"""
Compute probe vehicle penetration rate at Iowa & University intersection.

For each (direction, 15-min window):
    p = TomTom_sampleSize / GRIDSMART_volume

Then aggregate to a global penetration rate (volume-weighted) for use in the rest
of the corridor.

Inputs:
  outputs/calibration/gridsmart_thursday_avg.csv   — 5-Thursday averaged 15-min volumes
  outputs/calibration/iowa_tomtom_last_segments.csv — TomTom sampleSize per (dir, timeSet)

Output:
  outputs/calibration/penetration_rate.json
  outputs/calibration/penetration_samples.csv  (diagnostic: per-sample p values)
"""

import os
import sys
import json
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)

OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "calibration")
GRIDSMART_CSV = os.path.join(OUTPUT_DIR, "gridsmart_thursday_avg.csv")
TOMTOM_CSV = os.path.join(OUTPUT_DIR, "iowa_tomtom_last_segments.csv")


def main():
    print("=" * 60)
    print("  Probe Vehicle Penetration Rate Estimation")
    print("=" * 60)

    if not os.path.exists(GRIDSMART_CSV):
        print(f"ERROR: {GRIDSMART_CSV} not found (run parse_gridsmart.py first)")
        sys.exit(1)
    if not os.path.exists(TOMTOM_CSV):
        print(f"ERROR: {TOMTOM_CSV} not found (run extract_iowa_tomtom.py first)")
        sys.exit(1)

    gs = pd.read_csv(GRIDSMART_CSV)
    tt = pd.read_csv(TOMTOM_CSV)

    print(f"\nGRIDSMART: {len(gs)} rows (15-min windows)")
    print(f"TomTom:    {len(tt)} rows (dir × timeSet)")

    # Pivot GRIDSMART to have columns [time_hhmm, EB, WB, NB, SB]
    # Already in wide format, just rename
    gs_long = pd.melt(
        gs[["time_hhmm", "NB_avg", "EB_avg", "SB_avg", "WB_avg"]],
        id_vars="time_hhmm",
        value_vars=["NB_avg", "EB_avg", "SB_avg", "WB_avg"],
        var_name="direction",
        value_name="V_gridsmart",
    )
    gs_long["direction"] = gs_long["direction"].str.replace("_avg", "")

    # Merge with TomTom on (direction, time_hhmm)
    merged = tt.merge(
        gs_long,
        on=["direction", "time_hhmm"],
        how="inner",
    )

    print(f"\nMerged records: {len(merged)} (after joining TomTom timeSets with GRIDSMART times)")

    if merged.empty:
        print("ERROR: No matching records between TomTom and GRIDSMART times")
        sys.exit(1)

    # Compute per-sample penetration
    # V_gridsmart is per 15-min, sampleSize is per 15-min → directly comparable
    merged = merged[merged["V_gridsmart"] > 0].copy()
    merged["p"] = merged["sampleSize"] / merged["V_gridsmart"]

    print("\n--- Per-sample penetration distribution ---")
    print(f"  Count: {len(merged)}")
    print(f"  Mean:  {merged['p'].mean():.4f}")
    print(f"  Std:   {merged['p'].std():.4f}")
    print(f"  Min:   {merged['p'].min():.4f}")
    print(f"  Max:   {merged['p'].max():.4f}")
    print(f"  Median:{merged['p'].median():.4f}")

    # Global weighted average (weight = GRIDSMART volume)
    # Use EB+WB only because they are on the main arterial (University Ave),
    # which is the primary target of calibration. NB/SB are on Iowa Ave with
    # very short TomTom segments (10m) that may capture multi-movement traffic
    # near the intersection, leading to over-counts relative to GRIDSMART.
    arterial_mask = merged["direction"].isin(["EB", "WB"])
    arterial = merged[arterial_mask]
    w_art = arterial["V_gridsmart"].values
    p_art = arterial["p"].values
    p_arterial = float(np.sum(p_art * w_art) / np.sum(w_art))

    # Also compute all-direction version for reference
    w = merged["V_gridsmart"].values
    p = merged["p"].values
    p_all = float(np.sum(p * w) / np.sum(w))

    # Use arterial as the canonical p
    p_global = p_arterial

    # By direction (also weighted)
    by_dir = {}
    for d in ["EB", "WB", "NB", "SB"]:
        sub = merged[merged["direction"] == d]
        if len(sub) > 0:
            ww = sub["V_gridsmart"].values
            pp = sub["p"].values
            by_dir[d] = {
                "p": float(np.sum(pp * ww) / np.sum(ww)),
                "n": int(len(sub)),
                "total_volume": int(sub["V_gridsmart"].sum()),
                "total_probe": int(sub["sampleSize"].sum()),
            }

    # By timeSet (not weighted, since each timeSet has just 4 dirs)
    by_ts = {}
    for ts in sorted(merged["timeSet"].unique()):
        sub = merged[merged["timeSet"] == ts]
        ww = sub["V_gridsmart"].values
        pp = sub["p"].values
        by_ts[int(ts)] = float(np.sum(pp * ww) / np.sum(ww)) if np.sum(ww) > 0 else 0.0

    # Diagnostics (CV computed on arterial-only for consistency with p_global)
    cv_arterial = arterial["p"].std() / arterial["p"].mean() if arterial["p"].mean() > 0 else 0
    cv_all = merged["p"].std() / merged["p"].mean() if merged["p"].mean() > 0 else 0

    print(f"\n--- Global Penetration Rate (EB+WB arterial, volume-weighted) ---")
    print(f"  p_global    = {p_arterial:.4f}  ({p_arterial*100:.2f}%)  [USE THIS]")
    print(f"  p_all_dirs  = {p_all:.4f}  ({p_all*100:.2f}%)  [reference, includes NB/SB]")
    print(f"  CV_arterial = {cv_arterial:.3f}  {'⚠️ HIGH' if cv_arterial > 0.3 else '✓'}")
    print(f"  CV_all      = {cv_all:.3f}")
    print(f"\n  NOTE: NB/SB excluded from global p — their very short segments (10m) on")
    print(f"  Iowa Ave appear to over-count traffic (p_NB={by_dir.get('NB',{}).get('p',0):.2f}).")

    print(f"\n--- By Direction ---")
    for d, info in by_dir.items():
        print(f"  {d}: p={info['p']:.4f}  (n={info['n']}, "
              f"V_total={info['total_volume']}, probe_total={info['total_probe']})")

    print(f"\n--- By TimeSet (sampling temporal variation) ---")
    for ts in sorted(by_ts.keys())[:8]:  # show first 8
        h = 7 + (ts - 2) * 0.25
        print(f"  timeSet={ts}  (t≈{h:.2f}h):  p={by_ts[ts]:.4f}")
    print(f"  ...")

    # Save JSON
    result = {
        "p_global": round(p_global, 5),
        "p_all_directions": round(p_all, 5),
        "p_source": "EB+WB (arterial only, volume-weighted)",
        "mean_unweighted": round(float(merged["p"].mean()), 5),
        "std": round(float(merged["p"].std()), 5),
        "min": round(float(merged["p"].min()), 5),
        "max": round(float(merged["p"].max()), 5),
        "cv_arterial": round(cv_arterial, 4),
        "cv_all_directions": round(cv_all, 4),
        "assumption_valid": bool(cv_arterial <= 0.3),
        "n_samples": int(len(merged)),
        "n_samples_arterial": int(len(arterial)),
        "notes": "NB/SB excluded because short Iowa Ave segments (10m) "
                 "yield p>1.0, suggesting TomTom hit-counting overestimates "
                 "vehicle count on short segments at intersections.",
        "by_direction": {d: {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in info.items()}
                          for d, info in by_dir.items()},
        "by_timeset": {str(ts): round(p, 5) for ts, p in by_ts.items()},
    }

    json_path = os.path.join(OUTPUT_DIR, "penetration_rate.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Saved: {json_path}")

    # Save per-sample diagnostic CSV
    diag_path = os.path.join(OUTPUT_DIR, "penetration_samples.csv")
    merged[["direction", "timeSet", "time_hhmm", "sampleSize",
            "V_gridsmart", "p"]].to_csv(diag_path, index=False)
    print(f"✓ Saved: {diag_path}")

    if not result["assumption_valid"]:
        print(f"\n⚠️  CV_arterial = {cv_arterial:.3f} > 0.3 → uniform p assumption weak.")
        print("   Consider using by_timeset values instead of p_global.")
    else:
        print(f"\n✓ CV_arterial = {cv_arterial:.3f} ≤ 0.3 → uniform p assumption OK.")


if __name__ == "__main__":
    main()

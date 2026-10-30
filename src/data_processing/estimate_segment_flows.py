"""
Estimate vehicle flow for every segment across all 29 approaches using:
    V_segment = TomTom_sampleSize / p_global

Outputs:
  outputs/calibration/segment_flows_estimated.csv
      - Per-segment per-timeSet flow (veh/15min, veh/hr)
  outputs/calibration/approach_inflow.csv
      - Per-approach-direction per-timeSet inflow (using first/farthest segment)
      - This represents traffic entering the corridor from each fringe
"""

import os
import sys
import json
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)

APPROACH_FOLDER = os.path.join(ROOT_DIR, "data", "geojson-1-15-min-interval", "processed")
CORRIDOR_LAYOUT = os.path.join(ROOT_DIR, "data", "geojson-1-15-min-interval", "corridor_layout.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "calibration")
PENETRATION_JSON = os.path.join(OUTPUT_DIR, "penetration_rate.json")

# TomTom timeSet=2 corresponds to 7:00
TOMTOM_START_HOUR = 7.0
TOMTOM_MIN_TIMESET = 2


def timeset_to_sim_hour(ts, start_hour=TOMTOM_START_HOUR, min_ts=TOMTOM_MIN_TIMESET):
    """timeSet=2 → 7.00, timeSet=3 → 7.25, ..."""
    return start_hour + (ts - min_ts) * 0.25


def get_approach_segment_order(layout_csv):
    """Parse corridor_layout.csv → dict {approach_id: (first_seg, last_seg)}."""
    layout = pd.read_csv(layout_csv)
    result = {}
    for _, row in layout.iterrows():
        name = row["intersection"]
        for direction in ["eb", "wb", "nb", "sb"]:
            val = row.get(direction)
            if pd.isna(val) or not str(val).strip():
                continue
            cleaned = str(val).replace("(", "").replace(")", "").replace("'", "")
            parts = [p.strip() for p in cleaned.split(",")]
            if len(parts) >= 2:
                approach_id = f"{name}_{direction}"
                result[approach_id] = (parts[0], parts[-1])  # (first=farthest, last=closest)
    return result


def main():
    print("=" * 60)
    print("  Segment Flow Estimation (全走廊)")
    print("=" * 60)

    # Load penetration rate
    if not os.path.exists(PENETRATION_JSON):
        print(f"ERROR: {PENETRATION_JSON} not found (run compute_penetration.py first)")
        sys.exit(1)

    with open(PENETRATION_JSON) as f:
        pen = json.load(f)
    p_arterial = pen["p_global"]

    # Cross streets (NB/SB) had TomTom over-counting issues (short Iowa Ave
    # segments capture multi-movement traffic). Apply a LARGER effective p to
    # deflate their estimated flow. p_cross=1.0 means sampleSize is used
    # directly without amplification (3x reduction vs p_arterial=0.348).
    p_cross = float(os.environ.get("P_CROSS", "1.0"))

    print(f"\nUsing p_arterial = {p_arterial:.4f} ({p_arterial*100:.2f}%) for EB/WB")
    print(f"Using p_cross    = {p_cross:.4f} ({p_cross*100:.2f}%) for NB/SB")
    print(f"Source: {pen['p_source']}")

    # Load approach first/last mapping
    first_last = get_approach_segment_order(CORRIDOR_LAYOUT)
    print(f"\n{len(first_last)} approaches in corridor_layout.csv")

    # Process each approach CSV
    all_segments = []
    for fname in sorted(os.listdir(APPROACH_FOLDER)):
        if not fname.endswith("_delay_time_results.csv"):
            continue
        approach_id = fname.replace("_delay_time_results.csv", "")
        df = pd.read_csv(os.path.join(APPROACH_FOLDER, fname))
        df["approach_id"] = approach_id
        df["simpleID"] = df["simpleID"].astype(str)

        # Pick penetration rate by direction: NB/SB use p_cross, EB/WB use p_arterial
        direction_suffix = approach_id.rsplit("_", 1)[-1].lower()
        if direction_suffix in ("nb", "sb"):
            p_used = p_cross
        else:
            p_used = p_arterial

        # Estimate flow (sampleSize is probes per 15-min window)
        df["V_veh_per_15min"] = (df["sampleSize"] / p_used).round().astype(int)
        df["V_veh_per_hour"] = df["V_veh_per_15min"] * 4
        df["p_used"] = p_used
        df["sim_hour"] = df["timeSet"].apply(timeset_to_sim_hour)

        # Mark first/last segments
        if approach_id in first_last:
            first_seg, last_seg = first_last[approach_id]
            df["is_first"] = df["simpleID"] == first_seg
            df["is_last"] = df["simpleID"] == last_seg
        else:
            df["is_first"] = False
            df["is_last"] = False

        keep_cols = ["approach_id", "simpleID", "timeSet", "sim_hour",
                     "sampleSize", "distance", "speedLimit",
                     "harmonicAverageSpeed",
                     "p_used", "V_veh_per_15min", "V_veh_per_hour",
                     "is_first", "is_last"]
        all_segments.append(df[keep_cols])

    combined = pd.concat(all_segments, ignore_index=True)
    combined = combined.sort_values(["approach_id", "timeSet", "simpleID"]).reset_index(drop=True)

    # Save segment-level
    seg_path = os.path.join(OUTPUT_DIR, "segment_flows_estimated.csv")
    combined.to_csv(seg_path, index=False)
    print(f"\n✓ Saved: {seg_path}")
    print(f"  {len(combined)} rows ({combined['approach_id'].nunique()} approaches × "
          f"{combined['timeSet'].nunique()} timeSets × segments)")

    # Approach-level inflow = flow of first (farthest) segment
    # This represents traffic ENTERING the corridor from this direction
    inflow = combined[combined["is_first"]].copy()
    inflow = inflow[["approach_id", "timeSet", "sim_hour", "simpleID",
                     "sampleSize", "V_veh_per_15min", "V_veh_per_hour"]]
    inflow = inflow.rename(columns={"simpleID": "first_segID"})

    inflow_path = os.path.join(OUTPUT_DIR, "approach_inflow.csv")
    inflow.to_csv(inflow_path, index=False)
    print(f"\n✓ Saved: {inflow_path}")
    print(f"  {len(inflow)} rows ({inflow['approach_id'].nunique()} approaches × "
          f"{inflow['timeSet'].nunique()} timeSets)")

    # Summary by approach: hourly profile
    print("\n--- Approach inflow summary (total veh/hr averaged over 6h window) ---")
    summary = inflow.groupby("approach_id").agg(
        mean_vph=("V_veh_per_hour", "mean"),
        max_vph=("V_veh_per_hour", "max"),
        min_vph=("V_veh_per_hour", "min"),
        total_15min_count=("V_veh_per_15min", "sum"),
    ).round(0).astype(int)
    summary = summary.sort_values("mean_vph", ascending=False)
    print(summary.to_string())

    # Comparison with GRIDSMART at Iowa
    print("\n--- Iowa sanity check (estimated inflow vs GRIDSMART 7am-1pm total) ---")
    iowa = inflow[inflow["approach_id"].str.startswith("Iowa_")]
    gs = pd.read_csv(os.path.join(OUTPUT_DIR, "gridsmart_thursday_avg.csv"))
    window = gs[(gs["time_hhmm"] >= "07:00") & (gs["time_hhmm"] < "13:00")]
    for dir_sfx, gs_col in [("eb", "EB_avg"), ("wb", "WB_avg"),
                             ("nb", "NB_avg"), ("sb", "SB_avg")]:
        est_sum = iowa[iowa["approach_id"] == f"Iowa_{dir_sfx}"]["V_veh_per_15min"].sum()
        gs_sum = window[gs_col].sum()
        ratio = est_sum / gs_sum if gs_sum > 0 else 0
        print(f"  Iowa_{dir_sfx}: estimated={est_sum}  GRIDSMART={gs_sum}  ratio={ratio:.2f}")


if __name__ == "__main__":
    main()

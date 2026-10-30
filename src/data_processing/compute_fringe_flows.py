"""
Compute hourly fringe flows for each boundary edge by aggregating
approach-level inflow from 15-min timeSets to hourly buckets.

Mapping from boundary_key → approach_id (from corridor_boundary_edges.json):
  arterial:
    uni_west_entry_EB  ← Lime_eb  first segment inflow
    uni_east_entry_WB  ← Iowa_wb  first segment inflow
  cross:
    {Cross}_south_entry_NB ← {Cross}_nb first segment inflow
    {Cross}_north_entry_SB ← {Cross}_sb first segment inflow

Direction convention for entry:
  "south_entry_NB" = car enters from south fringe, travels NB → observed on NB approach
  "north_entry_SB" = car enters from north fringe, travels SB → observed on SB approach

Output: outputs/calibration/fringe_flows.csv
  Columns: boundary_key, hour, V_veh_per_hour, V_veh_per_15min_samples
"""

import os
import sys
import json
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)

OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "calibration")
BOUNDARY_JSON = os.path.join(OUTPUT_DIR, "corridor_boundary_edges.json")
INFLOW_CSV = os.path.join(OUTPUT_DIR, "approach_inflow.csv")
SEGMENT_FLOWS_CSV = os.path.join(OUTPUT_DIR, "segment_flows_estimated.csv")


def get_approach_flow_by_hour(segment_flows, approach_id, seg_id):
    """Get hourly vehicle flow for one approach's specific segment.

    Aggregates 4 × 15-min timeSets per hour.

    Returns dict: {hour: V_veh_per_hour}
    """
    sf = segment_flows[
        (segment_flows["approach_id"] == approach_id)
        & (segment_flows["simpleID"].astype(str) == str(seg_id))
    ]
    if sf.empty:
        return {}

    result = {}
    # Each row is 1 timeSet = 15 min. Hour = floor(sim_hour)
    sf = sf.copy()
    sf["hour_bucket"] = sf["sim_hour"].astype(int)
    for h, grp in sf.groupby("hour_bucket"):
        # Sum 4 quarters → hourly volume
        total = int(grp["V_veh_per_15min"].sum())
        result[int(h)] = total
    return result


def main():
    print("=" * 60)
    print("  Fringe Flow Aggregation")
    print("=" * 60)

    if not os.path.exists(BOUNDARY_JSON):
        print(f"ERROR: {BOUNDARY_JSON} not found")
        sys.exit(1)
    if not os.path.exists(SEGMENT_FLOWS_CSV):
        print(f"ERROR: {SEGMENT_FLOWS_CSV} not found")
        sys.exit(1)

    with open(BOUNDARY_JSON) as f:
        boundaries = json.load(f)
    seg_flows = pd.read_csv(SEGMENT_FLOWS_CSV)

    print(f"\nBoundaries: arterial={len(boundaries['arterial'])}, "
          f"cross={sum(len(v) for v in boundaries['cross'].values())}")
    print(f"Segment flows: {len(seg_flows)} rows")

    # Build list of (boundary_key, approach, seg_id, role)
    # Entry: use first_mapped_seg (already in boundaries)
    # Exit:  use last_mapped_seg
    rows = []

    # Arterial entries
    for key, info in boundaries["arterial"].items():
        approach = info["approach"]
        if "entry" in key:
            seg = info.get("first_mapped_seg")
        else:
            seg = info.get("last_mapped_seg")
        if not seg:
            continue
        hourly = get_approach_flow_by_hour(seg_flows, approach, seg)
        for h, vph in hourly.items():
            rows.append({
                "boundary_key": key,
                "boundary_type": "arterial",
                "approach": approach,
                "seg_id": seg,
                "edge_id": info["edge"],
                "hour": h,
                "V_veh_per_hour": vph,
            })

    # Cross street entries and exits
    for cross_name, cross_info in boundaries["cross"].items():
        for key, info in cross_info.items():
            approach = info["approach"]
            if "entry" in key:
                seg = info.get("first_mapped_seg")
            else:
                seg = info.get("last_mapped_seg")
            if not seg:
                continue
            hourly = get_approach_flow_by_hour(seg_flows, approach, seg)
            full_key = f"{cross_name}_{key}"
            for h, vph in hourly.items():
                rows.append({
                    "boundary_key": full_key,
                    "boundary_type": "cross",
                    "approach": approach,
                    "seg_id": seg,
                    "edge_id": info["edge"],
                    "hour": h,
                    "V_veh_per_hour": vph,
                })

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["boundary_type", "boundary_key", "hour"]).reset_index(drop=True)

    # Save
    out_path = os.path.join(OUTPUT_DIR, "fringe_flows.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n✓ Saved: {out_path}")
    print(f"  {len(out_df)} rows ({out_df['boundary_key'].nunique()} boundary keys × "
          f"{out_df['hour'].nunique()} hours)")

    # Summary: average hourly flow per boundary
    summary = out_df.groupby(["boundary_type", "boundary_key"]).agg(
        mean_vph=("V_veh_per_hour", "mean"),
        max_vph=("V_veh_per_hour", "max"),
        min_vph=("V_veh_per_hour", "min"),
        n_hours=("hour", "count"),
    ).round(0).astype(int)
    print("\n--- Summary by boundary ---")
    print(summary.to_string())

    # Show morning peak sample
    print("\n--- Morning peak (8:00) samples ---")
    peak = out_df[out_df["hour"] == 8].sort_values("V_veh_per_hour", ascending=False)
    print(peak[["boundary_type", "boundary_key", "V_veh_per_hour"]].to_string(index=False))


if __name__ == "__main__":
    main()

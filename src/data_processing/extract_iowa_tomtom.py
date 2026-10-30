"""
Extract TomTom probe data for Iowa & University intersection by averaging the
LAST TWO segments (closest to intersection) of each approach.

Using the average of the last 2 segments (instead of just 1) gives more stable
sample-size estimates because:
  - Reduces noise from short segments (10m) with detection glitches
  - Smooths over segment geometry variations at the intersection
  - The same vehicle passes both segments, so average ≈ typical probe count

From corridor_layout.csv approach CSVs (segments ordered farthest → nearest):
    Iowa EB: ...,429,...,435  → last 2 segments of the approach
    Iowa WB: ...,661,...,602  → last 2
    Iowa NB: ...,553,...,312  → last 2
    Iowa SB: ...,304,...,29   → last 2

Output: outputs/calibration/iowa_tomtom_last_segments.csv
  (columns: direction, timeSet, time_hhmm, sampleSize_avg,
            sampleSize_seg1, sampleSize_seg2, ...)
"""

import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)

APPROACH_FOLDER = os.path.join(ROOT_DIR, "data", "geojson-1-15-min-interval", "processed")
CORRIDOR_LAYOUT = os.path.join(ROOT_DIR, "data", "geojson-1-15-min-interval", "corridor_layout.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "calibration")

# TomTom data covers morning block starting at 7:00 AM
# timeSet=2 → 7:00-7:15, timeSet=3 → 7:15-7:30, ..., timeSet=25 → 12:45-13:00
TOMTOM_START_HOUR = 7.0
TOMTOM_MIN_TIMESET = 2


def timeset_to_clock(ts, start_hour=TOMTOM_START_HOUR, min_ts=TOMTOM_MIN_TIMESET):
    """Convert timeSet to HH:MM string (start of interval)."""
    offset_min = (ts - min_ts) * 15
    total_min = int(start_hour * 60 + offset_min)
    h = total_min // 60
    m = total_min % 60
    return f"{h:02d}:{m:02d}"


N_LAST_SEGMENTS = 5  # Number of segments closest to intersection to average


def get_approach_segment_order(approach_folder, approach_id):
    """Return ordered list of simpleIDs for approach (farthest → nearest)."""
    fpath = os.path.join(approach_folder, f"{approach_id}_delay_time_results.csv")
    if not os.path.exists(fpath):
        return []
    df = pd.read_csv(fpath)
    min_ts = df["timeSet"].min()
    df_t = df[df["timeSet"] == min_ts].reset_index(drop=True)
    return df_t["simpleID"].astype(str).tolist()


def main():
    print("=" * 60)
    print(f"  Iowa&University TomTom Last-{N_LAST_SEGMENTS}-Segments Extractor")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get last N segment IDs per direction
    print(f"\nIowa last-{N_LAST_SEGMENTS}-segments mapping (closest → to intersection):")
    last_segs_per_dir = {}
    for direction in ["eb", "wb", "nb", "sb"]:
        approach_id = f"Iowa_{direction}"
        seg_list = get_approach_segment_order(APPROACH_FOLDER, approach_id)
        if not seg_list:
            print(f"  Iowa_{direction}: approach not found")
            continue
        # Take last N (closest to intersection)
        last_n = seg_list[-N_LAST_SEGMENTS:]
        last_segs_per_dir[direction] = last_n
        print(f"  Iowa_{direction}: {last_n}  (of {len(seg_list)} total segments)")

    if not last_segs_per_dir:
        print("ERROR: No Iowa approaches found")
        sys.exit(1)

    # Read each approach CSV and extract all selected segments
    rows = []
    for direction, seg_ids in last_segs_per_dir.items():
        fname = f"Iowa_{direction}_delay_time_results.csv"
        fpath = os.path.join(APPROACH_FOLDER, fname)
        if not os.path.exists(fpath):
            print(f"  WARN: {fname} not found")
            continue

        df = pd.read_csv(fpath)
        df["simpleID"] = df["simpleID"].astype(str)
        seg_df = df[df["simpleID"].isin(seg_ids)].copy()

        if seg_df.empty:
            continue

        # For each timeSet, average the N segments' sampleSize & speed
        for ts, grp in seg_df.groupby("timeSet"):
            # Average across the N=2 segments
            sample_avg = grp["sampleSize"].mean()
            speed_avg = grp["harmonicAverageSpeed"].mean()
            distance_sum = grp["distance"].sum()  # total distance covered by the 2 segments
            street = grp["streetName"].iloc[0]
            speed_limit = grp["speedLimit"].iloc[0] if "speedLimit" in grp else 0

            # Also store per-segment values for diagnostics
            seg_vals = grp.set_index("simpleID")["sampleSize"].to_dict()

            rows.append({
                "direction": direction.upper(),
                "segments": ",".join(seg_ids),
                "timeSet": int(ts),
                "time_hhmm": timeset_to_clock(int(ts)),
                "sampleSize": round(sample_avg, 1),  # AVERAGE of last 2 (used downstream)
                "sampleSize_raw": seg_vals,          # per-segment (diagnostic)
                "sampleSize_sum": int(grp["sampleSize"].sum()),
                "harmonicAverageSpeed": round(speed_avg, 2),
                "distance_m": round(distance_sum, 2),
                "streetName": street,
                "speedLimit_kph": float(speed_limit),
                "n_segments_averaged": len(grp),
            })

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["direction", "timeSet"]).reset_index(drop=True)

    # Save (drop the dict column which doesn't CSV well)
    save_df = out_df.drop(columns=["sampleSize_raw"])
    out_path = os.path.join(OUTPUT_DIR, "iowa_tomtom_last_segments.csv")
    save_df.to_csv(out_path, index=False)
    print(f"\n✓ Saved: {out_path}")
    print(f"  {len(out_df)} rows (4 directions × {out_df['timeSet'].nunique()} timeSets)")
    print(f"  sampleSize column is AVERAGE of last {N_LAST_SEGMENTS} segments")

    # Summary by direction
    print("\n--- Summary by direction ---")
    summary = out_df.groupby("direction").agg(
        n_timesets=("timeSet", "count"),
        total_samples_avg=("sampleSize", "sum"),
        mean_samples_avg=("sampleSize", "mean"),
        min_samples_avg=("sampleSize", "min"),
        max_samples_avg=("sampleSize", "max"),
        total_samples_sum=("sampleSize_sum", "sum"),
    ).round(1)
    print(summary.to_string())

    # Show a few rows for inspection
    print("\n--- Sample rows (timeSet 2-5) ---")
    sample = out_df[out_df["timeSet"].isin([2, 3, 4, 5])]
    print(sample[["direction", "segments", "timeSet", "time_hhmm",
                  "sampleSize", "sampleSize_sum",
                  "harmonicAverageSpeed"]].to_string(index=False))


if __name__ == "__main__":
    main()

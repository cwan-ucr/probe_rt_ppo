"""
Load and prepare TomTom observed data as calibration targets.

Each target is a (approach_id, timeSet) pair with a distance-weighted
harmonic average speed that the simulation should match.
"""

import os
import json
import pandas as pd
import numpy as np

from demand_calibration.stop_bar_segments import (load_stop_bar_simple_ids,
                                                  load_nearest_stop_bar_sids)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def _load_chosen_stop_bar_sids():
    """Load per-approach stop-bar simpleID mapping selected after a
    preliminary sim run (if available); fall back to chain[-2]."""
    chosen_path = os.path.join(ROOT_DIR, "outputs", "calibration",
                                "stop_bar_chosen.json")
    if os.path.exists(chosen_path):
        with open(chosen_path) as f:
            return {k: int(v) for k, v in json.load(f).items()}
    return load_stop_bar_simple_ids()


class CalibrationTargets:
    """Load TomTom approach-level speed targets for calibration."""

    def __init__(self, processed_folder, time_period="morning",
                 stop_bar_only=True):
        """
        Parameters
        ----------
        processed_folder : str
            Path to the folder containing *_delay_time_results.csv files
        time_period : str
            'morning' or 'afternoon'
        stop_bar_only : bool
            If True, restrict each approach's target to ONLY the stop-bar
            segment. The mapping is loaded from
            outputs/calibration/stop_bar_chosen.json if present (selected
            after one preliminary sim to skip approaches whose chain[-2]
            segment lies beyond the simulated demand). Otherwise falls
            back to chain[-2] from the TomTom approach CSV.
        """
        self.processed_folder = processed_folder
        self.time_period = time_period
        self.stop_bar_only = stop_bar_only
        # {approach_id: [up to N simpleIDs]} — the N segments nearest the stop bar
        # (incl. stop bar). Speeds are matched per-segment over these.
        self.stop_bar_sids = (load_nearest_stop_bar_sids()
                              if stop_bar_only else None)
        self._targets = None
        self._load()

    def _load(self):
        rows = []
        for fname in os.listdir(self.processed_folder):
            if not fname.endswith("_delay_time_results.csv"):
                continue
            approach_id = fname.replace("_delay_time_results.csv", "")

            fpath = os.path.join(self.processed_folder, fname)
            df = pd.read_csv(fpath)

            # Stage A: restrict to the N segments nearest the stop bar (incl. it);
            # each segment is matched to its own sim speed (per-segment objective).
            if self.stop_bar_only:
                sids = self.stop_bar_sids.get(approach_id)
                if not sids:
                    continue
                df = df[df["simpleID"].isin(sids)]
                if df.empty:
                    continue

            # One target per (segment, timeSet): group includes simpleID so the
            # 5 nearest segments stay separate rows (not aggregated).
            group_keys = ["simpleID", "timeSet"] if self.stop_bar_only else ["timeSet"]
            for keys, grp in df.groupby(group_keys):
                sid = keys[0] if self.stop_bar_only else None
                ts = keys[1] if self.stop_bar_only else keys
                distances = grp["distance"].values
                speeds = grp["harmonicAverageSpeed"].values
                sample_sizes = grp["sampleSize"].values

                # Filter out zero/negative speeds
                valid = speeds > 0
                if valid.sum() == 0:
                    continue

                d = distances[valid]
                s = speeds[valid]
                ss = sample_sizes[valid]

                # Distance-weighted harmonic mean speed
                total_dist = d.sum()
                if total_dist <= 0:
                    continue
                travel_times = d / (s / 3.6)  # seconds per segment
                total_travel_time = travel_times.sum()
                if total_travel_time <= 0:
                    continue
                weighted_speed_kph = (total_dist / total_travel_time) * 3.6

                # Weight for objective function: confidence * distance
                avg_sample = ss.mean()
                weight = min(avg_sample, 100) / 100.0 * total_dist

                rows.append({
                    "approach_id": approach_id,
                    "detector_id": int(sid) if sid is not None else -1,
                    "timeSet": int(ts),
                    "target_speed_kph": round(weighted_speed_kph, 2),
                    "total_distance_m": round(total_dist, 1),
                    "avg_sample_size": round(avg_sample, 1),
                    "weight": round(weight, 2),
                    "num_segments": int(valid.sum()),
                })

        self._targets = pd.DataFrame(rows)

    def get_target_matrix(self):
        """Return full target DataFrame."""
        return self._targets.copy()

    def get_approach_ids(self):
        """Return list of unique approach IDs."""
        return sorted(self._targets["approach_id"].unique().tolist())

    def get_corridor_targets(self):
        """Aggregate to corridor-level: average speed per timeSet across all approaches.
        Used for Phase 1 (total volume calibration).
        """
        df = self._targets.copy()
        corridor = df.groupby("timeSet", group_keys=False).apply(
            lambda g: pd.Series({
                "target_speed_kph": np.average(
                    g["target_speed_kph"], weights=g["total_distance_m"]
                ),
                "total_weight": g["weight"].sum(),
                "num_approaches": len(g),
            }),
            include_groups=False,
        ).reset_index()
        return corridor

    def get_timeset_to_sim_interval(self, sim_start_hour=7):
        """Map timeSet indices to simulation time intervals.

        Assumes timeSet=2 corresponds to the first 15-min interval
        starting at sim_start_hour.

        Returns dict: {timeSet: (sim_begin_sec, sim_end_sec)}
        """
        mapping = {}
        min_ts = self._targets["timeSet"].min()
        for ts in sorted(self._targets["timeSet"].unique()):
            interval_idx = ts - min_ts
            begin = interval_idx * 900
            end = begin + 900
            mapping[ts] = (begin, end)
        return mapping

    def __repr__(self):
        n_approaches = len(self.get_approach_ids())
        n_timesets = self._targets["timeSet"].nunique()
        n_total = len(self._targets)
        return (f"CalibrationTargets({n_approaches} approaches, "
                f"{n_timesets} time periods, {n_total} targets)")

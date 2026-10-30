"""
Wrap SUMO simulation as a callable function for calibration.

Supports two parameter types:
  - DemandParameters (legacy, randomTrips-based)
  - FlowParameters (flow-based, uses <flow> definitions from fringe data)

Parameters → generate trips → run SUMO → return speed results
"""

import os
import sys
import json
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)

# Add src to path for imports
sys.path.insert(0, SRC_DIR)

DEFAULT_BOUNDARY_JSON = os.path.join(ROOT_DIR, "outputs", "calibration",
                                       "corridor_boundary_edges.json")


class SimulationRunner:
    """Run one SUMO simulation with given demand parameters and return results."""

    def __init__(self, sumocfg, additional_file, approach_folder, net_file,
                 sim_start_hour=7, sim_duration_hours=6, probe_prob=0.3478,
                 probe_prob_cross=0.75,
                 boundaries_json=DEFAULT_BOUNDARY_JSON,
                 stop_bar_only=True, output_dir=None):
        # per-runner output dir -> parallel GA workers each isolate their trips/
        # results files (no clash on the shared default folder).
        self.output_dir = output_dir
        self.sumocfg = sumocfg
        self.additional_file = additional_file
        self.approach_folder = approach_folder
        self.net_file = net_file
        self.sim_start_hour = sim_start_hour
        self.sim_duration_hours = sim_duration_hours
        self.probe_prob = probe_prob
        self.probe_prob_cross = probe_prob_cross
        self.boundaries_json = boundaries_json
        self.stop_bar_only = stop_bar_only
        self.run_count = 0
        # Lazy-load ranked stop-bar candidates (most-downstream first)
        self._stop_bar_ranked = None
        if stop_bar_only:
            from demand_calibration.stop_bar_segments import load_stop_bar_simple_ids
            self._stop_bar_ranked = load_stop_bar_simple_ids(ranked=True)
        # Effective sid chosen per (approach_id) — populated lazily inside
        # parse_sim_results based on which sids actually have data this run
        self._chosen_stop_bar_sid = {}

    def run(self, params, output_dir=None):
        """Run simulation with the given demand parameters.

        Dispatches based on params type:
          - FlowParameters → generate flow XML (new pipeline)
          - DemandParameters → generate trips via randomTrips (legacy)
        """
        from tsc_sumo_sim import run_simulation

        output_dir = (output_dir or self.output_dir
                      or os.path.join(ROOT_DIR, "outputs", "calibration"))
        os.makedirs(output_dir, exist_ok=True)

        self.run_count += 1
        run_id = f"run_{os.getpid()}_{self.run_count:04d}"   # pid -> parallel-safe

        trips_probe_file = os.path.join(output_dir, f"trips_{run_id}_probe.xml")

        # Detect parameter type and generate demand
        if hasattr(params, "flows"):  # FlowParameters
            params.to_flow_xml(
                boundaries_json=self.boundaries_json,
                out_file=trips_probe_file,
                probe_prob=self.probe_prob,
                probe_prob_cross=self.probe_prob_cross,
            )
            trips_file = None
        else:  # DemandParameters (legacy)
            from generate_daily_demand_with_probe import generate_demand
            trips_file = os.path.join(output_dir, f"trips_{run_id}.xml")
            end_hour = self.sim_start_hour + self.sim_duration_hours
            generate_demand(
                hourly_volumes=params.hourly_volumes,
                net_file=self.net_file,
                out_file=trips_file,
                out_file_probe=trips_probe_file,
                probe_prob=self.probe_prob,
                seed=42,
                arterial_factor=params.arterial_factor,
                hours_range=(self.sim_start_hour, end_hour),
                verbose=False,
            )

        # Run simulation
        output_csv = os.path.join(output_dir, f"results_{run_id}.csv")
        end_time = self.sim_duration_hours * 3600

        results_df = run_simulation(
            sumocfg=self.sumocfg,
            additional_file=self.additional_file,
            approach_folder=self.approach_folder,
            output_csv=output_csv,
            use_gui=False,
            end_time=end_time,
            return_results=True,
            trips_file=trips_probe_file,
        )

        # Clean up intermediate files
        for f in [trips_file, trips_probe_file, output_csv]:
            if f and os.path.exists(f):
                os.remove(f)

        return results_df

    def parse_sim_results(self, results_df):
        """Convert simulation results to approach-level speed per interval.

        Returns DataFrame with: approach_id, interval_index, sim_speed_kph
        """
        if results_df is None or results_df.empty:
            return pd.DataFrame(columns=["approach_id", "interval_begin",
                                          "sim_speed_kph"])

        # Group by approach and interval, compute distance-weighted harmonic speed
        # The simulation already computes harmonic_mean_speed_kph per segment
        df = results_df.copy()
        df["interval_begin"] = df["interval_begin"].astype(int)
        # Normalize to 900s-grid: tsc_sumo_sim emits 0, 901, 1801, ... but
        # targets use clean 0, 900, 1800, ... — align them for the merge.
        df["interval_begin"] = (df["interval_begin"] // 900) * 900

        # Stage A: restrict to stop-bar simpleID per approach.
        # The chosen sid mapping is loaded from a JSON written by
        # `tools/choose_stop_bar_sids.py` after one preliminary sim run.
        if self.stop_bar_only:
            df["detector_id"] = df["detector_id"].astype(int)
            # keep the N segments nearest the stop bar (incl. it) per approach,
            # matched per-segment to the target.
            from demand_calibration.stop_bar_segments import load_nearest_stop_bar_sids
            nearest = load_nearest_stop_bar_sids()
            keep_mask = df.apply(
                lambda r: r["detector_id"] in nearest.get(r["approach_id"], []),
                axis=1,
            )
            df = df[keep_mask]

        # One sim speed per (approach, segment, interval) so each of the N
        # nearest segments is matched to its own target (per-segment objective).
        group_cols = (["approach_id", "detector_id", "interval_begin"]
                      if self.stop_bar_only
                      else ["approach_id", "interval_begin"])
        rows = []
        for keys, grp in df.groupby(group_cols):
            valid = grp[grp["harmonic_mean_speed_kph"] > 0]
            if valid.empty:
                continue
            # Weight by vehicle count
            counts = valid["vehicle_count"].values.astype(float)
            speeds = valid["harmonic_mean_speed_kph"].values
            total_count = counts.sum()
            if total_count > 0:
                avg_speed = np.dot(counts, speeds) / total_count
            else:
                avg_speed = speeds.mean()

            row = {"approach_id": keys[0], "interval_begin": keys[-1],
                   "sim_speed_kph": round(avg_speed, 2),
                   "total_vehicles": int(total_count)}
            if self.stop_bar_only:
                row["detector_id"] = int(keys[1])
            rows.append(row)

        return pd.DataFrame(rows)

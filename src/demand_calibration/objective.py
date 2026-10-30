"""
Objective function for demand calibration.

Computes weighted RMSE between simulated and observed approach-level speeds.
"""

import numpy as np
import pandas as pd


class ObjectiveFunction:
    """Compute calibration objective: weighted RMSE of speed differences."""

    def __init__(self, targets, runner):
        """
        Parameters
        ----------
        targets : CalibrationTargets
        runner : SimulationRunner
        """
        self.targets = targets
        self.runner = runner
        self.history = []  # (iteration, Z, params_vector)

    def evaluate(self, params):
        """Run simulation and compute objective function value.

        Returns
        -------
        float : weighted RMSE in kph
        """
        # Run simulation
        results_df = self.runner.run(params)
        sim_df = self.runner.parse_sim_results(results_df)

        # Get targets
        target_df = self.targets.get_target_matrix()
        ts_mapping = self.targets.get_timeset_to_sim_interval()

        # Add sim_begin to targets for matching
        target_df["interval_begin"] = target_df["timeSet"].map(
            lambda ts: ts_mapping.get(ts, (None, None))[0]
        )
        target_df = target_df.dropna(subset=["interval_begin"])
        target_df["interval_begin"] = target_df["interval_begin"].astype(int)

        # Merge sim and target. When matching per-segment (the N nearest stop-bar
        # segments), include detector_id so each segment's sim speed is compared to
        # its own target; otherwise match per (approach, interval).
        keys = ["approach_id", "interval_begin"]
        if "detector_id" in target_df.columns and "detector_id" in sim_df.columns:
            keys = ["approach_id", "detector_id", "interval_begin"]
        merged = target_df.merge(
            sim_df[keys + ["sim_speed_kph"]],
            on=keys,
            how="left",
        )

        # Drop unmatched rows: missing sim data should not be penalised as
        # (target − 0)². If a target row has no sim observation, exclude it.
        merged = merged.dropna(subset=["sim_speed_kph"])

        if merged.empty:
            return 999.0

        # Compute weighted RMSE
        diff_sq = (merged["sim_speed_kph"] - merged["target_speed_kph"]) ** 2
        weights = merged["weight"].values

        total_weight = weights.sum()
        if total_weight == 0:
            return 999.0

        wrmse = np.sqrt(np.sum(weights * diff_sq) / total_weight)

        # Record history
        self.history.append({
            "iteration": len(self.history),
            "wrmse": round(wrmse, 3),
            "params": params.to_vector().tolist(),
            "n_matched": int(merged["sim_speed_kph"].gt(0).sum()),
            "n_total": len(merged),
        })

        return wrmse

    def evaluate_corridor(self, params):
        """Evaluate at corridor level (for Phase 1).

        Computes RMSE of corridor-wide average speed per time interval.
        """
        results_df = self.runner.run(params)
        sim_df = self.runner.parse_sim_results(results_df)

        if sim_df.empty:
            return 999.0

        # Corridor-level sim: average speed per interval
        sim_corridor = sim_df.groupby("interval_begin").agg(
            sim_speed_kph=("sim_speed_kph", "mean"),
        ).reset_index()

        # Corridor-level target
        target_corridor = self.targets.get_corridor_targets()
        ts_mapping = self.targets.get_timeset_to_sim_interval()
        target_corridor["interval_begin"] = target_corridor["timeSet"].map(
            lambda ts: ts_mapping.get(ts, (None, None))[0]
        )
        target_corridor = target_corridor.dropna(subset=["interval_begin"])
        target_corridor["interval_begin"] = target_corridor["interval_begin"].astype(int)

        merged = target_corridor.merge(
            sim_corridor, on="interval_begin", how="left"
        )
        merged = merged.dropna(subset=["sim_speed_kph"])

        if merged.empty:
            return 999.0

        diff_sq = (merged["sim_speed_kph"] - merged["target_speed_kph"]) ** 2
        rmse = np.sqrt(diff_sq.mean())

        self.history.append({
            "iteration": len(self.history),
            "wrmse": round(rmse, 3),
            "params": params.to_vector().tolist(),
            "n_matched": int(merged["sim_speed_kph"].gt(0).sum()),
            "n_total": len(merged),
        })

        return rmse

    def get_history_df(self):
        return pd.DataFrame(self.history)

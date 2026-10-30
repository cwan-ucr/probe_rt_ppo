"""Smoke-test the legacy Diaz dual-loop observation in SUMO.

Run from the repository root with SUMO/TraCI available::

    python -m src.rl.verify_diaz_detector_flow --intervals 32

The original feature sums E1 vehicle presence at each simulation step. It is
not an entry count, so comparing it to SUMO's nVehEntered would be invalid.
This checks loop placement and the aggregated 160-value state instead.
"""
import os
import argparse

os.environ["RL_METHOD"] = "diaz"
os.environ["RL_CONTROL_LAYOUT"] = "all10"
os.environ["RL_RESIDUAL"] = "0"
os.environ["RL_SPLIT"] = "0"
os.environ["RL_OFFSET_DEVIATION_PENALTY"] = "0"
os.environ.setdefault("RL_COLLECTOR", "probe")

import numpy as np

from . import config as C
from .diaz_style_env import (
    ADVANCE_POS_M,
    STOP_POS_M,
    DiazCorridorEnv,
    advance_layout,
    detector_layout,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", type=int, default=2,
                        help="number of consecutive 15-minute intervals (32 = 8 h)")
    args = parser.parse_args()
    if args.intervals < 2:
        parser.error("--intervals must be at least 2")
    stopbar, _edges = detector_layout()
    advance = advance_layout()
    approach_sites = {(node, direction) for node, direction, _length
                      in stopbar.values()}
    advance_approaches = {(node, direction) for node, direction, _pos, _length
                          in advance.values()}
    if approach_sites != advance_approaches:
        missing = approach_sites - advance_approaches
        raise AssertionError(f"advance/stop-bar approaches differ: {missing}")
    if set(advance) != set(stopbar):
        raise AssertionError("legacy advance and stop-bar loops must share lanes")
    if any(pos != ADVANCE_POS_M or pos >= STOP_POS_M
           for _node, _direction, pos, _length
           in advance.values()):
        raise AssertionError("advance detector is not at the legacy position")
    print(f"detector layout: {len(advance)} advance loops, "
          f"{len(stopbar)} stop-bar loops on the same lanes", flush=True)
    env = DiazCorridorEnv(
        backbone="coord_actuated",
        episode_end=args.intervals * C.DECISION_INTERVAL_S)
    observed = {}
    states = {}
    try:
        states[0] = env.reset()
        observed[0] = env._diaz_interval_detector_counts.copy()
        for begin in range(C.DECISION_INTERVAL_S,
                           args.intervals * C.DECISION_INTERVAL_S,
                           C.DECISION_INTERVAL_S):
            states[begin], _, _, _ = env.step(
                np.zeros(C.ACTION_DIM, dtype=np.float32))
            observed[begin] = env._diaz_interval_detector_counts.copy()
    finally:
        env.close()
    for begin, counts in sorted(observed.items()):
        flow = states[begin].reshape(C.N_AGENTS, len(C.DIRECTIONS), 4)
        aggregated = np.zeros((C.N_AGENTS, len(C.DIRECTIONS), 2))
        for detector, (node, direction, site) in env._diaz_detectors.items():
            aggregated[node, direction, site] += counts[detector]
        expected_flow = aggregated / np.maximum(env._n_lanes, 1)
        expected_flow /= (1800 * C.DECISION_INTERVAL_S / 3600)
        if not np.allclose(flow[:, :, (0, 2)], expected_flow,
                           rtol=1e-6, atol=1e-7):
            raise AssertionError(f"normalized policy flow mismatch at {begin} s")
        print(f"{begin:4d}–{begin + C.DECISION_INTERVAL_S:4d} s: "
              f"{len(counts)} loops, {sum(counts.values())} "
              "vehicle-presence steps")
    print("PASS: legacy loop placement and normalized policy flow verified")


if __name__ == "__main__":
    main()

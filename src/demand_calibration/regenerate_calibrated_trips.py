"""
Regenerate trips_turning_probe.xml from calibration_result_turning.json using
the CURRENT corridor_boundary_edges.json.

Fixes the stale-boundary issue where trips were generated against an older JSON
(.bak) that mapped Iowa.south_entry_NB to dead-end edge 1109915080. The current
JSON correctly uses 1109915091.

Usage:
  python src/demand_calibration/regenerate_calibrated_trips.py [--validate]
"""

import argparse
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.dirname(SCRIPT_DIR)
ROOT_DIR   = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from demand_calibration.turning_parameters import TurningParameters

DEFAULT_RESULT = os.path.join(ROOT_DIR, "outputs", "calibration",
                              "calibration_result_turning.json")
DEFAULT_BOUNDARY = os.path.join(ROOT_DIR, "outputs", "calibration",
                                "corridor_boundary_edges.json")
DEFAULT_OUT = os.path.join(ROOT_DIR, "outputs", "calibration",
                           "trips_turning_probe.xml")
DEFAULT_NET = os.path.join(ROOT_DIR, "data", "corridor-osm-network",
                           "osm.net.xml.gz")


def load_params(result_path):
    with open(result_path) as f:
        r = json.load(f)
    flows = {k: {int(h): int(v) for h, v in per_hour.items()}
             for k, per_hour in r["flows"].items()}
    tr = r["turning_ratios"]
    return TurningParameters(
        flows=flows,
        left_art=tr["left_art"], right_art=tr["right_art"],
        # new per-direction cross keys, fallback to old combined keys
        left_cross_nb=tr.get("left_cross_nb", tr.get("left_cross")),
        right_cross_nb=tr.get("right_cross_nb", tr.get("right_cross")),
        left_cross_sb=tr.get("left_cross_sb", tr.get("left_cross")),
        right_cross_sb=tr.get("right_cross_sb", tr.get("right_cross")),
        calibration_hours=tuple(r["calibration_hours"]),
    )


def validate_with_duarouter(net_file, trips_file):
    sumo_bin = "/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/bin/duarouter"
    if not os.path.exists(sumo_bin):
        sumo_bin = "duarouter"
    out = subprocess.run(
        [sumo_bin, "--net-file", net_file, "--route-files", trips_file,
         "--output-file", "/tmp/duarouter_validate.rou.xml", "--ignore-errors"],
        capture_output=True, text=True,
    )
    stderr = (out.stderr or "") + (out.stdout or "")
    n_no_conn = stderr.count("No connection between")
    n_no_route = stderr.count("has no valid route")
    return n_no_conn, n_no_route, stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=DEFAULT_RESULT)
    ap.add_argument("--boundary", default=DEFAULT_BOUNDARY)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--net", default=DEFAULT_NET)
    # Real-world TomTom probe penetration is ~10% and is uniform across road
    # types (same probe fleet drives main + side streets). Using a single 0.10 for
    # both keeps the RL reward's probe `count` an unbiased flow proxy (no main/cross
    # weighting bias). Penetration does not affect the speed-matched calibration —
    # it only relabels the calibrated total into probe/reg.
    ap.add_argument("--probe-prob", type=float, default=0.10)
    ap.add_argument("--probe-prob-cross", type=float, default=0.10)
    ap.add_argument("--validate", action="store_true",
                    help="Run duarouter on the generated file")
    args = ap.parse_args()

    print(f"Loading calibrated params from: {args.result}")
    p = load_params(args.result)
    print(f"  {p}")

    print(f"\nGenerating trips with boundary: {args.boundary}")
    print(f"  → {args.out}")
    p.to_flow_xml(
        boundaries_json=args.boundary,
        out_file=args.out,
        probe_prob=args.probe_prob,
        probe_prob_cross=args.probe_prob_cross,
    )

    n_flows = subprocess.check_output(
        ["grep", "-c", "<flow", args.out], text=True).strip()
    print(f"  Wrote {n_flows} <flow> entries")

    if args.validate:
        print(f"\nValidating with duarouter against {args.net}")
        n_no_conn, n_no_route, stderr = validate_with_duarouter(
            args.net, args.out)
        if n_no_conn == 0 and n_no_route == 0:
            print(f"  PASS: 0 'No connection' warnings, 0 invalid routes")
        else:
            print(f"  FAIL: {n_no_conn} 'No connection' warnings, "
                  f"{n_no_route} invalid routes")
            print("\n  First offending lines:")
            for line in stderr.split("\n"):
                if "No connection" in line or "no valid route" in line:
                    print("   ", line)
                    break
            sys.exit(1)


if __name__ == "__main__":
    main()

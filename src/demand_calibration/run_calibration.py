"""
CLI entry point for demand calibration.

Usage:
    python -m demand_calibration.run_calibration [options]

    # or from project root:
    python src/demand_calibration/run_calibration.py [options]

Example:
    python src/demand_calibration/run_calibration.py \
        --sumocfg data/corridor-osm-network/osm.sumocfg \
        --additional data/corridor-osm-network/detectors_v8_fixed.add.xml,data/corridor-osm-network/actuated_signals.add.xml \
        --approach-folder data/geojson-1-15-min-interval/processed \
        --net-file data/corridor-osm-network/osm.net.xml.gz \
        --output-dir outputs/calibration/ \
        --max-iter 30 \
        --arterial-factor 2.0
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Ensure src/ is on the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from demand_calibration.calibration_targets import CalibrationTargets
from demand_calibration.demand_parameters import DemandParameters
from demand_calibration.flow_parameters import FlowParameters
from demand_calibration.simulation_runner import SimulationRunner
from demand_calibration.objective import ObjectiveFunction
from demand_calibration.optimizer import SPSAOptimizer
from demand_calibration.ga_optimizer import GAOptimizer
from demand_calibration.turning_parameters import TurningParameters


def run_calibration(args):
    print("=" * 60)
    print(f"  Demand Calibration Pipeline (mode: {args.param_mode})")
    print("=" * 60)

    # Step 1: Load calibration targets
    print("\n[1] Loading TomTom calibration targets...")
    targets = CalibrationTargets(args.approach_folder)
    print(f"    {targets}")

    corridor = targets.get_corridor_targets()
    print(f"    Corridor avg speed range: "
          f"{corridor['target_speed_kph'].min():.1f} - "
          f"{corridor['target_speed_kph'].max():.1f} kph")

    # Step 2: Initialize simulation runner
    print("\n[2] Initializing simulation runner...")
    runner = SimulationRunner(
        sumocfg=args.sumocfg,
        additional_file=args.additional,
        approach_folder=args.approach_folder,
        net_file=args.net_file,
        sim_start_hour=args.start_hour,
        sim_duration_hours=args.duration_hours,
        probe_prob=args.probe_prob,
        probe_prob_cross=args.probe_prob_cross,
        boundaries_json=args.boundaries_json,
    )
    print(f"    Sim window: {args.start_hour}:00 - "
          f"{args.start_hour + args.duration_hours}:00")
    print(f"    Probe prob: {args.probe_prob}")

    # Step 3: Set up initial parameters
    print("\n[3] Setting initial parameters...")
    cal_hours = (args.start_hour, args.start_hour + args.duration_hours)
    if args.param_mode == "flow":
        initial_params = FlowParameters.from_fringe_flows_csv(
            csv_path=args.fringe_flows_csv,
            calibration_hours=cal_hours,
        )
        print(f"    {initial_params}")
        phase_name = "flow_calibration"
        spsa_c = args.spsa_c if args.spsa_c != 5.0 else 100.0
    elif args.param_mode == "turning":
        if args.seed_from_result:
            # warm-start from a previous calibration_result_turning.json (already
            # speed-calibrated); the GA only adapts it to the new signal plan.
            r = json.load(open(args.seed_from_result))
            flows = {k: {int(h): int(v) for h, v in per.items()}
                     for k, per in r["flows"].items()}
            tr = r["turning_ratios"]
            import copy
            # accept both the new per-direction keys and the old combined cross keys
            initial_params = TurningParameters(
                flows=flows, left_art=tr["left_art"], right_art=tr["right_art"],
                left_cross_nb=tr.get("left_cross_nb", tr.get("left_cross")),
                right_cross_nb=tr.get("right_cross_nb", tr.get("right_cross")),
                left_cross_sb=tr.get("left_cross_sb", tr.get("left_cross")),
                right_cross_sb=tr.get("right_cross_sb", tr.get("right_cross")),
                calibration_hours=tuple(r["calibration_hours"]),
                _initial_flows=copy.deepcopy(flows))   # anchor project_bounds to seed
            print(f"    warm-start from {args.seed_from_result}")
        else:
            initial_params = TurningParameters.from_fringe_flows_csv(
                csv_path=args.fringe_flows_csv,
                calibration_hours=cal_hours,
            )
        # per-movement-type GA search bounds (factor x seed flow)
        initial_params.art_lower_factor, initial_params.art_upper_factor = args.art_bounds
        initial_params.cross_lower_factor, initial_params.cross_upper_factor = args.cross_bounds
        print(f"    {initial_params}")
        print(f"    bounds  arterial x[{args.art_bounds[0]},{args.art_bounds[1]}]  "
              f"cross x[{args.cross_bounds[0]},{args.cross_bounds[1]}]")
        phase_name = "turning_calibration"
        spsa_c = args.spsa_c if args.spsa_c != 5.0 else 100.0
    else:
        initial_params = DemandParameters(
            calibration_hours=cal_hours,
            arterial_factor=args.arterial_factor,
        )
        print(f"    {initial_params}")
        phase_name = "phase1_volume"
        spsa_c = args.spsa_c

    # Step 4: Create objective function
    objective = ObjectiveFunction(targets, runner)
    obj_fn = (objective.evaluate
              if args.param_mode in ("flow", "turning")
              else objective.evaluate_corridor)

    # Optional parallel batch evaluator (A): score each generation's population
    # on a process pool, each worker using the probe collector (B). Only for the
    # turning param-mode (the picklable TurningParameters template).
    batch_fn = None
    par_obj = None
    if args.n_workers > 1 and args.param_mode == "turning":
        from demand_calibration.parallel_eval import ParallelObjective
        cfg = dict(sumocfg=args.sumocfg, additional=args.additional,
                   approach_folder=args.approach_folder, net_file=args.net_file,
                   start_hour=args.start_hour, hours=args.duration_hours,
                   probe_prob=args.probe_prob, probe_prob_cross=args.probe_prob_cross,
                   boundaries_json=args.boundaries_json, stop_bar_only=True,
                   template=initial_params)
        par_obj = ParallelObjective(cfg, args.n_workers)
        batch_fn = par_obj.evaluate_batch
        print(f"    Parallel eval: {args.n_workers} workers, probe collector")

    # Step 5: Run optimization
    if args.optimizer == "ga":
        print("\n[4] Running GA optimization...")
        optimizer = GAOptimizer(
            objective_fn=obj_fn,
            initial_params=initial_params,
            population_size=args.ga_pop,
            n_generations=args.ga_gens,
            elite_frac=args.ga_elite_frac,
            mutation_rate=args.ga_mut_rate,
            mutation_sigma=args.ga_mut_sigma,
            crossover_rate=args.ga_cx_rate,
            init_jitter=args.ga_init_jitter,
            output_dir=args.output_dir,
            phase_name=phase_name,
            seed=args.ga_seed,
            batch_objective_fn=batch_fn,
        )
    else:
        print("\n[4] Running SPSA optimization...")
        optimizer = SPSAOptimizer(
            objective_fn=obj_fn,
            initial_params=initial_params,
            max_iterations=args.max_iter,
            a=args.spsa_a,
            c=spsa_c,
            convergence_threshold=0.05,
            output_dir=args.output_dir,
            phase_name=phase_name,
        )

    best_params, best_z, log = optimizer.optimize()
    if par_obj is not None:
        par_obj.close()

    # Step 6: Report results
    print("\n" + "=" * 60)
    print("  Calibration Results")
    print("=" * 60)
    print(f"  Best WRMSE: {best_z:.2f} kph")
    print(f"  Total simulation runs: {runner.run_count}")

    # Save calibrated parameters
    result = {
        "timestamp": datetime.now().isoformat(),
        "param_mode": args.param_mode,
        "best_wrmse": round(best_z, 3),
        "total_runs": runner.run_count,
        "calibration_hours": list(cal_hours),
    }

    if args.param_mode == "flow":
        result["flows"] = best_params.flows
    elif args.param_mode == "turning":
        result["flows"] = best_params.flows
        result["turning_ratios"] = {
            "left_art":       best_params.left_art,
            "right_art":      best_params.right_art,
            "left_cross_nb":  best_params.left_cross_nb,
            "right_cross_nb": best_params.right_cross_nb,
            "left_cross_sb":  best_params.left_cross_sb,
            "right_cross_sb": best_params.right_cross_sb,
        }
    else:
        result["hourly_volumes"] = best_params.hourly_volumes
        result["arterial_factor"] = best_params.arterial_factor

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = {"flow": "flow", "turning": "turning", "legacy": "legacy"}[args.param_mode]
    result_path = os.path.join(args.output_dir, f"calibration_result_{suffix}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved: {result_path}")

    return best_params, best_z


def main():
    parser = argparse.ArgumentParser(description="Demand calibration via SPSA")
    parser.add_argument("--sumocfg",
                        default="data/corridor-osm-network/osm.sumocfg")
    parser.add_argument("--additional",
                        default="data/corridor-osm-network/detectors_v8_fixed.add.xml,"
                                "data/corridor-osm-network/actuated_signals.add.xml")
    parser.add_argument("--approach-folder",
                        default="data/geojson-1-15-min-interval/processed")
    parser.add_argument("--net-file",
                        default="data/corridor-osm-network/osm.net.xml.gz")
    parser.add_argument("--output-dir",
                        default="outputs/calibration/")
    parser.add_argument("--start-hour", type=int, default=7,
                        help="Simulation start hour (default: 7)")
    parser.add_argument("--duration-hours", type=int, default=6,
                        help="Simulation duration in hours (default: 6)")
    parser.add_argument("--probe-prob", type=float, default=0.3478,
                        help="Probe penetration rate for arterial EB/WB (default: 0.3478)")
    parser.add_argument("--probe-prob-cross", type=float, default=0.75,
                        help="Probe penetration rate for cross-street NB/SB (default: 0.75, range 0.5-1.0)")
    parser.add_argument("--arterial-factor", type=float, default=2.0,
                        help="[legacy mode] Initial arterial factor (default: 2.0)")
    parser.add_argument("--param-mode",
                        choices=["flow", "turning", "legacy"],
                        default="flow",
                        help="'flow' = point-to-point through-flows (78 params); "
                             "'turning' = flows + 4 turning ratios (82 params); "
                             "'legacy' = randomTrips-based")
    parser.add_argument("--fringe-flows-csv",
                        default="outputs/calibration/fringe_flows.csv",
                        help="[flow mode] Initial flow values")
    parser.add_argument("--boundaries-json",
                        default="outputs/calibration/corridor_boundary_edges.json",
                        help="[flow mode] Corridor boundary edges")
    parser.add_argument("--max-iter", type=int, default=30,
                        help="Max SPSA iterations (default: 30)")
    parser.add_argument("--spsa-a", type=float, default=100.0,
                        help="SPSA gain parameter a")
    parser.add_argument("--spsa-c", type=float, default=5.0,
                        help="SPSA perturbation parameter c (auto-raised for flow mode)")
    # GA options
    parser.add_argument("--optimizer", choices=["spsa", "ga"], default="ga",
                        help="Optimization algorithm (default: ga)")
    parser.add_argument("--ga-pop", type=int, default=12,
                        help="GA population size (default: 12)")
    parser.add_argument("--ga-gens", type=int, default=20,
                        help="GA number of generations (default: 20)")
    parser.add_argument("--ga-elite-frac", type=float, default=0.25,
                        help="GA elite fraction (default: 0.25)")
    parser.add_argument("--ga-mut-rate", type=float, default=0.2,
                        help="GA per-gene mutation probability (default: 0.2)")
    parser.add_argument("--ga-mut-sigma", type=float, default=0.15,
                        help="GA mutation sigma as frac of bound width (default: 0.15)")
    parser.add_argument("--ga-cx-rate", type=float, default=0.7,
                        help="GA crossover probability (default: 0.7)")
    parser.add_argument("--ga-init-jitter", type=float, default=0.25,
                        help="GA initial population jitter (default: 0.25)")
    parser.add_argument("--ga-seed", type=int, default=42,
                        help="GA random seed (default: 42)")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel sim workers for population eval (turning "
                             "mode only); each uses the probe collector. 1 = serial.")
    parser.add_argument("--seed-from-result", default=None,
                        help="Warm-start turning GA from a calibration_result_*.json "
                             "instead of fringe_flows.csv (adapts it to a new signal).")
    parser.add_argument("--art-bounds", type=float, nargs=2, default=[0.7, 2.5],
                        metavar=("LOW", "HIGH"),
                        help="Arterial (EB/WB) flow GA bounds as factor x seed.")
    parser.add_argument("--cross-bounds", type=float, nargs=2, default=[0.9, 5.0],
                        metavar=("LOW", "HIGH"),
                        help="Cross-street (NB/SB) flow GA bounds as factor x seed.")
    args = parser.parse_args()

    run_calibration(args)


if __name__ == "__main__":
    main()

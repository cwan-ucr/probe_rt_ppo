"""Evaluate every recorded 310-gene GA generation on PPO's three online scenarios.

This is post-hoc monitoring only: held-out results never select or modify a GA
genome.  The first three scenarios of the shared 72-point design are exactly
those used by PPO's ``--eval-scenarios 3`` online evaluation.
"""

import argparse
import csv
import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import numpy as np

# The GA module sets RL_CONTROL_LAYOUT=all10 before importing config. Import it
# first; otherwise this evaluator silently initializes the default layout.
from . import train_ga_static_offset as ga
from . import config as C
from . import demand, scenarios
from .traci_ports import resolve_port_base


METRICS = ("probe_delay_vehh", "speed_kmh", "wait_s_per_veh",
           "delay_s_per_veh")


def load_generation_winners(path, n_genes, expected_generations):
    winners = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            generation = int(row["generation"])
            if generation != len(winners) + 1:
                raise ValueError(f"GA training log has a gap at generation {generation}")
            genome = tuple(json.loads(row["best_adjustments_s"]))
            if (len(genome) != n_genes or
                    any(type(value) is not int or value < ga._LOW or
                        value > ga._HIGH for value in genome)):
                raise ValueError(f"invalid {n_genes}-gene winner at generation "
                                 f"{generation}")
            winners.append(genome)
    if len(winners) != expected_generations:
        raise ValueError(f"expected {expected_generations} GA generations, "
                         f"found {len(winners)}")
    return winners


def _evaluate_one(task):
    generation, scenario, params, genome = task
    env = ga._ENV
    try:
        # PPO's online evaluation and the GA's final held-out evaluation both
        # use per-vehicle tripinfo. SUMO's aggregate statistics are rounded and
        # fail the exact baseline-realization check below.
        env.set_metrics_mode("tripinfo")
        env.set_demand(params)
        probe_delay, info = ga._rollout(env, genome)
        return {
            "generation": generation,
            "scenario": scenario,
            "g": params[0], "main": params[1],
            "side": params[2], "direction": params[3],
            "probe_delay_vehh": probe_delay,
            "speed_kmh": float(info["vmt_per_vht"]) * 1.609344,
            "wait_s_per_veh": float(info["wait_s_per_veh"]),
            "delay_s_per_veh": float(info["timeloss_s_per_veh"]),
            "n_veh": int(info["n_veh"]),
        }
    finally:
        env.close()


def _load_completed_rows(path, n_eval, n_generations):
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_generation = defaultdict(list)
    for row in rows:
        generation, scenario = int(row["generation"]), int(row["scenario"])
        if not 1 <= generation <= n_generations or not 0 <= scenario < n_eval:
            raise ValueError(f"unexpected generation/scenario in {path}")
        by_generation[generation].append(row)
    for generation, group in by_generation.items():
        if sorted(int(row["scenario"]) for row in group) != list(range(n_eval)):
            raise ValueError(f"incomplete/duplicated generation {generation} "
                             f"in {path}")
    return by_generation


def _check_reference_baselines(path, baselines):
    if not path.exists():
        raise FileNotFoundError(f"final GA evaluation is required to verify "
                                f"the three fixed baselines: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reference = {int(row["scenario"]): row for row in csv.DictReader(stream)}
    mapping = {
        "probe_delay_vehh": "inc_probe_delay_vehh",
        "speed_kmh": "inc_speed_kmh",
        "wait_s_per_veh": "inc_wait",
        "delay_s_per_veh": "inc_loss",
    }
    for scenario, baseline in baselines.items():
        if scenario not in reference:
            raise ValueError(f"scenario {scenario} absent from {path}")
        for metric, column in mapping.items():
            expected = float(reference[scenario][column])
            if not np.isclose(baseline[metric], expected,
                              rtol=1e-8, atol=1e-6):
                raise ValueError(f"baseline mismatch: scenario {scenario}, "
                                 f"{metric}: observed={baseline[metric]:.9g}, "
                                 f"reference={expected:.9g}; check PPO/GA "
                                 "scenario realization")


def _attach_baseline(row, baseline):
    result = dict(row)
    for metric in METRICS:
        result[f"baseline_{metric}"] = baseline[metric]
    return result


def summarize(rows_by_generation, population):
    summary = []
    for generation in sorted(rows_by_generation):
        rows = rows_by_generation[generation]
        result = {"generation": generation,
                  "training_rollouts": generation * population,
                  "eval_count": len(rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            baseline = [float(row[f"baseline_{metric}"]) for row in rows]
            result[f"eval_{metric}"] = float(np.mean(values))
            result[f"eval_{metric}_min"] = float(np.min(values))
            result[f"eval_{metric}_max"] = float(np.max(values))
            result[f"baseline_{metric}"] = float(np.mean(baseline))
        summary.append(result)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="coord_actuated")
    parser.add_argument("--run-tag", default=(
        "ga_interval310_probe_delay_v1_ottawa_bank1000_all10_8h"))
    parser.add_argument("--eval-scenarios", type=Path,
                        default=scenarios.DEFAULT_EVAL_FILE)
    parser.add_argument("--n-eval", type=int, default=3)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-generations", type=int, default=16)
    parser.add_argument("--port-base", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    args.backbone = C.canonical_backbone(args.backbone)
    if args.n_eval != 3:
        parser.error("this comparison requires PPO's first three eval scenarios")
    if args.workers < 1 or args.batch_generations < 1:
        parser.error("workers and batch-generations must be positive")
    if C.RL_METHOD != "probe" or C.CONTROL_LAYOUT != "all10" or \
            C.SPLIT_CONTROL or not C.RESIDUAL_ACTION or \
            not demand.explicit_demand_enabled():
        parser.error("requires probe/all10/residual/offset-only/explicit demand")

    output_dir = args.output_dir or Path(C.rl_output_dir(args.backbone))
    stem = f"{args.run_tag}_{args.backbone}"
    log_path = output_dir / f"ga_interval_training_{stem}.csv"
    solution_path = output_dir / f"ga_interval_solution_{stem}.json"
    final_eval_path = output_dir / f"eval_ga_interval_{stem}.csv"
    rows_path = output_dir / f"ga_interval_online_eval3_{stem}_rows.csv"
    summary_path = output_dir / f"ga_interval_online_eval3_{stem}_summary.csv"
    manifest_path = output_dir / f"ga_interval_online_eval3_{stem}_manifest.json"

    solution = json.loads(solution_path.read_text(encoding="utf-8"))
    signature = solution["signature"]
    expected_intervals = ((int(signature["episode_end"]) - C.WARMUP_S) //
                          C.DECISION_INTERVAL_S)
    n_genes = expected_intervals * C.N_AGENTS
    from .train_ppo_parallel import _sha256, _sha256_tree
    current_inputs = {
        "signal_sha256": _sha256(C.signal_path(args.backbone)),
        "trips_sha256": _sha256(C.TRIPS),
        "detectors_sha256": _sha256(C.DETECTORS),
        "approach_geometry_sha256": _sha256_tree(C.APPROACH_DIR),
    }
    if signature["schema"] != "ga_interval_residual_probe_delay_v1" or \
            signature["backbone"] != args.backbone or n_genes != 310 or \
            signature["eval_scenarios_sha256"] != scenarios.file_sha256(
                args.eval_scenarios) or \
            any(signature[key] != value for key, value in
                current_inputs.items()) or \
            not np.isclose(signature["probe_penetration"],
                           demand.probe_penetration()):
        raise ValueError("GA solution, environment, or evaluation design "
                         "does not match the 310-gene experiment")
    winners = load_generation_winners(log_path, n_genes,
                                      int(signature["generations"]))
    all_eval, _ = scenarios.load_scenario_file(args.eval_scenarios)
    eval_params = all_eval[:args.n_eval]
    manifest = {
        "source_log_sha256": scenarios.file_sha256(log_path),
        "source_solution_sha256": scenarios.file_sha256(solution_path),
        "eval_scenarios_sha256": scenarios.file_sha256(args.eval_scenarios),
        "scenario_indices": list(range(args.n_eval)),
        "backbone": args.backbone,
        "seed": int(signature["seed"]),
        "episode_end": int(signature["episode_end"]),
        "n_generations": len(winners),
        "n_genes": n_genes,
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("existing online-evaluation files have different "
                             "inputs; use a distinct run tag")
    else:
        if rows_path.exists() or summary_path.exists():
            raise ValueError("output CSV exists without manifest; refusing to "
                             "mix evaluation runs")
        output_dir.mkdir(parents=True, exist_ok=True)
        ga._atomic_json(manifest_path, manifest)

    completed = _load_completed_rows(rows_path, args.n_eval, len(winners))
    port_base = resolve_port_base(args.port_base, required_span=args.workers)
    ctx = mp.get_context("spawn")
    counter, lock = ctx.Value("i", 0), ctx.Lock()
    with ctx.Pool(args.workers, initializer=ga._worker_init,
                  initargs=(args.backbone, int(signature["seed"]),
                            int(signature["episode_end"]), port_base,
                            counter, lock)) as pool:
        baseline_tasks = [(0, scenario, params, (0,) * n_genes)
                          for scenario, params in enumerate(eval_params)]
        baselines = {row["scenario"]: row for row in
                     pool.imap_unordered(_evaluate_one, baseline_tasks)}
        _check_reference_baselines(final_eval_path, baselines)
        print("verified PPO scenarios 0, 1, 2 against final GA baselines",
              flush=True)

        pending = [generation for generation in range(1, len(winners) + 1)
                   if generation not in completed]
        for start in range(0, len(pending), args.batch_generations):
            batch = pending[start:start + args.batch_generations]
            tasks = [(generation, scenario, params, winners[generation - 1])
                     for generation in batch
                     for scenario, params in enumerate(eval_params)]
            new_rows = [_attach_baseline(row, baselines[row["scenario"]])
                        for row in pool.imap_unordered(_evaluate_one, tasks,
                                                       chunksize=1)]
            for generation in batch:
                group = [row for row in new_rows
                         if row["generation"] == generation]
                completed[generation] = sorted(group,
                                               key=lambda row: row["scenario"])
            all_rows = [row for generation in sorted(completed)
                        for row in completed[generation]]
            ga._write_csv(rows_path, all_rows)
            ga._write_csv(summary_path,
                          summarize(completed, int(signature["population"])))
            print(f"evaluated generations {batch[0]}–{batch[-1]} / "
                  f"{len(winners)} ({len(completed)} complete)", flush=True)

        if not pending:
            ga._write_csv(summary_path,
                          summarize(completed, int(signature["population"])))

    print(f"online evaluation complete: {rows_path}\nsummary: {summary_path}",
          flush=True)


if __name__ == "__main__":
    main()

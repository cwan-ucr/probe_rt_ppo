"""Train demand-robust residual offset vectors by GA.

Static mode uses one integer adjustment per intersection for all decisions.
Interval mode uses a separate adjustment per intersection and 15-minute
policy-controlled interval. Both add adjustments to the active program's
incumbent offsets. Unlike PPO, neither schedule reacts to observations or
changes between demand scenarios. Fitness is the mean episode-total
probe-estimated delay, using the estimator in ``corridor_env.step``.

Each of 500 generations evaluates 24 genomes on one shared training scenario
(CRN): 24 rollouts/update and 12,000 training rollouts, matching PPO's rollout
budget.  A bounded set of generation winners is re-ranked on a common sample
from the *training* bank; held-out evaluation is used only after selection.
"""
import os

# These settings must precede imports of config and the environment in the main
# process and in spawned workers.
os.environ["RL_CONTROL_LAYOUT"] = "all10"
os.environ["RL_METHOD"] = "probe"
os.environ["RL_RESIDUAL"] = "1"
os.environ["RL_SPLIT"] = "0"
os.environ.setdefault("RL_COLLECTOR", "probe")
os.environ.setdefault("RL_EXPLICIT_DEMAND", "1")
os.environ.setdefault("RL_EXPLICIT_DEMAND_SEED", "42")
os.environ.setdefault("RL_PROBE_MEMBERSHIP_SEED", "42")
os.environ.setdefault("RL_PROBE_PENETRATION", "0.1")

import argparse
import csv
import json
import multiprocessing as mp
import re
import time
from pathlib import Path

import numpy as np

from . import config as C
from . import demand, scenarios
from .corridor_env_all10 import CorridorEnvAll10
from .traci_ports import resolve_port_base


_ENV = None
_ZERO = tuple(0 for _ in C.AGENT_ORDER)
_LOW = -int(C.ACTION_HALF_RANGE)
_HIGH = int(C.ACTION_HALF_RANGE) - 1


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def _write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _sample_indices(rng, order, cursor, n):
    """One no-replacement pass through the nested training bank at a time."""
    selected = []
    while len(selected) < n:
        if cursor == len(order):
            order = rng.permutation(len(order)).tolist()
            cursor = 0
        take = min(n - len(selected), len(order) - cursor)
        selected.extend(order[cursor:cursor + take])
        cursor += take
    return selected, order, cursor


def _initial_population(size, rng, n_genes, schedule):
    zero = (0,) * n_genes
    population = [zero]
    seen = {zero}
    while len(population) < size:
        if schedule == "interval":
            # Start with viable low-dimensional schedules, then let crossover
            # and mutation discover interval-specific corrections.
            if len(population) % 2:
                block = rng.integers(_LOW, _HIGH + 1, size=C.N_AGENTS)
                candidate = np.tile(block, n_genes // C.N_AGENTS)
            else:
                blocks = rng.integers(-10, 11, size=(4, C.N_AGENTS))
                candidate = np.repeat(blocks, 8, axis=0)[:
                    n_genes // C.N_AGENTS].reshape(-1)
            genome = tuple(int(x) for x in candidate)
        else:
            genome = tuple(int(x) for x in rng.integers(
                _LOW, _HIGH + 1, size=n_genes))
        if genome not in seen:
            seen.add(genome)
            population.append(genome)
    return population


def _next_population(population, scores, rng, elite_count, mutation_rate,
                     mutation_sigma, n_genes):
    zero = (0,) * n_genes
    order = np.argsort(scores, kind="stable")
    winners = [population[int(i)] for i in order[:elite_count]]
    next_pop = [zero]
    seen = {zero}
    for genome in winners:
        if genome not in seen:
            next_pop.append(genome)
            seen.add(genome)
    attempts = 0
    while len(next_pop) < len(population):
        attempts += 1
        if attempts > len(population) * 1000:
            raise RuntimeError("could not generate distinct GA offspring")
        parents = []
        for _ in range(2):
            contenders = rng.choice(len(population), size=min(3, len(population)),
                                    replace=False)
            parents.append(population[min(contenders, key=lambda i: scores[i])])
        mask = rng.random(n_genes) < 0.5
        child = np.where(mask, parents[0], parents[1]).astype(int)
        mutate = rng.random(n_genes) < mutation_rate
        if not mutate.any():
            mutate[int(rng.integers(n_genes))] = True
        child[mutate] += np.rint(rng.normal(
            0.0, mutation_sigma, size=int(mutate.sum()))).astype(int)
        child = np.clip(child, _LOW, _HIGH)
        genome = tuple(int(x) for x in child)
        if genome not in seen:
            next_pop.append(genome)
            seen.add(genome)
    return next_pop


def _selection_finalists(archive, history, limit, n_genes):
    """Bound the common-scenario re-ranking cost after a long GA run.

    Keep the incumbent, recent winners, and winners with the largest paired
    improvement on their generation's common scenario.  The latter ranking is
    only a shortlist; final selection uses the same training scenarios for
    every shortlisted genome.
    """
    zero = (0,) * n_genes
    finalists = [zero]
    seen = {zero}

    def add(genome):
        candidate = tuple(genome)
        if candidate not in seen and len(finalists) < limit:
            finalists.append(candidate)
            seen.add(candidate)

    for genome in reversed(archive[-(limit // 2):]):
        add(genome)
    ranked = sorted(zip(archive, history),
                    key=lambda item: item[1]["best_delta_vs_inc_vehh"],
                    reverse=True)
    for genome, _row in ranked:
        add(genome)
    return finalists


def _worker_init(backbone, seed, episode_end, port_base, counter, lock):
    global _ENV
    with lock:
        worker_index = counter.value
        counter.value += 1
    _ENV = CorridorEnvAll10(backbone=backbone, seed=seed,
                            episode_end=episode_end, port=port_base + worker_index)


def _rollout(env, genome):
    """Return unscaled probe delay and SUMO physical metrics for one episode."""
    actions = np.asarray(genome, dtype=np.float32).reshape(-1, C.N_AGENTS)
    actions /= C.ACTION_HALF_RANGE
    n_steps = (env.episode_end - C.WARMUP_S) // C.DECISION_INTERVAL_S
    if len(actions) not in (1, n_steps):
        raise ValueError(f"expected 1 or {n_steps} action rows, got {len(actions)}")
    env.baseline = None
    env.reset()
    done = False
    total_delay = 0.0
    info = {}
    step = 0
    while not done:
        action = actions[0 if len(actions) == 1 else step]
        _state, _reward, done, info = env.step(action)
        total_delay += float(info["total_delay_vehh"])
        step += 1
    if step != n_steps:
        raise RuntimeError(f"GA rollout ended after {step}/{n_steps} decisions")
    if not np.isfinite(total_delay):
        raise RuntimeError(f"nonfinite probe delay for genome {genome}")
    return total_delay, info


def _fitness_task(task):
    candidate, genome, scenario, params = task
    env = _ENV
    try:
        env.set_metrics_mode("statistics")
        env.set_demand(params)
        delay, _info = _rollout(env, genome)
        return candidate, scenario, delay
    finally:
        env.close()


def _evaluate_population(pool, population, scenario_indices, bank):
    tasks = [(candidate, genome, index, bank[index])
             for candidate, genome in enumerate(population)
             for index in scenario_indices]
    totals = np.zeros(len(population), dtype=float)
    counts = np.zeros(len(population), dtype=int)
    for candidate, _scenario, delay in pool.imap_unordered(
            _fitness_task, tasks, chunksize=1):
        totals[candidate] += delay
        counts[candidate] += 1
    if not np.all(counts == len(scenario_indices)):
        raise RuntimeError("GA fitness batch is incomplete")
    return totals / counts


def _eval_task(task):
    scenario, params, genome, origin_metrics = task
    env = _ENV
    try:
        env.set_metrics_mode("tripinfo" if origin_metrics else "statistics")
        env.keep_metrics_files = origin_metrics
        env.set_demand(params)
        baseline_probe, baseline = _rollout(env, _ZERO)
        if origin_metrics:
            from .eval_policy import _origin_group_trip_metrics
            baseline_groups = _origin_group_trip_metrics(env._tripinfo)
        ga_probe, ga = _rollout(env, genome)
        if origin_metrics:
            ga_groups = _origin_group_trip_metrics(env._tripinfo)
        speed0, speed1 = baseline["vmt_per_vht"], ga["vmt_per_vht"]
        wait0, wait1 = baseline["wait_s_per_veh"], ga["wait_s_per_veh"]
        loss0, loss1 = baseline["timeloss_s_per_veh"], ga["timeloss_s_per_veh"]
        saving = lambda before, after: (
            100.0 * (before - after) / before if before else 0.0)
        row = {
            "scenario": scenario, "g": params[0], "main": params[1],
            "side": params[2], "direction": params[3],
            "inc_probe_delay_vehh": baseline_probe,
            "ga_probe_delay_vehh": ga_probe,
            "probe_delay_saving_pct": saving(baseline_probe, ga_probe),
            "inc_mph": speed0, "pol_mph": speed1,
            "inc_speed_kmh": speed0 * 1.609344,
            "ga_speed_kmh": speed1 * 1.609344,
            "dmph_pct": 100.0 * (speed1 - speed0) / speed0 if speed0 else 0.0,
            "inc_wait": wait0, "pol_wait": wait1,
            "dwait_pct": 100.0 * (wait1 - wait0) / wait0 if wait0 else 0.0,
            "waiting_saving_pct": saving(wait0, wait1),
            "inc_loss": loss0, "pol_loss": loss1,
            "dloss_pct": 100.0 * (loss1 - loss0) / loss0 if loss0 else 0.0,
            "delay_saving_pct": saving(loss0, loss1),
            "n_veh": baseline["n_veh"],
            "inc_vmt_mi": baseline["vmt_mi"],
            "pol_vmt_mi": ga["vmt_mi"],
            "inc_vht_h": baseline["vht_h"],
            "pol_vht_h": ga["vht_h"],
        }
        if origin_metrics:
            from .eval_policy import _add_origin_group_metrics
            _add_origin_group_metrics(row, baseline_groups, ga_groups)
        return row
    finally:
        env.close()
        if env.keep_metrics_files:
            Path(env._tripinfo).unlink(missing_ok=True)


def _signature(args, train_file, eval_file, master_indices, n_genes):
    from .train_ppo_parallel import _sha256, _sha256_tree
    signature = {
        "schema": ("ga_static_residual_probe_delay_v1" if args.schedule == "static"
                   else "ga_interval_residual_probe_delay_v1"),
        "backbone": args.backbone,
        "agent_order": list(C.AGENT_ORDER),
        "cycle_s": C.CYCLE_S,
        "episode_end": args.episode_end,
        "seed": args.seed,
        "master_bank_size": args.master_bank_size,
        "bank_limit": args.bank_limit,
        "subset_seed": args.subset_seed,
        "master_indices": master_indices,
        "train_scenarios_sha256": scenarios.file_sha256(train_file),
        "eval_scenarios_sha256": scenarios.file_sha256(eval_file),
        "signal_sha256": _sha256(C.signal_path(args.backbone)),
        "trips_sha256": _sha256(C.TRIPS),
        "detectors_sha256": _sha256(C.DETECTORS),
        "approach_geometry_sha256": _sha256_tree(C.APPROACH_DIR),
        "probe_penetration": demand.probe_penetration(),
        "collector": os.environ["RL_COLLECTOR"],
        "explicit_demand": demand.explicit_demand_enabled(),
        "explicit_demand_seed": demand.explicit_demand_seed(),
        "probe_membership_seed": demand.probe_membership_seed(),
        "use_expected_delay": C.USE_EXPECTED_DELAY,
        "expected_delay_quadrature": C.EXPECTED_DELAY_QUADRATURE,
        "fitness": "mean_episode_total_probe_estimated_delay_vehh",
        "population": args.population,
        "generations": args.generations,
        "scenario_batch": args.scenario_batch,
        "selection_scenarios": args.selection_scenarios,
        "selection_candidates": args.selection_candidates,
        "eval_count": args.eval_count,
        "save_origin_group_metrics": args.save_origin_group_metrics,
        "elite_count": args.elite_count,
        "mutation_rate": args.mutation_rate,
        "mutation_sigma": args.mutation_sigma,
    }
    if args.schedule == "interval":
        signature["schedule"] = args.schedule
        signature["n_genes"] = n_genes
    return signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", choices=("static", "interval"),
                        default="static", help="one vector for all decisions or "
                        "one vector per 15-minute controlled interval")
    parser.add_argument("--backbone", default="coord_actuated")
    parser.add_argument("--train-scenarios", type=Path,
                        default=scenarios.DEFAULT_TRAIN_FILE)
    parser.add_argument("--eval-scenarios", type=Path,
                        default=scenarios.DEFAULT_EVAL_FILE)
    parser.add_argument("--master-bank-size", type=int, default=2000)
    parser.add_argument("--bank-limit", type=int, default=1000)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--scenario-batch", type=int, default=1)
    parser.add_argument("--selection-scenarios", type=int, default=72)
    parser.add_argument("--selection-candidates", type=int, default=24)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--mutation-sigma", type=float, default=5.0)
    parser.add_argument("--eval-count", type=int, default=72)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--episode-end", type=int, default=28800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port-base", type=int, default=0)
    parser.add_argument("--run-tag", default="ga_static_probe_delay_v2_ottawa_bank1000_all10_8h")
    parser.add_argument("--output-dir", type=Path,
                        help="artifact directory (default: outputs/rl/<backbone>)")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--save-origin-group-metrics", action="store_true")
    parser.add_argument("--no-eval", action="store_true", help="training-only smoke test")
    args = parser.parse_args()
    args.backbone = C.canonical_backbone(args.backbone)
    if C.RL_METHOD != "probe" or C.CONTROL_LAYOUT != "all10" or C.SPLIT_CONTROL \
            or not C.RESIDUAL_ACTION or C.N_ACT_PER_INT != 1:
        parser.error("GA requires probe/all10/residual/offset-only control")
    if os.environ["RL_COLLECTOR"] != "probe" or not demand.explicit_demand_enabled():
        parser.error("GA requires RL_COLLECTOR=probe and RL_EXPLICIT_DEMAND=1")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_tag):
        parser.error("--run-tag may contain only letters, digits, '_' and '-'")
    if not 2 <= args.population or not 1 <= args.elite_count < args.population:
        parser.error("population must be >=2 and elite-count in [1,population)")
    if not 0.0 <= args.mutation_rate <= 1.0 or args.mutation_sigma <= 0:
        parser.error("mutation rate must be in [0,1] and sigma positive")
    if args.generations < 1 or args.workers < 1 or args.eval_count < 1:
        parser.error("generations, workers and eval-count must be positive")
    if args.episode_end <= C.WARMUP_S or (
            args.episode_end - C.WARMUP_S) % C.DECISION_INTERVAL_S:
        parser.error("episode-end must equal warmup plus whole decision intervals")
    n_intervals = (args.episode_end - C.WARMUP_S) // C.DECISION_INTERVAL_S
    n_genes = C.N_AGENTS * (n_intervals if args.schedule == "interval" else 1)
    zero = (0,) * n_genes
    all_train, _meta = scenarios.load_scenario_file(
        args.train_scenarios, expected_n=args.master_bank_size)
    all_eval, _meta = scenarios.load_scenario_file(args.eval_scenarios)
    if not 1 <= args.bank_limit <= len(all_train):
        parser.error("bank-limit must be between 1 and master-bank-size")
    if not 1 <= args.scenario_batch <= args.bank_limit:
        parser.error("scenario-batch must be between 1 and bank-limit")
    if not 1 <= args.selection_scenarios <= args.bank_limit:
        parser.error("selection-scenarios must be between 1 and bank-limit")
    if not 2 <= args.selection_candidates <= args.population:
        parser.error("selection-candidates must be between 2 and population")
    if args.eval_count > len(all_eval):
        parser.error("eval-count exceeds the held-out evaluation design")
    subset_rng = np.random.default_rng(args.subset_seed)
    master_indices = sorted(int(i) for i in subset_rng.permutation(
        len(all_train))[:args.bank_limit])
    bank = [all_train[index] for index in master_indices]
    signature = _signature(args, args.train_scenarios, args.eval_scenarios,
                           master_indices, n_genes)
    output_dir = args.output_dir or Path(C.rl_output_dir(args.backbone))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.run_tag}_{args.backbone}"
    prefix = "ga_static" if args.schedule == "static" else "ga_interval"
    state_path = output_dir / f"{prefix}_trainstate_{stem}.json"
    log_path = output_dir / f"{prefix}_training_{stem}.csv"
    solution_path = output_dir / f"{prefix}_solution_{stem}.json"
    eval_path = output_dir / f"eval_{prefix}_{stem}.csv"
    summary_path = output_dir / f"eval_{prefix}_{stem}_summary.json"

    rng = np.random.default_rng(args.seed)
    if args.auto_resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["signature"] != signature:
            raise RuntimeError("GA training state has incompatible configuration")
        rng.bit_generator.state = state["rng_state"]
        print(f"resuming GA at generation {state['generation_next']}/"
              f"{args.generations}", flush=True)
    else:
        state = {
            "signature": signature, "generation_next": 0,
            "population": [list(g) for g in _initial_population(
                args.population, rng, n_genes, args.schedule)],
            "archive": [], "history": [],
            "scenario_order": rng.permutation(len(bank)).tolist(),
            "scenario_cursor": 0,
            "rng_state": rng.bit_generator.state,
        }
        if state_path.exists():
            raise RuntimeError(f"existing state at {state_path}; use --auto-resume "
                               "or a different --run-tag")
    if state["history"]:
        _write_csv(log_path, state["history"])
    print(f"GA: schedule={args.schedule} genes={n_genes} "
          f"backbone={args.backbone} bank={len(bank)}/{len(all_train)} "
          f"pop={args.population} gens={args.generations} "
          f"batch={args.scenario_batch} workers={args.workers} "
          f"probe_p={demand.probe_penetration()} "
          f"expected_training_rollouts={args.population * args.generations * args.scenario_batch}",
          flush=True)
    port_base = resolve_port_base(args.port_base, required_span=args.workers)
    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    lock = ctx.Lock()
    with ctx.Pool(args.workers, initializer=_worker_init,
                  initargs=(args.backbone, args.seed, args.episode_end,
                            port_base, counter, lock)) as pool:
        for generation in range(state["generation_next"], args.generations):
            tic = time.monotonic()
            indices, order, cursor = _sample_indices(
                rng, state["scenario_order"], state["scenario_cursor"],
                args.scenario_batch)
            population = [tuple(g) for g in state["population"]]
            scores = _evaluate_population(pool, population, indices, bank)
            best_idx = int(np.argmin(scores))
            best = population[best_idx]
            row = {
                "generation": generation + 1,
                "mean_probe_delay_vehh": float(np.mean(scores)),
                "best_probe_delay_vehh": float(scores[best_idx]),
                "inc_probe_delay_vehh": float(scores[population.index(zero)]),
                "best_delta_vs_inc_vehh": float(
                    scores[population.index(zero)] - scores[best_idx]),
                "best_adjustments_s": json.dumps(best),
                "scenario_master_indices": json.dumps(
                    [master_indices[i] for i in indices]),
                "elapsed_s": round(time.monotonic() - tic, 2),
            }
            state["history"].append(row)
            state["archive"].append(list(best))
            state["population"] = [list(g) for g in _next_population(
                population, scores, rng, args.elite_count,
                args.mutation_rate, args.mutation_sigma, n_genes)]
            state["scenario_order"] = order
            state["scenario_cursor"] = cursor
            state["generation_next"] = generation + 1
            state["rng_state"] = rng.bit_generator.state
            _atomic_json(state_path, state)
            _write_csv(log_path, state["history"])
            print(f"gen {generation+1:3d}/{args.generations}: "
                  f"best={scores[best_idx]:.4f} incumbent="
                  f"{scores[population.index(zero)]:.4f} veh-h "
                  f"elapsed={row['elapsed_s']:.0f}s", flush=True)

        if not solution_path.exists():
            # Re-rank generation winners on one common sample drawn only from
            # the training bank. Never select a genome using held-out outcomes.
            finalists = _selection_finalists(
                state["archive"], state["history"], args.selection_candidates,
                n_genes)
            selection_rng = np.random.default_rng(args.seed + 101)
            selection_indices = selection_rng.choice(
                len(bank), size=args.selection_scenarios,
                replace=False).tolist()
            scores = _evaluate_population(pool, finalists, selection_indices, bank)
            best_idx = int(np.argmin(scores))
            best = finalists[best_idx]
            solution = {
                "signature": signature,
                "offset_adjustment_s": (
                    dict(zip(C.AGENT_ORDER, best)) if args.schedule == "static"
                    else {name: list(best[i::C.N_AGENTS])
                          for i, name in enumerate(C.AGENT_ORDER)}),
                "offset_adjustment_vector_s": list(best),
                "selection_mean_probe_delay_vehh": float(scores[best_idx]),
                "incumbent_selection_mean_probe_delay_vehh": float(scores[0]),
                "selection_master_indices": [
                    master_indices[index] for index in selection_indices],
                "number_of_distinct_finalists": len(finalists),
                "training_rollouts": args.population * args.generations * args.scenario_batch,
                "selection_rollouts": len(finalists) * args.selection_scenarios,
            }
            _atomic_json(solution_path, solution)
        else:
            solution = json.loads(solution_path.read_text(encoding="utf-8"))
            if solution["signature"] != signature:
                raise RuntimeError("GA solution has incompatible configuration")
            best = tuple(solution["offset_adjustment_vector_s"])
        if args.schedule == "static":
            print(f"selected fixed residual offsets (s): "
                  f"{dict(zip(C.AGENT_ORDER, best))}", flush=True)
        else:
            print(f"selected interval schedule: {n_intervals} x {C.N_AGENTS} "
                  f"offsets; saved to {solution_path}", flush=True)
        if args.no_eval:
            return
        if eval_path.exists() and summary_path.exists():
            print(f"evaluation already exists: {eval_path}", flush=True)
            return
        tasks = [(index, params, best, args.save_origin_group_metrics)
                 for index, params in enumerate(all_eval[:args.eval_count])]
        rows = sorted(pool.imap_unordered(_eval_task, tasks, chunksize=1),
                      key=lambda row: row["scenario"])
        _write_csv(eval_path, rows)
        summary = {
            "signature": signature,
            "solution": str(solution_path),
            "n_scenarios": len(rows),
            "means": {key: float(np.mean([row[key] for row in rows]))
                      for key in ("inc_probe_delay_vehh", "ga_probe_delay_vehh",
                                  "probe_delay_saving_pct", "inc_mph", "pol_mph",
                                  "inc_speed_kmh", "ga_speed_kmh", "dmph_pct",
                                  "inc_wait", "pol_wait", "waiting_saving_pct",
                                  "inc_loss", "pol_loss", "delay_saving_pct")},
        }
        _atomic_json(summary_path, summary)
        print(f"evaluation complete: {eval_path}", flush=True)


if __name__ == "__main__":
    main()

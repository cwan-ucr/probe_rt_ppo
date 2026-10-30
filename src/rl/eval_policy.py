"""
Evaluate a trained PPO policy on a shared held-out demand design.

For each scenario the selected incumbent backbone and the DETERMINISTIC
policy (action = distribution mean, no exploration) are rolled out on the SAME scaled
demand and simulation seed (common random numbers), so the two are paired. Reports
per-scenario and aggregate true trip metrics (VMT/VHT mph, mean waiting, mean time
loss) and the policy's improvement over the incumbent.

The action space (offset-only vs offset+split) and the collector (probe vs fast) are
read from the environment, so run this with the SAME RL_SPLIT / RL_COLLECTOR that the
checkpoint was trained with, e.g.:

  RL_COLLECTOR=probe python -m src.rl.eval_policy \
      --ckpt outputs/rl/coord_actuated_skip/ppo_parallel_ckpt_<tag>_coord_actuated_skip_500.pt \
      --arch gru --backbone coord_actuated_skip --n-scenarios 72 --workers 24

Outputs: outputs/rl/<backbone>/eval_<tag>.csv (+ console table + JSON summary).
"""
import os
import csv
import json
import time
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
import multiprocessing as mp

from . import config as C
from . import demand
from . import scenarios
from .corridor_env import CorridorEnv
from .traci_ports import resolve_port_base
if C.RL_METHOD == "diaz":
    from .diaz_style_env import DiazCorridorEnv as EvaluationEnv
elif C.CONTROL_LAYOUT == "all10":
    from .corridor_env_all10 import CorridorEnvAll10 as EvaluationEnv
else:
    EvaluationEnv = CorridorEnv
from .train_ppo_parallel import (
    ATTENTION_CRITIC_POOLING,
    ATTENTION_DROPOUT,
    ATTENTION_HEADS,
    ATTENTION_POSITION_ENCODING,
    RECURRENT_ARCHS,
    build_net,
)

OUT = os.path.join(C.ROOT, "outputs", "rl")
_G = {}   # per-worker cache (net + config), populated by _init


def _init(ckpt, arch, lstm_hid, backbone, episode_end, port_base, model_residual,
          save_action_trace, save_origin_group_metrics):
    """Pool initializer: build the net once per worker and load the checkpoint."""
    net = build_net(arch, lstm_hid)
    sd = torch.load(ckpt, map_location="cpu")
    net.load_state_dict(sd)                    # fails loudly if ACTION_DIM mismatches
    net.eval()
    _G.update(net=net, arch=arch, backbone=backbone, episode_end=episode_end,
              port_base=port_base, model_residual=model_residual,
              save_action_trace=save_action_trace,
              save_origin_group_metrics=save_origin_group_metrics)


def _origin_group_trip_metrics(path):
    """Aggregate SUMO trip metrics by the vehicle's corridor-entry class.

    Calibrated arterial-entry movements begin with ``uni_``. All other
    movements enter from a cross street. The classification is intentionally
    origin-based: a side-origin vehicle remains in that group after turning
    onto University Avenue.
    """
    totals = {
        "main_origin": {
            "n": 0, "route_length_total_m": 0.0,
            "duration_total_s": 0.0, "waiting_total_s": 0.0,
            "delay_total_s": 0.0,
        },
        "side_origin": {
            "n": 0, "route_length_total_m": 0.0,
            "duration_total_s": 0.0, "waiting_total_s": 0.0,
            "delay_total_s": 0.0,
        },
    }
    for _event, element in ET.iterparse(path, events=("end",)):
        if element.tag == "tripinfo":
            vehicle_id = element.get("id", "")
            group = "main_origin" if vehicle_id.startswith("uni_") else "side_origin"
            totals[group]["n"] += 1
            totals[group]["route_length_total_m"] += float(
                element.get("routeLength", 0.0) or 0.0)
            totals[group]["duration_total_s"] += float(
                element.get("duration", 0.0) or 0.0)
            totals[group]["waiting_total_s"] += float(
                element.get("waitingTime", 0.0) or 0.0)
            totals[group]["delay_total_s"] += float(
                element.get("timeLoss", 0.0) or 0.0)
        element.clear()
    for values in totals.values():
        values["speed_kmh"] = (
            3.6 * values["route_length_total_m"]
            / values["duration_total_s"]
            if values["duration_total_s"] else 0.0)
        values["waiting_s_per_veh"] = (
            values["waiting_total_s"] / values["n"]
            if values["n"] else 0.0)
        values["delay_s_per_veh"] = (
            values["delay_total_s"] / values["n"] if values["n"] else 0.0)
    return totals


def _add_origin_group_metrics(row, incumbent, policy):
    """Append paired main-/side-origin speed, waiting, and delay fields."""
    for group in ("main_origin", "side_origin"):
        inc = incumbent[group]
        pol = policy[group]
        row.update({
            f"inc_{group}_n": int(inc["n"]),
            f"pol_{group}_n": int(pol["n"]),
            f"inc_{group}_speed_kmh": float(inc["speed_kmh"]),
            f"pol_{group}_speed_kmh": float(pol["speed_kmh"]),
            f"{group}_speed_gain_pct": (
                100.0 * (pol["speed_kmh"] - inc["speed_kmh"])
                / inc["speed_kmh"] if inc["speed_kmh"] else 0.0),
            f"inc_{group}_waiting_total_s": float(inc["waiting_total_s"]),
            f"pol_{group}_waiting_total_s": float(pol["waiting_total_s"]),
            f"inc_{group}_waiting_s_per_veh": float(
                inc["waiting_s_per_veh"]),
            f"pol_{group}_waiting_s_per_veh": float(
                pol["waiting_s_per_veh"]),
            f"delta_{group}_waiting_s_per_veh": float(
                pol["waiting_s_per_veh"] - inc["waiting_s_per_veh"]),
            f"{group}_waiting_saving_pct": (
                100.0 * (inc["waiting_s_per_veh"]
                         - pol["waiting_s_per_veh"])
                / inc["waiting_s_per_veh"]
                if inc["waiting_s_per_veh"] else 0.0),
            f"inc_{group}_delay_total_s": float(inc["delay_total_s"]),
            f"pol_{group}_delay_total_s": float(pol["delay_total_s"]),
            f"inc_{group}_delay_s_per_veh": float(inc["delay_s_per_veh"]),
            f"pol_{group}_delay_s_per_veh": float(pol["delay_s_per_veh"]),
            # This matches Delta D = D_RL - D_baseline; negative is beneficial.
            f"delta_{group}_delay_s_per_veh": float(
                pol["delay_s_per_veh"] - inc["delay_s_per_veh"]),
            # Positive values are easier to interpret as operational savings.
            f"{group}_delay_saving_pct": (
                100.0 * (inc["delay_s_per_veh"] - pol["delay_s_per_veh"])
                / inc["delay_s_per_veh"]
                if inc["delay_s_per_veh"] else 0.0),
        })


def _wrap_cycle(x, cycle):
    """Signed circular difference in [-cycle/2, cycle/2)."""
    return (float(x) + 0.5 * cycle) % cycle - 0.5 * cycle


def _rollout(env, net, arch, collect_trace=False, scenario=None, params=None):
    """One deterministic episode. net=None -> incumbent (a=0). Returns the end-of-episode
    info dict (true VMT/VHT/waiting/time-loss) and optional per-decision action trace."""
    s = env.reset()
    done, info = False, {}
    trace = []
    h = None
    a_prev = np.zeros(C.ACTION_DIM, dtype=np.float32)
    while not done:
        if net is None:
            a_np = np.zeros(C.ACTION_DIM, dtype=np.float32)      # incumbent plan
        else:
            st = torch.as_tensor(s, dtype=torch.float32)
            with torch.no_grad():
                if arch in RECURRENT_ARCHS:
                    mu, _v, h = net.act_mean(st, torch.as_tensor(a_prev), h)
                else:
                    mu, _std, _v = net.forward(st)
            a_np = mu.numpy()
        s, step_reward, done, info = env.step(a_np)
        if collect_trace:
            a_mat = np.asarray(a_np, dtype=np.float32).reshape(
                C.N_AGENTS, C.N_ACT_PER_INT)
            offsets = info.get("offsets", {})
            start_s = float(info.get(
                "interval_start_s",
                info.get("interval_end_s", 0.0) - C.DECISION_INTERVAL_S))
            end_s = float(info.get(
                "interval_end_s", start_s + C.DECISION_INTERVAL_S))
            step = int(info.get("decision_step", len(trace) // C.N_AGENTS))
            for k, name in enumerate(C.AGENT_ORDER):
                deployed = float(offsets[name])
                base = float(env._ga_offset[name])
                trace.append({
                    "scenario": int(scenario) if scenario is not None else -1,
                    "g": float(params[0]) if params is not None else 1.0,
                    "main": float(params[1]) if params is not None else 1.0,
                    "side": float(params[2]) if params is not None else 1.0,
                    "direction": float(params[3]) if params is not None else 0.5,
                    "decision_step": step,
                    "interval_start_s": start_s,
                    "interval_end_s": end_s,
                    "clock_hour": 7.0 + start_s / 3600.0,
                    "time_set": 1 + int(start_s // C.DECISION_INTERVAL_S),
                    "intersection": name,
                    "action_norm": float(a_mat[k, 0]),
                    "base_offset_s": base,
                    "deployed_offset_s": deployed,
                    "residual_offset_s": _wrap_cycle(
                        deployed - base, C.CYCLE_S),
                    "step_reward": float(step_reward),
                })
        a_prev = a_np
    return info, trace


def _eval_one(task):
    """Run incumbent + policy on one scenario (paired). Returns a per-scenario dict."""
    idx, params = task
    g = _G
    env = EvaluationEnv(
        backbone=g["backbone"], seed=42, episode_end=g["episode_end"],
        port=g["port_base"] + idx)
    keep_group_metrics = bool(g["save_origin_group_metrics"])
    if keep_group_metrics:
        # The split requires individual tripinfo records. Parse the incumbent
        # file before the policy reset overwrites the same per-environment path.
        env.set_metrics_mode("tripinfo")
        env.keep_metrics_files = True
    inc_groups = pol_groups = None
    try:
        env.set_demand(params)                 # scale demand (same file for both rollouts)
        env.baseline = None
        # incumbent = the true coordinated-actuated plan (a=0 under RESIDUAL semantics),
        # independent of the policy's own action mode.
        C.RESIDUAL_ACTION = True
        inc, _ = _rollout(env, None, g["arch"])
        if keep_group_metrics:
            inc_groups = _origin_group_trip_metrics(env._tripinfo)
        C.RESIDUAL_ACTION = g["model_residual"]   # policy in its trained action mode
        pol, trace = _rollout(
            env, g["net"], g["arch"],
            collect_trace=g["save_action_trace"], scenario=idx, params=params)
        if keep_group_metrics:
            pol_groups = _origin_group_trip_metrics(env._tripinfo)
    finally:
        env.close()
        if keep_group_metrics:
            Path(env._tripinfo).unlink(missing_ok=True)

    def pick(d):
        return (d.get("vmt_per_vht", 0.0), d.get("wait_s_per_veh", 0.0),
                d.get("timeloss_s_per_veh", 0.0))
    imph, iw, il = pick(inc)
    pmph, pw, pl = pick(pol)
    pct = lambda a, b: (100.0 * (b - a) / a if a else 0.0)
    row = {"scenario": idx, "g": params[0], "main": params[1],
           "side": params[2], "direction": params[3],
           "inc_mph": imph, "pol_mph": pmph, "dmph_pct": pct(imph, pmph),
           "inc_wait": iw, "pol_wait": pw, "dwait_pct": pct(iw, pw),
           "inc_loss": il, "pol_loss": pl, "dloss_pct": pct(il, pl)}
    if keep_group_metrics:
        _add_origin_group_metrics(row, inc_groups, pol_groups)
    return row, trace


def _validate_checkpoint_metadata(args, model_residual):
    meta_path = args.ckpt + ".meta.json"
    if not os.path.exists(meta_path):
        print(
            f"WARNING: checkpoint metadata not found: {meta_path}; "
            "relying on tensor-shape validation only", flush=True)
        return
    with open(meta_path) as f:
        metadata = json.load(f)
    signature = metadata.get("signature", {})
    expected = {
        "control_layout": C.CONTROL_LAYOUT,
        "state_schema": C.STATE_SCHEMA,
        "direction_features": list(C.DIR_FEATURES),
        "delay_ratio_saturation": float(C.DELAY_RATIO_SATURATION),
        "state_dim": C.STATE_DIM,
        "action_dim": C.ACTION_DIM,
        "policy_distribution": (
            "tanh_normal_state_std_v1"
            if C.CONTROL_LAYOUT == "all10"
            else "legacy_normal"),
        "state_dependent_std": C.CONTROL_LAYOUT == "all10",
        "split_control": bool(C.SPLIT_CONTROL),
        "residual_action": bool(model_residual),
        "backbone": args.backbone,
        "arch": args.arch,
        "lstm_hid": int(args.lstm_hid),
    }
    if C.RL_METHOD == "diaz":
        expected["comparison_method"] = "diaz_near_stopbar_queue_step_presence_v1"
    if "spatiotemporal_layer_norm" in signature:
        from .train_ppo_parallel import USE_SPATIOTEMPORAL_LAYER_NORM
        expected["spatiotemporal_layer_norm"] = bool(
            USE_SPATIOTEMPORAL_LAYER_NORM)
    if args.arch in ("attention", "attention_gru") and \
            "attention_heads" in signature:
        expected.update({
            "attention_dim": int(args.lstm_hid),
            "attention_heads": int(ATTENTION_HEADS),
            "attention_dropout": float(ATTENTION_DROPOUT),
            "attention_position_encoding": ATTENTION_POSITION_ENCODING,
            "critic_pooling": ATTENTION_CRITIC_POOLING,
        })
    # New EXP5 checkpoints fingerprint the explicit vehicle realization and
    # penetration. Retain compatibility with older EXP2--EXP4 metadata that did
    # not contain these fields.
    if "explicit_demand" in signature:
        expected.update({
            "explicit_demand": bool(demand.explicit_demand_enabled()),
            "probe_penetration": (
                float(demand.probe_penetration())
                if demand.explicit_demand_enabled() else None),
            "explicit_demand_seed": (
                int(demand.explicit_demand_seed())
                if demand.explicit_demand_enabled() else None),
        })
        if "probe_membership_seed" in signature:
            expected["probe_membership_seed"] = (
                int(demand.probe_membership_seed())
                if demand.explicit_demand_enabled() else None)
    mismatches = {
        key: {"expected": value, "actual": signature.get(key)}
        for key, value in expected.items()
        if signature.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Checkpoint configuration mismatch for {args.ckpt}: {mismatches}")


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="trained policy state_dict (.pt)")
    ap.add_argument(
        "--arch",
        choices=["mlp", "bilstm", "attention", "gru", "bilstm_gru",
                 "attention_gru"],
        required=True)
    ap.add_argument("--lstm-hid", type=int, default=128)
    ap.add_argument("--backbone", choices=C.BACKBONE_CHOICES, default=None)
    ap.add_argument(
        "--base", choices=["fixed", "actuated"], default=None,
        help="deprecated compatibility alias")
    ap.add_argument("--episode-end", type=int, default=C.EPISODE_END_S)
    ap.add_argument("--n-scenarios", type=int, default=72)
    ap.add_argument("--scenario-seed", type=int, default=54321,
                    help="legacy fallback seed when --scenario-file is empty")
    ap.add_argument(
        "--scenario-file", default=str(scenarios.DEFAULT_EVAL_FILE),
        help="common versioned LHS evaluation design")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--port-base", type=int, default=0,
                    help="0 -> derive from $RL_PORT_BASE / SLURM_JOB_ID / pid")
    ap.add_argument(
        "--tag", default="",
        help="output suffix (default: arch, plus _all10 in the 10-node environment)")
    ap.add_argument(
        "--save-action-trace", action="store_true",
        help=("save one long-form row per scenario, decision, and intersection "
              "with normalized actions and physical offsets"))
    ap.add_argument(
        "--save-origin-group-metrics", action="store_true",
        help=("also report SUMO speed, waiting time, and timeLoss separately "
              "for mainline-origin and side-street-origin vehicles; requires "
              "per-vehicle tripinfo"))
    args = ap.parse_args()
    if args.backbone is not None and args.base is not None:
        if C.canonical_backbone(args.backbone) != C.canonical_backbone(args.base):
            ap.error("--backbone and deprecated --base select different controllers")
    args.backbone = C.canonical_backbone(args.backbone or args.base)
    args.base = args.backbone
    OUT = C.rl_output_dir(args.backbone)

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(args.ckpt)
    if args.n_scenarios < 1 or args.workers < 1:
        ap.error("--n-scenarios and --workers must both be positive")
    tag = args.tag or (
        f"{args.arch}_all10"
        if C.CONTROL_LAYOUT == "all10"
        else args.arch
    )
    os.makedirs(OUT, exist_ok=True)

    # Match training: keep SUMO listeners below Linux's ephemeral client ports.
    port_base = resolve_port_base(
        args.port_base, required_span=max(args.workers, args.n_scenarios))

    # Use the same checked-in LHS evaluation design for every experiment.  The RNG
    # fallback is kept only for compatibility with an explicitly empty path.
    if args.scenario_file:
        all_scenarios, scenario_metadata = scenarios.load_scenario_file(
            args.scenario_file)
        if args.n_scenarios > len(all_scenarios):
            ap.error(
                f"--n-scenarios={args.n_scenarios}, but {args.scenario_file} "
                f"contains only {len(all_scenarios)} points")
        demand_scenarios = all_scenarios[:args.n_scenarios]
        scenario_design = scenario_metadata["sampler"]
        scenario_seed = int(scenario_metadata["seed"])
        scenario_sha = scenarios.file_sha256(args.scenario_file)
    else:
        rng = np.random.default_rng(args.scenario_seed)
        demand_scenarios = [
            demand.sample_params(rng) for _ in range(args.n_scenarios)]
        scenario_design = "legacy_independent_uniform"
        scenario_seed = int(args.scenario_seed)
        scenario_sha = None
    model_residual = os.environ.get("RL_RESIDUAL", "1") == "1"
    _validate_checkpoint_metadata(args, model_residual)

    print(f"Eval | ckpt={os.path.basename(args.ckpt)} arch={args.arch} "
          f"split={'on' if C.SPLIT_CONTROL else 'off'} action_dim={C.ACTION_DIM} "
          f"residual={model_residual} "
          f"std={'state-dependent' if C.CONTROL_LAYOUT == 'all10' else 'global'} | "
          f"backbone={args.backbone} | {args.n_scenarios} scenarios "
          f"({scenario_design}) | workers={args.workers} port_base={port_base}",
          flush=True)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init,
                  initargs=(args.ckpt, args.arch, args.lstm_hid, args.backbone,
                            args.episode_end, port_base, model_residual,
                            args.save_action_trace,
                            args.save_origin_group_metrics)) as pool:
        results = pool.map(_eval_one, list(enumerate(demand_scenarios)))
    rows = [r for r, _trace in results]
    trace_rows = [x for _r, trace in results for x in trace]
    rows.sort(key=lambda r: r["scenario"])
    trace_rows.sort(key=lambda r: (
        r["scenario"], r["decision_step"],
        C.AGENT_ORDER.index(r["intersection"])))

    # per-scenario table
    print("\n==== per-scenario (policy vs incumbent) ====")
    hdr = (f"{'sc':>3} {'g':>5} {'main':>5} {'side':>5} {'dir':>5} | "
           f"{'inc_wait':>8} {'pol_wait':>8} "
           f"{'dwait%':>7} | {'inc_mph':>7} {'pol_mph':>7} {'dmph%':>6} | {'dloss%':>7}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['scenario']:>3} {r['g']:>5.2f} {r['main']:>5.2f} "
              f"{r['side']:>5.2f} {r['direction']:>5.2f} | "
              f"{r['inc_wait']:>8.1f} {r['pol_wait']:>8.1f} {r['dwait_pct']:>+7.1f} | "
              f"{r['inc_mph']:>7.2f} {r['pol_mph']:>7.2f} {r['dmph_pct']:>+6.1f} | "
              f"{r['dloss_pct']:>+7.1f}")

    # aggregate
    def agg(k):
        return float(np.mean([r[k] for r in rows]))
    summary = {
        "n_scenarios": len(rows), "arch": args.arch, "ckpt": args.ckpt,
        "backbone": args.backbone,
        "scenario_seed": scenario_seed, "residual": model_residual,
        "scenario_design": scenario_design,
        "scenario_file": (
            os.path.relpath(args.scenario_file, C.ROOT)
            if args.scenario_file else None),
        "scenario_sha256": scenario_sha,
        "state_dependent_std": C.CONTROL_LAYOUT == "all10",
        "inc_wait": agg("inc_wait"), "pol_wait": agg("pol_wait"), "dwait_pct": agg("dwait_pct"),
        "inc_mph": agg("inc_mph"), "pol_mph": agg("pol_mph"), "dmph_pct": agg("dmph_pct"),
        "inc_loss": agg("inc_loss"), "pol_loss": agg("pol_loss"), "dloss_pct": agg("dloss_pct"),
        "wait_win_rate": float(np.mean([r["dwait_pct"] < 0 for r in rows])),
    }
    if args.save_origin_group_metrics:
        for group in ("main_origin", "side_origin"):
            for key in (
                    f"inc_{group}_speed_kmh",
                    f"pol_{group}_speed_kmh",
                    f"{group}_speed_gain_pct",
                    f"inc_{group}_waiting_s_per_veh",
                    f"pol_{group}_waiting_s_per_veh",
                    f"delta_{group}_waiting_s_per_veh",
                    f"{group}_waiting_saving_pct",
                    f"inc_{group}_delay_s_per_veh",
                    f"pol_{group}_delay_s_per_veh",
                    f"delta_{group}_delay_s_per_veh",
                    f"{group}_delay_saving_pct"):
                summary[key] = agg(key)
    print("\n==== aggregate over %d scenarios ====" % len(rows))
    print(f"  waiting  : incumbent {summary['inc_wait']:.2f} -> policy "
          f"{summary['pol_wait']:.2f} s/veh  ({summary['dwait_pct']:+.1f}%)")
    print(f"  mph      : incumbent {summary['inc_mph']:.3f} -> policy "
          f"{summary['pol_mph']:.3f}  ({summary['dmph_pct']:+.1f}%)")
    print(f"  time loss: incumbent {summary['inc_loss']:.2f} -> policy "
          f"{summary['pol_loss']:.2f} s/veh  ({summary['dloss_pct']:+.1f}%)")
    print(f"  scenarios where policy lowers waiting: "
          f"{summary['wait_win_rate']*100:.0f}%  ({time.time()-t0:.0f}s)")
    if args.save_origin_group_metrics:
        print("\n==== metrics by vehicle origin (policy vs incumbent) ====")
        for group, label in (("main_origin", "mainline origin"),
                             ("side_origin", "side-street origin")):
            print(
                f"  {label:<18}: speed "
                f"{summary[f'inc_{group}_speed_kmh']:.2f} -> "
                f"{summary[f'pol_{group}_speed_kmh']:.2f} km/h "
                f"({summary[f'{group}_speed_gain_pct']:+.2f}%); waiting "
                f"{summary[f'inc_{group}_waiting_s_per_veh']:.2f} -> "
                f"{summary[f'pol_{group}_waiting_s_per_veh']:.2f} s/veh "
                f"(saving={summary[f'{group}_waiting_saving_pct']:+.2f}%); "
                f"delay {summary[f'inc_{group}_delay_s_per_veh']:.2f} -> "
                f"{summary[f'pol_{group}_delay_s_per_veh']:.2f} s/veh "
                f"(saving={summary[f'{group}_delay_saving_pct']:+.2f}%)")

    # save CSV + JSON
    csv_path = os.path.join(OUT, f"eval_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    json_path = os.path.join(OUT, f"eval_{tag}_summary.json")
    json.dump(summary, open(json_path, "w"), indent=2)
    saved = [csv_path, json_path]
    if args.save_action_trace:
        if not trace_rows:
            raise RuntimeError(
                "--save-action-trace was requested, but no policy decisions "
                "were recorded")
        trace_path = os.path.join(OUT, f"eval_{tag}_action_trace.csv")
        with open(trace_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
            w.writeheader()
            w.writerows(trace_rows)
        saved.append(trace_path)
    print("\nSaved " + "\nSaved ".join(saved), flush=True)


if __name__ == "__main__":
    main()

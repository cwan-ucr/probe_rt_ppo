# Aggregate-Probe-Based Adaptive Corridor Signal Control

Reproducibility guide for residual PPO offset control and two comparator methods
on the University Avenue corridor, Riverside, California.

> **Release status.** This is a release-ready README for an RL-focused copy of
> the project. The working `signal_control` repository also contains demand
> calibration, figure-generation, paper, and legacy experiment files. Keep only
> the files listed under [Required inputs and layout](#required-inputs-and-layout)
> in a clean public release. The archived result bundles described below are
> locally available; check their contents and data rights before uploading them.

## What this repository does

The main method learns **15-minute offset adjustments** for ten coordinated
signals while preserving each intersection's local signal program and the common
60-second cycle. Its observations and reward are computed from aggregate
probe-vehicle segment measurements; all-vehicle SUMO outputs are reserved for
performance evaluation. The study compares:

1. **R-ST-PPO:** residual, probe-based PPO with configurable state encoders;
2. **Diaz-style MLP:** detector-observation, absolute-offset, queue-reward PPO
   adapted to this corridor; and
3. **GA-310:** an open-loop genetic algorithm with a separate ten-offset vector
   for each of the 31 controlled 15-minute intervals.

This is a **simulation study**, not an operational deployment. It does not
provide raw commercial probe-vehicle trajectories.

## Experimental setting

| Item | Formal Ottawa-demand experiments |
|---|---|
| Corridor | University Avenue, Riverside, California |
| Controlled intersections, west to east | Lime, Mulberry, Park, Victoria, Eucalyptus, Kansas, Ottawa, Chicago, Cranford, Iowa |
| Signal cycle | 60 s |
| Simulation horizon | 8 h (28,800 s); the calibrated demand file has arrivals over the first 6 h |
| Control interval | 15 min (900 s) |
| Initial interval | Incumbent warm-up; 31 subsequent intervals are policy-controlled |
| Main training design | 2,000 four-dimensional LHS demand points, seed 42; a fixed nested 1,000-point subset is used for training |
| Training budget | 500 PPO updates × 24 simulated trajectories/update = 12,000 trajectories |
| Online monitoring | First 3 fixed held-out scenarios after each PPO update |
| Final evaluation | 72 held-out LHS scenarios, seed 54321 |
| Default physical vehicle/probe seeds | 42 / 42 |
| Reference probe penetration | 10% |

The four demand dimensions are `global`, `main`, `side` in `[0.8, 1.3]` and
`direction` in `[0.2, 0.8]`. Given calibrated eastbound (EB), westbound (WB),
and side-street flows, the multipliers are

```text
EB:   global × main × 2 × direction
WB:   global × main × 2 × (1 − direction)
Side: global × side
```

Thus `direction = 0.5` preserves the EB/WB balance in the calibrated input.
The training and evaluation LHS files contain **parameters**, not individual
vehicle trajectories. In explicit-demand runs, each parameter vector produces a
fixed vehicle realization; penetration changes only nested probe membership,
not the physical vehicle set.

### Signal backbones

| CLI value | Signal plan | Local control |
|---|---|---|
| `coord_fixed` | `fixed_timing_optimized_nema.add.xml` | Coordinated fixed-time |
| `coord_actuated` | `coordinated_actuated_nema.add.xml` | Coordinated actuated, without phase skipping |
| `coord_actuated_skip` | `coordinated_actuated_dual_loop_skipping_nema.add.xml` | Coordinated actuated with detector-call phase skipping |

The RL controller changes **offsets only**. It does not optimize cycle length,
phase order, or green splits in the formal experiments. Eucalyptus is represented
by two directional SUMO traffic-light objects but receives one policy action.

## Required inputs and layout

Run commands from the repository root. The following relative paths are used by
`src/rl/config.py` and the formal Slurm scripts; preserve them when building a
clean release:

```text
signal_control/
├── src/
│   ├── rl/                         # environments, PPO, Diaz, GA, evaluation
│   └── tsc_sumo_sim.py             # shared probe-segment parsing/collection
├── slurm/                          # HPCC launchers and scratch setup
├── data/
│   ├── corridor-osm-network/
│   │   ├── osm.sumocfg
│   │   ├── osm_cr_tl.net.xml
│   │   ├── detectors_v8_fixed.add.xml
│   │   ├── fixed_timing_optimized_nema.add.xml
│   │   ├── coordinated_actuated_nema.add.xml
│   │   └── coordinated_actuated_dual_loop_skipping_nema.add.xml
│   ├── geojson-1-15-min-interval/processed/
│   │                                  # segment/approach geometry inputs
│   └── rl_scenarios/
│       ├── train_lhs_n2000_seed42.json
│       └── eval_lhs_n72_seed54321.json
├── outputs/
│   ├── calibration/trips_turning_probe.xml
│   └── rl/<backbone>/              # generated caches, logs, checkpoints, evals
└── README.md
```

Some network files and processed segment inputs were derived from external
sources. **Verify redistribution rights before publishing them.** If an input
cannot be redistributed, document how authorized users can obtain it and keep
the same expected relative path. The trip file is a simulated calibrated demand
input, not observed individual trajectories.

## Software and machine setup

The HPCC scripts used the UCR Slurm `epyc` partition, Python 3.9 in a project
Conda environment, and SUMO 1.20.0. Core Python imports include `numpy`,
`torch`, `traci`, and `sumolib`; analysis/plotting scripts may additionally need
`pandas`, `scipy`, and `matplotlib`. Reproduce the published results with the
same software versions where possible. The repository does **not** currently
contain a lockfile, so a public release should add a tested environment export.

Set `SUMO_HOME` and make SUMO's Python tools importable before running a Python
entry point. On the original HPCC, each Slurm script performs `module load sumo`
and sets `PYTHONPATH` automatically. Before reusing those scripts elsewhere,
edit their hard-coded `PY=/rhome/cwan025/bigdata/.conda/envs/sumo/bin/python`,
Slurm partition/resource directives, and scratch root. Submit them **from the
repository root on the cluster**, not from a local Mac shell.

Validate the shared scenario files first:

```bash
python -m src.rl.scenarios --check \
  --train-file data/rl_scenarios/train_lhs_n2000_seed42.json --train-size 2000 \
  --eval-file data/rl_scenarios/eval_lhs_n72_seed54321.json --eval-size 72
```

Do not silently regenerate these files when comparing to released checkpoints:
their hashes and scenario order are part of the experiment protocol.

## Reproduce the formal PPO suite

The complete Ottawa-demand launch script first builds one 2,000-scenario
incumbent cache per backbone, then submits 15 PPO training jobs. Each job trains
on the same fixed 1,000-scenario subset and evaluates its final checkpoint on
the same 72 held-out scenarios. The preparation jobs must complete successfully
before their dependent training jobs can start.

```bash
mkdir -p logs
export RL_TMP_ROOT=/path/to/writable/large/scratch
bash slurm/submit_ottawa_formal_suite.sh
squeue -u "$USER"
```

`REBUILD_BANKS=1` is the launcher default. Use `REBUILD_BANKS=0` only if the
cached files were built with the **same** scenario file, calibrated demand,
signal-plan XML, probe estimator, and physical/probe seeds. In a new checkout,
rebuilding is safer.

| Experiment | New training jobs | Reference reused |
|---|---:|---|
| EXP2: MLP, GRU, BiLSTM, BiLSTM-GRU | 4 | — |
| EXP4: residual action × difference reward | 3 | EXP2 GRU is the residual + difference cell |
| EXP5: 5%, 20%, 30% penetration | 3 | EXP2 GRU is the 10% cell |
| Bank-size ablation: 200, 500, 2,000 | 3 | EXP2 GRU is the 1,000-cell reference |
| EXP6: fixed and phase-skipping backbones | 2 | EXP2 GRU is the actuated reference |

Formal PPO settings are 32 rollout workers, 24 trajectories per update, 500
updates, 10 PPO epochs, AdamW learning rate `3e-4`, weight decay `1e-4`, global
gradient norm clip `0.5`, discount `0.99`, GAE `0.95`, clipping `0.2`, and
entropy coefficient annealed from `0.002` to `0.0001` by update 300. State
encoders use hidden width 128 where applicable. The formal jobs enable spatial
and temporal LayerNorm where the selected encoder has those layers and cap the
state-dependent log standard deviation at `-1.5`. Source-of-truth details are
the corresponding Slurm script **and checkpoint `.pt.meta.json`**, not this
overview if a run intentionally differs.

### Main method: observation, action, reward

The ten-node probe observation has 322 values: for each node, four approach
directions (`EB`, `WB`, `NB`, `SB`), seven aggregate-probe features per direction
and four approach-presence flags, followed by two time-of-day values. The seven
features are stop-bar/mean/minimum speed ratios, a normalized low-speed queue
extent, log probe count, staleness, and a normalized probe-delay ratio. Missing
approaches are masked. Recurrent encoders additionally receive the previous
action; the current offset is not appended to the observation.

One action changes the offset at each of the ten nodes. In residual mode,
`a_i ∈ [-1,1]` maps to a wrapped adjustment of up to 30 s around the active
incumbent program's offset. The first 15-minute interval uses that incumbent
offset; the policy acts on the remaining 31 intervals. The default difference
reward at interval `t` is

```text
r_t = 5 × [D_base,t − D_policy,t − 0.05 × mean_i((wrapped offset change_i / 30 s)^2)]
```

`D` is **probe-estimated** segment delay in vehicle-hours. It integrates
`max(0, 1/v − 1/v_free)` over the 21 reported speed percentiles using the
composite trapezoidal rule, then weights by probe count and segment length.
The plain-delay EXP4 cells instead use the separately scaled `-0.03 D` traffic
term. Do not compare episode returns across reward definitions or probe
penetration levels as if they were the same physical quantity.

## Reproduce the comparator methods

### Diaz-style detector PPO

```bash
sbatch slurm/train_diaz_style.sh
```

This launcher uses the common coordinated-actuated plan, demand design, 500
updates, 24 trajectories/update, 32 workers, an MLP, and a 72-scenario final
evaluation. Its state is 160 values: ten nodes × four directions × two passive
E1 detector sites × (`flow`, `occupancy`). It selects ten **absolute** offsets
from the 60-second cycle and uses the negative length-normalized queue measure
as reward, without a probe-delay difference term or incumbent-offset penalty.
The detector loops are added for observation and do not actuate the existing
signal plan.

**Interpretation limit:** this is a *Diaz-style corridor adaptation*, not an
exact reimplementation of Diaz et al.'s original network. The current legacy
configuration puts 1.8-m advance and stop-bar loops approximately 8.3 m and
2.3 m upstream of the stop bar. Its `flow` feature accumulates per-step vehicle
presence, so a vehicle remaining on a loop can be counted repeatedly; it is
not SUMO's unique-vehicle 15-minute entry count. Preserve this disclosure when
using the archived comparison, or change the implementation and retrain under
a distinct version tag. The script's default tag is `diaz_mlp_v15_legacy_...`.

### Open-loop 310-gene GA

```bash
GA_JOB=$(sbatch --parsable slurm/train_ga_interval_offset.sh)
sbatch --dependency="afterok:${GA_JOB%%;*}" slurm/eval_ga_interval_history.sh
```

GA-310 has **31 intervals × 10 offsets = 310 integer adjustment genes**. It
starts from the incumbent offset schedule and evaluates 24 genomes over one
shared training scenario in each of 500 generations, yielding 12,000 training
rollouts. Fitness is episode-total probe-estimated delay. Candidate selection
is re-ranked on common **training-bank** scenarios; the 72 held-out scenarios
are reserved for final evaluation. GA is open loop: one selected interval
schedule is applied to every demand scenario and cannot react to observed
traffic. A separate ten-gene static-offset GA exists in the codebase but is
not the equal-dimensional comparator for the 31-decision PPO policy.

The second launcher evaluates every generation's training-best schedule on
the **first three fixed held-out scenarios** used by PPO online monitoring.
These 1,500 additional trajectories produce a comparable three-scenario
learning-curve visualization; they do not select the GA winner.

## Results and file conventions

New runs write under `outputs/rl/<backbone>/`. Typical files are:

| Pattern | Meaning |
|---|---|
| `baseline_cache_*.json` | Paired incumbent results for a particular backbone, scenario design, penetration, and input signature |
| `ppo_parallel_log_*.csv` | One row per PPO update, including rollout statistics and the three online evaluation scenarios |
| `ppo_parallel_ckpt_*_500.pt` | Final PyTorch policy checkpoint |
| `*.pt.meta.json` | Checkpoint architecture, configuration, and input hashes |
| `eval_*.csv` / `eval_*_summary.json` | Paired final evaluation and aggregate metadata |
| `ga_interval_training_*.csv` | One row per GA generation |
| `ga_interval_solution_*.json` | Selected GA schedule and selection metadata |
| `ga_interval_online_eval3_*` | Three-scenario post-hoc GA history evaluation |

Downloaded formal PPO results are currently archived at
`outputs/hpcc_rl_results_v11_ottawa/`; the downloaded 310-gene GA results are
at `outputs/hpcc_ga_interval310/`, and its three-scenario history evaluation
is at `outputs/ga310_eval3/`. These archive folders are **copies of outputs**,
not where a fresh training job writes. Do not mix `v10`, `v11`, and Diaz tags in
one comparison without checking their plans, demand files, scenario hashes,
and checkpoint metadata.

### Evaluate a released checkpoint without retraining

This example evaluates the archived EXP2 GRU policy on the formal 72-scenario
design. It writes a new `eval_*.csv` and `eval_*_summary.json` under
`outputs/rl/coord_actuated/`; it does not alter the archived checkpoint.

```bash
export RL_METHOD=probe RL_COLLECTOR=probe RL_EXPLICIT_DEMAND=1
export RL_EXPLICIT_DEMAND_SEED=42 RL_PROBE_MEMBERSHIP_SEED=42
export RL_PROBE_PENETRATION=0.1 RL_RESIDUAL=1 RL_SPLIT=0
export RL_SPATIOTEMPORAL_LAYER_NORM=1 RL_STATE_LOG_STD_MAX=-1.5
python -m src.rl.eval_policy_all10 \
  --ckpt outputs/hpcc_rl_results_v11_ottawa/coord_actuated/ppo_parallel_ckpt_exp2_v11_adamw1e4_clip05_ottawa_bank1000_coord_actuated_all10_8h_gru_coord_actuated_500.pt \
  --arch gru --lstm-hid 128 --backbone coord_actuated \
  --scenario-file data/rl_scenarios/eval_lhs_n72_seed54321.json \
  --n-scenarios 72 --episode-end 28800 --workers 8 \
  --save-origin-group-metrics --tag release_gru_eval72
```

Run this only after restoring all required network, segment, and calibrated
demand inputs. Use the checkpoint's adjacent `.pt.meta.json` to confirm the
expected input hashes before interpreting the new result.

Final evaluation rows pair policy and incumbent on the same demand scenario.
Key network columns include `inc_mph`/`pol_mph`, `inc_wait`/`pol_wait`, and
`inc_loss`/`pol_loss`. `loss` is SUMO all-vehicle `timeLoss` in seconds per
vehicle; it is **not** reconstructed from the probe reward. For reporting
savings, use

```text
speed improvement (%) = 100 × (pol_mph − inc_mph) / inc_mph
waiting saving (%)    = 100 × (inc_wait − pol_wait) / inc_wait
delay saving (%)      = 100 × (inc_loss − pol_loss) / inc_loss
```

Speed in the result CSVs is usually mph; multiply by `1.609344` for km/h.
Columns prefixed `inc_main_origin_`, `pol_main_origin_`, `inc_side_origin_`, and
`pol_side_origin_` separate vehicles by **network entry origin**. This does
not mean every subsequent segment of a “main-origin” vehicle lies on the
mainline. Compare physical evaluation metrics across methods; use training
returns only within the same reward definition and scale.

## Reproducibility safeguards and troubleshooting

- **Match the whole experiment signature.** A checkpoint requires matching
  layout (`all10`), encoder, residual/absolute action mode, backbone, state
  schema, signal plans, penetration, scenario design, and collector settings.
- **Do not reuse a stale baseline cache.** Rebuild after changing the calibrated
  trip file, signal XML, probe-delay estimator, LHS file, or explicit-demand
  seed. The cache metadata and checkpoint sidecar store hashes for auditing.
- **Keep demand realizations paired.** Baseline and learned policy must share
  vehicle IDs, departures, and routes; penetration experiments relabel only
  nested probe membership.
- **Use a writable scratch filesystem.** `slurm/setup_rl_scratch.sh` allocates
  one directory per job and sets `TMPDIR`/`RL_RUNTIME_DIR`. If SUMO reports
  `No space left on device`, check both quota and the actual job scratch path,
  not only `df -h $HOME`.
- **Inspect Slurm failures before resubmitting.** `DependencyNeverSatisfied`
  means an upstream job failed; inspect `sacct` and the corresponding
  `logs/*.err`. A new job does not repair the old dependency automatically.
- **Resume carefully.** PPO `--auto-resume` requires a matching training-state
  signature. Loading only a model checkpoint is not a bit-for-bit continuation
  of optimizer, sampler, and random-number state.
- **Treat `.pt` files as trusted-input-only.** PyTorch deserialization can run
  code. Load released checkpoints only from a trusted source and use their
  `.meta.json` files to verify the experiment before evaluation.

### Before publishing the clean copy

Keep the final `*.pt`, adjacent `*.pt.meta.json`, training CSV, final
evaluation CSV, and evaluation summary for every method being reported. GA's
selected controller is a JSON schedule rather than a PyTorch checkpoint. The
local `v11` results archive currently contains Diaz training/evaluation CSVs
but **no Diaz `.pt` checkpoint**; retrieve that checkpoint and its metadata
from the cluster before claiming that the released bundle reproduces Diaz
evaluation directly. Also include the GA-310 solution JSON and the three-
scenario history files if those curves appear in the paper.

## Publication and citation

The associated manuscript describes the full method and the interpretation of
the experiments. Add its final citation and a persistent release DOI when they
are available. The current working repository does not have a top-level
`LICENSE` file; choose and add an explicit code/data license before declaring
the clean copy open source. License the software and redistributable result
files separately from any third-party map or commercial probe-derived inputs.

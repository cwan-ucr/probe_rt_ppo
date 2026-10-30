# Reinforcement-Learning Traffic Signal Control Results

## Overview

This directory contains the processed training and evaluation results supporting a study of aggregate-probe-based adaptive traffic signal control for an urban arterial corridor. The experiments were conducted in SUMO using a calibrated model of the University Avenue corridor in Riverside, California. The modeled corridor contains ten controlled intersections:

1. Lime
2. Mulberry
3. Park
4. Victoria
5. Eucalyptus
6. Kansas
7. Ottawa
8. Chicago
9. Cranford
10. Iowa

The proposed controller uses residual proximal policy optimization to adjust the ten intersection offsets while preserving the local signal-control logic. Unless otherwise indicated by a filename or metadata file, the results use aggregate probe observations, a 15-minute decision interval, a 10% probe penetration rate, and the coordinated-actuated controller as the signal-control backbone.

This release contains simulation outputs and processed performance results. It does not contain personally identifiable information, individual real-world vehicle trajectories, or proprietary raw commercial probe data.

The `coord_actuated` directory contains the state-encoder comparison, residual-RL ablation, probe-penetration analysis, and training demand-bank-size analysis. The other two directories contain the additional controller backbones used in the signal-controller comparison.

## Experiments

| Label | Purpose | Main variants |
|---|---|---|
| `exp2` | State-encoder comparison | MLP, BiLSTM, GRU, and BiLSTM-GRU |
| `exp4` | Residual-action and reward ablation | residual/non-residual actions and difference/plain-delay rewards |
| `exp5` | Probe-penetration analysis | 5%, 10%, 20%, and 30% probe penetration; the 10% case reuses the Exp2 GRU result |
| `exp6` | Signal-controller backbone comparison | coordinated fixed-time, coordinated actuated, and coordinated actuated with phase skipping; the coordinated-actuated case reuses the Exp2 GRU result |
| `bank_ablation` | Training demand-bank-size analysis | nested subsets of a common master demand bank |

## File types

| Filename pattern | Description |
|---|---|
| `ppo_parallel_log_*.csv` | One row per PPO update. Contains training-rollout statistics and online evaluation results for three fixed held-out scenarios. |
| `eval_*.csv` | Final-checkpoint paired evaluation results. Each row represents one held-out demand scenario. |
| `eval_*_summary.json` | Aggregate statistics and evaluation metadata for the corresponding evaluation CSV. |
| `*.pt.meta.json` | Model architecture, training configuration, demand design, random seeds, and input-file hashes associated with a checkpoint. |

The final evaluation CSV files contain 72 held-out Latin-hypercube demand scenarios. Training logs contain 500 PPO updates, with 24 trajectories sampled per update and 32 parallel simulation workers in the principal experiments.

## Filename conventions

Important filename tokens include:

| Token | Meaning |
|---|---|
| `v11_adamw1e4_clip05_ottawa` | Result version using AdamW, weight decay of 0.0001, maximum gradient norm of 0.5, and the updated Ottawa demand |
| `bank1000` or `n1000` | Effective number of training demand scenarios |
| `all10` | All ten intersections are controlled |
| `8h` | Eight-hour simulation configuration |
| `mlp`, `bilstm`, `gru`, `bilstm_gru` | Policy state encoder |
| `pen005`, `pen020`, `pen030` | Probe penetration rates of 5%, 20%, and 30% |
| `noresidual` | Absolute/non-residual action formulation |
| `nodiff` | Plain delay reward without the paired difference-reward formulation |
| `explicit_pen01` | Explicit vehicle realization with 10% probe membership for final evaluation |
| `500` in a checkpoint name | Checkpoint saved after PPO update 500 |

## Training-log variables

All speed variables ending in `_mph` are stored in miles per hour. Variables ending in `_s_per_veh` are in seconds per vehicle.

| Variable | Description |
|---|---|
| `iter` | Zero-based PPO update index |
| `episodes` | Number of training trajectories used in the update |
| `ep_return_mean` | Mean training episode return |
| `probe_delay_vehh` | Aggregate probe-estimated delay used by the reward calculation, in vehicle-hours |
| `offset_deviation_penalty` | Mean penalty associated with deviation from the incumbent offsets |
| `mean_abs_offset_adjustment_s` | Mean absolute residual offset adjustment, in seconds |
| `time_loss_s_per_veh` | SUMO network mean time loss, in seconds per vehicle |
| `vmt_mi` | Vehicle-miles traveled |
| `vht_h` | Vehicle-hours traveled |
| `vmt_per_vht` | Network space-mean speed computed as VMT/VHT, in mph |
| `wait_s_per_veh` | Mean waiting time, in seconds per vehicle |
| `ga_mph` | Paired baseline speed, in mph |
| `n_transitions` | Number of state-action transitions collected for the PPO update |
| `secs` | Wall-clock time for the PPO update, in seconds |
| `ent_coef` | Entropy coefficient used during the update |
| `eval_mph` | Mean policy speed across the three online evaluation scenarios |
| `eval_ga_mph` | Mean paired-baseline speed across the three online evaluation scenarios |
| `eval_wait_s` | Mean policy waiting time across the online evaluation scenarios |
| `eval_time_loss_s_per_veh` | Mean policy time loss across the online evaluation scenarios |
| `eval0_*`, `eval1_*`, `eval2_*` | Scenario-specific return and physical performance for the three online evaluation scenarios |

The online evaluation values are monitoring results recorded during training. Statistical comparisons reported from final models should use the separate `eval_*.csv` files.

## Final-evaluation variables

### Demand variables

| Variable | Description |
|---|---|
| `scenario` | Held-out evaluation-scenario identifier |
| `g` | Overall demand multiplier |
| `main` | Mainline demand multiplier |
| `side` | Side-street demand multiplier |
| `direction` | Mainline directional-allocation coefficient, denoted by `d` |

For the released design, the mainline directional multipliers are `EB = 2d` and `WB = 2(1-d)`. These directional terms are combined with the overall and mainline multipliers. Demand ranges and scenario hashes are recorded in the checkpoint metadata and evaluation summary files.

### Network performance

| Variable | Description |
|---|---|
| `inc_mph`, `pol_mph` | Incumbent and learned-policy mean speed, in mph |
| `dmph_pct` | Percentage speed change, `(policy - incumbent) / incumbent × 100` |
| `inc_wait`, `pol_wait` | Incumbent and policy mean waiting time, in seconds per vehicle |
| `dwait_pct` | Percentage waiting-time change, `(policy - incumbent) / incumbent × 100` |
| `inc_loss`, `pol_loss` | Incumbent and policy mean time loss, in seconds per vehicle |
| `dloss_pct` | Percentage time-loss change, `(policy - incumbent) / incumbent × 100` |

For `dmph_pct`, positive values represent improvement. For `dwait_pct` and `dloss_pct`, negative values represent improvement. If reporting all outcomes as *savings*, reverse the sign of the waiting-time and time-loss percentage changes.

### Mainline- and side-street-origin performance

Columns beginning with `inc_main_origin_` and `pol_main_origin_` report incumbent and policy outcomes for vehicles entering from mainline origins. Columns beginning with `inc_side_origin_` and `pol_side_origin_` report the corresponding outcomes for vehicles entering from side-street origins.

For each origin group, the CSV provides:

- vehicle count (`*_n`);
- mean speed (`*_speed_kmh`);
- total and per-vehicle waiting time;
- total and per-vehicle time loss;
- absolute policy-minus-incumbent changes; and
- percentage improvements or savings.

These origin classifications describe where a vehicle entered the modeled network. They do not restrict the subsequent route to only the mainline or side streets.

## Experimental design

- Training demand parameters were generated using Latin hypercube sampling.
- The principal model uses a nested subset of 1,000 scenarios drawn from a 2,000-scenario master bank.
- Final evaluation uses 72 held-out Latin-hypercube scenarios generated with seed 54321.
- Baseline and learned-policy simulations are paired by demand scenario and vehicle realization where applicable.
- The main policy controls ten residual offset adjustments within ±30 seconds.
- The state schema is `probe7_expected_delay_ratio_v1`; the exact feature list and model dimensions are provided in checkpoint metadata.
- The primary GRU model uses AdamW with a learning rate of 0.0003, weight decay of 0.0001, and maximum gradient norm of 0.5.

The metadata JSON files should be treated as the definitive source whenever settings differ across experiments.

## Example: loading the results

```python
from pathlib import Path
import pandas as pd

root = Path("hpcc_rl_results_v11_ottawa/coord_actuated")

training = pd.read_csv(
    root
    / "ppo_parallel_log_exp2_v11_adamw1e4_clip05_ottawa_bank1000_"
      "coord_actuated_all10_8h_gru.csv"
)

evaluation = pd.read_csv(
    root
    / "eval_exp2_v11_adamw1e4_clip05_ottawa_bank1000_"
      "coord_actuated_all10_8h_gru_explicit_pen01.csv"
)

evaluation["speed_kmh"] = evaluation["pol_mph"] * 1.609344
evaluation["delay_saving_pct"] = -evaluation["dloss_pct"]
evaluation["waiting_saving_pct"] = -evaluation["dwait_pct"]
```

## Reproducibility and interpretation notes

1. Compare policies using paired rows from the same evaluation CSV or identically defined scenario files.
2. Do not compare episode returns across experiments with different reward definitions or reward scales without explicitly accounting for those differences.
3. Use physical measures such as speed, waiting time, and time loss for comparisons across reward formulations.
4. Training curves use three fixed online evaluation scenarios and are intended to show learning dynamics. Final performance claims should use the 72-scenario evaluation files.
5. Model checkpoints are PyTorch serialization files. Load them only in a trusted environment with the matching model code and metadata.
6. Results are generated from a calibrated simulation and should not be interpreted as field deployment measurements.

## Citation

When using these data, cite the associated manuscript and identify the dataset version as `v11_adamw1e4_clip05_ottawa`. The complete bibliographic citation and persistent dataset identifier should be added here when they become available.


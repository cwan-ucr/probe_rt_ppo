#!/bin/bash
# EXP4: three additional cells of the residual-action / difference-reward 2x2
# ablation. The full residual + paired-difference reference is EXP2 GRU.
# Every task uses the same nested n=1,000 subset of the Ottawa n=2,000 master
# cache and the same explicit 10% probe-vehicle realization.
#SBATCH --job-name=exp4_residual
#SBATCH --partition=epyc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/exp4_ottawa_%A_%a.out
#SBATCH --error=logs/exp4_ottawa_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -d src/rl ] || cd ..
[ -d src/rl ] || { echo "ERROR: submit from repository root"; exit 1; }
mkdir -p logs

source slurm/setup_rl_scratch.sh
module unload java 2>/dev/null || true
module load sumo
export PYTHONPATH="$PWD/src:$SUMO_HOME/tools:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RL_COLLECTOR=probe
export RL_SPLIT=0
export RL_SPATIOTEMPORAL_LAYER_NORM=1
export RL_STATE_LOG_STD_MAX=-1.5
export RL_OFFSET_DEVIATION_PENALTY=0.05
export RL_REWARD_SCALE=5.0
export RL_EXPLICIT_DEMAND=1
export RL_EXPLICIT_DEMAND_SEED=42
export RL_PROBE_MEMBERSHIP_SEED=42
export RL_PROBE_PENETRATION=0.1

PY=/rhome/cwan025/bigdata/.conda/envs/sumo/bin/python
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
BACKBONE="${BACKBONE:-coord_actuated}"
[ "$BACKBONE" = coord_actuated ] || {
    echo "ERROR: formal EXP4 reference backbone is coord_actuated"; exit 2;
}

# task 0: residual + plain delay reward
# task 1: absolute offset + paired difference reward
# task 2: absolute offset + plain delay reward
case "$TASK_ID" in
    0)
        export RL_RESIDUAL=1
        SUFFIX=nodiff
        EXTRA_ARGS=(--no-diff-reward --plain-reward-scale 0.03)
        ;;
    1)
        export RL_RESIDUAL=0
        SUFFIX=noresidual
        EXTRA_ARGS=()
        ;;
    2)
        export RL_RESIDUAL=0
        SUFFIX=noresidual_nodiff
        EXTRA_ARGS=(--no-diff-reward --plain-reward-scale 0.03)
        ;;
    *) echo "ERROR: unexpected array task $TASK_ID"; exit 3 ;;
esac

ARCH=gru
HIDDEN=128
WORKERS=32
TRAJ=24
ITERS=500
MASTER_BANK=2000
TRAIN_BANK=1000
SUBSET_SEED=42
EPISODE_END=28800
NEVAL=72
PROTOCOL_TAG="${PROTOCOL_TAG:-v11_adamw1e4_clip05_ottawa}"
TRAIN_SCENARIOS=data/rl_scenarios/train_lhs_n2000_seed42.json
EVAL_SCENARIOS=data/rl_scenarios/eval_lhs_n72_seed54321.json
OUTDIR="outputs/rl/$BACKBONE"
MASTER_CACHE="$OUTDIR/baseline_cache_v10_ottawa_all10_${BACKBONE}_lhs4d0813_n2000_p010_explicit.json"
TAG="exp4_${PROTOCOL_TAG}_bank1000_${BACKBONE}_all10_8h_gru_${SUFFIX}"
mkdir -p "$OUTDIR"

[ -s "$MASTER_CACHE" ] || {
    echo "ERROR: missing Ottawa master cache: $MASTER_CACHE"; exit 4;
}

$PY -u -m src.rl.train_ppo_all10 \
    --arch "$ARCH" --lstm-hid "$HIDDEN" --run-tag "$TAG" \
    --backbone "$BACKBONE" --random-demand \
    --demand-bank-size "$MASTER_BANK" \
    --demand-bank-limit "$TRAIN_BANK" \
    --demand-subset-seed "$SUBSET_SEED" \
    --train-scenario-file "$TRAIN_SCENARIOS" \
    --eval-scenario-file "$EVAL_SCENARIOS" \
    --bank-path "$MASTER_CACHE" --demand-sampling without_replacement \
    --episode-end "$EPISODE_END" --auto-resume \
    --workers "$WORKERS" --traj-per-update "$TRAJ" --iters "$ITERS" \
    --max-grad-norm 0.5 --weight-decay 1e-4 --log-std-init -2.5 \
    --ent-coef0 0.002 --ent-coef-final 0.0001 --ent-decay-iters 300 \
    --epochs 10 --batch-size 64 --eval-every 1 --eval-scenarios 3 \
    --ckpt-every 100 "${EXTRA_ARGS[@]}"

$PY -u -m src.rl.eval_policy_all10 \
    --ckpt "$OUTDIR/ppo_parallel_ckpt_${TAG}_${BACKBONE}_${ITERS}.pt" \
    --arch "$ARCH" --lstm-hid "$HIDDEN" --backbone "$BACKBONE" \
    --n-scenarios "$NEVAL" --scenario-file "$EVAL_SCENARIOS" \
    --episode-end "$EPISODE_END" --workers 24 \
    --save-origin-group-metrics \
    --tag "${TAG}_explicit_pen01"

echo "completed EXP4 task=$TASK_ID residual=$RL_RESIDUAL variant=$SUFFIX"

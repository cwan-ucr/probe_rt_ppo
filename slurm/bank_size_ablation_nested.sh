#!/bin/bash
# Bank-size ablation using nested subsets of one 2,000-scenario master cache.
# BANK_TIER=standard trains n={200,500,2000}; BANK_TIER=small trains
# n={24,50,100}. The n=1,000 GRU reference comes from matching EXP2.
# All jobs reuse the same incumbent rollouts and differ only in the number
# of master scenarios PPO may sample.
#SBATCH --job-name=bank_nested
#SBATCH --partition=epyc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/bank_nested_%A_%a.out
#SBATCH --error=logs/bank_nested_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -d src/rl ] || cd ..
[ -d src/rl ] || { echo "ERROR: submit from the repository root"; exit 1; }
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
export RL_RESIDUAL=1
export RL_OFFSET_DEVIATION_PENALTY=0.05
export RL_REWARD_SCALE=5.0
export RL_SPATIOTEMPORAL_LAYER_NORM=1
export RL_STATE_LOG_STD_MAX=-1.5
export RL_EXPLICIT_DEMAND=1
export RL_EXPLICIT_DEMAND_SEED=42
export RL_PROBE_MEMBERSHIP_SEED=42
export RL_PROBE_PENETRATION=0.1

PY=/rhome/cwan025/bigdata/.conda/envs/sumo/bin/python
case "${BANK_TIER:-standard}" in
    standard) SIZES=(200 500 2000) ;;
    small) SIZES=(24 50 100) ;;
    *) echo "ERROR: BANK_TIER must be standard or small"; exit 2 ;;
esac
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( TASK_ID < 0 || TASK_ID >= ${#SIZES[@]} )); then
    echo "ERROR: array task $TASK_ID is outside 0-$((${#SIZES[@]} - 1))"
    exit 2
fi
EFFECTIVE_BANK="${SIZES[$TASK_ID]}"

ARCH=gru
BACKBONE="${BACKBONE:-coord_actuated}"
case "$BACKBONE" in
    coord_actuated|coord_actuated_skip) ;;
    *) echo "ERROR: BACKBONE must be coord_actuated or coord_actuated_skip"; exit 2 ;;
esac

HIDDEN=128
WORKERS=32
TRAJ=24
ITERS=500
MASTER_BANK=2000
SUBSET_SEED=42
EPISODE_END=28800
NEVAL=72
PROTOCOL_TAG="${PROTOCOL_TAG:-v11_adamw1e4_clip05_ottawa}"
TRAIN_SCENARIOS=data/rl_scenarios/train_lhs_n2000_seed42.json
EVAL_SCENARIOS=data/rl_scenarios/eval_lhs_n72_seed54321.json
OUTDIR="outputs/rl/$BACKBONE"
MASTER_CACHE="$OUTDIR/baseline_cache_v10_ottawa_all10_${BACKBONE}_lhs4d0813_n2000_p010_explicit.json"
TAG="bank_ablation_${PROTOCOL_TAG}_n${EFFECTIVE_BANK}_${BACKBONE}_all10_8h_gru"
mkdir -p "$OUTDIR"

[ -s "$MASTER_CACHE" ] || {
    echo "ERROR: missing 2,000-scenario master cache: $MASTER_CACHE"
    echo "Run slurm/prepare_exp2_bank_2000.sh first."
    exit 3
}

$PY -u -m src.rl.train_ppo_all10 \
    --arch "$ARCH" --lstm-hid "$HIDDEN" --run-tag "$TAG" \
    --backbone "$BACKBONE" --random-demand \
    --demand-bank-size "$MASTER_BANK" \
    --demand-bank-limit "$EFFECTIVE_BANK" \
    --demand-subset-seed "$SUBSET_SEED" \
    --train-scenario-file "$TRAIN_SCENARIOS" \
    --eval-scenario-file "$EVAL_SCENARIOS" \
    --bank-path "$MASTER_CACHE" --demand-sampling without_replacement \
    --episode-end "$EPISODE_END" --auto-resume \
    --workers "$WORKERS" --traj-per-update "$TRAJ" --iters "$ITERS" \
    --max-grad-norm 0.5 --weight-decay 1e-4 --log-std-init -2.5 \
    --ent-coef0 0.002 --ent-coef-final 0.0001 --ent-decay-iters 300 \
    --epochs 10 --batch-size 64 --eval-every 1 --eval-scenarios 3 \
    --ckpt-every 100

$PY -u -m src.rl.eval_policy_all10 \
    --ckpt "$OUTDIR/ppo_parallel_ckpt_${TAG}_${BACKBONE}_${ITERS}.pt" \
    --arch "$ARCH" --lstm-hid "$HIDDEN" --backbone "$BACKBONE" \
    --n-scenarios "$NEVAL" --scenario-file "$EVAL_SCENARIOS" \
    --episode-end "$EPISODE_END" --workers 24 \
    --save-origin-group-metrics \
    --tag "${TAG}_explicit_pen01"

echo "completed nested bank-size run: $EFFECTIVE_BANK/$MASTER_BANK"

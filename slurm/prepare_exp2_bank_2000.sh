#!/bin/bash
# Build the shared 2,000-scenario paired incumbent cache for EXP2.
# Select BACKBONE through --export; default is coordinated actuated.
#SBATCH --job-name=prep_exp2_n2000
#SBATCH --partition=epyc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/prep_exp2_n2000_%j.out
#SBATCH --error=logs/prep_exp2_n2000_%j.err

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
export RL_REWARD_SCALE=5.0
export RL_EXPLICIT_DEMAND=1
export RL_EXPLICIT_DEMAND_SEED=42
export RL_PROBE_MEMBERSHIP_SEED=42
export RL_PROBE_PENETRATION=0.1

PY=/rhome/cwan025/bigdata/.conda/envs/sumo/bin/python
BACKBONE="${BACKBONE:-coord_actuated}"
case "$BACKBONE" in
    coord_fixed|coord_actuated|coord_actuated_skip) ;;
    *) echo "ERROR: unsupported coordinated BACKBONE=$BACKBONE"; exit 2 ;;
esac

TRAIN_SCENARIOS=data/rl_scenarios/train_lhs_n2000_seed42.json
EVAL_SCENARIOS=data/rl_scenarios/eval_lhs_n72_seed54321.json
BANK=2000
WORKERS=32
EPISODE_END=28800
OUTDIR="outputs/rl/$BACKBONE"
BANKPATH="$OUTDIR/baseline_cache_v10_ottawa_all10_${BACKBONE}_lhs4d0813_n2000_p010_explicit.json"
mkdir -p "$OUTDIR"

REBUILD_ARGS=()
if [ "${REBUILD_BANKS:-0}" = "1" ]; then
    REBUILD_ARGS+=(--rebuild-bank)
    echo "REBUILD_BANKS=1: rebuilding the 2,000-scenario master cache"
fi

$PY -m src.rl.scenarios --check \
    --train-file "$TRAIN_SCENARIOS" --train-size "$BANK" \
    --eval-file "$EVAL_SCENARIOS" --eval-size 72

$PY -u -m src.rl.train_ppo_all10 \
    --backbone "$BACKBONE" --arch gru --random-demand \
    --demand-bank-size "$BANK" --bank-path "$BANKPATH" \
    --train-scenario-file "$TRAIN_SCENARIOS" \
    --eval-scenario-file "$EVAL_SCENARIOS" \
    --episode-end "$EPISODE_END" --workers "$WORKERS" \
    --traj-per-update 1 --iters 1 --eval-every 0 --build-bank-only \
    "${REBUILD_ARGS[@]}"

echo "EXP2 cache ready: $BANKPATH"

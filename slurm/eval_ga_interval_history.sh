#!/bin/bash
# Post-hoc evaluation of all 500 GA-310 generation winners on the same first
# three held-out LHS scenarios used by PPO's online evaluation. No GA retraining.
#SBATCH --job-name=ga310_eval3
#SBATCH --partition=epyc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ga310_eval3_%j.out
#SBATCH --error=logs/ga310_eval3_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
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
export RL_METHOD=probe
export RL_SPLIT=0
export RL_RESIDUAL=1
export RL_OFFSET_DEVIATION_PENALTY=0.05
export RL_REWARD_SCALE=5.0
export RL_EXPLICIT_DEMAND=1
export RL_EXPLICIT_DEMAND_SEED=42
export RL_PROBE_MEMBERSHIP_SEED=42
export RL_PROBE_PENETRATION=0.1

PY=/rhome/cwan025/bigdata/.conda/envs/sumo/bin/python
BACKBONE="${BACKBONE:-coord_actuated}"
TAG=ga_interval310_probe_delay_v1_ottawa_bank1000_all10_8h
OUTDIR="outputs/rl/$BACKBONE"

for path in \
    data/rl_scenarios/eval_lhs_n72_seed54321.json \
    "$OUTDIR/ga_interval_training_${TAG}_${BACKBONE}.csv" \
    "$OUTDIR/ga_interval_solution_${TAG}_${BACKBONE}.json" \
    "$OUTDIR/eval_ga_interval_${TAG}_${BACKBONE}.csv"; do
    [ -s "$path" ] || { echo "ERROR: missing $path"; exit 2; }
done

$PY -u -m src.rl.eval_ga_interval_history \
    --backbone "$BACKBONE" --run-tag "$TAG" \
    --eval-scenarios data/rl_scenarios/eval_lhs_n72_seed54321.json \
    --n-eval 3 --workers 32 --batch-generations 16

#!/bin/bash
# =============================================================================
# Evaluate the FINAL BiLSTM-GRU checkpoint on a fixed 72-scenario LHS design.
#
# Default (coordinated-actuated):
#   sbatch slurm/eval_final_checkpoint_72.sh
#
# Coordinated-actuated with phase skipping:
#   sbatch --export=ALL,BACKBONE=coord_actuated_skip \
#       slurm/eval_final_checkpoint_72.sh
#
# Override the checkpoint explicitly when needed:
#   sbatch --export=ALL,BACKBONE=coord_actuated,CKPT=/absolute/path/model.pt \
#       slurm/eval_final_checkpoint_72.sh
# =============================================================================
#SBATCH --job-name=eval72_final
#SBATCH --partition=epyc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval72_%j.out
#SBATCH --error=logs/eval72_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -d src/rl ] || cd ..
[ -d src/rl ] || { echo "ERROR: submit from the repository root"; exit 1; }
mkdir -p logs

module unload java 2>/dev/null || true
module load sumo

export PYTHONPATH="$PWD/src:$SUMO_HOME/tools:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# These settings must match the final EXP2 checkpoint configuration.
export RL_COLLECTOR=probe
export RL_SPLIT=0
export RL_RESIDUAL=1
export RL_OFFSET_DEVIATION_PENALTY=0.05
export RL_REWARD_SCALE=5.0
export RL_SPATIOTEMPORAL_LAYER_NORM=1
export RL_STATE_LOG_STD_MAX=-1.5

PY=/rhome/cwan025/bigdata/.conda/envs/sumo/bin/python
ARCH="${ARCH:-bilstm_gru}"
BACKBONE="${BACKBONE:-coord_actuated}"
FINAL_UPDATE="${FINAL_UPDATE:-500}"
N_SCENARIOS=72
WORKERS="${WORKERS:-24}"
EPISODE_END=28800
SCENARIO_SEED=54321
SCENARIO_FILE="data/rl_scenarios/eval_lhs_n72_seed${SCENARIO_SEED}.json"

case "$BACKBONE" in
    coord_actuated|coord_actuated_skip)
        EXPERIMENT="exp2"
        ;;
    *)
        echo "ERROR: BACKBONE must be coord_actuated or coord_actuated_skip"
        exit 2
        ;;
esac

TRAIN_TAG="${EXPERIMENT}_v8_demand0813_${BACKBONE}_all10_8h_${ARCH}"
DEFAULT_CKPT="outputs/rl/${BACKBONE}/ppo_parallel_ckpt_${TRAIN_TAG}_${BACKBONE}_${FINAL_UPDATE}.pt"
CKPT="${CKPT:-$DEFAULT_CKPT}"
EVAL_TAG="${TRAIN_TAG}_final${FINAL_UPDATE}_eval72_seed${SCENARIO_SEED}"

ulimit -n 4096 2>/dev/null || true

$PY -c "import torch, traci, sumolib; print('dependencies OK')"
$PY -c 'import sys; from src.rl.scenarios import load_scenario_file, file_sha256; p,m=load_scenario_file(sys.argv[1], int(sys.argv[2])); print("scenarios OK: n={} seed={} sha256={}".format(len(p), m["seed"], file_sha256(sys.argv[1])[:12]))' "$SCENARIO_FILE" "$N_SCENARIOS"
[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 3; }

echo "Evaluating final checkpoint only"
echo "  backbone : $BACKBONE"
echo "  checkpoint: $CKPT"
echo "  scenarios : $SCENARIO_FILE"
echo "  workers   : $WORKERS"
echo "  output tag: $EVAL_TAG"

$PY -u -m src.rl.eval_policy_all10 \
    --ckpt "$CKPT" \
    --arch "$ARCH" \
    --backbone "$BACKBONE" \
    --n-scenarios "$N_SCENARIOS" \
    --scenario-file "$SCENARIO_FILE" \
    --episode-end "$EPISODE_END" \
    --workers "$WORKERS" \
    --tag "$EVAL_TAG"

OUTDIR="outputs/rl/$BACKBONE"
echo "Done. Results:"
echo "  $OUTDIR/eval_${EVAL_TAG}.csv"
echo "  $OUTDIR/eval_${EVAL_TAG}_summary.json"

#!/bin/bash
# Diaz-style MLP comparator on the coordinated-actuated backbone.
# Same demand protocol: 2,000 LHS master scenarios, nested 1,000 training
# scenarios, three online held-out evaluations, then final paired evaluation on 72.
#SBATCH --job-name=diaz_mlp
#SBATCH --partition=epyc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/diaz_mlp_%j.out
#SBATCH --error=logs/diaz_mlp_%j.err

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
export RL_EXPLICIT_DEMAND=1
export RL_EXPLICIT_DEMAND_SEED=42
export RL_PROBE_MEMBERSHIP_SEED=42
export RL_PROBE_PENETRATION=0.1

PY=/rhome/cwan025/bigdata/.conda/envs/sumo/bin/python
BACKBONE=coord_actuated
OUTDIR="outputs/rl/$BACKBONE"
TRAIN_SCENARIOS=data/rl_scenarios/train_lhs_n2000_seed42.json
EVAL_SCENARIOS=data/rl_scenarios/eval_lhs_n72_seed54321.json
# Restore the original near-stop-bar dual-loop, per-step-presence observation.
# Use a distinct tag so prior v11/v13/v14 results and checkpoints remain intact.
# The incumbent bank is reusable: its physical rollout does not depend on the
# Diaz observation definition.
TAG=diaz_mlp_v15_legacy_ottawa_bank1000_all10_8h
BANKPATH="$OUTDIR/baseline_cache_diaz_v1_all10_${BACKBONE}_n2000_p010_explicit.json"
CKPT="$OUTDIR/ppo_parallel_ckpt_${TAG}_${BACKBONE}_500.pt"
mkdir -p "$OUTDIR"

for path in "$TRAIN_SCENARIOS" "$EVAL_SCENARIOS" \
    outputs/calibration/trips_turning_probe.xml \
    data/corridor-osm-network/coordinated_actuated_nema.add.xml; do
    [ -s "$path" ] || { echo "ERROR: missing $path"; exit 2; }
done
[ -d data/geojson-1-15-min-interval/processed ] || {
    echo "ERROR: missing probe segment geometry"; exit 2;
}

$PY -m src.rl.scenarios --check \
    --train-file "$TRAIN_SCENARIOS" --train-size 2000 \
    --eval-file "$EVAL_SCENARIOS" --eval-size 72
$PY -c 'import torch, traci, sumolib; print("Python/SUMO dependencies OK")'
# The unique-ID-versus-E1-entry audit is not applicable to the legacy
# per-step-presence feature. Check layout and feature aggregation separately.
$PY -u -m src.rl.verify_diaz_detector_flow --intervals 2

$PY -u -m src.rl.train_diaz_style \
    --backbone "$BACKBONE" --random-demand \
    --demand-bank-size 2000 --demand-bank-limit 1000 \
    --demand-subset-seed 42 --demand-sampling without_replacement \
    --train-scenario-file "$TRAIN_SCENARIOS" \
    --eval-scenario-file "$EVAL_SCENARIOS" \
    --bank-path "$BANKPATH" --run-tag "$TAG" \
    --workers 32 --traj-per-update 24 --iters 500 \
    --episode-end 28800 --auto-resume \
    --max-grad-norm 0.5 --weight-decay 1e-4 \
    --epochs 10 --batch-size 64 \
    --ent-coef0 0.002 --ent-coef-final 0.0001 --ent-decay-iters 300 \
    --eval-every 1 --eval-scenarios 3 --ckpt-every 100

[ -s "$CKPT" ] || { echo "ERROR: final checkpoint not found: $CKPT"; exit 3; }
$PY -u -m src.rl.eval_diaz_style \
    --ckpt "$CKPT" --backbone "$BACKBONE" \
    --n-scenarios 72 --scenario-file "$EVAL_SCENARIOS" \
    --episode-end 28800 --workers 24 \
    --save-origin-group-metrics \
    --tag "${TAG}_explicit_pen01"

echo "Diaz-style training and final evaluation completed: $CKPT"

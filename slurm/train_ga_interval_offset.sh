#!/bin/bash
# Open-loop 15-minute offset-schedule GA: 31 controlled intervals x 10 nodes
# = 310 integer genes. The first 15 minutes remain incumbent warmup, as in PPO.
# 24 genomes x 500 generations x 1 shared scenario = 12,000 train rollouts.
# Uses the same demand bank, probe-delay estimator, and held-out design as the
# static-offset GA; final candidate selection uses training scenarios only.
#SBATCH --job-name=ga_310
#SBATCH --partition=epyc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ga_310_%j.out
#SBATCH --error=logs/ga_310_%j.err

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
TAG="ga_interval310_probe_delay_v1_ottawa_bank1000_all10_8h"

for path in outputs/calibration/trips_turning_probe.xml \
    data/corridor-osm-network/coordinated_actuated_nema.add.xml \
    data/rl_scenarios/train_lhs_n2000_seed42.json \
    data/rl_scenarios/eval_lhs_n72_seed54321.json; do
    [ -s "$path" ] || { echo "ERROR: missing $path"; exit 2; }
done
[ -d data/geojson-1-15-min-interval/processed ] || {
    echo "ERROR: missing probe segment geometry"; exit 2;
}

$PY -m src.rl.scenarios --check \
    --train-file data/rl_scenarios/train_lhs_n2000_seed42.json --train-size 2000 \
    --eval-file data/rl_scenarios/eval_lhs_n72_seed54321.json --eval-size 72

$PY -u -m src.rl.train_ga_static_offset \
    --schedule interval --backbone "$BACKBONE" --run-tag "$TAG" \
    --train-scenarios data/rl_scenarios/train_lhs_n2000_seed42.json \
    --eval-scenarios data/rl_scenarios/eval_lhs_n72_seed54321.json \
    --master-bank-size 2000 --bank-limit 1000 --subset-seed 42 \
    --population 24 --generations 500 --scenario-batch 1 \
    --selection-scenarios 72 --selection-candidates 24 --eval-count 72 \
    --workers 32 --episode-end 28800 --seed 42 \
    --mutation-rate 0.01 --mutation-sigma 5.0 \
    --save-origin-group-metrics --auto-resume

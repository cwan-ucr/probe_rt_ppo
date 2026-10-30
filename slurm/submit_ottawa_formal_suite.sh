#!/bin/bash
# Submit the complete Ottawa-demand formal experiment suite.
#
# Phase 1: rebuild n=2,000 explicit-p=10% incumbent caches for all backbones.
# Phase 2: launch 15 independent PPO training jobs through five Slurm arrays:
#   EXP2 encoders 4; residual ablation 3; penetration 3;
#   bank-size ablation 3; controller backbones 2.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

REBUILD_BANKS="${REBUILD_BANKS:-1}"
TMPROOT="${TMPROOT:-/rhome/cwan025/bigdata/tmp}"
PROTOCOL_TAG="${PROTOCOL_TAG:-v11_adamw1e4_clip05_ottawa}"
mkdir -p "$TMPROOT"
chmod 700 "$TMPROOT"

submit_id () {
    local output
    output=$(sbatch --parsable "$@")
    echo "${output%%;*}"
}

PREP_ACT=$(submit_id \
    --export="ALL,BACKBONE=coord_actuated,REBUILD_BANKS=$REBUILD_BANKS,RL_TMP_ROOT=$TMPROOT,TMPDIR=$TMPROOT" \
    slurm/prepare_exp2_bank_2000.sh)
PREP_FIXED=$(submit_id \
    --export="ALL,BACKBONE=coord_fixed,REBUILD_BANKS=$REBUILD_BANKS,RL_TMP_ROOT=$TMPROOT,TMPDIR=$TMPROOT" \
    slurm/prepare_exp2_bank_2000.sh)
PREP_SKIP=$(submit_id \
    --export="ALL,BACKBONE=coord_actuated_skip,REBUILD_BANKS=$REBUILD_BANKS,RL_TMP_ROOT=$TMPROOT,TMPDIR=$TMPROOT" \
    slurm/prepare_exp2_bank_2000.sh)

EXP2=$(submit_id \
    --dependency="afterok:$PREP_ACT" \
    --export="ALL,BACKBONE=coord_actuated,PROTOCOL_TAG=$PROTOCOL_TAG,RL_TMP_ROOT=$TMPROOT,TMPDIR=$TMPROOT" \
    slurm/exp2_state_repr_2000_parallel.sh)

EXP4=$(submit_id \
    --dependency="afterok:$PREP_ACT" \
    --export="ALL,BACKBONE=coord_actuated,PROTOCOL_TAG=$PROTOCOL_TAG,RL_TMP_ROOT=$TMPROOT,TMPDIR=$TMPROOT" \
    slurm/exp4_residual_ablation.sh)

EXP5=$(submit_id \
    --dependency="afterok:$PREP_ACT" \
    --export="ALL,BACKBONE=coord_actuated,PROTOCOL_TAG=$PROTOCOL_TAG,REBUILD_BANKS=$REBUILD_BANKS,RL_TMP_ROOT=$TMPROOT,TMPDIR=$TMPROOT" \
    slurm/exp5_penetration.sh)

BANK=$(submit_id \
    --dependency="afterok:$PREP_ACT" \
    --export="ALL,BACKBONE=coord_actuated,PROTOCOL_TAG=$PROTOCOL_TAG,RL_TMP_ROOT=$TMPROOT,TMPDIR=$TMPROOT" \
    slurm/bank_size_ablation_nested.sh)

EXP6=$(submit_id \
    --dependency="afterok:$PREP_FIXED:$PREP_SKIP" \
    --export="ALL,PROTOCOL_TAG=$PROTOCOL_TAG,RL_TMP_ROOT=$TMPROOT,TMPDIR=$TMPROOT" \
    slurm/exp6_backbone_ablation.sh)

echo "Ottawa formal suite submitted"
echo "  protocol tag                  : $PROTOCOL_TAG"
echo "  baseline coord_actuated      : $PREP_ACT"
echo "  baseline coord_fixed         : $PREP_FIXED"
echo "  baseline coord_actuated_skip : $PREP_SKIP"
echo "  EXP2 encoders (4)            : $EXP2"
echo "  EXP4 residual ablation (3)   : $EXP4"
echo "  EXP5 penetration (3)         : $EXP5"
echo "  bank sizes 200/500/2000 (3)  : $BANK"
echo "  EXP6 other backbones (2)     : $EXP6"
echo "  total new PPO runs           : 15"
echo "Monitor: squeue -j $PREP_ACT,$PREP_FIXED,$PREP_SKIP,$EXP2,$EXP4,$EXP5,$BANK,$EXP6"

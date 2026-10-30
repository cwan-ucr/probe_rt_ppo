#!/bin/bash
# Source BEFORE module loading. The module command itself needs writable temp
# storage; CorridorEnv gives RL_RUNTIME_DIR precedence over SLURM_TMPDIR.
# A unique directory also prevents equal worker PIDs on different nodes from
# colliding when scratch is on the shared bigdata filesystem.

rl_setup_scratch() {
    local scratch_root job_key probe_file
    scratch_root="${RL_TMP_ROOT:-${TMPROOT:-/rhome/cwan025/bigdata/tmp}}"
    case "$scratch_root" in
        /*) ;;
        *) echo "ERROR: RL_TMP_ROOT must be an absolute path" >&2; return 1 ;;
    esac
    mkdir -p "$scratch_root" || return 1
    job_key="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-0}"
    RL_JOB_SCRATCH=$(mktemp -d "$scratch_root/rl_${job_key}.XXXXXX") || return 1
    export RL_JOB_SCRATCH
    export TMPDIR="$RL_JOB_SCRATCH"
    export TMP="$RL_JOB_SCRATCH"
    export TEMP="$RL_JOB_SCRATCH"
    export RL_RUNTIME_DIR="$RL_JOB_SCRATCH"

    # Keep fallback TraCI locks shared by this user's jobs on the same node.
    # Do not place locks in the per-job directory.
    export RL_TRACI_LOCK_DIR="$scratch_root/rl_traci_locks/$(hostname -s)"
    mkdir -p "$RL_TRACI_LOCK_DIR" || return 1
    echo "RL scratch: $RL_JOB_SCRATCH"
    echo "Scheduler scratch (overridden for RL): ${SLURM_TMPDIR:-unset}"
    df -h "$RL_JOB_SCRATCH" || return 1
    df -i "$RL_JOB_SCRATCH" || return 1
    probe_file="$RL_JOB_SCRATCH/.write_probe"
    if ! dd if=/dev/zero of="$probe_file" bs=1048576 count=1 2>/dev/null; then
        echo "ERROR: cannot write 1 MiB to $RL_JOB_SCRATCH; check space/quota" >&2
        return 1
    fi
    rm -f "$probe_file"
    # Retain the isolated directory for diagnosis. Never remove another job's
    # files or shared scratch from a running batch script.
}

rl_setup_scratch

"""TraCI port allocation helpers for concurrent local and Slurm runs."""
import os


_SAFE_PORT_LOW = 10_000
_DEFAULT_EPHEMERAL_LOW = 32_768
_MIN_RANGE_STRIDE = 128


def _linux_ephemeral_low():
    """Return the first kernel ephemeral port, with a conservative fallback."""
    try:
        with open("/proc/sys/net/ipv4/ip_local_port_range") as f:
            return int(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return _DEFAULT_EPHEMERAL_LOW


def _validate_port_base(port_base, required_span):
    if port_base < 1024 or port_base + required_span - 1 > 65_535:
        raise ValueError(
            f"TraCI port base {port_base} with span {required_span} is outside "
            "the usable TCP port range")
    return port_base


def resolve_port_base(explicit=0, required_span=1):
    """Choose a job-specific server-port range below Linux ephemeral ports.

    Keeping TraCI listening ports below the ephemeral range matters when many
    workers start together: a client's ephemeral source port can otherwise claim
    another worker's not-yet-opened listening port.
    """
    required_span = max(1, int(required_span))
    if explicit:
        return _validate_port_base(int(explicit), required_span)

    env_value = os.environ.get("RL_PORT_BASE")
    if env_value:
        try:
            return _validate_port_base(int(env_value), required_span)
        except ValueError as exc:
            raise ValueError(f"invalid RL_PORT_BASE={env_value!r}: {exc}") from exc

    safe_high = min(_linux_ephemeral_low() - 1, _DEFAULT_EPHEMERAL_LOW - 1)
    stride = max(_MIN_RANGE_STRIDE, required_span)
    slots = (safe_high - _SAFE_PORT_LOW + 1) // stride
    if slots < 1:
        raise RuntimeError(
            f"no non-ephemeral TraCI port range can fit {required_span} ports")

    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    base_id = int(job_id) if job_id and job_id.isdigit() else os.getpid()
    return _SAFE_PORT_LOW + (base_id % slots) * stride

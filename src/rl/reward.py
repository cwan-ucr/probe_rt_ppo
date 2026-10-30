"""
Detector-based corridor reward (probe-only).

For each corridor segment with a probe reading this 15-min interval:
    delay_seg = N_seg * (L_km) * (1/v_seg - 1/v_free)     [vehicle-hours]
with v in kph, L in km, N = raw probe count (penetration is a constant scale and
does not change the argmax). For percentile observations, the expectation is
approximated by trapezoidal integration over quantile probability. Reward is
``-REWARD_SCALE * sum(delay_seg)``.

Only OFFSET is controlled, so the reward responds mainly through mainline
through-segment speeds (better progression -> higher v -> less delay).
"""
from . import config as C


def quantile_trapezoid_weights(percentiles):
    """Return composite-trapezoid weights for a quantile grid in percent.

    The grid must span the complete probability interval [0, 100]. For the
    project's 0, 5, ..., 100 grid, this returns endpoint weights 0.025 and
    interior weights 0.05. The weights sum to one.
    """
    q = [float(value) for value in percentiles]
    if len(q) < 2:
        raise ValueError("quantile trapezoid integration needs at least two points")
    if abs(q[0]) > 1e-12 or abs(q[-1] - 100.0) > 1e-12:
        raise ValueError("quantile grid must span 0 to 100 percent")
    if any(right <= left for left, right in zip(q, q[1:])):
        raise ValueError("quantile grid must be strictly increasing")

    weights = [0.0] * len(q)
    for index in range(len(q) - 1):
        half_width = (q[index + 1] - q[index]) / 200.0
        weights[index] += half_width
        weights[index + 1] += half_width
    return weights


_EXPECTED_DELAY_WEIGHTS = quantile_trapezoid_weights(C.SPEED_PERCENTILES)


def _seg_delay(speed_kph, count, length_m, vfree_kph):
    """Delay from the aggregate (harmonic-mean) segment speed [veh-hours]."""
    if count <= 0 or speed_kph is None or speed_kph < 0 or vfree_kph <= 0:
        return 0.0
    v = max(speed_kph, C.REWARD_MIN_SPEED_KPH)
    inv = (1.0 / v) - (1.0 / vfree_kph)          # h/km
    if inv <= 0:
        return 0.0
    return count * (length_m / 1000.0) * inv     # veh-hours


def _seg_delay_expected(pct_speeds, count, length_m, vfree_kph):
    """Estimate expected delay by trapezoidal quantile integration.

    ``D = count * L_km * integral_0^1 max(0, 1/Q_v(u)-1/v_free) du``.
    The integral is evaluated on the configured 21-point percentile grid. This
    avoids assigning a full 1/21 probability mass to both boundary quantiles.
    """
    if count <= 0 or vfree_kph <= 0 or not pct_speeds:
        return 0.0
    if len(pct_speeds) != len(C.SPEED_PERCENTILES):
        raise ValueError(
            f"expected {len(C.SPEED_PERCENTILES)} speed percentiles, "
            f"received {len(pct_speeds)}")
    if any(v is None or v < 0 for v in pct_speeds):
        return 0.0
    acc = 0.0
    for weight, v in zip(_EXPECTED_DELAY_WEIGHTS, pct_speeds):
        vv = max(v, C.REWARD_MIN_SPEED_KPH)
        d = (1.0 / vv) - (1.0 / vfree_kph)       # h/km, per percentile
        if d > 0:
            acc += weight * d
    return count * (length_m / 1000.0) * acc     # veh-hours


def segment_delay_vehh(reading, length_m, vfree_kph):
    """Return the probe-estimated delay represented by one segment reading.

    ``reading`` follows the environment contract ``(harmonic_speed, count)`` or
    ``(harmonic_speed, count, percentile_speeds)``.  Keeping this dispatch in one
    public helper guarantees that the observation's delay-ratio feature and the
    reward use the same expected-delay estimator and aggregate-speed fallback.
    """
    if reading is None or len(reading) < 2:
        return 0.0
    pct = reading[2] if len(reading) > 2 else None
    if C.USE_EXPECTED_DELAY and pct:
        return _seg_delay_expected(pct, reading[1], length_m, vfree_kph)
    return _seg_delay(reading[0], reading[1], length_m, vfree_kph)


def corridor_reward(readings, geom, return_breakdown=False):
    """Return scalar reward (=-total delay). Optionally a per-intersection dict.

    Each reading is (harmonic_speed, count) or (harmonic_speed, count,
    percentile_speeds). With USE_EXPECTED_DELAY and percentiles present, use the
    expected-delay-over-percentiles form; else the aggregate-speed form."""
    total = 0.0
    per_int = {}
    # Adjacent all10 approach windows intentionally share some TomTom segments so
    # each node can observe its local upstream/downstream context.  That is useful
    # in the state, but a centralized corridor reward must count each physical
    # segment once.  Keep legacy8 unchanged for checkpoint reproducibility.
    deduplicate = C.CONTROL_LAYOUT == "all10"
    counted_sids = set()
    for name in C.AGENT_ORDER:
        d_int = 0.0
        for d in C.DIRECTIONS:
            segs = geom[name][d]
            if not segs:
                continue
            seg_rd = readings.get(f"{name}_{d}", {})
            for s in segs:
                rd = seg_rd.get(s.sid)
                if rd is None:
                    continue
                if deduplicate and s.sid in counted_sids:
                    continue
                if deduplicate:
                    counted_sids.add(s.sid)
                d_int += segment_delay_vehh(rd, s.length, s.vfree)
        per_int[name] = d_int
        total += d_int
    reward = -C.REWARD_SCALE * total
    if return_breakdown:
        return reward, {"total_delay_vehh": total, "per_intersection": per_int}
    return reward

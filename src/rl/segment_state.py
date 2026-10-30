"""
Probe-only state construction for the centralized agent.

Input per 15-min interval:
    readings = {
        approach_id: {
            sid(str): (harmonic_speed_kph, probe_count, speed_percentiles)
        }
    }
(the percentile vector may be omitted by legacy callers, in which case the
harmonic-speed delay estimator is used.)

Per (intersection, direction) we aggregate the variable-length segment list into
7 fixed features. Per intersection we append a 4-d approach-presence mask in
[eb, wb, nb, sb] order, then append a global time-of-day (sin/cos). Everything is
derivable from segment speed percentiles + count -> deployable on probe data alone.
"""
import math
import numpy as np

from . import config as C
from .reward import segment_delay_vehh


def _dir_features(segs, seg_readings):
    """7 features for one approach.  segs: nearest-first list[SegInfo]."""
    v_ratios, lengths, has_data, counts = [], [], [], []
    delay_vehh = 0.0
    freeflow_vehh = 0.0
    for s in segs:
        rd = seg_readings.get(s.sid)
        if rd is not None and rd[1] > 0 and rd[0] is not None and rd[0] >= 0 \
                and s.vfree > 0:
            vr = min(rd[0] / s.vfree, C.VRATIO_CLIP)
            v_ratios.append(vr); has_data.append(True); counts.append(rd[1])
            delay_vehh += segment_delay_vehh(rd, s.length, s.vfree)
            freeflow_vehh += (
                float(rd[1]) * (float(s.length) / 1000.0) / float(s.vfree))
        else:
            v_ratios.append(None); has_data.append(False); counts.append(0)
        lengths.append(s.length)

    valid = [v for v in v_ratios if v is not None]
    if not valid:                      # no probe anywhere on this approach
        return np.array(
            [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    # nearest-stop-bar valid segment
    v_stop = next(v for v in v_ratios if v is not None)
    v_mean = float(np.mean(valid))
    v_min = float(np.min(valid))

    # queue length: from the stop bar upstream, accumulate consecutive congested
    # (or treat a data gap as "unknown" and stop)
    qlen = 0.0
    for v, L, hd in zip(v_ratios, lengths, has_data):
        if hd and v < C.QUEUE_VRATIO_THRESH:
            qlen += L
        else:
            break
    qlen_norm = min(qlen / C.QUEUE_LEN_NORM_M, 2.0)

    n_probes = float(sum(counts))
    n_log = min(math.log1p(n_probes) / 5.0, 1.5)
    staleness = 1.0 - (sum(has_data) / len(segs))

    # Exposure-weighted direction-level delay ratio. The numerator uses exactly
    # the same 21-quantile trapezoidal estimator as the reward (with the same
    # harmonic-speed fallback). Dividing by probe free-flow vehicle-hours removes
    # the raw effects of segment length and observed demand. A log transform keeps
    # severe low-speed tails numerically bounded while preserving delta=0 at free
    # flow and mapping delta=1 (travel time = 2x free flow) to 0.5.
    delay_ratio = delay_vehh / freeflow_vehh if freeflow_vehh > 0 else 0.0
    delay_ratio_norm = min(
        math.log1p(max(delay_ratio, 0.0)) /
        math.log1p(C.DELAY_RATIO_SATURATION),
        1.0,
    )

    return np.array(
        [v_stop, v_mean, v_min, qlen_norm, n_log, staleness,
         delay_ratio_norm], dtype=np.float32)


def build_state(readings, geom, offsets, tod_frac):
    """Return (flat_state[float32], structured) for the centralized policy.

    offsets:  kept for call-site compatibility; not included in the observation.
    tod_frac: fraction through the episode window in [0,1].
    """
    structured = {}
    blocks = []
    for name in C.AGENT_ORDER:
        per_dir = {}
        approach_mask = []
        for d in C.DIRECTIONS:
            segs = geom[name][d]
            approach_mask.append(1.0 if segs else 0.0)
            seg_rd = readings.get(f"{name}_{d}", {})
            feats = _dir_features(segs, seg_rd) if segs else \
                np.array([1, 1, 1, 0, 0, 1, 0], dtype=np.float32)
            per_dir[d] = feats
            blocks.append(feats)
        mask = np.array(approach_mask, dtype=np.float32)
        blocks.append(mask)
        structured[name] = {"dirs": per_dir, "approach_mask": mask,
                            "offset": offsets.get(name, 0)}

    tod = np.array([math.sin(2 * math.pi * tod_frac),
                    math.cos(2 * math.pi * tod_frac)], dtype=np.float32)
    blocks.append(tod)
    flat = np.concatenate(blocks).astype(np.float32)
    assert flat.shape[0] == C.STATE_DIM, (flat.shape[0], C.STATE_DIM)
    return flat, structured


# ---- helper: parse the offline detector CSV into per-interval readings ----
def _parse_percentiles(cell):
    """The CSV 'speed_percentiles' column is a list repr like '[12.3, 15.1, ...]'."""
    if isinstance(cell, (list, tuple)):
        return [float(x) for x in cell]
    try:
        import ast
        return [float(x) for x in ast.literal_eval(cell)]
    except Exception:
        return []


def readings_from_detector_df(df, interval_begin):
    """{approach_id: {sid: (harmonic_speed, count, percentile_speeds)}} for one
    interval_begin (percentiles -> expected-delay reward, same as the live env)."""
    sub = df[df["interval_begin"] == interval_begin]
    has_pct = "speed_percentiles" in df.columns
    out = {}
    for _, r in sub.iterrows():
        app = r["approach_id"]
        pct = _parse_percentiles(r["speed_percentiles"]) if has_pct else []
        out.setdefault(app, {})[str(int(r["detector_id"]))] = (
            float(r["harmonic_mean_speed_kph"]), int(r["vehicle_count"]), pct)
    return out

"""
Static corridor geometry for state building: for each (intersection, direction)
the ordered list of TomTom segments with distance-from-intersection, length and
free-flow speed. Loaded once from the approach CSVs (timeSet 2 row per segment).

Segments are ordered NEAREST-the-stop-bar first (cum_dist ascending), which is
what queue_len / v_ratio_stopbar need.
"""
import os
import numpy as np
import pandas as pd

from . import config as C


class SegInfo:
    __slots__ = ("sid", "cum_dist", "length", "vfree")

    def __init__(self, sid, cum_dist, length, vfree):
        self.sid = sid
        self.cum_dist = cum_dist
        self.length = length
        self.vfree = vfree


def _load_approach(approach_id):
    """Return list[SegInfo] ordered nearest->farthest, or [] if no CSV."""
    fp = os.path.join(C.APPROACH_DIR, f"{approach_id}_delay_time_results.csv")
    if not os.path.exists(fp):
        return []
    df = pd.read_csv(fp)
    t0 = df["timeSet"].min()
    dt = df[df["timeSet"] == t0]
    dist = dt["distance"].astype(float).values          # per-segment length
    sids = dt["simpleID"].astype(str).values
    vfree = dt["speedLimit"].astype(float).values
    cum = np.flip(np.cumsum(np.flip(dist)))             # distance from intersection
    segs = [SegInfo(sids[i], float(cum[i]), float(dist[i]), float(vfree[i]))
            for i in range(len(sids))]
    segs.sort(key=lambda s: s.cum_dist)                 # nearest stop bar first
    return segs


def load_geometry(intersections=None):
    """Return approach geometry for the requested intersections.

    By default the intersections selected by the active control layout are
    loaded (`legacy8` or `all10`). Visualization recorders may pass an explicit
    list when they need a different corridor subset.
    """
    names = list(intersections) if intersections is not None else C.AGENT_ORDER
    geom = {}
    for name in names:
        geom[name] = {}
        for d in C.DIRECTIONS:
            geom[name][d] = _load_approach(f"{name}_{d}")
    return geom


def summary(geom):
    for name in geom:
        parts = []
        for d in C.DIRECTIONS:
            segs = geom[name][d]
            if segs:
                parts.append(f"{d}:{len(segs)}seg({segs[0].cum_dist:.0f}-"
                             f"{segs[-1].cum_dist:.0f}m)")
            else:
                parts.append(f"{d}:-")
        print(f"  {name:9s} " + "  ".join(parts))


if __name__ == "__main__":
    g = load_geometry()
    summary(g)

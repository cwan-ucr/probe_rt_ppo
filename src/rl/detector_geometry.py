"""Static TomTom segment boundaries for the probe-vehicle collector.

The v8 induction loops are geometric points.  Probe collection only needs their
lane and position, so loading all loops into SUMO and querying them over TraCI on
every episode is unnecessary.  Parse the same additional file once per worker
and attach immutable boundary points to the segment definitions.
"""
from functools import lru_cache
import xml.etree.ElementTree as ET

from tsc_sumo_sim import parse_segments_v8


@lru_cache(maxsize=None)
def load_probe_segments(approach_folder, detector_file):
    """Return cached v8 segments with static entry/exit detector geometry."""
    points = {}
    for det in ET.parse(detector_file).getroot().findall(".//inductionLoop"):
        det_id = det.get("id", "")
        lane = det.get("lane")
        pos = det.get("pos")
        if det_id.startswith("segLoop_") and lane and pos is not None:
            points[det_id] = (lane, float(pos))

    detector_ids = tuple(points)
    kept = []
    for raw in parse_segments_v8(approach_folder):
        seg = dict(raw)
        sid = seg["sid"]
        end_prefix = f"segLoop_{sid}_END_"
        start_prefix = f"segLoop_{sid}_"
        exit_ids = sorted(d for d in detector_ids if d.startswith(end_prefix))
        entry_ids = sorted(
            d for d in detector_ids
            if d.startswith(start_prefix) and not d.startswith(end_prefix)
        )
        if not entry_ids or not exit_ids:
            continue
        seg["entry"] = entry_ids
        seg["exit"] = exit_ids
        seg["entry_points"] = tuple(points[d] for d in entry_ids)
        seg["exit_points"] = tuple(points[d] for d in exit_ids)
        kept.append(seg)
    return kept

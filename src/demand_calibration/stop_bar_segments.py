"""
Lookup: approach_id → stop-bar simpleID for matching TomTom and SUMO.

Important convention:
  TomTom defines each simpleID as a segment with its own length and speed.
  SUMO's tsc_sumo_sim builds segments by pairing CONSECUTIVE simpleIDs:
    segment(i)  has  up_id=simpleID[i],  down_id=simpleID[i+1]
  This means the LAST simpleID in a chain has no segment of its own —
  it's only used as the down_id of the previous segment.
  So the SUMO output's `detector_id` column never contains the last
  simpleID; it only contains up_ids = simpleID[0 .. n-2].

For Stage A (stop-bar-only calibration):
  We want the segment IMMEDIATELY before the intersection — this is
  the LAST SUMO segment, whose up_id = simpleID[n-2] of the chain.
  Both the TomTom target speed and the SUMO sim speed are taken at
  this simpleID. (Despite the name "stop bar", physically this is
  just the last segment in the approach chain.)

Source for chain order: data/geojson-1-15-min-interval/processed/
  *_delay_time_results.csv (rows ordered by upstream → downstream
  within timeSet=2).
"""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.dirname(SCRIPT_DIR)
ROOT_DIR   = os.path.dirname(SRC_DIR)

DEFAULT_APPROACH_DIR = os.path.join(ROOT_DIR, "data",
                                    "geojson-1-15-min-interval", "processed")


def load_stop_bar_simple_ids(approach_dir=DEFAULT_APPROACH_DIR,
                              ranked=False):
    """Return {approach_id: simpleID_int} for stop-bar matching.

    For each approach, returns simpleID[n-2] of the chain — the up_id
    of the last SUMO segment in the chain. This is the same simpleID
    that appears as `detector_id` in the SUMO sim output for the
    most-downstream segment.

    If `ranked=True`, return {approach_id: [simpleID_int, ...]} listing
    candidate stop-bar simpleIDs from the last SUMO segment going
    backwards (chain[n-2], chain[n-3], ..., chain[0]). This lets callers
    fall back to an upstream segment when the preferred stop-bar segment
    has no traffic in simulation (e.g., when it's beyond the corridor
    boundary).
    """
    # Mirror tsc_sumo_sim.APPROACHES_REVERSED_CHAIN — approaches whose
    # CSV chain is in REVERSED traffic order must be flipped before
    # picking the stop-bar simpleID.
    APPROACHES_REVERSED_CHAIN = {"Cranford_wb"}

    out = {}
    for fname in sorted(os.listdir(approach_dir)):
        if not fname.endswith("_delay_time_results.csv"):
            continue
        approach_id = fname.replace("_delay_time_results.csv", "")
        df = pd.read_csv(os.path.join(approach_dir, fname))
        # Use timeSet=2 (the first timeSet) to get the canonical chain order
        chain = df[df["timeSet"] == 2]["simpleID"].astype(int).tolist()
        if len(chain) < 2:
            continue
        if approach_id in APPROACHES_REVERSED_CHAIN:
            chain = list(reversed(chain))
        if ranked:
            # All up_ids in descending preference order (most downstream first)
            out[approach_id] = list(reversed(chain[:-1]))
        else:
            # SUMO last segment's up_id = simpleID[n-2]
            out[approach_id] = chain[-2]
    return out


# Number of most-downstream segments (incl. the stop-bar segment) used by the
# calibration objective. Speeds are matched per-segment over these N. Kept small
# (3) so the objective stays near the stop bar — including far-upstream segments
# forces the GA to over-inflate demand to reproduce their (queued) speeds.
N_NEAREST_STOP_BAR = 3


def load_nearest_stop_bar_sids(approach_dir=DEFAULT_APPROACH_DIR,
                               n=N_NEAREST_STOP_BAR):
    """{approach_id: [simpleID, ...]} — the n segments nearest the stop bar
    (most-downstream first; the stop-bar segment chain[n-2] is element 0)."""
    ranked = load_stop_bar_simple_ids(approach_dir, ranked=True)
    return {k: v[:n] for k, v in ranked.items()}


if __name__ == "__main__":
    m = load_stop_bar_simple_ids()
    for k, v in sorted(m.items()):
        print(f"  {k:<15} → {v}")
    print(f"\nTotal: {len(m)} approaches")

"""
Generate NEMA-based fixed-time signal timing plans from TomTom 15-min flow data.

For each corridor intersection, for each 15-min TomTom interval:
  1. Read approach flows (last segment = closest to stop line)
  2. Apply Webster's formula to compute optimal cycle + green splits
  3. Preserve existing SUMO phase state strings; only adjust durations

Output: data/corridor-osm-network/fixed_timing.add.xml
  - One <tlLogic> program per (intersection × 15-min interval)
  - <WAUT> + <wautJunction> for automatic plan switching

NEMA phase classification (heuristic):
  - Phase group A (arterial EB+WB through): phase with the most G's in state string
    in the first half of the phase sequence
  - Phase group B (cross-street NB+SB through): phase with the most G's in second half
  - Yellow/all-red clearance phases: fixed durations (not adjusted)

Usage:
    python src/signal_timing/generate_fixed_timing.py
    python src/signal_timing/generate_fixed_timing.py --out data/corridor-osm-network/fixed_timing.add.xml
"""

import os
import sys
import argparse
import json
import math
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.dirname(SCRIPT_DIR)
ROOT_DIR   = os.path.dirname(SRC_DIR)

sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import sumolib

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_NET     = os.path.join(ROOT_DIR, "data", "corridor-osm-network", "osm.net.xml.gz")
DEFAULT_FLOWS   = os.path.join(ROOT_DIR, "outputs", "calibration", "segment_flows_estimated.csv")
DEFAULT_TOMTOM_DIR = os.path.join(ROOT_DIR, "data", "geojson-1-15-min-interval", "processed")
DEFAULT_PENETRATION = os.path.join(ROOT_DIR, "outputs", "calibration", "penetration_rate.json")
DEFAULT_CALIBRATED_RESULT = os.path.join(ROOT_DIR, "outputs", "calibration",
                                         "calibration_result_turning.json")
DEFAULT_BND     = os.path.join(ROOT_DIR, "outputs", "calibration", "corridor_boundary_edges.json")
DEFAULT_OUT     = os.path.join(ROOT_DIR, "data", "corridor-osm-network", "fixed_timing.add.xml")

# Number of TomTom segments closest to the stop line to average for timing demand.
DEFAULT_LAST_N_SEGMENTS = 5

# Corridor intersections that have TomTom cross-street data
CORRIDOR = ["Lime", "Park", "Victoria", "Kansas", "Chicago", "Iowa"]

FULL_CORRIDOR_ORDER = [
    "Lime", "Park", "Victoria", "Eucalyptus",
    "Kansas", "Ottawa", "Chicago", "Cranford", "Iowa",
]

# TL → intersection mapping (derived from approach-edge matching)
TL_INTERSECTION = {
    "54553323":                                                 "Lime",
    "54424419":                                                 "Park",
    "GS_cluster_5910940017_5910976489":                         "Victoria",
    "GS_cluster_54287971_5910976488":                           "Kansas",
    "GS_cluster_4689533026_54483626_5910945475_5910945476":     "Chicago",
    "GS_cluster_5910933593_9013582988":                         "Iowa",
}

# Saturation flow per lane (vph/lane)
SAT_THROUGH   = 1650
SAT_LEFT      = 1650

# Assumed left-turn share when no turning-count data are available.
LEFT_SHARE_ART   = 0.20
LEFT_SHARE_CROSS = 0.30

# Cycle bounds (seconds)
C_MIN = 60
C_MAX = 150

# Lost time per phase transition (yellow + all-red, seconds)
LOST_TIME_PER_PHASE = 4.0

# Minimum green time (seconds)
MIN_GREEN = 7.0

# TomTom time mapping: timeSet=2 → 07:00, each step = 15 min
TOMTOM_START_HOUR = 7
TOMTOM_MIN_TIMESET = 2
SIM_START_HOUR = 7  # simulation starts at 07:00
ARTERIAL_ROAD_NAME = "University Avenue"


# ── Phase classification ───────────────────────────────────────────────────────

def classify_phases(phases):
    """
    Classify phase indices into arterial-through, cross-through, and clearance groups.

    Heuristic:
      - Count uppercase G in each state string
      - Phases with G_count > 0 and duration > 5s are "through" candidates
      - The phase(s) with the most G's in the first half → arterial group
      - The phase(s) with the most G's in the second half → cross-street group
      - Short phases (duration ≤ 5s) or all-lowercase → clearance/yellow

    Returns
    -------
    art_idx   : list[int] – phase indices serving arterial through
    cross_idx : list[int] – phase indices serving cross-street through
    clear_idx : list[int] – yellow / all-red clearance phases (fixed duration)
    """
    n = len(phases)
    g_counts = [p.state.count('G') for p in phases]
    durations = [p.duration for p in phases]

    # Mark clearance phases (duration ≤ 5s OR only lowercase/r states)
    is_clearance = [
        d <= 5.5 or p.state.count('G') == 0
        for d, p in zip(durations, phases)
    ]

    through_indices = [i for i in range(n) if not is_clearance[i]]
    if not through_indices:
        # Fallback: treat first half as arterial, second half as cross
        mid = n // 2
        return list(range(0, mid, 2)), list(range(mid, n, 2)), \
               [i for i in range(n) if i not in range(0, mid, 2) and i not in range(mid, n, 2)]

    # Split through phases into two groups by position in sequence
    mid = n // 2
    first_half  = [i for i in through_indices if i < mid]
    second_half = [i for i in through_indices if i >= mid]

    if not first_half:
        first_half = through_indices[:len(through_indices)//2 or 1]
    if not second_half:
        second_half = through_indices[len(through_indices)//2:]

    clear_idx = [i for i in range(n) if is_clearance[i]]
    return first_half, second_half, clear_idx


def active_link_indices(state):
    """Return controlled-link indices that are green or permissive-green."""
    return [i for i, color in enumerate(state) if color in ("G", "g")]


def _bearing(p0, p1):
    return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))


def _turn_type(in_lane, out_lane):
    """Classify one SUMO lane-to-lane connection as left/through/right."""
    in_shape = in_lane.getEdge().getShape()
    out_shape = out_lane.getEdge().getShape()
    if len(in_shape) < 2 or len(out_shape) < 2:
        return "through"

    in_bearing = _bearing(in_shape[-2], in_shape[-1])
    out_bearing = _bearing(out_shape[0], out_shape[1])
    delta = (out_bearing - in_bearing + 540) % 360 - 180
    if delta > 35:
        return "left"
    if delta < -35:
        return "right"
    return "through"


def build_signal_link_table(tl):
    """Map each SUMO controlled-link index to incoming lane and turn metadata."""
    by_index = {}
    for in_lane, out_lane, link_index in tl.getConnections():
        by_index.setdefault(link_index, []).append({
            "in_lane": in_lane,
            "out_lane": out_lane,
            "in_lane_id": in_lane.getID(),
            "in_edge_id": in_lane.getEdge().getID(),
            "turn": _turn_type(in_lane, out_lane),
        })
    return by_index


def classify_phases_by_incoming_road(tl, phases,
                                     arterial_road_name=ARTERIAL_ROAD_NAME):
    """
    Classify green phases by the road name of active incoming SUMO links.

    The previous phase-position heuristic fails at some intersections because
    their NEMA phase order is reversed. Incoming road name is more reliable:
    University Avenue incoming links are arterial phases; all other incoming
    links are cross-street phases.
    """
    link_table = build_signal_link_table(tl)
    art_idx = []
    cross_idx = []
    clear_idx = []

    for idx, phase in enumerate(phases):
        if phase.duration <= 5.5 or phase.state.count("G") == 0:
            clear_idx.append(idx)
            continue

        art_count = 0
        cross_count = 0
        for link_index in active_link_indices(phase.state):
            for link in link_table.get(link_index, []):
                in_name = link["in_lane"].getEdge().getName()
                if in_name == arterial_road_name:
                    art_count += 1
                else:
                    cross_count += 1

        if art_count == 0 and cross_count == 0:
            clear_idx.append(idx)
        elif art_count >= cross_count:
            art_idx.append(idx)
        else:
            cross_idx.append(idx)

    if not art_idx or not cross_idx:
        fallback_art, fallback_cross, fallback_clear = classify_phases(phases)
        return fallback_art, fallback_cross, fallback_clear

    return art_idx, cross_idx, clear_idx


def critical_lane_count(state, link_table, movement):
    """
    Count lanes serving a phase movement on the critical incoming edge.

    For a two-way phase, opposing approaches are usually active together. Webster
    uses the critical lane group, so this counts unique active incoming lanes per
    incoming edge and returns the maximum, not the sum across opposing approaches.
    """
    lanes_by_edge = {}
    for link_index in active_link_indices(state):
        for link in link_table.get(link_index, []):
            if movement == "left" and link["turn"] != "left":
                continue
            if movement == "through" and link["turn"] == "left":
                continue
            lanes_by_edge.setdefault(link["in_edge_id"], set()).add(link["in_lane_id"])

    if not lanes_by_edge:
        for link_index in active_link_indices(state):
            for link in link_table.get(link_index, []):
                lanes_by_edge.setdefault(link["in_edge_id"], set()).add(link["in_lane_id"])

    if not lanes_by_edge:
        return 1
    return max(1, max(len(lanes) for lanes in lanes_by_edge.values()))


def build_phase_specs(tl, phases, art_idx, cross_idx,
                      left_share_art=LEFT_SHARE_ART,
                      left_share_cross=LEFT_SHARE_CROSS):
    """Build Webster phase groups with movement labels and detected lane counts."""
    link_table = build_signal_link_table(tl)
    specs = []

    def _add_group(indices, approach_group, left_share):
        if not indices:
            return
        has_left_phase = len(indices) > 1
        for pos, idx in enumerate(indices):
            movement = "through" if pos == 0 else "left"
            flow_share = 1.0 - left_share if movement == "through" and has_left_phase else 1.0
            if movement == "left":
                flow_share = left_share
            specs.append({
                "phase_index": idx,
                "approach_group": approach_group,
                "movement": movement,
                "flow_share": flow_share,
                "sat_per_lane": SAT_LEFT if movement == "left" else SAT_THROUGH,
                "lanes": critical_lane_count(phases[idx].state, link_table, movement),
                "min_green": MIN_GREEN,
            })

    _add_group(art_idx, "art", left_share_art)
    _add_group(cross_idx, "cross", left_share_cross)
    return specs


# ── Webster's formula ─────────────────────────────────────────────────────────

def webster_timing(q_art, q_cross, L_actual, n_lanes_art=2, n_lanes_cross=1):
    """
    Compute Webster optimal cycle and effective green times for 2 phase groups.

    Parameters
    ----------
    q_art     : arterial critical flow (vph) – max(EB, WB)
    q_cross   : cross-street critical flow (vph) – max(NB, SB)
    L_actual  : actual total lost time = sum of all clearance phase durations (s)

    Returns
    -------
    cycle_s       : optimal cycle length (seconds), clipped to [C_MIN, C_MAX]
    g_art_eff     : effective green for arterial group (seconds)
    g_cross_eff   : effective green for cross-street group (seconds)

    Invariant: g_art_eff + g_cross_eff + L_actual == cycle_s
    """
    s_art   = SAT_THROUGH * n_lanes_art
    s_cross = SAT_THROUGH * n_lanes_cross

    y_art   = q_art   / s_art   if s_art   > 0 else 0
    y_cross = q_cross / s_cross if s_cross > 0 else 0
    Y = y_art + y_cross

    # Webster formula — use L_actual as the total lost time
    Y_safe = min(Y, 0.90)
    if Y_safe <= 0 or L_actual <= 0:
        C = C_MIN
    else:
        C_raw = (1.5 * L_actual + 5.0) / (1.0 - Y_safe)
        C = float(np.clip(C_raw, C_MIN, C_MAX))

    # Available green time (must accommodate both groups)
    g_total = max(C - L_actual, 2 * MIN_GREEN)
    # Adjust C so that it equals g_total + L_actual exactly
    C = round(g_total + L_actual, 1)

    if Y <= 0:
        g_art_eff   = g_total / 2
        g_cross_eff = g_total / 2
    else:
        g_art_eff   = max(g_total * (y_art   / Y), MIN_GREEN)
        g_cross_eff = max(g_total * (y_cross / Y), MIN_GREEN)
        # Re-normalize if min green pushed total over budget
        if g_art_eff + g_cross_eff > g_total:
            scale = g_total / (g_art_eff + g_cross_eff)
            g_art_eff   *= scale
            g_cross_eff *= scale
        # Force exact sum to close rounding gaps
        g_cross_eff = round(g_total - round(g_art_eff, 1), 1)

    return round(C, 1), round(g_art_eff, 1), round(g_cross_eff, 1)


def webster_timing_for_signal(q_art, q_cross, L_actual,
                               n_art_phases, n_cross_phases,
                               n_lanes_art=2, n_lanes_cross=1):
    """Wrapper that applies per-phase minimum green before returning."""
    # Minimum green per group = MIN_GREEN × number of phases in that group
    min_g_art   = MIN_GREEN * n_art_phases
    min_g_cross = MIN_GREEN * n_cross_phases

    s_art   = SAT_THROUGH * n_lanes_art
    s_cross = SAT_THROUGH * n_lanes_cross

    y_art   = q_art   / s_art   if s_art   > 0 else 0
    y_cross = q_cross / s_cross if s_cross > 0 else 0
    Y = y_art + y_cross

    Y_safe = min(Y, 0.90)
    if Y_safe <= 0 or L_actual <= 0:
        C_raw = C_MIN
    else:
        C_raw = (1.5 * L_actual + 5.0) / (1.0 - Y_safe)
    C = float(np.clip(C_raw, C_MIN, C_MAX))

    g_total = max(C - L_actual, min_g_art + min_g_cross)
    C = round(g_total + L_actual, 1)

    if Y <= 0:
        g_art   = g_total / 2
        g_cross = g_total / 2
    else:
        g_art   = max(g_total * (y_art   / Y), min_g_art)
        g_cross = max(g_total * (y_cross / Y), min_g_cross)
        if g_art + g_cross > g_total:
            scale = g_total / (g_art + g_cross)
            g_art   = round(g_art   * scale, 1)
            g_cross = round(g_cross * scale, 1)
        # Close any rounding gap
        g_cross = round(g_total - round(g_art, 1), 1)

    return round(C, 1), round(g_art, 1), round(g_cross, 1)


def distribute_green(phases, art_idx, cross_idx, g_art, g_cross):
    """
    Distribute effective green time across the through-phase indices in each group.
    Clearance phases keep their original durations.

    Returns list of new durations (same length as phases).
    """
    new_dur = [p.duration for p in phases]  # start from existing

    def _assign(indices, g_total):
        if not indices:
            return
        per_phase = round(g_total / len(indices), 1)
        for i in indices:
            new_dur[i] = per_phase

    _assign(art_idx, g_art)
    _assign(cross_idx, g_cross)
    return new_dur


def webster_timing_for_phase_specs(q_art, q_cross, L_actual, phase_specs,
                                   movement_demands=None):
    """
    Compute Webster cycle and phase greens for detected phase/lane groups.

    Flow assumptions:
      - arterial through/right = q_art * (1 - LEFT_SHARE_ART) if a protected
        arterial left phase exists, otherwise q_art
      - arterial left = q_art * LEFT_SHARE_ART
      - cross through/right = q_cross * (1 - LEFT_SHARE_CROSS) if a protected
        cross left phase exists, otherwise q_cross
      - cross left = q_cross * LEFT_SHARE_CROSS
    """
    specs = []
    for spec in phase_specs:
        q_total = q_art if spec["approach_group"] == "art" else q_cross
        q = q_total * spec["flow_share"]
        if movement_demands is not None:
            key = (spec["approach_group"], spec["movement"])
            q = movement_demands.get(key, q)
        saturation = spec["sat_per_lane"] * spec["lanes"]
        y = q / saturation if saturation > 0 else 0
        spec = dict(spec)
        spec.update({
            "q": q,
            "saturation": saturation,
            "y": y,
        })
        specs.append(spec)

    Y = sum(spec["y"] for spec in specs)
    Y_safe = min(Y, 0.90)
    if Y_safe <= 0 or L_actual <= 0:
        C_raw = C_MIN
    else:
        C_raw = (1.5 * L_actual + 5.0) / (1.0 - Y_safe)
    C = float(np.clip(C_raw, C_MIN, C_MAX))

    min_green_total = sum(spec["min_green"] for spec in specs)
    g_total = max(C - L_actual, min_green_total)
    C = round(g_total + L_actual, 1)

    if not specs:
        return C, {}, specs

    min_green_total = sum(spec["min_green"] for spec in specs)
    residual_green = max(g_total - min_green_total, 0.0)

    if Y <= 0:
        extra = residual_green / len(specs)
        greens = {
            spec["phase_index"]: spec["min_green"] + extra
            for spec in specs
        }
    else:
        greens = {
            spec["phase_index"]: spec["min_green"] + residual_green * spec["y"] / Y
            for spec in specs
        }

    rounded = {idx: round(g, 1) for idx, g in greens.items()}
    if rounded:
        last_idx = list(rounded)[-1]
        gap = round(g_total - sum(rounded.values()), 1)
        rounded[last_idx] = round(rounded[last_idx] + gap, 1)

    return C, rounded, specs


def apply_phase_greens(phases, phase_greens):
    """Apply computed green durations while keeping clearance phases unchanged."""
    new_dur = [p.duration for p in phases]
    for idx, green in phase_greens.items():
        new_dur[idx] = green
    return new_dur


# ── Flow lookup ───────────────────────────────────────────────────────────────

def build_flow_table_from_segment_last(flows_csv):
    """
    Returns DataFrame indexed by (approach_id, timeSet) → V_veh_per_hour
    using the LAST segment (closest to stop line) for each approach.
    """
    df = pd.read_csv(flows_csv)
    last = df[df["is_last"]].copy()
    # Keep one row per (approach_id, timeSet) — should be unique
    last = last.drop_duplicates(subset=["approach_id", "timeSet"])
    last = last.set_index(["approach_id", "timeSet"])
    return last["V_veh_per_hour"]


def build_flow_table_from_tomtom_last_n(tomtom_dir, penetration_json, last_n=5):
    """
    Build timing demand from raw TomTom probe counts.

    For every approach CSV, this uses the last N segments closest to the stop
    line, averages their TomTom sampleSize per 15-min interval, scales by the
    global probe penetration rate, and converts the result to veh/h:

        V_veh_per_hour = (mean(sampleSize_last_N) / p_global) * 4

    This matches the validation plot's TomTom last-segment averaging convention.
    """
    with open(penetration_json) as f:
        penetration = json.load(f)
    p_global = float(penetration["p_global"])
    if p_global <= 0:
        raise ValueError(f"Invalid p_global in {penetration_json}: {p_global}")

    rows = []
    for fname in sorted(os.listdir(tomtom_dir)):
        if not fname.endswith("_delay_time_results.csv"):
            continue

        approach_id = fname.replace("_delay_time_results.csv", "")
        fpath = os.path.join(tomtom_dir, fname)
        df = pd.read_csv(fpath)
        if df.empty:
            continue

        df["simpleID"] = df["simpleID"].astype(str)
        min_ts = df["timeSet"].min()
        segment_order = df[df["timeSet"] == min_ts]["simpleID"].astype(str).tolist()
        selected = segment_order[-last_n:] if len(segment_order) > last_n else segment_order
        if not selected:
            continue

        sub = df[df["simpleID"].isin(selected)].copy()
        for ts, grp in sub.groupby("timeSet"):
            sample_avg = float(grp["sampleSize"].mean())
            veh_15min = sample_avg / p_global
            rows.append({
                "approach_id": approach_id,
                "timeSet": int(ts),
                "V_veh_per_hour": veh_15min * 4.0,
                "sampleSize_avg": sample_avg,
                "p_used": p_global,
                "n_segments": int(grp["simpleID"].nunique()),
            })

    if not rows:
        raise ValueError(f"No TomTom approach flow rows found in {tomtom_dir}")

    flow_df = pd.DataFrame(rows)
    flow_df = flow_df.drop_duplicates(subset=["approach_id", "timeSet"])
    flow_df = flow_df.set_index(["approach_id", "timeSet"])
    return flow_df["V_veh_per_hour"]


def _timeset_to_hour(timeset):
    return TOMTOM_START_HOUR + int((timeset - TOMTOM_MIN_TIMESET) // 4)


def _get_hourly_flow(flows, key, hour):
    per_hour = flows.get(key, {})
    if str(hour) in per_hour:
        return float(per_hour[str(hour)])
    if hour in per_hour:
        return float(per_hour[hour])
    return 0.0


def _crosses_with_exit(boundaries, exit_role):
    return [
        cross for cross in FULL_CORRIDOR_ORDER
        if cross in boundaries.get("cross", {})
        and exit_role in boundaries["cross"][cross]
    ]


def _remaining_turn_destinations(pool, intersection, direction):
    current = FULL_CORRIDOR_ORDER.index(intersection)
    if direction == "EB":
        return [c for c in pool if FULL_CORRIDOR_ORDER.index(c) >= current]
    return [c for c in pool if FULL_CORRIDOR_ORDER.index(c) <= current]


def build_flow_table_from_calibrated_turning(result_json, boundaries_json,
                                             start_timeset, end_timeset):
    """
    Build stop-line approach demands from calibrated boundary flows and
    calibrated turning ratios.

    The calibration result stores hourly entry flows at the corridor boundary.
    This expands them to 15-min timing intervals and estimates each controlled
    intersection's approach total, left-turn, and non-left demand. Arterial
    turns are distributed using the same equal-destination convention as
    TurningParameters.to_flow_xml().
    """
    with open(result_json) as f:
        result = json.load(f)
    with open(boundaries_json) as f:
        boundaries = json.load(f)

    flows = result["flows"]
    tr = result.get("turning_ratios", {})
    left_art = float(tr.get("left_art", LEFT_SHARE_ART))
    right_art = float(tr.get("right_art", 0.0))
    left_cross = float(tr.get("left_cross", LEFT_SHARE_CROSS))
    right_cross = float(tr.get("right_cross", 0.0))
    straight_art = max(0.0, 1.0 - left_art - right_art)

    nb_exits = _crosses_with_exit(boundaries, "north_exit_NB")
    sb_exits = _crosses_with_exit(boundaries, "south_exit_SB")

    rows = []

    def add_row(intersection, direction, timeset, total, left):
        rows.append({
            "approach_id": f"{intersection}_{direction.lower()}",
            "timeSet": int(timeset),
            "V_veh_per_hour": float(total),
            "left_vph": float(left),
            "nonleft_vph": max(float(total) - float(left), 0.0),
        })

    for ts in range(start_timeset, end_timeset + 1):
        hour = _timeset_to_hour(ts)
        eb_in = _get_hourly_flow(flows, "uni_west_entry_EB", hour)
        wb_in = _get_hourly_flow(flows, "uni_east_entry_WB", hour)

        eb_left_each = eb_in * left_art / len(nb_exits) if nb_exits else 0.0
        eb_right_each = eb_in * right_art / len(sb_exits) if sb_exits else 0.0
        wb_left_each = wb_in * left_art / len(sb_exits) if sb_exits else 0.0
        wb_right_each = wb_in * right_art / len(nb_exits) if nb_exits else 0.0

        for intersection in CORRIDOR:
            eb_left = eb_left_each if intersection in nb_exits else 0.0
            eb_remaining_left = len(_remaining_turn_destinations(
                nb_exits, intersection, "EB")) * eb_left_each
            eb_remaining_right = len(_remaining_turn_destinations(
                sb_exits, intersection, "EB")) * eb_right_each
            eb_total = eb_in * straight_art + eb_remaining_left + eb_remaining_right
            add_row(intersection, "EB", ts, eb_total, eb_left)

            wb_left = wb_left_each if intersection in sb_exits else 0.0
            wb_remaining_left = len(_remaining_turn_destinations(
                sb_exits, intersection, "WB")) * wb_left_each
            wb_remaining_right = len(_remaining_turn_destinations(
                nb_exits, intersection, "WB")) * wb_right_each
            wb_total = wb_in * straight_art + wb_remaining_left + wb_remaining_right
            add_row(intersection, "WB", ts, wb_total, wb_left)

            nb_total = _get_hourly_flow(
                flows, f"{intersection}_south_entry_NB", hour)
            sb_total = _get_hourly_flow(
                flows, f"{intersection}_north_entry_SB", hour)
            add_row(intersection, "NB", ts, nb_total, nb_total * left_cross)
            add_row(intersection, "SB", ts, sb_total, sb_total * left_cross)

    flow_df = pd.DataFrame(rows)
    if flow_df.empty:
        raise ValueError(f"No calibrated flow rows found in {result_json}")
    return flow_df.set_index(["approach_id", "timeSet"]), {
        "left_art": left_art,
        "right_art": right_art,
        "left_cross": left_cross,
        "right_cross": right_cross,
    }


def _lookup_flow_row(flow_table, approach_id, timeset):
    key = (approach_id, timeset)
    if key not in flow_table.index:
        return None
    return flow_table.loc[key]


def get_approach_flow(flow_table, intersection, direction, timeset, default=100):
    """Return stop-line flow (vph) for one approach direction at one time step."""
    approach_id = f"{intersection}_{direction.lower()}"
    row = _lookup_flow_row(flow_table, approach_id, timeset)
    if row is not None:
        v = row["V_veh_per_hour"] if isinstance(row, pd.Series) else row
        return max(int(v), 0)
    return default


def get_approach_components(flow_table, intersection, direction, timeset,
                            default=0, left_share=0.0):
    approach_id = f"{intersection}_{direction.lower()}"
    row = _lookup_flow_row(flow_table, approach_id, timeset)
    if row is None:
        total = float(default)
        left = total * left_share
        return {
            "total": total,
            "left": left,
            "nonleft": max(total - left, 0.0),
        }

    if isinstance(row, pd.Series) and "left_vph" in row.index:
        total = float(row["V_veh_per_hour"])
        left = float(row["left_vph"])
        nonleft = float(row["nonleft_vph"])
    else:
        total = float(row)
        left = total * left_share
        nonleft = max(total - left, 0.0)
    return {"total": total, "left": left, "nonleft": nonleft}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net",    default=DEFAULT_NET)
    ap.add_argument("--flow-source",
                    choices=["calibrated-turning", "tomtom-last-n", "segment-last"],
                    default="tomtom-last-n",
                    help="Timing demand source. Default uses TomTom last-N sampleSize / p_global.")
    ap.add_argument("--flows",  default=DEFAULT_FLOWS)
    ap.add_argument("--calibrated-result", default=DEFAULT_CALIBRATED_RESULT)
    ap.add_argument("--tomtom-dir", default=DEFAULT_TOMTOM_DIR)
    ap.add_argument("--penetration", default=DEFAULT_PENETRATION)
    ap.add_argument("--last-n", type=int, default=DEFAULT_LAST_N_SEGMENTS,
                    help="Number of TomTom segments closest to the stop line to average")
    ap.add_argument("--bnd",    default=DEFAULT_BND)
    ap.add_argument("--out",    default=DEFAULT_OUT)
    ap.add_argument("--start-timeset", type=int, default=2,
                    help="First TomTom timeSet (default=2 → 07:00)")
    ap.add_argument("--end-timeset",   type=int, default=25,
                    help="Last TomTom timeSet inclusive (default=25 → 12:45)")
    args = ap.parse_args()

    print("=" * 60)
    print("  NEMA Fixed-Time Signal Timing Generator")
    print("=" * 60)

    # Load SUMO network
    print(f"\n[1] Loading network: {args.net}")
    net = sumolib.net.readNet(args.net, withPrograms=True)
    tls_by_id = {tl.getID(): tl for tl in net.getTrafficLights()}

    # Load timing demand data
    turning_ratios = {
        "left_art": LEFT_SHARE_ART,
        "right_art": 0.0,
        "left_cross": LEFT_SHARE_CROSS,
        "right_cross": 0.0,
    }
    if args.flow_source == "calibrated-turning":
        print(f"[2] Loading calibrated turning demand: {args.calibrated_result}")
        print(f"    Boundary geometry: {args.bnd}")
        flow_table, turning_ratios = build_flow_table_from_calibrated_turning(
            args.calibrated_result, args.bnd, args.start_timeset, args.end_timeset
        )
        print("    Calibrated turning ratios: "
              f"L_art={turning_ratios['left_art']:.3f}, "
              f"R_art={turning_ratios['right_art']:.3f}, "
              f"L_cross={turning_ratios['left_cross']:.3f}, "
              f"R_cross={turning_ratios['right_cross']:.3f}")
    elif args.flow_source == "tomtom-last-n":
        print(f"[2] Loading TomTom last-{args.last_n} flow data: {args.tomtom_dir}")
        print(f"    Scaling sampleSize by penetration: {args.penetration}")
        flow_table = build_flow_table_from_tomtom_last_n(
            args.tomtom_dir, args.penetration, last_n=args.last_n
        )
    else:
        print(f"[2] Loading segment-last flow data: {args.flows}")
        flow_table = build_flow_table_from_segment_last(args.flows)

    time_steps = list(range(args.start_timeset, args.end_timeset + 1))
    print(f"    {len(time_steps)} time steps: timeSet {time_steps[0]} "
          f"(07:00) → {time_steps[-1]} ({7 + (time_steps[-1]-2)*0.25:.2f})")

    # Build XML
    root = ET.Element("additional")

    # ── Process each corridor intersection ───────────────────────────────────
    print(f"\n[3] Generating timing plans for {len(TL_INTERSECTION)} intersections...")

    for tl_id, intersection in TL_INTERSECTION.items():
        if tl_id not in tls_by_id:
            print(f"  WARN: TL {tl_id} not found in network — skipping")
            continue

        tl = tls_by_id[tl_id]
        base_prog = list(tl.getPrograms().values())[0]
        phases = base_prog._phases
        n_phases = len(phases)

        art_idx, cross_idx, clear_idx = classify_phases_by_incoming_road(
            tl, phases
        )

        # Actual lost time = sum of all clearance phase durations (yellow + all-red)
        L_actual = sum(phases[i].duration for i in clear_idx)

        has_nb = f"{intersection}_nb" in [k[0] for k in flow_table.index]
        has_sb = f"{intersection}_sb" in [k[0] for k in flow_table.index]
        has_cross_data = has_nb or has_sb
        phase_specs = build_phase_specs(
            tl, phases, art_idx, cross_idx,
            left_share_art=turning_ratios["left_art"],
            left_share_cross=turning_ratios["left_cross"],
        )

        print(f"\n  {intersection} ({tl_id})")
        print(f"    phases={n_phases}  art={art_idx}  cross={cross_idx}  clear={clear_idx}")
        for spec in phase_specs:
            print(f"      P{spec['phase_index']}: {spec['approach_group']} "
                  f"{spec['movement']} lanes={spec['lanes']} "
                  f"flow_share={spec['flow_share']:.2f}")

        # Generate one program per 15-min interval
        for ts in time_steps:
            sim_min = (ts - TOMTOM_MIN_TIMESET) * 15  # minutes after 07:00
            prog_id = f"ts{ts:02d}"  # e.g. "ts02", "ts03", ...

            # Get approach flows and movement components.
            eb = get_approach_components(
                flow_table, intersection, "eb", ts, default=100,
                left_share=turning_ratios["left_art"])
            wb = get_approach_components(
                flow_table, intersection, "wb", ts, default=100,
                left_share=turning_ratios["left_art"])
            nb = get_approach_components(
                flow_table, intersection, "nb", ts, default=0,
                left_share=turning_ratios["left_cross"])
            sb = get_approach_components(
                flow_table, intersection, "sb", ts, default=0,
                left_share=turning_ratios["left_cross"])

            q_eb = eb["total"]
            q_wb = wb["total"]
            q_nb = nb["total"]
            q_sb = sb["total"]

            q_art   = max(q_eb, q_wb)
            q_cross = max(q_nb, q_sb) if has_cross_data else 50  # fallback
            movement_demands = {
                ("art", "through"): max(eb["nonleft"], wb["nonleft"]),
                ("art", "left"): max(eb["left"], wb["left"]),
                ("cross", "through"): max(nb["nonleft"], sb["nonleft"]),
                ("cross", "left"): max(nb["left"], sb["left"]),
            }

            # Webster timing using estimated left-turn flows and detected lane counts.
            cycle, phase_greens, timed_specs = webster_timing_for_phase_specs(
                q_art, q_cross, L_actual, phase_specs,
                movement_demands=movement_demands
            )

            # Apply green times to the corresponding phases.
            new_durations = apply_phase_greens(phases, phase_greens)

            # Verify cycle (sum of all phases)
            total = round(sum(new_durations), 1)

            # Write tlLogic element
            tl_logic = ET.SubElement(root, "tlLogic",
                                      id=tl_id,
                                      type="static",
                                      programID=prog_id,
                                      offset="0")

            for i, (ph, dur) in enumerate(zip(phases, new_durations)):
                ET.SubElement(tl_logic, "phase",
                               duration=str(round(dur, 1)),
                               state=ph.state)

            if ts == time_steps[0]:
                spec_summary = ", ".join(
                    f"P{s['phase_index']} {s['approach_group']}-{s['movement']}:"
                    f"q={s['q']:.0f},lanes={s['lanes']},y={s['y']:.3f},"
                    f"g={phase_greens.get(s['phase_index'], 0):.1f}"
                    for s in timed_specs
                )
                print(f"    Example ts={ts} (07:00): "
                      f"q_art={q_art:.0f} q_cross={q_cross:.0f} → "
                      f"C={cycle}s  actual_total={total}s")
                print(f"      {spec_summary}")

        # WAUT for automatic switching
        waut_id = f"waut_{tl_id}"
        waut = ET.SubElement(root, "WAUT",
                              startProg=f"ts{time_steps[0]:02d}",
                              id=waut_id,
                              refTime="0")

        for ts in time_steps:
            sim_sec = (ts - TOMTOM_MIN_TIMESET) * 900  # 15-min = 900s
            ET.SubElement(waut, "wautSwitch",
                          time=str(sim_sec),
                          to=f"ts{ts:02d}")

        ET.SubElement(root, "wautJunction",
                      junctionID=tl_id,
                      wautID=waut_id,
                      procedure="stretch")

    # ── Write output ─────────────────────────────────────────────────────────
    ET.indent(root, space="  ")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ET.ElementTree(root).write(args.out, encoding="utf-8", xml_declaration=True)
    print(f"\n✓ Written: {args.out}")
    print(f"  {len(TL_INTERSECTION)} intersections × {len(time_steps)} time steps = "
          f"{len(TL_INTERSECTION) * len(time_steps)} tlLogic programs")


if __name__ == "__main__":
    main()

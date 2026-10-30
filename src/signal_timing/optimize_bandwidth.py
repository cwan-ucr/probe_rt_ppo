"""
Phase 2: GA optimization of cycle length + per-signal offsets for the cleaned
fixed-timing baseline, maximizing the flow-weighted bidirectional green-wave
bandwidth across the 6 corridor signals.

Inputs:
  - data/corridor-osm-network/fixed_timing_60s.add.xml   (Module A output)
  - data/corridor-osm-network/osm.net.xml.gz
  - data/geojson-1-15-min-interval/processed/            (segment speed limits)
  - outputs/calibration/calibration_result_turning.json  (EB/WB flow weights)

Outputs:
  - data/corridor-osm-network/fixed_timing_optimized.add.xml
  - outputs/optimization/bandwidth_result.json

Bandwidth model:
  For one direction d on a corridor with signals i = 0..N-1, given:
    C            cycle length
    g_start_i    when TL i's d-through movement turns green (in local clock)
    g_dur_i      duration of that green phase
    offset_i     when TL i's local clock = 0 in global time
    tau_i        cumulative travel time from reference TL to TL i
  the "feasible departure window" at the reference TL is the cyclic
  intersection over i of arcs of length g_dur_i starting at
  (offset_i + g_start_i - tau_i) mod C. Bandwidth = length of that intersection.

GA: 7 continuous variables [C, off_2, ..., off_7]; offset_0 anchored at 0.
After convergence: round C to nearest multiple of 10, integer offsets, small
local search.
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sumolib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.dirname(SCRIPT_DIR)
ROOT_DIR   = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, SCRIPT_DIR)

from clean_fixed_timing import try_clean, apply as apply_phases  # Module A helpers

NET_FILE       = os.path.join(ROOT_DIR, "data", "corridor-osm-network", "osm.net.xml.gz")
APPROACH_DIR   = os.path.join(ROOT_DIR, "data", "geojson-1-15-min-interval", "processed")
LAYOUT_CSV     = os.path.join(ROOT_DIR, "data", "geojson-1-15-min-interval", "corridor_layout.csv")
DETECTORS_ADD  = os.path.join(ROOT_DIR, "data", "corridor-osm-network", "detectors_v5_with_tl.add.xml")
INPUT_60S      = os.path.join(ROOT_DIR, "data", "corridor-osm-network", "fixed_timing_60s.add.xml")
OUTPUT_FILE    = os.path.join(ROOT_DIR, "data", "corridor-osm-network", "fixed_timing_optimized.add.xml")
RESULT_JSON    = os.path.join(ROOT_DIR, "outputs", "optimization", "bandwidth_result.json")
CAL_JSON       = os.path.join(ROOT_DIR, "outputs", "calibration", "calibration_result_turning.json")

# Order including minor / non-GA-optimized intersections (so travel-time
# accumulation is correct). Mulberry sits just east of Lime.
CORRIDOR_ALL_W2E = [
    "Lime", "Mulberry", "Park", "Victoria", "Eucalyptus", "Kansas",
    "Ottawa", "Chicago", "Cranford", "Iowa",
]
SIGNALIZED_W2E = ["Lime", "Park", "Victoria", "Kansas", "Ottawa", "Chicago", "Iowa"]


# ============================================================================
# Step 1: Corridor edge directions + TL → intersection mapping
# ============================================================================

def build_corridor_edge_dirs(net, road_name="University Avenue"):
    """Return (eb_edges, wb_edges) ID sets for main corridor."""
    eb, wb = set(), set()
    for e in net.getEdges():
        if e.getID().startswith(":"):
            continue
        if e.getName() != road_name:
            continue
        sh = e.getShape()
        if len(sh) < 2:
            continue
        dx = sh[-1][0] - sh[0][0]
        if dx > 1e-6:    eb.add(e.getID())
        elif dx < -1e-6: wb.add(e.getID())
    return eb, wb


def map_tls_to_intersections(net):
    """Return {intersection_name: tl_id} for the 7 signalized corridor intersections.

    Reuses logic from setup_signal_params: parse detectors_v5_with_tl.add.xml for
    segLoop_{segID} → SUMO edge mapping, then for each EB approach's end segment
    find the closest TL junction.
    """
    # Build segID → edgeID from detector file
    seg_to_edge = {}
    if os.path.exists(DETECTORS_ADD):
        tree = ET.parse(DETECTORS_ADD)
        for det in tree.findall(".//inductionLoop"):
            det_id = det.get("id", "")
            if not det_id.startswith("segLoop_"):
                continue
            parts = det_id.split("_")
            if len(parts) < 3:
                continue
            sid = parts[1]
            lane_id = det.get("lane", "")
            eid = "_".join(lane_id.split("_")[:-1])
            seg_to_edge.setdefault(sid, eid)

    # End segment of each EB approach per corridor_layout.csv
    layout = pd.read_csv(LAYOUT_CSV)
    end_seg = {}
    for _, row in layout.iterrows():
        eb_val = row.get("eb", "")
        if not isinstance(eb_val, str) or not eb_val.strip():
            continue
        cleaned = eb_val.strip().strip("()").replace("'", "")
        parts = [p.strip() for p in cleaned.split(",")]
        if len(parts) >= 2:
            end_seg[row["intersection"]] = parts[1]

    # TL junction positions
    tl_pos = {}
    for tl in net.getTrafficLights():
        xs, ys = [], []
        seen = set()
        for from_lane, _to_lane, _li in tl.getConnections():
            jid = from_lane.getEdge().getToNode().getID()
            if jid in seen: continue
            seen.add(jid)
            x, y = net.getNode(jid).getCoord()
            xs.append(x); ys.append(y)
        if xs:
            tl_pos[tl.getID()] = (np.mean(xs), np.mean(ys))

    mapping = {}
    for int_name, seg in end_seg.items():
        eid = seg_to_edge.get(seg)
        if not eid:
            continue
        try:
            tx, ty = net.getEdge(eid).getToNode().getCoord()
        except KeyError:
            continue
        best, best_d = None, float("inf")
        for tl_id, (x, y) in tl_pos.items():
            d = (x - tx) ** 2 + (y - ty) ** 2
            if d < best_d:
                best_d = d; best = tl_id
        if best and best_d ** 0.5 < 200:
            mapping[int_name] = best
    return mapping


# ============================================================================
# Step 2: Identify per-TL EB-through and WB-through link indices
# ============================================================================

def find_through_link_indices(net, eb_edges, wb_edges):
    """For each traffic light in the net, return (eb_link_idx, wb_link_idx) sets
    for through-movement (from-edge and to-edge both on corridor in same direction).
    """
    out = {}
    for tl in net.getTrafficLights():
        eb_idx, wb_idx = set(), set()
        for from_lane, to_lane, link_idx in tl.getConnections():
            f = from_lane.getEdge().getID()
            t = to_lane.getEdge().getID()
            if f in eb_edges and t in eb_edges:
                eb_idx.add(link_idx)
            elif f in wb_edges and t in wb_edges:
                wb_idx.add(link_idx)
        out[tl.getID()] = (eb_idx, wb_idx)
    return out


# ============================================================================
# Step 3: Inter-intersection travel times (segment-level speed limits)
# ============================================================================

def per_segment_travel_time(csv_path):
    if not os.path.exists(csv_path):
        return 0.0
    df = pd.read_csv(csv_path)
    df = df[df["timeSet"] == 2].reset_index(drop=True) if (df["timeSet"] == 2).any() else df
    if df.empty:
        return 0.0
    speed_ms = 0.85 * df["speedLimit"].astype(float) / 3.6 # assume average speed is 85 % of speed limit, converted to m/s
    speed_ms = speed_ms.where(speed_ms > 1e-3, 1e-3)
    return float((df["distance"].astype(float) / speed_ms).sum())


def cumulative_times():
    """Return (cum_eb, cum_wb): dicts mapping signalized intersection → cumulative
    travel time from reference (Lime for EB, Iowa for WB).
    """
    # EB: iterate W→E, accumulate using {name}_eb (excludes Lime, which is pre-corridor)
    cum_eb_all = OrderedDict()
    total = 0.0
    for i, name in enumerate(CORRIDOR_ALL_W2E):
        if i == 0:
            cum_eb_all[name] = 0.0
            continue
        seg_t = per_segment_travel_time(
            os.path.join(APPROACH_DIR, f"{name}_eb_delay_time_results.csv"))
        total += seg_t
        cum_eb_all[name] = total

    # WB: iterate E→W, accumulate using {name}_wb (Iowa is the reference, excluded)
    cum_wb_all = OrderedDict()
    total = 0.0
    for i, name in enumerate(reversed(CORRIDOR_ALL_W2E)):
        if i == 0:
            cum_wb_all[name] = 0.0
            continue
        seg_t = per_segment_travel_time(
            os.path.join(APPROACH_DIR, f"{name}_wb_delay_time_results.csv"))
        total += seg_t
        cum_wb_all[name] = total

    cum_eb = OrderedDict((n, cum_eb_all[n]) for n in SIGNALIZED_W2E if n in cum_eb_all)
    cum_wb = OrderedDict((n, cum_wb_all[n]) for n in SIGNALIZED_W2E if n in cum_wb_all)
    return cum_eb, cum_wb


# ============================================================================
# Step 4: Through-green window extraction
# ============================================================================

def extract_through_window(phases: List[Tuple[str, int]], links: set):
    """Given phase list [(state, duration)] and through-link indices, return
    (green_start_in_cycle, green_duration). If green is split across the cycle
    wrap, returns the longest contiguous run (wrap-aware).
    Yellow on through-link is NOT counted as green.
    """
    cycle = sum(d for _, d in phases)
    if not links or cycle == 0:
        return (0, 0)
    # Build per-second green flag (1s resolution)
    flags = []
    for state, dur in phases:
        ok = all(idx < len(state) and state[idx] in "Gg" for idx in links)
        flags.extend([ok] * int(dur))
    flags = np.array(flags[:cycle], dtype=bool)
    if not flags.any():
        return (0, 0)
    # Find longest contiguous run on a circle
    # Double the array, scan for longest run of True, length <= cycle
    doubled = np.concatenate([flags, flags])
    best_start = 0; best_len = 0
    i = 0
    while i < len(doubled):
        if doubled[i]:
            j = i
            while j < len(doubled) and doubled[j]:
                j += 1
            run_len = min(j - i, cycle)
            if run_len > best_len:
                best_len = run_len
                best_start = i % cycle
            i = j
        else:
            i += 1
    return (best_start, best_len)


# ============================================================================
# Step 5: Bandwidth computation
# ============================================================================

def bandwidth_direction(windows, taus, offsets, C):
    """Return cyclic-intersection length (seconds, integer) of arcs:
       arc_i = [offset_i + g_start_i - tau_i  (mod C),  length = g_dur_i]
    """
    C_int = int(round(C))
    if C_int <= 0:
        return 0
    feasible = np.ones(C_int, dtype=bool)
    for (gs, gd), tau, off in zip(windows, taus, offsets):
        if gd <= 0:
            return 0
        start = int(round(off + gs - tau)) % C_int
        L = min(int(round(gd)), C_int)
        arc = np.zeros(C_int, dtype=bool)
        end = start + L
        if end <= C_int:
            arc[start:end] = True
        else:
            arc[start:] = True
            arc[:end - C_int] = True
        feasible &= arc
        if not feasible.any():
            return 0
    return int(feasible.sum())


# ============================================================================
# Step 6: Baseline phase loading + cycle rescaling
# ============================================================================

def load_baseline_phases(tree, tl_id, program_id=None):
    """Return list[(state, int_duration)] for one program of one TL.

    Strategy:
      1. If program_id is given, try matching that exactly first.
      2. Fall back to the first program found for this tl_id (handles
         dummy / single-program inserts like Ottawa).
    """
    matched = None
    first = None
    for tl in tree.findall(".//tlLogic"):
        if tl.get("id") != tl_id:
            continue
        if first is None:
            first = tl
        if program_id is None or tl.get("programID") == program_id:
            matched = tl
            break
    use = matched if matched is not None else first
    if use is None:
        return []
    return [(p.get("state"), int(round(float(p.get("duration")))))
            for p in use.findall("phase")]


def rescale_phases_to_cycle(phases, target_cycle):
    """Yellow phases fixed; non-yellow scaled to fit target_cycle. Integer durations.
    Returns new list[(state, dur)] or None if infeasible (min-green < 7).
    """
    if not phases:
        return None
    yellows = ["y" in s for s, _ in phases]
    sum_yellow = sum(d for (_, d), y in zip(phases, yellows) if y)
    target_non_yellow = target_cycle - sum_yellow
    sum_non_yellow = sum(d for (_, d), y in zip(phases, yellows) if not y)
    if sum_non_yellow <= 0 or target_non_yellow <= 0:
        return None
    scale = target_non_yellow / sum_non_yellow
    new = []
    for (state, d), y in zip(phases, yellows):
        nd = int(round(d)) if y else int(round(d * scale))
        new.append((state, nd))
    delta = target_cycle - sum(d for _, d in new)
    if delta != 0:
        idxs = [i for i, y in enumerate(yellows) if not y]
        longest = max(idxs, key=lambda i: new[i][1])
        new[longest] = (new[longest][0], new[longest][1] + delta)
    # Min-green check
    for (state, d), y in zip(new, yellows):
        if not y and d < 7:
            return None
    return new


# ============================================================================
# Step 7: Fitness function
# ============================================================================

class BandwidthFitness:
    def __init__(self, tl_order, baseline_phases, link_idx, cum_eb, cum_wb,
                 w_eb, w_wb):
        """
        tl_order: list of TL ids in W→E order
        baseline_phases: {tl_id: [(state, dur_int)]} at 60s
        link_idx: {tl_id: (eb_set, wb_set)}
        cum_eb: list of cumulative EB times in TL order (cum_eb[0]=0)
        cum_wb: list of cumulative WB times measured from the EASTMOST TL
                (cum_wb[-1]=0, cum_wb[0]=largest); we re-orient inside.
        w_eb, w_wb: flow weights (normalized)
        """
        self.tl_order = tl_order
        self.baseline_phases = baseline_phases
        self.link_idx = link_idx
        self.cum_eb = np.array(cum_eb)
        self.cum_wb = np.array(cum_wb)
        self.w_eb = w_eb
        self.w_wb = w_wb

    def _windows(self, C):
        eb_windows, wb_windows = [], []
        for tl_id in self.tl_order:
            base = self.baseline_phases[tl_id]
            rescaled = rescale_phases_to_cycle(base, int(round(C)))
            if rescaled is None:
                return None, None
            eb_links, wb_links = self.link_idx[tl_id]
            eb_windows.append(extract_through_window(rescaled, eb_links))
            wb_windows.append(extract_through_window(rescaled, wb_links))
        return eb_windows, wb_windows

    def evaluate(self, C, offsets):
        """offsets: array of len = N (one per TL); first is fixed to 0 by caller.
        Always returns (score, b_EB_s, b_WB_s)."""
        eb_w, wb_w = self._windows(C)
        if eb_w is None:
            return (-1.0, 0, 0)
        b_eb = bandwidth_direction(eb_w, self.cum_eb, offsets, C)
        b_wb = bandwidth_direction(wb_w, self.cum_wb, offsets, C)
        return (self.w_eb * b_eb + self.w_wb * b_wb, b_eb, b_wb)


# ============================================================================
# Step 8: GA
# ============================================================================

def ga_offsets_at_cycle(fit_obj, C, n_tls, pop_size=60, n_generations=100,
                         elite_frac=0.25, n_seeds=3, verbose=False):
    """Run multi-seed offset-only GA for a fixed cycle C. Returns best
    (score, b_eb, b_wb, offsets array of length n_tls with [0]=0)."""
    n_off = n_tls - 1
    best = (-1.0, 0, 0, None)
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed * 1009 + 42)

        def rand_ind():
            return rng.uniform(0, C, size=n_off)

        def eval_ind(ind):
            offs = np.concatenate([[0.0], ind % C])
            return fit_obj.evaluate(C, offs)

        pop = np.array([rand_ind() for _ in range(pop_size)])
        fits_full = [eval_ind(ind) for ind in pop]
        fits = np.array([f[0] for f in fits_full])

        elite_n = max(2, int(round(elite_frac * pop_size)))
        for g in range(n_generations):
            order = np.argsort(-fits)
            pop, fits = pop[order], fits[order]
            elite = pop[:elite_n]
            offspring = []
            while len(offspring) < pop_size - elite_n:
                i, j = rng.integers(0, elite_n, size=2)
                child = (pop[i] + pop[j]) / 2.0
                if rng.random() < 0.6:
                    child += rng.normal(0, 4.0, size=n_off)
                child = child % C
                offspring.append(child)
            pop = np.vstack([elite, np.array(offspring)])
            fits = np.array([fit_obj.evaluate(C, np.concatenate([[0.0], ind % C]))[0]
                              for ind in pop])

        # Best of this seed
        order = np.argsort(-fits)
        ind_best = pop[order[0]]
        # Local search at integer res
        offs0 = np.concatenate([[0.0], np.round(ind_best).astype(int) % C])
        score, b_eb, b_wb = fit_obj.evaluate(C, offs0)
        improved = True; iters = 0
        deltas = [-5, -3, -2, -1, 1, 2, 3, 5]
        while improved and iters < 20:
            improved = False; iters += 1
            for k in range(1, len(offs0)):
                base = offs0[k]
                for delta in deltas:
                    cand = offs0.copy()
                    cand[k] = int(base + delta) % C
                    s, be, bw = fit_obj.evaluate(C, cand)
                    if s > score:
                        score, b_eb, b_wb = s, be, bw
                        offs0 = cand
                        improved = True
        if score > best[0]:
            best = (score, b_eb, b_wb, offs0.copy())
        if verbose:
            print(f"    seed {seed}: best_score={score:.2f}  b_EB={b_eb}s  b_WB={b_wb}s")
    return best


def optimize_all_cycles(fit_obj, n_tls, c_step=10, c_min=60, c_max=120, **ga_kwargs):
    """Run independent GA for each candidate discrete cycle. Returns best
    (score, C, offsets, b_eb, b_wb)."""
    overall_best = (-1.0, None, None, None, None)
    for C in range(c_min, c_max + 1, c_step):
        print(f"\n--- C={C}s ---")
        score, b_eb, b_wb, offsets = ga_offsets_at_cycle(
            fit_obj, C, n_tls, verbose=True, **ga_kwargs)
        print(f"  C={C}s best: score={score:.2f}  b_EB={b_eb}s  b_WB={b_wb}s")
        if score > overall_best[0]:
            overall_best = (score, C, offsets.tolist(), b_eb, b_wb)
    return overall_best


# ============================================================================
# Step 9: Discretize (round C to nearest 10, integer offsets) + local search
# ============================================================================

def discretize_and_polish(ind, fit_obj, c_step=10):
    """For every multiple of c_step in [60, 120], seed offsets from `ind` and
    run a coordinate-descent local search at integer-second granularity.
    Returns the global best across all candidate cycles."""
    offs_raw = ind[1:]
    c_candidates = list(range(60, 121, c_step))
    best = (-1.0, None, None, None, None)  # score, C, offsets, b_eb, b_wb
    for C in c_candidates:
        offs0 = np.concatenate([[0.0], np.round(offs_raw).astype(int) % C])
        score, b_eb, b_wb = fit_obj.evaluate(C, offs0)
        improved = True
        iters = 0
        # Wider neighborhood (±5) and more iterations
        deltas = [-5, -3, -2, -1, 1, 2, 3, 5]
        while improved and iters < 20:
            improved = False
            iters += 1
            for k in range(1, len(offs0)):
                base = offs0[k]
                for delta in deltas:
                    cand = offs0.copy()
                    cand[k] = int(base + delta) % C
                    s, be, bw = fit_obj.evaluate(C, cand)
                    if s > score:
                        score = s; b_eb = be; b_wb = bw
                        offs0 = cand
                        improved = True
        if score > best[0]:
            best = (score, C, offs0.tolist(), b_eb, b_wb)
    return best


# ============================================================================
# Step 10: Apply optimized params back to fixed_timing XML
# ============================================================================

def apply_to_xml(input_xml, output_xml, tl_to_offset, target_cycle):
    """Rescale all programs of all corridor TLs to target_cycle and set offset.
    Non-corridor tlLogic entries are left untouched."""
    tree = ET.parse(input_xml)
    root = tree.getroot()
    for tl in root.findall(".//tlLogic"):
        tl_id = tl.get("id")
        if tl_id not in tl_to_offset:
            continue
        phases = [(p.get("state"), int(round(float(p.get("duration")))))
                  for p in tl.findall("phase")]
        new = rescale_phases_to_cycle(phases, target_cycle)
        if new is None:
            print(f"  WARN: could not rescale {tl_id} prog={tl.get('programID')} to {target_cycle}s, leaving as-is")
            continue
        for ph_elem, (state, dur) in zip(tl.findall("phase"), new):
            ph_elem.set("duration", str(dur))
        tl.set("offset", str(int(tl_to_offset[tl_id])))
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)


# ============================================================================
# Step 11: Flow weights from calibration result
# ============================================================================

def get_flow_weights(cal_json_path):
    """Sum hourly vph for uni_west_entry_EB and uni_east_entry_WB across the
    calibration window. Returns normalized (w_eb, w_wb)."""
    with open(cal_json_path) as f:
        cal = json.load(f)
    flows = cal["flows"]
    eb_total = sum(flows["uni_west_entry_EB"].values()) if "uni_west_entry_EB" in flows else 0
    wb_total = sum(flows["uni_east_entry_WB"].values()) if "uni_east_entry_WB" in flows else 0
    if eb_total + wb_total == 0:
        return 0.5, 0.5
    w_eb = eb_total / (eb_total + wb_total)
    return w_eb, 1.0 - w_eb


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=INPUT_60S)
    ap.add_argument("--output", default=OUTPUT_FILE)
    ap.add_argument("--pop", type=int, default=80)
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--program", default=None,
                    help="programID to use as the reference for window extraction (default: first program)")
    ap.add_argument("--cycle-min", type=int, default=60)
    ap.add_argument("--cycle-max", type=int, default=120)
    ap.add_argument("--cycle-step", type=int, default=10)
    args = ap.parse_args()

    print(f"Reading network {NET_FILE} ...")
    net = sumolib.net.readNet(NET_FILE, withInternal=True)
    eb_edges, wb_edges = build_corridor_edge_dirs(net)
    print(f"  EB corridor edges: {len(eb_edges)}, WB: {len(wb_edges)}")

    print("Mapping TLs to intersections ...")
    name_to_tl = map_tls_to_intersections(net)
    print(f"  Found: {dict(name_to_tl)}")

    # Drop intersections without a TL discovered
    tl_order = [t for n in SIGNALIZED_W2E for t in [name_to_tl.get(n)] if t]
    ordered_names = [n for n in SIGNALIZED_W2E if n in name_to_tl]
    print(f"  Ordered W→E: {list(zip(ordered_names, tl_order))}")

    print("Finding through-link indices ...")
    link_idx = find_through_link_indices(net, eb_edges, wb_edges)
    for n, t in zip(ordered_names, tl_order):
        eb, wb = link_idx[t]
        print(f"  {n:10s} {t:50s}  EB={sorted(eb)}  WB={sorted(wb)}")

    print("Computing per-direction cumulative travel times ...")
    cum_eb, cum_wb = cumulative_times()
    cum_eb_arr = [cum_eb[n] for n in ordered_names]
    # WB cum is from Iowa to each name (in W→E order, cum_wb decreases as we move E)
    cum_wb_arr = [cum_wb[n] for n in ordered_names]
    for n, e, w in zip(ordered_names, cum_eb_arr, cum_wb_arr):
        print(f"  {n:10s}  τ_EB={e:6.1f}s  τ_WB={w:6.1f}s")

    print("Loading baseline phases (60s cycle) ...")
    tree = ET.parse(args.input)
    if args.program is None:
        # pick first program of first TL as reference
        first_tl = tree.find(f".//tlLogic[@id='{tl_order[0]}']")
        ref_prog = first_tl.get("programID")
    else:
        ref_prog = args.program
    print(f"  Reference program: '{ref_prog}'")
    baseline_phases = {t: load_baseline_phases(tree, t, ref_prog) for t in tl_order}

    # Drop any TL with no phases (e.g., Ottawa not in fixed_timing.add.xml)
    missing = [n for n, t in zip(ordered_names, tl_order) if not baseline_phases[t]]
    if missing:
        print(f"  WARN: dropping TLs without baseline phases: {missing}")
    keep = [(n, t) for n, t in zip(ordered_names, tl_order) if baseline_phases[t]]
    ordered_names = [n for n, _ in keep]
    tl_order = [t for _, t in keep]
    cum_eb_arr = [cum_eb[n] for n in ordered_names]
    cum_wb_arr = [cum_wb[n] for n in ordered_names]
    baseline_phases = {t: baseline_phases[t] for t in tl_order}

    for n, t in zip(ordered_names, tl_order):
        durs = [d for _, d in baseline_phases[t]]
        print(f"  {n:10s} phases: {durs}  sum={sum(durs)}")

    print("Loading EB/WB flow weights ...")
    w_eb, w_wb = get_flow_weights(CAL_JSON)
    print(f"  w_EB={w_eb:.3f}  w_WB={w_wb:.3f}")

    fit = BandwidthFitness(
        tl_order=tl_order,
        baseline_phases=baseline_phases,
        link_idx={t: link_idx[t] for t in tl_order},
        cum_eb=cum_eb_arr,
        cum_wb=cum_wb_arr,
        w_eb=w_eb, w_wb=w_wb,
    )

    # Sanity: bandwidth at baseline (C=60, all offsets=0)
    s0 = fit.evaluate(60, np.zeros(len(tl_order)))
    print(f"\nBaseline (C=60, offsets=0): score={s0[0]:.2f}  b_EB={s0[1]}s  b_WB={s0[2]}s")

    print(f"\nRunning per-cycle GA: pop={args.pop}, gens={args.generations}, seeds=3")
    print(f"  Cycle search: [{args.cycle_min}, {args.cycle_max}] step {args.cycle_step}")
    score, C_final, offsets_final, b_eb, b_wb = optimize_all_cycles(
        fit, n_tls=len(tl_order),
        c_min=args.cycle_min, c_max=args.cycle_max, c_step=args.cycle_step,
        pop_size=args.pop, n_generations=args.generations,
    )
    print(f"\nOverall best: C={C_final}s  offsets={offsets_final}")
    print(f"  b_EB={b_eb}s  b_WB={b_wb}s  weighted_score={score:.2f}")

    # Build TL → offset mapping (excluding any non-corridor TLs in file)
    tl_to_offset = {tl: int(off) for tl, off in zip(tl_order, offsets_final)}

    # Apply to ALL programs of these TLs
    print(f"\nWriting {args.output} ...")
    apply_to_xml(args.input, args.output, tl_to_offset, C_final)

    # Save result JSON
    os.makedirs(os.path.dirname(RESULT_JSON), exist_ok=True)
    result = {
        "input_file": args.input,
        "output_file": args.output,
        "reference_program": ref_prog,
        "cycle_length_s": C_final,
        "flow_weights": {"EB": w_eb, "WB": w_wb},
        "baseline_score_at_C60_offsets0": {
            "score": s0[0], "b_EB_s": s0[1], "b_WB_s": s0[2]},
        "optimized_score": {
            "weighted_score": score, "b_EB_s": b_eb, "b_WB_s": b_wb},
        "intersections": [
            {"name": n, "tl_id": t, "offset_s": int(off),
             "tau_EB_s": round(cum_eb_arr[i], 2),
             "tau_WB_s": round(cum_wb_arr[i], 2)}
            for i, (n, t, off) in enumerate(zip(ordered_names, tl_order, offsets_final))
        ],
    }
    with open(RESULT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Wrote {RESULT_JSON}")


if __name__ == "__main__":
    main()

"""Rewrite baseline signal plans into standard NEMA leading-left order.

Target order per intersection: NS-left, NS-through, WE-left, WE-through.
The mainline (WE) through phase therefore ends the cycle, matching the
coordinated-actuated force-off convention used by the other backbones.

The transformation preserves every movement's green seconds exactly:

* lag-PPLT block ``[ALL(T green, L permissive), trans(T yellow, L green),
  L-protected, L-yellow]`` becomes ``[L-protected(dur+trans dur), ALL,
  ALL-yellow(L-yellow dur)]``.  The left turn keeps its protected+permissive
  green total; the through keeps its green and its clearance.
* protected-lag block ``[T, Y, L, Y]`` (Iowa NS) swaps the (green, yellow)
  pairs to ``[L, Y, T, Y]``.
* single-green blocks are kept intact.
* street blocks are reordered NS first, WE last.

Offsets are compensated independently for every program so the mainline
through green keeps its absolute time window:
``offset_new = (offset_old + p_old - p_new) mod cycle``.  Time-of-day programs
may therefore have different offset attributes when their phase splits differ.

Usage:
  python -m src.signal_timing.make_nema_lead_left           # write *_nema files
  python -m src.signal_timing.make_nema_lead_left --check   # verify only
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NET_DIR = ROOT / "data" / "corridor-osm-network"
NET_FILE = NET_DIR / "osm_cr_tl.net.xml"

TARGETS = [
    NET_DIR / "fixed_timing.add.xml",
    NET_DIR / "fixed_timing_optimized.add.xml",
    NET_DIR / "actuated_signals.add.xml",
]

# Corridor signals to transform (ids present in the fixed baseline). Off-corridor
# signals inside actuated_signals.add.xml keep their existing programs.
CORRIDOR_TLS = None  # filled from fixed_timing.add.xml in main()

if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = (
        "/Library/Frameworks/EclipseSUMO.framework/Versions/Current/"
        "EclipseSUMO/share/sumo")
sys.path.append(str(Path(os.environ["SUMO_HOME"]) / "tools"))
import sumolib  # noqa: E402

import math  # noqa: E402


def _card(lane):
    sh = lane.getShape()
    a = math.degrees(math.atan2(sh[-1][1] - sh[0][1], sh[-1][0] - sh[0][0]))
    return ("EB" if -45 <= a < 45 else "NB" if 45 <= a < 135
            else "WB" if (a >= 135 or a < -135) else "SB")


def build_link_movements(net):
    """{tl_id: {link_index: (approach_card, turn)}} from netconvert topology."""
    mv = {}
    for tl in net.getTrafficLights():
        m = {}
        for inl, outl, i in tl.getConnections():
            turn = "?"
            for c in inl.getOutgoing():
                if c.getToLane() == outl:
                    turn = c.getDirection()
                    break
            turn = {"s": "T", "l": "L", "r": "R", "t": "U",
                    "L": "L", "R": "R"}.get(turn, turn)
            m[i] = (_card(inl), turn)
        mv[tl.getID()] = m
    return mv


def _street(card):
    return "WE" if card in ("EB", "WB") else "NS"


def phase_street(state, moves):
    """Street served by a green phase, judged on T/L movements only.

    Returns None for phases with no through/left green (all-red, right-only):
    those attach to the surrounding block like a clearance."""
    streets = set()
    for i, ch in enumerate(state):
        if ch in "Gg" and i in moves and moves[i][1] in ("T", "L"):
            streets.add(_street(moves[i][0]))
    if len(streets) > 1:
        raise ValueError(f"ambiguous phase street {streets} for state {state}")
    return streets.pop() if streets else None


def is_clearance(state):
    return "y" in state


def yellowize(state):
    return "".join("y" if ch in "Gg" else ch for ch in state)


def left_links(moves, street):
    return {i for i, (c, t) in moves.items()
            if _street(c) == street and t in ("L", "U")}


def split_blocks(phases, moves):
    """[(street, [phase elements...])], clearances attach to the open block."""
    blocks, pending = [], []
    for ph in phases:
        st = ph.get("state")
        street = None if is_clearance(st) else phase_street(st, moves)
        if street is None:
            (blocks[-1][1] if blocks else pending).append(ph)
            continue
        if blocks and blocks[-1][0] == street:
            blocks[-1][1].append(ph)
        else:
            blocks.append((street, [ph]))
            if pending:                      # leading all-red joins block 1
                blocks[-1][1][:0] = pending
                pending = []
    if pending:
        raise ValueError("program has no green phase")
    return blocks


def _greens(block):
    return [p for p in block if not is_clearance(p.get("state"))]


def _dur(ph):
    return int(float(ph.get("duration")))


def _bump(ph, extra):
    """Add ``extra`` seconds to duration and any minDur/maxDur."""
    for key in ("duration", "minDur", "maxDur"):
        if ph.get(key) is not None:
            ph.set(key, str(int(float(ph.get(key))) + extra))


def transform_block(street, block, moves):
    """Return the block's phases in leading-left order (green-preserving)."""
    greens = _greens(block)
    if len(greens) == 1:
        return list(block)
    if len(greens) != 2:
        raise ValueError(f"{len(greens)} green phases in one street block")

    g_all, g_left = greens
    lefts = left_links(moves, street)
    on_left = {i for i, ch in enumerate(g_left.get("state")) if ch in "Gg"}
    if not on_left or not on_left <= lefts | {
            i for i, (c, t) in moves.items() if t == "R"}:
        # Not a lagging-left pattern (e.g. a lead-lag through split): the
        # leading-left flip does not apply, keep the block's internal order.
        return list(block)

    idx = {id(p): k for k, p in enumerate(block)}
    trans = block[idx[id(g_all)] + 1]
    tail = block[idx[id(g_left)] + 1:]
    if not is_clearance(trans.get("state")) or not tail:
        raise ValueError("unexpected block layout")

    trans_has_left_green = any(
        ch in "Gg" and i in lefts
        for i, ch in enumerate(trans.get("state")))

    new_left = copy.deepcopy(g_left)
    new_all = copy.deepcopy(g_all)
    if trans_has_left_green:
        # lag-PPLT: fold the (T-yellow, L-green) transition into the lead left,
        # close the block with a full yellow lasting the old L-yellow duration.
        _bump(new_left, _dur(trans))
        new_yellow = copy.deepcopy(tail[-1])
        new_yellow.set("state", yellowize(g_all.get("state")))
        return [new_left, new_all, new_yellow]
    # protected-lag: swap the (green, yellow) pairs.
    y_all = copy.deepcopy(trans)
    return [new_left] + [copy.deepcopy(p) for p in tail] + [new_all, y_all]


def mainline_start(phases, moves):
    """Cycle position where the WE through green begins."""
    pos = 0
    for ph in phases:
        st = ph.get("state")
        if not is_clearance(st):
            served = {moves[i] for i, ch in enumerate(st)
                      if ch in "Gg" and i in moves}
            if any(_street(c) == "WE" and t == "T" for c, t in served):
                return pos
        pos += _dur(ph)
    return None


def link_green_seconds(phases):
    n = len(phases[0].get("state"))
    tot = [0] * n
    for ph in phases:
        d = _dur(ph)
        for i, ch in enumerate(ph.get("state")):
            if ch in "Gg":
                tot[i] += d
    return tot


def transform_program(tl, moves):
    """Rewrite one tlLogic element in place. Returns a report dict."""
    phases = tl.findall("phase")
    cycle = sum(_dur(p) for p in phases)
    old_start = mainline_start(phases, moves)
    old_green = link_green_seconds(phases)

    blocks = split_blocks(phases, moves)
    if len(blocks) < 2:
        return {"changed": False, "cycle": cycle}
    if len(blocks) > 2:
        raise ValueError(f"{tl.get('id')}: {len(blocks)} street blocks")

    new_blocks = sorted(blocks, key=lambda b: 0 if b[0] == "NS" else 1)
    new_phases = []
    for street, block in new_blocks:
        new_phases.extend(transform_block(street, block, moves))

    for ph in phases:
        tl.remove(ph)
    params = tl.findall("param")
    for p in params:
        tl.remove(p)
    for ph in new_phases:
        tl.append(ph)
    for p in params:
        tl.append(p)

    new_cycle = sum(_dur(p) for p in new_phases)
    if new_cycle != cycle:
        raise ValueError(f"{tl.get('id')}: cycle {cycle} -> {new_cycle}")
    new_green = link_green_seconds(new_phases)
    if new_green != old_green:
        bad = [(i, a, b) for i, (a, b) in enumerate(zip(old_green, new_green))
               if a != b]
        raise ValueError(f"{tl.get('id')}: green seconds changed: {bad}")

    new_start = mainline_start(new_phases, moves)
    delta = None
    if old_start is not None and new_start is not None:
        delta = (old_start - new_start) % cycle
    return {"changed": True, "cycle": cycle, "delta": delta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify existing *_nema files instead of writing")
    args = ap.parse_args()

    net = sumolib.net.readNet(str(NET_FILE))
    movements = build_link_movements(net)
    corridor = {tl.get("id")
                for tl in ET.parse(TARGETS[0]).getroot().iter("tlLogic")}

    for src in TARGETS:
        out = src.with_name(src.stem.replace(".add", "") + "_nema.add.xml")
        tree = ET.parse(src)
        n_changed = n_kept = 0
        for tl in tree.getroot().iter("tlLogic"):
            tid = tl.get("id")
            if tid not in corridor or tid not in movements:
                n_kept += 1
                continue
            rep = transform_program(tl, movements[tid])
            n_changed += 1 if rep["changed"] else 0
            n_kept += 0 if rep["changed"] else 1
            if rep.get("delta") is not None:
                old_off = int(float(tl.get("offset", "0")))
                tl.set(
                    "offset",
                    str((old_off + rep["delta"]) % rep["cycle"]),
                )
        if args.check:
            if not out.exists():
                raise FileNotFoundError(f"missing generated plan: {out}")
            expected = ET.tostring(tree.getroot(), encoding="unicode")
            actual = ET.tostring(
                ET.parse(out).getroot(), encoding="unicode"
            )
            if expected != actual:
                raise ValueError(
                    f"{out.name} does not match the current transformation"
                )
        else:
            tree.write(out, encoding="utf-8", xml_declaration=True)
        print(f"{src.name}: {n_changed} programs reordered, "
              f"{n_kept} kept -> {out.name}"
              f"{' [verified]' if args.check else ''}")


if __name__ == "__main__":
    main()

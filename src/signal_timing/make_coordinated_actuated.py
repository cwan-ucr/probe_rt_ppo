"""
Generate a coordinated-actuated signal plan from the GA-optimized fixed plan.

Idea (per SUMO coordinated-actuated semantics):
  - Lock a common cycle:  param cycleTime=60, param coordinated=true, offset=GA.
  - ROTATE the phase ring so the mainline-through ("coordination") phase is the
    LAST phase, then pin its end to the cycle boundary with
    earliestEnd=latestEnd=cycleTime (cycle clock: timeInCycle=(simTime-offset)%C).
    Because it is last and held to the boundary, it is the slack-absorber: every
    other phase gaps out early and the freed time accrues as extra mainline green
    held until the boundary, so the cycle CANNOT restart early -> no "breathing",
    realized cycle is a hard 60 s. The offset is shifted by the rotation amount so
    the green band stays at the same absolute position (GA wave preserved).
  - Side-street / left-lag phases stay free actuated (minDur<=maxDur=nominal, no
    earliestEnd/latestEnd) so they gap out when no demand. Their maxDur stays at
    the nominal value so the per-cycle budget never overflows 60 s.

Input : data/corridor-osm-network/fixed_timing_optimized.add.xml  (C=60, GA offsets)
The Eucalyptus, Cranford, and Mulberry definitions are supplied by
integrate_minor_signals_60s so they use the same coordinated-actuated
construction as the checked-in signal plan.

Output: data/corridor-osm-network/coordinated_actuated.add.xml
"""
import argparse
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sumolib
import xml.etree.ElementTree as ET
from integrate_minor_signals_60s import MINOR, build_coordinated_tl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NET = os.path.join(ROOT, "data", "corridor-osm-network", "osm_cr_tl.net.xml")
OPT = os.path.join(ROOT, "data", "corridor-osm-network", "fixed_timing_optimized.add.xml")
OUT = os.path.join(ROOT, "data", "corridor-osm-network", "coordinated_actuated.add.xml")

CYCLE = 60
EXT = 10          # how many sec the mainline phase may extend past its nominal band end
SIDE_MIN = 5      # min green for actuated side / left-lag phases

MAIN_IDS = ["54553323", "54424419", "GS_cluster_5910940017_5910976489",
            "GS_cluster_54287971_5910976488", "GS_cluster_5910940035_5910945467",
            "GS_cluster_4689533026_54483626_5910945475_5910945476",
            "GS_cluster_5910933593_9013582988"]
DIRMAP = {"s": "T", "l": "L", "r": "R", "t": "U", "L": "L", "R": "R"}


def card(lane):
    sh = lane.getShape()
    a = math.degrees(math.atan2(sh[-1][1] - sh[0][1], sh[-1][0] - sh[0][0]))
    return "EB" if -45 <= a < 45 else "NB" if 45 <= a < 135 else "WB" if (a >= 135 or a < -135) else "SB"


def link_moves(net):
    mv = {}
    for tl in net.getTrafficLights():
        m = {}
        for inl, outl, i in tl.getConnections():
            dd = "?"
            for c in inl.getOutgoing():
                if c.getToLane() == outl:
                    dd = DIRMAP.get(c.getDirection(), c.getDirection())
                    break
            m[i] = f"{card(inl)}-{dd}"
        mv[tl.getID()] = m
    return mv


def is_yellow(state):
    return "y" in state and "G" not in state and "g" not in state


def find_main_phase(phases, m):
    """Index of first phase whose green links include a mainline through (EB-T/WB-T)."""
    for i, p in enumerate(phases):
        gr = {m.get(j, "") for j, c in enumerate(p.get("state")) if c in "Gg"}
        if "EB-T" in gr or "WB-T" in gr:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=OPT,
                    help="fixed plan to rotate (default: GA-optimized fixed)")
    ap.add_argument("--output", default=OUT)
    args = ap.parse_args()

    net = sumolib.net.readNet(NET, withInternal=True)
    mv = link_moves(net)
    tree = ET.parse(args.input)

    # first program per main id
    first = {}
    for tl in tree.findall(".//tlLogic"):
        if tl.get("id") in MAIN_IDS and tl.get("id") not in first:
            first[tl.get("id")] = tl
    root = ET.Element("additional")
    summary = []
    for tid in MAIN_IDS:
        src = first[tid]
        m = mv[tid]
        phases = src.findall("phase")
        durs = [int(round(float(p.get("duration")))) for p in phases]
        states = [p.get("state") for p in phases]
        n = len(phases)
        main_i = find_main_phase(phases, m)
        band_dur = durs[main_i]
        band_end = sum(durs[:main_i + 1])         # old cycle pos where main green ends
        old_off = int(round(float(src.get("offset"))))

        # rotate the ring so the coordinated phase is LAST: new order starts right
        # after it (phases[main_i+1 ..], then wraps to [.. main_i]); its trailing
        # clearance becomes the first phase of the next cycle (clearance preserved).
        s = (main_i + 1) % n
        order = list(range(s, n)) + list(range(0, s))     # old indices; order[-1]==main_i
        new_off = (old_off + band_end) % CYCLE            # keep band's absolute position

        tl = ET.SubElement(root, "tlLogic", {
            "id": tid, "type": "actuated", "programID": "coord_act",
            "offset": str(new_off)})
        ET.SubElement(tl, "param", {"key": "coordinated", "value": "true"})
        ET.SubElement(tl, "param", {"key": "cycleTime", "value": str(CYCLE)})

        for j in order:
            st = states[j]
            d = durs[j]
            ph = {"duration": str(d), "state": st}
            if j == main_i:
                # coordinated mainline phase, now LAST: pin its end to the cycle
                # boundary so it holds all slack -> hard 60 s cycle, no breathing.
                ph["minDur"] = str(band_dur)
                ph["maxDur"] = str(CYCLE)
                ph["earliestEnd"] = str(CYCLE)
                ph["latestEnd"] = str(CYCLE)
            elif is_yellow(st):
                ph["minDur"] = str(d); ph["maxDur"] = str(d)
            else:
                # free actuated side / left-lag: gap out; maxDur stays nominal so the
                # per-cycle budget never exceeds 60 s (mainline absorbs the rest).
                ph["minDur"] = str(min(SIDE_MIN, d))
                ph["maxDur"] = str(d)
            ET.SubElement(tl, "phase", ph)
        summary.append((tid, main_i, band_dur, band_end, old_off, new_off))

    # Eucalyptus/Cranford stay fixed within an actuated-coordinated shell;
    # Mulberry keeps its 60 s template and real actuated min/max bounds.
    for tid, (offset, phases) in MINOR.items():
        root.append(build_coordinated_tl(tid, offset, phases))

    ET.indent(tree := ET.ElementTree(root), space="  ")
    tree.write(args.output, encoding="utf-8", xml_declaration=True)

    print(f"Wrote {args.output}")
    print(f"  {'tl':9s} main_phase band_dur  old_off -> new_off  (coord phase pinned last @ {CYCLE}s)")
    lab = {"54553323": "Lime", "54424419": "Park",
           "GS_cluster_5910940017_5910976489": "Victoria",
           "GS_cluster_54287971_5910976488": "Kansas",
           "GS_cluster_5910940035_5910945467": "Ottawa",
           "GS_cluster_4689533026_54483626_5910945475_5910945476": "Chicago",
           "GS_cluster_5910933593_9013582988": "Iowa"}
    for tid, mi, bd, _be, oo, no in summary:
        print(f"  {lab[tid]:9s} phase{mi:<2d}     {bd:>3d}s      {oo:>3d} -> {no:>3d}")


if __name__ == "__main__":
    main()

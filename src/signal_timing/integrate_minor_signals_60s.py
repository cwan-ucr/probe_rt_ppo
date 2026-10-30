"""
Put the corridor's minor signals on the common 60s cycle with a travel-time
offset, so the whole corridor shares one metronome:

  - Eucalyptus EB orphan (5910940029)  : main EB green, 90% green
  - Eucalyptus WB orphan (5910975175)  : main WB green, 90% green
  - Cranford            (4685892089)   : both-through, 90% green
  - Mulberry            (joinedS_...)   : 42s main + 12s cross green

Fixed signal files keep static programs.  In actuated_signals.add.xml all four
controllers become local actuated programs.  In coordinated_actuated.add.xml
they become coordinated-actuated programs.  Mulberry retains the states and
nominal 42/3/12/3 timing with the original actuated bounds (main 15..50s,
cross 7..30s).  Every coordinated green is rotated to the final phase and its
offset is shifted by the same rotation so the absolute green window is
unchanged.
"""
import os
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NETDIR = os.path.join(ROOT, "data", "corridor-osm-network")

FILES = ["fixed_timing.add.xml", "fixed_timing_60s.add.xml",
         "fixed_timing_optimized.add.xml", "coordinated_actuated.add.xml",
         "actuated_signals.add.xml"]

# id -> (offset, [(state, dur), ...])  60s.
# Eucalyptus/Cranford: no cross street -> 90% main green.
# Mulberry: now has NB (Mulberry St) demand -> real main + NB-cross phases.
G, Y, R = 54, 3, 3
MINOR = {
    "5910940029":                 (20, [("GG", G), ("yy", Y), ("rr", R)]),            # Eucalyptus EB
    "5910975175":                 (20, [("GG", G), ("yy", Y), ("rr", R)]),            # Eucalyptus WB
    "4685892089":                 (7,  [("GGGGGG", G), ("yyyyyy", Y), ("rrrrrr", R)]),# Cranford
    "joinedS_12329405771_54512111": (8, [("rrrgGGrrrrGGgg", 42),   # main: EB/WB through+turns green, NB red
                                          ("rrryyyrrrryyyy", 3),    # main yellow
                                          ("gGGrrrGGGGrrrr", 12),   # cross: NB merge; link 0 yields
                                          ("yyyrrryyyyrrrr", 3)]),  # cross yellow
}
PROG = "minor60"
CYCLE = 60
COORD_FILE = "coordinated_actuated.add.xml"
ACTUATED_FILE = "actuated_signals.add.xml"

# Bounds are indexed in the original MINOR phase order.  Mulberry's values come
# from its actuated controller parameters; the other controllers are fixed.
ACTUATED_BOUNDS = {
    "5910940029": [(54, 54), (3, 3), (3, 3)],
    "5910975175": [(54, 54), (3, 3), (3, 3)],
    "4685892089": [(54, 54), (3, 3), (3, 3)],
    "joinedS_12329405771_54512111": [(15, 50), (3, 3), (7, 30), (3, 3)],
}
ACTUATED_PARAMS = {
    # This approach lane serves both a yielding right turn (g) and priority
    # through movement (G). SUMO permits its detector for actuation when jammed
    # calls are bounded, instead of requiring both movements to be priority G.
    "joinedS_12329405771_54512111": {"jam-threshold": "30"},
}

# Bounds used only by the local, non-coordinated actuated file.  With no
# competing vehicle phase, Eucalyptus and Cranford retain the original 54-s
# maximum but may gap out after a 3-s minimum.
LOCAL_ACTUATED_BOUNDS = {
    "5910940029": [(3, 54), (3, 3), (3, 3)],
    "5910975175": [(3, 54), (3, 3), (3, 3)],
    "4685892089": [(3, 54), (3, 3), (3, 3)],
    "joinedS_12329405771_54512111": [(15, 50), (3, 3), (7, 30), (3, 3)],
}


def build_tl(tid, offset, phases):
    tl = ET.Element("tlLogic", {"id": tid, "type": "static",
                                "programID": PROG, "offset": str(offset)})
    for state, dur in phases:
        ET.SubElement(tl, "phase", {"duration": str(dur), "state": state})
    return tl


def build_coordinated_tl(tid, old_offset, phases):
    """Rotate phase 0 (the coordinated through green) to the end of the ring."""
    main_idx = 0
    band_end = sum(dur for _, dur in phases[:main_idx + 1])
    offset = (old_offset + band_end) % CYCLE
    order = list(range(main_idx + 1, len(phases))) + list(range(main_idx + 1))

    tl = ET.Element("tlLogic", {
        "id": tid, "type": "actuated", "programID": "coord_act",
        "offset": str(offset),
    })
    ET.SubElement(tl, "param", {"key": "coordinated", "value": "true"})
    ET.SubElement(tl, "param", {"key": "cycleTime", "value": str(CYCLE)})
    for key, value in ACTUATED_PARAMS.get(tid, {}).items():
        ET.SubElement(tl, "param", {"key": key, "value": value})

    bounds = ACTUATED_BOUNDS[tid]
    for idx in order:
        state, duration = phases[idx]
        min_dur, max_dur = bounds[idx]
        attrs = {
            "duration": str(duration),
            "state": state,
            "minDur": str(min_dur),
            "maxDur": str(max_dur),
        }
        if idx == main_idx:
            attrs["earliestEnd"] = str(CYCLE)
            attrs["latestEnd"] = str(CYCLE)
        ET.SubElement(tl, "phase", attrs)
    return tl


def build_local_actuated_tl(tid, phases):
    """Build an uncoordinated detector-responsive program in original order."""
    tl = ET.Element("tlLogic", {
        "id": tid, "type": "actuated", "programID": "actuated_local",
        "offset": "0",
    })
    ET.SubElement(tl, "param", {"key": "detector-gap", "value": "3"})
    ET.SubElement(tl, "param", {"key": "build-all-detectors", "value": "true"})
    for key, value in ACTUATED_PARAMS.get(tid, {}).items():
        ET.SubElement(tl, "param", {"key": key, "value": value})
    for (state, duration), (min_dur, max_dur) in zip(
        phases, LOCAL_ACTUATED_BOUNDS[tid]
    ):
        ET.SubElement(tl, "phase", {
            "duration": str(duration), "state": state,
            "minDur": str(min_dur), "maxDur": str(max_dur),
        })
    return tl


def main():
    for fn in FILES:
        p = os.path.join(NETDIR, fn)
        if not os.path.exists(p):
            print(f"  skip (missing) {fn}"); continue
        tree = ET.parse(p)
        root = tree.getroot()
        # remove existing tlLogic for these ids
        removed = 0
        for tl in list(root.findall("tlLogic")):
            if tl.get("id") in MINOR:
                root.remove(tl); removed += 1
        # append fresh 60s versions
        for tid, (off, phases) in MINOR.items():
            if fn == COORD_FILE:
                root.append(build_coordinated_tl(tid, off, phases))
            elif fn == ACTUATED_FILE:
                root.append(build_local_actuated_tl(tid, phases))
            else:
                root.append(build_tl(tid, off, phases))
        ET.indent(tree, space="  ")
        tree.write(p, encoding="UTF-8", xml_declaration=True)
        mode = (
            "coordinated-actuated" if fn == COORD_FILE else
            "local-actuated" if fn == ACTUATED_FILE else
            "static"
        )
        print(f"  {fn}: removed {removed} old, added {len(MINOR)} ({mode}, 60s)")


if __name__ == "__main__":
    main()

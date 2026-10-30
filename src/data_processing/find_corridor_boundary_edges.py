"""
Identify SUMO boundary edges for flow-based demand generation.

Strategy: For each approach, find the FIRST (farthest from intersection) segment,
get its SUMO edge from the detector mapping, then trace UPSTREAM to find the edge
that has no valid predecessor inside the corridor (i.e., the network entry).

Outputs:
  outputs/calibration/corridor_boundary_edges.json

Structure:
  {
    "arterial": {
        "uni_west_entry_EB": {"edge": "...", "approach": "Lime_eb", "first_seg": "466"},
        "uni_east_entry_WB": {"edge": "...", "approach": "Iowa_wb", "first_seg": "661"},
        ...
    },
    "cross": {
        "Iowa": {
            "south_entry_NB": {...},    # cars from south going north
            "north_entry_SB": {...}     # cars from north going south
        },
        ...
    }
  }
"""

import os
import sys
import json
import pandas as pd
import xml.etree.ElementTree as ET

if "SUMO_HOME" not in os.environ:
    # Try known macOS install
    candidate = "/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/share/sumo"
    if os.path.isdir(candidate):
        os.environ["SUMO_HOME"] = candidate
    else:
        raise EnvironmentError("SUMO_HOME not set")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import sumolib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)

NET_FILE = os.path.join(ROOT_DIR, "data", "corridor-osm-network", "osm.net.xml.gz")
DETECTOR_XML = os.path.join(ROOT_DIR, "data", "corridor-osm-network",
                             "detectors_v5_with_tl.add.xml")
LAYOUT_FILE = os.path.join(ROOT_DIR, "data", "geojson-1-15-min-interval", "corridor_layout.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "calibration")


def build_segment_to_edge(detector_xml):
    """Parse detectors additional file to build segmentID → SUMO edge mapping.

    Detector IDs follow pattern: segLoop_{segID}_{edgeID}_{laneIdx} or
                                  segLoop_{segID}_{numericSuffix}.
    We extract the lane attribute and strip the _laneIdx suffix to get the edge.
    """
    tree = ET.parse(detector_xml)
    root = tree.getroot()

    seg_to_edge = {}
    for det in root.findall(".//inductionLoop"):
        det_id = det.get("id", "")
        if not det_id.startswith("segLoop_"):
            continue
        parts = det_id.split("_")
        if len(parts) < 2:
            continue
        seg_id = parts[1]
        lane = det.get("lane", "")
        if not lane:
            continue
        # Strip trailing "_{laneIdx}" from lane to get edge ID
        if "_" in lane:
            edge = "_".join(lane.split("_")[:-1])
        else:
            edge = lane
        if seg_id not in seg_to_edge:
            seg_to_edge[seg_id] = edge
    return seg_to_edge


def get_first_segments(layout_csv):
    """Parse corridor_layout.csv → dict {approach_id: first_seg_id}.
    'first' = farthest from intersection = upstream entry to the approach.
    """
    layout = pd.read_csv(layout_csv)
    result = {}
    for _, row in layout.iterrows():
        name = row["intersection"]
        for direction in ["eb", "wb", "nb", "sb"]:
            val = row.get(direction)
            if pd.isna(val) or not str(val).strip():
                continue
            cleaned = str(val).replace("(", "").replace(")", "").replace("'", "")
            parts = [p.strip() for p in cleaned.split(",")]
            if len(parts) >= 2:
                result[f"{name}_{direction}"] = parts[0]  # first = farthest
    return result


def get_approach_segment_list(approach_folder, approach_id):
    """Return ordered list of simpleIDs for approach (farthest → nearest)."""
    fpath = os.path.join(approach_folder, f"{approach_id}_delay_time_results.csv")
    if not os.path.exists(fpath):
        return []
    df = pd.read_csv(fpath)
    min_ts = df["timeSet"].min()
    df_t = df[df["timeSet"] == min_ts].reset_index(drop=True)
    return df_t["simpleID"].astype(str).tolist()


def find_earliest_mapped_edge(segment_list, seg_to_edge):
    """Walk segment list (farthest first) to find first segment with SUMO edge.
    Returns (segment_id, edge_id) or (None, None).
    """
    for sid in segment_list:
        edge = seg_to_edge.get(str(sid))
        if edge:
            return str(sid), edge
    return None, None


def find_latest_mapped_edge(segment_list, seg_to_edge):
    """Walk segment list in reverse (nearest first) to find last segment with edge."""
    for sid in reversed(segment_list):
        edge = seg_to_edge.get(str(sid))
        if edge:
            return str(sid), edge
    return None, None


def collect_approach_edges(segment_list, seg_to_edge):
    """Return ordered, globally-deduplicated list of SUMO edges along the approach.

    Segments are ordered farthest → nearest to intersection. Multiple segments
    often share the same SUMO edge; we keep only the FIRST occurrence of each
    edge to ensure a monotonic path (no back-and-forth).
    """
    seen = set()
    edges = []
    for sid in segment_list:
        edge = seg_to_edge.get(str(sid))
        if not edge:
            continue
        if edge in seen:
            continue
        seen.add(edge)
        edges.append(edge)
    return edges


def find_upstream_entry(net, edge_id, max_hops=5):
    """From the given edge, trace upstream until reaching an edge with no predecessor
    or max_hops reached.

    Returns: (entry_edge_id, n_hops) or (edge_id, 0) if already terminal.
    """
    try:
        edge = net.getEdge(edge_id)
    except KeyError:
        return edge_id, -1

    current = edge
    for hops in range(max_hops):
        from_node = current.getFromNode()
        incoming = from_node.getIncoming()
        # Filter out internal/connector edges
        valid = [e for e in incoming if not e.getID().startswith(":") and e.allows("passenger")]
        if not valid:
            return current.getID(), hops
        # Pick longest incoming as "main" predecessor
        valid.sort(key=lambda e: e.getLength(), reverse=True)
        current = valid[0]
    return current.getID(), max_hops


def find_downstream_exit(net, edge_id, max_hops=5):
    """From the given edge, trace downstream until reaching an edge with no successor.

    Returns: (exit_edge_id, n_hops)
    """
    try:
        edge = net.getEdge(edge_id)
    except KeyError:
        return edge_id, -1

    current = edge
    for hops in range(max_hops):
        to_node = current.getToNode()
        outgoing = to_node.getOutgoing()
        valid = [e for e in outgoing if not e.getID().startswith(":") and e.allows("passenger")]
        if not valid:
            return current.getID(), hops
        valid.sort(key=lambda e: e.getLength(), reverse=True)
        current = valid[0]
    return current.getID(), max_hops


def get_sumo_fringes(net):
    """Get SUMO's automatic fringe edges (entry/exit)."""
    entries, exits = set(), set()
    for edge in net.getEdges():
        if edge.getID().startswith(":"):
            continue
        if not edge.allows("passenger"):
            continue
        from_node = edge.getFromNode()
        to_node = edge.getToNode()
        # Entry: from_node has no valid incoming
        fn_in = [e for e in from_node.getIncoming()
                 if not e.getID().startswith(":") and e.allows("passenger")]
        if not fn_in:
            entries.add(edge.getID())
        # Exit: to_node has no valid outgoing
        tn_out = [e for e in to_node.getOutgoing()
                  if not e.getID().startswith(":") and e.allows("passenger")]
        if not tn_out:
            exits.add(edge.getID())
    return entries, exits


def main():
    print("=" * 60)
    print("  Corridor Boundary Edge Identifier")
    print("=" * 60)

    net = sumolib.net.readNet(NET_FILE)
    print(f"\nNetwork: {NET_FILE}")
    print(f"  Total edges: {len(net.getEdges())}")

    seg_to_edge = build_segment_to_edge(DETECTOR_XML)
    print(f"  SegmentID → SUMO edge mapping: {len(seg_to_edge)}")

    first_segs = get_first_segments(LAYOUT_FILE)
    print(f"  Approaches: {len(first_segs)}")

    sumo_entries, sumo_exits = get_sumo_fringes(net)
    print(f"  SUMO auto-detected fringe: {len(sumo_entries)} entries, {len(sumo_exits)} exits")

    # For each approach, get the full segment list
    approach_folder = os.path.join(ROOT_DIR, "data", "geojson-1-15-min-interval", "processed")
    approach_segs = {}
    for app_id in first_segs:
        approach_segs[app_id] = get_approach_segment_list(approach_folder, app_id)

    result = {"arterial": {}, "cross": {}}

    # === Arterial boundaries (University Ave) ===
    # West entry for EB traffic ← Lime_eb (westernmost) first segment
    # East entry for WB traffic ← Iowa_wb (easternmost) first segment
    print("\n--- Arterial (University Ave) boundaries ---")

    arterial_map = {
        "uni_west_entry_EB": ("Lime_eb", "upstream", "entry"),
        "uni_east_entry_WB": ("Iowa_wb", "upstream", "entry"),
        # For exits, use the LAST segment of the exit approaches
        # Actually: for EB traffic to exit, it continues east past Iowa → Iowa_eb last segment's downstream
        # For WB traffic to exit, it continues west past Lime → Lime_wb last segment's downstream
    }

    for key, (approach, _direction, _role) in arterial_map.items():
        seg_list = approach_segs.get(approach, [])
        if not seg_list:
            print(f"  {key}: approach '{approach}' not found")
            continue
        # Find earliest segment (farthest from intersection) that has a detector
        seg_id, edge_id = find_earliest_mapped_edge(seg_list, seg_to_edge)
        if not edge_id:
            print(f"  {key}: no segment in '{approach}' has detector mapping")
            continue
        # Use the segment's edge directly as spawn point (see note below).
        entry_edge = edge_id
        approach_edges = collect_approach_edges(seg_list, seg_to_edge)
        result["arterial"][key] = {
            "edge": entry_edge,
            "approach": approach,
            "first_mapped_seg": seg_id,
            "seg_edge": edge_id,
            "upstream_hops": 0,
            "is_sumo_fringe": entry_edge in sumo_entries,
            "approach_edges": approach_edges,  # full ordered edge sequence (for via=)
        }
        print(f"  {key}: seg={seg_id} → edge={edge_id} "
              f"({len(approach_edges)} edges in approach)")

    # Arterial exits: trace downstream from LAST segment
    # We need last_segs for this
    layout_df = pd.read_csv(LAYOUT_FILE)
    last_segs = {}
    for _, row in layout_df.iterrows():
        name = row["intersection"]
        for direction in ["eb", "wb", "nb", "sb"]:
            val = row.get(direction)
            if pd.isna(val) or not str(val).strip():
                continue
            cleaned = str(val).replace("(", "").replace(")", "").replace("'", "")
            parts = [p.strip() for p in cleaned.split(",")]
            if len(parts) >= 2:
                last_segs[f"{name}_{direction}"] = parts[-1]

    arterial_exit_map = {
        "uni_east_exit_EB": "Iowa_eb",   # EB traffic exits east past Iowa
        "uni_west_exit_WB": "Lime_wb",   # WB traffic exits west past Lime
    }
    for key, approach in arterial_exit_map.items():
        seg_list = approach_segs.get(approach, [])
        if not seg_list:
            continue
        seg_id, edge_id = find_latest_mapped_edge(seg_list, seg_to_edge)
        if not edge_id:
            print(f"  {key}: no segment in '{approach}' has detector mapping")
            continue
        # Use the segment's edge directly as exit point (see note on entries).
        exit_edge = edge_id
        approach_edges = collect_approach_edges(seg_list, seg_to_edge)
        result["arterial"][key] = {
            "edge": exit_edge,
            "approach": approach,
            "last_mapped_seg": seg_id,
            "seg_edge": edge_id,
            "downstream_hops": 0,
            "is_sumo_fringe": exit_edge in sumo_exits,
            "approach_edges": approach_edges,
        }
        print(f"  {key}: seg={seg_id} → edge={edge_id}")

    # === Cross street boundaries ===
    print("\n--- Cross street boundaries ---")
    cross_streets = ["Iowa", "Chicago", "Kansas", "Park", "Lime", "Victoria"]

    for cross in cross_streets:
        result["cross"][cross] = {}

        for direction_suffix, entry_key, exit_key in [
            ("nb", "south_entry_NB", "north_exit_NB"),
            ("sb", "north_entry_SB", "south_exit_SB"),
        ]:
            approach = f"{cross}_{direction_suffix}"
            seg_list = approach_segs.get(approach, [])
            if not seg_list:
                continue

            # Entry: use earliest mapped segment's edge directly (no upstream trace).
            # Upstream tracing can find edges on different streets, causing the
            # router to bypass the approach's detectors.
            approach_edges = collect_approach_edges(seg_list, seg_to_edge)
            seg_id, edge_id = find_earliest_mapped_edge(seg_list, seg_to_edge)
            if edge_id:
                result["cross"][cross][entry_key] = {
                    "edge": edge_id,
                    "approach": approach,
                    "first_mapped_seg": seg_id,
                    "seg_edge": edge_id,
                    "upstream_hops": 0,
                    "is_sumo_fringe": edge_id in sumo_entries,
                    "approach_edges": approach_edges,
                }
                print(f"  {cross} {entry_key}: seg={seg_id} → {edge_id} "
                      f"({len(approach_edges)} edges in approach)")

            # Exit: use latest mapped segment's edge directly
            seg_id2, edge_id2 = find_latest_mapped_edge(seg_list, seg_to_edge)
            if edge_id2:
                result["cross"][cross][exit_key] = {
                    "edge": edge_id2,
                    "approach": approach,
                    "last_mapped_seg": seg_id2,
                    "seg_edge": edge_id2,
                    "downstream_hops": 0,
                    "is_sumo_fringe": edge_id2 in sumo_exits,
                    "approach_edges": approach_edges,
                }

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "corridor_boundary_edges.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Saved: {out_path}")

    # Summary
    n_art = len(result["arterial"])
    n_cross = sum(len(v) for v in result["cross"].values())
    print(f"\nArterial boundaries: {n_art}")
    print(f"Cross street boundaries: {n_cross}")
    print(f"Total: {n_art + n_cross} boundary points")


if __name__ == "__main__":
    main()

import collections
import csv
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mplconfig")
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = os.path.join(tempfile.gettempdir(), "xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sumolib
import traci


Point = Tuple[float, float]

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DETECTOR_PLACEMENT_LOG = os.path.join(
    ROOT_DIR, "outputs", "calibration", "detector_placement_log.csv"
)
DEFAULT_STOP_BAR_CHOICES = os.path.join(
    ROOT_DIR, "outputs", "calibration", "stop_bar_chosen.json"
)

SIGNAL_LABEL_OVERRIDES = {
    "5910940029": "Eucalyptus",
    "5910975175": "Eucalyptus",
    "4685892089": "Cranford",
}


@dataclass
class SignalInfo:
    tl_id: str
    distance_m: float
    link_indices: List[int]
    label: str


@dataclass
class LandmarkInfo:
    distance_m: float
    label: str


@dataclass
class CorridorDirection:
    key: str
    title: str
    edge_ids: List[str]
    edge_set: set
    edge_base_m: Dict[str, float]
    total_length_m: float
    signal_infos: List[SignalInfo]
    landmark_infos: List[LandmarkInfo]
    plot_min_m: float = 0.0
    plot_max_m: Optional[float] = None


class MainlineTimeSpaceRecorder:
    def __init__(
        self,
        sumocfg_path: str,
        output_dir: str,
        road_name: str = "University Avenue",
        sample_interval_s: float = 1.0,
    ) -> None:
        self.sumocfg_path = sumocfg_path
        self.output_dir = output_dir
        self.road_name = road_name
        self.sample_interval_s = max(sample_interval_s, 0.1)
        self.next_sample_time = 0.0

        self.net_path = resolve_net_file_from_sumocfg(sumocfg_path)
        self.net = sumolib.net.readNet(self.net_path, withInternal=False)
        self.junction_cross_streets = build_junction_cross_street_lookup(self.net, self.road_name)
        self.directions = build_corridor_directions(
            self.net, self.road_name, self.junction_cross_streets)

        self.trajectories = {
            direction.key: collections.defaultdict(lambda: {"time_s": [], "distance_m": []})
            for direction in self.directions.values()
        }
        self.signal_history = {
            direction.key: {
                signal.tl_id: {"time_s": [], "is_green": [], "distance_m": signal.distance_m, "label": signal.label}
                for signal in direction.signal_infos
            }
            for direction in self.directions.values()
        }

        # Lookup tables for runtime recording
        self.edge_to_direction: Dict[str, str] = {}
        self.edge_end_m: Dict[str, float] = {}
        for direction in self.directions.values():
            for edge_id in direction.edge_ids:
                self.edge_to_direction[edge_id] = direction.key
                try:
                    e = self.net.getEdge(edge_id)
                    self.edge_end_m[edge_id] = direction.edge_base_m[edge_id] + e.getLength()
                except KeyError:
                    self.edge_end_m[edge_id] = direction.edge_base_m[edge_id]

        # Per-vehicle state used to interpolate through internal (`:`) lanes when
        # a vehicle is blocked inside an intersection (common under fixed-time
        # signals with long red queues). Holds the last corridor (direction_key,
        # base_m_at_edge_end) so internal-lane lane_pos can be added on top.
        self.veh_last_state: Dict[str, Tuple[str, float]] = {}

    def record_step(self, sim_time: float) -> None:
        if sim_time + 1e-9 < self.next_sample_time:
            return
        self.next_sample_time = sim_time + self.sample_interval_s

        # Single pass over all currently active vehicles. This catches vehicles
        # on corridor edges (record directly) AND vehicles stuck on internal
        # junction lanes between two corridor edges (record via last-state
        # interpolation).
        for veh_id in traci.vehicle.getIDList():
            road_id = traci.vehicle.getRoadID(veh_id)
            if not road_id:
                continue

            if road_id.startswith(":"):
                state = self.veh_last_state.get(veh_id)
                if state is None:
                    continue  # vehicle never touched the corridor yet
                dir_key, base_m = state
                # Pin to the upstream edge's downstream end. Adding lane_pos
                # here would overshoot, because edge_base_m only cumulates
                # corridor edge lengths and excludes internal lane lengths —
                # the next corridor edge starts back at exactly `base_m`, so
                # advancing through the internal lane would cause a backward
                # jump on re-entry. Holding the distance constant draws the
                # queued vehicle as a horizontal line at the intersection.
                self.trajectories[dir_key][veh_id]["time_s"].append(sim_time)
                self.trajectories[dir_key][veh_id]["distance_m"].append(base_m)
            else:
                dir_key = self.edge_to_direction.get(road_id)
                if dir_key is None:
                    continue  # off-corridor edge
                direction = self.directions[dir_key]
                lane_pos = traci.vehicle.getLanePosition(veh_id)
                self.trajectories[dir_key][veh_id]["time_s"].append(sim_time)
                self.trajectories[dir_key][veh_id]["distance_m"].append(
                    direction.edge_base_m[road_id] + lane_pos)
                # Save state so the next sample on an internal lane continues
                # from this edge's downstream end.
                self.veh_last_state[veh_id] = (dir_key, self.edge_end_m[road_id])

        # Drop state for vehicles that finished this step to keep the dict bounded.
        for vid in traci.simulation.getArrivedIDList():
            self.veh_last_state.pop(vid, None)

        for direction in self.directions.values():
            for signal in direction.signal_infos:
                state = traci.trafficlight.getRedYellowGreenState(signal.tl_id)
                is_green = any(idx < len(state) and state[idx] in ("G", "g") for idx in signal.link_indices)
                history = self.signal_history[direction.key][signal.tl_id]
                history["time_s"].append(sim_time)
                history["is_green"].append(is_green)

    def finalize(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        save_trajectory_csvs(self.output_dir, self.trajectories)
        save_signal_csvs(self.output_dir, self.signal_history)
        for direction in self.directions.values():
            plot_direction_time_space(
                output_dir=self.output_dir,
                direction=direction,
                trajectories=self.trajectories[direction.key],
                signal_history=self.signal_history[direction.key],
            )


def resolve_net_file_from_sumocfg(sumocfg_path: str) -> str:
    root = ET.parse(sumocfg_path).getroot()
    net_elem = root.find("./input/net-file")
    if net_elem is None:
        raise ValueError(f"net-file not found in {sumocfg_path}")
    value = net_elem.get("value")
    if not value:
        raise ValueError(f"net-file value missing in {sumocfg_path}")
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(os.path.dirname(sumocfg_path), value))


def build_junction_cross_street_lookup(net, road_name: str) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for node in net.getNodes():
        cross_names = set()
        for edge in node.getIncoming() + node.getOutgoing():
            name = edge.getName()
            if not name or name == road_name:
                continue
            cross_names.add(name)
        lookup[node.getID()] = "; ".join(sorted(cross_names))
    return lookup


def build_corridor_directions(net, road_name: str, cross_lookup: Dict[str, str]) -> Dict[str, CorridorDirection]:
    edge_infos = []
    for edge in net.getEdges():
        if edge.getID().startswith(":") or edge.getName() != road_name:
            continue
        shape = edge.getShape()
        if len(shape) < 2:
            continue
        start_x = shape[0][0]
        end_x = shape[-1][0]
        delta_x = end_x - start_x
        if abs(delta_x) < 1e-6:
            continue
        edge_infos.append(
            {
                "edge_id": edge.getID(),
                "start_x": start_x,
                "end_x": end_x,
                "mid_x": sum(pt[0] for pt in shape) / len(shape),
                "length_m": edge.getLength(),
                "to_node": edge.getToNode().getID(),
            }
        )

    east_edges = [row for row in edge_infos if row["end_x"] > row["start_x"]]
    west_edges = [row for row in edge_infos if row["end_x"] < row["start_x"]]

    east_edges.sort(key=lambda row: (row["start_x"], row["mid_x"]))
    west_edges.sort(key=lambda row: (-row["start_x"], -row["mid_x"]))

    directions = {
        "west_to_east": build_direction_from_edges(
            net=net,
            rows=east_edges,
            key="west_to_east",
            title=f"{road_name}: West to East",
            cross_lookup=cross_lookup,
        ),
        "east_to_west": build_direction_from_edges(
            net=net,
            rows=west_edges,
            key="east_to_west",
            title=f"{road_name}: East to West",
            cross_lookup=cross_lookup,
        ),
    }
    add_detector_landmarks(directions, net)
    apply_iowa_plot_bounds(directions)
    return directions


def build_direction_from_edges(net, rows, key: str, title: str, cross_lookup: Dict[str, str]) -> CorridorDirection:
    edge_ids = [row["edge_id"] for row in rows]
    edge_base_m: Dict[str, float] = {}
    running = 0.0
    for row in rows:
        edge_base_m[row["edge_id"]] = running
        running += row["length_m"]

    signal_map: Dict[str, SignalInfo] = {}
    tl_by_id = {tl.getID(): tl for tl in net.getTrafficLights()}

    for tl_id, tl in tl_by_id.items():
        link_indices = set()
        signal_distance = None
        signal_label = ""
        for in_lane, out_lane, link_index in tl.getConnections():
            in_edge = in_lane.getEdge()
            out_edge = out_lane.getEdge()
            if in_edge.getID() not in edge_base_m or out_edge.getID() not in edge_base_m:
                continue
            link_indices.add(link_index)
            candidate_distance = edge_base_m[in_edge.getID()] + in_edge.getLength()
            signal_distance = candidate_distance if signal_distance is None else max(signal_distance, candidate_distance)
            signal_label = shorten_street_name(cross_lookup.get(in_edge.getToNode().getID(), ""))

        if not link_indices or signal_distance is None:
            continue

        if tl_id in SIGNAL_LABEL_OVERRIDES:
            signal_label = SIGNAL_LABEL_OVERRIDES[tl_id]
        elif not signal_label:
            signal_label = tl_id

        signal_map[tl_id] = SignalInfo(
            tl_id=tl_id,
            distance_m=signal_distance,
            link_indices=sorted(link_indices),
            label=signal_label,
        )

    signal_infos = sorted(signal_map.values(), key=lambda row: row.distance_m)
    return CorridorDirection(
        key=key,
        title=title,
        edge_ids=edge_ids,
        edge_set=set(edge_ids),
        edge_base_m=edge_base_m,
        total_length_m=running,
        signal_infos=signal_infos,
        landmark_infos=[],
        plot_max_m=running,
    )


def add_detector_landmarks(directions: Dict[str, CorridorDirection], net) -> None:
    """Add named non-signal locations from calibrated detector placement data."""
    if not (os.path.exists(DEFAULT_DETECTOR_PLACEMENT_LOG)
            and os.path.exists(DEFAULT_STOP_BAR_CHOICES)):
        return

    with open(DEFAULT_STOP_BAR_CHOICES) as f:
        stop_choices = json.load(f)

    placement_rows = {}
    with open(DEFAULT_DETECTOR_PLACEMENT_LOG, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            approach_id = row.get("approach_id") or row.get("approach")
            segment_id = row.get("segment_id") or row.get("simpleID")
            placement_rows[(approach_id, str(segment_id))] = row

    approach_for_direction = {
        "west_to_east": ("Cranford_eb", "Eucalyptus_eb"),
        "east_to_west": ("Cranford_wb", "Eucalyptus_wb"),
    }

    for direction_key, approach_ids in approach_for_direction.items():
        direction = directions.get(direction_key)
        if direction is None:
            continue
        existing_labels = {signal.label for signal in direction.signal_infos}
        landmarks = []
        for approach_id in approach_ids:
            label = approach_id.rsplit("_", 1)[0]
            if label in existing_labels:
                continue
            segment_id = stop_choices.get(approach_id)
            row = placement_rows.get((approach_id, str(segment_id)))
            if not row:
                continue
            edge_id = row.get("edge_id") or row.get("edge")
            if edge_id not in direction.edge_base_m:
                continue
            try:
                edge = net.getEdge(edge_id)
                distance_m = direction.edge_base_m[edge_id] + edge.getLength()
            except KeyError:
                distance_m = direction.edge_base_m[edge_id]
            landmarks.append(LandmarkInfo(distance_m=distance_m, label=label))
        direction.landmark_infos = sorted(landmarks, key=lambda row: row.distance_m)


def apply_iowa_plot_bounds(directions: Dict[str, CorridorDirection]) -> None:
    """Hide University Avenue edges east of Iowa where the calibrated EB trips end."""
    for direction in directions.values():
        direction.plot_min_m = 0.0
        direction.plot_max_m = direction.total_length_m

    west_to_east = directions.get("west_to_east")
    if west_to_east:
        iowa = next((s for s in west_to_east.signal_infos if s.label == "Iowa"), None)
        if iowa:
            west_to_east.plot_max_m = iowa.distance_m

    east_to_west = directions.get("east_to_west")
    if east_to_west:
        iowa = next((s for s in east_to_west.signal_infos if s.label == "Iowa"), None)
        if iowa:
            east_to_west.plot_min_m = iowa.distance_m


def shorten_street_name(name: str) -> str:
    if not name:
        return ""
    suffix_tokens = {
        "Avenue",
        "Street",
        "Drive",
        "Road",
        "Boulevard",
        "Lane",
        "Way",
        "Court",
        "Place",
        "Terrace",
        "West",
        "East",
        "North",
        "South",
    }
    labels = []
    for raw_name in name.split(";"):
        tokens = raw_name.strip().split()
        while tokens and tokens[-1] in suffix_tokens:
            tokens.pop()
        labels.append(" ".join(tokens) if tokens else raw_name.strip())
    return " / ".join(label for label in labels if label)


def compress_signal_runs(times_s: List[float], states: List[bool]) -> List[Tuple[float, float, bool]]:
    if not times_s:
        return []
    runs = []
    start = times_s[0]
    current = states[0]
    prev = times_s[0]
    if len(times_s) > 1:
        step = max(times_s[1] - times_s[0], 0.1)
    else:
        step = 1.0

    for t, state in zip(times_s[1:], states[1:]):
        if state != current:
            runs.append((start, prev + step, current))
            start = t
            current = state
        prev = t
    runs.append((start, prev + step, current))
    return runs


def plot_direction_time_space(output_dir: str, direction: CorridorDirection, trajectories, signal_history) -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=220)

    plot_min_m = direction.plot_min_m
    plot_max_m = direction.plot_max_m if direction.plot_max_m is not None else direction.total_length_m
    max_time_s = 0.0
    for series in trajectories.values():
        if len(series["time_s"]) < 2:
            continue
        in_window = [
            plot_min_m <= distance_m <= plot_max_m
            for distance_m in series["distance_m"]
        ]
        if not any(in_window):
            continue
        time_min = [t / 60.0 for t in series["time_s"]]
        dist_km = [d / 1000.0 for d in series["distance_m"]]
        max_time_s = max(max_time_s, series["time_s"][-1])
        ax.plot(time_min, dist_km, color="#1f2937", linewidth=0.1, alpha=0.4)

    band_half_height_km = max(direction.total_length_m / 1000.0 * 0.006, 0.015)
    for history in signal_history.values():
        if not (plot_min_m <= history["distance_m"] <= plot_max_m):
            continue
        runs = compress_signal_runs(history["time_s"], history["is_green"])
        green_runs = []
        red_runs = []
        for start_s, end_s, is_green in runs:
            bucket = green_runs if is_green else red_runs
            bucket.append((start_s / 60.0, (end_s - start_s) / 60.0))
        y0 = history["distance_m"] / 1000.0 - band_half_height_km
        band_h = 2 * band_half_height_km
        if red_runs:
            ax.broken_barh(red_runs, (y0, band_h), facecolors="#fca5a5", alpha=0.22, edgecolors="none", zorder=0)
        if green_runs:
            ax.broken_barh(green_runs, (y0, band_h), facecolors="#86efac", alpha=0.35, edgecolors="none", zorder=1)

        label_x = max_time_s / 60.0 + 0.8 if max_time_s > 0 else 0.8
        ax.text(
            label_x,
            history["distance_m"] / 1000.0,
            history["label"],
            fontsize=8,
            va="center",
            ha="left",
            color="#374151",
        )

    for landmark in direction.landmark_infos:
        if not (plot_min_m <= landmark.distance_m <= plot_max_m):
            continue
        y = landmark.distance_m / 1000.0
        ax.axhline(y, color="#7c3aed", linewidth=0.7, linestyle=(0, (3, 3)), alpha=0.7)
        label_x = max_time_s / 60.0 + 0.8 if max_time_s > 0 else 0.8
        ax.text(
            label_x,
            y,
            landmark.label,
            fontsize=8,
            va="center",
            ha="left",
            color="#5b21b6",
        )

    ax.set_title(f"Mainline Time-Space Diagram ({direction.title})")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Distance Along Mainline (km)")
    ax.grid(True, color="#d1d5db", linewidth=0.5, alpha=0.6)

    if max_time_s > 0:
        ax.set_xlim(0, max_time_s / 60.0 + 3.0)
    ax.set_ylim(plot_min_m / 1000.0, max(plot_max_m / 1000.0, 0.1))

    legend_items = [
        Patch(facecolor="#86efac", alpha=0.35, label="Mainline green"),
        Patch(facecolor="#fca5a5", alpha=0.22, label="Mainline not green"),
        Patch(facecolor="#7c3aed", alpha=0.7, label="Named detector landmark"),
    ]
    ax.legend(handles=legend_items, loc="upper left")

    fig.tight_layout()
    out_path = os.path.join(output_dir, f"mainline_time_space_{direction.key}.svg")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_trajectory_csvs(output_dir: str, trajectories) -> None:
    out_path = os.path.join(output_dir, "mainline_trajectories.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as csvfile:
        csvfile.write("direction,vehicle_id,time_s,distance_m\n")
        for direction, vehicle_rows in trajectories.items():
            for veh_id, series in vehicle_rows.items():
                for time_s, distance_m in zip(series["time_s"], series["distance_m"]):
                    csvfile.write(f"{direction},{veh_id},{time_s:.2f},{distance_m:.2f}\n")


def save_signal_csvs(output_dir: str, signal_history) -> None:
    out_path = os.path.join(output_dir, "mainline_signal_states.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as csvfile:
        csvfile.write("direction,tl_id,label,time_s,distance_m,is_green\n")
        for direction, signal_rows in signal_history.items():
            for tl_id, history in signal_rows.items():
                for time_s, is_green in zip(history["time_s"], history["is_green"]):
                    csvfile.write(
                        f"{direction},{tl_id},{history['label']},{time_s:.2f},{history['distance_m']:.2f},{int(is_green)}\n"
                    )

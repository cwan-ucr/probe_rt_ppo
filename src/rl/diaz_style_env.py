"""Detector-observation/queue-reward comparator on the common signal backbone.

This is a corridor adaptation of the Diaz-style information and control design,
not a claim to reproduce that paper's original network or phase-transition code.
The existing PPO learner, demand files, 60-s plans, 15-min decisions, and SUMO
physical evaluation metrics remain shared with the probe-based experiments.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from functools import lru_cache

import numpy as np

from . import config as C
from .corridor_env import CorridorEnv, collect_step_data
import sumolib
import traci


# Legacy v11 observation: both passive loops are on each controlled incoming
# lane, approximately 6 m apart near the stop bar. Neither actuates a TLS.
STOP_POS_M = -2.3
ADVANCE_POS_M = -8.3
LOOP_LENGTH_M = 1.8
QUEUE_VEHICLE_LENGTH_M = 5.0
SATURATION_FLOW_VPHPL = 1800.0


def _arrival_direction(shape):
    """Direction of travel into the intersection, not the outgoing turn."""
    (x0, y0), (x1, y1) = shape[-2:]
    dx, dy = x1 - x0, y1 - y0
    if abs(dx) >= abs(dy):
        return "eb" if dx > 0 else "wb"
    return "nb" if dy > 0 else "sb"


@lru_cache(maxsize=1)
def detector_layout():
    """Stable node/direction/lane ownership from controlled incoming links."""
    net = sumolib.net.readNet(C.NET)
    lanes = {}
    edges = {}
    for node_idx, name in enumerate(C.AGENT_ORDER):
        incoming = {}
        for tlid in C.AGENT_TLIDS[name]:
            for from_lane, _to_lane, _link_index in net.getTLS(tlid).getConnections():
                incoming[from_lane.getID()] = from_lane
        for lane_id, lane in sorted(incoming.items()):
            edge = lane.getEdge()
            direction = _arrival_direction(edge.getShape())
            dir_idx = C.DIRECTIONS.index(direction)
            # A physical lane can appear in more than one TLS connection, but
            # it needs only one stop-bar sensor and one approach assignment.
            previous = lanes.get(lane_id)
            if previous is not None and previous[:2] != (node_idx, dir_idx):
                raise ValueError(f"incoming lane assigned twice: {lane_id}")
            lanes[lane_id] = (node_idx, dir_idx, float(lane.getLength()))
            edges[edge.getID()] = float(edge.getLength())
    if not lanes or not edges:
        raise RuntimeError("no controlled incoming lanes were found")
    return lanes, edges


@lru_cache(maxsize=1)
def advance_layout():
    """Place the legacy advance loop 6 m upstream of each stop-bar loop."""
    lanes, _edges = detector_layout()
    return {lane_id: (node_idx, dir_idx, ADVANCE_POS_M, length)
            for lane_id, (node_idx, dir_idx, length) in lanes.items()}


def build_detector_file(path, output_file="NUL"):
    """Write passive E1 loops and return ID -> (node, direction, site).

    The legacy flow is a sum of per-step detector presence, not SUMO's native
    15-minute entry count. The optional output file is diagnostic only.
    """
    lanes, _edges = detector_layout()
    upstream = advance_layout()
    root = ET.Element("additional")
    mapping = {}
    for site_idx, (site, sites) in enumerate((
            ("advance", upstream),
            ("stopbar", {lane_id: (node_idx, dir_idx, STOP_POS_M, length)
                         for lane_id, (node_idx, dir_idx, length)
                         in lanes.items()}))):
        for lane_index, (lane_id, (node_idx, dir_idx, pos, length)) in enumerate(
                sorted(sites.items())):
            if abs(pos) + LOOP_LENGTH_M > length:
                raise ValueError(f"{site} detector does not fit: {lane_id}")
            detector_id = f"diaz_obs_{lane_index}_{site}"
            ET.SubElement(root, "inductionLoop", {
                "id": detector_id, "lane": lane_id, "pos": str(pos),
                "length": str(LOOP_LENGTH_M),
                "period": str(C.DECISION_INTERVAL_S),
                "file": str(output_file), "friendlyPos": "true",
            })
            mapping[detector_id] = (node_idx, dir_idx, site_idx)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return mapping


class DiazCorridorEnv(CorridorEnv):
    """160-value observation (10 nodes x 4 approaches x 2 sites x 2 values).

    For each approach and site, legacy flow sums the per-step vehicle count
    over the preceding 15 minutes, averages across lanes, then divides by 450
    (=1800 veh/h/lane). A vehicle remaining on a loop can count repeatedly.
    Occupancy is mean detector occupancy
    as a fraction in [0,1]. Absent approaches are exactly zero.  The reward is
    the negative time-mean of sum_e(5 m * halting_vehicles_e / length_e) over
    unique controlled incoming links. No difference term or offset prior enters.
    """

    def __init__(self, *args, **kwargs):
        detector_output_file = kwargs.pop("detector_output_file", "NUL")
        if C.RL_METHOD != "diaz" or C.CONTROL_LAYOUT != "all10":
            raise RuntimeError("DiazCorridorEnv requires RL_METHOD=diaz and all10")
        if C.SPLIT_CONTROL or C.RESIDUAL_ACTION or C.OFFSET_DEVIATION_PENALTY:
            raise RuntimeError(
                "Diaz comparator requires absolute offset, no split, no offset prior")
        super().__init__(*args, **kwargs)
        filename = (f"diaz_sensors_pid{os.getpid()}_"
                    f"port{self._port if self._port is not None else 'auto'}.add.xml")
        sensor_path = os.path.join(self._runtime_dir, filename)
        self._diaz_detectors = build_detector_file(
            sensor_path, output_file=detector_output_file)
        self._diaz_sensor_path = sensor_path
        self._diaz_interval_detector_counts = {}
        self.extra_additional_files.append(sensor_path)
        _controlled_lanes, self._diaz_edges = detector_layout()
        self._n_lanes = np.zeros(
            (C.N_AGENTS, len(C.DIRECTIONS), 2), dtype=np.float32)
        for node_idx, dir_idx, site_idx in self._diaz_detectors.values():
            self._n_lanes[node_idx, dir_idx, site_idx] += 1
        self._diaz_subscribed = False

    def _subscribe_diaz(self):
        tc = traci.constants
        live_ids = set(traci.inductionloop.getIDList())
        missing = set(self._diaz_detectors) - live_ids
        if missing:
            raise RuntimeError(f"Diaz observation loops missing in SUMO: {sorted(missing)[:3]}")
        variables = [tc.LAST_STEP_VEHICLE_NUMBER, tc.LAST_STEP_OCCUPANCY]
        for detector_id in self._diaz_detectors:
            traci.inductionloop.subscribe(detector_id, variables)
        for edge_id in self._diaz_edges:
            traci.edge.subscribe(edge_id, [tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
        self._diaz_subscribed = True

    def _policy_state(self, readings, time_fraction):
        return self._diaz_state.copy()

    def _run(self, seconds):
        """Collect passive detector readings and time-average physical queues."""
        if not self._diaz_subscribed:
            self._subscribe_diaz()
        tc = traci.constants
        dt = self._dt
        n = int(round(seconds / dt))
        counts = np.zeros((C.N_AGENTS, len(C.DIRECTIONS), 2), dtype=np.float64)
        # Legacy v11 semantics: count vehicle presence every 0.5-s step.
        # This is deliberately not a unique-vehicle flow count.
        detector_counts = {detector_id: 0 for detector_id in self._diaz_detectors}
        occupancy = np.zeros_like(counts)
        queue_sum = 0.0
        t = traci.simulation.getTime()
        target = t + n * dt
        for i in range(n):
            if (t >= self._demand_end and (i & 0x3F) == 0
                    and traci.simulation.getMinExpectedNumber() <= 0):
                traci.simulationStep(target)
                break
            traci.simulationStep()
            t += dt
            # Retain the existing probe collector only for comparable monitoring
            # columns; the Diaz policy neither observes nor rewards this data.
            collect_step_data(self.segments, self._entries, self._spd,
                              sim_time=t, step_length=dt)
            loop_results = traci.inductionloop.getAllSubscriptionResults()
            for detector_id, (node_idx, dir_idx, site_idx) in self._diaz_detectors.items():
                values = loop_results.get(detector_id, {})
                detector_counts[detector_id] += values.get(
                    tc.LAST_STEP_VEHICLE_NUMBER, 0)
                occupancy[node_idx, dir_idx, site_idx] += values.get(
                    tc.LAST_STEP_OCCUPANCY, 0.0) / 100.0
            edge_results = traci.edge.getAllSubscriptionResults()
            queue_sum += sum(
                QUEUE_VEHICLE_LENGTH_M
                * edge_results.get(edge_id, {}).get(
                    tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
                / edge_length
                for edge_id, edge_length in self._diaz_edges.items())
        self._diaz_interval_detector_counts = detector_counts
        for detector_id, (node_idx, dir_idx, site_idx) in self._diaz_detectors.items():
            counts[node_idx, dir_idx, site_idx] += (
                self._diaz_interval_detector_counts[detector_id])
        # Canonical flattening: node, [eb,wb,nb,sb],
        # [advance_flow,advance_occ,stopbar_flow,stopbar_occ].
        lane_count = np.maximum(self._n_lanes, 1.0)
        flow = counts / lane_count / (SATURATION_FLOW_VPHPL * seconds / 3600.0)
        occ = occupancy / lane_count / max(n, 1)
        state = np.stack((flow[:, :, 0], occ[:, :, 0],
                          flow[:, :, 1], occ[:, :, 1]), axis=-1)
        self._diaz_state = state.astype(np.float32).reshape(-1)
        self._diaz_queue_cost = float(queue_sum / max(n, 1))
        if self._diaz_state.size != C.STATE_DIM:
            raise RuntimeError("Diaz observation dimension does not match config")

    def reset(self):
        self._diaz_subscribed = False
        super().reset()
        return self._diaz_state.copy()

    def step(self, action):
        # The parent's segment delay remains in info solely as an offline
        # diagnostic; it must not participate in the Diaz reward.
        self.baseline = None
        _probe_state, _probe_reward, done, info = super().step(action)
        reward = -self._diaz_queue_cost
        info["traffic_reward"] = reward
        info["offset_deviation_penalty"] = 0.0
        info["diaz_queue_cost"] = self._diaz_queue_cost
        return self._diaz_state.copy(), reward, done, info

    def compute_baseline(self):
        # A fixed absolute action cannot represent the hourly program-specific
        # incumbent offsets. Temporarily use the parent's zero-residual action
        # so the cached baseline really is the same ATSC controller throughout.
        original = C.RESIDUAL_ACTION
        try:
            C.RESIDUAL_ACTION = True
            return super().compute_baseline()
        finally:
            C.RESIDUAL_ACTION = original

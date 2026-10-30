"""
FlowParameters: parameter space for SPSA calibration of flow-based demand.

Replaces the old DemandParameters (randomTrips-based). Now parameters are:
  - Arterial EB/WB flow per hour (University Ave)
  - Cross street flows per hour per direction

Each flow gets a bounds constraint of [0.5x, 2.0x] of its initial value
(derived from fringe_flows.csv).
"""

import os
import json
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Optional


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)

DEFAULT_FRINGE_CSV = os.path.join(ROOT_DIR, "outputs", "calibration", "fringe_flows.csv")
DEFAULT_BOUNDARY_JSON = os.path.join(ROOT_DIR, "outputs", "calibration",
                                       "corridor_boundary_edges.json")
DEFAULT_PENETRATION = os.path.join(ROOT_DIR, "outputs", "calibration",
                                     "penetration_rate.json")

# Entry → Exit mapping (same as generate_demand_from_flows.py)
ARTERIAL_PAIRS = {
    "uni_west_entry_EB": "uni_east_exit_EB",
    "uni_east_entry_WB": "uni_west_exit_WB",
}


def build_cross_pairs(boundaries):
    pairs = {}
    for cross, info in boundaries["cross"].items():
        if "south_entry_NB" in info and "north_exit_NB" in info:
            pairs[f"{cross}_south_entry_NB"] = f"{cross}_north_exit_NB"
        if "north_entry_SB" in info and "south_exit_SB" in info:
            pairs[f"{cross}_north_entry_SB"] = f"{cross}_south_exit_SB"
    return pairs


def get_boundary_edge(boundaries, key):
    if key in boundaries["arterial"]:
        return boundaries["arterial"][key]["edge"]
    for cross, info in boundaries["cross"].items():
        for sub, detail in info.items():
            if f"{cross}_{sub}" == key:
                return detail["edge"]
    return None


def get_boundary_approach_edges(boundaries, key):
    if key in boundaries["arterial"]:
        return boundaries["arterial"][key].get("approach_edges", [])
    for cross, info in boundaries["cross"].items():
        for sub, detail in info.items():
            if f"{cross}_{sub}" == key:
                return detail.get("approach_edges", [])
    return []


def build_via_string(entry_edges, exit_edges, from_edge, to_edge):
    """Force routing through full approach (entry + exit), excluding from/to endpoints."""
    via = []
    seen = {from_edge, to_edge}
    for e in entry_edges:
        if e == from_edge or e in seen:
            continue
        via.append(e)
        seen.add(e)
    for e in exit_edges:
        if e == to_edge or e in seen:
            continue
        via.append(e)
        seen.add(e)
    return " ".join(via)


@dataclass
class FlowParameters:
    """Calibration parameters: hourly flows per boundary entry key.

    Structure: flows[entry_key][hour] = vehs_per_hour
    """
    flows: dict = field(default_factory=dict)
    calibration_hours: tuple = (7, 13)
    bounds_lower_factor: float = 0.5
    bounds_upper_factor: float = 2.0
    _initial_flows: Optional[dict] = None  # for bounds reference

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------
    @classmethod
    def from_fringe_flows_csv(cls, csv_path=DEFAULT_FRINGE_CSV,
                               calibration_hours=(7, 13)):
        """Build initial parameters from fringe_flows.csv.

        Only entries corresponding to ENTRY boundaries are used (exits are
        determined by routing).
        """
        df = pd.read_csv(csv_path)
        flows = {}

        # Only keep entry keys (those paired with exits in entry→exit mapping)
        # Easier approach: keep keys containing "_entry_" or "entry_EB/WB"
        entry_mask = df["boundary_key"].str.contains("entry", case=False)
        df_entry = df[entry_mask]

        start, end = calibration_hours
        df_entry = df_entry[(df_entry["hour"] >= start) & (df_entry["hour"] < end)]

        for key, grp in df_entry.groupby("boundary_key"):
            flows[key] = {int(h): int(v) for h, v in
                           zip(grp["hour"], grp["V_veh_per_hour"])}

        # Ensure all hours are present (fill with 0 if missing)
        for key, per_hour in flows.items():
            for h in range(start, end):
                if h not in per_hour:
                    per_hour[h] = 0

        instance = cls(
            flows=flows,
            calibration_hours=calibration_hours,
            _initial_flows=deepcopy(flows),
        )
        return instance

    # -------------------------------------------------------------------------
    # Vector representation (for SPSA)
    # -------------------------------------------------------------------------
    def _ordered_keys(self):
        """Consistent ordering of (entry_key, hour) pairs for vectorization."""
        keys = sorted(self.flows.keys())
        pairs = []
        start, end = self.calibration_hours
        for k in keys:
            for h in range(start, end):
                pairs.append((k, h))
        return pairs

    def to_vector(self):
        """Flatten flows to 1D numpy array."""
        pairs = self._ordered_keys()
        return np.array([self.flows[k].get(h, 0) for k, h in pairs], dtype=float)

    def from_vector_update(self, vec):
        """Update self in-place from vector (same ordering as to_vector)."""
        pairs = self._ordered_keys()
        for (k, h), v in zip(pairs, vec):
            if k not in self.flows:
                self.flows[k] = {}
            self.flows[k][h] = int(max(0, round(v)))

    @classmethod
    def from_vector(cls, vec, template):
        """Construct new instance from vector using template's structure."""
        new_params = template.copy()
        new_params.from_vector_update(vec)
        return new_params

    def perturb(self, epsilon, direction):
        """Create perturbed copy for SPSA gradient estimation."""
        vec = self.to_vector() + epsilon * direction
        return FlowParameters.from_vector(vec, self)

    def project_bounds(self):
        """Clip flows to [factor_low × initial, factor_high × initial].
        If no initial reference, clip to [0, 3000] as a safety net.
        """
        if not self._initial_flows:
            for k, per_hour in self.flows.items():
                for h, v in per_hour.items():
                    per_hour[h] = int(np.clip(v, 0, 3000))
            return

        for k, per_hour in self.flows.items():
            init_per_hour = self._initial_flows.get(k, {})
            for h, v in per_hour.items():
                v0 = init_per_hour.get(h, 0)
                lo = max(0, v0 * self.bounds_lower_factor)
                hi = max(10, v0 * self.bounds_upper_factor)
                per_hour[h] = int(np.clip(v, lo, hi))

    def copy(self):
        return deepcopy(self)

    # -------------------------------------------------------------------------
    # SUMO flow XML generation
    # -------------------------------------------------------------------------
    def to_flow_xml(self, boundaries_json, out_file, probe_prob,
                    probe_prob_cross=None, out_file_probe=None, seed=42):
        """Render current flows to SUMO <flow>-based XML.

        Similar structure to src/generate_demand_from_flows.py but uses
        self.flows instead of reading fringe_flows.csv.

        probe_prob       : penetration rate for arterial (EB/WB) flows
        probe_prob_cross : penetration rate for cross-street (NB/SB) flows;
                           defaults to probe_prob if not provided
        """
        if probe_prob_cross is None:
            probe_prob_cross = probe_prob

        with open(boundaries_json) as f:
            boundaries = json.load(f)

        pairs = dict(ARTERIAL_PAIRS)
        pairs.update(build_cross_pairs(boundaries))

        root = ET.Element("routes")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("xsi:noNamespaceSchemaLocation",
                 "http://sumo.dlr.de/xsd/routes_file.xsd")
        ET.SubElement(root, "vType", id="probe_vehicle", vClass="passenger")
        ET.SubElement(root, "vType", id="regular_vehicle", vClass="passenger")

        start_hour, end_hour = self.calibration_hours

        # SUMO loads route files incrementally and stops at the first flow
        # whose `begin` time is in the future. So we MUST emit flows sorted
        # by begin time (then by entry_key for stability).
        flow_records = []
        for entry_key in sorted(self.flows.keys()):
            exit_key = pairs.get(entry_key)
            if not exit_key:
                continue
            from_edge = get_boundary_edge(boundaries, entry_key)
            to_edge = get_boundary_edge(boundaries, exit_key)
            if not from_edge or not to_edge:
                continue

            # via= constraint to force routing through full approach
            entry_app_edges = get_boundary_approach_edges(boundaries, entry_key)
            exit_app_edges = get_boundary_approach_edges(boundaries, exit_key)
            via_str = build_via_string(entry_app_edges, exit_app_edges,
                                        from_edge, to_edge)

            # Arterial entries start with "uni_"; cross-street entries have an
            # intersection name prefix (e.g. "Iowa_south_entry_NB")
            is_arterial = entry_key.startswith("uni_")
            p = probe_prob if is_arterial else probe_prob_cross

            for hour in range(start_hour, end_hour):
                vph = self.flows[entry_key].get(hour, 0)
                if vph <= 0:
                    continue

                begin = (hour - start_hour) * 3600
                end = begin + 3600

                # Split into regular and probe sub-flows
                v_probe = int(round(vph * p))
                v_reg = vph - v_probe

                attrs = {"from": from_edge, "to": to_edge,
                         "begin": str(begin), "end": str(end)}
                if via_str:
                    attrs["via"] = via_str

                if v_reg > 0:
                    flow_records.append((begin, entry_key, hour, "reg",
                                         {**attrs, "vehsPerHour": str(v_reg)},
                                         "regular_vehicle"))
                if v_probe > 0:
                    flow_records.append((begin, entry_key, hour, "probe",
                                         {**attrs, "vehsPerHour": str(v_probe)},
                                         "probe_vehicle"))

        # Emit flows in (begin, entry_key) order
        flow_records.sort(key=lambda r: (r[0], r[1], r[3]))
        for begin, entry_key, hour, kind, attrs, vtype in flow_records:
            ET.SubElement(
                root, "flow",
                id=f"{entry_key}_h{hour}_{kind}", type=vtype,
                attrib=attrs,
            )

        ET.indent(root, space="  ")
        tree = ET.ElementTree(root)
        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
        target = out_file_probe if out_file_probe else out_file
        tree.write(target, encoding="utf-8", xml_declaration=True)
        return target

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    def summary(self):
        lines = []
        start, end = self.calibration_hours
        for k in sorted(self.flows.keys()):
            per_hour = [self.flows[k].get(h, 0) for h in range(start, end)]
            total = sum(per_hour)
            lines.append(f"  {k}: total={total}, per_hour={per_hour}")
        return "\n".join(lines)

    def __repr__(self):
        n_params = len(self._ordered_keys())
        return (f"FlowParameters({len(self.flows)} boundaries, "
                f"{n_params} vars, hours={self.calibration_hours})")


if __name__ == "__main__":
    # Quick self-test
    params = FlowParameters.from_fringe_flows_csv()
    print(params)
    print(params.summary())
    print(f"\nVector length: {len(params.to_vector())}")
    print(f"First 10 values: {params.to_vector()[:10]}")

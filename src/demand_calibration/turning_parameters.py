"""
TurningParameters: parameter space for SPSA / GA calibration of demand with
explicit turning movements.

Parameter layout (82 total):
  • 78 hourly flow values  (13 boundary entries × 6 hours)
  • 4 turning ratios:
      - left_art    : fraction of arterial vehicles that turn left
      - right_art   : fraction of arterial vehicles that turn right
      - left_cross  : fraction of cross-street vehicles that turn left
      - right_cross : fraction of cross-street vehicles that turn right
    (straight share = 1 - left - right per category)

Demand generation matches src/generate_demand_turning.py:
  - Arterial entries (EB/WB) split into:
      straight        → far arterial exit on the same direction
      left/right turns → equally distributed across cross-street exits on
                         the appropriate side of the corridor
  - Cross-street entries (NB/SB) split into:
      straight  → opposite cross-street exit at same intersection
      left turn → far arterial exit on the perpendicular direction
      right turn → far arterial exit on the other side
"""

import os
import json
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from copy import deepcopy
from typing import Optional
from dataclasses import dataclass, field

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.dirname(SCRIPT_DIR)
ROOT_DIR   = os.path.dirname(SRC_DIR)

DEFAULT_FRINGE_CSV    = os.path.join(ROOT_DIR, "outputs", "calibration",
                                     "fringe_flows.csv")
DEFAULT_BOUNDARY_JSON = os.path.join(ROOT_DIR, "outputs", "calibration",
                                     "corridor_boundary_edges.json")

CORRIDOR_ORDER = [
    "Lime", "Park", "Victoria", "Eucalyptus",
    "Kansas", "Ottawa", "Chicago", "Cranford", "Iowa",
]

# Default initial turning ratios (taken from generate_demand_turning.py CLI defaults)
DEFAULT_LEFT_ART    = 0.20
DEFAULT_RIGHT_ART   = 0.10
DEFAULT_LEFT_CROSS  = 0.30
DEFAULT_RIGHT_CROSS = 0.10

# Bounds on individual turning ratios
TURN_LOW  = 0.02
TURN_HIGH = 0.45
# Constraint: left + right ≤ 0.95 (leaves ≥ 5% for straight)
TURN_PAIR_SUM_MAX = 0.95


def _direction(bkey: str) -> str:
    return bkey.split("_")[-1]


def _is_arterial(direction: str) -> bool:
    return direction in ("EB", "WB")


def _build_straight_via(entry_edges, exit_edges, from_edge, to_edge) -> str:
    via, seen = [], {from_edge, to_edge}
    for e in list(entry_edges) + list(exit_edges):
        if e not in seen:
            via.append(e); seen.add(e)
    return " ".join(via)


@dataclass
class TurningParameters:
    """82-parameter demand model with turning movements.

    flows[entry_key][hour] = vehs_per_hour (78 vars)
    + 4 scalar turning fractions
    """
    flows: dict = field(default_factory=dict)
    left_art:    float = DEFAULT_LEFT_ART
    right_art:   float = DEFAULT_RIGHT_ART
    # cross-street turn ratios split by entry direction (NB vs SB)
    left_cross_nb:  float = DEFAULT_LEFT_CROSS
    right_cross_nb: float = DEFAULT_RIGHT_CROSS
    left_cross_sb:  float = DEFAULT_LEFT_CROSS
    right_cross_sb: float = DEFAULT_RIGHT_CROSS
    calibration_hours: tuple = (7, 13)
    bounds_lower_factor: float = 0.5
    bounds_upper_factor: float = 2.0
    # per-movement-type GA search bounds: factor x seed flow. Cross streets get a
    # wider upper bound (under-congested in sim -> may need much more demand).
    art_lower_factor:   float = 0.7
    art_upper_factor:   float = 2.5
    cross_lower_factor: float = 0.9
    cross_upper_factor: float = 5.0
    _initial_flows: Optional[dict] = None

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------
    @classmethod
    def from_fringe_flows_csv(cls, csv_path=DEFAULT_FRINGE_CSV,
                              calibration_hours=(7, 13),
                              left_art=DEFAULT_LEFT_ART,
                              right_art=DEFAULT_RIGHT_ART,
                              left_cross=DEFAULT_LEFT_CROSS,
                              right_cross=DEFAULT_RIGHT_CROSS,
                              left_cross_nb=None, right_cross_nb=None,
                              left_cross_sb=None, right_cross_sb=None):
        df = pd.read_csv(csv_path)
        entry_mask = df["boundary_key"].str.contains("entry", case=False)
        df_entry = df[entry_mask]
        start, end = calibration_hours
        df_entry = df_entry[(df_entry["hour"] >= start) &
                             (df_entry["hour"] < end)]

        flows = {}
        for key, grp in df_entry.groupby("boundary_key"):
            flows[key] = {int(h): int(v) for h, v in
                          zip(grp["hour"], grp["V_veh_per_hour"])}
        for key, per_hour in flows.items():
            for h in range(start, end):
                if h not in per_hour:
                    per_hour[h] = 0

        return cls(
            flows=flows,
            left_art=left_art, right_art=right_art,
            left_cross_nb=left_cross_nb if left_cross_nb is not None else left_cross,
            right_cross_nb=right_cross_nb if right_cross_nb is not None else right_cross,
            left_cross_sb=left_cross_sb if left_cross_sb is not None else left_cross,
            right_cross_sb=right_cross_sb if right_cross_sb is not None else right_cross,
            calibration_hours=calibration_hours,
            _initial_flows=deepcopy(flows),
        )

    # -------------------------------------------------------------------------
    # Vector representation
    # -------------------------------------------------------------------------
    def _ordered_flow_keys(self):
        keys = sorted(self.flows.keys())
        pairs = []
        start, end = self.calibration_hours
        for k in keys:
            for h in range(start, end):
                pairs.append((k, h))
        return pairs

    def _factors_for_key(self, k):
        """(lower, upper) bound factor for an entry key, by movement type."""
        if _is_arterial(_direction(k)):
            return self.art_lower_factor, self.art_upper_factor
        return self.cross_lower_factor, self.cross_upper_factor

    def vector_bounds(self):
        """(lo, hi) arrays aligned with to_vector(): per-flow factor bounds by
        arterial/cross type, then the 4 turning ratios in [TURN_LOW, TURN_HIGH]."""
        pairs = self._ordered_flow_keys()
        v0 = self.to_vector()
        n = len(pairs)
        lo = np.empty(n); hi = np.empty(n)
        for i, (k, _h) in enumerate(pairs):
            lof, hif = self._factors_for_key(k)
            lo[i] = max(0.0, v0[i] * lof)
            hi[i] = max(10.0, v0[i] * hif)
        lo = np.concatenate([lo, np.full(6, TURN_LOW)])   # 6 turning ratios
        hi = np.concatenate([hi, np.full(6, TURN_HIGH)])
        return lo, hi

    def to_vector(self) -> np.ndarray:
        pairs = self._ordered_flow_keys()
        flow_vec = np.array([self.flows[k].get(h, 0) for k, h in pairs],
                            dtype=float)
        turn_vec = np.array([self.left_art, self.right_art,
                             self.left_cross_nb, self.right_cross_nb,
                             self.left_cross_sb, self.right_cross_sb], dtype=float)
        return np.concatenate([flow_vec, turn_vec])

    def from_vector_update(self, vec: np.ndarray):
        pairs = self._ordered_flow_keys()
        n_flows = len(pairs)
        # flow part
        for (k, h), v in zip(pairs, vec[:n_flows]):
            if k not in self.flows:
                self.flows[k] = {}
            self.flows[k][h] = int(max(0, round(v)))
        # turning part
        self.left_art       = float(vec[n_flows + 0])
        self.right_art      = float(vec[n_flows + 1])
        self.left_cross_nb  = float(vec[n_flows + 2])
        self.right_cross_nb = float(vec[n_flows + 3])
        self.left_cross_sb  = float(vec[n_flows + 4])
        self.right_cross_sb = float(vec[n_flows + 5])

    @classmethod
    def from_vector(cls, vec: np.ndarray, template: "TurningParameters"):
        new = template.copy()
        new.from_vector_update(vec)
        return new

    def project_bounds(self):
        """Clip flows to [factor_low × initial, factor_high × initial], and
        clip turning ratios to [TURN_LOW, TURN_HIGH] with pair-sum constraint."""
        # Flows
        if self._initial_flows:
            for k, per_hour in self.flows.items():
                lof, hif = self._factors_for_key(k)
                init_per_hour = self._initial_flows.get(k, {})
                for h, v in per_hour.items():
                    v0 = init_per_hour.get(h, 0)
                    lo = max(0, v0 * lof)
                    hi = max(10, v0 * hif)
                    per_hour[h] = int(np.clip(v, lo, hi))
        else:
            for k, per_hour in self.flows.items():
                for h, v in per_hour.items():
                    per_hour[h] = int(np.clip(v, 0, 3000))

        # Turning ratios (arterial + cross NB + cross SB)
        for r in ("left_art", "right_art", "left_cross_nb", "right_cross_nb",
                  "left_cross_sb", "right_cross_sb"):
            setattr(self, r, float(np.clip(getattr(self, r), TURN_LOW, TURN_HIGH)))
        # Pair-sum constraint: scale down if left + right > 0.95
        for pair in (("left_art", "right_art"),
                     ("left_cross_nb", "right_cross_nb"),
                     ("left_cross_sb", "right_cross_sb")):
            l = getattr(self, pair[0]); r = getattr(self, pair[1])
            if l + r > TURN_PAIR_SUM_MAX:
                scale = TURN_PAIR_SUM_MAX / (l + r)
                setattr(self, pair[0], float(l * scale))
                setattr(self, pair[1], float(r * scale))

    def copy(self):
        return deepcopy(self)

    # -------------------------------------------------------------------------
    # SUMO trips XML generation
    # -------------------------------------------------------------------------
    def to_flow_xml(self, boundaries_json, out_file, probe_prob,
                    probe_prob_cross=None, out_file_probe=None, seed=42):
        """Generate trips XML with turning movements + probe/regular split.

        Same signature as FlowParameters.to_flow_xml so SimulationRunner can
        call either interchangeably. Writes flows sorted by `begin` to avoid
        SUMO's incremental-load truncation bug.
        """
        if probe_prob_cross is None:
            probe_prob_cross = probe_prob

        with open(boundaries_json) as f:
            bnd = json.load(f)

        # Lookup tables
        eb_exit = bnd["arterial"]["uni_east_exit_EB"]
        wb_exit = bnd["arterial"]["uni_west_exit_WB"]
        EB_EXIT_EDGE = eb_exit["edge"]
        WB_EXIT_EDGE = wb_exit["edge"]
        EB_EXIT_APS  = eb_exit.get("approach_edges", [])
        WB_EXIT_APS  = wb_exit.get("approach_edges", [])

        nb_exits, sb_exits = {}, {}
        for cross, info in bnd["cross"].items():
            if "north_exit_NB" in info:
                nb_exits[cross] = {"edge": info["north_exit_NB"]["edge"],
                                   "aps":  info["north_exit_NB"].get("approach_edges", []),
                                   "arterial_turn_exit": info["north_exit_NB"].get(
                                       "arterial_turn_exit", True)}
            if "south_exit_SB" in info:
                sb_exits[cross] = {"edge": info["south_exit_SB"]["edge"],
                                   "aps":  info["south_exit_SB"].get("approach_edges", []),
                                   "arterial_turn_exit": info["south_exit_SB"].get(
                                       "arterial_turn_exit", True)}
        # Some synthetic cross-street flows are added only to exercise a local
        # side-street phase.  They must not silently change the existing arterial
        # turning-destination distribution.
        NB_CROSSES = [c for c in CORRIDOR_ORDER
                      if c in nb_exits and nb_exits[c]["arterial_turn_exit"]]
        SB_CROSSES = [c for c in CORRIDOR_ORDER
                      if c in sb_exits and sb_exits[c]["arterial_turn_exit"]]

        def lookup_entry(bkey):
            if bkey in bnd["arterial"]:
                info = bnd["arterial"][bkey]
                return info["edge"], info.get("approach_edges", []), \
                       _direction(bkey), None
            for cross, cinfo in bnd["cross"].items():
                for role, detail in cinfo.items():
                    if f"{cross}_{role}" == bkey:
                        return detail["edge"], detail.get("approach_edges", []), \
                               _direction(bkey), cross
            return None, [], None, None

        # Turning ratios
        p_left_art    = self.left_art
        p_right_art   = self.right_art
        p_str_art     = max(0.0, 1.0 - p_left_art  - p_right_art)
        # cross-street turn fractions by entry direction (NB vs SB)
        cross_turns = {
            "NB": (self.left_cross_nb, self.right_cross_nb),
            "SB": (self.left_cross_sb, self.right_cross_sb),
        }

        # Collect all flow records, then sort by begin before writing
        records = []  # (begin, kind, flow_id, from, to, via, vph_total, is_arterial)
        start_h, end_h = self.calibration_hours

        for entry_key in sorted(self.flows.keys()):
            from_edge, entry_aps, direction, cross = lookup_entry(entry_key)
            if from_edge is None:
                continue
            is_art = _is_arterial(direction)

            for hour in range(start_h, end_h):
                vph = self.flows[entry_key].get(hour, 0)
                if vph <= 0:
                    continue
                begin = (hour - start_h) * 3600
                end   = begin + 3600
                fid   = f"{entry_key}_h{hour}"

                if is_art:
                    if direction == "EB":
                        st_edge, st_aps = EB_EXIT_EDGE, EB_EXIT_APS
                        left_pool, right_pool = NB_CROSSES, SB_CROSSES
                        l_exits, r_exits = nb_exits, sb_exits
                    else:
                        st_edge, st_aps = WB_EXIT_EDGE, WB_EXIT_APS
                        left_pool, right_pool = SB_CROSSES, NB_CROSSES
                        l_exits, r_exits = sb_exits, nb_exits

                    # Straight
                    via = _build_straight_via(entry_aps, st_aps,
                                              from_edge, st_edge)
                    records.append((begin, "str", f"{fid}_str",
                                    from_edge, st_edge, via,
                                    vph * p_str_art, True))
                    # Left turns — equal share per exit, no via
                    if left_pool and p_left_art > 0:
                        v_each = vph * p_left_art / len(left_pool)
                        for c in left_pool:
                            records.append((begin, "L", f"{fid}_L_{c}",
                                            from_edge, l_exits[c]["edge"],
                                            "", v_each, True))
                    # Right turns
                    if right_pool and p_right_art > 0:
                        v_each = vph * p_right_art / len(right_pool)
                        for c in right_pool:
                            records.append((begin, "R", f"{fid}_R_{c}",
                                            from_edge, r_exits[c]["edge"],
                                            "", v_each, True))
                else:
                    # Cross-street
                    if direction == "NB":
                        if cross not in nb_exits:
                            continue
                        st_edge = nb_exits[cross]["edge"]
                        l_edge, r_edge = WB_EXIT_EDGE, EB_EXIT_EDGE
                    else:
                        if cross not in sb_exits:
                            continue
                        st_edge = sb_exits[cross]["edge"]
                        l_edge, r_edge = EB_EXIT_EDGE, WB_EXIT_EDGE

                    p_left_cross, p_right_cross = cross_turns[direction]
                    p_str_cross = max(0.0, 1.0 - p_left_cross - p_right_cross)
                    # Straight (no via)
                    if from_edge != st_edge:
                        records.append((begin, "str", f"{fid}_str",
                                        from_edge, st_edge, "",
                                        vph * p_str_cross, False))
                    # Left turn
                    records.append((begin, "L", f"{fid}_L",
                                    from_edge, l_edge, "",
                                    vph * p_left_cross, False))
                    # Right turn
                    records.append((begin, "R", f"{fid}_R",
                                    from_edge, r_edge, "",
                                    vph * p_right_cross, False))

        # Build XML root
        root = ET.Element("routes")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("xsi:noNamespaceSchemaLocation",
                 "http://sumo.dlr.de/xsd/routes_file.xsd")
        ET.SubElement(root, "vType", id="probe_vehicle",   vClass="passenger")
        ET.SubElement(root, "vType", id="regular_vehicle", vClass="passenger")

        # Sort by begin (then by id for stability) — bypasses SUMO's
        # incremental-load truncation bug
        records.sort(key=lambda r: (r[0], r[2]))

        for begin, kind, fid, from_edge, to_edge, via, vph_total, is_art in records:
            if from_edge == to_edge:
                continue
            vph_int = int(round(vph_total))
            if vph_int <= 0:
                continue
            end = begin + 3600
            p = probe_prob if is_art else probe_prob_cross
            v_probe = max(0, int(round(vph_int * p)))
            v_reg = vph_int - v_probe

            common = {"from": from_edge, "to": to_edge,
                      "begin": str(begin), "end": str(end),
                      "departLane": "best", "departSpeed": "max"}
            if via:
                common["via"] = via

            if v_reg > 0:
                ET.SubElement(root, "flow",
                              id=f"{fid}_reg", type="regular_vehicle",
                              attrib={**common,
                                      "vehsPerHour": str(v_reg)})
            if v_probe > 0:
                ET.SubElement(root, "flow",
                              id=f"{fid}_probe", type="probe_vehicle",
                              attrib={**common,
                                      "vehsPerHour": str(v_probe)})

        ET.indent(root, space="  ")
        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
        target = out_file_probe if out_file_probe else out_file
        ET.ElementTree(root).write(target, encoding="utf-8",
                                    xml_declaration=True)
        return target

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    def summary(self):
        lines = [
            f"Turning ratios:",
            f"  left_art    = {self.left_art:.3f}",
            f"  right_art   = {self.right_art:.3f}",
            f"  left_cross_nb  = {self.left_cross_nb:.3f}  right_cross_nb = {self.right_cross_nb:.3f}",
            f"  left_cross_sb  = {self.left_cross_sb:.3f}  right_cross_sb = {self.right_cross_sb:.3f}",
            f"  → arterial straight = {1 - self.left_art - self.right_art:.3f}",
            f"\nFlows (vph per boundary per hour):",
        ]
        start, end = self.calibration_hours
        for k in sorted(self.flows.keys()):
            per_hour = [self.flows[k].get(h, 0) for h in range(start, end)]
            lines.append(f"  {k}: total={sum(per_hour)}, per_hour={per_hour}")
        return "\n".join(lines)

    def __repr__(self):
        n_vars = len(self._ordered_flow_keys()) + 6
        return (f"TurningParameters({len(self.flows)} boundaries, "
                f"{n_vars} vars [{n_vars-6} flows + 6 turning], "
                f"hours={self.calibration_hours}, "
                f"L_art={self.left_art:.2f} R_art={self.right_art:.2f} "
                f"cross_nb(L={self.left_cross_nb:.2f},R={self.right_cross_nb:.2f}) "
                f"cross_sb(L={self.left_cross_sb:.2f},R={self.right_cross_sb:.2f}))")


if __name__ == "__main__":
    p = TurningParameters.from_fringe_flows_csv()
    print(p)
    print()
    print(p.summary()[:500])
    print(f"\nVector length: {len(p.to_vector())}")
    print(f"Last 4 (turning ratios): {p.to_vector()[-4:]}")

    # Sanity: write a test trips XML
    out = "/tmp/test_turning.xml"
    p.to_flow_xml(DEFAULT_BOUNDARY_JSON, out_file=out,
                  probe_prob=0.3478, probe_prob_cross=0.75)
    import subprocess
    n = subprocess.check_output(
        ["grep", "-c", "<flow", out], text=True).strip()
    print(f"Generated {n} flow entries in {out}")

"""
Demand generation for randomized-demand RL training.

Each demand realization is parameterized by 4 controllable knobs:
  global_scale      : overall traffic level (multiplies every flow)
  main_mult         : average mainline amplification
  side_mult         : side-street amplification
  direction_balance : allocation of the mainline multiplier to eastbound and
                      westbound traffic; 0.5 is balanced and larger values favor EB

A flow's hourly rate is rescaled by
    EB:   base_vph * global_scale * main_mult * 2 * direction_balance
    WB:   base_vph * global_scale * main_mult * 2 * (1 - direction_balance)
    side: base_vph * global_scale * side_mult

Mainline flows are exactly those whose id starts with 'uni_' (the two corridor-end
entries that travel EB/WB along University Ave); everything else is a cross-street
entry. Because the demand is encoded as <flow vehsPerHour=...> (not individual
<trip>s), scaling is a pure multiply on the rate -- no per-vehicle resampling.

For the standard experiments, the generated route file retains SUMO ``<flow>``
elements.  EXP5 can instead set ``RL_EXPLICIT_DEMAND=1`` and obtain a fixed list
of explicit vehicles for each (global, main, side, direction) scenario. Across probe
penetration levels, those vehicles keep the same ids, departure times, origins,
destinations, and routes; only their otherwise-identical SUMO type is relabeled.
"""
import copy
import hashlib
import math
import os
import xml.etree.ElementTree as ET

from . import config as C

BASE_TRIPS = C.TRIPS

# Common parameter domain for LHS training and evaluation scenario designs.
DEFAULT_RANGES = {
    "global": (0.8, 1.3),
    "main": (0.8, 1.3),
    "side": (0.8, 1.3),
    "direction": (0.2, 0.8),
}

# parse the calibrated base flows ONCE: (id, from, base_vph, is_mainline)
_BASE = None
_EXPLICIT_BASE = None

_PROBE_SUFFIX = "_probe"
_REGULAR_SUFFIX = "_reg"
_FLOW_ONLY_ATTRIBUTES = {
    "begin", "end", "vehsPerHour", "period", "probability", "number",
    "personsPerHour", "containerNumber", "personNumber",
}


def is_mainline(flow_id):
    return flow_id.startswith("uni_")


def mainline_direction(flow_id):
    """Return ``EB``/``WB`` for a mainline entry flow, else ``None``.

    The calibrated demand uses west-corridor entries for eastbound traffic and
    east-corridor entries for westbound traffic. Failing loudly on an unknown
    ``uni_`` flow prevents a new mainline movement from silently receiving the
    wrong multiplier.
    """
    movement = _movement_id(flow_id)
    if not is_mainline(movement):
        return None
    if movement.startswith("uni_west_entry_EB_"):
        return "EB"
    if movement.startswith("uni_east_entry_WB_"):
        return "WB"
    raise ValueError(f"Cannot determine mainline direction from flow id: {flow_id}")


def _validate_direction_balance(direction_balance):
    value = float(direction_balance)
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"direction_balance must be in [0, 1], got {direction_balance}")
    return value


def demand_multiplier(flow_id, main_mult, side_mult, direction_balance=0.5):
    """Return the non-global multiplier for one calibrated movement."""
    direction = mainline_direction(flow_id)
    if direction is None:
        return float(side_mult)
    balance = _validate_direction_balance(direction_balance)
    directional_weight = 2.0 * (
        balance if direction == "EB" else 1.0 - balance)
    return float(main_mult) * directional_weight


def explicit_demand_enabled():
    """Whether route generation should materialize fixed individual vehicles."""
    return os.environ.get("RL_EXPLICIT_DEMAND", "0") == "1"


def probe_penetration():
    """Probe fraction used only by the explicit-demand EXP5 workflow."""
    value = float(os.environ.get("RL_PROBE_PENETRATION", "0.1"))
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"RL_PROBE_PENETRATION must be in [0, 1], got {value}")
    return value


def explicit_demand_seed():
    """Common realization seed, deliberately independent of rollout worker id."""
    return int(os.environ.get("RL_EXPLICIT_DEMAND_SEED", "42"))


def probe_membership_seed():
    """Probe-label seed, separated from the physical vehicle realization seed.

    By default this equals ``RL_EXPLICIT_DEMAND_SEED`` for backward compatibility.
    The penetration-estimator experiment varies only this seed while holding vehicle
    ids, departures, origins, destinations, and routes fixed.
    """
    return int(os.environ.get(
        "RL_PROBE_MEMBERSHIP_SEED", str(explicit_demand_seed())))


def probe_membership_score(vehicle_id, params, seed=None):
    """Stable U[0,1) vehicle-level probe score for nested penetration samples."""
    if seed is None:
        seed = probe_membership_seed()
    return _unit_interval("probe", int(seed), param_key(params), vehicle_id)


def _unit_interval(*parts):
    """Stable U[0,1) variate that is independent of Python hash randomization."""
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    raw = hashlib.sha256(payload).digest()[:8]
    return int.from_bytes(raw, "big") / float(1 << 64)


def _movement_id(flow_id):
    if flow_id.endswith(_PROBE_SUFFIX):
        return flow_id[:-len(_PROBE_SUFFIX)]
    if flow_id.endswith(_REGULAR_SUFFIX):
        return flow_id[:-len(_REGULAR_SUFFIX)]
    return flow_id


def _load_base(base=BASE_TRIPS):
    """Cache the base flow rates so generation is a cheap multiply + write."""
    global _BASE
    if _BASE is None or _BASE[0] != base:
        tree = ET.parse(base)
        rows = [(fl.get("id"), float(fl.get("vehsPerHour", 0.0)),
                 is_mainline(fl.get("id"))) for fl in tree.findall(".//flow")]
        _BASE = (base, tree, rows)
    return _BASE


def _load_explicit_base(base=BASE_TRIPS):
    """Merge each calibrated ``_probe``/``_reg`` flow pair into one movement.

    The calibrated XML already partitions the same physical demand into probe and
    regular flows.  Their rates must be recombined before individual vehicles are
    generated; otherwise changing penetration would change departure processes.
    """
    global _EXPLICIT_BASE
    if _EXPLICIT_BASE is not None and _EXPLICIT_BASE[0] == base:
        return _EXPLICIT_BASE

    tree = ET.parse(base)
    root = tree.getroot()
    groups = {}
    order = []
    for flow in root.findall("flow"):
        movement = _movement_id(flow.get("id", ""))
        if movement not in groups:
            groups[movement] = {
                "template": copy.deepcopy(flow),
                "base_vph": 0.0,
            }
            order.append(movement)
        groups[movement]["base_vph"] += float(
            flow.get("vehsPerHour", 0.0))

    movements = []
    for movement in order:
        item = groups[movement]
        template = item["template"]
        template.set("id", movement)
        template.attrib.pop("type", None)
        movements.append((movement, item["base_vph"], template))
    vtypes = [copy.deepcopy(node) for node in root.findall("vType")]
    _EXPLICIT_BASE = (base, root.tag, dict(root.attrib), vtypes, movements)
    return _EXPLICIT_BASE


def generate_trips(out_path, global_scale=1.0, main_mult=1.0, side_mult=1.0,
                   direction_balance=0.5, base=BASE_TRIPS):
    """Write a four-parameter-scaled copy of the base flows to ``out_path``.

    ``direction_balance=0.5`` reproduces the former direction-neutral behavior.
    Returns out_path. Reuses the cached base tree (only vehsPerHour is rewritten).
    """
    _b, tree, rows = _load_base(base)
    flows = tree.findall(".//flow")
    for fl, (_id, vph, _is_main) in zip(flows, rows):
        mult = demand_multiplier(
            _id, main_mult, side_mult, direction_balance)
        fl.set("vehsPerHour", f"{vph * global_scale * mult:.4f}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def generate_explicit_trips(out_path, global_scale=1.0, main_mult=1.0,
                            side_mult=1.0, direction_balance=0.5,
                            penetration=None, seed=None,
                            probe_seed=None, base=BASE_TRIPS):
    """Write one fixed explicit-vehicle realization for an EXP5 scenario.

    The realization hash depends on the demand parameters and ``seed``, but not on
    ``penetration``.  Probe membership uses a second stable hash and is nested, so
    every 5% probe also appears in the 10%, 20%, 50%, and 100% samples. Thus the output
    files differ only in each trip's ``type`` attribute.
    """
    if penetration is None:
        penetration = probe_penetration()
    penetration = float(penetration)
    if not 0.0 <= penetration <= 1.0:
        raise ValueError(f"penetration must be in [0, 1], got {penetration}")
    if seed is None:
        seed = explicit_demand_seed()
    if probe_seed is None:
        probe_seed = probe_membership_seed()

    _base, root_tag, root_attrib, vtypes, movements = _load_explicit_base(base)
    root = ET.Element(root_tag, root_attrib)
    for vtype in vtypes:
        root.append(copy.deepcopy(vtype))

    params = (global_scale, main_mult, side_mult, direction_balance)
    scenario = param_key(params)
    trips = []
    for movement, base_vph, template in movements:
        begin = float(template.get("begin", 0.0))
        end = float(template.get("end", begin))
        duration = end - begin
        if duration <= 0:
            continue
        multiplier = demand_multiplier(
            movement, main_mult, side_mult, direction_balance)
        expected = base_vph * global_scale * multiplier * duration / 3600.0
        count = int(math.floor(expected))
        if _unit_interval("count", seed, scenario, movement) < expected - count:
            count += 1
        if count == 0:
            continue

        trip_attrib = {
            key: value for key, value in template.attrib.items()
            if key not in _FLOW_ONLY_ATTRIBUTES and key not in {"id", "type"}
        }
        for index in range(count):
            vehicle_id = f"{movement}_veh{index:05d}"
            # One jittered departure within each equal-width stratum prevents the
            # synchronized bursts caused by independently split SUMO flows.
            jitter = _unit_interval(
                "depart", seed, scenario, vehicle_id)
            depart = begin + ((index + jitter) / count) * duration
            attrib = dict(trip_attrib)
            attrib.update({
                "id": vehicle_id,
                "depart": f"{depart:.2f}",
                "type": (
                    "probe_vehicle"
                    if probe_membership_score(
                        vehicle_id,
                        params,
                        seed=probe_seed)
                    < penetration
                    else "regular_vehicle"
                ),
            })
            trips.append((depart, vehicle_id, ET.Element("trip", attrib)))

    for _depart, _vehicle_id, trip in sorted(
            trips, key=lambda item: (item[0], item[1])):
        root.append(trip)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ET.ElementTree(root).write(
        out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def sample_params(rng, ranges=DEFAULT_RANGES):
    """Sample a (global, main, side, direction) demand tuple."""
    return (round(float(rng.uniform(*ranges["global"])), 3),
            round(float(rng.uniform(*ranges["main"])), 3),
            round(float(rng.uniform(*ranges["side"])), 3),
            round(float(rng.uniform(*ranges["direction"])), 3))


def param_key(p):
    if len(p) != 4:
        raise ValueError(
            "Demand parameters must be (global, main, side, direction); "
            f"received {p}")
    return (f"g{p[0]:.3f}_m{p[1]:.3f}_s{p[2]:.3f}_"
            f"d{p[3]:.3f}")

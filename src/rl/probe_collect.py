"""
Probe-vehicle drop-in for tsc_sumo_sim.setup_subscriptions / collect_step_data.

**Same output contract** as fast_collect.collect_step_data (per-segment probe
travel-time speed samples in `speed_buffer`), but it flips the per-step question:

  fast_collect : "for each of ~1490 induction loops, which vehicles are on it?"
                 -> SUMO serializes ~1490 subscription objects every step
                 (~16 ms/step = 82% of the whole step; profiled, scales with the
                 number of subscribed objects, not with traffic).

  probe_collect: "for each of the ~60-100 probe vehicles on the net, what lane +
                 lane-position is it at?"  -> ~60-100 subscription objects/step.
                 Object count drops ~15-25x -> the per-step marshalling collapses.

Equivalence principle
---------------------
A v8 induction loop is just a point (lane, pos). "Probe fired the loop this step"
== "the probe's front crossed `pos` on `lane` this step". Given each probe's
(lane, pos) sampled per step we reconstruct the SAME entry/exit crossings:
  - crossing a segment ENTRY (START) boundary -> record entry_time
  - crossing the matching EXIT  (END)   boundary -> travel_time = now - entry_time,
    speed = distance / travel_time  (clamped to MAX_SPEED), appended to the buffer.

The two-pass (all EXIT boundaries before all ENTRY boundaries) and the per-segment
`entries.pop` / `not in entries` dedup are kept identical to the reference, so a
probe seen at a segment's entry and exit in the SAME step is not closed that step,
and repeated hits across a segment's multiple lanes stay idempotent.

Timing matches the reference convention exactly: entry recorded at
`sim_time - step_length` (start of the crossing step), exit travel time measured to
`sim_time` (end of the crossing step).

This is an APPROXIMATION of the loop method (positions are sampled each step, so a
crossing is attributed to a step rather than to the exact sub-step instant). Its
fidelity is validated against the loop method / detector_full_v8fixed.csv by
src/rl/validate_probe_collect.py. To A/B in-process, point corridor_env's import
back at fast_collect (or set RL_COLLECTOR=fast).
"""
import traci
from tsc_sumo_sim import MAX_SPEED

_VLANE = traci.constants.VAR_LANE_ID
_VPOS = traci.constants.VAR_LANEPOSITION
_DEP = traci.constants.VAR_DEPARTED_VEHICLES_IDS
_TVAR = traci.constants.VAR_TELEPORT_STARTING_VEHICLES_IDS

# Module state is keyed by the cached segment-list identity. Static probe geometry
# therefore survives reset() while per-vehicle history is cleared each episode.
_state = {
    "key": None, "prev": {}, "probes": set(),
    "entry_map": {}, "exit_map": {},
    "traversal_observer": None,
}


def _is_probe(vehicle_id):
    """Identify probes by vType so vehicle ids can stay fixed across EXP5."""
    return traci.vehicle.getTypeID(vehicle_id) == "probe_vehicle"


def _edge(lane):
    """Edge id of a SUMO lane id `{edge}_{laneIndex}`."""
    i = lane.rfind("_")
    return lane[:i] if i > 0 else lane


def _build_lane_maps(segments):
    """{lane_id: [(pos, seg_idx), ...]} for entry and exit boundaries.

    Probe-mode segments carry points parsed from the detector XML. The TraCI
    fallback keeps this collector usable with legacy callers.
    """
    entry_map, exit_map = {}, {}
    for idx, seg in enumerate(segments):
        entry_points = seg.get("entry_points")
        if entry_points is None:
            entry_points = tuple(
                (traci.inductionloop.getLaneID(d),
                 traci.inductionloop.getPosition(d))
                for d in seg["entry"]
            )
        exit_points = seg.get("exit_points")
        if exit_points is None:
            exit_points = tuple(
                (traci.inductionloop.getLaneID(d),
                 traci.inductionloop.getPosition(d))
                for d in seg["exit"]
            )
        for lane, pos in entry_points:
            entry_map.setdefault(lane, []).append((pos, idx))
        for lane, pos in exit_points:
            exit_map.setdefault(lane, []).append((pos, idx))
    return entry_map, exit_map


def setup_subscriptions(segments=None):
    """Subscribe the simulation domain (newly-departed + teleporting lists) and any
    probes already on the net. Per-vehicle (lane, pos) subscriptions are added as
    probes depart; SUMO auto-drops them when a vehicle leaves."""
    traci.simulation.subscribe([_DEP, _TVAR])
    probes = set()
    for v in traci.vehicle.getIDList():
        if _is_probe(v):
            traci.vehicle.subscribe(v, [_VLANE, _VPOS])
            probes.add(v)
    if segments is None:
        _state["key"] = None
    elif _state["key"] is not segments:
        em, xm = _build_lane_maps(segments)
        _state.update(key=segments, entry_map=em, exit_map=xm)
    _state["prev"] = {}           # veh -> (lane, pos) at the previous step
    _state["probes"] = probes


def set_traversal_observer(observer=None):
    """Optionally receive every completed segment traversal.

    The callback signature is ``(segment_index, vehicle_id, entry_time_s,
    exit_time_s, travel_time_s, speed_kph)``. Normal training leaves this unset;
    the penetration-estimator experiment enables it while every physical vehicle
    is observable, allowing an exact sum of individual segment travel-time delay.
    """
    _state["traversal_observer"] = observer


def _crossed(plane, ppos, clane, cpos, lane_map):
    """Segment indices whose boundary (in lane_map) the probe crossed this step,
    going from (plane, ppos) to (clane, cpos). Windows are deliberately generous at
    lane/edge changes (over-triggering a segment is idempotent via the per-segment
    dedup; missing a crossing would lose a sample)."""
    if plane is None:                                  # first sighting after depart
        return [seg for pos, seg in lane_map.get(clane, ()) if pos <= cpos]
    if plane == clane:                                 # same lane: pos is monotone
        return [seg for pos, seg in lane_map.get(clane, ()) if ppos < pos <= cpos]
    same_edge = _edge(plane) == _edge(clane)
    out = []
    hi = cpos if same_edge else float("inf")           # old lane: passed downstream
    out += [seg for pos, seg in lane_map.get(plane, ()) if ppos < pos <= hi]
    lo = ppos if same_edge else float("-inf")          # new lane: up to current pos
    out += [seg for pos, seg in lane_map.get(clane, ()) if lo < pos <= cpos]
    return out


def collect_step_data(segments, veh_entry_times, speed_buffer,
                      sim_time=None, step_length=None):
    """Drop-in for fast_collect.collect_step_data (identical output contract).

    See module docstring. `sim_time` / `step_length` are passed in by the caller's
    hot loop to avoid a getTime()/getDeltaT() round trip; omitted -> queried.
    """
    if _state["key"] is not segments:                  # new episode -> rebuild maps
        em, xm = _build_lane_maps(segments)
        _state.update(key=segments, entry_map=em, exit_map=xm)

    sim_sub = traci.simulation.getSubscriptionResults() or {}
    probes = _state["probes"]
    for v in sim_sub.get(_DEP, ()):                     # subscribe freshly-departed
        if _is_probe(v):
            traci.vehicle.subscribe(v, [_VLANE, _VPOS])
            probes.add(v)
    teleporting = set(sim_sub.get(_TVAR, ()))
    if sim_time is None:
        sim_time = traci.simulation.getTime()
    if step_length is None:
        step_length = traci.simulation.getDeltaT()

    prev = _state["prev"]
    em, xm = _state["entry_map"], _state["exit_map"]
    res = traci.vehicle.getAllSubscriptionResults()

    # one geometric pass over the ~60-100 subscribed probes -> crossing events
    exit_events, entry_events, seen = [], [], set()
    for veh, d in res.items():
        if veh not in probes:
            continue
        clane = d.get(_VLANE)
        cpos = d.get(_VPOS)
        if clane is None or cpos is None:
            continue
        if veh in teleporting:                         # drop state; emit nothing
            prev.pop(veh, None)
            probes.discard(veh)
            for e in veh_entry_times:
                e.pop(veh, None)
            continue
        seen.add(veh)
        p = prev.get(veh)
        plane, ppos = p if p else (None, None)
        for seg in _crossed(plane, ppos, clane, cpos, xm):
            exit_events.append((seg, veh))
        for seg in _crossed(plane, ppos, clane, cpos, em):
            entry_events.append((seg, veh))
        prev[veh] = (clane, cpos)

    # forget probes that left the network (entry records left as-is, like the ref)
    if len(prev) > len(seen):
        for veh in [v for v in prev if v not in seen]:
            prev.pop(veh, None)
            probes.discard(veh)

    # Pass 1: exit boundaries complete trips (strictly before entries).
    for idx, veh in exit_events:
        entry_t = veh_entry_times[idx].pop(veh, None)
        if entry_t is not None:
            travel_time = sim_time - entry_t
            if travel_time > 0:
                speed_kph = (segments[idx]["distance"] / travel_time) * 3.6
                if speed_kph > MAX_SPEED:
                    speed_kph = MAX_SPEED
                if speed_kph > 0:
                    speed_buffer[idx].append(speed_kph)
                    observer = _state.get("traversal_observer")
                    if observer is not None:
                        observer(
                            idx, veh, float(entry_t), float(sim_time),
                            float(travel_time), float(speed_kph))

    # Pass 2: entry boundaries record entry time.
    entry_t0 = sim_time - step_length
    for idx, veh in entry_events:
        entries = veh_entry_times[idx]
        if veh not in entries:
            entries[veh] = entry_t0

"""
Faster drop-in for tsc_sumo_sim.setup_subscriptions / collect_step_data, used by
the RL env's hot loop. **Output is identical** to the originals — only the
per-step work is reduced:

  1. teleport list via a simulation-domain subscription (pushed with the step
     response, read from the client cache -> no per-step round trip);
  2. sim_time / step_length passed in by the caller -> no getTime()/getDeltaT()
     round trip per step;
  3. the per-step segment scan is **inverted**: instead of scanning all ~1600
     segments every step (the original did this regardless of traffic), we touch
     only the detectors that actually fired this step, via a cached
     detector->segment map. Per-step cost now scales with real probe traffic.

The original `tsc_sumo_sim` functions are intentionally left untouched as a
fallback — swap the import in corridor_env back to `tsc_sumo_sim` to restore the
reference behavior if anything looks off.

Equivalence notes (why output matches the original exactly):
  - Two passes are kept (all EXIT boundaries before all ENTRY boundaries) so a
    probe seen at a segment's entry and exit in the SAME step is not closed that
    step (would otherwise fabricate a one-step travel time).
  - `entries.pop(veh_id, None)` / `if veh_id not in entries` make repeated hits of
    the same probe on multiple detectors of one segment idempotent, matching the
    original's per-segment set() dedup.
"""
import os
import traci
from tsc_sumo_sim import MAX_SPEED

_TVAR = traci.constants.VAR_TELEPORT_STARTING_VEHICLES_IDS
_VID = traci.constants.LAST_STEP_VEHICLE_ID_LIST

# Safety net: set RL_REFERENCE_COLLECT=1 (or toggle this flag) to run the original
# scan-all-segments algorithm instead of the inverted-loop fast path, while still
# accepting the cheap sim_time/step_length/teleport-subscription preamble. Lets you
# A/B the two in-process if the fast path is ever suspected.
USE_REFERENCE = os.environ.get("RL_REFERENCE_COLLECT") == "1"

# single-entry cache of the detector->segment maps, keyed by the `segments` object
# identity (rebuilt automatically when a new episode rebuilds the segment list).
_map_cache = {"key": None, "entry": None, "exit": None}


def _det_segment_maps(segments):
    """{detector_id: [segment_idx, ...]} for entry and exit boundaries."""
    if _map_cache["key"] is segments:
        return _map_cache["entry"], _map_cache["exit"]
    entry_map, exit_map = {}, {}
    for idx, seg in enumerate(segments):
        for d in seg["entry"]:
            entry_map.setdefault(d, []).append(idx)
        for d in seg["exit"]:
            exit_map.setdefault(d, []).append(idx)
    _map_cache.update(key=segments, entry=entry_map, exit=exit_map)
    return entry_map, exit_map


def setup_subscriptions():
    """Subscribe segLoop detectors (vehicle-id list) + the starting-teleport list."""
    for det_id in traci.inductionloop.getIDList():
        if det_id.startswith("segLoop_"):
            traci.inductionloop.subscribe(det_id, [_VID])
    # starting-teleport list rides the step response -> no round trip in the loop
    traci.simulation.subscribe([_TVAR])


def collect_step_data(segments, veh_entry_times, speed_buffer,
                      sim_time=None, step_length=None):
    """v8 endpoint collection (inverted-loop, fired-detectors-only variant).

    Parameters
    ----------
    segments : list of segment dicts ('entry'/'exit' detector-id lists, 'distance')
    veh_entry_times : list[dict] indexed by segment: {veh_id: entry_time}
    speed_buffer : dict {segment_index: [speeds]}
    sim_time, step_length : pass them in from the caller's hot loop to avoid a
        getTime()/getDeltaT() round trip each step; omitted -> queried (original
        behavior).
    """
    all_results = traci.inductionloop.getAllSubscriptionResults()

    # teleport list from the simulation-domain subscription cache (no round trip);
    # fall back to an explicit query if the subscription was not set up.
    sim_sub = traci.simulation.getSubscriptionResults()
    if sim_sub and _TVAR in sim_sub:
        teleporting = set(sim_sub[_TVAR])
    else:
        teleporting = set(traci.simulation.getStartingTeleportIDList())
    if sim_time is None:
        sim_time = traci.simulation.getTime()
    if step_length is None:
        step_length = traci.simulation.getDeltaT()

    if USE_REFERENCE:
        _collect_reference(segments, veh_entry_times, speed_buffer,
                           all_results, teleporting, sim_time, step_length)
        return

    entry_map, exit_map = _det_segment_maps(segments)

    # resolve the probe set on each FIRED detector once; skip empty detectors
    # (the common case every step) with no per-segment work.
    fired = {}
    for det_id, res in all_results.items():
        vlist = res.get(_VID, ())
        if not vlist:
            continue
        probes = [v for v in vlist if "_probe" in v and v not in teleporting]
        if probes:
            fired[det_id] = probes
    if not fired:
        return

    # Pass 1: exit boundaries complete trips (strictly before entries).
    for det_id, probes in fired.items():
        for idx in exit_map.get(det_id, ()):
            seg = segments[idx]
            entries = veh_entry_times[idx]
            buf = speed_buffer[idx]
            dist = seg["distance"]
            for veh_id in probes:
                entry_t = entries.pop(veh_id, None)
                if entry_t is not None:
                    travel_time = sim_time - entry_t
                    if travel_time > 0:
                        speed_kph = (dist / travel_time) * 3.6
                        if speed_kph > MAX_SPEED:
                            speed_kph = MAX_SPEED
                        if speed_kph > 0:
                            buf.append(speed_kph)

    # Pass 2: entry boundaries record entry time.
    entry_t0 = sim_time - step_length
    for det_id, probes in fired.items():
        for idx in entry_map.get(det_id, ()):
            entries = veh_entry_times[idx]
            for veh_id in probes:
                if veh_id not in entries:
                    entries[veh_id] = entry_t0


def _collect_reference(segments, veh_entry_times, speed_buffer,
                       all_results, teleporting, sim_time, step_length):
    """Reference algorithm: the original scan-all-segments loop from
    tsc_sumo_sim.collect_step_data, byte-for-byte in its segment processing. Used
    only when USE_REFERENCE is set, to A/B against the fast path."""
    def probes_on(dets):
        vs = set()
        for d in dets:
            for veh_id in all_results.get(d, {}).get(_VID, ()):
                if "_probe" in veh_id and veh_id not in teleporting:
                    vs.add(veh_id)
        return vs

    for idx, seg in enumerate(segments):
        entries = veh_entry_times[idx]
        for veh_id in probes_on(seg["exit"]):
            if veh_id in entries:
                travel_time = sim_time - entries.pop(veh_id)
                if travel_time > 0:
                    speed_kph = (seg["distance"] / travel_time) * 3.6
                    if speed_kph > MAX_SPEED:
                        speed_kph = MAX_SPEED
                    if speed_kph > 0:
                        speed_buffer[idx].append(speed_kph)
        for veh_id in probes_on(seg["entry"]):
            if veh_id not in entries:
                entries[veh_id] = sim_time - step_length

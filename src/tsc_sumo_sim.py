import os
import sys
import csv
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import pickle
import json
from collections import defaultdict
# ---------------------------------------------------------------------
# Make sure SUMO tools are available
# ---------------------------------------------------------------------
if "SUMO_HOME" not in os.environ:
    raise EnvironmentError("SUMO_HOME is not set. Please set it to your SUMO installation path.")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci
from sumolib import checkBinary
from mainline_time_space import MainlineTimeSpaceRecorder

INTERVAL = 900
PERCENTILES = np.arange(0, 101, 5)
OUTPUT_CSV = "detector_results.csv"

CSV_FIELDS = [
    "interval_begin", "interval_end", "detector_id", "approach_id",
    "vehicle_count",  "mean_speed_kph",
    "harmonic_mean_speed_kph",
    "speed_percentiles",
]

MAX_SPEED = 70 # kph

# v8 endpoint detector convention (detectors_v8_fixed.add.xml):
#   START detectors: segLoop_{sid}_<lane>        (entry boundary of segment)
#   END   detectors: segLoop_{sid}_END_<lane>    (exit boundary of segment)
# Travel time is measured across each TomTom segment's OWN two boundaries, so
# the measured distance matches TomTom's per-segment definition (vs the old v5
# center-to-center pairing which mixed half of two adjacent segments).
DROP_SIDS = {110, 112, 113, 205, 466, 510}  # boundaries outside SUMO network
MIN_SEG_LEN_M = 5.0  # skip degenerate short segments (boundaries collapse)


def parse_segments_v8(approach_folder):
    """Return list of segments [{sid, approach_id, distance}] from approach CSVs.

    Each TomTom segment is independent — its travel time is measured between its
    own START and END boundary detectors. No up/down chain pairing needed.
    """
    segments = []
    for file in sorted(os.listdir(approach_folder)):
        if not file.endswith("results.csv"):
            continue
        approach_id = os.path.splitext(file)[0].replace("_delay_time_results", "")
        df = pd.read_csv(os.path.join(approach_folder, file))
        df = df[df["timeSet"] == 2].reset_index(drop=True)
        for _, row in df.iterrows():
            sid = int(row["simpleID"])
            if sid in DROP_SIDS:
                continue
            if float(row["distance"]) < MIN_SEG_LEN_M:
                continue
            segments.append({
                "sid": sid,
                "approach_id": approach_id,
                "distance": float(row["distance"]),
            })
    return segments


def build_segment_detectors_v8(segments):
    """Fill each segment with entry (START) and exit (END) detector IDs.

    Must be called AFTER traci.start(). Returns a filtered list dropping any
    segment that lacks either boundary in the loaded detector file.
    """
    sumo_ids = set(traci.inductionloop.getIDList())
    kept = []
    for seg in segments:
        sid = seg["sid"]
        end_prefix = f"segLoop_{sid}_END_"
        start_prefix = f"segLoop_{sid}_"
        exit_d = sorted(d for d in sumo_ids if d.startswith(end_prefix))
        entry = sorted(d for d in sumo_ids
                       if d.startswith(start_prefix) and d not in exit_d)
        if not entry or not exit_d:
            continue
        seg["entry"] = entry
        seg["exit"] = exit_d
        kept.append(seg)
    return kept

def setup_subscriptions():
    """Subscribe to all segLoop detectors once at startup.

    Only LAST_STEP_VEHICLE_ID_LIST is needed: collect_step_data identifies
    probe vehicles by the `_probe` substring in their id (flow ids end in
    `_probe`), so the heavy LAST_STEP_VEHICLE_DATA payload (per-vehicle entry/
    exit times + type for every detector every step) is avoided — that payload
    on 1600+ v8 detectors under peak flow roughly doubled wall-clock time.
    """
    for det_id in traci.inductionloop.getIDList():
        if det_id.startswith("segLoop_"):
            traci.inductionloop.subscribe(
                det_id, [traci.constants.LAST_STEP_VEHICLE_ID_LIST])
      
def collect_step_data(segments, veh_entry_times, speed_buffer):
    """v8 endpoint collection: each probe vehicle's travel time is measured
    between a segment's own START (entry) and END (exit) boundary detectors.

    Parameters
    ----------
    segments : list of dict, each with 'entry'/'exit' detector-id lists (filled
        by build_segment_detectors_v8) and 'distance'.
    veh_entry_times : list of dict (indexed by segment position): {veh: entry_t}
    speed_buffer : dict {segment_index: [speeds]}
    """
    all_results = traci.inductionloop.getAllSubscriptionResults()
    teleporting = set(traci.simulation.getStartingTeleportIDList())
    sim_time = traci.simulation.getTime()
    step_length = traci.simulation.getDeltaT()
    VID = traci.constants.LAST_STEP_VEHICLE_ID_LIST

    def probes_on(dets):
        vs = set()
        for d in dets:
            for veh_id in all_results.get(d, {}).get(VID, ()):
                # probe vehicles have flow ids ending in `_probe` → `_probe`
                # appears in the instance id `{flow_id}.{n}`
                if "_probe" in veh_id and veh_id not in teleporting:
                    vs.add(veh_id)
        return vs

    for idx, seg in enumerate(segments):
        entries = veh_entry_times[idx]
        # Step 1: exit boundary — complete trips
        for veh_id in probes_on(seg["exit"]):
            if veh_id in entries:
                travel_time = sim_time - entries.pop(veh_id)
                if travel_time > 0:
                    speed_kph = (seg["distance"] / travel_time) * 3.6
                    if speed_kph > MAX_SPEED:
                        speed_kph = MAX_SPEED
                    if speed_kph > 0:
                        speed_buffer[idx].append(speed_kph)
        # Step 2: entry boundary — record entry time
        for veh_id in probes_on(seg["entry"]):
            if veh_id not in entries:
                entries[veh_id] = sim_time - step_length

def write_interval_results(writer, interval_begin, interval_end, speed_buffer, segments):
    for idx, seg in enumerate(segments):
        speeds = [s for s in speed_buffer[idx] if s > 0]
        vehicle_count = len(speeds)

        if speeds:
            mean_speed = sum(speeds) / len(speeds)
            harmonic_speed = len(speeds) / sum(1.0 / s for s in speeds)
            percentile_values = [float(np.percentile(speeds, p)) for p in PERCENTILES]
        else:
            mean_speed = -1
            harmonic_speed = -1
            percentile_values = [-1] * len(PERCENTILES)

        writer.writerow({
            "interval_begin":          interval_begin,
            "interval_end":            interval_end,
            "detector_id":             seg["sid"],
            "approach_id":             seg["approach_id"],
            "vehicle_count":           vehicle_count,
            "mean_speed_kph":          round(mean_speed , 1) if mean_speed >= 0 else -1,
            "harmonic_mean_speed_kph":  round(harmonic_speed, 1) if harmonic_speed >= 0 else -1,
            "speed_percentiles":        [round(p, 1) for p in percentile_values],
        })

        speed_buffer[idx].clear()

def save_signal_program():
    """Retrieve signal programs for all traffic lights in the network and save to a pickle file.
    """
    
    signal_ID_list = traci.trafficlight.getIDList()
    
    Signal_Programs = {}
    
    for signal_id in signal_ID_list:
        program = traci.trafficlight.getCompleteRedYellowGreenDefinition(signal_id)
        Signal_Programs[signal_id] = program

    # Save the signal programs to a pickle file for later reference
    with open("signal_programs.pkl", "wb") as f:
        pickle.dump(Signal_Programs, f)
    return Signal_Programs

def change_signal_program(signal_id, offset, cycle_length,
                          minDur, maxDur, earliestEnd, lastestEnd):
    """Change the signal program for a specific traffic light.

    Args:
        signal_id (_type_): _description_
        offset (_type_): _description_
        cycle_length (_type_): _description_
        minDur (_type_): _description_
        maxDur (_type_): _description_
        earliestEnd (_type_): _description_
        lastestEnd (_type_): _description_
    """
    
    # Get the current signal program
    current_program = traci.trafficlight.getCompleteRedYellowGreenDefinition(signal_id)[0]
    
    # Modify the program as needed
    traci.trafficlight.setParameter(signal_id, "cycleTime", cycle_length)
    traci.trafficlight.setParameter(signal_id, "offset", offset)
    for phase_idx, phase in enumerate(current_program.phases):
        phase.minDur = minDur[phase_idx]
        phase.maxDur = maxDur[phase_idx]
        phase.earliestEnd = earliestEnd[phase_idx]
        phase.lastestEnd = lastestEnd[phase_idx]

    # Set the modified program back to the traffic light
    traci.trafficlight.setCompleteRedYellowGreenDefinition(signal_id, current_program)
    
def adjust_signal_programs():
    """_summary_
    """
    
    for signal_id in traci.trafficlight.getIDList():
        # Change the signal program for each traffic light as needed
        cycle_length = 60
        num_phases = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(signal_id)[0].phases)
        minDur = [cycle_length // num_phases] * num_phases
        maxDur = [cycle_length // num_phases] * num_phases
        earliestEnd = [cycle_length // num_phases] * num_phases
        lastestEnd = [cycle_length // num_phases] * num_phases
        offset = 10
        
        change_signal_program(signal_id, offset, cycle_length, minDur,
                              maxDur, earliestEnd, lastestEnd)

def run_simulation(
    sumocfg,
    additional_file,
    approach_folder,
    output_csv="detector_15min_snapshots_v5.csv",
    use_gui=False,
    end_time=None,
    return_results=False,
    trips_file=None,
    time_space_output_dir=None,
    mainline_road_name="University Avenue",
    time_space_sample_interval=1.0,
):
    """
    Start SUMO, run the simulation, and collect detector snapshots every 15 minutes.

    Parameters
    ----------
    sumocfg : str
        Path to .sumocfg file
    additional_file : str
        Path to detectors.add.xml containing inductionLoop definitions
    approach_folder : str
        Path to approach folder containing approach results.csv files
    output_csv : str
        CSV path for saved detector snapshots
    use_gui : bool
        If True, start sumo-gui instead of sumo
    end_time : float, optional
        Simulation end time in seconds
    return_results : bool
        If True, return results as a pandas DataFrame
    trips_file : str, optional
        Override the route file in the sumocfg (for calibration runs)
    time_space_output_dir : str, optional
        If provided, save mainline time-space plots and sampled raw data to this folder
    mainline_road_name : str
        Corridor road name used to identify the mainline in the SUMO network
    time_space_sample_interval : float
        Sampling interval in seconds for mainline trajectories and signal states
    """

    binary = checkBinary("sumo-gui" if use_gui else "sumo")

    cmd = [
        binary,
        "-c", sumocfg,
        "--additional-files", additional_file,
    ]

    if trips_file:
        cmd.extend(["--route-files", trips_file])

    # If using sumo-gui with TraCI, set --start so the simulation begins immediately.
    if use_gui:
        cmd.append("--start")

    # explicit TraCI port (set by parallel calibration workers) avoids the free-port
    # race between concurrently-launched SUMO instances; unset -> traci auto-picks.
    _port = os.environ.get("RL_TRACI_PORT")
    if _port:
        traci.start(cmd, port=int(_port))
    else:
        traci.start(cmd)

    # Pluggable probe collector: RL_COLLECTOR=probe subscribes the ~60-100 probe
    # VEHICLES instead of ~1490 induction loops (validated corr 0.999 / 0.2 kph MAE),
    # collapsing the per-step serialization -> ~4x faster sims. Same speed_buffer
    # output, so calibration speeds are unchanged. Default keeps the loop method.
    if os.environ.get("RL_COLLECTOR") == "probe":
        from rl.probe_collect import (setup_subscriptions as _setup_subs,
                                       collect_step_data as _collect)
    else:
        _setup_subs, _collect = setup_subscriptions, collect_step_data

    _setup_subs()
    segments = parse_segments_v8(approach_folder)
    segments = build_segment_detectors_v8(segments)  # needs traci connection
    veh_entry_times = [dict() for _ in segments]      # per-segment {veh: entry_t}
    speed_buffer = defaultdict(list)                  # {segment_index: [speeds]}
    time_space_recorder = None
    if time_space_output_dir:
        time_space_recorder = MainlineTimeSpaceRecorder(
            sumocfg_path=sumocfg,
            output_dir=time_space_output_dir,
            road_name=mainline_road_name,
            sample_interval_s=time_space_sample_interval,
        )

    all_rows = []

    # create output with header (ensure dir exists — a missing output dir makes
    # the open() below fail, which crashes TraCI with "peer shutdown")
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()

        interval_begin = 0
        interval_end   = INTERVAL  # 900

        sim_time = 0
        while traci.simulation.getMinExpectedNumber() > 0 and (end_time is None or sim_time < end_time):
            traci.simulationStep()
            sim_time = traci.simulation.getTime()

            _collect(segments, veh_entry_times, speed_buffer)
            if time_space_recorder is not None:
                time_space_recorder.record_step(sim_time)

            if sim_time >= interval_end:
                write_interval_results(writer, interval_begin, interval_end, speed_buffer, segments)
                csvfile.flush()

                interval_begin = interval_end + 1   # 901, 1801, ...
                interval_end   = interval_begin - 1 + INTERVAL  # 1800, 2700, ...

    traci.close()
    if time_space_recorder is not None:
        time_space_recorder.finalize()

    if return_results:
        return pd.read_csv(output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sumocfg", default="data/corridor-osm-network/osm.sumocfg", help="Path to SUMO config file (.sumocfg)")
    parser.add_argument("--additional", default="data/corridor-osm-network/detectors_v8_fixed.add.xml", help="Path to detectors additional-file (.add.xml)")
    parser.add_argument("--approach-folder", default="data/geojson-1-15-min-interval/processed", help="Path to approach folder containing approach results.csv files")
    parser.add_argument("--output", default="detector_15min_snapshots_v2.csv", help="CSV output file")
    parser.add_argument("--gui", action="store_true", default=False, help="Use sumo-gui instead of sumo")
    parser.add_argument("--end-time", type=float, default=90000, help="Optional simulation end time in seconds")
    parser.add_argument("--trips-file", default=None, help="Override route file in sumocfg")
    parser.add_argument("--time-space-output-dir", default=None, help="Output folder for mainline time-space plots")
    parser.add_argument("--mainline-road-name", default="University Avenue", help="Road name used as the mainline corridor")
    parser.add_argument("--time-space-sample-interval", type=float, default=1.0, help="Sampling interval in seconds for the time-space recorder")
    args = parser.parse_args()

    run_simulation(
        sumocfg=args.sumocfg,
        additional_file=args.additional,
        approach_folder=args.approach_folder,
        output_csv=args.output,
        use_gui=args.gui,
        end_time=args.end_time,
        trips_file=args.trips_file,
        time_space_output_dir=args.time_space_output_dir,
        mainline_road_name=args.mainline_road_name,
        time_space_sample_interval=args.time_space_sample_interval)

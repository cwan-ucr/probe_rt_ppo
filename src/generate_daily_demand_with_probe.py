"""
Generate full-day traffic demand for SUMO with morning/evening peak characteristics.
Calls randomTrips.py once per hour with volume adjusted to reflect daily traffic patterns.
Output: data/corridor-osm-network/trips_daily.xml

Supports:
- Parameterized hourly volumes (for calibration)
- Edge weight files for arterial OD preference (--weights-prefix)
- Callable API for automated calibration loops
"""

import os
import subprocess
import xml.etree.ElementTree as ET
import tempfile
import random

# ── Configuration ────────────────────────────────────────────────────────────
SUMO_HOME   = os.environ.get("SUMO_HOME", "/opt/homebrew/share/sumo")
RANDOM_TRIPS = os.path.join(SUMO_HOME, "tools", "randomTrips.py")
NET_FILE    = "data/corridor-osm-network/osm.net.xml.gz"
OUT_FILE    = "data/corridor-osm-network/trips_daily.xml"
OUT_FILE_PROBE = "data/corridor-osm-network/trips_daily_probe.xml"
PROBE_PROB  = 0.1   # probability of a vehicle being a probe vehicle
SEED        = 42
MIN_DIST    = 450   # metres – same as existing trips
FRINGE      = 5     # favour network-edge edges as O/D (good for corridor network)

# ── Default hourly demand (vehicles/hour) ────────────────────────────────────
# Typical urban arterial profile
DEFAULT_HOURLY_DEMAND = {
    0:  80,
    1:  60,
    2:  40,
    3:  40,
    4:  80,
    5:  300,
    6:  700,
    7:  1200,
    8:  1600,   # morning peak
    9:  1300,
    10: 1000,
    11: 800,
    12: 750,
    13: 780,
    14: 900,
    15: 1100,
    16: 1300,
    17: 1600,   # evening peak
    18: 1500,
    19: 1100,
    20: 700,
    21: 450,
    22: 250,
    23: 120,
}

# Keep backward compatibility
HOURLY_DEMAND = DEFAULT_HOURLY_DEMAND

# ── Script root (so relative paths resolve correctly) ─────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)

def abs_path(rel):
    return os.path.join(ROOT_DIR, rel)


def generate_hour(hour: int, volume: int, tmp_dir: str,
                   net_file=None, weights_prefix=None, seed=None,
                   min_dist=None, fringe=None, verbose=True,
                   time_offset=0) -> str | None:
    """Run randomTrips.py for one hour; return path to the generated trips file.

    Parameters
    ----------
    time_offset : int
        Seconds to subtract from absolute depart times.
        E.g., for hour=7 with time_offset=25200, depart starts at 0.
    """
    if volume == 0:
        return None

    net_file = net_file or abs_path(NET_FILE)
    seed = seed if seed is not None else (SEED + hour)
    min_dist = min_dist or MIN_DIST
    fringe = fringe or FRINGE

    begin  = hour * 3600 - time_offset
    end    = begin + 3600
    period = round(3600 / volume, 4)   # seconds between vehicle insertions
    out    = os.path.join(tmp_dir, f"trips_h{hour:02d}.xml")

    cmd = [
        "python", RANDOM_TRIPS,
        "-n", net_file,
        "-o", out,
        "-b", str(begin),
        "-e", str(end),
        "--period", str(period),
        "--min-distance", str(min_dist),
        "--fringe-factor", str(fringe),
        "--seed", str(seed),
        "--validate",
    ]

    # Add edge weight file or speed exponent for arterial OD preference
    if weights_prefix:
        if isinstance(weights_prefix, (int, float)):
            # Use speed exponent: higher values favor faster (arterial) edges
            cmd.extend(["--speed-exponent", str(weights_prefix)])
        else:
            cmd.extend(["--weights-prefix", weights_prefix])

    if verbose:
        print(f"  Hour {hour:02d}:00 — {volume:4d} veh/hr  (period={period}s)")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        if verbose:
            print(f"    WARNING: randomTrips failed for hour {hour}:")
            print(result.stderr[-400:])
        return None

    return out


def merge_trips(trip_files: list[str], output_path: str):
    """Merge multiple trips XML files into one, re-sequencing trip IDs."""
    root = ET.Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation",
             "http://sumo.dlr.de/xsd/routes_file.xsd")

    trip_id = 0
    for fpath in trip_files:
        if fpath is None or not os.path.exists(fpath):
            continue
        tree = ET.parse(fpath)
        for trip in tree.getroot().iter("trip"):
            trip.set("id", str(trip_id))
            root.append(trip)
            trip_id += 1

    ET.indent(root, space="    ")
    tree_out = ET.ElementTree(root)
    ET.ElementTree.write(tree_out, output_path,
                         encoding="unicode", xml_declaration=True)
    return trip_id

def assign_probe_vehicles(probe_prob, file_path, output_path, seed=0):
    random.seed(seed)
    
    # read the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # vType elements to be added at the beginning of the XML file
    vtype1 = ET.Element("vType", {"id": "probe_vehicle", "vClass": "passenger"})
    vtype2 = ET.Element("vType", {"id": "regular_vehicle", "vClass": "passenger"})
    root.insert(0, vtype2)
    root.insert(0, vtype1)

    # iterate through all trip elements
    for trip in root.findall("trip"):
        # add vtype attribute to each trip element
        if random.random() < probe_prob:
            trip.set("type", "probe_vehicle")
        else:
            trip.set("type", "regular_vehicle")

    # write the modified XML to a new file
    ET.indent(tree, space="    ", level=0)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

def generate_arterial_weights(net_file, output_prefix, arterial_factor=2.0):
    """Generate SUMO edge weight files that give arterial (highway.primary) edges
    higher OD selection probability.

    Creates {output_prefix}.src.xml and {output_prefix}.dst.xml for use with
    randomTrips.py --weights-prefix.

    Parameters
    ----------
    net_file : str
        Path to the SUMO network file
    output_prefix : str
        Prefix for the output weight files
    arterial_factor : float
        Weight multiplier for highway.primary edges (default 2.0)
    """
    import sys
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", ""), "tools"))
    import sumolib

    net = sumolib.net.readNet(net_file)

    for suffix in ["src", "dst"]:
        root = ET.Element("meandata")
        interval = ET.SubElement(root, "interval", begin="0", end="86400")

        for edge in net.getEdges():
            edge_id = edge.getID()
            edge_type = edge.getType()
            # Assign higher weight to primary roads (arterials)
            if "primary" in edge_type:
                weight = arterial_factor
            elif "secondary" in edge_type:
                weight = max(1.0, arterial_factor * 0.7)
            elif "motorway" in edge_type:
                weight = 0.5  # discourage motorway as local OD
            else:
                weight = 1.0

            ET.SubElement(interval, "edge", id=edge_id, value=str(round(weight, 2)))

        tree = ET.ElementTree(root)
        out_path = f"{output_prefix}.{suffix}.xml"
        tree.write(out_path, encoding="unicode", xml_declaration=True)

    return output_prefix


def generate_demand(hourly_volumes=None, net_file=None, out_file=None,
                    out_file_probe=None, probe_prob=0.1, seed=42,
                    weights_prefix=None, arterial_factor=None,
                    hours_range=None, verbose=True):
    """Generate traffic demand. Callable API for calibration.

    Parameters
    ----------
    hourly_volumes : dict
        Hour → vehicles/hr. Defaults to DEFAULT_HOURLY_DEMAND.
    net_file : str
        Network file path. Defaults to NET_FILE.
    out_file : str
        Output trips XML. Defaults to OUT_FILE.
    out_file_probe : str
        Output probe trips XML. Defaults to OUT_FILE_PROBE.
    probe_prob : float
        Probe vehicle probability (default 0.1).
    seed : int
        Base random seed.
    weights_prefix : str
        Edge weight file prefix for randomTrips. If None and arterial_factor
        is set, auto-generates weight files.
    arterial_factor : float
        If set, generates arterial weight files before demand generation.
    hours_range : tuple (start_hour, end_hour)
        Only generate for these hours. Default: all 24 hours.

    Returns
    -------
    int : number of trips generated
    """
    hourly_volumes = hourly_volumes or DEFAULT_HOURLY_DEMAND
    net_file = abs_path(net_file or NET_FILE)
    out_file = abs_path(out_file or OUT_FILE)
    out_file_probe = abs_path(out_file_probe or OUT_FILE_PROBE)

    if not os.path.exists(RANDOM_TRIPS):
        raise FileNotFoundError(
            f"randomTrips.py not found at {RANDOM_TRIPS}\n"
            "Set the SUMO_HOME environment variable to your SUMO installation."
        )

    # Use speed exponent to favor arterial edges in OD selection.
    # Higher speed_exponent → more trips on higher-speed (arterial) roads.
    # arterial_factor=2.0 → speed_exponent=2.0 → P(edge) ∝ speed^2
    # Primary (27.8 m/s) vs unclassified (13.9 m/s): 4x probability
    if arterial_factor and not weights_prefix:
        weights_prefix = float(arterial_factor)  # pass as speed exponent
        if verbose:
            print(f"Using speed exponent={arterial_factor} "
                  f"for arterial OD preference")

    start_hour = hours_range[0] if hours_range else 0
    end_hour = hours_range[1] if hours_range else 24

    if verbose:
        total = sum(hourly_volumes.get(h, 0) for h in range(start_hour, end_hour))
        print(f"Generating demand: hours {start_hour}-{end_hour}, "
              f"total={total:,} vehicles")

    # When generating a partial day, shift depart times so sim starts at t=0
    time_offset = start_hour * 3600 if hours_range else 0

    trip_files = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for hour in range(start_hour, end_hour):
            volume = hourly_volumes.get(hour, 0)
            f = generate_hour(hour, volume, tmp_dir,
                              net_file=net_file,
                              weights_prefix=weights_prefix,
                              seed=seed + hour,
                              verbose=verbose,
                              time_offset=time_offset)
            trip_files.append(f)

        n = merge_trips(trip_files, out_file)

    if verbose:
        print(f"Done — {n:,} trips written to {out_file}")
        if time_offset:
            print(f"  (depart times shifted by -{time_offset}s so sim starts at t=0)")

    assign_probe_vehicles(probe_prob, out_file, out_file_probe, seed=seed)
    return n


def main(probe_prob=0.1):
    if not os.path.exists(RANDOM_TRIPS):
        raise FileNotFoundError(
            f"randomTrips.py not found at {RANDOM_TRIPS}\n"
            "Set the SUMO_HOME environment variable to your SUMO installation."
        )

    print("=" * 55)
    print("  Generating full-day traffic demand")
    print("=" * 55)
    print(f"Network : {abs_path(NET_FILE)}")
    print(f"Output  : {abs_path(OUT_FILE)}")
    print()

    total_vehicles = sum(HOURLY_DEMAND.values())
    print(f"Total daily vehicles (target): {total_vehicles:,}")
    print()

    trip_files = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for hour in range(24):
            volume = HOURLY_DEMAND[hour]
            f = generate_hour(hour, volume, tmp_dir)
            trip_files.append(f)

        print()
        print("Merging all hourly trips …")
        n = merge_trips(trip_files, abs_path(OUT_FILE))

    print(f"Done — {n:,} trips written to {abs_path(OUT_FILE)}")
    print()

    # Assign probe vehicles to the merged trips
    assign_probe_vehicles(probe_prob, abs_path(OUT_FILE), abs_path(OUT_FILE_PROBE), seed=SEED)


if __name__ == "__main__":
    main(probe_prob=PROBE_PROB)

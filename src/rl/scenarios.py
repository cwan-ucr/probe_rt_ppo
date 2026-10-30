"""Versioned Latin-hypercube demand designs shared by every RL experiment.

The files contain demand *parameters only*.  Signal-backbone- and penetration-
specific incumbent trajectories live in separate baseline-cache files keyed by the
scenario-file hash.  This separation guarantees that architecture, reward,
penetration, and backbone comparisons use the same demand points.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import config as C
from . import demand


SCENARIO_DIR = Path(C.ROOT) / "data" / "rl_scenarios"
DEFAULT_TRAIN_FILE = SCENARIO_DIR / "train_lhs_n2000_seed42.json"
DEFAULT_EVAL_FILE = SCENARIO_DIR / "eval_lhs_n72_seed54321.json"
SCHEMA_VERSION = 2
DIMENSIONS = ("global", "main", "side", "direction")


def _canonical_ranges(ranges=None):
    ranges = ranges or demand.DEFAULT_RANGES
    return {key: [float(ranges[key][0]), float(ranges[key][1])]
            for key in DIMENSIONS}


def latin_hypercube(n: int, seed: int, ranges=None) -> list[tuple[float, ...]]:
    """Randomized LHS with one sample in every marginal stratum.

    This NumPy-only implementation is stable on both the macOS workstation and the
    HPCC environment and avoids a SciPy-version-dependent optimization step.
    """
    if n < 1:
        raise ValueError("Latin-hypercube size must be positive")
    canonical = _canonical_ranges(ranges)
    rng = np.random.default_rng(int(seed))
    unit = np.empty((n, len(DIMENSIONS)), dtype=float)
    for column in range(len(DIMENSIONS)):
        unit[:, column] = (rng.permutation(n) + rng.random(n)) / n
    out = np.empty_like(unit)
    for column, name in enumerate(DIMENSIONS):
        low, high = canonical[name]
        out[:, column] = low + unit[:, column] * (high - low)
    # Six decimals preserve the LHS strata at the current bank sizes while
    # keeping the JSON readable and reproducible across platforms.
    return [tuple(round(float(value), 6) for value in row) for row in out]


def write_scenario_file(path, n: int, seed: int, ranges=None):
    path = Path(path)
    canonical = _canonical_ranges(ranges)
    points = latin_hypercube(n, seed, canonical)
    payload = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "sampler": "latin_hypercube_numpy_v1",
            "scrambled": True,
            "seed": int(seed),
            "n_scenarios": int(n),
            "dimensions": list(DIMENSIONS),
            "ranges": canonical,
        },
        "scenarios": [
            {"index": index, "params": list(point)}
            for index, point in enumerate(points)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)
    return path


def load_scenario_file(path, expected_n=None):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    metadata = payload.get("metadata", {})
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported scenario schema in {path}: expected "
            f"version {SCHEMA_VERSION} with dimensions {DIMENSIONS}; "
            "regenerate the common LHS files")
    if tuple(metadata.get("dimensions", ())) != DIMENSIONS:
        raise ValueError(f"unexpected demand dimensions in {path}")
    expected_ranges = _canonical_ranges()
    if metadata.get("ranges") != expected_ranges:
        raise ValueError(
            f"scenario ranges in {path} are {metadata.get('ranges')}, expected "
            f"{expected_ranges}; regenerate the common LHS files")
    rows = payload.get("scenarios", [])
    points = [tuple(float(value) for value in row["params"]) for row in rows]
    if len(points) != int(metadata.get("n_scenarios", -1)):
        raise ValueError(f"scenario count does not match metadata in {path}")
    if expected_n is not None and len(points) != int(expected_n):
        raise ValueError(
            f"{path} contains {len(points)} scenarios; expected {expected_n}")
    if len(set(points)) != len(points):
        raise ValueError(f"duplicate demand points in {path}")
    return points, metadata


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--eval-size", type=int, default=72)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=54321)
    parser.add_argument("--check", action="store_true",
                        help="validate existing files rather than rewriting them")
    args = parser.parse_args()
    if args.check:
        for path, n in ((args.train_file, args.train_size),
                        (args.eval_file, args.eval_size)):
            points, metadata = load_scenario_file(path, n)
            print(f"valid {path}: n={len(points)} seed={metadata['seed']} "
                  f"sha256={file_sha256(path)[:12]}")
        return
    for path, n, seed in (
            (args.train_file, args.train_size, args.train_seed),
            (args.eval_file, args.eval_size, args.eval_seed)):
        write_scenario_file(path, n, seed)
        print(f"wrote {path}: n={n} seed={seed} "
              f"sha256={file_sha256(path)[:12]}")


if __name__ == "__main__":
    main()

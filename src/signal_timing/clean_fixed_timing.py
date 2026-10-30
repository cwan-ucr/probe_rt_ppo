"""
Phase 1 baseline cleanup for fixed-timing signal plans.

For each <tlLogic> block in fixed_timing.add.xml:
  - Round phase durations to integer seconds
  - Yellow phases (state contains 'y') stay at their integer-rounded value
  - Non-yellow durations are re-scaled so the cycle sums to the target
  - Rounding remainder is dumped into the longest non-yellow phase
  - If any non-yellow phase ends up below MIN_GREEN_S, retry with the
    fallback cycle (70s instead of 60s)
  - Offset reset to 0 (optimization happens in Phase 2)
  - WAUT blocks preserved unchanged

Usage:
  python src/signal_timing/clean_fixed_timing.py            # writes _60s file
  python src/signal_timing/clean_fixed_timing.py --in-place # overwrite source
"""

import argparse
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_INPUT  = os.path.join(ROOT_DIR, "data", "corridor-osm-network", "fixed_timing.add.xml")
DEFAULT_OUTPUT = os.path.join(ROOT_DIR, "data", "corridor-osm-network", "fixed_timing_60s.add.xml")

PRIMARY_CYCLE  = 60
FALLBACK_CYCLE = 70
MIN_GREEN_S    = 7


def is_yellow(state: str) -> bool:
    return "y" in state


def try_clean(tl: ET.Element, target_cycle: int) -> Tuple[bool, Optional[str], list]:
    """Return (ok, error_msg, new_durations) without mutating tl yet."""
    phases = tl.findall("phase")
    if not phases:
        return False, "no phases", []

    durations = [float(p.get("duration")) for p in phases]
    yellows = [is_yellow(p.get("state")) for p in phases]

    sum_yellow = sum(round(d) for d, y in zip(durations, yellows) if y)
    target_non_yellow = target_cycle - sum_yellow
    sum_non_yellow = sum(d for d, y in zip(durations, yellows) if not y)

    if sum_non_yellow <= 0:
        return False, "no non-yellow phases", []
    if target_non_yellow <= 0:
        return False, f"yellow total ({sum_yellow}s) >= target cycle ({target_cycle}s)", []

    scale = target_non_yellow / sum_non_yellow

    new_durations = []
    for d, y in zip(durations, yellows):
        new_durations.append(int(round(d)) if y else int(round(d * scale)))

    # Adjust rounding remainder on the longest non-yellow phase
    delta = target_cycle - sum(new_durations)
    if delta != 0:
        non_yellow_idx = [i for i, y in enumerate(yellows) if not y]
        longest = max(non_yellow_idx, key=lambda i: new_durations[i])
        new_durations[longest] += delta

    # Validate min-green
    for i, (d, y) in enumerate(zip(new_durations, yellows)):
        if not y and d < MIN_GREEN_S:
            return False, f"phase[{i}] = {d}s < min {MIN_GREEN_S}s", new_durations

    return True, None, new_durations


def apply(tl: ET.Element, new_durations: list) -> None:
    for ph, d in zip(tl.findall("phase"), new_durations):
        ph.set("duration", str(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--in-place", action="store_true",
                    help="Overwrite the input file instead of writing --output")
    ap.add_argument("--reset-offset", action="store_true", default=True,
                    help="Set offset='0' on every cleaned tlLogic (default: yes)")
    args = ap.parse_args()

    out_path = args.input if args.in_place else args.output

    tree = ET.parse(args.input)
    root = tree.getroot()

    tls = root.findall(".//tlLogic")
    stats = {"60s": 0, "70s": 0, "failed": 0}
    fallback_by_tl = defaultdict(int)
    failures = []

    for tl in tls:
        tl_id = tl.get("id")
        prog_id = tl.get("programID")

        ok, _, new_durs = try_clean(tl, PRIMARY_CYCLE)
        if ok:
            apply(tl, new_durs)
            if args.reset_offset:
                tl.set("offset", "0")
            stats["60s"] += 1
            continue

        ok, _, new_durs = try_clean(tl, FALLBACK_CYCLE)
        if ok:
            apply(tl, new_durs)
            if args.reset_offset:
                tl.set("offset", "0")
            stats["70s"] += 1
            fallback_by_tl[tl_id] += 1
            continue

        # Could not satisfy at either cycle — try 70s anyway with the
        # min-green floor (we accept the violation rather than leaving stale
        # floats). Re-run without the validation step.
        ok2, err, new_durs = try_clean(tl, FALLBACK_CYCLE)
        # Even if min-green failed, we still got durations; force-apply.
        if new_durs:
            apply(tl, new_durs)
            if args.reset_offset:
                tl.set("offset", "0")
        stats["failed"] += 1
        failures.append((tl_id, prog_id, err))

    tree.write(out_path, encoding="utf-8", xml_declaration=True)

    print(f"Processed {len(tls)} <tlLogic> blocks")
    print(f"  cleaned to 60s : {stats['60s']}")
    print(f"  fell back to 70s: {stats['70s']}")
    print(f"  could not satisfy min-green at 70s: {stats['failed']}")
    if fallback_by_tl:
        print(f"\nTLs that needed 70s fallback (program count):")
        for tl_id, n in sorted(fallback_by_tl.items()):
            print(f"  {tl_id}: {n} programs")
    if failures:
        print(f"\nFailures (first 5):")
        for tl_id, prog_id, err in failures[:5]:
            print(f"  {tl_id} prog '{prog_id}': {err}")
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()

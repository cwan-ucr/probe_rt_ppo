"""
Plot GA iteration progress: best-so-far + per-generation best/mean ± std.

Usage:
    python src/demand_calibration/plot_ga_progress.py
    python src/demand_calibration/plot_ga_progress.py --log <path/to/ga_log_*.json>
"""

import os
import sys
import glob
import json
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.dirname(SCRIPT_DIR)
ROOT_DIR   = os.path.dirname(SRC_DIR)

DEFAULT_LOG_GLOB = os.path.join(ROOT_DIR, "outputs", "calibration",
                                "ga_log_*_*.json")
DEFAULT_OUT      = os.path.join(ROOT_DIR, "outputs", "calibration",
                                "ga_progress.png")


def find_latest_log():
    matches = sorted(glob.glob(DEFAULT_LOG_GLOB))
    return matches[-1] if matches else None


def plot_progress(log_file, out_file):
    with open(log_file) as f:
        log = json.load(f)
    if not log:
        print(f"  Empty log: {log_file}")
        return

    gens     = [e["generation"]            for e in log]
    best     = [e["best_z"]                for e in log]
    gen_best = [e.get("gen_best_z", e["best_z"]) for e in log]
    mean_z   = [e["mean_z"]                for e in log]
    std_z    = [e["std_z"]                 for e in log]

    mean_arr = np.array(mean_z)
    std_arr  = np.array(std_z)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.fill_between(gens, mean_arr - std_arr, mean_arr + std_arr,
                    alpha=0.2, color="#1f77b4", label="Mean ± Std (population)")
    ax.plot(gens, mean_z,   "-o", color="#1f77b4", lw=1.2, ms=4,
            label="Generation mean")
    ax.plot(gens, gen_best, "--s", color="#ff7f0e", lw=1.2, ms=4,
            label="Generation best")
    ax.plot(gens, best,     "-",  color="#2ca02c", lw=2.4,
            label="Best-so-far")

    # Annotate final best
    ax.annotate(f"best = {best[-1]:.2f} kph",
                xy=(gens[-1], best[-1]),
                xytext=(8, 8), textcoords="offset points",
                fontsize=10, color="#2ca02c", fontweight="bold")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Weighted RMSE (kph)")
    ax.set_title(f"GA Calibration Progress\n{os.path.basename(log_file)}",
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_file}")
    print(f"  Generations: {len(gens)}, "
          f"initial best: {best[0]:.3f}, final best: {best[-1]:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=None,
                        help="Path to ga_log_*.json (default: latest)")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"Output PNG path (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    log_file = args.log or find_latest_log()
    if not log_file or not os.path.exists(log_file):
        print(f"No log file found. Pattern: {DEFAULT_LOG_GLOB}")
        sys.exit(1)

    plot_progress(log_file, args.out)


if __name__ == "__main__":
    main()

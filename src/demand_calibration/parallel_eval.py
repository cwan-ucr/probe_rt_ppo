"""
Parallel population evaluation for the demand-calibration GA.

Each GA individual = one full SUMO calibration sim (WRMSE of simulated vs TomTom
segment speeds). Individuals are independent, so a generation's population is
evaluated on a persistent ProcessPool. Each worker:
  - forces RL_COLLECTOR=probe -> the probe-vehicle collector (~4x faster sims,
    validated corr 0.999 / 0.2 kph MAE vs the loop method);
  - isolates its trips/results files in a private temp dir (parallel-safe);
  - builds runner + targets ONCE (reused across the worker's evals).

Used by run_calibration when --n-workers > 1; the serial path is unchanged.
"""
import os
import sys
import shutil
import atexit
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# per-worker state (built once in the initializer)
_RUNNER = None
_TARGETS = None
_TEMPLATE = None
_WORK = None


def _init_worker(cfg):
    global _RUNNER, _TARGETS, _TEMPLATE, _WORK
    os.environ["RL_COLLECTOR"] = "probe"          # B: probe collector in every sim
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    # distinct TraCI port per worker -> no free-port race between parallel SUMOs
    # (worker reuses its port across its serial evals; previous SUMO has closed).
    os.environ["RL_TRACI_PORT"] = str(8900 + os.getpid() % 1000)
    from demand_calibration.simulation_runner import SimulationRunner
    from demand_calibration.calibration_targets import CalibrationTargets
    _WORK = tempfile.mkdtemp(prefix="cal_w_")
    atexit.register(lambda: shutil.rmtree(_WORK, ignore_errors=True))
    _RUNNER = SimulationRunner(
        sumocfg=cfg["sumocfg"], additional_file=cfg["additional"],
        approach_folder=cfg["approach_folder"], net_file=cfg["net_file"],
        sim_start_hour=cfg["start_hour"], sim_duration_hours=cfg["hours"],
        probe_prob=cfg["probe_prob"], probe_prob_cross=cfg["probe_prob_cross"],
        boundaries_json=cfg["boundaries_json"], stop_bar_only=cfg["stop_bar_only"],
        output_dir=_WORK)
    _TARGETS = CalibrationTargets(cfg["approach_folder"])
    _TEMPLATE = cfg["template"]


def _eval_one(vec):
    from demand_calibration.objective import ObjectiveFunction
    from demand_calibration.turning_parameters import TurningParameters
    try:
        params = TurningParameters.from_vector(vec, _TEMPLATE)
        params.project_bounds()
        return float(ObjectiveFunction(_TARGETS, _RUNNER).evaluate(params))
    except Exception:
        import traceback
        traceback.print_exc()
        return 999.0


class ParallelObjective:
    """Persistent process pool. evaluate_batch(vectors) -> list of WRMSE."""

    def __init__(self, cfg, n_workers):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        self.n_workers = n_workers
        self.pool = ProcessPoolExecutor(
            max_workers=n_workers, mp_context=mp.get_context("spawn"),
            initializer=_init_worker, initargs=(cfg,))

    def evaluate_batch(self, vecs):
        return list(self.pool.map(_eval_one, list(vecs)))

    def close(self):
        self.pool.shutdown(wait=True)

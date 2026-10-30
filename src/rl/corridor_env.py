"""
Centralized SUMO/TraCI environment for probe-driven 15-min offset control.

MDP:
  state  : probe-only corridor state (segment_state.build_state)           [STATE_DIM]
  action : 8 offsets, one per controlled intersection (continuous [-1,1]
           mapped to [0, CYCLE-1]; rounded to int when applied)
  reward : -corridor segment delay (reward.corridor_reward) over the interval
  step   : 15 min of simulation under the chosen offsets
  backbone: coordinated-fixed, coordinated-actuated, or coordinated-actuated
            with detector-call phase skipping

Segment measurement reuses the validated v8 entry/exit logic from tsc_sumo_sim.
Offset is applied at each decision boundary by re-aligning the program to the
phase position implied by the new offset (works for the static fixed base); for
the actuated base the coordination `offset` parameter is also set.
"""
import os
import sys
import math
import socket
import tempfile
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from functools import lru_cache
import numpy as np

from . import config as C
from .corridor_geometry import load_geometry
from .segment_state import build_state
from .reward import corridor_reward
from .signal_phases import sum_preserving_split

if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = ("/Library/Frameworks/EclipseSUMO.framework/"
                               "Versions/Current/EclipseSUMO/share/sumo")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
import traci
from sumolib import checkBinary
from sumolib.miscutils import getFreeSocketPort
# segment parsing/detector building reuse the originals; the hot-loop collection
# uses the optimized drop-in (identical output, much less per-step work). To fall
# back, change this import to pull setup_subscriptions/collect_step_data from
# tsc_sumo_sim instead.
from tsc_sumo_sim import parse_segments_v8, build_segment_detectors_v8
from .detector_geometry import load_probe_segments
# Step-data collector is swappable (identical output contract):
#   RL_COLLECTOR=fast  (default) -> fast_collect: subscribe 1490 induction loops;
#   RL_COLLECTOR=probe          -> probe_collect: subscribe ~60-100 probe vehicles
#       instead (object count drops ~15-25x -> the per-step subscription
#       serialization that dominates the SUMO step collapses; ~speeds validated to
#       corr 0.999 / 0.2 kph MAE vs the loop method by validate_probe_collect).
COLLECTOR_NAME = os.environ.get("RL_COLLECTOR", "fast")
USE_PROBE_COLLECTOR = COLLECTOR_NAME == "probe"
if USE_PROBE_COLLECTOR:
    from .probe_collect import setup_subscriptions, collect_step_data
else:
    from .fast_collect import setup_subscriptions, collect_step_data


@lru_cache(maxsize=None)
def _route_demand_end(route_file):
    """Latest scheduled departure in a route file; 0 keeps legacy polling."""
    try:
        root = ET.parse(route_file).getroot()
    except (ET.ParseError, OSError):
        return 0.0
    ends = []
    for elem in root.iter():
        value = elem.get("end") if elem.tag == "flow" else elem.get("depart")
        if value is None or value in ("triggered", "containerTriggered"):
            continue
        try:
            ends.append(float(value))
        except ValueError:
            pass
    return max(ends, default=0.0)


def _runtime_dir():
    """Node-local scratch for per-worker route and metrics files."""
    root = (os.environ.get("RL_RUNTIME_DIR")
            or os.environ.get("SLURM_TMPDIR")
            or os.environ.get("TMPDIR")
            or tempfile.gettempdir())
    path = os.path.join(root, "probe_signal_rl")
    os.makedirs(path, exist_ok=True)
    return path


@contextmanager
def _traci_launch_lock():
    """Serialize fallback port selection and SUMO bind across this user's jobs.

    getFreeSocketPort() releases its probe socket before SUMO binds. Without this
    lock, workers launched together can all observe the same port as available.
    """
    # Use node-global /tmp rather than TMPDIR/SLURM_TMPDIR, which may be isolated
    # per job and would not coordinate simultaneous jobs on the same compute node.
    lock_root = os.environ.get("RL_TRACI_LOCK_DIR", "/tmp")
    lock_path = os.path.join(
        lock_root, f"probe_signal_traci_{os.getuid()}.lock")
    with open(lock_path, "a+") as lock_file:
        try:
            import fcntl
        except ImportError:
            yield
            return
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _port_is_available(port):
    """Check a preferred port before TraCI can connect to an unrelated SUMO."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("", int(port)))
        return True
    except OSError:
        return False


def _harmonic(speeds):
    s = [x for x in speeds if x > 0]
    if not s:
        return -1.0, 0
    return len(s) / sum(1.0 / x for x in s), len(s)


class CorridorEnv:
    # The legacy environment keeps its historical action behavior.  The all10
    # subclass enables a policy-level split mask for fixed-duration nodes.
    USE_SPLIT_ACTION_MASK = False

    def __init__(self, base=None, gui=False, seed=42,
                 episode_end=C.EPISODE_END_S, port=None, backbone=None):
        if base is not None and backbone is not None:
            if C.canonical_backbone(base) != C.canonical_backbone(backbone):
                raise ValueError(
                    f"conflicting base={base!r} and backbone={backbone!r}")
        # Preserve the historical direct-environment default; production PPO
        # scripts pass their canonical backbone explicitly.
        backbone = C.canonical_backbone(backbone or base or "coord_fixed")
        if (os.environ.get("RL_EXPLICIT_DEMAND", "0") == "1"
                and not USE_PROBE_COLLECTOR):
            raise ValueError(
                "RL_EXPLICIT_DEMAND requires RL_COLLECTOR=probe because explicit "
                "EXP5 vehicles keep penetration-invariant ids and are identified "
                "by their vType")
        if episode_end <= C.WARMUP_S:
            raise ValueError("episode_end must exceed the warmup interval")
        if ((episode_end - C.WARMUP_S) % C.DECISION_INTERVAL_S) != 0:
            raise ValueError(
                "episode_end - warmup must be an integer number of "
                "decision intervals")
        # fixed TraCI port avoids the free-port race when many envs start SUMO
        # concurrently (parallel workers / baseline pool); None -> traci auto-picks.
        self._port = port
        # ``base`` remains as a compatibility attribute for older diagnostics.
        self.backbone = backbone
        self.base = backbone
        self.gui = gui
        self.seed = seed
        self.episode_end = episode_end
        self.sig = C.signal_path(backbone)
        self.geom = load_geometry()
        self.state_dim = C.STATE_DIM
        self.action_dim = C.ACTION_DIM
        self._started = False
        self._runtime_dir = _runtime_dir()
        self._static_segments = (
            load_probe_segments(C.APPROACH_DIR, C.DETECTORS)
            if USE_PROBE_COLLECTOR else None
        )
        self._demand_end = _route_demand_end(C.TRIPS)
        # per-env tripinfo file (true VMT/VHT, same definition as the 4-scenario
        # compare_signal_metrics): VMT=Sum routeLength, VHT=Sum duration, over all
        # vehicles incl. those still en route at episode end (write-unfinished).
        # keyed by the (unique) TraCI port too, so PARALLEL workers that share a seed
        # (e.g. eval on many scenarios, or same-seed training workers) never write the
        # same tripinfo file -> no XML corruption from concurrent writers.
        _uid = f"pid{os.getpid()}_p{port if port is not None else 'auto'}"
        self._tripinfo = os.path.join(
            self._runtime_dir, f"tripinfo_{backbone}_seed{seed}_{_uid}.xml")
        self._statistics = os.path.join(
            self._runtime_dir, f"statistics_{backbone}_seed{seed}_{_uid}.xml")
        self.metrics_mode = os.environ.get("RL_METRICS_MODE", "tripinfo")
        if self.metrics_mode not in ("tripinfo", "statistics"):
            raise ValueError(
                f"RL_METRICS_MODE={self.metrics_mode!r}; expected tripinfo or statistics")
        self.keep_metrics_files = os.environ.get("RL_KEEP_METRICS", "0") == "1"
        # Optional SUMO-native outputs used by offline diagnostics.  Empty by
        # default, so training/evaluation commands are unchanged.
        self.extra_additional_files = []
        self.extra_sumo_args = []
        self.time_to_teleport = 300
        # demand route-file: defaults to the calibrated base trips; set_demand()
        # swaps in a (global, main, side, direction)-scaled copy for training.
        # per-process file name avoids clashes between parallel workers.
        self._trips = C.TRIPS
        self._trips_path = os.path.join(
            self._runtime_dir, f"trips_pid{os.getpid()}.xml")
        self.demand_params = None
        # base GA/coordinated offsets that the delta action adjusts around
        self._ga_offset_by_program = C.read_program_offsets(self.sig)
        self._ga_offset = C.read_base_offsets(
            self.sig, allow_program_variation=True)
        self._coordinated_tlids = self._read_coordinated_tlids(self.sig)
        # per-phase split (maxDur) control: layout of the controllable non-coordinated
        # green phases per intersection (padded to K_MAX_SPLIT + mask). Only on the
        # actuated base (fixed base has no min/maxDur to actuate).
        self._split_layout = None
        self._split_applied = {}      # {name: {phase_idx: maxDur}} last applied (change-guard)
        self.action_mask = np.ones(C.ACTION_DIM, dtype=np.float32)
        if C.SPLIT_CONTROL and backbone == "coord_actuated":
            from .signal_phases import (
                build_policy_action_mask,
                build_split_layout,
                load_signal_phases,
            )
            _phases = load_signal_phases(self.sig)
            _kmax, self._split_layout = build_split_layout(_phases, k_max=C.K_MAX_SPLIT)
            if self.USE_SPLIT_ACTION_MASK:
                self.action_mask = build_policy_action_mask(self._split_layout)
        # per-decision-step baseline delay (set by compute_baseline) -> difference
        # reward = baseline_delay[k] - current_delay[k]; None -> scaled plain-delay
        self.baseline = None
        # Base delay coefficient used by the plain-delay ablation. The global
        # numerical conditioning factor C.REWARD_SCALE is applied separately.
        self.plain_reward_scale = 1.0
        # Optional read-only callback invoked whenever one probe-observation
        # interval is harvested.  Offline diagnostics can reuse the exact
        # 15-minute observations seen by the policy without extra TraCI calls.
        self.probe_observer = None

    # ---------- offset application ----------
    @staticmethod
    def _read_coordinated_tlids(sig_path):
        """Traffic-light ids whose loaded add.xml program uses SUMO coordination.

        These support TraCI's offset parameter directly. Any static or
        non-coordinated program needs explicit phase realignment instead.
        """
        out = set()
        try:
            root = ET.parse(sig_path).getroot()
        except (ET.ParseError, OSError):
            return out
        for logic in root.findall(".//tlLogic"):
            for param in logic.findall("param"):
                if (param.get("key") == "coordinated"
                        and param.get("value", "").lower() == "true"):
                    out.add(logic.get("id"))
                    break
        return out

    def _active_phase_durs(self, tlid):
        """Phase durations of the CURRENTLY active program (handles WAUT / the
        net-default program coexisting with the additional-file program)."""
        prog = traci.trafficlight.getProgram(tlid)
        logics = traci.trafficlight.getAllProgramLogics(tlid)
        active = next((l for l in logics if getattr(l, "programID", None) == prog),
                      logics[-1])
        return [float(p.duration) for p in active.phases]

    def _realign_phase_to_offset(self, tlid, offset):
        """Set the active phase/remaining duration so local cycle time matches offset."""
        durs = self._active_phase_durs(tlid)
        cyc = sum(durs) or C.CYCLE_S
        t = traci.simulation.getTime()
        pos = (t - offset) % cyc
        acc = 0.0
        for i, d in enumerate(durs):
            if pos < acc + d:
                traci.trafficlight.setPhase(tlid, i)
                traci.trafficlight.setPhaseDuration(tlid, (acc + d) - pos)
                break
            acc += d

    def _apply_offset(self, name, offset):
        for tlid in C.AGENT_TLIDS[name]:
            if tlid in self._coordinated_tlids:
                # coordinated-actuated: offset drives timeInCycle=(t-offset)%cycleTime
                traci.trafficlight.setParameter(tlid, "offset", str(int(offset)))
            else:
                # Fixed or non-coordinated programs need explicit phase realignment.
                self._realign_phase_to_offset(tlid, offset)

    def _current_program_base_offset(self, name):
        """Incumbent offset for the currently active time-of-day program."""
        programs = self._ga_offset_by_program.get(name, {})
        values = []
        for tlid in C.AGENT_TLIDS[name]:
            program = traci.trafficlight.getProgram(tlid)
            if program in programs:
                values.append(programs[program])
        if not values:
            return self._ga_offset[name]
        if len(set(values)) > 1:
            raise RuntimeError(
                f"Controllers for {name} have inconsistent active-program "
                f"base offsets: {values}")
        return values[0]

    def _init_split_applied(self):
        """Currently-applied maxDur per controllable phase = the freshly-loaded
        program (base). Seeds the change-guard so an a=0 apply matches and skips the
        setProgramLogic reset (which otherwise costs ~1% mph on the coordinated
        actuated program even when nothing changes)."""
        out = {}
        if not self._split_layout:
            return out
        for name, lay in self._split_layout.items():
            d = {int(lay["slots"][j]): int(round(lay["base"][j]))
                 for j in range(C.K_MAX_SPLIT) if lay["mask"][j] > 0.5}
            if d:
                out[name] = d
        return out

    def _apply_split(self, name, slot_actions):
        """SUM-PRESERVING green split over [mainline] + [side phases]. slot_actions[0]
        is the coordinated MAINLINE share; slot_actions[1:] are the side shares. The
        total green G = cycle - clearances is held fixed (cycle preserved), and the
        partition among all phases -- INCLUDING the mainline -- is optimized (see
        signal_phases.sum_preserving_split). Only side maxDurs are written; the
        coordinated phase absorbs G - sum(side), so raising the mainline share shrinks
        the side caps and gives the mainline more green. a=0 -> base plan.

        GUARD: setProgramLogic RESETS the coordinated-actuated runtime state (~-1% mph)
        every call, even when the maxDurs are unchanged. So rewrite the program ONLY
        when a target maxDur actually differs from what's applied. At a=0 (the
        difference-reward baseline) the target equals base -> no setProgramLogic ->
        the baseline stays the clean coordinated-actuated plan."""
        lay = self._split_layout.get(name)
        if not lay or sum(lay["mask"]) == 0:
            return
        target = sum_preserving_split(lay, slot_actions, C.K_MAX_SPLIT)
        if self._split_applied.get(name) == target:      # unchanged -> don't disturb
            return
        tlid = C.AGENTS[name]
        prog = traci.trafficlight.getProgram(tlid)
        logics = traci.trafficlight.getAllProgramLogics(tlid)
        logic = next((l for l in logics if getattr(l, "programID", None) == prog),
                     logics[-1])
        for idx, m in target.items():
            logic.phases[idx].maxDur = m
            logic.phases[idx].duration = m               # keep nominal within [min,max]
        traci.trafficlight.setProgramLogic(tlid, logic)
        self._split_applied[name] = target

    def _apply_action(self, action):
        # Per intersection: [offset (+ split slots)].
        #   RESIDUAL (default): offset = (GA + a*ACTION_HALF_RANGE) mod CYCLE
        #       -> a=0 reproduces the incumbent (the incumbent-anchored design).
        #   ABSOLUTE (RL_RESIDUAL=0, residual-action ablation): offset = (a+1)/2 * CYCLE
        #       -> the policy sets the offset outright, with NO anchoring to the
        #          incumbent (a=0 -> mid-cycle, not the GA plan); isolates the value
        #          of residual anchoring.
        #   split : per-phase maxDur rescale (see _apply_split); a=0 -> GA maxDur
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != C.ACTION_DIM:
            raise ValueError(
                f"Action has {action.size} entries, expected {C.ACTION_DIM}")
        action = (action * self.action_mask).reshape(
            C.N_AGENTS, C.N_ACT_PER_INT)
        offsets = {}
        for k, name in enumerate(C.AGENT_ORDER):
            # A fixed time-of-day plan can change program (and therefore its
            # compensated incumbent offset) at an hourly boundary. Resolve the
            # base from the live program before applying the residual action.
            self._ga_offset[name] = self._current_program_base_offset(name)
            if C.SPLIT_CONTROL and self._split_layout is not None:
                self._apply_split(name, action[k, 1:])   # rewrite maxDur first
            a = float(np.clip(action[k, 0], -1, 1))
            if C.RESIDUAL_ACTION:
                off = int(round(self._ga_offset[name] + a * C.ACTION_HALF_RANGE)) % C.CYCLE_S
            else:                                          # absolute offset (ablation)
                off = int(round((a + 1.0) * 0.5 * C.CYCLE_S)) % C.CYCLE_S
            offsets[name] = off
            self._apply_offset(name, off)              # then the coordination offset
        self._cur_offsets = offsets

    # ---------- simulation ----------
    def _sumo_cmd(self):
        b = checkBinary("sumo-gui" if self.gui else "sumo")
        additional = [self.sig] + list(self.extra_additional_files)
        if not USE_PROBE_COLLECTOR:
            additional.insert(0, C.DETECTORS)
        cmd = [b, "-c", C.SUMOCFG,
               "--additional-files", ",".join(additional),
               "--route-files", self._trips,
               "--end", str(self.episode_end)]
        if self.metrics_mode == "statistics":
            cmd += ["--statistic-output", self._statistics]
        else:
            cmd += ["--tripinfo-output", self._tripinfo,
                    "--tripinfo-output.write-unfinished", "true"]
        return cmd + ["--no-warnings", "true", "--no-step-log", "true",
                      "--seed", str(self.seed), "--time-to-teleport",
                      str(self.time_to_teleport)] + list(
                          self.extra_sumo_args)

    def _run(self, seconds):
        """Advance `seconds`, collecting probe segment data each sim step.

        Hot loop: the only unavoidable TraCI round trip per step is
        simulationStep(). dt is cached, sim_time is tracked locally, the teleport
        list rides the step subscription. Once no current or future vehicles remain,
        SUMO is advanced directly to the interval boundary. This preserves the fixed
        15-minute decision grid through the full episode without paying for empty
        one-second steps during the two-hour dissipation window.
        """
        dt = self._dt
        n = int(round(seconds / dt))
        t = traci.simulation.getTime()
        target = t + n * dt
        for i in range(n):
            if (t >= self._demand_end and (i & 0x3F) == 0
                    and traci.simulation.getMinExpectedNumber() <= 0):
                traci.simulationStep(target)
                return
            traci.simulationStep()
            t += dt
            collect_step_data(self.segments, self._entries, self._spd,
                              sim_time=t, step_length=dt)

    def _harvest(self):
        """Return {approach_id:{sid:(harmonic_speed, count, percentile_speeds)}}
        from buffers, then clear. The 21 percentiles (0..100 step 5) are the only
        distributional info real probe feeds give -> the reward's expected delay is
        computed from them (deployment-faithful)."""
        readings = {}
        for idx, seg in enumerate(self.segments):
            speeds = self._spd[idx]
            spd, cnt = _harmonic(speeds)
            pct = (list(np.percentile(speeds, C.SPEED_PERCENTILES))
                   if speeds else [])
            readings.setdefault(seg["approach_id"], {})[str(seg["sid"])] = (
                spd, cnt, pct)
            self._spd[idx].clear()
        if self.probe_observer is not None:
            self.probe_observer(
                readings, float(traci.simulation.getTime()))
        return readings

    def _policy_state(self, readings, time_fraction):
        """Construct the policy observation; comparator environments may override."""
        state, _ = build_state(
            readings, self.geom, self._cur_offsets, time_fraction)
        return state

    def reset(self):
        if self._started:
            traci.close()
        # Preferred: the assigned per-worker port (no intra-job free-port race). If it
        # is busy (leftover SUMO, or another job/user on a shared HPCC node -> SUMO
        # 'address already in use'), fall back to fresh OS-assigned free ports with
        # retry, so a port collision degrades gracefully instead of killing the job.
        import time as _t
        cmd = self._sumo_cmd()
        started = False
        last_error = None
        if self._port is not None:
            try:
                # Besides preventing an allocation race, the pre-bind check is a
                # correctness guard. traci.start() can otherwise connect to an old
                # SUMO already listening on this port after its new SUMO process
                # fails to bind, silently attaching this env to the wrong episode.
                with _traci_launch_lock():
                    if not _port_is_available(self._port):
                        raise RuntimeError(
                            f"preferred TraCI port {self._port} is already in use")
                    traci.start(cmd, port=self._port, numRetries=20)
                started = True
            except Exception as exc:
                last_error = exc
                try:
                    traci.close(False)
                except Exception:
                    pass
        for _ in range(30):
            if started:
                break
            try:
                # Hold the lock until TraCI is connected, proving that SUMO has
                # completed the bind before another worker probes for a port.
                with _traci_launch_lock():
                    fallback_port = getFreeSocketPort()
                    if not isinstance(fallback_port, int):
                        raise RuntimeError("could not allocate a fallback TraCI port")
                    traci.start(cmd, port=fallback_port, numRetries=20)
                started = True
            except Exception as exc:
                last_error = exc
                try:
                    traci.close(False)
                except Exception:
                    pass
                _t.sleep(0.3)
        if not started:
            raise RuntimeError(
                "SUMO failed to start after the preferred port and 30 serialized "
                f"fallback attempts (preferred={self._port}, last_error={last_error!r})"
            ) from last_error
        self._started = True
        self._dt = traci.simulation.getDeltaT()   # constant; cached for the hot loop
        # Probe geometry is static and cached per worker. The fast detector-based
        # fallback still discovers detector ids from the live SUMO instance.
        if USE_PROBE_COLLECTOR:
            self.segments = self._static_segments
            setup_subscriptions(self.segments)
        else:
            segs = parse_segments_v8(C.APPROACH_DIR)
            self.segments = build_segment_detectors_v8(segs)
            setup_subscriptions()
        self._entries = [dict() for _ in self.segments]
        self._spd = {i: [] for i in range(len(self.segments))}
        for name in C.AGENT_ORDER:
            self._ga_offset[name] = self._current_program_base_offset(name)
        self._cur_offsets = dict(self._ga_offset)   # before first action: GA plan
        self._split_applied = self._init_split_applied()  # program starts at base maxDur
        self._k = 0                                  # policy-controlled step index
        # First interval uses the incumbent action a=0. Its detector readings become
        # the first observation; the first learned action then controls the next
        # 15-minute interval.
        self._run(C.WARMUP_S)
        readings = self._harvest()
        self._last_readings = readings
        self._t = traci.simulation.getTime()
        return self._policy_state(readings, self._t / self.episode_end)

    def step(self, action):
        self._apply_action(action)
        self._run(C.DECISION_INTERVAL_S)
        readings = self._harvest()
        self._t = traci.simulation.getTime()
        _r, info = corridor_reward(readings, self.geom, return_breakdown=True)
        delay = info["total_delay_vehh"]
        # difference reward against the GA baseline at this decision step (positive
        # = better than the coordinated plan); falls back to plain -delay if no
        # baseline is set (e.g. while computing the baseline itself).
        if self.baseline is not None:
            if self._k >= len(self.baseline):
                raise RuntimeError(
                    f"Difference-reward baseline has {len(self.baseline)} steps, "
                    f"but the episode reached decision step {self._k}. Rebuild the "
                    "demand bank with the current episode configuration.")
            r = C.REWARD_SCALE * (self.baseline[self._k] - delay)
            info["baseline_delay"] = self.baseline[self._k]
        else:
            r = self.plain_reward_scale * (-delay)
        traffic_reward = float(r)
        normalized_deviations = []
        absolute_deviations_s = []
        for name in C.AGENT_ORDER:
            delta = (
                (self._cur_offsets[name] - self._ga_offset[name]
                 + C.CYCLE_S / 2.0)
                % C.CYCLE_S
                - C.CYCLE_S / 2.0
            )
            absolute_deviations_s.append(abs(float(delta)))
            normalized_deviations.append(
                float(delta / C.ACTION_HALF_RANGE) ** 2)
        # Under the paired difference reward, scale the full objective—not just
        # its traffic term—so RL_REWARD_SCALE changes numerical conditioning
        # without weakening the incumbent-offset prior. The plain-reward
        # ablation retains its independently tuned --plain-reward-scale.
        objective_scale = C.REWARD_SCALE if self.baseline is not None else 1.0
        offset_penalty = (
            objective_scale
            * C.OFFSET_DEVIATION_PENALTY
            * float(np.mean(normalized_deviations))
        )
        r = traffic_reward - offset_penalty
        info["traffic_reward"] = traffic_reward
        info["offset_deviation_penalty"] = offset_penalty
        info["mean_abs_offset_adjustment_s"] = float(
            np.mean(absolute_deviations_s))
        self._k += 1
        done = self._t >= self.episode_end
        state = self._policy_state(
            readings, min(self._t / self.episode_end, 1.0))
        info["offsets"] = dict(self._cur_offsets)
        # Evaluation diagnostics: expose the physical decision interval without
        # changing the state, reward, or termination semantics used for training.
        # The action chosen before this call is active over
        # [interval_start_s, interval_end_s).
        info["interval_end_s"] = float(self._t)
        info["interval_start_s"] = float(self._t - C.DECISION_INTERVAL_S)
        info["decision_step"] = int(self._k - 1)
        if done:
            # End of episode: close SUMO to flush the selected metrics output.
            self.close()
            info.update(self._read_metrics())
        return state, r, done, info

    @staticmethod
    def _read_tripinfo_file(path, retries=20, wait=0.25):
        """(VMT[mi], VHT[veh-h], n_veh, wait_s, loss_s) from a tripinfo file.
        routeLength[m], duration[s], waitingTime[s] and timeLoss[s] summed over all
        records (SUMO writes these per <tripinfo> by default).
          waitingTime = time standing (speed < 0.1 m/s); timeLoss = time lost vs the
          ideal (uncongested) travel time. Both are corridor totals in seconds.

        SUMO finishes writing the file slightly after traci.close() returns, so we
        wait until the closing </tripinfos> tag is present before parsing (avoids
        an intermittent ParseError on a half-written file)."""
        import time as _time
        txt = None
        for _ in range(retries):
            if os.path.exists(path):
                with open(path) as f:
                    txt = f.read()
                if txt.rstrip().endswith("</tripinfos>"):
                    break
            _time.sleep(wait)
        if not txt:
            return 0.0, 0.0, 0, 0.0, 0.0
        if not txt.rstrip().endswith("</tripinfos>"):   # never completed: salvage
            txt = txt[:txt.rfind("</tripinfo>") + len("</tripinfo>")] + "</tripinfos>"
        sum_len_m = sum_dur_s = sum_wait_s = sum_loss_s = 0.0
        n = 0
        for el in ET.fromstring(txt).iter("tripinfo"):
            sum_len_m += float(el.get("routeLength", 0.0) or 0.0)
            sum_dur_s += float(el.get("duration", 0.0) or 0.0)
            sum_wait_s += float(el.get("waitingTime", 0.0) or 0.0)
            sum_loss_s += float(el.get("timeLoss", 0.0) or 0.0)
            n += 1
        return sum_len_m / 1609.344, sum_dur_s / 3600.0, n, sum_wait_s, sum_loss_s

    def _read_tripinfo(self):
        vmt, vht, n, wait_s, loss_s = self._read_tripinfo_file(self._tripinfo)
        return self._metrics_dict(
            vmt, vht, n, wait_s, loss_s, source="tripinfo", complete=True)

    @staticmethod
    def _metrics_dict(vmt, vht, n, wait_s, loss_s, source, complete):
        return {
            "vmt_mi": vmt,
            "vht_h": vht,
            "vmt_per_vht": (vmt / vht if vht > 0 else 0.0),
            "n_veh": n,
            "wait_h": wait_s / 3600.0,
            "wait_s_per_veh": (wait_s / n if n > 0 else 0.0),
            "timeloss_h": loss_s / 3600.0,
            "timeloss_s_per_veh": (loss_s / n if n > 0 else 0.0),
            "metrics_source": source,
            "metrics_complete": bool(complete),
        }

    @staticmethod
    def _read_statistics_file(path, retries=20, wait=0.25):
        """Read SUMO's compact aggregate vehicleTripStatistics output."""
        import time as _time
        root = None
        for _ in range(retries):
            try:
                root = ET.parse(path).getroot()
                break
            except (ET.ParseError, OSError):
                _time.sleep(wait)
        if root is None:
            return 0.0, 0.0, 0, 0.0, 0.0, False
        trips = root.find("vehicleTripStatistics")
        vehicles = root.find("vehicles")
        if trips is None:
            return 0.0, 0.0, 0, 0.0, 0.0, False
        n = int(float(trips.get("count", 0) or 0))
        avg_len_m = float(trips.get("routeLength", 0.0) or 0.0)
        avg_dur_s = float(trips.get("duration", 0.0) or 0.0)
        avg_wait_s = float(trips.get("waitingTime", 0.0) or 0.0)
        avg_loss_s = float(trips.get("timeLoss", 0.0) or 0.0)
        running = int(float(vehicles.get("running", 0) or 0)) if vehicles is not None else 0
        return (avg_len_m * n / 1609.344,
                avg_dur_s * n / 3600.0,
                n,
                avg_wait_s * n,
                avg_loss_s * n,
                running == 0)

    def _read_metrics(self):
        path = self._statistics if self.metrics_mode == "statistics" else self._tripinfo
        try:
            if self.metrics_mode == "statistics":
                vmt, vht, n, wait_s, loss_s, complete = \
                    self._read_statistics_file(path)
                return self._metrics_dict(
                    vmt, vht, n, wait_s, loss_s,
                    source="statistics", complete=complete)
            return self._read_tripinfo()
        finally:
            if not self.keep_metrics_files:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass

    def set_metrics_mode(self, mode):
        """Select compact training statistics or exact per-vehicle tripinfo."""
        if mode not in ("tripinfo", "statistics"):
            raise ValueError(f"unknown metrics mode: {mode}")
        if self._started:
            raise RuntimeError("metrics mode can only change between episodes")
        self.metrics_mode = mode

    def set_demand(self, params):
        """Scale demand by ``(global, main, side, direction)`` and use the
        result as this env's route-file. params=None -> revert to the calibrated
        base demand. In EXP5 explicit-demand mode, materialize one deterministic
        vehicle list and change only its probe labels across penetration levels.
        Takes effect on the next reset()/episode."""
        from . import demand
        self.demand_params = params
        if demand.explicit_demand_enabled():
            g, m, s, d = (
                params if params is not None else (1.0, 1.0, 1.0, 0.5))
            self._trips = demand.generate_explicit_trips(
                self._trips_path, g, m, s, d,
                penetration=demand.probe_penetration(),
                seed=demand.explicit_demand_seed(),
                probe_seed=demand.probe_membership_seed())
        elif params is None:
            self._trips = C.TRIPS
        else:
            g, m, s, d = params
            self._trips = demand.generate_trips(
                self._trips_path, g, m, s, d)
        return self._trips

    def incumbent_action(self):
        """Policy-space action that reproduces the loaded incumbent plan.

        In residual mode this is the all-zero action. In absolute-offset mode,
        zero means 30 s, so each offset component must instead be the inverse of
        the absolute mapping. Split components remain zero, which preserves the
        base green partition.
        """
        action = np.zeros(
            (C.N_AGENTS, C.N_ACT_PER_INT), dtype=np.float32)
        if not C.RESIDUAL_ACTION:
            for k, name in enumerate(C.AGENT_ORDER):
                action[k, 0] = (
                    2.0 * float(self._ga_offset[name]) / C.CYCLE_S - 1.0)
        return action.reshape(-1)

    def compute_baseline(self):
        """Run one episode at the incumbent coordinated plan and store its
        per-decision-step delay as the difference-reward baseline.

        The incumbent action depends on the action parameterization: it is zero in
        residual mode and the inverse absolute-offset mapping otherwise.
        """
        self.baseline = None                      # plain -delay while measuring
        self.reset()
        base, done = [], False
        incumbent = self.incumbent_action()
        info = {}
        while not done:
            _s, _r, done, info = self.step(incumbent)
            base.append(info["total_delay_vehh"])
        self.baseline = base
        ref = {k: info[k] for k in ("vmt_mi", "vht_h", "vmt_per_vht", "n_veh",
                                    "wait_s_per_veh", "timeloss_s_per_veh")
               if k in info}
        return base, ref

    def close(self):
        if self._started:
            traci.close()
            self._started = False

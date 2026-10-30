"""
Central configuration for the SLE-ATSC reinforcement-learning corridor problem.

Problem (settled with the user):
  - Decision step = 15 min (matches the TomTom probe aggregation granularity).
  - Action = per-intersection OFFSET only (continuous, rounded to int at deploy);
    common cycle fixed at 60 s; green splits NOT optimized.
  - Three signal backbones: coordinated-fixed, coordinated-actuated, and
    coordinated-actuated with detector-call phase skipping. RL sets their offsets
    every 15 min without changing the local phase sequence.
  - State + reward are PROBE-ONLY (per-segment speed percentiles + probe count).
    Reward is detector-based (segment delay). True VMT/VHT is for offline
    evaluation only.
  - Start CENTRALIZED single-agent.  The legacy environment outputs 8 offsets;
    the opt-in all10 environment also controls Eucalyptus and Cranford.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ----- timing -----
DECISION_INTERVAL_S = 900          # 15 min control interval
WARMUP_S = 900                     # first interval runs incumbent action a=0, then yields obs
EPISODE_START_S = 0                # demand begins at 0 (= 7:00 AM)
EPISODE_END_S = 28800             # 8 h window: 32 intervals; policy controls intervals 2-32
CYCLE_S = 60                       # common cycle (fixed)

# ----- controlled intersections (centralized policy; corridor order West -> East) -----
# Keep the established 8-node environment as the default so old experiments and
# checkpoints remain reproducible.  The new all10 environment is selected before
# importing this module with RL_CONTROL_LAYOUT=all10 (see train_ppo_all10.py).
CONTROL_LAYOUT = os.environ.get("RL_CONTROL_LAYOUT", "legacy8")
if CONTROL_LAYOUT not in ("legacy8", "all10"):
    raise ValueError(
        f"RL_CONTROL_LAYOUT={CONTROL_LAYOUT!r}; expected 'legacy8' or 'all10'")

_ALL_AGENT_TLIDS = {
    "Lime":       ("54553323",),
    "Mulberry":   ("joinedS_12329405771_54512111",),
    "Park":       ("54424419",),
    "Victoria":   ("GS_cluster_5910940017_5910976489",),
    # Eucalyptus is one physical control node represented by directional SUMO TLS.
    # A single node offset action is applied to both controllers.
    "Eucalyptus": ("5910940029", "5910975175"),
    "Kansas":     ("GS_cluster_54287971_5910976488",),
    "Ottawa":     ("GS_cluster_5910940035_5910945467",),
    "Chicago":    ("GS_cluster_4689533026_54483626_5910945475_5910945476",),
    "Cranford":   ("4685892089",),
    "Iowa":       ("GS_cluster_5910933593_9013582988",),
}
_LEGACY_ORDER = [
    "Lime", "Mulberry", "Park", "Victoria",
    "Kansas", "Ottawa", "Chicago", "Iowa",
]
_ALL10_ORDER = [
    "Lime", "Mulberry", "Park", "Victoria", "Eucalyptus",
    "Kansas", "Ottawa", "Chicago", "Cranford", "Iowa",
]

AGENT_ORDER = _ALL10_ORDER if CONTROL_LAYOUT == "all10" else _LEGACY_ORDER
AGENT_TLIDS = {name: _ALL_AGENT_TLIDS[name] for name in AGENT_ORDER}
# Primary TLS is used for phase-layout parsing.  Multi-TLS node offset application
# must use AGENT_TLIDS instead.
AGENTS = {name: tlids[0] for name, tlids in AGENT_TLIDS.items()}
N_AGENTS = len(AGENT_ORDER)

# adjacency on the corridor chain (for later GAT; west<->east neighbours)
NEIGHBORS = {AGENT_ORDER[i]: [AGENT_ORDER[j]
             for j in (i - 1, i + 1) if 0 <= j < N_AGENTS]
             for i in range(N_AGENTS)}

# ----- approaches / state -----
DIRECTIONS = ["eb", "wb", "nb", "sb"]   # fixed 4 slots; absent approaches zero-filled
# per-direction probe features (all derivable from segment speed distributions
# and probe count)
DIR_FEATURES = ["v_ratio_stopbar", "v_ratio_mean", "v_ratio_min",
                "queue_len_norm", "n_probes_log", "staleness",
                "delay_ratio_norm"]
STATE_SCHEMA = "probe7_expected_delay_ratio_v1"
N_DIR_FEATURES = len(DIR_FEATURES)
QUEUE_VRATIO_THRESH = 0.5     # seg counts toward queue_len if speed < 0.5*freeflow
QUEUE_LEN_NORM_M = 200.0      # normalize queue length by this
VRATIO_CLIP = 1.3             # cap speed/freeflow
# Per-direction delay ratio delta = probe delay / probe free-flow travel time is
# transformed as log(1+delta)/log(1+3) and clipped to one. Thus delta=1 maps to
# 0.5 and the feature saturates when observed travel time reaches 4x free flow.
DELAY_RATIO_SATURATION = 3.0

# per-intersection extra context appended to the 4*7 direction block.
# Approach-presence mask follows DIRECTIONS order: [has_eb, has_wb, has_nb, has_sb].
# Offset is intentionally excluded from the observation; recurrent policies receive
# the previous action directly, and the fixed base offset is part of the environment.
N_APPROACH_MASK = len(DIRECTIONS)
N_CTX_PER_AGENT = N_APPROACH_MASK

# global state size
N_TOD = 2                     # time-of-day sin/cos over the 8 h window
STATE_DIM = N_AGENTS * (len(DIRECTIONS) * N_DIR_FEATURES + N_CTX_PER_AGENT) + N_TOD

# Opt-in Diaz-style detector state.  The centralized MLP sees every controlled
# intersection, so the target and its one-hop neighbors are already present in
# this non-duplicated corridor vector.  Each direction has advance/stop-bar
# flow and occupancy; nonexistent approaches remain zero.
RL_METHOD = os.environ.get("RL_METHOD", "probe")
if RL_METHOD not in ("probe", "diaz"):
    raise ValueError(f"unknown RL_METHOD={RL_METHOD!r}")
if RL_METHOD == "diaz":
    DIR_FEATURES = ["advance_flow", "advance_occupancy",
                    "stopbar_flow", "stopbar_occupancy"]
    N_DIR_FEATURES = len(DIR_FEATURES)
    N_CTX_PER_AGENT = 0
    N_TOD = 0
    STATE_SCHEMA = "diaz_near_stopbar_step_presence_occupancy_v1"
    STATE_DIM = N_AGENTS * len(DIRECTIONS) * N_DIR_FEATURES

# ----- action -----
# DELTA mode: policy outputs in [-1,1] -> ADJUSTMENT around the GA plan; a=0
# reproduces the GA/coordinated plan (the difference-reward baseline) and the actor
# is warm-started at a~0 (zero-init mu) so it begins at the GA plan.
#
# Per intersection the policy emits 1 OFFSET (+ a SUM-PRESERVING green-split block of
# 1 MAINLINE knob + K_MAX_SPLIT side knobs when RL_SPLIT=1). The K side slots are
# padded/masked to the real non-coordinated green phases (Lime/Kansas/Chicago/Iowa
# have 3, Park 2, Victoria/Ottawa/Mulberry 1; see
# signal_phases.build_split_layout):
#   offset : deployed = (GA_offset + a*ACTION_HALF_RANGE) mod CYCLE
#   split  : redistribute the FIXED total green G = cycle - clearances over the pool
#            [coordinated mainline] + [side phases] via exp-share softmax (see
#            signal_phases.sum_preserving_split). slot 0 = mainline share, slots 1..K
#            = side shares. Only side maxDurs are written; the coordinated phase
#            absorbs G - sum(side) so the mainline max green is optimized too WITHOUT
#            changing the cycle. a=0 -> base plan.
# Split is opt-in via the RL_SPLIT env var so spawned workers agree on the action
# size (like RL_COLLECTOR). Only meaningful on the actuated base.
SPLIT_CONTROL = os.environ.get("RL_SPLIT", "0") == "1"
K_MAX_SPLIT = 3                    # per-intersection SIDE split slots (pad+mask; corridor max)
N_SPLIT_PER_INT = 1 + K_MAX_SPLIT  # mainline share + K side shares
N_ACT_PER_INT = (1 + N_SPLIT_PER_INT) if SPLIT_CONTROL else 1   # offset (+ split block)
ACTION_DIM = N_AGENTS * N_ACT_PER_INT
OFFSET_MIN, OFFSET_MAX = 0, CYCLE_S - 1
ACTION_HALF_RANGE = CYCLE_S / 2.0   # +/- half a cycle (30 s) around the GA offset
SPLIT_HALF_RANGE = 0.6              # exp-share softmax gain for the green-split reallocation
# Residual (incumbent-anchored) vs ABSOLUTE offset action. Default residual: a=0
# reproduces the incumbent. RL_RESIDUAL=0 -> absolute offset (no anchoring); the
# residual-action ablation ("no micro-adjustment"). Opt-in via env var so spawned
# workers agree (like RL_SPLIT / RL_COLLECTOR).
RESIDUAL_ACTION = os.environ.get("RL_RESIDUAL", "1") == "1"
# Operational prior: retain the incumbent offset unless the traffic benefit is
# large enough to justify a change. The penalty is evaluated on the wrapped
# PHYSICAL offset deviation, normalized by half a cycle, so residual and absolute
# action parameterizations optimize the same objective.
OFFSET_DEVIATION_PENALTY = float(
    os.environ.get("RL_OFFSET_DEVIATION_PENALTY", "0.05"))
if OFFSET_DEVIATION_PENALTY < 0:
    raise ValueError("RL_OFFSET_DEVIATION_PENALTY must be nonnegative")

# ----- reward -----
# Paired mode: r = c_r * [delay difference - lambda_o*q].
# The plain-delay ablation retains its independently tuned numerical scale.
# The traffic term is the paired delay difference by default and all delay inputs
# come from probe readings:
#   delay_seg = N_seg * L_seg * max(0, 1/v_seg - 1/v_free)   [veh-hours]
# N_seg uses raw probe count (consistent relative scale; penetration is a const).
# Numerical scaling of the paired difference objective.  PPO standardizes the
# advantage, so this primarily places critic targets on a healthier scale; it
# does not change the optimum.  Both components of the paired objective are
# multiplied by the same factor so its internal trade-off is preserved.
REWARD_SCALE = float(os.environ.get("RL_REWARD_SCALE", "5.0"))
if REWARD_SCALE <= 0:
    raise ValueError("RL_REWARD_SCALE must be positive")
REWARD_MIN_SPEED_KPH = 1.0    # floor to avoid div-by-zero
USE_TRUE_REWARD = False       # if True (training only), use SUMO true delay instead
# Expected delay over the probe SPEED PERCENTILES (0..100 every 5% -> 21 values,
# the only distributional info real probe feeds provide). Delay ~ 1/v is non-linear,
# so the aggregate mean speed masks a slow tail. Integrate delay over quantile
# probability with the composite trapezoidal rule: the 0th/100th percentiles each
# receive weight 0.025 and the 19 interior percentiles each receive weight 0.05.
# D_seg = count * L_km * integral_0^1 max(0, 1/Q_v(u) - 1/v_free) du.
USE_EXPECTED_DELAY = True
SPEED_PERCENTILES = list(range(0, 101, 5))   # must match tsc_sumo_sim.PERCENTILES
EXPECTED_DELAY_QUADRATURE = "trapezoid_quantile_v1"

# ----- paths -----
APPROACH_DIR = os.path.join(ROOT, "data", "geojson-1-15-min-interval", "processed")
SUMOCFG = os.path.join(ROOT, "data", "corridor-osm-network", "osm.sumocfg")
NET = os.path.join(ROOT, "data", "corridor-osm-network", "osm_cr_tl.net.xml")
DETECTORS = os.path.join(ROOT, "data", "corridor-osm-network", "detectors_v8_fixed.add.xml")
# Base demand trips (probe/reg split baked in). RL_TRIPS overrides it so a penetration
# sweep can point every process at a re-split copy (see rl/make_penetration_trips.py).
TRIPS = os.environ.get("RL_TRIPS") or os.path.join(
    ROOT, "outputs", "calibration", "trips_turning_probe.xml")
# NEMA leading-left plan (make_nema_lead_left.py): phase order NS-L, NS-T,
# WE-L, WE-T with the coordinated mainline through ending the cycle. Green
# splits are preserved exactly and each time-of-day program is compensated
# independently, so its GA green window is unchanged; the coord_fixed baseline
# cache must still be rebuilt.
SIG_FIXED = os.path.join(
    ROOT, "data", "corridor-osm-network", "fixed_timing_optimized_nema.add.xml")
SIG_ACTUATED = os.path.join(
    ROOT, "data", "corridor-osm-network", "coordinated_actuated_nema.add.xml")
SIG_ACTUATED_SKIP = os.path.join(
    ROOT, "data", "corridor-osm-network",
    "coordinated_actuated_dual_loop_skipping_nema.add.xml")

# Canonical signal-backbone registry.  Keep the historical ``fixed`` and
# ``actuated`` names as CLI aliases, but write canonical ids into new checkpoints,
# baseline caches, and evaluation summaries so unlike controllers can never share
# an artifact by accident.
BACKBONES = {
    "coord_fixed": {
        "signal": SIG_FIXED,
        "actuated": False,
        "phase_skipping": False,
    },
    "coord_actuated": {
        "signal": SIG_ACTUATED,
        "actuated": True,
        "phase_skipping": False,
    },
    "coord_actuated_skip": {
        "signal": SIG_ACTUATED_SKIP,
        "actuated": True,
        "phase_skipping": True,
    },
}
BACKBONE_ALIASES = {
    "fixed": "coord_fixed",
    "actuated": "coord_actuated",
}
BACKBONE_CHOICES = tuple(BACKBONES)


def canonical_backbone(name):
    """Return the canonical backbone id, accepting the two legacy aliases."""
    value = name or "coord_actuated"
    value = BACKBONE_ALIASES.get(value, value)
    if value not in BACKBONES:
        allowed = sorted(set(BACKBONES) | set(BACKBONE_ALIASES))
        raise ValueError(f"unknown signal backbone {name!r}; expected one of {allowed}")
    return value


def rl_output_dir(backbone):
    """Backbone-isolated directory for training, evaluation, and bank artifacts."""
    return os.path.join(ROOT, "outputs", "rl", canonical_backbone(backbone))


def signal_path(backbone):
    return BACKBONES[canonical_backbone(backbone)]["signal"]


def backbone_is_actuated(backbone):
    return bool(BACKBONES[canonical_backbone(backbone)]["actuated"])
# offline-verification snapshot (per-15-min detector readings)
DETECTOR_CSV = os.path.join(ROOT, "outputs", "calibration", "detector_full_v8fixed.csv")


def read_program_offsets(sig_path):
    """Return ``{intersection: {program_id: offset_s}}`` from a signal plan.

    A time-of-day fixed plan may legitimately use a different offset for each
    program when phase-order changes require program-specific compensation.
    Multiple physical controllers assigned to one policy node must still agree
    whenever they share a program id.
    """
    import xml.etree.ElementTree as ET
    id2name = {
        tid: name
        for name, tlids in AGENT_TLIDS.items()
        for tid in tlids
    }
    out = {name: {} for name in AGENT_ORDER}
    for tl in ET.parse(sig_path).findall(".//tlLogic"):
        if tl.get("id") not in id2name:
            continue
        name = id2name[tl.get("id")]
        program = tl.get("programID", "")
        value = int(float(tl.get("offset", "0")))
        previous = out[name].get(program)
        if previous is not None and previous != value:
            raise ValueError(
                f"Physical control node {name!r} has conflicting offsets for "
                f"program {program!r} in {sig_path}: {previous} and {value}"
            )
        out[name][program] = value
    return out


def read_base_offsets(sig_path, *, allow_program_variation=False):
    """{intersection_name: GA offset (int s)} read from a signal add.xml.
    These are the coordinated/GA offsets the delta action adjusts around.
    When variation is allowed, the first declared program supplies the initial
    value and the live environment resolves later program changes dynamically.
    """
    by_program = read_program_offsets(sig_path)
    out = {
        name: next(iter(programs.values()), 0)
        for name, programs in by_program.items()
    }
    conflicts = {
        name: programs
        for name, programs in by_program.items()
        if len(set(programs.values())) > 1
    }
    if conflicts and not allow_program_variation:
        raise ValueError(
            f"Physical control node has program-specific base offsets in {sig_path}: "
            f"{conflicts}")
    return out

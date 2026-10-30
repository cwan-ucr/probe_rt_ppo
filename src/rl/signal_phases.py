"""
Parse the coordinated-actuated signal program into the per-intersection data the
green-split (maxDur) action needs.

Control principle (SUMO "coordinated phase absorbs the slack" pattern):
  Each controlled TL has ONE coordinated green phase whose maxDur ~= the whole
  cycle (it soaks up whatever the other phases gap out, holding cycleTime fixed).
  The RL adjusts the maxDur of the NON-coordinated green phases; the coordinated
  phase rebalances automatically, so the cycle (and thus the coordination) is
  preserved without manual renormalization.

What counts as a controllable phase:
  A phase is ACTUATABLE iff `maxDur > minDur` (SUMO can stretch it between those
  bounds). Clearance phases (yellow / all-red) have `minDur == maxDur == duration`
  and are excluded -- their letters can include a permissive 'g', so do NOT classify
  by the state string. The coordinated phase = the actuatable phase with the largest
  maxDur; the rest are the controllable non-coordinated splits.

Feasibility ceiling (per non-coord phase p):
  ceil_p = available_green - minDur_coord - sum_{q != p, non-coord} minDur_q
  i.e. p may grow only until every OTHER green phase (incl. the coordinated one)
  is squeezed to its own minDur. Staying within [minDur_p, ceil_p] guarantees the
  cycle stays feasible at 60 s for any action combination.

  available_green = cycle - sum(clearance durations)

Usage:
  python -m src.rl.signal_phases          # print the table
  from .signal_phases import load_signal_phases, build_split_layout
"""
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from . import config as C

DEFAULT_SIG = os.path.join(C.ROOT, "data", "corridor-osm-network",
                           "coordinated_actuated.add.xml")


@dataclass
class IntersectionPhases:
    name: str
    tlid: str
    present: bool                       # has a coord_act program in the file
    cycle: float = 0.0
    offset: int = 0
    n_phases: int = 0
    fixed_time: float = 0.0             # sum of clearance (min==max) durations
    available_green: float = 0.0        # cycle - fixed_time
    coord_phase: int = -1               # index of the coordinated (slack) green
    coord_mindur: float = 0.0           # minDur of the coordinated phase
    coord_base_green: float = 0.0       # nominal coord green = available_green - sum(side base)
    noncoord_phases: list = field(default_factory=list)   # controllable indices
    base_maxdur: dict = field(default_factory=dict)       # {idx: maxDur}
    mindur: dict = field(default_factory=dict)            # {idx: minDur}
    ceiling: dict = field(default_factory=dict)           # {idx: feasibility cap}

    @property
    def n_controllable(self):
        return len(self.noncoord_phases)


def _cycle_of(logic):
    """cycleTime is stored as a <param>; fall back to sum(duration)."""
    for p in logic.findall("param"):
        if p.get("key") == "cycleTime":
            return float(p.get("value"))
    return sum(float(ph.get("duration")) for ph in logic.findall("phase"))


def load_signal_phases(sig_path=DEFAULT_SIG, agents=None):
    """Return {name: IntersectionPhases} for every controlled intersection.

    Intersections absent from the file are returned with present=False and no
    controllable phases."""
    agents = agents or C.AGENTS
    tree = ET.parse(sig_path)
    by_id = {L.get("id"): L for L in tree.findall(".//tlLogic")}

    out = {}
    for name in C.AGENT_ORDER:
        tlid = agents[name]
        logic = by_id.get(tlid)
        if logic is None:
            out[name] = IntersectionPhases(name=name, tlid=tlid, present=False)
            continue

        phases = logic.findall("phase")
        cycle = _cycle_of(logic)

        # classify: actuatable iff maxDur > minDur (clearance phases have min==max)
        actuatable, fixed_time = [], 0.0
        mindur, maxdur = {}, {}
        for i, ph in enumerate(phases):
            d = float(ph.get("duration"))
            mn = float(ph.get("minDur", d))
            mx = float(ph.get("maxDur", d))
            mindur[i], maxdur[i] = mn, mx
            if mx > mn:
                actuatable.append(i)
            else:
                fixed_time += d

        # coordinated phase = actuatable green with the largest maxDur (~= cycle)
        coord = max(actuatable, key=lambda i: maxdur[i]) if actuatable else -1
        noncoord = [i for i in actuatable if i != coord]

        available_green = cycle - fixed_time
        # feasibility ceiling per non-coord phase
        ceiling = {}
        for p in noncoord:
            others_min = sum(mindur[q] for q in noncoord if q != p)
            coord_min = mindur[coord] if coord >= 0 else 0.0
            ceiling[p] = available_green - coord_min - others_min

        side_base_sum = sum(maxdur[i] for i in noncoord)
        out[name] = IntersectionPhases(
            name=name, tlid=tlid, present=True, cycle=cycle,
            offset=int(float(logic.get("offset", "0"))), n_phases=len(phases),
            fixed_time=fixed_time, available_green=available_green,
            coord_phase=coord,
            coord_mindur=(mindur[coord] if coord >= 0 else 0.0),
            coord_base_green=(available_green - side_base_sum if coord >= 0 else 0.0),
            noncoord_phases=noncoord,
            base_maxdur={i: maxdur[i] for i in noncoord},
            mindur={i: mindur[i] for i in noncoord},
            ceiling=ceiling,
        )
    return out


def build_split_layout(phases, k_max=None):
    """Flatten the per-intersection controllable phases into a fixed-width action
    block with a mask, for the centralized policy.

    Returns (k_max, layout) where layout[name] = {
        "slots":   [phase_idx or -1] * k_max,   # which phase each action slot drives
        "mask":    [1.0/0.0] * k_max,           # 1 = active slot
        "base":    [base_maxDur] * k_max,        # 0 for inactive
        "min":     [minDur] * k_max,
        "ceil":    [ceiling] * k_max,
    }
    Action vector for splits = concat over AGENT_ORDER of the k_max slots; the env
    applies only masked-in slots. k_max defaults to the corridor max."""
    if k_max is None:
        k_max = max((p.n_controllable for p in phases.values()), default=0)
    # Guard: k_max must cover every intersection's controllable phases. Otherwise the
    # padding `[-1]*(k_max - n)` collapses to [] for n > k_max and sum_preserving_split
    # (which loops range(k_max)) SILENTLY DROPS the extra phases and mis-sizes the green
    # budget. Fail loudly so a new corridor raises C.K_MAX_SPLIT instead of misbehaving.
    worst = max((phases[n].n_controllable for n in C.AGENT_ORDER), default=0)
    if worst > k_max:
        offenders = {n: phases[n].n_controllable for n in C.AGENT_ORDER
                     if phases[n].n_controllable > k_max}
        raise ValueError(
            f"build_split_layout: k_max={k_max} < max controllable phases {worst} "
            f"({offenders}); raise C.K_MAX_SPLIT to >= {worst} so the split action "
            f"covers every intersection (ACTION_DIM will grow accordingly).")
    layout = {}
    for name in C.AGENT_ORDER:
        p = phases[name]
        slots = list(p.noncoord_phases) + [-1] * (k_max - p.n_controllable)
        mask = [1.0] * p.n_controllable + [0.0] * (k_max - p.n_controllable)
        base = [p.base_maxdur.get(i, 0.0) for i in p.noncoord_phases] + \
               [0.0] * (k_max - p.n_controllable)
        mn = [p.mindur.get(i, 0.0) for i in p.noncoord_phases] + \
             [0.0] * (k_max - p.n_controllable)
        cl = [p.ceiling.get(i, 0.0) for i in p.noncoord_phases] + \
             [0.0] * (k_max - p.n_controllable)
        # coordinated (mainline) phase enters the split budget as a virtual slot so
        # its max green is optimized too (see sum_preserving_split). has_main = the
        # intersection has a coord phase AND >=1 controllable side (else no partition).
        has_main = 1.0 if (p.present and p.coord_phase >= 0 and p.n_controllable >= 1) else 0.0
        coord_ceil = p.available_green - sum(p.mindur.get(i, 0.0) for i in p.noncoord_phases)
        layout[name] = {"slots": slots, "mask": mask, "base": base,
                        "min": mn, "ceil": cl,
                        "has_main": has_main,
                        "coord_base": p.coord_base_green,
                        "coord_min": p.coord_mindur,
                        "coord_ceil": coord_ceil,
                        "avail": p.available_green}
    return k_max, layout


def build_policy_action_mask(layout):
    """Return the flattened policy mask for [offset, main, side_0..side_K].

    Offset is active at every control node.  Split entries are active only when
    the parsed signal program exposes the corresponding coordinated/main and
    non-coordinated phase.  A fixed-duration node therefore receives
    [1, 0, 0, ...]: its offset remains controllable while all split outputs are
    excluded from action sampling and the policy loss.
    """
    if not C.SPLIT_CONTROL:
        return np.ones(C.ACTION_DIM, dtype=np.float32)
    rows = []
    for name in C.AGENT_ORDER:
        lay = layout[name]
        rows.append(
            [1.0, float(lay.get("has_main", 0.0))]
            + [float(v) for v in lay["mask"]]
        )
    mask = np.asarray(rows, dtype=np.float32).reshape(-1)
    if mask.size != C.ACTION_DIM:
        raise ValueError(
            f"Action mask has {mask.size} entries, expected ACTION_DIM={C.ACTION_DIM}")
    return mask


def sum_preserving_split(lay, slot_actions, k_max, gain=None):
    """SUM-PRESERVING green split over ALL green phases -- the coordinated MAINLINE
    phase included -- holding the total green fixed so the cycle (and the offset's
    meaning) is preserved while the mainline/side partition is optimized.

    Budget: total green G = available_green = cycle - clearances is fixed. The pool is
    [coordinated mainline] + [controllable side phases], with base greens
    [coord_base_green, side_base...] (summing to G). The action vector is

        slot_actions[0]      -> a_main  (the coordinated MAINLINE phase's share)
        slot_actions[1..K]   -> a_side  (the K padded side slots; masked-in are used)

    Mapping (exp-share softmax over the pool): w_i = base_i * exp(gain * a_i);
    m_i = G * w_i / sum(w). a=0 -> m_i = base_i (baseline preserved exactly). Values
    are clipped to each phase's [min, ceil] with residual redistribution and integer
    largest-remainder rounding so the pool sums to round(G) exactly.

    Physical realization: only the SIDE maxDurs (force-off caps) are written; the
    coordinated phase stays the slack absorber (maxDur ~= cycle) and therefore
    automatically receives G - sum(side) = m_main. Raising a_main shrinks the side
    caps -> the mainline holds more green; lowering it does the reverse -- WITHOUT
    changing the cycle. A single-side intersection is no longer locked: it trades the
    mainline against its one side phase (1 DOF).

    Returns {side_phase_idx: maxDur_int} for the masked-in SIDE slots (the coordinated
    phase is not written). Empty if the intersection has no controllable side phase.
    """
    if gain is None:
        gain = C.SPLIT_HALF_RANGE
    side_idx, base, mn, ceil, act = [], [], [], [], []
    for j in range(k_max):
        if lay["mask"][j] < 0.5:
            continue
        side_idx.append(int(lay["slots"][j]))
        base.append(float(lay["base"][j]))
        mn.append(float(lay["min"][j]))
        ceil.append(float(lay["ceil"][j]))
        act.append(float(np.clip(slot_actions[1 + j], -1.0, 1.0)))   # +1: slot0 = a_main
    n_side = len(side_idx)
    if n_side == 0:
        return {}

    has_main = lay.get("has_main", 0.0) > 0.5
    if has_main:
        # pool = [coord] + sides; coord (index 0) is optimized but NOT written.
        pool_idx = [None] + side_idx
        base = [float(lay["coord_base"])] + base
        mn = [float(lay["coord_min"])] + mn
        ceil = [float(lay["coord_ceil"])] + ceil
        act = [float(np.clip(slot_actions[0], -1.0, 1.0))] + act
    else:                                        # no coord partition -> side-only pool
        pool_idx = list(side_idx)
    base = np.asarray(base); mn = np.asarray(mn); ceil = np.asarray(ceil)
    act = np.asarray(act)
    n = len(pool_idx)
    G = float(base.sum())
    if n == 1:                                   # nothing to trade against -> locked
        return {pool_idx[0]: int(round(base[0]))} if pool_idx[0] is not None else {}

    w = base * np.exp(gain * act)
    m = G * w / w.sum()
    for _ in range(3):                           # restore sum=G under [mn,ceil] box
        m = np.clip(m, mn, ceil)
        resid = G - float(m.sum())
        if abs(resid) < 1e-6:
            break
        room = (ceil - m) if resid > 0 else (m - mn)
        tot = float(room.sum())
        if tot < 1e-9:
            break
        m = m + resid * room / tot
    m = np.clip(m, mn, ceil)
    # integer largest-remainder rounding so the pool sums EXACTLY to round(G) within
    # the integer box (total green held to the second -> cycle preserved).
    lo = np.ceil(mn - 1e-9).astype(int)
    hi = np.floor(ceil + 1e-9).astype(int)
    fl = np.clip(np.floor(m).astype(int), lo, hi)
    target = int(min(max(round(G), int(lo.sum())), int(hi.sum())))
    resid = target - int(fl.sum())
    frac = m - np.floor(m)
    if resid > 0:
        for i in list(np.argsort(-frac)) + list(range(n)) * resid:
            if resid == 0:
                break
            if fl[i] < hi[i]:
                fl[i] += 1; resid -= 1
    elif resid < 0:
        for i in list(np.argsort(frac)) + list(range(n)) * (-resid):
            if resid == 0:
                break
            if fl[i] > lo[i]:
                fl[i] -= 1; resid += 1
    # write only the side phases; the coordinated phase absorbs the remainder
    return {pool_idx[i]: int(fl[i]) for i in range(n) if pool_idx[i] is not None}


if __name__ == "__main__":
    ph = load_signal_phases()
    print(f"{'name':9s} {'pres':4s} {'cyc':>4s} {'off':>4s} {'nph':>3s} "
          f"{'coord':>5s} {'ctrl':>4s}  non-coord [idx: min..base→ceil]")
    n_ctrl = 0
    for name in C.AGENT_ORDER:
        p = ph[name]
        if not p.present:
            print(f"{name:9s} {'NO':4s}    -    -   -      -    -   (no coord_act program)")
            continue
        segs = "  ".join(
            f"ph{i}[{p.mindur[i]:.0f}..{p.base_maxdur[i]:.0f}→{p.ceiling[i]:.0f}]"
            for i in p.noncoord_phases)
        print(f"{name:9s} {'yes':4s} {p.cycle:4.0f} {p.offset:4d} {p.n_phases:3d} "
              f"ph{p.coord_phase:<3d} {p.n_controllable:4d}  {segs}")
        n_ctrl += p.n_controllable
    k_max, _ = build_split_layout(ph)
    print(f"\nTotal controllable non-coord green phases = {n_ctrl}  |  k_max = {k_max}")
    print(f"Split action block = {C.N_AGENTS} agents x {k_max} = "
          f"{C.N_AGENTS * k_max} dims (masked to {n_ctrl} active)")

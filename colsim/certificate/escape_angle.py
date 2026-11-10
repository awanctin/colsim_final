from __future__ import annotations

import math

import numpy as np

from .geometry import (
    R_MIN_M,
    TWO_PI,
    ONE_NAUTICAL_MILE_M,
    RADAR_DETECTION_RANGE_NM,
    VISIBLE_RANGE_M,
    OVERTAKING_THRESHOLD,
    HEAD_ON_THRESHOLD,
    RECIPROCAL_THRESHOLD,
    TS_SPEED_RANGE_MPS,
    R_VESSEL_M,
    HULL_CONTACT_M,
    HOLD_RAD,
    GIVEWAY_MIN_APPARENT_RAD,
    DELTA_MIN_APPARENT_RAD,
    Vessel,
    _wrap,
)
from . import alpha as _alpha
from .duty import situation as _label
from .tube import clearance_many

COLREGS_RULE = {
    "Head-on": ("Rule 14", "give-way", -1, "both turn to STARBOARD, pass port-to-port"),
    "Crossing_GiveWay": (
        "Rule 15",
        "give-way",
        -1,
        "TS on your starboard: keep clear, turn STARBOARD / slow",
    ),
    "Crossing_StandOn": (
        "Rule 17",
        "stand-on",
        0,
        "TS on your port: hold course & speed (stand-on)",
    ),
    "Overtaking_GiveWay": (
        "Rule 13",
        "give-way",
        -1,
        "you are overtaking: keep clear, pass well off",
    ),
    "Being_Overtaken_StandOn": (
        "Rule 17",
        "stand-on",
        0,
        "TS is overtaking you: hold course & speed",
    ),
    "Safe": ("Rule 8", "none", 0, "no risk of collision: no action required"),
}

COMPLEMENT = {
    "Head-on": "Head-on",
    "Crossing_GiveWay": "Crossing_StandOn",
    "Crossing_StandOn": "Crossing_GiveWay",
    "Overtaking_GiveWay": "Being_Overtaken_StandOn",
    "Being_Overtaken_StandOn": "Overtaking_GiveWay",
    "Safe": "Safe",
}


def cpa(os_boat, ts_boat):
    """Compute TCPA, DCPA and relative speed of two vessels.

    Args:
        os_boat (Vessel): Own vessel
        ts_boat (Vessel): Target vessel

    Returns:
        tuple: TCPA (s), DCPA (m) and relative speed (m/s)
    """
    pO = np.array([os_boat.x, os_boat.y])
    pT = np.array([ts_boat.x, ts_boat.y])
    vO = os_boat.v * np.array([np.cos(os_boat.theta), np.sin(os_boat.theta)])
    vT = ts_boat.v * np.array([np.cos(ts_boat.theta), np.sin(ts_boat.theta)])
    vrel_vec = vT - vO
    r_vec = pT - pO
    vrel = float(np.hypot(vrel_vec[0], vrel_vec[1]))
    R = float(np.hypot(r_vec[0], r_vec[1]))
    if vrel < 1e-9:
        return np.inf, R, vrel
    cos_a = np.clip(np.dot(vrel_vec, -r_vec) / (vrel * R), -1.0, 1.0)
    dcpa = R * np.sqrt(1.0 - cos_a**2)
    tcpa = R * cos_a / vrel
    return tcpa, dcpa, vrel


def classify_colregs(os_boat, ts_boat, dcpa_gate=None, tcpa_gate=None):
    """Classify the COLREGs situation of own vessel toward a target.

    Args:
        os_boat (Vessel): Own vessel
        ts_boat (Vessel): Target vessel
        dcpa_gate (float): DCPA above which it is safe, m, or None
        tcpa_gate (float): TCPA above which it is safe, s, or None

    Returns:
        str: Situation name
    """
    ox, oy, ov, oth = os_boat.x, os_boat.y, os_boat.v, os_boat.theta
    tx, ty, tv, tth = ts_boat.x, ts_boat.y, ts_boat.v, ts_boat.theta
    dx, dy = tx - ox, ty - oy
    if dx * dx + dy * dy > VISIBLE_RANGE_M * VISIBLE_RANGE_M:
        return "Safe"
    rvx = tv * math.cos(tth) - ov * math.cos(oth)
    rvy = tv * math.sin(tth) - ov * math.sin(oth)
    vv = rvx * rvx + rvy * rvy
    if vv < 1e-12:
        tcpa = math.inf
    else:
        tcpa = -(rvx * dx + rvy * dy) / vv
        if tcpa < 0.0:
            return "Safe"
    if dcpa_gate is not None:
        dcpa = (
            math.hypot(dx, dy)
            if tcpa == math.inf
            else math.hypot(dx + tcpa * rvx, dy + tcpa * rvy)
        )
        if dcpa > dcpa_gate:
            return "Safe"
    if tcpa_gate is not None and tcpa > tcpa_gate:
        return "Safe"
    b_os_ts = math.remainder(math.atan2(dy, dx) - oth, TWO_PI)
    b_ts_os = math.remainder(math.atan2(-dy, -dx) - tth, TWO_PI)
    os_abaft_ts = abs(b_ts_os) > OVERTAKING_THRESHOLD
    ts_abaft_os = abs(b_os_ts) > OVERTAKING_THRESHOLD
    if os_abaft_ts and not ts_abaft_os:
        return "Overtaking_GiveWay"
    if ts_abaft_os and not os_abaft_ts:
        return "Being_Overtaken_StandOn"
    if os_abaft_ts and ts_abaft_os:
        return "Safe"
    if (
        abs(b_os_ts) < HEAD_ON_THRESHOLD
        and abs(b_ts_os) < HEAD_ON_THRESHOLD
        and abs(math.remainder(tth - oth, TWO_PI)) > RECIPROCAL_THRESHOLD
    ):
        return "Head-on"

    os_gives_way = b_os_ts < 0.0
    ts_gives_way = b_ts_os < 0.0
    if os_gives_way and not ts_gives_way:
        return "Crossing_GiveWay"
    if ts_gives_way and not os_gives_way:
        return "Crossing_StandOn"
    return "Safe"


def colregs_advice(os_boat, ts_boat, **gate):
    """Get the situation, rule, role, side and action toward a target.

    Args:
        os_boat (Vessel): Own vessel
        ts_boat (Vessel): Target vessel
        **gate (dict): DCPA and TCPA gates

    Returns:
        dict: Situation, rule, role, side and action
    """
    sit = classify_colregs(os_boat, ts_boat, **gate)
    rule, role, side, action = COLREGS_RULE[sit]
    return dict(situation=sit, rule=rule, role=role, side=side, action=action)


HORIZON_MIN_S = 60.0
HORIZON_MAX_S = 300.0


def _horizon_n(boats):
    """Pick a look ahead time from the TCPA of every pair.

    Args:
        boats (list): All vessels

    Returns:
        float: Look ahead time, s
    """
    tc = [
        cpa(boats[i], boats[j])[0]
        for i in range(len(boats))
        for j in range(i + 1, len(boats))
    ]
    return float(min(max([t for t in tc if t > 0] + [HORIZON_MIN_S]), HORIZON_MAX_S))


def _kappa(boat, T, react=0.0):
    """Compute the max heading change in the horizon after the reaction delay.

    Args:
        boat (Vessel): Vessel
        T (float): Look ahead time, s
        react (float): Share of the horizon lost to reaction

    Returns:
        float: Max heading change, rad
    """
    return float(min(boat.rate_of_turn * T * (1.0 - react), np.pi))


def _vel(boat, dpsi):
    """Get the velocity after a heading change.

    Args:
        boat (Vessel): Vessel
        dpsi (float): Heading change, rad

    Returns:
        ndarray: Velocity (vx, vy), m/s
    """
    h = boat.theta + dpsi
    return boat.v * np.array([np.cos(h), np.sin(h)])


def _compliant_intervals(situation, kappa, delta_min=None):
    """Get the heading changes allowed by one situation.

    Args:
        situation (str): Situation name
        kappa (float): Max heading change, rad
        delta_min (float): Smallest lawful turn, rad, or None

    Returns:
        list: Allowed intervals, rad
    """
    d = GIVEWAY_MIN_APPARENT_RAD if delta_min is None else float(delta_min)
    d = min(max(d, 0.0), kappa)
    if situation in ("Head-on", "Crossing_GiveWay"):
        return [(-kappa, -d)]
    if situation == "Overtaking_GiveWay":
        if d <= 0.0:
            return [(-kappa, kappa)]
        return [(-kappa, -d), (d, kappa)]
    if situation in ("Crossing_StandOn", "Being_Overtaken_StandOn"):
        h = min(HOLD_RAD, kappa)
        return [(-h, h)]
    return [(-kappa, kappa)]


def _intersect(list_a, list_b):
    """Intersect two lists of intervals.

    Args:
        list_a (list): First intervals
        list_b (list): Second intervals

    Returns:
        list: Common intervals
    """
    out = []
    for a0, a1 in list_a:
        for b0, b1 in list_b:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi - lo > 1e-9:
                out.append((lo, hi))
    return sorted(out)


def _total_width(intervals):
    """Sum the widths of intervals.

    Args:
        intervals (list): Pairs (low, high)

    Returns:
        float: Total width
    """
    return float(sum(b - a for a, b in intervals))


def _os_compliant_intervals(os_boat, threats, kappa0, delta_min=None, gate=None):
    """Get the heading changes allowed by every situation at once.

    Args:
        os_boat (Vessel): Own vessel
        threats (list): Target vessels
        kappa0 (float): Max heading change, rad
        delta_min (float): Smallest lawful turn, rad, or None
        gate (dict): Gate settings, or None

    Returns:
        list: Allowed intervals, rad
    """
    cur = [(-kappa0, kappa0)]
    for ts in threats:
        sit = colregs_advice(os_boat, ts, **(gate or {}))["situation"]
        duty = _compliant_intervals(sit, kappa0, delta_min)
        cur = _intersect(cur, duty)
        if not cur:
            return []
    return cur


def _in_cone_vec(z, V, r_d=HULL_CONTACT_M, T=math.inf):
    """Check which relative velocities lead to contact.

    Args:
        z (ndarray): Relative position, m
        V (ndarray): Relative velocities, m/s
        r_d (float): Contact distance, m
        T (float): Look ahead time, s

    Returns:
        ndarray: True where contact happens
    """
    V = np.asarray(V, float)
    vv = np.einsum("...i,...i->...", V, V)
    with np.errstate(divide="ignore", invalid="ignore"):
        tcpa = -(V @ z) / vv
    closing = (vv > 1e-12) & (tcpa > 0.0)
    miss = z + tcpa[..., None] * V
    dcpa = np.hypot(miss[..., 0], miss[..., 1])
    hit = closing & (dcpa <= r_d)
    if math.isinf(T):
        return hit
    with np.errstate(divide="ignore", invalid="ignore"):
        t_in = tcpa - np.sqrt(np.maximum(r_d * r_d - dcpa * dcpa, 0.0) / vv)
    return hit & (t_in <= T)


def _safe_mask(os_boat, threats, d0_grid, threat_offsets, r_d, T=math.inf):
    """Mark the own headings that avoid every target heading sample.

    Args:
        os_boat (Vessel): Own vessel
        threats (list): Target vessels
        d0_grid (ndarray): Own heading changes, rad
        threat_offsets (list): Heading changes per target, rad
        r_d (float): Contact distance, m
        T (float): Look ahead time, s

    Returns:
        ndarray: True where safe
    """
    V0 = np.stack([_vel(os_boat, d) for d in d0_grid])
    mask = np.ones(len(d0_grid), bool)
    for ts, offs in zip(threats, threat_offsets):
        z = np.array([os_boat.x - ts.x, os_boat.y - ts.y])
        Vj = np.stack([_vel(ts, dj) for dj in offs])
        rel = V0[:, None, :] - Vj[None, :, :]
        blocks = _in_cone_vec(z, rel, r_d, T).all(axis=1)
        mask &= ~blocks
    return mask


TUBE_RESOLUTION_M = 25.0


def _tube_times(os_boat, tubes, T):
    """Choose the time samples for the tube check.

    Args:
        os_boat (Vessel): Own vessel
        tubes (list): Target tubes
        T (float): Look ahead time, s

    Returns:
        ndarray: Times, s
    """
    rate = os_boat.v + max(tb.v_max * (1.0 + math.pi) for tb in tubes)
    n = max(int(math.ceil(T * rate / TUBE_RESOLUTION_M)), 1)
    return np.linspace(0.0, T, n + 1)


def _safe_mask_tube(os_boat, d0_grid, threats, tubes, r_d, T):
    """Mark the own headings that keep clear of every target tube.

    Args:
        os_boat (Vessel): Own vessel
        d0_grid (ndarray): Own heading changes, rad
        threats (list): Target vessels
        tubes (list): Target tubes
        r_d (float): Contact distance, m
        T (float): Look ahead time, s

    Returns:
        ndarray: True where safe
    """
    t = _tube_times(os_boat, tubes, T)
    h = os_boat.theta + np.asarray(d0_grid, float)
    px = os_boat.x + os_boat.v * np.cos(h)[:, None] * t[None, :]
    py = os_boat.y + os_boat.v * np.sin(h)[:, None] * t[None, :]

    mask = np.ones(len(h), bool)
    for ts, tb in zip(threats, tubes):
        mask &= (clearance_many(px, py, tb, ts, t[None, :]) >= r_d).all(axis=1)
    return mask


def _sample(intervals, n):
    """Spread n samples over intervals by width.

    Args:
        intervals (list): Pairs (low, high)
        n (int): Number of samples

    Returns:
        ndarray: Samples
    """
    live = [(a, b) for a, b in intervals if b - a > 1e-9]
    if not live:
        return np.array([intervals[0][0]] if intervals else [0.0], float)
    total = _total_width(live)
    out = []
    for a, b in live:
        k = max(int(round(n * (b - a) / total)), 1)
        out.append(np.linspace(a, b, k))
    return np.concatenate(out)


def escape_measure(
    os_boat,
    threats,
    ts2=None,
    T=None,
    react=0.0,
    n0=361,
    nj=41,
    r_d=HULL_CONTACT_M,
    delta_min=None,
    gate=None,
    tubes=None,
):
    """Measure the physical and compliant escape widths on a grid.

    Args:
        os_boat (Vessel): Own vessel
        threats (object): Target vessel or list of them
        ts2 (Vessel): Second target, or None
        T (float): Look ahead time, s, or None
        react (float): Share of the horizon lost to reaction
        n0 (int): Number of own heading samples
        nj (int): Number of target heading samples
        r_d (float): Contact distance, m
        delta_min (float): Smallest lawful turn, rad, or None
        gate (dict): Gate settings, or None
        tubes (list): Target tubes, or None

    Returns:
        dict: Horizon, reach, widths, class and flags
    """
    if ts2 is not None:
        threats = [threats, ts2]
    elif not isinstance(threats, (list, tuple)):
        threats = [threats]
    threats = list(threats)

    boats = [os_boat] + threats
    T = _horizon_n(boats) if T is None else T
    k0 = _kappa(os_boat, T, react)
    kj = [_kappa(t, T, react) for t in threats]

    comp_offs, phys_offs = [], []
    for t, kt in zip(threats, kj):
        duty = _compliant_intervals(
            colregs_advice(t, os_boat, **(gate or {}))["situation"], kt, delta_min
        )
        comp_offs.append(_sample(duty, nj))
        phys_offs.append(np.linspace(-kt, kt, nj))

    d0p = np.linspace(-k0, k0, n0)
    if tubes is None:
        mp = _safe_mask(os_boat, threats, d0p, phys_offs, r_d, T)
    else:
        mp = _safe_mask_tube(os_boat, d0p, threats, tubes, r_d, T)
    alpha_phys = mp.mean() * (2.0 * k0)

    Dc0 = _os_compliant_intervals(os_boat, threats, k0, delta_min, gate)
    if not Dc0:
        alpha_col, wc, safe_col = 0.0, 0.0, False
    else:
        wc = _total_width(Dc0)
        d0c = _sample(Dc0, n0)
        if tubes is None:
            mc = _safe_mask(os_boat, threats, d0c, comp_offs, r_d, T)
        else:
            mc = _safe_mask_tube(os_boat, d0c, threats, tubes, r_d, T)
        alpha_col = mc.mean() * wc
        safe_col = bool(mc.any())

    safe_phys = bool(mp.any())
    ics = not safe_phys
    deadlock = (not safe_col) and safe_phys
    cls = "ICS" if ics else ("deadlock" if deadlock else "escapable")
    return dict(
        T=T,
        kappa0=k0,
        n_threats=len(threats),
        alpha_phys=alpha_phys,
        alpha_col=alpha_col,
        alpha_over_kappa=(alpha_col / k0 if k0 > 1e-9 else 0.0),
        compliant_width=wc,
        alpha_frac_compliant=(alpha_col / wc if wc > 1e-9 else 0.0),
        deontic=(not Dc0),
        cls=cls,
        deadlock=deadlock,
        ics=ics,
    )


def wedge_endpoints_closedform(os_boat, ts, r_d=HULL_CONTACT_M):
    """Find the edges of the collision wedge of one target in closed form.

    Args:
        os_boat (Vessel): Own vessel
        ts (Vessel): Target vessel
        r_d (float): Contact distance, m

    Returns:
        list: Heading changes, rad
    """
    z = np.array([os_boat.x - ts.x, os_boat.y - ts.y])
    Z = float(np.hypot(*z))
    if Z < 1e-9:
        return []
    lam = np.arctan2(ts.y - os_boat.y, ts.x - os_boat.x)
    a_cone = np.arcsin(min(r_d / Z, 1.0))
    v0, vj, thj = os_boat.v, ts.v, ts.theta
    out = []
    for g in (lam - a_cone, lam + a_cone):
        s = (vj / v0) * np.sin(thj - g)
        if abs(s) <= 1.0:
            for base in (np.arcsin(s), np.pi - np.arcsin(s)):
                out.append(_wrap((g + base) - os_boat.theta))
    return out


def _cone_blocks(os_boat, ts, d, r_d):
    """Check whether one own heading leads to contact with a target.

    Args:
        os_boat (Vessel): Own vessel
        ts (Vessel): Target vessel
        d (float): Heading change, rad
        r_d (float): Contact distance, m

    Returns:
        bool: True if blocked
    """
    z = np.array([os_boat.x - ts.x, os_boat.y - ts.y])
    return bool(_in_cone_vec(z, _vel(os_boat, d) - _vel(ts, 0.0), r_d))


def _split_arc(a, b):
    """Split an arc that wraps past pi.

    Args:
        a (float): Arc start, rad
        b (float): Arc end, rad

    Returns:
        list: One or two intervals
    """
    if a <= b:
        return [(a, b)]
    return [(a, math.pi), (-math.pi, b)]


def _wedge_intervals(os_boat, ts, r_d=HULL_CONTACT_M, T=math.inf):
    """Get the forbidden heading changes for one target.

    Args:
        os_boat (Vessel): Own vessel
        ts (Vessel): Target vessel
        r_d (float): Contact distance, m
        T (float): Look ahead time, s

    Returns:
        list: Forbidden intervals, rad
    """
    if math.hypot(os_boat.x - ts.x, os_boat.y - ts.y) <= r_d:
        return [(-math.pi, math.pi)]
    ends = [float(e) for e in wedge_endpoints_closedform(os_boat, ts, r_d)]
    blocks = lambda d: _cone_blocks(os_boat, ts, d, r_d)
    if not math.isinf(T):
        ends += [float(e) for e in _alpha.horizon_breakpoints(os_boat, ts, S=r_d, T=T)]
        blocks = lambda d: _alpha.forbidden(d, os_boat, ts, S=r_d, T=T)
    ends = sorted(ends)
    if not ends:
        return [(-math.pi, math.pi)] if blocks(0.0) else []

    out = []
    for a, b in zip(ends, ends[1:] + [ends[0] + TWO_PI]):
        if b - a <= 1e-12:
            continue
        if not blocks(_wrap(0.5 * (a + b))):
            continue
        out += (
            [(-math.pi, math.pi)]
            if b - a >= TWO_PI - 1e-12
            else _split_arc(_wrap(a), _wrap(b))
        )
    return out


def _union_measure(forbidden, allowed):
    """Measure the allowed width left after removing the forbidden intervals.

    Args:
        forbidden (list): Forbidden intervals
        allowed (list): Allowed intervals

    Returns:
        float: Free width, rad
    """
    events = sorted([(a, 1) for a, _ in forbidden] + [(b, -1) for _, b in forbidden])
    covered, depth, start = [], 0, None
    for x, step in events:
        if depth == 0 and step > 0:
            start = x
        depth += step
        if depth == 0 and step < 0:
            covered.append((start, x))

    free = 0.0
    for lo, hi in allowed:
        blocked = sum(max(0.0, min(hi, b) - max(lo, a)) for a, b in covered)
        free += (hi - lo) - blocked
    return free


def escape_closed(
    os_boat, threats, T=None, react=0.0, r_d=HULL_CONTACT_M, delta_min=None, gate=None
):
    """Closed form version of escape_measure.

    Args:
        os_boat (Vessel): Own vessel
        threats (list): Target vessels
        T (float): Look ahead time, s, or None
        react (float): Share of the horizon lost to reaction
        r_d (float): Contact distance, m
        delta_min (float): Smallest lawful turn, rad, or None
        gate (dict): Gate settings, or None

    Returns:
        dict: Horizon, reach, widths, class and flags
    """
    threats = list(threats)
    T = _horizon_n([os_boat] + threats) if T is None else T
    k0 = _kappa(os_boat, T, react)
    forbidden = [iv for ts in threats for iv in _wedge_intervals(os_boat, ts, r_d, T)]

    alpha_phys = _union_measure(forbidden, [(-k0, k0)])
    Dc0 = _os_compliant_intervals(os_boat, threats, k0, delta_min, gate)
    alpha_col = _union_measure(forbidden, Dc0) if Dc0 else 0.0

    safe_phys, safe_col = alpha_phys > 1e-12, alpha_col > 1e-12
    cls = "ICS" if not safe_phys else ("escapable" if safe_col else "deadlock")
    return dict(
        T=T,
        kappa0=k0,
        n_threats=len(threats),
        alpha_phys=alpha_phys,
        alpha_col=alpha_col,
        compliant_width=_total_width(Dc0) if Dc0 else 0.0,
        cls=cls,
        deadlock=cls == "deadlock",
        ics=cls == "ICS",
    )


def _wedge_from_grid(os_boat, ts, k0, r_d=HULL_CONTACT_M, n=4001):
    """Find the forbidden heading runs of one target on a fine grid.

    Args:
        os_boat (Vessel): Own vessel
        ts (Vessel): Target vessel
        k0 (float): Max heading change, rad
        r_d (float): Contact distance, m
        n (int): Number of samples

    Returns:
        list: Forbidden intervals, rad
    """
    z = np.array([os_boat.x - ts.x, os_boat.y - ts.y])
    Vj = _vel(ts, 0.0)
    g = np.linspace(-k0, k0, n)
    inc = _in_cone_vec(z, np.stack([_vel(os_boat, d) for d in g]) - Vj, r_d)
    runs, i = [], 0
    while i < n:
        if inc[i]:
            j = i
            while j + 1 < n and inc[j + 1]:
                j += 1
            runs.append((g[i], g[j]))
            i = j + 1
        else:
            i += 1
    return runs


SITUATIONS = (
    "Head-on",
    "Crossing_GiveWay",
    "Crossing_StandOn",
    "Overtaking_GiveWay",
    "Being_Overtaken_StandOn",
    "Safe",
)

_SITUATION_ARCS = {
    "Head-on": [(-HEAD_ON_THRESHOLD, HEAD_ON_THRESHOLD)],
    "Crossing_GiveWay": [(-OVERTAKING_THRESHOLD, -HEAD_ON_THRESHOLD)],
    "Crossing_StandOn": [(HEAD_ON_THRESHOLD, OVERTAKING_THRESHOLD)],
    "Overtaking_GiveWay": [(-OVERTAKING_THRESHOLD, OVERTAKING_THRESHOLD)],
    "Being_Overtaken_StandOn": [
        (OVERTAKING_THRESHOLD, math.pi),
        (-math.pi, -OVERTAKING_THRESHOLD),
    ],
    "Safe": [(-math.pi, math.pi)],
}


def collision_ranges(v0, v, T, beta):
    """Compute the target distances that lead to contact at time T.

    Args:
        v0 (float): Own speed, m/s
        v (float): Target speed, m/s
        T (float): Time to contact, s
        beta (float): Target bearing from the own course, rad

    Returns:
        list: Distances, m
    """
    disc = v * v - (v0 * math.sin(beta)) ** 2
    if disc < 0.0:
        return []
    root = math.sqrt(disc)
    mid = v0 * math.cos(beta)
    return [T * z for z in (mid + root, mid - root) if T * z > 0.0]


def _closing_rates(v0, v, beta):
    """Compute the closing speeds that lead to contact.

    Args:
        v0 (float): Own speed, m/s
        v (float): Target speed, m/s
        beta (float): Target bearing from the own course, rad

    Returns:
        list: Closing speeds, m/s
    """
    disc = v * v - (v0 * math.sin(beta)) ** 2
    if disc < 0.0:
        return []
    root = math.sqrt(disc)
    mid = v0 * math.cos(beta)
    return [r for r in (mid + root, mid - root) if r > 1e-9]


def _sample_arc(rng, arcs):
    """Draw an angle uniformly over a set of arcs.

    Args:
        rng (Generator): Random generator
        arcs (list): Pairs (low, high), rad

    Returns:
        float: Angle, rad
    """
    widths = [hi - lo for lo, hi in arcs]
    k = (
        0
        if len(arcs) == 1
        else int(rng.choice(len(arcs), p=np.array(widths) / sum(widths)))
    )
    lo, hi = arcs[k]
    return float(rng.uniform(lo, hi))


def place_threat(
    rng,
    os_b,
    situation,
    *,
    name="TS1",
    t_range=(60.0, 3600.0),
    v_range=(5.0, 15.0),
    z_range=(1500.0, 7000.0),
    jitter=0.10,
    ts_speed_cap=False,
    max_tries=24,
):
    """Place a target that meets own vessel in a given situation.

    Args:
        rng (Generator): Random generator
        os_b (Vessel): Own vessel
        situation (str): Situation name
        name (str): Target name
        t_range (tuple): Time to contact range, s
        v_range (tuple): Target speed range, m/s
        z_range (tuple): Start distance range, m
        jitter (float): Relative noise on the aim point
        ts_speed_cap (bool): Whether to cap the target speed at own speed
        max_tries (int): Max tries

    Returns:
        tuple: Target (or None) and number of tries
    """
    v_lo, v_hi = v_range
    if ts_speed_cap:
        v_hi = min(v_hi, os_b.v)
    if situation == "Overtaking_GiveWay":
        v_hi = min(v_hi, 0.95 * os_b.v)
    elif situation == "Being_Overtaken_StandOn":
        v_lo = max(v_lo, 1.05 * os_b.v)
    if v_hi <= v_lo:
        return None, 0
    arcs = _SITUATION_ARCS[situation]
    for tries in range(1, max_tries + 1):
        v = float(rng.uniform(v_lo, v_hi))
        if v < os_b.v:
            b_max = math.asin(min(1.0, v / os_b.v))
            clipped = [(max(lo, -b_max), min(hi, b_max)) for lo, hi in arcs]
            clipped = [(lo, hi) for lo, hi in clipped if hi > lo]
            if not clipped:
                continue
            beta = _sample_arc(rng, clipped)
        else:
            beta = _sample_arc(rng, arcs)
        lam = os_b.theta + beta
        if situation == "Safe":
            Z = float(rng.uniform(*z_range))
            p = (os_b.x + Z * math.cos(lam), os_b.y + Z * math.sin(lam))
            theta = float(_wrap(lam + rng.uniform(-0.6, 0.6)))
            ts = Vessel(name, p[0], p[1], v, theta, os_b.r_min)
            if _label(os_b, ts) == "Safe":
                return ts, tries
            continue
        Z = float(rng.uniform(*z_range))
        p = (os_b.x + Z * math.cos(lam), os_b.y + Z * math.sin(lam))
        for denom in _closing_rates(os_b.v, v, beta):
            T = Z / denom
            if not (t_range[0] <= T <= t_range[1]):
                continue
            p_star = (
                os_b.x + os_b.v * T * math.cos(os_b.theta),
                os_b.y + os_b.v * T * math.sin(os_b.theta),
            )
            theta = math.atan2(p_star[1] - p[1], p_star[0] - p[0])
            theta = float(_wrap(theta + rng.uniform(-jitter, jitter)))
            ts = Vessel(name, p[0], p[1], v, theta, os_b.r_min)
            if _label(os_b, ts) == situation:
                return ts, tries
    return None, max_tries


def place_encounter(rng, situations, *, os_theta=None, v0_range=(6.0, 13.0), **kw):
    """Place own vessel and one target per situation.

    Args:
        rng (Generator): Random generator
        situations (list): Situation names
        os_theta (float): Own heading, rad, or None for random
        v0_range (tuple): Own speed range, m/s
        **kw (dict): Extra settings for place_threat

    Returns:
        tuple: Own vessel, targets and placed situations
    """
    v0 = float(rng.uniform(*v0_range))
    theta0 = (
        float(rng.uniform(-math.pi, math.pi)) if os_theta is None else float(os_theta)
    )
    os_b = Vessel("OS", 0.0, 0.0, v0, theta0, R_MIN_M)
    threats, placed = [], []
    for k, sit in enumerate(situations, start=1):
        ts, _ = place_threat(rng, os_b, sit, name=f"TS{k}", **kw)
        if ts is None:
            if k == 1:
                return os_b, [], []
            continue
        threats.append(ts)
        placed.append(sit)
    return os_b, threats, placed


OS_SPEED_RANGE_MPS = (5.14, 9.26)
TCPA_RANGE_S = (120.0, 360.0)
SPAWN_RANGE_M = (1500.0, 7000.0)
AIM_JITTER_RAD = 0.25


def _rand_encounter(
    rng, n_threats=2, on_collision=0.75, ts_speed_cap=False, os_theta=None
):
    """Draw a random encounter where targets aim near a common point.

    Args:
        rng (Generator): Random generator
        n_threats (int): Number of targets
        on_collision (float): Share of targets aimed at own vessel
        ts_speed_cap (bool): Whether to cap the target speed at own speed
        os_theta (float): Own heading, rad, or None for random

    Returns:
        tuple: Own vessel and targets
    """
    v0 = rng.uniform(*OS_SPEED_RANGE_MPS)
    theta0 = float(rng.uniform(-np.pi, np.pi)) if os_theta is None else float(os_theta)
    os_b = Vessel("OS", 0.0, 0.0, v0, theta0, R_MIN_M)
    Tc = rng.uniform(*TCPA_RANGE_S)
    p_star = np.array(
        [os_b.x + v0 * Tc * np.cos(os_b.theta), os_b.y + v0 * Tc * np.sin(os_b.theta)]
    )
    ts = []
    for k in range(1, n_threats + 1):
        Z = rng.uniform(*SPAWN_RANGE_M)
        lam = os_b.theta + rng.uniform(-np.pi, np.pi)
        p = np.array([Z * np.cos(lam), Z * np.sin(lam)])
        v_lo, v_hi = TS_SPEED_RANGE_MPS
        v = rng.uniform(v_lo, v0 if ts_speed_cap else v_hi)
        if rng.random() < on_collision:
            d = p_star - p
            th = np.arctan2(d[1], d[0]) + rng.uniform(-AIM_JITTER_RAD, AIM_JITTER_RAD)
        else:
            th = rng.uniform(-np.pi, np.pi)
        ts.append(
            Vessel(f"TS{k}", float(p[0]), float(p[1]), float(v), float(th), R_MIN_M)
        )
    return os_b, ts

from __future__ import annotations

import math
from enum import Enum

import numpy as np

from .geometry import (
    GIVEWAY_MIN_APPARENT_RAD,
    HOLD_RAD,
    HULL_CONTACT_M,
    VISIBLE_RANGE_M,
)

KAPPA_MAX_RAD = 0.55
DELTA_MIN_RAD = GIVEWAY_MIN_APPARENT_RAD
HOLD_BAND_RAD = HOLD_RAD
HORIZON_S = 240.0
SAFETY_RADIUS_M = HULL_CONTACT_M

O_ABAFT_RAD = np.radians(112.5)
O_DOUBT_RAD = np.radians(5.0)
PSI_RULE13_RAD = np.radians(30.0)
R_RECIPROCAL_RAD = np.radians(170.0)
H_HEADON_RAD = np.radians(6.0)

GIVE_WAY = ("OGW", "HO", "CGW")
STAND_ON = ("BO", "CSO")
NO_DUTY = "NONE"

PRIORITY = {"OGW": 1, "BO": 1, "HO": 2, "CGW": 3, "CSO": 3, NO_DUTY: 4}


def wrap(a):
    """Wrap an angle into [-pi, pi].

    Args:
        a (float): Angle, rad

    Returns:
        float: Wrapped angle, rad
    """
    return math.remainder(a, 2.0 * math.pi)


def pair_angles(os_boat, ts):
    """Compute the bearing, relative heading and back bearing of a target.

    Args:
        os_boat (Vessel): Own vessel
        ts (Vessel): Target vessel

    Returns:
        tuple: Bearing, relative heading and back bearing, rad
    """
    a_j = math.atan2(ts.y - os_boat.y, ts.x - os_boat.x)
    beta = wrap(a_j - os_boat.theta)
    psi = wrap(ts.theta - os_boat.theta)
    b = wrap(beta + math.pi - psi)
    return beta, psi, b


def situations_firing(
    os_boat,
    ts,
    *,
    O=O_ABAFT_RAD,
    Psi=PSI_RULE13_RAD,
    R=R_RECIPROCAL_RAD,
    H=H_HEADON_RAD,
    D=O_DOUBT_RAD,
):
    """List every COLREGs situation whose test holds.

    Args:
        os_boat (Vessel): Own vessel
        ts (Vessel): Target vessel
        O (float): Abaft the beam angle, rad
        Psi (float): Max heading gap for overtaking, rad
        R (float): Min heading gap for head on, rad
        H (float): Max bearing for head on, rad
        D (float): Margin of doubt, rad

    Returns:
        list: Role codes
    """
    beta, psi, b = pair_angles(os_boat, ts)
    faster = os_boat.v > ts.v
    slower = ts.v > os_boat.v
    out = []
    if abs(b) > O - D and abs(psi) < Psi and faster:
        out.append("OGW")
    if abs(beta) > O - D and abs(psi) < Psi and slower:
        out.append("BO")
    if abs(beta) < H and abs(psi) > R:
        out.append("HO")
    if -O < beta < 0:
        out.append("CGW")
    if 0 <= beta < O:
        out.append("CSO")
    return out


OVERTAKE_ROLES = ("OGW", "BO")
KRASOWSKI_T_CHECK_S = 420.0


class OvertakeExit(Enum):
    OPENING = "opening"
    DISTANCE = "distance"


OVERTAKE_EXIT = OvertakeExit.OPENING
OVERTAKE_EXIT_M = 1852.0
BEAM_RAD = math.pi / 2.0


def role(os_boat, ts, *, gate=None, memory=None, **thresholds):
    """Get the duty role of own vessel toward a target.

    Args:
        os_boat (Vessel): Own vessel
        ts (Vessel): Target vessel
        gate (float): DCPA above which there is no duty, m, or None
        memory (str): Stored overtaking role, or None
        **thresholds (dict): Angle thresholds of the situation tests

    Returns:
        str: Role code
    """
    if gate is not None and straight_dcpa(os_boat, ts) > float(gate):
        return NO_DUTY
    if memory in OVERTAKE_ROLES:
        return memory
    if gate is not None and not closing(os_boat, ts, HORIZON_S):
        return NO_DUTY
    firing = situations_firing(os_boat, ts, **thresholds)
    if not firing:
        return NO_DUTY
    return min(firing, key=lambda r: PRIORITY[r])


def finally_past_and_clear(overtaker, overtaken):
    """Check that an overtaking is over.

    Args:
        overtaker (Vessel): Vessel overtaking
        overtaken (Vessel): Vessel overtaken

    Returns:
        bool: True if past and clear
    """
    bearing = wrap(
        math.atan2(overtaker.y - overtaken.y, overtaker.x - overtaken.x)
        - overtaken.theta
    )
    if abs(bearing) >= BEAM_RAD:
        return False
    dx, dy = overtaker.x - overtaken.x, overtaker.y - overtaken.y
    rvx = overtaker.v * math.cos(overtaker.theta) - overtaken.v * math.cos(
        overtaken.theta
    )
    rvy = overtaker.v * math.sin(overtaker.theta) - overtaken.v * math.sin(
        overtaken.theta
    )
    closing = -(dx * rvx + dy * rvy) / max(math.hypot(dx, dy), 1e-9)
    if closing > 0.0:
        return False
    if OVERTAKE_EXIT is OvertakeExit.OPENING:
        return True
    return math.hypot(dx, dy) > OVERTAKE_EXIT_M


def update_overtaking(memory, os_boat, ts, *, gate=None, **thresholds):
    """Update the stored overtaking role.

    Args:
        memory (str): Stored role, or None
        os_boat (Vessel): Own vessel
        ts (Vessel): Target vessel
        gate (float): DCPA gate, m, or None
        **thresholds (dict): Angle thresholds of the situation tests

    Returns:
        str: New stored role, or None
    """
    if memory in OVERTAKE_ROLES:
        overtaker, overtaken = (os_boat, ts) if memory == "OGW" else (ts, os_boat)
        return None if finally_past_and_clear(overtaker, overtaken) else memory
    r = role(os_boat, ts, gate=gate, **thresholds)
    return r if r in OVERTAKE_ROLES else None


def straight_dcpa(os_boat, ts):
    """Compute the DCPA if both vessels keep course and speed.

    Args:
        os_boat (Vessel): Own vessel
        ts (Vessel): Target vessel

    Returns:
        float: DCPA, m
    """
    dx, dy = ts.x - os_boat.x, ts.y - os_boat.y
    rvx = ts.v * math.cos(ts.theta) - os_boat.v * math.cos(os_boat.theta)
    rvy = ts.v * math.sin(ts.theta) - os_boat.v * math.sin(os_boat.theta)
    vv = rvx * rvx + rvy * rvy
    if vv < 1e-12:
        return math.hypot(dx, dy)
    t = -(rvx * dx + rvy * dy) / vv
    if t <= 0.0:
        return math.hypot(dx, dy)
    return math.hypot(dx + t * rvx, dy + t * rvy)


def closing(os_boat, ts, horizon=math.inf):
    """Check that the CPA is ahead in time and within the horizon.

    Args:
        os_boat (Vessel): Own vessel
        ts (Vessel): Target vessel
        horizon (float): Max TCPA, s

    Returns:
        bool: True if closing
    """
    dx, dy = ts.x - os_boat.x, ts.y - os_boat.y
    rvx = ts.v * math.cos(ts.theta) - os_boat.v * math.cos(os_boat.theta)
    rvy = ts.v * math.sin(ts.theta) - os_boat.v * math.sin(os_boat.theta)
    vv = rvx * rvx + rvy * rvy
    if vv < 1e-12:
        return False
    t = -(rvx * dx + rvy * dy) / vv
    return 0.0 < t <= horizon


SITUATION_NAME = {
    "HO": "Head-on",
    "CGW": "Crossing_GiveWay",
    "OGW": "Overtaking_GiveWay",
    "CSO": "Crossing_StandOn",
    "BO": "Being_Overtaken_StandOn",
}


def situation(a, b, gate=None):
    """Get the situation name of a toward b.

    Args:
        a (Vessel): First vessel
        b (Vessel): Second vessel
        gate (float): DCPA gate, m, or None

    Returns:
        str: Situation name, or Safe
    """
    dx, dy = b.x - a.x, b.y - a.y
    if dx * dx + dy * dy > VISIBLE_RANGE_M * VISIBLE_RANGE_M or not closing(a, b):
        return "Safe"
    return SITUATION_NAME.get(role(a, b, gate=gate), "Safe")


def kappa(vessel, T=HORIZON_S, kappa_max=KAPPA_MAX_RAD):
    """Compute the largest heading change a vessel can make in the horizon.

    Args:
        vessel (Vessel): Vessel
        T (float): Look ahead time, s
        kappa_max (float): Upper limit, rad

    Returns:
        float: Max heading change, rad
    """
    rt = getattr(vessel, "rate_of_turn", None)
    hull = float("inf") if rt is None else rt * T
    return float(min(hull, kappa_max, math.pi))


def duty_intervals(
    situation, k, delta_min=DELTA_MIN_RAD, h=HOLD_BAND_RAD, relaxed=False
):
    """Get the heading changes allowed by one role.

    Args:
        situation (str): Role code
        k (float): Max heading change, rad
        delta_min (float): Smallest lawful turn, rad
        h (float): Hold band, rad
        relaxed (bool): Whether Rule 17 is relaxed

    Returns:
        list: Allowed intervals, rad
    """
    d = min(max(float(delta_min), 0.0), k)
    hh = min(max(float(h), 0.0), k)
    if situation in ("HO", "CGW"):
        return [(-k, -d)]
    if situation == "OGW":
        return [(-k, -d), (d, k)]
    if situation in ("CSO", "BO"):
        if not relaxed:
            return [(-hh, hh)]
        allowed = [(-k, -d), (-hh, hh)]
        if situation == "BO":
            allowed.append((d, k))
        return _merge(allowed)
    return [(-k, k)]


def _merge(intervals):
    """Sort and join intervals that overlap.

    Args:
        intervals (list): Pairs (low, high)

    Returns:
        list: Merged intervals
    """
    out = []
    for lo, hi in sorted(intervals):
        if hi - lo <= 1e-12:
            continue
        if out and lo <= out[-1][1] + 1e-12:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return [tuple(iv) for iv in out]


def _intersect(a, b):
    """Intersect two lists of intervals.

    Args:
        a (list): First intervals
        b (list): Second intervals

    Returns:
        list: Common intervals
    """
    out = []
    for a0, a1 in a:
        for b0, b1 in b:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi - lo > 1e-12:
                out.append((lo, hi))
    return sorted(out)


def lawful_intervals(
    os_boat,
    threats,
    *,
    T=HORIZON_S,
    kappa_max=KAPPA_MAX_RAD,
    delta_min=DELTA_MIN_RAD,
    h=HOLD_BAND_RAD,
    gate=None,
    relaxed=None,
    memory=None,
    **thresholds,
):
    """Get the heading changes allowed by every duty at once.

    Args:
        os_boat (Vessel): Own vessel
        threats (list): Target vessels
        T (float): Look ahead time, s
        kappa_max (float): Max heading change, rad
        delta_min (float): Smallest lawful turn, rad
        h (float): Hold band, rad
        gate (float): DCPA gate, m, or None
        relaxed (list): Relaxation flag per target, or None
        memory (list): Stored role per target, or None
        **thresholds (dict): Angle thresholds of the situation tests

    Returns:
        list: Lawful intervals, rad, empty if none
    """
    k = kappa(os_boat, T, kappa_max)
    cur = [(-k, k)]
    for i, ts in enumerate(threats):
        m = memory[i] if memory is not None and i < len(memory) else None
        r = role(os_boat, ts, gate=gate, memory=m, **thresholds)
        rx = bool(relaxed[i]) if relaxed is not None and i < len(relaxed) else False
        cur = _intersect(cur, duty_intervals(r, k, delta_min, h, rx))
        if not cur:
            return []
    return cur


def roles(os_boat, threats, *, gate=None, memory=None, **thresholds):
    """Get the role toward every target.

    Args:
        os_boat (Vessel): Own vessel
        threats (list): Target vessels
        gate (float): DCPA gate, m, or None
        memory (list): Stored role per target, or None
        **thresholds (dict): Angle thresholds of the situation tests

    Returns:
        list: Role codes
    """
    memory = memory or [None] * len(threats)
    return [
        role(os_boat, ts, gate=gate, memory=m, **thresholds)
        for ts, m in zip(threats, memory)
    ]


def conflict(role_list, delta_min=DELTA_MIN_RAD, h=HOLD_BAND_RAD):
    """Check whether give way and stand on duties clash.

    Args:
        role_list (list): Role codes
        delta_min (float): Smallest lawful turn, rad
        h (float): Hold band, rad

    Returns:
        bool: True if the duties clash
    """
    gw = any(r in GIVE_WAY for r in role_list)
    so = any(r in STAND_ON for r in role_list)
    return bool(gw and so and (delta_min > h))


def balance_index(role_list):
    """Measure how even the give way and stand on roles are.

    Args:
        role_list (list): Role codes

    Returns:
        float: 1 if even, 0 if one sided, None if no duty
    """
    gw = sum(1 for r in role_list if r in GIVE_WAY)
    so = sum(1 for r in role_list if r in STAND_ON)
    if gw + so == 0:
        return None
    return 1.0 - abs(gw - so) / (gw + so)


def n_conflict_labelings(n, ordered=True):
    """Count the role labellings of n targets that hold a clash.

    Args:
        n (int): Number of targets
        ordered (bool): Whether target order matters

    Returns:
        int: Number of labellings
    """
    if ordered:
        return 5**n - 3**n - 2**n
    from math import comb

    return comb(n + 4, 4) - comb(n + 2, 2) - (n + 1)

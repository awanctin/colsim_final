from __future__ import annotations

import math

import numpy as np

from ..certificate.escape_angle import HULL_CONTACT_M, ONE_NAUTICAL_MILE_M

V_MAX_MPS = 20.0
DELTA_HEAD_ON_RAD = math.radians(5.0)
DELTA_NO_TURN_RAD = math.radians(10.0)
DELTA_LARGE_TURN_RAD = math.radians(20.0)
T_CHECK_S = 420.0
T_COLL_S = 300.0
T_REACT_S = 60.0
T_MANEUVER_S = 60.0
OVERTAKE_ORIENT_RAD = math.radians(67.5)
SIDE_SECTOR_RAD = math.radians(112.5)
CONE_RADIUS_M = HULL_CONTACT_M

R_COL_M = HULL_CONTACT_M
R_PREF_M = ONE_NAUTICAL_MILE_M
DELAY_MAX = 0.5
APPARENT_TURN_RAD = math.radians(30.0)
DETECTABLE_TURN_RAD = math.radians(2.0)
APPARENT_SLOWDOWN = 0.5
DETECTABLE_DV_MPS = 0.2
NON_APPARENT_MAX = 0.5
NON_APPARENT_THRESH = 0.3
STAND_ON_MAX = 0.5

OK, VIOLATED, UNDECIDED, NOT_APPLICABLE = "ok", "violated", "undecided", "n/a"
DUTY_RULES = ("R3", "R4", "R5", "R6")


def _wrap(a):
    """Wrap angles into [-pi, pi).

    Args:
        a (ndarray): Angles, rad

    Returns:
        ndarray: Wrapped angles, rad
    """
    return (np.asarray(a) + math.pi) % (2.0 * math.pi) - math.pi


def _mod2pi(a):
    """Wrap angles into [0, 2 pi).

    Args:
        a (ndarray): Angles, rad

    Returns:
        ndarray: Wrapped angles, rad
    """
    return np.asarray(a) % (2.0 * math.pi)


def collision_possible(ego, other, t_horizon, cone_m=CONE_RADIUS_M):
    """Check at each step whether a contact can happen soon.

    Args:
        ego (ndarray): Own track, one row per step (x, y, heading, speed)
        other (ndarray): Other track, same layout
        t_horizon (float): Look ahead time, s
        cone_m (float): Contact radius, m

    Returns:
        ndarray: True per step where contact is possible
    """
    px, py = other[:, 0] - ego[:, 0], other[:, 1] - ego[:, 1]
    rng = np.hypot(px, py)
    wx = ego[:, 3] * np.cos(ego[:, 2]) - other[:, 3] * np.cos(other[:, 2])
    wy = ego[:, 3] * np.sin(ego[:, 2]) - other[:, 3] * np.sin(other[:, 2])
    w = np.hypot(wx, wy)

    inside = rng <= cone_m
    with np.errstate(divide="ignore", invalid="ignore"):
        half = np.arcsin(np.clip(cone_m / rng, 0.0, 1.0))
        cos_ang = (wx * px + wy * py) / (w * rng)
        in_cone = np.arccos(np.clip(cos_ang, -1.0, 1.0)) <= half
        soon = (rng - cone_m) / w <= t_horizon
    return inside | ((w > 1e-9) & in_cone & soon)


def _bearing(ego, other):
    """Compute the bearing of the other vessel from own heading at each step.

    Args:
        ego (ndarray): Own track, one row per step (x, y, heading, speed)
        other (ndarray): Other track, same layout

    Returns:
        ndarray: Relative bearing per step, rad
    """
    return _wrap(
        np.arctan2(other[:, 1] - ego[:, 1], other[:, 0] - ego[:, 0]) - ego[:, 2]
    )


def in_front_sector(ego, other):
    """Check at each step that the other vessel is ahead.

    Args:
        ego (ndarray): Own track, one row per step (x, y, heading, speed)
        other (ndarray): Other track, same layout

    Returns:
        ndarray: True per step
    """
    return np.abs(_bearing(ego, other)) <= DELTA_HEAD_ON_RAD


def in_right_sector(ego, other):
    """Check at each step that the other vessel is on starboard.

    Args:
        ego (ndarray): Own track, one row per step (x, y, heading, speed)
        other (ndarray): Other track, same layout

    Returns:
        ndarray: True per step
    """
    b = _bearing(ego, other)
    return (b < -DELTA_HEAD_ON_RAD) & (b >= -SIDE_SECTOR_RAD)


def in_left_sector(ego, other):
    """Check at each step that the other vessel is on port.

    Args:
        ego (ndarray): Own track, one row per step (x, y, heading, speed)
        other (ndarray): Other track, same layout

    Returns:
        ndarray: True per step
    """
    b = _bearing(ego, other)
    return (b > DELTA_HEAD_ON_RAD) & (b <= SIDE_SECTOR_RAD)


def in_behind_sector(ego, other):
    """Check at each step that the other vessel is astern.

    Args:
        ego (ndarray): Own track, one row per step (x, y, heading, speed)
        other (ndarray): Other track, same layout

    Returns:
        ndarray: True per step
    """
    return np.abs(_bearing(ego, other)) > SIDE_SECTOR_RAD


def orientation_delta(ego, other, delta, offset):
    """Check at each step that the heading gap, plus an offset, is at least delta.

    Args:
        ego (ndarray): Own track, one row per step (x, y, heading, speed)
        other (ndarray): Other track, same layout
        delta (float): Min gap, rad
        offset (float): Offset added to the gap, rad

    Returns:
        ndarray: True per step
    """
    d = _mod2pi(other[:, 2] - ego[:, 2] + offset)
    return (d >= delta) & (d <= 2.0 * math.pi - delta)


def orientation_towards_right(ego, other, delta=DELTA_HEAD_ON_RAD):
    """Check at each step that the other vessel heads to the right.

    Args:
        ego (ndarray): Own track, one row per step (x, y, heading, speed)
        other (ndarray): Other track, same layout
        delta (float): Margin, rad

    Returns:
        ndarray: True per step
    """
    d = _wrap(other[:, 2] - ego[:, 2])
    return (d >= -math.pi + delta) & (d <= -delta)


def orientation_towards_left(ego, other, delta=DELTA_HEAD_ON_RAD):
    """Check at each step that the other vessel heads to the left.

    Args:
        ego (ndarray): Own track, one row per step (x, y, heading, speed)
        other (ndarray): Other track, same layout
        delta (float): Margin, rad

    Returns:
        ndarray: True per step
    """
    d = _wrap(other[:, 2] - ego[:, 2])
    return (d >= delta) & (d <= math.pi - delta)


def overtake(ego, other, t=T_CHECK_S, cone_m=CONE_RADIUS_M):
    """Check at each step that own vessel is overtaking.

    Args:
        ego (ndarray): Own track, one row per step (x, y, heading, speed)
        other (ndarray): Other track, same layout
        t (float): Look ahead time, s
        cone_m (float): Contact radius, m

    Returns:
        ndarray: True per step
    """
    return (
        collision_possible(ego, other, t, cone_m)
        & in_behind_sector(other, ego)
        & (ego[:, 3] > other[:, 3])
        & ~orientation_delta(ego, other, OVERTAKE_ORIENT_RAD, 0.0)
    )


def head_on(ego, other, t=T_CHECK_S, cone_m=CONE_RADIUS_M):
    """Check at each step that the vessels meet head on.

    Args:
        ego (ndarray): Own track, one row per step (x, y, heading, speed)
        other (ndarray): Other track, same layout
        t (float): Look ahead time, s
        cone_m (float): Contact radius, m

    Returns:
        ndarray: True per step
    """
    return (
        collision_possible(ego, other, t, cone_m)
        & in_front_sector(ego, other)
        & ~orientation_delta(ego, other, DELTA_HEAD_ON_RAD, math.pi)
    )


def crossing(ego, other, t=T_CHECK_S, cone_m=CONE_RADIUS_M):
    """Check at each step that own vessel must give way in a crossing.

    Args:
        ego (ndarray): Own track, one row per step (x, y, heading, speed)
        other (ndarray): Other track, same layout
        t (float): Look ahead time, s
        cone_m (float): Contact radius, m

    Returns:
        ndarray: True per step
    """
    return (
        collision_possible(ego, other, t, cone_m)
        & in_right_sector(ego, other)
        & orientation_towards_left(ego, other)
    )


def keep(ego, other, t=T_CHECK_S, cone_m=CONE_RADIUS_M):
    """Check at each step that own vessel must stand on.

    Args:
        ego (ndarray): Own track, one row per step (x, y, heading, speed)
        other (ndarray): Other track, same layout
        t (float): Look ahead time, s
        cone_m (float): Contact radius, m

    Returns:
        ndarray: True per step
    """
    return (
        collision_possible(ego, other, t, cone_m)
        & in_left_sector(ego, other)
        & orientation_towards_right(ego, other)
    ) | overtake(other, ego, t, cone_m)


def _rising_edges(s):
    """Find the steps where a flag turns on.

    Args:
        s (ndarray): Flag per step

    Returns:
        ndarray: Step indexes
    """
    prev = np.concatenate(([False], s[:-1]))
    return np.flatnonzero(s & ~prev)


def _give_way_rule(s, ego, dt, starboard):
    """Check that own vessel turned enough, in time, each time it had to give way.

    Args:
        s (ndarray): Give way flag per step
        ego (ndarray): Own track
        dt (float): Time step, s
        starboard (bool): Whether the turn must be to starboard

    Returns:
        str: Verdict
    """
    k_react = int(round(T_REACT_S / dt))
    k_man = int(round(T_MANEUVER_S / dt))
    n = len(s)
    heading = np.unwrap(ego[:, 2])
    verdicts = []
    for edge in _rising_edges(s):
        k = edge - 1

        hold = s[edge : k + k_react + 1]
        if len(hold) < k_react:
            if hold.all():
                verdicts.append(UNDECIDED)
            continue
        if not hold.all():
            continue

        turn = heading - heading[max(k, 0)]
        manoeuvre = (np.abs(turn) >= DELTA_LARGE_TURN_RAD) & np.concatenate(
            (s[:1], s[:-1])
        )
        if starboard:
            manoeuvre &= turn < 0.0
        end1 = k + k_react + k_man
        met1 = manoeuvre[max(k, 0) : end1 + 1].any()

        end2 = k + k_react + 2 * k_man
        met2 = (~s[k + k_react : end2 + 1]).any()

        if met1 and met2:
            verdicts.append(OK)
        elif (not met1 and end1 < n) or (not met2 and end2 < n):
            verdicts.append(VIOLATED)
        else:
            verdicts.append(UNDECIDED)
    return _combine(verdicts)


def _stand_on_licence(s, other, dt):
    """Mark the steps where the stand on vessel may act because the other did not.

    Args:
        s (ndarray): Stand on flag per step
        other (ndarray): Other track
        dt (float): Time step, s

    Returns:
        ndarray: True per licensed step
    """
    k_due = int(round((T_REACT_S + T_MANEUVER_S) / dt))
    heading = np.unwrap(other[:, 2])
    licensed = np.zeros(len(s), bool)
    for edge in _rising_edges(s):
        k = max(edge - 1, 0)
        due = k + k_due
        if due >= len(s):
            continue
        acted = (
            np.abs(heading[k : due + 1] - heading[k]) >= DELTA_LARGE_TURN_RAD
        ).any()
        if acted:
            continue
        end = due
        while end < len(s) and s[end]:
            end += 1
        licensed[due:end] = True
    return licensed


def _stand_on_rule(s, ego, licensed=None):
    """Check that own vessel kept its heading while it had to stand on.

    Args:
        s (ndarray): Stand on flag per step
        ego (ndarray): Own track
        licensed (ndarray): Licensed steps, or None

    Returns:
        str: Verdict
    """
    if not s.any():
        return NOT_APPLICABLE
    if licensed is not None:
        s = s & ~licensed
    heading = np.unwrap(ego[:, 2])
    edge = np.zeros(len(s), bool)
    edge[_rising_edges(s)] = True
    onset = np.maximum.accumulate(np.where(edge, np.arange(len(s)), 0))
    ts = np.maximum(onset - 1, 0)
    turned_next = np.abs(heading[1:] - heading[ts[:-1]]) >= DELTA_NO_TURN_RAD
    return VIOLATED if (s[:-1] & turned_next).any() else OK


def _combine(verdicts):
    """Merge verdicts, violated first, then undecided.

    Args:
        verdicts (list): Verdicts

    Returns:
        str: Merged verdict
    """
    if not verdicts:
        return NOT_APPLICABLE
    for v in (VIOLATED, UNDECIDED):
        if v in verdicts:
            return v
    return OK


def check_pair(ego, other, dt, cone_m=CONE_RADIUS_M):
    """Check every rule for own vessel against one other vessel.

    Args:
        ego (ndarray): Own track, one row per step (x, y, heading, speed)
        other (ndarray): Other track, same layout
        dt (float): Time step, s
        cone_m (float): Contact radius, m

    Returns:
        dict: Verdict per rule
    """
    kept = keep(ego, other, cone_m=cone_m)
    return {
        "R1": (
            VIOLATED if collision_possible(ego, other, T_COLL_S, cone_m).any() else OK
        ),
        "R2": OK if ((ego[:, 3] >= 0.0) & (ego[:, 3] <= V_MAX_MPS)).all() else VIOLATED,
        "R3": _give_way_rule(
            crossing(ego, other, cone_m=cone_m), ego, dt, starboard=True
        ),
        "R4": _give_way_rule(
            head_on(ego, other, cone_m=cone_m), ego, dt, starboard=True
        ),
        "R5": _give_way_rule(
            overtake(ego, other, cone_m=cone_m), ego, dt, starboard=False
        ),
        "R6": _stand_on_rule(kept, ego, _stand_on_licence(kept, other, dt)),
        "R6_raw": _stand_on_rule(kept, ego),
    }


def _segment_min(ego, other):
    """Find the closest distance between two tracks, between steps too.

    Args:
        ego (ndarray): Own track, one row per step (x, y, heading, speed)
        other (ndarray): Other track, same layout

    Returns:
        tuple: Min distance (m) and step index
    """
    r = ego[:, :2] - other[:, :2]
    if len(r) < 2:
        return float(np.hypot(*r[0])), 0
    r0, d = r[:-1], r[1:] - r[:-1]
    dd = np.einsum("ij,ij->i", d, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(
            dd == 0.0, 0.0, np.clip(-np.einsum("ij,ij->i", r0, d) / dd, 0.0, 1.0)
        )
    dist = np.hypot(*(r0 + t[:, None] * d).T)
    i = int(np.argmin(dist))
    return float(dist[i]), i + int(t[i] > 0.5)


def _safety(r_cpa):
    """Score the closest distance between contact and preferred range.

    Args:
        r_cpa (float): Closest distance, m

    Returns:
        float: Score between 0 and 1
    """
    return float(np.clip((r_cpa - R_COL_M) / (R_PREF_M - R_COL_M), 0.0, 1.0))


def _give_way_score(ego, other, k0, r_cpa):
    """Score how well own vessel gave way.

    Args:
        ego (ndarray): Own track, one row per step (x, y, heading, speed)
        other (ndarray): Other track, same layout
        k0 (int): Step where the duty starts
        r_cpa (float): Closest distance, m

    Returns:
        float: Score
    """
    heading = np.unwrap(ego[:, 2])
    dev = np.abs(heading[k0:] - heading[k0])
    rng = np.hypot(*(ego[k0:, :2] - other[k0:, :2]).T)

    started = np.flatnonzero(dev >= DETECTABLE_TURN_RAD)
    r_detect = float(rng[0])
    r_man = min(float(rng[started[0]]) if len(started) else r_cpa, r_detect)
    score = _safety(r_cpa) * (
        1.0 - DELAY_MAX * (r_detect - r_man) / max(r_detect, 1e-9)
    )

    turn = float(dev.max())
    p_turn = (
        0.0
        if turn > APPARENT_TURN_RAD
        else NON_APPARENT_MAX * ((APPARENT_TURN_RAD - turn) / APPARENT_TURN_RAD)
    )
    v0 = max(float(ego[k0, 3]), 1e-9)
    slow = (v0 - float(ego[k0:, 3].min())) / v0
    p_speed = (
        0.0
        if slow >= APPARENT_SLOWDOWN
        else NON_APPARENT_MAX * ((APPARENT_SLOWDOWN - slow) / APPARENT_SLOWDOWN)
    )
    if p_turn >= NON_APPARENT_THRESH and p_speed >= NON_APPARENT_THRESH:
        score *= (1.0 - p_speed) * (1.0 - p_turn)
    return score


def _stand_on_score(ego, k0, k_cpa, r_cpa):
    """Score how well own vessel stood on.

    Args:
        ego (ndarray): Own track
        k0 (int): Step where the duty starts
        k_cpa (int): Step of closest approach
        r_cpa (float): Closest distance, m

    Returns:
        float: Score
    """
    score = _safety(r_cpa)
    heading = np.unwrap(ego[:, 2])
    dev = np.abs(heading[k0:] - heading[k0])
    started = np.flatnonzero(dev >= DETECTABLE_TURN_RAD)
    if len(started) and k0 + started[0] > k_cpa:
        return score

    turn = float(dev[: max(k_cpa - k0, 0) + 1].max())
    if turn > APPARENT_TURN_RAD:
        score -= STAND_ON_MAX
    elif turn >= DETECTABLE_TURN_RAD:
        score -= (
            STAND_ON_MAX
            * (turn - DETECTABLE_TURN_RAD)
            / (APPARENT_TURN_RAD - DETECTABLE_TURN_RAD)
        )

    v0 = max(float(ego[k0, 3]), 1e-9)
    v_hi, v_lo = float(ego[k0:, 3].max()), float(ego[k0:, 3].min())
    if max(v_hi - v0, v0 - v_lo) >= DETECTABLE_DV_MPS:
        score *= (v0 / v_hi) ** 2
        score -= STAND_ON_MAX * (v0 - v_lo) / v0
    return max(score, 0.0)


def woerner_score(ego, other):
    """Score the encounter with the Woerner method.

    Args:
        ego (ndarray): Own track, one row per step (x, y, heading, speed)
        other (ndarray): Other track, same layout

    Returns:
        float: Score, or None if own vessel had no duty
    """
    give = crossing(ego, other) | head_on(ego, other) | overtake(ego, other)
    stand = keep(ego, other)
    r_cpa, k_cpa = _segment_min(ego, other)
    if give.any():
        return _give_way_score(ego, other, int(np.argmax(give)), r_cpa)
    if stand.any():
        return _stand_on_score(ego, int(np.argmax(stand)), k_cpa, r_cpa)
    return None


def check_episode(trace, dt, own=0, cone_m=CONE_RADIUS_M):
    """Check every pair of an episode and score own vessel.

    Args:
        trace (ndarray): Tracks, shape (steps, vessels, 4)
        dt (float): Time step, s
        own (int): Index of own vessel
        cone_m (float): Contact radius, m

    Returns:
        dict: Pair verdicts, own violations, compliance flags and scores
    """
    trace = np.asarray(trace, float)
    n = trace.shape[1]
    pairs = {
        (i, j): check_pair(trace[:, i], trace[:, j], dt, cone_m)
        for i in range(n)
        for j in range(n)
        if i != j
    }

    def violations(i, rules=DUTY_RULES, verdict=VIOLATED):
        """List the rules a vessel broke with each other vessel.

        Args:
            i (int): Vessel index
            rules (tuple): Rules to check
            verdict (str): Verdict to look for

        Returns:
            list: Entries like R3:2
        """
        return sorted(
            f"{r}:{j}"
            for (a, j), v in pairs.items()
            if a == i
            for r in rules
            if v[r] == verdict
        )

    raw_rules = tuple("R6_raw" if r == "R6" else r for r in DUTY_RULES)

    scores = {
        j: woerner_score(trace[:, own], trace[:, j]) for j in range(n) if j != own
    }
    held = [s for s in scores.values() if s is not None]
    return dict(
        pairs=pairs,
        own_violations=violations(own),
        own_compliant=not violations(own),
        own_violations_raw=violations(own, raw_rules),
        own_compliant_raw=not violations(own, raw_rules),
        vessels_compliant=[not violations(i) for i in range(n)],
        own_undecided=violations(own, verdict=UNDECIDED),
        vessels_undecided=[bool(violations(i, verdict=UNDECIDED)) for i in range(n)],
        undecided=sum(v == UNDECIDED for p in pairs.values() for v in p.values()),
        woerner=scores,
        woerner_min=min(held) if held else None,
    )

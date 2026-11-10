"""Provably safe RL under a COLREGS statechart, the hybrid state-of-the-art arm (safety layer).

Krasowski & Althoff (2024), "Provable traffic rule compliance in safe reinforcement learning on the
open sea", IEEE Transactions on Intelligent Vehicles, doi 10.1109/TIV.2024.3400597,
arXiv 2402.08502.
"""

from __future__ import annotations

import math
import os
from enum import Enum

import numpy as np

from ...simulator.experiment_constants import (
    SIM_DT_S,
    SPEED_MAX_MPS,
    SPEED_MIN_MPS,
    SURGE_ACCEL_MAX_MPS2,
    SURGE_DECEL_MAX_MPS2,
    TS_SPEED_RANGE_MPS,
)
from ...certificate.escape_angle import R_MIN_M

TWO_PI = 2.0 * math.pi
DEG = math.pi / 180.0

LENGTH_M = 150.0
WIDTH_M = 25.0
HULL_RADIUS_M = 0.5 * math.hypot(LENGTH_M, WIDTH_M)
K_V = SPEED_MAX_MPS / 9.5

DELTA_AHEAD = 45.0 * DEG
V_PM_MAX_MPS = TS_SPEED_RANGE_MPS[1] + 1.0
A_PM_MAX_MPS2 = 0.045
D_RESOLVED_M = 2.0 * LENGTH_M
D_OBS_SAFETY_M = 2.0 * LENGTH_M
DELTA_HEAD_ON = 5.0 * DEG
DELTA_NO_TURN = 10.0 * DEG
DELTA_LARGE_TURN = 20.0 * DEG
T_PRED_S = 180.0
T_MAX_M_S = 200.0
DELTA_STERN = 20.0 * DEG
A_STERN_MPS2 = 0.2 * 0.24
D_MIN_AHEAD_M = 3.0 * LENGTH_M
T_CHECK_S = 420.0
T_REACT_S = 60.0
T_M_S = 40.0
V_EPS_MPS = 1.0
N_V_SAMPLES = 5
CONE_RADIUS_M = 3.0 * LENGTH_M
SIDE_SECTOR = 112.5 * DEG
OVERTAKE_ORIENT = 67.5 * DEG

LAMBDA_1 = 4.0
LAMBDA_2 = 0.04
DV_OMEGA = 0.3
V_DESIRED_MPS = 6.0 * K_V
AHEAD_TARGET_M = 10.0 * LENGTH_M
BASE_TARGET_M = 3.0 * LENGTH_M

A_SET = (-0.048, -0.032, -0.016, 0.0, 0.016, 0.032, 0.048)
W_SET = (-0.018, -0.012, -0.006, 0.0, 0.006, 0.012, 0.018)
N_REGULAR = len(A_SET) * len(W_SET)
EMERGENCY = N_REGULAR
N_ACTIONS = N_REGULAR + 1
KEEP = A_SET.index(0.0) * len(W_SET) + W_SET.index(0.0)
A_TURN_RIGHT = tuple(
    A_SET.index(0.0) * len(W_SET) + j
    for j, w in enumerate(W_SET)
    if w < 0 and -w * T_M_S >= DELTA_LARGE_TURN
)
A_TURN_LEFT = tuple(
    A_SET.index(0.0) * len(W_SET) + j
    for j, w in enumerate(W_SET)
    if w > 0 and w * T_M_S >= DELTA_LARGE_TURN
)
A_ACC = tuple(
    i * len(W_SET) + W_SET.index(0.0) for i, a in enumerate(A_SET) if a >= 0.0
)
STEPS_PER_SEGMENT = int(round(T_M_S / SIM_DT_S))
ROLL_DT_S = 2.0
STERN_HOLD_S = 20.0


def action_values(i):
    """Turn an action index into acceleration and turn rate.

    Args:
        i (int): Action index

    Returns:
        tuple: Acceleration (m/s2) and turn rate (rad/s)
    """
    return A_SET[i // len(W_SET)], W_SET[i % len(W_SET)]


class Rho(Enum):
    NO_CONFLICT = 0
    STAND_ON = 1
    HEAD_ON = 2
    CROSSING = 3
    OVERTAKE = 4
    EMERGENCY = 5


GIVE_WAY = (Rho.HEAD_ON, Rho.CROSSING, Rho.OVERTAKE)
PRIORITY = {
    Rho.EMERGENCY: 3,
    Rho.HEAD_ON: 2,
    Rho.CROSSING: 2,
    Rho.OVERTAKE: 2,
    Rho.STAND_ON: 1,
    Rho.NO_CONFLICT: 0,
}


class Mode(Enum):
    AHEAD = 0
    STERN = 1
    BASE = 2


def _bearing_cw(l, m):
    """Compute the clockwise bearing of m seen from l.

    Args:
        l (tuple): Own state (x, y, v, heading)
        m (tuple): Other state (x, y, v, heading)

    Returns:
        float: Bearing in [0, 2 pi), rad
    """
    return (l[3] - math.atan2(m[1] - l[1], m[0] - l[0])) % TWO_PI


def in_sector(l, m, lo, hi):
    """Check that m lies in a bearing sector of l.

    Args:
        l (tuple): Own state (x, y, v, heading)
        m (tuple): Other state (x, y, v, heading)
        lo (float): Sector start, rad
        hi (float): Sector end, rad

    Returns:
        bool: True if inside
    """
    b, lo, hi = _bearing_cw(l, m), lo % TWO_PI, hi % TWO_PI
    return lo <= b <= hi if lo <= hi else (b >= lo or b <= hi)


def in_front(l, m):
    """Check that m is ahead of l.

    Args:
        l (tuple): Own state (x, y, v, heading)
        m (tuple): Other state (x, y, v, heading)

    Returns:
        bool: True if ahead
    """
    return in_sector(l, m, -DELTA_HEAD_ON, DELTA_HEAD_ON)


def in_left(l, m):
    """Check that m is on the port side of l.

    Args:
        l (tuple): Own state (x, y, v, heading)
        m (tuple): Other state (x, y, v, heading)

    Returns:
        bool: True if on port
    """
    return in_sector(l, m, -SIDE_SECTOR, -DELTA_HEAD_ON)


def in_right(l, m):
    """Check that m is on the starboard side of l.

    Args:
        l (tuple): Own state (x, y, v, heading)
        m (tuple): Other state (x, y, v, heading)

    Returns:
        bool: True if on starboard
    """
    return in_sector(l, m, DELTA_HEAD_ON, SIDE_SECTOR)


def in_behind(l, m):
    """Check that m is astern of l.

    Args:
        l (tuple): Own state (x, y, v, heading)
        m (tuple): Other state (x, y, v, heading)

    Returns:
        bool: True if astern
    """
    return in_sector(l, m, SIDE_SECTOR, TWO_PI - SIDE_SECTOR)


def orientation_delta(l, m, delta, offset):
    """Check that the heading gap, plus an offset, is at least delta.

    Args:
        l (tuple): Own state (x, y, v, heading)
        m (tuple): Other state (x, y, v, heading)
        delta (float): Min gap, rad
        offset (float): Offset added to the gap, rad

    Returns:
        bool: True if the gap is large enough
    """
    return delta <= (m[3] - l[3] + offset) % TWO_PI <= TWO_PI - delta


def towards_right(l, m, delta):
    """Check that m heads to the right of l.

    Args:
        l (tuple): Own state (x, y, v, heading)
        m (tuple): Other state (x, y, v, heading)
        delta (float): Margin, rad

    Returns:
        bool: True if heading right
    """
    return math.pi + delta <= (m[3] - l[3]) % TWO_PI <= TWO_PI - delta


def towards_left(l, m, delta):
    """Check that m heads to the left of l.

    Args:
        l (tuple): Own state (x, y, v, heading)
        m (tuple): Other state (x, y, v, heading)
        delta (float): Margin, rad

    Returns:
        bool: True if heading left
    """
    return delta <= (m[3] - l[3]) % TWO_PI <= math.pi - delta


def collision_possible(l, m, horizon=T_CHECK_S):
    """Check whether a close pass can happen for some own speed near the current one.

    Args:
        l (tuple): Own state (x, y, v, heading)
        m (tuple): Other state (x, y, v, heading)
        horizon (float): Look ahead time, s

    Returns:
        bool: True if a close pass is possible
    """
    px, py = m[0] - l[0], m[1] - l[1]
    mvx, mvy = m[2] * math.cos(m[3]), m[2] * math.sin(m[3])
    c, s = math.cos(l[3]), math.sin(l[3])
    for lam in np.linspace(max(l[2] - V_EPS_MPS, 0.0), l[2] + V_EPS_MPS, N_V_SAMPLES):
        vx, vy = mvx - lam * c, mvy - lam * s
        vv = vx * vx + vy * vy
        t = 0.0 if vv < 1e-12 else min(max(-(px * vx + py * vy) / vv, 0.0), horizon)
        if math.hypot(px + vx * t, py + vy * t) <= CONE_RADIUS_M:
            return True
    return False


def overtake(l, m):
    """Check that l is overtaking m.

    Args:
        l (tuple): Own state (x, y, v, heading)
        m (tuple): Other state (x, y, v, heading)

    Returns:
        bool: True if overtaking
    """
    return (
        collision_possible(l, m)
        and in_behind(m, l)
        and l[2] > m[2]
        and not orientation_delta(l, m, OVERTAKE_ORIENT, 0.0)
    )


def head_on(l, m):
    """Check that l and m meet head on.

    Args:
        l (tuple): Own state (x, y, v, heading)
        m (tuple): Other state (x, y, v, heading)

    Returns:
        bool: True if head on
    """
    return (
        collision_possible(l, m)
        and in_front(l, m)
        and not orientation_delta(l, m, DELTA_HEAD_ON, math.pi)
    )


def crossing(l, m):
    """Check that m crosses from the right, so l gives way.

    Args:
        l (tuple): Own state (x, y, v, heading)
        m (tuple): Other state (x, y, v, heading)

    Returns:
        bool: True if crossing
    """
    return (
        collision_possible(l, m)
        and in_right(l, m)
        and towards_left(l, m, DELTA_HEAD_ON)
    )


def keep(l, m):
    """Check that l must keep course and speed.

    Args:
        l (tuple): Own state (x, y, v, heading)
        m (tuple): Other state (x, y, v, heading)

    Returns:
        bool: True if l stands on
    """
    return collision_possible(l, m) and (
        (in_left(l, m) and towards_right(l, m, DELTA_HEAD_ON)) or overtake(m, l)
    )


_GIVE_WAY_PRED = {Rho.HEAD_ON: head_on, Rho.CROSSING: crossing, Rho.OVERTAKE: overtake}


def _ahead(s, t):
    """Move a state forward in a straight line.

    Args:
        s (tuple): State (x, y, v, heading)
        t (float): Time, s

    Returns:
        tuple: Future state
    """
    return (
        s[0] + t * s[2] * math.cos(s[3]),
        s[1] + t * s[2] * math.sin(s[3]),
        s[2],
        s[3],
    )


def persistent(pred, ego, obs):
    """Check that a rule holds over the whole reaction time.

    Args:
        pred (callable): Rule test pred(ego, obs)
        ego (tuple): Own state (x, y, v, heading)
        obs (tuple): Obstacle state (x, y, v, heading)

    Returns:
        bool: True if it holds at every step
    """
    steps = int(T_REACT_S / SIM_DT_S)
    return all(
        pred(_ahead(ego, k * SIM_DT_S), _ahead(obs, k * SIM_DT_S))
        for k in range(steps + 1)
    )


def rollout(s, controls, hold_s, dt=ROLL_DT_S):
    """Simulate own motion under a list of held controls.

    Args:
        s (tuple): Start state (x, y, v, heading)
        controls (list): Pairs (acceleration, turn rate)
        hold_s (float): Time each control is held, s
        dt (float): Integration step, s

    Returns:
        tuple: Times, x values, y values and final state
    """

    x, y, v, th = s
    n = int(round(hold_s / dt))
    ts, xs, ys = [0.0], [x], [y]
    t = 0.0
    for a, w in controls:
        for _ in range(n):
            w_lim = abs(v) / R_MIN_M
            w_c = min(max(w, -w_lim), w_lim)
            x += dt * v * math.cos(th)
            y += dt * v * math.sin(th)
            v = min(max(v + dt * a, SPEED_MIN_MPS), SPEED_MAX_MPS)
            th += dt * w_c
            t += dt
            ts.append(t)
            xs.append(x)
            ys.append(y)
    return np.array(ts), np.array(xs), np.array(ys), (x, y, v, th)


def _pm_hits(obs, t, ex, ey, extra):
    """Check whether own path enters the reachable set of the obstacle.

    Args:
        obs (tuple): Obstacle state (x, y, v, heading)
        t (ndarray): Times, s
        ex (ndarray): Own x values, m
        ey (ndarray): Own y values, m
        extra (float): Extra margin, m

    Returns:
        bool: True if a hit is possible
    """
    cx = obs[0] + obs[2] * math.cos(obs[3]) * t
    cy = obs[1] + obs[2] * math.sin(obs[3]) * t
    r1 = 0.5 * A_PM_MAX_MPS2 * t * t + extra
    r2 = V_PM_MAX_MPS * t + extra
    d1 = np.hypot(ex - cx, ey - cy)
    d2 = np.hypot(ex - obs[0], ey - obs[1])
    return bool(np.any((d1 <= r1) & (d2 <= r2)))


def is_emergency(ego, obs, controls=None, hold_s=T_PRED_S):
    """Check whether the obstacle can reach own path in the prediction time.

    Args:
        ego (tuple): Own state (x, y, v, heading)
        obs (tuple): Obstacle state (x, y, v, heading)
        controls (list): Own controls, or None for a straight line
        hold_s (float): Time each control is held, s

    Returns:
        bool: True if this is an emergency
    """
    if controls is None:
        t = np.arange(0.0, T_PRED_S + ROLL_DT_S / 2, ROLL_DT_S)
        ex = ego[0] + ego[2] * math.cos(ego[3]) * t
        ey = ego[1] + ego[2] * math.sin(ego[3]) * t
    else:
        t, ex, ey, _ = rollout(ego, controls, hold_s)
        keep_t = t <= T_PRED_S
        t, ex, ey = t[keep_t], ex[keep_t], ey[keep_t]
    extra = 2.0 * HULL_RADIUS_M + 0.5 * ROLL_DT_S * (SPEED_MAX_MPS + V_PM_MAX_MPS)
    return _pm_hits(obs, t, ex, ey, extra)


def range_opening(ego, obs):
    """Check that the distance between the two vessels is growing.

    Args:
        ego (tuple): Own state (x, y, v, heading)
        obs (tuple): Obstacle state (x, y, v, heading)

    Returns:
        bool: True if opening
    """
    dx, dy = obs[0] - ego[0], obs[1] - ego[1]
    return (
        dx * (obs[2] * math.cos(obs[3]) - ego[2] * math.cos(ego[3]))
        + dy * (obs[2] * math.sin(obs[3]) - ego[2] * math.sin(ego[3]))
    ) > 0


def is_resolved(ego, obs):
    """Check that the encounter is over.

    Args:
        ego (tuple): Own state (x, y, v, heading)
        obs (tuple): Obstacle state (x, y, v, heading)

    Returns:
        bool: True if opening and far enough
    """
    return (
        range_opening(ego, obs)
        and math.hypot(obs[0] - ego[0], obs[1] - ego[1]) >= D_RESOLVED_M
    )


def tracking_control(ego, target):
    """Compute the command that steers toward a point.

    Args:
        ego (tuple): Own state (x, y, v, heading)
        target (tuple): Point (x, y)

    Returns:
        tuple: Acceleration (m/s2) and turn rate (rad/s)
    """
    x, y, v, th = ego
    dx, dy = target[0] - x, target[1] - y
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return 0.0, 0.0
    wx, wy = dx / dist, dy / dist
    reach = V_DESIRED_MPS * SIM_DT_S
    pdx, pdy = (dx, dy) if dist <= reach else (reach * wx, reach * wy)
    cos_e = math.cos(th) * wx + math.sin(th) * wy
    sin_e = -math.sin(th) * wx + math.cos(th) * wy
    w_lim = abs(v) / R_MIN_M
    v_omega = 1.0 - cos_e * cos_e
    if cos_e <= 1e-6:
        w = w_lim if sin_e >= 0 else -w_lim
    else:
        w = LAMBDA_1 * v_omega / (2.0 * sin_e * cos_e) if abs(sin_e) > 1e-9 else 0.0
    w_err = abs(math.atan2(sin_e, cos_e)) / SIM_DT_S
    w = min(max(w, -w_lim, -w_err), w_lim, w_err)
    along = pdx * v * math.cos(th) + pdy * v * math.sin(th)
    a = 0.0
    if v_omega <= DV_OMEGA and along > 0:
        a = LAMBDA_2 * 0.5 * (pdx * pdx + pdy * pdy) / along
    return min(max(a, SURGE_DECEL_MAX_MPS2), SURGE_ACCEL_MAX_MPS2), w


def _pass_side(ego, obs):
    """Find on which side the obstacle will pass.

    Args:
        ego (tuple): Own state (x, y, v, heading)
        obs (tuple): Obstacle state (x, y, v, heading)

    Returns:
        int: 1 port, -1 starboard, 0 dead ahead
    """
    px, py = obs[0] - ego[0], obs[1] - ego[1]
    vx = obs[2] * math.cos(obs[3]) - ego[2] * math.cos(ego[3])
    vy = obs[2] * math.sin(obs[3]) - ego[2] * math.sin(ego[3])
    vv = vx * vx + vy * vy
    t = 0.0 if vv < 1e-12 else max(-(px * vx + py * vy) / vv, 0.0)
    cx, cy = px + vx * t, py + vy * t
    lateral = -math.sin(ego[3]) * cx + math.cos(ego[3]) * cy
    return 0 if abs(lateral) < 1e-6 else (1 if lateral > 0 else -1)


def _stern_intercept(ego, obs, iterations=5):
    """Find a point behind the obstacle that own vessel can reach.

    Args:
        ego (tuple): Own state (x, y, v, heading)
        obs (tuple): Obstacle state (x, y, v, heading)
        iterations (int): Number of refinement loops

    Returns:
        tuple: Point (x, y)
    """
    c, s = math.cos(obs[3]), math.sin(obs[3])
    lateral = -s * (ego[0] - obs[0]) + c * (ego[1] - obs[1])
    side = 1.0 if lateral >= 0 else -1.0
    back = (
        obs[0] - BASE_TARGET_M * (c + side * s),
        obs[1] - BASE_TARGET_M * (s - side * c),
    )
    t = 0.0
    for _ in range(iterations):
        p = (back[0] + t * obs[2] * c, back[1] + t * obs[2] * s)
        t = math.hypot(p[0] - ego[0], p[1] - ego[1]) / max(ego[2], SPEED_MIN_MPS)
        if t > T_PRED_S:
            return back
    return p


class Emergency:

    def __init__(self, ego, obs):
        """Pick the emergency mode: ahead, stern or base.

        Args:
            ego (tuple): Own state (x, y, v, heading)
            obs (tuple): Obstacle state (x, y, v, heading)
        """
        self.start = (ego[0], ego[1])
        self.heading0 = ego[3]
        ahead = not orientation_delta(ego, obs, DELTA_AHEAD, math.pi) and in_sector(
            ego, obs, -DELTA_AHEAD, DELTA_AHEAD
        )
        n_acc = int(T_REACT_S / STERN_HOLD_S)
        u_acc = [(A_STERN_MPS2, 0.0)] * n_acc + [(0.0, 0.0)] * (
            int(T_PRED_S / STERN_HOLD_S) - n_acc
        )
        stern = in_sector(
            ego, obs, math.pi / 2 + DELTA_STERN, 3 * math.pi / 2 - DELTA_STERN
        ) and not is_emergency(ego, obs, u_acc, STERN_HOLD_S)
        self.mode = Mode.AHEAD if ahead else (Mode.STERN if stern else Mode.BASE)
        self.turn = -1.0 if _pass_side(ego, obs) >= 0 else 1.0
        self.t = 0.0

    def control(self, ego, obs):
        """Compute the emergency command for the current mode.

        Args:
            ego (tuple): Own state (x, y, v, heading)
            obs (tuple): Obstacle state (x, y, v, heading)

        Returns:
            tuple: Acceleration (m/s2) and turn rate (rad/s)
        """
        if (
            self.mode is Mode.AHEAD
            and math.hypot(ego[0] - self.start[0], ego[1] - self.start[1])
            > D_MIN_AHEAD_M
        ):
            self.mode = Mode.BASE
        if self.mode is Mode.STERN and self.t >= T_REACT_S:
            self.mode = Mode.BASE
        if self.mode is Mode.STERN:
            self.t += SIM_DT_S
            return min(A_STERN_MPS2, SURGE_ACCEL_MAX_MPS2), 0.0
        if self.mode is Mode.AHEAD:
            h = self.heading0 + self.turn * math.pi / 2
            return tracking_control(
                ego,
                (
                    ego[0] + AHEAD_TARGET_M * math.cos(h),
                    ego[1] + AHEAD_TARGET_M * math.sin(h),
                ),
            )
        return tracking_control(ego, _stern_intercept(ego, obs))


def maneuver_verified(seq, ego, obs):
    """Check that a sequence of actions keeps clear of the obstacle.

    Args:
        seq (tuple): Action indexes
        ego (tuple): Own state (x, y, v, heading)
        obs (tuple): Obstacle state (x, y, v, heading)

    Returns:
        bool: True if the manoeuvre is safe
    """
    t, ex, ey, end = rollout(ego, [action_values(i) for i in seq], T_M_S)
    ox = obs[0] + obs[2] * math.cos(obs[3]) * t
    oy = obs[1] + obs[2] * math.sin(obs[3]) * t
    r_obs = 0.5 * math.hypot(LENGTH_M + D_OBS_SAFETY_M, WIDTH_M + D_OBS_SAFETY_M)
    gap = HULL_RADIUS_M + r_obs + 0.5 * ROLL_DT_S * (SPEED_MAX_MPS + obs[2])
    if np.any(np.hypot(ex - ox, ey - oy) <= gap):
        return False
    return not collision_possible(end, _ahead(obs, t[-1]))


def build_tree(ac, ego, obs):
    """Grow action sequences from one first action and keep the safe ones.

    Args:
        ac (int): First action index
        ego (tuple): Own state (x, y, v, heading)
        obs (tuple): Obstacle state (x, y, v, heading)

    Returns:
        list: Safe action sequences
    """
    if maneuver_verified((ac,), ego, obs):
        return [(ac,)]
    level, t_end = [(ac,)], T_M_S
    while t_end < T_MAX_M_S:
        t_end += T_M_S
        nxt = []
        for seq in level:
            if seq[-1] == ac:
                nxt.append(seq + (ac,))
                nxt.extend(seq + (a,) for a in A_ACC)
            else:
                nxt.append(seq + (seq[-1],))
        level = list(dict.fromkeys(nxt))
        ok = [s for s in level if maneuver_verified(s, ego, obs)]
        if ok:
            return ok
    return []


def safe_sequences(rho, ego, obs):
    """List the safe manoeuvres for a COLREGs role.

    Args:
        rho (Rho): Current role
        ego (tuple): Own state (x, y, v, heading)
        obs (tuple): Obstacle state (x, y, v, heading)

    Returns:
        list: Safe action sequences
    """
    if rho is Rho.OVERTAKE:
        cands = (
            A_TURN_LEFT if math.remainder(obs[3] - ego[3], TWO_PI) < 0 else A_TURN_RIGHT
        )
    else:
        cands = A_TURN_RIGHT
    out = []
    for ac in cands:
        out.extend(build_tree(ac, ego, obs))
    return out


class _Pair:

    def __init__(self):
        """Start the pair with no conflict."""
        self.rho = Rho.NO_CONFLICT
        self.plan = None
        self.depth = 0
        self.hold = 0
        self.held = None
        self.em = None

    def _enter(self, rho):
        """Switch to a new role and clear the plan.

        Args:
            rho (Rho): New role
        """
        self.rho, self.plan, self.depth, self.hold, self.held, self.em = (
            rho,
            None,
            0,
            0,
            None,
            None,
        )

    def transition(self, ego, obs):
        """Update the role of the pair from the current states.

        Args:
            ego (tuple): Own state (x, y, v, heading)
            obs (tuple): Obstacle state (x, y, v, heading)
        """
        if self.rho is Rho.EMERGENCY:
            if not is_resolved(ego, obs):
                return
            self._enter(Rho.NO_CONFLICT)
        if not range_opening(ego, obs) and is_emergency(ego, obs):
            self._enter(Rho.EMERGENCY)
            self.em = Emergency(ego, obs)
            return
        if self.rho in GIVE_WAY:
            if self.hold > 0 or collision_possible(ego, obs):
                return
            self._enter(Rho.NO_CONFLICT)
        if self.rho is Rho.STAND_ON:
            if keep(ego, obs):
                return
            self._enter(Rho.NO_CONFLICT)
        if not collision_possible(ego, obs):
            return
        if keep(ego, obs):
            self._enter(Rho.STAND_ON)
            return
        for rho, pred in _GIVE_WAY_PRED.items():
            if persistent(pred, ego, obs):
                self._enter(rho)
                return

    def allowed(self, ego, obs):
        """List the actions allowed by the current role and plan.

        Args:
            ego (tuple): Own state (x, y, v, heading)
            obs (tuple): Obstacle state (x, y, v, heading)

        Returns:
            tuple: Allowed action indexes
        """
        if self.rho is Rho.NO_CONFLICT:
            return range(N_REGULAR)
        if self.rho is Rho.EMERGENCY:
            return (EMERGENCY,)
        if self.rho is Rho.STAND_ON:
            return (KEEP,)
        if self.hold > 0:
            return (self.held,)
        if self.plan is None or (self.plan and self.depth >= len(self.plan[0])):
            self.plan, self.depth = safe_sequences(self.rho, ego, obs), 0
        if not self.plan:
            return (KEEP,)
        return tuple(dict.fromkeys(s[self.depth] for s in self.plan))

    def commit(self, action):
        """Keep only the plans that match the chosen action.

        Args:
            action (int): Chosen action index
        """
        if self.rho not in GIVE_WAY:
            return
        if self.hold > 0:
            self.hold -= 1
            if self.held != action:
                self.plan, self.hold = None, 0
            return
        if not self.plan:
            return
        kept = [s for s in self.plan if s[self.depth] == action]
        if not kept:
            self.plan = None
            return
        self.plan, self.depth = kept, self.depth + 1
        self.held, self.hold = action, STEPS_PER_SEGMENT - 1


class Shield:

    def __init__(self):
        """Start with no tracked pairs."""
        self.pairs = {}

    def mask(self, ego, others):
        """Build the mask of allowed actions from the most urgent pair.

        Args:
            ego (tuple): Own state (x, y, v, heading)
            others (dict): Obstacle states by identifier

        Returns:
            ndarray: Boolean mask over actions
        """
        for tid in list(self.pairs):
            if tid not in others:
                del self.pairs[tid]
        per = []
        for tid, obs in others.items():
            p = self.pairs.setdefault(tid, _Pair())
            p.transition(ego, obs)
            m = np.zeros(N_ACTIONS, bool)
            m[list(p.allowed(ego, obs))] = True
            dist = math.hypot(obs[0] - ego[0], obs[1] - ego[1])
            per.append((PRIORITY[p.rho], -dist, m))
        if not per:
            m = np.zeros(N_ACTIONS, bool)
            m[:N_REGULAR] = True
            return m
        both = np.logical_and.reduce([m for _, _, m in per])
        if both.any():
            return both
        return max(per, key=lambda e: (e[0], e[1]))[2]

    def control(self, action, ego, others):
        """Turn the chosen action into a command.

        Args:
            action (int): Chosen action index
            ego (tuple): Own state (x, y, v, heading)
            others (dict): Obstacle states by identifier

        Returns:
            tuple: Acceleration (m/s2) and turn rate (rad/s)
        """
        for p in self.pairs.values():
            p.commit(action)
        if action != EMERGENCY:
            return action_values(action)
        em = [
            (math.hypot(others[t][0] - ego[0], others[t][1] - ego[1]), t)
            for t, p in self.pairs.items()
            if p.rho is Rho.EMERGENCY and t in others
        ]
        if not em:
            return action_values(KEEP)
        _, tid = min(em)
        return self.pairs[tid].em.control(ego, others[tid])

    def emergency_active(self):
        """Check whether any pair is in emergency.

        Returns:
            bool: True if one pair is in emergency
        """
        return any(p.rho is Rho.EMERGENCY for p in self.pairs.values())


D_SENSE_M = 8000.0
D_HULL_M = 2000.0
K_MAX = 170
D_GOAL_M = 4500.0 * K_V
N_OBS = 27
_SECTORS = (in_front, in_left, in_right, in_behind)
_NO_VESSEL = (
    1.0,
    0.0,
    0.0,
)


class Terminal(Enum):
    TIME = 0
    AREA = 1
    STOPPED = 2
    COLLISION = 3
    GOAL = 4


def path_frame(p, start, goal):
    """Express a point along and across the start to goal line.

    Args:
        p (tuple): Point (x, y)
        start (tuple): Start point
        goal (tuple): Goal point

    Returns:
        tuple: Distance along and across the line, m
    """
    ux, uy = goal[0] - start[0], goal[1] - start[1]
    n = math.hypot(ux, uy) or 1.0
    ux, uy = ux / n, uy / n
    dx, dy = p[0] - start[0], p[1] - start[1]
    return dx * ux + dy * uy, -dx * uy + dy * ux


def observe(ego, control, start, goal, k, others, prev_range, terminal=None):
    """Build the KRAS observation vector.

    Args:
        ego (tuple): Own state (x, y, v, heading)
        control (tuple): Last command
        start (tuple): Start point
        goal (tuple): Goal point
        k (int): Current step
        others (dict): Obstacle states by identifier
        prev_range (dict): Ranges at the last step
        terminal (object): Terminal flag, or None

    Returns:
        ndarray: Observation vector
    """
    d_goal = math.hypot(goal[0] - ego[0], goal[1] - ego[1])
    d_long, d_lat = path_frame(ego, start, goal)
    beta_goal = math.remainder(
        math.atan2(goal[1] - ego[1], goal[0] - ego[0]) - ego[3], TWO_PI
    )
    out = [
        ego[2] / SPEED_MAX_MPS,
        math.remainder(ego[3], TWO_PI) / math.pi,
        control[0] / max(A_SET),
        control[1] / max(W_SET),
        d_goal / D_GOAL_M,
        max(K_MAX - k, 0) / K_MAX,
        beta_goal / math.pi,
        d_long / D_GOAL_M,
        d_lat / D_HULL_M,
        float(min(abs(d_lat), abs(d_long)) > D_HULL_M),
    ]

    best = [None] * len(_SECTORS)
    for tid, obs in others.items():
        d = math.hypot(obs[0] - ego[0], obs[1] - ego[1])
        rate = (d - prev_range.get(tid, d)) / SIM_DT_S
        prev_range[tid] = d
        if d > D_SENSE_M:
            continue
        j = next(i for i, inside in enumerate(_SECTORS) if inside(ego, obs))
        if best[j] is None or d < best[j][0]:
            rel = math.remainder(
                math.atan2(obs[1] - ego[1], obs[0] - ego[0]) - ego[3], TWO_PI
            )
            best[j] = (d, rel, rate)
    for b in best:
        out.extend(
            _NO_VESSEL
            if b is None
            else (b[0] / D_SENSE_M, b[1] / math.pi, b[2] / (2.0 * SPEED_MAX_MPS))
        )
    out.extend(float(terminal is t) for t in Terminal)
    return np.asarray(out, np.float32)


POLICY_ENV = "COLSIM_KRAS_POLICY"
_policies = {}


def policy(path=None):
    """Load a trained KRAS policy once and cache it.

    Args:
        path (str): Policy zip file, or None to read the environment variable

    Returns:
        MaskablePPO: Loaded policy
    """
    path = path or os.environ.get(POLICY_ENV)
    if not path:
        raise RuntimeError(f"set {POLICY_ENV} to a train_kras.py policy zip")
    if path not in _policies:
        from sb3_contrib import MaskablePPO

        _policies[path] = MaskablePPO.load(path, device="cpu")
    return _policies[path]

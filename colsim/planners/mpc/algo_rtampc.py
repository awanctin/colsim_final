"""
Wang, Tian, Sun, Zhou, Huang, Zeng et al. (2026), "Bridging symmetric dynamics and asymmetric
semantic objectives: runtime-assured predictive safety control for autonomous surface vehicles",
Symmetry 18(7):1123, doi 10.3390/sym18071123.
"""

from __future__ import annotations

import math
import warnings
from enum import Enum

import numpy as np
from scipy.optimize import minimize

L_SCALE = 75.0
T_SCALE = 11.25

D_SAFE_M = 5.0 * L_SCALE
T_SAFE_S = 30.0 * T_SCALE
W_D, W_T = 0.6, 0.4
RISK_ACTIVE = 0.30
M_ACTIVE = 3
HEAD_ON_BEARING = math.radians(10.0)
HEAD_ON_COURSE = math.radians(150.0)
SIDE_LIMIT = math.radians(112.5)

DT_S = 10.0
N_STEPS = 17
D0_M = 3.0 * L_SCALE
K_SIGMA = 1.5
K_SPEED_S = 1.0
POS_NOISE_M = 0.5 * L_SCALE
ALPHA = 0.2
Q_TRACK = 1.0e-6
R_YAW = 1.0e4
R_ACC = 1.0e2
LAMBDA_SEM = 1.0e1
LAMBDA_SAFE = 1.0e5
H_SCALE_M2 = 1.0e6
SLACK_EMERGENCY = 1.0e-2
FEAS_TOL = 1.0e-6
RECOVERY_STEPS = 5
HOLD_BAND = math.radians(5.0)


def _d_safe(own, tgt, j):
    """Compute the safe distance at each step, growing with noise and relative speed.

    Args:
        own (tuple): Own state (x, y, v, heading)
        tgt (tuple): Target state (x, y, v, heading)
        j (ndarray): Step indexes

    Returns:
        ndarray: Safe distance per step, m
    """
    q = math.hypot(
        tgt[2] * math.cos(tgt[3]) - own[2] * math.cos(own[3]),
        tgt[2] * math.sin(tgt[3]) - own[2] * math.sin(own[3]),
    )
    return D0_M + K_SIGMA * POS_NOISE_M * np.sqrt(j) + K_SPEED_S * q


class Mode(Enum):
    NOMINAL = "nominal"
    EMERGENCY = "emergency"


class Encounter(Enum):
    NONE = "non-conflict"
    HEAD_ON = "head-on"
    GIVE_WAY = "starboard crossing"
    STAND_ON = "port crossing"
    OVERTAKING = "overtaking"


def _nav(theta):
    """Convert a math heading to a nautical one.

    Args:
        theta (float): Heading from the x axis, rad

    Returns:
        float: Heading from north, clockwise, rad
    """
    return math.remainder(math.pi / 2.0 - theta, 2.0 * math.pi)


def risk(own, tgt):
    """Compute the collision risk of one target from DCPA and TCPA.

    Args:
        own (tuple): Own state (x, y, v, heading)
        tgt (tuple): Target state (x, y, v, heading)

    Returns:
        tuple: Risk, DCPA (m) and TCPA (s)
    """
    px, py = tgt[0] - own[0], tgt[1] - own[1]
    vx = tgt[2] * math.cos(tgt[3]) - own[2] * math.cos(own[3])
    vy = tgt[2] * math.sin(tgt[3]) - own[2] * math.sin(own[3])
    vv = vx * vx + vy * vy
    if math.sqrt(vv) < 0.05:
        return 0.0, math.hypot(px, py), float("inf")
    tcpa = -(px * vx + py * vy) / (vv + 1e-6)
    dcpa = math.hypot(px + vx * tcpa, py + vy * tcpa)
    if tcpa <= 0.0 or dcpa >= D_SAFE_M:
        return 0.0, dcpa, tcpa
    g_d = max(0.0, 1.0 - dcpa / D_SAFE_M)
    g_t = max(0.0, 1.0 - tcpa / T_SAFE_S)
    return W_D * g_d + W_T * g_t, dcpa, tcpa


def encounter(own, tgt):
    """Classify the encounter with one target by COLREGs.

    Args:
        own (tuple): Own state (x, y, v, heading)
        tgt (tuple): Target state (x, y, v, heading)

    Returns:
        Encounter: Encounter type
    """
    bearing_nav = math.atan2(tgt[0] - own[0], tgt[1] - own[1])
    beta = math.remainder(bearing_nav - _nav(own[3]), 2.0 * math.pi)
    d_psi = abs(math.remainder(_nav(tgt[3]) - _nav(own[3]), 2.0 * math.pi))
    from_target = math.remainder(
        math.atan2(own[0] - tgt[0], own[1] - tgt[1]) - _nav(tgt[3]), 2.0 * math.pi
    )
    if abs(from_target) > SIDE_LIMIT and own[2] > tgt[2]:
        return Encounter.OVERTAKING
    if abs(beta) <= HEAD_ON_BEARING and d_psi >= HEAD_ON_COURSE:
        return Encounter.HEAD_ON
    if HEAD_ON_BEARING < beta < SIDE_LIMIT:
        return Encounter.GIVE_WAY
    if -SIDE_LIMIT < beta < -HEAD_ON_BEARING:
        return Encounter.STAND_ON
    return Encounter.NONE


def _rollout(own, U):
    """Simulate own motion under a sequence of commands.

    Args:
        own (tuple): Own state (x, y, v, heading)
        U (ndarray): Commands (acceleration, turn rate) per step

    Returns:
        tuple: Arrays of x, y and heading
    """
    x, y, v, th = own
    xs, ys, ths = [], [], []
    for a, r in U:
        x += DT_S * v * math.cos(th)
        y += DT_S * v * math.sin(th)
        v = max(v + DT_S * a, 0.0)
        th += DT_S * r
        xs.append(x)
        ys.append(y)
        ths.append(th)
    return np.array(xs), np.array(ys), np.array(ths)


class RTAMPC:

    def __init__(self):
        """Start in nominal mode with zero commands."""
        self.mode = Mode.NOMINAL
        self.calm = 0
        self.U = np.zeros((N_STEPS, 2))
        self.emergency_steps = 0
        self.switches = 0

    def _solve(self, own, active, u_nom, path, rate, acc_box):
        """Solve the MPC with CBF and COLREGs constraints.

        Args:
            own (tuple): Own state (x, y, v, heading)
            active (list): Pairs (target state, encounter)
            u_nom (tuple): Nominal (acceleration, turn rate)
            path (tuple): Start and goal points
            rate (float): Max turn rate, rad/s
            acc_box (tuple): Acceleration bounds

        Returns:
            tuple: First command, feasible flag and max safety slack
        """
        n_t = len(active)
        j = np.arange(1, N_STEPS + 1)
        preds = []
        for tgt, enc in active:
            mu = np.stack(
                [
                    tgt[0] + tgt[2] * math.cos(tgt[3]) * DT_S * j,
                    tgt[1] + tgt[2] * math.sin(tgt[3]) * DT_S * j,
                ],
                axis=1,
            )
            preds.append((mu, _d_safe(own, tgt, j), enc))
        h0 = [
            ((own[0] - t[0]) ** 2 + (own[1] - t[1]) ** 2)
            - _d_safe(own, t, np.array([0]))[0] ** 2
            for t, _ in active
        ]

        n_u = 2 * N_STEPS

        def unpack(z):
            """Split the decision vector into commands and slacks.

            Args:
                z (ndarray): Decision vector

            Returns:
                tuple: Commands, COLREGs slack and safety slacks
            """
            return z[:n_u].reshape(N_STEPS, 2), z[n_u], z[n_u + 1 :]

        (px0, py0), (gx, gy) = path
        ex, ey = gx - px0, gy - py0
        norm = math.hypot(ex, ey) or 1.0
        nx, ny = -ey / norm, ex / norm

        def cost(z):
            """Compute the MPC cost of a decision vector.

            Args:
                z (ndarray): Decision vector

            Returns:
                float: Cost
            """
            U, s_sem, s_safe = unpack(z)
            xs, ys, _ = _rollout(own, U)
            cross = (xs - px0) * nx + (ys - py0) * ny
            du = U - np.array(u_nom)[None, :]
            return (
                Q_TRACK * np.sum(cross**2)
                + R_ACC * np.sum(du[:, 0] ** 2)
                + R_YAW * np.sum(du[:, 1] ** 2)
                + LAMBDA_SEM * s_sem**2
                + LAMBDA_SAFE * np.sum(s_safe**2)
            )

        def cbf(z):
            """Compute the CBF constraints over the horizon.

            Args:
                z (ndarray): Decision vector

            Returns:
                ndarray: Constraint values, must stay >= 0
            """
            U, _s, s_safe = unpack(z)
            xs, ys, _ = _rollout(own, U)
            out = []
            for k, (mu, d_safe, _enc) in enumerate(preds):
                h = (xs - mu[:, 0]) ** 2 + (ys - mu[:, 1]) ** 2 - d_safe**2
                prev = np.concatenate(([h0[k]], h[:-1]))
                out.append((h - (1.0 - ALPHA) * prev) / H_SCALE_M2 + s_safe[k])
            return np.concatenate(out) if out else np.zeros(1)

        def sem(z):
            """Compute the COLREGs turn constraints.

            Args:
                z (ndarray): Decision vector

            Returns:
                ndarray: Constraint values, must stay >= 0
            """
            U, s_sem, _ = unpack(z)
            _, _, ths = _rollout(own, U)
            turn = ths - own[3]
            g = []
            for _mu, _d, enc in preds:
                if enc in (Encounter.HEAD_ON, Encounter.GIVE_WAY):
                    g.append(-turn)
                elif enc is Encounter.STAND_ON:
                    g.append(HOLD_BAND - np.abs(turn))
            if not g:
                return np.ones(1)
            return np.concatenate(g) + s_sem

        z0 = np.concatenate(
            [np.vstack([self.U[1:], self.U[-1:]]).ravel(), [0.0], np.zeros(n_t)]
        )
        bounds = (
            [acc_box, (-rate, rate)] * N_STEPS + [(0.0, None)] + [(0.0, None)] * n_t
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = minimize(
                cost,
                z0,
                method="SLSQP",
                bounds=bounds,
                constraints=[
                    {"type": "ineq", "fun": cbf},
                    {"type": "ineq", "fun": sem},
                ],
                options={"maxiter": 60, "ftol": 1e-6},
            )
        U, _s, s_safe = unpack(res.x)
        self.U = U
        feasible = (
            float(np.min(cbf(res.x))) >= -FEAS_TOL
            and float(np.min(sem(res.x))) >= -FEAS_TOL
        )
        return U[0], feasible, float(np.max(s_safe)) if n_t else 0.0

    def decide(self, own, others, goal, start, u_nom, rate, acc_box):
        """Pick the command and switch to emergency mode when the MPC fails.

        Args:
            own (tuple): Own state (x, y, v, heading)
            others (list): Target states
            goal (tuple): Goal point
            start (tuple): Start point
            u_nom (tuple): Nominal (acceleration, turn rate)
            rate (float): Max turn rate, rad/s
            acc_box (tuple): Acceleration bounds

        Returns:
            tuple: Command (acceleration, turn rate) and mode
        """
        scored = sorted(((risk(own, t)[0], t) for t in others), key=lambda s: -s[0])
        active = [
            (t, encounter(own, t)) for r, t in scored[:M_ACTIVE] if r >= RISK_ACTIVE
        ]

        if not active:
            ok, u0 = True, tuple(u_nom)
        else:
            u0, ok, slack = self._solve(
                own, active, u_nom, (start, goal), rate, acc_box
            )
            ok = ok and slack <= SLACK_EMERGENCY

        if self.mode is Mode.NOMINAL and not ok:
            self.mode, self.calm = Mode.EMERGENCY, 0
            self.switches += 1
        elif self.mode is Mode.EMERGENCY:
            self.calm = self.calm + 1 if ok else 0
            if self.calm >= RECOVERY_STEPS:
                self.mode = Mode.NOMINAL
        if self.mode is Mode.EMERGENCY:
            self.emergency_steps += 1
            return (0.0, -rate), self.mode
        return (float(u0[0]), float(u0[1])), self.mode

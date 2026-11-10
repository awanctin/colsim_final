"""

Johansen, Perez & Cristofaro (2016), "Ship collision avoidance and COLREGS compliance using
simulation-based control behavior selection with predictive hazard assessment", IEEE T-ITS
17(12):3407-3422
"""

from __future__ import annotations

import math

import numpy as np

T_HORIZON_S = 400.0
TS_S = 5.0
D_CLOSE_M = 1000.0
D_SAFE_M = 200.0
P_EXP = 1.0
Q_EXP = 4.0
K_COLL = 1.0
KAPPA = 3.0
LAMBDA_TC = 100.0
K_P = 10.5
K_DP = 0.5
K_CHI_SB = 1.5
K_CHI_P = 10.5
K_DCHI_SB = 0.5
K_DCHI_P = 0.9
PHI_AH_RAD = math.radians(68.5)
PHI_OT_RAD = math.radians(68.5)
PHI_HO_RAD = math.radians(22.5)
PHI_CR_RAD = math.radians(68.5)
MOVING_MPS = 0.25
CHI_OFFSETS_RAD = np.radians(np.arange(-90.0, 91.0, 15.0))
SPEED_FACTORS = np.array([1.0, 0.5, 0.0])


def _unit(a):
    """Build unit vectors from angles.

    Args:
        a (ndarray): Angles, rad

    Returns:
        ndarray: Unit vectors, last axis (cos, sin)
    """
    return np.stack([np.cos(a), np.sin(a)], axis=-1)


def _situation_now(p0, v0, pi, vi, d_close):
    """Classify the encounter with one target at the current time.

    Args:
        p0 (ndarray): Own position, m
        v0 (ndarray): Own velocity, m/s
        pi (ndarray): Target position, m
        vi (ndarray): Target velocity, m/s
        d_close (float): Distance under which the target counts, m

    Returns:
        tuple: Flags (starboard, overtaking, overtaken, crossing)
    """
    d = pi - p0
    dist = float(np.hypot(*d))
    s0, si = float(np.hypot(*v0)), float(np.hypot(*vi))
    heading = math.atan2(v0[1], v0[0])
    bearing = math.remainder(math.atan2(d[1], d[0]) - heading, 2.0 * math.pi)
    starboard = bearing < 0.0
    close = dist <= d_close
    los = d / max(dist, 1e-9)
    ahead = v0 @ los > math.cos(PHI_AH_RAD) * s0
    same_way = v0 @ vi > math.cos(PHI_OT_RAD) * s0 * si
    overtaking = close and ahead and si < s0 and (si < MOVING_MPS or same_way)
    overtaken = close and not ahead and si > s0 and same_way
    crossing = (
        close
        and ahead
        and v0 @ vi < math.cos(PHI_CR_RAD) * s0 * si
        and not overtaking
        and not overtaken
    )
    return starboard, overtaking, overtaken, crossing


def best_behaviour(own, obstacles, chi_d, u_d, last, v_floor=0.0, l_obs=150.0):
    """Pick the course offset and speed factor with the lowest SB-MPC cost.

    Args:
        own (tuple): Own state (x, y, v, heading)
        obstacles (list): Target states
        chi_d (float): Wanted course, rad
        u_d (float): Wanted speed, m/s
        last (tuple): Previous (course offset, speed factor)
        v_floor (float): Lowest allowed speed, m/s
        l_obs (float): Safety distance, m

    Returns:
        tuple: Course offset, speed factor and cost
    """
    chi_last, p_last = last
    x0 = np.array(own[:2], float)
    t = np.arange(TS_S, T_HORIZON_S + 1e-9, TS_S)
    chi = CHI_OFFSETS_RAD[:, None] + np.zeros_like(SPEED_FACTORS)[None, :]
    P = np.zeros_like(CHI_OFFSETS_RAD)[:, None] + SPEED_FACTORS[None, :]
    chi, P = chi.ravel(), P.ravel()
    speed = np.maximum(u_d * P, v_floor)
    course = chi_d + chi
    v_os = _unit(course) * speed[:, None]
    p_os = x0[None, None, :] + v_os[:, None, :] * t[None, :, None]

    v0_now = _unit(np.array(own[3])) * own[2]
    d_chi = chi - chi_last
    f_base = K_P * (1.0 - P) ** 2 + K_DP * (p_last - P) ** 2
    k_chi = np.where(d_chi > 0.0, K_CHI_P, K_CHI_SB)
    k_dchi = np.where(d_chi > 0.0, K_DCHI_P, np.where(d_chi < 0.0, K_DCHI_SB, 0.0))
    f = f_base + k_chi * chi**2 + k_dchi * d_chi**2
    f_sym = (
        f_base + K_CHI_SB * chi**2 + np.where(d_chi != 0.0, K_DCHI_SB, 0.0) * d_chi**2
    )
    cost = f.copy() if not obstacles else np.full(len(chi), -np.inf)
    d_safe = D_SAFE_M + l_obs / 2.0
    for ob in obstacles:
        pi0 = np.array(ob[:2], float)
        vi = _unit(np.array(ob[3])) * ob[2]
        p_ob = pi0[None, :] + vi[None, :] * t[:, None]
        d = p_ob[None, :, :] - p_os
        dist = np.maximum(np.hypot(d[..., 0], d[..., 1]), 1.0)
        close = dist <= D_CLOSE_M

        rel = np.hypot(*(v_os - vi[None, :]).T)
        R = np.where(
            dist <= d_safe, (1.0 / t[None, :] ** P_EXP) * (d_safe / dist) ** Q_EXP, 0.0
        )
        CR = K_COLL * rel[:, None] ** 2 * R

        s0, si = speed[:, None], float(np.hypot(*vi))
        los = d / dist[..., None]
        heading = course[:, None]
        bearing = (
            np.remainder(np.arctan2(d[..., 1], d[..., 0]) - heading + np.pi, 2 * np.pi)
            - np.pi
        )
        starboard = bearing < 0.0
        dot_v = (v_os @ vi)[:, None]
        head_on = (
            (si > MOVING_MPS)
            & (dot_v < -math.cos(PHI_HO_RAD) * s0 * si)
            & (np.einsum("kd,ktd->kt", v_os, los) > math.cos(PHI_AH_RAD) * s0)
        )
        crossed = dot_v < math.cos(PHI_CR_RAD) * s0 * si
        overtaken = (dot_v > math.cos(PHI_OT_RAD) * s0 * si) & (si > s0)
        mu = close & starboard & (head_on | (crossed & ~overtaken))

        sb0, ot0, obt0, cr0 = _situation_now(x0, v0_now, pi0, vi, D_CLOSE_M)
        flipped = starboard != sb0
        to_port = (chi > chi_last)[:, None]
        trans = ((ot0 | obt0) & flipped) | (cr0 & sb0 & starboard & to_port)

        h = CR + KAPPA * mu + LAMBDA_TC * trans
        sym = (ot0 or obt0) and float(np.hypot(*(pi0 - x0))) > D_SAFE_M
        cost = np.maximum(cost, h.max(axis=1) + (f_sym if sym else f))

    k = int(np.argmin(cost))
    return float(chi[k]), float(P[k]), float(cost[k])

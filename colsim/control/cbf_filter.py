"""CBF safety filter Gundersen, Hinostroza & Lekkas (ECC 2026)"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np


class SafetyFilter(Enum):
    NONE = "none"
    CBF = "cbf"


L_DOMAIN_M = 600.0
GAMMA1 = 0.02
GAMMA2 = 0.02
ALPHA_D_RAD = {
    "Overtaking_GiveWay": math.radians(25.0),
    "Head-on": math.radians(36.0),
    "Crossing_GiveWay": math.radians(36.0),
}
FEAS_TOL = 1e-9


def latched_deflection(latch, key, alpha_now, closing):
    """Keep the first non zero deflection while the target is closing.

    Args:
        latch (dict): Stored deflection per target
        key (hashable): Target key
        alpha_now (float): Deflection asked now, rad
        closing (bool): Whether the target is getting closer

    Returns:
        float: Deflection to use, rad
    """

    if not closing:
        latch.pop(key, None)
        return alpha_now
    if key in latch:
        return latch[key]
    if alpha_now != 0.0:
        latch[key] = alpha_now
    return alpha_now


def _constraint(own, tgt, alpha_d, l_m=L_DOMAIN_M, g1=GAMMA1, g2=GAMMA2):
    """Build the linear CBF constraint for one target.

    Args:
        own (tuple): Own state (x, y, v, heading)
        tgt (tuple): Target state (x, y, v, heading)
        alpha_d (float): Wanted deflection angle, rad
        l_m (float): Safety distance, m
        g1 (float): First CBF gain
        g2 (float): Second CBF gain

    Returns:
        tuple: Constraint (A, b) with A u >= b, or None if the vessels overlap
    """
    x, y, v, th = own
    tx, ty, tv, tth = tgt
    rx, ry = x - tx, y - ty
    e = (math.cos(th), math.sin(th))
    rdx = v * e[0] - tv * math.cos(tth)
    rdy = v * e[1] - tv * math.sin(tth)
    rr = rx * rx + ry * ry
    if rr < 1e-9:
        return None

    alpha = math.atan2(ry, rx) + alpha_d
    n = (math.cos(alpha), math.sin(alpha))
    n_perp = (-n[1], n[0])
    alpha_dot = (rx * rdy - ry * rdx) / rr
    n_dot = (alpha_dot * n_perp[0], alpha_dot * n_perp[1])

    h = n[0] * rx + n[1] * ry - l_m
    h_dot = n_dot[0] * rx + n_dot[1] * ry + n[0] * rdx + n[1] * rdy
    drift = 2.0 * (n_dot[0] * rdx + n_dot[1] * rdy)

    # n . r'' = a (n . e) + omega v (n . e_perp)
    A = (n[0] * e[0] + n[1] * e[1], v * (-n[0] * e[1] + n[1] * e[0]))
    b = -(drift + (g1 + g2) * h_dot + g1 * g2 * h)
    return A, b


def _qp2(z_nom, G, hvec):
    """Find the command closest to the nominal one that meets all constraints.

    Args:
        z_nom (ndarray): Nominal command
        G (ndarray): Constraint rows
        hvec (ndarray): Constraint bounds

    Returns:
        tuple: Command and True if feasible, else the least violating command and False
    """
    cands = [np.asarray(z_nom, float)]
    for g, hi in zip(G, hvec):
        gg = g @ g
        if gg > 1e-18:
            cands.append(z_nom + max(0.0, hi - g @ z_nom) / gg * g)
    m = len(G)
    for i in range(m):
        for j in range(i + 1, m):
            M = np.array([G[i], G[j]])
            if abs(np.linalg.det(M)) > 1e-12:
                cands.append(np.linalg.solve(M, [hvec[i], hvec[j]]))
    slack = [np.min(G @ c - hvec) for c in cands]
    feas = [c for c, s in zip(cands, slack) if s >= -FEAS_TOL]
    if feas:
        return min(feas, key=lambda c: float((c - z_nom) @ (c - z_nom))), True
    return cands[int(np.argmax(slack))], False


def filter_command(own, targets, u_nom, box, gammas=(GAMMA1, GAMMA2)):
    """Correct the nominal command so that every CBF constraint holds.

    Args:
        own (tuple): Own state (x, y, v, heading)
        targets (list): Pairs (target state, deflection angle)
        u_nom (tuple): Nominal (acceleration, turn rate)
        box (tuple): Bounds ((a_lo, a_hi), (w_lo, w_hi))
        gammas (tuple): The two CBF gains

    Returns:
        tuple: Filtered command, active flag and feasible flag
    """
    (a_lo, a_hi), (w_lo, w_hi) = box
    s = np.array([max(a_hi - a_lo, 1e-9), max(w_hi - w_lo, 1e-9)])
    rows, rhs = [], []
    for tgt, alpha_d in targets:
        c = _constraint(own, tgt, alpha_d, g1=gammas[0], g2=gammas[1])
        if c is not None:
            rows.append(np.array(c[0]) * s)
            rhs.append(c[1])
    for axis, lo, hi in ((0, a_lo, a_hi), (1, w_lo, w_hi)):
        g = np.zeros(2)
        g[axis] = s[axis]
        rows += [g, -g]
        rhs += [lo, -hi]
    G, hvec = np.array(rows), np.array(rhs)

    u_box = np.clip(np.asarray(u_nom, float), [a_lo, w_lo], [a_hi, w_hi])
    z_nom = u_box / s
    z, feasible = _qp2(z_nom, G, hvec)
    u = np.clip(z * s, [a_lo, w_lo], [a_hi, w_hi])
    active = bool(np.any(np.abs(u - u_box) > 1e-9))
    return u, active, feasible

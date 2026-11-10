from __future__ import annotations

import math

from . import tube as _tube_mod
from .duty import (
    DELTA_MIN_RAD,
    HOLD_BAND_RAD,
    HORIZON_S,
    KAPPA_MAX_RAD,
    SAFETY_RADIUS_M,
    kappa,
    lawful_intervals,
    wrap,
)


def wedge_offsets(os_boat, ts, S=SAFETY_RADIUS_M):
    """Find the heading offsets at the edges of the collision wedge of one target.

    Args:
        os_boat (Vessel): Own vessel
        ts (Vessel): Target vessel
        S (float): Safety radius, m

    Returns:
        list: Heading offsets, rad
    """
    dx, dy = ts.x - os_boat.x, ts.y - os_boat.y
    R = math.hypot(dx, dy)
    if R <= S:
        return []
    g = math.asin(min(S / R, 1.0))
    beta = wrap(math.atan2(dy, dx) - os_boat.theta)
    psi = wrap(ts.theta - os_boat.theta)
    k = ts.v / os_boat.v if os_boat.v > 1e-9 else float("inf")
    out = []
    for sigma in (-g, g):
        y = psi - beta - sigma
        s = k * math.sin(y)
        if abs(s) > 1.0:
            continue
        chi_p = math.asin(s)
        for chi in (chi_p, math.pi - chi_p):
            if math.cos(chi) > k * math.cos(y):
                out.append(wrap(beta + sigma + chi))
    return out


def horizon_breakpoints(os_boat, ts, S=SAFETY_RADIUS_M, T=HORIZON_S):
    """Find the heading offsets where the contact time crosses the horizon.

    Args:
        os_boat (Vessel): Own vessel
        ts (Vessel): Target vessel
        S (float): Safety radius, m
        T (float): Look ahead time, s

    Returns:
        list: Heading offsets, rad
    """
    dx, dy = ts.x - os_boat.x, ts.y - os_boat.y
    cx, cy = ts.v * math.cos(ts.theta), ts.v * math.sin(ts.theta)
    mx, my = dx + T * cx, dy + T * cy
    zx, zy = mx + T * cx, my + T * cy
    nm, nz = math.hypot(mx, my), math.hypot(zx, zy)
    v0T = os_boat.v * T
    out = []
    if nm > 1e-9 and v0T > 1e-9:
        Q = (nm * nm + v0T * v0T - S * S) / (2.0 * v0T * nm)
        if -1.0 <= Q <= 1.0:
            mu = wrap(math.atan2(my, mx) - os_boat.theta)
            out += [wrap(mu - math.acos(Q)), wrap(mu + math.acos(Q))]
    if nz > 1e-9 and os_boat.v > 1e-9:
        Qp = ((mx * cx + my * cy) + os_boat.v**2 * T) / (os_boat.v * nz)
        if -1.0 <= Qp <= 1.0:
            zeta = wrap(math.atan2(zy, zx) - os_boat.theta)
            out += [wrap(zeta - math.acos(Qp)), wrap(zeta + math.acos(Qp))]
    return out


def forbidden(delta, os_boat, ts, S=SAFETY_RADIUS_M, T=HORIZON_S):
    """Check whether a heading offset leads to contact within the horizon.

    Args:
        delta (float): Heading offset, rad
        os_boat (Vessel): Own vessel
        ts (Vessel): Target vessel
        S (float): Safety radius, m
        T (float): Look ahead time, s

    Returns:
        bool: True if the heading is forbidden
    """
    dx, dy = ts.x - os_boat.x, ts.y - os_boat.y
    R = math.hypot(dx, dy)
    if R <= S:
        return True
    a = os_boat.theta + delta
    wx = ts.v * math.cos(ts.theta) - os_boat.v * math.cos(a)
    wy = ts.v * math.sin(ts.theta) - os_boat.v * math.sin(a)
    nw = math.hypot(wx, wy)
    if nw < 1e-12:
        return False
    g = math.asin(min(S / R, 1.0))
    cos_ang = (-wx * dx - wy * dy) / (nw * R)
    if math.acos(max(-1.0, min(1.0, cos_ang))) > g:
        return False
    t_cpa = -(dx * wx + dy * wy) / (nw * nw)
    if t_cpa <= T:
        return True
    return math.hypot(dx + T * wx, dy + T * wy) < S


def _disc_breach(delta, os_boat, ts, tube, S, T, t0=0.0):
    """Check whether a heading enters the growing disc of a target tube.

    Args:
        delta (float): Heading offset, rad
        os_boat (Vessel): Own vessel
        ts (Vessel): Target vessel
        tube (Tube): Target reachable tube
        S (float): Safety radius, m
        T (float): Look ahead time, s
        t0 (float): Start time, s

    Returns:
        bool: True if the disc is entered
    """
    if t0 > T:
        return False
    ex, ey = os_boat.x - ts.x, os_boat.y - ts.y
    a_ang = os_boat.theta + delta
    ux, uy = math.cos(a_ang), math.sin(a_ang)
    v0, vt = os_boat.v, tube.v_max
    A = v0 * v0 - vt * vt
    B = 2.0 * (v0 * (ex * ux + ey * uy) - S * vt)
    C = ex * ex + ey * ey - S * S

    cands = [t0, T]
    if A > 0.0:
        cands.append(min(max(-B / (2.0 * A), t0), T))
    return min(A * t * t + B * t + C for t in cands) < 0.0


def forbidden_tube(delta, os_boat, ts, tube, S=SAFETY_RADIUS_M, T=HORIZON_S, n_t=32):
    """Check whether a heading can meet any path the target tube allows.

    Args:
        delta (float): Heading offset, rad
        os_boat (Vessel): Own vessel
        ts (Vessel): Target vessel
        tube (Tube): Target reachable tube
        S (float): Safety radius, m
        T (float): Look ahead time, s
        n_t (int): Number of time samples

    Returns:
        bool: True if the heading is forbidden
    """
    if tube.is_line:
        return forbidden(delta, os_boat, ts, S, T)
    t_sat = min(tube.saturation_time(), T)
    if _disc_breach(delta, os_boat, ts, tube, S, T, t0=t_sat):
        return True
    if t_sat <= 0.0:
        return False
    a_ang = os_boat.theta + delta
    ux, uy = math.cos(a_ang), math.sin(a_ang)
    for i in range(int(n_t) + 1):
        t = t_sat * i / float(n_t)
        p = (os_boat.x + os_boat.v * ux * t, os_boat.y + os_boat.v * uy * t)
        if _tube_mod.clearance(p, tube, ts, t) < S:
            return True
    return False


def breakpoints(os_boat, threats, ambient, S=SAFETY_RADIUS_M, T=HORIZON_S):
    """List every heading offset where the forbidden set can change.

    Args:
        os_boat (Vessel): Own vessel
        threats (list): Target vessels
        ambient (list): Allowed heading intervals
        S (float): Safety radius, m
        T (float): Look ahead time, s

    Returns:
        list: Heading offsets, rad
    """
    pts = []
    for lo, hi in ambient:
        pts += [lo, hi]
    for ts in threats:
        pts += wedge_offsets(os_boat, ts, S)
        pts += horizon_breakpoints(os_boat, ts, S, T)
    return pts


def surviving_set(os_boat, threats, ambient, S=SAFETY_RADIUS_M, T=HORIZON_S):
    """Compute the width and intervals of headings that avoid every target.

    Args:
        os_boat (Vessel): Own vessel
        threats (list): Target vessels
        ambient (list): Allowed heading intervals
        S (float): Safety radius, m
        T (float): Look ahead time, s

    Returns:
        tuple: Total width (rad) and surviving intervals
    """
    return _alpha_impl(os_boat, threats, ambient, S, T)


def alpha(os_boat, threats, ambient, S=SAFETY_RADIUS_M, T=HORIZON_S):
    """Compute the width of headings that avoid every target.

    Args:
        os_boat (Vessel): Own vessel
        threats (list): Target vessels
        ambient (list): Allowed heading intervals
        S (float): Safety radius, m
        T (float): Look ahead time, s

    Returns:
        float: Total width, rad
    """
    return _alpha_impl(os_boat, threats, ambient, S, T)[0]


def alpha_tube(
    os_boat, threats, tubes, ambient, S=SAFETY_RADIUS_M, T=HORIZON_S, n_delta=180
):
    """Same as surviving_set, with reachable tubes for the targets.

    Args:
        os_boat (Vessel): Own vessel
        threats (list): Target vessels
        tubes (list): Target reachable tubes
        ambient (list): Allowed heading intervals
        S (float): Safety radius, m
        T (float): Look ahead time, s
        n_delta (int): Number of heading samples

    Returns:
        tuple: Total width (rad) and surviving intervals
    """
    if not tubes or all(tb.is_line for tb in tubes):
        return _alpha_impl(os_boat, threats, ambient, S, T)
    span = sum(hi - lo for lo, hi in ambient) or 1.0
    total = 0.0
    surviving = []
    for lo, hi in ambient:
        if hi - lo <= 1e-12:
            continue
        n = max(2, int(round(n_delta * (hi - lo) / span)))
        cuts = [lo + (hi - lo) * i / n for i in range(n + 1)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            mid = 0.5 * (a + b)
            if not any(
                forbidden_tube(mid, os_boat, ts, tb, S, T)
                for ts, tb in zip(threats, tubes)
            ):
                total += b - a
                surviving.append((a, b))
    return total, _merge(surviving)


def _merge(intervals, tol=1e-12):
    """Join intervals that touch.

    Args:
        intervals (list): Sorted pairs (low, high)
        tol (float): Max gap to join

    Returns:
        list: Merged intervals
    """
    out = []
    for a, b in intervals:
        if out and a - out[-1][1] <= tol:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return [tuple(x) for x in out]


def _alpha_impl(os_boat, threats, ambient, S=SAFETY_RADIUS_M, T=HORIZON_S):
    """Split the headings at the breakpoints and keep the safe pieces.

    Args:
        os_boat (Vessel): Own vessel
        threats (list): Target vessels
        ambient (list): Allowed heading intervals
        S (float): Safety radius, m
        T (float): Look ahead time, s

    Returns:
        tuple: Total width (rad) and surviving intervals
    """
    total = 0.0
    surviving = []
    for lo, hi in ambient:
        if hi - lo <= 1e-12:
            continue
        cuts = sorted(
            {lo, hi}
            | {
                p
                for p in breakpoints(os_boat, threats, [(lo, hi)], S, T)
                if lo < p < hi
            }
        )
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a <= 1e-12:
                continue
            mid = 0.5 * (a + b)
            if not any(forbidden(mid, os_boat, ts, S, T) for ts in threats):
                total += b - a
                surviving.append((a, b))
    return total, surviving


def measure(
    os_boat,
    threats,
    *,
    S=SAFETY_RADIUS_M,
    T=HORIZON_S,
    kappa_max=KAPPA_MAX_RAD,
    delta_min=None,
    h=None,
    gate=None,
    **thresholds,
):
    """Compute the physical and lawful escape widths and the deadlock flags.

    Args:
        os_boat (Vessel): Own vessel
        threats (list): Target vessels
        S (float): Safety radius, m
        T (float): Look ahead time, s
        kappa_max (float): Max heading change, rad
        delta_min (float): Smallest lawful turn, rad, or None
        h (float): Hold band, rad, or None
        gate (object): Gate value, or None
        **thresholds (dict): Extra duty thresholds

    Returns:
        dict: Reach, widths, lawful intervals and deadlock flags
    """
    delta_min = DELTA_MIN_RAD if delta_min is None else delta_min
    h = HOLD_BAND_RAD if h is None else h

    k = kappa(os_boat, T, kappa_max)
    reach = [(-k, k)]
    lawful = lawful_intervals(
        os_boat,
        threats,
        T=T,
        kappa_max=kappa_max,
        delta_min=delta_min,
        h=h,
        gate=gate,
        **thresholds,
    )

    a_phys = alpha(os_boat, threats, reach, S, T)
    a_col = alpha(os_boat, threats, lawful, S, T) if lawful else 0.0

    regulatory = not lawful
    physical = bool(lawful) and a_col <= 0.0
    return dict(
        kappa=k,
        alpha_phys=a_phys,
        alpha_col=a_col,
        lawful_width=float(sum(hi - lo for lo, hi in lawful)),
        reach_width=2.0 * k,
        regulatory_deadlock=regulatory,
        physical_deadlock=physical,
        ics_bound=bool(a_phys <= 0.0),
        n_threats=len(threats),
    )

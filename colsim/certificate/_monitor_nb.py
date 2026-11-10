from __future__ import annotations

import ctypes
import math

import llvmlite.binding as llvm
import numpy as np
from numba import njit, types

from . import duty as _duty
from .behaviour import APPARENT_TURN_RAD
from .escape_angle import VISIBLE_RANGE_M

_libm = ctypes.CDLL("libm.so.6")


def _extern(name, n_args):
    """Bind a glibc maths function for numba.

    Args:
        name (str): Function name in libm
        n_args (int): Number of float arguments

    Returns:
        ExternalFunction: Numba callable
    """
    symbol = "colsim_libm_" + name
    llvm.add_symbol(symbol, ctypes.cast(getattr(_libm, name), ctypes.c_void_p).value)
    return types.ExternalFunction(symbol, types.float64(*([types.float64] * n_args)))


c_sin = _extern("sin", 1)
c_cos = _extern("cos", 1)
c_pow = _extern("pow", 2)
c_remainder = _extern("remainder", 2)
c_hypot = _extern("hypot", 2)

TWO_PI = 2.0 * math.pi
T27 = 134217729.0

X, Y, V, TH, ROT = 0, 1, 2, 3, 4

R_NONE, R_OGW, R_BO, R_HO, R_CGW, R_CSO = 0, 1, 2, 3, 4, 5
PRIORITY = np.array([4, 1, 1, 2, 3, 3], dtype=np.int64)

C_COMPLIANT, C_STRAIGHT, C_ADVERSARIAL, C_INTERCEPTOR = 0, 1, 2, 3
CLASS_CODE = {
    "compliant": C_COMPLIANT,
    "adversarial": C_ADVERSARIAL,
    "interceptor": C_INTERCEPTOR,
}

ARM_NONE, ARM_DEONTIC, ARM_GEOMETRIC = 0, 1, 2

MAX_IV = 64
O_ABAFT = float(_duty.O_ABAFT_RAD)
O_DOUBT = float(_duty.O_DOUBT_RAD)
GATE_HORIZON_S = float(_duty.HORIZON_S)
PSI13 = float(_duty.PSI_RULE13_RAD)
R_RECIP = float(_duty.R_RECIPROCAL_RAD)
H_HEADON = float(_duty.H_HEADON_RAD)
VIS_M = float(VISIBLE_RANGE_M)
APPARENT = float(APPARENT_TURN_RAD)


@njit(cache=True)
def py_hypot(x, y):
    """Hypot that matches CPython bit for bit.

    Args:
        x (float): First value
        y (float): Second value

    Returns:
        float: sqrt(x*x + y*y)
    """
    x, y = abs(x), abs(y)
    if math.isinf(x) or math.isinf(y):
        return math.inf
    if math.isnan(x) or math.isnan(y):
        return math.nan
    mx = x if x > y else y
    if mx == 0.0:
        return mx
    _m, max_e = math.frexp(mx)
    csum, frac1, frac2, frac3 = 1.0, 0.0, 0.0, 0.0
    if max_e < -1023:
        for v in (x, y):
            v = v / mx
            v = v * v
            old = csum
            csum += v
            frac1 += (old - csum) + v
        return mx * math.sqrt(csum - 1.0 + frac1)
    scale = math.ldexp(1.0, -max_e)
    for v in (x, y):
        v = v * scale
        t = v * T27
        hi = t - (t - v)
        lo = v - hi
        v = hi * hi
        old = csum
        csum += v
        frac1 += (old - csum) + v
        v = 2.0 * hi * lo
        old = csum
        csum += v
        frac2 += (old - csum) + v
        frac3 += lo * lo
    h = math.sqrt(csum - 1.0 + (frac1 + frac2 + frac3))
    v = h
    t = v * T27
    hi = t - (t - v)
    lo = v - hi
    v = -hi * hi
    old = csum
    csum += v
    frac1 += (old - csum) + v
    v = -2.0 * hi * lo
    old = csum
    csum += v
    frac2 += (old - csum) + v
    v = -lo * lo
    old = csum
    csum += v
    frac3 += (old - csum) + v
    v = csum - 1.0 + (frac1 + frac2 + frac3)
    return (h + v / (2.0 * h)) / scale


@njit(cache=True)
def py_min(a, b):
    """Min that matches Python on ties and NaN.

    Args:
        a (float): First value
        b (float): Second value

    Returns:
        float: Smaller value
    """
    return b if b < a else a


@njit(cache=True)
def py_max(a, b):
    """Max that matches Python on ties and NaN.

    Args:
        a (float): First value
        b (float): Second value

    Returns:
        float: Larger value
    """
    return b if b > a else a


@njit(cache=True)
def wrap(a):
    """Wrap an angle into [-pi, pi] with remainder.

    Args:
        a (float): Angle, rad

    Returns:
        float: Wrapped angle, rad
    """
    return c_remainder(a, TWO_PI)


@njit(cache=True)
def ea_wrap(a):
    """Wrap an angle into [-pi, pi] with atan2.

    Args:
        a (float): Angle, rad

    Returns:
        float: Wrapped angle, rad
    """
    return math.atan2(c_sin(a), c_cos(a))


@njit(cache=True)
def _less(a0, a1, b0, b1):
    """Compare two intervals by start, then end.

    Args:
        a0 (float): First start
        a1 (float): First end
        b0 (float): Second start
        b1 (float): Second end

    Returns:
        bool: True if the first is smaller
    """
    return a0 < b0 or (a0 == b0 and a1 < b1)


@njit(cache=True)
def _sort_iv(buf, n):
    """Sort intervals in place.

    Args:
        buf (ndarray): Intervals
        n (int): Number used
    """
    for i in range(1, n):
        lo, hi = buf[i, 0], buf[i, 1]
        j = i - 1
        while j >= 0 and _less(lo, hi, buf[j, 0], buf[j, 1]):
            buf[j + 1, 0], buf[j + 1, 1] = buf[j, 0], buf[j, 1]
            j -= 1
        buf[j + 1, 0], buf[j + 1, 1] = lo, hi


@njit(cache=True)
def intersect(a, na, b, nb, out):
    """Intersect two interval lists into out.

    Args:
        a (ndarray): First intervals
        na (int): Number used in a
        b (ndarray): Second intervals
        nb (int): Number used in b
        out (ndarray): Output intervals

    Returns:
        int: Number of output intervals
    """
    n = 0
    for i in range(na):
        for j in range(nb):
            lo, hi = py_max(a[i, 0], b[j, 0]), py_min(a[i, 1], b[j, 1])
            if hi - lo > 1e-12:
                if n == out.shape[0]:
                    raise ValueError("interval buffer full")
                out[n, 0], out[n, 1] = lo, hi
                n += 1
    _sort_iv(out, n)
    return n


@njit(cache=True)
def merge(buf, n, out):
    """Sort and join overlapping intervals into out.

    Args:
        buf (ndarray): Intervals
        n (int): Number used
        out (ndarray): Output intervals

    Returns:
        int: Number of output intervals
    """
    _sort_iv(buf, n)
    m = 0
    for i in range(n):
        lo, hi = buf[i, 0], buf[i, 1]
        if hi - lo <= 1e-12:
            continue
        if m and lo <= out[m - 1, 1] + 1e-12:
            out[m - 1, 1] = py_max(out[m - 1, 1], hi)
        else:
            out[m, 0], out[m, 1] = lo, hi
            m += 1
    return m


@njit(cache=True)
def kappa(s, T, kappa_max):
    """Compute the largest heading change in the horizon.

    Args:
        s (ndarray): Vessel row
        T (float): Look ahead time, s
        kappa_max (float): Upper limit, rad

    Returns:
        float: Max heading change, rad
    """
    k = s[ROT] * T
    if kappa_max < k:
        k = kappa_max
    if math.pi < k:
        k = math.pi
    return k


@njit(cache=True)
def straight_dcpa(o, t):
    """Compute the DCPA if both vessels keep course and speed.

    Args:
        o (ndarray): First row
        t (ndarray): Second row

    Returns:
        float: DCPA, m
    """
    dx, dy = t[X] - o[X], t[Y] - o[Y]
    rvx = t[V] * c_cos(t[TH]) - o[V] * c_cos(o[TH])
    rvy = t[V] * c_sin(t[TH]) - o[V] * c_sin(o[TH])
    vv = rvx * rvx + rvy * rvy
    if vv < 1e-12:
        return py_hypot(dx, dy)
    tt = -(rvx * dx + rvy * dy) / vv
    if tt <= 0.0:
        return py_hypot(dx, dy)
    return py_hypot(dx + tt * rvx, dy + tt * rvy)


@njit(cache=True)
def closing(o, t, horizon):
    """Check that the CPA is ahead in time and within the horizon.

    Args:
        o (ndarray): First row
        t (ndarray): Second row
        horizon (float): Max TCPA, s

    Returns:
        bool: True if closing
    """
    dx, dy = t[X] - o[X], t[Y] - o[Y]
    rvx = t[V] * c_cos(t[TH]) - o[V] * c_cos(o[TH])
    rvy = t[V] * c_sin(t[TH]) - o[V] * c_sin(o[TH])
    vv = rvx * rvx + rvy * rvy
    if vv < 1e-12:
        return False
    tt = -(rvx * dx + rvy * dy) / vv
    return 0.0 < tt <= horizon


@njit(cache=True)
def role(o, t, use_gate, gate):
    """Get the duty role code of o toward t.

    Args:
        o (ndarray): Own row
        t (ndarray): Target row
        use_gate (bool): Whether to apply the DCPA gate
        gate (float): DCPA gate, m

    Returns:
        int: Role code
    """
    if use_gate and straight_dcpa(o, t) > gate:
        return R_NONE
    if use_gate and not closing(o, t, GATE_HORIZON_S):
        return R_NONE
    a_j = math.atan2(t[Y] - o[Y], t[X] - o[X])
    beta = wrap(a_j - o[TH])
    psi = wrap(t[TH] - o[TH])
    b = wrap(beta + math.pi - psi)
    faster = o[V] > t[V]
    slower = t[V] > o[V]
    best = R_NONE
    if abs(b) > O_ABAFT - O_DOUBT and abs(psi) < PSI13 and faster:
        best = R_OGW
    if abs(beta) > O_ABAFT - O_DOUBT and abs(psi) < PSI13 and slower:
        if best == R_NONE or PRIORITY[R_BO] < PRIORITY[best]:
            best = R_BO
    if abs(beta) < H_HEADON and abs(psi) > R_RECIP:
        if best == R_NONE or PRIORITY[R_HO] < PRIORITY[best]:
            best = R_HO
    if -O_ABAFT < beta < 0:
        if best == R_NONE or PRIORITY[R_CGW] < PRIORITY[best]:
            best = R_CGW
    if 0 <= beta < O_ABAFT:
        if best == R_NONE or PRIORITY[R_CSO] < PRIORITY[best]:
            best = R_CSO
    return best


@njit(cache=True)
def roles_of(o, ts, use_gate, gate, out):
    """Get the role code toward every target.

    Args:
        o (ndarray): Own row (x, y, v, heading, turn rate)
        ts (ndarray): Target rows, same layout
        use_gate (bool): Whether to apply the DCPA gate
        gate (float): DCPA gate, m
        out (ndarray): Role codes, filled
    """
    for i in range(ts.shape[0]):
        out[i] = role(o, ts[i], use_gate, gate)


@njit(cache=True)
def role_mem(o, t, use_gate, gate, m):
    """Get the role code, keeping a stored overtaking role.

    Args:
        o (ndarray): Own row
        t (ndarray): Target row
        use_gate (bool): Whether to apply the DCPA gate
        gate (float): DCPA gate, m
        m (int): Stored role code

    Returns:
        int: Role code
    """
    if m == R_OGW or m == R_BO:
        if use_gate and straight_dcpa(o, t) > gate:
            return R_NONE
        return m
    return role(o, t, use_gate, gate)


@njit(cache=True)
def roles_of_mem(o, ts, use_gate, gate, mem, out):
    """Get the role code toward every target, with memory.

    Args:
        o (ndarray): Own row (x, y, v, heading, turn rate)
        ts (ndarray): Target rows, same layout
        use_gate (bool): Whether to apply the DCPA gate
        gate (float): DCPA gate, m
        mem (ndarray): Stored role code per target
        out (ndarray): Role codes, filled
    """
    for i in range(ts.shape[0]):
        out[i] = role_mem(o, ts[i], use_gate, gate, mem[i])


ROLE_NAMES = ("NONE", "OGW", "BO", "HO", "CGW", "CSO")


@njit(cache=True)
def duty_intervals(r, k, delta_min, h, relaxed, out, tmp):
    """Get the heading changes allowed by one role.

    Args:
        r (int): Role code
        k (float): Max heading change, rad
        delta_min (float): Smallest lawful turn, rad
        h (float): Hold band, rad
        relaxed (bool): Whether Rule 17 is relaxed
        out (ndarray): Output intervals
        tmp (ndarray): Work buffer

    Returns:
        int: Number of output intervals
    """
    d = py_min(py_max(delta_min, 0.0), k)
    hh = py_min(py_max(h, 0.0), k)
    if r == R_HO or r == R_CGW:
        out[0, 0], out[0, 1] = -k, -d
        return 1
    if r == R_OGW:
        out[0, 0], out[0, 1] = -k, -d
        out[1, 0], out[1, 1] = d, k
        return 2
    if r == R_CSO or r == R_BO:
        if not relaxed:
            out[0, 0], out[0, 1] = -hh, hh
            return 1
        tmp[0, 0], tmp[0, 1] = -k, -d
        tmp[1, 0], tmp[1, 1] = -hh, hh
        n = 2
        if r == R_BO:
            tmp[2, 0], tmp[2, 1] = d, k
            n = 3
        return merge(tmp, n, out)
    out[0, 0], out[0, 1] = -k, k
    return 1


G_DX, G_DY, G_R, G_G, G_BETA, G_PSI, G_CT, G_ST = 0, 1, 2, 3, 4, 5, 6, 7
N_GEO = 8


@njit(cache=True)
def geo_fill(o, ts, S, geo):
    """Store the range, bearing and wedge angle of each target.

    Args:
        o (ndarray): Own row (x, y, v, heading, turn rate)
        ts (ndarray): Target rows, same layout
        S (float): Safety radius, m
        geo (ndarray): Cached geometry per target
    """
    for j in range(ts.shape[0]):
        dx, dy = ts[j, X] - o[X], ts[j, Y] - o[Y]
        R = py_hypot(dx, dy)
        geo[j, G_DX] = dx
        geo[j, G_DY] = dy
        geo[j, G_R] = R
        geo[j, G_G] = math.asin(py_min(S / R, 1.0)) if R > S else math.nan
        geo[j, G_BETA] = wrap(math.atan2(dy, dx) - o[TH])
        geo[j, G_PSI] = wrap(ts[j, TH] - o[TH])


@njit(cache=True)
def trig_fill(ts, geo):
    """Store the cosine and sine of each target heading.

    Args:
        ts (ndarray): Target rows, same layout
        geo (ndarray): Cached geometry per target
    """
    for j in range(ts.shape[0]):
        geo[j, G_CT] = c_cos(ts[j, TH])
        geo[j, G_ST] = c_sin(ts[j, TH])


@njit(cache=True)
def gated_geo(o, t, gj, co, so, gate):
    """Check whether a target is outside the DCPA gate, using the cache.

    Args:
        o (ndarray): Own row
        t (ndarray): Target row (x, y, v, heading, turn rate)
        gj (ndarray): Cached geometry of the target
        co (float): Cosine of own heading
        so (float): Sine of own heading
        gate (float): DCPA gate, m

    Returns:
        bool: True if outside the gate
    """
    dx, dy = gj[G_DX], gj[G_DY]
    rvx = t[V] * gj[G_CT] - o[V] * co
    rvy = t[V] * gj[G_ST] - o[V] * so
    vv = rvx * rvx + rvy * rvy
    if vv < 1e-12:
        d = gj[G_R]
    else:
        tt = -(rvx * dx + rvy * dy) / vv
        if tt <= 0.0:
            d = gj[G_R]
        else:
            d = py_hypot(dx + tt * rvx, dy + tt * rvy)
    return d > gate


@njit(cache=True)
def role_geo(o, t, gj, co, so, use_gate, gate):
    """Get the role code using the cached geometry.

    Args:
        o (ndarray): Own row
        t (ndarray): Target row (x, y, v, heading, turn rate)
        gj (ndarray): Cached geometry of the target
        co (float): Cosine of own heading
        so (float): Sine of own heading
        use_gate (bool): Whether to apply the DCPA gate
        gate (float): DCPA gate, m

    Returns:
        int: Role code
    """
    if use_gate:
        dx, dy = gj[G_DX], gj[G_DY]
        rvx = t[V] * gj[G_CT] - o[V] * co
        rvy = t[V] * gj[G_ST] - o[V] * so
        vv = rvx * rvx + rvy * rvy
        if vv < 1e-12:
            d = gj[G_R]
        else:
            tt = -(rvx * dx + rvy * dy) / vv
            if tt <= 0.0:
                d = gj[G_R]
            else:
                d = py_hypot(dx + tt * rvx, dy + tt * rvy)
        if d > gate:
            return R_NONE
        if vv < 1e-12 or not (0.0 < -(rvx * dx + rvy * dy) / vv <= GATE_HORIZON_S):
            return R_NONE
    beta = gj[G_BETA]
    psi = gj[G_PSI]
    b = wrap(beta + math.pi - psi)
    faster = o[V] > t[V]
    slower = t[V] > o[V]
    best = R_NONE
    if abs(b) > O_ABAFT - O_DOUBT and abs(psi) < PSI13 and faster:
        best = R_OGW
    if abs(beta) > O_ABAFT - O_DOUBT and abs(psi) < PSI13 and slower:
        if best == R_NONE or PRIORITY[R_BO] < PRIORITY[best]:
            best = R_BO
    if abs(beta) < H_HEADON and abs(psi) > R_RECIP:
        if best == R_NONE or PRIORITY[R_HO] < PRIORITY[best]:
            best = R_HO
    if -O_ABAFT < beta < 0:
        if best == R_NONE or PRIORITY[R_CGW] < PRIORITY[best]:
            best = R_CGW
    if 0 <= beta < O_ABAFT:
        if best == R_NONE or PRIORITY[R_CSO] < PRIORITY[best]:
            best = R_CSO
    return best


@njit(cache=True)
def lawful(
    o,
    ts,
    geo,
    co,
    so,
    use_gate,
    gate,
    relaxed,
    mem,
    T,
    kappa_max,
    delta_min,
    h,
    cur,
    work,
):
    """Get the heading changes allowed by every duty at once.

    Args:
        o (ndarray): Own row (x, y, v, heading, turn rate)
        ts (ndarray): Target rows, same layout
        geo (ndarray): Cached geometry per target
        co (float): Cosine of own heading
        so (float): Sine of own heading
        use_gate (bool): Whether to apply the DCPA gate
        gate (float): DCPA gate, m
        relaxed (ndarray): Relaxation flag per target
        mem (ndarray): Stored role code per target
        T (float): Look ahead time, s
        kappa_max (float): Max heading change, rad
        delta_min (float): Smallest lawful turn, rad
        h (float): Hold band, rad
        cur (ndarray): Output intervals
        work (ndarray): Work buffers

    Returns:
        int: Number of lawful intervals
    """
    k = kappa(o, T, kappa_max)
    cur[0, 0], cur[0, 1] = -k, k
    n = 1
    duty, tmp, nxt = work[0], work[1], work[2]
    for i in range(ts.shape[0]):
        m = mem[i]
        if m == R_OGW or m == R_BO:
            r = (
                R_NONE
                if (use_gate and gated_geo(o, ts[i], geo[i], co, so, gate))
                else m
            )
        else:
            r = role_geo(o, ts[i], geo[i], co, so, use_gate, gate)
        nd = duty_intervals(r, k, delta_min, h, relaxed[i] != 0, duty, tmp)
        n = intersect(cur, n, duty, nd, nxt)
        cur[:n] = nxt[:n]
        if n == 0:
            return 0
    return n


@njit(cache=True)
def _push_if_inside(p, lo, hi, buf, n):
    """Add a value to a buffer if it lies strictly inside a range.

    Args:
        p (float): Value
        lo (float): Range start
        hi (float): Range end
        buf (ndarray): Buffer
        n (int): Number used

    Returns:
        int: New number used
    """
    if lo < p < hi:
        buf[n] = p
        n += 1
    return n


@njit(cache=True)
def breakpoints(o, ts, geo, S, T, ov2, lo, hi, buf):
    """Collect the heading changes where the forbidden set can change.

    Args:
        o (ndarray): Own row (x, y, v, heading, turn rate)
        ts (ndarray): Target rows, same layout
        geo (ndarray): Cached geometry per target
        S (float): Safety radius, m
        T (float): Look ahead time, s
        ov2 (float): Own speed squared
        lo (float): Range start, rad
        hi (float): Range end, rad
        buf (ndarray): Output buffer

    Returns:
        int: Number of values
    """
    buf[0], buf[1] = lo, hi
    n = 2
    ov, oth = o[V], o[TH]
    for j in range(ts.shape[0]):
        tv = ts[j, V]
        dx, dy, R = geo[j, G_DX], geo[j, G_DY], geo[j, G_R]
        if R > S:
            g = geo[j, G_G]
            beta = geo[j, G_BETA]
            psi = geo[j, G_PSI]
            kk = tv / ov if ov > 1e-9 else math.inf
            for sigma in (-g, g):
                y = psi - beta - sigma
                s = kk * c_sin(y)
                if abs(s) > 1.0:
                    continue
                chi_p = math.asin(s)
                kcy = kk * c_cos(y)
                for chi in (chi_p, math.pi - chi_p):
                    if c_cos(chi) > kcy:
                        n = _push_if_inside(wrap(beta + sigma + chi), lo, hi, buf, n)
        cx, cy = tv * geo[j, G_CT], tv * geo[j, G_ST]
        mx, my = dx + T * cx, dy + T * cy
        zx, zy = mx + T * cx, my + T * cy
        nm, nz = py_hypot(mx, my), py_hypot(zx, zy)
        v0T = ov * T
        if nm > 1e-9 and v0T > 1e-9:
            Q = (nm * nm + v0T * v0T - S * S) / (2.0 * v0T * nm)
            if -1.0 <= Q <= 1.0:
                mu = wrap(math.atan2(my, mx) - oth)
                aq = math.acos(Q)
                n = _push_if_inside(wrap(mu - aq), lo, hi, buf, n)
                n = _push_if_inside(wrap(mu + aq), lo, hi, buf, n)
        if nz > 1e-9 and ov > 1e-9:
            Qp = ((mx * cx + my * cy) + ov2 * T) / (ov * nz)
            if -1.0 <= Qp <= 1.0:
                zeta = wrap(math.atan2(zy, zx) - oth)
                aq = math.acos(Qp)
                n = _push_if_inside(wrap(zeta - aq), lo, hi, buf, n)
                n = _push_if_inside(wrap(zeta + aq), lo, hi, buf, n)
    return n


@njit(cache=True)
def forbidden_past_range(ca, sa, o, t, gj, S, T):
    """Check whether a heading leads to contact within the horizon.

    Args:
        ca (float): Cosine of the heading
        sa (float): Sine of the heading
        o (ndarray): Own row
        t (ndarray): Target row (x, y, v, heading, turn rate)
        gj (ndarray): Cached geometry of the target
        S (float): Safety radius, m
        T (float): Look ahead time, s

    Returns:
        bool: True if forbidden
    """
    dx, dy, R = gj[G_DX], gj[G_DY], gj[G_R]
    wx = t[V] * gj[G_CT] - o[V] * ca
    wy = t[V] * gj[G_ST] - o[V] * sa
    nw = py_hypot(wx, wy)
    if nw < 1e-12:
        return False
    g = gj[G_G]
    cos_ang = (-wx * dx - wy * dy) / (nw * R)
    if math.acos(py_max(-1.0, py_min(1.0, cos_ang))) > g:
        return False
    t_cpa = -(dx * wx + dy * wy) / (nw * nw)
    if t_cpa <= T:
        return True
    return py_hypot(dx + T * wx, dy + T * wy) < S


@njit(cache=True)
def heading_blocked(delta, o, ts, geo, S, T):
    """Check whether a heading change leads to contact with any target.

    Args:
        delta (float): Heading change, rad
        o (ndarray): Own row (x, y, v, heading, turn rate)
        ts (ndarray): Target rows, same layout
        geo (ndarray): Cached geometry per target
        S (float): Safety radius, m
        T (float): Look ahead time, s

    Returns:
        bool: True if blocked
    """
    have = False
    ca, sa = 0.0, 0.0
    for j in range(ts.shape[0]):
        if geo[j, G_R] <= S:
            return True
        if not have:
            a = o[TH] + delta
            ca, sa = c_cos(a), c_sin(a)
            have = True
        if forbidden_past_range(ca, sa, o, ts[j], geo[j], S, T):
            return True
    return False


@njit(cache=True)
def _isort(buf, n):
    """Sort values in place.

    Args:
        buf (ndarray): Values
        n (int): Number used
    """
    for i in range(1, n):
        v = buf[i]
        j = i - 1
        while j >= 0 and v < buf[j]:
            buf[j + 1] = buf[j]
            j -= 1
        buf[j + 1] = v


@njit(cache=True)
def _sorted_cuts(buf, n):
    """Sort the cut points, with numpy if a NaN is present.

    Args:
        buf (ndarray): Values
        n (int): Number used

    Returns:
        ndarray: Sorted values
    """
    for i in range(n):
        if buf[i] != buf[i]:
            return np.sort(buf[:n])
    _isort(buf, n)
    return buf[:n]


@njit(cache=True)
def alpha(o, ts, geo, amb, n_amb, S, T, ov2, buf, stop_at_free):
    """Measure the width of headings that avoid every target.

    Args:
        o (ndarray): Own row (x, y, v, heading, turn rate)
        ts (ndarray): Target rows, same layout
        geo (ndarray): Cached geometry per target
        amb (ndarray): Allowed intervals
        n_amb (int): Number of allowed intervals
        S (float): Safety radius, m
        T (float): Look ahead time, s
        ov2 (float): Own speed squared
        buf (ndarray): Work buffer
        stop_at_free (bool): Whether to stop at the first free piece

    Returns:
        float: Free width, rad
    """
    total = 0.0
    for i in range(n_amb):
        lo, hi = amb[i, 0], amb[i, 1]
        if hi - lo <= 1e-12:
            continue
        n = breakpoints(o, ts, geo, S, T, ov2, lo, hi, buf)
        cuts = _sorted_cuts(buf, n)
        prev = cuts[0]
        for c in range(1, n):
            b = cuts[c]
            a = prev
            prev = b
            if b - a <= 1e-12:
                continue
            if not heading_blocked(0.5 * (a + b), o, ts, geo, S, T):
                total += b - a
                if stop_at_free:
                    return total
    return total


@njit(cache=True)
def gives_way(thr, own):
    """Check whether a target must give way to own vessel.

    Args:
        thr (ndarray): Target row
        own (ndarray): Own row

    Returns:
        bool: True if the target gives way
    """
    dx, dy = own[X] - thr[X], own[Y] - thr[Y]
    if dx * dx + dy * dy > VIS_M * VIS_M:
        return False
    if not closing(thr, own, math.inf):
        return False
    r = role(thr, own, False, 0.0)
    return r == R_OGW or r == R_HO or r == R_CGW


@njit(cache=True)
def roll_threat(t, o, cls, dt, turned, has_aim, ax, ay, out, gj, co, so):
    """Move a target one step with the behaviour of its class.

    Args:
        t (ndarray): Target row (x, y, v, heading, turn rate)
        o (ndarray): Own row
        cls (int): Behaviour class code
        dt (float): Time step, s
        turned (float): Turn already made, rad
        has_aim (bool): Whether an aim point is set
        ax (float): Aim x, m
        ay (float): Aim y, m
        out (ndarray): Moved row, filled
        gj (ndarray): Cached geometry of the target
        co (float): Cosine of own heading
        so (float): Sine of own heading

    Returns:
        float: Total turn, rad
    """
    rate = t[ROT] * dt
    d = 0.0
    if cls == C_ADVERSARIAL:
        d = py_min(py_max(ea_wrap(intercept_heading(t, o) - t[TH]), -rate), rate)
    elif cls == C_INTERCEPTOR:
        px, py = (ax, ay) if has_aim else (o[X], o[Y])
        want = math.atan2(py - t[Y], px - t[X])
        d = py_min(py_max(ea_wrap(want - t[TH]), -rate), rate)
    elif cls == C_COMPLIANT and gives_way(t, o):
        remaining = py_max(APPARENT - abs(turned), 0.0)
        d = -py_min(rate, remaining)
    a = t[TH] + d
    if a == t[TH] and a != 0.0:
        theta = math.atan2(gj[G_ST], gj[G_CT])
    else:
        theta = math.atan2(c_sin(a), c_cos(a))
    ct, st = c_cos(theta), c_sin(theta)
    out[X] = t[X] + t[V] * ct * dt
    out[Y] = t[Y] + t[V] * st * dt
    out[V] = t[V]
    out[TH] = theta
    out[ROT] = t[ROT]
    gj[G_CT] = ct
    gj[G_ST] = st
    return turned + d


@njit(cache=True)
def rollout(
    o0,
    ts0,
    classes,
    aims,
    has_aim,
    n_steps,
    dt,
    commit,
    n_commit,
    use_gate,
    gate,
    relaxed,
    mem,
    T,
    kappa_max,
    delta_min,
    h,
    S,
    os_out,
    ts_out,
):
    """Roll one future forward until a certificate fires.

    Args:
        o0 (ndarray): Own row
        ts0 (ndarray): Target rows
        classes (ndarray): Class code per target
        aims (ndarray): Aim point per target
        has_aim (ndarray): Aim flag per target
        n_steps (int): Number of steps
        dt (float): Time step, s
        commit (ndarray): Commitment intervals
        n_commit (int): Number of commitment intervals
        use_gate (bool): Whether to apply the DCPA gate
        gate (float): DCPA gate, m
        relaxed (ndarray): Relaxation flag per target
        mem (ndarray): Stored role code per target
        T (float): Look ahead time, s
        kappa_max (float): Max heading change, rad
        delta_min (float): Smallest lawful turn, rad
        h (float): Hold band, rad
        S (float): Safety radius, m
        os_out (ndarray): Final own row, filled
        ts_out (ndarray): Final target rows, filled

    Returns:
        tuple: Fire time, certificate code and steps used
    """
    n = ts0.shape[0]
    o = o0.copy()
    ts = ts0.copy()
    nxt = np.empty_like(ts)
    turned = np.zeros(n)
    cur = np.empty((MAX_IV, 2))
    work = np.empty((3, MAX_IV, 2))
    geo = np.empty((n, N_GEO))
    buf = np.empty(2 + 8 * n)
    trig_fill(ts, geo)
    co, so = c_cos(o[TH]), c_sin(o[TH])
    ov2 = c_pow(o[V], 2.0)
    for i in range(n_steps + 1):
        t = i * dt
        geo_fill(o, ts, S, geo)
        k = lawful(
            o,
            ts,
            geo,
            co,
            so,
            use_gate,
            gate,
            relaxed,
            mem,
            T,
            kappa_max,
            delta_min,
            h,
            cur,
            work,
        )
        if n_commit > 0:
            nx = work[2]
            k = intersect(cur, k, commit, n_commit, nx)
            cur[:k] = nx[:k]
        if k == 0:
            return t, ARM_DEONTIC, i + 1
        if alpha(o, ts, geo, cur, k, S, T, ov2, buf, True) <= 0.0:
            os_out[:] = o
            ts_out[:] = ts
            return t, ARM_GEOMETRIC, i + 1
        if i == n_steps:
            break
        x = o[X] + o[V] * co * dt
        y = o[Y] + o[V] * so * dt
        o[X], o[Y] = x, y
        for j in range(n):
            turned[j] = roll_threat(
                ts[j],
                o,
                classes[j],
                dt,
                turned[j],
                has_aim[j],
                aims[j, 0],
                aims[j, 1],
                nxt[j],
                geo[j],
                co,
                so,
            )
        ts[:] = nxt
    return math.nan, ARM_NONE, n_steps + 1


@njit(cache=True)
def step_cert_now(
    o, ts, commit, n_commit, use_gate, gate, relaxed, mem, T, kappa_max, delta_min, h, S
):
    """Check the deontic and geometric certificates at the current state.

    Args:
        o (ndarray): Own row (x, y, v, heading, turn rate)
        ts (ndarray): Target rows, same layout
        commit (ndarray): Commitment intervals
        n_commit (int): Number of commitment intervals
        use_gate (bool): Whether to apply the DCPA gate
        gate (float): DCPA gate, m
        relaxed (ndarray): Relaxation flag per target
        mem (ndarray): Stored role code per target
        T (float): Look ahead time, s
        kappa_max (float): Max heading change, rad
        delta_min (float): Smallest lawful turn, rad
        h (float): Hold band, rad
        S (float): Safety radius, m

    Returns:
        tuple: Deontic flag, geometric flag, compliant width, physical width, lawful width and reach
    """
    n = ts.shape[0]
    cur = np.empty((MAX_IV, 2))
    work = np.empty((3, MAX_IV, 2))
    geo = np.empty((n, N_GEO))
    buf = np.empty(2 + 8 * n)
    trig_fill(ts, geo)
    geo_fill(o, ts, S, geo)
    co, so = c_cos(o[TH]), c_sin(o[TH])
    ov2 = c_pow(o[V], 2.0)
    k = kappa(o, T, kappa_max)
    m = lawful(
        o,
        ts,
        geo,
        co,
        so,
        use_gate,
        gate,
        relaxed,
        mem,
        T,
        kappa_max,
        delta_min,
        h,
        cur,
        work,
    )
    if n_commit > 0:
        nx = work[2]
        m = intersect(cur, m, commit, n_commit, nx)
        cur[:m] = nx[:m]
    deontic = m == 0
    a_col = alpha(o, ts, geo, cur, m, S, T, ov2, buf, False) if m > 0 else 0.0
    reach = work[1]
    reach[0, 0], reach[0, 1] = -k, k
    a_phys = alpha(o, ts, geo, reach, 1, S, T, ov2, buf, False)
    width = 0.0
    for i in range(m):
        width += cur[i, 1] - cur[i, 0]
    geometric = m > 0 and a_col <= 0.0
    return deontic, geometric, a_col, a_phys, width, k


@njit(cache=True)
def predict_core(
    o,
    ts,
    codes,
    aims,
    has_aim,
    relaxed_fut,
    n_steps,
    dt,
    commit,
    n_commit,
    use_gate,
    gate,
    relaxed_now,
    mem,
    T,
    kappa_max,
    delta_min,
    h,
    S,
    t_out,
    arm_out,
    ns_out,
    os_out,
    ts_out,
):
    """Roll every future and check the certificates now.

    Args:
        o (ndarray): Own row (x, y, v, heading, turn rate)
        ts (ndarray): Target rows, same layout
        codes (ndarray): Class codes per future and target
        aims (ndarray): Aim point per target
        has_aim (ndarray): Aim flags per future and target
        relaxed_fut (ndarray): Relaxation flags per future and target
        n_steps (int): Number of steps
        dt (float): Time step, s
        commit (ndarray): Commitment intervals
        n_commit (int): Number of commitment intervals
        use_gate (bool): Whether to apply the DCPA gate
        gate (float): DCPA gate, m
        relaxed_now (ndarray): Relaxation flag per target now
        mem (ndarray): Stored role code per target
        T (float): Look ahead time, s
        kappa_max (float): Max heading change, rad
        delta_min (float): Smallest lawful turn, rad
        h (float): Hold band, rad
        S (float): Safety radius, m
        t_out (ndarray): Fire time per future, filled
        arm_out (ndarray): Certificate code per future, filled
        ns_out (ndarray): Steps used per future, filled
        os_out (ndarray): Final own row per future, filled
        ts_out (ndarray): Final target rows per future, filled

    Returns:
        tuple: Certificate result at the current state
    """
    for p in range(codes.shape[0]):
        t_out[p], arm_out[p], ns_out[p] = rollout(
            o,
            ts,
            codes[p],
            aims,
            has_aim[p],
            n_steps,
            dt,
            commit,
            n_commit,
            use_gate,
            gate,
            relaxed_fut[p],
            mem,
            T,
            kappa_max,
            delta_min,
            h,
            S,
            os_out[p],
            ts_out[p],
        )
    return step_cert_now(
        o,
        ts,
        commit,
        n_commit,
        use_gate,
        gate,
        relaxed_now,
        mem,
        T,
        kappa_max,
        delta_min,
        h,
        S,
    )


@njit(cache=True)
def observe_core(
    o, ts, mid, use_gate, gate, relaxed, mem, T, kappa_max, delta_min, h, S, lawful_out
):
    """Compute the blocked bearing bins and the lawful intervals.

    Args:
        o (ndarray): Own row (x, y, v, heading, turn rate)
        ts (ndarray): Target rows, same layout
        mid (ndarray): Centre of each heading bin, rad
        use_gate (bool): Whether to apply the DCPA gate
        gate (float): DCPA gate, m
        relaxed (ndarray): Relaxation flag per target
        mem (ndarray): Stored role code per target
        T (float): Look ahead time, s
        kappa_max (float): Max heading change, rad
        delta_min (float): Smallest lawful turn, rad
        h (float): Hold band, rad
        S (float): Safety radius, m
        lawful_out (ndarray): Lawful intervals, filled

    Returns:
        tuple: Blocked bins, number of lawful intervals, physical width and compliant width
    """
    n = ts.shape[0]
    geo = np.empty((n, N_GEO))
    buf = np.empty(2 + 8 * n)
    trig_fill(ts, geo)
    geo_fill(o, ts, S, geo)
    co, so = c_cos(o[TH]), c_sin(o[TH])
    ov2 = c_pow(o[V], 2.0)
    blocked = np.zeros(mid.shape[0], dtype=np.float32)
    for b in range(mid.shape[0]):
        if heading_blocked(mid[b], o, ts, geo, S, T):
            blocked[b] = 1.0
    work = np.empty((3, MAX_IV, 2))
    m = lawful(
        o,
        ts,
        geo,
        co,
        so,
        use_gate,
        gate,
        relaxed,
        mem,
        T,
        kappa_max,
        delta_min,
        h,
        lawful_out,
        work,
    )
    k = kappa(o, T, kappa_max)
    reach = work[1]
    reach[0, 0], reach[0, 1] = -k, k
    a_phys = alpha(o, ts, geo, reach, 1, S, T, ov2, buf, False)
    a_col = alpha(o, ts, geo, lawful_out, m, S, T, ov2, buf, False) if m > 0 else 0.0
    return blocked, m, a_phys, a_col


@njit(cache=True)
def intercept_heading(t, o):
    """Find the heading a target needs to meet own vessel.

    Args:
        t (ndarray): Target row (x, y, v, heading, turn rate)
        o (ndarray): Own row

    Returns:
        float: Heading, rad
    """
    dx, dy = o[X] - t[X], o[Y] - t[Y]
    d = py_hypot(dx, dy)
    if d < 1e-9:
        return t[TH]
    rx, ry = dx / d, dy / d
    ox, oy = o[V] * c_cos(o[TH]), o[V] * c_sin(o[TH])
    b = ox * rx + oy * ry
    disc = b * b - (c_pow(o[V], 2.0) - c_pow(t[V], 2.0))
    if disc < 0.0:
        return math.atan2(dy, dx)
    c = -b + math.sqrt(disc)
    if c <= 0.0:
        return math.atan2(dy, dx)
    return math.atan2(oy + c * ry, ox + c * rx)


@njit(cache=True)
def cpa_obs(o, t):
    """Compute TCPA and DCPA for the observation.

    Args:
        o (ndarray): Own row
        t (ndarray): Target row (x, y, v, heading, turn rate)

    Returns:
        tuple: TCPA (s) and DCPA (m)
    """
    vox, voy = o[V] * c_cos(o[TH]), o[V] * c_sin(o[TH])
    vtx, vty = t[V] * c_cos(t[TH]), t[V] * c_sin(t[TH])
    vrx, vry = vtx - vox, vty - voy
    rx, ry = t[X] - o[X], t[Y] - o[Y]
    vrel = c_hypot(vrx, vry)
    R = c_hypot(rx, ry)
    if vrel < 1e-9:
        return math.inf, R
    cos_a = (vrx * -rx + vry * -ry) / (vrel * R)
    cos_a = min(max(cos_a, -1.0), 1.0)
    dcpa = R * math.sqrt(1.0 - c_pow(cos_a, 2.0))
    tcpa = R * cos_a / vrel
    return tcpa, dcpa


@njit(cache=True)
def observe_full(
    o,
    ts,
    mid,
    use_gate,
    gate,
    relaxed,
    mem,
    n_track,
    elapsed_frac,
    flip,
    speed_max,
    range_norm,
    tcpa_norm,
    T,
    kappa_max,
    delta_min,
    h,
    S,
):
    """Build the full policy observation vector.

    Args:
        o (ndarray): Own row (x, y, v, heading, turn rate)
        ts (ndarray): Target rows, same layout
        mid (ndarray): Centre of each heading bin, rad
        use_gate (bool): Whether to apply the DCPA gate
        gate (float): DCPA gate, m
        relaxed (ndarray): Relaxation flag per target
        mem (ndarray): Stored role code per target
        n_track (int): Number of tracked targets
        elapsed_frac (float): Share of the episode used
        flip (bool): Whether to mirror port and starboard
        speed_max (float): Speed used to normalise, m/s
        range_norm (float): Range used to normalise, m
        tcpa_norm (float): TCPA used to normalise, s
        T (float): Look ahead time, s
        kappa_max (float): Max heading change, rad
        delta_min (float): Smallest lawful turn, rad
        h (float): Hold band, rad
        S (float): Safety radius, m

    Returns:
        tuple: Observation, physical width, compliant width and reach
    """
    n_bins = mid.shape[0]
    lawful_out = np.empty((MAX_IV, 2))
    blocked, n_iv, a_phys, a_col = observe_core(
        o,
        ts,
        mid,
        use_gate,
        gate,
        relaxed,
        mem,
        T,
        kappa_max,
        delta_min,
        h,
        S,
        lawful_out,
    )
    k0 = kappa(o, T, kappa_max)
    obs = np.zeros(2 * n_bins + 6 * n_track + 5, dtype=np.float32)
    obs[:n_bins] = blocked
    for b in range(n_bins):
        c = np.float32(0.0)
        for i in range(n_iv):
            if mid[b] >= lawful_out[i, 0] and mid[b] <= lawful_out[i, 1]:
                c += np.float32(1.0)
        obs[n_bins + b] = c

    n = ts.shape[0]
    order = np.arange(n)
    rng = np.empty(n)
    for i in range(n):
        rng[i] = py_hypot(ts[i, X] - o[X], ts[i, Y] - o[Y])
    for i in range(1, n):
        j = i
        while j > 0 and rng[order[j]] < rng[order[j - 1]]:
            order[j], order[j - 1] = order[j - 1], order[j]
            j -= 1
    base = 2 * n_bins
    for slot in range(min(n, n_track)):
        t = ts[order[slot]]
        dx, dy = t[X] - o[X], t[Y] - o[Y]
        tcpa, dcpa = cpa_obs(o, t)
        k = base + 6 * slot
        obs[k] = py_min(py_hypot(dx, dy) / range_norm, 1.0)
        obs[k + 1] = ea_wrap(math.atan2(dy, dx) - o[TH]) / math.pi
        obs[k + 2] = ea_wrap(t[TH] - o[TH]) / math.pi
        obs[k + 3] = t[V] / speed_max
        obs[k + 4] = py_min(dcpa / range_norm, 1.0)
        obs[k + 5] = (
            min(max(tcpa / tcpa_norm, -1.0), 1.0) if math.isfinite(tcpa) else 1.0
        )

    k = base + 6 * n_track
    obs[k] = o[V] / speed_max
    obs[k + 1] = a_col / k0 if k0 else 0.0
    obs[k + 2] = a_phys / k0 if k0 else 0.0
    obs[k + 3] = elapsed_frac
    obs[k + 4] = flip
    for i in range(obs.shape[0]):
        obs[i] = min(max(obs[i], np.float32(-1.0)), np.float32(1.0))
    return obs, a_phys, a_col, k0


def _selfcheck():
    """Compare the numba kernels with the Python versions on random cases."""
    import random

    from . import alpha as _alpha
    from . import monitor as _monitor
    from .escape_angle import Vessel

    rng = random.Random(11)
    S, T = _duty.SAFETY_RADIUS_M, _duty.HORIZON_S
    n_cases = 3000
    for case in range(n_cases):
        os_v = Vessel(
            "OS",
            rng.uniform(-3000, 3000),
            rng.uniform(-3000, 3000),
            rng.uniform(2, 10),
            rng.uniform(-math.pi, math.pi),
        )
        ts_v = [
            Vessel(
                f"T{j}",
                rng.uniform(-6000, 6000),
                rng.uniform(-6000, 6000),
                rng.uniform(0, 12),
                rng.uniform(-math.pi, math.pi),
            )
            for j in range(rng.choice((1, 2, 3)))
        ]
        gate = rng.choice((None, 500.0, 3000.0))
        relaxed = [rng.random() < 0.3 for _ in ts_v]
        memory = [rng.choice((None, None, "OGW", "BO")) for _ in ts_v]
        o, ts = to_rows(os_v), to_rows_list(ts_v)
        use_gate, g = (False, 0.0) if gate is None else (True, gate)
        rel = np.array(relaxed, dtype=np.int8)
        mem = memory_codes(memory, len(ts_v))
        commit = np.empty((1, 2))

        ref = _monitor.step_certificate(
            os_v, ts_v, gate=gate, relaxed=relaxed, memory=memory
        )
        got = step_cert_now(
            o,
            ts,
            commit,
            0,
            use_gate,
            g,
            rel,
            mem,
            T,
            _duty.KAPPA_MAX_RAD,
            _duty.DELTA_MIN_RAD,
            _duty.HOLD_BAND_RAD,
            S,
        )
        assert got[0] == ref["deontic"] and got[1] == ref["geometric"], case
        assert got[2] == ref["alpha_col"] and got[3] == ref["alpha_phys"], case
        assert got[4] == ref["lawful_width"] and got[5] == ref["kappa"], case
        amb = [(-0.4, 0.3)]
        geo = np.empty((len(ts_v), N_GEO))
        trig_fill(ts, geo)
        geo_fill(o, ts, S, geo)
        buf = np.empty(2 + 8 * len(ts_v))
        got_a = alpha(o, ts, geo, np.array(amb), 1, S, T, o[V] ** 2.0, buf, False)
        assert got_a == _alpha.alpha(os_v, ts_v, amb, S, T)
    print(f"_monitor_nb selfcheck OK ({n_cases} states)")


_ROLE_CODE = {"OGW": R_OGW, "BO": R_BO}


def memory_codes(memory, n):
    """Turn stored roles into role codes.

    Args:
        memory (list): Stored role per target, or None
        n (int): Number of targets

    Returns:
        ndarray: Role code per target
    """
    out = np.zeros(n, np.int64)
    for i, m in enumerate(memory or ()):
        if i < n:
            out[i] = _ROLE_CODE.get(m, R_NONE)
    return out


_ROW_ONE = np.empty(5)
_ROW_N = {}


def to_rows(v):
    """Copy a Vessel into a reusable row.

    Args:
        v (Vessel): Vessel

    Returns:
        ndarray: Row (x, y, v, heading, turn rate)
    """
    o = _ROW_ONE
    o[0], o[1], o[2], o[3], o[4] = v.x, v.y, v.v, v.theta, v.rate_of_turn
    return o


def to_rows_list(vs):
    """Copy a list of Vessels into a reusable array.

    Args:
        vs (list): Vessels

    Returns:
        ndarray: One row per vessel
    """
    n = len(vs)
    rows = _ROW_N.get(n)
    if rows is None:
        rows = np.empty((n, 5))
        _ROW_N[n] = rows
    for i, v in enumerate(vs):
        rows[i, 0], rows[i, 1], rows[i, 2] = v.x, v.y, v.v
        rows[i, 3], rows[i, 4] = v.theta, v.rate_of_turn
    return rows


if __name__ == "__main__":
    _selfcheck()

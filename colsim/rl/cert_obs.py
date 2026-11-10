from __future__ import annotations

import functools
import math

import numpy as np

from ..certificate import alpha as _alpha
from ..certificate import behaviour as _behaviour
from ..certificate import duty as _duty
from ..certificate import monitor as _monitor
from ..certificate import tube as _tube
from ..certificate.escape_angle import HULL_CONTACT_M, R_MIN_M, _wrap, cpa
from ..simulator.experiment_constants import (
    SIM_DT_S,
    SPEED_MAX_MPS,
    SPEED_MIN_MPS,
    SURGE_ACCEL_MAX_MPS2,
    SURGE_DECEL_MAX_MPS2,
)

N_BINS = 32

N_TRACK = 2

RELEASE_LOOKAHEAD = 0

VETO_MARGIN_M = HULL_CONTACT_M

ACTION_SHIELD = 0

SHIELD_GRID = 21

SHIELD_MARGIN_M = 250.0

MARGIN_STEPS = 0

MARGIN_NORM_M = 500.0

ALPHA_COL_RELEASE = 0.0

MAX_STEPS = 60

RANGE_NORM_M = 8000.0

TCPA_NORM_S = 300.0

def seg_min(p0, p1):
    """Find the closest distance between own track and every target track, between steps too.

    Args:
        p0 (list): Own positions per step
        p1 (list): Target positions per step, own first

    Returns:
        float: Closest distance, m
    """
    ox0, oy0 = p0[0]
    ox1, oy1 = p1[0]
    best = float("inf")
    for (tx0, ty0), (tx1, ty1) in zip(p0[1:], p1[1:]):
        best = min(best, _seg_pair(ox0, oy0, ox1, oy1, tx0, ty0, tx1, ty1))
    return best

def _seg_pair(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1):
    """Compute the closest distance between two moving points over one step.

    Args:
        ax0 (float): First point x at start, m
        ay0 (float): First point y at start, m
        ax1 (float): First point x at end, m
        ay1 (float): First point y at end, m
        bx0 (float): Second point x at start, m
        by0 (float): Second point y at start, m
        bx1 (float): Second point x at end, m
        by1 (float): Second point y at end, m

    Returns:
        float: Closest distance, m
    """
    rx0, ry0 = ax0 - bx0, ay0 - by0
    rx1, ry1 = ax1 - bx1, ay1 - by1
    dx, dy = rx1 - rx0, ry1 - ry0
    dd = dx * dx + dy * dy
    t = 0.0 if dd == 0.0 else max(0.0, min(1.0, -(rx0 * dx + ry0 * dy) / dd))
    return math.hypot(rx0 + t * dx, ry0 + t * dy)

def rate_margin(os_b, threats, yaw, n, dt, tubes=None):
    """Find the smallest separation if own vessel holds a turn rate for n steps.

    Args:
        os_b (Vessel): Own vessel
        threats (list): Target vessels
        yaw (float): Turn rate, rad/s
        n (int): Number of steps
        dt (float): Time step, s
        tubes (list): Target reachable tubes, or None

    Returns:
        float: Smallest separation, m
    """
    x, y, th = os_b.x, os_b.y, os_b.theta
    others = [(b.x, b.y, b.v * math.cos(b.theta), b.v * math.sin(b.theta)) for b in threats]
    worst = float("inf")
    for k in range(1, int(n) + 1):
        x0, y0 = x, y
        th += yaw * dt
        x += os_b.v * math.cos(th) * dt
        y += os_b.v * math.sin(th) * dt
        t0, t1 = (k - 1) * dt, k * dt
        if tubes is not None:
            for p in ((x0, y0), (x, y)):
                for ts, tb in zip(threats, tubes):
                    worst = min(worst, _tube.clearance(p, tb, ts, t1))
            continue
        p0 = [(x0, y0)] + [(bx + vx * t0, by + vy * t0) for bx, by, vx, vy in others]
        p1 = [(x, y)] + [(bx + vx * t1, by + vy * t1) for bx, by, vx, vy in others]
        worst = min(worst, seg_min(p0, p1))
    return worst

def shielded_yaw(os_b, threats, yaw, rate, dt, *, fire_n, margin, pick_n=None,
                 grid_n=SHIELD_GRID, tubes=None):
    """Keep the turn rate if its margin is enough, otherwise pick the safest rate on a grid.

    Args:
        os_b (Vessel): Own vessel
        threats (list): Target vessels
        yaw (float): Proposed turn rate, rad/s
        rate (float): Max turn rate, rad/s
        dt (float): Time step, s
        fire_n (int): Steps used to test the proposed rate
        margin (float): Separation needed, m
        pick_n (int): Steps used to rank the grid, or None
        grid_n (int): Number of grid rates
        tubes (list): Target reachable tubes, or None

    Returns:
        tuple: Turn rate and whether it was replaced
    """
    if rate_margin(os_b, threats, yaw, fire_n, dt, tubes) >= margin:
        return yaw, False
    n_pick = fire_n if pick_n is None else pick_n
    grid = np.linspace(-rate, rate, grid_n)
    best = max(grid, key=lambda c: rate_margin(os_b, threats, float(c), n_pick, dt, tubes))
    return float(best), True

def margin_channel(os_v, threats, mid, *, steps, dt=SIM_DT_S, r_min=R_MIN_M):
    """Compute the separation margin of each heading bin, scaled to [0, 1].

    Args:
        os_v (Vessel): Own vessel
        threats (list): Target vessels
        mid (ndarray): Bin centre heading changes, rad
        steps (int): Number of steps
        dt (float): Time step, s
        r_min (float): Min turn radius, m

    Returns:
        ndarray: Margin per bin
    """
    rate = float(os_v.v) / float(r_min)
    out = np.empty(len(mid), dtype=np.float32)
    for i, d in enumerate(mid):
        yaw = float(np.clip(float(d) / _duty.HORIZON_S, -rate, rate))
        m = rate_margin(os_v, threats, yaw, steps, dt)
        out[i] = 1.0 if not math.isfinite(m) else m / (m + MARGIN_NORM_M)
    return np.clip(out, 0.0, 1.0)

def _bin_fields(os_v, threats, mid, S=_duty.SAFETY_RADIUS_M, T=_duty.HORIZON_S, tubes=None,
                n_t=16):
    """Compute the room, contact time and load of each heading bin.

    Args:
        os_v (Vessel): Own vessel
        threats (list): Target vessels
        mid (ndarray): Bin centre heading changes, rad
        S (float): Safety radius, m
        T (float): Look ahead time, s
        tubes (list): Target reachable tubes, or None
        n_t (int): Number of time samples

    Returns:
        tuple: Room, time and load arrays
    """
    n = len(mid)
    if not threats:
        return (np.ones(n, np.float32), np.ones(n, np.float32), np.zeros(n, np.float32))
    if tubes is not None:
        return _bin_fields_tube(os_v, threats, mid, tubes, S=S, T=T, n_t=n_t)
    room = np.full(n, np.inf)
    when = np.ones(n)
    load = np.zeros(n)
    a = os_v.theta + mid
    ovx, ovy = os_v.v * np.cos(a), os_v.v * np.sin(a)
    for ts in threats:
        dx, dy = ts.x - os_v.x, ts.y - os_v.y
        wx = ts.v * math.cos(ts.theta) - ovx
        wy = ts.v * math.sin(ts.theta) - ovy
        ww = wx * wx + wy * wy
        t = np.where(ww > 1e-24, -(dx * wx + dy * wy) / np.maximum(ww, 1e-24), 0.0)
        t = np.clip(t, 0.0, T)
        m = np.hypot(dx + t * wx, dy + t * wy)
        hit = m < S
        room = np.minimum(room, m)
        when = np.where(hit, np.minimum(when, t / T), when)
        load += hit
    return (
        (room / (room + S)).astype(np.float32),
        when.astype(np.float32),
        (load / len(threats)).astype(np.float32),
    )

def _lawful_signed(mid, lawful_iv, k0):
    """Compute the signed distance of each bin to the lawful intervals, scaled to [-1, 1].

    Args:
        mid (ndarray): Bin centre heading changes, rad
        lawful_iv (list): Lawful heading intervals
        k0 (float): Max heading change, rad

    Returns:
        ndarray: Signed distance per bin
    """
    if not lawful_iv:
        return np.full(len(mid), -1.0, dtype=np.float32)
    d = np.full(len(mid), -np.inf)
    for lo, hi in lawful_iv:
        d = np.maximum(d, np.minimum(mid - lo, hi - mid))
    return np.clip(d / k0 if k0 else d * 0.0, -1.0, 1.0).astype(np.float32)

def _bin_fields_tube(os_v, threats, mid, tubes, *, S=_duty.SAFETY_RADIUS_M, T=_duty.HORIZON_S,
                     n_t=16):
    """Same as _bin_fields, with reachable tubes for the targets.

    Args:
        os_v (Vessel): Own vessel
        threats (list): Target vessels
        mid (ndarray): Bin centre heading changes, rad
        tubes (list): Target reachable tubes
        S (float): Safety radius, m
        T (float): Look ahead time, s
        n_t (int): Number of time samples

    Returns:
        tuple: Room, time and load arrays
    """
    n = len(mid)
    room = np.full(n, np.inf)
    when = np.ones(n)
    load = np.zeros(n)
    ts_grid = np.linspace(0.0, T, int(max(n_t, 2)))
    a = os_v.theta + mid
    ovx, ovy = os_v.v * np.cos(a), os_v.v * np.sin(a)
    for ts, tb in zip(threats, tubes):
        m = np.full(n, np.inf)
        at = np.ones(n)
        for t in ts_grid:
            px, py = os_v.x + ovx * t, os_v.y + ovy * t
            d = np.array([_tube.clearance((float(x), float(y)), tb, ts, float(t))
                          for x, y in zip(px, py)])
            closer = d < m
            at = np.where(closer, t / T, at)
            m = np.minimum(m, d)
        hit = m < S
        room = np.minimum(room, m)
        when = np.where(hit, np.minimum(when, at), when)
        load += hit
    return (
        np.asarray(room / (room + S), dtype=np.float32),
        np.asarray(when, dtype=np.float32),
        np.asarray(load / len(threats), dtype=np.float32),
    )

def _nb_observe_core(os_v, threats, mid, gate, relaxed, memory=None):
    """Compute the blocked bins and lawful intervals with the numba kernel.

    Args:
        os_v (Vessel): Own vessel
        threats (list): Target vessels
        mid (ndarray): Bin centre heading changes, rad
        gate (float): Duty gate, m, or None
        relaxed (list): Rule 17 relaxation flags
        memory (list): Stored roles

    Returns:
        tuple: Blocked bins, lawful intervals, physical and lawful widths
    """
    nb = _monitor._nb
    lawful_out = np.empty((nb.MAX_IV, 2))
    blocked, n, a_phys, a_col = nb.observe_core(
        nb.to_rows(os_v), nb.to_rows_list(threats), np.ascontiguousarray(mid, dtype=float),
        gate is not None, 0.0 if gate is None else float(gate),
        _monitor._nb_flags(relaxed, len(threats)), nb.memory_codes(memory, len(threats)),
        _duty.HORIZON_S, _duty.KAPPA_MAX_RAD, _duty.DELTA_MIN_RAD, _duty.HOLD_BAND_RAD,
        _duty.SAFETY_RADIUS_M, lawful_out,
    )
    lawful_iv = [(float(lo), float(hi)) for lo, hi in lawful_out[:n]]
    return blocked, lawful_iv, float(a_phys), float(a_col)

@functools.lru_cache(maxsize=64)
def _bin_mids(k0, n_bins):
    """Get the centre heading change of each bin, cached read only.

    Args:
        k0 (float): Max heading change, rad
        n_bins (int): Number of bins

    Returns:
        ndarray: Bin centres, rad
    """
    edges = np.linspace(-k0, k0, n_bins + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    mid.setflags(write=False)
    return mid

def observe(os_v, threats, *, gate, n_bins=N_BINS, elapsed_frac=0.0, relaxed=None,
            flip_engaged=0.0, margin_steps=0, dt=SIM_DT_S, r_min=R_MIN_M, n_track=N_TRACK,
            bin_fields=False, tubes=None, posterior=None, memory=None):
    """Build the policy observation from the certificate quantities.

    Args:
        os_v (Vessel): Own vessel
        threats (list): Target vessels
        gate (float): Duty gate, m, or None
        n_bins (int): Number of heading bins
        elapsed_frac (float): Share of the episode used
        relaxed (list): Rule 17 relaxation flags
        flip_engaged (bool): Whether a role flip is engaged
        margin_steps (int): Steps of the margin channel, 0 for none
        dt (float): Time step, s
        r_min (float): Min turn radius, m
        n_track (int): Number of tracked targets
        bin_fields (bool): Whether to add the per bin fields
        tubes (list): Target reachable tubes, or None
        posterior (ndarray): Behaviour class probabilities, or None
        memory (list): Stored roles

    Returns:
        tuple: Observation, physical width, lawful width and max heading change
    """
    if (_monitor._nb is not None and not bin_fields and margin_steps <= 0
            and posterior is None):
        nb = _monitor._nb
        obs, a_phys, a_col, k0 = nb.observe_full(
            nb.to_rows(os_v), nb.to_rows_list(threats), _bin_mids(_duty.kappa(os_v), int(n_bins)),
            gate is not None, 0.0 if gate is None else float(gate),
            _monitor._nb_flags(relaxed, len(threats)), nb.memory_codes(memory, len(threats)),
            int(n_track), float(elapsed_frac),
            float(flip_engaged), float(SPEED_MAX_MPS), float(RANGE_NORM_M), float(TCPA_NORM_S),
            _duty.HORIZON_S, _duty.KAPPA_MAX_RAD, _duty.DELTA_MIN_RAD, _duty.HOLD_BAND_RAD,
            _duty.SAFETY_RADIUS_M,
        )
        return obs, float(a_phys), float(a_col), float(k0)

    k0 = _duty.kappa(os_v)
    edges = np.linspace(-k0, k0, n_bins + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    fast = _monitor._nb is not None and not bin_fields
    if fast:
        blocked, lawful_iv, a_phys, a_col = _nb_observe_core(os_v, threats, mid, gate, relaxed,
                                                              memory)
    else:
        lawful_iv = _duty.lawful_intervals(os_v, threats, gate=gate, relaxed=relaxed,
                                           memory=memory)
    raw = []
    post = None
    if bin_fields:
        room, when, load = _bin_fields(os_v, threats, mid, tubes=tubes)
        bins = [room, when, load, _lawful_signed(mid, lawful_iv, k0)]
    else:
        if not fast:
            blocked = np.array(
                [any(_alpha.forbidden(float(d), os_v, ts) for ts in threats) for d in mid],
                dtype=np.float32,
            )
        lawful = np.zeros(n_bins, dtype=np.float32)
        for lo, hi in lawful_iv:
            lawful += ((mid >= lo) & (mid <= hi)).astype(np.float32)
        bins = [blocked, np.clip(lawful, 0.0, 1.0)]

        order = sorted(
            range(len(threats)),
            key=lambda i: math.hypot(threats[i].x - os_v.x, threats[i].y - os_v.y),
        )
        for i in order[:int(n_track)]:
            t = threats[i]
            dx, dy = t.x - os_v.x, t.y - os_v.y
            tcpa, dcpa, _ = cpa(os_v, t)
            raw += [
                min(math.hypot(dx, dy) / RANGE_NORM_M, 1.0),
                float(_wrap(math.atan2(dy, dx) - os_v.theta)) / math.pi,
                float(_wrap(t.theta - os_v.theta)) / math.pi,
                t.v / SPEED_MAX_MPS,
                min(dcpa / RANGE_NORM_M, 1.0),
                float(np.clip(tcpa / TCPA_NORM_S, -1.0, 1.0)) if np.isfinite(tcpa) else 1.0,
            ]
        raw += [0.0] * (6 * int(n_track) - len(raw))
        if posterior is not None:
            p = np.asarray(posterior, dtype=np.float32).reshape(-1, len(_behaviour.CLASSES))
            post = [p[i] for i in order[:int(n_track)]]
            post += [np.full(len(_behaviour.CLASSES), 1.0 / len(_behaviour.CLASSES), np.float32)
                     ] * (int(n_track) - len(post))

    if not fast:
        a_phys = _alpha.alpha(os_v, threats, [(-k0, k0)])
        a_col = _alpha.alpha(os_v, threats, lawful_iv) if lawful_iv else 0.0
    tail = [
        os_v.v / SPEED_MAX_MPS,
        a_col / k0 if k0 else 0.0,
        a_phys / k0 if k0 else 0.0,
        float(elapsed_frac),
        float(flip_engaged),
    ]
    blocks = list(bins)
    if bin_fields:
        blocks.append(np.asarray(tail, np.float32))
    elif margin_steps > 0:
        blocks.append(np.asarray(raw, np.float32))
        blocks.append(np.asarray(tail, np.float32))
        blocks.append(margin_channel(os_v, threats, mid, steps=margin_steps, dt=dt, r_min=r_min))
    else:
        blocks.append(np.asarray(raw, np.float32))
        blocks.append(np.asarray(tail, np.float32))
    if post is not None:
        blocks.append(np.concatenate(post))
    obs = np.concatenate(blocks).astype(np.float32)
    return np.clip(obs, -1.0, 1.0), a_phys, a_col, k0

def decode_action(action, *, v, dt, yaw_max, speed_action=True):
    """Turn a policy action into acceleration and turn rate.

    Args:
        action (ndarray): Policy action in [-1, 1]
        v (float): Current speed, m/s
        dt (float): Time step, s
        yaw_max (float): Max turn rate, rad/s
        speed_action (bool): Whether the policy also sets speed

    Returns:
        tuple: Acceleration (m/s2) and turn rate (rad/s)
    """
    a = np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0)
    yaw = float(a[0]) * float(yaw_max)
    v_des = (SPEED_MIN_MPS + 0.5 * (float(a[1]) + 1.0) * (SPEED_MAX_MPS - SPEED_MIN_MPS)
             if speed_action else float(v))
    acc = float(
        np.clip(
            (v_des - float(v)) / float(dt), SURGE_DECEL_MAX_MPS2, SURGE_ACCEL_MAX_MPS2
        )
    )
    return acc, yaw

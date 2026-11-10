"""`SeaObject._steer_to_point` and the integration in `update, numba version"""

import math

import numpy as np
from numba import njit

from ..certificate import _monitor_nb as _nb
from ..certificate import duty as _duty
from ..certificate import geometry as _geometry
from .._nb_cache import drop_if_stale

drop_if_stale(__file__, [_nb.__file__, _duty.__file__, _geometry.__file__])

X, Y, V, TH = _nb.X, _nb.Y, _nb.V, _nb.TH
c_sin, c_cos = _nb.c_sin, _nb.c_cos

QUARTER_PI = float(np.pi / 4)
DELTA_MIN = float(_duty.DELTA_MIN_RAD)
HORIZON = float(_duty.HORIZON_S)
HOLD_BAND = float(_duty.HOLD_BAND_RAD)

F_PORT_VETO, F_GIVEWAY_FLOOR, F_STANDON_SNAP = 1, 2, 4


@njit(cache=True)
def command(own, rows, use_roles, gate, tx, ty, dt, cap_on, cap, out, mem):
    """Steer to a waypoint with the COLREGs veto, floor and snap, then integrate one step.

    Args:
        own (ndarray): Own state (x, y, v, heading, extra)
        rows (ndarray): Target states, one per row
        use_roles (bool): Whether to apply the COLREGs roles
        gate (float): Distance under which a duty applies, m
        tx (float): Waypoint x, m
        ty (float): Waypoint y, m
        dt (float): Time step, s
        cap_on (bool): Whether to limit the turn rate
        cap (float): Max turn rate, rad/s
        out (ndarray): Filled with the new (x, y, v, heading)
        mem (ndarray): Role memory per target

    Returns:
        int: Flags of the rules that changed the command
    """
    thetabar = math.atan2(ty - own[Y], tx - own[X])
    d_th = thetabar - own[TH]
    error = math.atan2(c_sin(d_th), c_cos(d_th))

    no_port = False
    n_give_way = 0
    all_ogw = True
    stand_on = False
    if use_roles:
        for j in range(rows.shape[0]):
            c = _nb.role_mem(own, rows[j], True, gate, mem[j])
            if c == _nb.R_HO or c == _nb.R_CGW or c == _nb.R_CSO:
                no_port = True
            if c == _nb.R_OGW or c == _nb.R_HO or c == _nb.R_CGW:
                n_give_way += 1
                if c != _nb.R_OGW:
                    all_ogw = False
            if c == _nb.R_BO or c == _nb.R_CSO:
                stand_on = True

    flags = 0
    if error > 0.0 and no_port:
        error = 0.0
        flags |= F_PORT_VETO

    u2 = min(max(0.2 / dt * error, -QUARTER_PI), QUARTER_PI)

    if n_give_way > 0:
        magnitude = DELTA_MIN / HORIZON
        port_allowed = all_ogw and u2 > 0.0
        floor = magnitude if port_allowed else -magnitude
        if (port_allowed and u2 < floor) or ((not port_allowed) and u2 > floor):
            u2 = floor
            flags |= F_GIVEWAY_FLOOR
    elif stand_on:
        want = -u2 * HORIZON
        if HOLD_BAND < want < DELTA_MIN:
            midpoint = 0.5 * (HOLD_BAND + DELTA_MIN)
            edge = DELTA_MIN if want > midpoint else HOLD_BAND
            u2 = -edge / HORIZON
            flags |= F_STANDON_SNAP

    v, th = own[V], own[TH]
    x = own[X] + dt * v * c_cos(th)
    y = own[Y] + dt * v * c_sin(th)
    yaw = u2
    if cap_on:
        yaw = min(max(yaw, -cap), cap)
    th2 = th + dt * yaw
    out[0], out[1], out[2] = x, y, v
    out[3] = math.atan2(c_sin(th2), c_cos(th2))
    return flags


@njit(cache=True)
def command_reference(own, rows, use_roles, gate, tx, ty, dt, cap_on, cap, out):
    """Same as command, without role memory, used as reference.

    Args:
        own (ndarray): Own state (x, y, v, heading, extra)
        rows (ndarray): Target states, one per row
        use_roles (bool): Whether to apply the COLREGs roles
        gate (float): Distance under which a duty applies, m
        tx (float): Waypoint x, m
        ty (float): Waypoint y, m
        dt (float): Time step, s
        cap_on (bool): Whether to limit the turn rate
        cap (float): Max turn rate, rad/s
        out (ndarray): Filled with the new (x, y, v, heading)

    Returns:
        int: Flags of the rules that changed the command
    """
    thetabar = math.atan2(ty - own[Y], tx - own[X])
    d_th = thetabar - own[TH]
    error = math.atan2(c_sin(d_th), c_cos(d_th))

    no_port = False
    n_give_way = 0
    all_ogw = True
    stand_on = False
    if use_roles:
        for j in range(rows.shape[0]):
            c = _nb.role(own, rows[j], True, gate)
            if c == _nb.R_HO or c == _nb.R_CGW or c == _nb.R_CSO:
                no_port = True
            if c == _nb.R_OGW or c == _nb.R_HO or c == _nb.R_CGW:
                n_give_way += 1
                if c != _nb.R_OGW:
                    all_ogw = False
            if c == _nb.R_BO or c == _nb.R_CSO:
                stand_on = True

    flags = 0
    if error > 0.0 and no_port:
        error = 0.0
        flags |= F_PORT_VETO

    u2 = min(max(0.2 / dt * error, -QUARTER_PI), QUARTER_PI)

    if n_give_way > 0:
        magnitude = DELTA_MIN / HORIZON
        port_allowed = all_ogw and u2 > 0.0
        floor = magnitude if port_allowed else -magnitude
        if (port_allowed and u2 < floor) or ((not port_allowed) and u2 > floor):
            u2 = floor
            flags |= F_GIVEWAY_FLOOR
    elif stand_on:
        want = -u2 * HORIZON
        if HOLD_BAND < want < DELTA_MIN:
            midpoint = 0.5 * (HOLD_BAND + DELTA_MIN)
            edge = DELTA_MIN if want > midpoint else HOLD_BAND
            u2 = -edge / HORIZON
            flags |= F_STANDON_SNAP

    v, th = own[V], own[TH]
    x = own[X] + dt * v * c_cos(th)
    y = own[Y] + dt * v * c_sin(th)
    yaw = u2
    if cap_on:
        yaw = min(max(yaw, -cap), cap)
    th2 = th + dt * yaw
    out[0], out[1], out[2] = x, y, v
    out[3] = math.atan2(c_sin(th2), c_cos(th2))
    return flags


def _selfcheck(n=20000, seed=0):
    """Compare the numba kernel with a Python version on random states.

    Args:
        n (int): Number of random states
        seed (int): Random seed
    """
    import random

    rnd = random.Random(seed)
    own = np.empty(5)
    rows = np.empty((2, 5))
    out = np.empty(4)
    bad = 0
    for _ in range(n):
        own[0], own[1] = rnd.uniform(-5e3, 5e3), rnd.uniform(-5e3, 5e3)
        own[2], own[3] = rnd.uniform(0.0, 12.0), rnd.uniform(-math.pi, math.pi)
        own[4] = 0.0
        for j in range(2):
            rows[j, 0], rows[j, 1] = rnd.uniform(-5e3, 5e3), rnd.uniform(-5e3, 5e3)
            rows[j, 2], rows[j, 3] = rnd.uniform(0.0, 12.0), rnd.uniform(
                -math.pi, math.pi
            )
            rows[j, 4] = 0.0
        tx, ty = rnd.uniform(-5e3, 5e3), rnd.uniform(-5e3, 5e3)
        dt = 10.0
        flags = command(
            own, rows, True, 500.0, tx, ty, dt, False, 0.0, out, np.zeros(2, np.int64)
        )

        codes = np.empty(2, np.int64)
        _nb.roles_of(own, rows, True, 500.0, codes)
        roles = [_nb.ROLE_NAMES[c] for c in codes]
        thetabar = math.atan2(ty - own[1], tx - own[0])
        d_th = thetabar - own[3]
        error = math.atan2(math.sin(d_th), math.cos(d_th))
        py_flags = 0
        if error > 0.0 and any(r in ("HO", "CGW", "CSO") for r in roles):
            error = 0.0
            py_flags |= F_PORT_VETO
        u2 = min(max(0.2 / dt * error, -QUARTER_PI), QUARTER_PI)
        give_way = [r for r in roles if r in _duty.GIVE_WAY]
        if give_way:
            magnitude = DELTA_MIN / HORIZON
            port_allowed = all(r == "OGW" for r in give_way) and u2 > 0.0
            floor = magnitude if port_allowed else -magnitude
            if (port_allowed and u2 < floor) or (not port_allowed and u2 > floor):
                u2 = floor
                py_flags |= F_GIVEWAY_FLOOR
        elif any(r in _duty.STAND_ON for r in roles):
            want = -u2 * HORIZON
            if HOLD_BAND < want < DELTA_MIN:
                midpoint = 0.5 * (HOLD_BAND + DELTA_MIN)
                edge = DELTA_MIN if want > midpoint else HOLD_BAND
                u2 = -edge / HORIZON
                py_flags |= F_STANDON_SNAP
        x = own[0] + dt * own[2] * math.cos(own[3])
        y = own[1] + dt * own[2] * math.sin(own[3])
        th2 = own[3] + dt * u2
        th2 = math.atan2(math.sin(th2), math.cos(th2))
        if (
            flags != py_flags
            or out[0] != x
            or out[1] != y
            or out[2] != own[2]
            or out[3] != th2
        ):
            bad += 1
    if bad:
        raise AssertionError(f"_steer_nb selfcheck {bad} / {n} mismatches")
    print(f"_steer_nb selfcheck OK ({n} states)")


if __name__ == "__main__":
    _selfcheck()

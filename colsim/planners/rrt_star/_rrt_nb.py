"""Numba port of the local-avoidance RRT*"""

from __future__ import annotations

import array
import ctypes
import math
import random

import llvmlite.binding as llvm
import numpy as np
from numba import njit, types

from ...certificate import _monitor_nb as _monitor_nb_module
from ...certificate._monitor_nb import c_cos, c_pow, c_sin
from ..._nb_cache import drop_if_stale

drop_if_stale(__file__, [_monitor_nb_module.__file__])

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


c_hypot = _extern("hypot", 2)
c_atan2 = _extern("atan2", 2)
c_log = _extern("log", 1)
c_fmod = _extern("fmod", 2)
c_floor = _extern("floor", 1)
c_copysign = _extern("copysign", 2)

NEED_SITUATIONS = -1
GOAL_TRAPPED = 0
DIRECT = 1
FOUND = 2
FAILED = 3

SIT_OTHER, SIT_HEAD_ON, SIT_CROSS_GIVE_WAY, SIT_OVERTAKING, SIT_STAND_ON = 0, 1, 2, 3, 4
SIT_CODE = {
    "Head_on": SIT_HEAD_ON,
    "Crossing_give_way": SIT_CROSS_GIVE_WAY,
    "Overtaking_give_way": SIT_OVERTAKING,
    "Crossing_stand_on": SIT_STAND_ON,
    "Being_overtaken_stand_on": SIT_STAND_ON,
}

TURN_TOLERANCE_RAD = 0.05
NEIGHBOUR_SLACK = 1e-9
GOAL_BIAS = 0.1
SAMPLE_TRIES = 15
GOAL_WALK_STEP_M = 200.0
SEGMENT_CHECK_M = 50
TRACE_LIMIT = 5000
PRUNE_COST_SLACK_M = 10.0
ELASTIC_ITERATIONS = 50

MT_N, MT_M = 624, 397
MT_MATRIX_A = np.uint64(0x9908B0DF)
MT_UPPER = np.uint64(0x80000000)
MT_LOWER = np.uint64(0x7FFFFFFF)
MT_MASK32 = np.uint64(0xFFFFFFFF)


@njit(cache=True)
def py_floordiv(vx, wx):
    """Floor division that matches Python bit for bit.

    Args:
        vx (float): Dividend
        wx (float): Divisor

    Returns:
        float: Floor of vx / wx
    """
    mod = c_fmod(vx, wx)
    div = (vx - mod) / wx
    if mod != 0.0:
        if (wx < 0.0) != (mod < 0.0):
            mod += wx
            div -= 1.0
    if div != 0.0:
        floordiv = c_floor(div)
        if div - floordiv > 0.5:
            floordiv += 1.0
        return floordiv
    return c_copysign(0.0, vx / wx)


@njit(cache=True)
def py_mod(vx, wx):
    """Modulo that matches Python bit for bit.

    Args:
        vx (float): Dividend
        wx (float): Divisor

    Returns:
        float: Remainder with the sign of wx
    """
    mod = c_fmod(vx, wx)
    if mod != 0.0:
        if (wx < 0.0) != (mod < 0.0):
            mod += wx
    else:
        mod = c_copysign(0.0, wx)
    return mod


@njit(cache=True)
def mt_uint32(mt, pos):
    """Draw the next 32 bit integer from a Mersenne Twister state.

    Args:
        mt (ndarray): Generator state words
        pos (ndarray): Position in the state, updated

    Returns:
        int: Random integer
    """
    if pos[0] >= MT_N:
        for kk in range(MT_N - MT_M):
            y = (mt[kk] & MT_UPPER) | (mt[kk + 1] & MT_LOWER)
            mt[kk] = (
                mt[kk + MT_M]
                ^ (y >> np.uint64(1))
                ^ (MT_MATRIX_A if y & np.uint64(1) else np.uint64(0))
            )
        for kk in range(MT_N - MT_M, MT_N - 1):
            y = (mt[kk] & MT_UPPER) | (mt[kk + 1] & MT_LOWER)
            mt[kk] = (
                mt[kk + MT_M - MT_N]
                ^ (y >> np.uint64(1))
                ^ (MT_MATRIX_A if y & np.uint64(1) else np.uint64(0))
            )
        y = (mt[MT_N - 1] & MT_UPPER) | (mt[0] & MT_LOWER)
        mt[MT_N - 1] = (
            mt[MT_M - 1]
            ^ (y >> np.uint64(1))
            ^ (MT_MATRIX_A if y & np.uint64(1) else np.uint64(0))
        )
        pos[0] = 0
    y = mt[pos[0]]
    pos[0] += 1
    y ^= y >> np.uint64(11)
    y ^= (y << np.uint64(7)) & np.uint64(0x9D2C5680)
    y ^= (y << np.uint64(15)) & np.uint64(0xEFC60000)
    y ^= y >> np.uint64(18)
    return y & MT_MASK32


@njit(cache=True)
def mt_random(mt, pos):
    """Draw a float in [0, 1) like random.random.

    Args:
        mt (ndarray): Generator state words
        pos (ndarray): Position in the state

    Returns:
        float: Random float
    """
    a = mt_uint32(mt, pos) >> np.uint64(5)
    b = mt_uint32(mt, pos) >> np.uint64(6)
    return (float(a) * 67108864.0 + float(b)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def mt_uniform(mt, pos, a, b):
    """Draw a float in [a, b) like random.uniform.

    Args:
        mt (ndarray): Generator state words
        pos (ndarray): Position in the state
        a (float): Lower bound
        b (float): Upper bound

    Returns:
        float: Random float
    """
    return a + (b - a) * mt_random(mt, pos)


def _mt_view():
    """Get direct access to the state of the Python random module, if safe.

    Returns:
        tuple: Generator, index view and words view, or None
    """
    try:
        n_bytes = 4 + 4 * MT_N
        if random.Random.__basicsize__ < object.__basicsize__ + n_bytes:
            return None
        buf = (ctypes.c_char * n_bytes).from_address(
            id(random._inst) + object.__basicsize__
        )
        raw = np.frombuffer(buf, dtype=np.uint8)
        index, words = raw[:4].view(np.int32), raw[4:].view(np.uint32)
        saved = random.getstate()
        try:
            for seed in (0, 12345):
                random.seed(seed)
                random.random()
                ref = random.getstate()[1]
                if tuple(words.tolist()) != ref[:MT_N] or int(index[0]) != ref[MT_N]:
                    return None
        finally:
            random.setstate(saved)
        return random._inst, index, words
    except Exception:
        return None


_MT_VIEW = _mt_view()


def rng_export():
    """Copy the Python random state into numpy arrays.

    Returns:
        tuple: State words, position and saved Python state
    """
    if _MT_VIEW is not None:
        _inst, index, words = _MT_VIEW
        return words.astype(np.uint64), np.array([index[0]], dtype=np.int64), None
    state = random.getstate()
    words = state[1]
    mt = np.frombuffer(array.array("I", words[:MT_N]), dtype=np.uint32).astype(
        np.uint64
    )
    return mt, np.array([words[MT_N]], dtype=np.int64), state


def rng_import(mt, pos, state):
    """Write the numpy random state back into Python random.

    Args:
        mt (ndarray): State words
        pos (ndarray): Position in the state
        state (tuple): Saved Python state
    """
    if _MT_VIEW is not None:
        _inst, index, words = _MT_VIEW
        words[:] = mt
        index[0] = pos[0]
        return
    random.setstate((state[0], tuple(mt.tolist()) + (int(pos[0]),), state[2]))


@njit(cache=True)
def grid_build(obs, x_min, x_max, y_min, y_max, cell):
    """Build the CSR grid that maps each cell to its obstacles.

    Args:
        obs (ndarray): Obstacle discs (x, y, radius)
        x_min (float): Lower x limit, m
        x_max (float): Upper x limit, m
        y_min (float): Lower y limit, m
        y_max (float): Upper y limit, m
        cell (float): Cell side, m

    Returns:
        tuple: Last cell indexes, cell start offsets and obstacle indexes
    """
    nx_max = int(py_floordiv(x_max - x_min, cell))
    ny_max = int(py_floordiv(y_max - y_min, cell))
    n_cells = (nx_max + 1) * (ny_max + 1)
    lo_x = np.empty(obs.shape[0], np.int64)
    hi_x = np.empty(obs.shape[0], np.int64)
    lo_y = np.empty(obs.shape[0], np.int64)
    hi_y = np.empty(obs.shape[0], np.int64)
    count = np.zeros(n_cells + 1, np.int64)
    for k in range(obs.shape[0]):
        x, y, r = obs[k, 0], obs[k, 1], obs[k, 2]
        lo_x[k] = max(0, int(py_floordiv(x - r - x_min, cell)))
        hi_x[k] = min(nx_max, int(py_floordiv(x + r - x_min, cell)))
        lo_y[k] = max(0, int(py_floordiv(y - r - y_min, cell)))
        hi_y[k] = min(ny_max, int(py_floordiv(y + r - y_min, cell)))
        for cx in range(lo_x[k], hi_x[k] + 1):
            for cy in range(lo_y[k], hi_y[k] + 1):
                count[cx * (ny_max + 1) + cy + 1] += 1
    for c in range(n_cells):
        count[c + 1] += count[c]
    items = np.empty(count[n_cells], np.int64)
    fill = count[:n_cells].copy()
    for k in range(obs.shape[0]):
        for cx in range(lo_x[k], hi_x[k] + 1):
            for cy in range(lo_y[k], hi_y[k] + 1):
                c = cx * (ny_max + 1) + cy
                items[fill[c]] = k
                fill[c] += 1
    return nx_max, ny_max, count, items


@njit(cache=True)
def local_obstacles(
    ox,
    oy,
    others,
    tcpa,
    n_samples,
    extra,
    extra_after,
    margin,
    sx,
    sy,
    x_min,
    x_max,
    y_min,
    y_max,
    cell,
):
    """Build the obstacles of the local planner from the predicted target positions.

    Args:
        ox (float): Own x, m
        oy (float): Own y, m
        others (ndarray): Targets (x, y, v, heading, radius)
        tcpa (float): Time to closest approach, s
        n_samples (int): Samples per target
        extra (ndarray): Extra obstacle discs
        extra_after (int): Target index after which the extra discs go
        margin (float): Clearance kept around own vessel, m
        sx (float): Shift along x, m
        sy (float): Shift along y, m
        x_min (float): Lower x limit, m
        x_max (float): Upper x limit, m
        y_min (float): Lower y limit, m
        y_max (float): Upper y limit, m
        cell (float): Cell side, m

    Returns:
        tuple: Obstacles and their grid arrays
    """
    ns = n_samples + 1
    total = others.shape[0] * ns + extra.shape[0]
    w = np.empty((total, 3))
    k = 0
    for i in range(others.shape[0]):
        vx = others[i, 2] * c_cos(others[i, 3])
        vy = others[i, 2] * c_sin(others[i, 3])
        for s in range(ns):
            t = tcpa * float(s) / float(n_samples)
            w[k, 0] = others[i, 0] + vx * t
            w[k, 1] = others[i, 1] + vy * t
            w[k, 2] = others[i, 4]
            k += 1
        if i == extra_after:
            for e in range(extra.shape[0]):
                w[k, 0], w[k, 1], w[k, 2] = extra[e, 0], extra[e, 1], extra[e, 2]
                k += 1

    out = np.empty((k, 3))
    m = 0
    for j in range(k):
        d = c_hypot(w[j, 0] - ox, w[j, 1] - oy)
        inside = d < w[j, 2] + margin
        x = d - margin - 1.0
        new_r = 0.0 if x < 0.0 else x
        if inside and not new_r > 1e-4:
            continue
        out[m, 0] = w[j, 0] - sx
        out[m, 1] = w[j, 1] - sy
        out[m, 2] = new_r if inside else w[j, 2]
        m += 1
    obs = out[:m].copy()
    nxm, nym, start, items = grid_build(obs, x_min, x_max, y_min, y_max, cell)
    return obs, nxm, nym, start, items


@njit(cache=True)
def grid_free(x, y, margin, obs, bounds, cell, nx_max, ny_max, start, items):
    """Check that a point is inside the map and clear of obstacles.

    Args:
        x (float): Point x, m
        y (float): Point y, m
        margin (float): Extra clearance, m
        obs (ndarray): Obstacle discs (x, y, radius)
        bounds (ndarray): Map limits (x_min, x_max, y_min, y_max)
        cell (float): Cell side, m
        nx_max (int): Last cell index along x
        ny_max (int): Last cell index along y
        start (ndarray): Cell start offsets
        items (ndarray): Obstacle indexes

    Returns:
        bool: True if the point is free
    """
    if x <= bounds[0] or x >= bounds[1] or y <= bounds[2] or y >= bounds[3]:
        return False
    cx0 = int(py_floordiv(x - bounds[0], cell))
    cy0 = int(py_floordiv(y - bounds[2], cell))
    reach = 1 + int(py_floordiv(margin, cell))
    for dx in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            cx, cy = cx0 + dx, cy0 + dy
            if cx < 0 or cx > nx_max or cy < 0 or cy > ny_max:
                continue
            c = cx * (ny_max + 1) + cy
            for m in range(start[c], start[c + 1]):
                k = items[m]
                dist_sq = c_pow(x - obs[k, 0], 2.0) + c_pow(y - obs[k, 1], 2.0)
                if dist_sq <= c_pow(obs[k, 2] + margin, 2.0):
                    return False
    return True


@njit(cache=True)
def segment_valid(
    ax, ay, bx, by, margin, own_on, own, obs, bounds, cell, nxm, nym, start, items
):
    """Check that a segment is free and does not start behind own vessel.

    Args:
        ax (float): Start x, m
        ay (float): Start y, m
        bx (float): End x, m
        by (float): End y, m
        margin (float): Extra clearance, m
        own_on (bool): Whether own vessel is known
        own (ndarray): Own (x, y, heading)
        obs (ndarray): Obstacle discs (x, y, radius)
        bounds (ndarray): Map limits (x_min, x_max, y_min, y_max)
        cell (float): Cell side, m
        nxm (int): Last cell index along x
        nym (int): Last cell index along y
        start (ndarray): Cell start offsets
        items (ndarray): Obstacle indexes

    Returns:
        bool: True if the segment is valid
    """
    if own_on:
        at_a = abs(ax - own[0]) < 0.1 and abs(ay - own[1]) < 0.1
        at_b = abs(bx - own[0]) < 0.1 and abs(by - own[1]) < 0.1
        if at_a or at_b:
            seg_heading = (
                c_atan2(by - ay, bx - ax) if at_a else c_atan2(ay - by, ax - bx)
            )
            diff = py_mod(seg_heading - own[2] + np.pi, 2 * np.pi) - np.pi
            if abs(diff) > np.pi / 2.0:
                return False
    dist = c_hypot(bx - ax, by - ay)
    steps = max(3, int(dist / SEGMENT_CHECK_M))
    for i in range(1, steps):
        t = i / steps
        x = ax + t * (bx - ax)
        y = ay + t * (by - ay)
        if not grid_free(x, y, margin, obs, bounds, cell, nxm, nym, start, items):
            return False
    return True


@njit(cache=True)
def heading_change(ax, ay, bx, by, heading):
    """Compute the turn needed to follow a segment.

    Args:
        ax (float): Start x, m
        ay (float): Start y, m
        bx (float): End x, m
        by (float): End y, m
        heading (float): Current heading, rad

    Returns:
        float: Heading change, rad, positive to port
    """
    dx, dy = bx - ax, by - ay
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    seg = c_atan2(dy, dx)
    cur = c_atan2(c_sin(heading), c_cos(heading))
    planned = c_atan2(c_sin(seg), c_cos(seg))
    d = planned - cur
    return c_atan2(c_sin(d), c_cos(d))


@njit(cache=True)
def edge_penalty(ax, ay, bx, by, own, sits, others):
    """Sum the COLREGs penalties of a segment.

    Args:
        ax (float): Start x, m
        ay (float): Start y, m
        bx (float): End x, m
        by (float): End y, m
        own (ndarray): Own (x, y, heading)
        sits (ndarray): Situation code per target
        others (ndarray): Targets (x, y, heading)

    Returns:
        float: Total penalty
    """
    penalty = 0.0
    for j in range(sits.shape[0]):
        s = sits[j]
        ox, oy, oth = others[j, 0], others[j, 1], others[j, 2]
        if s == SIT_HEAD_ON:
            if not heading_change(ax, ay, bx, by, own[2]) < 0:
                penalty += 100000
        elif s == SIT_CROSS_GIVE_WAY:
            if not heading_change(ax, ay, bx, by, own[2]) < 0:
                penalty += 50000
            rx, ry = ox - own[0], oy - own[1]
            behind = True
            if not math.sqrt(rx * rx + ry * ry) < 1e-6:
                ux, uy = c_cos(oth), c_sin(oth)
                projection = rx * ux + ry * uy
                nx = rx - (bx - ax)
                ny = ry - (by - ay)
                behind = nx * ux + ny * uy <= projection
            if not behind:
                penalty += 50000
        elif s == SIT_OVERTAKING:
            ux, uy = c_cos(oth), c_sin(oth)
            rx, ry = own[0] - ox, own[1] - oy
            nx = rx + (bx - ax)
            ny = ry + (by - ay)
            if not ((ux * ry - uy * rx < 0) and (ux * ny - uy * nx < 0)):
                penalty += 30000
        elif s == SIT_STAND_ON:
            dh = heading_change(ax, ay, bx, by, own[2])
            if dh > TURN_TOLERANCE_RAD:
                penalty += 200000
            elif dh < -TURN_TOLERANCE_RAD:
                penalty += 25000
    return penalty


@njit(cache=True)
def _nearest(px, py, tree, n, sx, sy):
    """Find the tree node closest to a point.

    Args:
        px (ndarray): Node x values
        py (ndarray): Node y values
        tree (ndarray): Node indexes of the tree
        n (int): Tree size
        sx (float): Point x, m
        sy (float): Point y, m

    Returns:
        int: Index of the closest node
    """
    best, best_d = 0, math.inf
    for k in range(n):
        dx = px[tree[k]] - sx
        dy = py[tree[k]] - sy
        d = dx * dx + dy * dy
        if d < best_d:
            best, best_d = k, d
    return tree[best]


@njit(cache=True)
def _is_ancestor(node, potential, parent, stamp, mark):
    """Check whether a node is above another in the tree, or a loop was found.

    Args:
        node (int): Start node
        potential (int): Node to look for
        parent (ndarray): Parent of each node
        stamp (ndarray): Visit marks
        mark (int): Mark for this search

    Returns:
        bool: True if found or a loop is hit
    """
    curr = node
    while curr >= 0:
        if curr == potential:
            return True
        if stamp[curr] == mark:
            return True
        stamp[curr] = mark
        curr = parent[curr]
    return False


@njit(cache=True)
def plan_core(
    start,
    goal,
    map_size,
    step_size,
    max_iter,
    search_radius,
    margin,
    obs,
    bounds,
    cell,
    nxm,
    nym,
    gstart,
    items,
    own_on,
    own,
    pen_on,
    sits_known,
    sits,
    others,
    comp_on,
    comp,
    comp_side,
    mt,
    pos,
    px,
    py,
    cost,
    parent,
    trees,
    lens,
    out,
):
    """Run the two tree RRT* search in numba.

    Args:
        start (ndarray): Start point
        goal (ndarray): Goal point
        map_size (float): Map size, m
        step_size (float): Max edge length, m
        max_iter (int): Max iterations
        search_radius (float): Rewire radius, m
        margin (float): Extra clearance, m
        obs (ndarray): Obstacle discs (x, y, radius)
        bounds (ndarray): Map limits (x_min, x_max, y_min, y_max)
        cell (float): Cell side, m
        nxm (int): Last cell index along x
        nym (int): Last cell index along y
        gstart (ndarray): Cell start offsets
        items (ndarray): Obstacle indexes
        own_on (bool): Whether own vessel is known
        own (ndarray): Own (x, y, heading)
        pen_on (bool): Whether to add COLREGs penalties
        sits_known (bool): Whether situations are already computed
        sits (ndarray): Situation code per target
        others (ndarray): Targets (x, y, heading)
        comp_on (bool): Whether to force the compliant side
        comp (ndarray): Reference line of the compliant side
        comp_side (float): Side to keep, 1 or -1
        mt (ndarray): Random state words
        pos (ndarray): Position in the random state
        px (ndarray): Node x values, filled
        py (ndarray): Node y values, filled
        cost (ndarray): Node costs, filled
        parent (ndarray): Node parents, filled
        trees (ndarray): Node indexes of both trees, filled
        lens (ndarray): Size of both trees, filled
        out (ndarray): Result data, filled

    Returns:
        int: Status code (FOUND, DIRECT, FAILED or GOAL_TRAPPED)
    """
    gx, gy = goal[0], goal[1]
    px[0], py[0], cost[0], parent[0] = start[0], start[1], 0.0, -1
    px[1], py[1], cost[1], parent[1] = gx, gy, 0.0, -1
    n_nodes = 2
    trees[0, 0], trees[1, 0] = 0, 1
    lens[0], lens[1] = 1, 1
    out[0], out[1], out[4] = gx, gy, 0.0
    active_c = comp_on

    if not grid_free(gx, gy, margin, obs, bounds, cell, nxm, nym, gstart, items):
        dx, dy = start[0] - gx, start[1] - gy
        dist = c_hypot(dx, dy)
        if dist < 1e-6:
            return GOAL_TRAPPED
        ux, uy = dx / dist, dy / dist
        max_steps = max(int(dist / GOAL_WALK_STEP_M), 1)
        moved = False
        for step in range(1, max_steps + 1):
            cx = gx + ux * GOAL_WALK_STEP_M * step
            cy = gy + uy * GOAL_WALK_STEP_M * step
            if grid_free(cx, cy, margin, obs, bounds, cell, nxm, nym, gstart, items):
                gx, gy = cx, cy
                px[1], py[1] = cx, cy
                out[0], out[1] = cx, cy
                out[4] = 1.0
                moved = True
                break
        if not moved:
            return GOAL_TRAPPED

    if segment_valid(
        start[0],
        start[1],
        gx,
        gy,
        margin,
        own_on,
        own,
        obs,
        bounds,
        cell,
        nxm,
        nym,
        gstart,
        items,
    ):
        compliant_goal = True
        if active_c:
            signed = (gx - comp[0]) * (-c_sin(comp[2])) + (gy - comp[1]) * c_cos(
                comp[2]
            )
            compliant_goal = comp_side * signed <= comp[3]
        if compliant_goal:
            lawful = True
            if pen_on:
                if not sits_known:
                    return NEED_SITUATIONS
                lawful = (
                    edge_penalty(start[0], start[1], gx, gy, own, sits, others) <= 0.0
                )
            if lawful:
                return DIRECT

    stamp = np.zeros(px.shape[0], np.int64)
    mark = 0
    nb = np.empty(px.shape[0], np.int64)
    for i in range(max_iter):
        if active_c and comp[4] == 0.0 and i == max_iter // 2:
            active_c = False
        a, b = (0, 1) if i % 2 == 0 else (1, 0)

        if mt_random(mt, pos) < GOAL_BIAS:
            sx, sy = gx, gy
        else:
            sx = sy = 0.0
            for _ in range(SAMPLE_TRIES):
                sx = mt_uniform(mt, pos, -map_size[0], map_size[0])
                sy = mt_uniform(mt, pos, -map_size[1], map_size[1])
                if not active_c:
                    break
                signed = (sx - comp[0]) * (-c_sin(comp[2])) + (sy - comp[1]) * c_cos(
                    comp[2]
                )
                if comp_side * signed <= comp[3]:
                    break

        near = _nearest(px, py, trees[a], lens[a], sx, sy)

        dist = c_hypot(sx - px[near], sy - py[near])
        step = dist if dist < step_size else step_size
        if dist < 1e-6:
            new = near
        else:
            angle = c_atan2(sy - py[near], sx - px[near])
            new = n_nodes
            px[new] = px[near] + step * c_cos(angle)
            py[new] = py[near] + step * c_sin(angle)
            cost[new], parent[new] = 0.0, -1

        if not segment_valid(
            px[near],
            py[near],
            px[new],
            py[new],
            margin,
            own_on,
            own,
            obs,
            bounds,
            cell,
            nxm,
            nym,
            gstart,
            items,
        ):
            continue
        if new == n_nodes:
            n_nodes += 1

        m = lens[a]
        r = 5.0 * math.sqrt(c_log(float(m)) / m)
        r = r if r < search_radius else search_radius
        r = step_size * 1.5 if step_size * 1.5 > r else r
        r_sq = c_pow(r, 2.0)
        n_nb = 0
        for k in range(m):
            node = trees[a, k]
            dx = px[node] - px[new]
            dy = py[node] - py[new]
            if dx * dx + dy * dy < r_sq * (1.0 + NEIGHBOUR_SLACK):
                if (
                    c_pow(px[node] - px[new], 2.0) + c_pow(py[node] - py[new], 2.0)
                    < r_sq
                ):
                    nb[n_nb] = node
                    n_nb += 1

        if pen_on and not sits_known:
            return NEED_SITUATIONS

        parent[new] = near
        c0 = c_hypot(px[new] - px[near], py[new] - py[near])
        if pen_on:
            c0 += edge_penalty(px[near], py[near], px[new], py[new], own, sits, others)
        cost[new] = cost[near] + c0
        for q in range(n_nb):
            node = nb[q]
            ce = c_hypot(px[new] - px[node], py[new] - py[node])
            if pen_on:
                ce += edge_penalty(
                    px[node], py[node], px[new], py[new], own, sits, others
                )
            c = cost[node] + ce
            if c < cost[new] and segment_valid(
                px[node],
                py[node],
                px[new],
                py[new],
                margin,
                own_on,
                own,
                obs,
                bounds,
                cell,
                nxm,
                nym,
                gstart,
                items,
            ):
                cost[new] = c
                parent[new] = node

        trees[a, lens[a]] = new
        lens[a] += 1

        for q in range(n_nb):
            node = nb[q]
            if parent[new] == node:
                continue
            if new == node:
                continue
            ce = c_hypot(px[node] - px[new], py[node] - py[new])
            if pen_on:
                ce += edge_penalty(
                    px[new], py[new], px[node], py[node], own, sits, others
                )
            new_cost = cost[new] + ce
            if new_cost < cost[node] and segment_valid(
                px[new],
                py[new],
                px[node],
                py[node],
                margin,
                own_on,
                own,
                obs,
                bounds,
                cell,
                nxm,
                nym,
                gstart,
                items,
            ):
                mark += 1
                if _is_ancestor(new, node, parent, stamp, mark):
                    continue
                parent[node] = new
                cost[node] = new_cost

        target = _nearest(px, py, trees[b], lens[b], px[new], py[new])
        if c_hypot(px[new] - px[target], py[new] - py[target]) < step_size * 2:
            if segment_valid(
                px[new],
                py[new],
                px[target],
                py[target],
                margin,
                own_on,
                own,
                obs,
                bounds,
                cell,
                nxm,
                nym,
                gstart,
                items,
            ):
                out[2], out[3] = new, target
                return FOUND
    return FAILED


@njit(cache=True)
def in_tree(node, tree, n):
    """Check whether a node is in a tree.

    Args:
        node (int): Node index
        tree (ndarray): Node indexes of the tree
        n (int): Tree size

    Returns:
        bool: True if found
    """
    for k in range(n):
        if tree[k] == node:
            return True
    return False


@njit(cache=True)
def trace_back(node, px, py, parent, stamp, mark, buf, n):
    """Copy the points from a node up to its root into a buffer.

    Args:
        node (int): Start node
        px (ndarray): Node x values
        py (ndarray): Node y values
        parent (ndarray): Parent of each node
        stamp (ndarray): Visit marks
        mark (int): Mark for this trace
        buf (ndarray): Output points
        n (int): Next free row in buf

    Returns:
        int: Next free row after the copy
    """
    curr = node
    count = 0
    while curr >= 0:
        if stamp[curr] == mark:
            break
        stamp[curr] = mark
        buf[n, 0], buf[n, 1] = px[curr], py[curr]
        n += 1
        count += 1
        curr = parent[curr]
        if count > TRACE_LIMIT:
            break
    return n


@njit(cache=True)
def prune_core(
    path,
    n,
    margin,
    own_on,
    own,
    pen_on,
    sits,
    others,
    obs,
    bounds,
    cell,
    nxm,
    nym,
    gstart,
    items,
    keep,
):
    """Remove path points that can be skipped in a straight line.

    Args:
        path (ndarray): Path points
        n (int): Number of points
        margin (float): Extra clearance, m
        own_on (bool): Whether own vessel is known
        own (ndarray): Own (x, y, heading)
        pen_on (bool): Whether to refuse COLREGs penalties
        sits (ndarray): Situation code per target
        others (ndarray): Targets (x, y, heading)
        obs (ndarray): Obstacle discs (x, y, radius)
        bounds (ndarray): Map limits (x_min, x_max, y_min, y_max)
        cell (float): Cell side, m
        nxm (int): Last cell index along x
        nym (int): Last cell index along y
        gstart (ndarray): Cell start offsets
        items (ndarray): Obstacle indexes
        keep (ndarray): Indexes of kept points, filled

    Returns:
        int: Number of kept points
    """
    if n < 3:
        for k in range(n):
            keep[k] = k
        return n
    keep[0] = 0
    m = 1
    i = 0
    while i < n - 1:
        furthest = i + 1
        for j in range(i + 2, n):
            ax, ay, bx, by = path[i, 0], path[i, 1], path[j, 0], path[j, 1]
            if segment_valid(
                ax,
                ay,
                bx,
                by,
                margin,
                own_on,
                own,
                obs,
                bounds,
                cell,
                nxm,
                nym,
                gstart,
                items,
            ):
                dist = c_hypot(bx - ax, by - ay)
                c = c_hypot(bx - ax, by - ay)
                if pen_on:
                    c += edge_penalty(ax, ay, bx, by, own, sits, others)
                if c > dist + PRUNE_COST_SLACK_M:
                    break
                furthest = j
            else:
                break
        keep[m] = furthest
        m += 1
        i = furthest
    return m


@njit(cache=True)
def elastic_core(xs, ys, margin, obs, bounds, cell, nxm, nym, gstart, items):
    """Smooth a path by pulling each point toward its neighbours.

    Args:
        xs (ndarray): Point x values, updated
        ys (ndarray): Point y values, updated
        margin (float): Extra clearance, m
        obs (ndarray): Obstacle discs (x, y, radius)
        bounds (ndarray): Map limits (x_min, x_max, y_min, y_max)
        cell (float): Cell side, m
        nxm (int): Last cell index along x
        nym (int): Last cell index along y
        gstart (ndarray): Cell start offsets
        items (ndarray): Obstacle indexes
    """
    last = xs.shape[0] - 1
    for _ in range(ELASTIC_ITERATIONS):
        moved = False
        for i in range(1, last):
            nx = xs[i] + 0.5 * (xs[i - 1] + xs[i + 1] - 2 * xs[i])
            ny = ys[i] + 0.5 * (ys[i - 1] + ys[i + 1] - 2 * ys[i])
            if nx == xs[i] and ny == ys[i]:
                continue
            if grid_free(nx, ny, margin, obs, bounds, cell, nxm, nym, gstart, items):
                xs[i], ys[i] = nx, ny
                moved = True
        if not moved:
            break


def grid_arrays(obstacles, bounds, cell):
    """Build all grid arrays from a list of obstacles.

    Args:
        obstacles (list): Obstacle discs (x, y, radius)
        bounds (list): Map limits (x_min, x_max, y_min, y_max)
        cell (float): Cell side, m

    Returns:
        tuple: Obstacles, bounds, cell and CSR arrays
    """
    obs = np.array(obstacles, dtype=float).reshape(-1, 3)
    b = np.array([float(v) for v in bounds])
    c = float(cell)
    nxm, nym, start, items = grid_build(obs, b[0], b[1], b[2], b[3], c)
    return obs, b, c, nxm, nym, start, items


def supports(rrt):
    """Check whether an RRT* instance can use the numba path.

    Args:
        rrt (RRTStar): Planner instance

    Returns:
        bool: True if the numba version gives the same result
    """
    from .algo_rttstar import MaritimeValidator
    from .colregs_planner import COLREGsPlanner

    v = rrt.validator
    if rrt.plot or type(v) is not MaritimeValidator or not hasattr(v.grid, "csr"):
        return False
    planner = v.planner
    return planner is None or (
        type(planner).colregs_violation_penalty
        is COLREGsPlanner.colregs_violation_penalty
    )


_PLAN_BUF = {}


def _plan_buffers(max_iter):
    """Get reusable node buffers for a given iteration count.

    Args:
        max_iter (int): Max iterations

    Returns:
        tuple: Buffers for x, y, cost, parent and trees
    """
    buf = _PLAN_BUF.get(max_iter)
    if buf is None:
        capacity = max_iter + 2
        buf = (
            np.empty(capacity),
            np.empty(capacity),
            np.empty(capacity),
            np.empty(capacity, np.int64),
            np.empty((2, max_iter + 1), np.int64),
        )
        _PLAN_BUF[max_iter] = buf
    return buf


def edge_penalty_on(v):
    """Check whether COLREGs edge penalties apply.

    Args:
        v (MaritimeValidator): Validator of the planner

    Returns:
        bool: True if penalties apply
    """
    from .algo_rttstar import COLREGS_EDGE_PENALTY

    return bool(COLREGS_EDGE_PENALTY and v.planner and v.own_ship and v.sea_objects)


def plan(rrt):
    """Run RRT* with the numba kernels and store the path on the planner.

    Args:
        rrt (RRTStar): Planner instance

    Returns:
        list: Path points, or None if no path is found
    """
    from .algo_rttstar import Node, PathOptimizer

    v = rrt.validator
    rrt.start_tree = [rrt.start]
    rrt.goal_tree = [Node(rrt.goal.x, rrt.goal.y)]
    rrt._xy = {}
    rrt._compliant_active = rrt.compliant_side != 0 and rrt.compliant_ref is not None

    grid = v.grid.csr()
    own_on = v.own_ship is not None
    sxy = (
        v.start_xy
        if v.start_xy is not None
        else (v.own_ship.x, v.own_ship.y) if own_on else None
    )
    own = (
        np.array([float(sxy[0]), float(sxy[1]), float(v.own_ship.theta)])
        if own_on
        else np.zeros(3)
    )
    pen_on = edge_penalty_on(v)
    comp = np.zeros(5)
    if rrt._compliant_active:
        comp[:] = (
            rrt.compliant_ref[0],
            rrt.compliant_ref[1],
            rrt.compliant_heading,
            rrt.compliant_tol,
            float(bool(getattr(rrt, "compliant_hold", False))),
        )
    max_iter = int(rrt.max_iter)
    px, py, cost, parent, trees = _plan_buffers(max_iter)
    lens = np.zeros(2, np.int64)
    out = np.zeros(5)
    start = np.array([float(rrt.start.x), float(rrt.start.y)])
    goal = np.array([float(rrt.goal.x), float(rrt.goal.y)])
    map_size = np.array([float(rrt.map_size[0]), float(rrt.map_size[1])])

    mt0, pos0, rng_state = rng_export()
    sits_known, sits, others = False, np.zeros(0, np.int64), np.zeros((0, 3))
    while True:
        mt, pos = mt0.copy(), pos0.copy()
        status = plan_core(
            start,
            goal,
            map_size,
            float(rrt.step_size),
            max_iter,
            float(rrt._search_radius),
            float(v.safety_margin),
            *grid,
            own_on,
            own,
            pen_on,
            sits_known,
            sits,
            others,
            rrt._compliant_active,
            comp,
            int(rrt.compliant_side),
            mt,
            pos,
            px,
            py,
            cost,
            parent,
            trees,
            lens,
            out,
        )
        if status != NEED_SITUATIONS:
            break
        try:
            pairs = v.planner.edge_situations(v.own_ship, v.sea_objects)
            sits = np.array(
                [SIT_CODE.get(s, SIT_OTHER) for _o, s in pairs], dtype=np.int64
            )
            others = np.array(
                [(float(o.x), float(o.y), float(o.theta)) for o, _s in pairs]
            )
            others = others.reshape(-1, 3)
        except Exception:
            pen_on = False
        sits_known = True
    rng_import(mt, pos, rng_state)

    nodes = {0: rrt.start}
    goal_root = Node(px[1], py[1])
    nodes[1] = goal_root
    if out[4]:
        rrt.goal = Node(px[1], py[1])
    for t in range(2):
        for idx in trees[t, : lens[t]]:
            if idx not in nodes:
                nodes[idx] = Node(px[idx], py[idx])
    for idx, node in nodes.items():
        node.cost = cost[idx]
        node.parent = nodes[parent[idx]] if parent[idx] >= 0 else None
    rrt.start_tree = [nodes[i] for i in trees[0, : lens[0]]]
    rrt.goal_tree = [nodes[i] for i in trees[1, : lens[1]]]

    if status == DIRECT:
        rrt.path = [[rrt.start.x, rrt.start.y], [rrt.goal.x, rrt.goal.y]]
        return rrt.path
    if status != FOUND:
        return None

    node_a, node_b = nodes[int(out[2])], nodes[int(out[3])]
    rrt.connection_nodes = (node_a, node_b)
    n_start, n_goal = (
        (node_a, node_b)
        if rrt._is_node_in_tree(node_a, rrt.start_tree)
        else (node_b, node_a)
    )
    path_start = _trace_back(n_start)
    path_start.reverse()
    raw_path = path_start + _trace_back(n_goal)
    if not rrt.opti:
        rrt.path = raw_path
        return rrt.path

    keep = np.empty(len(raw_path), np.int64)
    n_keep = prune_core(
        np.array(raw_path, dtype=float).reshape(-1, 2),
        len(raw_path),
        float(v.safety_margin),
        own_on,
        own,
        pen_on,
        sits,
        others,
        *grid,
        keep,
    )
    pruned = raw_path if len(raw_path) < 3 else [raw_path[k] for k in keep[:n_keep]]
    smoothed = PathOptimizer.bspline_smooth(pruned)
    xs = np.array([float(p[0]) for p in smoothed])
    ys = np.array([float(p[1]) for p in smoothed])
    if len(smoothed) >= 3:
        elastic_core(xs, ys, float(v.safety_margin), *grid)
    rrt.path = PathOptimizer.first_leg_or(
        [[x, y] for x, y in zip(xs.tolist(), ys.tolist())], pruned, v.own_ship
    )
    return rrt.path


def _trace_back(node, limit=TRACE_LIMIT):
    """Follow the parents of a node up to the root.

    Args:
        node (Node): Start node
        limit (int): Max number of points

    Returns:
        list: Path points from node to root
    """
    path = []
    curr = node
    visited = set()
    while curr:
        if id(curr) in visited:
            break
        visited.add(id(curr))
        path.append([curr.x, curr.y])
        curr = curr.parent
        if len(path) > limit:
            break
    return path

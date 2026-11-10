from __future__ import annotations

import math

from . import duty as _duty
from .geometry import R_MIN_M, Vessel, _wrap

STRAIGHT_CLASSES = ("breakdown", "oblivious")
FREE_CLASSES = ("adversarial", "interceptor", "mistaken", "erratic")
COMPLIANT_CLASSES = ("compliant", "late", "wide_berth")


class Tube:

    __slots__ = ("v_max", "v_min", "omega", "side", "turn_cap", "cls")

    def __init__(self, v_max, omega, side=0, turn_cap=math.inf, cls="", v_min=None):
        """Describe the set of positions a target can reach.

        Args:
            v_max (float): Max speed, m/s
            omega (float): Max turn rate, rad/s
            side (int): Allowed turn side, -1 starboard, 1 port, 0 both
            turn_cap (float): Max total turn, rad
            cls (str): Behaviour class
            v_min (float): Min speed, m/s, or None for v_max
        """
        self.v_max = float(v_max)
        self.v_min = float(v_max if v_min is None else v_min)
        self.omega = float(omega)
        self.side = int(side)
        self.turn_cap = float(turn_cap)
        self.cls = str(cls)

    def __repr__(self):
        """Describe the tube in plain text.

        Returns:
            str: Text description
        """
        return (
            f"Tube(v={self.v_min:.2f}-{self.v_max:.2f}, omega={self.omega:.4f}, "
            f"side={self.side}, turn_cap={self.turn_cap:.3f}, cls={self.cls!r})"
        )

    @property
    def is_line(self):
        """Check whether the target can only go straight at fixed speed.

        Returns:
            bool: True if the tube is a line
        """
        return (self.omega <= 0.0 or self.turn_cap <= 0.0) and self.v_min == self.v_max

    def saturation_time(self):
        """Compute when the turn spread stops growing.

        Returns:
            float: Time, s, inf if it never turns
        """
        if self.omega <= 0.0 or self.turn_cap <= 0.0:
            return math.inf
        return min(math.pi, self.turn_cap) / self.omega

    def half_angle(self, t):
        """Compute the turn spread at a given time.

        Args:
            t (float): Time, s

        Returns:
            float: Half angle, rad
        """
        if self.omega <= 0.0 or self.turn_cap <= 0.0:
            return 0.0
        return min(math.pi, min(self.omega * float(t), self.turn_cap))

    def bearings(self, t):
        """Get the bearing range of the tube at a given time.

        Args:
            t (float): Time, s

        Returns:
            tuple: Lower and upper bearing, rad
        """
        h = self.half_angle(t)
        if h >= math.pi:
            return -math.pi, math.pi
        if self.side < 0:
            return -h, 0.0
        if self.side > 0:
            return 0.0, h
        return -h, h

    def radius(self, t):
        """Get the outer radius of the tube at a given time.

        Args:
            t (float): Time, s

        Returns:
            float: Radius, m
        """
        return self.v_max * float(t)

    def radius_in(self, t):
        """Get the inner radius of the tube at a given time.

        Args:
            t (float): Time, s

        Returns:
            float: Radius, m
        """
        if self.omega > 0.0 and self.turn_cap > 0.0:
            return 0.0
        return self.v_min * float(t)


def tube_for(cls, ts, os_v=None, *, r_min=R_MIN_M, gate=None, v_min=None):
    """Build the tube for a behaviour class.

    Args:
        cls (str): Behaviour class
        ts (Vessel): Target vessel
        os_v (Vessel): Own vessel, or None
        r_min (float): Min turn radius, m
        gate (dict): Gate settings, or None
        v_min (float): Min speed, m/s, or None

    Returns:
        Tube: Reachable tube
    """
    omega = float(ts.v) / float(r_min)
    lo = None if v_min is None else min(float(v_min), float(ts.v))
    if cls in STRAIGHT_CLASSES:
        return Tube(ts.v, 0.0, 0, 0.0, cls, lo)
    if cls in FREE_CLASSES:
        return Tube(ts.v, omega, 0, math.inf, cls, lo)
    if cls not in COMPLIANT_CLASSES:
        raise ValueError(
            f"no tube for behaviour class {cls!r}; want one of "
            f"{STRAIGHT_CLASSES + FREE_CLASSES + COMPLIANT_CLASSES}"
        )
    if os_v is None:
        return Tube(ts.v, omega, 0, math.inf, cls, lo)
    if _duty.role(ts, os_v, gate=(gate or {}).get("dcpa_gate")) == "OGW":
        return Tube(ts.v, omega, 0, math.inf, cls, lo)
    return Tube(ts.v, omega, -1, math.inf, cls, lo)


def worst_case(ts, *, r_min=R_MIN_M, v_min=None):
    """Build the tube where the target can turn freely both ways.

    Args:
        ts (Vessel): Target vessel
        r_min (float): Min turn radius, m
        v_min (float): Min speed, m/s, or None

    Returns:
        Tube: Reachable tube
    """
    lo = None if v_min is None else min(float(v_min), float(ts.v))
    return Tube(ts.v, float(ts.v) / float(r_min), 0, math.inf, "worst", lo)


def tubes_for(classes, threats, os_v=None, *, r_min=R_MIN_M, gate=None, v_min=None):
    """Build one tube per target.

    Args:
        classes (object): One class, a list of classes, or None for worst case
        threats (list): Target vessels
        os_v (Vessel): Own vessel, or None
        r_min (float): Min turn radius, m
        gate (dict): Gate settings, or None
        v_min (float): Min speed, m/s, or None

    Returns:
        list: Reachable tubes
    """
    if classes is None:
        return [worst_case(t, r_min=r_min, v_min=v_min) for t in threats]
    if isinstance(classes, str):
        classes = [classes] * len(threats)
    return [
        tube_for(c, t, os_v, r_min=r_min, gate=gate, v_min=v_min)
        for c, t in zip(classes, threats)
    ]


def contains(tube, ts, point, t, *, tol=1e-9):
    """Check whether a point is inside the tube at a given time.

    Args:
        tube (Tube): Reachable tube
        ts (Vessel): Target vessel
        point (tuple): Point (x, y), m
        t (float): Time, s
        tol (float): Tolerance

    Returns:
        bool: True if inside
    """
    dx, dy = point[0] - ts.x, point[1] - ts.y
    r = math.hypot(dx, dy)
    if not (tube.radius_in(t) - tol <= r <= tube.radius(t) + tol):
        return False
    lo, hi = tube.bearings(t)
    if hi - lo >= 2.0 * math.pi - 1e-12 or r <= tol:
        return True
    b = _wrap(math.atan2(dy, dx) - ts.theta)
    return (lo - tol) <= b <= (hi + tol)


def clearance(point, tube, ts, t):
    """Compute the distance from a point to the tube at a given time.

    Args:
        point (tuple): Point (x, y), m
        tube (Tube): Reachable tube
        ts (Vessel): Target vessel
        t (float): Time, s

    Returns:
        float: Distance, m, 0 if inside
    """
    dx, dy = point[0] - ts.x, point[1] - ts.y
    r = math.hypot(dx, dy)
    R_out, R_in = tube.radius(t), tube.radius_in(t)
    lo, hi = tube.bearings(t)
    full = hi - lo >= 2.0 * math.pi - 1e-12
    b = 0.0 if r <= 1e-12 else _wrap(math.atan2(dy, dx) - ts.theta)
    if full or (lo <= b <= hi):
        return max(r - R_out, R_in - r, 0.0)
    best = math.inf
    for edge in (lo, hi):
        ex, ey = math.cos(ts.theta + edge), math.sin(ts.theta + edge)
        proj = min(R_out, max(R_in, dx * ex + dy * ey))
        best = min(best, math.hypot(dx - proj * ex, dy - proj * ey))
    return best


def clearance_many(px, py, tube, ts, t):
    """Array version of clearance.

    Args:
        px (ndarray): Point x values, m
        py (ndarray): Point y values, m
        tube (Tube): Reachable tube
        ts (Vessel): Target vessel
        t (ndarray): Times, s

    Returns:
        ndarray: Distances, m
    """
    import numpy as np

    px, py, t = np.broadcast_arrays(
        np.asarray(px, float), np.asarray(py, float), np.asarray(t, float)
    )
    dx, dy = px - ts.x, py - ts.y
    r = np.hypot(dx, dy)

    r_out = tube.v_max * t
    turning = tube.omega > 0.0 and tube.turn_cap > 0.0
    r_in = np.zeros_like(t) if turning else tube.v_min * t

    if turning:
        h = np.minimum(math.pi, np.minimum(tube.omega * t, tube.turn_cap))
    else:
        h = np.zeros_like(t)
    lo = -h if tube.side <= 0 else np.zeros_like(h)
    hi = h if tube.side >= 0 else np.zeros_like(h)
    lo = np.where(h >= math.pi, -math.pi, lo)
    hi = np.where(h >= math.pi, math.pi, hi)
    full = (hi - lo) >= 2.0 * math.pi - 1e-12

    b = np.where(
        r <= 1e-12,
        0.0,
        np.arctan2(
            np.sin(np.arctan2(dy, dx) - ts.theta), np.cos(np.arctan2(dy, dx) - ts.theta)
        ),
    )
    in_span = full | ((lo <= b) & (b <= hi))
    radial = np.maximum(np.maximum(r - r_out, r_in - r), 0.0)
    best = np.full(r.shape, np.inf)
    for edge in (lo, hi):
        ex, ey = np.cos(ts.theta + edge), np.sin(ts.theta + edge)
        proj = np.minimum(r_out, np.maximum(r_in, dx * ex + dy * ey))
        best = np.minimum(best, np.hypot(dx - proj * ex, dy - proj * ey))
    return np.where(in_span, radial, best)


def _selfcheck():
    """Check the tube geometry on known cases."""
    ts = Vessel("TS", 0.0, 0.0, 8.0, 0.0, R_MIN_M)
    tb = worst_case(ts)
    dt, x, y, th = 10.0, 0.0, 0.0, 0.0
    for k in range(1, 21):
        th += tb.omega * dt
        x += ts.v * math.cos(th) * dt
        y += ts.v * math.sin(th) * dt
        assert contains(tb, ts, (x, y), k * dt, tol=1e-6), (k, x, y)
    line = Tube(8.0, 0.0)
    assert line.is_line and line.half_angle(100.0) == 0.0 and line.radius(10.0) == 80.0
    assert abs(clearance((0.0, 500.0), line, ts, 10.0) - math.hypot(80.0, 500.0)) < 1e-9
    assert clearance((80.0, 0.0), line, ts, 10.0) == 0.0
    assert abs(clearance((40.0, 0.0), line, ts, 10.0) - 40.0) < 1e-9
    assert abs(tb.saturation_time() - math.pi * R_MIN_M / 8.0) < 1e-9

    import numpy as np

    rng = np.random.default_rng(0)
    shapes = (
        tb,
        line,
        Tube(8.0, tb.omega, -1, math.inf),
        Tube(8.0, tb.omega, 1, 0.4, v_min=5.0),
    )
    for shape in shapes:
        px, py = rng.uniform(-3000, 3000, 400), rng.uniform(-3000, 3000, 400)
        tt = rng.uniform(0.0, 300.0, 400)
        many = clearance_many(px, py, shape, ts, tt)
        one = [clearance((x, y), shape, ts, s) for x, y, s in zip(px, py, tt)]
        assert np.allclose(many, one, rtol=0.0, atol=1e-6), shape
    print("tube selfcheck ok")


if __name__ == "__main__":
    _selfcheck()

from __future__ import annotations

import math

import numpy as np

R_MIN_M = 500.0

TWO_PI = 2.0 * math.pi
ONE_NAUTICAL_MILE_M = 1852.0
RADAR_DETECTION_RANGE_NM = 8.0
VISIBLE_RANGE_M = RADAR_DETECTION_RANGE_NM * ONE_NAUTICAL_MILE_M
OVERTAKING_THRESHOLD = np.deg2rad(112.5)
HEAD_ON_THRESHOLD = np.deg2rad(6.0)
RECIPROCAL_THRESHOLD = np.deg2rad(170.0)
R_VESSEL_M = 75.0
HULL_CONTACT_M = 2.0 * R_VESSEL_M
TS_SPEED_RANGE_MPS = (5.14, 9.26)
HOLD_RAD = np.radians(3.0)
GIVEWAY_MIN_APPARENT_RAD = np.radians(15.0)
DELTA_MIN_APPARENT_RAD = GIVEWAY_MIN_APPARENT_RAD


class Vessel:
    __slots__ = ("name", "x", "y", "v", "theta", "r_min", "rate_of_turn")

    def __init__(self, name, x, y, v, theta, r_min=R_MIN_M):
        """Store the state of a vessel and its max turn rate.

        Args:
            name (str): Vessel name
            x (float): x position, m
            y (float): y position, m
            v (float): Speed, m/s
            theta (float): Heading, rad
            r_min (float): Min turn radius, m
        """
        self.name = name
        self.x = float(x)
        self.y = float(y)
        self.v = float(v)
        self.theta = float(theta)
        self.r_min = float(r_min)
        self.rate_of_turn = float(v) / float(r_min)

    @classmethod
    def from_sea_object(cls, obj, r_min=None, name=None):
        """Build a Vessel from a simulator object.

        Args:
            obj (SeaObject): Simulator object
            r_min (float): Min turn radius, m, or None to derive it
            name (str): Vessel name, or None to use the object name

        Returns:
            Vessel: New vessel
        """
        if r_min is None:
            rate = getattr(obj, "max_rot_rad_s", None)
            v = float(getattr(obj, "v", 0.0))
            r_min = (v / rate) if (rate not in (None, 0)) and v > 0 else R_MIN_M
        return cls(
            name if name is not None else getattr(obj, "name", "vessel"),
            obj.x,
            obj.y,
            obj.v,
            obj.theta,
            r_min,
        )

    @property
    def state(self):
        """Get the state as a tuple.

        Returns:
            tuple: State (x, y, v, heading)
        """
        return (self.x, self.y, self.v, self.theta)

    def __repr__(self):
        """Describe the vessel in plain text.

        Returns:
            str: Text description
        """
        return (
            f"Vessel({self.name}, ({self.x:.1f}, {self.y:.1f}) m, "
            f"{self.v:.2f} m/s, {np.degrees(self.theta):.1f} deg)"
        )


def _wrap(a):
    """Wrap an angle into [-pi, pi].

    Args:
        a (float): Angle, rad

    Returns:
        float: Wrapped angle, rad
    """
    if isinstance(a, float):
        return math.atan2(math.sin(a), math.cos(a))
    return np.arctan2(np.sin(a), np.cos(a))

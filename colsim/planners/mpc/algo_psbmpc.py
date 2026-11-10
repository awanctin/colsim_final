"""
Tengesdal, Rothmund, Basso, Schmidt-Didlaukies & Johansen (2024), "Obstacle intention awareness
in automatic collision avoidance: full scale experiments in confined waters", Field Robotics
4(1):211-245, doi 10.55417/fr.2024007.
"""

from __future__ import annotations

import ctypes
import math
import os

import numpy as np

_DEFAULT_LIB = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "..",
    "..",
    "third_party",
    "local",
    "lib",
    "libpsbmpc_bridge.so",
)
LIB_PATH = os.environ.get("COLSIM_PSBMPC_LIB", _DEFAULT_LIB)

L_SCALE = 30.0
T_SCALE = 6.0

LENGTH_M = 150.0
WIDTH_M = 25.0
T_U_S = 5.0 * T_SCALE
T_CHI_S = 5.0 * T_SCALE
R_A_M = 5.0 * L_SCALE
LOS_LD_M = 66.0 * L_SCALE
HORIZON_S = 120.0 * T_SCALE
DT_S = 1.0 * T_SCALE
T_TS_S = 60.0 * T_SCALE
D_SAFE_M = 4.5 * L_SCALE
D_DO_RELEVANT_M = 500.0 * L_SCALE
D_SO_RELEVANT_M = 150.0 * L_SCALE
T_COLL_S = 100.0 * T_SCALE
POS_VAR_M2 = 25.0 * L_SCALE**2
VEL_VAR_M2S2 = 0.025 * (L_SCALE / T_SCALE) ** 2
OBSTACLE_ABCD_M = (75.0, 75.0, 12.5, 12.5)

_lib = None


def _load():
    """Load the PSB-MPC C++ library once and set its function types.

    Returns:
        ctypes.CDLL: The loaded library
    """
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(os.path.abspath(LIB_PATH))
        lib.psb_create.restype = ctypes.c_void_p
        lib.psb_create.argtypes = [ctypes.c_double] * 15
        lib.psb_destroy.argtypes = [ctypes.c_void_p]
        lib.psb_seed.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong]
        dp = ctypes.POINTER(ctypes.c_double)
        lib.psb_step.argtypes = [
            ctypes.c_void_p,
            dp,
            ctypes.c_double,
            ctypes.c_double,
            dp,
            ctypes.c_int,
            dp,
            ctypes.c_int,
            ctypes.c_double,
            dp,
        ]
        _lib = lib
    return _lib


def _ptr(a):
    """Get a C double pointer to a numpy array.

    Args:
        a (ndarray): Array of floats

    Returns:
        ctypes.POINTER: Pointer to the data
    """
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def to_ned(x, y, theta):
    """Convert a pose from the simulator frame to North East Down.

    Args:
        x (float): East position, m
        y (float): North position, m
        theta (float): Heading from the x axis, rad

    Returns:
        tuple: North, east and course, rad
    """
    return y, x, math.remainder(math.pi / 2.0 - theta, 2.0 * math.pi)


class PSBMPC:

    def __init__(self, seed=0):
        """Create a PSB-MPC instance in the C++ library.

        Args:
            seed (int): Random seed
        """
        self._h = _load().psb_create(
            LENGTH_M,
            WIDTH_M,
            T_U_S,
            T_CHI_S,
            R_A_M,
            LOS_LD_M,
            HORIZON_S,
            DT_S,
            T_TS_S,
            D_SAFE_M,
            D_DO_RELEVANT_M,
            D_SO_RELEVANT_M,
            T_COLL_S,
            POS_VAR_M2,
            VEL_VAR_M2S2,
        )
        _lib.psb_seed(self._h, int(seed) & 0xFFFFFFFFFFFFFFFF)

    def __del__(self):
        """Free the C++ instance."""
        if getattr(self, "_h", None) and _lib is not None:
            _lib.psb_destroy(self._h)
            self._h = None

    def __deepcopy__(self, _memo):
        """Return a fresh instance, since the C++ handle cannot be copied.

        Args:
            _memo (dict): Unused

        Returns:
            PSBMPC: New instance
        """
        return PSBMPC()

    def decide(self, own, obstacles, start, goal, u_d, dt):
        """Run one PSB-MPC step and get the new heading and speed.

        Args:
            own (tuple): Own state (x, y, v, heading)
            obstacles (list): Targets (x, y, v, heading, id)
            start (tuple): Start waypoint (x, y), m
            goal (tuple): Goal waypoint (x, y), m
            u_d (float): Wanted speed, m/s
            dt (float): Time step, s

        Returns:
            tuple: Commanded heading, speed and course offset
        """
        n, e, chi = to_ned(own[0], own[1], own[3])
        own_ned = np.array([n, e, chi, own[2]], float)
        wps = np.array([[start[1], goal[1]], [start[0], goal[0]]], float)
        chi_d = math.atan2(goal[0] - own[0], goal[1] - own[1])

        cols = []
        for x, y, v, th, tid in obstacles:
            cols.append(
                [y, x, v * math.sin(th), v * math.cos(th), *OBSTACLE_ABCD_M, float(tid)]
            )
        obs = np.asarray(cols, float).reshape(-1, 9)
        out = np.zeros(2)
        _lib.psb_step(
            self._h,
            _ptr(own_ned),
            float(u_d),
            float(chi_d),
            _ptr(np.asfortranarray(wps)),
            2,
            _ptr(np.ascontiguousarray(obs)),
            len(cols),
            float(dt),
            _ptr(out),
        )
        u_opt, chi_opt = out
        theta_c = math.pi / 2.0 - (chi_d + chi_opt)
        return theta_c, float(u_opt), float(chi_opt)

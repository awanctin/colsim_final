from collections import namedtuple

import numpy as np

from ...certificate.duty import HORIZON_S
from ...simulator.experiment_constants import RuleMode

VOThreat = namedtuple("VOThreat", ["x", "y", "vx", "vy", "r_combined", "role"])
VOParams = namedtuple(
    "VOParams", ["detection_range", "time_horizon", "sweep_deg", "step_deg"]
)

DEFAULT_PARAMS = VOParams(
    detection_range=15000.0,
    time_horizon=HORIZON_S,
    sweep_deg=90.0,
    step_deg=2.0,
)


def _ang_diff(a, b):
    """Signed smallest angle a-b in (-pi, pi]."""
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def _in_collision_cone(os_pos, os_vel, threat, horizon):
    """Check whether a velocity leads to a close pass with a threat.

    Args:
        os_pos (ndarray): Own position, m
        os_vel (ndarray): Own velocity, m/s
        threat (object): Threat with position, velocity and r_combined
        horizon (float): Look ahead time, s

    Returns:
        bool: True if the closest distance is under r_combined
    """
    p = np.array([threat.x, threat.y]) - os_pos
    v_rel = os_vel - np.array([threat.vx, threat.vy])
    vv = float(v_rel @ v_rel)
    if vv < 1e-9:
        return float(np.linalg.norm(p)) < threat.r_combined
    t_star = float(p @ v_rel) / vv
    if t_star <= 0:
        return False
    t_star = min(t_star, horizon)
    dcpa = float(np.linalg.norm(-p + t_star * v_rel))
    return dcpa < threat.r_combined


def _worst_clearance(os_pos, os_vel, threats, horizon):
    """Find the smallest closest distance over all threats.

    Args:
        os_pos (ndarray): Own position, m
        os_vel (ndarray): Own velocity, m/s
        threats (list): Threats
        horizon (float): Look ahead time, s

    Returns:
        float: Smallest closest distance, m
    """
    worst = np.inf
    for t in threats:
        p = np.array([t.x, t.y]) - os_pos
        v_rel = os_vel - np.array([t.vx, t.vy])
        vv = float(v_rel @ v_rel)
        if vv < 1e-9:
            d = float(np.linalg.norm(p))
        else:
            t_star = max(0.0, min(float(p @ v_rel) / vv, horizon))
            d = float(np.linalg.norm(-p + t_star * v_rel))
        worst = min(worst, d)
    return worst


def select_heading(
    os_x,
    os_y,
    os_v,
    os_theta,
    goal_x,
    goal_y,
    threats,
    params=DEFAULT_PARAMS,
    rule_mode=RuleMode.COLREGS,
):
    """Choose the heading closest to the goal that avoids every collision cone.

    Args:
        os_x (float): Own x, m
        os_y (float): Own y, m
        os_v (float): Own speed, m/s
        os_theta (float): Own heading, rad
        goal_x (float): Goal x, m
        goal_y (float): Goal y, m
        threats (list): Threats with role and position
        params (object): VO parameters
        rule_mode (RuleMode): How COLREGs are applied

    Returns:
        float: Chosen heading, rad
    """
    goal_heading = np.arctan2(goal_y - os_y, goal_x - os_x)
    os_pos = np.array([os_x, os_y])

    active = [
        t
        for t in threats
        if t.role in ("give_way", "stand_on")
        and np.hypot(t.x - os_x, t.y - os_y) <= params.detection_range
    ]
    if not active:
        return goal_heading
    give_way = [t for t in active if t.role == "give_way"]
    if not give_way:
        return goal_heading

    sweep = np.deg2rad(params.sweep_deg)
    step = np.deg2rad(params.step_deg)
    n = int(sweep / step)
    candidates = [os_theta + k * step for k in range(-n, n + 1)]

    def feasible(h):
        """Check that a heading avoids every collision cone.

        Args:
            h (float): Heading, rad

        Returns:
            bool: True if no cone is hit
        """
        v = os_v * np.array([np.cos(h), np.sin(h)])
        return not any(
            _in_collision_cone(os_pos, v, t, params.time_horizon) for t in active
        )

    feas = [h for h in candidates if feasible(h)]

    starboard = [h for h in feas if _ang_diff(h, os_theta) <= np.deg2rad(1.0)]
    pool = feas if rule_mode is RuleMode.RELAXED else (starboard or feas)
    if pool:
        return min(pool, key=lambda h: abs(_ang_diff(h, goal_heading)))

    return max(
        candidates,
        key=lambda h: _worst_clearance(
            os_pos, os_v * np.array([np.cos(h), np.sin(h)]), active, params.time_horizon
        ),
    )

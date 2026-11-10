"""Kalami Heris, Mostapha (2023). Python Implementation of Path Planning and Obstacle Avoidance using PSO [Computer software].
Retrieved from https://github.com/smkalami/path-planning
"""

from .display.calcul_tools import *
from .display.draw import *
from ..planners.reactive.potential_fields import *
from ..planners.grid_search.algo_astar import *
from ..planners.metaheuristics.algo_aco import *
from ..planners.grid_search.algo_dstar_lite import *
from ..planners.metaheuristics.algo_pso import *
from ..planners.rrt_star.algo_rttstar import *
from .vessels.boat_pid import PID
from ..planners.rrt_star.colregs_planner import COLREGsPlanner
from ..certificate.escape_angle import R_MIN_M, Vessel as _DutyVessel
from ..certificate import duty as _duty
from ..certificate.monitor import _nb as _monitor_nb
from .experiment_constants import (
    CRI_W_DCPA,
    CRI_W_TCPA,
    ONE_NAUTICAL_MILE_M,
    T_HORIZON_S,
    WAYPOINT_ACCEPT_M,
    D_SAFE_MULT,
    DUTY_GATE_M,
    LATCH_DCPA_M,
    LATCH_TCPA_MIN_S,
    LATCH_TCPA_S,
    THREAT_FILTER_MULT,
    VISIBLE_RANGE_M,
    SPEED_MAX_MPS,
    SPEED_MIN_MPS,
    SURGE_ACCEL_MAX_MPS2,
    SURGE_DECEL_MAX_MPS2,
    MAKING_WAY_MPS,
    MAKING_WAY_BAND_MPS,
    OVERTAKING_DOUBT_BAND_DEG,
    MPS_TO_KNOT,
    NO_CLOSING_TCPA_S,
    R_VESSEL_M,
    RuleMode,
)
from ..control.speed_governor import SpeedGovernor, TargetInfo
from ..planners.reactive.algo_velocity_obstacle import select_heading, VOThreat
from ..planners.mpc import algo_sbmpc as _sbmpc
from ..planners.mpc import algo_psbmpc as _psbmpc
from ..planners.mpc import algo_rtampc as _rtampc
from ..planners.learned import algo_kras as _kras

from ..control import cbf_filter as _cbf
from ..certificate.escape_angle import classify_colregs as _classify, cpa as _cpa


import builtins
import numpy as np
import math
import os
import time
from typing import Tuple

try:
    if os.environ.get("COLSIM_MONITOR_NUMBA", "1") == "0":
        raise ImportError
    from ..control import _steer_nb
except Exception:
    _steer_nb = None

_OVERTAKING_ENTER = np.deg2rad(112.5) - np.deg2rad(OVERTAKING_DOUBT_BAND_DEG)
_HEAD_ON_ANGLE = np.deg2rad(6.0)
_RECIPROCAL_TOL = np.deg2rad(20)
_OFF_BOW_SKIP = np.deg2rad(60.0)
_WP_STALL_STEPS = 4


def _round1(x):
    """Round to one decimal, matching numpy, but faster for scalars.

    Args:
        x (float): Value

    Returns:
        float: Rounded value
    """
    if (type(x) is float or type(x) is np.float64) and abs(x) < 1e14:
        y = float(x) * 10.0
        return np.float64(math.copysign(float(builtins.round(y)), y) / 10.0)
    return np.round(x, 1)


def _cpa_xy(x_os, y_os, v_os_x, v_os_y, x_ts, y_ts, v_ts_x, v_ts_y):
    """Compute DCPA and TCPA between two vessels.

    Args:
        x_os (float): Own x, m
        y_os (float): Own y, m
        v_os_x (float): Own speed along x, m/s
        v_os_y (float): Own speed along y, m/s
        x_ts (float): Target x, m
        y_ts (float): Target y, m
        v_ts_x (float): Target speed along x, m/s
        v_ts_y (float): Target speed along y, m/s

    Returns:
        tuple: DCPA (m) and TCPA (s)
    """
    R = math.sqrt((x_ts - x_os) ** 2 + (y_ts - y_os) ** 2)
    v_x_rel = v_ts_x - v_os_x
    v_y_rel = v_ts_y - v_os_y
    relative_speed = math.sqrt(v_x_rel**2 + v_y_rel**2)
    psi_Vrel = math.atan2(v_ts_y - v_os_y, v_ts_x - v_os_x)
    psi_LOS = math.atan2(y_os - y_ts, x_os - x_ts)
    alpha = psi_LOS - psi_Vrel
    DCPA = np.float64(abs(R * math.sin(alpha)))
    if relative_speed != 0:
        TCPA = np.float64(R * math.cos(alpha) / relative_speed)
    else:
        TCPA = NO_CLOSING_TCPA_S
    return DCPA, TCPA


def _vel(obj):
    """Get the velocity of an object.

    Args:
        obj (SeaObject): Object

    Returns:
        tuple: Speed along x and y, m/s
    """
    v, th = float(obj.v), float(obj.theta)
    return v * math.cos(th), v * math.sin(th)


COMPLIANT_TAIL_ALGO = "RRT*"

NO_PORT_ROLES = ("HO", "CGW", "CSO")
_MEMORY_CODES = {"OGW": 1, "BO": 2}
PORT_VETO_GATE_M = DUTY_GATE_M
CONTROL_ALGOS = ("letter", "straight_goal", "SB-MPC", "PSB-MPC", "RTA-MPC", "KRAS")
RELAXED_SITUATION = "Relaxed"
RELEASE_LEG_M = 1000.0
EXTREMIS_DCPA_HULLS = 4.0
EXTREMIS_DCPA_DSAFE = 0.5
EXTREMIS_RANGE_M = ONE_NAUTICAL_MILE_M
RESUME_LEAD_S = 60.0
RESUME_HULLS = 4.0
REENTRY_WINDOW_S = 120.0
RESUME_OFFSET_FRAC = 0.25
RESUME_OFFSET_MIN_M = 500.0
OVERTAKE_RESUME_RANGE_FRAC = 0.25
OVERTAKE_RESUME_LEAD_FRAC = 0.10
OVERTAKE_RESUME_STRETCH = 1.5
OVERTAKE_RESUME_OFFSET_MIN_M = 1000.0
_ROLE_SITUATION = {
    "HO": "Head_on",
    "CGW": "Crossing_give_way",
    "CSO": "Crossing_stand_on",
    "OGW": "Overtaking_give_way",
    "BO": "Being_overtaken_stand_on",
}
LATCH_OPEN_M = 50.0
WAYPOINT_REACHED_M = 10.0


def Jφ0(p):
    """Compute the Jacobian of the reference field.

    Args:
        p (ndarray): Point (p1, p2)

    Returns:
        ndarray: 2 x 2 Jacobian
    """
    p1, p2 = p.flatten()
    return array(
        [
            [-3 * p1**2 - p2**2 + 1, -2 * p1 * p2 - 1],
            [-2 * p1 * p2 + 1, -3 * p2**2 - p1**2 + 1],
        ]
    )


def dφ(x, c, D):
    """Compute the change of the field along the heading.

    Args:
        x (ndarray): State (x, y, v, heading)
        c (ndarray): Field centre
        D (ndarray): Field scale matrix

    Returns:
        ndarray: Field derivative
    """
    p1, p2, v, θ = x.flatten()
    z = inv(D) @ array([[p1 - c[0, 0]], [p2 - c[1, 0]]])
    dv = D @ Jφ0(z) @ inv(D) @ array([[cos(θ)], [sin(θ)]])
    return dv.flatten()


def control(x, φ, c, D, k, r):
    """Compute the command that follows a vector field.

    Args:
        x (ndarray): State (x, y, v, heading)
        φ (callable): Field function
        c (ndarray): Field centre
        D (ndarray): Field scale matrix
        k (float): Field gain
        r (float): Field radius

    Returns:
        ndarray: Command (acceleration, turn rate)
    """

    dφ1, dφ2 = dφ(x, c, D)
    x, y, v, θ = x.flatten()
    φ1, φ2 = φ(x, y, c, D, k, r)
    u1 = 0
    u2 = -sawtooth(θ - arctan2(φ2, φ1)) - (φ2 * dφ1 - φ1 * dφ2) / ((φ1**2) + (φ2**2))
    return array([[u1], [u2]])


def path_smoother(path, map):
    """Remove grid path points that can be skipped.

    Args:
        path (list): Path cells
        map (ndarray): Occupancy grid

    Returns:
        list: Shorter path
    """
    new_path = [path[0]]
    for i in range(1, len(path) - 1):
        if not is_obstacle_between(new_path[-1], path[i + 1], map):
            continue
        new_path.append(path[i])
    new_path.append(path[-1])
    return new_path


def is_obstacle_between(p1, p2, map):
    """Check whether a grid line between two cells hits an obstacle.

    Args:
        p1 (tuple): First cell
        p2 (tuple): Second cell
        map (ndarray): Occupancy grid

    Returns:
        bool: True if blocked
    """
    x1, y1 = p1
    x2, y2 = p2

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while (x1, y1) != (x2, y2):
        if map[x1][y1] != 0:
            return True
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return False


class SeaObject:
    def __init__(
        self,
        mmsi,
        x,
        y,
        v,
        theta,
        algo,
        destination_distance,
        map_size=10000,
        sea_objects=None,
        goal=None,
        record_path=None,
    ):
        """Set the state, goal, rule settings and memory of a sea object.

        Args:
            mmsi (int): Vessel identifier
            x (float): Start x position, m
            y (float): Start y position, m
            v (float): Start speed, m/s
            theta (float): Start heading, rad
            algo (str): Name of the path planning algorithm
            destination_distance (float): Distance to the goal, m
            map_size (float): Half size of the map, m
            sea_objects (list): Other sea objects, or None
            goal (tuple): Goal position, m, or None
            record_path (str): Folder for saved plots, or None
        """
        self.mmsi = mmsi
        self.x = x
        self.y = y
        self.v = v
        self.theta = theta
        self.algo = algo
        self.destination_distance = destination_distance
        assert (
            destination_distance < 2 * map_size
        ), f"Destination distance {destination_distance} should be less than twice the map size {2 * map_size}."
        self.save_graph = False
        self.history = []
        self.collision_history = {}
        self.collision_risk = 0
        self.history_lenght = 1000
        self.collision_coordinates = []
        self.map_size = map_size
        if goal is None:
            goal = self.straight_goal(distance=destination_distance, map_size=map_size)
        assert goal is not None, "Goal cannot be None."
        self.phat = goal
        self.privilege = 0
        self.r = R_VESSEL_M
        self.cross_path = False
        self.final = 0
        self.set_goal(self.phat[0], self.phat[1])
        self.waypoints = []
        self.sea_objects = [] if sea_objects is None else sea_objects
        self.trajectory = [[self.x], [self.y]]
        self.planner = COLREGsPlanner()
        self.list_colregs = []
        self.record_path = record_path if record_path is not None else "plots"
        self.steps_count = 0
        self.paths = []
        self.in_collision = False
        self.calculate_colregs = None
        self.nb_replan = 0
        self.previous_path = None
        self.completed_waypoints = []

        self.nav_phase = "TRANSIT"
        self.avoidance_replan_counter = 0
        self.LOCAL_REPLAN_INTERVAL = 3
        self.letter_commit_mmsi = None
        self.letter_target_mmsi = None
        self.letter_role = None
        self._letter_ot_roles = {}
        self.letter_rule13d = True
        self.T_ALERT = LATCH_TCPA_S
        self.SAFE_DCPA_M = LATCH_DCPA_M
        self.SAFE_TCPA_MIN_S = LATCH_TCPA_MIN_S
        self.VISIBLE_RANGE_M = VISIBLE_RANGE_M
        self._max_rot_rad_s = None
        self.D_SAFE_MULT = D_SAFE_MULT
        self.giveway_action_dist_m = 3704.0
        self.giveway_min_apparent_deg = 15.0
        self.resume_waypoint = None
        self.emergency_avoidance = True
        self.avoidance_side = 0
        self.avoidance_ref_heading = None

        self.avoidance_ref_pos = None
        self.frozen_role = None
        self._giveway_role = None
        self._giveway_streak = 0
        self._resolve_counter = 0
        self.avoidance_entry_heading = None
        self._released = None
        self._released_age_s = 0.0
        self.K_COMMIT = 1
        self.colregs_clamp = False
        self.colregs_clamp_gate = None
        self.clamp_steps = 0
        self.clamp_infeasible_steps = 0
        self._clamp_context = None
        self.colregs_port_veto = algo not in CONTROL_ALGOS
        self.rule_mode = RuleMode.COLREGS
        self.safety_filter = _cbf.SafetyFilter.NONE
        self.cbf_active_steps = 0
        self.cbf_infeasible_steps = 0
        self._cbf_latch = {}
        self.port_veto_steps = 0
        self.giveway_floor_steps = 0
        self.standon_snap_steps = 0
        self.overtaking = {}
        self.N_RESOLVE = 5
        self.threat_ts = None
        self.heading_pid = PID(Kp=0.5, Ki=0.01, Kd=0.3)
        self.speed_pid = PID(Kp=0.1, Ki=0.001, Kd=0.05)
        self.verbose = False
        self.j_cascade = 0.0
        self._speed_gov: SpeedGovernor | None = None
        self.log_obligations = False
        self._obligations = []
        self._primary_threat = None

    @property
    def position(self):
        """Get the position as an array.

        Returns:
            ndarray: Position (x, y), m
        """
        return np.array([self.x, self.y])

    def __str__(self):
        """Describe the object state in plain text.

        Returns:
            str: Text description
        """
        return f"{self.name} at ({self.x:.2f},{self.y:.2f}), heading {math.degrees(self.theta):.2f}° at {self.v:.2f} m/s ({self.v * MPS_TO_KNOT:.2f} knots). collision coordinates = {self.collision_coordinates}, goal = ({self.goal_x:.2f},{self.goal_y:.2f}), algo = {self.algo})"

    def __repr__(self):
        """Describe the object state for debugging.

        Returns:
            str: Text description
        """
        return f"{self.name}(x: {self.x},y: {self.y},theta: {self.theta},v: {self.v},coll: {self.collision_coordinates}; goal: ({self.goal_x}, {self.goal_y}), algo: {self.algo})"

    def set_goal(self, gx, gy):
        """Set the goal and remember its direction for the sliding goal.

        Args:
            gx (float): Goal x, m
            gy (float): Goal y, m
        """
        self.phat = [gx, gy]
        self.goal_x, self.goal_y = gx, gy
        self._goal0 = (float(gx), float(gy))
        gdx, gdy = self._goal0[0] - float(self.x), self._goal0[1] - float(self.y)
        lookahead = math.hypot(gdx, gdy)
        self._goal_origin = (float(self.x), float(self.y))
        self._goal_dir = (gdx / lookahead, gdy / lookahead) if lookahead > 0 else None

    def _slide_goal(self):
        """Move the goal forward as the vessel moves along its route."""
        d = getattr(self, "_goal_dir", None)
        if d is None:
            return
        along = max(
            0.0,
            (float(self.x) - self._goal_origin[0]) * d[0]
            + (float(self.y) - self._goal_origin[1]) * d[1],
        )
        self.goal_x = self._goal0[0] + along * d[0]
        self.goal_y = self._goal0[1] + along * d[1]

    def straight_goal(self, distance=10000, map_size=10000):
        """Put the goal straight ahead of the vessel.

        Args:
            distance (float): Distance to the goal, m
            map_size (float): Half size of the map, m

        Returns:
            list: Goal point (x, y), m
        """
        assert (
            distance < 2 * map_size
        ), "Distance should be less than twice the map size."
        goal_x = self.x + distance * np.cos(self.theta)
        goal_y = self.y + distance * np.sin(self.theta)
        return [goal_x, goal_y]

    @property
    def max_rot_rad_s(self):
        """Get the max turn rate, from the speed if not set.

        Returns:
            float: Max turn rate, rad/s
        """
        if self._max_rot_rad_s is not None:
            return self._max_rot_rad_s
        return abs(float(self.v)) / R_MIN_M

    @max_rot_rad_s.setter
    def max_rot_rad_s(self, value):
        """Set the max turn rate.

        Args:
            value (float): Max turn rate, rad/s, or None to use the speed
        """
        self._max_rot_rad_s = None if value is None else float(value)

    @property
    def _rulebook_latch(self):
        """Check whether this vessel keeps its COLREGs roles over time.

        Returns:
            bool: True if roles are latched
        """
        return self.algo not in CONTROL_ALGOS

    def set_rule_mode(self, mode):
        """Set how COLREGs are applied and turn the port veto on or off.

        Args:
            mode (RuleMode): How COLREGs are applied
        """
        self.rule_mode = mode
        self.colregs_port_veto = (
            mode is RuleMode.COLREGS and self.algo not in CONTROL_ALGOS
        )

    @staticmethod
    def _nearest_in(intervals, x):
        """Find the value closest to x inside a list of intervals.

        Args:
            intervals (list): Pairs (low, high)
            x (float): Value

        Returns:
            float: Closest allowed value, or None
        """
        best = None
        for lo, hi in intervals:
            near = min(max(x, lo), hi)
            if best is None or abs(near - x) < abs(best - x):
                best = near
        return best

    def _lawful_yaw(self, yaw, dt):
        """Clamp the turn rate to the headings allowed by COLREGs.

        Args:
            yaw (float): Wanted turn rate, rad/s
            dt (float): Time step, s

        Returns:
            float: Allowed turn rate, rad/s
        """
        if not self.colregs_clamp or not self._clamp_context:
            return yaw
        others = [o for o in self._clamp_context if o is not self]
        if not others:
            return yaw
        try:
            os_v = _DutyVessel.from_sea_object(self, r_min=R_MIN_M)
            ts = [_DutyVessel.from_sea_object(o, r_min=R_MIN_M) for o in others]
            iv = _duty.lawful_intervals(
                os_v,
                ts,
                gate=self.colregs_clamp_gate,
                memory=[self.overtaking.get(o.mmsi) for o in others],
            )
        except Exception:
            return yaw
        if not iv:
            self.clamp_infeasible_steps += 1
            return yaw
        want = yaw * _duty.HORIZON_S
        got = self._nearest_in(iv, want)
        if got is None or abs(got - want) < 1e-12:
            return yaw
        self.clamp_steps += 1
        return got / _duty.HORIZON_S

    def _cbf_command(self, u):
        """Pass the command through the CBF safety filter.

        Args:
            u (ndarray): Command (acceleration, turn rate)

        Returns:
            ndarray: Filtered command
        """
        others = [o for o in (self._clamp_context or []) if o is not self]
        if not others:
            return u
        own = _DutyVessel.from_sea_object(self, r_min=R_MIN_M)
        targets = []
        for o in others:
            tv = _DutyVessel.from_sea_object(o)
            alpha = _cbf.latched_deflection(
                self._cbf_latch,
                o.mmsi,
                _cbf.ALPHA_D_RAD.get(_classify(own, tv), 0.0),
                closing=_cpa(own, tv)[0] > 0.0,
            )
            targets.append(((o.x, o.y, o.v, o.theta), alpha))
        rate = self.max_rot_rad_s or 0.0
        a_lo = 0.0 if self.v <= SPEED_MIN_MPS else SURGE_DECEL_MAX_MPS2
        a_hi = 0.0 if self.v >= SPEED_MAX_MPS else SURGE_ACCEL_MAX_MPS2
        safe, active, feasible = _cbf.filter_command(
            (self.x, self.y, self.v, self.theta),
            targets,
            (u[0][0], u[1][0]),
            ((a_lo, a_hi), (-rate, rate)),
        )
        self.cbf_active_steps += int(active)
        self.cbf_infeasible_steps += int(not feasible)
        return np.array([[safe[0]], [safe[1]]])

    def update(self, u, dt):
        """Apply a command and move the vessel one step.

        Args:
            u (ndarray): Command (acceleration, turn rate)
            dt (float): Time step, s
        """
        if self.safety_filter is _cbf.SafetyFilter.CBF:
            u = self._cbf_command(u)
        x, y, v, theta = self.x, self.y, self.v, self.theta
        self.x += dt * v * math.cos(theta)
        self.y += dt * v * math.sin(theta)
        self.v = v + dt * u[0][0]
        yaw = u[1][0]
        if self.max_rot_rad_s is not None:
            yaw = float(min(max(yaw, -self.max_rot_rad_s), self.max_rot_rad_s))
        yaw = self._lawful_yaw(yaw, dt)
        self.theta += dt * yaw
        self.theta = self._normalize_ang(self.theta)
        self.update_history(self.history_lenght)
        self.trajectory[0].append(self.x)
        self.trajectory[1].append(self.y)

    def get_state_vector(self):
        """Get the state and goal as a column vector.

        Returns:
            ndarray: 6 x 1 vector (x, y, v, heading, goal x, goal y)
        """
        x = self.x
        if type(x) is float or type(x) is np.float64:
            out = np.empty((6, 1))
            out[0, 0], out[1, 0], out[2, 0] = x, self.y, self.v
            out[3, 0], out[4, 0], out[5, 0] = self.theta, self.goal_x, self.goal_y
            return out
        return np.array(
            [[x], [self.y], [self.v], [self.theta], [self.goal_x], [self.goal_y]]
        )

    VALID_ALGOS = (
        "APF",
        "A*",
        "D*Lite",
        "ACO",
        "PSO",
        "RRT*",
        "VO",
        "straight_goal",
        "hybrid",
        "letter",
        "SB-MPC",
        "PSB-MPC",
        "RTA-MPC",
        "KRAS",
    )

    GIVE_WAY_SITUATIONS = ("Crossing_give_way", "Head_on", "Overtaking_give_way")
    STAND_ON_SITUATIONS = ("Crossing_stand_on", "Being_overtaken_stand_on")

    def _dispatch_move(
        self,
        record_data,
        sea_objects,
        mmsi_list,
        rules,
        table,
        ax,
        eps,
        s,
        k,
        dt,
        verbose=False,
    ):
        """Call the move method that matches the planning algorithm.

        Args:
            record_data (bool): Whether to store the trajectory
            sea_objects (list): All sea objects
            mmsi_list (list): Identifiers of all sea objects
            rules (list): Active rules table
            table (matplotlib.table.Table): Rules table display, or None
            ax (matplotlib.axes.Axes): Axes to draw on, or None
            eps (float): Safety margin added to the radius, m
            s (float): Half size of the map, m
            k (int): Current step index
            dt (float): Time step, s
            verbose (bool): Whether to print debug output

        Returns:
            list: Identifier, name, state vector, collision risk and history
        """
        self._clamp_context = sea_objects
        self.update_overtaking(sea_objects)
        self._slide_goal()
        algo = self.algo
        if algo == "APF":
            return self.move_apf(
                record_data, sea_objects, mmsi_list, rules, table, ax, eps, s, k, dt
            )
        if algo == "A*":
            return self.move_astar(
                record_data, sea_objects, mmsi_list, rules, table, ax, eps, s, k, dt
            )
        if algo == "D*Lite":
            return self.move_dstarl(
                record_data, sea_objects, mmsi_list, rules, table, ax, eps, s, k, dt
            )
        if algo == "ACO":
            return self.move_aco(
                record_data, sea_objects, mmsi_list, rules, table, ax, eps, s, k, dt
            )
        if algo == "PSO":
            return self.move_pso(
                record_data, sea_objects, mmsi_list, rules, table, ax, eps, s, k, dt
            )
        if algo == "RRT*":
            return self.move_rrtstar(
                record_data,
                sea_objects,
                mmsi_list,
                rules,
                table,
                ax,
                eps,
                s,
                k,
                dt,
                verbose,
            )
        if algo == "VO":
            return self.move_velocity_obstacle(
                record_data,
                sea_objects,
                mmsi_list,
                rules,
                table,
                ax,
                eps,
                s,
                k,
                dt,
                verbose,
            )
        if algo == "KRAS":
            return self.move_kras(
                record_data, sea_objects, mmsi_list, rules, table, ax, eps, s, k, dt
            )
        if algo == "RTA-MPC":
            return self.move_rtampc(
                record_data, sea_objects, mmsi_list, rules, table, ax, eps, s, k, dt
            )
        if algo == "PSB-MPC":
            return self.move_psbmpc(
                record_data, sea_objects, mmsi_list, rules, table, ax, eps, s, k, dt
            )
        if algo == "SB-MPC":
            return self.move_sbmpc(
                record_data, sea_objects, mmsi_list, rules, table, ax, eps, s, k, dt
            )
        if algo == "straight_goal":
            return self.move_straight_goal(
                record_data, sea_objects, mmsi_list, rules, table, ax, eps, s, k, dt
            )
        if algo == "hybrid":
            return self.move_hybrid(
                record_data,
                sea_objects,
                mmsi_list,
                rules,
                table,
                ax,
                eps,
                s,
                k,
                dt,
                verbose=verbose,
            )
        if algo == "letter":
            return self.move_letter(
                record_data,
                sea_objects,
                mmsi_list,
                rules,
                table,
                ax,
                eps,
                s,
                k,
                dt,
                verbose,
            )
        raise ValueError(
            f"Unknown algo {algo!r}. Valid algorithms: {', '.join(self.VALID_ALGOS)}."
        )

    def move_straight_apf(self):
        """Steer toward the goal with a simple vector field.

        Returns:
            ndarray: Command applied
        """
        vhat = array([[1], [1]])
        wp = vhat - 2 * (array([[self.x], [self.y]]) - self.phat)
        thetabar_p = arctan2(wp[1, 0], wp[0, 0])

        up = array([[0], [10 * arctan(tan(0.5 * (thetabar_p - self.theta)))]])
        self.update(up, 0.1)
        return up

    def avoid_collision_apf(
        self, record_data, obstacle, mmsi_list, rules, table, ax, eps, s, r, k
    ):
        """Steer around one obstacle with a vector field.

        Args:
            record_data (bool): Whether to store the trajectory
            obstacle (SeaObject): Obstacle to avoid
            mmsi_list (list): Identifiers of all sea objects
            rules (list): Active rules table
            table (matplotlib.table.Table): Rules table display, or None
            ax (matplotlib.axes.Axes): Axes to draw on, or None
            eps (float): Safety margin added to the radius, m
            s (float): Half size of the map, m
            r (float): Field radius, m
            k (int): Current step index

        Returns:
            ndarray: Command applied
        """
        px, py, pv, ptheta, _, _ = self.get_state_vector().flatten()
        qx, qy, qv, qtheta, _, _ = obstacle.get_state_vector().flatten()

        scalar_pdt = geo_scalar_prod(qv, pv, qtheta, ptheta)

        c = array([[qx], [qy]])
        D = array([[r, 0], [0, r]])

        if dist(array([[qx], [qy]]), array([[px], [py]])) < r + eps:
            if scalar_pdt >= 0:

                if (px < qx - eps) and self.cross_path:
                    φ = φrep
                    rule = 1

                elif (px > qx - eps) and self.cross_path:
                    φ = φrep
                    rule = 1

                elif py > qy + eps:
                    φ = φrep
                    rule = 1

                elif (py < qy + eps) and (px < qx):
                    φ = φcw
                    rule = 2

                elif (
                    (py < qy + eps)
                    and (px < qx)
                    and (self.goal_y < qy + eps)
                    and (self.goal_x > qx)
                ):
                    φ = φccw
                    self.cross_path = True
                    rule = 4

                elif (
                    (py < qy + eps)
                    and (px > qx)
                    and (self.phat[1] < qy + eps)
                    and (self.phat[0] < qx)
                ):
                    φ = φcw
                    self.cross_path = True
                    rule = 3

                else:
                    φ = φccw
                    rule = 3

                up = control(array([[px], [py], [pv], [ptheta]]), φ, c, D, k, r)
                if not record_data:
                    for row in arange(len(rules)):
                        table[row, mmsi_list.index(self.mmsi) + 1].set_facecolor(
                            "white"
                        )
                    table[rule, mmsi_list.index(self.mmsi) + 1].set_facecolor("green")
                    draw_field_around_c_new(ax, φ, -s, s, -s, s, 0.9, c, D, k, r)

            else:
                if py > qy - eps:
                    φ = φccw
                    rule = 4
                    up = control(array([[px], [py], [pv], [ptheta]]), φ, c, D, k, r)

                elif (
                    (py > qy - eps)
                    and (px > qx)
                    and (scalar_pdt < abs(qv * pv) * cos(2.5))
                ):
                    up = array([[0], [0]])
                    rule = 4

                elif py > qy - eps and px > qx and scalar_pdt > abs(qv * pv) * cos(2.5):
                    φ = φccw
                    rule = 4
                    up = control(array([[px], [py], [pv], [ptheta]]), φ, c, D, k, r)

                else:
                    φ = φrep
                    rule = 1
                    up = control(array([[px], [py], [pv], [ptheta]]), φ, c, D, k, r)

        if not record_data:
            for row in arange(len(rules)):
                table[row, mmsi_list.index(self.mmsi) + 1].set_facecolor("white")
            table[rule, mmsi_list.index(self.mmsi) + 1].set_facecolor("green")
            draw_field_around_c_new(ax, φ, -s, s, -s, s, 0.9, c, D, k, r)
        return up

    def move_apf(
        self, record_data, sea_objects, mmsi_list, rules, table, ax, eps, s, k, dt
    ):
        """Move one step with the potential field method.

        Args:
            record_data (bool): Whether to store the trajectory
            sea_objects (list): All sea objects
            mmsi_list (list): Identifiers of all sea objects
            rules (list): Active rules table
            table (matplotlib.table.Table): Rules table display, or None
            ax (matplotlib.axes.Axes): Axes to draw on, or None
            eps (float): Safety margin added to the radius, m
            s (float): Half size of the map, m
            k (int): Current step index
            dt (float): Time step, s

        Returns:
            list: Identifier, name, state vector, collision risk and history
        """
        in_collision = False

        for other_object in sea_objects:

            if self != other_object:
                if (
                    dist(
                        array([[other_object.x], [other_object.y]]),
                        array([[self.x], [self.y]]),
                    )
                    < max(self.r, other_object.r) + eps
                ):
                    self.collision_risk = 1
                    other_object.collision_risk = 1
                    self.collision_coordinates.append([self.x, self.y])
                    self.collision_coordinates.append([other_object.x, other_object.y])
                    self.save_graph = True

                    if other_object.mmsi not in self.collision_history:
                        self.collision_history[other_object.mmsi] = []
                        self.collision_history[other_object.mmsi].append(
                            [other_object.history]
                        )
                    if self.privilege <= other_object.privilege:
                        up = self.avoid_collision_apf(
                            record_data,
                            other_object,
                            mmsi_list,
                            rules,
                            table,
                            ax,
                            eps,
                            s,
                            max(self.r, other_object.r),
                            k,
                        )
                        in_collision = True
                        self.save_graph = True

        if (
            not in_collision
            and sqrt((self.phat[0] - self.x) ** 2 + (self.phat[1] - self.y) ** 2) > 1
        ):
            self.collision_risk = 0
            other_object.collision_risk = 0
            up = self.move_straight_apf()

        if sqrt((self.phat[0] - self.x) ** 2 + (self.phat[1] - self.y) ** 2) < 1:
            up = array([[0], [0]])
            if self.final == 0:
                self.final = 1

        else:
            self.update(up, dt)
        return [
            self.mmsi,
            self.name,
            self.get_state_vector(),
            self.collision_risk,
            self.collision_history,
        ]

    def create_grid(
        self, sea_objects, mmsi_list, rules, table, ax, eps, s, k, dt, grid
    ):
        """Mark the cells blocked by other objects and find start and goal cells.

        Args:
            sea_objects (list): All sea objects
            mmsi_list (list): Identifiers of all sea objects
            rules (list): Active rules table
            table (matplotlib.table.Table): Rules table display, or None
            ax (matplotlib.axes.Axes): Axes to draw on, or None
            eps (float): Safety margin added to the radius, m
            s (float): Half size of the map, m
            k (int): Current step index
            dt (float): Time step, s
            grid (list): Occupancy grid

        Returns:
            tuple: Grid, start cell and goal cell
        """
        grid_w, grid_h = len(grid[0]), len(grid)

        for row in range(grid_w):
            for col in range(grid_h):
                for other_object in sea_objects:
                    if self != other_object:
                        xx = (grid_w - row) * (s + s) / grid_w - s
                        yy = col * (s + s) / grid_h - s
                        if (
                            dist(
                                array([[other_object.x], [other_object.y]]),
                                array([[xx], [yy]]),
                            )
                            < self.r + eps
                        ):
                            grid[row][col] = 1

        start = (
            int(-(np.round(self.x) + s) * grid_w / (s + s) + grid_w),
            int((np.round(self.y) + s) * grid_h / (s + s)),
        )
        end = (
            int(-(np.round(self.goal_x) + s) * grid_w / (s + s) + grid_w),
            int((np.round(self.goal_y) + s) * grid_h / (s + s)),
        )
        show = False
        if show:
            print("start (3) is at", start)
            print("end (4) is at", end)
            grid[start[0], start[1]] = "3"
            grid[end[0], end[1]] = "4"
            print(grid)
            grid[start[0], start[1]] = "0"
            grid[end[0], end[1]] = "0"

        return grid, start, end

    def go_to(self, path, s, grid):
        """Steer toward the next cell of a grid path.

        Args:
            path (list): Path cells
            s (float): Half size of the map, m
            grid (list): Occupancy grid

        Returns:
            ndarray: Command applied
        """
        grid_w, grid_h = len(grid[0]), len(grid)
        next_point = (grid_w - path[1][0]) * (s + s) / grid_w - s, path[1][1] * (
            s + s
        ) / grid_h - s

        thetabar = arctan2(next_point[1] - self.y, next_point[0] - self.x)

        error = arctan2(sin(thetabar - self.theta), cos(thetabar - self.theta))

        if error > np.pi / 4:
            u2 = 1
        elif error < -np.pi / 4:
            u2 = -1
        else:
            u2 = 0.5 * error

        up = array([[0], [u2]])
        return up

    def move_astar(
        self, record_data, sea_objects, mmsi_list, rules, table, ax, eps, s, k, dt
    ):
        """Move one step with A*.

        Args:
            record_data (bool): Whether to store the trajectory
            sea_objects (list): All sea objects
            mmsi_list (list): Identifiers of all sea objects
            rules (list): Active rules table
            table (matplotlib.table.Table): Rules table display, or None
            ax (matplotlib.axes.Axes): Axes to draw on, or None
            eps (float): Safety margin added to the radius, m
            s (float): Half size of the map, m
            k (int): Current step index
            dt (float): Time step, s

        Returns:
            list: Identifier, name, state vector, collision risk and history
        """
        in_collision = False

        for other_object in sea_objects:
            if self != other_object:
                if (
                    dist(
                        array([[other_object.x], [other_object.y]]),
                        array([[self.x], [self.y]]),
                    )
                    < max(self.r, other_object.r) + eps
                ):
                    self.collision_risk = 1
                    other_object.collision_risk = 1
                    self.collision_coordinates.append([self.x, self.y])
                    self.collision_coordinates.append([other_object.x, other_object.y])
                    self.save_graph = True

                    if other_object.mmsi not in self.collision_history:
                        self.collision_history[other_object.mmsi] = []
                        self.collision_history[other_object.mmsi].append(
                            [other_object.history]
                        )
                    in_collision = True
        if len(self.waypoints) == 0 or (
            self.steps_count % 10 == 0
            and sqrt((self.goal_x - self.x) ** 2 + (self.goal_y - self.y) ** 2) > 1
        ):
            self.waypoints, _, _ = self.planner.enhanced_colreg_planning(
                self,
                sea_objects,
                [self.map_size, self.map_size],
                record_path=self.record_path,
                algo="A*",
            )

        if sqrt((self.goal_x - self.x) ** 2 + (self.goal_y - self.y) ** 2) < 1:
            up = array([[0], [0]])
            if self.final == 0:
                self.final = 1
                self.update(up, dt)

        elif len(self.waypoints) != 0:
            x0, y0 = self.waypoints[0]
            if sqrt((self.x - x0) ** 2 + (self.y - y0) ** 2) < 1:
                self.waypoints.pop(0)
            self.waypoint_following(self.waypoints, dt, speed_setpoint=self.v)

        if not in_collision:
            self.collision_risk = 0
            self.save_graph = False

        return [
            self.mmsi,
            self.name,
            self.get_state_vector(),
            self.collision_risk,
            self.collision_history,
        ]

    def move_aco(
        self, record_data, sea_objects, mmsi_list, rules, table, ax, eps, s, k, dt
    ):
        """Move one step with ant colony optimisation.

        Args:
            record_data (bool): Whether to store the trajectory
            sea_objects (list): All sea objects
            mmsi_list (list): Identifiers of all sea objects
            rules (list): Active rules table
            table (matplotlib.table.Table): Rules table display, or None
            ax (matplotlib.axes.Axes): Axes to draw on, or None
            eps (float): Safety margin added to the radius, m
            s (float): Half size of the map, m
            k (int): Current step index
            dt (float): Time step, s

        Returns:
            list: Identifier, name, state vector, collision risk and history
        """
        in_collision = False

        for other_object in sea_objects:
            if self != other_object:
                if (
                    dist(
                        array([[other_object.x], [other_object.y]]),
                        array([[self.x], [self.y]]),
                    )
                    < max(self.r, other_object.r) + eps
                ):
                    self.collision_risk = 1
                    other_object.collision_risk = 1
                    self.collision_coordinates.append([self.x, self.y])
                    self.collision_coordinates.append([other_object.x, other_object.y])
                    self.save_graph = True

                    if other_object.mmsi not in self.collision_history:
                        self.collision_history[other_object.mmsi] = []
                        self.collision_history[other_object.mmsi].append(
                            [other_object.history]
                        )
                    in_collision = True

        step = 2
        grid_w, grid_h = int(2 * s / step), int(2 * s / step)
        Obstacles = []
        grid = np.zeros((int(2 * s / step), int(2 * s / step)))

        for row in range(grid_w):
            for col in range(grid_h):
                for other_object in sea_objects:
                    if self != other_object:
                        xx = (grid_w - row) * (s + s) / grid_w - s
                        yy = col * (s + s) / grid_h - s
                        if (
                            dist(
                                array([[other_object.x], [other_object.y]]),
                                array([[xx], [yy]]),
                            )
                            < self.r + eps
                        ):
                            Obstacles.append((row, col))
                            grid[row][col] = 1

        start = (
            int(-(np.round(self.x) + s) * grid_w / (s + s) + grid_w),
            int((np.round(self.y) + s) * grid_h / (s + s)),
        )
        end = (
            int(-(np.round(self.goal_x) + s) * grid_w / (s + s) + grid_w),
            int((np.round(self.goal_y) + s) * grid_h / (s + s)),
        )

        aco_pdg = ACO_PDG(
            grid_w,
            grid_h,
            Obstacles,
            ant_count=20,
            step_count=100,
            alpha=1.1,
            beta=12,
            gamma=0.02,
            evaporation_rate=0.5,
            start=start,
            end=end,
            brushfire_iter=5,
            colony_iter=20,
        )

        heuristic_grid = aco_pdg.heuristic_grid
        path1 = aco_pdg.move_ant()
        path2 = aco_pdg.geometric_path_optimizer(path1)
        o = aco_pdg.run_ant_colony_noshow(10)

        if not o:

            return [
                self.mmsi,
                self.name,
                self.get_state_vector(),
                self.collision_risk,
                self.collision_history,
            ]
        path = o[0][1]

        if path == None:
            return [
                self.mmsi,
                self.name,
                self.get_state_vector(),
                self.collision_risk,
                self.collision_history,
            ]

        if path[0] == path[-1]:
            up = array([[0], [0]])
            if self.final == 0:
                self.final = 1

        else:
            path = path_smoother(path, grid)
            next_point = (grid_w - path[1][0]) * (s + s) / grid_w - s, path[1][1] * (
                s + s
            ) / grid_h - s
            thetabar = arctan2(next_point[1] - self.y, next_point[0] - self.x)
            error = arctan2(sin(thetabar - self.theta), cos(thetabar - self.theta))

            if error > np.pi / 4:
                u2 = 1
            elif error < -np.pi / 4:
                u2 = -1
            else:
                u2 = 0.5 * error

            up = array([[0], [u2]])
            self.update(up, dt)

        if not in_collision:
            self.collision_risk = 0
            self.save_graph = False

        return [
            self.mmsi,
            self.name,
            self.get_state_vector(),
            self.collision_risk,
            self.collision_history,
        ]

    def init_dstarlite(
        self, sea_objects, mmsi_list, rules, table, ax, eps, s, k, dt, grid
    ):
        """Build the grid and the obstacle lists for D* Lite.

        Args:
            sea_objects (list): All sea objects
            mmsi_list (list): Identifiers of all sea objects
            rules (list): Active rules table
            table (matplotlib.table.Table): Rules table display, or None
            ax (matplotlib.axes.Axes): Axes to draw on, or None
            eps (float): Safety margin added to the radius, m
            s (float): Half size of the map, m
            k (int): Current step index
            dt (float): Time step, s
            grid (list): Occupancy grid

        Returns:
            tuple: Grid, obstacle x, obstacle y, start cell and goal cell
        """

        grid_w, grid_h = len(grid[0]), len(grid)
        ox, oy = [], []
        s = s * 2
        for i in range(-s, s + 1):
            ox.append(i)
            oy.append(s)
            ox.append(s)
            oy.append(i)
            ox.append(i)
            oy.append(-s)
            ox.append(-s)
            oy.append(i)

        for row in range(grid_w):
            for col in range(grid_h):
                for other_object in sea_objects:
                    if self != other_object:
                        xx = (grid_w - row) * (s + s) / grid_w - s
                        yy = col * (s + s) / grid_h - s
                        if (
                            dist(
                                array([[other_object.x], [other_object.y]]),
                                array([[xx], [yy]]),
                            )
                            < self.r + eps
                        ):
                            grid[row][col] = 1
                            ox.append(row)
                            oy.append(col)

        start = (
            int(-(np.round(self.x) + s) * grid_w / (s + s) + grid_w),
            int((np.round(self.y) + s) * grid_h / (s + s)),
        )
        end = (
            int(-(np.round(self.goal_x) + s) * grid_w / (s + s) + grid_w),
            int((np.round(self.goal_y) + s) * grid_h / (s + s)),
        )

        return grid, ox, oy, start, end

    def move_dstarl(
        self, record_data, sea_objects, mmsi_list, rules, table, ax, eps, s, k, dt
    ):
        """Move one step with D* Lite.

        Args:
            record_data (bool): Whether to store the trajectory
            sea_objects (list): All sea objects
            mmsi_list (list): Identifiers of all sea objects
            rules (list): Active rules table
            table (matplotlib.table.Table): Rules table display, or None
            ax (matplotlib.axes.Axes): Axes to draw on, or None
            eps (float): Safety margin added to the radius, m
            s (float): Half size of the map, m
            k (int): Current step index
            dt (float): Time step, s

        Returns:
            list: Identifier, name, state vector, collision risk and history
        """
        in_collision = False

        for other_object in sea_objects:
            if self != other_object:
                if (
                    dist(
                        array([[other_object.x], [other_object.y]]),
                        array([[self.x], [self.y]]),
                    )
                    < max(self.r, other_object.r) + eps
                ):
                    self.collision_risk = 1
                    other_object.collision_risk = 1
                    self.collision_coordinates.append([self.x, self.y])
                    self.collision_coordinates.append([other_object.x, other_object.y])
                    self.save_graph = True

                    if other_object.mmsi not in self.collision_history:
                        self.collision_history[other_object.mmsi] = []
                        self.collision_history[other_object.mmsi].append(
                            [other_object.history]
                        )
                    in_collision = True
        if len(self.waypoints) == 0:
            step = 1
            empty_grid = np.zeros((int(2 * s / step), int(2 * s / step)))
            grid, ox, oy, start, end = self.init_dstarlite(
                sea_objects, mmsi_list, rules, table, ax, eps, s, k, dt, empty_grid
            )

            obstacles = []
            for other_object in sea_objects:
                if self != other_object:
                    obstacles.append((other_object.x, other_object.y, other_object.r))
            dstarlite = DStarLite(ox, oy)
            path = dstarlite.dstarl(
                Node(x=start[0], y=start[1]), Node(x=end[0], y=end[1])
            )
            self.waypoints = path

        path = self.waypoints
        if path[0] == path[-1]:
            up = array([[0], [0]])
            if self.final == 0:
                self.final = 1

        else:
            up = self.go_to(path, s, grid)
            self.update(up, dt)

        if not in_collision:
            self.collision_risk = 0
            self.save_graph = False

        return [
            self.mmsi,
            self.name,
            self.get_state_vector(),
            self.collision_risk,
            self.collision_history,
        ]

    def move_pso(
        self, record_data, sea_objects, mmsi_list, rules, table, ax, eps, s, k, dt
    ):
        """Move one step with particle swarm optimisation.

        Args:
            record_data (bool): Whether to store the trajectory
            sea_objects (list): All sea objects
            mmsi_list (list): Identifiers of all sea objects
            rules (list): Active rules table
            table (matplotlib.table.Table): Rules table display, or None
            ax (matplotlib.axes.Axes): Axes to draw on, or None
            eps (float): Safety margin added to the radius, m
            s (float): Half size of the map, m
            k (int): Current step index
            dt (float): Time step, s

        Returns:
            list: Identifier, name, state vector, collision risk and history
        """
        in_collision = False

        for other_object in sea_objects:
            if self != other_object:
                if (
                    dist(
                        array([[other_object.x], [other_object.y]]),
                        array([[self.x], [self.y]]),
                    )
                    < max(self.r, other_object.r) + eps
                ):
                    self.collision_risk = 1
                    other_object.collision_risk = 1
                    self.collision_coordinates.append([self.x, self.y])
                    self.collision_coordinates.append([other_object.x, other_object.y])
                    self.save_graph = True

                    if other_object.mmsi not in self.collision_history:
                        self.collision_history[other_object.mmsi] = []
                        self.collision_history[other_object.mmsi].append(
                            [other_object.history]
                        )
                    in_collision = True
        if len(self.waypoints) == 0:
            start = (self.x, self.y)
            goal = (self.goal_x, self.goal_y)
            limits = [-s, s, -s, s]
            layout = PathPlanning(start, goal, limits)

            obstacles = []
            for other_object in sea_objects:
                if self != other_object:
                    layout.add_circle(
                        x=other_object.x,
                        y=other_object.y,
                        r=other_object.r + eps,
                        Kv=100,
                    )
                    obstacles.append((other_object.x, other_object.y, other_object.r))

            nRun = 2
            nPts = 3
            d = 30
            nPop = 50
            epochs = 50
            f_interp = "linear"
            Xinit = None

            t0 = time.time()
            layout.optimize(
                nPts=nPts,
                ns=1 + (nPts + 1) * d,
                nPop=nPop,
                epochs=epochs,
                f_interp=f_interp,
                Xinit=Xinit,
            )
            t1 = time.time()
            self.execution_time = t1 - t0
            path = [
                [layout.sol[3][0][i], layout.sol[4][0][i]]
                for i in range(len(layout.sol[3][0]))
            ]
            layout.plot_path()
            self.waypoints = path

        if sqrt((self.goal_x - self.x) ** 2 + (self.goal_y - self.y) ** 2) < 1:
            up = array([[0], [0]])
            if self.final == 0:
                self.final = 1
                self.update(up, dt)

        elif len(self.waypoints) != 0:
            x0, y0 = self.waypoints[0]
            if sqrt((self.x - x0) ** 2 + (self.y - y0) ** 2) < 1:
                self.waypoints.pop(0)
            self.waypoint_following(self.waypoints, dt, speed_setpoint=10)

        if not in_collision:
            self.collision_risk = 0
            self.save_graph = False

        return [
            self.mmsi,
            self.name,
            self.get_state_vector(),
            self.collision_risk,
            self.collision_history,
        ]

    def move_straight_goal(
        self, record_data, sea_objects, mmsi_list, rules, table, ax, eps, s, k, dt
    ):
        """Steer straight to the goal and stop on arrival.

        Args:
            record_data (bool): Whether to store the trajectory
            sea_objects (list): All sea objects
            mmsi_list (list): Identifiers of all sea objects
            rules (list): Active rules table
            table (matplotlib.table.Table): Rules table display, or None
            ax (matplotlib.axes.Axes): Axes to draw on, or None
            eps (float): Safety margin added to the radius, m
            s (float): Half size of the map, m
            k (int): Current step index
            dt (float): Time step, s

        Returns:
            list: Identifier, name, state vector, collision risk and history
        """
        arrival_radius = max(self.v * dt, 150.0)
        dist_to_goal = sqrt((self.goal_x - self.x) ** 2 + (self.goal_y - self.y) ** 2)
        if dist_to_goal < arrival_radius or self.final:
            self.v = 0.0
            self.update(array([[0], [0]]), dt)
            self.final = 1
        else:
            self._steer_to_point([self.goal_x, self.goal_y], dt)
        return [
            self.mmsi,
            self.name,
            self.get_state_vector(),
            self.collision_risk,
            self.collision_history,
        ]

    def move_sbmpc(
        self, record_data, sea_objects, mmsi_list, rules, table, ax, eps, s, k, dt
    ):
        """Move one step with SB-MPC.

        Args:
            record_data (bool): Whether to store the trajectory
            sea_objects (list): All sea objects
            mmsi_list (list): Identifiers of all sea objects
            rules (list): Active rules table
            table (matplotlib.table.Table): Rules table display, or None
            ax (matplotlib.axes.Axes): Axes to draw on, or None
            eps (float): Safety margin added to the radius, m
            s (float): Half size of the map, m
            k (int): Current step index
            dt (float): Time step, s

        Returns:
            list: Identifier, name, state vector, collision risk and history
        """
        arrival_radius = max(self.v * dt, 150.0)
        if (
            math.hypot(self.goal_x - self.x, self.goal_y - self.y) < arrival_radius
            or self.final
        ):
            self.v = 0.0
            self.update(np.array([[0.0], [0.0]]), dt)
            self.final = 1
            return [
                self.mmsi,
                self.name,
                self.get_state_vector(),
                self.collision_risk,
                self.collision_history,
            ]

        if getattr(self, "_sbmpc_ud", None) is None:
            self._sbmpc_ud = float(self.v)
            self._sbmpc_last = (0.0, 1.0)
        chi_d = math.atan2(self.goal_y - self.y, self.goal_x - self.x)
        obstacles = [(o.x, o.y, o.v, o.theta) for o in sea_objects if o is not self]
        chi_m, p_factor, _cost = _sbmpc.best_behaviour(
            (self.x, self.y, self.v, self.theta),
            obstacles,
            chi_d,
            self._sbmpc_ud,
            self._sbmpc_last,
            v_floor=SPEED_MIN_MPS,
        )
        self._sbmpc_last = (chi_m, p_factor)

        err = math.remainder(chi_d + chi_m - float(self.theta), 2.0 * math.pi)
        yaw = min(max(0.2 / dt * err, -np.pi / 4), np.pi / 4)
        v_cmd = max(self._sbmpc_ud * p_factor, SPEED_MIN_MPS)
        acc = min(
            max((v_cmd - self.v) / dt, SURGE_DECEL_MAX_MPS2), SURGE_ACCEL_MAX_MPS2
        )
        self.update(np.array([[acc], [yaw]]), dt)
        return [
            self.mmsi,
            self.name,
            self.get_state_vector(),
            self.collision_risk,
            self.collision_history,
        ]

    def move_rtampc(
        self, record_data, sea_objects, mmsi_list, rules, table, ax, eps, s, k, dt
    ):
        """Move one step with RTA-MPC.

        Args:
            record_data (bool): Whether to store the trajectory
            sea_objects (list): All sea objects
            mmsi_list (list): Identifiers of all sea objects
            rules (list): Active rules table
            table (matplotlib.table.Table): Rules table display, or None
            ax (matplotlib.axes.Axes): Axes to draw on, or None
            eps (float): Safety margin added to the radius, m
            s (float): Half size of the map, m
            k (int): Current step index
            dt (float): Time step, s

        Returns:
            list: Identifier, name, state vector, collision risk and history
        """
        arrival_radius = max(self.v * dt, 150.0)
        if (
            math.hypot(self.goal_x - self.x, self.goal_y - self.y) < arrival_radius
            or self.final
        ):
            self.v = 0.0
            self.update(np.array([[0.0], [0.0]]), dt)
            self.final = 1
            return [
                self.mmsi,
                self.name,
                self.get_state_vector(),
                self.collision_risk,
                self.collision_history,
            ]

        if getattr(self, "_rtampc", None) is None:
            self._rtampc = _rtampc.RTAMPC()
            self._rtampc_start = (float(self.x), float(self.y))
        rate = self.max_rot_rad_s or 0.0
        err = math.remainder(
            math.atan2(self.goal_y - self.y, self.goal_x - self.x) - float(self.theta),
            2.0 * math.pi,
        )
        u_nom = (0.0, min(max(0.2 / dt * err, -rate), rate))
        a_lo = 0.0 if self.v <= SPEED_MIN_MPS else SURGE_DECEL_MAX_MPS2
        others = [(o.x, o.y, o.v, o.theta) for o in sea_objects if o is not self]
        (acc, yaw), _mode = self._rtampc.decide(
            (self.x, self.y, self.v, self.theta),
            others,
            (self.goal_x, self.goal_y),
            self._rtampc_start,
            u_nom,
            rate,
            (a_lo, SURGE_ACCEL_MAX_MPS2),
        )
        self.update(np.array([[acc], [yaw]]), dt)
        return [
            self.mmsi,
            self.name,
            self.get_state_vector(),
            self.collision_risk,
            self.collision_history,
        ]

    def move_kras(
        self, record_data, sea_objects, mmsi_list, rules, table, ax, eps, s, k, dt
    ):
        """Move one step with the KRAS policy and its shield.

        Args:
            record_data (bool): Whether to store the trajectory
            sea_objects (list): All sea objects
            mmsi_list (list): Identifiers of all sea objects
            rules (list): Active rules table
            table (matplotlib.table.Table): Rules table display, or None
            ax (matplotlib.axes.Axes): Axes to draw on, or None
            eps (float): Safety margin added to the radius, m
            s (float): Half size of the map, m
            k (int): Current step index
            dt (float): Time step, s

        Returns:
            list: Identifier, name, state vector, collision risk and history
        """
        arrival_radius = max(self.v * dt, 150.0)
        if (
            math.hypot(self.goal_x - self.x, self.goal_y - self.y) < arrival_radius
            or self.final
        ):
            self.v = 0.0
            self.update(np.array([[0.0], [0.0]]), dt)
            self.final = 1
            return [
                self.mmsi,
                self.name,
                self.get_state_vector(),
                self.collision_risk,
                self.collision_history,
            ]

        if getattr(self, "_kras", None) is None:
            self._kras = _kras.Shield()
            self._kras_start = (float(self.x), float(self.y))
            self._kras_control = (0.0, 0.0)
            self._kras_range = {}
            self._kras_k = 0
        ego = (float(self.x), float(self.y), float(self.v), float(self.theta))
        others = {
            o.mmsi: (float(o.x), float(o.y), float(o.v), float(o.theta))
            for o in sea_objects
            if o is not self
        }
        obs = _kras.observe(
            ego,
            self._kras_control,
            self._kras_start,
            (self.goal_x, self.goal_y),
            self._kras_k,
            others,
            self._kras_range,
        )
        mask = self._kras.mask(ego, others)
        action, _ = _kras.policy().predict(obs, action_masks=mask, deterministic=True)
        acc, yaw = self._kras.control(int(action), ego, others)
        acc = min(
            max(acc, (SPEED_MIN_MPS - self.v) / dt), (SPEED_MAX_MPS - self.v) / dt
        )
        self._kras_control = (acc, yaw)
        self._kras_k += 1
        self.update(np.array([[acc], [yaw]]), dt)
        return [
            self.mmsi,
            self.name,
            self.get_state_vector(),
            self.collision_risk,
            self.collision_history,
        ]

    def move_psbmpc(
        self, record_data, sea_objects, mmsi_list, rules, table, ax, eps, s, k, dt
    ):
        """Move one step with PSB-MPC.

        Args:
            record_data (bool): Whether to store the trajectory
            sea_objects (list): All sea objects
            mmsi_list (list): Identifiers of all sea objects
            rules (list): Active rules table
            table (matplotlib.table.Table): Rules table display, or None
            ax (matplotlib.axes.Axes): Axes to draw on, or None
            eps (float): Safety margin added to the radius, m
            s (float): Half size of the map, m
            k (int): Current step index
            dt (float): Time step, s

        Returns:
            list: Identifier, name, state vector, collision risk and history
        """
        arrival_radius = max(self.v * dt, 150.0)
        if (
            math.hypot(self.goal_x - self.x, self.goal_y - self.y) < arrival_radius
            or self.final
        ):
            self.v = 0.0
            self.update(np.array([[0.0], [0.0]]), dt)
            self.final = 1
            return [
                self.mmsi,
                self.name,
                self.get_state_vector(),
                self.collision_risk,
                self.collision_history,
            ]

        if getattr(self, "_psbmpc", None) is None:
            seed = (
                hash(
                    (
                        np.round(float(self.x), 3),
                        np.round(float(self.y), 3),
                        np.round(float(self.theta), 6),
                    )
                )
                & 0x7FFFFFFFFFFFFFFF
            )
            self._psbmpc = _psbmpc.PSBMPC(seed=seed)
            self._psbmpc_ud = float(self.v)
            self._psbmpc_start = (float(self.x), float(self.y))
        obstacles = [
            (o.x, o.y, o.v, o.theta, o.mmsi) for o in sea_objects if o is not self
        ]
        theta_c, p_factor, _chi = self._psbmpc.decide(
            (self.x, self.y, self.v, self.theta),
            obstacles,
            self._psbmpc_start,
            (self.goal_x, self.goal_y),
            self._psbmpc_ud,
            dt,
        )

        err = math.remainder(theta_c - float(self.theta), 2.0 * math.pi)
        yaw = min(max(0.2 / dt * err, -np.pi / 4), np.pi / 4)
        v_cmd = max(self._psbmpc_ud * p_factor, SPEED_MIN_MPS)
        acc = min(
            max((v_cmd - self.v) / dt, SURGE_DECEL_MAX_MPS2), SURGE_ACCEL_MAX_MPS2
        )
        self.update(np.array([[acc], [yaw]]), dt)
        return [
            self.mmsi,
            self.name,
            self.get_state_vector(),
            self.collision_risk,
            self.collision_history,
        ]

    def move_velocity_obstacle(
        self,
        record_data,
        sea_objects,
        mmsi_list,
        rules,
        table,
        ax,
        eps,
        s,
        k,
        dt,
        verbose=False,
    ):
        """Move one step with velocity obstacles.

        Args:
            record_data (bool): Whether to store the trajectory
            sea_objects (list): All sea objects
            mmsi_list (list): Identifiers of all sea objects
            rules (list): Active rules table
            table (matplotlib.table.Table): Rules table display, or None
            ax (matplotlib.axes.Axes): Axes to draw on, or None
            eps (float): Safety margin added to the radius, m
            s (float): Half size of the map, m
            k (int): Current step index
            dt (float): Time step, s
            verbose (bool): Whether to print debug output

        Returns:
            list: Identifier, name, state vector, collision risk and history
        """

        arrival_radius = max(self.v * dt, 150.0)
        dist_to_goal = sqrt((self.goal_x - self.x) ** 2 + (self.goal_y - self.y) ** 2)
        if dist_to_goal < arrival_radius or self.final:
            self.v = 0.0
            self.update(array([[0], [0]]), dt)
            self.final = 1
            return [
                self.mmsi,
                self.name,
                self.get_state_vector(),
                self.collision_risk,
                self.collision_history,
            ]

        d_safe = self.D_SAFE_MULT * self.r
        threats = []
        for other in sea_objects:
            if other is self:
                continue
            situation = self.COLREGs_situation(other)
            self.list_colregs.append([self.mmsi, other.mmsi, situation])
            if situation in self.GIVE_WAY_SITUATIONS or situation == RELAXED_SITUATION:
                role = "give_way"
            elif situation in self.STAND_ON_SITUATIONS:
                role = "stand_on"
            else:
                role = "safe"
            threats.append(
                VOThreat(
                    x=other.x,
                    y=other.y,
                    vx=other.v * np.cos(other.theta),
                    vy=other.v * np.sin(other.theta),
                    r_combined=d_safe + other.r,
                    role=role,
                )
            )

        heading = select_heading(
            self.x,
            self.y,
            self.v,
            self.theta,
            self.goal_x,
            self.goal_y,
            threats,
            rule_mode=self.rule_mode,
        )
        leg = max(2000.0, d_safe * 2.0)
        target = [self.x + leg * np.cos(heading), self.y + leg * np.sin(heading)]
        self._steer_to_point(target, dt, verbose=verbose)

        return [
            self.mmsi,
            self.name,
            self.get_state_vector(),
            self.collision_risk,
            self.collision_history,
        ]

    def move_letter(
        self,
        record_data,
        sea_objects,
        mmsi_list,
        rules,
        table,
        ax,
        eps,
        s,
        k,
        dt,
        verbose=False,
    ):
        """Move one step with the rule by the letter method.

        Args:
            record_data (bool): Whether to store the trajectory
            sea_objects (list): All sea objects
            mmsi_list (list): Identifiers of all sea objects
            rules (list): Active rules table
            table (matplotlib.table.Table): Rules table display, or None
            ax (matplotlib.axes.Axes): Axes to draw on, or None
            eps (float): Safety margin added to the radius, m
            s (float): Half size of the map, m
            k (int): Current step index
            dt (float): Time step, s
            verbose (bool): Whether to print debug output

        Returns:
            list: Identifier, name, state vector, collision risk and history
        """
        _rec = [
            self.mmsi,
            self.name,
            self.get_state_vector(),
            self.collision_risk,
            self.collision_history,
        ]
        arrival_radius = max(self.v * dt, 150.0)
        dist_to_goal = sqrt((self.goal_x - self.x) ** 2 + (self.goal_y - self.y) ** 2)
        if dist_to_goal < arrival_radius or self.final:
            self.v = 0.0
            self.update(array([[0], [0]]), dt)
            self.final = 1
            return _rec

        committed = (
            bool(getattr(self, "avoidance_side", 0))
            and self.avoidance_ref_heading is not None
        )
        target = None
        if self.letter_commit_mmsi is not None:
            target = next(
                (
                    o
                    for o in sea_objects
                    if o.mmsi == self.letter_commit_mmsi and o is not self
                ),
                None,
            )
        if target is None and self.letter_target_mmsi is not None:
            target = next(
                (
                    o
                    for o in sea_objects
                    if o.mmsi == self.letter_target_mmsi and o is not self
                ),
                None,
            )
        if target is None and self.letter_target_mmsi is None:
            pos_os = np.array([self.x, self.y])
            vel_os = np.array(
                [self.v * np.cos(self.theta), self.v * np.sin(self.theta)]
            )
            best = None
            for other in sea_objects:
                if other is self:
                    continue
                situation = self._letter_situation(other)
                if situation == "Safe":
                    continue
                pos_ts = np.array([other.x, other.y])
                vel_ts = np.array(
                    [other.v * np.cos(other.theta), other.v * np.sin(other.theta)]
                )
                _, tcpa = self._cpa(pos_os, vel_os, pos_ts, vel_ts)
                if not np.isfinite(tcpa):
                    continue
                key = (self._action_priority(situation), tcpa)
                if best is None or key < best[0]:
                    best = (key, other, situation)
            if best is not None:
                target = best[1]

        if target is None:
            if committed:
                self._release_letter_commit()
            blocker = next(
                (
                    o
                    for o in sea_objects
                    if o is not self and not self._resume_would_clear(o)
                ),
                None,
            )
            if blocker is not None:
                self._steer_to_point(self._letter_hold_point(), dt)
            else:
                self._steer_to_point([self.goal_x, self.goal_y], dt)
            return _rec

        situation = self._letter_situation(target)
        for o in sea_objects:
            if o is not self:
                self.list_colregs.append([self.mmsi, o.mmsi, self._letter_situation(o)])

        give_way_engaged = committed or (situation in self.GIVE_WAY_SITUATIONS)
        if give_way_engaged:
            if not committed:
                self.avoidance_entry_heading = self.theta
                self._commit_avoidance_side(target, situation)
                self.letter_target_mmsi = target.mmsi
            if situation in self.GIVE_WAY_SITUATIONS:
                gh = self._letter_giveway_heading(target)
                leg = max(2000.0, self.D_SAFE_MULT * self.r * 2.0)
                self._steer_to_point(
                    [self.x + leg * np.cos(gh), self.y + leg * np.sin(gh)], dt
                )
            else:
                pos_os = np.array([self.x, self.y])
                vel_os = np.array(
                    [self.v * np.cos(self.theta), self.v * np.sin(self.theta)]
                )
                pos_ts = np.array([target.x, target.y])
                vel_ts = np.array(
                    [target.v * np.cos(target.theta), target.v * np.sin(target.theta)]
                )
                _, tcpa = self._cpa(pos_os, vel_os, pos_ts, vel_ts)
                if tcpa <= 0.0 and self._resume_would_clear(target):
                    self._release_letter_commit()
                    self._steer_to_point([self.goal_x, self.goal_y], dt)
                else:
                    self._steer_to_point(self._letter_hold_point(), dt)
        else:
            self._steer_to_point(self._letter_hold_point(), dt)
        return _rec

    def _resume_would_clear(self, target):
        """Check that heading back to the goal passes the target safely.

        Args:
            target (SeaObject): Target vessel

        Returns:
            bool: True if the DCPA stays safe
        """
        gh = np.arctan2(self.goal_y - self.y, self.goal_x - self.x)
        v_os = self.v * np.array([np.cos(gh), np.sin(gh)])
        v_ts = np.array(
            [target.v * np.cos(target.theta), target.v * np.sin(target.theta)]
        )
        r0 = np.array([self.x - target.x, self.y - target.y])
        v = v_os - v_ts
        vv = float(v @ v)
        D_safe = self.D_SAFE_MULT * self.r
        if vv < 1e-9:
            return float(np.hypot(*r0)) >= D_safe
        tcpa = -float(r0 @ v) / vv
        if tcpa <= 0.0:
            return True
        dcpa = float(np.hypot(*(r0 + v * tcpa)))
        return dcpa >= D_safe

    def _letter_hold_point(self):
        """Put a point far ahead on the current heading.

        Returns:
            list: Point (x, y), m
        """
        leg = max(2000.0, self.D_SAFE_MULT * self.r * 2.0)
        return [self.x + leg * np.cos(self.theta), self.y + leg * np.sin(self.theta)]

    def _release_letter_commit(self):
        """Forget the side and target of the current manoeuvre."""
        self.avoidance_side = 0
        self.avoidance_ref_heading = None
        self.avoidance_ref_pos = None
        self.avoidance_entry_heading = None
        self.letter_target_mmsi = None
        self.letter_role = None
        self._letter_ot_roles.clear()

    def _letter_situation(self, target):
        """Get the situation, keeping overtaking roles once set.

        Args:
            target (SeaObject): Target vessel

        Returns:
            str: COLREGs situation
        """
        raw = self.COLREGs_situation(target)
        if not self.letter_rule13d:
            return raw
        m = target.mmsi
        if m in self._letter_ot_roles:
            return self._letter_ot_roles[m] if raw != "Safe" else "Safe"
        if raw == "Overtaking_give_way" and self._abaft_beam(target, self):
            self._letter_ot_roles[m] = raw
        elif raw == "Being_overtaken_stand_on" and self._abaft_beam(self, target):
            self._letter_ot_roles[m] = raw
        return raw

    @staticmethod
    def _abaft_beam(front, back):
        """Check that one vessel is more than 22.5 deg abaft the beam of another.

        Args:
            front (SeaObject): Vessel in front
            back (SeaObject): Vessel behind

        Returns:
            bool: True if abaft the beam
        """
        rel = np.array([back.x - front.x, back.y - front.y])
        bearing = np.arctan2(rel[1], rel[0]) - front.theta
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
        return abs(bearing) > np.radians(112.5)

    def _letter_giveway_heading(self, target):
        """Find the smallest turn that gives a safe pass, astern when possible.

        Args:
            target (SeaObject): Target vessel

        Returns:
            float: Heading to steer, rad
        """
        h0 = (
            self.avoidance_ref_heading
            if self.avoidance_ref_heading is not None
            else self.theta
        )
        side = self.avoidance_side if self.avoidance_side else +1
        D_safe = self.D_SAFE_MULT * self.r
        D_action = self.giveway_action_dist_m
        min_deg = self.giveway_min_apparent_deg
        r0 = np.array([self.x - target.x, self.y - target.y])
        v_ts = np.array(
            [target.v * np.cos(target.theta), target.v * np.sin(target.theta)]
        )
        ts_fwd = np.array([np.cos(target.theta), np.sin(target.theta)])

        def _dcpa_astern(th):
            """Compute the DCPA for a heading and check the pass is astern.

            Args:
                th (float): Heading to test, rad

            Returns:
                tuple: DCPA (m) and True if passing astern
            """
            v_os = self.v * np.array([np.cos(th), np.sin(th)])
            v = v_os - v_ts
            vv = float(v @ v)
            if vv < 1e-9:
                return float(np.hypot(*r0)), True
            tcpa = -float(r0 @ v) / vv
            dcpa = (
                float(np.hypot(*r0))
                if tcpa <= 0.0
                else float(np.hypot(*(r0 + v * tcpa)))
            )
            os_cpa = np.array([self.x, self.y]) + v_os * max(tcpa, 0.0)
            ts_cpa = np.array([target.x, target.y]) + v_ts * max(tcpa, 0.0)
            return dcpa, float((os_cpa - ts_cpa) @ ts_fwd) <= 0.0

        dcpa0, astern0 = _dcpa_astern(h0)
        if dcpa0 >= D_action and astern0:
            return h0
        best_clear = None
        for deg in range(int(min_deg), 181, 2):
            th = h0 - side * np.radians(deg)
            dcpa, astern = _dcpa_astern(th)
            if dcpa >= D_safe:
                if best_clear is None:
                    best_clear = th
                if astern or deg >= 90:
                    return th
        return best_clear if best_clear is not None else h0 - side * (np.pi / 2.0)

    def _get_worst_cpa(self, sea_objects, situations=None):
        """Find the target with the most urgent close pass.

        Args:
            sea_objects (list): All sea objects
            situations (dict): Known situations, or None

        Returns:
            tuple: TCPA, DCPA, CPA point and target, or None
        """
        D_SAFE = self.D_SAFE_MULT * self.r
        worst = None

        xo, yo = float(self.x), float(self.y)
        vxo, vyo = _vel(self)
        for other in sea_objects:
            if other is self:
                continue

            vxt, vyt = _vel(other)
            dcpa, tcpa = _cpa_xy(
                xo, yo, vxo, vyo, float(other.x), float(other.y), vxt, vyt
            )

            if tcpa <= 0 or dcpa > D_SAFE * THREAT_FILTER_MULT:
                continue

            cpa_coords = (
                self.x + self.v * math.cos(self.theta) * tcpa,
                self.y + self.v * math.sin(self.theta) * tcpa,
            )

            situation = (
                situations[id(other)]
                if situations is not None and id(other) in situations
                else self.COLREGs_situation(other)
            )
            priority = self._action_priority(situation)

            threatening = dcpa < D_SAFE
            key = (not threatening, priority, tcpa)
            if worst is None or key < worst[0]:
                worst = (key, tcpa, dcpa, cpa_coords, other)

        if worst is None:
            return None
        return (worst[1], worst[2], worst[3], worst[4])

    def _action_priority(self, situation):
        """Rank a situation: give way first, then stand on.

        Args:
            situation (str): COLREGs situation

        Returns:
            int: 0 give way, 1 stand on, 2 other
        """
        if situation in self.GIVE_WAY_SITUATIONS:
            return 0
        if situation in self.STAND_ON_SITUATIONS:
            return 1
        return 2

    def _record_obligations(self, sea_objects):
        """Store the COLREGs duty with each target.

        Args:
            sea_objects (list): All sea objects
        """
        obligations = []
        pos_os = np.array([self.x, self.y])
        vel_os = np.array([self.v * np.cos(self.theta), self.v * np.sin(self.theta)])
        for other in sea_objects:
            if other is self:
                continue
            situation = self.COLREGs_situation(other)
            if situation == "Safe":
                continue
            pos_ts = np.array([other.x, other.y])
            vel_ts = np.array(
                [other.v * np.cos(other.theta), other.v * np.sin(other.theta)]
            )
            dcpa, tcpa = self._cpa(pos_os, vel_os, pos_ts, vel_ts)
            priority = self._action_priority(situation)
            role = (
                "give_way"
                if priority == 0
                else "stand_on" if priority == 1 else "other"
            )
            obligations.append(
                {
                    "mmsi": other.mmsi,
                    "situation": situation,
                    "role": role,
                    "dcpa": float(dcpa),
                    "tcpa": float(tcpa),
                    "_priority": priority,
                }
            )
        self._obligations = obligations
        if obligations:
            primary = min(obligations, key=lambda o: (o["_priority"], o["tcpa"]))
            self._primary_threat = primary["mmsi"]
        else:
            self._primary_threat = None

    def _goal_directed_cpa_threat(self, sea_objects, verbose=False) -> bool:
        """Check whether heading to the goal leads to an unsafe pass.

        Args:
            sea_objects (list): All sea objects
            verbose (bool): Whether to print debug output

        Returns:
            bool: True if there is a threat
        """
        D_SAFE = self.D_SAFE_MULT * self.r
        dist_to_goal = np.hypot(self.goal_x - self.x, self.goal_y - self.y)
        if verbose:
            print("D_SAFE =", D_SAFE)
            print("dist_to_goal =", dist_to_goal)
        if dist_to_goal < 100:
            if verbose:
                print("Close to goal, ignoring CPA threat check")
            return False

        goal_vx = self.v * (self.goal_x - self.x) / dist_to_goal
        goal_vy = self.v * (self.goal_y - self.y) / dist_to_goal
        if verbose:
            goal_vel = np.array([goal_vx, goal_vy])
            pos_os = np.array([self.x, self.y])
            print("Checking CPA threat when heading straight to goal:")
            print("Current position:", (self.x, self.y))
            print("Goal position:", (self.goal_x, self.goal_y))
            print("Goal velocity vector:", goal_vel)
        for other in sea_objects:
            if other is self:
                if verbose:
                    print(f"Skipping object {other.name} (final={other.final})")
                continue
            vxt, vyt = _vel(other)
            dcpa, tcpa = _cpa_xy(
                float(self.x),
                float(self.y),
                float(goal_vx),
                float(goal_vy),
                float(other.x),
                float(other.y),
                vxt,
                vyt,
            )
            if verbose:
                pos_ts = np.array([other.x, other.y])
                vel_ts = np.array([vxt, vyt])
                print(
                    f"Object {other.name}: pos={pos_ts}, vel={vel_ts}, tcpa={tcpa:.1f}, dcpa={dcpa:.1f}"
                )
            D_SAFE = self.D_SAFE_MULT * self.r
            if tcpa <= 0 or dcpa >= D_SAFE:
                continue

            if tcpa > _duty.HORIZON_S and not (
                tcpa < self.T_ALERT and self._gives_way_to(other)
            ):
                continue
            if verbose:
                print(
                    f"  -> Goal-directed CPA threat detected with {other.name} (DCPA={dcpa:.1f}, TCPA={tcpa:.1f})"
                )
            return True
        if verbose:
            print("No goal-directed CPA threats detected")
        return False

    def _commit_avoidance_side(self, threat_obj, situation):
        """Choose and keep the side of the avoidance manoeuvre.

        Args:
            threat_obj (SeaObject): Threat vessel
            situation (str): COLREGs situation
        """
        self.avoidance_ref_heading = self.theta
        self.avoidance_ref_pos = (self.x, self.y)
        self.extremis_side_hold = situation in self.STAND_ON_SITUATIONS
        if situation == RELAXED_SITUATION:
            self.avoidance_side = 0
            return
        if situation in ("Head_on", "Crossing_give_way"):
            self.avoidance_side = +1
            return
        if situation == "Overtaking_give_way":
            if self._starboard_duty_live(threat_obj):
                self.avoidance_side = +1
                return
            port_x = -np.sin(self.theta)
            port_y = np.cos(self.theta)
            rel_x = threat_obj.x - self.x
            rel_y = threat_obj.y - self.y
            self.avoidance_side = +1 if (rel_x * port_x + rel_y * port_y) > 0 else -1
            return
        self.avoidance_side = +1

    def _gives_way_to(self, other):
        """Check whether this vessel must give way to another.

        Args:
            other (SeaObject): Other vessel

        Returns:
            bool: True if it gives way
        """
        situation = self.COLREGs_situation(other)
        return situation in self.GIVE_WAY_SITUATIONS or situation == RELAXED_SITUATION

    def _starboard_duty_live(self, besides):
        """Check whether another vessel still needs a starboard turn.

        Args:
            besides (SeaObject): Vessel to ignore

        Returns:
            bool: True if a starboard duty is still active
        """
        xo, yo = float(self.x), float(self.y)
        vxo, vyo = _vel(self)
        for other in self.sea_objects or []:
            if other is self or other is besides or other.final:
                continue
            if self.locked_situations.get(other.mmsi) not in (
                "Head_on",
                "Crossing_give_way",
            ):
                continue
            _dcpa, tcpa = _cpa_xy(
                xo, yo, vxo, vyo, float(other.x), float(other.y), *_vel(other)
            )
            if 0.0 < tcpa < self.T_ALERT:
                return True
        return False

    def _in_extremis(self, threat, tcpa, dcpa):
        """Check whether the threat is so close that the stand on vessel must act.

        Args:
            threat (SeaObject): Threat vessel
            tcpa (float): Time to closest approach, s
            dcpa (float): Distance at closest approach, m

        Returns:
            bool: True if in extremis
        """
        d_safe = self.D_SAFE_MULT * self.r
        extremis_dcpa = max(self.r * EXTREMIS_DCPA_HULLS, d_safe * EXTREMIS_DCPA_DSAFE)
        dist = np.hypot(self.x - threat.x, self.y - threat.y)
        return 0.0 < tcpa and dcpa < extremis_dcpa and dist <= EXTREMIS_RANGE_M

    def _meets_entry_test(self, threat, tcpa, dcpa, situation):
        """Check whether a threat calls for an avoidance manoeuvre.

        Args:
            threat (SeaObject): Threat vessel
            tcpa (float): Time to closest approach, s
            dcpa (float): Distance at closest approach, m
            situation (str): COLREGs situation

        Returns:
            bool: True if avoidance must start
        """
        if not (0.0 < tcpa < self.T_ALERT and dcpa < self.D_SAFE_MULT * self.r):
            return False
        if situation in self.GIVE_WAY_SITUATIONS or situation == RELAXED_SITUATION:
            return True
        if not (self.emergency_avoidance and situation in self.STAND_ON_SITUATIONS):
            return False
        return self._in_extremis(threat, tcpa, dcpa)

    def _retarget_avoidance(self, cpa_data, situations, verbose=False):
        """Switch the avoidance to a more urgent threat if needed.

        Args:
            cpa_data (tuple): TCPA, DCPA, CPA point and threat
            situations (dict): Known situations
            verbose (bool): Whether to print debug output
        """
        tcpa, dcpa, _coords, threat = cpa_data
        situation = situations.get(id(threat)) or self.COLREGs_situation(threat)
        if not self._meets_entry_test(threat, tcpa, dcpa, situation):
            return

        held = self.threat_ts
        if held is not None and not held.final:
            h_dcpa, h_tcpa = _cpa_xy(
                float(self.x),
                float(self.y),
                *_vel(self),
                float(held.x),
                float(held.y),
                *_vel(held),
            )
            h_sit = situations.get(id(held)) or self.COLREGs_situation(held)
            if self._meets_entry_test(held, h_tcpa, h_dcpa, h_sit):
                return

        self._enter_avoidance(threat, tcpa, situation, verbose)

    def _enter_avoidance(self, threat, tcpa, situation, verbose=False):
        """Start an avoidance manoeuvre against a threat.

        Args:
            threat (SeaObject): Threat vessel
            tcpa (float): Time to closest approach, s
            situation (str): COLREGs situation
            verbose (bool): Whether to print debug output
        """
        give_way = (
            situation in self.GIVE_WAY_SITUATIONS or situation == RELAXED_SITUATION
        )
        self.nav_phase = "AVOIDANCE"
        self.threat_ts = threat
        self.frozen_role = situation if give_way else None
        self.avoidance_entry_heading = self.theta
        self._commit_avoidance_side(threat, situation)
        self._reuse_released_reference(threat)
        self.resume_waypoint = self._compute_resume_waypoint(
            threat, tcpa, situation, verbose=verbose
        )
        self.avoidance_replan_counter = 0
        self._resolve_counter = 0
        self.waypoints = []

    def _reuse_released_reference(self, threat):
        """Reuse the last reference heading if the same threat comes back soon.

        Args:
            threat (SeaObject): Threat vessel
        """
        if self._released is None or self._released_age_s > REENTRY_WINDOW_S:
            return
        released, h_entry, h_ref, ref_pos = self._released
        if released is not threat or h_ref is None:
            return
        self.avoidance_entry_heading = h_entry
        self.avoidance_ref_heading = h_ref
        self.avoidance_ref_pos = ref_pos

    def _age_release(self, dt):
        """Count the time since the last avoidance ended.

        Args:
            dt (float): Time step, s
        """
        self._released_age_s += dt

    def _clear_avoidance(self):
        """Reset every avoidance setting."""
        self.threat_ts = None
        self.resume_waypoint = None
        self.avoidance_side = 0
        self.avoidance_ref_heading = None
        self.avoidance_ref_pos = None
        self.frozen_role = None
        self.avoidance_entry_heading = None
        self.extremis_side_hold = False
        self._resolve_counter = 0
        self.avoidance_replan_counter = 0

    def _release_avoidance(self):
        """End the avoidance and add a waypoint to rejoin the route."""
        h0 = (
            self.avoidance_entry_heading
            if self.avoidance_entry_heading is not None
            else self.theta
        )
        side = self.avoidance_side if self.avoidance_side else +1
        leg = RELEASE_LEG_M
        perp_x, perp_y = np.sin(h0) * side, -np.cos(h0) * side
        resume_pt = [
            self.x + leg * np.cos(h0) + perp_x * (self.r * 4.0),
            self.y + leg * np.sin(h0) + perp_y * (self.r * 4.0),
        ]
        self.nav_phase = "TRANSIT"
        self.waypoints = [resume_pt, [self.goal_x, self.goal_y]]
        self._released = (
            self.threat_ts,
            self.avoidance_entry_heading,
            self.avoidance_ref_heading,
            self.avoidance_ref_pos,
        )
        self._released_age_s = 0.0
        self._clear_avoidance()

    def _compliant_reference(self):
        """Get the line that the path must not cross during avoidance.

        Returns:
            tuple: Reference x, y, heading and side, or None
        """
        if not getattr(self, "avoidance_side", 0) or self.avoidance_ref_heading is None:
            return None
        rx, ry = (
            self.avoidance_ref_pos
            if self.avoidance_ref_pos is not None
            else (self.x, self.y)
        )
        return (rx, ry, float(self.avoidance_ref_heading), int(self.avoidance_side))

    def _compute_resume_waypoint(self, threat_obj, tcpa, situation=None, verbose=False):
        """Compute where to rejoin the route after the threat has passed.

        Args:
            threat_obj (SeaObject): Threat vessel
            tcpa (float): Time to closest approach, s
            situation (str): COLREGs situation, or None
            verbose (bool): Whether to print debug output

        Returns:
            list: Waypoint (x, y), m
        """
        if situation == "Overtaking_give_way":
            ts_cpa_x = threat_obj.x + threat_obj.v * np.cos(threat_obj.theta) * tcpa
            ts_cpa_y = threat_obj.y + threat_obj.v * np.sin(threat_obj.theta) * tcpa
            dist = max(np.hypot(self.x - ts_cpa_x, self.y - ts_cpa_y), 1e-4)

            fwd_x = np.cos(threat_obj.theta)
            fwd_y = np.sin(threat_obj.theta)

            cross_product = fwd_x * (self.y - ts_cpa_y) - fwd_y * (self.x - ts_cpa_x)
            if cross_product >= 0:
                perp_x, perp_y = -fwd_y, fwd_x
            else:
                perp_x, perp_y = fwd_y, -fwd_x

            projection_distance = (
                dist * OVERTAKE_RESUME_RANGE_FRAC + dist * OVERTAKE_RESUME_LEAD_FRAC
            ) * OVERTAKE_RESUME_STRETCH
            offset_dist = max(
                dist * OVERTAKE_RESUME_RANGE_FRAC, OVERTAKE_RESUME_OFFSET_MIN_M
            )

            rx = ts_cpa_x + fwd_x * projection_distance + perp_x * offset_dist
            ry = ts_cpa_y + fwd_y * projection_distance + perp_y * offset_dist

            rx = float(np.clip(rx, -self.map_size, self.map_size))
            ry = float(np.clip(ry, -self.map_size, self.map_size))
            return [rx, ry]

        ts_heading = threat_obj.theta
        fwd_x = np.cos(ts_heading)
        fwd_y = np.sin(ts_heading)

        min_clearance = (threat_obj.v * RESUME_LEAD_S) + (
            max(self.r, getattr(threat_obj, "r", R_VESSEL_M)) * RESUME_HULLS
        )
        cross_s = min_clearance

        cross_x = threat_obj.x + fwd_x * cross_s
        cross_y = threat_obj.y + fwd_y * cross_s

        if verbose:
            print(f"Goal projection bounded to safe minimum: s={cross_s:.1f}m")

        rel_x = self.x - threat_obj.x
        rel_y = self.y - threat_obj.y
        cross_product = fwd_x * rel_y - fwd_y * rel_x

        inter_vessel_dist = np.hypot(rel_x, rel_y)
        offset_dist = max(inter_vessel_dist * RESUME_OFFSET_FRAC, RESUME_OFFSET_MIN_M)

        ref = self._compliant_reference()
        if ref is not None and situation in ("Head_on", "Crossing_give_way"):
            _, _, h0, side = ref
            perp_x = np.sin(h0) * side
            perp_y = -np.cos(h0) * side
        elif situation == "Head_on":
            os_fwd_x = np.cos(self.theta)
            os_fwd_y = np.sin(self.theta)
            perp_x = os_fwd_y
            perp_y = -os_fwd_x
        else:
            if cross_product >= 0:
                perp_x = fwd_y
                perp_y = -fwd_x
            else:
                perp_x = -fwd_y
                perp_y = fwd_x

        rx = cross_x + perp_x * offset_dist
        ry = cross_y + perp_y * offset_dist

        if ref is not None and situation == "Crossing_give_way":
            _, _, h0, side = ref
            fx, fy = np.cos(h0), np.sin(h0)
            if (rx - self.x) * fx + (ry - self.y) * fy <= 0.0:
                col_x = self.x + fx * self.v * tcpa
                col_y = self.y + fy * self.v * tcpa
                perp_x = np.sin(h0) * side
                perp_y = -np.cos(h0) * side
                rx = col_x + perp_x * offset_dist
                ry = col_y + perp_y * offset_dist

        rx = float(np.clip(rx, -self.map_size, self.map_size))
        ry = float(np.clip(ry, -self.map_size, self.map_size))

        return [rx, ry]

    _duty_buf = None

    def _duty_buffers(self, n):
        """Get reusable arrays for the duty kernels.

        Args:
            n (int): Number of other vessels

        Returns:
            tuple: Own state, target rows and role codes
        """
        buf = self._duty_buf
        if buf is None or buf[1].shape[0] != n:
            buf = (np.zeros(5), np.zeros((n, 5)), np.empty(n, np.int64))
            self._duty_buf = buf
        return buf

    def _memory_codes(self, others):
        """Encode the overtaking memory of each vessel for numba.

        Args:
            others (list): Other vessels

        Returns:
            ndarray: Memory code per vessel
        """
        mem = self._mem_buf
        if mem is None or mem.shape[0] != len(others):
            mem = np.zeros(len(others), np.int64)
            self._mem_buf = mem
        for i, o in enumerate(others):
            mem[i] = _MEMORY_CODES.get(self.overtaking.get(o.mmsi), 0)
        return mem

    def update_overtaking(self, sea_objects=None):
        """Update the overtaking memory with each vessel.

        Args:
            sea_objects (list): All sea objects, or None
        """
        if not self._rulebook_latch:
            return
        if self.sea_objects is None:
            self.sea_objects = sea_objects
        others = [o for o in (self.sea_objects or []) if o is not self]
        geo = self._live_duty_roles(memory=False)
        kept = {}
        for o, r in zip(others, geo):
            m = self.overtaking.get(o.mmsi)
            if m in _duty.OVERTAKE_ROLES:
                over, under = (self, o) if m == "OGW" else (o, self)
                if not _duty.finally_past_and_clear(over, under):
                    kept[o.mmsi] = m
            elif r in _duty.OVERTAKE_ROLES:
                kept[o.mmsi] = r
        self.overtaking = kept

    def _live_duty_roles(self, memory=True):
        """Compute the current COLREGs role with each vessel.

        Args:
            memory (bool): Whether to use the overtaking memory

        Returns:
            list: Role per vessel
        """
        others = [o for o in (self.sea_objects or []) if o is not self]
        if not others:
            return []
        try:
            if _monitor_nb is not None:
                own, rows, codes = self._duty_buffers(len(others))
                own[0], own[1], own[2], own[3] = self.x, self.y, self.v, self.theta
                for i, o in enumerate(others):
                    rows[i, 0], rows[i, 1] = o.x, o.y
                    rows[i, 2], rows[i, 3] = o.v, o.theta
                if memory and self.overtaking:
                    _monitor_nb.roles_of_mem(
                        own,
                        rows,
                        True,
                        float(PORT_VETO_GATE_M),
                        self._memory_codes(others),
                        codes,
                    )
                else:
                    _monitor_nb.roles_of(
                        own, rows, True, float(PORT_VETO_GATE_M), codes
                    )
                return [_monitor_nb.ROLE_NAMES[c] for c in codes]
            os_v = _DutyVessel.from_sea_object(self, r_min=R_MIN_M)
            return [
                _duty.role(
                    os_v,
                    _DutyVessel.from_sea_object(o, r_min=R_MIN_M),
                    gate=PORT_VETO_GATE_M,
                    memory=self.overtaking.get(o.mmsi) if memory else None,
                )
                for o in others
            ]
        except Exception:
            return []

    _steer_out = None
    _mem_buf = None

    def _steer_fast(self, target, dt):
        """Steer to a point with the numba kernel.

        Args:
            target (tuple): Point (x, y), m
            dt (float): Time step, s

        Returns:
            bool: True if the kernel was used
        """
        others = (
            [o for o in (self.sea_objects or []) if o is not self]
            if self.colregs_port_veto
            else []
        )
        n = len(others)
        own, rows, _codes = self._duty_buffers(n)
        own[0], own[1], own[2], own[3] = self.x, self.y, self.v, self.theta
        for i, o in enumerate(others):
            rows[i, 0], rows[i, 1] = o.x, o.y
            rows[i, 2], rows[i, 3] = o.v, o.theta

        out = self._steer_out
        if out is None:
            out = np.empty(4)
            self._steer_out = out
        cap = self.max_rot_rad_s
        try:
            flags = _steer_nb.command(
                own,
                rows,
                n > 0,
                float(PORT_VETO_GATE_M),
                float(target[0]),
                float(target[1]),
                float(dt),
                cap is not None,
                0.0 if cap is None else float(cap),
                out,
                self._memory_codes(others),
            )
        except Exception:
            return False

        if flags & _steer_nb.F_PORT_VETO:
            self.port_veto_steps += 1
        if flags & _steer_nb.F_GIVEWAY_FLOOR:
            self.giveway_floor_steps += 1
        if flags & _steer_nb.F_STANDON_SNAP:
            self.standon_snap_steps += 1

        self.x, self.y, self.v, self.theta = out[0], out[1], out[2], out[3]
        self.update_history(self.history_lenght)
        self.trajectory[0].append(self.x)
        self.trajectory[1].append(self.y)
        return True

    def _steer_to_point(self, target, dt, verbose=False):
        """Steer to a point with the COLREGs veto and move one step.

        Args:
            target (tuple): Point (x, y), m
            dt (float): Time step, s
            verbose (bool): Whether to print debug output
        """
        if (
            _steer_nb is not None
            and not verbose
            and self._speed_gov is None
            and self.safety_filter is _cbf.SafetyFilter.NONE
            and not (self.colregs_clamp and self._clamp_context)
            and self._steer_fast(target, dt)
        ):
            return

        thetabar = math.atan2(float(target[1] - self.y), float(target[0] - self.x))
        d_th = thetabar - float(self.theta)
        error = math.atan2(math.sin(d_th), math.cos(d_th))
        roles = self._live_duty_roles() if self.colregs_port_veto else []

        if error > 0.0 and any(r in NO_PORT_ROLES for r in roles):
            error = 0.0
            self.port_veto_steps += 1

        u2 = min(max(0.2 / dt * error, -np.pi / 4), np.pi / 4)

        give_way = [r for r in roles if r in _duty.GIVE_WAY]
        if give_way:
            magnitude = _duty.DELTA_MIN_RAD / _duty.HORIZON_S
            port_allowed = all(r == "OGW" for r in give_way) and u2 > 0.0
            floor = magnitude if port_allowed else -magnitude
            if (port_allowed and u2 < floor) or (not port_allowed and u2 > floor):
                u2 = floor
                self.giveway_floor_steps += 1

        elif any(r in _duty.STAND_ON for r in roles):
            want = -u2 * _duty.HORIZON_S
            if _duty.HOLD_BAND_RAD < want < _duty.DELTA_MIN_RAD:
                midpoint = 0.5 * (_duty.HOLD_BAND_RAD + _duty.DELTA_MIN_RAD)
                edge = _duty.DELTA_MIN_RAD if want > midpoint else _duty.HOLD_BAND_RAD
                u2 = -edge / _duty.HORIZON_S
                self.standon_snap_steps += 1

        if self._speed_gov is not None:
            speed_targets = self._collect_speed_targets()
            active_role = self._resolve_active_role(speed_targets)
            v_cmd = self._speed_gov.decide_speed(
                v=self.v,
                theta=self.theta,
                role=active_role,
                targets=speed_targets,
                j_cascade=self.j_cascade,
                dt=dt,
            )
            u0 = self._speed_gov.accel_command(self.v, v_cmd)
        else:
            u0 = 0.0

        self.update(np.array([[u0], [u2]]), dt)

        if self._speed_gov is not None:
            self.v = self._speed_gov.envelope_clip(self.v)

        if verbose:
            print(
                f"Steering to point {target}: thetabar={np.degrees(thetabar):.1f}°, error={np.degrees(error):.1f}°, u2={np.degrees(u2):.1f}°/s"
            )

    def _collect_speed_targets(self):
        """Build the target list for the speed governor.

        Returns:
            list: Target information
        """
        targets = []
        pos_os = np.array([self.x, self.y])
        vel_os = np.array([self.v * np.cos(self.theta), self.v * np.sin(self.theta)])
        for other in self.sea_objects:
            if other is self:
                continue
            pos_ts = np.array([other.x, other.y])
            vel_ts = np.array(
                [other.v * np.cos(other.theta), other.v * np.sin(other.theta)]
            )
            dcpa, tcpa = self._cpa(pos_os, vel_os, pos_ts, vel_ts)
            dx, dy = other.x - self.x, other.y - self.y
            range_m = float(np.hypot(dx, dy))
            if range_m > 0:
                rel_vel = vel_ts - vel_os
                range_rate = float((dx * rel_vel[0] + dy * rel_vel[1]) / range_m)
            else:
                range_rate = 0.0
            targets.append(
                TargetInfo(dcpa=dcpa, tcpa=tcpa, range_m=range_m, range_rate=range_rate)
            )
        return targets

    def _resolve_active_role(self, speed_targets):
        """Find the COLREGs role with the most risky vessel.

        Args:
            speed_targets (list): Target information

        Returns:
            str: COLREGs situation, or None
        """
        if self.frozen_role is not None:
            return self.frozen_role
        if not self.sea_objects:
            return None
        worst_other = None
        worst_cri = -1.0

        pos_os = np.array([self.x, self.y])
        vel_os = np.array([self.v * np.cos(self.theta), self.v * np.sin(self.theta)])
        for other in self.sea_objects:
            if other is self:
                continue
            pos_ts = np.array([other.x, other.y])
            vel_ts = np.array(
                [other.v * np.cos(other.theta), other.v * np.sin(other.theta)]
            )
            dcpa, tcpa = self._cpa(pos_os, vel_os, pos_ts, vel_ts)
            if tcpa <= 0 or tcpa > T_HORIZON_S:
                continue
            f_d = max(0.0, min(1.0, 1.0 - dcpa / ONE_NAUTICAL_MILE_M))
            f_t = max(0.0, min(1.0, 1.0 - tcpa / T_HORIZON_S))
            cri = CRI_W_DCPA * f_d + CRI_W_TCPA * f_t
            if cri > worst_cri:
                worst_cri = cri
                worst_other = other
        if worst_other is None:
            return None
        return self.COLREGs_situation(worst_other)

    def _follow_waypoints(self, dt, verbose=False):
        """Steer to the next waypoint, or to the goal if none is left.

        Args:
            dt (float): Time step, s
            verbose (bool): Whether to print debug output
        """
        if not self.waypoints:
            self._steer_to_goal(dt, verbose)
            return

        self.prune_stale_waypoints(verbose=verbose)
        if not self.waypoints:
            self._steer_to_goal(dt, verbose)
            return

        x0, y0 = self.waypoints[0]
        acceptance_radius = WAYPOINT_ACCEPT_M

        dist_to_wp = np.hypot(self.x - x0, self.y - y0)

        wp_key = (_round1(x0), _round1(y0))
        if getattr(self, "_active_wp_key", None) != wp_key:
            self._active_wp_key = wp_key
            self._active_wp_min_dist = dist_to_wp
            self._active_wp_stall = 0
        bearing = math.atan2(float(y0 - self.y), float(x0 - self.x))
        d_th = bearing - float(self.theta)
        off_bow = abs(math.atan2(math.sin(d_th), math.cos(d_th)))

        if dist_to_wp < acceptance_radius:
            self.waypoints.pop(0)
            self._active_wp_key = None
        elif len(self.waypoints) > 1 and (
            (off_bow > _OFF_BOW_SKIP and dist_to_wp > self._active_wp_min_dist + 50.0)
            or self._active_wp_stall >= _WP_STALL_STEPS
        ):
            dropped = self.waypoints.pop(0)
            stalled = self._active_wp_stall >= _WP_STALL_STEPS
            resume = getattr(self, "resume_waypoint", None)
            if stalled and resume is not None and np.allclose(dropped, resume):
                self.resume_waypoint = None
            self._active_wp_key = None
        else:
            if off_bow > _OFF_BOW_SKIP:
                self._active_wp_stall += 1
            else:
                self._active_wp_stall = 0
            self._active_wp_min_dist = min(self._active_wp_min_dist, dist_to_wp)

        if not self.waypoints:
            self._steer_to_goal(dt, verbose)
        elif not self.final:
            self._steer_to_point(self.waypoints[0], dt, verbose=verbose)

    def _steer_to_goal(self, dt, verbose=False):
        """Steer to the goal unless already arrived.

        Args:
            dt (float): Time step, s
            verbose (bool): Whether to print debug output
        """
        if not self.final:
            self._steer_to_point([self.goal_x, self.goal_y], dt, verbose=verbose)

    def move_rrtstar(
        self,
        record_data,
        sea_objects,
        mmsi_list,
        rules,
        table,
        ax,
        eps,
        s,
        k,
        dt,
        verbose=False,
    ):
        """Move one step with RRT* and the COLREGs avoidance logic.

        Args:
            record_data (bool): Whether to store the trajectory
            sea_objects (list): All sea objects
            mmsi_list (list): Identifiers of all sea objects
            rules (list): Active rules table
            table (matplotlib.table.Table): Rules table display, or None
            ax (matplotlib.axes.Axes): Axes to draw on, or None
            eps (float): Safety margin added to the radius, m
            s (float): Half size of the map, m
            k (int): Current step index
            dt (float): Time step, s
            verbose (bool): Whether to print debug output

        Returns:
            list: Identifier, name, state vector, collision risk and history
        """
        self.s = s
        self._age_release(dt)

        dx_goal = self.goal_x - self.x
        dy_goal = self.goal_y - self.y
        if (dx_goal * dx_goal + dy_goal * dy_goal) < 90000.0 or self.final:
            if verbose:
                print("We are arrived at goal", self.goal_x, self.goal_y)
            self.v = 0.0
            self.update(np.array([[0], [0]]), dt)
            self.final = 1
            self.waypoints = []
            if self.log_obligations:
                self._obligations = []
                self._primary_threat = None
            return [
                self.mmsi,
                self.name,
                self.get_state_vector(),
                self.collision_risk,
                self.collision_history,
            ]

        self.in_collision = False
        self.collision_risk = 0

        step_situations = {}
        for other in sea_objects:
            if other is self:
                continue

            dx, dy = self.x - other.x, self.y - other.y
            dist_sq = dx * dx + dy * dy
            thresh = max(self.r, other.r) + eps * 2

            if dist_sq < (thresh * thresh):
                if verbose:
                    print("We are in collision between ", self, " and ", other)
                self.collision_risk = 1
                self.in_collision = True
                self.collision_coordinates.append([self.x, self.y])
                self.collision_coordinates.append([other.x, other.y])

            situation = self.COLREGs_situation(other)
            step_situations[id(other)] = situation
            self.list_colregs.append([self.mmsi, other.mmsi, situation])

        cpa_data = self._get_worst_cpa(sea_objects, step_situations)
        D_SAFE = self.D_SAFE_MULT * self.r
        safe_v = max(self.v, 0.5)

        if not cpa_data:
            self._giveway_role = None
            self._giveway_streak = 0

        if self.nav_phase == "TRANSIT" and cpa_data:
            tcpa, dcpa, _, threat_obj = cpa_data

            if tcpa < self.T_ALERT and dcpa < D_SAFE:
                tid = id(threat_obj)
                situation = (
                    step_situations[tid]
                    if tid in step_situations
                    else self.COLREGs_situation(threat_obj)
                )
                GIVEWAY = {
                    "Crossing_give_way",
                    "Head_on",
                    "Overtaking_give_way",
                    RELAXED_SITUATION,
                }

                if situation in GIVEWAY:
                    if situation == self._giveway_role:
                        self._giveway_streak += 1
                    else:
                        self._giveway_role = situation
                        self._giveway_streak = 1
                else:
                    self._giveway_role = None
                    self._giveway_streak = 0

                if self._giveway_streak >= self.K_COMMIT:
                    self._enter_avoidance(threat_obj, tcpa, self._giveway_role, verbose)

                elif self.emergency_avoidance and situation in {
                    "Crossing_stand_on",
                    "Being_overtaken_stand_on",
                }:
                    if self._in_extremis(threat_obj, tcpa, dcpa):
                        self._enter_avoidance(threat_obj, tcpa, situation, verbose)

        elif self.nav_phase == "AVOIDANCE":
            if cpa_data and cpa_data[3] is not self.threat_ts:
                self._retarget_avoidance(cpa_data, step_situations, verbose)
            goal_safe = not self._goal_directed_cpa_threat(sea_objects, verbose=verbose)

            if self.threat_ts is not None and not self.threat_ts.final:
                x_os, y_os = float(self.x), float(self.y)
                vx_os, vy_os = _vel(self)
                vx_ts, vy_ts = _vel(self.threat_ts)

                f_dcpa, f_tcpa = _cpa_xy(
                    x_os,
                    y_os,
                    vx_os,
                    vy_os,
                    float(self.threat_ts.x),
                    float(self.threat_ts.y),
                    vx_ts,
                    vy_ts,
                )

                if cpa_data is None:
                    t_cpa = max(f_tcpa, 0.0)
                    cpa_data = (
                        t_cpa,
                        f_dcpa,
                        (
                            x_os + vx_os * t_cpa,
                            y_os + vy_os * t_cpa,
                        ),
                        self.threat_ts,
                    )

                dx_t, dy_t = self.x - self.threat_ts.x, self.y - self.threat_ts.y
                physical_dist = (dx_t * dx_t + dy_t * dy_t) ** 0.5
            else:
                physical_dist = float("inf")

            resolved_now = goal_safe
            self._resolve_counter = (self._resolve_counter + 1) if resolved_now else 0
            if self._resolve_counter >= self.N_RESOLVE:
                self._release_avoidance()

            elif (
                self.threat_ts is None
                or self.threat_ts.final
                or physical_dist > self.VISIBLE_RANGE_M
            ):
                self._release_avoidance()

        if self.nav_phase == "TRANSIT":
            if self.waypoints:
                self._follow_waypoints(dt, verbose=verbose)
            else:
                self._steer_to_point([self.goal_x, self.goal_y], dt, verbose=verbose)

        elif self.nav_phase == "AVOIDANCE":
            path_is_invalid = True
            if self.waypoints:
                path_is_invalid = not self.planner._path_is_safe_and_compliant(
                    {"path": self.waypoints}, self, sea_objects, verbose=False
                )

            need_replan = (not self.waypoints) or (
                path_is_invalid
                and (self.avoidance_replan_counter % self.LOCAL_REPLAN_INTERVAL == 0)
            )
            if need_replan:
                if cpa_data:
                    tcpa, _, cpa_coords, threat_obj = cpa_data

                    if self.resume_waypoint is None:
                        self.resume_waypoint = self._compute_resume_waypoint(
                            threat_obj,
                            tcpa,
                            self.frozen_role or self.COLREGs_situation(threat_obj),
                            verbose=verbose,
                        )

                    new_path = self.planner.plan_local_avoidance(
                        self,
                        threat_obj,
                        cpa_coords,
                        tcpa,
                        self.resume_waypoint,
                        sea_objects,
                        self.record_path,
                        verbose=verbose,
                    )

                    if new_path and len(new_path) > 0:
                        self.waypoints = (
                            new_path + [[self.goal_x, self.goal_y]]
                            if new_path[-1] != [self.goal_x, self.goal_y]
                            else new_path
                        )
                        self.nb_replan += 1
                    else:
                        escape_dist = max(1000.0, safe_v * 120.0)
                        _side = self.avoidance_side if self.avoidance_side else +1
                        _h0 = (
                            self.avoidance_ref_heading
                            if self.avoidance_ref_heading is not None
                            else self.theta
                        )
                        escape_heading = self._normalize_ang(_h0 - _side * np.pi / 4)
                        self.waypoints = [
                            [
                                self.x + escape_dist * np.cos(escape_heading),
                                self.y + escape_dist * np.sin(escape_heading),
                            ]
                        ]

                elif self.resume_waypoint and not self.waypoints:
                    self.waypoints = [self.resume_waypoint, [self.goal_x, self.goal_y]]

            self.avoidance_replan_counter += 1
            self._follow_waypoints(dt, verbose=verbose)
            self.steps_count += 1
        if verbose:
            print(
                f"End of move_rrtstar step {self.steps_count}, current position = ({self.x:.1f}, {self.y:.1f}), velocity = {self.v:.1f}, heading = {np.degrees(self.theta):.1f}°, collision risk = {self.collision_risk}, nav_phase = {self.nav_phase}"
            )

        if self.log_obligations:
            self._record_obligations(sea_objects)

        return [
            self.mmsi,
            self.name,
            self.get_state_vector(),
            self.collision_risk,
            self.collision_history,
        ]

    def _on_emergency_handover(self):
        """Drop the plan when the emergency policy takes over."""
        self.waypoints = []
        self.nav_phase = "TRANSIT"
        self._clear_avoidance()

    def _release_tail_step(self, mmsi_list, rules, eps, s, k, dt):
        """Build the function that replays the compliant planner on copies.

        Args:
            mmsi_list (list): Identifiers of all sea objects
            rules (list): Active rules table
            eps (float): Safety margin added to the radius, m
            s (float): Half size of the map, m
            k (int): Current step index
            dt (float): Time step, s

        Returns:
            callable: Function that moves the copies one step
        """
        replay = []

        def step(ghosts):
            """Move the copies one step with the compliant planner.

            Args:
                ghosts (list): Copies of the sea objects
            """
            for ghost in ghosts:
                if ghost.mmsi == self.mmsi:
                    ghost.algo = COMPLIANT_TAIL_ALGO
            host = getattr(self, "_tail_host", None)
            host = host() if host is not None else None
            if host is not None:
                if not replay:
                    replay.append(host.tail_replayer())
                replay[0](ghosts)
                return
            for ghost in ghosts:
                ghost.move(
                    False, ghosts, mmsi_list, rules, None, None, eps, s, k, dt, False
                )

        return step

    def _move_hybrid_cert(
        self,
        record_data,
        sea_objects,
        mmsi_list,
        rules,
        table,
        ax,
        eps,
        s,
        k,
        dt,
        verbose: bool = False,
    ):
        """Move one step with the Hybrid controller and its certificate switch.

        Args:
            record_data (bool): Whether to store the trajectory
            sea_objects (list): All sea objects
            mmsi_list (list): Identifiers of all sea objects
            rules (list): Active rules table
            table (matplotlib.table.Table): Rules table display, or None
            ax (matplotlib.axes.Axes): Axes to draw on, or None
            eps (float): Safety margin added to the radius, m
            s (float): Half size of the map, m
            k (int): Current step index
            dt (float): Time step, s
            verbose (bool): Whether to print debug output

        Returns:
            list: Identifier, name, state vector, collision risk and history
        """
        from ..certificate.monitor import DECISION_HORIZON_S
        from ..certificate.cert_switch import DEFAULT_GATE, EMERGENCY, CertificateSwitch
        from ..rl.cert_obs import (
            ALPHA_COL_RELEASE,
            MARGIN_STEPS,
            N_TRACK,
            SHIELD_MARGIN_M,
            VETO_MARGIN_M,
        )

        if not hasattr(self, "_cert_switch"):
            self._cert_switch = CertificateSwitch(
                getattr(self, "hybrid_policy", None),
                gate=getattr(self, "hybrid_gate", DEFAULT_GATE),
                horizon=getattr(self, "hybrid_horizon", None) or DECISION_HORIZON_S,
                p_switch=getattr(self, "hybrid_p_switch", 0.0),
                trigger_k=getattr(self, "hybrid_trigger_k", None),
                release_lookahead=getattr(self, "hybrid_release_lookahead", 0),
                veto_margin=getattr(self, "hybrid_veto_margin", None) or VETO_MARGIN_M,
                action_shield=getattr(self, "hybrid_action_shield", 0),
                shield_margin=getattr(self, "hybrid_shield_margin", None)
                or SHIELD_MARGIN_M,
                shield_pick=getattr(self, "hybrid_shield_pick", None),
                margin_steps=getattr(self, "hybrid_margin_steps", None) or MARGIN_STEPS,
                n_track=getattr(self, "hybrid_n_track", None) or N_TRACK,
                speed_action=getattr(self, "hybrid_speed_action", None),
                alpha_col_release=(
                    ALPHA_COL_RELEASE
                    if getattr(self, "hybrid_alpha_col_release", None) is None
                    else float(self.hybrid_alpha_col_release)
                ),
                rule17b_relax=bool(getattr(self, "hybrid_rule17b_relax", False)),
                bin_fields=bool(getattr(self, "hybrid_bin_fields", False)),
                max_emergency_steps=getattr(self, "hybrid_max_emergency_steps", None),
                dt=dt,
                verbose=verbose,
            )
            self._hybrid_steps = 0
            self._hybrid_ppo_steps = 0

        if self.final:
            self._cert_switch.reset()
            return self.move_rrtstar(
                record_data,
                sea_objects,
                mmsi_list,
                rules,
                table,
                ax,
                eps,
                s,
                k,
                dt,
                verbose=verbose,
            )

        _tail_step = self._release_tail_step(mmsi_list, rules, eps, s, k, dt)

        decision = self._cert_switch.decide(self, sea_objects, tail_step=_tail_step)
        self._hybrid_steps += 1
        self.cert_state = decision["state"]
        self.cert_p_fire = decision["p_fire"]
        self.cert_fires_now = decision["fires_now"]

        if decision["state"] != EMERGENCY or "yaw" not in decision:
            return self.move_rrtstar(
                record_data,
                sea_objects,
                mmsi_list,
                rules,
                table,
                ax,
                eps,
                s,
                k,
                dt,
                verbose=verbose,
            )

        self._hybrid_ppo_steps += 1
        self._on_emergency_handover()
        self.update(np.array([[decision["accel"]], [decision["yaw"]]]), dt)
        self.steps_count += 1

        if self.log_obligations:
            self._record_obligations(sea_objects)
        if verbose:
            print(
                f"[hybrid-cert] EMERGENCY p_fire={decision['p_fire']:.2f} "
                f"alpha_phys/k0={decision['alpha_phys'] / decision['kappa0']:.3f} "
                f"yaw={np.degrees(decision['yaw']):+.1f}°/s v={self.v:.1f} m/s"
            )

        return [
            self.mmsi,
            self.name,
            self.get_state_vector(),
            self.collision_risk,
            self.collision_history,
        ]

    def move_hybrid(
        self,
        record_data,
        sea_objects,
        mmsi_list,
        rules,
        table,
        ax,
        eps,
        s,
        k,
        dt,
        tau_trigger: float = 0.5,
        tau_return: float = 0.3,
        N_hysteresis: int = 3,
        protocol: str = "hard",
        ppo_model=None,
        verbose: bool = False,
    ):
        """Move one step with the Hybrid controller.

        Args:
            record_data (bool): Whether to store the trajectory
            sea_objects (list): All sea objects
            mmsi_list (list): Identifiers of all sea objects
            rules (list): Active rules table
            table (matplotlib.table.Table): Rules table display, or None
            ax (matplotlib.axes.Axes): Axes to draw on, or None
            eps (float): Safety margin added to the radius, m
            s (float): Half size of the map, m
            k (int): Current step index
            dt (float): Time step, s
            tau_trigger (float): Unused, kept for old callers
            tau_return (float): Unused, kept for old callers
            N_hysteresis (int): Unused, kept for old callers
            protocol (str): Unused, kept for old callers
            ppo_model (object): Unused, kept for old callers
            verbose (bool): Whether to print debug output

        Returns:
            list: Identifier, name, state vector, collision risk and history
        """
        if getattr(self, "hybrid_certificate", False):
            return self._move_hybrid_cert(
                record_data,
                sea_objects,
                mmsi_list,
                rules,
                table,
                ax,
                eps,
                s,
                k,
                dt,
                verbose=verbose,
            )

        raise ValueError(
            "algo='hybrid' needs hybrid_certificate=True: the J_cascade arm is "
            "retired (ADR 0025)"
        )

    def prune_stale_waypoints(self, verbose=False):
        """Drop the waypoints already passed.

        Args:
            verbose (bool): Whether to print debug output
        """

        if verbose:
            print("Pruning stale waypoints, current position = ", (self.x, self.y))
        if not self.waypoints:
            if verbose:
                print("No waypoints to prune")
            return

        while len(self.waypoints) > 1:
            wx = float(self.waypoints[0][0]) - float(self.x)
            wy = float(self.waypoints[0][1]) - float(self.y)
            dist_to_wp = math.sqrt(wx * wx + wy * wy)

            if dist_to_wp < WAYPOINT_REACHED_M:
                self.waypoints.pop(0)
                continue

            th = float(self.theta)
            is_behind = wx * math.cos(th) + wy * math.sin(th) < 0
            if not is_behind:
                break
            self.waypoints.pop(0)

    def plot_waypoints(self):
        """Plot the waypoints and the current position."""
        if len(self.waypoints) != 0:
            x_waypoints = [wp[0] for wp in self.waypoints if wp is not None]
            y_waypoints = [wp[1] for wp in self.waypoints if wp is not None]
            plt.scatter(x_waypoints, y_waypoints, label="Waypoints", s=10, c="blue")
            plt.plot(self.x, self.y, "ro", markersize=5, label="Current Position")
            plt.title(f"Waypoints for MMSI: {self.mmsi}")
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            plt.legend()
            plt.grid()
            plt.gca().set_aspect("equal", adjustable="box")
            plt.axhline(0, color="black", lw=0.5)
            plt.axvline(0, color="black", lw=0.5)
            plt.show()
            plt.close()

    def update_history(self, lenght):
        """Add the current state to the history, keeping a fixed length.

        Args:
            lenght (int): Max history length
        """
        if len(self.history) == lenght:
            self.history.pop(0)
        self.history.append(self.get_state_vector())

    def bearing_to(self, target):
        """Compute the bearing from the vessel to a point.

        Args:
            target (tuple): Point (x, y), m

        Returns:
            float: Bearing, rad
        """
        dx = target[0] - self.x
        dy = target[1] - self.y
        return math.atan2(dy, dx)

    def waypoint_following(self, waypoints, dt, speed_setpoint=10.0):
        """Steer toward the first waypoint and record the path.

        Args:
            waypoints (list): Waypoints (x, y)
            dt (float): Time step, s
            speed_setpoint (float): Wanted speed, m/s

        Returns:
            list: Recorded path
        """
        current_wp = 0
        wp_radius = max(2.0, self.v * dt * 1.5)

        path = []
        i = 0
        if current_wp < len(waypoints):
            i += 1
            target = waypoints[current_wp]

            desired_heading = self.bearing_to(target)
            heading_error = (
                (desired_heading - self.theta + np.pi) % (2 * np.pi)
            ) - np.pi

            u2 = self.heading_pid.update(heading_error, dt)

            max_rudder_rate = np.deg2rad(5)
            u2 = np.clip(u2, -max_rudder_rate, max_rudder_rate)

            speed_error = speed_setpoint - self.v
            u1 = self.speed_pid.update(speed_error, dt)

            self.update(np.array([[u1], [u2]]), dt)

            distance = math.hypot(target[0] - self.x, target[1] - self.y)
            if distance < wp_radius:
                current_wp += 1

        return path

    def plot(
        self,
        algo=None,
        path=None,
        execution_time=None,
        iterations=0,
        sea_objects=None,
        save_path=None,
    ):
        """Plot the trajectory, the path and the other objects.

        Args:
            algo (str): Algorithm name, or None
            path (list): Planned path, or None
            execution_time (float): Planning time, s
            iterations (int): Number of iterations
            sea_objects (list): All sea objects, or None
            save_path (str): Output file, or None to show
        """
        plt.figure(figsize=(10, 8))
        i = 0
        for ox, oy, size in sea_objects:
            circle = plt.Circle(
                (ox, oy), size, color="gray", label="Obstacle" if i == 0 else ""
            )
            i += 1
            plt.gca().add_artist(circle)

        if path:
            path = np.array(path)
            for i in range(len(path) - 1):

                plt.plot(
                    path[i : i + 2, 0],
                    path[i : i + 2, 1],
                    "-xr",
                    label="Path" if i == 0 else "",
                )

        plt.plot(self.x, self.y, "xr", label="Boat")
        plt.plot(self.goal_x, self.goal_y, "xr", label="Goal")

        date_time = time.strftime("%Y%m%d_%H%M%S")
        if not os.path.exists("plots"):
            os.makedirs("plots")
        plt.grid(True)
        plt.xlim(-self.map_size, self.map_size)
        plt.ylim(-self.map_size, self.map_size)
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()
        plt.title(
            f"{algo} Path Planning\nExecution Time: {execution_time:.3f} seconds\nIterations: {iterations}"
        )

        if save_path is None:
            save_path = "plots"
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        algo_name = algo.replace("*", "star")
        plt.savefig(f"{save_path}/{algo}_{date_time}.png")
        plt.close()

    def normalize_angle(self, angle):
        """Wrap an angle into [-pi, pi].

        Args:
            angle (float): Angle, rad

        Returns:
            float: Wrapped angle, rad
        """
        return np.arctan2(np.sin(angle), np.cos(angle))

    def _normalize_ang(self, a: float) -> float:
        """Wrap an angle into [-pi, pi], faster for scalars.

        Args:
            a (float): Angle, rad

        Returns:
            float: Wrapped angle, rad
        """
        if isinstance(a, float):
            return np.float64(math.atan2(math.sin(a), math.cos(a)))
        return np.arctan2(np.sin(a), np.cos(a))

    def _cpa(self, pos1, vel1, pos2, vel2) -> Tuple[float, float]:
        """Compute DCPA and TCPA from positions and velocities.

        Args:
            pos1 (ndarray): Own position, m
            vel1 (ndarray): Own velocity, m/s
            pos2 (ndarray): Target position, m
            vel2 (ndarray): Target velocity, m/s

        Returns:
            Tuple[float, float]: DCPA (m) and TCPA (s)
        """
        return _cpa_xy(
            float(pos1[0]),
            float(pos1[1]),
            float(vel1[0]),
            float(vel1[1]),
            float(pos2[0]),
            float(pos2[1]),
            float(vel2[0]),
            float(vel2[1]),
        )

    def _cpa_numpy(self, pos1, vel1, pos2, vel2) -> Tuple[float, float]:
        """Numpy version of _cpa.

        Args:
            pos1 (ndarray): Own position, m
            vel1 (ndarray): Own velocity, m/s
            pos2 (ndarray): Target position, m
            vel2 (ndarray): Target velocity, m/s

        Returns:
            Tuple[float, float]: DCPA (m) and TCPA (s)
        """

        x_os, y_os = pos1
        x_ts, y_ts = pos2
        v_os_x, v_os_y = vel1
        v_ts_x, v_ts_y = vel2
        R = np.sqrt((x_ts - x_os) ** 2 + (y_ts - y_os) ** 2)

        v_x_rel = v_ts_x - v_os_x
        v_y_rel = v_ts_y - v_os_y

        relative_speed = np.sqrt(v_x_rel**2 + v_y_rel**2)

        psi_Vrel = np.arctan2(v_ts_y - v_os_y, v_ts_x - v_os_x)

        psi_LOS = np.arctan2(y_os - y_ts, x_os - x_ts)

        alpha = psi_LOS - psi_Vrel

        DCPA = abs(R * np.sin(alpha))
        if relative_speed != 0:
            TCPA = R * np.cos(alpha) / relative_speed
        else:
            TCPA = NO_CLOSING_TCPA_S

        return DCPA, TCPA

    def _rel_bearing(self, os_pos, ts_pos, os_hdg):
        """Compute the bearing of a target relative to the heading.

        Args:
            os_pos (ndarray): Own position, m
            ts_pos (ndarray): Target position, m
            os_hdg (float): Own heading, rad

        Returns:
            float: Relative bearing, rad
        """
        rel_pos = ts_pos - os_pos
        bearing = np.arctan2(rel_pos[1], rel_pos[0])
        return self._normalize_ang(bearing - os_hdg)

    _SITUATION_COMPLEMENT = {
        "Head_on": "Head_on",
        "Crossing_give_way": "Crossing_stand_on",
        "Crossing_stand_on": "Crossing_give_way",
        "Overtaking_give_way": "Being_overtaken_stand_on",
        "Being_overtaken_stand_on": "Overtaking_give_way",
    }

    def _commit_situation(self, other: "SeaObject", situation: str) -> str:
        """Store a situation, and its opposite on the other vessel.

        Args:
            other (SeaObject): Other vessel
            situation (str): COLREGs situation

        Returns:
            str: Stored situation
        """
        self.locked_situations[other.mmsi] = situation
        if self._rulebook_latch:
            return situation
        if not hasattr(other, "locked_situations"):
            other.locked_situations = {}
        other.locked_situations.setdefault(
            self.mmsi, self._SITUATION_COMPLEMENT.get(situation, situation)
        )
        return situation

    def _overtaking_latch_holds(self, other: "SeaObject") -> bool:
        """Check that a stored overtaking role is still active.

        Args:
            other (SeaObject): Other vessel

        Returns:
            bool: True if the role still holds
        """
        if not self._rulebook_latch:
            return False
        latched = self.locked_situations.get(other.mmsi)
        if latched == "Overtaking_give_way":
            return not _duty.finally_past_and_clear(self, other)
        if latched == "Being_overtaken_stand_on":
            return not _duty.finally_past_and_clear(other, self)
        return False

    def COLREGs_situation(self, other: "SeaObject") -> str:
        """Get the COLREGs situation with another vessel.

        Args:
            other (SeaObject): Other vessel

        Returns:
            str: COLREGs situation
        """
        situation = self._rule_situation(other)
        if self.rule_mode is RuleMode.RELAXED and situation != "Safe":
            return RELAXED_SITUATION
        return situation

    def _rule_situation(self, other: "SeaObject") -> str:
        """Get the situation, keeping latched roles until past and clear.

        Args:
            other (SeaObject): Other vessel

        Returns:
            str: COLREGs situation
        """
        if not hasattr(self, "locked_situations"):
            self.locked_situations = {}
        if not self._rulebook_latch:
            return self._sector_situation(other)

        label = self._held_past_cpa(other)
        if label is None:
            label = self._sector_situation(other)
        if label == "Safe":
            self._past_cpa_range.pop(other.mmsi, None)
            return label

        sit = _ROLE_SITUATION.get(
            self._duty_role_of(other, self.overtaking.get(other.mmsi))
        )
        if sit is None or sit == label:
            return label
        self.locked_situations[other.mmsi] = sit
        return sit

    @property
    def _past_cpa_range(self):
        """Get the dict of ranges stored after CPA.

        Returns:
            dict: Range per vessel identifier
        """
        ref = self.__dict__.get("_past_cpa_ref")
        if ref is None:
            ref = self.__dict__["_past_cpa_ref"] = {}
        return ref

    def _held_past_cpa(self, other: "SeaObject"):
        """Keep the latched situation after CPA until the range opens.

        Args:
            other (SeaObject): Other vessel

        Returns:
            str: Latched situation, or None
        """
        latched = self.locked_situations.get(other.mmsi)
        if not latched:
            return None
        vxo, vyo = _vel(self)
        vxt, vyt = _vel(other)
        _d, tcpa = _cpa_xy(
            float(self.x),
            float(self.y),
            vxo,
            vyo,
            float(other.x),
            float(other.y),
            vxt,
            vyt,
        )
        if tcpa > 0:
            self._past_cpa_range.pop(other.mmsi, None)
            return None
        rng = math.hypot(other.x - self.x, other.y - self.y)
        r0 = self._past_cpa_range.setdefault(other.mmsi, rng)
        if rng < r0 + LATCH_OPEN_M and rng <= self.VISIBLE_RANGE_M:
            return latched
        return None

    def _duty_role_of(
        self, other: "SeaObject", memory=None, gate=PORT_VETO_GATE_M
    ) -> str:
        """Get the duty role with another vessel from the duty model.

        Args:
            other (SeaObject): Other vessel
            memory (object): Overtaking memory, or None
            gate (float): Distance under which a duty applies, m

        Returns:
            str: Role name, or NONE on failure
        """
        try:
            ov = _DutyVessel.from_sea_object(self, r_min=R_MIN_M)
            tv = _DutyVessel.from_sea_object(other, r_min=R_MIN_M)
            return _duty.role(ov, tv, gate=gate, memory=memory)
        except Exception:
            return "NONE"

    def _sector_situation(self, other: "SeaObject") -> str:
        """Classify the encounter from bearing sectors, DCPA and TCPA.

        Args:
            other (SeaObject): Other vessel

        Returns:
            str: COLREGs situation
        """
        OVERTAKING_ENTER = _OVERTAKING_ENTER
        HEAD_ON_ANGLE = _HEAD_ON_ANGLE
        SAFE_DCPA_THRESHOLD = self.SAFE_DCPA_M
        SAFE_TCPA_THRESHOLD = max(
            self.SAFE_TCPA_MIN_S, getattr(self, "T_ALERT", LATCH_TCPA_S)
        )
        VISIBLE_RANGE = self.VISIBLE_RANGE_M

        xo, yo, xt, yt = float(self.x), float(self.y), float(other.x), float(other.y)
        vxo, vyo = _vel(self)
        vxt, vyt = _vel(other)
        rdx, rdy = xt - xo, yt - yo
        rng = math.sqrt(rdx * rdx + rdy * rdy)
        if rng > VISIBLE_RANGE:
            self.locked_situations.pop(other.mmsi, None)
            return "Safe"

        dcpa, tcpa = _cpa_xy(xo, yo, vxo, vyo, xt, yt, vxt, vyt)

        if tcpa <= 0 and not self._overtaking_latch_holds(other):
            self.locked_situations.pop(other.mmsi, None)
            return "Safe"
        if other.mmsi not in self.locked_situations:
            if dcpa > SAFE_DCPA_THRESHOLD or tcpa > SAFE_TCPA_THRESHOLD:
                return "Safe"

        if other.mmsi in self.locked_situations:
            return self.locked_situations[other.mmsi]

        making_low = MAKING_WAY_MPS - MAKING_WAY_BAND_MPS
        making_high = MAKING_WAY_MPS + MAKING_WAY_BAND_MPS
        if tcpa > 0 and dcpa < SAFE_DCPA_THRESHOLD:
            if other.v < making_low and self.v >= making_high:
                return self._commit_situation(other, "Crossing_give_way")
            if self.v < making_low and other.v >= making_high:
                return self._commit_situation(other, "Crossing_stand_on")

        if self._rulebook_latch:
            role = self._duty_role_of(other, self.overtaking.get(other.mmsi), gate=None)
            situation = _ROLE_SITUATION.get(role, "Safe")
            if situation != "Safe":
                return self._commit_situation(other, situation)
            return situation

        bearing_os_to_ts = self._normalize_ang(math.atan2(rdy, rdx) - float(self.theta))
        bearing_ts_to_os = self._normalize_ang(
            math.atan2(yo - yt, xo - xt) - float(other.theta)
        )

        heading_diff = abs(self._normalize_ang(self.theta - other.theta))
        reciprocal = abs(heading_diff - np.pi) < _RECIPROCAL_TOL

        situation = "Safe"
        if abs(bearing_ts_to_os) > OVERTAKING_ENTER:
            situation = "Overtaking_give_way"
        elif abs(bearing_os_to_ts) > OVERTAKING_ENTER:
            situation = "Being_overtaken_stand_on"
        elif abs(bearing_os_to_ts) < HEAD_ON_ANGLE and reciprocal:
            situation = "Head_on"
        elif -OVERTAKING_ENTER < bearing_os_to_ts < 0:
            situation = "Crossing_give_way"
        elif 0 < bearing_os_to_ts < OVERTAKING_ENTER:
            situation = "Crossing_stand_on"

        if situation != "Safe":
            return self._commit_situation(other, situation)

        return situation

import os
import math
import numpy as np
import copy
import warnings
import matplotlib.pyplot as plt
from .algo_rttstar import RRTStar, Node as RRTNode, SpatialGrid as RRTSpatialGrid
from .rrt_spatial_grid import _nb as _rrt_nb


def _np_clip(x, lo, hi):
    """Clip a value like numpy, keeping NaN.

    Args:
        x (float): Value
        lo (float): Lower bound
        hi (float): Upper bound

    Returns:
        float: Clipped value
    """
    x, lo, hi = float(x), float(lo), float(hi)
    m = x if (x != x or x > lo) else lo
    return m if (m != m or m < hi) else hi


def _append_samples(blocks, other, tcpa, sample_i, n_samples):
    """Add the predicted positions of a target as obstacle discs.

    Args:
        blocks (list): List of obstacle arrays, updated in place
        other (SeaObject): Target vessel
        tcpa (float): Time to closest approach, s
        sample_i (ndarray): Sample indexes
        n_samples (int): Number of samples
    """
    _vx = other.v * np.cos(other.theta)
    _vy = other.v * np.sin(other.theta)
    _r_obs = getattr(other, "r", 75.0) * 1.5
    t_sample = tcpa * sample_i / n_samples
    blocks.append(
        np.column_stack(
            (
                other.x + _vx * t_sample,
                other.y + _vy * t_sample,
                np.full(n_samples + 1, _r_obs, dtype=float),
            )
        )
    )


from ..grid_search.algo_astar import AStar
from ...simulator.experiment_constants import MAKING_WAY_MPS, TUBE_DCPA_MULT, TUBE_TCPA_S

DOMAIN_AXES_LENGTHS = (4.0, 2.0, 1.5)
CORRIDOR_MAX_M = 5000.0


class COLREGsPlanner:
    def __init__(self, prediction_horizon=10, safety_margin=1.0):
        """Set the prediction horizon and the safety margin.

        Args:
            prediction_horizon (float): Prediction horizon
            safety_margin (float): Safety margin factor
        """
        self.prediction_horizon = prediction_horizon
        self.safety_margin = safety_margin
        self.previous_paths = []

    def calculate_cpa(self, own_ship, other_ship):
        """Compute the closest point of approach with another vessel.

        Args:
            own_ship (SeaObject): Own vessel
            other_ship (SeaObject): Target vessel

        Returns:
            tuple: TCPA (s), DCPA (m) and own position at CPA
        """
        dx = other_ship.x - own_ship.x
        dy = other_ship.y - own_ship.y
        dvx = other_ship.v * np.cos(other_ship.theta) - own_ship.v * np.cos(
            own_ship.theta
        )
        dvy = other_ship.v * np.sin(other_ship.theta) - own_ship.v * np.sin(
            own_ship.theta
        )

        rel_vel_sq = dvx**2 + dvy**2

        if rel_vel_sq < 1e-12:
            dcpa = np.sqrt(dx**2 + dy**2)
            return 0.0, dcpa, (own_ship.x, own_ship.y)

        tcpa = -(dx * dvx + dy * dvy) / rel_vel_sq
        if tcpa < 0:
            tcpa = 0.0

        dcpa_x = dx + dvx * tcpa
        dcpa_y = dy + dvy * tcpa
        dcpa = np.sqrt(dcpa_x**2 + dcpa_y**2)

        cpa_coordinates = (
            own_ship.x + own_ship.v * np.cos(own_ship.theta) * tcpa,
            own_ship.y + own_ship.v * np.sin(own_ship.theta) * tcpa,
        )

        return tcpa, dcpa, cpa_coordinates

    def _get_effective_length(self, own_ship, other_ship):
        """Compute the core radius of a target and its distance.

        Args:
            own_ship (SeaObject): Own vessel
            other_ship (SeaObject): Target vessel

        Returns:
            tuple: Core radius and distance, m
        """
        dist = max(np.hypot(own_ship.x - other_ship.x, own_ship.y - other_ship.y), 1e-6)
        core_r = max(getattr(other_ship, "r", 75), dist * 0.05)
        return core_r, dist

    def _compute_domain_axes(self, other_ship, situation):
        """Compute the size of the ship domain around a target.

        Args:
            other_ship (SeaObject): Target vessel
            situation (str): COLREGs situation

        Returns:
            tuple: Forward, aft and side lengths and COLREGs offset, m
        """
        vessel_length = max(getattr(other_ship, "r", 75) * 2, 50.0)
        fwd_len, aft_len, lat_len = getattr(
            other_ship, "DOMAIN_AXES_LENGTHS", DOMAIN_AXES_LENGTHS
        )
        a_fwd = max(fwd_len * vessel_length, 300.0)
        a_aft = max(aft_len * vessel_length, 150.0)
        c_lat = max(lat_len * vessel_length, 100.0)
        d_colregs = (
            c_lat * 0.5
            if situation in ["Head_on", "Crossing_give_way", "Overtaking_give_way"]
            else 0.0
        )
        return a_fwd, a_aft, c_lat, d_colregs

    def generate_asymmetric_tube(self, own_ship, other_ship, situation, verbose=False):
        """Build obstacle discs that push own vessel to the correct side.

        Args:
            own_ship (SeaObject): Own vessel
            other_ship (SeaObject): Target vessel
            situation (str): COLREGs situation
            verbose (bool): Whether to print debug output

        Returns:
            list: Obstacle discs (x, y, radius)
        """
        core_r, dist = self._get_effective_length(own_ship, other_ship)
        obstacles = [(other_ship.x, other_ship.y, core_r)]
        if verbose:
            print(
                f"Generating asymmetric tube for situation '{situation}' with core radius {core_r:.1f}m and distance {dist:.1f}m."
            )
        if situation in ["Safe", "Crossing_stand_on", "Being_overtaken_stand_on"]:
            if verbose:
                print(
                    f"Situation '{situation}' does not require an asymmetric tube. Returning single obstacle at target vessel."
                )
            return obstacles

        tcpa, dcpa, _ = self.calculate_cpa(own_ship, other_ship)

        a_fwd, a_aft, c_lat, d_colregs = self._compute_domain_axes(
            other_ship, situation
        )

        heading = other_ship.theta
        fwd_x = np.cos(heading)
        fwd_y = np.sin(heading)
        ts_port_x = np.cos(other_ship.theta + np.pi / 2)
        ts_port_y = np.sin(other_ship.theta + np.pi / 2)

        step_size = max(c_lat / 2.0, 1e-4)
        if verbose:
            print(
                f"Computed tube parameters: a_fwd={a_fwd:.1f}m, a_aft={a_aft:.1f}m, c_lat={c_lat:.1f}m, d_colregs={d_colregs:.1f}m, step_size={step_size:.1f}m"
            )
            print(
                "Generating forward obstacles along the TS track with lateral offset for COLREGs compliance..."
            )
        for s in np.arange(0.0, a_fwd, step_size):
            t = s / a_fwd
            r_circle = c_lat * np.sqrt(max(1.0 - t**2, 0.0))
            if r_circle < 1e-4:
                continue
            cx = other_ship.x + fwd_x * s - ts_port_x * d_colregs
            cy = other_ship.y + fwd_y * s - ts_port_y * d_colregs
            obstacles.append((cx, cy, r_circle))
        if verbose:
            print(
                f"Generated {len(obstacles)-1} forward obstacles along the TS track for the asymmetric tube."
            )
        for s in np.arange(step_size, a_aft, step_size):
            t = s / a_aft
            r_circle = c_lat * np.sqrt(max(1.0 - t**2, 0.0))
            if r_circle < 1e-4:
                continue
            obstacles.append(
                (other_ship.x - fwd_x * s, other_ship.y - fwd_y * s, r_circle)
            )
        if verbose:
            print(
                f"Generated {len(obstacles)-1} aft obstacles along the TS track for the asymmetric tube."
            )
        return obstacles

    def generate_no_cross_corridor(
        self,
        own_ship,
        other_ship,
        situation,
        tcpa,
        wall_thickness=3,
        verbose=False,
    ):
        """Build a wall of discs ahead of the target so the path does not cross its bow.

        Args:
            own_ship (SeaObject): Own vessel
            other_ship (SeaObject): Target vessel
            situation (str): COLREGs situation
            tcpa (float): Time to closest approach, s
            wall_thickness (int): Number of disc rows
            verbose (bool): Whether to print debug output

        Returns:
            list: Obstacle discs (x, y, radius)
        """
        if situation not in ("Crossing_give_way", "Head_on", "Overtaking_give_way"):
            if verbose:
                print(f"Situation is {situation}, no-cross corridor not needed.")
            return []
        if tcpa <= 0:
            if verbose:
                print(f"TCPA is {tcpa:.1f}s, no-cross corridor not needed.")
            return []

        fwd_x = np.cos(other_ship.theta)
        fwd_y = np.sin(other_ship.theta)

        rel_x = own_ship.x - other_ship.x
        rel_y = own_ship.y - other_ship.y
        inter_dist = np.hypot(rel_x, rel_y)

        if situation == "Head_on":
            os_fwd_x = np.cos(own_ship.theta)
            os_fwd_y = np.sin(own_ship.theta)
            perp_x = -os_fwd_y
            perp_y = os_fwd_x
        else:
            cross = fwd_x * rel_y - fwd_y * rel_x
            if cross >= 0:
                perp_x = -fwd_y
                perp_y = fwd_x
            else:
                perp_x = fwd_y
                perp_y = -fwd_x

        corridor_length = min(other_ship.v * tcpa, CORRIDOR_MAX_M)

        base_radius = max(getattr(own_ship, "r", 75.0), getattr(other_ship, "r", 75.0))
        obs_radius = max(200.0, base_radius * 2.0)

        obstacles = []
        n_along = max(int(corridor_length / (obs_radius * 1.2)), 5)

        if verbose:
            print(
                f"Building continuous no-cross wall: length={corridor_length:.1f}m, obs_radius={obs_radius:.1f}m"
            )

        for i in range(n_along + 1):
            s = corridor_length * i / n_along
            bx = other_ship.x + fwd_x * s
            by = other_ship.y + fwd_y * s

            for layer in range(wall_thickness):
                offset = layer * obs_radius * 1.2
                obstacles.append(
                    (bx + perp_x * offset, by + perp_y * offset, obs_radius)
                )

        return obstacles

    def _get_heading_change(self, segment_start, segment_end, current_heading_rad):
        """Compute the turn needed to follow a segment.

        Args:
            segment_start (tuple): Segment start
            segment_end (tuple): Segment end
            current_heading_rad (float): Current heading, rad

        Returns:
            float: Heading change, rad, positive to port
        """
        dx = segment_end[0] - segment_start[0]
        dy = segment_end[1] - segment_start[1]

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0

        seg = math.atan2(dy, dx)
        cur = math.atan2(math.sin(current_heading_rad), math.cos(current_heading_rad))
        planned = math.atan2(math.sin(seg), math.cos(seg))
        d = planned - cur
        return math.atan2(math.sin(d), math.cos(d))

    def is_starboard_maneuver(self, segment_start, segment_end, own_ship):
        """Check that a segment turns own vessel to starboard.

        Args:
            segment_start (tuple): Segment start
            segment_end (tuple): Segment end
            own_ship (SeaObject): Own vessel

        Returns:
            bool: True for a starboard turn
        """
        delta_heading = self._get_heading_change(
            segment_start, segment_end, own_ship.theta
        )
        return delta_heading < 0

    def passes_behind_stand_on_vessel(
        self, segment_start, segment_end, own_ship, stand_on_ship
    ):
        """Check that a segment passes astern of the stand on vessel.

        Args:
            segment_start (tuple): Segment start
            segment_end (tuple): Segment end
            own_ship (SeaObject): Own vessel
            stand_on_ship (SeaObject): Stand on vessel

        Returns:
            bool: True if the segment passes astern
        """
        rx = stand_on_ship.x - own_ship.x
        ry = stand_on_ship.y - own_ship.y
        if math.sqrt(rx * rx + ry * ry) < 1e-6:
            return True

        ux = math.cos(stand_on_ship.theta)
        uy = math.sin(stand_on_ship.theta)
        projection = rx * ux + ry * uy

        nx = rx - (segment_end[0] - segment_start[0])
        ny = ry - (segment_end[1] - segment_start[1])
        return nx * ux + ny * uy <= projection

    def overtakes_on_correct_side(
        self, segment_start, segment_end, own_ship, other_ship
    ):
        """Check that an overtaking segment stays on the allowed side.

        Args:
            segment_start (tuple): Segment start
            segment_end (tuple): Segment end
            own_ship (SeaObject): Own vessel
            other_ship (SeaObject): Vessel being overtaken

        Returns:
            bool: True if the side is correct
        """
        ux = math.cos(other_ship.theta)
        uy = math.sin(other_ship.theta)
        rx = own_ship.x - other_ship.x
        ry = own_ship.y - other_ship.y
        nx = rx + (segment_end[0] - segment_start[0])
        ny = ry + (segment_end[1] - segment_start[1])
        return (ux * ry - uy * rx < 0) and (ux * ny - uy * nx < 0)

    def edge_situations(self, own_ship, sea_objects):
        """List the COLREGs situation with every other vessel, with a cache.

        Args:
            own_ship (SeaObject): Own vessel
            sea_objects (list): All sea objects

        Returns:
            list: Pairs (vessel, situation)
        """
        _state = (id(own_ship), own_ship.x, own_ship.y, own_ship.theta, own_ship.v)
        if getattr(self, "_sit_state", None) != _state:
            self._sit_state, self._sit_cache = _state, {}
        _latch = getattr(own_ship, "locked_situations", None) or {}

        out = []
        for other_ship in sea_objects:
            if other_ship == own_ship:
                continue

            _key = (
                other_ship.mmsi,
                other_ship.x,
                other_ship.y,
                other_ship.theta,
                other_ship.v,
                _latch.get(other_ship.mmsi),
            )
            situation = self._sit_cache.get(_key)
            if situation is None:
                situation = own_ship.COLREGs_situation(other_ship)
                self._sit_cache[_key] = situation
            out.append((other_ship, situation))
        return out

    def colregs_violation_penalty(
        self, own_ship, segment_start, segment_end, sea_objects
    ):
        """Sum the penalties of a segment that breaks COLREGs.

        Args:
            own_ship (SeaObject): Own vessel
            segment_start (tuple): Segment start
            segment_end (tuple): Segment end
            sea_objects (list): All sea objects

        Returns:
            float: Total penalty
        """
        penalty = 0.0
        turn_tolerance_rad = 0.05

        for other_ship, situation in self.edge_situations(own_ship, sea_objects):
            if situation == "Head_on":
                if not self.is_starboard_maneuver(segment_start, segment_end, own_ship):
                    penalty += 100000

            elif situation == "Crossing_give_way":
                if not self.is_starboard_maneuver(segment_start, segment_end, own_ship):
                    penalty += 50000
                if not self.passes_behind_stand_on_vessel(
                    segment_start, segment_end, own_ship, other_ship
                ):
                    penalty += 50000

            elif situation == "Overtaking_give_way":
                if not self.overtakes_on_correct_side(
                    segment_start, segment_end, own_ship, other_ship
                ):
                    penalty += 30000

            elif situation in ["Crossing_stand_on", "Being_overtaken_stand_on"]:
                delta_heading = self._get_heading_change(
                    segment_start, segment_end, own_ship.theta
                )
                if delta_heading > turn_tolerance_rad:
                    penalty += 200000
                elif delta_heading < -turn_tolerance_rad:
                    penalty += 25000

        return penalty

    def _path_still_relevant(self, path, own_ship, max_deviation=2.0, verbose=False):
        """Check that own vessel is still close to a stored path.

        Args:
            path (list): Path points
            own_ship (SeaObject): Own vessel
            max_deviation (float): Max cross track error, m
            verbose (bool): Whether to print debug output

        Returns:
            bool: True if the path can be kept
        """
        if verbose:
            print(
                f"Checking path relevance: own_ship position=({own_ship.x:.1f}, {own_ship.y:.1f}), max_deviation={max_deviation:.1f}m"
            )
        if not path or len(path) < 2:
            if verbose:
                print("Path is too short to be relevant, treating as not relevant.")
            return False

        min_cross_track_error = float("inf")
        sx, sy = float(own_ship.x), float(own_ship.y)

        for i in range(len(path) - 1):
            x1, y1 = float(path[i][0]), float(path[i][1])
            x2, y2 = float(path[i + 1][0]), float(path[i + 1][1])
            gx, gy = x2 - x1, y2 - y1
            hx, hy = sx - x1, sy - y1

            segment_len_sq = gx * gx + gy * gy

            if segment_len_sq < 1e-9:
                dist_to_p1 = np.float64(math.sqrt(hx * hx + hy * hy))
                min_cross_track_error = min(min_cross_track_error, dist_to_p1)
                continue

            t = (hx * gx + hy * gy) / segment_len_sq

            if t < 0:
                cx, cy = x1, y1
            elif t > 1:
                cx, cy = x2, y2
            else:
                cx, cy = x1 + t * gx, y1 + t * gy

            dx, dy = sx - cx, sy - cy
            distance_to_segment = np.float64(math.sqrt(dx * dx + dy * dy))
            min_cross_track_error = min(min_cross_track_error, distance_to_segment)
        if verbose:
            print(f"Minimum cross-track error to path: {min_cross_track_error:.1f}m")
            print(
                f"Path is {'still relevant' if min_cross_track_error <= max_deviation else 'no longer relevant'} based on cross-track error."
            )
        return min_cross_track_error <= max_deviation

    def plan_rrt_with_enhanced_obstacles(
        self,
        own_ship,
        obstacles,
        map_size,
        sea_objects=None,
        colregs_situation=None,
        record_path="plots",
        previous_path=None,
        opti=False,
        verbose=False,
    ):
        """Plan an RRT* path around the obstacles and COLREGs discs.

        Args:
            own_ship (SeaObject): Own vessel
            obstacles (list): Obstacle discs (x, y, radius)
            map_size (float): Half size of the map, m
            sea_objects (list): All sea objects, or None
            colregs_situation (str): COLREGs situation, or None
            record_path (str): Folder for saved plots
            previous_path (dict): Last path, or None
            opti (bool): Whether to keep improving the path
            verbose (bool): Whether to print debug output

        Returns:
            tuple: Path points and path structure
        """
        _safety_margin = getattr(own_ship, "r", 75.0) + 200
        cleared_obstacles = []
        if verbose:
            print(
                f"Planning RRT* with enhanced obstacles. Original number of obstacles: {len(obstacles)}. Applying safety margin of {_safety_margin:.1f}m around own ship."
            )

        for ox, oy, r_obs in obstacles:
            dist_to_start = np.hypot(ox - own_ship.x, oy - own_ship.y)
            if dist_to_start < r_obs + _safety_margin:
                new_r = max(0.0, dist_to_start - _safety_margin - 1.0)
                if new_r > 1e-4:
                    cleared_obstacles.append((ox, oy, new_r))
            else:
                cleared_obstacles.append((ox, oy, r_obs))
        obstacles = cleared_obstacles
        if verbose:
            print(
                f"Number of obstacles after applying safety margin: {len(obstacles)}. Obstacles that were too close to start have been shrunk or removed."
            )
        max_x = max(obstacle[0] for obstacle in obstacles) if obstacles else 0
        max_x = max(max_x, own_ship.goal_x + 1, own_ship.x + 1, map_size[0] * 1.5)
        min_x = min(obstacle[0] for obstacle in obstacles) if obstacles else 0
        min_x = min(min_x, own_ship.goal_x - 1, own_ship.x - 1, -map_size[0] * 1.5)
        max_y = max(obstacle[1] for obstacle in obstacles) if obstacles else 0
        max_y = max(max_y, own_ship.goal_y + 1, own_ship.y + 1, map_size[1] * 1.5)
        min_y = min(obstacle[1] for obstacle in obstacles) if obstacles else 0
        min_y = min(min_y, own_ship.goal_y - 1, own_ship.y - 1, -map_size[1] * 1.5)

        map_size_param = max(max_x, max_y, -min_x, -min_y)
        if verbose:
            print(
                f"Calculated map size parameter for RRT*: {map_size_param:.1f}m (based on obstacle extents and start/goal positions)"
            )

        rrt = RRTStar(
            start=[own_ship.x, own_ship.y],
            goal=[own_ship.goal_x, own_ship.goal_y],
            num_obstacles=len(obstacles),
            map_size=[map_size_param, map_size_param],
            step_size=200,
            obstacles=obstacles,
            max_iter=2000,
            opti=opti,
        )
        rrt.safety_margin = _safety_margin
        rrt.search_radius = 200
        rrt.colregs_planner = self
        rrt.own_ship = own_ship
        rrt.sea_objects = sea_objects
        rrt.plot = verbose
        rrt.opti = opti
        rrt.plot = False

        if (
            previous_path
            and isinstance(previous_path, dict)
            and "path" in previous_path
        ):
            if verbose:
                print(
                    "Previous path provided, checking if it is still valid and relevant..."
                )
            if (
                previous_path["path"]
                and rrt.is_path_valid(previous_path["path"])
                and self._path_still_relevant(previous_path["path"], own_ship)
            ):
                if verbose:
                    print(
                        "Previous path is still valid and relevant. Reusing it for RRT* initialization."
                    )
                rrt.path = previous_path["path"]
            else:
                if verbose:
                    print(
                        "Previous path is no longer valid or relevant. RRT* will plan from scratch."
                    )
                rrt.plan()
        else:
            if verbose:
                print("No previous path provided, planning from scratch.")
            rrt.plan()

        object_path = rrt.save_situation_as_obj(rrt.path, colregs_situation)
        self.previous_paths.append(object_path)
        if len(self.previous_paths) > 50:
            del self.previous_paths[:-50]

        if rrt.path:
            if verbose:
                print(f"RRT* found a path with {len(rrt.path)} waypoints.")
                print(
                    "First waypoint: ({:.1f}, {:.1f}), Last waypoint: ({:.1f}, {:.1f})".format(
                        rrt.path[0][0], rrt.path[0][1], rrt.path[-1][0], rrt.path[-1][1]
                    )
                )
            return rrt.path, object_path
        else:
            if verbose:
                print(
                    "RRT* failed to find a path. Returning only the goal position as a fallback."
                )
            return [[own_ship.goal_x, own_ship.goal_y]], object_path

    def _segment_intersects_circle(self, p1, p2, center, radius):
        """Check whether a segment crosses a circle.

        Args:
            p1 (ndarray): Segment start
            p2 (ndarray): Segment end
            center (ndarray): Circle centre
            radius (float): Circle radius

        Returns:
            bool: True if they intersect
        """
        d = p2 - p1
        f = p1 - center
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - radius**2
        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return False

        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a + 1e-9)
        t2 = (-b + discriminant) / (2 * a + 1e-9)

        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            return True
        return t1 < 0 < 1 < t2

    def _path_is_safe_and_compliant(
        self, path_structure, own_ship, sea_objects, verbose=False
    ):
        """Check that a stored path still avoids every vessel and follows COLREGs.

        Args:
            path_structure (dict): Stored path
            own_ship (SeaObject): Own vessel
            sea_objects (list): All sea objects
            verbose (bool): Whether to print debug output

        Returns:
            bool: True if the path can be kept
        """
        if verbose:
            print(
                "We are in _path_is_safe_and_compliant with path_structure:",
                path_structure,
            )
        if (
            not path_structure
            or "path" not in path_structure
            or not path_structure["path"]
        ):
            if verbose:
                print("No valid path found in path_structure, treating as unsafe.")
            return False

        path = path_structure["path"]
        kinematic_tolerance = max(getattr(own_ship, "r", 75.0), 100.0)
        if not self._path_still_relevant(
            path, own_ship, max_deviation=kinematic_tolerance, verbose=verbose
        ):
            if verbose:
                print(
                    "Path is no longer relevant to current position, treating as unsafe."
                )
            return False

        current_obstacles = []

        for other in sea_objects:
            if other is own_ship:
                continue

            dist_to_other = np.hypot(own_ship.x - other.x, own_ship.y - other.y)
            if dist_to_other < (
                getattr(own_ship, "r", 10.0) + getattr(other, "r", 10.0) + 1.0
            ):
                if verbose:
                    print(
                        f"Other ship at ({other.x:.1f}, {other.y:.1f}) is too close to own ship, treating path as unsafe."
                    )
                    print(
                        "Distance to other ship: {:.1f}m, combined radius + margin: {:.1f}m".format(
                            dist_to_other,
                            getattr(own_ship, "r", 10.0)
                            + getattr(other, "r", 10.0)
                            + 1.0,
                        )
                    )
                return False

            situation = own_ship.COLREGs_situation(other)

            if situation != "Safe":
                if verbose:
                    print(
                        f"Other ship at ({other.x:.1f}, {other.y:.1f}) is a COLREGs threat (situation: {situation}), generating asymmetric tube obstacles."
                    )
                zones = self.generate_asymmetric_tube(
                    own_ship, other, situation, verbose=verbose
                )
                current_obstacles.extend(zones)

            current_obstacles.append(
                (other.x, other.y, getattr(other, "r", 10.0) + 1.5)
            )

        for i in range(len(path) - 1):
            p1 = np.array(path[i])
            p2 = np.array(path[i + 1])
            for ox, oy, r in current_obstacles:
                if self._segment_intersects_circle(p1, p2, np.array([ox, oy]), r):
                    if verbose:
                        print(
                            f"Path segment from ({p1[0]:.1f}, {p1[1]:.1f}) to ({p2[0]:.1f}, {p2[1]:.1f}) intersects obstacle at ({ox:.1f}, {oy:.1f}) with radius {r:.1f}, treating path as unsafe."
                        )
                    return False

        return True

    def plan_local_avoidance(
        self,
        own_ship,
        threat_obj,
        cpa_coords,
        tcpa,
        resume_waypoint,
        sea_objects,
        record_path="plots",
        verbose=False,
    ):
        """Plan a short RRT* detour around one threat back to a waypoint.

        Args:
            own_ship (SeaObject): Own vessel
            threat_obj (SeaObject): Threat vessel
            cpa_coords (tuple): Position at CPA
            tcpa (float): Time to closest approach, s
            resume_waypoint (tuple): Waypoint to rejoin
            sea_objects (list): All sea objects
            record_path (str): Folder for saved plots
            verbose (bool): Whether to print debug output

        Returns:
            list: Path points, empty if no path is found
        """

        if verbose:
            print(f"Planning local avoidance: TCPA={tcpa:.1f}s")

        min_x = min(own_ship.x, threat_obj.x, resume_waypoint[0])
        max_x = max(own_ship.x, threat_obj.x, resume_waypoint[0])
        min_y = min(own_ship.y, threat_obj.y, resume_waypoint[1])
        max_y = max(own_ship.y, threat_obj.y, resume_waypoint[1])

        padding = (
            max(getattr(own_ship, "r", 75.0), getattr(threat_obj, "r", 75.0)) * 8.0
            + 1500.0
        )

        heading_diff = abs(
            (own_ship.theta - threat_obj.theta + np.pi) % (2 * np.pi) - np.pi
        )
        if (
            heading_diff > np.deg2rad(120.0)
            and own_ship.COLREGs_situation(threat_obj) != "Head_on"
        ):
            padding += 2000.0

        cpa_x = (min_x + max_x) / 2.0
        cpa_y = (min_y + max_y) / 2.0

        grid_half_width = (max_x - min_x) / 2.0 + padding
        grid_half_height = (max_y - min_y) / 2.0 + padding

        grid_half = max(grid_half_width, grid_half_height)
        grid_half = min(grid_half, own_ship.map_size)

        if verbose:
            print(
                f"Dynamic grid bounded to objects: center=({cpa_x:.1f}, {cpa_y:.1f}), half-size={grid_half:.1f}m"
            )

        local_goal_w = [
            _np_clip(resume_waypoint[0], cpa_x - grid_half, cpa_x + grid_half),
            _np_clip(resume_waypoint[1], cpa_y - grid_half, cpa_y + grid_half),
        ]

        frozen = getattr(own_ship, "frozen_role", None)
        situation = (
            frozen
            if frozen and threat_obj is getattr(own_ship, "threat_ts", None)
            else own_ship.COLREGs_situation(threat_obj)
        )
        obstacles_world = []
        N_TRAJ_SAMPLES = 50
        _sample_i = np.arange(N_TRAJ_SAMPLES + 1)
        others, extra, extra_after = [], [], -1

        for other in sea_objects:
            if other is own_ship:
                continue

            if _rrt_nb is not None:
                others.append(
                    (
                        other.x,
                        other.y,
                        other.v,
                        other.theta,
                        getattr(other, "r", 75.0) * 1.5,
                    )
                )
                extra_after = len(others) - 1 if other is threat_obj else extra_after
                blocks = extra
            else:
                _append_samples(obstacles_world, other, tcpa, _sample_i, N_TRAJ_SAMPLES)
                blocks = obstacles_world

            if (
                other is threat_obj
                and situation
                not in (
                    "Safe",
                    "Crossing_stand_on",
                    "Being_overtaken_stand_on",
                )
                and getattr(other, "v", 0.0) >= MAKING_WAY_MPS
            ):
                try:
                    corridor = self.generate_no_cross_corridor(
                        own_ship,
                        other,
                        situation,
                        tcpa,
                        wall_thickness=3,
                        verbose=verbose,
                    )
                    blocks.append(np.array(corridor, dtype=float).reshape(-1, 3))
                except Exception as e:
                    if verbose:
                        print(f"No-cross corridor generation failed: {e}")

                future_ts_obj = copy.copy(other)
                future_ts_obj.x = other.x + other.v * np.cos(other.theta) * tcpa
                future_ts_obj.y = other.y + other.v * np.sin(other.theta) * tcpa
                try:
                    tube = self.generate_asymmetric_tube(
                        own_ship, future_ts_obj, situation, verbose=verbose
                    )
                    blocks.append(np.array(tube, dtype=float).reshape(-1, 3))
                except Exception as e:
                    if verbose:
                        print(f"Tube generation failed: {e}")

        _safety_margin = getattr(own_ship, "r", 75.0) + 200
        sx, sy = cpa_x, cpa_y
        start_local = [own_ship.x - sx, own_ship.y - sy]
        goal_local = [local_goal_w[0] - sx, local_goal_w[1] - sy]
        _bounds = [-grid_half - 50, grid_half + 50, -grid_half - 50, grid_half + 50]
        _tmp_grid = RRTSpatialGrid(_bounds, cell_size=100)

        if _rrt_nb is not None:
            _b = np.array([float(b) for b in _bounds])
            _cell = float(_tmp_grid.cell_size)
            obs_local, nxm, nym, g_start, g_items = _rrt_nb.local_obstacles(
                float(own_ship.x),
                float(own_ship.y),
                np.array(others, dtype=float).reshape(-1, 5),
                float(tcpa),
                N_TRAJ_SAMPLES,
                np.concatenate(extra) if extra else np.empty((0, 3)),
                extra_after,
                float(_safety_margin),
                float(sx),
                float(sy),
                _b[0],
                _b[1],
                _b[2],
                _b[3],
                _cell,
            )
            _tmp_grid.add_obstacle_array(
                obs_local, csr=(obs_local, _b, _cell, nxm, nym, g_start, g_items)
            )
        else:
            _w = (
                np.concatenate(obstacles_world) if obstacles_world else np.empty((0, 3))
            )
            dist_to_start = np.hypot(_w[:, 0] - own_ship.x, _w[:, 1] - own_ship.y)
            inside = dist_to_start < _w[:, 2] + _safety_margin
            new_r = np.maximum(0.0, dist_to_start - _safety_margin - 1.0)
            keep = ~inside | (new_r > 1e-4)
            r_cleared = np.where(inside, new_r, _w[:, 2])
            obs_local = np.column_stack(
                (_w[keep, 0] - sx, _w[keep, 1] - sy, r_cleared[keep])
            )
            _tmp_grid.add_obstacle_array(obs_local)

        if not _tmp_grid.is_collision_free(
            goal_local[0], goal_local[1], _safety_margin
        ):
            astern_local_x = np.cos(threat_obj.theta + np.pi)
            astern_local_y = np.sin(threat_obj.theta + np.pi)

            for _step in range(1, 30):
                _gx = goal_local[0] + astern_local_x * 200 * _step
                _gy = goal_local[1] + astern_local_y * 200 * _step
                if abs(_gx) < grid_half and abs(_gy) < grid_half:
                    if _tmp_grid.is_collision_free(_gx, _gy, _safety_margin):
                        goal_local = [_gx, _gy]
                        break

        step_size = max(100.0, grid_half * 0.05)
        search_radius = max(200.0, grid_half * 0.10)

        rrt = RRTStar(
            start=start_local,
            goal=goal_local,
            num_obstacles=len(obs_local),
            map_size=[grid_half, grid_half],
            step_size=step_size,
            max_iter=1500,
            spatial_grid=_tmp_grid,
        )
        rrt.safety_margin = _safety_margin
        rrt.search_radius = search_radius
        rrt.colregs_planner = self
        rrt.own_ship = own_ship
        rrt.sea_objects = sea_objects
        rrt.plot = False
        rrt.opti = True

        ref = (
            own_ship._compliant_reference()
            if hasattr(own_ship, "_compliant_reference")
            else None
        )
        if ref is not None:
            _rx, _ry, _h0, _side = ref
            rrt.compliant_ref = (_rx - sx, _ry - sy)
            rrt.compliant_heading = _h0
            rrt.compliant_side = _side
            rrt.compliant_tol = step_size
            rrt.compliant_hold = bool(getattr(own_ship, "extremis_side_hold", False))

        if _rrt_nb is not None and _rrt_nb.supports(rrt):
            rrt.obstacles = obs_local
        else:
            rrt.obstacles = [tuple(row) for row in obs_local.tolist()]
        rrt.plan()

        try:
            for _n in rrt.start_tree + rrt.goal_tree:
                _n.x += sx
                _n.y += sy
            if isinstance(rrt.obstacles, np.ndarray):
                _o = rrt.obstacles
                rrt.obstacles = np.column_stack(
                    (_o[:, 0] + sx, _o[:, 1] + sy, _o[:, 2])
                )
            else:
                rrt.obstacles = [(ox + sx, oy + sy, r) for ox, oy, r in rrt.obstacles]
            rrt.start.x += sx
            rrt.start.y += sy
            rrt.goal.x += sx
            rrt.goal.y += sy
            if rrt.path:
                rrt.path = [[p[0] + sx, p[1] + sy] for p in rrt.path]
            self.previous_paths.append(rrt.save_situation_as_obj(rrt.path, situation))
            if len(self.previous_paths) > 50:
                del self.previous_paths[:-50]
        except Exception as e:
            if verbose:
                print(f"RRT* snapshot capture failed: {e}")

        if rrt.path:
            return [[p[0], p[1]] for p in rrt.path]
        return []

    def enhanced_colreg_planning(
        self,
        own_ship,
        sea_objects,
        map_size,
        record_path="plots",
        algo="RRT*",
        calculated_colregs=None,
        previous_path=None,
        opti=False,
        verbose=False,
    ):
        """Build the COLREGs obstacles and plan, or keep the last path if still valid.

        Args:
            own_ship (SeaObject): Own vessel
            sea_objects (list): All sea objects
            map_size (float): Half size of the map, m
            record_path (str): Folder for saved plots
            algo (str): Planner name
            calculated_colregs (str): Situation forced by the caller, or None
            previous_path (dict): Last path, or None
            opti (bool): Whether to keep improving the path
            verbose (bool): Whether to print debug output

        Returns:
            tuple: Path points, path structure and situation
        """
        enhanced_obstacles = []
        situation = "Safe"

        for other_ship in sea_objects:
            if other_ship == own_ship:
                continue
            enhanced_obstacles.extend(
                [(other_ship.x, other_ship.y, getattr(other_ship, "r", 10.0) * 1.5)]
            )

            if calculated_colregs:
                situation = calculated_colregs
            else:
                situation = own_ship.COLREGs_situation(other_ship)

            if situation in ["Crossing_stand_on", "Being_overtaken_stand_on"]:
                continue
            tcpa, dcpa, _ = self.calculate_cpa(own_ship, other_ship)
            SAFE_DCPA = getattr(own_ship, "r", 75.0) * 2.0
            if verbose:
                print(
                    f"Evaluating COLREGs situation with {other_ship.name if hasattr(other_ship, 'name') else 'unknown object'}: situation={situation}, TCPA={tcpa:.1f}s, DCPA={dcpa:.1f}m"
                )
                print(
                    f"Using SAFE_DCPA threshold of {SAFE_DCPA:.1f}m for enhanced obstacle generation."
                )
            if 0 < tcpa < TUBE_TCPA_S and dcpa < SAFE_DCPA * TUBE_DCPA_MULT:
                if verbose:
                    print(
                        "tcpa treshold is 900s and dcpa threshold is {}m (3x SAFE_DCPA)".format(
                            SAFE_DCPA * 3
                        )
                    )
                enhanced_obstacles.extend(
                    self.generate_asymmetric_tube(
                        own_ship, other_ship, situation, verbose=verbose
                    )
                )
                if verbose:
                    print("Added asymmetric tube obstacles for this encounter.")
                try:
                    if verbose:
                        print(
                            "Attempting to generate no-cross corridor for this encounter (situation: {}, TCPA: {:.1f}s)".format(
                                situation, tcpa
                            )
                        )
                    corridor = self.generate_no_cross_corridor(
                        own_ship,
                        other_ship,
                        situation,
                        tcpa,
                        wall_thickness=3,
                        verbose=verbose,
                    )
                    enhanced_obstacles.extend(corridor)
                except Exception as e:
                    warnings.warn(f"No-cross corridor generation failed: {e}")

        if previous_path and self._path_is_safe_and_compliant(
            previous_path, own_ship, sea_objects, verbose=verbose
        ):
            if verbose:
                print(
                    "Previous path is still safe and compliant with COLREGs. Reusing it."
                )
            return previous_path["path"], previous_path, situation

        if algo == "RRT*":
            if verbose:
                print(
                    "Planning with RRT* using enhanced obstacles. Total number of obstacles:",
                    len(enhanced_obstacles),
                )
            plan, object_path = self.plan_rrt_with_enhanced_obstacles(
                own_ship,
                enhanced_obstacles,
                map_size,
                sea_objects=sea_objects,
                colregs_situation=situation,
                record_path=record_path,
                previous_path=previous_path,
                opti=opti,
                verbose=verbose,
            )
            return plan, object_path, situation

        elif algo == "A*":
            astar = AStar(
                start=[own_ship.x, own_ship.y],
                goal=[own_ship.goal_x, own_ship.goal_y],
                num_obstacles=len(enhanced_obstacles),
                map_size=map_size,
                grid_resolution=0.5,
                safety_margin=0.2,
                obstacles=enhanced_obstacles,
            )
            astar.plan()
            return (
                astar.path if astar.path else [[own_ship.goal_x, own_ship.goal_y]],
                None,
                situation,
            )
        raise ValueError(
            f"enhanced_colreg_planning: unknown algo {algo!r}, want 'RRT*' or 'A*'"
        )

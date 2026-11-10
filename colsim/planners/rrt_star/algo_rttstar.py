import math
import os
import numpy as np
import random
import time
import copy
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import scipy.interpolate as interp
from ...simulator.experiment_constants import WAYPOINT_ACCEPT_M
from .rrt_spatial_grid import SpatialGrid
from .rrt_spatial_grid import _nb as _rrt_nb

COLREGS_EDGE_PENALTY = False

_NEIGHBOUR_SLACK = 1e-9
_XY_MIN_CAPACITY = 256


LIVE_SEARCH_PLOT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "output", "rrt_search"
)


class Node:
    def __init__(self, x, y):
        """Create a tree node at a point.

        Args:
            x (float): Node x, m
            y (float): Node y, m
        """
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0
        self.children = []


class PathOptimizer:

    @staticmethod
    def prune(path, validator):
        """Remove path points that can be skipped without a higher cost.

        Args:
            path (list): Path points
            validator (MaritimeValidator): Checks segments and costs

        Returns:
            list: Shorter path
        """
        if len(path) < 3:
            return path
        optimized = [path[0]]
        i = 0
        while i < len(path) - 1:
            furthest = i + 1
            for j in range(i + 2, len(path)):
                n_i = Node(*path[i])
                n_j = Node(*path[j])

                if validator.is_segment_valid(n_i, n_j):
                    dist = np.hypot(n_j.x - n_i.x, n_j.y - n_i.y)
                    cost = validator.get_cost(n_i, n_j)

                    if cost > dist + 10.0:
                        break

                    furthest = j
                else:
                    break
            optimized.append(path[furthest])
            i = furthest
        return optimized

    @staticmethod
    def first_leg_or(path, fallback, own_ship):
        """Use the fallback path if the first leg points astern.

        Args:
            path (list): Path points
            fallback (list): Path to use instead
            own_ship (SeaObject): Own vessel, or None

        Returns:
            list: Path to follow
        """
        if own_ship is None or len(path) < 2:
            return path
        (x0, y0), (x1, y1) = path[0][:2], path[1][:2]
        astern = (
            abs(
                math.remainder(
                    math.atan2(y1 - y0, x1 - x0) - own_ship.theta, 2 * math.pi
                )
            )
            > math.pi / 2
        )
        if astern and math.hypot(x1 - x0, y1 - y0) >= WAYPOINT_ACCEPT_M:
            return fallback
        return path

    @staticmethod
    def bspline_smooth(path, resolution=200):
        """Smooth a path with a B-spline.

        Args:
            path (list): Path points
            resolution (float): Distance between output points, m

        Returns:
            list: Smoothed path
        """
        if len(path) < 3:
            return path
        pts = np.array(path)
        dist = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        pts = np.vstack([pts[0], pts[1:][dist > 1e-4]])

        if len(pts) < 3:
            return path

        try:
            tck, u = interp.splprep(
                [pts[:, 0], pts[:, 1]], s=0.5, k=min(3, len(pts) - 1)
            )
            u_new = np.linspace(0, 1, max(int(np.sum(dist) / resolution), len(pts)))
            new_pts = interp.splev(u_new, tck)
            smoothed = list(zip(new_pts[0], new_pts[1]))
            return smoothed
        except Exception as e:
            return path

    @staticmethod
    def elastic_band(path, validator, iterations=50):
        """Smooth a path by pulling each point toward its neighbours.

        Args:
            path (list): Path points
            validator (MaritimeValidator): Checks that points stay free
            iterations (int): Max number of passes

        Returns:
            list: Smoothed path
        """
        if len(path) < 3:
            return path
        xs = [float(p[0]) for p in path]
        ys = [float(p[1]) for p in path]
        free = validator.grid.is_collision_free
        margin = validator.safety_margin
        last = len(xs) - 1
        for _ in range(iterations):
            moved = False
            for i in range(1, last):
                nx = xs[i] + 0.5 * (xs[i - 1] + xs[i + 1] - 2 * xs[i])
                ny = ys[i] + 0.5 * (ys[i - 1] + ys[i + 1] - 2 * ys[i])
                if nx == xs[i] and ny == ys[i]:
                    continue
                if free(nx, ny, margin):
                    xs[i], ys[i] = nx, ny
                    moved = True

            if not moved:
                break
        return [[x, y] for x, y in zip(xs, ys)]


class MaritimeValidator:
    def __init__(
        self, spatial_grid, own_ship=None, sea_objects=None, planner=None, start_xy=None
    ):
        """Store the grid and the context used to check nodes and segments.

        Args:
            spatial_grid (SpatialGrid): Obstacle grid
            own_ship (SeaObject): Own vessel, or None
            sea_objects (list): All sea objects, or None
            planner (COLREGsPlanner): Gives the COLREGs penalties, or None
            start_xy (tuple): Start point, or None to use own position
        """
        self.grid = spatial_grid
        self.own_ship = own_ship
        self.start_xy = start_xy
        self.sea_objects = sea_objects
        self.planner = planner
        self.safety_margin = 0.0

    def is_node_valid(self, node):
        """Check that a node is free.

        Args:
            node (Node): Node to check

        Returns:
            bool: True if free
        """
        return self.grid.is_collision_free(node.x, node.y, self.safety_margin)

    def is_segment_valid(self, n1, n2):
        """Check that a segment is free and does not start behind own vessel.

        Args:
            n1 (Node): Segment start
            n2 (Node): Segment end

        Returns:
            bool: True if the segment is valid
        """
        if self.own_ship is not None:
            sx, sy = (
                self.start_xy
                if self.start_xy is not None
                else (self.own_ship.x, self.own_ship.y)
            )
            far = None
            if abs(n1.x - sx) < 0.1 and abs(n1.y - sy) < 0.1:
                far = n2
            elif abs(n2.x - sx) < 0.1 and abs(n2.y - sy) < 0.1:
                far = n1
            if far is not None:
                seg_heading = np.arctan2(far.y - sy, far.x - sx)

                diff = (seg_heading - self.own_ship.theta + np.pi) % (2 * np.pi) - np.pi

                if abs(diff) > np.pi / 2.0:
                    return False

        dist = np.hypot(n2.x - n1.x, n2.y - n1.y)
        steps = max(3, int(dist / 50))
        for i in range(1, steps):
            t = i / steps
            x = n1.x + t * (n2.x - n1.x)
            y = n1.y + t * (n2.y - n1.y)
            if not self.grid.is_collision_free(x, y, self.safety_margin):
                return False
        return True

    def get_cost(self, n1, n2):
        """Compute the length of a segment plus its COLREGs penalty.

        Args:
            n1 (Node): Segment start
            n2 (Node): Segment end

        Returns:
            float: Segment cost
        """
        dist = np.hypot(n2.x - n1.x, n2.y - n1.y)
        penalty = 0.0
        if COLREGS_EDGE_PENALTY and self.planner and self.own_ship and self.sea_objects:
            try:
                penalty = self.planner.colregs_violation_penalty(
                    self.own_ship, [n1.x, n1.y], [n2.x, n2.y], self.sea_objects
                )
            except Exception:
                pass
        return dist + penalty

    def is_path_valid(self, path):
        """Check every node and segment of a path.

        Args:
            path (list): Path points

        Returns:
            bool: True if the whole path is valid
        """
        if not path or len(path) < 2:
            return False
        for i in range(len(path) - 1):
            n1 = Node(path[i][0], path[i][1])
            n2 = Node(path[i + 1][0], path[i + 1][1])
            if not self.is_node_valid(n1) or not self.is_node_valid(n2):
                return False
            if not self.is_segment_valid(n1, n2):
                return False
        return True


class RRTStar:
    def __init__(
        self,
        start,
        goal,
        num_obstacles,
        map_size,
        step_size,
        obstacles=None,
        max_iter=1000,
        opti=False,
        plot=False,
        spatial_grid=None,
    ):
        """Set up the planner, its obstacles and its grid.

        Args:
            start (tuple): Start point (x, y)
            goal (tuple): Goal point (x, y)
            num_obstacles (int): Number of random obstacles if none are given
            map_size (tuple): Half size of the map along x and y
            step_size (float): Max edge length, m
            obstacles (list): Obstacle discs (x, y, radius), or None
            max_iter (int): Max iterations
            opti (bool): Whether to keep improving the path
            plot (bool): Whether to print and plot the search
            spatial_grid (SpatialGrid): Obstacle grid, or None to build one
        """
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.map_size = map_size
        self.max_iter = max_iter
        self.step_size = step_size
        self.opti = opti
        self.plot = plot

        if self.plot:
            print(f"[RRT* Init] Initializing RRT* from {start} to {goal}")
            print(
                f"[RRT* Init] Params: max_iter={max_iter}, step_size={step_size}, opti={opti}"
            )
            print(
                f"[RRT* Init] Map size: {map_size}, initial obstacle count: {num_obstacles}"
            )

        bounds = [
            -map_size[0] - 50,
            map_size[0] + 50,
            -map_size[1] - 50,
            map_size[1] + 50,
        ]
        self.obstacles = obstacles or []
        self.spatial_grid = spatial_grid
        if self.spatial_grid is None:
            self.spatial_grid = SpatialGrid(bounds, cell_size=100)
            for ox, oy, size in self.obstacles:
                self.spatial_grid.add_obstacle(ox, oy, size)

        self.validator = MaritimeValidator(
            self.spatial_grid, start_xy=(self.start.x, self.start.y)
        )
        self.start_tree = [self.start]
        self.goal_tree = [Node(*goal)]
        self.path = None
        self.connection_nodes = None
        self._search_radius = 5.0

        self.compliant_ref = None
        self.compliant_heading = None
        self.compliant_side = 0
        self.compliant_hold = False
        self._compliant_active = False

    @property
    def colregs_planner(self):
        """Get the COLREGs planner of the validator.

        Returns:
            COLREGsPlanner: Planner, or None
        """
        return self.validator.planner

    @colregs_planner.setter
    def colregs_planner(self, value):
        """Set the COLREGs planner of the validator.

        Args:
            value (COLREGsPlanner): Planner, or None
        """
        self.validator.planner = value

    @property
    def own_ship(self):
        """Get own vessel from the validator.

        Returns:
            SeaObject: Own vessel, or None
        """
        return self.validator.own_ship

    @own_ship.setter
    def own_ship(self, value):
        """Set own vessel on the validator.

        Args:
            value (SeaObject): Own vessel
        """
        self.validator.own_ship = value

    @property
    def sea_objects(self):
        """Get the sea objects from the validator.

        Returns:
            list: All sea objects
        """
        return self.validator.sea_objects

    @sea_objects.setter
    def sea_objects(self, value):
        """Set the sea objects on the validator.

        Args:
            value (list): All sea objects
        """
        self.validator.sea_objects = value

    @property
    def safety_margin(self):
        """Get the safety margin of the validator.

        Returns:
            float: Extra clearance, m
        """
        return self.validator.safety_margin

    @safety_margin.setter
    def safety_margin(self, value):
        """Set the safety margin of the validator.

        Args:
            value (float): Extra clearance, m
        """
        self.validator.safety_margin = value

    @property
    def search_radius(self):
        """Get the rewire radius.

        Returns:
            float: Rewire radius, m
        """
        return self._search_radius

    @search_radius.setter
    def search_radius(self, value):
        """Set the rewire radius.

        Args:
            value (float): Rewire radius, m
        """
        self._search_radius = value

    def is_path_valid(self, path):
        """Check a path with the validator.

        Args:
            path (list): Path points

        Returns:
            bool: True if the path is valid
        """
        return self.validator.is_path_valid(path)

    def plot_live_search(self):
        """Plot the obstacles, both trees and the path, then save the figure."""
        if self.plot:
            print("[RRT* Plot] Generating live search plot...")

        scale_factor = 1000.0
        plt.figure(figsize=(10, 10))
        ax = plt.gca()

        all_x, all_y = [], []

        for ox, oy, size in self.obstacles:
            s_ox, s_oy, s_size = (
                ox / scale_factor,
                oy / scale_factor,
                size / scale_factor,
            )
            circle = plt.Circle(
                (s_ox, s_oy), s_size, color="black", alpha=0.5, clip_on=True
            )
            ax.add_artist(circle)
            all_x.extend([s_ox - s_size, s_ox + s_size])
            all_y.extend([s_oy - s_size, s_oy + s_size])

        for tree, color in [(self.start_tree, "blue"), (self.goal_tree, "green")]:
            for node in tree:
                s_nx, s_ny = node.x / scale_factor, node.y / scale_factor
                all_x.append(s_nx)
                all_y.append(s_ny)
                if node.parent:
                    plt.plot(
                        [s_nx, node.parent.x / scale_factor],
                        [s_ny, node.parent.y / scale_factor],
                        color=color,
                        alpha=0.3,
                        linewidth=0.5,
                        marker="o",
                        markersize=3,
                    )

        plt.scatter(
            self.start.x / scale_factor,
            self.start.y / scale_factor,
            c="blue",
            marker="o",
            s=100,
            label="Start",
            zorder=5,
        )
        plt.scatter(
            self.goal.x / scale_factor,
            self.goal.y / scale_factor,
            c="green",
            marker="x",
            s=100,
            label="Goal",
            zorder=5,
        )
        all_x.extend([self.start.x / scale_factor, self.goal.x / scale_factor])
        all_y.extend([self.start.y / scale_factor, self.goal.y / scale_factor])

        if self.path:
            path_x = [p[0] / scale_factor for p in self.path]
            path_y = [p[1] / scale_factor for p in self.path]
            plt.plot(
                path_x,
                path_y,
                color="red",
                linewidth=2,
                label="Final Path",
                marker="o",
                markersize=5,
                zorder=4,
            )

        if all_x and all_y:
            buffer = 2.0
            ax.set_xlim(min(all_x) - buffer, max(all_x) + buffer)
            ax.set_ylim(min(all_y) - buffer, max(all_y) + buffer)

        plt.title(f"RRT* Search State (Path Found: {self.path is not None})")
        plt.xlabel("X (km)")
        plt.ylabel("Y (km)")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()

        folder = LIVE_SEARCH_PLOT_DIR
        if not os.path.exists(folder):
            os.makedirs(folder)
        unique_name = f"{folder}/rrtstar_search_{int(time.time())}.png"
        plt.savefig(unique_name, bbox_inches="tight")
        plt.close()
        plt.close("all")
        if self.plot:
            print("[RRT* Plot] Search state successfully saved to:", unique_name)

    def _rebuild_grid(self):
        """Rebuild the obstacle grid and keep the validator context."""
        if self.plot:
            print(
                f"[RRT* Core] Rebuilding spatial grid with {len(self.obstacles)} obstacles..."
            )

        bounds = [
            -self.map_size[0] - 50,
            self.map_size[0] + 50,
            -self.map_size[1] - 50,
            self.map_size[1] + 50,
        ]
        old_planner = self.validator.planner
        old_own_ship = self.validator.own_ship
        old_sea_objects = self.validator.sea_objects
        old_safety_margin = self.validator.safety_margin

        self.spatial_grid = SpatialGrid(bounds, cell_size=100)
        for ox, oy, size in self.obstacles:
            self.spatial_grid.add_obstacle(ox, oy, size)
        self.validator = MaritimeValidator(
            self.spatial_grid, start_xy=(self.start.x, self.start.y)
        )
        self.validator.planner = old_planner
        self.validator.own_ship = old_own_ship
        self.validator.sea_objects = old_sea_objects
        self.validator.safety_margin = old_safety_margin

        if self.plot:
            print("[RRT* Core] Spatial grid rebuild complete.")

    def _clear_start(self):
        """Shrink the obstacles that cover the start point."""
        if self.plot:
            print(
                f"[RRT* Core] Checking if start node at ({self.start.x:.1f}, {self.start.y:.1f}) is inside any obstacles..."
            )

        safety = self.validator.safety_margin
        new_obstacles = []
        for ox, oy, r in self.obstacles:
            d = np.hypot(ox - self.start.x, oy - self.start.y)
            if d < r + safety:
                new_r = max(0.0, d - safety - 1.0)
                if self.plot:
                    print(
                        f"[RRT* Core] Start node trapped! Shrinking obstacle at ({ox:.1f}, {oy:.1f}) from {r:.1f} to {new_r:.1f}"
                    )
                if new_r > 1e-4:
                    new_obstacles.append((ox, oy, new_r))
            else:
                new_obstacles.append((ox, oy, r))
        self.obstacles = new_obstacles
        self._rebuild_grid()

    def _walk_goal_to_free(self):
        """Move the goal toward the start until it is free.

        Returns:
            bool: True if a free goal was found
        """

        if self.plot:
            print(
                f"[RRT* Core] Goal at ({self.goal.x:.1f}, {self.goal.y:.1f}) is invalid. Attempting to walk goal to free space..."
            )

        dx = self.start.x - self.goal.x
        dy = self.start.y - self.goal.y
        dist = np.hypot(dx, dy)
        if dist < 1e-6:
            if self.plot:
                print("[RRT* Core] Goal walk failed: Distance to start is zero.")
            return False

        ux, uy = dx / dist, dy / dist
        step_m = 200.0
        max_steps = max(int(dist / step_m), 1)

        for step in range(1, max_steps + 1):
            cx = self.goal.x + ux * step_m * step
            cy = self.goal.y + uy * step_m * step
            candidate = Node(cx, cy)
            if self.validator.is_node_valid(candidate):
                self.goal = candidate
                self.goal_tree = [Node(cx, cy)]
                if self.plot:
                    print(
                        f"[RRT* Core] Success! Goal moved to free space at ({cx:.1f}, {cy:.1f}) in {step} steps."
                    )
                return True

        if self.plot:
            print(
                "[RRT* Core] Goal walk failed: No free space found between start and goal."
            )
        return False

    def plan(self):
        """Search a path with a two tree RRT*, using numba when possible.

        Returns:
            list: Path points, or None if no path is found
        """
        if _rrt_nb is not None and _rrt_nb.supports(self):
            return _rrt_nb.plan(self)
        if self.plot:
            print("=" * 100)
            print(f"[RRT* Plan] Starting path planning sequence...")

        self.start_tree = [self.start]
        self.goal_tree = [Node(self.goal.x, self.goal.y)]
        self._xy = {}
        self._compliant_active = (
            self.compliant_side != 0 and self.compliant_ref is not None
        )

        if self.plot:
            print("[RRT* Plan] Validating goal node...")

        if not self.validator.is_node_valid(self.goal):
            if not self._walk_goal_to_free():
                if self.plot:
                    print(
                        "[RRT* Plan] Aborting plan: Goal node is hopelessly trapped inside an obstacle."
                    )
                return None

        if self.plot:
            print("[RRT* Plan] Checking if direct straight-line path is valid...")

        if (
            self.validator.is_segment_valid(self.start, self.goal)
            and (not self._compliant_active or self._node_compliant(self.goal))
            and self._direct_is_lawful()
        ):
            self.path = [[self.start.x, self.start.y], [self.goal.x, self.goal.y]]
            if self.plot:
                print(
                    "[RRT* Plan] Direct path from start to goal is valid. No search tree needed."
                )
                self.plot_live_search()
                time.sleep(0.5)
            return self.path

        if self.plot:
            print(
                f"[RRT* Plan] Direct path blocked. Initiating Bi-directional RRT* search loop (Max iterations: {self.max_iter})..."
            )

        for i in range(self.max_iter):
            if self.plot and i > 0 and i % 500 == 0:
                print(
                    f"[RRT* Plan] Search in progress... Iteration {i}/{self.max_iter}"
                )
            if (
                self._compliant_active
                and not self.compliant_hold
                and i == self.max_iter // 2
            ):
                self._compliant_active = False
                if self.plot:
                    print(
                        "[RRT* Plan] Compliant half-plane unsolved at half budget "
                        "— relaxing constraint to preserve collision-avoidance."
                    )

            active, target = (
                (self.start_tree, self.goal_tree)
                if i % 2 == 0
                else (self.goal_tree, self.start_tree)
            )

            sample = self._sample()
            nearest = self._get_nearest(sample, active)
            new_node = self._steer(nearest, sample)

            if self.validator.is_segment_valid(nearest, new_node):
                neighbors = self._get_neighbors(new_node, active)
                self._choose_parent(new_node, neighbors, nearest)
                active.append(new_node)
                self._rewire(new_node, neighbors, active)

                if self._attempt_connect(new_node, target):
                    if self.plot:
                        print(
                            f"[RRT* Plan] BINGO! Both trees successfully connected at iteration {i}."
                        )

                    self._finalize_path()

                    if self.plot:
                        print(
                            f"[RRT* Plan] Path extraction complete. Total waypoints: {len(self.path)}"
                        )
                        print("[RRT* Plan] Visualizing final successful path...")
                        self.plot_live_search()
                        time.sleep(0.5)

                    return self.path

        if self.plot:
            print(
                f"[RRT* Plan] SEARCH FAILED. Maximum iterations ({self.max_iter}) reached without connecting trees."
            )
            print("[RRT* Plan] Visualizing failed search state...")
            self.plot_live_search()
            time.sleep(0.5)

        return None

    def _direct_is_lawful(self):
        """Check that the straight line to the goal has no COLREGs penalty.

        Returns:
            bool: True if the straight line is lawful
        """
        v = self.validator
        if not (COLREGS_EDGE_PENALTY and v.planner and v.own_ship and v.sea_objects):
            return True
        try:
            penalty = v.planner.colregs_violation_penalty(
                v.own_ship,
                [self.start.x, self.start.y],
                [self.goal.x, self.goal.y],
                v.sea_objects,
            )
        except Exception:
            return True
        return penalty <= 0.0

    def _node_compliant(self, node):
        """Check that a node lies on the compliant side.

        Args:
            node (Node): Node to check

        Returns:
            bool: True if on the allowed side
        """
        if not self._compliant_active:
            return True
        rx, ry = self.compliant_ref
        h = self.compliant_heading
        signed = (node.x - rx) * (-np.sin(h)) + (node.y - ry) * np.cos(h)
        return self.compliant_side * signed <= self.compliant_tol

    def _sample(self):
        """Draw a random node, the goal one time in ten.

        Returns:
            Node: Sampled node
        """
        if random.random() < 0.1:
            return Node(self.goal.x, self.goal.y)
        for _ in range(15):
            cand = Node(
                random.uniform(-self.map_size[0], self.map_size[0]),
                random.uniform(-self.map_size[1], self.map_size[1]),
            )
            if self._node_compliant(cand):
                return cand
        return cand

    def _tree_xy(self, tree):
        """Get the node positions of a tree as an array, with a cache.

        Args:
            tree (list): Tree nodes

        Returns:
            ndarray: Positions, one row per node
        """
        buf, n, ref = self._xy.get(id(tree), (None, 0, None))
        m = len(tree)
        if ref is not tree or m < n:
            buf, n = None, 0
        if buf is None or m > len(buf):
            grown = np.empty((max(2 * m, _XY_MIN_CAPACITY), 2))
            if n:
                grown[:n] = buf[:n]
            buf = grown
        for k in range(n, m):
            buf[k, 0] = tree[k].x
            buf[k, 1] = tree[k].y
        self._xy[id(tree)] = (buf, m, tree)
        return buf[:m]

    def _get_nearest(self, sample, tree):
        """Find the tree node closest to a sample.

        Args:
            sample (Node): Sampled node
            tree (list): Tree nodes

        Returns:
            Node: Closest node
        """
        pts = self._tree_xy(tree)
        dist = np.sum((pts - [sample.x, sample.y]) ** 2, axis=1)
        return tree[np.argmin(dist)]

    def _steer(self, from_node, to_node):
        """Move from one node toward another by at most one step.

        Args:
            from_node (Node): Start node
            to_node (Node): Target node

        Returns:
            Node: New node
        """
        dist = np.hypot(to_node.x - from_node.x, to_node.y - from_node.y)
        step = min(self.step_size, dist)
        if dist < 1e-6:
            return from_node
        angle = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
        return Node(
            from_node.x + step * np.cos(angle), from_node.y + step * np.sin(angle)
        )

    def _choose_parent(self, new_node, neighbors, nearest):
        """Pick the neighbour that gives the new node the lowest cost.

        Args:
            new_node (Node): New node
            neighbors (list): Nearby nodes
            nearest (Node): Closest node
        """
        new_node.parent = nearest
        new_node.cost = nearest.cost + self.validator.get_cost(nearest, new_node)
        for n in neighbors:
            cost = n.cost + self.validator.get_cost(n, new_node)
            if cost < new_node.cost and self.validator.is_segment_valid(n, new_node):
                new_node.cost = cost
                new_node.parent = n

    def _is_ancestor(self, node, potential_ancestor):
        """Check whether a node is above another in the tree, or a loop was found.

        Args:
            node (Node): Start node
            potential_ancestor (Node): Node to look for

        Returns:
            bool: True if found or a loop is hit
        """
        curr = node
        visited = set()

        while curr is not None:
            if curr is potential_ancestor:
                return True

            if id(curr) in visited:
                return True

            visited.add(id(curr))
            curr = curr.parent

        return False

    def _rewire(self, new_node, neighbors, tree):
        """Reconnect neighbours through the new node when it is cheaper.

        Args:
            new_node (Node): New node
            neighbors (list): Nearby nodes
            tree (list): Tree nodes
        """
        for n in neighbors:
            if new_node.parent is n:
                continue
            if new_node is n:
                continue

            new_cost = new_node.cost + self.validator.get_cost(new_node, n)
            if new_cost < n.cost and self.validator.is_segment_valid(new_node, n):
                if self._is_ancestor(new_node, n):
                    continue

                n.parent = new_node
                n.cost = new_cost

    def _get_neighbors(self, node, tree):
        """List the tree nodes inside the rewire radius.

        Args:
            node (Node): Centre node
            tree (list): Tree nodes

        Returns:
            list: Nearby nodes
        """
        r = min(self._search_radius, 5.0 * np.sqrt(np.log(len(tree)) / len(tree)))
        r = max(r, self.step_size * 1.5)
        r_sq = r**2
        pts = self._tree_xy(tree)
        d2 = (pts[:, 0] - node.x) ** 2 + (pts[:, 1] - node.y) ** 2
        neighbors = []
        for k in np.flatnonzero(d2 < r_sq * (1.0 + _NEIGHBOUR_SLACK)):
            n = tree[k]
            if (n.x - node.x) ** 2 + (n.y - node.y) ** 2 < r_sq:
                neighbors.append(n)
        return neighbors

    def _attempt_connect(self, new_node, target_tree):
        """Try to link a new node to the other tree.

        Args:
            new_node (Node): New node
            target_tree (list): Other tree nodes

        Returns:
            bool: True if the trees are linked
        """
        nearest_target = self._get_nearest(new_node, target_tree)
        if (
            np.hypot(new_node.x - nearest_target.x, new_node.y - nearest_target.y)
            < self.step_size * 2
        ):
            if self.validator.is_segment_valid(new_node, nearest_target):
                self.connection_nodes = (new_node, nearest_target)
                if self.plot:
                    print(
                        f"[RRT* Core] Connection segment validated between nodes ({new_node.x:.1f}, {new_node.y:.1f}) and ({nearest_target.x:.1f}, {nearest_target.y:.1f})."
                    )
                return True
        return False

    def _finalize_path(self):
        """Join both trees into one path from start to goal.

        Returns:
            list: Path points
        """
        if self.plot:
            print("[RRT* Post] Tracing back node parents to assemble raw path...")

        node_a, node_b = self.connection_nodes

        if self._is_node_in_tree(node_a, self.start_tree):
            n_start, n_goal = node_a, node_b
        else:
            n_start, n_goal = node_b, node_a

        def trace_back(node, limit=5000):
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

        path_start = trace_back(n_start)
        path_start.reverse()
        path_goal = trace_back(n_goal)

        raw_path = path_start + path_goal

        if self.plot:
            print(
                f"[RRT* Post] Raw path assembled successfully. Length: {len(raw_path)} nodes."
            )

        if self.opti:
            if self.plot:
                print(
                    "[RRT* Post] Path optimization enabled. Executing Pruning -> Spline Smoothing -> Elastic Band..."
                )
            pruned = PathOptimizer.prune(raw_path, self.validator)
            smoothed = PathOptimizer.bspline_smooth(pruned)
            self.path = PathOptimizer.first_leg_or(
                PathOptimizer.elastic_band(smoothed, self.validator),
                pruned,
                self.own_ship,
            )
            if self.plot:
                print(
                    f"[RRT* Post] Optimization complete. Final smoothed path length: {len(self.path)} nodes."
                )
        else:
            if self.plot:
                print("[RRT* Post] Path optimization disabled. Returning raw path.")
            self.path = raw_path

    def _is_node_in_tree(self, node, tree):
        """Check whether a node is in a tree.

        Args:
            node (Node): Node
            tree (list): Tree nodes

        Returns:
            bool: True if found
        """
        return any(node is n for n in tree)

    def save_situation_as_obj(self, path=None, colregs_situation=None):
        """Store the search result in a dict.

        Args:
            path (list): Path to store, or None for self.path
            colregs_situation (str): COLREGs situation, or None

        Returns:
            dict: Map, trees, obstacles, path, start and goal
        """
        if self.plot:
            print(
                f"[RRT* State] Exporting search situation dictionary. Goal reached: {self.path is not None}"
            )

        return {
            "map_size": self.map_size,
            "goal_reached": self.path is not None,
            "node_list": self.start_tree + self.goal_tree,
            "obstacles": self.obstacles,
            "path": path if path else self.path,
            "path_original": self.path,
            "start": [self.start.x, self.start.y],
            "goal": [self.goal.x, self.goal.y],
            "colregs_situation": colregs_situation,
        }

    def add_situation_to_plot(
        self, situation, color=None, alpha=0.7, label=None, show_tree=True, color_code=0
    ):
        """Draw a stored search result on the current figure.

        Args:
            situation (dict): Stored search result
            color (str): Colour, or None to pick one
            alpha (float): Transparency
            label (str): Legend label
            show_tree (bool): Whether to draw the trees
            color_code (int): Index used to pick a colour
        """
        scale_factor = 1000.0
        ax = plt.gca()

        if color is None:
            colors = ["blue", "green", "orange", "purple", "brown", "gray"]
            color = colors[color_code % len(colors)]

        obstacles = situation.get("obstacles", [])
        path = situation.get("path", None)
        start_pos = situation.get("start", [0, 0])
        goal_pos = situation.get("goal", [0, 0])
        node_list = situation.get("node_list", [])

        for ox, oy, size in obstacles:
            circle = plt.Circle(
                (ox / scale_factor, oy / scale_factor),
                size / scale_factor,
                color=color,
                alpha=alpha * 0.4,
                clip_on=True,
            )
            ax.add_artist(circle)

        if show_tree and node_list:
            nodes_to_plot = (
                node_list
                if len(node_list) < 500
                else random.Random(0).sample(node_list, 500)
            )
            for node in nodes_to_plot:
                if node.parent:
                    plt.plot(
                        [node.x / scale_factor, node.parent.x / scale_factor],
                        [node.y / scale_factor, node.parent.y / scale_factor],
                        color=color,
                        alpha=alpha * 0.2,
                        linewidth=0.5,
                    )

        if path:
            path_arr = np.array(path) / scale_factor
            plt.plot(
                path_arr[:, 0],
                path_arr[:, 1],
                color=color,
                alpha=alpha,
                linewidth=2,
                label=label,
            )
            plt.scatter(path_arr[:, 0], path_arr[:, 1], color=color, s=15, alpha=alpha)

        plt.scatter(
            start_pos[0] / scale_factor,
            start_pos[1] / scale_factor,
            marker="o",
            color=color,
            s=50,
            edgecolors="k",
            zorder=10,
        )
        plt.scatter(
            goal_pos[0] / scale_factor,
            goal_pos[1] / scale_factor,
            marker="s",
            color=color,
            s=50,
            edgecolors="k",
            zorder=10,
        )

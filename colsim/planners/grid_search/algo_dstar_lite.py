"""
D* Lite grid planning
author: vss2sn (28676655+vss2sn@users.noreply.github.com)
Link to papers:
D* Lite (Link: http://idm-lab.org/bib/abstracts/papers/aaai02b.pd)
Improved Fast Replanning for Robot Navigation in Unknown Terrain
(Link: http://www.cs.cmu.edu/~maxim/files/dlite_icra02.pdf)
"""

import math
import matplotlib.pyplot as plt
import random
import numpy as np
from typing import Tuple

show_animation = False
pause_time = 0.001
p_create_random_obstacle = 0


class Node:
    def __init__(self, x: int = 0, y: int = 0, cost: float = 0.0):
        """Create a grid node with a cost.

        Args:
            x (int): Grid x index
            y (int): Grid y index
            cost (float): Move cost
        """
        self.x = x
        self.y = y
        self.cost = cost


def add_coordinates(node1: Node, node2: Node):
    """Add two nodes position and cost.

    Args:
        node1 (Node): First node
        node2 (Node): Second node

    Returns:
        Node: Sum node
    """
    new_node = Node()
    new_node.x = node1.x + node2.x
    new_node.y = node1.y + node2.y
    new_node.cost = node1.cost + node2.cost
    return new_node


def compare_coordinates(node1: Node, node2: Node):
    """Check that two nodes have the same position.

    Args:
        node1 (Node): First node
        node2 (Node): Second node

    Returns:
        bool: True if equal
    """
    return node1.x == node2.x and node1.y == node2.y


class DStarLite:

    motions = [
        Node(1, 0, 1),
        Node(0, 1, 1),
        Node(-1, 0, 1),
        Node(0, -1, 1),
        Node(1, 1, math.sqrt(2)),
        Node(1, -1, math.sqrt(2)),
        Node(-1, 1, math.sqrt(2)),
        Node(-1, -1, math.sqrt(2)),
    ]

    def __init__(self, ox: list, oy: list):
        """Build the grid from the obstacle points.

        Args:
            ox (list): Obstacle x values
            oy (list): Obstacle y values
        """
        self.x_min_world = int(min(ox))
        self.y_min_world = int(min(oy))
        self.x_max = int(abs(max(ox) - self.x_min_world))
        self.y_max = int(abs(max(oy) - self.y_min_world))
        self.obstacles = [
            Node(x - self.x_min_world, y - self.y_min_world) for x, y in zip(ox, oy)
        ]
        self.obstacles_xy = np.array(
            [[obstacle.x, obstacle.y] for obstacle in self.obstacles]
        )
        self.start = Node(0, 0)
        self.goal = Node(0, 0)
        self.U = list()
        self.km = 0.0
        self.kold = 0.0
        self.rhs = self.create_grid(float("inf"))
        self.g = self.create_grid(float("inf"))
        self.detected_obstacles_xy = np.empty((0, 2))
        self.xy = np.empty((0, 2))
        if show_animation:
            self.detected_obstacles_for_plotting_x = list()
            self.detected_obstacles_for_plotting_y = list()
        self.initialized = False

    def create_grid(self, val: float):
        """Create a grid filled with one value.

        Args:
            val (float): Fill value

        Returns:
            ndarray: Grid
        """
        return np.full((self.x_max, self.y_max), val)

    def is_obstacle(self, node: Node):
        """Check whether a node is a known or detected obstacle.

        Args:
            node (Node): Node to check

        Returns:
            bool: True if blocked
        """
        x = np.array([node.x])
        y = np.array([node.y])
        obstacle_x_equal = self.obstacles_xy[:, 0] == x
        obstacle_y_equal = self.obstacles_xy[:, 1] == y
        is_in_obstacles = (obstacle_x_equal & obstacle_y_equal).any()

        is_in_detected_obstacles = False
        if self.detected_obstacles_xy.shape[0] > 0:
            is_x_equal = self.detected_obstacles_xy[:, 0] == x
            is_y_equal = self.detected_obstacles_xy[:, 1] == y
            is_in_detected_obstacles = (is_x_equal & is_y_equal).any()

        return is_in_obstacles or is_in_detected_obstacles

    def c(self, node1: Node, node2: Node):
        """Give the cost to move between two neighbour nodes.

        Args:
            node1 (Node): Start node
            node2 (Node): End node

        Returns:
            float: Move cost, inf if blocked
        """
        if self.is_obstacle(node2):
            # Attempting to move from or to an obstacle
            return math.inf
        new_node = Node(node1.x - node2.x, node1.y - node2.y)
        detected_motion = list(
            filter(lambda motion: compare_coordinates(motion, new_node), self.motions)
        )
        return detected_motion[0].cost

    def h(self, s: Node):
        """Give the heuristic, here a constant.

        Args:
            s (Node): Node

        Returns:
            float: Heuristic value
        """
        return 1

    def calculate_key(self, s: Node):
        """Compute the priority key of a node.

        Args:
            s (Node): Node

        Returns:
            tuple: Key pair
        """
        return (
            min(self.g[s.x][s.y], self.rhs[s.x][s.y]) + self.h(s) + self.km,
            min(self.g[s.x][s.y], self.rhs[s.x][s.y]),
        )

    def is_valid(self, node: Node):
        """Check that a node is inside the grid.

        Args:
            node (Node): Node

        Returns:
            bool: True if inside
        """
        if 0 <= node.x < self.x_max and 0 <= node.y < self.y_max:
            return True
        return False

    def get_neighbours(self, u: Node):
        """List the neighbour nodes inside the grid.

        Args:
            u (Node): Node

        Returns:
            list: Neighbour nodes
        """
        return [
            add_coordinates(u, motion)
            for motion in self.motions
            if self.is_valid(add_coordinates(u, motion))
        ]

    def pred(self, u: Node):
        """List the predecessors of a node.

        Args:
            u (Node): Node

        Returns:
            list: Predecessor nodes
        """
        return self.get_neighbours(u)

    def succ(self, u: Node):
        """List the successors of a node.

        Args:
            u (Node): Node

        Returns:
            list: Successor nodes
        """
        return self.get_neighbours(u)

    def initialize(self, start: Node, goal: Node):
        """Set the start and goal and reset the search on first call.

        Args:
            start (Node): Start node
            goal (Node): Goal node
        """

        self.start.x = start.x - self.x_min_world
        self.start.y = start.y - self.y_min_world
        self.goal.x = goal.x - self.x_min_world
        self.goal.y = goal.y - self.y_min_world

        if not self.initialized:
            self.initialized = True
            self.U = list()
            self.km = 0.0
            self.rhs = self.create_grid(math.inf)
            self.g = self.create_grid(math.inf)
            self.rhs[self.goal.x][self.goal.y] = 0
            self.U.append((self.goal, self.calculate_key(self.goal)))
            self.detected_obstacles_xy = np.empty((0, 2))

    def update_vertex(self, u: Node):
        """Update the cost of a node and its place in the queue.

        Args:
            u (Node): Node
        """
        if not compare_coordinates(u, self.goal):
            self.rhs[u.x][u.y] = min(
                [
                    self.c(u, sprime) + self.g[sprime.x][sprime.y]
                    for sprime in self.succ(u)
                ]
            )
        if any([compare_coordinates(u, node) for node, key in self.U]):
            self.U = [
                (node, key) for node, key in self.U if not compare_coordinates(node, u)
            ]
            self.U.sort(key=lambda x: x[1])
        if self.g[u.x][u.y] != self.rhs[u.x][u.y]:
            self.U.append((u, self.calculate_key(u)))
            self.U.sort(key=lambda x: x[1])

    def compare_keys(
        self, key_pair1: Tuple[float, float], key_pair2: Tuple[float, float]
    ):
        """Compare two priority keys.

        Args:
            key_pair1 (Tuple[float, float]): First key
            key_pair2 (Tuple[float, float]): Second key

        Returns:
            bool: True if the first key is smaller
        """
        return key_pair1[0] < key_pair2[0] or (
            key_pair1[0] == key_pair2[0] and key_pair1[1] < key_pair2[1]
        )

    def compute_shortest_path(self):
        """Expand nodes until the start cost is settled."""
        self.U.sort(key=lambda x: x[1])
        has_elements = len(self.U) > 0
        start_key_not_updated = self.compare_keys(
            self.U[0][1], self.calculate_key(self.start)
        )
        rhs_not_equal_to_g = (
            self.rhs[self.start.x][self.start.y] != self.g[self.start.x][self.start.y]
        )
        while has_elements and start_key_not_updated or rhs_not_equal_to_g:
            self.kold = self.U[0][1]
            u = self.U[0][0]
            self.U.pop(0)
            if self.compare_keys(self.kold, self.calculate_key(u)):
                self.U.append((u, self.calculate_key(u)))
                self.U.sort(key=lambda x: x[1])
            elif (self.g[u.x, u.y] > self.rhs[u.x, u.y]).any():
                self.g[u.x, u.y] = self.rhs[u.x, u.y]
                for s in self.pred(u):
                    self.update_vertex(s)
            else:
                self.g[u.x, u.y] = math.inf
                for s in self.pred(u) + [u]:
                    self.update_vertex(s)
            self.U.sort(key=lambda x: x[1])
            start_key_not_updated = self.compare_keys(
                self.U[0][1], self.calculate_key(self.start)
            )
            rhs_not_equal_to_g = (
                self.rhs[self.start.x][self.start.y]
                != self.g[self.start.x][self.start.y]
            )

    def detect_changes(self):
        """Add the new obstacles seen at this step.

        Returns:
            list: Nodes that changed
        """
        changed_vertices = list()
        if len(self.spoofed_obstacles) > 0:
            for spoofed_obstacle in self.spoofed_obstacles[0]:
                if compare_coordinates(
                    spoofed_obstacle, self.start
                ) or compare_coordinates(spoofed_obstacle, self.goal):
                    continue
                changed_vertices.append(spoofed_obstacle)
                self.detected_obstacles_xy = np.concatenate(
                    (
                        self.detected_obstacles_xy,
                        [[spoofed_obstacle.x, spoofed_obstacle.y]],
                    )
                )
                if show_animation:
                    self.detected_obstacles_for_plotting_x.append(
                        spoofed_obstacle.x + self.x_min_world
                    )
                    self.detected_obstacles_for_plotting_y.append(
                        spoofed_obstacle.y + self.y_min_world
                    )
                    plt.plot(
                        self.detected_obstacles_for_plotting_x,
                        self.detected_obstacles_for_plotting_y,
                        ".k",
                    )
                    plt.pause(pause_time)
            self.spoofed_obstacles.pop(0)

        random.seed()
        if random.random() > 1 - p_create_random_obstacle:
            x = random.randint(0, self.x_max - 1)
            y = random.randint(0, self.y_max - 1)
            new_obs = Node(x, y)
            if compare_coordinates(new_obs, self.start) or compare_coordinates(
                new_obs, self.goal
            ):
                return changed_vertices
            changed_vertices.append(Node(x, y))
            self.detected_obstacles_xy = np.concatenate(
                (self.detected_obstacles_xy, [[x, y]])
            )
            if show_animation:
                self.detected_obstacles_for_plotting_x.append(x + self.x_min_world)
                self.detected_obstacles_for_plotting_y.append(y + self.y_min_world)
                plt.plot(
                    self.detected_obstacles_for_plotting_x,
                    self.detected_obstacles_for_plotting_y,
                    ".k",
                )
                plt.pause(pause_time)
        return changed_vertices

    def compute_current_path(self):
        """Follow the lowest costs from start to goal.

        Returns:
            list: Path nodes
        """
        path = list()
        current_point = Node(self.start.x, self.start.y)
        while not compare_coordinates(current_point, self.goal):
            path.append(current_point)
            current_point = min(
                self.succ(current_point),
                key=lambda sprime: self.c(current_point, sprime)
                + self.g[sprime.x][sprime.y],
            )
        path.append(self.goal)
        return path

    def compare_paths(self, path1: list, path2: list):
        """Check that two paths are the same.

        Args:
            path1 (list): First path
            path2 (list): Second path

        Returns:
            bool: True if equal
        """
        if len(path1) != len(path2):
            return False
        for node1, node2 in zip(path1, path2):
            if not compare_coordinates(node1, node2):
                return False
        return True

    def display_path(self, path: list, colour: str, alpha: float = 1.0):
        """Plot a path on the current figure.

        Args:
            path (list): Path nodes
            colour (str): Plot style
            alpha (float): Transparency

        Returns:
            list: Plotted lines
        """
        px = [(node.x + self.x_min_world) for node in path]
        py = [(node.y + self.y_min_world) for node in path]
        drawing = plt.plot(px, py, colour, alpha=alpha)
        plt.pause(pause_time)
        return drawing

    def dstarl(
        self, start: Node, goal: Node, spoofed_ox: list = [], spoofed_oy: list = []
    ):
        """Move from start to goal and replan when obstacles appear.

        Args:
            start (Node): Start node
            goal (Node): Goal node
            spoofed_ox (list): x values of obstacles that appear, per step
            spoofed_oy (list): y values of obstacles that appear, per step

        Returns:
            list: Visited nodes, or a failure tuple if no path exists
        """
        self.spoofed_obstacles = [
            [
                Node(x - self.x_min_world, y - self.y_min_world)
                for x, y in zip(rowx, rowy)
            ]
            for rowx, rowy in zip(spoofed_ox, spoofed_oy)
        ]
        pathx = []
        pathy = []
        path = []

        self.initialize(start, goal)
        last = self.start
        self.compute_shortest_path()
        pathx.append(self.start.x + self.x_min_world)
        pathy.append(self.start.y + self.y_min_world)
        path.append((pathx[-1], pathy[-1]))

        if show_animation:
            current_path = self.compute_current_path()
            previous_path = current_path.copy()
            previous_path_image = self.display_path(previous_path, ".c", alpha=0.3)
            current_path_image = self.display_path(current_path, ".c")

        outer_iterations = 0
        while not compare_coordinates(self.goal, self.start):
            outer_iterations += 1
            if self.g[self.start.x][self.start.y] == math.inf:
                print("No path possible")
                return False, pathx, pathy
            self.start = min(
                self.succ(self.start),
                key=lambda sprime: self.c(self.start, sprime)
                + self.g[sprime.x][sprime.y],
            )
            pathx.append(self.start.x + self.x_min_world)
            pathy.append(self.start.y + self.y_min_world)
            path.append((pathx[-1], pathy[-1]))

            if show_animation:
                current_path.pop(0)
                plt.plot(pathx, pathy, "-r")
                plt.pause(pause_time)
            changed_vertices = self.detect_changes()
            if len(changed_vertices) != 0:
                print("New obstacle detected")
                self.km += self.h(last)
                last = self.start
                for u in changed_vertices:
                    if compare_coordinates(u, self.start):
                        continue
                    self.rhs[u.x][u.y] = math.inf
                    self.g[u.x][u.y] = math.inf
                    self.update_vertex(u)
                self.compute_shortest_path()

                if show_animation:
                    new_path = self.compute_current_path()
                    if not self.compare_paths(current_path, new_path):
                        current_path_image[0].remove()
                        previous_path_image[0].remove()
                        previous_path = current_path.copy()
                        current_path = new_path.copy()
                        previous_path_image = self.display_path(
                            previous_path, ".c", alpha=0.3
                        )
                        current_path_image = self.display_path(current_path, ".c")
                        plt.pause(pause_time)
        return path


def main():
    """Run the D* Lite demo on a walled map."""

    sx = 10  # [m]
    sy = 10  # [m]
    gx = 50  # [m]
    gy = 50  # [m]

    # set obstacle positions
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)

    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")
        label_column = [
            "Start",
            "Goal",
            "Path taken",
            "Current computed path",
            "Previous computed path",
            "Obstacles",
        ]
        columns = [
            plt.plot([], [], symbol, color=colour, alpha=alpha)[0]
            for symbol, colour, alpha in [
                ["o", "g", 1],
                ["x", "b", 1],
                ["-", "r", 1],
                [".", "c", 1],
                [".", "c", 0.3],
                [".", "k", 1],
            ]
        ]
        plt.legend(
            columns,
            label_column,
            bbox_to_anchor=(1, 1),
            title="Key:",
            fontsize="xx-small",
        )
        plt.plot()
        plt.pause(pause_time)

    spoofed_ox = [[0, 9, 4], [0, 7], [], [], [], [], [], [5]]
    spoofed_oy = [[2, 2, 0], [1, 7], [], [], [], [], [], [4]]

    dstarlite = DStarLite(ox, oy)
    path = dstarlite.dstarl(Node(x=sx, y=sy), Node(x=gx, y=gy))
    print("the final path is ", path)


if __name__ == "__main__":
    main()

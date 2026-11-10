import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
from heapq import heappush, heappop
import math


class Node:
    def __init__(self, x, y, g=float("inf"), h=0, parent=None):
        """Create a grid node with its costs.

        Args:
            x (int): Grid x index
            y (int): Grid y index
            g (float): Cost from the start
            h (float): Estimated cost to the goal
            parent (Node): Previous node, or None
        """
        self.x = x
        self.y = y
        self.g = g  # Cost from start
        self.h = h  # Heuristic cost to goal
        self.f = g + h  # Total cost
        self.parent = parent

    def __repr__(self):
        """Describe the node position and total cost.

        Returns:
            str: Text description
        """
        return f"Node(x={self.x}, y={self.y}, f={self.f:.2f})"

    def __lt__(self, other):
        """Compare two nodes by total cost for the heap.

        Args:
            other (Node): Node to compare with

        Returns:
            bool: True if this node costs less
        """
        return self.f < other.f


class AStar:
    def __init__(
        self,
        start,
        goal,
        num_obstacles,
        map_size,
        grid_resolution=0.5,
        obstacles=None,
        safety_margin=0.1,
    ):
        """Build the grid, the obstacles and the start and goal cells.

        Args:
            start (tuple): Start point (x, y)
            goal (tuple): Goal point (x, y)
            num_obstacles (int): Number of random obstacles if none are given
            map_size (tuple): Half size of the map along x and y
            grid_resolution (float): Cell side
            obstacles (list): Obstacle discs (x, y, radius), or None
            safety_margin (float): Extra clearance around obstacles
        """
        self.start_pos = start
        self.goal_pos = goal
        self.map_size = map_size
        self.grid_resolution = grid_resolution
        self.safety_margin = safety_margin

        # Calculate grid dimensions
        self.grid_width = int((2 * map_size[0]) / grid_resolution) + 1
        self.grid_height = int((2 * map_size[1]) / grid_resolution) + 1

        self.start = self.world_to_grid(start[0], start[1])
        self.goal = self.world_to_grid(goal[0], goal[1])

        self.obstacles = []
        self.obstacles = (
            obstacles if obstacles else self.generate_random_obstacles(num_obstacles)
        )

        self.obstacle_grid = self.create_obstacle_grid()

        self.path = None
        self.path_original = None
        self.goal_reached = False
        self.iterations_used = 0

    def world_to_grid(self, x, y):
        """Convert a map point to grid indexes.

        Args:
            x (float): Point x
            y (float): Point y

        Returns:
            tuple: Grid indexes
        """
        grid_x = int((x + self.map_size[0]) / self.grid_resolution)
        grid_y = int((y + self.map_size[1]) / self.grid_resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid indexes to a map point.

        Args:
            grid_x (int): Grid x index
            grid_y (int): Grid y index

        Returns:
            tuple: Point (x, y)
        """
        x = grid_x * self.grid_resolution - self.map_size[0]
        y = grid_y * self.grid_resolution - self.map_size[1]
        return (x, y)

    def generate_random_obstacles(self, num_obstacles):
        """Place random obstacles away from the start, goal and edges.

        Args:
            num_obstacles (int): Number of obstacles wanted

        Returns:
            list: Obstacle discs (x, y, radius)
        """
        obstacles = []
        attempts = 0
        max_attempts = num_obstacles * 10

        while len(obstacles) < num_obstacles and attempts < max_attempts:
            attempts += 1
            ox = random.uniform(-self.map_size[0] + 2, self.map_size[0] - 2)
            oy = random.uniform(-self.map_size[1] + 2, self.map_size[1] - 2)
            size = 4

            if self.check_obstacle_position(ox, oy, size):
                obstacles.append((ox, oy, size))

        return obstacles

    def check_obstacle_position(self, x, y, size):
        """Check that an obstacle stays in the map and away from start and goal.

        Args:
            x (float): Obstacle x
            y (float): Obstacle y
            size (float): Obstacle radius

        Returns:
            bool: True if the spot is allowed
        """
        # Check map boundaries
        if (
            x - size < -self.map_size[0]
            or x + size > self.map_size[0]
            or y - size < -self.map_size[1]
            or y + size > self.map_size[1]
        ):
            return False

        # Check goal region
        goal_world = self.grid_to_world(self.goal[0], self.goal[1])
        if np.linalg.norm([x - goal_world[0], y - goal_world[1]]) < size + 1:
            return False

        # Check start region
        start_world = self.grid_to_world(self.start[0], self.start[1])
        if np.linalg.norm([x - start_world[0], y - start_world[1]]) < size + 1:
            return False

        # Check other obstacles
        for o in self.obstacles:
            if np.linalg.norm([x - o[0], y - o[1]]) < size + o[2]:
                return False

        return True

    def create_obstacle_grid(self):
        """Mark every cell covered by an obstacle and its margin.

        Returns:
            ndarray: Boolean grid, True where blocked
        """
        grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)

        for ox, oy, size in self.obstacles:
            # Convert obstacle center to grid coordinates
            center_gx, center_gy = self.world_to_grid(ox, oy)

            # Calculate obstacle radius in grid cells
            radius_cells = int((size + self.safety_margin) / self.grid_resolution) + 1

            # Mark all cells within obstacle radius
            for dy in range(-radius_cells, radius_cells + 1):
                for dx in range(-radius_cells, radius_cells + 1):
                    gx = center_gx + dx
                    gy = center_gy + dy

                    if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                        # Check if cell is within obstacle radius
                        cell_world = self.grid_to_world(gx, gy)
                        dist = np.linalg.norm([cell_world[0] - ox, cell_world[1] - oy])
                        if dist <= size + self.safety_margin:
                            grid[gy, gx] = True

        return grid

    def plan(self):
        """Search the shortest path with A* and store it in self.path."""

        open_list = []
        closed_set = set()
        g_costs = {}

        start_node = Node(
            self.start[0], self.start[1], g=0, h=self.heuristic(self.start, self.goal)
        )
        heappush(open_list, start_node)
        g_costs[self.start] = 0

        iteration_count = 0

        while open_list and iteration_count < 10000:
            iteration_count += 1

            # Get node with lowest f-cost
            current = heappop(open_list)
            current_pos = (current.x, current.y)

            # Check if goal reached
            if current_pos == self.goal:
                self.path_original = self.reconstruct_path(current)
                self.path = self.optimize_path(self.path_original)
                self.goal_reached = True
                self.iterations_used = iteration_count
                return

            closed_set.add(current_pos)

            # Explore neighbors
            for neighbor_pos in self.get_neighbors(current_pos):
                if neighbor_pos in closed_set:
                    continue

                if self.is_collision(neighbor_pos):
                    continue

                # Calculate costs
                tentative_g = current.g + self.distance(current_pos, neighbor_pos)

                if neighbor_pos not in g_costs or tentative_g < g_costs[neighbor_pos]:
                    g_costs[neighbor_pos] = tentative_g
                    h_cost = self.heuristic(neighbor_pos, self.goal)
                    neighbor_node = Node(
                        neighbor_pos[0],
                        neighbor_pos[1],
                        g=tentative_g,
                        h=h_cost,
                        parent=current,
                    )
                    heappush(open_list, neighbor_node)

        # No path found
        self.iterations_used = iteration_count
        self.path = None

    def heuristic(self, pos1, pos2):
        """Estimate the distance between two cells.

        Args:
            pos1 (tuple): First cell
            pos2 (tuple): Second cell

        Returns:
            float: Straight line distance
        """
        # Euclidean distance heuristic
        dx = abs(pos1[0] - pos2[0]) * self.grid_resolution
        dy = abs(pos1[1] - pos2[1]) * self.grid_resolution
        return math.sqrt(dx * dx + dy * dy)

    def distance(self, pos1, pos2):
        """Compute the distance between two neighbour cells.

        Args:
            pos1 (tuple): First cell
            pos2 (tuple): Second cell

        Returns:
            float: Distance
        """
        # Actual distance between adjacent cells
        dx = abs(pos1[0] - pos2[0]) * self.grid_resolution
        dy = abs(pos1[1] - pos2[1]) * self.grid_resolution
        return math.sqrt(dx * dx + dy * dy)

    def get_neighbors(self, pos):
        """List the 8 neighbour cells inside the grid.

        Args:
            pos (tuple): Current cell

        Returns:
            list: Neighbour cells
        """
        neighbors = []
        x, y = pos

        # 8-connected grid (including diagonal moves)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                new_x, new_y = x + dx, y + dy

                if 0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height:
                    neighbors.append((new_x, new_y))

        return neighbors

    def is_collision(self, pos):
        """Check whether a cell is blocked.

        Args:
            pos (tuple): Cell

        Returns:
            bool: True if blocked
        """
        x, y = pos
        return self.obstacle_grid[y, x]

    def reconstruct_path(self, goal_node):
        """Follow the parents back from the goal to build the path.

        Args:
            goal_node (Node): Goal node

        Returns:
            list: Path points from start to goal
        """
        path = []
        current = goal_node

        while current is not None:
            world_pos = self.grid_to_world(current.x, current.y)
            path.append([world_pos[0], world_pos[1]])
            current = current.parent

        return path[::-1]  # Reverse to get start-to-goal path

    def optimize_path(self, path):
        """Remove the points that can be skipped in a straight line.

        Args:
            path (list): Path points

        Returns:
            list: Shorter path
        """
        if not path or len(path) < 3:
            return path

        optimized_path = [path[0]]
        current_idx = 0

        while current_idx < len(path) - 1:
            # Find furthest visible point
            furthest_idx = current_idx + 1

            for test_idx in range(current_idx + 2, len(path)):
                if self.line_of_sight(path[current_idx], path[test_idx]):
                    furthest_idx = test_idx
                else:
                    break

            optimized_path.append(path[furthest_idx])
            current_idx = furthest_idx

        return optimized_path

    def line_of_sight(self, pos1, pos2):
        """Check that the straight line between two points is free.

        Args:
            pos1 (tuple): First point
            pos2 (tuple): Second point

        Returns:
            bool: True if no cell on the line is blocked
        """
        # Check if direct line between two points is obstacle-free
        x1, y1 = pos1
        x2, y2 = pos2

        # Convert to grid coordinates
        gx1, gy1 = self.world_to_grid(x1, y1)
        gx2, gy2 = self.world_to_grid(x2, y2)

        # Bresenham's line algorithm for grid traversal
        dx = abs(gx2 - gx1)
        dy = abs(gy2 - gy1)
        x, y = gx1, gy1

        x_inc = 1 if gx1 < gx2 else -1
        y_inc = 1 if gy1 < gy2 else -1
        error = dx - dy

        dx *= 2
        dy *= 2

        while True:
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                if self.obstacle_grid[y, x]:
                    return False

            if x == gx2 and y == gy2:
                break

            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

        return True

    def plot(self, path=None, execution_time=None, iterations=None):
        """Plot the obstacles, the path and the run statistics.

        Args:
            path (list): Path points, or None
            execution_time (float): Run time, s
            iterations (int): Number of iterations
        """
        plt.figure(figsize=(12, 10))
        plt.grid(True, alpha=0.3)
        plt.xlim(-self.map_size[0], self.map_size[0])
        plt.ylim(-self.map_size[1], self.map_size[1])
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title(
            f"A* Path Planning\nExecution Time: {execution_time:.3f} seconds\nIterations: {iterations}"
        )

        # Plot obstacles
        for ox, oy, size in self.obstacles:
            circle = plt.Circle((ox, oy), size, color="gray", alpha=0.7)
            plt.gca().add_artist(circle)

        # Plot grid (optional, for small grids)
        if self.grid_width < 100:
            for i in range(self.grid_width):
                for j in range(self.grid_height):
                    if self.obstacle_grid[j, i]:
                        world_pos = self.grid_to_world(i, j)
                        plt.plot(
                            world_pos[0],
                            world_pos[1],
                            "s",
                            color="red",
                            markersize=1,
                            alpha=0.3,
                        )

        # Plot paths
        if path:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], "-b", linewidth=2, label="Optimized Path")

        if hasattr(self, "path_original") and self.path_original:
            path_orig = np.array(self.path_original)
            plt.plot(
                path_orig[:, 0],
                path_orig[:, 1],
                "--r",
                alpha=0.5,
                label="Original Path",
            )

        # Plot start and goal
        start_world = self.grid_to_world(self.start[0], self.start[1])
        goal_world = self.grid_to_world(self.goal[0], self.goal[1])
        plt.plot(start_world[0], start_world[1], "go", markersize=10, label="Start")
        plt.plot(goal_world[0], goal_world[1], "ro", markersize=10, label="Goal")

        plt.legend()
        plt.tight_layout()

        # Save plot
        date_time = time.strftime("%Y%m%d-%H%M%S")
        if not os.path.exists("plots"):
            os.makedirs("plots")
        plt.savefig(f"plots/astar_{date_time}.png", dpi=300)
        # plt.show()
        plt.close()


def run_astar(start, goal, num_obstacles, map_size, astar_parameters):
    """Run A* once, plot the result and time it.

    Args:
        start (tuple): Start point
        goal (tuple): Goal point
        num_obstacles (int): Number of random obstacles
        map_size (tuple): Half size of the map
        astar_parameters (dict): Grid resolution and safety margin

    Returns:
        tuple: Run time, path and number of iterations
    """
    t0 = time.time()
    astar = AStar(
        start,
        goal,
        num_obstacles,
        map_size,
        grid_resolution=astar_parameters["grid_resolution"],
        safety_margin=astar_parameters["safety_margin"],
    )
    astar.plan()
    t1 = time.time()
    execution_time = t1 - t0

    astar.plot(astar.path, execution_time, astar.iterations_used)
    return execution_time, astar.path, astar.iterations_used


def create_graph_iteration_time(tot_time_list, tot_iter_list):
    """Plot the run time and iterations against the number of obstacles.

    Args:
        tot_time_list (list): Run times, s
        tot_iter_list (list): Iteration counts
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "tab:blue"
    ax1.set_xlabel("Number of Obstacles")
    ax1.set_ylabel("Execution Time (s)", color=color)
    ax1.plot(
        range(len(tot_time_list)), tot_time_list, color=color, marker="o", label="Time"
    )
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Number of Iterations", color=color)
    ax2.plot(
        range(len(tot_iter_list)),
        tot_iter_list,
        color=color,
        marker="s",
        label="Iterations",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    for i, val in enumerate(tot_time_list):
        ax1.annotate(
            f"{val:.3f}",
            (i, val),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="tab:blue",
        )

    for i, val in enumerate(tot_iter_list):
        ax2.annotate(
            f"{int(val)}",
            (i, val),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            color="tab:red",
        )

    plt.title("A* Performance vs Number of Obstacles")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)

    fig.tight_layout()
    plt.savefig("astar_time_iter.png", dpi=300)


if __name__ == "__main__":
    from tqdm import tqdm

    start = [-20, -20]
    goal = [20, 20]
    map_size = [25, 25]

    astar_parameters = {
        "grid_resolution": 0.5,  # Smaller values = higher resolution
        "safety_margin": 1.0,
    }

    tot_time_list = []
    tot_iter_list = []

    for i in range(12, 20):
        print(f"\nRunning A* with {i} obstacles")
        time_list = []
        iter_list = []

        for j in tqdm(range(1), desc=f"Obstacle count {i}", unit="run"):
            num_obstacles = i
            execution_time, path, iterations = run_astar(
                start, goal, num_obstacles, map_size, astar_parameters
            )
            time_list.append(execution_time)
            iter_list.append(iterations)

            with open("astar_times.txt", "a") as f:
                f.write(f"{i},{execution_time},{iterations}\n")

        tot_time_list.append(np.mean(time_list))
        tot_iter_list.append(np.mean(iter_list))

    print("Time list:", tot_time_list)
    print("Iter list:", tot_iter_list)
    create_graph_iteration_time(tot_time_list, tot_iter_list)

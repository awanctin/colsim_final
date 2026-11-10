#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find here : https://github.com/Haghrah/ACO---Robot-Path-Planning/tree/master
Created on Thu Nov 16 12:56:52 2017

@author: arslan
"""

import matplotlib.pyplot as plt
import numpy as np


class ACO_PDG:

    def __init__(
        self,
        width,
        height,
        obstacles,
        ant_count,
        step_count,
        alpha,
        beta,
        gamma,
        evaporation_rate,
        start,
        end,
        brushfire_iter,
        colony_iter,
    ):
        """Set up the grid, the colony and the pheromone map.

        Args:
            width (int): Grid width, cells
            height (int): Grid height, cells
            obstacles (list): Blocked cells (i, j)
            ant_count (int): Number of ants per iteration
            step_count (int): Max steps per ant
            alpha (float): Weight of the pheromone
            beta (float): Weight of the heuristic
            gamma (float): Pheromone diffusion gain
            evaporation_rate (float): Share of pheromone lost per iteration
            start (tuple): Start cell
            end (tuple): Goal cell
            brushfire_iter (int): Iterations for the obstacle distance map
            colony_iter (int): Number of colony iterations
        """
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.ant_count = ant_count
        self.step_count = step_count
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.evaporation_rate = evaporation_rate
        self.start = start
        self.end = end
        self.colony_iter = colony_iter
        self.brushfire_iter = brushfire_iter
        self.pheromone_grid = np.ones(shape=(self.width, self.height))
        self.heuristic_grid = self.Potential_Field_Heuristic(brushfire_iter)
        self.ants_movements = []

    def U_att(self, p, p_e):
        """Compute the attractive potential toward the goal.

        Args:
            p (ndarray): Current point
            p_e (ndarray): Goal point

        Returns:
            float: Attractive potential
        """
        K_a = 1.0
        return K_a * np.linalg.norm(p - p_e) ** 2 / 2.0

    def U_rep(self, rho, rho_o):
        """Compute the repulsive potential of an obstacle.

        Args:
            rho (float): Distance to the obstacle
            rho_o (float): Range of influence

        Returns:
            float: Repulsive potential, 0 outside the range
        """
        K_r = 25.0
        if rho <= rho_o:
            return K_r * (1 / rho - 1 / rho_o) ** 2 / 2.0
        else:
            return 0

    def U_tot(self, p, p_e, rho, rho_o):
        """Sum the attractive and repulsive potentials.

        Args:
            p (ndarray): Current point
            p_e (ndarray): Goal point
            rho (float): Distance to the obstacle
            rho_o (float): Range of influence

        Returns:
            float: Total potential
        """
        return self.U_att(p, p_e) + self.U_rep(rho, rho_o)

    def pheromone_diffusion(self, phi, r, xi):
        """Compute the pheromone spread to a nearby cell.

        Args:
            phi (float): Pheromone at the source cell
            r (float): Diffusion radius
            xi (float): Distance to the source cell

        Returns:
            float: Pheromone added
        """
        return self.gamma * phi * (r - xi) / r

    def Brushfire_Algorithm(self, max_iter):
        """Find the shortest distance from an obstacle for each grid using Brushfire algorithm."""
        sdo = float("inf") * np.ones(shape=(self.width, self.height))
        grid_traverse_mask = np.zeros(shape=(self.width, self.height))
        grid_list = []
        for obstacle in self.obstacles:
            sdo[obstacle[0], obstacle[1]] = 0
            grid_list.append(obstacle)
        for iteration in range(max_iter):
            grid_list_tmp = []
            for grid in grid_list:
                for i in range(max(grid[0] - 1, 0), min(grid[0] + 2, self.width)):
                    for j in range(max(grid[1] - 1, 0), min(grid[1] + 2, self.height)):
                        if grid_traverse_mask[i, j] == 0:
                            sdo[i, j] = min(
                                sdo[i, j],
                                sdo[grid[0], grid[1]]
                                + np.linalg.norm([grid[0] - i, grid[1] - j]),
                            )
                            grid_list_tmp.append((i, j))
            for grid in grid_list:
                grid_traverse_mask[grid[0], grid[1]] = 1
            grid_list = grid_list_tmp
        return sdo

    def Potential_Field_Heuristic(self, brushfire_iter):
        """Heuristic value over obstacles are not valid and should not be used."""
        heuristic_grid = np.zeros(shape=(self.width, self.height))
        sdo = self.Brushfire_Algorithm(brushfire_iter)
        for i in range(self.width):
            for j in range(self.height):
                if sdo[i, j] != 0:
                    heuristic_grid[i, j] = self.U_tot(
                        np.array([i, j]), self.end, sdo[i, j], 4
                    )
        return 5 * (1 - heuristic_grid / np.max(heuristic_grid))

    def can_go_grid_list(self, position, previous):
        """List the free neighbour cells an ant can move to.

        Args:
            position (tuple): Current cell
            previous (tuple): Previous cell

        Returns:
            list: Reachable cells
        """
        x = position[0]
        y = position[1]
        x_p = previous[0]
        y_p = previous[1]
        grid_list = []
        for i in range(max(x - 1, 0), min(x + 2, self.width)):
            for j in range(max(y - 1, 0), min(y + 2, self.height)):
                if (
                    (not (i, j) == (x_p, y_p))
                    and (not (i, j) == (x, y))
                    and (not (i, j) in self.obstacles)
                ):
                    grid_list.append((i, j))
        return grid_list

    def move_ant(self):
        """Walk one ant from the start, picking cells by pheromone and heuristic.

        Returns:
            list: Visited cells
        """
        position = self.start
        visited_grids = [(position[0], position[1])]
        for i in range(self.step_count):
            can_go_list = self.can_go_grid_list(visited_grids[i], visited_grids[i - 1])
            p = []  # Probabilities
            for can_go in can_go_list:
                p.append(
                    self.pheromone_grid[can_go[0], can_go[1]] ** self.alpha
                    * self.heuristic_grid[can_go[0], can_go[1]] ** self.beta
                )
            p = np.array(p)
            p /= np.sum(p)
            p = np.cumsum(p)
            r = np.random.random()
            for j in range(len(p)):
                if p[j] >= r:
                    break
            visited_grids.append(can_go_list[j])
            if can_go_list[j][0] == self.end[0] and can_go_list[j][1] == self.end[1]:
                break
        return visited_grids

    def path_len(self, path):
        """Compute the length of a path.

        Args:
            path (list): Cells of the path

        Returns:
            float: Path length, cells
        """
        length = 0
        for i in range(1, len(path)):
            length += np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
        return length

    def geometric_path_optimizer(self, path):
        """Shorten a path by skipping cells that can be cut.

        Args:
            path (list): Cells of the path

        Returns:
            list: Shorter path
        """
        new_path = [path[0]]
        i = 1
        while i < len(path) - 1:
            grid = path[i]
            new_path.append(grid)

            tmp_grids = []
            for j in range(max(grid[0] - 1, 0), min(grid[0] + 2, self.width)):
                for k in range(max(grid[1] - 1, 0), min(grid[1] + 2, self.height)):
                    if not (j, k) in self.obstacles:
                        tmp_grids.append((j, k))
            tmp_grids.remove(grid)
            tmp_grids = [g for g in tmp_grids if g != path[i - 1]]

            for tmp_grid in tmp_grids:
                if tmp_grid in path[i:]:
                    for j in range(path[i:].count(tmp_grid)):
                        i = path[i:].index(tmp_grid) + i

        new_path.append(path[-1])
        return new_path

    def run_ant_colony(self, Q):
        """Run the colony and show the pheromone map at each iteration.

        Args:
            Q (float): Pheromone deposit constant

        Returns:
            list: Pairs (length, path) of the last iteration
        """
        complete_paths = []
        for t in range(self.colony_iter):
            plt.figure()
            plt.imshow(self.pheromone_grid)
            plt.show()
            complete_paths = []
            d_pheromone = np.zeros(shape=(self.width, self.height))
            for i in range(self.ant_count):
                ant_path = self.move_ant()
                if ant_path[-1] == (self.end[0], self.end[1]):
                    ant_path = self.geometric_path_optimizer(ant_path)
                    complete_paths.append((self.path_len(ant_path), ant_path))
                    for grid in ant_path:
                        d_pheromone[grid[0], grid[1]] += Q / self.path_len(
                            complete_paths[-1][1]
                        )
                        for j in range(
                            max(grid[0] - 1, 0), min(grid[0] + 2, self.width)
                        ):
                            for k in range(
                                max(grid[1] - 1, 0), min(grid[1] + 2, self.height)
                            ):
                                if not (j, k) in self.obstacles:
                                    d_pheromone[j, k] += self.pheromone_diffusion(
                                        self.pheromone_grid[j, k]
                                        + d_pheromone[grid[0], grid[1]],
                                        2,
                                        np.linalg.norm(
                                            np.array(grid) - np.array((j, k))
                                        ),
                                    )
            self.pheromone_grid = (
                1 - self.evaporation_rate
            ) * self.pheromone_grid + d_pheromone
        complete_paths = sorted(complete_paths, key=lambda x: x[0])
        return complete_paths

    def run_ant_colony_noshow(self, Q):
        """Run the colony without any plot.

        Args:
            Q (float): Pheromone deposit constant

        Returns:
            list: Pairs (length, path) of the last iteration
        """
        complete_paths = []
        for t in range(self.colony_iter):
            complete_paths = []
            d_pheromone = np.zeros(shape=(self.width, self.height))
            for i in range(self.ant_count):
                ant_path = self.move_ant()
                if ant_path[-1] == (self.end[0], self.end[1]):
                    ant_path = self.geometric_path_optimizer(ant_path)
                    complete_paths.append((self.path_len(ant_path), ant_path))
                    for grid in ant_path:
                        d_pheromone[grid[0], grid[1]] += Q / self.path_len(
                            complete_paths[-1][1]
                        )
                        for j in range(
                            max(grid[0] - 1, 0), min(grid[0] + 2, self.width)
                        ):
                            for k in range(
                                max(grid[1] - 1, 0), min(grid[1] + 2, self.height)
                            ):
                                if not (j, k) in self.obstacles:
                                    d_pheromone[j, k] += self.pheromone_diffusion(
                                        self.pheromone_grid[j, k]
                                        + d_pheromone[grid[0], grid[1]],
                                        2,
                                        np.linalg.norm(
                                            np.array(grid) - np.array((j, k))
                                        ),
                                    )
            self.pheromone_grid = (
                1 - self.evaporation_rate
            ) * self.pheromone_grid + d_pheromone
        complete_paths = sorted(complete_paths, key=lambda x: x[0])
        return complete_paths

    def plot(self, path):
        """Show the grid with obstacles and the path.

        Args:
            path (list): Cells of the path

        Returns:
            ndarray: Grid image values
        """
        table = np.ones(shape=(self.width, self.height))
        for obstacle in self.obstacles:
            table[obstacle[0], obstacle[1]] = 0.0
        for grid in path:
            table[grid[0], grid[1]] = 0.5
        plt.figure()
        plt.imshow(table)
        plt.show()
        return table

    def __repr__(self):
        """Describe the grid size and its obstacles.

        Returns:
            str: Text description
        """
        o = str(self.width) + "x" + str(self.height) + " grid with obstacles:\n"
        o += str(self.obstacles)
        return o


if __name__ == "__main__":

    Obstacles = [
        (0, 2),
        (0, 3),
        (0, 10),
        (0, 11),
        (1, 5),
        (2, 4),
        (2, 5),
        (2, 6),
        (2, 7),
        (2, 17),
        (3, 9),
        (3, 10),
        (3, 16),
        (3, 17),
        (3, 19),
        (4, 11),
        (4, 16),
        (4, 19),
        (5, 5),
        (5, 11),
        (5, 12),
        (5, 16),
        (5, 19),
        (6, 1),
        (6, 5),
        (6, 6),
        (6, 10),
        (6, 11),
        (7, 1),
        (7, 2),
        (7, 5),
        (7, 6),
        (7, 7),
        (7, 11),
        (8, 2),
        (8, 6),
        (8, 14),
        (9, 6),
        (9, 11),
        (9, 14),
        (10, 3),
        (10, 6),
        (10, 11),
        (10, 12),
        (10, 14),
        (10, 15),
        (11, 3),
        (11, 6),
        (11, 12),
        (11, 14),
        (11, 15),
        (13, 8),
        (13, 9),
        (14, 3),
        (14, 5),
        (14, 8),
        (14, 9),
        (15, 4),
        (15, 5),
        (15, 9),
        (15, 10),
        (15, 13),
        (15, 14),
        (17, 3),
        (17, 6),
    ]

    aco_pdg = ACO_PDG(
        20,
        20,
        Obstacles,
        10,
        50,
        1,
        10,
        0.01,
        0.5,
        np.array([0, 0]),
        np.array([19, 19]),
        3,
        10,
    )

    heuristic_grid = aco_pdg.heuristic_grid
    path1 = aco_pdg.move_ant()
    path2 = aco_pdg.geometric_path_optimizer(path1)
    o = aco_pdg.run_ant_colony_noshow(10)
    aco_pdg.plot(o[0][1])
    print("path3", o[0][1])
    print("Shortest path length: " + str(o[0][0]))

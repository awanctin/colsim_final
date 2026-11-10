from ..sea_object import *


class Fisherman(SeaObject):

    def __init__(
        self,
        mmsi,
        x,
        y,
        v,
        theta,
        algo="straight_goal",
        destination_distance=10000,
        map_size=12000,
        goal=None,
    ):
        """Create a fishing vessel with high privilege.

        Args:
            mmsi (int): Vessel identifier
            x (float): Start x position, m
            y (float): Start y position, m
            v (float): Start speed, m/s
            theta (float): Start heading, rad
            algo (str): Name of the path planning algorithm
            destination_distance (float): Distance to the goal, m
            map_size (float): Half size of the map, m
            goal (tuple): Goal position, m, or None
        """
        super().__init__(
            mmsi, x, y, v, theta, algo, destination_distance, map_size, goal=goal
        )
        self.privilege = 100
        self.color = "yellow"
        self.name = "Fisherman"

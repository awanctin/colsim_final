from ..experiment_constants import R_VESSEL_M
from ..sea_object import SeaObject
from ..display.calcul_tools import *
from ..display.draw import *


class Agent(SeaObject):

    def __init__(
        self,
        mmsi,
        x,
        y,
        v,
        theta,
        algo="APF",
        destination_distance=10000,
        map_size=12000,
        sea_objects=None,
        goal=None,
    ):
        """Create the learning agent vessel.

        Args:
            mmsi (int): Vessel identifier
            x (float): Start x position, m
            y (float): Start y position, m
            v (float): Start speed, m/s
            theta (float): Start heading, rad
            algo (str): Name of the path planning algorithm
            destination_distance (float): Distance to the goal, m
            map_size (float): Half size of the map, m
            sea_objects (list): Other sea objects, unused
            goal (tuple): Goal position, m, or None
        """
        super().__init__(
            mmsi, x, y, v, theta, algo, destination_distance, map_size, goal=goal
        )
        self.privilege = 0
        self.r = R_VESSEL_M
        self.name = "Agent"

    def move(
        self,
        record_data,
        boats,
        mmsi_list,
        rules,
        table,
        ax,
        Ɛ,
        s,
        k,
        dt,
        verbose=False,
    ):
        """Returns mmsi and same state vector"""
        # The agent moves with RL, not with the simulation
        return [
            self.mmsi,
            self.name,
            self.get_state_vector(),
            self.collision_risk,
            self.collision_history,
        ]

    def draw(self, ax, Ɛ, alpha=1):
        """Draw the agent, its speed vector and its two safety circles.

        Args:
            ax (matplotlib.axes.Axes): Axes to draw on
            Ɛ (float): Safety margin added to the radius, m
            alpha (float): Transparency
        """
        draw_boat_and_vector(ax, self.get_state_vector())  # Display of the boat
        # DCPA zone to avoid related to the boat
        draw_circle(ax, self.x, self.y, self.r, "red", alpha=alpha)
        # DCPA zone extended for safety : manoeuvring area
        draw_circle(ax, self.x, self.y, self.r + Ɛ, "magenta", alpha=alpha)

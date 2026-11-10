from ..display.draw import *
from ..sea_object import *


class Boat(SeaObject):

    def __init__(
        self,
        mmsi,
        x,
        y,
        v,
        theta,
        algo,
        destination_distance=10000,
        map_size=12000,
        sea_objects=None,
        goal=None,
        record_path="plots",
    ):
        """Create a boat with zero privilege that must give way.

        Args:
            mmsi (int): Vessel identifier
            x (float): Start x position, m
            y (float): Start y position, m
            v (float): Start speed, m/s
            theta (float): Start heading, rad
            algo (str): Name of the path planning algorithm
            destination_distance (float): Distance to the goal, m
            map_size (float): Half size of the map, m
            sea_objects (list): Other sea objects
            goal (tuple): Goal position, m, or None
            record_path (str): Folder for saved plots
        """
        # call the superclass's constructor
        super().__init__(
            mmsi, x, y, v, theta, algo, destination_distance, map_size, goal=goal
        )
        self.privilege = 0
        self.r = R_VESSEL_M
        self.length = 150.0
        self.breadth = 25.0
        self.name = "Boat"
        self.agent = False
        self.sea_objects = (
            [] if sea_objects is None else sea_objects
        )  # no shared default (F046)
        self.algo = algo
        # self.algo = "RRT*"
        self.record_path = record_path
        self.x0, self.y0 = x, y  # Initial position of the boat

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
        """Returns mmsi and state vector, depending on the path planning algorithm"""
        return self._dispatch_move(
            record_data, boats, mmsi_list, rules, table, ax, Ɛ, s, k, dt, verbose
        )

    # Draw circle around boat

    def draw(self, ax, Ɛ, alpha=1):
        """Draw the boat, its speed vector and its two safety circles.

        Args:
            ax (matplotlib.axes.Axes): Axes to draw on
            Ɛ (float): Safety margin added to the radius, m
            alpha (float): Transparency
        """
        draw_boat_and_vector(ax, self.get_state_vector())  # Display of the boat
        # DCPA zone to avoid related to the boat
        draw_circle(ax, self.x, self.y, self.r, "red", alpha)
        # DCPA zone extended for safety : manoeuvring area
        draw_circle(ax, self.x, self.y, self.r + Ɛ, "magenta", alpha)

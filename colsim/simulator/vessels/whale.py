from ..sea_object import *
from ..display.calcul_tools import *
from ..display.draw import *


class Whale(SeaObject):

    def __init__(
        self,
        mmsi,
        x,
        y,
        v,
        theta,
        algo,
        destination_distance=20000,
        map_size=12000,
        sea_objects=None,
        goal=None,
    ):
        """Create a whale, an obstacle with very high privilege.

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
        # call the superclass's constructor
        super().__init__(
            mmsi, x, y, v, theta, algo, destination_distance, map_size, goal=goal
        )
        self.privilege = 500
        self.r = 15
        self.name = "Whale"

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

    def draw(self, ax, Ɛ, col="blue", coef=0.1, w=2, alpha=1):
        """Display of the whale"""
        M = coef * array(
            [
                [-1, 5, 7, 7, 5, -1, -4, -5, -7, -7, -5, -4, -1],
                [-3, -3, -2, 2, 3, 3, 1, 1, 3, -3, -1, -1, -3],
            ]
        )
        M = add1(M)
        draw_arrow(ax, self.x, self.y, self.theta, norm(self.v), "red")
        M_transformed = tran2H(self.x, self.y) @ rot2H(self.theta) @ M
        ax.plot(
            M_transformed[0],
            M_transformed[1],
            color=col,
            linewidth=w,
            zorder=1,
            alpha=alpha,
        )
        ax.add_patch(
            Polygon(
                M_transformed[:2].T,
                facecolor=col,
                edgecolor=None,
                zorder=0,
                alpha=alpha,
            )
        )
        """ Display of the zones """
        draw_circle(
            ax, self.x, self.y, self.r, "red", alpha=alpha
        )  # DCPA zone to avoid related to the boat
        # DCPA zone extended for safety : manoeuvring area
        draw_circle(ax, self.x, self.y, self.r + Ɛ, "magenta", alpha=alpha)

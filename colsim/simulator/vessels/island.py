from ..sea_object import *
from ..display.calcul_tools import *
from ..display.draw import *


class Island(SeaObject):

    def __init__(
        self,
        mmsi,
        x,
        y,
        v,
        theta,
        algo="APF",
        destination_distance=0,
        map_size=25,
        sea_objects=None,
        goal=None,
    ):
        """Create a fixed island with a 100 m radius.

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
        self.name = "Island"

        # call the superclass's constructor
        super().__init__(
            mmsi, x, y, v, theta, algo, destination_distance=0, map_size=1, goal=goal
        )
        self.privilege = 1000
        self.r = 100
        # An island never moves

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
        return [
            self.mmsi,
            self.name,
            self.get_state_vector(),
            self.collision_risk,
            self.collision_history,
        ]

    def draw(
        self, ax, Ɛ, col1="darkkhaki", col2="limegreen", r1=0.3, r2=0.15, w=2, alpha=1
    ):
        """Display of the island"""
        M1 = r1 * np.array(
            [
                [
                    -5,
                    -4.5,
                    -3,
                    -2,
                    -1,
                    1,
                    2,
                    3,
                    2,
                    1,
                    1,
                    1.5,
                    2.5,
                    4,
                    3,
                    2,
                    1,
                    -1,
                    -1.5,
                    -3,
                    -4,
                    -5,
                ],
                [
                    0,
                    2,
                    3,
                    2.5,
                    4,
                    4,
                    3.75,
                    3,
                    2.5,
                    1.5,
                    0.75,
                    0,
                    0,
                    -1,
                    -2.5,
                    -3.5,
                    -3.5,
                    -2.5,
                    -2.5,
                    -1.25,
                    -1,
                    0,
                ],
            ]
        )
        M1 = add1(M1)
        draw_arrow(ax, self.x, self.y, self.theta, np.linalg.norm(self.v), "red")
        M1_transformed = tran2H(self.x, self.y) @ rot2H(self.theta) @ M1
        ax.plot(
            M1_transformed[0],
            M1_transformed[1],
            color=col1,
            linewidth=w,
            zorder=1,
            alpha=alpha,
        )
        ax.add_patch(
            Polygon(
                M1_transformed[:2].T,
                facecolor=col1,
                edgecolor=None,
                zorder=0,
                alpha=alpha,
            )
        )

        M2 = r2 * np.array(
            [
                [
                    -5,
                    -4.5,
                    -3,
                    -2,
                    -1,
                    1,
                    2,
                    4,
                    2,
                    1,
                    1,
                    1.5,
                    2.5,
                    6,
                    3,
                    2,
                    1,
                    -1,
                    -1.5,
                    -3,
                    -4,
                    -5,
                ],
                [
                    0,
                    2,
                    3,
                    2.5,
                    4,
                    4,
                    3.75,
                    6,
                    2.5,
                    1.5,
                    0.75,
                    0,
                    0,
                    -4,
                    -2.5,
                    -3.5,
                    -3.5,
                    -2.5,
                    -2.5,
                    -1.25,
                    -1,
                    0,
                ],
            ]
        )
        M2 = add1(M2)
        M2_transformed = tran2H(self.x, self.y - 0.25) @ rot2H(self.theta) @ M2
        ax.plot(
            M2_transformed[0],
            M2_transformed[1],
            color=col2,
            linewidth=w,
            zorder=1,
            alpha=alpha,
        )
        ax.add_patch(
            Polygon(
                M2_transformed[:2].T,
                facecolor=col2,
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
        draw_circle(ax, self.x, self.y, self.r + Ɛ, "green", alpha=alpha)

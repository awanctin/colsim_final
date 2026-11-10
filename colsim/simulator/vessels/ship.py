from ..display.draw import *
from ..sea_object import *


class Ship(SeaObject):

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
        """Create a ship with medium privilege.

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
        self.privilege = 30
        self.r = R_VESSEL_M
        self.name = "Ship"

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
        """Move the ship one step with its planning algorithm.

        Args:
            record_data (bool): Whether to store the trajectory
            boats (list): All sea objects
            mmsi_list (list): Identifiers of all sea objects
            rules (list): Active rules table
            table (matplotlib.table.Table): Rules table display, or None
            ax (matplotlib.axes.Axes): Axes to draw on, or None
            Ɛ (float): Safety margin added to the radius, m
            s (float): Half size of the map, m
            k (int): Current step index
            dt (float): Time step, s
            verbose (bool): Whether to print debug output

        Returns:
            tuple: Identifier and new state vector
        """
        return self._dispatch_move(
            record_data, boats, mmsi_list, rules, table, ax, Ɛ, s, k, dt, verbose
        )

    def draw(
        self,
        ax,
        Ɛ,
        col1="royalblue",
        col2="steelblue",
        col3="slategray",
        r1=0.3,
        w=1,
        alpha=1,
    ):
        """Draw the ship hull, its speed vector and its two safety circles.

        Args:
            ax (matplotlib.axes.Axes): Axes to draw on
            Ɛ (float): Safety margin added to the radius, m
            col1 (str): Hull colour
            col2 (str): Deck colour
            col3 (str): Cabin colour
            r1 (float): Scale of the hull shape
            w (float): Line width
            alpha (float): Transparency
        """
        M1 = r1 * np.array(
            [[-6.5, -6, 4, 7, 4, -6, -6.5, -6.5], [1.5, 3, 3, 0, -3, -3, -1.5, 1.5]]
        )
        M1 = add1(M1)
        draw_arrow(ax, self.x, self.y, self.theta, np.linalg.norm(self.v), "red")
        M1_transformed = tran2H(self.x, self.y) @ rot2H(self.theta) @ M1
        ax.plot(
            M1_transformed[0],
            M1_transformed[1],
            color="black",
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

        M2 = r1 * np.array([[-5, 2, 2, -5, -5], [2, 2, -2, -2, 2]])
        M2 = add1(M2)
        M2_transformed = tran2H(self.x, self.y) @ rot2H(self.theta) @ M2
        ax.plot(
            M2_transformed[0],
            M2_transformed[1],
            color="black",
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

        M3 = r1 * np.array([[-4, 1, 1, -4, -4], [1, 1, -1, -1, 1]])
        M3 = add1(M3)
        M3_transformed = tran2H(self.x, self.y) @ rot2H(self.theta) @ M3
        ax.plot(
            M3_transformed[0],
            M3_transformed[1],
            color="black",
            linewidth=w,
            zorder=1,
            alpha=alpha,
        )
        ax.add_patch(
            Polygon(
                M3_transformed[:2].T,
                facecolor=col3,
                edgecolor=None,
                zorder=0,
                alpha=alpha,
            )
        )

        M4 = r1 * np.array([[3, 5, 3, 3], [2, 0, -2, 2]])
        M4 = add1(M4)
        M4_transformed = tran2H(self.x, self.y) @ rot2H(self.theta) @ M4
        ax.plot(
            M4_transformed[0],
            M4_transformed[1],
            color="black",
            linewidth=w,
            zorder=1,
            alpha=alpha,
        )
        ax.add_patch(
            Polygon(
                M4_transformed[:2].T,
                facecolor=col2,
                edgecolor=None,
                zorder=0,
                alpha=alpha,
            )
        )

        """ Display of the zones """
        draw_circle(ax, self.x, self.y, self.r, "red", alpha=alpha)
        draw_circle(ax, self.x, self.y, self.r + Ɛ, "magenta", alpha=alpha)

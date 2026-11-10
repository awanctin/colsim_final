from collections import defaultdict
import matplotlib.pyplot as plt


class SpatialGrid:
    def __init__(self, cell_size, map_size):
        """Create an empty grid of square cells.

        Args:
            cell_size (float): Cell side, m
            map_size (float): Map size, m
        """
        self.cell_size = cell_size
        self.grid = defaultdict(list)
        self.map_size = map_size

    def get_cell(self, x, y):
        """Find the cell that holds a point.

        Args:
            x (float): Point x, m
            y (float): Point y, m

        Returns:
            tuple: Cell indices (cx, cy)
        """
        return (int(x // self.cell_size), int(y // self.cell_size))

    def add_object(self, obj):
        """Register an object in the cells around it and around its goal.

        Args:
            obj (SeaObject): Object to add
        """
        cells = self.get_relevant_cells(obj.x, obj.y, obj.r)
        for cell in cells:
            self.grid[cell].append(obj)
        if obj.name != "Island":
            goal_cells = self.get_relevant_cells(obj.goal_x, obj.goal_y, obj.r)
            for cell in goal_cells:
                self.grid[cell].append(obj)

    def get_relevant_cells(self, x, y, r):
        """List the cells covered by a disc plus one cell of margin.

        Args:
            x (float): Disc centre x, m
            y (float): Disc centre y, m
            r (float): Disc radius, m

        Returns:
            set: Cell indices (cx, cy)
        """
        affected_cells = set()
        min_x = x - r - self.cell_size
        max_x = x + r + self.cell_size
        min_y = y - r - self.cell_size
        max_y = y + r + self.cell_size

        for cx in range(int(min_x // self.cell_size), int(max_x // self.cell_size) + 1):
            for cy in range(
                int(min_y // self.cell_size), int(max_y // self.cell_size) + 1
            ):
                affected_cells.add((cx, cy))
        return affected_cells

    def plot_grid(self, ax=None, color="lightgray", path="spatial_grid.png"):
        """Draw the filled cells and their objects, then save the figure.

        Args:
            ax (matplotlib.axes.Axes): Unused, a new figure is made
            color (str): Cell edge colour
            path (str): Output image file
        """
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)
        for (cx, cy), objs in self.grid.items():
            rect = plt.Rectangle(
                (cx * self.cell_size, cy * self.cell_size),
                self.cell_size,
                self.cell_size,
                linewidth=1,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)
            for obj in objs:
                obj.draw(ax, 1)
        ax.set_aspect("equal", adjustable="box")
        plt.grid(True)
        plt.savefig(path, dpi=600)
        plt.show()

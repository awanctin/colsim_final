import os

try:
    # COLSIM_RRT_NUMBA=0 forces the Python reference, for A/B traces.
    if os.environ.get("COLSIM_RRT_NUMBA", "1") == "0":
        raise ImportError("disabled by COLSIM_RRT_NUMBA=0")
    from . import _rrt_nb as _nb
except ImportError:
    _nb = None


class SpatialGrid:

    def __init__(self, bounds, cell_size=20.0):
        """Create an empty obstacle grid over the map bounds.

        Args:
            bounds (list): Map limits [x_min, x_max, y_min, y_max], m
            cell_size (float): Cell side, m
        """
        self.cell_size = cell_size
        self.bounds = bounds  # [x_min, x_max, y_min, y_max]
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        # Obstacles are recorded, and each index is built on first use: the dict for the Python
        # queries, CSR arrays for the numba kernels (main_program/_rrt_nb.py). A planner that
        # only runs the kernels never pays for the dict.
        self._grid = {}
        self._pending = []
        self._obstacles = []
        # An (N, 3) float array from `add_obstacle_array`, turned into tuples only when needed.
        self._array = None
        self._csr = None

    @property
    def grid(self):
        """Build the dict index on first use and return it.

        Returns:
            dict: Cell to list of obstacles (x, y, radius)
        """
        self._unpack_array()
        for x, y, radius in self._pending:
            self._insert(x, y, radius)
        self._pending = []
        return self._grid

    def csr(self):
        """Build the CSR arrays for the numba kernels on first use.

        Returns:
            tuple: Grid arrays for _rrt_nb
        """
        if self._csr is None:
            obs = self._obstacles if self._array is None else self._array
            self._csr = _nb.grid_arrays(obs, self.bounds, self.cell_size)
        return self._csr

    def _unpack_array(self):
        """Turn the pending obstacle array into tuples."""
        if self._array is None:
            return
        rows = [tuple(row) for row in self._array.tolist()]
        self._array = None
        self._pending.extend(rows)
        self._obstacles.extend(rows)

    def _get_cell(self, x, y):
        """Convert world coordinates to grid cell"""
        cell_x = int((x - self.x_min) // self.cell_size)
        cell_y = int((y - self.y_min) // self.cell_size)
        return (cell_x, cell_y)

    def add_obstacle(self, x, y, radius):
        """Add obstacle to spatial grid"""
        self._unpack_array()
        self._pending.append((x, y, radius))
        self._obstacles.append((x, y, radius))
        self._csr = None

    def add_obstacles(self, obstacles):
        """`add_obstacle` for each `(x, y, radius)`, in order."""
        self._unpack_array()
        self._pending.extend(obstacles)
        self._obstacles.extend(obstacles)
        self._csr = None

    def add_obstacle_array(self, obstacles, csr=None):
        """`add_obstacles` for an (N, 3) float array; the grid keeps it as an array.
        `csr` is the `csr()` tuple of an empty grid given these obstacles, when already built.
        """
        if self._obstacles or self._array is not None:
            self.add_obstacles([tuple(row) for row in obstacles.tolist()])
            return
        self._array = obstacles
        self._csr = csr

    def _insert(self, x, y, radius):
        """File one obstacle in every cell it touches.

        Args:
            x (float): Obstacle x, m
            y (float): Obstacle y, m
            radius (float): Obstacle radius, m
        """
        # One shared tuple, one dict probe per cell.
        ob = (x, y, radius)
        grid = self._grid
        for cell in self._get_affected_cells(x, y, radius):
            grid.setdefault(cell, []).append(ob)

    def _get_affected_cells(self, x, y, radius):
        """Get all cells that an obstacle affects"""
        nx_max = int((self.x_max - self.x_min) // self.cell_size)
        ny_max = int((self.y_max - self.y_min) // self.cell_size)
        # Clip to the grid's own bounds. Unclipped, one obstacle of radius 223 130 m on a
        # +/-7867 m grid enumerated 19 922 832 cells and took 8.3 GB of anon RSS before the
        # kernel killed the process (2026-08-21, big_traffic s=30000 seed 10; the same 9.5 GB
        # was charged to a joblib worker on 2026-08-20). Nothing can query a cell outside the
        # bounds: is_collision_free rejects out-of-bounds points before any lookup, and an
        # obstacle covering an in-bounds point also occupies that point's own cell.
        min_cell_x = max(0, int((x - radius - self.x_min) // self.cell_size))
        max_cell_x = min(nx_max, int((x + radius - self.x_min) // self.cell_size))
        min_cell_y = max(0, int((y - radius - self.y_min) // self.cell_size))
        max_cell_y = min(ny_max, int((y + radius - self.y_min) // self.cell_size))

        cells = []
        for cx in range(min_cell_x, max_cell_x + 1):
            for cy in range(min_cell_y, max_cell_y + 1):
                cells.append((cx, cy))
        return cells

    def is_collision_free(self, x, y, safety_margin=0):
        """Check that a point is inside the map and clear of obstacles.

        Args:
            x (float): Point x, m
            y (float): Point y, m
            safety_margin (float): Extra clearance, m

        Returns:
            bool: True if the point is free
        """
        if _nb is not None:
            return bool(
                _nb.grid_free(float(x), float(y), float(safety_margin), *self.csr())
            )
        return self._is_collision_free_py(x, y, safety_margin)

    def _is_collision_free_py(self, x, y, safety_margin=0):
        """Pure Python version of is_collision_free.

        Args:
            x (float): Point x, m
            y (float): Point y, m
            safety_margin (float): Extra clearance, m

        Returns:
            bool: True if the point is free
        """
        # Check if point is within bounds
        if x <= self.x_min or x >= self.x_max or y <= self.y_min or y >= self.y_max:
            return False

        cell = self._get_cell(x, y)

        # Every cell within the margin of this one (F059): an obstacle is filed by its radius
        # only, so its edge can sit up to margin / cell cells away. Was the 8 neighbours.
        reach = 1 + int(safety_margin // self.cell_size)
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                check_cell = (cell[0] + dx, cell[1] + dy)
                if check_cell in self.grid:
                    for ox, oy, radius in self.grid[check_cell]:
                        dist_sq = (x - ox) ** 2 + (y - oy) ** 2
                        threshold_sq = (radius + safety_margin) ** 2
                        if dist_sq <= threshold_sq:
                            return False
        return True

    def query_obstacles(self, x, y, radius, safety_margin=0):
        """
        Queries for obstacles within a given radius of a point (x, y).
        """

        # Set to take into account the obstacles overlqping
        candidate_obstacles = set()

        # Bounding box
        min_ix = int((x - radius - self.bounds[0]) / self.cell_size)
        max_ix = int((x + radius - self.bounds[0]) / self.cell_size)
        min_iy = int((y - radius - self.bounds[2]) / self.cell_size)
        max_iy = int((y + radius - self.bounds[2]) / self.cell_size)

        for ix in range(min_ix, max_ix + 1):
            for iy in range(min_iy, max_iy + 1):
                cell_key = (ix, iy)
                if cell_key in self.grid:
                    for obstacle in self.grid[cell_key]:
                        candidate_obstacles.add(obstacle)

        intersecting_obstacles = []
        for obs in candidate_obstacles:
            ox, oy, o_size = obs
            dist_sq = (x - ox) ** 2 + (y - oy) ** 2
            effective_obs_radius = o_size + safety_margin

            if dist_sq < (effective_obs_radius + radius) ** 2:
                intersecting_obstacles.append(obs)

        return intersecting_obstacles


def _selfcheck():
    """A huge obstacle must not enumerate cells outside the grid, and must still be seen."""
    g = SpatialGrid([-1000.0, 1000.0, -1000.0, 1000.0], cell_size=100.0)
    g.add_obstacle(0.0, 0.0, 200000.0)
    assert len(g.grid) == 21 * 21, len(g.grid)
    assert not g.is_collision_free(500.0, 500.0)
    assert not g.is_collision_free(-999.0, 0.0)

    g2 = SpatialGrid([-1000.0, 1000.0, -1000.0, 1000.0], cell_size=100.0)
    g2.add_obstacle(500.0, 500.0, 50.0)
    assert not g2.is_collision_free(520.0, 500.0)
    assert g2.is_collision_free(0.0, 0.0)
    assert not g2.is_collision_free(5000.0, 0.0)
    print("rrt_spatial_grid selfcheck OK")


if __name__ == "__main__":
    _selfcheck()

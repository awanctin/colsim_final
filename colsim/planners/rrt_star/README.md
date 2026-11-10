# rrt_star

Default planner.
RRT* with COLREGs edge penalties, wrapped by a planner that predicts the targets.

| File | Role |
|---|---|
| `colregs_planner.py` | `COLREGsPlanner`: adds predicted target positions as obstacle discs, then plans |
| `algo_rttstar.py` | `RRTStar`, `PathOptimizer`, `MaritimeValidator`: Python RRT* and path smoothing |
| `_rrt_nb.py` | Numba port of the local RRT*, bit-identical to Python (own Mersenne Twister, libm bindings) |
| `rrt_spatial_grid.py` | `SpatialGrid`: obstacle lookup for the tree, `_selfcheck()` included |

`_rrt_nb.supports` decides whether an instance can use the numba path.

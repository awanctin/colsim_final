# grid_search

Graph search on an occupancy grid.

| File | Role |
|---|---|
| `algo_astar.py` | `AStar`, plus run time against obstacle count |
| `algo_dstar_lite.py` | `DStarLite`, replans when the grid changes, demo in `main()` |

`planners/rrt_star/colregs_planner.py` imports `algo_astar`.

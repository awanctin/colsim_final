# planners

Collision avoidance methods.
One subfolder per family, one method per `algo_*.py`.

| Folder | Methods |
|---|---|
| `rrt_star/` | RRT* with COLREGs penalties, the default planner |
| `mpc/` | SB-MPC, PSB-MPC, RTA-MPC |
| `reactive/` | Velocity obstacle, potential fields |
| `grid_search/` | A*, D* Lite |
| `metaheuristics/` | PSO, ACO |
| `learned/` | KRAS, RL under a COLREGs statechart |

`simulator/sea_object.py` picks the method of each vessel.

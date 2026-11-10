# simulator

The world and its step loop.

| File | Role |
|---|---|
| `main.py` | Entry point: 100 runs of Imazu scenario 2, writes `simulation_trajectories_0.{txt,png}` |
| `simulation_runner.py` | `SimulationRunner`: builds a scenario (fixed, Imazu or random) and runs it |
| `simu.py` | `Simulation`: moves and draws every object each step, saves states to CSV |
| `sea_object.py` | `SeaObject`: base class of every object, holds state, planner choice and steering |
| `experiment_constants.py` | `RuleMode` and shared knobs, `apply_capability` sets them on every vessel |
| `spatial_grid.py` | `SpatialGrid`: cell index of objects, used to place them on free spots |

Subfolders: `vessels/` (object types), `display/` (figures).

`sea_object.py` is the largest file (4 151 lines).
It picks the planner from `planners/` and calls `control/` for the command.

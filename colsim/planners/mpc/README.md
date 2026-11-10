# mpc

Model predictive control baselines.

| File | Method | Source |
|---|---|---|
| `algo_sbmpc.py` | SB-MPC: picks a course offset and speed factor from a finite set | Johansen, Perez and Cristofaro (2016) |
| `algo_psbmpc.py` | PSB-MPC through a C++ library (ctypes) | Tengesdal et al. (2024), NTNU |
| `algo_rtampc.py` | RTA-MPC: risk from DCPA and TCPA, encounter class, rollout | Wang et al. (2026) |

PSB-MPC needs the built `libpsbmpc_bridge.so`, path in `COLSIM_PSBMPC_LIB`.

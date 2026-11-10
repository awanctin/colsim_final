# control

Turns a target heading or speed into the command applied each step.

| File | Role |
|---|---|
| `_steer_nb.py` | Numba kernel: steer to a waypoint with the COLREGs veto, floor and snap, then integrate one step |
| `speed_governor.py` | `SpeedGovernor`: commanded speed from the collision risk index (DCPA, TCPA) of the most risky target |
| `cbf_filter.py` | `SafetyFilter`: CBF correction of the nominal command, after Gundersen, Hinostroza and Lekkas (ECC 2026) |

`_steer_nb.command_reference` is the Python reference.
`_selfcheck()` compares it with the kernel on random states.

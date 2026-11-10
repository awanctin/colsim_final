# vessels

Object types placed in the world.
All except `PID` inherit from `SeaObject` (`../sea_object.py`).

| File | Class | Behaviour |
|---|---|---|
| `ship.py` | `Ship` | Moves with its planning algorithm, the vessel used in evaluations |
| `boat.py` | `Boat` | Moves with its planning algorithm, smaller drawing |
| `agent.py` | `Agent` | Kept still, returns the same state |
| `fisherman.py` | `Fisherman` | Fishing vessel, reuses the `SeaObject` motion |
| `whale.py` | `Whale` | Moving obstacle |
| `island.py` | `Island` | Static obstacle |
| `boat_pid.py` | `PID` | PID controller, one command per step |

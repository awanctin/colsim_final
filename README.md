# COLSim

COLSim simulates several vessels at sea and tests COLREGs collision avoidance on them.
It ships pluggable planners (RRT* by default, VO, SB-MPC, PSB-MPC, CBF, RTA-MPC), a duty model, and the Hybrid controller: RRT* with an RL emergency policy behind a safety certificate.

Sorted on 2026-11-10 from `aaaa_colsim`.
Comments and module docstrings were stripped and every function carries a Google-style docstring.

## Quick start (Linux, Python 3.11)

```bash
# 1. Clone
git clone https://github.com/awanctin/colsim_final.git
# 2. Enter the folder
cd colsim_final
# 3. Virtual environment, CPU-only torch
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
# 4. Example: Imazu case 2 with RRT*, writes out/imazu2.png and out/imazu2.gif (about 15 s)
python examples/imazu2.py
```

Use `pip install -e ".[kras]"` to train the KRAS baseline (`sb3-contrib`).
Linux only: `colsim/certificate/_monitor_nb.py` loads glibc `libm.so.6` at import.
On Windows, use WSL.

The numba kernels compile on the first call, so the first run is slower.
Later runs load them from `__pycache__/`.

## Run

Commands work from any folder once the package is installed.
Paths below assume the repository root as the current folder.

```bash
# 1. Plain simulation: 100 runs of Imazu scenario 2, writes simulation_trajectories_0.{txt,png}
python -m colsim.simulator.main

# 2. Train a Hybrid policy (smoke size)
colsim-train --bank data/train_alarm.csv --gate 500 --envs 1 --steps 512 --eval-episodes 2 \
    --episode-start alarm --include-free --seed 1 --run-dir out/train

# 3. Evaluate the compliant planner, then the trained policy, on 2 bank rows
colsim-eval --gate 500 --arms compliant --bank data/far_bank/cat_q_w567.csv --n 2 --pair "" --json out/ev.json
colsim-eval --run out/train --arms hybrid --bank data/far_bank/cat_q_w567.csv --gate 500 --n 2 --pair "" --json out/ev_h.json
```

`colsim-train` is `python -m colsim.rl.train_cert`, `colsim-eval` is `python -m colsim.rl.evaluate_cert`.
Both take `--help`.

## Layout

```
requirements.txt   pinned dependencies, read by pyproject.toml
examples/          imazu2.py: one Imazu case, static figure and GIF
pyproject.toml     package definition, colsim-train and colsim-eval
data/              banks: far_bank/cat_q_w567.csv (evaluation), train_alarm.csv (training)
colsim/
  simulator/       world, step loop, sea objects, entry point main.py
    vessels/       object types: Ship, Boat, Agent, Fisherman, Whale, Island, PID
    display/       2D drawing and 3D trajectory view
  control/         low-level commands: steering kernel, speed governor, CBF filter
  planners/        collision avoidance methods, one subfolder per family
  certificate/     COLREGs duty model, escape measure, safety certificate, monitor
  evaluation/      bank of encounters and the rule compliance check
  rl/              Hybrid emergency policy: environment, training, evaluation, bank mining
  _nb_cache.py     drops stale numba caches of control/_steer_nb.py and planners/rrt_star/_rrt_nb.py
```

Each folder has its own README.

## Licence

GPL-3.0-or-later, see `LICENSE`.

## How the parts connect

```
simulator  --asks a heading-->  planners  --uses-->  certificate (duty, lawful headings)
    |                                                      |
    +--applies the command-->  control  <--switch, monitor-+
rl  --wraps the simulator as a Gymnasium env, trains the policy cert_switch runs-->  certificate
evaluation  --reads tracks and banks-->  certificate
```

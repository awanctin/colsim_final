from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time

import numpy as np

from ..evaluation import bank as _bank
from ..certificate import duty as _duty
from ..certificate.behaviour import class_mix
from ..certificate.escape_angle import (
    HULL_CONTACT_M,
    Vessel,
    cpa,
    place_encounter,
    place_threat,
)
from .cert_env import MAX_STEPS, CertifiedEscapeEnv
from .evaluate_cert import _ARM_KW

GATE_M = 500.0
STRATUM = "reactive"
RELAXED_ARM = "relaxed"
PLANNER_ARMS = ("compliant", "letter", "vo")
N_REFINE = 2
PLACE_TRIES = 50
PROGRESS_EVERY = 25
SPEED_RANGE_MPS = (5.14, 9.26)
T_MEET_RANGE_S = (120.0, 480.0)

COMPATIBLE_QUADRANTS = {
    "Head-on": ("Stbd-Ahead", "Port-Ahead"),
    "Crossing_GiveWay": ("Stbd-Ahead", "Stbd-Astern"),
    "Crossing_StandOn": ("Port-Ahead", "Port-Astern"),
    "Overtaking_GiveWay": ("Port-Ahead", "Stbd-Ahead"),
    "Being_Overtaken_StandOn": ("Port-Astern", "Stbd-Astern"),
}
ATTEMPT_FIELDS = ["cell", "attempt", "seed", "planner", "placed", "refine_iters",
                  "min_sep_planner", "min_sep_relaxed", "accepted"]
EXTRA_FIELDS = ["cell_ts1", "cell_ts2", "cell_quadrant", "planner", "refine_iters",
                "min_sep_planner", "min_sep_relaxed"]


def cells():
    """List every cell of the bank grid: two situations and a quadrant.

    Returns:
        list: Cells (situation 1, situation 2, quadrant)
    """
    return [(s1, s2, q) for s1 in COMPATIBLE_QUADRANTS
            for s2, quads in COMPATIBLE_QUADRANTS.items() for q in quads]


def quadrant(os_v, ts):
    """Name the quadrant of a target around own vessel.

    Args:
        os_v (Vessel): Own vessel
        ts (Vessel): Target vessel

    Returns:
        str: Port or Stbd, Ahead or Astern
    """
    dx, dy = ts.x - os_v.x, ts.y - os_v.y
    ax = dx * math.cos(os_v.theta) + dy * math.sin(os_v.theta)
    ay = -dx * math.sin(os_v.theta) + dy * math.cos(os_v.theta)
    side = "Port" if ay >= 0 else "Stbd"
    return f"{side}-{'Ahead' if ax >= 0 else 'Astern'}"


def place_row(rng, cell):
    """Place own vessel, a first target and a second target in the cell quadrant.

    Args:
        rng (Generator): Random generator
        cell (tuple): Bank cell

    Returns:
        tuple: Own vessel and targets, or None
    """
    sit1, sit2, quad = cell
    for _ in range(PLACE_TRIES):
        os_v, threats, _placed = place_encounter(rng, [sit1], t_range=T_MEET_RANGE_S)
        if not threats:
            continue
        ts2, _tries = place_threat(rng, os_v, sit2, name="TS2", t_range=T_MEET_RANGE_S)
        if ts2 is not None and quadrant(os_v, ts2) == quad:
            return os_v, [threats[0], ts2]
    return None


def _row(os_v, threats, idx):
    """Build a bank row dict from vessels.

    Args:
        os_v (Vessel): Own vessel
        threats (list): Target vessels
        idx (int): Row index

    Returns:
        dict: Bank row
    """
    return dict(idx=str(idx), stratum=STRATUM, gate=GATE_M, os=os_v, threats=list(threats))


_ENVS = {}


def _env_for(arm, row, mix):
    """Get the environment of an arm, built once and reused.

    Args:
        arm (str): Arm name
        row (dict): Bank row
        mix (object): Target behaviour mix

    Returns:
        CertifiedEscapeEnv: Environment
    """
    env = _ENVS.get(arm)
    if env is None:
        env = CertifiedEscapeEnv([row], gate=GATE_M, strata=(STRATUM,), ts_classes=mix,
                                 score_after_handback=False, **_ARM_KW[arm])
        _ENVS[arm] = env
    env.rows = [row]
    return env


def simulate(row, arm, seed, mix=None, window=MAX_STEPS):
    """Run one arm on one row and keep the closest separation.

    Args:
        row (dict): Bank row
        arm (str): Arm name
        seed (int): Random seed
        mix (object): Target behaviour mix
        window (int): Number of steps

    Returns:
        tuple: Closest separation (m) and own track
    """
    env = _env_for(arm, row, mix)
    env.reset(seed=seed, options={"row": 0})
    track = [(env._os.x, env._os.y)]
    best = float("inf")
    for _ in range(window):
        _obs, _r, _term, _trunc, info = env.step(None)
        best = min(best, float(info["min_separation"]))
        track.append((env._os.x, env._os.y))
    return best, track


def reaim_ts2(os0, ts2, track, dt):
    """Turn the second target so it meets own track at its CPA time.

    Args:
        os0 (Vessel): Own vessel at start
        ts2 (Vessel): Second target
        track (list): Own positions per step
        dt (float): Time step, s

    Returns:
        Vessel: Re-aimed target, or None
    """
    t_meet = cpa(os0, ts2)[0]
    if not math.isfinite(t_meet) or t_meet <= 0.0:
        return None
    k = min(int(round(t_meet / dt)), len(track) - 1)
    px, py = track[k]
    dist = math.hypot(px - ts2.x, py - ts2.y)
    v = min(max(dist / t_meet, SPEED_RANGE_MPS[0]), SPEED_RANGE_MPS[1])
    return Vessel(ts2.name, ts2.x, ts2.y, v, math.atan2(py - ts2.y, px - ts2.x), ts2.r_min)


def attempt(cell, planner, seed, dt, n_refine=N_REFINE, mix=None):
    """Place a row, then refine it until the planner is trapped.

    Args:
        cell (tuple): Bank cell
        planner (str): Planner arm
        seed (int): Random seed
        dt (float): Time step, s
        n_refine (int): Max refine rounds
        mix (object): Target behaviour mix

    Returns:
        tuple: Attempt record and accepted row, or None
    """
    rng = np.random.default_rng(seed)
    rec = dict(cell="|".join(cell), seed=seed, planner=planner, placed=0, refine_iters=0,
               min_sep_planner="", min_sep_relaxed="", accepted=0)
    placed = place_row(rng, cell)
    if placed is None:
        return rec, None
    os_v, (ts1, ts2) = placed
    rec["placed"] = 1

    sep, track = simulate(_row(os_v, [ts1, ts2], 0), planner, seed, mix)
    for it in range(n_refine):
        if sep < HULL_CONTACT_M:
            break
        new = reaim_ts2(os_v, ts2, track, dt)
        if new is None or _duty.situation(os_v, new) != cell[1] or quadrant(os_v, new) != cell[2]:
            break
        ts2 = new
        rec["refine_iters"] = it + 1
        sep, track = simulate(_row(os_v, [ts1, ts2], 0), planner, seed, mix)
    rec["min_sep_planner"] = sep
    if sep >= HULL_CONTACT_M:
        return rec, None

    relaxed, _ = simulate(_row(os_v, [ts1, ts2], 0), RELAXED_ARM, seed, mix)
    rec["min_sep_relaxed"] = relaxed
    if relaxed < HULL_CONTACT_M:
        return rec, None
    rec["accepted"] = 1
    return rec, (os_v, [ts1, ts2], sep, relaxed)


def _load_attempts(path):
    """Read the attempt log CSV.

    Args:
        path (str): CSV path

    Returns:
        list: Rows as dicts
    """
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _bank_record(idx, seed, cell, planner, os_v, threats, iters, sep, relaxed):
    """Build the bank CSV record of an accepted row.

    Args:
        idx (int): Row index
        seed (int): Random seed
        cell (tuple): Bank cell
        planner (str): Planner arm
        os_v (Vessel): Own vessel
        threats (list): Target vessels
        iters (int): Refine rounds
        sep (float): Planner closest separation, m
        relaxed (float): Relaxed closest separation, m

    Returns:
        dict: CSV record
    """
    lab = _bank.label(os_v, threats, gate=GATE_M)
    lab["stratum"] = STRATUM
    return dict(idx=idx, seed=seed, **lab, cell_ts1=cell[0], cell_ts2=cell[1],
                cell_quadrant=cell[2], planner=planner, refine_iters=iters,
                min_sep_planner=sep, min_sep_relaxed=relaxed)


def main():
    """Build the reactive bank from the command line, resuming partial files."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--planner", choices=PLANNER_ARMS, default="compliant")
    ap.add_argument("--out-dir", default="reactive_bank")
    ap.add_argument("--per-cell", type=int, default=300, help="accepted rows wanted per cell")
    ap.add_argument("--max-attempts", type=int, default=3000, help="attempts per cell, cap")
    ap.add_argument("--refine", type=int, default=N_REFINE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--cells", default=None, help="comma list of cell indices, default all")
    ap.add_argument("--ts-mix", default=None, metavar="MIX",
                    help="target behaviour mix, e.g. 'compliant=1,late=1'; default all compliant")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    tag = f"{a.planner}_{a.shard}"
    att_path = os.path.join(a.out_dir, f"attempts_{tag}.csv")
    bank_path = os.path.join(a.out_dir, f"bank_{tag}.csv")
    grid = cells()
    wanted = range(len(grid)) if a.cells is None else [int(c) for c in a.cells.split(",")]

    done = _load_attempts(att_path)
    seen = {(r["cell"], int(r["attempt"])) for r in done}
    accepted = {}
    for r in done:
        accepted[r["cell"]] = accepted.get(r["cell"], 0) + int(r["accepted"])
    per_shard = math.ceil(a.per_cell / a.shards)

    att_new, bank_new = not os.path.exists(att_path), not os.path.exists(bank_path)
    fields = _bank.FIELDS + EXTRA_FIELDS
    t0, n_run = time.time(), 0
    with open(att_path, "a", newline="", encoding="utf-8") as fa, \
            open(bank_path, "a", newline="", encoding="utf-8") as fb:
        wa = csv.DictWriter(fa, fieldnames=ATTEMPT_FIELDS)
        wb = csv.DictWriter(fb, fieldnames=fields, extrasaction="ignore")
        if att_new:
            wa.writeheader()
        if bank_new:
            wb.writeheader()
        for ci in wanted:
            cell = grid[ci]
            key = "|".join(cell)
            for n in range(a.shard, a.max_attempts, a.shards):
                if accepted.get(key, 0) >= per_shard:
                    break
                if (key, n) in seen:
                    continue
                seed = a.seed + 1_000_000 * ci + n
                rec, row = attempt(cell, a.planner, seed, dt=10.0, n_refine=a.refine,
                                   mix=class_mix(a.ts_mix) if a.ts_mix else None)
                rec["attempt"] = n
                wa.writerow(rec)
                fa.flush()
                if row is not None:
                    accepted[key] = accepted.get(key, 0) + 1
                    os_v, threats, sep, relaxed = row
                    wb.writerow(_bank_record(f"{tag}_{ci}_{n}", seed, cell, a.planner, os_v,
                                             threats, rec["refine_iters"], sep, relaxed))
                    fb.flush()
                n_run += 1
                if n_run % PROGRESS_EVERY == 0:
                    total = sum(accepted.values())
                    print(f"  {n_run} attempts  {n_run / (time.time() - t0):.2f}/s  "
                          f"accepted {total}  cell {ci} {key}: {accepted.get(key, 0)}",
                          flush=True)
    print(json.dumps({"planner": a.planner, "shard": a.shard, "attempts": n_run,
                      "accepted_per_cell": accepted}), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from enum import Enum

import numpy as np

from ..evaluation import bank as _bank
from ..certificate import duty as _duty
from ..certificate.escape_angle import (
    HULL_CONTACT_M,
    Vessel,
    cpa,
    place_encounter,
    place_threat,
)
from .cert_env import MAX_STEPS, CertifiedEscapeEnv, OwnShipPresence
from .evaluate_cert import _ARM_KW
from .reactive_bank import (
    COMPATIBLE_QUADRANTS,
    GATE_M,
    N_REFINE,
    PLACE_TRIES,
    PROGRESS_EVERY,
    RELAXED_ARM,
    SPEED_RANGE_MPS,
    T_MEET_RANGE_S,
    quadrant,
    reaim_ts2,
)

STRATUM = "fleet"
SITUATIONS = tuple(COMPATIBLE_QUADRANTS)
MAX_TARGETS = 4
REFERENCE_ARM = "compliant"
RULE_ARMS = ("compliant", "letter", "vo")
CHARACTERS = ("wide_berth", "compliant", "late")
MAX_SWITCHES = 2
SWITCH_STEPS = (6, 48)
NEIGHBOUR_DCPA_M = (1000.0, 3000.0)
NEIGHBOUR_TCPA_S = (0.0, 600.0)
NEIGHBOUR_RANGE_M = (1500.0, 7000.0)
NEIGHBOUR_TRIES = 500
SIM_DT_S = 10.0


class Verdict(Enum):

    REJECTED = "rejected"
    ACCEPTED = "accepted"
    HARD = "hard"
    TT_INDEPENDENT = "tt_independent"


ATTEMPT_FIELDS = ["cell", "attempt", "seed", "placed", "refine_iters", "sep_compliant",
                  "sep_letter", "sep_vo", "sep_relaxed", "trapped_by", "verdict", "wall_s"]
EXTRA_FIELDS = ["cell_n", "cell_k", "cell_situation", "quadrants", "plan", "n_switches",
                "refine_iters", "sep_compliant", "sep_letter", "sep_vo", "sep_relaxed",
                "trapped_by", "tt_min_sep", "tt_step", "tt_attribution"]


def cells():
    """List every cell of the bank grid: targets, colliding targets, situation.

    Returns:
        list: Cells (n, k, situation)
    """
    return [(n, k, s) for n in range(1, MAX_TARGETS + 1) for k in range(1, n + 1)
            for s in SITUATIONS]


def cell_key(cell):
    """Turn a cell into its string key.

    Args:
        cell (tuple): Bank cell

    Returns:
        str: Key n
    """
    return f"{cell[0]}|{cell[1]}|{cell[2]}"


def place_neighbour(rng, os_v, name):
    """Place a non colliding target near own vessel.

    Args:
        rng (Generator): Random generator
        os_v (Vessel): Own vessel
        name (str): Target name

    Returns:
        Vessel: Target, or None
    """
    for _ in range(NEIGHBOUR_TRIES):
        z = float(rng.uniform(*NEIGHBOUR_RANGE_M))
        lam = float(rng.uniform(-math.pi, math.pi))
        ts = Vessel(name, os_v.x + z * math.cos(lam), os_v.y + z * math.sin(lam),
                    float(rng.uniform(*SPEED_RANGE_MPS)), float(rng.uniform(-math.pi, math.pi)),
                    os_v.r_min)
        tcpa, dcpa, _ = cpa(os_v, ts)
        if (NEIGHBOUR_TCPA_S[0] <= tcpa <= NEIGHBOUR_TCPA_S[1]
                and NEIGHBOUR_DCPA_M[0] <= dcpa <= NEIGHBOUR_DCPA_M[1]):
            return ts
    return None


def place_fleet(rng, cell):
    """Place own vessel and the targets of a cell.

    Args:
        rng (Generator): Random generator
        cell (tuple): Bank cell

    Returns:
        tuple: Own vessel and targets, or None
    """
    n, k, sit = cell
    for _ in range(PLACE_TRIES):
        os_v, first, _placed = place_encounter(rng, [sit], t_range=T_MEET_RANGE_S)
        if not first:
            continue
        t1 = float(cpa(os_v, first[0])[0])
        fleet = [first[0]]
        for j in range(2, k + 1):
            ts, _tries = place_threat(rng, os_v, str(rng.choice(SITUATIONS)), name=f"TS{j}",
                                      t_range=(t1, T_MEET_RANGE_S[1]))
            if ts is None:
                break
            fleet.append(ts)
        if len(fleet) < k:
            continue
        for j in range(k + 1, n + 1):
            ts = place_neighbour(rng, os_v, f"TS{j}")
            if ts is None:
                break
            fleet.append(ts)
        if len(fleet) == n:
            return os_v, fleet
    return None


def draw_plan(rng, n):
    """Draw the behaviour switch plan of each target.

    Args:
        rng (Generator): Random generator
        n (int): Number of targets

    Returns:
        list: Segments per target
    """
    plan = []
    steps = np.arange(SWITCH_STEPS[0], SWITCH_STEPS[1] + 1)
    for _ in range(n):
        c = str(rng.choice(CHARACTERS))
        segs = [[0, c]]
        m = int(rng.integers(0, MAX_SWITCHES + 1))
        for step in sorted(rng.choice(steps, size=m, replace=False)):
            c = str(rng.choice([x for x in CHARACTERS if x != c]))
            segs.append([int(step), c])
        plan.append(segs)
    return plan


def _row(os_v, threats, plan, idx=0):
    """Build a bank row dict from vessels.

    Args:
        os_v (Vessel): Own vessel
        threats (list): Target vessels
        plan (list): Behaviour plan
        idx (int): Row index

    Returns:
        dict: Bank row
    """
    return dict(idx=str(idx), stratum=STRATUM, gate=GATE_M, os=os_v, threats=list(threats),
                plan=plan)


_ENVS = {}


def simulate(row, arm, seed, presence=OwnShipPresence.PRESENT, window=MAX_STEPS):
    """Run one arm on one row and record separation, track and attribution.

    Args:
        row (dict): Bank row
        arm (str): Arm name
        seed (int): Random seed
        presence (OwnShipPresence): Whether targets see own vessel
        window (int): Number of steps

    Returns:
        dict: Separation, track, attribution and its step
    """
    key = (arm, presence)
    env = _ENVS.get(key)
    if env is None:
        env = CertifiedEscapeEnv([row], gate=GATE_M, strata=(STRATUM,),
                                 score_after_handback=False, own_ship=presence, **_ARM_KW[arm])
        _ENVS[key] = env
    env.rows = [row]
    env.reset(seed=seed, options={"row": 0})
    track = [(env._os.x, env._os.y)]
    sep, tt, tt_step = float("inf"), float("inf"), None
    for t in range(1, window + 1):
        _obs, _r, _term, _trunc, info = env.step(None)
        sep = min(sep, float(info["min_separation"]))
        if env.ep_ts_cpa < tt:
            tt, tt_step = float(env.ep_ts_cpa), t
        track.append((env._os.x, env._os.y))
    return dict(sep=sep, track=track, tt=tt, tt_step=tt_step)


def refine(os_v, threats, k, ref, plan, seed, n_refine):
    """Re-aim the colliding targets until the reference arm is trapped.

    Args:
        os_v (Vessel): Own vessel
        threats (list): Target vessels
        k (int): Number of colliding targets
        ref (dict): Reference simulation
        plan (list): Behaviour plan
        seed (int): Random seed
        n_refine (int): Max refine rounds

    Returns:
        tuple: Targets, reference simulation and rounds used
    """
    rounds = 0
    for it in range(n_refine):
        if ref["sep"] < HULL_CONTACT_M:
            break
        new = list(threats)
        for j in range(1, k):
            ts = reaim_ts2(os_v, threats[j], ref["track"], SIM_DT_S)
            if ts is None:
                continue
            if (_duty.situation(os_v, ts) == _duty.situation(os_v, threats[j])
                    and quadrant(os_v, ts) == quadrant(os_v, threats[j])):
                new[j] = ts
        if all(a is b for a, b in zip(new, threats)):
            break
        threats, rounds = new, it + 1
        ref = simulate(_row(os_v, threats, plan), REFERENCE_ARM, seed)
    return threats, ref, rounds


def judge(n, seps):
    """Decide whether a row is kept from the rule arms it traps.

    Args:
        n (int): Number of targets
        seps (dict): Closest separation per arm

    Returns:
        tuple: Verdict and trapping arms
    """
    trapped = [a for a in RULE_ARMS if seps[a] < HULL_CONTACT_M]
    if n == 1:
        return Verdict.ACCEPTED, trapped
    if not trapped:
        return Verdict.REJECTED, trapped
    if seps[RELAXED_ARM] < HULL_CONTACT_M:
        return Verdict.HARD, trapped
    return Verdict.ACCEPTED, trapped


def screen_tt(verdict, attribution):
    """Mark a kept row whose contact does not depend on own vessel.

    Args:
        verdict (Verdict): Verdict
        attribution (str): Contact attribution

    Returns:
        Verdict: Updated verdict
    """
    if verdict in (Verdict.ACCEPTED, Verdict.HARD) and attribution == "os_independent":
        return Verdict.TT_INDEPENDENT
    return verdict


def attempt(cell, seed, n_refine=N_REFINE):
    """Place, refine, simulate and judge one row of a cell.

    Args:
        cell (tuple): Bank cell
        seed (int): Random seed
        n_refine (int): Max refine rounds

    Returns:
        tuple: Attempt record and results
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    rec = dict(cell=cell_key(cell), seed=seed, placed=0, refine_iters=0, trapped_by="",
               verdict=Verdict.REJECTED.value)
    placed = place_fleet(rng, cell)
    if placed is None:
        rec["wall_s"] = time.perf_counter() - t0
        return rec, {}
    os_v, threats = placed
    plan = draw_plan(rng, len(threats))
    rec["placed"] = 1
    out = {"random": dict(os=os_v, threats=list(threats), refine_iters=0)}
    n, k, _sit = cell

    ref = simulate(_row(os_v, threats, plan), REFERENCE_ARM, seed)
    threats, ref, rec["refine_iters"] = refine(os_v, threats, k, ref, plan, seed, n_refine)
    row = _row(os_v, threats, plan)
    seps = {REFERENCE_ARM: ref["sep"]}
    for arm in ("letter", "vo", RELAXED_ARM):
        seps[arm] = simulate(row, arm, seed)["sep"]
    verdict, trapped = judge(n, seps)
    rec.update(verdict=verdict.value, trapped_by="|".join(trapped),
               sep_compliant=seps["compliant"], sep_letter=seps["letter"], sep_vo=seps["vo"],
               sep_relaxed=seps[RELAXED_ARM])

    if verdict is not Verdict.REJECTED:
        attribution = ""
        if ref["tt"] < HULL_CONTACT_M:
            cf = simulate(row, REFERENCE_ARM, seed, presence=OwnShipPresence.ABSENT)["tt"]
            attribution = "os_independent" if cf < HULL_CONTACT_M else "os_induced"
        verdict = screen_tt(verdict, attribution)
        rec["verdict"] = verdict.value

    if verdict in (Verdict.ACCEPTED, Verdict.HARD):
        out[verdict] = dict(os=os_v, threats=threats, refine_iters=rec["refine_iters"],
                            tt_min_sep=ref["tt"], tt_step=ref["tt_step"],
                            tt_attribution=attribution, trapped_by=rec["trapped_by"],
                            **{f"sep_{a}": s for a, s in seps.items()})
    rec["wall_s"] = time.perf_counter() - t0
    out = {name: dict(r, plan=plan) for name, r in out.items()}
    return rec, out


def bank_record(idx, seed, cell, r):
    """Build the bank CSV record of a kept row.

    Args:
        idx (int): Row index
        seed (int): Random seed
        cell (tuple): Bank cell
        r (dict): Row results

    Returns:
        dict: CSV record
    """
    lab = _bank.label(r["os"], r["threats"], gate=GATE_M)
    lab["stratum"] = STRATUM
    return dict(idx=idx, seed=seed, **lab, cell_n=cell[0], cell_k=cell[1], cell_situation=cell[2],
                quadrants=json.dumps([quadrant(r["os"], t) for t in r["threats"]]),
                plan=json.dumps(r["plan"]),
                n_switches=json.dumps([len(segs) - 1 for segs in r["plan"]]),
                **{k: v for k, v in r.items() if k not in ("os", "threats", "plan")})


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


def _open_csv(path, fields):
    """Open a CSV in append mode and write the header if new.

    Args:
        path (str): CSV path
        fields (list): Column names

    Returns:
        tuple: File handle and writer
    """
    new = not os.path.exists(path)
    fh = open(path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    if new:
        w.writeheader()
    return fh, w


def main():
    """Build the fleet bank from the command line, by shard, resuming partial files."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--out-dir", default="fleet_bank")
    ap.add_argument("--per-cell", type=int, default=50, help="accepted rows wanted per cell")
    ap.add_argument("--max-attempts", type=int, default=3000, help="attempts per cell, cap")
    ap.add_argument("--refine", type=int, default=N_REFINE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--cells", default=None, help="comma list of cell indices, default all")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    grid = cells()
    wanted = range(len(grid)) if a.cells is None else [int(c) for c in a.cells.split(",")]
    wanted = [ci for ci in wanted if ci % a.shards == a.shard]

    att_path = os.path.join(a.out_dir, f"attempts_{a.shard}.csv")
    done = _load_attempts(att_path)
    seen = {(r["cell"], int(r["attempt"])) for r in done}
    accepted = {}
    for r in done:
        accepted[r["cell"]] = accepted.get(r["cell"], 0) + (r["verdict"] == Verdict.ACCEPTED.value)

    fields = _bank.FIELDS + EXTRA_FIELDS
    fa, wa = _open_csv(att_path, ATTEMPT_FIELDS)
    sinks = {name: _open_csv(os.path.join(a.out_dir, f"{prefix}_{a.shard}.csv"), fields)
             for name, prefix in (("random", "bank_random"), (Verdict.ACCEPTED, "bank"),
                                  (Verdict.HARD, "bank_hard"))}
    t0, n_run = time.time(), 0
    try:
        for ci in wanted:
            cell = grid[ci]
            key = cell_key(cell)
            for n in range(a.max_attempts):
                if accepted.get(key, 0) >= a.per_cell:
                    break
                if (key, n) in seen:
                    continue
                seed = a.seed + 1_000_000 * ci + n
                rec, rows = attempt(cell, seed, n_refine=a.refine)
                rec["attempt"] = n
                wa.writerow(rec)
                fa.flush()
                for name, r in rows.items():
                    fh, w = sinks[name]
                    w.writerow(bank_record(f"{STRATUM}_{ci}_{n}", seed, cell, r))
                    fh.flush()
                accepted[key] = accepted.get(key, 0) + (rec["verdict"] == Verdict.ACCEPTED.value)
                n_run += 1
                if n_run % PROGRESS_EVERY == 0:
                    print(f"  {n_run} attempts  {n_run / (time.time() - t0):.2f}/s  "
                          f"accepted {sum(accepted.values())}  cell {ci} {key}: "
                          f"{accepted.get(key, 0)}", flush=True)
    finally:
        fa.close()
        for fh, _w in sinks.values():
            fh.close()
    print(json.dumps({"shard": a.shard, "attempts": n_run, "accepted_per_cell": accepted}),
          flush=True)


if __name__ == "__main__":
    main()

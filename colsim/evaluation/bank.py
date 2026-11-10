from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from ..certificate import duty as _duty
from ..certificate import monitor as _monitor
from ..certificate.escape_angle import (
    SITUATIONS,
    Vessel,
    _rand_encounter,
    place_encounter,
)

STRATA = ("certified", "pre_certified", "free")

FIELDS = [
    "idx",
    "seed",
    "stratum",
    "arm",
    "t_first",
    "p_hat",
    "deontic_now",
    "geometric_now",
    "alpha_col",
    "alpha_phys",
    "lawful_width",
    "kappa",
    "n_threats",
    "gate",
    "horizon",
    "os",
    "threats",
    "situations",
]


def _vessel_dict(v):
    """Turn a Vessel into a dict for the CSV.

    Args:
        v (Vessel): Vessel

    Returns:
        dict: Vessel fields
    """
    return dict(name=v.name, x=v.x, y=v.y, v=v.v, theta=v.theta, r_min=v.r_min)


def _vessel(d):
    """Build a Vessel from a dict.

    Args:
        d (dict): Vessel fields

    Returns:
        Vessel: New vessel
    """
    return Vessel(d["name"], d["x"], d["y"], d["v"], d["theta"], d["r_min"])


def label(os_v, threats, *, gate, horizon=_monitor.DECISION_HORIZON_S, rng=None, **kw):
    """Label an encounter with the monitor prediction.

    Args:
        os_v (Vessel): Own vessel
        threats (list): Target vessels
        gate (object): Gate value
        horizon (float): Look ahead time, s
        rng (Generator): Random generator, or None
        **kw (dict): Extra monitor settings

    Returns:
        dict: Stratum, first breach time, measures and vessels
    """
    r = _monitor.predict(os_v, threats, gate=gate, horizon=horizon, rng=rng, **kw)
    now = r["now"]
    if r["fires_now"]:
        stratum = "certified"
    elif r["p_fire"] > 0.0:
        stratum = "pre_certified"
    else:
        stratum = "free"
    p_hat = next((p for p in r["p_hat"] if p is not None), None)
    return dict(
        stratum=stratum,
        arm=r["arms"][0] or "",
        t_first=r["t_first_min"] if r["t_first_min"] != float("inf") else "",
        p_hat="" if p_hat is None else p_hat,
        deontic_now=int(now["deontic"]),
        geometric_now=int(now["geometric"]),
        alpha_col=now["alpha_col"],
        alpha_phys=now["alpha_phys"],
        lawful_width=now["lawful_width"],
        kappa=now["kappa"],
        n_threats=len(threats),
        gate="" if gate is None else gate,
        horizon=horizon,
        os=json.dumps(_vessel_dict(os_v)),
        threats=json.dumps([_vessel_dict(t) for t in threats]),
        situations=json.dumps([_duty.situation(os_v, t) for t in threats]),
    )


def sample_row(
    idx,
    *,
    seed,
    n_threats,
    gate,
    horizon,
    on_collision=0.75,
    ts_speed_cap=False,
    os_theta=None,
    situations=None,
    **kw,
):
    """Draw and label one random encounter.

    Args:
        idx (int): Row index
        seed (int): Base seed
        n_threats (int): Number of targets
        gate (object): Gate value
        horizon (float): Look ahead time, s
        on_collision (float): Share of targets aimed at own vessel
        ts_speed_cap (bool): Whether to cap the target speed
        os_theta (float): Own heading, rad, or None for random
        situations (list): Situations to draw from, or None
        **kw (dict): Extra monitor settings

    Returns:
        dict: Labelled row, or None if no target was placed
    """
    rng = np.random.default_rng([seed, idx])
    if situations is None:
        os_v, threats = _rand_encounter(
            rng,
            n_threats=n_threats,
            on_collision=on_collision,
            ts_speed_cap=ts_speed_cap,
            os_theta=os_theta,
        )
    else:
        os_v, threats, _ = place_encounter(
            rng, situations, os_theta=os_theta, ts_speed_cap=ts_speed_cap
        )
        if not threats:
            return None
    row = label(os_v, threats, gate=gate, horizon=horizon, rng=rng, **kw)
    row.update(idx=idx, seed=seed)
    return row


def parse_threats(spec):
    """Read a fleet size spec like 3, 1-5 or 1,3,5.

    Args:
        spec (str): Fleet size spec

    Returns:
        tuple: Fleet sizes
    """
    spec = str(spec).strip()
    if "-" in spec:
        lo, hi = (int(p) for p in spec.split("-", 1))
        return tuple(range(lo, hi + 1))
    return tuple(int(p) for p in spec.split(",") if p)


def build_balanced(
    path,
    n,
    *,
    seed=42,
    n_threats=(1, 2, 3, 4, 5),
    gate=None,
    horizon=_monitor.DECISION_HORIZON_S,
    on_collision=0.75,
    strata=("certified", "pre_certified"),
    max_attempts=60,
    progress=None,
    ts_speed_cap=False,
    os_theta=None,
    situations=SITUATIONS,
):
    """Write a bank with the same number of useful rows per fleet size and situation.

    Args:
        path (str): Output CSV file
        n (int): Total rows to keep
        seed (int): Base seed
        n_threats (tuple): Fleet sizes
        gate (object): Gate value, or None
        horizon (float): Look ahead time, s
        on_collision (float): Share of targets aimed at own vessel
        strata (tuple): Strata to keep
        max_attempts (int): Attempt budget per kept row
        progress (callable): Progress callback, or None
        ts_speed_cap (bool): Whether to cap the target speed
        os_theta (float): Own heading, rad, or None for random
        situations (tuple): Situations to draw from

    Returns:
        int: Number of rows kept
    """
    sizes = tuple(int(s) for s in n_threats)
    cells = tuple(situations) if situations else (None,)
    target = max(int(n) // (len(sizes) * len(cells)), 1)
    budget = max_attempts * target
    idx = 0
    kept_total = 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for nt in sizes:
            for cell in cells:
                got = tried = 0
                while got < target and tried < budget:
                    rng = np.random.default_rng([seed, idx, 7])
                    sits = (
                        None
                        if cell is None
                        else (
                            [cell]
                            + [
                                SITUATIONS[int(k)]
                                for k in rng.integers(0, len(SITUATIONS), size=nt - 1)
                            ]
                        )
                    )
                    row = sample_row(
                        idx,
                        seed=seed,
                        n_threats=nt,
                        gate=gate,
                        horizon=horizon,
                        on_collision=on_collision,
                        ts_speed_cap=ts_speed_cap,
                        os_theta=os_theta,
                        situations=sits,
                    )
                    tried += 1
                    idx += 1
                    if row is not None and row["stratum"] in strata:
                        w.writerow(row)
                        fh.flush()
                        got += 1
                        kept_total += 1
                        if progress and got % progress == 0:
                            print(
                                f"  n_threats={nt} {cell or 'any'}: {got}/{target} kept, "
                                f"{tried} tried",
                                flush=True,
                            )
                print(
                    f"  n_threats={nt} {cell or 'any'}: kept {got}/{target} from {tried} "
                    f"attempts ({got / max(tried, 1):.1%} usable)"
                    + ("  <-- BUDGET EXHAUSTED" if got < target else ""),
                    flush=True,
                )
    return kept_total


def build(
    path,
    n,
    *,
    seed=42,
    n_threats=3,
    gate=None,
    horizon=_monitor.DECISION_HORIZON_S,
    on_collision=0.75,
    resume=True,
    progress=None,
    **kw,
):
    """Write a bank of random labelled rows, resuming a partial file.

    Args:
        path (str): Output CSV file
        n (int): Number of rows
        seed (int): Base seed
        n_threats (int): Number of targets
        gate (object): Gate value, or None
        horizon (float): Look ahead time, s
        on_collision (float): Share of targets aimed at own vessel
        resume (bool): Whether to continue an existing file
        progress (callable): Progress callback, or None
        **kw (dict): Extra settings for sample_row

    Returns:
        dict: Row count per stratum
    """
    done = 0
    if resume and os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline="", encoding="utf-8") as fh:
            done = sum(1 for _ in csv.DictReader(fh))
    mode = "a" if done else "w"
    with open(path, mode, newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if not done:
            w.writeheader()
        for i in range(done, n):
            row = sample_row(
                i,
                seed=seed,
                n_threats=n_threats,
                gate=gate,
                horizon=horizon,
                on_collision=on_collision,
                **kw,
            )
            if row is None:
                continue
            w.writerow(row)
            fh.flush()
            if progress and (i + 1) % progress == 0:
                print(f"  {i + 1}/{n}", flush=True)
    return counts(load(path))


def load(path, stratum=None):
    """Read a bank CSV into rows with Vessel objects.

    Args:
        path (str): Bank CSV file
        stratum (str): Keep only this stratum, or None

    Returns:
        list: Rows
    """
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if stratum is not None and r["stratum"] != stratum:
                continue
            r["os"] = _vessel(json.loads(r["os"]))
            r["threats"] = [_vessel(d) for d in json.loads(r["threats"])]
            for k in ("alpha_col", "alpha_phys", "lawful_width", "kappa"):
                r[k] = float(r[k])
            r["t_first"] = float(r["t_first"]) if r["t_first"] else float("inf")
            r["p_hat"] = float(r["p_hat"]) if r["p_hat"] else None
            if r.get("plan"):
                r["plan"] = json.loads(r["plan"])
            out.append(r)
    return out


def counts(rows):
    """Count the rows of each stratum.

    Args:
        rows (list): Rows

    Returns:
        dict: Row count per stratum
    """
    return {s: sum(1 for r in rows if r["stratum"] == s) for s in STRATA}


def main():
    """Build a bank from the command line."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--n", type=int, default=1000, help="total rows in the bank")
    ap.add_argument("--out", default="trap_bank.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--threats",
        default="3",
        help="fleet size: `3`, or a range `1-5`, or a list `1,3,5`. Anything but a "
        "single value implies --balance.",
    )
    ap.add_argument(
        "--balance",
        action="store_true",
        help="equal *usable* (non-free) rows per fleet size, with a per-size attempt "
        "budget. Implied by a multi-valued --threats. --n is the total kept.",
    )
    ap.add_argument(
        "--max-attempts",
        type=int,
        default=60,
        metavar="K",
        help="per-size attempt budget, as a multiple of the per-size target",
    )
    ap.add_argument(
        "--gate",
        type=float,
        default=None,
        help="Rule 7 risk gate in metres (ADR 0008). Omit for ungated.",
    )
    ap.add_argument("--horizon", type=float, default=_monitor.DECISION_HORIZON_S)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument(
        "--ts-speed-cap",
        action="store_true",
        help="draw each threat's speed in [5, v_os] instead of [5, 15]. Off "
        "reproduces every bank written before 2026-08-19.",
    )
    ap.add_argument(
        "--os-theta",
        type=float,
        default=None,
        metavar="RAD",
        help="fix the own-ship heading. Omit for the uniform draw.",
    )
    a = ap.parse_args()
    sizes = parse_threats(a.threats)
    if a.balance or len(sizes) > 1:
        kept = build_balanced(
            a.out,
            a.n,
            seed=a.seed,
            n_threats=sizes,
            gate=a.gate,
            horizon=a.horizon,
            max_attempts=a.max_attempts,
            progress=max(a.n // (20 * len(sizes)), 1),
            ts_speed_cap=a.ts_speed_cap,
            os_theta=a.os_theta,
        )
        print(
            f"bank: {a.out}  kept={kept}  gate={a.gate}  horizon={a.horizon:.0f} s  "
            f"threats={sizes}  (free rows not written)"
        )
        return
    c = build(
        a.out,
        a.n,
        seed=a.seed,
        n_threats=sizes[0],
        gate=a.gate,
        horizon=a.horizon,
        resume=not a.no_resume,
        progress=max(a.n // 20, 1),
        ts_speed_cap=a.ts_speed_cap,
        os_theta=a.os_theta,
    )
    total = sum(c.values()) or 1
    print(
        f"bank: {a.out}  n={total}  gate={a.gate}  horizon={a.horizon:.0f} s  "
        f"threats={sizes[0]}"
    )
    for s in STRATA:
        print(f"  {s:<14} {c[s]:6d}  {c[s] / total:6.1%}")


if __name__ == "__main__":
    main()

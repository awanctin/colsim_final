from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
PINK, VERMILLION, GREY = "#CC79A7", "#D55E00", "#666666"
CONTACT_M, HULL_M = 200.0, 150.0


def setup():
    """Set the shared matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "legend.frameon": False, "lines.linewidth": 1.6,
    })


def _stamp(ax, n, extra=""):
    """Write the sample size in the top left corner.

    Args:
        ax (Axes): Axes
        n (int): Sample size
        extra (str): Extra text
    """
    ax.text(0.01, 0.99, f"n = {n:,}{('  ·  ' + extra) if extra else ''}", transform=ax.transAxes,
            va="top", ha="left", fontsize=7, color=GREY)


def _threshold(ax, y=CONTACT_M):
    """Draw the contact threshold line.

    Args:
        ax (Axes): Axes
        y (float): Threshold, m
    """
    ax.axhline(y, color=VERMILLION, ls="--", lw=1.2, zorder=1)
    ax.text(0.995, y, f" {y:.0f} m — scored contact", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7, color=VERMILLION)


def _save(fig, path):
    """Save a figure as PNG and PDF, then close it.

    Args:
        fig (Figure): Figure
        path (str): Path without extension
    """
    tmp = path + ".tmp.png"
    fig.savefig(tmp)
    os.replace(tmp, path + ".png")
    fig.savefig(path + ".pdf")
    plt.close(fig)


def evals(night):
    """Read every evaluation record of a night run.

    Args:
        night (str): Night run directory

    Returns:
        dict: Records per evaluation name
    """
    out = {}
    for p in sorted(glob.glob(os.path.join(night, "evals", "*.json"))):
        try:
            out[os.path.basename(p)[:-5]] = json.load(open(p))["records"]
        except Exception:
            pass
    for p in sorted(glob.glob(os.path.join(night, "evals", "*.jsonl"))):
        k = os.path.basename(p)[:-6]
        if k in out:
            continue
        try:
            recs = [json.loads(l) for l in open(p) if l.strip()]
            if recs:
                out[k] = recs
        except Exception:
            pass
    return out


def runs(night):
    """Read the config and episodes of every training run of a night.

    Args:
        night (str): Night run directory

    Returns:
        dict: Config and episodes per run
    """
    out = {}
    for d in sorted(glob.glob(os.path.join(night, "runs", "*"))):
        if not os.path.isdir(d):
            continue
        cfg_p, ep_p = os.path.join(d, "config.json"), os.path.join(d, "episodes.csv")
        cfg = {}
        try:
            cfg = json.load(open(cfg_p))
        except Exception:
            pass
        done, n_ep = 0, 0
        try:
            rows = list(csv.DictReader(open(ep_p)))
            n_ep = len(rows)
            if rows:
                done = int(rows[-1]["timestep"])
        except Exception:
            pass
        out[os.path.basename(d)] = dict(
            dir=d, target=int(cfg.get("steps") or 0), done=done, episodes=n_ep,
            finished=os.path.exists(os.path.join(d, "model.zip")),
            mtime=os.path.getmtime(ep_p) if os.path.exists(ep_p) else 0.0)
    return out


def _below(recs, thr=CONTACT_M):
    """Count the records whose CPA is under a threshold.

    Args:
        recs (list): Evaluation records
        thr (float): Threshold, m

    Returns:
        int: Count
    """
    return sum(1 for r in recs if float(r["ep_cpa"]) < thr)


def fig_board(night, jobs, out):
    """Draw the job status board of a night run.

    Args:
        night (str): Night run directory
        jobs (dict): Job status
        out (str): Output path without extension
    """
    setup()
    rows = jobs.get("jobs", [])
    h = max(3.0, 0.42 * len(rows) + 2.2)
    fig, ax = plt.subplots(figsize=(13.5, h))
    ax.set_axis_off()
    now = time.time()
    dl = jobs.get("deadline", 0)
    ax.text(0, 1.0, f"COLSim overnight run   ·   {time.strftime('%Y-%m-%d %H:%M:%S')}   ·   "
                    f"hard stop {time.strftime('%H:%M', time.localtime(dl))} "
                    f"({max(0, (dl - now)) / 3600:.1f} h left)",
            transform=ax.transAxes, fontsize=12, va="bottom", weight="bold")
    y, dy = 0.93, 1.0 / max(len(rows) + 3, 8)
    for j in rows:
        frac = float(j.get("frac", 0.0))
        state = j.get("state", "queued")
        col = {"running": BLUE, "done": GREEN, "failed": VERMILLION,
               "skipped": GREY, "queued": "#CCCCCC"}.get(state, GREY)
        ax.text(0.0, y, j.get("name", "?")[:34], transform=ax.transAxes, fontsize=8.5,
                family="monospace", va="center")
        ax.add_patch(plt.Rectangle((0.30, y - 0.011), 0.30, 0.022, transform=ax.transAxes,
                                   facecolor="#EEEEEE", edgecolor="none"))
        ax.add_patch(plt.Rectangle((0.30, y - 0.011), 0.30 * min(frac, 1.0), 0.022,
                                   transform=ax.transAxes, facecolor=col, edgecolor="none"))
        ax.text(0.615, y, f"{100 * frac:5.1f}%  {state:<8}", transform=ax.transAxes,
                fontsize=8, family="monospace", va="center", color=col)
        ax.text(0.73, y, j.get("why", "")[:60], transform=ax.transAxes, fontsize=7.5,
                va="center", color=GREY, style="italic")
        y -= dy
    _save(fig, out)


def fig_paired(a_recs, b_recs, label_a, label_b, out, thr=CONTACT_M):
    """Plot the CPA of two arms paired by seed.

    Args:
        a_recs (list): Records of arm A
        b_recs (list): Records of arm B
        label_a (str): Arm A label
        label_b (str): Arm B label
        out (str): Output path without extension
        thr (float): Contact threshold, m

    Returns:
        bool: True if the figure was made
    """
    setup()
    A = {r["seed"]: r for r in a_recs}
    B = {r["seed"]: r for r in b_recs}
    keys = sorted(set(A) & set(B))
    if len(keys) < 25:
        return False
    ca = np.array([float(A[k]["ep_cpa"]) for k in keys])
    cb = np.array([float(B[k]["ep_cpa"]) for k in keys])
    fa, fb = ca < thr, cb < thr
    fixed, broke = int((fa & ~fb).sum()), int((~fa & fb).sum())
    try:
        from scipy.stats import binomtest
        p = binomtest(min(fixed, broke), fixed + broke, 0.5).pvalue if fixed + broke else 1.0
    except Exception:
        p = float("nan")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.4),
                                 gridspec_kw={"width_ratios": [2.1, 1]})
    moved = [i for i in range(len(keys)) if fa[i] != fb[i]]
    for i in moved:
        good = fa[i] and not fb[i]
        ax.plot([0, 1], [ca[i], cb[i]], color=(GREEN if good else VERMILLION),
                alpha=0.85, marker="o", ms=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels([label_a, label_b]); ax.set_xlim(-0.25, 1.25)
    ax.set_yscale("log"); ax.set_ylabel("episode closest approach (m, log)")
    _threshold(ax, thr)
    _stamp(ax, len(keys), f"{len(moved)} episodes changed side")
    ax.set_title(f"green = fixed ({fixed})   ·   red = broken ({broke})", fontsize=9)
    ax.text(0.99, 0.02, "↑ higher is better", transform=ax.transAxes, ha="right",
            fontsize=7, style="italic", color=GREY)

    cells = [[int((~fa & ~fb).sum()), broke], [fixed, int((fa & fb).sum())]]
    bx.imshow([[0, 1], [1, 0]], cmap="Greys", alpha=0.10)
    for r in range(2):
        for c in range(2):
            bx.text(c, r, f"{cells[r][c]}", ha="center", va="center", fontsize=15,
                    color=(VERMILLION if (r, c) == (0, 1) else
                           GREEN if (r, c) == (1, 0) else GREY))
    bx.set_xticks([0, 1]); bx.set_xticklabels([f"{label_b}\nok", f"{label_b}\nfailed"], fontsize=8)
    bx.set_yticks([0, 1]); bx.set_yticklabels([f"{label_a}\nok", f"{label_a}\nfailed"], fontsize=8)
    bx.grid(False)
    bx.set_title(f"McNemar p = {p:.3f}", fontsize=10,
                 color=(GREEN if p < 0.05 and fixed > broke else GREY))
    fig.suptitle(f"{label_b} vs {label_a} — episodes below {thr:.0f} m: "
                 f"{int(fa.sum())} → {int(fb.sum())} of {len(keys)}", fontsize=11)
    _save(fig, out)
    return True


def fig_rhold(ev, out, thr=CONTACT_M):
    """Plot contacts against the hold radius.

    Args:
        ev (dict): Records per evaluation name
        out (str): Output path without extension
        thr (float): Contact threshold, m

    Returns:
        bool: True if the figure was made
    """
    pts = []
    for k, recs in ev.items():
        if not k.startswith("rh"):
            continue
        try:
            v = float(k.split("_")[0][2:].replace("p", "."))
        except ValueError:
            continue
        pts.append((v, _below(recs, thr), len(recs)))
    if len(pts) < 2:
        return False
    pts.sort()
    setup()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    try:
        from .cert_env import reward_ordering_bound
        b = reward_ordering_bound(0.99, 100.0, 60)
        lo, hi = float(b["lo"]), float(b["hi"])
        ax.axvspan(lo, hi, color=GREEN, alpha=0.13, zorder=0)
        ax.text((lo + hi) / 2, 0.97, "admissible: return > timeout > contact",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=7.5, color=GREEN)
    except Exception:
        pass
    x = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    n = pts[0][2]
    frac = y / max(n, 1)
    err = 1.96 * np.sqrt(np.clip(frac * (1 - frac), 0, None) / max(n, 1)) * n
    ax.errorbar(x, y, yerr=err, marker="o", color=BLUE, capsize=3, zorder=3)
    ax.axvline(5.0, color=VERMILLION, ls=":", lw=1.4)
    ax.text(5.0, 0.98, " shipped r_hold = 5.0", transform=ax.get_xaxis_transform(),
            rotation=90, va="top", fontsize=7, color=VERMILLION)
    ax.set_xlabel("r_hold (per-step cost of holding emergency control)")
    ax.set_ylabel(f"episodes below {thr:.0f} m (of {n:,})")
    _stamp(ax, n, "95 % Wilson bars")
    ax.text(0.99, 0.98, "↓ lower is better", transform=ax.transAxes, ha="right", va="top",
            fontsize=7, style="italic", color=GREY)
    ax.set_title("r_hold sweep at γ = 0.99 — the green band is derived, not chosen")
    _save(fig, out)
    return True


def fig_frontier(night, out):
    """Plot the held out frontier of the training runs.

    Args:
        night (str): Night run directory
        out (str): Output path without extension

    Returns:
        bool: True if the figure was made
    """
    pts = []
    for d in sorted(glob.glob(os.path.join(night, "runs", "*rv*"))):
        ho = os.path.join(d, "heldout_episodes.csv")
        if not os.path.exists(ho):
            continue
        try:
            rows = [r for r in csv.DictReader(open(ho)) if r.get("arm", "policy") == "policy"]
            if len(rows) < 25:
                continue
            v = float(os.path.basename(d).split("rv")[-1].split("_")[0].replace("p", "."))
            broke = [float(r["rules_broken"] or 0) for r in rows]
            hb = [float(r["handback_step"]) for r in rows
                  if r["handback_step"] not in ("", "None")]
            bad = sum(1 for r in rows if float(r["ep_cpa"]) < CONTACT_M)
            pts.append((v, float(np.mean(broke)), float(np.median(hb)) if hb else math.nan,
                        bad, len(rows)))
        except Exception:
            continue
    if len(pts) < 2:
        return False
    pts.sort()
    setup()
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    x = [p[1] for p in pts]
    y = [p[2] for p in pts]
    ax.plot(x, y, color=GREY, lw=1.0, alpha=0.6, zorder=1)
    ax.scatter(x, y, s=[20 + 9 * p[3] for p in pts], c=BLUE, zorder=3,
               alpha=0.85, edgecolor="white", linewidth=0.8)
    for v, bx_, by_, bad, _n in pts:
        ax.annotate(f"$r_{{viol}}$={v:g}\n{bad} below 200 m", (bx_, by_),
                    textcoords="offset points", xytext=(10, 5), fontsize=7.5, color=GREY)
    ax.set_xlabel("mean COLREGs duties broken per episode")
    ax.set_ylabel("median steps to hand back")
    ax.set_title("The behaviour family: breaking fewer rules costs time in emergency control\n"
                 "marker area = episodes below 200 m")
    ax.annotate("fast, rule-breaking", (min(x), min(y)), textcoords="offset points",
                xytext=(0, -28), fontsize=8, style="italic", color=GREY)
    _stamp(ax, pts[0][4])
    _save(fig, out)
    return True


def fig_summary(ev, out, thr=CONTACT_M):
    """Plot the contact count of every evaluation.

    Args:
        ev (dict): Records per evaluation name
        out (str): Output path without extension
        thr (float): Contact threshold, m

    Returns:
        bool: True if the figure was made
    """
    items = [(k, _below(v, thr), len(v)) for k, v in sorted(ev.items()) if len(v) >= 100]
    if not items:
        return False
    setup()
    fig, ax = plt.subplots(figsize=(8.6, max(3.0, 0.36 * len(items) + 1.6)))
    names = [i[0] for i in items]
    vals = [1000.0 * i[1] / i[2] for i in items]
    cols = [GREY if "incumbent" in n else BLUE for n in names]
    ax.barh(range(len(items)), vals, color=cols, height=0.62)
    for i, (nm, b, n) in enumerate(items):
        ax.text(vals[i] + 0.4, i, f"{b} / {n}", va="center", fontsize=8, color=GREY)
    ax.set_yticks(range(len(items))); ax.set_yticklabels(names, fontsize=8, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel(f"episodes below {thr:.0f} m, per 1000")
    ax.axvline(7, color=GREEN, ls="--", lw=1.2)
    ax.text(7, -0.8, " 0.7 % — proven floor", color=GREEN, fontsize=7.5)
    ax.set_title("Every arm on one axis   ·   grey = incumbent   ·   ← lower is better")
    _save(fig, out)
    return True


def once(night, board_only=False):
    """Make every figure of a night run once.

    Args:
        night (str): Night run directory
        board_only (bool): Whether to draw only the board

    Returns:
        list: Figures made
    """
    figs = os.path.join(night, "figures")
    os.makedirs(figs, exist_ok=True)
    made = []
    try:
        jobs = json.load(open(os.path.join(night, "status.json")))
    except Exception:
        jobs = {}
    if jobs:
        fig_board(night, jobs, os.path.join(figs, "board")); made.append("board")
    if board_only:
        return made
    ev = evals(night)
    base = None
    for k in ("incumbent_s9999", "incumbent"):
        if k in ev:
            base = ev[k]
            break
    if base:
        for k, recs in ev.items():
            if k.startswith("incumbent"):
                continue
            if fig_paired(base, recs, "incumbent", k,
                          os.path.join(figs, f"paired_{k}")):
                made.append(f"paired_{k}")
    for fn, name in ((fig_rhold, "rhold"), (fig_summary, "summary")):
        try:
            if fn(ev, os.path.join(figs, name)):
                made.append(name)
        except Exception as e:
            print(f"  [figs] {name} skipped: {e}", flush=True)
    try:
        if fig_frontier(night, os.path.join(figs, "frontier")):
            made.append("frontier")
    except Exception as e:
        print(f"  [figs] frontier skipped: {e}", flush=True)
    print(f"  [figs] {time.strftime('%H:%M:%S')}  regenerated: "
          f"{', '.join(made) if made else 'nothing yet'}", flush=True)
    return made


def main():
    """Make the night figures from the command line, once or in a loop."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--night", required=True)
    ap.add_argument("--watch", type=int, default=0, metavar="SEC",
                    help="loop forever, regenerating every SEC seconds")
    ap.add_argument("--until", type=float, default=0.0, help="epoch to stop watching")
    ap.add_argument("--board-only", action="store_true",
                    help="redraw only the status board — cheap, so it can run on a fast loop "
                         "while the result figures stay on the slow one")
    a = ap.parse_args()
    if not a.watch:
        once(a.night, a.board_only)
        return
    while True:
        try:
            once(a.night, a.board_only)
        except Exception as e:
            print(f"  [figs] refresh failed, continuing: {e}", flush=True)
        if a.until and time.time() > a.until:
            return
        time.sleep(a.watch)


if __name__ == "__main__":
    main()

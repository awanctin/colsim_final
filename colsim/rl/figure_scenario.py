from __future__ import annotations

import argparse
import math
import os

import numpy as np

from ..evaluation import bank as _bank
from .cert_env import (
    HULL_CONTACT_M,
    MAX_STEPS,
    N_BINS,
    R_MIN_M,
    SIM_DT_S,
    SPEED_MAX_MPS,
    SPEED_MIN_MPS,
    CertifiedEscapeEnv,
    observe,
    seg_min,
)
from .train_cert import UNIFORM_MIX

C_FREE = "#009E73"
C_BLOCKED = "#D55E00"
HATCH_UNLAWFUL = "///"
C_OS = "#0072B2"
C_TS = "#CC79A7"
C_TAIL = "#56B4E9"
C_INK = "#222222"

HORIZON_S = 600.0


C_CLASS = {
    "compliant": "#CC79A7",
    "adversarial": "#D55E00",
    "breakdown": "#E69F00",
    "constant": "#999999",
}


def _roll(env, policy, seed):
    """Run one episode with a policy and record every track.

    Args:
        env (CertifiedEscapeEnv): Environment
        policy (object): Trained policy, or None
        seed (int): Random seed

    Returns:
        tuple: Start info, tracks, handback step, outcome and last info
    """
    obs, info0 = env.reset(seed=seed)
    tracks = [[(b.x, b.y, b.theta)] for b in env._boats]
    handback = None
    outcome = "truncated"
    while True:
        action = (np.array([0.0, 1.0], dtype=np.float32) if policy is None
                  else policy.predict(obs, deterministic=True)[0])
        obs, _r, term, trunc, info = env.step(action)
        for tr, b in zip(tracks, env._boats):
            tr.append((b.x, b.y, b.theta))
        if handback is None and info.get("handback_step") is not None:
            handback = len(tracks[0]) - 1
        if term or trunc:
            outcome = info.get("outcome", "truncated")
            break
    return info0, [np.asarray(t) for t in tracks], handback, outcome, info


def _min_sep(tracks):
    """Find the closest separation between own track and any target track.

    Args:
        tracks (list): Tracks, own first

    Returns:
        tuple: Separation (m), step and target index
    """
    best = (np.inf, 0, 1)
    own = tracks[0]
    for j, tr in enumerate(tracks[1:], start=1):
        n = min(len(own), len(tr))
        if n == 1:
            d = float(np.hypot(own[0, 0] - tr[0, 0], own[0, 1] - tr[0, 1]))
            best = min(best, (d, 0, j))
            continue
        for k in range(n - 1):
            d = seg_min([tuple(own[k]), tuple(tr[k])], [tuple(own[k + 1]), tuple(tr[k + 1])])
            if d < best[0]:
                best = (float(d), k, j)
    return best


def _ship(ax, x, y, theta, *, colour, length=HULL_CONTACT_M, label=None, z=5):
    """Draw a ship shape.

    Args:
        ax (Axes): Axes
        x (float): Position x, m
        y (float): Position y, m
        theta (float): Heading, rad
        colour (str): Colour
        length (float): Drawn length, m
        label (str): Legend label
        z (int): Draw order
    """
    w = length * 0.42
    pts = np.array([[length * 0.6, 0.0], [length * 0.15, w], [-length * 0.4, w],
                    [-length * 0.4, -w], [length * 0.15, -w]])
    rot = np.array([[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)]])
    pts = pts @ rot.T + np.array([x, y])
    ax.fill(pts[:, 0], pts[:, 1], color=colour, zorder=z, ec=C_INK, lw=0.6)
    if label:
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(10, 10),
                    fontsize=8, color=C_INK, zorder=z + 1,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))


def panel_geometry(ax, env, info0):
    """Draw the escape fan: reachable, lawful and blocked headings.

    Args:
        ax (Axes): Axes
        env (CertifiedEscapeEnv): Environment
        info0 (dict): Start info

    Returns:
        tuple: Own vessel, targets, max heading change, reach, physical and lawful widths
    """
    import matplotlib.patches as mp

    os_v, threats = env._vessels()
    obs, a_phys, a_col, k0 = observe(os_v, threats, gate=env.gate, n_bins=N_BINS)
    blocked, lawful = obs[:N_BINS], obs[N_BINS:2 * N_BINS]
    reach = os_v.v * HORIZON_S
    edges = np.linspace(-k0, k0, N_BINS + 1)

    for i in range(N_BINS):
        free, law = blocked[i] < 0.5, lawful[i] > 0.5
        t0 = math.degrees(os_v.theta + edges[i])
        t1 = math.degrees(os_v.theta + edges[i + 1])
        ax.add_patch(mp.Wedge((os_v.x, os_v.y), reach, t0, t1,
                              fc=C_FREE if free else C_BLOCKED, alpha=0.5,
                              hatch=None if law else HATCH_UNLAWFUL,
                              ec="white", lw=0.35, zorder=1))

    for r, lab in ((1000, "1 km"), (2000, "2 km"), (5000, "5 km")):
        if r < reach * 1.25:
            ax.add_patch(mp.Circle((os_v.x, os_v.y), r, fill=False, ec=C_INK,
                                   lw=0.5, ls=":", alpha=0.5, zorder=2))
            ax.annotate(lab, (os_v.x, os_v.y + r), fontsize=7, color=C_INK,
                        ha="center", va="bottom", alpha=0.8, zorder=3)
    ax.add_patch(mp.Circle((os_v.x, os_v.y), 500.0, fill=False, ec=C_OS, lw=1.0,
                           ls="--", zorder=3))
    ax.add_patch(mp.Circle((os_v.x, os_v.y), HULL_CONTACT_M, fc=C_BLOCKED, alpha=0.30,
                           ec=C_BLOCKED, lw=0.8, zorder=4))

    _ship(ax, os_v.x, os_v.y, os_v.theta, colour=C_OS, label="own ship")
    for n, t in enumerate(threats, start=1):
        ax.add_patch(mp.Circle((t.x, t.y), HULL_CONTACT_M, fc=C_TS, alpha=0.22,
                               ec=C_TS, lw=0.7, zorder=4))
        _ship(ax, t.x, t.y, t.theta, colour=C_TS, label=f"TS{n}")
        ax.arrow(t.x, t.y, math.cos(t.theta) * t.v * 120, math.sin(t.theta) * t.v * 120,
                 width=18, color=C_TS, alpha=0.85, zorder=5, length_includes_head=True)
    ax.arrow(os_v.x, os_v.y, math.cos(os_v.theta) * os_v.v * 120,
             math.sin(os_v.theta) * os_v.v * 120, width=18, color=C_OS, zorder=6,
             length_includes_head=True)

    ax.set_title(f"A — what the policy sees at t = 0   (bank row {info0['idx']}, "
                 f"{info0['stratum']})", fontsize=10, loc="left")
    return os_v, threats, k0, reach, a_phys, a_col


def panel_episode(ax, tracks, handback, outcome, k0, classes=None, contact_m=HULL_CONTACT_M):
    """Draw the tracks of an episode with the handback point.

    Args:
        ax (Axes): Axes
        tracks (list): Tracks, own first
        handback (int): Handback step, or None
        outcome (str): Episode outcome
        k0 (float): Max heading change, rad
        classes (list): Target behaviour classes
        contact_m (float): Contact distance, m
    """
    import matplotlib.patches as mp

    own = tracks[0]
    cut = handback if handback is not None else len(own)
    ax.plot(own[:cut, 0], own[:cut, 1], color=C_OS, lw=2.4, zorder=5,
            label="own ship: PPO (certificate holds control)")
    if handback is not None:
        ax.plot(own[cut - 1:, 0], own[cut - 1:, 1], color=C_TAIL, lw=2.4, zorder=5,
                label="own ship: RRT* compliant planner (after hand-back)")
        ax.plot(own[cut - 1, 0], own[cut - 1, 1], "o", ms=9, mfc="white", mec=C_INK,
                mew=1.4, zorder=7)
        ax.annotate(f"hand-back\nstep {cut - 1}", (own[cut - 1, 0], own[cut - 1, 1]),
                    textcoords="offset points", xytext=(12, -22), fontsize=8,
                    color=C_INK, zorder=8,
                    bbox=dict(fc="white", ec=C_INK, lw=0.5, alpha=0.9, pad=2))
    seen = set()
    for n, tr in enumerate(tracks[1:], start=1):
        cls = (classes or [])[n - 1] if classes and len(classes) >= n else None
        col = C_CLASS.get(cls, C_TS)
        lab = None
        if cls is None and "ts" not in seen:
            lab, _ = "target ships", seen.add("ts")
        elif cls is not None and cls not in seen:
            lab, _ = f"target ship: {cls}", seen.add(cls)
        ax.plot(tr[:, 0], tr[:, 1], color=col, lw=1.5, alpha=0.9, zorder=4, label=lab)
        _ship(ax, tr[-1, 0], tr[-1, 1], tr[-1, 2], colour=col, length=220, label=f"TS{n}")
    _ship(ax, own[0, 0], own[0, 1], own[0, 2], colour=C_OS, length=220, label="start")

    d, k, j = _min_sep(tracks)
    p, q = tracks[0][k], tracks[j][k]
    ax.plot([p[0], q[0]], [p[1], q[1]], color=C_BLOCKED, lw=1.4, ls="--", zorder=8)
    ax.add_patch(mp.Circle((p[0], p[1]), contact_m / 2, fc="none", ec=C_BLOCKED,
                           lw=0.9, zorder=8))
    ax.annotate(f"closest approach {d:,.0f} m\n(step {k}; contact below {contact_m:.0f} m)",
                (0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1])),
                textcoords="offset points", xytext=(12, 12), fontsize=8, color=C_BLOCKED,
                zorder=9, bbox=dict(fc="white", ec=C_BLOCKED, lw=0.5, alpha=0.9, pad=2))
    ax.set_title(f"B — the episode, {len(own) - 1} steps of {SIM_DT_S:.0f} s   "
                 f"→ outcome: {outcome}", fontsize=10, loc="left")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def _limits_box(ax, os_v, k0, reach):
    """Write the physical limits of the simulation in a box.

    Args:
        ax (Axes): Axes
        os_v (Vessel): Own vessel
        k0 (float): Max heading change, rad
        reach (float): Reach, m
    """
    txt = (
        "physical limits of the simulation\n"
        f"time step {SIM_DT_S:.0f} s · window {MAX_STEPS} steps = {MAX_STEPS * SIM_DT_S / 60:.0f} min\n"
        f"speed {SPEED_MIN_MPS:.0f}–{SPEED_MAX_MPS:.1f} m/s "
        f"({SPEED_MIN_MPS * 1.944:.0f}–{SPEED_MAX_MPS * 1.944:.0f} kn); now {os_v.v:.1f} m/s\n"
        f"turn radius R_min = {R_MIN_M:.0f} m → yaw ≤ v/R_min\n"
        f"heading fan ±{math.degrees(k0):.1f}° over {N_BINS} bins "
        f"of {2 * math.degrees(k0) / N_BINS:.2f}°\n"
        f"fan reach {reach / 1000:.1f} km in {HORIZON_S / 60:.0f} min\n"
        f"contact below {HULL_CONTACT_M:.0f} m (two {HULL_CONTACT_M / 2:.0f} m hulls)\n"
        f"clear of traffic at t = 0 within 500 m"
    )
    ax.text(0.015, 0.015, txt, transform=ax.transAxes, fontsize=7.6, va="bottom",
            ha="left", color=C_INK, family="monospace", zorder=10,
            bbox=dict(fc="white", ec=C_INK, lw=0.6, alpha=0.92, pad=4))


def _legend_fan(ax):
    """Draw the legend of the escape fan.

    Args:
        ax (Axes): Axes
    """
    import matplotlib.patches as mp
    handles = [
        mp.Patch(fc=C_FREE, alpha=0.5, ec="white", label="reachable and lawful — the escape"),
        mp.Patch(fc=C_FREE, alpha=0.5, ec="white", hatch=HATCH_UNLAWFUL,
                 label="reachable, but COLREGs forbids it"),
        mp.Patch(fc=C_BLOCKED, alpha=0.5, ec="white", label="lawful, but a threat blocks it"),
        mp.Patch(fc=C_BLOCKED, alpha=0.5, ec="white", hatch=HATCH_UNLAWFUL,
                 label="blocked and unlawful"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9,
              title="heading bins  (hatched = unlawful)", title_fontsize=8)


def _inset_hull(ax, os_v, threats):
    """Draw an inset zoom on own hull and the closest targets.

    Args:
        ax (Axes): Axes
        os_v (Vessel): Own vessel
        threats (list): Target vessels
    """
    import matplotlib.patches as mp
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    ins = inset_axes(ax, width="30%", height="30%", loc="lower right", borderpad=1.1)
    ins.add_patch(mp.Circle((os_v.x, os_v.y), 500.0, fill=False, ec=C_OS, lw=1.2, ls="--"))
    ins.add_patch(mp.Circle((os_v.x, os_v.y), HULL_CONTACT_M, fc=C_BLOCKED, alpha=0.3,
                            ec=C_BLOCKED, lw=1.0))
    _ship(ins, os_v.x, os_v.y, os_v.theta, colour=C_OS, length=150)
    for t in threats:
        _ship(ins, t.x, t.y, t.theta, colour=C_TS, length=150)
    ins.annotate(f"{HULL_CONTACT_M:.0f} m\ncontact", (os_v.x, os_v.y - HULL_CONTACT_M),
                 fontsize=6.5, ha="center", va="top", color=C_BLOCKED)
    ins.annotate("500 m clear at t = 0", (os_v.x, os_v.y + 520), fontsize=6.5,
                 ha="center", va="bottom", color=C_OS)
    ins.set_xlim(os_v.x - 700, os_v.x + 700)
    ins.set_ylim(os_v.y - 700, os_v.y + 700)
    ins.set_aspect("equal")
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_title("zoom: 1.4 km across", fontsize=7, pad=2)
    for s in ins.spines.values():
        s.set_edgecolor(C_INK)
        s.set_linewidth(0.8)


def _square(ax, pad=1.12):
    """Make the axes square around their content.

    Args:
        ax (Axes): Axes
        pad (float): Scale factor of the half width
    """
    ax.set_aspect("equal", adjustable="datalim")
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    h = 0.5 * max(x1 - x0, y1 - y0) * pad
    ax.set_xlim(cx - h, cx + h)
    ax.set_ylim(cy - h, cy + h)
    ax.set_xlabel("east (m)", fontsize=8)
    ax.set_ylabel("north (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15, lw=0.5)


def build(bank, gate, *, model=None, seed=3, out="figures/scenario", title=None, **env_kw):
    """Build the scenario figure for one bank row and save it.

    Args:
        bank (str): Bank CSV path
        gate (float): Duty gate, m
        model (str): Policy path, or None
        seed (int): Random seed
        out (str): Output path
        title (str): Figure title
        **env_kw (dict): Extra environment knobs

    Returns:
        str: Output path
    """
    import matplotlib
    matplotlib.use("Agg")
    cell = dict(ts_classes=UNIFORM_MIX, seed=9999, alpha_col_release=1.0, r_hold=5.0)
    cell.update(env_kw)
    import matplotlib.pyplot as plt

    rows = list(_bank.load(bank))
    policy = None
    if model:
        from .policy_io import load_policy
        policy = load_policy(model)

    env = CertifiedEscapeEnv(rows, gate=gate, **cell)
    env.reset(seed=seed)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6))
    os_v, threats, k0, reach, a_phys, a_col = panel_geometry(axes[0], env, env._row)
    _legend_fan(axes[0])
    _limits_box(axes[0], os_v, k0, reach)
    _square(axes[0])
    _inset_hull(axes[0], os_v, threats)

    env2 = CertifiedEscapeEnv(rows, gate=gate, **cell)
    info0, tracks, handback, outcome, _info = _roll(env2, policy, seed)
    panel_episode(axes[1], tracks, handback, outcome, k0, classes=list(env2._classes),
                  contact_m=float(env_kw.get('contact_m', HULL_CONTACT_M)))
    _square(axes[1])

    head = title or ("COLSim certified-escape environment — "
                     f"row {info0['idx']} ({info0['stratum']}), "
                     f"α_phys {a_phys:.2f}, α_col {a_col:.2f}")
    fig.suptitle(head, fontsize=12, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    print(f"wrote {out}.pdf and {out}.png   "
          f"(row {info0['idx']}, outcome {outcome}, "
          f"min separation {_min_sep(tracks)[0]:,.0f} m)")
    return out


def _selfcheck(bank, gate):
    """Check the drawn quantities against the observation on bank rows.

    Args:
        bank (str): Bank CSV path
        gate (float): Duty gate, m
    """
    rows = list(_bank.load(bank))[:80]
    env = CertifiedEscapeEnv(rows, gate=gate, ts_classes=UNIFORM_MIX, seed=9999)
    env.reset(seed=3)
    os_v, threats = env._vessels()
    obs, a_phys, a_col, k0 = observe(os_v, threats, gate=gate, n_bins=N_BINS)
    assert len(obs) == 2 * N_BINS + 17, len(obs)
    assert 0.0 < k0 <= 0.55, k0
    blocked, lawful = obs[:N_BINS], obs[N_BINS:2 * N_BINS]
    assert set(np.unique(blocked)) <= {0.0, 1.0} and set(np.unique(lawful)) <= {0.0, 1.0}
    print(f"  PASS  fan is {2 * math.degrees(k0):.1f}° over {N_BINS} bins, "
          f"{int(blocked.sum())} blocked, {int(lawful.sum())} lawful")

    _i, tracks, _hb, outcome, _info = _roll(env, None, 3)
    d, k, j = _min_sep(tracks)
    assert len(tracks) >= 2 and all(len(t) == len(tracks[0]) for t in tracks)
    assert (d < HULL_CONTACT_M) == (outcome == "contact"), (d, outcome)
    print(f"  PASS  tracks aligned, min separation {d:,.0f} m at step {k} vs TS{j}, "
          f"outcome {outcome}")
    print("\n=== figure_scenario: ALL PASS ===")


def main():
    """Build the figure from the command line.

    Returns:
        int: Exit code from the self check, or None
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None,
                    help="training run directory: its config.json sets every environment knob "
                         "(release action, speed action, veto margin, reward cell); the flags "
                         "below override it")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--gate", type=float, required=True)
    ap.add_argument("--model", default=None, help="SB3 .zip; omitted = hold-course null arm")
    ap.add_argument("--seed", type=int, default=3,
                    help="reset seed: picks the bank row and the target-ship classes")
    ap.add_argument("--out", default="figures/scenario", help="path without extension")
    ap.add_argument("--title", default=None)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--bin-fields", action="store_true",
                    help="133-dim ship-count-free observation (must match the policy)")
    ap.add_argument("--contact-m", type=float, default=None,
                    help="separation that terminates in contact; empty = 150 (hull)")
    ap.add_argument("--release-lookahead", type=int, default=None,
                    help="steps of compliant tail checked before a hand-back is permitted")
    ap.add_argument("--eval-stack", action="store_true",
                    help="score_after_handback=False, the configuration evalpol.py runs, so the "
                         "picture is the evaluation episode rather than the training one")
    a = ap.parse_args()
    if a.selfcheck:
        return _selfcheck(a.bank, a.gate)
    env_kw = {}
    if a.run:
        from . import episode_figs
        env_kw = episode_figs.env_kwargs(episode_figs.load_config(a.run))
        env_kw.pop("gate", None)
    if a.bin_fields:
        env_kw["bin_fields"] = True
    if a.contact_m is not None:
        env_kw["contact_m"] = a.contact_m
    if a.release_lookahead is not None:
        env_kw["release_lookahead"] = a.release_lookahead
    if a.eval_stack:
        env_kw["score_after_handback"] = False
    build(a.bank, a.gate, model=a.model, seed=a.seed, out=a.out, title=a.title, **env_kw)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..evaluation import bank as _bank
from .cert_env import (
    N_BINS,
    R_CONTACT,
    R_RETURN,
    REWARD_SCALE,
    SPEED_MAX_MPS,
    SPEED_MIN_MPS,
    CertifiedEscapeEnv,
)
from .figure_scenario import C_CLASS, C_OS, C_TS
from .figure_scenario import _ship, _square, panel_episode
from .night_figs import BLUE, GREEN, GREY, ORANGE, PINK, VERMILLION
from .night_figs import _save, setup
from .train_cert import UNIFORM_MIX, split_bank

ENV_KEYS = (
    "gate", "gamma", "contact_m", "bin_fields", "tubes", "n_track", "margin_steps",
    "min_hold_steps", "alpha_col_release", "release_lookahead", "veto_margin", "release_action",
    "action_shield", "r_hold", "r_timeout", "r_violation", "tail_steps", "max_steps",
    "rule17b_relax", "horizon", "hard_frac", "speed_control",
    "speed_action", "behaviour_obs",
)
CFG_KEYS = ("r_hold", "r_timeout", "r_violation", "gamma", "contact_m", "release_lookahead")

OUTCOME_COLOUR = {
    "return": GREEN,
    "timeout": BLUE,
    "contact": VERMILLION,
    "contact_after_handback": PINK,
}
DIM_COLOUR = (BLUE, ORANGE, GREEN)
_DIM_YAW = "a0 — yaw command (x max rate)"
_DIM_SPEED = "speed command"
_DIM_RELEASE = "release ask"


def dim_labels(n_dims, speed_action=None):
    """Name each action dimension.

    Args:
        n_dims (int): Number of action dimensions
        speed_action (bool): Whether the policy sets speed, or None to guess

    Returns:
        tuple: Labels
    """
    if speed_action is None:
        speed_action = n_dims >= 3
    out = [_DIM_YAW]
    if speed_action:
        out.append(f"a{len(out)} — {_DIM_SPEED}")
    if len(out) < n_dims:
        out.append(f"a{len(out)} — {_DIM_RELEASE}")
    return tuple(out)


DIM_LABEL = dim_labels(3, speed_action=True)

SATURATED = 0.95
ROLLOUT_SEED = 0


def load_config(run_dir):
    """Read the config of a training run.

    Args:
        run_dir (str): Run directory

    Returns:
        dict: Config
    """
    with open(os.path.join(run_dir, "config.json")) as fh:
        return json.load(fh)


def env_kwargs(cfg):
    """Build the environment arguments from a run config.

    Args:
        cfg (dict): Run config

    Returns:
        dict: Environment arguments
    """
    kw = {k: cfg[k] for k in ENV_KEYS if cfg.get(k) is not None}
    kw["ts_classes"] = cfg.get("mix") or UNIFORM_MIX
    if cfg.get("obs_dim"):
        kw["bin_fields"] = int(cfg["obs_dim"]) == 4 * N_BINS + 5
    return kw


def cfg_from_env(env, **extra):
    """Build a config dict from a live environment.

    Args:
        env (CertifiedEscapeEnv): Environment
        **extra (dict): Extra fields

    Returns:
        dict: Config
    """
    cfg = {k: getattr(env, k) for k in CFG_KEYS}
    cfg["obs_dim"] = int(env.observation_space.shape[0])
    cfg.update(extra)
    return cfg


def population(cfg, split="heldout"):
    """Get the train or held out rows of a run.

    Args:
        cfg (dict): Run config
        split (str): train or heldout

    Returns:
        list: Bank rows
    """
    keep = _bank.STRATA if cfg.get("include_free") else ("certified", "pre_certified")
    rows = [r for r in _bank.load(cfg["bank"]) if r["stratum"] in keep]
    train_rows, held = split_bank(rows, float(cfg.get("holdout", 0.2)))
    return train_rows if split == "train" else held


def roll_episode(env, policy, seed, *, deterministic=True, index=0):
    """Run one episode with a policy and record tracks, actions and rewards.

    Args:
        env (CertifiedEscapeEnv): Environment
        policy (object): Trained policy
        seed (int): Random seed
        deterministic (bool): Whether to take the mean action
        index (int): Episode index

    Returns:
        dict: Episode record
    """
    obs, _info0 = env.reset(seed=seed)
    tracks = [[(b.x, b.y, b.theta)] for b in env._boats]
    acts, rews, broken, info = [], [], [], {}
    handback = None
    while True:
        action, _ = policy.predict(obs, deterministic=deterministic)
        obs, r, term, trunc, info = env.step(action)
        for tr, b in zip(tracks, env._boats):
            tr.append((b.x, b.y, b.theta))
        acts.append(np.asarray(action, dtype=float).ravel())
        rews.append(float(r))
        broken.append(float(info.get("rules_broken", 0.0)))
        if handback is None and info.get("handback_step") is not None:
            handback = len(tracks[0]) - 1
        if term or trunc:
            break
    return {
        "episode": int(index),
        "seed": int(seed),
        "actions": np.asarray(acts),
        "rewards": np.asarray(rews),
        "broken": np.asarray(broken),
        "tracks": [np.asarray(t) for t in tracks],
        "classes": list(getattr(env, "_classes", [])),
        "handback": handback,
        "outcome": str(info.get("outcome", "unknown")),
        "ep_return": float(np.sum(rews)),
        "ep_len": len(rews),
        "ep_cpa": float(info.get("ep_cpa", float("nan"))),
        "min_separation": float(info.get("min_separation", float("nan"))),
    }


def rollout(env, policy, n_episodes, *, seed=ROLLOUT_SEED, deterministic=True, progress=None):
    """Run several episodes with a policy.

    Args:
        env (CertifiedEscapeEnv): Environment
        policy (object): Trained policy
        n_episodes (int): Number of episodes
        seed (int): First seed
        deterministic (bool): Whether to take the mean action
        progress (str): Progress bar label, or None

    Returns:
        list: Episode records
    """
    it = range(n_episodes)
    if progress:
        from tqdm import tqdm
        it = tqdm(it, desc=progress, unit="ep", leave=True)
    return [roll_episode(env, policy, seed + i, deterministic=deterministic, index=i)
            for i in it]


def decompose(ep, *, r_hold, r_violation, r_timeout):
    """Split the rewards of an episode into shaping, hold, violation and terminal parts.

    Args:
        ep (dict): Episode record
        r_hold (float): Per step hold cost
        r_violation (float): Per rule violation cost
        r_timeout (float): Survival reward

    Returns:
        dict: Reward parts per step
    """
    raw = ep["rewards"] / REWARD_SCALE
    terminal = np.zeros_like(raw)
    if ep["outcome"] in ("contact", "contact_after_handback"):
        terminal[-1] = R_CONTACT
    elif ep["outcome"] == "return":
        terminal[-1] = R_RETURN
    elif ep["outcome"] == "timeout":
        terminal[-1] = float(r_timeout)
    hold = np.full_like(raw, -float(r_hold))
    violation = -float(r_violation) * ep["broken"]
    shaping = raw - terminal - hold - violation
    return {k: v * REWARD_SCALE for k, v in
            (("shaping", shaping), ("hold", hold), ("violation", violation),
             ("terminal", terminal))}


def decompose_all(eps, cfg):
    """Split the rewards of every episode.

    Args:
        eps (list): Episode records
        cfg (dict): Run config

    Returns:
        list: Reward parts per episode
    """
    return [decompose(e, r_hold=cfg.get("r_hold", 5.0),
                      r_violation=cfg.get("r_violation", 0.0),
                      r_timeout=cfg.get("r_timeout", 100.0)) for e in eps]


def spread_stats(eps):
    """Compute the spread of actions between and within episodes.

    Args:
        eps (list): Episode records

    Returns:
        dict: Spread statistics
    """
    per_step = [e["actions"] for e in eps]
    allacts = np.concatenate(per_step)
    between = np.asarray([a.mean(axis=0) for a in per_step])
    within = np.asarray([a.std(axis=0) for a in per_step])
    return {
        "n_episodes": len(eps),
        "n_steps": int(len(allacts)),
        "mean": allacts.mean(axis=0).tolist(),
        "std_overall": allacts.std(axis=0).tolist(),
        "std_between_episodes": between.std(axis=0).tolist(),
        "std_within_episode": within.mean(axis=0).tolist(),
        "saturated_frac": (np.abs(allacts) > SATURATED).mean(axis=0).tolist(),
        "unique_rounded": [int(len(np.unique(np.round(allacts[:, d], 2))))
                           for d in range(allacts.shape[1])],
    }


def reward_stats(eps, parts):
    """Compute return means and the share of each reward part.

    Args:
        eps (list): Episode records
        parts (list): Reward parts per episode

    Returns:
        dict: Reward statistics
    """
    rets = np.asarray([e["ep_return"] for e in eps], dtype=float)
    total_abs = float(sum(np.abs(e["rewards"]).sum() for e in eps))
    term_abs = float(sum(np.abs(p["terminal"]).sum() for p in parts))
    by_outcome = {}
    for e in eps:
        by_outcome.setdefault(e["outcome"], []).append(e["ep_return"])
    return {
        "ep_return_mean": float(rets.mean()),
        "ep_return_std": float(rets.std()),
        "ep_return_min": float(rets.min()),
        "ep_return_max": float(rets.max()),
        "ep_return_cv": float(rets.std() / abs(rets.mean())) if rets.mean() else float("nan"),
        "terminal_share_of_abs_reward": term_abs / total_abs if total_abs else float("nan"),
        "outcomes": {k: {"n": len(v), "mean": float(np.mean(v)), "std": float(np.std(v))}
                     for k, v in sorted(by_outcome.items())},
    }


def _stamp(ax, text):
    """Write a note in the top left corner.

    Args:
        ax (Axes): Axes
        text (str): Text
    """
    ax.text(0.01, 0.985, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=7, color=GREY)


def _outcome_legend(ax, eps, loc="lower right"):
    """Add a legend of the outcomes present.

    Args:
        ax (Axes): Axes
        eps (list): Episode records
        loc (str): Legend position
    """
    seen = [o for o in OUTCOME_COLOUR if any(e["outcome"] == o for e in eps)]
    for o in seen:
        ax.plot([], [], color=OUTCOME_COLOUR[o], lw=1.4, label=o)
    if seen:
        ax.legend(fontsize=7, loc=loc, ncol=2)


def _title(cfg, mode, n):
    """Build the figure title from the run config.

    Args:
        cfg (dict): Run config
        mode (str): deterministic or sampled
        n (int): Number of episodes

    Returns:
        str: Title
    """
    return (f"{cfg.get('launched', 'run')} — {n} episodes, {mode} policy, "
            f"gamma {cfg.get('gamma')}, contact {cfg.get('contact_m')} m, "
            f"release lookahead {cfg.get('release_lookahead')}")


def _relative(tracks, k):
    """Express every track relative to own vessel at step k, own heading up.

    Args:
        tracks (list): Tracks, own first
        k (int): Step

    Returns:
        list: Relative positions
    """
    own = tracks[0][min(k, len(tracks[0]) - 1)]
    a = math.pi / 2 - own[2]
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for tr in tracks[1:]:
        p = tr[min(k, len(tr) - 1)]
        dx, dy = p[0] - own[0], p[1] - own[1]
        out.append((dx * ca - dy * sa, dx * sa + dy * ca, p[2] + a))
    return out


def _equal(ax):
    """Make the axes equal with a light grid.

    Args:
        ax (Axes): Axes
    """
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15, lw=0.5)
    ax.tick_params(labelsize=7)


def _rings(ax, radii, labels):
    """Draw range rings with labels.

    Args:
        ax (Axes): Axes
        radii (list): Radii, m
        labels (list): Labels
    """
    import matplotlib.patches as mp
    for i, (r, lab) in enumerate(zip(radii, labels)):
        ang = math.radians(-40.0 - 25.0 * i)
        ax.add_patch(mp.Circle((0, 0), r, fc="none", ec=GREY, ls=":", lw=0.9, zorder=1))
        ax.annotate(lab, (r * math.cos(ang), r * math.sin(ang)), textcoords="offset points",
                    xytext=(5, -6), fontsize=7, color=GREY, zorder=2)


def fig_actions(eps, cfg, *, deterministic=True):
    """Plot the action distribution of every dimension.

    Args:
        eps (list): Episode records
        cfg (dict): Run config
        deterministic (bool): Whether actions were the mean

    Returns:
        Figure: Figure
    """
    setup()
    dim = eps[0]["actions"].shape[1]
    labels = dim_labels(dim, cfg.get("speed_action"))
    fig, axes = plt.subplots(1, dim + 1, figsize=(5.6 * (dim + 1), 5.2))
    mode = "deterministic" if deterministic else "sampled"

    for d in range(dim):
        a = axes[d]
        a.axhspan(SATURATED, 1.0, color=GREY, alpha=0.12, zorder=0)
        a.axhspan(-1.0, -SATURATED, color=GREY, alpha=0.12, zorder=0)
        for e in eps:
            a.plot(np.arange(1, e["ep_len"] + 1), e["actions"][:, d], lw=1.0, alpha=0.55,
                   color=OUTCOME_COLOUR.get(e["outcome"], GREY))
        a.set_ylim(-1.08, 1.08)
        a.set_xlabel("step")
        a.set_ylabel(labels[d] if d < len(labels) else f"a{d}")
        a.set_title(f"{chr(65 + d)} — command {d} per episode ({mode})", fontsize=10, loc="left")
        vals = np.concatenate([e["actions"][:, d] for e in eps])
        _stamp(a, f"saturated |a| > {SATURATED}: {(np.abs(vals) > SATURATED).mean() * 100:.0f} %"
                  f"   ·   distinct values (2 dp): {len(np.unique(np.round(vals, 2)))}"
                  f"   ·   sd {vals.std():.3f}")
        if d == 0:
            _outcome_legend(a, eps)
        if d == 1:
            a.text(0.99, 0.02, f"-1 = {SPEED_MIN_MPS:.1f} m/s   +1 = {SPEED_MAX_MPS:.1f} m/s",
                   transform=a.transAxes, ha="right", va="bottom", fontsize=7, color=GREY)

    a = axes[dim]
    bins = np.linspace(-1.0, 1.0, 41)
    for d in range(dim):
        vals = np.concatenate([e["actions"][:, d] for e in eps])
        a.hist(vals, bins=bins, histtype="step", density=True, lw=1.6,
               color=DIM_COLOUR[d % len(DIM_COLOUR)], label=f"a{d}")
    a.axvspan(SATURATED, 1.0, color=GREY, alpha=0.12, zorder=0)
    a.axvspan(-1.0, -SATURATED, color=GREY, alpha=0.12, zorder=0)
    a.set_xlabel("command value")
    a.set_ylabel("density over all steps")
    a.set_title(f"{chr(65 + dim)} — repartition of the commands", fontsize=10, loc="left")
    a.legend(fontsize=7)
    _stamp(a, f"{sum(e['ep_len'] for e in eps):,} steps over {len(eps)} episodes")

    fig.suptitle(_title(cfg, mode, len(eps)), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def fig_reward(eps, cfg, *, deterministic=True):
    """Plot the reward parts over the episode.

    Args:
        eps (list): Episode records
        cfg (dict): Run config
        deterministic (bool): Whether actions were the mean

    Returns:
        Figure: Figure
    """
    setup()
    parts = decompose_all(eps, cfg)
    stats = reward_stats(eps, parts)
    mode = "deterministic" if deterministic else "sampled"
    fig, (a_step, a_hist, a_len) = plt.subplots(1, 3, figsize=(16.0, 5.2))

    trun = max(1, max(e["ep_len"] - 1 for e in eps))
    def _mean_series(key):
        """Compute the mean and spread of one reward part per step.

        Args:
            key (str): Reward part name

        Returns:
            tuple: Mean and standard deviation per step
        """
        grid = np.full((len(eps), trun), np.nan)
        for i, (e, p) in enumerate(zip(eps, parts)):
            n = min(e["ep_len"] - 1, trun)
            if n > 0:
                grid[i, :n] = p[key][:n] if key else e["rewards"][:n]
        return np.nanmean(grid, axis=0), np.nanstd(grid, axis=0)

    steps = np.arange(1, trun + 1)
    total, sd = _mean_series(None)
    a_step.fill_between(steps, total - sd, total + sd, color=GREY, alpha=0.20, lw=0)
    a_step.plot(steps, total, color=GREY, lw=2.0, label="step reward (mean +- 1 sd)")
    for key, colour in (("shaping", BLUE), ("hold", ORANGE), ("violation", PINK)):
        m, _sd = _mean_series(key)
        if np.nanmax(np.abs(m)) > 0:
            a_step.plot(steps, m, color=colour, lw=1.3, label=key)
    a_step.axhline(0.0, color=GREY, lw=0.8, ls=":")
    a_step.set_xlabel("step   (terminal step excluded)")
    a_step.set_ylabel("reward per step")
    a_step.set_title("A — where the reward comes from", fontsize=10, loc="left")
    a_step.legend(fontsize=7, loc="best")
    _stamp(a_step, f"terminal carries {stats['terminal_share_of_abs_reward'] * 100:.0f} % of the "
                   f"episode's total |reward|")

    rets = np.asarray([e["ep_return"] for e in eps])
    lo, hi = float(rets.min()), float(rets.max())
    pad = 0.05 * max(hi - lo, 1.0)
    bins = np.linspace(lo - pad, hi + pad, 25)
    for o, colour in OUTCOME_COLOUR.items():
        sel = [e["ep_return"] for e in eps if e["outcome"] == o]
        if sel:
            a_hist.hist(sel, bins=bins, color=colour, alpha=0.75, label=f"{o} (n = {len(sel)})")
    a_hist.set_xlabel("episodic return")
    a_hist.set_ylabel("episodes")
    a_hist.set_title("B — episodic return, split by outcome", fontsize=10, loc="left")
    a_hist.legend(fontsize=7, loc="upper right")
    _stamp(a_hist, f"mean {stats['ep_return_mean']:.2f}   sd {stats['ep_return_std']:.2f}   "
                   f"range [{lo:.2f}, {hi:.2f}]")

    tmax = max(e["ep_len"] for e in eps)
    for o, colour in OUTCOME_COLOUR.items():
        sel = [(e["ep_len"], e["ep_return"]) for e in eps if e["outcome"] == o]
        if sel:
            a_len.plot([s[0] for s in sel], [s[1] for s in sel], "o", ms=5, alpha=0.8,
                       color=colour, label=o)
    tgrid = np.arange(1, tmax + 1)
    r_hold = float(cfg.get("r_hold", 5.0))
    for o, terminal in (("contact", R_CONTACT), ("return", R_RETURN),
                        ("timeout", float(cfg.get("r_timeout", 100.0)))):
        if any(e["outcome"].startswith(o) for e in eps):
            a_len.plot(tgrid, REWARD_SCALE * (terminal - r_hold * tgrid), ls="--", lw=1.0,
                       color=OUTCOME_COLOUR[o], alpha=0.6)
    a_len.set_xlabel("episode length (steps)")
    a_len.set_ylabel("episodic return")
    a_len.set_title("C — return is length plus outcome", fontsize=10, loc="left")
    a_len.legend(fontsize=7, loc="upper right")
    _stamp(a_len, f"dashed: {REWARD_SCALE} x (terminal - r_hold x T), r_hold = {r_hold}")

    fig.suptitle(_title(cfg, mode, len(eps)), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def fig_positions(eps, cfg, *, when="initial", deterministic=True):
    """Plot the target positions relative to own vessel at the start or the end.

    Args:
        eps (list): Episode records
        cfg (dict): Run config
        when (str): initial or final
        deterministic (bool): Whether actions were the mean

    Returns:
        Figure: Figure
    """
    setup()
    first = when == "initial"
    mode = "deterministic" if deterministic else "sampled"
    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    contact_m = float(cfg.get("contact_m", 200.0))
    gate = float(cfg.get("gate", 500.0) or 500.0)

    points = []
    for e in eps:
        k = 0 if first else len(e["tracks"][0]) - 1
        for n, (x, y, th) in enumerate(_relative(e["tracks"], k)):
            cls = e["classes"][n] if n < len(e["classes"]) else None
            points.append((x, y, th, cls, e["outcome"]))
    reach = max([contact_m] + [math.hypot(p[0], p[1]) for p in points])
    tick = max(260.0, reach * 0.05)

    seen = set()
    for x, y, th, cls, outcome in points:
        colour = C_CLASS.get(cls, C_TS) if first else OUTCOME_COLOUR.get(outcome, GREY)
        key = cls if first else outcome
        lab = None
        if key is not None and key not in seen:
            lab = f"target ship: {key}" if first else key
            seen.add(key)
        ax.plot([x], [y], "o", ms=4.5, alpha=0.85, color=colour, label=lab, zorder=4)
        ax.plot([x, x + tick * math.cos(th)], [y, y + tick * math.sin(th)],
                color=colour, lw=1.0, alpha=0.75, zorder=3)
    ax.plot([], [], marker="^", ls="none", color=C_OS, label="own ship (origin, heading up)")
    _ship(ax, 0.0, 0.0, math.pi / 2, colour=C_OS, length=max(260.0, reach * 0.06))
    _rings(ax, (contact_m, gate), (f"{contact_m:.0f} m contact", f"{gate:.0f} m gate"))
    ax.set_xlim(-reach * 1.15, reach * 1.15)
    ax.set_ylim(-reach * 1.15, reach * 1.15)
    _equal(ax)
    ax.set_xlabel("metres to starboard of the own ship")
    ax.set_ylabel("metres ahead of the own ship")
    ax.set_title(f"{'initial' if first else 'final'} traffic, own-ship frame "
                 f"({'class' if first else 'outcome'} by colour)", fontsize=10, loc="left")
    ax.legend(fontsize=7, loc="upper right")
    _stamp(ax, f"{len(eps)} episodes   ·   {len(points)} target ships   ·   "
               f"heading tick = {tick:,.0f} m")
    fig.suptitle(_title(cfg, mode, len(eps)), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def fig_episode(ep, cfg):
    """Plot one episode: tracks, actions and rewards.

    Args:
        ep (dict): Episode record
        cfg (dict): Run config

    Returns:
        Figure: Figure
    """
    setup()
    part = decompose(ep, r_hold=cfg.get("r_hold", 5.0),
                     r_violation=cfg.get("r_violation", 0.0),
                     r_timeout=cfg.get("r_timeout", 100.0))
    contact_m = float(cfg.get("contact_m", 200.0))
    fig = plt.figure(figsize=(16.5, 9.6))
    gs = fig.add_gridspec(2, 3, width_ratios=(1.35, 1.0, 1.0))
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_i, ax_f = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    ax_a, ax_r = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])

    panel_episode(ax_traj, ep["tracks"], ep["handback"], ep["outcome"], 0.0,
                  classes=ep["classes"], contact_m=contact_m)
    ax_traj.set_title(f"A — episode {ep['episode']} (seed {ep['seed']}), {ep['ep_len']} steps "
                      f"→ {ep['outcome']}", fontsize=10, loc="left")
    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")
    _square(ax_traj)

    for ax, k, name in ((ax_i, 0, "B — initial state"),
                        (ax_f, len(ep["tracks"][0]) - 1, "C — final state")):
        reach = contact_m
        for n, (x, y, th) in enumerate(_relative(ep["tracks"], k)):
            cls = ep["classes"][n] if n < len(ep["classes"]) else None
            colour = C_CLASS.get(cls, C_TS)
            _ship(ax, x, y, th, colour=colour, length=220, label=f"TS{n + 1}"
                  + (f" ({cls})" if cls else ""))
            reach = max(reach, math.hypot(x, y))
        _ship(ax, 0.0, 0.0, math.pi / 2, colour=C_OS, length=220, label="OS")
        _rings(ax, (contact_m,), (f"{contact_m:.0f} m",))
        ax.set_xlim(-reach * 1.25, reach * 1.25)
        ax.set_ylim(-reach * 1.25, reach * 1.25)
        _equal(ax)
        ax.set_title(f"{name}, own-ship frame   ·   step {k}", fontsize=10, loc="left")
        ax.set_xlabel("m to starboard")
        ax.set_ylabel("m ahead")

    steps = np.arange(1, ep["ep_len"] + 1)
    ax_a.axhspan(SATURATED, 1.0, color=GREY, alpha=0.12, zorder=0)
    ax_a.axhspan(-1.0, -SATURATED, color=GREY, alpha=0.12, zorder=0)
    _lab = dim_labels(ep["actions"].shape[1], cfg.get("speed_action"))
    for d in range(ep["actions"].shape[1]):
        ax_a.plot(steps, ep["actions"][:, d], color=DIM_COLOUR[d % len(DIM_COLOUR)],
                  lw=1.5, marker="o", ms=3,
                  label=_lab[d] if d < len(_lab) else f"a{d}")
    ax_a.set_ylim(-1.08, 1.08)
    ax_a.set_xlabel("step")
    ax_a.set_ylabel("command")
    ax_a.set_title("D — the commands", fontsize=10, loc="left")
    ax_a.legend(fontsize=7, loc="lower right")

    ax_r.plot(steps, ep["rewards"], color=GREY, lw=1.8, marker="o", ms=3, label="step reward")
    for key, colour in (("shaping", BLUE), ("hold", ORANGE), ("violation", PINK),
                        ("terminal", VERMILLION)):
        if np.any(part[key]):
            ax_r.plot(steps, part[key], color=colour, lw=1.2, label=key)
    ax_r.axhline(0.0, color=GREY, lw=0.8, ls=":")
    ax_r.set_xlabel("step")
    ax_r.set_ylabel("reward")
    ax_r.set_title(f"E — the reward   ·   return {ep['ep_return']:.2f}", fontsize=10, loc="left")
    ax_r.legend(fontsize=7, loc="best")

    fig.suptitle(f"{cfg.get('launched', 'run')} — episode {ep['episode']}: "
                 f"{ep['outcome']}, ep_cpa {ep['ep_cpa']:,.0f} m, "
                 f"return {ep['ep_return']:.2f}", fontsize=11)
    fig.text(0.01, 0.012,
             "ep_cpa is the environment's segment minimum over the whole episode including the "
             "compliant tail; panel A's closest approach is measured between drawn step "
             "positions, so the two differ by the between-step geometry.",
             fontsize=7, color=GREY, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return fig


def write_actions_csv(eps, path):
    """Write every action of every episode to CSV.

    Args:
        eps (list): Episode records
        path (str): CSV path
    """
    dim = eps[0]["actions"].shape[1]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["episode", "seed", "step"] + [f"a{d}" for d in range(dim)]
                   + ["reward", "rules_broken", "outcome", "ep_len", "ep_return"])
        for e in eps:
            for t in range(e["ep_len"]):
                w.writerow([e["episode"], e["seed"], t + 1]
                           + [round(float(x), 5) for x in e["actions"][t]]
                           + [round(e["rewards"][t], 5), e["broken"][t], e["outcome"],
                              e["ep_len"], round(e["ep_return"], 5)])


def numbers_of(eps, cfg, *, deterministic=True):
    """Collect the numbers shown in the figures.

    Args:
        eps (list): Episode records
        cfg (dict): Run config
        deterministic (bool): Whether actions were the mean

    Returns:
        dict: Numbers
    """
    parts = decompose_all(eps, cfg)
    return {
        "spec": {"run": cfg.get("launched"), "bank": cfg.get("bank"), "episodes": len(eps),
                 "deterministic": bool(deterministic),
                 "action_dim": int(eps[0]["actions"].shape[1]),
                 **{k: cfg.get(k) for k in CFG_KEYS + ("obs_dim",)}},
        "action_spread": spread_stats(eps),
        "reward": reward_stats(eps, parts),
    }


def write_set(eps, cfg, out_dir, *, deterministic=True, every=100, overview=True):
    """Write the figure set, the CSV and the numbers of a rollout.

    Args:
        eps (list): Episode records
        cfg (dict): Run config
        out_dir (str): Output directory
        deterministic (bool): Whether actions were the mean
        every (int): Draw one episode figure every n episodes
        overview (bool): Whether to add the overview figure

    Returns:
        tuple: Numbers and written paths
    """
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, fig in (("actions", fig_actions(eps, cfg, deterministic=deterministic)),
                      ("reward", fig_reward(eps, cfg, deterministic=deterministic)),
                      ("positions_initial",
                       fig_positions(eps, cfg, when="initial", deterministic=deterministic)),
                      ("positions_final",
                       fig_positions(eps, cfg, when="final", deterministic=deterministic))):
        _save(fig, os.path.join(out_dir, name))
        written.append(name)
    if overview:
        _save(overview_figure(eps, cfg, deterministic=deterministic),
              os.path.join(out_dir, "episode_anatomy"))
        written.append("episode_anatomy")

    if every:
        ep_dir = os.path.join(out_dir, "episodes")
        os.makedirs(ep_dir, exist_ok=True)
        for stale in sorted(os.listdir(ep_dir)):
            if stale.startswith("ep") and stale.endswith((".png", ".pdf")):
                os.remove(os.path.join(ep_dir, stale))
        for e in eps:
            if e["episode"] % every:
                continue
            _save(fig_episode(e, cfg), os.path.join(ep_dir, f"ep{e['episode']:04d}"))
            written.append(f"episodes/ep{e['episode']:04d}")

    numbers = numbers_of(eps, cfg, deterministic=deterministic)
    with open(os.path.join(out_dir, "numbers.json"), "w") as fh:
        json.dump(numbers, fh, indent=2)
    write_actions_csv(eps, os.path.join(out_dir, "actions.csv"))
    return numbers, written


def dump_training_episode(env, policy, out_dir, *, index, seed, deterministic=True, **cfg_extra):
    """Run and draw one episode during training, never raising.

    Args:
        env (CertifiedEscapeEnv): Environment
        policy (object): Current policy
        out_dir (str): Output directory
        index (int): Episode index
        seed (int): Random seed
        deterministic (bool): Whether to take the mean action
        **cfg_extra (dict): Extra config fields

    Returns:
        dict: Episode record, or None on failure
    """
    try:
        ep = roll_episode(env, policy, seed, deterministic=deterministic, index=index)
        cfg = cfg_from_env(env, **cfg_extra)
        os.makedirs(out_dir, exist_ok=True)
        _save(fig_episode(ep, cfg), os.path.join(out_dir, f"ep{index:06d}"))
        return ep
    except Exception as exc:
        print(f"[episode_figs] filmstrip at episode {index} failed: {exc!r}")
        return None


def _span(labels, ncol=12):
    """Repeat each label to fill a number of columns.

    Args:
        labels (list): Labels
        ncol (int): Number of columns

    Returns:
        list: Labels per column
    """
    k = ncol // len(labels)
    return [lab for lab in labels for _ in range(k)]


def overview_figure(eps, cfg, *, deterministic=True):
    """Draw the one page overview of a rollout.

    Args:
        eps (list): Episode records
        cfg (dict): Run config
        deterministic (bool): Whether actions were the mean

    Returns:
        Figure: Figure
    """
    setup()
    dim = eps[0]["actions"].shape[1]
    labels = dim_labels(dim, cfg.get("speed_action"))
    parts = decompose_all(eps, cfg)
    stats = reward_stats(eps, parts)
    mode = "deterministic" if deterministic else "sampled"
    mosaic = [_span([f"a{d}" for d in range(dim)]), _span(["hist", "reward", "ret", "len"])]
    fig, ax = plt.subplot_mosaic(mosaic, figsize=(16.0, 8.4))

    for d in range(dim):
        a = ax[f"a{d}"]
        a.axhspan(SATURATED, 1.0, color=GREY, alpha=0.12, zorder=0)
        a.axhspan(-1.0, -SATURATED, color=GREY, alpha=0.12, zorder=0)
        for e in eps:
            a.plot(np.arange(1, e["ep_len"] + 1), e["actions"][:, d], lw=1.0, alpha=0.55,
                   color=OUTCOME_COLOUR.get(e["outcome"], GREY))
        a.set_ylim(-1.08, 1.08)
        a.set_xlabel("step")
        a.set_ylabel(labels[d] if d < len(labels) else f"a{d}")
        a.set_title(f"{chr(65 + d)} — command {d} per episode ({mode})", fontsize=10, loc="left")
        vals = np.concatenate([e["actions"][:, d] for e in eps])
        _stamp(a, f"saturated |a| > {SATURATED}: {(np.abs(vals) > SATURATED).mean() * 100:.0f} %"
                  f"   ·   distinct values (2 dp): {len(np.unique(np.round(vals, 2)))}"
                  f"   ·   sd {vals.std():.3f}")
        if d == 0:
            _outcome_legend(a, eps)

    a = ax["hist"]
    bins = np.linspace(-1.0, 1.0, 41)
    for d in range(dim):
        vals = np.concatenate([e["actions"][:, d] for e in eps])
        a.hist(vals, bins=bins, histtype="step", density=True, lw=1.6,
               color=DIM_COLOUR[d % len(DIM_COLOUR)], label=f"a{d}")
    a.set_xlabel("command value")
    a.set_ylabel("density over all steps")
    a.set_title(f"{chr(65 + dim)} — repartition of the commands", fontsize=10, loc="left")
    a.legend(fontsize=7)
    _stamp(a, f"{sum(e['ep_len'] for e in eps):,} steps over {len(eps)} episodes")

    a = ax["reward"]
    trun = max(1, max(e["ep_len"] - 1 for e in eps))
    def _mean_series(key):
        """Compute the mean and spread of one reward part per step.

        Args:
            key (str): Reward part name

        Returns:
            tuple: Mean and standard deviation per step
        """
        grid = np.full((len(eps), trun), np.nan)
        for i, (e, p) in enumerate(zip(eps, parts)):
            n = min(e["ep_len"] - 1, trun)
            if n > 0:
                grid[i, :n] = p[key][:n] if key else e["rewards"][:n]
        return np.nanmean(grid, axis=0), np.nanstd(grid, axis=0)
    steps = np.arange(1, trun + 1)
    total, sd = _mean_series(None)
    a.fill_between(steps, total - sd, total + sd, color=GREY, alpha=0.20, lw=0)
    a.plot(steps, total, color=GREY, lw=2.0, label="step reward (mean +- 1 sd)")
    for key, colour in (("shaping", BLUE), ("hold", ORANGE), ("violation", PINK)):
        m, _sd = _mean_series(key)
        if np.nanmax(np.abs(m)) > 0:
            a.plot(steps, m, color=colour, lw=1.3, label=key)
    a.axhline(0.0, color=GREY, lw=0.8, ls=":")
    a.set_xlabel("step   (terminal step excluded)")
    a.set_ylabel("reward per step")
    a.set_title(f"{chr(66 + dim)} — where the reward comes from", fontsize=10, loc="left")
    a.legend(fontsize=7, loc="best")
    _stamp(a, f"terminal carries {stats['terminal_share_of_abs_reward'] * 100:.0f} % of the "
              f"episode's total |reward|")

    a = ax["ret"]
    rets = np.asarray([e["ep_return"] for e in eps])
    lo, hi = float(rets.min()), float(rets.max())
    pad = 0.05 * max(hi - lo, 1.0)
    bins = np.linspace(lo - pad, hi + pad, 25)
    for o, colour in OUTCOME_COLOUR.items():
        sel = [e["ep_return"] for e in eps if e["outcome"] == o]
        if sel:
            a.hist(sel, bins=bins, color=colour, alpha=0.75, label=f"{o} (n = {len(sel)})")
    a.set_xlabel("episodic return")
    a.set_ylabel("episodes")
    a.set_title(f"{chr(67 + dim)} — episodic return, split by outcome", fontsize=10, loc="left")
    a.legend(fontsize=7, loc="upper right")
    _stamp(a, f"mean {stats['ep_return_mean']:.2f}   sd {stats['ep_return_std']:.2f}   "
              f"range [{lo:.2f}, {hi:.2f}]")

    a = ax["len"]
    tmax = max(e["ep_len"] for e in eps)
    for o, colour in OUTCOME_COLOUR.items():
        sel = [(e["ep_len"], e["ep_return"]) for e in eps if e["outcome"] == o]
        if sel:
            a.plot([s[0] for s in sel], [s[1] for s in sel], "o", ms=5, alpha=0.8,
                   color=colour, label=o)
    tgrid = np.arange(1, tmax + 1)
    r_hold = float(cfg.get("r_hold", 5.0))
    for o, terminal in (("contact", R_CONTACT), ("return", R_RETURN),
                        ("timeout", float(cfg.get("r_timeout", 100.0)))):
        if any(e["outcome"].startswith(o) for e in eps):
            a.plot(tgrid, REWARD_SCALE * (terminal - r_hold * tgrid), ls="--", lw=1.0,
                   color=OUTCOME_COLOUR[o], alpha=0.6)
    a.set_xlabel("episode length (steps)")
    a.set_ylabel("episodic return")
    a.set_title(f"{chr(68 + dim)} — return is length plus outcome", fontsize=10, loc="left")
    a.legend(fontsize=7, loc="upper right")
    _stamp(a, f"dashed: {REWARD_SCALE} x (terminal - r_hold x T), r_hold = {r_hold}")

    fig.suptitle(_title(cfg, mode, len(eps)), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _fake_episodes(rng, cfg, n_eps=4, n_ts=2):
    """Build synthetic episodes for the self check.

    Args:
        rng (Generator): Random generator
        cfg (dict): Run config
        n_eps (int): Number of episodes
        n_ts (int): Number of targets

    Returns:
        list: Episode records
    """
    eps = []
    for i, outcome in enumerate(("contact", "return", "timeout", "contact_after_handback")[:n_eps]):
        n = 5 + i
        broken = rng.integers(0, 2, n).astype(float)
        raw = rng.normal(0, 0.3, n) - cfg["r_hold"] - cfg["r_violation"] * broken
        raw[-1] += {"contact": R_CONTACT, "contact_after_handback": R_CONTACT,
                    "return": R_RETURN, "timeout": cfg["r_timeout"]}[outcome]
        tracks = [np.column_stack([np.linspace(0, 900, n + 1) + 300 * j,
                                   np.linspace(0, 600, n + 1) - 250 * j,
                                   np.full(n + 1, 0.4 * j)])
                  for j in range(n_ts + 1)]
        eps.append({"episode": i, "seed": i, "actions": rng.uniform(-1, 1, (n, 2)),
                    "rewards": raw * REWARD_SCALE, "broken": broken, "outcome": outcome,
                    "tracks": tracks, "classes": ["compliant", "adversarial"][:n_ts],
                    "handback": None, "ep_return": float((raw * REWARD_SCALE).sum()),
                    "ep_len": n, "ep_cpa": 300.0, "min_separation": 300.0})
    return eps


def _selfcheck():
    """Check the reward split and the figure writers on synthetic episodes.

    Returns:
        bool: True if every check passes
    """
    import tempfile
    ok = True
    rng = np.random.default_rng(0)
    cfg = {"r_hold": 5.0, "r_violation": 0.5, "r_timeout": 100.0, "gamma": 0.95,
           "contact_m": 200.0, "gate": 500.0, "release_lookahead": 60, "launched": "selfcheck"}
    eps = _fake_episodes(rng, cfg)

    for e in eps:
        p = decompose(e, r_hold=cfg["r_hold"], r_violation=cfg["r_violation"],
                      r_timeout=cfg["r_timeout"])
        got = p["shaping"] + p["hold"] + p["violation"] + p["terminal"]
        good = np.allclose(got, e["rewards"], atol=1e-9)
        print(f"  {'ok  ' if good else 'FAIL'} decompose sums to the reward ({e['outcome']})")
        ok &= good

    s = spread_stats(eps)
    good = s["n_steps"] == sum(e["ep_len"] for e in eps) and len(s["mean"]) == 2
    print(f"  {'ok  ' if good else 'FAIL'} spread_stats over {s['n_steps']} steps, "
          f"{len(s['mean'])} dimensions")
    ok &= good

    for theta in (0.0, math.pi / 2, -2.1):
        tracks = [np.array([[0.0, 0.0, theta]]),
                  np.array([[1000 * math.cos(theta), 1000 * math.sin(theta), theta]])]
        x, y, _th = _relative(tracks, 0)[0]
        good = abs(x) < 1e-6 and abs(y - 1000.0) < 1e-6
        print(f"  {'ok  ' if good else 'FAIL'} own-ship frame at heading {theta:+.2f}: "
              f"({x:+.1f}, {y:+.1f})")
        ok &= good

    kw = env_kwargs({"gate": 500.0, "algo": "ppo", "n_track": None, "contact_m": 200.0,
                     "mix": {"compliant": 1.0}})
    good = "algo" not in kw and "n_track" not in kw and kw["contact_m"] == 200.0
    print(f"  {'ok  ' if good else 'FAIL'} env_kwargs drops the learner and the nulls: "
          f"{sorted(kw)}")
    ok &= good

    good = _span(["a0", "a1"]) == ["a0"] * 6 + ["a1"] * 6 and len(_span(["a", "b", "c", "d"])) == 12
    print(f"  {'ok  ' if good else 'FAIL'} mosaic rows are 12 columns wide")
    ok &= good

    with tempfile.TemporaryDirectory() as d:
        numbers, written = write_set(eps, cfg, d, every=2)
        files = sorted(os.path.relpath(os.path.join(r, f), d)
                       for r, _dirs, fs in os.walk(d) for f in fs if f.endswith(".png"))
        want = {"actions.png", "reward.png", "positions_initial.png", "positions_final.png",
                "episode_anatomy.png"}
        good = want <= set(files) and any(f.startswith("episodes/") for f in files)
        print(f"  {'ok  ' if good else 'FAIL'} write_set produced {len(files)} pages: "
              f"{', '.join(files)}")
        ok &= good
        good = numbers["reward"]["ep_return_std"] > 0 and os.path.exists(
            os.path.join(d, "actions.csv"))
        print(f"  {'ok  ' if good else 'FAIL'} numbers.json + actions.csv beside them")
        ok &= good
    plt.close("all")
    return ok


def main():
    """Roll a trained policy and write its figures from the command line."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--run", help="a training run directory (holds config.json and model.zip)")
    ap.add_argument("--model", default=None, help="override the checkpoint inside --run")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--every", type=int, default=100,
                    help="full graphical interpretation of every Nth episode (0 = none)")
    ap.add_argument("--seed", type=int, default=ROLLOUT_SEED)
    ap.add_argument("--split", choices=("heldout", "train"), default="heldout")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample the policy instead of taking its mean — what training saw")
    ap.add_argument("--out", default=None,
                    help="output directory (default <run>/figures/episode_figs)")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()

    if a.selfcheck:
        raise SystemExit(0 if _selfcheck() else 1)
    if not a.run:
        ap.error("--run is required (or --selfcheck)")

    cfg = load_config(a.run)
    model_path = a.model or os.path.join(a.run, "model.zip")
    from .policy_io import algo_of, load_policy
    policy = load_policy(model_path, device="cpu")
    rows = population(cfg, a.split)
    env = CertifiedEscapeEnv(rows, seed=a.seed, **env_kwargs(cfg))
    width = env.observation_space.shape[0]
    if cfg.get("obs_dim") and width != int(cfg["obs_dim"]):
        raise SystemExit(f"[episode_figs] the environment built here is {width} wide and the run "
                         f"trained at {cfg['obs_dim']} — the figure would diagnose a policy that "
                         f"was never trained. Check the knobs {a.run}/config.json records.")
    print(f"[episode_figs] {algo_of(model_path)} {model_path}")
    print(f"[episode_figs] {len(rows)} {a.split} rows, obs {width}, "
          f"action {env.action_space.shape[0]}, contact {env.contact_m:.0f} m, "
          f"release_lookahead {env.release_lookahead}")

    eps = rollout(env, policy, a.episodes, seed=a.seed,
                  deterministic=not a.stochastic, progress="rollout")
    out_dir = a.out or os.path.join(a.run, "figures", "episode_figs")
    numbers, written = write_set(eps, cfg, out_dir, deterministic=not a.stochastic, every=a.every)

    sp, rw = numbers["action_spread"], numbers["reward"]
    print(f"[episode_figs] saturated {['%.0f %%' % (f * 100) for f in sp['saturated_frac']]}, "
          f"distinct values {sp['unique_rounded']}, "
          f"sd within/between {['%.3f' % v for v in sp['std_within_episode']]} / "
          f"{['%.3f' % v for v in sp['std_between_episodes']]}")
    print(f"[episode_figs] return {rw['ep_return_mean']:.2f} +- {rw['ep_return_std']:.2f} "
          f"in [{rw['ep_return_min']:.2f}, {rw['ep_return_max']:.2f}], "
          f"terminal carries {rw['terminal_share_of_abs_reward'] * 100:.0f} % of |reward|")
    print(f"[episode_figs] wrote {len(written)} pages to {out_dir}")


if __name__ == "__main__":
    main()

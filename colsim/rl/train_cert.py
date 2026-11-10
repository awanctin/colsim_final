from __future__ import annotations

import argparse
import collections
import csv
import datetime
import json
import os
import sys
import time

import numpy as np

from ..evaluation import bank as _bank
from ..certificate import behaviour as _behaviour
from .cert_env import (
    ALARM_TRIGGER_K,
    GAMMA,
    MAX_STEPS,
    R_TIMEOUT,
    REWARD_SCALE,
    VETO_MARGIN_M,
    W_ALPHA_COL,
    W_ALPHA_PHYS,
    W_SPEED,
    CertifiedEscapeEnv,
    EpisodeStart,
    reward_ordering_bound,
)

UNIFORM_MIX = _behaviour.class_mix("uniform")
HOLDOUT_EVERY = 5


def split_bank(rows, holdout=0.2):
    """Split bank rows into train and held out, one row in every n held out.

    Args:
        rows (list): Bank rows
        holdout (float): Held out share

    Returns:
        tuple: Train rows and held out rows
    """
    if holdout <= 0:
        return list(rows), []
    every = max(int(round(1.0 / holdout)), 2)
    train = [r for i, r in enumerate(rows) if i % every]
    evalr = [r for i, r in enumerate(rows) if not i % every]
    return train, evalr


def train_strata(rows, include_free=False):
    """Get the strata used for training.

    Args:
        rows (list): Bank rows
        include_free (bool): Whether to keep free rows

    Returns:
        tuple: Strata names
    """
    if not include_free:
        return ("certified", "pre_certified")
    return tuple(sorted({r["stratum"] for r in rows}))


def stack_for_policy(env, policy):
    """Wrap the environment with frame stacking if the policy expects it.

    Args:
        env (Env): Environment
        policy (object): Trained policy

    Returns:
        Env: Environment, stacked if needed
    """
    space = getattr(policy, "observation_space", None)
    if space is None:
        return env
    want = int(np.prod(space.shape))
    have = int(np.prod(env.observation_space.shape))
    if want == have or have == 0:
        return env
    n, rem = divmod(want, have)
    if rem or n < 2:
        raise ValueError(
            f"policy observes {want} and this environment produces {have}, which is not a whole "
            f"number of stacked frames. That is a different observation, not a deeper history — "
            f"check `behaviour_obs`, `n_track`, `margin_steps` and `bin_fields` against the run's "
            f"config.json."
        )
    return _stacked(env, n)


def _stacked(env, n):
    """Stack and flatten n frames of observation.

    Args:
        env (Env): Environment
        n (int): Number of frames

    Returns:
        Env: Stacked environment, or env if n is 1
    """
    if int(n) <= 1:
        return env
    from gymnasium.wrappers import FlattenObservation, FrameStack

    return FlattenObservation(FrameStack(env, int(n)))


def make_env(rows_or_path, *, gate, gamma, ts_classes, seed, horizon=None, holdout=0.2,
             split="train", frame_stack=1, include_free=False, **kw):
    """Get a function that builds one training or held out environment.

    Args:
        rows_or_path (object): Bank CSV path or list of rows
        gate (float): Duty gate, m
        gamma (float): Discount factor
        ts_classes (dict): Target behaviour mix
        seed (int): Random seed
        horizon (float): Certificate horizon, s
        holdout (float): Held out share
        split (str): train or heldout
        frame_stack (int): Number of stacked frames
        include_free (bool): Whether to keep free rows
        **kw (dict): Extra environment arguments

    Returns:
        function: Environment builder
    """
    def _thunk():
        """Build the environment.

        Returns:
            Env: Environment
        """
        rows = rows_or_path
        if isinstance(rows, str):
            loaded = _bank.load(rows)
            keep = train_strata(loaded, include_free)
            loaded = [r for r in loaded if r["stratum"] in keep]
            train_rows, eval_rows = split_bank(loaded, holdout)
            rows = train_rows if split == "train" else eval_rows
        env = CertifiedEscapeEnv(rows, gate=gate, gamma=gamma, ts_classes=ts_classes,
                                 seed=seed, **({} if horizon is None else {"horizon": horizon}),
                                 **kw)
        return _stacked(env, frame_stack)
    return _thunk


CONTACT_OUTCOMES = ("contact", "contact_after_handback")


def contact_rate(rates):
    """Sum the rates of the contact outcomes.

    Args:
        rates (dict): Rate per outcome

    Returns:
        float: Contact rate
    """
    return sum(rates.get(k, 0.0) for k in CONTACT_OUTCOMES)


def plateaued(contacts, patience):
    """Check that the contact rate has not improved over the last evaluations.

    Args:
        contacts (list): Contact rate per evaluation
        patience (int): Number of evaluations without gain

    Returns:
        bool: True if plateaued
    """
    if patience <= 0 or len(contacts) <= patience:
        return False
    return min(contacts[-patience:]) >= min(contacts[:-patience])


def rollout_outcomes(env, policy=None, n_episodes=50, seed=0, log=None, progress=None):
    """Run held out episodes and count the outcomes.

    Args:
        env (Env): Environment
        policy (object): Policy
        n_episodes (int): Number of episodes
        seed (int): First seed
        log (list): Per episode log, appended, or None
        progress (str): Progress bar label, or None

    Returns:
        tuple: Rate per outcome and mean steps in control
    """
    out = collections.Counter()
    steps = []
    episodes = range(n_episodes)
    if progress:
        from tqdm import tqdm
        episodes = tqdm(episodes, desc=progress, unit="ep", leave=True)
    for i in episodes:
        obs, info0 = env.reset(seed=seed + i)
        n = 0
        ret = 0.0
        while True:
            if policy is None:
                action = np.array([0.0, 1.0, 1.0],
                                  dtype=np.float32)[:env.action_space.shape[0]]
            else:
                action, _ = policy.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(action)
            n += 1
            ret += float(_r)
            if term or trunc:
                break
        out[info.get("outcome", "truncated")] += 1
        steps.append(n)
        if log is not None:
            log.append({
                "ep_return": ret,
                "episode": i, "seed": seed + i, "row": info0.get("idx"),
                "stratum": info0.get("stratum"), "outcome": info.get("outcome", "truncated"),
                "steps": n, "min_separation": info.get("min_separation"),
                "ep_cpa": info.get("ep_cpa", ""),
                "ep_ts_cpa": info.get("ep_ts_cpa", ""),
                "handback_step": info.get("handback_step", ""),
                "tail_steps": info.get("tail_steps", ""),
                "alpha_col": info.get("alpha_col"), "alpha_phys": info.get("alpha_phys"),
                "rules_broken": info.get("rules_broken_sum", ""),
                "violation_steps": info.get("violation_steps", ""),
                "permit_steps": info.get("permit_steps", ""),
                "ask_steps": info.get("ask_steps", ""),
                "veto_steps": info.get("veto_steps", ""),
                "shield_steps": info.get("shield_steps", ""),
            })
    total = sum(out.values()) or 1
    return {k: v / total for k, v in out.items()}, float(np.mean(steps))


FIG_ROLL = 25
FIG_ROLL_LONG = 1000
FILMSTRIP_EVERY = 100
FILMSTRIP_SEED = 4242
SUCCESS_WINDOW = 100
_SUCCESS = ("return",)
HULL_CONTACT_M = 150.0
_OUTCOMES = (("contact", "tab:red"), ("contact_after_handback", "tab:purple"),
             ("return", "tab:green"), ("timeout", "tab:orange"))
_DIAG_KEYS = (("train/value_loss", "value loss", "tab:red"),
              ("train/policy_gradient_loss", "policy gradient loss", "tab:blue"),
              ("train/entropy_loss", "entropy loss", "tab:green"),
              ("train/explained_variance", "explained variance", "tab:purple"))


def _rolling(y, w):
    """Compute a rolling mean.

    Args:
        y (list): Values
        w (int): Window size

    Returns:
        ndarray: Rolling mean
    """
    y = np.asarray(y, dtype=float)
    w = max(1, min(int(w), len(y)))
    return np.convolve(y, np.ones(w), "valid") / w


def _window(n, target):
    """Pick a rolling window size for n points.

    Args:
        n (int): Number of points
        target (int): Wanted window

    Returns:
        int: Window size
    """
    return int(max(1, min(target, n // 10 or 1)))


def _rolling_std(y, w):
    """Compute a rolling standard deviation.

    Args:
        y (list): Values
        w (int): Window size

    Returns:
        ndarray: Rolling standard deviation
    """
    y = np.asarray(y, dtype=float)
    y = y - y.mean() if len(y) else y
    w = max(1, min(int(w), len(y)))
    c1 = np.concatenate(([0.0], np.cumsum(y)))
    c2 = np.concatenate(([0.0], np.cumsum(y * y)))
    mean = (c1[w:] - c1[:-w]) / w
    var = (c2[w:] - c2[:-w]) / w - mean * mean
    return np.sqrt(np.maximum(var, 0.0))


def _commit():
    """Get the git commit of the code, if any.

    Returns:
        str: Commit hash, or None
    """
    import subprocess
    try:
        return subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)),
             "describe", "--always", "--dirty", "--exclude", "*"],
            capture_output=True, text=True, timeout=5, check=True).stdout.strip() or None
    except Exception:
        return None


def make_run_dir(root="output/runs", tag=None, stamp=None):
    """Create a run directory with checkpoint and figure folders.

    Args:
        root (str): Parent directory
        tag (str): Run tag
        stamp (str): Time stamp, or None for now

    Returns:
        str: Run directory
    """
    stamp = stamp or datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = os.path.join(root, f"{stamp}_{tag}" if tag else stamp)
    for sub in ("checkpoints", "figures"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir


def _run_callback(run_dir, eval_env, every, n_episodes, roll=FIG_ROLL, log_csv=True,
                  filmstrip_every=FILMSTRIP_EVERY, filmstrip_fixed_row=False,
                  plateau=0, plateau_min_steps=0):
    """Build the callback that logs episodes, runs held out evaluations and draws figures.

    Args:
        run_dir (str): Run directory
        eval_env (Env): Held out environment
        every (int): Steps between evaluations
        n_episodes (int): Held out episodes per evaluation
        roll (int): Rolling window size
        log_csv (str): Episode log CSV path
        filmstrip_every (int): Draw one episode every n episodes, 0 for none
        filmstrip_fixed_row (bool): Whether the drawn episode keeps one seed
        plateau (int): Plateau patience, 0 for none
        plateau_min_steps (int): Steps before the plateau stop can fire

    Returns:
        BaseCallback: Callback
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from stable_baselines3.common.callbacks import BaseCallback

    fig_dir = os.path.join(run_dir, "figures")
    csv_path = os.path.join(run_dir, "episodes.csv")

    def _band(ax, xs, ys, colour, label):
        """Plot points with a rolling mean and spread band.

        Args:
            ax (Axes): Axes
            xs (list): x values
            ys (list): y values
            colour (str): Colour
            label (str): Legend label
        """
        ax.plot(xs, ys, "o", alpha=0.15, color=colour, label="episode")
        if len(ys) > 1:
            w = min(roll, len(ys))
            mean = _rolling(ys, w)
            std = np.array([np.std(ys[max(0, i - w):i]) for i in range(w, len(ys) + 1)])
            ax.plot(xs[w - 1:], mean, color=colour, label=f"{label} ({w} ep)")
            ax.fill_between(xs[w - 1:], mean - std, mean + std, alpha=0.2, color=colour,
                            label="+/-1 std")
        med = float(np.median(ys))
        ax.axhline(med, color="r", linestyle="--", label=f"median {med:.1f}")

    def _p_heldout_outcomes(ax, ev):
        """Plot the held out outcome rates over training.

        Args:
            ax (Axes): Axes
            ev (list): Evaluations
        """
        xs = [e[0] for e in ev]
        for key, colour in _OUTCOMES:
            ax.plot(xs, [e[1].get(key, 0.0) for e in ev], marker="o", color=colour, label=key)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("training steps")
        ax.set_ylabel(f"held-out rate (n={n_episodes})")
        ax.set_title("held-out outcomes")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    def _p_heldout_control(ax, ev):
        """Plot the held out mean steps in control over training.

        Args:
            ax (Axes): Axes
            ev (list): Evaluations
        """
        ax.plot([e[0] for e in ev], [e[2] for e in ev], marker="o", color="tab:blue",
                label="mean steps in control")
        ax.axhline(2.4, color="r", linestyle="--", label="2.4 = clear-and-hand-back (2026-08-05)")
        ax.set_xlabel("training steps")
        ax.set_ylabel("steps")
        ax.set_title("time in control before hand-back")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    def _p_return(ax, eps):
        """Plot the training episode return.

        Args:
            ax (Axes): Axes
            eps (list): Episode records
        """
        _band(ax, [e["timestep"] for e in eps], [e["ep_return"] for e in eps],
              "tab:blue", "rolling mean")
        ax.set_xlabel("training steps")
        ax.set_ylabel("episode return")
        ax.set_title("training episode return")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    def _p_length(ax, eps):
        """Plot the training episode length.

        Args:
            ax (Axes): Axes
            eps (list): Episode records
        """
        _band(ax, [e["timestep"] for e in eps], [e["ep_len"] for e in eps],
              "tab:purple", "rolling mean")
        ax.set_xlabel("training steps")
        ax.set_ylabel("steps in control")
        ax.set_title("training episode length")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    def _p_share(ax, eps):
        """Plot the rolling share of each outcome.

        Args:
            ax (Axes): Axes
            eps (list): Episode records
        """
        xs = [e["timestep"] for e in eps]
        for key, colour in _OUTCOMES:
            flag = [1.0 if e["outcome"] == key else 0.0 for e in eps]
            w = min(roll, len(flag))
            ax.plot(xs[w - 1:], _rolling(flag, w), color=colour, label=key)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("training steps")
        ax.set_ylabel(f"share over {min(roll, len(eps))} episodes")
        ax.set_title("training outcome share")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    def _p_collision_rates(ax, eps):
        """Plot the rolling contact rates.

        Args:
            ax (Axes): Axes
            eps (list): Episode records
        """
        xs = [e["timestep"] for e in eps]
        w = min(roll, len(eps)) or 1
        for key, colour in (("contact", "tab:red"), ("contact_after_handback", "tab:purple")):
            flag = [1.0 if e["outcome"] == key else 0.0 for e in eps]
            ys = _rolling(flag, w)
            final = float(ys[-1]) if len(ys) else float("nan")
            ax.plot(xs[w - 1:], ys, color=colour, label=f"{key} (final {final*100:.1f} %)")
        ax.axhline(0.0, color="k", linestyle=":", label="zero")
        ax.set_ylim(-0.01, None)
        ax.set_xlabel("training steps")
        ax.set_ylabel(f"collision rate over {w} episodes")
        ax.set_title("collisions by channel")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    def _p_min_separation(ax, eps):
        """Plot the closest separation of each episode.

        Args:
            ax (Axes): Axes
            eps (list): Episode records
        """
        xs = [e["timestep"] for e in eps]
        ys = [float(e.get("ep_cpa") or e["min_separation"]) for e in eps]
        ax.plot(xs, ys, "o", ms=3, alpha=0.12, color="tab:green", label="episode")
        w = min(roll, len(ys))
        if len(ys) > w:
            arr = np.asarray(ys, dtype=float)
            win = np.lib.stride_tricks.sliding_window_view(arr, w)
            for q, style, lab in ((0.5, "-", "median"), (0.25, "--", "p25"), (0.05, ":", "p5")):
                ax.plot(xs[w - 1:], np.quantile(win, q, axis=1), style, color="darkgreen",
                        lw=1.6, label=f"{lab} ({w} ep)")
        ax.axhline(HULL_CONTACT_M, color="k", linestyle=":", label="hull contact 150 m")
        ax.set_yscale("log")
        ax.set_xlabel("training steps")
        ax.set_ylabel("closest approach over the episode (m)")
        ax.set_title("closest approach")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

    def _p_diag(key, label, colour):
        """Get a plot function for one training diagnostic.

        Args:
            key (str): Logger key
            label (str): Axis label
            colour (str): Colour

        Returns:
            function: Plot function
        """
        def _plot(ax, diag):
            """Plot one training diagnostic over steps.

            Args:
                ax (Axes): Axes
                diag (list): Diagnostics per step
            """
            pts = [(t, d[key]) for t, d in diag if key in d]
            if not pts:
                ax.set_axis_off()
                return
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=colour,
                    label=label)
            if key.endswith("explained_variance"):
                ax.axhline(0.0, color="k", linestyle=":", label="critic no better than mean")
                ax.set_ylim(-1.05, 1.05)
            ax.set_xlabel("training steps")
            ax.set_ylabel(label)
            ax.set_title(key)
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)

        return _plot

    def _p_alpha(ax, eps):
        """Plot the physical escape width at episode end.

        Args:
            ax (Axes): Axes
            eps (list): Episode records
        """
        _band(ax, [e["timestep"] for e in eps], [e["alpha_phys"] for e in eps],
              "tab:orange", "rolling mean")
        ax.set_xlabel("training steps")
        ax.set_ylabel("alpha_phys at episode end (rad)")
        ax.set_title("escape freedom on release")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    def _p_reward_stats(ax, eps):
        """Plot the episode return with rolling mean and spread.

        Args:
            ax (Axes): Axes
            eps (list): Episode records
        """
        ys = np.array([e["ep_return"] for e in eps], dtype=float)
        xs = np.arange(1, len(ys) + 1)
        w = _window(len(ys), FIG_ROLL_LONG)
        ax.fill_between(xs, 0.0, ys, color="tab:orange", alpha=0.15, label="episode")
        mean, std = _rolling(ys, w), _rolling_std(ys, w)
        ax.fill_between(xs[w - 1:], mean - std, mean + std, color="tab:blue", alpha=0.25,
                        label="+/-1 std")
        ax.plot(xs[w - 1:], mean, color="tab:blue", label=f"rolling mean ({w} ep)")
        for key, colour in _OUTCOMES:
            sel = np.array([i for i, e in enumerate(eps) if e["outcome"] == key], dtype=int)
            if len(sel) < 5:
                continue
            wo = _window(len(sel), max(w // 4, 5))
            ax.plot(sel[wo - 1:] + 1, _rolling(ys[sel], wo), color=colour, lw=1.0, alpha=0.75,
                    label=f"{key} ({len(sel)})")
        tail = eps[-w:]
        yt = np.array([e["ep_return"] for e in tail], dtype=float)
        lt = np.array([float(e["ep_len"]) for e in tail], dtype=float)
        note = [f"std {float(np.std(yt)):.2f} over the last {len(tail)} episodes"]
        vt = float(np.var(yt))
        if vt > 0:
            groups = [yt[[i for i, e in enumerate(tail) if e["outcome"] == k]]
                      for k, _ in _OUTCOMES]
            groups = [g for g in groups if len(g)]
            between = sum(len(g) * (float(np.mean(g)) - float(np.mean(yt))) ** 2
                          for g in groups) / len(yt)
            note.append(f"outcome mix {100 * between / vt:.0f} % of variance")
            if float(np.std(lt)) > 0:
                r = float(np.corrcoef(yt, lt)[0, 1])
                note.append(f"episode length {100 * r * r:.0f} % (r = {r:+.2f})")
        ax.text(0.02, 0.03, "\n".join(note), transform=ax.transAxes, fontsize=7, va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
        ax.set_xlabel("episode")
        ax.set_ylabel("episode return")
        ax.set_title(f"reward statistics (window {w})")
        ax.legend(loc="lower right", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    def _p_success_rate(ax, eps):
        """Plot the rolling success rate.

        Args:
            ax (Axes): Axes
            eps (list): Episode records
        """
        flag = np.array([100.0 if e["outcome"] in _SUCCESS else 0.0 for e in eps])
        w = _window(len(flag), SUCCESS_WINDOW)
        rate = _rolling(flag, w)
        xs = np.arange(w, len(flag) + 1)
        std = _rolling_std(rate, w)
        ax.fill_between(xs[len(xs) - len(std):], rate[len(rate) - len(std):] - std,
                        rate[len(rate) - len(std):] + std, color="tab:blue", alpha=0.25,
                        label=f"std dev {float(np.std(rate)):.2f}")
        ax.plot(xs, rate, color="tab:blue", label=f"success rate / {w} episodes (%)")
        ax.set_ylim(-2, 102)
        ax.set_xlabel("episode")
        ax.set_ylabel("success rate (%)")
        ax.set_title("success rate")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    class _RunLogger(BaseCallback):
        def __init__(self):
            """Set up the logger state."""
            super().__init__(verbose=0)
            self.evals = []
            self.episodes = []
            self.diag = []
            self._ret = None
            self._len = None
            self._last = 0
            self._tail_total = 0
            self._agent_total = 0
            self._t0 = time.perf_counter()
            if log_csv:
                with open(csv_path, "w", newline="") as fh:
                    csv.writer(fh).writerow(["timestep", "env", "ep_return", "ep_len", "outcome",
                                             "min_separation", "ep_cpa", "ep_ts_cpa", "row",
                                             "alpha_phys",
                                             "rules_broken", "tail_steps"])

        def _collect(self):
            """Record every finished episode of the step."""
            n = self.training_env.num_envs
            if self._ret is None:
                self._ret = np.zeros(n)
                self._len = np.zeros(n, dtype=int)
            self._ret += np.asarray(self.locals["rewards"], dtype=float)
            self._len += 1
            rows = []
            for i, done in enumerate(self.locals["dones"]):
                if not done:
                    continue
                info = self.locals["infos"][i]
                rec = {"timestep": self.num_timesteps, "env": i,
                       "ep_return": float(self._ret[i]), "ep_len": int(self._len[i]),
                       "outcome": info.get("outcome", "truncated"),
                       "min_separation": float(info.get("min_separation", np.nan)),
                       "ep_cpa": float(info.get("ep_cpa", np.nan)),
                       "ep_ts_cpa": float(info.get("ep_ts_cpa", np.nan)),
                       "row": info.get("idx", ""),
                       "alpha_phys": float(info.get("alpha_phys", np.nan)),
                       "rules_broken": int(info.get("rules_broken_sum", 0)),
                       "tail_steps": int(info.get("tail_steps", 0) or 0)}
                self._tail_total += rec["tail_steps"]
                self._agent_total += rec["ep_len"]
                self.episodes.append(rec)
                if (filmstrip_every and eval_env is not None
                        and len(self.episodes) % filmstrip_every == 0):
                    from . import episode_figs as _efigs
                    n_done = len(self.episodes)
                    _efigs.dump_training_episode(
                        eval_env, self.model, os.path.join(fig_dir, "episodes"),
                        index=n_done,
                        seed=FILMSTRIP_SEED + (0 if filmstrip_fixed_row else n_done),
                        launched=os.path.basename(run_dir))
                rows.append([rec[k] for k in ("timestep", "env", "ep_return", "ep_len",
                                              "outcome", "min_separation", "ep_cpa", "ep_ts_cpa",
                                              "row", "alpha_phys", "rules_broken", "tail_steps")])
                self._ret[i] = 0.0
                self._len[i] = 0
            if rows and log_csv:
                with open(csv_path, "a", newline="") as fh:
                    csv.writer(fh).writerows(rows)

        def figures(self):
            """Draw and save every training figure."""
            singles = [("heldout_outcomes", _p_heldout_outcomes, self.evals),
                       ("heldout_steps_in_control", _p_heldout_control, self.evals),
                       ("training_episode_return", _p_return, self.episodes),
                       ("training_episode_length", _p_length, self.episodes),
                       ("training_outcome_share", _p_share, self.episodes),
                       ("training_min_separation", _p_min_separation, self.episodes),
                       ("training_collision_rates", _p_collision_rates, self.episodes),
                       ("training_alpha_phys", _p_alpha, self.episodes),
                       ("reward_stats", _p_reward_stats, self.episodes),
                       ("success_rate", _p_success_rate, self.episodes)]
            singles += [(key.replace("train/", "train_"), _p_diag(key, label, colour), self.diag)
                        for key, label, colour in _DIAG_KEYS]
            for name, fn, data in singles:
                if not data:
                    continue
                fig, ax = plt.subplots(figsize=(8, 4.5))
                fn(ax, data)
                fig.tight_layout()
                fig.savefig(os.path.join(fig_dir, name + ".png"), dpi=110)
                plt.close(fig)
            panels = [("panel_heldout", (_p_heldout_outcomes, _p_heldout_control), self.evals),
                      ("panel_training", (_p_return, _p_share), self.episodes),
                      ("panel_optimisation",
                       (_p_diag(*_DIAG_KEYS[0]), _p_diag(*_DIAG_KEYS[3])), self.diag)]
            for name, fns, data in panels:
                if not data:
                    continue
                fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
                for ax, fn in zip(axes, fns):
                    fn(ax, data)
                fig.tight_layout()
                fig.savefig(os.path.join(fig_dir, name + ".png"), dpi=110)
                plt.close(fig)

        def _on_step(self):
            """Collect episodes and, every n steps, evaluate and draw.

            Returns:
                bool: False to stop training
            """
            self._collect()
            if self.num_timesteps - self._last < every:
                return True
            self._last = self.num_timesteps
            self._report_rates()
            if eval_env is not None and n_episodes:
                rates, mean_steps = rollout_outcomes(eval_env, self.model, n_episodes)
                self.evals.append((self.num_timesteps, rates, mean_steps))
            self.figures()
            return not self._plateau_stop()

        def _plateau_stop(self):
            """Check whether the plateau stop fires.

            Returns:
                bool: True to stop
            """
            if not plateau or self.num_timesteps < plateau_min_steps:
                return False
            contacts = [contact_rate(r) for _t, r, _m in self.evals]
            if not plateaued(contacts, plateau):
                return False
            print(f"\n[plateau] stopping at {self.num_timesteps} steps: contact rate "
                  f"{contacts[-1]:.3f}, best {min(contacts):.3f}, no gain in {plateau} evals",
                  flush=True)
            return True

        def _report_rates(self):
            """Print the agent and simulation step rates."""
            el = max(time.perf_counter() - self._t0, 1e-9)
            agent = self.num_timesteps / el
            sim = (self._agent_total + self._tail_total) / el
            share = (self._tail_total / max(self._agent_total + self._tail_total, 1)) * 100.0
            print(f"[rates] {self.num_timesteps:>8} steps | agent {agent:5.1f}/s | "
                  f"simulated {sim:5.1f}/s | tail {share:4.1f} % of simulation "
                  f"({self._tail_total} steps)")

        def _snapshot_diag(self):
            """Store the training diagnostics if they changed."""
            recorded = {k: float(v) for k, v in self.logger.name_to_value.items()
                        if k.startswith("train/")}
            if recorded and (not self.diag or self.diag[-1][1] != recorded):
                self.diag.append((self.num_timesteps, recorded))

        def _on_rollout_start(self):
            """Store the training diagnostics."""
            self._snapshot_diag()

        def _on_training_end(self):
            """Store the diagnostics and draw the figures."""
            self._snapshot_diag()
            self.figures()

    return _RunLogger()


def resolve_lookahead(release_lookahead, max_steps):
    """Turn a negative release lookahead into the episode length.

    Args:
        release_lookahead (int): Lookahead, steps, negative for the full window
        max_steps (int): Episode length, or None

    Returns:
        int: Lookahead, steps
    """
    k = int(release_lookahead)
    return int(max_steps or MAX_STEPS) if k < 0 else k


def train(bank_path, *, gate, steps=1_000_000, n_envs=4, n_steps=None, gamma=GAMMA,
          holdout=0.2,
          ts_classes=None, out=None, seed=0, horizon=None, eval_episodes=50,
          verbose=1, subproc=False, save_every=20_000, plot_every=20_000, plot_episodes=25,
          plateau=0, plateau_min_steps=0,
          tb=False, run_root="output/runs", tag=None, run_dir=None, tail_steps=None,
          margin_steps=0, strict_reward_ordering=False,
          r_hold=0.0, r_timeout=R_TIMEOUT, min_hold_steps=0, max_steps=None,
          alpha_col_release=0.0, rule17b_relax=False, init_from=None,
          no_progress=False, ent_coef=None, net_arch=None,
          algo="ppo", utd=1.0, buffer_size=300_000, learning_starts=5_000, lr=None,
          r_violation=0.0, release_action=False, release_lookahead=0, action_shield=0,
          veto_margin=VETO_MARGIN_M,
          feasible_only=False, feasible_at=None, contact_m=None, n_track=None, hard_frac=0.0,
          bin_fields=False, tubes=False, wall=None, wall_reserve=900.0,
          filmstrip_every=FILMSTRIP_EVERY, filmstrip_fixed_row=False,
          speed_action=False, behaviour_obs=False, frame_stack=1, include_free=False,
          random_episode_every=0, episode_start=EpisodeStart.ROW,
          alarm_trigger_k=ALARM_TRIGGER_K):
    """Train the emergency policy and evaluate it on the held out rows.

    Args:
        bank_path (str): Bank CSV path
        gate (float): Duty gate, m
        steps (int): Training steps
        n_envs (int): Number of environments
        n_steps (int): Rollout length per environment
        gamma (float): Discount factor
        holdout (float): Held out share
        ts_classes (dict): Target behaviour mix
        out (str): Model path
        seed (int): Random seed
        horizon (float): Certificate horizon, s
        eval_episodes (int): Held out episodes per evaluation
        verbose (int): Verbosity
        subproc (bool): Whether environments run in subprocesses
        save_every (int): Steps between checkpoints
        plot_every (int): Steps between evaluations and figures
        plot_episodes (int): Held out episodes for figures
        plateau (int): Plateau patience, 0 for none
        plateau_min_steps (int): Steps before the plateau stop can fire
        tb (str): Tensorboard directory, or None
        run_root (str): Parent of run directories
        tag (str): Run tag
        run_dir (str): Run directory, or None for a new one
        tail_steps (int): Steps run after handback, or None
        margin_steps (int): Steps of the margin channel, 0 for none
        strict_reward_ordering (bool): Whether a bad reward ordering raises
        r_hold (float): Per step hold cost
        r_timeout (float): Survival reward
        min_hold_steps (int): Min emergency steps before release
        max_steps (int): Episode length, or None
        alpha_col_release (float): Lawful width share needed to release
        rule17b_relax (bool): Whether Rule 17(b) relaxation is on
        init_from (str): Policy to start from, or None
        no_progress (bool): Whether to hide progress bars
        ent_coef (object): Entropy coefficient, or None
        net_arch (list): Hidden layer sizes, or None
        algo (str): ppo or sac
        utd (int): SAC gradient steps per step
        buffer_size (int): SAC replay buffer size
        learning_starts (int): SAC steps before learning
        lr (float): Learning rate, or None
        r_violation (float): Per rule violation cost
        release_action (bool): Whether the policy asks for release
        release_lookahead (int): Steps replayed before a release
        action_shield (int): Steps of the action shield, 0 for none
        veto_margin (float): Separation a release needs, m
        feasible_only (bool): Whether to keep only feasible rows
        feasible_at (float): Feasibility CPA threshold, m
        contact_m (float): Contact distance, m
        n_track (int): Number of tracked targets
        hard_frac (float): Share of episodes drawn from failed rows
        bin_fields (bool): Whether to add the per bin fields
        tubes (bool): Whether to use reachable tubes
        wall (float): Wall clock budget, s, or None
        wall_reserve (float): Time kept for the final evaluation, s
        filmstrip_every (int): Draw one episode every n episodes, 0 for none
        filmstrip_fixed_row (bool): Whether the drawn episode keeps one seed
        speed_action (bool): Whether the policy sets speed
        behaviour_obs (bool): Whether to add behaviour class probabilities
        frame_stack (int): Number of stacked frames
        include_free (bool): Whether to keep free rows
        random_episode_every (int): One uniform random episode every n, 0 for none
        episode_start (EpisodeStart): Where the episode starts
        alarm_trigger_k (int): Lead cap of the alarm pre roll, steps

    Returns:
        tuple: Model and held out report
    """
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from . import policy_io

    algo = str(algo).lower()
    if algo not in policy_io.ALGOS:
        raise ValueError(f"--algo {algo} is not one of {policy_io.ALGOS}")

    if run_dir:
        for sub in ("checkpoints", "figures"):
            os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    else:
        run_dir = make_run_dir(run_root, tag)
    _max_steps = {} if max_steps is None else {"max_steps": int(max_steps)}
    release_lookahead = resolve_lookahead(release_lookahead, max_steps)
    _obs = {}
    if n_track is not None:
        _obs["n_track"] = int(n_track)
    if contact_m is not None:
        _obs["contact_m"] = float(contact_m)
    if bin_fields:
        _obs["bin_fields"] = True
    if tubes:
        _obs["tubes"] = True
    if speed_action:
        _obs["speed_action"] = True
    if behaviour_obs:
        _obs["behaviour_obs"] = True
    episode_start = EpisodeStart(episode_start)
    if episode_start is EpisodeStart.ALARM:
        _obs["episode_start"] = episode_start
        _obs["alarm_trigger_k"] = int(alarm_trigger_k)
    all_rows = _bank.load(bank_path)
    strata = train_strata(all_rows, include_free)
    if include_free:
        _obs["strata"] = strata
    if action_shield:
        print(f"[train_cert] WARNING: --action-shield {action_shield} shields the policy during "
              f"training. The rollout buffer stores the sampled action while the environment "
              f"executes the substituted one, so the policy gradient is credited to an action "
              f"that never ran. Train at 0 and evaluate with the guard instead.")
    rows = [r for r in all_rows if r["stratum"] in strata]
    if not rows:
        raise ValueError(f"{bank_path} holds no certified or pre-certified rows")
    train_rows, eval_rows = split_bank(rows, holdout)
    if feasible_only:
        keep = lambda rs: [r for r in rs if str(r.get("feasible", "1")) not in ("0", "False")]
        n0, m0 = len(train_rows), len(eval_rows)
        train_rows, eval_rows = keep(train_rows), keep(eval_rows)
        print(f"feasible-only: train {n0} -> {len(train_rows)}, eval {m0} -> {len(eval_rows)}")
        if not train_rows:
            raise ValueError(f"{bank_path} has no feasible rows — is the `feasible` column there?")
    if feasible_at is not None:
        n0, m0 = len(train_rows), len(eval_rows)
        def keep_at(rs):
            """Keep the rows whose feasibility CPA reaches the threshold.

            Args:
                rs (list): Bank rows

            Returns:
                list: Kept rows
            """
            out = []
            for r in rs:
                c = r.get("feas_cpa", "")
                if c == "" or c is None:
                    raise ValueError(
                        f"--feasible-at {feasible_at} needs a `feas_cpa` column on every row of "
                        f"{bank_path}; row {r.get('idx')} has none. Run "
                        f"`python3 -m COLSim_pkg.main_program.RL.label_feasible --bank {bank_path}` "
                        f"first."
                    )
                if float(c) >= float(feasible_at):
                    out.append(r)
            return out
        train_rows, eval_rows = keep_at(train_rows), keep_at(eval_rows)
        print(f"feasible-at {feasible_at:.0f} m: train {n0} -> {len(train_rows)}, "
              f"eval {m0} -> {len(eval_rows)}")
        if not train_rows:
            raise ValueError(f"no row in {bank_path} reaches {feasible_at} m under the probe")
    mix = UNIFORM_MIX if ts_classes is None else ts_classes

    source = bank_path if subproc else train_rows
    thunks = [make_env(source, gate=gate, gamma=gamma, ts_classes=mix, seed=seed + i,
                       horizon=horizon, holdout=holdout, split="train",
                       tail_steps=tail_steps, r_hold=r_hold, r_timeout=r_timeout,
                       margin_steps=margin_steps,
                       strict_reward_ordering=strict_reward_ordering,
                       min_hold_steps=min_hold_steps, **_max_steps,
                       alpha_col_release=alpha_col_release, r_violation=r_violation,
                       release_action=release_action, release_lookahead=release_lookahead,
                       veto_margin=veto_margin,
                       action_shield=action_shield, hard_frac=hard_frac, **_obs,
                       frame_stack=frame_stack, include_free=include_free,
                       rule17b_relax=rule17b_relax) for i in range(n_envs)]
    vec = (SubprocVecEnv(thunks) if (subproc and n_envs > 1) else DummyVecEnv(thunks))

    tb_log = os.path.join(run_dir, "tb") if tb else None
    policy_kwargs = {"net_arch": list(net_arch)} if net_arch else None
    if ent_coef is None:
        ent_coef = "auto" if algo == "sac" else 0.0
    kw = dict(gamma=gamma, verbose=verbose, seed=seed, batch_size=256, ent_coef=ent_coef,
              tensorboard_log=tb_log)
    if n_steps and algo == "ppo":
        kw["n_steps"] = int(n_steps)
    if lr is not None:
        kw["learning_rate"] = ((lambda p, a=abs(float(lr)): a * p) if float(lr) < 0
                               else float(lr))
    if algo == "sac":
        kw.update(buffer_size=int(buffer_size), learning_starts=int(learning_starts),
                  train_freq=1, gradient_steps=max(1, int(round(float(utd) * n_envs))))
    if init_from:
        saved = policy_io.algo_of(init_from)
        if saved != algo:
            raise ValueError(f"--init-from {init_from} holds a {saved} policy while --algo is "
                             f"{algo}; there is no cross-algorithm resume")
        model = policy_io.algo_class(algo).load(init_from, env=vec, **kw)
    else:
        model = policy_io.algo_class(algo)("MlpPolicy", vec, policy_kwargs=policy_kwargs, **kw)

    config = {"launched": os.path.basename(run_dir), "commit": _commit(),
              "bank": os.path.abspath(bank_path),
              "gate": gate, "steps": int(steps), "n_envs": n_envs, "gamma": gamma,
              "holdout": holdout, "mix": mix, "seed": seed, "horizon": horizon,
              "subproc": bool(subproc), "save_every": save_every, "plot_every": plot_every,
              "plateau": plateau, "plateau_min_steps": plateau_min_steps,
              "plot_episodes": plot_episodes, "filmstrip_every": filmstrip_every,
              "filmstrip_fixed_row": bool(filmstrip_fixed_row),
              "tail_steps": tail_steps, "r_hold": r_hold,
              "margin_steps": margin_steps,
              "r_timeout": r_timeout, "min_hold_steps": min_hold_steps,
              "max_steps": max_steps,
              "alpha_col_release": alpha_col_release, "batch_size": model.batch_size,
              "reward_ordering": reward_ordering_bound(
                  gamma, r_timeout, int(max_steps) if max_steps else MAX_STEPS),
              "lr": float(lr) if lr is not None else None,
              "algo": algo, "utd": float(utd) if algo == "sac" else None,
              "buffer_size": int(buffer_size) if algo == "sac" else None,
              "learning_starts": int(learning_starts) if algo == "sac" else None,
              "gradient_steps": int(model.gradient_steps) if algo == "sac" else None,
              "ent_coef": ent_coef if isinstance(ent_coef, str) else float(ent_coef),
              "net_arch": list(net_arch) if net_arch else None,
              "w_alpha_col": W_ALPHA_COL, "w_alpha_phys": W_ALPHA_PHYS, "w_speed": W_SPEED,
              "r_violation": r_violation, "release_action": bool(release_action),
              "release_lookahead": int(release_lookahead), "action_shield": int(action_shield),
              "veto_margin": float(veto_margin),
              "feasible_only": bool(feasible_only), "feasible_at": feasible_at,
              "contact_m": contact_m, "n_track": n_track, "hard_frac": float(hard_frac),
              "action_dim": int(vec.action_space.shape[0]),
              "reward_scale": REWARD_SCALE, "rule17b_relax": bool(rule17b_relax),
              "tubes": bool(tubes),
              "bin_fields": bool(bin_fields),
              "speed_action": bool(speed_action), "behaviour_obs": bool(behaviour_obs),
              "frame_stack": int(frame_stack), "include_free": bool(include_free),
              "episode_start": episode_start.value, "alarm_trigger_k": int(alarm_trigger_k),
              "random_episode_every": int(random_episode_every),
              "wall": None if wall is None else float(wall),
              "wall_reserve": float(wall_reserve),
              "obs_dim": int(vec.observation_space.shape[0]),
              "train_rows": len(train_rows), "eval_rows": len(eval_rows)}
    if steps:
        with open(os.path.join(run_dir, "config.json"), "w") as fh:
            json.dump(config, fh, indent=2)

    cbs = []
    if save_every:
        cbs.append(CheckpointCallback(save_freq=max(save_every // n_envs, 1),
                                      save_path=os.path.join(run_dir, "checkpoints"),
                                      name_prefix=algo))
    if plot_every:
        penv = None
        if plot_episodes:
            penv = CertifiedEscapeEnv(eval_rows or train_rows, gate=gate, gamma=gamma,
                                      ts_classes=mix, seed=7777, tail_steps=tail_steps,
                                      margin_steps=margin_steps,
                                      r_hold=r_hold, r_timeout=r_timeout,
                                      min_hold_steps=min_hold_steps, **_max_steps,
                                      alpha_col_release=alpha_col_release,
                                      r_violation=r_violation, release_action=release_action,
                                      release_lookahead=release_lookahead,
                                      veto_margin=veto_margin,
                                      action_shield=action_shield, **_obs,
                                      rule17b_relax=rule17b_relax,
                                      **({} if horizon is None else {"horizon": horizon}))
            penv = _stacked(penv, frame_stack)
        cbs.append(_run_callback(run_dir, penv, plot_every, plot_episodes,
                                 filmstrip_every=filmstrip_every,
                                 filmstrip_fixed_row=filmstrip_fixed_row,
                                 plateau=plateau, plateau_min_steps=plateau_min_steps))
    if wall:
        cbs.append(_wall_callback(float(wall), float(wall_reserve)))
    if random_episode_every:
        cbs.append(_install_random_episodes(model, n_envs, random_episode_every, seed=seed))
    model_path = os.path.join(run_dir, "model")
    previous_sigterm = _install_sigterm()
    try:
        if steps:
            model.learn(total_timesteps=int(steps), callback=cbs,
                        progress_bar=sys.stdout.isatty() and not no_progress)
    except KeyboardInterrupt as exc:
        model.save(model_path + "_interrupted")
        why = "terminated" if isinstance(exc, _Terminated) else "interrupted"
        print(f"\n{why} at {model.num_timesteps} steps — "
              f"saved {model_path}_interrupted.zip; the held-out evaluation still runs")
    finally:
        _restore_sigterm(previous_sigterm)
    if steps:
        model.save(model_path)
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        model.save(out)
    vec.close()

    report = {"run_dir": run_dir, "bank": bank_path, "gate": gate,
              "train_rows": len(train_rows), "eval_rows": len(eval_rows),
              "steps": int(steps), "gamma": gamma, "mix": mix}
    if eval_rows and eval_episodes:
        eenv = CertifiedEscapeEnv(eval_rows, gate=gate, gamma=gamma, ts_classes=mix, seed=9999, margin_steps=margin_steps,
                                  tail_steps=tail_steps, r_hold=r_hold,
                                  r_timeout=r_timeout, min_hold_steps=min_hold_steps,
                                  **_max_steps,
                                  alpha_col_release=alpha_col_release,
                                  r_violation=r_violation, release_action=release_action,
                                  release_lookahead=release_lookahead,
                                  veto_margin=veto_margin,
                                  action_shield=action_shield, **_obs,
                                  rule17b_relax=rule17b_relax,
                                  **({} if horizon is None else {"horizon": horizon}))
        eenv = _stacked(eenv, frame_stack)
        per_episode = []
        for arm, pol in (("null", None), ("policy", model)):
            log = []
            report[arm], report[arm + "_steps"] = rollout_outcomes(
                eenv, pol, eval_episodes, log=log,
                progress=None if no_progress else f"eval {arm}")
            for rec in log:
                per_episode.append({"arm": arm, **rec})
        with open(os.path.join(run_dir, "heldout_episodes.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_episode[0]))
            w.writeheader()
            w.writerows(per_episode)
    with open(os.path.join(run_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    return model, report


def _install_random_episodes(model, n_envs, every, seed=0):
    """Make the policy act uniformly at random in one episode out of n.

    Args:
        model (object): Model
        n_envs (int): Number of environments
        every (int): One random episode every n
        seed (int): Random seed

    Returns:
        BaseCallback: Callback that redraws at each episode end
    """
    import torch as th
    from stable_baselines3.common.callbacks import BaseCallback

    if not hasattr(model.policy, "evaluate_actions"):
        raise ValueError("--random-episode-every is a PPO knob; SAC explores through "
                         "--learning-starts and its own entropy temperature")

    every = int(every)
    if every < 1:
        raise ValueError(f"--random-episode-every must be >= 1, got {every}")

    n_envs = int(n_envs)
    rng = np.random.default_rng(int(seed) + 90_210)
    lo = np.asarray(model.action_space.low, dtype=np.float64)
    hi = np.asarray(model.action_space.high, dtype=np.float64)
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError("--random-episode-every needs a bounded action space, got "
                         f"low {lo} high {hi}")
    mask = np.zeros(n_envs, dtype=bool)
    drawn = [0, 0]

    def _draw(n):
        """Draw which environments run a random episode.

        Args:
            n (int): Number of environments

        Returns:
            ndarray: Random flags
        """
        return rng.random(n) < (1.0 / every)

    def _restart(where):
        """Redraw the random flag of the environments that just ended.

        Args:
            where (ndarray): Done flags
        """
        k = int(np.count_nonzero(where))
        if not k:
            return
        picks = _draw(k)
        mask[where] = picks
        drawn[0] += int(picks.sum())
        drawn[1] += k

    _restart(np.ones(n_envs, dtype=bool))
    base_forward = model.policy.forward

    def forward(obs, deterministic=False):
        """Replace the actions of random episodes with uniform actions.

        Args:
            obs (Tensor): Observations
            deterministic (bool): Whether to take the mean action

        Returns:
            tuple: Actions, values and log probabilities
        """
        actions, values, log_prob = base_forward(obs, deterministic=deterministic)
        if deterministic or not mask.any():
            return actions, values, log_prob
        idx = np.flatnonzero(mask)
        draw = rng.uniform(lo, hi, size=(idx.size, lo.size))
        actions = actions.clone()
        actions[th.as_tensor(idx, device=actions.device)] = th.as_tensor(
            draw, dtype=actions.dtype, device=actions.device)
        values, log_prob, _ = model.policy.evaluate_actions(obs, actions)
        return actions, values, log_prob

    model.policy.forward = forward

    class _RandomEpisodes(BaseCallback):

        def _on_step(self):
            """Redraw the flags of finished episodes.

            Returns:
                bool: True to keep training
            """
            dones = self.locals.get("dones")
            if dones is not None:
                _restart(np.asarray(dones, dtype=bool))
            return True

        def _on_training_end(self):
            """Print the share of random episodes drawn."""
            if drawn[1]:
                print(f"[random-episodes] {drawn[0]}/{drawn[1]} episodes drawn uniformly "
                      f"(1 in {every} asked, {drawn[0] / drawn[1]:.1%} realised)")

    return _RandomEpisodes()


def _wall_callback(wall, reserve):
    """Build the callback that stops training before the wall clock budget.

    Args:
        wall (float): Wall clock budget, s
        reserve (float): Time kept for the final evaluation, s

    Returns:
        BaseCallback: Callback
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class _Wall(BaseCallback):
        def __init__(self):
            """Set up the timer state."""
            super().__init__()
            self.t0 = None
            self.stopped = False

        def _on_training_start(self):
            """Start the timer."""
            self.t0 = time.time()

        def _on_step(self):
            """Stop training once the budget minus the reserve is spent.

            Returns:
                bool: False to stop
            """
            if time.time() - self.t0 < max(float(wall) - float(reserve), 0.0):
                return True
            self.stopped = True
            print(f"\n[wall] stopping training at {self.num_timesteps} steps: "
                  f"{time.time() - self.t0:.0f} s of a {wall:.0f} s budget used, "
                  f"{reserve:.0f} s reserved for the held-out evaluation", flush=True)
            return False

    return _Wall()


class _Terminated(KeyboardInterrupt):
    pass


def _install_sigterm():
    """Turn SIGTERM into an exception so the run can save.

    Returns:
        object: Previous handler, or None
    """
    import signal

    def _raise(_signum, _frame):
        """Raise the termination exception.

        Args:
            _signum (int): Signal number
            _frame (frame): Current frame
        """
        raise _Terminated()

    try:
        return signal.signal(signal.SIGTERM, _raise)
    except ValueError:
        return None


def _restore_sigterm(previous):
    """Restore the previous SIGTERM handler.

    Args:
        previous (object): Previous handler, or None
    """
    import signal

    if previous is not None:
        try:
            signal.signal(signal.SIGTERM, previous)
        except ValueError:
            pass


def _ent_coef(v):
    """Read the entropy coefficient: a float or an auto value.

    Args:
        v (object): Command line value, or None

    Returns:
        object: Coefficient, or None
    """
    if v is None:
        return None
    return v if str(v).startswith("auto") else float(v)


def main():
    """Train the emergency policy from the command line."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--bank", required=False, default="trap_bank.csv")
    ap.add_argument("--gate", type=float, default=None)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=None, metavar="N",
                    help="PPO rollout per env (SB3 default 2048). Set N x --envs equal to the "
                         "control's rollout when oversubscribing workers.")
    ap.add_argument("--subproc", action="store_true",
                    help="SubprocVecEnv instead of DummyVecEnv; works, but only 1.08x at 4 "
                         "workers / 1.62x at 10 — run sweep points as separate processes")
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--horizon", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--run-root", default="output/runs")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--mix",
                    default="uniform",
                    help="behaviour every threat is driven by. `uniform` stays the 3-class mix "
                         "every run before 2026-08-19 trained on, so the route interceptor is "
                         "opt-in and no earlier arm moves; run it as its own arm. A weighted "
                         "spec, `compliant=0.4,late=0.3,wide_berth=0.3`, draws a fleet whose "
                         "vessels differ, which is what a behaviour posterior is there to read.")
    ap.add_argument("--save-every", type=int, default=20_000)
    ap.add_argument("--plot-every", type=int, default=20_000)
    ap.add_argument("--plot-episodes", type=int, default=25)
    ap.add_argument("--filmstrip-every", type=int, default=FILMSTRIP_EVERY,
                    help="every N finished episodes, write the full graphical interpretation of "
                         "one episode (trajectory, initial and final state, commands, reward) to "
                         "<run>/figures/episodes/. Always the same bank row, so the pages read as "
                         "one encounter watched as the policy changes. 0 turns it off. Needs "
                         "--plot-episodes > 0, which is what builds the environment it rolls. "
                         "Each dump is a different bank row unless --filmstrip-fixed-row.")
    ap.add_argument("--filmstrip-fixed-row", action="store_true",
                    help="pin every filmstrip page to the same reset seed, so the sequence is one "
                         "encounter redrawn as the policy changes. Off by default: the pages are "
                         "then a sample of what the policy actually meets, which is the question "
                         "the cadence was asked for.")
    ap.add_argument("--tail-steps", type=int, default=None)
    ap.add_argument("--r-hold", type=float, default=0.0,
                    help="per-step cost of holding emergency control; sweep it, do not "
                         "tune it. At 0.0 a timeout scores as a success with no offset, "
                         "so the optimal policy is to never return control.")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="episode window in steps (env default 60). It also normalises the "
                         "observation's elapsed fraction, so changing it moves the policy "
                         "off-distribution — for diagnosis, or retrain.")
    ap.add_argument("--r-timeout", type=float, default=R_TIMEOUT,
                    help="terminal reward for surviving the window without ever handing "
                         "back. ADR 0004 pays +100, tying 'never release' with a clean "
                         "return; 0.0 encodes the operational definition — an episode "
                         "succeeds iff the latch is released.")
    ap.add_argument("--min-hold", type=int, default=0,
                    help="earliest step at which the latch may be released, whatever the "
                         "certificate says. 0 reproduces the old rule.")
    ap.add_argument("--alpha-col-release", type=float, default=0.0,
                    help="lawful-escape margin required to hand control back, on top of "
                         "p_fire == 0.0; sweep it, do not tune it. 0.0 is the bare boundary "
                         "the policy learns to walk to (2026-08-11: 31%% of releases ended in "
                         "contact, and alpha_col separated those at AUC 0.83)")
    ap.add_argument("--rule17b-relax", action="store_true",
                    help="ADR 0009: a stand-on role flips to give-way while the give-way "
                         "vessel is demonstrably not acting. Off by default — it is a "
                         "disclosed ablation layer, and it changes what `lawful` means")
    ap.add_argument("--ent-coef", default=None,
                    help="entropy coefficient. PPO: a bonus on the policy loss, SB3 defaults it "
                         "to 0.0, so every run before 2026-08-13 explored only through the "
                         "Gaussian head's learned log_std; 0.005 is the low end of the usual "
                         "continuous-control range. SAC: the temperature of the maximum-entropy "
                         "objective, where the default is \"auto\" (tuned against a target "
                         "entropy of -dim(A)) and a fixed number is a hyperparameter nobody has "
                         "swept. Unset means each algorithm's own default. Sweep, do not tune.")
    ap.add_argument("--lr", type=float, default=None, metavar="R",
                    help="optimiser step size; SB3's default is 3e-4 for both learners and no run "
                         "has ever swept it. A NEGATIVE value anneals linearly from |R| to 0 over "
                         "the run (e.g. -3e-4), which is the standard PPO recipe and the one knob "
                         "aimed at the late-training oscillation that widens the ep_return band.")
    ap.add_argument("--algo", choices=("ppo", "sac"), default="ppo",
                    help="the learner. ppo reproduces every run to 2026-08-18. sac replays each "
                         "transition instead of discarding it after one rollout, which is the "
                         "direct attack on the one axis that measured a both-seed-range effect "
                         "that month (budget: 45 -> 28 -> 17 failures per 1000 at 180k/300k/600k). "
                         "It changes the learner only — same env, same bank, same observation. "
                         "SAC costs 3.0x PPO's wall clock per environment step at --utd 1, so "
                         "pair an arm against the PPO control at equal wall clock as well as at "
                         "equal steps.")
    ap.add_argument("--utd", type=float, default=1.0, metavar="R",
                    help="SAC only: gradient updates per environment transition (update-to-data "
                         "ratio). Implemented as train_freq=1 with gradient_steps = R * --envs, "
                         "so the ratio is independent of the worker count (SB3's own default is "
                         "1/--envs, which trains less the more workers you add). It costs wall "
                         "clock, measured 2026-08-18: PPO 145 fps against SAC 48 / 31 / 16 fps "
                         "at R = 1 / 2 / 4 -- ~13 ms per update at every ratio, so linear in "
                         "R, and unmoved by OMP_NUM_THREADS. Sweep it.")
    ap.add_argument("--buffer-size", type=int, default=300_000, metavar="N",
                    help="SAC only: replay capacity in transitions. SB3 defaults to 1e6, which at "
                         "a 133-dim float32 observation is ~1.1 GB of RAM; 300k is ~0.3 GB and "
                         "still ~5000 complete 60-step episodes, which is more than the whole "
                         "bank of held-out rows.")
    ap.add_argument("--learning-starts", type=int, default=5_000, metavar="N",
                    help="SAC only: transitions collected before the first gradient step. SB3 "
                         "defaults to 100, i.e. under two episodes of this env; 5000 is ~83 "
                         "episodes, so the first update sees the outcome distribution rather "
                         "than one episode's tail.")
    ap.add_argument("--net-arch", type=int, nargs="+", default=None,
                    help="hidden layer sizes for MlpPolicy (SB3 default 64 64). Ignored on "
                         "--init-from: a saved policy keeps the architecture it was built with.")
    ap.add_argument("--r-violation", type=float, default=0.0,
                    help="per-step cost per duty the realised turn fails to discharge — the "
                         "rules-broken side of the rules-broken/time-to-return trade-off asked "
                         "for on 2026-08-12. 0.0 reports the count without pricing it, which "
                         "reproduces every earlier run; sweep it to draw the frontier.")
    ap.add_argument("--release-action", action="store_true",
                    help="ADR 0010: Box(3), the third dimension asks to hand back and the "
                         "certificate is demoted from trigger to permission. Invalidates every "
                         "checkpoint (the action space changes). Pair it with --r-timeout 0.0 — "
                         "the agent can now choose to never ask, so the ADR 0004 tie is live.")
    ap.add_argument("--veto-margin", type=float, default=VETO_MARGIN_M, metavar="M",
                    help="separation the replayed compliant tail must hold for a release to be "
                         "permitted, in metres. Only bites at --release-lookahead > 0. The "
                         "default is the hull contact threshold, which certifies nothing beyond "
                         "the hull: of 19 failures on the screening sample, 11 were hand-backs "
                         "whose tail ran down to 159-191 m, inside a 150 m veto and outside a "
                         "250 m one (docs/the-handback-problem.md section 4). 250 removed all "
                         "eleven and touched nothing else.")
    ap.add_argument("--release-lookahead", type=int, default=0, metavar="K",
                    help="certify the tail, not the instant: refuse permission unless the "
                         "compliant run-out stays clear for K steps from here. At K >= the "
                         "window (60) a granted release cannot end in contact, because the "
                         "run-out that follows replays the trajectory the check just watched. "
                         "Costs one compliant tail per candidate release step. 0 is off; "
                         "negative means the whole window, resolved against --max-steps so the "
                         "run config records the number that ran.")
    ap.add_argument("--action-shield", type=int, default=0, metavar="N",
                    help="veto the policy's own turn with an N-step constant-velocity "
                         "lookahead (safety_check.check_heading, whose r_os + r_ts is exactly "
                         "the 150 m contact threshold) and substitute the nearest heading that "
                         "does not predict contact. Attacks in-control contact, the half the "
                         "release lookahead cannot reach. 0 is off.")
    ap.add_argument("--margin-steps", type=int, default=0, metavar="K",
                    help="per-bin margin observation channel: the room a constant turn toward "
                         "each heading bin buys over K steps. 0 = off (81-dim observation, every "
                         "existing checkpoint). Turning it on widens the observation to 113 and "
                         "**invalidates every checkpoint**, so an arm using it cannot be paired "
                         "against one that does not. Use --margin-steps 60: the channel only "
                         "discriminates inside ~v*dt*K, and at K=12 it is noise beyond 600 m.")
    ap.add_argument("--strict-reward-ordering", action="store_true",
                    help="raise instead of warning when r_hold sits outside the interval that "
                         "keeps return > timeout > contact at this gamma. The bound is 5.80 at "
                         "gamma 0.95 and 2.26 at 0.99 with r_timeout 0, so the settled r_hold=5.0 "
                         "inverts the objective at 0.99. See cert_env.reward_ordering_bound.")
    ap.add_argument("--feasible-only", action="store_true",
                    help="keep only rows whose `feasible` column is not 0 — states where some "
                         "controller avoids contact. The bank is a *trap* bank and contains "
                         "inevitable-collision states by construction; scoring a policy on those "
                         "measures the bank, not the policy. Disclose the exclusion wherever the "
                         "resulting numbers are reported.")
    ap.add_argument("--feasible-at", type=float, default=None, metavar="M",
                    help="keep only rows whose `feas_cpa` column reaches M metres — the "
                         "threshold-free version of --feasible-only, which is a bool labelled at "
                         "150 m while the score moved to 200. Needs a bank written by "
                         "label_feasible.py.")
    ap.add_argument("--contact-m", type=float, default=None, metavar="M",
                    help="separation at which an episode terminates in contact and pays "
                         "R_CONTACT. Default 150 (hull contact), which is what every run before "
                         "2026-08-17 trained on while the figures scored 200 — between the two "
                         "the discounted return is flat, so no reward or guard arm could reach "
                         "the quantity it was judged on. Moves the terminal set, not a shaping "
                         "term, and leaves the observation width alone, so --init-from applies.")
    ap.add_argument("--n-track", type=int, default=None, metavar="N",
                    help="threats given a per-threat block (range, bearing, relative heading, "
                         "speed, DCPA, TCPA) in the observation. Default 2 on a bank built at "
                         "n_threats=3, so the third vessel currently reaches the policy only "
                         "through the blocked/lawful bins. 3 adds 6 dimensions and "
                         "**invalidates every checkpoint**.")
    ap.add_argument("--speed-action", action="store_true",
                    help="restore ADR 0007's speed command. Off since 2026-08-25: forcing full "
                         "ahead reproduced the policy's own output on 250 of 250 episodes, "
                         "0 fixed / 0 broken, p = 1.000.")
    ap.add_argument("--behaviour-obs", action="store_true",
                    help="feed the behaviour tracker's posterior (3 floats per tracked threat) "
                         "into the observation. The tracker's estimate, never the simulator's "
                         "label. **Invalidates every checkpoint.**")
    ap.add_argument("--include-free", action="store_true",
                    help="keep the `free` rows — states where the certificate never fires — in "
                         "the training bank. With the release dimension on, handing back at step "
                         "1 is a learnable response to a benign state. Changes the bank "
                         "composition, so it is not comparable with any earlier run.")
    ap.add_argument("--episode-start", choices=[e.value for e in EpisodeStart],
                    default=EpisodeStart.ROW.value,
                    help="row: the policy takes the helm at the bank row's state. alarm: on "
                         "far-start rows the compliant planner drives until the deployed switch "
                         "would engage, and the episode (--max-steps) starts there.")
    ap.add_argument("--alarm-trigger-k", type=int, default=ALARM_TRIGGER_K, metavar="K",
                    help="alarm start: engage on a predicted fire within K steps "
                         "(3 = v0, 30 s; 0 = v2, a present certificate only).")
    ap.add_argument("--frame-stack", type=int, default=1, metavar="N",
                    help="N observation frames concatenated, so the policy reads history instead "
                         "of a posterior computed from it. 6 matches behaviour.WINDOW_STEPS. "
                         "**Invalidates every checkpoint.**")
    ap.add_argument("--wall", type=float, default=None, metavar="SECONDS",
                    help="total wall budget for this run. Training stops at `wall - "
                         "wall-reserve` so the held-out evaluation is inside the cap instead of "
                         "being cut off by it (backlog A9). SIGTERM is handled either way.")
    ap.add_argument("--wall-reserve", type=float, default=900.0, metavar="SECONDS",
                    help="seconds held back from --wall for the held-out evaluation. 250 "
                         "episodes x 2 arms measured about 11 min on the box, hence 900.")
    ap.add_argument("--tubes", action="store_true",
                    help="project threats as reachable sets rather than as points, in both the "
                         "observation and the action shield (backlog A5). Strictly more "
                         "conservative: every bin reads at most as clear as it does now. The "
                         "sets are worst-case, not per-class -- the guard may not read the "
                         "simulator's behaviour labels. Note the saturation horizon: at 15 m/s "
                         "and R_min 300 m the sector is the whole circle by 63 s, so this says "
                         "little over the full 150 s duty horizon. Invalidates no checkpoint "
                         "(the width is unchanged) but changes what the policy sees.")
    ap.add_argument("--bin-fields", action="store_true",
                    help="ship-count-independent observation: the two binary bin channels become "
                         "four graded ones (room / when / load / signed lawful distance) and the "
                         "per-threat block is dropped entirely, so the width is 4*n_bins + 5 for "
                         "any fleet size and --n-track is ignored. `room < 0.5` reproduces the "
                         "old blocked bit exactly, so nothing is lost. **Invalidates every "
                         "checkpoint** — train it cold against a cold control at the same budget, "
                         "never against the warm incumbent.")
    ap.add_argument("--hard-frac", type=float, default=0.0, metavar="P",
                    help="fraction of training episodes drawn from the rows this worker has "
                         "failed on (ep_cpa < contact threshold, per-row EMA), the rest uniform. "
                         "0.0 is the uniform draw every run before 2026-08-17 used, which spends "
                         "~97 %% of the budget on rows already solved. Evaluation stays uniform.")
    ap.add_argument("--random-episode-every", type=int, default=0, metavar="N",
                    help="PPO only: draw one entirely random episode in N, per worker, instead "
                         "of perturbing single steps (Thomas Chaffre, 2026-08-19). The random "
                         "action is what the rollout buffer stores, so credit assignment is "
                         "unaffected. 0 is off and reproduces every earlier run.")
    ap.add_argument("--plateau", type=int, default=0, metavar="K",
                    help="stop training when the periodic held-out contact rate (--plot-every, "
                         "--plot-episodes) has not beaten its best for K evaluations. 0 is off.")
    ap.add_argument("--plateau-min-steps", type=int, default=0, metavar="N",
                    help="no plateau stop before N steps")
    ap.add_argument("--tb", action="store_true")
    ap.add_argument("--init-from", default=None,
                    help="resume from a saved .zip instead of a fresh policy; with "
                         "--steps 0 this runs the held-out evaluation only, and leaves "
                         "config.json and model.zip in the run directory untouched")
    ap.add_argument("--no-progress", action="store_true",
                    help="suppress the rich progress bar even on a tty")
    a = ap.parse_args()
    mix = _behaviour.class_mix(a.mix)
    _model, report = train(a.bank, gate=a.gate, steps=a.steps, n_envs=a.envs,
                           n_steps=a.n_steps, gamma=a.gamma,
                           holdout=a.holdout, ts_classes=mix, out=a.out, seed=a.seed,
                           horizon=a.horizon, eval_episodes=a.eval_episodes,
                           subproc=a.subproc, save_every=a.save_every,
                           plot_every=a.plot_every, plot_episodes=a.plot_episodes,
                           plateau=a.plateau, plateau_min_steps=a.plateau_min_steps,
                           tb=a.tb, run_root=a.run_root, tag=a.tag, run_dir=a.run_dir,
                           tail_steps=a.tail_steps, r_hold=a.r_hold,
                           margin_steps=a.margin_steps,
                           strict_reward_ordering=a.strict_reward_ordering,
                           r_timeout=a.r_timeout, min_hold_steps=a.min_hold,
                           max_steps=a.max_steps,
                           alpha_col_release=a.alpha_col_release,
                           rule17b_relax=a.rule17b_relax, init_from=a.init_from,
                           no_progress=a.no_progress, ent_coef=_ent_coef(a.ent_coef),
                           lr=a.lr, algo=a.algo, utd=a.utd, buffer_size=a.buffer_size,
                           learning_starts=a.learning_starts,
                           net_arch=a.net_arch, r_violation=a.r_violation,
                           release_action=a.release_action,
                           release_lookahead=a.release_lookahead,
                           veto_margin=a.veto_margin,
                           action_shield=a.action_shield,
                           feasible_only=a.feasible_only, feasible_at=a.feasible_at,
                           contact_m=a.contact_m, n_track=a.n_track, hard_frac=a.hard_frac,
                           bin_fields=a.bin_fields, tubes=a.tubes,
                           wall=a.wall, wall_reserve=a.wall_reserve,
                           filmstrip_every=a.filmstrip_every,
                           filmstrip_fixed_row=a.filmstrip_fixed_row,
                           speed_action=a.speed_action,
                           behaviour_obs=a.behaviour_obs, frame_stack=a.frame_stack,
                           include_free=a.include_free,
                           episode_start=a.episode_start, alarm_trigger_k=a.alarm_trigger_k,
                           random_episode_every=a.random_episode_every)
    print("\n--- held-out evaluation ---")
    for k in ("null", "policy"):
        if k in report:
            rates = "  ".join(f"{n} {v:.0%}" for n, v in sorted(report[k].items()))
            print(f"  {k:<7} {rates}   mean {report[k + '_steps']:.1f} steps")
    print(f"  run: {report['run_dir']}   train rows {report['train_rows']}  "
          f"eval rows {report['eval_rows']}")


if __name__ == "__main__":
    main()

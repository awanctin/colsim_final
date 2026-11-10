from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import time

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .kras_env import KrasEnv

TOTAL_STEPS = 3_000_000
CHECKPOINT_STEPS = 100_000
NET_ARCH = [64, 64]


class _Progress(BaseCallback):

    def __init__(self, out):
        """Set up the progress callback.

        Args:
            out (str): Run directory
        """
        super().__init__()
        self.out = out
        self.returns, self.terminals = [], collections.Counter()
        self.t0, self.n0 = time.time(), 0

    def _on_training_start(self):
        """Store the step count at start, for the fps."""
        self.n0 = self.num_timesteps

    def _on_step(self):
        """Count episodes and, every checkpoint, save the model and a progress line.

        Returns:
            bool: True to keep training
        """
        for info in self.locals["infos"]:
            if "episode" in info:
                self.returns.append(info["episode"]["r"])
                self.terminals[info.get("terminal")] += 1
        if self.num_timesteps % CHECKPOINT_STEPS:
            return True
        self.model.save(os.path.join(self.out, f"ckpt_{self.num_timesteps}"))
        fps = (self.num_timesteps - self.n0) / max(time.time() - self.t0, 1e-9)
        row = dict(steps=self.num_timesteps, fps=round(fps, 1), episodes=len(self.returns),
                   mean_return=(round(sum(self.returns) / len(self.returns), 2)
                                if self.returns else None),
                   terminals=dict(self.terminals))
        with open(os.path.join(self.out, "progress.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        self.returns, self.terminals = [], collections.Counter()
        self.t0, self.n0 = time.time(), self.num_timesteps
        return True


def _last_checkpoint(out):
    """Find the newest checkpoint of a run.

    Args:
        out (str): Run directory

    Returns:
        str: Checkpoint path, or None
    """
    ck = glob.glob(os.path.join(out, "ckpt_*.zip"))
    return max(ck, key=lambda p: int(re.search(r"ckpt_(\d+)", p).group(1))) if ck else None


def main():
    """Train the KRAS agent with masked PPO, resuming from the last checkpoint."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=TOTAL_STEPS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(1)

    env = Monitor(KrasEnv(), info_keywords=("terminal",))
    env.reset(seed=args.seed)
    last = _last_checkpoint(args.out)
    if last:
        model = MaskablePPO.load(last, env=env, device="cpu")
        print(f"resuming from {last}", flush=True)
    else:
        model = MaskablePPO("MlpPolicy", env, seed=args.seed, device="cpu",
                            policy_kwargs=dict(net_arch=NET_ARCH))
    remaining = args.steps - model.num_timesteps
    if remaining > 0:
        model.learn(remaining, callback=_Progress(args.out), reset_num_timesteps=False)
    model.save(os.path.join(args.out, "final"))
    print(f"wrote {os.path.join(args.out, 'final.zip')}", flush=True)


if __name__ == "__main__":
    main()

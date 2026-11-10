from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from ..planners.learned.algo_kras import (
    D_GOAL_M, D_HULL_M, D_SENSE_M, EMERGENCY, K_MAX, K_V, N_ACTIONS, N_OBS, TWO_PI, Shield,
    Terminal, is_emergency, observe, path_frame, _ahead,
)
from ..simulator.experiment_constants import SIM_DT_S, SPEED_MAX_MPS, SPEED_MIN_MPS, TS_SPEED_RANGE_MPS
from ..certificate.escape_angle import HULL_CONTACT_M, R_MIN_M

D_ENCOUNTER_M = (2000.0 * K_V, 3500.0 * K_V)
EGO_SPEED_MPS = (3.0 * K_V, SPEED_MAX_MPS)
HEADING_NOISE = 0.05
SPEED_NOISE_MPS = 0.1
GOAL_HALF_LENGTH_M = 200.0 * K_V
GOAL_HALF_WIDTH_M = 30.0
AREA_HALF_WIDTH_M = 4000.0
AREA_MARGIN_M = 2000.0

V_LOW_MPS = 2.5 * K_V
V_HIGH_MPS = 8.0 * K_V
C_TERMINAL = {Terminal.TIME: -25.0, Terminal.AREA: -5.0, Terminal.GOAL: 50.0,
              Terminal.STOPPED: -40.0, Terminal.COLLISION: -50.0}
C_EMERGENCY = -0.5
C_REACH = 1.5
C_V = -2.0
C_DEVIATE = -0.001

ALPHA_X = 75.0
GAMMA_THETA_DYN = 1.0
MEYER_SCALE = 150.0 / D_SENSE_M
STARBOARD_SECTOR = 112.5 * math.pi / 180.0
ZETA = {"starboard": (0.007, 0.004, 0.05), "port": (0.009, 0.007, 0.005),
        "stern": (0.01, 0.007, 0.005)}


def colregs_penalty(ego, others):
    """Compute the COLREGs penalty from the targets in sensing range.

    Args:
        ego (tuple): Own state (x, y, speed, heading)
        others (list): Target states

    Returns:
        float: Penalty
    """
    r = 0.0
    for obs in others:
        dx, dy = obs[0] - ego[0], obs[1] - ego[1]
        d = math.hypot(dx, dy)
        if d > D_SENSE_M or d < 1e-9:
            continue
        phi = -math.remainder(math.atan2(dy, dx) - ego[3], TWO_PI)
        rvx = obs[2] * math.cos(obs[3]) - ego[2] * math.cos(ego[3])
        rvy = obs[2] * math.sin(obs[3]) - ego[2] * math.sin(ego[3])
        v_y = -(rvx * dx + rvy * dy) / d
        side = ("starboard" if 0.0 < phi < STARBOARD_SECTOR else
                "port" if -STARBOARD_SECTOR < phi <= 0.0 else "stern")
        z_x, z_in, z_out = ZETA[side]
        z_v = z_in if v_y >= 0 else z_out
        raw = ALPHA_X * math.exp(-z_x * d * MEYER_SCALE + z_v * v_y)
        r -= raw / (1.0 + math.exp(GAMMA_THETA_DYN * abs(phi)))
    return r


def step_vessel(s, a, w):
    """Move a vessel one step with bounded turn rate.

    Args:
        s (tuple): State (x, y, speed, heading)
        a (float): Acceleration, m/s2
        w (float): Turn rate, rad/s

    Returns:
        tuple: New state
    """
    x, y, v, th = s
    lim = abs(v) / R_MIN_M
    return (x + SIM_DT_S * v * math.cos(th), y + SIM_DT_S * v * math.sin(th),
            min(v + SIM_DT_S * a, SPEED_MAX_MPS), th + SIM_DT_S * min(max(w, -lim), lim))


class KrasEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self):
        """Set up the observation and action spaces."""
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (N_OBS,), np.float32)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self._mask = None

    def _scenario(self):
        """Draw a random critical two vessel encounter and a goal.

        Returns:
            tuple: Own state, target state and goal
        """
        rng = self.np_random
        th0 = rng.uniform(-math.pi, math.pi)
        v_ego = rng.uniform(*EGO_SPEED_MPS)
        d_enc = rng.uniform(*D_ENCOUNTER_M)
        t_meet = d_enc / v_ego
        meet = (d_enc * math.cos(th0), d_enc * math.sin(th0))
        th_obs = th0 + rng.uniform(-math.pi, math.pi)
        v_obs = rng.uniform(*TS_SPEED_RANGE_MPS)
        obs = (meet[0] - v_obs * t_meet * math.cos(th_obs),
               meet[1] - v_obs * t_meet * math.sin(th_obs),
               v_obs + rng.uniform(-SPEED_NOISE_MPS, SPEED_NOISE_MPS),
               th_obs + rng.uniform(-HEADING_NOISE, HEADING_NOISE))
        goal = (D_GOAL_M * math.cos(th0), D_GOAL_M * math.sin(th0))
        return (0.0, 0.0, v_ego, th0), obs, goal

    def reset(self, *, seed=None, options=None):
        """Draw encounters until one starts outside emergency, then reset the episode.

        Args:
            seed (int): Random seed, or None
            options (dict): Unused

        Returns:
            tuple: Observation and empty info
        """
        super().reset(seed=seed)
        while True:
            self.ego, self.obs, self.goal = self._scenario()
            gap = math.hypot(self.obs[0] - self.ego[0], self.obs[1] - self.ego[1])
            if gap >= D_ENCOUNTER_M[0] and not is_emergency(self.ego, self.obs):
                break
        self.start = (0.0, 0.0)
        self.k = 0
        self.control = (0.0, 0.0)
        self.shield = Shield()
        self.prev_range = {}
        self._mask = None
        return observe(self.ego, self.control, self.start, self.goal, self.k, {1: self.obs},
                       self.prev_range), {}

    def action_masks(self):
        """Get the shield mask of allowed actions, cached per step.

        Returns:
            ndarray: Allowed action flags
        """
        if self._mask is None:
            self._mask = self.shield.mask(self.ego, {1: self.obs})
        return self._mask

    def _terminal(self):
        """Find whether the episode ended and why.

        Returns:
            Terminal: End reason, or None
        """
        d_long, d_lat = path_frame(self.ego, self.start, self.goal)
        if math.hypot(self.obs[0] - self.ego[0], self.obs[1] - self.ego[1]) < HULL_CONTACT_M:
            return Terminal.COLLISION
        if abs(d_long - D_GOAL_M) <= GOAL_HALF_LENGTH_M and abs(d_lat) <= GOAL_HALF_WIDTH_M:
            return Terminal.GOAL
        if self.ego[2] <= SPEED_MIN_MPS:
            return Terminal.STOPPED
        if (abs(d_lat) > AREA_HALF_WIDTH_M or d_long < -AREA_MARGIN_M
                or d_long > D_GOAL_M + AREA_MARGIN_M):
            return Terminal.AREA
        if self.k >= K_MAX:
            return Terminal.TIME
        return None

    def step(self, action):
        """Apply one shielded action and compute the reward.

        Args:
            action (int): Action index

        Returns:
            tuple: Observation, reward, done, truncated and info
        """
        action = int(action)
        mask = self.action_masks()
        assert mask[action], "action outside the shield's mask"
        before = math.hypot(self.goal[0] - self.ego[0], self.goal[1] - self.ego[1])
        self.control = self.shield.control(action, self.ego, {1: self.obs})
        self.ego = step_vessel(self.ego, *self.control)
        self.obs = _ahead(self.obs, SIM_DT_S)
        self.k += 1
        self._mask = None

        term = self._terminal()
        after = math.hypot(self.goal[0] - self.ego[0], self.goal[1] - self.ego[1])
        _, d_lat = path_frame(self.ego, self.start, self.goal)
        v = self.ego[2]
        r = C_TERMINAL.get(term, 0.0) + (C_EMERGENCY if action == EMERGENCY else 0.0)
        r += colregs_penalty(self.ego, [self.obs])
        r += C_REACH * (before - after)
        r += C_V * (v - V_HIGH_MPS) if v > V_HIGH_MPS else (C_V * (V_LOW_MPS - v) if v < V_LOW_MPS else 0.0)
        r += C_DEVIATE * min(abs(d_lat), D_HULL_M)

        ob = observe(self.ego, self.control, self.start, self.goal, self.k, {1: self.obs},
                     self.prev_range, term)
        done = term is not None and term is not Terminal.TIME
        truncated = term is Terminal.TIME
        return ob, float(r), done, truncated, {"terminal": None if term is None else term.name}

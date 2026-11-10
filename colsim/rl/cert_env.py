from __future__ import annotations

import copy
import functools
import math
import random
import weakref
from collections import namedtuple
from enum import Enum

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError("cert_env needs gymnasium: pip install gymnasium") from exc

from ..certificate import behaviour as _behaviour
from ..certificate.behaviour import MOTION_CLASSES as _behaviour_class_names
from ..certificate import alpha as _alpha
from ..evaluation import bank as _bank
from ..certificate import duty as _duty
from ..certificate import monitor as _monitor
from ..certificate import tube as _tube
from ..certificate.escape_angle import HULL_CONTACT_M, Vessel, _wrap, cpa
from ..simulator.vessels.boat import Boat
from ..control.cbf_filter import SafetyFilter
from ..simulator.experiment_constants import (
    MAP_HALF_EXTENT_M,
    SIM_DT_S,
    SPEED_MAX_MPS,
    SPEED_MIN_MPS,
    SURGE_ACCEL_MAX_MPS2,
    SURGE_DECEL_MAX_MPS2,
    RuleMode,
    apply_capability,
)

from ..certificate.escape_angle import R_MIN_M

GOAL_DISTANCE_M = float(MAP_HALF_EXTENT_M)

from .cert_obs import (
    ACTION_SHIELD,
    ALPHA_COL_RELEASE,
    MARGIN_NORM_M,
    MARGIN_STEPS,
    MAX_STEPS,
    N_BINS,
    N_TRACK,
    RANGE_NORM_M,
    RELEASE_LOOKAHEAD,
    SHIELD_GRID,
    SHIELD_MARGIN_M,
    TCPA_NORM_S,
    VETO_MARGIN_M,
    _bin_fields,
    _bin_fields_tube,
    _bin_mids,
    _lawful_signed,
    _nb_observe_core,
    _seg_pair,
    decode_action,
    margin_channel,
    observe,
    rate_margin,
    seg_min,
    shielded_yaw,
)

GAMMA = 0.99
DUTY_TOL_RAD = 1e-9
FROZEN_DUTY_ROLE = {"Head_on": "HO", "Crossing_give_way": "CGW", "Overtaking_give_way": "OGW"}
W_ALPHA_COL = 1.0
W_ALPHA_PHYS = 0.25
W_SPEED = 0.0
SPEED_ACTION = False
BEHAVIOUR_OBS = False
R_CONTACT = -100.0
R_RETURN = +100.0
R_TIMEOUT = +100.0
CONTACT_M = None
MIN_HOLD_STEPS = 0
R_HOLD = 0.0
R_VIOLATION = 0.0
REWARD_SCALE = 0.1
HARD_FRAC = 0.0
HARD_LR = 0.1

_K_GAIN = 0.5
_EPS_M = 20

BehaviourSchedule = namedtuple("BehaviourSchedule", ["n_switches", "interval_steps"])


class ScheduleSource(Enum):

    ROW = "row"


class DeviantSource(Enum):

    ROW = "row"


DEVIANT_SALT = 20260929


class EpisodeStart(Enum):

    ROW = "row"
    ALARM = "alarm"


ALARM_MAX_DRAWS = 50
ALARM_TRIGGER_K = 3
SWITCH_PREMISES = ("compliant", "adversarial")


class OwnShipPresence(Enum):

    PRESENT = "present"
    ABSENT = "absent"


ABSENT_OFFSET_M = 1.0e6


def seg_min_pairs(p0, p1):
    """Find the closest distance between any two tracks, between steps too.

    Args:
        p0 (list): Positions of every vessel at step start
        p1 (list): Positions of every vessel at step end

    Returns:
        float: Closest distance, m
    """
    best = float("inf")
    n = len(p0)
    for i in range(1, n):
        for j in range(i + 1, n):
            best = min(best, _seg_pair(p0[i][0], p0[i][1], p1[i][0], p1[i][1],
                                       p0[j][0], p0[j][1], p1[j][0], p1[j][1]))
    return best


def reward_ordering_bound(gamma, r_timeout, max_steps, r_contact=R_CONTACT, r_return=R_RETURN):
    """Compute the range of hold costs that keeps surviving better than contact.

    Args:
        gamma (float): Discount factor
        r_timeout (float): Survival reward
        max_steps (int): Episode length, steps
        r_contact (float): Contact reward
        r_return (float): Handback reward

    Returns:
        dict: Low and high bounds, ok flag and tie value
    """
    g, T = float(gamma), int(max_steps)
    s = T if g == 1.0 else (1.0 - g ** T) / (1.0 - g)
    last = g ** (T - 1)
    if s <= 1.0:
        return {"lo": 0.0, "hi": float("inf"), "ok": True, "tie": r_timeout == r_return}
    hi = (last * float(r_timeout) - float(r_contact)) / (s - 1.0)
    lo = (last * float(r_timeout) - float(r_return)) / (s - 1.0)
    return {"lo": lo, "hi": hi, "ok": True, "tie": float(r_timeout) == float(r_return)}


def check_reward_ordering(r_hold, gamma, r_timeout, max_steps, *, strict=False):
    """Check that the hold cost lies inside the reward ordering bound.

    Args:
        r_hold (float): Per step hold cost
        gamma (float): Discount factor
        r_timeout (float): Survival reward
        max_steps (int): Episode length, steps
        strict (bool): Whether to raise instead of warn

    Returns:
        dict: Bounds and ok flag
    """
    b = reward_ordering_bound(gamma, r_timeout, max_steps)
    b["ok"] = b["lo"] < float(r_hold) < b["hi"]
    if not b["ok"]:
        msg = (
            f"r_hold={r_hold} is outside the interval that keeps the discounted outcome ordering "
            f"return > timeout > contact at gamma={gamma}, r_timeout={r_timeout}, "
            f"max_steps={max_steps}: the admissible range is ({b['lo']:.2f}, {b['hi']:.2f}). "
            f"Above the upper bound the reward pays the policy to end the episode in contact. "
            f"See cert_env.reward_ordering_bound."
        )
        if strict:
            raise ValueError(msg)
        print(f"[cert_env] WARNING: {msg}")
    if b["tie"]:
        print(f"[cert_env] NOTE: r_timeout == R_RETURN == {r_timeout}, so a release on the final "
              f"step and a timeout score identically (the ADR 0004 tie).")
    return b


class CertifiedEscapeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        bank,
        *,
        gate=None,
        strata=("certified", "pre_certified"),
        dt=SIM_DT_S,
        max_steps=MAX_STEPS,
        horizon=_monitor.DECISION_HORIZON_S,
        algo="RRT*",
        gamma=GAMMA,
        seed=None,
        n_bins=N_BINS,
        n_track=N_TRACK,
        bin_fields=False,
        tubes=False,
        contact_m=CONTACT_M,
        hard_frac=HARD_FRAC,
        hard_lr=HARD_LR,
        ts_classes=None,
        episode_start=EpisodeStart.ROW,
        alarm_trigger_k=ALARM_TRIGGER_K,
        os_algo=None,
        os_rule_mode=RuleMode.COLREGS,
        os_safety_filter=SafetyFilter.NONE,
        speed_control=False,
        speed_action=SPEED_ACTION,
        behaviour_obs=BEHAVIOUR_OBS,
        score_after_handback=True,
        tail_steps=None,
        premises=("compliant", "adversarial"),
        r_hold=R_HOLD,
        r_timeout=R_TIMEOUT,
        strict_reward_ordering=False,
        margin_steps=MARGIN_STEPS,
        min_hold_steps=MIN_HOLD_STEPS,
        alpha_col_release=ALPHA_COL_RELEASE,
        release_lookahead=RELEASE_LOOKAHEAD,
        action_shield=ACTION_SHIELD,
        shield_margin=SHIELD_MARGIN_M,
        veto_margin=VETO_MARGIN_M,
        shield_pick=None,
        r_violation=R_VIOLATION,
        release_action=False,
        rule17b_relax=False,
        r17_p_on=_behaviour.P_ON,
        r17_p_off=_behaviour.P_OFF,
        r17_n_hysteresis=_behaviour.N_HYSTERESIS,
        hybrid_policy=None,
        p_switch=0.0,
        trigger_k=None,
        max_emergency_steps=None,
        emergency_avoidance=True,
        capability=None,
        behaviour_schedule=None,
        own_ship=OwnShipPresence.PRESENT,
        deviant=None,
    ):
        """Set up the emergency control environment over a bank of encounters.

        Args:
            bank (object): Bank CSV path or list of rows
            gate (float): Duty gate, m, or None
            strata (tuple): Bank strata to keep
            dt (float): Time step, s
            max_steps (int): Episode length, steps
            horizon (float): Certificate horizon, s
            algo (str): Target planner
            gamma (float): Discount factor
            seed (int): Random seed, or None
            n_bins (int): Number of heading bins
            n_track (int): Number of tracked targets
            bin_fields (bool): Whether to add the per bin fields
            tubes (bool): Whether to use reachable tubes
            contact_m (float): Contact distance, m
            hard_frac (float): Share of episodes drawn from failed rows
            hard_lr (float): Update rate of the row failure weights
            ts_classes (dict): Target behaviour mix, or None
            episode_start (EpisodeStart): Where the episode starts
            alarm_trigger_k (int): Lead cap of the alarm pre roll, steps
            os_algo (str): Own planner, or None for the target planner
            os_rule_mode (RuleMode): Own rule mode
            os_safety_filter (SafetyFilter): Own safety filter
            speed_control (bool): Whether vessels control speed
            speed_action (bool): Whether the policy sets speed
            behaviour_obs (bool): Whether to add behaviour class probabilities
            score_after_handback (bool): Whether to run the episode on after handback
            tail_steps (int): Steps run after handback, or None
            premises (tuple): Behaviour premises of the switch
            r_hold (float): Per step hold cost
            r_timeout (float): Survival reward
            strict_reward_ordering (bool): Whether a bad reward ordering raises
            margin_steps (int): Steps of the margin channel, 0 for none
            min_hold_steps (int): Min emergency steps before release
            alpha_col_release (float): Lawful width share needed to release
            release_lookahead (int): Steps replayed before a release
            action_shield (int): Steps of the action shield, 0 for none
            shield_margin (float): Separation the shield needs, m
            veto_margin (float): Separation a release needs, m
            shield_pick (int): Steps used to rank shield rates, or None
            r_violation (float): Per rule violation cost
            release_action (bool): Whether the policy asks for release
            rule17b_relax (bool): Whether Rule 17(b) relaxation is on
            r17_p_on (float): Relaxation on threshold
            r17_p_off (float): Relaxation off threshold
            r17_n_hysteresis (int): Relaxation hysteresis, steps
            hybrid_policy (object): Policy run by the targets, or None
            p_switch (float): Premise share that must trap own vessel
            trigger_k (int): Lead cap of the switch, steps, or None
            max_emergency_steps (int): Release cap, steps, or None
            emergency_avoidance (bool): Whether the in extremis override is on
            capability (dict): Capability knobs per vessel, or None
            behaviour_schedule (object): Behaviour switch schedule, or None
            own_ship (OwnShipPresence): Whether targets see own vessel
            deviant (object): Deviant target settings, or None
        """
        super().__init__()
        rows = _bank.load(bank) if isinstance(bank, str) else list(bank)
        self.rows = [r for r in rows if r["stratum"] in strata]
        if not self.rows:
            raise ValueError(f"no bank rows in strata {strata}")
        gates = {r["gate"] for r in self.rows}
        if gate is not None and gates and str(gate) not in {str(g) for g in gates}:
            raise ValueError(
                f"env gate {gate} does not match the bank's {gates} — the "
                f"episode would start and end under different rulebooks"
            )
        self.gate = gate
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.horizon = float(horizon)
        self.algo = algo
        self.gamma = float(gamma)
        self.n_bins = int(n_bins)
        self.n_track = int(n_track)
        self.bin_fields = bool(bin_fields)
        self.tubes = bool(tubes)
        self.contact_m = float(HULL_CONTACT_M if contact_m is None else contact_m)
        self.hard_frac = float(hard_frac)
        self.hard_lr = float(hard_lr)
        self.ts_classes = dict(ts_classes) if ts_classes else None
        self._classes = []
        self._aims = []
        self.behaviour_schedule = behaviour_schedule
        self.own_ship = own_ship
        self.deviant = deviant
        self._deviant_class = ""
        self._erratic = []
        self._rng_dev = np.random.default_rng(DEVIANT_SALT)
        self._plan = []
        self._seg = []
        self._starts = []
        self._dsafe0 = []
        self.os_algo = os_algo or algo
        self.episode_start = EpisodeStart(episode_start)
        self.alarm_trigger_k = int(alarm_trigger_k)
        self._t0 = 0
        self.os_rule_mode = os_rule_mode
        self.os_safety_filter = os_safety_filter
        self.speed_control = bool(speed_control)
        self.speed_action = bool(speed_action)
        self.behaviour_obs = bool(behaviour_obs)
        if self.behaviour_obs and self.bin_fields:
            raise ValueError(
                "behaviour_obs needs the per-threat blocks to attach to; bin_fields has none"
            )
        self._classifier = _behaviour.BehaviourClassifier() if self.behaviour_obs else None
        self.score_after_handback = bool(score_after_handback)
        self.tail_steps = None if tail_steps is None else int(tail_steps)
        self.premises = tuple(premises)
        self.r_hold = float(r_hold)
        self.r_timeout = float(r_timeout)
        self.reward_bound = check_reward_ordering(
            self.r_hold, self.gamma, self.r_timeout, self.max_steps,
            strict=bool(strict_reward_ordering),
        )
        self.margin_steps = int(margin_steps)
        self.min_hold_steps = int(min_hold_steps)
        self.alpha_col_release = float(alpha_col_release)
        self.release_lookahead = int(release_lookahead)
        self.action_shield = int(action_shield)
        self.shield_margin = float(shield_margin)
        self.veto_margin = float(veto_margin)
        self.shield_pick = None if shield_pick is None else int(shield_pick)
        self.r_violation = float(r_violation)
        self.release_action = bool(release_action)
        self.rule17b_relax = bool(rule17b_relax)
        self.hybrid_policy = hybrid_policy
        self.p_switch = float(p_switch)
        self.trigger_k = None if trigger_k is None else int(trigger_k)
        self.max_emergency_steps = max_emergency_steps
        self.emergency_avoidance = bool(emergency_avoidance)
        self.capability = dict(capability) if capability else None
        apply_capability([], self.capability)
        self._relaxer = (
            _behaviour.Rule17Relaxation(p_on=r17_p_on, p_off=r17_p_off,
                                        n_hysteresis=r17_n_hysteresis)
            if self.rule17b_relax else None
        )
        self._relaxed = None
        self._flip = 0.0

        self.action_space = spaces.Box(
            -1.0, 1.0,
            shape=(1 + int(self.speed_action) + int(self.release_action),),
            dtype=np.float32,
        )
        if self.bin_fields:
            obs_dim = 4 * self.n_bins + 5
        else:
            obs_dim = (2 * self.n_bins + 6 * self.n_track + 5
                       + (self.n_bins if self.margin_steps > 0 else 0)
                       + (len(_behaviour.CLASSES) * self.n_track if self.behaviour_obs else 0))
        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self._boats = []
        self._step = 0
        self._phi = 0.0
        self._broken_sum = 0
        self._broken_steps = 0
        self._permit_steps = 0
        self._ask_steps = 0
        self._veto_steps = 0
        self._shield_steps = 0
        self._replay = False
        self._tail_probe = None
        self._last_cpa = float("inf")
        self.ep_cpa = float("inf")
        self._last_ts_cpa = float("inf")
        self.ep_ts_cpa = float("inf")
        self._row = None
        self._row_fail = np.zeros(len(self.rows), dtype=np.float64)
        self._row_i = 0
        self._rules = [
            ["Rules :"],
            ["Finish OT"],
            ["Left OT"],
            ["Right OT"],
            ["R to R"],
        ]

    def _make_boats(self, row):
        """Build the simulator vessels of a bank row.

        Args:
            row (dict): Bank row

        Returns:
            list: Boats, own first
        """
        boats = []
        for i, v in enumerate([row["os"]] + list(row["threats"])):
            x = v.x + (ABSENT_OFFSET_M if i == 0 and self.own_ship is OwnShipPresence.ABSENT
                       else 0.0)
            goal = [
                x + GOAL_DISTANCE_M * math.cos(v.theta),
                v.y + GOAL_DISTANCE_M * math.sin(v.theta),
            ]
            b = Boat(
                i,
                x,
                v.y,
                v.v,
                v.theta,
                algo=(self.os_algo if i == 0 else self.algo),
                goal=goal,
                map_size=int(MAP_HALF_EXTENT_M),
            )
            b.map_size = int(MAP_HALF_EXTENT_M)
            b.max_rot_rad_s = v.v / R_MIN_M
            boats.append(b)
        for b in boats:
            b.sea_objects = boats
            b.emergency_avoidance = self.emergency_avoidance
        boats[0].set_rule_mode(self.os_rule_mode)
        boats[0].safety_filter = self.os_safety_filter
        apply_capability(boats, self.capability)
        if self.os_algo == "hybrid":
            os_b = boats[0]
            os_b._tail_host = weakref.ref(self)
            os_b.hybrid_certificate = True
            os_b.hybrid_policy = self.hybrid_policy
            os_b.hybrid_gate = self.gate
            os_b.hybrid_horizon = self.horizon
            os_b.hybrid_alpha_col_release = self.alpha_col_release
            os_b.hybrid_release_lookahead = self.release_lookahead
            os_b.hybrid_veto_margin = self.veto_margin
            os_b.hybrid_action_shield = self.action_shield
            os_b.hybrid_shield_margin = self.shield_margin
            os_b.hybrid_shield_pick = self.shield_pick
            os_b.hybrid_margin_steps = self.margin_steps
            os_b.hybrid_n_track = self.n_track
            os_b.hybrid_speed_action = self.speed_action
            os_b.hybrid_rule17b_relax = self.rule17b_relax
            os_b.hybrid_bin_fields = self.bin_fields
            os_b.hybrid_p_switch = self.p_switch
            os_b.hybrid_trigger_k = self.trigger_k
            os_b.hybrid_max_emergency_steps = self.max_emergency_steps
        if self.speed_control:
            from ..control.speed_governor import SpeedGovernor

            boats[0]._speed_gov = SpeedGovernor(v_nominal=boats[0].v)
        return boats

    def _draw_classes(self, n):
        """Draw a behaviour class for each target.

        Args:
            n (int): Number of targets

        Returns:
            list: Class names
        """
        if not self.ts_classes:
            return ["compliant"] * n
        names = list(_behaviour_class_names)
        p = np.array([self.ts_classes.get(c, 0.0) for c in names], dtype=float)
        p = p / p.sum()
        return [names[int(i)] for i in self._rng.choice(len(names), size=n, p=p)]

    def _draw_schedule(self, classes):
        """Draw the behaviour switch plan of each target.

        Args:
            classes (list): Start class per target

        Returns:
            list: Segments per target
        """
        plan = [[(0, c)] for c in classes]
        sch = self.behaviour_schedule
        if sch is None or sch.n_switches <= 0:
            return plan
        names = list(_behaviour_class_names)
        mix = self.ts_classes or {"compliant": 1.0}
        for segs in plan:
            for k in range(1, int(sch.n_switches) + 1):
                prev = segs[-1][1]
                p = np.array([0.0 if c == prev else mix.get(c, 0.0) for c in names], dtype=float)
                if p.sum() <= 0.0:
                    break
                segs.append((k * int(sch.interval_steps),
                             names[int(self._rng.choice(len(names), p=p / p.sum()))]))
        return plan

    def _late_holds(self, j, b):
        """Check whether a late acting target still holds its course.

        Args:
            j (int): Target index
            b (Boat): Target boat

        Returns:
            bool: True while it holds
        """
        if self._starts[j] is None:
            worst = b._get_worst_cpa(self._boats)
            if worst is None or not (worst[0] < b.T_ALERT and worst[1] < b.D_SAFE_MULT * b.r):
                return False
            self._starts[j] = self._step
        return self._step - self._starts[j] < _behaviour.LATE_ACTION_STEPS

    def _apply_schedule(self):
        """Switch target behaviours that reach a new plan segment."""
        os0 = None
        for j, (b, segs) in enumerate(zip(self._boats[1:], self._plan)):
            k = self._seg[j]
            if k + 1 >= len(segs) or self._step < segs[k + 1][0]:
                continue
            k += 1
            while k + 1 < len(segs) and self._step >= segs[k + 1][0]:
                k += 1
            old, new = self._classes[j], segs[k][1]
            self._seg[j], self._classes[j], self._starts[j] = k, new, self._step
            if new == "late":
                self._starts[j] = None
            if old == "wide_berth":
                b.D_SAFE_MULT = self._dsafe0[j]
            if new == "wide_berth":
                b.D_SAFE_MULT = _behaviour.WIDE_BERTH_MULT
            if new == "interceptor":
                os0 = os0 or Vessel.from_sea_object(self._os, r_min=R_MIN_M)
                self._aims[j] = _monitor.route_intercept_point(
                    Vessel.from_sea_object(b, r_min=R_MIN_M), os0)

    @property
    def _os(self):
        """Get own boat.

        Returns:
            Boat: Own boat
        """
        return self._boats[0]

    def _vessels(self):
        """Get own vessel and targets as certificate vessels.

        Returns:
            tuple: Own vessel and targets
        """
        os_v = Vessel.from_sea_object(self._os, r_min=R_MIN_M)
        ts = [Vessel.from_sea_object(b, r_min=R_MIN_M) for b in self._boats[1:]]
        return os_v, ts

    def _os_memory(self):
        """Get the stored overtaking role toward each target.

        Returns:
            list: Stored roles
        """
        if not getattr(self, "_boats", None):
            return []
        self._os.update_overtaking(self._boats)
        return [self._os.overtaking.get(b.mmsi) for b in self._boats[1:]]

    def _update_relaxation(self, os_v, threats):
        """Update the Rule 17 relaxation flags.

        Args:
            os_v (Vessel): Own vessel
            threats (list): Target vessels
        """
        if self._relaxer is None:
            self._relaxed, self._flip = None, 0.0
            return
        flags = self._relaxer.update(self._step * self.dt, os_v, threats)
        self._relaxed = flags
        self._flip = 1.0 if any(flags) else 0.0

    def _observe(self, os_v, threats):
        """Build the policy observation of the current state.

        Args:
            os_v (Vessel): Own vessel
            threats (list): Target vessels

        Returns:
            tuple: Observation, physical width, lawful width and max heading change
        """
        post = None
        if self._classifier is not None:
            self._classifier.observe(self._step * self.dt, os_v, threats)
            post = self._classifier.as_array(os_v, threats)
        return observe(
            os_v,
            threats,
            gate=self.gate,
            n_bins=self.n_bins,
            elapsed_frac=(self._step - self._t0) / self.max_steps,
            relaxed=self._relaxed,
            flip_engaged=self._flip,
            margin_steps=self.margin_steps,
            dt=self.dt,
            n_track=self.n_track,
            bin_fields=self.bin_fields,
            tubes=self._tubes(threats),
            posterior=post,
            memory=self._os_memory(),
        )

    def _potential(self, a_col, a_phys, k0, v):
        """Compute the shaping potential from escape widths and speed.

        Args:
            a_col (float): Lawful width, rad
            a_phys (float): Physical width, rad
            k0 (float): Max heading change, rad
            v (float): Speed, m/s

        Returns:
            float: Potential
        """
        norm = (1.0 / k0) if k0 else 0.0
        return (W_ALPHA_COL * a_col * norm
                + W_ALPHA_PHYS * a_phys * norm
                + W_SPEED * (v / SPEED_MAX_MPS))

    def _rules_broken(self, os_v, threats, delta):
        """Count the duties a heading change breaks.

        Args:
            os_v (Vessel): Own vessel
            threats (list): Target vessels
            delta (float): Heading change, rad

        Returns:
            int: Number of broken duties
        """
        k = _duty.kappa(os_v)
        frozen = self._frozen_duty()
        memory = self._os_memory()
        n = 0
        for i, ts in enumerate(threats):
            r = frozen.get(i) or _duty.role(os_v, ts, gate=self.gate,
                                            memory=memory[i] if i < len(memory) else None)
            rx = (bool(self._relaxed[i])
                  if self._relaxed is not None and i < len(self._relaxed) else False)
            if not any(lo - DUTY_TOL_RAD <= delta <= hi + DUTY_TOL_RAD
                       for lo, hi in _duty.duty_intervals(r, k, relaxed=rx)):
                n += 1
        return n

    def _frozen_duty(self):
        """Get the role own planner froze during avoidance.

        Returns:
            dict: Role per target index
        """
        if not getattr(self, "_boats", None):
            return {}
        os_b = self._os
        if getattr(os_b, "nav_phase", None) != "AVOIDANCE":
            return {}
        role = FROZEN_DUTY_ROLE.get(getattr(os_b, "frozen_role", None))
        threat = getattr(os_b, "threat_ts", None)
        if role is None or threat is None:
            return {}
        idx = next((i for i, b in enumerate(self._boats[1:]) if b is threat), None)
        return {} if idx is None else {idx: role}

    def _asks_release(self, action):
        """Check whether the action asks to hand back control.

        Args:
            action (ndarray): Policy action, or None

        Returns:
            bool: True if release is asked
        """
        if not self.release_action or action is None:
            return True
        a = np.asarray(action, dtype=float).ravel()
        i = 1 + int(self.speed_action)
        return True if a.size <= i else bool(a[i] > 0.0)

    def open_loop(self, yaw, speed=1.0, ask_release=False):
        """Build an action from a turn, a speed and a release flag.

        Args:
            yaw (float): Turn command in [-1, 1]
            speed (float): Speed command in [-1, 1]
            ask_release (bool): Release flag

        Returns:
            ndarray: Action
        """
        a = [float(yaw)]
        if self.speed_action:
            a.append(float(speed))
        if self.release_action:
            a.append(1.0 if ask_release else -1.0)
        return np.asarray(a, dtype=np.float32)

    def _draw_row(self):
        """Draw a bank row, favouring failed rows when hard sampling is on.

        Returns:
            int: Row index
        """
        n = len(self.rows)
        if self.hard_frac <= 0.0 or self._rng.random() >= self.hard_frac:
            return int(self._rng.integers(n))
        total = float(self._row_fail.sum())
        if total <= 0.0:
            return int(self._rng.integers(n))
        return int(self._rng.choice(n, p=self._row_fail / total))

    def reset(self, *, seed=None, options=None):
        """Start an episode on a bank row.

        Args:
            seed (int): Random seed, or None
            options (dict): Row, start mode and draw count, or None

        Returns:
            tuple: Observation and info
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            random.seed(seed)
        idx = (options or {}).get("row")
        self._row_i = int(idx) if idx is not None else self._draw_row()
        self._row = self.rows[self._row_i]
        self._boats = self._make_boats(self._row)
        row_plan = self._row.get("plan")
        self._classes = ([segs[0][1] for segs in row_plan] if row_plan
                         else self._draw_classes(len(self._boats) - 1))
        row_seed = int(self._row.get("seed", self._row_i) or 0)
        self._rng_dev = np.random.default_rng([row_seed, DEVIANT_SALT])
        self._deviant_class = ""
        if self.deviant is DeviantSource.ROW and self._classes:
            names = _behaviour.DEVIANT_CLASSES
            self._deviant_class = names[int(self._rng_dev.integers(len(names)))]
            self._classes[0] = self._deviant_class
        self._erratic = [None] * len(self._classes)
        os0 = Vessel.from_sea_object(self._boats[0], r_min=R_MIN_M)
        self._aims = [
            _monitor.route_intercept_point(Vessel.from_sea_object(b, r_min=R_MIN_M), os0)
            if c == "interceptor" else None
            for b, c in zip(self._boats[1:], self._classes)
        ]
        if row_plan:
            self._plan = [list(segs) if self.behaviour_schedule is ScheduleSource.ROW
                          else [tuple(segs[0])] for segs in row_plan]
        elif self.behaviour_schedule is ScheduleSource.ROW:
            raise ValueError(f"row {self._row.get('idx')} has no plan to play")
        else:
            self._plan = self._draw_schedule(self._classes)
        if self._deviant_class:
            self._plan[0] = [(0, self._deviant_class)]
        self._seg = [0] * len(self._classes)
        self._starts = [None if c == "late" else 0 for c in self._classes]
        self._dsafe0 = [b.D_SAFE_MULT for b in self._boats[1:]]
        for b, c in zip(self._boats[1:], self._classes):
            if c == "wide_berth":
                b.D_SAFE_MULT = _behaviour.WIDE_BERTH_MULT
        self._step = 0
        self._broken_sum = 0
        self._broken_steps = 0
        self._permit_steps = 0
        self._ask_steps = 0
        self._veto_steps = 0
        self._shield_steps = 0
        self._replay = False
        self._tail_probe = None
        self._last_cpa = self._raw_min()
        self.ep_cpa = self._last_cpa
        pts = [(b.x, b.y) for b in self._boats]
        self._last_ts_cpa = seg_min_pairs(pts, pts)
        self.ep_ts_cpa = self._last_ts_cpa
        if self._relaxer is not None:
            self._relaxer.reset()
        if self._classifier is not None:
            self._classifier.reset()
        self._t0 = 0
        rolled = self.episode_start is EpisodeStart.ALARM
        if rolled and not self._roll_to_alarm():
            draws = (options or {}).get("_alarm_draws", 0) + 1
            if draws >= ALARM_MAX_DRAWS:
                raise RuntimeError(f"no alarm in {ALARM_MAX_DRAWS} far-start draws")
            return self.reset(options={"_alarm_draws": draws})
        os_v, ts = self._vessels()
        if not rolled:
            self._update_relaxation(os_v, ts)
        obs, a_phys, a_col, k0 = self._observe(os_v, ts)
        self._phi = self._potential(a_col, a_phys, k0, os_v.v)
        return obs, {
            "stratum": self._row["stratum"],
            "idx": self._row["idx"],
            "alpha_col": a_col,
            "alpha_phys": a_phys,
        }

    def _roll_to_alarm(self):
        """Run the compliant planner until the switch would engage.

        Returns:
            bool: True if the alarm fired in the window
        """
        limit = int(self._row.get("window_steps") or self.max_steps)
        while self._step < limit:
            os_v, ts = self._vessels()
            self._update_relaxation(os_v, ts)
            pred = _monitor.predict(os_v, ts, gate=self.gate, horizon=self.horizon, rng=self._rng,
                                    premises=SWITCH_PREMISES, n_perturb=0,
                                    rule17b_relax=self.rule17b_relax, relaxed_now=self._relaxed,
                                    memory=self._os_memory())
            engage = pred["fires_now"] or (
                pred["p_fire"] > 0.0
                and float(pred["t_first_min"]) <= self.alarm_trigger_k * self.dt)
            if engage:
                self._t0 = self._step
                self._last_cpa = self._raw_min()
                self.ep_cpa = self._last_cpa
                pts = [(b.x, b.y) for b in self._boats]
                self._last_ts_cpa = seg_min_pairs(pts, pts)
                self.ep_ts_cpa = self._last_ts_cpa
                return True
            if self._classifier is not None:
                self._classifier.observe(self._step * self.dt, os_v, ts)
            self._apply_action(None)
            self._step += 1
            if self._min_separation() < self.contact_m:
                return False
        return False

    def _move_colsim(self, b):
        """Move one boat one step with its COLSim planner.

        Args:
            b (Boat): Boat
        """
        b.move(
            True,
            self._boats,
            [x.mmsi for x in self._boats],
            self._rules,
            None,
            None,
            _EPS_M,
            int(MAP_HALF_EXTENT_M),
            _K_GAIN,
            self.dt,
            False,
        )

    def tail_replayer(self):
        """Get a function that replays the next step on copies of the boats.

        Returns:
            function: Step function on a boat list
        """
        ghost = copy.copy(self)
        ghost._classes, ghost._aims, ghost._seg = list(self._classes), list(self._aims), list(self._seg)
        ghost._starts, ghost._erratic = list(self._starts), list(self._erratic)
        ghost._rng, ghost._rng_dev = copy.deepcopy(self._rng), copy.deepcopy(self._rng_dev)
        ghost._replay = True

        def step(boats):
            """Replay one step on the given boats.

            Args:
                boats (list): Boat copies
            """
            ghost._boats = boats
            ghost._apply_action(None)
            ghost._step += 1
        return step

    def _apply_action(self, action):
        """Apply the policy action, or the planner if none, and move every boat.

        Args:
            action (ndarray): Policy action, or None
        """
        self._apply_schedule()
        p0 = [(b.x, b.y) for b in self._boats]
        os_b = self._os
        if action is None:
            self._move_colsim(os_b)
        else:
            rate = os_b.max_rot_rad_s or (os_b.v / R_MIN_M)
            acc, yaw = decode_action(action, v=os_b.v, dt=self.dt, yaw_max=rate,
                                     speed_action=self.speed_action)
            if self.action_shield > 0:
                yaw = self._shielded_yaw(os_b, yaw, rate)
            os_b.update(np.array([[acc], [yaw]]), self.dt)
        for j, (b, cls) in enumerate(zip(self._boats[1:], self._classes)):
            if cls in ("compliant", "wide_berth"):
                self._move_colsim(b)
            elif cls == "late":
                if self._late_holds(j, b):
                    b.update_overtaking(self._boats)
                    b.update(np.array([[0.0], [0.0]]), self.dt)
                else:
                    self._move_colsim(b)
            elif cls == "mistaken":
                theta0 = b.theta
                self._move_colsim(b)
                if getattr(b, "nav_phase", "") == "AVOIDANCE":
                    b.theta = b._normalize_ang(theta0 - _wrap(b.theta - theta0))
            elif cls == "erratic":
                st = self._erratic[j]
                if st is None or self._step >= st[0]:
                    lo, hi = _behaviour.ERRATIC_INTERVAL_STEPS
                    turn = self._rng_dev.uniform(-_behaviour.ERRATIC_MAX_TURN_RAD,
                                                 _behaviour.ERRATIC_MAX_TURN_RAD)
                    st = (self._step + int(self._rng_dev.integers(lo, hi + 1)),
                          float(b.theta + turn), float(b.theta))
                    self._erratic[j] = st
                rate = b.max_rot_rad_s or (b.v / R_MIN_M)
                yaw = float(np.clip(_wrap(st[1] - b.theta) / self.dt, -rate, rate))
                b.update(np.array([[0.0], [yaw]]), self.dt)
            elif cls in ("adversarial", "interceptor"):
                if cls == "adversarial":
                    want = _monitor.intercept_heading(
                        Vessel.from_sea_object(b, r_min=R_MIN_M),
                        Vessel.from_sea_object(os_b, r_min=R_MIN_M),
                    )
                else:
                    px, py = self._aims[j] or (os_b.x, os_b.y)
                    want = float(np.arctan2(py - b.y, px - b.x))
                rate = b.max_rot_rad_s or (b.v / R_MIN_M)
                b.update(
                    np.array(
                        [
                            [0.0],
                            [
                                float(
                                    np.clip(
                                        _wrap(want - b.theta) / self.dt, -rate, rate
                                    )
                                )
                            ],
                        ]
                    ),
                    self.dt,
                )
            else:
                b.update(np.array([[0.0], [0.0]]), self.dt)
        p1 = [(b.x, b.y) for b in self._boats]
        self._last_cpa = seg_min(p0, p1)
        self._last_ts_cpa = seg_min_pairs(p0, p1)
        self.ep_ts_cpa = min(self.ep_ts_cpa, self._last_ts_cpa)
        if not self._replay:
            self.ep_cpa = min(self.ep_cpa, self._last_cpa)

    def _rate_margin(self, os_b, yaw, n):
        """Find the smallest separation if own vessel holds a turn rate.

        Args:
            os_b (Boat): Own boat
            yaw (float): Turn rate, rad/s
            n (int): Number of steps

        Returns:
            float: Smallest separation, m
        """
        return rate_margin(os_b, self._boats[1:], yaw, n, self.dt)

    def _rate_is_safe(self, os_b, yaw, n):
        """Check that a turn rate keeps the contact distance.

        Args:
            os_b (Boat): Own boat
            yaw (float): Turn rate, rad/s
            n (int): Number of steps

        Returns:
            bool: True if safe
        """
        return self._rate_margin(os_b, yaw, n) >= self.contact_m

    def _tubes(self, threats):
        """Get the target tubes if tubes are on.

        Args:
            threats (list): Target vessels

        Returns:
            list: Tubes, or None
        """
        if not self.tubes:
            return None
        return _tube.tubes_for(None, threats, r_min=R_MIN_M)

    def _shielded_yaw(self, os_b, yaw, rate):
        """Run the action shield on a turn rate.

        Args:
            os_b (Boat): Own boat
            yaw (float): Proposed turn rate, rad/s
            rate (float): Max turn rate, rad/s

        Returns:
            float: Turn rate to apply
        """
        out, fired = shielded_yaw(os_b, self._boats[1:], yaw, rate, self.dt,
                                  fire_n=self.action_shield, margin=self.shield_margin,
                                  pick_n=self.shield_pick,
                                  tubes=self._tubes([Vessel.from_sea_object(b, r_min=R_MIN_M)
                                                     for b in self._boats[1:]]))
        self._shield_steps += 1 if fired else 0
        return out

    def _raw_min(self):
        """Get the current distance to the closest target.

        Returns:
            float: Distance, m
        """
        return min(
            math.hypot(b.x - self._os.x, b.y - self._os.y) for b in self._boats[1:]
        )

    def _min_separation(self):
        """Get the closest separation so far.

        Returns:
            float: Separation, m
        """
        return self._last_cpa

    def _run_out_after_handback(self):
        """Run the planner after handback and track the closest separation.

        Returns:
            tuple: Contact flag, steps run and closest separation
        """
        n = 0
        worst = self._min_separation()
        limit = self._t0 + self.max_steps
        if self.tail_steps is not None:
            limit = min(limit, self._step + self.tail_steps)
        while self._step + n < limit:
            self._apply_action(None)
            n += 1
            sep = self._min_separation()
            worst = min(worst, sep)
            if sep < self.contact_m:
                return True, n, worst
        return False, n, worst

    def _tail_is_clear(self, k):
        """Replay the next steps on copies to check a release is safe.

        Args:
            k (int): Number of steps

        Returns:
            bool: True if no contact
        """
        snapshot = copy.deepcopy(self._boats)
        schedule_state = (list(self._classes), list(self._aims), list(self._seg),
                          list(self._starts), list(self._erratic))
        py_state, np_state = random.getstate(), self._rng.bit_generator.state
        dev_state = self._rng_dev.bit_generator.state
        step0, cpa0 = self._step, self._last_cpa
        limit = min(self._t0 + self.max_steps, self._step + int(k))
        clear = True
        worst = self._min_separation()
        self._tail_probe = None
        self._replay = True
        try:
            while self._step < limit:
                self._apply_action(None)
                self._step += 1
                sep = self._min_separation()
                worst = min(worst, sep)
                if sep < self.veto_margin:
                    clear = False
                    break
        finally:
            self._replay = False
        if clear:
            self._tail_probe = (step0, limit, self._step - step0, worst)
        self._boats = snapshot
        self._classes, self._aims, self._seg, self._starts, self._erratic = schedule_state
        self._step, self._last_cpa = step0, cpa0
        random.setstate(py_state)
        self._rng.bit_generator.state = np_state
        self._rng_dev.bit_generator.state = dev_state
        return clear

    def _cached_tail(self):
        """Reuse the release probe result if it still covers this step.

        Returns:
            tuple: Contact flag, steps and separation, or None
        """
        probe, self._tail_probe = self._tail_probe, None
        if probe is None or self.veto_margin < self.contact_m:
            return None
        step0, limit, n, worst = probe
        run_out = (self._t0 + self.max_steps if self.tail_steps is None
                   else min(self._t0 + self.max_steps, self._step + self.tail_steps))
        if step0 != self._step or limit != run_out:
            return None
        return False, n, worst

    def step(self, action):
        """Apply one action and compute reward, end flags and info.

        Args:
            action (ndarray): Policy action

        Returns:
            tuple: Observation, reward, terminated, truncated and info
        """
        theta0 = self._os.theta
        self._apply_action(action)
        self._step += 1

        os_v, ts = self._vessels()
        self._update_relaxation(os_v, ts)
        obs, a_phys, a_col, k0 = self._observe(os_v, ts)

        delta = float(np.clip(_wrap(self._os.theta - theta0) / self.dt * _duty.HORIZON_S,
                              -k0, k0))
        broken = self._rules_broken(os_v, ts, delta)
        self._broken_sum += broken
        self._broken_steps += 1 if broken else 0

        phi = self._potential(a_col, a_phys, k0, os_v.v)
        reward = self.gamma * phi - self._phi - self.r_hold - self.r_violation * broken
        self._phi = phi

        info = {
            "alpha_col": a_col,
            "alpha_phys": a_phys,
            "min_separation": self._min_separation(),
            "rules_broken": broken,
        }
        terminated, truncated = False, False

        if info["min_separation"] < self.contact_m:
            reward += R_CONTACT
            terminated, info["outcome"] = True, "contact"
        else:
            pred = _monitor.predict(
                os_v,
                ts,
                gate=self.gate,
                horizon=self.horizon,
                rng=self._rng,
                premises=self.premises,
                n_perturb=0,
                rule17b_relax=self.rule17b_relax,
                relaxed_now=self._relaxed,
                memory=self._os_memory(),
            )
            info["certified"] = bool(pred["fires_now"])
            info["p_fire"] = pred["p_fire"]
            permitted = (not pred["fires_now"] and pred["p_fire"] == 0.0
                         and a_col >= self.alpha_col_release
                         and self._step - self._t0 >= self.min_hold_steps)
            asked = self._asks_release(action)
            if permitted and asked and self.release_lookahead > 0:
                permitted = self._tail_is_clear(self.release_lookahead)
                self._veto_steps += 0 if permitted else 1
            info["permitted"], info["asked"] = bool(permitted), bool(asked)
            self._permit_steps += 1 if permitted else 0
            self._ask_steps += 1 if asked else 0
            if permitted and asked:
                info["handback_step"] = self._step - self._t0
                if not self.score_after_handback:
                    tail_contact, tail_steps, tail_min = False, 0, info["min_separation"]
                else:
                    cached = self._cached_tail()
                    tail_contact, tail_steps, tail_min = (
                        cached if cached is not None else self._run_out_after_handback())
                    if cached is not None:
                        self.ep_cpa = min(self.ep_cpa, tail_min)
                info["tail_steps"] = tail_steps
                info["min_separation"] = min(info["min_separation"], tail_min)
                if tail_contact:
                    reward += R_CONTACT
                    terminated, info["outcome"] = True, "contact_after_handback"
                else:
                    reward += R_RETURN
                    terminated, info["outcome"] = True, "return"
            elif self._step - self._t0 >= self.max_steps:
                reward += self.r_timeout
                terminated, info["outcome"] = True, "timeout"

        if terminated or truncated:
            reward -= self.gamma * phi
            info["rules_broken_sum"] = self._broken_sum
            info["violation_steps"] = self._broken_steps
            info["permit_steps"] = self._permit_steps
            info["ask_steps"] = self._ask_steps
            info["idx"] = self._row["idx"]
            info["veto_steps"] = self._veto_steps
            info["shield_steps"] = self._shield_steps
            info["ep_cpa"] = self.ep_cpa
            info["ep_ts_cpa"] = self.ep_ts_cpa
            info["row"] = self._row_i
            failed = 1.0 if self.ep_cpa < self.contact_m else 0.0
            self._row_fail[self._row_i] = (
                (1.0 - self.hard_lr) * self._row_fail[self._row_i] + self.hard_lr * failed
            )

        return obs, float(reward * REWARD_SCALE), terminated, truncated, info

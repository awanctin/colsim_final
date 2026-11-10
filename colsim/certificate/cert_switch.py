from __future__ import annotations

import copy
import json
import os
import random

import numpy as np

from . import behaviour as _behaviour
from . import monitor as _monitor
from .escape_angle import Vessel
from ..rl.cert_obs import (
    ACTION_SHIELD,
    ALPHA_COL_RELEASE,
    MARGIN_STEPS,
    MAX_STEPS,
    N_BINS,
    N_TRACK,
    R_MIN_M,
    RELEASE_LOOKAHEAD,
    SHIELD_MARGIN_M,
    VETO_MARGIN_M,
    decode_action,
    observe,
    seg_min,
    shielded_yaw,
)
from ..simulator.experiment_constants import DUTY_GATE_M, SIM_DT_S

DEFAULT_GATE = DUTY_GATE_M
RELEASE_CAP_STEPS = 2
COMPLIANT = "COMPLIANT"
EMERGENCY = "EMERGENCY"


class CertificateSwitch:

    def __init__(
        self,
        policy=None,
        *,
        gate=DEFAULT_GATE,
        horizon=_monitor.DECISION_HORIZON_S,
        p_switch=0.0,
        trigger_k=None,
        n_bins=N_BINS,
        bin_fields=False,
        r_min=R_MIN_M,
        dt=SIM_DT_S,
        dists=None,
        seed=0,
        verbose=False,
        use_commitment=True,
        alpha_col_release=ALPHA_COL_RELEASE,
        release_lookahead=RELEASE_LOOKAHEAD,
        veto_margin=VETO_MARGIN_M,
        action_shield=ACTION_SHIELD,
        shield_margin=SHIELD_MARGIN_M,
        shield_pick=None,
        margin_steps=MARGIN_STEPS,
        n_track=N_TRACK,
        speed_action=None,
        rule17b_relax=False,
        r17_p_on=_behaviour.P_ON,
        r17_p_off=_behaviour.P_OFF,
        r17_n_hysteresis=_behaviour.N_HYSTERESIS,
        max_emergency_steps=None,
        policy_max_steps=None,
    ):
        """Set up the switch between the compliant planner and the RL emergency policy.

        Args:
            policy (object): Trained policy or its path, or None
            gate (float): Distance under which a duty applies, m
            horizon (float): Look ahead time of the monitor, s
            p_switch (float): Fire probability above which to switch
            trigger_k (int): Max steps before the first breach to switch, or None
            n_bins (int): Number of bearing bins in the observation
            bin_fields (bool): Whether to use the binned field observation
            r_min (float): Min vessel radius, m
            dt (float): Time step, s
            dists (object): Target distributions for the monitor, or None
            seed (int): Random seed
            verbose (bool): Whether to print each decision
            use_commitment (bool): Whether the monitor keeps a committed manoeuvre
            alpha_col_release (float): Policy output above which a release is asked
            release_lookahead (int): Steps replayed to check a release is safe
            veto_margin (float): Extra distance for the action veto, m
            action_shield (int): Steps checked by the action shield, 0 to turn it off
            shield_margin (float): Extra distance for the action shield, m
            shield_pick (int): Number of candidate turns for the shield, or None
            margin_steps (int): Steps of margin added to the observation
            n_track (int): Number of tracked targets in the observation
            speed_action (bool): Whether the policy also controls speed, or None to read the config
            rule17b_relax (bool): Whether to relax Rule 17(b) on a stand on vessel
            r17_p_on (float): Probability to turn the relaxation on
            r17_p_off (float): Probability to turn the relaxation off
            r17_n_hysteresis (int): Steps of hysteresis for the relaxation
            max_emergency_steps (int): Max steps in emergency mode, or None
            policy_max_steps (int): Episode length used in training, or None to read the config
        """
        speed_action = speed_action_for(policy, speed_action)
        self.policy_max_steps = policy_max_steps_for(policy, policy_max_steps)
        self.policy = _load_policy(policy)
        self.max_emergency_steps = (
            None if max_emergency_steps is None else int(max_emergency_steps)
        )
        self.gate = gate
        self.horizon = float(horizon)
        self.p_switch = float(p_switch)
        self.trigger_k = None if trigger_k is None else int(trigger_k)
        self.n_bins = int(n_bins)
        self.bin_fields = bool(bin_fields)
        self.r_min = float(r_min)
        self.dt = float(dt)
        self.dists = dists
        self.verbose = bool(verbose)
        self.use_commitment = bool(use_commitment)
        self.alpha_col_release = float(alpha_col_release)
        self.release_lookahead = int(release_lookahead)
        self.veto_margin = float(veto_margin)
        self.action_shield = int(action_shield)
        self.shield_margin = float(shield_margin)
        self.shield_pick = None if shield_pick is None else int(shield_pick)
        self.margin_steps = int(margin_steps)
        self.n_track = int(n_track)
        self.rule17b_relax = bool(rule17b_relax)
        self._relaxer = (
            _behaviour.Rule17Relaxation(
                p_on=r17_p_on, p_off=r17_p_off, n_hysteresis=r17_n_hysteresis
            )
            if self.rule17b_relax
            else None
        )
        self._rng = np.random.default_rng(seed)

        want = (
            (4 * self.n_bins + 5,)
            if self.bin_fields
            else (
                2 * self.n_bins
                + 6 * self.n_track
                + 5
                + (self.n_bins if self.margin_steps > 0 else 0),
            )
        )
        space = getattr(self.policy, "observation_space", None)
        if space is not None and tuple(space.shape) != want:
            raise ValueError(
                f"policy expects observations of shape {tuple(space.shape)}, but this switch "
                f"produces {want} (n_bins={self.n_bins}, bin_fields={self.bin_fields}, "
                f"n_track={self.n_track}, margin_steps={self.margin_steps}). Either the cell "
                f"was not threaded through to the switch, or this is a different policy "
                f"generation — the Discrete(3)/15-dim Phase-3 model belongs to "
                f"switching_fsm.SwitchingFSM, not here."
            )
        act = getattr(self.policy, "action_space", None)
        shape = None if act is None else (tuple(getattr(act, "shape", ()) or ()))
        self.speed_action = bool(speed_action)
        n_yaw_speed = 1 + int(self.speed_action)
        if shape is not None and shape not in ((n_yaw_speed,), (n_yaw_speed + 1,)):
            raise ValueError(
                f"policy action space {act} is neither the Box({n_yaw_speed}) command this "
                f"switch decodes at speed_action={self.speed_action} (ADR 0007) nor the "
                f"Box({n_yaw_speed + 1}) that adds the release request (ADR 0010)."
            )
        self._release_dim = shape == (n_yaw_speed + 1,)
        self.reset()

    def reset(self):
        """Go back to compliant mode and reset every counter."""
        if self._relaxer is not None:
            self._relaxer.reset()
        self.state = COMPLIANT
        self.steps = 0
        self.emergency_steps = 0
        self.fire_steps = 0
        self.n_switches = 0
        self.first_engage_step = None
        self.veto_steps = 0
        self.shield_steps = 0
        self.excursion_steps = 0
        self.cap_releases = 0
        self.cap_overruns = 0
        self.cap_overrun_steps = 0
        self._replay_failed = False
        self.last = {}

    def _tail_is_clear(self, own_ship, sea_objects, tail_step):
        """Replay the next steps on copies to check a release is safe.

        Args:
            own_ship (SeaObject): Own vessel
            sea_objects (list): All sea objects
            tail_step (callable): Function that moves the copies one step

        Returns:
            bool: True if no contact happens in the replay
        """
        py_state, np_state = random.getstate(), self._rng.bit_generator.state
        npg_state = np.random.get_state()
        try:
            ghosts = copy.deepcopy(list(sea_objects))
        except Exception as exc:
            if not self._replay_failed:
                self._replay_failed = True
                print(
                    f"[cert_switch] tail replay unavailable ({type(exc).__name__}: {exc}); "
                    f"refusing every release rather than granting an uncertified one"
                )
            return False
        idx = next((i for i, o in enumerate(sea_objects) if o is own_ship), 0)
        clear = True
        try:
            for _ in range(self.release_lookahead):
                p0 = [(o.x, o.y) for o in ghosts]
                tail_step(ghosts)
                p1 = [(o.x, o.y) for o in ghosts]
                order = [idx] + [i for i in range(len(ghosts)) if i != idx]
                if (
                    seg_min([p0[i] for i in order], [p1[i] for i in order])
                    < self.veto_margin
                ):
                    clear = False
                    break
        finally:
            random.setstate(py_state)
            self._rng.bit_generator.state = np_state
            np.random.set_state(npg_state)
        return clear

    def _within_lead(self, pred):
        """Check the predicted breach is close enough to act now.

        Args:
            pred (dict): Monitor prediction

        Returns:
            bool: True if the breach is within trigger_k steps
        """
        if self.trigger_k is None:
            return True
        return float(pred["t_first_min"]) <= self.trigger_k * self.dt

    def _cap_reached(self):
        """Check the emergency excursion hit its step limit.

        Returns:
            bool: True if the limit is reached
        """
        return (
            self.max_emergency_steps is not None
            and self.excursion_steps >= self.max_emergency_steps
        )

    def _elapsed_frac(self, k):
        """Compute the share of the training episode already used.

        Args:
            k (int): Steps spent in emergency mode

        Returns:
            float: Share between 0 and 1
        """
        return min(max(k, 0), self.policy_max_steps) / self.policy_max_steps

    def decide(self, own_ship, sea_objects, tail_step=None):
        """Decide the mode for this step and, in emergency mode, the policy command.

        Args:
            own_ship (SeaObject): Own vessel
            sea_objects (list): All sea objects
            tail_step (callable): Function that moves copies one step, needed for releases

        Returns:
            dict: Mode, fire flags and, in emergency mode, acceleration and turn rate
        """
        threats = [o for o in sea_objects if o is not own_ship]
        os_v = Vessel.from_sea_object(own_ship, r_min=self.r_min)
        ts_v = [Vessel.from_sea_object(o, r_min=self.r_min) for o in threats]

        if not ts_v:
            self.state = COMPLIANT
            self.steps += 1
            self.last = {"state": COMPLIANT, "fires_now": False, "p_fire": 0.0}
            return dict(self.last)

        relaxed = None
        if self._relaxer is not None:
            relaxed = self._relaxer.update(self.steps * self.dt, os_v, ts_v)

        memory = None
        if hasattr(own_ship, "update_overtaking"):
            own_ship.update_overtaking(sea_objects)
            memory = [own_ship.overtaking.get(o.mmsi) for o in threats]

        pred = _monitor.predict(
            own_ship,
            ts_v,
            gate=self.gate,
            horizon=self.horizon,
            rule17b_relax=self.rule17b_relax,
            relaxed_now=relaxed,
            premises=("compliant", "adversarial"),
            dists=self.dists,
            rng=self._rng,
            use_commitment=self.use_commitment,
            memory=memory,
        )
        fires_now, p_fire = bool(pred["fires_now"]), float(pred["p_fire"])
        if fires_now or p_fire > 0.0:
            self.fire_steps += 1

        obs = a_phys = a_col = k0 = action = None
        if self.state == COMPLIANT:
            if fires_now or (p_fire > self.p_switch and self._within_lead(pred)):
                self.state = EMERGENCY
                self.n_switches += 1
                if self.first_engage_step is None:
                    self.first_engage_step = self.steps
                self.excursion_steps = 0
        elif not fires_now and p_fire == 0.0:
            obs, a_phys, a_col, k0 = observe(
                os_v,
                ts_v,
                gate=self.gate,
                n_bins=self.n_bins,
                elapsed_frac=self._elapsed_frac(self.excursion_steps),
                relaxed=relaxed,
                flip_engaged=(1.0 if relaxed and any(relaxed) else 0.0),
                margin_steps=self.margin_steps,
                dt=self.dt,
                n_track=self.n_track,
                bin_fields=self.bin_fields,
                memory=memory,
            )
            ask = True
            if self._release_dim and self.policy is not None:
                action, _ = self.policy.predict(obs, deterministic=True)
                ask = bool(np.asarray(action, dtype=float).ravel()[-1] > 0.0)
            voluntary = a_col >= self.alpha_col_release and ask
            permitted = voluntary or self._cap_reached()
            if permitted and self.release_lookahead > 0:
                if tail_step is None:
                    raise ValueError(
                        "release_lookahead > 0 needs a `tail_step` callable to replay the "
                        "compliant tail on; `_move_hybrid_cert` supplies one. Without it the "
                        "switch would release on the instantaneous rule while reporting that it "
                        "certified the tail (ADR 0011)."
                    )
                permitted = self._tail_is_clear(own_ship, sea_objects, tail_step)
                self.veto_steps += 0 if permitted else 1
            if permitted:
                self.state = COMPLIANT
                self.cap_releases += 0 if voluntary else 1

        out = {
            "state": self.state,
            "fires_now": fires_now,
            "p_fire": p_fire,
            "t_first_min": float(pred["t_first_min"]),
            "committed": bool(pred["committed"]),
        }

        if self.state == EMERGENCY:
            self.emergency_steps += 1
            if self._cap_reached():
                self.cap_overruns += (
                    1 if self.excursion_steps == self.max_emergency_steps else 0
                )
                self.cap_overrun_steps += 1
            self.excursion_steps += 1
            if obs is None:
                obs, a_phys, a_col, k0 = observe(
                    os_v,
                    ts_v,
                    gate=self.gate,
                    n_bins=self.n_bins,
                    elapsed_frac=self._elapsed_frac(self.excursion_steps - 1),
                    relaxed=relaxed,
                    flip_engaged=(1.0 if relaxed and any(relaxed) else 0.0),
                    margin_steps=self.margin_steps,
                    dt=self.dt,
                    n_track=self.n_track,
                    bin_fields=self.bin_fields,
                    memory=memory,
                )
            out.update(alpha_phys=a_phys, alpha_col=a_col, kappa0=k0)
            if self.policy is not None:
                if action is None:
                    action, _ = self.policy.predict(obs, deterministic=True)
                yaw_max = getattr(own_ship, "max_rot_rad_s", None) or (
                    own_ship.v / self.r_min
                )
                accel, yaw = decode_action(
                    action,
                    v=own_ship.v,
                    dt=self.dt,
                    yaw_max=yaw_max,
                    speed_action=self.speed_action,
                )
                if self.action_shield > 0:
                    yaw, fired = shielded_yaw(
                        own_ship,
                        threats,
                        yaw,
                        yaw_max,
                        self.dt,
                        fire_n=self.action_shield,
                        margin=self.shield_margin,
                        pick_n=self.shield_pick,
                    )
                    self.shield_steps += 1 if fired else 0
                out.update(accel=accel, yaw=yaw)

        self.steps += 1
        self.last = out
        if self.verbose:
            print(
                f"[cert_switch] {self.state} fires_now={fires_now} p_fire={p_fire:.2f}"
                + (
                    f" yaw={np.degrees(out['yaw']):+.1f}deg/s accel={out['accel']:+.3f}"
                    if "yaw" in out
                    else ""
                )
            )
        return out


def speed_action_for(policy, speed_action=None):
    """Find whether the policy also controls speed.

    Args:
        policy (object): Policy or its path
        speed_action (bool): Value forced by the caller, or None

    Returns:
        bool: True if the policy controls speed
    """
    if speed_action is not None:
        return bool(speed_action)
    if isinstance(policy, str):
        cfg = os.path.join(os.path.dirname(os.path.abspath(policy)), "config.json")
        try:
            with open(cfg, encoding="utf-8") as fh:
                return bool(json.load(fh).get("speed_action", False))
        except (OSError, ValueError):
            pass
    return False


def policy_max_steps_for(policy, max_steps=None):
    """Find the episode length used to train the policy.

    Args:
        policy (object): Policy or its path
        max_steps (int): Value forced by the caller, or None

    Returns:
        int: Episode length, steps
    """
    if max_steps is not None:
        return int(max_steps)
    if isinstance(policy, str):
        cfg = os.path.join(os.path.dirname(os.path.abspath(policy)), "config.json")
        try:
            with open(cfg, encoding="utf-8") as fh:
                value = json.load(fh).get("max_steps")
            if value is not None:
                return int(value)
        except (OSError, ValueError):
            pass
    return MAX_STEPS


def _load_policy(policy):
    """Load the policy from disk when given a path.

    Args:
        policy (object): Policy or its path

    Returns:
        object: Loaded policy, or the input unchanged
    """
    if policy is None or not isinstance(policy, str):
        return policy
    from ..rl.policy_io import load_policy

    return load_policy(policy, device="cpu")

from __future__ import annotations

import math
from typing import List, NamedTuple, Optional, Tuple

from ..simulator.experiment_constants import (
    SURGE_TIME_CONSTANT_S,
    SPEED_MAX_MPS,
    SPEED_MIN_MPS,
    SURGE_ACCEL_MAX_MPS2,
    SURGE_DECEL_MAX_MPS2,
    SURGE_REVERSE_DECEL_MPS2,
    CRI_ARM,
    CRI_STOP,
    CRI_REV,
    T_HORIZON_S,
    T_STOP_S,
    T_REV_S,
    DV_MIN_APPARENT_MPS,
    T_DWELL_S,
    CLEAR_RANGE_M,
    ONE_NAUTICAL_MILE_M,
    CRI_W_DCPA,
    CRI_W_TCPA,
)

_SAFE_DCPA_M: float = ONE_NAUTICAL_MILE_M


class _TargetInfo(NamedTuple):

    dcpa: float
    tcpa: float
    range_m: float
    range_rate: float


TargetInfo = _TargetInfo


class SpeedGovernor:

    _GIVE_WAY_ROLES = frozenset(("Crossing_give_way", "Head_on", "Overtaking_give_way"))
    _STAND_ON_ROLES = frozenset(("Crossing_stand_on", "Being_overtaken_stand_on"))

    def __init__(
        self,
        v_nominal: float,
        *,
        tau_v: float = SURGE_TIME_CONSTANT_S,
        v_max: float = SPEED_MAX_MPS,
        v_min: float = SPEED_MIN_MPS,
        a_acc_max: float = SURGE_ACCEL_MAX_MPS2,
        a_dec_max: float = SURGE_DECEL_MAX_MPS2,
        enable_reverse: bool = False,
        a_rev_max: float = SURGE_REVERSE_DECEL_MPS2,
        cri_arm: float = CRI_ARM,
        cri_stop: float = CRI_STOP,
        cri_rev: float = CRI_REV,
        t_horizon: float = T_HORIZON_S,
        t_stop: float = T_STOP_S,
        t_rev: float = T_REV_S,
        dv_min_apparent: float = DV_MIN_APPARENT_MPS,
        t_dwell: float = T_DWELL_S,
        clear_range_m: float = CLEAR_RANGE_M,
    ) -> None:
        """Set the speed limits, risk thresholds and timing of the governor.

        Args:
            v_nominal (float): Cruise speed, m/s
            tau_v (float): Speed time constant, s
            v_max (float): Max speed, m/s
            v_min (float): Min speed, m/s
            a_acc_max (float): Max acceleration, m/s2
            a_dec_max (float): Max deceleration, negative, m/s2
            enable_reverse (bool): Whether going astern is allowed
            a_rev_max (float): Max astern deceleration, m/s2
            cri_arm (float): Risk to start slowing down
            cri_stop (float): Risk to stop
            cri_rev (float): Risk to go astern
            t_horizon (float): Look ahead time, s
            t_stop (float): TCPA under which to stop, s
            t_rev (float): TCPA under which to go astern, s
            dv_min_apparent (float): Smallest speed change worth doing, m/s
            t_dwell (float): Min time between two changes, s
            clear_range_m (float): Range at which a passed target is clear, m
        """

        self.v_nominal = v_nominal
        self.tau_v = tau_v
        self.v_max = v_max
        self.v_min = v_min
        self.a_acc_max = a_acc_max
        self.a_dec_max = a_dec_max
        self.enable_reverse = enable_reverse
        self.a_rev_max = a_rev_max

        self.cri_arm = cri_arm
        self.cri_stop = cri_stop
        self.cri_rev = cri_rev
        self.t_horizon = t_horizon
        self.t_stop = t_stop
        self.t_rev = t_rev
        self.dv_min_apparent = dv_min_apparent
        self.t_dwell = t_dwell
        self.clear_range_m = clear_range_m

        self._v_cmd: float = v_nominal
        self._current_rung: int = 0
        self._dwell_elapsed: float = 0.0

    def accel_command(self, v: float, v_cmd: float) -> float:
        """Compute the acceleration toward the commanded speed.

        Args:
            v (float): Current speed, m/s
            v_cmd (float): Commanded speed, m/s

        Returns:
            float: Acceleration, m/s2
        """
        u0_unsat = (v_cmd - v) / self.tau_v
        return float(max(self.a_dec_max, min(self.a_acc_max, u0_unsat)))

    def envelope_clip(self, v: float) -> float:
        """Keep a speed within the allowed range.

        Args:
            v (float): Speed, m/s

        Returns:
            float: Clipped speed, m/s
        """
        v_floor = 0.0 if self._current_rung >= 2 else self.v_min
        if self.enable_reverse:
            v_floor = float("-inf")
        return float(max(v_floor, min(self.v_max, v)))

    def decide_speed(
        self,
        *,
        v: float,
        theta: float,
        role: Optional[str],
        targets: List[TargetInfo],
        j_cascade: float,
        dt: float,
    ) -> float:
        """Choose the commanded speed from the most risky target.

        Args:
            v (float): Current speed, m/s
            theta (float): Current heading, rad
            role (str): COLREGs role, or None
            targets (list): Target information
            j_cascade (float): Cascade risk score
            dt (float): Time step, s

        Returns:
            float: Commanded speed, m/s
        """

        self._dwell_elapsed += dt

        if not targets:
            return self._resume_cruise(v)

        worst = self._worst_target(targets)
        if worst is None:
            return self._resume_cruise(v)

        dcpa, tcpa, range_m, range_rate = worst
        cri = self._compute_cri(dcpa, tcpa)

        past_and_clear = tcpa < 0 and range_rate > 0.0 and range_m > self.clear_range_m
        if past_and_clear and self._current_rung > 0:
            return self._resume_cruise(v)

        is_armed = (
            cri >= self.cri_arm and 0 < tcpa <= self.t_horizon
        ) or j_cascade >= self.cri_stop
        if not is_armed:
            return self._resume_cruise(v)

        is_stand_on = role in self._STAND_ON_ROLES
        in_extremis = (
            cri >= self.cri_rev or (0 < tcpa <= self.t_rev) or j_cascade >= self.cri_rev
        )
        if is_stand_on and not in_extremis:
            return self._hold_v_cmd(v)

        target_rung = self._required_rung(cri, tcpa)

        if target_rung == 1:
            v_target = self._geometric_v_target(v, dcpa, tcpa)
            new_v_cmd = float(max(self.v_min, min(v_target, self.v_nominal)))
        elif target_rung == 2:
            new_v_cmd = 0.0
        elif target_rung == 3 and self.enable_reverse:
            new_v_cmd = -abs(self.v_min)
        else:
            new_v_cmd = 0.0 if target_rung >= 2 else self._v_cmd

        new_v_cmd = self._apply_antichatter(new_v_cmd, target_rung)

        return self._v_cmd

    def _compute_cri(self, dcpa: float, tcpa: float) -> float:
        """Compute the collision risk index from DCPA and TCPA.

        Args:
            dcpa (float): Distance at closest approach, m
            tcpa (float): Time to closest approach, s

        Returns:
            float: Risk index between 0 and 1
        """
        f_d = max(0.0, min(1.0, 1.0 - dcpa / _SAFE_DCPA_M))
        if tcpa > 0:
            f_t = max(0.0, min(1.0, 1.0 - tcpa / self.t_horizon))
        else:
            f_t = 1.0
        return CRI_W_DCPA * f_d + CRI_W_TCPA * f_t

    def _worst_target(
        self, targets: List[TargetInfo]
    ) -> Optional[Tuple[float, float, float, float]]:
        """Find the target with the highest risk inside the horizon.

        Args:
            targets (list): Target information

        Returns:
            tuple: DCPA, TCPA, range and range rate, or None
        """
        best = None
        best_cri = -1.0
        for t in targets:
            if t.tcpa <= 0 or t.tcpa > self.t_horizon:
                continue
            cri = self._compute_cri(t.dcpa, t.tcpa)
            if cri > best_cri:
                best_cri = cri
                best = (t.dcpa, t.tcpa, t.range_m, t.range_rate)
        return best

    def _required_rung(self, cri: float, tcpa: float) -> int:
        """Pick the response level from the risk and TCPA.

        Args:
            cri (float): Risk index
            tcpa (float): Time to closest approach, s

        Returns:
            int: 0 cruise, 1 slow, 2 stop, 3 astern
        """
        if cri >= self.cri_rev or (0 < tcpa <= self.t_rev):
            return 3
        if cri >= self.cri_stop or (0 < tcpa <= self.t_stop):
            return 2
        if cri >= self.cri_arm and tcpa > 0:
            return 1
        return 0

    def _geometric_v_target(self, v: float, dcpa: float, tcpa: float) -> float:
        """Compute the speed that gives a safe DCPA and can be reached in time.

        Args:
            v (float): Current speed, m/s
            dcpa (float): Distance at closest approach, m
            tcpa (float): Time to closest approach, s

        Returns:
            float: Target speed, m/s
        """
        if dcpa <= 0:
            return 0.0
        v_needed = v * (_SAFE_DCPA_M / max(dcpa, 1.0))
        v_needed = min(v_needed, self.v_nominal)

        if tcpa > 0:
            v_reachable = max(0.0, v + self.a_dec_max * tcpa)
        else:
            v_reachable = 0.0

        return max(0.0, min(v_needed, v_reachable))

    def _apply_antichatter(self, proposed_v_cmd: float, proposed_rung: int) -> float:
        """Accept a new command only if it is large enough or more urgent.

        Args:
            proposed_v_cmd (float): Proposed speed, m/s
            proposed_rung (int): Proposed response level

        Returns:
            float: Commanded speed, m/s
        """
        escalating = proposed_rung > self._current_rung
        change = abs(proposed_v_cmd - self._v_cmd)

        suppress = not escalating and (
            change < self.dv_min_apparent or self._dwell_elapsed < self.t_dwell
        )

        if not suppress:
            self._v_cmd = proposed_v_cmd
            self._current_rung = proposed_rung
            self._dwell_elapsed = 0.0

        return self._v_cmd

    def _resume_cruise(self, v: float) -> float:
        """Ask to go back to cruise speed.

        Args:
            v (float): Current speed, m/s, unused

        Returns:
            float: Commanded speed, m/s
        """
        return self._apply_antichatter(self.v_nominal, 0)

    def _hold_v_cmd(self, v: float) -> float:
        """Keep the current commanded speed.

        Args:
            v (float): Current speed, m/s, unused

        Returns:
            float: Commanded speed, m/s
        """
        self._dwell_elapsed += 0.0
        return self._v_cmd

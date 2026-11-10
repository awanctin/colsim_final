from __future__ import annotations

import argparse
import collections
import functools
import gc
import hashlib
import json
import os
import math
import time
from enum import Enum

import numpy as np

from ..evaluation import bank as _bank
from ..certificate.behaviour import class_mix
from ..evaluation import compliance_check as _compliance
from ..certificate import duty as _duty
from ..certificate.escape_angle import HULL_CONTACT_M, Vessel, cpa
from ..control.cbf_filter import SafetyFilter
from ..certificate.cert_switch import RELEASE_CAP_STEPS
from ..simulator.experiment_constants import RuleMode
from ..simulator.sea_object import PORT_VETO_GATE_M
from .cert_env import (
    MAX_STEPS, N_BINS, R_MIN_M, RELEASE_LOOKAHEAD, VETO_MARGIN_M, BehaviourSchedule,
    CertifiedEscapeEnv, DeviantSource, OwnShipPresence, ScheduleSource,
)
from .train_cert import UNIFORM_MIX, split_bank, stack_for_policy

ARMS = ("compliant", "compliant_speed", "hybrid", "letter", "policy")


class ContactAttribution(Enum):

    OFF = "off"
    COUNTERFACTUAL = "counterfactual"


class SeedSource(Enum):

    INDEX = "index"
    BANK = "bank"


class Population(Enum):

    HELDOUT = "heldout"
    ALL = "all"


class ComplianceCheck(Enum):

    OFF = "off"
    ON = "on"


class RunLength(Enum):

    WINDOW = "window"
    CLEAR = "clear"


FIRST_TURN_RAD = math.radians(2.0)
STOP_CLEAR_SEP_M = 500.0


def _fleet_state(base):
    """Get the position, heading and speed of every boat.

    Args:
        base (CertifiedEscapeEnv): Environment

    Returns:
        list: States, own first
    """
    return [(b.x, b.y, b.theta, b.v) for b in base._boats]


class Verdict(Enum):

    COMPLIANT = "compliant"
    VIOLATED = "violated"
    UNDECIDED_AT_CONTACT = "undecided_at_contact"
    UNDECIDED_AT_END = "undecided_at_end"


SENSITIVITY_CONE_M = PORT_VETO_GATE_M


class R6Cause(Enum):

    EXTREMIS = "extremis"
    CONFLICT = "conflict"
    OTHER = "other"
    UNLATCHED = "unlatched"


_STAND_ON = ("Crossing_stand_on", "Being_overtaken_stand_on")
_GIVE_WAY = ("Crossing_give_way", "Head_on", "Overtaking_give_way")
_R6_RANK = (R6Cause.OTHER, R6Cause.CONFLICT, R6Cause.EXTREMIS)


def _r6_update(base, flags):
    """Flag the targets whose speed broke Rule 6 at this step.

    Args:
        base (CertifiedEscapeEnv): Environment
        flags (dict): Rule 6 flags per target, updated in place
    """
    own = base._os
    sit = getattr(own, "locked_situations", {})
    threat = getattr(own, "threat_ts", None)
    for j, b in enumerate(base._boats[1:], 1):
        if sit.get(b.mmsi) not in _STAND_ON:
            continue
        cause = R6Cause.OTHER
        if any(sit.get(o.mmsi) in _GIVE_WAY for o in base._boats[1:] if o is not b):
            cause = R6Cause.CONFLICT
        if (getattr(own, "nav_phase", None) == "AVOIDANCE" and threat is b
                and not getattr(own, "frozen_role", None)):
            cause = R6Cause.EXTREMIS
        if j not in flags or _R6_RANK.index(cause) > _R6_RANK.index(flags[j]):
            flags[j] = cause


def _r6_causes(flags, violations):
    """Keep the Rule 6 violations that have a recorded cause.

    Args:
        flags (dict): Rule 6 flags per target
        violations (list): Violation labels rule:index

    Returns:
        list: Violations with their cause
    """
    out = []
    for v in violations:
        rule, j = v.split(":")
        if rule != "R6":
            continue
        out.append(flags.get(int(j), R6Cause.UNLATCHED).value)
    return out


def _verdict(violated, undecided, contact):
    """Turn violation and undecided lists into a verdict.

    Args:
        violated (list): Violated rules
        undecided (list): Undecided rules
        contact (bool): Whether the episode ended in contact

    Returns:
        Verdict: Verdict
    """
    if violated:
        return Verdict.VIOLATED
    if not undecided:
        return Verdict.COMPLIANT
    return Verdict.UNDECIDED_AT_CONTACT if contact else Verdict.UNDECIDED_AT_END


def _verdict_fields(ep, contact, prefix):
    """Build the verdict fields of own vessel and every vessel.

    Args:
        ep (dict): Compliance result of an episode
        contact (bool): Whether the episode ended in contact
        prefix (str): Field name prefix

    Returns:
        dict: Verdict fields
    """
    n = len(ep["vessels_compliant"])
    return {
        f"{prefix}own_verdict": _verdict(ep["own_violations"], ep["own_undecided"], contact).value,
        f"{prefix}vessels_verdict": [
            _verdict(not ep["vessels_compliant"][i], ep["vessels_undecided"][i], contact).value
            for i in range(n)],
    }


def _compliance_fields(trace, dt, contact=False):
    """Check the rules of an episode trace, normal and wide cone.

    Args:
        trace (list): Fleet states per step
        dt (float): Time step, s
        contact (bool): Whether the episode ended in contact

    Returns:
        dict: Compliance fields
    """
    ep = _compliance.check_episode(trace, dt)
    wide = _compliance.check_episode(trace, dt, cone_m=SENSITIVITY_CONE_M)
    return dict(
        kr_own_undecided=ep["own_undecided"],
        **_verdict_fields(ep, contact, "kr_"),
        kr500_own_violations=wide["own_violations"],
        kr500_own_undecided=wide["own_undecided"],
        **_verdict_fields(wide, contact, "kr500_"),
        kr_own_violations=ep["own_violations"],
        kr_own_compliant=ep["own_compliant"],
        kr_own_violations_raw=ep["own_violations_raw"],
        kr_own_compliant_raw=ep["own_compliant_raw"],
        kr_vessels_compliant=ep["vessels_compliant"],
        kr_undecided=ep["undecided"],
        woerner_own=[ep["woerner"][j] for j in sorted(ep["woerner"])],
        woerner_own_min=ep["woerner_min"],
    )


_ARM_KW = {
    "compliant": dict(os_algo="RRT*"),
    "compliant_speed": dict(os_algo="RRT*", speed_control=True),
    "hybrid": dict(os_algo="hybrid"),
    "letter": dict(os_algo="letter"),
    "policy": dict(os_algo="RRT*"),
    "relaxed": dict(os_algo="RRT*", os_rule_mode=RuleMode.RELAXED),
    "vo": dict(os_algo="VO"),
    "vo_relaxed": dict(os_algo="VO", os_rule_mode=RuleMode.RELAXED),
    "sbmpc": dict(os_algo="SB-MPC"),
    "psbmpc": dict(os_algo="PSB-MPC"),
    "rtampc": dict(os_algo="RTA-MPC"),
    "kras": dict(os_algo="KRAS"),
    "cbf": dict(os_algo="straight_goal", os_safety_filter=SafetyFilter.CBF),
    "rrt_cbf": dict(os_algo="RRT*", os_safety_filter=SafetyFilter.CBF),
}


def _force_release(action, base, t, force_release_step):
    """Set the release flag of the action from a given step on.

    Args:
        action (ndarray): Policy action, or None
        base (CertifiedEscapeEnv): Environment
        t (int): Current step
        force_release_step (int): Step to force release, or None

    Returns:
        ndarray: Action
    """
    if force_release_step is None or action is None:
        return action

    a = np.asarray(action, dtype=float).ravel().copy()
    i = 1 + int(getattr(base, "speed_action", False))
    if a.size <= i:
        return action

    a[i] = 1.0 if t >= force_release_step else -1.0
    return a.reshape(np.shape(action))


def _goal_track(base, x0, y0, theta0):
    """Compute own progress along and across the start heading.

    Args:
        base (CertifiedEscapeEnv): Environment
        x0 (float): Start x, m
        y0 (float): Start y, m
        theta0 (float): Start heading, rad

    Returns:
        tuple: Along and cross distances, m
    """
    os_b = base._os
    dx, dy = os_b.x - x0, os_b.y - y0
    along = dx * math.cos(theta0) + dy * math.sin(theta0)
    cross = -dx * math.sin(theta0) + dy * math.cos(theta0)
    return along, cross


def _switch_counters(base):
    """Read the certificate switch counters of own vessel.

    Args:
        base (CertifiedEscapeEnv): Environment

    Returns:
        dict: Counters, empty if no switch
    """
    sw = getattr(base._os, "_cert_switch", None)
    if sw is None:
        return {}
    return dict(
        switch_engagements=int(sw.n_switches),
        switch_emergency_steps=int(sw.emergency_steps),
        switch_fire_steps=int(sw.fire_steps),
        switch_first_engage_step=sw.first_engage_step,
        switch_veto_steps=int(sw.veto_steps),
        switch_shield_steps=int(sw.shield_steps),
        switch_cap_releases=int(sw.cap_releases),
        switch_cap_overruns=int(sw.cap_overruns),
        switch_cap_overrun_steps=int(sw.cap_overrun_steps),
    )


def _tt_duty(base, contact_m):
    """Name the roles of the closest target pair when they make contact.

    Args:
        base (CertifiedEscapeEnv): Environment
        contact_m (float): Contact distance, m

    Returns:
        str: Roles of both targets, or None
    """
    ts = [Vessel.from_sea_object(b, r_min=R_MIN_M) for b in base._boats[1:]]
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            tcpa, dcpa, _ = cpa(ts[i], ts[j])
            if dcpa < contact_m and 0.0 < tcpa <= _duty.HORIZON_S:
                return f"{_duty.role(ts[i], ts[j])}/{_duty.role(ts[j], ts[i])}"
    return None


def _counterfactual_tt(env_cf, row_i, seed, window):
    """Run a row without own vessel and keep the closest target separation.

    Args:
        env_cf (CertifiedEscapeEnv): Environment with own vessel absent
        row_i (int): Row index
        seed (int): Random seed
        window (int): Number of steps

    Returns:
        float: Closest separation, m
    """
    env_cf.reset(seed=seed, options={"row": row_i})
    best = float("inf")
    for _ in range(window):
        _obs, _r, _term, _trunc, info = env_cf.step(None)
        best = min(best, float(info.get("ep_ts_cpa", float("inf"))))
    return best


def run_row(env, row_i, arm, *, model=None, seed=0, window=MAX_STEPS,
            contact_m=HULL_CONTACT_M, force_release_step=None, env_cf=None,
            compliance=ComplianceCheck.OFF, run_length=RunLength.WINDOW):
    """Run one arm on one bank row and record the outcome.

    Args:
        env (CertifiedEscapeEnv): Environment
        row_i (int): Row index
        arm (str): Arm name
        model (object): Trained policy, or None
        seed (int): Random seed
        window (int): Number of steps
        contact_m (float): Contact distance, m
        force_release_step (int): Step to force release, or None
        env_cf (CertifiedEscapeEnv): Environment without own vessel, or None
        compliance (bool): Whether to check the rules
        run_length (object): Run length mode

    Returns:
        dict: Row record
    """
    obs, info0 = env.reset(seed=seed, options={"row": row_i})
    base = env.unwrapped
    situation = _situation(base)
    ts_class = (list(getattr(base, "_classes", [])) or [""])[0]
    contact = contact_in_control = False
    handback = None
    min_sep = float("inf")
    ts_min_sep = float("inf")
    os0 = base._os
    x0, y0, theta0 = os0.x, os0.y, os0.theta
    tt_duty, tt_duty_step = "", None
    trace = [_fleet_state(base)] if compliance is ComplianceCheck.ON else None
    r6_flags = {}
    cert_first_step = None
    theta_start = [b.theta for b in base._boats[1:]]
    first_turn_tcpa = [None] * len(theta_start)
    t = 0
    for t in range(1, window + 1):
        in_control = (arm == "policy" and handback is None)
        action = model.predict(obs, deterministic=True)[0] if in_control else None
        action = _force_release(action, base, t, force_release_step)
        obs, _r, _term, _trunc, info = env.step(action)
        min_sep = min(min_sep, info["min_separation"])
        ts_min_sep = min(ts_min_sep, float(info.get("ep_ts_cpa", float("inf"))))
        if trace is not None:
            trace.append(_fleet_state(base))
            _r6_update(base, r6_flags)
        if env_cf is not None and tt_duty_step is None:
            label = _tt_duty(base, contact_m)
            tt_duty, tt_duty_step = (label, t) if label else ("", None)
        if cert_first_step is None and info.get("certified"):
            cert_first_step = t
        _first_turns(base, theta_start, first_turn_tcpa)
        if info["min_separation"] < contact_m:
            contact, contact_in_control = True, in_control
            break
        if in_control and info.get("outcome") == "return":
            handback = t
        if run_length is RunLength.CLEAR and min_sep > STOP_CLEAR_SEP_M and _all_opening(base):
            break
    steps_held = (handback if handback is not None else t) if arm == "policy" else 0
    along, cross = _goal_track(base, x0, y0, theta0)
    comp = _compliance_fields(trace, base.dt, contact) if trace is not None else {}
    attribution = {}
    if env_cf is not None:
        cf = (_counterfactual_tt(env_cf, row_i, seed, window) if ts_min_sep < contact_m
              else float("inf"))
        attribution = dict(
            ts_min_sep_cf=cf,
            tt_attribution=("" if ts_min_sep >= contact_m
                            else "targets" if cf < contact_m else "own_ship"),
            tt_duty=tt_duty, tt_duty_step=tt_duty_step,
        )
    return dict(
        arm=arm, row=row_i, contact=contact, contact_in_control=contact_in_control,
        handback=handback, min_sep=min_sep, steps=t,
        along_track_m=along,
        cross_track_m=cross,
        forced_release=force_release_step,
        released=handback is not None,
        release_step=handback,
        held_steps=steps_held,
        timeout=bool(arm == "policy" and handback is None and not contact),
        override_steps=int(getattr(base, "_shield_steps", 0)),
        veto_steps=int(getattr(base, "_veto_steps", 0)),
        **_switch_counters(base),
        **({"ts_plan": base._plan} if getattr(base, "behaviour_schedule", None) else {}),
        ts_segments=list(getattr(base, "_seg", [])),
        ts_deviant=getattr(base, "_deviant_class", ""),
        idx=base._row.get("idx", ""),
        **attribution,
        **({"cbf_active_steps": int(base._os.cbf_active_steps),
            "cbf_infeasible_steps": int(base._os.cbf_infeasible_steps)}
           if base._os.safety_filter is SafetyFilter.CBF else {}),
        **comp,
        **({"kr_own_r6_causes": _r6_causes(r6_flags, comp["kr_own_violations"])} if comp else {}),
        **({"traj": [[[round(x), round(y)] for x, y, _h, _v in s] for s in trace]}
           if trace is not None else {}),
        situation=situation,
        ts_class=ts_class,
        ts_min_sep=ts_min_sep,
        stratum=info0.get("stratum", ""),
        cert_first_step=cert_first_step,
        ts_first_turn_tcpa_s=first_turn_tcpa,
    )


def _first_turns(env, theta_start, first_turn_tcpa):
    """Record the TCPA at which each target first turns.

    Args:
        env (CertifiedEscapeEnv): Environment
        theta_start (list): Start heading per target, rad
        first_turn_tcpa (list): First turn TCPA per target, updated in place
    """
    for j, b in enumerate(env._boats[1:]):
        if first_turn_tcpa[j] is not None:
            continue
        off = (b.theta - theta_start[j] + math.pi) % (2.0 * math.pi) - math.pi
        if abs(off) > FIRST_TURN_RAD:
            first_turn_tcpa[j] = float(cpa(env._os, b)[0])


def _all_opening(env):
    """Check that every pair of boats is moving apart.

    Args:
        env (CertifiedEscapeEnv): Environment

    Returns:
        bool: True if every pair opens
    """
    boats = env._boats
    for i, a in enumerate(boats):
        for b in boats[i + 1:]:
            tcpa = cpa(a, b)[0]
            if math.isfinite(tcpa) and tcpa > 0.0:
                return False
    return True


def _situation(env):
    """Get the situation of own vessel toward the first target.

    Args:
        env (CertifiedEscapeEnv): Environment

    Returns:
        str: Situation name, empty if none
    """
    from ..certificate import duty as _duty
    from ..certificate.escape_angle import Vessel
    from .cert_env import R_MIN_M

    boats = getattr(env, "_boats", None)
    if not boats or len(boats) < 2:
        return ""
    os_v = Vessel.from_sea_object(boats[0], r_min=R_MIN_M)
    return _duty.situation(os_v, Vessel.from_sea_object(boats[1], r_min=R_MIN_M))


def evaluate(bank_path, *, gate, model_path=None, n=100, window=MAX_STEPS, seed=0,
             ts_classes=None, holdout=0.2, horizon=None, arms=ARMS,
             alpha_col_release=0.0, rule17b_relax=False, contact_m=HULL_CONTACT_M,
             release_lookahead=RELEASE_LOOKAHEAD, veto_margin=VETO_MARGIN_M,
             speed_action=None, release_action=None,
             behaviour_obs=None, include_free=False,
             n_track=None, progress=True, jsonl=None, force_release_step=None,
             p_switch=0.0, max_emergency_steps=None, emergency_avoidance=True, capability=None,
             trigger_k=None, action_shield=None,
             behaviour_schedule=None, deviant=None, attribution=ContactAttribution.OFF,
             compliance=ComplianceCheck.OFF, seed_source=SeedSource.INDEX,
             population=Population.HELDOUT, shard=0, shards=1, run_length=RunLength.WINDOW):
    """Run every arm on the bank rows and collect the records.

    Args:
        bank_path (str): Bank CSV path
        gate (float): Duty gate, m
        model_path (str): Policy path, or None
        n (int): Max rows
        window (int): Number of steps
        seed (int): Random seed
        ts_classes (dict): Target behaviour mix
        holdout (float): Held out share of the bank
        horizon (float): Certificate horizon, s
        arms (list): Arm names
        alpha_col_release (float): Lawful width share needed to release
        rule17b_relax (bool): Whether Rule 17(b) relaxation is on
        contact_m (float): Contact distance, m
        release_lookahead (int): Steps replayed before a release
        veto_margin (float): Separation a release needs, m
        speed_action (bool): Whether the policy sets speed
        release_action (bool): Whether the policy asks for release
        behaviour_obs (bool): Whether to add behaviour class probabilities
        include_free (bool): Whether to keep free rows
        n_track (int): Number of tracked targets
        progress (bool): Whether to show a progress bar
        jsonl (str): Path of the per row log, or None
        force_release_step (int): Step to force release, or None
        p_switch (float): Premise share that must trap own vessel
        max_emergency_steps (int): Release cap, steps, or None
        emergency_avoidance (bool): Whether the in extremis override is on
        capability (dict): Capability knobs, or None
        trigger_k (int): Lead cap of the switch, steps, or None
        action_shield (int): Steps of the action shield, 0 for none
        behaviour_schedule (object): Behaviour switch schedule, or None
        deviant (object): Deviant target settings, or None
        attribution (bool): Whether to run the counterfactual without own vessel
        compliance (bool): Whether to check the rules
        seed_source (object): How each row seed is drawn
        population (Population): Which rows to run
        shard (int): Shard index
        shards (int): Number of shards
        run_length (object): Run length mode

    Returns:
        dict: Records per population
    """
    rows = _bank.load(bank_path)
    trapped = [r for r in rows if r["stratum"] != "free"]
    free = [r for r in rows if r["stratum"] == "free"]
    if population is Population.ALL:
        order = np.random.default_rng(seed).permutation(len(rows))
        populations = {"all (shuffled)": [rows[int(j)] for j in order][:n]}
    elif include_free:
        _tr, held = split_bank(trapped + free, holdout)
        populations = {"held-out (all strata)": held[:n]}
    else:
        _train, held = split_bank(trapped, holdout)
        populations = {"certified (held-out)": held[:n], "free (no-regression)": free[:n]}
    mix = UNIFORM_MIX if ts_classes is None else ts_classes

    model = None
    if model_path:
        from .policy_io import load_policy
        model = load_policy(model_path)
    elif "policy" in arms:
        arms = tuple(a for a in arms if a != "policy")

    bin_fields = (model is not None
                  and int(np.prod(model.observation_space.shape)) == 4 * N_BINS + 5)

    results = collections.defaultdict(list)
    done, kept = set(), []
    if jsonl and os.path.exists(jsonl):
        with open(jsonl, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kept.append(line if line.endswith("\n") else line + "\n")
                done.add((rec["population"], rec["arm"], rec["row"]))
                results[rec.pop("population")].append(rec)
    sink = open(jsonl, "w", encoding="utf-8") if jsonl else None
    if sink:
        sink.writelines(kept)
        sink.flush()
    for pop_name, pop in populations.items():
        if not pop:
            continue
        for arm in arms:
            make_env = functools.partial(CertifiedEscapeEnv, pop,
                                     gate=gate, ts_classes=mix, seed=seed,
                                     strata=tuple({r["stratum"] for r in pop}),
                                     score_after_handback=False,
                                     alpha_col_release=alpha_col_release,
                                     rule17b_relax=rule17b_relax,
                                     contact_m=contact_m,
                                     release_lookahead=release_lookahead,
                                     veto_margin=veto_margin,
                                     bin_fields=bin_fields,
                                     **({} if speed_action is None
                                        else {"speed_action": bool(speed_action)}),
                                     **({} if release_action is None
                                        else {"release_action": bool(release_action)}),
                                     **({} if behaviour_obs is None
                                        else {"behaviour_obs": bool(behaviour_obs)}),
                                     **({} if n_track is None else {"n_track": int(n_track)}),
                                     **({} if horizon is None else {"horizon": horizon}),
                                     **({"hybrid_policy": model, "p_switch": p_switch,
                                         "max_emergency_steps": max_emergency_steps}
                                        if arm == "hybrid" else {}),
                                     **({"trigger_k": int(trigger_k)}
                                        if arm == "hybrid" and trigger_k is not None else {}),
                                     **({"action_shield": int(action_shield)}
                                        if arm == "hybrid" and action_shield is not None else {}),
                                     emergency_avoidance=emergency_avoidance,
                                     capability=capability,
                                     behaviour_schedule=behaviour_schedule,
                                     deviant=deviant,
                                     **_ARM_KW[arm])
            env = make_env()
            env_cf = (make_env(own_ship=OwnShipPresence.ABSENT)
                      if attribution is ContactAttribution.COUNTERFACTUAL else None)
            if model is not None:
                env = stack_for_policy(env, model)
            rows_it = range(len(pop))
            gc.collect()
            gc.freeze()
            if progress:
                from tqdm import tqdm
                rows_it = tqdm(rows_it, desc=f"{arm} / {pop_name.split()[0]}", unit="ep",
                               leave=True)
            for i in rows_it:
                if (pop_name, arm, i) in done:
                    continue
                if i % shards != shard:
                    continue

                row_seed = (int(pop[i]["seed"]) if seed_source is SeedSource.BANK
                            else seed + 1000 * i)
                t_row = time.perf_counter()
                rec = run_row(env, i, arm, model=model, seed=row_seed,
                              window=window, contact_m=contact_m,
                              force_release_step=(force_release_step
                                                  if arm == "policy" else None),
                              env_cf=env_cf, compliance=compliance, run_length=run_length)
                rec["wall_s"] = time.perf_counter() - t_row
                results[pop_name].append(rec)
                if sink:
                    sink.write(json.dumps(dict(rec, population=pop_name)) + "\n")
                    sink.flush()
    if sink:
        sink.close()
    return results


def summarise(results):
    """Compute the outcome rates of each arm per population.

    Args:
        results (dict): Records per population

    Returns:
        dict: Rates per population and arm
    """
    table = {}
    for pop, rows in results.items():
        by = collections.defaultdict(list)
        for r in rows:
            by[r["arm"]].append(r)
        table[pop] = {
            arm: dict(
                n=len(rs),
                contact=float(np.mean([r["contact"] for r in rs])),
                contact_in_control=float(np.mean([r["contact_in_control"] for r in rs])),
                handback=float(np.mean([r["handback"] is not None for r in rs])),
                median_min_sep=float(np.median([min(r["min_sep"], 1e6) for r in rs])),
                engaged=float(np.mean([r.get("switch_engagements", 0) > 0 for r in rs])),
                mean_emergency_steps=float(np.mean(
                    [r.get("switch_emergency_steps", 0) for r in rs])),
            )
            for arm, rs in by.items()
        }
    return table


def summarise_cells(results, arms=("policy",)):
    """Compute the outcome rates per arm and bank cell.

    Args:
        results (dict): Records per population
        arms (list): Arm names

    Returns:
        list: Rows per cell
    """
    out = []
    for pop, rows in results.items():
        by = collections.defaultdict(list)
        for r in rows:
            if r["arm"] in arms:
                by[(r["arm"], r.get("situation", ""), r.get("ts_class", ""))].append(r)
        for (arm, sit, cls), rs in sorted(by.items()):
            rel = [r for r in rs if r.get("release_step") is not None]
            out.append({
                "population": pop, "arm": arm, "situation": sit, "ts_class": cls, "n": len(rs),
                "contact": float(np.mean([r["contact"] for r in rs])),
                "contact_in_control": float(np.mean([r["contact_in_control"] for r in rs])),
                "released": float(np.mean([r["released"] for r in rs])),
                "timeout": float(np.mean([r["timeout"] for r in rs])),
                "median_release_step": (float(np.median([r["release_step"] for r in rel]))
                                        if rel else None),
                "median_held_steps": float(np.median([r["held_steps"] for r in rs])),
                "override_rate": float(np.mean([r["override_steps"] > 0 for r in rs])),
                "veto_rate": float(np.mean([r["veto_steps"] > 0 for r in rs])),
                "median_min_sep": float(np.median([min(r["min_sep"], 1e6) for r in rs])),
            })
    return out


def paired(results, arm_a="compliant", arm_b="policy"):
    """Pair two arms row by row.

    Args:
        results (dict): Records per population
        arm_a (str): First arm
        arm_b (str): Second arm

    Returns:
        list: Paired results per population
    """
    out = []
    for pop, rows in results.items():
        by_arm = collections.defaultdict(dict)
        for r in rows:
            by_arm[r["arm"]][(r["row"], r.get("ts_class", ""))] = r
        a, b = by_arm.get(arm_a, {}), by_arm.get(arm_b, {})
        keys = sorted(set(a) & set(b))
        pairs = []
        for k in keys:
            ra, rb = a[k], b[k]
            pairs.append({
                "population": pop, "row": k[0], "ts_class": k[1],
                "situation": rb.get("situation", ""),
                f"{arm_a}_contact": bool(ra["contact"]), f"{arm_b}_contact": bool(rb["contact"]),
                f"{arm_a}_min_sep": float(min(ra["min_sep"], 1e6)),
                f"{arm_b}_min_sep": float(min(rb["min_sep"], 1e6)),
                "delta_min_sep": float(min(rb["min_sep"], 1e6) - min(ra["min_sep"], 1e6)),
                "fixed": bool(ra["contact"] and not rb["contact"]),
                "broken": bool(rb["contact"] and not ra["contact"]),
            })
        out.append({"population": pop, "arm_a": arm_a, "arm_b": arm_b, "n": len(pairs),
                    "pairs": pairs, "summary": _pair_summary(pairs, arm_a, arm_b)})
    return out


def _pair_summary(pairs, arm_a, arm_b):
    """Count fixed and broken rows of a pairing and test them.

    Args:
        pairs (list): Paired rows
        arm_a (str): First arm
        arm_b (str): Second arm

    Returns:
        dict: Counts and p value
    """
    if not pairs:
        return {"n": 0}
    fixed = sum(p["fixed"] for p in pairs)
    broken = sum(p["broken"] for p in pairs)
    d = np.array([p["delta_min_sep"] for p in pairs], dtype=float)
    return {
        "n": len(pairs), "fixed": int(fixed), "broken": int(broken),
        "p_mcnemar": _mcnemar(fixed, broken),
        "delta_min_sep_q": {str(q): float(np.quantile(d, q))
                            for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "delta_positive": float(np.mean(d > 0.0)),
        f"{arm_a}_contact": float(np.mean([p[f"{arm_a}_contact"] for p in pairs])),
        f"{arm_b}_contact": float(np.mean([p[f"{arm_b}_contact"] for p in pairs])),
    }


def _mcnemar(fixed, broken):
    """Compute the exact two sided McNemar p value.

    Args:
        fixed (int): Rows fixed by arm A
        broken (int): Rows broken by arm A

    Returns:
        float: p value
    """
    n = fixed + broken
    if n == 0:
        return 1.0
    k = min(fixed, broken)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * tail))


def _cap_steps(text):
    """Read a step cap, none for no cap.

    Args:
        text (str): Command line value

    Returns:
        int: Cap, or None
    """
    return None if text.lower() == "none" else int(text)


def main():
    """Run the evaluation from the command line and write the JSON."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--run", default=None, metavar="RUN_DIR",
                    help="take the bank, the gate, the behaviour mix, the action space and the "
                         "release rule off this run's config.json, and default --model to the "
                         "model.zip beside it. Anything also given explicitly wins.")
    ap.add_argument("--bank", default="trap_bank.csv")
    ap.add_argument("--gate", type=float, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--window", type=int, default=MAX_STEPS)
    ap.add_argument("--horizon", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--json", default=None)
    ap.add_argument("--jsonl", default=None,
                    help="per-encounter records, one JSON object per line, flushed as they "
                         "land. Defaults to --json with an l appended, so a killed run keeps "
                         "every encounter it finished.")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--alpha-col-release", type=float, default=0.0,
                    help="lawful-escape margin required to hand back; must match the "
                         "policy's training value — read it off the run's config.json")
    ap.add_argument("--rule17b-relax", action="store_true",
                    help="ADR 0009 stand-on role flip; must match the policy's training value")
    ap.add_argument("--contact-m", type=float, default=HULL_CONTACT_M, metavar="M",
                    help="separation counted as contact, in the environment and in this "
                         "harness's own tally. Default 150 (hull). Set it to the policy's "
                         "training value — a policy trained to hold 200 m and scored at 150 is "
                         "being asked a different question, which is the mistake --alpha-col-"
                         "release already carries a warning about.")
    ap.add_argument("--n-track", type=int, default=None, metavar="N",
                    help="threats given a per-threat observation block; must match the "
                         "policy's training value or the model will not load")
    ap.add_argument("--veto-margin", type=float, default=VETO_MARGIN_M, metavar="M",
                    help="separation the replayed compliant tail must hold for a release to be "
                         "permitted. Only bites at --release-lookahead > 0, and like it, MUST "
                         "match the policy's training value; --run reads it off the run config.")
    ap.add_argument("--release-lookahead", type=int, default=RELEASE_LOOKAHEAD, metavar="K",
                    help="ADR 0011 tail check: the release is permitted only if the escape "
                         "still holds K steps ahead. Defaults to 0 and MUST match the policy's "
                         "training value. capped_s0 trained at 60 and reads 36.7 %% contact "
                         "scored at 0 against 21.5 %% at 60 on the same 600 rows.")
    ap.add_argument("--force-release-step", type=int, default=None, metavar="K",
                    help="meeting-24 ask 1: make the policy hold the helm until step K, then "
                         "ask to hand back at every step after it. Sweeping K and reading "
                         "which values come back clean is the per-cause attribution: released "
                         "too soon, released too late, or lost whatever it did. Forces the "
                         "policy's ask only — the certificate's permission still has to agree.")
    ap.add_argument("--ts-mix", default=None, metavar="MIX",
                    help="threat behaviour mix, overriding the run's training mix, e.g. "
                         "'breakdown=1,adversarial=1,interceptor=1' for classes never trained on")
    ap.add_argument("--p-switch", type=float, default=0.0, metavar="P",
                    help="engage threshold of the deployed switch on the hybrid arm")
    ap.add_argument("--trigger-k", type=int, default=None, metavar="K",
                    help="lead-time cap of the hybrid arm: engage only when the first predicted "
                         "fire is within K steps. Default: no cap (switch default). Recorded in "
                         "the config hash; replaces the eval_trigger_k.py runtime patch (F086).")
    ap.add_argument("--action-shield", type=int, default=None, metavar="N",
                    help="action shield of the hybrid arm's switch, steps of constant-velocity "
                         "veto on the policy's turn. Default: the env default (0). Recorded in "
                         "the config hash (N2 ablation, 2026-10-24).")
    ap.add_argument("--max-emergency-steps", type=_cap_steps, default=RELEASE_CAP_STEPS,
                    metavar="K|none",
                    help="release cap on the hybrid arm: hand back at K steps once the "
                         "certificate is clear, log every step past K while it still fires. "
                         f"Default {RELEASE_CAP_STEPS} (cert_switch.RELEASE_CAP_STEPS); 'none' "
                         "is the uncapped switch every run before 2026-09-24 used.")
    ap.add_argument("--switches", type=int, default=0, metavar="N",
                    help="mid-episode behaviour change, evaluation only: each target changes "
                         "class N times, every --switch-interval steps, within the episode's mix")
    ap.add_argument("--switch-interval", type=int, default=10, metavar="K",
                    help="steps between two behaviour changes of one target")
    ap.add_argument("--schedule", choices=["none", ScheduleSource.ROW.value], default="none",
                    help="'row' plays the behaviour schedule each bank v2 row carries (its "
                         "`plan` column); 'none' keeps every target on its starting character")
    ap.add_argument("--deviant", choices=["none", DeviantSource.ROW.value], default="none",
                    help="'row' gives the first colliding target a deviant behaviour (mistaken, "
                         "breakdown, erratic, interceptor), drawn from the row's seed")
    ap.add_argument("--population", choices=[p.value for p in Population],
                    default=Population.HELDOUT.value,
                    help="'all' scores the whole bank in a seeded shuffled order (bank v2); "
                         "'heldout' the split every run before 2026-09-25 used")
    ap.add_argument("--stop-clear", action="store_true",
                    help="end a row once every pair opens with min separation above "
                         f"{STOP_CLEAR_SEP_M:.0f} m (late mining); off by default")
    ap.add_argument("--check-compliance", action="store_true",
                    help="score every episode with the third-party checker (Krasowski & Althoff "
                         "2021 rules R1-R6, Woerner et al. 2019 score): kr_* and woerner_* fields")
    ap.add_argument("--attribute-tt", action="store_true",
                    help="replay every target-target contact without the own ship and label it "
                         "own_ship / targets, with the duty roles the pair held")
    ap.add_argument("--no-emergency-avoidance", action="store_true",
                    help="remove L4, the in-extremis override, from every vessel")
    ap.add_argument("--capability", default=None, metavar="JSON",
                    help='disclosed layer knobs on every vessel, e.g. '
                         '\'{"SAFE_DCPA_M": 926, "DOMAIN_AXES_LENGTHS": [4, 2, 6]}\'')
    ap.add_argument("--pair", default="compliant,policy", metavar="ARM_A,ARM_B",
                    help="emit per-row pairs between these two arms and their McNemar test "
                         "(backlog A13). Empty string turns it off.")
    ap.add_argument("--row-seed", choices=[s.value for s in SeedSource],
                    default=SeedSource.INDEX.value,
                    help="'bank' seeds each row with the seed its bank row recorded, so a "
                         "reactive-bank row faces the target draw it was accepted under. Pair "
                         "it with the bank's --ts-mix.")
    ap.add_argument("--shard", type=int, default=0, metavar="I",
                    help="score only rows i with i %% --shards == I, to split one slow arm over "
                         "several processes. Each part writes its own --json.")
    ap.add_argument("--shards", type=int, default=1, metavar="K")
    a = ap.parse_args()
    if not 0 <= a.shard < a.shards:
        ap.error(f"--shard {a.shard} outside 0..{a.shards - 1}")

    cfg, ts_classes, include_free = {}, None, False
    if a.run:
        with open(os.path.join(a.run, "config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
        defaults = ap.parse_args([])
        if a.bank == defaults.bank and cfg.get("bank"):
            a.bank = cfg["bank"]
        if a.gate is None:
            a.gate = cfg.get("gate")
        _mz = os.path.join(a.run, "model.zip")
        a.model = a.model or (_mz if os.path.exists(_mz) else None)
        for name in ("alpha_col_release", "contact_m", "release_lookahead", "veto_margin",
                     "n_track", "rule17b_relax"):
            if getattr(a, name) == getattr(defaults, name) and cfg.get(name) is not None:
                setattr(a, name, cfg[name])
        ts_classes = cfg.get("mix") or None
        include_free = bool(cfg.get("include_free"))
        print(f"run {a.run}: bank {a.bank} gate {a.gate} "
              f"speed_action {cfg.get('speed_action')} release_action {cfg.get('release_action')} "
              f"behaviour_obs {cfg.get('behaviour_obs')} include_free {include_free}")

    if a.ts_mix:
        ts_classes = class_mix(a.ts_mix)
    capability = json.loads(a.capability) if a.capability else None
    if capability and "DOMAIN_AXES_LENGTHS" in capability:
        capability["DOMAIN_AXES_LENGTHS"] = tuple(capability["DOMAIN_AXES_LENGTHS"])
    config = {k: v for k, v in sorted(vars(a).items())
              if k not in ("json", "jsonl", "no_progress", "shard", "shards")}
    if a.row_seed == SeedSource.INDEX.value:
        del config["row_seed"]
    if a.schedule == "none":
        del config["schedule"]
    if a.deviant == "none":
        del config["deviant"]
    if a.population == Population.HELDOUT.value:
        del config["population"]
    if a.trigger_k is None:
        del config["trigger_k"]
    if a.action_shield is None:
        del config["action_shield"]
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    print(f"config_sha {config_sha} {json.dumps(config, sort_keys=True)}")

    jsonl = a.jsonl or (a.json + "l" if a.json else None)
    res = evaluate(a.bank, gate=a.gate, model_path=a.model, n=a.n, window=a.window,
                   seed=a.seed, horizon=a.horizon, arms=tuple(a.arms.split(",")),
                   alpha_col_release=a.alpha_col_release, rule17b_relax=a.rule17b_relax,
                   contact_m=a.contact_m, n_track=a.n_track,
                   release_lookahead=a.release_lookahead, veto_margin=a.veto_margin,
                   ts_classes=ts_classes, include_free=include_free,
                   speed_action=cfg.get("speed_action"),
                   release_action=cfg.get("release_action"),
                   behaviour_obs=cfg.get("behaviour_obs"),
                   progress=not a.no_progress, jsonl=jsonl,
                   force_release_step=a.force_release_step,
                   p_switch=a.p_switch, max_emergency_steps=a.max_emergency_steps,
                   trigger_k=a.trigger_k, action_shield=a.action_shield,
                   emergency_avoidance=not a.no_emergency_avoidance, capability=capability,
                   behaviour_schedule=(ScheduleSource.ROW if a.schedule == ScheduleSource.ROW.value
                                       else BehaviourSchedule(a.switches, a.switch_interval)
                                       if a.switches > 0 else None),
                   deviant=DeviantSource.ROW if a.deviant == DeviantSource.ROW.value else None,
                   attribution=(ContactAttribution.COUNTERFACTUAL if a.attribute_tt
                                else ContactAttribution.OFF),
                   compliance=(ComplianceCheck.ON if a.check_compliance
                               else ComplianceCheck.OFF),
                   seed_source=SeedSource(a.row_seed), population=Population(a.population),
                   shard=a.shard, shards=a.shards,
                   run_length=RunLength.CLEAR if a.stop_clear else RunLength.WINDOW)
    table = summarise(res)
    for pop, arms in table.items():
        print(f"\n--- {pop} ---")
        print(f"  {'arm':<16}{'n':>5}{'contact':>10}{'in-control':>12}"
              f"{'handback':>10}{'med min-sep':>13}{'engaged':>10}{'emerg st':>10}")
        for arm, v in arms.items():
            sep = "inf" if not math.isfinite(v["median_min_sep"]) else f"{v['median_min_sep']:.0f}"
            print(f"  {arm:<16}{v['n']:>5}{v['contact']:>9.1%}{v['contact_in_control']:>12.1%}"
                  f"{v['handback']:>10.1%}{sep:>13}{v['engaged']:>10.1%}"
                  f"{v['mean_emergency_steps']:>10.1f}")
    cells = summarise_cells(res, arms=tuple(a.arms.split(",")))
    if cells:
        print("\n--- availability per situation x behaviour class ---")
        print("  (released / timeout / rel@ / held are policy-arm quantities and read 0 "
              "for every arm that never holds the helm)")
        print(f"  {'arm':<10}{'situation':<24}{'class':<13}{'n':>4}{'contact':>9}"
              f"{'released':>10}{'timeout':>9}{'rel@':>6}{'held':>6}{'over':>7}")
        for c in cells:
            rel = "-" if c["median_release_step"] is None else f"{c['median_release_step']:.0f}"
            print(f"  {c['arm'][:9]:<10}{c['situation'][:23]:<24}{c['ts_class'][:12]:<13}"
                  f"{c['n']:>4}{c['contact']:>8.0%}{c['released']:>10.0%}{c['timeout']:>9.0%}"
                  f"{rel:>6}{c['median_held_steps']:>6.0f}{c['override_rate']:>7.0%}")

    pairs = []
    if a.pair:
        arm_a, arm_b = a.pair.split(",")
        pairs = paired(res, arm_a, arm_b)
        for blk in pairs:
            sm = blk["summary"]
            if not sm.get("n"):
                continue
            q = sm["delta_min_sep_q"]
            print(f"\n--- paired {arm_a} vs {arm_b}: {blk['population']} ---")
            print(f"  n={sm['n']}  fixed {sm['fixed']}  broken {sm['broken']}  "
                  f"p={sm['p_mcnemar']:.4f}")
            print(f"  delta min-sep (m)  q05 {q['0.05']:.0f}  q25 {q['0.25']:.0f}  "
                  f"med {q['0.5']:.0f}  q75 {q['0.75']:.0f}  q95 {q['0.95']:.0f}  "
                  f"({sm['delta_positive']:.0%} of rows improved)")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"config_sha": config_sha, "config": config,
                       "table": table, "cells": cells,
                       "paired": [{k: v for k, v in b.items() if k != "pairs"} for b in pairs],
                       "pairs": [p for b in pairs for p in b["pairs"]]}, fh, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()

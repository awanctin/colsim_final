from __future__ import annotations

import math

import numpy as np

from . import alpha as _alpha
from . import duty as _duty
from .behaviour import APPARENT_TURN_RAD, CLASSES
from .certificate import POSE_SIGMA_HEADING_RAD, POSE_SIGMA_M, _perturb
from .escape_angle import R_MIN_M, VISIBLE_RANGE_M, Vessel, _wrap

try:
    import os as _os

    if _os.environ.get("COLSIM_MONITOR_NUMBA", "1") == "0":
        raise ImportError("disabled by COLSIM_MONITOR_NUMBA=0")
    from . import _monitor_nb as _nb
except ImportError:
    _nb = None

SIM_DT_S = 10.0
DECISION_HORIZON_S = _duty.HORIZON_S
N_FUTURES = 8
N_PERTURB_AT_HIT = 50
GATE = None


def commitment_interval(os_theta, ref_heading, side, k):
    """Get the headings that keep the committed side.

    Args:
        os_theta (float): Own heading, rad
        ref_heading (float): Reference heading, rad
        side (int): Committed side
        k (float): Max heading change, rad

    Returns:
        list: Allowed interval, or empty
    """
    phi = float(_wrap(ref_heading - os_theta))
    lo, hi = (phi - math.pi, phi) if side > 0 else (phi, phi + math.pi)
    lo, hi = max(lo, -k), min(hi, k)
    return [(lo, hi)] if hi - lo > 1e-12 else []


def latched(os_obj, os_theta, k):
    """Get the commitment interval of a vessel, if any.

    Args:
        os_obj (SeaObject): Own vessel
        os_theta (float): Own heading, rad
        k (float): Max heading change, rad

    Returns:
        list: Allowed interval, or None
    """
    ref = getattr(os_obj, "_compliant_reference", None)
    if not callable(ref):
        return None
    r = ref()
    if r is None:
        return None
    ref_heading, side = r[2], r[3]
    return commitment_interval(os_theta, ref_heading, side, k)


def step_certificate(
    os_v,
    threats,
    *,
    commitment=None,
    gate=GATE,
    T=_duty.HORIZON_S,
    kappa_max=_duty.KAPPA_MAX_RAD,
    delta_min=_duty.DELTA_MIN_RAD,
    h=_duty.HOLD_BAND_RAD,
    S=_duty.SAFETY_RADIUS_M,
    need_phys=True,
    relaxed=None,
    memory=None,
):
    """Check the deontic and geometric certificates at one instant.

    Args:
        os_v (Vessel): Own vessel
        threats (list): Target vessels
        commitment (list): Commitment interval, or None
        gate (object): Gate settings
        T (float): Look ahead time, s
        kappa_max (float): Max heading change, rad
        delta_min (float): Smallest lawful turn, rad
        h (float): Hold band, rad
        S (float): Safety radius, m
        need_phys (bool): Whether to compute the physical width
        relaxed (list): Relaxation flag per target, or None
        memory (list): Stored role per target, or None

    Returns:
        dict: Flags, widths and reach
    """
    k = _duty.kappa(os_v, T, kappa_max)
    lawful = _duty.lawful_intervals(
        os_v,
        threats,
        T=T,
        kappa_max=kappa_max,
        delta_min=delta_min,
        h=h,
        gate=gate,
        relaxed=relaxed,
        memory=memory,
    )
    if commitment:
        lawful = _duty._intersect(lawful, commitment)

    deontic = not lawful
    a_col = _alpha.alpha(os_v, threats, lawful, S, T) if lawful else 0.0
    a_phys = _alpha.alpha(os_v, threats, [(-k, k)], S, T) if need_phys else float("nan")
    return dict(
        deontic=bool(deontic),
        geometric=bool(lawful) and a_col <= 0.0,
        alpha_col=float(a_col),
        alpha_phys=float(a_phys),
        lawful_width=float(sum(hi - lo for lo, hi in lawful)),
        kappa=float(k),
        ics_bound=bool(need_phys and a_phys <= 0.0),
    )


def _p_hat(
    os_v,
    threats,
    *,
    commitment=None,
    gate=GATE,
    n_perturb=N_PERTURB_AT_HIT,
    sigma_m=POSE_SIGMA_M,
    sigma_hdg=POSE_SIGMA_HEADING_RAD,
    rng=None,
    memory=None,
    **kw,
):
    """Compute the share of noisy copies that still fire the certificate.

    Args:
        os_v (Vessel): Own vessel
        threats (list): Target vessels
        commitment (list): Commitment interval, or None
        gate (object): Gate settings
        n_perturb (int): Number of noisy copies
        sigma_m (float): Position noise, m
        sigma_hdg (float): Heading noise, rad
        rng (Generator): Random generator, or None
        memory (list): Stored role per target, or None
        **kw (dict): Extra settings for step_certificate

    Returns:
        float: Share between 0 and 1
    """
    rng = np.random.default_rng() if rng is None else rng
    hits = 0
    for _ in range(n_perturb):
        os_p = _perturb(os_v, rng, sigma_m, sigma_hdg)
        c = commitment
        if commitment:
            d = float(_wrap(os_p.theta - os_v.theta))
            c = [(lo - d, hi - d) for lo, hi in commitment]
        hits += bool(
            step_certificate(
                os_p,
                threats,
                commitment=c,
                gate=gate,
                need_phys=False,
                memory=memory,
                **kw,
            )["geometric"]
        )
    return hits / n_perturb if n_perturb else 0.0


def roll_own(os_v, dt=SIM_DT_S):
    """Move own vessel one step in a straight line.

    Args:
        os_v (Vessel): Own vessel
        dt (float): Time step, s

    Returns:
        Vessel: Moved vessel
    """
    return Vessel(
        os_v.name,
        os_v.x + os_v.v * math.cos(os_v.theta) * dt,
        os_v.y + os_v.v * math.sin(os_v.theta) * dt,
        os_v.v,
        os_v.theta,
        os_v.r_min,
    )


def intercept_heading(ts, os_v):
    """Find the heading a target needs to meet own vessel.

    Args:
        ts (Vessel): Target vessel
        os_v (Vessel): Own vessel

    Returns:
        float: Heading, rad
    """
    dx, dy = os_v.x - ts.x, os_v.y - ts.y
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return ts.theta
    rx, ry = dx / d, dy / d
    ox, oy = os_v.v * math.cos(os_v.theta), os_v.v * math.sin(os_v.theta)
    b = ox * rx + oy * ry
    disc = b * b - (os_v.v**2 - ts.v**2)
    if disc < 0.0:
        return math.atan2(dy, dx)
    c = -b + math.sqrt(disc)
    if c <= 0.0:
        return math.atan2(dy, dx)
    return math.atan2(oy + c * ry, ox + c * rx)


def route_intercept_time(ts, os_v):
    """Find when a target can reach the route of own vessel.

    Args:
        ts (Vessel): Target vessel
        os_v (Vessel): Own vessel

    Returns:
        float: Time, s, or None
    """
    dx, dy = os_v.x - ts.x, os_v.y - ts.y
    ox, oy = os_v.v * math.cos(os_v.theta), os_v.v * math.sin(os_v.theta)
    a = os_v.v**2 - ts.v**2
    b = dx * ox + dy * oy
    c = dx * dx + dy * dy
    if abs(a) < 1e-9:
        if abs(b) < 1e-12:
            return None
        t = -c / (2.0 * b)
        return t if t > 0.0 else None
    disc = b * b - a * c
    if disc < 0.0:
        return None
    root = math.sqrt(disc)
    roots = sorted(t for t in ((-b - root) / a, (-b + root) / a) if t > 0.0)
    return roots[0] if roots else None


def route_intercept_point(ts, os_v):
    """Find where a target can reach the route of own vessel.

    Args:
        ts (Vessel): Target vessel
        os_v (Vessel): Own vessel

    Returns:
        tuple: Point (x, y), m, or None
    """
    t = route_intercept_time(ts, os_v)
    if t is None:
        return None
    return (
        os_v.x + os_v.v * math.cos(os_v.theta) * t,
        os_v.y + os_v.v * math.sin(os_v.theta) * t,
    )


def _gives_way(ts, os_v):
    """Check whether a target must give way to own vessel.

    Args:
        ts (Vessel): Target vessel
        os_v (Vessel): Own vessel

    Returns:
        bool: True if the target gives way
    """
    dx, dy = os_v.x - ts.x, os_v.y - ts.y
    if dx * dx + dy * dy > VISIBLE_RANGE_M * VISIBLE_RANGE_M:
        return False
    return _duty.closing(ts, os_v) and _duty.role(ts, os_v) in _duty.GIVE_WAY


def roll_threat(ts, os_v, cls, dt=SIM_DT_S, turned=0.0, aim=None):
    """Move a target one step with the behaviour of its class.

    Args:
        ts (Vessel): Target vessel
        os_v (Vessel): Own vessel
        cls (str): Behaviour class
        dt (float): Time step, s
        turned (float): Turn already made, rad
        aim (tuple): Aim point, or None

    Returns:
        tuple: Moved target and total turn
    """
    rate = ts.rate_of_turn * dt
    d = 0.0
    if cls == "adversarial":
        d = min(max(float(_wrap(intercept_heading(ts, os_v) - ts.theta)), -rate), rate)
    elif cls == "interceptor":
        px, py = aim if aim is not None else (os_v.x, os_v.y)
        want = math.atan2(py - ts.y, px - ts.x)
        d = min(max(float(_wrap(want - ts.theta)), -rate), rate)
    elif cls == "compliant" and _gives_way(ts, os_v):
        remaining = max(APPARENT_TURN_RAD - abs(turned), 0.0)
        d = -min(rate, remaining)
    theta = float(_wrap(ts.theta + d))
    return (
        Vessel(
            ts.name,
            ts.x + ts.v * math.cos(theta) * dt,
            ts.y + ts.v * math.sin(theta) * dt,
            ts.v,
            theta,
            ts.r_min,
        ),
        turned + d,
    )


def _sample_classes(dists, rng):
    """Draw a behaviour class for each target.

    Args:
        dists (list): Class probabilities per target
        rng (Generator): Random generator

    Returns:
        list: Class per target
    """
    return [rng.choice(CLASSES, p=[d[c] for c in CLASSES]) for d in dists]


def _rollout(
    os_v,
    threats,
    classes,
    *,
    horizon,
    dt,
    commitment,
    gate,
    rng,
    n_perturb=N_PERTURB_AT_HIT,
    rule17b_relax=False,
    memory=None,
    **kw,
):
    """Roll one future forward until a certificate fires.

    Args:
        os_v (Vessel): Own vessel
        threats (list): Target vessels
        classes (list): Class per target
        horizon (float): Look ahead time, s
        dt (float): Time step, s
        commitment (list): Commitment interval, or None
        gate (object): Gate settings
        rng (Generator): Random generator
        n_perturb (int): Number of noisy copies
        rule17b_relax (bool): Whether to relax Rule 17(b)
        memory (list): Stored role per target, or None
        **kw (dict): Extra settings for step_certificate

    Returns:
        tuple: Fire time, certificate name, noisy share and steps used
    """
    os_r, ts_r = os_v, list(threats)
    turned = [0.0] * len(ts_r)
    aims = [
        route_intercept_point(ts, os_v) if classes[j] == "interceptor" else None
        for j, ts in enumerate(ts_r)
    ]
    n_steps = int(round(horizon / dt))
    relaxed = [c != "compliant" for c in classes] if rule17b_relax else None
    for i in range(n_steps + 1):
        t = i * dt
        c = step_certificate(
            os_r,
            ts_r,
            commitment=commitment,
            gate=gate,
            need_phys=False,
            relaxed=relaxed,
            memory=memory,
            **kw,
        )
        if c["deontic"]:
            return t, "deontic", None, i + 1
        if c["geometric"]:
            return (
                t,
                "geometric",
                (
                    _p_hat(
                        os_r,
                        ts_r,
                        commitment=commitment,
                        gate=gate,
                        rng=rng,
                        n_perturb=n_perturb,
                        relaxed=relaxed,
                        memory=memory,
                        **kw,
                    )
                    if n_perturb
                    else None
                ),
                i + 1,
            )
        if i == n_steps:
            break
        os_r = roll_own(os_r, dt)
        rolled = [
            roll_threat(ts, os_r, classes[j], dt, turned[j], aim=aims[j])
            for j, ts in enumerate(ts_r)
        ]
        ts_r = [v for v, _ in rolled]
        turned = [w for _, w in rolled]
    return None, None, None, n_steps + 1


def predict(
    os_obj,
    threats,
    *,
    dists=None,
    premises=("compliant",),
    horizon=DECISION_HORIZON_S,
    dt=SIM_DT_S,
    gate=GATE,
    n_futures=N_FUTURES,
    n_perturb=N_PERTURB_AT_HIT,
    rng=None,
    use_commitment=True,
    rule17b_relax=False,
    relaxed_now=None,
    memory=None,
    **kw,
):
    """Predict whether and when a certificate will fire over sampled futures.

    Args:
        os_obj (object): Own vessel or simulator object
        threats (list): Target vessels or objects
        dists (list): Class probabilities per target, or None
        premises (tuple): Classes used when dists is None
        horizon (float): Look ahead time, s
        dt (float): Time step, s
        gate (object): Gate settings
        n_futures (int): Number of sampled futures
        n_perturb (int): Number of noisy copies
        rng (Generator): Random generator, or None
        use_commitment (bool): Whether to keep the committed side
        rule17b_relax (bool): Whether to relax Rule 17(b)
        relaxed_now (list): Relaxation flag per target now, or None
        memory (list): Stored role per target, or None
        **kw (dict): Extra settings for step_certificate

    Returns:
        dict: Fire flags, share of futures that fire, fire times and arms
    """
    rng = np.random.default_rng() if rng is None else rng
    os_v = os_obj if isinstance(os_obj, Vessel) else Vessel.from_sea_object(os_obj)
    ts_v = [t if isinstance(t, Vessel) else Vessel.from_sea_object(t) for t in threats]

    k = _duty.kappa(
        os_v, kw.get("T", _duty.HORIZON_S), kw.get("kappa_max", _duty.KAPPA_MAX_RAD)
    )
    commitment = latched(os_obj, os_v.theta, k) if use_commitment else None

    if dists is None:
        futures = [[c] * len(ts_v) for c in premises]
    else:
        futures = [_sample_classes(dists, rng) for _ in range(max(int(n_futures), 1))]

    if _nb is not None and not kw:
        now, results = _nb_predict(
            os_v,
            ts_v,
            futures,
            horizon=horizon,
            dt=dt,
            commitment=commitment,
            gate=gate,
            rng=rng,
            n_perturb=n_perturb,
            rule17b_relax=rule17b_relax,
            relaxed_now=(relaxed_now if rule17b_relax else None),
            memory=memory,
        )
    else:
        now = step_certificate(
            os_v,
            ts_v,
            commitment=commitment,
            gate=gate,
            relaxed=(relaxed_now if rule17b_relax else None),
            memory=memory,
            **kw,
        )
        results = [
            _rollout(
                os_v,
                ts_v,
                classes,
                horizon=horizon,
                dt=dt,
                commitment=commitment,
                gate=gate,
                rng=rng,
                n_perturb=n_perturb,
                rule17b_relax=rule17b_relax,
                memory=memory,
                **kw,
            )
            for classes in futures
        ]
    ts_first, arms, p_hats, n_states = [], [], [], 0
    for t, arm, ph, n in results:
        ts_first.append(t)
        arms.append(arm)
        p_hats.append(ph)
        n_states += n

    fired = [t for t in ts_first if t is not None]
    return dict(
        fires_now=bool(now["deontic"] or now["geometric"]),
        now=now,
        p_fire=len(fired) / len(futures),
        t_first=ts_first,
        t_first_min=min(fired) if fired else float("inf"),
        arms=arms,
        p_hat=p_hats,
        n_futures=len(futures),
        n_states=n_states,
        horizon=float(horizon),
        committed=bool(commitment),
    )


_NO_INTERVALS = np.empty((0, 2))
_ARM_NAMES = {1: "deontic", 2: "geometric"}


def _nb_intervals(intervals):
    """Turn intervals into an array for numba.

    Args:
        intervals (list): Pairs (low, high)

    Returns:
        tuple: Array and count
    """
    if not intervals:
        return _NO_INTERVALS, 0
    return np.array(intervals, dtype=float).reshape(-1, 2), len(intervals)


_FLAG_BUF = {}


def _nb_flags(flags, n):
    """Turn flags into a reusable int8 array for numba.

    Args:
        flags (list): Flags, or None
        n (int): Array length

    Returns:
        ndarray: Flags as 0 or 1
    """
    out = _FLAG_BUF.get(n)
    if out is None:
        out = np.zeros(n, dtype=np.int8)
        _FLAG_BUF[n] = out
    else:
        out[:] = 0
    for i, f in enumerate(() if flags is None else flags):
        if i >= n:
            break
        out[i] = bool(f)
    return out


_PRED_BUF = {}


def _pred_buffers(n_fut, n):
    """Get reusable arrays for the numba prediction.

    Args:
        n_fut (int): Number of futures
        n (int): Number of targets

    Returns:
        tuple: Work arrays
    """
    key = (n_fut, n)
    buf = _PRED_BUF.get(key)
    if buf is None:
        buf = (
            np.empty((n_fut, n), dtype=np.int64),
            np.zeros((n_fut, n), dtype=np.bool_),
            np.zeros((n_fut, n), dtype=np.int8),
            np.zeros((n, 2)),
            np.empty(n_fut),
            np.empty(n_fut, dtype=np.int64),
            np.empty(n_fut, dtype=np.int64),
            np.empty((n_fut, 5)),
            np.empty((n_fut, n, 5)),
        )
        _PRED_BUF[key] = buf
    else:
        buf[1][:] = False
        buf[2][:] = 0
        buf[3][:] = 0.0
    return buf


def _nb_predict(
    os_v,
    threats,
    futures,
    *,
    horizon,
    dt,
    commitment,
    gate,
    rng,
    n_perturb,
    rule17b_relax,
    relaxed_now,
    memory=None,
):
    """Run the future rollouts with the numba kernel.

    Args:
        os_v (Vessel): Own vessel
        threats (list): Target vessels
        futures (list): Class per target, per future
        horizon (float): Look ahead time, s
        dt (float): Time step, s
        commitment (list): Commitment interval, or None
        gate (object): Gate settings
        rng (Generator): Random generator
        n_perturb (int): Number of noisy copies
        rule17b_relax (bool): Whether to relax Rule 17(b)
        relaxed_now (list): Relaxation flag per target now, or None
        memory (list): Stored role per target, or None

    Returns:
        tuple: Current certificate and one result per future
    """
    n, n_fut = len(threats), len(futures)
    codes, has_aim, relaxed_fut, aims, t_out, arm_out, ns_out, os_out, ts_out = (
        _pred_buffers(n_fut, n)
    )
    aim_found = [None] * n
    for f, classes in enumerate(futures):
        for j, c in enumerate(classes):
            codes[f, j] = _nb.CLASS_CODE.get(c, _nb.C_STRAIGHT)
            if c == "interceptor":
                if aim_found[j] is None:
                    p = route_intercept_point(threats[j], os_v)
                    aim_found[j] = p is not None
                    if p is not None:
                        aims[j] = p
                has_aim[f, j] = aim_found[j]
        if rule17b_relax:
            relaxed_fut[f] = _nb_flags([c != "compliant" for c in classes], n)
    commit, n_commit = _nb_intervals(commitment)

    deontic, geometric, a_col, a_phys, width, k = _nb.predict_core(
        _nb.to_rows(os_v),
        _nb.to_rows_list(threats),
        codes,
        aims,
        has_aim,
        relaxed_fut,
        int(round(horizon / dt)),
        float(dt),
        commit,
        n_commit,
        gate is not None,
        0.0 if gate is None else float(gate),
        _nb_flags(relaxed_now, n),
        _nb.memory_codes(memory, n),
        _duty.HORIZON_S,
        _duty.KAPPA_MAX_RAD,
        _duty.DELTA_MIN_RAD,
        _duty.HOLD_BAND_RAD,
        _duty.SAFETY_RADIUS_M,
        t_out,
        arm_out,
        ns_out,
        os_out,
        ts_out,
    )
    now = dict(
        deontic=bool(deontic),
        geometric=bool(geometric),
        alpha_col=float(a_col),
        alpha_phys=float(a_phys),
        lawful_width=float(width),
        kappa=float(k),
        ics_bound=bool(a_phys <= 0.0),
    )

    results = []
    for f, classes in enumerate(futures):
        arm, n_states = int(arm_out[f]), int(ns_out[f])
        if arm == _nb.ARM_NONE:
            results.append((None, None, None, n_states))
            continue
        t = float(t_out[f])
        if arm == _nb.ARM_DEONTIC or not n_perturb:
            results.append((t, _ARM_NAMES[arm], None, n_states))
            continue
        relaxed = [c != "compliant" for c in classes] if rule17b_relax else None
        os_r = Vessel(
            os_v.name,
            os_out[f, 0],
            os_out[f, 1],
            os_out[f, 2],
            os_out[f, 3],
            os_v.r_min,
        )
        ts_r = [
            Vessel(ts.name, *ts_out[f, j, :4], ts.r_min) for j, ts in enumerate(threats)
        ]
        ph = _p_hat(
            os_r,
            ts_r,
            commitment=commitment,
            gate=gate,
            rng=rng,
            n_perturb=n_perturb,
            relaxed=relaxed,
            memory=memory,
        )
        results.append((t, _ARM_NAMES[arm], ph, n_states))
    return now, results


def horizon_curve(results, horizons=None):
    """Compute the share of fired futures for each horizon.

    Args:
        results (list): Outputs of predict
        horizons (ndarray): Horizons, s, or None for 0 to 600

    Returns:
        tuple: Horizons and fired share
    """
    horizons = (
        np.arange(0.0, 601.0, 30.0) if horizons is None else np.asarray(horizons, float)
    )
    results = list(results)
    if not results:
        return horizons, np.zeros_like(horizons)
    frac = []
    for H in horizons:
        hits = [
            np.mean([(t is not None and t <= H) for t in r["t_first"]]) for r in results
        ]
        frac.append(float(np.mean(hits)))
    return horizons, np.array(frac)


class _FakeOS:

    def __init__(self, x, y, v, theta, ref_heading, side):
        """Build a minimal own vessel with a committed side, for tests.

        Args:
            x (float): x position, m
            y (float): y position, m
            v (float): Speed, m/s
            theta (float): Heading, rad
            ref_heading (float): Reference heading, rad
            side (int): Committed side
        """
        self.name = "OS"
        self.x, self.y, self.v, self.theta = x, y, v, theta
        self.avoidance_side = side
        self.avoidance_ref_heading = ref_heading
        self.avoidance_ref_pos = None
        self.max_rot_rad_s = abs(float(v)) / R_MIN_M

    def _compliant_reference(self):
        """Get the commitment reference, like SeaObject.

        Returns:
            tuple: Reference x, y, heading and side, or None
        """
        if not self.avoidance_side or self.avoidance_ref_heading is None:
            return None
        return (
            self.x,
            self.y,
            float(self.avoidance_ref_heading),
            int(self.avoidance_side),
        )

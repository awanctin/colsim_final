from __future__ import annotations

import numpy as np

from . import alpha as _alpha
from . import duty as _duty
from .escape_angle import (
    HULL_CONTACT_M,
    Vessel,
    _horizon_n,
    _kappa,
    _os_compliant_intervals,
    escape_measure,
)

MODEL = "manuscript"

POSE_SIGMA_M = 500.0
POSE_SIGMA_HEADING_RAD = np.radians(5.0)
N_PERTURB = 50


def deontic_certificate(
    os_boat, threats, *, delta_min=None, gate=None, T=None, react=0.0, model=None
):
    """Check whether no lawful heading is left for own vessel.

    Args:
        os_boat (Vessel): Own vessel
        threats (list): Target vessels
        delta_min (float): Smallest lawful turn, rad, or None
        gate (float): Distance under which a duty applies, or None
        T (float): Look ahead time, s, or None
        react (float): Reaction delay, s
        model (str): Model name, or None for the default

    Returns:
        bool: True if no lawful heading exists
    """

    model = MODEL if model is None else model
    if model == "manuscript":
        kw = {} if delta_min is None else dict(delta_min=delta_min)
        return not _duty.lawful_intervals(
            os_boat,
            list(threats),
            T=_duty.HORIZON_S if T is None else T,
            gate=gate,
            **kw,
        )
    return _deontic_legacy(
        os_boat, threats, delta_min=delta_min, gate=gate, T=T, react=react
    )


def _legacy_gate(gate):
    """Turn a gate value into the dict the old model expects.

    Args:
        gate (object): Gate value, dict or None

    Returns:
        dict: Gate settings, or None
    """
    if gate is None or isinstance(gate, dict):
        return gate
    return dict(dcpa_gate=float(gate))


def _deontic_legacy(os_boat, threats, *, delta_min=None, gate=None, T=None, react=0.0):
    """Old model version of deontic_certificate.

    Args:
        os_boat (Vessel): Own vessel
        threats (list): Target vessels
        delta_min (float): Smallest lawful turn, rad, or None
        gate (object): Gate value, or None
        T (float): Look ahead time, s, or None
        react (float): Reaction delay, s

    Returns:
        bool: True if no lawful heading exists
    """
    gate = _legacy_gate(gate)
    T = _horizon_n([os_boat] + list(threats)) if T is None else T
    k0 = _kappa(os_boat, T, react)
    return not _os_compliant_intervals(
        os_boat, list(threats), k0, delta_min=delta_min, gate=gate
    )


def geometric_certificate(
    os_boat,
    threats,
    *,
    delta_min=None,
    gate=None,
    n0=361,
    nj=41,
    r_d=HULL_CONTACT_M,
    T=None,
    react=0.0,
    model=None,
):
    """Check whether every reachable heading leads to contact.

    Args:
        os_boat (Vessel): Own vessel
        threats (list): Target vessels
        delta_min (float): Smallest lawful turn, rad, or None
        gate (object): Gate value, or None
        n0 (int): Number of headings tested
        nj (int): Number of time samples
        r_d (float): Contact distance, m
        T (float): Look ahead time, s, or None
        react (float): Reaction delay, s
        model (str): Model name, or None for the default

    Returns:
        tuple: True if in deadlock, and the full measure
    """
    model = MODEL if model is None else model
    if model == "manuscript":
        if react:
            raise ValueError(
                f"model='manuscript' has no reaction delay, got react={react}"
            )
        kw = {} if delta_min is None else dict(delta_min=delta_min)
        m = _alpha.measure(
            os_boat,
            list(threats),
            S=r_d,
            gate=gate,
            T=_duty.HORIZON_S if T is None else T,
            **kw,
        )
        return bool(m["physical_deadlock"]), m
    return _geometric_legacy(
        os_boat,
        threats,
        delta_min=delta_min,
        gate=gate,
        n0=n0,
        nj=nj,
        r_d=r_d,
        T=T,
        react=react,
    )


def _geometric_legacy(
    os_boat,
    threats,
    *,
    delta_min=None,
    gate=None,
    n0=361,
    nj=41,
    r_d=HULL_CONTACT_M,
    T=None,
    react=0.0,
):
    """Old model version of geometric_certificate.

    Args:
        os_boat (Vessel): Own vessel
        threats (list): Target vessels
        delta_min (float): Smallest lawful turn, rad, or None
        gate (object): Gate value, or None
        n0 (int): Number of headings tested
        nj (int): Number of time samples
        r_d (float): Contact distance, m
        T (float): Look ahead time, s, or None
        react (float): Reaction delay, s

    Returns:
        tuple: True if in deadlock, and the full measure
    """
    gate = _legacy_gate(gate)
    m = escape_measure(
        os_boat,
        list(threats),
        T=T,
        react=react,
        n0=n0,
        nj=nj,
        r_d=r_d,
        delta_min=delta_min,
        gate=gate,
    )
    return bool(m["deadlock"]), m


def _perturb(os_boat, rng, sigma_m, sigma_hdg):
    """Add random noise to the position and heading of a vessel.

    Args:
        os_boat (Vessel): Own vessel
        rng (Generator): Random generator
        sigma_m (float): Position noise, m
        sigma_hdg (float): Heading noise, rad

    Returns:
        Vessel: Noisy copy
    """
    return Vessel(
        os_boat.name,
        os_boat.x + rng.normal(0.0, sigma_m),
        os_boat.y + rng.normal(0.0, sigma_m),
        os_boat.v,
        os_boat.theta + rng.normal(0.0, sigma_hdg),
        os_boat.r_min,
    )


def certify(
    os_boat,
    threats,
    *,
    delta_min=None,
    gate=None,
    n_perturb=N_PERTURB,
    sigma_m=POSE_SIGMA_M,
    sigma_hdg=POSE_SIGMA_HEADING_RAD,
    n0=181,
    nj=21,
    r_d=HULL_CONTACT_M,
    react=0.0,
    rng=None,
    model=None,
):
    """Run both certificates and repeat the geometric one under pose noise.

    Args:
        os_boat (Vessel): Own vessel
        threats (list): Target vessels
        delta_min (float): Smallest lawful turn, rad, or None
        gate (object): Gate value, or None
        n_perturb (int): Number of noisy copies
        sigma_m (float): Position noise, m
        sigma_hdg (float): Heading noise, rad
        n0 (int): Number of headings tested
        nj (int): Number of time samples
        r_d (float): Contact distance, m
        react (float): Reaction delay, s
        rng (Generator): Random generator, or None
        model (str): Model name, or None for the default

    Returns:
        dict: Deontic flag, nominal geometric flag, share of noisy deadlocks and class
    """
    rng = np.random.default_rng() if rng is None else rng
    model = MODEL if model is None else model
    threats = list(threats)
    T = _duty.HORIZON_S if model == "manuscript" else _horizon_n([os_boat] + threats)

    deo = deontic_certificate(
        os_boat, threats, delta_min=delta_min, gate=gate, T=T, react=react, model=model
    )
    geo_nom, m = geometric_certificate(
        os_boat,
        threats,
        delta_min=delta_min,
        gate=gate,
        n0=n0,
        nj=nj,
        r_d=r_d,
        T=T,
        react=react,
        model=model,
    )

    hits = 0
    for _ in range(n_perturb):
        os_p = _perturb(os_boat, rng, sigma_m, sigma_hdg)
        hit, _ = geometric_certificate(
            os_p,
            threats,
            delta_min=delta_min,
            gate=gate,
            n0=n0,
            nj=nj,
            r_d=r_d,
            T=T,
            react=react,
            model=model,
        )
        hits += bool(hit)

    return dict(
        deontic=bool(deo),
        geometric_nominal=bool(geo_nom),
        p_hat=(hits / n_perturb if n_perturb else float(geo_nom)),
        n_perturb=int(n_perturb),
        sigma_m=float(sigma_m),
        alpha_col=float(m["alpha_col"]),
        alpha_phys=float(m["alpha_phys"]),
        alpha_over_kappa=float(
            m.get(
                "alpha_over_kappa",
                m["alpha_col"] / m["kappa"] if m.get("kappa") else 0.0,
            )
        ),
        kappa0=float(m.get("kappa0", m.get("kappa", 0.0))),
        T=float(m.get("T", T)),
        cls=str(m["cls"]) if "cls" in m else _manuscript_cls(m),
        model=model,
        n_threats=int(m["n_threats"]),
    )


def _manuscript_cls(m):
    """Name the class of a measure: ICS, deadlock or escapable.

    Args:
        m (dict): Measure

    Returns:
        str: Class name
    """
    if m.get("ics_bound"):
        return "ICS"
    return "deadlock" if m.get("physical_deadlock") else "escapable"


def is_certified(result, p):
    """Check whether a result counts as certified at level p.

    Args:
        result (dict): Output of certify
        p (float): Required share of noisy deadlocks

    Returns:
        bool: True if certified
    """
    return bool(result["deontic"] or (result["p_hat"] > 0.0 and result["p_hat"] >= p))


def p_curve(results, ps=None):
    """Compute the certified share for each level p.

    Args:
        results (list): Outputs of certify
        ps (ndarray): Levels to test, or None for 0 to 1

    Returns:
        tuple: Levels and certified share
    """
    ps = np.linspace(0.0, 1.0, 21) if ps is None else np.asarray(ps, float)
    results = list(results)
    if not results:
        return ps, np.zeros_like(ps)
    frac = np.array([np.mean([is_certified(r, float(p)) for r in results]) for p in ps])
    return ps, frac

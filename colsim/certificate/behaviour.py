from __future__ import annotations

import math
from collections import deque

import numpy as np

from . import duty as _duty
from .escape_angle import HOLD_RAD, _wrap, cpa

GIVE_WAY = ("Head-on", "Crossing_GiveWay", "Overtaking_GiveWay")
STAND_ON = ("Crossing_StandOn", "Being_Overtaken_StandOn")

CLASSES = ("compliant", "breakdown", "adversarial")

MOTION_CLASSES = CLASSES + ("interceptor", "late", "wide_berth", "mistaken", "erratic")
DEVIANT_CLASSES = ("mistaken", "breakdown", "erratic", "interceptor")
ERRATIC_MAX_TURN_RAD = math.radians(30.0)
ERRATIC_INTERVAL_STEPS = (3, 6)

LATE_ACTION_STEPS = 12
WIDE_BERTH_MULT = 20.0


def class_mix(name):
    """Read a behaviour mix name or spec into class weights.

    Args:
        name (str): Class name, uniform, or a spec like a=0.5,b=0.5

    Returns:
        dict: Weight per class, summing to 1
    """
    if "=" in str(name):
        w = {}
        for part in str(name).split(","):
            k, _sep, p = part.partition("=")
            k = k.strip()
            if k not in MOTION_CLASSES:
                raise ValueError(
                    f"unknown behaviour {k!r}; want one of {MOTION_CLASSES}"
                )
            w[k] = float(p)
            if w[k] < 0.0:
                raise ValueError(f"behaviour mix {name!r}: negative weight for {k!r}")
        if sum(w.values()) <= 0.0:
            raise ValueError(f"behaviour mix {name!r} carries no positive weight")
        return {k: v / sum(w.values()) for k, v in w.items()}
    if name == "uniform":
        return {c: 1.0 / len(CLASSES) for c in CLASSES}
    if name not in MOTION_CLASSES:
        raise ValueError(
            f"unknown behaviour mix {name!r}; want 'uniform' or one of "
            f"{MOTION_CLASSES}"
        )
    return {name: 1.0}


APPARENT_TURN_RAD = np.radians(15.0)
APPARENT_DV_MPS = 1.0
WINDOW_STEPS = 6


def _uniform_prior():
    """Give the same weight to every class.

    Returns:
        dict: Weight per class
    """
    return {c: 1.0 / len(CLASSES) for c in CLASSES}


def _normalise(w):
    """Scale weights so they sum to 1.

    Args:
        w (dict): Weight per class

    Returns:
        dict: Normalised weights
    """
    total = float(sum(w.values()))
    if total <= 0:
        return _uniform_prior()
    return {k: v / total for k, v in w.items()}


class BehaviourTracker:

    def __init__(self, window=WINDOW_STEPS):
        """Create an empty history of one target.

        Args:
            window (int): Number of steps kept
        """
        self.window = int(window)
        self.hist = deque(maxlen=self.window + 1)

    def observe(self, t, ts, os_boat):
        """Add the current state of the target to the history.

        Args:
            t (float): Time, s
            ts (Vessel): Target vessel
            os_boat (Vessel): Own vessel
        """
        _tcpa, dcpa, _vrel = cpa(os_boat, ts)
        self.hist.append(
            (
                float(t),
                float(ts.x),
                float(ts.y),
                float(ts.v),
                float(ts.theta),
                float(dcpa),
            )
        )

    @property
    def ready(self):
        """Check that the history has at least two states.

        Returns:
            bool: True if ready
        """
        return len(self.hist) >= 2

    def deltas(self):
        """Compute the change in heading, speed and DCPA over the window.

        Returns:
            tuple: Heading change (rad), speed change (m/s) and DCPA change (m)
        """
        if not self.ready:
            return 0.0, 0.0, 0.0
        t0, _x0, _y0, v0, th0, dc0 = self.hist[0]
        t1, _x1, _y1, v1, th1, dc1 = self.hist[-1]
        return _wrap(th1 - th0), (v1 - v0), (dc1 - dc0)

    def span_steps(self):
        """Count the steps covered by the history.

        Returns:
            int: Number of steps
        """
        return max(len(self.hist) - 1, 0)


def classify_behaviour(
    tracker,
    situation,
    *,
    turn_sign=-1.0,
    apparent_turn=APPARENT_TURN_RAD,
    apparent_dv=APPARENT_DV_MPS,
    hold_band=HOLD_RAD,
):
    """Estimate how likely the target is compliant, broken down or adversarial.

    Args:
        tracker (BehaviourTracker): Target history
        situation (str): Situation of the target
        turn_sign (float): Sign of the correct turn
        apparent_turn (float): Smallest turn that counts, rad
        apparent_dv (float): Smallest speed change that counts, m/s
        hold_band (float): Heading band for keeping course, rad

    Returns:
        dict: Probability per class
    """
    if not tracker.ready:
        return _uniform_prior()

    d_head, d_speed, d_dcpa = tracker.deltas()
    turned = abs(d_head) >= apparent_turn
    turned_correct_side = turned and (
        (situation == "Overtaking_GiveWay" and d_dcpa >= 0.0)
        or math.copysign(1.0, d_head) == math.copysign(1.0, turn_sign)
    )
    slackened = d_speed <= -apparent_dv
    still = (abs(d_head) < hold_band) and (abs(d_speed) < apparent_dv)
    closing = d_dcpa < 0.0

    w = {c: 0.0 for c in CLASSES}

    if situation not in GIVE_WAY and situation not in STAND_ON:
        return {"compliant": 1.0, "breakdown": 0.0, "adversarial": 0.0}

    if situation in GIVE_WAY:
        if turned_correct_side or slackened:
            w["compliant"] += 3.0
        if still:
            w["breakdown"] += 2.0
            w["compliant"] += 0.5
        if turned and not turned_correct_side:
            w["adversarial"] += 1.5
        if closing and not (turned_correct_side or slackened):
            w["adversarial"] += 1.5
        if not any(w.values()):
            w["compliant"] += 1.0
    else:
        if still:
            w["compliant"] += 2.0
            w["breakdown"] += 2.0
        if turned and closing:
            w["adversarial"] += 3.0
        elif turned:
            w["compliant"] += 2.0

    dist = _normalise(w)

    frac = min(tracker.span_steps() / max(tracker.window, 1), 1.0)
    prior = _uniform_prior()
    return {c: frac * dist[c] + (1.0 - frac) * prior[c] for c in CLASSES}


class BehaviourClassifier:

    def __init__(self, window=WINDOW_STEPS, gate=None):
        """Create a classifier with one tracker per target.

        Args:
            window (int): Number of steps kept
            gate (dict): Gate settings, or None
        """
        self.window = int(window)
        self.gate = dict(gate or {})
        self._trackers = {}

    def _key(self, ts, i):
        """Build the tracker key of a target.

        Args:
            ts (Vessel): Target vessel
            i (int): Target index

        Returns:
            str: Tracker key
        """
        mmsi = getattr(ts, "mmsi", None)
        return f"mmsi:{mmsi}" if mmsi is not None else f"i:{i}"

    def observe(self, t, os_boat, threats):
        """Add the current state of every target.

        Args:
            t (float): Time, s
            os_boat (Vessel): Own vessel
            threats (list): Target vessels
        """
        for i, ts in enumerate(threats):
            k = self._key(ts, i)
            if k not in self._trackers:
                self._trackers[k] = BehaviourTracker(self.window)
            self._trackers[k].observe(t, ts, os_boat)

    def classify(self, os_boat, threats):
        """Estimate the class probabilities of every target.

        Args:
            os_boat (Vessel): Own vessel
            threats (list): Target vessels

        Returns:
            list: Probability per class, per target
        """
        out = []
        for i, ts in enumerate(threats):
            tracker = self._trackers.get(self._key(ts, i))
            if tracker is None:
                out.append(_uniform_prior())
                continue
            sit = _duty.situation(ts, os_boat, gate=self.gate.get("dcpa_gate"))
            out.append(classify_behaviour(tracker, sit))
        return out

    def as_array(self, os_boat, threats):
        """Same as classify, as an array.

        Args:
            os_boat (Vessel): Own vessel
            threats (list): Target vessels

        Returns:
            ndarray: One row per target, one column per class
        """
        return np.array(
            [[d[c] for c in CLASSES] for d in self.classify(os_boat, threats)],
            dtype=np.float32,
        )

    def spans(self, threats):
        """Count the history steps of every target.

        Args:
            threats (list): Target vessels

        Returns:
            list: Number of steps per target
        """
        return [
            (self._trackers[k].span_steps() if k in self._trackers else 0)
            for k in (self._key(ts, i) for i, ts in enumerate(threats))
        ]

    def reset(self):
        """Forget every target."""
        self._trackers.clear()


P_ON = 0.8
P_OFF = 0.5
N_HYSTERESIS = 3


class Rule17Relaxation:

    def __init__(
        self,
        *,
        p_on=P_ON,
        p_off=P_OFF,
        n_hysteresis=N_HYSTERESIS,
        window=WINDOW_STEPS,
        gate=None,
    ):
        """Set the thresholds for relaxing Rule 17 when a give way target fails to act.

        Args:
            p_on (float): Probability to turn the relaxation on
            p_off (float): Probability to turn it off
            n_hysteresis (int): Steps below p_off before turning off
            window (int): Number of steps kept
            gate (dict): Gate settings, or None
        """
        if not 0.0 < p_off <= p_on <= 1.0:
            raise ValueError(f"need 0 < p_off <= p_on <= 1, got {p_off} / {p_on}")
        if p_on <= 2.0 / len(CLASSES):
            raise ValueError(
                f"p_on={p_on} is at or below the uniform prior's not-acting mass "
                f"({2.0 / len(CLASSES):.3f}); the flip would engage without evidence"
            )
        self.p_on = float(p_on)
        self.p_off = float(p_off)
        self.n_hysteresis = int(n_hysteresis)
        self.window = int(window)
        self.classifier = BehaviourClassifier(window=window, gate=gate)
        self.gate = dict(gate or {})
        self._engaged = {}
        self._below = {}

    def reset(self):
        """Forget every target and every flag."""
        self.classifier.reset()
        self._engaged.clear()
        self._below.clear()

    def update(self, t, os_boat, threats):
        """Update the relaxation flag of every target.

        Args:
            t (float): Time, s
            os_boat (Vessel): Own vessel
            threats (list): Target vessels

        Returns:
            list: Relaxation flag per target
        """
        self.classifier.observe(t, os_boat, threats)
        dists = self.classifier.classify(os_boat, threats)
        spans = self.classifier.spans(threats)
        out = []
        for i, ts in enumerate(threats):
            key = self.classifier._key(ts, i)
            if (
                _duty.situation(ts, os_boat, gate=self.gate.get("dcpa_gate"))
                not in GIVE_WAY
            ):
                self._engaged[key] = False
                self._below[key] = 0
                out.append(False)
                continue
            d = dists[i]
            bad = float(d["breakdown"] + d["adversarial"])
            engaged = self._engaged.get(key, False)
            if not engaged:
                if spans[i] >= self.window and bad >= self.p_on:
                    engaged = True
                    self._below[key] = 0
            else:
                if bad <= self.p_off:
                    self._below[key] = self._below.get(key, 0) + 1
                    if self._below[key] >= self.n_hysteresis:
                        engaged = False
                        self._below[key] = 0
                else:
                    self._below[key] = 0
            self._engaged[key] = engaged
            out.append(engaged)
        return out

    def flags(self, threats):
        """Get the current relaxation flag of every target.

        Args:
            threats (list): Target vessels

        Returns:
            list: Relaxation flag per target
        """
        return [
            self._engaged.get(self.classifier._key(ts, i), False)
            for i, ts in enumerate(threats)
        ]

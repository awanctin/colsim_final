from __future__ import annotations

from enum import Enum

from ..certificate.escape_angle import (
    ONE_NAUTICAL_MILE_M,
    R_VESSEL_M,
    TS_SPEED_RANGE_MPS,
    VISIBLE_RANGE_M,
)


class RuleMode(Enum):

    COLREGS = "colregs"
    RELAXED = "relaxed"


CAPABILITY_KNOBS: frozenset = frozenset(
    {
        "SAFE_DCPA_M",
        "SAFE_TCPA_MIN_S",
        "VISIBLE_RANGE_M",
        "D_SAFE_MULT",
        "DOMAIN_AXES_LENGTHS",
        "LOCAL_REPLAN_INTERVAL",
        "K_COMMIT",
    }
)


def apply_capability(vessels, capability):
    """Set each capability knob on every vessel.

    Args:
        vessels (list): Vessels to update
        capability (dict): Knob name to value, or None
    """
    if not capability:
        return
    unknown = set(capability) - CAPABILITY_KNOBS
    if unknown:
        raise ValueError(
            f"capability: unknown knob(s) {sorted(unknown)}; "
            f"allowed: {sorted(CAPABILITY_KNOBS)}"
        )
    for vessel in vessels:
        for knob, value in capability.items():
            setattr(vessel, knob, value)


KNOT_TO_MPS: float = ONE_NAUTICAL_MILE_M / 3600.0
MPS_TO_KNOT: float = 1.0 / KNOT_TO_MPS


SIM_DT_S: float = 10.0

NO_CLOSING_TCPA_S: float = 9999.0

THREAT_FILTER_MULT: float = 3.0
LATCH_DCPA_M: float = ONE_NAUTICAL_MILE_M
LATCH_TCPA_S: float = 2400.0
LATCH_TCPA_MIN_S: float = 900.0
D_SAFE_MULT: float = 10.0
DUTY_GATE_M: float = 500.0
TUBE_DCPA_MULT: float = 3.0
TUBE_TCPA_S: float = 900.0
MAP_HALF_EXTENT_M: float = 20000.0
GOAL_HORIZON_MARGIN: float = 1.10

TS_DIST_RANGE_M: tuple = (3000.0, 8000.0)
DEFAULT_OS_SPEED_MPS: float = 25.0 * KNOT_TO_MPS

MAKING_WAY_MPS: float = 2.0
MAKING_WAY_BAND_MPS: float = 0.5

OVERTAKING_DOUBT_BAND_DEG: float = 5.0

WAYPOINT_ACCEPT_M: float = 400.0

COLLISION_FLAG_M: float = 200.0


BREACH_CPA_M: float = 500.0

BREACH_DWELL_STEPS: int = 2
BREACH_MARGIN_M: float = 50.0


SURGE_TIME_CONSTANT_S: float = 60.0

SPEED_MAX_MPS: float = DEFAULT_OS_SPEED_MPS
SPEED_MIN_MPS: float = 2.0

SURGE_ACCEL_MAX_MPS2: float = 0.10
SURGE_DECEL_MAX_MPS2: float = -0.30
SURGE_REVERSE_DECEL_MPS2: float = -0.50

CRI_ARM: float = 0.50
CRI_STOP: float = 0.75
CRI_REV: float = 0.90

T_HORIZON_S: float = 300.0
T_STOP_S: float = 120.0
T_REV_S: float = 60.0


DV_MIN_APPARENT_MPS: float = 1.0
T_DWELL_S: float = 60.0

CLEAR_RANGE_M: float = ONE_NAUTICAL_MILE_M

CRI_W_DCPA: float = 0.5
CRI_W_TCPA: float = 0.5

"""Observation / action / stage contract shared by data collection, training and deployment.

Every consumer — collector, exporter, trainer, bridge server and policy client — imports
this module and validates feature *names and dimensions* against it, failing closed on any
mismatch. Earlier iterations of the project (18D, then a 28D layout with the task phase
embedded in the state) showed that silent schema drift yields a policy that loads cleanly
but behaves incorrectly; a single source of truth removes that failure mode.

Mirrors `excavator_common/vla_observation_contract.py` and
`excavator_common/deployment_contract.py` from the development repository.
"""

from __future__ import annotations

import math
from typing import Sequence

# --------------------------------------------------------------------------------------
# Joints
# --------------------------------------------------------------------------------------

# Canonical order used by the dataset, the model and the bridge. The USD articulation
# reports DOFs in its own order ([bucket, arm, boom, swing] in some builds), so the bridge
# resolves indices by *name* rather than by position.
DOF_ORDER: tuple[str, ...] = ("swing", "boom", "arm", "bucket")

# Runtime limits override the ±90° URDF limits exported from SolidWorks.
JOINT_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "swing": (-180.0, 180.0),
    "boom": (-75.0, 75.0),
    "arm": (-95.0, 95.0),
    "bucket": (-120.0, 90.0),
}

# Conservative limits used by the planner (a few degrees inside the hard limits).
PLANNER_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "swing": (-180.0, 180.0),
    "boom": (-70.0, 74.0),
    "arm": (-89.0, 92.0),
    "bucket": (-118.0, 88.0),
}

# Max commanded joint speed for the scripted expert (rad/s).
DQ_MAX: dict[str, float] = {"swing": 1.0, "boom": 0.7, "arm": 0.8, "bucket": 1.0}

# --------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------

PHYSICS_HZ = 60.0          # PhysX step rate
POLICY_FPS = 10.0          # dataset / policy rate -> 6 physics ticks per action
PROTOCOL_VERSION = 2

# --------------------------------------------------------------------------------------
# Task stages (supervision labels only — never a policy input in the deployed model)
# --------------------------------------------------------------------------------------

# Launchers refuse dataset metadata whose stage order differs from this tuple.
STAGE_NAMES: tuple[str, ...] = (
    "pre_dig",                # 0  move above the chosen dig point
    "approach_contact",       # 1  lower bucket to the surface at attack angle
    "insert_cut",             # 2  push the tip into the pile
    "pull_mid_cut",           # 3  drag through the material
    "curl_to_hold_material",  # 4  curl bucket to trap the load
    "pull_exit_cut",          # 5  exit the cut
    "secure_load",            # 6  hold while particles settle
    "lift_carry",             # 7  lift with bucket locked at carry angle
    "loaded_transit",         # 8  clearance route / swing toward the truck
    "unload_to_bin",          # 9  open bucket over the truck bed
)
STAGE_ID = {name: i for i, name in enumerate(STAGE_NAMES)}

# --------------------------------------------------------------------------------------
# State vectors
# --------------------------------------------------------------------------------------

# 28D policy state. All spatial terms are expressed in the *upper-structure* frame
# (heading = base_yaw + swing), so the policy is invariant to where the excavator is parked
# and which way it initially faces.
STATE_NAMES: tuple[str, ...] = (
    "swing", "boom", "arm", "bucket",
    "swing_velocity", "boom_velocity", "arm_velocity", "bucket_velocity",
    "swing_tracking_error", "boom_tracking_error", "arm_tracking_error", "bucket_tracking_error",
    "swing_prev_action", "boom_prev_action", "arm_prev_action", "bucket_prev_action",
    "bucket_pour_axis_forward", "bucket_pour_axis_up",
    "dig_delta_forward", "dig_delta_left", "dig_delta_up",
    "unload_delta_forward", "unload_delta_left", "unload_delta_up",
    "truck_heading_relative_sin", "truck_heading_relative_cos",
    "bucket_fill_fraction", "bucket_fill_rate",
)

EFFORT_NAMES: tuple[str, ...] = tuple(f"{j}_measured_effort" for j in DOF_ORDER)
ACTION_NAMES: tuple[str, ...] = tuple(f"{j}_cmd_velocity" for j in DOF_ORDER)
CAMERA_KEYS: tuple[str, ...] = ("observation.images.0", "observation.images.1", "observation.images.2")
CAMERA_SHAPE = (3, 256, 256)

STATE_SCHEMA = "excavator_state_v4_28d_plus_4effort_categorical_phase10"

BUCKET_CAPACITY_PARTICLES = 6400.0   # normalises fill fraction
FILL_RATE_TAU_S = 0.4                # EMA time constant for fill rate


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def to_heading_frame(delta_world_xy: Sequence[float], heading: float) -> tuple[float, float]:
    """Rotate a world XY delta into (forward, left) of a frame with yaw `heading`."""
    dx, dy = float(delta_world_xy[0]), float(delta_world_xy[1])
    c, s = math.cos(heading), math.sin(heading)
    return (c * dx + s * dy, -s * dx + c * dy)


class FillRateEMA:
    """Bucket fill rate (fraction / s) smoothed with a first-order EMA."""

    def __init__(self, tau_s: float = FILL_RATE_TAU_S, clip: float = 5.0):
        self.tau_s, self.clip = tau_s, clip
        self._prev: float | None = None
        self.value = 0.0

    def update(self, fill_fraction: float, dt: float) -> float:
        if self._prev is not None and dt > 0:
            raw = (fill_fraction - self._prev) / dt
            alpha = 1.0 - math.exp(-dt / self.tau_s)
            self.value += alpha * (raw - self.value)
            self.value = max(-self.clip, min(self.clip, self.value))
        self._prev = fill_fraction
        return self.value


def build_state_28d(
    q: Sequence[float],
    dq: Sequence[float],
    q_cmd: Sequence[float],
    prev_action: Sequence[float],
    base_yaw: float,
    bucket_tip: Sequence[float],
    bucket_load: Sequence[float],
    bucket_pour: Sequence[float],
    dig_target: Sequence[float],
    unload_landing: Sequence[float],
    truck_yaw: float,
    bucket_particles: float,
    fill_rate: float,
) -> list[float]:
    """Assemble the 28D state. All spatial terms live in the upper-structure frame."""
    heading = base_yaw + float(q[0])
    q_err = [wrap_angle(q_cmd[0] - q[0])] + [float(c - m) for c, m in zip(q_cmd[1:], q[1:])]

    pour_f, _ = to_heading_frame((bucket_pour[0] - bucket_load[0], bucket_pour[1] - bucket_load[1]), heading)
    pour_up = float(bucket_pour[2] - bucket_load[2])
    n = math.hypot(pour_f, pour_up) or 1.0

    dig_f, dig_l = to_heading_frame((dig_target[0] - bucket_tip[0], dig_target[1] - bucket_tip[1]), heading)
    unl_f, unl_l = to_heading_frame((unload_landing[0] - bucket_load[0], unload_landing[1] - bucket_load[1]), heading)
    rel_truck = wrap_angle(truck_yaw - heading)

    state = (
        list(map(float, q)) + list(map(float, dq)) + q_err + list(map(float, prev_action))
        + [pour_f / n, pour_up / n]
        + [dig_f, dig_l, float(dig_target[2] - bucket_tip[2])]
        + [unl_f, unl_l, float(unload_landing[2] - bucket_load[2])]
        + [math.sin(rel_truck), math.cos(rel_truck)]
        + [min(1.5, bucket_particles / BUCKET_CAPACITY_PARTICLES), fill_rate]
    )
    if len(state) != len(STATE_NAMES) or not all(math.isfinite(v) for v in state):
        raise ValueError("observation.state violates the 28D contract")
    return state


def validate_feature_names(kind: str, expected: Sequence[str], actual: Sequence[str]) -> None:
    """Fail closed. Truncating or padding features is explicitly unsupported."""
    if tuple(expected) != tuple(actual):
        missing = [n for n in expected if n not in actual]
        extra = [n for n in actual if n not in expected]
        raise ValueError(
            f"{kind} contract mismatch: expected {len(expected)}D, got {len(actual)}D; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

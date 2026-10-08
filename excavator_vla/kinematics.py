"""Excavator kinematics for the scripted expert.

The 4-DOF arm decomposes neatly:
  * swing  = rotation about world Z  ->  solved in closed form with atan2
  * boom / arm / bucket = a planar 3-link chain in the swing plane

For the planar chain we *search* over the bucket's absolute angle a3 (the variable the
expert actually cares about — attack angle, carry angle, dump angle), subtract the bucket
link to get a wrist target, then solve boom+arm analytically as a 2-link problem with both
elbow branches. Candidates are filtered (body below ground, tip too deep, joint limits)
and the cheapest by weighted joint motion wins.

Condensed from `solve_priority_ik_to_target` / `two_link_ik_2d` in the runtime and
`floor_safe_effector_target_z` in `ik_calculation.py`.

Simplification: the planar model uses URDF link lengths and treats each joint zero as
collinear with its parent link. The production runtime instead reads FK from the USD
articulation (including mesh/frame offsets), so absolute feasible radii differ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .contract import PLANNER_LIMITS_DEG, wrap_angle

DEG = math.pi / 180.0


@dataclass
class ExcavatorGeometry:
    """Planar link lengths (m) and the swing-plane pivot of the boom (from URDF_real3)."""
    boom_pivot: tuple[float, float, float] = (0.53, 0.0, 1.37)   # base->swing 0.60 + swing->boom 0.77
    boom_len: float = 5.715                                       # boom joint -> arm joint
    arm_len: float = 2.95                                         # arm joint -> bucket joint
    # Bucket reference points in the bucket-link frame (runtime constants)
    tip_local: tuple[float, float, float] = (0.75, 0.0, -0.18)
    load_local: tuple[float, float, float] = (0.35, 0.0, 0.08)
    pour_local: tuple[float, float, float] = (0.85, 0.0, 0.30)
    joint_weights: tuple[float, float, float, float] = (1.15, 1.0, 0.9, 0.7)
    limits_deg: dict = field(default_factory=lambda: dict(PLANNER_LIMITS_DEG))


def two_link_ik_2d(target: np.ndarray, l1: float, l2: float, seed: Sequence[float]):
    """Both elbow solutions for a planar 2-link chain, sorted by distance to `seed`."""
    r = float(np.hypot(target[0], target[1]))
    if r > l1 + l2 or r < abs(l1 - l2):
        return []
    c = (r * r - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    c = max(-1.0, min(1.0, c))
    base = math.atan2(float(target[1]), float(target[0]))
    sols = []
    for rel in (math.acos(c), -math.acos(c)):
        a1 = base - math.atan2(l2 * math.sin(rel), l1 + l2 * math.cos(rel))
        a2 = a1 + rel
        cost = abs(wrap_angle(a1 - seed[0])) + abs(wrap_angle(a2 - seed[1]))
        sols.append((cost, a1, a2))
    return sorted(sols)


def forward_planar(geo: ExcavatorGeometry, a1: float, a2: float, a3: float) -> dict[str, np.ndarray]:
    """Absolute link angles -> 2D points (radial, z) in the swing plane."""
    p0 = np.array([geo.boom_pivot[0], geo.boom_pivot[2]])
    p1 = p0 + geo.boom_len * np.array([math.cos(a1), math.sin(a1)])
    p2 = p1 + geo.arm_len * np.array([math.cos(a2), math.sin(a2)])
    rot = np.array([[math.cos(a3), -math.sin(a3)], [math.sin(a3), math.cos(a3)]])
    pts = {"boom_pivot": p0, "arm_joint": p1, "bucket_joint": p2}
    for name, loc in (("tip", geo.tip_local), ("load", geo.load_local), ("pour", geo.pour_local)):
        pts[name] = p2 + rot @ np.array([loc[0], loc[2]])
    return pts


def chain_to_joints(a1: float, a2: float, a3: float) -> tuple[float, float, float]:
    """Absolute link angles -> relative joint angles (boom, arm, bucket)."""
    return a1, wrap_angle(a2 - a1), wrap_angle(a3 - a2)


@dataclass
class IKResult:
    q: np.ndarray            # [swing, boom, arm, bucket] rad
    cost: float
    points: dict


def solve_bucket_ik(
    geo: ExcavatorGeometry,
    target_xyz: Sequence[float],
    q_seed: Sequence[float],
    bucket_world_angles_deg: Iterable[float],
    effector: str = "tip",
    ground_z: float = 0.0,
    min_body_clearance: float = 0.15,
) -> IKResult | None:
    """Priority IK: put `effector` on `target_xyz` while trying each bucket world angle."""
    tx, ty, tz = map(float, target_xyz)
    swing = math.atan2(ty, tx)
    radial = math.hypot(tx, ty)
    best: IKResult | None = None

    loc = {"tip": geo.tip_local, "load": geo.load_local, "pour": geo.pour_local}[effector]
    for a3_deg in bucket_world_angles_deg:
        a3 = a3_deg * DEG
        rot = np.array([[math.cos(a3), -math.sin(a3)], [math.sin(a3), math.cos(a3)]])
        wrist = np.array([radial, tz]) - rot @ np.array([loc[0], loc[2]])
        rel = wrist - np.array([geo.boom_pivot[0], geo.boom_pivot[2]])
        for _, a1, a2 in two_link_ik_2d(rel, geo.boom_len, geo.arm_len, (q_seed[1], q_seed[1] + q_seed[2])):
            boom, arm, bucket = chain_to_joints(a1, a2, a3)
            q = np.array([swing, boom, arm, bucket])
            if not _within_limits(geo, q):
                continue
            pts = forward_planar(geo, a1, a2, a3)
            if min(pts["arm_joint"][1], pts["bucket_joint"][1]) < ground_z + min_body_clearance:
                continue  # reject: boom/arm body below clearance
            dq = np.abs([wrap_angle(a - b) for a, b in zip(q, q_seed)])
            cost = float(np.dot(geo.joint_weights, dq))
            if best is None or cost < best.cost:
                best = IKResult(q=q, cost=cost, points=pts)
    return best


def _within_limits(geo: ExcavatorGeometry, q: np.ndarray) -> bool:
    for name, val in zip(("swing", "boom", "arm", "bucket"), q):
        lo, hi = geo.limits_deg[name]
        if not (lo * DEG <= val <= hi * DEG):
            return False
    return True


def floor_safe_effector_target_z(
    geo: ExcavatorGeometry,
    effector: str,
    bucket_world_angle_deg: float,
    ground_z: float,
    max_tip_depth: float,
    max_body_depth: float,
    margin: float = 0.015,
) -> float:
    """Lowest target Z for `effector` that keeps every bucket point above its floor limit.

    For a fixed bucket world angle the bucket is a rigid body, so each reference point
    has a constant Z offset from the effector. The binding constraint is whichever
    point would punch through its floor first.
    """
    a3 = bucket_world_angle_deg * DEG
    rot = np.array([[math.cos(a3), -math.sin(a3)], [math.sin(a3), math.cos(a3)]])
    z_of = {n: (rot @ np.array([p[0], p[2]]))[1]
            for n, p in (("tip", geo.tip_local), ("load", geo.load_local), ("pour", geo.pour_local))}
    floors = {"tip": ground_z - max_tip_depth, "load": ground_z - max_body_depth, "pour": ground_z - max_body_depth}
    ref = z_of[effector]
    return max(floors[n] + margin - (z_of[n] - ref) for n in z_of)

"""Dig-target selection and the staged scoop plan.

The expert has privileged access to every sand particle. It:
  1. grids the pile, scores each cell (reach, expected fill, density, depth, ...),
  2. turns the best few into full 9-stage plans via IK,
  3. keeps the plan with the lowest expert cost,
  4. picks an unload landing on a 5x5 grid over the truck bed (lowest fill first).

Condensed from `rank_dig_targets`, `dig_plan_specs_from_candidate` and
`choose_unload_landing` in the original runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..kinematics import ExcavatorGeometry, floor_safe_effector_target_z, solve_bucket_ik

# Score weights for dig candidates
W = {"reach": 32, "fill": 26, "swept_density": 22, "depth": 12, "center": 10, "approach": 8, "motion": 6}
GRID_N = 7
TOP_K_CANDIDATES = 8
FULL_PLAN_TOP_K = 3
DENSITY_RADIUS_M = 0.34
MIN_LOCAL_PARTICLES = 24
REACH_WINDOW_M = (5.6, 7.45)

# Bucket world angles (deg) per cut phase, before slope correction (±18° max)
BUCKET_ATTACK, BUCKET_INSERT, BUCKET_PULL, BUCKET_EXIT = -52.0, -48.0, -92.0, -96.0
CURL_HOLD_TARGET_DEG = -120.0
DIG_MAX_TIP_DEPTH, DIG_MAX_BODY_DEPTH = 0.28, 0.10


@dataclass
class DigCandidate:
    xyz: np.ndarray
    score: float
    local_particles: int
    detail: dict


@dataclass
class StageTarget:
    stage: str
    target: np.ndarray | None     # world xyz for the effector (None = keep position)
    bucket_world_deg: float | None
    bucket_mode: str | None       # None | "hold" (lock current) | "carry" (mouth-up 18°)
    effector: str                 # "tip" or "load"
    duration_s: float


def rank_dig_targets(particles: np.ndarray, pile_center: np.ndarray, pile_half: float,
                     q_now: np.ndarray, ground_z: float = 0.0) -> list[DigCandidate]:
    """Score a GRID_N x GRID_N grid of candidate dig points over the pile."""
    xs = np.linspace(-pile_half, pile_half, GRID_N) + pile_center[0]
    ys = np.linspace(-pile_half, pile_half, GRID_N) + pile_center[1]
    out: list[DigCandidate] = []
    for x in xs:
        for y in ys:
            d = np.hypot(particles[:, 0] - x, particles[:, 1] - y)
            local = particles[d < DENSITY_RADIUS_M]
            if len(local) < MIN_LOCAL_PARTICLES:
                continue
            surface_z = float(np.percentile(local[:, 2], 90))
            r = math.hypot(x, y)
            reach = 1.0 - min(1.0, abs(r - np.mean(REACH_WINDOW_M)) / (0.5 * np.ptp(REACH_WINDOW_M)))
            # particles a bucket would sweep when dragged toward the cab
            toward = -np.array([x, y]) / max(r, 1e-6)
            swept = particles[(np.abs(np.cross(particles[:, :2] - [x, y], toward)) < 0.4)
                              & ((particles[:, :2] - [x, y]) @ toward > 0)
                              & ((particles[:, :2] - [x, y]) @ toward < 1.2)]
            swing_goal = math.atan2(y, x)
            parts = {
                "reach": reach,
                "fill": min(1.0, len(local) / 400.0),
                "swept_density": min(1.0, len(swept) / 1200.0),
                "depth": min(1.0, (surface_z - ground_z) / 0.6),
                "center": 1.0 - min(1.0, math.dist((x, y), pile_center[:2]) / (pile_half * 1.42)),
                "approach": 1.0,  # simplified: real code checks approach corridor obstacles
                "motion": 1.0 - min(1.0, abs(swing_goal - q_now[0]) / math.pi),
            }
            score = sum(W[k] * v for k, v in parts.items())
            out.append(DigCandidate(np.array([x, y, surface_z]), score, len(local), parts))
    out.sort(key=lambda c: -c.score)
    return out[:TOP_K_CANDIDATES]


def build_stage_plan(geo: ExcavatorGeometry, cand: DigCandidate, unload_xyz: np.ndarray,
                     ground_z: float = 0.0, insert_depth: float = 0.18) -> list[StageTarget]:
    """Nine stage targets for one scoop (loaded_transit is inserted later by clearance routing)."""
    t = cand.xyz
    outward = t[:2] / max(np.linalg.norm(t[:2]), 1e-6)   # away from the cab

    def at(dist_out: float, z: float) -> np.ndarray:
        return np.array([*(t[:2] + outward * dist_out), z])

    # Never let any bucket point punch through the floor: insert depth is clamped by the
    # floor-safe Z for the bucket angle actually used in that stage.
    insert_z = max(t[2] - insert_depth, floor_safe_effector_target_z(
        geo, "tip", BUCKET_INSERT, ground_z, DIG_MAX_TIP_DEPTH, DIG_MAX_BODY_DEPTH))
    mid_z = max(t[2] - 0.6 * insert_depth, floor_safe_effector_target_z(
        geo, "load", BUCKET_PULL, ground_z, DIG_MAX_TIP_DEPTH, DIG_MAX_BODY_DEPTH))

    return [
        StageTarget("pre_dig",               at(0.25, t[2] + 0.55), -25.0, None, "tip", 1.10),
        StageTarget("approach_contact",      at(0.25, t[2] + 0.02), BUCKET_ATTACK, None, "tip", 1.00),
        StageTarget("insert_cut",            at(0.10, insert_z),    BUCKET_INSERT, None, "tip", 1.15),
        StageTarget("pull_mid_cut",          at(-0.45, mid_z),      BUCKET_PULL, None, "load", 2.20),
        StageTarget("curl_to_hold_material", at(-0.60, mid_z + 0.05), CURL_HOLD_TARGET_DEG, None, "load", 1.45),
        StageTarget("pull_exit_cut",         at(-0.80, t[2] + 0.15), BUCKET_EXIT, None, "load", 1.45),
        StageTarget("secure_load",           None, None, "hold", "load", 0.85),
        StageTarget("lift_carry",            at(-0.80, t[2] + 1.40), None, "hold", "load", 1.20),
        StageTarget("unload_to_bin",         unload_xyz, None, "carry", "load", 1.35),
    ]


def plan_cost(geo: ExcavatorGeometry, plan: list[StageTarget], q0: np.ndarray) -> tuple[float, list[np.ndarray]] | None:
    """IK every stage in sequence; total weighted joint motion, or None if infeasible."""
    q, total, qs = q0.copy(), 0.0, []
    for st in plan:
        if st.target is None:
            qs.append(q.copy()); continue
        angles = [st.bucket_world_deg] if st.bucket_world_deg is not None else np.arange(-140, 41, 10)
        sol = solve_bucket_ik(geo, st.target, q, angles, effector=st.effector)
        if sol is None:
            return None
        total += sol.cost
        q = sol.q
        qs.append(q.copy())
    return total, qs


def choose_unload_landing(bed_center: np.ndarray, bed_yaw: float, bed_half: float,
                          fill_height_fn, wall_margin: float = 0.60, release_h: float = 1.45) -> np.ndarray:
    """5x5 grid over the truck bed: lowest fill height, then nearest the centre."""
    c, s = math.cos(bed_yaw), math.sin(bed_yaw)
    best, best_key = None, None
    for u in np.linspace(-bed_half + wall_margin, bed_half - wall_margin, 5):
        for v in np.linspace(-bed_half + wall_margin, bed_half - wall_margin, 5):
            p = bed_center[:2] + np.array([c * u - s * v, s * u + c * v])
            key = (round(fill_height_fn(p), 2), math.hypot(u, v))
            if best_key is None or key < best_key:
                best, best_key = p, key
    return np.array([best[0], best[1], bed_center[2] + release_h])

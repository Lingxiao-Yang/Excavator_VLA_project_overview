"""Reproducible scene randomization for parallel data collection.

Every (run, worker, attempt) triple maps to a stable 64-bit seed, so any rejected or
interesting episode can be replayed exactly — even when 4 Isaac Sim workers on 2 GPUs
are collecting concurrently. Mirrors `excavator_common/scene_randomization.py`.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass

# Ranges from the final report (Table 6-2) / runtime constants.
TRUCK_RADIUS_M = (8.0, 10.8)
TRUCK_ANGLE_DEG = (-170.0, -120.0)
TRUCK_JITTER_M = 1.2
ROBOT_YAW_DEG = (-170.0, 170.0)
SAND_RADIUS_M = (5.8, 7.35)
SAND_ANGLE_DEG = (82.0, 98.0)
SAND_AMOUNT = (3.0, 15.0)
MIN_SAND_TO_UNLOAD_M = 6.0
MIN_ROBOT_TO_TRUCK_M = 3.5
MAX_TRIES = 96


def stable_scene_seed(run_identity: str, attempt_index: int, base_seed: int = 410700,
                      worker_identity: str = "") -> int:
    payload = json.dumps(
        {"schema": "excavator_scene_seed_v2", "base_seed": int(base_seed),
         "run_identity": str(run_identity), "worker_identity": str(worker_identity),
         "attempt_index": int(attempt_index)},
        ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    )
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def area_uniform_radius(u: float, radius_range: tuple[float, float]) -> float:
    """Sample a radius so points are uniform over the annulus *area*, not the radius."""
    lo, hi = min(radius_range), max(radius_range)
    u = min(1.0, max(0.0, float(u)))
    return math.sqrt(lo * lo + u * (hi * hi - lo * lo))


@dataclass
class SceneSample:
    seed: int
    robot_yaw_deg: float
    sand_xy: tuple[float, float]
    sand_amount: float
    truck_xy: tuple[float, float]
    truck_yaw_deg: float


def _polar(rng: random.Random, r_range, a_range_deg) -> tuple[float, float]:
    r = area_uniform_radius(rng.random(), r_range)
    a = math.radians(rng.uniform(*a_range_deg))
    return r * math.cos(a), r * math.sin(a)


def sample_scene(run_identity: str, attempt: int, worker: str = "") -> SceneSample:
    seed = stable_scene_seed(run_identity, attempt, worker_identity=worker)
    rng = random.Random(seed)
    for _ in range(MAX_TRIES):
        sand = _polar(rng, SAND_RADIUS_M, SAND_ANGLE_DEG)
        tx, ty = _polar(rng, TRUCK_RADIUS_M, TRUCK_ANGLE_DEG)
        truck = (tx + rng.uniform(-TRUCK_JITTER_M, TRUCK_JITTER_M), ty + rng.uniform(-TRUCK_JITTER_M, TRUCK_JITTER_M))
        if math.dist(sand, truck) < MIN_SAND_TO_UNLOAD_M or math.hypot(*truck) < MIN_ROBOT_TO_TRUCK_M:
            continue
        # keep the truck's rear (tailgate) facing the excavator within ±90°
        face_robot = math.degrees(math.atan2(-truck[1], -truck[0]))
        truck_yaw = face_robot + rng.uniform(-90.0, 90.0)
        return SceneSample(
            seed=seed,
            robot_yaw_deg=rng.uniform(*ROBOT_YAW_DEG),
            sand_xy=sand,
            sand_amount=rng.uniform(*SAND_AMOUNT),
            truck_xy=truck,
            truck_yaw_deg=truck_yaw,
        )
    raise RuntimeError(f"no valid scene after {MAX_TRIES} tries (seed={seed})")

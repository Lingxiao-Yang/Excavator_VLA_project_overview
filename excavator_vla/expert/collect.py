"""Automated data-collection loop (runs inside one Isaac Sim process).

    Randomize scene -> Select targets -> Plan motion -> Execute & record
        -> Check episode quality -> accept (export later) | reject + retry

The `sim` object stands in for the ~40k-line Isaac runtime; its methods are named after
the real entry points (`auto_collect_*`) so this reads as a map of that code.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

from .. import randomization
from ..contract import POLICY_FPS, STAGE_ID
from ..kinematics import ExcavatorGeometry
from .dig_planner import (FULL_PLAN_TOP_K, build_stage_plan, choose_unload_landing,
                          plan_cost, rank_dig_targets)
from .quality import score_scoop

log = logging.getLogger("collect")
SETTLE_FRAMES = 180


def collect_one_episode(sim, geo: ExcavatorGeometry, run_id: str, attempt: int, out_dir: Path) -> dict:
    worker = os.environ.get("EXCAVATOR_AUTO_RUN_ID_SUFFIX", "")
    scene = randomization.sample_scene(run_id, attempt, worker)

    # 1-3: scene + pose gates (cheap, before any physics)
    sim.apply_scene(scene)
    if not sim.scene_pre_sample_gate():
        return {"accepted": False, "reason": "scene_gate"}

    # 4-5: home pose -> stable sand reset -> settle; camera + action-channel preflight
    sim.move_home()
    sim.reset_sand(scene.sand_amount)
    sim.step(SETTLE_FRAMES)
    if not sim.camera_preflight():
        # camera failure stops collection instead of producing a false episode
        raise RuntimeError("dataset cameras not producing fresh frames")

    # 6: plan — rank candidates cheaply, fully plan the top few, keep the cheapest
    q0 = sim.joint_positions()
    particles = sim.particle_positions()
    candidates = rank_dig_targets(particles, sim.pile_center(), sim.pile_half(), q0)
    unload = choose_unload_landing(sim.truck_bed_center(), sim.truck_yaw(), 1.25, sim.bed_fill_height)
    best = None
    for cand in candidates[:FULL_PLAN_TOP_K]:
        plan = build_stage_plan(geo, cand, unload)
        res = plan_cost(geo, plan, q0)
        if res and (best is None or res[0] < best[0]):
            best = (res[0], plan, res[1])
    if best is None:
        return {"accepted": False, "reason": "planning_gate"}
    _, plan, joint_goals = best

    # 7-9: execute with interpolated control points; record on a 10 Hz *sim-time* grid
    ep_dir = out_dir / f"episode_{attempt:06d}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    rows, counts = [], {"max_bucket": 0, "lift_bucket": 0}
    sim.cameras_start_background()
    for st, q_goal in zip(plan, joint_goals):
        if st.stage == "unload_to_bin":
            for wp in sim.find_clearance_route(sim.joint_positions(), q_goal):   # loaded_transit
                rows += sim.move_to(wp, stage_id=STAGE_ID["loaded_transit"], sample_hz=POLICY_FPS)
        rows += sim.move_to(q_goal, stage_id=STAGE_ID[st.stage], duration=st.duration_s,
                            bucket_mode=st.bucket_mode, sample_hz=POLICY_FPS)
        n = sim.bucket_particle_count()
        counts["max_bucket"] = max(counts["max_bucket"], n)
        if st.stage == "lift_carry":
            counts["lift_bucket"] = n
    sim.dump(angle_deg=45.0, hold_s=0.9)

    # 10: quality gate
    report = score_scoop(counts["max_bucket"], counts["lift_bucket"], sim.bucket_particle_count(),
                         sim.truck_bed_gain(), execution_success=True, freeze_events=sim.freeze_events())
    with open(ep_dir / "trajectory.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = {"seed": scene.seed, "scene": scene.__dict__, "score": report.score,
            "spill_ratio": report.spill_ratio, "failures": report.failures, "accepted": report.accepted}
    (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=lambda o: np.asarray(o).tolist()))
    return meta


def auto_collect_loop(sim, success_count: int, max_attempts: int | None = None, out_root="excavator_auto_dataset"):
    geo = ExcavatorGeometry()
    run_id = sim.run_identity()
    out_dir = Path(out_root) / run_id
    max_attempts = max_attempts or success_count * 5
    ok = 0
    for attempt in range(max_attempts):
        meta = collect_one_episode(sim, geo, run_id, attempt, out_dir)
        ok += bool(meta.get("accepted"))
        log.info("attempt %d -> %s (%d/%d accepted)", attempt,
                 "OK" if meta.get("accepted") else meta.get("reason") or meta.get("failures"), ok, success_count)
        if ok >= success_count:
            break
    (out_dir / "summary.json").write_text(json.dumps({"accepted": ok, "attempts": attempt + 1}))
    return ok

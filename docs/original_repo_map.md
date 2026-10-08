# Correspondence with the development repository

The development repository (`ExcavatorVLA`, 224 commits, May–July 2026) is organized into long-lived
branches. Isaac Sim data collection and SmolVLA training require separate Python environments, so
each workflow is maintained on its own branch.

| Branch | Purpose |
| --- | --- |
| `main` | Handover index: README, handover record and release workflow |
| `isaac` | Scene construction, randomization, scripted expert, parallel collection, dashboard and LeRobot export |
| `smolvla` | Training extensions, offline evaluation, bridge server, policy client and deployment launchers |
| `video` | `smolvla` with overview-camera recording, used for the Design Expo videos |
| `mujoco` | Early URDF-to-MJCF prototype, superseded by Isaac Sim |

## Module correspondence

| Reference implementation (`excavator_vla/`) | Development source |
| --- | --- |
| `contract.py` | `excavator_common/vla_observation_contract.py`, `deployment_contract.py` |
| `protocol.py` | `excavator_common/bridge_protocol.py`, `deployment_contract.py` (tick scheduler, handshake) |
| `kinematics.py` | `excavator_runtime.py` (`two_link_ik_2d`, `solve_priority_ik_to_target`), `ik_calculation.py` |
| `randomization.py` | `excavator_common/scene_randomization.py` and runtime randomization constants |
| `expert/dig_planner.py` | `excavator_runtime.py` (`rank_dig_targets`, `dig_plan_specs_from_candidate`, unload selection) |
| `expert/quality.py` | `excavator_runtime.py` (`compute_episode_quality_score`, `multi_scoop_cycle_report`) |
| `expert/collect.py` | `excavator_runtime.py` (`auto_collect_loop`, `auto_collect_one_episode`) |
| `expert/parallel.py` | `run_parallel_auto_collect.py` |
| `data/lerobot_export.py` | `excavator_dataset_tools.py` |
| `sim/bridge_server.py` | `run_simulation.py` |
| `sim/scene.py`, `sim/sand.py` | `sand_site_runtime.py`, `excavator_dataset_camera.py` |
| `policy/client.py` | `scripts/bridge_test/smolvla_policy_client.py` |
| `policy/ensemble.py` | client replanning and `temporal_ensemble_action` |
| `policy/supervisor.py` | client `apply_excavation_sequence_supervisor` |
| `training/stage_losses.py` | `scripts/training/patch_smolvla_*.py` (six extensions to LeRobot's SmolVLA) |
| `evaluation/` | `scripts/evaluation/evaluate_smolvla_episodes.py`, `analyze_action_stage_chunk_stitch.py` |

## Simplifications

* The production runtime is a single module of about 40,000 lines, with extensive tuned constants,
  recovery paths, diagnostics and dashboard integration. The reference implementation preserves the
  algorithmic structure and production thresholds. It omits recovery branches, staged and progressive
  planning fallbacks, multi-scoop logic, and the full RRT implementation.
* `ExcavatorGeometry` is a planar model built from the URDF link lengths, whereas the production code
  uses articulation forward kinematics. Feasible dig radii in the reference model therefore differ
  from those of the production scene.
* The policy client uses LeRobot's public `predict_action_chunk` interface. The production client
  implements the flow-matching sampler directly, so that it can read the stage head at the final
  denoising step.
* Supervisor unload targets are placeholders. The production client derives them for each scene.
* Large assets (USD, FBX, meshes and Blender models), the 15 GB dataset and the trained checkpoints
  are not included.

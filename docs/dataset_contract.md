# Dataset contract

Implementation: [`excavator_vla/contract.py`](../excavator_vla/contract.py),
[`excavator_vla/data/lerobot_export.py`](../excavator_vla/data/lerobot_export.py)

Demonstrations are stored in LeRobot v3 format at **10 fps**: one Parquet table for low-dimensional
data and one MP4 stream per camera per episode. The schema identifier is
`excavator_state_v4_28d_plus_4effort_categorical_phase10`.

## Features

| Key | Shape | Type | Description |
| --- | --- | --- | --- |
| `observation.images.0` | 256 × 256 × 3 | video | arm-mounted bucket camera |
| `observation.images.1` | 256 × 256 × 3 | video | forward camera on the upper structure |
| `observation.images.2` | 256 × 256 × 3 | video | elevated overhead camera on the upper structure |
| `observation.state` | 28 | float32 | robot and task state (below) |
| `observation.effort` | 4 | float32 | measured joint effort, [swing, boom, arm, bucket] |
| `action` | 4 | float32 | commanded joint velocity for the next step, rad/s, [swing, boom, arm, bucket] |
| `observation.stage_current_id` | 1 | int64 | task stage 0–9, used as a training label only |
| `task` | — | string | natural-language instruction |

## `observation.state` (28D)

All spatial quantities are expressed in the **upper-structure frame**, whose heading is
`base_yaw + swing`. The state is therefore invariant to the excavator's parking position and
initial orientation, which is what lets one policy generalize across randomized scenes.

| Index | Names | Unit | Description |
| --- | --- | --- | --- |
| 0–3 | `swing, boom, arm, bucket` | rad | joint positions |
| 4–7 | `*_velocity` | rad/s | joint velocities (backward difference) |
| 8–11 | `*_tracking_error` | rad | commanded minus measured position (swing wrapped) |
| 12–15 | `*_prev_action` | rad/s | previously commanded velocity |
| 16–17 | `bucket_pour_axis_{forward,up}` | — | unit vector from bucket load point to pour point |
| 18–20 | `dig_delta_{forward,left,up}` | m | dig target relative to the bucket tip |
| 21–23 | `unload_delta_{forward,left,up}` | m | truck landing point relative to the bucket load point |
| 24–25 | `truck_heading_relative_{sin,cos}` | — | truck yaw relative to the upper-structure heading |
| 26 | `bucket_fill_fraction` | — | particles in the bucket / 6400, clipped to 1.5 |
| 27 | `bucket_fill_rate` | 1/s | EMA-filtered rate of change of the fill fraction (τ = 0.4 s), clipped to ±5 |

The task stage is not part of the state. It is exported separately as a supervision target for
the auxiliary stage head.

## Actions

The action is the velocity the expert *commanded* for the next transition,
`action[t] = (q_cmd[t+1] − q_cmd[t]) / Δt`, with three refinements:

* rows in which the expert deliberately holds position (for example `secure_load`) are set to
  **exactly zero**, so waiting is an explicit, learnable behaviour;
* when the command jumps by more than 6 rad/s (a replanning set-point discontinuity), the
  measured joint velocity is used instead;
* values are clipped to ±12 rad/s.

During deployment the bridge integrates the predicted velocities back into joint position targets,
so training and execution share one action convention.

## Task stages

| ID | Stage | ID | Stage |
| --- | --- | --- | --- |
| 0 | `pre_dig` | 5 | `pull_exit_cut` |
| 1 | `approach_contact` | 6 | `secure_load` |
| 2 | `insert_cut` | 7 | `lift_carry` |
| 3 | `pull_mid_cut` | 8 | `loaded_transit` |
| 4 | `curl_to_hold_material` | 9 | `unload_to_bin` |

## Contract enforcement

Earlier iterations of the dataset used other state layouts: an 18D vector, and a 28D layout with the
task phase embedded in the state and positions expressed in the world frame. Each change produced
models that loaded successfully but behaved incorrectly. The current contract is enforced in four ways:

1. **Single source of truth.** The collector, exporter, trainer, bridge server and policy client all
   import the same contract module.
2. **Fail closed.** Feature names and dimensions are validated at every boundary. Truncating or
   padding features is not supported.
3. **Handshake.** The bridge handshake carries the schema identifier, every feature name, the control
   rate and a SHA-256 digest of the normalization statistics. A policy paired with the wrong
   statistics is rejected before the first motion.
4. **Canonical stage order.** Dataset metadata with a non-canonical stage ordering is refused.

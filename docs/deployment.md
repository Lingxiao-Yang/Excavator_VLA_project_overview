# Closed-loop deployment

Implementation: [`protocol.py`](../excavator_vla/protocol.py), [`sim/bridge_server.py`](../excavator_vla/sim/bridge_server.py),
[`policy/`](../excavator_vla/policy/), [`scripts/03_deploy_demo.sh`](../scripts/03_deploy_demo.sh)

Isaac Sim ships with its own Python interpreter, while LeRobot and SmolVLA run in a separate conda
environment. The two processes communicate over **local TCP**.

## Bridge protocol (version 2)

* **Framing.** Each message is a 4-byte big-endian length followed by a UTF-8 JSON body. Images are
  transmitted as zlib-compressed, base64-encoded uint8 arrays.
* **Handshake.** The client sends the protocol version, observation schema, every state, effort and
  action name, the camera keys, the training frame rate and a normalization hash. The server validates
  all fields and replies with the physics rate, the timing mode (`deterministic_lockstep`) and the
  canonical-to-articulation joint index map, resolved by joint *name*. Any mismatch aborts the
  session before motion begins.
* **Step request.** The client sends only `joint_velocities[4]` in rad/s. Fields from the earlier
  tick-based protocol are rejected.
* **Observation.** The server returns `observation_state[28]`, `observation_effort[4]`, the three
  camera images, bucket and truck particle counts, the instruction text and a per-component timing
  breakdown.

## Deterministic lockstep

```text
client: send v_t ─────────────► server: resume; 6 × (q_target += v_t·Δt; clip; step); pause
client: ◄─────────── obs_{t+1}          render 3 cameras → 256² → encode
client: inference (simulation paused)
```

The simulation timeline is paused between requests, so neither inference time nor rendering time
accrues as simulated time. Every action lasts exactly 0.1 s of physics. When the control rate does
not divide 60 Hz evenly, the tick scheduler carries the fractional remainder forward to avoid drift.

## Inference loop

At each control step, the client:

1. Sends the current command and receives the next observation. The first command is all zeros.
2. Evaluates the success detector, with optional early termination.
3. Decodes the camera images and normalizes state and effort. It rejects non-finite values and clamps
   inputs to ±8 σ, logging any clipping as out-of-distribution.
4. **Replans** every three steps, or whenever the current chunk is exhausted. Replanning predicts a
   new 50 × 4 action chunk with stage logits and discards the unexecuted remainder.
5. Selects the action for the current step. Temporal ensembling over the newest *w* chunks, with
   weights `exp(−0.35 · age)`, is supported; the demonstration uses *w* = 1.
6. Optionally applies the execution supervisor (below).
7. Logs the predicted action, the executed action, any supervisor interventions and the timing.

The client applies no additional gain, scaling or clipping to predicted actions. The only limit is
the server's joint-limit clip. Measured latency is **2.37 ms** (median forward pass) and **6.47 ms**
(median end-to-end client processing).

## Execution supervisor

```text
dig → curl_secure → lift_loaded → transfer_loaded → position_over_truck → dump → complete
```

The supervisor is monotonic (its phase never regresses), deterministic, bounded and fully logged. It
reads only target-relative quantities from the 28D state.

* **dig:** latches a minimum boom-down velocity once the bucket tip approaches the dig target or the
  policy predicts a cutting stage, and halves the bucket command while the tip is still well above
  the target.
* **curl_secure:** holds swing and boom and enforces bucket curl until the bucket is closed and loaded.
* **lift_loaded, transfer_loaded, position_over_truck:** bounded proportional control toward the
  unload configuration. Velocity limits are 0.18 rad/s for the boom, 0.12 rad/s for the arm and
  0.45 rad/s for swing.
* **dump:** enforces bucket opening past the dump angle.
* **complete:** commands zero velocity.

The supervisor was introduced for the final Design Expo trials to ensure a reliable demonstration
and is disabled by default. Results obtained with it enabled characterise the deployed system; see
[results.md](results.md).

## Recording

* `camera_0/1/2.mp4` contain the exact 256² policy inputs, one frame per control step (10 fps).
* `camera_overview.mp4` is a fixed 1024² viewpoint for human viewing only.
* `recording_metadata.json` records frame counts and the policy-step range. Warm-up frames are excluded.

The videos in [`media/`](../media/) are from run
`isaacsim_seed2_ckpt18450_replan3_ensemble1_supervisor1_300steps_20260727_224304`, re-encoded to
H.264 for browser playback.

## Start-up procedure

Select artifacts → preflight checks → validate contract → initialize scene → connect client →
execute → record → archive.

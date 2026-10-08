#!/usr/bin/env bash
# Stage 3 — closed-loop demo: Isaac Sim bridge server + SmolVLA client.
# Two Python environments on purpose: Isaac Sim's bundled Python vs. the smolvla conda env.
set -euo pipefail

CKPT=${CKPT:?.../checkpoints/018450/pretrained_model}
VLM=${VLM:-checkpoints/SmolVLM2-500M-Video-Instruct}
LOG=${LOG:-runs/demo_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$LOG"

# 1) simulator (needs a DISPLAY: persistent viewports are used for camera capture)
setsid /isaac-sim/python.sh -m excavator_vla.sim.bridge_server \
  --scene-seed 2 --sand-amount 10 > "$LOG/sim.log" 2>&1 &
SIM_PID=$!
trap 'kill -- -$SIM_PID 2>/dev/null || true' EXIT

until grep -q "\[bridge\] TCP server listening" "$LOG/sim.log"; do sleep 1; done

# 2) policy client
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/opt/conda/envs/smolvla/bin/python -m excavator_vla.policy.client \
  --ckpt "$CKPT" --vlm "$VLM" \
  --replan-interval 3 --temporal-ensemble-width 1 \
  --enable-supervisor --record-overview \
  --steps 300 --trace-log "$LOG/trace.jsonl"

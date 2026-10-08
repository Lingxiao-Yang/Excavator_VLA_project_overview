#!/usr/bin/env bash
# Stage 1 — automated expert data collection in Isaac Sim (isaac environment).
# Each worker is an independent headless Isaac Sim process with its own seed suffix.
set -euo pipefail

WORKERS=${WORKERS:-4}
SUCCESS=${SUCCESS:-600}
GPUS=${GPUS:-0,1}

python -m excavator_vla.expert.parallel \
  --workers "$WORKERS" --success-count "$SUCCESS" --max-attempts $((SUCCESS * 3)) \
  --active-gpus "$GPUS" --stagger-seconds 5

# Review episodes in the dashboard, move accepted ones into .dashboard_success/,
# then export once to LeRobot v3 (10 fps, 3 x 256^2 video streams):
#   python excavator_dataset_tools.py excavator_auto_dataset/.dashboard_success \
#     --export-lerobot-v3 --export-dir lerobot_v3 --export-require-vla \
#     --export-quality-policy gold-v1

#!/usr/bin/env bash
# Stage 2 — fine-tune SmolVLA with LeRobot (smolvla conda env, offline).
set -euo pipefail

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Isaac Sim leaks its own Python paths; make sure none of them reach LeRobot.
unset PYTHONPATH ISAAC_PATH EXP_PATH CARB_APP_PATH LD_PRELOAD

DATASET=${DATASET:?path to LeRobot v3 dataset}
BASE=${BASE:-checkpoints/smolvla_base}
VLM=${VLM:-checkpoints/SmolVLM2-500M-Video-Instruct}
OUT=${OUT:-outputs/train/excavator_smolvla}

# (original repo) apply the stage-head / direction-prior / low-motion patches to LeRobot first:
#   for p in stage_training stage_seq50 dynamic_stage_seq50 stage_batch_and_logging \
#            stage_action_direction_loss stage_action_low_motion_loss; do
#     python scripts/training/patch_smolvla_${p}.py; done

lerobot-train \
  --policy.type=smolvla \
  --policy.pretrained_path="$BASE" \
  --policy.vlm_model_name="$VLM" \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/excavator_smolvla \
  --dataset.root="$DATASET" \
  --policy.input_features='{"observation.state":{"type":"STATE","shape":[28]},"observation.effort":{"type":"STATE","shape":[4]},"observation.images.0":{"type":"VISUAL","shape":[3,256,256]},"observation.images.1":{"type":"VISUAL","shape":[3,256,256]},"observation.images.2":{"type":"VISUAL","shape":[3,256,256]}}' \
  --policy.output_features='{"action":{"type":"ACTION","shape":[4]}}' \
  --policy.optimizer_lr=5e-5 \
  --batch_size=144 \
  --steps=16650 \
  --save_freq=300 \
  --output_dir="$OUT" \
  --wandb.enable=false

# Fixed-scene refinement: copy the latest checkpoint, edit train_config.json
# (steps=18450, dataset=5 seed-2 episodes), then:
#   lerobot-train --config_path=$OUT_REFINE/train_config.json --resume=true

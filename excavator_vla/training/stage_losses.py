"""Auxiliary losses added on top of SmolVLA's flow-matching objective.

Vanilla SmolVLA fine-tuning on excavator data learned the *average* motion well but mixed
up phases — e.g. swinging toward the truck while still cutting. Three additions fixed most
of that. In the original repo they are applied as exact-text patches to the installed
LeRobot package (`scripts/training/patch_smolvla_*.py`, applied in a fixed order); here
they are written as a standalone module for readability.

  L = L_flow
    + w_stage      * CE(stage_head(suffix_tokens), stage_sequence_50)     # 0.5, x10 on rare stages
    + w_direction  * relu(-prior_direction[stage] * a_pred) * prior_weight # 0.20
    + w_low_motion * relu(|a_swing| * swing_scale - margin)  in dig/lift   # 0.5, margin 0.10

The vision encoder stays frozen; the action expert, state projection and stage head train.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..contract import STAGE_ID, STAGE_NAMES

N_STAGES = len(STAGE_NAMES)
SWING = 0
# Stages where the upper structure should NOT rotate
NO_SWING_STAGES = [STAGE_ID[s] for s in ("approach_contact", "insert_cut", "pull_mid_cut",
                                         "curl_to_hold_material", "pull_exit_cut", "secure_load", "lift_carry")]


class StageHead(nn.Module):
    """Per-action-token stage classifier on the action-expert (suffix) output."""

    def __init__(self, hidden: int, n_stages: int = N_STAGES):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, n_stages))

    def forward(self, suffix_out: torch.Tensor) -> torch.Tensor:   # [B, 50, H] -> [B, 50, S]
        return self.proj(suffix_out)


def stage_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
               exception_weight: float = 10.0, rare_stages: tuple[int, ...] = (6, 8)) -> torch.Tensor:
    """Cross-entropy over the 50-step stage sequence; rare transition stages are up-weighted.

    `stage_exception_weight=10` is the production value; which stages count as "rare" here
    is illustrative (short transitional stages such as secure_load / loaded_transit).
    """
    w = torch.ones(N_STAGES, device=logits.device)
    w[list(rare_stages)] = exception_weight
    ce = F.cross_entropy(logits.flatten(0, 1), target.flatten(), weight=w, reduction="none")
    m = valid.flatten().float()
    return (ce * m).sum() / m.sum().clamp_min(1.0)


def stage_action_direction_loss(action: torch.Tensor, stage: torch.Tensor,
                                prior_direction: torch.Tensor, prior_weight: torch.Tensor,
                                min_prior_weight: float = 0.8) -> torch.Tensor:
    """Penalise predicted velocities whose *sign* contradicts the expert prior for that stage.

    prior_direction[s, j] in {-1, 0, +1}: e.g. boom must go down in insert_cut,
    bucket must curl (negative) in curl_to_hold_material, swing toward truck in loaded_transit.
    Priors are mined from the expert dataset; only joints with prior_weight >= 0.8 are used.
    """
    d = prior_direction[stage]                     # [B, T, 4]
    w = prior_weight[stage] * (prior_weight[stage] >= min_prior_weight)
    return (F.relu(-d * action) * w).mean()


def low_motion_loss(action: torch.Tensor, stage: torch.Tensor,
                    margin: float = 0.10, swing_scale: float = 2.178) -> torch.Tensor:
    """Keep swing near zero while digging / lifting (swing in normalised units is ~2x larger)."""
    mask = torch.zeros_like(stage, dtype=torch.bool)
    for s in NO_SWING_STAGES:
        mask |= stage == s
    excess = F.relu(action[..., SWING].abs() * swing_scale - margin)
    return (excess * mask).sum() / mask.sum().clamp_min(1)


def total_loss(flow_loss, stage_logits, batch, action_pred, priors,
               w_stage=0.5, w_dir=0.20, w_low=0.5) -> dict[str, torch.Tensor]:
    st = batch["observation.stage_sequence_50"]
    parts = {
        "flow": flow_loss,
        "stage": stage_loss(stage_logits, st, batch["observation.stage_valid_50"]),
        "direction": stage_action_direction_loss(action_pred, st, priors["direction"], priors["weight"]),
        "low_motion": low_motion_loss(action_pred, st),
    }
    parts["total"] = parts["flow"] + w_stage * parts["stage"] + w_dir * parts["direction"] + w_low * parts["low_motion"]
    return parts

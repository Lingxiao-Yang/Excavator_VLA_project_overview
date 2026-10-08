"""Episode quality gates.

Three filters keep bad demonstrations out of the dataset:
  scene gate     — before simulation: reachable pile, no truck overlap, enough particles
  planning gate  — a full IK-feasible, collision-free plan exists
  quality gate   — after execution: the bucket actually carried sand into the truck

Only the third is shown here (from `multi_scoop_cycle_report` /
`compute_episode_quality_score`). Thresholds are the production values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

TARGET_BUCKET_PARTICLES = 25
MIN_BUCKET_PARTICLES = 5
MIN_DUMP_PARTICLES = 3
MAX_SPILL_RATIO = 0.30
MIN_SCORE = 70.0          # also the "gold-v1" export threshold


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass
class ScoopReport:
    score: float
    spill_ratio: float
    failures: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return not self.failures


def score_scoop(max_bucket: int, lift_bucket: int, final_bucket: int, bin_gain: int,
                execution_success: bool, freeze_events: int) -> ScoopReport:
    """Score one dig-carry-dump cycle from particle counts sampled at key moments.

    max_bucket   peak particles inside the bucket volume during the cut
    lift_bucket  particles still in the bucket after lift (retention)
    final_bucket particles left in the bucket after dumping
    bin_gain     new particles inside the truck-bed volume
    """
    spill = max(0, lift_bucket - final_bucket - bin_gain)
    spill_ratio = spill / max(1.0, float(lift_bucket))

    score = (
        35.0 * _clamp01(max_bucket / TARGET_BUCKET_PARTICLES)            # dig load
        + 30.0 * _clamp01(bin_gain / max(1.0, float(max(lift_bucket, max_bucket))))  # transfer
        + 15.0 * _clamp01(lift_bucket / max(1.0, float(max_bucket)))     # retention
        + 10.0 * 0.5                                                     # smoothness (neutral)
        + (10.0 if execution_success else 0.0)                           # completion
        - min(20.0, 20.0 * spill_ratio)
    )

    failures = []
    if max_bucket < MIN_BUCKET_PARTICLES:
        failures.append(f"low_bucket_particles:{max_bucket}")
    if lift_bucket < MIN_BUCKET_PARTICLES:
        failures.append(f"low_lift_particles:{lift_bucket}")
    if bin_gain < MIN_DUMP_PARTICLES:
        failures.append(f"low_bin_gain:{bin_gain}")
    if spill_ratio > MAX_SPILL_RATIO:
        failures.append(f"high_spill_ratio:{spill_ratio:.3f}")
    if score < MIN_SCORE:
        failures.append(f"score_below_minimum:{score:.1f}")
    if freeze_events > 0:
        failures.append(f"freeze_detected:{freeze_events}")
    return ScoopReport(score=score, spill_ratio=spill_ratio, failures=failures)


def detect_freeze(cmd_err_deg: float, motion_deg: float) -> bool:
    """A joint is 'frozen' when it is far from its command yet not moving (stuck in sand)."""
    return cmd_err_deg > 2.0 and motion_deg < 0.12

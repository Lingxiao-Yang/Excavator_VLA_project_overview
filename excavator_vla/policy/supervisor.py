"""Excavation-sequence supervisor: a bounded execution safeguard.

Reporting note: the 83.5 % (167/200) closed-loop success rate presented at the Design Expo
was measured with this supervisor enabled. The supervisor is deterministic, monotonic (its
phase never regresses) and logs every step it modifies, but its presence means the figure
characterises the deployed VLA *system* rather than the policy in isolation. It is disabled
by default.

Division of labour: the VLA selects where and how to dig; the supervisor constrains only the
phases in which the raw policy was observed to hesitate, namely committing to the downward
cut, curling before lifting, and completing the swing-and-dump once a load is confirmed.

    dig -> curl_secure -> lift_loaded -> transfer_loaded -> position_over_truck -> dump -> complete

All inputs are taken from the 28D state, so every trigger is expressed relative to the dig
target, the unload landing point, or the bucket fill fraction.

Condensed from `apply_excavation_sequence_supervisor` in `smolvla_policy_client.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PHASES = ("dig", "curl_secure", "lift_loaded", "transfer_loaded", "position_over_truck", "dump", "complete")
SWING, BOOM, ARM, BUCKET = range(4)


def bounded_p(q: float, target: float, gain: float, max_abs: float) -> float:
    """Proportional velocity command toward `target`, saturated at `max_abs` rad/s."""
    return float(np.clip(gain * (target - q), -max_abs, max_abs))


@dataclass
class SupervisorConfig:
    descent_trigger_height: float = 2.1     # m above the dig target — start the forced descent
    descent_velocity: float = 0.20          # rad/s, minimum boom-down command
    bucket_half_scale_dist: float = 0.85    # m — halve bucket action while still this far above the target
    curl_handover_dist: float = 0.75        # m — hand over to curl once the tip is this close
    curl_velocity: float = 0.85             # rad/s
    curl_done_bucket: float = -2.04         # rad — bucket considered closed
    load_trigger_fill: float = 0.08         # bucket fill fraction treated as "loaded" (~500 particles)
    lift_max_steps: int = 30
    # The unload_* / dump_* values are placeholders. The production client derives them for each
    # scene from the deployment scene contract (unload landing point -> IK) at start-up.
    unload_swing: float = 0.0
    unload_boom: float = 0.55
    unload_arm: float = -0.95
    unload_bucket_hold: float = -1.9
    dump_velocity: float = 1.2
    dump_target_bucket: float = 0.3


@dataclass
class ExcavationSupervisor:
    cfg: SupervisorConfig = field(default_factory=SupervisorConfig)
    phase_rank: int = 0
    steps_in_phase: int = 0
    descent_latched: bool = False

    @property
    def phase(self) -> str:
        return PHASES[self.phase_rank]

    def _advance(self, new_phase: str) -> None:
        r = PHASES.index(new_phase)
        if r > self.phase_rank:                 # monotonic: never regress
            self.phase_rank, self.steps_in_phase = r, 0

    def __call__(self, action: np.ndarray, obs: dict, predicted_stage: int) -> tuple[np.ndarray, list[str]]:
        """Return the (possibly modified) action and the reasons for any modification.

        obs keys: q [4], tip_above_dig_target (m), load_above_unload (m), fill_fraction.
        """
        a = np.array(action, dtype=np.float32)
        c, reasons = self.cfg, []
        q = obs["q"]
        height = obs["tip_above_dig_target"]
        self.steps_in_phase += 1

        if self.phase == "dig":
            if height <= c.descent_trigger_height or 2 <= predicted_stage <= 5:
                self.descent_latched = True     # one-shot latch
            if self.descent_latched:
                a[BOOM] = min(a[BOOM], -c.descent_velocity)
                reasons.append("forced_descent")
                if height > c.bucket_half_scale_dist:
                    a[BUCKET] *= 0.5
            if height <= c.curl_handover_dist:
                self._advance("curl_secure")

        elif self.phase == "curl_secure":
            a[SWING] = a[BOOM] = 0.0
            a[BUCKET] = min(a[BUCKET], -c.curl_velocity)
            reasons.append("curl_before_lift")
            if q[BUCKET] <= c.curl_done_bucket and obs["fill_fraction"] >= c.load_trigger_fill:
                self._advance("lift_loaded")

        elif self.phase == "lift_loaded":
            a[SWING] = 0.0
            a[BOOM] = bounded_p(q[BOOM], c.unload_boom, 0.8, 0.18)
            a[ARM] = bounded_p(q[ARM], c.unload_arm, 0.8, 0.12)
            a[BUCKET] = bounded_p(q[BUCKET], c.unload_bucket_hold, 1.5, 0.8)
            if abs(q[BOOM] - c.unload_boom) < 0.05 or self.steps_in_phase >= c.lift_max_steps:
                self._advance("transfer_loaded")

        elif self.phase in ("transfer_loaded", "position_over_truck"):
            a[SWING] = bounded_p(q[SWING], c.unload_swing, 1.0, 0.45)
            if obs["load_above_unload"] < 0.0:
                a[BOOM] = 0.12                  # keep raising until the load clears the landing height
            if abs(q[SWING] - c.unload_swing) < 0.05:
                self._advance("dump")

        elif self.phase == "dump":
            a[BUCKET] = max(a[BUCKET], c.dump_velocity)
            if q[BUCKET] >= c.dump_target_bucket:
                self._advance("complete")

        else:  # complete
            a[:] = 0.0

        return a, reasons

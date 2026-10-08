"""Excavator VLA — demo / showcase package.

This package is a condensed, readable re-implementation of the core ideas in the
ME 450 Group 19 ExcavatorVLA codebase. It is NOT the production code: the real
runtime is a ~40k-line Isaac Sim module plus a 4k-line policy client spread over
two long-lived branches (`isaac`, `smolvla`). Everything here is written to be
read first and run second.

Layout
------
contract          Shared observation / action / stage contract (single source of truth)
protocol          Protocol-v2 TCP framing, handshake, drift-free tick scheduler
kinematics        Planar 2-link IK + bucket-angle search, floor-safe depth
randomization     Reproducible per-worker scene seeding and sampling
expert/           Scripted expert: dig-target ranking, stage plan, quality gates, collection loop
data/             LeRobot v3 export (feature schema + action derivation)
sim/              Isaac Sim bridge server (sketch — needs Isaac Sim to run)
policy/           SmolVLA policy client: replanning, temporal ensemble, supervisor, success check
training/         Extra SmolVLA losses (stage head, stage-action direction prior, low-motion)
evaluation/       Offline metrics and open-loop chunk stitching

Pure-Python modules (contract, protocol, kinematics, randomization, policy.ensemble,
policy.supervisor, evaluation) only need numpy and are covered by `tests/`.
"""

__version__ = "0.1.0-demo"

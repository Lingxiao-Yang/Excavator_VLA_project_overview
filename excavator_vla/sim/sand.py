"""PhysX particle "sand" presets.

Sand is a PhysX particle system (not a height map) so the bucket can genuinely scoop,
carry, spill and dump material — "material interaction level 3" in the spec. The cost
is performance: particle count is the main knob trading realism for loop rate.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ParticleMaterial:
    # "Soft Dig" preset used for all final data and evaluation
    viscosity: float = 0.85
    cohesion: float = 0.035
    adhesion: float = 0.015
    damping: float = 0.45
    static_friction: float = 0.85
    dynamic_friction: float = 0.82
    particle_friction_scale: float = 0.90
    restitution: float = 0.0
    gravity_scale: float = 1.0
    max_velocity_m_s: float = 12.0
    bulk_density_ref_kg_m3: float = 1600.0


@dataclass(frozen=True)
class ParticleResolution:
    n_particles: int
    spacing_m: float
    radius_m: float
    solver_iterations: int


EFFICIENCY = ParticleResolution(45_000, 0.085, 0.032, 16)    # selected baseline
REALISTIC = ParticleResolution(260_000, 0.034, 0.014, 24)

PILE_INNER_SIZE_M = 2.50      # 2.5 x 2.5 m pile region and truck bed
WALL_THICKNESS_M = 0.05
WALL_HEIGHT_M = 0.60


def target_particle_count(area_scale: float, realism: float) -> int:
    """N = A * [(1 - f) * 45k + f * 260k],  f in [0, 1] blends efficiency -> realistic."""
    f = max(0.0, min(1.0, realism))
    return int(area_scale * ((1 - f) * EFFICIENCY.n_particles + f * REALISTIC.n_particles))

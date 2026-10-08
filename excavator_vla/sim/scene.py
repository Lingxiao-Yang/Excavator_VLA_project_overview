"""Scene construction (sketch). Requires Isaac Sim; omni/isaacsim imports happen lazily.

/World
 ├── construction_site     (FBX environment + HDR sky)
 ├── excavator             (URDF_real3 -> USD articulation, fixed base, 4 DOF)
 │    ├── swing_link/Camera_1   "main"      t=(0,-1.2,1.4)  r=(65,0,0)
 │    ├── swing_link/Camera_2   "overhead"  t=(0,0,3.0)     r=(0,-70,0)
 │    └── arm_link/Camera_0     "bucket"    t=(0,0,0.45)    r=(0,-60,0)
 ├── truck                 (assets/fbx/truck/truck.usd, bed volume used for success counting)
 ├── sand_site             (2.5 x 2.5 m pile region, PhysX particle system)
 └── OverviewCamera        (1024^2 recording-only view, never a model input)

Camera clock rule: world.step(render=True) -> camera tick -> get_rgb(). Reading a camera
without the explicit tick returns the *previous* frame; this caused weeks of stale /
duplicated / black frames in early datasets.
"""

from __future__ import annotations

from dataclasses import dataclass

from .sand import EFFICIENCY, ParticleMaterial


@dataclass(frozen=True)
class CameraSpec:
    name: str
    parent: str
    translate: tuple[float, float, float]
    rotate_deg: tuple[float, float, float]
    render_res: int = 1024
    output_res: int = 256
    focal_length: float = 18.0


CAMERAS = (
    CameraSpec("0", "arm_link", (0.0, 0.0, 0.45), (0.0, -60.0, 0.0)),
    CameraSpec("1", "swing_link", (0.0, -1.2, 1.4), (65.0, 0.0, 0.0)),
    CameraSpec("2", "swing_link", (0.0, 0.0, 3.0), (0.0, -70.0, 0.0)),
)

INITIAL_POSES_DEG = {           # swing, boom, arm, bucket
    "safe_front": (0, 42, -58, -18),
    "safe_left_45": (45, 46, -62, -20),
    "safe_right_45": (-45, 46, -62, -20),
    "safe_high_front": (0, 56, -70, -24),
    "deploy_seed2": (40, 46, -62, -20),
}


def build_scene(seed: int, sand_amount: float):
    """Return (world, robot, cameras, scene) — see the original `run_simulation.py`."""
    from isaacsim.core.api import World                     # noqa: F401  (Isaac Sim 4.x)
    from isaacsim.core.utils.stage import open_stage        # noqa: F401

    # 1. open assets/usd/excavator_scene.usd; set physics dt = 1/60, GPU dynamics on
    # 2. place truck + pile from the fixed scene profile (configs/deployment_scene_seed2.json)
    # 3. lift the excavator out of the way, spawn particles (EFFICIENCY, ParticleMaterial())
    #    scaled by sand_amount, let them settle, then lower the excavator back
    # 4. create the 3 dataset cameras + overview camera, warm them up (discard first frames)
    # 5. wrap the articulation, apply drive gains, move to INITIAL_POSES_DEG["deploy_seed2"]
    raise NotImplementedError("demo sketch — see docs/simulation.md for the full procedure")


_ = (EFFICIENCY, ParticleMaterial)

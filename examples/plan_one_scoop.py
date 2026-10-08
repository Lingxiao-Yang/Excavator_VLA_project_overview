"""Runnable without Isaac Sim: plan one scoop on a synthetic sand pile and plot it.

    python examples/plan_one_scoop.py            # writes media/generated/plan_one_scoop.png

The pile is a random Gaussian mound of points standing in for PhysX particles. The script
runs the same target ranking -> 9-stage plan -> IK pipeline the expert uses, then draws the
excavator configuration at every stage in the swing plane.

Note: `ExcavatorGeometry` is a simplified planar model (URDF link lengths, joint zeros treated
as collinear). Its feasible dig radius (~8 m+) is not the production scene's 5.6-7.45 m
reach window, so the synthetic pile is placed at r = 8.2 m.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from excavator_vla.contract import STAGE_NAMES  # noqa: E402
from excavator_vla.expert.dig_planner import (FULL_PLAN_TOP_K, build_stage_plan, plan_cost,  # noqa: E402
                                              rank_dig_targets)
from excavator_vla.kinematics import ExcavatorGeometry, forward_planar  # noqa: E402


def synthetic_pile(center=(0.0, 8.2), n=20000, height=0.6, sigma=0.55, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.normal(center, sigma, size=(n, 2))
    r2 = np.sum((xy - center) ** 2, axis=1)
    z = height * np.exp(-r2 / (2 * sigma ** 2)) * rng.random(n)
    return np.column_stack([xy, z])


def main():
    geo = ExcavatorGeometry()
    q0 = np.radians([90.0, 42.0, -58.0, -18.0])           # "safe_front" rotated to face the pile
    pile = synthetic_pile()
    cands = rank_dig_targets(pile, np.array([0.0, 8.2]), 1.25, q0)
    print(f"{len(cands)} dig candidates; best score {cands[0].score:.1f} at {np.round(cands[0].xyz, 2)}")

    unload = np.array([-5.8, 4.9, 2.2])                      # pretend truck-bed landing point
    best = None
    for c in cands[:FULL_PLAN_TOP_K]:
        res = plan_cost(geo, build_stage_plan(geo, c, unload), q0)
        print(f"  candidate {np.round(c.xyz, 2)} -> {'cost %.2f' % res[0] if res else 'infeasible'}")
        if res and (best is None or res[0] < best[0]):
            best = (res[0], c, res[1])
    if best is None:
        print("no feasible plan — the real expert would fall back to staged / progressive planning")
        return

    _, cand, qs = best
    for name, q in zip([s for s in STAGE_NAMES if s != "loaded_transit"], qs):
        print(f"  {name:<22} q(deg) = {np.round(np.degrees(q), 1)}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    r_pile = np.hypot(pile[:, 0], pile[:, 1])
    ax.scatter(r_pile[::10], pile[::10, 2], s=1, c="#c8a165", label="sand (synthetic)")
    cmap = plt.get_cmap("viridis")
    for i, q in enumerate(qs[:-1]):                          # dig stages live in the pile's swing plane
        a1 = q[1]; a2 = a1 + q[2]; a3 = a2 + q[3]
        pts = forward_planar(geo, a1, a2, a3)
        chain = np.array([pts["boom_pivot"], pts["arm_joint"], pts["bucket_joint"], pts["tip"]])
        ax.plot(chain[:, 0], chain[:, 1], "-o", ms=3, color=cmap(i / 8), label=STAGE_NAMES[i])
    ax.set_xlabel("radial distance from swing axis [m]"); ax.set_ylabel("z [m]")
    ax.set_aspect("equal"); ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_title(f"Scripted expert: one scoop at r={math.hypot(*cand.xyz[:2]):.2f} m")
    out = Path(__file__).resolve().parents[1] / "media" / "generated" / "plan_one_scoop.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

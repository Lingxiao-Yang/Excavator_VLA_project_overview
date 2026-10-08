"""Parallel collection launcher: one Isaac Sim process per worker.

Kit / RTX / PhysX state is process-global, so threads are useless — each worker is a
separate `python.sh run_vla_train_scene.py --headless --auto-collect` with its own
seed suffix, GPU and log file. Successful episodes are later pooled (via the dashboard)
into an immutable `.dashboard_success/` folder and exported to LeRobot once.

Usage (illustrative):
    python -m excavator_vla.expert.parallel --workers 4 --success-count 600 --active-gpus 0,1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ISAAC_PYTHON = os.environ.get("ISAAC_PYTHON", "/isaac-sim/python.sh")


def split_total(total: int, n: int) -> list[int]:
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--success-count", type=int, default=100)
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--active-gpus", default="0")
    p.add_argument("--stagger-seconds", type=float, default=5.0)
    p.add_argument("--log-dir", default="parallel_logs")
    a = p.parse_args(argv)

    gpus = [g.strip() for g in a.active_gpus.split(",") if g.strip()]
    succ = split_total(a.success_count, a.workers)
    atts = split_total(a.max_attempts or a.success_count * 3, a.workers)
    Path(a.log_dir).mkdir(exist_ok=True)

    procs = []
    for i in range(a.workers):
        env = dict(os.environ,
                   EXCAVATOR_AUTO_RUN_ID_SUFFIX=f"w{i:02d}",       # -> seed + run_*_wNN dir
                   EXCAVATOR_PARALLEL_WORKER_INDEX=str(i),
                   EXCAVATOR_PARALLEL_WORKER_COUNT=str(a.workers),
                   CUDA_VISIBLE_DEVICES=gpus[i % len(gpus)])
        cmd = [ISAAC_PYTHON, "run_vla_train_scene.py", "--headless", "--auto-collect",
               "--success-count", str(succ[i]), "--max-attempts", str(atts[i]),
               "--random-sand-amount", "--disable-export"]
        log = open(Path(a.log_dir) / f"worker_{i:02d}.log", "w")
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
        time.sleep(a.stagger_seconds)   # avoid every worker compiling shaders at once

    (Path(a.log_dir) / "manifest.json").write_text(json.dumps(
        {"workers": a.workers, "success_per_worker": succ, "gpus": gpus,
         "pids": [pr.pid for pr in procs]}, indent=2))
    codes = [pr.wait() for pr in procs]
    sys.exit(max(codes))


if __name__ == "__main__":
    main()

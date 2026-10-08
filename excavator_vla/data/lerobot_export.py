"""Export accepted expert episodes to a LeRobot v3 dataset.

Raw episodes are `trajectory.jsonl` rows sampled on a 10 Hz sim-time grid, each with
q, q_cmd, privileged scene info and three 256x256 camera frames. The exporter:

  * builds the 28D policy state from privileged scene information,
  * derives the action as the *next* command velocity  (q_cmd[t+1] - q_cmd[t]) / dt,
  * forces hold rows to exactly 0 (so "wait for sand to settle" is learnable),
  * falls back to measured-q velocity when the command jumps (re-plan discontinuity),
  * writes stage labels as supervision-only columns.

Condensed from `excavator_dataset_tools.py` (`_hold_aware_command_actions`,
`build_lerobot_state_28d`, the LeRobot v3 feature dictionary). Uses the public
`lerobot.datasets.lerobot_dataset.LeRobotDataset` API.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import contract as C

FPS = int(C.POLICY_FPS)
COMMAND_DISCONTINUITY_RAD_S = 6.0
ACTION_HARD_MAX_RAD_S = 12.0


def lerobot_features(state_names=C.STATE_NAMES, include_effort: bool = True) -> dict:
    feats = {
        "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": list(state_names)},
        "action": {"dtype": "float32", "shape": (4,), "names": list(C.ACTION_NAMES)},
        "observation.stage_current_id": {"dtype": "int64", "shape": (1,), "names": None},
    }
    if include_effort:
        feats["observation.effort"] = {"dtype": "float32", "shape": (4,), "names": list(C.EFFORT_NAMES)}
    for key in C.CAMERA_KEYS:
        feats[key] = {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]}
    return feats


def hold_aware_actions(q_cmd: np.ndarray, q_real: np.ndarray, t: np.ndarray, is_hold: np.ndarray) -> np.ndarray:
    """Next-command velocity per row, with hold / discontinuity handling."""
    T = len(q_cmd)
    act = np.zeros((T, 4), dtype=np.float32)
    for i in range(T - 1):
        dt = float(t[i + 1] - t[i])
        if dt <= 1e-9 or is_hold[i + 1]:
            continue                                    # hold rows stay exactly zero
        v = (q_cmd[i + 1] - q_cmd[i]) / dt
        if np.max(np.abs(v)) > COMMAND_DISCONTINUITY_RAD_S:
            v = (q_real[i + 1] - q_real[i]) / dt        # setpoint jump -> use what really happened
        act[i] = np.clip(v, -ACTION_HARD_MAX_RAD_S, ACTION_HARD_MAX_RAD_S)
    return act


def export_episodes(episode_dirs: list[Path], out_root: Path, repo_id: str, task_text: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # heavy import, keep local

    ds = LeRobotDataset.create(repo_id=repo_id, root=out_root, fps=FPS,
                               features=lerobot_features(), use_videos=True)
    for ep_dir in episode_dirs:
        rows = [json.loads(l) for l in open(ep_dir / "trajectory.jsonl")]
        t = np.array([r["t"] for r in rows])
        if not np.all(np.diff(t) > 0):
            raise ValueError(f"{ep_dir}: timestamps must be strictly monotonic")
        actions = hold_aware_actions(np.array([r["q_cmd"] for r in rows]), np.array([r["q"] for r in rows]),
                                     t, np.array([r.get("semantic") == "hold" for r in rows]))
        for r, a in zip(rows, actions):
            frame = {
                "observation.state": np.asarray(r["state_28d"], dtype=np.float32),
                "observation.effort": np.asarray(r["effort"], dtype=np.float32),
                "observation.stage_current_id": np.array([C.STAGE_ID[r["stage"]]], dtype=np.int64),
                "action": a,
            }
            for cam, key in zip("012", C.CAMERA_KEYS):
                frame[key] = _load_rgb(ep_dir / r["images"][cam])   # fresh, unique, non-black (pre-validated)
            ds.add_frame(frame, task=task_text)
        ds.save_episode()
    return ds


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


TASK_TEXT = (
    "Excavate one scoop of sand from the sand pile in front of the excavator's initial base pose, "
    "then carry and dump the collected material into the truck bed to the right of the excavator's "
    "initial base pose."
)

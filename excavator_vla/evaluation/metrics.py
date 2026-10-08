"""Offline per-frame metrics on held-in episodes (condensed `evaluate_smolvla_episodes.py`).

For every frame t the model predicts a 50-step chunk; we compare it with the expert's next
50 actions. Sign accuracy matters more than MAE for an excavator — a slightly slow boom is
fine, a boom moving the wrong way is a failed scoop.
"""

from __future__ import annotations

import numpy as np


def action_metrics(pred: np.ndarray, true: np.ndarray, valid: np.ndarray,
                   tol: np.ndarray, deadband: float = 1e-3) -> dict[str, float]:
    """pred/true: [T, H, 4]; valid: [T, H]; tol: [4] per-joint tolerance in rad/s."""
    m = valid.astype(bool)
    p, t = pred[m], true[m]                               # [N, 4]
    moving = np.abs(t) > deadband
    out = {
        "action_chunk_within_tolerance_accuracy": float(np.mean(np.all(np.abs(p - t) <= tol, axis=-1))),
        "action_chunk_sign_accuracy": float(np.mean((np.sign(p) == np.sign(t))[moving])) if moving.any() else float("nan"),
    }
    cur = valid[:, 0].astype(bool)
    out["action_current_within_tolerance_accuracy"] = float(
        np.mean(np.all(np.abs(pred[cur, 0] - true[cur, 0]) <= tol, axis=-1)))
    for j, name in enumerate(("swing", "boom", "arm", "bucket")):
        out[f"mae_{name}"] = float(np.mean(np.abs(p[:, j] - t[:, j])))
    return out


def stage_metrics(pred_stage: np.ndarray, true_stage: np.ndarray, valid: np.ndarray, n: int = 10) -> dict[str, float]:
    m = valid.astype(bool)
    p, t = pred_stage[m], true_stage[m]
    f1s = []
    for s in range(n):
        tp = np.sum((p == s) & (t == s)); fp = np.sum((p == s) & (t != s)); fn = np.sum((p != s) & (t == s))
        if tp + fp + fn:
            f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
    return {"stage_current_accuracy": float(np.mean(pred_stage[valid[:, 0].astype(bool), 0] == true_stage[valid[:, 0].astype(bool), 0])),
            "stage_chunk_macro_f1": float(np.mean(f1s)) if f1s else float("nan")}


def success_summary(results: list[dict]) -> dict:
    """Aggregate closed-loop episodes: success rate and time-to-success distribution."""
    ok = [r for r in results if r["success"]]
    times = np.array([r["time_to_success_s"] for r in ok]) if ok else np.array([np.nan])
    return {
        "episodes": len(results), "successes": len(ok),
        "success_rate": len(ok) / max(1, len(results)),
        "time_to_success_s": {"min": float(np.min(times)), "mean": float(np.mean(times)),
                              "median": float(np.median(times)), "p95": float(np.percentile(times, 95)),
                              "max": float(np.max(times))},
    }

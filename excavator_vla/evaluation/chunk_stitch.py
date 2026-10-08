"""Open-loop chunk stitching: "how many actions can we execute before we must re-infer?"

From per-frame 50-step predictions we simulate executing the first k actions of a chunk,
then jumping to the prediction made k frames later, for k in {1, 2, 3, 5, 10, ..., 50}.
Re-scoring the stitched sequence shows how fast quality decays with execution horizon.
k = 1 must reproduce the per-frame evaluator exactly (a built-in self-check).

This analysis motivated replanning every 3 steps at deployment. Note it is an *offline*
proxy: it is not the same thing as the live replan interval, because in closed loop the
observations themselves change with the executed actions.
"""

from __future__ import annotations

import numpy as np

DEFAULT_CHUNKS = (1, 2, 3, 5, 10, 15, 20, 30, 50)


def stitch(pred: np.ndarray, true: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """pred, true: [T, H, A] per-frame chunks -> stitched [T, A] executed vs expert."""
    T, H, _ = pred.shape
    if not 1 <= k <= H:
        raise ValueError(f"chunk size must be in [1, {H}]")
    p_out, t_out = [], []
    for start in range(0, T, k):
        take = min(k, T - start)
        p_out.append(pred[start, :take])
        t_out.append(true[start, :take])
    return np.concatenate(p_out), np.concatenate(t_out)


def sweep(pred: np.ndarray, true: np.ndarray, tol: np.ndarray, chunks=DEFAULT_CHUNKS) -> dict[int, float]:
    out = {}
    for k in chunks:
        if k > pred.shape[1]:
            continue
        p, t = stitch(pred, true, k)
        out[k] = float(np.mean(np.all(np.abs(p - t) <= tol, axis=-1)))
    return out

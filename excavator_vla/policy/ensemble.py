"""Action-chunk bookkeeping: replanning and temporal ensembling.

SmolVLA predicts a chunk of 50 future actions (5 s at 10 Hz). Executing all 50 open-loop
drifts badly once the bucket meets sand, so the client re-infers every `replan_interval`
steps (3 in the final demo) and discards the unused tail. Optionally, overlapping chunks
are blended with exponentially decaying weights (ACT-style temporal ensembling); the
final configuration used width 1, i.e. no blending.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class ChunkHistory:
    def __init__(self, width: int = 1, decay: float = 0.35):
        if width < 1:
            raise ValueError("ensemble width must be >= 1")
        self.width, self.decay = width, decay
        self._chunks: deque[tuple[int, np.ndarray]] = deque(maxlen=width)

    def add(self, origin_step: int, chunk: np.ndarray) -> None:
        self._chunks.append((int(origin_step), np.asarray(chunk, dtype=np.float32)))

    def action_at(self, step: int) -> np.ndarray:
        """Blend the newest `width` chunks that cover `step`, weight = exp(-decay * age)."""
        cands = []
        for origin, chunk in reversed(self._chunks):
            k = step - origin
            if 0 <= k < len(chunk):
                cands.append((step - origin, chunk[k, :4]))
        if not cands:
            raise LookupError(f"no chunk covers step {step}; replan first")
        ages = np.array([a for a, _ in cands], dtype=np.float64)
        w = np.exp(-self.decay * ages)
        w /= w.sum()
        return np.sum(np.stack([c for _, c in cands]) * w[:, None], axis=0).astype(np.float32)


def needs_replan(step: int, replan_interval: int, chunk_origin: int | None, chunk_len: int) -> bool:
    return chunk_origin is None or step - chunk_origin >= chunk_len or step % replan_interval == 0

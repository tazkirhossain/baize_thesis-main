"""Validation-based checkpoint selection.

Centralized and federated training both keep a running best by validation
PR-AUC and only then touch the test set. The helper exists so the two paths
cannot drift (the library centralized loop previously returned the last epoch).
"""

from __future__ import annotations

import numpy as np


def best_checkpoint(
    snapshots: list[np.ndarray], scores: list[float]
) -> tuple[np.ndarray, int, float]:
    """Return (params, 1-based index, score) of the maximum validation score.

    Ties keep the earliest checkpoint, matching "first time we hit the best
    validation score" rather than a later numerically-equal wobble.
    """
    if not snapshots or not scores:
        raise ValueError("snapshots and scores must be non-empty")
    if len(snapshots) != len(scores):
        raise ValueError(
            f"length mismatch: {len(snapshots)} snapshots vs {len(scores)} scores"
        )
    i = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    return np.asarray(snapshots[i], dtype=np.float64).copy(), i + 1, float(scores[i])

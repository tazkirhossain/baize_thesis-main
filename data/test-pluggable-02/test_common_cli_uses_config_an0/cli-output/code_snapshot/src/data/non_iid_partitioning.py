"""Seeded Dirichlet ownership skew; records are never duplicated or discarded."""
import math

import numpy as np


def dirichlet_indices(labels, categories, seed=0, n_clients=5, *, alpha=0.5,
                      stratify_on="category"):
    """Draw one client-share vector per label or category, then permute rows.

    Largest-remainder rounding preserves each stratum's exact row count. Smaller
    alpha generally permits stronger skew; it does not guarantee a particular
    realized distribution. Sizes can differ. No seed retries or split inspection.
    """
    if n_clients != 5:
        raise ValueError("this baseline requires exactly five clients")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be a positive finite number")
    if stratify_on not in ("label", "category"):
        raise ValueError("stratify_on must be label or category")
    y, cats = np.asarray(labels), np.asarray(categories)
    if y.ndim != 1 or y.shape != cats.shape or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("parallel nonempty binary labels and categories are required")
    strata = y if stratify_on == "label" else cats
    rng = np.random.default_rng(seed)
    pieces = [[] for _ in range(n_clients)]
    for value in np.unique(strata):
        rows = rng.permutation(np.flatnonzero(strata == value))
        shares = rng.dirichlet(np.full(n_clients, alpha))
        if not np.isfinite(shares).all() or not np.isclose(shares.sum(), 1.0):
            raise ValueError("alpha produced numerically invalid Dirichlet shares")
        expected = shares * len(rows)
        counts = np.floor(expected).astype(np.int64)
        remainder = len(rows) - int(counts.sum())
        order = sorted(rng.permutation(n_clients), key=lambda i: -(expected[i] - counts[i]))
        counts[order[:remainder]] += 1
        for cid, chunk in enumerate(np.split(rows, np.cumsum(counts)[:-1])):
            pieces[cid].append(chunk)
    return [np.sort(np.concatenate(chunks)) for chunks in pieces]

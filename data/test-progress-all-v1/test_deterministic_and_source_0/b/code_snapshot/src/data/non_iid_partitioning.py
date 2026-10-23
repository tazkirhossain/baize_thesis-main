"""Seeded Dirichlet ownership skew; records are never duplicated or discarded."""
import math

import numpy as np
from src.data.sampling import binary_pools, consume_pool, verify_coverage


def validate_dirichlet_parameters(alpha=0.5, stratify_on="category"):
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be a positive finite number")
    if stratify_on not in ("label", "category"):
        raise ValueError("stratify_on must be label or category")


def dirichlet_indices(labels, categories, seed=0, n_clients=5, *, alpha=0.5,
                      stratify_on="category"):
    """Draw one client-share vector per label or category, then permute rows.

    Largest-remainder rounding preserves each stratum's exact row count. Smaller
    alpha generally permits stronger skew; it does not guarantee a particular
    realized distribution. Sizes can differ. No seed retries or split inspection.
    """
    if n_clients != 5:
        raise ValueError("this baseline requires exactly five clients")
    validate_dirichlet_parameters(alpha, stratify_on)
    y, cats = np.asarray(labels), np.asarray(categories)
    if y.ndim != 1 or y.shape != cats.shape or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("parallel nonempty binary labels and categories are required")
    strata = y if stratify_on == "label" else cats
    rng = np.random.default_rng(seed)
    pools = binary_pools(y, rng)
    pieces = [[] for _ in range(n_clients)]
    for value in np.unique(strata):
        rows = (pools[int(value)] if stratify_on == "label" else
                np.concatenate([pool[cats[pool] == value] for pool in pools.values()]))
        shares = rng.dirichlet(np.full(n_clients, alpha))
        if not np.isfinite(shares).all() or not np.isclose(shares.sum(), 1.0):
            raise ValueError("alpha produced numerically invalid Dirichlet shares")
        expected = shares * len(rows)
        counts = np.floor(expected).astype(np.int64)
        remainder = len(rows) - int(counts.sum())
        order = sorted(rng.permutation(n_clients), key=lambda i: -(expected[i] - counts[i]))
        counts[order[:remainder]] += 1
        for cid, chunk in enumerate(consume_pool(rows, counts)):
            pieces[cid].append(chunk)
    parts = [np.sort(np.concatenate(chunks)) for chunks in pieces]
    verify_coverage(parts, len(y))
    return parts

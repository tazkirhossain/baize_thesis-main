"""Exact-quota IID strategy and compatibility entry point."""
import numpy as np


N_CLIENTS = 5


def stratified_iid_indices(labels, categories, seed: int = 0, n_clients: int = 5):
    """Partition row indices without replacement with integer-balanced quotas.

    Each binary-class count, total count, and (label,category) stratum count
    differs by at most one between clients. Category marginals also differ by
    at most one when each category maps to a single binary class, as here.
    """
    if n_clients != N_CLIENTS:
        raise ValueError("this IID baseline requires exactly five clients")
    y = np.asarray(labels)
    cats = np.asarray(categories)
    if y.ndim != 1 or cats.ndim != 1 or y.shape != cats.shape:
        raise ValueError("labels and categories must be parallel one-dimensional arrays")
    if not np.isin(y, [0, 1]).all() or any(np.count_nonzero(y == c) < 5 for c in (0, 1)):
        raise ValueError("binary labels and at least five records of each class are required")
    rng = np.random.default_rng(seed)
    totals = np.full(5, len(y) // 5, dtype=np.int64)
    totals[rng.permutation(5)[:len(y) % 5]] += 1
    class_sizes = np.array([np.count_nonzero(y == c) for c in (0, 1)])
    quotas = np.repeat((class_sizes // 5)[:, None], 5, axis=1)
    remaining = totals - quotas.sum(axis=0)
    for c in (0, 1):
        order = sorted(rng.permutation(5), key=lambda i: -remaining[i])
        for i in order[:int(class_sizes[c] % 5)]:
            quotas[c, i] += 1
            remaining[i] -= 1
    if np.any(remaining):
        raise RuntimeError("binary quota allocation failed")
    pieces = [[] for _ in range(5)]
    for c in (0, 1):
        groups = [np.flatnonzero((y == c) & (cats == category))
                  for category in np.unique(cats[y == c])]
        # Largest-remainder first, randomly breaking equal-size ties.
        order = rng.permutation(len(groups))
        groups = [groups[i] for i in sorted(order, key=lambda i: -(len(groups[i]) % 5))]
        capacity = quotas[c] - sum(len(group) // 5 for group in groups)
        for group in groups:
            shuffled = rng.permutation(group)
            sizes = np.full(5, len(group) // 5, dtype=np.int64)
            order = sorted(rng.permutation(5), key=lambda i: -capacity[i])
            for i in order[:len(group) % 5]:
                if capacity[i] <= 0:
                    raise RuntimeError("category quota allocation failed")
                sizes[i] += 1
                capacity[i] -= 1
            for i, part in enumerate(np.split(shuffled, np.cumsum(sizes)[:-1])):
                pieces[i].append(part)
        if np.any(capacity):
            raise RuntimeError("unfilled category quota")
    parts = [np.sort(np.concatenate(items)) for items in pieces]
    joined = np.concatenate(parts)
    if not np.array_equal(np.sort(joined), np.arange(len(y))):
        raise RuntimeError("IID partition does not cover each row exactly once")
    return parts


def partition_prepared_iid(source, output, *, seed=0, split_seed=0, n_clients=5):
    """Backward-compatible IID entry point using the shared preparation engine."""
    from src.data.partitioning import partition_prepared
    return partition_prepared(source, output, strategy="iid", seed=seed,
                              split_seed=split_seed, n_clients=n_clients)

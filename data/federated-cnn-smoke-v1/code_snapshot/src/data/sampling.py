"""Shared seeded pools and sequential sampling without replacement."""
import numpy as np


SAMPLING_POLICY = "binary_shuffled_pools_sequential_v1"


def binary_pools(labels, rng):
    """Shuffle Adult and Non-adult eligible row indices independently."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("one-dimensional binary labels are required")
    return {label: rng.permutation(np.flatnonzero(labels == label)) for label in (1, 0)}


def validate_counts(counts, n_clients=5):
    if not isinstance(counts, (list, tuple, np.ndarray)) or len(counts) != n_clients:
        raise ValueError("each allocation requires exactly five client counts")
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0 for n in counts):
        raise ValueError("allocation counts must be nonnegative integers")


def consume_pool(shuffled_pool, counts):
    """Allocate to clients 1–5 in order and remove each prefix from availability.

    A prefix of a uniform random permutation is a uniform sample from the pool.
    Consuming successive prefixes is equivalent to repeated random sampling
    without replacement, with no costly repeated row deletion.
    """
    validate_counts(counts)
    available = np.asarray(shuffled_pool)
    if sum(map(int, counts)) != len(available):
        raise ValueError("allocation counts must exhaust the eligible pool exactly")
    parts = []
    for count in counts:
        count = int(count)
        parts.append(available[:count])
        available = available[count:]
    if len(available):
        raise RuntimeError("eligible pool was not exhausted")
    return parts


def verify_coverage(parts, n_records):
    """Union equals the original population; no overlap, omission or replacement."""
    if len(parts) != 5:
        raise ValueError("exactly five client partitions are required")
    if any(np.asarray(part).ndim != 1 or not np.issubdtype(np.asarray(part).dtype, np.integer)
           for part in parts):
        raise ValueError("client partitions must contain one-dimensional integer indices")
    joined = np.concatenate(parts)
    if not np.array_equal(np.sort(joined), np.arange(n_records)):
        raise ValueError("partitions must contain every eligible original record exactly once")

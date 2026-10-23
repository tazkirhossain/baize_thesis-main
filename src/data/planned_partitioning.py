"""Explicit label/category allocation plans with random record selection."""
import numpy as np

from src.data.schema import KNOWN_CATEGORIES
from src.data.sampling import binary_pools, consume_pool, validate_counts, verify_coverage


def validate_label_plan(allocation_plan=None):
    if not isinstance(allocation_plan, dict) or set(allocation_plan) != {"Adult", "Non-adult"}:
        raise ValueError("label allocation_plan must contain Adult and Non-adult count lists")
    for counts in allocation_plan.values():
        validate_counts(counts)


def validate_category_plan(allocation_plan=None):
    if not isinstance(allocation_plan, dict) or not allocation_plan:
        raise ValueError("a nonempty category allocation_plan is required")
    if any(name not in KNOWN_CATEGORIES for name in allocation_plan):
        raise ValueError("category plan keys must be known normalized original category names")
    for counts in allocation_plan.values():
        validate_counts(counts)


def planned_indices(labels, categories, seed=0, n_clients=5, *, allocation_plan=None, axis="label"):
    if n_clients != 5:
        raise ValueError("this baseline requires exactly five clients")
    if axis not in ("label", "category"):
        raise ValueError("plan axis must be label or category")
    (validate_label_plan if axis == "label" else validate_category_plan)(allocation_plan)
    y, cats = np.asarray(labels), np.asarray(categories)
    if y.ndim != 1 or y.shape != cats.shape or not len(y):
        raise ValueError("parallel nonempty labels and categories are required")
    pools = binary_pools(y, np.random.default_rng(seed))
    pieces = [[] for _ in range(5)]
    if axis == "label":
        groups = [(pools[label], allocation_plan[name]) for label, name in ((1, "Adult"), (0, "Non-adult"))]
    else:
        names = sorted(KNOWN_CATEGORIES)
        codes = {name:i for i, name in enumerate(names)}
        if set(np.unique(cats)) != {codes[name] for name in allocation_plan}:
            raise ValueError("category plan must cover exactly the eligible original categories")
        groups = []
        for name in sorted(allocation_plan):
            code = codes[name]
            label = int(name == "adult")
            if np.any(y[cats == code] != label):
                raise ValueError("category and binary label are inconsistent")
            pool = pools[label]
            groups.append((pool[cats[pool] == code], allocation_plan[name]))
    # All per-pool totals are checked before consuming any records.
    if any(sum(map(int, counts)) != len(pool) for pool, counts in groups):
        raise ValueError("allocation plan totals must match every eligible pool exactly")
    for pool, counts in groups:
        for cid, chosen in enumerate(consume_pool(pool, counts)):
            pieces[cid].append(chosen)
    parts = [np.sort(np.concatenate(chunks)) for chunks in pieces]
    verify_coverage(parts, len(y))
    return parts

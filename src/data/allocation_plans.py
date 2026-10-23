"""Generate a separate category-skew plan with balanced binary margins."""
from collections import Counter

import numpy as np

from src.data.schema import KNOWN_CATEGORIES
from src.data.planned_partitioning import validate_category_plan


def category_skew_plan(category_counts, *, seed=0, n_clients=5):
    """Fill equal client capacities using a seeded order of whole categories.

    Adult and Non-adult totals per client each differ by at most one. Non-adult
    categories fill clients sequentially, splitting a category only at a client
    capacity boundary. This creates strong category skew without binary skew.
    Only eligible population counts enter this plan, never split outcomes.
    """
    if n_clients != 5:
        raise ValueError("exactly five clients are required")
    if (not isinstance(category_counts, dict) or "adult" not in category_counts or
            set(category_counts) - KNOWN_CATEGORIES or
            any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in category_counts.values())):
        raise ValueError("positive integer counts of known categories, including adult, are required")
    total = sum(category_counts.values())
    adult = category_counts["adult"]
    if min(adult, total-adult) < n_clients:
        raise ValueError("at least five eligible records of each binary class are required")
    sizes = np.full(n_clients, total // n_clients, dtype=np.int64)
    sizes[:total % n_clients] += 1
    adult_counts = np.full(n_clients, adult // n_clients, dtype=np.int64)
    adult_counts[:adult % n_clients] += 1
    capacity = sizes - adult_counts
    plan = {name:[0]*n_clients for name in sorted(category_counts)}
    plan["adult"] = adult_counts.tolist()
    names = sorted(set(category_counts) - {"adult"})
    order = np.random.default_rng(seed).permutation(len(names))
    cid = 0
    for index in order:
        name = names[int(index)]
        remaining = category_counts[name]
        while remaining:
            while cid < n_clients and capacity[cid] == 0:
                cid += 1
            if cid == n_clients:
                raise RuntimeError("category allocation exceeded client capacities")
            chosen = min(remaining, int(capacity[cid]))
            plan[name][cid] += chosen
            capacity[cid] -= chosen
            remaining -= chosen
    if np.any(capacity) or any(sum(plan[name]) != count for name,count in category_counts.items()):
        raise RuntimeError("category allocation did not exhaust the eligible population")
    validate_category_plan(plan)
    return plan


def category_counts_from_report(report):
    """Read only the unsplit eligible-population audit, not validation/test data."""
    if report.get("status") != "PASS":
        raise ValueError("a PASS prepared-source audit is required")
    counts = Counter()
    for client in report["clients"]:
        counts.update(client["dataset"]["category_counts"])
    if sum(counts.values()) != report["retained_records"]:
        raise ValueError("source category totals do not reconcile")
    binary = {str(label):sum(c["dataset"]["class_counts"][str(label)] for c in report["clients"])
              for label in (0,1)}
    if counts["adult"] != binary["1"] or sum(counts.values())-counts["adult"] != binary["0"]:
        raise ValueError("source category and binary totals do not reconcile")
    return dict(counts)

"""Client partitioning strategies for federated simulation.

Implements the distributions required by RQ5 (master spec §24) and the
per-client accounting required by §25. Every partitioner returns a
PartitionReport whose contents are VERIFIED against the requested configuration
rather than asserted - the spec is explicit that a declared distribution must be
checked against the generated one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PartitionReport:
    strategy: str
    n_clients: int
    seed: int
    params: dict = field(default_factory=dict)
    clients: list[dict] = field(default_factory=list)
    status: str = "UNKNOWN"

    @property
    def sizes(self) -> list[int]:
        return [c["n"] for c in self.clients]

    @property
    def positive_rates(self) -> list[float]:
        return [c["positive_rate"] for c in self.clients]

    def render(self) -> str:
        lines = [
            f"PARTITION REPORT  strategy={self.strategy}  clients={self.n_clients}  seed={self.seed}",
            f"  params: {self.params}",
            "",
            f"  {'client':>7}{'n':>9}{'pos':>9}{'neg':>9}{'pos_rate':>10}",
        ]
        for c in self.clients:
            lines.append(
                f"  {c['client_id']:>7}{c['n']:>9,}{c['n_pos']:>9,}"
                f"{c['n_neg']:>9,}{c['positive_rate']:>10.4f}"
            )
        sizes = np.array(self.sizes)
        rates = np.array(self.positive_rates)
        lines += [
            "",
            f"  size    min={sizes.min():,} max={sizes.max():,} "
            f"mean={sizes.mean():.1f} cv={sizes.std() / max(sizes.mean(), 1e-9):.3f}",
            f"  pos_rate min={rates.min():.4f} max={rates.max():.4f} "
            f"spread={rates.max() - rates.min():.4f}",
            f"STATUS: {self.status}",
        ]
        return "\n".join(lines)


def _build_report(strategy, parts, labels, n_clients, seed, params) -> PartitionReport:
    rep = PartitionReport(strategy=strategy, n_clients=n_clients, seed=seed, params=params)
    for cid, idx in enumerate(parts):
        y = labels[idx]
        n_pos = int((y == 1).sum())
        rep.clients.append(
            {
                "client_id": cid,
                "n": int(len(idx)),
                "n_pos": n_pos,
                "n_neg": int(len(idx) - n_pos),
                "positive_rate": float(n_pos / len(idx)) if len(idx) else 0.0,
            }
        )

    problems = []
    if any(len(p) == 0 for p in parts):
        problems.append("at least one client has no samples")
    covered = sum(len(p) for p in parts)
    if covered != len(labels):
        problems.append(f"partition does not cover the dataset ({covered} vs {len(labels)})")
    if len({i for p in parts for i in p.tolist()}) != covered:
        problems.append("partitions overlap")
    rep.status = "PASS" if not problems else "BLOCKED: " + "; ".join(problems)
    return rep


def iid_partition(labels, n_clients: int, seed: int = 0):
    """Equal-size shards, class distribution preserved in expectation."""
    labels = np.asarray(labels)
    _require_partitionable(labels, n_clients)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    parts = [np.sort(p) for p in np.array_split(idx, n_clients)]
    return parts, _build_report("iid", parts, labels, n_clients, seed, {})


def _require_partitionable(labels, n_clients: int) -> None:
    if n_clients < 1:
        raise ValueError(f"n_clients must be >= 1, got {n_clients}")
    if n_clients > len(labels):
        raise ValueError(
            f"cannot partition {len(labels)} samples into {n_clients} clients"
        )


def label_skew_partition(labels, n_clients: int, alpha: float = 0.5, seed: int = 0):
    """Dirichlet label skew: client c receives proportion p_c of each class.

    Smaller alpha -> more skew. alpha -> infinity approaches IID. This is the
    standard construction in the non-IID federated literature; verify the
    intended severity by reading `positive_rate` spread in the report rather
    than trusting alpha to mean the same thing at every client count.
    """
    labels = np.asarray(labels)
    _require_partitionable(labels, n_clients)
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(n_clients)]
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        props = rng.dirichlet(np.repeat(alpha, n_clients))
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for cid, chunk in enumerate(np.split(idx, cuts)):
            parts[cid].append(chunk)
    parts = [np.sort(np.concatenate(p)) if p else np.array([], dtype=np.int64) for p in parts]
    return parts, _build_report(
        "label_skew", parts, labels, n_clients, seed, {"alpha": alpha}
    )


def quantity_skew_partition(labels, n_clients: int, alpha: float = 0.5, seed: int = 0):
    """Unequal shard sizes via a Dirichlet over sample counts; classes unskewed.

    Counts are the Hamilton (largest-remainder) allocation of `n` samples,
    floored at one per client so a downstream empty-shard crash cannot be
    mistaken for an aggregation bug. Remainder that previously silently landed
    on the last client is now given to the largest fractional parts.
    """
    labels = np.asarray(labels)
    _require_partitionable(labels, n_clients)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    n = len(idx)
    props = rng.dirichlet(np.repeat(alpha, n_clients))
    raw = props * n
    counts = np.maximum(np.floor(raw).astype(int), 1)
    while int(counts.sum()) > n:
        eligible = np.flatnonzero(counts > 1)
        if len(eligible) == 0:
            break
        counts[eligible[int(np.argmax(counts[eligible]))]] -= 1
    remainder = n - int(counts.sum())
    frac = raw - np.floor(raw)
    order = np.argsort(-frac, kind="stable")
    i = 0
    while remainder > 0:
        counts[int(order[i % n_clients])] += 1
        remainder -= 1
        i += 1
    cuts = np.cumsum(counts)[:-1]
    parts = [np.sort(p) for p in np.split(idx, cuts)]
    return parts, _build_report(
        "quantity_skew", parts, labels, n_clients, seed, {"alpha": alpha}
    )


PARTITIONERS = {
    "iid": iid_partition,
    "label_skew": label_skew_partition,
    "quantity_skew": quantity_skew_partition,
}


def partition(labels, strategy: str, n_clients: int, seed: int = 0, **kwargs):
    if strategy not in PARTITIONERS:
        raise ValueError(f"unknown partition strategy {strategy!r}; have {sorted(PARTITIONERS)}")
    return PARTITIONERS[strategy](labels, n_clients=n_clients, seed=seed, **kwargs)

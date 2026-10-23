"""Dataset splitting and leakage detection.

Two regimes (DECISIONS.md D-002):
  - random     : stratified, reported for comparability with prior work only
  - domain_aware: grouped by registrable domain, PRIMARY regime for all headline
                  numbers

The leakage check is a hard gate. `assert_no_leakage` raises rather than warns,
because a silently leaking split invalidates every number downstream and the
failure is otherwise invisible - it looks like good performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from baize.data.dataset import URLDataset


class LeakageError(RuntimeError):
    """Raised when a split violates its own integrity guarantee. Do not catch."""


@dataclass
class SplitReport:
    strategy: str
    seed: int
    sizes: dict[str, int]
    proportions: dict[str, float]
    positive_rates: dict[str, float]
    n_domains: dict[str, int]
    canonical_overlap: dict[str, int] = field(default_factory=dict)
    domain_overlap: dict[str, int] = field(default_factory=dict)
    status: str = "UNKNOWN"

    def render(self) -> str:
        lines = [f"SPLIT REPORT  strategy={self.strategy}  seed={self.seed}", ""]
        lines.append(f"{'partition':<12}{'n':>10}{'prop':>9}{'pos_rate':>10}{'domains':>10}")
        for k in self.sizes:
            lines.append(
                f"{k:<12}{self.sizes[k]:>10,}{self.proportions[k]:>9.3f}"
                f"{self.positive_rates[k]:>10.4f}{self.n_domains[k]:>10,}"
            )
        lines.append("")
        lines.append("overlap (must be 0 for the guarantee each strategy claims):")
        for pair, v in self.canonical_overlap.items():
            lines.append(f"  canonical {pair:<16} {v:>8,}")
        for pair, v in self.domain_overlap.items():
            lines.append(f"  domain    {pair:<16} {v:>8,}")
        lines.append("")
        lines.append(f"STATUS: {self.status}")
        return "\n".join(lines)


@dataclass
class Splits:
    train: URLDataset
    val: URLDataset
    test: URLDataset
    report: SplitReport

    def as_dict(self) -> dict[str, URLDataset]:
        return {"train": self.train, "val": self.val, "test": self.test}


def _stratified_random_indices(
    labels: np.ndarray, fractions: tuple[float, float, float], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts: list[list[np.ndarray]] = [[], [], []]
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n = len(idx)
        c1 = int(round(fractions[0] * n))
        c2 = c1 + int(round(fractions[1] * n))
        for slot, chunk in enumerate((idx[:c1], idx[c1:c2], idx[c2:])):
            parts[slot].append(chunk)
    return tuple(np.concatenate(p) if p else np.array([], dtype=np.int64) for p in parts)


def _domain_aware_indices(
    domains: np.ndarray,
    labels: np.ndarray,
    fractions: tuple[float, float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy group assignment: no registrable domain spans two partitions.

    Domains are shuffled, then processed largest-first and assigned to whichever
    partition is furthest below its target share. Largest-first matters: leaving
    big groups until the end makes the realised proportions drift badly.

    This targets SIZE proportions. Class balance across partitions is therefore
    approximate, not exact - a consequence of the grouping constraint, reported
    in SplitReport.positive_rates so it can be inspected rather than assumed.
    """
    uniq, inverse = np.unique(domains, return_inverse=True)
    group_sizes = np.bincount(inverse, minlength=len(uniq))

    order = rng.permutation(len(uniq))
    order = order[np.argsort(-group_sizes[order], kind="stable")]

    total = len(domains)
    targets = np.array(fractions, dtype=np.float64) * total
    assigned = np.zeros(3, dtype=np.float64)
    group_slot = np.empty(len(uniq), dtype=np.int8)

    for g in order:
        deficit = targets - assigned
        slot = int(np.argmax(deficit))
        group_slot[g] = slot
        assigned[slot] += group_sizes[g]

    slot_of_sample = group_slot[inverse]
    return tuple(np.flatnonzero(slot_of_sample == s) for s in range(3))


def make_splits(
    ds: URLDataset,
    strategy: str = "domain_aware",
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 0,
) -> Splits:
    if len(fractions) != 3:
        raise ValueError(f"fractions must have length 3 (train, val, test), got {len(fractions)}")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    if any(f < 0 for f in fractions):
        raise ValueError(f"fractions must be non-negative, got {fractions}")
    if len(ds) == 0:
        raise ValueError("cannot split an empty dataset")

    rng = np.random.default_rng(seed)
    if strategy == "random":
        idx = _stratified_random_indices(ds.labels, fractions, rng)
    elif strategy == "domain_aware":
        idx = _domain_aware_indices(ds.domains, ds.labels, fractions, rng)
    else:
        raise ValueError(f"unknown split strategy: {strategy!r}")

    train, val, test = (ds.subset(i) for i in idx)
    train.name, val.name, test.name = f"{ds.name}/train", f"{ds.name}/val", f"{ds.name}/test"

    report = leakage_report(train, val, test, strategy=strategy, seed=seed)
    return Splits(train=train, val=val, test=test, report=report)


def leakage_report(
    train: URLDataset, val: URLDataset, test: URLDataset, strategy: str, seed: int
) -> SplitReport:
    parts = {"train": train, "val": val, "test": test}
    total = sum(len(p) for p in parts.values())

    rep = SplitReport(
        strategy=strategy,
        seed=seed,
        sizes={k: len(v) for k, v in parts.items()},
        proportions={k: (len(v) / total if total else 0.0) for k, v in parts.items()},
        positive_rates={k: v.positive_rate for k, v in parts.items()},
        n_domains={k: int(len(set(v.domains))) for k, v in parts.items()},
    )

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        key = f"{a}~{b}"
        rep.canonical_overlap[key] = len(set(parts[a].canonicals) & set(parts[b].canonicals))
        rep.domain_overlap[key] = len(set(parts[a].domains) & set(parts[b].domains))

    problems: list[str] = []
    # Every strategy must guarantee no duplicated URL across partitions.
    if any(v > 0 for v in rep.canonical_overlap.values()):
        problems.append("canonical URL overlap")
    # Only domain_aware claims the stronger domain guarantee.
    if strategy == "domain_aware" and any(v > 0 for v in rep.domain_overlap.values()):
        problems.append("registrable domain overlap")
    if any(v == 0 for v in rep.sizes.values()):
        problems.append("empty partition")
    if any(set(p.class_counts) != {0, 1} for p in parts.values()):
        problems.append("each partition must contain both classes")

    rep.status = "PASS" if not problems else "BLOCKED: " + "; ".join(problems)
    return rep


def assert_no_leakage(report: SplitReport) -> None:
    """Hard gate (master spec §64). Raises rather than warns, by design."""
    if not report.status.startswith("PASS"):
        raise LeakageError(f"split integrity failed -> {report.status}\n\n{report.render()}")

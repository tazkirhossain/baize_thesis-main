"""Dataset validation and deduplication (Phase 1 gate).

Order of operations in the pipeline is load -> validate -> deduplicate -> split.
Deduplicating AFTER splitting would be wrong: identical URLs would already have
been scattered across partitions, and removing them then would not undo the
leakage that has already been baked into the split.

The validator is deliberately schema-light. Source-specific schema checks belong
in the loader that produces the URLDataset (`baize/data/loaders.py`).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from baize.data.dataset import URLDataset


class ValidationError(RuntimeError):
    pass


@dataclass
class ValidationReport:
    dataset_name: str
    n_records: int
    n_dropped_at_load: int
    drop_reasons: dict[str, int]
    class_counts: dict[int, int]
    positive_rate: float
    n_unique_canonical: int
    n_unique_domains: int
    n_exact_duplicates: int
    domain_concentration: list[tuple[str, int]]
    url_length: dict[str, float]
    findings: list[str] = field(default_factory=list)
    status: str = "UNKNOWN"

    def render(self) -> str:
        L = [
            f"DATASET VALIDATION  {self.dataset_name}",
            "=" * 60,
            f"  records retained      : {self.n_records:,}",
            f"  dropped at load       : {self.n_dropped_at_load:,}",
        ]
        for r, c in sorted(self.drop_reasons.items(), key=lambda kv: -kv[1]):
            L.append(f"      {r:<28} {c:>8,}")
        L += [
            "",
            f"  class counts          : {self.class_counts}",
            f"  positive rate         : {self.positive_rate:.4f}",
            f"  unique canonical URLs : {self.n_unique_canonical:,}",
            f"  exact duplicates      : {self.n_exact_duplicates:,}",
            f"  unique domains        : {self.n_unique_domains:,}",
            "",
            "  URL length (host+path):",
            f"      min={self.url_length['min']:.0f}  p50={self.url_length['p50']:.0f}  "
            f"p95={self.url_length['p95']:.0f}  max={self.url_length['max']:.0f}",
            "",
            "  top domains by sample count:",
        ]
        for d, c in self.domain_concentration:
            L.append(f"      {d:<40} {c:>8,}")
        if self.findings:
            L += ["", "  FINDINGS:"]
            L += [f"      - {f}" for f in self.findings]
        L += ["", f"STATUS: {self.status}"]
        return "\n".join(L)


def validate_dataset(
    ds: URLDataset,
    min_records: int = 100,
    max_domain_share: float = 0.25,
    min_positive_rate: float = 0.001,
) -> ValidationReport:
    """Characterise the dataset and decide whether it may proceed to splitting."""
    n = len(ds)
    canon = ds.canonicals
    domains = ds.domains
    lengths = np.array([len(u.feature_text) for u in ds.urls], dtype=float) if n else np.zeros(1)

    dom_counts = Counter(domains.tolist())
    top = dom_counts.most_common(10)

    rep = ValidationReport(
        dataset_name=ds.name,
        n_records=n,
        n_dropped_at_load=int(sum(ds.dropped.values())),
        drop_reasons=dict(ds.dropped),
        class_counts=ds.class_counts,
        positive_rate=ds.positive_rate,
        n_unique_canonical=int(len(set(canon.tolist()))),
        n_unique_domains=int(len(dom_counts)),
        n_exact_duplicates=int(n - len(set(canon.tolist()))),
        domain_concentration=top,
        url_length={
            "min": float(lengths.min()),
            "p50": float(np.percentile(lengths, 50)),
            "p95": float(np.percentile(lengths, 95)),
            "max": float(lengths.max()),
        },
    )

    findings, blocking = [], []

    if n < min_records:
        blocking.append(f"only {n} records, below minimum {min_records}")
    if len(rep.class_counts) < 2:
        blocking.append(f"dataset is single-class: {rep.class_counts}")
    elif rep.positive_rate < min_positive_rate or rep.positive_rate > 1 - min_positive_rate:
        blocking.append(f"degenerate class balance: positive rate {rep.positive_rate:.5f}")

    if rep.n_exact_duplicates:
        findings.append(
            f"{rep.n_exact_duplicates:,} exact duplicate canonical URLs "
            f"({rep.n_exact_duplicates / n:.2%}) - must deduplicate before splitting"
        )
    if top and top[0][1] / n > max_domain_share:
        findings.append(
            f"domain concentration: {top[0][0]} accounts for {top[0][1] / n:.1%} of records; "
            "a domain-aware split will place this entire block in one partition"
        )
    if rep.n_unique_domains < 20:
        blocking.append(
            f"only {rep.n_unique_domains} unique domains - a domain-aware split "
            "cannot be constructed meaningfully (DECISIONS.md D-002 reversal condition)"
        )
    if rep.n_dropped_at_load / max(n + rep.n_dropped_at_load, 1) > 0.10:
        findings.append(
            f"{rep.n_dropped_at_load:,} records dropped at load "
            "- inspect drop reasons before accepting the corpus"
        )

    rep.findings = findings + blocking
    rep.status = "PASS" if not blocking else "BLOCKED: " + "; ".join(blocking)
    if not blocking and findings:
        rep.status = "PASS WITH FINDINGS"
    return rep


@dataclass
class DedupReport:
    strategy: str
    n_before: int
    n_after: int
    n_removed: int
    n_label_conflicts: int
    conflict_examples: list[tuple[str, list[int]]] = field(default_factory=list)

    def render(self) -> str:
        L = [
            f"DEDUPLICATION  strategy={self.strategy}",
            f"  before          : {self.n_before:,}",
            f"  after           : {self.n_after:,}",
            f"  removed         : {self.n_removed:,}",
            f"  label conflicts : {self.n_label_conflicts:,}",
        ]
        if self.conflict_examples:
            L.append("  examples (same URL, disagreeing labels):")
            for c, labs in self.conflict_examples[:5]:
                L.append(f"      {c[:60]:<60} labels={labs}")
        return "\n".join(L)


def deduplicate(
    ds: URLDataset, strategy: str = "canonical", conflict: str = "drop"
) -> tuple[URLDataset, DedupReport]:
    """Remove duplicate URLs, keeping the first occurrence.

    Records whose canonical form appears more than once with DISAGREEING labels
    are label conflicts. These are direct evidence of noise already present in
    the source corpus, so the count is reported, not silently resolved - it
    feeds the label-noise floor estimate in LABELING_POLICY.md §6.

    conflict='drop'  : discard all copies of a conflicting URL (default, safest)
    conflict='first' : keep the first occurrence and its label
    """
    if strategy != "canonical":
        raise ValueError(f"unknown dedup strategy {strategy!r}")
    if conflict not in ("drop", "first"):
        raise ValueError(f"unknown conflict resolution {conflict!r}; must be 'drop' or 'first'")

    keys = ds.canonicals
    groups: dict[str, list[int]] = {}
    for i, k in enumerate(keys.tolist()):
        groups.setdefault(k, []).append(i)

    keep, conflicts, examples = [], 0, []
    for k, idxs in groups.items():
        labs = sorted({int(ds.labels[i]) for i in idxs})
        if len(labs) > 1:
            conflicts += 1
            if len(examples) < 20:
                examples.append((k, labs))
            if conflict == "drop":
                continue
        keep.append(idxs[0])

    keep = np.sort(np.array(keep, dtype=np.int64))
    out = ds.subset(keep)
    out.name = ds.name
    rep = DedupReport(
        strategy=strategy,
        n_before=len(ds),
        n_after=len(out),
        n_removed=len(ds) - len(out),
        n_label_conflicts=conflicts,
        conflict_examples=examples,
    )
    return out, rep


def assert_validation_passed(report: ValidationReport) -> None:
    """Hard gate (master spec §64)."""
    if report.status.startswith("BLOCKED"):
        raise ValidationError(f"{report.status}\n\n{report.render()}")

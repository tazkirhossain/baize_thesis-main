"""Dataset container and a synthetic corpus for pipeline validation.

The synthetic generator is a TEST FIXTURE. It must never appear in a results
table. Any figure produced from it is labelled SYNTHETIC.

The real corpus is identified in `docs/DATASET.md` (D-010 closed). This module
still does not assume a source schema: `URLDataset.from_records` is the single
entry point a loader must satisfy.
"""

from __future__ import annotations

import hashlib
from itertools import zip_longest
from dataclasses import dataclass, field

import numpy as np

from baize.data.url import NormalizedURL, normalize_url

POSITIVE = 1  # Adult
NEGATIVE = 0  # Non-Adult


def binary_label(value) -> int:
    """Validate before narrowing to int8; never truncate fractions or overflow."""
    if isinstance(value, str):
        if value.strip() not in ("0", "1"):
            raise ValueError("label must be exactly 0 or 1")
        return int(value.strip())
    if not np.isscalar(value) or value not in (0, 1):
        raise ValueError("label must be exactly 0 or 1")
    return int(value)


@dataclass
class URLDataset:
    """Normalised URLs with binary labels.

    Invariant: every retained record has is_valid=True. Invalid records are
    dropped at construction and counted in `dropped`.
    """

    urls: list[NormalizedURL]
    labels: np.ndarray
    name: str = "unnamed"
    dropped: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.labels)
        if values.ndim != 1:
            raise ValueError("labels must be a one-dimensional vector")
        self.labels = np.array([binary_label(y) for y in values], dtype=np.int8)
        if len(self.urls) != len(self.labels):
            raise ValueError(f"length mismatch: {len(self.urls)} urls vs {len(self.labels)} labels")
        if any(not u.is_valid for u in self.urls):
            raise ValueError("dataset contains an invalid normalized URL")

    def __len__(self) -> int:
        return len(self.urls)

    @property
    def texts(self) -> list[str]:
        return [u.feature_text for u in self.urls]

    @property
    def domains(self) -> np.ndarray:
        return np.array([u.registrable_domain for u in self.urls], dtype=object)

    @property
    def canonicals(self) -> np.ndarray:
        return np.array([u.canonical for u in self.urls], dtype=object)

    @property
    def class_counts(self) -> dict[int, int]:
        vals, counts = np.unique(self.labels, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}

    @property
    def positive_rate(self) -> float:
        return float(self.labels.mean()) if len(self) else 0.0

    def subset(self, idx) -> "URLDataset":
        idx = np.asarray(idx, dtype=np.int64)
        return URLDataset(
            urls=[self.urls[i] for i in idx],
            labels=self.labels[idx],
            name=f"{self.name}[subset:{len(idx)}]",
            dropped=dict(self.dropped),
        )

    def fingerprint(self) -> str:
        """Content hash for provenance (master spec §6, §50).

        Hashes canonical URL + label in order, so any silent modification of the
        dataset changes the hash and invalidates the experiment record.
        """
        h = hashlib.sha256()
        for u, y in zip(self.urls, self.labels):
            h.update(u.canonical.encode("utf-8", "replace"))
            h.update(b"\x00")
            h.update(str(int(y)).encode())
            h.update(b"\n")
        return h.hexdigest()

    @classmethod
    def from_records(cls, raw_urls, raw_labels, name: str = "unnamed") -> "URLDataset":
        """Build from parallel sequences. Drops unparseable URLs and bad labels.

        This is the interface any real dataset loader must produce. Label mapping
        from the source taxonomy to {0,1} happens BEFORE this call and is
        documented per LABELING_POLICY.md §4 - it is deliberately not done here,
        so that the mapping is always an explicit, reviewable step.
        """
        urls: list[NormalizedURL] = []
        labels: list[int] = []
        dropped: dict[str, int] = {}

        missing = object()
        for raw_u, raw_y in zip_longest(raw_urls, raw_labels, fillvalue=missing):
            if raw_u is missing or raw_y is missing:
                raise ValueError("length mismatch between URLs and labels")
            n = normalize_url(raw_u)
            if not n.is_valid:
                dropped[n.reason] = dropped.get(n.reason, 0) + 1
                continue
            try:
                y = binary_label(raw_y)
            except (TypeError, ValueError):
                dropped["unparseable_label"] = dropped.get("unparseable_label", 0) + 1
                continue
            if y not in (NEGATIVE, POSITIVE):
                dropped[f"label_out_of_range:{y}"] = dropped.get(f"label_out_of_range:{y}", 0) + 1
                continue
            urls.append(n)
            labels.append(y)

        return cls(urls=urls, labels=np.array(labels, dtype=np.int8), name=name, dropped=dropped)


# --------------------------------------------------------------------------- #
# Synthetic corpus
# --------------------------------------------------------------------------- #

# Deliberately abstract markers. The pipeline only needs a *learnable string
# signal* to exercise every component; using real explicit vocabulary would put
# it in the repository for no scientific gain (LABELING_POLICY.md §7).
_POS_MARKERS = ["zqvx", "krellon", "vashtu", "brimlok", "quenyx"]
_NEG_MARKERS = ["ledger", "harbour", "council", "almanac", "foundry"]
_NEUTRAL = ["about", "index", "contact", "archive", "page", "list", "view", "home"]
_TLDS = ["com", "net", "org", "co.uk", "de", "io"]


def make_synthetic_dataset(
    n_domains: int = 400,
    pages_per_domain: tuple[int, int] = (1, 12),
    positive_rate: float = 0.25,
    hard_case_rate: float = 0.12,
    seed: int = 0,
    name: str = "synthetic",
) -> URLDataset:
    """Generate a corpus with realistic *structure*, not realistic content.

    Two properties matter for testing:

    1. Labels are assigned per DOMAIN, and a domain contributes several pages.
       Without this, a random split and a domain-aware split are identical and
       the leakage machinery cannot be exercised.
    2. `hard_case_rate` produces the two irreducible error types described in
       LABELING_POLICY.md §5: positive-labelled sites whose URL looks neutral,
       and negative-labelled sites whose URL carries a positive marker. These put
       a ceiling below 100% accuracy, as in the real task.
    """
    rng = np.random.default_rng(seed)
    raw_urls: list[str] = []
    labels: list[int] = []

    for d in range(n_domains):
        is_pos = rng.random() < positive_rate
        # Hard case: the domain's label does not match its surface string.
        misleading = rng.random() < hard_case_rate

        show_pos_marker = is_pos != misleading  # XOR
        pool = _POS_MARKERS if show_pos_marker else _NEG_MARKERS
        marker = pool[int(rng.integers(len(pool)))]
        host = f"{marker}{int(rng.integers(1000))}.{_TLDS[int(rng.integers(len(_TLDS)))]}"
        if rng.random() < 0.3:
            host = f"{_NEUTRAL[int(rng.integers(len(_NEUTRAL)))]}.{host}"

        n_pages = int(rng.integers(pages_per_domain[0], pages_per_domain[1] + 1))
        for _ in range(n_pages):
            depth = int(rng.integers(0, 3))
            segs = [_NEUTRAL[int(rng.integers(len(_NEUTRAL)))] for _ in range(depth)]
            if show_pos_marker and rng.random() < 0.5:
                segs.append(pool[int(rng.integers(len(pool)))])
            path = "/" + "/".join(segs) if segs else "/"
            raw_urls.append(f"https://{host}{path}")
            labels.append(POSITIVE if is_pos else NEGATIVE)

    return URLDataset.from_records(raw_urls, labels, name=name)

"""Loader for the 15-category directory-style URL corpus (`url.csv`).

Source format: headerless CSV, three fields, `id,url,label`.
Label column is a 15-value directory taxonomy:

    Adult, Arts, Business, Computers, Games, Health, Home, Kids, News,
    Recreation, Reference, Science, Shopping, Society, Sports

The taxonomy matches the Open Directory Project (DMOZ) top-level categories.
I have NOT verified the corpus provenance from the file itself - the taxonomy
match is strong evidence but not proof, and the collection date, methodology and
licence are unknown. Record them in DATASET.md once confirmed with the source,
and treat provenance as UNVERIFIED until then.

LABEL MAPPING (LABELING_POLICY.md §4)
------------------------------------
    Adult          -> 1
    all 14 others  -> 0

This is the mapping the original repository uses and it is retained for
comparability. Two caveats that must reach the limitations chapter:

1. DEFINITION MISMATCH. LABELING_POLICY.md §1 defines Adult as "primary purpose
   is sexually explicit material". A directory's Adult category is an editorial
   grouping which need not coincide with that. The mapping is therefore an
   ASSUMPTION about the source taxonomy, not a measurement, and is exactly what
   the §6 label audit is meant to test.

2. FALSE-NEGATIVE LEAKAGE ACROSS CATEGORIES. Sexually explicit sites filed under
   another category (e.g. Shopping, Society) become label 0. Every such row is a
   silent false negative in the ground truth. The audit should sample non-Adult
   categories specifically, not only Adult.

Conversely, categories that our policy explicitly EXCLUDES from Adult - Health
(sexual health), Society (LGBTQ+ resources), Kids - already map to 0 here, which
agrees with the policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from baize.data.dataset import URLDataset

SOURCE_COLUMNS = ["id", "url", "label"]
POSITIVE_CATEGORY = "adult"
KNOWN_CATEGORIES = frozenset({
    "adult", "arts", "business", "computers", "games", "health", "home",
    "kids", "news", "recreation", "reference", "science", "shopping",
    "society", "sports",
})


@dataclass
class LabelMappingReport:
    source_categories: dict[str, int] = field(default_factory=dict)
    mapped_positive: list[str] = field(default_factory=list)
    mapped_negative: list[str] = field(default_factory=list)
    n_rows_read: int = 0
    n_unmapped: int = 0

    def render(self) -> str:
        total = sum(self.source_categories.values()) or 1
        L = [
            "SOURCE LABEL MAPPING (LABELING_POLICY.md §4)",
            "=" * 60,
            f"  rows read            : {self.n_rows_read:,}",
            f"  source categories    : {len(self.source_categories)}",
            "",
            f"  {'category':<14}{'count':>12}{'share':>9}  -> BAIZE",
        ]
        for cat, n in sorted(self.source_categories.items(), key=lambda kv: -kv[1]):
            key = cat.lower()
            if key == POSITIVE_CATEGORY:
                tgt = "1"
            elif key in KNOWN_CATEGORIES:
                tgt = "0"
            else:
                tgt = "unmapped"
            L.append(f"  {cat:<14}{n:>12,}{n / total:>8.2%}  ->   {tgt}")
        pos = sum(n for c, n in self.source_categories.items() if c.lower() == POSITIVE_CATEGORY)
        mapped_neg = sum(
            n for c, n in self.source_categories.items()
            if c.lower() in KNOWN_CATEGORIES and c.lower() != POSITIVE_CATEGORY
        )
        L += [
            "",
            f"  mapped to 1 (Adult)  : {pos:,}  ({pos / total:.4%})",
            f"  mapped to 0 (other)  : {mapped_neg:,}",
            f"  unmapped (dropped)   : {self.n_unmapped:,}",
            "",
            "  ASSUMPTION: the source 'Adult' category is taken to mean the same",
            "  thing as LABELING_POLICY.md §1. Untested until the label audit runs.",
        ]
        return "\n".join(L)


def load_url_csv(
    path: str,
    nrows: int | None = None,
    name: str = "url-csv",
    chunksize: int = 200_000,
) -> tuple[URLDataset, LabelMappingReport]:
    """Load the corpus and map the source taxonomy to BAIZE's binary target.

    Reads in chunks so that the 1.5M-row file does not need to be held twice in
    memory during parsing. No shuffling and no subsampling happen here: the file
    is SORTED BY CATEGORY, so any positional subsample is a category subsample.
    Selecting a subset is the splitter's job, after a proper shuffle.
    """
    rep = LabelMappingReport()
    urls: list[str] = []
    raw_labels: list[int] = []

    reader = pd.read_csv(
        path,
        names=SOURCE_COLUMNS,
        usecols=["url", "label"],
        chunksize=chunksize,
        nrows=nrows,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace",
    )

    for chunk in reader:
        chunk = chunk[chunk["url"].notna() & chunk["label"].notna()]
        for cat, n in chunk["label"].value_counts().items():
            rep.source_categories[str(cat)] = rep.source_categories.get(str(cat), 0) + int(n)
        for url, cat in zip(chunk["url"].tolist(), chunk["label"].tolist()):
            key = str(cat).strip().lower()
            if key not in KNOWN_CATEGORIES:
                rep.n_unmapped += 1
                continue
            urls.append(url)
            raw_labels.append(1 if key == POSITIVE_CATEGORY else 0)
        rep.n_rows_read += len(chunk)

    rep.mapped_positive = [c for c in rep.source_categories if c.lower() == POSITIVE_CATEGORY]
    rep.mapped_negative = [
        c for c in rep.source_categories
        if c.lower() in KNOWN_CATEGORIES and c.lower() != POSITIVE_CATEGORY
    ]

    ds = URLDataset.from_records(urls, raw_labels, name=name)
    return ds, rep


def sample_url_csv(
    path: str,
    n: int,
    seed: int = 0,
    name: str = "url-csv-sample",
    chunksize: int = 200_000,
) -> tuple[URLDataset, LabelMappingReport]:
    """Uniform reservoir sample of mapped rows.

    `url.csv` is sorted by category. A positional prefix (`nrows=`) is therefore
    a category subsample, not a corpus subsample — Adult would dominate. The
    reservoir walks the whole file so a demo-sized draw keeps the source mix.
    """
    if n <= 0:
        raise ValueError(f"sample size must be positive, got {n}")

    rng = np.random.default_rng(seed)
    rep = LabelMappingReport()
    reservoir: list[tuple[str, int]] = []
    seen = 0

    reader = pd.read_csv(
        path,
        names=SOURCE_COLUMNS,
        usecols=["url", "label"],
        chunksize=chunksize,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace",
    )

    for chunk in reader:
        chunk = chunk[chunk["url"].notna() & chunk["label"].notna()]
        for cat, count in chunk["label"].value_counts().items():
            rep.source_categories[str(cat)] = rep.source_categories.get(str(cat), 0) + int(count)
        for url, cat in zip(chunk["url"].tolist(), chunk["label"].tolist()):
            key = str(cat).strip().lower()
            if key not in KNOWN_CATEGORIES:
                rep.n_unmapped += 1
                continue
            y = 1 if key == POSITIVE_CATEGORY else 0
            seen += 1
            if len(reservoir) < n:
                reservoir.append((url, y))
            else:
                j = int(rng.integers(0, seen))
                if j < n:
                    reservoir[j] = (url, y)
        rep.n_rows_read += len(chunk)

    rep.mapped_positive = [c for c in rep.source_categories if c.lower() == POSITIVE_CATEGORY]
    rep.mapped_negative = [
        c for c in rep.source_categories
        if c.lower() in KNOWN_CATEGORIES and c.lower() != POSITIVE_CATEGORY
    ]
    urls = [u for u, _ in reservoir]
    raw_labels = [y for _, y in reservoir]
    ds = URLDataset.from_records(urls, raw_labels, name=name)
    return ds, rep


def load_with_categories(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Load raw rows keeping the source category, for per-category error analysis.

    Needed because once labels are binarised the information about WHICH
    non-Adult category a false positive came from is gone, and that breakdown is
    the most informative part of the error analysis (LABELING_POLICY.md §2:
    false positives on Health and Society are the harmful ones).
    """
    return pd.read_csv(
        path,
        names=SOURCE_COLUMNS,
        usecols=["url", "label"],
        nrows=nrows,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace",
    )

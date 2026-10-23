"""Stateless hashed-character/L2 features retained for prepared-data inspection.

Neural training uses the separate fixed character or pretrained tokenizer encoder.
No vocabulary or normalization statistics are fitted on evaluation data.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import HashingVectorizer


def build_vectorizer(
    kind: str = "hashing",
    n_features: int = 2**18,
    ngram_range: tuple[int, int] = (3, 5),
    analyzer: str = "char_wb",
    lowercase: bool = True,
    **kwargs,
):
    if kind == "hashing":
        return HashingVectorizer(
            n_features=n_features,
            ngram_range=ngram_range,
            analyzer=analyzer,
            lowercase=lowercase,
            alternate_sign=False,  # keep features non-negative; norms stay interpretable
            norm="l2",
            **kwargs,
        )
    raise ValueError(f"only stateless hashing is supported, got {kind!r}")

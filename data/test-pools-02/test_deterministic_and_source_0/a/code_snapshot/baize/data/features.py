"""Feature extraction for URL strings.

Default is a HASHING vectoriser over character n-grams, not TF-IDF.

Reason (federated correctness, not convenience): a TF-IDF vectoriser must be fit
on a corpus to build a vocabulary. In a federated setting that vocabulary would
either have to be built centrally from client data - defeating the premise - or
negotiated across clients, which leaks the character n-grams present in each
client's URLs. Feature hashing needs no vocabulary, so every client computes
identical features from a shared seed with zero information exchange.

Cost: hash collisions, and no inverse mapping from feature index back to n-gram,
which makes error analysis harder. Use `kind="tfidf"` for interpretability work
on the CENTRALISED baseline only, and never for a federated run.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer


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
    if kind == "tfidf":
        return TfidfVectorizer(
            max_features=n_features,
            ngram_range=ngram_range,
            analyzer=analyzer,
            lowercase=lowercase,
            **kwargs,
        )
    raise ValueError(f"unknown vectorizer kind {kind!r}")

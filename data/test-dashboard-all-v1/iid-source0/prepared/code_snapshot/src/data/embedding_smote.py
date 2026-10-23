"""Lazy standard SMOTE in continuous, client-local encoder feature space.

Only original local training embeddings are accepted. No tokenizer/character IDs
or holdouts enter neighbors, interpolation or balancing targets. Synthetic vectors
are constructed per item, so a balanced corpus need not be materialized in RAM.
"""
import hashlib
import numpy as np
import torch
from sklearn import config_context
from sklearn.neighbors import NearestNeighbors

from src.data.balancing import training_labels, ratio, selection_report


class EmbeddingTrainingSet:
    split = "train"

    def __init__(self, features, labels, *, provenance, split="train"):
        self.labels = training_labels(labels, split=split)
        if (not provenance or not isinstance(features, np.ndarray) or features.ndim != 2 or
                features.shape[0] != len(self.labels) or features.shape[1] < 1 or
                features.dtype.kind != "f"):
            raise ValueError("floating encoder training embeddings and provenance are required")
        # Chunked check avoids a corpus-sized boolean temporary allocation.
        if any(not np.isfinite(features[i:i+4096]).all() for i in range(0, len(features), 4096)):
            raise ValueError("nonfinite encoder training embeddings")
        self.features, self.provenance = features, provenance

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.features[index], dtype=np.float32, copy=True)), int(self.labels[index])


class SyntheticEmbeddingSet:
    split = "train"

    def __init__(self, original, bases, neighbors, coefficients):
        self.original, self.bases, self.neighbors, self.coefficients = original, bases, neighbors, coefficients

    def __len__(self):
        return len(self.original) + len(self.bases)

    def __getitem__(self, index):
        if index < len(self.original):
            return self.original[index]
        index -= len(self.original)
        first = np.array(self.original.features[self.bases[index]], dtype=np.float32, copy=True)
        second = self.original.features[self.neighbors[index]]
        return torch.from_numpy(first + self.coefficients[index]*(second-first)), 1


def smote_embeddings(data, cfg):
    if not isinstance(data, EmbeddingTrainingSet):
        raise TypeError("SMOTE requires an EmbeddingTrainingSet, never URL strings or token IDs")
    cfg.validate()
    if cfg.method != "smote":
        raise ValueError("expected SMOTE configuration")
    y = training_labels(data.labels, split=data.split)
    adult = np.flatnonzero(y == 1)
    non = int(np.count_nonzero(y == 0))
    r = ratio(cfg.adult_to_non_adult)
    target = (non*r.numerator + r.denominator-1)//r.denominator
    count = max(0, target-len(adult))
    k, status = cfg.k_neighbors, "applied"
    if not count:
        status = "unchanged_already_at_or_above_target"
    elif len(adult) < 2 or (len(adult) <= k and cfg.insufficient_minority == "skip"):
        status, count = "skipped_insufficient_minority", 0
    elif len(adult) <= k:
        k, status = len(adult)-1, "adjusted_k"
    bases = neighbors = np.array([], dtype=np.int64)
    coefficients = np.array([], dtype=np.float32)
    if count:
        minority = np.asarray(data.features[adult], dtype=np.float32)
        # Bound pairwise-distance workspace; feature cache itself stays on disk.
        with config_context(working_memory=32):
            nearest = NearestNeighbors(n_neighbors=k+1, n_jobs=1).fit(minority).kneighbors(minority, return_distance=False)
        nearest = np.asarray([row[row != i][:k] for i, row in enumerate(nearest)])
        rng = np.random.default_rng(cfg.seed)
        choices = rng.integers(len(adult), size=count)
        bases = adult[choices]
        neighbors = adult[nearest[choices, rng.integers(k, size=count)]]
        coefficients = rng.random(count).astype(np.float32)
    selected = SyntheticEmbeddingSet(data, bases, neighbors, coefficients)
    # Counts, not a synthetic URL/record list, are retained in the report.
    report = selection_report(y, np.concatenate([y, np.ones(count, dtype=np.int8)]), cfg,
        status=status, synthetic_examples=count, effective_k_neighbors=k if count else 0,
        requested_k_neighbors=cfg.k_neighbors, backend="standard_smote_encoder_embeddings_v1",
        feature_dimension=int(data.features.shape[1]), feature_scaling="none",
        training_provenance=data.provenance, generated_urls=False,
        plan_sha256=hashlib.sha256(bases.tobytes()+neighbors.tobytes()+coefficients.tobytes()).hexdigest())
    return selected, report

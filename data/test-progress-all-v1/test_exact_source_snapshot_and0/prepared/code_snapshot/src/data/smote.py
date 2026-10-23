"""Standard minority-neighbor interpolation on engineered numerical TRAIN only.

Uses the standard SMOTE rule x_new=x_i+u*(x_neighbor-x_i), u~Uniform[0,1).
Synthetic vectors are numerical examples, never synthetic URLs/token sequences.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.data.balancing import ratio, selection_report
from src.data.numerical_features import NumericalTrainingMatrix


def smote_training(data, config):
    config.validate()
    if config.method != "smote":
        raise ValueError("SMOTE configuration required")
    if not isinstance(data, NumericalTrainingMatrix):
        raise TypeError("SMOTE accepts only a NumericalTrainingMatrix from local training features")
    data.validate()
    X, y = data.X, data.y
    counts = np.bincount(y, minlength=2)
    r = ratio(config.adult_to_non_adult)
    target = (int(counts[0])*r.numerator + r.denominator-1)//r.denominator
    n_new = max(0, target-int(counts[1]))
    k, status = config.k_neighbors, "applied"
    if not n_new:
        status = "unchanged_already_at_or_above_target"
    elif counts[1] < 2:
        status = "skipped_insufficient_minority"
        n_new = 0
    elif counts[1] <= k:
        if config.insufficient_minority == "skip":
            status, n_new = "skipped_insufficient_minority", 0
        else:
            k, status = int(counts[1])-1, "adjusted_k"
    if n_new:
        minority = X[y == 1]
        neighbors = NearestNeighbors(n_neighbors=k+1, n_jobs=1).fit(minority).kneighbors(minority, return_distance=False)
        # Explicitly exclude self even when duplicate vectors tie at distance zero.
        neighbors = np.asarray([row[row != i][:k] for i,row in enumerate(neighbors)])
        rng = np.random.default_rng(config.seed)
        base = rng.integers(len(minority), size=n_new)
        neighbor = neighbors[base, rng.integers(k, size=n_new)]
        u = rng.random((n_new,1))
        synthetic = minority[base] + u * (minority[neighbor] - minority[base])
        X_new = np.vstack([X, synthetic])
        y_new = np.concatenate([y, np.ones(n_new, dtype=np.int8)])
    else:
        X_new, y_new = X, y
    report = selection_report(y, y_new, config, status=status, synthetic_examples=n_new,
        requested_k_neighbors=config.k_neighbors, effective_k_neighbors=k if n_new else None,
        backend="standard_smote_numpy_sklearn_v1", feature_schema=data.schema,
        scaler_fingerprint=data.scaler_fingerprint, generated_urls=False)
    return X_new, y_new, report

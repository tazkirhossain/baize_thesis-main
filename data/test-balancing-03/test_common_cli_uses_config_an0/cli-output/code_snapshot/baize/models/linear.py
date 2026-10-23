"""Logistic regression with an explicit parameter-vector interface.

Written directly rather than wrapping sklearn because federated aggregation
needs three things sklearn does not expose cleanly together: get/set of a flat
parameter vector, local training that resumes from arbitrary given parameters,
and a fixed parameter layout that is identical across clients.

The model is intentionally small. Browser deployability is a design constraint
on the whole project (RESEARCH.md §4.4), and a linear model over hashed
character n-grams is the honest first baseline. Escalate only on measured
evidence.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    """Numerically stable logistic function."""
    arr = np.asarray(z, dtype=np.float64)
    out = np.empty_like(arr)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    ez = np.exp(arr[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class LogisticSGD:
    """Binary logistic regression trained by mini-batch SGD with L2 penalty."""

    def __init__(self, n_features: int, l2: float = 1e-5, seed: int = 0) -> None:
        self.n_features = int(n_features)
        self.l2 = float(l2)
        self.w = np.zeros(self.n_features, dtype=np.float64)
        self.b = 0.0
        self._rng = np.random.default_rng(seed)

    # -- parameter vector interface (what the federated layer talks to) ------ #

    @property
    def n_params(self) -> int:
        return self.n_features + 1

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.w, np.array([self.b])])

    def set_params(self, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=np.float64)
        if vec.shape != (self.n_params,):
            raise ValueError(f"expected parameter vector of shape ({self.n_params},), got {vec.shape}")
        if not np.all(np.isfinite(vec)):
            raise ValueError("refusing to set non-finite parameters (NaN/Inf) - master spec §64")
        self.w = vec[:-1].copy()
        self.b = float(vec[-1])

    # -- training / inference ----------------------------------------------- #

    def decision_function(self, X) -> np.ndarray:
        return np.asarray(X @ self.w).ravel() + self.b

    def predict_proba(self, X) -> np.ndarray:
        return sigmoid(self.decision_function(X))

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int8)

    def fit(
        self,
        X,
        y,
        epochs: int = 1,
        batch_size: int = 64,
        lr: float = 0.5,
        rng: np.random.Generator | None = None,
        indices: np.ndarray | None = None,
        compute_loss: bool = True,
    ) -> dict:
        """Train in place from the current parameters. Returns a training record.

        `indices` restricts training to a subset of rows of X WITHOUT
        materialising a slice. At corpus scale a 1M-row sparse slice costs
        hundreds of MB, and holding one per split regime is what exhausts memory;
        batching directly out of the shared matrix avoids the copy entirely.
        """
        rng = rng or self._rng
        y = np.asarray(y, dtype=np.float64)
        rows_all = (
            np.arange(X.shape[0]) if indices is None else np.asarray(indices, dtype=np.int64)
        )
        n = len(rows_all)
        if n == 0:
            return {"n_samples": 0, "n_steps": 0, "final_loss": float("nan")}
        if len(y) != n:
            raise ValueError(f"y has {len(y)} labels for {n} selected rows")

        batch_size = max(1, min(batch_size, n))
        steps = 0

        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                sel = order[start : start + batch_size]
                Xb = X[rows_all[sel]]
                yb = y[sel]
                p = sigmoid(np.asarray(Xb @ self.w).ravel() + self.b)
                err = p - yb
                m = len(sel)
                if sp.issparse(Xb):
                    grad_w = np.asarray(Xb.T @ err).ravel() / m
                else:
                    grad_w = (Xb.T @ err) / m
                grad_w += self.l2 * self.w
                self.w -= lr * grad_w
                self.b -= lr * (err.sum() / m)
                steps += 1

        if not compute_loss:
            return {"n_samples": int(n), "n_steps": steps, "final_loss": float("nan")}

        p = self.predict_proba_chunked(X, rows_all)
        eps = 1e-12
        loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
        return {"n_samples": int(n), "n_steps": steps, "final_loss": loss}

    def predict_proba_chunked(self, X, indices=None, chunk: int = 200_000) -> np.ndarray:
        """Probabilities for selected rows, evaluated in chunks to bound memory."""
        rows = np.arange(X.shape[0]) if indices is None else np.asarray(indices, dtype=np.int64)
        out = np.empty(len(rows), dtype=np.float64)
        for start in range(0, len(rows), chunk):
            sl = rows[start : start + chunk]
            out[start : start + len(sl)] = self.predict_proba(X[sl])
        return out


def init_params(n_features: int) -> np.ndarray:
    """Zero initialisation.

    Deliberate: for a convex linear model this removes initialisation as a
    source of cross-seed variance, so the seed variance measured in Phase 2
    (sigma_c, DECISIONS.md D-008) reflects data ordering and splitting rather
    than a random starting point.
    """
    return np.zeros(n_features + 1, dtype=np.float64)

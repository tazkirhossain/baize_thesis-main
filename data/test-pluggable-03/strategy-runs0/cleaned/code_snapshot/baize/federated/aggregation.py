"""Aggregation strategies for the federated server.

All aggregators operate on UPDATE DELTAS (client_params - global_params), not on
raw parameter vectors. This matters for the robust methods: deltas from honest
clients are centred near a common direction, whereas raw parameter vectors are
dominated by the shared global state, which compresses the differences the
median and trimmed mean are supposed to act on.

Every aggregator declares its threat model and failure mode in ASSUMPTIONS.
Nothing here may be described as "robust" in prose without citing the
corresponding entry (THREAT_MODEL.md §7).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ASSUMPTIONS: dict[str, dict[str, str]] = {
    "fedavg": {
        "defends_against": "nothing - this is the non-robust baseline",
        "assumes": "all clients honest and following the protocol",
        "fails_when": "any client sends an arbitrary or scaled update; a single "
        "large update can dominate the weighted mean",
    },
    "coordinate_median": {
        "defends_against": "THREAT_MODEL.md dimension B, untargeted update poisoning",
        "assumes": "fewer than 50% of clients corrupted per coordinate; honest "
        "updates unimodal around a common direction",
        "fails_when": "honest updates are multimodal, which is exactly what "
        "non-IID partitioning produces (RQ5/H5); or majority corrupted",
    },
    "trimmed_mean": {
        "defends_against": "THREAT_MODEL.md dimension B, untargeted update poisoning",
        "assumes": "corrupted fraction < beta, and beta small enough not to "
        "discard the honest tail",
        "fails_when": "beta mis-specified; legitimate minority clients occupy "
        "the trimmed tails under non-IID data. Sample-count weights are "
        "discarded (same limitation as the coordinate-wise median).",
    },
    "norm_clip_fedavg": {
        "defends_against": "scaled-update attacks and exploding updates only",
        "assumes": "malicious updates have anomalously large norm",
        "fails_when": "the attacker constrains its update norm to look ordinary; "
        "provides NO defence against a norm-bounded adversary",
    },
}


@dataclass
class AggregationReport:
    method: str
    n_clients: int
    params: dict = field(default_factory=dict)
    update_norms: list[float] = field(default_factory=list)
    clipped_clients: list[int] = field(default_factory=list)
    excluded_clients: list[int] = field(default_factory=list)
    aggregate_norm: float = 0.0
    status: str = "PASS"


def _stack(deltas: list[np.ndarray]) -> np.ndarray:
    if not deltas:
        raise ValueError("no client updates to aggregate")
    D = np.vstack([np.asarray(d, dtype=np.float64).ravel() for d in deltas])
    if not np.all(np.isfinite(D)):
        bad = np.flatnonzero(~np.isfinite(D).all(axis=1))
        raise ValueError(
            f"non-finite values in client updates from clients {bad.tolist()} "
            "- master spec §64 requires BLOCK, not silent repair"
        )
    return D


def _weights(weights, n: int) -> np.ndarray:
    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (n,):
            raise ValueError(f"expected {n} weights, got {w.shape}")
        if np.any(w < 0):
            raise ValueError("negative aggregation weights")
    total = w.sum()
    if total <= 0:
        raise ValueError("aggregation weights sum to zero")
    return w / total


def clip_by_norm(D: np.ndarray, max_norm: float) -> tuple[np.ndarray, list[int]]:
    """Scale any update whose L2 norm exceeds max_norm down to max_norm."""
    norms = np.linalg.norm(D, axis=1)
    clipped = np.flatnonzero(norms > max_norm)
    out = D.copy()
    for i in clipped:
        out[i] *= max_norm / norms[i]
    return out, clipped.tolist()


def fedavg(deltas, weights=None, **kwargs) -> tuple[np.ndarray, AggregationReport]:
    """Sample-count-weighted mean. The baseline, not a defence."""
    D = _stack(deltas)
    w = _weights(weights, len(D))
    agg = (w[:, None] * D).sum(axis=0)
    rep = AggregationReport(
        method="fedavg",
        n_clients=len(D),
        update_norms=np.linalg.norm(D, axis=1).tolist(),
        aggregate_norm=float(np.linalg.norm(agg)),
    )
    return agg, rep


def coordinate_median(deltas, weights=None, **kwargs) -> tuple[np.ndarray, AggregationReport]:
    """Coordinate-wise median. Ignores weights by construction.

    Note the cost: because the median is taken per coordinate independently, the
    aggregate is not in general any client's update, and sample-count weighting
    is discarded. Under quantity skew that means a client holding 1% of the data
    counts as much as one holding 40%.
    """
    D = _stack(deltas)
    agg = np.median(D, axis=0)
    rep = AggregationReport(
        method="coordinate_median",
        n_clients=len(D),
        params={"note": "sample-count weights discarded by construction"},
        update_norms=np.linalg.norm(D, axis=1).tolist(),
        aggregate_norm=float(np.linalg.norm(agg)),
    )
    return agg, rep


def trimmed_mean(
    deltas, weights=None, beta: float = 0.2, **kwargs
) -> tuple[np.ndarray, AggregationReport]:
    """Coordinate-wise trimmed mean: drop the top and bottom beta fraction.

    beta is the fraction trimmed from EACH tail. beta=0.2 with 10 clients trims
    2 high and 2 low values per coordinate, averaging the remaining 6.

    Sample-count weights are discarded by construction — after per-coordinate
    trimming there is no well-defined client identity left to re-weight. Under
    quantity skew a 1% client counts the same as a 40% client, which is why
    quantity-skew experiments must not treat trimmed mean as a drop-in FedAvg.
    """
    D = _stack(deltas)
    n = len(D)
    if not (0.0 <= beta < 0.5):
        raise ValueError(f"beta must be in [0.0, 0.5), got {beta}")
    k = int(np.floor(beta * n))
    if 2 * k >= n:
        raise ValueError(f"beta={beta} trims {2 * k} of {n} clients; nothing would remain")
    if k == 0:
        agg = D.mean(axis=0)
    else:
        S = np.sort(D, axis=0)
        agg = S[k : n - k].mean(axis=0)
    params: dict = {"beta": beta, "trimmed_per_tail": k, "retained": n - 2 * k}
    if weights is not None:
        params["note"] = "sample-count weights discarded: coordinate-wise trim is unweighted"
    rep = AggregationReport(
        method="trimmed_mean",
        n_clients=n,
        params=params,
        update_norms=np.linalg.norm(D, axis=1).tolist(),
        aggregate_norm=float(np.linalg.norm(agg)),
    )
    return agg, rep


def norm_clip_fedavg(
    deltas, weights=None, clip_norm: float | None = None, clip_quantile: float = 0.5, **kwargs
) -> tuple[np.ndarray, AggregationReport]:
    """Clip each update to a norm bound, then weighted-average.

    If clip_norm is None the bound is set to a quantile of the observed client
    norms this round (adaptive). Adaptive clipping is convenient but note that
    the bound is then a function of the attacker's own submissions, which an
    adaptive adversary can exploit by inflating the quantile.
    """
    if clip_norm is not None and clip_norm <= 0:
        raise ValueError(f"clip_norm must be positive, got {clip_norm}")
    if not (0.0 <= clip_quantile <= 1.0):
        raise ValueError(f"clip_quantile must be in [0.0, 1.0], got {clip_quantile}")
    D = _stack(deltas)
    norms = np.linalg.norm(D, axis=1)
    bound = float(clip_norm) if clip_norm is not None else float(np.quantile(norms, clip_quantile))
    Dc, clipped = clip_by_norm(D, bound)
    w = _weights(weights, len(Dc))
    agg = (w[:, None] * Dc).sum(axis=0)
    rep = AggregationReport(
        method="norm_clip_fedavg",
        n_clients=len(D),
        params={"clip_norm": bound, "adaptive": clip_norm is None},
        update_norms=norms.tolist(),
        clipped_clients=clipped,
        aggregate_norm=float(np.linalg.norm(agg)),
    )
    return agg, rep


AGGREGATORS = {
    "fedavg": fedavg,
    "coordinate_median": coordinate_median,
    "trimmed_mean": trimmed_mean,
    "norm_clip_fedavg": norm_clip_fedavg,
}


def get_aggregator(name: str):
    if name not in AGGREGATORS:
        raise ValueError(f"unknown aggregator {name!r}; have {sorted(AGGREGATORS)}")
    return AGGREGATORS[name]

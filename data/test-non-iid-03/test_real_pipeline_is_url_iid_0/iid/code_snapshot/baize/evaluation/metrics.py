"""Classification metrics.

Primary metric is PR-AUC (DECISIONS.md D-003). It is reported first and is the
only metric permitted to drive a hypothesis decision. Everything else is
descriptive.

PR-AUC depends on class balance, so it is not comparable across datasets with
different positive rates. `positive_rate` is therefore returned alongside it and
must be quoted wherever PR-AUC is quoted.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


def classification_metrics(
    y_true, y_prob, threshold: float = 0.5, n_calibration_bins: int = 10
) -> dict[str, float]:
    """Full metric block for one model on one dataset."""
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(y_prob, dtype=np.float64)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true {y.shape} vs y_prob {p.shape}")
    if len(y) == 0:
        raise ValueError("cannot compute metrics on an empty set")
    if not np.all(np.isfinite(p)):
        raise ValueError("non-finite probabilities - master spec §64 requires BLOCK")

    single_class = len(np.unique(y)) < 2
    yhat = (p >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    out = {
        # primary
        "pr_auc": float("nan") if single_class else float(average_precision_score(y, p)),
        # descriptive
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y, p)),
        "accuracy": float((yhat == y).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "threshold": float(threshold),
        # calibration
        "brier": float(brier_score_loss(y, p)) if not single_class else float(np.mean((p - y) ** 2)),
        "ece": float(expected_calibration_error(y, p, n_bins=n_calibration_bins)),
    }
    return out


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Equal-width-bin ECE.

    Equal-width binning is the common convention but is sensitive to bin count
    and leaves sparse bins in the middle of the range when predictions are
    concentrated near 0 and 1, which is typical here. Treat ECE as indicative;
    read the reliability curve rather than the scalar.
    """
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        ece += (sel.mean()) * abs(y[sel].mean() - p[sel].mean())
    return float(ece)


def reliability_curve(y_true, y_prob, n_bins: int = 10) -> dict[str, list[float]]:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    conf, acc, frac = [], [], []
    for b in range(n_bins):
        sel = idx == b
        conf.append(float(p[sel].mean()) if sel.any() else float("nan"))
        acc.append(float(y[sel].mean()) if sel.any() else float("nan"))
        frac.append(float(sel.mean()))
    return {"mean_predicted": conf, "observed_frequency": acc, "bin_fraction": frac}


def precision_at_base_rate(recall: float, fpr: float, base_rate: float) -> float:
    """Precision the model would achieve at an ASSUMED deployment base rate.

    RESEARCH.md §4.9. Test-set class balance is not deployment class balance. If
    adult sites are a small fraction of real traffic, a benign-looking FPR
    produces many false blocks in absolute terms.

    This is a PROJECTION from an assumed prior, not a measurement. Report the
    assumed base rate every time this number appears, and never present it as an
    observed result.
    """
    if not 0.0 <= base_rate <= 1.0:
        raise ValueError(f"base_rate must be in [0,1], got {base_rate}")
    tp = recall * base_rate
    fp = fpr * (1.0 - base_rate)
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def selective_metrics(y_true, y_prob, lower: float, upper: float) -> dict[str, float]:
    """Abstention: predict only outside the uncertainty band [lower, upper].

    Returns coverage (fraction answered) and metrics on the answered subset.
    Master spec §16 requires abstention to be EVALUATED, not assumed to help:
    accuracy on the retained subset always rises trivially as coverage falls, so
    the meaningful comparison is the coverage/risk curve, not a single point.
    """
    if lower > upper:
        raise ValueError(f"lower must be <= upper, got lower={lower}, upper={upper}")
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(y_prob, dtype=np.float64)
    answered = (p < lower) | (p > upper)
    coverage = float(answered.mean())
    out = {"coverage": coverage, "lower": float(lower), "upper": float(upper),
           "n_answered": int(answered.sum()), "n_abstained": int((~answered).sum())}
    if answered.sum() == 0:
        out.update({"selective_error": float("nan"), "selective_accuracy": float("nan")})
        return out
    yhat = (p[answered] >= 0.5).astype(np.int8)
    err = float((yhat != y[answered]).mean())
    out.update({"selective_error": err, "selective_accuracy": 1.0 - err})
    return out


def coverage_risk_curve(y_true, y_prob, n_points: int = 20) -> list[dict[str, float]]:
    """Sweep a symmetric uncertainty band around 0.5 and record coverage vs risk."""
    curve = []
    for half in np.linspace(0.0, 0.5, n_points, endpoint=False):
        curve.append(selective_metrics(y_true, y_prob, 0.5 - half, 0.5 + half))
    return curve


def format_metrics(m: dict, title: str = "") -> str:
    L = [f"{title}" if title else "", "-" * 52]
    L.append(f"  PR-AUC (primary)  {m['pr_auc']:.4f}   [pos_rate {m['positive_rate']:.4f}]")
    L.append(f"  ROC-AUC           {m['roc_auc']:.4f}")
    L.append(f"  F1                {m['f1']:.4f}")
    L.append(f"  Precision/Recall  {m['precision']:.4f} / {m['recall']:.4f}")
    L.append(f"  FPR / FNR         {m['fpr']:.4f} / {m['fnr']:.4f}")
    L.append(f"  Accuracy          {m['accuracy']:.4f}")
    L.append(f"  Brier / ECE       {m['brier']:.4f} / {m['ece']:.4f}")
    L.append(f"  Confusion         tp={m['tp']} fp={m['fp']} tn={m['tn']} fn={m['fn']}  (n={m['n']})")
    return "\n".join(L)

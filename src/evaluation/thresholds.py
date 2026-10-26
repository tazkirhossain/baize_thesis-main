"""Fixed-grid threshold selection from coarse training-selection summaries.

Rows are true labels (0=Non-adult, 1=Adult); columns are probability intervals.
Only aggregated counts are encoded. No example scores, URLs or IDs are retained.
This module accepts no validation/test inputs for selecting a threshold.
"""
import json
from numbers import Integral, Real

import numpy as np

from src.evaluation.classification import confusion_metrics


MAX_COUNT = int(np.iinfo(np.int64).max)
MAX_PAYLOAD_BYTES = 64 * 1024


def _bins(bins):
    if type(bins) is not int or not 21 <= bins <= 1001:
        raise ValueError("threshold bins must be an integer between 21 and 1001")
    return bins


def validate_threshold_selection(cfg):
    """Validate the optional public contract; never infer settings from data."""
    settings = cfg.get("threshold_selection")
    if settings is None:
        return None
    if (not isinstance(settings, dict)
            or set(settings) != {"source", "metric", "bins", "tie_break"}
            or settings["source"] != "training_selection"
            or settings["metric"] != "adult_f1"
            or settings["tie_break"] != "highest_threshold"):
        raise ValueError("threshold_selection requires training_selection, adult_f1, bins and highest_threshold")
    _bins(settings["bins"])
    if (not isinstance(cfg.get("model_selection"), dict)
            or cfg["model_selection"].get("source") != "training_selection"):
        raise ValueError("threshold_selection requires a training-only model_selection split")
    return dict(settings)


def histogram_counts(probabilities, labels, bins):
    """Return class-by-score counts, preserving direct >= grid comparisons.

    The mathematical bin is floor(p*(bins-1)). Comparing against the fixed grid
    avoids multiplication rounding placing an exact grid probability below its
    threshold (or placing nextafter(threshold, -inf) above it).
    """
    _bins(bins)
    try:
        scores = np.asarray(probabilities, dtype=np.float64)
        y = np.asarray(labels)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid probability/label arrays") from exc
    if (scores.ndim != 1 or y.shape != scores.shape
            or (y.size and y.dtype.kind not in "iu") or not np.isin(y, (0, 1)).all()
            or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()):
        raise ValueError("finite probabilities in [0,1] and binary integer labels are required")
    grid = np.arange(bins, dtype=np.float64) / (bins - 1)
    columns = np.searchsorted(grid, scores, side="right") - 1
    cells = y.astype(np.int64) * bins + columns
    return np.bincount(cells, minlength=2 * bins).reshape(2, bins).astype(np.int64)


def encode_probability_histogram(probabilities, labels, bins):
    return json.dumps(histogram_counts(probabilities, labels, bins).tolist(), separators=(",", ":"))


def _checked_histogram(histogram, bins=None):
    try:
        values = np.asarray(histogram, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid probability histogram") from exc
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError("probability histogram must have two class rows")
    inferred = _bins(values.shape[1])
    if bins is not None and inferred != _bins(bins):
        raise ValueError("probability histogram bin count mismatch")
    counts = list(values.flat)
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
           or not 0 <= value <= MAX_COUNT for value in counts):
        raise ValueError("histogram counts must be nonnegative bounded integers")
    if sum(int(value) for value in counts) > MAX_COUNT:
        raise ValueError("probability histogram count overflow")
    return np.asarray(values, dtype=np.int64)


def parse_histogram(payload, bins, expected_counts=None):
    """Decode strict JSON counts and optionally reconcile class supports [0,1]."""
    _bins(bins)
    if not isinstance(payload, str) or len(payload.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise ValueError("invalid or oversized probability histogram payload")
    try:
        decoded = json.loads(payload)
    except (ValueError, RecursionError) as exc:
        raise ValueError("invalid probability histogram JSON") from exc
    if (type(decoded) is not list or len(decoded) != 2
            or any(type(row) is not list or len(row) != bins for row in decoded)
            or any(type(value) is not int for row in decoded for value in row)):
        raise ValueError("histogram JSON must contain two rows of native integer counts")
    histogram = _checked_histogram(decoded, bins)
    if expected_counts is not None:
        try:
            supports = list(expected_counts)
        except TypeError as exc:
            raise ValueError("expected class counts must have two integer supports") from exc
        if (len(supports) != 2 or any(isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral) or not 0 <= value <= MAX_COUNT for value in supports)):
            raise ValueError("expected class counts must have two integer supports")
        actual = [sum(int(value) for value in row) for row in histogram]
        if actual != supports:
            raise ValueError("probability histogram class counts do not reconcile")
    return histogram


def pool_histograms(histograms):
    """Add summaries with Python integers before checked int64 conversion."""
    items = list(histograms)
    if not items:
        raise ValueError("at least one probability histogram is required")
    checked = [_checked_histogram(item) for item in items]
    bins = checked[0].shape[1]
    if any(item.shape[1] != bins for item in checked):
        raise ValueError("probability histograms must use the same grid")
    pooled = np.zeros((2, bins), dtype=object)
    for item in checked:
        pooled += item.astype(object)
    return _checked_histogram(pooled, bins)


def metrics_at_threshold(histogram, threshold):
    """Compute counts at an exact grid threshold; endpoints are supported."""
    values = _checked_histogram(histogram)
    bins = values.shape[1]
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real) or not np.isfinite(threshold):
        raise ValueError("threshold must be a finite grid value")
    index = round(float(threshold) * (bins - 1))
    if not 0 <= index < bins or float(threshold) != index / (bins - 1):
        raise ValueError("threshold must belong to the fixed probability grid")
    tn = sum(int(value) for value in values[0, :index])
    fp = sum(int(value) for value in values[0, index:])
    fn = sum(int(value) for value in values[1, :index])
    tp = sum(int(value) for value in values[1, index:])
    return dict(confusion_metrics([[tn, fp], [fn, tp]]), threshold=float(threshold))


def select_threshold(pooled_histogram):
    """Maximize pooled training-selection Adult F1, highest threshold on ties.

    Scores are compared as integer fractions, avoiding floating-point tie drift.
    Validation/test counts cannot be supplied through a separate selection input.
    """
    values = _checked_histogram(pooled_histogram)
    if any(sum(int(value) for value in row) == 0 for row in values):
        raise ValueError("pooled training-selection histogram requires both classes")
    bins = values.shape[1]
    adult_total = sum(int(value) for value in values[1])
    fp = sum(int(value) for value in values[0, 1:])
    tp = sum(int(value) for value in values[1, 1:])
    best_index, best_numerator, best_denominator = 1, -1, 1
    for index in range(1, bins - 1):
        numerator = 2 * tp
        denominator = adult_total + tp + fp
        # Iteration is ascending, so equality deliberately retains the higher t.
        if numerator * best_denominator >= best_numerator * denominator:
            best_index, best_numerator, best_denominator = index, numerator, denominator
        fp -= int(values[0, index])
        tp -= int(values[1, index])
    return metrics_at_threshold(values, best_index / (bins - 1))

"""Grid arithmetic, support reconciliation and training-only threshold contract."""
import json

import numpy as np
import pytest

from src.evaluation.thresholds import (MAX_COUNT, encode_probability_histogram,
    histogram_counts, metrics_at_threshold, parse_histogram, pool_histograms,
    select_threshold, validate_threshold_selection)


def config():
    return {"model_selection": {"source": "training_selection"},
            "threshold_selection": {"source": "training_selection", "metric": "adult_f1",
                                    "bins": 201, "tie_break": "highest_threshold"}}


@pytest.mark.parametrize("bins", [21, 201, 1001])
def test_every_grid_boundary_matches_direct_predictions(bins):
    grid = np.arange(bins) / (bins - 1)
    scores = np.concatenate([grid, np.nextafter(grid[1:], -np.inf), np.nextafter(grid[:-1], np.inf)])
    labels = np.arange(len(scores)) % 2
    histogram = histogram_counts(scores, labels, bins)
    for threshold in grid:
        predicted = scores >= threshold
        actual = [[int(np.sum((labels == 0) & ~predicted)), int(np.sum((labels == 0) & predicted))],
                  [int(np.sum((labels == 1) & ~predicted)), int(np.sum((labels == 1) & predicted))]]
        assert metrics_at_threshold(histogram, threshold)["confusion_matrix"] == actual


def test_payload_roundtrip_supports_and_empty_client_pooling():
    scores, labels = [0, .1, .5, .9, 1], [0, 0, 1, 1, 1]
    payload = encode_probability_histogram(scores, labels, 201)
    histogram = parse_histogram(payload, 201, expected_counts=[2, 3])
    empty = histogram_counts([], [], 201)
    np.testing.assert_array_equal(pool_histograms([empty, histogram]), histogram)
    assert histogram.dtype == np.int64 and histogram.sum() == 5
    with pytest.raises(ValueError, match="reconcile"):
        parse_histogram(payload, 201, expected_counts=[3, 2])
    with pytest.raises(ValueError, match="both classes"):
        select_threshold(empty)


def test_best_f1_is_exact_and_highest_threshold_wins_ties():
    histogram = histogram_counts([.1, .2, .8, .9], [0, 0, 1, 1], 201)
    best = select_threshold(histogram)
    assert best["threshold"] == .8 and best["adult_f1"] == 1
    # With no positives detected at any interior threshold, all F1 values tie0.
    all_zero_f1 = histogram_counts([1, 0], [0, 1], 201)
    assert select_threshold(all_zero_f1)["threshold"] == .995
    candidates = [metrics_at_threshold(histogram, index/200) for index in range(1, 200)]
    assert best == max(candidates, key=lambda row: (row["adult_f1"], row["threshold"]))


@pytest.mark.parametrize("change", [{"source": "val"}, {"source": "test"}, {"metric": "accuracy"},
    {"bins": 20}, {"bins": 1002}, {"bins": True}, {"bins": 201.0},
    {"tie_break": "lowest_threshold"}, {"unexpected": 1}])
def test_configuration_rejects_holdout_selection_and_ambiguous_grids(change):
    cfg = config()
    cfg["threshold_selection"].update(change)
    with pytest.raises(ValueError):
        validate_threshold_selection(cfg)


def test_selection_contract_requires_training_selection_and_returns_copy():
    assert validate_threshold_selection({}) is None
    cfg = config()
    assert validate_threshold_selection(cfg) == cfg["threshold_selection"]
    assert validate_threshold_selection(cfg) is not cfg["threshold_selection"]
    cfg.pop("model_selection")
    with pytest.raises(ValueError, match="training-only"):
        validate_threshold_selection(cfg)


@pytest.mark.parametrize("payload", ["null", "{}", "not-json", "[]", "[[]]", "[[0],[0]]", " " * 65537],
                         ids=["null", "object", "invalid-json", "empty", "one-row", "wrong-width", "oversized"])
def test_invalid_or_oversized_payloads_fail(payload):
    with pytest.raises(ValueError):
        parse_histogram(payload, 201)


@pytest.mark.parametrize("value", [True, -1, .5, float("nan"), MAX_COUNT+1])
def test_noninteger_negative_nonfinite_or_overflow_counts_fail(value):
    histogram = [[0]*21 for _ in range(2)]
    histogram[0][0] = value
    with pytest.raises(ValueError):
        parse_histogram(json.dumps(histogram), 21)


def test_pooling_overflow_grid_mismatch_and_missing_classes_fail():
    huge = np.zeros((2, 21), dtype=np.int64)
    huge[0, 0] = MAX_COUNT
    extra = np.zeros((2, 21), dtype=np.int64)
    extra[1, 0] = 1
    with pytest.raises(ValueError, match="overflow"):
        pool_histograms([huge, extra])
    with pytest.raises(ValueError, match="same grid"):
        pool_histograms([extra, np.zeros((2, 201), dtype=np.int64)])
    with pytest.raises(ValueError, match="at least one"):
        pool_histograms([])
    with pytest.raises(ValueError, match="both classes"):
        select_threshold(extra)


@pytest.mark.parametrize("scores,labels", [([-.1], [0]), ([1.1], [1]), ([float("nan")], [1]),
    ([.5], [2]), ([.5], [True]), ([.5], [.0]), ([[.5]], [[0]]), ([.5], [])])
def test_invalid_probabilities_and_labels_fail(scores, labels):
    with pytest.raises(ValueError):
        histogram_counts(scores, labels, 201)


@pytest.mark.parametrize("threshold", [-.1, 1.1, .1234, float("nan"), True])
def test_threshold_must_be_an_exact_grid_value(threshold):
    histogram = histogram_counts([.1, .9], [0, 1], 201)
    with pytest.raises(ValueError):
        metrics_at_threshold(histogram, threshold)

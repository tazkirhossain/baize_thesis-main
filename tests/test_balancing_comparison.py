import pytest
from src.evaluation.balancing_comparison import pooled_confusion_metrics, summarize_balancing


def test_pooled_metrics_derive_from_confusion_counts_not_mean_client_f1():
    m = pooled_confusion_metrics([[90,10],[5,15]])
    assert m["adult_precision"] == 15/25
    assert m["adult_recall"] == 15/20
    assert m["adult_f1"] == 30/45
    assert m["balanced_accuracy"] == (15/20+90/100)/2
    assert m["accuracy"] == 105/120


def test_comparison_rejects_neural_result_family():
    with pytest.raises(ValueError):
        summarize_balancing({"model_family":"deberta_v3_base","clients":[]})


def test_empty_or_invalid_confusion_matrix_is_rejected():
    for matrix in ([[0,0],[0,0]], [[-1,1],[1,1]], [[1,2,3]]):
        with pytest.raises(ValueError):
            pooled_confusion_metrics(matrix)

"""Train-only resampling, ratio semantics, deterministic selections and synthetic guards."""
from dataclasses import replace

import numpy as np
import pytest

from src.data.protection import mark_protected
from src.client.data import LocalSplit
from src.data.balancing import (BalanceConfig, class_counts, ensemble_training_indices,
    sample_training_indices)
from src.data.numerical_features import FEATURE_NAMES, NumericalTrainingMatrix, URLNumericTransformer
from src.data.smote import smote_training
from src.evaluation.balancing_metrics import balancing_metrics


@pytest.mark.parametrize("method", ["undersampling","oversampling"])
@pytest.mark.parametrize("ratio", ["1:1","1:5"])
def test_random_methods_exact_counts_preserve_minority_and_reproduce(method,ratio):
    labels = np.array([1]*10+[0]*100)
    original = labels.copy()
    cfg = BalanceConfig(method=method,adult_to_non_adult=ratio,seed=9)
    selected = sample_training_indices(labels,cfg,split="train")
    same = sample_training_indices(labels,cfg,split="train")
    changed = sample_training_indices(labels,replace(cfg,seed=10),split="train")
    np.testing.assert_array_equal(selected.indices,same.indices)
    assert not np.array_equal(selected.indices,changed.indices)
    assert set(range(10)) <= set(selected.indices)
    if method == "undersampling":
        expected = 10 if ratio == "1:1" else 50
        assert class_counts(labels[selected.indices]) == {"Adult":10,"Non-adult":expected}
        assert len(np.unique(selected.indices)) == len(selected.indices)
    else:
        expected = 100 if ratio == "1:1" else 20
        assert class_counts(labels[selected.indices]) == {"Adult":expected,"Non-adult":100}
        assert set(selected.indices) == set(range(110))
        assert selected.report["duplicated_examples"] == expected-10
    np.testing.assert_array_equal(labels,original)
    assert not selected.indices.flags.writeable


@pytest.mark.parametrize("method", ["none","undersampling","oversampling","ensemble_undersampling"])
@pytest.mark.parametrize("split", ["val","test"])
def test_every_index_method_refuses_evaluation_labels(method,split):
    cfg = BalanceConfig(method=method)
    function = ensemble_training_indices if method == "ensemble_undersampling" else sample_training_indices
    with pytest.raises(ValueError,match="training"):
        function(np.array([0,1]),cfg,split=split)
    with pytest.raises(ValueError,match="training"):
        function(mark_protected([0,1]),cfg,split="train")


def numeric_data(n_adult=3,n_non=30):
    X = np.arange((n_adult+n_non)*len(FEATURE_NAMES),dtype=float).reshape(-1,len(FEATURE_NAMES))
    return NumericalTrainingMatrix(X,np.array([1]*n_adult+[0]*n_non),"fitted-training-fingerprint")


@pytest.mark.parametrize("ratio", ["1:1","1:5"])
def test_smote_generates_only_floating_numerical_training_vectors(ratio):
    data = numeric_data()
    cfg = BalanceConfig(method="smote",adult_to_non_adult=ratio,k_neighbors=2)
    X,y,report = smote_training(data,cfg)
    X2,y2,_ = smote_training(data,cfg)
    expected = 30 if ratio == "1:1" else 6
    assert class_counts(y) == {"Adult":expected,"Non-adult":30}
    assert report["synthetic_examples"] == expected-3
    assert report["duplicated_examples"] == 0
    np.testing.assert_array_equal(X[:len(data.y)],data.X)
    np.testing.assert_array_equal(X,X2)
    np.testing.assert_array_equal(y,y2)
    synthetic = X[len(data.y):]
    assert np.all(synthetic >= data.X[:3].min(axis=0))
    assert np.all(synthetic <= data.X[:3].max(axis=0))
    assert report["generated_urls"] is False


def test_smote_adjusts_neighbors_or_skips_explicitly():
    cfg = BalanceConfig(method="smote",adult_to_non_adult="1:1",k_neighbors=5)
    _,_,adjusted = smote_training(numeric_data(3),cfg)
    assert adjusted["status"] == "adjusted_k" and adjusted["effective_k_neighbors"] == 2
    X,y,skipped = smote_training(numeric_data(1),cfg)
    assert skipped["status"] == "skipped_insufficient_minority" and len(y) == 31
    _,_,skipped = smote_training(numeric_data(3),replace(cfg,insufficient_minority="skip"))
    assert skipped["synthetic_examples"] == 0 and skipped["status"].startswith("skipped")


def test_smote_refuses_raw_urls_tokens_and_holdouts():
    cfg = BalanceConfig(method="smote")
    for raw in (["https://site.com"],np.array([[101,25,102]])):
        with pytest.raises(TypeError):
            smote_training(raw,cfg)
    data = numeric_data()
    for invalid in (replace(data,X=data.X.astype(int)),replace(data,schema="deberta_token_ids"),
                    replace(data,split="test"),replace(data,y=mark_protected(data.y))):
        with pytest.raises(ValueError):
            smote_training(invalid,cfg)


def test_scaler_fits_training_only_and_does_not_refit_on_holdouts():
    train = LocalSplit("train",("https://a.com/a","https://b.com/bb"),np.array([1,0]),np.array([1,2]),"a")
    val = LocalSplit("val",("https://very-long-domain-name.com/huge/path",),mark_protected([1]),np.array([3]),"b")
    transformer = URLNumericTransformer()
    data = transformer.fit_training(train)
    original = transformer.scaler_.mean_.copy()
    transformer.transform(val)
    np.testing.assert_array_equal(original,transformer.scaler_.mean_)
    assert transformer.scaler_.n_samples_seen_ == 2
    assert data.X.dtype.kind == "f"
    with pytest.raises(ValueError,match="training"):
        URLNumericTransformer().fit_training(val)


def test_ensemble_subsets_retain_all_adult_and_use_independent_seeds():
    labels = np.array([1]*10+[0]*100)
    cfg = BalanceConfig(method="ensemble_undersampling",adult_to_non_adult="1:1",ensemble_members=3)
    selections = ensemble_training_indices(labels,cfg,split="train")
    assert len({s.report["member_seed"] for s in selections}) == 3
    assert len({s.report["selection_sha256"] for s in selections}) == 3
    for selection in selections:
        assert set(range(10)) <= set(selection.indices)
        assert len(selection.indices) == 20


def test_metrics_include_both_class_macro_and_fixed_confusion_matrix():
    metrics = balancing_metrics([0,0,1,1],[0.1,0.9,0.8,0.2])
    assert metrics["confusion_matrix"] == [[1,1],[1,1]]
    for key in ("adult_precision","adult_recall","adult_f1","macro_f1","balanced_accuracy","accuracy"):
        assert metrics[key] == 0.5


def test_unreachable_targets_are_logged_without_discarding_adult():
    labels = np.array([1]*10+[0]*20)
    for method in ("undersampling","oversampling"):
        selected = sample_training_indices(labels,BalanceConfig(method=method,adult_to_non_adult="1:5"),split="train")
        assert selected.report["status"].startswith("unchanged")
        assert set(selected.indices) == set(range(30))


def test_oversampling_rounds_up_and_selection_view_is_model_agnostic():
    labels = np.array([1]*2+[0]*13)
    selected = sample_training_indices(labels,BalanceConfig(method="oversampling",adult_to_non_adult="1:5"),split="train")
    assert selected.report["after"]["Adult"] == 3
    dataset = [f"original-url-{i}" for i in range(15)]
    view = selected.apply(dataset)
    assert len(view) == 16 and view[0] == dataset[selected.indices[0]]
    with pytest.raises(ValueError):
        selected.apply(dataset[:3])


def test_equal_class_default_and_all_presets():
    import json
    from pathlib import Path
    assert BalanceConfig().adult_to_non_adult == "1:1"
    presets = Path(__file__).resolve().parents[1] / "config/balancing"
    for method in ("undersampling", "oversampling", "smote", "ensemble_undersampling"):
        cfg = BalanceConfig(**json.loads((presets / f"{method}.json").read_text())["balancing"])
        cfg.validate()
        assert cfg.adult_to_non_adult == "1:1"
        labels = np.array([1]*3 + [0]*30)
        if method == "smote":
            _, _, report = smote_training(numeric_data(), cfg)
            reports = [report]
        elif method == "ensemble_undersampling":
            reports = [item.report for item in ensemble_training_indices(labels, cfg, split="train")]
        else:
            reports = [sample_training_indices(labels, cfg, split="train").report]
        for report in reports:
            assert report["after"]["Adult"] == report["after"]["Non-adult"]

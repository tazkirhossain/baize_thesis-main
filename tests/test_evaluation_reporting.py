"""Metric arithmetic, historical-source checks and research report exports."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score

from src.evaluation.classification import confusion_metrics
from src.evaluation.reporting import (checked_metrics, comparison_coverage, federated_rows,
    numerical_rows, report_markdown, verify_comparability, checked_client_training)

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = json.loads((ROOT / "config/reporting/comparison.json").read_text())


@pytest.mark.parametrize("matrix", [[[90, 10], [5, 15]], [[1000, 0], [24, 0]], [[0, 10], [20, 0]], [[10, 0], [0, 3]]])
def test_both_class_metrics_match_independent_sklearn(matrix):
    tn, fp, fn, tp = np.asarray(matrix).ravel()
    labels = [0]*(tn+fp)+[1]*(fn+tp)
    predicted = [0]*tn+[1]*fp+[0]*fn+[1]*tp
    independent = classification_report(labels, predicted, labels=[0, 1], output_dict=True, zero_division=0)
    result = confusion_metrics(matrix)
    for class_id, name in (("0", "non_adult"), ("1", "adult")):
        for suffix, key in (("precision", "precision"), ("recall", "recall"), ("f1", "f1-score")):
            assert result[f"{name}_{suffix}"] == pytest.approx(independent[class_id][key])
    assert result["macro_f1"] == pytest.approx(f1_score(labels, predicted, average="macro", zero_division=0))
    assert result["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(labels, predicted))
    assert result["adult_prevalence"] == (fn+tp)/len(labels)


def test_majority_only_accuracy_does_not_hide_zero_adult_recall():
    result = confusion_metrics([[9763, 0], [237, 0]])
    assert result["accuracy"] == 0.9763
    assert result["adult_recall"] == result["adult_f1"] == 0
    assert result["balanced_accuracy"] == 0.5
    assert result["non_adult_recall"] == 1
    with pytest.raises(ValueError, match="disagrees"):
        checked_metrics(dict(result, adult_recall=0.9))


def test_zero_division_and_invalid_confusion_counts():
    assert confusion_metrics([[5, 0], [0, 0]])["adult_f1"] == 0
    for matrix in ([[0, 0], [0, 0]], [[1.5, 1], [0, 1]], [[np.nan, 0], [1, 2]]):
        with pytest.raises(ValueError):
            confusion_metrics(matrix)


def test_historical_numerical_results_derive_negative_metrics_without_retraining():
    source = ROOT / "data/balancing-four-methods-equal-v2/summary.json"
    if not source.exists():
        pytest.skip("private completed numerical summary is not present in this checkout")
    before = source.read_bytes()
    rows, metadata, skipped = numerical_rows(source, SETTINGS)
    assert len(rows) == 24 and len(metadata) == 12 and not skipped
    assert {row["partition"] for row in rows} == {"iid", "label_skew", "category_skew"}
    assert all("non_adult_f1" in row["metrics"] for row in rows)
    assert all(row["timing"]["training_seconds_sum"] is None for row in rows)
    assert source.read_bytes() == before
    coverage = comparison_coverage(rows, SETTINGS)
    assert all(item["status"] == "not measured" for item in coverage if item["family"] == "character_cnn" and item["method"] != "smote")


@pytest.mark.parametrize('change', [{'aggregation_weight':'resampled_training'},
                                   {'optimizer':{'name':'adamw','weight_decay':0.0}}])
def test_duplicate_or_incomparable_experiments_are_rejected(change):
    row = {"scope": "federated global model", "family": "character_cnn", "data_kind": "real",
           "seed": 42, "partition": "iid", "method": "undersampling", "split": "val", "round": 1,
           "experiment_id": "a"}
    cfg = json.loads((ROOT / "config/federated/character_cnn.json").read_text())
    metadata = {"a": {"configuration": cfg, "data_contracts": {}}}
    with pytest.raises(ValueError, match="duplicate"):
        verify_comparability([row, dict(row)], metadata)
    second = dict(row, method="oversampling", experiment_id="b")
    changed = deepcopy(cfg)
    changed.update(change)
    metadata["b"] = {"configuration": changed, "data_contracts": {}}
    with pytest.raises(ValueError, match="incompatible"):
        verify_comparability([row, second], metadata)


def test_standalone_figures_and_tables_have_measured_scope(tmp_path):
    pytest.importorskip("matplotlib")
    from src.evaluation.plots import export_plots
    rows = []
    for partition in ("iid", "label_skew", "category_skew"):
        for method in SETTINGS["required_balancing_methods"]:
            for split in ("val", "test"):
                rows.append({"family": "engineered_numeric", "classifier": "sgd_logistic", "data_kind": "real", "scope": "pooled independent local numerical models",
                             "partition": partition, "method": method, "ratio": "1:1", "seed": 42, "split": split,
                             "round": None, "metrics": confusion_metrics([[90, 10], [5, 15]])})
    report = {"rows": rows, "experiments": {}, "limitations": ["Synthetic unit-test values only."],
              "coverage": comparison_coverage(rows, SETTINGS)}
    names = export_plots(report, tmp_path, ("png",))
    assert len(names) == 4 and all((tmp_path / name).stat().st_size > 1000 for name in names)
    markdown = report_markdown(report, names)
    assert "Non-adult P" in markdown and "No method is declared best" in markdown


def test_publication_pdf_and_high_resolution_png_export(tmp_path):
    pytest.importorskip("matplotlib")
    from PIL import Image
    from src.evaluation.plots import export_plots

    rows = []
    for rnd in (1, 2, 3):
        metrics = dict(confusion_metrics([[90-rnd, rnd], [4, 16]]), loss=1/rnd)
        rows.append({"experiment_id": "publication", "family": "character_cnn", "data_kind": "real",
                     "scope": "federated global model", "partition": "iid", "method": "oversampling",
                     "input_mode": "url_only", "split": "val", "round": rnd, "metrics": metrics,
                     "seed": 42, "classifier": None, "ratio": "1:1",
                     "timing": {"client_training_seconds_sum": 10+rnd},
                     "communication": {"total_bytes": 1024**2*(20+rnd)}})
    report = {"rows": rows, "experiments": {"publication": {"configuration": {"rounds": 3}}}}
    names = export_plots(report, tmp_path, ("png", "pdf"))
    assert {name.rsplit(".", 1)[1] for name in names} == {"png", "pdf"}
    assert all((tmp_path/name).stat().st_size > 1000 for name in names)
    with Image.open(tmp_path/"05_convergence_publication.png") as image:
        assert image.info["dpi"][0] == pytest.approx(300, abs=1)


def test_client_training_statistics_are_validated_and_plotted(tmp_path):
    pytest.importorskip("matplotlib")
    from src.evaluation.plots import export_plots

    cfg = {"n_clients": 5, "clients_per_round": 5, "aggregation_weight": "original_training"}
    client_rounds = []
    for rnd in (1, 2):
        clients = [{"client_id": cid, "original_train": 100+cid, "resampled_train": 190,
                    "original_adult": 5+cid, "original_non_adult": 95,
                    "used_adult": 95, "used_non_adult": 95, "train_loss": 0.5+cid/10+rnd/100,
                    "train_seconds": 2+cid/10, "aggregation_weight": 100+cid}
                   for cid in range(5)]
        report = {"round": rnd, "clients": clients}
        client_rounds.append(checked_client_training(report, cfg))
    metrics = dict(confusion_metrics([[90, 5], [2, 8]]), loss=0.4)
    rows = [{"experiment_id": "clients", "family": "character_cnn", "data_kind": "real",
             "scope": "federated global model", "partition": "label_skew", "method": "undersampling",
             "input_mode": "url_only", "split": "val", "round": rnd, "metrics": metrics,
             "seed": 42, "classifier": "character_cnn", "ratio": "1:1"}
            for rnd in (1, 2)]
    report = {"rows": rows, "experiments": {"clients": {"configuration": {"rounds": 2, "n_clients": 5},
              "client_training_rounds": client_rounds}}}
    names = export_plots(report, tmp_path, ("png",))
    assert "08_client_balance_clients.png" in names
    assert "09_client_heterogeneity_clients.png" in names
    assert all((tmp_path/name).stat().st_size > 1000 for name in names)

    damaged = {**client_rounds[0]["clients"][0], "aggregation_weight": 99}
    with pytest.raises(ValueError, match="aggregation weight"):
        checked_client_training({"round": 1, "clients": [damaged, *client_rounds[0]["clients"][1:]]}, cfg)


def test_neural_figures_include_class_bias_and_fixed_final_round(tmp_path):
    pytest.importorskip("matplotlib")
    import xml.etree.ElementTree as ET
    from src.evaluation.plots import export_plots

    rows = []
    for rnd in (1, 2):
        for split, final_matrix in (("train", [[1, 3], [2, 4]]), ("val", [[0, 4], [0, 2]])):
            metrics = dict(confusion_metrics(final_matrix if rnd == 2 else [[123, 0], [0, 1]]), loss=0.8/rnd)
            rows.append({"experiment_id": "smoke", "family": "character_cnn", "data_kind": "synthetic",
                         "scope": "federated global model", "partition": "unknown", "method": "oversampling",
                         "split": split, "round": rnd, "metrics": metrics})
    report = {"rows": rows, "experiments": {"smoke": {"configuration": {"rounds": 2}}}}
    names = export_plots(report, tmp_path, ("svg",))
    assert set(names) == {"05_convergence_smoke.svg", "07_confusion_smoke.svg"}

    def labels(name):
        return " ".join("".join(node.itertext()) for node in ET.parse(tmp_path/name).iter()
                        if node.tag.endswith("}text"))

    curves = labels("05_convergence_smoke.svg")
    assert all(title in curves for title in ("Accuracy", "Balanced Accuracy", "Macro F1", "Adult Precision", "Adult Recall", "Adult F1", "Non Adult F1"))
    matrix = labels("07_confusion_smoke.svg")
    assert "round 2" in matrix and "Original train · N=10" in matrix and "Validation · N=6" in matrix
    assert "123" not in matrix and "Final test" not in matrix
    assert "True class" in matrix and "Predicted class" in matrix and "100.0%" in matrix


@pytest.mark.parametrize('family,best_round', [('character_cnn', 1), ('deberta_v3_base', 2)])
def test_best_last_figures_use_selected_checkpoint_and_actual_test_round(tmp_path, family, best_round):
    import xml.etree.ElementTree as ET
    from src.evaluation.plots import export_plots

    rows = []
    for rnd in (1, 2):
        # Validation is better in round 2, independently of the recorded selection.
        metrics = dict(confusion_metrics([[77, 3], [6, 4]] if rnd == 1 else [[79, 1], [1, 9]]),
                       loss=.8/rnd, threshold=.5+rnd/10)
        rows.append(dict(experiment_id='selected', family=family, data_kind='synthetic',
                         scope='federated global model', partition='unknown', method='undersampling',
                         split='val', round=rnd, checkpoint_round=rnd, metrics=metrics))
    rows.append(dict(rows[-1], split='test', round=2, checkpoint_round=best_round,
                     metrics=dict(confusion_metrics([[333, 7], [3, 17]]), threshold=.6)))
    report = {'rows': rows, 'experiments': {'selected': {
        'configuration': {'rounds': 2, 'model_selection': {'metric': 'adult_f1'}},
        'selection': {'round': best_round, 'source': 'training_selection'}}}}
    names = export_plots(report, tmp_path, ('svg',))

    def labels(name):
        return ' '.join(''.join(node.itertext()) for node in ET.parse(tmp_path/name).iter()
                        if node.tag.endswith('}text'))

    assert '10_best_vs_last_metrics_selected.svg' in names
    metrics = labels('10_best_vs_last_metrics_selected.svg')
    assert f'Selected best · round {best_round}' in metrics and 'Last · round 2' in metrics
    assert '40.0%' in metrics if best_round == 1 else '90.0%' in metrics
    matrices = labels('11_best_vs_last_confusion_selected.svg')
    assert '333' in matrices and 'Final test · N=360' in matrices
    assert ('Not evaluated' in matrices) == (best_round != 2)
    curves = labels('05_convergence_selected.svg')
    assert (f'selected best round {best_round}' if best_round == 1 else 'selected best = last round 2') in curves

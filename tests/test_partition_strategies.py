"""Strategy interchange, non-IID integrity and comparison contract gates."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.data.preprocessing import (
    PreprocessingConfig, PreprocessingError, prepare_clients, split_client, load_local_split,
)
from src.data.comparison import compare_partitions
from src.data.non_iid_partitioning import dirichlet_indices
from src.data.partitioning import partition_prepared


@pytest.mark.parametrize("axis", ["category", "label"])
def test_dirichlet_is_repeatable_without_replacement_and_changes_ownership(axis):
    categories = np.repeat(np.arange(8), 1000)
    labels = (categories == 0).astype(int)
    a = dirichlet_indices(labels, categories, seed=7, alpha=0.5, stratify_on=axis)
    b = dirichlet_indices(labels, categories, seed=7, alpha=0.5, stratify_on=axis)
    changed = dirichlet_indices(labels, categories, seed=8, alpha=0.5, stratify_on=axis)
    np.testing.assert_array_equal(np.sort(np.concatenate(a)), np.arange(len(labels)))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert any(not np.array_equal(x, y) for x, y in zip(a, changed))
    assert np.ptp([len(p) for p in a]) > 1
    assert np.ptp([labels[p].mean() for p in a]) > 0.01


@pytest.mark.parametrize("parameters", [{"alpha": 0}, {"alpha": -1}, {"alpha": float("nan")},
    {"alpha": True}, {"stratify_on": "invalid"}, {"unexpected": 2}])
def test_invalid_strategy_options_fail_before_touching_data(tmp_path, parameters):
    with pytest.raises(ValueError):
        partition_prepared(tmp_path / "missing", tmp_path / "output",
                           strategy="dirichlet", parameters=parameters)
    assert not (tmp_path / "output").exists()


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("strategy-runs")
    csv_path = root / "raw.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        for domain in range(1600):
            for page in range(3):
                writer.writerow([f"{domain:06}-{page}", f"https://example{domain}.com/{page}",
                                 ("Adult", "Arts", "Business", "Health")[domain % 4]])
    source = root / "cleaned"
    prepare_clients(csv_path, source, PreprocessingConfig(n_features=64))
    for strategy in ("iid", "dirichlet"):
        partition_prepared(source, root / strategy, strategy=strategy, seed=0,
                           parameters={"alpha":2.0} if strategy == "dirichlet" else {})
        for cid in range(5):
            split_client(root / strategy / f"client_{cid:03d}")
    return root


def test_same_pipeline_loader_and_evaluation_population_for_both_strategies(runs):
    comparison = compare_partitions([runs / "iid", runs / "dirichlet"])
    assert comparison["status"] == "PASS"
    assert comparison["comparison_contract"]["retained_records"] == 4800
    per_run = []
    for strategy in ("iid", "dirichlet"):
        report = json.loads((runs / strategy / "report.json").read_text())
        assert report["partitioning"]["name"] == strategy
        assert all(c["dataset"]["class_counts"]["1"] > 0 for c in report["clients"])
        if strategy == "dirichlet":
            assert report["distribution_spreads"]["n_records"]["spread"] > 1
            assert "binary_category_size_spreads_at_most_one" not in report["checks"]
        populations, roles = {name:set() for name in ("train", "val", "test")}, {}
        for cid in range(5):
            for name in populations:
                for line in (runs / strategy / f"client_{cid:03d}/splits/{name}.jsonl").read_text().splitlines():
                    row = json.loads(line)
                    assert roles.setdefault(row["domain"], name) == name
                    assert row["source_row"] not in populations[name]
                    populations[name].add(row["source_row"])
        assert sum(map(len, populations.values())) == 4800
        X, y = load_local_split(runs / strategy / "client_000", "val")
        assert X.shape == (len(y), 64) and y.protected
        assert not (populations["train"] & populations["val"] or
                    populations["train"] & populations["test"] or populations["val"] & populations["test"])
        per_run.append(populations)
    assert per_run[0] == per_run[1]  # same global evaluation examples, different client owners


@pytest.mark.parametrize("field,value", [("split_policy", {"type":"shared_domain_hash_v1", "seed":99}),
    ("features", {"n_features":123}), ("population_sha256", "changed"), ("n_clients", 6)])
def test_comparison_rejects_mismatched_data_or_evaluation_rules(runs, tmp_path, field, value):
    changed = tmp_path / "changed"
    changed.mkdir()
    report = json.loads((runs / "dirichlet/report.json").read_text())
    report["comparison_contract"][field] = value
    (changed / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="incompatible"):
        compare_partitions([runs / "iid", changed])


def test_common_cli_uses_config_and_wont_replace_an_existing_run(runs, tmp_path):
    cfg = {"source": str(runs / "cleaned"), "output": str(tmp_path / "cli-output"),
           "strategy":"dirichlet", "parameters":{"alpha":2.0, "stratify_on":"category"},
           "n_clients":5, "seed":0, "split_seed":0}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/partition_clients.py"),
               "--config", str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((Path(cfg["output"]) / "report.json").read_text())
    assert report["clients"] == json.loads((runs / "dirichlet/report.json").read_text())["clients"]
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_missing_client_class_fails_without_retries_or_partial_output(runs, tmp_path):
    with pytest.raises(PreprocessingError, match="both binary classes"):
        partition_prepared(runs / "cleaned", tmp_path / "extreme", strategy="dirichlet",
                           parameters={"alpha":0.001})
    assert not (tmp_path / "extreme").exists()


def test_unknown_strategy_or_iid_options_are_rejected(tmp_path):
    for strategy, params in (("typo", {}), ("iid", {"alpha":0.5})):
        with pytest.raises(ValueError):
            partition_prepared(tmp_path / "missing", tmp_path / "out", strategy=strategy, parameters=params)

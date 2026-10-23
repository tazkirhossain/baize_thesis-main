"""Independent label/category scenarios, exact presets and category-only margins."""
import json
import csv
from pathlib import Path

import numpy as np
import pytest

from src.data.allocation_plans import category_counts_from_report, category_skew_plan
from src.data.planned_partitioning import planned_indices
from src.data.schema import KNOWN_CATEGORIES
from src.data.preprocessing import PreprocessingConfig, prepare_clients
from src.data.partitioning import partition_prepared
from src.data.verification import verify_partitions

ROOT = Path(__file__).resolve().parents[1]


def test_label_preset_preserves_exact_user_targets_and_alias():
    cfg = json.loads((ROOT / "config/five_client_label_skew.json").read_text())
    alias = json.loads((ROOT / "config/five_client_non_iid.json").read_text())
    assert cfg == alias
    plan = cfg["parameters"]["allocation_plan"]
    assert cfg["strategy"] == "label_skew"
    assert plan["Adult"] == [12000,8500,6500,4500,3311]
    assert plan["Non-adult"] == [281649,285149,287149,289148,290337]
    assert sum(plan["Adult"]) == 34811
    assert sum(plan["Non-adult"]) == 1433432
    assert [a+b for a,b in zip(plan["Adult"],plan["Non-adult"])] == [293649]*3 + [293648]*2


def test_category_preset_is_separate_and_has_no_binary_skew():
    label = json.loads((ROOT / "config/five_client_label_skew.json").read_text())
    category = json.loads((ROOT / "config/five_client_category_skew.json").read_text())
    assert category["strategy"] == "category_skew"
    assert label["output"] != category["output"]
    assert label["source"] == category["source"]
    assert label["split_seed"] == category["split_seed"]
    plan = category["parameters"]["allocation_plan"]
    assert len(plan) == 15
    assert sum(plan["adult"]) == 34811
    non = np.sum([v for k,v in plan.items() if k != "adult"], axis=0)
    assert sum(non) == 1433432
    assert np.ptp(non) <= 1 and np.ptp(plan["adult"]) <= 1
    assert np.ptp(non + plan["adult"]) <= 1
    assert any(np.ptp(v) > 10000 for k,v in plan.items() if k != "adult")


@pytest.mark.parametrize("seed", [0,1,9])
def test_category_plan_is_conservative_repeatable_and_balances_binary_margins(seed):
    counts = {"adult":113, "arts":233, "health":97, "business":139, "sports":121}
    plan = category_skew_plan(counts, seed=seed)
    assert plan == category_skew_plan(counts, seed=seed)
    assert all(sum(plan[k]) == n for k,n in counts.items())
    non = np.sum([v for k,v in plan.items() if k != "adult"], axis=0)
    assert np.ptp(non) <= 1 and np.ptp(plan["adult"]) <= 1
    assert np.ptp(non + plan["adult"]) <= 1
    names = sorted(KNOWN_CATEGORIES)
    cats = np.concatenate([np.full(count,names.index(name)) for name,count in sorted(counts.items())])
    labels = (cats == names.index("adult")).astype(int)
    parts = planned_indices(labels,cats,seed=seed,axis="category",allocation_plan=plan)
    assert len(np.unique(np.concatenate(parts))) == sum(counts.values())
    for category,targets in plan.items():
        assert [np.count_nonzero(cats[p] == names.index(category)) for p in parts] == targets


def test_category_plan_seed_changes_categories_and_totals_stay_fixed():
    counts = {"adult":113, "arts":233, "health":97, "business":139, "sports":121}
    assert category_skew_plan(counts,seed=0) != category_skew_plan(counts,seed=1)


@pytest.mark.parametrize("counts", [{}, {"adult":0,"arts":100}, {"adult":10,"unknown":10},
                                   {"adult":True,"arts":100}, {"adult":10,"arts":1.5}])
def test_invalid_source_counts_are_rejected(counts):
    with pytest.raises(ValueError):
        category_skew_plan(counts)


def test_source_report_reconciliation_rejects_inconsistent_counts():
    report = {"status":"PASS", "retained_records":30, "clients":[{"dataset":{
        "category_counts":{"adult":10,"arts":20}, "class_counts":{"1":10,"0":20}}}]}
    assert category_counts_from_report(report) == {"adult":10,"arts":20}
    report["retained_records"] = 31
    with pytest.raises(ValueError,match="reconcile"):
        category_counts_from_report(report)


@pytest.fixture(scope="module")
def tiny_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("verify-source")
    csv_path = root / "input.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        for domain in range(90):
            for page in range(2):
                writer.writerow([f"{domain:04}-{page}", f"https://source{domain}.com/{page}",
                                 ("Adult", "Arts", "Health")[domain % 3]])
    source = root / "source"
    prepare_clients(csv_path, source, PreprocessingConfig(n_features=64))
    return source


@pytest.mark.parametrize("strategy", ["label_skew", "category_skew"])
def test_full_record_verifier_checks_both_independent_scenarios(tiny_source, tmp_path, strategy):
    if strategy == "label_skew":
        plan = {"Adult":[20,15,10,8,7], "Non-adult":[30,27,24,21,18]}
    else:
        plan = category_skew_plan({"adult":60,"arts":60,"health":60})
    root = tmp_path / strategy
    partition_prepared(tiny_source, root, strategy=strategy, parameters={"allocation_plan":plan})
    result = verify_partitions(tiny_source, root)
    assert result["status"] == "PASS" and result["retained_records"] == 180
    assert result["local_split_directories"] == 0
    assert all(result["checks"].values())
    file = root / "client_000/data.jsonl"
    with file.open("ab") as stream:
        stream.write(b'{}\n')
    with pytest.raises(ValueError, match="checksum"):
        verify_partitions(tiny_source, root)

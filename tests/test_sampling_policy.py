"""Sequential shuffled pools, exact declared plans and exactly-once coverage."""
import numpy as np
import pytest
import csv
import json

from src.data.schema import KNOWN_CATEGORIES
from src.data.planned_partitioning import planned_indices
from src.data.sampling import binary_pools, consume_pool, verify_coverage
from src.data.strategies import resolve_strategy
from src.data.partitioning import partition_prepared
from src.data.preprocessing import PreprocessingConfig, prepare_clients


def test_separate_binary_pools_are_shuffled_and_consumed_sequentially():
    labels = np.array([1] * 20 + [0] * 50)
    pools = binary_pools(labels, np.random.default_rng(42))
    repeated = binary_pools(labels, np.random.default_rng(42))
    parts = [[] for _ in range(5)]
    for label, counts in ((1, [2, 3, 4, 5, 6]), (0, [20, 15, 8, 5, 2])):
        assert np.all(labels[pools[label]] == label)
        np.testing.assert_array_equal(pools[label], repeated[label])
        assert not np.array_equal(pools[label], np.sort(pools[label]))
        chunks = consume_pool(pools[label], counts)
        np.testing.assert_array_equal(np.concatenate(chunks), pools[label])
        offset = 0
        for cid, (chunk, count) in enumerate(zip(chunks, counts)):
            np.testing.assert_array_equal(chunk, pools[label][offset:offset+count])
            parts[cid].append(chunk)
            offset += count
    verify_coverage([np.concatenate(chunks) for chunks in parts], len(labels))


def test_explicit_label_counts_randomize_actual_records_and_exhaust_both_pools():
    labels = np.array([1] * 20 + [0] * 50)
    plan = {"Adult":[2,3,4,5,6], "Non-adult":[20,15,8,5,2]}
    allocate, descriptor = resolve_strategy("label_skew", {"allocation_plan":plan})
    parts = allocate(labels, np.zeros(70, dtype=int), seed=7)
    same = allocate(labels, np.zeros(70, dtype=int), seed=7)
    changed = allocate(labels, np.zeros(70, dtype=int), seed=8)
    verify_coverage(parts, 70)
    for cid, indices in enumerate(parts):
        assert np.count_nonzero(labels[indices] == 1) == plan["Adult"][cid]
        assert np.count_nonzero(labels[indices] == 0) == plan["Non-adult"][cid]
        np.testing.assert_array_equal(indices, same[cid])
    assert any(not np.array_equal(x,y) for x,y in zip(parts,changed))
    assert descriptor["strategy"] == "planned_label_skew"


def test_category_plan_selects_within_original_category_and_exact_targets():
    names = sorted(KNOWN_CATEGORIES)
    cats = np.repeat([names.index("adult"), names.index("arts"), names.index("health")], [20,30,20])
    labels = (cats == names.index("adult")).astype(int)
    plan = {"adult":[2,3,4,5,6], "arts":[10,8,6,4,2], "health":[2,3,4,5,6]}
    allocate, _ = resolve_strategy("category_skew", {"allocation_plan":plan})
    parts = allocate(labels, cats, seed=42)
    verify_coverage(parts, len(labels))
    for category, counts in plan.items():
        for cid, indices in enumerate(parts):
            assert np.count_nonzero(cats[indices] == names.index(category)) == counts[cid]


@pytest.mark.parametrize("plan", [
    {"Adult":[1,1,1,1,1], "Non-adult":[1,1,1,1,1]},
    {"Adult":[2,2,2,2,2], "Non-adult":[1,1,1,1,1]},
])
def test_wrong_population_totals_are_rejected_instead_of_adjusted(plan):
    with pytest.raises(ValueError, match="totals"):
        planned_indices([1]*10 + [0]*20, [0]*30, allocation_plan=plan)


@pytest.mark.parametrize("counts", [[1,2], [1,2,3,-1,5], [1,2,3,4,0.5], [True,1,1,1,1]])
def test_invalid_declared_counts_are_rejected(counts):
    with pytest.raises(ValueError):
        resolve_strategy("label_skew", {"allocation_plan":{"Adult":counts, "Non-adult":[1]*5}})


def test_coverage_guard_rejects_overlap_omission_and_invalid_indices():
    for parts in ([np.array([0]), np.array([0]), np.array([2]), np.array([3]), np.array([4])],
                  [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([5])],
                  [np.array([0.0]), np.array([1]), np.array([2]), np.array([3]), np.array([4])]):
        with pytest.raises(ValueError):
            verify_coverage(parts, 5)


def test_wrong_category_plan_and_oversampling_are_rejected():
    names = sorted(KNOWN_CATEGORIES)
    with pytest.raises(ValueError):
        resolve_strategy("category_skew", {"allocation_plan":{"unknown":[1]*5}})
    with pytest.raises(ValueError, match="exactly the eligible"):
        planned_indices([1]*10, [names.index("adult")]*10, axis="category",
                        allocation_plan={"arts":[2]*5})
    with pytest.raises(ValueError, match="exhaust"):
        consume_pool(np.arange(10), [3]*5)


@pytest.fixture(scope="module")
def cleaned_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("planned-source")
    path = root / "input.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        for domain in range(300):
            for page in range(2):
                writer.writerow([f"id-{domain}-{page}", f"https://site{domain}.com/{page}",
                                 ("Adult", "Arts", "Health")[domain % 3]])
    source = root / "cleaned"
    prepare_clients(path, source, PreprocessingConfig(n_features=64))
    return source


@pytest.mark.parametrize("strategy,plan", [
    ("label_skew", {"Adult":[20,30,40,50,60], "Non-adult":[180,100,60,40,20]}),
    ("category_skew", {"adult":[20,30,40,50,60], "arts":[90,50,30,20,10],
                       "health":[90,50,30,20,10]}),
])
def test_declared_plan_survives_shared_pipeline_without_splitting_or_balancing(cleaned_source, tmp_path, strategy, plan):
    root = tmp_path / strategy
    report = partition_prepared(cleaned_source, root, strategy=strategy,
                               parameters={"allocation_plan":plan}, seed=42)
    assert report["retained_records"] == 600
    assert report["partitioning"]["sampling_policy"] == "binary_shuffled_pools_sequential_v1"
    seen = set()
    for cid, client in enumerate(report["clients"]):
        data = client["dataset"]
        assert data["class_counts"]["1"] == (plan["Adult"] if strategy == "label_skew" else plan["adult"])[cid]
        if strategy == "label_skew":
            assert data["class_counts"]["0"] == plan["Non-adult"][cid]
        else:
            assert all(data["category_counts"][name] == counts[cid] for name, counts in plan.items())
        directory = root / f"client_{cid:03d}"
        assert not (directory / "splits").exists()
        for line in (directory / "data.jsonl").read_text().splitlines():
            row = json.loads(line)
            assert row["source_row"] not in seen
            assert row["record_id"].startswith("id-")
            seen.add(row["source_row"])
    assert len(seen) == 600

"""IID allocation quotas, identity coverage, repeatability and global role safety."""
import csv
import json
import shutil

import numpy as np
import pytest

from src.data.preprocessing import (
    PreprocessingConfig, PreprocessingError, prepare_clients, split_client, load_local_split,
)
from src.data.iid_partitioning import stratified_iid_indices, partition_prepared_iid
from src.data.split_policy import shared_domain_split


@pytest.mark.parametrize("seed", [0, 1, 23])
def test_exact_binary_category_and_size_quotas_without_replacement(seed):
    # Several uneven categories make a naive independent array_split accumulate
    # all remainder records on the same client. All three margins must balance.
    counts = [113, 71, 37, 21, 17, 9]
    cats = np.repeat(np.arange(len(counts)), counts)
    labels = (cats == 0).astype(np.int8)
    parts = stratified_iid_indices(labels, cats, seed)
    assert len(parts) == 5
    np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(len(labels)))
    assert max(map(len, parts)) - min(map(len, parts)) <= 1
    for label in (0, 1):
        sizes = [int((labels[p] == label).sum()) for p in parts]
        assert max(sizes)-min(sizes) <= 1
    for category in np.unique(cats):
        sizes = [int((cats[p] == category).sum()) for p in parts]
        assert max(sizes)-min(sizes) <= 1


def test_repeatability_seed_effect_and_exactly_five():
    cats = np.repeat(np.arange(4), 100)
    labels = (cats == 0).astype(int)
    a = stratified_iid_indices(labels, cats, seed=7)
    b = stratified_iid_indices(labels, cats, seed=7)
    changed = stratified_iid_indices(labels, cats, seed=8)
    assert all(np.array_equal(p, q) for p, q in zip(a, b))
    assert any(not np.array_equal(p, q) for p, q in zip(a, changed))
    for count in (1, 4, 6):
        with pytest.raises(ValueError, match="exactly five"):
            stratified_iid_indices(labels, cats, n_clients=count)
    with pytest.raises(ValueError, match="parallel"):
        stratified_iid_indices(labels, cats[:-1])
    with pytest.raises(ValueError, match="binary"):
        stratified_iid_indices([0.5] * 20, ["Arts"] * 20)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    directory = tmp_path_factory.mktemp("iid-source")
    source = directory / "raw.csv"
    categories = ("Adult", "Arts", "Health", "Business")
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        for domain in range(800):
            for page in range(3):
                writer.writerow([f"{domain:05d}-{page}", f"https://site{domain}.com/{page}",
                                 categories[domain % 4]])
    root = directory / "prepared"
    prepare_clients(source, root, PreprocessingConfig(n_clients=5, n_features=64))
    return root


def test_real_pipeline_is_url_iid_and_clients_split_without_global_files(prepared, tmp_path):
    root = tmp_path / "iid"
    report = partition_prepared_iid(prepared, root, seed=0, split_seed=0)
    assert len(report["clients"]) == 5
    assert report["retained_records"] == 2400
    assert all(v["spread"] <= 1 for v in report["distribution_spreads"].values())
    numbers, canonical, domain_clients = set(), set(), {}
    for cid in range(5):
        for line in (root / f"client_{cid:03d}/data.jsonl").read_text().splitlines():
            row = json.loads(line)
            assert row["source_row"] not in numbers and row["url"] not in canonical
            numbers.add(row["source_row"])
            canonical.add(row["url"])
            domain_clients.setdefault(row["domain"], set()).add(cid)
            assert row["record_id"].startswith("0")  # leading zeros preserved
    assert len(numbers) == 2400
    assert any(len(owners) > 1 for owners in domain_clients.values())
    roles = {}
    for cid in range(5):
        # Copy only this client's own directory, with no provisioning report.
        isolated = tmp_path / f"isolated-{cid}"
        shutil.copytree(root / f"client_{cid:03d}", isolated)
        local = split_client(isolated)
        assert local["split_policy"]["type"] == "shared_domain_hash_v1"
        assert sum(r["n_records"] for r in local["splits"].values()) == 480
        for name in ("train", "val", "test"):
            for line in (isolated / f"splits/{name}.jsonl").read_text().splitlines():
                domain = json.loads(line)["domain"]
                assert roles.setdefault(domain, name) == name
        X, y = load_local_split(isolated, "val")
        assert X.shape == (len(y), 64) and y.protected
    assert (root / "exclusions.jsonl").read_bytes() == (prepared / "exclusions.jsonl").read_bytes()


def test_artifacts_reproduce_and_input_tampering_blocks_publication(prepared, tmp_path):
    a = partition_prepared_iid(prepared, tmp_path / "a", seed=19)
    b = partition_prepared_iid(prepared, tmp_path / "b", seed=19)
    assert a["clients"] == b["clients"]
    with pytest.raises(FileExistsError):
        partition_prepared_iid(prepared, tmp_path / "a")
    modified = tmp_path / "modified"
    shutil.copytree(prepared, modified)
    with (modified / "client_000/data.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(PreprocessingError):
        partition_prepared_iid(modified, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_shared_role_is_stable_and_has_all_three_roles():
    roles = [shared_domain_split(f"site{i}.com", 0) for i in range(1000)]
    assert set(roles) == {0, 1, 2}
    assert roles == [shared_domain_split(f"site{i}.com", 0) for i in range(1000)]
    assert roles != [shared_domain_split(f"site{i}.com", 1) for i in range(1000)]
    with pytest.raises(ValueError):
        shared_domain_split("site.com", 0, (1, 0, 0))


def test_unknown_shared_split_policy_is_rejected(prepared, tmp_path):
    isolated = tmp_path / "client"
    shutil.copytree(prepared / "client_000", isolated)
    settings = json.loads((isolated / "preprocessing.json").read_text())
    settings["split_policy"] = {"type": "unrecognized", "seed": 0}
    (isolated / "preprocessing.json").write_text(json.dumps(settings))
    with pytest.raises(PreprocessingError, match="unknown client split policy"):
        split_client(isolated)
    assert not (isolated / "splits").exists()


def test_client_cannot_silently_change_the_shared_split_rule(prepared, tmp_path):
    root = tmp_path / "iid"
    partition_prepared_iid(prepared, root)
    client = root / "client_000"
    file = client / "preprocessing.json"
    settings = json.loads(file.read_text())
    settings["split_policy"]["seed"] += 1
    file.write_text(json.dumps(settings))
    with pytest.raises(PreprocessingError, match="configuration checksum"):
        split_client(client)
    assert not (client / "splits").exists()

"""Integrity tests for client-local preprocessing, including failed publication."""
import csv
import json
import shutil

import numpy as np
import pytest

from src.data.schema import binary_label
from src.data.balancing import BalanceConfig, sample_training_indices
from src.data.preprocessing import (
    PreprocessingConfig, PreprocessingError, load_client_split, prepare_clients,
    split_client, load_local_split,
)
from src.data.url import normalize_url


def corpus(tmp_path, extras=()):
    path = tmp_path / "source.csv"
    rows = []
    for i in range(120):
        # Several pages per domain exercise ownership and local leakage gates.
        for page in range(2):
            rows.append([str(i), f"https://site{i}.com/page{page}?token=SECRET",
                         "Adult" if i % 2 else "Health"])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows(rows)
        writer.writerows(extras)
    return path


def test_all_rows_accounted_and_domains_owned_once(tmp_path):
    source = corpus(tmp_path, [
        ["duplicate", "http://site1.com/page0", "Adult"],
        ["conflict", "http://site2.com/page0", "Adult"],
        ["unknown", "https://unknown.com/", "NotACategory"],
        ["missing", "", "Adult"],
        ["unlabelled", "https://unlabelled.com/", ""],
        ["bad"],
        ["invalid", "https://alice@TOPSECRET@localhost/", "Adult"],
    ])
    root = tmp_path / "clients"
    report = prepare_clients(source, root, PreprocessingConfig(n_clients=3, n_features=32))
    assert report["ingestion"]["input_rows"] == 247
    assert report["retained_records"] == 239
    assert report["duplicate_rows_removed"] == 1
    assert report["conflicting_source_rows"] == 2
    assert report["rejected_rows"] == 5
    assert all(report["checks"].values())
    ownership, canonical = {}, set()
    for client in report["clients"]:
        directory = root / f"client_{client['client_id']:03d}"
        assert (directory / "data.jsonl").exists()
        assert not (directory / "splits").exists()
        local_report = split_client(directory)
        for split in ("train", "val", "test"):
            path = directory / "splits" / f"{split}.jsonl"
            assert "SECRET" not in path.read_text()
            for line in path.read_text().splitlines():
                row = json.loads(line)
                owner = (client["client_id"], split)
                assert ownership.setdefault(row["domain"], owner) == owner
                key = normalize_url(row["url"]).canonical
                assert key not in canonical
                canonical.add(key)
    quarantine = (root / "quarantine.jsonl").read_text()
    assert "TOPSECRET" not in quarantine
    assert "label_conflict" in quarantine and "unlabelled" in quarantine


def test_deterministic_and_source_unchanged(tmp_path):
    source = corpus(tmp_path)
    original = source.read_bytes()
    cfg = PreprocessingConfig(n_clients=3)
    a = prepare_clients(source, tmp_path / "a", cfg)
    b = prepare_clients(source, tmp_path / "b", cfg)
    assert a["clients"] == b["clients"]
    for i in range(3):
        assert split_client(tmp_path / "a" / f"client_{i:03d}") == split_client(
            tmp_path / "b" / f"client_{i:03d}")
    assert source.read_bytes() == original
    with pytest.raises(FileExistsError):
        prepare_clients(source, tmp_path / "a", cfg)


def test_client_features_and_protected_evaluation(tmp_path):
    root = tmp_path / "clients"
    prepare_clients(corpus(tmp_path), root, PreprocessingConfig(n_clients=3, n_features=64))
    split_client(root / "client_000")
    for split in ("train", "val", "test"):
        X, y = load_client_split(root, 0, split)
        assert X.shape == (len(y), 64)
        np.testing.assert_allclose(np.asarray(X.multiply(X).sum(axis=1)).ravel(), 1)
        assert set(y) == {0, 1}
        if split != "train":
            assert not y.flags.writeable
            with pytest.raises(ValueError, match="training"):
                sample_training_indices(y, BalanceConfig(), split="train")
    file = root / "client_000/splits/train.jsonl"
    file.write_text(file.read_text() + "{}\n")
    with pytest.raises(PreprocessingError, match="checksum"):
        load_client_split(root, 0, "train")


@pytest.mark.parametrize("fractions", [(0.7, 0.2, 0.2), (float('nan'), 0.5, 0.5), (1, 0, 0)])
def test_invalid_configuration_does_not_publish(tmp_path, fractions):
    out = tmp_path / "clients"
    with pytest.raises(PreprocessingError):
        prepare_clients(corpus(tmp_path), out, PreprocessingConfig(fractions=fractions))
    assert not out.exists()


def test_single_class_and_insufficient_domains_block(tmp_path):
    path = tmp_path / "small.csv"
    path.write_text("1,https://a.com/,Adult\n2,https://b.com/,Arts\n")
    out = tmp_path / "clients"
    with pytest.raises(PreprocessingError):
        prepare_clients(path, out)
    assert not out.exists()
    assert not list(tmp_path.glob(".preparing-*"))


def test_strict_binary_labels_reject_fractional_and_overflow_values():
    for invalid in (0.9, 1.9, 256, -1, "adult", [0, 1]):
        with pytest.raises(ValueError):
            binary_label(invalid)
    assert binary_label("0") == 0 and binary_label("1") == 1


def test_invalid_credential_and_control_character_privacy():
    for raw in ("https://alice@TOPSECRET@localhost/", "alice@TOPSECRET@localhost"):
        result = normalize_url(raw)
        assert not result.is_valid
        assert "TOPSECRET" not in result.raw
    assert not normalize_url("https://a.com/\nsecret").is_valid


def test_client_splits_and_loads_without_access_to_global_or_other_client_files(tmp_path):
    root = tmp_path / "provisioned"
    prepare_clients(corpus(tmp_path), root, PreprocessingConfig(n_clients=3, n_features=32))
    isolated = tmp_path / "isolated-client"
    shutil.copytree(root / "client_001", isolated)
    original = (isolated / "data.jsonl").read_bytes()
    result = split_client(isolated)
    assert result["state"] == "split_locally"
    assert sum(s["n_records"] for s in result["splits"].values()) == 80
    assert (isolated / "data.jsonl").read_bytes() == original
    X, y = load_local_split(isolated, "val")
    assert X.shape[0] == len(y) and y.protected
    assert not (root / "client_001/splits").exists()
    with pytest.raises(FileExistsError):
        split_client(isolated)


def test_unsplit_tampering_blocks_local_split(tmp_path):
    root = tmp_path / "clients"
    prepare_clients(corpus(tmp_path), root, PreprocessingConfig(n_clients=3))
    client = root / "client_000"
    with (client / "data.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(PreprocessingError, match="checksum"):
        split_client(client)
    assert not (client / "splits").exists()


def test_local_class_coverage_failure_does_not_publish_splits(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("\n".join(
        [f"{i},https://adult.com/{i},Adult" for i in range(30)] +
        [f"{i},https://negative{i}.com/,Arts" for i in range(30)]))
    root = tmp_path / "clients"
    prepare_clients(source, root, PreprocessingConfig(n_clients=1))
    with pytest.raises(PreprocessingError, match="records of each class"):
        split_client(root / "client_000")
    assert not (root / "client_000/splits").exists()
    assert not list((root / "client_000").glob(".splitting-*"))


def test_ids_original_categories_and_complete_exclusion_ledger(tmp_path):
    source = corpus(tmp_path, [
        ["repeat-id", "http://site1.com/page0", " Adult "],
        ["conflict-id", "http://site2.com/page0", "Adult"],
        ["repeated-conflict-id", "http://site2.com/page0", "Health"],
        ["", "", ""],
        ["missing-label", "https://unlabelled.com/", ""],
    ])
    root = tmp_path / "prepared"
    report = prepare_clients(source, root, PreprocessingConfig(n_clients=3))
    assert report["missing_values"] == {"id": 1, "url": 1, "label": 2}
    entries = [json.loads(line) for line in (root / "exclusions.jsonl").read_text().splitlines()]
    assert len(entries) == report["ingestion"]["input_rows"] - report["retained_records"]
    assert len({r["source_row"] for r in entries}) == len(entries)
    duplicate = next(r for r in entries if r["record_id"] == "repeat-id")
    assert duplicate["original_category"] == " Adult "
    assert duplicate["reason"] == "duplicate_url" and duplicate["kept_record_id"] == "1"
    conflicts = [r for r in entries if r["reason"] == "label_conflict"]
    assert {r["record_id"] for r in conflicts} == {"2", "conflict-id", "repeated-conflict-id"}
    missing = next(r for r in entries if r["record_id"] == "")
    assert set(missing["reasons"]) == {"missing_id", "missing_url", "unlabelled"}
    retained = []
    for p in root.glob("client_*/data.jsonl"):
        retained.extend(json.loads(line) for line in p.read_text().splitlines())
    assert all(r["target"] == ("Adult" if r["label"] else "Non-adult") for r in retained)
    assert all("record_id" in r and "original_category" in r for r in retained)
    assert {r["original_category"] for r in retained} == {"Adult", "Health"}


@pytest.mark.parametrize("header", ["id,url,wrong", "id,id,label", "url,label", "id,url,label,extra"])
def test_bad_declared_headers_fail_without_publishing(tmp_path, header):
    source = corpus(tmp_path)
    source.write_text(header + "\n" + source.read_text())
    out = tmp_path / "bad"
    with pytest.raises(PreprocessingError, match="CSV header"):
        prepare_clients(source, out, PreprocessingConfig(header="present", n_clients=3))
    assert not out.exists()


def test_reordered_named_columns_preserve_exact_source_identity(tmp_path):
    source = corpus(tmp_path)
    rows = list(csv.reader(source.open()))
    with source.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "id", "url"])
        for record_id, url, label in rows:
            writer.writerow([f" {label} ", f"ID-{record_id}", url])
    root = tmp_path / "prepared"
    report = prepare_clients(source, root, PreprocessingConfig(header="present", n_clients=3))
    assert report["input_schema"]["columns"] == ["label", "id", "url"]
    assert report["ingestion"]["input_rows"] == 240
    row = json.loads(next(root.glob("client_*/data.jsonl")).read_text().splitlines()[0])
    assert row["record_id"].startswith("ID-")
    assert row["original_category"] == f" {row['category'].title()} "


def test_exact_source_snapshot_and_runtime_lock_are_written(tmp_path):
    import hashlib
    root = tmp_path / "prepared"
    report = prepare_clients(corpus(tmp_path), root, PreprocessingConfig(n_clients=3))
    for name, expected in report["reproduction"]["source_files_sha256"].items():
        assert hashlib.sha256((root / "code_snapshot" / name).read_bytes()).hexdigest() == expected
    lock = (root / "runtime-requirements.txt").read_text()
    assert "numpy==" in lock and "tldextract==" in lock and "packaging==" in lock
    assert "http" not in lock and "@" not in lock
    assert not report["reproduction"]["cold_environment_reproduction_verified"]

"""Read-only full-record verification for outputs from the shared sampler."""
from array import array
from collections import Counter
import hashlib
import heapq
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np

from src.data.schema import KNOWN_CATEGORIES
from src.data.preprocessing import _file_hash
from src.data.url import normalize_url
from src.data.sampling import SAMPLING_POLICY
from src.data.strategies import resolve_strategy


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ordered_lines(path):
    previous = -1
    with path.open("rb") as stream:
        for line in stream:
            number = json.loads(line)["source_row"]
            _require(number > previous, "records are not uniquely sorted by source row")
            previous = number
            yield number, line


def verify_partitions(source, root):
    """Verify actual records, original bytes, audit hashes and seeded membership.

    Does not split, normalize feature vectors, balance or modify any dataset.
    Older sampling versions require their saved code instead of this verifier.
    """
    source, root = Path(source), Path(root)
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    original = json.loads((source / "report.json").read_text(encoding="utf-8"))
    settings = json.loads((root / "preprocessing.json").read_text(encoding="utf-8"))
    partitioning = report["partitioning"]
    _require(report.get("status") == original.get("status") == "PASS", "non-PASS audit")
    _require(partitioning.get("sampling_policy") == SAMPLING_POLICY,
             "sampling version differs; use the run's archived code")
    _require(report["prepared_source_report_sha256"] == _file_hash(source / "report.json"), "source audit changed")
    _require(json.loads((root / "cleaning_report.json").read_text()) == original, "cleaning audit changed")
    _require(len(report["clients"]) == settings["config"]["n_clients"] == 5, "expected five clients")
    _require([c["client_id"] for c in report["clients"]] == list(range(5)), "client IDs differ")
    for filename, expected in (("exclusions.jsonl", original["excluded_records"]["sha256"]),
                               ("quarantine.jsonl", original["quarantine_sha256"])):
        _require(_file_hash(source / filename) == _file_hash(root / filename) == expected,
                 "source exclusion/quarantine audit changed")
    names = sorted(KNOWN_CATEGORIES)
    codes = {name:i for i,name in enumerate(names)}
    numbers, labels, categories, owners = array("q"), array("b"), array("h"), array("b")
    seen_urls = set()
    actual_clients = []
    for cid, expected in enumerate(report["clients"]):
        directory = root / f"client_{cid:03d}"
        manifest = json.loads((directory / "manifest.json").read_text())
        _require(manifest == expected, "client manifest differs from root audit")
        config_hash = _file_hash(directory / "preprocessing.json")
        _require(config_hash == manifest["preprocessing_sha256"] == _file_hash(root / "preprocessing.json"),
                 "client preprocessing contract changed")
        file = directory / "data.jsonl"
        _require(_file_hash(file) == expected["dataset"]["sha256"], "client file checksum mismatch")
        by_class, by_category = Counter(), Counter()
        with file.open("rb") as stream:
            for line in stream:
                row = json.loads(line)
                normalized = normalize_url(row["url"])
                _require(normalized.is_valid and normalized.registrable_domain == row["domain"], "invalid URL/domain")
                _require(normalized.canonical not in seen_urls, "canonical URL appears more than once")
                seen_urls.add(normalized.canonical)
                numbers.append(row["source_row"])
                labels.append(row["label"])
                categories.append(codes[row["category"]])
                owners.append(cid)
                by_class[str(row["label"])] += 1
                by_category[row["category"]] += 1
        info = expected["dataset"]
        _require(sum(by_class.values()) == info["n_records"], "client row count mismatch")
        _require(dict(by_class) == info["class_counts"] and dict(by_category) == info["category_counts"],
                 "actual class/category counts differ from report")
        actual_clients.append({"client_id":cid, "n_records":info["n_records"],
                               "class_counts":dict(by_class), "category_counts":dict(by_category)})
        print(f"Client {cid+1}: actual counts, URL uniqueness and fingerprints PASS", flush=True)
    _require(len(numbers) == len(seen_urls) == report["retained_records"] == original["retained_records"],
             "population total changed")
    order = np.argsort(numbers)
    _require(np.all(np.diff(np.asarray(numbers)[order]) > 0), "a source record appears on multiple clients")
    allocate, _ = resolve_strategy(partitioning["name"], partitioning["parameters"])
    parts = allocate(np.asarray(labels)[order], np.asarray(categories)[order],
                     seed=partitioning["seed"], n_clients=5)
    actual_owners = np.asarray(owners)[order]
    _require(all(np.all(actual_owners[indices] == cid) for cid,indices in enumerate(parts)),
             "fixed-seed replay gives different client memberships")
    # Compare every complete record byte to the eligible original population.
    original_paths = [source / f"client_{c['client_id']:03d}/data.jsonl" for c in original["clients"]]
    for path, client in zip(original_paths, original["clients"]):
        _require(_file_hash(path) == client["dataset"]["sha256"], "prepared source changed")
    before = heapq.merge(*(_ordered_lines(path) for path in original_paths), key=lambda p:p[0])
    after = heapq.merge(*(_ordered_lines(root / f"client_{cid:03d}/data.jsonl") for cid in range(5)), key=lambda p:p[0])
    population = hashlib.sha256()
    for original_record, allocated_record in zip_longest(before, after):
        _require(original_record == allocated_record, "eligible original records were changed, omitted or duplicated")
        population.update(allocated_record[1])
    _require(population.hexdigest() == report["comparison_contract"]["population_sha256"],
             "population fingerprint mismatch")
    for relative, expected in report["reproduction"]["source_files_sha256"].items():
        _require(_file_hash(root / "code_snapshot" / relative) == expected, "archived preparation code changed")
    return {"status":"PASS", "retained_records":len(numbers), "clients":actual_clients,
        "allocation_report_sha256":_file_hash(root / "report.json"),
        "population_sha256":population.hexdigest(),
        "local_split_directories":sum((root / f"client_{cid:03d}/splits").exists() for cid in range(5)),
        "checks":{"canonical_url_uniqueness":True, "disjoint_client_ownership":True,
                  "every_eligible_original_record_exactly_once":True, "complete_original_record_bytes_preserved":True,
                  "seeded_membership_replay":True, "actual_label_and_category_counts":True,
                  "shared_local_preprocessing_contract":True, "source_audit_unchanged":True,
                  "saved_preparation_code_fingerprints":True}}

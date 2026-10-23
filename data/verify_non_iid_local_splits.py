"""Private full-record audit of client-local splits and global domain roles."""
from collections import Counter
import hashlib
import heapq
from itertools import zip_longest
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baize.data.preprocessing import _file_hash, load_local_split
from src.data.split_policy import shared_domain_split

scenarios = {
    "label_skew": ROOT / "data/client-label-skew-v1",
    "category_skew": ROOT / "data/client-category-skew-v1",
    "historical_iid": ROOT / "data/client-iid-v1",
}
roles = {}
populations = {}
reports = {}


def ordered(path):
    previous = -1
    with path.open("rb") as stream:
        for line in stream:
            number = json.loads(line)["source_row"]
            assert number > previous
            previous = number
            yield number, line


for scenario, root in scenarios.items():
    provision = json.loads((root / "report.json").read_text())
    verified = json.loads((root / "verification.json").read_text())
    assert verified["status"] == provision["status"] == "PASS"
    assert verified["allocation_report_sha256"] == _file_hash(root / "report.json")
    settings = json.loads((root / "preprocessing.json").read_text())
    assert settings["config"]["fractions"] == [0.7,0.15,0.15]
    assert settings["split_policy"] == {"type":"shared_domain_hash_v1", "seed":0}
    population = np.full(provision["ingestion"]["input_rows"] + 2, -1, dtype=np.int8)
    clients = []
    totals = Counter()
    all_labels = {name:Counter() for name in ("train","val","test")}
    for cid in range(5):
        directory = root / f"client_{cid:03d}"
        parent = json.loads((directory / "manifest.json").read_text())
        local = json.loads((directory / "splits/manifest.json").read_text())
        assert parent == provision["clients"][cid]
        assert local["state"] == "split_locally" and local["client_id"] == cid
        assert local["source_sha256"] == parent["dataset"]["sha256"] == _file_hash(directory / "data.jsonl")
        assert (_file_hash(directory / "preprocessing.json") == parent["preprocessing_sha256"] ==
                _file_hash(directory / "splits/preprocessing.json") == local["preprocessing_sha256"])
        assert local["split_policy"] == settings["split_policy"]
        assert json.loads((directory / "splits/preprocessing.json").read_text()) == settings
        stats = {}
        split_counts = Counter()
        split_categories = Counter()
        for slot, split in enumerate(("train","val","test")):
            digest = hashlib.sha256()
            classes, categories, domains = Counter(), Counter(), set()
            with (directory / f"splits/{split}.jsonl").open("rb") as stream:
                for line in stream:
                    digest.update(line)
                    row = json.loads(line)
                    domain = row["domain"]
                    if domain not in roles:
                        roles[domain] = shared_domain_split(domain, 0, (0.7,0.15,0.15))
                    assert roles[domain] == slot
                    number = row["source_row"]
                    assert population[number] == -1
                    population[number] = slot
                    classes[str(row["label"])] += 1
                    categories[row["category"]] += 1
                    domains.add(domain)
            info = local["splits"][split]
            count = sum(classes.values())
            assert digest.hexdigest() == info["sha256"]
            assert count == info["n_records"] and dict(classes) == info["class_counts"]
            assert dict(categories) == info["category_counts"] and len(domains) == info["n_domains"]
            assert all(classes[str(label)] >= settings["config"]["min_per_class"] for label in (0,1))
            split_counts.update(classes)
            split_categories.update(categories)
            totals[split] += count
            all_labels[split].update(classes)
            stats[split] = {"n_records":count, "class_counts":dict(classes),
                            "actual_fraction":count / parent["dataset"]["n_records"],
                            "n_domains":len(domains)}
        assert dict(split_counts) == parent["dataset"]["class_counts"]
        assert dict(split_categories) == parent["dataset"]["category_counts"]
        merged = heapq.merge(*(ordered(directory / f"splits/{s}.jsonl")
                               for s in ("train","val","test")), key=lambda p:p[0])
        for original, reconstructed in zip_longest(ordered(directory / "data.jsonl"), merged):
            assert original == reconstructed
        clients.append({"client":cid+1, "client_id":cid, "n_records":parent["dataset"]["n_records"],
                        "splits":stats})
        print(f"{scenario} Client {cid+1}: counts, hashes, complete record coverage and domain roles PASS",flush=True)
    assert int(np.count_nonzero(population >= 0)) == provision["retained_records"] == sum(totals.values())
    populations[scenario] = population
    feature = None
    if scenario != "historical_iid":
        matrix, labels = load_local_split(root / "client_000", "val")
        norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
        assert matrix.shape[1] == 65536 and labels.protected
        assert np.allclose(norms[norms > 0],1)
        feature = {"client":1,"split":"val","shape":list(matrix.shape),
                   "l2_normalized":True,"labels_protected":True}
    reports[scenario] = {"status":"PASS", "scenario":scenario,
        "configured_fractions":{"train":0.7,"val":0.15,"test":0.15},
        "split_policy":settings["split_policy"], "n_clients":5,
        "n_records":provision["retained_records"], "clients":clients,
        "totals":dict(totals),"class_counts_by_split":{k:dict(v) for k,v in all_labels.items()},
        "feature_check":feature, "allocation_report_sha256":_file_hash(root / "report.json"),
        "verification_script_sha256":_file_hash(Path(__file__)),
        "checks":{"all_15_local_split_files_verified":True,"complete_original_record_bytes_preserved":True,
                  "unique_source_row_coverage":True,"no_cross_split_domain_leakage":True,
                  "shared_policy_config_checksums":True,"both_classes_in_every_local_split":True,
                  "unsplit_client_datasets_unchanged":True}}

assert np.array_equal(populations["label_skew"], populations["category_skew"])
assert np.array_equal(populations["label_skew"], populations["historical_iid"])
for scenario in reports:
    reports[scenario]["checks"]["same_global_evaluation_records_across_scenarios"] = True
    reports[scenario]["global_domain_role_conflicts"] = 0
    reports[scenario]["global_domains_verified"] = len(roles)
    if scenario != "historical_iid":
        with (scenarios[scenario] / "local_split_verification.json").open("x",encoding="utf-8") as stream:
            stream.write(json.dumps(reports[scenario],indent=2) + "\n")
summary = {"status":"PASS", "scope":"data split verification; no model training",
    "global_domain_role_conflicts":0,"global_domains_verified":len(roles),
    "same_global_evaluation_records_across_scenarios":True,
    "runs":[reports[name] for name in ("label_skew","category_skew")]}
out = ROOT / "results/client_local_split_summary_v1.json"
out.parent.mkdir(parents=True,exist_ok=True)
with out.open("x",encoding="utf-8") as stream:
    stream.write(json.dumps(summary,indent=2) + "\n")
print(f'PASS: all 45 split files, {len(roles):,} domain roles and identical global split membership across all three scenarios')
print(json.dumps({name:report["totals"] for name,report in reports.items()},indent=2))

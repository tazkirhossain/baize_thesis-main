"""Independent audit of all four real-client experiments and immutable inputs."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baize.data.preprocessing import _file_hash

output = ROOT / "data/balancing-four-methods-equal-v2"
summary = json.loads((output / "summary.json").read_text())
assert summary["status"] == "PASS" and summary["model_family"] == "engineered_numeric"
assert len(summary["clients"]) == 15
assert summary["neural_models_trained"] == []
warnings = Counter()
for client in summary["clients"]:
    directory = ROOT / "data" / client["partition"] / f"client_{client['client_id']:03d}"
    owned = json.loads((directory / "manifest.json").read_text())
    local = json.loads((directory / "splits/manifest.json").read_text())
    assert _file_hash(directory / "data.jsonl") == owned["dataset"]["sha256"]
    hashes = {name:_file_hash(directory / f"splits/{name}.jsonl") for name in ("train","val","test")}
    assert hashes == client["input_split_hashes"]
    assert all(hashes[name] == local["splits"][name]["sha256"] for name in hashes)
    assert len(client["runs"]) == 4
    assert {r["method"] for r in client["runs"]} == {"undersampling","oversampling","smote","ensemble_undersampling"}
    train = local["splits"]["train"]
    a,n = train["class_counts"]["1"],train["class_counts"]["0"]
    transform = client["runs"][0]["feature_transform"]
    for run in client["runs"]:
        assert run["status"] == "PASS" and run["neural_models_trained"] == []
        assert run["experiment_family"] == run["model"]["family"] == "engineered_numeric"
        assert run["model"]["parameters"]["class_weight"] is None
        assert run["model"]["parameters"]["early_stopping"] is False
        assert run["data_contract"] == hashes and all(run["checks"].values())
        assert run["training_original_counts"] == {"Adult":a,"Non-adult":n}
        assert run["feature_transform"] == transform
        assert transform["n_fit_records"] == train["n_records"] and transform["fit_split"] == "train"
        cfg = run["balancing_config"]
        assert cfg["adult_to_non_adult"] == "1:1"
        assert cfg["seed"] == 42+client["client_id"]
        sampling = run["sampling"]
        if run["method"] == "ensemble_undersampling":
            members = sampling["members"]
            assert len(members) == 5 and sampling["aggregation"] == "mean_adult_probability"
            assert len({m["selection_sha256"] for m in members}) == 5
            assert len({m["member_seed"] for m in members}) == 5
            assert all(m["before"] == {"Adult":a,"Non-adult":n} for m in members)
            assert all(m["after"] == {"Adult":a,"Non-adult":a} for m in members)
            assert all(m["duplicated_examples"] == m["synthetic_examples"] == 0 for m in members)
            audits = [m["fit_audit"] for m in members]
        else:
            assert sampling["before"] == {"Adult":a,"Non-adult":n}
            if run["method"] == "undersampling":
                assert sampling["after"] == {"Adult":a,"Non-adult":a}
                assert sampling["removed_training_examples"] == n-a
                assert sampling["duplicated_examples"] == sampling["synthetic_examples"] == 0
            else:
                target = n
                assert sampling["after"] == {"Adult":target,"Non-adult":n}
                generated_key = "synthetic_examples" if run["method"] == "smote" else "duplicated_examples"
                assert sampling[generated_key] == target-a
                assert sampling["removed_training_examples"] == 0
                if run["method"] == "smote":
                    assert sampling["effective_k_neighbors"] == 5 and not sampling["generated_urls"]
            audits = [sampling["fit_audit"]]
        for audit in audits:
            warnings.update(w["category"] for w in audit["warnings"])
        for split in ("val","test"):
            m = run["metrics"][split]
            expected = local["splits"][split]
            assert m["n_records"] == expected["n_records"]
            assert m["adult_count"] == expected["class_counts"]["1"]
            assert m["non_adult_count"] == expected["class_counts"]["0"]
            assert m["threshold"] == 0.5
            assert sum(sum(row) for row in m["confusion_matrix"]) == expected["n_records"]
    print(f"{client['partition']} Client {client['client']}: four methods, class counts and unchanged input files PASS",flush=True)
for relative,expected in summary["reproduction"]["source_files_sha256"].items():
    assert _file_hash(output / "code_snapshot" / relative) == expected
result = {"status":"PASS","n_clients":15,"n_method_runs":60,"n_independent_models_fitted":120,
    "original_client_datasets_verified":15,"original_split_files_verified":45,
    "fit_warnings":dict(warnings),"summary_sha256":_file_hash(output / "summary.json"),
    "verification_script_sha256":_file_hash(Path(__file__)),
    "checks":{"all_original_input_files_unchanged":True,"training_only_resampling":True,
        "evaluation_class_distributions_unchanged":True,"four_methods_kept_separate":True,
        "exact_class_counts_and_generated_counts":True,"training_only_scaler_fit":True,
        "numerical_neural_experiments_distinguished":True,"independent_ensemble_members":True,
        "no_class_weight_or_focal_substitution":True,"fixed_predeclared_ratio_seed_and_threshold":True,
        "code_snapshot_fingerprints":True}}
with (output / "verification.json").open("x",encoding="utf-8") as stream:
    stream.write(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))

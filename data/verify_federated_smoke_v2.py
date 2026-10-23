"""Independent after-run accounting; reads synthetic data only, never real URLs."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baize.data.preprocessing import _file_hash
from src.models.federated_state import state_hash
from src.server.config import contract_hash

output = ROOT / "data/federated-cnn-smoke-v2"
metadata = json.loads((output / "server/run.json").read_text())
config = json.loads((output / "experiment.json").read_text())
assert metadata["status"] == "PASS" and metadata["completed_rounds"] == 2
assert metadata["contract"] == contract_hash(config)
assert config["aggregation_weight"] == "original_training"
assert config["model"]["family"] == "character_cnn" and not config["evaluate_final_test"]
assert metadata["initial_global_hash"] != metadata["global_hash"]
previous = metadata["initial_global_hash"]
for rnd in (1, 2):
    report = json.loads((output / "server" / f"round_{rnd:04d}.json").read_text())
    assert [row["client_id"] for row in report["clients"]] == list(range(5))
    for row in report["clients"]:
        assert row["base_hash"] == previous and row["round"] == rnd
        assert row["original_train"] == row["aggregation_weight"] == 12
        assert row["used_adult"] == row["used_non_adult"] == row["client_id"] + 2
        assert row["resampled_train"] == row["examples_processed"] == 2*row["used_adult"]
        assert row["duplicated"] == 0 and row["ensemble_members"] == 1
        assert not {"urls", "labels", "source_rows", "predictions"} & set(row)
    assert report["val"]["n_records"] == 20 and "test" not in report
    assert sum(map(sum, report["val"]["confusion_matrix"])) == 20
    previous = report["global_hash"]
assert previous == metadata["global_hash"]
with np.load(output / "server/global_final.npz", allow_pickle=False) as saved:
    arrays = [saved[key] for key in sorted(saved.files)]
    assert state_hash(arrays) == metadata["global_hash"]
    assert len(arrays) == len(metadata["schema"])
    for array, (_, name, shape, dtype) in zip(arrays, metadata["schema"]):
        assert array.shape == tuple(shape) and array.dtype.str == dtype
        assert np.isfinite(array).all()
        if name.endswith("num_batches_tracked"):
            assert int(array) == 0
for cid in range(5):
    directory = output / "synthetic-clients" / f"client_{cid:03d}" / "splits"
    manifest = json.loads((directory / "manifest.json").read_text())
    for split in ("train", "val", "test"):
        assert _file_hash(directory / f"{split}.jsonl") == manifest["splits"][split]["sha256"]
reproduction = json.loads((output / "reproduction.json").read_text())
for name, digest in reproduction["source_files_sha256"].items():
    assert _file_hash(output / "code_snapshot" / name) == digest
assert _file_hash(output / "neural-runtime-requirements.txt") == reproduction["neural_runtime_requirements_sha256"]
verification = {"status": "PASS", "scope": "synthetic actual-CNN Flower smoke, not real-data performance",
                "n_clients": 5, "rounds": 2, "local_fit_calls": 10,
                "validation_calls": 10, "unchanged_local_split_files": 15,
                "weighting_policy": "original_training", "final_hash": metadata["global_hash"],
                "checks": {"global_model_changes": True, "round_hash_chain": True,
                           "class_balance_training_only": True, "no_test_evaluation": True,
                           "only_coarse_statistics_in_round_reports": True,
                           "checkpoint_schema_and_hash": True, "integer_buffers_preserved": True,
                           "source_and_runtime_snapshots": True}}
with (output / "verification.json").open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(verification, indent=2) + "\n")
print(json.dumps(verification, indent=2))

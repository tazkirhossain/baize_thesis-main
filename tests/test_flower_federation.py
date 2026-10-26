"""Actual Flower round orchestration and client-local neural training invariants."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
fl = pytest.importorskip("flwr")
from flwr.common import FitRes, Status, Code, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import SimpleClientManager

from scripts.federated import synthetic_clients
from src.client.flower_client import URLFlowerClient
from src.client.neural_data import make_collator
from src.evaluation.federated import aggregate_validation
from src.models.federated_state import StateCodec, state_hash
from src.server.config import contract_hash, validate_config
from src.server.flower_strategy import FiveClientFedAvg
from src.server.local_transport import run_local

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def configuration():
    cfg = json.loads((ROOT / "config/federated/character_cnn.json").read_text())
    cfg.update(rounds=2, clients_per_round=3, batch_size=4)
    cfg["model"]["sequence_length"] = 16
    return cfg


class TinyCNN(torch.nn.Module):
    def __init__(self, characters):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(characters)
        self.fc = torch.nn.Linear(characters, 2)

    def forward(self, x):
        return self.fc(self.bn(x.transpose(1, 2)).mean(dim=2))


def tiny_models(cfg):
    count = cfg["balancing"]["ensemble_members"] if cfg["balancing"]["method"] == "ensemble_undersampling" else 1
    torch.manual_seed(cfg["seed"])
    return [TinyCNN(len(cfg["model"]["characters"]) + 1) for _ in range(count)]


def endpoints(cfg, root):
    return [URLFlowerClient(cid, root / f"client_{cid:03d}", cfg, models=tiny_models(cfg)) for cid in range(5)]


def test_actual_flower_rounds_repeat_and_keep_holdouts_local(tmp_path, monkeypatch):
    import src.client.flower_client as module
    synthetic_clients(tmp_path / "clients")
    before = {p: p.read_bytes() for p in (tmp_path / "clients").rglob("*.jsonl")}
    calls, original = [], module.read_local_split

    def tracked(directory, split):
        calls.append((Path(directory).name, split))
        return original(directory, split)

    monkeypatch.setattr(module, "read_local_split", tracked)
    cfg = configuration()
    fingerprints = []
    for repeat in range(2):
        strategy = FiveClientFedAvg(cfg, tiny_models(cfg), output=tmp_path / f"run{repeat}")
        initial = state_hash(strategy.current)
        _, history, _ = run_local(strategy, endpoints(cfg, tmp_path / "clients"))
        assert len(history.losses_distributed) == 2
        assert state_hash(strategy.current) != initial
        fingerprints.append(state_hash(strategy.current))
        assert len(strategy.round_reports) == 2
        for report in strategy.round_reports.values():
            assert len(report["clients"]) == 3 and report["val"]["n_records"] == 20
            assert report["train"]["n_records"] == 60
            assert all("non_adult_f1" in report[split] for split in ("train", "val"))
            assert report["timing"]["client_training_seconds_sum"] >= 0
            communication = report["communication"]
            assert communication["total_bytes"] == sum(communication[key] for key in
                ("fit_down_bytes", "fit_up_bytes", "evaluation_down_bytes", "evaluation_up_bytes"))
            assert communication["fit_down_bytes"] > 0 and communication["evaluation_up_bytes"] > 0
            assert report["weighting_policy"] == "original_training"
            for client in report["clients"]:
                assert client["aggregation_weight"] == client["original_train"] == 12
                assert client["used_adult"] == client["used_non_adult"]
                assert not any(key in client for key in ("urls", "labels", "source_rows", "predictions"))
        assert (tmp_path / f"run{repeat}/global_final.npz").is_file()
    assert fingerprints[0] == fingerprints[1]
    assert {name for name, split in calls if split == "val"} == {f"client_{i:03d}" for i in range(5)}
    assert not any(split == "test" for _, split in calls)
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("method", ["oversampling", "ensemble_undersampling"])
def test_balancing_methods_and_optional_final_test(tmp_path, method):
    cfg = configuration()
    cfg.update(rounds=1, clients_per_round=5, evaluate_final_test=True)
    cfg["balancing"].update(method=method, ensemble_members=2)
    synthetic_clients(tmp_path / "clients")
    strategy = FiveClientFedAvg(cfg, tiny_models(cfg))
    run_local(strategy, endpoints(cfg, tmp_path / "clients"))
    report = strategy.round_reports[1]
    assert report["test"]["n_records"] == report["val"]["n_records"] == 20
    for client in report["clients"]:
        assert client["used_adult"] == client["used_non_adult"]
        assert client["examples_processed"] == client["resampled_train"] * client["ensemble_members"]
        if method == "oversampling":
            assert client["used_non_adult"] == client["original_non_adult"]


def fake_round(cfg):
    strategy = FiveClientFedAvg(cfg, tiny_models(cfg))
    manager = SimpleClientManager()
    for cid in range(5):
        properties = {"client_id": cid, "contract": strategy.contract, "schema": strategy.codec.fingerprint}
        proxy = SimpleNamespace(cid=str(cid), get_properties=lambda ins, timeout, group_id, p=properties:
                                SimpleNamespace(properties=p, status=Status(Code.OK, "")))
        manager.register(proxy)
    requests = strategy.configure_fit(1, ndarrays_to_parameters(strategy.current), manager)
    results = []
    for proxy, ins in requests:
        cid, n = int(proxy.cid), int(proxy.cid) + 3
        resampled = 2*(n-1)
        metrics = {"client_id": cid, **{key: ins.config[key] for key in ("round", "contract", "schema", "base_hash")},
                   "original_train": n, "resampled_train": resampled,
                   "original_adult": 1, "original_non_adult": n-1,
                   "used_adult": n-1, "used_non_adult": n-1, "duplicated": n-2,
                   "ensemble_members": 1, "examples_processed": resampled*cfg["local_epochs"], "train_loss": 0.5,
                   "train_seconds": 0.1, "fit_seconds": 0.2}
        values = [np.full_like(array, cid+1 if array.dtype.kind == "f" else 99) for array in strategy.current]
        count = n if cfg["aggregation_weight"] == "original_training" else resampled
        results.append((proxy, FitRes(Status(Code.OK, ""), ndarrays_to_parameters(values), count, metrics)))
    return strategy, manager, results


@pytest.mark.parametrize("policy", ["original_training", "resampled_training"])
def test_weighted_tensor_average_and_integer_buffer_policy(policy):
    cfg = configuration()
    cfg.update(clients_per_round=5, aggregation_weight=policy)
    cfg["balancing"]["method"] = "oversampling"
    strategy, _, results = fake_round(cfg)
    original = strategy.current
    arrays, _ = strategy.aggregate_fit(1, results[::-1], [])
    arrays = parameters_to_ndarrays(arrays)
    expected = sum((int(proxy.cid)+1)*response.num_examples for proxy, response in results)/sum(response.num_examples for _, response in results)
    for before, after in zip(original, arrays):
        if before.dtype.kind == "f":
            np.testing.assert_allclose(after, expected, rtol=1e-6)
        else:
            np.testing.assert_array_equal(after, before)


@pytest.mark.parametrize("mutation", ["weight", "stale", "raw", "nonfinite", "duplicate", "failure"])
def test_invalid_or_private_update_is_rejected(mutation):
    cfg = configuration()
    cfg["balancing"]["method"] = "oversampling"
    strategy, _, results = fake_round(cfg)
    failures = []
    if mutation == "weight":
        results[0][1].num_examples += 1
    elif mutation == "stale":
        results[0][1].metrics["base_hash"] = "stale"
    elif mutation == "raw":
        results[0][1].metrics["urls"] = "not permitted"
    elif mutation == "nonfinite":
        arrays = parameters_to_ndarrays(results[0][1].parameters)
        arrays[0].flat[0] = np.nan
        results[0][1].parameters = ndarrays_to_parameters(arrays)
    elif mutation == "duplicate":
        results[-1] = results[0]
    else:
        failures = [RuntimeError("client failed")]
    with pytest.raises((ValueError, RuntimeError)):
        strategy.aggregate_fit(1, results, failures)


def test_identity_config_and_local_ownership_checks(tmp_path):
    cfg = configuration()
    synthetic_clients(tmp_path / "clients")
    with pytest.raises(ValueError, match="own assigned"):
        URLFlowerClient(0, tmp_path / "clients/client_001", cfg, models=tiny_models(cfg))
    for key, value in (("n_clients", 4), ("clients_per_round", 6), ("learning_rate", float("nan"))):
        changed = deepcopy(cfg)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_config(changed)
    changed = deepcopy(cfg)
    changed["balancing"]["method"] = "smote"
    with pytest.raises(ValueError, match="SMOTE"):
        validate_config(changed)
    assert contract_hash(cfg) != contract_hash(dict(cfg, learning_rate=0.002))


def test_codec_and_fixed_encoder_reject_incompatible_inputs():
    cfg = configuration()
    codec = StateCodec(tiny_models(cfg))
    with pytest.raises(ValueError, match="tensor count"):
        codec.set(codec.get()[:-1])
    collate = make_collator({"family": "character_cnn", "characters": "ab", "sequence_length": 8})
    features, labels = collate([("a?", 1), ("b", 0)])
    assert features["x"].shape == (2, 8, 3)
    assert features["x"][0, 1, 2] == 1  # UNK
    assert features["x"][1, 1:].sum() == 0  # PAD
    assert labels.tolist() == [1, 0]
    metrics = aggregate_validation([(4, {"tn": 1, "fp": 1, "fn": 0, "tp": 2, "loss_sum": 2.0})])
    assert metrics["adult_recall"] == 1 and metrics["accuracy"] == 0.75


@pytest.mark.parametrize('optimizer', [None, {'name':'adamw','weight_decay':0.0}])
def test_deberta_forward_interface_in_actual_federation(tmp_path, optimizer):
    pytest.importorskip("transformers")
    from transformers import DebertaV2Config
    from src.models.deberta_v3_base.model import DebertaV3BaseClassifier
    cfg = json.loads((ROOT / "config/federated/deberta_v3_base.json").read_text())
    if optimizer is not None:
        cfg['optimizer'] = optimizer
    cfg.update(rounds=1, clients_per_round=2, batch_size=4)
    cfg["model"]["max_length"] = 8
    synthetic_clients(tmp_path / "clients")

    def models():
        return [DebertaV3BaseClassifier(DebertaV2Config(
            vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=4, max_position_embeddings=16, relative_attention=True,
            pos_att_type=["p2c", "c2p"], position_biased_input=False,
        ))]

    def collate(rows):
        ids = torch.tensor([[2] + [ord(char) % 60 + 4 for char in url[:6]] + [3] for url, _ in rows])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, torch.tensor([label for _, label in rows])

    strategy = FiveClientFedAvg(cfg, models())
    clients = [URLFlowerClient(cid, tmp_path / "clients" / f"client_{cid:03d}", cfg,
                               models=models(), collator=collate) for cid in range(5)]
    initial = state_hash(strategy.current)
    run_local(strategy, clients)
    assert state_hash(strategy.current) != initial
    assert strategy.round_reports[1]["val"]["n_records"] == 20

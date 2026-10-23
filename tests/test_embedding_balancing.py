"""Continuous-feature SMOTE, frozen encoder semantics and four-method FL paths."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("flwr")
from scripts.federated import synthetic_clients
from src.client.embedding_training import train_batches, train_embedding_member
from src.client.data import LocalSplit
from src.client.neural_data import LocalURLDataset
from src.client.flower_client import URLFlowerClient
from src.data.balancing import BalanceConfig
from src.data.embedding_smote import EmbeddingTrainingSet, smote_embeddings
from src.data.protection import mark_protected
from src.models.character_cnn import CharacterCNNConfig, CharacterLevelCNN
from src.models.deberta_v3_base.model import DebertaV3BaseClassifier
from src.server.config import validate_config
from src.server.flower_strategy import FiveClientFedAvg
from src.server.local_transport import run_local
from src.evaluation.reporting import verify_comparability

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_standard_smote_is_deterministic_interpolation_with_minority_safety():
    features = np.arange(44, dtype=np.float32).reshape(11, 4)
    y = np.array([1]*3+[0]*8)
    data = EmbeddingTrainingSet(features, y, provenance="training-sha")
    cfg = BalanceConfig(method="smote", k_neighbors=5)
    view, report = smote_embeddings(data, cfg)
    again, same = smote_embeddings(data, cfg)
    assert report == same and report["status"] == "adjusted_k"
    assert report["after"] == {"Adult": 8, "Non-adult": 8}
    assert report["synthetic_examples"] == 5 and report["effective_k_neighbors"] == 2
    for index in range(len(data), len(view)):
        position = index-len(data)
        vector, label = view[index]
        expected = features[view.bases[position]] + view.coefficients[position]*(features[view.neighbors[position]]-features[view.bases[position]])
        np.testing.assert_allclose(vector, expected)
        torch.testing.assert_close(vector, again[index][0])
        assert label == 1 and view.bases[position] != view.neighbors[position]
    one = EmbeddingTrainingSet(features[:9], np.array([1]+[0]*8), provenance="train")
    assert smote_embeddings(one, cfg)[1]["status"] == "skipped_insufficient_minority"
    assert smote_embeddings(data, replace(cfg, insufficient_minority="skip"))[1]["synthetic_examples"] == 0
    for invalid in (features.astype(int), ["https://example.org"]):
        with pytest.raises(ValueError):
            EmbeddingTrainingSet(invalid, y, provenance="train")
    with pytest.raises(ValueError, match="training"):
        EmbeddingTrainingSet(features, mark_protected(y), provenance="train")
    with pytest.raises(ValueError, match="training"):
        EmbeddingTrainingSet(features, y, provenance="train", split="test")
    with pytest.raises(TypeError):
        smote_embeddings(np.array([[101, 102]]), cfg)


@pytest.mark.parametrize("family", ["character_cnn", "deberta_v3_base"])
def test_existing_logits_equal_encoder_then_head_and_only_head_trains(family):
    from transformers import DebertaV2Config
    if family == "character_cnn":
        model = CharacterLevelCNN(CharacterCNNConfig(5, dropout_input=0.1), 2)
        features = {"x": torch.nn.functional.one_hot(torch.randint(5, (3, 16)), 5).float()}
    else:
        model = DebertaV3BaseClassifier(DebertaV2Config(vocab_size=64, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=4,
            max_position_embeddings=32, relative_attention=True, position_biased_input=False))
        features = {"input_ids": torch.randint(4, 64, (3, 8)), "attention_mask": torch.ones(3, 8, dtype=torch.long)}
    model.eval()
    with torch.no_grad():
        actual = model(**features)
        actual = actual.logits if hasattr(actual, "logits") else actual
        embedded = model.extract_features(**features)
        torch.testing.assert_close(actual, model.classify_features(embedded))
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(embedded, torch.tensor([0, 1, 1])), batch_size=3)
    train_batches(model, loader, epochs=1, learning_rate=0.01, device=torch.device("cpu"), seed=17, embeddings=True)
    changed = {key for key, value in model.state_dict().items() if not torch.equal(before[key], value)}
    assert changed
    assert all(key.startswith(("fc.", "pooler.", "classifier.")) for key in changed)


class MiniCNN(torch.nn.Module):
    def __init__(self, characters):
        super().__init__()
        self.encoder = torch.nn.Linear(characters, 8)
        self.fc = torch.nn.Linear(8, 2)

    def extract_features(self, x):
        return torch.relu(self.encoder(x.mean(dim=1)))

    def classify_features(self, values):
        return self.fc(values)

    def feature_head(self):
        return self.fc

    def forward(self, x):
        return self.classify_features(self.extract_features(x))


@pytest.mark.parametrize("family", ["character_cnn", "deberta_v3_base"])
@pytest.mark.parametrize("method", ["undersampling", "oversampling", "smote", "ensemble_undersampling"])
def test_all_four_methods_in_five_client_federation_keep_holdouts_original(tmp_path, family, method):
    from transformers import DebertaV2Config
    cfg = json.loads((ROOT/f"config/federated/embedding_balancing/{family}_{method}.json").read_text())
    cfg.update(rounds=1, batch_size=4, evaluate_final_test=True)
    cfg["balancing"]["ensemble_members"] = 2
    cfg["transport"]["chunk_bytes"] = 1024
    cfg["model"]["sequence_length" if family == "character_cnn" else "max_length"] = 16
    validate_config(cfg)
    synthetic_clients(tmp_path/"clients")
    before = {p: p.read_bytes() for p in (tmp_path/"clients").rglob("*.jsonl")}

    def models():
        count = 2 if method == "ensemble_undersampling" else 1
        if family == "character_cnn":
            return [MiniCNN(len(cfg["model"]["characters"])+1) for _ in range(count)]
        return [DebertaV3BaseClassifier(DebertaV2Config(vocab_size=64, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=4,
            max_position_embeddings=32, relative_attention=True, position_biased_input=False)) for _ in range(count)]

    def collate(rows):
        ids = torch.tensor([[2]+[ord(char)%60+4 for char in url[:6]]+[3] for url, _ in rows])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, torch.tensor([label for _, label in rows])

    strategy = FiveClientFedAvg(cfg, models(), output=tmp_path/"server")
    initial = [a.copy() for a in strategy.current]
    clients = [URLFlowerClient(cid, tmp_path/f"clients/client_{cid:03d}", cfg, models=models(),
               collator=collate if family == "deberta_v3_base" else None) for cid in range(5)]
    run_local(strategy, clients)
    result = strategy.round_reports[1]
    assert result["val"]["n_records"] == result["test"]["n_records"] == 20
    assert result["train"]["n_records"] == 60
    for client in result["clients"]:
        assert client["used_adult"] == client["used_non_adult"]
        assert client["aggregation_weight"] == 12
        assert client["warmup_examples_processed"] == 12*client["ensemble_members"]
        assert client["examples_processed"] == client["warmup_examples_processed"]+client["resampled_train"]*client["ensemble_members"]
    if method == "smote":
        assert result["clients"][0]["synthetic"] == 8
    assert any(not np.array_equal(a, b) for a, b in zip(initial, strategy.current))
    assert all(path.read_bytes() == content for path, content in before.items())
    assert not list((tmp_path/"clients").rglob("training.npy"))
    assert len(list((tmp_path/"clients").rglob("round_0001_member_*.json"))) == 5*(2 if method == "ensemble_undersampling" else 1)


def test_mixed_end_to_end_and_embedding_protocols_cannot_be_pooled():
    cfg = json.loads((ROOT/"config/federated/character_cnn.json").read_text())
    embedded = deepcopy(cfg)
    embedded["representation"] = {"mode": "encoder_embeddings", "warmup_epochs": 1}
    base = {"scope": "federated global model", "family": "character_cnn", "data_kind": "real",
            "seed": 42, "partition": "iid", "method": "undersampling", "split": "val", "round": 1, "experiment_id": "a"}
    with pytest.raises(ValueError, match="incompatible"):
        verify_comparability([base, dict(base, method="smote", experiment_id="b")],
            {"a": {"configuration": cfg}, "b": {"configuration": embedded}})


def test_holdout_embedding_warmup_is_rejected_before_any_weight_change(tmp_path):
    data = LocalSplit("val", ("https://example.org", "https://example.net"), mark_protected([0, 1]), np.arange(2), "val")
    model = MiniCNN(5)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="training"):
        train_embedding_member(model, LocalURLDataset(data), data, {}, BalanceConfig(method="smote"),
            collate=None, device=torch.device("cpu"), seed=42, selection=None, cache_root=tmp_path/"cache")
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert not (tmp_path/"cache").exists()

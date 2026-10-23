"""Opt-in real localhost gRPC test (no sockets needed by the default suite)."""
import json
import importlib
import os
from pathlib import Path
import socket
import threading

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("BAIZE_TEST_FLOWER_NETWORK") != "1",
                                reason="opt in to real loopback sockets")
fl = pytest.importorskip("flwr")
torch = pytest.importorskip("torch")
from flwr.server.client_manager import SimpleClientManager
from scripts.federated import synthetic_clients, MAX_MESSAGE
from src.client.flower_client import URLFlowerClient
from src.server.flower_strategy import FederatedServer, FiveClientFedAvg


@pytest.mark.parametrize("family,method", [("character_cnn", "undersampling"), ("deberta_v3_base", "ensemble_undersampling"), ("deberta_v3_base", "smote")])
def test_server_five_clients_over_grpc(tmp_path, monkeypatch, family, method):
    # Flower normally runs the server in the main process. Only this threaded
    # test bypasses signal registration; CLI server behavior is unchanged.
    monkeypatch.setattr(importlib.import_module("flwr.compat.server.app"),
                        "register_signal_handlers", lambda **kwargs: None)
    cfg = json.loads((Path(__file__).resolve().parents[1] / f"config/federated/{family}.json").read_text())
    cfg.update(rounds=1, batch_size=8)
    cfg["model"]["sequence_length" if family == "character_cnn" else "max_length"] = 16
    cfg["balancing"].update(method=method, ensemble_members=2)
    cfg["transport"] = {"chunk_bytes": 1024, "max_transfer_bytes": 1024*1024}
    if method == "smote":
        cfg["representation"] = {"mode": "encoder_embeddings", "warmup_epochs": 1}
    synthetic_clients(tmp_path / "clients")

    class NetworkModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(len(cfg["model"]["characters"]) + 1, 2)

        def forward(self, x):
            return self.linear(x.mean(dim=1))

    def models():
        if family == "character_cnn":
            return [NetworkModel()]
        from transformers import DebertaV2Config
        from src.models.deberta_v3_base.model import DebertaV3BaseClassifier
        return [DebertaV3BaseClassifier(DebertaV2Config(vocab_size=64, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=4,
            max_position_embeddings=32, relative_attention=True, position_biased_input=False))
            for _ in range(2 if method == "ensemble_undersampling" else 1)]

    def collate(rows):
        ids = torch.tensor([[2]+[ord(c)%60+4 for c in url[:6]]+[3] for url, _ in rows])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, torch.tensor([y for _, y in rows])

    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    address = f"127.0.0.1:{port}"
    strategy = FiveClientFedAvg(cfg, models(), output=tmp_path / "server")
    server = FederatedServer(client_manager=SimpleClientManager(), strategy=strategy)
    server.max_workers = 1
    errors = []

    def serve():
        try:
            fl.server.start_server(server_address=address, server=server,
                                   config=fl.server.ServerConfig(num_rounds=1), grpc_max_message_length=8192)
        except Exception as error:
            errors.append(error)

    def connect(cid):
        try:
            endpoint = URLFlowerClient(cid, tmp_path / "clients" / f"client_{cid:03d}", cfg,
                                       models=models(), collator=collate if family == "deberta_v3_base" else None)
            fl.client.start_client(server_address=address, client=endpoint.to_client(),
                                   insecure=True, grpc_max_message_length=8192,
                                   max_retries=2, max_wait_time=30)
        except Exception as error:
            errors.append(error)

    workers = [threading.Thread(target=serve, daemon=True)]
    workers += [threading.Thread(target=connect, args=(cid,), daemon=True) for cid in range(5)]
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=45)
        assert not errors, errors
        assert not any(worker.is_alive() for worker in workers)
        assert len(strategy.round_reports[1]["clients"]) == 5
        assert strategy.round_reports[1]["val"]["n_records"] == 20
        assert (tmp_path / "server/global_final.npz").is_file()
        if family == "deberta_v3_base":
            # Full states exceed this gRPC limit; successful fit proves chunk use.
            assert sum(a.nbytes for a in strategy.current) > 8192
    finally:
        torch.set_num_threads(previous)

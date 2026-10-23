"""Flower neural server/client entry points plus a reproducible local harness."""
import argparse
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import flwr as fl
import torch
from flwr.server.client_manager import SimpleClientManager

from src.data.preprocessing import _file_hash, _snapshot_reproduction
from src.client.flower_client import URLFlowerClient
from src.models.federated_state import make_models
from src.server.config import validate_config
from src.server.flower_strategy import FederatedServer, FiveClientFedAvg
from src.server.local_transport import run_local

MAX_MESSAGE = 1024 * 1024 * 1024


def synthetic_clients(root):
    """Tiny, explicitly synthetic splits for exercising the actual CNN/FL stack."""
    for cid in range(5):
        directory = root / f"client_{cid:03d}"
        splits = directory / "splits"
        splits.mkdir(parents=True, exist_ok=False)
        (directory / "manifest.json").write_text(json.dumps({"client_id": cid, "synthetic": True}))
        settings = splits / "preprocessing.json"
        settings.write_text(json.dumps({"synthetic": True}))
        manifest = {"preprocessing_sha256": _file_hash(settings), "splits": {}}
        for index, split in enumerate(("train", "val", "test")):
            labels = [1] * (cid + 2) + [0] * (10 - cid) if split == "train" else [0, 1, 0, 1]
            path = splits / f"{split}.jsonl"
            with path.open("x", encoding="utf-8") as stream:
                for row, label in enumerate(labels):
                    stream.write(json.dumps({"url": f"https://c{cid}-{split}-{row}.example/{'adult' if label else 'news'}",
                                             "label": label, "source_row": cid*1000+index*100+row}) + "\n")
            manifest["splits"][split] = {"n_records": len(labels), "sha256": _file_hash(path)}
        (splits / "manifest.json").write_text(json.dumps(manifest))


def snapshot(output):
    record = _snapshot_reproduction(output)
    for relative in ("scripts/federated.py", "config/federated/character_cnn.json", "config/federated/deberta_v3_base.json"):
        source, target = ROOT / relative, output / "code_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        record["source_files_sha256"][relative] = _file_hash(target)
    # The preparation helper's original tree hash precedes the extra FL files.
    import hashlib
    record["source_tree_sha256"] = hashlib.sha256(json.dumps(record["source_files_sha256"], sort_keys=True).encode()).hexdigest()
    record["runtime"] = {name: importlib.metadata.version(name) for name in ("torch", "flwr", "numpy", "protobuf", "grpcio")}
    pending = ["torch", "flwr"]
    for optional in ("transformers", "sentencepiece"):
        try:
            importlib.metadata.version(optional)
            pending.append(optional)
        except importlib.metadata.PackageNotFoundError:
            pass
    installed = {}
    while pending:
        distribution = importlib.metadata.distribution(pending.pop())
        name = distribution.metadata["Name"]
        if name.lower() in installed:
            continue
        installed[name.lower()] = (name, distribution.version)
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    runtime_path = output / "neural-runtime-requirements.txt"
    runtime_path.write_text("\n".join(f"{name}=={version}" for name, version in sorted(installed.values())) + "\n")
    record["neural_runtime_requirements_sha256"] = _file_hash(runtime_path)
    record["python"] = platform.python_version()
    (output / "reproduction.json").write_text(json.dumps(record, indent=2) + "\n")


def local_experiment(cfg, partition, output, *, device="cpu"):
    strategy = FiveClientFedAvg(cfg, make_models(cfg), output=output / "server")
    endpoints = [URLFlowerClient(cid, partition / f"client_{cid:03d}", cfg, device=device) for cid in range(5)]
    return run_local(strategy, endpoints)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for name in ("server", "client", "local", "smoke"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=ROOT / "config/federated/character_cnn.json")
        command.add_argument("--threads", type=int, default=2)
        if name != "client":
            command.add_argument("--output", type=Path, required=True)
        if name in ("server", "client"):
            command.add_argument("--address", default="127.0.0.1:8080")
            command.add_argument("--ca", type=Path)
        if name == "server":
            command.add_argument("--cert", type=Path)
            command.add_argument("--key", type=Path)
        if name in ("client", "local"):
            command.add_argument("--device", default="cpu")
        if name == "client":
            command.add_argument("--client-id", type=int, required=True)
            command.add_argument("--client-dir", type=Path, required=True)
        if name == "local":
            command.add_argument("--partition-root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cfg = validate_config(json.loads(args.config.read_text()))
    if args.mode in ("server", "client"):
        local_address = args.address.startswith(("127.0.0.1:", "localhost:", "[::1]:"))
        if not local_address and args.ca is None:
            parser.error("remote connections require TLS --ca (and server --cert/--key)")
    if args.mode == "client":
        endpoint = URLFlowerClient(args.client_id, args.client_dir, cfg, device=args.device)
        fl.client.start_client(server_address=args.address, client=endpoint.to_client(),
                               insecure=args.ca is None, root_certificates=args.ca.read_bytes() if args.ca else None,
                               grpc_max_message_length=MAX_MESSAGE)
        return
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "data"):
        parser.error("save checkpoints and local run artifacts under ignored data/")
    output.mkdir(parents=True, exist_ok=False)
    if args.mode == "smoke":
        if cfg["model"]["family"] != "character_cnn":
            parser.error("synthetic smoke uses the actual character CNN")
        cfg = deepcopy(cfg)
        cfg.update(rounds=2, batch_size=8)
        cfg["experiment"] = {"partition_strategy": "unknown", "data_kind": "synthetic",
                             "evaluation_population_id": "synthetic-five-client-smoke-v1"}
        cfg["model"]["sequence_length"] = 32
        synthetic_clients(output / "synthetic-clients")
    snapshot(output)
    (output / "experiment.json").write_text(json.dumps(cfg, indent=2) + "\n")
    if args.mode in ("local", "smoke"):
        partition = args.partition_root if args.mode == "local" else output / "synthetic-clients"
        _, _, seconds = local_experiment(cfg, partition, output, device=getattr(args, "device", "cpu"))
        print(f"Completed {cfg['rounds']} Flower rounds, five clients, in {seconds:.2f}s; artifacts: {output}")
    else:
        if bool(args.ca) != bool(args.cert) or bool(args.ca) != bool(args.key):
            parser.error("provide all three server TLS files together")
        strategy = FiveClientFedAvg(cfg, make_models(cfg), output=output / "server")
        server = FederatedServer(client_manager=SimpleClientManager(), strategy=strategy)
        server.max_workers = 1
        certificates = (args.ca.read_bytes(), args.cert.read_bytes(), args.key.read_bytes()) if args.ca else None
        fl.server.start_server(server_address=args.address, server=server,
                               config=fl.server.ServerConfig(num_rounds=cfg["rounds"]),
                               certificates=certificates, grpc_max_message_length=MAX_MESSAGE)


if __name__ == "__main__":
    main()

"""Five-client neural training endpoint; only tensors and aggregate scalars leave."""
from copy import deepcopy
from dataclasses import replace
import json
import math
import time
from pathlib import Path

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader

from baize.data.preprocessing import _file_hash
from src.client.data import read_local_split
from src.client.neural_data import LocalURLDataset, make_collator
from src.data.balancing import (BalanceConfig, ensemble_training_indices,
                               print_balance_report, sample_training_indices)
from src.models.federated_state import StateCodec, make_models, state_hash
from src.server.config import contract_hash


class URLFlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, directory, cfg, *, device="cpu", models=None, collator=None):
        self.contract = contract_hash(cfg)
        if type(client_id) is not int or client_id not in range(5):
            raise ValueError("client storage ID must be 0-4")
        self.directory, self.client_id, self.cfg = Path(directory), client_id, deepcopy(cfg)
        if json.loads((self.directory / "manifest.json").read_text())["client_id"] != client_id:
            raise ValueError("the client must use its own assigned data directory")
        self.models = models if models is not None else make_models(cfg)
        expected_members = cfg["balancing"].get("ensemble_members", 5) if cfg["balancing"]["method"] == "ensemble_undersampling" else 1
        if len(self.models) != expected_members:
            raise ValueError("ensemble member count mismatch")
        self.device = torch.device(device)
        self.codec = StateCodec(self.models)
        self.collate = collator if collator is not None else make_collator(cfg["model"])
        self.last_fit_round = 0
        self.row_ids = {}

    def get_properties(self, config):
        manifest = json.loads((self.directory / "splits/manifest.json").read_text())
        preprocessing = json.loads((self.directory / "splits/preprocessing.json").read_text())
        synthetic = json.loads((self.directory / "manifest.json").read_text()).get("synthetic", False)
        return {"client_id": self.client_id, "contract": self.contract,
                "schema": self.codec.fingerprint,
                **{f"{name}_sha256": manifest["splits"][name]["sha256"] for name in ("train", "val", "test")},
                "partition_strategy": preprocessing.get("partitioning", {}).get("strategy", "unknown"),
                "data_kind": "synthetic" if synthetic else "real"}

    def get_parameters(self, config):
        return self.codec.get()

    def _check(self, parameters, config):
        if config["contract"] != self.contract or config["schema"] != self.codec.fingerprint:
            raise ValueError("server/client experiment or model schema mismatch")
        if state_hash(parameters) != config["base_hash"]:
            raise ValueError("global model checksum mismatch")
        self.codec.set(parameters)

    def _read(self, split):
        data = read_local_split(self.directory, split)
        rows = set(int(row) for row in data.source_rows)
        if len(rows) != len(data):
            raise ValueError("duplicate source rows inside a local split")
        if split != "train" and "train" not in self.row_ids:
            training = read_local_split(self.directory, "train")
            self.row_ids["train"] = set(int(row) for row in training.source_rows)
        if any(rows & other for name, other in self.row_ids.items() if name != split):
            raise ValueError("client-local training/holdout source-row leakage")
        self.row_ids[split] = rows
        return data, LocalURLDataset(data)

    def _unchanged(self, split, digest):
        if _file_hash(self.directory / "splits" / f"{split}.jsonl") != digest:
            raise ValueError("local split changed during training/evaluation")

    @staticmethod
    def _logits(model, features):
        output = model(**features)
        return output.logits if hasattr(output, "logits") else output

    def fit(self, parameters, config):
        fit_started = time.perf_counter()
        self._check(parameters, config)
        rnd = config["round"]
        if type(rnd) is not int or rnd <= self.last_fit_round:
            raise ValueError("training round is stale or repeated")
        for key in ("local_epochs", "batch_size", "learning_rate"):
            if config[key] != self.cfg[key]:
                raise ValueError("local training settings differ from the agreed contract")
        data, dataset = self._read("train")
        seed = int(np.random.SeedSequence([self.cfg["balancing"]["seed"], self.client_id, rnd]).generate_state(1)[0])
        balance = replace(BalanceConfig(**self.cfg["balancing"]), seed=seed)
        selections = (ensemble_training_indices(data.labels, balance, split="train")
                      if balance.method == "ensemble_undersampling" else
                      [sample_training_indices(data.labels, balance, split="train")])
        loss_sum, seen = 0.0, 0
        counts = []
        training_started = time.perf_counter()
        for member, (model, selection) in enumerate(zip(self.models, selections)):
            if selection.report["status"].startswith("skipped"):
                raise ValueError("training balancing skipped: local training class is missing")
            print(f"Client {self.client_id+1}, round {rnd}, member {member+1}", flush=True)
            print_balance_report(selection.report)
            view = selection.apply(dataset)
            generator = torch.Generator().manual_seed((seed + member) % 2**32)
            loader = DataLoader(view, batch_size=self.cfg["batch_size"], shuffle=True,
                                generator=generator, collate_fn=self.collate, num_workers=0)
            model.to(self.device).train()
            # Plain SGD, reset each round. No class-weight/focal replacement.
            optimizer = torch.optim.SGD(model.parameters(), lr=self.cfg["learning_rate"])
            with torch.random.fork_rng(devices=[] if self.device.type == "cpu" else [self.device]):
                torch.manual_seed((seed + member) % 2**32)
                for _ in range(self.cfg["local_epochs"]):
                    for features, labels in loader:
                        features = {key: value.to(self.device) for key, value in features.items()}
                        labels = labels.to(self.device)
                        optimizer.zero_grad(set_to_none=True)
                        loss = torch.nn.functional.cross_entropy(self._logits(model, features), labels)
                        if not torch.isfinite(loss):
                            raise ValueError("nonfinite local training loss")
                        loss.backward()
                        optimizer.step()
                        loss_sum += float(loss.detach()) * len(labels)
                        seen += len(labels)
            model.cpu()
            counts.append(len(view))
        train_seconds = time.perf_counter() - training_started
        self._unchanged("train", data.sha256)
        self.last_fit_round = rnd
        arrays = self.codec.get()
        self.codec.validate(arrays)
        weight = len(data) if self.cfg["aggregation_weight"] == "original_training" else counts[0]
        if len(set(counts)) != 1:
            raise ValueError("ensemble members must have equal resampled counts")
        metrics = {"client_id": self.client_id, "round": rnd, "contract": self.contract,
                   "schema": self.codec.fingerprint, "base_hash": config["base_hash"],
                   "original_train": len(data), "resampled_train": counts[0],
                   "original_adult": int(np.count_nonzero(data.labels == 1)),
                   "original_non_adult": int(np.count_nonzero(data.labels == 0)),
                   "used_adult": selections[0].report["after"]["Adult"],
                   "used_non_adult": selections[0].report["after"]["Non-adult"],
                   "duplicated": selections[0].report["duplicated_examples"],
                   "ensemble_members": len(self.models), "examples_processed": seen,
                   "train_loss": loss_sum / seen, "train_seconds": train_seconds,
                   "fit_seconds": time.perf_counter() - fit_started}
        return arrays, weight, metrics

    def evaluate(self, parameters, config):
        self._check(parameters, config)
        split = config["split"]
        if split not in ("val", "test") or (split == "test" and not self.cfg["evaluate_final_test"]):
            raise ValueError("only agreed holdout evaluation is allowed")
        loss, count, metrics = self._evaluate_split(split)
        if split == "val" and self.cfg.get("evaluate_training", True):
            _, train_count, training = self._evaluate_split("train")
            metrics.update({"train_n_records": train_count,
                            **{f"train_{key}": value for key, value in training.items()}})
        metrics.update({"client_id": self.client_id, "contract": self.contract,
                        "schema": self.codec.fingerprint, "round": config["round"],
                        "base_hash": config["base_hash"], "split": split})
        return loss, count, metrics

    def _evaluate_split(self, split):
        started = time.perf_counter()
        data, dataset = self._read(split)
        loader = DataLoader(dataset, batch_size=self.cfg["batch_size"], shuffle=False,
                            collate_fn=self.collate, num_workers=0)
        # Counts and source-row disjointness are checked within the client only.
        manifest = json.loads((self.directory / "splits/manifest.json").read_text())
        if len(data) != manifest["splits"][split]["n_records"]:
            raise ValueError("holdout count mismatch")
        matrix = np.zeros((2, 2), dtype=np.int64)
        loss_sum = 0.0
        for model in self.models:
            model.to(self.device).eval()
        with torch.no_grad():
            for features, labels in loader:
                features = {key: value.to(self.device) for key, value in features.items()}
                probabilities = torch.stack([
                    self._logits(model, features).softmax(dim=1) for model in self.models
                ]).mean(dim=0).cpu()
                if not torch.isfinite(probabilities).all():
                    raise ValueError("nonfinite evaluation probabilities")
                loss_sum += float(torch.nn.functional.nll_loss(
                    probabilities.clamp_min(1e-12).log(), labels, reduction="sum"))
                predicted = (probabilities[:, 1] >= self.cfg["threshold"]).long()
                matrix += np.bincount((2*labels + predicted).numpy(), minlength=4).reshape(2, 2)
        for model in self.models:
            model.cpu()
        self._unchanged(split, data.sha256)
        metrics = dict(zip(("tn", "fp", "fn", "tp"), [int(x) for x in matrix.ravel()]))
        metrics.update({"loss_sum": loss_sum, "evaluation_seconds": time.perf_counter() - started})
        return loss_sum / len(data), len(data), metrics

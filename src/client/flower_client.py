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

from src.data.preprocessing import _file_hash
from src.client.data import read_local_split
from src.client.neural_data import LocalURLDataset, make_collator
from src.client.progress import BatchProgress
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
        self.enrichment = None
        if cfg.get('website_enrichment'):
            from src.data.enrichment.store import EnrichmentReader
            self.enrichment = EnrichmentReader(self.directory, cfg['website_enrichment'])
        self.collate = collator if collator is not None else make_collator(cfg["model"], cfg.get('website_enrichment'))
        self.last_fit_round = 0
        self.row_ids = {}
        self.selection_membership = None
        self.selection_summary = None

    def to_client(self):
        from src.transport import ChunkedClient
        return ChunkedClient(super().to_client(), contract=self.contract,
            directory=self.directory / ".transport", limits=self.cfg.get("transport"))

    def get_properties(self, config):
        manifest = json.loads((self.directory / "splits/manifest.json").read_text())
        preprocessing = json.loads((self.directory / "splits/preprocessing.json").read_text())
        synthetic = json.loads((self.directory / "manifest.json").read_text()).get("synthetic", False)
        properties = {"client_id": self.client_id, "contract": self.contract,
                "schema": self.codec.fingerprint,
                "device": str(self.device), "torch_threads": torch.get_num_threads(),
                "cuda_runtime": torch.version.cuda or "none",
                **{f"{name}_sha256": manifest["splits"][name]["sha256"] for name in ("train", "val", "test")},
                "partition_strategy": preprocessing.get("partitioning", {}).get("strategy", "unknown"),
                "data_kind": "synthetic" if synthetic else "real"}
        if self.enrichment:
            manifest = self.enrichment.manifest
            properties.update(enrichment_digest=manifest['input_digest'], enrichment_n_records=manifest['n_records'],
                              enrichment_database_sha256=manifest['database_sha256'])
        if self.cfg.get('model_selection'):
            if self.selection_summary is None:
                data, _ = self._read('train')
                self._training_views(data)
            properties.update({f'{name}_membership_sha256': values['membership_sha256']
                               for name, values in self.selection_summary.items() if isinstance(values, dict)})
        return properties

    def _training_views(self, data):
        from src.client.selection import selection_mask, selection_views, membership_summary
        if self.selection_membership is None:
            synthetic = json.loads((self.directory/'manifest.json').read_text()).get('synthetic', False)
            mask = selection_mask(data, fraction=self.cfg['model_selection']['fraction'],
                                  seed=self.cfg['seed'], synthetic=synthetic)
            self.selection_membership = (data.sha256, mask)
        digest, mask = self.selection_membership
        if digest != data.sha256:
            raise ValueError('original training membership changed during model selection')
        training, selection = selection_views(data, mask)
        if self.selection_summary is None:
            from src.server.checkpoints import atomic_json
            self.selection_summary = membership_summary(training, selection)
            self.selection_summary.update(fraction=self.cfg['model_selection']['fraction'], seed=self.cfg['seed'],
                                          parent_train_sha256=data.sha256, contract=self.contract)
            audit = self.directory/'.runtime'/f'training-selection-{self.contract}.json'
            audit.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(audit, self.selection_summary)
            print(f'Client {self.client_id+1} training-only selection: assigned={len(data):,}; '
                  f'optimization={len(training):,}; selection={len(selection):,}. '
                  'Original validation/test unchanged.', flush=True)
        return training, selection

    def get_parameters(self, config):
        return self.codec.get()

    def _check(self, parameters, config):
        if config["contract"] != self.contract or config["schema"] != self.codec.fingerprint:
            raise ValueError("server/client experiment or model schema mismatch")
        if state_hash(parameters) != config["base_hash"]:
            raise ValueError("global model checksum mismatch")
        self.codec.set(parameters)

    def _read(self, split):
        if self.enrichment:
            self.enrichment.assert_unchanged()
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
        return data, LocalURLDataset(data, self.enrichment)

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
        assigned_count = len(data)
        selection_count = 0
        if self.cfg.get('model_selection'):
            data, held = self._training_views(data)
            selection_count = len(held)
            dataset = LocalURLDataset(data, self.enrichment)
        seed = int(np.random.SeedSequence([self.cfg["balancing"]["seed"], self.client_id, rnd]).generate_state(1)[0])
        balance = replace(BalanceConfig(**self.cfg["balancing"]), seed=seed)
        selections = ([None] if balance.method == "smote" else ensemble_training_indices(data.labels, balance, split="train")
                      if balance.method == "ensemble_undersampling" else
                      [sample_training_indices(data.labels, balance, split="train")])
        loss_sum, seen = 0.0, 0
        counts, reports, warmup_seen = [], [], 0
        training_started = time.perf_counter()
        for member, (model, selection) in enumerate(zip(self.models, selections)):
            if self.cfg.get("representation"):
                from src.client.embedding_training import train_embedding_member
                print(f"Client {self.client_id+1}, round {rnd}, member {member+1}", flush=True)
                loss, n, report, warm = train_embedding_member(model, dataset, data, self.cfg, balance,
                    collate=self.collate, device=self.device, seed=(seed+member) % 2**32,
                    selection=selection, cache_root=self.directory / ".runtime",
                    progress_label=f'Client {self.client_id+1} | round {rnd} | member {member+1}')
                audit = self.directory / ".runtime" / f"round_{rnd:04d}_member_{member:03d}.json"
                audit.write_text(json.dumps(dict(report, round=rnd, member=member,
                    warmup_examples_processed=warm), indent=2, allow_nan=False) + "\n")
                loss_sum += loss
                seen += n
                warmup_seen += warm
                counts.append(sum(report["after"].values()))
                reports.append(report)
                continue
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
            progress = BatchProgress(f'Client {self.client_id+1} | round {rnd} | member {member+1} | training view',
                total_batches=len(loader)*self.cfg['local_epochs'], total_examples=len(view)*self.cfg['local_epochs'],
                epochs=self.cfg['local_epochs'], interval_seconds=self.cfg.get('progress_interval_seconds', 30))
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
                        batch_loss = float(loss.detach())
                        loss_sum += batch_loss * len(labels)
                        seen += len(labels)
                        progress.update(len(labels), loss=batch_loss)
            model.cpu()
            counts.append(len(view))
            reports.append(selection.report)
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
                   "used_adult": reports[0]["after"]["Adult"],
                   "used_non_adult": reports[0]["after"]["Non-adult"],
                   "duplicated": reports[0]["duplicated_examples"],
                   "ensemble_members": len(self.models), "examples_processed": seen,
                   "train_loss": loss_sum / seen, "train_seconds": train_seconds,
                   "fit_seconds": time.perf_counter() - fit_started}
        if self.cfg.get("representation"):
            metrics.update({"warmup_examples_processed": warmup_seen,
                "synthetic": reports[0]["synthetic_examples"],
                "smote_effective_k": reports[0].get("effective_k_neighbors", 0),
                "balancing_status": reports[0]["status"],
                "feature_space": "encoder_embeddings"})
        if self.cfg.get('model_selection'):
            metrics.update(assigned_train=assigned_count, selection_train=selection_count)
        return arrays, weight, metrics

    def evaluate(self, parameters, config):
        self._check(parameters, config)
        split = config["split"]
        if split not in ("val", "test") or (split == "test" and not self.cfg["evaluate_final_test"]):
            raise ValueError("only agreed holdout evaluation is allowed")
        loss, count, metrics = self._evaluate_split(split, round_number=config['round'])
        if split == "val" and self.cfg.get("evaluate_training", True):
            _, train_count, training = self._evaluate_split("train", round_number=config['round'])
            metrics.update({"train_n_records": train_count,
                            **{f"train_{key}": value for key, value in training.items()}})
        if split == 'val' and self.cfg.get('model_selection'):
            _, selection_count, held = self._evaluate_split('selection', round_number=config['round'])
            metrics.update(selection_n_records=selection_count,
                           **{f'selection_{key}': value for key, value in held.items()})
        metrics.update({"client_id": self.client_id, "contract": self.contract,
                        "schema": self.codec.fingerprint, "round": config["round"],
                        "base_hash": config["base_hash"], "split": split})
        return loss, count, metrics

    def _evaluate_split(self, split, *, round_number=None):
        started = time.perf_counter()
        parent = 'train' if split == 'selection' else split
        data, dataset = self._read(parent)
        # Counts and source-row disjointness are checked within the client only.
        manifest = json.loads((self.directory / "splits/manifest.json").read_text())
        if len(data) != manifest["splits"][parent]["n_records"]:
            raise ValueError("holdout count mismatch")
        if self.cfg.get('model_selection') and split in ('train', 'selection'):
            views = self._training_views(data)
            data = views[1 if split == 'selection' else 0]
            dataset = LocalURLDataset(data, self.enrichment)
        # Empty local selection is legitimate for a small/domain-grouped client.
        # No fallback, replacement, seed search or evaluation-pool migration.
        if not len(data):
            if split != 'selection':
                raise ValueError('empty local evaluation pool')
            self._unchanged(parent, data.sha256)
            return 0.0, 0, dict(tn=0, fp=0, fn=0, tp=0, loss_sum=0.0,
                                evaluation_seconds=time.perf_counter()-started)
        loader = DataLoader(dataset, batch_size=self.cfg['batch_size'], shuffle=False,
                            collate_fn=self.collate, num_workers=0)
        matrix = np.zeros((2, 2), dtype=np.int64)
        loss_sum = 0.0
        for model in self.models:
            model.to(self.device).eval()
        progress = BatchProgress(f'Client {self.client_id+1} | round {round_number or "?"} | original {split} evaluation',
            total_batches=len(loader), total_examples=len(data), interval_seconds=self.cfg.get('progress_interval_seconds', 30))
        with torch.no_grad():
            for features, labels in loader:
                features = {key: value.to(self.device) for key, value in features.items()}
                probabilities = torch.stack([
                    self._logits(model, features).softmax(dim=1) for model in self.models
                ]).mean(dim=0).cpu()
                if not torch.isfinite(probabilities).all():
                    raise ValueError("nonfinite evaluation probabilities")
                batch_loss_sum = float(torch.nn.functional.nll_loss(
                    probabilities.clamp_min(1e-12).log(), labels, reduction="sum"))
                loss_sum += batch_loss_sum
                predicted = (probabilities[:, 1] >= self.cfg["threshold"]).long()
                matrix += np.bincount((2*labels + predicted).numpy(), minlength=4).reshape(2, 2)
                progress.update(len(labels), loss=batch_loss_sum/len(labels))
        for model in self.models:
            model.cpu()
        self._unchanged(parent, data.sha256)
        metrics = dict(zip(("tn", "fp", "fn", "tp"), [int(x) for x in matrix.ravel()]))
        metrics.update({"loss_sum": loss_sum, "evaluation_seconds": time.perf_counter() - started})
        return loss_sum / len(data), len(data), metrics

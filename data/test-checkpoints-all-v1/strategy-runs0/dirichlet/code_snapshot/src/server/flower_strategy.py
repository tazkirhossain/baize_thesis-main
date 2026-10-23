"""Deterministic five-client Flower FedAvg with distributed holdout evaluation."""
import json
import math
import time
import platform
from copy import deepcopy
from pathlib import Path

import flwr as fl
from flwr.common import (EvaluateIns, FitIns, GetPropertiesIns, ndarrays_to_parameters,
                         parameters_to_ndarrays)
from flwr.common.serde import (fit_ins_to_proto, fit_res_to_proto,
                              evaluate_ins_to_proto, evaluate_res_to_proto)
import numpy as np

from src.evaluation.federated import aggregate_validation
from src.models.federated_state import StateCodec, state_hash
from src.server.config import contract_hash
from src.transport import ChunkedProxy, MARKER, parameter_size, DEFAULT_LIMITS
from src.data.balancing import ratio
from src.server.checkpoints import atomic_json, save_weights, load_checkpoint, copy_best


class FiveClientFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, cfg, models, *, output=None):
        self.cfg, self.contract = deepcopy(cfg), contract_hash(cfg)
        self.codec = StateCodec(models)
        self.current = self.codec.get()
        self.initial_hash = state_hash(self.current)
        self.output = Path(output) if output else None
        if self.output:
            self.output.mkdir(parents=True, exist_ok=False)
        self.round_reports, self.original_counts = {}, {}
        self.pending_fit, self.pending_eval = {}, {}
        self.round_started, self.evaluation_started = {}, {}
        self.evaluation_native_bytes = {}
        self.communication, self.client_data_contracts = {}, {}
        self.client_runtime = {}
        self.best = None
        self.best_arrays = None  # Only used by output-less unit/local harnesses.
        super().__init__(initial_parameters=ndarrays_to_parameters(self.current),
                         fraction_fit=1, fraction_evaluate=1, min_fit_clients=cfg["clients_per_round"],
                         min_evaluate_clients=5, min_available_clients=5, accept_failures=False,
                         evaluate_fn=None)

    def _clients(self, manager):
        if not manager.wait_for(5, timeout=60):
            raise RuntimeError("all five assigned clients must be available")
        proxies = list(manager.all().values())
        if len(proxies) != 5:
            raise ValueError("exactly five clients must register")
        registry = {}
        protocols = []
        for proxy in proxies:
            response = proxy.get_properties(GetPropertiesIns({}), timeout=30, group_id=None)
            if response.status.code != fl.common.Code.OK:
                raise ValueError("client registration failed")
            properties = response.properties
            protocols.append(properties.get("parameter_transport") == MARKER)
            cid = properties.get("client_id")
            if (type(cid) is not int or cid not in range(5) or cid in registry or
                    properties.get("contract") != self.contract or
                    properties.get("schema") != self.codec.fingerprint):
                raise ValueError("client identity, experiment or model schema mismatch")
            if properties.get("parameter_transport") == MARKER:
                proxy = ChunkedProxy(proxy, contract=self.contract, limits=self.cfg.get("transport"))
            elif self.cfg["model"]["family"] == "deberta_v3_base" and self.cfg["balancing"]["method"] == "ensemble_undersampling":
                raise ValueError("DeBERTa ensembles require clients with chunked parameter transport")
            registry[cid] = proxy
            self.client_data_contracts[str(cid)] = {key: properties[key] for key in
                ("train_sha256", "val_sha256", "test_sha256", "partition_strategy", "data_kind",
                 "optimization_membership_sha256", "selection_membership_sha256")
                if key in properties}
            if self.cfg.get('model_selection'):
                for name in ('optimization', 'selection'):
                    digest = properties.get(f'{name}_membership_sha256')
                    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                        raise ValueError('client training-selection membership contract is missing')
            self.client_runtime[str(cid)] = {key: properties[key] for key in
                ("device", "torch_threads", "cuda_runtime") if key in properties}
        if any(protocols) and not all(protocols):
            raise ValueError("all clients must use the same parameter transport protocol")
        return registry

    def _message(self, server_round, parameters):
        arrays = parameters_to_ndarrays(parameters)
        self.codec.validate(arrays)
        return {"round": server_round, "contract": self.contract,
                "schema": self.codec.fingerprint, "base_hash": state_hash(arrays)}

    def configure_fit(self, server_round, parameters, client_manager):
        self.round_started[server_round] = time.perf_counter()
        clients = self._clients(client_manager)
        self.current = parameters_to_ndarrays(parameters)
        config = self._message(server_round, parameters)
        config.update({key: self.cfg[key] for key in ("local_epochs", "batch_size", "learning_rate")})
        rng = np.random.default_rng(np.random.SeedSequence([self.cfg["seed"], server_round]))
        selected = sorted(int(cid) for cid in rng.choice(5, self.cfg["clients_per_round"], replace=False))
        self.pending_fit[server_round] = ({clients[cid].cid: cid for cid in selected}, config)
        requests = [(clients[cid], FitIns(parameters, dict(config))) for cid in selected]
        self.communication[server_round] = {
            "fit_down_bytes": sum(fit_ins_to_proto(ins).ByteSize() for proxy, ins in requests
                                  if not isinstance(proxy, ChunkedProxy) or parameter_size(ins.parameters) <= DEFAULT_LIMITS["chunk_bytes"]),
            "fit_up_bytes": 0, "evaluation_down_bytes": 0, "evaluation_up_bytes": 0,
        }
        return requests

    def _validate_responses(self, server_round, results, failures, *, evaluation=False):
        pending = self.pending_eval if evaluation else self.pending_fit
        expected, config = pending[server_round]
        if failures or len(results) != len(expected):
            raise RuntimeError("round failed: missing/failed client response; partial aggregation is disabled")
        seen = set()
        for proxy, result in results:
            if result.status.code != fl.common.Code.OK:
                raise ValueError("client reported a failed response")
            metrics = result.metrics
            if proxy.cid in seen or proxy.cid not in expected:
                raise ValueError("duplicate or unrequested client response")
            seen.add(proxy.cid)
            required = {"client_id": expected[proxy.cid], "round": server_round,
                        "contract": self.contract, "schema": self.codec.fingerprint,
                        "base_hash": config["base_hash"]}
            if evaluation:
                required["split"] = config["split"]
            if any(metrics.get(key) != value for key, value in required.items()):
                raise ValueError("incompatible or stale client response")
        return expected

    def aggregate_fit(self, server_round, results, failures):
        self._validate_responses(server_round, results, failures)
        results = sorted(results, key=lambda item: item[1].metrics["client_id"])
        arrays, weights, records = [], [], []
        allowed = {"client_id", "round", "contract", "schema", "base_hash", "original_train",
                   "resampled_train", "original_adult", "original_non_adult", "used_adult",
                   "used_non_adult", "duplicated", "ensemble_members", "examples_processed", "train_loss",
                   "train_seconds", "fit_seconds"}
        if self.cfg.get("representation"):
            allowed |= {"warmup_examples_processed", "synthetic", "smote_effective_k", "balancing_status", "feature_space"}
        if self.cfg.get('model_selection'):
            allowed |= {'assigned_train', 'selection_train'}
        for proxy, result in results:
            metrics = result.metrics
            if set(metrics) != allowed:
                raise ValueError("only agreed aggregate training statistics may leave a client")
            for key in ("original_train", "resampled_train", "examples_processed", "ensemble_members"):
                if type(metrics[key]) is not int or metrics[key] <= 0:
                    raise ValueError("invalid training count")
            for key in ("original_adult", "original_non_adult", "used_adult", "used_non_adult", "duplicated"):
                if type(metrics[key]) is not int or metrics[key] < 0:
                    raise ValueError("invalid training class count")
            if metrics["original_adult"] + metrics["original_non_adult"] != metrics["original_train"]:
                raise ValueError("original class counts do not sum to original training count")
            if self.cfg.get('model_selection') and (
                    type(metrics['assigned_train']) is not int or type(metrics['selection_train']) is not int
                    or metrics['selection_train'] < 0
                    or metrics['assigned_train'] != metrics['original_train'] + metrics['selection_train']):
                raise ValueError('training/selection counts do not sum to assigned training count')
            if metrics["used_adult"] + metrics["used_non_adult"] != metrics["resampled_train"]:
                raise ValueError("balanced class counts do not sum to resampled training count")
            expected_members = self.cfg["balancing"].get("ensemble_members", 5) if self.cfg["balancing"]["method"] == "ensemble_undersampling" else 1
            warmup = 0
            if self.cfg.get("representation"):
                warmup = metrics["original_train"] * expected_members * self.cfg["representation"]["warmup_epochs"]
                if (metrics["warmup_examples_processed"] != warmup or
                        metrics["feature_space"] != "encoder_embeddings" or
                        type(metrics["synthetic"]) is not int or metrics["synthetic"] < 0 or
                        type(metrics["smote_effective_k"]) is not int or metrics["smote_effective_k"] < 0 or
                        metrics["balancing_status"] not in ("applied", "adjusted_k", "skipped_insufficient_minority",
                            "unchanged_already_at_or_above_target", "unchanged", "skipped_missing_class")):
                    raise ValueError("invalid embedding training statistics")
                if self.cfg["balancing"]["method"] == "smote":
                    if (metrics["used_non_adult"] != metrics["original_non_adult"] or metrics["duplicated"] or
                            metrics["used_adult"] != metrics["original_adult"] + metrics["synthetic"]):
                        raise ValueError("SMOTE counts violate the training-only interpolation contract")
                    balance = self.cfg["balancing"]
                    r = ratio(balance["adult_to_non_adult"])
                    adult, non = metrics["original_adult"], metrics["original_non_adult"]
                    expected_synthetic = max(0, (non*r.numerator+r.denominator-1)//r.denominator-adult)
                    k = balance.get("k_neighbors", 5)
                    status = "applied"
                    if not expected_synthetic:
                        k, status = 0, "unchanged_already_at_or_above_target"
                    elif adult < 2 or (adult <= k and balance.get("insufficient_minority", "adjust") == "skip"):
                        expected_synthetic, k, status = 0, 0, "skipped_insufficient_minority"
                    elif adult <= k:
                        k, status = adult-1, "adjusted_k"
                    if (metrics["synthetic"], metrics["smote_effective_k"], metrics["balancing_status"]) != (expected_synthetic, k, status):
                        raise ValueError("SMOTE targets/neighbors/status violate the training-count policy")
                elif metrics["synthetic"] or metrics["smote_effective_k"]:
                    raise ValueError("non-SMOTE method reported synthetic embeddings")
            if (metrics["ensemble_members"] != expected_members or metrics["examples_processed"] != warmup +
                    metrics["resampled_train"] * expected_members * self.cfg["local_epochs"]):
                raise ValueError("local epoch/example counts violate the experiment contract")
            if not math.isfinite(metrics["train_loss"]) or metrics["train_loss"] < 0:
                raise ValueError("invalid training loss")
            for key in ("train_seconds", "fit_seconds"):
                if not math.isfinite(metrics[key]) or metrics[key] < 0:
                    raise ValueError("invalid local timing")
            cid = metrics["client_id"]
            if cid in self.original_counts and self.original_counts[cid] != metrics["original_train"]:
                raise ValueError("original local training population changed across rounds")
            self.original_counts[cid] = metrics["original_train"]
            expected_weight = metrics["original_train"] if self.cfg["aggregation_weight"] == "original_training" else metrics["resampled_train"]
            if type(result.num_examples) is not int or result.num_examples != expected_weight:
                raise ValueError("client's FedAvg weight violates the experiment policy")
            values = parameters_to_ndarrays(result.parameters)
            self.codec.validate(values)
            arrays.append(values)
            weights.append(expected_weight)
            records.append(dict(metrics, aggregation_weight=expected_weight))
        # Full-state weighted averaging equals averaging deltas from the common
        # global state. Integer counters stay at their global reference values.
        total = sum(weights)
        aggregate = []
        for index, reference in enumerate(self.current):
            if np.issubdtype(reference.dtype, np.floating):
                combined = np.zeros_like(reference)
                for values, weight in zip(arrays, weights):
                    combined += values[index] * (weight / total)
                aggregate.append(combined)
            else:
                aggregate.append(reference.copy())
        self.codec.validate(aggregate)
        self.current = aggregate
        if all(isinstance(proxy, ChunkedProxy) and "fit" in proxy.traffic for proxy, _ in results):
            self.communication[server_round]["fit_down_bytes"] = sum(proxy.traffic["fit"][0] for proxy, _ in results)
            self.communication[server_round]["fit_up_bytes"] = sum(proxy.traffic["fit"][1] for proxy, _ in results)
        else:
            self.communication[server_round]["fit_up_bytes"] = sum(fit_res_to_proto(res).ByteSize() for _, res in results)
        self.round_reports[server_round] = {
            "round": server_round, "weighting_policy": self.cfg["aggregation_weight"],
            "clients": sorted(records, key=lambda row: row["client_id"]),
            "global_hash": state_hash(aggregate),
            "optimization": {"loss": sum(record["train_loss"]*record["examples_processed"] for record in records)/sum(record["examples_processed"] for record in records),
                             "examples_processed": sum(record["examples_processed"] for record in records),
                             "scope": "original-training encoder warmup plus balanced head views; not global train evaluation" if self.cfg.get("representation") else "balanced training views, epochs and ensemble members; not global train evaluation"},
            "balancing": {"training_protocol": "encoder_embeddings" if self.cfg.get("representation") else "end_to_end",
                          "synthetic_embeddings": sum(record.get("synthetic", 0) for record in records),
                          "smote_clients_skipped": sum(record.get("balancing_status") == "skipped_insufficient_minority" for record in records),
                          "smote_clients_adjusted": sum(record.get("balancing_status") == "adjusted_k" for record in records)},
            "timing": {"client_training_seconds_sum": sum(record["train_seconds"] for record in records),
                       "client_training_seconds_max": max(record["train_seconds"] for record in records),
                       "fit_round_wall_seconds": time.perf_counter()-self.round_started[server_round]},
        }
        self._write(server_round)
        self._checkpoint(server_round, evaluated=False)
        return ndarrays_to_parameters(aggregate), {"total_aggregation_weight": total}

    def configure_evaluate(self, server_round, parameters, client_manager):
        return self.evaluation_requests(server_round, parameters, client_manager, split="val")

    def evaluation_requests(self, server_round, parameters, manager, *, split):
        self.evaluation_started[server_round, split] = time.perf_counter()
        clients = self._clients(manager)
        config = dict(self._message(server_round, parameters), split=split)
        self.pending_eval[server_round] = ({proxy.cid: cid for cid, proxy in clients.items()}, config)
        requests = [(clients[cid], EvaluateIns(parameters, dict(config))) for cid in range(5)]
        native = sum(evaluate_ins_to_proto(ins).ByteSize() for proxy, ins in requests
                     if not isinstance(proxy, ChunkedProxy) or parameter_size(ins.parameters) <= DEFAULT_LIMITS["chunk_bytes"])
        self.evaluation_native_bytes[server_round, split] = native
        self.communication[server_round]["evaluation_down_bytes"] += native
        return requests

    def aggregate_evaluate(self, server_round, results, failures):
        self._validate_responses(server_round, results, failures, evaluation=True)
        allowed = {"tn", "fp", "fn", "tp", "loss_sum", "client_id", "contract", "schema",
                   "round", "base_hash", "split", "evaluation_seconds"}
        split = self.pending_eval[server_round][1]["split"]
        if split == "val" and self.cfg.get("evaluate_training", True):
            allowed |= {"train_n_records", "train_tn", "train_fp", "train_fn", "train_tp",
                        "train_loss_sum", "train_evaluation_seconds"}
        if split == 'val' and self.cfg.get('model_selection'):
            allowed |= {'selection_n_records', 'selection_tn', 'selection_fp', 'selection_fn',
                        'selection_tp', 'selection_loss_sum', 'selection_evaluation_seconds'}
        if any(set(result.metrics) != allowed for _, result in results):
            raise ValueError("only aggregate evaluation counts/loss may leave a client")
        results = sorted(results, key=lambda item: item[1].metrics["client_id"])
        aggregate = aggregate_validation([(result.num_examples, result.metrics) for _, result in results])
        selection = None
        if split == 'val' and self.cfg.get('model_selection'):
            held = []
            for _, res in results:
                n = res.metrics['selection_n_records']
                counts = {key: res.metrics[f'selection_{key}'] for key in ('tn', 'fp', 'fn', 'tp', 'loss_sum')}
                if type(n) is not int or n < 0:
                    raise ValueError('invalid selection count')
                if not n:
                    if any(type(counts[key]) is not int or counts[key] != 0 for key in ('tn','fp','fn','tp')) or counts['loss_sum'] != 0:
                        raise ValueError('nonzero metrics on empty selection pool')
                else:
                    held.append((n, counts))
            if not held:
                raise ValueError('empty pooled training-selection data; no best checkpoint can be selected')
            selection = aggregate_validation(held)
            if not selection['adult_count'] or not selection['non_adult_count']:
                raise ValueError('pooled training-selection data requires both classes; no seed retuning is allowed')
        self.round_reports[server_round][split] = aggregate
        self.round_reports[server_round][split]["threshold"] = self.cfg["threshold"]
        if selection is not None:
            self.round_reports[server_round]['selection'] = dict(selection, threshold=self.cfg['threshold'],
                scope='fixed domain-disjoint subset of original training only; never optimized or balanced')
        if split == "val" and self.cfg.get("evaluate_training", True):
            training = [(res.metrics["train_n_records"], {key: res.metrics[f"train_{key}"]
                         for key in ("tn", "fp", "fn", "tp", "loss_sum")}) for _, res in results]
            self.round_reports[server_round]["train"] = dict(aggregate_validation(training),
                threshold=self.cfg["threshold"], scope=(
                    'updated global model on unresampled optimization pools; excludes training-selection data'
                    if self.cfg.get('model_selection') else
                    'updated global model on original local training splits; all five clients'))
        timing = self.round_reports[server_round]["timing"]
        for _, res in results:
            for key in ("evaluation_seconds", "train_evaluation_seconds", "selection_evaluation_seconds"):
                if key in res.metrics and (not math.isfinite(res.metrics[key]) or res.metrics[key] < 0):
                    raise ValueError("invalid evaluation timing")
        timing[f"{split}_client_evaluation_seconds_sum"] = sum(res.metrics["evaluation_seconds"] for _, res in results)
        timing[f"{split}_phase_wall_seconds"] = time.perf_counter()-self.evaluation_started[server_round, split]
        if split == "val":
            timing["round_wall_seconds"] = time.perf_counter()-self.round_started[server_round]
            if self.cfg.get("evaluate_training", True):
                timing["train_evaluation_seconds_sum"] = sum(res.metrics["train_evaluation_seconds"] for _, res in results)
            if selection is not None:
                timing['selection_evaluation_seconds_sum'] = sum(res.metrics['selection_evaluation_seconds'] for _, res in results)
        if all(isinstance(proxy, ChunkedProxy) and "evaluate" in proxy.traffic for proxy, _ in results):
            # configure_evaluate already counted native small requests.
            self.communication[server_round]["evaluation_down_bytes"] -= self.evaluation_native_bytes[server_round, split]
            self.communication[server_round]["evaluation_down_bytes"] += sum(proxy.traffic["evaluate"][0] for proxy, _ in results)
            self.communication[server_round]["evaluation_up_bytes"] += sum(proxy.traffic["evaluate"][1] for proxy, _ in results)
        else:
            self.communication[server_round]["evaluation_up_bytes"] += sum(evaluate_res_to_proto(res).ByteSize() for _, res in results)
        self._write(server_round)
        if split == 'val':
            self._checkpoint(server_round, evaluated=True)
            if selection is not None:
                self._select_best(server_round, selection)
        from src.evaluation.console import print_round_summary
        print_round_summary(self.round_reports[server_round], split=split, rounds=self.cfg['rounds'])
        return aggregate["loss"], {key: value for key, value in aggregate.items()
                                  if isinstance(value, (int, float))}

    def _checkpoint(self, rnd, *, evaluated):
        if self.output is None:
            return
        directory = self.output/'checkpoints'
        directory.mkdir(exist_ok=True)
        path = directory/f'round_{rnd:04d}.npz'
        if not evaluated:
            if path.exists():
                raise FileExistsError('refusing to overwrite a round checkpoint')
            save_weights(path, self.current)
        metadata = {'round':rnd, 'status':'evaluated' if evaluated else 'aggregated_pending_evaluation',
            'configuration':self.cfg, 'contract':self.contract,
            'schema':self.codec.entries, 'schema_hash':self.codec.fingerprint,
            'global_hash':state_hash(self.current), 'client_data_contracts':self.client_data_contracts,
            'metrics':{name:self.round_reports[rnd][name] for name in ('train','selection','val')
                       if evaluated and name in self.round_reports[rnd]},
            'purpose':'global model weights; includes all ensemble members; not a complete resumable training session'}
        atomic_json(path.with_suffix('.json'), metadata)
        print(f'Saved global checkpoint: round {rnd}, {metadata["status"]}: {path}', flush=True)

    def _select_best(self, rnd, selection):
        score = selection['adult_f1']
        if self.best is not None and score <= self.best['score']:
            print(f'Best checkpoint remains round {self.best["round"]}: selection Adult F1={self.best["score"]:.4f}', flush=True)
            return
        best = {'round':rnd, 'score':score, 'metric':'adult_f1', 'source':'training_selection',
            'tie_break':'earliest round', 'global_hash':state_hash(self.current),
            'checkpoint':f'checkpoints/round_{rnd:04d}.npz', 'selection_metrics':selection,
            'scope':'best checkpoint within this run only; not a best model/method claim across experiments'}
        if self.output is not None:
            path = self.output/best['checkpoint']
            copy_best(path, self.output/'best_model.npz')
            metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            atomic_json(self.output/'best_model.json', dict(metadata, selection=best))
        else:
            self.best_arrays = [value.copy() for value in self.current]
        self.best = best
        print(f'BEST CHECKPOINT: round {rnd}, training-selection Adult F1={score:.4f}; validation/test not used for selection.', flush=True)

    def selected_parameters(self):
        if self.best is None:
            raise RuntimeError('no evaluated training-selection checkpoint is available')
        if self.output is not None:
            arrays, _ = load_checkpoint(self.output/self.best['checkpoint'], self.codec, expected_contract=self.contract)
        else:
            arrays = self.best_arrays
        return ndarrays_to_parameters(arrays)

    def _write(self, server_round):
        if server_round in self.round_reports:
            measurements = dict(self.communication[server_round])
            measurements["total_bytes"] = sum(measurements.values())
            measurements["measurement"] = "serialized Flower Fit/Evaluate and parameter-chunk GetProperties protobuf payloads, both directions; excludes registration/gRPC/TLS/retry overhead"
            self.round_reports[server_round]["communication"] = measurements
        if self.output:
            path = self.output / f"round_{server_round:04d}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.round_reports[server_round], indent=2, allow_nan=False) + "\n")
            temporary.replace(path)

    def save_final(self):
        if self.output:
            save_weights(self.output / 'global_final.npz', self.current)
            metadata = {"status": "PASS", "configuration": self.cfg, "contract": self.contract,
                        "schema": self.codec.entries, "schema_hash": self.codec.fingerprint,
                        "initial_global_hash": self.initial_hash,
                        "client_data_contracts": self.client_data_contracts,
                        "client_runtime": self.client_runtime,
                        "server_runtime": {"platform": platform.platform(), "processor": platform.processor() or "unavailable"},
                        "experiment": self.cfg.get("experiment", {}),
                        "global_hash": state_hash(self.current), "completed_rounds": len(self.round_reports),
                        "privacy": "model states and aggregate statistics only; no DP or secure aggregation",
                        "selection": (self.best if self.cfg.get('model_selection') else
                                      "fixed final round, no holdout-driven tuning")}
            (self.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


class FederatedServer(fl.server.Server):
    """Same Flower orchestration for the network endpoint and local harness."""
    def fit(self, num_rounds, timeout):
        result = super().fit(num_rounds, timeout)
        if self.strategy.cfg["evaluate_final_test"]:
            evaluation_parameters = self.parameters
            if self.strategy.cfg.get('model_selection'):
                evaluation_parameters = self.strategy.selected_parameters()
                self.strategy.round_reports[num_rounds]['test_checkpoint'] = {
                    'round':self.strategy.best['round'], 'global_hash':self.strategy.best['global_hash'],
                    'source':'training_selection'}
            requests = self.strategy.evaluation_requests(num_rounds, evaluation_parameters, self._client_manager, split="test")
            responses = [(proxy, proxy.evaluate(ins, timeout, num_rounds)) for proxy, ins in requests]
            self.strategy.aggregate_evaluate(num_rounds, responses, [])
        self.strategy.save_final()
        return result

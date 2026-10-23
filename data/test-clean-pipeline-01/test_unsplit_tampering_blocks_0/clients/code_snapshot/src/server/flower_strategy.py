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
        self.communication, self.client_data_contracts = {}, {}
        self.client_runtime = {}
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
        for proxy in proxies:
            response = proxy.get_properties(GetPropertiesIns({}), timeout=30, group_id=None)
            if response.status.code != fl.common.Code.OK:
                raise ValueError("client registration failed")
            properties = response.properties
            cid = properties.get("client_id")
            if (type(cid) is not int or cid not in range(5) or cid in registry or
                    properties.get("contract") != self.contract or
                    properties.get("schema") != self.codec.fingerprint):
                raise ValueError("client identity, experiment or model schema mismatch")
            registry[cid] = proxy
            self.client_data_contracts[str(cid)] = {key: properties[key] for key in
                ("train_sha256", "val_sha256", "test_sha256", "partition_strategy", "data_kind")
                if key in properties}
            self.client_runtime[str(cid)] = {key: properties[key] for key in
                ("device", "torch_threads", "cuda_runtime") if key in properties}
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
            "fit_down_bytes": sum(fit_ins_to_proto(ins).ByteSize() for _, ins in requests),
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
            if metrics["used_adult"] + metrics["used_non_adult"] != metrics["resampled_train"]:
                raise ValueError("balanced class counts do not sum to resampled training count")
            expected_members = self.cfg["balancing"].get("ensemble_members", 5) if self.cfg["balancing"]["method"] == "ensemble_undersampling" else 1
            if (metrics["ensemble_members"] != expected_members or metrics["examples_processed"] !=
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
        self.communication[server_round]["fit_up_bytes"] = sum(fit_res_to_proto(res).ByteSize() for _, res in results)
        self.round_reports[server_round] = {
            "round": server_round, "weighting_policy": self.cfg["aggregation_weight"],
            "clients": sorted(records, key=lambda row: row["client_id"]),
            "global_hash": state_hash(aggregate),
            "optimization": {"loss": sum(record["train_loss"]*record["examples_processed"] for record in records)/sum(record["examples_processed"] for record in records),
                             "examples_processed": sum(record["examples_processed"] for record in records),
                             "scope": "balanced training views, epochs and ensemble members; not global train evaluation"},
            "timing": {"client_training_seconds_sum": sum(record["train_seconds"] for record in records),
                       "client_training_seconds_max": max(record["train_seconds"] for record in records),
                       "fit_round_wall_seconds": time.perf_counter()-self.round_started[server_round]},
        }
        self._write(server_round)
        return ndarrays_to_parameters(aggregate), {"total_aggregation_weight": total}

    def configure_evaluate(self, server_round, parameters, client_manager):
        return self.evaluation_requests(server_round, parameters, client_manager, split="val")

    def evaluation_requests(self, server_round, parameters, manager, *, split):
        self.evaluation_started[server_round, split] = time.perf_counter()
        clients = self._clients(manager)
        config = dict(self._message(server_round, parameters), split=split)
        self.pending_eval[server_round] = ({proxy.cid: cid for cid, proxy in clients.items()}, config)
        requests = [(clients[cid], EvaluateIns(parameters, dict(config))) for cid in range(5)]
        self.communication[server_round]["evaluation_down_bytes"] += sum(evaluate_ins_to_proto(ins).ByteSize() for _, ins in requests)
        return requests

    def aggregate_evaluate(self, server_round, results, failures):
        self._validate_responses(server_round, results, failures, evaluation=True)
        allowed = {"tn", "fp", "fn", "tp", "loss_sum", "client_id", "contract", "schema",
                   "round", "base_hash", "split", "evaluation_seconds"}
        split = self.pending_eval[server_round][1]["split"]
        if split == "val" and self.cfg.get("evaluate_training", True):
            allowed |= {"train_n_records", "train_tn", "train_fp", "train_fn", "train_tp",
                        "train_loss_sum", "train_evaluation_seconds"}
        if any(set(result.metrics) != allowed for _, result in results):
            raise ValueError("only aggregate evaluation counts/loss may leave a client")
        results = sorted(results, key=lambda item: item[1].metrics["client_id"])
        aggregate = aggregate_validation([(result.num_examples, result.metrics) for _, result in results])
        self.round_reports[server_round][split] = aggregate
        self.round_reports[server_round][split]["threshold"] = self.cfg["threshold"]
        if split == "val" and self.cfg.get("evaluate_training", True):
            training = [(res.metrics["train_n_records"], {key: res.metrics[f"train_{key}"]
                         for key in ("tn", "fp", "fn", "tp", "loss_sum")}) for _, res in results]
            self.round_reports[server_round]["train"] = dict(aggregate_validation(training),
                threshold=self.cfg["threshold"], scope="updated global model on original local training splits; all five clients")
        timing = self.round_reports[server_round]["timing"]
        for _, res in results:
            for key in ("evaluation_seconds", "train_evaluation_seconds"):
                if key in res.metrics and (not math.isfinite(res.metrics[key]) or res.metrics[key] < 0):
                    raise ValueError("invalid evaluation timing")
        timing[f"{split}_client_evaluation_seconds_sum"] = sum(res.metrics["evaluation_seconds"] for _, res in results)
        timing[f"{split}_phase_wall_seconds"] = time.perf_counter()-self.evaluation_started[server_round, split]
        if split == "val":
            timing["round_wall_seconds"] = time.perf_counter()-self.round_started[server_round]
            if self.cfg.get("evaluate_training", True):
                timing["train_evaluation_seconds_sum"] = sum(res.metrics["train_evaluation_seconds"] for _, res in results)
        self.communication[server_round]["evaluation_up_bytes"] += sum(evaluate_res_to_proto(res).ByteSize() for _, res in results)
        self._write(server_round)
        print(f"Round {server_round} global {split}: Adult F1={aggregate['adult_f1']:.4f}, "
              f"balanced accuracy={aggregate['balanced_accuracy']:.4f}", flush=True)
        return aggregate["loss"], {key: value for key, value in aggregate.items()
                                  if isinstance(value, (int, float))}

    def _write(self, server_round):
        if server_round in self.round_reports:
            measurements = dict(self.communication[server_round])
            measurements["total_bytes"] = sum(measurements.values())
            measurements["measurement"] = "serialized Flower Fit/Evaluate protobuf payloads, both directions; excludes gRPC/TLS/control/retry overhead"
            self.round_reports[server_round]["communication"] = measurements
        if self.output:
            path = self.output / f"round_{server_round:04d}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.round_reports[server_round], indent=2, allow_nan=False) + "\n")
            temporary.replace(path)

    def save_final(self):
        if self.output:
            np.savez(self.output / "global_final.npz", **{f"tensor_{i:04d}": array for i, array in enumerate(self.current)})
            metadata = {"status": "PASS", "configuration": self.cfg, "contract": self.contract,
                        "schema": self.codec.entries, "schema_hash": self.codec.fingerprint,
                        "initial_global_hash": self.initial_hash,
                        "client_data_contracts": self.client_data_contracts,
                        "client_runtime": self.client_runtime,
                        "server_runtime": {"platform": platform.platform(), "processor": platform.processor() or "unavailable"},
                        "experiment": self.cfg.get("experiment", {}),
                        "global_hash": state_hash(self.current), "completed_rounds": len(self.round_reports),
                        "privacy": "model states and aggregate statistics only; no DP or secure aggregation",
                        "selection": "fixed final round, no holdout-driven tuning"}
            (self.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


class FederatedServer(fl.server.Server):
    """Same Flower orchestration for the network endpoint and local harness."""
    def fit(self, num_rounds, timeout):
        result = super().fit(num_rounds, timeout)
        if self.strategy.cfg["evaluate_final_test"]:
            requests = self.strategy.evaluation_requests(num_rounds, self.parameters, self._client_manager, split="test")
            responses = [(proxy, proxy.evaluate(ins, timeout, num_rounds)) for proxy, ins in requests]
            self.strategy.aggregate_evaluate(num_rounds, responses, [])
        self.strategy.save_final()
        return result

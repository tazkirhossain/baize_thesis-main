"""Shared public experiment contract (contains no client data locations)."""
import hashlib
import json
import math

from src.data.balancing import BalanceConfig


def validate_config(cfg):
    allowed = {"n_clients", "rounds", "clients_per_round", "local_epochs", "batch_size",
               "learning_rate", "seed", "model", "balancing", "aggregation_weight",
               "threshold", "evaluate_final_test"}
    optional = {"evaluate_training", "experiment", "representation", "transport", "progress_interval_seconds", "model_selection"}
    if not allowed <= set(cfg) or set(cfg) - allowed - optional:
        raise ValueError(f"federated configuration fields must be {sorted(allowed)}")
    if type(cfg.get("evaluate_training", True)) is not bool:
        raise ValueError("evaluate_training must be boolean")
    selection = cfg.get('model_selection')
    if selection is not None:
        if (not isinstance(selection, dict) or set(selection) != {'source', 'fraction', 'metric'}
                or selection['source'] != 'training_selection' or selection['metric'] != 'adult_f1'
                or type(selection['fraction']) not in (int, float)
                or not math.isfinite(selection['fraction']) or not 0 < selection['fraction'] <= .5):
            raise ValueError('model_selection requires training_selection, adult_f1 and fraction in (0, 0.5]')
    interval = cfg.get('progress_interval_seconds', 30)
    if type(interval) not in (int, float) or not math.isfinite(interval) or interval < 0:
        raise ValueError('progress_interval_seconds must be finite and nonnegative (0 disables progress)')
    if "experiment" in cfg:
        experiment = cfg["experiment"]
        if (set(experiment) != {"partition_strategy", "data_kind", "evaluation_population_id"} or
                experiment["partition_strategy"] not in ("iid", "label_skew", "category_skew", "unknown") or
                experiment["data_kind"] not in ("real", "synthetic") or
                not isinstance(experiment["evaluation_population_id"], str) or
                not experiment["evaluation_population_id"]):
            raise ValueError("invalid public experiment/evaluation population metadata")
    for key in ("n_clients", "rounds", "clients_per_round", "local_epochs", "batch_size", "seed"):
        if type(cfg[key]) is not int or cfg[key] < (0 if key == "seed" else 1):
            raise ValueError(f"invalid {key}")
    if cfg["n_clients"] != 5 or cfg["clients_per_round"] > 5 or cfg["seed"] >= 2**32:
        raise ValueError("exactly five clients, 1-5 participants and a 32-bit seed are required")
    for key in ("learning_rate", "threshold"):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"invalid {key}")
    if cfg["threshold"] >= 1 or type(cfg["evaluate_final_test"]) is not bool:
        raise ValueError("invalid evaluation settings")
    if cfg["aggregation_weight"] not in ("original_training", "resampled_training"):
        raise ValueError("unknown FedAvg weighting policy")
    balance = BalanceConfig(**cfg["balancing"])
    balance.validate()
    representation = cfg.get("representation")
    if representation is not None:
        if (not isinstance(representation, dict) or set(representation) != {"mode", "warmup_epochs"} or
                representation["mode"] != "encoder_embeddings" or
                type(representation["warmup_epochs"]) is not int or representation["warmup_epochs"] < 1):
            raise ValueError("representation requires encoder_embeddings and positive warmup_epochs")
    if balance.method == "smote" and representation is None:
        raise ValueError("SMOTE requires explicit encoder_embeddings representation; never interpolate tokens")
    transport = cfg.get("transport", {"chunk_bytes": 4*1024**2, "max_transfer_bytes": 16*1024**3})
    if (not isinstance(transport, dict) or set(transport) != {"chunk_bytes", "max_transfer_bytes"} or
            any(type(value) is not int for value in transport.values()) or
            not 1024 <= transport["chunk_bytes"] <= 32*1024**2 or
            not transport["chunk_bytes"] <= transport["max_transfer_bytes"] <= 64*1024**3):
        raise ValueError("invalid chunked transport limits")
    model = cfg["model"]
    if model["family"] == "character_cnn":
        if set(model) != {"family", "characters", "sequence_length", "dropout_input"}:
            raise ValueError("invalid character CNN fields")
        alphabet = model["characters"]
        if not isinstance(alphabet, str) or not alphabet or len(set(alphabet)) != len(alphabet):
            raise ValueError("characters must be a unique, fixed public alphabet")
        if type(model["sequence_length"]) is not int or model["sequence_length"] < 8:
            raise ValueError("CNN sequence_length must be >= 8 (including singleton training batches)")
        if not 0 <= model["dropout_input"] <= 1:
            raise ValueError("invalid input dropout")
    elif model["family"] == "deberta_v3_base":
        if set(model) != {"family", "source", "revision", "max_length"}:
            raise ValueError("invalid DeBERTa fields")
        if type(model["max_length"]) is not int or not 2 <= model["max_length"] <= 512:
            raise ValueError("invalid token length")
    else:
        raise ValueError("unknown neural family")
    return cfg


def contract_hash(cfg):
    validate_config(cfg)
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

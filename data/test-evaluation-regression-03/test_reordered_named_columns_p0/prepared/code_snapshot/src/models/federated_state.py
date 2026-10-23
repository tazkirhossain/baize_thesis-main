"""Strict ordered tensor transport, independent of URLs or datasets."""
import hashlib
import json

import numpy as np
import torch


class StateCodec:
    def __init__(self, models):
        self.models = models
        self.entries = [(member, name, tuple(value.shape), value.cpu().numpy().dtype.str)
                        for member, model in enumerate(models)
                        for name, value in model.state_dict().items()]
        self.fingerprint = hashlib.sha256(json.dumps(self.entries).encode()).hexdigest()

    def get(self):
        return [value.detach().cpu().numpy().copy() for model in self.models
                for value in model.state_dict().values()]

    def validate(self, arrays):
        if len(arrays) != len(self.entries):
            raise ValueError("model state tensor count mismatch")
        for array, (_, name, shape, dtype) in zip(arrays, self.entries):
            if (not isinstance(array, np.ndarray) or array.shape != shape or
                    array.dtype.str != dtype or not np.isfinite(array).all()):
                raise ValueError(f"incompatible/nonfinite model tensor: {name}")

    def set(self, arrays):
        self.validate(arrays)
        states = [dict() for _ in self.models]
        for array, (member, name, _, _) in zip(arrays, self.entries):
            states[member][name] = torch.from_numpy(array.copy())
        for model, state in zip(self.models, states):
            model.load_state_dict(state, strict=True)


def state_hash(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def make_models(cfg):
    from src.data.balancing import BalanceConfig
    balance = BalanceConfig(**cfg["balancing"])
    count = balance.ensemble_members if balance.method == "ensemble_undersampling" else 1
    models = []
    for member in range(count):
        seed = (cfg["seed"] + member) % 2**32
        if cfg["model"]["family"] == "character_cnn":
            from src.models.character_cnn import CharacterCNNConfig, CharacterLevelCNN
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                # Extra channel is UNK; all-zero vectors are padding.
                args = CharacterCNNConfig(len(cfg["model"]["characters"]) + 1,
                                          dropout_input=cfg["model"]["dropout_input"])
                models.append(CharacterLevelCNN(args, 2))
        else:
            from src.models.deberta_v3_base.model import build_model
            settings = cfg["model"]
            models.append(build_model(settings["source"], revision=settings["revision"], seed=seed))
    return models

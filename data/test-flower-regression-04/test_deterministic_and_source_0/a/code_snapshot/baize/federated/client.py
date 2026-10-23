"""Federated client.

A client owns its local feature matrix and labels, receives global parameters,
trains locally, and returns a DELTA plus its sample count. It never returns its
data, and the server never asks for it.

Adversarial behaviour (THREAT_MODEL.md §4) is implemented here rather than in
the server, because that is where it would actually live: a malicious client is
one that runs different code, not one the server has flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from baize.models.linear import LogisticSGD


@dataclass
class ClientConfig:
    local_epochs: int = 3
    batch_size: int = 64
    lr: float = 0.5
    l2: float = 1e-5
    # THREAT_MODEL.md §4 simulated attacks. "honest" is the default and must
    # remain the default: an attack silently on by default would corrupt every
    # baseline in the project.
    behaviour: str = "honest"  # honest | sign_flip | scale | gaussian
    attack_scale: float = 10.0
    attack_sigma: float = 1.0
    # Training loss costs one extra full pass over the shard per round. Useful
    # for diagnostics, wasteful in a large sweep.
    compute_loss: bool = True


@dataclass
class ClientUpdate:
    client_id: int
    delta: np.ndarray
    n_samples: int
    train_loss: float
    behaviour: str
    update_norm: float
    meta: dict = field(default_factory=dict)


class FederatedClient:
    def __init__(
        self,
        client_id: int,
        X,
        y: np.ndarray,
        n_features: int,
        config: ClientConfig | None = None,
        seed: int = 0,
        indices: np.ndarray | None = None,
    ) -> None:
        """`indices` lets the client address rows of a SHARED feature matrix
        instead of owning a slice.

        This is a simulation-efficiency measure only. It does not weaken the
        federated abstraction: the client still reads exclusively its own rows
        and still returns only a parameter delta. At corpus scale, materialising
        one slice per client duplicates the entire training matrix in memory.
        """
        self.client_id = client_id
        self.X = X
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)
        self.y = np.asarray(y, dtype=np.int8)
        self.n_features = n_features
        self.config = config or ClientConfig()
        self.rng = np.random.default_rng(seed)
        self.model = LogisticSGD(n_features, l2=self.config.l2, seed=seed)

    def __len__(self) -> int:
        return int(len(self.indices) if self.indices is not None else self.X.shape[0])

    @property
    def class_counts(self) -> dict[int, int]:
        v, c = np.unique(self.y, return_counts=True)
        return {int(a): int(b) for a, b in zip(v, c)}

    def local_train(self, global_params: np.ndarray) -> ClientUpdate:
        cfg = self.config
        self.model.set_params(global_params)

        record = self.model.fit(
            self.X,
            self.y,
            epochs=cfg.local_epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            rng=self.rng,
            indices=self.indices,
            compute_loss=cfg.compute_loss,
        )
        delta = self.model.get_params() - global_params
        delta = self._apply_behaviour(delta)

        if not np.all(np.isfinite(delta)):
            raise ValueError(
                f"client {self.client_id} produced a non-finite update - "
                "master spec §64 requires BLOCK"
            )

        return ClientUpdate(
            client_id=self.client_id,
            delta=delta,
            n_samples=len(self),
            train_loss=record["final_loss"],
            behaviour=cfg.behaviour,
            update_norm=float(np.linalg.norm(delta)),
            meta={"n_steps": record["n_steps"], "class_counts": self.class_counts},
        )

    def _apply_behaviour(self, delta: np.ndarray) -> np.ndarray:
        b = self.config.behaviour
        if b == "honest":
            return delta
        if b == "sign_flip":
            return -delta
        if b == "scale":
            return delta * self.config.attack_scale
        if b == "gaussian":
            return self.rng.normal(0.0, self.config.attack_sigma, size=delta.shape)
        raise ValueError(f"unknown client behaviour {b!r}")

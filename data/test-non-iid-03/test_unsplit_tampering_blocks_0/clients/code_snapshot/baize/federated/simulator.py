"""Federated simulation loop.

    global params
        -> distribute to selected clients
        -> local training
        -> update validation (finiteness, norm)
        -> aggregation
        -> new global params
        -> evaluate on the server's clean validation set

The server's validation set is clean by assumption (DECISIONS.md D-004). That
assumption is strong and unrealistic; it is recorded in the run record so it
cannot be forgotten when results are written up.

The test set never appears in this module. Final test evaluation happens once,
in the runner, after all selection is complete.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from baize.evaluation.metrics import classification_metrics
from baize.federated.aggregation import get_aggregator
from baize.federated.client import FederatedClient
from baize.models.linear import LogisticSGD, init_params


class FederatedError(RuntimeError):
    pass


@dataclass
class FederatedConfig:
    rounds: int = 30
    participation: float = 1.0
    aggregation: str = "fedavg"
    aggregation_params: dict = field(default_factory=dict)
    weight_by_samples: bool = True
    max_update_norm: float | None = None  # pre-aggregation validation, not a defence
    eval_every: int = 1
    seed: int = 0


@dataclass
class RoundRecord:
    round: int
    n_selected: int
    client_ids: list[int]
    update_norms: list[float]
    mean_train_loss: float
    aggregate_norm: float
    rejected_clients: list[int]
    val_metrics: dict | None = None


@dataclass
class FederatedRun:
    config: FederatedConfig
    history: list[RoundRecord]
    final_params: np.ndarray
    best_params: np.ndarray
    best_round: int
    best_val_pr_auc: float
    wall_seconds: float
    assumptions: list[str] = field(default_factory=list)

    def curve(self, metric: str = "pr_auc") -> list[tuple[int, float]]:
        return [
            (r.round, r.val_metrics[metric])
            for r in self.history
            if r.val_metrics is not None
        ]


def run_federated(
    clients: list[FederatedClient],
    n_features: int,
    X_val,
    y_val: np.ndarray,
    config: FederatedConfig | None = None,
    verbose: bool = False,
) -> FederatedRun:
    cfg = config or FederatedConfig()
    if not clients:
        raise FederatedError("no clients")
    if cfg.rounds < 1:
        raise FederatedError(f"rounds must be >= 1, got {cfg.rounds}")
    if cfg.eval_every < 1:
        raise FederatedError(f"eval_every must be >= 1, got {cfg.eval_every}")
    if not (0.0 < cfg.participation <= 1.0):
        raise FederatedError(f"participation must be in (0.0, 1.0], got {cfg.participation}")

    rng = np.random.default_rng(cfg.seed)
    aggregate = get_aggregator(cfg.aggregation)

    global_params = init_params(n_features)
    eval_model = LogisticSGD(n_features)

    best_params = global_params.copy()
    best_pr_auc, best_round = -np.inf, -1
    history: list[RoundRecord] = []
    t0 = time.time()

    n_select = max(1, int(round(cfg.participation * len(clients))))

    for rnd in range(1, cfg.rounds + 1):
        selected = rng.choice(len(clients), size=n_select, replace=False)
        selected = np.sort(selected)

        updates, rejected = [], []
        for cid in selected:
            upd = clients[int(cid)].local_train(global_params)
            # Update validation (master spec §23). This rejects broken updates;
            # it is NOT a robustness defence, because an attacker can trivially
            # stay under the norm bound. Kept separate from the aggregator for
            # exactly that reason.
            if cfg.max_update_norm is not None and upd.update_norm > cfg.max_update_norm:
                rejected.append(upd.client_id)
                continue
            updates.append(upd)

        if not updates:
            raise FederatedError(
                f"round {rnd}: every client update was rejected "
                f"(max_update_norm={cfg.max_update_norm}) - configuration error, not a result"
            )

        deltas = [u.delta for u in updates]
        weights = (
            np.array([u.n_samples for u in updates], dtype=np.float64)
            if cfg.weight_by_samples
            else None
        )
        agg_delta, agg_rep = aggregate(deltas, weights=weights, **cfg.aggregation_params)

        global_params = global_params + agg_delta
        if not np.all(np.isfinite(global_params)):
            raise FederatedError(f"round {rnd}: global parameters became non-finite - BLOCK")

        rec = RoundRecord(
            round=rnd,
            n_selected=len(updates),
            client_ids=[u.client_id for u in updates],
            update_norms=[u.update_norm for u in updates],
            mean_train_loss=float(np.mean([u.train_loss for u in updates])),
            aggregate_norm=agg_rep.aggregate_norm,
            rejected_clients=rejected,
        )

        if rnd % cfg.eval_every == 0 or rnd == cfg.rounds:
            eval_model.set_params(global_params)
            rec.val_metrics = classification_metrics(y_val, eval_model.predict_proba(X_val))
            if rec.val_metrics["pr_auc"] > best_pr_auc:
                best_pr_auc = rec.val_metrics["pr_auc"]
                best_params = global_params.copy()
                best_round = rnd
            if verbose:
                print(
                    f"  round {rnd:>3}  loss={rec.mean_train_loss:.4f}  "
                    f"val_pr_auc={rec.val_metrics['pr_auc']:.4f}  "
                    f"|agg|={rec.aggregate_norm:.4f}"
                )

        history.append(rec)

    return FederatedRun(
        config=cfg,
        history=history,
        final_params=global_params,
        best_params=best_params,
        best_round=best_round,
        best_val_pr_auc=float(best_pr_auc),
        wall_seconds=time.time() - t0,
        assumptions=[
            "server holds a CLEAN validation set (DECISIONS.md D-004) - strong and "
            "unrealistic; model selection would be much harder without it",
            "full-information simulation: no communication cost, latency, dropout "
            "or stragglers are modelled",
        ],
    )

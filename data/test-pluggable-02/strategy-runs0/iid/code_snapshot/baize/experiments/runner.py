"""End-to-end experiment runner.

Enforces the pipeline order and the gates:

    load -> validate -> deduplicate -> split -> [leakage gate]
         -> partition -> [partition gate] -> noise -> [noise gate]
         -> train (centralized or federated)
         -> select on VALIDATION
         -> evaluate ONCE on the clean test set

The test set is touched exactly once per run, at the end, using the parameters
already chosen on validation. `ExperimentResult` records the provenance fields
required by master spec §50.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from baize.provenance import environment_record

from baize.data.dataset import URLDataset
from baize.data.features import build_vectorizer
from baize.data.noise import apply_client_noise, apply_noise, assert_noise_valid, mark_protected
from baize.data.partition import partition
from baize.data.splits import assert_no_leakage, make_splits
from baize.data.validation import assert_validation_passed, deduplicate, validate_dataset
from baize.evaluation.budget import describe_budget, passes_per_example
from baize.evaluation.metrics import classification_metrics, precision_at_base_rate
from baize.evaluation.selection import best_checkpoint
from baize.federated.client import ClientConfig, FederatedClient
from baize.federated.simulator import FederatedConfig, run_federated
from baize.models.linear import LogisticSGD


@dataclass
class ExperimentConfig:
    experiment_id: str = "BAIZE-EXP-000"
    seed: int = 0

    # data
    split_strategy: str = "domain_aware"
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)

    # features — library defaults stay small so tests are fast.
    # Production scripts must use baize.hparams (2**18 features, lr=160, ...).
    vectorizer: str = "hashing"
    n_features: int = 2**16
    ngram_range: tuple[int, int] = (3, 5)

    # mode
    mode: str = "federated"  # centralized | federated

    # federated
    n_clients: int = 10
    partition_strategy: str = "iid"
    partition_params: dict = field(default_factory=dict)
    rounds: int = 30
    local_epochs: int = 3
    batch_size: int = 64
    lr: float = 0.5
    l2: float = 1e-5
    aggregation: str = "fedavg"
    aggregation_params: dict = field(default_factory=dict)
    participation: float = 1.0
    max_update_norm: float | None = None

    # noise (training labels only)
    noise_process: str = "none"  # none | symmetric | class_conditional | client_heterogeneous
    noise_rate: float = 0.0
    noise_class_rates: dict | None = None
    noise_client_rates: list[float] | None = None
    noise_tolerance: float | None = None  # None -> statistical (noise.effective_tolerance)

    # adversarial clients
    malicious_clients: list[int] = field(default_factory=list)
    malicious_behaviour: str = "sign_flip"

    # centralized budget
    centralized_epochs: int = 10

    # reporting
    assumed_deployment_base_rates: tuple[float, ...] = (0.05, 0.01)


@dataclass
class ExperimentResult:
    config: dict
    provenance: dict
    dataset_report: str
    dedup_report: str
    split_report: str
    partition_report: str | None
    noise_report: str | None
    val_metrics: dict
    test_metrics: dict
    projected_precision: dict
    history: list[dict] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    wall_seconds: float = 0.0
    training_budget: dict = field(default_factory=dict)
    selected_checkpoint: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    def summary(self) -> str:
        from baize.evaluation.metrics import format_metrics

        L = [
            f"EXPERIMENT {self.config['experiment_id']}  seed={self.config['seed']}",
            f"  mode={self.config['mode']}  split={self.config['split_strategy']}  "
            f"agg={self.config['aggregation']}  noise={self.config['noise_process']}"
            f"@{self.config['noise_rate']}",
            f"  dataset fingerprint {self.provenance['dataset_fingerprint'][:16]}",
            "",
            format_metrics(self.val_metrics, "VALIDATION (used for selection)"),
            "",
            format_metrics(self.test_metrics, "CLEAN TEST (evaluated once)"),
            "",
            "  Projected precision at assumed deployment base rates",
            "  (PROJECTION from an assumed prior, not a measurement):",
        ]
        for br, pr in self.projected_precision.items():
            L.append(f"      base rate {br:<8} -> precision {pr:.4f}")
        if self.assumptions:
            L += ["", "  ASSUMPTIONS IN FORCE:"]
            L += [f"      - {a}" for a in self.assumptions]
        return "\n".join(L)


def _provenance(ds: URLDataset, cfg: ExperimentConfig) -> dict:
    rec = environment_record()
    rec.update({
        "dataset_name": ds.name,
        "dataset_fingerprint": ds.fingerprint(),
        "n_records": len(ds),
        "seed": cfg.seed,
    })
    return rec


def run_experiment(ds: URLDataset, cfg: ExperimentConfig, verbose: bool = False) -> ExperimentResult:
    t0 = time.time()
    assumptions: list[str] = []
    if cfg.mode == "centralized" and cfg.centralized_epochs < 1:
        raise ValueError(f"centralized_epochs must be >= 1, got {cfg.centralized_epochs}")
    if cfg.mode == "federated" and cfg.vectorizer != "hashing":
        raise ValueError(
            f"vectorizer {cfg.vectorizer!r} is not allowed for federated mode; "
            "federated training requires a stateless hashing vectorizer to prevent n-gram leakage (features.py)"
        )

    # ---- validate ---------------------------------------------------------- #
    vrep = validate_dataset(ds)
    assert_validation_passed(vrep)

    # ---- deduplicate (before splitting, never after) ----------------------- #
    ds, drep = deduplicate(ds, conflict="drop")
    assert_validation_passed(validate_dataset(ds))
    if drep.n_label_conflicts:
        assumptions.append(
            f"{drep.n_label_conflicts} URLs carried conflicting labels in the source "
            "and were dropped; this is a lower bound on pre-existing label noise"
        )

    # ---- split + leakage gate ---------------------------------------------- #
    splits = make_splits(ds, strategy=cfg.split_strategy, fractions=cfg.fractions, seed=cfg.seed)
    assert_no_leakage(splits.report)

    # ---- features ---------------------------------------------------------- #
    vec = build_vectorizer(cfg.vectorizer, n_features=cfg.n_features, ngram_range=cfg.ngram_range)
    if cfg.vectorizer == "hashing":
        X_train = vec.transform(splits.train.texts)
    else:
        # TF-IDF is fit on TRAIN ONLY. Fitting on the full corpus would leak test
        # vocabulary and inflate results.
        X_train = vec.fit_transform(splits.train.texts)
        assumptions.append(
            "TF-IDF vectoriser fit on training data only; not valid for a federated "
            "run because it requires a globally shared vocabulary (features.py)"
        )
    X_val = vec.transform(splits.val.texts)
    X_test = vec.transform(splits.test.texts)
    n_features = X_train.shape[1]

    y_train = splits.train.labels.copy()
    y_val = mark_protected(splits.val.labels)
    y_test = mark_protected(splits.test.labels)

    partition_report = None
    noise_report = None
    history: list[dict] = []
    selected_checkpoint: dict = {}
    # ---- train ------------------------------------------------------------- #
    if cfg.mode == "centralized":
        if cfg.noise_process not in ("none", "symmetric", "class_conditional"):
            raise ValueError(f"noise process {cfg.noise_process!r} requires federated mode")
        if cfg.noise_process != "none":
            y_train, nrep = apply_noise(
                y_train,
                process=cfg.noise_process,
                rate=cfg.noise_rate,
                class_rates=cfg.noise_class_rates,
                seed=cfg.seed,
                tolerance=cfg.noise_tolerance,
            )
            assert_noise_valid(nrep)
            noise_report = nrep.render()

        model = LogisticSGD(n_features, l2=cfg.l2, seed=cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        snapshots, scores = [], []
        for ep in range(cfg.centralized_epochs):
            model.fit(X_train, y_train, epochs=1, batch_size=cfg.batch_size, lr=cfg.lr, rng=rng)
            m = classification_metrics(y_val, model.predict_proba(X_val))
            snapshots.append(model.get_params())
            scores.append(m["pr_auc"])
            history.append({"epoch": ep + 1, "val_pr_auc": m["pr_auc"]})
            if verbose:
                print(f"  epoch {ep + 1:>3}  val_pr_auc={m['pr_auc']:.4f}")
        best_params, best_ep, best_val = best_checkpoint(snapshots, scores)
        model.set_params(best_params)
        final = model
        selected_checkpoint = {"kind": "epoch", "index": best_ep, "val_pr_auc": best_val}
        assumptions.append(
            f"centralized model selected at epoch {best_ep} by validation PR-AUC"
        )
        budget = describe_budget(
            len(y_train),
            passes_per_example(centralized_epochs=cfg.centralized_epochs),
        )

    elif cfg.mode == "federated":
        parts, prep = partition(
            y_train,
            strategy=cfg.partition_strategy,
            n_clients=cfg.n_clients,
            seed=cfg.seed,
            **cfg.partition_params,
        )
        if not prep.status.startswith("PASS"):
            raise RuntimeError(f"partition gate failed: {prep.status}\n\n{prep.render()}")
        partition_report = prep.render()

        if cfg.noise_process == "client_heterogeneous":
            rates = cfg.noise_client_rates
            if rates is None or len(rates) != cfg.n_clients:
                raise ValueError("client_heterogeneous noise needs one rate per client")
            y_train, nrep = apply_client_noise(
                y_train, parts, rates, seed=cfg.seed, aggregate_tolerance=cfg.noise_tolerance
            )
            assert_noise_valid(nrep)
            noise_report = nrep.render()
        elif cfg.noise_process != "none":
            y_train, nrep = apply_noise(
                y_train,
                process=cfg.noise_process,
                rate=cfg.noise_rate,
                class_rates=cfg.noise_class_rates,
                seed=cfg.seed,
                tolerance=cfg.noise_tolerance,
            )
            assert_noise_valid(nrep)
            noise_report = nrep.render()

        clients = []
        for cid, idx in enumerate(parts):
            ccfg = ClientConfig(
                local_epochs=cfg.local_epochs,
                batch_size=cfg.batch_size,
                lr=cfg.lr,
                l2=cfg.l2,
                behaviour=cfg.malicious_behaviour if cid in cfg.malicious_clients else "honest",
            )
            clients.append(
                FederatedClient(
                    client_id=cid,
                    X=X_train,
                    y=y_train[idx],
                    n_features=n_features,
                    config=ccfg,
                    seed=cfg.seed * 1000 + cid,
                    indices=idx,
                )
            )
        if cfg.malicious_clients:
            assumptions.append(
                f"clients {cfg.malicious_clients} run behaviour "
                f"'{cfg.malicious_behaviour}' (THREAT_MODEL.md §4, untargeted only)"
            )

        run = run_federated(
            clients,
            n_features,
            X_val,
            y_val,
            FederatedConfig(
                rounds=cfg.rounds,
                participation=cfg.participation,
                aggregation=cfg.aggregation,
                aggregation_params=cfg.aggregation_params,
                max_update_norm=cfg.max_update_norm,
                seed=cfg.seed,
            ),
            verbose=verbose,
        )
        assumptions.extend(run.assumptions)
        history = [
            {"round": r.round, "loss": r.mean_train_loss,
             "val_pr_auc": r.val_metrics["pr_auc"] if r.val_metrics else None}
            for r in run.history
        ]
        final = LogisticSGD(n_features, l2=cfg.l2)
        final.set_params(run.best_params)  # selected on VALIDATION only
        selected_checkpoint = {
            "kind": "round",
            "index": run.best_round,
            "val_pr_auc": run.best_val_pr_auc,
        }
        assumptions.append(
            f"global model selected at round {run.best_round} by validation PR-AUC"
        )
        budget = describe_budget(
            len(y_train),
            passes_per_example(
                rounds=cfg.rounds,
                local_epochs=cfg.local_epochs,
                participation=cfg.participation,
            ),
        )
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")

    # ---- evaluate: validation, then the clean test set exactly once --------- #
    val_metrics = classification_metrics(y_val, final.predict_proba(X_val))
    test_metrics = classification_metrics(y_test, final.predict_proba(X_test))

    projected = {
        str(br): precision_at_base_rate(test_metrics["recall"], test_metrics["fpr"], br)
        for br in cfg.assumed_deployment_base_rates
    }

    return ExperimentResult(
        config=asdict(cfg),
        provenance=_provenance(ds, cfg),
        dataset_report=vrep.render(),
        dedup_report=drep.render(),
        split_report=splits.report.render(),
        partition_report=partition_report,
        noise_report=noise_report,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        projected_precision=projected,
        history=history,
        assumptions=assumptions,
        wall_seconds=time.time() - t0,
        training_budget=budget,
        selected_checkpoint=selected_checkpoint,
    )

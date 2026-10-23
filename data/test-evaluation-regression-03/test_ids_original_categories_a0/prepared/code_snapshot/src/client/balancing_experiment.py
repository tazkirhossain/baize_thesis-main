"""One client's reproducible numerical-feature balancing experiments.

All four methods train and evaluate here using the SAME engineered feature
family. Row-index samplers are separately reusable by future neural trainers.
This is local model evaluation, not Flower/FedAvg or DeBERTa/CNN training.
"""
from dataclasses import asdict, replace
from pathlib import Path
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits

from baize.data.preprocessing import _file_hash
from src.client.data import read_local_split
from src.data.balancing import BalanceConfig, class_counts, print_balance_report, sample_training_indices
from src.data.numerical_features import URLNumericTransformer
from src.data.smote import smote_training
from src.evaluation.balancing_metrics import balancing_metrics, print_metrics
from src.models.numerical_logistic import numerical_classifier
from src.models.undersampling_ensemble import UndersamplingEnsemble
from src.models.training_audit import fit_recorded


def fit_numerical_training(data, config, model_factory, *, verbose=True):
    """Reusable fitting API returning the trained numerical model and sampling log.

    The transformer is fitted separately on original training only. Callers can
    retain/persist the returned model in ignored local checkpoint storage.
    """
    config.validate()
    data.validate()
    if config.method == "ensemble_undersampling":
        model = UndersamplingEnsemble(model_factory,config).fit_training(data.X,data.y,verbose=verbose)
        sampling = {"method":config.method,"members":model.reports_,
                    "aggregation":"mean_adult_probability", "ensemble_members":config.ensemble_members,
                    "seed":config.seed}
    elif config.method == "smote":
        X,y,sampling = smote_training(data,config)
        if verbose:
            print_balance_report(sampling)
            print(f"  SMOTE neighbors: requested={config.k_neighbors}, used={sampling['effective_k_neighbors']}; "
                  f"status={sampling['status']}",flush=True)
        if sampling["status"].startswith("skipped"):
            return None,sampling
        model = fit_recorded(model_factory(config.seed),X,y)
    else:
        selection = sample_training_indices(data.y,config,split=data.split)
        sampling = selection.report
        if verbose:
            print_balance_report(sampling)
        if sampling["status"] == "skipped_missing_class":
            return None,sampling
        model = fit_recorded(model_factory(config.seed),data.X[selection.indices],data.y[selection.indices])
    if config.method != "ensemble_undersampling":
        sampling["fit_audit"] = model.fit_audit_
        if verbose:
            print(f"  Fit iterations={model.fit_audit_['iterations']}; warnings={len(model.fit_audit_['warnings'])}",flush=True)
    return model,sampling


def run_local_experiments(client_directory, experiment, *, methods=None, verbose=True):
    allowed = {"balancing","model","threshold"}
    if set(experiment)-allowed:
        raise ValueError(f"unknown experiment fields: {sorted(set(experiment)-allowed)}")
    base = BalanceConfig(**experiment["balancing"])
    base.validate()
    model_cfg = experiment["model"]
    if set(model_cfg)-{"family","classifier","parameters"} or model_cfg.get("family") != "engineered_numeric":
        raise ValueError("this runner trains engineered_numeric only; DeBERTa/char CNN need separate trainers")
    classifier = model_cfg.get("classifier","sgd_logistic")
    parameters = model_cfg.get("parameters",{})
    numerical_classifier(classifier,seed=base.seed,parameters=parameters)
    threshold = experiment.get("threshold",0.5)
    if not isinstance(threshold,(int,float)) or not 0<threshold<1:
        raise ValueError("fixed threshold must be between zero and one")
    methods = list(methods) if methods is not None else [base.method]
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("select distinct balancing methods")
    for method in methods:
        replace(base,method=method).validate()
    directory = Path(client_directory)
    train = read_local_split(directory,"train")
    if verbose:
        print(f"Local training only: {class_counts(train.labels)}; numerical features, classifier={classifier}",flush=True)
    if len(np.unique(train.labels)) != 2:
        return {"status":"SKIPPED", "reason":"training split lacks a binary class", "runs":[]}
    transformer = URLNumericTransformer()
    data = transformer.fit_training(train)
    # Holdouts are read only after the training transform has been fitted.
    holdouts = {name:read_local_split(directory,name) for name in ("val","test")}
    for name, split in holdouts.items():
        if np.intersect1d(train.source_rows,split.source_rows).size:
            raise ValueError("training and evaluation source records overlap")
    if np.intersect1d(holdouts["val"].source_rows,holdouts["test"].source_rows).size:
        raise ValueError("validation and test records overlap")
    evaluation = {name:transformer.transform(split) for name,split in holdouts.items()}
    contracts = {split.split:split.sha256 for split in (train,*holdouts.values())}
    results = []
    for method in methods:
        cfg = replace(base,method=method)
        start = time.perf_counter()
        if verbose:
            print(f"\n=== {method}: engineered_numeric / {classifier}, fixed threshold={threshold} ===",flush=True)
        factory = lambda seed: numerical_classifier(classifier,seed=seed,parameters=parameters)
        with threadpool_limits(limits=1):
            model, sampling = fit_numerical_training(data,cfg,factory,verbose=verbose)
            training_seconds = time.perf_counter() - start
            if model is None:
                results.append({"status":"SKIPPED", "method":method, "sampling":sampling,
                                "experiment_family":"engineered_numeric", "metrics":{},
                                "data_contract":contracts})
                continue
            metrics = {}
            member_metrics = {}
            for name, split in holdouts.items():
                positive = int(np.flatnonzero(model.classes_ == 1)[0])
                metrics[name] = balancing_metrics(split.labels,model.predict_proba(evaluation[name])[:,positive],threshold=threshold)
                if verbose:
                    print_metrics(metrics[name],split=name)
                if method == "ensemble_undersampling":
                    member_metrics[name] = [balancing_metrics(split.labels,m.predict_proba(evaluation[name])[:,int(np.flatnonzero(m.classes_==1)[0])],threshold=threshold)
                                            for m in model.models_]
        # Refuse to report a run if any original split changed during training.
        for name, expected in contracts.items():
            if _file_hash(directory / f"splits/{name}.jsonl") != expected:
                raise ValueError("original training/evaluation file changed during experiment")
        results.append({"status":"PASS", "experiment_family":"engineered_numeric",
            "neural_models_trained":[],"method":method,"balancing_config":asdict(cfg),
            "model":{"family":"engineered_numeric","classifier":classifier,
                     "parameters":factory(cfg.seed).get_params()},
            "training_original_counts":class_counts(train.labels),"sampling":sampling,
            "metrics":metrics,"ensemble_member_metrics":member_metrics,
            "training_seconds": training_seconds,
            "evaluation_and_audit_seconds": time.perf_counter()-start-training_seconds,
            "elapsed_seconds":time.perf_counter()-start,"data_contract":contracts,
            "feature_transform":transformer.report(),
            "checks":{"balancing_training_only":True,"scaler_fit_training_only":True,
                      "evaluation_files_unchanged":True,"original_training_file_unchanged":True,
                      "fixed_threshold_no_holdout_tuning":True}})
        del model
    return {"status":"PASS" if all(r["status"]=="PASS" for r in results) else "COMPLETED_WITH_SKIPS",
            "runs":results,"input_split_hashes":contracts}

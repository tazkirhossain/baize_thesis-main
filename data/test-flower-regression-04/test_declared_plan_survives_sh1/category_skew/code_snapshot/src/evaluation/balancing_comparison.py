"""Aggregate local prediction counts without implying a global federated model."""
from collections import defaultdict

import numpy as np


def pooled_confusion_metrics(matrix):
    cm = np.asarray(matrix)
    if (cm.shape != (2,2) or not np.isfinite(cm).all() or (cm < 0).any() or cm.sum() <= 0 or
            not np.array_equal(cm,np.floor(cm))):
        raise ValueError("a nonnegative nonempty 2x2 confusion matrix is required")
    tn,fp,fn,tp = [int(v) for v in cm.ravel()]
    divide = lambda a,b:a/b if b else 0.0
    p,r = divide(tp,tp+fp),divide(tp,tp+fn)
    f1 = divide(2*tp,2*tp+fp+fn)
    return {"adult_precision":p,"adult_recall":r,"adult_f1":f1,
            "macro_f1":(f1+divide(2*tn,2*tn+fp+fn))/2,
            "balanced_accuracy":(r+divide(tn,tn+fp))/2,
            "accuracy":divide(tp+tn,tn+fp+fn+tp),"confusion_matrix":[[tn,fp],[fn,tp]],
            "n_records":tn+fp+fn+tp}


def summarize_balancing(summary):
    if summary.get("model_family") != "engineered_numeric":
        raise ValueError("this summary is for numerical-feature experiments only")
    grouped = defaultdict(list)
    skipped = []
    for client in summary["clients"]:
        reference = None
        for run in client["runs"]:
            if run["status"] != "PASS":
                skipped.append({"partition":client["partition"],"client":client["client"],
                                "method":run["method"],"status":run["status"]})
                continue
            signature = (run["data_contract"],run["feature_transform"]["fingerprint"],
                         run["model"],run["metrics"]["val"]["threshold"])
            if reference is not None and signature != reference:
                raise ValueError("method comparison has mismatched inputs, transform, classifier or threshold")
            reference = signature
            grouped[client["partition"],run["method"]].append(run)
    groups = []
    for (partition,method),runs in sorted(grouped.items()):
        metrics = {}
        for split in ("val","test"):
            matrix = np.sum([r["metrics"][split]["confusion_matrix"] for r in runs],axis=0)
            metrics[split] = pooled_confusion_metrics(matrix)
            metrics[split]["mean_client_adult_f1"] = float(np.mean([r["metrics"][split]["adult_f1"] for r in runs]))
            metrics[split]["mean_client_average_precision"] = float(np.mean([
                r["metrics"][split]["pr_auc_average_precision"] for r in runs]))
        audits = []
        for run in runs:
            if method == "ensemble_undersampling":
                audits.extend(m["fit_audit"] for m in run["sampling"]["members"])
            else:
                audits.append(run["sampling"]["fit_audit"])
        groups.append({"partition":partition,"method":method,"n_clients":len(runs),
                       "metrics":metrics,"fit_warning_count":sum(len(a["warnings"]) for a in audits)})
    return {"scope":"pooled local numerical-model predictions; not a global FL model or neural results",
            "model_family":"engineered_numeric","automatic_method_selection":False,
            "groups":groups,"skipped":skipped}


def comparison_markdown(result):
    lines = ["# Four-method numerical comparison", "", result["scope"], "",
        "Single predeclared-seed comparison. No method was automatically selected. "
        "All methods use original holdouts; warnings and per-client metrics are retained in the source reports.", "",
        "Metrics below pool the confusion counts of independent client models on their own holdouts. "
        "They do not describe a model trained by federated aggregation.", "",
        "| Partition | Method | Val Adult F1 | Test Adult precision | Test Adult recall | Test Adult F1 | Test macro-F1 | Test balanced accuracy | Test accuracy | Fit warnings |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in result["groups"]:
        m = group["metrics"]["test"]
        values = [group["metrics"]["val"]["adult_f1"],m["adult_precision"],m["adult_recall"],
                  m["adult_f1"],m["macro_f1"],m["balanced_accuracy"],m["accuracy"]]
        lines.append(f"| {group['partition']} | {group['method']} | "+" | ".join(f"{v:.4f}" for v in values)+
                     f" | {group['fit_warning_count']} |")
    lines += ["", "Confusion-matrix order is [Non-adult, Adult]. Detailed matrices and client-level "
              "results are in the JSON reports. No DeBERTa or character CNN results are included.", ""]
    return "\n".join(lines)

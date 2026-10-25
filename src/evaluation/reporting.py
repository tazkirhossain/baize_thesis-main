"""Source-backed comparison adapters. No raw datasets or predictions are read."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from src.evaluation.classification import confusion_metrics

METRICS = ("adult_precision", "adult_recall", "adult_f1", "non_adult_precision",
           "non_adult_recall", "non_adult_f1", "macro_f1", "balanced_accuracy", "accuracy")


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_metrics(metrics):
    computed = confusion_metrics(metrics["confusion_matrix"])
    for key in (*METRICS, "n_records", "adult_count", "non_adult_count"):
        if key in metrics and not np.isclose(metrics[key], computed[key], atol=1e-10):
            raise ValueError(f"reported {key} disagrees with confusion counts")
    return {**computed, **{key: metrics[key] for key in ("threshold", "loss") if key in metrics}}


def numerical_rows(path, settings):
    path = Path(path)
    summary = json.loads(path.read_text())
    if summary.get("model_family") != "engineered_numeric" or not summary.get("clients"):
        raise ValueError("expected completed numerical local-experiment summary")
    source_id = fingerprint(path)
    grouped, metadata, missing = defaultdict(list), {}, []
    references = {}
    for client in summary["clients"]:
        partition = settings["partition_aliases"].get(client["partition"], client["partition"])
        for run in client["runs"]:
            if run["status"] != "PASS":
                missing.append({"partition": partition, "method": run["method"], "status": run["status"]})
                continue
            if run["balancing_config"]["adult_to_non_adult"] != settings["expected_adult_to_non_adult"]:
                raise ValueError("balancing ratio differs from the requested reporting cohort")
            for split in ("val", "test"):
                checked_metrics(run["metrics"][split])
            key = (partition, run["method"])
            reference_key = partition, client["client_id"]
            reference = (run["data_contract"], run["feature_transform"]["fingerprint"],
                         run["model"], run["metrics"]["val"]["threshold"])
            if reference_key in references and references[reference_key] != reference:
                raise ValueError("balancing methods use different inputs, feature transforms, models or thresholds")
            references[reference_key] = reference
            grouped[key].append((client["client_id"], run))
    rows = []
    for (partition, method), runs in sorted(grouped.items()):
        if len(runs) != settings["expected_clients"] or len({cid for cid, _ in runs}) != len(runs):
            raise ValueError("a numerical comparison group must contain all five unique clients")
        group_id = f"{source_id[:12]}:{partition}:{method}"
        metadata[group_id] = {
            "configuration": summary["configuration"], "runtime": summary.get("environment", {}),
            "code_tree_sha256": summary.get("reproduction", {}).get("source_tree_sha256"),
            "source_summary_sha256": source_id,
            "clients": [{"client_id": cid, "balancing": r["balancing_config"], "model": r["model"],
                         "data_contract": r["data_contract"], "feature_transform_fingerprint": r["feature_transform"]["fingerprint"]}
                        for cid, r in sorted(runs)],
        }
        warnings = 0
        for _, run in runs:
            audits = run["sampling"].get("members", [run["sampling"]])
            warnings += sum(len(member.get("fit_audit", {}).get("warnings", [])) for member in audits)
        for split in ("val", "test"):
            matrix = np.sum([r["metrics"][split]["confusion_matrix"] for _, r in runs], axis=0)
            rows.append({"experiment_id": group_id, "scope": "pooled independent local numerical models",
                         "family": "engineered_numeric", "data_kind": "real", "partition": partition,
                         "classifier": runs[0][1]["model"]["classifier"], "method": method,
                         "ratio": settings["expected_adult_to_non_adult"], "split": split, "round": None,
                         "seed": summary["configuration"]["balancing"]["seed"],
                         "metrics": dict(confusion_metrics(matrix), threshold=runs[0][1]["metrics"][split]["threshold"]),
                         "timing": {"total_elapsed_seconds_sum": sum(r["elapsed_seconds"] for _, r in runs),
                                    "training_seconds_sum": sum(r["training_seconds"] for _, r in runs)
                                    if all("training_seconds" in r for _, r in runs) else None},
                         "communication": None, "fit_warning_count": warnings})
    return rows, metadata, missing


def checked_selection(cfg, run, rounds):
    """Verify a completed run's selected model from selection counts, not holdouts."""
    if not cfg.get('model_selection'):
        return
    decision = run.get('selection')
    if (not isinstance(decision, dict) or decision.get('source') != 'training_selection'
            or decision.get('metric') != 'adult_f1' or decision.get('tie_break') != 'earliest round'):
        raise ValueError('invalid training-only checkpoint selection metadata')
    candidates = []
    for report in rounds:
        measured = checked_metrics(report['selection'])
        if not measured['adult_count'] or not measured['non_adult_count']:
            raise ValueError('training-selection data must include both classes')
        candidates.append((measured['adult_f1'], report['round'], report['global_hash']))
    best = max(candidates, key=lambda item:(item[0], -item[1]))
    if (decision.get('score') != best[0] or decision.get('round') != best[1]
            or decision.get('global_hash') != best[2]):
        raise ValueError('selected checkpoint disagrees with training-selection counts')
    for report in rounds:
        if 'test' in report:
            reference = report.get('test_checkpoint', {})
            if reference.get('round') != best[1] or reference.get('global_hash') != best[2] or reference.get('source') != 'training_selection':
                raise ValueError('test model does not match selected checkpoint')


def federated_rows(path, settings):
    path = Path(path)
    run = json.loads((path / "run.json").read_text())
    if run.get("status") != "PASS":
        raise ValueError("only successful completed federated runs may be compared")
    cfg = run["configuration"]
    props = run.get("client_data_contracts", {})
    kinds = {value.get("data_kind") for value in props.values()} - {None}
    partitions = {value.get("partition_strategy") for value in props.values()} - {None}
    declared = run.get("experiment", cfg.get("experiment", {}))
    data_kind = next(iter(kinds)) if len(kinds) == 1 else declared.get("data_kind", "unknown")
    partition = next(iter(partitions)) if len(partitions) == 1 else declared.get("partition_strategy", "unknown")
    partition = settings["partition_aliases"].get(partition, partition)
    source_id = fingerprint(path / "run.json")
    experiment_id = source_id[:16]
    rows, rounds = [], []
    expected = list(range(1, cfg["rounds"]+1))
    files = sorted(path.glob("round_*.json"))
    if len(files) != len(expected):
        raise ValueError("federated round history is incomplete")
    source_hashes = {"run.json": source_id}
    for index, file in zip(expected, files):
        report = json.loads(file.read_text())
        if report["round"] != index:
            raise ValueError("federated rounds must be contiguous")
        source_hashes[file.name] = fingerprint(file)
        rounds.append(report)
        for split in ("train", "val", "test"):
            if split not in report:
                continue
            if split == "test" and index != cfg["rounds"]:
                raise ValueError("test data were evaluated before the fixed final round")
            rows.append({"experiment_id": experiment_id, "scope": "federated global model",
                         "family": cfg["model"]["family"], "data_kind": data_kind, "partition": partition,
                         "classifier": cfg["model"]["family"], "method": cfg["balancing"]["method"],
                         "ratio": cfg["balancing"]["adult_to_non_adult"], "seed": cfg["seed"],
                         "training_protocol": "encoder_embeddings" if cfg.get("representation") else "end_to_end",
                         "input_mode": "url_plus_metadata" if cfg.get('website_enrichment') else "url_only",
                         "split": split, "round": index, "metrics": checked_metrics(report[split]),
                         "checkpoint_round": report.get('test_checkpoint', {}).get('round', index) if split == 'test' else index,
                         "checkpoint_hash": report.get('test_checkpoint', {}).get('global_hash', report.get('global_hash')) if split == 'test' else report.get('global_hash'),
                         "optimization": report.get("optimization"), "timing": report.get("timing"),
                         "balancing": report.get("balancing"),
                         "communication": report.get("communication"), "fit_warning_count": None})
    checked_selection(cfg, run, rounds)
    metadata = {experiment_id: {"configuration": cfg, "scope": "federated global model",
                               "data_contracts": props, "source_files_sha256": source_hashes,
                               "final_global_hash": run["global_hash"],
                               "client_runtime": run.get("client_runtime", {}),
                               "server_runtime": run.get("server_runtime", {}),
                               "selection": run["selection"], "declared_experiment": declared}}
    parent = path.parent
    if (parent / "reproduction.json").exists():
        metadata[experiment_id]["reproduction"] = json.loads((parent / "reproduction.json").read_text())
    # Never export raw per-client records; only pooled round rows and provenance.
    return rows, metadata


def verify_comparability(rows, metadata):
    """Require matching contracts within each method-comparison stratum.

    Different model families, execution scopes, data kinds, seeds and policies
    remain separate. Cross-partition population identity is an operator-declared
    audited contract; identical counts alone are not proof of equal membership.
    """
    strata, populations = defaultdict(list), defaultdict(list)
    unique = set()
    for row in rows:
        identity = (row["scope"], row["family"], row["data_kind"], row["seed"],
                    row["partition"], row["method"], row["split"], row["round"])
        if identity in unique:
            raise ValueError("duplicate comparison cell; separate seeds/configurations into different reports")
        unique.add(identity)
        if row["scope"] != "federated global model" or row["split"] != "val":
            continue
        cfg = metadata[row["experiment_id"]]["configuration"]
        key = row["family"], row["data_kind"], row["seed"], row["round"]
        signature = {name: cfg[name] for name in ("model", "local_epochs", "batch_size", "learning_rate", "rounds", "clients_per_round", "aggregation_weight", "threshold")}
        balance = dict(cfg["balancing"])
        balance.pop("method", None)
        signature["balancing_settings"] = balance
        signature["representation"] = cfg.get("representation")
        signature['model_selection'] = cfg.get('model_selection')
        signature['website_enrichment'] = cfg.get('website_enrichment')
        from src.data.enrichment.config import pooled_digest
        signature['website_input_digest'] = pooled_digest(metadata[row['experiment_id']].get('data_contracts', {}))
        signature["evaluation_population_id"] = cfg.get("experiment", {}).get("evaluation_population_id")
        populations[(*key, row["partition"])].append(json.dumps(metadata[row["experiment_id"]].get("data_contracts"), sort_keys=True))
        strata[key].append(json.dumps(signature, sort_keys=True))
    if (any(len(set(signatures)) > 1 for signatures in strata.values()) or
            any(len(set(signatures)) > 1 for signatures in populations.values())):
        raise ValueError("federated method comparisons have incompatible data/settings/weighting policies")


def comparison_coverage(rows, settings):
    coverage = []
    for family in ("engineered_numeric", "character_cnn", "deberta_v3_base"):
        for partition in ("iid", "label_skew", "category_skew"):
            for method in settings["required_balancing_methods"]:
                actual = [row for row in rows if row["family"] == family and row["partition"] == partition
                          and row["method"] == method and row["data_kind"] == "real"]
                coverage.append({"family": family, "partition": partition, "method": method,
                                 "validation_measured": any(row["split"] == "val" for row in actual),
                                 "final_test_measured": any(row["split"] == "test" for row in actual),
                                 "status": "measured" if actual else "not measured"})
    return coverage


def report_markdown(report, plot_files):
    lines = ["# Evaluation and comparison results", "",
             "No method is declared best. Validation and final test measurements are reported separately; "
             "single-seed results and convergence warnings limit conclusions.", "",
             "Numerical rows pool predictions from independent local numerical models. "
             "Flower rows describe a shared federated model. Synthetic smoke results are not real-data evidence.", "",
             f"Recorded numerical fit warnings: **{report.get('fit_warning_count_total', 'not supplied')}**. "
             "Counts are retained per method/scenario in report.json; warnings are not hidden or used to retune against holdouts.", "",
             "Encoder-embedding rows use training-only encoder warmup and a frozen-encoder balanced head phase; they are a distinct protocol from end-to-end training. Never interpolate raw strings or token IDs.", "",
             "With training-only checkpoint selection, train metrics exclude the selection pool. The test is measured once after the FL budget on the selected checkpoint; its checkpoint round may differ from the final communication round. Per-experiment selection metadata and checkpoint hashes/rounds are retained in report.json.", "",
             "Website-enriched and URL-only inputs are separate experiment contracts. The input mode is shown alongside model/protocol; source configuration and pooled feature fingerprints remain in report.json.", "",
             "## All requested classification metrics", "",
             "| Family / data / protocol | Partition | Method | Split / round | Adult P | Adult R | Adult F1 | Non-adult P | Non-adult R | Non-adult F1 | Macro-F1 | Balanced accuracy | Accuracy | N |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["rows"]:
        m = row["metrics"]
        values = " | ".join(f"{m[key]:.4f}" for key in METRICS)
        lines.append(f"| {row['family']} / {row['data_kind']} / {row.get('training_protocol', 'historical')} / {row.get('input_mode', 'historical')} | {row['partition']} | {row['method']} | "
                     f"{row['split']} / {row['round'] or 'local'} | {values} | {m['n_records']:,} |")
    lines += ["", "Confusion matrices, class supports, prevalence, source hashes, exact configuration, timing "
              "and communication measurements are retained in report.json. Undefined class ratios use zero; "
              "macro-F1 and balanced recall use the fixed two-class label set.", "",
              "## Confusion matrices", "",
              "True rows and predicted columns use [Non-adult, Adult].", "",
              "| Family / data | Partition | Method | Split / round | TN | FP | FN | TP |",
              "|---|---|---|---|---:|---:|---:|---:|"]
    for row in report["rows"]:
        tn, fp, fn, tp = np.asarray(row["metrics"]["confusion_matrix"]).ravel()
        lines.append(f"| {row['family']} / {row['data_kind']} / {row.get('training_protocol', 'historical')} | {row['partition']} | {row['method']} | "
                     f"{row['split']} / {row['round'] or 'local'} | {tn:,} | {fp:,} | {fn:,} | {tp:,} |")
    lines += ["", "## Timing and communication", "",
              "One row per experiment/round, not repeated for each metric split. Missing measurements are not zeros.", "",
              "| Family / data | Partition | Method | Round | Sum client training seconds | Total local elapsed seconds | Round wall seconds | Protobuf payload MiB |",
              "|---|---|---|---|---:|---:|---:|---:|"]
    display = lambda value: "not recorded" if value is None else f"{value:.3f}"
    for row in report["rows"]:
        if row["split"] != "val":
            continue
        timing = row.get("timing") or {}
        communication = row.get("communication") or {}
        payload = communication.get("total_bytes")
        values = [timing.get("training_seconds_sum", timing.get("client_training_seconds_sum")),
                  timing.get("total_elapsed_seconds_sum"), timing.get("round_wall_seconds"),
                  payload/1024**2 if payload is not None else None]
        lines.append(f"| {row['family']} / {row['data_kind']} / {row.get('training_protocol', 'historical')} | {row['partition']} | {row['method']} | "
                     f"{row['round'] or 'local'} | " + " | ".join(display(value) for value in values) + " |")
    lines += ["",
              "## Figures", ""]
    for file in plot_files:
        if file.endswith(".png"):
            lines.append(f"![{Path(file).stem.replace('_', ' ')}]({file})\n")
    lines += ["## Measurement limits", ""]
    lines.extend(f"- {note}" for note in report["limitations"])
    lines += ["", "## Real-data experiment coverage", "",
              "| Family | Partition | Method | Validation | Final test | Status |", "|---|---|---|---|---|---|"]
    for item in report["coverage"]:
        lines.append(f"| {item['family']} | {item['partition']} | {item['method']} | "
                     f"{'measured' if item['validation_measured'] else 'not measured'} | "
                     f"{'measured' if item['final_test_measured'] else 'not measured'} | {item['status']} |")
    return "\n".join(lines) + "\n"

"""Frozen experiment numbers for the demo dashboard.

The frontend does not rerun E1–E9. It reads the committed JSON under
`experiments/results/` so the visualisation stays in lock-step with RESULTS.md.
H1 is still an unmatched-budget comparison; that caveat is part of the payload.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"


def _load(name: str) -> dict | None:
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def research_snapshot() -> dict:
    e1 = _load("E1_summary_lr160.json") or {}
    h1 = _load("E2_H1_decision.json") or {}
    h4 = _load("E5_H4_decision.json") or {}
    h3 = _load("E9_H3_decision.json") or {}
    h5 = _load("E8_H5_decision.json") or {}

    sigma_c = e1.get("sigma_c", h1.get("sigma_c"))

    return {
        "sigma_c": sigma_c,
        "primary_metric": "PR-AUC",
        "positive_rate_note": "Adult is ~2.4% of corpus rows; PR-AUC must be quoted with that base rate.",
        "corpus_era": "Pre-~2010 directory web (HTTPS 0.068% of rows).",
        "hypotheses": [
            {
                "id": "RQ1",
                "title": "Random splits inflate PR-AUC",
                "decision": "MEASURED",
                "bars": [
                    {"label": "Random split", "value": e1.get("random", {}).get("mean_pr_auc")},
                    {"label": "Domain-aware (primary)", "value": e1.get("domain_aware", {}).get("mean_pr_auc")},
                ],
                "delta": e1.get("inflation_abs"),
                "sigma": e1.get("inflation_sigma_c"),
                "n_seeds": len(e1.get("domain_aware", {}).get("seeds") or []),
                "note": "A fifth of random-split score is registrable-domain memorisation.",
            },
            {
                "id": "H1",
                "title": "Federation is nearly free under IID data",
                "decision": h1.get("H1", "UNKNOWN"),
                "bars": [
                    {"label": "Centralized", "value": h1.get("centralized_mean")},
                    {"label": "FedAvg IID", "value": h1.get("fedavg_mean")},
                ],
                "delta": h1.get("mean_gap"),
                "sigma": h1.get("gap_in_sigma_c"),
                "n_seeds": len(h1.get("per_seed_gap") or {}),
                "note": (
                    "SUPPORTED at this compute, but not matched-budget: "
                    "centralized 35 passes/example vs FedAvg rounds×local_epochs."
                ),
            },
            {
                "id": "H4",
                "title": "Heterogeneity is the real cost",
                "decision": h4.get("H4", "UNKNOWN"),
                "bars": [
                    {"label": "FedAvg IID", "value": h4.get("iid_mean")},
                    {"label": "Label skew α=1", "value": h4.get("noniid_mean")},
                ],
                "delta": h4.get("mean_gap"),
                "sigma": h4.get("gap_sigma_c"),
                "n_seeds": len(h4.get("seeds") or []),
                "note": "Architecture costs 0.003; dropping IID costs ~8.5× more.",
            },
            {
                "id": "H3",
                "title": "Robust agg pays only vs identifiable corruption",
                "decision": h3.get("H3", "UNKNOWN"),
                "bars": [
                    {"label": "FedAvg, 3/10 clients @ 40% noise", "value": _mean(h3.get("fedavg"))},
                    {"label": "Trimmed mean β=0.3, same noise", "value": _mean(h3.get("trimmed"))},
                ],
                "delta": h3.get("mean_gain"),
                "sigma": h3.get("gain_sigma_c"),
                "n_seeds": len(h3.get("seeds") or []),
                "note": (
                    "Heterogeneous noise (E9) is the decided condition. Homogeneous "
                    "20% noise (one seed) gave −0.004 — trimming hurts when there "
                    "is nothing in the tails."
                ),
            },
            {
                "id": "H5",
                "title": "Robust agg cost under benign skew",
                "decision": h5.get("H5", "UNKNOWN"),
                "bars": [
                    {"label": "FedAvg, label skew", "value": _mean(h5.get("fedavg"))},
                    {"label": "Trimmed mean, label skew", "value": _mean(h5.get("trimmed"))},
                ],
                "delta": h5.get("mean_cost"),
                "sigma": h5.get("cost_sigma_c"),
                "n_seeds": 3,
                "note": "NOT SUPPORTED: cost is indistinguishable from zero here.",
            },
        ],
        "pipeline": [
            {
                "id": "minimise",
                "title": "URL minimisation",
                "module": "baize.data.url",
                "keeps": "scheme, host, path",
                "drops": "query, fragment, userinfo, port",
            },
            {
                "id": "features",
                "title": "Hashed char 3–5-grams",
                "module": "baize.data.features",
                "keeps": "feature hashing, no vocabulary",
                "drops": "TF-IDF vocab (federated leak)",
            },
            {
                "id": "model",
                "title": "Logistic SGD",
                "module": "baize.models.linear",
                "keeps": "flat parameter vector for FedAvg",
                "drops": "page content, images, HTML",
            },
            {
                "id": "abstain",
                "title": "Abstention band",
                "module": "baize.evaluation.metrics",
                "keeps": "p outside [0.40, 0.60]",
                "drops": "forced 0/1 near 0.5",
            },
            {
                "id": "federated",
                "title": "FedAvg + robust aggregators",
                "module": "baize.federated",
                "keeps": "simulation of noisy / skewed clients",
                "drops": "real user devices, DP, secure aggregation",
            },
        ],
        "gates": [
            "LeakageError — duplicate URLs or domain overlap under domain-aware split",
            "NoiseValidationError — realised noise off-target, or noising a protected array",
            "ValidationError — single-class / degenerate corpus",
        ],
    }


def _mean(xs) -> float | None:
    if not xs:
        return None
    return float(sum(xs) / len(xs))

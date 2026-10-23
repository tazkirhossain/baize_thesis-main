"""Compare preparation contracts and aggregate distributions, not model quality."""
import json
from pathlib import Path


def compare_partitions(roots):
    roots = [Path(root) for root in roots]
    if len(roots) < 2:
        raise ValueError("at least two partition outputs are required")
    reports = [json.loads((root / "report.json").read_text(encoding="utf-8")) for root in roots]
    for report in reports:
        if report.get("status") != "PASS" or not report.get("comparison_contract"):
            raise ValueError("comparison requires PASS reports from the configurable pipeline; "
                             "regenerate older allocations into a new output directory")
    reference = reports[0]["comparison_contract"]
    for report in reports[1:]:
        other = report["comparison_contract"]
        differences = [key for key in reference.keys() | other.keys() if reference.get(key) != other.get(key)]
        if differences:
            raise ValueError(f"incompatible data/evaluation contracts: {sorted(differences)}")
    return {"status": "PASS", "comparison_contract": reference,
        "scope": "data preparation only; model performance has not been compared",
        "runs": [{"partitioning": report["partitioning"],
                  "distribution_spreads": report["distribution_spreads"],
                  "clients": [{"client_id": c["client_id"],
                    "n_records": c["dataset"]["n_records"],
                    "class_counts": c["dataset"]["class_counts"],
                    "category_counts": c["dataset"]["category_counts"]} for c in report["clients"]]}
                 for report in reports]}

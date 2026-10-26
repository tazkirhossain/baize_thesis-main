"""Export measured comparisons as JSON, Markdown tables and standalone PNG/SVG."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.reporting import (comparison_coverage, federated_rows, fingerprint,
    numerical_rows, report_markdown, verify_comparability)
from src.evaluation.plots import export_plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/reporting/comparison.json")
    parser.add_argument("--balancing-summary", type=Path, action="append", default=[])
    parser.add_argument("--federated-run", type=Path, action="append", default=[], help="server output directory containing run.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.balancing_summary and not args.federated_run:
        parser.error("provide completed numerical summaries and/or federated runs")
    if args.output.exists():
        raise FileExistsError("choose a fresh report output; historical results are never overwritten")
    settings = json.loads(args.config.read_text())
    if settings["automatic_method_selection"] is not False:
        raise ValueError("automatic winner selection is not supported")
    rows, metadata, missing = [], {}, []
    inputs = []
    for path in args.balancing_summary:
        extracted, provenance, skipped = numerical_rows(path, settings)
        rows.extend(extracted)
        metadata.update(provenance)
        missing.extend(skipped)
        inputs.append({"kind": "numerical summary", "name": path.name, "sha256": fingerprint(path)})
    for path in args.federated_run:
        extracted, provenance = federated_rows(path, settings)
        rows.extend(extracted)
        metadata.update(provenance)
        inputs.append({"kind": "federated run", "name": path.name, "sha256": fingerprint(path / "run.json")})
    verify_comparability(rows, metadata)
    report = {"schema_version": 1, "automatic_method_selection": False, "best_method": None,
              "fit_warning_count_total": sum(row["fit_warning_count"] or 0 for row in rows
                                             if row["split"] == "val" and row["round"] is None),
              "selection_policy": "no accuracy-only selection; validation and final test measurements required; no automatic ranking",
              "rows": rows, "experiments": metadata, "inputs": inputs, "missing_or_skipped": missing,
              "coverage": comparison_coverage(rows, settings), "report_configuration": settings,
              "limitations": [
                  "Existing real-data results are single-seed numerical experiments, not CNN/DeBERTa FL performance.",
                  "Neural real-data comparisons remain not measured until those experiments run; synthetic smoke is never substituted.",
                  "Convergence warnings are preserved. Single-seed measured validation/test results do not establish a robust best method.",
                  "Historical elapsed times include training/evaluation/audits; separate training time is null where it was not recorded.",
                  "Local numerical runs have no federated communication measurement. Historical missing timing/bytes remain null, not zero.",
                  "Flower byte counts cover serialized application payloads including chunk staging/fetch RPCs; they exclude registration, gRPC/TLS framing and retries.",
                  "Global train metrics use unresampled local optimization pools, excluding training-selection records when configured; optimization loss uses balanced views and is distinct.",
                  "Per-class undefined divisions use zero; macro-F1/balanced recall average the fixed [Non-adult, Adult] label set.",
                  "Data minimization is not differential privacy or secure aggregation; only counts and model states cross the FL API.",
                  "A declared cross-partition population label is not proof of identical membership; retain the preprocessing leakage/coverage audit.",
              ],
              "reproduction": {"python": platform.python_version(), "packages": {name: importlib.metadata.version(name)
                  for name in ("numpy", "matplotlib", "scikit-learn")},
                  "report_sources_sha256": {relative: fingerprint(ROOT / relative) for relative in
                      ("scripts/report_evaluation.py", "src/evaluation/reporting.py", "src/evaluation/plots.py", "src/evaluation/classification.py", "src/evaluation/thresholds.py")},
                  "settings_sha256": fingerprint(args.config)}}
    args.output.mkdir(parents=True)
    plots = export_plots(report, args.output, settings["formats"])
    report["plots"] = {name: fingerprint(args.output / name) for name in plots}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "README.md").write_text(report_markdown(report, plots), encoding="utf-8")
    print(f"Saved {len(rows)} measured metric rows and {len(plots)} plot files to {args.output}")
    print("No best method selected; missing experiments remain explicitly not measured.")


if __name__ == "__main__":
    main()

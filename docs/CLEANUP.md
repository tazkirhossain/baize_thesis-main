# Current-pipeline cleanup

## Data artifact cleanup, 26 October 2026

The owner requested removal of unnecessary data-folder files and generated test
cases. Disposable `data/test-*` fixtures, repeated synthetic smoke checkpoints,
plot previews and the removed dashboard's npm cache/artifacts are removed.
Small aggregate records and loose historical logs/scripts are relocated under
`results/maintenance/data-cleanup-20261026-*/`; initialization/training diagnostics
move to `results/diagnostics/`. The manifest lists each removed entry, logical
bytes, tracked-file counts, and protected-file SHA-256 values verified afterward.
Original datasets, all client-owned train/validation/test data, pretrained
DeBERTa assets, required runtime packages, enrichment caches and original recovery
archive remain. Source regression tests in `tests/` remain required for pipeline
verification and are not run during ordinary training. Launcher temporary files
and transport staging are retained because active training may use them.
The older cleanup below remains historical context.

Authorized by the user on 23 October 2026: remove the previous person's work
outside the current research pipeline. This supersedes earlier instructions to
preserve the old simulator in the active tree.

## Retained

- Audited preprocessing, strict label/schema validation, safe canonical URLs,
  exclusion reporting and reproducibility snapshots.
- Five-client IID, separate label-skew/category-skew and optional Dirichlet configuration.
- Shared domain-role leakage prevention and client-local nominal 70/15/15 splitting.
- Training-only balancing, SMOTE numerical utility and train-only feature transforms.
- Character CNN, DeBERTa-v3-base, Flower clients/FedAvg and privacy-aware evaluation.
- Metric/report exports, current pipeline tests, safety CI and source license attribution.
- All existing raw/cleaned/client data, local runs and original source snapshots.

## Removed from the active project

- `baize/`: old linear models, simulator, attack/noise experiments, checkpoint
  selection, inference API and demo server. Required helpers were migrated first.
- `web/`, `scripts/serve.py` and the earlier `scripts/smoke.py` demo.
- `experiments/`, old phase/E1/E2/FedAvg scripts and their old research results.
- Old research decisions, threat/labeling/privacy specifications, `RESULTS.md`
  and unrelated push instructions.
- Discord notification script and workflow; repository safety CI stays.
- Tests solely for those removed components. URL privacy tests were retained
  separately, and all current preparation/model/FL/evaluation tests remain.
- Intermediate `results/evaluation_reporting_v1/`, v2 and v3 report drafts.
  Final v4 and the current study's other aggregate records remain.
- Unused pandas/PyYAML dependencies, TF-IDF branch and old JSONL Git-ignore exceptions.

## Migration

| Previous location | Current location |
|---|---|
| `baize/data/preprocessing.py` | `src/data/preprocessing.py` |
| `baize/data/url.py` | `src/data/url.py` |
| `baize/data/features.py` hashing/L2 branch | `src/data/features.py` |
| Binary label validator and category constants | `src/data/schema.py` |
| Protected evaluation-label array helpers | `src/data/protection.py` |
| `baize/provenance.py` | `src/provenance.py` |

Imports, entry points, packaging, source snapshots, tests and documentation use
the new locations. Do not add compatibility shims that restore the old pipeline.
The `baize-domain-role-v1` hash prefix is deliberately unchanged: it is part of
the existing split contract, and changing it would alter evaluation membership.

## Recovery and data integrity

Before cleanup, 248 source/config/document/result files were copied into the
verified local `data/cleanup-2026-10-23/before-cleanup.zip`. Each archived file was
read back and checked against its SHA-256; `manifest.json` records the hashes.
`changes.json` records removed paths. The archive is ignored and contains the
pre-cleanup tree, not an additional active pipeline. Existing snapshots inside
older runs remain historical evidence and were not edited.

Integrity metadata for 191 existing data artifacts was captured before changes.
Cleanup does not resample, redistribute, regenerate or rewrite client data.
The final verification record is saved locally as `verification.json` after
regression and command checks.

## Cleanup verification, 23 October 2026

**159 current pipeline tests passed**, with one opt-in localhost gRPC test skipped.
No legacy UI tests remain or need deselection. The real Character CNN completed
two Flower rounds across five synthetic clients; the report CLI exported four
metric rows and four PNG/SVG files from that run. These are integration checks,
not real-data neural performance results.

Verified all **80 original client dataset/split content hashes**, the raw CSV
SHA-256 and integrity metadata for 191 existing data artifacts. Both model
architectures and the shared domain-role implementation match their pre-cleanup
syntax trees; LICENSE is byte-identical. All 69 active Python files parse, direct
project imports resolve without the old package, and active documentation links
resolve. The final historical report hash is unchanged.

Evidence: `data/cleanup-2026-10-23/verification.json`, `verify_cleanup.py`,
`tests.log`, `smoke.log` and `report.log`. The verified recovery archive remains
ignored and local. README describes the current commands and model limitations.

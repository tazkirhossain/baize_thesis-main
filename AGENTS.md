# Project instructions

## Scope and structure

The user authorized removal of the earlier person's pipeline on 23 October 2026.
The active project is the five-client Character CNN / DeBERTa-v3-base Flower
pipeline. Do not restore the old linear models, simulator, demo website, Discord
notifications or old experimental claims. Read `docs/PROJECT_STRUCTURE.md`,
`docs/CLEANUP.md` and `docs/version2_readme.md` before structural changes.

Use `config/`, ignored `data/`, `src/data/`, named `src/models/` folders,
`src/client/`, `src/server/`, `src/evaluation/`, `scripts/`, `results/` and `tests/`.
Shared helpers now live in `src/data/` and `src/provenance.py`; there is no active
`baize` package. All current imports, packaging and snapshots must use `src`.
Preserve LICENSE attribution for adapted code.

## Data and leakage

Read `docs/PREPROCESSING.md`, `IID_PARTITIONING.md`, `NON_IID_PARTITIONING.md` and
`PARTITION_SAMPLING.md`. Preserve the CSV and audited records, IDs, categories,
exclusion reasons and checksums. Do not silently change totals. Reuse the existing
verified source and three scenario roots under `data/`; never overwrite them.

Exactly five clients, storage 0–4 / human 1–5. Independently shuffle Adult and
Non-adult pools with fixed seeds and assign quotas sequentially without
replacement. Verify all eligible records occur exactly once. Keep IID, label
skew and category skew independent. Domains may span clients but must retain one
shared train/validation/test role everywhere. Preserve the existing
`baize-domain-role-v1` hash namespace as a data compatibility constant; changing
it would change evaluation membership. Clients split locally, nominal 70/15/15.

## Models and balancing

Read `docs/NEURAL_MODELS.md` and `CLASS_BALANCING.md`. Preserve the user-provided
CNN architecture and the pinned Microsoft DeBERTa checkpoint/tokenizer. Offline
DeBERTa loading must fail if pretrained files are missing, never silently use a
random encoder. Shared labels are 0=Non-adult, 1=Adult.

Use train-only undersampling, oversampling or independent undersampling ensembles
with either model. Read `docs/NEURAL_BALANCING.md` for continuous-embedding SMOTE
and chunked transport. SMOTE requires explicit `representation` configuration;
all four methods support this matched two-stage warmup/head protocol. Default Adult:Non-adult ratio is 1:1; 1:5 is an explicit alternative.
Log before/after counts, seeds, duplicates and member counts. No class-weighted
or focal loss replacement. Keep engineered-numerical SMOTE/scaling utilities separate from encoder-embedding
SMOTE; there is no classical numerical classifier runner. Never SMOTE URLs or tokens.
Multi-DeBERTa ensembles use bounded, verified chunked parameter RPCs. Chunking
does not solve RAM requirements; use capable hardware. Learned transforms fit only
original local training data and transform holdouts with frozen statistics.

## Federation and reporting

Read `docs/FEDERATED_TRAINING.md` and `EVALUATION_REPORTING.md`. Use Flower 1.22.0
with `scripts/federated.py`. The server must never load client datasets. Send
compatible model tensors and agreed coarse statistics only. Primary FedAvg
weights are ORIGINAL local training counts; declare resampled weighting
separately. Preserve seeded sampling, model/schema/config contracts and fail-closed
rounds. All five clients validate the updated global model. Original-training
metrics are a read-only evaluation pass distinct from balanced optimization loss.
Test evaluation is optional and once at the fixed final round.
The user approved training-only checkpoint selection on 23 October 2026.
Read docs/CHECKPOINTS.md. When model_selection is enabled, reserve a fixed
domain-disjoint training subset, protect its labels, exclude it from optimization,
balancing and FedAvg original-training counts, and choose by pooled selection
Adult F1 only (earliest round wins ties). Validation/test never select the model.
Optional test runs once after the FL budget on the selected checkpoint, with
its actual checkpoint round/hash recorded separately from final-round weights.
Save every global round and retain final weights separately. Do not claim that
weight checkpoints implement full session resume or a cross-experiment winner.
Keep timed batch progress separate from global evaluation. The optional
progress_interval_seconds defaults to 30 (0 disables batch logs); report only
after completed batches. ETA is phase-local. Console round summaries must use
the measured report, without adding evaluation passes or parameter selection.

Report both classes' precision/recall/F1, macro-F1, balanced accuracy, confusion
counts, accuracy, timing and exact serialized application payload sizes. Sum
confusion counts before computing global metrics. Missing historical measurements
are null. Never tune sampling, weights, thresholds or model parameters from
holdouts, use accuracy alone or claim a best method without measurements.
Separate model families, real/synthetic data, local/federated scope and compatible
experiment settings. Full real-data neural studies and LLM review remain pending.
Read docs/WEBSITE_ENRICHMENT.md for optional client-owned website inputs. Keep
collection label-independent, bounded, robots-aware and separate from offline
cleaning and training. Never rewrite original splits, drop failed pages, fetch
private addresses or crawl from the server. Enriched training requires frozen,
verified local snapshots. Preserve input/cohort digests for fair comparisons.

## Reproduction and safety

Run `python -m pytest tests -q` from the repository root. The real gRPC test is
opt-in. Existing snapshots, original data and past aggregate reports are immutable;
keep them readable as history, not active model implementations. A verified local
recovery archive is in `data/cleanup-2026-10-23/`.

Publication policy changed explicitly by the user on 23 October 2026: publish
the whole data folder and remove ignore patterns. This supersedes earlier
dataset/artifact exclusions. Actual credentials/private keys remain prohibited;
the updated repository safety check and CI enforce that. Data, exports, model
states, caches, dependencies and historical artifacts are authorized for this
publication. Use Git LFS for large files and preserve nested test-repo .git
metadata in the documented archive, never as submodules. Do not publish the
root repository's .git metadata. No notification workflow is authorized.
The user requested publishing on
codex/federated-url-pipeline, keeping main untouched. The authorized origin is
https://github.com/tazkirhossain/baize_thesis-main.git. The destination was empty
when inspected before the first push. Preserve any remote history on later pushes.

## Active experiment configuration

Routine neural selection now uses `config/experiment.json` with
`scripts/run_experiment.py`; see `docs/RUN_EXPERIMENT.md`. Preserve the old full
presets as model defaults and low-level interfaces. The launcher selects audited
roots, checks strategy/own-client metadata, captures requested/effective settings,
and invokes the existing Flower/reporting scripts. Never silently enable SMOTE
on token IDs or change end-to-end/embedding budgets. Server mode must not inspect
client directories. Keep auto outputs as fresh immediate children of data/.
Check-only mode must not train/write runs. The React dashboard and its dedicated
adapter/server were removed at the user's request; standalone Matplotlib PNG/SVG
reporting remains supported.

# Completed work and current project status

Updated 23 October 2026. This is the current handover record. The complete earlier
version is recoverable in `data/cleanup-2026-10-23/before-cleanup.zip`; earlier
person's research claims and linear-model pipeline are no longer active.

## Goal and implemented workflow

Classify URL links as Adult or Non-adult using five federated clients with
Character CNN or DeBERTa-v3-base. Stages are configurable and separate: offline
audit/cleaning, unique client ownership, local splitting, local encoding/scaling,
training-only balancing, Flower aggregation, then local evaluation/reporting.

Implemented preprocessing validates columns, IDs, categories and URLs; strips
credentials/query/fragment/port; canonicalizes URLs; preserves source IDs and
categories; deduplicates same-label URLs and excludes entire conflicting URLs.
Every excluded row has a reason. Original data is never silently rewritten.
Exact sources/configs, code/runtime snapshots and audit fingerprints are saved.

## Audited data

| Group | Records |
|---|---:|
| Original CSV | 1,562,978 |
| Adult retained | 34,811 |
| Non-adult retained | 1,433,432 |
| Total retained | 1,468,243 |
| Invalid/missing excluded | 1,047 |
| Same-label duplicates excluded | 89,385 |
| Conflicting-label records excluded | 4,303 |

Conflicts cover 148 canonical URLs. The raw CSV is `data/raw/url.csv`; audited
source is `data/client-datasets-v3/`. Raw SHA-256:
`79e32aeafc74a5c875add374d1ea5d163f867ee9109a47907c99271bb76002a4`.
Full category counts are in `DATASET.md`. Audit fields are not predictors.

## Five-client partitioning

Storage clients 0–4 are human Clients 1–5. Separate seeded Adult/Non-adult pools
are shuffled and sequentially consumed without replacement. IID preserves
proportional binary/category quotas; label skew and category skew are independent
plans. Every original eligible record is owned once; no cross-client duplicates.
All source partitions remain unbalanced: balancing occurs only during training.

| Scenario | Client | Adult | Non-adult | Total |
|---|---:|---:|---:|---:|
| IID | 1 | 6,962 | 286,686 | 293,648 |
| IID | 2 | 6,962 | 286,686 | 293,648 |
| IID | 3 | 6,962 | 286,687 | 293,649 |
| IID | 4 | 6,963 | 286,686 | 293,649 |
| IID | 5 | 6,962 | 286,687 | 293,649 |
| Label skew | 1 | 12,000 | 281,649 | 293,649 |
| Label skew | 2 | 8,500 | 285,149 | 293,649 |
| Label skew | 3 | 6,500 | 287,149 | 293,649 |
| Label skew | 4 | 4,500 | 289,148 | 293,648 |
| Label skew | 5 | 3,311 | 290,337 | 293,648 |
| Category skew | 1 | 6,963 | 286,686 | 293,649 |
| Category skew | 2 | 6,962 | 286,687 | 293,649 |
| Category skew | 3 | 6,962 | 286,687 | 293,649 |
| Category skew | 4 | 6,962 | 286,686 | 293,648 |
| Category skew | 5 | 6,962 | 286,686 | 293,648 |

Existing roots: `data/client-iid-v1/`, `data/client-label-skew-v1/`,
`data/client-category-skew-v1/`. Their immutable allocation reports and local
split manifests are authoritative. Optional Dirichlet remains a separate preset.

## Client-local splits and leakage

Each client has `splits/train.jsonl`, `val.jsonl`, `test.jsonl`. Nominal fractions
are 70/15/15. Domains may span clients but use one deterministic global split role
to prevent training/evaluation leakage. Domain indivisibility makes row fractions
approximate; never force exact fractions by breaking the role rule.

| Scenario | Client | Training | Validation | Test |
|---|---:|---:|---:|---:|
| IID | 1 | 211,166 | 38,987 | 43,495 |
| IID | 2 | 211,249 | 39,109 | 43,290 |
| IID | 3 | 211,377 | 38,992 | 43,280 |
| IID | 4 | 211,989 | 38,460 | 43,200 |
| IID | 5 | 210,928 | 39,092 | 43,629 |
| Label skew | 1 | 211,230 | 39,032 | 43,387 |
| Label skew | 2 | 211,478 | 38,919 | 43,252 |
| Label skew | 3 | 211,344 | 38,922 | 43,383 |
| Label skew | 4 | 211,502 | 38,521 | 43,625 |
| Label skew | 5 | 211,155 | 39,246 | 43,247 |
| Category skew | 1 | 212,733 | 35,792 | 45,124 |
| Category skew | 2 | 212,646 | 39,027 | 41,976 |
| Category skew | 3 | 209,079 | 41,838 | 42,732 |
| Category skew | 4 | 213,960 | 36,165 | 43,523 |
| Category skew | 5 | 208,291 | 41,818 | 43,539 |

Each scenario globally has 1,056,709 training records (25,137 Adult), 194,640
validation and 216,894 test records. Role membership is shared across scenarios.
The earlier audit verified 864,157 domains with no cross-role leakage.

## Encoding and balancing

CNN uses fixed public character features. DeBERTa uses its pinned pretrained
tokenizer. These do not fit a vocabulary on holdouts. The separate hashed/L2
inspection helper remains stateless; it is not the neural input path.
Numerical scaling fits only original local training and uses frozen holdout
transforms. It remains for the SMOTE preprocessing utility.

Balancing presets default to Adult:Non-adult 1:1. Undersampling retains Adult
examples and samples Non-adult; oversampling retains originals and duplicates
Adult selections; each CNN ensemble member trains independently on a different
undersampled view and predictions average probabilities. Validation/test stay
unchanged. No class-weighted/focal substitution. Fixed seeds and before/after
counts, duplicates and member counts are logged.

SMOTE now also supports continuous neural encoder embeddings in the explicit
two-stage protocol described in `NEURAL_BALANCING.md`; never apply it to raw URLs
or token/character IDs.
There is no numerical classifier runner after the user's removal request.
The four-method historical numerical runs remain auditable under ignored data:
`data/balancing-four-methods-v1/` (1:5) and `data/balancing-four-methods-equal-v2/`
(1:1). The latter had 60 method runs / 120 independent numerical fits and 74
recorded convergence warnings. Those results are historical, not evidence of
neural SMOTE or current CNN/DeBERTa performance.

## Models and Flower

- `src/models/character_cnn/model.py`: supplied eight-convolution residual CNN,
  adaptive pooling and classifier. Transpose precedes channel Dropout1d.
- `src/models/deberta_v3_base/model.py`: Microsoft's pretrained DeBERTa-v3-base,
  fixed tokenizer and binary head, revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`.
  Full pretrained files are now installed and hash-verified locally. Offline
  loading and a full encoder/head training step passed on a synthetic example.
  Missing files still fail instead of substituting a random encoder.
- `src/client/flower_client.py`: owns local encoding/training/balancing/evaluation.
- `src/server/flower_strategy.py`: five-client Flower 1.22.0 FedAvg with deterministic
  participation, compatible state/contracts and strict round validation.

Rounds, sampled clients, local epochs, batch size, learning rate, model and
balancing are configurable. Primary aggregation weights use original local
training counts; resampled weighting is an explicit separate experiment. No
raw URLs/datasets are sent to the server. All five clients evaluate original
training and validation after each round; optional test runs once at the fixed
final round. No holdout-driven hyperparameter/threshold/checkpoint selection.
The local harness is a simulation, not separate device/storage isolation.
Independent ensembles are supported for both models with chunked parameter RPCs.

## Evaluation and measured scope

Shared metrics include both classes' precision, recall and F1, macro-F1, balanced
accuracy, confusion matrix, overall accuracy and class supports/prevalence.
Global metrics derive from summed confusion counts. Per-round train/validation
metrics, optimization loss, client/round times and exact application protobuf
payload sizes are separate. These byte counts exclude physical gRPC/TLS overhead.
No differential privacy or secure aggregation is claimed.

`scripts/report_evaluation.py` exports reproducible JSON, Markdown, PNG and SVG
with source/config/runtime hashes and compatibility checks. Supported comparisons
cover IID versus both non-IID types, balancing by scenario, Adult recall/F1,
convergence and costs. Missing runs remain unmeasured and no winner is selected.
`results/evaluation_reporting_v4/` is the immutable historical aggregate export:
24 real numerical evaluation rows plus five synthetic CNN rows. Real-data neural
experiments have not run. Actual CNN federated smoke and tiny-encoder DeBERTa
integration checks establish wiring, not full-model classification quality.

## Cleanup and current files

The old `baize/` package, linear simulator, website, phase/E1/E2 experiment scripts,
old research claims and Discord workflow were removed from the active tree.
Required preprocessing, URL, protection, schema and provenance helpers now live
under `src/`; imports, packaging and snapshots follow those locations.
README/instructions reflect this current pipeline. Old JSONL version-control
exceptions and unused dependencies were removed. `LICENSE` attribution remains.

The verified pre-cleanup archive and hashes are local at
`data/cleanup-2026-10-23/`; see `CLEANUP.md`. Raw data, partitions, local runs,
source snapshots and final current-study reports were preserved. Preparation,
local splitting, federation and report commands are in README and focused docs.

## Pending work

Run declared real-data neural
experiments, repeated seeds and final evaluation before choosing a method.
Implement LLM quality/noise/label review and sitemap/robots/metadata enrichment
as separate audited stages. Large ensembles require suitable memory and compute.
No publication, remote deployment or dataset upload has occurred.

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

## Full DeBERTa checkpoint prepared, 23 October 2026

Downloaded the four files from Microsoft's public `microsoft/deberta-v3-base`
checkpoint at immutable revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3` into
`data/pretrained/deberta-v3-base/`. `scripts/prepare_deberta.py` reproduces the
public download without credentials or client-data transfer. SHA-256/LFS hashes,
Git blob IDs and file sizes are checked against the pinned Hub metadata and saved
in ignored `download_manifest.json`. The weight file is 371,146,213 bytes with
SHA-256 `691d48a2800b926a19e3051def466fc2cca4f59a15e42ce4a0cf7f1b380b5e33`.

The actual 184,423,682-parameter pretrained model passed **offline loading,
production client tokenization, inference, backward and one plain-SGD step**.
Both an encoder parameter and the new binary head changed; all parameters were
trainable. The test used one synthetic example, max length 16, CPU, seed 42 and
the configured learning rate 0.00002. The normal max length 256 and all experiment
settings remain unchanged. No corpus/validation/test records were used, and the
original pretrained files were not modified or replaced with the smoke-trained
weights. This establishes full-checkpoint training compatibility, not detection
quality or full five-client training performance.

Evidence: `data/pretrained/deberta-pretrained-verification.json`, its `.log`,
`verify_deberta_pretrained.py` and `deberta-focused-tests.log` (22 tests passed).
The new pooler and Adult/Non-adult classifier remain untrained for the real task.
Full real-data CNN/DeBERTa experiments and fair comparisons remain pending.

This local machine has approximately 8 GB RAM and no CUDA. Single-model checking
passed, but the five-client local simulation retains a server model, five client
models and aggregation buffers simultaneously. Full DeBERTa experiments need
sufficient additional memory and compute, preferably a suitable GPU machine or
separate client devices. Do not infer that this small check established the
resource requirements of the full run.

For a fair model comparison, reuse the same verified client ownership and split
membership, balancing policy, fixed evaluation threshold and declared seeds.
Record each model's encoding, optimizer/training budget and costs. Assess Adult
recall/F1 and macro-F1 alongside the other required metrics using a declared
validation/final-test protocol; neither model has been selected as best.

## Both-model SMOTE and undersampling ensembles, 23 October 2026

All four methods now support both Character CNN and DeBERTa through the explicit
`encoder_embeddings` protocol. Eight complete presets are in
`config/federated/embedding_balancing/`; each accepts IID, label-skew or
category-skew inputs, giving 24 available combinations. This does not mean
those real-data experiments have been run or that a winning method is known.

Each round/client/member first warms up on original local TRAINING records,
extracts continuous training features with the encoder in evaluation mode, then
balances those features and trains the original classifier head with the encoder
frozen. CNN uses its original pooled 512 features and fc head; DeBERTa uses its
768 encoder features and original pooler/classifier. No raw URL, integer token
or character IDs are interpolated. Numerical-feature SMOTE remains separate.
Use the same two-stage protocol for all four methods when comparing methods;
reports reject mixed end-to-end/two-stage comparisons. Original end-to-end
presets remain available for non-SMOTE methods.

SMOTE performs standard seeded nearest-neighbor interpolation, adjusts/skips
unsupported k from training counts alone, logs actual counts/status and generates
vectors lazily from a temporary disk feature cache. Protected holdouts are
rejected BEFORE warmup or cache creation. No evaluation files are balanced or
rewritten. Per-member local audits and global synthetic/skip/adjustment counters
are retained. Warmup and balanced-head processed counts/loss budgets are explicit.

Independent ensembles now work for both models. Corresponding member states are
federated separately within the state schema and evaluation averages member
probabilities. At least two members are required. `src/transport.py` chunks large
model updates into ordered, contract/size/SHA-256-verified RPC transfers. Default
chunk size is 4 MiB and transfer limit is 16 GiB; wire counters include staging/
fetch application payloads. This handles messages larger than a single gRPC
payload without changing dtype or averaging members together. RAM requirements
for full model copies/gradients/aggregation remain; full pretrained DeBERTa
ensembles were not run on this machine.

Verification:

- 174 regression tests passed; three opt-in network cases were skipped in that run.
- All three actual localhost gRPC tests passed separately, including tiny real
  DeBERTa SMOTE and two-member ensemble with an 8-KiB RPC limit.
- Final transport/aggregation contract checks: 16 tests passed after the last
  compatibility-accounting adjustment.
- Actual user CNN: two rounds, five clients, embedding SMOTE, 20 synthetic vectors
  and 140 warmup/head processed examples each round; original global train N=60
  and validation N=20. Repeated smoke runs produced identical final state hashes.
- Actual full pretrained DeBERTa: one model, six synthetic training examples,
  training-only warmup, 768 continuous features, two SMOTE vectors and head update.
  Encoder/head changed in memory; the original checkpoint files remain unchanged.
- All 191 protected data artifacts' captured integrity metadata is unchanged;
  all four pretrained-file SHA-256 values and the historical CNN tensor layout
  were independently verified. No private corpus training or quality claim.

Main guide: `NEURAL_BALANCING.md`. Private evidence:
`data/neural-balancing-support-verification.json`, `verify_neural_balancing_support.py`,
`data/test-all-neural-balancing-01.log`, `test-neural-wire-01.log`,
`test-neural-final-contracts-01.log`, `data/cnn-embedding-smote-v2/` and
`data/pretrained/deberta-embedding-smote-verification-v1.json`.

## First IID / oversampling / Character CNN preset, 23 October 2026

Added `config/federated/iid_oversampling_character_cnn.json`: exactly five clients,
all five participating, two FL rounds, one local epoch, batch size 8, learning
rate 0.001, seed 42 and training-only Adult:Non-adult random oversampling at 1:1.
This is full end-to-end CNN training, with no additional encoder warmup. Primary
FedAvg weights use original training counts; final test evaluation is disabled.
The real-data command selects the existing `data/client-iid-v1` root and does
not repartition or change its files. Instructions: `IID_OVERSAMPLING_CNN.md`.

The synthetic smoke passed in 19.52 seconds: two actual Flower rounds, five
clients, no fit/evaluation failures, original training N=60, resampled training
N=80 (40 Adult / 40 Non-adult) and 20 duplicates in each round. All five clients
train for one epoch per round; original training and validation metrics are
evaluated after aggregation. Evidence:
`data/iid-oversampling-cnn-smoke-20261023-v1/` and its sibling `.log`.
Smoke uses length 32 and synthetic fixtures with unequal label proportions, so
metadata correctly records synthetic/unknown rather than real IID. Its scores
are not URL classification quality measurements. The full real IID run uses
length 256 and 2,063,144 balanced training examples per round; it has not been
launched by this smoke check. All artifacts remain under ignored local `data/`.

## Real IID readiness and Matplotlib figures, 23 October 2026

Read-only preflight checked all 1,468,243 records across the existing five IID
clients: original/client/split hashes, recorded row and class counts, unique
source-row ownership and train/validation/test disjointness. Each client's
training-only oversampling retained all original training rows and achieved 1:1.
One actual real-URL training batch per client at length 256 passed forward,
finite-loss/gradient, backward and SGD weight-update checks. These in-memory
updates were discarded; original partitions and checkpoints were unchanged.
No validation/test inference or full epoch was performed. Evidence:
`data/iid-cnn-real-readiness-20261023-v1.json` and its `.log`;
reproduction script `data/check_iid_cnn_readiness_20261023.py`.

Matplotlib 3.10.3 was already installed. Extended shared neural reporting with
eight original-training/validation curves (loss, accuracy, balanced accuracy,
macro-F1, Adult precision/recall/F1 and Non-adult F1), plus fixed final-round
confusion matrices with counts and within-class percentages. Existing time and
communication plots remain. Test panels appear only for measured final tests.
No best round or method is selected. The exporter is shared across both models
and all partitioning scenarios and runs separately after a completed experiment.

The user's completed run `data/cnn-over-smoke-20261023-163747/` now has a
`visualizations/` report with four metric rows and six PNG/SVG files. Convergence
and confusion figures were visually inspected, all six hashes verified, PNGs
decoded and SVGs parsed. These are explicitly synthetic figures; final all-Adult
predictions do not establish real-data quality. Historical reports were preserved.

Validation: 24 focused training/reporting tests passed; full current suite
175 passed, 3 opt-in network tests skipped, with one joblib physical-core warning.
Logs: `data/test-iid-cnn-ready-20261023-v1.log` and
`data/test-readiness-visualizations-all-20261023-v1.log`.
Updated run instructions: `IID_OVERSAMPLING_CNN.md`.

## Removed local federated experiment dashboard, 25 October 2026

The React/TypeScript/Vite/Recharts dashboard and its dedicated adapter/server
were removed from the active project at the user's request. It was independent
of training, splitting, balancing, aggregation and evaluation, so removing it
does not change the federated pipeline. Matplotlib PNG/SVG reporting remains in
`src/evaluation/plots.py` and is generated by the experiment launcher when
`generate_plots` is true.

Implemented overview KPIs; interactive metric/round curves; IID/label/category
scenario comparison and real measurement coverage; original and balanced class
counts; a recorded five-client/FedAvg diagram with local detail; confusion and
per-class metrics; history filters/search/sorting; multi-run overlays; scoped CSV
download; and a complete configuration planner defaulting to batch 64, one epoch,
ten rounds. These draft defaults never replace actual saved run settings. SMOTE
configuration uses the explicit continuous-embedding warmup/head protocol;
token-ID SMOTE and incompatible comparison claims remain rejected.

Real/synthetic/unknown classification, optional final-test checkpoints, missing
metrics and protocol/budget/data comparability remain explicit. Test accuracy
never falls back to validation. Status is recorded, not live activity. Prediction
probabilities and example-level errors are unavailable and not fabricated.
Seven valid historical neural checks were loaded from actual files, with the
user's oversampling smoke selected initially; one incompatible oldest log is
listed in the source issues panel. Audited real corpus counts are separate from
synthetic model results. No demonstration metrics were needed.

Validation: production build passed; 181 Python tests passed, three opt-in network
tests skipped; final adapter guards passed six focused tests; three frontend
tests passed. Browser controls/CSV/config downloads, client detail, keyboard
focus and responsive rendering were checked. Downloaded CSV had 14 validation
rows; downloaded SMOTE config passed the existing training config validator with
64/1/10 settings. HTTP checks rejected raw-file paths, traversal and POST writes.
Runtime dependency audit returned zero known vulnerabilities. Browser error
console was clean. Full real model quality remains unmeasured.

Evidence: `data/test-dashboard-all-v1.log`, `test-dashboard-adapter-final.log`,
`dashboard-frontend-tests-final.log`, `dashboard-build-final-v2.log`,
`dashboard-http-verification-v1.json`, `dashboard-npm-audit.json`, and
`dashboard-full-preview-20261023.png`.

## One-file experiment launcher, 23 October 2026

Implemented the user's requested shared control point:
`config/experiment.json` plus `scripts/run_experiment.py`.
`src/server/experiment_config.py` resolves the model defaults and selects the
existing IID, label-skew or category-skew root. Common controls cover model,
method/protocol, rounds, local epochs, batch, learning rate, sequence length,
seed, ratio, ensemble/SMOTE parameters, weighting, evaluation, device/threads
and automatic Matplotlib export. The shipped active combination remains
IID -> oversampling -> Character CNN, rounds 2, local epoch 1, batch 8. A null
learning rate selects the existing model default: CNN .001 / DeBERTa .00002.
The old full presets remain supported and hold architecture/checkpoint defaults.

The launcher checks metadata/strategy and own-client ownership before training;
the original loader still verifies actual records/checksums/leakage. It does not
repartition, change split files or change ML algorithms. SMOTE explicitly requires
encoder_embeddings and its additional warmup budget is printed; there is no
automatic switch from end-to-end training. Overrides of the data root require an
explicit declared evaluation population identity. Unknown/duplicate/invalid JSON
fields are rejected. CUDA/MPS availability is checked for real client training.

`--check` validates without training or writing run artifacts. Normal mode runs
all five local clients; `--mode smoke` retains the existing explicit synthetic
overrides; server/client modes preserve the existing TLS and local-data boundary.
Server-only preflight never opens client folders. Auto outputs are unique fresh
immediate `data/` children, discoverable by the dashboard. Requested and effective
settings are saved separately with captured launcher/resolver/model-default
sources. Failure/interruption provenance is preserved where output exists; failed
training does not run plotting. Post-training report failure is reported separately.

Successful runs automatically produce `report/` JSON/Markdown metrics and PNG/SVG
figures when generate_plots=true. Reporting uses the declared Adult:Non-adult
ratio, including explicit 1:5 experiments, without score-based tuning. No winner
is selected. Raw datasets, private runtime artifacts and checkpoints stay ignored.

Validation: 266 project tests passed, three optional network tests skipped.
The one-file tests cover both models, three distributions, five methods and both
protocols (invalid token/SMOTE combinations are rejected), schema failures,
strategy checks, server/client boundaries, output safety, no-work check mode,
automatic reporting and success/failure/interruption provenance. Metadata/file
preflight passed all six real model/distribution choices. An actual five-client
two-round CNN oversampling smoke completed with 80 processed training examples
per round, 1:1 views and automatic four-row/six-plot report. This is synthetic
compatibility evidence; full real-data quality was not measured by this work.

Evidence: `data/test-unified-config-all-v1.log`,
`data/unified-config-real-preflight-v1.json`,
`data/unified-config-smoke-20261023-v1/` and its sibling `.log`.
Read `docs/RUN_EXPERIMENT.md` for exact Windows setup, shared-machine instructions,
config values and advanced server/client commands.

## Periodic progress and complete round summaries (23 October 2026)

Added `src/client/progress.py` and `src/evaluation/console.py`. Both neural
models now report client-local batch progress during end-to-end training,
embedding warmup/extraction/head training and original-split evaluation. The
optional `progress_interval_seconds` is 30 by default and set to 30 in
`config/experiment.json`; zero disables batch progress. Older configs without
this field remain valid. No other active training settings were changed.

Progress prints the phase start, first completed batch, periodic updates and
completion, with client/round/member, epoch, batches/examples, running loss when
defined, elapsed time, examples/second and approximate phase ETA. Logging happens
after completed batches; it cannot report mid-batch or continuously during SMOTE
neighbor computation. It reports no raw URLs or examples and adds no model/RNG
operations or evaluation passes.

After every completed global validation round, the console prints the original
validation metrics and, when enabled, original-training metrics: both classes'
precision/recall/F1, macro-F1, balanced accuracy, accuracy, loss and confusion
matrix, plus round time and serialized application payload size. Metrics use
summed client confusion counts. Test remains optional once at the fixed final
round. Balanced-view running loss and original-population global evaluation are
explicitly separate. No tuning or best-method selection was added.

Existing round JSON, dashboard refresh and final Matplotlib reports remain in
place. Batch progress is console-only. The user's already running real training
process was not stopped; these source/config changes apply on the next launch.
See `docs/RUN_EXPERIMENT.md` for controls and interpretation.

Verification: 277 tests passed and three optional network tests skipped in
`data/test-progress-all-v1.log`. New deterministic tests cover timer behavior,
weighted loss/ETA, disabled logging, interval validation and old-config support,
known confusion metrics, and identical model updates/evaluation metrics with
logging on versus off in both end-to-end and embedding protocols. Client data
files stayed byte-identical in those tests. The active real-data metadata
preflight passed in `data/progress-config-check-v1.log`; this is a readiness
check, not evidence of real-data detection quality.

## Every-round checkpoints and training-only best model (23 October 2026)

The user requested all model checkpoints and explicitly approved a selection
subset derived only from TRAINING, rather than selecting on validation/test.
`config/experiment.json` now enables `model_selection` with source
`training_selection`, fraction 0.1 and metric `adult_f1`. Other active parameters,
including the user's CPU thread setting, were preserved.

Clients use `src/client/selection.py` to deterministically reserve whole domains
inside their immutable original training split. The distinct shared hash and
experiment seed give the same domain role across clients/scenarios. No labels
or evaluation scores determine the role. The selection subset is protected and
excluded from encoder warmup, feature fitting, balancing and optimization.
No original dataset/split file or validation/test membership was changed.
Counts, class supports, seed and membership fingerprints are audited locally;
no URLs or IDs leave the client. Empty local selection is supported, but pooled
selection must contain both classes or best-model selection fails explicitly.

FedAvg original-training weights count original OPTIMIZATION records after
removing selection records, before balancing/duplication/synthesis. The server
checks assigned_train = original_train + selection_train. Comparison contracts
include the selection policy so these runs cannot silently mix with older runs
that optimized the complete original training split.

`src/server/checkpoints.py` provides atomic pickle-free NPZ storage and verified
reloads (keys, schema, shapes, dtypes, finiteness, contract and global checksum).
Every successfully aggregated round saves `server/checkpoints/round_NNNN.npz`
with JSON metadata, before evaluation, and marks metadata evaluated only after
valid evaluation returns. A later interruption preserves completed checkpoints;
unassessed models are not promoted to best and failed runs are not marked PASS.
All ensemble members are saved in each global checkpoint for both model families.

The server promotes `best_model.npz`/`.json` only on a strict improvement in
pooled training-selection Adult F1; equal scores keep the earlier round. Final
weights remain separate in `global_final.npz`. Optional test runs once after
the FL budget on the selected checkpoint; its actual round/hash are recorded
separately. The active final-test flag was not enabled implicitly. Console,
report metadata/plots and the read-only dashboard describe the new policy;
export readers recompute the decision from selection confusion counts and
reject a mismatched tested checkpoint. No validation/test-based selection or
automatic cross-experiment winner ranking was added.

The enlarged synthetic smoke fixture is explicitly conditional on selection
being enabled, so tiny minority supports are adequate for exercising the split.
Legacy full configs remain compatible, save every round and retain fixed-final
selection when the new option is absent/null. Checkpoints are loadable GLOBAL
WEIGHTS; automatic session resume and mid-round recovery are not implemented.
Read `docs/CHECKPOINTS.md` for files, semantics, reload example and limitations.

Verification: 298 tests passed, three optional network tests skipped in
`data/test-checkpoints-all-v1.log`. New tests cover label-independent shared
domain membership, complete/disjoint protected selection, invalid policies,
all four balancing methods, end-to-end/embedding protocols, checkpoint reload
and corrupt-state rejection, earliest-round ties, independence from val/test
scores, interruption before evaluation, selected-model final-test routing,
actual tiny DeBERTa prediction parity after reload, and report/dashboard
rejection of a mislabeled tested checkpoint. No full pretrained DeBERTa quality
claim is made from a tiny-encoder compatibility test.

An actual Character CNN completed the one-file launcher's two-round five-client
oversampling smoke at `data/checkpoint-selection-smoke-v1/`. From 200 assigned
synthetic training records it retained 172 for optimization and 28 for selection;
each round trained balanced views of 121 Adult and 121 Non-adult examples.
Both round checkpoints and best/final files passed schema/contract/checksum
reloads. Equal selection F1 retained round 1. Original files were rehashed and
unchanged. Automatic reporting produced four train/validation metric rows and
six PNG/SVG plot files; the updated confusion plot was visually inspected.
Test was not loaded, matching the active false setting. This is synthetic
execution/integrity evidence, not real-corpus model quality. Audit:
`data/verify_checkpoint_selection_v1.py` and the run's `verification.json`.
The active real-data metadata preflight is `data/checkpoint-config-check-v1.log`.

## Complete data publication (owner instruction, 23 October 2026)

The owner explicitly superseded the earlier dataset exclusion and requested
the complete data folder with no Git ignore patterns. Root .gitignore now has
no patterns. Publication includes raw data, client partitions/splits, pretrained
weights, checkpoints, old runs, caches, runtime packages, dependencies and test
artifacts. Actual credentials/private keys remain prohibited; the publication
safety guard/CI check those and regular GitHub file limits. Exact known fake
safety-test values remain test fixtures, not actual credentials.

Large files are staged through Git LFS, and data/runtime attributes disable
newline conversion to preserve dataset/manifest checksums across operating
systems. Historical nested test-repository .git directories cannot be ordinary
Git paths, so their metadata is preserved in a ZIP and SHA-256 manifest under
data/ while ordinary files are published at their existing locations. Root
repository metadata is never uploaded. No local source data is deleted or
repartitioned by this work. Use Git/LFS clone and git lfs pull to obtain actual
large files; GitHub browser ZIPs may contain pointers. See PUBLISHED_DATA.md.

The ML pipeline was unchanged. The complete regression suite passed 300 tests
with three optional network tests skipped after the publication policy change.
The focused safety tests were also rerun to verify public CA bundles, exact
synthetic payloads, token/private-key blocking and duplicate-content handling.

## Website enrichment and neural inputs — 25 October 2026

Added `src/data/enrichment/`, `scripts/enrich_client.py`,
`config/website_collection.json` and `config/experiment_website.json`. The main
experiment file keeps `website_enrichment: null` for URL-only reproduction. The
new optional input mode works with either model, IID/label/category skew and all
four existing training-only balancing methods. No new classifier, label inference
or LLM service was substituted.

Each client collects robots rules, bounded sitemap/index/gzip context, title,
description/Open Graph/Twitter metadata, keywords/rating/language and non-script
HTML text into its own resumable SQLite snapshot. Public-address pinning, TLS,
redirect/domain checks, robots access/pacing and resource limits bound collection.
Original rows, IDs, labels and splits remain unchanged. Failed pages are retained
with missing features and explicit reasons; full coverage is required to freeze.
Complete snapshots are verified and reused without network access. Training
never crawls, and the server receives only tensors, metrics and cache hashes/counts.

Both neural collators now allocate separate URL/title/metadata/page/site budgets,
so long URLs cannot truncate every website field. CNN adds a fixed whitespace
alphabet; DeBERTa keeps its pretrained tokenizer. Sampling/SMOTE/ensembles remain
optimization-only and share the same frozen input reader. Selection, validation
and test roles remain protected. Cohort guards and plot/report labels distinguish
URL-only/enriched inputs and require pooled content fingerprints for comparisons.

The actual CNN completed two synthetic Flower rounds with five clients at
`data/website-cnn-smoke-v1/`, with round/best/final checkpoints and six verified
PNG/SVG files. The public example.org transport/HTML probe succeeded. A bounded
five-record REAL Client 1 collection pilot retained all five rows: one metadata
success, two robots-denied and two robots-unreachable. Its cache is explicitly
partial; full-corpus enrichment/training has NOT been completed or quality-proven.

Independent evidence in `data/website-enrichment-verification-v1.json` rehashes all
90 original files across all scenarios and verifies unchanged totals (1,468,243
rows; 34,811 Adult; 1,433,432 Non-adult), smoke checkpoints and report artifacts.
Read docs/WEBSITE_ENRICHMENT.md for the exact pilot/resume/five-client/training
commands and limits. Regression evidence: data/website-all-tests-v3.log.

Regression verification: 357 tests passed, three optional network tests skipped.
Two further offline-reuse checks passed. `scripts/reuse_enrichment.py` lets the
dataset owner provision a new scenario's assigned client features from the five
previously frozen source snapshots, without new crawling or labels. It is not a
server/runtime client operation. Original URL/source-row/split matches are
required, copied rows belong only to the target client, partial copies resume,
and the pooled feature fingerprint stays identical across client reallocations.
This supports fair IID/non-IID comparisons using exactly the same website inputs.

## Adult false-alarm investigation — 26 October 2026

The user reported weak Adult precision during a real IID undersampling CNN run.
The read-only audit verified correct labels, retained minority examples, majority
sampling without replacement, unchanged original files and correct fixed character
encoding. Client 1's ORIGINAL OPTIMIZATION pool (selection excluded) contains
4,564 Adult and 181,920 Non-adult records. No Adult URL is truncated at length 256
or contains an unknown encoded character; only two Non-adult URLs truncate.
There are no duplicate/conflicting 256-character prefixes in that inspected pool.
This is training-only evidence, not a raw validation/test audit. Detailed counts
and limitations are in [ADULT_PERFORMANCE_DIAGNOSIS.md](ADULT_PERFORMANCE_DIAGNOSIS.md).

Added versioned `stable_classifier_v1` CNN initialization: only the final
1024-to-2 layer uses normal weights with standard deviation 0.01 and zero bias.
The hidden layers and supplied architecture remain intact. Missing settings keep
the legacy `fan_out` initialization for historical reproduction. Synthetic startup
behavior does not prove improved final detection quality.

The active experiment enables optional `threshold_selection` from the existing
protected training-selection domains: 201 class-conditional bins, Adult F1, and
higher-threshold ties. Clients return only 2-by-201 counts; the server uses pooled
selection counts only. Fixed 0.5 reference metrics are retained. Validation/test
never choose thresholds or checkpoints. Earliest-round checkpoint ties remain;
the checkpoint records its threshold, and optional final test freezes both.
This adds no differential privacy or secure aggregation guarantee.

A fresh run is required. Existing running processes/checkpoints do not change.
Primary comparisons must use matching initialization and threshold policies;
improvement remains unmeasured until the new experiment and reserved final test
complete. The four balancing methods, original partitions/splits and FedAvg
original-optimization-count weighting remain in place.

## Data folder cleaned — 26 October 2026

Removed 111,713 disposable files (1.072 GiB) across 148 entries: generated
`data/test-*` fixtures/logs, repeated synthetic smoke checkpoints, preview graphs,
and obsolete dashboard/npm artifacts. No generated `data/test-*` directories
remain. Source `tests/` retains 24 Python files for leakage/model/FL checks;
ordinary training does not run those tests.

All 313 recorded protected raw/active-partition/pretrained file hashes match
their pre-cleanup SHA-256 values. Client train/validation/test files, raw CSV,
all audited dataset versions, required Python runtimes, enrichment caches and
the original recovery archive remain. Transport staging and launcher temporary
configs were retained for possible active jobs. Real-data launcher preflight
passes after cleanup; no new training or full test suite was run during cleanup.

Small historical loose files and compact verification evidence are relocated to
`results/maintenance/data-cleanup-20261026-015555/`. CNN diagnostic scripts and
measurements live in `results/diagnostics/`. The cleanup manifest records each
removed path, size, tracked-file count, and protected file checksum. This cleanup
does not rewrite Git history or push changes to GitHub.

## CNN learning investigation and local optimizer — 26 October 2026

After stable initialization, the user reported loss near 0.693 and weak Adult
discrimination. A frozen round-1 checkpoint and controlled client-local probes
used original TRAIN only, preserving optimization/selection domain roles.
The current SGD 0.001 barely changed the encoder and failed to fit a 64-example
balanced training sample after 60 updates. AdamW at the same rate reduced recent
training loss to 0.028 on that memorization check; this is not a holdout result.

Larger matched pilots used 1,024 balanced optimization examples, three epochs
(48 updates) and separate unbalanced 4,096-row training-selection samples per
client. Selection Adult F1 increased from 0.0414 to 0.2623 on Client 1 and from
0.0330 to 0.2195 on Client 2. ROC-AUC increased from 0.5025 to 0.7366 and from
0.4778 to 0.7445 respectively. F1 uses the existing 201-bin selection-only policy.
These are single-seed, two-client local pilot results, not a full FedAvg study;
Adult recall at selected F1 thresholds remains limited.

Added a reusable local optimizer factory for both model families and all four
balancing methods, including encoder warmup and frozen-head training. Explicit
AdamW/SGD settings enter configuration, hashes, snapshots, console output and
comparison checks. Missing settings reproduce historical plain SGD exactly.
State resets per member/phase/round; server aggregation remains FedAvg.
The active file now selects AdamW at 0.001, zero weight decay. Original datasets,
balancing ratio, train/selection/validation/test memberships and epoch/round
budgets remain unchanged. Validation/test were not balanced or used for tuning.

Verification: 427 full-suite tests passed, three skipped; a later exact-update
parity regression also passed. Actual Character CNN five-client/two-round AdamW
integration exercised checkpoint selection and frozen-threshold final testing.
Source tests and compact evidence/logs remain; generated test fixtures were
removed afterward. `scripts/diagnose_cnn_learning.py` reproduces bounded probes;
CSV/JSON evidence and source fingerprints are under `results/diagnostics/`.
Read [CNN_LEARNING_DIAGNOSIS.md](CNN_LEARNING_DIAGNOSIS.md). A fresh full run is
required to establish actual FL/final-test improvements. No best-method or
guaranteed-good-result claim is made. These changes are local until committed
and pushed.

# Federated Adult and Non-adult URL Detection

A reproducible five-client research pipeline using **Character CNN** and
**DeBERTa-v3-base**, Flower FedAvg, client-local training and evaluation.

The earlier linear-model simulator, demo website, research claims and unrelated
scripts were removed. Required URL cleaning and audit helpers now live in `src/`.
See [cleanup details](docs/CLEANUP.md) and the
[completed-work record](docs/version2_readme.md).

## Workflow

1. Validate the local CSV, minimize URLs, map labels, deduplicate and audit every exclusion.
2. Assign every eligible record to exactly one of five clients using IID, label skew or category skew.
3. Each client splits its own records into nominal 70% train, 15% validation and 15% test.
4. Encode locally and balance training views only; keep original files and holdouts unchanged.
5. Run Flower FedAvg and aggregate local evaluation counts, timing and model updates.
6. Export both-class metrics, reproducibility metadata, comparison tables and plots.

A shared deterministic domain role prevents a domain appearing in training on
one client and evaluation on another. Domain groups make exact split percentages
approximate. Client storage IDs 0–4 are displayed as Clients 1–5.

## Setup

Use Python 3.11 or newer in a project environment, then install the extras needed:

```powershell
python -m pip install -e ".[dev,federated,deberta,reporting]"
```

The pinned DeBERTa pretrained files are installed and verified in ignored local
storage in this workspace. On another machine, run `python scripts/prepare_deberta.py`
after installing the DeBERTa extra. Loading never falls back to random encoder
weights. See [model setup](docs/NEURAL_MODELS.md).
Exact installed dependency versions and code hashes are saved with each run;
repository dependency pins do not prove a fresh environment was tested.

## Existing data

The local CSV is `data/raw/url.csv`. The audited source is
`data/client-datasets-v3/`: **1,468,243** retained records, comprising **34,811
Adult** and **1,433,432 Non-adult**. All three scenarios already have five clients
with local splits:

| Scenario | Existing partition root |
|---|---|
| IID | `data/client-iid-v1/` |
| Label skew | `data/client-label-skew-v1/` |
| Category skew | `data/client-category-skew-v1/` |

Reuse these verified files. Preparation commands require fresh output paths and
refuse to overwrite existing runs. No dataset is bundled or uploaded.

```powershell
# Only when preparing a new audited source:
python scripts/prepare_clients.py --input data/raw/url.csv --output data/prepared-new --clients 5
# Repartition an existing audited source without changing its records:
python scripts/partition_clients.py --config config/five_client_iid.json --output data/iid-new
# Run separately for each client's own directory:
python scripts/split_client.py --client-dir data/iid-new/client_000
```

See [preparation and audit rules](docs/PREPROCESSING.md),
[IID](docs/IID_PARTITIONING.md), [non-IID](docs/NON_IID_PARTITIONING.md) and
[configuration](docs/EXPERIMENT_CONFIGURATION.md).

## Training and evaluation

**Edit `config/experiment.json` and run `python scripts/run_experiment.py`.**
That one file selects the model, IID/label/category distribution, balancing,
rounds, epochs, batch, learning rate, device and automatic Matplotlib export.
The launcher selects the correct existing client root. Its current defaults are
IID -> random oversampling -> Character CNN, one local epoch, two rounds, batch 8.
Use `--check` to validate without training, or `--mode smoke` for synthetic
verification. See [single-file configuration and team setup](docs/RUN_EXPERIMENT.md).
The earlier complete presets and low-level commands below remain available.

The new [local experiment dashboard](dashboard/README.md) visualizes current
Flower results with interactive filters, round curves, client audits, comparison
history and CSV export. Start the built dashboard with
`python scripts/dashboard.py`, then open http://127.0.0.1:8765. A fresh checkout
requires `npm ci` and `npm run build` in `dashboard/` first. It is read-only and
keeps synthetic checks distinct from real experiments.

```powershell
# Tiny synthetic integration check with the actual Character CNN:
python scripts/federated.py smoke --output data/fl-smoke-new
# A configured real-data experiment; this can require substantial compute:
python scripts/federated.py local --config config/federated/character_cnn.json --partition-root data/client-iid-v1 --output data/fl-iid-cnn-new
# Export a completed run's measured metrics and plots:
python scripts/report_evaluation.py --federated-run data/fl-iid-cnn-new/server --output results/fl-iid-cnn-new
```

Choose model, rounds, client participation, local epochs, batch size, learning
rate and balancing in `config/federated/`. **All four balancing methods now work
with both models** through the explicit encoder-embedding experiment presets in
`config/federated/embedding_balancing/`. SMOTE operates only on continuous local
training embeddings after encoder warmup; it never interpolates URLs or IDs.
Existing end-to-end presets retain their original protocol. Ensembles train
independent members and average probabilities; large DeBERTa states use chunked
RPCs. See [neural balancing and limits](docs/NEURAL_BALANCING.md).

The primary FedAvg policy weights updates by **original local training counts**.
The server receives model tensors and aggregate statistics, not raw URLs or
local datasets. All five clients evaluate the updated global model on original
optimization and validation data; optional final test evaluation occurs once after
all configured rounds. The active configuration reserves 10% of training domains
for checkpoint selection by pooled Adult F1; those records are excluded from
optimization, balancing and original-training FedAvg counts. Every global round
is saved, and best/final model files remain separate. Validation/test never select
the model. See [checkpoint files and selection](docs/CHECKPOINTS.md),
[Flower design and commands](docs/FEDERATED_TRAINING.md)
and [metric definitions/reporting](docs/EVALUATION_REPORTING.md).

Reports include both classes' precision/recall/F1, macro-F1, balanced accuracy,
confusion matrices, overall accuracy, per-round metrics, training time and
serialized Flower payload sizes. No automatic winner is selected.

Full real-data neural comparisons are **not yet measured**. Existing
[historical tables and plots](results/evaluation_reporting_v4/README.md) distinguish
removed numerical-model experiments from synthetic CNN checks. They do not
establish real-data CNN/DeBERTa performance. LLM quality/noise review and
sitemap/robots/metadata enrichment remain pending.

## Verification and data safety

```powershell
python -m pytest tests -q
# Before committing from a real Git checkout:
python scripts/check_repository_safety.py
```

The real localhost Flower test is opt-in; see `docs/FEDERATED_TRAINING.md`.
The owner explicitly authorized publishing the complete data/artifact tree on
23 October 2026. Root ignore patterns were removed. Actual credentials and
private keys remain blocked by the repository safety check. Large files use
Git LFS; initialize LFS and run `git lfs pull` after cloning to obtain actual
datasets/pretrained weights rather than pointer files. Included Windows runtime
packages are historical local artifacts; install dependencies appropriate to
your own machine. See [published data guide](docs/PUBLISHED_DATA.md).
The project uses branch
`codex/federated-url-pipeline` at
[tazkirhossain/baize_thesis-main](https://github.com/tazkirhossain/baize_thesis-main),
as requested by the user; publishing does not target `main`. `LICENSE` retains the
original source-code attribution and does not license the dataset.

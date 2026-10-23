# One-file experiment configuration

Edit **`config/experiment.json`** for ordinary experiments. Run
**`python scripts/run_experiment.py`** from the repository root. The launcher
resolves model defaults, chooses the matching existing five-client dataset,
checks its declared strategy, and invokes the existing Flower runner. It never
repartitions data, changes splits or replaces the training/aggregation code.

The current active combination is IID -> random oversampling -> Character CNN,
ten FL rounds, one local epoch and batch size 64. These are the user's declared
first real-data settings, not tuned recommendations. Check the file and the
printed effective settings before each launch.

## Main controls

| Field | Values / meaning |
|---|---|
| `model_name` | `character_cnn`, `deberta_v3_base` |
| `partition_type` | `iid`, `label_skew`, `category_skew` |
| `balancing_method` | `none`, `undersampling`, `oversampling`, `smote`, `ensemble_undersampling` |
| `training_protocol` | `end_to_end` or `encoder_embeddings` |
| `rounds` | Number of FL communication rounds |
| `local_epochs` | Full-model epochs in end-to-end mode; balanced head epochs in embedding mode |
| `batch_size` | Local batch size, applied as specified |
| `learning_rate` | Positive number, or null for model default (CNN 0.001; DeBERTa 0.00002) |
| `sequence_length` | CNN character length / DeBERTa token maximum length |
| `seed` | Model, training and balancing seed; does not regenerate partitions |
| `n_clients` | Exactly 5 |
| `clients_per_round` | 1–5; validation still uses all five clients |
| `device` | `cpu`, `cuda`, `cuda:0`, etc., or supported `mps` |
| `threads` | Positive CPU thread limit |
| `adult_to_non_adult` | `1:1`, `1:5`, or another supported Adult:Non-adult ratio |
| `ensemble_members` | Member count when using ensemble undersampling (at least 2) |
| `smote_k_neighbors` | Training-only SMOTE neighbor count |
| `smote_insufficient_minority` | `adjust` or `skip` |
| `warmup_epochs` | Additional original-training warmup epochs in embedding mode only |
| `aggregation_weight` | Primary `original_training`; explicit alternative `resampled_training` |
| `threshold` | Predeclared fixed Adult decision threshold |
| `evaluate_training` | Evaluate updated global model on original training splits |
| `evaluate_final_test` | Evaluate untouched test splits once at the fixed final round |
| `generate_plots` | Automatically export Matplotlib metrics/tables/PNG/SVG after success |
| `progress_interval_seconds` | Optional; default 30 seconds between batch progress updates; 0 disables batch progress |
| `model_selection` | Optional; training-only domain selection and best Adult F1 checkpoint; see `CHECKPOINTS.md`; null uses fixed final round |
| `pretrained_source` | Local DeBERTa checkpoint directory, used only by DeBERTa |
| `partition_root` | null for automatic selection; override only for explicitly audited custom roots |
| `evaluation_population_id` | null for the known default corpus; explicit identity required with a custom root |

Do not write comments inside JSON. Unknown, missing required, duplicate or invalid fields
are rejected. The launcher prints the resolved settings, including an inherited
learning rate, before starting. It does not inspect validation/test scores to
choose any parameter. Balancing remains local to TRAINING only.

## Console progress and round metrics

The default progress interval is 30 seconds. Configure it in the active file:

```json
"progress_interval_seconds": 30
```

Each client prints its phase start, first completed batch, periodic progress and
phase completion. Lines identify client, round and ensemble member where relevant,
epoch, completed/total batches and examples, running loss when applicable,
elapsed time, examples/second and approximate phase ETA. Updates happen after
completed batches; a single slow batch or SMOTE neighbor computation can still
take longer than the interval. ETA covers the current phase only. The same
reporter supports both models, end-to-end training, embedding warmup/extraction,
balanced head training and evaluation. Embedding extraction has no loss.

After every FL round, the server prints the updated global model's full validation
metrics, and original-training metrics when `evaluate_training=true`: both classes'
precision/recall/F1, macro-F1, balanced accuracy, accuracy, loss and confusion
matrix `[[TN, FP], [FN, TP]]`. It also prints measured round time and serialized
application payload size. Rates are fractions from 0 to 1. Counts are summed
across all five clients before calculating global metrics. Periodic running
optimization loss describes the balanced training view; it is not the same
measurement as evaluating the aggregated global model on original training data.

Full holdout evaluation stays at round boundaries; timer updates add no validation
or test passes. Test metrics are printed only once after all configured rounds when
`evaluate_final_test=true`. With training-only selection enabled, this evaluates
the selected checkpoint and records its round explicitly; otherwise it evaluates
the final-round model. The current active setting is false. No threshold,
sampling or model parameter is changed from validation/test scores. Existing per-round JSON
and final Matplotlib exports remain available. The dashboard can refresh saved
round results; batch progress is console-only.

These source/config changes apply on the next launch. A training process already
running retains the code and settings it loaded; it is not interrupted or resumed
automatically to enable logging.

The active file now enables training-only selection with a 0.1 domain fraction.
Every aggregated global round is checkpointed. The best Adult F1 checkpoint and
the final-round weights are stored separately; ties keep the earlier round.
Selection records are removed from optimization/balancing and remain local.
FedAvg original-training weights exclude those selection records. Original
client files and validation/test membership are unchanged. See `CHECKPOINTS.md`
for exact files, local count audits, frozen selection roles, reload checks and
the distinction between model checkpoints and automatic session resume.

The files `config/federated/character_cnn.json` and `deberta_v3_base.json` now
serve as model-specific defaults for this entry point. They retain the public
CNN alphabet/input dropout and pinned DeBERTa revision. Common values in the
active file override the preset's rounds, epochs, batch, method, etc. Existing
advanced full presets and `scripts/federated.py` remain directly usable. Editing
an unrelated preset does not change a run launched through `config/experiment.json`.
The dashboard's existing configuration download produces an advanced full Flower
config for `scripts/federated.py`; it is not this active-file schema. Use the
single active file for the simple launcher, or the displayed advanced command
for those dashboard downloads.

## Automatic data selection

| `partition_type` | Default local root |
|---|---|
| `iid` | `data/client-iid-v1` |
| `label_skew` | `data/client-label-skew-v1` |
| `category_skew` | `data/client-category-skew-v1` |

These directories must already contain their five immutable client partitions,
local splits, manifests and preparation settings. Preflight checks their declared
strategy and IDs; the original client loader still verifies actual split hashes,
labels and source-row leakage at runtime. Preflight is not a new full data audit.
A custom root must declare the right strategy and an explicit audited evaluation
population ID; a different dataset is never silently labelled as the default
corpus. Shared population labels are declarations, not cryptographic proof.

## Commands on this Windows machine

```powershell
Set-Location 'D:\baize_thesis-main\baize_thesis-main'
$pythonExe = 'C:\Users\hossa\anaconda3\python.exe'
$env:PYTHONPATH = "$PWD\data\flower-runtime;$PWD\data\neural-runtime"

# Validate config and required local files; no model training/run artifacts:
& $pythonExe scripts/run_experiment.py --check

# Existing tiny synthetic execution check, currently Character CNN only:
& $pythonExe scripts/run_experiment.py --mode smoke

# Actual five-client FL on the automatically selected FULL real dataset:
& $pythonExe scripts/run_experiment.py
```

The existing smoke runner deliberately overrides rounds=2, batch=8, length=32
and CPU, retaining configured local epochs and balancing. Its effective metadata
is synthetic/unknown, regardless of the requested real scenario. It cannot
establish real IID/non-IID detection quality. Use the normal local command to
honor every active training value on real data. No real-data subset is silently
created. Full IID 1:1 oversampling processes 2,063,144 examples per local epoch
across clients per round, plus evaluation passes; use suitable compute.

The default output is a unique immediate `data/RUN/` directory containing model,
method, requested partition and timestamp. Override with `--output data/NEW_RUN`
if needed. Existing or nested output directories are rejected. Keeping runs
directly under `data/` makes them discoverable by the dashboard on refresh.
The real-data launcher uses the configured device; unavailable CUDA/MPS fails
early rather than silently switching devices.

Successful runs contain:

- `experiment-request.json`: the exact one-file request, including runtime controls.
- `experiment.json`: the effective public Flower configuration (including smoke overrides).
- `launcher_reproduction.json` and `launcher_snapshot/`: captured launcher,
  resolver and model-preset source fingerprints from before training.
- Existing `code_snapshot/`, runtime versions, server round metrics and final checkpoint.
- `report/`: automatic JSON/Markdown reports and Matplotlib PNG/SVG figures
  when enabled. Reporting ratio is taken from the declared config, not results.

Requested settings/provenance are also preserved for ordinary endpoint failures
or keyboard interruption when an output directory has been created. A failed
training endpoint does not launch reporting. If reporting fails after successful
training, the launcher says so and retains the measured training artifacts.
No failed or interrupted run is claimed as successful.

## SMOTE and fair comparisons

Set both:

```json
"balancing_method": "smote",
"training_protocol": "encoder_embeddings"
```

SMOTE cannot run in `end_to_end` mode. The embedding protocol adds the configured
encoder warmup and then trains the head on continuous balanced features. All five
balancing choices support embedding mode, so use that same protocol/budget for
comparisons that include SMOTE. Never interpolate URLs or integer character/token
IDs. See `NEURAL_BALANCING.md`. No method or model is declared best automatically.

## Share with another machine

Send project source plus `config/experiment.json`. The receiver edits that one
file for normal experiment choices, installs the project extras in their Python
environment (`pip install -e ".[federated,deberta,reporting]"`), and provides the
approved local client partitions. The workspace-specific `PYTHONPATH` above is
not needed in a normal installed environment. Dependency installation on a fresh
machine is not established by the current workspace tests; use the saved runtime
versions and an appropriate PyTorch build for the receiver's hardware.

Data and pretrained files are ignored and are not included merely by sharing
source. Keep private datasets/checkpoints out of GitHub. For DeBERTa provide its
four pretrained files at `pretrained_source`, or use the existing download helper
`scripts/prepare_deberta.py`; no random-encoder fallback is permitted.

For independently running devices, the same one-file launcher also supports:

```powershell
# Server machine: no client dataset access is required:
python scripts/run_experiment.py --mode server
# Each client device, once per ID 0–4, using ONLY its own local data:
python scripts/run_experiment.py --mode client --client-id 0 --client-dir data/client-iid-v1/client_000
```

For remote connections use the existing TLS flags `--address HOST:8080 --ca
CA.pem`, plus `--cert SERVER.pem --key SERVER.key` on the server. Keep private
keys outside version control. Public active/model configurations must match
between endpoints; client-specific local data directories need not match. Server
preflight never opens client directories, and client preflight checks only its
own provided folder. No raw URLs are transferred to the server.

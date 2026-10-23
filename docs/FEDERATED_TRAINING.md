# Five-client Flower neural training

Implemented 23 October 2026. This is the active research training path. The old
linear simulator and unrelated work were removed; existing client data is preserved.
It uses **Flower 1.22.0**, pinned for the tested NumPyClient/ClientProxy/Server
interfaces and Python 3.13 compatibility. The current Flower docs also describe
newer Message APIs; this project deliberately uses the pinned compatibility API.
No Ray installation is needed for the local harness.
Reference: [Flower 1.22.0 package metadata](https://pypi.org/project/flwr/1.22.0/)
and the installed version's Server, NumPyClient and FedAvg implementations.

## Components and data boundary

- `src/server/flower_strategy.py`: five-client registration, deterministic client
  sampling, FedAvg, federated validation and server checkpoints.
- `src/client/flower_client.py`: each client's own immutable split reader,
  training-only sampling, neural optimization and local holdout evaluation.
- `src/client/neural_data.py`: fixed character encoding or original pretrained
  SentencePiece tokenization; lazy local URL dataset and batch collation.
- `src/models/federated_state.py`: model construction and strict tensor-state
  transport, including ensemble member ordering, shapes, dtypes and schema hash.
- `src/evaluation/federated.py`: pooled confusion counts and loss sums.
- `src/client/progress.py`: configurable batch progress without extra inference
  passes or RNG/model changes; default 30 seconds, after completed batches.
- `src/evaluation/console.py`: full measured global metrics after each validation
  round and optional fixed final test; see `RUN_EXPERIMENT.md` for console fields.
- `scripts/federated.py`: separate server/client endpoints and local/smoke modes.

The server command has **no client-directory or dataset argument**. Only client
processes receive their own `client_000` through `client_004` paths. Human Client
1 is storage ID 0, through Client 5 at ID 4. No client can be configured with a
different client's manifest ID. The server receives model tensors, sample counts,
training losses and the agreed aggregate evaluation scalars. URL strings, token
sequences, labels, source IDs and per-example predictions are never returned.
Unknown training/evaluation response fields are rejected by the aggregation API.

The local harness uses the actual Flower server, strategy, FitIns/FitRes and
EvaluateIns/EvaluateRes interfaces with a socket-free ClientProxy transport.
It is a single-process simulation, not process isolation. Use the separate
network endpoints for independently running clients. Neither mode pools raw
records for central training.

## Round protocol

1. The server registers exactly five unique IDs and checks the experiment and
   model schema hashes. It samples `clients_per_round` IDs without replacement
   using a seed derived from the experiment seed and round number.
2. Every selected client receives the **same current global model**, its hash,
   round number and declared local settings. Stale/incompatible states fail.
3. Each client reads its own existing training split, verifies its checksum,
   creates the configured training-only sampling view, and trains locally for
   `local_epochs` using the configured batch size and learning rate. Plain SGD
   is reset each round; there is no momentum or optimizer state aggregation.
   Sampling and loader/dropout seeds are deterministic from client/round/member.
   No validation/test data chooses ratios, optimizer settings or thresholds.
4. Clients return full compatible states plus original/resampled training counts,
   original/resulting class counts, duplicates, examples processed and mean loss.
   Full-state averaging is mathematically equivalent to averaging deltas from
   the same global model. Every selected response must succeed; missing, duplicate,
   incompatible, stale and nonfinite responses fail the round. Partial-result
   aggregation is disabled.
5. The server aggregates and sends the **new global model** to all five clients,
   including clients not selected for training. Each evaluates its original local
   validation split. No centralized validation dataset exists in this path.

Each validation response contains only `tn`, `fp`, `fn`, `tp`, count and loss sum,
plus protocol identity/hashes and aggregate timing. Reporting added on 23 October
also returns prefixed original-training confusion counts/loss sums by default,
evaluated on the same updated global model without local fitting. The server sums confusion matrices in
[Non-adult, Adult] order and computes Adult precision/recall/F1, macro-F1,
balanced accuracy and overall accuracy; loss is weighted by validation count.
It does not average F1 scores or claim pooled PR-AUC from uncollected predictions.
Both classes' precision, recall and F1 are explicit. Per-round timing, original-
training/validation metrics and serialized protobuf byte counts are documented
in `EVALUATION_REPORTING.md`; these are application payloads, not total network
traffic. Additional client properties are whole-file split fingerprints and
coarse runtime metadata, never records or raw labels.
Local source-row disjointness and file hashes are checked at the clients;
the earlier preprocessing/domain-role audits remain the cross-client leakage gate.

The active experiment now selects a checkpoint using a separate fixed subset of
TRAINING domains and pooled Adult F1; equal scores retain the earlier round.
Selection records never participate in optimization, balancing or SMOTE. Every
global round is checkpointed, and best/final weights remain separate. Validation
is reported each round but never changes settings or selects the checkpoint.
Legacy presets without `model_selection` retain the fixed-final policy. Set
`evaluate_final_test=true` before a run to test once after all rounds, on the
selected model when selection is enabled, otherwise on the final model. Test
records never determine checkpoint selection. See `CHECKPOINTS.md` for counts,
membership/weighting changes, exact storage and reload checks.

## FedAvg weighting policy

Both primary presets use `aggregation_weight="original_training"`:

`global_next = sum(original_train_count_i * client_state_i) / sum(original_train_count_i)`

Counts are the original local training split sizes **before balancing**, not
the entire client partition, not the number processed across epochs, and not
the validation size. Duplicating minority records must not manufacture additional
client ownership or increase that client's weight. This policy keeps the weighting
population constant across undersampling, oversampling and the separate ensemble
comparison while balancing can still change the local optimization objective.
Training class balancing does not turn client aggregation into class-balanced
FedAvg; original-population weighting is a deliberate scientific choice.

`resampled_training` is available only as an explicit alternative experiment.
It weights by the resampled view size per member, excluding epoch repetitions
and ensemble multiplicity. Use a separate configuration/output, justify the
change, and do not mix policies within an IID/non-IID/balancing comparison.
The selected policy and each client's original/resampled counts and weight are
recorded in every round. Original counts cannot change across later rounds.

All floating model parameters and buffers, including CNN BatchNorm running means
and variances, use the same weighted mean. Integer buffers such as
`num_batches_tracked` retain the previous global value; averaging integer counters
would create incompatible states. This is a documented FedAvg buffer policy,
not FedBN. With the supplied BatchNorm's fixed momentum, the retained counter is
not used to set a cumulative averaging factor.

## Models and balancing

The character CNN now has a local encoder with a fixed configured alphabet,
case-preserving character IDs, an extra UNK channel, all-zero padding, and fixed
right truncation/padding. No vocabulary is learned from evaluation data. The
federated CNN config requires length >=8 so singleton batches remain valid for
BatchNorm after the two pooling stages. It does not use hashed text features or
fit a numerical scaler.

DeBERTa uses its original pretrained tokenizer and pinned checkpoint revision,
attention masks and explicit maximum length. Its full pretrained files must be
available locally on the server and clients; the runner never downloads them or
substitutes random encoder weights. See `NEURAL_MODELS.md`.

Training-only under/oversampling, unbalanced `none` and independent
`ensemble_undersampling` work with both CNN and DeBERTa. Ensemble members have
aligned independent global states, independent local subsets/optimizers and
probability-averaged evaluation. Large DeBERTa ensembles now use checksum-verified
chunked parameter RPCs rather than one oversized message.

SMOTE is supported with both models in the explicit continuous-embedding
warmup/frozen-head protocol. It is never applied to URL strings or IDs. All four
methods have matching protocol presets for fair comparisons. Read
[NEURAL_BALANCING.md](NEURAL_BALANCING.md) for implementation, budgets, original-
count weighting, byte accounting and memory/hardware requirements.

## Configuration and commands

For routine experiments, edit `config/experiment.json` and run
`python scripts/run_experiment.py`; `--check` validates without training and
`--mode smoke` runs synthetic verification. The launcher resolves the correct
model preset and existing IID/label/category client root, then invokes this same
Flower path. See [one-file setup](RUN_EXPERIMENT.md). Common active-file controls
override the base presets; completed runs retain both requested/effective configs
and can generate Matplotlib plots automatically. Low-level commands remain below.

Presets: `config/federated/character_cnn.json`, `deberta_v3_base.json`. Configure
`clients_per_round` (1-5), `local_epochs`, `batch_size`, `rounds`, `learning_rate`,
seed, model encoding, balancing, weighting policy and fixed threshold. None of
the preset values are claimed to be tuned or optimal. Exactly five clients are
required. Choose the existing IID, label-skew or category-skew root without
repartitioning or changing its splits.

Install into a project environment with `pip install -e ".[federated]"` for CNN,
or `pip install -e ".[federated,deberta]"` for DeBERTa. Flower is an optional
dependency; data preparation and historical report reading do not require it. The tested local
runtime is Python 3.13.5, Flower 1.22.0, torch 2.8.0+cpu, NumPy 2.2.3,
protobuf 4.25.8 and grpcio 1.74.0. Dependencies were installed in ignored
`data/flower-runtime/` for verification, without modifying global packages.
This workspace's temporary setup uses
`$env:PYTHONPATH = (Join-Path (Get-Location) 'data/flower-runtime') + ';' + (Join-Path (Get-Location) 'data/neural-runtime')`.
Normal installed environments do not need that temporary path.

```powershell
# Two rounds on tiny synthetic data, actual user CNN, all five clients:
python scripts/federated.py smoke --output data/fl-smoke-new

# Five local clients on an existing partition; choose a fresh output:
python scripts/federated.py local --config config/federated/character_cnn.json --partition-root data/client-iid-v1 --output data/fl-iid-cnn-new
# Swap root to data/client-label-skew-v1 or data/client-category-skew-v1.

# Server in one terminal; no dataset argument:
python scripts/federated.py server --config config/federated/character_cnn.json --output data/fl-network-server-new
# Run once per client in separate terminals/devices; IDs 0,1,2,3,4:
python scripts/federated.py client --config config/federated/character_cnn.json --client-id 0 --client-dir data/client-iid-v1/client_000
```

For client 1 use storage `--client-id 0`; repeat with IDs 1-4 and matching folders.
For DeBERTa swap the configuration, after placing its four pretrained files.
Client `--device` is explicit (default cpu); `--threads` bounds local CPU threads.
Clients must use the same public experiment config; model settings, optimizer,
batch size and balancing differences are rejected by the contract hash.

Network defaults bind loopback only. Remote addresses require TLS: server
`--address HOST:8080 --ca CA.pem --cert SERVER.pem --key SERVER.key`, clients
`--address HOST:8080 --ca CA.pem`. Keep keys/certificates outside version control.
Transport permits messages up to 1 GiB for the full single DeBERTa state.

## Artifacts, privacy and verification limits

Fresh ignored `data/` outputs contain per-round aggregate JSON, final tensor
checkpoint, full public settings, schema/parameter hashes and code/runtime
snapshots. No existing dataset or historical run is overwritten. Local balancing
logs print before/after counts, duplicates and seeds. The original client
partitions and all training/validation/test files remain unchanged.

This is data minimization with local evaluation, **not differential privacy or
secure aggregation**. The server sees each client's updates and coarse counts;
those can reveal information. The honest-client prototype does not provide
cryptographic proof of sample counts, prevent gradient inference, or authenticate
client IDs against impersonation. TLS protects remote transport, not the server
from seeing updates. No formal anonymity or privacy guarantee is claimed.

The synthetic smoke exercises real CNN training and Flower rounds, not URL
detection quality. Tiny-encoder tests cover the DeBERTa forward interface without
claiming pretrained-base fine-tuning. No full-corpus neural FL result has been
produced. The historical linear simulator still uses its documented central
validation design; do not attribute that design to this new local-evaluation path.

Verified on 23 October 2026: 239 regression tests passed, one opt-in network test
skipped in that run, and three existing localhost UI tests deselected because of
sandbox restrictions. The actual five-client gRPC test passed separately with
loopback access. The actual-CNN two-round smoke passed an independent checkpoint,
round-chain, count and immutable-file audit at
`data/federated-cnn-smoke-v2/verification.json`.

```powershell
python -m pytest tests/test_flower_federation.py -q
# Optional network integration check, where loopback sockets are permitted:
$env:BAIZE_TEST_FLOWER_NETWORK = '1'
python -m pytest tests/test_flower_network.py -q
```

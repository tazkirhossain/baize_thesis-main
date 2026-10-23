# Four balancing methods with Character CNN and DeBERTa

Implemented 23 October 2026. Both model families now support random undersampling,
random oversampling, embedding SMOTE and independent undersampling ensembles in
five-client Flower experiments. The same trainer accepts IID, label-skew and
category-skew client roots. There are 24 model/method/scenario combinations.

## Two explicit training protocols

Existing presets without `representation` retain ordinary end-to-end training.
They support undersampling, oversampling, unbalanced training and ensembles.
SMOTE is rejected in this mode because URLs and integer token/character IDs are
not a continuous interpolation space.

The new `encoder_embeddings` protocol supports all four methods with either
model. Each client, each round, each ensemble member:

1. Receive the corresponding global model state.
2. Warm up the encoder and original head using only the client's ORIGINAL
   training records for the configured `warmup_epochs`. This phase preserves the
   natural class imbalance; it is not claimed to be balanced training.
3. Freeze encoder behavior by entering evaluation mode. Extract continuous
   training embeddings in original record order into a temporary local disk cache.
4. Construct a balanced training view in that feature space. SMOTE interpolates
   Adult neighbors; under/oversampling and ensembles select original embeddings.
5. Keep the encoder frozen, including dropout and CNN batch-normalization state,
   while training the existing classifier head for `local_epochs`.
6. Return ordinary compatible model updates and agreed coarse statistics.

CNN features are the existing 512-dimensional pooled convolution output; the
original `fc` head is unchanged. DeBERTa features are the 768-dimensional first
encoder token; the original pooler, dropout and classifier form its head.
`extract_features()` followed by `classify_features()` matches ordinary evaluation
logits. Layer names, tensor shapes and the supplied CNN architecture are unchanged.
Evaluation always starts from original URLs with the complete model. No synthetic
URLs are generated, decoded or added to validation/test data.

This is a distinct two-stage neural experiment. It must not be described as
SMOTE applied directly to DeBERTa tokens, raw URL strings, or an unchanged
end-to-end training loop. The earlier engineered-numerical SMOTE utility remains
separate. The additional encoder warmup is recorded in budgets, processed counts,
loss scope and timing. Use the same two-stage protocol for all four balancing
methods when comparing them; the report exporter rejects mixed-protocol method
comparisons. No model or balancing method is presumed best.

## SMOTE implementation and boundaries

`src/data/embedding_smote.py` implements the standard rule
`new = first + u * (neighbor - first)` with a deterministic seed and uniform
`u` in [0,1). Neighbors come exclusively from Adult TRAINING embeddings.
Self-neighbors are excluded even for tied/duplicate vectors. This is the
[standard SMOTE interpolation rule](https://imbalanced-learn.org/stable/over_sampling.html#sample-generation),
applied here to a declared neural feature experiment; its detection quality
must be measured independently.

No fitted scaler is applied to these embeddings: `feature_scaling=none` is
recorded. Avoiding client-specific scaling keeps returned classifier parameters
in the shared encoder coordinates. The numerical URL-feature utility has its
own train-only StandardScaler and is not reused implicitly here. No learned
transform, neighbor count, ratio or threshold is fitted from holdouts.

Feature caches use float32 NumPy memory maps and are deleted after training,
including exception cleanup. Nearest-neighbor distance workspace is bounded;
synthetic vectors are produced lazily during minibatching rather than creating a
corpus-sized synthetic matrix in RAM. Temporary embeddings remain client-local.

If Adult count is below `k+1`, `adjust` reduces k using training support alone;
`skip` leaves the original head-training view and explicitly reports a skipped
SMOTE phase. Fewer than two Adult examples always skips interpolation. Targets
already met are explicit no-ops. Skips are not presented as successful synthetic
balancing: per-client status/k/counts, local member reports and pooled round skip/
adjustment counters are retained. Default target is Adult:Non-adult 1:1; 1:5
remains configurable. Original partition and split files never change.

## Ensembles and bounded model transfer

Every ensemble member has its own model, random seed, optimizer and independently
undersampled head-training subset. In end-to-end mode, its original URL training
view is undersampled. In embedding mode, each member has the same declared
original-data warmup then a separately sampled balanced embedding view. Each
available Adult example is retained. Predictions average member probabilities,
not logits or different members' weights. FedAvg averages corresponding members
across clients using ORIGINAL local training counts in the primary policy.

`src/transport.py` stages large Fit/Evaluate parameter lists through ordered
GetProperties RPC chunks. Individual chunks default to 4 MiB; transfer files are
limited to 16 GiB by default, with configurable bounds. Identifiers, contract,
chunk ordering, sizes and SHA-256 are checked before restoring the model state.
Only the existing serialized NumPy tensors are transported; no pickle objects,
URLs, feature matrices or datasets are sent. Small payloads retain native RPCs.

The server and clients use private temporary files under ignored `data/` or the
owning client's runtime directory. Successful transfers delete their staging
files. Explicit endpoint closure removes temporary workspaces; abrupt process
termination can leave ignored private runtime files requiring local cleanup.
The server retains no client feature cache. Actual chunk request/response
application protobuf sizes are included in fit/evaluation traffic counters;
registration, outer transport framing, gRPC/TLS and retries remain excluded.

Chunking addresses message-size limits. It does NOT make a multi-model DeBERTa
experiment fit into 8 GB RAM. Model copies, gradients and aggregation buffers
still require suitable hardware. The current local harness holds server and five
client model sets simultaneously. Full pretrained ensembles were not run on this
machine. A successful tiny-encoder test is not a full-scale resource benchmark.

## Presets and usage

Eight complete presets are in `config/federated/embedding_balancing/`:
`character_cnn_{method}.json` and `deberta_v3_base_{method}.json`, with method
`undersampling`, `oversampling`, `smote` or `ensemble_undersampling`.
They share the explicit representation protocol, warmup count, ratio and declared
training settings within each model family. Default ensemble size is five.

```powershell
# Actual CNN, tiny synthetic five-client check:
python scripts/federated.py smoke --config config/federated/embedding_balancing/character_cnn_smote.json --output data/cnn-smote-new
# Real experiment on capable hardware; choose one method and a fresh output:
python scripts/federated.py local --config config/federated/embedding_balancing/deberta_v3_base_smote.json --partition-root data/client-iid-v1 --output data/deberta-iid-smote-new
# Use client-label-skew-v1 or client-category-skew-v1 for independent non-IID scenarios.
```

The same configuration can be used with separate `server` and five `client`
processes/devices. Keep the DeBERTa assets locally available on each required
machine. Nothing automatically runs or ranks all 24 combinations.

Private per-member logs: each client's `.runtime/round_NNNN_member_NNN.json`
contains seeds, counts, duplicate/synthetic totals, SMOTE status/effective k and
training provenance. Global round JSON contains original/resampled/processed
counts, warmup counts, synthetic/skip/adjustment totals, metrics, timing and wire
costs. Exported JSON and plots label `training_protocol=encoder_embeddings`.
All final test and original-count FedAvg policies remain unchanged.

## Completed verification

174 regression tests passed; all three opt-in localhost network cases passed
separately, using an 8-KiB RPC cap and tiny real DeBERTa encoders for network
ensemble/SMOTE checks. Final aggregation/transport checks passed 16 tests.
The actual CNN completed a reproducible two-round five-client SMOTE smoke run.
The actual full pretrained DeBERTa completed one training-only embedding-SMOTE
check on six synthetic examples. Full pretrained DeBERTa ensembles and all
real-corpus model-quality comparisons remain unmeasured.

The independent audit is `data/neural-balancing-support-verification.json`.
It confirms unchanged source-data integrity metadata and pretrained hashes,
unchanged CNN tensor layout, exact synthetic/processed/wire counts and repeated
CNN final-state hashes. No best model/method is selected.

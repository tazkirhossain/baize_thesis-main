# Global checkpoints and training-only model selection

Implemented 23 October 2026 for both Character CNN and DeBERTa-v3-base, including
all members of an undersampling ensemble. It applies independently to each run
under IID, label-skew or category-skew partitioning and every balancing method.

## Files saved

Each new run writes under its ignored `data/<run>/server/` directory:

```text
checkpoints/
  round_0001.npz       All global state tensors after round 1 aggregation
  round_0001.json      Configuration, schema, checksum, membership hashes, metrics
  round_0002.npz
  round_0002.json
  ...
best_model.npz        Copy of the selected round's global weights
best_model.json       Matching schema/config/checksum plus selection decision
global_final.npz      Global weights from the fixed final round, always separate
run.json              Completed-run metadata, selected round, criterion and score
round_0001.json       Existing per-round training/selection/validation measurements
...
```

Round weights are written immediately after successful aggregation, before
evaluation. Metadata initially says `aggregated_pending_evaluation`, then
`evaluated` after all five clients return valid evaluation statistics. An
interruption before evaluation finishes can leave an aggregated checkpoint but
cannot promote it to best. A failed run is not marked PASS. Completed earlier
rounds survive later interruptions. Incomplete `.tmp` files are not checkpoints.

Writes use temporary files and atomic replacement. NPZ arrays are loaded with
`allow_pickle=False`; reload checks exact keys, tensor shape/dtype/finiteness,
model schema, experiment contract and global state checksum. No URLs, raw IDs or
client datasets are stored in server checkpoints. Each file contains the entire
global model or ensemble; individual client updates are not archived here.

## Select the best checkpoint

### Optional threshold stored with each checkpoint, 26 October 2026

The active experiment also enables `threshold_selection` with source
`training_selection`, metric `adult_f1`, 201 score bins and tie-break
`highest_threshold`. It requires the existing domain-disjoint `model_selection`
reservation. Clients supply only class-conditional bin counts. Each round's
threshold maximizes pooled TRAINING-SELECTION Adult F1 on the declared grid;
ties choose the higher threshold. Round selection then uses that selection F1,
retaining the earliest round on equal scores. The checkpoint metadata binds its
threshold to the selected model; optional final test uses that exact model and
threshold once after the configured budget. Validation/test never choose either.

Fixed-0.5 reference measurements remain available alongside selected-threshold
metrics. Configurations without `threshold_selection` retain the existing fixed
threshold. Initialization and threshold policies form part of the experiment
contract; a new run is needed, and old checkpoints must not be silently labelled
with the new policies. See [diagnosis](ADULT_PERFORMANCE_DIAGNOSIS.md).

The active `config/experiment.json` now declares:

```json
"model_selection": {
  "source": "training_selection",
  "fraction": 0.1,
  "metric": "adult_f1"
}
```

`fraction` is a probability target over original TRAINING domains, not an exact
row quota. The distinct `shared-training-selection-domain-v1` hash and experiment
seed assign every training domain to optimization or selection. Client ID,
labels, model scores and scenario do not enter the role assignment. With the
same seed/fraction, a domain has the same role across clients and IID/non-IID
scenarios. The original `baize-domain-role-v1` train/validation/test rule is
unchanged. No original data file, partition, split or corpus total is rewritten.

Approximately 90% of original training domains remain eligible for optimization;
10% become the selection subset. Actual class/row counts vary with domain sizes.
Selection rows are immutable/protected and are excluded from balancing,
duplication, SMOTE feature fitting, encoder warmup and head/full-model training.
Class ratios and sampling use optimization labels only. Original validation/test
membership and distribution remain unchanged. Clients log assigned, optimization
and selection counts, class counts, seed, policy and membership hashes locally
under `.runtime/training-selection-<contract>.json` without exporting raw IDs.

FedAvg `original_training` now counts the unresampled ORIGINAL OPTIMIZATION
records, excluding selection records, duplicates and synthetic examples.
`assigned_train = original_train + selection_train` is checked by the server.
This gives weight only to data eligible for optimization. Keep the same selection
policy for all primary comparisons. Configuration contracts and comparison guards
separate runs with different selection policies, including historical runs that
optimized their entire original training split.

All five clients evaluate selection rows after every global round and send only
counts/loss/time. The server sums confusion counts and computes pooled Adult F1.
It promotes a checkpoint only on a strict F1 improvement; equal scores retain
the earliest round. No validation/test score, loss or accuracy breaks ties.
An empty local selection subset contributes zero counts; pooled selection must
contain both classes or the experiment fails explicitly. Do not search seeds or
move evaluation records to manufacture supports. Missing optimization classes
remain subject to the existing balancing safety checks.

Validation is still measured each round for reporting. Train curves now describe
the unresampled optimization pool. If `evaluate_final_test=true`, test is evaluated
ONCE after the full configured FL budget on the selected checkpoint. The final
round report records `test_checkpoint.round` and `global_hash`, and exported test
rows retain checkpoint round/hash so an earlier selected model is not presented
as the final round's weights. The final global checkpoint remains separately
saved. Current active `evaluate_final_test` remains false; no test was enabled
implicitly. Selection supports appear in round JSON and console summaries.

"Best" means highest training-selection Adult F1 WITHIN ONE RUN. It does not
establish the best model family, balancing method or distribution across studies.
Those comparisons still require measured validation/final test results and
matched protocols/budgets; no cross-experiment winner is declared automatically.

Older full configurations without `model_selection` remain compatible: every
round is saved, the fixed final model is used, and no best checkpoint is selected.
Set `model_selection` to null to explicitly use that same fixed-final policy.

## Reload a saved model

Using the saved configuration and matching local model/tokenizer resources:

```python
from src.models.federated_state import StateCodec, make_models
from src.server.checkpoints import load_checkpoint
from src.server.config import contract_hash

models = make_models(configuration)
codec = StateCodec(models)
arrays, metadata = load_checkpoint(checkpoint_path, codec,
                                  expected_contract=contract_hash(configuration))
codec.set(arrays)
for model in models:
    model.eval()
```

Use `best_model.npz` or a `checkpoints/round_NNNN.npz` path. Preserve adjacent JSON
and the original configuration/source snapshot. For DeBERTa the original local
tokenizer/pretrained resources must still exist to reconstruct the model.

These are loadable global MODEL checkpoints, not complete training-session
snapshots: automatic resume, client optimizer/RNG state restoration and mid-round
continuation are not implemented. Starting the launcher still begins a new run.
Storage grows with rounds, model size and ensemble count; no round is deleted by
retention. All checkpoints/data remain ignored by Git.

# Client-local class balancing

Updated 23 October 2026: numerical logistic classifiers, their numerical ensemble
and fitting helpers, `src/client/balancing_experiment.py`, and
`scripts/run_balancing.py` have been removed at the user's request. Historical
reports and immutable source snapshots remain for audit. They describe past
experiments, not a currently runnable numerical classifier.

## Current model support

| Method | Character CNN | DeBERTa-v3-base |
|---|---|---|
| Random undersampling | Supported | Supported |
| Random oversampling | Supported | Supported |
| Ensemble with undersampling | Supported, probability averaging | Supported, probability averaging and chunked updates |
| SMOTE | Supported on continuous encoder embeddings | Supported on continuous encoder embeddings |

Use `config/federated/embedding_balancing/` for consistent four-method comparisons.
The explicit two-stage protocol warms the encoder on original local training,
freezes it, then balances embeddings and trains the existing head. See
[NEURAL_BALANCING.md](NEURAL_BALANCING.md) for exact semantics, caches, safeguards,
skip/adjust policies, communication accounting and hardware limits.

The earlier numerical-feature SMOTE utility and train-only StandardScaler remain
separate, with no numerical classifier runner. Never apply SMOTE to raw URL
strings or character/tokenizer IDs. Original end-to-end training remains supported
for the other methods; reports reject mixing its budget with the two-stage
protocol as if they were identical comparisons.

## Training-only rules

Each client uses its existing training split; never rewrite its source partition,
training file, validation file or test file. Validation/test labels are protected.
The client verifies split checksums and source-record disjointness. Balancing
uses training labels only and logs original/resulting class counts, seeds,
duplicate counts and member counts. All four presets default to Adult:Non-adult
**1:1**; `1:5` remains an explicit configurable ratio.

- Undersampling retains every available Adult example and samples Non-adult
  without replacement. Unused majority rows stay in the original files.
- Oversampling retains originals and duplicates Adult training selections with
  replacement to the target ratio.
- Each CNN ensemble member uses its own deterministic seed and independently
  undersampled training subset, retaining Adult examples. Evaluation averages
  member probabilities; members have independent optimizers and aligned global
  states across clients.
- SMOTE interpolates floating continuous training vectors only. It
  adjusts neighbors or explicitly skips when minority training support is too
  small. It never generates URLs or integer token sequences.

## Select one method

Use a complete preset in `config/federated/embedding_balancing/` for four-method
neural comparisons. Original end-to-end runs use the `balancing` section in
`config/federated/character_cnn.json` or `config/federated/deberta_v3_base.json`. Model-independent presets in
`config/balancing/` contain only that section; copy it into a supported neural
configuration. They are not standalone experiment configurations.

```python
from src.data.balancing import BalanceConfig, sample_training_indices

config = BalanceConfig(method="oversampling", adult_to_non_adult="1:1", seed=42)
selection = sample_training_indices(training_labels, config, split="train")
training_view = selection.apply(original_training_dataset)
```

`ensemble_training_indices()` returns an independent selection per member.
For Flower commands, models, aggregation weights and supported combinations,
see `FEDERATED_TRAINING.md`. Compare methods using separate runs with the same
partitions and local splits. Neither class counts nor historical numerical
results establish the best neural method.

## Historical numerical experiments

The following sections record completed experiments before classifier removal.
Their data, reports, warnings and original code snapshots remain unchanged.
Historical `scripts/summarize_balancing.py` and `scripts/report_evaluation.py`
can still read saved summaries without fitting a classifier.

## Completed first real-data comparison

The declared 1:5 comparison completed on all 15 clients (five each in historical
IID, label skew and category skew): **60 method runs and 120 independently fitted
numerical classifiers**, counting five members in each ensemble. All runs
completed; no SMOTE run needed neighbor adjustment on these clients.

Example: IID Client 1's ORIGINAL TRAINING split has 5,102 Adult and 206,064
Non-adult records. The methods produced:

| Method | Adult used | Non-adult used | Duplicated / synthetic |
|---|---:|---:|---|
| Undersampling | 5,102 | 25,510 | 0 |
| Oversampling | 41,213 | 206,064 | 36,111 duplicated Adult training selections |
| SMOTE | 41,213 | 206,064 | 36,111 synthetic numerical training vectors |
| Ensemble, each of 5 members | 5,102 | 25,510 | 0 within each member; independent subsets |

All original records remain stored unchanged. These are training-view counts,
not rewritten partition sizes. Client 1 is storage client_000.

Artifacts:

- `data/balancing-four-methods-v1/run.log`: all printed count/effect/metric output.
- `data/balancing-four-methods-v1/summary.json`: all configurations, 60 runs,
  per-client reports, source split fingerprints and numerical transform metadata.
- `data/balancing-four-methods-v1/verification.json`: independent after-run audit.
- `results/balancing_numerical_comparison_v1.md` and `.json`: sanitized comparison
  metrics pooled from independent client predictions. This is not a global FL
  model. Pooling confusion counts differs from averaging client F1 scores; both
  are explicitly named in the JSON.

Regenerate a summary with a fresh output filename:

```powershell
python scripts/summarize_balancing.py data/balancing-four-methods-v1/summary.json --output results/balancing-comparison-new.json
```

The independent audit checked exact target/duplicate/synthetic counts, distinct
ensemble selections, unchanged validation/test supports, training-only scaler
metadata and every original client/split file fingerprint. All 15 original client
datasets and 45 split files remained unchanged. Eleven ConvergenceWarnings were
recorded at the predeclared 30-epoch cap, including ten ensemble member fits.
Those warnings were not used to retune the experiment after seeing holdouts.
This single-seed initial comparison does not establish a robust winning method.
Method effects can differ by representation and model: compare neural methods
within their own family later, retaining numerical SMOTE as its separate study.

Validation: 214 tests passed, with three existing localhost HTTP tests deselected
because connections are blocked in this environment. Tests exercise both 1:1
and 1:5 sampling, training/holdout guards, numerical SMOTE, neighbor adjustment
and skip policies, train-only scaling, ensemble independence/averaging, required
metrics, deterministic results, CLI output and one-method selection.

## Completed equal-class run, 21 October 2026

Re-ran all four methods independently for all five clients in each of IID,
label-skew non-IID and category-skew non-IID: **60 successful method runs and
120 independently fitted numerical classifiers**. Every resulting training view
has equal Adult and Non-adult counts; all five members of each undersampling
ensemble satisfy the same equality. Oversampling and SMOTE retain the original
Non-adult count; undersampling and ensembles retain the original Adult count.

Example IID Client 1, originally 5,102 Adult / 206,064 Non-adult:

| Method | Adult | Non-adult | Added Adult examples |
|---|---:|---:|---|
| Random undersampling | 5,102 | 5,102 | None |
| Random oversampling | 206,064 | 206,064 | 200,962 duplicate selections |
| Numerical SMOTE | 206,064 | 206,064 | 200,962 synthetic feature vectors |
| Ensemble, each member | 5,102 | 5,102 | None |

The independent audit verified all 15 original client dataset hashes and all
45 train/validation/test file hashes, holdout class supports, training-only scaler
fits, exact duplicate/synthetic counts, and distinct ensemble seeds/selections.
Results are in `data/balancing-four-methods-equal-v2/summary.json`, `run.log`,
per-client reports and `verification.json`. The audit script is
`data/verify_balancing_equal_v2.py`. Sanitized evaluation metrics are in
`results/balancing_numerical_equal_v2.md` and `.json`.

These remain numerical-feature experiments, not DeBERTa, character CNN or a
federated aggregated model. There were **74 recorded ConvergenceWarnings** at
the unchanged 30-epoch cap. Equal class counts do not establish model quality;
no method or settings were selected or changed based on these holdout results.
The earlier 1:5 run remains intact. Focused verification: **32 tests passed**.

### Exact equal-class counts per client

Each cell below is the count **in each class**, not their combined total.
Undersampling ensemble counts apply separately to every member.

| Partition | Client | Undersampling / each ensemble member | Oversampling / SMOTE |
|---|---:|---:|---:|
| client-iid-v1 | 1 | 5,102 | 206,064 |
| client-iid-v1 | 2 | 4,972 | 206,277 |
| client-iid-v1 | 3 | 5,026 | 206,351 |
| client-iid-v1 | 4 | 5,011 | 206,978 |
| client-iid-v1 | 5 | 5,026 | 205,902 |
| client-label-skew-v1 | 1 | 8,679 | 202,551 |
| client-label-skew-v1 | 2 | 6,124 | 205,354 |
| client-label-skew-v1 | 3 | 4,701 | 206,643 |
| client-label-skew-v1 | 4 | 3,220 | 208,282 |
| client-label-skew-v1 | 5 | 2,413 | 208,742 |
| client-category-skew-v1 | 1 | 5,025 | 207,708 |
| client-category-skew-v1 | 2 | 5,016 | 207,630 |
| client-category-skew-v1 | 3 | 5,036 | 204,043 |
| client-category-skew-v1 | 4 | 5,032 | 208,928 |
| client-category-skew-v1 | 5 | 5,028 | 203,263 |


## Verification after removal

Verification after removal: **242 tests passed**, one opt-in network test
skipped and three existing localhost UI tests deselected. Retained neural
training/evaluation, partitioning, train-only balancing, SMOTE/scaling utilities
and historical report readers passed. Historical source-summary and final-report
SHA-256 hashes remain unchanged.

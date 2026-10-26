# CNN optimization investigation — 26 October 2026

Validation/test data was not balanced or used to choose these changes. All raw
records accessed came from clients' existing original TRAIN files. The fixed
domain-selection reservation was applied before drawing optimization samples.
URLs, individual scores and record IDs were not exported.

## Saved global checkpoint

The saved round-1 checkpoint was scored without fitting on a fixed sample of
Client 1's training-selection domains: 4,096 records, 72 Adult. ROC-AUC was
0.4506; AP was 0.0288. Finite-grid and exact score thresholds both gave weak
Adult F1. This sample identifies weak ranking, not merely a threshold issue.
It is one client sample, not the pooled global validation result.

## Can the supplied CNN learn its training examples?

Same stable CNN initialization, 64 optimization examples (32 per class), input
length 256, batch 32, 60 updates, same shuffle/dropout seeds and CPU two threads:

| Local optimizer | Learning rate | Last-ten-update mean CE | Optimization ROC-AUC |
|---|---:|---:|---:|
| SGD, no momentum | 0.001 | 0.6921 | 0.4482 |
| SGD, no momentum | 0.01 | 0.6722 | 0.7637 |
| AdamW, zero weight decay | 0.001 | 0.0280 | 0.9980 |

The first-convolution relative parameter change was 0.000301 with SGD at 0.001,
versus 0.198997 with AdamW at 0.001. Gradients reached both the encoder and head.
The current SGD rate/budget barely trained the encoder; the model could learn
the small sample with adaptive updates. This is a memorization check, not
generalization evidence. Training-mode loss and eval-mode loss differ because
dropout and BatchNorm behave differently; no holdout statistics fit BatchNorm.

## Matched local pilots on original training domains

For each of Clients 1 and 2, draw 512 Adult and 512 Non-adult optimization
records without replacement. Train for three local epochs (48 updates, batch
64). Score a separate fixed uniform sample of 4,096 original training-selection
records without balancing. Both optimizer arms use identical data, stable
initialization, dropout/shuffle seeds, architecture and update budget. The
selection samples contain 82 and 68 Adult records respectively.

| Client | Optimizer | Selection ROC-AUC | Selection AP | Selection Adult precision | Selection Adult recall | Selection Adult F1 |
|---|---|---:|---:|---:|---:|---:|
| 1 | SGD 0.001 | 0.5025 | 0.0219 | 0.0213 | 0.7073 | 0.0414 |
| 1 | AdamW 0.001 | 0.7366 | 0.2185 | 0.4000 | 0.1951 | 0.2623 |
| 2 | SGD 0.001 | 0.4778 | 0.0165 | 0.0170 | 0.5441 | 0.0330 |
| 2 | AdamW 0.001 | 0.7445 | 0.1631 | 0.6429 | 0.1324 | 0.2195 |

Classification metrics use the 201-bin Adult-F1 threshold selected on that
training-selection sample. This is selection-set performance and can be
optimistic. AUC/AP use locally held scores; scores remain local. These two
client-local samples are not independent statistical replications: the shared
domain-role policy allows domains to appear in multiple clients. This is not
five-client FedAvg, a completed main experiment or a final-test comparison.
Recall at the selected F1 operating point remains low; no good-result guarantee
or best-method/model claim is supported.

## Implemented next configuration

`config/experiment.json` now explicitly declares:

```json
"learning_rate": 0.001,
"optimizer": {"name": "adamw", "weight_decay": 0.0}
```

All other active data/distribution/balancing/budget settings remain as provided.
The CNN architecture, training-only balancing, original-count FedAvg weighting,
unchanged validation/test populations and training-only checkpoint/threshold
selection remain. `src/client/optimizers.py` is shared by end-to-end training,
encoder warmup and balanced head training, for both models and all methods.
Local AdamW state resets per phase/member/round and is not aggregated; the
server remains FedAvg. Historical configs without `optimizer` retain exact
plain SGD behavior. Optimizer settings participate in the public contract and
report comparability checks. No class-weighted/focal replacement was added.

A fresh process is required; the running SGD experiment retains its original
configuration. A full new run must measure whether the improvement survives
five-client aggregation, more training data and final testing. No optimizer
has been established as universally optimal. Model/ratio/epoch changes were
not selected from validation/test results.

## Reproduce and inspect

`scripts/diagnose_cnn_learning.py` exposes `sanity`, `pilot` and `checkpoint`
stages. Output directories must be fresh. It reads only the own-client train
file, uses fixed domain roles and records source/membership fingerprints.
Each arm explicitly overrides optimizer/rate; remaining model settings match.
Run from the repository root using the configured project Python environment:

```powershell
& $pythonExe scripts/diagnose_cnn_learning.py --stage sanity --steps 60 --output results/diagnostics/NEW_SANITY
& $pythonExe scripts/diagnose_cnn_learning.py --stage pilot --arms sgd_001 adamw_001 --epochs 3 --per-class 512 --selection-size 4096 --output results/diagnostics/NEW_PILOT
```

Evidence lives under `results/diagnostics/`:

- `cnn-round1-scores-v1/measurements.json`
- `cnn-learnability-sanity-v1/measurements.json`
- `cnn-optimizer-pilot-client1-v1/measurements.json`
- `cnn-optimizer-pilot-client2-v1/measurements.json`

These are bounded real-training diagnostics; synthetic integration tests
separately verify the actual Flower pipeline, not Adult detection quality.

Structured comparison rows, including source hashes, are saved in
`results/diagnostics/cnn-optimizer-comparison-v1.csv` and `.json`. Each diagnostic
folder retains its original executable source snapshot and provenance record.

Verification: the full regression suite passed 427 tests with 3 optional tests
skipped. A subsequent exact-update parity check passed, confirming the production
factory matches the AdamW implementation measured by the diagnostic. Coverage
includes all four balancing methods, both end-to-end/embedding paths, both neural
model interfaces, and an actual two-round/five-client CNN AdamW synthetic run
with saved/reloaded selected checkpoint and frozen-threshold final evaluation.
Compact synthetic server metadata is in `results/verification/adamw-flower-smoke-v1/`.
Disposable generated test directories were removed afterward; source tests/logs
remain, and `data/` was not repopulated with test fixtures.

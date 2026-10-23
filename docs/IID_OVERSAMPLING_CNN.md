# First Character CNN run

Routine setup now uses **`config/experiment.json`** and
**`python scripts/run_experiment.py`**, which selects the correct IID root and
can generate figures automatically. See [the one-file guide](RUN_EXPERIMENT.md).
The earlier complete preset and low-level commands below remain supported.

Use `config/federated/iid_oversampling_character_cnn.json` for five clients,
two Flower rounds, one local epoch per participating client per round, batch
size 8, SGD learning rate 0.001 and seed 42. All five clients participate in
each round. Change `rounds`, `local_epochs`, `batch_size` and `learning_rate`
in that file before starting a new run.

This preset trains the complete CNN end to end. It has no embedding warmup or
separate head-training stage. Each client oversamples its own TRAINING split
to Adult:Non-adult 1:1. Original files and evaluation distributions stay intact.
FedAvg weights by original training counts, before duplication. Test evaluation
is disabled; training and validation metrics are collected after each round.
The unused ensemble settings do not create multiple models for oversampling.

## Run on this Windows machine

Open PowerShell and run:

```powershell
Set-Location 'D:\baize_thesis-main\baize_thesis-main'
$pythonExe = 'C:\Users\hossa\anaconda3\python.exe'
$env:PYTHONPATH = (Join-Path (Get-Location) 'data/flower-runtime') + ';' + (Join-Path (Get-Location) 'data/neural-runtime')
$runStamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'

# Quick execution check; tiny synthetic fixtures, CPU, all five clients.
& $pythonExe scripts/federated.py smoke --config config/federated/iid_oversampling_character_cnn.json --output "data/iid-over-cnn-smoke-$runStamp" --threads 2
if ($LASTEXITCODE -ne 0) { throw 'Smoke test failed' }
```

The existing smoke command explicitly uses two rounds, batch size 8 and URL
length 32, retaining the configured local epoch count and balancing method.
Its five synthetic fixtures have different label proportions: they are NOT the
real IID partitions. Run metadata therefore marks them synthetic with unknown
partition strategy. Passing smoke verifies training, balancing, aggregation and
metrics; its scores do not measure real URL detection quality. Inspect the
output `experiment.json` for effective settings.

To train the selected combination on the **existing full real IID splits**, run:

```powershell
$runStamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
& $pythonExe scripts/federated.py local --config config/federated/iid_oversampling_character_cnn.json --partition-root data/client-iid-v1 --output "data/iid-over-cnn-real-$runStamp" --device cpu --threads 2
if ($LASTEXITCODE -ne 0) { throw 'IID training failed' }
```

The partition root selects actual records; experiment metadata alone does not
select a distribution. `local` simulates all five Flower clients sequentially
in one process, so no extra terminals or devices are needed. It honors length
256 and the other preset settings without smoke overrides. A local epoch visits
the complete oversampled training view, not one batch. Real IID training has
1,056,709 original training records and 2,063,144 resampled examples per round
at 1:1, plus original training/validation evaluation passes. Two rounds can take
a long time on this CPU-only machine. This is a full-data run, not a small pilot.

Each output directory must be new. Outputs are ignored local artifacts under
`data/`: `experiment.json`, `reproduction.json`, `server/round_0001.json`,
`server/round_0002.json`, the final model checkpoint and `server/run.json`.
Client logs print original/resulting counts and duplicate counts. Raw local
datasets are not sent to the server.

After a successful run, generate structured tables and convergence plots:

```powershell
# Replace RUN_DIRECTORY with the exact directory printed by the runner.
& $pythonExe scripts/report_evaluation.py --federated-run data/RUN_DIRECTORY/server --output data/RUN_DIRECTORY/report
```

Matplotlib 3.10.3 is already installed on this machine. The exporter saves PNG
and SVG without requiring a desktop plotting window. Convergence panels show
original-training/validation loss, accuracy, balanced accuracy, macro-F1, Adult
precision/recall/F1 and Non-adult F1. Additional plots show the fixed final-round
confusion matrices and per-round training time/communication payload sizes.
Test confusion matrices appear only when final-test evaluation was enabled.
Run the exporter after training completes; it is a separate command, not an
automatic part of `federated.py`. It works for both models and all scenarios.

The user's completed synthetic smoke has been plotted at
`data/cnn-over-smoke-20261023-163747/visualizations/README.md`. The confusion
matrix confirms its final model predicted every synthetic record as Adult.
Those figures are execution-check evidence, not real-data performance.

Readiness checks on 23 October 2026 verified all existing real IID split hashes,
counts and disjoint source rows (1,468,243 records), complete-original retention
and 1:1 training-only oversampling, and one real training forward/backward/SGD
batch for each client at configured length 256. All gradients/losses were finite
and model weights changed in memory. No split files or checkpoints were changed;
no validation or test inference was performed by this check. Evidence:
`data/iid-cnn-real-readiness-20261023-v1.json`. This establishes compatibility,
not complete-epoch convergence, available resources throughout a long run, or
real URL classification quality.

Assess Adult recall/F1, macro-F1 and balanced accuracy alongside overall accuracy.
A two-round run is an execution/initial-learning check, not evidence that this
combination is best. For subsequent comparisons that include SMOTE, use the
matched two-stage presets documented in `NEURAL_BALANCING.md`; do not present
those runs and this end-to-end run as an identical training protocol.

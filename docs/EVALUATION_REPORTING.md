# Evaluation, timing and research comparisons

Numerical logistic classifiers and their training runner were removed on
23 October 2026. Historical numerical reports remain readable; new research
training uses Character CNN or DeBERTa-v3-base via `scripts/federated.py`.
Continuous-embedding SMOTE and ensembles now support both neural models; see
`NEURAL_BALANCING.md` for the explicit protocol and chunked transport.

Implemented 23 October 2026. Use `scripts/report_evaluation.py` to export
reproducible JSON, Markdown tables and standalone PNG/SVG figures. It consumes
completed aggregate result files, never raw URL datasets. Existing source runs
and real partitions remain immutable. Outputs require a fresh directory.

## Classification metrics

### Training-only threshold metrics, 26 October 2026

Optional `threshold_selection` evaluates a declared 201-bin grid using pooled
TRAINING-SELECTION class-conditional score counts. It requires the existing
`model_selection` reservation. Adult F1 chooses the threshold, with higher
thresholds winning equal scores. Each round retains fixed-0.5 reference metrics
alongside selected-threshold metrics. Validation and optional original-training
metrics use that round's selected threshold; optional final test uses the frozen
selected checkpoint and its own recorded threshold. Validation/test never select
or adjust either. Model loss is independent of the decision threshold.

Clients return 2-by-201 histograms rather than raw scores, records or individual
predictions. These coarse statistics are visible to the server and add no formal
privacy/security guarantee. Grid-based curves or operating points must be
labelled approximate; do not present them as exact score-level PR-AUC. Missing
historical histograms remain missing. Preserve the configured threshold policy
and versioned CNN initialization in comparison metadata, and do not mix legacy
fixed-threshold runs with the new active policy as identical methods. See
[diagnosis and limitations](ADULT_PERFORMANCE_DIAGNOSIS.md).

The shared calculator `src/evaluation/classification.py` reports Adult and
Non-adult precision, recall and F1, macro-F1, balanced accuracy, overall accuracy,
confusion matrix, both class supports and observed Adult prevalence.
Matrices use true rows / predicted columns in [Non-adult, Adult] order:
`[[TN, FP], [FN, TP]]`. Undefined class divisions use zero; macro-F1 and balanced
recall average the fixed two-label set. The audited real-data splits contain
both classes. Empty matrices, negative/fractional/nonfinite counts fail.

Per-class formulas are independently tested against scikit-learn. Report inputs
are checked against their saved confusion counts; contradictory derived metrics
fail instead of being silently accepted. Historical negative-class metrics are
derived from saved confusion matrices without retraining or changing their totals.
Pooling means **sum confusion counts first, then calculate metrics**. Averaging
client F1 values is a different quantity and is not substituted for pooled F1.

Adult is approximately 2.37% of the cleaned corpus. An all-Non-adult classifier
can therefore achieve about 97.63% accuracy while Adult recall/F1 are zero.
Report each split's actual measured prevalence; do not assume it is exactly the
full-corpus proportion after domain-aware splitting. Accuracy is included as
context and is never an automatic selection criterion.

## Per-round neural evaluation

After FedAvg, **all five clients** evaluate the updated global model locally on
their original training and validation splits. Original-training evaluation is
enabled by default through `evaluate_training=true`; it is an additional read-only
pass, with dropout off and no optimizer/scaler fitting. It preserves the natural
training population for interpretable train/validation curves. It is not accuracy
on the balanced optimization view. Set False only as an explicitly recorded
cost tradeoff; missing training metrics are then not invented by the exporter.

Separately, `optimization.loss` records the example-weighted loss observed during
local training on balanced views, across configured epochs and ensemble members.
It is not the fixed global model's original-training loss. No validation/test
data changes sampling ratios, weights, learning rate, thresholds or model state.

Only confusion counts, loss sums and aggregate timing leave clients; local URLs,
raw labels, token sequences, source-row IDs and individual probabilities do not.
Clients also provide whole-file train/validation/test fingerprints and coarse
device/thread/runtime metadata for provenance, never file contents. These are
data minimization measures, not differential privacy or secure aggregation.

With `model_selection` enabled, the best round uses pooled Adult F1 on a fixed
training-only selection subset, with earlier rounds retained on ties. Those rows
are excluded from optimization/balancing. Legacy presets without that option
retain the fixed final round. Validation is reported every round and never selects
the model. Optional test occurs once after all rounds when configured, on the
selected checkpoint (or the final checkpoint in legacy mode). Exports identify
the tested checkpoint's round/hash separately from the final communication round.
This is checkpoint selection within one run, not a cross-method winner claim.
See `CHECKPOINTS.md`; test data must not drive tuning.

## Timing and communication definitions

- `train_seconds`: client wall time for member training loops, including lazy
  encoding, optimizer initialization, device work and return to CPU. Includes
  warm-up overhead; it is not isolated accelerator kernel time.
- `fit_seconds`: the broader client fit call, including local reads, balancing,
  validation of incoming parameters and state extraction.
- `client_training_seconds_sum` / `max`: summed client work and longest client
  work. The sum is not parallel elapsed time.
- `fit_round_wall_seconds`, `val_phase_wall_seconds`, `round_wall_seconds`:
  server-observed phase/round elapsed time. Test phase is recorded separately.
- `communication`: exact byte sizes of serialized Flower Fit/Evaluate protobuf
  payloads, both directions. This includes tensor serialization and metric fields,
  not gRPC/TLS framing, registration/control traffic, retransmissions or retries.
  It is application-payload accounting, not measured physical network traffic.
  An optional final-test exchange increases the final round's payload total.

Original and resampled training counts, examples processed, FedAvg weighting
policy, global state hashes, model/config schema and seeds remain recorded.
The original-training-count primary weighting policy is unchanged.

New numerical local runs separately record training time and evaluation/audit
time. Their historical `elapsed_seconds` includes both and cannot be retrospectively
called pure training time. Reports keep missing old measurements null, not zero.
Local numerical experiments have no FL communication cost. Byte counts and
neural runtime measurements are not imputed for old numerical runs.

## Reproduction and comparison guards

`config/reporting/comparison.json` declares partition aliases, expected five
clients, the four original methods, required 1:1 ratio, output formats and no
automatic method selection. A 1:5 source requires a separate explicitly changed
report configuration; old runs are never relabelled as 1:1.

The exporter records exact model/classifier, feature-transform fingerprint,
client sampling seeds, ratio, threshold, optimizer/training settings, original
split fingerprints, runtime versions, input/source-code hashes and plotting
configuration. Federated run metadata retains the parent code/runtime snapshot.
`experiment` in the FL config holds public partition/data-kind/cohort labels;
client-reported preparation strategy and synthetic/real status resolve the actual
kind where available. A public cohort label is not cryptographic proof of equal
evaluation membership: retain the prior preparation/domain-role audit.

Method comparisons reject mismatched inputs/transforms/thresholds or duplicate
cells. Federated comparisons also reject different optimizer/model settings,
weighting policies or local data contracts within the same scenario. Cross-partition
experiments require matching declared evaluation cohorts and settings. Different
families, local-versus-FL scope, synthetic-versus-real data and seeds are separate
plot cohorts; they are never pooled into one performance result. Multi-seed
uncertainty is not fabricated from a single seed or from client-to-client spread.

Real-data coverage tables explicitly say **not measured** for neural runs that
have not occurred. Token-ID SMOTE remains invalid. Continuous-embedding SMOTE and DeBERTa
ensembles are supported; unrun combinations remain not measured. Historical numerical
results remain readable; no numerical classifier experiment runner remains.

## Commands and outputs

Install plotting with `pip install -e ".[reporting]"` in the project environment.
The tested Matplotlib version is 3.10.3. Run from the repository root.

```powershell
python scripts/report_evaluation.py --balancing-summary data/balancing-four-methods-equal-v2/summary.json --federated-run data/federated-cnn-evaluation-v4/server --output results/evaluation-new
```

Repeat `--federated-run` to include completed compatible IID, label-skew and
category-skew neural experiments when available. Export without either input
type when only the other is available; missing comparisons remain explicit.

Output includes `report.json` (all requested metrics, confusion matrices,
supports, timing, costs and reproduction metadata), `README.md` tables, and
publication-ready 300-DPI PNG plus SVG/PDF vector figures:

1. IID versus separate label/category non-IID comparisons.
2. Balancing methods under IID.
3. Balancing methods under label-skew and category-skew non-IID.
4. Adult recall versus F1 across methods, with explicitly focused scatter axes.
5. Original-training/validation loss, accuracy, balanced accuracy, macro-F1,
   Adult precision/recall/F1 and Non-adult F1 across FL rounds.
6. Additional measured client-time and protobuf-payload charts.
7. Fixed final-round original-training/validation confusion matrices, plus
   final test only when measured. Counts and percentages within each true class
   expose majority-only/minority-only predictions; panels have separate count
   color scales. No round is selected using evaluation scores.
8. Five-client original/balanced Adult prevalence and declared FedAvg weights.
9. Per-client local optimization-loss and training-time heatmaps across rounds;
   missing cells identify clients not sampled in a round.

The client figures contain only coarse counts, local training loss/time and
aggregation weights already received by the Flower server. They contain no URLs,
record identifiers, examples, predictions or local validation metrics.

The first four figures compare the existing **real numerical** experiments.
The current convergence/cost figures use an explicitly labelled **synthetic
character-CNN smoke run**. These do not fill missing real CNN/DeBERTa results.
No data or checkpoints are exported with the aggregate report. All actual model
states, synthetic/local datasets and sensitive run artifacts stay in ignored data/.

## Verified current export

Final artifacts are `results/evaluation_reporting_v4/README.md`, `report.json`,
six figures in both PNG and SVG, and `verification.json`. There are 29 measured
rows: 24 real numerical val/test groups and five synthetic CNN per-round/final-
test rows. The original 1:1 numerical summary is unchanged. Its 74 fit warnings
remain explicit; no source result was relabelled or retrained for this report.

Verification: 248 regression tests passed, with one opt-in network test skipped
and three existing sandbox-restricted localhost UI tests deselected. The real
five-client gRPC integration test passed separately. All 29 exported rows were
independently recomputed from confusion counts, original numerical aggregation
was checked, timing/byte sums were verified, and all 12 image files/hashes were
checked. Six final PNG figures were visually inspected; SVGs parsed successfully.
No browser/mobile or formal accessibility certification is claimed. Audit source:
ignored `data/verify_evaluation_reporting_v4.py`.

## Console monitoring added 23 October 2026

The current neural pipeline prints full global validation metrics after each
round, with original-training metrics when enabled. Both classes' precision,
recall and F1, macro-F1, balanced accuracy, accuracy, loss, confusion matrices,
round time and serialized application payload size use the already measured
round report. Final test metrics appear only when enabled, once at the fixed
final round. This adds no score-based selection or extra holdout evaluations.

Batch progress runs locally at a configurable interval (default 30 seconds after
completed batches), reporting running loss, throughput and phase ETA. That loss
describes the phase being executed, not a fresh global evaluation. Embedding
extraction has no loss. Existing round JSON, dashboard refresh and Matplotlib
exports are unchanged. See `RUN_EXPERIMENT.md` for the active config field and
the distinction between an already running process and a newly launched run.

# Adult detection diagnosis and correction — 26 October 2026

The reported IID, URL-only Character CNN undersampling run has many false Adult
alarms. The supplied round-6 validation report contains 3,073 true positives,
36,015 false positives, 1,743 false negatives and 153,809 true negatives. Adult
precision is therefore 7.86%, recall 63.81% and F1 14.00%. These observations
describe that run; they do not establish the best configuration. Validation/test
results must not be used to choose replacement settings.

Equal class counts in the optimization view do not imply equal precision or
recall on the original population. Undersampling changes the optimization class
prior to 50% Adult, while the original evaluation population remains strongly
Non-adult. A fixed 0.5 probability threshold need not be an appropriate operating
point after that change. This is a hypothesis about the observed false alarms,
not proof that the scores are calibrated or that a different threshold will
improve final-test quality.

## What the read-only audit checked

Label mapping remains 0=Non-adult and 1=Adult. The actual Client 1 round-1
undersampling seed produced 4,564 Adult and 4,564 Non-adult examples, retained all
eligible Adult records, and sampled without replacement. The source partition
and original training/validation/test files were not rewritten.

Encoding checks used only Client 1's ORIGINAL OPTIMIZATION records, after
excluding 24,682 training-selection records. No raw validation/test records were
inspected and no URLs are reproduced here.

| Optimization-only check | Adult | Non-adult |
|---|---:|---:|
| Records | 4,564 | 181,920 |
| Distinct hostnames | 3,417 | 129,167 |
| Median URL length, characters | 34 | 32 |
| Maximum URL length, characters | 97 | 587 |
| URLs truncated at 256 characters | 0 | 2 |
| URLs containing unknown characters in the encoded prefix | 0 | 15 |
| URLs with an empty/root path | 2,325 | 104,310 |

There were no duplicate 256-character URL prefixes or conflicting-label prefixes
in this optimization pool. Truncation and unknown characters cannot explain the
Adult problem in this inspected client. Many inputs contain only a hostname and
root path, which limits URL-only semantic information; it does not prove that
website enrichment will improve performance.

## Versioned CNN initialization

The legacy CNN applies Kaiming `fan_out` initialization to every convolution and
linear layer, including the final 1024-to-2 classifier. Its output-weight standard
deviation is about 1.0. Across three seeded synthetic batches, the initial
cross-entropy was 3.88-6.91 and mean confidence 92.9%-96.5%. Resetting only the
output weights to standard deviation 0.01 gave cross-entropy 0.677-0.693 and mean
confidence 52.6%-52.7%. The hidden weights were identical in each pair. These
checks establish saturated random startup without using corpus holdouts.

The corrected active experiment uses:

```json
"cnn_initialization": "stable_classifier_v1"
```

The corresponding low-level CNN model setting is
`"initialization": "stable_classifier_v1"`. Only the final classification layer
changes to zero-mean normal weights with standard deviation 0.01 and zero bias.
Hidden-layer initialization, all eight convolution blocks, pooling, residual
connections, dropout and head dimensions retain the supplied architecture.
Missing initialization settings mean `legacy_fan_out` for compatibility.
Initialization is part of the experiment contract; do not compare policies as if
they were identical runs or silently relabel old checkpoints.

## Training-only operating-point selection

The optional policy is:

```json
"threshold_selection": {
  "source": "training_selection",
  "metric": "adult_f1",
  "bins": 201,
  "tie_break": "highest_threshold"
}
```

It requires the existing `model_selection` training-domain reservation. Clients
return 2-by-201 class-conditional score-bin counts for selection, validation and
optional original-training/test evaluation. Raw scores, individual predictions,
URLs and record identifiers remain local. Coarse histograms are additional
statistics visible to the server; they do not provide differential privacy,
secure aggregation or a formal privacy guarantee.

The server chooses a grid threshold using pooled TRAINING-SELECTION Adult F1
only, separately for each evaluated round. Equal F1 values choose the higher
threshold. It reports the original fixed-0.5 reference alongside metrics at the
selected threshold. Validation/test histograms only measure the chosen operating
point; they never select or change it. This is a finite-grid operating point,
not an exact score-level optimum or a calibration guarantee.
Grid comparisons use float64 Adult probabilities to avoid scalar rounding at
threshold boundaries. Inference should load the saved checkpoint threshold and
compare `adult_probability.double() >= saved_threshold` using the same rule.

Checkpoint selection still uses pooled training-selection Adult F1, with the
earliest round winning ties. Each checkpoint records its threshold. Optional
final test freezes BOTH the selected checkpoint and its threshold; final-round
weights do not substitute for the selected checkpoint. Old configurations
without this option retain the fixed-threshold policy.

## Reproduction and limits

The active experiment enables the stable initialization and training-only
threshold policy. A fresh run is required. An already running process and its
historical checkpoints do not acquire these changes.

Keep initialization, threshold-selection policy, selection membership, model,
optimization budget, balancing ratio and seeds consistent in primary method
comparisons. Preserve the four balancing methods and original class counts.
Do not tune settings from validation/test or promise a higher Adult F1 before a
new training-only-selected experiment and its reserved final test are measured.

The changes address concrete startup behavior and an explicit operating-point
policy. Remaining limitations include few unique Adult examples, domain-disjoint
generalization, reduced majority variety under undersampling and ambiguous
URL-only inputs. They are not solved automatically by balancing.

## Verification evidence

The synthetic initialization evidence is in `results/diagnostics/cnn-init-audit/measure.json`.
A bounded real-training diagnostic is in
`results/diagnostics/cnn-training-audit-v1/measurements.json`, with its reproducible script.
It used only Client 1's original training file: a fixed 512-example balanced
optimization sample and a uniform 2,048-example training-selection sample. It
never read validation/test files. With one epoch/eight SGD batches, initial
training loss changed from 5.0366 (legacy) to 0.6917 (stable). Both resulting
selection F1 scores remained weak (below 0.03 after finite-grid threshold
selection). This confirms the startup correction, not successful Adult detection
or superiority on a completed five-client experiment. No diagnostic hyperparameter
search or full-corpus retraining was performed.

Regression verification: 404 tests passed and 3 were skipped in the full suite.
After final boundary/report-integrity repairs, 14 focused checkpoint/report tests
and the client probability-boundary regression passed. The actual Character CNN
two-round/five-client synthetic Flower smoke completed, reloaded its selected
checkpoint/threshold for final testing, and exported 15 PNG/SVG/PDF figure files.
The smoke results establish implementation behavior only. Configuration preflight
for the existing real IID client data passed without starting full training.

The subsequent weak round-1 discrimination was investigated separately in
[CNN_LEARNING_DIAGNOSIS.md](CNN_LEARNING_DIAGNOSIS.md). Matched training-only
optimizer pilots showed that AdamW learned more effectively than the current
SGD setting on two clients. The active experiment now declares that optimizer;
the original eight-batch initialization check above remains historical evidence.

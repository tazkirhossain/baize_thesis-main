# Evaluation and comparison results

No method is declared best. Validation and final test measurements are reported separately; single-seed results and convergence warnings limit conclusions.

Numerical rows pool predictions from independent local numerical models. Flower rows describe a shared federated model. Synthetic smoke results are not real-data evidence.

Recorded numerical fit warnings: **0**. Counts are retained per method/scenario in report.json; warnings are not hidden or used to retune against holdouts.

## All requested classification metrics

| Family / data | Partition | Method | Split / round | Adult P | Adult R | Adult F1 | Non-adult P | Non-adult R | Non-adult F1 | Macro-F1 | Balanced accuracy | Accuracy | N |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| character_cnn / synthetic | unknown | undersampling | train / 1 | 0.3390 | 1.0000 | 0.5063 | 1.0000 | 0.0250 | 0.0488 | 0.2776 | 0.5125 | 0.3500 | 60 |
| character_cnn / synthetic | unknown | undersampling | val / 1 | 0.5000 | 0.9000 | 0.6429 | 0.5000 | 0.1000 | 0.1667 | 0.4048 | 0.5000 | 0.5000 | 20 |
| character_cnn / synthetic | unknown | undersampling | train / 2 | 0.0000 | 0.0000 | 0.0000 | 0.6667 | 1.0000 | 0.8000 | 0.4000 | 0.5000 | 0.6667 | 60 |
| character_cnn / synthetic | unknown | undersampling | val / 2 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 1.0000 | 0.6667 | 0.3333 | 0.5000 | 0.5000 | 20 |

Confusion matrices, class supports, prevalence, source hashes, exact configuration, timing and communication measurements are retained in report.json. Undefined class ratios use zero; macro-F1 and balanced recall use the fixed two-class label set.

## Confusion matrices

True rows and predicted columns use [Non-adult, Adult].

| Family / data | Partition | Method | Split / round | TN | FP | FN | TP |
|---|---|---|---|---:|---:|---:|---:|
| character_cnn / synthetic | unknown | undersampling | train / 1 | 1 | 39 | 0 | 20 |
| character_cnn / synthetic | unknown | undersampling | val / 1 | 1 | 9 | 1 | 9 |
| character_cnn / synthetic | unknown | undersampling | train / 2 | 40 | 0 | 20 | 0 |
| character_cnn / synthetic | unknown | undersampling | val / 2 | 10 | 0 | 10 | 0 |

## Timing and communication

One row per experiment/round, not repeated for each metric split. Missing measurements are not zeros.

| Family / data | Partition | Method | Round | Sum client training seconds | Total local elapsed seconds | Round wall seconds | Protobuf payload MiB |
|---|---|---|---|---:|---:|---:|---:|
| character_cnn / synthetic | unknown | undersampling | 1 | 6.649 | not recorded | 8.809 | 178.599 |
| character_cnn / synthetic | unknown | undersampling | 2 | 0.593 | not recorded | 3.066 | 178.599 |

## Figures

![05 convergence faa8ac946dec02b5](05_convergence_faa8ac946dec02b5.png)

![06 costs faa8ac946dec02b5](06_costs_faa8ac946dec02b5.png)

## Measurement limits

- Existing real-data results are single-seed numerical experiments, not CNN/DeBERTa FL performance.
- Neural real-data comparisons remain not measured until those experiments run; synthetic smoke is never substituted.
- Convergence warnings are preserved. Single-seed measured validation/test results do not establish a robust best method.
- Historical elapsed times include training/evaluation/audits; separate training time is null where it was not recorded.
- Local numerical runs have no federated communication measurement. Historical missing timing/bytes remain null, not zero.
- Flower byte counts cover serialized application protobuf payloads, not gRPC/TLS framing, control messages or retries.
- Global train metrics use original local training splits, evaluated after aggregation; optimization loss uses balanced views and is distinct.
- Per-class undefined divisions use zero; macro-F1/balanced recall average the fixed [Non-adult, Adult] label set.
- Data minimization is not differential privacy or secure aggregation; only counts and model states cross the FL API.
- A declared cross-partition population label is not proof of identical membership; retain the preprocessing leakage/coverage audit.

## Real-data experiment coverage

| Family | Partition | Method | Validation | Final test | Status |
|---|---|---|---|---|---|
| engineered_numeric | iid | undersampling | not measured | not measured | not measured |
| engineered_numeric | iid | oversampling | not measured | not measured | not measured |
| engineered_numeric | iid | smote | not measured | not measured | not measured |
| engineered_numeric | iid | ensemble_undersampling | not measured | not measured | not measured |
| engineered_numeric | label_skew | undersampling | not measured | not measured | not measured |
| engineered_numeric | label_skew | oversampling | not measured | not measured | not measured |
| engineered_numeric | label_skew | smote | not measured | not measured | not measured |
| engineered_numeric | label_skew | ensemble_undersampling | not measured | not measured | not measured |
| engineered_numeric | category_skew | undersampling | not measured | not measured | not measured |
| engineered_numeric | category_skew | oversampling | not measured | not measured | not measured |
| engineered_numeric | category_skew | smote | not measured | not measured | not measured |
| engineered_numeric | category_skew | ensemble_undersampling | not measured | not measured | not measured |
| character_cnn | iid | undersampling | not measured | not measured | not measured |
| character_cnn | iid | oversampling | not measured | not measured | not measured |
| character_cnn | iid | smote | not measured | not measured | not applicable in this implementation |
| character_cnn | iid | ensemble_undersampling | not measured | not measured | not measured |
| character_cnn | label_skew | undersampling | not measured | not measured | not measured |
| character_cnn | label_skew | oversampling | not measured | not measured | not measured |
| character_cnn | label_skew | smote | not measured | not measured | not applicable in this implementation |
| character_cnn | label_skew | ensemble_undersampling | not measured | not measured | not measured |
| character_cnn | category_skew | undersampling | not measured | not measured | not measured |
| character_cnn | category_skew | oversampling | not measured | not measured | not measured |
| character_cnn | category_skew | smote | not measured | not measured | not applicable in this implementation |
| character_cnn | category_skew | ensemble_undersampling | not measured | not measured | not measured |
| deberta_v3_base | iid | undersampling | not measured | not measured | not measured |
| deberta_v3_base | iid | oversampling | not measured | not measured | not measured |
| deberta_v3_base | iid | smote | not measured | not measured | not applicable in this implementation |
| deberta_v3_base | iid | ensemble_undersampling | not measured | not measured | not applicable in this implementation |
| deberta_v3_base | label_skew | undersampling | not measured | not measured | not measured |
| deberta_v3_base | label_skew | oversampling | not measured | not measured | not measured |
| deberta_v3_base | label_skew | smote | not measured | not measured | not applicable in this implementation |
| deberta_v3_base | label_skew | ensemble_undersampling | not measured | not measured | not applicable in this implementation |
| deberta_v3_base | category_skew | undersampling | not measured | not measured | not measured |
| deberta_v3_base | category_skew | oversampling | not measured | not measured | not measured |
| deberta_v3_base | category_skew | smote | not measured | not measured | not applicable in this implementation |
| deberta_v3_base | category_skew | ensemble_undersampling | not measured | not measured | not applicable in this implementation |

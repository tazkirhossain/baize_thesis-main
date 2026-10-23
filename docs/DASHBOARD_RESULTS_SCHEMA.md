# Local dashboard results interface v1

`GET /api/experiments` returns the read-only snapshot produced by
`src/evaluation/dashboard.py`. `GET /api/health` checks server availability.
There are no write, file-upload, raw-data, model-inference or training endpoints.
Only built assets under `dashboard/dist/` are served as files. Unknown API routes
and traversal outside that directory are rejected. The server binds localhost.

Top-level fields:

| Field | Meaning |
|---|---|
| `schema_version` | Integer 1 |
| `generated_at` | UTC ISO timestamp of snapshot extraction |
| `runs` | Independent experiment histories; never pooled across runs |
| `dataset` | Optional audited real IID corpus/allocation counts, separate from scores |
| `issues` | Excluded unreadable/inconsistent aggregate sources and bounded reasons |
| `training_templates` | Public checked-in CNN/DeBERTa base/embedding configs for draft downloads |
| `capabilities` | Read-only=true, live_telemetry=false, raw_records=false |

Each run has a stable `experiment_id` derived from its local run-folder name,
`name`, `model_name` (`character_cnn` / `deberta_v3_base`), `partition_type`
(`iid`, `label_skew`, `category_skew`, `unknown`), `imbalance_method` (`none`,
`undersampling`, `oversampling`, `smote`, `ensemble_undersampling`),
`data_kind` (`real`, `synthetic`, `unknown`), `training_protocol`
(`end_to_end`, `encoder_embeddings`), `status` (`completed`, `incomplete`),
`configuration`, `rounds`, `duration_seconds`, `sources`, `updated_at`,
`evaluation_protocol` and `evaluation_population_id`.

Client-reported data kind and partition take precedence over declared public
metadata; disagreement between clients is rejected. Missing historical metadata
remains unknown. A completed history must have PASS and all configured rounds;
partial histories say incomplete, because saved files cannot establish liveness.

Each round contains:

```json
{
  "round_number": 1,
  "metrics": {"train": "optional metric object", "val": "optional metric object", "test": "optional metric object"},
  "optimization_loss": null,
  "timing": {"client_training_seconds_sum": null, "round_wall_seconds": null},
  "communication": {"total_bytes": null},
  "clients": []
}
```

This fragment documents types; it is not observation data. The actual metric
object has Adult and Non-adult precision, recall and F1; macro-F1, balanced
accuracy, accuracy, optional loss, n_records, both class counts, and a 2x2
confusion matrix. Rows are true classes and columns predicted classes, both in
`[Non-adult, Adult]` order: `[[TN, FP], [FN, TP]]`. Metrics are independently
recomputed by the existing shared calculator and contradictory values rejected.
Undefined divisions use the pipeline's documented zero policy. Missing metrics
remain absent/null and are rendered “Not measured,” not as measured zeros.

Test data may appear only at the fixed final round, and only when the saved
configuration enabled it. Global test accuracy never falls back to validation.
Training metrics evaluate the updated global model on original local training
data. Optimization loss is separate and may cover resampled data or encoder
warmup/head phases; it is not substituted for original-training evaluation loss.

Each client entry uses display ID 1–5 and contains allowlisted coarse statistics:
original/resampled training count, original/used class counts, duplicated and
synthetic examples, ensemble members, examples processed, optional warmup count,
loss, update/training time, aggregation weight and SMOTE neighbors. Status is
`Aggregated` only for an existing saved aggregate. Unsampled clients have no
entry, so the UI says no recorded update. Configured local epochs are displayed
for saved successful updates; no live epoch counter is claimed.

`comparison_group` hashes model, data kind, representation, seed, balancing seed,
learning rate, rounds, batch, local/warmup epochs, sampling participation, ratio,
weighting, threshold and operator-declared evaluation population. It deliberately
excludes the method and partition being compared. Same-partition input equality
also requires `partition_data_signature`, a hash of client-reported split
contracts. Different partitions depend on the previously audited shared domain
roles and the declared population identity; metadata is not cryptographic proof
of equal population. Incompatible overlaid curves carry a visible caveat and
are never silently treated as a matched scientific comparison.

Provenance includes logical aggregate filenames and SHA-256 values. No raw
URLs, record IDs, source rows, tokens, individual predictions, model tensors,
private paths, environment variables or credentials are returned. Extraction
does not recurse into snapshots, archives or client data folders. Counts come
only from audited manifests. Source files and artifacts are never rewritten.

CSV uses one row per matching run/round for the selected split and range, with
the configuration, metrics, confusion matrix, durations and payload count.
It preserves real/synthetic/protocol labels and empty missing values. A full
field list lives in `dashboard/src/data.ts`; nested client statistics stay in
the JSON API. CSV export quotes values and escapes spreadsheet formula prefixes.

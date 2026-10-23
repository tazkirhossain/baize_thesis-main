# Federated URL experiment dashboard

Local React 19 / TypeScript / Vite dashboard with Recharts. The read-only Python
adapter observes the existing Flower outputs; the ML pipeline is unchanged.
No dataset, checkpoint, raw URL, prediction example or credential is bundled or
served. All frontend dependencies are pinned in `package-lock.json`.

## Launch on this Windows machine

The dashboard is already built in this workspace. Open PowerShell:

```powershell
Set-Location 'D:\baize_thesis-main\baize_thesis-main'
$env:PYTHONPATH = "$PWD\data\flower-runtime;$PWD\data\neural-runtime"
& 'C:\Users\hossa\anaconda3\python.exe' scripts/dashboard.py
```

Open **http://127.0.0.1:8765**. Stop the server with Ctrl+C. It binds loopback
only and has no write or training-start endpoint. No separate Node process is
needed to serve the production build. NumPy and the project's metric helpers
are required by the results adapter; the existing Python environment has them.

After cloning or changing frontend source, build once before starting Python:

```powershell
Set-Location dashboard
npm.cmd ci --cache ../data/npm-cache
npm.cmd run build
Set-Location ..
```

For frontend development, keep the Python server running and run `npm.cmd run
dev` from `dashboard/` in a second terminal. Open http://127.0.0.1:5173. Vite
proxies `/api` to the same local Python server. No external analytics, fonts,
image service or data upload is used.

## Connect experiment results

Keep the existing Flower layout:

```text
data/YOUR_RUN/
  experiment.json
  server/
    run.json
    round_0001.json
    round_0002.json
    ...
```

Refresh results in the dashboard or enable the 15-second file-refresh checkbox.
The API caches snapshots for at most five seconds. It scans only immediate
`data/*/server/` folders, excluding nested code snapshots and recovery archives.
Completed runs require PASS, contiguous rounds and valid confusion-derived
metrics. Unsupported or invalid logs are listed visibly as source issues.
Partial logs can be shown as **Partial**, never implicitly labelled actively
training. A finalized round can appear before validation is saved; absent
metrics stay unavailable. Exact file SHA-256 values appear in source details.

The current useful preview uses your actual
`cnn-over-smoke-20261023-163747` run. It is labelled **Synthetic smoke** and its
two rounds/batch 8 are retained. No fictional metrics were added. IID/non-IID and
real-model coverage remain unmeasured until those experiments finish. Audited
real corpus/allocation counts are a separate source, not results of the smoke.

## Controls and semantics

- Model, partition, balancing, real/synthetic and protocol filters intersect.
  All removes a restriction; Reset restores all filters and clears comparisons.
- The selected experiment drives KPIs, diagnostics, counts and client details.
  The round range drives per-round plots, KPIs and exported rows. History uses
  the latest measured round within that range; its time column covers the whole
  saved run. Missing observations remain null, never substituted with zero.
- Validation is the initial split. Global **test** accuracy is shown only for a
  saved test checkpoint. Choosing test without a test measurement shows empty
  metrics/charts. Tests are permitted only at a declared final checkpoint.
- Check runs in Experiment history, then View selected curves / Overview to
  compare them alongside the currently selected experiment. Different
  models/data/representation/budgets/cohorts/seeds are
  flagged as descriptive comparisons. Same-partition runs also need identical
  recorded client split contracts. No ranking automatically selects a winner.
- Client nodes drill down into original counts, resampled counts, processed
  examples, members, configured completed epochs and local update timings.
  The diagram is a **recorded snapshot**, not live client/server telemetry.
- Export CSV contains all filtered runs with measurements in the current split
  and round range, also respecting the history name search. That search narrows
  history/export only and does not change the selected run or its charts.
  Counts/metrics use the same reviewed source; nulls are empty,
  and cells are quoted with spreadsheet formula protection. Raw URLs are absent.
- Plan experiment starts a **draft** at batch 64, local epochs 1, rounds 10.
  Its working controls download a complete JSON configuration for the existing
  runner. Save it under `config/federated/`, then use the displayed command with
  a new `data/` output. The planner does not modify source configs or start
  training. DeBERTa defaults to LR 0.00002; CNN to 0.001. These are not tuned.

SMOTE uses the project's explicit `encoder_embeddings` protocol, never raw
URLs or token IDs. It has an extra encoder-warmup stage plus frozen-head epochs;
that budget is shown separately. Use matching protocols for fair four-method
comparisons. The ensemble counts are per member; processed examples include
member/epoch repetitions. Original-training-count FedAvg remains the primary
policy. Validation and test data are not resampled.

Recorded round time sums server-observed round wall times; it is not full
process startup time. Local training time is a separate sum of client work.
Communication is exact serialized application payload sizes, excluding network
framing, retries and registration. Precision–recall curves and error URLs are
explicitly unavailable because the logs do not contain individual predictions.

## Data interface and maintenance

Read [the schema and trust boundaries](../docs/DASHBOARD_RESULTS_SCHEMA.md).
`src/evaluation/dashboard.py` owns extraction and sanitization;
`scripts/dashboard.py` owns HTTP and static serving;
`dashboard/src/data.ts` owns frontend transformations/export;
`components.tsx` owns reusable panels and charts; `App.tsx` composes views.

```powershell
# Project root, with the documented Python runtime:
python -m pytest tests/test_dashboard.py tests/test_evaluation_reporting.py -q
# Dashboard directory:
npm.cmd test
npm.cmd run build
```

Python remains the authoritative metrics calculator. Recharts/React render
saved measurements; they never train, select thresholds or infer missing
prediction counts. Chart APIs follow the official
[Recharts guide](https://recharts.github.io/en-US/guide/sizes/) and
[Vite server configuration](https://vite.dev/config/server-options).

## Verified in this workspace

- Production TypeScript/Vite build passed; chart code is a separate local chunk.
- 181 Python tests passed; three optional network tests skipped. The final
  strict configuration/representation guard also passed all six adapter tests.
- Three frontend tests passed for range/missing-test semantics, CSV scope and
  formula escaping, and comparison-group/input guards.
- Browser checks covered filters, missing metrics, metric/range changes,
  comparison warnings, client drill-down, history search and sorting, CSV
  download, default 64/1/10 config planning, valid SMOTE config download, keyboard
  focus trapping/Escape, responsive layouts and a clean browser error console.
- All nine HTTP checks passed, including raw-path/traversal rejection and no
  POST endpoint. Runtime dependency audit reported zero known vulnerabilities.
- Seven valid saved runs are displayed; one older incompatible run is listed
  as an issue. These are synthetic/unknown checks, not measured real CNN or
  DeBERTa quality. No live telemetry or individual prediction data was added.

Private verification artifacts and screenshots are under ignored `data/`.

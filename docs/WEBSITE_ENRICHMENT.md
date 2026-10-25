# Frozen client-local website enrichment

Implemented 25 October 2026. This optional input mode supports Character CNN and
DeBERTa-v3-base, IID/label-skew/category-skew and all four balancing methods. It
does not replace the audited source, repartition records, alter labels, or change
train/selection/validation/test membership. Extra inputs do not guarantee improved
Adult precision, recall or F1; that requires measured enriched experiments.

## Configuration and collection

The original `config/experiment.json` keeps `"website_enrichment": null` for
URL-only reproduction. The example `config/experiment_website.json` selects the
same IID/oversampling/CNN combination with 512 input characters and:

```json
"website_enrichment": {
  "mode": "url_plus_metadata",
  "snapshot_id": "website-v1",
  "collection_config": "config/website_collection.json"
}
```

You can put this object into the main experiment file instead. Model, partition,
balancing and training settings still live in that experiment file. Shared
network/extraction limits live in `config/website_collection.json`; exact values
are expanded into the saved public experiment contract. Changing collection
settings requires a **new snapshot_id**. Enriched and URL-only models are distinct
input experiments, never interchangeable checkpoints.

From the actual repository root in the project's working Python environment:

```powershell
# Optional end-to-end check: actual CNN, 5 synthetic clients, 2 rounds, no network.
python scripts/run_experiment.py --config config/experiment_website.json --mode smoke

# Optional pilot: at most 100 NEW original records. This leaves a partial cache.
python scripts/enrich_client.py --config config/experiment_website.json --client-dir data/client-iid-v1/client_000 --limit 100

# Resume and complete all original records for Client 1.
python scripts/enrich_client.py --config config/experiment_website.json --client-dir data/client-iid-v1/client_000
```

Each device runs separately using its OWN directory. In the existing one-machine
simulation, this loop invokes the same local collector sequentially:

```powershell
0..4 | ForEach-Object {
    $clientDirectory = 'data/client-iid-v1/client_{0:D3}' -f $_
    python scripts/enrich_client.py --config config/experiment_website.json --client-dir $clientDirectory
    if ($LASTEXITCODE -ne 0) { throw "Collection failed: $clientDirectory" }
}

python scripts/run_experiment.py --config config/experiment_website.json --check
python scripts/run_experiment.py --config config/experiment_website.json
```

If `$pythonExe` points to your working interpreter, replace `python` with
`& $pythonExe`. Existing Windows bundled runtimes require matching Python 3.13;
CUDA requires CUDA-enabled PyTorch. A running URL-only process cannot reload
the new code or inputs. Choose CPU/CUDA as usual in the new experiment file.

For non-IID, select `label_skew` or `category_skew` and collect each directory
under `data/client-label-skew-v1/` or `data/client-category-skew-v1/` respectively.

## Collected signals and network policy

1. Origin `/robots.txt`: access rules, advertised sitemaps and crawl pacing.
2. Bounded advertised sitemaps and `/sitemap.xml` fallback, including bounded
   indexes and `.gz` files. Same-registrable-domain URL paths become context;
   discovered pages are **not** added as new labelled examples.
3. The retained minimized page URL: HTML title, description/Open Graph/Twitter
   fallbacks, keywords, rating, language and a bounded non-script HTML-text prefix.
   Scripts, style, templates, noscript and SVG content are excluded. No JavaScript
   is executed and login/browser-only content is not bypassed.

Robots rules control crawling; they are not Adult labels. Sitemaps supply page
locations, not labels. No target/category/model score is consulted during
collection. Earlier preprocessing removed queries/credentials/ports; the
collector cannot reconstruct them and visits the retained minimized URL.

Only public HTTP(S), default ports and verified TLS are used. DNS answers are
validated and the connection is pinned to the validated address. Redirects are
bounded, restricted to the original registrable domain and checked against the
destination's robots policy. There are no cookies, authentication, environment
proxies, file URLs or private network requests. Responses, sitemap decompression,
files/entries and extracted text are capped. Raw HTML and images are not stored.
The socket timeout is bounded; OS DNS resolution may take additional time.

Per-origin spacing and advertised crawl delay/request rate are applied. Excessive
delay records a failure instead of ignoring the rule. Robots 4xx permits access
except 429; unreachable/error/oversized robots blocks collection conservatively.
Equally specific robots groups merge; longest rules match and Allow wins ties.
Robots caches used for new requests expire after 24 hours.

Sources: [RFC 9309](https://www.rfc-editor.org/rfc/rfc9309.html),
[Sitemap protocol](https://www.sitemaps.org/protocol.html).

## Frozen local storage and audit

```text
client_000/enrichment/website-v1/
  features.sqlite3     Extracted features, statuses, resource cache and provenance
  manifest.json        Counts, missing fields, original split/config/DB hashes
  .collection.lock     OS-released exclusive writer lock
```

Records are checked against original source-row, split and URL fingerprints.
Collection streams local files and commits each completed record. Interrupting
preserves committed records; rerunning skips them. A concurrent writer to the
same client snapshot is rejected. Completed snapshots are frozen: rerunning the
CLI verifies and skips them without network access. Refresh under a new ID.

Failed, blocked, expired, non-HTML and oversized pages remain original examples
with missing feature strings and explicit reasons. There is no silent filtering,
relabeling, replacement, retry loop or evaluation resampling. Progress and final
counts/missingness are printed; resource status/content hash/timestamps remain
local. The manifest requires zero excluded original records and full train/val/test
coverage before freezing. A pilot/partial cache cannot train. Collecting holdout
inputs using a fixed rule does not fit transforms or tune on holdout labels.

Training is offline. The reader verifies the DB fingerprint/source contract and
checks split/URL on each record lookup. File changes during a run fail closed.
Missing enrichment never silently changes an enriched run to URL-only.

## Model inputs and leakage prevention

Five separately budgeted sections are encoded: URL, title, metadata, HTML text
and site context. Proportions are 25/15/25/20/15 percent after section markers for
CNN characters, and after special tokens for DeBERTa tokens. The public CNN
alphabet adds space/newline; other Unicode uses UNK. DeBERTa retains its pinned
tokenizer and Unicode support. No vocabulary/scaler/budget is fitted on holdouts.
Enriched minimum lengths are 64 characters / 32 tokens; DeBERTa max remains 512.
The CNN's supplied architecture is retained with a declared new input vocabulary.

The shared domain roles and training-only selection policy remain unchanged.
Cross-domain fetched content is excluded, avoiding a resource bridge across those
roles. Website metadata can be predictive; dataset labels/categories/IDs are never
model inputs. Live content may have changed since original labelling; retain
collection dates and audit that drift rather than silently changing targets.

Balancing still affects optimization data ONLY. Oversampling reuses cached
features without duplicate requests. SMOTE extracts enriched continuous encoder
embeddings after original-training warmup; strings/tokens are not interpolated.
Selection/validation/test features are used only for their evaluation passes.

Only model tensors, agreed coarse metrics and snapshot hashes/counts leave a
client. The server never opens website feature DBs. Allocation-independent pooled
feature digests and collection contracts protect report/dashboard comparison
cohorts. Independently crawling different live content under one snapshot name
does not prove identical inputs. Preserve the frozen features for matched studies;
use the offline owner-operated reuse tool for matched scenario inputs:

```powershell
# First finish and freeze ALL five source IID client snapshots.
# Copy only records assigned to this target client; no network or labels used.
python scripts/reuse_enrichment.py --config config/experiment_website.json --source-root data/client-iid-v1 --target-client-dir data/client-label-skew-v1/client_000
```

Repeat for target Clients 0–4 and category-skew directories. This is explicit
research provisioning by the dataset owner, before client deployment; the Flower
server and runtime clients never open other clients' caches. Reuse checks source
snapshot hashes, original target IDs/URLs and protected split membership. It
copies no unassigned examples and rejects conflicting existing features. Global
feature fingerprints are preserved across changed client allocations. A failed
or interrupted copy remains partial/resumable. Completed destinations are verified
and skipped. Runtime clients subsequently read only their own prepared directory.

## Resource limits and verification scope

Start with the bounded training-only pilot. Full 1.46M-URL collection is potentially
slow and network/storage intensive, with missing/changed sites. Pilot coverage is
not a dataset-wide result. Collection is sequential per client; origin pacing is
local to each collector, so coordinate aggregate traffic across devices sharing
origins. Full pretrained DeBERTa memory/speed tests, full-corpus enriched accuracy
and LLM label/model-quality review are separate work. Select models only using
the existing training-selection policy; preserve final test independence.

## Verification recorded in this workspace

The full regression log is `data/website-all-tests-v3.log`: **357 passed, 3 optional
network tests skipped**. Two additional offline-reuse checks passed in
`data/website-reuse-tests-v2.log`. New tests exercise
public-address pinning, TLS hostname retention, response/XML/gzip limits, robots
rules, redirects, resumable collection, failed-page retention, exclusive writers,
snapshot/source corruption checks, label independence and field/token budgets.
All 24 family/method/scenario configurations run two Flower rounds using tiny
explicit synthetic model/data fixtures, including selection and final-test routing.
The DeBERTa fixtures check the real classifier implementation with a tiny encoder;
they are not pretrained DeBERTa quality measurements.

`data/website-cnn-smoke-v1/` separately ran the ACTUAL supplied CNN architecture
for two rounds on five synthetic clients and frozen synthetic website inputs.
It generated two round checkpoints, best/final weights and six PNG/SVG files.
These toy results establish execution/integrity only, not useful model quality.

The public `example.org` probe extracted its title and text after robots returned
404. The real Client 1 pilot collected five optimization/train records: one page
yielded metadata, two were robots-disallowed and two had unreachable robots. All
five remain in a PARTIAL, resumable `website-v1` cache. This small result does not
estimate full-corpus availability. Full collection has NOT run on all five clients.

`data/website-enrichment-verification-v1.json` independently verifies checkpoint
hashes, report artifacts and all **90** original dataset/preprocessing/split hashes
across the three five-client scenarios. Each scenario retains exactly 1,468,243
original records: 34,811 Adult and 1,433,432 Non-adult.

# Reproducible data preparation and leakage prevention

The implementation is now `src/data/preprocessing.py`; required helpers were
migrated from the earlier package without changing the established record rules.
The neural Flower path is documented separately in `FEDERATED_TRAINING.md`.

## Source validation and audit

The CSV contract is exactly `id,url,label`. Use `--header absent` for the uploaded
headerless file, `present` for named columns in any order, or `auto` for header
detection. Named headers with missing, duplicate, unknown or extra columns fail.
Every source record must have a nonblank ID, valid URL and recognized category.
Missing IDs/URLs/labels, malformed rows, invalid URLs and unknown labels are
excluded with explicit reasons, never guessed or silently imputed.

`src/data/schema.py` contains the 15 known categories and strict 0/1 validation.
Adult maps to 1; all 14 other known categories map to 0. Preserve the exact source
ID and original category alongside normalized category, target and `source_row`.
IDs need not be unique; the source row identifies an original record. Category,
ID and source-row fields are audit metadata, never model predictors.

`src/data/url.py` normalizes hosts with an offline public-suffix snapshot. Only
HTTP/HTTPS are accepted; bare hosts assume HTTP. Reject control characters, inputs
over 2048 characters, missing/unparseable hosts, invalid host characters and hosts
without a registrable domain. Strip userinfo, port, query and fragment even when
input is invalid. Do not fetch any URLs.

Canonical duplicate keys are scheme-insensitive host plus path, with trailing
slashes stripped. This documented policy can merge differently written URLs.
Keep the first source row for same-binary-label duplicates. Exclude **every**
occurrence of a canonical URL with conflicting binary labels. Retained records
and excluded source rows must account for the input exactly once.

Audit outputs under ignored local storage include:

- `report.json`: input/accepted/retained/excluded counts, missing counts, class
  and category counts, duplicate/conflict counts and fingerprints.
- `exclusions.jsonl`: one entry for every excluded source row, original ID and
  category, primary/all applicable reasons and duplicate-retained-row references.
- `quarantine.jsonl`: invalid/unlabelled/unknown/conflicting records for review;
  invalid raw URLs are omitted, safe minimized forms retained where available.
- Immutable client `data.jsonl`, manifests and shared preparation settings.
- `code_snapshot/` and `runtime-requirements.txt`: exact code/config hashes and
  actual installed dependency closure. A saved environment is not proof of a
  successful fresh-environment installation.

Missing-value and secondary-reason counts can overlap. Primary exclusion reasons
are mutually exclusive. See `DATASET.md` for the current audited totals.

## Preparation and client allocation are separate stages

```powershell
python scripts/prepare_clients.py --input data/raw/url.csv --output data/prepared-new --header absent --clients 5
python scripts/partition_clients.py --config config/five_client_iid.json --output data/iid-new
```

The first command validates/deduplicates and provisions a domain-exclusive source
allocation. The second reads an audited source named in its config and creates
the requested five-client strategy. It is the configured partitioner that defines
the IID/non-IID experiment. Use `five_client_label_skew.json` or
`five_client_category_skew.json` for the two separate non-IID scenarios.

The existing audited source is `data/client-datasets-v3/`; current five-client
scenario roots are `data/client-iid-v1/`, `data/client-label-skew-v1/` and
`data/client-category-skew-v1/`. Reuse them without overwriting. No resampling or
balancing happens during client ownership assignment. Each eligible original
record is assigned once without replacement; validate the complete union and
source fingerprints. See `PARTITION_SAMPLING.md` for shuffled-pool quotas.

## Client-local train/validation/test splitting

```powershell
python scripts/split_client.py --client-dir data/iid-new/client_000
```

Run separately for each client directory; no raw CSV, other client directory or
central validation set is needed. Verify the local input checksum first. Write
`splits/`, containing
`train.jsonl`, `val.jsonl`, `test.jsonl`, its manifest and preprocessing contract.
Existing splits are never overwritten. Insufficient class coverage fails.

The targets are 70%/15%/15%. For URL-level IID and both current non-IID scenarios,
a domain can be owned across clients but has one deterministic shared split role
everywhere (`src/data/split_policy.py`). Therefore train/evaluation domain leakage
is prevented across the entire federation. Exact row fractions are approximate
because domain groups cannot be divided across roles. Historical domain-exclusive
source allocations use their declared local splitting policy, not a retroactive
change to the new scenario policy.

## Encoding and scaling after splitting

`src/client/data.py` reads and verifies local splits and marks evaluation labels
read-only/protected. The neural trainer uses fixed public character encoding for
CNN or the pinned pretrained DeBERTa tokenizer. No vocabulary is learned from
validation or test. All clients share the model/encoding contract.

The optional inspection API `src.data.preprocessing.load_local_split()` retains
65,536 hashed character 3–5-gram features with per-row L2 normalization. This is
stateless and does not train either neural model. It is not a learned scaler.
The removed TF-IDF/linear-classifier branch is not available.

`src/data/numerical_features.py` retains engineered numerical features and a
StandardScaler fitted only on original local training records; validation/test
use frozen transforms. This supports the SMOTE utility, not a numerical model
runner. Never apply SMOTE to URLs, character IDs or tokenizer sequences.
Balancing changes only a client's training view; it never edits source files or
holdout distributions. See `CLASS_BALANCING.md`.

## Safety and remaining scope

New preparation outputs are atomic and refuse existing destinations. Original
records and old source snapshots stay immutable. Recheck stored checksums when
loading. `.gitignore` excludes all local data, JSONL exports, archives, model
states, credentials and private keys. The repository safety check and CI inspect
tracked files, including force-added ignored artifacts. Run the check locally
before uploading from a real Git checkout; CI after a push cannot prevent upload.
This exported workspace has no readable Git index or verified remote history.

Website collection is now a separate optional client-local stage AFTER these fixed
splits; it does not change this offline preparation or original labels. Read
WEBSITE_ENRICHMENT.md for robots/sitemap/metadata/text collection and frozen neural
inputs. LLM noise/label review remains unimplemented. Quarantined records cannot
be reintroduced without checking protected roles.
There is no validation/test-driven sampling, class weighting, threshold tuning,
checkpoint selection or gradient fitting in this preparation stage.

# Client-owned preprocessing and local splitting

The workflow has two explicit stages. Global offline provisioning cleans the
uploaded CSV and assigns unique datasets to clients. Each client subsequently
splits only its own dataset. Feature extraction and normalization happen after
that local split, when the client loads training, validation or test data.

## Stage 1: offline cleaning and client allocation

```powershell
python scripts/prepare_clients.py --clients 5
```

This creates `data/client-datasets-v3/`. Existing outputs are never overwritten;
use `--output data/another-allocation` for another experiment. Use the Python
environment containing the project's requirements.

Preparation performs strict CSV parsing, URL minimization, explicit binary label
mapping, global canonical deduplication and conflict removal. Missing URLs,
malformed records, invalid URLs, missing labels and unknown categories have
separate rejection counts. Unknown/missing labels are quarantined, never guessed.

The known `Adult` category maps to 1 and the other 14 known categories map to 0.
Same-label duplicates keep their first source row/category; all occurrences of
conflicting binary labels are excluded. The canonical policy is scheme-insensitive
and removes trailing slashes, an explicit modelling assumption. No websites are
fetched. Query strings, fragments, credentials and ports are not retained.

Whole registrable domains have exactly one client owner. The seeded allocator
targets similar positive/negative counts across clients. This provides unique
ownership and prevents another client's training domain from appearing in a
client's evaluation data. The provisioner does not create train/val/test splits
and does not extract or scale features.

All valid unique labelled records are retained. No class balancing, majority
undersampling, minority replication or LLM labelling is performed. Training-time
imbalance handling is a separate decision. Source rows, accepted rows, duplicates,
conflicts and rejections are accounted for before publishing the output atomically.
The original CSV is never changed.

```text
data/client-datasets-v3/
  preprocessing.json       shared settings and default 70/15/15 split policy
  report.json              global provisioning counts and source checksum
  quarantine.jsonl         excluded records for offline review
  client_000/
    data.jsonl             this client's entire unique, unsplit dataset
    manifest.json          dataset count, class/category counts and checksum
    preprocessing.json     local copy of the shared configuration
  client_001/ ...
```

Offline provisioning is simulation setup, not a training server role. Deploy each
client directory only to its owner. The root report and quarantine do not need to
be present on clients or the training server. Actual process/storage access
isolation is a deployment concern, not enforced by assigning files on one disk.

## Stage 2: each client splits its own data

Run this command on the client, pointing only to that client's directory:

```powershell
python scripts/split_client.py --client-dir data/client-datasets-v3/client_000
```

Each of the remaining clients runs the same command with its own local directory.
This command needs no original CSV, global report, root configuration or other
client's directory. It verifies its local input checksum, then assigns whole local
domains to training/validation/test, targeting 70/15/15. Both classes are required
in every split (default CLI provisioning policy: at least ten records per class).
Insufficient class coverage blocks publication; existing splits are not overwritten.

```text
client_000/
  data.jsonl
  manifest.json
  preprocessing.json
  splits/
    train.jsonl
    val.jsonl
    test.jsonl
    manifest.json          local split counts and checksums
    preprocessing.json     shared transformation contract
```

For the hypothetical 600,000-record, six-client experiment, the targets are
100,000 records per client and then 70,000 train / 15,000 validation / 15,000 test
on each client. Actual sizes may differ because domains cannot be divided across
owners or splits. Domain isolation takes priority over exact counts. Inspect the
reported counts rather than assuming exact proportions.

## Stage 3: client-side encoding and normalization

```python
from baize.data.preprocessing import load_local_split

client_dir = "data/client-datasets-v3/client_000"
X_train, y_train = load_local_split(client_dir, "train")
X_val, y_val = load_local_split(client_dir, "val")
X_test, y_test = load_local_split(client_dir, "test")
```

The loader verifies each split's checksum. Validation/test labels are read-only
and tagged against artificial noise injection. All clients use the same hashed
character 3-5-grams, dimensions, analyzer, lowercasing and per-record L2
normalization. There is no fitted vocabulary or corpus-wide scaling statistic.
Thus features are prepared after splitting without learning from evaluation data.
No StandardScaler or MinMaxScaler is needed for this URL-text feature branch.

If numeric metadata features are added later, fit any learned imputation/scaling
on training data only; transform validation/test with the frozen training-derived
transformation. Feature coordinates must remain compatible across FL clients.
Source category, CSV ID and source row number are audit fields, never predictors.
The category determines the label, and row ordering is correlated with category.

## Compatibility and scope

The older `data/clients/` and `data/client-datasets/` outputs are preserved as
historical allocations. They lack the complete v3 identity and exclusion audit. It is not the output of the current two-stage workflow.
`load_client_split(root, client_id, split)` supports both formats; for isolated
clients, use `load_local_split` directly.

The legacy training simulator and result-reproduction scripts have not been
converted to client-side evaluation. This change implements data provisioning,
local splitting and local feature loading, not a new FL training loop. Sitemap,
robots.txt, metadata collection, LLM labelling and numeric feature scaling remain
separate future additions. Quarantined records must never be added to training
without checking their domains against the protected evaluation splits.


## V3 schema, identity, exclusion audit and reproducibility

The CSV contract has exactly three required columns: `id`, `url`, `label`.
Use `--header absent` for a declared headerless file (positions id,url,label) or
`--header present` for a named header, whose order may vary. In named-header mode,
missing, duplicate, unknown or extra column names fail the entire preparation.
The default `--header auto` recognizes common header names; use an explicit mode
if the first record ID could be confused with a header name. Record IDs must be
nonblank; their exact original strings, including leading zeros, are preserved.
IDs are not assumed globally unique; `source_row` identifies each source record.

Every retained record now contains `record_id`, exact `original_category`,
normalized `category`, numeric `label` and explicit `target` (`Adult` or `Non-adult`).
Only URL text enters the feature extractor. Audit fields are never predictors.

Additional private output artifacts:

- `exclusions.jsonl`: one entry for EVERY excluded source row, with original ID,
  original category, source-row reference, primary reason and all applicable reasons.
  Same-label duplicates reference the retained row/ID. All conflicting occurrences,
  including the first occurrence, are individually listed. Invalid raw URLs are
  omitted; safe minimized URLs are included when available.
- `quarantine.jsonl`: invalid, unlabelled, unknown-category and conflicting records;
  ordinary same-label duplicates are in the exclusion ledger only.
- `report.json`: independent missing counts for all three columns; schema/header
  interpretation; raw original category counts BEFORE URL rejection; accepted
  category counts before deduplication; retained counts per client/category/class;
  exclusion counts/checksums and source-row accounting. Missing-value and secondary
  reason counts can overlap; primary exclusion reasons remain mutually exclusive.
- `code_snapshot/`: exact preparation code, configuration files and documentation,
  with per-file checksums and a source-tree fingerprint even without Git metadata.
- `runtime-requirements.txt`: exact INSTALLED dependencies and their required
  dependency closure. This records the environment that ran, rather than claiming
  it matched the repository's desired pins. Use the recorded Python/platform for
  reproduction. A fresh-environment installation has not been verified offline.

The ledger + retained records must partition the complete set of input source
rows, with no omissions and no duplicate ledger identities. Preparation verifies
ledger counts before publishing. All these artifacts remain under ignored local
storage. Previous outputs are not modified, so choose the v3 root for future use.

## Training-only information policy

This preparation pipeline performs no resampling, class-weight estimation,
threshold tuning or model training. Split fractions are configuration values,
not selected from evaluation scores. Class labels are used for grouped
stratification, not to tune sampling ratios. Future learned transformations,
class weights or sampling choices must use training data only. Validation/test
records must not contribute optimizer gradients or learned scaling statistics.

The pre-existing legacy experiments select a checkpoint using validation scores;
this is model selection, not gradient fitting. They are explicitly outside this
preparation workflow. If an experiment forbids validation-based selection too,
use a fixed training-only selection policy in the future client training loop;
do not present legacy results as satisfying that stricter policy.

## Version-control protection

`.gitignore` excludes local data, dataset exports, client folders, checkpoints,
model arrays, archives, environment files, common private-key formats and credential
folders. Known pre-existing metric JSONL files have a narrow exception; the safety
guard requires those files to contain only numeric aggregate metric fields.

Before committing/pushing in an actual Git checkout, run:

```powershell
python scripts/check_repository_safety.py
```

The guard checks the Git index, including force-added ignored files, common
credential-like token patterns, private-key headers and unexpected large artifacts.
It prints paths/reasons, never secret values. The same check is configured in
`.github/workflows/repository-safety.yml` for pushes and pull requests. CI runs
AFTER a push; it cannot prevent an upload by itself. Run the guard locally first.
Ignoring files does not untrack already committed files or erase remote history.
The current exported checkout has no Git index, so its tracked/remote state cannot
be verified here; the guard reports NOT VERIFIED rather than a false PASS.
No GitHub upload is performed by preprocessing or this change. Pattern checks
are safeguards, not a guarantee of discovering every possible secret format.

# Client-owned preprocessing and local splitting

The workflow has two explicit stages. Global offline provisioning cleans the
uploaded CSV and assigns unique datasets to clients. Each client subsequently
splits only its own dataset. Feature extraction and normalization happen after
that local split, when the client loads training, validation or test data.

## Stage 1: offline cleaning and client allocation

```powershell
python scripts/prepare_clients.py --clients 5
```

This creates `data/client-datasets/`. Existing outputs are never overwritten;
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
data/client-datasets/
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
python scripts/split_client.py --client-dir data/client-datasets/client_000
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

client_dir = "data/client-datasets/client_000"
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

The older `data/clients/` output, if present, is preserved as a historical centrally
presplit allocation. It is not the output of the current two-stage workflow.
`load_client_split(root, client_id, split)` supports both formats; for isolated
clients, use `load_local_split` directly.

The legacy training simulator and result-reproduction scripts have not been
converted to client-side evaluation. This change implements data provisioning,
local splitting and local feature loading, not a new FL training loop. Sitemap,
robots.txt, metadata collection, LLM labelling and numeric feature scaling remain
separate future additions. Quarantined records must never be added to training
without checking their domains against the protected evaluation splits.

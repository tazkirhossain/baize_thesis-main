# Project structure

Updated 23 October 2026 after removal of the previous implementation.

```text
config/                 Active experiment.json plus partition/model/report defaults
data/                   Ignored CSV, client partitions, splits, runs and recovery archive
src/
  data/                 Cleaning, audit, schema, URL minimization, partitioning,
                        local splitting, protected labels, balancing and features
  models/
    character_cnn/      User-supplied CNN model
    deberta_v3_base/    Pretrained Microsoft encoder, binary head and tokenizer helpers
    federated_state.py Compatible model state transport
  client/               Local data ownership, encoding, training and evaluation
  server/               Flower FedAvg, contracts and local simulation transport
  evaluation/           Both-class metrics, report checks, tables and plots
  transport.py          Bounded model-parameter RPC chunks and verification
  provenance.py         Runtime and optional Git identity
scripts/                Current preparation, verification, federation and report CLIs
results/                Sanitized current-study aggregate reports and plots
tests/                  Current pipeline integrity, leakage, models, FL and metrics
dashboard/              Local React/TypeScript experiment dashboard; no embedded datasets
docs/                   Current design, completed work and cleanup record
README.md               Setup and reproduction
LICENSE                 Preserved source-code attribution
```

The required preparation helpers were migrated from `baize/data/` to `src/data/`;
provenance is now `src/provenance.py`. Active code has no `baize` imports. Package
discovery includes `src*` only. The old simulator, linear models, UI, phase/E1/E2
experiments and notification workflow were removed; see `CLEANUP.md`.

Preserve the agreed order: offline cleaning, unique client allocation, local
train/validation/test splitting, then local encoding/scaling and training-only
balancing. The training server receives no raw URLs or datasets. Keep learned
transforms, sample ratios and model fitting independent of evaluation data.

Existing data, historical source snapshots and the current study's aggregate
results remain unchanged. New outputs use fresh paths. Old source snapshots may
contain the removed package because they record the code that actually ran;
never rewrite them to imply those runs used the migrated implementation.

The dashboard added at the user's request is a new observer of the current
Flower logs, not the removed legacy website. It uses `src/evaluation/dashboard.py`
and `scripts/dashboard.py`; training, partitioning and aggregation are unchanged.

Routine experiment selection uses `config/experiment.json`, resolved by
`src/server/experiment_config.py` and launched with `scripts/run_experiment.py`.
Existing `config/federated/` model presets remain defaults/advanced full configs;
the launcher calls the same training pipeline and selects audited data roots.

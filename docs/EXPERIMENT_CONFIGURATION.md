# Configurable experiments

Numerical logistic classifiers and their training runner were removed on
23 October 2026. Historical numerical reports remain readable; new research
training uses Character CNN or DeBERTa-v3-base via the one-file
`config/experiment.json` / `scripts/run_experiment.py` entry point, which calls
the existing `scripts/federated.py` pipeline. SMOTE now supports the explicit
continuous-encoder-embedding protocol with both models; see `NEURAL_BALANCING.md`.

## Routine training controls

Use [RUN_EXPERIMENT.md](RUN_EXPERIMENT.md) to select model, distribution, balancing,
epochs, batch, rounds, device and automatic plots from one active JSON file.
The remaining sections document the separate data-preparation configurations;
training reuses their existing audited partitions and does not regenerate them.

Agreed 18 October 2026: make experiment choices interchangeable through
configuration. Preserve a common data/model interface rather than copying an
entire pipeline for each method. The preparation commands below configure
partitioning. Model selection, training and measured reporting now use the
separate neural launcher described above.

## Select the partition method

All methods follow [the required shuffled-pool policy](PARTITION_SAMPLING.md).
Explicit `label_skew` and `category_skew` strategies accept per-client target
count plans. Dirichlet remains available when quotas should be generated from
alpha instead. Assignment, local splitting and training-only balancing are
separate stages. Sampling policy versions are recorded in reports; reproduce
both sides of new comparisons into fresh outputs under the current version.

Both presets consume the same audited cleaned source and preserve every record:

| Configuration | Strategy | Client distributions |
|---|---|---|
| `config/five_client_iid.json` | `iid` | Sizes, binary counts and category counts differ by at most one |
| `config/five_client_label_skew.json` | `label_skew` | Exact user-provided Adult/Non-adult counts; approximately equal sizes |
| `config/five_client_non_iid.json` | `label_skew` | Alias of the initial label-skew preset |
| `config/five_client_category_skew.json` | `category_skew` | Category skew with balanced binary counts and approximately equal sizes |
| `config/five_client_dirichlet.json` | `dirichlet` | Optional generated label/category skew; client sizes can differ |

Use one entry point with either preset. Choose fresh output directories:

```powershell
python scripts/partition_clients.py --config config/five_client_iid.json --output data/iid-run-001
python scripts/partition_clients.py --config config/five_client_non_iid.json --output data/non-iid-run-001
```

Each client then independently runs the same local command against its chosen
directory, for example:

```powershell
python scripts/split_client.py --client-dir data/non-iid-run-001/client_000
```

Repeat for clients 001–004. Loading features uses the same
`src.data.preprocessing.load_local_split(client_directory, split)` API for
every strategy. All presets currently require exactly five clients. Changing
the configured count intentionally fails until that protocol is extended.

Custom configurations use the following fields; relative source/output paths
are resolved against the repository root:

```json
{
  "strategy": "dirichlet",
  "parameters": {"alpha": 0.5, "stratify_on": "category"},
  "n_clients": 5,
  "seed": 0,
  "split_seed": 0,
  "source": "data/client-datasets-v3",
  "output": "data/non-iid-custom-001"
}
```

`stratify_on` accepts `category` or `label`. A seeded Dirichlet draw determines
client shares for each stratum. Integer largest-remainder rounding preserves
its full count, and a permutation assigns rows without replacement. `alpha`
must be positive and finite; smaller values generally permit stronger skew.
It does not guarantee a particular observed distribution. Choose alpha and
seeds before inspecting evaluation outcomes. The optional Dirichlet preset is an example
experiment specification, not a measured best setting.

There are no automatic retries, balancing or record drops. A partition missing
either class on a client fails explicitly. Each local splitter also fails if
one of its train/validation/test sets lacks the required class coverage. Strong
skew or small datasets may not satisfy this protocol. No seed is searched to
repair validation/test coverage.

## Shared rules for comparison

The same fixed domain hash assigns train/validation/test roles across clients
and strategies. With identical inputs and split settings, global evaluation
records are identical even when client ownership differs. Fractions remain
approximate at row level. Feature hashing and per-record L2 normalization stay
unchanged; future fitted transforms and class weighting must use training only.

Every new allocation report records a `comparison_contract`: exact cleaned
population fingerprint, source fingerprint, retained total, client count,
feature settings, split seed/policy and split fractions. Compare these reports:

```powershell
python scripts/compare_partitions.py data/iid-run-001 data/non-iid-run-001 --output results/partition-comparison-001.json
```

This checks matching preparation contracts and exports aggregate distributions.
It does not train models or select a performance winner. Mismatched contracts
fail. Outputs are never overwritten. Existing `data/client-iid-v1/` is preserved
and predates the comparison-contract field; generate a fresh IID output with
the current entry point when preparing a comparison. Do not manually amend its
historical report to claim the new provenance.

For subsequent model comparisons, also fix the model initialization, optimizer,
training budget, client participation, feature settings and training seeds.
Report per-client and federation metrics, using a consistent definition for
aggregation. Use validation for permitted model selection under a declared
protocol and reserve test for final evaluation. Do not change data sampling,
class weights or evaluation membership based on validation/test outcomes.
Repeat predeclared seeds to assess variability. No assumption is made that IID
or non-IID will perform better before those experiments run.

## Adding methods without copying the pipeline

- Add an allocation function in `src/data/` returning five index arrays. Indices
  must cover the population once, without replacement; input order is original
  source-row order. Algorithms receive labels, categories, allocation seed and
  client count, never evaluation metrics.
- Use the shared binary pools and sequential consumption helpers in
  `src/data/sampling.py` for every new allocation method. Preserve complete,
  unique record coverage and the declared target counts.
- Register a `Strategy` in `src/data/strategies.py`: callable, option defaults,
  validator and metadata description. Provide `balanced_margins` only when the
  method promises balanced sizes/classes/categories. Add a preset and behavioral
  tests for its declared invariants.
- `src/data/partitioning.py` owns common input validation, coverage checks,
  hashes, audit preservation, atomic publication and reports. Keep these shared.
- Reuse local splitting, loading and the future client training interface.
  Preserve `partition_prepared_iid` and `scripts/partition_iid.py` compatibility.

For future models, balancing, aggregation and LLM review, follow the same pattern:
model-specific implementations, explicit configuration, a stable shared stage
interface, separate output/provenance per run, and tests for stage invariants.
These are architectural requirements for future work, not implemented switches.
LLM label review must preserve originals and keep evaluation labels untouched.
All record-level outputs and sensitive artifacts stay in ignored local storage.

## Neural federated configuration (23 October 2026)

The Flower path is now implemented separately from the preparation commands.
`config/federated/character_cnn.json` and `deberta_v3_base.json` configure round
count, participants, local epochs, batch size, learning rate, balancing, fixed
evaluation threshold and explicit aggregation weighting. See
`FEDERATED_TRAINING.md` for five-client server/client/local commands and exact
supported method/model combinations. Primary comparisons use original local
training counts for FedAvg; validation stays client-local. The local runner
accepts an IID, label-skew or category-skew partition root without rewriting it.

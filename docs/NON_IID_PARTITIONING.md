# Independent five-client non-IID scenarios

Two configurations separate label skew from category skew. Both consume the
same audited cleaned corpus, preserve all 1,468,243 records and use fixed seed
0 with shared domain split seed 0. Client 1–5 here map to storage IDs 0–4.

## Initial experiment: label skew

`config/five_client_label_skew.json` contains the user's exact targets:

| Client | Adult | Non-adult | Total |
|---|---:|---:|---:|
| 1 | 12,000 | 281,649 | 293,649 |
| 2 | 8,500 | 285,149 | 293,649 |
| 3 | 6,500 | 287,149 | 293,649 |
| 4 | 4,500 | 289,148 | 293,648 |
| 5 | 3,311 | 290,337 | 293,648 |
| Total | 34,811 | 1,433,432 | 1,468,243 |

These totals were checked against the actual prepared-source audit before
preparation. There is no separate category quota in this scenario. The actual
records are selected randomly from the two seeded binary pools and consumed
sequentially without replacement.

```powershell
python scripts/partition_clients.py --config config/five_client_label_skew.json
```

Output: ignored local `data/client-label-skew-v1/`. For convenience the existing
`config/five_client_non_iid.json` now contains the same label-skew configuration.
The previous optional Dirichlet preset is retained separately as
`config/five_client_dirichlet.json`; it is not the initial label-skew experiment.

## Separate experiment: category skew

`config/five_client_category_skew.json` fixes a separate allocation table. Adult
and Non-adult counts each differ by at most one between clients, as do total
sizes. This isolates original-category skew while controlling binary prevalence
and client size. No label-skew target list is applied to this scenario.

The plan generator orders the 14 Non-adult categories with seed 0, then fills
five approximately equal Non-adult client capacities using whole categories,
splitting a category only at a capacity boundary. This is a declared strong
category-skew scenario, not an optimized setting. Adult counts are balanced.
Actual rows within every category are drawn from the shared shuffled pools.
The configuration records the complete exact per-category count table.

```powershell
python scripts/partition_clients.py --config config/five_client_category_skew.json
```

Output: ignored local `data/client-category-skew-v1/`. To reproduce plan
generation without overwriting the saved configuration:

```powershell
python scripts/create_category_skew_config.py --config-output config/category-skew-copy.json --data-output data/category-skew-copy
```

The generated category-only binary margins are:

| Client | Adult | Non-adult | Total |
|---|---:|---:|---:|
| 1 | 6,963 | 286,686 | 293,649 |
| 2 | 6,962 | 286,687 | 293,649 |
| 3 | 6,962 | 286,687 | 293,649 |
| 4 | 6,962 | 286,686 | 293,648 |
| 5 | 6,962 | 286,686 | 293,648 |

Exact counts for all 15 original categories are recorded in the saved
configuration, each client's manifest, and the aggregate comparison report.

The generator uses only eligible population counts from the audit, not local
validation/test counts. An edited plan must still exhaust every category exactly.
Both preparation and plan generation refuse to overwrite existing outputs.

## Common integrity and later stages

All original IDs, categories and minimized URLs are preserved. The shared
preparation engine validates canonical URL uniqueness, input checksums, required
identity fields, exact pool totals, disjoint assignment and complete coverage.
Each report records its exact targets, realized counts, sampling-policy version,
seed, source fingerprint and comparison contract. Source exclusion/quarantine
audits are copied with verified hashes. No additional balancing, exclusions or
relabeling are performed.

Partitioning produces five **unsplit** datasets per scenario. Local splitting
remains a separate client action, for example:

```powershell
python scripts/split_client.py --client-dir data/client-label-skew-v1/client_000
```

This local step has now completed separately on all five clients in both
scenarios. Each client's `splits/manifest.json` records its local split state,
counts and hashes; the immutable provisioning manifests retain allocation history.
Existing splits are never overwritten. Configured fractions are 70%/15%/15%,
but actual row counts differ because the shared domain rule keeps each domain
in one role across all clients. See the current split tables in `version2_readme.md`.

Both scenarios carry the same fixed domain-to-split rule for subsequent local
splitting. Training-set balancing and model training are separate stages. No
model quality or IID/non-IID performance winner follows from these allocations.
Use matching data/evaluation/model protocols for subsequent comparisons. The
historical IID allocation stays intact; create a fresh IID run under the current
sampling policy for a new comparison. All record-level artifacts remain local
and ignored by version control.

## Full-corpus verification

Both scenarios were generated and independently verified against the complete
eligible source. Each provisioning verification passed for all 1,468,243 records.
Local splits were created afterward; those historical reports remain unchanged.
To recheck allocation read-only:

```powershell
python scripts/verify_partitions.py data/client-label-skew-v1
python scripts/verify_partitions.py data/client-category-skew-v1
```

These checks validate every canonical URL, actual class/category counts,
original source row coverage, complete source record bytes, input/output hashes,
fixed-seed membership replay and archived preparation code. Use `--output` with
a new local filename to save another report. The completed full-corpus checks
are saved as `verification.json` inside each scenario root.

`results/non_iid_partition_comparison_v1.json` contains aggregate distributions
and a successful shared data/evaluation contract check. It contains no row-level
data and makes no model-performance comparison. Tests: 183 passed; three existing
localhost HTTP tests deselected due to environment restrictions.

After local splitting, separate `local_split_verification.json` reports and
`results/client_local_split_summary_v1.json` record full checks of all new split
files. Each client's split files reconstruct its original owned dataset exactly.
Global split memberships match historical IID, with zero domain-role conflicts.
Per scenario, actual totals are 1,056,709 training (71.9710%), 194,640 validation
(13.2567%) and 216,894 test (14.7724%). The 70/15/15 configuration remains a
domain-level target; row percentages reflect indivisible domain sizes.

# Five-client IID baseline

All five existing IID clients already have local train/validation/test files.
Their 15 split-file hashes and five owned-data hashes were rechecked, retaining
the prior full-record/domain-leakage verification. Current verification is in
`data/client-iid-v1/local_split_verification.json`; per-client counts are in
`version2_readme.md`. Configured fractions are 70/15/15; actual global row shares
are 71.9710%/13.2567%/14.7724% under the shared domain rule. Files were not replaced.

The historical full-corpus allocation below is preserved. Future allocations
follow the revised [shared shuffled-pool policy](PARTITION_SAMPLING.md), recorded
as `binary_shuffled_pools_sequential_v1`. A fixed seed reproduces assignments
within that policy/code version; replay older outputs with their code snapshots.

The user selected URL-level IID allocation on 18 October 2026. Each cleaned URL
belongs to exactly one of five clients (0–4). Domains may appear on several
clients, but a shared domain rule prevents a domain used for training anywhere
from appearing in validation or test anywhere else.

## Allocation

`src/data/iid_partitioning.py` consumes the audited, already cleaned
`data/client-datasets-v3/` corpus. It validates source fingerprints, identity
fields, canonical URL uniqueness and category/label consistency. It preserves
every retained record byte-for-byte, including record ID and original category.
It copies the original exclusions and quarantine ledgers with verified hashes.
No balancing, relabelling, oversampling or additional cleaning is performed.

Rows are ordered by original source row. NumPy PCG64 with the configured seed
permutes records independently within binary-label/original-category strata,
without replacement. Integer quotas make total sizes, binary class counts and
each normalized category count differ by at most one between clients. Random
ties distribute indivisible remainders. This produces similar distributions;
it preserves the corpus's class imbalance rather than correcting it.

## Independent local splitting

Each client runs `scripts/split_client.py` with only its own directory. The
shared `shared_domain_hash_v1` policy hashes the domain and split seed using
SHA-256. The fixed 70%/15%/15% hash intervals assign train/validation/test roles.
The role does not depend on client identity, labels or observed performance.
Every URL from that domain receives the same role across the federation.
The preprocessing configuration checksum is bound into each client's manifest;
the splitter and loader reject silent configuration changes.

Because domains are indivisible for splitting, 70/15/15 are probability targets,
not exact row quotas. Local class proportions can vary. Do not retry seeds or
move records to improve evaluation distributions. Insufficient class coverage
must fail explicitly. No validation/test examples determine class weights,
sampling ratios, thresholds or model parameters. Future balancing must use
training data only. Current hashing and per-row L2 normalization learn no fitted
statistics; any future learned transformation must fit on local training only.

## Reproduction

Run from the repository root using the recorded dependency environment:

```powershell
python scripts/partition_iid.py --config config/five_client_iid.json
0..4 | ForEach-Object {
    python scripts/split_client.py --client-dir ('data/client-iid-v1/client_{0:D3}' -f $_)
    if ($LASTEXITCODE -ne 0) { throw 'Client split failed' }
}
```

The configuration fixes exactly five clients, allocation seed 0 and split seed
0. Existing output directories are never overwritten. Use `--output` with a new
local directory to reproduce an allocation. Reproducibility is tied to the saved
source hashes, settings, source snapshot and runtime versions, not to arbitrary
future dependency versions. Partitioning requires the audited source; after
provisioning, local splitting/loading requires only the client's directory.

`data/client-iid-v1/report.json` contains exact allocations, category/class
spreads, coverage checks and reproduction metadata. `cleaning_report.json`
preserves the source audit. All raw and generated datasets remain ignored local
artifacts. These files do not implement a new Flower training loop.

The full-corpus run retained 1,468,243 records (34,811 Adult and 1,433,432
Non-adult). Independent `verification.json` checks passed, including exact
source-record preservation, seeded membership replay and all 15 local split
files. Of 864,157 domains, 53,398 span multiple clients; no domain crosses split
roles. Exact client and local split counts are in `version2_readme.md`.

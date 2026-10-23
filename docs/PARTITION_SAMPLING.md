# Required random assignment policy

Agreed 18 October 2026. All partition strategies use
`binary_shuffled_pools_sequential_v1` from `src/data/sampling.py`.

1. Use only eligible records from the audited cleaned population, ordered by
   original source row before any random operation.
2. With the fixed allocation seed, independently permute Adult and Non-adult
   indices into separate random pools.
3. Establish quotas without using local validation/test outcomes. Assign Client
   1 its required counts from those shuffled pools and consume the selected
   prefixes, then repeat for Clients 2–5. A prefix of a uniform random permutation
   is a random sample without replacement; consuming its prefix removes those
   rows from availability. There is no need to redraw or delete dataframe rows.
4. For category strata, filter the appropriate shuffled binary pool by original
   category and consume that category's quotas in the same client order.
5. Verify every index occurs exactly once in the complete union. Reject any
   overlap, omission, invalid index or inconsistent pool total.

Client labels 1–5 here refer to assignment order. Existing storage identifiers
remain `client_000` through `client_004`. Within each output file, records are
sorted by original source row for reproducibility; this does not change their
random client assignment.

## Quotas and configurable strategies

- `iid`: proportional Adult/Non-adult quotas with integer rounding and category
  stratification. Sizes, binary counts and category counts differ by at most one.
- `label_skew`: a supplied Adult/Non-adult allocation plan determines exact
  counts; the sampler randomly chooses the actual records for each client.
- `category_skew`: a supplied normalized original-category plan determines exact
  category counts; selection within each category is random.
- `dirichlet`: the configured seed/alpha generates label or category quotas;
  those quotas are filled from the same shuffled pools. This is an optional
  generated-plan strategy, not a substitute for explicitly supplied targets.

All plans must account for each eligible pool exactly. Counts must be five
nonnegative integers in client order. There is no replacement, hidden resampling,
automatic seed retry, oversampling or count correction. The existing protocol
also requires both binary classes on each client and each eventual local split;
insufficient coverage fails explicitly.

For example, **on a toy corpus with 20 Adult and 50 Non-adult records**, select:

```json
{
  "strategy": "label_skew",
  "parameters": {
    "allocation_plan": {
      "Adult": [2, 3, 4, 5, 6],
      "Non-adult": [20, 15, 8, 5, 2]
    }
  }
}
```

For a toy corpus with 20 Adult, 30 Arts and 20 Health records:

```json
{
  "strategy": "category_skew",
  "parameters": {
    "allocation_plan": {
      "adult": [2, 3, 4, 5, 6],
      "arts": [10, 8, 6, 4, 2],
      "health": [2, 3, 4, 5, 6]
    }
  }
}
```

These are illustrative counts, not proposed allocations for the real corpus.
Add the common source/output, five-client count, seed and split seed fields
described in `EXPERIMENT_CONFIGURATION.md`. Supply real target counts before
running a planned non-IID experiment. No real non-IID targets have been invented.

## Separate stages and reproducibility

Partitioning publishes unsplit client datasets only. Each client subsequently
splits its own records under the unchanged shared domain rule. Balancing is a
separate future training-only stage and must not alter validation/test data.

New reports record the sampling policy, fixed seed, client order, strategy and
allocation parameters. The comparison contract also includes the sampling
policy. Reproducibility requires matching policy/code and runtime versions as
well as the seed. This revision changes how future records are randomized;
existing historical partition files and reports are preserved. Replay older
allocations with their archived code snapshots instead of this revised sampler.
Use fresh output directories when preparing comparisons under the new policy.

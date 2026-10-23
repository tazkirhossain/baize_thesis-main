# Local dataset and audit identity

This document describes the uploaded dataset used in the current pipeline.
Earlier research conclusions, collection-date estimates and phase decisions are
not evidence for this study and were removed from the active documentation.

| Field | Current audited value |
|---|---|
| Local input | `data/raw/url.csv` |
| Format | Three columns: id, url, label; uploaded file is headerless |
| Raw records | 1,562,978 |
| Raw SHA-256 | `79e32aeafc74a5c875add374d1ea5d163f867ee9109a47907c99271bb76002a4` |
| Audited source | `data/client-datasets-v3/` |
| Retained | 1,468,243 |
| Adult | 34,811 |
| Non-adult | 1,433,432 |
| Invalid/missing exclusions | 1,047 |
| Same-label duplicate exclusions | 89,385 |
| Conflicting-label exclusions | 4,303 records across 148 canonical URLs |

Adult maps to 1; the other 14 recognized source categories map to 0. Preserve the
original category and record ID; `source_row` is the unique source-record identity.
This taxonomy mapping does not independently verify a website's current content.
Source provenance, collection date and dataset licensing remain unverified;
the source-code MIT license does not cover the CSV.

| Original category | Retained records |
|---|---:|
| Adult | 34,811 |
| Arts | 237,804 |
| Business | 235,405 |
| Computers | 111,101 |
| Games | 52,546 |
| Health | 58,023 |
| Home | 26,886 |
| Kids | 37,077 |
| News | 8,373 |
| Recreation | 102,423 |
| Reference | 53,654 |
| Science | 97,763 |
| Shopping | 93,233 |
| Society | 226,372 |
| Sports | 92,772 |

Missing-value counts in the input audit: id=0, url=3, label=0. Missing and secondary
reason counts may overlap; primary exclusion counts partition all excluded rows.
`report.json` and private `exclusions.jsonl` in the audited source are authoritative.
See `PREPROCESSING.md` for rules and `version2_readme.md` for current client counts.
The dataset must remain local and ignored by version control.

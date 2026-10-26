# Published data and artifacts

## Local cleanup, 26 October 2026

The owner's new cleanup request supersedes retaining disposable test outputs
in the working data folder. Generated `data/test-*` fixtures, repeated synthetic
smoke checkpoints, preview graphs and obsolete dashboard/npm artifacts are
removed locally. Small historical verification records, including the nested
test Git metadata archive, are relocated under
`results/maintenance/data-cleanup-20261026-*/`. CNN diagnostic evidence lives in
`results/diagnostics/`. Checksummed cleanup manifests record the changes.
Source `tests/`, original CSV, audited datasets/client splits, pretrained weights,
required Python runtimes, and client-local enrichment caches are retained.
This does not rewrite Git history or publish the cleanup automatically. The
following sections describe the earlier publication snapshot.

On 23 October 2026 the owner explicitly requested publishing the complete data
folder and removing Git ignore patterns. This overrides the older instructions
to exclude datasets, checkpoints, local runs, caches and dependencies. Older
source snapshots and documentation inside historical runs record the old
policy and are preserved as history. The root .gitignore now has no patterns.

The publication includes the raw CSV, all existing client dataset versions,
IID/label-skew/category-skew partitions, local splits, pretrained DeBERTa files,
checkpoints, generated reports/logs, dependency/runtime folders, test outputs
and caches. These are provided at their existing paths. Source code and pipeline
behavior remain unchanged by this publication. Actual credentials/private keys
are not authorized for publication and remain blocked by CI. Exact, visibly fake
safety-test payloads under data/test-* are test fixtures, not real credentials.
Public CA certificate bundles are not private keys.
The exact-file non-secret review records hashes for eight generated safety-test
artifacts and an installed public cryptography self-test among them. Its upstream
source was compared after normalizing Windows line endings. Reviewed files are
exempt only while their SHA-256 matches; edits require review again. This does
not authorize actual credentials or arbitrary files with a similar name.

## Clone with large files

Install Git LFS, then:

```powershell
git lfs install
git -c core.longpaths=true clone --branch codex/federated-url-pipeline https://github.com/tazkirhossain/baize_thesis-main.git
cd baize_thesis-main
git lfs pull
```

Large files (10 MiB and larger at publication time) use exact-path LFS rules in
.gitattributes. Their GitHub view can show a small pointer instead of file data;
Git LFS retrieves the real contents. Browser ZIP downloads may contain pointers,
so use a Git/LFS clone. Storage/download quotas depend on the repository owner's
GitHub plan. No billing settings are changed by the publication.

## Nested test Git metadata

Historical tests created hundreds of temporary repositories inside data/test-*.
Git cannot commit a .git directory as a normal path, and submodule entries would
not contain those tests' actual files. Ordinary files are published normally;
the nested metadata is preserved in data/nested-test-git-metadata.zip with a
path/SHA-256 manifest alongside it. This archive is not required for training.
To restore those historical test repositories, optionally extract the archive
at the project root after inspecting its manifest. Root repository metadata is
never included. All original local files are retained; nothing is deleted to
make the publication.

Local Windows/Python runtimes and node_modules are included at the owner's
request, but other platforms should install the project dependencies normally.
Existing checkpoints do not automatically resume a training session. Use
CHECKPOINTS.md and RUN_EXPERIMENT.md for training and evaluation behavior.

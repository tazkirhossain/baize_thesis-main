# Published project data

The repository owner authorized publishing this complete folder on 23 October
2026. Large files use Git LFS. Run `git lfs install` and `git lfs pull` after
cloning so datasets and model weights are real files rather than LFS pointers.

Current training roots (automatically selected by config/experiment.json):

- `client-iid-v1/`: five IID clients and their immutable local splits.
- `client-label-skew-v1/`: five label-skew clients and local splits.
- `client-category-skew-v1/`: five category-skew clients and local splits.
- `raw/url.csv`: original source dataset.
- `pretrained/deberta-v3-base/`: pinned pretrained encoder/tokenizer files.

Cleanup authorized on 26 October 2026 removes generated `test-*` directories,
repeated synthetic smoke checkpoints, graph previews and obsolete dashboard/npm
artifacts. These are disposable verification outputs, separate from each
client's original `splits/test.jsonl`, which remains required and preserved.

The project's source regression tests remain in `../tests/`. They do not run
during normal training. New test runs may create fresh temporary outputs here.

Other audited dataset versions and the original recovery archive remain as
history; they are not automatically selected for a new experiment. Required
`flower-runtime/` and `neural-runtime/` packages are retained for this machine,
along with transport staging and client-owned runtime/enrichment directories.
Launcher temporary configurations are retained because running jobs may use
them. Install dependencies for your own Python/OS; included Windows packages
are not a portable environment installer.

Small historical logs/verification evidence and the cleanup manifest are under
`../results/maintenance/data-cleanup-20261026-*/`. CNN initialization/training
diagnostics are under `../results/diagnostics/`. The manifest records removed
paths, sizes, tracked-file counts and before/after protected-file checksums.

See docs/PUBLISHED_DATA.md for the clone instructions and historical publication
policy. See docs/RUN_EXPERIMENT.md for training
and docs/CHECKPOINTS.md for global weight checkpoints and best-model selection.

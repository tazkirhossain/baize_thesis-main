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

Other dataset versions, tests, caches, runtime packages, logs and snapshots are
historical artifacts included at the owner's request. They are not automatically
selected for a new experiment. Install dependencies for your own Python/OS;
included Windows runtime packages are not a portable environment installer.

See docs/PUBLISHED_DATA.md for the clone instructions and the archive preserving
nested test repositories' Git metadata. See docs/RUN_EXPERIMENT.md for training
and docs/CHECKPOINTS.md for global weight checkpoints and best-model selection.

# Character CNN and DeBERTa-v3-base

Website inputs were added 25 October 2026 as an optional client-local mode.
`config/experiment_website.json` enables frozen title/metadata/text/sitemap context
for both models; URL-only remains available. Inputs use fixed field budgets, a
declared CNN whitespace vocabulary and the existing DeBERTa tokenizer. Collection
and training are separate; no model fetches websites. Read WEBSITE_ENRICHMENT.md
before starting an enriched experiment or reloading an older URL-only checkpoint.

CNN setup recorded 22 October 2026; DeBERTa implementation added 23 October 2026.

The two research models have separate files:

- `src/models/character_cnn/model.py`: the user-supplied `CharacterLevelCNN`.
- `src/models/deberta_v3_base/model.py`: Microsoft's DeBERTa-v3-base pretrained
  encoder loader, binary sequence-classification head and tokenizer helpers.

The numerical logistic classifiers and their training runner were removed at
the user's request. Their historical results remain available for audit. The earlier demo and linear simulator have also been removed. The Flower neural path added on 23 October is documented in
`FEDERATED_TRAINING.md`. Synthetic federated training is verified; a full-corpus
neural performance study has not run.

## Character CNN contract

The architecture retains eight convolution blocks with batch normalization and
ReLU, two factor-two max pools, residual additions at blocks 3 through 6,
adaptive average pooling, and a 512 -> 1024 -> number_of_classes head with 0.5
classifier dropout. Convolution channels are 256 in blocks 1-6 and 512 in blocks
7-8. Kernel sizes are 7, 7, 3, 3, 3, 3, 3, 3, with length-preserving padding.
Conv/hidden-linear weights use the supplied Kaiming initialization and zero
biases. The optional versioned final-classifier initialization is described below.

### CNN initialization correction, 26 October 2026

The active launcher setting `cnn_initialization="stable_classifier_v1"` maps to
the low-level CNN `model.initialization` setting of the same name. It initializes
only the final 1024-to-2 classifier with zero-mean normal weights, standard
deviation 0.01 and zero bias. Hidden layers and the supplied architecture remain
unchanged. Missing settings retain `legacy_fan_out`, including its final-layer
Kaiming initialization, so historical experiments are not silently redefined.
The policy is bound into the experiment contract and must match across primary
comparisons. Synthetic startup checks establish initialization behavior only;
they do not establish improved Adult detection. See
[the diagnosis](ADULT_PERFORMANCE_DIAGNOSIS.md) for the optimization-only audit
and training-only threshold selection. Existing runs require no migration;
launch a fresh run to use the correction.

One dimensional correction is intentional: input is transposed before
`Dropout1d`, so this layer sees (batch, characters, sequence_length) and drops
character channels. The supplied ordering would drop sequence positions.

Input must be floating character features shaped (batch, sequence_length,
number_of_characters + len(extra_characters)); integer token IDs and the old
hashed/L2 or numerical feature matrices cannot be passed directly. Sequence
length must be at least four for the two pooling stages. Training BatchNorm also
requires more than one value per channel; batch=1 with pooled length=1 is not
valid in training mode. Use `model.eval()` for inference.

Output is raw logits of shape (batch, number_of_classes). For this task pass
number_of_classes=2 and retain the existing mapping 0=Non-adult, 1=Adult.
`CrossEntropyLoss` takes logits and integer class labels; evaluation can convert
logits to probabilities with softmax and use column 1 for Adult.

The module does not read a working-directory `config.json` during import.
Configuration is passed explicitly through `CharacterCNNConfig` or an args
object with the same three attributes. No vocabulary, character ordering or
maximum input length is inferred from validation/test records. A fixed character
encoder, padding/truncation policy and Flower trainer are implemented in the
separate federated path. All clients share the exact configured vocabulary and
model layout. See `FEDERATED_TRAINING.md` for the encoding and buffer policy.

```python
import torch
from src.models.character_cnn import CharacterCNNConfig, CharacterLevelCNN

# Illustrative alphabet only; this does not configure a real-data experiment.
base_characters = "abcdefghijklmnopqrstuvwxyz0123456789"
extra_characters = ".:/_-"
args = CharacterCNNConfig(
    number_of_characters=len(base_characters),
    extra_characters=extra_characters,
    dropout_input=0.1,
)
model = CharacterLevelCNN(args, number_of_classes=2)
model.eval()
n_characters = len(base_characters) + len(extra_characters)
ids = torch.zeros((2, 64), dtype=torch.long)  # Synthetic shape check only.
one_hot = torch.nn.functional.one_hot(ids, n_characters).float()
with torch.no_grad():
    adult_probability = model(one_hot).softmax(dim=1)[:, 1]
```

PyTorch is optional for data preparation and historical report reading. Install the neural
extra with `pip install -e ".[neural]"` in the project environment. The CNN was
checked using PyTorch 2.8.0+cpu; no CUDA device was available. Run focused checks
with `python -m pytest tests/test_character_cnn.py -q`. The tests use synthetic
one-hot tensors, verify forward and backward passes, channel dropout dimensions,
variable lengths and input guards; they do not establish model accuracy.

Training-only undersampling/oversampling index views are reused by the Flower
CNN trainer. The CNN undersampling ensemble is trained and reported separately.
SMOTE now supports continuous encoder embeddings with the two-stage protocol in
`NEURAL_BALANCING.md`. Never apply it to character or DeBERTa token IDs.

## DeBERTa-v3-base implementation and required files

The source selected by the user is
[Microsoft's DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base).
Its [official configuration](https://huggingface.co/microsoft/deberta-v3-base/blob/8ccc9b6f36199bec6961081d44eb72fb3f7353f3/config.json)
uses model_type `deberta-v2`, 12 layers, 768 hidden dimensions, 12 attention heads,
3072 intermediate dimensions, 128100 vocabulary entries and 512 maximum token
positions. Transformers supplies the actual encoder architecture; no copied or
approximate transformer is substituted. The official sequence-classification
implementation adds a context pooler, dropout and a two-logit linear head.
Labels are 0=Non-adult and 1=Adult, with ordinary unweighted cross-entropy when
labels are supplied. Its return value is a Transformers output object: access
`outputs.logits` and, when labels are provided, `outputs.loss`.

For the PyTorch files visible in the user's screenshot, place these in the
ignored local directory `data/pretrained/deberta-v3-base/`:

| File | Purpose |
|---|---|
| `config.json` | Official encoder architecture |
| `pytorch_model.bin` | Pretrained PyTorch weights |
| `spm.model` | Original SentencePiece vocabulary/tokenization model |
| `tokenizer_config.json` | Original tokenizer settings |

Only PyTorch weights are needed for this implementation, so `tf_model.h5` and
`rust_model.ot` are not required. A compatible `model.safetensors` checkpoint
can also be loaded by Transformers; it is not a file shown in the screenshot.
See the [official file list](https://huggingface.co/microsoft/deberta-v3-base/tree/8ccc9b6f36199bec6961081d44eb72fb3f7353f3).

Install optional dependencies using `pip install -e ".[deberta]"`. The tested
runtime is torch 2.8.0+cpu, transformers 4.55.4 and sentencepiece 0.2.1.
SentencePiece was installed into ignored `data/neural-runtime/` for local tests,
without altering the user's global packages. That temporary test setup requires
`$env:PYTHONPATH = (Join-Path (Get-Location) 'data/neural-runtime')` in PowerShell;
a normal installation of the optional extra does not.

```python
import torch
from src.models.deberta_v3_base import build_model, load_tokenizer, tokenize_urls

checkpoint = "data/pretrained/deberta-v3-base"
model = build_model(checkpoint, seed=42)  # Offline; missing files raise an error.
tokenizer = load_tokenizer(checkpoint)
# max_length=256 is a declared starting configuration, not a tuned optimum.
batch = tokenize_urls(tokenizer, ["https://example.org/news"], max_length=256)
model.eval()
with torch.no_grad():
    logits = model(**batch).logits
    adult_probability = logits.softmax(dim=1)[:, 1]
```

The base checkpoint is pretrained on general language, not this Adult/Non-adult
task. Its new pooler/classifier parameters must be fine-tuned; output before
fine-tuning is not a valid detector. `build_model()` validates the base architecture,
seeds any new head parameters, records loading diagnostics and rejects missing
encoder parameters. It never replaces unavailable pretrained weights with a
randomly initialized model. Direct class construction from a config is explicitly
random initialization, used for small offline tests only.

For an explicit Hub download, use the same immutable revision for both model and
tokenizer, rather than `main`. The default revision is
`8ccc9b6f36199bec6961081d44eb72fb3f7353f3`, checked on 23 October 2026. The helpers
default to `local_files_only=True`; pass False explicitly to download through
Transformers, with its cache kept under `data/pretrained/hf-cache/`. Alternatively,
the four-file download can be prepared once:

```python
from huggingface_hub import snapshot_download
from src.models.deberta_v3_base.model import MODEL_ID, MODEL_REVISION

snapshot_download(
    repo_id=MODEL_ID, revision=MODEL_REVISION,
    allow_patterns=["config.json", "pytorch_model.bin", "spm.model", "tokenizer_config.json"],
    local_dir="data/pretrained/deberta-v3-base",
)
```

`config/models/deberta_v3_base.json` records the starting model/tokenizer settings.
The Flower CLI uses `config/federated/deberta_v3_base.json` for its complete
experiment contract. The tokenizer is the fixed pretrained tokenizer and is
not fitted on training, validation or test records. It transforms each split
without resampling; masks accompany padding, and lengths are explicitly truncated.
No StandardScaler, L2 transform or SMOTE operates on DeBERTa token IDs.

For FL, the server must distribute one identical initial model (including head
and buffers), tokenizer revision, label mapping and token-length settings to all
clients. The Flower path supplies client-local loaders/trainers and aggregation,
and supports independent ensembles for both models with bounded chunked transfer.
Full real-data DeBERTa runs and real-corpus neural metrics remain future work.
See `NEURAL_BALANCING.md` for SMOTE protocol details and resource limits.

Verification uses synthetic data, a tiny real DebertaV2 encoder and a locally
trained synthetic SentencePiece vocabulary. It covers forward/backward passes,
mask behavior, checkpoint save/reload, tokenizer padding/truncation and failure
on missing/incompatible checkpoints. These checks do not load the full public
pretrained weights and do not claim Adult detection accuracy.
Combined validation: 10 tests passed (6 DeBERTa, 4 CNN) on 23 October 2026.

## Full DeBERTa checkpoint prepared, 23 October 2026

Downloaded the four files from Microsoft's public `microsoft/deberta-v3-base`
checkpoint at immutable revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3` into
`data/pretrained/deberta-v3-base/`. `scripts/prepare_deberta.py` reproduces the
public download without credentials or client-data transfer. SHA-256/LFS hashes,
Git blob IDs and file sizes are checked against the pinned Hub metadata and saved
in ignored `download_manifest.json`. The weight file is 371,146,213 bytes with
SHA-256 `691d48a2800b926a19e3051def466fc2cca4f59a15e42ce4a0cf7f1b380b5e33`.

The actual 184,423,682-parameter pretrained model passed **offline loading,
production client tokenization, inference, backward and one plain-SGD step**.
Both an encoder parameter and the new binary head changed; all parameters were
trainable. The test used one synthetic example, max length 16, CPU, seed 42 and
the configured learning rate 0.00002. The normal max length 256 and all experiment
settings remain unchanged. No corpus/validation/test records were used, and the
original pretrained files were not modified or replaced with the smoke-trained
weights. This establishes full-checkpoint training compatibility, not detection
quality or full five-client training performance.

Evidence: `data/pretrained/deberta-pretrained-verification.json`, its `.log`,
`verify_deberta_pretrained.py` and `deberta-focused-tests.log` (22 tests passed).
The new pooler and Adult/Non-adult classifier remain untrained for the real task.
Full real-data CNN/DeBERTa experiments and fair comparisons remain pending.

This local machine has approximately 8 GB RAM and no CUDA. Single-model checking
passed, but the five-client local simulation retains a server model, five client
models and aggregation buffers simultaneously. Full DeBERTa experiments need
sufficient additional memory and compute, preferably a suitable GPU machine or
separate client devices. Do not infer that this small check established the
resource requirements of the full run.

For a fair model comparison, reuse the same verified client ownership and split
membership, balancing policy, fixed evaluation threshold and declared seeds.
Record each model's encoding, optimizer/training budget and costs. Assess Adult
recall/F1 and macro-F1 alongside the other required metrics using a declared
validation/final-test protocol; neither model has been selected as best.

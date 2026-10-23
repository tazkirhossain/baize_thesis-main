"""Microsoft DeBERTa-v3-base with a binary sequence-classification head.

DeBERTa-v3 checkpoints use Transformers' DebertaV2 implementation. Loading is
explicit and offline by default: no download, config read or training on import.
"""
from copy import deepcopy
from pathlib import Path

import torch
from transformers import AutoConfig, DebertaV2Config, DebertaV2ForSequenceClassification


MODEL_ID = "microsoft/deberta-v3-base"
MODEL_REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
DEFAULT_CACHE = Path(__file__).resolve().parents[3] / "data" / "pretrained" / "hf-cache"
ID2LABEL = {0: "Non-adult", 1: "Adult"}
LABEL2ID = {name: index for index, name in ID2LABEL.items()}
BASE_ARCHITECTURE = {
    "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12,
    "intermediate_size": 3072, "vocab_size": 128100,
    "max_position_embeddings": 512, "relative_attention": True,
    "position_biased_input": False,
}


class DebertaV3BaseClassifier(DebertaV2ForSequenceClassification):
    """Two logits, with optional standard CrossEntropyLoss when labels are given.

    Direct construction from a config initializes RANDOM weights. Use build_model()
    for pretrained loading; tiny configs are useful only for offline unit tests.
    """

    def __init__(self, config):
        if not isinstance(config, DebertaV2Config) or config.model_type != "deberta-v2":
            raise ValueError("a DebertaV2Config is required for DeBERTa-v3")
        config = deepcopy(config)
        config.num_labels = 2
        config.id2label = dict(ID2LABEL)
        config.label2id = dict(LABEL2ID)
        config.problem_type = "single_label_classification"
        super().__init__(config)


def _loading_options(source, revision, cache_dir, local_files_only):
    source = str(source)
    options = {"cache_dir": str(cache_dir), "local_files_only": local_files_only}
    if not Path(source).is_dir():
        # Use immutable revisions, not a moving branch, for reproducible clients.
        if not isinstance(revision, str) or len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise ValueError("Hub revision must be a full immutable commit SHA")
        options["revision"] = revision
    return source, options


def build_model(source=MODEL_ID, *, revision=MODEL_REVISION, cache_dir=DEFAULT_CACHE,
                local_files_only=True, seed=42):
    """Load pretrained base weights and a reproducibly initialized binary head.

    Missing/incompatible weights fail; there is no random-backbone fallback. The
    public base checkpoint has no trained Adult/Non-adult head. All FL clients must
    receive the same initial model state from their server before local training.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    source, options = _loading_options(source, revision, cache_dir, local_files_only)
    if Path(source).is_dir() and not (Path(source) / "config.json").is_file():
        raise FileNotFoundError("the local model folder needs config.json and pretrained weights")
    config = AutoConfig.from_pretrained(source, trust_remote_code=False, **options)
    if not isinstance(config, DebertaV2Config) or any(
        getattr(config, name, None) != value for name, value in BASE_ARCHITECTURE.items()
    ):
        raise ValueError("checkpoint does not match the DeBERTa-v3-base architecture")
    # Preserve the caller's CPU RNG; the seed only controls new head parameters.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model, loading = DebertaV3BaseClassifier.from_pretrained(
            source, config=config, output_loading_info=True, **options
        )
    missing_backbone = [key for key in loading["missing_keys"]
                        if not key.startswith(("pooler.", "classifier."))]
    if missing_backbone or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError("pretrained encoder weights are incomplete or incompatible")
    model.loading_audit = {
        "source": source, "revision": options.get("revision"),
        "seed": seed, "pretrained_encoder": True,
        "new_head_keys": list(loading["missing_keys"]),
        "unused_checkpoint_keys": list(loading["unexpected_keys"]),
        "adult_non_adult_finetuning_performed": False,
    }
    return model


def load_tokenizer(source=MODEL_ID, *, revision=MODEL_REVISION,
                   cache_dir=DEFAULT_CACHE, local_files_only=True):
    """Load the checkpoint's original SentencePiece tokenizer, without refitting."""
    from transformers import DebertaV2Tokenizer

    source, options = _loading_options(source, revision, cache_dir, local_files_only)
    # The slow tokenizer directly uses spm.model, avoiding implicit conversion.
    return DebertaV2Tokenizer.from_pretrained(source, **options)


def tokenize_urls(tokenizer, urls, *, max_length):
    """Encode already-preprocessed URLs; no fitting, URL fetching or resampling.

    max_length is an explicit predeclared experiment setting, not chosen from
    holdout statistics. Padding has an attention mask, and truncation is explicit.
    """
    if type(max_length) is not int or not 2 <= max_length <= 512:
        raise ValueError("max_length must be between 2 and 512, including special tokens")
    if isinstance(urls, str):
        raise ValueError("pass a nonempty sequence of URL strings, not one string")
    urls = list(urls)
    if not urls or any(not isinstance(url, str) or not url.strip() for url in urls):
        raise ValueError("nonempty URL strings are required")
    return tokenizer(urls, padding=True, truncation=True, max_length=max_length,
                     return_tensors="pt", return_attention_mask=True,
                     return_token_type_ids=False)

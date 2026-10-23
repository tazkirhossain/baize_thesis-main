"""Offline DeBERTa contracts; tiny random models are not pretrained-base results."""
from io import BytesIO
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import BertConfig, DebertaV2Config
from src.models.deberta_v3_base.model import (
    BASE_ARCHITECTURE, MODEL_ID, MODEL_REVISION, AutoConfig,
    DebertaV3BaseClassifier, build_model, load_tokenizer, tokenize_urls,
)


@pytest.fixture(autouse=True)
def limited_cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


def tiny_config():
    return DebertaV2Config(
        vocab_size=128, hidden_size=24, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=4,
        max_position_embeddings=32, relative_attention=True,
        position_buckets=8, pos_att_type=["p2c", "c2p"],
        position_biased_input=False, hidden_dropout_prob=0,
        attention_probs_dropout_prob=0, pooler_dropout=0,
    )


def test_binary_loss_backward_and_label_mapping():
    model = DebertaV3BaseClassifier(tiny_config())
    ids = torch.tensor([[2, 7, 8, 3, 0], [2, 9, 3, 0, 0]])
    output = model(input_ids=ids, attention_mask=ids.ne(0).long(),
                   labels=torch.tensor([0, 1]))
    assert output.logits.shape == (2, 2)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    for parameter in (model.classifier.weight, model.deberta.embeddings.word_embeddings.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert model.config.id2label == {0: "Non-adult", 1: "Adult"}
    assert model.config.problem_type == "single_label_classification"


def test_attention_mask_and_saved_checkpoint_roundtrip(tmp_path):
    model = DebertaV3BaseClassifier(tiny_config()).eval()
    ids = torch.tensor([[2, 7, 3, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    with torch.no_grad():
        expected = model(input_ids=ids, attention_mask=mask).logits
        altered = model(input_ids=torch.tensor([[2, 7, 3, 11, 12]]), attention_mask=mask).logits
    torch.testing.assert_close(expected, altered)
    model.save_pretrained(tmp_path)
    restored = DebertaV3BaseClassifier.from_pretrained(tmp_path, local_files_only=True).eval()
    with torch.no_grad():
        torch.testing.assert_close(expected, restored(input_ids=ids, attention_mask=mask).logits)


def test_missing_checkpoint_and_wrong_architecture_fail_without_fallback(tmp_path):
    with pytest.raises(OSError):
        build_model(tmp_path)
    BertConfig().save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="architecture"):
        build_model(tmp_path)
    tiny_config().save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="architecture"):
        build_model(tmp_path)


@pytest.mark.parametrize("missing", [[], ["deberta.embeddings.word_embeddings.weight"]])
def test_pretrained_loader_pins_revision_and_rejects_missing_encoder(monkeypatch, missing):
    config = DebertaV2Config(**BASE_ARCHITECTURE)
    calls = []
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: config)

    def load(cls, source, **kwargs):
        calls.append((source, kwargs))
        return torch.nn.Identity(), {"missing_keys": missing + ["classifier.weight"],
            "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}

    monkeypatch.setattr(DebertaV3BaseClassifier, "from_pretrained", classmethod(load))
    if missing:
        with pytest.raises(ValueError, match="incomplete"):
            build_model()
    else:
        model = build_model()
        assert model.loading_audit["pretrained_encoder"]
    assert calls[0][0] == MODEL_ID
    assert calls[0][1]["revision"] == MODEL_REVISION
    assert calls[0][1]["local_files_only"] is True
    with pytest.raises(ValueError, match="immutable"):
        build_model(revision="main")


def test_sentencepiece_local_loading_padding_and_truncation(tmp_path):
    spm = pytest.importorskip("sentencepiece")
    writer = BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(["https://example.org/a", "https://example.org/a/long/path", "news science games"]*10),
        model_writer=writer, vocab_size=64, hard_vocab_limit=False,
        bos_id=-1, eos_id=-1, minloglevel=2,
    )
    (tmp_path / "spm.model").write_bytes(writer.getvalue())
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"do_lower_case": False}))
    tokenizer = load_tokenizer(tmp_path)
    encoded = tokenize_urls(tokenizer, ["a", "https://example.org/a/long/path"], max_length=12)
    assert encoded["input_ids"].shape == encoded["attention_mask"].shape == (2, 12)
    assert (encoded["attention_mask"] == 0).any()
    assert "token_type_ids" not in encoded
    model = DebertaV3BaseClassifier(tiny_config()).eval()
    with torch.no_grad():
        assert model(**encoded).logits.shape == (2, 2)
    for urls, length in ((["a"], 513), ("a", 12), ([""], 12)):
        with pytest.raises(ValueError):
            tokenize_urls(tokenizer, urls, max_length=length)

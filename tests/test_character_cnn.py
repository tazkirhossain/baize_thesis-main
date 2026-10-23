"""CNN tensor contract and backward pass."""
import pytest

torch = pytest.importorskip("torch")
from src.models.character_cnn.model import CharacterCNNConfig, CharacterLevelCNN


@pytest.fixture
def model():
    original_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    network = CharacterLevelCNN(CharacterCNNConfig(8, "_-", 0.1), 2)
    yield network
    torch.set_num_threads(original_threads)


def one_hot(batch, length):
    return torch.nn.functional.one_hot(torch.randint(0, 10, (batch, length)), 10).float()


@pytest.mark.parametrize("length", [16, 33])
def test_binary_logits_variable_lengths_and_eval_determinism(model, length):
    model.eval()
    x = one_hot(2, length)
    with torch.no_grad():
        logits = model(x)
        assert logits.shape == (2, 2)
        assert torch.isfinite(logits).all()
        torch.testing.assert_close(logits, model(x))


def test_backward_reaches_all_blocks_and_dropout_uses_character_channels(model):
    seen = []
    hook = model.dropout_input.register_forward_pre_hook(
        lambda module, inputs: seen.append(tuple(inputs[0].shape))
    )
    logits = model(one_hot(2, 16))
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    hook.remove()
    assert seen == [(2, 10, 16)]
    for index in range(1, 9):
        gradient = getattr(model, f"conv{index}")[0].weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    assert torch.isfinite(model.fc[-1].weight.grad).all()


def test_rejects_token_ids_bad_channels_and_too_short_sequences(model):
    with pytest.raises(TypeError, match="token IDs"):
        model(torch.ones(2, 16, 10, dtype=torch.long))
    with pytest.raises(ValueError, match="expected"):
        model(torch.zeros(2, 16, 9))
    with pytest.raises(ValueError, match="length"):
        model(torch.zeros(2, 3, 10))



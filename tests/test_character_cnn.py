"""CNN tensor contract and backward pass."""
import pytest
from types import SimpleNamespace

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


def test_versioned_initialization_preserves_hidden_states_and_historical_callers(model):
    torch.manual_seed(42)
    explicit = CharacterLevelCNN(CharacterCNNConfig(8, "_-", 0.1, "legacy_fan_out"), 2)
    torch.manual_seed(42)
    old_args = CharacterLevelCNN(SimpleNamespace(number_of_characters=8, extra_characters="_-", dropout_input=0.1), 2)
    torch.manual_seed(42)
    stable = CharacterLevelCNN(CharacterCNNConfig(8, "_-", 0.1, "stable_classifier_v1"), 2)
    for key, expected in model.state_dict().items():
        torch.testing.assert_close(explicit.state_dict()[key], expected, rtol=0, atol=0)
        torch.testing.assert_close(old_args.state_dict()[key], expected, rtol=0, atol=0)
        if key != "fc.3.weight":
            torch.testing.assert_close(stable.state_dict()[key], expected, rtol=0, atol=0)
    assert not torch.equal(stable.fc[-1].weight, explicit.fc[-1].weight)
    with pytest.raises(ValueError, match="initialization"):
        CharacterLevelCNN(CharacterCNNConfig(8, initialization="unknown"), 2)


@pytest.mark.parametrize("seed", [7, 42, 101])
def test_stable_classifier_avoids_saturated_random_predictions_and_can_backpropagate(model, seed):
    # A fresh binary classifier should not be confidently wrong on arbitrary
    # balanced inputs. This failed with ReLU fan_out applied to the output layer.
    torch.manual_seed(seed)
    network = CharacterLevelCNN(CharacterCNNConfig(10, initialization="stable_classifier_v1"), 2).train()
    x = one_hot(16, 64)
    labels = torch.arange(16) % 2
    logits = network(x)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    assert float(logits.softmax(1).max(1).values.mean().detach()) < 0.8
    assert 0.4 < float(loss.detach()) < 1.2
    loss.backward()
    assert torch.isfinite(network.conv1[0].weight.grad).all()
    assert network.conv1[0].weight.grad.abs().sum() > 0



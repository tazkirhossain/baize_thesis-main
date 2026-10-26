"""Strict optimizer contracts, historical behavior and fresh local state."""
from copy import deepcopy
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
from src.client.optimizers import make_optimizer, optimizer_settings
from src.server.config import contract_hash, validate_config
from src.server.experiment_config import read_json, resolve_experiment

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize('settings', [True, {}, {'name':'adam'},
    {'name':'adamw','weight_decay':-1}, {'name':'adamw','weight_decay':True},
    {'name':'adamw','weight_decay':float('nan')},
    {'name':'adamw','weight_decay':0,'momentum':.9},
    {'name':'sgd','weight_decay':0}, {'name':'sgd','weight_decay':0,'momentum':1},
    {'name':'sgd','weight_decay':0,'momentum':float('inf')}])
def test_invalid_optimizer_fails_before_training(settings):
    request = read_json(ROOT/'config/experiment.json')
    request['optimizer'] = settings
    with pytest.raises(ValueError, match='optimizer'):
        resolve_experiment(request, ROOT)


def test_legacy_sgd_matches_original_update_exactly_and_new_state_is_local():
    weights = [torch.nn.Parameter(torch.tensor([1., -2.])) for _ in range(2)]
    legacy = torch.optim.SGD([weights[0]], lr=.001)
    current = make_optimizer([weights[1]], learning_rate=.001)
    for _ in range(3):
        for weight, optimizer in zip(weights, (legacy, current)):
            optimizer.zero_grad()
            weight.square().sum().backward()
            optimizer.step()
    torch.testing.assert_close(weights[0], weights[1], rtol=0, atol=0)
    assert optimizer_settings() == dict(name='sgd', momentum=0., weight_decay=0.)
    settings = dict(name='adamw', weight_decay=0.)
    optimizer = make_optimizer([weights[0]], learning_rate=.001, settings=settings)
    optimizer.zero_grad()
    weights[0].square().sum().backward()
    optimizer.step()
    assert optimizer.state
    # A new client/round optimizer never inherits another local Adam state.
    fresh = make_optimizer([weights[0]], learning_rate=.001, settings=settings)
    assert not fresh.state


def test_adamw_production_updates_match_the_measured_diagnostic_exactly():
    parameters = [torch.nn.Parameter(torch.tensor([1., -2.])) for _ in range(2)]
    diagnostic = torch.optim.AdamW([parameters[0]], lr=.001, weight_decay=0.)
    production = make_optimizer([parameters[1]], learning_rate=.001,
                                settings={'name':'adamw','weight_decay':0.})
    for _ in range(4):
        for parameter, optimizer in zip(parameters, (diagnostic, production)):
            optimizer.zero_grad()
            parameter.square().sum().backward()
            optimizer.step()
    torch.testing.assert_close(parameters[0], parameters[1], rtol=0, atol=0)


@pytest.mark.parametrize('family', ['character_cnn', 'deberta_v3_base'])
def test_optimizer_is_captured_in_contract_and_input_is_not_mutated(family):
    request = read_json(ROOT/'config/experiment.json')
    request.update(model_name=family, learning_rate=None, optimizer=dict(name='adamw',weight_decay=0.))
    before = deepcopy(request)
    cfg = resolve_experiment(request, ROOT).configuration
    assert cfg['optimizer'] == request['optimizer']
    assert request == before
    baseline = deepcopy(cfg)
    baseline.pop('optimizer')
    validate_config(baseline)
    assert contract_hash(cfg) != contract_hash(baseline)


def test_actual_character_cnn_adamw_five_client_rounds_and_selected_test(tmp_path):
    import json
    import numpy as np
    pytest.importorskip('flwr')
    from scripts.federated import synthetic_clients
    from src.client.flower_client import URLFlowerClient
    from src.models.federated_state import make_models, state_hash
    from src.server.flower_strategy import FiveClientFedAvg
    from src.server.local_transport import run_local
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        request = read_json(ROOT/'config/experiment.json')
        cfg = resolve_experiment(request, ROOT).configuration
        cfg.update(rounds=2, batch_size=8, local_epochs=1, evaluate_training=False,
                   evaluate_final_test=True, progress_interval_seconds=0)
        cfg['optimizer'] = {'name':'adamw','weight_decay':0.0}
        cfg['balancing']['method'] = 'undersampling'
        cfg['model'].update(sequence_length=32, initialization='stable_classifier_v1')
        cfg['experiment'].update(data_kind='synthetic', partition_strategy='unknown',
                                 evaluation_population_id='adamw-synthetic-integration')
        synthetic_clients(tmp_path/'clients', selection_ready=True)
        contents = {p:p.read_bytes() for p in (tmp_path/'clients').rglob('*.jsonl')}
        strategy = FiveClientFedAvg(cfg, make_models(cfg), output=tmp_path/'server')
        initial_hash = state_hash(strategy.current)
        initial_encoder = strategy.current[0].copy()
        clients = [URLFlowerClient(cid, tmp_path/'clients'/f'client_{cid:03d}',cfg)
                   for cid in range(5)]
        run_local(strategy, clients)
        assert state_hash(strategy.current) != initial_hash
        assert not np.array_equal(initial_encoder, strategy.current[0])
        assert len(strategy.round_reports) == 2
        for report in strategy.round_reports.values():
            assert len(report['clients']) == 5
            assert all(c['used_adult'] == c['used_non_adult'] for c in report['clients'])
            assert report['val']['n_records'] == 20
        final = strategy.round_reports[2]
        assert final['test']['threshold'] == strategy.best['threshold']
        metadata = json.loads((tmp_path/'server/best_model.json').read_text())
        assert metadata['configuration']['optimizer']['name'] == 'adamw'
        assert metadata['decision_threshold'] == strategy.best['threshold']
        assert all(path.read_bytes() == original for path,original in contents.items())
    finally:
        torch.set_num_threads(old_threads)

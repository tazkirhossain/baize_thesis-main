"""One-file orchestration chooses datasets/settings without changing ML semantics."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_experiment as launcher
from src.server.experiment_config import (PARTITIONS, check_plan, read_json, resolve_experiment)

ROOT = Path(__file__).parents[1]


def request():
    return read_json(ROOT/'config/experiment.json')


@pytest.mark.parametrize('family', ['character_cnn', 'deberta_v3_base'])
@pytest.mark.parametrize('partition', ['iid', 'label_skew', 'category_skew'])
@pytest.mark.parametrize('method', ['none', 'undersampling', 'oversampling', 'smote', 'ensemble_undersampling'])
@pytest.mark.parametrize('protocol', ['end_to_end', 'encoder_embeddings'])
def test_model_partition_balance_combinations_and_explicit_protocol(family, partition, method, protocol):
    value = request()
    value.update(model_name=family, partition_type=partition, balancing_method=method, training_protocol=protocol,
                 rounds=7, local_epochs=2, batch_size=16, sequence_length=128, seed=123, learning_rate=None)
    before = deepcopy(value)
    if method == 'smote' and protocol == 'end_to_end':
        with pytest.raises(ValueError, match='encoder_embeddings'):
            resolve_experiment(value, ROOT)
        return
    plan = resolve_experiment(value, ROOT)
    cfg = plan.configuration
    assert cfg['model']['family'] == family and cfg['balancing']['method'] == method
    assert cfg['rounds'] == 7 and cfg['local_epochs'] == 2 and cfg['batch_size'] == 16
    assert cfg['balancing']['seed'] == cfg['seed'] == 123
    assert cfg['learning_rate'] == (0.001 if family == 'character_cnn' else 0.00002)
    assert cfg['model']['sequence_length' if family == 'character_cnn' else 'max_length'] == 128
    assert cfg['experiment']['partition_strategy'] == partition
    assert plan.partition_root == ROOT/'data'/PARTITIONS[partition]
    assert ('representation' in cfg) == (protocol == 'encoder_embeddings')
    assert cfg['aggregation_weight'] == 'original_training'
    assert value == before


@pytest.mark.parametrize('patch', [dict(model_name='unknown'), dict(partition_type='non_iid'),
    dict(training_protocol='auto'), dict(rounds=0), dict(batch_size=True), dict(n_clients=6),
    dict(threads=0), dict(generate_plots='yes'), dict(device='gpu'), dict(learning_rate=-1),
    dict(warmup_epochs=0), dict(sequence_length=3), dict(evaluation_population_id=''),
    dict(partition_root='data/custom'), dict(unknown_parameter=3)])
def test_invalid_or_ambiguous_settings_fail_before_run(patch):
    value = request()
    value.update(patch)
    with pytest.raises((ValueError, TypeError)):
        resolve_experiment(value, ROOT)


def test_custom_source_requires_declared_identity_and_explicit_lr():
    value = request()
    value.update(partition_root='data/custom-iid', evaluation_population_id='audited-custom-population-v1',
                 learning_rate=.0003, adult_to_non_adult='1:5')
    cfg = resolve_experiment(value, ROOT).configuration
    assert cfg['learning_rate'] == .0003
    assert cfg['experiment']['evaluation_population_id'] == 'audited-custom-population-v1'
    assert cfg['balancing']['adult_to_non_adult'] == '1:5'


@pytest.mark.parametrize('policy', ['legacy_fan_out', 'stable_classifier_v1'])
def test_cnn_initialization_is_explicit_in_contract_but_not_deberta(policy):
    from src.server.config import contract_hash, validate_config
    value = request()
    value['cnn_initialization'] = policy
    cnn = resolve_experiment(value, ROOT).configuration
    assert cnn['model']['initialization'] == policy
    changed = deepcopy(cnn)
    changed['model']['initialization'] = 'legacy_fan_out' if policy == 'stable_classifier_v1' else 'stable_classifier_v1'
    assert contract_hash(cnn) != contract_hash(changed)
    value['model_name'] = 'deberta_v3_base'
    assert 'initialization' not in resolve_experiment(value, ROOT).configuration['model']
    value['cnn_initialization'] = 'unknown'
    with pytest.raises(ValueError, match='cnn_initialization'):
        resolve_experiment(value, ROOT)
    changed['model']['initialization'] = 'unknown'
    with pytest.raises(ValueError, match='initialization'):
        validate_config(changed)


def test_missing_initialization_preserves_legacy_config_and_training_selection_policy():
    value = request()
    value.pop('cnn_initialization', None)
    cfg = resolve_experiment(value, ROOT).configuration
    assert 'initialization' not in cfg['model']
    assert cfg['threshold_selection'] == value['threshold_selection']


def test_duplicate_json_keys_are_rejected(tmp_path):
    source = tmp_path/'bad.json'
    source.write_text('{"rounds": 2, "rounds": 10}')
    with pytest.raises(ValueError, match='duplicate'):
        read_json(source)


def write_client(root, cid, partition='iid'):
    directory = root/f'client_{cid:03d}'
    splits = directory/'splits'
    splits.mkdir(parents=True)
    (directory/'manifest.json').write_text(json.dumps({'client_id': cid}))
    (splits/'manifest.json').write_text(json.dumps({'client_id':cid, 'splits':{
        name: {'n_records':2, 'sha256':'0'*64} for name in ('train','val','test')}}))
    (splits/'preprocessing.json').write_text(json.dumps({'partitioning':{'strategy':partition}}))
    for name in ('train','val','test'):
        (splits/f'{name}.jsonl').write_text('private fixture not parsed by preflight')
    return directory


def test_selected_strategy_must_match_actual_client_metadata(tmp_path):
    value = request()
    value.update(partition_root=str(tmp_path), evaluation_population_id='test-fixture-not-a-result')
    plan = resolve_experiment(value, ROOT)
    (tmp_path/'report.json').write_text(json.dumps({'status':'PASS','partitioning':{'name':'iid'}}))
    for cid in range(5):
        write_client(tmp_path, cid)
    check_plan(plan)
    bad = tmp_path/'client_003/splits/preprocessing.json'
    bad.write_text(json.dumps({'partitioning':{'strategy':'label_skew'}}))
    with pytest.raises(ValueError, match='distribution'):
        check_plan(plan)


def test_server_preflight_never_opens_client_data_and_client_uses_only_own_folder(tmp_path):
    value = request()
    value.update(partition_root=str(tmp_path/'missing-other-clients'), evaluation_population_id='test-only')
    plan = resolve_experiment(value, ROOT)
    check_plan(plan, mode='server')
    own = write_client(tmp_path/'device', 3)
    check_plan(plan, mode='client', client_id=3, client_directory=own)
    with pytest.raises(ValueError, match='ownership'):
        check_plan(plan, mode='client', client_id=2, client_directory=own)


def test_output_is_fresh_immediate_ignored_child_and_commands_reuse_runner():
    plan = resolve_experiment(request(), ROOT)
    output = launcher.output_path(plan)
    assert output.parent == ROOT/'data' and not output.exists()
    for bad in ('README.md', 'data', 'data/client-iid-v1', 'data/client-iid-v1/nested'):
        with pytest.raises(ValueError, match='output'):
            launcher.output_path(plan, bad)
    args = SimpleNamespace(mode='local')
    command = launcher.training_command(plan, ROOT/'data/test-resolved.json', args, output)
    assert command[1] == str(ROOT/'scripts/federated.py')
    assert command[command.index('--partition-root')+1] == str(ROOT/'data/client-iid-v1')


def test_check_does_not_create_artifacts_or_launch_training(tmp_path, monkeypatch, capsys):
    value = request()
    source = tmp_path/'experiment.json'
    source.write_text(json.dumps(value))
    monkeypatch.setattr(launcher, 'check_plan', lambda *a, **k: None)
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **k: pytest.fail('check launched training'))
    target = ROOT/'data/check-must-not-create-this-folder'
    assert not target.exists()
    assert launcher.main(['--config', str(source), '--check', '--output', str(target)]) == 0
    assert not target.exists()
    assert 'No training' in capsys.readouterr().out


@pytest.mark.parametrize('outcome', ['success', 'training_failure', 'plot_failure', 'interrupt'])
def test_launcher_provenance_auto_plots_and_failure_boundaries(tmp_path, monkeypatch, outcome):
    value = request()
    value['adult_to_non_adult'] = '1:5'
    source = tmp_path/'request.json'
    source.write_text(json.dumps(value))
    target = tmp_path/'result'
    monkeypatch.setattr(launcher, 'check_plan', lambda *a, **k: None)
    monkeypatch.setattr(launcher, 'output_path', lambda *a, **k: target)
    calls = []

    def subprocess_run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            cfg = read_json(command[command.index('--config')+1])
            target.mkdir()
            (target/'experiment.json').write_text(json.dumps(cfg))
            if outcome == 'interrupt':
                raise KeyboardInterrupt()
            return SimpleNamespace(returncode=3 if outcome == 'training_failure' else 0)
        settings = read_json(command[command.index('--config')+1])
        assert settings['expected_adult_to_non_adult'] == '1:5'
        assert command[1] == str(ROOT/'scripts/report_evaluation.py')
        return SimpleNamespace(returncode=5 if outcome == 'plot_failure' else 0)

    monkeypatch.setattr(launcher.subprocess, 'run', subprocess_run)
    status = launcher.main(['--config', str(source), '--mode', 'smoke'])
    assert status == {'success':0, 'training_failure':3, 'plot_failure':5, 'interrupt':130}[outcome]
    assert len(calls) == (1 if outcome in ('interrupt', 'training_failure') else 2)
    assert read_json(target/'experiment-request.json') == value
    provenance = read_json(target/'launcher_reproduction.json')
    assert len(provenance['source_sha256']) == 3
    assert (target/'launcher_snapshot/config/federated/character_cnn.json').read_bytes() == (ROOT/'config/federated/character_cnn.json').read_bytes()

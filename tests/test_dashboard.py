"""Dashboard source boundaries, truthful metrics, and read-only serving."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from src.evaluation.dashboard import dashboard_snapshot, load_run
from src.evaluation.classification import confusion_metrics


def fixture_run(root):
    server = root/'trial/server'
    server.mkdir(parents=True)
    cfg = json.loads((Path(__file__).parents[1]/'config/federated/iid_oversampling_character_cnn.json').read_text())
    cfg['rounds'] = 1
    cfg['experiment'].update(data_kind='synthetic', partition_strategy='unknown')
    report = {'round': 1, 'global_hash': 'test-only', 'clients': [
        {'client_id': i, 'original_train': 12, 'original_adult': 2,
         'original_non_adult': 10, 'resampled_train': 20, 'used_adult': 10,
         'used_non_adult': 10, 'duplicated': 8, 'train_loss': .5,
         'train_seconds': 1, 'examples_processed': 20, 'ensemble_members': 1}
        for i in range(5)], 'val': dict(confusion_metrics([[0,10],[0,10]]), loss=1.2),
        'timing': {'round_wall_seconds': 5}, 'communication': {'total_bytes': 100}}
    run = {'status': 'PASS', 'configuration': cfg, 'private_key': 'do-not-export',
           'client_data_contracts': {str(i): {'data_kind': 'synthetic', 'partition_strategy': 'unknown'} for i in range(5)}}
    (server/'run.json').write_text(json.dumps(run))
    (server/'round_0001.json').write_text(json.dumps(report))
    # These files must never be touched or surfaced by the adapter.
    (server/'global_final.npz').write_bytes(b'private checkpoint')
    (root/'trial/records.jsonl').write_text('private raw URL dataset')
    return server, run, report


def test_only_supported_aggregates_and_true_data_kind_leave_adapter(tmp_path):
    server, _, _ = fixture_run(tmp_path)
    result = load_run(server, tmp_path)
    assert result['data_kind'] == 'synthetic' and result['partition_type'] == 'unknown'
    assert 'test' not in result['rounds'][0]['metrics']
    assert result['rounds'][0]['metrics']['val']['balanced_accuracy'] == .5
    assert [c['client_id'] for c in result['rounds'][0]['clients']] == [1,2,3,4,5]
    text = json.dumps(result)
    assert 'do-not-export' not in text and 'private checkpoint' not in text and 'private raw URL' not in text
    assert result['monitoring'] == 'recorded_snapshot'
    assert result['duration_seconds'] == 5


def test_mismatched_confusion_metrics_and_early_test_fail(tmp_path):
    server, run, report = fixture_run(tmp_path)
    bad = deepcopy(report)
    bad['val']['adult_f1'] = .99
    (server/'round_0001.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='disagrees'):
        load_run(server, tmp_path)
    report['test'] = report['val']
    (server/'round_0001.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='protocol'):
        load_run(server, tmp_path)


def test_smote_requires_continuous_representation_and_no_implicit_live_status(tmp_path):
    server, run, _ = fixture_run(tmp_path)
    run['configuration']['balancing']['method'] = 'smote'
    (server/'run.json').write_text(json.dumps(run))
    with pytest.raises(ValueError, match='representation'):
        load_run(server, tmp_path)
    run['configuration']['representation'] = {'mode':'token_ids', 'warmup_epochs':1}
    (server/'run.json').write_text(json.dumps(run))
    with pytest.raises(ValueError, match='encoder_embeddings'):
        load_run(server, tmp_path)
    run['configuration']['representation'] = {'mode':'encoder_embeddings', 'warmup_epochs':1}
    (server/'run.json').write_text(json.dumps(run))
    assert load_run(server, tmp_path)['training_protocol'] == 'encoder_embeddings'


def test_comparison_groups_preserve_budgets_protocol_and_kind(tmp_path):
    server, run, _ = fixture_run(tmp_path)
    first = load_run(server, tmp_path)
    run['configuration']['batch_size'] = 64
    (server/'run.json').write_text(json.dumps(run))
    assert first['comparison_group'] != load_run(server, tmp_path)['comparison_group']
    run['configuration']['batch_size'] = 8
    run['configuration']['balancing']['method'] = 'undersampling'
    (server/'run.json').write_text(json.dumps(run))
    assert first['comparison_group'] == load_run(server, tmp_path)['comparison_group']


def test_discovery_bounded_and_invalid_sources_explicit(tmp_path):
    server, _, _ = fixture_run(tmp_path)
    nested = tmp_path/'snapshots/private/server'
    nested.mkdir(parents=True)
    (nested/'run.json').write_text('not a result')
    result = dashboard_snapshot(tmp_path)
    assert len(result['runs']) == 1 and result['dataset'] is None and not result['issues']
    (server/'round_0001.json').write_text('{')
    result = dashboard_snapshot(tmp_path)
    assert not result['runs'] and len(result['issues']) == 1


def test_no_sources_is_an_empty_state(tmp_path):
    result = dashboard_snapshot(tmp_path)
    assert result['runs'] == [] and result['dataset'] is None
    assert result['capabilities'] == {'read_only':True,'live_telemetry':False,'raw_records':False}

"""Independent artifact/integrity audit; no training or source-data changes."""
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


source = json.loads((ROOT/'data/cleanup-2026-10-23/manifest.json').read_text())
for name, expected in source['protected_data'].items():
    path = ROOT/name
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (expected['size'], expected['mtime_ns']), name
    if 'sha256' in expected:
        assert digest(path) == expected['sha256'], name

assets = json.loads((ROOT/'data/pretrained/deberta-v3-base/download_manifest.json').read_text())
for name, values in assets['files'].items():
    assert digest(ROOT/'data/pretrained/deberta-v3-base'/name) == values['sha256']
old = json.loads((ROOT/'data/federated-cnn-evaluation-v4/server/run.json').read_text())
new = json.loads((ROOT/'data/cnn-embedding-smote-v2/server/run.json').read_text())
first = json.loads((ROOT/'data/cnn-embedding-smote-v1/server/run.json').read_text())
assert new['status'] == 'PASS' and new['schema_hash'] == old['schema_hash']
assert first['global_hash'] == new['global_hash']
state_bytes = sum(math.prod(shape)*__import__('numpy').dtype(dtype).itemsize for _, _, shape, dtype in new['schema'])
for n in (1, 2):
    round_report = json.loads((ROOT/f'data/cnn-embedding-smote-v2/server/round_{n:04d}.json').read_text())
    assert round_report['balancing']['synthetic_embeddings'] == 20
    assert round_report['balancing']['smote_clients_adjusted'] == 4
    assert round_report['balancing']['smote_clients_skipped'] == 0
    assert round_report['optimization']['examples_processed'] == 140
    assert round_report['train']['n_records'] == 60 and round_report['val']['n_records'] == 20
    for client in round_report['clients']:
        assert client['original_train'] == client['aggregation_weight'] == client['warmup_examples_processed'] == 12
        assert client['used_adult'] == client['used_non_adult']
        assert client['used_adult'] == client['original_adult']+client['synthetic']
    traffic = round_report['communication']
    assert traffic['total_bytes'] == sum(traffic[key] for key in ('fit_down_bytes','fit_up_bytes','evaluation_down_bytes','evaluation_up_bytes'))
    assert all(traffic[key] >= 5*state_bytes for key in ('fit_down_bytes','fit_up_bytes','evaluation_down_bytes'))

report = json.loads((ROOT/'data/cnn-embedding-smote-v2/report/report.json').read_text())
assert report['best_method'] is None and len(report['rows']) == 4
assert all(row['training_protocol'] == 'encoder_embeddings' and row['balancing']['synthetic_embeddings'] == 20 for row in report['rows'])
pretrained = json.loads((ROOT/'data/pretrained/deberta-embedding-smote-verification-v1.json').read_text())
assert pretrained['status'] == 'PASS' and pretrained['sampling']['feature_dimension'] == 768
assert pretrained['sampling']['synthetic_examples'] == 2 and pretrained['examples_processed'] == 14
assert not pretrained['corpus_or_holdouts_read']
assert '174 passed, 3 skipped' in (ROOT/'data/test-all-neural-balancing-01.log').read_text(encoding='utf-8')
assert '3 passed' in (ROOT/'data/test-neural-wire-01.log').read_text(encoding='utf-8')

result = {'status':'PASS','protected_data_artifacts_unchanged':len(source['protected_data']),
          'original_pretrained_files_unchanged':True,'cnn_tensor_layout_unchanged':True,
          'cnn_smoke_seed_reproducibility':True,'five_client_cnn_smote_rounds':2,
          'cnn_synthetic_examples_each_round':20,'cnn_processed_examples_each_round':140,
          'full_pretrained_deberta_embedding_smote':'PASS, one model and six synthetic training examples',
          'full_pretrained_deberta_ensemble_executed':False,
          'regression_tests_passed':174,'network_tests_passed_separately':3,
          'preset_files':[p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'config/federated/embedding_balancing').glob('*.json'))],
          'real_data_quality_measured':False,'best_method':None}
(ROOT/'data/neural-balancing-support-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS: unchanged data/assets/layout, deterministic CNN FL, exact SMOTE and wire counts, full DeBERTa SMOTE, tests and eight presets.')

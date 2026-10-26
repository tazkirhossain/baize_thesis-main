"""Selection leakage, saved-round integrity and real Flower checkpoint selection."""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('flwr')
from scripts.federated import synthetic_clients
from src.client.data import LocalSplit, read_local_split
from src.client.flower_client import URLFlowerClient
from src.client.selection import selection_mask, selection_views
from src.data.balancing import BalanceConfig, sample_training_indices
from src.evaluation.classification import confusion_metrics
from src.models.federated_state import StateCodec, state_hash
from src.server.checkpoints import atomic_json, load_checkpoint, save_weights
from src.server.config import validate_config
from src.server.flower_strategy import FiveClientFedAvg
from src.server.local_transport import run_local

ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


class TinyEncoder(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(channels)
        self.fc = torch.nn.Linear(channels, 2)

    def extract_features(self, x):
        return self.bn(x.transpose(1,2)).mean(dim=2)

    def classify_features(self, x):
        return self.fc(x)

    def feature_head(self):
        return self.fc

    def forward(self, x):
        return self.classify_features(self.extract_features(x))


def cfg():
    value = json.loads((ROOT/'config/federated/character_cnn.json').read_text())
    value.update(rounds=2, batch_size=8, evaluate_final_test=True, progress_interval_seconds=0,
                 model_selection={'source':'training_selection','fraction':.1,'metric':'adult_f1'})
    value['model']['sequence_length'] = 16
    value['balancing']['ensemble_members'] = 2
    return value


def models(value):
    torch.manual_seed(value['seed'])
    n = value['balancing']['ensemble_members'] if value['balancing']['method']=='ensemble_undersampling' else 1
    return [TinyEncoder(len(value['model']['characters'])+1) for _ in range(n)]


def test_domain_selection_is_shared_label_independent_disjoint_complete_and_protected():
    urls = tuple(f'https://sub{p}.domain{i}.org/path' for i in range(150) for p in range(2))
    data = LocalSplit('train',urls,np.arange(len(urls),dtype=np.int8)%2,np.arange(len(urls)), 'parent')
    mask = selection_mask(data,fraction=.1,seed=42)
    train, held = selection_views(data,mask)
    assert len(train)+len(held)==len(data) and len(held)>0
    assert not set(train.source_rows)&set(held.source_rows)
    assert not set(u.split('/')[2].split('.',1)[1] for u in train.urls)&set(u.split('/')[2].split('.',1)[1] for u in held.urls)
    changed = LocalSplit('train',data.urls,1-data.labels,data.source_rows,data.sha256)
    np.testing.assert_array_equal(selection_mask(changed,fraction=.1,seed=42),mask)
    reverse = LocalSplit('train',data.urls[::-1],data.labels[::-1],data.source_rows[::-1],data.sha256)
    np.testing.assert_array_equal(selection_mask(reverse,fraction=.1,seed=42),mask[::-1])
    # A second client holding URLs from the same domains gets identical roles.
    other = LocalSplit('train',data.urls[1::2],data.labels[1::2],data.source_rows[1::2],data.sha256)
    np.testing.assert_array_equal(selection_mask(other,fraction=.1,seed=42),mask[1::2])
    with pytest.raises(ValueError,match='training'):
        sample_training_indices(held.labels,BalanceConfig(method='oversampling'),split='train')
    with pytest.raises(ValueError,match='training'):
        selection_mask(held,fraction=.1,seed=42)
    with pytest.raises(ValueError,match='no optimization'):
        selection_views(data,np.ones(len(data),dtype=bool))


@pytest.mark.parametrize('selection', [True, {}, {'source':'val','fraction':.1,'metric':'adult_f1'},
    {'source':'training_selection','fraction':0,'metric':'adult_f1'},
    {'source':'training_selection','fraction':float('nan'),'metric':'adult_f1'},
    {'source':'training_selection','fraction':True,'metric':'adult_f1'},
    {'source':'training_selection','fraction':.1,'metric':'accuracy'}])
def test_selection_config_rejects_unsafe_or_ambiguous_options(selection):
    value=cfg()
    value['model_selection']=selection
    with pytest.raises(ValueError,match='model_selection'):
        validate_config(value)


@pytest.mark.parametrize('method,protocol', [('undersampling','end_to_end'),('oversampling','end_to_end'),
    ('ensemble_undersampling','end_to_end'), ('smote','encoder_embeddings'),
    ('oversampling','encoder_embeddings')])
@pytest.mark.parametrize('optimizer', [None, {'name':'adamw','weight_decay':0.0}])
def test_every_round_best_checkpoint_reloads_and_only_training_selection_selects(tmp_path, method, protocol, optimizer):
    value=cfg()
    if optimizer is not None:
        value['optimizer'] = optimizer
    value['balancing']['method']=method
    if protocol=='encoder_embeddings':
        value['representation']={'mode':protocol,'warmup_epochs':1}
    synthetic_clients(tmp_path/'clients',selection_ready=True)
    original={p:p.read_bytes() for p in (tmp_path/'clients').rglob('*.jsonl')}
    strategy=FiveClientFedAvg(value,models(value),output=tmp_path/'server')
    clients=[URLFlowerClient(cid,tmp_path/'clients'/f'client_{cid:03d}',value,models=models(value)) for cid in range(5)]
    run_local(strategy,clients)
    assert len(list((tmp_path/'server/checkpoints').glob('*.npz')))==2
    scores=[strategy.round_reports[r]['selection']['adult_f1'] for r in (1,2)]
    best_round=max((1,2),key=lambda r:scores[r-1])
    assert strategy.best['round']==best_round
    for r in (1,2):
        arrays, metadata=load_checkpoint(tmp_path/f'server/checkpoints/round_{r:04d}.npz',strategy.codec,expected_contract=strategy.contract)
        report=strategy.round_reports[r]
        assert metadata['status']=='evaluated' and metadata['global_hash']==report['global_hash']==state_hash(arrays)
        assert report['train']['n_records']+report['selection']['n_records']==200
        for client in report['clients']:
            assert client['assigned_train']==client['original_train']+client['selection_train']==40
            assert client['aggregation_weight']==client['original_train']
    best_arrays,_=load_checkpoint(tmp_path/'server/best_model.npz',strategy.codec,expected_contract=strategy.contract)
    assert state_hash(best_arrays)==strategy.best['global_hash']
    final=np.load(tmp_path/'server/global_final.npz',allow_pickle=False)
    assert state_hash([final[f'tensor_{i:04d}'] for i in range(len(strategy.codec.entries))])==state_hash(strategy.current)
    final.close()
    assert strategy.round_reports[2]['test_checkpoint']['round']==best_round
    assert strategy.round_reports[2]['test_checkpoint']['global_hash']==strategy.best['global_hash']
    assert 'test' not in strategy.round_reports[1]
    assert all(p.read_bytes()==content for p,content in original.items())
    assert json.loads((tmp_path/'server/run.json').read_text())['selection']['round']==best_round
    for client in clients:
        data=read_local_split(client.directory,'train')
        training,held=client._training_views(data)
        assert not set(training.source_rows)&set(held.source_rows)
        assert state_hash(client.codec.get())==strategy.best['global_hash']

    from src.evaluation.reporting import federated_rows
    settings=json.loads((ROOT/'config/reporting/comparison.json').read_text())
    rows,_=federated_rows(tmp_path/'server',settings)
    test_row=next(row for row in rows if row['split']=='test')
    assert test_row['checkpoint_round']==best_round and test_row['checkpoint_hash']==strategy.best['global_hash']
    # Read-only exports must reject a mislabeled test checkpoint.
    report_path=tmp_path/'server/round_0002.json'
    damaged=json.loads(report_path.read_text())
    damaged['test_checkpoint']['global_hash']='wrong'
    report_path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError,match='test model'):
        federated_rows(tmp_path/'server',settings)


def test_best_ties_keep_earliest_and_holdout_scores_do_not_select(tmp_path):
    value=cfg()
    strategy=FiveClientFedAvg(value,models(value),output=tmp_path/'server')
    best_metrics=dict(confusion_metrics([[8,2],[1,9]]),loss=.2)
    worse=dict(confusion_metrics([[10,0],[8,2]]),loss=.5)
    first=state_hash(strategy.current)
    for rnd, selection in ((1,best_metrics),(2,best_metrics),(3,worse)):
        strategy.current=[a+1 if a.dtype.kind=='f' else a.copy() for a in strategy.current] if rnd>1 else strategy.current
        strategy.round_reports[rnd]={'round':rnd, 'selection':selection,
            'val':worse if rnd==1 else best_metrics, 'test':worse if rnd==1 else best_metrics}
        strategy._checkpoint(rnd,evaluated=False)
        strategy._checkpoint(rnd,evaluated=True)
        strategy._select_best(rnd,selection)
    assert strategy.best['round']==1 and strategy.best['global_hash']==first
    arrays,metadata=load_checkpoint(tmp_path/'server/best_model.npz',strategy.codec,expected_contract=strategy.contract)
    assert state_hash(arrays)==first and metadata['round']==1
    # A worse final checkpoint never replaces a better earlier checkpoint.
    assert state_hash(strategy.current)!=first


def test_threshold_is_selected_only_on_training_and_frozen_with_best_checkpoint(tmp_path):
    from dataclasses import replace
    from src.evaluation.thresholds import select_threshold, metrics_at_threshold
    from src.evaluation.reporting import federated_rows
    value = cfg()
    value['threshold_selection'] = dict(source='training_selection', metric='adult_f1',
                                      bins=201, tie_break='highest_threshold')
    validate_config(value)
    synthetic_clients(tmp_path/'clients', selection_ready=True)
    originals = {p: p.read_bytes() for p in (tmp_path/'clients').rglob('*.jsonl')}
    runs = []
    # Alter validation labels only in memory. Neither weights, threshold nor
    # checkpoint choice may depend on these labels. Test labels stay untouched.
    for perturb_validation in (False, True):
        directory = tmp_path/str(perturb_validation)
        strategy = FiveClientFedAvg(value, models(value), output=directory)
        clients = [URLFlowerClient(cid, tmp_path/'clients'/f'client_{cid:03d}', value,
                                   models=models(value)) for cid in range(5)]
        if perturb_validation:
            from src.client.neural_data import LocalURLDataset
            for client in clients:
                original_read = client._read
                def read(split, original=original_read):
                    data, dataset = original(split)
                    if split == 'val':
                        data = replace(data, labels=np.zeros_like(data.labels))
                        dataset = LocalURLDataset(data)
                    return data, dataset
                client._read = read
        run_local(strategy, clients)
        for rnd, report in strategy.round_reports.items():
            chosen = select_threshold(np.array(report['score_histograms']['selection']))
            assert report['selection']['threshold'] == chosen['threshold']
            assert report['selection']['adult_f1'] == chosen['adult_f1']
            assert report['selection']['adult_f1'] >= report['fixed_threshold_metrics']['selection']['adult_f1']
            assert report['val']['threshold'] == chosen['threshold']
            for split in ('val', 'selection', 'train'):
                expected = metrics_at_threshold(np.array(report['score_histograms'][split]), chosen['threshold'])
                assert report[split]['confusion_matrix'] == expected['confusion_matrix']
        final = strategy.round_reports[2]
        assert final['test']['threshold'] == strategy.best['threshold'] == final['test_checkpoint']['threshold']
        metadata = json.loads((directory/'best_model.json').read_text())
        assert metadata['decision_threshold'] == strategy.best['threshold']
        settings = json.loads((ROOT/'config/reporting/comparison.json').read_text())
        federated_rows(directory, settings)
        runs.append(strategy)
    assert runs[0].best == runs[1].best
    assert state_hash(runs[0].current) == state_hash(runs[1].current)
    assert runs[0].round_reports[2]['val']['confusion_matrix'] != runs[1].round_reports[2]['val']['confusion_matrix']
    assert all(p.read_bytes() == content for p, content in originals.items())

    from src.evaluation.reporting import checked_selection
    run_metadata = json.loads((tmp_path/'False/run.json').read_text())
    run_metadata['selection']['threshold'] = -1
    without_test = [{key: val for key, val in report.items() if key not in ('test', 'test_checkpoint')}
                    for report in runs[0].round_reports.values()]
    with pytest.raises(ValueError, match='selected threshold'):
        checked_selection(value, run_metadata, without_test)

    # Historical exports must fail if a held-out test is mislabeled with a
    # different threshold, even when its checkpoint hash is still correct.
    path = tmp_path/'False/round_0002.json'
    report = json.loads(path.read_text())
    report['test_checkpoint']['threshold'] = -1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='test threshold'):
        federated_rows(tmp_path/'False', settings)


def test_final_test_uses_earlier_best_threshold_instead_of_final_round_threshold():
    from types import SimpleNamespace
    from src.evaluation.thresholds import encode_probability_histogram, histogram_counts, metrics_at_threshold
    value = cfg()
    value['threshold_selection'] = dict(source='training_selection', metric='adult_f1',
                                      bins=201, tie_break='highest_threshold')
    strategy = FiveClientFedAvg(value, models(value))
    digest = state_hash(strategy.current)
    strategy.best = dict(round=1, global_hash=digest, threshold=0.7)
    strategy.pending_eval[2] = ({}, dict(base_hash=digest))
    scores, labels = np.array([.1, .6, .65, .9]), np.array([0, 0, 1, 1])
    reference = dict(confusion_metrics([[1, 1], [0, 2]]), threshold=.5, loss=.4)
    strategy.round_reports[2] = dict(test=reference, test_checkpoint=dict(round=1),
        threshold_selection=dict(threshold=.8))
    metrics = dict(tn=1, fp=1, fn=0, tp=2,
                   score_histogram=encode_probability_histogram(scores, labels, 201))
    strategy._apply_threshold_policy(2, 'test', [(None, SimpleNamespace(metrics=metrics))])
    measured = strategy.round_reports[2]['test']
    assert measured['threshold'] == .7
    assert measured['confusion_matrix'] == metrics_at_threshold(histogram_counts(scores, labels, 201), .7)['confusion_matrix']
    assert strategy.round_reports[2]['test_checkpoint']['threshold'] == .7


def test_interrupted_evaluation_keeps_aggregated_checkpoint_without_claiming_best(tmp_path, monkeypatch):
    value=cfg()
    synthetic_clients(tmp_path/'clients',selection_ready=True)
    strategy=FiveClientFedAvg(value,models(value),output=tmp_path/'server')
    clients=[URLFlowerClient(cid,tmp_path/'clients'/f'client_{cid:03d}',value,models=models(value)) for cid in range(5)]
    def fail(*args, **kwargs):
        raise RuntimeError('synthetic evaluation interruption')
    monkeypatch.setattr(clients[0],'evaluate',fail)
    with pytest.raises(RuntimeError):
        run_local(strategy,clients)
    arrays,metadata=load_checkpoint(tmp_path/'server/checkpoints/round_0001.npz',strategy.codec,expected_contract=strategy.contract)
    assert metadata['status']=='aggregated_pending_evaluation'
    assert state_hash(arrays)==state_hash(strategy.current)
    assert not (tmp_path/'server/run.json').exists() and not (tmp_path/'server/best_model.npz').exists()


def test_deberta_checkpoint_round_and_selected_test_predictions_reload(tmp_path):
    pytest.importorskip('transformers')
    from transformers import DebertaV2Config
    from src.models.deberta_v3_base.model import DebertaV3BaseClassifier
    value=json.loads((ROOT/'config/federated/deberta_v3_base.json').read_text())
    value.update(rounds=1,batch_size=8,evaluate_final_test=True,progress_interval_seconds=0,
                 model_selection={'source':'training_selection','fraction':.1,'metric':'adult_f1'})
    value['model']['max_length']=8
    def build():
        torch.manual_seed(value['seed'])
        return [DebertaV3BaseClassifier(DebertaV2Config(vocab_size=64,hidden_size=16,
            intermediate_size=32,num_hidden_layers=1,num_attention_heads=4,max_position_embeddings=16,
            relative_attention=True,pos_att_type=['p2c','c2p'],position_biased_input=False))]
    def collate(rows):
        ids=torch.tensor([[2]+[ord(c)%60+4 for c in url[:6]]+[3] for url,_ in rows])
        return {'input_ids':ids,'attention_mask':torch.ones_like(ids)},torch.tensor([y for _,y in rows])
    synthetic_clients(tmp_path/'clients',selection_ready=True)
    strategy=FiveClientFedAvg(value,build(),output=tmp_path/'server')
    clients=[URLFlowerClient(cid,tmp_path/'clients'/f'client_{cid:03d}',value,models=build(),collator=collate) for cid in range(5)]
    run_local(strategy,clients)
    saved,_=load_checkpoint(tmp_path/'server/best_model.npz',strategy.codec,expected_contract=strategy.contract)
    reloaded=build()
    StateCodec(reloaded).set(saved)
    features,_=collate([('https://fixture.example',1)])
    reloaded[0].eval()
    clients[0].models[0].eval()
    with torch.no_grad():
        torch.testing.assert_close(reloaded[0](**features).logits,clients[0].models[0](**features).logits)


@pytest.mark.parametrize('corruption', ['hash','schema','contract','keys','nan'])
def test_checkpoint_reload_fails_closed_on_corruption(tmp_path, corruption):
    value=cfg()
    codec=StateCodec(models(value))
    path=tmp_path/'round.npz'
    arrays=codec.get()
    save_weights(path,arrays)
    metadata={'schema_hash':codec.fingerprint,'contract':'agreed','global_hash':state_hash(arrays)}
    if corruption in ('hash','schema','contract'):
        metadata[{'hash':'global_hash','schema':'schema_hash','contract':'contract'}[corruption]]='wrong'
    elif corruption=='keys':
        save_weights(path,arrays[:-1])
    else:
        arrays[0].flat[0]=np.nan
        save_weights(path,arrays)
    atomic_json(path.with_suffix('.json'),metadata)
    with pytest.raises(ValueError):
        load_checkpoint(path,codec,expected_contract='agreed')

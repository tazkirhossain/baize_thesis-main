"""Progress is timed, clearly scoped, and does not change model updates."""
from copy import deepcopy
from pathlib import Path

import pytest
from src.client.progress import BatchProgress
from src.evaluation.classification import confusion_metrics
from src.evaluation.console import print_round_summary
from src.server.config import validate_config
from src.server.experiment_config import read_json, resolve_experiment

ROOT = Path(__file__).parents[1]


def test_time_interval_first_batch_completion_weighted_loss_and_eta():
    times = iter((0, 1, 2, 31, 32))
    lines = []
    progress = BatchProgress('Client 1 | training', total_batches=4, total_examples=7, epochs=2,
                             interval_seconds=30, clock=lambda:next(times), emit=lines.append)
    for size, loss in ((2,1),(2,2),(2,3),(1,4)):
        progress.update(size, loss=loss)
    assert len(lines) == 4  # start, first batch, interval, final batch
    assert 'START' in lines[0] and 'batch 1/4' in lines[1]
    assert 'batch 2/4' not in '\n'.join(lines)
    assert 'epoch 2/2' in lines[2] and 'batch 3/4' in lines[2]
    assert 'DONE' in lines[-1] and 'examples 7/7' in lines[-1]
    assert 'running loss=2.2857' in lines[-1] and 'phase ETA~00:00:00' in lines[-1]


def test_zero_disables_progress_and_no_loss_is_invented_for_extraction():
    lines = []
    progress = BatchProgress('extraction', total_batches=1, total_examples=2, interval_seconds=0, emit=lines.append)
    progress.update(2)
    assert lines == []
    progress = BatchProgress('extraction', total_batches=1, total_examples=2, emit=lines.append)
    progress.update(2)
    assert 'running loss' not in '\n'.join(lines)


@pytest.mark.parametrize('interval', [-1, True, '30', float('nan'), float('inf')])
def test_invalid_intervals_rejected_by_shared_contract_and_launcher(interval):
    cfg = read_json(ROOT/'config/federated/character_cnn.json')
    cfg['progress_interval_seconds'] = interval
    with pytest.raises(ValueError, match='progress_interval'):
        validate_config(cfg)
    value = read_json(ROOT/'config/experiment.json')
    value['progress_interval_seconds'] = interval
    with pytest.raises(ValueError, match='progress_interval'):
        resolve_experiment(value, ROOT)


def test_old_active_configs_remain_compatible_without_optional_progress():
    value = read_json(ROOT/'config/experiment.json')
    value.pop('progress_interval_seconds', None)
    plan = resolve_experiment(value, ROOT)
    assert plan.configuration.get('progress_interval_seconds',30) == 30


def test_round_summary_uses_measured_metrics_and_preserves_data(capsys):
    metrics = dict(confusion_metrics([[8,2],[3,7]]), loss=.25)
    report = {'round':2, 'val':metrics, 'train':metrics,
              'timing':{'round_wall_seconds':5,'client_training_seconds_sum':4},
              'communication':{'total_bytes':1048576}}
    before = deepcopy(report)
    print_round_summary(report, split='val', rounds=10)
    output = capsys.readouterr().out
    assert 'round 2/10' in output and 'Original train' in output and 'Validation' in output
    assert 'Adult:     precision=0.7778  recall=0.7000' in output
    assert 'Non-adult:' in output and 'Macro-F1' in output and 'Balanced' in output
    assert '[[8, 2], [3, 7]]' in output and '1.00 MiB' in output
    assert 'Final test' not in output
    assert report == before


@pytest.mark.parametrize('protocol', ['end_to_end', 'encoder_embeddings'])
def test_progress_leaves_flower_updates_metrics_and_holdouts_identical(tmp_path, protocol):
    pytest.importorskip('torch')
    pytest.importorskip('flwr')
    import torch
    from scripts.federated import synthetic_clients
    from src.client.flower_client import URLFlowerClient
    from src.models.federated_state import state_hash
    from src.server.flower_strategy import FiveClientFedAvg
    from src.server.local_transport import run_local

    class TinyClassifier(torch.nn.Module):
        def __init__(self, characters):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(characters)
            self.fc = torch.nn.Linear(characters, 2)

        def forward(self, x):
            return self.classify_features(self.extract_features(x))

        def extract_features(self, x):
            return self.bn(x.transpose(1, 2)).mean(dim=2)

        def classify_features(self, x):
            return self.fc(x)

        def feature_head(self):
            return self.fc

    def tiny_models(cfg):
        torch.manual_seed(cfg['seed'])
        return [TinyClassifier(len(cfg['model']['characters'])+1)]

    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        synthetic_clients(tmp_path/'clients')
        files = {p:p.read_bytes() for p in (tmp_path/'clients').rglob('*.jsonl')}
        cfg = read_json(ROOT/'config/federated/character_cnn.json')
        cfg.update(rounds=1,batch_size=4,evaluate_final_test=False)
        cfg['model']['sequence_length']=16
        if protocol == 'encoder_embeddings':
            cfg['representation'] = {'mode':protocol, 'warmup_epochs':1}
        states, metrics = [], []
        for interval in (0, .00001):
            cfg['progress_interval_seconds']=interval
            strategy=FiveClientFedAvg(cfg,tiny_models(cfg),output=tmp_path/f'output-{interval}')
            clients=[URLFlowerClient(cid,tmp_path/'clients'/f'client_{cid:03d}',cfg,models=tiny_models(cfg)) for cid in range(5)]
            run_local(strategy,clients)
            states.append(state_hash(strategy.current))
            metrics.append({s:strategy.round_reports[1][s] for s in ('train','val')})
        assert states[0]==states[1] and metrics[0]==metrics[1]
        assert all(p.read_bytes()==content for p,content in files.items())
    finally:
        torch.set_num_threads(previous)

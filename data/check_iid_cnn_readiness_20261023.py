"""Local read-only corpus audit plus one real training batch per client.

No full experiment, holdout tuning, URL printing or checkpoint saving.
Run from the repository root with the existing local Flower runtime on PYTHONPATH.
"""
import gc
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.client.data import read_local_split
from src.client.neural_data import make_collator
from src.data.balancing import BalanceConfig, sample_training_indices
from src.data.preprocessing import _file_hash
from src.models.federated_state import make_models
from src.server.config import validate_config

def require(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    output = ROOT/'data/iid-cnn-real-readiness-20261023-v1.json'
    require(not output.exists(), 'choose a new readiness report')
    started = time.perf_counter()
    cfg = validate_config(json.loads((ROOT/'config/federated/iid_oversampling_character_cnn.json').read_text()))
    require(cfg['balancing']['method'] == 'oversampling' and cfg['rounds'] == 2 and cfg['local_epochs'] == 1,
            'unexpected selected experiment')
    torch.set_num_threads(2)
    report = {'status': 'PASS', 'scope': 'immutable full IID splits and bounded real training batches; not full training or quality evaluation',
              'configuration': cfg, 'clients': [], 'runtime': {name: importlib.metadata.version(name)
               for name in ('torch', 'flwr', 'numpy', 'matplotlib')},
              'cuda_available': torch.cuda.is_available(), 'test_evaluated': False}
    partition = ROOT/'data/client-iid-v1'
    previous = json.loads((partition/'local_split_verification.json').read_text())
    require(previous['status'] == 'PASS', 'prior local split audit failed')
    seen = set()
    totals = dict.fromkeys(('train', 'val', 'test'), 0)
    for cid in range(5):
        directory = partition/f'client_{cid:03d}'
        client_manifest = json.loads((directory/'manifest.json').read_text())
        manifest = json.loads((directory/'splits/manifest.json').read_text())
        require(client_manifest['client_id'] == manifest['client_id'] == cid, 'client ownership mismatch')
        require(_file_hash(directory/'data.jsonl') == client_manifest['dataset']['sha256'] == manifest['source_sha256'],
                'original partition fingerprint mismatch')
        item = {'client': cid+1, 'splits': {}}
        for split in ('train', 'val', 'test'):
            data = read_local_split(directory, split)
            counts = np.bincount(data.labels, minlength=2)
            expected = manifest['splits'][split]
            require(counts.tolist() == [expected['class_counts'][str(i)] for i in range(2)], 'class count mismatch')
            require(np.all(counts > 0), 'missing class in existing split')
            rows = set(map(int, data.source_rows))
            require(len(rows) == len(data) and not rows.intersection(seen), 'duplicate original source row')
            seen.update(rows)
            require(len(data) == previous['clients'][cid]['counts'][split], 'split differs from prior audit')
            totals[split] += len(data)
            item['splits'][split] = {'n_records': len(data), 'Non-adult': int(counts[0]),
                                     'Adult': int(counts[1]), 'sha256': data.sha256}
            if split == 'train':
                selection = sample_training_indices(data.labels, BalanceConfig(**cfg['balancing']), split='train')
                require(selection.report['after']['Adult'] == selection.report['after']['Non-adult'], 'imbalance after sampling')
                require(np.array_equal(np.unique(selection.indices), np.arange(len(data))), 'oversampling dropped originals')
                item['balancing'] = selection.report
                # Read only real TRAIN examples; this small optimizer step checks
                # actual configured length 256, unlike synthetic smoke length 32.
                indices = np.random.default_rng(cfg['seed']+cid).choice(selection.indices, cfg['batch_size'], replace=False)
                features, labels = make_collator(cfg['model'])([(data.urls[i], int(data.labels[i])) for i in indices])
                model = make_models(cfg)[0].train()
                optimizer = torch.optim.SGD(model.parameters(), lr=cfg['learning_rate'])
                parameter = next(model.parameters())
                before = parameter.detach().clone()
                loss = torch.nn.functional.cross_entropy(model(**features), labels)
                require(bool(torch.isfinite(loss)), 'nonfinite real training loss')
                loss.backward()
                require(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()), 'nonfinite gradients')
                optimizer.step()
                require(not torch.equal(before, parameter), 'no model update')
                item['one_batch_check'] = {'n_examples': len(labels), 'sequence_length': features['x'].shape[1],
                                           'loss': float(loss.detach()), 'finite_gradients': True, 'weights_changed': True}
                del model, optimizer, before, parameter, loss, features, labels, selection
            require(_file_hash(directory/'splits'/f'{split}.jsonl') == data.sha256, 'source file changed during check')
            del rows, data
            gc.collect()
        require(sum(s['n_records'] for s in item['splits'].values()) == client_manifest['dataset']['n_records'], 'client total mismatch')
        report['clients'].append(item)
        print(f'Client {cid+1}: source/split hashes, counts, disjointness, 1:1 oversampling and real length-256 backward/update PASS', flush=True)
    require(totals == previous['totals'], 'global split totals changed')
    require(len(seen) == sum(totals.values()) == 1468243, 'global population mismatch')
    report.update(totals=totals, eligible_records=len(seen), elapsed_seconds=time.perf_counter()-started,
                  limitations=['Existing domain/URL leakage audit retained by exact verified file hashes; no new split was created.',
                               'One batch per client is a compatibility check, not a complete epoch or measured validation/test result.'])
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2)+'\n')
    print(f'PASS: all {len(seen):,} existing records; report {output}', flush=True)

if __name__ == '__main__':
    main()

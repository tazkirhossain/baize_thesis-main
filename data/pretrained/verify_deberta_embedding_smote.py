"""Actual full pretrained DeBERTa warmup and embedding-SMOTE check, synthetic only."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from pathlib import Path
import sys
import json
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.client.data import LocalSplit
from src.client.neural_data import LocalURLDataset, make_collator
from src.client.embedding_training import train_embedding_member
from src.data.balancing import BalanceConfig
from src.models.deberta_v3_base.model import build_model, MODEL_REVISION

torch.set_num_threads(2)
torch.manual_seed(42)
cfg = json.loads((ROOT/'config/federated/embedding_balancing/deberta_v3_base_smote.json').read_text())
cfg['batch_size'] = 2
cfg['model']['max_length'] = 16
cfg['model']['source'] = str(ROOT/'data/pretrained/deberta-v3-base')
model = build_model(cfg['model']['source'], seed=42)
print('Loaded actual full pretrained DeBERTa encoder', flush=True)
labels = np.array([1, 1, 0, 0, 0, 0], dtype=np.int8)
urls = tuple(f'https://example.org/synthetic/{i}' for i in range(6))
data = LocalSplit('train', urls, labels, np.arange(6), 'synthetic-training-only')
encoder = model.deberta.encoder.layer[0].attention.self.query_proj.weight
before_encoder, before_head = encoder.detach().clone(), model.classifier.weight.detach().clone()
started = time.perf_counter()
loss, processed, report, warmup = train_embedding_member(model, LocalURLDataset(data), data, cfg,
    BalanceConfig(**cfg['balancing']), collate=make_collator(cfg['model']), device=torch.device('cpu'),
    seed=42, selection=None, cache_root=ROOT/'data/pretrained/smote-check-cache')
assert report['synthetic_examples'] == 2 and report['after'] == {'Adult': 4, 'Non-adult': 4}
assert processed == 14 and warmup == 6
assert report['feature_dimension'] == 768 and report['effective_k_neighbors'] == 1
assert report['status'] == 'adjusted_k'
assert not torch.equal(encoder, before_encoder) and not torch.equal(model.classifier.weight, before_head)
assert not list((ROOT/'data/pretrained/smote-check-cache').rglob('training.npy'))
result = {'status': 'PASS', 'scope': 'one full pretrained model on six synthetic training examples, not full FL or detection quality',
          'model': 'microsoft/deberta-v3-base', 'revision': MODEL_REVISION,
          'corpus_or_holdouts_read': False, 'pretrained_files_modified': False,
          'training_protocol': 'encoder_embeddings', 'warmup_examples_processed': warmup,
          'examples_processed': processed, 'sampling': report,
          'optimization_loss': loss/processed, 'seconds': time.perf_counter()-started,
          'test_config_overrides': {'batch_size': 2, 'max_length': 16},
          'encoder_and_head_changed': True, 'temporary_embedding_cache_removed': True}
(ROOT/'data/pretrained/deberta-embedding-smote-verification-v1.json').write_text(json.dumps(result, indent=2)+'\n')
print('PASS: full pretrained DeBERTa warmup + training-only continuous-embedding SMOTE + classifier-head update', flush=True)

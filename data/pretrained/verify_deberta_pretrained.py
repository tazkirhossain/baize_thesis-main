"""One-model offline full-checkpoint training check on a synthetic batch only."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from pathlib import Path
import sys
import json
import hashlib
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from src.models.deberta_v3_base.model import build_model, MODEL_REVISION, BASE_ARCHITECTURE
from src.client.neural_data import make_collator

torch.set_num_threads(2)
torch.manual_seed(42)
checkpoint = ROOT / 'data/pretrained/deberta-v3-base'
manifest = json.loads((checkpoint / 'download_manifest.json').read_text())
assert manifest['status'] == 'PASS' and manifest['revision'] == MODEL_REVISION
for name, record in manifest['files'].items():
    digest = hashlib.sha256()
    with (checkpoint / name).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == record['sha256']
print('PASS: downloaded file hashes verified before offline loading', flush=True)
started = time.perf_counter()
model = build_model(checkpoint, seed=42)
load_seconds = time.perf_counter() - started
assert model.loading_audit['pretrained_encoder']
assert all(getattr(model.config, key) == value for key, value in BASE_ARCHITECTURE.items())
assert model.config.id2label == {0: 'Non-adult', 1: 'Adult'}
parameter_count = sum(parameter.numel() for parameter in model.parameters())
print(f'PASS: full pretrained encoder loaded offline; {parameter_count:,} parameters', flush=True)
cfg = json.loads((ROOT / 'config/federated/deberta_v3_base.json').read_text())
model_cfg = dict(cfg['model'], source=str(checkpoint), max_length=16)
collate = make_collator(model_cfg)
features, labels = collate([('https://example.org/news', 0)])
assert features['input_ids'].shape[0] == 1 and features['input_ids'].shape[1] <= 16
assert 'attention_mask' in features and labels.tolist() == [0]
model.eval()
with torch.no_grad():
    output = model(**features)
assert output.logits.shape == (1, 2) and torch.isfinite(output.logits).all()
print('PASS: production client collator and full-checkpoint inference', flush=True)
encoder_name, encoder = next((name, parameter) for name, parameter in model.named_parameters()
                             if name.startswith('deberta.encoder.layer.0.') and name.endswith('query_proj.weight'))
before_encoder = encoder.detach().clone()
before_head = model.classifier.weight.detach().clone()
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=cfg['learning_rate'])
optimizer.zero_grad(set_to_none=True)
started = time.perf_counter()
loss = torch.nn.functional.cross_entropy(model(**features).logits, labels)
assert torch.isfinite(loss)
loss.backward()
assert encoder.grad is not None and torch.isfinite(encoder.grad).all() and encoder.grad.abs().sum() > 0
assert model.classifier.weight.grad is not None and torch.isfinite(model.classifier.weight.grad).all()
optimizer.step()
step_seconds = time.perf_counter() - started
encoder_change = float((encoder.detach() - before_encoder).abs().max())
head_change = float((model.classifier.weight.detach() - before_head).abs().max())
assert encoder_change > 0 and head_change > 0
optimizer.zero_grad(set_to_none=True)
model.eval()
with torch.no_grad():
    after = model(**features).logits
assert torch.isfinite(after).all()
result = {'status': 'PASS', 'model': 'microsoft/deberta-v3-base', 'revision': MODEL_REVISION,
          'scope': 'one full pretrained model, one synthetic example; not federation or model quality',
          'dataset_read': False, 'checkpoint_modified': False, 'offline_loading': True,
          'parameter_count': parameter_count, 'loading_audit': model.loading_audit,
          'model_architecture': BASE_ARCHITECTURE,
          'synthetic_batch_size': 1, 'smoke_max_length': 16,
          'production_max_length_unchanged': cfg['model']['max_length'],
          'all_parameters_trainable': all(p.requires_grad for p in model.parameters()),
          'optimizer': 'SGD', 'learning_rate': cfg['learning_rate'],
          'loss': float(loss.detach()), 'encoder_probe': encoder_name,
          'encoder_max_update': encoder_change, 'classifier_max_update': head_change,
          'load_seconds': load_seconds, 'step_seconds': step_seconds,
          'device': 'cpu', 'torch': torch.__version__}
(ROOT / 'data/pretrained/deberta-pretrained-verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(f'PASS: full encoder/head forward, backward and SGD step; loss={float(loss.detach()):.6f}', flush=True)

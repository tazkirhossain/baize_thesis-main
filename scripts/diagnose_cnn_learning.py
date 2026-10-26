"""Bounded client-local learning diagnosis using original TRAIN data only.

Sanity checks memorization; pilot compares optimization on fixed real samples.
Neither is a five-client performance result. No validation/test files are read,
no raw URLs/scores are exported, and no active experiment settings are changed.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Subset

from src.client.data import read_local_split
from src.client.neural_data import LocalURLDataset, make_collator
from src.client.optimizers import make_optimizer
from src.client.selection import membership_summary, selection_mask, selection_views
from src.evaluation.classification import confusion_metrics
from src.evaluation.thresholds import histogram_counts, metrics_at_threshold, select_threshold
from src.models.federated_state import StateCodec, make_models
from src.server.checkpoints import load_checkpoint
from src.server.experiment_config import read_json, resolve_experiment


ARMS = {'sgd_001': ('sgd', .001), 'sgd_01': ('sgd', .01), 'adamw_001': ('adamw', .001)}


def digest(rows):
    return hashlib.sha256(np.sort(rows).astype('<i8').tobytes()).hexdigest()


def evaluate(model, loader):
    model.eval()
    scores, labels, loss_sum = [], [], 0.0
    with torch.no_grad():
        for features, y in loader:
            logits = model(**features)
            loss_sum += float(torch.nn.functional.cross_entropy(logits, y, reduction='sum'))
            scores.extend(logits.softmax(1)[:, 1].double().tolist())
            labels.extend(y.tolist())
    scores, labels = np.asarray(scores), np.asarray(labels)
    matrix = np.bincount(2*labels+(scores >= .5).astype(int), minlength=4).reshape(2, 2)
    summary = dict(confusion_metrics(matrix), loss=loss_sum/len(labels), threshold=.5,
                   roc_auc=float(roc_auc_score(labels, scores)),
                   average_precision=float(average_precision_score(labels, scores)),
                   score_range=[float(scores.min()), float(scores.max())])
    # This optimization is allowed only on the reserved TRAINING-selection pool
    # (or as a memorization diagnostic on optimization data), never holdouts.
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = np.divide(2*precision[:-1]*recall[:-1], precision[:-1]+recall[:-1],
                   out=np.zeros_like(precision[:-1]), where=(precision[:-1]+recall[:-1]) != 0)
    best = max(range(len(thresholds)), key=lambda i: (f1[i], thresholds[i]))
    summary['exact_selection_threshold'] = float(thresholds[best])
    summary['exact_selection_adult_f1'] = float(f1[best])
    for bins in (201, 1001):
        histogram = histogram_counts(scores, labels, bins)
        summary[f'grid_{bins}'] = select_threshold(histogram)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('sanity', 'pilot', 'checkpoint'), required=True)
    parser.add_argument('--client-dir', type=Path, default=ROOT/'data/client-iid-v1/client_000')
    parser.add_argument('--config', type=Path, default=ROOT/'config/experiment.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--arms', nargs='+', choices=tuple(ARMS), default=list(ARMS))
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--per-class', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--selection-size', type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a fresh output folder')
    if min(args.steps, args.per_class, args.epochs, args.selection_size) < 1:
        parser.error('positive diagnostic budgets required')
    if args.stage == 'checkpoint' and args.checkpoint is None:
        parser.error('--checkpoint is required for checkpoint diagnosis')
    if args.checkpoint:
        cfg = read_json(args.checkpoint.with_suffix('.json'))['configuration']
    else:
        cfg = resolve_experiment(read_json(args.config), ROOT).configuration
    if cfg['model']['family'] != 'character_cnn' or cfg.get('website_enrichment'):
        parser.error('this diagnostic uses the existing URL-only Character CNN')
    if not cfg.get('model_selection'):
        parser.error('the reserved training-selection split must be configured')
    torch.set_num_threads(2)
    args.output.mkdir(parents=True)
    source = Path(__file__).read_bytes()
    (args.output/'diagnostic_source.py').write_bytes(source)
    started = time.perf_counter()
    print('Loading own-client ORIGINAL TRAIN only; deriving existing domain roles.', flush=True)
    data = read_local_split(args.client_dir, 'train')
    optimization, selection = selection_views(data, selection_mask(data,
        fraction=cfg['model_selection']['fraction'], seed=cfg['seed']))
    rng = np.random.default_rng(cfg['seed'])
    per_class = 32 if args.stage == 'sanity' else args.per_class
    indexes = np.concatenate([rng.choice(np.flatnonzero(optimization.labels == y), per_class, replace=False)
                              for y in (0, 1)])
    rng.shuffle(indexes)
    held_indexes = rng.choice(len(selection), min(args.selection_size, len(selection)), replace=False)
    collate = make_collator(cfg['model'])
    batch_size = 32 if args.stage == 'sanity' else 64
    train_data = Subset(LocalURLDataset(optimization), indexes.tolist())
    def loader(dataset, shuffle=False):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
            generator=torch.Generator().manual_seed(cfg['seed']), collate_fn=collate)
    train_eval = loader(train_data)
    held_loader = loader(Subset(LocalURLDataset(selection), held_indexes.tolist()))
    report = dict(scope='client-local TRAIN-only learning diagnostic; not full FL/test evidence',
        diagnostic_source_sha256=hashlib.sha256(source).hexdigest(),
        environment=dict(torch=str(torch.__version__), numpy=np.__version__),
        configuration=cfg, stage=args.stage, client_directory=args.client_dir.name,
        source_train_sha256=data.sha256, original_membership=membership_summary(optimization, selection),
        optimization_sample_sha256=digest(optimization.source_rows[indexes]),
        selection_sample_sha256=digest(selection.source_rows[held_indexes]),
        sample_plan=dict(per_class=per_class, selection_sampling='uniform, label independent, without replacement',
                         selection_size=len(held_indexes), batch_size=batch_size, steps=args.steps,
                         epochs=args.epochs, threads=2, device='cpu'), measurements=[])
    arms = ['checkpoint'] if args.stage == 'checkpoint' else args.arms
    for arm in arms:
        phase_started = time.perf_counter()
        model = make_models(cfg)[0]
        item = dict(arm=arm)
        if args.stage == 'checkpoint':
            arrays, metadata = load_checkpoint(args.checkpoint, StateCodec([model]))
            StateCodec([model]).set(arrays)
            item['checkpoint'] = dict(round=metadata['round'], global_hash=metadata['global_hash'])
        else:
            name, rate = ARMS[arm]
            item.update(optimizer=name, learning_rate=rate, weight_decay=0)
            settings = dict(name=name, weight_decay=0.0)
            if name == 'sgd':
                settings['momentum'] = 0.0
            optimizer = make_optimizer(model.parameters(), learning_rate=rate, settings=settings)
            train_loader = loader(train_data, shuffle=True)
            total_steps = args.steps if args.stage == 'sanity' else args.epochs*len(train_loader)
            losses, step = [], 0
            first_weight = model.conv1[0].weight.detach().clone()
            print(f'ARM {arm}: {total_steps} updates; {len(train_data)} optimization records.', flush=True)
            iterator = iter(train_loader)
            while step < total_steps:
                try:
                    features, labels = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    features, labels = next(iterator)
                model.train()
                torch.manual_seed(cfg['seed']+step)
                optimizer.zero_grad(set_to_none=True)
                logits = model(**features)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite optimization loss')
                loss.backward()
                if step == 0:
                    item['initial_grad_norm'] = dict(encoder=float(model.conv1[0].weight.grad.norm()),
                                                     output=float(model.fc[-1].weight.grad.norm()))
                optimizer.step()
                losses.append(float(loss.detach()))
                step += 1
                if step == 1 or step % 20 == 0 or step == total_steps:
                    print(f'  update {step}/{total_steps}; recent mean loss={np.mean(losses[-20:]):.4f}; '
                          f'elapsed={time.perf_counter()-phase_started:.1f}s', flush=True)
            item.update(updates=step, first_10_loss=float(np.mean(losses[:10])),
                last_10_loss=float(np.mean(losses[-10:])),
                encoder_relative_update=float((model.conv1[0].weight.detach()-first_weight).norm()/first_weight.norm()))
        item['optimization'] = evaluate(model, train_eval)
        if args.stage != 'sanity':
            item['training_selection'] = evaluate(model, held_loader)
        item['elapsed_seconds'] = time.perf_counter()-phase_started
        report['measurements'].append(item)
        report['elapsed_seconds'] = time.perf_counter()-started
        (args.output/'measurements.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(item, allow_nan=False), flush=True)
        del model
    print(f'COMPLETE: {args.output}', flush=True)


if __name__ == '__main__':
    main()

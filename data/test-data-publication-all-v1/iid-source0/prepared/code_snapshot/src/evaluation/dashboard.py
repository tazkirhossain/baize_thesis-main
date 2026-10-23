"""Read-only, allowlisted aggregate data for the local research dashboard.

Never opens URL datasets, prediction files, checkpoints, or reproduction trees.
The dashboard is an observer: no training or hyperparameter selection happens here.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from src.evaluation.reporting import checked_metrics, checked_selection, fingerprint
from src.server.config import validate_config

ALIASES = {"stratified_url_iid": "iid", "planned_label_skew": "label_skew",
           "planned_category_skew": "category_skew"}
PARTITIONS = ("iid", "label_skew", "category_skew", "unknown")
METHODS = ("none", "undersampling", "oversampling", "smote", "ensemble_undersampling")
CLIENT_FIELDS = ("original_train", "resampled_train", "original_adult", "original_non_adult",
                 "used_adult", "used_non_adult", "duplicated", "synthetic", "examples_processed",
                 "ensemble_members", "warmup_examples_processed", "train_loss", "train_seconds",
                 "fit_seconds", "aggregation_weight", "smote_effective_k")
TIMING_FIELDS = ("client_training_seconds_sum", "client_training_seconds_max", "fit_round_wall_seconds",
                 "val_phase_wall_seconds", "round_wall_seconds", "train_evaluation_seconds_sum")
BYTE_FIELDS = ("fit_down_bytes", "fit_up_bytes", "evaluation_down_bytes", "evaluation_up_bytes", "total_bytes")


def _json(path, boundary):
    path = Path(path)
    if not path.resolve().is_relative_to(Path(boundary).resolve()):
        raise ValueError("source is outside the approved local root")
    if path.stat().st_size > 16*1024*1024:
        raise ValueError("aggregate file exceeds 16 MiB")
    return json.loads(path.read_text(encoding="utf-8"))


def _numbers(data, fields):
    output = {}
    for key in fields:
        value = data.get(key)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError(f"invalid aggregate field: {key}")
        output[key] = value
    return output


def load_run(server, data_root):
    server, data_root = Path(server), Path(data_root)
    completed = (server/'run.json').exists()
    run = _json(server/'run.json', data_root) if completed else {}
    cfg = run.get('configuration') or _json(server.parent/'experiment.json', data_root)
    cfg = validate_config(cfg)
    model, balance = cfg['model']['family'], cfg['balancing']['method']
    if model not in ('character_cnn', 'deberta_v3_base') or balance not in METHODS or cfg['n_clients'] != 5:
        raise ValueError('not a supported five-client neural run')
    if completed and run.get('status') != 'PASS':
        raise ValueError('completed run is not PASS')
    contracts = run.get('client_data_contracts', {})
    declared = run.get('experiment', cfg.get('experiment', {}))
    kinds = {x.get('data_kind') for x in contracts.values()} - {None}
    parts = {x.get('partition_strategy') for x in contracts.values()} - {None}
    if len(kinds) > 1 or len(parts) > 1:
        raise ValueError('clients disagree on data kind/partition')
    kind = next(iter(kinds), declared.get('data_kind', 'unknown'))
    partition = next(iter(parts), declared.get('partition_strategy', 'unknown'))
    partition = ALIASES.get(partition, partition)
    if kind not in ('real', 'synthetic', 'unknown') or partition not in PARTITIONS:
        raise ValueError('unsupported data classification')
    protocol = 'encoder_embeddings' if cfg.get('representation') else 'end_to_end'
    if balance == 'smote' and protocol != 'encoder_embeddings':
        raise ValueError('SMOTE run has no valid continuous representation protocol')
    configuration = {name: cfg.get(name) for name in ('batch_size', 'local_epochs', 'rounds', 'clients_per_round',
        'learning_rate', 'seed', 'threshold', 'aggregation_weight', 'evaluate_training', 'evaluate_final_test')}
    configuration.update(n_clients=5, sequence_length=cfg['model'].get('sequence_length', cfg['model'].get('max_length')),
                         adult_to_non_adult=cfg['balancing']['adult_to_non_adult'],
                         warmup_epochs=cfg.get('representation', {}).get('warmup_epochs', 0))
    rounds, sources, raw_rounds = [], [], []
    round_files = sorted(server.glob('round_*.json'))
    for expected, file in enumerate(round_files, 1):
        row = _json(file, data_root)
        raw_rounds.append(row)
        if row['round'] != expected or expected > cfg['rounds']:
            raise ValueError('rounds are not contiguous or exceed configured rounds')
        # Only finalized aggregate rounds can be interpreted, even during a run.
        if not row.get('global_hash'):
            continue
        if 'test' in row and (not cfg.get('evaluate_final_test') or expected != cfg['rounds']):
            raise ValueError('test result violates fixed-final-round protocol')
        clients, seen = [], set()
        for client in row['clients']:
            cid = client['client_id']
            if type(cid) is not int or cid not in range(5) or cid in seen:
                raise ValueError('duplicate or invalid client ID')
            seen.add(cid)
            clients.append(dict(client_id=cid+1, status='Aggregated', **_numbers(client, CLIENT_FIELDS)))
        metrics = {split: checked_metrics(row[split]) for split in ('train', 'val', 'test') if split in row}
        rounds.append({'round_number': expected, 'metrics': metrics, 'clients': sorted(clients, key=lambda c: c['client_id']),
                       'optimization_loss': _numbers(row.get('optimization', {}), ('loss',))['loss'],
                       'timing': _numbers(row.get('timing', {}), TIMING_FIELDS),
                       'communication': _numbers(row.get('communication', {}), BYTE_FIELDS)})
        sources.append({'file': file.name, 'sha256': fingerprint(file)})
    if completed and (len(rounds) != cfg['rounds'] or not rounds or 'val' not in rounds[-1]['metrics']):
        raise ValueError('completed run has incomplete metrics/round history')
    if completed:
        checked_selection(cfg, run, raw_rounds)
    source = server/'run.json' if completed else server.parent/'experiment.json'
    sources.append({'file': source.name, 'sha256': fingerprint(source)})
    population = declared.get('evaluation_population_id')
    # Different methods/partitions can be compared only with matching budgets,
    # representation, cohort, seed and evaluation policy. Missing cohort is unique.
    comparison = {key: configuration[key] for key in ('batch_size', 'local_epochs', 'rounds', 'clients_per_round',
                  'learning_rate', 'seed', 'threshold', 'aggregation_weight', 'adult_to_non_adult', 'warmup_epochs')}
    comparison.update(model=model, data_kind=kind, protocol=protocol, population=population,
                      model_configuration=cfg['model'], balancing_seed=cfg['balancing'].get('seed'),
                      model_selection=cfg.get('model_selection'))
    comparison_id = hashlib.sha256(json.dumps(comparison, sort_keys=True).encode()).hexdigest()[:16] if population else None
    name = server.parent.name
    contract_signature = hashlib.sha256(json.dumps(contracts, sort_keys=True).encode()).hexdigest() if contracts else None
    return {'experiment_id': hashlib.sha256(name.encode()).hexdigest()[:12], 'name': name,
            'model_name': model, 'partition_type': partition, 'imbalance_method': balance, 'data_kind': kind,
            'training_protocol': protocol, 'status': 'completed' if completed else 'incomplete',
            'configuration': configuration, 'rounds': rounds, 'comparison_group': comparison_id,
            'partition_data_signature': contract_signature,
            'evaluation_population_id': population, 'evaluation_protocol': (
                f'Training-only domain selection, {cfg["model_selection"]["fraction"]:.0%} of training domains; '
                f'best Adult F1 checkpoint; optional test once after final FL round. '
                f'Selected checkpoint: {run.get("selection", {}).get("round", "pending")}.'
                if cfg.get('model_selection') else
                'Fixed final round; optional test once; no automatic best-model selection'),
            'duration_seconds': sum(r['timing']['round_wall_seconds'] for r in rounds)
                if rounds and all(r['timing']['round_wall_seconds'] is not None for r in rounds) else None,
            'updated_at': datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat(),
            'sources': sources, 'monitoring': 'recorded_snapshot',
            'prediction_probabilities_available': False, 'example_predictions_available': False}


def dataset_summary(data_root):
    """Counts from audited manifests only; these are NOT experiment outcomes."""
    root = Path(data_root)/'client-iid-v1'
    if not (root/'report.json').exists():
        return None
    audit = _json(root/'report.json', data_root)
    clients, counts, categories = [], {'adult': 0, 'non_adult': 0}, {}
    for cid in range(5):
        manifest = _json(root/f'client_{cid:03d}/manifest.json', data_root)
        local = manifest['dataset']
        adult, non = local['class_counts']['1'], local['class_counts']['0']
        counts['adult'] += adult
        counts['non_adult'] += non
        for key, value in local.get('category_counts', {}).items():
            categories[key] = categories.get(key, 0)+value
        clients.append({'client_id': cid+1, 'total': adult+non, 'adult': adult, 'non_adult': non})
    if counts['adult']+counts['non_adult'] != audit['retained_records']:
        raise ValueError('dataset manifest totals disagree')
    return {'scope': 'audited real IID partitions; separate from selected experiment metrics',
            'n_records': audit['retained_records'], 'class_counts': counts,
            'category_counts': categories, 'clients': clients,
            'source': 'client-iid-v1/report.json', 'source_sha256': fingerprint(root/'report.json')}


def dashboard_snapshot(data_root):
    data_root = Path(data_root)
    runs, issues = [], []
    # Deliberately bounded: current run directories only, never snapshots/archives.
    for server in sorted(data_root.glob('*/server')):
        if not server.is_dir() or not ((server/'run.json').exists() or (server.parent/'experiment.json').exists()):
            continue
        try:
            runs.append(load_run(server, data_root))
        except (ValueError, KeyError, OSError, TypeError) as error:
            issues.append({'run': server.parent.name, 'reason': f'Invalid or incomplete aggregate results ({type(error).__name__})'})
    try:
        dataset = dataset_summary(data_root)
    except (ValueError, KeyError, OSError, TypeError):
        dataset = None
        issues.append({'run': 'Dataset audit', 'reason': 'Audited counts unavailable or inconsistent'})
    templates = {}
    config_root = Path(__file__).resolve().parents[2]/'config/federated'
    for family in ('character_cnn', 'deberta_v3_base'):
        file = config_root/f'{family}.json'
        if file.exists():
            templates[family] = _json(file, config_root)
        file = config_root/'embedding_balancing'/f'{family}_smote.json'
        if file.exists():
            templates[f'{family}_embeddings'] = _json(file, config_root)
    return {'schema_version': 1, 'generated_at': datetime.now(timezone.utc).isoformat(),
            'runs': sorted(runs, key=lambda r: r['updated_at'], reverse=True), 'dataset': dataset, 'issues': issues,
            'training_templates': templates,
            'capabilities': {'read_only': True, 'live_telemetry': False, 'raw_records': False}}

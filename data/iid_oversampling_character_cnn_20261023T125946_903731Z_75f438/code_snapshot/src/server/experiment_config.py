"""Resolve one user-facing experiment file into the existing Flower contract.

Only orchestration changes: no partitioning, resampling, training or evaluation
algorithm lives here. Model-specific architecture/checkpoint defaults stay in
their existing named presets.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re

from src.server.config import validate_config

PARTITIONS = {'iid': 'client-iid-v1', 'label_skew': 'client-label-skew-v1',
              'category_skew': 'client-category-skew-v1'}
PARTITION_ALIASES = {'stratified_url_iid': 'iid', 'planned_label_skew': 'label_skew',
                     'planned_category_skew': 'category_skew'}
FIELDS = {'model_name', 'partition_type', 'balancing_method', 'training_protocol',
          'n_clients', 'clients_per_round', 'rounds', 'local_epochs', 'batch_size',
          'learning_rate', 'sequence_length', 'seed', 'adult_to_non_adult',
          'ensemble_members', 'smote_k_neighbors', 'smote_insufficient_minority',
          'warmup_epochs', 'aggregation_weight', 'threshold', 'evaluate_training',
          'evaluate_final_test', 'pretrained_source', 'partition_root', 'device',
          'threads', 'generate_plots', 'evaluation_population_id'}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate configuration key: {key}')
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=_unique_object)


@dataclass(frozen=True)
class ExperimentPlan:
    configuration: dict
    partition_root: Path
    device: str
    threads: int
    generate_plots: bool
    default_file: Path
    default_source: bytes


def resolve_experiment(request, root):
    root = Path(root).resolve()
    if not isinstance(request, dict) or set(request) != FIELDS:
        raise ValueError(f'experiment fields differ: missing {sorted(FIELDS-set(request)) if isinstance(request, dict) else sorted(FIELDS)}; '
                         f'unknown {sorted(set(request)-FIELDS) if isinstance(request, dict) else []}')
    family, partition = request['model_name'], request['partition_type']
    if family not in ('character_cnn', 'deberta_v3_base'):
        raise ValueError('model_name must be character_cnn or deberta_v3_base')
    if partition not in PARTITIONS:
        raise ValueError('partition_type must be iid, label_skew or category_skew')
    protocol = request['training_protocol']
    if protocol not in ('end_to_end', 'encoder_embeddings'):
        raise ValueError('training_protocol must be end_to_end or encoder_embeddings')
    if request['balancing_method'] == 'smote' and protocol != 'encoder_embeddings':
        raise ValueError('SMOTE requires training_protocol="encoder_embeddings"; never interpolate URL strings or token IDs')
    if type(request['warmup_epochs']) is not int or request['warmup_epochs'] < 1:
        raise ValueError('warmup_epochs must be a positive integer (used only by encoder_embeddings)')
    if type(request['threads']) is not int or request['threads'] < 1:
        raise ValueError('threads must be a positive integer')
    if type(request['generate_plots']) is not bool:
        raise ValueError('generate_plots must be boolean')
    if not isinstance(request['device'], str) or not re.fullmatch(r'cpu|mps|cuda(?::\d+)?', request['device']):
        raise ValueError('device must be cpu, mps, cuda or cuda:INDEX')
    if not isinstance(request['pretrained_source'], str) or not request['pretrained_source'].strip():
        raise ValueError('pretrained_source must be a local checkpoint directory')
    override = request['partition_root']
    if override is not None and (not isinstance(override, str) or not override.strip()):
        raise ValueError('partition_root must be null or a local directory path')
    population = request['evaluation_population_id']
    if population is not None and (not isinstance(population, str) or not population.strip()):
        raise ValueError('evaluation_population_id must be null or a nonempty declared cohort identity')
    if override and population is None:
        raise ValueError('a custom partition_root requires an explicit evaluation_population_id; do not silently claim the default dataset')
    default_file = root/'config/federated'/f'{family}.json'
    default_source = default_file.read_bytes()
    cfg = deepcopy(json.loads(default_source.decode('utf-8-sig'), object_pairs_hook=_unique_object))
    if cfg['model']['family'] != family:
        raise ValueError('model preset family differs from its filename')
    for key in ('n_clients', 'clients_per_round', 'rounds', 'local_epochs', 'batch_size', 'seed',
                'aggregation_weight', 'threshold', 'evaluate_training', 'evaluate_final_test'):
        cfg[key] = request[key]
    # Null chooses the model's existing documented LR, never holdout tuning.
    if request['learning_rate'] is not None:
        cfg['learning_rate'] = request['learning_rate']
    cfg['balancing'] = {'method': request['balancing_method'], 'adult_to_non_adult': request['adult_to_non_adult'],
                        'seed': request['seed'], 'ensemble_members': request['ensemble_members'],
                        'k_neighbors': request['smote_k_neighbors'],
                        'insufficient_minority': request['smote_insufficient_minority']}
    cfg.pop('representation', None)
    if protocol == 'encoder_embeddings':
        cfg['representation'] = {'mode': protocol, 'warmup_epochs': request['warmup_epochs']}
    cfg['experiment'] = {'partition_strategy': partition, 'data_kind': 'real',
                        'evaluation_population_id': population or 'client-datasets-v3:shared_domain_hash_v1:seed=0'}
    length_key = 'sequence_length' if family == 'character_cnn' else 'max_length'
    cfg['model'][length_key] = request['sequence_length']
    if family == 'deberta_v3_base':
        cfg['model']['source'] = request['pretrained_source']
    validate_config(cfg)
    partition_root = Path(override) if override else Path('data')/PARTITIONS[partition]
    if not partition_root.is_absolute():
        partition_root = root/partition_root
    return ExperimentPlan(cfg, partition_root.resolve(), request['device'], request['threads'],
                          request['generate_plots'], default_file, default_source)


def check_client_metadata(directory, cid, partition=None):
    """Own-client fingerprints and counts only; never reads records or URLs."""
    directory = Path(directory)
    manifest = read_json(directory/'manifest.json')
    splits = read_json(directory/'splits/manifest.json')
    if manifest['client_id'] != cid or splits.get('client_id', cid) != cid:
        raise ValueError(f'Client {cid+1}: directory/manifest ownership mismatch')
    for split in ('train', 'val', 'test'):
        item = splits['splits'][split]
        if (not (directory/'splits'/f'{split}.jsonl').is_file() or
                type(item['n_records']) is not int or item['n_records'] <= 0 or
                not isinstance(item.get('sha256'), str) or len(item['sha256']) != 64):
            raise ValueError(f'Client {cid+1}: missing or invalid {split} split')
    if not (directory/'splits/preprocessing.json').is_file():
        raise ValueError(f'Client {cid+1}: local preprocessing contract is missing')
    if partition is not None:
        preprocessing = read_json(directory/'splits/preprocessing.json')
        descriptor = preprocessing.get('partitioning', {})
        actual = descriptor.get('name', descriptor.get('strategy'))
        if manifest.get('synthetic', False) or PARTITION_ALIASES.get(actual, actual) != partition:
            raise ValueError(f'Client {cid+1}: actual data distribution differs from selected partition_type')


def check_plan(plan, *, mode='local', client_id=None, client_directory=None):
    """Preflight paths/declared strategy without loading any model or raw dataset.

    The actual client loader still checks all split bytes, labels and source-row
    disjointness. The server-only mode never checks/opens client directories.
    """
    if mode == 'smoke':
        if plan.configuration['model']['family'] != 'character_cnn':
            raise ValueError('the existing synthetic smoke command supports Character CNN only')
        return
    family = plan.configuration['model']['family']
    if family == 'deberta_v3_base':
        source = Path(plan.configuration['model']['source'])
        if not source.is_absolute():
            source = plan.default_file.parents[2]/source
        missing = [name for name in ('config.json', 'tokenizer_config.json', 'spm.model', 'pytorch_model.bin')
                   if not (source/name).is_file()]
        if missing:
            raise ValueError(f'pretrained DeBERTa files are missing: {", ".join(missing)}')
    if mode == 'server':
        return
    if mode == 'client':
        if type(client_id) is not int or client_id not in range(5):
            raise ValueError('client mode requires --client-id 0, 1, 2, 3 or 4')
        check_client_metadata(client_directory or plan.partition_root/f'client_{client_id:03d}', client_id,
                              plan.configuration['experiment']['partition_strategy'])
        return
    if mode != 'local':
        raise ValueError('unsupported run mode')
    root = plan.partition_root
    report = read_json(root/'report.json')
    descriptor = report.get('partitioning', {})
    actual = descriptor.get('name', descriptor.get('strategy'))
    if report.get('status') != 'PASS' or PARTITION_ALIASES.get(actual, actual) != plan.configuration['experiment']['partition_strategy']:
        raise ValueError('partition_root audit does not match the selected partition_type')
    if {p.name for p in root.glob('client_*') if p.is_dir()} != {f'client_{i:03d}' for i in range(5)}:
        raise ValueError('partition_root must contain exactly the five assigned client directories')
    for cid in range(5):
        check_client_metadata(root/f'client_{cid:03d}', cid, plan.configuration['experiment']['partition_strategy'])

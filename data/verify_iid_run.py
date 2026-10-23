"""Independent full-corpus verification; private local audit artifact."""
from array import array
from collections import Counter
import hashlib
import heapq
from itertools import zip_longest
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baize.data.preprocessing import _file_hash, load_local_split
from baize.data.url import normalize_url
from src.data.iid_partitioning import stratified_iid_indices
from src.data.split_policy import shared_domain_split

source = ROOT / 'data/client-datasets-v3'
root = ROOT / 'data/client-iid-v1'
report = json.loads((root / 'report.json').read_text())
old = json.loads((source / 'report.json').read_text())
assert len(report['clients']) == 5
assert report['prepared_source_report_sha256'] == _file_hash(source / 'report.json')
assert json.loads((root / 'cleaning_report.json').read_text()) == old
for filename in ('exclusions.jsonl', 'quarantine.jsonl'):
    assert _file_hash(root / filename) == _file_hash(source / filename)

numbers, labels, categories, owners = array('q'), array('b'), array('h'), array('b')
category_names = sorted(report['distribution_spreads'].keys() - {'n_records', 'adult', 'non_adult'})
# Adult is both a binary summary key and a source category.
category_names = sorted(set(category_names) | {'adult'})
codes = {name: i for i, name in enumerate(category_names)}
domain_masks = {}
roles = {}
canonical = set()
all_counts = Counter()
client_results = []

def lines(path):
    previous = -1
    with path.open('rb') as stream:
        for line in stream:
            row = json.loads(line)
            assert row['source_row'] > previous
            previous = row['source_row']
            yield previous, line

for cid in range(5):
    directory = root / f'client_{cid:03d}'
    manifest = json.loads((directory / 'manifest.json').read_text())
    local = json.loads((directory / 'splits/manifest.json').read_text())
    settings = json.loads((directory / 'preprocessing.json').read_text())
    assert manifest['dataset'] == report['clients'][cid]['dataset']
    assert _file_hash(directory / 'data.jsonl') == manifest['dataset']['sha256']
    assert _file_hash(directory / 'preprocessing.json') == manifest['preprocessing_sha256']
    assert _file_hash(directory / 'splits/preprocessing.json') == manifest['preprocessing_sha256']
    class_counts, counts = Counter(), Counter()
    with (directory / 'data.jsonl').open('rb') as stream:
        for line in stream:
            row = json.loads(line)
            normalized = normalize_url(row['url'])
            assert normalized.is_valid and normalized.canonical not in canonical
            canonical.add(normalized.canonical)
            assert row['domain'] == normalized.registrable_domain
            domain_masks[row['domain']] = domain_masks.get(row['domain'], 0) | (1 << cid)
            numbers.append(row['source_row'])
            labels.append(row['label'])
            categories.append(codes[row['category']])
            owners.append(cid)
            class_counts[str(row['label'])] += 1
            counts[row['category']] += 1
    assert dict(class_counts) == manifest['dataset']['class_counts']
    assert dict(counts) == manifest['dataset']['category_counts']
    all_counts.update(counts)
    for slot, split in enumerate(('train', 'val', 'test')):
        path = directory / f'splits/{split}.jsonl'
        assert _file_hash(path) == local['splits'][split]['sha256']
        actual, by_class, by_category = 0, Counter(), Counter()
        with path.open('rb') as stream:
            for line in stream:
                row = json.loads(line)
                domain = row['domain']
                assert roles.setdefault(domain, slot) == slot
                assert shared_domain_split(domain, settings['split_policy']['seed'],
                                           settings['config']['fractions']) == slot
                actual += 1
                by_class[str(row['label'])] += 1
                by_category[row['category']] += 1
        info = local['splits'][split]
        assert actual == info['n_records']
        assert dict(by_class) == info['class_counts']
        assert dict(by_category) == info['category_counts']
    merged = heapq.merge(*(lines(directory / f'splits/{s}.jsonl')
                         for s in ('train', 'val', 'test')), key=lambda p:p[0])
    for original, reconstructed in zip_longest(lines(directory / 'data.jsonl'), merged):
        assert original == reconstructed
    client_results.append({'client_id':cid, 'dataset':manifest['dataset'],
                           'splits':local['splits']})
    print(f'Client {cid}: hashes, counts, domain roles and exact split coverage PASS', flush=True)

assert len(canonical) == len(numbers) == report['retained_records'] == old['retained_records']
expected_categories = Counter()
for client in old['clients']:
    expected_categories.update(client['dataset']['category_counts'])
assert all_counts == expected_categories
order = np.argsort(numbers)
ordered_numbers = np.asarray(numbers)[order]
assert np.all(np.diff(ordered_numbers) > 0)
ordered_owners = np.asarray(owners)[order]
parts = stratified_iid_indices(np.asarray(labels)[order], np.asarray(categories)[order],
                               report['partitioning']['seed'])
for cid, part in enumerate(parts):
    assert np.all(ordered_owners[part] == cid)

# Independent byte comparison proves all original IDs, labels and categories survived.
original = heapq.merge(*(lines(source / f'client_{cid:03d}/data.jsonl') for cid in range(5)), key=lambda p:p[0])
allocated = heapq.merge(*(lines(root / f'client_{cid:03d}/data.jsonl') for cid in range(5)), key=lambda p:p[0])
for before, after in zip_longest(original, allocated):
    assert before == after
X, y = load_local_split(root / 'client_000', 'val')
assert y.protected and X.shape[1] == 65536
norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
assert np.allclose(norms[norms > 0], 1)
result = {'status':'PASS', 'retained_records':len(numbers),
    'category_counts':dict(all_counts), 'clients':client_results,
    'unique_domains':len(domain_masks),
    'domains_on_multiple_clients':sum(mask.bit_count() > 1 for mask in domain_masks.values()),
    'domain_role_conflicts':0,
    'checks':{'exact_source_record_bytes_preserved':True, 'canonical_url_uniqueness':True,
        'exactly_once_source_row_coverage':True, 'seeded_membership_replay':True,
        'category_and_class_counts_verified':True, 'all_15_splits_verified':True,
        'global_domain_role_consistency':True, 'local_feature_normalization':True,
        'evaluation_labels_protected':True, 'cleaning_audit_unchanged':True},
    'feature_check':{'client':0, 'split':'val', 'shape':list(X.shape)},
    'allocation_report_sha256':_file_hash(root / 'report.json'),
    'verification_script_sha256':_file_hash(Path(__file__))}
(root / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('clients', 'category_counts')}, indent=2))

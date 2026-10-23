"""Fixed client-local selection roles inside TRAIN only; never move holdouts."""
import hashlib
import math
from urllib.parse import urlsplit

import numpy as np

from src.client.data import LocalSplit
from src.data.protection import mark_protected
from src.data.url import normalize_url

POLICY = 'shared-training-selection-domain-v1'


def selection_mask(data, *, fraction, seed, synthetic=False):
    if (type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction <= .5
            or type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError('invalid training-selection fraction or seed')
    if data.split != 'train' or getattr(data.labels, 'protected', False):
        raise ValueError('model selection must originate exclusively from training data')
    roles = {}
    mask = []
    for url in data.urls:
        normalized = normalize_url(url)
        domain = normalized.registrable_domain if normalized.is_valid else None
        # Explicit synthetic fixtures use reserved .example hosts absent from PSL.
        if not domain and synthetic:
            host = urlsplit(url).hostname or ''
            domain = host if host.endswith('.example') else None
        if not domain:
            raise ValueError('training-selection domain could not be determined')
        if domain not in roles:
            value = int.from_bytes(hashlib.sha256(f'{POLICY}:{seed}:{domain}'.encode()).digest(), 'big')
            roles[domain] = value < int(fraction * (1 << 256))
        mask.append(roles[domain])
    return np.asarray(mask, dtype=bool)


def selection_views(data, mask):
    if data.split != 'train' or getattr(data.labels, 'protected', False):
        raise ValueError('selection views require original training data')
    if mask.dtype != np.bool_ or mask.shape != (len(data),):
        raise ValueError('training-selection membership mismatch')
    views = []
    for split, indexes in (('train', np.flatnonzero(~mask)), ('selection', np.flatnonzero(mask))):
        labels, rows = data.labels[indexes].copy(), data.source_rows[indexes].copy()
        if split == 'selection':
            labels = mark_protected(labels)
        labels.setflags(write=False)
        rows.setflags(write=False)
        views.append(LocalSplit(split, tuple(data.urls[i] for i in indexes), labels, rows, data.sha256))
    if not len(views[0]):
        raise ValueError('training-selection split left no optimization records; do not retune the seed')
    return tuple(views)


def membership_summary(training, selection):
    def describe(data):
        # Hashes, supports and counts only. Raw IDs remain inside the client.
        digest = hashlib.sha256(np.sort(data.source_rows).astype('<i8').tobytes()).hexdigest()
        return {'n_records': len(data), 'adult': int(np.count_nonzero(data.labels == 1)),
                'non_adult': int(np.count_nonzero(data.labels == 0)), 'membership_sha256': digest}
    return {'policy': POLICY, 'optimization': describe(training), 'selection': describe(selection)}

"""Atomic global weight checkpoints with strict, pickle-free reload checks."""
import json
import shutil
from pathlib import Path

import numpy as np

from src.models.federated_state import state_hash


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def save_weights(path, arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    # A file handle avoids NumPy appending .npz to the temporary filename.
    with temporary.open('wb') as stream:
        np.savez(stream, **{f'tensor_{i:04d}': array for i, array in enumerate(arrays)})
    temporary.replace(path)


def load_checkpoint(path, codec, *, expected_contract=None):
    path = Path(path)
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if metadata['schema_hash'] != codec.fingerprint:
        raise ValueError('checkpoint model schema mismatch')
    if expected_contract is not None and metadata['contract'] != expected_contract:
        raise ValueError('checkpoint experiment contract mismatch')
    with np.load(path, allow_pickle=False) as archive:
        keys = [f'tensor_{i:04d}' for i in range(len(codec.entries))]
        if set(archive.files) != set(keys):
            raise ValueError('checkpoint tensor keys mismatch')
        arrays = [archive[key] for key in keys]
    codec.validate(arrays)
    if state_hash(arrays) != metadata['global_hash']:
        raise ValueError('checkpoint weights checksum mismatch')
    return arrays, metadata


def copy_best(source, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    shutil.copyfile(source, temporary)
    temporary.replace(destination)

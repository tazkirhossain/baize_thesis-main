"""Offline research provisioning across scenarios; never called by FL server/client.

The dataset owner runs this before deployment. Only features belonging to the
target client's immutable assigned IDs/URLs/splits are copied. No network or labels.
"""
import json
from pathlib import Path

from src.data.enrichment.config import digest
from src.data.enrichment.store import CollectionStore, EnrichmentReader


def reuse_client_snapshot(source_root, target_directory, enrichment, *, batch_size=256):
    if type(batch_size) is not int or not 1 <= batch_size <= 256:
        raise ValueError('reuse batch_size must be in [1, 256]')
    source_root, target_directory = Path(source_root).resolve(), Path(target_directory).resolve()
    if target_directory.parent == source_root:
        raise ValueError('reuse requires a different scenario root, not the source itself')
    readers = []
    store = None
    try:
        for cid in range(5):
            reader = EnrichmentReader(source_root/f'client_{cid:03d}', enrichment)
            if reader.manifest['identity']['client_id'] != cid:
                reader.close()
                raise ValueError('source snapshot ownership mismatch')
            readers.append(reader)
        store = CollectionStore(target_directory, enrichment)
        copied = 0

        def copy_batch(targets):
            nonlocal copied
            ids = [row[0] for row in targets]
            found = {}
            placeholders = ','.join('?' for _ in ids)
            for reader in readers:
                reader.assert_unchanged()
                query = f'SELECT source_row,split,url_sha256,input_digest,payload FROM records WHERE source_row IN ({placeholders})'
                for row in reader.db.execute(query, ids):
                    if row[0] in found:
                        raise ValueError('source snapshots duplicate an original record across clients')
                    found[row[0]] = (row, reader.manifest['database_sha256'])
            for source_row, split, url in targets:
                item = found.get(source_row)
                if item is None or item[0][1:3] != (split, digest(url)):
                    raise ValueError('source snapshot does not contain the same assigned URL and protected split')
                row, source_hash = item
                previous = store.db.execute('SELECT split,url_sha256,input_digest FROM records WHERE source_row=?',
                                            (source_row,)).fetchone()
                if previous:
                    if previous != row[1:4]:
                        raise ValueError('target already contains different live features; choose a new snapshot')
                    continue
                payload = json.loads(row[4])
                payload['reused_from_database_sha256'] = source_hash
                store.db.execute('INSERT INTO records VALUES (?,?,?,?,?)',
                    (source_row,split,row[2],row[3],json.dumps(payload, ensure_ascii=True)))
                copied += 1
            store.db.commit()

        for split in ('train', 'val', 'test'):
            batch = []
            with (target_directory/'splits'/f'{split}.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    batch.append((row['source_row'], split, row['url']))
                    if len(batch) == batch_size:
                        copy_batch(batch)
                        batch.clear()
                if batch:
                    copy_batch(batch)
            print(f'Client {store.identity["client_id"]+1}: reused {split}; new records={copied:,}; network=none', flush=True)
        for reader in readers:
            reader.assert_unchanged()
        report = store.write_manifest(complete=True)
        return report
    except BaseException:
        if store is not None:
            store.db.rollback()
            store.write_manifest(complete=False)
        raise
    finally:
        if store is not None:
            store.close()
        for reader in readers:
            reader.close()

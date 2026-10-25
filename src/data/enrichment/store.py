"""Resumable collection database; immutable, fingerprinted snapshots for training."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import re
import threading

from src.data.enrichment.config import digest, validate_enrichment
from src.data.preprocessing import _file_hash

INPUT_FIELDS = ('url', 'title', 'description', 'keywords', 'rating', 'language', 'text',
                'sitemap_paths', 'page_status', 'robots_status', 'sitemap_status')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def input_fields(record):
    return {key: record.get(key, '') for key in INPUT_FIELDS}


def source_identity(directory):
    directory = Path(directory)
    owner = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))['client_id']
    source = json.loads((directory/'splits/manifest.json').read_text(encoding='utf-8'))
    if type(owner) is not int or owner not in range(5):
        raise ValueError('invalid client ownership')
    identity = {'client_id': owner, 'splits': {}}
    for name in ('train', 'val', 'test'):
        info = source['splits'][name]
        if _file_hash(directory/'splits'/f'{name}.jsonl') != info['sha256']:
            raise ValueError('original local split fingerprint mismatch')
        identity['splits'][name] = dict(sha256=info['sha256'], n_records=info['n_records'])
    return identity


def snapshot_directory(directory, enrichment):
    validate_enrichment(enrichment)
    return Path(directory)/'enrichment'/enrichment['snapshot_id']


def atomic_json(path, value):
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


class CollectionStore:
    def __init__(self, directory, enrichment):
        from src.data.enrichment.locking import SnapshotLock
        destination = snapshot_directory(directory, enrichment)
        destination.mkdir(parents=True, exist_ok=True)
        self.lock = SnapshotLock(destination/'.collection.lock')
        try:
            self._initialize(directory, enrichment)
        except BaseException:
            if hasattr(self, 'db'):
                self.db.close()
            self.lock.close()
            raise

    def _initialize(self, directory, enrichment):
        self.identity = source_identity(directory)
        self.cfg = enrichment
        self.directory = snapshot_directory(directory, enrichment)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory/'features.sqlite3'
        self.manifest_path = self.directory/'manifest.json'
        if self.manifest_path.exists():
            old = json.loads(self.manifest_path.read_text())
            if old['identity'] != self.identity or old['configuration'] != enrichment:
                raise ValueError('snapshot source/settings changed; use a NEW snapshot_id')
            if old['status'] == 'complete':
                raise ValueError('snapshot is frozen; use --verify-only or a NEW snapshot_id')
        self.db = sqlite3.connect(self.path)
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS records (
                source_row INTEGER PRIMARY KEY, split TEXT NOT NULL, url_sha256 TEXT NOT NULL,
                input_digest TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS resources (
                key TEXT PRIMARY KEY, payload TEXT NOT NULL);
        ''')
        expected = json.dumps({'identity': self.identity, 'configuration': enrichment}, sort_keys=True)
        previous = self.db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
        if previous and json.loads(previous[0]) != json.loads(expected):
            self.close()
            raise ValueError('collection database belongs to different input/settings')
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('contract', ?)", (expected,))
        self.db.commit()
        self.write_manifest(complete=False)

    def has_record(self, row, split, url):
        item = self.db.execute('SELECT split,url_sha256 FROM records WHERE source_row=?', (row,)).fetchone()
        if item is None:
            return False
        if item != (split, digest(url)):
            raise ValueError('record ownership or URL changed during collection')
        return True

    def add_record(self, row, split, record):
        signature = digest({'source_row': row, 'input': input_fields(record)})
        self.db.execute('INSERT INTO records VALUES (?,?,?,?,?)',
            (row, split, digest(record['url']), signature, json.dumps(record, ensure_ascii=True)))
        self.db.commit()

    def resource(self, key):
        row = self.db.execute('SELECT payload FROM resources WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def save_resource(self, key, value):
        self.db.execute('INSERT OR REPLACE INTO resources VALUES (?,?)', (key, json.dumps(value)))
        self.db.commit()

    def write_manifest(self, *, complete):
        counts, statuses, missing = Counter(), Counter(), Counter()
        signature = 0
        total = 0
        for split, item_digest, raw in self.db.execute('SELECT split,input_digest,payload FROM records'):
            payload = json.loads(raw)
            total += 1
            counts[split] += 1
            statuses[payload['page_status']] += 1
            for field in ('title', 'description', 'keywords', 'rating', 'language', 'text', 'sitemap_paths'):
                missing[field] += int(not payload.get(field))
            signature = (signature+int(item_digest, 16)) % (1 << 256)
        expected = {name: value['n_records'] for name, value in self.identity['splits'].items()}
        if complete and any(counts[name] != count for name, count in expected.items()):
            raise ValueError('cannot freeze an incomplete enrichment snapshot')
        record = {'schema': 'website-enrichment-v1', 'status': 'complete' if complete else 'partial',
                  'configuration': self.cfg, 'configuration_sha256': digest(self.cfg),
                  'identity': self.identity, 'required_counts': expected, 'collected_counts': dict(counts),
                  'n_records': total, 'page_status_counts': dict(statuses), 'missing_fields': dict(missing),
                  'excluded_records': 0, 'input_digest': f'{signature:064x}',
                  'updated_at': utc_now(), 'database_sha256': _file_hash(self.path) if complete else None,
                  'policy': 'all original rows retained; no labels used; raw HTML not stored; offline training'}
        atomic_json(self.manifest_path, record)
        return record

    def close(self):
        self.db.close()
        self.lock.close()


def check_snapshot(directory, enrichment, *, verify_bytes=False):
    path = snapshot_directory(directory, enrichment)
    try:
        manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise ValueError('website snapshot missing; run scripts/enrich_client.py for this client first') from None
    source = json.loads((Path(directory)/'splits/manifest.json').read_text(encoding='utf-8'))
    owner = json.loads((Path(directory)/'manifest.json').read_text(encoding='utf-8'))['client_id']
    expected = {'client_id': owner, 'splits': {name: dict(sha256=source['splits'][name]['sha256'],
                n_records=source['splits'][name]['n_records']) for name in ('train', 'val', 'test')}}
    if (manifest.get('schema') != 'website-enrichment-v1' or manifest.get('status') != 'complete'
            or manifest.get('configuration') != enrichment or manifest.get('configuration_sha256') != digest(enrichment)
            or manifest.get('identity') != expected or manifest.get('excluded_records') != 0
            or not isinstance(manifest.get('input_digest'), str)
            or re.fullmatch('[0-9a-f]{64}', manifest['input_digest']) is None
            or manifest.get('n_records') != sum(x['n_records'] for x in expected['splits'].values())
            or any(manifest.get('collected_counts', {}).get(name, 0) != item['n_records']
                   for name, item in expected['splits'].items())
            or not (path/'features.sqlite3').is_file()):
        raise ValueError('website snapshot is incomplete or its source/configuration does not match')
    if verify_bytes:
        source_identity(directory)  # Verify original bytes as well as manifest declarations.
        if _file_hash(path/'features.sqlite3') != manifest.get('database_sha256'):
            raise ValueError('website snapshot database checksum mismatch')
    return manifest


class EnrichmentReader:
    """No network access or fitting during training, selection, validation or test."""
    def __init__(self, directory, enrichment):
        self.manifest = check_snapshot(directory, enrichment, verify_bytes=True)
        path = snapshot_directory(directory, enrichment)/'features.sqlite3'
        self.path = path
        self.stamp = (path.stat().st_size, path.stat().st_mtime_ns)
        # Flower invokes a client on worker threads. Serialize read-only lookups;
        # DataLoader workers stay at zero and no SQLite writes are ever allowed.
        self.read_lock = threading.RLock()
        self.db = sqlite3.connect(path.resolve().as_uri()+'?mode=ro&immutable=1', uri=True, check_same_thread=False)

    def assert_unchanged(self):
        if (self.path.stat().st_size, self.path.stat().st_mtime_ns) != self.stamp:
            raise ValueError('frozen website snapshot changed during the experiment')

    def read(self, source_row, split, url):
        with self.read_lock:
            row = self.db.execute('SELECT split,url_sha256,payload FROM records WHERE source_row=?',
                                  (int(source_row),)).fetchone()
        parent = 'train' if split == 'selection' else split
        if row is None or row[:2] != (parent, digest(url)):
            raise ValueError('missing/mismatched frozen website features for a local record')
        return input_fields(json.loads(row[2]))

    def close(self):
        with self.read_lock:
            self.db.close()

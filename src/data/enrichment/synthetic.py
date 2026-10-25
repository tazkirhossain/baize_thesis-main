"""Explicit fake website inputs for synthetic smoke only; never visits a URL."""
import json
from pathlib import Path

from src.data.enrichment.store import CollectionStore


def freeze_synthetic_client(directory, enrichment):
    directory = Path(directory)
    if json.loads((directory/'manifest.json').read_text()).get('synthetic') is not True:
        raise ValueError('fake website features are allowed only for explicit synthetic clients')
    store = CollectionStore(directory, enrichment)
    try:
        for split in ('train', 'val', 'test'):
            for line in (directory/'splits'/f'{split}.jsonl').read_text().splitlines():
                row = json.loads(line)
                store.add_record(row['source_row'], split, {'url': row['url'],
                    'title': 'Synthetic website fixture', 'description': 'Offline pipeline check only',
                    'text': 'This is fixed synthetic website text without target or category information.',
                    'sitemap_paths': '/about /articles', 'page_status': 'synthetic_fixture',
                    'robots_status': 'synthetic_fixture', 'sitemap_status': 'synthetic_fixture'})
        return store.write_manifest(complete=True)
    finally:
        store.close()

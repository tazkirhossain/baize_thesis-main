"""Public, label-independent collection and input contracts."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re


@dataclass(frozen=True)
class CollectionConfig:
    user_agent: str = 'BaizeResearchBot/1.0'
    timeout_seconds: float = 10.0
    request_delay_seconds: float = 1.0
    max_crawl_delay_seconds: float = 60.0
    max_response_bytes: int = 1048576
    max_robots_bytes: int = 512000
    max_redirects: int = 5
    max_sitemap_files: int = 3
    max_sitemap_entries: int = 100
    max_title_chars: int = 256
    max_description_chars: int = 768
    max_text_chars: int = 2048
    max_sitemap_chars: int = 768

    def validate(self):
        if (not isinstance(self.user_agent, str) or not self.user_agent.isascii()
                or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]+/[0-9.]+', self.user_agent)):
            raise ValueError('use an explicit ASCII crawler product/version user_agent')
        for key in ('timeout_seconds', 'request_delay_seconds', 'max_crawl_delay_seconds'):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 60:
                raise ValueError(f'{key} must be finite and in (0, 60]')
        if self.max_crawl_delay_seconds < self.request_delay_seconds:
            raise ValueError('max_crawl_delay_seconds must cover request_delay_seconds')
        for key, maximum in [('max_response_bytes', 4*1024**2), ('max_robots_bytes', 1024**2),
                ('max_redirects', 10), ('max_sitemap_files', 20), ('max_sitemap_entries', 1000),
                ('max_title_chars', 1024), ('max_description_chars', 4096),
                ('max_text_chars', 16384), ('max_sitemap_chars', 4096)]:
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f'{key} must be in [1, {maximum}]')
        return self


def digest(value):
    def canonical(item):
        # JSON 10 and 10.0 are the same collection limit, including on resume.
        if type(item) is float and math.isfinite(item) and item.is_integer():
            return int(item)
        if isinstance(item, dict):
            return {key: canonical(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [canonical(val) for val in item]
        return item
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, ensure_ascii=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_enrichment(value):
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {'mode', 'snapshot_id', 'collection'}
            or value['mode'] != 'url_plus_metadata'
            or not isinstance(value['snapshot_id'], str)
            or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', value['snapshot_id'])
            or not isinstance(value['collection'], dict)):
        raise ValueError('website_enrichment requires url_plus_metadata, a safe snapshot_id and collection settings')
    cfg = CollectionConfig(**value['collection']).validate()
    if set(value['collection']) != set(asdict(cfg)):
        raise ValueError('resolved collection settings must be explicit and complete')
    return value


def resolve_enrichment(value, root):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {'mode', 'snapshot_id', 'collection_config'}:
        raise ValueError('website_enrichment requires mode, snapshot_id and collection_config')
    path = Path(value['collection_config'])
    path = path if path.is_absolute() else Path(root)/path
    collection = json.loads(path.read_text(encoding='utf-8-sig'))
    cfg = CollectionConfig(**collection).validate()
    return validate_enrichment(dict(mode=value['mode'], snapshot_id=value['snapshot_id'], collection=asdict(cfg)))


def pooled_digest(contracts):
    """Allocation-independent fingerprint of all frozen enriched model inputs."""
    values = [v['enrichment_digest'] for v in contracts.values() if 'enrichment_digest' in v]
    return f'{sum(int(x, 16) for x in values) % (1 << 256):064x}' if values else None

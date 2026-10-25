"""One client's label-independent collection, robots policy and sitemap cache."""
from datetime import datetime, timezone
from collections import OrderedDict
import hashlib
import json
import time
from urllib.parse import urljoin, urlsplit
import xml.etree.ElementTree as ET

from src.data.enrichment.config import CollectionConfig
from src.data.enrichment.extract import RobotsRules, extract_page, parse_sitemap, sitemap_context
from src.data.enrichment.fetch import PublicFetcher, checked_url
from src.data.enrichment.store import CollectionStore, utc_now
from src.data.url import normalize_url


class WebsiteCollector:
    def __init__(self, store, *, fetcher=None):
        self.store = store
        self.cfg = CollectionConfig(**store.cfg['collection']).validate()
        self.fetcher = fetcher or PublicFetcher(self.cfg)
        self.rules = OrderedDict()

    @staticmethod
    def origin(url):
        parts = urlsplit(url)
        return f'{parts.scheme}://{parts.netloc}'

    def robots(self, url):
        origin = self.origin(url)
        if origin in self.rules:
            rule = self.rules[origin]
            timestamp = datetime.fromisoformat(rule.record['collected_at'])
            if (datetime.now(timezone.utc)-timestamp).total_seconds() < 86400:
                self.rules.move_to_end(origin)
                return rule
        key = 'robots:'+origin
        saved = self.store.resource(key)
        if saved and (datetime.now(timezone.utc)-datetime.fromisoformat(saved['collected_at'])).total_seconds() >= 86400:
            saved = None
        if saved is None:
            domain = normalize_url(url).registrable_domain
            response = self.fetcher.get(origin+'/robots.txt', domain=domain, limit=self.cfg.max_robots_bytes)
            saved = {'status': response.status, 'http_status': response.http_status,
                     'text': response.body.decode('utf-8', errors='replace') if response.status == 'ok' else '',
                     'collected_at': utc_now(), 'body_sha256': hashlib.sha256(response.body).hexdigest()}
            if saved['text'].lstrip().lower().startswith(('<!doctype html', '<html')):
                saved.update(status='invalid_robots', http_status=0, text='')
            self.store.save_resource(key, saved)
        self.rules[origin] = RobotsRules(saved, self.cfg.user_agent)
        self.rules.move_to_end(origin)
        if len(self.rules) > 128:
            self.rules.popitem(last=False)
        return self.rules[origin]

    def allow(self, url):
        rules = self.robots(url)
        if rules.delay() > self.cfg.max_crawl_delay_seconds:
            return 'crawl_delay_exceeds_budget'
        return rules.blocked(url)

    def delay(self, url):
        return max(self.cfg.request_delay_seconds, self.robots(url).delay())

    def sitemaps(self, url):
        origin = self.origin(url)
        key = 'sitemaps:'+origin
        saved = self.store.resource(key)
        if saved is not None:
            return saved
        domain = normalize_url(url).registrable_domain
        queue = self.robots(url).sitemaps() + [origin+'/sitemap.xml']
        seen, paths, reports = set(), [], []
        while queue and len(seen) < self.cfg.max_sitemap_files:
            candidate = queue.pop(0)
            try:
                candidate = checked_url(urljoin(origin+'/', candidate), domain)
            except ValueError:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            response = self.fetcher.get(candidate, domain=domain, limit=self.cfg.max_response_bytes,
                                        allow=self.allow, delay=self.delay)
            status = response.status
            if status == 'ok':
                try:
                    kind, urls = parse_sitemap(response.body, domain=domain, config=self.cfg)
                    if kind == 'sitemapindex':
                        queue.extend(urls)
                    else:
                        paths.extend(urls)
                except (ValueError, ET.ParseError, OSError, EOFError):
                    status = 'invalid_sitemap'
            reports.append({'status': status, 'http_status': response.http_status,
                            'body_sha256': hashlib.sha256(response.body).hexdigest()})
        saved = {'status': 'ok' if paths else (reports[0]['status'] if reports else 'unavailable'),
                 'paths': sitemap_context(paths[:self.cfg.max_sitemap_entries], self.cfg.max_sitemap_chars),
                 'resources': reports, 'collected_at': utc_now()}
        self.store.save_resource(key, saved)
        return saved

    def collect(self, url):
        normalized = normalize_url(url)
        if not normalized.is_valid or normalized.raw != url:
            raise ValueError('collection expects the original already-minimized URL')
        result = {'url': url, 'collected_at': utc_now()}
        reason = self.allow(url)
        rules = self.robots(url)
        result['robots_status'] = rules.record['status']
        sitemap = self.sitemaps(url)
        result.update(sitemap_status=sitemap['status'], sitemap_paths=sitemap['paths'])
        if reason:
            result.update(page_status=reason, http_status=0)
            return result
        response = self.fetcher.get(url, domain=normalized.registrable_domain, limit=self.cfg.max_response_bytes,
                                    allow=self.allow, delay=self.delay)
        result.update(page_status=response.status, http_status=response.http_status,
                      response_bytes=response.bytes_received, final_url=response.url,
                      body_sha256=hashlib.sha256(response.body).hexdigest())
        if response.status == 'ok':
            mime = response.content_type.split(';', 1)[0].strip().lower()
            if mime not in ('text/html', 'application/xhtml+xml'):
                result['page_status'] = 'unsupported_content_type'
            else:
                result.update(extract_page(response.body, self.cfg, response.content_type))
        return result


def collect_client(directory, enrichment, *, limit=None, fetcher=None, progress_every=100):
    """Stream only this client's immutable splits. Labels are never read/used."""
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be a positive number of NEW records per invocation')
    store = CollectionStore(directory, enrichment)
    collector = WebsiteCollector(store, fetcher=fetcher)
    started, processed, complete = time.monotonic(), 0, True
    last_log = started
    print(f'Client {store.identity["client_id"]+1}: collection/resume START; snapshot={enrichment["snapshot_id"]}; '
          'original train/val/test unchanged; cached rows skipped.', flush=True)
    try:
        for split in ('train', 'val', 'test'):
            with (store.directory.parents[1]/'splits'/f'{split}.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    record = json.loads(line)
                    row, url = record['source_row'], record['url']
                    if store.has_record(row, split, url):
                        continue
                    if limit is not None and processed >= limit:
                        complete = False
                        break
                    payload = collector.collect(url)
                    store.add_record(row, split, payload)
                    processed += 1
                    if processed == 1 or processed % progress_every == 0 or time.monotonic()-last_log >= 30:
                        print(f'Client {store.identity["client_id"]+1} enrichment: new={processed:,}; '
                              f'elapsed={time.monotonic()-started:.1f}s; last_status={payload["page_status"]}', flush=True)
                        last_log = time.monotonic()
            if not complete:
                break
        report = store.write_manifest(complete=complete)
        print(f'Client {store.identity["client_id"]+1}: snapshot={report["status"]}; '
              f'collected={report["n_records"]:,}; page statuses={report["page_status_counts"]}; '
              f'missing fields={report["missing_fields"]}', flush=True)
        return report
    except BaseException:
        store.write_manifest(complete=False)
        raise
    finally:
        store.close()

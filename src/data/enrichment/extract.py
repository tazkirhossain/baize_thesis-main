"""Bounded non-executing HTML extraction, robots rules and sitemap parsing."""
import gzip
import io
from html.parser import HTMLParser
import re
from urllib.parse import quote, urlsplit
from urllib.robotparser import RobotFileParser
import xml.etree.ElementTree as ET

from src.data.enrichment.fetch import checked_url


def clean_text(value, limit):
    # Drop controls, collapse whitespace. Preserve Unicode for the pretrained tokenizer.
    value = value if isinstance(value, str) else ''
    return ' '.join(''.join(c if c.isprintable() else ' ' for c in value).split())[:limit]


class PageParser(HTMLParser):
    def __init__(self, config):
        super().__init__(convert_charrefs=True)
        self.cfg = config
        self.title, self.parts = [], []
        self.meta, self.language = {}, ''
        self.in_title = False
        self.skip = 0
        self.text_size = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ('script', 'style', 'noscript', 'svg', 'template'):
            self.skip += 1
        if tag == 'title':
            self.in_title = True
        if tag == 'html':
            self.language = clean_text(attrs.get('lang', ''), 32)
        if tag == 'meta':
            name = str(attrs.get('name', attrs.get('property', ''))).lower()
            if name in ('description', 'og:description', 'og:title', 'twitter:title',
                        'twitter:description', 'keywords', 'rating', 'content-rating') and name not in self.meta:
                self.meta[name] = clean_text(attrs.get('content', ''), self.cfg.max_description_chars)

    def handle_endtag(self, tag):
        if tag == 'title':
            self.in_title = False
        if tag in ('script', 'style', 'noscript', 'svg', 'template'):
            self.skip = max(0, self.skip-1)

    def handle_data(self, data):
        if self.skip:
            return
        if self.in_title and sum(map(len, self.title)) < self.cfg.max_title_chars:
            self.title.append(data[:self.cfg.max_title_chars])
        elif self.text_size < self.cfg.max_text_chars:
            data = clean_text(data, self.cfg.max_text_chars-self.text_size)
            self.parts.append(data)
            self.text_size += len(data)+1


def extract_page(body, config, content_type='text/html'):
    charset = re.search(r'charset=["\s]*([a-zA-Z0-9_-]+)', content_type, re.I)
    try:
        html = body.decode(charset.group(1) if charset else 'utf-8', errors='replace')
    except LookupError:
        html = body.decode('utf-8', errors='replace')
    parser = PageParser(config)
    parser.feed(html)
    meta = parser.meta
    title = clean_text(' '.join(parser.title), config.max_title_chars)
    return {'title': title or meta.get('og:title', meta.get('twitter:title', ''))[:config.max_title_chars],
            'description': meta.get('description', meta.get('og:description', meta.get('twitter:description', ''))),
            'keywords': meta.get('keywords', '')[:256],
            'rating': meta.get('rating', meta.get('content-rating', ''))[:128],
            'language': parser.language,
            'text': clean_text(' '.join(parser.parts), config.max_text_chars)}


class RobotsRules:
    def __init__(self, record, user_agent):
        self.record, self.user_agent = record, user_agent
        self.parser = RobotFileParser()
        self.parser.parse(record.get('text', '').splitlines())
        # Match all equally specific groups, then longest rule, with Allow winning ties.
        # RobotFileParser alone uses first-match rules on supported Python versions.
        groups, agents, rules = [], [], []
        in_rules = False
        for line in record.get('text', '').splitlines():
            line = line.split('#', 1)[0].strip()
            if ':' not in line:
                continue
            name, value = (part.strip() for part in line.split(':', 1))
            name = name.lower()
            if name == 'user-agent':
                if in_rules:
                    groups.append((agents, rules))
                    agents, rules, in_rules = [], [], False
                agents.append(value.lower())
            elif agents:
                in_rules = True
                if name in ('allow', 'disallow') and value:
                    pattern = self._encoded_path(value)
                    anchor = pattern.endswith('$')
                    pattern = pattern[:-1] if anchor else pattern
                    regex = '^'+'.*'.join(re.escape(part) for part in pattern.split('*'))+('$' if anchor else '')
                    rules.append((len(pattern.replace('*', '')), name == 'allow', re.compile(regex)))
        groups.append((agents, rules))
        product = user_agent.split('/', 1)[0].lower()
        specific = [(max((len(a) for a in agents if a != '*' and a in product), default=0), rules)
                    for agents, rules in groups]
        best = max((score for score, _ in specific), default=0)
        self.rules = [rule for (agents, rules), (score, _) in zip(groups, specific)
                      if (score == best if best else '*' in agents) for rule in rules]

    @staticmethod
    def _encoded_path(path):
        path = quote(path, safe="/%:@!$&'()*+,;=-._~?")
        def replace(match):
            char = chr(int(match.group(1), 16))
            return char if char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~' else match.group(0).upper()
        return re.sub(r'%([a-fA-F0-9]{2})', replace, path)

    def blocked(self, url):
        status, code = self.record['status'], self.record.get('http_status', 0)
        if status == 'ok' and code == 200:
            path = self._encoded_path(urlsplit(url).path or '/')
            matched = [(size, allow) for size, allow, regex in self.rules if regex.match(path)]
            return None if not matched or max(matched)[1] else 'robots_disallowed'
        # RFC 9309 unavailable 4xx permits access; 429 is deliberately conservative.
        if 400 <= code < 500 and code != 429:
            return None
        return 'robots_unreachable'

    def delay(self):
        delay = self.parser.crawl_delay(self.user_agent) or 0
        rate = self.parser.request_rate(self.user_agent)
        return max(delay, rate.seconds/rate.requests if rate and rate.requests else 0)

    def sitemaps(self):
        return self.parser.site_maps() or []


def parse_sitemap(body, *, domain, config):
    if body.startswith(b'\x1f\x8b'):
        with gzip.GzipFile(fileobj=io.BytesIO(body)) as stream:
            body = stream.read(config.max_response_bytes+1)
    if len(body) > config.max_response_bytes:
        raise ValueError('sitemap_too_large')
    # Reject all DTD/entity declarations, including UTF-16/32 encodings, before XML parsing.
    probe = body.replace(b'\x00', b'').upper()
    if b'<!DOCTYPE' in probe or b'<!ENTITY' in probe:
        raise ValueError('unsafe_xml')
    root = ET.fromstring(body)
    kind = root.tag.rsplit('}', 1)[-1]
    if kind not in ('urlset', 'sitemapindex'):
        raise ValueError('invalid_sitemap')
    urls = []
    for child in root:
        expected = 'url' if kind == 'urlset' else 'sitemap'
        if child.tag.rsplit('}', 1)[-1] != expected:
            continue
        for element in child:
            if element.tag.rsplit('}', 1)[-1] == 'loc' and element.text:
                try:
                    url = checked_url(element.text.strip(), domain)
                except ValueError:
                    continue
                if url not in urls:
                    urls.append(url)
                break
        if len(urls) >= config.max_sitemap_entries:
            break
    return kind, urls


def sitemap_context(urls, limit):
    # Keep URL paths as context, never copy sitemap page labels or create new records.
    return clean_text(' '.join(urlsplit(url).path for url in urls), limit)

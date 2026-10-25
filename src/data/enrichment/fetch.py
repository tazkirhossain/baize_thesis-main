"""Bounded public HTTP(S), pinned DNS, verified TLS and explicit redirects.

No cookies, authentication, proxy environment, scripts, downloads or browser state.
Every redirect is checked before a connection; connections use the validated IP.
"""
from dataclasses import dataclass
from collections import OrderedDict
import http.client
import ipaddress
import socket
import ssl
import time
import threading
from urllib.parse import quote, urljoin, urlsplit

from src.data.url import normalize_url


@dataclass
class FetchResult:
    status: str
    url: str
    http_status: int = 0
    body: bytes = b''
    content_type: str = ''
    bytes_received: int = 0


def checked_url(url, domain):
    parts = urlsplit(url)
    normalized = normalize_url(url)
    if (not normalized.is_valid or normalized.registrable_domain != domain
            or parts.username is not None or parts.password is not None
            or parts.port not in (None, 80 if parts.scheme == 'http' else 443)):
        raise ValueError('unsafe_or_cross_domain_url')
    # Reuse the already documented query/credential minimization contract.
    return normalized.raw


def public_addresses(host, port):
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not addresses:
        raise ValueError('no_public_address')
    # Reject mixed public/private DNS as well as loopback, link-local and reserved IPs.
    for entry in addresses:
        address = ipaddress.ip_address(entry[4][0])
        if not address.is_global or (address.version == 6 and address.ipv4_mapped is not None):
            raise ValueError('non_public_address')
    return addresses


class PublicFetcher:
    def __init__(self, config):
        self.cfg = config
        self.last_request = OrderedDict()

    def _request(self, url, limit, delay):
        parts = urlsplit(url)
        port = 443 if parts.scheme == 'https' else 80
        addresses = public_addresses(parts.hostname, port)
        origin = f'{parts.scheme}://{parts.netloc}'
        wait = delay - (time.monotonic() - self.last_request.get(origin, 0))
        if wait > 0:
            time.sleep(wait)
        self.last_request[origin] = time.monotonic()
        self.last_request.move_to_end(origin)
        if len(self.last_request) > 4096:
            self.last_request.popitem(last=False)
        deadline = time.monotonic() + self.cfg.timeout_seconds
        connection = http.client.HTTPConnection(parts.hostname, port, timeout=self.cfg.timeout_seconds)
        sock = None
        watchdog = None
        try:
            # Do not resolve the hostname again after validating its addresses.
            family, socktype, protocol, _, endpoint = addresses[0]
            sock = socket.socket(family, socktype, protocol)
            sock.settimeout(max(.01, deadline-time.monotonic()))
            sock.connect(endpoint)
            if parts.scheme == 'https':
                sock = ssl.create_default_context().wrap_socket(sock, server_hostname=parts.hostname)
            connection.sock = sock
            def expire():
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            # Socket timeouts alone do not stop a peer trickling partial headers.
            watchdog = threading.Timer(max(.01, deadline-time.monotonic()), expire)
            watchdog.daemon = True
            watchdog.start()
            connection.request('GET', quote(parts.path or '/', safe="/%:@!$&'()*+,;=-._~"),
                headers={'User-Agent': self.cfg.user_agent, 'Accept-Encoding': 'identity',
                         'Accept': 'text/html,application/xhtml+xml,application/xml,text/xml,text/plain,*/*;q=0.1',
                         'Connection': 'close'})
            response = connection.getresponse()
            status = response.status
            headers = {name.lower(): value for name, value in response.getheaders()}
            if status in (301, 302, 303, 307, 308):
                return status, headers, b''
            if headers.get('content-encoding', 'identity').lower() not in ('identity', ''):
                raise ValueError('unsupported_content_encoding')
            declared = headers.get('content-length')
            if declared is not None and (not declared.isdigit() or int(declared) > limit):
                raise ValueError('response_too_large')
            body = bytearray()
            while True:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError()
                if connection.sock is not None:
                    connection.sock.settimeout(remaining)
                chunk = response.read1(min(65536, limit+1-len(body)))
                if not chunk:
                    break
                body.extend(chunk)
                if len(body) > limit:
                    raise ValueError('response_too_large')
            return status, headers, bytes(body)
        except (OSError, http.client.HTTPException):
            if time.monotonic() >= deadline:
                raise TimeoutError() from None
            raise
        finally:
            if watchdog is not None:
                watchdog.cancel()
            connection.close()
            if sock is not None:
                sock.close()

    def get(self, url, *, domain, limit, allow=None, delay=None):
        original = url
        seen = set()
        try:
            for _ in range(self.cfg.max_redirects+1):
                url = checked_url(url, domain)
                if url in seen:
                    return FetchResult('redirect_loop', url)
                seen.add(url)
                if allow is not None:
                    reason = allow(url)
                    if reason:
                        return FetchResult(reason, url)
                seconds = delay(url) if delay is not None else self.cfg.request_delay_seconds
                status, headers, body = self._request(url, limit, seconds)
                if status in (301, 302, 303, 307, 308):
                    location = headers.get('location')
                    if not location:
                        return FetchResult('invalid_redirect', url, status)
                    url = urljoin(url, location)
                    continue
                return FetchResult('ok' if 200 <= status < 300 else 'http_error', url, status,
                                   body, headers.get('content-type', ''), len(body))
            return FetchResult('too_many_redirects', url)
        except (TimeoutError, socket.timeout):
            return FetchResult('timeout', original)
        except ssl.SSLError:
            return FetchResult('tls_error', original)
        except socket.gaierror:
            return FetchResult('dns_error', original)
        except (OSError, http.client.HTTPException):
            return FetchResult('network_error', original)
        except ValueError as error:
            # Reasons are enumerated; never retain remote exception text or credentials.
            reason = str(error)
            return FetchResult(reason if reason in {'unsafe_or_cross_domain_url', 'no_public_address',
                'non_public_address', 'response_too_large', 'unsupported_content_encoding'} else 'invalid_url', original)

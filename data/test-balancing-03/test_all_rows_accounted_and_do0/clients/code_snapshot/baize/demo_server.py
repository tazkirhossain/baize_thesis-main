"""Stdlib HTTP frontend for the local URL checker.

Kept free of third-party web frameworks. `scripts/serve.py` is the CLI wrapper.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from baize.inference import UrlClassifier
from baize.research_snapshot import research_snapshot

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = REPO_ROOT / "web"
MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json; charset=utf-8",
    ".ico": "image/x-icon",
}

_STATE: dict = {}


class Handler(BaseHTTPRequestHandler):
    server_version = "BAIZEDemo/0.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/api/meta":
            self._send_json(200, _STATE["classifier"].meta)
            return
        if path == "/api/experiments":
            self._send_json(200, _STATE["snapshot"])
            return
        if path in ("/", "/index.html"):
            self._send_file(WEB_ROOT / "index.html")
            return
        rel = path.lstrip("/")
        target = safe_web_path(rel)
        if target is None or not target.is_file():
            self._send_json(404, {"error": "not_found"})
            return
        self._send_file(target)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/check":
            self._send_json(404, {"error": "not_found"})
            return
        raw_len = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_len)
            if length < 0:
                raise ValueError("negative content length")
        except ValueError:
            self._send_json(400, {"error": "bad_length"})
            return
        if length > 16_384:
            self._send_json(413, {"error": "payload_too_large"})
            return
        body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "invalid_json"})
            return
        url = payload.get("url", "")
        if not isinstance(url, str):
            self._send_json(400, {"error": "url_must_be_string"})
            return
        result = _STATE["classifier"].classify(url)
        self._send_json(200, result.to_dict())

    def _send_json(self, status: int, payload: dict) -> None:
        blob = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(blob)

    def _send_file(self, path: Path) -> None:
        data = path.read_bytes()
        ctype = MIME.get(path.suffix.lower(), "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def safe_web_path(rel: str, root: Path = WEB_ROOT) -> Path | None:
    if not rel or ".." in Path(rel).parts:
        return None
    resolved_root = root.resolve()
    target = (root / rel).resolve()
    try:
        target.relative_to(resolved_root)
    except ValueError:
        return None
    return target


def make_server(
    classifier: UrlClassifier,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    _STATE["classifier"] = classifier
    _STATE["snapshot"] = research_snapshot()
    return ThreadingHTTPServer((host, port), Handler)


def serve_in_thread(
    classifier: UrlClassifier,
    host: str = "127.0.0.1",
    port: int = 0,
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    httpd = make_server(classifier, host=host, port=port)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, t

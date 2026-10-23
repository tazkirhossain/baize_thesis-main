"""Serve the built React dashboard and read-only local aggregate-results API."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
import threading
import time
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.dashboard import dashboard_snapshot


def make_handler(data_root, static_root):
    lock, cache = threading.Lock(), {'at': 0, 'bytes': b''}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = unquote(urlsplit(self.path).path)
            if route == '/api/health':
                return self.reply(b'{"status":"ok","read_only":true}', 'application/json')
            if route == '/api/experiments':
                with lock:
                    if time.monotonic()-cache['at'] > 5:
                        cache['bytes'] = json.dumps(dashboard_snapshot(data_root), allow_nan=False).encode()
                        cache['at'] = time.monotonic()
                    payload = cache['bytes']
                return self.reply(payload, 'application/json')
            if route.startswith('/api/'):
                return self.send_error(404, 'Unknown read-only API route')
            # Never serve repository/data paths; only built assets under dist.
            path = (Path(static_root)/route.lstrip('/')).resolve()
            if not path.is_relative_to(Path(static_root).resolve()) or '\\' in route:
                return self.send_error(404)
            if route == '/':
                path = Path(static_root)/'index.html'
            if not path.is_file():
                return self.send_error(404, 'Build dashboard first: cd dashboard; npm run build')
            return self.reply(path.read_bytes(), mimetypes.guess_type(path.name)[0] or 'application/octet-stream')

        def reply(self, payload, content_type):
            self.send_response(200)
            self.send_header('Content-Type', content_type+'; charset=utf-8' if content_type.startswith(('text/', 'application/json')) else content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; script-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(payload)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(ROOT/'data', ROOT/'dashboard/dist'))
    print(f'Dashboard: http://127.0.0.1:{args.port} (read-only local aggregate results)', flush=True)
    print('Stop with Ctrl+C. Raw datasets and checkpoints are never served.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()

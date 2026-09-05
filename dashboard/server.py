"""Serve M³ experiments, versioned data and local model inference."""
import argparse
import hmac
import json
import math
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch
from tokenizers import Tokenizer
from mcubed.common import device_for, digest
from mcubed.model import GPT, GPTConfig

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).parent / 'static'
ID = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}')


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def metrics(path):
    rows = []
    try:
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # Running trainer may be appending the last line.
    except FileNotFoundError:
        pass
    return rows


class Lab:
    def __init__(self, root=ROOT, device='cpu'):
        self.root = Path(root).resolve()
        self.device = device_for(device)
        self.lock = threading.Lock()
        self.cache = None
        torch.set_num_threads(4)

    def child(self, kind, identifier):
        if not isinstance(identifier, str) or not ID.fullmatch(identifier):
            raise ValueError('Invalid identifier.')
        parent = (self.root / kind).resolve()
        path = (parent / identifier).resolve()
        if path.parent != parent or not path.is_dir():
            raise ValueError('Unknown identifier.')
        return path

    def run(self, identifier):
        path = self.child('runs', identifier)
        rows = metrics(path / 'metrics.jsonl')
        env = read_json(path / 'environment.json', {})
        return {'id': identifier, 'config': read_json(path / 'config.json', {}),
            'environment': {k: env.get(k) for k in ('device', 'parameters_total', 'git_commit', 'git_status', 'data_hash', 'torch', 'hourly_usd', 'initialization')},
            'summary': read_json(path / 'summary.json'), 'failure': read_json(path / 'failure.json'),
            'failure_at': (path / 'failure.json').stat().st_mtime if (path / 'failure.json').exists() else None,
            'metrics': rows, 'checkpoint_available': (path / 'best.pt').is_file(),
            'samples': read_json(path / 'story-samples.json'),
            'updated_at': (path / 'metrics.jsonl').stat().st_mtime if (path / 'metrics.jsonl').exists() else None}

    def datasets(self):
        result = []
        for path in sorted((self.root / 'datasets').glob('*/version.json')):
            version = read_json(path)
            manifest = read_json(path.parent / 'manifest.json')
            result.append({'version': version, 'manifest': manifest})
        return result

    def stories(self, identifier, split, offset, limit):
        path = self.child('datasets', identifier)
        if split not in ('train', 'val'):
            raise ValueError('The held-out test split stays sealed during model selection.')
        if offset < 0 or not 1 <= limit <= 20:
            raise ValueError('Invalid page.')
        version = read_json(path / 'version.json')
        data_root = (self.root / version['artifact_directory']).resolve()
        if not data_root.is_relative_to((self.root / 'data').resolve()):
            raise ValueError('Dataset must reside under the mounted data directory.')
        manifest = read_json(path / 'manifest.json')
        if digest(data_root / f'{split}.jsonl') != manifest['sha256'][f'{split}.jsonl']:
            raise ValueError('Story file changed since this dataset version was registered.')
        selected = []
        with (data_root / f'{split}.jsonl').open() as f:
            for i, line in enumerate(f):
                if i >= offset + limit:
                    break
                if i >= offset:
                    selected.append({'index': i, 'text': json.loads(line)['text']})
        return {'stories': selected, 'offset': offset, 'total': manifest['splits'][split]['stories']}

    def generate(self, body):
        path = self.child('runs', body.get('run'))
        prompt = body.get('prompt', '')
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 20000:
            raise ValueError('Enter a prompt of 1–20,000 characters.')
        count, temperature, seed = int(body.get('tokens', 120)), float(body.get('temperature', 0.7)), int(body.get('seed', 42))
        if not 1 <= count <= 256 or not math.isfinite(temperature) or not 0.1 <= temperature <= 1.5 or not 0 <= seed < 2**32:
            raise ValueError('Invalid generation settings.')
        checkpoint = path / 'best.pt'
        if not checkpoint.is_file():
            raise ValueError('This experiment has no saved checkpoint yet.')
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError('Inference is busy. Try again after the current generation finishes.')
        try:
            key = (str(checkpoint), checkpoint.stat().st_mtime_ns)
            if not self.cache or self.cache[0] != key:
                manifest = read_json(path / 'data-manifest.json')
                if not manifest or digest(path / 'tokenizer.json') != manifest['sha256']['tokenizer.json']:
                    raise ValueError('The checkpoint tokenizer does not match its manifest.')
                state = torch.load(checkpoint, map_location='cpu', weights_only=True)
                model = GPT(GPTConfig(**state['model_config'])).to(self.device).eval()
                model.load_state_dict(state['model'])
                tokenizer = Tokenizer.from_file(str(path / 'tokenizer.json'))
                self.cache = (key, model, tokenizer, state['step'])
            _, model, tokenizer, step = self.cache
            encoded = tokenizer.encode(prompt).ids
            ids = torch.tensor([encoded[-model.config.block_size:]], dtype=torch.long, device=self.device)
            eos = tokenizer.token_to_id('<|endoftext|>')
            torch.manual_seed(seed)
            start = time.monotonic()
            generated = []
            with torch.no_grad():
                for _ in range(count):
                    ids = model.generate(ids, 1, temperature=temperature, top_k=50)
                    token = ids[0, -1].item()
                    if token == eos:
                        break
                    generated.append(token)
            return {'text': tokenizer.decode(generated), 'tokens': len(generated), 'seconds': time.monotonic() - start,
                    'device': self.device, 'checkpoint_step': step,
                    'context_truncated': len(encoded) > model.config.block_size,
                    'context_limit': model.config.block_size, 'ended_at_eos': len(generated) < count}
        finally:
            self.lock.release()


class Handler(BaseHTTPRequestHandler):
    server_version = 'MCubed/1'

    def reply(self, status, data, content_type='application/json', filename=None):
        payload = data if isinstance(data, bytes) else json.dumps(data, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if filename:
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def authorized(self):
        # Host validation prevents DNS-rebinding access to a local service.
        host = self.headers.get('Host', '').split(':')[0]
        if host not in self.server.allowed_hosts:
            self.reply(403, {'error': 'Unrecognized host.'})
            return False
        if self.server.token and not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + self.server.token):
            self.reply(401, {'error': 'Access token required.'})
            return False
        return True

    def do_GET(self):
        route = urlparse(self.path)
        if route.path in ('/', '/app.js', '/style.css'):
            name = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}[route.path]
            mime = {'index.html': 'text/html; charset=utf-8', 'app.js': 'text/javascript', 'style.css': 'text/css'}[name]
            self.reply(200, (STATIC / name).read_bytes(), mime)
            return
        if not self.authorized():
            return
        try:
            q = parse_qs(route.query)
            if route.path == '/api/runs':
                result = [self.server.lab.run(p.name) for p in (self.server.lab.root / 'runs').iterdir()
                          if p.is_dir() and ID.fullmatch(p.name) and (p / 'config.json').exists()]
                self.reply(200, {'runs': sorted(result, key=lambda r: r['updated_at'] or 0, reverse=True), 'inference_device': self.server.lab.device})
            elif route.path == '/api/datasets':
                self.reply(200, {'datasets': self.server.lab.datasets()})
            elif route.path == '/api/stories':
                self.reply(200, self.server.lab.stories(q.get('id', [''])[0], q.get('split', ['train'])[0],
                    int(q.get('offset', ['0'])[0]), int(q.get('limit', ['5'])[0])))
            elif route.path == '/api/archive':
                version_dir = self.server.lab.child('datasets', q.get('id', [''])[0])
                version = read_json(version_dir / 'version.json')
                archive = (self.server.lab.root / version['archive']['path']).resolve()
                if not archive.is_relative_to((self.server.lab.root / 'data/archives').resolve()):
                    raise ValueError('Invalid archive location.')
                if digest(archive) != version['archive']['sha256']:
                    raise ValueError('Archive checksum mismatch.')
                self.reply(200, archive.read_bytes(), 'application/gzip', archive.name)
            else:
                self.reply(404, {'error': 'Not found.'})
        except (ValueError, KeyError, FileNotFoundError) as error:
            self.reply(400, {'error': str(error)})

    def do_POST(self):
        if not self.authorized():
            return
        origin = self.headers.get('Origin')
        if origin and urlparse(origin).netloc != self.headers.get('Host'):
            self.reply(403, {'error': 'Cross-origin requests are not accepted.'})
            return
        if self.path != '/api/generate':
            self.reply(404, {'error': 'Not found.'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 100000 or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                raise ValueError('Send a bounded JSON request.')
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError('Expected a JSON object.')
            self.reply(200, self.server.lab.generate(body))
        except BlockingIOError as error:
            self.reply(409, {'error': str(error)})
        except (ValueError, TypeError, KeyError, FileNotFoundError) as error:
            self.reply(400, {'error': str(error)})
        except Exception:
            self.reply(500, {'error': 'Inference failed. Check the server log.'})
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda', 'auto'], default='cpu')
    p.add_argument('--allowed-host', action='append', default=[])
    a = p.parse_args()
    token = os.environ.get('MCUBED_DASHBOARD_TOKEN', '')
    if a.host not in ('127.0.0.1', 'localhost') and len(token) < 24:
        p.error('Non-loopback binding requires MCUBED_DASHBOARD_TOKEN of at least 24 characters.')
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    server.lab = Lab(a.root, a.device)
    server.token = token
    server.allowed_hosts = {'127.0.0.1', 'localhost', *a.allowed_host}
    print(f'M³ dashboard: http://{a.host}:{a.port} (inference: {server.lab.device})', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

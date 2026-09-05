"""Exercise the local HTTP API without browser automation or a paid service."""
import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request


def check(base, run, dataset):
    def request(path, body=None, headers=None):
        merged = {'Content-Type': 'application/json'} if body is not None else {}
        token = os.environ.get('MCUBED_DASHBOARD_TOKEN', '')
        if token:
            merged['Authorization'] = 'Bearer ' + token
        merged.update(headers or {})
        req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None, headers=merged)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, r.read(), r.headers
        except urllib.error.HTTPError as error:
            return error.code, error.read(), error.headers
    status, raw, _ = request('/api/runs')
    assert status == 200 and any(r['id'] == run for r in json.loads(raw)['runs'])
    status, raw, _ = request('/api/datasets')
    assert status == 200
    version = next(d['version'] for d in json.loads(raw)['datasets'] if d['version']['id'] == dataset)
    status, raw, _ = request(f'/api/stories?id={dataset}&split=train&offset=0&limit=2')
    assert status == 200 and len(json.loads(raw)['stories']) == 2
    assert request(f'/api/stories?id={dataset}&split=test')[0] == 400
    assert request('/api/stories?id=..%2F..&split=train')[0] == 400
    assert request('/api/runs', headers={'Host': 'untrusted.example'})[0] == 403
    body = {'run': run, 'prompt': 'Once upon a time, a little rabbit', 'tokens': 12, 'temperature': 0.7, 'seed': 42}
    assert request('/api/generate', body, {'Origin': 'https://untrusted.example'})[0] == 403
    assert request('/api/generate', {**body, 'tokens': 100000})[0] == 400
    status, raw, _ = request('/api/generate', body)
    result = json.loads(raw)
    assert status == 200 and isinstance(result['text'], str) and result['tokens'] <= 12
    status, archive, headers = request(f'/api/archive?id={dataset}')
    assert status == 200 and hashlib.sha256(archive).hexdigest() == version['archive']['sha256']
    assert 'attachment' in headers['Content-Disposition']
    print(json.dumps({'passed': ['run records', 'dataset versions', 'story paging', 'sealed test split',
        'path traversal rejection', 'host rejection', 'origin rejection', 'generation bounds',
        'checkpoint inference', 'archive checksum and download'], 'inference_device': result['device']}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8765')
    p.add_argument('--run', default='coherence-small-001')
    p.add_argument('--dataset', default='tinystories-20k')
    a = p.parse_args()
    check(a.url, a.run, a.dataset)

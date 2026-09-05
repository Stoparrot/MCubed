import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import torch


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def device_for(name):
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable; run natively on macOS, or select --device cpu.')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable in this Python environment.')
    return name


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()


def provenance():
    env = os.environ.copy()
    if platform.system() == 'Darwin' and Path('/Library/Developer/CommandLineTools').exists():
        env.setdefault('DEVELOPER_DIR', '/Library/Developer/CommandLineTools')
    def git(*args):
        try:
            return subprocess.check_output(['git', *args], env=env, stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    files = sorted(p for root in ('mcubed', 'vendor', 'configs') for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in str(p))
    files += [Path('requirements.txt')]
    hashes = {str(p): digest(p) for p in files}
    return {'git_commit': git('rev-parse', 'HEAD'), 'git_status': git('status', '--porcelain'),
            'source_sha256': hashes, 'python': platform.python_version(), 'platform': platform.platform(),
            'torch': str(torch.__version__),
            'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}


def verify_data(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if digest(root / name) != expected:
            raise ValueError(f'Dataset file changed: {name}; prepare a new version.')
    return manifest

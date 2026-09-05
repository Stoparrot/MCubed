"""Package or restore an exact dataset version with verified content hashes."""
import argparse
import gzip
import io
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcubed.common import digest, verify_data, write_json


def pack(name):
    registered = Path('datasets') / name
    version = json.loads((registered / 'version.json').read_text())
    root = Path(version['artifact_directory'])
    manifest = verify_data(root)
    if digest(root / 'manifest.json') != version['manifest_sha256']:
        raise ValueError('Registered version does not match local data.')
    out = Path('data/archives') / f'{name}-{version["manifest_sha256"][:12]}.tar.gz'
    out.parent.mkdir(parents=True, exist_ok=True)
    notice = (registered / 'DATA_NOTICE.md').read_bytes()
    files = {p: (root / p).read_bytes() for p in [*manifest['sha256'], 'manifest.json']}
    files['DATA_NOTICE.md'] = notice
    # Sidecars identify which original data files were transformed by preparation.
    for p in manifest['sha256']:
        files[p + '.NOTICE'] = notice
    with out.open('xb') as f, gzip.GzipFile(filename='', mode='wb', fileobj=f, mtime=0, compresslevel=6) as gz:
        with tarfile.open(fileobj=gz, mode='w') as tar:
            for p, content in sorted(files.items()):
                info = tarfile.TarInfo(f'{name}/{p}')
                info.size = len(content)
                info.mode = 0o644
                info.mtime = 0
                tar.addfile(info, io.BytesIO(content))
    version['archive'] = {'path': out.as_posix(), 'sha256': digest(out), 'bytes': out.stat().st_size,
                          'format': 'tar.gz', 'published_url': None}
    write_json(registered / 'version.json', version)
    print(json.dumps(version['archive'], indent=2))


def restore(name, archive, out):
    registered = Path('datasets') / name
    version = json.loads((registered / 'version.json').read_text())
    if digest(archive) != version['archive']['sha256']:
        raise ValueError('Archive checksum mismatch.')
    manifest = json.loads((registered / 'manifest.json').read_text())
    expected = {'manifest.json', *manifest['sha256'], 'DATA_NOTICE.md',
                *(name + '.NOTICE' for name in manifest['sha256'])}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, 'r:gz') as tar:
        # Extract only expected regular files, never arbitrary archive paths or links.
        for filename in sorted(expected):
            member = tar.getmember(f'{name}/{filename}')
            if not member.isfile():
                raise ValueError('Expected a regular file in archive.')
            with tar.extractfile(member) as src, (out / filename).open('wb') as dest:
                import shutil
                shutil.copyfileobj(src, dest)
    if digest(out / 'manifest.json') != version['manifest_sha256']:
        raise ValueError('Restored manifest mismatch.')
    verify_data(out)
    print(f'Restored and verified {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['pack', 'restore'])
    p.add_argument('--name', required=True)
    p.add_argument('--archive')
    p.add_argument('--out')
    a = p.parse_args()
    if Path(a.name).name != a.name or a.name in ('.', '..'):
        p.error('Invalid dataset name.')
    if a.action == 'pack':
        pack(a.name)
    else:
        if not a.archive or not a.out:
            p.error('restore needs --archive and --out')
        restore(a.name, a.archive, a.out)

"""Commit-sized dataset provenance; large content stays under data/ or mounted storage."""
import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcubed.common import digest, provenance, verify_data, write_json


def register(root, name):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,79}', name):
        raise ValueError('Use a lowercase dataset version name with letters, numbers and hyphens.')
    root = Path(root)
    manifest = verify_data(root)
    target = Path('datasets') / name
    target.mkdir(parents=True, exist_ok=False)
    for file in ('manifest.json', 'tokenizer.json'):
        shutil.copyfile(root / file, target / file)
    write_json(target / 'version.json', {'id': name,
        'registered_at': datetime.now(timezone.utc).isoformat(),
        'manifest_sha256': digest(root / 'manifest.json'),
        'artifact_directory': root.as_posix(),
        'registration_source_commit': provenance()['git_commit'],
        'preparation_script_sha256_at_registration': digest('mcubed/prepare.py'),
        'source_revision': manifest['revision'],
        'storage': 'Local data/ directory now; mount the same relative layout on Linux. Large files are not in Git.',
        'recovery': 'Use pinned upstream URLs and recorded selection, plus the committed tokenizer; verify resulting file hashes before use.'})
    print(target)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--name', required=True)
    a = p.parse_args()
    register(a.data, a.name)

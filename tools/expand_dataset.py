"""Expand training stories while freezing the parent tokenizer and held-out splits."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcubed.common import digest, verify_data, write_json
from mcubed.prepare import select, stories


def expand(parent, out, count):
    parent, out = Path(parent), Path(out)
    manifest = verify_data(parent)
    if out.exists() or count <= manifest['splits']['train']['stories']:
        raise ValueError('Choose a new output directory and a larger training count.')
    seen = {hashlib.sha256(' '.join(json.loads(line)['text'].split()).encode()).hexdigest()
            for split in ('val', 'test') for line in (parent / f'{split}.jsonl').read_text().splitlines()}
    print(f'Streaming {count} training stories; excluding frozen held-out hashes...', flush=True)
    texts = select(stories(manifest['urls']['train']), count, seen)
    out.mkdir(parents=True)
    for name in ('tokenizer.json', 'val.jsonl', 'val.bin', 'test.jsonl', 'test.bin'):
        shutil.copyfile(parent / name, out / name)
    tokenizer = Tokenizer.from_file(str(out / 'tokenizer.json'))
    total = 0
    with (out / 'train.jsonl').open('w') as raw, (out / 'train.bin').open('wb') as binary:
        for text in texts:
            raw.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
            ids = tokenizer.encode(text).ids + [manifest['eos_id']]
            np.asarray(ids, dtype='<u2').tofile(binary)
            total += len(ids)
    manifest['parent_manifest_sha256'] = digest(parent / 'manifest.json')
    manifest['selection'] = 'first N unique upstream train stories excluding frozen val/test hashes; parent tokenizer and held-out files copied byte-for-byte'
    manifest['splits']['train'] = {'stories': count, 'tokens': total}
    manifest['sha256'] = {p.name: digest(p) for p in sorted(out.iterdir())}
    write_json(out / 'manifest.json', manifest)
    verify_data(out)
    print(json.dumps(manifest['splits'], indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', default='data/tinystories-20k')
    p.add_argument('--out', required=True)
    p.add_argument('--train-stories', type=int, default=200000)
    a = p.parse_args()
    expand(a.parent, a.out, a.train_stories)

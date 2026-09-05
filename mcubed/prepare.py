"""Stream a bounded, pinned TinyStories subset. Never train a tokenizer on held-out data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import requests
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from .common import digest, write_json

REVISION = 'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'
BASE = f'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{REVISION}'
EOS = '<|endoftext|>'


def stories(url):
    with requests.get(url, stream=True, timeout=(30, 90)) as response:
        response.raise_for_status()
        response.encoding = 'utf-8'
        lines = []
        for line in response.iter_lines(decode_unicode=True):
            if line.strip() == EOS:
                text = '\n'.join(lines).strip()
                if text:
                    yield text
                lines = []
            else:
                lines.append(line)
        if lines and '\n'.join(lines).strip():
            yield '\n'.join(lines).strip()


def select(source, count, seen):
    selected = []
    for text in source:
        key = hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        selected.append(text)
        if len(selected) == count:
            return selected
    raise ValueError(f'Only found {len(selected)} unique stories; requested {count}.')


def prepare(out, train_count=20000, heldout_count=1000, vocab_size=2048):
    out = Path(out)
    if out.exists():
        raise ValueError(f'{out} already exists. Choose a new directory to keep data immutable.')
    if train_count < 1 or heldout_count < 2 or not 257 <= vocab_size <= 65535:
        raise ValueError('Need positive training size, >=2 held-out stories and vocab 257..65535.')
    seen = set()
    urls = {s: f'{BASE}/TinyStories-{s}.txt' for s in ('train', 'valid')}
    print('Streaming training stories...', flush=True)
    train = select(stories(urls['train']), train_count, seen)
    print('Streaming held-out stories...', flush=True)
    heldout = select(stories(urls['valid']), heldout_count, seen)
    splits = {'train': train, 'val': heldout[::2], 'test': heldout[1::2]}
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(train, trainers.BpeTrainer(vocab_size=vocab_size,
        special_tokens=[EOS], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    out.mkdir(parents=True)
    tokenizer.save(str(out / 'tokenizer.json'))
    manifest = {'dataset': 'roneneldan/TinyStories', 'revision': REVISION, 'urls': urls,
        'variant': 'original TinyStories text files (not V2-GPT4)',
        'selection': 'first N unique stories in upstream order; whitespace-normalized SHA256 dedup across selected splits; valid alternates val/test',
        'vocab_size': tokenizer.get_vocab_size(), 'eos_id': tokenizer.token_to_id(EOS),
        'splits': {}, 'sha256': {}}
    for split, texts in splits.items():
        raw = out / f'{split}.jsonl'
        with raw.open('w') as f, (out / f'{split}.bin').open('wb') as binary:
            total = 0
            for text in texts:
                f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                ids = tokenizer.encode(text).ids + [manifest['eos_id']]
                np.asarray(ids, dtype='<u2').tofile(binary)
                total += len(ids)
        manifest['splits'][split] = {'stories': len(texts), 'tokens': total}
    for path in sorted(out.iterdir()):
        manifest['sha256'][path.name] = digest(path)
    write_json(out / 'manifest.json', manifest)
    print(json.dumps(manifest['splits'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='data/tinystories-20k')
    parser.add_argument('--train-stories', type=int, default=20000)
    parser.add_argument('--heldout-stories', type=int, default=1000)
    parser.add_argument('--vocab-size', type=int, default=2048)
    args = parser.parse_args()
    prepare(args.out, args.train_stories, args.heldout_stories, args.vocab_size)

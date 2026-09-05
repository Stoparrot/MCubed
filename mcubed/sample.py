"""Generate text from a local checkpoint; no external API."""
import argparse
import json
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from .model import GPT, GPTConfig
from .common import device_for, sync, write_json


def sample(args):
    if args.temperature <= 0 or args.tokens < 1:
        raise ValueError('Temperature and tokens must be positive.')
    device = device_for(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    checkpoint = Path(args.checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = GPT(GPTConfig(**state['model_config'])).to(device).eval()
    model.load_state_dict(state['model'])
    tokenizer = Tokenizer.from_file(str(checkpoint.parent / 'tokenizer.json'))
    ids = tokenizer.encode(args.prompt).ids or [tokenizer.token_to_id('<|endoftext|>')]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    sync(device)
    started = time.monotonic()
    with torch.no_grad():
        result = model.generate(x, args.tokens, temperature=args.temperature, top_k=50)
    sync(device)
    seconds = time.monotonic() - started
    output = {'prompt': args.prompt, 'text': tokenizer.decode(result[0].tolist(), skip_special_tokens=False),
              'checkpoint_step': state['step'], 'seed': args.seed, 'temperature': args.temperature,
              'device': device, 'new_tokens': args.tokens, 'seconds': seconds,
              'tokens_per_second': args.tokens / seconds,
              'timing_scope': 'generation only; includes first-call warmup, excludes model load; no KV cache'}
    if args.output:
        write_json(args.output, output)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--prompt', default='Once upon a time, a little rabbit')
    p.add_argument('--tokens', type=int, default=100)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    p.add_argument('--output')
    sample(p.parse_args())

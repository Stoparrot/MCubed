"""One-shot held-out test evaluation after model selection. Never used by training."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from .model import GPT, GPTConfig
from .common import device_for, digest, verify_data, write_json


def test(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError('Test report already exists. Preserve the original result.')
    torch.set_num_threads(4)
    device = device_for(args.device)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    root = Path(args.data)
    verify_data(root)
    if digest(root / 'manifest.json') != state['data_hash']:
        raise ValueError('Checkpoint and dataset do not match.')
    model = GPT(GPTConfig(**state['model_config'])).to(device).eval()
    model.load_state_dict(state['model'])
    tokens = np.memmap(root / 'test.bin', dtype='<u2', mode='r')
    block = state['model_config']['block_size']
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(tokens) - 1, block):
            n = min(block, len(tokens) - 1 - i)
            x = torch.tensor(np.array(tokens[i:i+n], dtype=np.int64)[None], device=device)
            y = torch.tensor(np.array(tokens[i+1:i+n+1], dtype=np.int64)[None], device=device)
            _, loss = model(x, y)
            total_loss += loss.item() * n
            count += n
    report = {'checkpoint': args.checkpoint, 'checkpoint_sha256': digest(args.checkpoint),
        'step': state['step'], 'tokens_scored': count, 'test_loss': total_loss/count,
        'test_perplexity': math.exp(total_loss/count), 'device': device,
        'protocol': 'all test tokens except first; nonoverlapping context windows; token-weighted NLL'}
    write_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', default='data/tinystories-20k')
    p.add_argument('--output', required=True)
    p.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    test(p.parse_args())

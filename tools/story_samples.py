"""Save every output from a fixed development prompt suite, using a local checkpoint."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tokenizers import Tokenizer
from mcubed.common import digest, write_json
from mcubed.model import GPT, GPTConfig


def generate_suite(checkpoint, output, suite_path):
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise ValueError('Preserve existing samples; choose a new output path.')
    suite = json.loads(Path(suite_path).read_text())
    torch.set_num_threads(4)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = GPT(GPTConfig(**state['model_config'])).eval()
    model.load_state_dict(state['model'])
    tokenizer = Tokenizer.from_file(str(checkpoint.parent / 'tokenizer.json'))
    eos = tokenizer.token_to_id('<|endoftext|>')
    samples = []
    for index, prompt in enumerate(suite['prompts']):
        torch.manual_seed(suite['seed'] + index)
        ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)
        start = time.monotonic()
        generated = []
        with torch.no_grad():
            for _ in range(suite['max_new_tokens']):
                result = model.generate(ids, 1, temperature=suite['temperature'], top_k=suite['top_k'])
                token = result[0, -1].item()
                if token == eos:
                    break
                generated.append(token)
                ids = result
        samples.append({'id': index + 1, 'prompt': prompt, 'continuation': tokenizer.decode(generated),
                        'new_tokens': len(generated), 'seconds': time.monotonic() - start})
    report = {'checkpoint_sha256': digest(checkpoint), 'checkpoint_step': state['step'],
              'suite_sha256': digest(suite_path), 'device': 'cpu', 'sampling': suite, 'samples': samples}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--suite', default='evals/story-prompts.json')
    a = p.parse_args()
    generate_suite(a.checkpoint, a.output, a.suite)

"""Single-device FP32 training of M³'s nanochat-derived GPT."""
import argparse
import json
import math
import signal
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from .model import GPT, GPTConfig
from .common import device_for, digest, provenance, sync, verify_data, write_json


def batch(tokens, config, generator, device):
    length = config['block_size']
    starts = torch.randint(len(tokens) - length, (config['batch_size'],), generator=generator).tolist()
    x = np.stack([tokens[i:i + length] for i in starts]).astype(np.int64)
    y = np.stack([tokens[i + 1:i + length + 1] for i in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def evaluate(model, tokens, config, device):
    # Identical evaluation windows each time, independent of training RNG.
    rng = torch.Generator().manual_seed(config['seed'] + 1)
    model.eval()
    losses = [model(*batch(tokens, config, rng, device))[1].item() for _ in range(config['eval_batches'])]
    model.train()
    return sum(losses) / len(losses)


def lr_at(step, c):
    if step < c['warmup_steps']:
        return c['learning_rate'] * (step + 1) / max(1, c['warmup_steps'])
    ratio = min(1, (step - c['warmup_steps']) / max(1, c['max_steps'] - c['warmup_steps'] - 1))
    return c['min_lr'] + 0.5 * (1 + math.cos(math.pi * ratio)) * (c['learning_rate'] - c['min_lr'])


def validate(c):
    for key in ('n_layer', 'n_head', 'n_embd', 'block_size', 'batch_size', 'gradient_accumulation',
                'max_steps', 'eval_interval', 'eval_batches', 'max_seconds'):
        if not isinstance(c[key], int) or c[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if c['n_embd'] % c['n_head'] or not 0 <= c['warmup_steps'] < c['max_steps']:
        raise ValueError('Embedding width must divide into heads; warmup must be below max_steps.')
    if not 0 < c['min_lr'] <= c['learning_rate'] or c['grad_clip'] <= 0 or c['weight_decay'] < 0:
        raise ValueError('Invalid optimizer settings')


def train(args):
    c = json.loads(Path(args.config).read_text())
    validate(c)
    if args.hourly_usd < 0 or not math.isfinite(args.hourly_usd):
        raise ValueError('Hourly rate must be a finite nonnegative value.')
    device = device_for(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(c['seed'])
    root, run = Path(args.data), Path(args.out)
    manifest = verify_data(root)
    data_hash = digest(root / 'manifest.json')
    tokens = {s: np.memmap(root / f'{s}.bin', dtype='<u2', mode='r') for s in ('train', 'val')}
    if any(len(t) <= c['block_size'] for t in tokens.values()):
        raise ValueError('Not enough data for the context length.')
    model_config = GPTConfig(**{k: c[k] for k in ('n_layer', 'n_head', 'n_embd', 'block_size')},
                             vocab_size=manifest['vocab_size'])
    model = GPT(model_config).to(device)
    optimizer = model.configure_optimizers(c['weight_decay'], c['learning_rate'], (0.9, 0.95), device)
    rng = torch.Generator().manual_seed(c['seed'])
    step, elapsed_prior, best, initial_val = 0, 0.0, float('inf'), None
    current_source = provenance()
    if args.resume:
        ckpt = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
        if ckpt['config'] != c or ckpt['data_hash'] != data_hash:
            raise ValueError('Resume requires the same config and dataset; create a new run for changes.')
        if ckpt['source_sha256'] != current_source['source_sha256']:
            raise ValueError('Source changed since checkpoint. Restore it before resuming.')
        if ckpt['device'] != device or ckpt['hourly_usd'] != args.hourly_usd:
            raise ValueError('Resume requires the same device and hourly rate.')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        rng.set_state(ckpt['batch_rng'])
        torch.set_rng_state(ckpt['torch_rng'])
        step, elapsed_prior, best, initial_val = (ckpt[k] for k in ('step', 'elapsed_seconds', 'best_val', 'initial_val'))
    else:
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / 'config.json', c)
        (run / 'tokenizer.json').write_bytes((root / 'tokenizer.json').read_bytes())
        write_json(run / 'data-manifest.json', manifest)
        write_json(run / 'environment.json', {**current_source, 'device': device,
            'parameters_total': sum(p.numel() for p in model.parameters()),
            'parameters_without_token_and_position_embeddings': sum(p.numel() for p in model.parameters())
                - model.embedding.weight.numel(),
            'hourly_usd': args.hourly_usd, 'cost_scope': 'process runtime x supplied rate; excludes idle VM/storage/network/electricity/API fees',
            'data_hash': data_hash})
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    started = time.monotonic()
    elapsed = lambda: elapsed_prior + time.monotonic() - started
    tokens_per_step = c['batch_size'] * c['block_size'] * c['gradient_accumulation']
    log = (run / 'metrics.jsonl').open('a', buffering=1)
    def emit(event):
        event.update(elapsed_seconds=round(elapsed(), 3), step=step, tokens_seen=step * tokens_per_step,
                     estimated_compute_usd=elapsed() / 3600 * args.hourly_usd)
        line = json.dumps(event, allow_nan=False)
        log.write(line + '\n')
        print(line, flush=True)
    def save(path):
        payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'model_config': asdict(model_config), 'config': c, 'step': step,
            'elapsed_seconds': elapsed(), 'best_val': best, 'initial_val': initial_val,
            'batch_rng': rng.get_state(), 'torch_rng': torch.get_rng_state(),
            'data_hash': data_hash, 'source_sha256': current_source['source_sha256'],
            'device': device, 'hourly_usd': args.hourly_usd}
        tmp = path.with_suffix('.tmp')
        torch.save(payload, tmp)
        tmp.replace(path)
    def measure():
        nonlocal best, initial_val
        val = evaluate(model, tokens['val'], c, device)
        train_loss = evaluate(model, tokens['train'], c, device)
        if not math.isfinite(val + train_loss):
            raise FloatingPointError('Nonfinite evaluation loss.')
        if initial_val is None:
            initial_val = val
        improved = val < best
        best = min(best, val)
        emit({'event': 'eval', 'train_loss': train_loss, 'val_loss': val, 'val_perplexity': math.exp(val)})
        if improved:
            save(run / 'best.pt')
        save(run / 'last.pt')
    try:
        measure()
        last_eval = step
        while step < c['max_steps'] and elapsed() < c['max_seconds'] and not stopped:
            if (run / 'STOP').exists():
                stopped = True
                break
            sync(device)
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            lr = lr_at(step, c)
            for group in optimizer.param_groups:
                group['lr'] = lr
            loss_sum = 0.0
            for _ in range(c['gradient_accumulation']):
                _, loss = model(*batch(tokens['train'], c, rng, device))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss.')
                loss_sum += loss.item()
                (loss / c['gradient_accumulation']).backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            sync(device)
            seconds = time.monotonic() - tick
            step += 1
            if step == 1 or step % 10 == 0:
                emit({'event': 'train', 'loss': loss_sum / c['gradient_accumulation'], 'lr': lr,
                      'grad_norm': float(norm), 'tokens_per_second': tokens_per_step / seconds,
                      'step_seconds': seconds})
            if step % c['eval_interval'] == 0:
                measure()
                last_eval = step
        if last_eval != step:
            measure()
        save(run / 'last.pt')
        status = 'stopped' if stopped else 'complete' if step == c['max_steps'] else 'time_limit'
        summary = {'status': status, 'step': step, 'best_val_loss': best, 'initial_val_loss': initial_val,
            'loss_reduced': best < initial_val, 'elapsed_seconds': elapsed(), 'tokens_seen': step * tokens_per_step,
            'estimated_compute_usd': elapsed() / 3600 * args.hourly_usd,
            'parameters_total': sum(p.numel() for p in model.parameters()),
            'weights_fp32_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
            'checkpoint_bytes': (run / 'last.pt').stat().st_size}
        write_json(run / 'summary.json', summary)
        emit({'event': 'end', 'status': status})
    except Exception as error:
        # Leave last known finite checkpoint intact.
        write_json(run / 'failure.json', {'step': step, 'error': str(error), 'elapsed_seconds': elapsed()})
        raise
    finally:
        log.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/pilot.json')
    p.add_argument('--data', default='data/tinystories-20k')
    p.add_argument('--out', required=True)
    p.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    p.add_argument('--hourly-usd', type=float, default=0)
    p.add_argument('--resume', action='store_true')
    train(p.parse_args())

"""Read-only terminal monitoring and compute estimates from local run records."""
import argparse
import json
import statistics
import time
from pathlib import Path


def report(root, hourly_usd=None):
    root = Path(root)
    rows = []
    for line in (root / 'metrics.jsonl').read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a concurrent writer may not yet have finished the last line
    if not rows:
        return {'status': 'waiting for metrics'}
    config = json.loads((root / 'config.json').read_text())
    environment = json.loads((root / 'environment.json').read_text())
    measured = [r['step_seconds'] for r in rows if r['event'] == 'train' and r['step'] > 1]
    evals = [r for r in rows if r['event'] == 'eval']
    current = rows[-1]
    seconds = statistics.median(measured) if measured else None
    rate = environment['hourly_usd'] if hourly_usd is None else hourly_usd
    if rate < 0:
        raise ValueError('Hourly rate must be nonnegative.')
    remaining = (config['max_steps'] - current['step']) * seconds if seconds else None
    return {'run': str(root), 'device': environment['device'], 'parameters': environment['parameters_total'],
        'last_event': current, 'latest_validation': evals[-1] if evals else None,
        'median_logged_step_seconds': seconds, 'remaining_training_seconds_estimate': remaining,
        'remaining_compute_usd_estimate': remaining / 3600 * rate if remaining is not None else None,
        'hourly_usd': rate,
        'estimate_scope': 'same hardware/config only; excludes evaluation, startup, VM idle, storage, network and APIs; provider rates must be supplied, not inferred'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--hourly-usd', type=float)
    p.add_argument('--watch', action='store_true')
    args = p.parse_args()
    try:
        while True:
            print(json.dumps(report(args.run, args.hourly_usd), indent=2), flush=True)
            if not args.watch:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        pass

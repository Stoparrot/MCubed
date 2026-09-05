import argparse
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from mcubed.model import GPT, GPTConfig
from mcubed.common import digest, verify_data, write_json
from mcubed.prepare import select
from tools.expand_dataset import expand
from mcubed.train import batch, evaluate, train
from mcubed.evaluate import test as score_test


def model_from_checkpoint(state):
    model = GPT(GPTConfig(**state['model_config']))
    model.load_state_dict(state['model'])
    return model


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_causal_attention(self):
        model = GPT(GPTConfig(vocab_size=16, block_size=8, n_layer=1, n_head=2, n_embd=16)).eval()
        # Enable the residual path so the test exercises attention at initialization.
        torch.nn.init.normal_(model.blocks[0].attn.proj.weight, std=0.1)
        x = torch.randint(16, (1, 8))
        changed = x.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 16
        a = model(x, x)[0]
        b = model(changed, changed)[0]
        torch.testing.assert_close(a[:, :4], b[:, :4], rtol=0, atol=0)

    def test_next_token_shift_and_evaluation_rng(self):
        c = {'block_size': 8, 'batch_size': 2, 'seed': 7, 'eval_batches': 2}
        rng = torch.Generator().manual_seed(7)
        data = np.arange(128, dtype=np.uint16) % 16
        x, y = batch(data, c, rng, 'cpu')
        torch.testing.assert_close(x[:, 1:], y[:, :-1])
        model = GPT(GPTConfig(vocab_size=16, block_size=8, n_layer=1, n_head=2, n_embd=16))
        before = rng.get_state().clone()
        self.assertEqual(evaluate(model, data, c, 'cpu'), evaluate(model, data, c, 'cpu'))
        self.assertTrue(torch.equal(before, rng.get_state()))

    def test_dedup_across_selected_splits(self):
        seen = set()
        self.assertEqual(select(iter(['a story', 'a  story', 'new']), 2, seen), ['a story', 'new'])
        self.assertEqual(select(iter(['a story', 'held out']), 1, seen), ['held out'])

    def test_expansion_preserves_heldout_and_tokenizer(self):
        from unittest.mock import patch
        from mcubed.prepare import prepare
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            parent, out = Path(tmp) / 'parent', Path(tmp) / 'expanded'
            with patch('mcubed.prepare.stories', side_effect=[iter(['a cat', 'a dog']), iter(['a bird', 'a fish'])]):
                prepare(parent, 2, 2, 257)
            with patch('tools.expand_dataset.stories', return_value=iter(['a bird', 'a fish', 'a cat', 'a  cat', 'a dog', 'a horse'])):
                expand(parent, out, 3)
            for name in ('tokenizer.json', 'val.bin', 'val.jsonl', 'test.bin', 'test.jsonl'):
                self.assertEqual(digest(parent / name), digest(out / name))
            self.assertEqual([json.loads(line)['text'] for line in (out / 'train.jsonl').read_text().splitlines()],
                             ['a cat', 'a dog', 'a horse'])
            self.assertEqual(verify_data(out)['parent_manifest_sha256'], digest(parent / 'manifest.json'))

    def test_heldout_scoring_covers_tail_and_preserves_report(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            (np.arange(20) % 16).astype('<u2').tofile(root / 'test.bin')
            write_json(root / 'manifest.json', {'sha256': {'test.bin': digest(root / 'test.bin')}})
            c = dict(vocab_size=16, block_size=8, n_layer=1, n_head=2, n_embd=16)
            model = GPT(GPTConfig(**c))
            torch.nn.init.zeros_(model.lm_head.weight)  # Uniform distribution: exact reference NLL.
            torch.save({'model_config': c, 'model': model.state_dict(), 'step': 0,
                'data_hash': digest(root / 'manifest.json')}, root / 'model.pt')
            args = argparse.Namespace(output=str(root / 'report.json'), device='cpu',
                checkpoint=str(root / 'model.pt'), data=str(root))
            score_test(args)
            report = json.loads(Path(args.output).read_text())
            self.assertEqual(report['tokens_scored'], 19)
            self.assertAlmostEqual(report['test_loss'], math.log(16), places=6)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                score_test(args)

    def test_stop_resume_matches_uninterrupted_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data = root / 'data'
            data.mkdir()
            for name in ('train', 'val'):
                (np.arange(512) % 16).astype('<u2').tofile(data / f'{name}.bin')
            (data / 'tokenizer.json').write_text('{}')
            write_json(data / 'manifest.json', {'vocab_size': 16,
                'sha256': {p.name: digest(p) for p in data.iterdir()}})
            c = json.loads(Path('configs/pilot.json').read_text())
            c.update(n_layer=1, n_head=2, n_embd=16, block_size=8, batch_size=2,
                max_steps=3, warmup_steps=1, eval_interval=2, eval_batches=2)
            config = root / 'config.json'
            write_json(config, c)
            def args(out, resume=False):
                return argparse.Namespace(config=str(config), data=str(data), out=str(out),
                    device='cpu', hourly_usd=0.0, resume=resume)
            full, resumed = root / 'full', root / 'resumed'
            train(args(full))
            # Stop after the first optimizer step, then resume with an unchanged schedule.
            from unittest.mock import patch
            original = torch.optim.AdamW.step
            def stop_after_step(opt, *a, **kw):
                result = original(opt, *a, **kw)
                (resumed / 'STOP').touch()
                return result
            with patch.object(torch.optim.AdamW, 'step', stop_after_step):
                train(args(resumed))
            self.assertEqual(json.loads((resumed / 'summary.json').read_text())['status'], 'stopped')
            (resumed / 'STOP').unlink()
            train(args(resumed, True))
            a = torch.load(full / 'last.pt', weights_only=True)
            b = torch.load(resumed / 'last.pt', weights_only=True)
            self.assertEqual(b['step'], 3)
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
            # Warm start loads only weights into a fresh run, with an explicit lineage.
            warm = root / 'warm'
            warm_args = args(warm)
            warm_args.init_from = str(full / 'last.pt')
            def check_first_step(opt, *aa, **kw):
                self.assertEqual(len(opt.state), 0)
                (warm / 'STOP').touch()
                return original(opt, *aa, **kw)
            with patch.object(torch.optim.AdamW, 'step', check_first_step):
                train(warm_args)
            w = torch.load(warm / 'last.pt', weights_only=True)
            self.assertEqual(w['step'], 1)
            self.assertEqual(w['initialization']['sha256'], digest(full / 'last.pt'))
            self.assertEqual(w['initialization']['ancestor_tokens_seen'], 3 * 2 * 8 * c['gradient_accumulation'])
            self.assertEqual(w['initial_val'], evaluate(model_from_checkpoint(a),
                np.memmap(data / 'val.bin', dtype='<u2', mode='r'), c, 'cpu'))
            c['n_layer'] = 2
            write_json(config, c)
            warm_args.out = str(root / 'incompatible')
            with self.assertRaisesRegex(ValueError, 'same model and dataset'):
                train(warm_args)
            with (data / 'val.bin').open('ab') as f:
                f.write(b'xx')
            with self.assertRaisesRegex(ValueError, 'Dataset file changed'):
                verify_data(data)


if __name__ == '__main__':
    unittest.main()

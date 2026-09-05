import json
import tempfile
import unittest
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from dashboard.server import Lab
from mcubed.common import digest, write_json
from mcubed.model import GPT, GPTConfig


class DashboardTests(unittest.TestCase):
    def test_data_versions_reject_traversal_test_access_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / 'data/v1'
            raw.mkdir(parents=True)
            (raw / 'train.jsonl').write_text('{"text":"First story."}\n{"text":"Second story."}\n')
            registered = root / 'datasets/v1'
            registered.mkdir(parents=True)
            write_json(registered / 'version.json', {'artifact_directory': 'data/v1'})
            write_json(registered / 'manifest.json', {'splits': {'train': {'stories': 2}},
                'sha256': {'train.jsonl': digest(raw / 'train.jsonl')}})
            lab = Lab(root)
            self.assertEqual(lab.stories('v1', 'train', 1, 1)['stories'][0]['text'], 'Second story.')
            for identifier in ('../v1', '/tmp', '..'):
                with self.assertRaises(ValueError):
                    lab.child('datasets', identifier)
            with self.assertRaisesRegex(ValueError, 'sealed'):
                lab.stories('v1', 'test', 0, 1)
            (raw / 'train.jsonl').write_text('{"text":"Changed."}\n')
            with self.assertRaisesRegex(ValueError, 'changed'):
                lab.stories('v1', 'train', 0, 1)

    def test_inference_uses_saved_tokenizer_and_bounds_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / 'runs/demo'
            run.mkdir(parents=True)
            tokenizer = Tokenizer(models.BPE())
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer.decoder = decoders.ByteLevel()
            tokenizer.train_from_iterator(['a tiny story'], trainers.BpeTrainer(vocab_size=257,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=['<|endoftext|>'], show_progress=False))
            tokenizer.save(str(run / 'tokenizer.json'))
            c = dict(vocab_size=tokenizer.get_vocab_size(), n_layer=1, n_embd=16, n_head=2, block_size=8)
            model = GPT(GPTConfig(**c))
            torch.save({'model_config': c, 'model': model.state_dict(), 'step': 3}, run / 'best.pt')
            write_json(run / 'data-manifest.json', {'sha256': {'tokenizer.json': digest(run / 'tokenizer.json')}})
            lab = Lab(root)
            request = {'run': 'demo', 'prompt': 'a tiny story ' * 4, 'tokens': 2, 'seed': 42}
            result = lab.generate(request)
            self.assertTrue(result['context_truncated'])
            self.assertLessEqual(result['tokens'], 2)
            self.assertEqual(result['checkpoint_step'], 3)
            self.assertEqual(result['text'], lab.generate(request)['text'])
            for change in ({'tokens': 257}, {'temperature': float('nan')}, {'run': '../demo'}, {'prompt': ''}):
                with self.assertRaises(ValueError):
                    lab.generate({**request, **change})
            lab.cache = None
            (run / 'tokenizer.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'tokenizer'):
                lab.generate(request)


if __name__ == '__main__':
    unittest.main()

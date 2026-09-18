import json
from pathlib import Path
import tempfile
import unittest

from collect import clean_html
from common import config, encode_example, read_jsonl
from prepare import split_pairs
from train import Collator, add_lora, validate_data
from style import markers


class DataTests(unittest.TestCase):
    def test_capitalized_phrase_before_punctuation(self):
        self.assertEqual(markers('Time to Do The Thing.')['capitalized_phrase'], 1)

    def test_inline_html_stays_readable(self):
        self.assertEqual(clean_html('<p>We <em>would</em> be excited.</p><script>bad()</script><p>Next.</p>'),
                         'We would be excited.\n\nNext.')

    def test_semantic_families_do_not_leak(self):
        rows = read_jsonl('style_pairs.jsonl')
        splits = split_pairs(rows, 0.1, 42)
        groups = [{r['group'] for r in splits[k]} for k in ('train', 'validation')]
        self.assertFalse(groups[0] & groups[1])
        self.assertEqual(sum(map(len, splits.values())), len(rows))
        reordered = split_pairs(list(reversed(rows)), 0.1, 42)
        for split in splits:
            self.assertEqual({r['id'] for r in splits[split]}, {r['id'] for r in reordered[split]})

    def test_real_template_mask_and_padding(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('data/tokenizer', local_files_only=True)
        messages = [{'role': 'system', 'content': 'Be helpful.'},
                    {'role': 'user', 'content': 'A brief invitation?'},
                    {'role': 'assistant', 'content': "We'd be excited to hear from you."}]
        example = encode_example(tokenizer, messages)
        supervised = tokenizer.decode([x for x in example['labels'] if x != -100])
        self.assertEqual(supervised, "We'd be excited to hear from you.<|im_end|>\n")
        batch = Collator(tokenizer.pad_token_id)([example, {k: v[:-2] for k, v in example.items()}])
        self.assertEqual(batch['labels'][1, -1].item(), -100)
        self.assertEqual(batch['attention_mask'][1, -1].item(), 0)

    def test_prepared_data_is_style_only(self):
        data, manifest = validate_data(config())
        originals = {r['id']: r for r in read_jsonl('style_pairs.jsonl')}
        for split in ('train', 'validation'):
            for row in read_jsonl(f'data/prepared/{split}.jsonl'):
                self.assertEqual(row['source'], 'original_style_pairs')
                self.assertEqual(row['messages'][-1]['content'], originals[row['id']]['response'])
                self.assertEqual(row['messages'][1]['content'], originals[row['id']]['prompt'])
        self.assertEqual(sum(len(x) for x in data.values()), len(originals))

    def test_corrupt_data_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in Path('data/prepared').glob('*.json*'):
                (root / path.name).write_bytes(path.read_bytes())
            with (root / 'train.jsonl').open('a') as handle:
                handle.write('\n')
            with self.assertRaisesRegex(ValueError, 'hash'):
                validate_data(config(), root)


class ModelTests(unittest.TestCase):
    def test_hybrid_lora_forward_backward_and_reload(self):
        import torch
        from peft import PeftModel
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Trainer, TrainingArguments
        cfg = Qwen3_5Config(
            text_config=dict(vocab_size=128, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                             head_dim=16, layer_types=['linear_attention', 'full_attention'],
                             linear_key_head_dim=16, linear_value_head_dim=16,
                             linear_num_key_heads=1, linear_num_value_heads=2,
                             rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0,
                                              'partial_rotary_factor': 1.0, 'mrope_section': [2, 3, 3]},
                             pad_token_id=0, eos_token_id=2, use_cache=False),
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                               out_hidden_size=32, num_position_embeddings=16),
        )
        torch.manual_seed(42)
        base = Qwen3_5ForConditionalGeneration(cfg)
        model = add_lora(base, {'lora_rank': 4, 'lora_alpha': 8})
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(any('linear_attn' in n for n in trainable))
        self.assertTrue(any('self_attn' in n for n in trainable))
        self.assertTrue(all('lora_' in n and 'language_model' in n for n in trainable))
        row = {'input_ids': list(range(10, 22)), 'attention_mask': [1] * 12,
               'labels': [-100] * 5 + list(range(15, 22))}
        batch = Collator(0)([row])
        model.train()
        loss = model(**batch).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad))
        torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3).step()
        model.eval()
        with torch.no_grad():
            expected = model(**batch).logits
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            torch.manual_seed(42)
            restored = PeftModel.from_pretrained(Qwen3_5ForConditionalGeneration(cfg), directory)
            restored.eval()
            with torch.no_grad():
                actual = restored(**batch).logits
            torch.testing.assert_close(actual, expected)
        # Exercise the real Trainer integration, including evaluation and checkpoints.
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=model,
                args=TrainingArguments(output_dir=directory, use_cpu=True, max_steps=1,
                                       per_device_train_batch_size=1, per_device_eval_batch_size=1,
                                       gradient_accumulation_steps=2, report_to='none',
                                       eval_strategy='epoch', save_strategy='epoch',
                                       load_best_model_at_end=True, metric_for_best_model='eval_loss',
                                       greater_is_better=False, prediction_loss_only=True,
                                       remove_unused_columns=False, dataloader_pin_memory=False,
                                       gradient_checkpointing=True,
                                       gradient_checkpointing_kwargs={'use_reentrant': False}),
                train_dataset=[row, row], eval_dataset=[row], data_collator=Collator(0),
            )
            result = trainer.train()
            self.assertEqual(result.global_step, 1)
            self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))


if __name__ == '__main__':
    unittest.main()

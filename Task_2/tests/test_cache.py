# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Validate physical packing, residual groups and absolute-position cache editing."""

import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import Cache, DynamicLayer

from Task_2.cache.adapter import BudgetCache, clone_dynamic_cache, model_step
from Task_2.cache.packing import PackedStream, pack_integers, payload_bytes, unpack_integers
from Task_2.cache.selection import question_importance, safeguards, selection_plan


def random_cache(length=1025, batch=1, layers=2):
    """Use actual KV tensor geometry with 64-dimensional heads and odd token lengths."""
    result = []
    for _ in range(layers):
        layer = DynamicLayer()
        layer.update(torch.randn(batch, 1, length, 64).half(), torch.randn(batch, 1, length, 64).half())
        result.append(layer)
    return Cache(layers=result)


class CacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(6304)

    def test_all_integer_values_roundtrip_real_bytes(self):
        for bits in [2, 4]:
            integers = (torch.arange(128) % (2 ** bits)).reshape(2, 64).to(torch.uint8)
            payload = pack_integers(integers, bits)
            self.assertEqual(payload.numel(), 128 * bits // 8)
            self.assertTrue(torch.equal(unpack_integers(payload, bits), integers))

    def test_every_policy_meets_actual_budget_and_binary_roundtrip(self):
        for length in [1024, 1025, 1087, 1088]:
            original = random_cache(length)
            geometry = (2, 1, 1, 64, length)
            budget = 32 + payload_bytes(*geometry, 4)
            for method in ["fp16", "sink_recent", "question_aware", "int4_token", "int4_channel", "hybrid"]:
                plan = selection_plan(method, geometry, budget, torch.linspace(0, 1, length))
                packed = BudgetCache(original, plan)
                payload = packed.serialize()
                self.assertEqual(len(payload), plan["expected_bytes"])
                if method != "fp16":
                    self.assertLessEqual(len(payload), budget)
                if plan["positions"]:
                    self.assertTrue(set(safeguards(length)).issubset(plan["positions"]))
                restored = BudgetCache.deserialize(payload)
                self.assertEqual(restored.serialize(), payload)
                for left, right in zip(packed.layers, restored.layers):
                    for first, second in zip(left.materialize(), right.materialize()):
                        self.assertTrue(torch.equal(first, second))

    def test_trailing_tokens_quantize_at_complete_group_and_stay_packed(self):
        keys, values = torch.randn(1, 1, 63, 64).half(), torch.randn(1, 1, 63, 64).half()
        stream = PackedStream(keys, values, 4, "channel")
        self.assertIsNone(stream.quantized_keys)
        stream.append(torch.randn(1, 1, 2, 64).half(), torch.randn(1, 1, 2, 64).half())
        self.assertEqual(stream.quantized_keys.tokens, 64)
        self.assertEqual(stream.residual_keys.shape[-2], 1)
        self.assertEqual(stream.length, 65)
        self.assertTrue(all(tensor.dtype in {torch.uint8, torch.float16} for tensor in stream.tensors()))

    def test_tiny_qwen_split_identity_eviction_and_question_scoring(self):
        configuration = Qwen2Config(vocab_size=256, hidden_size=128, intermediate_size=256,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=2048, attention_dropout=0., tie_word_embeddings=True,
            eos_token_id=255, pad_token_id=0)
        configuration._attn_implementation = "sdpa"
        model = Qwen2ForCausalLM(configuration).half().eval()
        tokens = torch.randint(2, 240, (1, 272))
        with torch.inference_mode():
            full_logits, _ = model_step(model, tokens)
            _, prefix = model_step(model, tokens[:, :256])
            split_logits, _ = model_step(model, tokens[:, 256:], clone_dynamic_cache(prefix), 256)
            torch.testing.assert_close(full_logits, split_logits, atol=.001, rtol=.002)
            geometry = (2, 1, 1, 64, 256)
            budget = 32 + payload_bytes(*geometry, 4)
            scores, scored_logits, _ = question_importance(model, prefix, tokens[:, 256:])
            self.assertEqual(scores.shape, (256,))
            self.assertTrue(torch.isfinite(scores).all())
            torch.testing.assert_close(scored_logits, split_logits, atol=.001, rtol=.002)
            identity = BudgetCache(prefix, selection_plan("fp16", geometry, budget))
            actual, _ = model_step(model, tokens[:, 256:], identity, 256)
            torch.testing.assert_close(actual, split_logits, atol=.001, rtol=.002)
            # This small geometry cannot fit 72 safeguards under INT4; test an
            # intentionally larger legal eviction budget without altering the policy.
            eviction_budget = 32 + (4 * 2 * 1 * 1 * 64 + 4) * 100
            plan = selection_plan("question_aware", geometry, eviction_budget, scores)
            edited = BudgetCache(prefix, plan)
            logits, _ = model_step(model, tokens[:, 256:], edited, 256)
            manual_layers = []
            indices = torch.tensor(plan["positions"])
            for original in prefix.layers:
                layer = DynamicLayer()
                layer.update(original.keys.index_select(-2, indices), original.values.index_select(-2, indices))
                manual_layers.append(layer)
            explicit = model.model(input_ids=tokens[:, 256:], past_key_values=Cache(layers=manual_layers),
                use_cache=True, position_ids=torch.arange(256, 272)[None], cache_position=torch.arange(100, 116), return_dict=True)
            manual_logits = model.lm_head(explicit.last_hidden_state[:, -1:]).float()
            torch.testing.assert_close(logits, manual_logits, atol=.001, rtol=.002)
            self.assertTrue(torch.isfinite(logits).all())
            self.assertEqual(edited.absolute_length, 272)
            self.assertEqual(edited.get_seq_length(), 100 + 16)
            self.assertEqual(edited.positions[-1].item(), 271)


if __name__ == "__main__":
    unittest.main()

# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Test joint laws, rejection, cache rollback, padding, EOS and final-token caps."""

import copy
import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from Task_3.models.runtime import clone_cache, step
from Task_3.sampling.exact import normalize_prompt, positive_residual, speculative_generate, target_generate
from Task_3.sampling.markov import validate_markov


def tiny_model(seed=6304, hidden=32, layers=1, device="cpu"):
    """Use random local Qwen weights; never download the assignment models in tests."""
    torch.manual_seed(seed)
    configuration = Qwen2Config(vocab_size=13, hidden_size=hidden, intermediate_size=hidden * 2,
        num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=2048, attention_dropout=0., eos_token_id=12, pad_token_id=0)
    configuration._attn_implementation = "sdpa"
    return Qwen2ForCausalLM(configuration).half().to(device).eval()


class SamplingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_100000_joint_markov_trials_and_rejected_negative_control(self):
        result = validate_markov()
        self.assertEqual(len(result["joint_categories"]), 108)
        self.assertTrue(all(row["passed"] for row in result["conditions"]))

    def test_residual_differs_from_target_and_handles_disjoint_support(self):
        p, q = torch.tensor([.6, .3, .1]), torch.tensor([.1, .3, .6])
        torch.testing.assert_close(positive_residual(p, q), torch.tensor([1., 0., 0.]))
        with self.assertRaises(FloatingPointError):
            positive_residual(p, p)

    def test_real_cache_rejection_bonus_short_proposals_and_pending_positions(self):
        teacher, draft = tiny_model(), tiny_model(6305)
        prompt = [1, 3, 2, 4]
        for gamma in [1, 2, 4]:
            for cap in [1, 3, 9]:
                def check(output, target_cache, draft_cache, pending):
                    expected = len(prompt) + len(output) - 1
                    self.assertEqual(target_cache.get_seq_length(), expected)
                    self.assertEqual(draft_cache.get_seq_length(), expected)
                    with torch.inference_mode():
                        cached, _ = step(teacher, torch.tensor([[pending]]), clone_cache(target_cache))
                        direct, _ = step(teacher, torch.tensor([prompt + output]))
                    torch.testing.assert_close(cached, direct, atol=.002, rtol=.01)
                result = speculative_generate(teacher, draft, prompt, cap, gamma, 6304, True, round_check=check)
                self.assertEqual(result["new_tokens"], cap)
                self.assertEqual(result["target_verification_passes"], len(result["rounds"]))
                self.assertTrue(all(row["examined"] <= row["proposed"] <= gamma for row in result["rounds"]))
        identical = speculative_generate(teacher, copy.deepcopy(teacher), prompt, 9, 4, 6304, True)
        self.assertTrue(identical["rounds"][0]["bonus"])
        self.assertEqual(identical["rounds"][0]["accepted"], 4)

    def test_eos_and_left_right_padding(self):
        model = tiny_model()
        class EOSHead(torch.nn.Module):
            def forward(self, hidden):
                result = torch.full(hidden.shape[:-1] + (13,), -1000., device=hidden.device)
                result[..., 12] = 1000.
                return result
        model.lm_head = EOSHead()
        for tokens, mask in [([0, 0, 1, 3], [0, 0, 1, 1]), ([1, 3, 0], [1, 1, 0])]:
            out = speculative_generate(model, copy.deepcopy(model), tokens, 9, 4, 6304, attention_mask=mask)
            self.assertEqual(out["token_ids"], [12])
            self.assertTrue(out["eos"])
            self.assertFalse(out["truncated"])
        self.assertEqual(target_generate(model, [1, 3], 9, 6304)["token_ids"], [12])
        with self.assertRaises(ValueError):
            normalize_prompt([1, 0, 3], "cpu", [1, 0, 1])

    @unittest.skipUnless(torch.cuda.is_available(), "Tiny CUDA sampler test requires CUDA.")
    def test_tiny_cuda_round_decomposition(self):
        teacher, draft = tiny_model(device="cuda"), tiny_model(6305, device="cuda")
        result = speculative_generate(teacher, draft, [1, 3, 4], 8, 4, 6304, True)
        self.assertEqual(result["new_tokens"], 8)
        self.assertGreater(result["peak_allocated_bytes"], 0)
        self.assertTrue(all(result[key] > 0 for key in ["proposal_seconds", "verification_seconds", "sampling_seconds", "maintenance_seconds"]))


if __name__ == "__main__":
    unittest.main()

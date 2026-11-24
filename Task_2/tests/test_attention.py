# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Check mathematical attention behavior without downloading pretrained models."""

import unittest
import copy
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as functional

from Task_2.attention.operators import (attention_operator, causal_mask, dense_attention,
                                       linear_attention, tiled_attention)


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(6304)

    def test_tiled_outputs_and_gradients_on_nondivisible_lengths(self):
        inputs = [torch.randn(2, 2, 23, 8, requires_grad=True) for _ in range(3)]
        upstream = torch.randn_like(inputs[0])
        expected = dense_attention(*inputs)
        gradients = torch.autograd.grad(expected, inputs, upstream)
        for tile in [1, 7, 32]:
            actual = tiled_attention(*inputs, tile)
            derivatives = torch.autograd.grad(actual, inputs, upstream)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            for first, second in zip(derivatives, gradients):
                torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-5)

    def test_future_perturbation_for_every_mqar_mechanism(self):
        q, k, v = [torch.randn(1, 4, 41, 8) for _ in range(3)]
        changed_k, changed_v = k.clone(), v.clone()
        changed_k[..., 20:, :] += 100
        changed_v[..., 20:, :] -= 100
        for method in ["dense", "window", "dilated", "linear", "local_content"]:
            expected = attention_operator(q, k, v, method, content_k=7)
            actual = attention_operator(q, changed_k, changed_v, method, content_k=7)
            torch.testing.assert_close(expected[..., :20, :], actual[..., :20, :], atol=2e-5, rtol=2e-5)

    def test_linear_chunking_matches_explicit_causal_kernel(self):
        q, k, v = [torch.randn(1, 4, 27, 8) for _ in range(3)]
        scores = (functional.elu(q) + 1) @ (functional.elu(k) + 1).transpose(-1, -2)
        scores = scores.masked_fill(~causal_mask(27, q.device), 0)
        expected = (scores @ v) / scores.sum(-1, keepdim=True)
        for chunk in [1, 9, 64]:
            torch.testing.assert_close(linear_attention(q, k, v, chunk), expected, atol=2e-6, rtol=2e-5)

    def test_mask_receptive_field_and_empty_rows(self):
        local = causal_mask(256, "cpu", window=64)
        dilated = causal_mask(256, "cpu", dilation=4)
        self.assertEqual(int(local[-1].sum()), 64)
        self.assertEqual(int(dilated[-1].sum()), 64)
        self.assertTrue(dilated[255, 3])
        self.assertFalse(dilated[255, 4])
        q, k, v = [torch.randn(1, 1, 8, 8, requires_grad=True) for _ in range(3)]
        output = dense_attention(q, k, v, torch.zeros(8, 8, dtype=torch.bool))
        self.assertEqual(int(output.count_nonzero()), 0)
        for gradient in torch.autograd.grad(output.sum(), [q, k, v]):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertEqual(int(gradient.count_nonzero()), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "Tiny CUDA timing smoke test requires CUDA.")
    def test_tiny_cuda_end_to_end_forward_backward_and_backend_measurements(self):
        from Task_2.attention.benchmarks import run_part_a
        from Task_2.common.artifacts import ArtifactStore, atomic_json, read_json
        from Task_2.common.config import load_configuration
        task = Path(__file__).resolve().parents[1]
        configuration = copy.deepcopy(load_configuration(task))
        configuration["attention"].update(lengths=[32], head_dimension=16, warmup=1,
                                            repetitions=2, reference_max_length=32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_json(root / "manifests/run_lock.json", {"config_id": "SYNTHETIC_TINY_CUDA_TEST_ONLY"})
            store = ArtifactStore({"task": task, "results": root}, configuration)
            for method in ["dense", "tiled", "sdpa"]:
                result = run_part_a(store, method)
                self.assertEqual(len(result), 2)
                self.assertTrue(all(row["status"] == "complete" for row in result))
                self.assertTrue(all(row["numerical_check"] == "passed" for row in result))
                self.assertTrue(all(len(row["forward_backward"]["samples_ms"]) == 2 for row in result))
            self.assertEqual(len(read_json(root / "metrics/attention/all_conditions.json")), 6)


if __name__ == "__main__":
    unittest.main()

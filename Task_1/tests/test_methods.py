# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Verify exact pruning budgets, low-bit payloads, and constrained allocations."""

import copy
import unittest
from pathlib import Path

import numpy as np
import torch

from Task_1.common.config import load_configuration, validate_configuration
from Task_1.methods.allocation import allocate_precision, allocate_sparsity
from Task_1.methods.pruning import row_prune, structured_prune
from Task_1.methods.quantization import (pack_integers, packed_projection_bytes,
                                         quantize_groups, reconstruct_packed,
                                         rescaling_equivalence, unpack_integers)


class CompressionMethodTests(unittest.TestCase):
    """Use tiny deterministic matrices, including padding and constant groups."""

    @classmethod
    def setUpClass(cls):
        cls.configuration = load_configuration(Path(__file__).resolve().parents[1])
        torch.set_num_threads(1)

    def test_exact_row_pruning_with_ties(self):
        weight = torch.ones(5, 11)
        compressed, mask = row_prune(weight, 17)
        self.assertEqual(int(mask.sum()), 17)
        self.assertEqual(int((compressed == 0).sum()), 17)
        self.assertEqual(mask.sum(dim=1).tolist(), [4, 4, 3, 3, 3])
        self.assertTrue(mask[0, :4].all())

    def test_wanda_changes_weight_ranking(self):
        weight = torch.tensor([[1., 2., 3., 4.]])
        _, magnitude = row_prune(weight, 2)
        _, wanda = row_prune(weight, 2, torch.tensor([100., 1., 1., 1.]))
        self.assertFalse(torch.equal(magnitude, wanda))
        self.assertFalse(bool(wanda[0, 0]))

    def test_existing_zeros_and_invalid_budget(self):
        weight = torch.tensor([[0., 3., 0., 1.]])
        compressed, _ = row_prune(weight, 2, torch.zeros(4))
        self.assertEqual(int((compressed == 0).sum()), 2)
        with self.assertRaises(ValueError):
            row_prune(weight, 1)

    def test_structured_mask(self):
        weight = torch.arange(1, 25).float().reshape(3, 8)
        compressed, mask = structured_prune(weight, torch.ones(8))
        self.assertTrue(torch.all(mask.reshape(3, 2, 4).sum(dim=-1) == 2))
        self.assertEqual(int((compressed == 0).sum()), 12)

    def test_all_bitstream_boundaries(self):
        for bits in [2, 3, 4, 8]:
            for columns in [1, 3, 7, 8, 9, 17, 128, 131]:
                values = (torch.arange(3 * columns).reshape(3, columns) % 2**bits).to(torch.uint8)
                packed = pack_integers(values, bits)
                recovered = unpack_integers(packed, bits, columns)
                self.assertTrue(torch.equal(values, recovered), (bits, columns))

    def test_quantizer_real_packing_and_byte_estimate(self):
        generator = torch.Generator().manual_seed(6304)
        weight = torch.randn(5, 131, generator=generator).half()
        for bits in [2, 3, 4, 8]:
            reconstructed, payload = quantize_groups(weight, bits)
            packed = pack_integers(payload["integers"], bits)
            decoded = reconstruct_packed({**payload, "packed": packed})
            self.assertTrue(torch.equal(reconstructed, decoded))
            sizes = packed_projection_bytes(5, 131, bits, self.configuration, compensation=True)
            measured = packed.numel() + payload["scales"].numel() * 2 + payload["zero_points"].numel() + 131 * 2 + 64
            measured += (-measured) % 16
            self.assertEqual(measured, sizes["packed_bytes"])

    def test_constant_and_zero_groups(self):
        weights = torch.tensor([[0.] * 128, [2.] * 128, [-3.] * 128]).half()
        for bits in [2, 3, 4, 8]:
            reconstructed, payload = quantize_groups(weights, bits)
            self.assertTrue(torch.isfinite(reconstructed).all())
            self.assertTrue(torch.allclose(weights, reconstructed, atol=.01))
            self.assertEqual(payload["scales"].dtype, torch.float16)
            self.assertEqual(payload["zero_points"].dtype, torch.uint8)

    def test_full_precision_channel_equivalence(self):
        torch.manual_seed(6304)
        result = rescaling_equivalence(torch.randn(7, 11), torch.randn(13, 11), torch.linspace(.1, 3, 11))
        self.assertLess(result["relative_l2"], 2e-5)

    def test_nonuniform_integer_sparsity_budget(self):
        profiles = {str(index): {"elements": 100 + 17 * index, "pruning_sensitivity": 10. ** index}
                    for index in range(6)}
        for target in [.5, .75]:
            plan = allocate_sparsity(profiles, target, 1)
            self.assertEqual(sum(row["zero_count"] for row in plan["allocations"].values()), round(target * sum(p["elements"] for p in profiles.values())))
            self.assertTrue(all(.1 <= row["sparsity"] <= .85 for row in plan["allocations"].values()))

    def test_precision_matches_bytes_not_average_bits(self):
        profiles = {}
        for index, name in enumerate(["model.layers.0.mlp.up_proj", "model.layers.0.mlp.down_proj", "model.layers.0.self_attn.q_proj"]):
            profiles[name] = {"name": name, "shape": [32, 128], "elements": 4096,
                              "awq": {str(bits): {"relative_mse": (index + 1) / (2. ** bits)} for bits in [2, 3, 4, 8]}}
        for reference in [3, 4]:
            plan = allocate_precision(profiles, reference, 12345, self.configuration)
            self.assertLessEqual(plan["packed_bytes"], plan["reference_budget_bytes"])
            self.assertTrue(set(plan["allocations"].values()).issubset({2, 3, 4, 8}))

    def test_fixed_protocol_cannot_silently_shrink(self):
        altered = copy.deepcopy(self.configuration)
        altered["evaluation"]["hellaswag_full_count"] = 16
        with self.assertRaises(ValueError):
            validate_configuration(altered)


if __name__ == "__main__":
    unittest.main()

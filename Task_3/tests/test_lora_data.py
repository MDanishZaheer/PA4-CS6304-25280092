# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Test unchanged assets, seeded FP32 adapters and genuine wrapper-free merging."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from Task_3.common.config import load_configuration, trained_variants
from Task_3.common.artifacts import ArtifactStore, sha256_file
from Task_3.data.prepare import verify_supplied_inputs
from Task_3.models.lora import LoRALinear, adapter_digest, adapter_state, attach_lora, merge_into_fresh
from Task_3.models.registry import ModelRegistry
from Task_3.tests.test_sampling import tiny_model
from Task_3.training.distill import atomic_torch_save


class LoRADataTests(unittest.TestCase):
    def test_fixed_data_and_four_run_protocol(self):
        task = Path(__file__).resolve().parents[1]
        self.assertEqual(len(verify_supplied_inputs(task / "resources/course_data")), 4)
        self.assertEqual(len(trained_variants(load_configuration(task))), 4)

    def test_fp32_parameters_identical_seeded_initialization_and_merge(self):
        torch.set_num_threads(1)
        settings = load_configuration(Path(__file__).resolve().parents[1])["training"]
        base = tiny_model()
        first = attach_lora(copy.deepcopy(base), settings, 6304)
        second = attach_lora(copy.deepcopy(base), settings, 6304)
        self.assertEqual(adapter_digest(first), adapter_digest(second))
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in first.parameters() if parameter.requires_grad))
        with torch.no_grad():
            for name, parameter in first.named_parameters():
                if name.endswith(".B"):
                    parameter.normal_(std=.01)
        merged = merge_into_fresh(copy.deepcopy(base), adapter_state(first), settings)
        self.assertFalse(any(isinstance(module, LoRALinear) for module in merged.modules()))
        ids = torch.tensor([[1, 3, 4, 5, 6]])
        with torch.inference_mode():
            left, right = first(ids).logits.float(), merged(ids).logits.float()
        self.assertLess(float((left - right).norm() / left.norm()), .01)

    def test_registry_saved_merge_reload_and_condition_integrity(self):
        torch.set_num_threads(1)
        task = Path(__file__).resolve().parents[1]
        configuration = load_configuration(task)
        base = tiny_model()
        wrapped = attach_lora(copy.deepcopy(base), configuration['training'], 6304)
        with torch.no_grad():
            for name, parameter in wrapped.named_parameters():
                if name.endswith('.B'):
                    parameter.normal_(std=.01)
        name = 'off_policy_seed6304'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = ArtifactStore({'task': task, 'results': root / 'results', 'data': root / 'data'}, configuration)
            store.save_json('manifests/run_lock.json', {'config_id': 'SYNTHETIC_REGISTRY_TEST_ONLY'})
            relative = f'checkpoints/{name}/adapter_final.pt'
            atomic_torch_save(store.path(relative), {'config_id': 'SYNTHETIC_REGISTRY_TEST_ONLY', 'adapter': adapter_state(wrapped)})
            store.save_json(f'metrics/training/{name}.json', {'adapter_checkpoint': relative, 'checkpoint_sha256': sha256_file(store.path(relative))})
            with patch('Task_3.models.registry.load_model', side_effect=lambda *a: copy.deepcopy(base)):
                registry = ModelRegistry(store)
                prompts = [{'id': f'SYNTHETIC_{i}', 'prompt_token_ids': [1, 2, 3, 4 + i]} for i in range(4)]
                result = registry.merge(name, prompts)
                self.assertEqual(len(result['numerical_checks']), 4)
                loaded = registry.merged(name)
                self.assertFalse(any(isinstance(module, LoRALinear) for module in loaded.modules()))
                with torch.inference_mode():
                    ids = torch.tensor([[1, 2, 3, 4]])
                    torch.testing.assert_close(loaded(ids).logits, wrapped(ids).logits, atol=.002, rtol=.02)
                checkpoint = torch.load(store.path(result['merged_checkpoint']), weights_only=True)
                checkpoint['adapter'] = 'on_policy_seed6304'
                atomic_torch_save(store.path(result['merged_checkpoint']), checkpoint)
                result['merged_checkpoint_sha256'] = sha256_file(store.path(result['merged_checkpoint']))
                store.save_json(f'metrics/merging/{name}.json', result)
                registry.clear_drafts()
                with self.assertRaises(ValueError):
                    registry.merged(name)


if __name__ == "__main__":
    unittest.main()

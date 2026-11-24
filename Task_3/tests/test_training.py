# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Exercise true soft-KL training, mixed quotas and exact interruption recovery."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from Task_3.common.artifacts import ArtifactStore, SessionTimeLimit, atomic_json, read_jsonl
from Task_3.common.config import load_configuration
from Task_3.data.prepare import read_gzip_records
from Task_3.models.lora import attach_lora
from Task_3.tests.test_sampling import tiny_model
from Task_3.training.distill import forward_kl_sum, prepare_teacher_pool, train_variant
from Task_3.training.schedule import PromptSchedule


class InterruptedStore(ArtifactStore):
    def __init__(self, paths, configuration):
        super().__init__(paths, configuration)
        self.calls = 0

    def check_time(self):
        self.calls += 1
        if self.calls == 2:
            raise SessionTimeLimit("SYNTHETIC stop after one successful update.")


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.task = Path(__file__).resolve().parents[1]
        self.configuration = copy.deepcopy(load_configuration(self.task))
        self.configuration["training"].update(quota={"math": 20, "short": 8}, tokens_per_update=8,
            response_caps={"math": 5, "short": 3})
        self.configuration["execution"]["recovery_every_updates"] = 5
        self.prompts = [{"id": f"SYNTHETIC_{domain}_{i}", "domain": domain,
            "prompt_token_ids": [1, 2, 3 + i], "prompt_sha256": f"SYNTHETIC_{domain}_{i}"}
            for domain, count in [("math", 5), ("short", 3)] for i in range(count)]

    def test_full_vocabulary_soft_kl_is_not_hard_token_cross_entropy(self):
        p = torch.tensor([[.5, .3, .2]])
        q = torch.tensor([[.2, .4, .4]], requires_grad=True)
        value = forward_kl_sum(q.log(), p.log())
        expected = (p * (p.log() - q.log())).sum()
        torch.testing.assert_close(value, expected)
        value.backward()
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_schedule_repeats_permutations_and_matches_between_methods(self):
        first = PromptSchedule(self.prompts, 6304, ["math", "math", "math", "short"])
        second = PromptSchedule(self.prompts, 6304, ["math", "math", "math", "short"])
        a = [first.next({"math": 1, "short": 1})[0]["id"] for _ in range(24)]
        b = [second.next({"math": 1, "short": 1})[0]["id"] for _ in range(24)]
        self.assertEqual(a, b)
        self.assertEqual(first.next({"math": 0, "short": 1})[0]["domain"], "short")

    def _run(self, method, device="cpu", interrupt=False):
        teacher = tiny_model(6304, device=device)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        variant = {"id": f"{method}_seed6304", "method": method, "seed": 6304}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {"task": self.task, "results": root}
            atomic_json(root / "manifests/run_lock.json", {"config_id": "SYNTHETIC_TRAINING_TEST_ONLY"})
            store = ArtifactStore(paths, self.configuration)
            if method == "off_policy":
                prepare_teacher_pool(store, teacher, self.prompts, 6304)
            def student():
                return attach_lora(tiny_model(6305, device=device), self.configuration["training"], 6304)
            if interrupt:
                with self.assertRaises(SessionTimeLimit):
                    train_variant(InterruptedStore(paths, self.configuration), teacher, student(), self.prompts, variant)
            result = train_variant(store, teacher, student(), self.prompts, variant)
            self.assertEqual(result["teacher_scored_tokens"], {"math": 20, "short": 8})
            self.assertEqual(result["successful_optimizer_updates"], 4)
            records = [row for relative in result["annotations"] for row in read_gzip_records(root / relative)]
            self.assertEqual(sum(row["teacher_scored_tokens"] for row in records), 28)
            self.assertTrue(all(row["source_id"].startswith("SYNTHETIC_") for row in records))
            history = read_jsonl(root / result["history"])
            self.assertEqual([row["teacher_scored_tokens"] for row in history], [8, 8, 8, 4])
            self.assertTrue(all(row["forward_kl"] >= -1e-5 for row in history))
            return result["adapter_sha256"]

    def test_both_methods_complete_exact_mixed_quotas(self):
        for method in ["off_policy", "on_policy"]:
            self._run(method)

    def test_resume_matches_uninterrupted_adapter_and_source_progress(self):
        self.assertEqual(self._run("on_policy"), self._run("on_policy", interrupt=True))

    @unittest.skipUnless(torch.cuda.is_available(), "Tiny CUDA LoRA training requires CUDA.")
    def test_tiny_cuda_fp32_adapters_scaler_and_full_soft_targets(self):
        self._run("on_policy", "cuda")


if __name__ == "__main__":
    unittest.main()

# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Verify batched causal scores, padding, checkpoint boundaries and GPU policy reuse."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from Task_1.cloud.colab import DriveSync
from Task_1.common.artifacts import ArtifactStore, SessionTimeLimit, read_jsonl
from Task_1.common.config import load_configuration, validate_configuration
from Task_1.evaluation.resources import benchmark_condition
from Task_1.evaluation.scoring import (batch_sequence_nll, evaluate_hellaswag,
                                       evaluate_perplexity, score_sequence_jobs, sequence_nll)
from Task_1.evaluation.workload import choose_workload, configure_gpu_workload, probe_workload
from Task_1.methods.calibration import collect_activations
from Task_1.models.compression import CompressionState


class CharacterTokenizer:
    """Use deterministic character tokens for tiny tests without network access."""

    def encode(self, text, add_special_tokens=False):
        return [ord(character) % 63 + 1 for character in text]


class BatchingTests(unittest.TestCase):
    """Keep score identities and saved progress independent of batching decisions."""

    @classmethod
    def setUpClass(cls):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        torch.set_num_threads(1)
        torch.manual_seed(6304)
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=512, tie_word_embeddings=True, attention_dropout=0.)
        config._attn_implementation = "sdpa"
        cls.reference_model = Qwen2ForCausalLM(config).eval().requires_grad_(False)
        cls.reference_configuration = load_configuration(Path(__file__).resolve().parents[1])

    def setUp(self):
        self.model = copy.deepcopy(self.reference_model)
        self.configuration = copy.deepcopy(self.reference_configuration)

    def test_batched_scores_exclude_padding_and_context_at_every_chunk_boundary(self):
        jobs = [(list(range(1, 13)), 1, 12), (list(range(1, 20)), 8, 17), (list(range(1, 8)), 4, 7)]
        reference = [sequence_nll(self.model, tokens, begin, end, 3) for tokens, begin, end in jobs]
        for chunk in [1, 5, 64]:
            actual = batch_sequence_nll(self.model, jobs, chunk)
            self.assertEqual([row[1] for row in actual], [11, 9, 3])
            torch.testing.assert_close(torch.tensor(actual), torch.tensor(reference), rtol=1e-6, atol=1e-5)
        with self.assertRaises(ValueError):
            batch_sequence_nll(self.model, [(list(range(8)), 0, 8)])
        with self.assertRaises(ValueError):
            batch_sequence_nll(self.model, [([1] * 513, 1, 513)])

    def test_token_budget_splits_batches_without_reordering_or_truncation(self):
        self.configuration["execution"]["gpu_workload"] = {
            "batch_sequences": 4, "padded_token_budget": 24, "head_chunk_tokens": 5}
        jobs = [(list(range(1, length + 1)), 1, length) for length in [8, 12, 7, 17, 6]]
        with patch.object(self.model.model, "forward", wraps=self.model.model.forward) as forward:
            actual = score_sequence_jobs(self.model, jobs, self.configuration)
        for call in forward.call_args_list:
            shape = call.kwargs["input_ids"].shape
            self.assertLessEqual(shape[0] * shape[1], 24)
        expected = [sequence_nll(self.model, tokens, begin, end) for tokens, begin, end in jobs]
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=1e-6, atol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "Padded FP16 scoring diagnostic requires CUDA.")
    def test_cuda_padded_batch_scores_match_independent_fp16_and_quantized_models(self):
        model = self.model.half().to("cuda")
        jobs = [(list(range(1, 17)), 1, 16), (list(range(1, 25)), 10, 24), (list(range(1, 10)), 5, 9)]
        state = CompressionState(model, self.configuration)
        for condition in [{"family": "fp16"}, {"family": "Q1", "bits": 4}]:
            state.apply(condition, {}, {})
            actual = batch_sequence_nll(model, jobs, 11)
            reference = [sequence_nll(model, ids, begin, end, 7) for ids, begin, end in jobs]
            self.assertEqual([count for _, count in actual], [count for _, count in reference])
            self.assertLess(max(abs(left[0] - right[0]) / left[1] for left, right in zip(actual, reference)), .003)

    @unittest.skipUnless(torch.cuda.is_available(), "Configured batch latency diagnostic requires CUDA.")
    def test_latency_benchmark_uses_the_configured_common_batch_size(self):
        self.configuration["benchmark"]["batch_size"] = 3
        validate_configuration(self.configuration)
        self.configuration["benchmark"]["prompt_lengths"] = [8]
        rows = [{"token_ids": [1] * 512} for _ in range(4)]
        model = self.model.half().to("cuda")
        with tempfile.TemporaryDirectory() as temporary:
            store = ArtifactStore({"results": Path(temporary)}, self.configuration)
            with patch.object(model.model, "forward", wraps=model.model.forward) as forward:
                result = benchmark_condition(model, rows, self.configuration, store, "fixture")
            self.assertEqual(result["batch_size"], 3)
            self.assertEqual(result["unit"], "milliseconds per batch")
            self.assertTrue(all(call.kwargs["input_ids"].shape[0] == 3 for call in forward.call_args_list))

    def test_batched_perplexity_resume_stops_at_the_same_durable_example_prefix(self):
        windows = [{"token_ids": [((index + offset) % 63) + 1 for offset in range(12)],
                    "source_row": index, "token_offset": 0, "score_target_offsets": [2, 11]} for index in range(37)]
        self.configuration["evaluation"]["scored_tokens_per_suite"] = 37 * 9
        self.configuration["execution"]["gpu_workload"] = {
            "batch_sequences": 6, "padded_token_budget": 6 * 512, "head_chunk_tokens": 13}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local, remote, fresh = root / "local", root / "Drive", root / "fresh"
            store = ArtifactStore({"results": local}, self.configuration, DriveSync(local, remote))
            with patch.object(store, "check_time", side_effect=[None] * 4 + [SessionTimeLimit("Interrupted")]):
                with self.assertRaises(SessionTimeLimit):
                    evaluate_perplexity(self.model, windows, "test", "fp16", store)
            self.assertEqual(len(read_jsonl(remote / "predictions/fp16/test.jsonl")), 16)
            sync = DriveSync(fresh, remote)
            sync.restore()
            store = ArtifactStore({"results": fresh}, self.configuration, sync)
            with patch.object(self.model.model, "forward", wraps=self.model.model.forward) as forward:
                summary = evaluate_perplexity(self.model, windows, "test", "fp16", store)
            self.assertEqual(forward.call_count, 4)
            self.assertEqual(summary["scored_tokens"], 333)
            records = read_jsonl(fresh / "predictions/fp16/test.jsonl")
            self.assertEqual([row["example_id"] for row in records], [f"test:{index}:0" for index in range(37)])
            expected = [sequence_nll(self.model, row["token_ids"], 2, 11)[0] for row in windows]
            torch.testing.assert_close(torch.tensor([row["nll_sum"] for row in records]), torch.tensor(expected), rtol=1e-6, atol=1e-5)

    def test_batched_hellaswag_preserves_all_candidate_scores_and_decisions(self):
        rows = [{"order": index, "source_row_index": index, "example": {
            "ctx_a": "A person", "ctx_b": "looks up", "activity_label": "Test",
            "endings": ["walks", "stops here", "runs away quickly", "turns"], "label": str(index % 4)}} for index in range(19)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequential = ArtifactStore({"results": root / "sequential"}, self.configuration)
            evaluate_hellaswag(self.model, CharacterTokenizer(), rows, "fp16", sequential)
            self.configuration["execution"]["gpu_workload"] = {
                "batch_sequences": 12, "padded_token_budget": 12 * 512, "head_chunk_tokens": 19}
            batched = ArtifactStore({"results": root / "batched"}, self.configuration)
            evaluate_hellaswag(self.model, CharacterTokenizer(), rows, "fp16", batched)
            expected = read_jsonl(sequential.path("predictions/fp16/hellaswag.jsonl"))
            actual = read_jsonl(batched.path("predictions/fp16/hellaswag.jsonl"))
            for left, right in zip(actual, expected):
                for key in ["example_id", "gold", "prediction", "prediction_normalized", "character_lengths", "continuation_token_counts"]:
                    self.assertEqual(left[key], right[key])
                torch.testing.assert_close(torch.tensor(left["log_likelihoods"]), torch.tensor(right["log_likelihoods"]), rtol=1e-6, atol=1e-5)

    def test_workload_selection_rejects_ooms_and_over_budget_fast_probes(self):
        probes = [{"status": "ok", "peak_allocated_bytes": memory, "scored_tokens_per_second": speed,
                   "batch_sequences": batch, "head_chunk_tokens": 256}
                  for memory, speed, batch in [(40, 100, 1), (80, 160, 4), (95, 300, 8)]]
        probes.append({"status": "tuning_cuda_oom", "batch_sequences": 16})
        self.assertEqual(choose_workload(probes, 90)["batch_sequences"], 4)
        with self.assertRaises(RuntimeError):
            choose_workload(probes, 20)

    @unittest.skipUnless(torch.cuda.is_available(), "GPU workload diagnostic requires CUDA.")
    def test_gpu_workload_is_frozen_and_reused_without_retuning(self):
        model = self.model.half().to("cuda")
        rows = [{"token_ids": [offset % 64 for offset in range(512)]} for _ in range(128)]
        self.configuration["execution"]["gpu_batch_candidates"] = [1, 2]
        self.configuration["execution"]["gpu_head_chunk_candidates"] = [32, 64]
        with tempfile.TemporaryDirectory() as temporary:
            store = ArtifactStore({"results": Path(temporary)}, self.configuration)
            plan = configure_gpu_workload(model, rows, self.configuration, store)
            with patch("Task_1.evaluation.workload.probe_workload", wraps=probe_workload) as probe:
                restored = configure_gpu_workload(model, rows, self.configuration, store)
            self.assertEqual(plan, restored)
            self.assertEqual(probe.call_count, 1)
            self.assertLessEqual(plan["selected_probe_peak_bytes"], plan["memory_budget_bytes"])
            self.configuration["execution"]["gpu_memory_target_fraction"] = .8
            with self.assertRaisesRegex(ValueError, "settings differ"):
                configure_gpu_workload(model, rows, self.configuration, store)

    @unittest.skipUnless(torch.cuda.is_available(), "Batched calibration recovery requires CUDA.")
    def test_batched_calibration_resumes_without_changing_samples_or_counts(self):
        model = self.model.half().to("cuda")
        rows = [{"token_ids": [(index + offset) % 64 for offset in range(512)]} for index in range(128)]
        self.configuration["execution"]["gpu_workload"] = {"calibration_batch_sequences": 8}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_store = ArtifactStore({"results": root / "baseline"}, self.configuration)
            expected, _ = collect_activations(model, rows, self.configuration, baseline_store)
            local, remote, fresh = root / "local", root / "Drive", root / "fresh"
            store = ArtifactStore({"results": local}, self.configuration, DriveSync(local, remote))
            with patch.object(store, "check_time", side_effect=[None] * 3 + [SessionTimeLimit("Interrupted")]):
                with self.assertRaises(SessionTimeLimit):
                    collect_activations(model, rows, self.configuration, store)
            sync = DriveSync(fresh, remote)
            sync.restore()
            store = ArtifactStore({"results": fresh}, self.configuration, sync)
            with patch.object(model.model, "forward", wraps=model.model.forward) as forward:
                actual, metadata = collect_activations(model, rows, self.configuration, store)
            self.assertEqual(forward.call_count, 14)
            self.assertEqual(metadata["resumed_from_sequences"], 16)
            for name in expected:
                self.assertEqual(actual[name]["token_count"], 128 * 512)
                self.assertEqual(actual[name]["samples"].shape[0], 128 * 4)
                for key in ["input_norm", "mean_absolute", "samples"]:
                    torch.testing.assert_close(actual[name][key], expected[name][key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

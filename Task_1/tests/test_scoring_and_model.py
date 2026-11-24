# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Check shared likelihood scoring and folded compensation on a random tiny Qwen."""

import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from Task_1.cloud.colab import DriveSync
from Task_1.common.artifacts import ArtifactStore, SessionTimeLimit
from Task_1.common.config import load_configuration
from Task_1.evaluation.scoring import format_hellaswag, sequence_nll, validate_progress
from Task_1.evaluation.statistics import align_records, paired_bootstrap
from Task_1.models.compression import (CompressionState, eligible_projections,
                                      excluded_parameter_digest, parameter_inventory)
from Task_1.methods.calibration import collect_activations


class ScoringAndModelTests(unittest.TestCase):
    """Exercise the real architecture/API without fetching any checkpoint."""

    @classmethod
    def setUpClass(cls):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        torch.set_num_threads(1)
        torch.manual_seed(6304)
        configuration = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                                    num_hidden_layers=2, num_attention_heads=4,
                                    num_key_value_heads=2, max_position_embeddings=128,
                                    tie_word_embeddings=True, attention_dropout=0.)
        configuration._attn_implementation = "sdpa"
        cls.model = Qwen2ForCausalLM(configuration).eval().requires_grad_(False)
        cls.configuration = load_configuration(Path(__file__).resolve().parents[1])

    def setUp(self):
        self.model = copy.deepcopy(type(self).model)

    def test_shared_scorer_matches_shifted_causal_loss(self):
        tokens = list(range(1, 17))
        summed, count = sequence_nll(self.model, tokens, chunk_tokens=3)
        input_ids = torch.tensor([tokens])
        with torch.inference_mode():
            reference = float(self.model(input_ids=input_ids, labels=input_ids, use_cache=False).loss)
        self.assertEqual(count, 15)
        self.assertAlmostEqual(summed / count, reference, places=5)

    def test_scored_offsets_exclude_context_and_first_token(self):
        tokens = list(range(1, 12))
        summed, count = sequence_nll(self.model, tokens, begin=6, end=10, chunk_tokens=2)
        with torch.inference_mode():
            logits = self.model(input_ids=torch.tensor([tokens]), use_cache=False).logits
            reference = torch.nn.functional.cross_entropy(logits[0, 5:9].float(), torch.tensor(tokens[6:10]), reduction="sum")
        self.assertEqual(count, 4)
        self.assertAlmostEqual(summed, float(reference), places=5)
        with self.assertRaises(ValueError):
            sequence_nll(self.model, tokens, begin=0)

    def test_inventory_counts_tied_embedding_once(self):
        inventory = parameter_inventory(self.model, self.configuration)
        self.assertEqual(len(inventory["projections"]), 14)
        self.assertEqual(inventory["parameters"], sum(p.numel() for p in self.model.parameters()))
        self.assertTrue(any("lm_head.weight" in group and "model.embed_tokens.weight" in group for group in inventory["tied_parameter_groups"]))

    def test_full_graph_rescaling_and_restore(self):
        state = CompressionState(self.model, self.configuration)
        profiles = {}
        for name, entry in eligible_projections(self.model, self.configuration).items():
            size = entry["module"].weight.shape[1]
            profiles[name] = {"scales": {"4": torch.linspace(.6, 1.4, size).half()}}
        tokens = torch.tensor([list(range(1, 17))])
        with torch.inference_mode():
            reference = self.model(input_ids=tokens, use_cache=False).logits
            state.apply_full_precision_rescaling(profiles, 4)
            adjusted = self.model(input_ids=tokens, use_cache=False).logits
        self.assertLess(float((reference - adjusted).norm() / reference.norm()), .005)
        self.assertEqual(excluded_parameter_digest(self.model, self.configuration), state.excluded_digest)
        state.restore()
        with torch.inference_mode():
            restored = self.model(input_ids=tokens, use_cache=False).logits
        self.assertTrue(torch.equal(restored, reference))

    def test_forward_hellaswag_format_and_characters(self):
        example = {"ctx_a": "A person", "ctx_b": "opens a box", "activity_label": "Activity",
                   "endings": ["é", "two words", "[note] yes", " [title] OK"], "label": "1"}
        prompt, endings, gold = format_hellaswag(example)
        self.assertEqual(prompt, "Activity: A person Opens a box")
        self.assertEqual(len(endings[0]), 1)
        self.assertEqual(gold, 1)

    def test_q2_and_same_bit_q3_use_identical_quantizer(self):
        state = CompressionState(self.model, self.configuration)
        profiles, statistics = {}, {}
        for name, entry in eligible_projections(self.model, self.configuration).items():
            columns = entry["module"].weight.shape[1]
            profiles[name] = {"scales": {str(bits): torch.linspace(.7, 1.3, columns).half() for bits in [2, 3, 4, 8]}}
            statistics[name] = {"input_norm": torch.ones(columns)}
        tokens = torch.tensor([list(range(1, 9))])
        state.apply({"family": "Q2", "bits": 4}, statistics, profiles)
        with torch.inference_mode():
            q2 = self.model(input_ids=tokens, use_cache=False).logits
        state.apply({"family": "Q3"}, statistics, profiles, {"allocations": {name: 4 for name in profiles}})
        with torch.inference_mode():
            q3 = self.model(input_ids=tokens, use_cache=False).logits
        self.assertTrue(torch.equal(q2, q3))
        self.assertEqual(excluded_parameter_digest(self.model, self.configuration), state.excluded_digest)

    @unittest.skipUnless(torch.cuda.is_available(), "Small CUDA cache diagnostic requires a GPU.")
    def test_cuda_dense_cache_and_backend_provenance(self):
        from Task_1.evaluation.resources import forward_last, sdpa_operations
        model = self.model.half().to("cuda")
        tokens = torch.tensor([list(range(1, 17))], device="cuda")
        with torch.inference_mode():
            full, _ = forward_last(model, tokens)
            _, cache = forward_last(model, tokens[:, :-1])
            split, _ = forward_last(model, tokens[:, -1:], cache)
        self.assertLess(float((full - split).float().norm() / full.float().norm()), .01)
        operations = sdpa_operations(model, tokens)
        self.assertTrue(any("scaled_dot_product" in operation for operation in operations))

    @unittest.skipUnless(torch.cuda.is_available(), "Calibration recovery diagnostic requires a GPU.")
    def test_calibration_resumes_verified_partial_checkpoint_with_identical_statistics(self):
        model = self.model.half().to("cuda")
        rows = [{"token_ids": [(index + offset) % 64 for offset in range(512)]} for index in range(128)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_store = ArtifactStore({"results": root / "reference"}, self.configuration)
            expected, _ = collect_activations(model, rows, self.configuration, reference_store)
            local, remote = root / "interrupted", root / "Drive/results/run"
            interrupted_store = ArtifactStore({"results": local}, self.configuration, DriveSync(local, remote))
            # The first 16 sequences are durable; the next four must be repeated.
            interruptions = [None] * 20 + [SessionTimeLimit("Simulated runtime disconnect")]
            with patch.object(interrupted_store, "check_time", side_effect=interruptions):
                with self.assertRaises(SessionTimeLimit):
                    collect_activations(model, rows, self.configuration, interrupted_store)
            partial = torch.load(remote / "checkpoints/calibration_activations.pt", weights_only=True)
            self.assertEqual(partial["completed_sequences"], 16)
            self.assertEqual(partial["status"], "collecting")
            self.assertTrue(all(state["token_count"] == 16 * 512 for state in partial["statistics"].values()))
            self.assertTrue(all(not module._forward_pre_hooks for module in model.modules()))
            fresh = root / "fresh_runtime"
            sync = DriveSync(fresh, remote)
            self.assertGreater(sync.restore(), 0)
            resumed_store = ArtifactStore({"results": fresh}, self.configuration, sync)
            with patch.object(model.model, "forward", wraps=model.model.forward) as forward:
                actual, metadata = collect_activations(model, rows, self.configuration, resumed_store)
            self.assertEqual(forward.call_count, 112)
            self.assertEqual(metadata["resumed_from_sequences"], 16)
            for name in expected:
                for key in ["input_norm", "mean_absolute", "samples"]:
                    torch.testing.assert_close(actual[name][key], expected[name][key], rtol=0, atol=0)
                self.assertEqual(actual[name]["token_count"], 128 * 512)
            completed = torch.load(remote / "checkpoints/calibration_activations.pt", weights_only=True)
            self.assertEqual(completed["status"], "complete")
            # A disconnect after the final tensor save must not lose metadata.
            (fresh / "metrics/calibration.json").unlink()
            with patch.object(model.model, "forward", side_effect=AssertionError("Completed calibration was rerun")):
                collect_activations(model, rows, self.configuration, resumed_store)
            self.assertTrue((fresh / "metrics/calibration.json").is_file())

    def test_bootstrap_pools_token_nll(self):
        left = [{"example_id": "a", "scored_tokens": 1, "nll_sum": 10.},
                {"example_id": "b", "scored_tokens": 9, "nll_sum": 0.}]
        right = [{"example_id": "a", "scored_tokens": 1, "nll_sum": 0.},
                 {"example_id": "b", "scored_tokens": 9, "nll_sum": 0.}]
        result = paired_bootstrap(left, right, "perplexity", resamples=100)
        self.assertAlmostEqual(result["difference"], math.e - 1)
        self.assertEqual(result["resampling_unit"], "whole fixed evaluation windows, paired")

    def test_paired_ids_and_progress_cannot_mix(self):
        left = [{"example_id": "a", "prompt_sha256": "first"}]
        right = [{"example_id": "a", "prompt_sha256": "second"}]
        with self.assertRaises(ValueError):
            align_records(left, right)
        with self.assertRaises(ValueError):
            validate_progress([{"example_id": "b", "condition": "x", "suite": "c4"}], ["a", "b"], "x", "c4")


if __name__ == "__main__":
    unittest.main()

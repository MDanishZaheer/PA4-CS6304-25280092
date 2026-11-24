# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Check prompt boundaries, answer extraction and all packed cache scoring branches."""

import unittest
import copy
import tempfile
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from Task_2.cache.adapter import BudgetCache, model_step
from Task_2.cache.packing import payload_bytes
from Task_2.cache.selection import selection_plan
from Task_2.data.retrieval import extract_code, load_retrieval_cases, tokenize_case
from Task_2.evaluation.kv import answer_nll, greedy_response, logit_kl, teacher_forced_logits
from Task_2.evaluation.statistics import paired_accuracy_interval


class CharacterTokenizer:
    """A deterministic test tokenizer; never used in real retrieval experiments."""

    eos_token_id = 255

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "SYSTEM\n" + messages[0]["content"] + "\nUSER\n" + messages[1]["content"] + "\nASSISTANT\n"

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(character) % 250 for character in text],
                "offset_mapping": [(index, index + 1) for index in range(len(text))]}

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(str(value) for value in tokens if value != self.eos_token_id)

    def encode(self, text, add_special_tokens=False):
        return [10 + int(character) for character in text if character.isdigit()]


class RetrievalTests(unittest.TestCase):
    def test_joint_tokenization_preserves_staff_boundary_and_gold_offsets(self):
        task = Path(__file__).resolve().parents[1]
        cases = load_retrieval_cases(task / "resources/course_data")
        self.assertEqual(len(cases), 180)
        result = tokenize_case(cases[0], CharacterTokenizer(), "Fixed test system")
        self.assertEqual(result["code"], cases[0]["code"])
        self.assertTrue(result["evidence_token_positions"])
        self.assertLess(max(result["evidence_token_positions"]), result["prefix_length"])
        self.assertEqual(result["prefix_length"] + result["suffix_length"], len(result["token_ids"]))

    def test_extractor_is_fixed_and_does_not_read_a_gold_answer(self):
        self.assertEqual(extract_code("Code: 12345."), "12345")
        self.assertEqual(extract_code("12345 then 67890"), "12345")
        self.assertIsNone(extract_code("123456"))
        self.assertIsNone(extract_code("no code"))

    def test_all_packed_policies_support_isolated_scoring_and_generation(self):
        torch.set_num_threads(1)
        torch.manual_seed(6304)
        settings = Qwen2Config(vocab_size=256, hidden_size=128, intermediate_size=256,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=2048, attention_dropout=0., eos_token_id=255, pad_token_id=0)
        settings._attn_implementation = "sdpa"
        model = Qwen2ForCausalLM(settings).half().eval()
        ids = torch.randint(2, 240, (1, 1092))
        with torch.inference_mode():
            _, original = model_step(model, ids[:, :1087])
            shape = (2, 1, 1, 64, 1087)
            budget = 32 + payload_bytes(*shape, 4)
            for method in ["fp16", "sink_recent", "question_aware", "int4_token", "int4_channel", "hybrid"]:
                packed = BudgetCache(original, selection_plan(method, shape, budget, torch.linspace(0, 1, 1087)))
                first, _ = model_step(model, ids[:, 1087:], packed, 1087)
                binary = packed.serialize()
                forced = teacher_forced_logits(model, first, packed, [17, 18, 19])
                metrics = answer_nll(forced, [17, 18, 19])
                self.assertTrue(metrics["mean_nll"] > 0)
                self.assertAlmostEqual(logit_kl(forced, forced), 0.)
                response = greedy_response(model, CharacterTokenizer(), first, packed.clone(), 3)
                self.assertGreater(response["new_tokens"], 0)
                self.assertEqual(packed.serialize(), binary)
                self.assertEqual(BudgetCache.deserialize(binary).serialize(), binary)

    def test_paired_bootstrap_identity_and_prompt_mismatch(self):
        records = [{"id": str(index), "prompt_sha256": str(index), "correct": index % 2 == 0} for index in range(180)]
        result = paired_accuracy_interval(records, records)
        self.assertEqual(result["paired_prompts"], 180)
        self.assertEqual(result["difference"], 0.)
        self.assertEqual(result["ci_low"], 0.)
        changed = [{**record, "prompt_sha256": "changed"} for record in records]
        with self.assertRaises(ValueError):
            paired_accuracy_interval(records, changed)

    def test_group_boundary_quantization_never_reads_future_teacher_tokens(self):
        torch.set_num_threads(1)
        torch.manual_seed(6304)
        settings = Qwen2Config(vocab_size=256, hidden_size=128, intermediate_size=256,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=2048, attention_dropout=0., eos_token_id=255, pad_token_id=0)
        settings._attn_implementation = "sdpa"
        model = Qwen2ForCausalLM(settings).half().eval()
        prefix_ids = torch.randint(2, 240, (1, 1085))
        with torch.inference_mode():
            initial, original = model_step(model, prefix_ids)
            shape = (2, 1, 1, 64, 1085)
            budget = 32 + payload_bytes(*shape, 4)
            for method in ["int4_token", "int4_channel", "hybrid"]:
                packed = BudgetCache(original, selection_plan(method, shape, budget, torch.linspace(0, 1, 1085)))
                first = teacher_forced_logits(model, initial, packed, [10, 11, 12, 13, 14, 15])
                second = teacher_forced_logits(model, initial, packed, [10, 11, 90, 91, 92, 93])
                torch.testing.assert_close(first[:, :3], second[:, :3], atol=0, rtol=0)
                # Multi-token callers receive the same causal appends as an
                # explicit one-token loop, including after a residual completes.
                batch_cache, sequential_cache = packed.clone(), packed.clone()
                joined, _ = model_step(model, torch.tensor([[10, 11, 12, 13]]), batch_cache, all_logits=True)
                singles = [model_step(model, torch.tensor([[token]]), sequential_cache)[0] for token in [10, 11, 12, 13]]
                torch.testing.assert_close(joined, torch.cat(singles, dim=1), atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "Tiny CUDA cache pipeline smoke requires CUDA.")
    def test_tiny_cuda_full_retrieval_and_packed_systems_paths(self):
        from Task_2.common.artifacts import ArtifactStore, atomic_json, read_json
        from Task_2.common.config import load_configuration
        from Task_2.evaluation.kv import evaluate_retrieval
        from Task_2.evaluation.kv_systems import run_cache_systems
        torch.set_num_threads(1)
        torch.manual_seed(6304)
        task = Path(__file__).resolve().parents[1]
        configuration = copy.deepcopy(load_configuration(task))
        configuration["kv"].update(max_new_tokens=3, qualitative_case_indices=[0],
            benchmark_lengths=[1024], benchmark_batches=[1, 2], warmup=1, repetitions=2)
        model_config = Qwen2Config(vocab_size=256, hidden_size=128, intermediate_size=256,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=2048, attention_dropout=0., eos_token_id=255, pad_token_id=0)
        model_config._attn_implementation = "sdpa"
        model = Qwen2ForCausalLM(model_config).half().cuda().eval()
        tokens = torch.randint(2, 240, (1092,)).tolist()
        case = {"id": "SYNTHETIC_TINY_CUDA_CASE", "token_ids": tokens, "prefix_length": 1087,
                "suffix_length": 5, "actual_prompt_tokens": 1092, "target_tokens": 2048,
                "position": "middle", "distractors": 0, "code": "12345",
                "prompt_sha256": "SYNTHETIC_TEST_ONLY", "evidence_token_positions": [8, 9]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_json(root / "manifests/run_lock.json", {"config_id": "SYNTHETIC_TINY_CUDA_TEST_ONLY"})
            store = ArtifactStore({"task": task, "results": root}, configuration)
            result = evaluate_retrieval(store, model, CharacterTokenizer(), [case])
            self.assertEqual(result["records"], 6)
            self.assertEqual(len(read_json(root / "metrics/kv/paired_comparisons.json")), 9)
            for method in configuration["kv"]["methods"]:
                row = read_json(root / f"predictions/kv/{case['id']}/{method}.json")
                self.assertEqual(row["status"], "complete")
                self.assertEqual(row["response"]["new_tokens"], len(row["response"]["token_ids"]))
            systems = run_cache_systems(store, model)
            self.assertEqual(len(systems), 12)
            self.assertTrue(all(row["status"] == "complete" for row in systems))
            self.assertTrue(all(row["serialized_bytes"] == row["calculated_bytes"] for row in systems))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()

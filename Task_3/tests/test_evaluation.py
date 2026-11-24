# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Exercise actual tiny-model banks, KL diagnostics, benchmarks, quality and isolated costs."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from Task_3.common.artifacts import ArtifactStore, atomic_json, read_json
from Task_3.common.config import draft_ids, load_configuration
from Task_3.evaluation.banks import evaluate_alignment, generate_banks, probability_metrics
from Task_3.evaluation.benchmark import benchmark
from Task_3.evaluation.quality import evaluate_quality, numeric_answer, score_answer
from Task_3.evaluation.statistics import break_even_acceptance, deployment_break_even, paired_interval, predicted_speedup
from Task_3.evaluation.systems import isolated_costs, validate_real_caches
from Task_3.tests.test_sampling import tiny_model


class TestTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return ' '.join(str(token) for token in token_ids if token != 12)


class EvaluationTests(unittest.TestCase):
    def test_full_vocabulary_metrics_and_acceptance_identity(self):
        p, q = torch.tensor([[.7, .2, .1]]), torch.tensor([[.1, .2, .7]])
        values = probability_metrics(p.log(), q.log())
        self.assertAlmostEqual(float(values['tv']), .6, places=6)
        self.assertAlmostEqual(float(values['expected_acceptance']), .4, places=6)
        same = probability_metrics(p.log(), p.log())
        self.assertAlmostEqual(float(same['forward_kl']), 0.)
        self.assertEqual(float(same['greedy_agreement']), 1.)

    def test_extractors_distinguish_format_from_correctness(self):
        self.assertEqual(numeric_answer('Work 3 + 4. #### 7'), '7')
        self.assertEqual(numeric_answer('The result is \\boxed{1,024}.'), '1024')
        self.assertIsNone(numeric_answer('No numeric result'))
        prompt = {'domain': 'short', 'family': 1, 'answer': '42'}
        self.assertTrue(score_answer(prompt, 'The integer is 42')['correct'])
        self.assertFalse(score_answer(prompt, 'The integer is 42')['format_compliant'])
        self.assertTrue(score_answer(prompt, '17')['format_compliant'])
        self.assertFalse(score_answer(prompt, '17')['correct'])

    def test_cost_model_nonexistence_and_prompt_pairing(self):
        self.assertIsNone(break_even_acceptance(4, 2., 1.))
        threshold = break_even_acceptance(4, .2, 1.)
        self.assertAlmostEqual(predicted_speedup(threshold, 4, .2, 1.), 1., places=7)
        self.assertIsNone(deployment_break_even(500, -.1))
        self.assertEqual(deployment_break_even(500, .1), 5000)
        same = paired_interval({'a': .2, 'b': .8}, {'a': .2, 'b': .8})
        self.assertEqual(same['ci_low'], 0.)
        with self.assertRaises(ValueError):
            paired_interval({'a': .1}, {'b': .1})

    def _tiny_pipeline(self, device):
        torch.set_num_threads(1)
        task = Path(__file__).resolve().parents[1]
        configuration = copy.deepcopy(load_configuration(task))
        configuration['banks'].update(generate_math_cap=6, generate_short_cap=3)
        configuration['sampling'].update(repetitions=1, fixed_length_math_prompts=1, quality_caps={'math': 6, 'short': 3})
        configuration['evaluation'].update(isolated_context_lengths=[8], warmup=1, timing_repetitions=2)
        teacher, student = tiny_model(device=device), tiny_model(6305, device=device)
        prompts = [{'id': f'SYNTHETIC_{i}', 'prompt_sha256': f'SYNTHETIC_{i}',
                    'prompt_token_ids': [1, 2, 3, 4, 5, 6], 'domain': 'math' if i == 0 else 'short',
                    'workload': 'math' if i == 0 else 'familiar', 'family': 1, 'answer': '7'} for i in range(2)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_json(root / 'manifests/run_lock.json', {'config_id': 'SYNTHETIC_EVALUATION_TEST_ONLY'})
            store = ArtifactStore({'task': task, 'results': root}, configuration)
            registry = lambda name: student
            identities = {name: f'SYNTHETIC_{name}' for name in ['teacher', *draft_ids(configuration)]}
            checks = validate_real_caches(teacher, student, prompts)
            self.assertEqual(checks['status'], 'passed')
            self.assertEqual(generate_banks(store, teacher, registry, prompts, identities)['banks'], 7)
            result = evaluate_alignment(store, teacher, registry, prompts)
            self.assertGreater(result['prefix_draft_records'], 0)
            result = benchmark(store, teacher, registry, TestTokenizer(), prompts, baseline=True)
            self.assertEqual(result['paired_records'], 4)
            result = benchmark(store, teacher, registry, TestTokenizer(), prompts, fixed_length=True)
            self.assertEqual(result['paired_records'], 10)
            records = list((root / 'benchmarks/fixed64').rglob('*.json'))
            interrupted = records[0]
            condition = read_json(interrupted)
            interrupted.unlink()
            self.assertEqual(benchmark(store, teacher, registry, TestTokenizer(), prompts, fixed_length=True)['paired_records'], 10)
            warm = read_json(root / f"metrics/warm_setup/fixed64_{condition['draft']}_rep{condition['repetition']}.json")
            self.assertEqual(len(warm['attempt_seconds']), 2)
            self.assertTrue(all(read_json(path)['speculative']['new_tokens'] == 64 for path in records))
            result = isolated_costs(store, teacher, student, 'original')
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['verification']['4']['input_tokens'], 5)
            self.assertEqual(evaluate_quality(store, registry, TestTokenizer(), prompts, ['original'])['records'], 2)

    def test_tiny_cpu_end_to_end_evaluation_paths(self):
        self._tiny_pipeline('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'Tiny CUDA evaluation paths require CUDA.')
    def test_tiny_cuda_end_to_end_evaluation_paths(self):
        self._tiny_pipeline('cuda')


if __name__ == '__main__':
    unittest.main()

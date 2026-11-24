# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Check staff data invariants, classification alignment and independent model budgets."""

import unittest
import copy
import tempfile
from pathlib import Path

import torch

from Task_2.common.config import load_configuration
from Task_2.data.mqar import load_evaluation_stream, load_generator, tensor_batch, validate_example, verify_supplied_inputs
from Task_2.evaluation.recall import aggregate_recall
from Task_2.models.mqar import RecallTransformer
from Task_2.training.protocol import evaluation_conditions, training_conditions
from Task_2.training.recall import learning_rate, model_digest


class RecallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.task = Path(__file__).resolve().parents[1]
        cls.configuration = load_configuration(cls.task)

    def test_course_assets_and_all_frozen_data_invariants(self):
        source = self.task / "resources/course_data"
        self.assertEqual(len(verify_supplied_inputs(source)), 4)
        self.assertEqual(len(load_evaluation_stream(source, "development")), 2304)
        self.assertEqual(len(load_evaluation_stream(source, "test")), 9216)
        generator = load_generator(source)
        for regime in ["near", "far", "mixed"]:
            for count in [4, 8, 16]:
                for index in range(10):
                    self.assertTrue(validate_example(generator.make_example("train", count, regime, index), generator))

    def test_query_position_labels_and_chance_denominators(self):
        example = load_generator(self.task / "resources/course_data").make_example("train", 4, "near", 0)
        batch = tensor_batch([example])
        self.assertEqual(batch["positions"].tolist()[0], example["query_positions"])
        self.assertEqual((batch["labels"] + 128).tolist()[0], example["answer_ids"])
        metrics = aggregate_recall([{**example, "predictions": example["answer_ids"], "query_nll": [0.] * 4}])
        self.assertEqual(metrics["accuracy"], 1.)
        self.assertEqual(metrics["queries"], 4)
        self.assertEqual(metrics["chance_label_space"], 1 / 64)
        self.assertEqual(metrics["chance_present_values"], 1 / 4)

    def test_exact_run_matrix_and_identical_initial_parameters(self):
        self.assertEqual(len(training_conditions(self.configuration)), 42)
        self.assertEqual(len(evaluation_conditions(self.configuration)), 54)
        digests, counts = [], []
        for method in ["dense", "window", "linear", "dilated", "local_content"]:
            torch.manual_seed(6304)
            model = RecallTransformer(self.configuration["mqar"], method)
            digests.append(model_digest(model))
            counts.append(model.parameter_count())
        self.assertEqual(len(set(digests)), 1)
        self.assertEqual(len(set(counts)), 1)

    def test_schedule_boundaries(self):
        settings = self.configuration["mqar"]
        self.assertAlmostEqual(learning_rate(1, settings), .0008 / 200)
        self.assertAlmostEqual(learning_rate(200, settings), .0008)
        self.assertAlmostEqual(learning_rate(4500, settings), .0008 * .15)

    def test_every_mechanism_backpropagates_query_position_loss(self):
        import torch.nn.functional as functional
        generator = load_generator(self.task / "resources/course_data")
        examples = [generator.make_example("train", 4, "near", index) for index in range(2)]
        batch = tensor_batch(examples)
        for method in ["dense", "window", "linear", "dilated", "local_content"]:
            torch.manual_seed(6304)
            model = RecallTransformer(self.configuration["mqar"], method, content_k=32)
            logits = model(batch["tokens"], batch["positions"])
            loss = functional.cross_entropy(logits.flatten(0, 1), batch["labels"].flatten())
            loss.backward()
            self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
            self.assertGreater(float(model.blocks[0].q_proj.weight.grad.norm()), 0.)
            self.assertGreater(float(model.blocks[0].v_proj.weight.grad.norm()), 0.)

    @unittest.skipUnless(torch.cuda.is_available(), "Tiny mixed-precision training smoke requires CUDA.")
    def test_every_mechanism_completes_tiny_cuda_optimizer_and_checkpoint_cycle(self):
        from Task_2.common.artifacts import ArtifactStore, atomic_json
        from Task_2.training.recall import train_condition
        configuration = copy.deepcopy(self.configuration)
        configuration["mqar"].update(hidden_dimension=32, updates=2, batch_size=2,
                                      warmup_updates=1, checkpoint_updates=[2])
        configuration["execution"].update(data_loader_workers=0, recovery_every_updates=100,
                                           evaluation_batch_size=2, progress_every_updates=100)
        development = load_evaluation_stream(self.task / "resources/course_data", "development")
        subset = [row for regime in ["near", "far", "mixed"] for row in
                  [item for item in development if item["K"] == 4 and item["regime"] == regime][:2]]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_json(root / "manifests/run_lock.json", {"config_id": "SYNTHETIC_CUDA_TRAINING_TEST_ONLY"})
            store = ArtifactStore({"task": self.task, "resources": self.task / "resources/course_data", "results": root}, configuration)
            for method in ["dense", "window", "linear", "dilated", "local_content"]:
                condition = {"id": f"tiny_cuda_{method}", "method": method, "train_K": 4,
                             "train_regime": "near", "seed": 6304}
                summary = train_condition(store, condition, subset, content_k=32, device="cuda")
                self.assertEqual(summary["status"], "complete")
                self.assertEqual(summary["completed_updates"], 2)
                self.assertEqual(len([path for path in summary["training_query_maps"] if path.endswith(".npz")]), 1)
                checkpoint = torch.load(store.path(f"checkpoints/mqar/{condition['id']}/recovery.pt"), map_location="cpu", weights_only=True)
                self.assertTrue(checkpoint["scaler"])
                self.assertEqual(checkpoint["next_training_index"], 4)


if __name__ == "__main__":
    unittest.main()

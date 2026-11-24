# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Compare interrupted training with an uninterrupted tiny run, including full state."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from Task_2.cloud.colab import DriveSync
from Task_2.common.artifacts import ArtifactStore, SessionTimeLimit, atomic_json, read_json
from Task_2.common.config import fingerprint, load_configuration
from Task_2.data.mqar import load_evaluation_stream, load_generator
from Task_2.training.recall import train_condition


class InterruptingStore(ArtifactStore):
    def __init__(self, paths, configuration):
        super().__init__(paths, configuration)
        self.calls = 0

    def check_time(self):
        self.calls += 1
        if self.calls == 2:
            raise SessionTimeLimit("Synthetic interruption after one completed optimizer update.")


class RecoveryTests(unittest.TestCase):
    def test_interrupted_optimizer_rng_sampler_and_schedule_resume_exactly(self):
        torch.set_num_threads(1)
        task = Path(__file__).resolve().parents[1]
        configuration = copy.deepcopy(load_configuration(task))
        configuration["mqar"].update(hidden_dimension=16, updates=2, batch_size=2,
                                      warmup_updates=1, checkpoint_updates=[2])
        configuration["execution"].update(data_loader_workers=0, recovery_every_updates=100,
                                           evaluation_batch_size=2, progress_every_updates=100)
        development = load_evaluation_stream(task / "resources/course_data", "development")
        subset = [row for regime in ["near", "far", "mixed"] for row in
                  [item for item in development if item["K"] == 4 and item["regime"] == regime][:2]]
        condition = {"id": "tiny_resume_test", "method": "dense", "train_K": 4, "train_regime": "near", "seed": 6304}
        with tempfile.TemporaryDirectory() as temporary:
            roots = [Path(temporary) / name for name in ["uninterrupted", "resumed"]]
            stores = []
            for index, root in enumerate(roots):
                paths = {"task": task, "resources": task / "resources/course_data", "results": root}
                atomic_json(root / "manifests/run_lock.json", {"config_id": "SYNTHETIC_RECOVERY_TEST_ONLY"})
                stores.append(ArtifactStore(paths, configuration) if index == 0 else InterruptingStore(paths, configuration))
            uninterrupted = train_condition(stores[0], condition, subset, device="cpu")
            with self.assertRaises(SessionTimeLimit):
                train_condition(stores[1], condition, subset, device="cpu")
            partial = torch.load(roots[1] / "checkpoints/mqar/tiny_resume_test/recovery.pt", map_location="cpu", weights_only=True)
            self.assertEqual(tuple(partial["pending_queries"].shape), (2, 4, 4))
            resumed = train_condition(ArtifactStore(stores[1].paths, configuration), condition, subset, device="cpu")
            self.assertEqual(uninterrupted["final_sha256"], resumed["final_sha256"])
            self.assertEqual(resumed["completed_updates"], 2)
            recovery = torch.load(roots[1] / "checkpoints/mqar/tiny_resume_test/recovery.pt", map_location="cpu", weights_only=True)
            self.assertEqual(recovery["next_training_index"], 4)
            self.assertEqual(recovery["schedule_state"]["completed_updates"], 2)
            self.assertTrue(recovery["optimizer"]["state"])
            self.assertEqual(len(recovery["history"]), 2)
            map_path = next(path for path in resumed["training_query_maps"] if path.endswith(".npz"))
            generator = load_generator(task / "resources/course_data")
            expected = []
            for index in range(4):
                example = generator.make_example("train", 4, "near", index)
                expected.append(list(zip(example["query_positions"], example["answer_ids"], example["source_positions"], example["distances"])))
            with np.load(roots[1] / map_path, allow_pickle=False) as maps:
                np.testing.assert_array_equal(maps["queries"], np.array(expected, dtype=np.uint8))

    def test_verified_drive_recovery_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local, remote = root / "local", root / "drive"
            atomic_json(local / "metrics/test.json", {"synthetic_test": True, "update": 100})
            sync = DriveSync(local, remote)
            sync.publish(["metrics/test.json"])
            restored = root / "restored"
            self.assertEqual(DriveSync(restored, remote).restore(), 1)
            self.assertEqual((restored / "metrics/test.json").read_bytes(), (local / "metrics/test.json").read_bytes())
            (remote / "metrics/test.json").write_text("corrupt", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                DriveSync(root / "another", remote).restore()

    def test_final_evaluation_oom_is_saved_without_partial_accuracy(self):
        from Task_2.training.pipeline import ComputeSession
        task = Path(__file__).resolve().parents[1]
        configuration = load_configuration(task)
        condition = {"id": "SYNTHETIC_FINAL_EVALUATION", "training_id": "SYNTHETIC_TRAINING",
                     "method": "dense", "seed": 6304, "train_K": 4, "train_regime": "mixed",
                     "test_method": "dense", "test_only_mask": False}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = ArtifactStore({"task": task, "resources": task / "resources/course_data", "results": root}, configuration)
            intervention = {"synthetic_test_fixture": True}
            session = ComputeSession.__new__(ComputeSession)
            session.store, session.paths, session.configuration = store, store.paths, configuration
            session.lock = {"config_id": "SYNTHETIC_FAILURE_TEST", "intervention_id": fingerprint(intervention), "mqar_final_test_started": False}
            store.save_json("manifests/run_lock.json", session.lock)
            store.save_json("configs/intervention_locked.json", intervention)
            store.save_json("metrics/mqar_training/SYNTHETIC_TRAINING.json",
                {"status": "complete", "final_checkpoint": "SYNTHETIC.pt", "content_k": 64})
            store.complete("train_SYNTHETIC_TRAINING", [])
            model = MagicMock()
            model.cuda.return_value = model
            with patch("Task_2.training.pipeline.training_conditions", return_value=[{"id": "SYNTHETIC_TRAINING"}]), \
                 patch("Task_2.training.pipeline.evaluation_conditions", return_value=[condition]), \
                 patch("Task_2.training.pipeline.load_evaluation_stream", return_value=[]), \
                 patch("Task_2.models.mqar.RecallTransformer", return_value=model), \
                 patch("torch.load", return_value={"config_id": session.lock["config_id"], "update": 4500, "model": {}}), \
                 patch("Task_2.evaluation.recall.evaluate_final_condition", side_effect=torch.OutOfMemoryError("SYNTHETIC TEST OOM")):
                session.evaluate_mqar()
            result = read_json(root / "metrics/mqar_test/SYNTHETIC_FINAL_EVALUATION.json")
            self.assertEqual(result["status"], "failed_cuda_oom")
            self.assertEqual(result["examples"], 0)
            self.assertEqual(result["cells"], [])
            self.assertNotIn("accuracy", result)
            self.assertTrue(store.completed("mqar_test_SYNTHETIC_FINAL_EVALUATION"))


if __name__ == "__main__":
    unittest.main()

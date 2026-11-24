# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Verify durable recovery and lightweight replay boundaries using temporary files."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from Task_1.cloud.colab import DriveSync, bootstrap_project
from Task_1.cloud.dependencies import matches_pin
from Task_1.common.artifacts import ArtifactStore, atomic_json
from Task_1.common.config import load_configuration
from Task_1.common.paths import resolve_paths
from Task_1.common.replay import validate_replay
from Task_1.data.prepare import (SUPPLIED_INPUTS, ensure_supplied_source,
                                 quarantine_incomplete_materializations, verify_supplied_inputs)
from Task_1.training.protocol import experiment_conditions


class ArtifactTests(unittest.TestCase):
    """Protect complete results, stale outputs, and recovery integrity."""

    def test_verified_backup_and_restore(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local, remote = root / "local", root / "drive"
            atomic_json(local / "metrics/example.json", {"value": 1})
            sync = DriveSync(local, remote)
            sync.publish(["metrics/example.json"])
            (local / "metrics/example.json").unlink()
            self.assertEqual(DriveSync(local, remote).restore(), 1)
            self.assertEqual(json.loads((local / "metrics/example.json").read_text())["value"], 1)
            (remote / "metrics/example.json").write_text("corrupt", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                DriveSync(local, remote).restore()

    def test_completed_stage_checks_file_content(self):
        configuration = load_configuration(Path(__file__).resolve().parents[1])
        with tempfile.TemporaryDirectory() as temporary:
            paths = {"results": Path(temporary)}
            store = ArtifactStore(paths, configuration)
            store.save_json("metrics/result.json", {"value": 1})
            store.complete("stage", ["metrics/result.json"])
            self.assertTrue(store.completed("stage"))
            store.path("metrics/result.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                store.completed("stage")

    def test_replay_import_does_not_import_torch(self):
        result = subprocess.run([sys.executable, "-c", "import sys; import Task_1.common.replay; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
                                cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_replay_does_not_fall_back_to_compute(self):
        configuration = load_configuration(Path(__file__).resolve().parents[1])
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "never downloads a model"):
                validate_replay(configuration, {"results": Path(temporary)})

    def test_complete_condition_matrix(self):
        conditions = experiment_conditions()
        self.assertEqual(len(conditions), 18)
        self.assertEqual(len({row["id"] for row in conditions}), 18)
        self.assertEqual(sum(row["hellaswag_count"] == 2000 for row in conditions), 10)

    def test_cuda_version_suffix_does_not_force_reinstall_loop(self):
        self.assertTrue(matches_pin("2.8.0+cu128", "2.8.0"))
        self.assertFalse(matches_pin("2.11.0+cu128", "2.8.0"))
        self.assertFalse(matches_pin(None, "2.8.0"))

    def test_interrupted_materialization_is_retained_for_recovery(self):
        configuration = load_configuration(Path(__file__).resolve().parents[1])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            public = source / "data/public_locked"
            public.mkdir(parents=True)
            interrupted = public / "c4_calibration_locked.jsonl.gz"
            interrupted.write_bytes(b"not a complete gzip file")
            store = ArtifactStore({"results": root / "results", "data": root / "Data/Task_1"}, configuration)
            records = quarantine_incomplete_materializations(source, store)
            self.assertEqual(len(records), 1)
            self.assertFalse(interrupted.exists())
            self.assertTrue(Path(records[0]["retained_at"]).is_file())

    def test_actual_supplied_task_inputs_match_checksums(self):
        source = Path(__file__).resolve().parents[2] / "Data/atml-pa4-datasets-main"
        if not source.is_dir():
            source = Path(__file__).resolve().parents[1] / "resources/course_data"
        self.assertEqual(len(verify_supplied_inputs(source)), len(SUPPLIED_INPUTS))

    def test_task_folder_alone_prepares_runtime_without_separate_data_upload(self):
        task_root = Path(__file__).resolve().parents[1]
        configuration = load_configuration(task_root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            drive = root / "MyDrive/ATML_PA4/PA_4"
            shutil.copytree(task_root, drive / "Task_1",
                            ignore=shutil.ignore_patterns("results", "__pycache__"))
            remote_result = drive / "Task_1/results/existing_run/metrics.json"
            atomic_json(remote_result, {"preserve": True})
            local = bootstrap_project(drive, root / "content/ATML_PA4/PA_4")
            self.assertFalse((drive / "Data").exists())
            self.assertFalse((local / "Task_1/results").exists())
            paths = resolve_paths(configuration, local)
            ensure_supplied_source(paths["source"], paths["task"] / "resources/course_data")
            self.assertEqual(len(verify_supplied_inputs(paths["source"])), len(SUPPLIED_INPUTS))
            self.assertEqual(json.loads(remote_result.read_text()), {"preserve": True})
            self.assertEqual(paths["drive_results"].as_posix(),
                             "/content/drive/MyDrive/ATML_PA4/PA_4/Task_1/results/task1_seed6304_v1")
            locked = paths["source"] / "data/local_source_revisions.json"
            atomic_json(locked, {"preserve": "existing revision lock"})
            ensure_supplied_source(paths["source"], paths["task"] / "resources/course_data")
            self.assertEqual(json.loads(locked.read_text()), {"preserve": "existing revision lock"})

    def test_bundled_public_script_inputs_reject_changed_gsm8k_ids(self):
        bundled = Path(__file__).resolve().parents[1] / "resources/course_data"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(bundled, root / "bundle")
            (root / "bundle/data/public_manifests/gsm8k_train_ids.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "gsm8k_train_ids"):
                ensure_supplied_source(root / "Data/source", root / "bundle")
            self.assertFalse((root / "Data").exists())


if __name__ == "__main__":
    unittest.main()

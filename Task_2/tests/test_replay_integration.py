# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Use explicitly synthetic temporary evidence to exercise the complete replay notebook."""

import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Task_2.analysis.evidence import build_evidence, collect_tables, create_figures, required_stages
from Task_2.cache.packing import payload_bytes
from Task_2.cache.selection import selection_plan
from Task_2.common.artifacts import ArtifactStore, atomic_json, atomic_jsonl, read_json, sha256_file
from Task_2.common.config import fingerprint, load_configuration, protocol_id
from Task_2.common.paths import resolve_paths
from Task_2.common.replay import validate_replay
from Task_2.data.mqar import load_evaluation_stream, load_generator, verify_supplied_inputs, write_gzip_records
from Task_2.data.retrieval import load_retrieval_cases
from Task_2.evaluation.recall import aggregate_recall
from Task_2.evaluation.statistics import KV_PAIRS, paired_accuracy_interval
from Task_2.training.protocol import evaluation_conditions, training_conditions


def fake_measure(milliseconds=1.):
    """Mark synthetic timing values clearly; never treat them as measured benchmarks."""
    return {"samples_ms": [milliseconds] * 10, "median_ms": milliseconds, "warmup_repetitions": 3,
            "timing_repetitions": 10, "peak_allocated_bytes": 1000000, "peak_reserved_bytes": 2000000,
            "additional_peak_allocated_bytes": 10000, "synthetic_test_fixture": True}


def synthetic_fixture(store):
    """Write full protocol-sized fake records only under a TemporaryDirectory."""
    configuration = store.configuration
    identity = protocol_id(configuration, store.paths["task"])
    data = {"supplied_hashes": verify_supplied_inputs(store.paths["resources"]), "synthetic_test_fixture": True}
    intervention = {"content_k": 64, "method": "local_content", "final_test_accessed": False, "synthetic_test_fixture": True}
    store.save_json("splits/data_manifest.json", data)
    store.save_json("configs/intervention_locked.json", intervention)
    store.save_json("manifests/run_lock.json", {"run_id": configuration["execution"]["run_id"], "config_id": identity,
        "data_id": fingerprint(data), "intervention_id": fingerprint(intervention), "mqar_final_test_started": True,
        "runtime": {"gpu": "SYNTHETIC TEST ONLY", "synthetic_test_fixture": True}})
    attention = []
    for method in ["dense", "tiled", "sdpa"]:
        selected = []
        for dtype in ["float16", "float32"]:
            for length in [256, 512, 1024, 2048, 4096, 8192]:
                row = {"method": method, "dtype": dtype, "length": length, "status": "complete",
                       "reference_status": "SYNTHETIC", "output_error": {"relative_l2": 1e-5},
                       "gradient_errors": [{"relative_l2": 1e-5}] * 3,
                       "forward": fake_measure(length / 256), "forward_backward": fake_measure(length / 128),
                       "observed_operators": ["SYNTHETIC TEST ONLY"], "synthetic_test_fixture": True}
                selected.append(row)
                attention.append(row)
        store.save_json(f"metrics/attention/{method}_conditions.json", selected)
    store.save_json("metrics/attention/all_conditions.json", attention)
    store.save_json("metrics/attention/correctness.json", {"status": "passed", "synthetic_test_fixture": True})
    for condition_index, condition in enumerate(training_conditions(configuration)):
        name = condition["id"]
        history = [{"update": update, "loss": 1 + math.exp(-update / 1000) + condition_index / 100,
                    "lr": .0008, "synthetic_test_fixture": True} for update in range(1, 4501)]
        history_path = f"histories/mqar/{name}.jsonl"
        atomic_jsonl(store.path(history_path), history)
        first_index = (condition["seed"] - 6304) * 4500 * 32
        stem = f"splits/mqar_training/seed{condition['seed']}_K{condition['train_K']}_{condition['train_regime']}/queries_{first_index:09d}_{first_index + 144000:09d}"
        map_path, receipt = stem + ".npz", stem + ".json"
        if not store.path(map_path).is_file():
            example = load_generator(store.paths["resources"]).make_example("train", condition["train_K"], condition["train_regime"], first_index)
            annotations = np.array(list(zip(example["query_positions"], example["answer_ids"], example["source_positions"], example["distances"])), dtype=np.uint8)
            store.path(map_path).parent.mkdir(parents=True, exist_ok=True)
            with store.path(map_path).open("wb") as handle:
                np.savez_compressed(handle, queries=np.broadcast_to(annotations, (144000, condition["train_K"], 4)), start_index=np.array([first_index], dtype=np.int64))
            store.save_json(receipt, {"seed": condition["seed"], "K": condition["train_K"], "regime": condition["train_regime"],
                "start_index": first_index, "stop_index": first_index + 144000, "examples": 144000,
                "sha256": sha256_file(store.path(map_path)), "generator_sha256": data["supplied_hashes"]["scripts/mqar_generator.py"],
                "synthetic_test_fixture": True})
        summary = {**condition, "status": "complete", "completed_updates": 4500,
            "checkpoint_updates": [500, 1000, 1500, 2000, 3000, 4500], "history": history_path,
            "trainable_parameters": 437824, "baseline_trainable_parameters": 437824,
            "initialization_sha256": fingerprint({"seed": condition["seed"]}), "final_sha256": fingerprint(condition),
            "training_seconds": 10., "skipped_optimizer_attempts": 0, "peak_allocated_bytes": 1000000,
            "training_query_maps": [map_path, receipt], "synthetic_test_fixture": True}
        store.save_json(f"metrics/mqar_training/{name}.json", summary)
    test = load_evaluation_stream(store.paths["resources"], "test")
    for condition_index, condition in enumerate(evaluation_conditions(configuration)):
        cells = []
        for count in [4, 8, 16]:
            for regime in ["near", "far", "mixed"]:
                original = [row for row in test if row["K"] == count and row["regime"] == regime]
                records = []
                for index, example in enumerate(original):
                    prediction = example["answer_ids"].copy()
                    if index % 10 < condition_index % 7:
                        prediction[0] = 128 + (prediction[0] - 128 + 1) % 64
                    records.append({key: example[key] for key in ["id", "split", "K", "regime", "query_positions",
                        "source_positions", "distances", "answer_ids"]} | {"predictions": prediction,
                        "query_nll": [1. + condition_index / 100] * count, "synthetic_test_fixture": True})
                relative = f"predictions/mqar/{condition['id']}/K{count}_{regime}_0000.jsonl.gz"
                write_gzip_records(store.path(relative), records)
                cells.append({"test_K": count, "regime": regime, **aggregate_recall(records)})
        store.save_json(f"metrics/mqar_test/{condition['id']}.json", {**condition, "status": "complete",
            "examples": 9216, "cells": cells, "synthetic_test_fixture": True})
    inference = [{"method": method, "length": length, "status": "complete", "timing": fake_measure(length / 128),
        "attended_pairs": length * 64, "theoretical_incremental_state_bytes": 1000, "implementation": "SYNTHETIC TEST ONLY"}
        for method in ["dense", "window", "linear", "dilated", "local_content"] for length in [256, 1024, 4096, 8192]]
    store.save_json("metrics/mqar_inference/all_conditions.json", inference)
    tokenized, predictions = [], []
    geometry = (2, 1, 1, 64, 1024)
    budget = 32 + payload_bytes(*geometry, 4)
    staff = load_retrieval_cases(store.paths["resources"])
    for index, case in enumerate(staff):
        prompt_hash = fingerprint({"id": case["id"], "synthetic_test_fixture": True})
        tokenized.append({**case, "prefix_length": 1024, "suffix_length": 16, "actual_prompt_tokens": 1040,
            "token_ids": [2] * 1024 + [3] * 16, "prompt_sha256": prompt_hash,
            "evidence_token_positions": [8, 9, 10], "messages": [{"role": "user", "content": case["user_text"]}],
            "synthetic_test_fixture": True})
        for method_index, method in enumerate(configuration["kv"]["methods"]):
            plan = selection_plan(method, geometry, budget, torch.linspace(0, 1, 1024))
            correct = method == "fp16" or (index + method_index) % 17 != 0
            response = case["code"] if correct else "00000"
            row = {"id": case["id"], "method": method, "status": "complete", "prompt_sha256": prompt_hash,
                "target_tokens": case["target_tokens"], "position": case["position"], "distractors": case["distractors"],
                "actual_prompt_tokens": 1040, "prefix_tokens": 1024, "suffix_tokens": 16,
                "gold_code": case["code"], "correct": correct, "extracted_code": response, "fp16_reference_correct": True,
                "fp16_reference_answer_token_ids": [1, 2, 3],
                "reference_answer_nll": {"nll_sum": 3., "tokens": 3, "mean_nll": 1.},
                "gold_code_nll": {"nll_sum": 3., "tokens": 3, "mean_nll": 1.},
                "post_prefill_logit_kl": method_index * .001, "reference_answer_teacher_forced_kl": method_index * .001,
                "reference_int4_budget_bytes": budget, "actual_serialized_prefix_bytes": plan["expected_bytes"],
                "full_fp16_prefix_bytes": 32 + 4 * 2 * 64 * 1024,
                "fraction_full_fp16_bytes": plan["expected_bytes"] / (32 + 4 * 2 * 64 * 1024),
                "selected_fp16_positions": plan["positions"], "evidence_positions": [8, 9, 10],
                "evidence_retention_fraction": len(set(plan["positions"]) & {8, 9, 10}) / 3 if plan["positions"] else 1.,
                "prefill_seconds": .02, "question_importance_seconds": .002 if method in {"question_aware", "hybrid"} else 0.,
                "packed_encode_seconds": .001, "question_suffix_seconds": .001, "teacher_forced_scoring_seconds": .002,
                "peak_allocated_bytes": 1000000, "response": {"text": response, "token_ids": [1, 2, 3],
                    "decode_seconds": .01, "new_tokens": 3, "eos": True, "truncated": False},
                "synthetic_test_fixture": True}
            store.save_json(f"predictions/kv/{case['id']}/{method}.json", row)
            predictions.append(row)
    write_gzip_records(store.path("splits/retrieval_tokenized.jsonl.gz"), tokenized)
    store.save_json("metrics/kv/model_geometry.json", {"model": configuration["kv"]["model"], "revision": configuration["kv"]["revision"],
        "layers": 2, "kv_heads": 1, "head_dimension": 64, "synthetic_test_fixture": True})
    store.save_json("metrics/kv/reference_checks.json", {"status": "passed", "synthetic_test_fixture": True})
    for family in ["eviction", "quantization", "hybrid"]:
        store.save_json(f"metrics/kv/policy_{family}.json", {"status": "passed", "conditions": [], "synthetic_test_fixture": True})
    intervals = [{"method": left, "reference": right, **paired_accuracy_interval(
        [row for row in predictions if row["method"] == left], [row for row in predictions if row["method"] == right])}
        for left, right in KV_PAIRS]
    store.save_json("metrics/kv/paired_comparisons.json", intervals)
    systems = []
    for length in [2048, 4096]:
        for batch in [1, 8]:
            shape = (2, batch, 1, 64, length)
            cap = 32 + payload_bytes(*shape, 4)
            for method in configuration["kv"]["methods"]:
                plan = selection_plan(method, shape, cap, torch.arange(length).float())
                systems.append({"method": method, "batch": batch, "context_length": length, "status": "complete",
                    "serialized_bytes": plan["expected_bytes"], "calculated_bytes": plan["expected_bytes"],
                    "encode": fake_measure(), "unpack": fake_measure(2.), "decode_step": fake_measure(3.), "synthetic_test_fixture": True})
    store.save_json("metrics/kv_systems/all_conditions.json", systems)
    for stage in required_stages(configuration):
        store.complete(stage, [], {"synthetic_test_fixture": True})


class ReplayIntegrationTests(unittest.TestCase):
    def test_full_evidence_and_notebook_replay_without_compute_imports(self):
        task = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            shutil.copytree(task, project / "Task_2", ignore=shutil.ignore_patterns("results", "__pycache__"))
            configuration = load_configuration(project / "Task_2")
            paths = resolve_paths(configuration, project)
            store = ArtifactStore(paths, configuration)
            synthetic_fixture(store)
            result = build_evidence(store)
            self.assertEqual(result["figures"], 8)
            self.assertEqual(validate_replay(configuration, paths)["status"], "complete")
            preview = os.environ.get("ATML_TASK2_QA_DIR")
            if preview:
                destination = Path(preview).resolve()
                workspace = task.parent.resolve()
                self.assertTrue(destination.is_relative_to(workspace / "tmp"))
                shutil.copytree(store.path("figures"), destination, dirs_exist_ok=True)
                shutil.copytree(store.path("tables"), destination / "tables", dirs_exist_ok=True)
            script = "\n".join([
                "import importlib.abc,sys",
                "class BlockComputeImports(importlib.abc.MetaPathFinder):",
                "    def find_spec(self, fullname, path=None, target=None):",
                "        if fullname.split('.')[0] in {'torch','transformers','accelerate','huggingface_hub'}:",
                "            raise AssertionError('Replay attempted a compute import: ' + fullname)",
                "sys.meta_path.insert(0,BlockComputeImports())",
                "from Task_2.scripts.validate_project import execute_replay_notebook",
                "print(execute_replay_notebook('.',working_directory='Task_2'))",
            ])
            process = subprocess.run([sys.executable, "-B", "-c", script], cwd=project, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertEqual(int(process.stdout.strip()), 38)
            # A failed final evaluation remains an explicit attempted condition,
            # even when its model trained successfully. It cannot acquire scores.
            condition = evaluation_conditions(configuration)[0]
            failed_path = f"metrics/mqar_test/{condition['id']}.json"
            store.save_json(failed_path, {**condition, "status": "failed_cuda_oom",
                "error": "SYNTHETIC TEST OOM", "training_status": "complete", "examples": 0, "cells": []})
            self.assertEqual(build_evidence(store)["status"], "complete_with_recorded_failures")
            self.assertEqual(validate_replay(configuration, paths, deep=False)["status"], "complete_with_recorded_failures")
            grid = pd.read_csv(store.path("tables/mqar_accuracy.csv"))
            unavailable = grid[grid["id"] == condition["id"]]
            self.assertEqual(len(unavailable), 9)
            self.assertTrue(unavailable["accuracy"].isna().all())
            self.assertTrue((unavailable["examples"] == 0).all())
            # A hash alone must not hide a changed answer: recheck its semantic scorer.
            relative = f"predictions/kv/{load_retrieval_cases(paths['resources'])[0]['id']}/fp16.json"
            changed = read_json(store.path(relative))
            changed["response"]["text"] = "no answer"
            atomic_json(store.path(relative), changed)
            manifest = read_json(store.path("manifests/run.json"))
            manifest["required_artifacts"][relative] = sha256_file(store.path(relative))
            atomic_json(store.path("manifests/run.json"), manifest)
            with self.assertRaises(ValueError):
                validate_replay(configuration, paths, deep=False)
            # Even an entirely unsuccessful MQAR section must produce readable
            # tables and empty plots, rather than a CSV/parser/plotting crash.
            for row in training_conditions(configuration):
                summary = read_json(store.path(f"metrics/mqar_training/{row['id']}.json"))
                atomic_jsonl(store.path(summary["history"]), [])
            for row in evaluation_conditions(configuration):
                store.save_json(f"metrics/mqar_test/{row['id']}.json", {**row,
                    "status": "failed_cuda_oom", "error": "SYNTHETIC TEST OOM", "examples": 0, "cells": []})
            frames = collect_tables(store)
            self.assertTrue(frames["curves"].empty)
            self.assertEqual(len(frames["recall"]), 54 * 9)
            self.assertTrue(frames["recall"]["accuracy"].isna().all())
            self.assertEqual(len(create_figures(store, frames)[0]), 8)
            self.assertTrue(pd.read_csv(store.path("tables/mqar_learning_curves.csv")).empty)


if __name__ == "__main__":
    unittest.main()

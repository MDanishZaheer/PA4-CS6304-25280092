# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Exercise all evidence generation and notebook replay using explicit fake fixtures."""

import gzip
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from Task_1.analysis.evidence import build_evidence
from Task_1.common.artifacts import ArtifactStore, atomic_json, sha256_file
from Task_1.common.config import (fingerprint, load_configuration,
                                  scientific_configuration, source_fingerprint)
from Task_1.common.paths import resolve_paths
from Task_1.common.replay import validate_replay
from Task_1.data.hellaswag import format_hellaswag
from Task_1.data.prepare import TASK_FILES, validate_task_data
from Task_1.training.protocol import experiment_conditions


def synthetic_fixture(store):
    """Write temporary synthetic evidence; never write into real result directories."""
    data = {"calibration": [], "development": [], "wikitext2": [], "c4": [], "hellaswag": []}
    for index in range(160):
        row = {"source_row": index, "token_offset": 0, "token_ids": [1] * 512, "raw_text_sha256": f"train-{index}"}
        data["calibration" if index < 128 else "development"].append(row)
    for suite in ["wikitext2", "c4"]:
        remaining = 50000
        while remaining:
            count = min(511, remaining)
            index = len(data[suite])
            data[suite].append({"source": suite, "source_row": index, "token_offset": 0,
                                "token_ids": [2] * (count + 1), "score_target_offsets": [1, count + 1],
                                "scored_tokens": count, "raw_text_sha256": f"{suite}-{index}"})
            remaining -= count
    for index in range(2000):
        data["hellaswag"].append({"order": index, "source_row_index": index, "example": {
            "ctx_a": "Synthetic fixture", "ctx_b": "does not contain experiment results", "activity_label": "Test",
            "endings": ["a", "b", "c", "d"], "label": str(index % 4)}})
    hashes = {}
    for name, filename in TASK_FILES.items():
        path = store.path(f"splits/materialized/{filename}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for row in data[name]:
                handle.write(json.dumps(row) + "\n")
        hashes[filename] = sha256_file(path)
    selection = {"summary": validate_task_data(data, store.configuration), "materialized_hashes": hashes,
                 "synthetic_test_fixture": True}
    store.save_json("splits/selection.json", selection)
    profiles = {}
    for name in ["model.layers.0.mlp.up_proj", "model.layers.0.mlp.down_proj"]:
        errors = {str(value): {"relative_mse": value / 10, "squared_error": value, "squared_reference": 10., "sample_tokens": 512}
                  for value in [.1, .25, .5, .75, .85]}
        profiles[name] = {"name": name, "shape": [8, 128], "elements": 1024, "layer": 0,
                          "pruning_sensitivity": .05, "quantization_sensitivity": .01, "probe_seconds": .01,
                          "activation_norms": {"quantiles": [1., 1., 1., 1., 1., 2., 2., 3., 4.]},
                          "magnitude_scores": {"quantiles": [.01] * 9}, "wanda_scores": {"quantiles": [.02] * 9},
                          "magnitude": {key: value for key, value in errors.items() if key in {"0.25", "0.5", "0.75"}}, "wanda": errors}
    plans = {"pruning": {}, "quantization": {}}
    for target in [.5, .75]:
        plans["pruning"][str(target)] = {"zero_count": round(target * 2048), "temperature": .25,
            "allocations": {name: {"zero_count": round(target * 1024), "sparsity": target} for name in profiles},
            "development": {"mean_nll": 1.}, "candidate_results": [{"seconds": .01}]}
    for bits in [3, 4]:
        plans["quantization"][str(bits)] = {"reference_budget_bytes": 10000, "packed_bytes": 9000,
            "unused_bytes": 1000, "allocations": {name: bits for name in profiles}}
    store.save_json("configs/allocations_locked.json", plans)
    store.save_json("metrics/sensitivity_profiles.json", profiles)
    store.save_json("metrics/calibration.json", {"seconds": .01, "synthetic_test_fixture": True})
    protocol_id = fingerprint({"scientific_configuration": scientific_configuration(store.configuration),
                               "source_id": source_fingerprint(store.paths["task"]), "conditions": experiment_conditions()})
    lock = {"task": "Task_1", "run_id": store.configuration["execution"]["run_id"], "config_id": protocol_id,
            "data_id": fingerprint(selection), "allocation_lock_id": fingerprint(plans),
            "runtime": {"gpu": "SYNTHETIC TEST ONLY"}, "synthetic_test_fixture": True}
    store.save_json("manifests/run_lock.json", lock)
    for condition_index, condition in enumerate(experiment_conditions()):
        name = condition["id"]
        mean_nll = 1 + condition_index * .01
        summary = {"condition": name, "status": "complete", "config_id": protocol_id, "data_id": lock["data_id"],
                   "compression_seconds_including_restore": .01, "mean_projection_calibration_relative_mse": condition_index / 100,
                   "storage": {"achieved_sparsity": condition.get("sparsity", 0), "estimated_packed_model_bytes": 9000,
                               "nominal_bits_per_eligible_weight": condition.get("bits", 16), "effective_nonzeros": 1024},
                   "quality": {}, "latency": {"shapes": {}}}
        for suite in ["wikitext2", "c4"]:
            predictions = [{"example_id": f"{suite}:{row['source_row']}:0", "condition": name, "suite": suite,
                            "token_ids_sha256": fingerprint(row["token_ids"]), "score_target_offsets": row["score_target_offsets"],
                            "scored_tokens": row["scored_tokens"], "nll_sum": mean_nll * row["scored_tokens"]}
                           for row in data[suite]]
            store.save_records(f"predictions/{name}/{suite}.jsonl", predictions)
            summary["quality"][suite] = {"mean_nll": mean_nll, "perplexity": math.exp(mean_nll), "scored_tokens": 50000}
        predictions = []
        for row in data["hellaswag"][:condition["hellaswag_count"]]:
            prompt, endings, gold = format_hellaswag(row["example"])
            scores = [-1., -2., -3., -4.]
            predictions.append({"example_id": f"hellaswag:{row['source_row_index']}", "order": row["order"], "condition": name,
                "suite": "hellaswag", "gold": gold, "prompt": prompt, "endings": endings,
                "prompt_sha256": fingerprint({"prompt": prompt, "endings": endings}),
                "log_likelihoods": scores, "character_lengths": [1] * 4, "character_normalized_scores": scores,
                "prediction": 0, "prediction_normalized": 0, "correct": int(gold == 0), "correct_normalized": int(gold == 0)})
        store.save_records(f"predictions/{name}/hellaswag.jsonl", predictions)
        summary["quality"]["hellaswag"] = {"examples": len(predictions), "accuracy": .25, "accuracy_normalized": .25}
        for length in ["512", "2048"]:
            summary["latency"]["shapes"][length] = {"prefill_ms": [1.] * 5, "decode_64_steps_ms": [2.] * 5,
                "prefill_median_ms": 1., "decode_median_ms": 2., "peak_allocated_bytes": 10000,
                "sdpa_observed_operators": ["SYNTHETIC TEST ONLY"]}
        store.save_json(f"metrics/{name}/summary.json", summary)


class ReplayIntegrationTests(unittest.TestCase):
    """Check every notebook cell against synthetic cloud-shaped evidence."""

    def test_evidence_and_fresh_process_notebook_replay(self):
        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            shutil.copytree(source, project / "Task_1", ignore=shutil.ignore_patterns("results", "__pycache__"))
            configuration = load_configuration(project / "Task_1")
            paths = resolve_paths(configuration, project)
            store = ArtifactStore(paths, configuration)
            synthetic_fixture(store)
            result = build_evidence(store)
            self.assertEqual(result["conditions"], 18)
            self.assertEqual(validate_replay(configuration, paths)["status"], "complete")
            # Mimic downloading only results into a second checkout. No raw
            # Data directory, Drive mount, or large recovery checkpoint is needed.
            local = project / "downloaded_local"
            shutil.copytree(source, local / "Task_1",
                            ignore=shutil.ignore_patterns("results", "__pycache__"))
            shutil.copytree(paths["results"].parent, local / "Task_1/results",
                            ignore=shutil.ignore_patterns("checkpoints"))
            self.assertFalse((local / "Data").exists())
            self.assertEqual(validate_replay(configuration, resolve_paths(configuration, local))["status"], "complete")
            script = "\n".join([
                "import importlib.abc, sys",
                "class BlockComputeImports(importlib.abc.MetaPathFinder):",
                "    def find_spec(self, fullname, path=None, target=None):",
                "        if fullname.split('.')[0] in {'torch', 'transformers', 'datasets', 'accelerate'}:",
                "            raise AssertionError('Replay attempted a compute import: ' + fullname)",
                "sys.meta_path.insert(0, BlockComputeImports())",
                "from Task_1.scripts.validate_project import execute_replay_notebook",
                "print(execute_replay_notebook('.', working_directory='Task_1'))",
            ])
            process = subprocess.run([sys.executable, "-c", script], cwd=local, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertGreater(int(process.stdout.strip()), 25)
            failure_path = paths["results"] / "metrics/p3_s75/summary.json"
            failed = json.loads(failure_path.read_text(encoding="utf-8"))
            failed["status"] = "failed_cuda_oom"
            failed["quality"] = {}
            failed["error"] = "SYNTHETIC TEST FAILURE ONLY"
            atomic_json(failure_path, failed)
            build_evidence(store)
            failed_manifest = validate_replay(configuration, paths)
            self.assertEqual(failed_manifest["status"], "complete_with_recorded_failures")
            self.assertIn("p3_s75", failed_manifest["recorded_failures"])
            # The aggregate validator must notice a changed candidate score even
            # if an attacker/bug also updates the superficial file hash.
            path = paths["results"] / "predictions/fp16/hellaswag.jsonl"
            contents = path.read_text(encoding="utf-8").splitlines()
            record = json.loads(contents[0])
            record["log_likelihoods"] = [-9., -1., -3., -4.]
            contents[0] = json.dumps(record)
            path.write_text("\n".join(contents) + "\n", encoding="utf-8")
            manifest_path = paths["results"] / "manifests/run.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["required_artifacts"]["predictions/fp16/hellaswag.jsonl"] = sha256_file(path)
            atomic_json(manifest_path, manifest)
            with self.assertRaises(ValueError):
                validate_replay(configuration, paths)


if __name__ == "__main__":
    unittest.main()

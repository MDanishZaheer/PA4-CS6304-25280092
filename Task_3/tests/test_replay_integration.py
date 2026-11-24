# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Use explicitly synthetic temporary evidence to test the complete CPU-only replay contract."""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from Task_3.analysis.evidence import build_evidence, required_stages
from Task_3.common.artifacts import ArtifactStore, atomic_jsonl, read_json, read_jsonl, sha256_file
from Task_3.common.config import bank_names, draft_ids, fingerprint, load_configuration, protocol_id, trained_variants
from Task_3.common.paths import resolve_paths
from Task_3.common.replay import validate_replay
from Task_3.data.prepare import read_gzip_records, verify_supplied_inputs, write_gzip_records
from Task_3.data.prefixes import selected_positions
from Task_3.evaluation.banks import position_bin
from Task_3.evaluation.quality import score_answer
from Task_3.sampling.markov import validate_markov


def fake_generation(length, elapsed=.2, cap=64, ignore=False):
    """Fabricate marked test metadata, never measured assignment results."""
    tokens = [1] * length
    if not ignore:
        tokens[-1] = 151645
    return {"synthetic_test_fixture": True, "token_ids": tokens, "new_tokens": length, "eos": not ignore,
        "truncated": ignore, "ignore_eos": ignore, "generation_seed": 6304, "prefill_seconds": .01,
        "decode_seconds": elapsed - .01, "elapsed_seconds": elapsed, "peak_allocated_bytes": 1000000,
        "peak_reserved_bytes": 2000000, "target_prefill_passes": 1, "target_decode_passes": length - 1,
        "target_passes": length, "draft_forward_passes": 0, "rounds": []}


def fake_speculation(prompt_length, length, gamma, ignore=False):
    """Construct internally consistent fake counters for all acceptance/bonus/cap paths."""
    rounds, output = [], []
    while len(output) < length:
        m = min(gamma, length - len(output))
        reject = len(rounds) % 2 == 1 and m > 1
        accepted = 1 if reject else m
        examined = accepted + 1 if reject else m
        proposed = [1] * m
        bonus = not reject and len(output) + m < length
        emitted = [1] * accepted + ([2] if reject else [3] if bonus else [])
        if len(output) + len(emitted) == length and not ignore:
            emitted[-1] = 151645
            if not reject and not bonus:
                proposed[-1] = 151645
        row = {"round": len(rounds), "output_offset": len(output), "proposed_tokens": proposed, "emitted_tokens": emitted,
            "proposed": m, "examined": examined, "accepted": accepted, "accepted_prefix_length": accepted,
            "bonus": bonus, "rejected": reject, "target_verification_passes": 1,
            "target_verification_input_tokens": m + int(bool(rounds)), "draft_forward_passes": m + int(bool(rounds)),
            "proposal_seconds": .003, "verification_seconds": .004, "sampling_seconds": .001, "maintenance_seconds": .001,
            "cache_length": prompt_length + len(output) + len(emitted) - 1, "pending_token": emitted[-1]}
        rounds.append(row)
        output.extend(emitted)
    result = fake_generation(length, .01 + .01 * len(rounds), ignore=ignore)
    result.update(token_ids=output, rounds=rounds, gamma=gamma)
    for field in ["accepted", "examined", "proposed", "draft_forward_passes", "proposal_seconds", "verification_seconds", "sampling_seconds", "maintenance_seconds"]:
        result[field] = sum(row[field] for row in rounds)
    result.update(accepted_examined_fraction=result["accepted"] / result["examined"],
        accepted_proposed_fraction=result["accepted"] / result["proposed"], mean_accepted_prefix=result["accepted"] / len(rounds),
        target_verification_passes=len(rounds), target_passes=len(rounds) + 1,
        target_passes_per_output_token=(len(rounds) + 1) / length, host_overhead_seconds=.001 * len(rounds),
        final_cache_length=prompt_length + length - 1, final_pending_token=output[-1])
    return result


def fake_measure(value):
    return {"samples_ms": [value] * 10, "median_ms": value, "warmup": 3, "repetitions": 10,
            "peak_allocated_bytes": 1000000, "peak_reserved_bytes": 2000000, "synthetic_test_fixture": True}


def synthetic_fixture(store):
    """Write a full-sized but clearly fake protocol exclusively in a TemporaryDirectory."""
    configuration, resources = store.configuration, store.paths["resources"]
    train, test = [], []
    for split, collection, count in [("train", train, 500), ("test", test, 60)]:
        ids = read_jsonl(resources / f"data/public_manifests/gsm8k_{split}_ids.jsonl")
        for i in range(count):
            collection.append({"id": f"GSM8K-{split}-{i:04d}", "text": f"SYNTHETIC TEST ONLY {split} math {i}",
                "answer": "#### 7", "domain": "math", "workload": "math", "family": None,
                "source_row_index": ids[i]["row_index"], "prompt_token_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                "prompt_sha256": fingerprint({"synthetic_test_fixture": True, "split": split, "i": i})})
        for source in read_jsonl(resources / f"data/short_form/short_{split}.jsonl"):
            collection.append({"id": source["id"], "text": source["input"], "answer": source["answer"], "domain": "short",
                "workload": source["split"], "family": source["family"], "prompt_token_ids": [1, 2, 3, 4, 5, 6],
                "prompt_sha256": fingerprint({"synthetic_test_fixture": True, "id": source["id"]})})
    for split, collection in [("train", train), ("test", test)]:
        write_gzip_records(store.path(f"splits/{split}_prompts.jsonl.gz"), collection)
    data = {"supplied_hashes": verify_supplied_inputs(resources), "train_prompts": 800, "test_prompts": 120,
            "train_ids": [x["id"] for x in train], "test_ids": [x["id"] for x in test], "synthetic_test_fixture": True}
    store.save_json("splits/data_manifest.json", data)
    store.save_json("manifests/run_lock.json", {"task": "Task_3", "run_id": configuration["execution"]["run_id"],
        "config_id": protocol_id(configuration, store.paths["task"]), "data_id": fingerprint(data),
        "runtime": {"gpu": "SYNTHETIC TEST ONLY", "synthetic_test_fixture": True}, "synthetic_test_fixture": True})
    store.save_json("metrics/sampler_markov.json", validate_markov())
    store.save_json("metrics/cache_checks.json", {"status": "passed", "synthetic_test_fixture": True})
    ledger = []
    producer = {"teacher": configuration["models"]["teacher_revision"], "original": configuration["models"]["student_revision"]}
    for seed in [6304, 6305]:
        for prompt in train:
            row = fake_generation(3, .03)
            row.update(source_id=prompt["id"], prompt_sha256=prompt["prompt_sha256"])
            store.save_json(f"rollouts/teacher_pool_seed{seed}/{prompt['id']}.json", row)
        store.save_json(f"metrics/teacher_pool_seed{seed}.json", {"seed": seed, "prompts": 800, "generation_seconds": 24.})
        ledger.append({"stage": f"pool_seed{seed}", "seconds": 25., "completed_attempt": True, "synthetic_test_fixture": True})
    for variant in trained_variants(configuration):
        name = variant["id"]
        producer[name] = fingerprint({"synthetic_adapter": name})
        history, annotations, counts = [], [], {"math": 0, "short": 0}
        for update in range(1, 158):
            budget = 256 if update < 157 else 64
            rows = []
            while budget:
                domain = "math" if counts["math"] < 30000 else "short"
                quota = 30000 if domain == "math" else 10000
                amount = min(budget, configuration["training"]["response_caps"][domain], quota - counts[domain])
                source = train[0] if domain == "math" else train[500]
                rows.append({"source_id": source["id"], "prompt_sha256": source["prompt_sha256"], "domain": domain,
                    "response": [1] * amount, "first_scored_response_index": 0, "teacher_scored_tokens": amount,
                    "rollout_seed": 6304, "producer_checkpoint": producer[name], "producer_update": update - 1,
                    "presentation": len(rows), "quota_truncated": False, "termination": "length_cap", "synthetic_test_fixture": True})
                budget -= amount
                counts[domain] += amount
            path = f"rollouts/training/{name}/update_{update:04d}.jsonl.gz"
            write_gzip_records(store.path(path), rows)
            annotations.append(path)
            history.append({**variant, "update": update, "teacher_scored_tokens": 256 if update < 157 else 64,
                "cumulative_tokens": sum(counts.values()), "cumulative_math_tokens": counts["math"], "cumulative_short_tokens": counts["short"],
                "forward_kl": 1 / update, "lr": .0003, "gradient_norm_before_clip": 1., "overflow_attempts": 0,
                "student_rollout_seconds": .01, "teacher_scoring_seconds": .01, "student_backward_and_optimizer_seconds": .01})
        history_path = f"histories/{name}.jsonl"
        atomic_jsonl(store.path(history_path), history)
        store.save_json(f"metrics/training/{name}.json", {**variant, "status": "complete", "teacher_scored_tokens": counts,
            "successful_optimizer_updates": 157, "initialization_sha256": fingerprint({"seed": variant["seed"]}),
            "adapter_sha256": producer[name], "overflow_attempts": 0, "trainable_parameters": 1081344, "total_parameters": 495114112,
            "history": history_path, "annotations": annotations, "teacher_pool_generation_seconds": 24. if variant["method"] == "off_policy" else 0.,
            "student_rollout_seconds": 1.57, "teacher_scoring_seconds": 1.57, "student_backward_and_optimizer_seconds": 1.57,
            "synthetic_test_fixture": True})
        store.save_json(f"metrics/merging/{name}.json", {"draft": name, "status": "complete", "fresh_fp16_backbone": True,
            "wrapper_in_primary_timing": False, "numerical_checks": [{"relative_l2": .0001}], "merge_compute_seconds": 1.,
            "merge_write_and_backup_seconds": 2., "synthetic_test_fixture": True})
        ledger.extend([{"stage": f"train_{name}", "seconds": 10., "synthetic_test_fixture": True},
                       {"stage": f"merge_{name}", "seconds": 3., "synthetic_test_fixture": True}])
    store.save_records("metrics/stage_wall_costs.jsonl", ledger)
    for bank_index, bank in enumerate(bank_names(configuration)):
        producing = "teacher" if bank == "teacher" else "original" if bank in {"original_draft", "exact_spec_original_gamma4"} else bank.removeprefix("own_")
        for prompt_index, prompt in enumerate(test):
            length = 80 if prompt["domain"] == "math" else 8
            selected = selected_positions(length, prompt["id"], bank, configuration)
            ids = [f"{bank}:{prompt['id']}:p{position:03d}" for position in selected]
            bank_path = f"banks/{bank}/{prompt['id']}.json"
            store.save_json(bank_path, {"prompt_id": prompt["id"], "prompt_sha256": prompt["prompt_sha256"], "bank": bank,
                "workload": prompt["workload"], "producer": producing, "producer_checkpoint": producer[producing],
                "response_token_ids": [1] * length, "selected_positions": selected, "prefix_ids": ids, "synthetic_test_fixture": True})
            bank_sha = sha256_file(store.path(bank_path))
            for name_index, name in enumerate(draft_ids(configuration)):
                tv = .1 + .01 * name_index + .001 * bank_index + .00001 * prompt_index
                rows = [{"prefix_id": prefix_id, "prompt_id": prompt["id"], "prompt_sha256": prompt["prompt_sha256"],
                    "bank": bank, "draft": name, "workload": prompt["workload"], "position": position, "position_bin": position_bin(position),
                    "producer_checkpoint": producer[producing], "prefix_bank_sha256": bank_sha,
                    "forward_kl": tv * 2, "reverse_kl": tv * 3, "tv": tv, "expected_acceptance": 1 - tv,
                    "greedy_agreement": .75, "target_next_token": 1, "student_next_token": 2, "synthetic_test_fixture": True}
                    for prefix_id, position in zip(ids, selected)]
                write_gzip_records(store.path(f"alignment/{bank}/{name}/{prompt['id']}.jsonl.gz"), rows)
    for wrapper in [False, True]:
        for name in (["off_policy_seed6304"] if wrapper else draft_ids(configuration)):
            for length in [64, 256, 512]:
                store.save_json(f"systems/{'unmerged_' if wrapper else ''}{name}_L{length}.json", {
                    "draft": name, "context_length": length, "unmerged_wrapper": wrapper, "status": "complete", "c": .4,
                    "target_single_step": fake_measure(1.), "draft_single_step": fake_measure(.4),
                    "verification": {str(gamma): {**fake_measure(1.2), "v_gamma": 1.2, "input_tokens": gamma + 1} for gamma in [2, 4]},
                    "synthetic_test_fixture": True})
    for suite, names, prompts in [("baseline", ["original"], test), ("primary", draft_ids(configuration), test),
                                 ("fixed64", draft_ids(configuration), test[:24])]:
        for index, name in enumerate(names):
            ratio = .8 + .1 * index
            for gamma in [2, 4]:
                for repetition in [0, 1]:
                    for prompt in prompts:
                        length = 64 if suite == "fixed64" else 32 if prompt["domain"] == "math" else 8
                        spec = fake_speculation(len(prompt["prompt_token_ids"]), length, gamma, suite == "fixed64")
                        target = fake_generation(length, spec["elapsed_seconds"] * ratio, ignore=suite == "fixed64")
                        store.save_json(f"benchmarks/{suite}/{name}/gamma{gamma}/rep{repetition}/{prompt['id']}.json", {
                            "prompt_id": prompt["id"], "prompt_sha256": prompt["prompt_sha256"], "prompt_tokens": len(prompt["prompt_token_ids"]),
                            "draft": name, "gamma": gamma, "repetition": repetition, "workload": prompt["workload"], "suite": suite,
                            "target": target, "speculative": spec, "adjacent_target_reference": True, "target_first": True,
                            "target_text": "SYNTHETIC TARGET", "speculative_text": "SYNTHETIC SPECULATION",
                            "end_to_end_tps_ratio": ratio, "steady_decode_tps_ratio": (spec["new_tokens"] / spec["decode_seconds"]) / (target["new_tokens"] / target["decode_seconds"]),
                            "request_latency_ratio": ratio, "synthetic_test_fixture": True})
    for name in ["target", *draft_ids(configuration)]:
        for repetition in [0, 1]:
            for prompt in test:
                text = '7' if prompt["domain"] == "math" else prompt["answer"]
                store.save_json(f"quality/{name}/rep{repetition}/{prompt['id']}.json", {"draft": name, "repetition": repetition,
                    "prompt_id": prompt["id"], "prompt_sha256": prompt["prompt_sha256"], "workload": prompt["workload"],
                    "response": fake_generation(3), "text": text, **score_answer(prompt, text), "synthetic_test_fixture": True})
    for stage in required_stages(configuration):
        store.complete(stage, [], {"synthetic_test_fixture": True})


class ReplayIntegrationTests(unittest.TestCase):
    def test_full_protocol_evidence_and_notebook_without_model_imports(self):
        task = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            shutil.copytree(task, project / "Task_3", ignore=shutil.ignore_patterns("results", "__pycache__"))
            configuration = load_configuration(project / "Task_3")
            paths = resolve_paths(configuration, project)
            store = ArtifactStore(paths, configuration)
            synthetic_fixture(store)
            self.assertEqual(build_evidence(store)["figures"], 9)
            self.assertEqual(validate_replay(configuration, paths)["status"], "complete")
            preview = os.environ.get("ATML_TASK3_QA_DIR")
            if preview:
                destination = Path(preview).resolve()
                self.assertTrue(destination.is_relative_to(task.parent / "tmp"))
                shutil.copytree(store.path("figures"), destination, dirs_exist_ok=True)
                shutil.copytree(store.path("tables"), destination / "tables", dirs_exist_ok=True)
            script = '\n'.join([
                'import importlib.abc,sys',
                'class BlockCompute(importlib.abc.MetaPathFinder):',
                '    def find_spec(self,fullname,path=None,target=None):',
                "        if fullname.split('.')[0] in {'torch','transformers','accelerate','huggingface_hub','datasets'}:",
                "            raise AssertionError('Replay imported a compute library: '+fullname)",
                'sys.meta_path.insert(0,BlockCompute())',
                'from Task_3.scripts.validate_project import execute_replay_notebook',
                "print(execute_replay_notebook('.',working_directory='Task_3'))"])
            result = subprocess.run([sys.executable, '-B', '-c', script], cwd=project, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(int(result.stdout.strip()), 29)
            # Updating the hash must not conceal a changed scientific outcome.
            relative = 'quality/original/rep0/GSM8K-test-0000.json'
            row = read_json(store.path(relative))
            row['correct'] = not row['correct']
            store.save_json(relative, row)
            manifest = read_json(store.path('manifests/run.json'))
            manifest['required_artifacts'][relative] = sha256_file(store.path(relative))
            store.save_json('manifests/run.json', manifest)
            with self.assertRaises(ValueError):
                validate_replay(configuration, paths, deep=False)


if __name__ == '__main__':
    unittest.main()

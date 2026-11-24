# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Expose the eleven assignment steps with strict identities and verified recovery."""

import gc
import subprocess
import sys
from datetime import datetime, timezone

import torch

from Task_2.cloud.colab import DriveSync
from Task_2.common.artifacts import ArtifactStore, read_json, sha256_file
from Task_2.common.config import fingerprint, protocol_id, scientific_configuration, source_fingerprint
from Task_2.common.runtime import runtime_metadata, seed_everything
from Task_2.data.mqar import load_evaluation_stream, load_generator, validate_example, verify_supplied_inputs
from Task_2.training.protocol import evaluation_conditions, training_conditions


class ComputeSession:
    """Run each required stage independently while preserving a single frozen protocol."""

    def __init__(self, configuration, paths):
        self.configuration, self.paths = configuration, paths
        self.sync = DriveSync(paths["results"], paths["drive_results"]) if configuration["execution"]["sync_to_drive"] else None
        if self.sync:
            self.sync.restore()
        self.store = ArtifactStore(paths, configuration, self.sync)
        seed_everything(configuration["seed"])
        self.runtime = runtime_metadata()
        identity = protocol_id(configuration, paths["task"])
        lock_path = self.store.path("manifests/run_lock.json")
        if lock_path.is_file():
            self.lock = read_json(lock_path)
            if self.lock["config_id"] != identity:
                raise ValueError("Scientific code/configuration changed. Restore it or use a new run_id; saved results cannot be mixed.")
            for key in ["gpu", "packages", "cuda", "compute_capability"]:
                if self.lock["runtime"][key] != self.runtime[key]:
                    raise ValueError(f"Resume runtime {key} differs. Use the original GPU/environment for matched comparisons.")
        else:
            self.lock = {"task": "Task_2", "run_id": configuration["execution"]["run_id"],
                         "config_id": identity, "source_id": source_fingerprint(paths["task"]),
                         "scientific_configuration": scientific_configuration(configuration), "runtime": self.runtime,
                         "created_utc": datetime.now(timezone.utc).isoformat(), "data_id": None,
                         "intervention_id": None, "mqar_final_test_started": False}
            self.store.save_json("manifests/run_lock.json", self.lock)
        self.store.save_json("configs/resolved_configuration.json", configuration)
        freeze = self.store.path("configs/environment_freeze.txt")
        if not freeze.is_file():
            freeze.write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True,
                                             capture_output=True, text=True).stdout, encoding="utf-8")
            self.store.backup(["configs/environment_freeze.txt"])
        self.development = None
        self.model = self.tokenizer = self.retrieval_cases = None
        print(f"Task 2 compute | {self.runtime['gpu']} | {self.lock['run_id']}", flush=True)

    def prepare_data(self):
        """Validate the original assets and development stream without scoring final tests."""
        hashes = verify_supplied_inputs(self.paths["resources"])
        self.development = load_evaluation_stream(self.paths["resources"], "development")
        generator = load_generator(self.paths["resources"])
        for count in [4, 8, 16]:
            for regime in ["near", "far", "mixed"]:
                for index in range(32):
                    validate_example(generator.make_example("train", count, regime, index), generator)
        # Validate the real tokenizer/template and all cache budgets before the
        # long training section. This downloads no pretrained model weights.
        from transformers import AutoConfig
        from Task_2.cache.packing import payload_bytes
        from Task_2.cache.selection import selection_plan
        from Task_2.data.retrieval import prepare_retrieval
        self.retrieval_cases, revision = prepare_retrieval(self.store)
        model_configuration = AutoConfig.from_pretrained(self.configuration["kv"]["model"],
            revision=revision["revision"], cache_dir=self.paths["data"] / "huggingface")
        expected_geometry = {"layers": model_configuration.num_hidden_layers,
            "kv_heads": model_configuration.num_key_value_heads,
            "head_dimension": getattr(model_configuration, "head_dim", None) or model_configuration.hidden_size // model_configuration.num_attention_heads}
        for case in self.retrieval_cases:
            geometry = (expected_geometry["layers"], 1, expected_geometry["kv_heads"], expected_geometry["head_dimension"], case["prefix_length"])
            budget = 32 + payload_bytes(*geometry, 4)
            scores = torch.linspace(0, 1, case["prefix_length"])
            for method in self.configuration["kv"]["methods"][1:]:
                if selection_plan(method, geometry, budget, scores)["expected_bytes"] > budget:
                    raise ValueError("Retrieval preflight found an infeasible matched cache budget.")
        data = {"supplied_hashes": hashes, "generator_master_seed": 6304,
                "mqar_development_examples": len(self.development), "mqar_final_test_examples": 9216,
                "retrieval_examples": 180, "generator_training_checks": 288,
                "tokenized_retrieval_sha256": sha256_file(self.store.path("splits/retrieval_tokenized.jsonl.gz")),
                "expected_kv_geometry": expected_geometry, "cache_budget_preflight_conditions": 900,
                "final_test_policy": "do not score final MQAR tests before intervention lock and all training runs"}
        data_id = fingerprint(data)
        if self.lock["data_id"] not in {None, data_id}:
            raise ValueError("Task 2 assets differ from the saved data identity.")
        self.lock["data_id"] = data_id
        self.store.save_json("splits/data_manifest.json", data)
        self.store.save_json("manifests/run_lock.json", self.lock)
        self.store.complete("data", ["splits/data_manifest.json", "splits/kv_model_revision.json",
            "splits/retrieval_tokenized.jsonl.gz", "splits/retrieval_selection.json"], data)
        return data

    def run_attention(self, method):
        """Run or reuse one of A1 dense, A2 tiled, or A3 optimized SDPA."""
        from Task_2.attention.benchmarks import run_part_a
        if method not in self.configuration["attention"]["methods"]:
            raise ValueError("Unknown Part A method.")
        if self.store.completed(f"attention_{method}"):
            if not self.store.completed("part_a") and all(self.store.completed(f"attention_{name}") for name in self.configuration["attention"]["methods"]):
                return run_part_a(self.store, method)
            return read_json(self.store.path(f"metrics/attention/{method}_conditions.json"))
        return run_part_a(self.store, method)

    def train_group(self, methods):
        """Train each listed architecture from its independent prescribed initialization."""
        from Task_2.training.recall import train_condition
        if self.development is None:
            self.prepare_data()
        results = []
        content_k = read_json(self.store.path("configs/intervention_locked.json"))["content_k"] if "local_content" in methods else 64
        for condition in training_conditions(self.configuration):
            if condition["method"] in methods:
                results.append(train_condition(self.store, condition, self.development, content_k))
        return results

    def freeze_intervention(self):
        """Record a development-motivated prediction before any final-test evaluation."""
        from Task_2.training.recall import lock_intervention
        for condition in training_conditions(self.configuration, include_intervention=False):
            if not self.store.completed(f"train_{condition['id']}"):
                raise RuntimeError("Finish every baseline seed before selecting the intervention.")
        if self.lock["mqar_final_test_started"] and not self.store.path("configs/intervention_locked.json").is_file():
            raise RuntimeError("Intervention selection cannot restart after final-test access.")
        intervention = lock_intervention(self.store, training_conditions(self.configuration))
        identity = fingerprint(intervention)
        if self.lock["intervention_id"] not in {None, identity}:
            raise ValueError("The pre-test intervention changed.")
        self.lock["intervention_id"] = identity
        self.store.save_json("manifests/run_lock.json", self.lock)
        self.store.complete("intervention", ["configs/intervention_locked.json"], {"intervention_id": identity})
        return intervention

    def evaluate_mqar(self):
        """Read final test labels only after all 42 training runs and the selection lock."""
        from Task_2.evaluation.recall import evaluate_final_condition
        from Task_2.models.mqar import RecallTransformer
        for condition in training_conditions(self.configuration):
            if not self.store.completed(f"train_{condition['id']}"):
                raise RuntimeError("Complete every required training attempt before final-test scoring.")
        intervention = read_json(self.store.path("configs/intervention_locked.json"))
        if fingerprint(intervention) != self.lock["intervention_id"]:
            raise ValueError("Intervention differs from its pre-test lock.")
        self.lock["mqar_final_test_started"] = True
        self.store.save_json("manifests/run_lock.json", self.lock)
        test = load_evaluation_stream(self.paths["resources"], "test")
        results, artifacts = [], []
        for condition in evaluation_conditions(self.configuration):
            summary_relative = f"metrics/mqar_test/{condition['id']}.json"
            if self.store.completed(f"mqar_test_{condition['id']}"):
                results.append(read_json(self.store.path(summary_relative)))
                artifacts.append(summary_relative)
                continue
            training = read_json(self.store.path(f"metrics/mqar_training/{condition['training_id']}.json"))
            if training["status"] != "complete":
                summary = {**condition, "status": "unavailable_training_failed", "training_status": training["status"], "cells": [], "examples": 0}
                self.store.save_json(summary_relative, summary)
                self.store.complete(f"mqar_test_{condition['id']}", [summary_relative], {"status": summary["status"]})
            else:
                model = checkpoint = None
                try:
                    checkpoint = torch.load(self.store.path(training["final_checkpoint"]), map_location="cuda", weights_only=True)
                    if checkpoint["config_id"] != self.lock["config_id"] or checkpoint["update"] != self.configuration["mqar"]["updates"]:
                        raise ValueError("MQAR evaluation checkpoint does not match the final prescribed update.")
                    model = RecallTransformer(self.configuration["mqar"], condition["method"], training["content_k"]).cuda()
                    model.load_state_dict(checkpoint["model"])
                    summary = evaluate_final_condition(self.store, model, test, condition)
                except (torch.OutOfMemoryError, FloatingPointError) as error:
                    summary = {**condition, "status": "failed_cuda_oom" if isinstance(error, torch.OutOfMemoryError) else "failed_nonfinite_evaluation",
                               "error": str(error), "training_status": training["status"], "cells": [], "examples": 0,
                               "partial_predictions": "Previously committed shards retained; no incomplete cell reported as a final metric."}
                    self.store.save_json(summary_relative, summary)
                    self.store.complete(f"mqar_test_{condition['id']}", [summary_relative], {"status": summary["status"]})
                finally:
                    model = checkpoint = None
                    gc.collect()
                    torch.cuda.empty_cache()
            results.append(summary)
            artifacts.append(summary_relative)
        self.store.save_json("metrics/mqar_test/all_conditions.json", results)
        artifacts.append("metrics/mqar_test/all_conditions.json")
        self.store.complete("mqar_final", artifacts, {"conditions": len(results)})
        return {"conditions": len(results), "training_runs": 42, "test_cells_per_completed_condition": 9}

    def run_mqar_inference(self):
        """Separate fixed-shape attention cost from learned recall outcomes."""
        from Task_2.attention.benchmarks import run_inference_sweep
        if self.store.completed("mqar_inference"):
            return read_json(self.store.path("metrics/mqar_inference/all_conditions.json"))
        width = read_json(self.store.path("configs/intervention_locked.json"))["content_k"]
        return run_inference_sweep(self.store, width)

    def ensure_kv_model(self):
        """Load the immutable instruction-tuned FP16 model only for Part C."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from Task_2.data.retrieval import prepare_retrieval
        if self.model is None:
            self.retrieval_cases, revision = prepare_retrieval(self.store)
            settings = self.configuration["kv"]
            seed_everything(self.configuration["seed"])
            cache_directory = self.paths["data"] / "huggingface"
            self.tokenizer = AutoTokenizer.from_pretrained(settings["model"], revision=revision["revision"], cache_dir=cache_directory, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(settings["model"], revision=revision["revision"],
                cache_dir=cache_directory, torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="sdpa")
            self.model.cuda().eval().requires_grad_(False)
            from Task_2.attention.benchmarks import observed_backend
            from Task_2.cache.adapter import clone_dynamic_cache, model_step
            probe = torch.full((1, 32), 32, device="cuda", dtype=torch.long)
            prefill_operators = observed_backend(lambda: model_step(self.model, probe))
            _, probe_cache = model_step(self.model, probe)
            decode_operators = observed_backend(lambda: model_step(self.model, probe[:, :1], clone_dynamic_cache(probe_cache), 32))
            self.store.save_json("metrics/kv/model_geometry.json", {
                "layers": self.model.config.num_hidden_layers, "kv_heads": self.model.config.num_key_value_heads,
                "query_heads": self.model.config.num_attention_heads,
                "head_dimension": self.model.model.layers[0].self_attn.head_dim,
                "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
                "model": settings["model"], "revision": revision["revision"], "attention_backend": "sdpa",
                "eos_token_ids": self.model.generation_config.eos_token_id,
                "sdpa_observed_prefill_operators": prefill_operators, "sdpa_observed_decode_operators": decode_operators})
        return self.model

    def validate_kv_reference(self):
        """Run C1 direct/split and 100%-retention controls before comparing policies."""
        from Task_2.evaluation.kv import validate_full_cache
        if self.store.completed("kv_reference"):
            return read_json(self.store.path("metrics/kv/reference_checks.json"))
        self.ensure_kv_model()
        result = validate_full_cache(self.model, self.retrieval_cases, self.configuration["kv"])
        self.store.save_json("metrics/kv/reference_checks.json", result)
        self.store.complete("kv_reference", ["metrics/kv/reference_checks.json", "metrics/kv/model_geometry.json",
            "splits/kv_model_revision.json", "splits/retrieval_tokenized.jsonl.gz", "splits/retrieval_selection.json"], result)
        return result

    def check_cache_policies(self, family):
        """Validate C2/C3/C4 byte budgets for every actual context length before C5."""
        from Task_2.cache.adapter import BudgetCache
        from Task_2.cache.packing import payload_bytes
        from Task_2.cache.selection import selection_plan
        methods = {"eviction": ["sink_recent", "question_aware"], "quantization": ["int4_token", "int4_channel"], "hybrid": ["hybrid"]}[family]
        if self.store.completed(f"kv_policy_{family}"):
            return read_json(self.store.path(f"metrics/kv/policy_{family}.json"))
        self.ensure_kv_model()
        geometry = read_json(self.store.path("metrics/kv/model_geometry.json"))
        rows = []
        for case in self.retrieval_cases:
            shape = (geometry["layers"], 1, geometry["kv_heads"], geometry["head_dimension"], case["prefix_length"])
            budget = 32 + payload_bytes(*shape, 4)
            scores = torch.linspace(0, 1, case["prefix_length"])
            for method in methods:
                plan = selection_plan(method, shape, budget, scores)
                if plan["expected_bytes"] > budget:
                    raise ValueError("A proposed cache layout exceeds its actual INT4 reference bytes.")
                rows.append({"id": case["id"], "method": method, "actual_prefix_tokens": case["prefix_length"],
                             "budget_bytes": budget, "planned_bytes": plan["expected_bytes"],
                             "selected_fp16_positions": len(plan["positions"]),
                             "validation_scores": "synthetic monotone scores for allocation validation only; actual question scores used in C5"})
        result = {"status": "passed", "family": family, "conditions": rows,
                  "quality_evaluation": "all six conditions run together in C5 to reuse a shared original prefill"}
        relative = f"metrics/kv/policy_{family}.json"
        self.store.save_json(relative, result)
        self.store.complete(f"kv_policy_{family}", [relative], {"cases": 180, "methods": methods})
        return {"status": "passed", "family": family, "cases": 180, "methods": methods}

    def run_kv_quality(self):
        """Execute C5's paired fidelity, answer-generation and qualitative evidence."""
        from Task_2.evaluation.kv import evaluate_retrieval
        if self.store.completed("kv_quality"):
            return read_json(self.store.path("manifests/kv_quality.json"))["summary"]
        for prerequisite in ["kv_reference", "kv_policy_eviction", "kv_policy_quantization", "kv_policy_hybrid"]:
            if not self.store.completed(prerequisite):
                raise RuntimeError(f"Finish {prerequisite} before cache quality evaluation.")
        self.ensure_kv_model()
        return evaluate_retrieval(self.store, self.model, self.tokenizer, self.retrieval_cases)

    def run_kv_systems(self):
        """Measure real packing and full-model decode at all required batch/length shapes."""
        from Task_2.evaluation.kv_systems import run_cache_systems
        if self.store.completed("kv_systems"):
            return read_json(self.store.path("metrics/kv_systems/all_conditions.json"))
        self.ensure_kv_model()
        return run_cache_systems(self.store, self.model)

    def build_evidence(self):
        """Require complete attempts before creating report tables and canonical figures."""
        from Task_2.analysis.evidence import build_evidence
        return build_evidence(self.store)

    def finish(self):
        """Validate the saved counts and checksums before printing the download location."""
        from Task_2.common.replay import validate_replay
        manifest = validate_replay(self.configuration, self.paths)
        print(f"Download results from {self.paths['drive_results'].parent}", flush=True)
        return {"status": manifest["status"], "computed_on": "Google Colab", "run_id": self.lock["run_id"],
                "results_to_download": str(self.paths["drive_results"].parent)}

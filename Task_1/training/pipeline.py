# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Resumable Colab compression/evaluation pipeline with pre-test allocation locks."""

import gc
import importlib.metadata
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import torch

from Task_1.cloud.colab import DriveSync
from Task_1.common.artifacts import ArtifactStore, atomic_json, read_json, sha256_file
from Task_1.common.config import fingerprint, scientific_configuration, source_fingerprint
from Task_1.data.prepare import TASK_FILES, load_saved_task_data, prepare_task_data
from Task_1.evaluation.resources import benchmark_condition, storage_accounting
from Task_1.evaluation.scoring import (development_nll, evaluate_hellaswag,
                                       evaluate_perplexity, sequence_nll)
from Task_1.evaluation.workload import configure_gpu_workload
from Task_1.methods.allocation import allocate_precision, allocate_sparsity
from Task_1.methods.calibration import (collect_activations, compressed_layer_errors,
                                        profile_projections)
from Task_1.models.compression import (CompressionState, load_frozen_model,
                                      parameter_inventory)
from Task_1.training.protocol import experiment_conditions


def runtime_metadata(configuration):
    """Capture exact software and accelerator identity for matched execution."""
    packages = {}
    for name in ["torch", "transformers", "huggingface_hub", "datasets", "accelerate", "numpy", "safetensors"]:
        packages[name] = importlib.metadata.version(name)
    if packages["transformers"] != "4.57.1" or packages["huggingface_hub"] != "0.36.2":
        raise RuntimeError("Install the supplied compute requirements before starting the run.")
    if not torch.cuda.is_available():
        raise RuntimeError("Task 1 compute requires a CUDA GPU. Local notebooks default to replay.")
    properties = torch.cuda.get_device_properties(0)
    return {"computed_on": "Google Colab", "python": platform.python_version(), "packages": packages,
            "cuda": torch.version.cuda, "gpu": properties.name, "gpu_memory_bytes": properties.total_memory,
            "compute_capability": list(torch.cuda.get_device_capability()),
            "precision": "float16 weights/activations; float32 error/scoring arithmetic; float64 NLL accumulation",
            "attention_implementation": configuration["model"]["attention_backend"],
            "tf32": False, "seed": configuration["seed"]}


class ComputeSession:
    """Coordinate task steps while retaining immutable, recoverable experiment state."""

    def __init__(self, configuration, paths):
        self.configuration = configuration
        self.paths = paths
        self.sync = DriveSync(paths["results"], paths["drive_results"]) if configuration["execution"]["sync_to_drive"] else None
        if self.sync:
            self.sync.restore()
        self.store = ArtifactStore(paths, configuration, self.sync)
        self.runtime = runtime_metadata(configuration)
        self.protocol_id = fingerprint({"scientific_configuration": scientific_configuration(configuration),
                                        "source_id": source_fingerprint(paths["task"]),
                                        "conditions": experiment_conditions()})
        self.lock_path = self.store.path("manifests/run_lock.json")
        if self.lock_path.is_file():
            self.lock = read_json(self.lock_path)
            if self.lock["config_id"] != self.protocol_id:
                raise ValueError("Saved run configuration/code differ. Restore the original code or use a new run_id; results cannot be mixed.")
            for key in ["gpu", "packages", "cuda", "attention_implementation"]:
                if self.lock["runtime"][key] != self.runtime[key]:
                    raise ValueError(f"Runtime {key} changed. Select the same runtime or create a separate run_id for matched comparisons.")
        else:
            self.lock = {"task": "Task_1", "run_id": configuration["execution"]["run_id"],
                         "config_id": self.protocol_id, "source_id": source_fingerprint(paths["task"]),
                         "created_utc": datetime.now(timezone.utc).isoformat(), "runtime": self.runtime,
                         "conditions": experiment_conditions(), "data_id": None,
                         "allocation_lock_id": None, "heldout_access_started": False,
                         "scientific_configuration": scientific_configuration(configuration)}
            self.store.save_json("manifests/run_lock.json", self.lock)
        self.store.save_json("configs/resolved_configuration.json", configuration)
        environment = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout
        freeze_path = self.store.path("configs/environment_freeze.txt")
        if not freeze_path.exists():
            freeze_path.write_text(environment, encoding="utf-8")
            self.store.backup(["configs/environment_freeze.txt"])
        self.model = self.tokenizer = self.state = None
        self.records = self.selection = self.statistics = self.profiles = None
        self.plans = None
        print(f"Task 1 compute | {self.runtime['gpu']} | run {configuration['execution']['run_id']}", flush=True)

    def prepare_data(self):
        """Prepare or restore the same immutable course data and tokenizer revision."""
        if self.store.path("splits/selection.json").is_file():
            self.records, self.selection = load_saved_task_data(self.store)
        else:
            self.records, self.selection = prepare_task_data(self.store)
        data_id = fingerprint(self.selection)
        if self.lock["data_id"] not in {None, data_id}:
            raise ValueError("Materialized data do not match this run's saved identity.")
        self.lock["data_id"] = data_id
        self.lock["model_revision"] = self.selection["source_revisions"]["revisions"]["model_base"]
        self.store.save_json("manifests/run_lock.json", self.lock)
        artifacts = ["splits/selection.json", "splits/local_source_revisions.json"]
        artifacts += [f"splits/materialized/{filename}" for filename in TASK_FILES.values()]
        self.store.complete("data", artifacts, self.selection["summary"])
        return self.selection["summary"]

    def ensure_model(self):
        """Lazily load one original FP16 model and record its exact parameter inventory."""
        if self.records is None:
            self.prepare_data()
        if self.model is None:
            cache = self.paths["data"] / "huggingface"
            self.model, self.tokenizer = load_frozen_model(self.configuration, self.lock["model_revision"], cache)
            inventory = parameter_inventory(self.model, self.configuration)
            self.state = CompressionState(self.model, self.configuration)
            self.store.save_json("metrics/model_inventory.json", inventory)
            tokens = self.records["calibration"][0]["token_ids"][:32]
            manual, count = sequence_nll(self.model, tokens, chunk_tokens=8)
            input_ids = torch.tensor([tokens], device="cuda")
            with torch.inference_mode():
                reference = float(self.model(input_ids=input_ids, labels=input_ids, use_cache=False).loss)
            if abs(manual / count - reference) > .003:
                raise RuntimeError("Shared NLL scorer disagrees with the checkpoint's standard shifted causal loss.")
            self.store.save_json("metrics/reference_diagnostics.json", {
                "calibration_sequence_index": 0, "tokens": 32, "scored_tokens": count,
                "manual_mean_nll": manual / count, "model_mean_nll": reference,
                "absolute_error": abs(manual / count - reference), "excluded_parameters_sha256": self.state.excluded_digest,
                "projection_count": len(inventory["projections"]), "status": "passed"})
            configure_gpu_workload(self.model, self.records["calibration"], self.configuration, self.store)
        return self.model

    def prepare_calibration_and_plan(self):
        """Lock every adaptive decision before baseline or compressed held-out scoring."""
        if self.store.completed("allocation_plan") and all(self.store.completed(condition["id"]) for condition in experiment_conditions()):
            self.plans = read_json(self.store.path("configs/allocations_locked.json"))
            return {"status": "all saved conditions complete; no model loading or recalibration needed"}
        self.ensure_model()
        self.state.restore()
        self.statistics, _ = collect_activations(self.model, self.records["calibration"], self.configuration, self.store)
        self.profiles = profile_projections(self.model, self.statistics, self.configuration, self.store)
        equivalence_relative = "metrics/awq_full_model_rescaling.json"
        if not self.store.path(equivalence_relative).is_file():
            inputs = torch.tensor([self.records["calibration"][0]["token_ids"][:32]], device="cuda")
            self.state.restore()
            with torch.inference_mode():
                reference = self.model(input_ids=inputs, use_cache=False).logits.float()
                equivalence = {}
                for bits in self.configuration["quantization"]["allowed_bits"]:
                    self.state.apply_full_precision_rescaling(self.profiles, bits)
                    adjusted = self.model(input_ids=inputs, use_cache=False).logits.float()
                    relative = float((adjusted - reference).norm() / reference.norm().clamp(min=1e-12))
                    equivalence[str(bits)] = {"relative_l2": relative, "maximum_absolute": float((adjusted - reference).abs().max()),
                                             "dtype": "FP16 execution before rounding", "tokens": 32, "calibration_sequence": 0}
                    if not torch.isfinite(adjusted).all() or relative > .02:
                        raise RuntimeError(f"Unquantized AWQ graph failed full-model equivalence for scale profile INT{bits}: {relative}")
                self.state.restore()
            self.store.save_json(equivalence_relative, equivalence)
        if self.store.completed("allocation_plan"):
            self.plans = read_json(self.store.path("configs/allocations_locked.json"))
            if fingerprint(self.plans) != self.lock["allocation_lock_id"]:
                raise ValueError("Frozen allocation identity changed.")
            return {"status": "loaded frozen allocations", "allocation_lock_id": self.lock["allocation_lock_id"]}
        if self.lock["heldout_access_started"]:
            raise RuntimeError("Adaptive selection cannot restart after held-out evaluation; restore the frozen allocation backup.")
        plans = {"pruning": {}, "quantization": {}, "selection_data": "128 C4 calibration + 32 separate C4 training development sequences"}
        for target in self.configuration["pruning"]["adaptive_sparsities"]:
            candidates = []
            for temperature in self.configuration["pruning"]["temperature_candidates"]:
                self.store.check_time()
                relative = f"metrics/selection/p3_s{round(target * 100)}_temperature_{temperature}.json"
                allocation = allocate_sparsity(self.profiles, target, temperature)
                if self.store.path(relative).is_file():
                    result = read_json(self.store.path(relative))
                    if result["allocation"] != allocation:
                        raise ValueError("Saved development candidate does not match its deterministic allocation.")
                else:
                    started = time.perf_counter()
                    self.state.apply({"family": "P3", "sparsity": target}, self.statistics, self.profiles, allocation)
                    result = {"allocation": allocation, "development": development_nll(self.model, self.records["development"], self.configuration),
                              "seconds": time.perf_counter() - started}
                    self.store.save_json(relative, result)
                candidates.append(result)
            selected = min(candidates, key=lambda candidate: (candidate["development"]["mean_nll"], candidate["allocation"]["temperature"]))
            plans["pruning"][str(target)] = {**selected["allocation"], "development": selected["development"],
                                              "candidate_results": candidates, "criterion": self.configuration["pruning"]["selection"]}
        inventory = read_json(self.store.path("metrics/model_inventory.json"))
        for reference_bits in [4, 3]:
            plans["quantization"][str(reference_bits)] = allocate_precision(
                self.profiles, reference_bits, inventory["uncompressed_bytes"] + 128, self.configuration)
        self.state.restore()
        self.plans = plans
        self.store.save_json("configs/allocations_locked.json", plans)
        self.lock["allocation_lock_id"] = fingerprint(plans)
        self.store.save_json("manifests/run_lock.json", self.lock)
        self.store.complete("allocation_plan", ["configs/allocations_locked.json", "metrics/sensitivity_profiles.json"],
                            {"allocation_lock_id": self.lock["allocation_lock_id"], "heldout_access_started": False})
        return {"status": "allocations frozen before held-out evaluation", "allocation_lock_id": self.lock["allocation_lock_id"]}

    def allocation_for(self, condition):
        """Read only the already frozen allocation for adaptive conditions."""
        if condition["family"] == "P3":
            return self.plans["pruning"][str(condition["sparsity"])]
        if condition["family"] == "Q3":
            return self.plans["quantization"][str(condition["reference_bits"])]
        return None

    def run_group(self, family):
        """Run the notebook's current required step, preserving independent starts."""
        conditions = [condition for condition in experiment_conditions() if condition["family"] == family]
        if not conditions:
            raise ValueError(f"Unknown experiment family: {family}")
        if all(self.store.completed(condition["id"]) for condition in conditions):
            return [read_json(self.store.path(f"metrics/{condition['id']}/summary.json")) for condition in conditions]
        if self.plans is None:
            self.prepare_calibration_and_plan()
        return [self.run_condition(condition) for condition in conditions]

    def run_condition(self, condition):
        """Save quality/resources independently and record OOM/numerical failures."""
        condition_id = condition["id"]
        summary_relative = f"metrics/{condition_id}/summary.json"
        if self.store.completed(condition_id):
            print(f"Reusing completed {condition_id}", flush=True)
            return read_json(self.store.path(summary_relative))
        self.store.check_time()
        if fingerprint(self.plans) != self.lock["allocation_lock_id"]:
            raise RuntimeError("Allocation policy changed after it was frozen.")
        self.lock["heldout_access_started"] = True
        self.store.save_json("manifests/run_lock.json", self.lock)
        started = time.perf_counter()
        artifacts = []
        summary = {"condition": condition_id, "specification": condition, "status": "running",
                   "config_id": self.protocol_id, "data_id": self.lock["data_id"],
                   "model_revision": self.lock["model_revision"], "runtime": self.runtime}
        try:
            torch.cuda.reset_peak_memory_stats()
            compression_started = time.perf_counter()
            details = self.state.apply(condition, self.statistics, self.profiles, self.allocation_for(condition))
            torch.cuda.synchronize()
            summary["compression_seconds_including_restore"] = time.perf_counter() - compression_started
            summary["compression_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            inventory = read_json(self.store.path("metrics/model_inventory.json"))
            storage = storage_accounting(self.model, inventory, condition, details, self.configuration)
            if condition["family"] == "P3":
                plan = self.allocation_for(condition)
                achieved_zeros = storage["eligible_parameters"] - storage["effective_nonzeros"]
                if achieved_zeros != plan["zero_count"]:
                    raise RuntimeError("Adaptive pruning does not match its global effective-zero budget.")
            if condition["family"] == "Q3":
                plan = self.allocation_for(condition)
                if storage["estimated_packed_model_bytes"] != plan["packed_bytes"] or plan["packed_bytes"] > plan["reference_budget_bytes"]:
                    raise RuntimeError("Actual mixed-precision accounting differs from the frozen byte budget.")
            self.store.save_json(f"metrics/{condition_id}/storage.json", storage)
            self.store.save_json(f"metrics/{condition_id}/projection_policy.json", details)
            errors = compressed_layer_errors(self.state, self.statistics, self.profiles, condition, details)
            self.store.save_json(f"metrics/{condition_id}/layer_errors.json", errors)
            artifacts += [f"metrics/{condition_id}/{name}.json" for name in ["storage", "projection_policy", "layer_errors"]]
            summary["storage"] = {key: value for key, value in storage.items() if key != "projection_storage"}
            summary["mean_projection_calibration_relative_mse"] = sum(row["relative_mse"] for row in errors.values()) / len(errors)
            summary["quality"] = {}
            for suite in ["wikitext2", "c4"]:
                summary["quality"][suite] = evaluate_perplexity(self.model, self.records[suite], suite, condition_id, self.store)
                artifacts += [f"metrics/{condition_id}/{suite}.json", f"predictions/{condition_id}/{suite}.jsonl"]
            summary["quality"]["hellaswag"] = evaluate_hellaswag(self.model, self.tokenizer,
                self.records["hellaswag"][:condition["hellaswag_count"]], condition_id, self.store)
            artifacts += [f"metrics/{condition_id}/hellaswag.json", f"predictions/{condition_id}/hellaswag.jsonl"]
            summary["peak_allocated_through_quality_bytes"] = torch.cuda.max_memory_allocated()
            try:
                summary["latency"] = benchmark_condition(self.model, self.records["calibration"], self.configuration, self.store, condition_id)
                artifacts.append(f"metrics/{condition_id}/latency.json")
                summary["status"] = "complete"
            except torch.OutOfMemoryError as error:
                if not self.configuration["execution"]["continue_after_cuda_oom"]:
                    raise
                summary["latency"] = {"status": "failed_cuda_oom", "error": str(error)}
                summary["status"] = "complete_with_timing_failure"
                gc.collect()
                torch.cuda.empty_cache()
        except (torch.OutOfMemoryError, FloatingPointError) as error:
            if isinstance(error, torch.OutOfMemoryError) and not self.configuration["execution"]["continue_after_cuda_oom"]:
                raise
            summary["status"] = "failed_cuda_oom" if isinstance(error, torch.OutOfMemoryError) else "failed_nonfinite_logits"
            summary["error"] = str(error)
            summary["partial_outputs_retained"] = True
            gc.collect()
            torch.cuda.empty_cache()
        summary["current_session_condition_seconds"] = time.perf_counter() - started
        peaks = [summary.get("compression_peak_allocated_bytes", 0),
                 summary.get("peak_allocated_through_quality_bytes", 0), torch.cuda.max_memory_allocated()]
        peaks += [shape["peak_allocated_bytes"] for shape in summary.get("latency", {}).get("shapes", {}).values()]
        summary["condition_peak_allocated_bytes"] = max(peaks)
        self.store.save_json(summary_relative, summary)
        artifacts.append(summary_relative)
        self.store.complete(condition_id, artifacts, {"condition_status": summary["status"]})
        print(f"Finished {condition_id}: {summary['status']}", flush=True)
        return summary

    def build_evidence(self):
        """Build all required tables/figures from saved predictions, then validate."""
        from Task_1.analysis.evidence import build_evidence

        missing = [condition["id"] for condition in experiment_conditions() if not self.store.completed(condition["id"])]
        if missing:
            raise RuntimeError(f"Cannot finalize before attempting every required condition: {missing}")
        return build_evidence(self.store)

    def finish(self):
        """Validate the saved evidence and show the Drive results folder to download."""
        from Task_1.common.replay import validate_replay

        manifest = validate_replay(self.configuration, self.paths)
        results_folder = self.paths["drive_results"].parent
        print(f"Download this results folder from Drive: {results_folder}", flush=True)
        return {"status": manifest["status"], "computed_on": manifest["computed_on"],
                "results_to_download": str(results_folder), "run_id": manifest["run_id"]}

# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Load immutable originals, swap adapters for diagnostics and use merged drafts for timing."""

import gc
import time

import torch

from Task_3.common.artifacts import read_json, read_jsonl, sha256_file
from Task_3.models.lora import LoRALinear, adapter_state, attach_lora, load_adapter, merge_into_fresh
from Task_3.models.runtime import load_model, step, synchronize
from Task_3.training.distill import atomic_torch_save


class ModelRegistry:
    """Keep only one timed draft resident, with a separate reusable diagnostic wrapper."""

    def __init__(self, store):
        self.store = store
        self.configuration = store.configuration
        self.cache = store.paths["data"] / "huggingface"
        self.target = self.current = self.wrapped = None
        self.current_name = self.wrapped_name = None
        self.states = {}
        self.load_seconds = {}

    def record_load(self, name, start, execution):
        """Preserve model setup separately from warmed request timing."""
        seconds = time.perf_counter() - start
        self.load_seconds[name] = seconds
        relative = "metrics/model_setup.jsonl"
        rows = read_jsonl(self.store.path(relative))
        rows.append({"model": name, "execution": execution, "load_seconds": seconds,
                     "included_in_request_timing": False})
        self.store.save_records(relative, rows)

    def teacher(self):
        if self.target is None:
            start = time.perf_counter()
            self.target = load_model(self.configuration, "teacher", self.cache)
            self.record_load("teacher", start, "frozen_teacher")
            self.store.save_json("metrics/model_geometry.json", {
                "teacher_parameters": sum(x.numel() for x in self.target.parameters()),
                "teacher_vocabulary": self.target.config.vocab_size,
                "teacher_model": self.configuration["models"]["teacher"],
                "teacher_revision": self.configuration["models"]["teacher_revision"],
                "student_model": self.configuration["models"]["student"],
                "student_revision": self.configuration["models"]["student_revision"],
                "original_backbones_frozen": True, "backend": "sdpa"})
        return self.target

    def clear_drafts(self):
        """Release other drafts before measuring memory of a primary timed condition."""
        self.current = self.wrapped = None
        self.current_name = self.wrapped_name = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def state(self, name):
        if name not in self.states:
            row = read_json(self.store.path(f"metrics/training/{name}.json"))
            path = self.store.path(row["adapter_checkpoint"])
            if sha256_file(path) != row["checkpoint_sha256"]:
                raise ValueError("A final adapter checkpoint is corrupt.")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint["config_id"] != read_json(self.store.path("manifests/run_lock.json"))["config_id"]:
                raise ValueError("The adapter belongs to a different scientific run.")
            self.states[name] = checkpoint["adapter"]
        return self.states[name]

    def unmerged(self, name):
        """Use one wrapper for matched-prefix diagnostics, with the original delta disabled."""
        if self.wrapped is None:
            self.clear_drafts()
            start = time.perf_counter()
            self.wrapped = attach_lora(load_model(self.configuration, "student", self.cache), self.configuration["training"], 6304).eval()
            self.record_load("diagnostic_wrapper", start, "unmerged_lora")
        if name != self.wrapped_name:
            if name == "original":
                for module in self.wrapped.modules():
                    if isinstance(module, LoRALinear):
                        module.enabled = False
            else:
                load_adapter(self.wrapped, self.state(name), True)
            self.wrapped_name = name
        return self.wrapped

    def merged(self, name):
        """Use ordinary FP16 projections for all primary benchmarks and standalone quality."""
        if self.current_name != name or self.current is None:
            self.clear_drafts()
            start = time.perf_counter()
            self.current = load_model(self.configuration, "student", self.cache)
            if name != "original":
                summary = read_json(self.store.path(f"metrics/merging/{name}.json"))
                path = self.store.path(summary["merged_checkpoint"])
                if sha256_file(path) != summary["merged_checkpoint_sha256"]:
                    raise ValueError("A merged draft checkpoint is corrupt.")
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                identity = read_json(self.store.path("manifests/run_lock.json"))["config_id"]
                if checkpoint["config_id"] != identity or checkpoint["adapter"] != name:
                    raise ValueError("The merged draft belongs to a different scientific condition.")
                self.current.load_state_dict(checkpoint["model"], strict=True)
                del checkpoint
            self.current_name = name
            self.record_load(name, start, "merged_fp16" if name != "original" else "original_fp16")
        if any(isinstance(module, LoRALinear) for module in self.current.modules()):
            raise RuntimeError("Primary timing must use a genuinely merged backbone.")
        return self.current

    def merge(self, name, prompts):
        """Save one merged checkpoint and numerical comparison per method/seed."""
        stage, relative = f"merge_{name}", f"metrics/merging/{name}.json"
        if self.store.completed(stage):
            return read_json(self.store.path(relative))
        wrapped = self.unmerged(name)
        state = self.state(name)
        device = next(wrapped.parameters()).device
        fresh = load_model(self.configuration, "student", self.cache)
        synchronize(device)
        start = time.perf_counter()
        fresh = merge_into_fresh(fresh, state, self.configuration["training"])
        synchronize(device)
        merging = time.perf_counter() - start
        checks = []
        with torch.inference_mode():
            for prompt in prompts:
                ids = torch.tensor([prompt["prompt_token_ids"]], device=device)
                left, _ = step(wrapped, ids)
                right, _ = step(fresh, ids)
                relative_l2 = float((left - right).norm() / left.norm().clamp_min(1e-12))
                if relative_l2 > self.configuration["evaluation"]["merge_relative_l2_tolerance"]:
                    raise RuntimeError("Merged versus unmerged logits exceed the predeclared tolerance.")
                checks.append({"prompt_id": prompt["id"], "relative_l2": relative_l2, "maximum_absolute": float((left - right).abs().max())})
        checkpoint_path = f"checkpoints/{name}/merged_fp16.pt"
        start = time.perf_counter()
        atomic_torch_save(self.store.path(checkpoint_path), {"model": {key: value.detach().cpu() for key, value in fresh.state_dict().items()},
            "config_id": read_json(self.store.path("manifests/run_lock.json"))["config_id"], "adapter": name})
        self.store.backup([checkpoint_path])
        write_seconds = time.perf_counter() - start
        row = {"draft": name, "status": "complete", "merge_compute_seconds": merging,
            "merge_write_and_backup_seconds": write_seconds, "merged_checkpoint": checkpoint_path,
            "merged_checkpoint_sha256": sha256_file(self.store.path(checkpoint_path)), "merged_checkpoint_bytes": self.store.path(checkpoint_path).stat().st_size,
            "numerical_checks": checks, "wrapper_in_primary_timing": False, "fresh_fp16_backbone": True}
        self.store.save_json(relative, row)
        self.store.complete(stage, [relative, checkpoint_path], {"status": "complete"})
        del fresh
        gc.collect()
        return row

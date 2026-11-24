# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Freeze Task 2's scientific protocol independently of operational settings."""

import copy
import hashlib
import json
from pathlib import Path

import yaml


def fingerprint(value):
    """Hash a stable JSON representation, rejecting non-finite configuration values."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def merge_configuration(base, extra):
    """Merge task overrides without mutating the shared scientific defaults."""
    merged = copy.deepcopy(base)
    for key, value in extra.items():
        merged[key] = merge_configuration(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
    return merged


def load_configuration(task_root):
    """Read the complete protocol and reject changes to assignment-fixed budgets."""
    root = Path(task_root)
    base = yaml.safe_load((root / "configs/base.yaml").read_text(encoding="utf-8"))
    extra = yaml.safe_load((root / "configs/task_2.yaml").read_text(encoding="utf-8")) or {}
    configuration = merge_configuration(base, extra)
    fixed = {
        "seed": 6304,
        "attention.batch": 2, "attention.heads": 4, "attention.head_dimension": 64,
        "attention.lengths": [256, 512, 1024, 2048, 4096, 8192],
        "attention.dtypes": ["float16", "float32"], "attention.methods": ["dense", "tiled", "sdpa"],
        "mqar.sequence_length": 256, "mqar.vocabulary": 256, "mqar.value_min": 128,
        "mqar.value_classes": 64, "mqar.hidden_dimension": 128, "mqar.heads": 4,
        "mqar.layers": 2, "mqar.mlp_multiplier": 4, "mqar.window": 64,
        "mqar.seeds": [6304, 6305, 6306], "mqar.loads": [4, 8, 16],
        "mqar.regimes": ["near", "far", "mixed"], "mqar.batch_size": 32,
        "mqar.updates": 4500, "mqar.warmup_updates": 200, "mqar.peak_lr": .0008,
        "mqar.final_lr_fraction": .15, "mqar.weight_decay": .01, "mqar.gradient_clip": 1.,
        "mqar.checkpoint_updates": [500, 1000, 1500, 2000, 3000, 4500],
        "mqar.test_examples_per_cell": 1024,
        "mqar.inference.lengths": [256, 1024, 4096, 8192],
        "kv.model": "Qwen/Qwen2.5-1.5B-Instruct", "kv.dtype": "float16", "kv.attention_backend": "sdpa",
        "kv.methods": ["fp16", "sink_recent", "question_aware", "int4_token", "int4_channel", "hybrid"],
        "kv.group_size": 64, "kv.sink_tokens": 8, "kv.recent_tokens": 64,
        "kv.uniform_bits": 4, "kv.hybrid_bits": 2,
        "kv.benchmark_batches": [1, 8], "kv.benchmark_lengths": [2048, 4096],
        "figures.font_family": "Times New Roman", "figures.font_size": 10,
    }
    for path, expected in fixed.items():
        actual = configuration
        for field in path.split("."):
            actual = actual.get(field) if isinstance(actual, dict) else None
        if actual != expected:
            raise ValueError(f"Assignment-fixed setting {path}: expected {expected!r}, got {actual!r}.")
    if configuration["attention"]["repetitions"] < 10 or configuration["kv"]["repetitions"] < 10:
        raise ValueError("Attention and packed-cache benchmarks require at least ten repetitions.")
    if configuration["kv"]["bootstrap_resamples"] < 2000:
        raise ValueError("Paired KV confidence intervals require at least 2,000 resamples.")
    for key in ["recovery_every_updates", "prediction_chunk_sequences", "evaluation_batch_size"]:
        if configuration["execution"][key] < 1:
            raise ValueError(f"execution.{key} must be positive.")
    if configuration["mqar"]["intervention"]["parameter_tolerance"] > .05:
        raise ValueError("The intervention cannot exceed the 5% parameter-count tolerance.")
    return configuration


def scientific_configuration(configuration):
    """Exclude paths, recovery frequency and plotting style from scientific identity."""
    result = copy.deepcopy(configuration)
    result.pop("execution", None)
    result.pop("figures", None)
    return result


def source_fingerprint(task_root):
    """Keep compute code locked while allowing independent presentation updates."""
    root = Path(task_root)
    files = []
    for directory in ["common", "data", "attention", "models", "training", "cache", "evaluation"]:
        files.extend(sorted((root / directory).rglob("*.py")))
    return fingerprint({path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files})


def protocol_id(configuration, task_root):
    """Bind all experiment stages to the same scientific configuration and source."""
    return fingerprint({"configuration": scientific_configuration(configuration), "source_id": source_fingerprint(task_root)})

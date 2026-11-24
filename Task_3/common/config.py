# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Freeze the assignment protocol independently of paths and plotting choices."""

import copy
import hashlib
import json
from pathlib import Path

import yaml


def fingerprint(value):
    """Produce a deterministic identity for JSON-compatible finite metadata."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def merge_configuration(base, extra):
    """Merge operational overrides without modifying the original defaults."""
    result = copy.deepcopy(base)
    for key, value in extra.items():
        result[key] = merge_configuration(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_configuration(task):
    """Reject changes to fixed experimental budgets before any model download."""
    task = Path(task)
    base = yaml.safe_load((task / "configs/base.yaml").read_text(encoding="utf-8"))
    result = merge_configuration(base, yaml.safe_load((task / "configs/task_3.yaml").read_text(encoding="utf-8")) or {})
    fixed = {"training.seeds": [6304, 6305], "training.methods": ["off_policy", "on_policy"],
        "training.quota": {"math": 30000, "short": 10000}, "training.response_caps": {"math": 192, "short": 32},
        "training.tokens_per_update": 256, "training.microbatch": 1, "training.lr": .0003, "training.weight_decay": .01,
        "training.gradient_clip": 1., "training.lora_rank": 8, "training.lora_alpha": 16, "training.lora_dropout": 0,
        "training.lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "models.teacher": "Qwen/Qwen2.5-1.5B-Instruct", "models.student": "Qwen/Qwen2.5-0.5B-Instruct",
        "models.dtype": "float16", "models.attention_backend": "sdpa",
        "sampling.temperature": 1., "sampling.top_p": 1., "sampling.top_k": 0, "sampling.gammas": [2, 4],
        "sampling.speed_cap": 64, "sampling.quality_caps": {"math": 512, "short": 32},
        "sampling.repetitions": 2, "sampling.fixed_length_math_prompts": 24,
        "data.math_train": 500, "data.short_train": 300, "data.math_test": 60, "data.short_test": 60,
        "figures.font_family": "Times New Roman", "figures.font_size": 10}
    for name, expected in fixed.items():
        actual = result
        for key in name.split("."):
            actual = actual[key]
        if actual != expected:
            raise ValueError(f"Assignment-fixed {name}: expected {expected!r}, got {actual!r}.")
    if result["sampling"]["markov_trials"] != 100000 or result["evaluation"]["bootstrap_resamples"] < 2000:
        raise ValueError("Use 100,000 Markov trials and at least 2,000 prompt bootstrap resamples.")
    if result["evaluation"]["timing_repetitions"] < 10:
        raise ValueError("Isolated timing requires at least ten warmed-up repetitions.")
    if result["evaluation"]["warmup"] < 1 or result["execution"]["recovery_every_updates"] < 1:
        raise ValueError("Timing warmups and checkpoint intervals must be positive.")
    return result


def protocol_id(configuration, task):
    """Hash scientific settings/source while permitting independent presentation updates."""
    scientific = copy.deepcopy(configuration)
    scientific.pop("execution", None)
    scientific.pop("figures", None)
    task = Path(task)
    sources = {path.relative_to(task).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ["common", "data", "models", "sampling", "training", "evaluation"]
        for path in sorted((task / folder).rglob("*.py"))}
    return fingerprint({"configuration": scientific, "source": sources})


def trained_variants(configuration):
    """List every independent learned draft in a stable, explicit order."""
    return [{"id": f"{method}_seed{seed}", "method": method, "seed": seed}
            for seed in configuration["training"]["seeds"] for method in configuration["training"]["methods"]]


def draft_ids(configuration):
    """Include the unchanged student and all four trained drafts."""
    return ["original", *[row["id"] for row in trained_variants(configuration)]]


def bank_names(configuration):
    """List three common controls and all four named own-student rollout banks."""
    return ["teacher", "original_draft", "exact_spec_original_gamma4", *["own_" + name for name in draft_ids(configuration) if name != "original"]]

# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Load configuration and reject departures from the fixed Task 1 protocol."""

import copy
import hashlib
import json
from pathlib import Path

import yaml


def fingerprint(value):
    """Return a stable identity for JSON-compatible protocol metadata."""
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def merge_configuration(base, override):
    """Recursively merge explicit overrides without modifying the input."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_configuration(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_configuration(task_root, override_path=None):
    """Load and validate the base protocol and task-specific settings."""
    root = Path(task_root)
    with (root / "configs/base.yaml").open(encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    selected = Path(override_path) if override_path else root / "configs/task_1.yaml"
    with selected.open(encoding="utf-8") as handle:
        configuration = merge_configuration(base, yaml.safe_load(handle) or {})
    validate_configuration(configuration)
    return configuration


def validate_configuration(configuration):
    """Enforce assignment budgets and supported numerical conventions."""
    required = {
        ("seed",): 6304,
        ("model", "name"): "Qwen/Qwen2.5-1.5B",
        ("model", "dtype"): "float16",
        ("model", "attention_backend"): "sdpa",
        ("model", "eligible"): ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ("calibration", "sequences"): 128,
        ("calibration", "development_sequences"): 32,
        ("calibration", "sequence_length"): 512,
        ("pruning", "uniform_sparsities"): [0.25, 0.5, 0.75],
        ("pruning", "adaptive_sparsities"): [0.5, 0.75],
        ("pruning", "minimum"): 0.10,
        ("pruning", "maximum"): 0.85,
        ("pruning", "sensitivity"): "isolated_wanda_relative_output_mse_at_50_percent",
        ("pruning", "allocation"): "clipped_inverse_sensitivity_water_filling",
        ("pruning", "selection"): "minimum_development_pooled_token_nll_then_smaller_temperature",
        ("quantization", "uniform_bits"): [8, 4, 3],
        ("quantization", "allowed_bits"): [2, 3, 4, 8],
        ("quantization", "group_size"): 128,
        ("quantization", "scale_dtype"): "float16",
        ("quantization", "zero_point_dtype"): "uint8",
        ("quantization", "rounding"): "nearest_ties_to_even",
        ("quantization", "include_zero_in_range"): True,
        ("quantization", "packing"): "little_endian_bitstream_per_row",
        ("quantization", "fold_down_inverse_into_up"): True,
        ("quantization", "tensor_alignment_bytes"): 16,
        ("quantization", "projection_header_bytes"): 64,
        ("quantization", "mixed_allocation"): "greedy_calibration_distortion_reduction_per_packed_byte",
        ("evaluation", "scored_tokens_per_suite"): 50000,
        ("evaluation", "window_length"): 512,
        ("evaluation", "hellaswag_full_count"): 2000,
        ("evaluation", "hellaswag_diagnostic_count"): 512,
        ("evaluation", "hellaswag_task_version"): "1.0",
        ("evaluation", "hellaswag_reference"): "lm-evaluation-harness_v0.4.9",
        ("evaluation", "confidence"): 0.95,
        ("benchmark", "prompt_lengths"): [512, 2048],
        ("benchmark", "decode_steps"): 64,
        ("benchmark", "ignore_eos_for_fixed_work"): True,
    }
    for keys, expected in required.items():
        observed = configuration
        for key in keys:
            observed = observed.get(key) if isinstance(observed, dict) else None
        if observed != expected:
            raise ValueError(f"Fixed protocol {'.'.join(keys)}: expected {expected!r}, got {observed!r}")
    if configuration["benchmark"]["repetitions"] < 5:
        raise ValueError("At least five warmed-up timing repetitions are required.")
    if type(configuration["benchmark"]["batch_size"]) is not int or configuration["benchmark"]["batch_size"] < 1:
        raise ValueError("Choose one positive benchmark batch size and keep it matched across all conditions.")
    if configuration["evaluation"]["bootstrap_resamples"] < 2000:
        raise ValueError("Use at least 2,000 paired bootstrap resamples.")
    for group, key in [("pruning", "temperature_candidates"), ("quantization", "scale_candidates")]:
        values = configuration[group][key]
        if not values or len(values) != len(set(values)) or any(x < 0 for x in values):
            raise ValueError(f"Invalid fixed candidate grid: {group}.{key}")
    if any(value <= 0 for value in configuration["pruning"]["temperature_candidates"]):
        raise ValueError("P3 candidates must use positive temperatures for nonuniform allocation; P2 is the uniform control.")
    probe_sparsities = configuration["calibration"]["pruning_probe_sparsities"]
    if not {0.25, 0.5, 0.75}.issubset(probe_sparsities) or any(not 0 <= value <= 1 for value in probe_sparsities):
        raise ValueError("Calibration probes must include the three uniform pruning sparsities.")
    if configuration["benchmark"]["warmup_repetitions"] < 1:
        raise ValueError("Warm up each measured path before recording latency.")
    if min(configuration["evaluation"]["head_chunk_tokens"], configuration["evaluation"]["save_every_examples"]) < 1:
        raise ValueError("Head chunks and progress-save intervals must be positive.")
    execution = configuration["execution"]
    if not 0.5 <= execution["gpu_memory_target_fraction"] < 1:
        raise ValueError("GPU workload selection requires a memory target in [0.5, 1).")
    for key in ["gpu_batch_candidates", "gpu_head_chunk_candidates"]:
        values = execution[key]
        if not values or values != sorted(set(values)) or any(type(value) is not int or value < 1 for value in values):
            raise ValueError(f"{key} must be increasing, distinct positive integers.")
    if execution["gpu_batch_candidates"][0] != 1 or execution["gpu_batch_candidates"][-1] > 128:
        raise ValueError("GPU batch probes must begin at 1 and fit the 128 calibration sequences.")
    if not 1 <= execution["calibration_save_every_sequences"] <= configuration["calibration"]["sequences"]:
        raise ValueError("Calibration checkpoint intervals must be between 1 and 128 sequences.")
    if not 0 <= execution["time_reserve_seconds"] < execution["maximum_session_seconds"]:
        raise ValueError("The recovery reserve must be nonnegative and smaller than the session limit.")
    if not 1 <= configuration["calibration"]["probe_tokens_per_sequence"] <= 512:
        raise ValueError("Calibration probe samples must come from every calibration sequence.")
    run_id = configuration["execution"]["run_id"]
    if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in run_id):
        raise ValueError("run_id must contain only letters, numbers, underscores, and hyphens.")


def scientific_configuration(configuration):
    """Exclude operational paths while retaining every scientific/timing setting."""
    return {key: value for key, value in configuration.items() if key not in {"execution", "experiment_name"}}


def source_fingerprint(task_root):
    """Identify implementation files so stale outputs cannot silently be reused."""
    root = Path(task_root)
    directories = ["common", "data", "models", "methods", "evaluation", "training"]
    hashes = {}
    for directory in directories:
        for path in sorted((root / directory).glob("*.py")):
            hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return fingerprint(hashes)

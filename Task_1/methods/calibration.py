# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Calibration activations and independently resumable projection sensitivity probes."""

import time

import torch

from Task_1.common.artifacts import read_json
from Task_1.common.config import fingerprint
from Task_1.methods.pruning import distribution_summary, pruning_scores, row_prune
from Task_1.methods.quantization import (quantize_groups, relative_output_error,
                                        rescaling_equivalence, search_awq_scale)
from Task_1.models.compression import eligible_projections


def atomic_torch_save(path, value):
    """Publish a complete tensor checkpoint before it is registered or backed up."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def save_calibration_progress(store, relative, statistics, completed, positions, selection_id, seconds, peak):
    """Preserve raw sums and ordered samples at a complete-sequence boundary."""
    snapshot = {name: {"sum_squared": state["sum_squared"].cpu(),
                       "sum_absolute": state["sum_absolute"].cpu(),
                       "token_count": state["token_count"],
                       "samples": torch.cat(state["sample_chunks"])}
                for name, state in statistics.items()}
    atomic_torch_save(store.path(relative), {
        "status": "collecting", "completed_sequences": completed, "tokens_per_sequence": 512,
        "probe_positions": positions.tolist(), "selection_id": selection_id,
        "statistics": snapshot, "processing_seconds": seconds, "peak_allocated_bytes": peak})
    store.backup([relative])


@torch.inference_mode()
def collect_activations(model, calibration_records, configuration, store):
    """Collect all fixed tokens, periodically backing up resumable raw statistics."""
    relative = "checkpoints/calibration_activations.pt"
    cache_path = store.path(relative)
    if len(calibration_records) != 128 or any(len(row["token_ids"]) != 512 for row in calibration_records):
        raise ValueError("Activation collection requires all 128 fixed 512-token sequences.")
    batch_size = configuration["execution"].get("gpu_workload", {}).get("calibration_batch_sequences", 1)
    selection_id = fingerprint({"tokens": [row["token_ids"] for row in calibration_records], "batch_sequences": batch_size})
    sample_count = configuration["calibration"]["probe_tokens_per_sequence"]
    positions = torch.linspace(0, 511, sample_count).round().long().unique()
    saved = None
    completed, seconds, peak = 0, 0.0, 0
    if cache_path.is_file():
        saved = torch.load(cache_path, map_location="cpu", weights_only=True)
        if saved.get("selection_id") != selection_id or saved["tokens_per_sequence"] != 512:
            raise ValueError("Incompatible saved activation statistics.")
        if saved.get("status") == "complete":
            if saved["sequences"] != 128 or saved["metadata"]["probe_positions_within_every_sequence"] != positions.tolist():
                raise ValueError("Incompatible completed activation statistics.")
            store.backup([relative])
            store.save_json("metrics/calibration.json", saved["metadata"])
            print("Loaded saved calibration activations.", flush=True)
            return saved["statistics"], saved["metadata"]
        completed = saved["completed_sequences"]
        if saved.get("status") != "collecting" or not 0 < completed < 128 or saved["probe_positions"] != positions.tolist():
            raise ValueError("Incompatible partial calibration checkpoint.")
        seconds, peak = saved["processing_seconds"], saved["peak_allocated_bytes"]
        print(f"Resuming calibration at {completed}/128 saved sequences.", flush=True)
    statistics = {}
    handles = []
    for name, entry in eligible_projections(model, configuration).items():
        columns = entry["module"].weight.shape[1]
        previous = saved["statistics"][name] if saved else None
        if previous and (previous["token_count"] != completed * 512 or previous["samples"].shape != (completed * len(positions), columns)
                         or previous["sum_squared"].shape != (columns,) or previous["sum_absolute"].shape != (columns,)):
            raise ValueError(f"Incomplete raw activation statistics for {name}.")
        statistics[name] = {"sum_squared": previous["sum_squared"].to("cuda") if previous else torch.zeros(columns, device="cuda", dtype=torch.float64),
                            "sum_absolute": previous["sum_absolute"].to("cuda") if previous else torch.zeros(columns, device="cuda", dtype=torch.float64),
                            "token_count": completed * 512, "sample_chunks": [previous["samples"]] if previous else []}

    for name, entry in eligible_projections(model, configuration).items():

        def record_inputs(module, arguments, projection_name=name):
            """Accumulate full-sequence feature statistics without retaining graphs."""
            inputs = arguments[0].detach()
            if inputs.ndim != 3:
                raise ValueError("Calibration projections must retain separate batch and sequence dimensions.")
            values = inputs.float()
            state = statistics[projection_name]
            state["sum_squared"] += values.square().sum(dim=1).double().sum(dim=0)
            state["sum_absolute"] += values.abs().sum(dim=1).double().sum(dim=0)
            state["token_count"] += inputs.shape[0] * inputs.shape[1]
            state["sample_chunks"].append(inputs[:, positions.to(inputs.device)].reshape(-1, inputs.shape[-1]).half().cpu())

        handles.append(entry["module"].register_forward_pre_hook(record_inputs))
    torch.cuda.reset_peak_memory_stats()
    checkpoint_every = configuration["execution"]["calibration_save_every_sequences"]
    try:
        index = completed
        while index < len(calibration_records):
            store.check_time()
            stop = min(index + batch_size, (index // checkpoint_every + 1) * checkpoint_every, 128)
            started = time.perf_counter()
            inputs = torch.tensor([record["token_ids"] for record in calibration_records[index:stop]], device="cuda")
            model.model(input_ids=inputs, use_cache=False, return_dict=True)
            torch.cuda.synchronize()
            seconds += time.perf_counter() - started
            peak = max(peak, torch.cuda.max_memory_allocated())
            if stop % checkpoint_every == 0 and stop < 128:
                save_calibration_progress(store, relative, statistics, stop, positions, selection_id, seconds, peak)
                print(f"Calibration checkpoint saved: {stop}/128 sequences", flush=True)
            if stop % 16 == 0:
                print(f"Calibration: {stop}/128 sequences (batch up to {batch_size})", flush=True)
            index = stop
    finally:
        for handle in handles:
            handle.remove()
    for name, state in statistics.items():
        if state["token_count"] != 128 * 512:
            raise RuntimeError(f"Incomplete activation collection for {name}.")
        statistics[name] = {"input_norm": state["sum_squared"].sqrt().float().cpu(),
                            "mean_absolute": (state["sum_absolute"] / state["token_count"]).float().cpu(),
                            "samples": torch.cat(state["sample_chunks"]), "token_count": state["token_count"]}
    metadata = {"seconds": seconds, "sequences": 128,
                "timing_scope": "completed forwards and activation hooks; excludes checkpoint I/O and repeated unsaved work",
                "peak_allocated_bytes": peak, "checkpoint_interval_sequences": checkpoint_every,
                "resumed_from_sequences": completed,
                "batch_sequences": batch_size,
                "tokens_per_sequence": 512, "probe_positions_within_every_sequence": positions.tolist(),
                "probe_tokens_per_projection": len(positions) * 128,
                "feature_norm_tokens_per_projection": 128 * 512,
                "note": "Full calibration feature norms; fixed, recorded within-sequence subsampling for local error probes."}
    atomic_torch_save(cache_path, {"status": "complete", "selection_id": selection_id,
                                   "sequences": 128, "tokens_per_sequence": 512,
                                   "statistics": statistics, "metadata": metadata})
    store.backup([relative])
    store.save_json("metrics/calibration.json", metadata)
    return statistics, metadata


@torch.inference_mode()
def profile_projections(model, statistics, configuration, store):
    """Save each completed sensitivity/scale search so restarts skip completed units."""
    profiles = {}
    entries = eligible_projections(model, configuration)
    for index, (name, entry) in enumerate(entries.items()):
        store.check_time()
        identifier = name.replace(".", "_")
        report_relative = f"metrics/probes/{identifier}.json"
        scale_relative = f"checkpoints/probes/{identifier}.pt"
        if store.path(report_relative).is_file() and store.path(scale_relative).is_file():
            report = read_json(store.path(report_relative))
            scales = torch.load(store.path(scale_relative), map_location="cpu", weights_only=True)
            profiles[name] = {**report, "scales": scales}
            continue
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        weight = entry["module"].weight.detach()
        state = statistics[name]
        samples = state["samples"].to(weight.device)
        report = {"name": name, "shape": list(weight.shape), "elements": weight.numel(), "layer": entry["layer"],
                  "magnitude_scores": distribution_summary(pruning_scores(weight)),
                  "wanda_scores": distribution_summary(pruning_scores(weight, state["input_norm"])),
                  "activation_norms": distribution_summary(state["input_norm"].to(weight.device)),
                  "magnitude": {}, "wanda": {}, "ptq": {}, "awq": {}}
        for sparsity in configuration["calibration"]["pruning_probe_sparsities"]:
            compressed, _ = row_prune(weight, round(sparsity * weight.numel()), state["input_norm"])
            report["wanda"][str(sparsity)] = relative_output_error(weight, compressed, samples)
            if sparsity in configuration["pruning"]["uniform_sparsities"]:
                compressed, _ = row_prune(weight, round(sparsity * weight.numel()))
                report["magnitude"][str(sparsity)] = relative_output_error(weight, compressed, samples)
        report["pruning_sensitivity"] = report["wanda"]["0.5"]["relative_mse"]
        scales = {}
        for bits in configuration["quantization"]["allowed_bits"]:
            reconstructed, _ = quantize_groups(weight, bits, configuration["quantization"]["group_size"])
            report["ptq"][str(bits)] = relative_output_error(weight, reconstructed, samples)
            scale, search = search_awq_scale(weight, samples, state["mean_absolute"], bits, configuration)
            scales[str(bits)] = scale
            report["awq"][str(bits)] = search
            report["awq"][str(bits)]["rescaling_equivalence"] = rescaling_equivalence(weight, samples, scale)
        report["quantization_sensitivity"] = report["awq"]["3"]["relative_mse"]
        report["probe_seconds"] = time.perf_counter() - started
        report["probe_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        atomic_torch_save(store.path(scale_relative), scales)
        store.backup([scale_relative])
        store.save_json(report_relative, report)
        profiles[name] = {**report, "scales": scales}
        print(f"Sensitivity/AWQ search: {index + 1}/{len(entries)} projections ({name})", flush=True)
    store.save_json("metrics/sensitivity_profiles.json", {name: {key: value for key, value in profile.items() if key != "scales"}
                                                         for name, profile in profiles.items()})
    return profiles


@torch.inference_mode()
def compressed_layer_errors(state, statistics, profiles, condition, details):
    """Measure effective per-projection errors with folding returned to original units."""
    errors = {}
    entries = eligible_projections(state.model, state.configuration)
    awq = condition["family"] in {"Q2", "Q3"}
    for name, entry in entries.items():
        weight = entry["module"].weight
        reference = state.originals[name].to(weight.device)
        candidate = weight.float()
        input_scale = None
        if awq:
            bits = details[name]["bits"]
            input_scale = profiles[name]["scales"][str(bits)].to(weight.device)
            if name.endswith("up_proj"):
                down_name = name.replace("up_proj", "down_proj")
                down_bits = details[down_name]["bits"]
                factor = profiles[down_name]["scales"][str(down_bits)].to(weight.device).float()
                candidate = candidate * factor[:, None]
        errors[name] = relative_output_error(reference, candidate, statistics[name]["samples"], input_scale)
    return errors

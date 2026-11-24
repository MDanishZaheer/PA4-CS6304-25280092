# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Separate packed representation estimates from dense execution measurements."""

import math
import statistics
import time

import torch

from Task_1.methods.quantization import packed_projection_bytes
from Task_1.models.compression import eligible_projections


def storage_accounting(model, inventory, condition, details, configuration):
    """Count eligible/excluded storage, alignment, metadata, and effective nonzeros."""
    projections = eligible_projections(model, configuration)
    family = condition["family"]
    records = {}
    total_packed = inventory["uncompressed_bytes"] + 128
    nonzeros = 0
    nominal_weight_bits = 0
    for name, entry in projections.items():
        weight = entry["module"].weight
        count = int(torch.count_nonzero(weight))
        nonzeros += count
        elements = weight.numel()
        if family in {"Q1", "Q2", "Q3"}:
            bits = details[name]["bits"]
            record = packed_projection_bytes(*weight.shape, bits, configuration,
                                             compensation=details[name]["explicit_compensation"])
        elif family == "fp16":
            bits = 16
            unaligned = 64 + elements * 2
            record = {"header_bytes": 64, "fp16_payload_bytes": elements * 2,
                      "alignment_padding_bytes": (-unaligned) % 16, "packed_bytes": unaligned + (-unaligned) % 16}
        else:
            bits = 16
            mask_bytes = math.ceil(elements / 8)
            unaligned = 64 + mask_bytes + count * 2
            record = {"header_bytes": 64, "mask_bytes": mask_bytes, "fp16_nonzero_bytes": count * 2,
                      "alignment_padding_bytes": (-unaligned) % 16, "packed_bytes": unaligned + (-unaligned) % 16}
            if family == "structured":
                index_bytes = math.ceil((elements // 4) * 3 / 8)
                structured_unaligned = 64 + index_bytes + (elements // 2) * 2
                record["alternative_2of4_index_bytes"] = index_bytes
                record["alternative_2of4_total_bytes"] = structured_unaligned + (-structured_unaligned) % 16
        nominal_weight_bits += bits * elements
        record.update({"elements": elements, "effective_nonzeros": count, "nominal_bits": bits})
        records[name] = record
        total_packed += record["packed_bytes"]
    eligible = inventory["eligible_parameters"]
    return {"dense_fp16_parameter_bytes": inventory["dense_fp16_bytes"],
            "uncompressed_parameter_bytes": inventory["uncompressed_bytes"], "model_header_bytes": 128,
            "estimated_packed_model_bytes": total_packed,
            "packed_size_kind": "justified custom binary-format estimate; no low-bit/sparse inference kernel",
            "packing_convention": "64-byte per-projection header, 16-byte tensor alignment, 128-byte model header; excluded FP16 tensors stored once",
            "effective_nonzeros": nonzeros, "eligible_parameters": eligible,
            "achieved_sparsity": 1 - nonzeros / eligible,
            "nominal_bits_per_eligible_weight": nominal_weight_bits / eligible,
            "effective_packed_bits_per_eligible_weight": 8 * (total_packed - inventory["uncompressed_bytes"] - 128) / eligible,
            "execution": "dense FP16 reconstructed weights; explicit input compensation for unfused AWQ scales",
            "projection_storage": records}


@torch.inference_mode()
def forward_last(model, tokens, cache=None):
    """Use ordinary dense model/cache execution and materialize only last logits."""
    output = model.model(input_ids=tokens, past_key_values=cache, use_cache=True, return_dict=True)
    return model.lm_head(output.last_hidden_state[:, -1:]), output.past_key_values


def sdpa_operations(model, tokens):
    """Record dispatched SDPA operators rather than infer a backend from GPU type."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        logits, cache = forward_last(model, tokens)
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        forward_last(model, token, cache)
        torch.cuda.synchronize()
    keys = sorted({event.key for event in profile.key_averages()
                   if any(marker in event.key for marker in ["scaled_dot_product", "flash_attention", "efficient_attention", "cudnn_attention"])})
    if not keys:
        raise RuntimeError("The profiler did not observe an SDPA operator; backend provenance is missing.")
    return keys


@torch.inference_mode()
def benchmark_condition(model, calibration, configuration, store, condition_id):
    """Benchmark paired fixed-shape prefill and 64 cached decode forward steps."""
    settings = configuration["benchmark"]
    prompt_pool = [token for row in calibration[:4] for token in row["token_ids"]]
    results = {"unit": "milliseconds per batch", "batch_size": settings["batch_size"], "warmup_repetitions": settings["warmup_repetitions"],
               "timing_repetitions": settings["repetitions"], "decode_steps": 64,
               "decode_definition": "64 cached next-token forward/argmax steps after an untimed prefill; EOS ignored for matched fixed work",
               "execution": "dense FP16 SDPA; no claimed sparse/low-bit hardware speedup", "shapes": {}}
    for length in settings["prompt_lengths"]:
        store.check_time()
        tokens = torch.tensor([prompt_pool[:length]], device="cuda", dtype=torch.long).repeat(settings["batch_size"], 1)
        backend = sdpa_operations(model, tokens)
        prefill_times, decode_times, prefill_peaks, decode_peaks = [], [], [], []
        total = settings["warmup_repetitions"] + settings["repetitions"]
        for repetition in range(total):
            store.check_time()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            logits, cache = forward_last(model, tokens)
            torch.cuda.synchronize()
            prefill_time = (time.perf_counter() - started) * 1000
            prefill_peak = torch.cuda.max_memory_allocated()
            token = logits[:, -1].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            for _ in range(settings["decode_steps"]):
                logits, cache = forward_last(model, token, cache)
                token = logits[:, -1].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()
            decode_time = (time.perf_counter() - started) * 1000
            decode_peak = torch.cuda.max_memory_allocated()
            if repetition >= settings["warmup_repetitions"]:
                prefill_times.append(prefill_time)
                decode_times.append(decode_time)
                prefill_peaks.append(prefill_peak)
                decode_peaks.append(decode_peak)
            del logits, cache, token
        results["shapes"][str(length)] = {
            "prompt_tokens": length, "sdpa_observed_operators": backend,
            "prefill_ms": prefill_times, "decode_64_steps_ms": decode_times,
            "prefill_median_ms": statistics.median(prefill_times),
            "decode_median_ms": statistics.median(decode_times),
            "prefill_peak_allocated_bytes": prefill_peaks,
            "decode_peak_allocated_bytes": decode_peaks,
            "peak_allocated_bytes": max(prefill_peaks + decode_peaks),
            "peak_reserved_bytes_at_end": torch.cuda.max_memory_reserved(),
        }
        store.save_json(f"metrics/{condition_id}/latency.json", results)
    return results

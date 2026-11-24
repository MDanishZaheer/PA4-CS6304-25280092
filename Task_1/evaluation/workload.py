# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Choose useful batched scoring work on calibration data before held-out access."""

import gc
import time

import torch

from Task_1.common.artifacts import read_json
from Task_1.common.config import fingerprint
from Task_1.evaluation.scoring import batch_sequence_nll, sequence_nll


def choose_workload(probes, memory_budget):
    """Prefer measured throughput among successful probes with memory headroom."""
    feasible = [row for row in probes if row["status"] == "ok" and row["peak_allocated_bytes"] <= memory_budget]
    if not feasible:
        raise RuntimeError("No scoring workload fits the requested memory headroom on this GPU.")
    return max(feasible, key=lambda row: (row["scored_tokens_per_second"], row["batch_sequences"], row["head_chunk_tokens"]))


def probe_workload(model, records, batch_size, chunk_tokens):
    """Measure real allocated memory and throughput; retain tuning OOMs separately."""
    jobs = [(row["token_ids"], 1, len(row["token_ids"])) for row in records[:batch_size]]
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = {"batch_sequences": batch_size, "head_chunk_tokens": chunk_tokens}
    try:
        scores = batch_sequence_nll(model, jobs, chunk_tokens)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        result.update(status="ok", seconds=seconds, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      scored_tokens_per_second=sum(count for _, count in scores) / seconds)
    except torch.OutOfMemoryError:
        result.update(status="tuning_cuda_oom", seconds=time.perf_counter() - started)
    return result


def configure_gpu_workload(model, records, configuration, store):
    """Freeze one calibration-selected batch/chunk policy and reuse it for all methods."""
    settings = configuration["execution"]
    relative = "metrics/gpu_workload_plan.json"
    requested = {key: settings[key] for key in ["gpu_memory_target_fraction", "gpu_batch_candidates", "gpu_head_chunk_candidates",
                                               "calibration_save_every_sequences"]}
    identity = fingerprint({"request": requested, "calibration_tokens": [row["token_ids"] for row in records]})
    if store.path(relative).is_file():
        plan = read_json(store.path(relative))
        if plan["selection_id"] != identity:
            raise ValueError("Saved GPU workload settings differ; restore them or select a new run_id.")
        check = probe_workload(model, records, plan["batch_sequences"], plan["head_chunk_tokens"])
        if check["status"] != "ok":
            raise RuntimeError("The frozen GPU workload no longer fits. Restore the original runtime/memory availability.")
        store.save_json("metrics/gpu_workload_resume_preflight.json", check)
    else:
        print("Selecting GPU batch sizes using calibration inputs only...", flush=True)
        started = time.perf_counter()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        baseline = torch.cuda.memory_allocated()
        memory_budget = min(int(total_bytes * settings["gpu_memory_target_fraction"]),
                            baseline + int(free_bytes * settings["gpu_memory_target_fraction"]))
        # Warm the unchanged model once; these are selection diagnostics, not reported latency benchmarks.
        batch_sequence_nll(model, [(records[0]["token_ids"], 1, 512)], 64)
        probes = []
        initial_chunk = min(settings["gpu_head_chunk_candidates"])
        for batch_size in settings["gpu_batch_candidates"]:
            store.check_time()
            probe = probe_workload(model, records, batch_size, initial_chunk)
            probes.append(probe)
            print(f"GPU batch probe: {batch_size} sequences | {probe['status']}", flush=True)
            if probe["status"] != "ok" or probe["peak_allocated_bytes"] > memory_budget:
                break
        batch_size = choose_workload(probes, memory_budget)["batch_sequences"]
        for chunk in settings["gpu_head_chunk_candidates"][1:]:
            store.check_time()
            if chunk > batch_size * 511:
                chunk = batch_size * 511
            if any(row["batch_sequences"] == batch_size and row["head_chunk_tokens"] == chunk for row in probes):
                continue
            probe = probe_workload(model, records, batch_size, chunk)
            probes.append(probe)
            if probe["status"] != "ok" or probe["peak_allocated_bytes"] > memory_budget:
                break
        selected = choose_workload(probes, memory_budget)
        plan = {"selection_id": identity, "batch_sequences": selected["batch_sequences"],
                "head_chunk_tokens": selected["head_chunk_tokens"],
                "padded_token_budget": selected["batch_sequences"] * 512,
                "calibration_batch_sequences": min(selected["batch_sequences"], settings["calibration_save_every_sequences"]),
                "memory_budget_bytes": memory_budget, "gpu_total_bytes": total_bytes,
                "target_memory_fraction": settings["gpu_memory_target_fraction"],
                "selected_probe_peak_bytes": selected["peak_allocated_bytes"], "probes": probes,
                "tuning_seconds": time.perf_counter() - started,
                "selection_rule": "highest observed scored-token throughput within memory headroom; calibration inputs only",
                "latency_benchmark_batch_size": configuration["benchmark"]["batch_size"],
                "note": "Headroom guides probe selection; no CUDA allocator quota or dummy VRAM allocation is applied."}
        # Check causal scoring against the original independent-sequence reference before held-out access.
        jobs = [(row["token_ids"][:64], 1, 64) for row in records[:4]]
        batched = batch_sequence_nll(model, jobs, min(plan["head_chunk_tokens"], 256))
        errors = [abs(nll / count - sequence_nll(model, tokens, begin, end)[0] / count)
                  for (nll, count), (tokens, begin, end) in zip(batched, jobs)]
        if max(errors) > .003:
            raise RuntimeError("Batched causal scores disagree with the independent FP16 reference.")
        plan["batch_reference_maximum_mean_nll_error"] = max(errors)
        plan["tuning_seconds"] = time.perf_counter() - started
        store.save_json(relative, plan)
    configuration["execution"]["gpu_workload"] = plan
    store.save_json("configs/resolved_configuration.json", configuration)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU workload: batch={plan['batch_sequences']} candidate sequences, head chunk={plan['head_chunk_tokens']}, "
          f"calibration batch={plan['calibration_batch_sequences']} | latency benchmark batch remains "
          f"{configuration['benchmark']['batch_size']}", flush=True)
    return plan

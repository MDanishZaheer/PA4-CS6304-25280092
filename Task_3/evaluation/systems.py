# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Validate real caches and measure isolated draft/target costs at matched shapes."""

import statistics

import torch

from Task_3.common.artifacts import read_json
from Task_3.models.runtime import clone_cache, step, synchronize, timed


def observe_backend(operation, device):
    """Record dispatched operators instead of inferring a backend from the GPU model."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        operation()
        synchronize(device)
    return sorted({item.key for item in profile.key_averages() if "scaled_dot_product" in item.key or "flash_attention" in item.key or "efficient_attention" in item.key})


@torch.inference_mode()
def validate_real_caches(target, draft, prompts, tolerance=.01):
    """Compare full/split prefill, crop and next-step positions before any real timing."""
    rows = []
    for name, model in [("teacher", target), ("original", draft)]:
        device = next(model.parameters()).device
        for prompt in prompts:
            ids = torch.tensor([prompt["prompt_token_ids"]], device=device)
            split = max(1, ids.shape[-1] // 2)
            full, _ = step(model, ids)
            _, cache = step(model, ids[:, :split])
            divided, cache = step(model, ids[:, split:], cache)
            error = float((full - divided).norm() / full.norm().clamp_min(1e-12))
            if error > tolerance or cache.get_seq_length() != ids.shape[-1]:
                raise RuntimeError("Real-model split prefill or cache positions are inconsistent.")
            cache.crop(split)
            restored, _ = step(model, ids[:, split:], cache)
            rollback = float((full - restored).norm() / full.norm().clamp_min(1e-12))
            if rollback > tolerance:
                raise RuntimeError("Real-model cache crop/advance changed the continuation.")
            rows.append({"model": name, "prompt_id": prompt["id"], "split_relative_l2": error,
                         "rollback_relative_l2": rollback, "tokens": ids.shape[-1], "status": "passed"})
    return {"status": "passed", "conditions": rows}


def measure_operation(operation, setup, device, warmup, repetitions):
    """Exclude independent cache cloning, keep all raw synchronized timing samples."""
    samples, allocated, reserved = [], [], []
    for index in range(warmup + repetitions):
        state = setup()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        seconds, value = timed(lambda: operation(state), device)
        if index >= warmup:
            samples.append(seconds * 1000)
            allocated.append(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
            reserved.append(torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0)
        del state, value
    return {"samples_ms": samples, "median_ms": statistics.median(samples), "warmup": warmup,
            "repetitions": repetitions, "peak_allocated_bytes": max(allocated), "peak_reserved_bytes": max(reserved),
            "scope": "model forward including dynamic cache append; untimed cache clone and resident model/reference cache excluded from latency"}


@torch.inference_mode()
def isolated_costs(store, target, draft, name, wrapper=False):
    """Measure c and v_gamma, explicitly including the pending verification token."""
    settings = store.configuration["evaluation"]
    device = next(target.parameters()).device
    artifacts, rows = [], []
    for length in settings["isolated_context_lengths"]:
        relative = f"systems/{'unmerged_' if wrapper else ''}{name}_L{length}.json"
        if store.path(relative).is_file():
            rows.append(read_json(store.path(relative)))
            artifacts.append(relative)
            continue
        store.check_time()
        # Fixed valid nonspecial tokens and the identical logical context for both.
        vocabulary = min(target.config.vocab_size, draft.config.vocab_size)
        ids = ((torch.arange(length, device=device) % (vocabulary - 3)) + 1)[None]
        token = ids[:, -1:]
        _, target_cache = step(target, ids)
        _, draft_cache = step(draft, ids)
        target_cost = measure_operation(lambda state: step(target, token, state), lambda: clone_cache(target_cache), device,
                                        settings["warmup"], settings["timing_repetitions"])
        draft_cost = measure_operation(lambda state: step(draft, token, state), lambda: clone_cache(draft_cache), device,
                                       settings["warmup"], settings["timing_repetitions"])
        verify = {}
        for gamma in store.configuration["sampling"]["gammas"]:
            block = token.repeat(1, gamma + 1)
            result = measure_operation(lambda state: step(target, block, state, all_logits=True), lambda: clone_cache(target_cache),
                                        device, settings["warmup"], settings["timing_repetitions"])
            result["input_tokens"] = gamma + 1
            result["v_gamma"] = result["median_ms"] / target_cost["median_ms"]
            verify[str(gamma)] = result
        row = {"draft": name, "context_length": length, "status": "complete", "unmerged_wrapper": wrapper,
            "target_single_step": target_cost, "draft_single_step": draft_cost,
            "c": draft_cost["median_ms"] / target_cost["median_ms"], "verification": verify,
            "target_operators": observe_backend(lambda: step(target, token, clone_cache(target_cache)), device),
            "draft_operators": observe_backend(lambda: step(draft, token, clone_cache(draft_cache)), device),
            "verification_shape": "gamma proposals plus one pending returned token; first round has no pending token"}
        store.save_json(relative, row)
        rows.append(row)
        artifacts.append(relative)
    store.complete(f"isolated_{'unmerged_' if wrapper else ''}{name}", artifacts, {"conditions": len(rows)})
    return rows

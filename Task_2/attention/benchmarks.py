# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Resumable Part A correctness/timing and Part B forward-only operator sweeps."""

import contextlib
import gc
import math
import statistics
import time

import torch
import torch.nn.functional as functional

from Task_2.attention.operators import (attention_operator, dense_attention,
                                       theoretical_resources, tiled_attention)
from Task_2.common.artifacts import read_json


def observed_backend(operation):
    """Read dispatched SDPA operations from a profiler instead of guessing by GPU."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        operation()
        torch.cuda.synchronize()
    return sorted({event.key for event in profile.key_averages() if any(
        word in event.key for word in ["scaled_dot_product", "flash_attention", "efficient_attention", "cudnn_attention"])})


def measure_cuda(operation, warmup, repetitions, store=None, setup=None):
    """Measure synchronized fixed work, preserving every sample and memory peak."""
    times, allocated, reserved, additional = [], [], [], []
    for repetition in range(warmup + repetitions):
        if store is not None:
            store.check_time()
        state = setup() if setup is not None else None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        initial_allocated = torch.cuda.memory_allocated()
        started = time.perf_counter()
        result = operation(state) if setup is not None else operation()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        if repetition >= warmup:
            times.append(elapsed)
            allocated.append(torch.cuda.max_memory_allocated())
            reserved.append(torch.cuda.max_memory_reserved())
            additional.append(max(0, torch.cuda.max_memory_allocated() - initial_allocated))
        del result, state
    return {"unit": "milliseconds per batch", "warmup_repetitions": warmup, "timing_repetitions": repetitions,
            "samples_ms": times, "median_ms": statistics.median(times),
            "peak_allocated_bytes": max(allocated), "peak_reserved_bytes": max(reserved),
            "additional_peak_allocated_bytes": max(additional)}


def error_metrics(actual, reference):
    """Use absolute and relative L2 errors without near-zero elementwise ratios."""
    actual, reference = actual.detach(), reference.detach()
    difference = actual.float() - reference.float()
    return {"maximum_absolute": float(difference.abs().max()),
            "relative_l2": float(difference.norm() / reference.float().norm().clamp(min=1e-12))}


def part_a_operator(method, tile_size):
    """Return one of the three required causal computations."""
    return {"dense": dense_attention,
            "tiled": lambda q, k, v: tiled_attention(q, k, v, tile_size),
            "sdpa": lambda q, k, v: functional.scaled_dot_product_attention(q, k, v, is_causal=True)}[method]


def reference_agreement(tensors, upstream, operator, settings):
    """Attempt dense numerical references only when feasible, retaining larger benchmarks."""
    batch, heads, length, _ = tensors[0].shape
    estimate = batch * heads * length * length * 4 * 6
    free, _ = torch.cuda.mem_get_info()
    if length > settings["reference_max_length"] or estimate > free * settings["memory_fraction"]:
        return {"reference_status": "not_attempted_memory_guard", "reference_estimated_working_bytes": estimate}
    references = expected = reference = actual = derivatives = None
    try:
        references = [tensor.detach().float().requires_grad_() for tensor in tensors]
        reference = dense_attention(*references)
        expected = torch.autograd.grad(reference, references, upstream.float())
        actual = operator(*tensors)
        derivatives = torch.autograd.grad(actual, tensors, upstream)
        result = {"reference_status": "dense_fp32_on_identical_input_values",
                  "output_error": error_metrics(actual, reference),
                  "gradient_errors": [error_metrics(first, second) for first, second in zip(derivatives, expected)]}
        maximum = max(result["output_error"]["relative_l2"], *[value["relative_l2"] for value in result["gradient_errors"]])
        tolerance = .005 if tensors[0].dtype == torch.float16 else .0001
        result.update(numerical_tolerance_relative_l2=tolerance,
                      numerical_check="passed" if math.isfinite(maximum) and maximum <= tolerance else "failed_reference_tolerance")
        return result
    except torch.OutOfMemoryError as error:
        return {"reference_status": "failed_cuda_oom", "reference_error": str(error)}
    finally:
        references = expected = reference = actual = derivatives = None
        gc.collect()
        torch.cuda.empty_cache()


def check_attention_numerics():
    """Exercise causality, extreme finite scores, all-masked rows and tile boundaries."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(6304)
    tensors = [torch.randn(1, 2, 37, 16, device=device, generator=generator, requires_grad=True) for _ in range(3)]
    reference = dense_attention(*tensors)
    upstream = torch.randn(reference.shape, device=device, generator=generator)
    reference_gradients = torch.autograd.grad(reference, tensors, upstream)
    checks = []
    for tile in [1, 8, 16, 64]:
        candidate = tiled_attention(*tensors, tile)
        gradients = torch.autograd.grad(candidate, tensors, upstream)
        output_error = error_metrics(candidate, reference)
        gradient_error = max(error_metrics(a, b)["relative_l2"] for a, b in zip(gradients, reference_gradients))
        if output_error["relative_l2"] > 2e-5 or gradient_error > 3e-5:
            raise RuntimeError("Tiled online softmax failed its output/gradient reference.")
        checks.append({"tile": tile, "output": output_error, "maximum_gradient_relative_l2": gradient_error})
    q, k, v = [item.detach().clone() for item in tensors]
    altered_k, altered_v = k.clone(), v.clone()
    altered_k[..., 20:, :] += 100
    altered_v[..., 20:, :] -= 100
    for method in [dense_attention, tiled_attention]:
        if not torch.allclose(method(q, k, v)[..., :20, :], method(q, altered_k, altered_v)[..., :20, :], atol=2e-5, rtol=2e-5):
            raise RuntimeError("Future keys or values changed causal attention outputs.")
    mask = torch.zeros(37, 37, dtype=torch.bool, device=device)
    empty = dense_attention(q, k, v, mask)
    if not torch.isfinite(empty).all() or empty.count_nonzero():
        raise RuntimeError("All-masked rows must return finite zero outputs.")
    extreme = tiled_attention(q * 100, k * 100, v, 16)
    if not torch.isfinite(extreme).all():
        raise RuntimeError("Online-softmax extreme-score check failed.")
    return {"status": "passed", "tiles": checks, "future_perturbation": "passed",
            "all_masked_dense_rows": "zero output", "extreme_scores": "finite",
            "tiled_backward": "custom blockwise recomputation; Python operators, not a fused kernel"}


def run_part_a(store, selected_method):
    """Attempt all 36 required method/dtype/length conditions and retain failures."""
    settings = store.configuration["attention"]
    if not store.path("metrics/attention/correctness.json").is_file():
        store.save_json("metrics/attention/correctness.json", check_attention_numerics())
    records, artifacts = [], ["metrics/attention/correctness.json"]
    for length in settings["lengths"]:
        for dtype_name in settings["dtypes"]:
            for method in [selected_method]:
                relative = f"metrics/attention/{method}_{dtype_name}_L{length}.json"
                if store.path(relative).is_file():
                    records.append(read_json(store.path(relative)))
                    artifacts.append(relative)
                    continue
                row = {"method": method, "dtype": dtype_name, "length": length, "batch": settings["batch"],
                       "heads": settings["heads"], "head_dimension": settings["head_dimension"],
                       "accumulation": "FP32 scores/statistics" if method in {"dense", "tiled"} else "SDPA backend-defined",
                       "reference_status": "not_attempted_above_fixed_reference_length"}
                tensors = reference_tensors = reference = output = None
                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                    free, _ = torch.cuda.mem_get_info()
                    # This conservative guard is reported, never treated as a measured OOM.
                    estimate = settings["batch"] * settings["heads"] * length * length * 4 * 6 if method == "dense" else 0
                    if estimate > free * settings["memory_fraction"]:
                        row.update(status="not_attempted_memory_guard", estimated_dense_working_bytes=estimate)
                    else:
                        generator = torch.Generator(device="cuda").manual_seed(store.configuration["seed"] + length)
                        shape = (settings["batch"], settings["heads"], length, settings["head_dimension"])
                        dtype = getattr(torch, dtype_name)
                        tensors = [torch.randn(shape, device="cuda", generator=generator, dtype=torch.float32).to(dtype).requires_grad_() for _ in range(3)]
                        upstream = torch.randn(shape, device="cuda", generator=generator, dtype=torch.float32).to(dtype)
                        operator = part_a_operator(method, settings["tile_size"])
                        with torch.no_grad():
                            if not torch.isfinite(operator(*tensors)).all():
                                raise FloatingPointError("Attention operator returned non-finite outputs.")
                        row.update(reference_agreement(tensors, upstream, operator, settings))
                        def forward():
                            with torch.no_grad():
                                return operator(*tensors)
                        def forward_backward():
                            for tensor in tensors:
                                tensor.grad = None
                            result = operator(*tensors)
                            (result.float() * upstream.float()).sum().backward()
                            return result
                        row["observed_operators"] = observed_backend(forward) if method == "sdpa" else ["explicit_dense_fp32" if method == "dense" else "python_tiled_custom_recomputed_backward"]
                        row["forward"] = measure_cuda(forward, settings["warmup"], settings["repetitions"], store)
                        row["forward_backward"] = measure_cuda(forward_backward, settings["warmup"], settings["repetitions"], store)
                        row["status"] = "failed_reference_tolerance" if row.get("numerical_check") == "failed_reference_tolerance" else "complete"
                except torch.OutOfMemoryError as error:
                    row.update(status="failed_cuda_oom", error=str(error))
                except FloatingPointError as error:
                    row.update(status="failed_nonfinite_outputs", error=str(error))
                except (NotImplementedError, RuntimeError) as error:
                    if not any(word in str(error).lower() for word in ["not implemented", "no available kernel", "not supported"]):
                        raise
                    row.update(status="unsupported", error=str(error))
                finally:
                    tensors = reference_tensors = reference = output = None
                    gc.collect()
                    torch.cuda.empty_cache()
                store.save_json(relative, row)
                records.append(row)
                artifacts.append(relative)
                print(f"Part A: {method} {dtype_name} L={length}: {row['status']}", flush=True)
    aggregate_path = f"metrics/attention/{selected_method}_conditions.json"
    store.save_json(aggregate_path, records)
    artifacts.append(aggregate_path)
    store.complete(f"attention_{selected_method}", artifacts, {"conditions": len(records), "statuses": {status: sum(row["status"] == status for row in records) for status in {row["status"] for row in records}}})
    if all(store.completed(f"attention_{method}") for method in settings["methods"]):
        all_records = [row for method in settings["methods"] for row in read_json(store.path(f"metrics/attention/{method}_conditions.json"))]
        store.save_json("metrics/attention/all_conditions.json", all_records)
        store.complete("part_a", ["metrics/attention/all_conditions.json"], {"conditions": len(all_records)})
    return records


def run_inference_sweep(store, content_k):
    """Benchmark mechanisms on fixed random Q/K/V without conflating training quality."""
    configuration = store.configuration["mqar"]
    settings = configuration["inference"]
    records, artifacts = [], []
    for length in settings["lengths"]:
        for method in ["dense", "window", "linear", "dilated", "local_content"]:
            relative = f"metrics/mqar_inference/{method}_L{length}.json"
            if store.path(relative).is_file():
                records.append(read_json(store.path(relative)))
                artifacts.append(relative)
                continue
            dimension = configuration["hidden_dimension"] // configuration["heads"]
            row = theoretical_resources(method, settings["batch"], configuration["heads"], length, dimension, content_k=content_k)
            inputs = None
            try:
                generator = torch.Generator(device="cuda").manual_seed(6304 + length)
                inputs = [torch.randn(settings["batch"], configuration["heads"], length, dimension,
                                      device="cuda", dtype=torch.float16, generator=generator) for _ in range(3)]
                def forward():
                    with torch.inference_mode():
                        return attention_operator(*inputs, method, content_k=content_k,
                                                  chunk_size=configuration["linear_chunk"], epsilon=configuration["linear_epsilon"])
                row["observed_operators"] = observed_backend(forward) or ["explicit_chunked_elu1_prefix_state"]
                row["timing"] = measure_cuda(forward, settings["warmup"], settings["repetitions"], store)
                row["status"] = "complete"
            except torch.OutOfMemoryError as error:
                row.update(status="failed_cuda_oom", error=str(error))
            finally:
                inputs = None
                gc.collect()
                torch.cuda.empty_cache()
            store.save_json(relative, row)
            records.append(row)
            artifacts.append(relative)
    store.save_json("metrics/mqar_inference/all_conditions.json", records)
    artifacts.append("metrics/mqar_inference/all_conditions.json")
    store.complete("mqar_inference", artifacts, {"conditions": len(records)})
    return records

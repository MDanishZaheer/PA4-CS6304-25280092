# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Measure real cache packing, unpacking and full-model one-step decode cost."""

import gc
import hashlib

import torch
from transformers.cache_utils import Cache, DynamicLayer

from Task_2.attention.benchmarks import measure_cuda, observed_backend
from Task_2.cache.adapter import BudgetCache, model_step
from Task_2.cache.packing import payload_bytes
from Task_2.cache.selection import selection_plan
from Task_2.common.artifacts import read_json


@torch.inference_mode()
def run_cache_systems(store, model):
    """Use all six policies at B=1/8, L=2048/4096 and ten synchronized repetitions."""
    settings = store.configuration["kv"]
    heads = model.config.num_key_value_heads
    dimension = model.model.layers[0].self_attn.head_dim
    layers = model.config.num_hidden_layers
    device = next(model.parameters()).device
    rows, artifacts = [], []
    for length in settings["benchmark_lengths"]:
        for batch in settings["benchmark_batches"]:
            generator = torch.Generator(device=device).manual_seed(6304 + length)
            originals = []
            for _ in range(layers):
                layer = DynamicLayer()
                keys, values = [torch.randn(1, heads, length, dimension, device=device,
                                            dtype=torch.float16, generator=generator).repeat(batch, 1, 1, 1) for _ in range(2)]
                layer.update(keys, values)
                originals.append(layer)
            reference = Cache(layers=originals)
            geometry = (layers, batch, heads, dimension, length)
            budget = 32 + payload_bytes(*geometry, 4)
            importance = torch.linspace(0, 1, length, device=device)
            token = torch.full((batch, 1), 32, device=device, dtype=torch.long)
            for method in settings["methods"]:
                relative = f"metrics/kv_systems/{method}_B{batch}_L{length}.json"
                if store.path(relative).is_file():
                    rows.append(read_json(store.path(relative)))
                    artifacts.append(relative)
                    continue
                store.check_time()
                packed = None
                row = {"method": method, "batch": batch, "context_length": length,
                       "geometry": {"layers": layers, "kv_heads": heads, "head_dimension": dimension},
                       "workload": "fixed synthetic FP16 KV tensors using the actual model geometry; identical tensors across policies",
                       "reference_int4_budget_bytes": budget, "unit": "milliseconds per batch",
                       "decode_step_scope": "one full Qwen SDPA step including cache append, new-group packing and dequantization",
                       "setup_scope": "untimed independent cache clone before each fixed-shape decode repetition",
                       "memory_scope": "includes model, immutable reference tensors, persistent packed cache and work copy; incremental peaks also reported"}
                try:
                    plan = selection_plan(method, geometry, budget, importance)
                    row["encode"] = measure_cuda(lambda: BudgetCache(reference, plan), settings["warmup"], settings["repetitions"], store)
                    packed = BudgetCache(reference, plan)
                    row["unpack"] = measure_cuda(lambda: [layer.materialize() for layer in packed.layers],
                                                  settings["warmup"], settings["repetitions"], store)
                    binary = packed.serialize()
                    row["serialized_bytes"] = len(binary)
                    row["binary_sha256"] = hashlib.sha256(binary).hexdigest()
                    row["calculated_bytes"] = plan["expected_bytes"]
                    if len(binary) != packed.serialized_bytes() or len(binary) != plan["expected_bytes"]:
                        raise RuntimeError("Packed-cache serialization size is inconsistent.")
                    del binary
                    row["sdpa_observed_operators"] = observed_backend(lambda: model_step(model, token, packed.clone(), length))
                    row["decode_step"] = measure_cuda(lambda state: model_step(model, token, state, length),
                        settings["warmup"], settings["repetitions"], store, setup=packed.clone)
                    row["status"] = "complete"
                except torch.OutOfMemoryError as error:
                    row.update(status="failed_cuda_oom", error=str(error))
                finally:
                    packed = None
                    gc.collect()
                    torch.cuda.empty_cache()
                store.save_json(relative, row)
                rows.append(row)
                artifacts.append(relative)
                print(f"KV systems {method}, B={batch}, L={length}: {row['status']}", flush=True)
            del reference, originals, keys, values
            gc.collect()
            torch.cuda.empty_cache()
    store.save_json("metrics/kv_systems/all_conditions.json", rows)
    artifacts.append("metrics/kv_systems/all_conditions.json")
    store.complete("kv_systems", artifacts, {"conditions": len(rows), "repetitions_per_condition": settings["repetitions"]})
    return rows

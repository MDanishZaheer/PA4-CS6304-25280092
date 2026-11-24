# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Adjacent target controls, counterbalanced draft orders and raw per-round timing."""

import time

from Task_3.common.artifacts import read_json
from Task_3.common.config import draft_ids
from Task_3.sampling.exact import speculative_generate, target_generate


def benchmark(store, target, get_draft, tokenizer, prompts, baseline=False, fixed_length=False):
    """Benchmark each prescribed prompt twice, preserving all output/work/stop diagnostics."""
    suite = "baseline" if baseline else "fixed64" if fixed_length else "primary"
    names = ["original"] if baseline else draft_ids(store.configuration)
    selected = [x for x in prompts if x["domain"] == "math"][:store.configuration["sampling"]["fixed_length_math_prompts"]] if fixed_length else prompts
    artifacts = []
    for repetition in range(store.configuration["sampling"]["repetitions"]):
        # Reverse the full model order in repetition two. Each method/seed and
        # gamma is retained; AB/BA also counterbalances the adjacent target call.
        model_order = names if repetition == 0 else list(reversed(names))
        for model_order_index, name in enumerate(model_order):
            condition_paths = [f"benchmarks/{suite}/{name}/gamma{gamma}/rep{repetition}/{prompt['id']}.json"
                for prompt in selected for gamma in store.configuration["sampling"]["gammas"]]
            if all(store.path(relative).is_file() for relative in condition_paths):
                artifacts.extend(condition_paths)
                continue
            draft = get_draft(name)
            warm_path = f"metrics/warm_setup/{suite}_{name}_rep{repetition}.json"
            previous = read_json(store.path(warm_path)) if store.path(warm_path).is_file() else {}
            start = time.perf_counter()
            warm = selected[0]
            target_generate(target, warm["prompt_token_ids"], 8, 6304, fixed_length)
            speculative_generate(target, draft, warm["prompt_token_ids"], 8, 4, 6304, fixed_length)
            seconds = time.perf_counter() - start
            attempts = previous.get("attempt_seconds", [previous["seconds"]] if previous else []) + [seconds]
            store.save_json(warm_path, {"seconds": seconds, "attempt_seconds": attempts,
                "excluded_from_primary_timing": True, "draft": name, "repetition": repetition,
                "model_order_index": model_order_index, "reason": "warm every newly loaded model, including resumed partial conditions"})
            rotation = (model_order_index * 17 + repetition * 31) % len(selected)
            prompt_order = selected[rotation:] + selected[:rotation]
            source_indices = {row["id"]: index for index, row in enumerate(prompts)}
            for prompt_order_index, prompt in enumerate(prompt_order):
                gammas = store.configuration["sampling"]["gammas"]
                if (prompt_order_index + repetition) % 2:
                    gammas = list(reversed(gammas))
                for gamma in gammas:
                    store.check_time()
                    relative = f"benchmarks/{suite}/{name}/gamma{gamma}/rep{repetition}/{prompt['id']}.json"
                    if store.path(relative).is_file():
                        row = read_json(store.path(relative))
                        if row["prompt_sha256"] != prompt["prompt_sha256"]:
                            raise ValueError("A paired timing record belongs to another prompt encoding.")
                        artifacts.append(relative)
                        continue
                    seed = 6304 + 10007 * source_indices[prompt["id"]] + 1000003 * repetition
                    target_first = (source_indices[prompt["id"]] + repetition + model_order_index + gamma // 2) % 2 == 0
                    operations = {"target": lambda: target_generate(target, prompt["prompt_token_ids"], 64, seed, fixed_length),
                        "speculative": lambda: speculative_generate(target, draft, prompt["prompt_token_ids"], 64, gamma, seed, fixed_length)}
                    outcomes = {}
                    for condition in (["target", "speculative"] if target_first else ["speculative", "target"]):
                        outcomes[condition] = operations[condition]()
                    reference, speculative = outcomes["target"], outcomes["speculative"]
                    row = {"prompt_id": prompt["id"], "prompt_sha256": prompt["prompt_sha256"], "prompt_tokens": len(prompt["prompt_token_ids"]),
                        "workload": prompt["workload"], "draft": name, "gamma": gamma, "repetition": repetition, "suite": suite,
                        "model_order_index": model_order_index, "prompt_order_index": prompt_order_index,
                        "target_first": target_first, "adjacent_target_reference": True, "fixed_length_ignore_eos": fixed_length,
                        "target": reference, "speculative": speculative,
                        "target_text": tokenizer.decode(reference["token_ids"], skip_special_tokens=True),
                        "speculative_text": tokenizer.decode(speculative["token_ids"], skip_special_tokens=True),
                        "end_to_end_tps_ratio": (speculative["new_tokens"] / speculative["elapsed_seconds"]) / (reference["new_tokens"] / reference["elapsed_seconds"]),
                        "steady_decode_tps_ratio": (speculative["new_tokens"] / speculative["decode_seconds"]) / (reference["new_tokens"] / reference["decode_seconds"]),
                        "request_latency_ratio": reference["elapsed_seconds"] / speculative["elapsed_seconds"],
                        "timing_scope": "pretokenized prompt; prefill and synchronized decoding; tokenization, loading, warming, output decoding and disk I/O excluded"}
                    store.save_json(relative, row)
                    artifacts.append(relative)
                if (prompt_order_index + 1) % 20 == 0:
                    print(f"{suite} {name}, rep {repetition}: {prompt_order_index + 1}/{len(selected)}", flush=True)
    store.complete(f"benchmark_{suite}", artifacts, {"drafts": len(names), "prompts": len(selected), "repetitions": 2, "gammas": [2, 4]})
    return {"suite": suite, "paired_records": len(artifacts), "drafts": len(names), "prompts": len(selected)}

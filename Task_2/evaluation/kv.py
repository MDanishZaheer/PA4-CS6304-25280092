# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Greedy retrieval and teacher-forced cache fidelity on all fixed staff prompts."""

import gc
import math
import time

import torch
import torch.nn.functional as functional

from Task_2.attention.benchmarks import error_metrics, measure_cuda
from Task_2.cache.adapter import BudgetCache, clone_dynamic_cache, model_step
from Task_2.cache.packing import payload_bytes
from Task_2.cache.selection import question_importance, selection_plan
from Task_2.common.artifacts import read_json
from Task_2.data.retrieval import extract_code
from Task_2.evaluation.statistics import KV_PAIRS, paired_accuracy_interval


def synchronize(device):
    """Use explicit synchronization around CUDA timings, including cache updates."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def greedy_response(model, tokenizer, initial_logits, cache, max_new_tokens):
    """Generate greedily with fixed EOS treatment and dynamically packed new KV entries."""
    logits, tokens = initial_logits, []
    eos = getattr(model.generation_config, "eos_token_id", None) or tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    started = time.perf_counter()
    synchronize(initial_logits.device)
    started = time.perf_counter()
    stopped = False
    for index in range(max_new_tokens):
        token = int(logits[0, -1].argmax())
        tokens.append(token)
        if token in eos_ids:
            stopped = True
            break
        if index + 1 < max_new_tokens:
            ids = torch.tensor([[token]], device=initial_logits.device, dtype=torch.long)
            logits, _ = model_step(model, ids, cache)
    synchronize(initial_logits.device)
    return {"token_ids": tokens, "text": tokenizer.decode(tokens, skip_special_tokens=True),
            "new_tokens": len(tokens), "eos": stopped, "truncated": not stopped and len(tokens) == max_new_tokens,
            "decode_seconds": time.perf_counter() - started, "final_live_serialized_cache_bytes": cache.serialized_bytes(),
            "final_physical_cache_tokens": cache.get_seq_length(), "final_absolute_cache_length": cache.absolute_length}


@torch.inference_mode()
def teacher_forced_logits(model, first_logits, cache, token_ids):
    """Score the same answer tokens from independent copies of the post-question cache."""
    if not token_ids:
        raise ValueError("Teacher-forced cache fidelity requires at least one answer token.")
    logits = [first_logits[:, -1:]]
    state = cache.clone()
    for token in token_ids[:-1]:
        ids = torch.tensor([[token]], device=first_logits.device, dtype=torch.long)
        continuation, _ = model_step(model, ids, state)
        logits.append(continuation)
    return torch.cat(logits, dim=1)


def answer_nll(logits, tokens):
    """Retain pooled NLL and the supervised answer-token denominator."""
    labels = torch.tensor(tokens, device=logits.device, dtype=torch.long)
    losses = functional.cross_entropy(logits[0].float(), labels, reduction="none")
    return {"nll_sum": float(losses.double().sum()), "tokens": len(tokens), "mean_nll": float(losses.double().mean())}


def logit_kl(reference, candidate):
    """Compute forward KL from FP16-reference logits on paired teacher-forced prefixes."""
    reference_log = reference.float().log_softmax(dim=-1)
    candidate_log = candidate.float().log_softmax(dim=-1)
    divergences = (reference_log.exp() * (reference_log - candidate_log)).sum(dim=-1)
    return float(divergences.double().mean().clamp(min=0))


@torch.inference_mode()
def validate_full_cache(model, tokenized_cases, settings):
    """Verify direct/split prefill, 100% retention and absolute-position continuation."""
    checks = []
    device = next(model.parameters()).device
    for index in [0, 30, 60, 90, 120, 150]:
        case = tokenized_cases[index]
        ids = torch.tensor([case["token_ids"]], device=device, dtype=torch.long)
        prefix, suffix = ids[:, :case["prefix_length"]], ids[:, case["prefix_length"]:]
        direct, _ = model_step(model, ids)
        _, cache = model_step(model, prefix)
        geometry = (len(cache.layers), ids.shape[0], cache.layers[0].keys.shape[1], cache.layers[0].keys.shape[-1], prefix.shape[-1])
        budget = 32 + payload_bytes(*geometry, 4)
        split, _ = model_step(model, suffix, clone_dynamic_cache(cache), prefix.shape[-1])
        identity_cache = BudgetCache(cache, selection_plan("fp16", geometry, budget))
        identity, _ = model_step(model, suffix, identity_cache, prefix.shape[-1])
        direct_error, retained_error = error_metrics(split, direct), error_metrics(identity, split)
        if max(direct_error["relative_l2"], retained_error["relative_l2"]) > settings["reference_tolerance_relative_l2"]:
            raise RuntimeError("Full-precision cache reference failed split/retention equivalence.")
        tokens_a, tokens_b = [], []
        original = BudgetCache(cache, selection_plan("fp16", geometry, budget))
        logits_a, _ = model_step(model, suffix, original, prefix.shape[-1])
        logits_b = identity
        for _ in range(3):
            token_a, token_b = int(logits_a[0, -1].argmax()), int(logits_b[0, -1].argmax())
            tokens_a.append(token_a)
            tokens_b.append(token_b)
            logits_a, _ = model_step(model, torch.tensor([[token_a]], device=device), original)
            logits_b, _ = model_step(model, torch.tensor([[token_b]], device=device), identity_cache)
        if tokens_a != tokens_b or identity_cache.absolute_length != len(case["token_ids"]) + 3:
            raise RuntimeError("100% retention changed greedy continuation or absolute positions.")
        checks.append({"id": case["id"], "direct_vs_split": direct_error, "split_vs_retained_100_percent": retained_error,
                       "three_step_greedy_tokens": tokens_a, "physical_tokens": identity_cache.get_seq_length(),
                       "absolute_tokens": identity_cache.absolute_length, "status": "passed"})
    return {"status": "passed", "cases": checks, "attention_backend": "sdpa",
            "positions": "original absolute rotary IDs; independent physical append positions"}


@torch.inference_mode()
def _evaluate_retrieval_until_complete(store, model, tokenizer, cases):
    """Run/resume 1,080 paired method-prompt records without dropping difficult cases."""
    settings = store.configuration["kv"]
    methods = settings["methods"]
    artifacts, all_records = [], []
    device = next(model.parameters()).device
    for case_index, case in enumerate(cases):
        paths = {method: f"predictions/kv/{case['id']}/{method}.json" for method in methods}
        existing = {method: read_json(store.path(relative)) for method, relative in paths.items() if store.path(relative).is_file()}
        if len(existing) == len(methods):
            for row in existing.values():
                if row["prompt_sha256"] != case["prompt_sha256"]:
                    raise ValueError("Saved KV prediction belongs to a different encoded prompt.")
            all_records.extend(existing.values())
            artifacts.extend(paths.values())
            continue
        store.check_time()
        store.kv_progress_context = {"case": case, "method": "shared_reference"}
        ids = torch.tensor([case["token_ids"]], device=device, dtype=torch.long)
        prefix_ids, suffix_ids = ids[:, :case["prefix_length"]], ids[:, case["prefix_length"]:]
        torch.cuda.reset_peak_memory_stats()
        synchronize(device)
        started = time.perf_counter()
        _, original = model_step(model, prefix_ids)
        synchronize(device)
        prefill_seconds = time.perf_counter() - started
        prefill_peak = torch.cuda.max_memory_allocated()
        first = original.layers[0].keys
        geometry = (len(original.layers), first.shape[0], first.shape[1], first.shape[-1], first.shape[-2])
        budget = 32 + payload_bytes(*geometry, 4)
        raw_prefix_bytes = 4 * geometry[0] * geometry[1] * geometry[2] * geometry[3] * geometry[4]
        synchronize(device)
        started = time.perf_counter()
        importance, reference_first, _ = question_importance(model, original, suffix_ids, settings["observation_max_tokens"])
        synchronize(device)
        importance_seconds = time.perf_counter() - started
        reference_state = BudgetCache(original, selection_plan("fp16", geometry, budget))
        reference_logits, _ = model_step(model, suffix_ids, reference_state, case["prefix_length"])
        torch.testing.assert_close(reference_first, reference_logits, atol=.02, rtol=.02)
        reference_response = greedy_response(model, tokenizer, reference_logits, reference_state.clone(), settings["max_new_tokens"])
        reference_tokens = reference_response["token_ids"]
        if "fp16" in existing and existing["fp16"]["status"] == "complete" and existing["fp16"]["response"]["token_ids"] != reference_tokens:
            raise ValueError("The resumed FP16 reference continuation changed; results cannot be mixed.")
        reference_teacher = teacher_forced_logits(model, reference_logits, reference_state, reference_tokens)
        gold_tokens = tokenizer.encode(case["code"], add_special_tokens=False)
        for method in methods:
            if method in existing:
                row = existing[method]
                if row["prompt_sha256"] != case["prompt_sha256"]:
                    raise ValueError("Saved KV prediction belongs to a different encoded prompt.")
                all_records.append(row)
                artifacts.append(paths[method])
                continue
            store.check_time()
            store.kv_progress_context = {"case": case, "method": method}
            torch.cuda.reset_peak_memory_stats()
            synchronize(device)
            started = time.perf_counter()
            plan = selection_plan(method, geometry, budget, importance)
            selected_seconds = time.perf_counter() - started
            synchronize(device)
            started = time.perf_counter()
            cache = BudgetCache(original, plan)
            synchronize(device)
            encode_seconds = time.perf_counter() - started
            prefix_bytes = cache.serialized_bytes()
            if method != "fp16" and prefix_bytes > budget:
                raise RuntimeError("A compressed-cache method exceeded the case-specific serialized INT4 budget.")
            if case_index in settings["qualitative_case_indices"] and len(cache.serialize()) != prefix_bytes:
                raise RuntimeError("Emitted packed-cache bytes differ from the reported size.")
            synchronize(device)
            reconstruction_started = time.perf_counter()
            reconstruction = {}
            if method in {"int4_token", "int4_channel", "hybrid"}:
                numerators = [0., 0.]
                denominators = [0., 0.]
                for left, right in zip(original.layers, cache.layers):
                    for index, (target, decoded) in enumerate(zip([left.keys, left.values], right.materialize())):
                        numerators[index] += float((decoded.float() - target.float()).square().sum(dtype=torch.float64))
                        denominators[index] += float(target.float().square().sum(dtype=torch.float64))
                reconstruction = {"key_relative_mse": numerators[0] / max(denominators[0], 1e-20),
                                  "value_relative_mse": numerators[1] / max(denominators[1], 1e-20)}
            synchronize(device)
            reconstruction_seconds = time.perf_counter() - reconstruction_started
            synchronize(device)
            started = time.perf_counter()
            first_logits, _ = model_step(model, suffix_ids, cache, case["prefix_length"])
            synchronize(device)
            suffix_seconds = time.perf_counter() - started
            after_suffix = cache.serialized_bytes()
            started = time.perf_counter()
            forced = teacher_forced_logits(model, first_logits, cache, reference_tokens)
            reference_nll = answer_nll(forced, reference_tokens)
            divergence = logit_kl(reference_teacher, forced)
            gold_logits = teacher_forced_logits(model, first_logits, cache, gold_tokens)
            gold_nll = answer_nll(gold_logits, gold_tokens)
            synchronize(device)
            scoring_seconds = time.perf_counter() - started
            response = greedy_response(model, tokenizer, first_logits, cache.clone(), settings["max_new_tokens"])
            response["cached_generated_tokens"] = response["final_absolute_cache_length"] - case["actual_prompt_tokens"]
            response["shared_generated_fp16_equivalent_bytes"] = 4 * geometry[0] * geometry[1] * geometry[2] * geometry[3] * response["cached_generated_tokens"]
            response["actual_live_byte_delta_from_budgeted_prefix"] = response["final_live_serialized_cache_bytes"] - prefix_bytes
            extracted = extract_code(response["text"])
            selected = set(plan["positions"])
            evidence = set(case["evidence_token_positions"])
            low_positions = [position for position in range(case["prefix_length"]) if position not in selected]
            residual_positions = set(low_positions[-(len(low_positions) % 64):]) if len(low_positions) % 64 else set()
            all_fp16 = selected | residual_positions if method == "hybrid" else selected
            row = {"id": case["id"], "method": method, "status": "complete", "prompt_sha256": case["prompt_sha256"],
                   "target_tokens": case["target_tokens"], "position": case["position"], "distractors": case["distractors"],
                   "actual_prompt_tokens": case["actual_prompt_tokens"], "prefix_tokens": case["prefix_length"], "suffix_tokens": case["suffix_length"],
                   "gold_code": case["code"], "extracted_code": extracted, "correct": extracted == case["code"],
                   "fp16_reference_correct": extract_code(reference_response["text"]) == case["code"],
                   "fp16_reference_answer_token_ids": reference_tokens, "reference_answer_nll": reference_nll,
                   "gold_code_nll": gold_nll, "post_prefill_logit_kl": logit_kl(reference_logits, first_logits),
                   "reference_answer_teacher_forced_kl": divergence, **reconstruction,
                   "reference_int4_budget_bytes": budget, "actual_serialized_prefix_bytes": prefix_bytes,
                   "full_fp16_prefix_bytes": raw_prefix_bytes + 32, "fraction_full_fp16_bytes": prefix_bytes / (raw_prefix_bytes + 32),
                   "selected_fp16_positions": plan["positions"], "evidence_positions": case["evidence_token_positions"],
                   "evidence_retention_fraction": len(evidence & selected) / len(evidence) if method in {"sink_recent", "question_aware", "hybrid"} else 1.,
                   "hybrid_any_fp16_evidence_fraction": len(evidence & all_fp16) / len(evidence) if method == "hybrid" else None,
                   "low_stream_fp16_residual_positions": sorted(residual_positions) if method == "hybrid" else [],
                   "prefill_seconds": prefill_seconds, "selection_seconds": selected_seconds,
                   "question_importance_seconds": importance_seconds if method in {"question_aware", "hybrid"} else 0.,
                   "packed_encode_seconds": encode_seconds, "question_suffix_seconds": suffix_seconds, "teacher_forced_scoring_seconds": scoring_seconds,
                   "diagnostic_reconstruction_seconds": reconstruction_seconds,
                   "live_serialized_bytes_after_question": after_suffix, "prefix_to_suffix_actual_byte_delta": after_suffix - prefix_bytes,
                   "shared_question_fp16_equivalent_bytes": 4 * geometry[0] * geometry[1] * geometry[2] * geometry[3] * case["suffix_length"],
                   "prefill_peak_allocated_bytes": prefill_peak, "peak_allocated_bytes": max(prefill_peak, torch.cuda.max_memory_allocated()),
                   "peak_scope": "full prefill, shared reference cache, policy construction, dequantized SDPA and fidelity/generation branches",
                   "cache_execution": "real packed persistent arrays; transient FP16 dequantization before ordinary SDPA",
                   "position_policy": "absolute rotary IDs preserved; physical cache length used for appending",
                   "response": response, "plan": {key: value for key, value in plan.items() if key != "positions"}}
            store.save_json(paths[method], row)
            all_records.append(row)
            artifacts.append(paths[method])
            del cache, first_logits, forced, gold_logits
            print(f"KV {case_index + 1}/{len(cases)} {case['id']} {method}: correct={row['correct']}, bytes={prefix_bytes}/{budget}", flush=True)
        del original, reference_state, reference_teacher, reference_logits, reference_first, importance
        _ = None
        gc.collect()
        torch.cuda.empty_cache()
    if len(all_records) != len(cases) * len(methods):
        raise RuntimeError("KV final evaluation is missing a required method/prompt record.")
    intervals = [{"method": left, "reference": right, **paired_accuracy_interval(
        [row for row in all_records if row["method"] == left], [row for row in all_records if row["method"] == right],
        settings["bootstrap_resamples"], store.configuration["seed"])} for left, right in KV_PAIRS]
    store.save_json("metrics/kv/paired_comparisons.json", intervals)
    artifacts.append("metrics/kv/paired_comparisons.json")
    store.complete("kv_quality", artifacts, {"prompt_records": len(all_records), "paired_intervals": len(intervals)})
    return {"status": "complete", "prompts": len(cases), "methods": len(methods), "records": len(all_records)}


def evaluate_retrieval(store, model, tokenizer, cases):
    """Retain OOM/non-finite attempts and continue without changing the fixed protocol."""
    while True:
        try:
            return _evaluate_retrieval_until_complete(store, model, tokenizer, cases)
        except (torch.OutOfMemoryError, FloatingPointError) as error:
            context = getattr(store, "kv_progress_context", None)
            if context is None:
                raise
            case, active = context["case"], context["method"]
            methods = store.configuration["kv"]["methods"] if active == "shared_reference" else [active]
            status = "failed_cuda_oom" if isinstance(error, torch.OutOfMemoryError) else "failed_nonfinite_logits"
            for method in methods:
                relative = f"predictions/kv/{case['id']}/{method}.json"
                if store.path(relative).is_file():
                    continue
                reference_path = store.path(f"predictions/kv/{case['id']}/fp16.json")
                reference_correct = read_json(reference_path).get("correct", False) if reference_path.is_file() else False
                store.save_json(relative, {"id": case["id"], "method": method, "status": status,
                    "prompt_sha256": case["prompt_sha256"], "target_tokens": case["target_tokens"],
                    "position": case["position"], "distractors": case["distractors"],
                    "prefix_tokens": case["prefix_length"], "suffix_tokens": case["suffix_length"],
                    "actual_prompt_tokens": case["actual_prompt_tokens"], "gold_code": case["code"],
                    "correct": False, "fp16_reference_correct": reference_correct, "error": str(error),
                    "failed_phase": active, "selected_fp16_positions": [], "evidence_positions": case["evidence_token_positions"],
                    "response": {"text": "", "token_ids": [], "new_tokens": 0, "eos": False,
                                 "truncated": False, "decode_seconds": None}})
                print(f"Retained KV failure: {case['id']} {method}: {status}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()

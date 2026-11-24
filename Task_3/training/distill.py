# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Token-budget-matched forward-KL training with exact full-state recovery."""

import copy
import gc
import random
import time

import numpy as np
import torch
from torch.nn import functional

from Task_3.common.artifacts import SessionTimeLimit, atomic_jsonl, read_json, read_jsonl, sha256_file
from Task_3.data.prepare import read_gzip_records, write_gzip_records
from Task_3.models.lora import adapter_digest, adapter_state, load_adapter
from Task_3.models.runtime import peak_memory, synchronize, timed
from Task_3.sampling.exact import target_generate
from Task_3.training.schedule import PromptSchedule, rollout_seed


def atomic_torch_save(path, value):
    """Publish complete trusted tensor/primitives checkpoints atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def capture_rng():
    """Keep NumPy/Python/Torch RNG data compatible with weights_only loading."""
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    """Restore training RNG after models, adapters and optimizer have been constructed."""
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.array(value[1], dtype=np.uint32), value[2], value[3], value[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def response_logits(model, prompt, response, first, count):
    """Project exactly the causal prefixes predicting the specified response-token range."""
    device = next(model.parameters()).device
    context = prompt + response[:first + count - 1]
    ids = torch.tensor([context], device=device, dtype=torch.long)
    output = model.model(input_ids=ids, use_cache=False, return_dict=True)
    start = len(prompt) - 1 + first
    logits = model.lm_head(output.last_hidden_state[:, start:start + count]).float()[0]
    if logits.shape[0] != count or not torch.isfinite(logits).all():
        raise FloatingPointError("Invalid causal response-logit selection during distillation.")
    return logits


def forward_kl_sum(student_logits, teacher_log_probs):
    """Compute full-vocabulary forward KL; sampled token IDs are never hard labels."""
    student_log_probs = functional.log_softmax(student_logits.float(), dim=-1)
    teacher_log_probs = teacher_log_probs.to(student_logits.device, dtype=torch.float32)
    return (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)).sum()


def prepare_teacher_pool(store, teacher, prompts, seed):
    """Generate all 800 immutable teacher responses before the off-policy updates."""
    relative = f"metrics/teacher_pool_seed{seed}.json"
    stage = f"teacher_pool_seed{seed}"
    if store.completed(stage):
        return read_json(store.path(relative))
    artifacts, rows = [], []
    for index, prompt in enumerate(prompts):
        store.check_time()
        path = f"rollouts/teacher_pool_seed{seed}/{prompt['id']}.json"
        if store.path(path).is_file():
            row = read_json(store.path(path))
            if row["prompt_sha256"] != prompt["prompt_sha256"]:
                raise ValueError("Teacher pool prompt identity changed.")
        else:
            gen_seed = rollout_seed(seed, prompt["id"], index, "frozen_teacher_pool")
            result = target_generate(teacher, prompt["prompt_token_ids"],
                store.configuration["training"]["response_caps"][prompt["domain"]], gen_seed)
            row = {"source_id": prompt["id"], "domain": prompt["domain"], "prompt_sha256": prompt["prompt_sha256"],
                   "rollout_seed": gen_seed, "teacher_scored_tokens": 0, "producer": "teacher", **result}
            store.save_json(path, row)
        rows.append(row)
        artifacts.append(path)
        if (index + 1) % 25 == 0:
            print(f"Frozen teacher pool seed {seed}: {index + 1}/{len(prompts)}", flush=True)
    summary = {"seed": seed, "prompts": len(rows), "math_prompts": sum(x["domain"] == "math" for x in rows),
        "short_prompts": sum(x["domain"] == "short" for x in rows), "generation_seconds": sum(x["elapsed_seconds"] for x in rows),
        "generated_tokens": sum(x["new_tokens"] for x in rows), "peak_allocated_bytes": max(x["peak_allocated_bytes"] for x in rows),
        "shared_between_methods": False, "cost_accounting": "one separate frozen 800-prompt pool for each off-policy training seed"}
    store.save_json(relative, summary)
    store.complete(stage, [*artifacts, relative], summary)
    return summary


def train_variant(store, teacher, student, prompts, variant):
    """Finish the fixed 30k/10k teacher-scored quota, restoring exact update boundaries."""
    name, settings = variant["id"], store.configuration["training"]
    stage, summary_path = f"train_{name}", f"metrics/training/{name}.json"
    if store.completed(stage):
        return read_json(store.path(summary_path))
    device = next(student.parameters()).device
    student.eval()
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if not parameters or any(parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError("Distillation requires FP32 trainable adapters on a frozen backbone.")
    optimizer = torch.optim.AdamW(parameters, lr=settings["lr"], weight_decay=settings["weight_decay"])
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    recovery_path = f"checkpoints/{name}/recovery.pt"
    history_path = f"histories/{name}.jsonl"
    identity = read_json(store.path("manifests/run_lock.json"))["config_id"]
    initialization = adapter_digest(student)
    schedule = PromptSchedule(prompts, variant["seed"], settings["category_cycle"])
    counts, current, history, artifacts = {"math": 0, "short": 0}, None, [], []
    update, overflows = 0, 0
    if store.path(recovery_path).is_file():
        checkpoint = torch.load(store.path(recovery_path), map_location="cpu", weights_only=True)
        if checkpoint["config_id"] != identity or checkpoint["variant"] != variant or checkpoint["initialization_sha256"] != initialization:
            raise ValueError("Training recovery identity/initialization differs.")
        load_adapter(student, checkpoint["adapter"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        counts, current, history = checkpoint["counts"], checkpoint["current_response"], checkpoint["history"]
        schedule.state = checkpoint["schedule"]
        update, overflows, artifacts = checkpoint["update"], checkpoint["overflows"], checkpoint["artifacts"]
        restore_rng(checkpoint["rng"])
        print(f"Resumed {name}: {sum(counts.values())}/40000 scored tokens", flush=True)
    def save_recovery():
        atomic_jsonl(store.path(history_path), history)
        atomic_torch_save(store.path(recovery_path), {"config_id": identity, "variant": variant,
            "initialization_sha256": initialization, "adapter": adapter_state(student), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "counts": counts, "current_response": current, "history": history,
            "schedule": schedule.state, "update": update, "overflows": overflows, "rng": capture_rng(), "artifacts": artifacts})
        store.backup([history_path, recovery_path])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    session_start = time.perf_counter()
    try:
        while sum(counts.values()) < sum(settings["quota"].values()):
            store.check_time()
            student.eval()
            producing_digest = adapter_digest(student)
            quota_left = {domain: settings["quota"][domain] - counts[domain] for domain in counts}
            update_tokens = min(settings["tokens_per_update"], sum(quota_left.values()))
            chunks, annotations, generation_seconds, scoring_seconds = [], [], 0., 0.
            remaining_update = update_tokens
            while remaining_update:
                if current is None:
                    prompt, presentation = schedule.next(quota_left)
                    if variant["method"] == "off_policy":
                        response = read_json(store.path(f"rollouts/teacher_pool_seed{variant['seed']}/{prompt['id']}.json"))
                        generation = 0.
                        producer = "frozen_teacher"
                    else:
                        seed = rollout_seed(variant["seed"], prompt["id"], presentation, "on_policy_training")
                        response = target_generate(student, prompt["prompt_token_ids"], settings["response_caps"][prompt["domain"]], seed)
                        response["rollout_seed"] = seed
                        generation = response["elapsed_seconds"]
                        producer = producing_digest
                    if not response["token_ids"]:
                        raise ValueError("A training rollout contains no teacher-scoreable response prefix.")
                    current = {"source_id": prompt["id"], "domain": prompt["domain"], "presentation": presentation,
                        "prompt_sha256": prompt["prompt_sha256"], "prompt": prompt["prompt_token_ids"],
                        "response": response["token_ids"], "cursor": 0, "rollout_seed": response["rollout_seed"],
                        "producer_checkpoint": producer, "producer_update": update,
                        "termination": "eos" if response["eos"] else "length_cap", "generation_seconds": generation}
                    generation_seconds += generation
                domain, first = current["domain"], current["cursor"]
                count = min(len(current["response"]) - first, quota_left[domain], remaining_update)
                if count <= 0:
                    raise RuntimeError("The distillation scheduler selected an exhausted response/category.")
                def score_teacher():
                    with torch.inference_mode():
                        logits = response_logits(teacher, current["prompt"], current["response"], first, count)
                        return functional.log_softmax(logits, dim=-1).cpu()
                seconds, log_probs = timed(score_teacher, device)
                scoring_seconds += seconds
                chunks.append({"prompt": current["prompt"], "response": current["response"], "first": first,
                               "count": count, "teacher_log_probs": log_probs})
                quota_left[domain] -= count
                remaining_update -= count
                current["cursor"] += count
                annotations.append({key: current[key] for key in ["source_id", "domain", "presentation", "prompt_sha256", "rollout_seed",
                    "producer_checkpoint", "producer_update", "termination", "response"]} | {
                    "first_scored_response_index": first, "teacher_scored_tokens": count,
                    "quota_truncated": quota_left[domain] == 0 and current["cursor"] < len(current["response"]),
                    "generation_seconds": current["generation_seconds"] if first == 0 else 0., "update": update + 1})
                if current["cursor"] == len(current["response"]) or quota_left[domain] == 0:
                    current = None
            synchronize(device)
            backward_start = time.perf_counter()
            student.train()
            for attempt in range(settings["maximum_overflow_retries"]):
                optimizer.zero_grad(set_to_none=True)
                total_kl = 0.
                for chunk in chunks:
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        q_logits = response_logits(student, chunk["prompt"], chunk["response"], chunk["first"], chunk["count"])
                        loss = forward_kl_sum(q_logits, chunk["teacher_log_probs"]) / update_tokens
                    if not torch.isfinite(loss):
                        raise FloatingPointError("The full-vocabulary forward-KL loss is not finite.")
                    total_kl += float(loss.detach())
                    scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_clip"])
                if not scaler.is_enabled() and not torch.isfinite(norm):
                    raise FloatingPointError("A distillation gradient norm is non-finite.")
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    break
                overflows += 1
            else:
                raise FloatingPointError("Repeated adapter gradient overflow; teacher targets were reused, not rescored.")
            synchronize(device)
            backward_seconds = time.perf_counter() - backward_start
            counts = {domain: settings["quota"][domain] - quota_left[domain] for domain in counts}
            update += 1
            annotation_path = f"rollouts/training/{name}/update_{update:04d}.jsonl.gz"
            write_gzip_records(store.path(annotation_path), annotations)
            store.backup([annotation_path])
            artifacts.append(annotation_path)
            history.append({"update": update, "teacher_scored_tokens": update_tokens, "cumulative_tokens": sum(counts.values()),
                "cumulative_math_tokens": counts["math"], "cumulative_short_tokens": counts["short"], "forward_kl": total_kl,
                "lr": settings["lr"], "gradient_norm_before_clip": float(norm), "overflow_attempts": overflows,
                "student_rollout_seconds": generation_seconds, "teacher_scoring_seconds": scoring_seconds,
                "student_backward_and_optimizer_seconds": backward_seconds, "teacher_scoring_calls": len(chunks),
                "teacher_targets_reused_for_overflow_retry": True})
            del chunks, log_probs, q_logits, loss
            if update % store.configuration["execution"]["recovery_every_updates"] == 0 or sum(counts.values()) == sum(settings["quota"].values()):
                save_recovery()
                print(f"{name}: {counts} | successful updates {update}, KL={total_kl:.5f}", flush=True)
    except SessionTimeLimit:
        save_recovery()
        raise
    except (torch.OutOfMemoryError, FloatingPointError) as error:
        store.save_json(f"metrics/training/{name}_failure.json", {"variant": variant, "error": str(error),
            "status": "failed_cuda_oom" if isinstance(error, torch.OutOfMemoryError) else "failed_nonfinite",
            "last_committed_tokens": history[-1]["cumulative_tokens"] if history else 0,
            "recovery": recovery_path, "required_condition_retained": True})
        raise
    final_path = f"checkpoints/{name}/adapter_final.pt"
    state = adapter_state(student)
    atomic_torch_save(store.path(final_path), {"config_id": identity, "variant": variant, "adapter": state,
        "scored_tokens": counts, "update": update, "initialization_sha256": initialization})
    store.backup([final_path])
    pool_seconds = read_json(store.path(f"metrics/teacher_pool_seed{variant['seed']}.json"))["generation_seconds"] if variant["method"] == "off_policy" else 0.
    summary = {**variant, "status": "complete", "teacher_scored_tokens": counts, "successful_optimizer_updates": update,
        "overflow_attempts": overflows, "initialization_sha256": initialization, "adapter_sha256": adapter_digest(state),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters), "total_parameters": sum(parameter.numel() for parameter in student.parameters()),
        "history": history_path, "annotations": artifacts, "adapter_checkpoint": final_path,
        "checkpoint_sha256": sha256_file(store.path(final_path)), "teacher_pool_generation_seconds": pool_seconds,
        "student_rollout_seconds": sum(x["student_rollout_seconds"] for x in history),
        "teacher_scoring_seconds": sum(x["teacher_scoring_seconds"] for x in history),
        "student_backward_and_optimizer_seconds": sum(x["student_backward_and_optimizer_seconds"] for x in history),
        "current_session_wall_seconds": time.perf_counter() - session_start, "final_selection": "last update at fixed token quota; no test-based selection",
        "adapter_precision": "float32", "backbone_precision": "float16", **peak_memory(device)}
    store.save_json(summary_path, summary)
    store.complete(stage, [summary_path, history_path, final_path, *artifacts], {"tokens": counts, "updates": update})
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    return summary

# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Independent MQAR training with complete optimizer/RNG recovery and fixed budgets."""

import gc
import hashlib
import random
import time

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from Task_2.common.artifacts import SessionTimeLimit, atomic_jsonl, read_json, sha256_file
from Task_2.common.runtime import seed_everything
from Task_2.data.mqar import TrainingStream, tensor_batch
from Task_2.evaluation.recall import development_metrics
from Task_2.models.mqar import RecallTransformer


def atomic_torch_save(path, value):
    """Publish one complete trusted recovery file at an optimizer-update boundary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def capture_rng():
    """Store only tensors and primitive values so weights_only loading is supported."""
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(),
             int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    """Restore all generators after model/optimizer/DataLoader construction."""
    random.setstate(state["python"])
    saved = state["numpy"]
    np.random.set_state((saved[0], np.array(saved[1], dtype=np.uint32), saved[2], saved[3], saved[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def model_digest(model):
    """Record identical independent initializations without serializing extra copies."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def learning_rate(update, settings):
    """Warm up for 200 updates, then linearly decay to 15% at update 4,500."""
    if update <= settings["warmup_updates"]:
        return settings["peak_lr"] * update / settings["warmup_updates"]
    progress = (update - settings["warmup_updates"]) / (settings["updates"] - settings["warmup_updates"])
    return settings["peak_lr"] * (1 - (1 - settings["final_lr_fraction"]) * progress)


def train_condition(store, condition, development, content_k=64, device="cuda"):
    """Finish exactly 4,500 successful optimizer updates or retain an explicit failure."""
    name = condition["id"]
    stage = f"train_{name}"
    summary_path = f"metrics/mqar_training/{name}.json"
    if store.completed(stage):
        return read_json(store.path(summary_path))
    session_started = time.perf_counter()
    checkpoint_io_seconds = 0.
    settings = store.configuration["mqar"]
    operational = store.configuration["execution"]
    lock = read_json(store.path("manifests/run_lock.json"))
    seed_everything(condition["seed"])
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model = RecallTransformer(settings, condition["method"], content_k).to(device)
    parameters = model.parameter_count()
    baseline_parameters = RecallTransformer(settings, "dense").parameter_count()
    if abs(parameters / baseline_parameters - 1) > settings["intervention"]["parameter_tolerance"]:
        raise ValueError("Intervention parameter count violates the baseline tolerance.")
    initialization = model_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["peak_lr"], weight_decay=settings["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    recovery = f"checkpoints/mqar/{name}/recovery.pt"
    completed, history, evaluations, skipped, elapsed = 0, [], [], 0, 0.
    saved = None
    if store.path(recovery).is_file():
        saved = torch.load(store.path(recovery), map_location=device, weights_only=True)
        if saved["config_id"] != lock["config_id"] or saved["condition"] != condition or saved["content_k"] != content_k:
            raise ValueError("MQAR recovery does not match this run, mechanism or frozen intervention.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        completed, history, evaluations = saved["completed_updates"], saved["history"], saved["evaluations"]
        skipped, elapsed, initialization = saved["skipped_updates"], saved["training_seconds"], saved["initialization_sha256"]
        if len(history) != completed or saved["next_training_index"] != completed * settings["batch_size"]:
            raise ValueError("MQAR saved sampler/history position is inconsistent.")
        print(f"Resuming {name} at {completed}/{settings['updates']} optimizer updates", flush=True)
    dataset = TrainingStream(store.paths["resources"], condition["train_K"], condition["train_regime"],
                             condition["seed"], settings["updates"], settings["batch_size"])
    loader_generator = torch.Generator().manual_seed(condition["seed"])
    loader = DataLoader(dataset, batch_size=settings["batch_size"],
                        sampler=range(completed * settings["batch_size"], len(dataset)),
                        collate_fn=tensor_batch, num_workers=operational["data_loader_workers"],
                        pin_memory=str(device).startswith("cuda"), generator=loader_generator,
                        persistent_workers=operational["data_loader_workers"] > 0)
    if saved:
        restore_rng(saved["rng"])
    historical = [f"checkpoints/mqar/{name}/update_{update:04d}.pt" for update in settings["checkpoint_updates"] if update <= completed]
    history_path = f"histories/mqar/{name}.jsonl"
    last_saved_update = completed if saved else None
    pending_queries = [saved["pending_queries"].cpu()] if saved and saved.get("pending_queries") is not None and saved["pending_queries"].numel() else []
    query_map_paths = list(saved.get("training_query_maps", [])) if saved else []

    def save_query_maps(force=False):
        """Deduplicate uint8 training annotations across methods with the same staff stream."""
        if not pending_queries:
            return
        if not force and completed % operational["recovery_every_updates"] and completed != settings["updates"]:
            return
        array = torch.cat(pending_queries).numpy()
        first_index = dataset.start_index + completed * settings["batch_size"] - len(array)
        stop_index = first_index + len(array)
        stem = f"splits/mqar_training/seed{condition['seed']}_K{condition['train_K']}_{condition['train_regime']}/queries_{first_index:09d}_{stop_index:09d}"
        relative, receipt = stem + ".npz", stem + ".json"
        target = store.path(relative)
        if target.is_file():
            with np.load(target, allow_pickle=False) as old:
                if not np.array_equal(old["queries"], array) or int(old["start_index"][0]) != first_index:
                    raise ValueError("Shared MQAR training annotations differ across matched mechanisms.")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".partial")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, queries=array, start_index=np.array([first_index], dtype=np.int64))
            temporary.replace(target)
            store.backup([relative])
        store.save_json(receipt, {"split": "train", "seed": condition["seed"], "K": condition["train_K"],
            "regime": condition["train_regime"], "start_index": first_index, "stop_index": stop_index,
            "examples": len(array), "columns": ["query_position", "answer_id", "source_position", "distance"],
            "dtype": "uint8", "sha256": sha256_file(target),
            "generator_sha256": sha256_file(store.paths["resources"] / "scripts/mqar_generator.py")})
        query_map_paths.extend([relative, receipt])
        pending_queries.clear()

    def save_recovery():
        """Keep sampler, schedule, full history and optimizer state in one commit."""
        nonlocal checkpoint_io_seconds, last_saved_update
        started = time.perf_counter()
        save_query_maps()
        snapshot = {"config_id": lock["config_id"], "condition": condition, "content_k": content_k,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "completed_updates": completed, "next_training_index": completed * settings["batch_size"],
                    "schedule_state": {"completed_updates": completed, "next_lr": learning_rate(min(completed + 1, settings["updates"]), settings)},
                    "rng": capture_rng(), "history": history, "evaluations": evaluations,
                    "skipped_updates": skipped, "training_seconds": elapsed, "initialization_sha256": initialization,
                    "pending_queries": torch.cat(pending_queries) if pending_queries else torch.empty((0, condition["train_K"], 4), dtype=torch.uint8),
                    "training_query_maps": query_map_paths}
        atomic_torch_save(store.path(recovery), snapshot)
        store.backup([recovery])
        atomic_jsonl(store.path(history_path), history)
        store.backup([history_path])
        checkpoint_io_seconds += time.perf_counter() - started
        last_saved_update = completed

    status, failure = "complete", None
    waiting = time.perf_counter()
    try:
        for batch in loader:
            store.check_time()
            wait_seconds = time.perf_counter() - waiting
            query_map = torch.stack([batch["positions"], batch["labels"] + 128, batch["sources"], batch["distances"]], dim=-1).to(torch.uint8)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items() if key in {"tokens", "positions", "labels"}}
            model.train()
            update = completed + 1
            rate = learning_rate(update, settings)
            for group in optimizer.param_groups:
                group["lr"] = rate
            update_started = time.perf_counter()
            for attempt in range(operational["maximum_overflow_retries"]):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=str(device).startswith("cuda")):
                    logits = model(batch["tokens"], batch["positions"])
                    loss = functional.cross_entropy(logits.flatten(0, 1).float(), batch["labels"].flatten())
                if not torch.isfinite(loss):
                    raise FloatingPointError("MQAR supervised loss became non-finite.")
                previous_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
                if not scaler.is_enabled() and not torch.isfinite(gradient_norm):
                    raise FloatingPointError("MQAR gradient norm became non-finite.")
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    break
                skipped += 1
            else:
                raise FloatingPointError("MQAR optimizer repeatedly overflowed; token/update budgets were not reduced.")
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            seconds = time.perf_counter() - update_started
            elapsed += seconds + wait_seconds
            completed = update
            pending_queries.append(query_map)
            history.append({"update": completed, "loss": float(loss.detach()), "lr": rate,
                            "gradient_norm_before_clip": float(gradient_norm), "update_seconds": seconds,
                            "batch_wait_seconds": wait_seconds, "training_seconds": elapsed,
                            "supervised_queries": settings["batch_size"] * condition["train_K"],
                            "skipped_optimizer_attempts": skipped})
            if completed in settings["checkpoint_updates"]:
                started = time.perf_counter()
                metrics = development_metrics(model, development, store.configuration, condition["train_K"])
                evaluations.append({"update": completed, "split": "development", "metrics": metrics,
                                    "evaluation_seconds": time.perf_counter() - started})
                historical_path = f"checkpoints/mqar/{name}/update_{completed:04d}.pt"
                atomic_torch_save(store.path(historical_path), {"config_id": lock["config_id"], "condition": condition,
                    "content_k": content_k, "update": completed, "model": model.state_dict()})
                store.backup([historical_path])
                historical.append(historical_path)
            if completed % operational["recovery_every_updates"] == 0 or completed in settings["checkpoint_updates"]:
                save_recovery()
            if completed % operational["progress_every_updates"] == 0:
                print(f"{name}: update {completed}/{settings['updates']}, loss={float(loss.detach()):.4f}, verified recovery <= {operational['recovery_every_updates']} updates", flush=True)
            waiting = time.perf_counter()
    except SessionTimeLimit:
        save_recovery()
        raise
    except (torch.OutOfMemoryError, FloatingPointError) as error:
        status = "failed_cuda_oom" if isinstance(error, torch.OutOfMemoryError) else "failed_nonfinite_training"
        failure = str(error)
        optimizer.zero_grad(set_to_none=True)
        save_query_maps(force=True)
        save_recovery()
    if status == "complete" and completed != settings["updates"]:
        raise RuntimeError("MQAR training ended without the required optimizer-update count.")
    if last_saved_update != completed:
        save_recovery()
    summary = {**condition, "status": status, "completed_updates": completed,
               "trainable_parameters": parameters, "baseline_trainable_parameters": baseline_parameters,
               "initialization_sha256": initialization, "final_sha256": model_digest(model),
               "training_seconds": elapsed, "skipped_optimizer_attempts": skipped, "content_k": content_k,
               "current_session_wall_seconds": time.perf_counter() - session_started,
               "checkpoint_and_backup_seconds_this_session": checkpoint_io_seconds,
               "saved_development_seconds": sum(row["evaluation_seconds"] for row in evaluations),
               "timing_scope": "training_seconds pools successful update/data-wait time across recovery; current-session wall time includes setup, evaluation and backup",
               "development_evaluations": evaluations, "final_checkpoint": historical[-1] if historical else None,
               "checkpoint_updates": [row["update"] for row in evaluations], "history": history_path,
               "training_query_maps": query_map_paths,
               "precision": settings["precision"], "failure": failure,
               "peak_scope": "model construction, training and scheduled development evaluations for this run",
               "peak_allocated_bytes": torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else 0}
    store.save_json(summary_path, summary)
    store.complete(stage, [summary_path, history_path, *historical, *query_map_paths], {"status": status, "completed_updates": completed})
    del model, optimizer, scaler, loader
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return summary


def lock_intervention(store, conditions):
    """Choose only the content-selection width from final development diagnostics."""
    relative = "configs/intervention_locked.json"
    if store.path(relative).is_file():
        return read_json(store.path(relative))
    settings = store.configuration["mqar"]["intervention"]
    evidence = []
    for seed in store.configuration["mqar"]["seeds"]:
        row = {"seed": seed}
        for method in ["dense", "window", "dilated"]:
            name = f"{method}_K16_mixed_seed{seed}"
            summary = read_json(store.path(f"metrics/mqar_training/{name}.json"))
            row[method] = summary["development_evaluations"][-1]["metrics"]["far"]["accuracy"] if summary["status"] == "complete" else None
        evidence.append(row)
    paired = [row["dense"] - row["dilated"] for row in evidence if row["dense"] is not None and row["dilated"] is not None]
    gap = sum(paired) / len(paired) if paired else None
    chosen = 64 if gap is None or gap >= settings["development_gap_threshold"] else 32
    lock = {"method": "local_content", "train_K": 16, "train_regime": "mixed", "content_k": chosen,
            "selection_rule": settings["selection"], "development_gap_threshold": settings["development_gap_threshold"],
            "observed_development_far_accuracy": evidence, "dense_minus_dilated_mean_development_gap": gap,
            "final_test_accessed": False, "prediction": {"metric": settings["prediction_metric"],
                "direction": settings["prediction_direction"], "comparator": "dilated_K16_mixed"},
            "persistent_state": "ordinary full-history keys and values for content score selection; no compressed recurrent state",
            "extra_trainable_parameters": 0, "local_heads": 2, "content_selected_heads": 2,
            "selection_cost": "all causal scores plus stable sort; no efficient sparse-kernel claim",
            "fallback": "64 positions if required development comparators failed" if gap is None else None}
    store.save_json(relative, lock)
    return lock

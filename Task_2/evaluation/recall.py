# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Query-weighted MQAR metrics and resumable paired final-test predictions."""

import math

from Task_2.common.artifacts import read_json, sha256_file
from Task_2.data.mqar import read_gzip_records, tensor_batch, write_gzip_records


def aggregate_recall(records):
    """Pool supervised positions, retaining true-distance bins and both chance baselines."""
    correct, total, nll = 0, 0, 0.
    bins = {f"{left}_{right}": {"correct": 0, "queries": 0} for left, right in [(0, 32), (32, 64), (64, 128), (128, 256)]}
    random_present = 0.
    for row in records:
        for answer, predicted, loss, distance in zip(row["answer_ids"], row["predictions"], row["query_nll"], row["distances"]):
            if not math.isfinite(loss) or not 128 <= predicted < 192:
                raise ValueError("Invalid MQAR prediction or non-finite supervised loss.")
            correct += int(answer == predicted)
            total += 1
            nll += loss
            random_present += 1 / row["K"]
            for left, right in [(0, 32), (32, 64), (64, 128), (128, 256)]:
                if left <= distance < right:
                    bins[f"{left}_{right}"]["correct"] += int(answer == predicted)
                    bins[f"{left}_{right}"]["queries"] += 1
    for cell in bins.values():
        cell["accuracy"] = cell["correct"] / cell["queries"] if cell["queries"] else None
    return {"examples": len(records), "queries": total, "correct": correct,
            "accuracy": correct / total if total else None, "mean_query_nll": nll / total if total else None,
            "chance_label_space": 1 / 64, "chance_present_values": random_present / total if total else None,
            "distance_bins": bins, "denominator": "all supervised query positions; no next-token shift"}


def predict_recall(model, records, batch_size, override_method=None):
    """Score complete equal-K batches while saving every query decision and NLL."""
    import torch
    import torch.nn.functional as functional

    model.eval()
    device = next(model.parameters()).device
    predictions = []
    with torch.inference_mode():
        for first in range(0, len(records), batch_size):
            batch = records[first:first + batch_size]
            if len({row["K"] for row in batch}) != 1:
                raise ValueError("MQAR batches must have the same number of supervised queries.")
            tensors = {key: value.to(device, non_blocking=True) for key, value in tensor_batch(batch).items()
                       if key in {"tokens", "positions", "labels"}}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(tensors["tokens"], tensors["positions"], override_method)
            logits = logits.float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("MQAR evaluation produced non-finite logits.")
            losses = functional.cross_entropy(logits.flatten(0, 1), tensors["labels"].flatten(), reduction="none").reshape(len(batch), -1).cpu().tolist()
            decisions = (logits.argmax(dim=-1) + 128).cpu().tolist()
            confidence = logits.softmax(dim=-1).amax(dim=-1).cpu().tolist()
            for row, decision, loss, probability in zip(batch, decisions, losses, confidence):
                predictions.append({key: row[key] for key in ["id", "split", "K", "regime", "query_positions",
                    "source_positions", "distances", "answer_ids"]} | {"predictions": decision, "query_nll": loss,
                    "predicted_probabilities": probability})
    return predictions


def development_metrics(model, development, settings, train_K):
    """Use only the frozen development stream for progress and intervention selection."""
    return {regime: aggregate_recall(predict_recall(model,
            [row for row in development if row["K"] == train_K and row["regime"] == regime],
            settings["execution"]["evaluation_batch_size"])) for regime in ["near", "far", "mixed"]}


def evaluate_final_condition(store, model, test_records, condition):
    """Resume checksummed 128-example shards and cover all nine final-test cells."""
    name = condition["id"]
    stage = f"mqar_test_{name}"
    summary_path = f"metrics/mqar_test/{name}.json"
    if store.completed(stage):
        return read_json(store.path(summary_path))
    batch_size = store.configuration["execution"]["evaluation_batch_size"]
    chunk_size = store.configuration["execution"]["prediction_chunk_sequences"]
    artifacts, cells = [], []
    for count in [4, 8, 16]:
        for regime in ["near", "far", "mixed"]:
            examples = [row for row in test_records if row["K"] == count and row["regime"] == regime]
            if len(examples) != 1024:
                raise ValueError("Each final MQAR cell must contain all 1,024 supplied examples.")
            completed = []
            for first in range(0, len(examples), chunk_size):
                store.check_time()
                relative = f"predictions/mqar/{name}/K{count}_{regime}_{first:04d}.jsonl.gz"
                receipt = relative.replace(".jsonl.gz", ".receipt.json")
                batch = examples[first:first + chunk_size]
                if store.path(receipt).is_file():
                    saved = read_json(store.path(receipt))
                    if not store.path(relative).is_file() or sha256_file(store.path(relative)) != saved["sha256"]:
                        raise ValueError("MQAR saved prediction shard is missing or corrupt.")
                    predictions = read_gzip_records(store.path(relative))
                else:
                    predictions = predict_recall(model, batch, batch_size,
                                                  condition["test_method"] if condition["test_only_mask"] else None)
                    write_gzip_records(store.path(relative), predictions)
                    store.backup([relative])
                    store.save_json(receipt, {"condition": name, "first": first, "examples": len(predictions),
                                              "sha256": sha256_file(store.path(relative))})
                if [row["id"] for row in predictions] != [row["id"] for row in batch]:
                    raise ValueError("MQAR prediction order/IDs differ from the frozen final stream.")
                completed.extend(predictions)
                artifacts.extend([relative, receipt])
            cells.append({"test_K": count, "regime": regime, **aggregate_recall(completed)})
            print(f"MQAR final {name}: K={count} {regime} 1024/1024", flush=True)
    summary = {**condition, "status": "complete", "examples": sum(row["examples"] for row in cells), "cells": cells}
    store.save_json(summary_path, summary)
    artifacts.append(summary_path)
    store.complete(stage, artifacts, {"status": "complete", "examples": summary["examples"]})
    return summary

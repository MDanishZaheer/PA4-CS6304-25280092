# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Token-exact likelihood scoring and HellaSwag character-normalized accuracy."""

import math
import time

import torch
import torch.nn.functional as functional

from Task_1.common.artifacts import read_jsonl, safe_perplexity
from Task_1.common.config import fingerprint
from Task_1.data.hellaswag import format_hellaswag


def encode_continuation(tokenizer, context, continuation):
    """Jointly tokenize a context and a space-prefixed ending without scoring context."""
    trailing = len(context) - len(context.rstrip())
    if trailing:
        continuation = context[-trailing:] + continuation
        context = context[:-trailing]
    context_tokens = tokenizer.encode(context, add_special_tokens=False)
    tokens = tokenizer.encode(context + continuation, add_special_tokens=False)
    if not context_tokens or tokens[:len(context_tokens)] != context_tokens:
        raise ValueError("Tokenizer changed the context prefix at the continuation boundary; scoring cannot silently shift labels.")
    if len(tokens) <= len(context_tokens):
        raise ValueError("The candidate ending has no scored tokens.")
    return tokens, len(context_tokens), len(tokens)


@torch.inference_mode()
def sequence_nll(model, token_ids, begin=1, end=None, chunk_tokens=64):
    """Sum only selected target NLLs; chunk the vocabulary head to bound memory."""
    end = len(token_ids) if end is None else end
    if not 1 <= begin < end <= len(token_ids):
        raise ValueError("Targets must have a preceding token and lie within the sequence.")
    device = next(model.parameters()).device
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
    maximum = getattr(model.config, "max_position_embeddings", 131072)
    if tokens.shape[1] > maximum:
        raise ValueError("An evaluation sequence exceeds the checkpoint context; truncation is forbidden.")
    hidden = model.model(input_ids=tokens, use_cache=False, return_dict=True).last_hidden_state
    total = torch.zeros((), device=device, dtype=torch.float64)
    for offset in range(begin, end, chunk_tokens):
        stop = min(end, offset + chunk_tokens)
        logits = model.lm_head(hidden[:, offset - 1:stop - 1]).float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation logits; retain this failed condition.")
        losses = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens[:, offset:stop].reshape(-1), reduction="none")
        total += losses.double().sum()
    return float(total), end - begin


@torch.inference_mode()
def batch_sequence_nll(model, jobs, chunk_tokens=256):
    """Score independent right-padded sequences; exclude context and every pad."""
    if not jobs or chunk_tokens < 1:
        raise ValueError("Provide scoring jobs and a positive vocabulary-head chunk size.")
    lengths = [len(tokens) for tokens, _, _ in jobs]
    maximum = getattr(model.config, "max_position_embeddings", 131072)
    if max(lengths) > maximum or any(not 1 <= begin < end <= len(tokens) for tokens, begin, end in jobs):
        raise ValueError("Invalid scored offsets/context length; truncation is forbidden.")
    device = next(model.parameters()).device
    pad_id = getattr(model.config, "pad_token_id", None) or 0
    tokens = torch.full((len(jobs), max(lengths)), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(tokens)
    scored = torch.zeros((len(jobs), max(lengths) - 1), dtype=torch.bool, device=device)
    for index, (ids, begin, end) in enumerate(jobs):
        tokens[index, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention[index, :len(ids)] = 1
        scored[index, begin - 1:end - 1] = True
    mask = attention if len(set(lengths)) > 1 else None
    hidden = model.model(input_ids=tokens, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
    rows, columns = scored.nonzero(as_tuple=True)
    total = torch.zeros(len(jobs), device=device, dtype=torch.float64)
    for offset in range(0, len(rows), chunk_tokens):
        batch_rows, batch_columns = rows[offset:offset + chunk_tokens], columns[offset:offset + chunk_tokens]
        logits = model.lm_head(hidden[batch_rows, batch_columns]).float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation logits; retain this failed condition.")
        labels = tokens[batch_rows, batch_columns + 1]
        losses = functional.cross_entropy(logits, labels, reduction="none")
        total.index_add_(0, batch_rows, losses.double())
        del logits, losses
    return [(value, end - begin) for value, (_, begin, end) in zip(total.tolist(), jobs)]


def score_sequence_jobs(model, jobs, configuration):
    """Use the frozen batch/token limits without reordering or truncating inputs."""
    plan = configuration["execution"].get("gpu_workload", {})
    batch_size = plan.get("batch_sequences", 1)
    token_limit = plan.get("padded_token_budget", batch_size * 512)
    chunk = plan.get("head_chunk_tokens", configuration["evaluation"]["head_chunk_tokens"])
    results, start = [], 0
    while start < len(jobs):
        stop = min(start + batch_size, len(jobs))
        while stop > start + 1 and (stop - start) * max(len(job[0]) for job in jobs[start:stop]) > token_limit:
            stop -= 1
        results.extend(batch_sequence_nll(model, jobs[start:stop], chunk))
        start = stop
    return results


def next_save_boundary(start, total, batch_size, interval):
    """Keep every completed batch inside the next durable checkpoint boundary."""
    return min(total, start + batch_size, (start // interval + 1) * interval)


def validate_progress(records, expected_ids, condition_id, suite):
    """Accept only an exact prefix of this condition's expected example order."""
    observed = [record["example_id"] for record in records]
    if observed != expected_ids[:len(observed)] or len(observed) > len(expected_ids):
        raise ValueError(f"Saved {suite} progress has wrong, duplicated, or reordered IDs.")
    if any(record.get("condition") != condition_id or record.get("suite") != suite for record in records):
        raise ValueError(f"Saved {suite} progress belongs to another condition.")


def evaluate_perplexity(model, windows, suite, condition_id, store):
    """Resume fixed independent windows and pool their exact token denominators."""
    relative = f"predictions/{condition_id}/{suite}.jsonl"
    predictions = read_jsonl(store.path(relative))
    identifiers = [f"{suite}:{row['source_row']}:{row['token_offset']}" for row in windows]
    validate_progress(predictions, identifiers, condition_id, suite)
    settings = store.configuration["evaluation"]
    started = time.perf_counter()
    batch_size = store.configuration["execution"].get("gpu_workload", {}).get("batch_sequences", 1)
    while len(predictions) < len(windows):
        store.check_time()
        start = len(predictions)
        stop = next_save_boundary(start, len(windows), batch_size, settings["save_every_examples"])
        batch = windows[start:stop]
        item_started = time.perf_counter()
        scores = score_sequence_jobs(model, [(row["token_ids"], *row["score_target_offsets"]) for row in batch], store.configuration)
        seconds_per_window = (time.perf_counter() - item_started) / len(batch)
        for index, (window, (nll, count)) in enumerate(zip(batch, scores), start):
            predictions.append({"example_id": identifiers[index], "condition": condition_id, "suite": suite,
                                "source_row": window["source_row"], "token_offset": window["token_offset"],
                                "token_ids_sha256": fingerprint(window["token_ids"]), "score_target_offsets": window["score_target_offsets"],
                                "scored_tokens": count, "nll_sum": nll, "scoring_seconds": seconds_per_window,
                                "scoring_seconds_scope": "batch wall time divided equally among windows"})
        if stop % settings["save_every_examples"] == 0 or stop == len(windows):
            store.save_records(relative, predictions)
            print(f"{condition_id} {suite}: {stop}/{len(windows)} windows", flush=True)
    count = sum(row["scored_tokens"] for row in predictions)
    if count != settings["scored_tokens_per_suite"]:
        raise RuntimeError("A perplexity suite did not score exactly 50,000 target tokens.")
    nll = math.fsum(row["nll_sum"] for row in predictions) / count
    summary = {"scored_tokens": count, "windows": len(predictions), "mean_nll": nll,
               "perplexity": safe_perplexity(nll), "perplexity_overflow": nll >= 709,
               "saved_successful_item_scoring_seconds": sum(row.get("scoring_seconds", 0) for row in predictions),
               "current_session_scoring_seconds": time.perf_counter() - started}
    store.save_json(f"metrics/{condition_id}/{suite}.json", summary)
    return summary


def evaluate_hellaswag(model, tokenizer, rows, condition_id, store):
    """Save all candidate scores with raw and character-normalized decisions."""
    relative = f"predictions/{condition_id}/hellaswag.jsonl"
    predictions = read_jsonl(store.path(relative))
    identifiers = [f"hellaswag:{row['source_row_index']}" for row in rows]
    validate_progress(predictions, identifiers, condition_id, "hellaswag")
    settings = store.configuration["evaluation"]
    started = time.perf_counter()
    batch_size = store.configuration["execution"].get("gpu_workload", {}).get("batch_sequences", 1)
    while len(predictions) < len(rows):
        store.check_time()
        start = len(predictions)
        stop = next_save_boundary(start, len(rows), max(1, batch_size // 4), settings["save_every_examples"])
        batch = rows[start:stop]
        item_started = time.perf_counter()
        formatted = [format_hellaswag(row["example"]) for row in batch]
        jobs = [encode_continuation(tokenizer, prompt, " " + ending)
                for prompt, endings, _ in formatted for ending in endings]
        likelihoods = score_sequence_jobs(model, jobs, store.configuration)
        seconds_per_example = (time.perf_counter() - item_started) / len(batch)
        for batch_index, (row, (prompt, endings, gold)) in enumerate(zip(batch, formatted)):
            index = start + batch_index
            candidate_scores = likelihoods[batch_index * 4:(batch_index + 1) * 4]
            scores, token_counts = [-nll for nll, _ in candidate_scores], [count for _, count in candidate_scores]
            lengths = [len(ending) for ending in endings]
            normalized = [score / length for score, length in zip(scores, lengths)]
            prediction = max(range(4), key=lambda choice: scores[choice])
            normalized_prediction = max(range(4), key=lambda choice: normalized[choice])
            predictions.append({"example_id": identifiers[index], "condition": condition_id, "suite": "hellaswag",
                            "order": row["order"], "source_row_index": row["source_row_index"],
                            "prompt": prompt, "endings": endings, "gold": gold, "log_likelihoods": scores,
                            "character_lengths": lengths, "continuation_token_counts": token_counts,
                            "character_normalized_scores": normalized, "prediction": prediction,
                            "prediction_normalized": normalized_prediction,
                            "correct": int(prediction == gold), "correct_normalized": int(normalized_prediction == gold),
                            "prompt_sha256": fingerprint({"prompt": prompt, "endings": endings}),
                            "task_version": "1.0", "format_reference": "lm-evaluation-harness v0.4.9; zero-shot; target delimiter=' '",
                            "scoring_seconds": seconds_per_example,
                            "scoring_seconds_scope": "batch wall time divided equally among examples"})
        if stop % settings["save_every_examples"] == 0 or stop == len(rows):
            store.save_records(relative, predictions)
            print(f"{condition_id} HellaSwag: {stop}/{len(rows)}", flush=True)
    count = len(predictions)
    if count != len(rows):
        raise RuntimeError("Incomplete HellaSwag evaluation.")
    summary = {"examples": count, "accuracy": sum(row["correct"] for row in predictions) / count,
               "accuracy_normalized": sum(row["correct_normalized"] for row in predictions) / count,
               "normalization": "preprocessed ending character count; excludes delimiter",
               "saved_successful_item_scoring_seconds": sum(row.get("scoring_seconds", 0) for row in predictions),
               "current_session_scoring_seconds": time.perf_counter() - started}
    store.save_json(f"metrics/{condition_id}/hellaswag.json", summary)
    return summary


@torch.inference_mode()
def development_nll(model, development, configuration):
    """Select allocations exclusively on the 32 C4 training-development windows."""
    scores = score_sequence_jobs(model, [(row["token_ids"], 1, len(row["token_ids"])) for row in development], configuration)
    total = math.fsum(nll for nll, _ in scores)
    count = sum(tokens for _, tokens in scores)
    if count != 32 * 511:
        raise ValueError("Allocation selection must use all 32 independent development sequences.")
    return {"mean_nll": total / count, "scored_tokens": count, "sequences": len(development)}

# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Gold-independent shared token selection under actual serialized cache budgets."""

import math

import torch

from Task_2.cache.packing import payload_bytes


def safeguards(length, sink=8, recent=64):
    """Keep the fixed prefix and trailing positions without duplicates."""
    return sorted(set(range(min(sink, length))) | set(range(max(0, length - recent), length)))


def select_positions(length, count, scores=None, sink=8, recent=64):
    """Fill a legal shared position budget using recency or question-only importance."""
    required = safeguards(length, sink, recent)
    if not len(required) <= count <= length:
        raise ValueError("Cache byte budget cannot satisfy its fixed safeguard positions.")
    selected = set(required)
    ranking = range(length - 1, -1, -1) if scores is None else torch.argsort(scores.detach().float().cpu(), descending=True, stable=True).tolist()
    for position in ranking:
        if len(selected) == count:
            break
        selected.add(position)
    return sorted(selected)


def selection_plan(method, geometry, budget, scores=None):
    """Maximize legal FP16 positions, explicitly counting a shared int32 index array."""
    layers, batch, heads, dimension, length = geometry
    full_position_bytes = 4 * layers * batch * heads * dimension
    header = 32
    if method == "fp16":
        return {"method": method, "positions": [], "expected_bytes": header + full_position_bytes * length,
                "bits": 16, "layout": "fp16"}
    if method in {"int4_channel", "int4_token"}:
        return {"method": method, "positions": [], "expected_bytes": header + payload_bytes(*geometry, 4),
                "bits": 4, "layout": "token" if method == "int4_token" else "channel"}
    if method in {"sink_recent", "question_aware"}:
        count = min(length, (budget - header) // (full_position_bytes + 4))
        positions = select_positions(length, count, None if method == "sink_recent" else scores)
        return {"method": method, "positions": positions, "expected_bytes": header + (full_position_bytes + 4) * count,
                "bits": 16, "layout": "fp16", "unused_budget_bytes": budget - header - (full_position_bytes + 4) * count}
    if method == "hybrid":
        minimum = len(safeguards(length))
        feasible = [count for count in range(minimum, length + 1)
                    if header + (full_position_bytes + 4) * count + payload_bytes(layers, batch, heads, dimension, length - count, 2) <= budget]
        if not feasible:
            raise ValueError("No hybrid allocation satisfies the reference bytes and mandatory safeguards.")
        count = max(feasible)
        positions = select_positions(length, count, scores)
        actual = header + (full_position_bytes + 4) * count + payload_bytes(layers, batch, heads, dimension, length - count, 2)
        return {"method": method, "positions": positions, "expected_bytes": actual, "bits": 2,
                "layout": "channel", "unused_budget_bytes": budget - actual,
                "low_precision_positions": length - count,
                "fp16_low_stream_residual": (length - count) % 64}
    raise ValueError(f"Unknown cache policy: {method}")


@torch.inference_mode()
def question_importance(model, prefix_cache, suffix_ids, observation_max=64):
    """Score past keys using actual rotary question queries; no gold annotations enter."""
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    from Task_2.cache.adapter import clone_dynamic_cache, model_step

    cache = clone_dynamic_cache(prefix_cache)
    length = prefix_cache.get_seq_length()
    device = suffix_ids.device
    positions = torch.arange(length, length + suffix_ids.shape[-1], device=device)[None]
    dummy = model.model.embed_tokens(suffix_ids)
    cos, sin = model.model.rotary_emb(dummy, positions)
    captured, handles = {}, []
    for index, layer in enumerate(model.model.layers):
        def save_query(module, arguments, output, layer_index=index):
            captured[layer_index] = output.detach()
        handles.append(layer.self_attn.q_proj.register_forward_hook(save_query))
    try:
        logits, _ = model_step(model, suffix_ids, cache, length)
    finally:
        for handle in handles:
            handle.remove()
    scores = torch.zeros(length, device=device, dtype=torch.float32)
    for index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        head_dimension = attention.head_dim
        query = captured[index].reshape(suffix_ids.shape[0], suffix_ids.shape[1], -1, head_dimension).transpose(1, 2)
        query, _ = apply_rotary_pos_emb(query, query, cos, sin)
        query = query[..., -observation_max:, :]
        keys = cache.layers[index].keys
        repetitions = query.shape[1] // keys.shape[1]
        keys = keys.repeat_interleave(repetitions, dim=1)
        attention_scores = (query.float() @ keys.float().transpose(-1, -2)) / math.sqrt(head_dimension)
        query_positions = positions[0, -query.shape[-2]:]
        allowed = torch.arange(keys.shape[-2], device=device)[None] <= query_positions[:, None]
        probabilities = attention_scores.masked_fill(~allowed, -torch.inf).softmax(dim=-1)
        scores += probabilities[..., :length].mean(dim=(0, 1, 2))
    return scores / len(model.model.layers), logits.detach(), cache

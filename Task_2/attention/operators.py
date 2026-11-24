# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Exact causal references and the MQAR attention mechanisms."""

import math

import torch
import torch.nn.functional as functional


def causal_mask(length, device, window=None, dilation=None):
    """Return True for legal key positions, including the current query position."""
    positions = torch.arange(length, device=device)
    offsets = positions[:, None] - positions[None, :]
    allowed = offsets >= 0
    if window is not None:
        allowed &= offsets < window
    if dilation is not None:
        allowed &= (offsets % dilation == 0) & (offsets <= 252)
    return allowed


def dense_attention(query, key, value, allowed=None):
    """Materialize FP32 scores/softmax; FP16 inputs use FP32 accumulation explicitly."""
    length = query.shape[-2]
    mask = causal_mask(length, query.device) if allowed is None else allowed
    scores = (query.float() @ key.float().transpose(-1, -2)) / math.sqrt(query.shape[-1])
    scores = scores.masked_fill(~mask, -torch.inf)
    valid = mask.any(dim=-1, keepdim=True)
    # Empty rows have defined zero output and zero gradients, rather than NaNs.
    safe = torch.where(valid, scores, torch.zeros_like(scores))
    weights = torch.softmax(safe, dim=-1).masked_fill(~mask, 0)
    return (weights @ value.float()).to(query.dtype)


class TiledAttention(torch.autograd.Function):
    """Online softmax forward and recomputed blockwise backward, without L-by-L state."""

    @staticmethod
    def forward(context, query, key, value, tile_size):
        length, dimension = query.shape[-2:]
        output = torch.empty_like(query, dtype=torch.float32)
        log_normalizer = torch.empty(query.shape[:-1], device=query.device, dtype=torch.float32)
        scale = dimension ** -.5
        for first in range(0, length, tile_size):
            stop = min(first + tile_size, length)
            q = query[..., first:stop, :].float()
            maximum = torch.full(q.shape[:-1] + (1,), -torch.inf, device=q.device)
            normalizer = torch.zeros_like(maximum)
            accumulator = torch.zeros_like(q)
            query_ids = torch.arange(first, stop, device=q.device)[:, None]
            for left in range(0, stop, tile_size):
                right = min(left + tile_size, length)
                scores = (q @ key[..., left:right, :].float().transpose(-1, -2)) * scale
                allowed = torch.arange(left, right, device=q.device)[None, :] <= query_ids
                scores.masked_fill_(~allowed, -torch.inf)
                next_maximum = torch.maximum(maximum, scores.amax(dim=-1, keepdim=True))
                correction = torch.exp(maximum - next_maximum)
                probabilities = torch.exp(scores - next_maximum)
                accumulator = correction * accumulator + probabilities @ value[..., left:right, :].float()
                normalizer = correction * normalizer + probabilities.sum(dim=-1, keepdim=True)
                maximum = next_maximum
            output[..., first:stop, :] = accumulator / normalizer
            log_normalizer[..., first:stop] = (maximum + normalizer.log()).squeeze(-1)
        context.save_for_backward(query, key, value, output, log_normalizer)
        context.tile_size = tile_size
        return output.to(query.dtype)

    @staticmethod
    def backward(context, gradient):
        query, key, value, output, log_normalizer = context.saved_tensors
        tile, length = context.tile_size, query.shape[-2]
        dq, dk, dv = [torch.zeros_like(item, dtype=torch.float32) for item in [query, key, value]]
        scale = query.shape[-1] ** -.5
        for first in range(0, length, tile):
            stop = min(first + tile, length)
            q, do = query[..., first:stop, :].float(), gradient[..., first:stop, :].float()
            delta = (output[..., first:stop, :] * do).sum(dim=-1, keepdim=True)
            query_ids = torch.arange(first, stop, device=query.device)[:, None]
            for left in range(0, stop, tile):
                right = min(left + tile, length)
                k, v = key[..., left:right, :].float(), value[..., left:right, :].float()
                scores = (q @ k.transpose(-1, -2)) * scale
                scores.masked_fill_(torch.arange(left, right, device=query.device)[None, :] > query_ids, -torch.inf)
                probabilities = torch.exp(scores - log_normalizer[..., first:stop, None])
                derivative = probabilities * (do @ v.transpose(-1, -2) - delta)
                dq[..., first:stop, :] += (derivative @ k) * scale
                dk[..., left:right, :] += (derivative.transpose(-1, -2) @ q) * scale
                dv[..., left:right, :] += probabilities.transpose(-1, -2) @ do
        return dq.to(query.dtype), dk.to(key.dtype), dv.to(value.dtype), None


def tiled_attention(query, key, value, tile_size=256):
    """Compute exact causal attention with FP32 running statistics and accumulators."""
    if tile_size < 1 or query.shape != key.shape or key.shape != value.shape:
        raise ValueError("Tiled attention expects equal Q/K/V shapes and a positive tile size.")
    return TiledAttention.apply(query, key, value, tile_size)


def linear_attention(query, key, value, chunk_size=64, epsilon=1e-6):
    """Apply causal ELU+1 kernel attention with chunked prefix states and FP32 sums."""
    q, k, v = functional.elu(query.float()) + 1, functional.elu(key.float()) + 1, value.float()
    state = torch.zeros(q.shape[:-2] + (q.shape[-1], v.shape[-1]), device=q.device)
    normalizer = torch.zeros(q.shape[:-2] + (q.shape[-1],), device=q.device)
    outputs = []
    for first in range(0, query.shape[-2], chunk_size):
        stop = min(first + chunk_size, query.shape[-2])
        keys, values = k[..., first:stop, :], v[..., first:stop, :]
        prefix = (keys[..., :, :, None] * values[..., :, None, :]).cumsum(dim=-3) + state[..., None, :, :]
        sums = keys.cumsum(dim=-2) + normalizer[..., None, :]
        current_q = q[..., first:stop, :]
        numerator = (current_q[..., :, :, None] * prefix).sum(dim=-2)
        denominator = (current_q * sums).sum(dim=-1, keepdim=True) + epsilon
        outputs.append(numerator / denominator)
        state, normalizer = prefix[..., -1, :, :], sums[..., -1, :]
    return torch.cat(outputs, dim=-2).to(query.dtype)


def attention_operator(query, key, value, method, window=64, content_k=64, chunk_size=64, epsilon=1e-6):
    """Change only the attention computation, keeping all learned projections shared."""
    length = query.shape[-2]
    if method == "dense":
        return functional.scaled_dot_product_attention(query, key, value, is_causal=True)
    if method == "linear":
        return linear_attention(query, key, value, chunk_size, epsilon)
    local = causal_mask(length, query.device, window=window)
    if method == "window":
        return functional.scaled_dot_product_attention(query, key, value, attn_mask=local)
    if method == "dilated":
        dilated = causal_mask(length, query.device, dilation=4)
        mask = torch.stack([local, local, dilated, dilated])
        return functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    if method == "local_content":
        # Content selection computes all causal scores; neither sparse time nor
        # a reduced persistent KV state is claimed for this correctness operator.
        local_output = functional.scaled_dot_product_attention(query[:, :2], key[:, :2], value[:, :2], attn_mask=local)
        q, k, v = query[:, 2:].float(), key[:, 2:].float(), value[:, 2:].float()
        causal = causal_mask(length, query.device)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        scores.masked_fill_(~causal, -torch.inf)
        count = min(content_k, length)
        # Column order gives deterministic tie breaking without changing scores.
        indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :count]
        selected = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, indices, True)
        weights = torch.softmax(scores.masked_fill(~selected, -torch.inf), dim=-1)
        content_output = (weights @ v).to(query.dtype)
        return torch.cat([local_output, content_output], dim=1)
    raise ValueError(f"Unknown attention mechanism: {method}")


def theoretical_resources(method, batch, heads, length, dimension, dtype_bytes=2, content_k=64):
    """Separate mathematical attended pairs, score work, persistent state and masks."""
    dense_pairs = length * (length + 1) // 2
    local_pairs = sum(min(i + 1, 64) for i in range(length))
    dilated_pairs = sum(min(i // 4 + 1, 64) for i in range(length))
    content_pairs = sum(min(i + 1, content_k) for i in range(length))
    if method == "linear":
        attended = None
        state = batch * heads * (dimension * dimension + dimension) * 4
        implementation = "chunked FP32 prefix tensors; autograd retains training intermediates"
        mask_bytes = 0
    else:
        per_head = {"dense": dense_pairs, "window": local_pairs,
                    "dilated": (local_pairs + dilated_pairs) / 2,
                    "local_content": (local_pairs + content_pairs) / 2}[method]
        attended = int(batch * heads * per_head)
        per_head_positions = ([min(64, length)] * heads if method == "window" else
                              [min(64, length)] * 2 + [min(253, length)] * 2 if method == "dilated" else
                              [min(64, length)] * 2 + [length] * 2 if method == "local_content" else [length] * heads)
        state = 2 * batch * sum(per_head_positions) * dimension * dtype_bytes
        mask_bytes = 0 if method == "dense" else length * length * (heads if method == "dilated" else 1)
        implementation = "SDPA" if method == "dense" else "dense boolean mask + SDPA" if method != "local_content" else "dense scores, stable sort and dense selected attention"
    return {"method": method, "batch": batch, "heads": heads, "length": length, "head_dimension": dimension,
            "attended_pairs": attended, "content_selection_score_pairs": batch * 2 * length * length if method == "local_content" else 0,
            "content_causal_candidate_pairs": batch * 2 * dense_pairs if method == "local_content" else 0,
            "explicit_content_score_bytes": batch * 2 * length * length * 4 if method == "local_content" else 0,
            "full_sort_index_bytes": batch * 2 * length * length * 8 if method == "local_content" else 0,
            "theoretical_incremental_state_bytes": state, "explicit_mask_bytes": mask_bytes,
            "mask_scope": "boolean mask passed to SDPA; construction/selection temporaries appear in measured peaks",
            "implementation": implementation, "state_note": "hypothetical minimal incremental per-head buffers/recurrent state, not measured full-sequence training allocation"}

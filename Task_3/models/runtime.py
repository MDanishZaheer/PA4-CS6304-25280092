# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Pinned frozen Qwen models and a cache-position-explicit forward interface."""

import copy
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def synchronize(device):
    """Make elapsed measurements include completed CUDA work."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def load_tokenizers(configuration, cache):
    """Check the complete vocabulary mapping before formatting any experiment."""
    settings = configuration["models"]
    tokenizers = [AutoTokenizer.from_pretrained(settings[name], revision=settings[name + "_revision"],
        cache_dir=str(cache), use_fast=True) for name in ["teacher", "student"]]
    if tokenizers[0].get_vocab() != tokenizers[1].get_vocab():
        raise ValueError("Teacher/student tokenizer token-to-ID mappings differ.")
    return tokenizers


def load_model(configuration, which, cache, device="cuda"):
    """Load exactly the pinned original teacher or student with ordinary SDPA."""
    settings = configuration["models"]
    model = AutoModelForCausalLM.from_pretrained(settings[which], revision=settings[which + "_revision"],
        cache_dir=str(cache), torch_dtype=torch.float16, attn_implementation="sdpa").to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def eos_ids(model):
    """Use the checkpoint's complete fixed EOS set, rather than only one tokenizer ID."""
    value = model.generation_config.eos_token_id
    return {value} if isinstance(value, int) else set(value or [])


def step(model, ids, cache=None, all_logits=False):
    """Append contiguous physical/absolute positions and project only needed hidden states."""
    length = cache.get_seq_length() if cache is not None else 0
    positions = torch.arange(length, length + ids.shape[-1], device=ids.device)
    output = model.model(input_ids=ids, past_key_values=cache, use_cache=True,
        cache_position=positions, position_ids=positions[None], return_dict=True)
    hidden = output.last_hidden_state if all_logits else output.last_hidden_state[:, -1:]
    logits = model.lm_head(hidden).float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Model logits contain non-finite values.")
    return logits, output.past_key_values


def crop_cache(cache, length):
    """Retain only committed tokens and check every layer after a speculative rollback."""
    cache.crop(length)
    if cache.get_seq_length() != length or any(layer.get_seq_length() != length for layer in cache.layers):
        raise RuntimeError("Cache rollback did not preserve the declared physical length.")


def clone_cache(cache):
    """Make benchmark setup independent of cache mutations inside timed operations."""
    return copy.deepcopy(cache)


def timed(operation, device):
    """Return a synchronized elapsed duration and the original operation result."""
    synchronize(device)
    start = time.perf_counter()
    result = operation()
    synchronize(device)
    return time.perf_counter() - start, result


def peak_memory(device):
    """Separate actual allocated/reserved GPU bytes from logical cache size."""
    if torch.device(device).type != "cuda":
        return {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}

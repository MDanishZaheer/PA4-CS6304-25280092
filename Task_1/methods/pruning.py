# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Deterministic per-output-row magnitude/Wanda and structured 2:4 pruning."""

import math

import torch


def pruning_scores(weight, input_norm=None):
    """Calculate magnitude or Wanda scores from the unchanged FP16 weights."""
    if weight.ndim != 2:
        raise ValueError("Pruning expects a two-dimensional projection matrix.")
    scores = weight.float().abs()
    if input_norm is not None:
        if input_norm.ndim != 1 or input_norm.numel() != weight.shape[1]:
            raise ValueError("Wanda activation norms must match input columns.")
        scores.mul_(input_norm.to(device=weight.device, dtype=torch.float32)[None, :])
    return scores


def row_prune(weight, zero_count, input_norm=None):
    """Remove exactly zero_count entries with row quotas differing by at most one."""
    rows, columns = weight.shape
    if not 0 <= zero_count <= weight.numel():
        raise ValueError("Requested zero count is outside the eligible matrix.")
    scores = pruning_scores(weight, input_norm)
    # Existing zeros must not become extra, unbudgeted zeros in score ties.
    scores[weight == 0] = -torch.inf
    ordered = torch.argsort(scores, dim=1, stable=True)
    per_row, remainder = divmod(int(zero_count), rows)
    quotas = torch.full((rows,), per_row, device=weight.device, dtype=torch.long)
    quotas[:remainder] += 1
    if torch.any((weight == 0).sum(dim=1) > quotas):
        raise ValueError("Existing row zeros exceed this pruning budget.")
    selected = torch.arange(columns, device=weight.device)[None, :] < quotas[:, None]
    mask = torch.zeros_like(weight, dtype=torch.bool)
    mask.scatter_(1, ordered, selected.expand(rows, -1))
    compressed = weight.clone()
    compressed[mask] = 0
    if int(mask.sum()) != zero_count or int((compressed == 0).sum()) != zero_count:
        raise RuntimeError("Pruning failed its exact selected/effective-zero budget check.")
    return compressed, mask


def structured_prune(weight, input_norm):
    """Keep the two largest Wanda scores in each contiguous input group of four."""
    if weight.shape[1] % 4:
        raise ValueError("2:4 pruning requires input columns divisible by four.")
    scores = pruning_scores(weight, input_norm).reshape(weight.shape[0], -1, 4)
    scores[weight.reshape_as(scores) == 0] = -torch.inf
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep.scatter_(-1, order[..., :2], True)
    mask = ~keep.reshape_as(weight)
    compressed = weight.clone()
    compressed[mask] = 0
    if not torch.all(keep.sum(dim=-1) == 2):
        raise RuntimeError("Invalid 2:4 structural mask.")
    return compressed, mask


def distribution_summary(values):
    """Preserve robust score quantiles and a fixed log-scale histogram."""
    flattened = values.detach().float().flatten()
    probabilities = torch.tensor([0, .01, .1, .25, .5, .75, .9, .99, 1.0], device=values.device)
    # Quantile has a size limit in some PyTorch versions; deterministic thinning
    # is for diagnostics only, never for ranking or mask construction.
    stride = max(1, math.ceil(flattened.numel() / 1000000))
    quantiles = torch.quantile(flattened[::stride], probabilities).cpu().tolist()
    positive = flattened[flattened > 0]
    edges = torch.linspace(-12, 8, 41, device=values.device)
    log_values = positive.log10().clamp(-12, 8)
    histogram = torch.histc(log_values, bins=40, min=-12, max=8).long().cpu().tolist()
    return {"quantile_probabilities": probabilities.cpu().tolist(), "quantiles": quantiles,
            "quantile_stride": stride, "log10_histogram_edges": edges.cpu().tolist(),
            "log10_histogram_counts": histogram, "zero_count": int((flattened == 0).sum()),
            "elements": flattened.numel()}

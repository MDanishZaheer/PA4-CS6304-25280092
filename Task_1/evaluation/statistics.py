# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Paired bootstrap comparisons with whole-window pooled perplexity."""

import math

import numpy as np


def align_records(left, right):
    """Align common IDs and reject duplicate or context-inconsistent pairs."""
    left_map = {row["example_id"]: row for row in left}
    right_map = {row["example_id"]: row for row in right}
    if len(left_map) != len(left) or len(right_map) != len(right):
        raise ValueError("Paired resampling requires unique example/window identifiers.")
    identifiers = [row["example_id"] for row in left if row["example_id"] in right_map]
    if not identifiers:
        raise ValueError("No common examples are available for a paired comparison.")
    pairs = [(left_map[key], right_map[key]) for key in identifiers]
    for first, second in pairs:
        for key in ["prompt_sha256", "token_ids_sha256", "score_target_offsets", "scored_tokens", "gold"]:
            if first.get(key) != second.get(key):
                raise ValueError(f"Paired examples have inconsistent {key}.")
    return pairs


def paired_bootstrap(left, right, metric, resamples=2000, seed=6304):
    """Return method-minus-reference 95% intervals over paired examples/windows."""
    pairs = align_records(left, right)
    generator = np.random.default_rng(seed)
    n = len(pairs)
    if metric in {"accuracy", "accuracy_normalized"}:
        key = "correct" if metric == "accuracy" else "correct_normalized"
        differences = np.array([a[key] - b[key] for a, b in pairs], dtype=np.float64)
        observed = float(differences.mean())
        sampled = np.empty(resamples)
        for repeat in range(resamples):
            sampled[repeat] = differences[generator.integers(0, n, n)].mean()
        return {"metric": metric, "difference": observed, "ci_low": float(np.quantile(sampled, .025)),
                "ci_high": float(np.quantile(sampled, .975)), "paired_examples": n,
                "resamples": resamples, "unit": "fraction; multiply by 100 for percentage points"}
    counts = np.array([a["scored_tokens"] for a, _ in pairs], dtype=np.float64)
    first = np.array([a["nll_sum"] for a, _ in pairs], dtype=np.float64)
    second = np.array([b["nll_sum"] for _, b in pairs], dtype=np.float64)
    def statistic(indices):
        """Pool token NLL before exponentiating; never average window perplexities."""
        denominator = counts[indices].sum()
        mean_first = first[indices].sum() / denominator
        mean_second = second[indices].sum() / denominator
        if metric == "mean_nll":
            return mean_first - mean_second
        if metric != "perplexity":
            raise ValueError("Unknown paired metric.")
        if max(mean_first, mean_second) >= 709:
            return math.nan
        return math.exp(mean_first) - math.exp(mean_second)
    observed = statistic(np.arange(n))
    sampled = np.array([statistic(generator.integers(0, n, n)) for _ in range(resamples)])
    finite = bool(np.isfinite(sampled).all() and math.isfinite(observed))
    return {"metric": metric, "difference": observed if finite else None,
            "ci_low": float(np.quantile(sampled, .025)) if finite else None,
            "ci_high": float(np.quantile(sampled, .975)) if finite else None,
            "overflow": not finite, "paired_windows": n, "scored_tokens": int(counts.sum()),
            "resamples": resamples, "unit": "nats/token" if metric == "mean_nll" else "perplexity",
            "resampling_unit": "whole fixed evaluation windows, paired"}

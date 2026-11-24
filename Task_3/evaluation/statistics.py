# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Prompt-paired uncertainty and the stated speculative cost model."""

import math

import numpy as np


def paired_interval(first, second=None, resamples=2000, seed=6304, geometric=False):
    """Resample matched prompt IDs, after averaging repetitions within each prompt."""
    keys = sorted(first)
    if second is not None and set(keys) != set(second):
        raise ValueError("A paired comparison must contain the same prompt IDs.")
    if not keys:
        return {"estimate": None, "ci_low": None, "ci_high": None, "paired_prompts": 0, "resamples": resamples}
    values = np.array([first[key] - second[key] if second is not None else first[key] for key in keys], dtype=float)
    if not np.isfinite(values).all() or geometric and np.any(values <= 0):
        raise ValueError("Bootstrap inputs are not finite/legal.")
    if geometric:
        values = np.log(values)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    transform = np.exp if geometric else lambda value: value
    low, high = np.quantile(samples, [.025, .975])
    return {"estimate": float(transform(values.mean())), "ci_low": float(transform(low)), "ci_high": float(transform(high)),
            "paired_prompts": len(values), "resamples": resamples, "repetition_unit": "two repetitions averaged within prompt before resampling"}


def predicted_speedup(alpha, gamma, c, verification_cost):
    """Use the assignment's approximately constant-overlap cost model."""
    return (1 + sum(alpha ** index for index in range(1, gamma + 1))) / (gamma * c + verification_cost)


def break_even_acceptance(gamma, c, verification_cost):
    """Return the boundary above which projected speedup exceeds one, or nonexistence."""
    denominator = gamma * c + verification_cost
    if denominator < 1:
        return 0.
    if denominator >= gamma + 1:
        return None
    lower, upper = 0., 1.
    for _ in range(60):
        middle = (lower + upper) / 2
        if 1 + sum(middle ** index for index in range(1, gamma + 1)) > denominator:
            upper = middle
        else:
            lower = middle
    return upper


def deployment_break_even(upfront_seconds, seconds_saved_per_request):
    """Never invent a positive amortization threshold when requests are slower."""
    return math.ceil(upfront_seconds / seconds_saved_per_request) if seconds_saved_per_request > 0 else None

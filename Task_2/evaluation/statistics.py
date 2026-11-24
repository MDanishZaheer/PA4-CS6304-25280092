# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Paired prompt bootstrap for the primary KV-cache comparisons."""

import numpy as np


KV_PAIRS = [(name, "fp16") for name in ["sink_recent", "question_aware", "int4_token", "int4_channel", "hybrid"]]
KV_PAIRS += [("question_aware", "sink_recent"), ("int4_channel", "int4_token"),
             ("hybrid", "question_aware"), ("hybrid", "int4_channel")]


def paired_accuracy_interval(left, right, resamples=2000, seed=6304):
    """Resample the complete paired prompt set; never run separate ten-case cell tests."""
    first, second = {row["id"]: row for row in left}, {row["id"]: row for row in right}
    if len(first) != len(left) or len(second) != len(right) or set(first) != set(second):
        raise ValueError("Paired KV comparison requires the same unique prompt IDs.")
    identifiers = [row["id"] for row in left]
    if not identifiers:
        raise ValueError("Paired bootstrap has no examples.")
    for key in identifiers:
        if first[key]["prompt_sha256"] != second[key]["prompt_sha256"]:
            raise ValueError("A paired cache comparison changed the encoded prompt.")
    differences = np.array([int(first[key].get("correct", False)) - int(second[key].get("correct", False))
                            for key in identifiers], dtype=np.float64)
    generator = np.random.default_rng(seed)
    sampled = differences[generator.integers(0, len(differences), (resamples, len(differences)))].mean(axis=1)
    return {"difference": float(differences.mean()), "ci_low": float(np.quantile(sampled, .025)),
            "ci_high": float(np.quantile(sampled, .975)), "paired_prompts": len(differences),
            "resamples": resamples, "unit": "fraction; multiply by 100 for percentage points",
            "failed_runs_counted_as_incorrect": True}

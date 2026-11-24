# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Calibration-only constrained sparsity and packed-byte precision allocation."""

import math

import numpy as np

from Task_1.methods.quantization import packed_projection_bytes


def allocate_sparsity(profiles, target, temperature, minimum=0.10, maximum=0.85):
    """Water-fill clipped inverse sensitivities and meet the exact integer budget."""
    names = sorted(profiles)
    sizes = np.array([profiles[name]["elements"] for name in names], dtype=np.int64)
    sensitivity = np.array([profiles[name]["pruning_sensitivity"] for name in names], dtype=np.float64)
    sensitivity = np.maximum(sensitivity, 1e-12)
    sensitivity /= np.median(sensitivity)
    inverse = sensitivity ** (-temperature)
    total = int(sizes.sum())
    required = int(round(target * total))
    lower = np.ceil(minimum * sizes).astype(np.int64)
    upper = np.floor(maximum * sizes).astype(np.int64)
    if not int(lower.sum()) <= required <= int(upper.sum()):
        raise ValueError("The global sparsity is infeasible under allocation bounds.")
    left, right = 0.0, maximum / float(inverse.min()) + 1
    for _ in range(100):
        midpoint = (left + right) / 2
        fractions = np.clip(midpoint * inverse, minimum, maximum)
        if np.dot(fractions, sizes) < required:
            left = midpoint
        else:
            right = midpoint
    desired = np.clip(((left + right) / 2) * inverse, minimum, maximum) * sizes
    counts = np.minimum(np.maximum(np.floor(desired).astype(np.int64), lower), upper)
    difference = required - int(counts.sum())
    direction = 1 if difference > 0 else -1
    priorities = np.argsort(-(desired - counts) if direction == 1 else desired - counts, kind="stable")
    while difference:
        changed = False
        for index in priorities:
            if difference == 0:
                break
            candidate = counts[index] + direction
            if lower[index] <= candidate <= upper[index]:
                counts[index] = candidate
                difference -= direction
                changed = True
        if not changed:
            raise RuntimeError("Integer sparsity allocation could not satisfy the budget.")
    allocations = {name: {"zero_count": int(counts[index]), "sparsity": float(counts[index] / sizes[index])}
                   for index, name in enumerate(names)}
    if sum(value["zero_count"] for value in allocations.values()) != required:
        raise RuntimeError("Sparsity allocation has an unmatched budget.")
    return {"target_sparsity": target, "temperature": temperature, "zero_count": required,
            "eligible_elements": total, "allocation_unit": "individual_projection",
            "allocations": allocations}


def quantization_cost(profile, bits, configuration, awq=True):
    """Use the same accounting function for allocation and final reporting."""
    folded_down = awq and profile["name"].endswith("down_proj")
    return packed_projection_bytes(*profile["shape"], bits, configuration,
                                   compensation=awq and not folded_down)["packed_bytes"]


def allocate_precision(profiles, reference_bits, uncompressed_bytes, configuration):
    """Greedily buy the largest calibration-distortion reduction per extra byte."""
    names = sorted(profiles)
    bits_allowed = configuration["quantization"]["allowed_bits"]
    assignment = {name: min(bits_allowed) for name in names}
    eligible_elements = sum(profiles[name]["elements"] for name in names)
    budget = uncompressed_bytes + sum(quantization_cost(profiles[name], reference_bits, configuration) for name in names)
    spent = uncompressed_bytes + sum(quantization_cost(profiles[name], assignment[name], configuration) for name in names)
    if spent > budget:
        raise ValueError("Minimum allowed precision already exceeds the reference budget.")
    steps = []
    while True:
        best = None
        for name in names:
            profile = profiles[name]
            current = assignment[name]
            current_error = profile["awq"][str(current)]["relative_mse"]
            for proposed in bits_allowed:
                if proposed <= current:
                    continue
                extra = quantization_cost(profile, proposed, configuration) - quantization_cost(profile, current, configuration)
                reduction = (current_error - profile["awq"][str(proposed)]["relative_mse"]) * profile["elements"] / eligible_elements
                if reduction <= 0 or extra <= 0 or spent + extra > budget:
                    continue
                key = (reduction / extra, reduction, -extra, name, proposed)
                if best is None or key > best[0]:
                    best = (key, name, proposed, extra, reduction)
        if best is None:
            break
        _, name, proposed, extra, reduction = best
        steps.append({"projection": name, "from_bits": assignment[name], "to_bits": proposed,
                      "extra_bytes": extra, "weighted_error_reduction": reduction})
        assignment[name] = proposed
        spent += extra
    if spent > budget:
        raise RuntimeError("Mixed precision exceeded its actual packed-byte budget.")
    if len(set(assignment.values())) < 2:
        # Uniform equal-sensitivity fixtures are valid but must not be mislabeled.
        allocation_kind = "uniform_outcome_of_adaptive_rule"
    else:
        allocation_kind = "mixed_precision"
    return {"reference_bits": reference_bits, "reference_budget_bytes": int(budget),
            "packed_bytes": int(spent), "unused_bytes": int(budget - spent),
            "uncompressed_bytes": int(uncompressed_bytes), "allocation_kind": allocation_kind,
            "selection_data": "C4 calibration only; no held-out labels or scores",
            "allocations": assignment, "upgrade_steps": steps}

# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Declare the complete mandatory comparison matrix before viewing test scores."""


def experiment_conditions():
    """Return all 18 required conditions, including the shared baseline and 2:4."""
    conditions = [{"id": "fp16", "family": "fp16", "label": "Unchanged FP16", "hellaswag_count": 2000}]
    for family in ["P1", "P2"]:
        for percentage in [25, 50, 75]:
            conditions.append({"id": f"{family.lower()}_s{percentage}", "family": family,
                               "sparsity": percentage / 100, "label": f"{family} {percentage}%",
                               "hellaswag_count": 2000 if percentage == 50 else 512})
    for percentage in [50, 75]:
        conditions.append({"id": f"p3_s{percentage}", "family": "P3", "sparsity": percentage / 100,
                           "label": f"P3 adaptive {percentage}%", "hellaswag_count": 2000 if percentage == 50 else 512})
    for family in ["Q1", "Q2"]:
        for bits in [8, 4, 3]:
            conditions.append({"id": f"{family.lower()}_int{bits}", "family": family, "bits": bits,
                               "label": f"{family} INT{bits}", "hellaswag_count": 2000 if bits in {4, 3} else 512})
    for bits in [4, 3]:
        conditions.append({"id": f"q3_budget_int{bits}", "family": "Q3", "reference_bits": bits,
                           "label": f"Q3 <= Q2 INT{bits} bytes", "hellaswag_count": 2000})
    conditions.append({"id": "wanda_2of4", "family": "structured", "sparsity": .5,
                       "label": "Wanda 2:4 diagnostic", "hellaswag_count": 512})
    return conditions


def adaptive_pairs():
    """Declare required paired confidence-interval comparisons in advance."""
    return [("p3_s50", "p2_s50"), ("p3_s75", "p2_s75"),
            ("q3_budget_int4", "q2_int4"), ("q3_budget_int3", "q2_int3")]

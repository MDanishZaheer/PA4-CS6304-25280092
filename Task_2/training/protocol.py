# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Enumerate all 42 independent training runs and 54 final-evaluation variants."""


def training_conditions(configuration, include_intervention=True):
    """Never continue one architecture or seed from another run's learned weights."""
    specifications = [("dense", count, "mixed") for count in [4, 8, 16]] + [("dense", 16, "near")]
    specifications += [("window", count, "near") for count in [4, 8, 16]] + [("window", 16, "mixed")]
    specifications += [("linear", count, "mixed") for count in [4, 8, 16]]
    specifications += [("dilated", count, "mixed") for count in [4, 16]]
    if include_intervention:
        specifications += [("local_content", 16, "mixed")]
    return [{"id": f"{method}_K{count}_{regime}_seed{seed}", "method": method, "train_K": count,
             "train_regime": regime, "seed": seed} for method, count, regime in specifications
            for seed in configuration["mqar"]["seeds"]]


def evaluation_conditions(configuration):
    """Apply test-only W64 masks to every learned dense control as separate variants."""
    conditions = []
    for training in training_conditions(configuration):
        conditions.append({**training, "training_id": training["id"], "test_method": training["method"], "test_only_mask": False})
        if training["method"] == "dense":
            conditions.append({**training, "id": training["id"] + "_test_window64", "training_id": training["id"],
                               "test_method": "window", "test_only_mask": True})
    return conditions

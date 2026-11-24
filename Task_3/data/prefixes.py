# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Pure-Python frozen prefix-position selection shared with model-free replay."""

import random


def selected_positions(length, prompt_id, bank, configuration):
    """Select at most 32 short positions and a separate 16-position long diagnostic."""
    rng = random.Random(f"PA4-banks-v1:6304:{prompt_id}:{bank}")
    positions = []
    for lower, upper, count in configuration["banks"]["short_strata"] + configuration["banks"]["long_strata"]:
        candidates = list(range(lower, min(upper, length) + 1))
        positions.extend(sorted(rng.sample(candidates, min(count, len(candidates)))))
    return positions

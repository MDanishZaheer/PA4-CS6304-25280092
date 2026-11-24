# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Deterministic category round-robin schedules with exact scored-token quotas."""

import hashlib
import random


def rollout_seed(seed, source_id, presentation, purpose):
    """Derive stable isolated seeds without relying on Python's randomized hash."""
    value = f"PA4-Task3:{seed}:{source_id}:{presentation}:{purpose}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little") % (2 ** 63 - 1)


class PromptSchedule:
    """Cycle each category's fixed permutation, skipping only an exhausted quota."""

    def __init__(self, prompts, seed, cycle, state=None):
        self.pools = {domain: [row for row in prompts if row["domain"] == domain] for domain in ["math", "short"]}
        for domain, rows in self.pools.items():
            random.Random(f"PA4-Task3-order:{seed}:{domain}").shuffle(rows)
            if not rows:
                raise ValueError("Both training quota categories must contain prompts.")
        self.cycle = cycle
        self.state = state or {"phase": 0, "cursors": {"math": 0, "short": 0}, "presentations": 0}

    def next(self, remaining):
        """Select a source ID, reusing the same permutation after category exhaustion."""
        if not any(value > 0 for value in remaining.values()):
            raise StopIteration
        while True:
            domain = self.cycle[self.state["phase"] % len(self.cycle)]
            self.state["phase"] += 1
            if remaining[domain] > 0:
                break
        index = self.state["cursors"][domain]
        row = self.pools[domain][index % len(self.pools[domain])]
        self.state["cursors"][domain] += 1
        presentation = self.state["presentations"]
        self.state["presentations"] += 1
        return row, presentation

# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Vectorized multi-step exact/incorrect Markov samplers with a 27-category joint test."""

import itertools

import numpy as np


TARGET = np.array([[.7, .2, .1], [.15, .65, .2], [.2, .2, .6]])
DRAFT = np.array([[.15, .2, .65], [.5, .1, .4], [.6, .3, .1]])


def categorical(distributions, generator):
    """Vectorize categorical draws without altering the provided probabilities."""
    return (generator.random(len(distributions))[:, None] > np.cumsum(distributions, axis=1)).sum(axis=1).clip(0, 2)


def joint_trials(trials, gamma, seed, wrong_residual=False):
    """Sample length-three sequences, allowing rejection and bonus paths across rounds."""
    rng = np.random.default_rng(seed)
    outputs = np.full((trials, 3), -1, dtype=np.int8)
    lengths = np.zeros(trials, dtype=np.int8)
    while np.any(lengths < 3):
        for count in range(3):
            members = np.flatnonzero(lengths == count)
            if not len(members):
                continue
            previous = outputs[members, count - 1] if count else np.zeros(len(members), dtype=np.int8)
            m = min(gamma, 3 - count)
            proposed, ps, qs = [], [], []
            state = previous.copy()
            for _ in range(m):
                p, q = TARGET[state], DRAFT[state]
                token = categorical(q, rng)
                proposed.append(token)
                ps.append(p)
                qs.append(q)
                state = token
            active = np.ones(len(members), dtype=bool)
            for index in range(m):
                local = np.flatnonzero(active)
                if not len(local):
                    break
                tokens = proposed[index][local]
                p, q = ps[index][local], qs[index][local]
                accepted = rng.random(len(local)) < np.minimum(1., p[np.arange(len(local)), tokens] / q[np.arange(len(local)), tokens])
                good, bad = local[accepted], local[~accepted]
                outputs[members[good], count + index] = proposed[index][good]
                lengths[members[good]] += 1
                if len(bad):
                    residual = ps[index][bad] if wrong_residual else np.maximum(ps[index][bad] - qs[index][bad], 0)
                    residual = residual / residual.sum(axis=1, keepdims=True)
                    outputs[members[bad], count + index] = categorical(residual, rng)
                    lengths[members[bad]] += 1
                    active[bad] = False
            if count + m < 3:
                local = np.flatnonzero(active)
                outputs[members[local], count + m] = categorical(TARGET[proposed[-1][local]], rng)
                lengths[members[local]] += 1
    indices = outputs[:, 0] * 9 + outputs[:, 1] * 3 + outputs[:, 2]
    return np.bincount(indices, minlength=27)


def validate_markov(trials=100000, seed=6304):
    """Require non-rejection of the exact sampler and rejection of the wrong residual."""
    from scipy.stats import chisquare
    sequences = list(itertools.product(range(3), repeat=3))
    expected = np.array([TARGET[0, a] * TARGET[a, b] * TARGET[b, c] for a, b, c in sequences])
    rows, conditions = [], []
    for gamma in [2, 4]:
        for wrong in [False, True]:
            observed = joint_trials(trials, gamma, seed, wrong)
            statistic, pvalue = chisquare(observed, expected * trials)
            passed = pvalue < .01 if wrong else pvalue >= .01
            conditions.append({"gamma": gamma, "negative_control": wrong, "trials": trials,
                "chi_square": float(statistic), "p_value": float(pvalue), "passed": bool(passed),
                "minimum_expected_count": float((expected * trials).min())})
            rows.extend({"gamma": gamma, "negative_control": wrong, "sequence": "".join(map(str, sequence)),
                "target_probability": float(probability), "observed_probability": int(count) / trials,
                "expected_count": float(probability * trials), "observed_count": int(count)}
                for sequence, probability, count in zip(sequences, expected, observed))
    if not all(row["passed"] and row["minimum_expected_count"] >= 10 for row in conditions):
        raise RuntimeError(f"Required joint-sequence sampler check failed: {conditions}")
    return {"status": "passed", "seed": seed, "sequence_length": 3, "target_transition": TARGET.tolist(),
            "draft_transition": DRAFT.tolist(), "initial_state": 0, "conditions": conditions, "joint_categories": rows}

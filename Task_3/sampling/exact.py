# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Student-written exact stochastic speculative sampling with one pending cache token."""

import time

import torch

from Task_3.models.runtime import crop_cache, eos_ids, peak_memory, step, synchronize, timed


def normalize_prompt(token_ids, device, attention_mask=None):
    """Support one unpadded sequence or contiguous left/right padding, never masked holes."""
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=device)
    if ids.ndim not in {1, 2} or ids.ndim == 2 and ids.shape[0] != 1:
        raise ValueError("The exact sampler operates on one sequence at a time.")
    ids = ids.reshape(1, -1)
    if attention_mask is not None:
        mask = torch.as_tensor(attention_mask, dtype=torch.bool, device=device).reshape(-1)
        if mask.numel() != ids.numel() or not mask.any():
            raise ValueError("Invalid prompt padding mask.")
        kept = mask.nonzero().flatten()
        if int(kept[-1] - kept[0] + 1) != kept.numel():
            raise ValueError("Only contiguous prompt tokens with edge padding are supported.")
        ids = ids[:, kept]
    if not ids.numel():
        raise ValueError("An autoregressive prompt must contain a token.")
    return ids


def probabilities(logits):
    """Use the full common vocabulary at temperature one and without top-k/top-p filtering."""
    distribution = logits.float().softmax(dim=-1).reshape(-1)
    if not torch.isfinite(distribution).all() or distribution.sum() <= 0:
        raise FloatingPointError("Invalid full-vocabulary sampling distribution.")
    return distribution


def sample(distribution, generator):
    """Sample a categorical ID with the explicit per-presentation RNG."""
    return int(torch.multinomial(distribution, 1, generator=generator).item())


def positive_residual(target, draft):
    """Use the normalized positive difference; silently substituting p is incorrect."""
    difference = (target.double() - draft.double()).clamp_min(0)
    mass = difference.sum()
    if not torch.isfinite(mass) or mass <= 0:
        raise FloatingPointError("A rejected proposal has no positive residual probability mass.")
    return (difference / mass).float()


@torch.inference_mode()
def target_generate(model, token_ids, maximum, seed, ignore_eos=False, attention_mask=None):
    """Ordinary target-distribution sampling using the same formatting and stopping rules."""
    device = next(model.parameters()).device
    ids = normalize_prompt(token_ids, device, attention_mask)
    generator = torch.Generator(device=device).manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    prefill, (logits, cache) = timed(lambda: step(model, ids), device)
    output, calls = [], 0
    decode_start = time.perf_counter()
    for index in range(maximum):
        token = sample(probabilities(logits[0, -1]), generator)
        output.append(token)
        if not ignore_eos and token in eos_ids(model):
            break
        if index + 1 < maximum:
            logits, cache = step(model, torch.tensor([[token]], device=device), cache)
            calls += 1
    synchronize(device)
    decode = time.perf_counter() - decode_start
    ended = bool(output and output[-1] in eos_ids(model) and not ignore_eos)
    return {"token_ids": output, "new_tokens": len(output), "eos": ended, "truncated": len(output) == maximum and not ended,
        "ignore_eos": ignore_eos, "generation_seed": seed, "prefill_seconds": prefill, "decode_seconds": decode,
        "elapsed_seconds": time.perf_counter() - started, "target_prefill_passes": 1, "target_decode_passes": calls,
        "target_passes": calls + 1, "draft_forward_passes": 0, "rounds": [], **peak_memory(device)}


@torch.inference_mode()
def speculative_generate(target, draft, token_ids, maximum, gamma, seed, ignore_eos=False,
                         attention_mask=None, round_check=None):
    """Verify proposals in one target pass, preserving the exact target law.

    After each round both caches contain the returned context except its last
    token. That token is pending: the next target verification consumes it together
    with the proposals. Draft consumption is charged to the proposal timer. This
    avoids an extra target forward pass for residual/bonus advancement.
    """
    if gamma < 1 or maximum < 1:
        raise ValueError("Positive proposal lengths and output caps are required.")
    device = next(target.parameters()).device
    ids = normalize_prompt(token_ids, device, attention_mask)
    generator = torch.Generator(device=device).manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    target_prefill, (first_target, target_cache) = timed(lambda: step(target, ids), device)
    draft_prefill, (draft_logits, draft_cache) = timed(lambda: step(draft, ids), device)
    if first_target.shape[-1] != draft_logits.shape[-1]:
        raise ValueError("Target and draft full vocabulary sizes differ.")
    output, rounds, pending = [], [], None
    stops = eos_ids(target)
    decode_start = time.perf_counter()
    while len(output) < maximum:
        base = ids.shape[-1] + len(output)
        proposal_limit = min(gamma, maximum - len(output))
        synchronize(device)
        phase = time.perf_counter()
        draft_calls = 0
        if pending is not None:
            draft_logits, draft_cache = step(draft, torch.tensor([[pending]], device=device), draft_cache)
            draft_calls += 1
        proposals, qs = [], []
        for _ in range(proposal_limit):
            q = probabilities(draft_logits[0, -1])
            token = sample(q, generator)
            proposals.append(token)
            qs.append(q)
            draft_logits, draft_cache = step(draft, torch.tensor([[token]], device=device), draft_cache)
            draft_calls += 1
            if not ignore_eos and token in stops:
                break
        synchronize(device)
        proposal_time = time.perf_counter() - phase
        verification_input = ([pending] if pending is not None else []) + proposals
        verification_time, (verified, target_cache) = timed(lambda: step(target,
            torch.tensor([verification_input], device=device), target_cache, all_logits=True), device)
        target_distributions = verified[0, :len(proposals)] if pending is not None else torch.cat(
            [first_target[0, -1:], verified[0, :len(proposals) - 1]], dim=0)
        synchronize(device)
        phase = time.perf_counter()
        accepted, examined, emitted, rejected = 0, 0, [], False
        for index, token in enumerate(proposals):
            p, q = probabilities(target_distributions[index]), qs[index]
            ratio = torch.minimum(torch.ones((), device=device), p[token] / q[token])
            examined += 1
            if float(torch.rand((), device=device, generator=generator)) < float(ratio):
                emitted.append(token)
                accepted += 1
                if not ignore_eos and token in stops:
                    break
            else:
                emitted.append(sample(positive_residual(p, q), generator))
                rejected = True
                break
        bonus = False
        if not rejected and accepted == len(proposals) and len(output) + len(emitted) < maximum:
            if ignore_eos or not emitted or emitted[-1] not in stops:
                emitted.append(sample(probabilities(verified[0, -1]), generator))
                bonus = True
        synchronize(device)
        sampling_time = time.perf_counter() - phase
        phase = time.perf_counter()
        output.extend(emitted)
        committed = ids.shape[-1] + len(output) - 1
        crop_cache(target_cache, committed)
        crop_cache(draft_cache, committed)
        pending = emitted[-1]
        synchronize(device)
        maintenance_time = time.perf_counter() - phase
        record = {"round": len(rounds), "output_offset": len(output) - len(emitted), "proposed_tokens": proposals,
            "emitted_tokens": emitted, "proposed": len(proposals), "examined": examined, "accepted": accepted,
            "accepted_prefix_length": accepted, "bonus": bonus, "rejected": rejected,
            "target_verification_passes": 1, "target_verification_input_tokens": len(verification_input),
            "draft_forward_passes": draft_calls, "proposal_seconds": proposal_time, "verification_seconds": verification_time,
            "sampling_seconds": sampling_time, "maintenance_seconds": maintenance_time,
            "cache_length": committed, "pending_token": pending}
        rounds.append(record)
        if round_check is not None:
            round_check(output, target_cache, draft_cache, pending)
        if not ignore_eos and output[-1] in stops:
            break
    synchronize(device)
    decode = time.perf_counter() - decode_start
    ended = bool(output[-1] in stops and not ignore_eos)
    totals = {key: sum(row[key] for row in rounds) for key in ["proposed", "examined", "accepted", "draft_forward_passes",
             "proposal_seconds", "verification_seconds", "sampling_seconds", "maintenance_seconds"]}
    return {"token_ids": output, "new_tokens": len(output), "eos": ended, "truncated": len(output) == maximum and not ended,
        "ignore_eos": ignore_eos, "generation_seed": seed, "gamma": gamma, "rounds": rounds, **totals,
        "accepted_examined_fraction": totals["accepted"] / totals["examined"],
        "accepted_proposed_fraction": totals["accepted"] / totals["proposed"],
        "mean_accepted_prefix": totals["accepted"] / len(rounds), "target_verification_passes": len(rounds),
        "target_prefill_passes": 1, "draft_prefill_passes": 1, "target_passes": 1 + len(rounds),
        "target_passes_per_output_token": (1 + len(rounds)) / len(output),
        "draft_forward_passes_including_prefill": 1 + totals["draft_forward_passes"],
        "prefill_seconds": target_prefill + draft_prefill, "target_prefill_seconds": target_prefill,
        "draft_prefill_seconds": draft_prefill, "decode_seconds": decode,
        "host_overhead_seconds": max(0., decode - sum(totals[key] for key in
            ["proposal_seconds", "verification_seconds", "sampling_seconds", "maintenance_seconds"])),
        "elapsed_seconds": time.perf_counter() - started, "final_cache_length": target_cache.get_seq_length(),
        "final_pending_token": pending, "cache_convention": "returned context minus exactly one pending last token",
        **peak_memory(device)}

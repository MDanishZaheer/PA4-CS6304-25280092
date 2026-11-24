# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Freeze shared and own-policy rollout banks, then score identical prefix IDs."""

import time

import torch
from torch.nn import functional

from Task_3.common.artifacts import read_json, sha256_file
from Task_3.common.config import bank_names, draft_ids
from Task_3.data.prepare import read_gzip_records, write_gzip_records
from Task_3.data.prefixes import selected_positions
from Task_3.models.runtime import synchronize
from Task_3.sampling.exact import speculative_generate, target_generate
from Task_3.training.distill import response_logits


COMMON_BANKS = ["teacher", "original_draft", "exact_spec_original_gamma4"]


def generate_banks(store, teacher, get_student, prompts, checkpoint_ids):
    """Materialize all seven banks without supplying held-out labels to generation."""
    artifacts = []
    for bank_index, bank in enumerate(bank_names(store.configuration)):
        producer = "teacher" if bank == "teacher" else "original" if bank in COMMON_BANKS else bank.removeprefix("own_")
        student = None if producer == "teacher" else get_student(producer)
        for prompt_index, prompt in enumerate(prompts):
            store.check_time()
            relative = f"banks/{bank}/{prompt['id']}.json"
            if store.path(relative).is_file():
                saved = read_json(store.path(relative))
                if saved["producer_checkpoint"] != checkpoint_ids[producer] or saved["prompt_sha256"] != prompt["prompt_sha256"]:
                    raise ValueError("A frozen prefix bank belongs to another prompt/checkpoint.")
                artifacts.append(relative)
                continue
            cap = store.configuration["banks"]["generate_math_cap" if prompt["domain"] == "math" else "generate_short_cap"]
            seed = 6304 + 10007 * prompt_index + (0 if bank == "teacher" else 123 if bank == "original_draft" else 246 if bank == "exact_spec_original_gamma4" else 1000 + bank_index)
            result = speculative_generate(teacher, student, prompt["prompt_token_ids"], cap, 4, seed) if bank == "exact_spec_original_gamma4" else target_generate(
                teacher if producer == "teacher" else student, prompt["prompt_token_ids"], cap, seed)
            positions = selected_positions(result["new_tokens"], prompt["id"], bank, store.configuration)
            row = {"prompt_id": prompt["id"], "prompt_sha256": prompt["prompt_sha256"], "workload": prompt["workload"],
                "bank": bank, "producer": producer, "producer_checkpoint": checkpoint_ids[producer],
                "response_token_ids": result["token_ids"], "selected_positions": positions,
                "prefix_ids": [f"{bank}:{prompt['id']}:p{position:03d}" for position in positions],
                "generation": result, "law": "target" if bank in {"teacher", "exact_spec_original_gamma4"} else producer,
                "materialization": "generated here because the supplied data-only checkout contains no materialized prefix bank"}
            store.save_json(relative, row)
            artifacts.append(relative)
            if (prompt_index + 1) % 20 == 0:
                print(f"Frozen {bank}: {prompt_index + 1}/{len(prompts)}", flush=True)
    store.complete("banks", artifacts, {"banks": len(bank_names(store.configuration)), "prompts_per_bank": len(prompts)})
    return {"banks": len(bank_names(store.configuration)), "prompts_per_bank": len(prompts)}


def probability_metrics(target_logits, student_logits):
    """Compute five full-vocabulary quantities at exactly the same response prefixes."""
    lp, lq = functional.log_softmax(target_logits.float(), dim=-1), functional.log_softmax(student_logits.float(), dim=-1)
    p, q = lp.exp(), lq.exp()
    tv = .5 * (p - q).abs().sum(dim=-1)
    return {"forward_kl": (p * (lp - lq)).sum(dim=-1).clamp_min(0),
        "reverse_kl": (q * (lq - lp)).sum(dim=-1).clamp_min(0), "tv": tv,
        "expected_acceptance": 1 - tv, "greedy_agreement": (target_logits.argmax(-1) == student_logits.argmax(-1)).float(),
        "target_next_token": target_logits.argmax(-1), "student_next_token": student_logits.argmax(-1)}


def position_bin(position):
    """Keep long-response diagnostics distinct from the first-64-position banks."""
    return "1_16" if position <= 16 else "17_32" if position <= 32 else "33_64" if position <= 64 else "65_192"


@torch.inference_mode()
def evaluate_alignment(store, teacher, get_student, prompts, tokenizer=None):
    """Score every draft on every frozen bank, including cross-student own-policy exposure."""
    artifacts, prefix_count = [], 0
    device = next(teacher.parameters()).device
    token_text = {}
    def decode_token(token):
        if token not in token_text:
            token_text[token] = tokenizer.decode([token], skip_special_tokens=False) if tokenizer is not None else str(token)
        return token_text[token]
    for bank in bank_names(store.configuration):
        for prompt in prompts:
            store.check_time()
            bank_path = f"banks/{bank}/{prompt['id']}.json"
            frozen = read_json(store.path(bank_path))
            bank_sha = sha256_file(store.path(bank_path))
            selected = frozen["selected_positions"]
            response = frozen["response_token_ids"]
            indices = torch.tensor([position - 1 for position in selected], device=device)
            missing = [name for name in draft_ids(store.configuration)
                       if not store.path(f"alignment/{bank}/{name}/{prompt['id']}.jsonl.gz").is_file()]
            if missing:
                synchronize(device)
                start = time.perf_counter()
                target_logits = response_logits(teacher, prompt["prompt_token_ids"], response, 0, max(selected)).index_select(0, indices)
                for name in missing:
                    student = get_student(name)
                    student_logits = response_logits(student, prompt["prompt_token_ids"], response, 0, max(selected)).index_select(0, indices)
                    values = {key: value.cpu().tolist() for key, value in probability_metrics(target_logits, student_logits).items()}
                    rows = [{"prefix_id": pid, "prompt_id": prompt["id"], "prompt_sha256": prompt["prompt_sha256"],
                        "bank": bank, "draft": name, "workload": prompt["workload"], "position": position,
                        "position_bin": position_bin(position), "producer_checkpoint": frozen["producer_checkpoint"],
                        "prefix_bank_sha256": bank_sha,
                        "target_next_token_text": decode_token(values["target_next_token"][index]),
                        "student_next_token_text": decode_token(values["student_next_token"][index]),
                        **{key: value[index] for key, value in values.items()}}
                        for index, (position, pid) in enumerate(zip(selected, frozen["prefix_ids"]))]
                    relative = f"alignment/{bank}/{name}/{prompt['id']}.jsonl.gz"
                    write_gzip_records(store.path(relative), rows)
                    store.backup([relative])
                    del student_logits
                del target_logits
                synchronize(device)
                store.save_json(f"metrics/alignment_cost/{bank}/{prompt['id']}.json", {"seconds": time.perf_counter() - start,
                    "new_draft_conditions": missing, "prefix_positions": len(selected), "teacher_projection_scope": "full response prefixes before selecting saved positions"})
            for name in draft_ids(store.configuration):
                relative = f"alignment/{bank}/{name}/{prompt['id']}.jsonl.gz"
                rows = read_gzip_records(store.path(relative))
                if [row["prefix_id"] for row in rows] != frozen["prefix_ids"]:
                    raise ValueError("Drafts were not evaluated on identical saved prefix IDs.")
                prefix_count += len(rows)
                artifacts.append(relative)
    store.complete("alignment", artifacts, {"drafts": len(draft_ids(store.configuration)), "prefix_draft_records": prefix_count})
    return {"prefix_draft_records": prefix_count, "banks": len(bank_names(store.configuration)), "drafts": len(draft_ids(store.configuration))}

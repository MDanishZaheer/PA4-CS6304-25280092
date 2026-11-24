# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Tokenize the frozen retrieval prompts while isolating gold spans from selection."""

import re
from collections import Counter
from pathlib import Path

from Task_2.common.artifacts import read_json, sha256_file
from Task_2.common.config import fingerprint
from Task_2.data.mqar import read_gzip_records, write_gzip_records


def extract_code(text):
    """Use the first standalone five-digit sequence without consulting the answer."""
    match = re.search(r"(?<!\d)\d{5}(?!\d)", text)
    return match.group(0) if match else None


def load_retrieval_cases(source):
    """Require all 180 original prompts and verify gold spans for evaluation only."""
    cases = read_gzip_records(Path(source) / "data/kv_retrieval/kv_retrieval_cases.jsonl.gz")
    expected = Counter({(length, position, distractors): 10 for length in [2048, 4096]
                        for position in ["start", "middle", "end"] for distractors in [0, 2, 8]})
    actual = Counter((row["target_tokens"], row["position"], row["distractors"]) for row in cases)
    if actual != expected or len({row["id"] for row in cases}) != 180:
        raise ValueError("Retrieval set differs from the frozen 180-case factorial design.")
    for row in cases:
        text = row["user_text"]
        if text[row["evidence_char_start"]:row["evidence_char_end"]] != row["evidence_text"]:
            raise ValueError("The staff evidence character span no longer matches its text.")
        if not re.fullmatch(r"\d{5}", row["code"]) or text.count("QUESTION_BEGIN\n") != 1:
            raise ValueError("Malformed fixed retrieval code or question boundary.")
    return cases


def tokenize_case(case, tokenizer, system_prompt):
    """Split one jointly encoded chat prompt; offsets map gold spans only afterward."""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": case["user_text"]}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if rendered.count(case["user_text"]) != 1:
        raise ValueError("Chat template must preserve the exact staff user text once.")
    user_start = rendered.index(case["user_text"])
    boundary = user_start + case["user_text"].index("QUESTION_BEGIN\n")
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    offsets, ids = encoded["offset_mapping"], encoded["input_ids"]
    split = next((index for index, (_, end) in enumerate(offsets) if end > boundary), None)
    if split is None or split < 128 or split == len(ids):
        raise ValueError("Invalid pre-question/suffix token boundary.")
    start, end = user_start + case["evidence_char_start"], user_start + case["evidence_char_end"]
    evidence = [index for index, (left, right) in enumerate(offsets) if right > start and left < end]
    if not evidence or max(evidence) >= split:
        raise ValueError("Evidence-token annotation must lie in the budgeted pre-question context.")
    return {"id": case["id"], "target_tokens": case["target_tokens"], "position": case["position"],
            "distractors": case["distractors"], "code": case["code"], "token_ids": ids,
            "prefix_length": split, "suffix_length": len(ids) - split, "actual_prompt_tokens": len(ids),
            "evidence_token_positions": evidence, "evidence_char_start": case["evidence_char_start"],
            "evidence_char_end": case["evidence_char_end"], "evidence_text": case["evidence_text"],
            "prompt_sha256": fingerprint({"rendered_prompt": rendered, "token_ids": ids}),
            "messages": messages}


def prepare_retrieval(store):
    """Save an immutable tokenizer/model revision before any model download or scoring."""
    settings = store.configuration["kv"]
    revision_path = "splits/kv_model_revision.json"
    if store.path(revision_path).is_file():
        revision = read_json(store.path(revision_path))
    else:
        requested = settings["revision"]
        if requested is None:
            from huggingface_hub import HfApi
            requested = HfApi().model_info(settings["model"]).sha
        if not re.fullmatch(r"[0-9a-f]{40}", requested):
            raise ValueError("The pretrained model/tokenizer must be pinned to an immutable SHA.")
        revision = {"model": settings["model"], "revision": requested, "tokenizer_revision": requested}
        store.save_json(revision_path, revision)
    relative = "splits/retrieval_tokenized.jsonl.gz"
    selection_path = "splits/retrieval_selection.json"
    existing = read_gzip_records(store.path(relative)) if store.path(relative).is_file() else None
    if existing is not None and store.path(selection_path).is_file():
        selection = read_json(store.path(selection_path))
        if len(existing) != 180 or selection["tokenized_sha256"] != sha256_file(store.path(relative)):
            raise ValueError("Saved tokenized retrieval prompts are incomplete or corrupt.")
        if [row["id"] for row in existing] != [row["id"] for row in load_retrieval_cases(store.paths["resources"])]:
            raise ValueError("Saved tokenized retrieval IDs differ from the staff order.")
        return existing, revision
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(settings["model"], revision=revision["revision"],
        cache_dir=store.paths["data"] / "huggingface", use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Retrieval span mapping requires the checkpoint's fast tokenizer offsets.")
    cases = [tokenize_case(case, tokenizer, settings["system_prompt"]) for case in load_retrieval_cases(store.paths["resources"])]
    if existing is not None and existing != cases:
        raise ValueError("Uncommitted tokenized retrieval file differs from the pinned source/template.")
    write_gzip_records(store.path(relative), cases)
    store.backup([relative])
    store.save_json(selection_path, {
        "examples": len(cases), "model": settings["model"], "revision": revision["revision"],
        "ids": [row["id"] for row in cases], "actual_prompt_tokens": [row["actual_prompt_tokens"] for row in cases],
        "tokenized_sha256": sha256_file(store.path(relative)),
        "scoring": "greedy; fixed first standalone five-digit extractor; all 180 prompts retained",
        "gold_usage": "span retention and answer scoring only; never passed to question-aware ranking",
        "system_prompt": settings["system_prompt"], "chat_template": tokenizer.chat_template,
    })
    return cases, revision

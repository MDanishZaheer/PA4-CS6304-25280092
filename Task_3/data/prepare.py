# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Materialize only the supplied GSM8K IDs and unchanged short-form prompts."""

import gzip
import json
from pathlib import Path

from Task_3.common.artifacts import read_json, read_jsonl, sha256_file
from Task_3.common.config import fingerprint


SUPPLIED_FILES = ["data/public_manifests/gsm8k_train_ids.jsonl", "data/public_manifests/gsm8k_test_ids.jsonl",
                  "data/short_form/short_train.jsonl", "data/short_form/short_test.jsonl"]


def write_gzip_records(path, records):
    """Atomically replace a complete compressed record collection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            for row in records:
                compressed.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode())
    temporary.replace(path)


def read_gzip_records(path):
    """Reject incomplete gzip streams instead of treating them as partial success."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_supplied_inputs(source):
    """Verify relevant original checksums without downloading the data repository."""
    source = Path(source)
    expected = read_json(source / "SHA256SUMS.json")
    for relative in SUPPLIED_FILES:
        if sha256_file(source / relative) != expected[relative]:
            raise ValueError(f"The supplied Task 3 file changed: {relative}")
    train = read_jsonl(source / SUPPLIED_FILES[2])
    test = read_jsonl(source / SUPPLIED_FILES[3])
    if len(train) != 300 or len(test) != 60 or len({x["id"] for x in train + test}) != 360:
        raise ValueError("Short-form IDs or counts differ from the supplied pool.")
    familiar = [x for x in test if x["split"] == "familiar"]
    shifted = [x for x in test if x["split"] == "shifted"]
    # Staff family numbers are local to each template namespace: shifted family
    # 0 is distinct from familiar/train family 0. The unchanged source hashes
    # freeze the actual templates; comparing bare integers would reject the data.
    if len(familiar) != 30 or len(shifted) != 30 or {x["input"] for x in train} & {x["input"] for x in test}:
        raise ValueError("Short-form split sizes or distinct prompt templates differ.")
    return {relative: expected[relative] for relative in SUPPLIED_FILES}


def format_prompt(tokenizer, text):
    """Use the same non-thinking user-only chat template for every condition."""
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)


def prepare_data(store, target_tokenizer, draft_tokenizer):
    """Freeze source revisions before downloading and retain all 800/120 prompt IDs."""
    source, settings = store.paths["resources"], store.configuration
    hashes = verify_supplied_inputs(source)
    revision = {"gsm8k": settings["data"]["gsm8k_revision"], "target": settings["models"]["teacher_revision"],
                "draft": settings["models"]["student_revision"]}
    store.save_json("splits/source_revisions.json", revision)
    files = ["splits/train_prompts.jsonl.gz", "splits/test_prompts.jsonl.gz"]
    if all(store.path(name).is_file() for name in files):
        train, test = [read_gzip_records(store.path(name)) for name in files]
    else:
        from datasets import load_dataset
        cache = str(store.paths["data"] / "huggingface")
        data = {split: load_dataset("openai/gsm8k", "main", split=split, revision=revision["gsm8k"], cache_dir=cache)
                for split in ["train", "test"]}
        collections = {}
        for split in ["train", "test"]:
            ids = read_jsonl(source / f"data/public_manifests/gsm8k_{split}_ids.jsonl")
            count = 500 if split == "train" else 60
            if len(ids) != count or [row["order"] for row in ids] != list(range(count)):
                raise ValueError("GSM8K row selection differs from the supplied order.")
            rows = []
            for item in ids:
                example = dict(data[split][item["row_index"]])
                rows.append({"id": f"GSM8K-{split}-{item['order']:04d}", "domain": "math", "split": split,
                    "workload": "math", "text": example["question"], "answer": example["answer"],
                    "source_row_index": item["row_index"], "source_revision": revision["gsm8k"],
                    "source_sha256": fingerprint(example), "family": None})
            for item in read_jsonl(source / f"data/short_form/short_{split}.jsonl"):
                rows.append({"id": item["id"], "domain": "short", "split": split, "workload": item["split"],
                    "text": item["input"], "answer": item["answer"], "family": item["family"],
                    "template_family": f"{'shifted' if item['split'] == 'shifted' else 'familiar'}:{item['family']}",
                    "source_sha256": fingerprint(item)})
            collections[split] = rows
        train, test = collections["train"], collections["test"]
    if len(train) != 800 or len(test) != 120 or {x["text"] for x in train} & {x["text"] for x in test}:
        raise ValueError("Training/test prompts overlap or their fixed counts differ.")
    for row in train + test:
        target_ids = format_prompt(target_tokenizer, row["text"])
        draft_ids = format_prompt(draft_tokenizer, row["text"])
        if target_ids != draft_ids:
            raise ValueError(f"Target/student chat token IDs differ: {row['id']}")
        rendered = target_tokenizer.apply_chat_template([{"role": "user", "content": row["text"]}], tokenize=False, add_generation_prompt=True)
        if rendered != draft_tokenizer.apply_chat_template([{"role": "user", "content": row["text"]}], tokenize=False, add_generation_prompt=True):
            raise ValueError("Target and student rendered chat templates differ.")
        if row.get("prompt_token_ids", target_ids) != target_ids:
            raise ValueError("A saved prompt encoding changed under the pinned tokenizer.")
        row.update(prompt_token_ids=target_ids, prompt_sha256=fingerprint({"text": rendered, "ids": target_ids}))
    for relative, rows in zip(files, [train, test]):
        if not store.path(relative).exists():
            write_gzip_records(store.path(relative), rows)
            store.backup([relative])
    manifest = {"supplied_hashes": hashes, "source_revisions": revision, "train_ids": [x["id"] for x in train],
        "test_ids": [x["id"] for x in test], "train_prompts": 800, "test_prompts": 120,
        "tokenizer_compatibility_prompts": 920, "chat_format": settings["data"]["format"],
        "data_hashes": {name: sha256_file(store.path(name)) for name in files}}
    store.save_json("splits/data_manifest.json", manifest)
    store.complete("data", [*files, "splits/data_manifest.json", "splits/source_revisions.json"], {"train": 800, "test": 120})
    return train, test, manifest

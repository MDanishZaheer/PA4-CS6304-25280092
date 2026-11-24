# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Use the unchanged staff generator and frozen MQAR development/test streams."""

import gzip
import importlib.util
import json
from collections import Counter
from pathlib import Path

from Task_2.common.artifacts import read_json, sha256_file


SUPPLIED_FILES = ["scripts/mqar_generator.py", "data/mqar/mqar_development.jsonl.gz",
                  "data/mqar/mqar_test.jsonl.gz", "data/kv_retrieval/kv_retrieval_cases.jsonl.gz"]


def verify_supplied_inputs(source):
    """Check only Task 2 assets against the supplied checksum list."""
    source = Path(source)
    expected = read_json(source / "SHA256SUMS.json")
    actual = {}
    for relative in SUPPLIED_FILES:
        path = source / relative
        if not path.is_file() or sha256_file(path) != expected.get(relative):
            raise ValueError(f"Supplied Task 2 input failed its course checksum: {relative}")
        actual[relative] = expected[relative]
    return actual


def load_generator(source):
    """Import the supplied module without editing or replacing its generation recipe."""
    path = Path(source) / "scripts/mqar_generator.py"
    specification = importlib.util.spec_from_file_location("staff_pa4_mqar_generator", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def read_gzip_records(path):
    """Read complete gzip JSONL and reject truncated or malformed data."""
    with Path(path).open("rb") as handle:
        with gzip.open(handle, "rt", encoding="utf-8") as compressed:
            return [json.loads(line) for line in compressed if line.strip()]


def write_gzip_records(path, records):
    """Publish a deterministic complete JSONL shard before registering its checksum."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as compressed:
            for row in records:
                compressed.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    temporary.replace(path)


def validate_example(example, generator=None):
    """Check exact ranges, query alignment, unique pairs, filler and true distances."""
    if generator is not None:
        generator.validate(example)
    tokens, count = example["tokens"], example["K"]
    query = example["query_positions"]
    sources, answers, distances = example["source_positions"], example["answer_ids"], example["distances"]
    if not len(sources) == len(answers) == len(distances) == count:
        raise ValueError("Every MQAR query must retain its source, answer and distance.")
    if len(tokens) != 256 or len(query) != count or len(set(query)) != count:
        raise ValueError("MQAR length/query-count invariant failed.")
    if len(set(tokens[:2 * count:2])) != count or len(set(tokens[1:2 * count:2])) != count:
        raise ValueError("MQAR source keys/values must each be distinct.")
    if not all(64 <= token < 128 for token in tokens[:2 * count:2]) or not all(128 <= token < 192 for token in tokens[1:2 * count:2]):
        raise ValueError("MQAR source key/value ranges differ from the supplied task.")
    if set(sources) != set(range(0, 2 * count, 2)):
        raise ValueError("Each source must be queried once.")
    for position, source, answer, distance in zip(query, sources, answers, distances):
        if tokens[position] != tokens[source] or tokens[source + 1] != answer or position - source != distance:
            raise ValueError("MQAR labels must be at query positions with actual source distances.")
    if any(not 2 <= token <= 63 for position, token in enumerate(tokens)
           if position >= 2 * count and position not in query):
        raise ValueError("An MQAR filler position contains a key or answer value.")
    limits = {"near": (1, 63), "far": (128, 250), "mixed": (1, 250)}
    lower, upper = limits[example["regime"]]
    if not all(lower <= distance <= upper for distance in distances):
        raise ValueError("MQAR query is outside its declared distance regime.")
    return True


def load_evaluation_stream(source, split):
    """Validate all supplied cells without modifying, filtering or resampling them."""
    if split not in {"development", "test"}:
        raise ValueError("Only development and final test files are frozen assets.")
    records = read_gzip_records(Path(source) / f"data/mqar/mqar_{split}.jsonl.gz")
    expected_per_cell = 256 if split == "development" else 1024
    counts = Counter((row["K"], row["regime"]) for row in records)
    if counts != Counter({(k, regime): expected_per_cell for k in [4, 8, 16] for regime in ["near", "far", "mixed"]}):
        raise ValueError("Frozen MQAR stream is incomplete.")
    if len({row["id"] for row in records}) != len(records) or any(row["split"] != split for row in records):
        raise ValueError("MQAR duplicate IDs or split contamination detected.")
    generator = load_generator(source)
    for row in records:
        validate_example(row, generator)
    far_distances = {distance for row in records if row["regime"] == "far" for distance in row["distances"]}
    if len(far_distances) < 10 or not any(distance % 4 for distance in far_distances):
        raise ValueError("Far examples must contain varied, off-grid source distances.")
    return records


class TrainingStream:
    """Map optimizer-update positions to reproducible, disjoint staff training IDs."""

    def __init__(self, source, count, regime, seed, updates=4500, batch_size=32):
        self.source = str(source)
        self.count, self.regime, self.seed = count, regime, seed
        self.total = updates * batch_size
        self.start_index = (seed - 6304) * self.total
        self.generator = None

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        if self.generator is None:
            self.generator = load_generator(self.source)
        return self.generator.make_example("train", self.count, self.regime, self.start_index + index)


def tensor_batch(records):
    """Collate all query labels at their own positions; never shift labels by one."""
    import torch
    return {"tokens": torch.tensor([row["tokens"] for row in records], dtype=torch.long),
            "positions": torch.tensor([row["query_positions"] for row in records], dtype=torch.long),
            "labels": torch.tensor([row["answer_ids"] for row in records], dtype=torch.long) - 128,
            "sources": torch.tensor([row["source_positions"] for row in records], dtype=torch.long),
            "distances": torch.tensor([row["distances"] for row in records], dtype=torch.long)}

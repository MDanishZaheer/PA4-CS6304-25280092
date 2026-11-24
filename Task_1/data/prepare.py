# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Materialize, audit, and preserve the course Task 1 dataset selections."""

import gzip
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from Task_1.common.artifacts import read_json, sha256_file


TASK_FILES = {
    "calibration": "c4_calibration_locked.jsonl.gz",
    "development": "c4_development_locked.jsonl.gz",
    "wikitext2": "wikitext2_test_windows_locked.jsonl.gz",
    "c4": "c4_validation_windows_locked.jsonl.gz",
    "hellaswag": "hellaswag_val_locked.jsonl.gz",
}

SUPPLIED_INPUTS = (
    "scripts/setup_data.py",
    "data/public_manifests/c4_candidate_rows.jsonl",
    "data/public_manifests/hellaswag_val_ids.jsonl",
    "data/public_manifests/public_corpus_materialization_spec.json",
    "data/public_manifests/gsm8k_train_ids.jsonl",
    "data/public_manifests/gsm8k_test_ids.jsonl",
)


def read_gzip_records(path):
    """Read every materialized record and expose truncated gzip files as errors."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_supplied_inputs(source):
    """Verify relevant bundled inputs without relying on a stale unrelated entry."""
    source = Path(source)
    checksums = read_json(source / "SHA256SUMS.json")
    verified = {}
    for relative in SUPPLIED_INPUTS:
        path = source / relative
        if relative not in checksums or not path.is_file() or sha256_file(path) != checksums[relative]:
            raise ValueError(f"Supplied Task 1 source failed its published checksum: {relative}")
        verified[relative] = checksums[relative]
    return verified


def ensure_supplied_source(source, bundled_source):
    """Stage the unchanged course files from the uploaded task into runtime Data/."""
    source, bundled_source = Path(source), Path(bundled_source)
    if (source / "scripts/setup_data.py").is_file():
        verify_supplied_inputs(source)
        return source
    verify_supplied_inputs(bundled_source)
    for relative in (*SUPPLIED_INPUTS, "SHA256SUMS.json", "requirements-public.txt"):
        original = bundled_source / relative
        destination = source / relative
        if destination.is_file() and sha256_file(destination) != sha256_file(original):
            raise ValueError(f"Existing course input differs from the bundled copy: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
    verify_supplied_inputs(source)
    return source


def preserve_initial_revisions(source, store):
    """Back up the course script's initial revision lock before long downloads."""
    source = Path(source)
    revision_path = source / "data/local_source_revisions.json"
    if not revision_path.is_file():
        specification = importlib.util.spec_from_file_location("course_pa4_materializer", source / "scripts/setup_data.py")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        module.revisions()
    revisions = read_json(revision_path)
    required = {"c4", "wikitext", "hellaswag", "gsm8k", "model_base", "model_target", "model_draft"}
    if not required.issubset(revisions.get("revisions", {})):
        raise ValueError("The supplied public materializer's source lock is incomplete.")
    if any(len(revisions["revisions"][key]) != 40 for key in required):
        raise ValueError("Every source/model revision must be an immutable commit SHA.")
    target = store.path("splits/local_source_revisions.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(revision_path, target)
    store.backup(["splits/local_source_revisions.json"])


def quarantine_incomplete_materializations(source, store):
    """Retain incomplete generated files and let the unchanged script rebuild them."""
    public_root = Path(source) / "data/public_locked"
    counts = {TASK_FILES["calibration"]: 128, TASK_FILES["development"]: 32,
              TASK_FILES["hellaswag"]: 2000, "gsm8k_train_locked.jsonl.gz": 500,
              "gsm8k_test_locked.jsonl.gz": 60}
    quarantined = []
    for filename in [*TASK_FILES.values(), "gsm8k_train_locked.jsonl.gz", "gsm8k_test_locked.jsonl.gz"]:
        path = public_root / filename
        if not path.is_file():
            continue
        try:
            records = read_gzip_records(path)
            if filename in counts and len(records) != counts[filename]:
                raise ValueError(f"Incomplete materialized count: {len(records)}/{counts[filename]}")
            if "windows_locked" in filename and sum(row["scored_tokens"] for row in records) != 50000:
                raise ValueError("Incomplete 50,000-token materialization.")
            if filename in {TASK_FILES["calibration"], TASK_FILES["development"]} and any(len(row["token_ids"]) != 512 for row in records):
                raise ValueError("Incomplete 512-token calibration/development window.")
        except (OSError, EOFError, ValueError, KeyError) as error:
            # These are uncommitted generated files only. Frozen run data are
            # loaded from checksummed results and never enter this function.
            destination = store.paths["data"] / "materialization_recovery" / f"{time.time_ns()}_{filename}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not path.resolve().is_relative_to(public_root.resolve()) or not destination.resolve().is_relative_to(store.paths["data"].resolve()):
                raise ValueError("Unsafe materialization recovery path.")
            digest = sha256_file(path)
            path.replace(destination)
            quarantined.append({"file": filename, "sha256": digest, "reason": str(error),
                                "retained_at": str(destination)})
    if quarantined:
        previous = read_json(store.path("splits/preparation_recovery.json")) if store.path("splits/preparation_recovery.json").is_file() else []
        store.save_json("splits/preparation_recovery.json", previous + quarantined)
        print(f"Preserved {len(quarantined)} incomplete generated file(s); rebuilding from the fixed source lock.", flush=True)
    return quarantined


def validate_task_data(records, configuration, source=None):
    """Enforce counts, scored-token boundaries, disjointness, and manifest order."""
    calibration = records["calibration"]
    development = records["development"]
    if len(calibration) != 128 or len(development) != 32:
        raise ValueError("Expected exactly 128 calibration and 32 development sequences.")
    for label, split in [("calibration", calibration), ("development", development)]:
        if any(len(record["token_ids"]) != 512 for record in split):
            raise ValueError(f"Every {label} record must contain exactly 512 tokens.")
        identifiers = [(record["source_row"], record["token_offset"]) for record in split]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"Duplicate {label} token windows.")
    calibration_hashes = {record["raw_text_sha256"] for record in calibration}
    development_hashes = {record["raw_text_sha256"] for record in development}
    if calibration_hashes & development_hashes:
        raise ValueError("C4 calibration and development contain overlapping documents.")
    training_rows = {record["source_row"] for record in calibration + development}
    if len(training_rows) != 160:
        raise ValueError("C4 calibration/development source IDs overlap.")
    for suite in ["wikitext2", "c4"]:
        rows = records[suite]
        target_total = 0
        occupied = {}
        for record in rows:
            tokens = record["token_ids"]
            begin, end = record["score_target_offsets"]
            if not 2 <= len(tokens) <= 512 or not 1 <= begin < end <= len(tokens):
                raise ValueError(f"Invalid {suite} prediction window; the first token cannot be scored.")
            if record["scored_tokens"] != end - begin:
                raise ValueError(f"Inconsistent {suite} scored-token denominator.")
            key = (record["source"], record["source_row"])
            interval = (record["token_offset"], record["token_offset"] + len(tokens))
            for previous in occupied.get(key, []):
                if max(previous[0], interval[0]) < min(previous[1], interval[1]):
                    raise ValueError(f"Overlapping independent evaluation windows in {suite}.")
            occupied.setdefault(key, []).append(interval)
            target_total += record["scored_tokens"]
        if target_total != configuration["evaluation"]["scored_tokens_per_suite"]:
            raise ValueError(f"{suite} must score exactly 50,000 target tokens, got {target_total}.")
    validation_hashes = {record["raw_text_sha256"] for record in records["c4"]}
    if validation_hashes & (calibration_hashes | development_hashes):
        raise ValueError("C4 held-out validation documents overlap training selections.")
    hellaswag = records["hellaswag"]
    if len(hellaswag) != 2000 or [row["order"] for row in hellaswag] != list(range(2000)):
        raise ValueError("HellaSwag must preserve the fixed 2,000-ID manifest order.")
    if len({row["source_row_index"] for row in hellaswag}) != 2000:
        raise ValueError("Duplicate HellaSwag source IDs.")
    for row in hellaswag:
        example = row["example"]
        if len(example["endings"]) != 4 or int(example["label"]) not in range(4):
            raise ValueError("Invalid four-way HellaSwag example.")
    if source is not None:
        manifest_path = Path(source) / "data/public_manifests/hellaswag_val_ids.jsonl"
        expected = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if [row["source_row_index"] for row in hellaswag] != [row["row_index"] for row in expected]:
            raise ValueError("Materialized HellaSwag IDs differ from the supplied manifest.")
        candidate_path = Path(source) / "data/public_manifests/c4_candidate_rows.jsonl"
        candidates = {json.loads(line)["row_index"] for line in candidate_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        if not training_rows.issubset(candidates):
            raise ValueError("C4 calibration/development IDs do not come from supplied candidates.")
    return {"calibration_sequences": 128, "development_sequences": 32,
            "calibration_tokens_per_sequence": 512, "wikitext2_scored_tokens": 50000,
            "c4_scored_tokens": 50000, "hellaswag_examples": 2000,
            "hellaswag_diagnostic_orders": [0, 512], "leakage_checks": "passed"}


def prepare_task_data(store):
    """Run the supplied public materializer on Colab and back up fixed records."""
    source = store.paths["source"]
    ensure_supplied_source(source, store.paths["task"] / "resources/course_data")
    supplied_hashes = verify_supplied_inputs(source)
    preparation_started = time.perf_counter()
    public_root = source / "data/public_locked"
    revision_path = source / "data/local_source_revisions.json"
    # Recover the exact prior materialization before considering a network request.
    saved_lock = store.path("splits/local_source_revisions.json")
    if saved_lock.is_file():
        if revision_path.is_file() and read_json(revision_path) != read_json(saved_lock):
            raise ValueError("Uploaded source revisions differ from this run's saved immutable lock.")
        revision_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(saved_lock, revision_path)
        public_root.mkdir(parents=True, exist_ok=True)
        for filename in TASK_FILES.values():
            saved = store.path(f"splits/materialized/{filename}")
            destination = public_root / filename
            if saved.is_file():
                if destination.is_file() and sha256_file(saved) != sha256_file(destination):
                    raise ValueError(f"Uploaded data differ from this run's saved records: {filename}")
                shutil.copyfile(saved, destination)
    if not revision_path.is_file() and any((public_root / name).is_file() for name in TASK_FILES.values()):
        raise ValueError("Materialized public assets exist without their matching source revision lock. Upload that lock; existing windows must not be relabeled with newly resolved revisions.")
    preserve_initial_revisions(source, store)
    quarantine_incomplete_materializations(source, store)
    if not all((public_root / name).is_file() for name in TASK_FILES.values()):
        # Use the course script unchanged; it also prepares the small GSM8K assets.
        environment = os.environ.copy()
        environment["HF_HOME"] = str(store.paths["data"] / "huggingface")
        subprocess.run([sys.executable, str(source / "scripts/setup_data.py"), "--public"],
                       check=True, cwd=source, env=environment)
    revisions = read_json(revision_path)
    model_revision = revisions["revisions"]["model_base"]
    if len(model_revision) != 40:
        raise ValueError("The base model must be pinned to an immutable 40-character commit SHA.")
    explicit_revision = store.configuration["model"]["revision"]
    if explicit_revision is not None and explicit_revision != model_revision:
        raise ValueError("Configured model revision differs from the tokenizer used to materialize the data.")
    records = {name: read_gzip_records(public_root / filename) for name, filename in TASK_FILES.items()}
    summary = validate_task_data(records, store.configuration, source)
    relatives = []
    for filename in TASK_FILES.values():
        relative = f"splits/materialized/{filename}"
        destination = store.path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(public_root / filename, destination)
        relatives.append(relative)
    destination = store.path("splits/local_source_revisions.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(revision_path, destination)
    relatives.append("splits/local_source_revisions.json")
    metadata = {"summary": summary, "source_revisions": revisions, "supplied_input_hashes": supplied_hashes,
                "materialized_hashes": {filename: sha256_file(public_root / filename) for filename in TASK_FILES.values()},
                "calibration_ids": [{key: row[key] for key in ["source_row", "token_offset", "raw_text_sha256"]} for row in records["calibration"]],
                "development_ids": [{key: row[key] for key in ["source_row", "token_offset", "raw_text_sha256"]} for row in records["development"]],
                "source_note": "Supplied self-service materializer; first-run HF revisions preserved, not a staff-wide snapshot."}
    store.backup(relatives)
    store.save_json("splits/selection.json", metadata)
    store.save_json("metrics/data_preparation.json", {"current_session_seconds": time.perf_counter() - preparation_started,
                                                       "source_lock_backed_up_before_download": True})
    return records, metadata


def load_saved_task_data(store):
    """Load the authoritative saved materialization without accessing the network."""
    metadata = read_json(store.path("splits/selection.json"))
    for filename, expected in metadata["materialized_hashes"].items():
        if sha256_file(store.path(f"splits/materialized/{filename}")) != expected:
            raise ValueError(f"Saved dataset content changed: {filename}")
    records = {name: read_gzip_records(store.path(f"splits/materialized/{filename}")) for name, filename in TASK_FILES.items()}
    validate_task_data(records, store.configuration)
    return records, metadata

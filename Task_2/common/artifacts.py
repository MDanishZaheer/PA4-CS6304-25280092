# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Atomic artifacts, compatible experiment identities, and strict replay checks."""

import hashlib
import json
import math
import os
import time
from pathlib import Path


class SessionTimeLimit(RuntimeError):
    """Request a clean restart after preserving completed work."""


def sha256_file(path):
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    """Read a UTF-8 JSON artifact."""
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path, value):
    """Publish valid JSON only after the complete temporary file is flushed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_jsonl(path):
    """Load saved records; partial lines are errors, never silently discarded."""
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_jsonl(path, records):
    """Replace a modest progress file with complete JSONL records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def finite_number(value):
    """Represent unbounded metrics explicitly instead of writing invalid JSON."""
    return float(value) if math.isfinite(float(value)) else None


def safe_perplexity(mean_nll):
    """Preserve NLL when its exponential exceeds floating-point range."""
    return math.exp(mean_nll) if mean_nll < 709 else None


class ArtifactStore:
    """Write run-bound artifacts and synchronize verified files when configured."""

    def __init__(self, paths, configuration, sync=None):
        self.paths = paths
        self.configuration = configuration
        self.root = paths["results"]
        self.sync = sync
        self.started = time.monotonic()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, relative):
        """Resolve an artifact path while preventing escapes from this run."""
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Artifact paths must stay inside the run directory.")
        return path

    def save_json(self, relative, value):
        """Atomically save JSON, then confirm its durable backup."""
        path = self.path(relative)
        atomic_json(path, value)
        self.backup([relative])
        return path

    def save_records(self, relative, records):
        """Save complete evaluation progress before continuing expensive work."""
        atomic_jsonl(self.path(relative), records)
        self.backup([relative])

    def backup(self, relatives):
        """Synchronize only the newly written files."""
        if self.sync is not None:
            self.sync.publish(relatives)

    def check_time(self):
        """Stop at a safe item boundary before the configured session deadline."""
        settings = self.configuration["execution"]
        limit = settings["maximum_session_seconds"] - settings["time_reserve_seconds"]
        if time.monotonic() - self.started > limit:
            raise SessionTimeLimit("Session time reserve reached. Saved progress is on Drive; reconnect and Run All to resume.")

    def completed(self, stage):
        """Require both a stage manifest and matching checksummed outputs."""
        path = self.path(f"manifests/{stage}.json")
        if not path.exists():
            return False
        manifest = read_json(path)
        if manifest.get("status") != "complete":
            return False
        lock_path = self.path("manifests/run_lock.json")
        if lock_path.exists() and manifest.get("config_id") != read_json(lock_path)["config_id"]:
            raise ValueError(f"Stage {stage} belongs to an incompatible experiment.")
        for relative, expected in manifest.get("artifacts", {}).items():
            artifact = self.path(relative)
            if not artifact.is_file() or sha256_file(artifact) != expected:
                raise ValueError(f"Corrupt or missing completed artifact: {relative}. Restore its verified Drive backup.")
        return True

    def complete(self, stage, relatives, summary=None):
        """Publish completion last, after every required artifact is valid."""
        hashes = {relative: sha256_file(self.path(relative)) for relative in relatives}
        lock_path = self.path("manifests/run_lock.json")
        identity = read_json(lock_path)["config_id"] if lock_path.exists() else None
        manifest = {"stage": stage, "status": "complete", "config_id": identity,
                    "artifacts": hashes, "summary": summary or {}}
        self.save_json(f"manifests/{stage}.json", manifest)
        return manifest

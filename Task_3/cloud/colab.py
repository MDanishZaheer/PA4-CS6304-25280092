# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Copy code safely and verify every durable result backup before publication."""

import json
import os
import shutil
from pathlib import Path

from Task_3.common.artifacts import atomic_json, read_json, sha256_file


def bootstrap_project(drive_project, local_project):
    """Copy the self-contained task without overwriting any Drive result backup."""
    source = Path(drive_project)
    destination = Path(local_project)
    if not (source / "Task_3/main_task_3.ipynb").is_file():
        raise FileNotFoundError(f"Upload Task_3 into {source} before running this notebook.")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / "Task_3", destination / "Task_3", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("results", "__pycache__", ".ipynb_checkpoints"))
    return destination


class DriveSync:
    """Use temporary copies, checksums, retained predecessors, and a final index."""

    def __init__(self, local_root, drive_root):
        self.local_root = Path(local_root)
        self.drive_root = Path(drive_root)
        self.drive_root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.drive_root / "backup_index.json"
        self.index = read_json(self.index_path) if self.index_path.is_file() else {"files": {}}

    def publish(self, relatives):
        """Confirm backup integrity; a failed sync must stop expensive execution."""
        for relative in relatives:
            local = (self.local_root / relative).resolve()
            if not local.is_relative_to(self.local_root.resolve()) or not local.is_file():
                raise ValueError(f"Cannot back up missing/outside artifact: {relative}")
            target = self.drive_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = sha256_file(local)
            temporary = target.with_name(target.name + ".uploading")
            shutil.copyfile(local, temporary)
            if temporary.stat().st_size != local.stat().st_size or sha256_file(temporary) != digest:
                raise RuntimeError(f"Drive copy verification failed: {relative}")
            previous = target.with_name(target.name + ".previous")
            if target.exists():
                os.replace(target, previous)
            os.replace(temporary, target)
            self.index["files"][relative] = {"sha256": digest, "bytes": target.stat().st_size}
            atomic_json(self.index_path, self.index)
            if previous.exists():
                previous.unlink()
        print(f"Verified Drive backup: {len(relatives)} file(s)", flush=True)

    def restore(self):
        """Restore only indexed, verified files; reject corrupt remote evidence."""
        count = 0
        for relative, metadata in self.index["files"].items():
            source = self.drive_root / relative
            expected = metadata["sha256"]
            if not source.is_file() or sha256_file(source) != expected:
                previous = source.with_name(source.name + ".previous")
                if previous.is_file() and sha256_file(previous) == expected:
                    source = previous
                else:
                    raise RuntimeError(f"Unverified Drive recovery artifact: {relative}")
            target = (self.local_root / relative).resolve()
            if not target.is_relative_to(self.local_root.resolve()):
                raise ValueError("Invalid backup-index path.")
            if target.is_file() and sha256_file(target) == expected:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".restoring")
            shutil.copyfile(source, temporary)
            if sha256_file(temporary) != expected:
                raise RuntimeError(f"Restore verification failed: {relative}")
            temporary.replace(target)
            count += 1
        print(f"Restored {count} verified result/progress files.", flush=True)
        return count

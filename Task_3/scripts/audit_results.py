# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Audit downloaded Task 3 results without loading models or doing GPU work."""

import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Task_3.common.artifacts import atomic_json
from Task_3.common.config import load_configuration
from Task_3.common.paths import resolve_paths
from Task_3.common.replay import validate_replay


def audit_results():
    project = Path(__file__).resolve().parents[2]
    configuration = load_configuration(project / "Task_3")
    paths = resolve_paths(configuration, project)
    manifest = validate_replay(configuration, paths)
    report = {"status": "passed", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest["run_id"], "counts": manifest["counts"], "registered_files_verified": len(manifest["required_artifacts"]),
        "computed_on": manifest["computed_on"], "gpu_work_performed": False}
    atomic_json(paths["results"] / "metrics/local_run_audit.json", report)
    print("PASS: saved Task 3 counts, prefixes, training quotas, sampling traces, uncertainty and figures verified.")
    return report


if __name__ == "__main__":
    audit_results()

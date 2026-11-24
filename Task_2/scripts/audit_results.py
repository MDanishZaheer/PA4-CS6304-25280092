# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Audit downloaded Task 2 evidence before report writing or deleting a download ZIP."""

import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Task_2.common.artifacts import atomic_json
from Task_2.common.config import load_configuration
from Task_2.common.paths import resolve_paths
from Task_2.common.replay import validate_replay


def audit_results():
    """Recalculate query metrics, primary paired intervals and budget invariants locally."""
    project = Path(__file__).resolve().parents[2]
    configuration = load_configuration(project / "Task_2")
    paths = resolve_paths(configuration, project)
    manifest = validate_replay(configuration, paths, deep=True)
    report = {"audited_at_utc": datetime.now(timezone.utc).isoformat(), "audited_on": "Local ATML; saved evidence only",
              "status": manifest["status"], "run_id": manifest["run_id"], "computed_on": manifest["computed_on"],
              "registered_artifacts_verified": len(manifest["required_artifacts"]), "counts": manifest["counts"],
              "recorded_failures": manifest["recorded_failures"], "gpu_work_performed": False,
              "report_readiness": "review recorded failures and limitations" if manifest["recorded_failures"] else "complete experimental evidence"}
    atomic_json(paths["results"] / "metrics/local_run_audit.json", report)
    print(f"PASS: Task 2 saved evidence verified; status={manifest['status']}; recorded failures={len(manifest['recorded_failures'])}.")
    return report


if __name__ == "__main__":
    audit_results()

# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Resolve project paths without embedding machine-specific locations."""

from pathlib import Path


def find_project_root(start=None):
    """Find a repository containing Task_1 from a notebook or module location."""
    start = Path(start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "Task_1/configs/base.yaml").is_file():
            return candidate
    raise FileNotFoundError("Run from the PA4 root or Task_1 folder; Task_1/configs/base.yaml is missing.")


def resolve_paths(configuration, project_root=None):
    """Return local fast-storage and optional durable Drive paths."""
    root = Path(project_root).resolve() if project_root else find_project_root()
    execution = configuration["execution"]
    paths = {
        "project": root,
        "task": root / "Task_1",
        "data": root / "Data/Task_1",
        "source": root / "Data" / execution["source_folder"],
        "results": root / "Task_1/results" / execution["run_id"],
        "drive_project": Path(execution["drive_mount"]) / "MyDrive" / execution["drive_project"],
    }
    paths["drive_results"] = paths["drive_project"] / "Task_1/results" / execution["run_id"]
    return paths

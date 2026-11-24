# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Resolve self-contained Task 3 code, runtime caches and named result directories."""

from pathlib import Path


def resolve_paths(configuration, project_root):
    """Keep downloaded models outside the uploaded Python package."""
    project = Path(project_root).resolve()
    settings = configuration["execution"]
    run_id = settings["run_id"]
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("The run ID must be a single directory name.")
    drive = Path(settings["drive_mount"]) / "MyDrive" / settings["drive_project"]
    return {"project": project, "task": project / "Task_3", "resources": project / "Task_3/resources/course_data",
        "data": project / "Data/Task_3", "results": project / "Task_3/results" / run_id,
        "drive_results": drive / "Task_3/results" / run_id}

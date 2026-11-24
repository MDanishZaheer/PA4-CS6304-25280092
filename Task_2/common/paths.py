# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Resolve local, Colab and durable Drive locations without cross-task dependencies."""

from pathlib import Path


def resolve_paths(configuration, project_root):
    """Keep all Task 2 outputs in a single named run directory."""
    project = Path(project_root).resolve()
    task = project / "Task_2"
    run_id = configuration["execution"]["run_id"]
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("Use a single directory name for execution.run_id.")
    drive = Path(configuration["execution"]["drive_mount"]) / "MyDrive" / configuration["execution"]["drive_project"]
    return {"project": project, "task": task, "data": project / "Data/Task_2",
            "resources": task / "resources/course_data", "results": task / "results" / run_id,
            "drive_results": drive / "Task_2/results" / run_id}

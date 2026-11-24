# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Validate Task 2 delivery and execute replay cells without Jupyter dependencies."""

import ast
import contextlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def execute_replay_notebook(project_root, working_directory=None):
    """Execute every code cell against explicit saved test evidence in a clean namespace."""
    project = Path(project_root).resolve()
    notebook = json.loads((project / "Task_2/main_task_2.ipynb").read_text(encoding="utf-8"))
    namespace, executed = {"__name__": "__main__"}, 0
    previous = Path.cwd()
    original_find_spec = importlib.util.find_spec
    def local_find_spec(name, *args, **kwargs):
        return None if name == "google.colab" else original_find_spec(name, *args, **kwargs)
    try:
        os.chdir(working_directory or project)
        with patch("importlib.util.find_spec", side_effect=local_find_spec), contextlib.redirect_stdout(io.StringIO()):
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
                    exec(compile(source, f"main_task_2_cell_{index}", "exec"), namespace)
                    executed += 1
    finally:
        os.chdir(previous)
    return executed


def validate_project():
    """Compile source and require all eleven steps, four RQs and standalone assets."""
    task = Path(__file__).resolve().parents[1]
    sources = sorted(task.rglob("*.py"))
    for path in sources:
        source = path.read_text(encoding="utf-8")
        if not path.is_relative_to(task / "resources/course_data") and not source.startswith("# Name : Muhammad Danish Zaheer Awan\n# REG id : 25280092"):
            raise ValueError(f"Missing student header: {path}")
        ast.parse(source, str(path))
        compile(source, str(path), "exec")
    notebook = json.loads((task / "main_task_2.ipynb").read_text(encoding="utf-8"))
    combined, count = [], 0
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        combined.append(source)
        if cell["cell_type"] == "code":
            compile(source, f"Task2_cell_{index}", "exec")
            count += 1
    text = "\n".join(combined)
    positions = [text.index(f"## Step {number}:") for number in range(1, 12)]
    if positions != sorted(positions) or text.index("# Required evidence") < positions[-1]:
        raise ValueError("Task 2 notebook steps/RQs are not in assignment order.")
    for number in range(1, 5):
        if f"## RQ-{number}:" not in text:
            raise ValueError("A Task 2 research question is missing.")
    if "Task_1" in text or "figures/tnr10" in text:
        raise ValueError("Task 2 must be standalone and use only canonical main figures.")
    for relative in ["README.md", "requirements-compute.txt", "requirements-replay.txt", "configs/base.yaml", "configs/task_2.yaml"]:
        if not (task / relative).is_file():
            raise FileNotFoundError(f"Required Task 2 delivery file is missing: {relative}")
    from Task_2.common.config import load_configuration
    from Task_2.data.mqar import verify_supplied_inputs
    from Task_2.training.protocol import evaluation_conditions, training_conditions
    configuration = load_configuration(task)
    if len(training_conditions(configuration)) != 42 or len(evaluation_conditions(configuration)) != 54:
        raise ValueError("The mandatory MQAR run matrix is incomplete.")
    verify_supplied_inputs(task / "resources/course_data")
    print(f"PASS: {len(sources)} Python files and {count} notebook code cells compile; all eleven steps, four RQs, 42 training runs and standalone assets are present.")


if __name__ == "__main__":
    validate_project()

# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Validate source/notebook syntax without downloads or full experiment execution."""

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
    """Execute every cell against supplied test evidence in an isolated namespace."""
    project_root = Path(project_root).resolve()
    notebook = json.loads((project_root / "Task_1/main_task_1.ipynb").read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    original_directory = Path.cwd()
    original_find_spec = importlib.util.find_spec

    def local_find_spec(name, *arguments, **keywords):
        """Force the local replay branch even when this test runs on Colab."""
        return None if name == "google.colab" else original_find_spec(name, *arguments, **keywords)

    try:
        os.chdir(working_directory or project_root)
        with patch("importlib.util.find_spec", side_effect=local_find_spec), contextlib.redirect_stdout(io.StringIO()):
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    source = cell["source"]
                    source = "".join(source) if isinstance(source, list) else source
                    exec(compile(source, f"main_task_1_cell_{index}", "exec"), namespace)
    finally:
        os.chdir(original_directory)
    return len([cell for cell in notebook["cells"] if cell["cell_type"] == "code"])


def validate_project():
    """Compile all delivered code and require the configured notebook structure."""
    task_root = Path(__file__).resolve().parents[1]
    python_files = sorted(task_root.rglob("*.py"))
    for path in python_files:
        source = path.read_text(encoding="utf-8")
        supplied_code = path.is_relative_to(task_root / "resources/course_data")
        if not supplied_code and not source.startswith("# Name : Muhammad Danish Zaheer Awan\n# REG id : 25280092"):
            raise ValueError(f"Missing student identification: {path}")
        ast.parse(source, filename=str(path))
        compile(source, str(path), "exec")
    notebook_path = task_root / "main_task_1.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    if notebook["nbformat"] != 4 or not notebook["cells"]:
        raise ValueError("The main notebook is not a valid version-4 notebook.")
    code_count = 0
    text = []
    for index, cell in enumerate(notebook["cells"]):
        source = cell["source"]
        source = "".join(source) if isinstance(source, list) else source
        text.append(source)
        if cell["cell_type"] == "code":
            compile(source, f"notebook_cell_{index}", "exec")
            code_count += 1
        if "NotImplementedError" in source or "TEMPLATE_CONFIGURED" in source or "<Assignment-specific" in source:
            raise ValueError("Unimplemented scaffold remains in the notebook.")
    combined = "\n".join(text)
    positions = [combined.index(f"## Step {number}:") for number in range(1, 10)]
    if positions != sorted(positions):
        raise ValueError("Assignment steps are not ordered.")
    if combined.index("# Required evidence") < positions[-1]:
        raise ValueError("RQ evidence must follow the assignment steps.")
    for number in range(1, 5):
        if f"## RQ-{number}:" not in combined:
            raise ValueError("A required research question is missing.")
    for relative in ["README.md", "requirements-compute.txt", "requirements-replay.txt", "configs/base.yaml", "configs/task_1.yaml"]:
        if not (task_root / relative).is_file():
            raise FileNotFoundError(f"Missing required upload file: {relative}")
    from Task_1.data.prepare import verify_supplied_inputs
    verify_supplied_inputs(task_root / "resources/course_data")
    print(f"PASS: {len(python_files)} Python files and {code_count} notebook code cells compile; all nine steps and four RQs are present.")


if __name__ == "__main__":
    validate_project()

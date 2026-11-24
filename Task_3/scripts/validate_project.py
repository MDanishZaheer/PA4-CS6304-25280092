# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Compile the self-contained task and execute replay cells in a fresh test process."""

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


def execute_replay_notebook(project, working_directory=None):
    """Execute all saved notebook code against explicitly supplied complete evidence."""
    project = Path(project).resolve()
    notebook = json.loads((project / "Task_3/main_task_3.ipynb").read_text(encoding="utf-8"))
    namespace, executed = {"__name__": "__main__"}, 0
    original, previous = importlib.util.find_spec, Path.cwd()
    try:
        os.chdir(working_directory or project)
        with patch("importlib.util.find_spec", side_effect=lambda name, *a, **k: None if name == "google.colab" else original(name, *a, **k)), contextlib.redirect_stdout(io.StringIO()):
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
                    exec(compile(source, f"main_task3_cell_{index}", "exec"), namespace)
                    executed += 1
    finally:
        os.chdir(previous)
    return executed


def validate_project():
    """Require seven assignment steps, four RQs and no dependency on another task."""
    task = Path(__file__).resolve().parents[1]
    sources = sorted(task.rglob("*.py"))
    for path in sources:
        source = path.read_text(encoding="utf-8")
        if not source.startswith("# Name : Muhammad Danish Zaheer Awan\n# REG id : 25280092"):
            raise ValueError(f"Missing student header: {path}")
        tree = ast.parse(source, str(path))
        compile(source, str(path), "exec")
        for node in ast.walk(tree):
            modules = [node.module or ""] if isinstance(node, ast.ImportFrom) else [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
            if any(name.split('.')[0] in {'Task_1', 'Task_2'} for name in modules):
                raise ValueError("Task 3 must upload independently.")
    notebook = json.loads((task / "main_task_3.ipynb").read_text(encoding="utf-8"))
    text, count = [], 0
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        text.append(source)
        if cell["cell_type"] == "code":
            compile(source, f"Task3_cell_{index}", "exec")
            count += 1
    combined = "\n".join(text)
    offsets = [combined.index(f"## Step {i}:") for i in range(1, 8)]
    if offsets != sorted(offsets) or any(f"## RQ-{i}:" not in combined for i in range(1, 5)):
        raise ValueError("Notebook assignment step/RQ order is incomplete.")
    from Task_3.common.config import load_configuration, trained_variants
    from Task_3.data.prepare import verify_supplied_inputs
    verify_supplied_inputs(task / "resources/course_data")
    assert len(trained_variants(load_configuration(task))) == 4
    print(f"PASS: {len(sources)} Python files, {count} notebook code cells, seven steps, four RQs and four training runs.")


if __name__ == "__main__":
    validate_project()

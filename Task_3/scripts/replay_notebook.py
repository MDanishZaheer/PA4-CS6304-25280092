# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Replay downloaded Task 3 evidence in this ATML interpreter; save outputs only on success."""

import json
import os
import queue
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from jupyter_client import KernelManager

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Task_3.common.artifacts import atomic_json
from Task_3.common.config import load_configuration
from Task_3.common.paths import resolve_paths


def execute_cell(client, source, timeout=900, store_history=True):
    """Collect standard notebook outputs, stopping on errors or missing completion."""
    message_id = client.execute(source, stop_on_error=True, allow_stdin=False, store_history=store_history)
    outputs = []
    execution_count = None
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Local replay cell exceeded its execution time limit.")
        try:
            message = client.get_iopub_msg(timeout=min(remaining, 5))
        except queue.Empty:
            continue
        if message.get("parent_header", {}).get("msg_id") != message_id:
            continue
        kind, content = message["msg_type"], message["content"]
        if kind == "execute_input":
            execution_count = content["execution_count"]
        elif kind == "stream":
            outputs.append({"output_type": kind, "name": content["name"], "text": content["text"]})
        elif kind in {"display_data", "execute_result"}:
            output = {"output_type": kind, "data": content["data"], "metadata": content.get("metadata", {})}
            if kind == "execute_result":
                output["execution_count"] = content["execution_count"]
            outputs.append(output)
        elif kind == "clear_output":
            outputs.clear()
        elif kind == "error":
            raise RuntimeError("\n".join(content["traceback"]))
        elif kind == "status" and content["execution_state"] == "idle":
            break
    while True:
        reply = client.get_shell_msg(timeout=10)
        if reply.get("parent_header", {}).get("msg_id") == message_id:
            if reply["content"]["status"] != "ok":
                raise RuntimeError(f"Replay cell failed: {reply['content']}")
            break
    return execution_count, outputs


def replay_notebook():
    """Use this interpreter for Run All and publish the notebook only on success."""
    task = Path(__file__).resolve().parents[1]
    project = task.parent
    configuration = load_configuration(task)
    paths = resolve_paths(configuration, project)
    if not (paths["results"] / "manifests/run.json").is_file():
        raise FileNotFoundError("Download the completed Colab Task_3/results folder before running local replay.")
    destination = task / "main_task_3.ipynb"
    notebook = json.loads(destination.read_text(encoding="utf-8"))
    connection = paths["results"] / "manifests/.local_replay_connection.json"
    connection.parent.mkdir(parents=True, exist_ok=True)
    manager = KernelManager(kernel_name="python3", connection_file=str(connection))
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}",
                               "--HistoryManager.enabled=False"]
    manager.kernel_spec.env = {}
    client = None
    profile = tempfile.TemporaryDirectory(prefix="task3_replay_")
    started = time.monotonic()
    try:
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "IPYTHONDIR": profile.name}
        manager.start_kernel(cwd=str(task), env=environment)
        client = manager.blocking_client()
        client.start_channels()
        client.wait_for_ready(timeout=30)
        # Force replay even on a machine with a google.colab package installed.
        # Blocking compute libraries also verifies that no saved result is rescored.
        preflight = "\n".join([
            "import importlib.abc, importlib.util, sys",
            f"assert sys.executable.lower() == {str(Path(sys.executable)).lower()!r}",
            "class BlockComputeImports(importlib.abc.MetaPathFinder):",
            "    def find_spec(self, fullname, path=None, target=None):",
            "        if fullname.split('.')[0] in {'torch', 'transformers', 'datasets', 'accelerate', 'huggingface_hub'}:",
            "            raise RuntimeError('Replay attempted a compute import: ' + fullname)",
            "sys.meta_path.insert(0, BlockComputeImports())",
            "_original_find_spec = importlib.util.find_spec",
            "importlib.util.find_spec = lambda name, *args, **kwargs: None if name == 'google.colab' else _original_find_spec(name, *args, **kwargs)",
        ])
        execute_cell(client, preflight, store_history=False)
        cells_executed = 0
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = cell["source"]
            source = "".join(source) if isinstance(source, list) else source
            count, outputs = execute_cell(client, source)
            cell["execution_count"] = count
            cell["outputs"] = outputs
            cells_executed += 1
            print(f"Replayed code cell {cells_executed} (notebook index {index}).", flush=True)
        execute_cell(client, "assert RUN_MODE == 'replay'\nassert not ({'torch', 'transformers', 'datasets', 'accelerate', 'huggingface_hub'} & set(sys.modules))", store_history=False)
        images = sum("image/png" in output.get("data", {}) for cell in notebook["cells"]
                     for output in cell.get("outputs", []))
        if images != 9:
            raise ValueError(f"Expected nine displayed report figures, found {images}.")
        report = {"status": "complete", "mode": "replay", "code_cells_executed": cells_executed,
                  "displayed_png_figures": images, "python_executable": sys.executable,
                  "executed_at_utc": datetime.now(timezone.utc).isoformat(),
                  "elapsed_seconds": time.monotonic() - started,
                  "compute_imports": [], "gpu_work_performed": False}
        notebook["metadata"]["local_replay"] = report
        notebook["metadata"]["language_info"]["version"] = sys.version.split()[0]
        atomic_json(destination, notebook)
        atomic_json(paths["results"] / "metrics/local_notebook_replay.json", report)
        print(f"PASS: saved {cells_executed} executed cells and {images} figures using {sys.executable}.")
    finally:
        if client is not None:
            client.stop_channels()
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        connection.unlink(missing_ok=True)
        profile.cleanup()


if __name__ == "__main__":
    replay_notebook()

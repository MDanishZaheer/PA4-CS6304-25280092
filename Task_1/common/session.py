# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Select a lightweight replay session or explicitly authorized Colab compute."""

import importlib.util

from Task_1.common.paths import resolve_paths


def open_session(configuration, project_root, mode):
    """Keep all torch/transformers imports inside the compute-only branch."""
    paths = resolve_paths(configuration, project_root)
    if mode == "replay":
        from Task_1.common.replay import ReplaySession
        return ReplaySession(configuration, paths)
    if mode != "compute":
        raise ValueError("Choose compute or replay explicitly.")
    try:
        in_colab = importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        in_colab = False
    if not in_colab:
        raise RuntimeError("The full experiments are reserved for Colab. Use local replay or the small diagnostic tests.")
    from Task_1.training.pipeline import ComputeSession
    return ComputeSession(configuration, paths)

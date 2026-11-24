# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Keep local replay independent of CUDA, model loading and training packages."""

import importlib.util

from Task_2.common.paths import resolve_paths


def open_session(configuration, project_root, mode):
    """Open exactly the requested execution mode; never fall back from replay to compute."""
    paths = resolve_paths(configuration, project_root)
    if mode == "replay":
        from Task_2.common.replay import ReplaySession
        return ReplaySession(configuration, paths)
    if mode != "compute":
        raise ValueError("Choose compute or replay explicitly.")
    try:
        in_colab = importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        in_colab = False
    if not in_colab:
        raise RuntimeError("Full Task 2 experiments run on Colab. Local Run All uses replay.")
    from Task_2.training.pipeline import ComputeSession
    return ComputeSession(configuration, paths)

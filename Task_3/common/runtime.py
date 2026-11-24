# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Seed stochastic computation and record the actual pinned compute runtime."""

import importlib.metadata
import platform
import random


def seed_everything(seed):
    """Seed all RNGs; explicit sampling generators isolate per-prompt draws."""
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def runtime_metadata():
    """Require CUDA for the full experiments and preserve exact package identities."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Task 3 experiments require a Colab CUDA GPU; local Run All uses replay.")
    packages = {name: importlib.metadata.version(name) for name in
                ["torch", "transformers", "huggingface_hub", "datasets", "numpy", "safetensors", "accelerate"]}
    if packages["transformers"] != "4.57.1" or packages["huggingface_hub"] != "0.36.2":
        raise RuntimeError("Use the supplied Transformers 4.57.1 / Hub 0.36.2 cache API pins.")
    gpu = torch.cuda.get_device_properties(0)
    return {"computed_on": "Google Colab", "python": platform.python_version(), "packages": packages,
        "gpu": gpu.name, "gpu_memory_bytes": gpu.total_memory, "cuda": torch.version.cuda,
        "compute_capability": list(torch.cuda.get_device_capability()), "tf32": False}

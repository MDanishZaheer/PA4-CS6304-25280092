# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Record runtime provenance and deterministic seeds only inside compute paths."""

import importlib.metadata
import platform
import random


def seed_everything(seed):
    """Seed Python, NumPy and Torch without requiring bitwise determinism from SDPA."""
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def runtime_metadata():
    """Capture the actual environment, including device and cache API pins."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Full Task 2 computation requires a Colab CUDA GPU; use replay locally.")
    versions = {name: importlib.metadata.version(name) for name in
                ["torch", "transformers", "huggingface_hub", "numpy", "safetensors", "accelerate"]}
    if versions["transformers"] != "4.57.1" or versions["huggingface_hub"] != "0.36.2":
        raise RuntimeError("Task 2 cache editing requires Transformers 4.57.1 and Hub 0.36.2.")
    properties = torch.cuda.get_device_properties(0)
    return {"computed_on": "Google Colab", "python": platform.python_version(), "packages": versions,
            "cuda": torch.version.cuda, "gpu": properties.name, "gpu_memory_bytes": properties.total_memory,
            "compute_capability": list(torch.cuda.get_device_capability()), "tf32": False}

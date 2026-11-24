# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Install pinned compute dependencies before importing the experiment stack."""

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path


def matches_pin(installed, requested):
    """Accept a CUDA local-version suffix under an exact public-version pin."""
    return installed is not None and installed.split("+", 1)[0] == requested


def install_compute_requirements(task_root):
    """Install only when necessary and require a restart for stale loaded modules."""
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    requirements = Path(task_root) / "requirements-compute.txt"
    mismatches = []
    for line in requirements.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        package, version = line.strip().split("==", 1)
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if not matches_pin(installed, version):
            mismatches.append((package, installed, version))
    if not mismatches:
        print("Pinned compute dependencies already match.")
        return
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirements)], check=True)
    aliases = {"PyYAML": "yaml", "huggingface_hub": "huggingface_hub"}
    loaded = [package for package, _, _ in mismatches if aliases.get(package, package) in sys.modules]
    if loaded:
        raise RuntimeError(f"Packages changed while already imported: {loaded}. Use Runtime > Restart session, then Run All; installed packages will be reused.")
    print("Installed pinned compute dependencies. Continuing with fresh imports.")

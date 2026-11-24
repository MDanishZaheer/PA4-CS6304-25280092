# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Include the course preparation inputs so Task_1 alone can be uploaded to Drive."""

import shutil
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Task_1.common.artifacts import sha256_file
from Task_1.data.prepare import SUPPLIED_INPUTS, verify_supplied_inputs


def prepare_upload():
    """Copy only unchanged setup code and fixed metadata, never raw datasets."""
    task_root = Path(__file__).resolve().parents[1]
    source = task_root.parent / "Data/atml-pa4-datasets-main"
    bundled = task_root / "resources/course_data"
    if not source.is_dir():
        verify_supplied_inputs(bundled)
        print(f"Task folder is ready to upload: {task_root}")
        return bundled
    verify_supplied_inputs(source)
    for relative in (*SUPPLIED_INPUTS, "SHA256SUMS.json", "requirements-public.txt"):
        original, destination = source / relative, bundled / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        if sha256_file(destination) != sha256_file(original):
            raise RuntimeError(f"Course input copy verification failed: {relative}")
    verify_supplied_inputs(bundled)
    print(f"Task folder is ready to upload: {task_root}")
    return bundled


if __name__ == "__main__":
    prepare_upload()

# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Stage unchanged small course inputs and personal fonts for direct Drive upload."""

import os
import shutil
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Task_3.analysis.fonts import configure_fonts
from Task_3.data.prepare import SUPPLIED_FILES, verify_supplied_inputs


def prepare_upload():
    task = Path(__file__).resolve().parents[1]
    source, destination = task.parent / "Data/atml-pa4-datasets-main", task / "resources/course_data"
    if source.is_dir():
        for relative in [*SUPPLIED_FILES, "SHA256SUMS.json", "requirements-public.txt"]:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target)
    verify_supplied_inputs(destination)
    if os.name == "nt":
        folder = task / "resources/fonts"
        folder.mkdir(parents=True, exist_ok=True)
        for name in ["times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"]:
            path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / name
            if path.is_file():
                shutil.copyfile(path, folder / name)
    print(configure_fonts(task))
    print("Upload the complete Task_3 folder directly to MyDrive/ATML_PA4/PA_4/. Personal fonts stay outside public Git.")


if __name__ == "__main__":
    prepare_upload()

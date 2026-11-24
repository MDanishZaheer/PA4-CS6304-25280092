# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Stage unchanged Task 2 assets and personal fonts for one-folder Drive upload."""

import os
import shutil
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Task_2.data.mqar import SUPPLIED_FILES, verify_supplied_inputs


def prepare_upload():
    """Never fetch GitHub or copy downloaded model files into the task code folder."""
    task = Path(__file__).resolve().parents[1]
    source = task.parent / "Data/atml-pa4-datasets-main"
    destination = task / "resources/course_data"
    if source.is_dir():
        verify_supplied_inputs(source)
        for relative in [*SUPPLIED_FILES, "SHA256SUMS.json", "requirements-public.txt"]:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target)
    verify_supplied_inputs(destination)
    if os.name == "nt":
        fonts = task / "resources/fonts"
        fonts.mkdir(parents=True, exist_ok=True)
        for name in ["times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"]:
            original = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / name
            if original.is_file():
                shutil.copyfile(original, fonts / name)
    from Task_2.analysis.fonts import configure_fonts
    print(configure_fonts(task))
    print("Task_2 is self-contained. Upload it directly under MyDrive/ATML_PA4/PA_4/.")
    print("Personal font files are ignored by Git; supply your own fonts when cloning publicly.")


if __name__ == "__main__":
    prepare_upload()

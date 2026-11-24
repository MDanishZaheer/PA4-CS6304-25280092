# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Require genuine Times New Roman and verify every exported text object at 10 pt."""

from pathlib import Path

from Task_2.common.artifacts import sha256_file


def configure_fonts(task_root):
    """Use personal supplied font files or installed system fonts without substitution."""
    import matplotlib
    from matplotlib import font_manager

    folder = Path(task_root) / "resources/fonts"
    for path in sorted(folder.glob("*.ttf")):
        font_manager.fontManager.addfont(str(path))
    try:
        resolved = font_manager.findfont(font_manager.FontProperties(family="Times New Roman"), fallback_to_default=False)
    except ValueError as error:
        raise RuntimeError("Times New Roman is required. Run scripts/prepare_upload.py on your Windows PC or supply your own Times New Roman fonts in Task_2/resources/fonts before the GPU run.") from error
    matplotlib.rcParams.update({"font.family": "Times New Roman", "font.size": 10,
        "axes.titlesize": 10, "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "figure.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold", "mathtext.default": "regular", "mathtext.fallback": None,
        "savefig.dpi": 300})
    return {"family": "Times New Roman", "size_pt": 10, "resolved_regular_font": str(resolved),
            "font_sha256": sha256_file(resolved), "dpi": 300}


def save_figure(store, figure, name, records):
    """Save only the main PNG and record actual text/font properties and its checksum."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.text import Text

    figure.canvas.draw()
    for text in figure.findobj(Text):
        text.set_fontfamily("Times New Roman")
        text.set_fontsize(10)
    figure.tight_layout()
    figure.canvas.draw()
    texts = [text for text in figure.findobj(Text) if text.get_text()]
    sizes = sorted({float(text.get_fontsize()) for text in texts})
    families = sorted({family for text in texts for family in text.get_fontfamily()})
    if sizes != [10.] or families != ["Times New Roman"]:
        raise ValueError("A report figure contains text outside Times New Roman 10 pt.")
    relative = f"figures/{name}.png"
    store.path("figures").mkdir(parents=True, exist_ok=True)
    figure.savefig(store.path(relative), dpi=300, bbox_inches="tight")
    records[relative] = {"sha256": sha256_file(store.path(relative)), "font_sizes_pt": sizes,
        "font_families": families, "text_objects": len(texts),
        "resolved_font_files": sorted({font_manager.findfont(text.get_fontproperties(), fallback_to_default=False) for text in texts})}
    plt.close(figure)
    store.backup([relative])
    return relative

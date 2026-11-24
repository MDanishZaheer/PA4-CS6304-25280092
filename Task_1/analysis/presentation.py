# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Maintain the main report figures while retaining cloud measurement provenance."""

import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from Task_1.common.artifacts import ArtifactStore, atomic_json, read_json, sha256_file
from Task_1.common.replay import display_question as display_saved_question


STYLE_DIRECTORY = "figures"
STYLE_MANIFEST = f"{STYLE_DIRECTORY}/style_manifest.json"
SOURCE_FILES = ["tables/all_conditions.csv", "tables/sensitivity_and_allocations.csv",
                "tables/pruning_local_errors.csv", "metrics/sensitivity_profiles.json"]


def presentation_inputs(root):
    """Identify the saved numerical inputs and the local plotting implementation."""
    run = read_json(root / "manifests/run.json")
    return {
        "saved_inputs": {relative: sha256_file(root / relative) for relative in SOURCE_FILES},
        "rendering_code": {name: sha256_file(Path(__file__).with_name(name))
                           for name in ["presentation.py", "evidence.py"]},
        "run_identity": {key: run[key]
                         for key in ["run_id", "config_id", "data_id"]},
        "font_family": "Times New Roman", "font_size_pt": 10, "dpi": 300,
    }


def valid_presentation(root, manifest):
    """Reject stale or modified main figures instead of displaying them."""
    if manifest.get("inputs") != presentation_inputs(root) or len(manifest.get("figures", {})) != 5:
        return False
    for record in manifest["figures"].values():
        path = (root / record["relative_path"]).resolve()
        if path.parent != (root / STYLE_DIRECTORY).resolve():
            return False
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            return False
    return True


def register_main_figures(root, style_manifest):
    """Update figure checksums, explicitly distinguishing local edits from cloud files."""
    run_path = root / "manifests/run.json"
    run = read_json(run_path)
    previous = run.get("figure_presentation", {}).get("original_cloud_figure_hashes", {})
    original_hashes = {relative: previous.get(relative, run["required_artifacts"][relative])
                       for relative in style_manifest["figures"]}
    for relative, record in style_manifest["figures"].items():
        run["required_artifacts"][relative] = record["sha256"]
    run["required_artifacts"][STYLE_MANIFEST] = sha256_file(root / STYLE_MANIFEST)
    run["figure_presentation"] = {
        "font_family": "Times New Roman", "font_size_pt": 10,
        "style_manifest": STYLE_MANIFEST, "updated_locally": True,
        "original_cloud_figure_hashes": original_hashes,
    }
    index_path = root / "backup_index.json"
    index = read_json(index_path) if index_path.is_file() else None
    original_index_hash = sha256_file(index_path) if index is not None else None
    atomic_json(run_path, run)
    if index is not None:
        changed = [*style_manifest["figures"], STYLE_MANIFEST, "manifests/run.json"]
        for relative in changed:
            old = index["files"].get(relative, {})
            record = {"sha256": sha256_file(root / relative), "bytes": (root / relative).stat().st_size,
                      "source": "local figure formatting"}
            if old and (old.get("source") != "local figure formatting" or "original_cloud_sha256" in old):
                record["original_cloud_sha256"] = old.get("original_cloud_sha256", old["sha256"])
                record["original_cloud_bytes"] = old.get("original_cloud_bytes", old["bytes"])
            index["files"][relative] = record
        index["local_presentation"] = {
            "original_cloud_index_sha256": index.get("local_presentation", {}).get(
                "original_cloud_index_sha256", original_index_hash),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "description": "Local Times New Roman 10 pt figures replace cloud PNGs; numerical evidence is unchanged.",
        }
        atomic_json(index_path, index)


def format_saved_figures(session):
    """Create and register the main 10 pt Times figures without rescoring results."""
    import matplotlib
    import pandas as pd
    from matplotlib import font_manager
    from PIL import Image

    from Task_1.analysis.evidence import create_figures

    root = session.paths["results"]
    manifest_path = root / STYLE_MANIFEST
    if manifest_path.is_file() and valid_presentation(root, read_json(manifest_path)):
        return {"status": "verified main figures", "figures": 5,
                "font": "Times New Roman, 10 pt", "directory": str(root / STYLE_DIRECTORY)}
    try:
        font_path = font_manager.findfont(font_manager.FontProperties(family="Times New Roman"),
                                          fallback_to_default=False)
    except ValueError:
        return {"status": "Times New Roman unavailable; displaying saved figures",
                "action": "Run local replay on Windows with Times New Roman installed."}
    inputs = presentation_inputs(root)
    cloud_manifest = read_json(root / "manifests/run.json")
    frame = pd.read_csv(root / "tables/all_conditions.csv")
    maps = pd.read_csv(root / "tables/sensitivity_and_allocations.csv")
    curves = pd.read_csv(root / "tables/pruning_local_errors.csv")
    profiles = read_json(root / "metrics/sensitivity_profiles.json")
    styles = {}
    with matplotlib.rc_context():
        generated = create_figures(ArtifactStore(session.paths, session.configuration),
            frame, maps, curves, profiles, font_family="Times New Roman", font_size=10,
            output_directory=STYLE_DIRECTORY, uniform_text=True, style_records=styles)
    figures = {}
    for relative in generated:
        with Image.open(root / relative) as image:
            dimensions = list(image.size)
        original = f"figures/{Path(relative).name}"
        figures[original] = {"relative_path": relative, "sha256": sha256_file(root / relative),
                             "dimensions_pixels": dimensions, **styles[relative]}
    manifest = {
        "purpose": "Main report figures rendered locally; numerical cloud measurements are unchanged.",
        "rendered_at_utc": datetime.now(timezone.utc).isoformat(),
        "rendered_on": {"platform": platform.platform(), "python_executable": sys.executable,
                        "python_version": platform.python_version(), "matplotlib": matplotlib.__version__},
        "original_execution": {key: cloud_manifest[key] for key in
                               ["run_id", "config_id", "data_id", "computed_on", "runtime"]},
        "inputs": inputs, "resolved_regular_font": font_path, "figures": figures,
    }
    if not valid_presentation(root, manifest):
        raise ValueError("The locally regenerated figures failed presentation verification.")
    atomic_json(manifest_path, manifest)
    register_main_figures(root, manifest)
    return {"status": "rendered and registered main figures", "figures": len(figures),
            "font": "Times New Roman, 10 pt", "directory": str(root / STYLE_DIRECTORY)}


def display_question(session, question, kind):
    """Display the canonical evidence paths after verifying the main figure style."""
    root = session.paths["results"]
    path = root / STYLE_MANIFEST
    if kind != "figures" or not path.is_file():
        return display_saved_question(session, question, kind)
    manifest = read_json(path)
    if not valid_presentation(root, manifest):
        raise ValueError("Main figures are stale or corrupt. Run Step 9 again to regenerate them.")
    return display_saved_question(session, question, kind)

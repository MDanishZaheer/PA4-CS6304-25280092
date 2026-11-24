# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Generate the finished stepwise notebook using only the Python standard library."""

import json
from pathlib import Path


def build_notebook():
    """Write configured compute/replay cells with the assignment's actual steps."""
    cells = []

    def markdown(text):
        """Append explanatory text as a separate notebook cell."""
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"})

    def code(text):
        """Append a short executable cell without saved or fabricated outputs."""
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip() + "\n"})

    markdown("""
# Task 1: Model Compression - Pruning and Quantization

Name: Muhammad Danish Zaheer Awan | REG id: 25280092

Compare magnitude/Wanda pruning, sensitivity-based allocation, asymmetric group PTQ,
AWQ-style scaling, mixed precision, and a 2:4 diagnostic on the **base** Qwen2.5-1.5B.
No fine-tuning is performed. All 18 mandatory conditions share fixed data and scoring.

**Colab:** compute/resume and verify Drive backups. **Local ATML environment:** replay
downloaded cloud evidence. Missing replay results raise an error; they never start training.
See `README.md` for upload locations, recovery, representation conventions, and limitations.
""")
    markdown("""
## Common setup

Keep `DRIVE_PROJECT` equal to your upload path relative to MyDrive, including both
parent folders (`ATML_PA4/PA_4`). On Colab, this notebook runs
the full protocol. Locally, its default is replay. Use the same GPU type and installed
environment when resuming a run; incompatible outputs are rejected.
""")
    code("""
# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092
from importlib.util import find_spec
from pathlib import Path
import sys

try:
    IN_COLAB = find_spec("google.colab") is not None
except ModuleNotFoundError:
    IN_COLAB = False
RUN_MODE = "compute" if IN_COLAB else "replay"
DRIVE_PROJECT = "ATML_PA4/PA_4"
DATA_SOURCE_FOLDER = "atml-pa4-datasets-main"
print(f"Task_1 | mode={RUN_MODE} | Colab={IN_COLAB}")
""")
    markdown("""
### Drive mount and fast local Colab storage

Upload only the complete `Task_1/` folder under `MyDrive/ATML_PA4/PA_4/` and open
`Task_1/main_task_1.ipynb`. Its bundled course setup files remove the need for a
separate Data upload. Code is copied to `/content/ATML_PA4/PA_4/`; results are saved to
`MyDrive/ATML_PA4/PA_4/Task_1/results/` throughout execution.
""")
    code("""
import shutil

if RUN_MODE == "compute":
    from google.colab import drive
    drive.mount("/content/drive")
    drive_root = Path("/content/drive/MyDrive") / DRIVE_PROJECT
    PROJECT_ROOT = Path("/content") / DRIVE_PROJECT
    if not (drive_root / "Task_1/main_task_1.ipynb").is_file():
        raise FileNotFoundError(f"Upload Task_1 under {drive_root} first.")
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(drive_root / "Task_1", PROJECT_ROOT / "Task_1", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("results", "__pycache__", ".ipynb_checkpoints"))
else:
    PROJECT_ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
                         if (p / "Task_1/configs/base.yaml").is_file()), None)
    if PROJECT_ROOT is None:
        raise FileNotFoundError("Open the notebook from the PA4 root or Task_1 folder.")
sys.path.insert(0, str(PROJECT_ROOT))
""")
    markdown("""
### Dependencies

Compute uses the supplied Transformers/Hub pins and the documented project pins.
If an already loaded dependency changes, restart the Colab session and Run All.
Local replay uses the existing ATML environment and never installs compute packages.
""")
    code("""
if RUN_MODE == "compute":
    from Task_1.cloud.dependencies import install_compute_requirements
    install_compute_requirements(PROJECT_ROOT / "Task_1")
else:
    print("Local replay: skipping compute dependencies, Drive, and model downloads.")
""")
    code("""
import pandas as pd
from IPython.display import display
from Task_1.common.config import load_configuration
from Task_1.common.session import open_session
from Task_1.analysis.presentation import display_question

pd.set_option("display.max_rows", 24)
configuration = load_configuration(PROJECT_ROOT / "Task_1")
configuration["execution"]["drive_project"] = DRIVE_PROJECT
configuration["execution"]["source_folder"] = DATA_SOURCE_FOLDER
print("Project:", PROJECT_ROOT)
print("Run:", configuration["execution"]["run_id"])
""")
    markdown("""
### Small correctness checks

These tests use tiny matrices and a randomly initialized tiny Qwen, never pretrained
downloads. They check pruning, packing, allocation, causal NLL, compensation/folding,
independent restoration, and verified recovery. They do not replace final experiments.
""")
    code("""
if RUN_MODE == "compute":
    import unittest
    import torch
    previous_threads = torch.get_num_threads()
    suite = unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "Task_1/tests"))
    outcome = unittest.TextTestRunner(verbosity=1).run(suite)
    torch.set_num_threads(previous_threads)
    if not outcome.wasSuccessful():
        raise RuntimeError("Correctness checks failed; full experiments have not started.")
else:
    print("Replay: using saved evidence; compute diagnostics are skipped.")
""")
    markdown("""
### Restore or validate saved progress

Compute restores files from Drive using the backup index and checksums before
opening a stage. Calibration checkpoints every 16 sequences, each projection search
and development candidate saves when complete, and evaluation saves every 16
examples/windows. Local replay validates the complete downloaded evidence, including
sample counts and regenerated aggregate scores. Incompatible runs are never merged.
""")
    code("""
session = open_session(configuration, PROJECT_ROOT, RUN_MODE)
print("Results:", session.paths["results"])
""")
    code("""
def show_stage(records):
    "Show compact completion states while preserving full saved measurements."
    rows = [{"condition": row["condition"], "status": row["status"],
             "WikiText-2 PPL": row.get("quality", {}).get("wikitext2", {}).get("perplexity"),
             "C4 PPL": row.get("quality", {}).get("c4", {}).get("perplexity"),
             "HellaSwag acc_norm": row.get("quality", {}).get("hellaswag", {}).get("accuracy_normalized")}
            for row in records]
    display(pd.DataFrame(rows))
""")
    markdown("""
## Data download and preparation

Use the supplied candidate/validation IDs unchanged. The course `--public` script
and fixed manifests are included under `Task_1/resources/course_data/`. Compute
stages their unchanged copies under runtime `Data/` before downloading datasets.
The materializer produces 128 x 512 C4 calibration tokens, 32 separate development sequences,
two independent 50,000-scored-token suites, and 2,000 HellaSwag examples.
First-run immutable HF revisions, token windows, IDs, hashes, and disjointness checks
are saved and backed up. The first 512 HellaSwag IDs are the diagnostic subset.
""")
    code("display(session.prepare_data())")
    markdown("""
## Step 1: Unchanged FP16 baseline and measurement protocol

Before **any held-out scoring**, collect shared calibration statistics, probe
projection sensitivity, choose P3 hyperparameters using only the 32 development
sequences, and freeze P3/Q3 maps. This early lock prevents baseline results from
influencing adaptive choices. Later steps apply these already frozen maps.

The unchanged model first selects a GPU batch/logit-chunk policy using calibration
inputs only, targeting useful throughput with 10% memory headroom. Quality and
development scoring reuse that saved policy; calibration batches stay within
16-sequence checkpoints. Masked padding never enters a scored-token denominator.
The separate latency benchmark batch size (default 1) is fixed across all methods
and may be configured before the run; Task 1 does not prescribe a batch size.

The scorer excludes every independent sequence's first token and scores each target
once. HellaSwag uses zero-shot conditional log likelihood and character normalization.
All conditions measure dense FP16 execution separately from estimated packed bytes.
""")
    code("display(session.prepare_calibration_and_plan())")
    code("show_stage(session.run_group(\"fp16\"))")
    steps = [
        (2, "P1: Magnitude pruning", "P1", "Prune the smallest absolute weights per output row at 25%, 50%, and 75%. Each condition restores the unchanged checkpoint; excluded parameters remain byte-identical."),
        (3, "P2: Activation-aware Wanda pruning", "P2", "Use |W_ij| times the full-calibration input-feature L2 norm, with the same row quotas and sparsities as P1. Save score distributions and local output distortion."),
        (4, "P3: Nonuniform Wanda allocation", "P3", "Apply the frozen projection-wise sensitivity allocation at global 50% and 75%. Ranking stays Wanda-based; sparsities remain within [0.10, 0.85], and integer zero counts match exactly. The temperature was selected before test evaluation."),
        (5, "Q1: Uniform asymmetric group PTQ", "Q1", "Evaluate INT8, INT4, and INT3 with 128 input-column groups. Use ties-to-even rounding, stored FP16 group scales, uint8 zero-points, zero-inclusive min/max ranges, and the documented row-aligned bit stream."),
        (6, "Q2: AWQ-style activation scaling", "Q2", "Search only the recorded calibration scale grid. Verify rescaling before quantization. Fold the down inverse into compatible up-projection rows; retain and account for independent channel compensations elsewhere. This is a simplified AWQ-style implementation, not an optimized AWQ inference engine."),
        (7, "Q3: Sensitivity-guided mixed precision", "Q3", "Apply the frozen greedy calibration-distortion allocation with bits {2,3,4,8}. Use exactly the Q2 scale search/quantizer; only bit assignments change. Total bytes include excluded weights, compensation, scales, zero-points, headers, and padding and cannot exceed Q2 INT4/INT3 budgets."),
        (8, "Structured 2:4 Wanda diagnostic", "structured", "Keep the two highest Wanda-ranked entries in each group of four contiguous input columns. Compare quality and representation with unstructured 50% Wanda; no T4 sparse-tensor-core acceleration is claimed."),
    ]
    for number, title, family, explanation in steps:
        markdown(f"## Step {number}: {title}\n\n{explanation}\n\nEach stage resumes compatible per-example progress or displays its saved completion state.")
        code(f"show_stage(session.run_group({family!r}))")
    markdown("""
## Step 9: Resource-quality comparisons and required evidence

Rebuild tables and figures from saved measurements. Compute paired 95% bootstrap
intervals for P3-P2 and Q3-Q2: resample HellaSwag examples and whole prediction
windows; pool token NLL before exponentiating. Failed conditions remain visible.
This step does not retrain or reload a model for plotting.

Local Windows replay also renders all figure text in **Times New Roman, 10 pt**
directly under `figures/` as the main report figures. Figure checksums are updated
with local formatting provenance; saved numerical measurements remain unchanged.
""")
    code("""
display(session.build_evidence())
if RUN_MODE == "replay":
    from Task_1.analysis.presentation import format_saved_figures
    display(format_saved_figures(session))
""")
    markdown("""
## Required evaluation audit

Every required condition attempts both perplexity suites. Full HellaSwag is used for
FP16, P1/P2/P3 at 50%, and all INT4/INT3 quantization variants; other conditions use
the first 512 IDs. The table distinguishes counts and compares FP16 on paired IDs.
Timing uses two warmups and at least five synchronized repetitions at fixed shapes
and the same declared benchmark batch size for every condition.
""")
    code("""
audit = pd.read_csv(session.paths["results"] / "tables/all_conditions.csv")
display(audit[["condition", "status", "wikitext2_scored_tokens", "c4_scored_tokens",
               "hellaswag_required_examples", "hellaswag_evaluated_examples"]])
""")
    markdown("# Required evidence\n\nThe sections below display saved cloud measurements. Write your own interpretation after inspecting the results; the PDF report's language, interpretation, and analysis must be your own.")
    questions = [
        ("rq1", "Magnitude, activation, and removal impact", "Compare local magnitude/Wanda reconstruction errors across sparsities and relate their differences to activation outliers. Complete projection score distributions are preserved in metrics/sensitivity_profiles.json."),
        ("rq2", "Sensitivity and constrained resource allocation", "Inspect pruning and quantization sensitivity maps, selected sparsities/bits, and exact budget checks. Isolated projection errors do not capture every interaction after simultaneous compression."),
        ("rq3", "Matched-budget quality and metric agreement", "Use pooled likelihood, raw/normalized answer selection, paired intervals, and aligned candidate-score examples. HellaSwag diagnostic plots use the same 512 IDs for every condition."),
        ("rq4", "Calibration cost, representation overhead, and real execution", "Compare estimated packed storage with measured dense FP16 memory and latency. Include activation collection, probe/search and development-selection costs; explicit AWQ compensation can add execution overhead."),
    ]
    for identifier, title, explanation in questions:
        markdown(f"## RQ-{identifier[-1]}: {title}\n\n{explanation}")
        code(f"display_question(session, {identifier!r}, 'tables')")
        code(f"display_question(session, {identifier!r}, 'figures')")
        code(f"display_question(session, {identifier!r}, 'metrics')")
    markdown("""
### Aligned qualitative cases

Cases use fixed manifest orders plus the first four adaptive/reference normalized
prediction disagreements per declared pair. The JSON retains the same prompt,
four endings, gold label, and all raw/character-normalized scores for each method.
""")
    code("""
from Task_1.common.artifacts import read_json
examples = read_json(session.paths["results"] / "examples/hellaswag_aligned.json")
print(examples["selection_rule"])
display(examples["examples"][:3])
""")
    markdown("""
## Completion, verified backup, and results download

Validate all registered artifacts and numerical counts before declaring completion.
Download `MyDrive/ATML_PA4/PA_4/Task_1/results/` and place it at your local
`Task_1/results/`, keeping the run folder inside it. Drive may package a folder
download as a ZIP; extract that folder without adding `results/results`.
Open this same notebook locally in the ATML environment and Run All to validate
and display the cloud tables, figures and per-example evidence for analysis.
""")
    code("display(session.finish())")
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "ATML", "language": "python", "name": "ATML"},
                  "language_info": {"name": "python", "file_extension": ".py", "mimetype": "text/x-python"},
                  "colab": {"name": "main_task_1.ipynb", "provenance": []}}, "nbformat": 4, "nbformat_minor": 4}
    destination = Path(__file__).resolve().parents[1] / "main_task_1.ipynb"
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile(cell["source"], f"cell_{index}", "exec")
    print(f"Built {destination.name}: {len(cells)} cells; all code cells compile.")


if __name__ == "__main__":
    build_notebook()

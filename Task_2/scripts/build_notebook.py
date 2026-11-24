# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Generate the stepwise Task 2 notebook without requiring Jupyter build packages."""

import json
from pathlib import Path


def build_notebook():
    """Expose common setup, all eleven assignment steps and all four RQ sections."""
    cells = []
    def markdown(source):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"})
    def code(source):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source.strip() + "\n"})
    markdown("""
# Task 2: Efficient Attention and Context Memory

Name: Muhammad Danish Zaheer Awan | REG id: 25280092

Part A compares exact attention implementations. Part B trains all 42 required
MQAR runs and preserves final checkpoints. Part C compares six cache policies on
the same 180 staff retrieval prompts, including actual packed-cache system costs.
Colab computes and resumes; local **ATML** Run All strictly replays saved evidence.
The eight main figures use **Times New Roman, 10 pt** from the first cloud run.
See `README.md` for methods, limitations, recovery and reproduction instructions.
""")
    markdown("## Common setup\n\nUpload this complete `Task_2/` folder to `MyDrive/ATML_PA4/PA_4/`. Raw model files download into runtime `Data/Task_2/`; supplied small benchmark assets are bundled.")
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
print(f"Task_2 | mode={RUN_MODE} | Colab={IN_COLAB}")
""")
    markdown("### Drive and local working storage\n\nFrequent reads run on `/content/ATML_PA4/PA_4/`. Results are backed up to Drive throughout the run. Copying code preserves the existing Drive results.")
    code("""
import shutil
if RUN_MODE == "compute":
    from google.colab import drive
    drive.mount("/content/drive")
    drive_root = Path("/content/drive/MyDrive") / DRIVE_PROJECT
    PROJECT_ROOT = Path("/content") / DRIVE_PROJECT
    if not (drive_root / "Task_2/main_task_2.ipynb").is_file():
        raise FileNotFoundError(f"Upload the complete Task_2 folder under {drive_root}.")
    shutil.copytree(drive_root / "Task_2", PROJECT_ROOT / "Task_2", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("results", "__pycache__", ".ipynb_checkpoints"))
else:
    PROJECT_ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
                         if (p / "Task_2/configs/base.yaml").is_file()), None)
    if PROJECT_ROOT is None:
        raise FileNotFoundError("Open this notebook from Task_2 or the PA4 root.")
sys.path.insert(0, str(PROJECT_ROOT))
""")
    markdown("### Dependencies and exact plotting fonts\n\nCompute installs the supplied cache API pins and explicit project pins. Restart Colab if an already imported package changes. Local replay installs nothing and loads no model. Font availability is checked before expensive computation.")
    code("""
if RUN_MODE == "compute":
    from Task_2.cloud.dependencies import install_compute_requirements
    install_compute_requirements(PROJECT_ROOT / "Task_2")
    from Task_2.analysis.fonts import configure_fonts
    print(configure_fonts(PROJECT_ROOT / "Task_2"))
else:
    print("Replay: no installation, Drive, downloads, model loading or training.")
""")
    code("""
import pandas as pd
from IPython.display import display
from Task_2.common.config import load_configuration
from Task_2.common.session import open_session
from Task_2.common.replay import display_question
pd.set_option("display.max_rows", 36)
configuration = load_configuration(PROJECT_ROOT / "Task_2")
configuration["execution"]["drive_project"] = DRIVE_PROJECT
print("Project:", PROJECT_ROOT)
print("Run:", configuration["execution"]["run_id"])
""")
    markdown("### Small correctness checks\n\nSynthetic tests check attention gradients, causality, true query-position labels, real packed-byte round trips, FP16 cache editing, recovery and replay isolation. They never download a pretrained checkpoint or replace the final experiments.")
    code("""
if RUN_MODE == "compute":
    import unittest
    import torch
    previous_threads = torch.get_num_threads()
    modules = ["test_attention", "test_cache", "test_mqar", "test_recovery", "test_retrieval"]
    suite = unittest.defaultTestLoader.loadTestsFromNames([f"Task_2.tests.{name}" for name in modules])
    outcome = unittest.TextTestRunner(verbosity=1).run(suite)
    torch.set_num_threads(previous_threads)
    if not outcome.wasSuccessful():
        raise RuntimeError("Correctness tests failed; full experiments have not started.")
else:
    print("Replay: validating saved cloud evidence; compute diagnostics are skipped.")
""")
    markdown("### Restore or validate saved progress\n\nReuse only compatible code/configuration, data, GPU and environment identities. Training saves full optimizer, schedule, scaler, sampler and RNG recovery every 100 updates, plus all six required historical checkpoints. Evaluations save checksummed shards or individual retrieval cases.")
    code("session = open_session(configuration, PROJECT_ROOT, RUN_MODE)\nprint('Results:', session.paths['results'])")
    code("""
def show_training(records):
    display(pd.DataFrame([{key: row[key] for key in ["id", "status", "completed_updates", "trainable_parameters", "training_seconds"]}
                          for row in records]))

def show_attention(records):
    display(pd.DataFrame([{key: row[key] for key in ["method", "dtype", "length", "status"]} |
                         {"forward_ms": row.get("forward", {}).get("median_ms"),
                          "forward_backward_ms": row.get("forward_backward", {}).get("median_ms")}
                         for row in records]))
""")
    markdown("## Data preparation\n\nVerify unchanged staff files and training-generator invariants. Development data are used for monitoring and intervention selection. All 9,216 final MQAR sequences remain fixed; no final scores are accessed until the intervention is locked and every training run has finished. Retrieval gold spans are for evaluation only.")
    code("display(session.prepare_data())")
    steps = [
        (1, "A1: Dense causal reference", "Explicit FP32 scores, stable causal softmax and value multiplication. FP16 input/output storage still uses FP32 accumulation in this reference. Use the prescribed B=2, H=4, d=64, six lengths and two dtypes; preserve unsupported/OOM conditions.", "show_attention(session.run_attention('dense'))"),
        (2, "A2: Online-softmax tiling", "The exact tiled forward keeps FP32 running maxima/normalizers. Its custom backward recomputes probability tiles. Compare output and all Q/K/V gradients with dense FP32 at fixed feasible lengths; this Python implementation is not a fused FlashAttention kernel.", "show_attention(session.run_attention('tiled'))"),
        (3, "A3: Optimized SDPA and complete operator cost", "Record actual dispatched SDPA operators. Measure forward and complete forward-plus-backward separately using three warmups and ten CUDA-synchronized repetitions at fixed shapes, with allocated and reserved memory.", "show_attention(session.run_attention('sdpa'))"),
        (4, "B1: Dense/local baselines", "Train dense mixed K=4/8/16 and dense near K=16; train W64 near K=4/8/16 and W64 mixed K=16. Every run uses all 4,500 updates and B=32. Labels classify the value at the query position, never the next position. Final-test evaluation is deferred until B3 freezes the intervention.", "show_training(session.train_group(['dense', 'window']))"),
        (5, "B2: Linear and local/dilated alternatives", "Train ELU+1 linear attention at mixed K=4/8/16 and two-local/two-dilated attention at mixed K=4/16, for all three seeds. Record dense-mask and chunked-prefix implementation costs explicitly; theoretical pair/state counts do not imply GPU savings.", "show_training(session.train_group(['linear', 'dilated']))"),
        (6, "B3: Controlled content-selection intervention", "Use only the completed dense/dilated K16 development results to freeze top-k=32 or 64 by the predeclared rule. Replace two dilated heads with causal content-selected heads and keep two W64 heads. Trainable parameters are unchanged. Full-history scoring is retained and charged; no persistent compressed-state equivalence is claimed. The saved prediction precedes final-test evaluation.", "display(session.freeze_intervention())"),
    ]
    for number, title, explanation, source in steps:
        markdown(f"## Step {number}: {title}\n\n{explanation}")
        code(source)
        if number == 6:
            code("show_training(session.train_group(['local_content']))")
            markdown("### Required final MQAR evaluation and test-only masks\n\nEvaluate every learned model on all nine frozen K/regime cells, each containing 1,024 sequences. Dense models also receive a W64 test-only mask as a separate control. Report the 1/64 label-space and 1/K present-value baselines, true source-distance bins, and training-seed dispersion.")
            code("display(session.evaluate_mqar())")
            markdown("### Independent forward-only operator sweep\n\nUse fixed random Q/K/V with B=2, H=4, d=32 at T=256/1024/4096/8192. This measures the operators independently of whether a trained model learned recall.")
            code("display(pd.DataFrame(session.run_mqar_inference()))")
    steps_c = [
        (7, "C1: Full-precision cache reference", "Load the fixed Qwen2.5-1.5B-Instruct revision in FP16/SDPA. Verify direct versus split prefill and 100% retention on six fixed prompt cases. Original absolute rotary IDs remain distinct from physical cache append indices.", "display(session.validate_kv_reference())"),
        (8, "C2: Matched-budget FP16 eviction", "Fix eight sinks and 64 recent positions. Compare recency with shared question-aware scores computed from real rotary question queries and cached keys in FP32. No correct code or gold coordinates reach this selector. Check each actual pre-question length; paired quality runs together in C5.", "display(session.check_cache_policies('eviction'))"),
        (9, "C3: Two real INT4 cache layouts", "Compare token-wise K/V with channel-wise K/token-wise V. Groups have 64 elements, FP16 scale and offset, ties-to-even rounding, and actual packed uint8 payloads. Both layouts keep the same trailing FP16 tokens, including new tokens until a complete group forms.", "display(session.check_cache_policies('quantization'))"),
        (10, "C4: Selective FP16 plus INT2 hybrid", "Keep safeguards and additional question-important positions in FP16, with all other positions in a channel-K/token-V INT2 stream. Search the legal FP16 count using actual group metadata, int32 indices, FP16 residuals and a 32-byte format header; stay within the identical INT4 reference budget.", "display(session.check_cache_policies('hybrid'))"),
        (11, "C5: Information retention and systems cost", "Evaluate all six policies on the identical 180 prompts. Save greedy generations, unconditional and FP16-correct-conditional accuracy, FP16-reference answer NLL, gold-code NLL, first-logit/teacher-forced KL, evidence masks, actual bytes and decomposed times. Confidence intervals resample all paired prompts, not individual ten-case cells.", "display(session.run_kv_quality())"),
    ]
    for number, title, explanation, source in steps_c:
        markdown(f"## Step {number}: {title}\n\n{explanation}")
        code(source)
    markdown("### Actual packed-cache systems sweep\n\nMeasure packing, unpacking and one full-model decode step including dequantization at B=1/8 and L=2048/4096, with ten synchronized repetitions each. Fixed synthetic caches use the real model geometry. Setup clones are outside timers. Packed bytes and allocated memory remain separate quantities.")
    code("display(pd.DataFrame(session.run_kv_systems()))")
    markdown("## Build required evidence\n\nCreate complete CSV grids and eight main PNGs, all with Times New Roman 10 pt text. No duplicate figure variants are created. A failed condition stays visible and qualifies completion rather than disappearing.")
    code("display(session.build_evidence())")
    markdown("# Required evidence\n\nUse these saved measurements to write your own interpretation. The assignment requires the PDF report’s language, interpretation and analysis to be your own.")
    for question, title in [("rq1", "Exact attention fidelity, backward cost and memory"),
                            ("rq2", "Distance, load, attention access and trained controls"),
                            ("rq3", "Matched-byte KV retention and answer fidelity"),
                            ("rq4", "Selection, packing and actual execution overhead")]:
        markdown(f"## RQ-{question[-1]}: {title}\n\nInspect the same saved tables and figures, keeping denominators, seeds, implementation costs and recorded failures visible.")
        for kind in ["tables", "figures", "metrics"]:
            code(f"display_question(session, {question!r}, {kind!r})")
    markdown("### Fixed aligned retrieval examples\n\nThe fixed six prompt IDs retain every policy’s response, gold span and FP16-selection mask. These examples are selected by supplied case order, never by whether a method won.")
    code("""
from Task_2.common.artifacts import read_json
examples = read_json(session.paths["results"] / "examples/kv_aligned.json")
display(pd.DataFrame([{key: row.get(key) for key in ["id", "method", "status", "correct", "evidence_retention_fraction"]} |
                      {"response": row["response"]["text"]} for row in examples]))
display({"id": examples[0]["id"], "gold_evidence_span": examples[0]["evidence_text"]})
print("Full unchanged prompts, evidence coordinates and token-selection masks: examples/kv_aligned.json")
""")
    markdown("## Completion and download\n\nValidate all registered artifacts, counts and saved predictions before declaring completion. Download `MyDrive/ATML_PA4/PA_4/Task_2/results/` into local `Task_2/results/`, preserving the run folder. Keep large checkpoints on Drive if only replay is needed. Run this same notebook locally in ATML; missing evidence raises an error and never launches training.")
    code("display(session.finish())")
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "ATML", "language": "python", "name": "ATML"},
        "language_info": {"name": "python", "file_extension": ".py", "mimetype": "text/x-python"},
        "colab": {"name": "main_task_2.ipynb", "provenance": []}}, "nbformat": 4, "nbformat_minor": 4}
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile(cell["source"], f"Task2_cell_{index}", "exec")
    destination = Path(__file__).resolve().parents[1] / "main_task_2.ipynb"
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Built {destination.name}: {len(cells)} cells; all code cells compile.")


if __name__ == "__main__":
    build_notebook()

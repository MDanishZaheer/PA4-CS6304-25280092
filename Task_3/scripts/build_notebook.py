# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Build the standalone seven-step Task 3 compute/replay notebook."""

import json
from pathlib import Path


def build_notebook():
    cells = []
    def markdown(source):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"})
    def code(source):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source.strip() + "\n"})
    markdown("""
# Task 3: Policy Distillation and Speculative Decoding

Name: Muhammad Danish Zaheer Awan | REG id: 25280092

Implement exact stochastic speculative sampling, train four independent LoRA
drafts, and compare probability alignment with actual execution costs.
Colab computes and resumes. Local **ATML** Run All validates and displays saved
Colab evidence. Every main figure uses **Times New Roman, 10 pt**.
""")
    markdown("## Common setup\n\nUpload the complete `Task_3/` folder to `MyDrive/ATML_PA4/PA_4/`. Code runs on local Colab storage; verified results and recovery checkpoints are backed up to Drive.")
    code("""
# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092
from importlib.util import find_spec
from pathlib import Path
import sys
try:
    IN_COLAB = find_spec('google.colab') is not None
except ModuleNotFoundError:
    IN_COLAB = False
RUN_MODE = 'compute' if IN_COLAB else 'replay'
DRIVE_PROJECT = 'ATML_PA4/PA_4'
print('Task_3 |', RUN_MODE, '| Colab:', IN_COLAB)
""")
    markdown("### Drive and working storage\n\nDownloaded model/data caches belong in runtime `Data/Task_3/`; this folder's `data/` package contains Python code. Preserve existing results when replacing uploaded code.")
    code("""
import shutil
if RUN_MODE == 'compute':
    from google.colab import drive
    drive.mount('/content/drive')
    drive_root = Path('/content/drive/MyDrive') / DRIVE_PROJECT
    PROJECT_ROOT = Path('/content') / DRIVE_PROJECT
    if not (drive_root / 'Task_3/main_task_3.ipynb').is_file():
        raise FileNotFoundError(f'Upload the complete Task_3 folder into {drive_root}.')
    shutil.copytree(drive_root / 'Task_3', PROJECT_ROOT / 'Task_3', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('results', '__pycache__', '.ipynb_checkpoints'))
else:
    PROJECT_ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
                         if (p / 'Task_3/configs/base.yaml').is_file()), None)
    if PROJECT_ROOT is None:
        raise FileNotFoundError('Open this notebook from Task_3 or the PA4 root.')
sys.path.insert(0, str(PROJECT_ROOT))
""")
    markdown("### Dependencies and exact plotting fonts\n\nCompute checks the pinned Transformers/cache stack and genuine Times New Roman files before expensive work. Restart the runtime if an already imported package changes. Replay installs nothing and loads no model.")
    code("""
if RUN_MODE == 'compute':
    from Task_3.cloud.dependencies import install_compute_requirements
    install_compute_requirements(PROJECT_ROOT / 'Task_3')
    from Task_3.analysis.fonts import configure_fonts
    print(configure_fonts(PROJECT_ROOT / 'Task_3'))
else:
    print('Local replay: no installation, downloads, training or model loading.')
""")
    code("""
import pandas as pd
from IPython.display import display
from Task_3.common.config import load_configuration
from Task_3.common.pipeline import open_session
from Task_3.common.replay import display_question
configuration = load_configuration(PROJECT_ROOT / 'Task_3')
configuration['execution']['drive_project'] = DRIVE_PROJECT
pd.set_option('display.max_rows', 36)
print('Run:', configuration['execution']['run_id'])
""")
    markdown("### Small mathematical and implementation checks\n\nThe joint Markov test uses 100,000 length-three sequences for each gamma and each correct/incorrect residual control. Tiny random Qwen tests check cache positions, bonus tokens, padding, EOS, merging, full soft-KL targets and complete training-state recovery. These are diagnostics, not assignment-model results.")
    code("""
if RUN_MODE == 'compute':
    import unittest
    import torch
    threads = torch.get_num_threads()
    modules = ['test_sampling', 'test_lora_data', 'test_training', 'test_evaluation', 'test_recovery']
    suite = unittest.defaultTestLoader.loadTestsFromNames([f'Task_3.tests.{name}' for name in modules])
    outcome = unittest.TextTestRunner(verbosity=1).run(suite)
    torch.set_num_threads(threads)
    if not outcome.wasSuccessful():
        raise RuntimeError('Small correctness checks failed; full experiments have not started.')
else:
    print('Replay verifies the saved cloud experiment instead of executing compute diagnostics.')
""")
    markdown("### Restore compatible progress or validate downloaded evidence\n\nRecovery includes adapters, optimizer/scaler, quota counters, prompt-schedule state, pending response, histories and RNGs every five successful updates. Scientific code, data, GPU and package identities must match. Local replay never falls back to computation.")
    code("session = open_session(configuration, PROJECT_ROOT, RUN_MODE)\nprint('Results:', session.paths['results'])")
    markdown("## Data preparation\n\nUse the exact 500 GSM8K training IDs and 300 supplied short-form training prompts. Freeze the 120 held-out prompts: 60 math, 30 familiar extraction, 30 shifted extraction. Download only the pinned GSM8K source rows; verify all 920 teacher/student chat encodings. No raw Data upload is needed.")
    code("display(session.prepare_data())")
    steps = [
        (1, 'Exact speculative sampling and cache correctness',
         'Use stochastic min(1,p/q) acceptance, the normalized positive residual on rejection and a target bonus token after full acceptance. A single target verification handles proposals plus the previous pending token. Both caches retain the committed context except that one pending token. Preserve actual draft/target work and all four synchronized round-time components. Validate joint laws and real-model split-prefill/crop equivalence before timing.',
         'display(session.validate_sampler())'),
        (2, 'Pretrained draft baseline',
         'Benchmark the original 0.5B draft at gamma=2/4, two paired repetitions on all 120 prompts, EOS-aware cap 64. Each speculative request has an adjacent target-only control. Save warm setup, prefill, steady decode, accepted/examined and accepted/proposed counters, target passes and actual draft work. Isolate single-step and verification costs at context lengths 64/256/512.',
         'display(session.baseline())'),
        (3, 'Off-policy full-vocabulary distillation',
         'For each seed 6304/6305, first freeze an 800-prompt teacher rollout pool. Independently initialize FP32 rank-8 LoRA adapters on the frozen FP16 original student. Score exactly 30,000 math and 10,000 short-form response prefixes, accumulating one update per 256 tokens with a final 64-token update. Teacher soft probabilities, never sampled hard tokens, define forward KL.',
         "display(pd.DataFrame(session.train_method('off_policy')))"),
        (4, 'On-policy full-vocabulary distillation',
         'Independently restart from the original backbone and matching seeded LoRA initialization. Generate fresh current-student responses using the same seeded category round-robin schedule and response caps. Teacher-score these prefixes to the identical 30,000/10,000 quotas. Freeze rollout generation from autograd; count student generation separately. Reuse scored soft targets during scaler retries instead of adding teacher-scored tokens.',
         "display(pd.DataFrame(session.train_method('on_policy')))"),
        (5, 'Common and own-policy prefix banks',
         'The supplied data-only checkout has no materialized prefix banks. Generate and freeze teacher, original-student and exact-original-speculative banks, plus all four trained students\' own banks. Score all five drafts on every bank with identical saved prefix IDs, up to 32 positions in 1-64 and a separate long diagnostic at 65-192. Teacher and exact-speculative banks are independent realizations of the same target law. Own-bank cross-evaluation retains named producing checkpoints.',
         'display(session.alignment())'),
        (6, 'Merged drafts and measured speculative execution',
         'Merge every adapter into a fresh FP16 backbone and verify logits against the corresponding unmerged model. Primary benchmarks use ordinary merged projections, all 120 prompts, both gamma values and two repetitions. Reverse model order across repetitions, alternate gamma order and counterbalance adjacent target/speculative order. Measure one fixed unmerged-adapter control separately.',
         'display(session.merged_execution())'),
        (7, 'Workload, response length, quality and deployment cost',
         'Run an additional fixed-64-token diagnostic on the first 24 supplied math prompts, ignoring EOS for timing only. Evaluate standalone teacher/draft answer quality separately at 512 math / 32 extraction tokens, retaining extraction failures and truncations. Build prompt-paired 95% intervals and compare measured costs with the analytical model; a nonpositive saving has no finite deployment break-even.',
         'display(session.workload_evaluation())')]
    for number, title, explanation, source in steps:
        markdown(f'## Step {number}: {title}\n\n{explanation}')
        code(source)
    markdown('## Build required evidence\n\nAll required stages must finish before the final evidence registry exists. Generate machine-readable tables and nine main PNGs, all Times New Roman 10 pt. Poor outcomes remain; there is no tuning on held-out prefixes or final accuracy.')
    code('display(session.build_evidence())')
    markdown('# Required evidence\n\nUse the saved numerical evidence to write your own report language, interpretation and analysis.')
    for question, title in [('rq1', 'Prefix distribution, soft distillation and target-law consistency'),
                            ('rq2', 'Workload, output length, prefix position and standalone quality'),
                            ('rq3', 'Probability overlap, verification work and projected versus measured speedup'),
                            ('rq4', 'Draft/target costs, adapter overhead, EOS and training amortization')]:
        markdown(f'## RQ-{question[-1]}: {title}')
        for kind in ['tables', 'figures', 'metrics']:
            code(f'display_question(session, {question!r}, {kind!r})')
    markdown('### Fixed matched-prompt continuations\n\nUse fixed prompt indices [0,30,60,90], repetition 0, with every draft and both proposal lengths. These are short speed-test continuations and must not be scored as completed mathematical answers.')
    code("""
from Task_3.common.artifacts import read_json
examples = read_json(session.paths['results'] / 'examples/matched_continuations.json')
display(pd.DataFrame([{key: row[key] for key in ['prompt_id', 'draft', 'gamma', 'workload', 'target_text', 'speculative_text']} for row in examples]))
prefixes = read_json(session.paths['results'] / 'examples/aligned_prefixes.json')
display(pd.DataFrame([{key: row.get(key) for key in ['prefix_id', 'draft', 'position', 'forward_kl', 'tv', 'target_next_token', 'student_next_token', 'target_next_token_text', 'student_next_token_text']} for row in prefixes]).head(30))
print('Full fixed matched-prefix contexts and disagreements: examples/aligned_prefixes.json')
""")
    markdown('## Completion and download\n\nDownload `MyDrive/ATML_PA4/PA_4/Task_3/results/` into local `Task_3/results/`, preserving the run directory. Large recovery and merged checkpoints may stay on Drive for replay. Local ATML Run All displays the verified Colab evidence. Delete the downloaded ZIP only after audit and replay pass.')
    code('display(session.finish())')
    for index, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            compile(cell['source'], f'Task3_cell_{index}', 'exec')
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'ATML', 'language': 'python', 'name': 'ATML'},
        'language_info': {'name': 'python', 'file_extension': '.py', 'mimetype': 'text/x-python'},
        'colab': {'name': 'main_task_3.ipynb', 'provenance': []}}, 'nbformat': 4, 'nbformat_minor': 4}
    destination = Path(__file__).resolve().parents[1] / 'main_task_3.ipynb'
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Built {len(cells)} cells, {sum(c["cell_type"] == "code" for c in cells)} code cells.')


if __name__ == '__main__':
    build_notebook()

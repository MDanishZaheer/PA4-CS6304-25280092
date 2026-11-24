# Task 1: Model compression

Name: Muhammad Danish Zaheer Awan  
REG id: 25280092

The entry point is **`main_task_1.ipynb`**, in this task's root. It implements all
18 required conditions, with no pretrained fine-tuning. Notebook cells expose the
assignment steps; modules contain the computation. Four research-question sections
display saved tables, figures, metrics, and aligned examples after the experiments.

## 1. Upload and launch on Colab

1. Use `MyDrive/ATML_PA4/PA_4/` and upload the complete local **`Task_1` folder**
   into it. The folder includes the unchanged course preparation script and fixed
   manifests, so a separate `Data` upload is unnecessary.
2. Open `MyDrive/ATML_PA4/PA_4/Task_1/main_task_1.ipynb` with Google Colab.
3. Select a CUDA GPU. A 16 GB or larger GPU is recommended; L4/A100 provide more
   memory headroom. Record and reuse the **same GPU type** for this run's matched
   timing comparisons. High system RAM is useful but is not a protocol requirement.
4. Keep `DRIVE_PROJECT = "ATML_PA4/PA_4"`, the nested path relative to MyDrive.
   The dataset folder name is configured beside it in the first setup cell.
5. Use **Run All**. Authorize the Drive mount when prompted. The setup copies code
   to `/content/ATML_PA4/PA_4/`, installs/checks compute dependencies, and runs small
   correctness tests before opening the experiment session.
6. If dependencies already imported by Colab change, the setup stops with a
   restart instruction. Use **Runtime > Restart session**, then Run All. Installed
   matching packages are reused.
7. The supplied `--public` materializer downloads C4/WikiText/HellaSwag and the
   small GSM8K selections. No Task 3 prefix banks are generated. C4 streaming may
   take time; progress is printed by the supplied script.
8. The session collects calibration features and freezes adaptive allocations
   **before evaluating even the baseline on held-out data**. Each condition then
   restores original eligible weights, evaluates, and measures resources.
9. Results are backed up throughout execution to
   `MyDrive/ATML_PA4/PA_4/Task_1/results/<run_id>/`. Wait for validated completion;
   the final cell prints the `results` folder to download for local replay.

Upload the folder directly; no upload archive or archive extraction step is needed.
Existing Drive results must be preserved when uploading updated code. Later,
`Task_2/` and `Task_3/` will be siblings under the same Drive project.

```text
MyDrive/ATML_PA4/PA_4/
|-- Task_1/
|   |-- main_task_1.ipynb
|   |-- configs/, common/, data/, models/, ...
|   |-- resources/course_data/      # Supplied setup script and fixed manifests
|   '-- results/
|       '-- task1_seed6304_v1/      # Written and backed up by Colab
|-- Task_2/                        # Add when implemented
'-- Task_3/                        # Add when implemented
```

To refresh the bundled setup files from the existing local data repository, use
`python Task_1/scripts/prepare_upload.py` in ATML. They are already included.

### Environment

`requirements-public.txt` is the only supplied dependency reference. It fixes
Transformers 4.57.1 and Hub 0.36.2 but does not supply a complete course lock.
`requirements-compute.txt` therefore adds explicit project pins, including PyTorch
2.8.0 and matching optional torchvision/torchaudio versions to avoid incompatible
preinstalled Colab packages. No packages are installed into the local ATML
environment by this notebook. Full `pip freeze`, Python/package/CUDA versions,
checkpoint SHA, device, precision, and observed SDPA operators are saved with the run.
The pinned Colab environment itself must pass the setup diagnostics on the actual
assigned runtime; local validation uses the existing ATML versions.

### GPU batching and useful VRAM usage

Task 1 prescribes no batch size. Before any held-out scoring, the unchanged FP16
model probes increasing sequence batches and vocabulary-head chunks using only
the supplied calibration inputs. The default selection target allows up to 90% of
device memory, also accounting for currently free memory, and chooses the highest
observed scoring throughput among probes within that headroom. This is a workload
selection heuristic; it does not reserve dummy tensors or impose an allocator quota.
Actual memory usage varies by phase and input length and may be below the target.

The selected sequence batch, logit chunk, memory peaks, tuning cost and all probe
outcomes are saved in `metrics/gpu_workload_plan.json` and backed up to Drive.
The same plan is reused for every compression condition and restored on resume.
Larger GPUs can process more work together; the code is no longer limited to one
sequence per forward pass. Calibration uses batches within each 16-sequence
checkpoint boundary. Quality/development scoring batches independent sequences;
HellaSwag batches candidate endings. Right padding is attention-masked and never
scored. Longer inputs reduce batch size under a fixed padded-token budget; no
context or ending is truncated. Small FP16 rounding differences from batching are
checked against the independent scoring reference before evaluation.

These execution choices preserve the required 128 calibration sequences,
32 development sequences, target-token totals, manifest order and scoring rules.
Checkpoint boundaries remain every 16 calibration sequences or evaluation items.
Settings are under `execution.gpu_memory_target_fraction`, `gpu_batch_candidates`
and `gpu_head_chunk_candidates`; change them before starting a run.

Latency benchmarks use a separately declared `benchmark.batch_size` (default 1),
held fixed for all methods. This default measures individual-request latency and
is our experimental choice, not an assignment-imposed batch cap. A larger common
value may be chosen before the run; timings are labeled milliseconds per batch.
The runtime never changes benchmark batch size independently for different methods.

All downloaded datasets, tokenizer/model caches, and temporary materializations
are under repository-root `Data/`, which is ignored by Git. The supplied data
repository's setup script, fixed public manifests, checksum list and requirements
are included unchanged under `resources/course_data/`. Colab copies these small
preparation inputs into runtime `Data/atml-pa4-datasets-main/`; actual datasets and
model weights download there on demand. No GitHub fetch is needed. Relevant inputs,
including the GSM8K IDs used by the unchanged public script, are verified against
the supplied SHA256 manifest. The repository's global
checksum list contains an unrelated absent legacy staff file, so Task 1 validates
its relevant script/manifests rather than treating that stale entry as missing data.

## 2. Recovery and compatible reuse

Durable results are under
`MyDrive/ATML_PA4/PA_4/Task_1/results/task1_seed6304_v1/`. Colab's working results are
under `/content/ATML_PA4/PA_4/Task_1/results/task1_seed6304_v1/`.

Writes use complete local files, a temporary Drive copy, size/checksum verification,
atomic publication, and a backup index published last. A previous verified version
is retained until replacement is confirmed. Sync failures stop execution instead
of continuing under an unverified backup. The verified index is used to restore
progress on a new runtime.

The initial source lock is backed up **before** public-data downloads. Interrupted,
uncommitted generated files are preserved in a recovery folder and rebuilt from
that same source lock. The data lock and actual materialized Task 1 records are backed up, so restarting
cannot resolve a new upstream revision or select different windows. Calibration
saves its raw feature sums, ordered probe samples, and completed sequence count
every **16 sequences**, then saves final activation statistics. These replace the
same checkpoint through verified Drive publication. A fresh runtime resumes from
the last saved prefix without double-counting tokens. The interval is configured
by `execution.calibration_save_every_sequences`.

Each completed projection's sensitivity/scale search is backed up independently.
Each development candidate and completed condition is also saved. Evaluations save
after every **16 examples/windows** and at the end of each suite; a restart reuses
the complete saved prefix and can repeat up to 15 unsaved items in calibration or
evaluation under the default intervals. No trained weights need recovery because this task performs
no training; compressed weights are rebuilt deterministically from the original
checkpoint and frozen policy.

On a fresh Colab runtime, select the same GPU type and Run All. Exact protocol,
source-code, data, model, software and hardware identities are checked. A changed
scientific setting or implementation requires a new `execution.run_id` or restoration
of the original configuration/code; incompatible evidence is never overwritten or
combined. Never change adaptive settings after viewing held-out outcomes.

The default session reserve stops at an item boundary after roughly 9 h 55 min.
This is a recovery convenience, not a promise of Colab runtime availability.
Reconnect and Run All to continue. A CUDA OOM or non-finite condition is recorded
and retained; unexpected implementation errors stop for correction. A run with
failed required conditions is labeled `complete_with_recorded_failures`, not an
unqualified success. Timing OOMs preserve completed quality results.

## 3. Local replay in ATML

1. Confirm final Drive completion and download
   `MyDrive/ATML_PA4/PA_4/Task_1/results/`.
2. Put that folder at **local `Task_1/results/`**, preserving
   `results/<run_id>/manifests/run.json` and its sibling evidence folders. Drive may
   ZIP a folder during download; extract it to this layout, avoiding
   `results/results`. Internal manifests verify the downloaded evidence.
3. Use the course environment:

```text
conda activate ATML
python -m pip install -r Task_1/requirements-replay.txt
python -m jupyter lab
```

4. Select the ATML kernel and Run All in `Task_1/main_task_1.ipynb`.

Local replay validates checksums, run identity, materialized window/choice identity,
sample counts, pooled NLL/perplexity, candidate scores, accuracy, required timings,
and RQ artifacts. It imports neither torch nor transformers, installs no compute
packages, downloads no raw data, loads no model, mounts no Drive, and does not
silently rescore or train. Missing evidence raises an actionable error. Recovery
checkpoints can remain on Drive; they are not needed to display results. Downloading
the whole results folder also works. Local Run All displays the saved tables,
figures and four RQ sections for our analysis.

Step 9 regenerates the five figures locally with **Times New Roman at 10 pt for
all text**, including legends, ticks, titles, and color bars. Use the PNG files in
`results/<run_id>/figures/` for the report. These are the only main figures.
`figures/style_manifest.json` records the
resolved font files, text sizes, original GPU provenance, and input/output hashes.
Figure checksums are updated with explicit local formatting provenance; numerical
cloud evidence remains unchanged. If Times New Roman is unavailable, replay
explicitly reports this and displays the existing saved figures.

## 4. Fixed experiment matrix and evaluation

| Family | Conditions | HellaSwag |
|---|---|---|
| FP16 | unchanged base model | 2,000 IDs |
| P1 magnitude | 25%, 50%, 75% | 2,000 at 50%; otherwise 512 |
| P2 Wanda | 25%, 50%, 75% | 2,000 at 50%; otherwise 512 |
| P3 nonuniform Wanda | global 50%, 75% | 2,000 at 50%; otherwise 512 |
| Q1 uniform group PTQ | INT8, INT4, INT3 | 2,000 at INT4/3; otherwise 512 |
| Q2 AWQ-style PTQ | INT8, INT4, INT3 | 2,000 at INT4/3; otherwise 512 |
| Q3 mixed AWQ-style | <= uniform-Q2 INT4 and INT3 bytes | 2,000 each |
| Structured Wanda | contiguous 2:4 | 512 |

Every condition attempts exactly 50,000 scored WikiText-2 test tokens and 50,000
separate C4 validation tokens. Documents/windows remain independent, no state is
carried between them, the first token is unscored, and each target is scored once.
The code saves window NLL sums and denominators; PPL is exp(pooled token NLL).
Vocabulary-head chunks reduce temporary logits memory without truncating contexts
or targets. HellaSwag scores all four full endings, with the same zero-shot context
and a space delimiter. `acc_norm` divides by **preprocessed ending character count**,
excluding the delimiter, not tokenizer token count. Task formatting follows the
version-1.0 HellaSwag task behavior in lm-evaluation-harness v0.4.9.

The complete summary clearly labels full versus diagnostic counts. FP16 changes
are paired on each method's actual IDs, and diagnostic plots use the same first
512 IDs across every method. Bootstrap intervals use 2,000 paired resamples of
HellaSwag examples or complete prediction windows, pooling NLL before exponentiation.

## 5. Compression and allocation definitions

Eligible matrices are exactly `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, and `down_proj` in every block. Embeddings, norms, biases and LM head stay
uncompressed and byte-identical; tied parameters are counted once. CPU originals
restore every condition independently, including removal of previous compensation
wrappers. The model is frozen and evaluated under inference mode.

### Pruning

P1 uses absolute weight magnitude. P2 uses magnitude times the input feature's L2
norm over all 128 x 512 calibration tokens. Stable column-index tie breaking is
used, with existing zeros ranked first. Exact integer per-projection zero counts
are spread over output rows with row quotas differing by at most one.

P3's sensitivity is the isolated relative projection-output MSE of 50% Wanda
pruning. For sensitivity `d_l`, assign
`s_l = clip(lambda * (d_l / median(d)) ** (-temperature), .10, .85)`.
Bisection finds lambda under the parameter-weighted budget, followed by bounded
largest-remainder integer adjustment. The fixed temperatures `{.25, .5, 1}` are
compared only using pooled NLL on all 32 development sequences, separately at each
target; ties choose the smaller temperature. The final actual eligible zero count
must equal its uniform target exactly. This is a reconstruction-based sensitivity
design, informed by Wanda and reconstruction/allocation literature; it is not OWL,
HAWQ or SparseGPT reproduced in full.

Norms and absolute activation statistics use **every calibration token**. To bound
local reconstruction/search cost, the probes use four deterministic positions per
sequence, covering every one of the same 128 sequences (512 sampled inputs per
projection). Positions, IDs, statistics and the sampling rule are saved. Probe
error curves and magnitude/Wanda score distributions are retained per projection.

### Q1 and the shared quantizer

Groups contain 128 contiguous input columns of one output row. Pad the final group
with zeros if necessary. Include zero in min/max, use an asymmetric integer range
`[0, 2**b - 1]`, ties-to-even rounding, a stored FP16 scale, and an integer zero-point
stored in one uint8 per group. The zero-point is computed from the **stored** scale.
All-zero groups use scale 1; the smallest positive stored scale is 2**(-24).
Numerical evaluation uses the resulting reconstructed FP16 weights.

The implemented bit packer uses a little-endian stream, byte-aligned per output
row, including cross-byte INT3 values. Round-trip tests compare actual payload
bytes and reconstructed weights. Production full-model sizes are explicitly
**format estimates**, as allowed when a low-bit/sparse execution kernel is absent.

The specified model format reserves a 128-byte global header and a 64-byte header
per eligible projection. Each projection's integer payload, FP16 scales, uint8
zero-points and explicit FP16 channel compensations are padded to 16-byte alignment.
Excluded raw FP16 parameters are included once. General-purpose PyTorch/ZIP overhead
is not presented as this binary format. Dense FP16 raw parameter bytes are shown
alongside the matched format estimate.

### Q2 scaling and compensation

Search scales `mean_abs_input ** exponent`, geometrically normalize, clamp to
`[.05, 20]`, and store in FP16. Search the fixed exponents `{0, .25, .5, .75, 1}` by
calibration output distortion for each projection and bit width. This reproduces
activation-derived channel scaling and local calibration search, with a documented
small grid. It omits published AWQ's optimized kernel/packing engine and broader
block/clipping search; it is labeled **AWQ-style** throughout.

Full-precision FP32 projection rescaling equivalence is verified before rounding.
The full FP16 model compensation graph is also tested on a calibration sequence,
including the algebraically compatible down/up fold. Because
`down(silu(gate(x)) * up(x))` permits dividing up-output rows by the down-input scale,
that inverse is folded into the up weights **before up is quantized**. Independent
q/k/v and gate/up scales cannot all be folded into their shared RMSNorm; unconstrained
o-input scales do not commute through GQA head repetition. These retain explicit
input divisions and serialized FP16 compensation vectors, whose storage and runtime
overhead are counted. Original biases are never quantized or changed.

### Q3 precision allocation

Use the identical cached Q2 scale search/quantizer at allowed bits `{2,3,4,8}`.
Sensitivity is isolated calibration output distortion at each candidate precision;
the visualization also shows the INT3 distortion. Start each projection at 2 bits,
then greedily select the feasible upgrade with the largest reduction in
parameter-weighted relative MSE per extra **packed byte**, allowing jumps over an
unhelpful intermediate precision. Stop when no beneficial feasible upgrade remains.
All metadata, compensations, excluded weights, headers and alignment are included.
Two policies are frozen under the uniform-Q2 INT4/INT3 total-byte ceilings.
Unused budget and the full upgrade trace are saved. No test scores or checkpoint
search enter this fixed rule. Equal sensitivities can yield a uniform outcome,
which is explicitly labeled rather than concealed.

Isolated sensitivity, sampled local reconstruction errors, and greedy allocation
do not model all interactions between simultaneously compressed projections.
Up/down folding also changes upstream quantization errors. The evidence includes
actual post-compression local errors (returned to original output units), full
held-out quality and a calibration-versus-held-out diagnostic to examine these limits.

## 6. Systems measurements and evidence locations

Unstructured sparse estimates use one mask bit per eligible weight plus 16 bits per
nonzero, with byte padding and the specified headers/alignment. The 2:4 diagnostic
also records an alternative three-bit index per group of four. Neither dense zeros
nor fake-quantized FP16 tensors are called hardware-accelerated representations.

Every condition measures prefill at 512/2,048 tokens and 64 cached decode
forward/argmax steps at both lengths. Decode prefill is untimed for that component;
the common benchmark batch size defaults to 1 and is explicitly recorded.
EOS is ignored only to keep this systems workload fixed. There are two warmups
and five CUDA-synchronized timed repetitions. Raw repetitions, medians, peak allocated
memory and observed SDPA dispatch operators are saved. Decode means **64 cached
forward steps**, not a claim about an EOS-aware end-to-end generation workload.
Compression/restore, calibration, probe/search, development selection and scoring
times are separately labeled. Calibration/probe peaks and condition-wide peaks are
saved alongside per-shape timed-execution peaks. Restarted sessions may repeat unsaved items; these
operational retries are not claimed to be free or included in per-item timing sums.

Within `results/<run_id>/`:

| Folder | Saved evidence |
|---|---|
| `configs/` | resolved settings, installed environment, frozen allocations |
| `splits/` | IDs, revision lock, hashes, exact materialized Task 1 records |
| `manifests/` | run identity, checksummed stage status, completion contract |
| `checkpoints/` | large calibration cache and scale-search recovery; keep on Drive |
| `metrics/` | quality, per-projection probes/errors/policies, storage, timing, intervals |
| `predictions/` | window NLL/counts and every HellaSwag candidate score |
| `tables/` | all-condition grid, paired intervals, maps, budgets, qualitative cases |
| `figures/` | five main 300-dpi Times New Roman 10 pt figures and their style manifest |
| `examples/` | aligned fixed cases and deterministically selected disagreements |

The main report figures use verified Times New Roman at 10 pt throughout. Local
replay replaces any initial cloud plots with these figures when the font is available.
Tables retain fractions; plots labeled `%` multiply
rates by 100. Absolute accuracy changes become percentage points when presented.
The student must write the PDF report's language, interpretation and analysis.

## 7. Local validation and implementation attribution

```text
conda activate ATML
python -m unittest discover -s Task_1/tests -v
python Task_1/scripts/validate_project.py
```

Tests cover exact row/global budgets, 2:4 structure, all bit-packing boundaries,
metadata bytes, constant groups, full-model scaling/folding/restoration on a random
tiny Qwen, chunked causal NLL against the model's standard loss, paired bootstrap
denominators, artifact corruption, Drive restoration, interrupted calibration with
exact resumed statistics, masked batched scores, batched checkpoint boundaries,
GPU policy reuse, configured benchmark batches, and the replay import boundary.
Project checks compile all Python/notebook cells and inspect the configured layout.
These are local diagnostics; full Qwen checkpoint quality, memory, timings and the
actual pinned Colab stack are validated only when the student runs Colab.

Methods were implemented for this assignment with LLM coding assistance. The
following references explain the algorithms and conventions; simplified designs
are not represented as complete reproductions of their optimized systems:

- Wanda: Sun et al., [A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695).
- AWQ: Lin et al., [Activation-aware Weight Quantization](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html).
- Reconstruction-based sensitivity context: [BRECQ](https://arxiv.org/abs/2102.05426),
  [HAWQ-V2](https://proceedings.neurips.cc/paper_files/paper/2020/hash/d77c703536718b95308130ff2e5cf9ee-Abstract.html).
- HellaSwag text preprocessing and character-normalized multiple-choice convention
  are independently implemented to match the MIT-licensed EleutherAI harness
  v0.4.9: [task preprocessing](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9/lm_eval/tasks/hellaswag/utils.py),
  [task version](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9/lm_eval/tasks/hellaswag/hellaswag.yaml),
  [score aggregation](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9/lm_eval/api/task.py),
  [license](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9/LICENSE.md).
  No harness model implementation or evaluation package is copied into this project.
- The course-supplied data repository remains unchanged under local
  `Data/atml-pa4-datasets-main/`. Its setup script and fixed public preparation
  inputs are copied unchanged into `Task_1/resources/course_data/` for uploading
  this task alone; their original hashes are verified and recorded.
- Pretrained weights/tokenizer: Qwen/Qwen2.5-1.5B base, resolved immutable revision.
  PyTorch, Transformers, Hugging Face Hub/Datasets, NumPy, pandas and Matplotlib
  versions are recorded with the run.

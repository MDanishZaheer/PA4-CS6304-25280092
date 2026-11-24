# Task 2: Efficient Attention and Context Memory

Name: Muhammad Danish Zaheer Awan  
REG id: 25280092

Use **`main_task_2.ipynb`**. It follows all eleven assignment steps and displays
the required evidence under four research questions. Full experiments run on
Colab; local Run All in **ATML** validates and displays downloaded cloud results.
The completed A100 run is saved under `results/task2_seed6304_v1/`. Its downloaded
artifacts, final predictions, tables, byte budgets and paired intervals passed the
local audit. All 38 notebook code cells have been replayed in ATML with model
imports blocked; the notebook contains its eight saved main figures.

Local verification receipts are `metrics/local_run_audit.json`,
`metrics/local_notebook_replay.json` and `metrics/local_readiness_review.json`
inside that run folder. These records preserve the original Google Colab provenance.

## Upload and run

1. Upload the complete **Task_2 folder** to `MyDrive/ATML_PA4/PA_4/`.
2. Open `Task_2/main_task_2.ipynb` in Colab and select a CUDA GPU. A 22 GB L4 or
   40 GB A100 can run the protocol; actual unsupported/OOM benchmark cases are
   retained. Use the same GPU type and environment when resuming matched timings.
3. Keep `DRIVE_PROJECT = "ATML_PA4/PA_4"`. Use Run All. Mount Drive when prompted.
4. If an already imported dependency changes, restart the Colab runtime and Run All.
5. Code runs under `/content/ATML_PA4/PA_4/Task_2/`; downloaded models stay under
   `/content/ATML_PA4/PA_4/Data/Task_2/`. Verified result backups go to
   `MyDrive/ATML_PA4/PA_4/Task_2/results/task2_seed6304_v1/` throughout execution.

This task is self-contained. Its unchanged staff generator, frozen MQAR files and
180 retrieval prompts are bundled under `resources/course_data/`; no separate Data
upload or GitHub dataset fetch is required. `data/` contains required Python code.
Initial data preparation downloads only the small pinned tokenizer/config, validates
all 180 chat prompts and all 900 compressed-cache budget plans, and then starts the
attention/training sections. The pretrained weight download is deferred to Part C.
Upload the folder directly. No upload ZIP is needed. Preserve existing Drive results
when replacing code. Task 1 is independent and is not imported or modified.

### Plotting fonts from the start

All eight main figures are 300-dpi PNGs directly in `results/<run_id>/figures/`.
Every text element, including ticks, legends, titles and colorbars, uses verified
**Times New Roman, 10 pt**. There are no alternative/duplicate figure folders.

The prepared local folder includes personal Windows font files under
`resources/fonts/` for the private Drive upload. They are ignored by Git. When
cloning the public repository, supply your own Times New Roman files there, or run
`python Task_2/scripts/prepare_upload.py` in your Windows ATML environment. Colab
checks the exact font before expensive experiments; it never silently substitutes
another family. Local replay of saved PNGs does not require installing fonts.

### Environment

The supplied requirements specify Transformers 4.57.1 and Hub 0.36.2; no additional
course lock was provided. `requirements-compute.txt` records explicit project pins,
including PyTorch 2.8.0 and matching optional vision/audio packages to avoid stale
Colab version conflicts. All actual package/CUDA/Python/GPU versions and `pip freeze`
are saved. The pretrained model/tokenizer are pinned to revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, the Qwen2.5-1.5B-Instruct revision
already resolved in the supplied-course workflow. The fixed revision is recorded
before model download or scoring.

## Experiments

### A: exact attention

The complete matrix contains 36 conditions: dense, tiled and SDPA; FP16/FP32;
L=256/512/1024/2048/4096/8192; B=2, H=4, head dimension 64. Every supported condition
has three warmups and ten synchronized timing repetitions. Forward-only and full
forward-plus-backward times, allocated/reserved memory and actual SDPA operators
are separate. Dense FP32 references are attempted at every feasible length using
a recorded memory guard. References through L=2048 are required for supported
conditions; larger shapes are still benchmarked when their dense reference is
infeasible or exhausts memory.

Dense/tiled references explicitly use FP32 score/statistic accumulation even for
FP16 input/output tensors. The tiled implementation has an online-softmax forward
and a custom blockwise-recomputed backward. It stores Q/K/V/output/normalizers,
not a complete L-by-L attention matrix. It is Python code, not a fused FlashAttention
kernel. Numerical tests cover irregular tiles, Q/K/V gradients, future perturbations,
extreme scores and all-masked dense rows. Conservative memory guards, actual OOMs
and unsupported operations have different explicit statuses.

### B: associative recall

All models have two pre-normalized blocks, hidden size 128, four 32-dimensional
heads, 4x MLPs, learned 256-ID token embeddings and sinusoidal positions. The output
classifier uses the 64 valid value IDs (128–191); labels are supervised **at query
positions**, with every other position masked. Chance baselines are 1/64 and 1/K.
All mechanisms have exactly 437,824 trainable parameters; only attention changes.

| Mechanism | Training conditions per seed |
|---|---|
| Dense | K=4/8/16 mixed; K=16 near |
| Window W64 | K=4/8/16 near; K=16 mixed |
| ELU+1 linear | K=4/8/16 mixed |
| Two local + two dilated heads | K=4/16 mixed |
| Two local + two content-selected heads | K=16 mixed intervention |

There are **14 conditions x 3 seeds = 42 independent training runs**, each with
4,500 successful AdamW updates and batch 32, peak LR 8e-4, weight decay .01, 200-update
warmup, linear decay to .15 of peak and global clipping 1.0. FP32 parameters and
optimizer states use FP16 autocast and dynamic scaling. Overflowed attempts are
recorded and retried on the same batch; they do not consume the optimizer-update
budget. The final checkpoint is used; final-test accuracy never selects a checkpoint.

Each seed has its own deterministic staff training-index range, shared between
matched mechanisms. All source/query positions, labels and true distances remain
auditable. Compact uint8 query maps explicitly save all four fields for every
completed training example; matched mechanisms reuse the same checksummed stream
files instead of storing duplicate datasets. Pending annotations are part of the
recovery checkpoint. The 2,304 development and 9,216 final-test sequences stay unchanged.
All 42 runs finish before any final-test evaluation. Every learned model is tested
on all nine K/regime cells (1,024 sequences each), and every dense model also receives
a W64 test-only mask, giving **54 final-evaluation variants**. Save per-query
predictions/NLLs in compressed shards and distinguish training load from test load.

The intervention is selected using only the final K16 dense/dilated development
far accuracies, averaged over paired seeds. Use content top-k=64 if their gap is
at least .02, otherwise 32; use 64 if a comparator failed. This predeclared decision
and its prediction are locked before intervention training/final testing. Two W64
heads remain; two heads select causal positions by content. It adds no parameters
and uses a full-history KV candidate pool, not a persistent compressed state.
It therefore makes no equal-state claim against linear attention. Dense score
computation, stable sorting and selection overhead are explicitly charged.

W64 has a two-layer maximum access distance of 126; supplied far queries lie
outside it. Mixed local/dilated heads have different multi-layer coverage and
must not be assigned that same limit. Sparse correctness paths use dense boolean
masks and SDPA. Linear attention uses FP32 recurrent sums and chunked prefix
tensors; autograd still retains training intermediates. A separate B=2, H=4, d=32
forward-only sweep at T=256/1024/4096/8192 isolates actual operator costs.

### C: pretrained KV memory

Use the frozen 180 staff prompts: 2 context targets x 3 evidence positions x
3 distractor counts x 10 cases. Jointly tokenize one unchanged chat prompt, then
split pre-question context from the identical question suffix. Retain actual
encoded lengths and map gold evidence offsets afterward, for evaluation only.

| Cache policy | Representation |
|---|---|
| FP16 reference | All positions, unchanged FP16 |
| Sink/recent | Largest legal FP16 position count; safeguards plus recency |
| Question-aware | Same FP16 count; safeguards plus question attention |
| INT4 token | Token-wise K and V |
| INT4 channel | Channel-wise K, token-wise V; reference-byte budget |
| Hybrid | Question-selected FP16 positions and channel-K/token-V INT2 remainder |

All five compressed policies use the case-specific **actual serialized uniform
INT4 channel-K/token-V prefix budget**. Groups contain 64 values, packed uint8
integer payload, FP16 scale and FP16 minimum offset (four metadata bytes), and
ties-to-even rounding; no integer zero-point is separately stored. A documented
32-byte binary header and one shared int32 position-index array, when required,
are included. Keys/values keep the same trailing incomplete token group in FP16.
New question/generated entries are quantized when complete groups form; existing
quantized groups are not repeatedly requantized.
Packed-cache suffixes append one token at a time so completing a 64-token group
cannot use future question/teacher-forced tokens in an earlier prediction. Answer
NLL is scored sequentially for every policy. This correctness implementation adds
packed suffix overhead and is explicitly separate from optimized kernel claims.

Both eviction rules and the hybrid preserve eight sinks and 64 recent positions.
Question scores use actual rotary Q projections against existing cached K in FP32,
causally normalized, then average over observation queries, heads and layers.
Selection is shared across heads/layers and receives no gold code/span. Generation
continues on the original SDPA model; no eager backend switch is used.

The hybrid maximizes its legal selected FP16 count, including all index/metadata
bytes and the residual FP16 low stream. Both selective-FP16 evidence retention and
any-FP16 retention including residuals are recorded. Absolute rotary position IDs
remain unchanged after eviction; physical lengths control appending and masks.

Greedy quality evaluation has a fixed 32-new-token cap and EOS stopping. The scorer
extracts the first standalone five-digit number. Keep all 180 cases, including
incorrect FP16 references. Report unconditional accuracy and accuracy conditional
on FP16 correctness. Save reference-answer NLL, gold-code NLL, post-prefill KL,
teacher-forced answer KL, K/V reconstruction errors, spans, selections, actual
bytes, prefill/scoring/packing/suffix/decode times, and measured memory.

The primary nine accuracy comparisons have 2,000-resample paired bootstrap
intervals over all 180 prompts. Ten-case factorial cells are descriptive. Fixed
qualitative indices [0,30,60,90,120,150] preserve every method’s response and mask.

Real packed-cache encoding/decoding is independently measured for all six policies
at B=1/8 and L=2048/4096, with ten warmed-up synchronized repetitions. Synthetic
cache values use the real model geometry and are identical across policies.
One full Qwen decode step includes append, packing and dequantization; independent
cache clones are outside timers to preserve fixed shapes. Actual binary lengths
and hashes verify byte accounting. Persistent arrays are packed, while ordinary
SDPA receives transient dequantized FP16. These are not low-bit attention kernels.

Prefill occurs before eviction, so eviction does not reduce that full-prefill
peak. Memory records explicitly identify retained reference/work buffers and give
incremental benchmark peaks. The initial budgeted prefix, shared question FP16
equivalent and actual live cache after appending are separate; residual-group
completion can change the packed byte increment.

## Recovery and local replay

Training recovery is verified on Drive every **100 optimizer updates**, at all
required evaluation checkpoints, and before a planned session stop. Save model,
optimizer, manual schedule position, scaler, next training index, full committed
history, development evaluations and Python/NumPy/Torch/CUDA RNG states. Retain
historical checkpoints at 500/1000/1500/2000/3000/4500. Every resumed condition must
match scientific code, configuration, assets, GPU type and software environment.

Final MQAR predictions save every 128 sequences; retrieval saves each method/case.
Writes and Drive copies are atomic and checksum-verified; backup failures stop
the run. Reconnect on the same runtime type and Run All to restore compatible
progress. The session reserve checkpoints before the configured stop. Failed
conditions are retained and never replaced using held-out scores.
If final MQAR scoring exhausts memory or becomes non-finite, its failure remains
in the full nine-cell condition grid with empty accuracies and zero completed
examples. Partial prediction shards remain saved without becoming final metrics.

Download `MyDrive/ATML_PA4/PA_4/Task_2/results/` into local `Task_2/results/` and
preserve `results/task2_seed6304_v1/`. Avoid `results/results`. Keep large recovery
checkpoints on Drive if only replay is needed. In ATML:

```text
conda activate ATML
python Task_2/scripts/audit_results.py
python Task_2/scripts/replay_notebook.py
```

The second command saves all successful notebook outputs using the same ATML
interpreter. Alternatively, open `main_task_2.ipynb`, select ATML and Run All.
Replay imports no Torch,
Transformers or model download stack; it verifies hashes, data identities,
training/evaluation counts, query predictions, NLLs, byte budgets, paired intervals
and figure font audits. Missing evidence raises an error. Delete a downloaded ZIP
only after audit and notebook replay pass. Saved provenance remains Google Colab.

## Local implementation checks

```text
conda activate ATML
python -m unittest discover -s Task_2/tests -v
python Task_2/scripts/validate_project.py
```

These tests use tiny tensors/random models and clearly marked temporary synthetic
evidence. They do not run the 42 real training runs or download the pretrained model.
The full pinned Colab stack and actual hardware must pass notebook checks before
expensive experiments. No synthetic test fixture belongs in real results.

## Attribution

Implementation uses LLM coding assistance. The student is responsible for every
line. The supplied generator and three frozen asset files are copied unchanged and
verified against their original SHA256 list. Atomic artifact/Drive/dependency
utilities adapt this repository’s Task 1 utilities so Task 2 remains independently
uploadable. No external optimized attention or cache-compression engine is copied.

Algorithms and conventions follow the assignment’s required readings: FlashAttention
(Dao et al., 2022), Longformer (Beltagy et al., 2020), Transformers are RNNs
(Katharopoulos et al., 2020), Zoology (Arora et al., 2024), SnapKV (Li et al., 2024)
and KIVI (Liu et al., 2024). Question scoring and quantization layouts are simplified
fixed policies; no full SnapKV/KIVI optimized implementation is claimed. The pinned
Transformers Qwen2/cache interfaces and PyTorch SDPA supply model execution.

Only numerical evidence and technical implementation documentation are generated.
The PDF report’s language, interpretation and analysis must be written by the student.

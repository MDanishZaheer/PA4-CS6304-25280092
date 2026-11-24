# Task 3: Policy Distillation and Speculative Decoding

Name: Muhammad Danish Zaheer Awan  
REG id: 25280092

Use **`main_task_3.ipynb`**. Its seven steps implement the assignment experiments;
the four RQ sections display saved evidence. Full computation runs on Colab.
Local **ATML** Run All validates and displays downloaded Colab results without
loading models, installing packages, downloading data or starting training.

## Upload and execute

1. Upload the complete **Task_3 folder** to `MyDrive/ATML_PA4/PA_4/`.
2. Open `Task_3/main_task_3.ipynb`, select a CUDA GPU and Run All. Use the same GPU
   type throughout a named run. A 22 GB L4 or 40 GB A100 is the intended runtime;
   the fixed microbatch and token quotas remain the assignment values.
3. Keep `DRIVE_PROJECT = "ATML_PA4/PA_4"`. Mount Drive when requested. If already
   imported packages change, restart the runtime and Run All.
4. Code executes under `/content/ATML_PA4/PA_4/Task_3/`. Pinned source rows,
   tokenizers and pretrained weights are cached in runtime `Data/Task_3/`.
5. Verified backups are saved throughout execution to
   `MyDrive/ATML_PA4/PA_4/Task_3/results/task3_seed6304_v1/`.

The folder contains the original four small course input files and checksum list.
No raw Data upload, ZIP upload or GitHub data fetch is needed. The runtime downloads
only the selected GSM8K dataset source, not Task 1's C4 corpora. No other task is
imported at runtime. Preserve existing Drive results when replacing code.

All nine main PNGs are written directly to `results/<run_id>/figures/`, at 300 dpi,
with **Times New Roman 10 pt on every text object**, including legends and ticks.
The supplied personal Windows fonts in `resources/fonts/` are ignored by Git and
intended for private Drive upload. Public cloners must supply their own fonts or
run `scripts/prepare_upload.py` in Windows ATML. Font checks occur before expensive
computation. Replay can display saved PNGs without installing fonts.

## Frozen data, revisions and formatting

Teacher/target: **Qwen/Qwen2.5-1.5B-Instruct**, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
Original draft/student: **Qwen/Qwen2.5-0.5B-Instruct**, revision
`7ae557604adf67be50417f59c2c2f167def9a775`.
GSM8K: **openai/gsm8k**, revision `740312add88f781978c0658806c59bc2815b9866`.
These immutable revisions were already resolved in the course-data workflow and
are fixed before model downloads. Task 3 stores its own independent revision lock.

Use all supplied 500 math/300 short-form training prompts and the 60 math/60
short-form held-out prompts. The short-form test set is 30 familiar and 30 shifted
templates. Staff family numbers 0-2 are local to each template namespace: shifted
family 0 and familiar family 0 have different templates. Preserve that distinction.
Gold answers never enter generation prompts or prefix-bank selection.

Every condition uses the same user-only Qwen chat template with an assistant
generation prompt. The templates' default system formatting is preserved.
Check the entire tokenizer vocabulary mapping, rendered prompts and token IDs on
all 920 prompts. Both models use their identical complete EOS set. No thinking
format, top-k filter or nucleus truncation is introduced.

The course requirements pin Transformers 4.57.1 and Hub 0.36.2. No other course
environment lock was provided. `requirements-compute.txt` records explicit project
pins; actual Python, CUDA, GPU, package versions and `pip freeze` are saved. LoRA is
a small transparent project implementation, so no PEFT dependency is required.

## Exact speculative sampler

`sampling/exact.py` samples full-vocabulary temperature-1 proposals. Each is
accepted with `min(1, p[x]/q[x])`; rejection uses the normalized positive residual
`max(p-q, 0)`. Full acceptance supplies a target bonus when the cap/EOS rule permits.
Do not replace this with greedy token equality or residual sampling from `p`.

Both caches use a documented **one-pending-token invariant**. After a round they
contain the returned context except its last token. The next target verification
consumes that pending token plus the new proposals in one pass. Draft consumption
is charged to the next proposal phase. Cropping removes every rejected/unexamined
proposal from all layers. The final pending token need not be processed after
generation stops. All caches, calls and timers are logged with this convention.

Draft proposals deliberately advance the draft cache after every proposed token;
the following round also consumes its pending returned token. These actual extra
draft calls are reported. This is a readable correctness implementation, not a
custom fused speculative kernel or a claim of optimal draft scheduling.

The required toy test uses 100,000 length-three sequences at each gamma, over all
27 joint outcomes. All expected counts exceed 10. The correct vectorized Markov
implementation must pass Pearson's 1% gate; a deliberately incorrect residual
control must reject. Tiny real-Qwen tests separately check Torch sampler paths,
variable proposal lengths, final-token caps, EOS, padding, full acceptance/bonus
and cache distributions after rollback. On Colab, pinned real-model split/crop
checks run on four fixed training prompts before benchmarks.

## Distillation protocol and recovery

Train **off-policy and on-policy independently at seeds 6304/6305**, four runs.
Each has exactly 30,000 math + 10,000 short-form teacher-scored response prefixes.
Microbatch is one. Accumulate full-vocabulary **forward KL**, normalized by the
actual supervised tokens in the optimizer update. There are 156 full 256-token
updates plus one final 64-token update. Sampled tokens define contexts, never hard
cross-entropy labels. Include EOS when it is a scored response token.

Use frozen FP16 backbones, FP32 LoRA tensors and optimizer states, FP16 autocast
with dynamic scaling, AdamW LR 3e-4, weight decay .01, and global clipping 1.0
after unscaling. LoRA rank 8, alpha 16, dropout 0 modifies only q/k/v/o projections.
Matched methods share identical seeded adapter initialization. The final fixed-quota
checkpoint is used; there is no held-out checkpoint or hyperparameter selection.

Each quota category has a seeded fixed prompt permutation; cycle the predeclared
math/math/math/short category order and reuse a category's permutation after
exhaustion. Skip a category only when its token quota is complete. Response caps
are 192 math / 32 short; temperature 1, top-p 1, top-k 0.

Off-policy first generates a complete frozen **800-prompt teacher pool per seed**.
Every pool's preparation cost is charged to its corresponding run. On-policy
generates each new presentation from the current student. A response may cross an
update boundary; its producing adapter digest/update remains fixed and explicit.
Save source ID, seed, full sampled response, termination, producer, scored index
range and quota truncation for every presentation chunk. No evaluation bank enters
training.

Teacher scoring includes full-vocabulary log-softmax and transfer of the scored
soft targets. If dynamic scaling overflows, reuse these cached soft targets for
the update retry. The quota is not reduced and the teacher is not rescored merely
to repeat a student backward attempt. Overflow attempts are retained.

Recovery checkpoints are atomically written and verified on Drive **every five
successful updates**, at completion and before a planned session stop. Save
adapters, AdamW/scaler, category quota counts, schedule cursors/permutations,
pending sampled response/cursor, committed histories, annotation paths and all
Python/NumPy/Torch/CUDA RNG states. A planned interruption resumes at the exact
saved update. Scientific source, data, GPU and package identities must match.
Failed numerical/OOM attempts are recorded and stop the incomplete stage; they
cannot become fabricated complete results. Already compatible work is reused.

## Banks, alignment and runtime comparisons

The supplied data-only checkout has **no materialized prefix banks**. This
implementation generates and freezes the three required common banks using its
own working decoder: teacher, original student, and exact speculative original at
gamma 4. It also freezes one own-policy bank per trained student, four additional
banks. All five drafts are evaluated on **all seven banks**, including each other's
frozen rollouts and both training seeds.

Use the staff's seeded stratified position selection: up to 8 positions in 1-16,
8 in 17-32, 16 in 33-64, plus a separate long bank of up to 16 positions in 65-192.
Math bank generation cap is 192; extraction cap is 32. If EOS occurs early, retain
only actual positions. The same saved prefix IDs are used for every draft. Save
forward/reverse KL, TV, `1-TV`, greedy agreement and both next-token IDs.

Teacher and exact-speculative prefixes have the same target law but independent
realizations. Their consistency comparison resamples matched **prompt means**
within workload/position strata; it never labels their different token histories
as identical prefixes. Within each frozen bank, draft comparisons genuinely share
the same prefix IDs. Alignment diagnostics use the FP32-adapter wrapper; the merge
error audit documents its numerical relation to the timed merged model.

Runtime matrices:

| Experiment | Conditions |
|---|---|
| Initial pretrained baseline | 120 prompts x 2 gamma values x 2 repetitions = 480 adjacent-control pairs |
| Final primary comparison | 5 drafts x 120 prompts x 2 gamma values x 2 repetitions = 2,400 pairs |
| Fixed-length diagnostic | 5 drafts x first 24 math prompts x 2 gamma values x 2 repetitions = 480 pairs |
| Standalone quality | Target + 5 drafts x 120 prompts x 2 repetitions = 1,440 responses |
| Isolated cost | 5 merged drafts x contexts 64/256/512, plus one unmerged-adapter control at all three contexts |

Primary speed uses EOS-aware cap 64. Fixed-length timing alone ignores EOS.
Standalone answer evaluation separately uses cap 512 math / 32 extraction with
temperature-1 sampling. Save mathematical extraction failures, independent
extraction-format compliance, EOS and length-cap flags. Speculative outputs follow
the target law and are not used to rank standalone student accuracy.

Merge each adapter into a fresh original FP16 backbone; preserve one merged
checkpoint per seed/method and verify logits on the same four fixed training
prompts. Primary comparisons use ordinary merged projections. One fixed unmerged
off-policy seed-6304 control quantifies wrapper cost separately.

The final primary matrix reruns the original draft, so its numbers are not reused
from the earlier baseline stage. Reverse full model order in repetition two,
alternate gamma order, rotate prompt order and counterbalance each adjacent
target/speculative pair. Record warm setup separately. Timers exclude model/data
loading, tokenization and result-writing. Inputs are pretokenized identically.
Every round separates draft passes, single target verification, acceptance/residual
sampling, and cache crop/advancement, with CUDA synchronization. Prefill and other
host overhead are separate. Preserve raw repetitions, tokens, actual model calls,
both acceptance denominators and allocated/reserved peaks.

The cost model uses measured `c` and `v_gamma`, with actual gamma+1 verification
inputs when a pending token exists. Match each prompt to the nearest isolated
context length; this is an approximation, not a claim of exact EOS-aware prediction.
Report projected/empirical-overlap speedups beside measured steady and end-to-end
TPS ratios and request latency ratios. Different output lengths make those distinct
quantities. No positive speedup or finite break-even is assumed.

Prompt-paired 95% bootstrap intervals use at least 2,000 resamples, averaging the
two repetitions within prompt before resampling. Training seeds remain separate.
The upfront cost ledger retains observed pool/training/merge wall time, including
partial planned attempts, setup and in-stage Drive I/O. Component timings are also
saved. For an interrupted stage whose earlier wall timer was not saved, account
for at least the full-history measured components. `accounted_upfront_seconds`
sums the larger of each stage's recorded wall cost and measured component cost;
`upfront_is_lower_bound` flags incomplete coverage. The recorded wall value is
retained separately. Deployment amortization uses measured adjacent-target
seconds saved per request; a zero/negative saving has **no finite break-even**.

The downloaded L4 run is complete: four 40,000-token runs, seven banks, 113,930
draft-prefix rows, 3,360 paired speed measurements and 1,440 standalone responses.
Its on-policy seed 6305 stage timer covers the final resumed session only. The
corrected upfront cost is therefore at least 2,088.81 seconds; exact lost wall time
cannot be recovered. On-policy generation resets CUDA peak counters, so the saved
training-memory numbers are observed peaks after the last reset, not certified
whole-run maxima. Per-request benchmark memory measurements retain their intended
scope. These limitations are recorded in `metrics/local_integrity_review.json`.

Local source formatting reproduces the original Colab fingerprint exactly. Seven
frozen source files retain CRLF through `.gitattributes`; other source files use LF.
Git preserves result and staff-data bytes because manifests validate exact hashes.
The original cloud manifest remains in `configs/original_colab_run_manifest.json`.
CPU cost/figure corrections are recorded in `metrics/local_evidence_rebuild.json`;
all underlying training, prefix, timing and quality records remain unchanged.

Fixed prompt indices [0,30,60,90] retain all draft/gamma continuations at repetition
0. Fixed first-three selected prefix positions from those prompts retain every
draft's aligned next-token disagreement, regardless of which method wins.

## Download and local replay

Download the Drive `Task_3/results/` folder into local `Task_3/results/`, preserving
`results/task3_seed6304_v1/`. Avoid `results/results`. Large recovery and merged
checkpoints may remain on Drive if only replay is needed; preserve those durable
cloud checkpoints for resumption. Compact evidence contains the full provenance,
histories, raw presentations, bank IDs, predictions, round traces, tables and PNGs.

```text
conda activate ATML
python Task_3/scripts/audit_results.py
python Task_3/scripts/replay_notebook.py
```

Alternatively select ATML and Run All in the main notebook. Replay verifies
scientific/data identities, file hashes, all four quotas, bank IDs/producer locks,
sampling counters/cache positions, complete condition matrices, answer scores,
prompt-paired intervals and figure font audits. It imports no Torch, Transformers,
Hub, datasets or Accelerate stack. Missing/incompatible evidence is an error.
Delete a downloaded ZIP only after audit and replay succeed.

## Local implementation checks

```text
conda activate ATML
python -m unittest discover -s Task_3/tests -v
python Task_3/scripts/validate_project.py
```

Tests use random tiny local models and explicitly synthetic TemporaryDirectory
fixtures. They never download pretrained models or run the actual four full
distillation experiments. Synthetic fixture outputs do not belong in real results.
The imported run now contains the actual Colab scores, independently audited
checkpoints/evidence and a saved local replay with all 29 code cells and nine PNGs.

## Attribution

Implementation uses LLM coding assistance; the student is responsible for every
submitted line. Artifact, Drive, font, dependency and replay utilities adapt this
repository's earlier-task utilities to keep direct one-folder upload independent.
The four staff prompt/selection files are unchanged and SHA256 verified.

The exact accept/residual/bonus algorithm follows Leviathan et al. (2023), *Fast
Inference from Transformers via Speculative Decoding*. Distillation comparisons
follow the assignment's simplified protocol informed by Zhou et al. (2024),
*DistillSpec*, Agarwal et al. (2024), *On-Policy Distillation of Language Models*,
and Hu et al. (2022), *LoRA*. No optimized speculative decoding engine or full
published DistillSpec/GKD implementation is copied or claimed. PyTorch SDPA and
the pinned Transformers Qwen/cache interfaces supply model execution.

Only technical implementation documentation and numerical evidence are generated.
The PDF report's language, interpretation and analysis must be written by the student.

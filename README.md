# Running Qwen3.8-Flash-Next (AWQ) on AMD RDNA4 / vLLM-ROCm

Four small patches to vLLM (plus a kernel-warmup shim for CUDA/HIP graph capture)
that get `leoncca/Qwen3.8-Flash-Next-AWQ-g32` — an AWQ-quantized version of
Alibaba's Qwen3.8-Flash-Next (a preview of the Qwen4 architecture: 125B MoE + a
51GB sparse n-gram embedding table, ~6B parameters active per token) — loading
and serving on consumer/workstation AMD GPUs.

None of these are architectural limitations. Each is a narrow, fixable gap in
day-2 support for a brand-new model architecture (`qwen4_exp`, merged into
vLLM ~Sept 2 2026) meeting an AWQ checkpoint layout and a ROCm fallback code
path that hadn't been exercised together before.

## Status (2026-09-04)

**Production config:** TP4 + expert parallelism, PIECEWISE CUDA graphs + MTP(2),
GPU P2P on, PLE table on host RAM — `tools/launch.sh` with its defaults. Single-stream
decode **~49 tok/s, flat from 55K to 120K tokens of context**, 272K-token KV pool at
`--max-model-len 131072`. `--enforce-eager` is **no longer used or recommended** (it was
the workaround for a graph-capture crash that is fixed by the QSA kernel warmup in this
repo; the earlier "stay on eager" recommendation below is kept as history). HumanEval on
the *eager* config was 96.34 %; the PIECEWISE config is numerically non-identical
(FP drift at graph-split boundaries, `GRAPH-CAPTURE-FIX.md`), and its full quality run on
the production EP seat is in progress as of this writing — see Validation.

## Environment

- Hardware: 4× AMD Radeon AI PRO R9700 (RDNA4, gfx1201, 32 GB each), TP4 — four of the six
  cards in an MC62-G40 box; BIOS ACS disabled and the P2P env (`HSA_ENABLE_IPC_MODE_LEGACY=0`),
  see `TUNING.md` §1
- vLLM: `vllm/vllm-openai-rocm:nightly-27a94d1ce4e3fc100c4732439ccec10f8246a804`
  (the Sept 2 2026 nightly — **required**; mainline v0.28.0 predates
  `qwen4_exp` architecture support entirely)
- Checkpoint: [`leoncca/Qwen3.8-Flash-Next-AWQ-g32`](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-AWQ-g32)
  (expert-only AWQ W4A16, group size 32; 129 GB on disk)

## The five problems, and the fixes

### 1. PLE embedding table doesn't fit in VRAM (`ple_cpu.py`, new file)

The checkpoint's PLE (n-gram) embedding table is 52 GB on disk. vLLM's
`VocabParallelEmbedding` shards it cleanly across TP ranks with no
redundancy — but even one rank's share pushes total per-GPU VRAM need to
~34.25 GB against 31.86 GB available, an OOM by a few GB.

The table is accessed as a **sparse hashed n-gram lookup** — roughly 16 rows
of 160 elements per token, not a bulk read — so keeping it on host RAM and
gathering only the touched rows per forward pass is cheap: worst case (an
8192-token prefill chunk) moves ~21 MB, well under a millisecond at typical
PCIe bandwidth. `ple_cpu.py` adds `PLECpuOffloadEmbedding`, a subclass that:

- keeps the table pinned on host RAM in its native FP8 storage (not upcast —
  halves the RAM cost and is required for correctness anyway, since dequant
  needs the scale tensor)
- gathers + dequantizes only the requested rows per step, then transfers
  the small result to GPU
- is opt-in via `VLLM_QWEN4EXP_PLE_CPU_OFFLOAD=1`; behavior for every other
  model/platform is unchanged

Two subtler bugs were found and fixed building this (see comments in the
file for the full story): a generic vLLM hook (`process_weights_after_loading`)
was round-tripping the entire CPU-resident table through VRAM for nothing on
every load, and a dtype attribute (`params_dtype`) was double-purposed by
unrelated code, causing dequantized output to get silently re-quantized on
its way out. Both are the actual reason this was worth documenting carefully
— the *idea* is simple, the failure modes weren't.

### 2. AWQ MoE expert loader crashes on a spurious checkpoint tensor (`routed_experts.py`)

`AttributeError: Layer ... has no parameter 'w2_weight' for checkpoint weight
'...down_proj.weight'`

This checkpoint's expert shapes aren't supported by vLLM's fast AWQ-Marlin
MoE kernel, so it falls back to the generic WNA16 path — whose registered
parameters are `w2_qweight`/`w2_qzeros`/`w2_scales`, not `w2_weight` (that
bare name is only created as a post-load alias, too late for the loading
loop). The checkpoint's weight stream additionally yields a small number of
spurious extra tensors per expert/shard (an artifact of the quantization
tooling — a bare `.weight` and a `.weight_scale_inv`, an FP8-scale leftover
irrelevant to this int4 checkpoint) that collide with that bare-name
resolution.

Fix: when a `_weight`-suffixed name can't resolve to a real parameter *and*
its `_qweight` sibling already exists (meaning the real data already loaded
correctly via the legitimate tensor), skip the spurious one rather than
erroring — or, the wrong first attempt this went through, silently
redirecting it into the real parameter and corrupting already-correct
weights. The patch comments document that failure mode explicitly; it's
the kind of bug that doesn't announce itself.

### 3. `enable_thinking`/output-dtype interplay — see `ple_cpu.py` comments

Folded into fix #1 above (the `params_dtype` double-purposing bug) rather
than broken out separately, but flagged here because it's conceptually
distinct: it isn't a loading bug, it's a **silent numerical-correctness**
bug that only a downstream shape/dtype mismatch happened to surface as a
crash. Worth a second read if you're adapting this patch to a different
checkpoint or architecture — "it loads" is not the same as "it's correct."

### 4. Expert parallelism crashes in the WNA16 MoE weight loader (`moe_wna16.py`)

`IndexError: index 128 is out of bounds for dimension 0 with size 128` (rank 0) and
`IndexError: index 1 is out of bounds for dimension 1 with size 1` (ranks ≥ 1) when
launching with `--enable-expert-parallel`.

Because this checkpoint takes the generic WNA16 path (problem 2), its expert
weights go through `moe_wna16_weight_loader`. That loader has two fast paths for
`w13_qzeros` / `w2_qzeros` which (a) write `param.data[expert_id]` using the
**global** expert id, and (b) slice the checkpoint tensor by
`get_tensor_model_parallel_rank()`. Both assumptions only hold without EP. Under
expert parallelism each rank owns `num_experts / ep_size` whole experts (here
512/4 = 128) and the MoE layer's own parallel config has `tp_size = 1`,
`tp_rank = 0` — experts are not split within a rank — so a global id ≥ 128 or a
TP rank ≥ 1 walks off the end of the local tensor. The delegate loader those
branches fall through to already maps global → local; the fast paths simply
never got the same treatment.

Fix (`moe_wna16.patch`, ~20 lines): map the expert id with
`layer._map_global_expert_id_to_local_expert_id()` and skip experts this rank
does not own, and take the slicing rank from `layer.moe_config.tp_rank`. This is
a plain vLLM bug (any WNA16-quantized MoE + EP), independent of the other three
patches.

Why you want EP on this model: TP8 is impossible for the AWQ-g32 checkpoint (the
640-wide experts shard to 80 columns per rank on the down-projection and
80 % 32 ≠ 0), so expert parallelism is the only way to spread this model across
8 GPUs — and it is also faster on 4 (see Validation).

### 5. Mixed-length concurrent prefills OOM in the PLE short-conv (`ple_layer.py`, second hunk)

Found 2026-09-04 by the first full bench suite: AutomationBench put four ~16K-token
prompts into prefill together and TP rank 1 died with
`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 290.00 MiB … 0 bytes free`
inside `_short_conv_dilated_prefill_batched` (`packed_tokens.transpose(1, 2).contiguous()`).
The engine core then hangs forever on the dead worker while `/health` keeps answering 200.

Cause (upstream vLLM code, not one of the RDNA4 patches): the batched prefill pads every
in-flight request to the longest one and materialises ~5 copies of that padded block
(packed, its transpose, `history`, the conv output and its transpose). vLLM's startup
memory profile sizes its batch as `max_num_seqs` *equal-length* chunks, so it never sees
the padded case. With `gpu_memory_utilization=0.95` the KV cache takes every byte the
profile left (peak activation 2.11 GiB), so a real batch of one long chunk plus a few
short prompts overruns it. Any tight-gmu seat would hit this; it is not RDNA4-specific.

Fix (`ple_layer.patch`, second hunk): when padding would exceed ~25 % of the real prefill
tokens, run the requests one at a time through the same packed math
(`_short_conv_dilated_prefill_per_request` → `_short_conv_dilated_prefill_core`). Each
request then costs at most one full-width chunk, which the profile does cover. Even-length
batches keep the packed, sync-free path; the fallback does one small device-to-host sync
for the lengths and is eager-only (prefill is never graph-captured). Verified bit-exact
against the packed path on CPU (outputs and conv-state write-back), and live on the
0.95 seat: `1×16K+3×short`, `4×16K`, `1×32K+3×2K`, `4×32K` concurrent prefills, no OOM
(`tools/stress_prefill.py`). Candidate for upstreaming.

## Applying the patches

No image rebuild needed — bind-mount the patched files over the installed copies.
`tools/launch.sh` does exactly this (all seven mounts, the P2P env, production knobs;
`EP=0` for the TP-only variant, `MTP=0` for no speculation, `GMU/MML/SEQS/LADDER` to
override). `TUNING.md` walks through how every setting was arrived at.

Seven files go in: the four fixes (`ple_layer.py`, `ple_cpu.py`, `routed_experts.py`,
`moe_wna16.py`) and the graph-capture warmup (`cudagraph_utils.py`, `kernel_warmup.py`,
`qwen4_exp_amd_qsa_warmup.py` — see "Graph capture" below).

```bash
CC=$(cat tools/compilation_config_seqs4.json)   # PIECEWISE + splitting_ops + ladder [1,2,3,4,6,8,12]
V=/usr/local/lib/python3.12/dist-packages/vllm
docker run -d --name flashnext \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --ipc host --shm-size 16g --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e VLLM_QWEN4EXP_PLE_CPU_OFFLOAD=1 \
  -e HSA_ENABLE_IPC_MODE_LEGACY=0 -e NCCL_PROTO=Simple \
  -e HIP_FORCE_DEV_KERNARG=1 -e SAFETENSORS_FAST_GPU=1 \
  -v $PWD/ple_layer.py:$V/models/qwen4_exp/amd/ple_layer.py:ro \
  -v $PWD/ple_cpu.py:$V/models/qwen4_exp/common/ple_cpu.py:ro \
  -v $PWD/routed_experts.py:$V/model_executor/layers/fused_moe/routed_experts.py:ro \
  -v $PWD/moe_wna16.py:$V/model_executor/layers/quantization/moe_wna16.py:ro \
  -v $PWD/cudagraph_utils.py:$V/v1/worker/gpu/cudagraph_utils.py:ro \
  -v $PWD/kernel_warmup.py:$V/model_executor/warmup/kernel_warmup.py:ro \
  -v $PWD/qwen4_exp_amd_qsa_warmup.py:$V/model_executor/warmup/qwen4_exp_amd_qsa_warmup.py:ro \
  -v /path/to/Qwen3.8-Flash-Next-AWQ-g32:/model:ro \
  -p 8011:8000 \
  vllm/vllm-openai-rocm:nightly-27a94d1ce4e3fc100c4732439ccec10f8246a804 \
  --model /model --served-model-name Qwen3.8-Flash-Next-AWQ \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --gpu-memory-utilization 0.95 --max-model-len 131072 \
  --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --kv-cache-dtype auto --quantization awq_marlin --enable-prefix-caching \
  --compilation-config "$CC" \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --trust-remote-code
```

Notes on the flags:
- `HSA_ENABLE_IPC_MODE_LEGACY=0` replaces the `NCCL_P2P_DISABLE=1 RCCL_NET=Socket`
  pair this README previously carried. The "gfx1201 RCCL bug" (`hipIpcGetMemHandle:
  invalid argument` on every vLLM-ROCm image ≥ 0.21) turned out to be that env var,
  baked into those images, forcing the legacy KFD IPC path which gfx12 rejects. With
  it off RCCL initialises `via P2P/IPC` (all-reduce 19.7 vs 11.6 GB/s busbw on 4×R9700;
  BIOS ACS must be disabled for real P2P DMA — `TUNING.md` §1). Keep `NCCL_PROTO=Simple`;
  that one is a separate RCCL LL-protocol deadlock on gfx12. Never re-add
  `NCCL_P2P_DISABLE` / `RCCL_NET=Socket`.
- `--enable-expert-parallel` (with `moe_wna16.py` mounted) is the production choice:
  +8–10 % decode on 4 GPUs and the only route to 8 (TP8 is impossible for this checkpoint,
  see fix #4). Drop it for the TP-only variant (~45 tok/s).
- `--compilation-config`: pure `PIECEWISE` cudagraph mode, the default attention
  `splitting_ops` **plus** `vllm::qwen4_exp_amd_ple_ngram_embedding` (the host-RAM PLE
  gather cannot be recorded into a graph), and an explicit `cudagraph_capture_sizes`
  ladder. `[1,2,3,4,6,8,12]` is right for `--max-num-seqs 4`; the default ladder scales
  with `max_num_seqs × (MTP+1)` and at seqs 32 balloons to 192 sizes, leaving 0 GiB for KV
  (`tools/compilation_config_seqs32.json` has the seqs-32 ladder).
- Production knobs that survived testing: `--gpu-memory-utilization 0.95` (0.97 loads
  but OOMs the engine as soon as prefills batch), `--max-num-seqs 4` (2 trips a
  torch.compile `ConstraintViolation` on `query_start_loc`). With those,
  `--max-model-len 131072` fits at 2.08× concurrency (272K-token KV pool).
- `--kv-cache-dtype auto` (i.e. BF16) is **mandatory**, not a choice — the
  architecture's attention mechanism (QSA) raises `NotImplementedError` on
  FP8 KV. This is unrelated to any patch here.
- `--speculative-config` enables the checkpoint's native MTP head (acceptance 68–73 %;
  bit-identical HumanEval output vs. no-MTP on the eager config — see Validation). Drop
  it for the no-MTP baseline, where `--max-model-len` can go back up to 262144 without
  MTP's extra KV reservation eating the margin.
- No `--cpu-offload-gb`. The PLE offload patch (fix #1) is what makes the model fit;
  generic weight offload is not used and was never the fix for anything here.
- `ple_layer.py` is shipped here as the full patched file (not a diff) since
  the upstream file may move between vLLM revisions — `ple_layer.patch` is
  included for reference against the exact commit above. Same for
  `routed_experts.patch` and `moe_wna16.patch`.

### Graph capture (why `--enforce-eager` used to be required, and isn't now)

Without `--enforce-eager` the original launches died during HIP graph capture with
`hipErrorStreamCaptureUnsupported`. Root cause (`GRAPH-CAPTURE-FIX.md`): QSA's Triton
attention kernels autotune lazily on first use, and Triton's candidate compilation —
which routinely and harmlessly discards configs invalid for gfx1201's WMMA v2 ISA —
cannot happen on a stream that is mid-capture. The fix is to resolve that autotuning in
eager mode *before* capture starts: `qwen4_exp_amd_qsa_warmup.py`, wired in through
`cudagraph_utils.py` + `kernel_warmup.py`. Second wall, once that was fixed: the PLE
host-RAM gather is not capturable on any hardware, hence the extra `splitting_ops` entry
and pure `PIECEWISE` mode above (the v1 default `FULL_AND_PIECEWISE` captures decode
batches as one monolithic graph and ignores `splitting_ops`).

Cost of PIECEWISE: output is not bit-identical to eager. That was first read as a
"quality regression" (a token-level diff of saved HumanEval generations, commit
`e7c9918`, showed 47/164 greedy generations diverging mid-output, all coherent), then
as "numerical drift", and finally measured properly on 2026-09-04 — see "Where the
drift comes from" below. Short version: eager is not a gold standard either; any two
implementations of this model in bf16 differ by the same amount, and PIECEWISE is just
one of them. Adopted for production on that basis; the speedup is ~2.7× at no-MTP and
the whole 45–49 tok/s figure depends on it.

## Validation

Real HumanEval pass@1 (lm-eval, chat-completions, generations re-scored
against the official HumanEval `check()` — lm-eval's own bundled scorer
misreads markdown-fenced chat answers as failures), needle-in-haystack, and
live `/metrics` — not vendor claims.

### Speed progression (single stream, this box)

| Config | Decode | Date |
|---|---|---|
| eager, no MTP, mml 65536, gmu 0.93, SHM env | 6.7–6.8 tok/s | 2026-09-03 |
| eager + MTP(2) | 8.3–11.0 tok/s | 2026-09-03 |
| PIECEWISE (QSA warmup), no MTP | 18.3–18.5 tok/s | 2026-09-03 |
| PIECEWISE + MTP(2), P2P env, gmu 0.95 / mml 131072 / seqs 4 — TP4 | 45.1 ± 2.0 @ 55K · 44.8 ± 3.2 @ 120K | 2026-09-04 |
| **same + `--enable-expert-parallel` (production)** | **49.4 ± 3.2 @ 55K · 48.3 ± 3.3 @ 120K** | 2026-09-04 |

The 2026-09-04 rows are n=8 per cell, 512-token generations, prefix-cached, box
otherwise idle (`tools/depthval.py`; methodology and the ±10–20 % run-to-run variance in
`TUNING.md` §2). Decode is flat with depth; cold 120K TTFT ≈ 42 s. KV pool 272K tokens
(2.08× at 131K) on both TP and EP.

### Quality

Eager config (2026-09-03, mml 65536, gmu 0.93):

| | No MTP | MTP enabled |
|---|---|---|
| HumanEval pass@1 | 96.34% (158/164) | 96.34% (158/164) — **identical failures** |
| Spec-decode acceptance | — | 61.8% (446/722 draft tokens) |
| Needle-in-haystack | clean @ 123.5K tokens | — |
| VRAM per rank | ~28 GiB | ~28 GiB |

The identical HumanEval score and identical six failed problems (`find_zero`,
`decode_cyclic`, `decode_shift`, `rounded_avg`, `fix_spaces`,
`order_by_points`) with MTP on vs. off is the expected signature of
*correct* speculative decoding — MTP verifies drafts against the base
model's own output distribution, so bit-identical greedy output (not just
"similar") is real evidence it's implemented correctly, not silently
degrading quality for speed.

PIECEWISE config (2026-09-03, TP4, no MTP, SHM env): 95.12 % @c1 · 95.73 % @c2 ·
93.90 % @c8 — numerical drift, see "Graph capture" above and `GRAPH-CAPTURE-FIX.md`.

**Production config (EP + PIECEWISE + MTP k=2, gmu 0.95, mml 131072, seqs 4), full suite
2026-09-04** — `johnny bench flashnext-awq-tp4-ep-p2p-piecewise-mtp-mml131072`, thinking
off, concurrency 4 unless noted, run on the live boot-profile seat:

| Suite | Flash-Next EP (prod) | Reference on the same box |
|---|---|---|
| HumanEval pass@1 | **95.73 %** (157/164) | 96.34 % eager (this model); 27B-FP8 93.29–96.34 %; gemma-4-26B 95.12–96.34 % |
| ARC-Challenge (first 400, CoT, 2048-tok budget) | **97.0 %** (388/400, 1 no-extraction) | gemma 93.75 %, Nemotron-3-Super 91.5 %, Qwen-122B 87.0 %, 27B 79–80 % — all at the old 512-tok budget |
| PlanBench task_1 exact plan (100) | **89.0 %** (prefix 89.2 %, 0 errors) | 27B 57 %, gemma 34 % |
| AutomationBench (30 tasks, 6 domains) | **40.0 %** pass, 66.8 % avg partial | 27B effort-low TP4 40.0 % / 61.2 %; gemma 6.7 % / 13.6 % |
| Code needle (16 targets, 30K corpus) | 15/16 | 15/16 for 27B and gemma |
| ICL pattern probe (16) | 1/16 | 0/16 for every other model here |
| Depth (llama-benchy, pp 2048 / tg 32) | 48.6 / 47.0 / 41.3 tok/s decode at 0 / 4K / 8K depth; ~1.5–1.6K tok/s prefill | — |
| Perf (bench.sh, 100-tok bursts) | 40.4 tok/s single · **120.0 tok/s** aggregate @c4 | fleet-comparable numbers; the 512-token single-stream figure is ~49 |
| ctxsafe (real needle requests at rising depth, disposable seat, live VRAM poll) | **verified safe to 131,072** (the full max_model_len; 2 trials at 124.5K and 131K), no crash, VRAM peak 31.7 GB | 27B-FP8 262K, Nemotron 123K (limited) |

The seven HumanEval failures are the eager run's six plus `minPath` — the PIECEWISE
numerical-drift signature, not a new failure mode. ARC's one miss rambled to the 2048
cap. The first ARC pass at the harness's old 512-token budget scored 87.75 % with 41
truncated answers; the budget was raised for both modes afterwards (johnny `0e87c3d`),
so the reference models' ARC numbers above will rise when re-run.

The suite also found and fixed problem 5 (mixed-length concurrent prefills OOM the PLE
short-conv — AutomationBench crashed the seat on its first attempt, then passed cleanly
on the patched file).

For reference, on the same hardware and methodology: a dense Qwen3.8-27B-FP8
scores 93.29% (153/164); Gemma-4-26B scores 96.34% (158/164, different 6
failures than Flash-Next).

## Where the drift comes from (numerics study, 2026-09-04)

Question: PIECEWISE greedy output differs from eager greedy output. Which op is
responsible, and is it a defect? Method (`tools/drift/`): the same weights on the same
four GPUs, launched under a series of configurations, each probed with the same twenty
prompts (the seven HumanEval problems this model fails, plus thirteen prose / code / math
/ structured prompts), greedy, MTP off, top-5 logprobs recorded per token. Two runs per
seat give the noise floor. The comparison that matters is at **position 0** — the very
first generated token, whose distribution is produced by a single prefill pass: no graph
replay is involved there, so it isolates kernel numerics from CUDA-graph capture.

| Configuration | vs. eager+kernels: top-1 agrees at pos 0 | top-5 max |Δ logprob| at pos 0 (median) | chosen-token mean |Δ| |
|---|---|---|---|
| eager+kernels, run 2 (noise floor) | 20/20 | **0.000** | 0.000 |
| PIECEWISE, run 2 vs run 1 (noise floor) | 20/20 | **0.000** | 0.000 |
| **PIECEWISE (production)** | 18/20 | 0.48 | 0.018 |
| PIECEWISE + rotary forced to native kernel | 19/20 | 0.45 | 0.019 |
| PIECEWISE + activation forced to native kernel | 19/20 | 0.51 | 0.019 |
| **eager, all ops torch-native (no compiler at all)** | 19/20 | 0.49 | 0.024 |
| eager, only the norms torch-native | 18/20 | 0.46 | 0.022 |
| eager, only rotary torch-native | 18/20 | 0.35 | 0.026 |
| eager, only the activation torch-native | 20/20 | 0.000 | 0.000 |

(PIECEWISE with the *norm* kernels forced native does not launch on this build — Dynamo
refuses to trace the ROCm RMSNorm wrapper, `device_index.__init__` "marked as skipped" —
which is why the bisection moved to the eager side.)

Findings:

1. **Both modes are deterministic.** Run-to-run, eager and PIECEWISE each reproduce every
   logprob to the last digit. The gap between them is a fixed implementation difference,
   not autotuning or scheduling nondeterminism.
2. **The gap is not the compiler's.** Eager with torch-native op implementations — no
   Inductor, no graphs, just different (unfused) PyTorch code for norm/rotary/activation —
   is as far from eager-with-kernels as PIECEWISE is (0.49 vs 0.48 nats), and equally far
   from PIECEWISE. Swapping *only* the norm implementation, or *only* rotary, produces the
   full-size effect on its own. Every implementation change lands in the same band.
3. **The amplifier is the bf16 logit grid.** Every logit gap in every run is an exact
   multiple of 1/16: the final logits are bf16 at magnitudes 8–32, i.e. a resolution of
   0.06–0.125 nats. Deltas of 0.3–1.7 nats on tail tokens are several ulps — accumulated
   bf16 rounding through ~100 layers of a different but equally valid summation order,
   surfacing on the coarse output grid. It does not grow with prompt length (a 13-token
   prompt drifts as much as a 407-token one), which is what a per-token systematic
   rounding difference looks like, not an accumulating error.
4. **What it does to output.** The token actually chosen moves by ~0.02 nats (≈2 % in
   probability). Greedy top-1 flips only on near-ties (the "poem" prompt: a 0.44-nat gap
   in eager became an exact tie under PIECEWISE), and the large moves are all on tail
   tokens greedy never picks. At any temperature above 0 this is far below sampling noise.
   The one-problem HumanEval difference between eager and PIECEWISE is this noise class.

Conclusion: nothing to fix. "Bit-exact with eager" is not a meaningful acceptance
criterion for a bf16 model whose eager path is itself one arbitrary point in the cloud;
the meaningful criterion is the benchmark table above, which the production config
passes. This should hold for any bf16 deployment of this model on any vendor's hardware.

## Known limitations and open items

- **PIECEWISE is not bit-exact with eager — and neither is anything else** (see the numerics
  study). If a workflow needs reproducibility, pin one configuration: each is deterministic
  run-to-run. Don't pay the 4–7× eager decode cost for "exactness" that eager doesn't have.
- **fp8 KV is impossible** — QSA hard-requires BF16 KV (`NotImplementedError`). That
  also rules out the fp8-KV-dependent `VLLM_ROCM_USE_AITER=1` path validated elsewhere
  on this hardware (AITER + BF16 KV crashes with an LDS overflow on other models here;
  untried on this one, likely incompatible as-is).
- **TP8 (and TP6) are impossible for this AWQ-g32 checkpoint** — 640-wide experts shard
  to 80 columns per rank on the down-projection and 80 % 32 ≠ 0. Expert parallelism is
  the 8-GPU path (`TUNING.md` §5). The compressed-tensors W4A16 repack of this model
  (`VnimanieAI/Qwen3.8-Flash-Next-W4A16`, group size 128) is worse: `160 % 128 ≠ 0`
  already fails at TP4, so it does not fit 4× R9700 at all.
- **No RDNA4-specific kernel tuning applied** — precedent on this hardware
  shows ~0% gain from this class of tuning on MoE architectures specifically
  (vs. +2–4% on dense models), so it's a low priority.
- **The KV pool varies ~20 % between launches of the identical config** — 271,558 tokens
  (hand launch), 250,557 and 220,867 (johnny launches, same image/knobs, 2026-09-04).
  Weights and profiled peak activation are identical each time; what moves is the
  "consumed (weights + non-torch)" figure vLLM measures after load (24.49 → 24.51 → 24.94
  GiB per GPU), i.e. HIP/RCCL/kernel-cache residency. Harmless for a personal seat (still
  ≥ 1.69× of 131K) but worth knowing before promising a concurrency number. Unexplained.
- **`moe_wna16.patch` is not upstreamed yet.** It is a plain vLLM bug for any WNA16 MoE
  under EP.
- All patched files were built and validated against the exact vLLM commit
  above; `qwen4_exp` support is days old at the time of writing — expect to need to
  re-verify against newer nightlies.

### Historical: the `--enforce-eager` era (2026-09-03, superseded)

Kept because the reasoning is still correct, only the conclusion moved. The first
launches needed `--enforce-eager` because HIP graph capture crashed
(`hipErrorStreamCaptureUnsupported`) — initially misattributed to an Inductor
autotuning bug, then root-caused to QSA's lazy Triton autotuning colliding with stream
capture. `cudagraph_mode: NONE` loaded but was *slower* than eager (5.7 vs 6.7 tok/s);
the QSA warmup + PIECEWISE combination was the fix, first measured at 18.3 tok/s and
initially **not** adopted because of the HumanEval movement, which the later token-level
diff (`e7c9918`) reclassified as numerical drift. Full log with configs, the c1/c2
isolation harness (`run_humaneval_c1.sh`, `run_humaneval_c2.sh`) and the divergence
analysis: `GRAPH-CAPTURE-FIX.md`.

## License

These files are modifications of vLLM source (Apache License 2.0) and are
distributed under the same license. See the original files in the vLLM
repository for full copyright headers.

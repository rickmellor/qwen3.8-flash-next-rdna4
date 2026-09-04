# Running Qwen3.8-Flash-Next (AWQ) on AMD RDNA4 / vLLM-ROCm

Four small patches to vLLM that get `leoncca/Qwen3.8-Flash-Next-AWQ-g32` — an
AWQ-quantized version of Alibaba's Qwen3.8-Flash-Next (a preview of the Qwen4
architecture: 125B MoE + a 51GB sparse n-gram embedding table, ~6B parameters
active per token) — loading and serving on consumer/workstation AMD GPUs.

None of these are architectural limitations. Each is a narrow, fixable gap in
day-2 support for a brand-new model architecture (`qwen4_exp`, merged into
vLLM ~Sept 2 2026) meeting an AWQ checkpoint layout and a ROCm fallback code
path that hadn't been exercised together before.

## Environment

- Hardware: 4× AMD Radeon AI PRO R9700 (RDNA4, gfx1201, 32 GB each), TP4 — on a 6-card MC62-G40 box;
  BIOS ACS disabled and the P2P env (`HSA_ENABLE_IPC_MODE_LEGACY=0`), see `TUNING.md` §1
- vLLM: `vllm/vllm-openai-rocm:nightly-27a94d1ce4e3fc100c4732439ccec10f8246a804`
  (the Sept 2 2026 nightly — **required**; mainline v0.28.0 predates
  `qwen4_exp` architecture support entirely)
- Checkpoint: [`leoncca/Qwen3.8-Flash-Next-AWQ-g32`](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-AWQ-g32)
  (expert-only AWQ W4A16, group size 32; 129 GB on disk)

## The four problems, and the fixes

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

## Applying the patches

`tools/launch.sh` wraps the command below (P2P env, all four patches, knobs via env, `EP=1`
for expert parallelism). `TUNING.md` walks through how every setting in it was arrived at —
GPU P2P root cause, memory budget, concurrency, expert parallelism — with the probes in `tools/`.


No image rebuild needed — bind-mount the three files over the installed
copies:

```bash
docker run -d --name flashnext \
  --device /dev/kfd --device /dev/dri --group-add 44 --group-add 992 \
  --ipc host --shm-size 16g \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e VLLM_QWEN4EXP_PLE_CPU_OFFLOAD=1 \
  -e HSA_ENABLE_IPC_MODE_LEGACY=0 -e NCCL_PROTO=Simple \
  -e HIP_FORCE_DEV_KERNARG=1 -e SAFETENSORS_FAST_GPU=1 \
  -v $PWD/ple_layer.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd/ple_layer.py \
  -v $PWD/ple_cpu.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/common/ple_cpu.py \
  -v $PWD/routed_experts.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/routed_experts.py \
  -v $PWD/moe_wna16.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/moe_wna16.py \
  -v /path/to/Qwen3.8-Flash-Next-AWQ-g32:/model \
  -p 8009:8000 \
  vllm/vllm-openai-rocm:nightly-27a94d1ce4e3fc100c4732439ccec10f8246a804 \
  --model /model --served-model-name Qwen3.8-Flash-Next-AWQ \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.93 --enforce-eager \
  --max-model-len 65536 --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --kv-cache-dtype auto --quantization awq_marlin \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --tool-call-parser qwen3_xml --reasoning-parser qwen3 --trust-remote-code
```

Notes on the flags:
- `HSA_ENABLE_IPC_MODE_LEGACY=0` replaces the `NCCL_P2P_DISABLE=1 RCCL_NET=Socket`
  pair this README previously carried. The "gfx1201 RCCL bug" (`hipIpcGetMemHandle:
  invalid argument` on every vLLM-ROCm image ≥ 0.21) turned out to be that env var,
  baked into those images, forcing the legacy KFD IPC path which gfx12 rejects. With
  it off RCCL initialises `via P2P/IPC` (all-reduce 19.7 vs 11.6 GB/s busbw on 4×R9700;
  BIOS ACS should be disabled for real P2P DMA). Keep `NCCL_PROTO=Simple` — that one is
  a separate RCCL LL-protocol deadlock on gfx12.
- Add `--enable-expert-parallel` (with `moe_wna16.py` mounted) for the EP variant —
  +8–10 % decode on 4 GPUs and the only route to 8.
- Production knobs that survived testing: `--gpu-memory-utilization 0.95` (0.97 loads
  but OOMs the engine as soon as prefills batch), `--max-num-seqs 4` (2 trips a
  torch.compile `ConstraintViolation` on `query_start_loc`), and an explicit
  `cudagraph_capture_sizes` ladder (`[1,2,3,4,6,8,12]` for seqs 4 — the default ladder
  at `--max-num-seqs 32` balloons to 192 sizes and leaves 0 GiB for KV). With those,
  `--max-model-len 131072` fits at 2.08× concurrency (272K-token KV pool).
- `--kv-cache-dtype auto` (i.e. BF16) is **mandatory**, not a choice — the
  architecture's attention mechanism (QSA) raises `NotImplementedError` on
  FP8 KV. This is unrelated to any patch here.
- `--enforce-eager` is required for an unrelated reason: a Triton/Inductor
  autotuning bug we hit but did not root-cause (see Known limitations).
- `--speculative-config` enables the checkpoint's native MTP head — see
  results below. Drop it for the no-MTP baseline (and `--max-model-len`
  can go back up to 262144 without MTP's extra KV reservation eating the
  margin).
- `ple_layer.py` is shipped here as the full patched file (not a diff) since
  the upstream file may move between vLLM revisions — `ple_layer.patch` is
  included for reference against the exact commit above.

## Validation

Real HumanEval pass@1 (lm-eval, chat-completions, generations re-scored
against the official HumanEval `check()` — lm-eval's own bundled scorer
misreads markdown-fenced chat answers as failures), needle-in-haystack, and
live `/metrics` — not vendor claims.

| | No MTP | MTP enabled |
|---|---|---|
| HumanEval pass@1 | 96.34% (158/164) | 96.34% (158/164) — **identical failures** |
| Single-stream speed | 6.7–6.8 tok/s | 8.3–11.0 tok/s |
| Spec-decode acceptance | — | 61.8% (446/722 draft tokens) |
| Needle-in-haystack | clean @ 123.5K tokens | — |
| VRAM per rank | ~28 GiB | ~28 GiB |

PIECEWISE graphs + MTP, P2P env, gmu 0.95 / mml 131072 / seqs 4 (2026-09-04, n=8
per cell, 512-token generations, prefix-cached, otherwise idle box):

| | 55K-token context | 120K-token context |
|---|---|---|
| TP4 | 45.1 ± 2.0 tok/s | 44.8 ± 3.2 tok/s |
| TP4 + expert parallel | **49.4 ± 3.2 tok/s** | **48.3 ± 3.3 tok/s** |

Decode is flat with depth. HumanEval has **not** yet been re-run on the EP
variant (arithmetic sanity only) — treat it as speed-validated, quality-pending,
on top of the PIECEWISE caveat below.

The identical HumanEval score and identical six failed problems (`find_zero`,
`decode_cyclic`, `decode_shift`, `rounded_avg`, `fix_spaces`,
`order_by_points`) with MTP on vs. off is the expected signature of
*correct* speculative decoding — MTP verifies drafts against the base
model's own output distribution, so bit-identical greedy output (not just
"similar") is real evidence it's implemented correctly, not silently
degrading quality for speed.

For reference, on the same hardware and methodology: a dense Qwen3.8-27B-FP8
scores 93.29% (153/164); Gemma-4-26B scores 96.34% (158/164, different 6
failures than Flash-Next).

## Known limitations (not yet fixed)

- **`--enforce-eager` remains the recommended, validated choice** — full graph
  capture now *works* (see below) but the fastest working config has an
  unresolved quality regression, so it is not recommended yet. Root cause of
  the original crash: `torch.compile`/Inductor tracing succeeds cleanly (QSA
  is already excluded from it via the same `no_compile_layers` mechanism the
  PLE gather uses). The real failure was one phase later, during CUDA/HIP
  **graph capture**: QSA's Triton attention kernels run autotuning that
  explores candidate configs invalid for gfx1201's WMMA v2 ISA — normally
  harmless (Triton discards failed candidates), but fatal mid-capture
  (`hipErrorStreamCaptureUnsupported`). Fixed via a kernel warmup
  (`qwen4_exp_amd_qsa_warmup.py`, wired in via `cudagraph_utils.py` +
  `kernel_warmup.py`) that resolves QSA's autotuning in eager mode before
  capture starts — validated via logs, no longer crashes. This warmup is
  independently correct and included in the patch set regardless of the
  point below.
- **Graph capture, once QSA stopped crashing first, hit a second, structural
  wall:** the PLE offload's host-RAM copy (`ple_cpu.py`) cannot be recorded
  into a CUDA/HIP graph — host-synchronizing ops are disallowed during
  capture on any hardware, not a ROCm quirk. Default `splitting_ops`
  (`CompilationConfig._attention_ops`) already excludes two other PLE-related
  ops from capture but not this one — likely a genuine upstream gap, since
  the op didn't exist before this repo's offload patch. Adding it explicitly
  (`compilation_config.json`) plus switching to pure `cudagraph_mode:
  PIECEWISE` (the v1 *default* is `FULL_AND_PIECEWISE`, which captures decode
  batches as one monolithic FULL graph regardless of `splitting_ops` — pure
  `PIECEWISE` is required so decode batches split too) **works and is
  substantially faster: 18.3–18.5 tok/s**, ~2.7x the no-MTP eager baseline.
  But HumanEval on this config regresses, and a three-point concurrency
  sweep has since ruled out the obvious explanation: **96.34% (158/164)
  eager baseline vs 95.12% @ c1, 95.73% @ c2, 93.90% @ c8.** Concurrency is
  clearly *part* of it — the extra-failure count tracks it — but the failing
  *sets* don't nest, and one problem (`minPath`) fails at every level tested
  **including fully serialized c1**. A regression that survives zero
  concurrent requests is not a race: PIECEWISE has an unresolved correctness
  gap in its own execution path (most likely floating-point drift at the
  graph-split boundary), with a second, sporadic concurrency-linked effect
  stacked on top. **Not adopted, and not recommended at any concurrency** —
  including single-stream, which is where it would otherwise be tempting to
  call it safe. Trading ~1-4% correctness for 2.7x speed without knowing why
  fails this repo's validated-results bar. The speedup is real and worth
  returning to; the concrete next step is a token-level diff of `minPath`'s
  output between eager and PIECEWISE at c1, to separate small numerical
  drift from an actual logic divergence. Full data, exact configs, and the
  c1/c2 harness (`run_humaneval_c1.sh`, `run_humaneval_c2.sh`):
  `GRAPH-CAPTURE-FIX.md`.
- **AITER untried** — `VLLM_ROCM_USE_AITER=1` is validated elsewhere on
  this hardware, but historically needs FP8 KV to avoid a separate LDS-
  overflow crash, which conflicts with this architecture's mandatory BF16
  KV. May simply be incompatible as-is.
- **No RDNA4-specific kernel tuning applied** — precedent on this hardware
  shows ~0% gain from this class of tuning on MoE architectures specifically
  (vs. +2–4% on dense models), so it's a low priority.
- Both patched files were built and validated against the exact vLLM commit
  above; a moving target (`qwen4_exp` support is ~2 days old at the time of
  writing) — expect to need to re-verify against newer nightlies.

## License

These files are modifications of vLLM source (Apache License 2.0) and are
distributed under the same license. See the original files in the vLLM
repository for full copyright headers.

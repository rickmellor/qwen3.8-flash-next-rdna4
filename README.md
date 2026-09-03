# Running Qwen3.8-Flash-Next (AWQ) on AMD RDNA4 / vLLM-ROCm

Three small patches to vLLM that get `leoncca/Qwen3.8-Flash-Next-AWQ-g32` — an
AWQ-quantized version of Alibaba's Qwen3.8-Flash-Next (a preview of the Qwen4
architecture: 125B MoE + a 51GB sparse n-gram embedding table, ~6B parameters
active per token) — loading and serving on consumer/workstation AMD GPUs.

None of these are architectural limitations. Each is a narrow, fixable gap in
day-2 support for a brand-new model architecture (`qwen4_exp`, merged into
vLLM ~Sept 2 2026) meeting an AWQ checkpoint layout and a ROCm fallback code
path that hadn't been exercised together before.

## Environment

- Hardware: 4× AMD Radeon AI PRO R9700 (RDNA4, gfx1201, 32 GB each), TP4
- vLLM: `vllm/vllm-openai-rocm:nightly-27a94d1ce4e3fc100c4732439ccec10f8246a804`
  (the Sept 2 2026 nightly — **required**; mainline v0.28.0 predates
  `qwen4_exp` architecture support entirely)
- Checkpoint: [`leoncca/Qwen3.8-Flash-Next-AWQ-g32`](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-AWQ-g32)
  (expert-only AWQ W4A16, group size 32; 129 GB on disk)

## The three problems, and the fixes

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

## Applying the patches

No image rebuild needed — bind-mount the three files over the installed
copies:

```bash
docker run -d --name flashnext \
  --device /dev/kfd --device /dev/dri --group-add 44 --group-add 992 \
  --ipc host --shm-size 16g \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e VLLM_QWEN4EXP_PLE_CPU_OFFLOAD=1 \
  -e NCCL_P2P_DISABLE=1 -e RCCL_NET=Socket -e NCCL_PROTO=Simple \
  -e HIP_FORCE_DEV_KERNARG=1 -e SAFETENSORS_FAST_GPU=1 \
  -v $PWD/ple_layer.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd/ple_layer.py \
  -v $PWD/ple_cpu.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/common/ple_cpu.py \
  -v $PWD/routed_experts.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/routed_experts.py \
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

- **`--enforce-eager` is required — root cause now understood, fix not yet found.**
  `torch.compile`/Inductor tracing actually succeeds cleanly (QSA is already
  correctly excluded from it via the same `no_compile_layers` opaque-custom-op
  pattern used for the PLE gather). The real failure is one phase later, during
  CUDA/HIP **graph capture**: QSA's Triton attention kernels
  (`amd/ops/qsa.py`) run autotuning that explores multiple candidate kernel
  configs, some invalid for gfx1201's WMMA v2 ISA — normally harmless (Triton
  silently discards failed candidates), but fatal when it happens on a stream
  that's actively being graph-captured (`hipErrorStreamCaptureUnsupported`).
  This is why eager mode works fine (autotuning never runs inside a capturing
  stream) despite hitting the identical kernels.
  Tried: `--compilation-config '{"cudagraph_mode":"NONE"}'` (disables graph
  capture, keeps Inductor fusion) — **loads and serves successfully, but is
  slightly slower than `--enforce-eager`** (4.8–5.7 tok/s vs. 6.7–6.8 tok/s),
  so not adopted. The real fix is pre-warming QSA's Triton kernels in eager
  mode *before* graph capture begins, so the autotuning search resolves and
  caches before any stream starts capturing — not yet implemented. Full
  investigation: `GRAPH-CAPTURE-FIX.md`.
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

# Draft forum post — Level1Techs (edit freely)

**Title:** Qwen3.8-Flash-Next (125B-A6B + 51B PLE, AWQ) on 4× Radeon AI Pro R9700 via vLLM-ROCm — 49 tok/s, full 131K context, and what "not bit-exact with eager" actually means

Repo with the patches, launch tooling, bench results and the write-ups:
**https://github.com/rickmellor/qwen3.8-flash-next-rdna4**

## The box

Threadripper PRO 3945WX / WRX80, 256 GB DDR4, six Radeon AI Pro R9700 (32 GB, RDNA4 / gfx1201)
on Gen5-switch risers, dual PSU. Four of the six run this model; the plan is eight via expert
parallelism (TP8 is impossible for this checkpoint — AWQ group-32 doesn't shard the 640-wide
experts eight ways).

## What it took (details and diffs in the README)

1. **PLE table doesn't fit.** The 51B parameter-learned-embedding table is gathered on the host
   (`VLLM_QWEN4EXP_PLE_CPU_OFFLOAD=1` + a small patch); weights + KV for 131K context then fit in
   4 × 32 GB.
2. **AWQ MoE loader crash** on a spurious checkpoint tensor — one-line skip.
3. **Graph capture crashed** (`hipErrorStreamCaptureUnsupported`): QSA's Triton kernels autotune
   lazily on first use and that can't happen mid-capture on gfx1201. Fix: warm the autotune in
   eager before capture, then run PIECEWISE cudagraphs with the PLE gather as a split point.
   Eager is 4–7× slower on decode; this is the difference between 7 and 45+ tok/s.
4. **Expert parallelism crashed** in vLLM's WNA16 MoE weight loader (global expert id used
   where a local one is needed) — ~20-line fix, not yet upstreamed. EP is +8–10 % over TP here
   and is the 8-GPU path.
5. **Mixed-length concurrent prefills OOM** in vLLM's PLE short-conv: it pads every in-flight
   request to the longest one and makes ~5 copies; the startup memory profile never sees the
   padded case. Found by a benchmark that put four ~16K prompts into prefill together, fixed with
   a bounded per-request fallback. Not RDNA4-specific — any tight-gmu deployment can hit it.

Plus one platform finding: **GPU P2P on gfx1201 was never an RCCL bug.** Upstream vLLM-ROCm
images from ~0.21 bake `HSA_ENABLE_IPC_MODE_LEGACY=1` into the image env, which forces the
pre-dma-buf KFD IPC path gfx12 rejects. `HSA_ENABLE_IPC_MODE_LEGACY=0` (with BIOS ACS off)
gives full-mesh P2P — RCCL all-reduce 19.7 vs 11.6 GB/s busbw, ~1.7× on collectives.

## Numbers (4× R9700, TP4 + EP, PIECEWISE + MTP, gmu 0.95, 131K context, 4 seqs)

- ~49 tok/s single-stream, flat from 0 to 120K context; ~120 tok/s aggregate at 4 concurrent
- KV pool ~220–270K tokens (varies between launches — see README)
- HumanEval 95.7 %, ARC-Challenge 97.0 % (first 400, CoT), PlanBench 89 % exact,
  AutomationBench 40 % pass / 67 % partial, code needle 15/16, context-safety probe clean to
  131,072 tokens

## The "not bit-exact with eager" thing

PIECEWISE output differs from eager output at temperature 0, and it's tempting to call that a
quality regression. Measured properly (README, "Where the drift comes from"): eager with
torch-native op implementations — no compiler in the loop — differs from eager-with-kernels
by exactly as much as PIECEWISE does. Swapping only the norm kernel, or only rotary, produces
the full effect. Every logit gap is a multiple of 1/16 — the logits are bf16 at magnitudes
8–32 — so any upstream ulp-level difference surfaces as 0.06–1 nats, mostly on tail tokens.
The chosen token moves ~0.02 nats; greedy flips only on near-ties. Both modes are perfectly
deterministic run-to-run. Nothing to fix; eager just isn't a gold standard. This should hold
for any bf16 deployment of this model on any hardware.

Happy to answer questions on any of it. Tooling for the P2P/RCCL probes, the depth validation,
the prefill stress test and the numerics study is all in `tools/`.

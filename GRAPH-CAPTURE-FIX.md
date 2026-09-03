# Investigating the `--enforce-eager` requirement (Inductor/CUDA-graph bug)

## Root cause: found, with real evidence (not the same bug reported earlier)

The original report (`AWQ-MOE-LOADER-FIX-v3.md`) hit `torch._inductor.exc.InductorError:
Failed to run autotuning code block` and worked around it with `--enforce-eager` without
capturing a full traceback. Reproducing tonight with full logging shows this description
was imprecise — **`torch.compile`/Inductor tracing actually succeeds** (114s, clean, both
attempts). The real failure is one phase later, during **CUDA/HIP graph capture**
(`Capturing CUDA graphs (PIECEWISE): 0/4`), and the true error is:

```
vllm/models/qwen4_exp/amd/ops/qsa.py:306:12: error: no matching matrix core intrinsic
for wmma version 2 with instruction shape [0, 0, 16] and element types A='bf16', B='bf16', C='f32'.
```
(also seen for shapes `[0, 0, 256]` and `[0, 0, 64]` across runs)

followed by a cascading, more generic-looking secondary error:
```
RuntimeError: Worker failed with error 'CUDA error: operation not permitted when stream is capturing
Search for `hipErrorStreamCaptureUnsupported'...
```

**Mechanism, confirmed by evidence, not guessed:**

1. `Qwen4ExpQSAAttention` (the novel QSA attention mechanism) is *already* correctly
   marked as an opaque custom op excluded from `torch.compile`/Inductor tracing via the
   same `no_compile_layers` pattern used for the PLE embedding gather (`vllm/models/qwen4_exp/amd/qsa.py`,
   `direct_register_custom_op` + `get_forward_context().no_compile_layers[layer_name]`).
   This is why Inductor compilation itself succeeds cleanly — the earlier report's
   "InductorError" framing was misleading; that mechanism was never actually broken.
2. QSA's attention math runs as raw `@triton.jit` kernels (`vllm/models/qwen4_exp/amd/ops/qsa.py`),
   invoked eagerly from Python even inside a compiled forward pass (that's the point of
   being "opaque"). Triton's autotuner explores multiple candidate kernel configs for these
   kernels and **expects some candidates to fail to compile** — this is normal, silent,
   non-fatal behavior during ordinary eager execution (confirmed: the validated
   `--enforce-eager` runs tonight, including a clean 96.34% HumanEval pass, exercise these
   same kernels without incident).
3. The failure is specifically that **Triton's on-the-fly candidate-compilation process is
   incompatible with happening on a HIP stream that is actively mid-graph-capture** — you
   cannot launch/benchmark new kernel candidates while a stream is being recorded for replay.
   When graph capture reaches a QSA layer for the first time, it triggers this same
   multi-candidate autotuning search, and doing so *during capture* is what's fatal — not
   the WMMA-shape errors themselves, which are routine autotuner noise in any other context.

This reframes the bug from "gfx1201 can't run this architecture's attention kernel" (which
would be a hard blocker) to "a lazy-JIT + CUDA-graph-capture ordering conflict" — a much
more tractable, well-understood class of problem.

## Attempted fix: `--compilation-config '{"cudagraph_mode":"NONE"}'`

Disables CUDA/HIP graph capture entirely while leaving `torch.compile`/Inductor active
(these are separate, independently-controllable mechanisms in vLLM — confirmed via
`vllm.config.compilation.CUDAGraphMode`: `NONE`, `PIECEWISE`, `FULL`, `FULL_DECODE_ONLY`,
`FULL_AND_PIECEWISE`).

**Result: loads and serves successfully without `--enforce-eager`.** `Application startup
complete`, confirmed via `/v1/models`. The WMMA errors still print during the warmup pass
(Triton's autotuner still explores the same bad candidates) but are now non-fatal, exactly
as the mechanism above predicts, because no stream is capturing when they occur.

**But it is NOT faster — it's slightly slower:**

| Config | Code probe | Prose probe |
|---|---|---|
| `--enforce-eager` (current default) | — | 6.7–6.8 t/s (documented baseline) |
| `--compilation-config cudagraph_mode=NONE` | 5.7 t/s | 4.8 t/s |

This is a real, honest negative result, not a partial win to adopt. The expected speedup
was always attributed to CUDA graph replay's per-step launch-overhead elimination
(see the parent session's earlier analysis) — Inductor's kernel fusion alone, without
graph capture, isn't enough to recover that, and for this launch-heavy MoE architecture may
even add net dispatch overhead relative to raw eager execution.

**Bisecting `cudagraph_capture_sizes` to exclude just the failing size(s) was considered
and rejected as unlikely to help**, not attempted further: the WMMA failures occurred with
*different* instruction shapes across otherwise-identical runs ([0,0,16]/[0,0,256] on one
attempt, [0,0,16]/[0,0,256]/[0,0,64] on another) — consistent with the autotuner's candidate
search itself being the variable, not a specific batch-size-dependent tensor shape. This
suggests every capture size would eventually trigger the same class of failure, not just one.

## Recommendation: stay on `--enforce-eager` for now

Neither tested configuration beats the current baseline. The real fix — recovering CUDA
graph capture's actual performance benefit — needs one of:

1. **Pre-warm QSA's Triton kernels in eager mode before graph capture begins**, so the
   autotuner's candidate search (and its expected-but-fatal-during-capture failures)
   completes and caches a working kernel *before* any stream starts capturing. vLLM already
   has a `kernel_warmup.py` mechanism (seen in logs: "JIT kernel warmup starting... finished
   in 0.00s") but it appears to be a no-op for QSA specifically — likely doesn't know to
   warm these particular kernels. Worth investigating whether QSA can register into that
   hook, or whether a manual dummy eager forward pass through a QSA layer before capture
   begins would force the caching.
2. **Fix the WMMA lowering gap directly** — genuinely out of scope locally (Triton AMD
   backend codegen), but worth an upstream report given it's reproducible and precisely
   characterized here.

No repo/registry changes made — the recommended launch command is unchanged from
`README.md`'s existing `--enforce-eager` configuration, which remains the empirically
faster, validated choice.

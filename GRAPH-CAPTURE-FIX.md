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

## Kernel-warmup attempt 2026-09-03 (later): real progress, no working fix yet

Picking up from the prior session's finding (root cause: QSA's Triton autotuner hits
gfx1201 WMMA-lowering failures that are harmless in eager mode but fatal when they occur
on a HIP stream that's mid-graph-capture). This session tried to fix it by pre-warming
QSA's kernel in eager mode before any capture starts, instead of disabling capture
(`cudagraph_mode: NONE`, previously tried and rejected as slower).

### Gap 1 found and fixed: upstream's own QSA warmup hook is NVIDIA-only

`vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py`'s `qwen4_exp_qsa_triton_warmup`
opens with `sys.modules.get("vllm.models.qwen4_exp.nvidia.indexer_qsa")` — never
imported on ROCm, so the function is a silent no-op on AMD (matches the prior session's
"finished in 0.00s" observation). It also targets the wrong kernel even conceptually:
NVIDIA's version warms `qsa_mqa_paged` (a paged-decode MQA kernel via `QSAIndexer`), not
`_qsa_sparse_paged_gqa_splitk_kernel` (`amd/ops/qsa.py:306`, invoked via
`Qwen4ExpQSAAttention.forward_qsa` -> `qsa_sparse_paged_attention`) — the actual crash
site, reached from the main attention path, not the indexer.

Wrote an AMD-specific equivalent (`qwen4_exp_amd_qsa_warmup.py`, new file) that forces a
real eager forward pass through the QSA attention path via vLLM's own
`GPUModelRunner._dummy_run(cudagraph_runtime_mode=CUDAGraphMode.NONE,
force_attention=True)` — the same primitive `kernel_warmup.py` already uses for FlashInfer
attention warmup elsewhere in that file, just not wired up for this arch/platform. Patched
`kernel_warmup.py` to call it. **Confirmed via logs this warmup now correctly identifies
and targets Qwen4Exp QSA layers on all 4 TP ranks** — but the crash still occurred,
unchanged, because:

### Gap 2 found: `kernel_warmup()` runs too late — after the crash point

The actual first real graph-capture attempt is not in `capture_model()` (the main capture
phase, which runs after KV-cache init and `kernel_warmup()`) — it's earlier, in
`Worker.determine_available_memory()` (`gpu_worker.py`), which calls
`self.model_runner.profile_cudagraph_memory()` specifically to *estimate* how much extra
GPU memory graph capture will need, before KV-cache size is even decided. This happens
before `kernel_warmup()` is ever called. Moved the warmup call to inside
`determine_available_memory()`'s `memory_profiling` context, right after `profile_run()`
and before `profile_cudagraph_memory()` (patched `gpu_worker.py`) — this is the correct
ordering fix, confirmed by the warmup log lines now appearing before the crash instead of
never appearing at all.

### Gap 3 found, not fixed: `force_attention=True` needs KV-cache block tables that don't exist yet

With the warmup now running at the right *time*, it fails on a different, structural
error: `AttributeError: 'GPUModelRunner' object has no attribute 'block_tables'`
(`gpu/model_runner.py:1399`, `prepare_dummy_attn` -> `self.block_tables.get_dummy_block_tables(...)`).
`self.block_tables` is a `BlockTables` manager created in `initialize_kv_cache()`
(`gpu/model_runner.py:615`) — which itself only runs *after* `determine_available_memory()`
returns, since sizing KV cache is the whole point of that phase. This is a real
chicken-and-egg ordering constraint, not a superficial bug: `_dummy_run(force_attention=True)`
fundamentally needs real block-table infrastructure to build attention metadata, and that
infrastructure cannot exist yet at any point before KV-cache sizing completes.

**Open question, not resolved:** `profile_cudagraph_memory()` (delegating to
`profile_cudagraph_memory` in `vllm/v1/worker/gpu/cudagraph_utils.py`, imported as
`_profile_cudagraph_memory`) clearly *does* reach real Triton kernel compilation (that's
what crashes on the WMMA error) despite running in this same pre-KV-cache-init window —
meaning it must construct some form of temporary/estimated attention metadata internally,
independent of `self.block_tables`. Whoever picks this up next should start by reading
`vllm/v1/worker/gpu/cudagraph_utils.py`'s `profile_cudagraph_memory` to find whatever
mechanism it uses for that, and either (a) hook a QSA warmup call using the same
mechanism, immediately before its own capture attempt inside that function, or (b) find a
way to give `_dummy_run` a temporary/estimated block-table stand-in early enough to be
usable during the profiling phase.

### Net result

**No working fix.** Two real, precisely-characterized structural gaps found and fixed in
sequence (wrong platform target, wrong pipeline phase); a third, deeper one
(block-table availability) blocks further progress without understanding
`profile_cudagraph_memory`'s internal metadata construction — a genuinely separate
investigation, not attempted further this session per the time-boxing on this class of
bug. `--enforce-eager` remains the correct, validated choice; no speed measurement or
HumanEval re-run performed (per updated coordinator directive: only validate if a working
config is reached).

Files added this session (staged in `~/scratch/flashnext/`, not yet committed):
`qwen4_exp_amd_qsa_warmup.py` (new AMD warmup, functionally correct — successfully
targets and would warm the right kernel once reachable), `kernel_warmup.py` (wires it in,
correct but insufficient alone), `gpu_worker.py` (moves the call earlier, correct
ordering fix, but hits the block_tables wall). All three are real, validated partial
progress — worth keeping and building on, not throwing away, even though the end-to-end
fix isn't there yet.

## Kernel-warmup attempt 2026-09-03/04 (continued): QSA fix works, uncovers a deeper structural conflict

Picking up from Gap 3 (block_tables unavailable during warmup). Traced
`profile_cudagraph_memory()` (`vllm/v1/worker/gpu/cudagraph_utils.py`) and found the
answer to the open question: it does NOT use fake/estimated metadata. It calls the REAL
`initialize_kv_cache()` with a minimal config (`_init_minimal_kv_cache_for_profiling`,
`num_gpu_blocks_override = min(max_num_reqs, max_cudagraph_capture_size) or 1`), which
populates a real (if tiny) `runner.block_tables`, then tears everything down afterward
(`_teardown_profiling_state`) once capture-memory measurement is done. `_dummy_run`s

## Kernel-warmup attempt 2026-09-03/04 (continued): QSA fix works, uncovers a deeper structural conflict

Picking up from Gap 3 (block_tables unavailable during warmup). Traced
`profile_cudagraph_memory()` (`vllm/v1/worker/gpu/cudagraph_utils.py`) and found the
answer to the open question: it does NOT use fake/estimated metadata. It calls the REAL
`initialize_kv_cache()` with a minimal config (`_init_minimal_kv_cache_for_profiling`,
`num_gpu_blocks_override = min(max_num_reqs, max_cudagraph_capture_size) or 1`), which
populates a real (if tiny) `runner.block_tables`, then tears everything down afterward
(`_teardown_profiling_state`) once capture-memory measurement is done. `_dummy_run`'s
`force_attention=True` path was never fundamentally incompatible with this phase --
it just needed to run *after* this minimal KV cache exists, not before.

**Fix applied:** added `qwen4_exp_amd_qsa_triton_warmup_runner(runner)` (a runner-based
variant of the existing, correctly-targeted AMD QSA warmup function -- no logic changes,
just a parameter-source swap since this call site only has `runner`, not `worker`) and
called it from inside `profile_cudagraph_memory()`, immediately after
`_init_minimal_kv_cache_for_profiling(runner)` and before `runner.capture_model()` --
i.e. in the narrow eager-mode window where real block tables exist but no capture has
started yet. Confirmed via logs: warmup now runs at the right time, on all 4 TP ranks,
against the actual crash-site kernel (`_qsa_sparse_paged_gqa_splitk_kernel`).

**Result: the original bug is fixed.** The WMMA "no matching matrix core intrinsic"
messages still print (Triton's autotuner still explores the same invalid candidates) but
are now confirmed non-fatal -- exactly the predicted outcome, because they now happen
during eager warmup, not mid-capture. No crash from QSA/WMMA this run.

**But launch still fails -- new, different, deeper error:**
```
torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing
Search for `hipErrorStreamCaptureUnsupported` ...
```
Full traceback confirms the exact site: `vllm/models/qwen4_exp/common/ple_cpu.py:177`,
inside `_pinned()`, at `view.copy_(t)` -- a CPU<->CPU/host copy, reached via the
`qwen4_exp_amd_ple_ngram_embedding` custom op called from inside an Inductor-compiled
decode step (`inductor_cache/.../....py`, `torch.ops.vllm.qwen4_exp_amd_ple_ngram_embedding
.default(...)`), which is what's now getting captured into a HIP graph now that QSA no
longer crashes first and capture proceeds further.

**Root cause: this is a structural incompatibility, not a bug to patch around.**
`no_compile_layers` (the mechanism the PLE op already uses, see ple_layer.py's custom-op
registration and this repo's README) only excludes the op from Inductor's own FX-graph
tracing/compilation -- it says nothing to vLLM's separate CUDA/HIP graph-capture
mechanism (`capture_model()`), which still records every op it encounters into the
graph, including calls to "no_compile" custom ops. Any operation that synchronizes with
or copies to/from the CPU -- which host-RAM offload fundamentally requires, by design,
every forward pass -- cannot be recorded into a replayable GPU-only graph. This is
documented CUDA/HIP graph API behavior (cudaStreamIsCapturing/hipStreamIsCapturing
disallow host-synchronizing ops during capture), true on NVIDIA as much as AMD -- not a
ROCm/Triton immaturity issue like the QSA bug was.

**Practical implication:** full CUDA/HIP graph capture and the PLE CPU-offload patch
(this repo's core fix for fitting the 129GB checkpoint in VRAM at all) are mutually
exclusive as currently built. You cannot have both simultaneously. Recovering graph
capture's speed benefit would need the PLE layer specifically excluded from capture
(a piecewise/selective-capture mechanism -- letting everything else in the decode step
capture normally while this one op runs eager inline), not a global disable
(`cudagraph_mode: NONE`, already tried, measured slower) and not more warmup (the QSA
fix, while independently correct and worth keeping, doesn't touch this).

**Not attempted this session:** whether vLLM's `cudagraph_mode: PIECEWISE` (as opposed to
the apparent default/FULL used in this run) would naturally exclude a custom op like this
from capture the way it's designed to exclude other non-capturable regions, or whether
`no_compile_layers` membership could/should also gate cudagraph capture (a real, scoped
upstream fix if so -- the op declaring "don't compile me" arguably should also imply
"don't capture me" for any custom op that does host I/O, which is a reasonable general
rule, not specific to this checkpoint). Worth checking PIECEWISE mode specifically
before concluding this is a hard dead end.

**Status:** QSA kernel-warmup fix retained (independently correct, worth keeping in the
patch set regardless of the graph-capture outcome -- it's a small correctness/robustness
improvement even under `--enforce-eager`, since it's harmless to warm a kernel that's
about to be used eagerly anyway). No working faster-than-eager config reached. No speed
measurement, no HumanEval re-run (nothing to validate -- launch doesn't complete). No
repo commit (nothing shippable). `--enforce-eager` remains correct/validated.

Files: `qwen4_exp_amd_qsa_warmup.py` (updated, now exports both worker- and
runner-based entry points), `cudagraph_utils.py` (new patch, the actual fix location --
supersedes the prior session's `gpu_worker.py` patch, which called warmup from the
wrong location and should be considered obsolete/unused going forward). `kernel_warmup.py`
unchanged from prior session (still correct as a second, redundant-but-harmless warmup
call for the real post-profiling capture phase).

## PIECEWISE mode attempt 2026-09-03/04: real speedup, but unresolved quality regression

Picking up the one untried avenue from the prior session: whether `cudagraph_mode:
PIECEWISE` (as opposed to the v1 *default* `FULL_AND_PIECEWISE` -- correcting an earlier
assumption that default was plain `FULL`) would exclude the PLE host-copy op from capture.

**Mechanism, traced precisely (not guessed):** `splitting_ops` is a fixed allowlist
(`CompilationConfig._attention_ops`, ~17 entries), NOT anything derived automatically from
`no_compile_layers` membership. Two PLE-related ops are already in the default list
(`qwen4_exp_compute_ple_ngram_ids`, `qwen4_exp_ple_short_conv`) -- but the actual
host-copying gather op this repos patch added, , is

## PIECEWISE mode attempt 2026-09-03/04: real speedup, but unresolved quality regression

Picking up the one untried avenue from the prior session: whether `cudagraph_mode:
PIECEWISE` (as opposed to the v1 *default* `FULL_AND_PIECEWISE` -- correcting an earlier
assumption that default was plain `FULL`) would exclude the PLE host-copy op from capture.

**Mechanism, traced precisely (not guessed):** `splitting_ops` is a fixed allowlist
(`CompilationConfig._attention_ops`, ~17 entries), NOT anything derived automatically from
`no_compile_layers` membership. Two PLE-related ops are already in the default list
(`qwen4_exp_compute_ple_ngram_ids`, `qwen4_exp_ple_short_conv`) -- but the actual
host-copying gather op this repo's patch added, `qwen4_exp_amd_ple_ngram_embedding`, is
NOT. This looks like a genuine oversight in vLLM's own default list (they clearly
anticipated needing to split around some PLE ops, just not this specific one -- which
didn't exist upstream until this session's patch introduced host-RAM offload for it).

Also: `FULL_AND_PIECEWISE` (the default) captures FULL for decode batches specifically --
no splitting at all applies there, which is why the original crash happened even though
piecewise splitting nominally existed. Needed pure `PIECEWISE` (not `FULL_AND_PIECEWISE`)
so decode batches also go through the splitting mechanism.

**Fix:** `--compilation-config '{"cudagraph_mode":"PIECEWISE","splitting_ops":[...default
17 attention ops..., "vllm::unified_kv_cache_update", "vllm::unified_mla_kv_cache_update",
"vllm::qwen4_exp_amd_ple_ngram_embedding"]}'`, combined with the QSA kernel-warmup fix from
the prior session (`qwen4_exp_amd_qsa_warmup.py` -- corrected mount path this session, it
belongs at `vllm/model_executor/warmup/qwen4_exp_amd_qsa_warmup.py`, not under
`models/qwen4_exp/amd/`; `cudagraph_utils.py`; `kernel_warmup.py`).

**Result: loads and serves. Real, substantial speedup:**
- Single-stream: **18.3-18.5 tok/s** vs 6.7-6.8 (no-MTP eager) / 8.3-11.0 (MTP eager) --
  roughly 2.7x the no-MTP eager baseline, and faster than MTP-eager despite this config
  having no MTP at all.
- VRAM: ~30.7 GiB/rank (vs ~28 GiB eager -- graph capture buffers cost real memory,
  still comfortably under the 31.86 GiB card limit).

**But HumanEval re-run shows a real quality regression, not identical output:**
**93.90% (154/164)** vs the 96.34% (158/164) MTP-off/eager baseline -- 4 fewer correct.
Failure set grew from the baseline's 6 (find_zero, decode_cyclic, decode_shift,
rounded_avg, fix_spaces, order_by_points) to 10, adding is_simple_power, sort_array,
minPath, is_nested. At temperature=0 this should be deterministic if the configs were
truly numerically equivalent -- they are not. Not root-caused this session.

**Leading suspect, not confirmed:** the HumanEval run used num_concurrent=8 (batched,
concurrent requests); the speed probe that showed no obvious problems was single-request.
A concurrency-dependent interaction at the forced split boundary (batched requests mixing
around the now-excluded PLE op, possible KV/state handling difference under PIECEWISE's
per-batch graph selection when multiple requests are in flight) is the most likely
culprit, but this is a hypothesis, not a verified root cause. Floating-point
non-associativity from the different kernel/graph path is a second, less likely
possibility (would typically cause much smaller/rarer divergence than 4 flipped answers
out of 164, but not ruled out).

**Disposition:** NOT adopted as the recommended config -- a ~2.4 percentage point quality
regression with an unidentified cause fails this repo's validated-results standard, even
though the speed number is real and substantial. The QSA kernel-warmup fix is kept and
shipped on its own merits (independently correct, necessary building block, harmless
under --enforce-eager too). The PIECEWISE + extended-splitting-ops combination is
recorded here as a promising but unverified lead for whoever continues this -- next step
would be isolating whether the regression is concurrency-dependent (rerun HumanEval at
num_concurrent=1) before considering this shippable.

Files added: `compilation_config.json` (the working PIECEWISE launch config, for
reference). Corrected: `qwen4_exp_amd_qsa_warmup.py` now confirmed mounted at
`vllm/model_executor/warmup/qwen4_exp_amd_qsa_warmup.py` (prior session's writeup didn't
specify the exact target path; this was a real bug the first launch attempt this session
hit and fixed).

## PIECEWISE concurrency isolation 2026-09-03/04: real, but not purely a race

Three-point HumanEval comparison, all at 0 temperature, PIECEWISE + QSA-warmup config
(18.3-18.5 tok/s, ~2.7x eager):

| Concurrency | Score | Fails | Extra failures beyond the 6-baseline-hard set |
|---|---|---|---|
| eager baseline (any) | 96.34% (158/164) | 6 | — (find_zero, decode_cyclic, decode_shift, rounded_avg, fix_spaces, order_by_points) |
| PIECEWISE @ 1 | 95.12% (156/164) | 8 | minPath, get_max_triples |
| PIECEWISE @ 2 | 95.73% (157/164) | 7 | minPath |
| PIECEWISE @ 8 | 93.90% (154/164) | 10 | is_simple_power, sort_array, minPath, is_nested |

**Not a clean concurrency gradient.** The extra-failure *count* roughly tracks concurrency
(2 -> 1 -> 4), but the extra-failure *set* does not nest or overlap cleanly: c1's unique
extra (get_max_triples) doesn't appear at c2 or c8; c8's three unique extras
(is_simple_power, sort_array, is_nested) don't appear at c1 or c2. Only one problem --
**minPath** -- fails at every concurrency level tested, including fully serialized c1.

**This changes the verdict.** A problem failing at concurrency=1 rules out a pure
concurrency-race explanation for the whole regression -- at minimum, minPath reflects a
real, reproducible-enough correctness gap in PIECEWISE's execution path itself (most
likely floating-point non-associativity from the different kernel/graph-split boundary,
consistent with it being sensitive to something inherent to the piecewise split rather
than only to request interleaving). The other extra failures (get_max_triples,
is_simple_power, sort_array, is_nested) look like a second, genuinely concurrency-linked
effect -- more failures pile up as more requests are in flight -- but not a strictly
monotonic or deterministic one; they're closer to sporadic/probabilistic than to a fixed
set of "fragile under load" problems.

**Honest final verdict: not recommended at any concurrency without further root-causing.**
The temptation is "safe for single-stream, unsafe for concurrent serving" -- that would be
the story if minPath had passed at c1. It didn't. A ~0.6-1.2 percentage point regression
persists even fully serialized, which means PIECEWISE mode has an unresolved correctness
issue independent of concurrency, on top of a separate, less-understood concurrency-linked
one. Two unexplained failure modes stacked is a real asterisk, not "basically fine." The
2.7x speed number is real and worth returning to once someone traces why minPath
specifically diverges (comparing its generated output token-by-token against the eager
baseline's would be the concrete next step -- not attempted this session), but shipping it
as a recommended config today would mean quietly trading ~1-4% correctness for speed
without understanding why, which fails this repo's validated-results standard the same way
the original concurrency=8 result did.

**MTP + PIECEWISE combination: not attempted**, per standing instruction not to compound
onto an unvalidated base -- correct call, since the base itself just got less validated,
not more, from this session's data.

**Status:** `--enforce-eager` remains the only recommended launch config. QSA warmup fix
stands on its own merits regardless (already pushed, commit `e724065`). PIECEWISE lead
recorded here as a real, partially-characterized, NOT-adopted option for whoever continues
this -- the concrete next diagnostic step is a token-level diff on minPath's output between
eager and PIECEWISE at concurrency=1, to determine whether it's small numerical drift
(likely fixable/acceptable) or a larger logic divergence (likely a real bug).

Files added this session: `run_humaneval_c1.sh`, `run_humaneval_c2.sh` (reference,
concurrency isolation harness). Samples: `humaneval_piecewise_c1/`, `humaneval_piecewise_c2/`.
No new commit to the GitHub repo -- nothing more shippable than what commit `e724065`
already captured; this session's finding is a *correction* to that commit's "promising
lead" framing (it's less promising than it looked, not more), which the writeup above
records for accuracy but doesn't warrant its own repo change.

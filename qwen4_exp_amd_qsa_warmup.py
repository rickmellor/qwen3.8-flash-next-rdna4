# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AMD (Triton/gfx1201) equivalent of qwen4_exp_qsa_warmup.py.

The upstream qwen4_exp_qsa_triton_warmup only recognizes the NVIDIA QSA
indexer module and is a silent no-op on AMD (it checks
sys.modules.get("vllm.models.qwen4_exp.nvidia.indexer_qsa"), which is never
imported on ROCm). Separately, that function targets the indexer's paged
*decode* kernel (qsa_mqa_paged) -- a different kernel from the one that
actually fails during CUDA-graph capture on this platform.

The real crash site is `_qsa_sparse_paged_gqa_splitk_kernel`
(amd/ops/qsa.py), reached via Qwen4ExpQSAAttention.forward_qsa ->
qsa_sparse_paged_attention -- the main QSA attention path, not the indexer.
Triton's autotuner explores multiple candidate kernel configs for this
Triton kernel; some are invalid for gfx1201's WMMA v2 ISA. That's silent and
harmless in ordinary eager execution (the bad candidates are just discarded)
but fatal if it happens on a HIP stream that's actively being graph-captured
(GRAPH-CAPTURE-FIX.md has the full mechanism/evidence).

Fix: force a real forward pass through the QSA attention path in eager mode
(cudagraph_runtime_mode=NONE, force_attention=True) *before* graph capture
begins, at each batch size graph capture will itself use. This is the same
mechanism vLLM's own kernel_warmup.py already uses for FlashInfer attention
warmup (see the `_dummy_run(..., force_attention=True, ...)` call in that
file) -- not a new pattern, just applying the existing one to a
platform/architecture combination upstream's warmup registry doesn't yet
cover.
"""

import sys

from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger

logger = init_logger(__name__)


def qwen4_exp_amd_qsa_triton_warmup(worker) -> None:
    """Force-run QSA's attention kernel in eager mode, at every batch size
    graph capture will use, so Triton's autotuner resolves + caches a
    working config before any HIP stream starts capturing."""
    qwen4_exp_amd_qsa_triton_warmup_runner(worker.model_runner)


def qwen4_exp_amd_qsa_triton_warmup_runner(runner) -> None:
    """Same as qwen4_exp_amd_qsa_triton_warmup, but takes a GPUModelRunner
    directly instead of a Worker -- for call sites that only have the
    runner (e.g. cudagraph_utils.profile_cudagraph_memory, which runs
    *before* kernel_warmup() and is the actual first place graph capture is
    attempted; see GRAPH-CAPTURE-FIX.md "Gap 2"). GPUModelRunner exposes the
    same get_model()/vllm_config/_dummy_run surface a Worker delegates to,
    so this is a plain parameter-source swap, not different logic.

    Must be called with runner.block_tables populated (i.e. after
    initialize_kv_cache() has run -- real or the minimal profiling one from
    _init_minimal_kv_cache_for_profiling) or _dummy_run(force_attention=True)
    raises AttributeError (see GRAPH-CAPTURE-FIX.md "Gap 3").
    """
    qsa_module = sys.modules.get("vllm.models.qwen4_exp.amd.qsa")
    if qsa_module is None:
        return
    has_qsa = any(
        isinstance(layer, qsa_module.Qwen4ExpQSAAttention)
        for layer in runner.get_model().modules()
    )
    if not has_qsa:
        return

    compilation_config = runner.vllm_config.compilation_config
    sizes = list(compilation_config.cudagraph_capture_sizes or [])
    if not sizes:
        sizes = [1]

    logger.info(
        "Warming up Qwen4Exp AMD QSA attention kernel (eager) at sizes: %s.",
        sizes,
    )
    for num_tokens in sizes:
        runner._dummy_run(  # noqa: SLF001
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            force_attention=True,
            skip_eplb=True,
            is_profile=True,
        )
    logger.info("Warmed up Qwen4Exp AMD QSA attention kernel.")

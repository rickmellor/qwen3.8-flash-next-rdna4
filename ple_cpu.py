# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-RAM-resident variant of PLEVocabParallelEmbedding for Qwen4Exp.

The PLE (n-gram) embedding table is large (~50GB checkpoint) but accessed as
a sparse row gather -- a handful to a few thousand rows per forward pass, not
a bulk transfer -- so keeping it host-resident and gathering+copying only the
touched rows is cheap relative to a decode/prefill step's time budget.

Opt-in via VLLM_QWEN4EXP_PLE_CPU_OFFLOAD=1. Off by default; behavior and
performance for every other model/platform are unchanged.

See ~/scratch/flashnext/PLE-OFFLOAD-PLAN.md and AWQ-MOE-LOADER-FIX-v2.md for
the research this is based on:
- PLE table measured 52.26 GB on disk (fp8), one dense layer, TP-sharded with
  no redundancy (each rank already holds a disjoint slice, not a full copy).
- The checkpoint's PLE table is genuinely FP8-quantized (F8_E4M3) with a
  companion per-tensor `weight_scale` (confirmed 2026-09-03: checkpoint shape
  [1], dtype BF16 -- a SINGLE scalar covering the entire table for this
  layer, not per-row/per-group/per-shard: there is exactly one
  `weight_scale` key in the checkpoint per PLE layer, shared by all 128
  `shard_N.weight` tensors). `Qwen4ExpNGramEmbedding.__init__` (amd/ple_layer.py)
  never passes `quant_config` to the embedding construction, so the stock
  (non-offload) `PLEVocabParallelEmbedding` also gets `UnquantizedEmbeddingMethod`
  and would hit the identical "no parameter named ...weight_scale" error if it
  ever reached weight-loading for PLE -- it never has, always OOMing (no
  offload) or crashing on unrelated MoE-loader bugs (with offload, before
  those were fixed) first. This is a pre-existing gap in the stock
  construction call, not something this subclass introduced; fixed here
  rather than in the shared construction site or common/ple.py since the
  stock path is moot for us (it can never reach this code without first
  OOMing on VRAM) and a narrower fix has smaller blast radius.
- Raw fp8 storage is kept as-is (no upcast at load time) -- HALVES the
  per-rank host-RAM footprint versus upcasting to bf16 at load
  (~12.8 GB/rank vs ~25.6 GB/rank) and is also now the *correctness*
  requirement: dequantizing needs the scale, so an unscaled dtype cast on
  load would have been silently wrong even before the loader started
  raising on the missing weight_scale parameter.
- use_fused_embedding is unconditionally False on ROCm already
  (current_platform.is_cuda() gates it), so there's no fused-kernel path to
  work around here.
- The custom PLE gather op (qwen4_exp_amd_ple_ngram_embedding) already runs
  outside HIP/Inductor graph capture by design (reads from
  no_compile_layers) -- this patch doesn't change that, and doesn't need to.
"""

import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.vocab_parallel_embedding import (
    get_masked_input_and_mask,
)

from .ple import PLEVocabParallelEmbedding


class PLECpuOffloadEmbedding(PLEVocabParallelEmbedding):
    """PLEVocabParallelEmbedding with its weight pinned on host RAM, raw fp8.

    Each rank keeps exactly the same vocab-parallel shard it would otherwise
    hold in VRAM (see VocabParallelEmbedding.__init__ / self.shard_indices) --
    this only relocates that shard's storage, it does not change sharding.
    The gathered fp8 rows are dequantized (via the per-tensor weight_scale)
    on the CPU side, on just the few thousand touched rows, before the
    (already small) transfer to the compute device -- not the whole table.
    """

    def __init__(self, *args, **kwargs) -> None:
        # Keep the checkpoint's native fp8 storage dtype instead of letting
        # it default to the model's compute dtype (bf16) -- see module
        # docstring. Only applies if the caller didn't already specify one
        # (it doesn't, today, but this stays a default rather than a
        # hard override).
        kwargs.setdefault("params_dtype", torch.float8_e4m3fn)
        # Escape whatever ambient torch.device(...) context the model loader
        # has active (normally the compute GPU) so create_weights() lands the
        # param on CPU instead. Nothing else in __init__ is device-bound --
        # tp_rank/tp_size come from the process group, not from a tensor op.
        with torch.device("cpu"):
            super().__init__(*args, **kwargs)
        # Pin *before* weight loading populates it, so the checkpoint copy
        # (copy_ple_embedding_shard_ in ../common/ple.py -- already
        # device-agnostic, no changes needed there; with matching fp8 dtypes
        # on both sides it's now a pure byte copy, no numeric cast) writes
        # straight into page-locked memory instead of pinning a large tensor
        # after the fact.
        self.weight.data = self.weight.data.pin_memory()
        # Per-tensor dequant scale. Checkpoint: shape [1], bf16, ONE value
        # for the whole table (not vocab-sharded) -- every TP rank loads the
        # identical scalar. No custom weight_loader needed: vLLM's generic
        # AutoWeightsLoader auto-copies into any same-shape registered
        # parameter it finds by name (`ngram_embedding.weight_scale`), which
        # is exactly what registering it here as a plain Parameter provides.
        # Explicit device="cpu" here (not relying on an outer `with
        # torch.device("cpu")` -- this line is intentionally outside that
        # block, see __init__ above) because this Parameter's creation is
        # NOT wrapped in one; omitting it silently inherits whatever GPU
        # device context is ambient at construction time, which is exactly
        # what happened here originally and produced a real
        # "cuda:N and cpu" RuntimeError in forward()'s dequant multiply
        # (self.weight_scale ended up on GPU while the gathered/cast rows
        # stayed on CPU). Found + fixed 2026-09-03.
        self.weight_scale = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.bfloat16, device="cpu"), requires_grad=False
        )
        # process_weights_after_loading() (model_loader/utils.py) wraps any
        # module with a real quant_method in device_loading_context, which
        # moves EVERY cpu-resident parameter on that module to GPU, runs the
        # hook, then moves it back -- for us that means round-tripping the
        # whole ~12GB fp8 table through VRAM for nothing:
        # UnquantizedEmbeddingMethod.process_weights_after_loading() is a
        # documented no-op on non-CPU platforms (only acts when
        # current_platform.is_cpu()). Confirmed via live OOM 2026-09-03: this
        # transient round-trip is exactly what blew the VRAM budget during
        # loading, independent of --max-model-len/--max-num-batched-tokens/
        # --max-num-seqs (none of which affect this at all -- it happens
        # before any KV cache allocation). create_weights() already ran
        # (inside super().__init__() above), so it's safe to drop
        # quant_method now: process_weights_after_loading's per-module loop
        # gates on `isinstance(module.quant_method, QuantizeMethodBase)`,
        # and our own forward() never touches self.quant_method (it gathers
        # via direct indexing + explicit dequant, not
        # self.quant_method.embedding(...)), so this is safe to neutralize.
        self.quant_method = None
        # `params_dtype` is dual-purposed by code outside this class: the
        # base VocabParallelEmbedding/create_weights path uses it (already
        # consumed, above) to size/type self.weight -- but the OUTER caller
        # in this same file (Qwen4ExpPLELayer's forward, the code that builds
        # `output = ngram_ids.new_empty(..., dtype=self.ngram_embedding
        # .params_dtype)` before invoking the qwen4_exp_amd_ple_ngram_embedding
        # custom op) ALSO reads this same attribute, to decide what dtype the
        # FINAL (post-dequant) output buffer should be. Setting params_dtype
        # to fp8 above (for the weight) leaked into that second, unrelated
        # use: the op's `output.copy_(result)` was writing our correctly
        # dequantized bf16 `result` into an fp8-typed buffer, silently
        # re-quantizing it with an unscaled cast -- reproducing, one level
        # up, the exact corruption this whole patch exists to avoid. This
        # is NOT a key_proj/value_proj bug (their weights are plain bf16 on
        # disk and correctly excluded from AWQ conversion via the
        # checkpoint's modules_to_not_convert=["ple"]; confirmed by reading
        # the checkpoint directly) -- they were only ever receiving already-
        # corrupted input. Reset params_dtype to the true output dtype now
        # that weight creation (the only consumer that needs the fp8 value)
        # is done, so the outer caller's buffer allocation gets it right.
        # Found + fixed 2026-09-03.
        self.params_dtype = torch.bfloat16

    def _dequant(self, gathered_fp8: torch.Tensor) -> torch.Tensor:
        # Cheap on purpose: this runs on the gathered rows only (a handful
        # to a few thousand, per the module docstring's bandwidth math), not
        # the full ~52 GB table.
        return gathered_fp8.to(torch.bfloat16) * self.weight_scale

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            idx_cpu = input_.long().cpu()
            gathered = self.weight[idx_cpu]  # fp8, cheap sparse gather on CPU
            dequant = self._dequant(gathered)
            return dequant.to(input_.device, non_blocking=True)

        masked_input, input_mask = get_masked_input_and_mask(
            input_,
            self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
            self.shard_indices.num_org_vocab_padding,
            self.shard_indices.added_vocab_start_index,
            self.shard_indices.added_vocab_end_index,
        )
        # The actual gather: indices -> CPU (tiny -- at most
        # max_num_batched_tokens int64s), plain-index gather (not
        # F.embedding -- avoids depending on fp8 dtype support in that op)
        # against the host-resident raw-fp8 weight, dequantized on just the
        # gathered rows, result back to the compute device. This is the
        # entire offload: everything else below is unchanged from
        # VocabParallelEmbedding.forward()'s non-fused fallback path, minus
        # the fp8-int8-view all-reduce trick (moot -- output is bf16 here
        # after dequant, not fp8, by the time it reaches the all-reduce).
        idx_cpu = masked_input.long().cpu()
        gathered = self.weight[idx_cpu]  # fp8
        dequant_cpu = self._dequant(gathered)  # bf16, cheap
        output_parallel = dequant_cpu.to(input_.device, non_blocking=True)

        output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
        return tensor_model_parallel_all_reduce(output_parallel)

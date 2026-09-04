# Dialing in Qwen3.8-Flash-Next on 4× R9700 — GPU P2P, memory budget, concurrency, expert parallelism

Working notes from 2026-09-03/04, in the order the findings were made. Everything below was
measured on one machine (Threadripper PRO 3945WX, MC62-G40, 6× Radeon AI PRO R9700 32 GB, four of
them used for this model), so treat the numbers as this box's; the *mechanisms* should travel.
Companion to `README.md` (the patches) and `GRAPH-CAPTURE-FIX.md` (the PIECEWISE investigation).
Tooling referenced here lives in `tools/`.

## TL;DR

| What | Result |
|---|---|
| The "gfx1201 RCCL bug" that forced `NCCL_P2P_DISABLE=1 RCCL_NET=Socket` on every seat | Not RCCL. vLLM-ROCm images ≥ 0.21 bake `HSA_ENABLE_IPC_MODE_LEGACY=1` into their env. Set it to `0`. |
| BIOS ACS | Disable it. KFD then exposes a full `p2p_links` mesh; D2D copies run at PCIe wire rate. |
| RCCL collectives with real P2P | 1.7× bandwidth, 1.5–1.8× lower latency for ≥ 512 KiB messages, tie at 8 KiB. |
| Seat-level effect at concurrency 1 | ≈ 0 for this model (TP4 MoE decode is not collective-bound at batch 1). |
| Memory budget | 24.5 GiB/GPU is weights; KV is the residual. ~62.7K tokens per GiB per GPU. |
| Knobs that matter | `--gpu-memory-utilization 0.95` (0.97 OOMs under batching), `--max-num-seqs 4` (2 crashes compile), an explicit `cudagraph_capture_sizes` ladder, `--max-model-len 131072` → 272K-token pool, 2.08× |
| Expert parallelism | Works after a 4th patch (`moe_wna16.py`); **+8–10 % decode** on 4 GPUs; the only way to 8 GPUs with this AWQ-g32 checkpoint. |
| Graph capture | PIECEWISE + the QSA kernel warmup is the production mode; `--enforce-eager` retired 2026-09-04 (README, `GRAPH-CAPTURE-FIX.md`). |
| Best config so far | TP4 + EP, PIECEWISE + MTP(2), P2P env: **49.4 ± 3.2 tok/s @ 55K, 48.3 ± 3.3 @ 120K**, single stream. |

---

## 1. GPU P2P: what was actually broken

### Symptom
Every vLLM-ROCm image newer than v0.20.x failed at RCCL init on gfx1201 with

```
NCCL WARN hipIpcGetMemHandle failed : invalid argument   (transport/p2p.cc)
```

For months the working answer was `NCCL_P2P_DISABLE=1` + `RCCL_NET=Socket` (+ `NCCL_PROTO=Simple`),
i.e. every tensor-parallel all-reduce bounced through host shared memory.

### Bisection
1. **Micro-repro** (`tools/rccl.py`, 2-rank all-reduce under `torchrun`): fails on 0.28.0 and the Sept-2
   nightly, passes on the 0.20.2 image with RCCL logging `via P2P/IPC`.
2. **Swap libraries.** Copy `librccl`, `libamdhip64`, `libhsa-runtime64` from the 0.20.2 image (ROCm 7.2.1)
   into the 0.28.0 container (ROCm 7.2.3) over the *real* paths (`/opt/rocm-7.2.3/lib/...`, not the
   `/opt/rocm` symlink — the first attempt mounted over the symlink target and changed nothing). Still
   fails. Same RCCL commit (`2.27.7-HEAD:96a25b5`) in both. So: not the libraries.
3. **Diff the image env.** `docker inspect` shows the ≥ 0.21 images set `HSA_ENABLE_IPC_MODE_LEGACY=1`;
   0.20.2 does not. That variable forces ROCr's pre-dma-buf KFD-ioctl IPC path, which gfx12 rejects.
4. `-e HSA_ENABLE_IPC_MODE_LEGACY=0` → 0.28.0 and the nightly init `via P2P/IPC`, 2-rank same/cross root
   complex and 4-rank. A bare `torch.multiprocessing` CUDA-tensor share (`tools/ipc.py`) hangs with the
   image default and passes with `=0`, so the breakage is at the HIP IPC layer, not RCCL.

### BIOS ACS
Disabling ACS (Access Control Services) on the root ports is what turns "IPC works" into "P2P DMA is
actually used": the kernel's `pci_p2pdma` refuses P2P through ACS-redirecting bridges. Verify without
root via KFD's topology — every GPU node should list every other GPU under `p2p_links`, `type=2`
(PCIe), and *no* `NO_PEER_TO_PEER_DMA` flag bit:

```
for n in /sys/class/kfd/kfd/topology/nodes/*; do ls $n/p2p_links 2>/dev/null | wc -l; done
```

and confirm every card kept its large BAR (`BAR0` = full VRAM in `/sys/bus/pci/devices/*/resource`).
Both held here after the change. `iommu=pt` is on the kernel command line.

### What P2P buys at the collective level (`tools/rccl.py`, `tools/rccl_lat.py`)
All-reduce, `NCCL_PROTO=Simple`, same four GPUs the model uses:

| | SHM (old env) | P2P/IPC |
|---|---|---|
| 2-rank, 256 MiB, busbw | 11.6 GB/s | **19.7 GB/s** |
| 4-rank, 8 KiB | 71–75 µs | 76 µs (tie) |
| 4-rank, 64 KiB | 63–80 µs | 70 µs |
| 4-rank, 512 KiB | 114–115 µs | **78 µs** |
| 4-rank, 4 MiB | 600–612 µs | **343 µs** |
| 4-rank, 64 MiB | 8.96–9.15 ms | **4.91 ms** |

Knobs tried on top: `NCCL_P2P_USE_CUDA_MEMCPY=1` is **4–10× worse** at every size (don't);
`NCCL_MIN_NCHANNELS=4` shaves small messages to ~50 µs (not tested in a seat).
D2D `copy_()` between any two cards, including across root complexes: ~25 GB/s (Gen4 x16 wire rate).

### What it buys at the seat level
| Seat | old env | P2P env |
|---|---|---|
| Qwen3.8-27B-FP8 TP2 (dense) | 27.7 single / 255 agg @16 | 27.9 / 264 |
| gemma-4-26B TP2, v0.28.0 | 1753 peak / 80.7 single | 1906 / 81.1 |
| gemma-4-26B TP2, v0.20.2 (already P2P) | 1943 / 99.9 | 1932 / 99.1 |
| **Flash-Next TP4, c=1** | 47.6 (n=3, prior night) | 45.7 ± 3.6 (n=8) — parity |

Interpretation: at batch 1 the all-reduce payload is one token's hidden state, a few KB, where the
transports tie. The transport starts to matter at ≥ 512 KiB, i.e. batched decode / prefill. Note also
that vLLM's custom all-reduce and QuickReduce are gated to MI300 (`platforms/rocm.py:
use_custom_allreduce`), so every gfx1201 seat dispatches plain `PYNCCL` regardless — the
`Using ['PYNCCL'] all-reduce backends` log line confirms it.

**Keep `NCCL_PROTO=Simple`.** That one is a separate RCCL bug on gfx12 (LL-protocol first-collective
deadlock, ROCm/rccl PR #2187) and was not re-tested here.

---

## 2. Measuring this seat honestly

This model's single-stream decode has **±10–20 % run-to-run variance** at 55K context on identical,
temperature-0, prefix-cached requests. Several apparent "findings" evaporated once that was controlled:

- **256-token generations are too short.** A 5-second decode window can read 53 or 36 on the same
  seat. Use 512-token generations, ≥ 8 samples, report mean ± sd and median.
- **Any other seat on the box costs ~10 tok/s** even when idle — vLLM workers busy-poll at ~200 % CPU
  each and this model's per-token PLE gather is host-RAM-bound. Every number in this file marked
  "idle box" had no other GPU seat running.
- **Generation length itself is not the cause** (interleaved 256 vs 768 on one warm seat: 40.4 ± 1.7
  vs 42.9 ± 3.7).
- **GPU clocks/thermals are not the cause**: sampled during a run (`tools/gpusample.sh`), sclk held
  3.0–3.2 GHz *rising* across the run, junction ≤ 73 °C, up to 428 W on one card.
- **CPU governor is not the cause**: with `amd-pstate-epp` in `powersave` / `balance_performance` the
  cores sit at 4216 of 4427 MHz under load. Leave it. (Side note: `powerprofilesctl get` D-Bus-activates
  `power-profiles-daemon`, which then pins the profile — don't poke it if you don't need it.)
- **Core pinning** (`docker update --cpuset-cpus`) didn't move the number.
- MTP acceptance was 68–73 % across every run, so speculation isn't the variable either.

`tools/depthval.py` is the probe used for every "n=8" number here: prime the prefix cache, then eight
512-token generations at ~55K and eight at ~120K tokens of context.

---

## 3. Memory budget (TP4, PLE table on host RAM via `ple_cpu.py`, no `--cpu-offload-gb`)

Per GPU at gmu 0.93, seqs 4, default capture ladder, mml 65536 (the earlier config):

| | GiB |
|---|---|
| weights + non-torch | 24.49 |
| peak activation | 2.32 |
| CUDA graphs | 1.11 |
| **KV cache** | **2.82** → 176,748 tokens |

So **~62.7K tokens per GiB per GPU** of KV, and the pool is shared across concurrent requests:
176K × 1, 64K × 2.7, 32K × 5.4, 16K × 10.8, 8K × 21.6.

Levers, with what happened when each was pulled:

| Lever | Effect |
|---|---|
| `--gpu-memory-utilization 0.97` | Loads (3.32 GiB KV) but **OOMs the engine** (`CUDA out of memory … 0 bytes free`) the moment four prefills batch. vLLM's activation estimate undershoots. **0.95** ran everything cleanly. |
| `--max-num-seqs 2` | Fails the first `torch.compile` pass: `ConstraintViolationError (L['query_start_loc'].size()[0])`. Use 4. |
| explicit `cudagraph_capture_sizes` | The default ladder scales with `max_num_seqs × (MTP+1)`: at seqs 32 it went to **192 sizes and left 0 GiB for KV**. `[1,2,3,4,6,8,12]` for seqs 4 (graph memory 1.11 → 0.9 GiB); `[1,2,4,8,12,16,24,32,48,64,96]` for seqs 32. Files: `tools/compilation_config_seqs{4,32}.json`. |
| `--kv-cache-dtype fp8` | Impossible: the QSA attention raises `NotImplementedError`. BF16 KV is mandatory. |
| `--cpu-offload-gb` (generic weight offload) | Not used anywhere in this repo's configs — the PLE offload patch is what makes the model fit. Untested as a KV-headroom lever: each GiB/GPU freed ≈ +63K tokens, but it streams weights per token and is expected to hurt decode. |
| drop MTP | Frees the draft head + its KV group (the no-MTP placement ran mml 262144) at ~2.7× the decode cost. Not worth it for a personal seat. |

**Resulting production config** (gmu 0.95, mml 131072, seqs 4, ladder ≤ 12, MTP 2, P2P env):
KV **272,282 tokens = 2.08× at 131K**, graph memory 0.9 GiB, decode **45.1 ± 2.0 @ 55K / 44.8 ± 3.2 @ 120K**
(TP only). Decode is flat with depth; cold 120K TTFT ≈ 42 s.

---

## 4. Concurrency ("for science")

seqs 32, mml 16384, gmu 0.94 (81K-token pool), ~2.3K-token distinct prompts, 256-token generations,
P2P env, MTP on (`tools/sweep.py`):

| concurrency | per-stream tok/s | decode aggregate | TTFT p50 |
|---|---|---|---|
| 1 | 45.4 | 45 | 4.3 s |
| 2 | 26.1 (artifact: 2nd stream's chunked prefill inside the 1st's window; 51.7 in a repeat) | 52 | 15.8 s |
| 4 | 43.6 | **175** | 16 s |
| 8 | 33.1 | **265 (peak)** | 32 s |
| 16 | 12.2 | 195 | 41 s |
| 32 | 6.9 | 221 | 78 s |

Aggregate peaks near c=8 at ~5.8× single-stream; past that the KV pool is the wall (c=32 × ~2.5K ≈
the whole pool) and the rest is queueing. The old-env sweep was deliberately *not* run, so the
"P2P wins at batch" prediction from §1 remains a prediction at the seat level.

---

## 5. Expert parallelism

### Why
TP8 is impossible for this checkpoint: experts are 640 wide → 80 columns per rank, and the AWQ group
size is 32; the down-projection shards along that dim and `80 % 32 ≠ 0`, which vLLM rejects at load
(TP4 works only because `160 % 32 = 0`). Attention (24 heads), GDN heads (48 / 16) and hidden (2560)
would all shard fine. Expert parallelism sidesteps it: 512 experts → whole experts per rank, no
intra-expert split, MoE traffic becomes all-to-all (`all2all_backend=allgather_reducescatter` by
default — the portable path, no NVIDIA-only kernels).

### What broke
`--enable-expert-parallel` crashed in `moe_wna16_weight_loader` — the generic WNA16 path this
checkpoint already uses (README §2):

1. `IndexError: index 128 is out of bounds for dimension 0 with size 128` on rank 0 — the
   `w13_qzeros` / `w2_qzeros` fast paths write `param.data[expert_id]` with the **global** expert id;
   under EP a rank holds 128 local experts.
2. After fixing (1), `IndexError: index 1 is out of bounds for dimension 1 with size 1` on ranks ≥ 1 —
   the same branches slice the checkpoint tensor by `get_tensor_model_parallel_rank()`, but the MoE
   layer's own parallel config has `tp_size = 1, tp_rank = 0` under EP.

`moe_wna16.py` / `moe_wna16.patch` map global → local via the layer's
`_map_global_expert_id_to_local_expert_id()` (skipping experts the rank doesn't own) and slice by
`layer.moe_config.tp_rank`. The delegate loader those branches fall through to already did both —
the fast paths never got the same treatment. It is a plain upstream bug for any WNA16 MoE + EP.

### Result (same config as §3's production seat, idle box, n=8)
| | 55K | 120K | KV pool |
|---|---|---|---|
| TP4 | 45.1 ± 2.0 | 44.8 ± 3.2 | 272,282 |
| **TP4 + EP** | **49.4 ± 3.2** | **48.3 ± 3.3** | 271,558 |

+8–10 %, same KV, MTP acceptance 68 %, `17*23 → 391`. **HumanEval has not been re-run on EP** — it is
speed-validated, quality-pending, on top of the PIECEWISE drift caveat in `GRAPH-CAPTURE-FIX.md`.

### Toward 8 GPUs
With 8 cards and EP (TP8 attention + 64 whole experts per rank): the 4-bit expert weights that dominate
today's 24.5 GiB/GPU halve, so KV headroom goes from ~2.8 to roughly 12 GiB per GPU (estimate) — on the
order of a 1M-token pool at ~63K tokens/GiB. Expect a meaningful single-stream gain but not 2×; the
8-rank collectives are exactly where the P2P transport stops being a tie. Board reality on this box: the
MC62-G40 has 7 physical slots, so an 8th card needs a riser; TP needs a power of two, so 7 buys nothing
over 6. `tools/launch.sh` (EP on by default) is the launcher. Not yet attempted — 6 cards installed.

---

## 6. Open items
- Quality on the production EP + PIECEWISE + MTP seat: a full suite (HumanEval, ARC, needle, PlanBench,
  AutomationBench, …) is running against it as of 2026-09-04; results go in README § Validation.
  (`run_humaneval_c1.sh PORT=<port>` is the standalone HumanEval harness.)
- Old-env concurrency sweep, if the batch-level P2P claim is ever to be a measurement.
- Upstream `moe_wna16.patch`.
- `--cpu-offload-gb` as a KV-headroom lever (each GiB/GPU ≈ +63K tokens — at what decode cost?).
- `NCCL_MIN_NCHANNELS=4` in a real seat.
- 8 GPUs via EP (§5), once the cards are in.

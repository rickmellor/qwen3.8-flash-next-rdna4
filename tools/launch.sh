#!/bin/bash
# Launch Qwen3.8-Flash-Next-AWQ on vLLM-ROCm with this repo's patches bind-mounted.
#
#   MODEL_DIR=/path/to/Qwen3.8-Flash-Next-AWQ-g32 tools/launch.sh [name] [port] [gpus]
#
# Knobs (env):  GMU (0.95)  MML (131072)  SEQS (4)  EP (0|1, default 0)  MTP (2; 0 = off)
#               LADDER (json list; default tools/compilation_config_seqs4.json's)
#               IMAGE (the Sept-2 2026 nightly)  VLLM_CACHE (host dir for the torch.compile cache)
# The env below is the P2P env: HSA_ENABLE_IPC_MODE_LEGACY=0 (NOT NCCL_P2P_DISABLE/RCCL_NET) — see TUNING.md.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
NAME=${1:-flashnext}; PORT=${2:-8011}; GPUS=${3:-0,1,2,3}
: "${MODEL_DIR:?set MODEL_DIR to the checkpoint directory}"
IMAGE=${IMAGE:-vllm/vllm-openai-rocm:nightly-27a94d1ce4e3fc100c4732439ccec10f8246a804}
V=/usr/local/lib/python3.12/dist-packages/vllm
GMU=${GMU:-0.95}; MML=${MML:-131072}; SEQS=${SEQS:-4}; EP=${EP:-0}; MTP=${MTP:-2}
CC=$(python3 - "$REPO" "${LADDER:-}" <<'PY'
import json,sys
cc=json.load(open(f"{sys.argv[1]}/tools/compilation_config_seqs4.json"))
if sys.argv[2]: cc["cudagraph_capture_sizes"]=json.loads(sys.argv[2])
print(json.dumps(cc))
PY
)
EXTRA=()
[ "$EP" = "1" ] && EXTRA+=(--enable-expert-parallel)
[ "$MTP" != "0" ] && EXTRA+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}")
CACHE=(); [ -n "${VLLM_CACHE:-}" ] && CACHE=(-v "$VLLM_CACHE:/root/.cache/vllm")
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --ipc host --shm-size 16g --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES="$GPUS" \
  -e VLLM_QWEN4EXP_PLE_CPU_OFFLOAD=1 \
  -e HSA_ENABLE_IPC_MODE_LEGACY=0 -e NCCL_PROTO=Simple \
  -e HIP_FORCE_DEV_KERNARG=1 -e SAFETENSORS_FAST_GPU=1 \
  -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT \
  -v "$REPO/ple_layer.py:$V/models/qwen4_exp/amd/ple_layer.py:ro" \
  -v "$REPO/ple_cpu.py:$V/models/qwen4_exp/common/ple_cpu.py:ro" \
  -v "$REPO/routed_experts.py:$V/model_executor/layers/fused_moe/routed_experts.py:ro" \
  -v "$REPO/moe_wna16.py:$V/model_executor/layers/quantization/moe_wna16.py:ro" \
  -v "$REPO/cudagraph_utils.py:$V/v1/worker/gpu/cudagraph_utils.py:ro" \
  -v "$REPO/kernel_warmup.py:$V/model_executor/warmup/kernel_warmup.py:ro" \
  -v "$REPO/qwen4_exp_amd_qsa_warmup.py:$V/model_executor/warmup/qwen4_exp_amd_qsa_warmup.py:ro" \
  -v "$MODEL_DIR:/model:ro" "${CACHE[@]}" \
  -p "$PORT:8000" \
  "$IMAGE" \
  --model /model --served-model-name Qwen3.8-Flash-Next-AWQ \
  --tensor-parallel-size 4 --gpu-memory-utilization "$GMU" \
  --max-model-len "$MML" --max-num-seqs "$SEQS" --max-num-batched-tokens 8192 \
  --kv-cache-dtype auto --quantization awq_marlin --enable-prefix-caching \
  --compilation-config "$CC" \
  "${EXTRA[@]}" \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 --trust-remote-code
echo "launched $NAME on :$PORT (gpus $GPUS, gmu $GMU, mml $MML, seqs $SEQS, ep $EP, mtp $MTP)"
echo "  ready when:   curl -sf localhost:$PORT/v1/models"
echo "  transport:    docker logs $NAME 2>&1 | grep -m1 -oE 'via (P2P/IPC|SHM/direct)'"
echo "  KV pool:      docker logs $NAME 2>&1 | grep -oE 'GPU KV cache size: .*'"

#!/bin/bash
# Numerics study driver: launch each configuration with tools/launch.sh, probe it, tear it down.
#   MODEL_DIR=/path/to/checkpoint tools/drift/run_variants.sh <outdir>
# Then: python3 tools/drift/compare.py <outdir>/L0-eager.a.json <outdir>/<variant>.a.json
# Position-0 (first generated token) statistics are the ones that isolate kernel numerics
# from graph replay; MAX_TOKENS=8 is enough for those, 256 also gives divergence indices.
set -u
OUT=${1:?outdir}; mkdir -p "$OUT"; REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
: "${MODEL_DIR:?}"; export MAX_TOKENS=${MAX_TOKENS:-8}
run(){ name=$1; shift
  echo "=== $(date +%T) $name ($*)"
  env "$@" MTP=0 tools/launch.sh fn-probe 8011 "${GPUS:-0,1,2,3}" > "$OUT/$name.launch.log" 2>&1 || { echo "launch failed"; return; }
  for i in $(seq 1 180); do curl -sf -m 2 localhost:8011/v1/models >/dev/null 2>&1 && break
    docker ps -q -f name=fn-probe | grep -q . || { echo "container died"; docker logs --tail 300 fn-probe > "$OUT/$name.died.log" 2>&1; return; }; sleep 10; done
  python3 tools/drift/probe.py 8011 "$OUT/$name.a.json" > "$OUT/$name.probe.log" 2>&1 && echo "probe ok"
  [ "${REPEAT:-0}" = 1 ] && python3 tools/drift/probe.py 8011 "$OUT/$name.b.json" >> "$OUT/$name.probe.log" 2>&1
  docker rm -f fn-probe >/dev/null 2>&1
}
REPEAT=1 run L0-eager            EAGER=1
REPEAT=1 run L1-piecewise        EAGER=0
run L4-native-rope               EAGER=0 CUSTOM_OPS=none,+rotary_embedding,+apply_rotary_emb,+hpc_rope_norm
run L5-native-act                EAGER=0 CUSTOM_OPS=none,+silu_and_mul,+mul_and_silu
# (CUSTOM_OPS=all / +rms_norm under compile do not launch on this build: Dynamo won't trace the ROCm RMSNorm wrapper)
run E1-eager-nativeops           EAGER=1 CUSTOM_OPS=none
run E2-eager-minus-norms         EAGER=1 CUSTOM_OPS=all,-rms_norm,-gemma_rms_norm,-rms_norm_gated,-fused_rms_norm_gated
run E3-eager-minus-rope          EAGER=1 CUSTOM_OPS=all,-rotary_embedding,-apply_rotary_emb,-hpc_rope_norm
run E4-eager-minus-act           EAGER=1 CUSTOM_OPS=all,-silu_and_mul,-mul_and_silu
echo done

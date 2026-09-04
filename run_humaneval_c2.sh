#!/bin/bash
# HumanEval pass@1 via lm-eval at num_concurrent=2; re-score the samples with the chat-aware scorer afterwards.
#   PORT=8011 OUT=./humaneval_c2 LMEVAL_PY=<python with lm_eval> run_humaneval_c2.sh
PORT=${PORT:-8011}; OUT=${OUT:-./humaneval_c2}
LMEVAL_PY=${LMEVAL_PY:-python3}   # a python with lm-eval installed
HF_ALLOW_CODE_EVAL=1 "$LMEVAL_PY" -m lm_eval run \
  --model local-chat-completions \
  --model_args "base_url=http://127.0.0.1:$PORT/v1/chat/completions,model=Qwen3.8-Flash-Next-AWQ,num_concurrent=2,max_retries=3,tokenized_requests=False,timeout=600" \
  --tasks humaneval --apply_chat_template --log_samples \
  --output_path "$OUT" --confirm_run_unsafe_code \
  --gen_kwargs '{"max_gen_toks": 2048, "until": [], "chat_template_kwargs": {"enable_thinking": false}}'

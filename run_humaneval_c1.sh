#!/bin/bash
HF_ALLOW_CODE_EVAL=1 /home/rick/.local/share/pipx/venvs/johnny-fleet/bin/python3 -m lm_eval run \
  --model local-chat-completions \
  --model_args "base_url=http://127.0.0.1:8009/v1/chat/completions,model=Qwen3.8-Flash-Next-AWQ,num_concurrent=1,max_retries=3,tokenized_requests=False,timeout=600" \
  --tasks humaneval --apply_chat_template --log_samples \
  --output_path /home/rick/scratch/flashnext/humaneval_piecewise_c1 --confirm_run_unsafe_code \
  --gen_kwargs '{"max_gen_toks": 2048, "until": [], "chat_template_kwargs": {"enable_thinking": false}}'

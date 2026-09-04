#!/usr/bin/env python3
"""Single-stream decode-speed probe with context-depth warmup.

Streams chat completions at increasing context depths, separating TTFT from
steady-state decode rate. Repeats each depth until the rate levels out
(prefix cache warm, kernels compiled). Prints one line per run.
"""
import json
import sys
import time
import urllib.request

import os
BASE = f"http://127.0.0.1:{os.environ.get('PORT', '8011')}/v1/chat/completions"
MODEL = os.environ.get("MODEL", "Qwen3.8-Flash-Next-AWQ")

# ~4 chars/token; one paragraph ~60 tokens. Vary content to defeat trivial compression.
PARA = (
    "Sensor node %d reports bus voltage %d.%02d V, load current %d mA, ambient "
    "%d.%d C, RSSI -%d dBm, uptime %d s, firmware rev %d.%d.%d, last fault code 0x%04x. "
)


def make_prompt(target_tokens):
    parts = []
    i = 0
    while len("".join(parts)) < target_tokens * 4:
        i += 1
        parts.append(
            PARA
            % (i, 11 + i % 3, i * 7 % 100, 120 + i * 13 % 800, 21 + i % 9,
               i % 10, 40 + i % 50, i * 37, 1 + i % 4, i % 10, i % 20, i * 2654435761 % 65536)
        )
    return "".join(parts)


def run(prompt_tokens, gen_tokens, tag):
    body = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": make_prompt(prompt_tokens)
                + "\n\nSummarize the overall health trends in this telemetry log in detail.",
            }
        ],
        "max_tokens": gen_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.time()
    tfirst = tlast = None
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            raw = raw.strip()
            if not raw.startswith(b"data: ") or raw == b"data: [DONE]":
                continue
            d = json.loads(raw[6:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices") and (d["choices"][0].get("delta") or {}).get("content"):
                now = time.time()
                if tfirst is None:
                    tfirst = now
                tlast = now
    ct = usage["completion_tokens"] if usage else -1
    pt = usage["prompt_tokens"] if usage else -1
    ttft = tfirst - t0 if tfirst else -1
    decode = (ct - 1) / (tlast - tfirst) if tfirst and tlast and tlast > tfirst else -1
    print(
        f"{tag:28s} prompt={pt:6d} gen={ct:4d} ttft={ttft:7.1f}s decode={decode:6.2f} tok/s",
        flush=True,
    )
    return decode


def main():
    import statistics
    run(50, 64, "warmup-tiny")
    run(1000, 256, "d1k-run1")
    run(31000, 256, "d55k-prime")          # populate prefix cache
    d1=[run(1000, 768, f"d1k-{n}") for n in range(1, 4)]
    d55=[run(31000, 768, f"d55k-{n}") for n in range(1, 6)]
    print(f"SUMMARY d1k  mean={statistics.mean(d1):.1f} sd={statistics.pstdev(d1):.1f}  runs={[round(x,1) for x in d1]}")
    print(f"SUMMARY d55k mean={statistics.mean(d55):.1f} sd={statistics.pstdev(d55):.1f}  runs={[round(x,1) for x in d55]}")


if __name__ == "__main__":
    main()


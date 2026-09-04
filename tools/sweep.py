#!/usr/bin/env python3
"""Concurrency sweep: c distinct ~2K-token prompts in flight, 256 gen tokens each.
Reports aggregate tok/s (sum completion tokens / wall) and per-stream decode."""
import concurrent.futures as cf, json, os, statistics as st, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depthprobe import make_prompt
PORT = os.environ.get("PORT", "8011"); MODEL = "Qwen3.8-Flash-Next-AWQ"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

def one(i, ptoks=2000, gen=256):
    body = {"model": MODEL, "max_tokens": gen, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": f"[stream {i}] " + make_prompt(ptoks) +
                          "\n\nSummarize the overall health trends in this telemetry log in detail."}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); tfirst = tlast = None; usage = None
    try:
      with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            raw = raw.strip()
            if not raw.startswith(b"data: ") or raw == b"data: [DONE]": continue
            d = json.loads(raw[6:])
            if d.get("usage"): usage = d["usage"]
            if d.get("choices") and (d["choices"][0].get("delta") or {}).get("content"):
                now = time.time(); tfirst = tfirst or now; tlast = now
    except Exception as e:
        print(f"   stream {i} FAILED: {str(e)[:80]}", flush=True); return 0, -1, 0
    ct = usage["completion_tokens"] if usage else 0
    return ct, (tfirst - t0) if tfirst else -1, (ct - 1) / (tlast - tfirst) if tfirst and tlast > tfirst else 0

one(0, 500, 32)  # warm
print(f"{'conc':>4} {'decode agg':>10} {'per-stream':>11} {'ttft p50':>9} {'wall':>6}  (decode agg = per-stream x streams that finished)")
for c in (1, 2, 4, 8, 16, 32):
    t0 = time.time()
    with cf.ThreadPoolExecutor(c) as ex:
        rs = list(ex.map(lambda i: one(1000 * c + i), range(c)))
    wall = time.time() - t0
    ok = [r for r in rs if r[0] > 0]
    if not ok: print(f"{c:>4}  all {c} streams failed", flush=True); break
    ttft = st.median(r[1] for r in ok); dec = st.median(r[2] for r in ok)
    print(f"{c:>4} {dec * len(ok):>10.1f} {dec:>11.1f} {ttft:>8.1f}s {wall:>5.0f}s  ({len(ok)}/{c} ok)", flush=True)

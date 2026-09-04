"""Reproduce the 2026-09-04 OOM batch: mixed-length concurrent prefills on the Flash-Next seat.
Usage: python stress_prefill.py [port]  — prints per-scenario wall time and any non-200s."""
import sys, time, json, concurrent.futures as cf, urllib.request
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8002
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MODEL = "Qwen3.8-Flash-Next-AWQ"
from tokenizers import Tokenizer
import os
tk = Tokenizer.from_file(os.path.join(os.environ["MODEL_DIR"], "tokenizer.json"))  # MODEL_DIR = checkpoint dir
import random
random.seed(7)
WORDS = open("/usr/share/dict/words").read().split() if __import__("os").path.exists("/usr/share/dict/words") else [f"w{i}" for i in range(5000)]
def prompt(ntok, tag):
    # random words defeat prefix caching so every request really prefills
    words = [random.choice(WORDS) for _ in range(ntok)]
    txt = " ".join(words)
    ids = tk.encode(txt).ids
    while len(ids) > ntok:
        txt = txt[: int(len(txt) * ntok / len(ids)) - 10]; ids = tk.encode(txt).ids
    return f"[{tag}] Summarize the following word list in one sentence.\n{txt}"
def call(ntok, tag):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt(ntok, tag)}],
                       "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(URL, body, {"content-type": "application/json"}), timeout=900)
        u = json.load(r)["usage"]; return (200, u["prompt_tokens"], round(time.time() - t, 1))
    except Exception as e:
        return (getattr(e, "code", str(e)[:60]), None, round(time.time() - t, 1))
SCEN = [("1x16K+3xshort", [16000, 300, 200, 100]), ("1x16K+3xshort", [16000, 300, 200, 100]),
        ("4x16K", [16000] * 4), ("1x32K+3x2K", [32000, 2000, 2000, 2000]), ("4x32K", [32000] * 4)]
for name, sizes in SCEN:
    t = time.time()
    with cf.ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda a: call(*a), [(n, f"{name}-{i}") for i, n in enumerate(sizes)]))
    print(f"{name:14s} wall={time.time()-t:6.1f}s  results={res}", flush=True)
    if any(r[0] != 200 for r in res): print("NON-200 — stop"); break
print("DONE")

"""Greedy top-5 logprob probe for the numerics study (README "Where the drift comes from").
Usage: MAX_TOKENS=256 probe.py <port> <out.json>. Raw /v1/completions, no chat template, temperature 0.
The seven HumanEval prompts are the problems this model fails (humaneval_fail_docs.json)."""
import sys, json, urllib.request
PORT, OUT = int(sys.argv[1]), sys.argv[2]
URL = f"http://127.0.0.1:{PORT}/v1/completions"
he = json.load(open(__file__.rsplit('/',1)[0] + '/humaneval_fail_docs.json'))
prompts = [(d['entry_point'], d['prompt']) for d in he]
prompts += [
 ("prose_history", "The history of the Roman Republic, from its founding to the rise of Augustus, can be summarized as follows:"),
 ("prose_science", "Explain, in plain language, why the sky is blue and sunsets are red.\n\n"),
 ("math_word", "Q: A train leaves at 3:15 pm travelling 72 km/h. A second train leaves the same station at 4:00 pm at 96 km/h on the same track. At what time does the second train catch the first? Show the working.\nA:"),
 ("code_py_sort", "# Python: implement merge sort on a list of integers, with a docstring and type hints.\n"),
 ("code_rust", "// Rust: a function that parses \"HH:MM\" into minutes since midnight, returning Result<u32, String>.\n"),
 ("json_gen", "Produce a JSON object describing a fictional 1987 sports car: make, model, engine, power_hp, weight_kg, zero_to_100_s.\n"),
 ("list_gen", "List twelve distinct uses for a paperclip, one per line:\n1."),
 ("translate", "Translate to French: 'The workshop was closed for the holiday, so we spent the afternoon repairing the old radio instead.'\n"),
 ("reasoning", "Three boxes are labelled Apples, Oranges, Mixed; every label is wrong. You may draw one fruit from one box. How do you relabel all three correctly? Reason step by step."),
 ("summarize", "Summarize in two sentences: PCIe Access Control Services (ACS) can force peer-to-peer traffic between devices under the same root complex to be routed through the root port for isolation. On systems where GPUs communicate directly, disabling ACS lets the fabric route device-to-device transfers without the detour, improving bandwidth and latency.\n\nSummary:"),
 ("sql", "-- SQL: for a table orders(id, customer_id, amount, created_at), return each customer's total spend in 2025 and rank them.\n"),
 ("poem", "A short poem about a Threadripper waking up at dawn:\n"),
 ("repeat_long", "Continue the sequence of prime numbers starting from 2, separated by commas: 2, 3, 5, 7,"),
]
res = []
for name, p in prompts:
    body = json.dumps({"model": "Qwen3.8-Flash-Next-AWQ", "prompt": p, "max_tokens": int(__import__("os").environ.get("MAX_TOKENS","256")), "temperature": 0,
                       "logprobs": 5, "seed": 0}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, body, {"content-type": "application/json"}), timeout=900))
    ch = r["choices"][0]; lp = ch["logprobs"]
    res.append({"name": name, "prompt_tokens": r["usage"]["prompt_tokens"], "text": ch["text"],
                "tokens": lp["tokens"], "token_logprobs": lp["token_logprobs"], "top": lp["top_logprobs"],
                "finish": ch["finish_reason"]})
    print(f"  {name:16s} {len(lp['tokens'])} tok", flush=True)
json.dump(res, open(OUT, "w"))
print("saved", OUT)

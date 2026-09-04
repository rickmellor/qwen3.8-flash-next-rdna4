#!/usr/bin/env python3
"""Compare two lm-eval HumanEval samples files generation-by-generation.

Usage: analyze_divergence.py <samples_a.jsonl> <samples_b.jsonl>

Reports which task generations diverge, where the first differing character
falls, and whether trajectories reconverge after the split -- distinguishes
numerical-drift token flips (mid-generation divergence, coherent
continuations, frequent reconvergence) from systematic corruption.
"""
import difflib
import json
import sys


def load(path):
    out = {}
    for line in open(path):
        d = json.loads(line)
        out[d["doc"]["task_id"]] = d["resps"][0][0]
    return out


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    ids = [t for t in a if t in b]
    diff = [t for t in ids if a[t] != b[t]]
    print(f"problems compared: {len(ids)}")
    print(f"byte-identical:    {len(ids) - len(diff)}")
    print(f"divergent:         {len(diff)}")
    reconverge = 0
    for t in sorted(diff, key=lambda t: int(t.split("/")[1])):
        x, y = a[t], b[t]
        i = next(
            (k for k in range(min(len(x), len(y))) if x[k] != y[k]),
            min(len(x), len(y)),
        )
        r = difflib.SequenceMatcher(None, x[i:], y[i:]).ratio()
        reconverge += r > 0.5
        print(
            f"  {t:16s} first-diff @ {i:5d}/{len(x):5d}"
            f" ({100 * i / max(len(x), 1):5.1f}%)  post-div similarity {r:.2f}"
        )
    if diff:
        print(f"reconverging (similarity > 0.5): {reconverge}/{len(diff)}")


if __name__ == "__main__":
    main()

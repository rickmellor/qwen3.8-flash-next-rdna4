"""compare.py <ref.json> <other.json> [label] — divergence of `other` from `ref` (both from probe.py)."""
import sys, json, statistics as st
A = {r["name"]: r for r in json.load(open(sys.argv[1]))}; B = {r["name"]: r for r in json.load(open(sys.argv[2]))}
label = sys.argv[3] if len(sys.argv) > 3 else sys.argv[2]
ident = 0; firstdiv = []; p0_top1 = 0; p0_maxd = []; prefix_meand = []
rows = []
for n, a in A.items():
    b = B[n]
    ta, tb = a["tokens"], b["tokens"]
    L = min(len(ta), len(tb)); d = next((i for i in range(L) if ta[i] != tb[i]), None)
    if d is None and len(ta) == len(tb): ident += 1; d_str = "same"
    else: d = L if d is None else d; firstdiv.append(d); d_str = str(d)
    # position 0 = pure prefill numerics: compare the top-5 distributions
    t0a, t0b = a["top"][0], b["top"][0]
    top1a = max(t0a, key=t0a.get); top1b = max(t0b, key=t0b.get)
    p0_top1 += (top1a == top1b)
    common = set(t0a) & set(t0b)
    md = max((abs(t0a[k] - t0b[k]) for k in common), default=float("nan")); p0_maxd.append(md)
    # mean |Δ logprob| of the chosen token over the shared prefix (before divergence)
    k = d if d_str != "same" else L
    diffs = [abs(a["token_logprobs"][i] - b["token_logprobs"][i]) for i in range(k) if a["token_logprobs"][i] is not None and b["token_logprobs"][i] is not None]
    prefix_meand.append(st.mean(diffs) if diffs else 0.0)
    rows.append((n, d_str, f"{md:.2e}", f"{prefix_meand[-1]:.2e}"))
print(f"== {label}: identical {ident}/{len(A)} · first divergence median {st.median(firstdiv) if firstdiv else '-'} (min {min(firstdiv) if firstdiv else '-'}) · "
      f"pos0 top-1 agree {p0_top1}/{len(A)} · pos0 top-5 max|Δlogprob| median {st.median(p0_maxd):.2e} max {max(p0_maxd):.2e} · "
      f"prefix mean|Δ| median {st.median(prefix_meand):.2e}")
for r in rows: print("   %-16s div=%-5s pos0maxΔ=%s prefixΔ=%s" % r)

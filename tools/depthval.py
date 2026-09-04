import os,sys,statistics as st
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from depthprobe import run
run(31000,256,"prime-55k")
a=[run(31000,512,f"d55k-{n}") for n in range(1,9)]
print(f"SUMMARY 55K  n=8 mean={st.mean(a):.1f} sd={st.pstdev(a):.1f} median={st.median(a):.1f} min={min(a):.1f} max={max(a):.1f}",flush=True)
run(68000,256,"prime-120k")
b=[run(68000,512,f"d120k-{n}") for n in range(1,9)]
print(f"SUMMARY 120K n=8 mean={st.mean(b):.1f} sd={st.pstdev(b):.1f} median={st.median(b):.1f} min={min(b):.1f} max={max(b):.1f}",flush=True)

import torch, time
n = torch.cuda.device_count()
print("devices:", n)
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} {p.name} bus={getattr(p,'pci_bus_id','?')} vram={p.total_memory>>30}GiB")
print("\npeer-access matrix (row can access col):")
for i in range(n):
    print(f"  {i}: " + ' '.join('-' if i==j else str(int(torch.cuda.can_device_access_peer(i,j))) for j in range(n)))
def bw(src, dst, mb=256, iters=20):
    a = torch.ones(mb<<20, dtype=torch.uint8, device=f'cuda:{src}')
    b = torch.empty_like(a, device=f'cuda:{dst}')
    b.copy_(a); torch.cuda.synchronize(src); torch.cuda.synchronize(dst)
    t = time.perf_counter()
    for _ in range(iters): b.copy_(a, non_blocking=True)
    torch.cuda.synchronize(dst); torch.cuda.synchronize(src)
    dt = time.perf_counter() - t
    ok = bool((b[::4096]==1).all())
    del a, b; torch.cuda.empty_cache()
    return mb*iters/1024/dt, ok
print("\nD2D copy bandwidth (256 MiB x20):")
for s,d in [(0,1),(1,0),(0,3),(0,5),(2,4),(3,5)]:
    if s<n and d<n:
        g,ok = bw(s,d); print(f"  cuda:{s} -> cuda:{d}: {g:6.2f} GB/s  verify={'ok' if ok else 'FAIL'}")

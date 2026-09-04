import os, time, torch, torch.distributed as dist
dist.init_process_group('nccl'); r=dist.get_rank(); w=dist.get_world_size(); torch.cuda.set_device(r)
out=[]
for kb in (8, 64, 512, 4096, 65536):
    x=torch.ones(kb*256, dtype=torch.float32, device='cuda')  # kb KiB
    for _ in range(20): dist.all_reduce(x)
    torch.cuda.synchronize(); dist.barrier()
    n=200 if kb<=4096 else 40
    t=time.perf_counter()
    for _ in range(n): dist.all_reduce(x)
    torch.cuda.synchronize(); dt=(time.perf_counter()-t)/n
    out.append(f"{kb:>6}KiB {dt*1e6:8.1f} us")
if r==0: print("RESULT " + " | ".join(out), flush=True)
dist.destroy_process_group()

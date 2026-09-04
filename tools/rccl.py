import os, time, torch, torch.distributed as dist
dist.init_process_group('nccl'); r=dist.get_rank(); w=dist.get_world_size(); torch.cuda.set_device(r)
x=torch.ones(64<<20, dtype=torch.float32, device='cuda')  # 256 MiB
dist.all_reduce(x); torch.cuda.synchronize()
assert x[0].item()==w, x[0].item()
dist.barrier(); t=time.perf_counter()
for _ in range(10): dist.all_reduce(x)
torch.cuda.synchronize(); dt=time.perf_counter()-t
sz=x.numel()*4; alg=sz*10/dt/1e9; bus=alg*2*(w-1)/w
if r==0: print(f"RESULT world={w} allreduce 256MiB x10 OK  algbw={alg:.1f} GB/s  busbw={bus:.1f} GB/s", flush=True)
dist.destroy_process_group()

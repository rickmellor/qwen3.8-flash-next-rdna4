import torch, torch.multiprocessing as mp
def child(q):
    t = q.get(); torch.cuda.synchronize(); print("  child got tensor on", t.device, "sum=", int(t.sum().item()), flush=True)
if __name__ == "__main__":
    mp.set_start_method("spawn")
    q = mp.Queue(); p = mp.Process(target=child, args=(q,)); p.start()
    t = torch.ones(1<<20, device="cuda:0")
    try: q.put(t); p.join(60); print("  parent: child exit", p.exitcode, "-> IPC", "OK" if p.exitcode==0 else "FAIL")
    except Exception as e: print("  parent: IPC FAIL:", repr(e)[:200]); p.kill()

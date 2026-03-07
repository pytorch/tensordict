#!/usr/bin/env python3
"""Minimal NCCL P2P test to verify send/recv works."""

import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    print(f"[rank {rank}] started, world_size={world_size}", flush=True)

    dist.barrier()
    print(f"[rank {rank}] past barrier 1", flush=True)

    if rank == 0:
        t = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        print(f"[rank {rank}] sending tensor {t}", flush=True)
        dist.send(t, dst=1)
        print(f"[rank {rank}] sent!", flush=True)
    elif rank == 1:
        t = torch.empty(3, device="cuda")
        print(f"[rank {rank}] receiving...", flush=True)
        dist.recv(t, src=0)
        print(f"[rank {rank}] received: {t}", flush=True)
        assert torch.equal(t, torch.tensor([1.0, 2.0, 3.0], device="cuda"))
        print(f"[rank {rank}] PASSED!", flush=True)

    dist.barrier()
    print(f"[rank {rank}] past barrier 2", flush=True)

    if rank == 0:
        print("[rank 0] ALL PASSED", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

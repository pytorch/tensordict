#!/usr/bin/env python3
"""Minimal NCCL P2P test to verify send/recv works."""

import json

import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    print(f"[rank {rank}] started, world_size={world_size}", flush=True)
    dist.barrier()

    # Test 1: raw tensor send/recv
    print(f"[rank {rank}] Test 1: raw tensor P2P", flush=True)
    if rank == 0:
        t = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        dist.send(t, dst=1)
        print(f"[rank {rank}] sent tensor", flush=True)
    elif rank == 1:
        t = torch.empty(3, device="cuda")
        dist.recv(t, src=0)
        assert torch.equal(t, torch.tensor([1.0, 2.0, 3.0], device="cuda"))
        print(f"[rank {rank}] Test 1 PASSED!", flush=True)
    dist.barrier()

    # Test 2: send JSON metadata as CUDA byte tensor (no tags)
    print(f"[rank {rank}] Test 2: JSON metadata via CUDA", flush=True)
    if rank == 0:
        obj = {"key": "value", "shape": [10, 20], "is_dtensor": True}
        data = json.dumps(obj).encode("utf-8")
        length_t = torch.tensor([len(data)], dtype=torch.int64, device="cuda")
        dist.send(length_t, dst=1)
        data_t = torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda()
        dist.send(data_t, dst=1)
        print(f"[rank {rank}] sent metadata", flush=True)
    elif rank == 1:
        length_t = torch.empty(1, dtype=torch.int64, device="cuda")
        dist.recv(length_t, src=0)
        length = int(length_t.item())
        data_t = torch.empty(length, dtype=torch.uint8, device="cuda")
        dist.recv(data_t, src=0)
        obj = json.loads(bytes(data_t.cpu().numpy()))
        assert obj == {"key": "value", "shape": [10, 20], "is_dtensor": True}
        print(f"[rank {rank}] Test 2 PASSED! Got: {obj}", flush=True)
    dist.barrier()

    # Test 3: metadata + tensor (simulating Strategy A)
    print(f"[rank {rank}] Test 3: metadata + tensor (Strategy A sim)", flush=True)
    if rank == 0:
        meta = {"a": {"shape": [4], "dtype": "float32"}}
        data = json.dumps(meta).encode("utf-8")
        length_t = torch.tensor([len(data)], dtype=torch.int64, device="cuda")
        dist.send(length_t, dst=1)
        data_t = torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda()
        dist.send(data_t, dst=1)
        tensor = torch.tensor([10.0, 20.0, 30.0, 40.0], device="cuda")
        dist.send(tensor, dst=1)
        print(f"[rank {rank}] sent meta + tensor", flush=True)
    elif rank == 1:
        length_t = torch.empty(1, dtype=torch.int64, device="cuda")
        dist.recv(length_t, src=0)
        length = int(length_t.item())
        data_t = torch.empty(length, dtype=torch.uint8, device="cuda")
        dist.recv(data_t, src=0)
        meta = json.loads(bytes(data_t.cpu().numpy()))
        buf = torch.empty(4, device="cuda")
        dist.recv(buf, src=0)
        assert torch.equal(buf, torch.tensor([10.0, 20.0, 30.0, 40.0], device="cuda"))
        print(f"[rank {rank}] Test 3 PASSED! meta={meta}, tensor={buf}", flush=True)
    dist.barrier()

    if rank == 0:
        print("[rank 0] ALL TESTS PASSED!", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Distributed TensorDict benchmark: leaf-by-leaf vs consolidated transport.

Benchmarks use NCCL backend with CUDA tensors, so requires GPU nodes.

Usage (2-node cluster via torchrun):
    # On node 0:
    torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
        --master_addr=$MASTER_ADDR --master_port=29500 bench_distributed.py
    # On node 1:
    torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
        --master_addr=$MASTER_ADDR --master_port=29500 bench_distributed.py
"""

import os
import time

import torch
import torch.distributed as dist

from tensordict import TensorDict


def make_td(num_tensors, tensor_size=1024, dtype=torch.float32, device="cuda"):
    """Create a TensorDict with `num_tensors` tensors of `tensor_size` elements."""
    d = {
        f"t{i}": torch.randn(tensor_size, dtype=dtype, device=device)
        for i in range(num_tensors)
    }
    return TensorDict(d, batch_size=[], device=device)


def bench_leaf_send_recv(td, n_iters, rank, warmup=3):
    """Benchmark leaf-by-leaf send/recv."""
    for _ in range(warmup):
        if rank == 0:
            td.send(dst=1)
        else:
            td.recv(src=0)

    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        if rank == 0:
            td.send(dst=1)
        else:
            td.recv(src=0)
    torch.cuda.synchronize()
    dist.barrier()
    return (time.perf_counter() - t0) / n_iters


def bench_consolidated_send_recv(td_sender, td_receiver, n_iters, rank, warmup=3):
    """Benchmark consolidated send/recv (steady-state)."""
    for _ in range(warmup):
        if rank == 0:
            td_sender.send(dst=1, consolidated=True)
        else:
            td_receiver.recv(src=0, consolidated=True)

    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        if rank == 0:
            td_sender.send(dst=1, consolidated=True)
        else:
            td_receiver.recv(src=0, consolidated=True)
    torch.cuda.synchronize()
    dist.barrier()
    return (time.perf_counter() - t0) / n_iters


def bench_broadcast(td, n_iters, rank, warmup=3):
    """Benchmark broadcast from rank 0."""
    for _ in range(warmup):
        td.broadcast(src=0)

    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        td.broadcast(src=0)
    torch.cuda.synchronize()
    dist.barrier()
    return (time.perf_counter() - t0) / n_iters


def bench_all_reduce(td, n_iters, rank, warmup=3):
    """Benchmark all_reduce."""
    for _ in range(warmup):
        td.all_reduce()

    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        td.all_reduce()
    torch.cuda.synchronize()
    dist.barrier()
    return (time.perf_counter() - t0) / n_iters


def bench_init_remote(td, n_iters, rank, warmup=3):
    """Benchmark init_remote / from_remote_init (uses broadcast internally)."""
    for _ in range(warmup):
        if rank == 0:
            td.init_remote(dst=1)
        else:
            TensorDict.from_remote_init(src=0, device=td.device)

    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        if rank == 0:
            td.init_remote(dst=1)
        else:
            TensorDict.from_remote_init(src=0, device=td.device)
    torch.cuda.synchronize()
    dist.barrier()
    return (time.perf_counter() - t0) / n_iters


def total_bytes(td):
    """Total bytes in all leaf tensors."""
    total = 0
    for v in td.values(True, True):
        if isinstance(v, torch.Tensor):
            total += v.numel() * v.element_size()
    return total


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    configs = [
        (10, 1024),
        (50, 1024),
        (100, 1024),
        (500, 1024),
        (10, 1024 * 1024),
        (50, 1024 * 1024),
    ]
    n_iters = 20

    if rank == 0:
        print(f"Backend: nccl | Device: cuda:{local_rank}")
        print(
            f"{'num_tensors':>12} {'tensor_size':>12} {'total_MB':>10} "
            f"{'leaf_ms':>10} {'consol_ms':>10} {'speedup':>8} "
            f"{'bcast_ms':>10} {'allred_ms':>10} {'initrem_ms':>10}"
        )
        print("-" * 112)

    for num_tensors, tensor_size in configs:
        td = make_td(num_tensors, tensor_size, device=f"cuda:{local_rank}")
        nbytes = total_bytes(td)
        mb = nbytes / 1e6

        td_recv_leaf = make_td(num_tensors, tensor_size, device=f"cuda:{local_rank}")
        td_recv_leaf.zero_()
        leaf_time = bench_leaf_send_recv(
            td if rank == 0 else td_recv_leaf, n_iters, rank
        )

        # Consolidated send/recv: setup phase via broadcast
        if rank == 0:
            td.init_remote(dst=1)
            td_c = td.consolidate(metadata=True)
        else:
            td_c = TensorDict.from_remote_init(src=0, device=f"cuda:{local_rank}")

        consol_time = bench_consolidated_send_recv(
            td_c if rank == 0 else None,
            td_c if rank == 1 else None,
            n_iters,
            rank,
        )

        bcast_time = bench_broadcast(
            td if rank == 0 else TensorDict({}, device=f"cuda:{local_rank}"),
            n_iters,
            rank,
        )
        allred_time = bench_all_reduce(td.clone(), n_iters, rank)
        initrem_time = bench_init_remote(td, n_iters, rank)

        if rank == 0:
            speedup = leaf_time / consol_time if consol_time > 0 else float("inf")
            print(
                f"{num_tensors:>12} {tensor_size:>12} {mb:>10.2f} "
                f"{leaf_time * 1000:>10.2f} {consol_time * 1000:>10.2f} {speedup:>8.1f}x "
                f"{bcast_time * 1000:>10.2f} {allred_time * 1000:>10.2f} {initrem_time * 1000:>10.2f}"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

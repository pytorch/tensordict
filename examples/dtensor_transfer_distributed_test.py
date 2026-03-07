#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed test for DTensor transfer strategies A, B.

Usage (4 GPUs, single-node):
    torchrun --nproc_per_node=4 examples/dtensor_transfer_distributed_test.py

Usage (8 GPUs, single-node):
    torchrun --nproc_per_node=8 examples/dtensor_transfer_distributed_test.py

Strategy A: rank 0 sends materialized full tensors to rank 1.
Strategy B: rank 0 sends local shards + metadata to rank 1.
Non-participating ranks wait at barriers.
"""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard
from torch.distributed.tensor import distribute_tensor

from tensordict import TensorDict


def log(msg: str):
    rank = dist.get_rank()
    print(f"[rank {rank}] {msg}", flush=True)


# ======================================================================
# Strategy A: materialize-and-reshard
# ======================================================================
def test_strategy_a_materialize():
    """Test Strategy A: rank 0 sends, rank 1 receives."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        log("=" * 60)
        log("Testing Strategy A: materialize (rank 0 -> rank 1)")
        log("=" * 60)

    mesh = DeviceMesh("cuda", torch.arange(world_size))

    full_a = torch.arange(
        world_size * 10, dtype=torch.float32, device="cuda"
    )
    full_b = torch.randn(4, world_size * 8, dtype=torch.float32, device="cuda")

    dt_a = distribute_tensor(full_a, mesh, [Shard(0)])
    dt_b = distribute_tensor(full_b, mesh, [Shard(1)])

    td_src = TensorDict(a=dt_a, b=dt_b)

    if rank == 0:
        log(f"  a local shape: {dt_a.to_local().shape}")
        log(f"  b local shape: {dt_b.to_local().shape}")

    dist.barrier()

    if rank == 0:
        td_src.dtensor_send(
            dst=1,
            strategy="materialize",
            transport="torch_distributed",
        )
        log("  Sent OK")
    elif rank == 1:
        td_recv = TensorDict(
            a=torch.empty_like(full_a),
            b=torch.empty_like(full_b),
        )
        td_recv.dtensor_recv(
            src=0,
            strategy="materialize",
            transport="torch_distributed",
        )

        assert torch.allclose(td_recv["a"], full_a), "a mismatch!"
        assert torch.allclose(td_recv["b"], full_b), "b mismatch!"
        log("  Verification PASSED!")

    dist.barrier()
    log("  Strategy A done")


# ======================================================================
# Strategy B: redistribute
# ======================================================================
def test_strategy_b_redistribute():
    """Test Strategy B: send local shards + placement metadata."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        log("=" * 60)
        log("Testing Strategy B: redistribute (rank 0 -> rank 1)")
        log("=" * 60)

    mesh = DeviceMesh("cuda", torch.arange(world_size))

    full_tensor = torch.arange(
        world_size * 12, dtype=torch.float32, device="cuda"
    )
    dt = distribute_tensor(full_tensor, mesh, [Shard(0)])
    td_src = TensorDict(weight=dt)

    if rank == 0:
        log(f"  Local shard shape: {dt.to_local().shape}")

    dist.barrier()

    if rank == 0:
        td_src.dtensor_send(
            dst=1,
            strategy="redistribute",
            transport="torch_distributed",
        )
        log("  Sent OK")
    elif rank == 1:
        local_size = len(full_tensor) // world_size
        td_recv = TensorDict(
            weight=torch.empty(local_size, device="cuda"),
        )
        td_recv.dtensor_recv(
            src=0,
            strategy="redistribute",
            transport="torch_distributed",
        )

        expected_local = list(full_tensor.chunk(world_size))[0]
        received = td_recv["weight"]
        assert torch.allclose(received, expected_local), (
            f"weight mismatch: got {received}, expected {expected_local}"
        )
        log("  Verification PASSED!")

    dist.barrier()
    log("  Strategy B done")


# ======================================================================
# Strategy A with non-DTensor
# ======================================================================
def test_strategy_a_plain_tensor():
    """Test that non-DTensor tensors pass through correctly."""
    rank = dist.get_rank()

    if rank == 0:
        log("=" * 60)
        log("Testing Strategy A: plain tensor (rank 0 -> rank 1)")
        log("=" * 60)

    plain = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    td_src = TensorDict(plain_tensor=plain)

    dist.barrier()

    if rank == 0:
        td_src.dtensor_send(
            dst=1,
            strategy="materialize",
            transport="torch_distributed",
        )
        log("  Sent non-DTensor OK")
    elif rank == 1:
        td_recv = TensorDict(
            plain_tensor=torch.empty(4, device="cuda"),
        )
        td_recv.dtensor_recv(
            src=0,
            strategy="materialize",
            transport="torch_distributed",
        )
        assert torch.equal(td_recv["plain_tensor"], plain)
        log("  Non-DTensor passthrough PASSED!")

    dist.barrier()
    log("  Plain tensor test done")


# ======================================================================
# Strategy A with multiple keys
# ======================================================================
def test_strategy_a_multi_key():
    """Test Strategy A with multiple DTensor keys."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        log("=" * 60)
        log("Testing Strategy A: multi-key DTensor (rank 0 -> rank 1)")
        log("=" * 60)

    mesh = DeviceMesh("cuda", torch.arange(world_size))

    full_w = torch.randn(8, world_size * 4, dtype=torch.float32, device="cuda")
    full_b = torch.randn(world_size * 4, dtype=torch.float32, device="cuda")

    dt_w = distribute_tensor(full_w, mesh, [Shard(1)])
    dt_b = distribute_tensor(full_b, mesh, [Shard(0)])

    td_src = TensorDict(bias=dt_b, weight=dt_w)

    dist.barrier()

    if rank == 0:
        td_src.dtensor_send(
            dst=1,
            strategy="materialize",
            transport="torch_distributed",
        )
        log("  Sent multi-key OK")
    elif rank == 1:
        td_recv = TensorDict(
            bias=torch.empty_like(full_b),
            weight=torch.empty_like(full_w),
        )
        td_recv.dtensor_recv(
            src=0,
            strategy="materialize",
            transport="torch_distributed",
        )
        assert torch.allclose(td_recv["weight"], full_w), "weight mismatch!"
        assert torch.allclose(td_recv["bias"], full_b), "bias mismatch!"
        log("  Multi-key Verification PASSED!")

    dist.barrier()
    log("  Multi-key test done")


# ======================================================================
# Main
# ======================================================================
def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(rank % torch.cuda.device_count())

    log(f"Initialized: world_size={world_size}, "
        f"device=cuda:{rank % torch.cuda.device_count()}")

    test_strategy_a_materialize()
    test_strategy_b_redistribute()
    test_strategy_a_plain_tensor()
    test_strategy_a_multi_key()

    if rank == 0:
        log("\n" + "=" * 60)
        log("ALL DISTRIBUTED TESTS PASSED!")
        log("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

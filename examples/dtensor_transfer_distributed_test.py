#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed test for DTensor transfer strategies A, B, C.

Usage (2 GPUs):
    torchrun --nproc_per_node=2 examples/dtensor_transfer_distributed_test.py

Usage (4 GPUs):
    torchrun --nproc_per_node=4 examples/dtensor_transfer_distributed_test.py

The test creates DTensors with known values, transfers them using each
strategy, and verifies correctness on the receiving side.
"""

import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor import distribute_tensor

from tensordict import TensorDict


def log(msg: str):
    rank = dist.get_rank()
    print(f"[rank {rank}] {msg}", flush=True)


def test_strategy_a_materialize(mesh: DeviceMesh):
    """Test Strategy A: materialize full tensor, send, receive."""
    log("=" * 50)
    log("Testing Strategy A: materialize")
    log("=" * 50)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    full_tensor_a = torch.arange(world_size * 10, dtype=torch.float32, device="cuda")
    full_tensor_b = torch.randn(4, world_size * 8, dtype=torch.float32, device="cuda")

    dt_a = distribute_tensor(full_tensor_a, mesh, [Shard(0)])
    dt_b = distribute_tensor(full_tensor_b, mesh, [Shard(1)])

    td_src = TensorDict(a=dt_a, b=dt_b)

    if rank == 0:
        log(f"  Source TD keys: {list(td_src.keys())}")
        log(f"  a.placements: {dt_a.placements}, local_shape: {dt_a.to_local().shape}")
        log(f"  b.placements: {dt_b.placements}, local_shape: {dt_b.to_local().shape}")

    dist.barrier()

    # Rank 0 sends to rank 1
    if rank == 0:
        td_src.dtensor_send(
            dst=1,
            strategy="materialize",
            transport="torch_distributed",
        )
        log("  Sent successfully")
    elif rank == 1:
        td_recv = TensorDict(
            a=torch.empty_like(full_tensor_a),
            b=torch.empty_like(full_tensor_b),
        )
        td_recv.dtensor_recv(
            src=0,
            strategy="materialize",
            transport="torch_distributed",
        )
        log("  Received successfully")

        # Verify
        assert torch.allclose(td_recv["a"], full_tensor_a), (
            f"a mismatch: {td_recv['a']} vs {full_tensor_a}"
        )
        assert torch.allclose(td_recv["b"], full_tensor_b), "b mismatch"
        log("  Verification PASSED!")

    dist.barrier()


def test_strategy_b_redistribute(mesh: DeviceMesh):
    """Test Strategy B: send local shards + placement metadata."""
    log("=" * 50)
    log("Testing Strategy B: redistribute")
    log("=" * 50)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    full_tensor = torch.arange(
        world_size * 10, dtype=torch.float32, device="cuda"
    )
    dt = distribute_tensor(full_tensor, mesh, [Shard(0)])

    td_src = TensorDict(weight=dt)

    if rank == 0:
        local = dt.to_local()
        log(f"  Source local shard shape: {local.shape}, values: {local[:5]}...")

    dist.barrier()

    if rank == 0:
        td_src.dtensor_send(
            dst=1,
            strategy="redistribute",
            transport="torch_distributed",
        )
        log("  Sent successfully")
    elif rank == 1:
        td_recv = TensorDict(
            weight=torch.empty(world_size * 10 // world_size, device="cuda"),
        )
        td_recv.dtensor_recv(
            src=0,
            strategy="redistribute",
            transport="torch_distributed",
        )
        log("  Received successfully")

        # Strategy B sends the local shard from the sender.
        # Rank 0's local shard of Shard(0) on a world_size mesh is the first chunk.
        expected_local = list(full_tensor.chunk(world_size))[0]
        received = td_recv["weight"]
        log(f"  Received shape: {received.shape}")
        log(f"  Expected shape: {expected_local.shape}")
        assert torch.allclose(received, expected_local), (
            f"weight mismatch: {received} vs {expected_local}"
        )
        log("  Verification PASSED!")

    dist.barrier()


def test_strategy_a_non_dtensor(mesh: DeviceMesh):
    """Test that non-DTensor tensors pass through correctly in Strategy A."""
    log("=" * 50)
    log("Testing Strategy A: non-DTensor passthrough")
    log("=" * 50)

    rank = dist.get_rank()

    plain = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    td_src = TensorDict(plain_tensor=plain)

    dist.barrier()

    if rank == 0:
        td_src.dtensor_send(
            dst=1,
            strategy="materialize",
            transport="torch_distributed",
        )
        log("  Sent non-DTensor successfully")
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


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(rank)

    log(f"Initialized: world_size={world_size}")

    mesh = DeviceMesh("cuda", torch.arange(world_size))

    test_strategy_a_materialize(mesh)
    test_strategy_b_redistribute(mesh)
    test_strategy_a_non_dtensor(mesh)

    if rank == 0:
        log("\n" + "=" * 50)
        log("ALL DISTRIBUTED TESTS PASSED!")
        log("=" * 50)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

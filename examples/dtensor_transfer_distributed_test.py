#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed test for DTensor transfer strategies A, B, C.

Usage (4 GPUs, single-node):
    torchrun --nproc_per_node=4 examples/dtensor_transfer_distributed_test.py

Usage (8 GPUs, single-node):
    torchrun --nproc_per_node=8 examples/dtensor_transfer_distributed_test.py

The test creates DTensors with known values, transfers them using each
strategy, and verifies correctness on the receiving side.

Strategy A and B use a single sender rank -> single receiver rank pattern.
Strategy C uses the full mesh-to-mesh optimal P2P transfer.
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


# ======================================================================
# Strategy A: materialize-and-reshard
# ======================================================================
def test_strategy_a_materialize():
    """Test Strategy A: rank 0 sends, rank 1 receives.

    Only ranks 0 and 1 participate in the P2P; others wait at the barrier.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        log("=" * 60)
        log("Testing Strategy A: materialize (rank 0 -> rank 1)")
        log("=" * 60)

    # Create a DeviceMesh covering all ranks
    mesh = DeviceMesh("cuda", torch.arange(world_size))

    full_a = torch.arange(
        world_size * 10, dtype=torch.float32, device="cuda"
    )
    full_b = torch.randn(4, world_size * 8, dtype=torch.float32, device="cuda")

    dt_a = distribute_tensor(full_a, mesh, [Shard(0)])
    dt_b = distribute_tensor(full_b, mesh, [Shard(1)])

    td_src = TensorDict(a=dt_a, b=dt_b)

    if rank == 0:
        log(f"  a local shape: {dt_a.to_local().shape}, b local shape: {dt_b.to_local().shape}")

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
        # Recv expects a buffer matching the local shard shape
        local_size = len(full_tensor) // world_size
        td_recv = TensorDict(
            weight=torch.empty(local_size, device="cuda"),
        )
        td_recv.dtensor_recv(
            src=0,
            strategy="redistribute",
            transport="torch_distributed",
        )

        # Strategy B sends rank 0's local shard
        expected_local = list(full_tensor.chunk(world_size))[0]
        received = td_recv["weight"]
        assert torch.allclose(received, expected_local), (
            f"weight mismatch: got {received}, expected {expected_local}"
        )
        log("  Verification PASSED!")

    dist.barrier()


# ======================================================================
# Strategy C: optimal P2P
# ======================================================================
def test_strategy_c_optimal():
    """Test Strategy C: optimal P2P mesh-to-mesh transfer.

    Simulates cross-mesh transfer by using the first half of ranks as
    "source mesh" and the second half as "destination mesh".
    All ranks participate in the P2P transfers.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size < 4:
        if rank == 0:
            log("SKIP Strategy C: need at least 4 GPUs")
        return

    if rank == 0:
        log("=" * 60)
        log("Testing Strategy C: optimal P2P mesh-to-mesh")
        log("=" * 60)

    half = world_size // 2
    src_ranks = list(range(half))
    dst_ranks = list(range(half, world_size))

    src_mesh = DeviceMesh("cuda", torch.tensor(src_ranks))
    dst_mesh = DeviceMesh("cuda", torch.tensor(dst_ranks))

    full_tensor = torch.arange(
        half * 16, dtype=torch.float32, device="cuda"
    )

    is_sender = rank in src_ranks
    is_receiver = rank in dst_ranks

    if rank == 0:
        log(f"  src_ranks={src_ranks}, dst_ranks={dst_ranks}")
        log(f"  Src placements: [Shard(0)] on {half} ranks")
        log(f"  Dst placements: [Shard(0)] on {half} ranks")

    dist.barrier()

    src_placements = (Shard(0),)
    dst_placements = (Shard(0),)

    if is_sender:
        dt = distribute_tensor(full_tensor, src_mesh, list(src_placements))
        td_src = TensorDict(data=dt)

        log(f"  Sender local shape: {dt.to_local().shape}")

        td_src.dtensor_send(
            dst=dst_ranks[0],
            dst_mesh=dst_mesh,
            dst_placements=dst_placements,
            strategy="optimal",
            transport="torch_distributed",
        )
        log("  Sent OK")

    if is_receiver:
        dt_recv = distribute_tensor(
            torch.zeros_like(full_tensor), dst_mesh, list(dst_placements)
        )
        td_recv = TensorDict(data=dt_recv)

        td_recv.dtensor_recv(
            src=src_ranks[0],
            src_mesh=src_mesh,
            src_placements=src_placements,
            strategy="optimal",
            transport="torch_distributed",
        )

        # Verify: gather the received DTensor and compare
        received_full = td_recv["data"].full_tensor()
        assert torch.allclose(received_full, full_tensor), (
            f"Mismatch on rank {rank}!"
        )
        log("  Verification PASSED!")

    dist.barrier()


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
    test_strategy_c_optimal()

    if rank == 0:
        log("\n" + "=" * 60)
        log("ALL DISTRIBUTED TESTS PASSED!")
        log("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed test for DTensor transfer strategies A, B.

Usage (4 GPUs, single-node):
    torchrun --nproc_per_node=4 examples/dtensor_transfer_distributed_test.py

Strategy A (materialize): all ranks collectively call full_tensor() to
    gather shards, then rank 0 sends the full tensor to rank 1 as a
    plain (non-DTensor) tensor via P2P.

Strategy B (redistribute): rank 0 sends its local shard + metadata to
    rank 1 via P2P. No collective needed.
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
# Strategy A: materialize-and-send
# ======================================================================
def test_strategy_a_materialize():
    """Test Strategy A.

    full_tensor() is a collective, so ALL ranks must participate.
    After materializing, only rank 0 sends the plain tensors to rank 1.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        log("=" * 60)
        log("Testing Strategy A: materialize (rank 0 -> rank 1)")
        log("=" * 60)

    mesh = DeviceMesh("cuda", torch.arange(world_size))

    torch.manual_seed(42)
    full_a = torch.arange(
        world_size * 10, dtype=torch.float32, device="cuda"
    )
    full_b = torch.randn(4, world_size * 8, dtype=torch.float32, device="cuda")

    dt_a = distribute_tensor(full_a, mesh, [Shard(0)])
    dt_b = distribute_tensor(full_b, mesh, [Shard(1)])

    # full_tensor() is COLLECTIVE - all ranks must call it
    materialized_a = dt_a.full_tensor()
    materialized_b = dt_b.full_tensor()

    if rank == 0:
        log(f"  Materialized a: {materialized_a.shape}")
        log(f"  Materialized b: {materialized_b.shape}")

    # Now send the plain (non-DTensor) tensors from rank 0 -> rank 1
    td_plain = TensorDict(a=materialized_a, b=materialized_b)

    dist.barrier()

    if rank == 0:
        td_plain.dtensor_send(
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
# Strategy B: redistribute (send local shard)
# ======================================================================
def test_strategy_b_redistribute():
    """Test Strategy B: send local shards + placement metadata.

    to_local() is NOT collective - only rank 0 calls it.
    Rank 0 sends its local shard to rank 1.
    """
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
# Strategy A with plain tensors only
# ======================================================================
def test_plain_tensor():
    """Test plain tensor P2P (no DTensor involved)."""
    rank = dist.get_rank()

    if rank == 0:
        log("=" * 60)
        log("Testing plain tensor (rank 0 -> rank 1)")
        log("=" * 60)

    plain = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    td_src = TensorDict(x=plain)

    dist.barrier()

    if rank == 0:
        td_src.dtensor_send(
            dst=1,
            strategy="materialize",
            transport="torch_distributed",
        )
        log("  Sent OK")
    elif rank == 1:
        td_recv = TensorDict(x=torch.empty(4, device="cuda"))
        td_recv.dtensor_recv(
            src=0,
            strategy="materialize",
            transport="torch_distributed",
        )
        assert torch.equal(td_recv["x"], plain)
        log("  Plain tensor PASSED!")

    dist.barrier()
    log("  Plain tensor test done")


# ======================================================================
# Strategy A with multiple keys (pre-materialized)
# ======================================================================
def test_multi_key():
    """Test multi-key DTensor transfer with pre-materialization."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        log("=" * 60)
        log("Testing multi-key DTensor (rank 0 -> rank 1)")
        log("=" * 60)

    mesh = DeviceMesh("cuda", torch.arange(world_size))

    torch.manual_seed(123)
    full_w = torch.randn(8, world_size * 4, dtype=torch.float32, device="cuda")
    full_b = torch.randn(world_size * 4, dtype=torch.float32, device="cuda")

    dt_w = distribute_tensor(full_w, mesh, [Shard(1)])
    dt_b = distribute_tensor(full_b, mesh, [Shard(0)])

    # Collective materialize
    mat_w = dt_w.full_tensor()
    mat_b = dt_b.full_tensor()

    td_plain = TensorDict(bias=mat_b, weight=mat_w)

    dist.barrier()

    if rank == 0:
        td_plain.dtensor_send(
            dst=1,
            strategy="materialize",
            transport="torch_distributed",
        )
        log("  Sent OK")
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
        log("  Multi-key PASSED!")

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

    test_plain_tensor()
    test_strategy_a_materialize()
    test_strategy_b_redistribute()
    test_multi_key()

    if rank == 0:
        log("\n" + "=" * 60)
        log("ALL DISTRIBUTED TESTS PASSED!")
        log("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

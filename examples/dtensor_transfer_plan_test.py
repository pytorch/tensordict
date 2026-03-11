#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test the DTensor transfer plan computation (no GPU / distributed needed).

Usage:
    python examples/dtensor_transfer_plan_test.py
"""

import torch

from tensordict._dtensor import (
    _compute_all_local_slices,
    _compute_transfer_plan,
)
from torch.distributed.tensor.placement_types import Replicate, Shard


def test_shard4_to_shard2():
    """Simulate Shard(0) on 4 ranks -> Shard(0) on 2 ranks."""
    print("=" * 60)
    print("Test: Shard(0) on 4 ranks -> Shard(0) on 2 ranks")
    print("=" * 60)

    full = torch.arange(100, dtype=torch.float32)

    plan = _compute_transfer_plan(
        global_shape=[100],
        src_mesh_shape=[4],
        src_placements=[Shard(0)],
        dst_mesh_shape=[2],
        dst_placements=[Shard(0)],
    )

    print(f"  Global shape: {plan.global_shape}")
    print(f"  Number of transfers: {len(plan.transfers)}")
    for t in plan.transfers:
        print(
            f"    rank {t.src_rank} -> rank {t.dst_rank}: "
            f"global {t.global_slices}, "
            f"src_local {t.src_slices}, dst_local {t.dst_slices}"
        )

    src_shards = list(full.chunk(4))
    dst_shards = [torch.zeros(50) for _ in range(2)]
    for t in plan.transfers:
        src_data = src_shards[t.src_rank][t.src_slices[0]]
        dst_shards[t.dst_rank][t.dst_slices[0]] = src_data

    expected = list(full.chunk(2))
    for i in range(2):
        assert torch.equal(dst_shards[i], expected[i]), (
            f"rank {i}: expected {expected[i]}, got {dst_shards[i]}"
        )
    print("  PASSED\n")


def test_2d_to_1d():
    """Simulate [Shard(0), Shard(1)] on 2x2 -> [Shard(0)] on 2."""
    print("=" * 60)
    print("Test: [Shard(0), Shard(1)] on 2x2 -> [Shard(0)] on 2")
    print("=" * 60)

    full = torch.arange(200, dtype=torch.float32).reshape(10, 20)

    plan = _compute_transfer_plan(
        global_shape=[10, 20],
        src_mesh_shape=[2, 2],
        src_placements=[Shard(0), Shard(1)],
        dst_mesh_shape=[2],
        dst_placements=[Shard(0)],
    )

    print(f"  Global shape: {plan.global_shape}")
    print(f"  Number of transfers: {len(plan.transfers)}")
    for t in plan.transfers:
        print(
            f"    rank {t.src_rank} -> rank {t.dst_rank}: "
            f"global {t.global_slices}"
        )

    top = full[:5]
    bottom = full[5:]
    src_shards = {
        0: top[:, :10],
        1: top[:, 10:],
        2: bottom[:, :10],
        3: bottom[:, 10:],
    }
    dst_shards = {0: torch.zeros(5, 20), 1: torch.zeros(5, 20)}

    for t in plan.transfers:
        src_data = src_shards[t.src_rank][t.src_slices]
        dst_shards[t.dst_rank][t.dst_slices] = src_data

    assert torch.equal(dst_shards[0], full[:5])
    assert torch.equal(dst_shards[1], full[5:])
    print("  PASSED\n")


def test_dp_tp_to_tp_only():
    """Realistic: FSDP+TP training -> TP-only inference."""
    print("=" * 60)
    print("Test: [Shard(0), Shard(1)] on DP=2,TP=2 -> [Shard(1)] on TP=2")
    print("  (FSDP+TP training -> TP-only inference)")
    print("=" * 60)

    full = torch.randn(8, 16)

    plan = _compute_transfer_plan(
        global_shape=[8, 16],
        src_mesh_shape=[2, 2],
        src_placements=[Shard(0), Shard(1)],
        dst_mesh_shape=[2],
        dst_placements=[Shard(1)],
    )

    print(f"  Global shape: {plan.global_shape}")
    print(f"  Number of transfers: {len(plan.transfers)}")

    src_specs = _compute_all_local_slices([8, 16], [2, 2], [Shard(0), Shard(1)])
    dst_specs = _compute_all_local_slices([8, 16], [2], [Shard(1)])

    print("\n  Source shards:")
    for s in src_specs:
        print(f"    rank {s.rank}: {s.slices}")
    print("  Destination shards:")
    for s in dst_specs:
        print(f"    rank {s.rank}: {s.slices}")

    print("\n  Transfers:")
    for t in plan.transfers:
        print(
            f"    rank {t.src_rank} -> rank {t.dst_rank}: "
            f"global {t.global_slices}"
        )

    src_shards = {s.rank: full[s.slices].clone() for s in src_specs}
    dst_buffers = {s.rank: torch.zeros_like(full[s.slices]) for s in dst_specs}

    for t in plan.transfers:
        src_data = src_shards[t.src_rank][t.src_slices]
        dst_buffers[t.dst_rank][t.dst_slices] = src_data

    for s in dst_specs:
        expected = full[s.slices]
        actual = dst_buffers[s.rank]
        assert torch.equal(actual, expected), (
            f"rank {s.rank}: mismatch!\n  expected: {expected}\n  got: {actual}"
        )
    print("  PASSED\n")


def test_replicate_to_shard():
    """Replicate on 4 ranks -> Shard(0) on 2 ranks."""
    print("=" * 60)
    print("Test: Replicate on 4 ranks -> Shard(0) on 2 ranks")
    print("=" * 60)

    full = torch.arange(100, dtype=torch.float32)

    plan = _compute_transfer_plan(
        global_shape=[100],
        src_mesh_shape=[4],
        src_placements=[Replicate()],
        dst_mesh_shape=[2],
        dst_placements=[Shard(0)],
    )

    print(f"  Number of transfers: {len(plan.transfers)}")
    for t in plan.transfers:
        print(
            f"    rank {t.src_rank} -> rank {t.dst_rank}: "
            f"global {t.global_slices}"
        )
    assert all(t.src_rank == 0 for t in plan.transfers), (
        "Expected deduplication to use only rank 0 as source"
    )
    print("  PASSED\n")


if __name__ == "__main__":
    test_shard4_to_shard2()
    test_2d_to_1d()
    test_dp_tp_to_tp_only()
    test_replicate_to_shard()
    print("All transfer plan tests passed!")

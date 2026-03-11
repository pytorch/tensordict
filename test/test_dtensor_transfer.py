# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the cross-mesh DTensor transfer plan computation.

All tests are pure logic -- no GPU, no torch.distributed needed.
"""

from __future__ import annotations

import pytest
import torch

from tensordict._dtensor import (
    _chunk_slice,
    _ChunkTransfer,
    _compute_all_local_slices,
    _compute_local_slices,
    _compute_transfer_plan,
    _intersect_1d,
    _intersect_slices,
    _ShardSpec,
    _slice_relative_to,
    _TransferPlan,
    execute_transfer_plan,
    ModelTransferPlan,
    ParameterPlan,
    ShardingDescriptor,
)


# ---------------------------------------------------------------------------
# Lightweight placement stubs for testing without torch.distributed
# ---------------------------------------------------------------------------


class _Shard:
    """Test stub matching the Shard placement interface."""

    def __init__(self, dim: int):
        self.dim = dim

    def is_replicate(self):
        return False

    def is_partial(self):
        return False

    def __eq__(self, other):
        return isinstance(other, _Shard) and self.dim == other.dim

    def __hash__(self):
        return hash(("Shard", self.dim))

    def __repr__(self):
        return f"Shard({self.dim})"


class _Replicate:
    """Test stub matching the Replicate placement interface."""

    def is_replicate(self):
        return True

    def is_partial(self):
        return False

    def __eq__(self, other):
        return isinstance(other, _Replicate)

    def __hash__(self):
        return hash("Replicate")

    def __repr__(self):
        return "Replicate()"


class _Partial:
    """Test stub matching the Partial placement interface."""

    def is_replicate(self):
        return False

    def is_partial(self):
        return True

    def __eq__(self, other):
        return isinstance(other, _Partial)

    def __hash__(self):
        return hash("Partial")

    def __repr__(self):
        return "Partial()"


# ---------------------------------------------------------------------------
# Slice helpers
# ---------------------------------------------------------------------------


class TestChunkSlice:
    def test_even_split(self):
        # 100 into 4 -> 25 each
        assert _chunk_slice(100, 4, 0) == slice(0, 25)
        assert _chunk_slice(100, 4, 1) == slice(25, 50)
        assert _chunk_slice(100, 4, 2) == slice(50, 75)
        assert _chunk_slice(100, 4, 3) == slice(75, 100)

    def test_uneven_split(self):
        # 10 into 3 -> chunks of size 4, 4, 2
        assert _chunk_slice(10, 3, 0) == slice(0, 4)
        assert _chunk_slice(10, 3, 1) == slice(4, 8)
        assert _chunk_slice(10, 3, 2) == slice(8, 10)

    def test_more_chunks_than_elements(self):
        # 2 into 4 -> chunk_size=1, last two are empty
        assert _chunk_slice(2, 4, 0) == slice(0, 1)
        assert _chunk_slice(2, 4, 1) == slice(1, 2)
        assert _chunk_slice(2, 4, 2) == slice(0, 0)
        assert _chunk_slice(2, 4, 3) == slice(0, 0)

    def test_single_chunk(self):
        assert _chunk_slice(50, 1, 0) == slice(0, 50)


class TestIntersect1D:
    def test_overlap(self):
        assert _intersect_1d(slice(0, 10), slice(5, 15)) == slice(5, 10)

    def test_no_overlap(self):
        assert _intersect_1d(slice(0, 5), slice(5, 10)) is None

    def test_contained(self):
        assert _intersect_1d(slice(0, 20), slice(5, 15)) == slice(5, 15)

    def test_identical(self):
        assert _intersect_1d(slice(3, 7), slice(3, 7)) == slice(3, 7)

    def test_empty(self):
        assert _intersect_1d(slice(0, 0), slice(0, 10)) is None


class TestIntersectSlices:
    def test_2d_overlap(self):
        a = (slice(0, 10), slice(0, 20))
        b = (slice(5, 15), slice(10, 30))
        result = _intersect_slices(a, b)
        assert result == (slice(5, 10), slice(10, 20))

    def test_2d_no_overlap(self):
        a = (slice(0, 10), slice(0, 5))
        b = (slice(10, 20), slice(5, 10))
        assert _intersect_slices(a, b) is None

    def test_partial_overlap_one_dim(self):
        a = (slice(0, 10), slice(0, 5))
        b = (slice(5, 15), slice(5, 10))
        assert _intersect_slices(a, b) is None


class TestSliceRelativeTo:
    def test_basic(self):
        result = _slice_relative_to(slice(5, 10), slice(3, 15))
        assert result == slice(2, 7)

    def test_same_start(self):
        result = _slice_relative_to(slice(5, 10), slice(5, 20))
        assert result == slice(0, 5)


# ---------------------------------------------------------------------------
# Local slices computation
# ---------------------------------------------------------------------------


class TestComputeLocalSlices:
    def test_1d_shard(self):
        # shape [100], mesh (4,), Shard(0)
        sl = _compute_local_slices([100], [4], [_Shard(0)], [0])
        assert sl == (slice(0, 25),)
        sl = _compute_local_slices([100], [4], [_Shard(0)], [3])
        assert sl == (slice(75, 100),)

    def test_1d_replicate(self):
        sl = _compute_local_slices([100], [4], [_Replicate()], [2])
        assert sl == (slice(0, 100),)

    def test_2d_shard_shard(self):
        # shape [100, 200], mesh (2, 4), [Shard(0), Shard(1)]
        # rank (0, 0) -> rows 0-50, cols 0-50
        sl = _compute_local_slices(
            [100, 200], [2, 4], [_Shard(0), _Shard(1)], [0, 0]
        )
        assert sl == (slice(0, 50), slice(0, 50))

        # rank (1, 3) -> rows 50-100, cols 150-200
        sl = _compute_local_slices(
            [100, 200], [2, 4], [_Shard(0), _Shard(1)], [1, 3]
        )
        assert sl == (slice(50, 100), slice(150, 200))

    def test_2d_shard_replicate(self):
        # mesh (2, 2), [Shard(0), Replicate()]
        sl = _compute_local_slices(
            [100, 200], [2, 2], [_Shard(0), _Replicate()], [0, 1]
        )
        assert sl == (slice(0, 50), slice(0, 200))

    def test_uneven_shard(self):
        # shape [10], mesh (3,), Shard(0) -> chunks 4,4,2
        sl = _compute_local_slices([10], [3], [_Shard(0)], [2])
        assert sl == (slice(8, 10),)


class TestComputeAllLocalSlices:
    def test_1d(self):
        specs = _compute_all_local_slices([100], [4], [_Shard(0)])
        assert len(specs) == 4
        assert specs[0] == _ShardSpec(rank=0, slices=(slice(0, 25),))
        assert specs[3] == _ShardSpec(rank=3, slices=(slice(75, 100),))

    def test_with_rank_map(self):
        rank_map = {(0,): 10, (1,): 20}
        specs = _compute_all_local_slices([100], [2], [_Shard(0)], rank_map)
        assert specs[0].rank == 10
        assert specs[1].rank == 20

    def test_2d_mesh(self):
        specs = _compute_all_local_slices(
            [100, 200], [2, 2], [_Shard(0), _Shard(1)]
        )
        assert len(specs) == 4
        # rank 0 = coords (0,0), rank 1 = coords (0,1), etc.
        assert specs[0].slices == (slice(0, 50), slice(0, 100))
        assert specs[1].slices == (slice(0, 50), slice(100, 200))
        assert specs[2].slices == (slice(50, 100), slice(0, 100))
        assert specs[3].slices == (slice(50, 100), slice(100, 200))


# ---------------------------------------------------------------------------
# Transfer plan computation
# ---------------------------------------------------------------------------


class TestTransferPlan:
    def test_shard_to_shard_same_size(self):
        """Same mesh size, same shard dim -> each rank sends to itself."""
        plan = _compute_transfer_plan(
            global_shape=[100],
            src_mesh_shape=[4],
            src_placements=[_Shard(0)],
            dst_mesh_shape=[4],
            dst_placements=[_Shard(0)],
        )
        assert len(plan.transfers) == 4
        for t in plan.transfers:
            assert t.src_rank == t.dst_rank

    def test_shard_to_shard_different_sizes(self):
        """src: 4 ranks, dst: 2 ranks, both Shard(0) on 100 elements."""
        plan = _compute_transfer_plan(
            global_shape=[100],
            src_mesh_shape=[4],
            src_placements=[_Shard(0)],
            dst_mesh_shape=[2],
            dst_placements=[_Shard(0)],
        )
        # src: [0-25, 25-50, 50-75, 75-100]
        # dst: [0-50, 50-100]
        # transfers: src0->dst0, src1->dst0, src2->dst1, src3->dst1
        assert len(plan.transfers) == 4

        dst0_recvs = plan.recvs_for_rank(0)
        assert len(dst0_recvs) == 2
        src_ranks = sorted(t.src_rank for t in dst0_recvs)
        assert src_ranks == [0, 1]

        dst1_recvs = plan.recvs_for_rank(1)
        assert len(dst1_recvs) == 2
        src_ranks = sorted(t.src_rank for t in dst1_recvs)
        assert src_ranks == [2, 3]

    def test_shard_to_replicate(self):
        """src: Shard(0) on 4 ranks, dst: Replicate on 2 ranks."""
        plan = _compute_transfer_plan(
            global_shape=[100],
            src_mesh_shape=[4],
            src_placements=[_Shard(0)],
            dst_mesh_shape=[2],
            dst_placements=[_Replicate()],
        )
        # Each dst rank needs the full tensor, so receives from all 4 src ranks
        for dst_rank in range(2):
            recvs = plan.recvs_for_rank(dst_rank)
            assert len(recvs) == 4

    def test_replicate_to_shard(self):
        """src: Replicate on 4 ranks, dst: Shard(0) on 2 ranks.

        Should deduplicate so only one src rank sends each piece.
        """
        plan = _compute_transfer_plan(
            global_shape=[100],
            src_mesh_shape=[4],
            src_placements=[_Replicate()],
            dst_mesh_shape=[2],
            dst_placements=[_Shard(0)],
        )
        # All src ranks hold the full tensor but are deduplicated to rank 0.
        # dst0 needs [0-50], dst1 needs [50-100] -> 2 transfers
        assert len(plan.transfers) == 2
        assert all(t.src_rank == 0 for t in plan.transfers)

    def test_replicate_to_replicate(self):
        """src: Replicate on 2, dst: Replicate on 3."""
        plan = _compute_transfer_plan(
            global_shape=[100],
            src_mesh_shape=[2],
            src_placements=[_Replicate()],
            dst_mesh_shape=[3],
            dst_placements=[_Replicate()],
        )
        # Deduplicated: only src rank 0 sends, to each of 3 dst ranks
        assert len(plan.transfers) == 3
        assert all(t.src_rank == 0 for t in plan.transfers)

    def test_2d_to_1d(self):
        """Training: [Shard(0), Shard(1)] on 2x2 mesh.
        Inference: [Shard(0)] on 2 mesh.
        Tensor shape: [100, 200].
        """
        plan = _compute_transfer_plan(
            global_shape=[100, 200],
            src_mesh_shape=[2, 2],
            src_placements=[_Shard(0), _Shard(1)],
            dst_mesh_shape=[2],
            dst_placements=[_Shard(0)],
        )
        # src: rank0=(0-50,0-100), rank1=(0-50,100-200),
        #      rank2=(50-100,0-100), rank3=(50-100,100-200)
        # dst: rank0=(0-50,0-200), rank1=(50-100,0-200)
        #
        # dst0 needs (0-50, 0-200): overlap with src0 (0-50,0-100) and src1 (0-50,100-200)
        # dst1 needs (50-100, 0-200): overlap with src2 (50-100,0-100) and src3 (50-100,100-200)
        assert len(plan.transfers) == 4

        dst0_recvs = plan.recvs_for_rank(0)
        assert len(dst0_recvs) == 2
        src_ranks = sorted(t.src_rank for t in dst0_recvs)
        assert src_ranks == [0, 1]

        dst1_recvs = plan.recvs_for_rank(1)
        assert len(dst1_recvs) == 2
        src_ranks = sorted(t.src_rank for t in dst1_recvs)
        assert src_ranks == [2, 3]

    def test_1d_to_2d(self):
        """Inference->Training direction: [Shard(0)] on 2 mesh -> [Shard(0), Shard(1)] on 2x2.
        Tensor: [100, 200].
        """
        plan = _compute_transfer_plan(
            global_shape=[100, 200],
            src_mesh_shape=[2],
            src_placements=[_Shard(0)],
            dst_mesh_shape=[2, 2],
            dst_placements=[_Shard(0), _Shard(1)],
        )
        # src: rank0=(0-50, 0-200), rank1=(50-100, 0-200)
        # dst: rank0=(0-50,0-100), rank1=(0-50,100-200),
        #      rank2=(50-100,0-100), rank3=(50-100,100-200)
        #
        # dst0: overlap with src0 on (0-50,0-100)
        # dst1: overlap with src0 on (0-50,100-200)
        # dst2: overlap with src1 on (50-100,0-100)
        # dst3: overlap with src1 on (50-100,100-200)
        assert len(plan.transfers) == 4
        assert plan.recvs_for_rank(0)[0].src_rank == 0
        assert plan.recvs_for_rank(1)[0].src_rank == 0
        assert plan.recvs_for_rank(2)[0].src_rank == 1
        assert plan.recvs_for_rank(3)[0].src_rank == 1

    def test_identity(self):
        """Same mesh, same placements -> self-transfers."""
        plan = _compute_transfer_plan(
            global_shape=[100, 200],
            src_mesh_shape=[2],
            src_placements=[_Shard(0)],
            dst_mesh_shape=[2],
            dst_placements=[_Shard(0)],
        )
        assert len(plan.transfers) == 2
        for t in plan.transfers:
            assert t.src_rank == t.dst_rank
            # Full local tensor
            assert t.src_slices == t.dst_slices

    def test_uneven_shard(self):
        """Tensor of 10 elements, src: 3 ranks Shard(0), dst: 2 ranks Shard(0)."""
        plan = _compute_transfer_plan(
            global_shape=[10],
            src_mesh_shape=[3],
            src_placements=[_Shard(0)],
            dst_mesh_shape=[2],
            dst_placements=[_Shard(0)],
        )
        # src: [0-4, 4-8, 8-10]
        # dst: [0-5, 5-10]
        # Overlaps:
        #   src0(0-4) & dst0(0-5) -> (0-4) -- full src0
        #   src1(4-8) & dst0(0-5) -> (4-5) -- 1 element from src1
        #   src1(4-8) & dst1(5-10) -> (5-8) -- 3 elements from src1
        #   src2(8-10) & dst1(5-10) -> (8-10) -- full src2
        assert len(plan.transfers) == 4

    def test_partial_to_shard(self):
        """src: Partial on 2 ranks, dst: Shard(0) on 2 ranks.

        Partial is deduplicated to one canonical src rank.
        """
        plan = _compute_transfer_plan(
            global_shape=[100],
            src_mesh_shape=[2],
            src_placements=[_Partial()],
            dst_mesh_shape=[2],
            dst_placements=[_Shard(0)],
        )
        # Partial: both src ranks hold full data, but deduplicated to rank 0
        assert len(plan.transfers) == 2
        assert all(t.src_rank == 0 for t in plan.transfers)

    def test_partial_to_replicate(self):
        """src: Partial on 2, dst: Replicate on 3."""
        plan = _compute_transfer_plan(
            global_shape=[100],
            src_mesh_shape=[2],
            src_placements=[_Partial()],
            dst_mesh_shape=[3],
            dst_placements=[_Replicate()],
        )
        assert len(plan.transfers) == 3
        assert all(t.src_rank == 0 for t in plan.transfers)

    def test_custom_rank_maps(self):
        """Verify that custom rank maps are respected."""
        plan = _compute_transfer_plan(
            global_shape=[100],
            src_mesh_shape=[2],
            src_placements=[_Shard(0)],
            dst_mesh_shape=[2],
            dst_placements=[_Shard(0)],
            src_rank_map={(0,): 10, (1,): 11},
            dst_rank_map={(0,): 20, (1,): 21},
        )
        assert len(plan.transfers) == 2
        src_ranks = {t.src_rank for t in plan.transfers}
        dst_ranks = {t.dst_rank for t in plan.transfers}
        assert src_ranks == {10, 11}
        assert dst_ranks == {20, 21}

    def test_sends_and_recvs_for_rank(self):
        plan = _compute_transfer_plan(
            global_shape=[100],
            src_mesh_shape=[4],
            src_placements=[_Shard(0)],
            dst_mesh_shape=[2],
            dst_placements=[_Shard(0)],
        )
        # src rank 0 sends to dst rank 0
        sends = plan.sends_for_rank(0)
        assert len(sends) == 1
        assert sends[0].dst_rank == 0

        # dst rank 1 receives from src ranks 2, 3
        recvs = plan.recvs_for_rank(1)
        assert len(recvs) == 2

    def test_transfer_slices_correctness(self):
        """Verify that src_slices and dst_slices correctly index into local tensors."""
        plan = _compute_transfer_plan(
            global_shape=[100],
            src_mesh_shape=[4],
            src_placements=[_Shard(0)],
            dst_mesh_shape=[2],
            dst_placements=[_Shard(0)],
        )
        # src0 holds [0-25], dst0 needs [0-50]
        # transfer: src0 -> dst0, global [0-25]
        t = [x for x in plan.transfers if x.src_rank == 0 and x.dst_rank == 0][0]
        # src_slices: relative to src0's local [0-25] -> slice(0, 25)
        assert t.src_slices == (slice(0, 25),)
        # dst_slices: relative to dst0's local [0-50] -> slice(0, 25)
        assert t.dst_slices == (slice(0, 25),)

        # src1 holds [25-50], dst0 needs [0-50]
        t = [x for x in plan.transfers if x.src_rank == 1 and x.dst_rank == 0][0]
        assert t.src_slices == (slice(0, 25),)  # full src1 local
        assert t.dst_slices == (slice(25, 50),)  # offset 25 in dst0 local

    def test_shard_different_dim(self):
        """src: Shard(0) on 2 ranks, dst: Shard(1) on 2 ranks.
        Tensor: [100, 200].
        """
        plan = _compute_transfer_plan(
            global_shape=[100, 200],
            src_mesh_shape=[2],
            src_placements=[_Shard(0)],
            dst_mesh_shape=[2],
            dst_placements=[_Shard(1)],
        )
        # src0: (0-50, 0-200), src1: (50-100, 0-200)
        # dst0: (0-100, 0-100), dst1: (0-100, 100-200)
        # Overlaps:
        #   src0 & dst0: (0-50, 0-100)
        #   src0 & dst1: (0-50, 100-200)
        #   src1 & dst0: (50-100, 0-100)
        #   src1 & dst1: (50-100, 100-200)
        assert len(plan.transfers) == 4

    def test_dp_tp_training_to_tp_inference(self):
        """Realistic scenario: FSDP+TP training -> TP-only inference.

        Training: 2D mesh [DP=2, TP=2], placements [Shard(0), Shard(1)]
        Inference: 1D mesh [TP=2], placements [Shard(1)]
        Tensor: [8, 16] (small for clarity)
        """
        plan = _compute_transfer_plan(
            global_shape=[8, 16],
            src_mesh_shape=[2, 2],
            src_placements=[_Shard(0), _Shard(1)],
            dst_mesh_shape=[2],
            dst_placements=[_Shard(1)],
        )
        # src: rank0=(0-4,0-8), rank1=(0-4,8-16),
        #      rank2=(4-8,0-8), rank3=(4-8,8-16)
        # dst: rank0=(0-8,0-8), rank1=(0-8,8-16)
        #
        # dst0 needs (0-8, 0-8): from src0 (0-4,0-8) and src2 (4-8,0-8)
        # dst1 needs (0-8, 8-16): from src1 (0-4,8-16) and src3 (4-8,8-16)
        assert len(plan.transfers) == 4

        dst0_recvs = plan.recvs_for_rank(0)
        assert len(dst0_recvs) == 2
        assert sorted(t.src_rank for t in dst0_recvs) == [0, 2]

        dst1_recvs = plan.recvs_for_rank(1)
        assert len(dst1_recvs) == 2
        assert sorted(t.src_rank for t in dst1_recvs) == [1, 3]

    def test_empty_shard(self):
        """Tensor smaller than mesh -> some ranks get empty shards."""
        plan = _compute_transfer_plan(
            global_shape=[2],
            src_mesh_shape=[4],
            src_placements=[_Shard(0)],
            dst_mesh_shape=[1],
            dst_placements=[_Replicate()],
        )
        # src ranks 2,3 have empty shards -> no transfers from them
        transfers_with_data = [
            t for t in plan.transfers
            if all(s.start < s.stop for s in t.global_slices)
        ]
        assert len(transfers_with_data) == 2
        src_ranks = sorted(t.src_rank for t in transfers_with_data)
        assert src_ranks == [0, 1]


class TestTransferPlanSimulation:
    """End-to-end simulation: use the transfer plan to actually move tensor data."""

    def test_shard4_to_shard2(self):
        """Simulate Shard(0) on 4 ranks -> Shard(0) on 2 ranks."""
        full = torch.arange(100, dtype=torch.float32)

        # Create local shards for src (4 ranks)
        src_shards = list(full.chunk(4))

        plan = _compute_transfer_plan(
            global_shape=[100],
            src_mesh_shape=[4],
            src_placements=[_Shard(0)],
            dst_mesh_shape=[2],
            dst_placements=[_Shard(0)],
        )

        # Simulate: allocate dst buffers and fill via the plan
        dst_shards = [torch.zeros(50) for _ in range(2)]
        for t in plan.transfers:
            src_data = src_shards[t.src_rank][t.src_slices[0]]
            dst_shards[t.dst_rank][t.dst_slices[0]] = src_data

        # Verify
        expected = list(full.chunk(2))
        for i in range(2):
            assert torch.equal(dst_shards[i], expected[i]), (
                f"dst rank {i}: expected {expected[i]}, got {dst_shards[i]}"
            )

    def test_2d_to_1d_simulation(self):
        """Simulate [Shard(0), Shard(1)] on 2x2 -> [Shard(0)] on 2."""
        full = torch.arange(200, dtype=torch.float32).reshape(10, 20)

        # src shards: 2x2 mesh, [Shard(0), Shard(1)]
        top = full[:5]
        bottom = full[5:]
        src_shards = {
            0: top[:, :10],   # (0-5, 0-10)
            1: top[:, 10:],   # (0-5, 10-20)
            2: bottom[:, :10],  # (5-10, 0-10)
            3: bottom[:, 10:],  # (5-10, 10-20)
        }

        plan = _compute_transfer_plan(
            global_shape=[10, 20],
            src_mesh_shape=[2, 2],
            src_placements=[_Shard(0), _Shard(1)],
            dst_mesh_shape=[2],
            dst_placements=[_Shard(0)],
        )

        # dst shards: 2 ranks, Shard(0) -> each gets 5 rows, full 20 cols
        dst_shards = {
            0: torch.zeros(5, 20),
            1: torch.zeros(5, 20),
        }

        for t in plan.transfers:
            src_data = src_shards[t.src_rank][t.src_slices]
            dst_shards[t.dst_rank][t.dst_slices] = src_data

        assert torch.equal(dst_shards[0], full[:5])
        assert torch.equal(dst_shards[1], full[5:])

    def test_replicate_to_shard_simulation(self):
        """Simulate Replicate on 2 -> Shard(0) on 4."""
        full = torch.arange(100, dtype=torch.float32)

        plan = _compute_transfer_plan(
            global_shape=[100],
            src_mesh_shape=[2],
            src_placements=[_Replicate()],
            dst_mesh_shape=[4],
            dst_placements=[_Shard(0)],
        )

        # src: only rank 0 used (deduplicated)
        src_full = full.clone()

        dst_shards = [torch.zeros(25) for _ in range(4)]
        for t in plan.transfers:
            src_data = src_full[t.src_slices[0]]
            dst_shards[t.dst_rank][t.dst_slices[0]] = src_data

        expected = list(full.chunk(4))
        for i in range(4):
            assert torch.equal(dst_shards[i], expected[i])

    def test_uneven_simulation(self):
        """Simulate uneven Shard(0): 10 elements, 3 src ranks -> 2 dst ranks."""
        full = torch.arange(10, dtype=torch.float32)
        src_shards = list(full.chunk(3))  # [4, 4, 2]

        plan = _compute_transfer_plan(
            global_shape=[10],
            src_mesh_shape=[3],
            src_placements=[_Shard(0)],
            dst_mesh_shape=[2],
            dst_placements=[_Shard(0)],
        )

        dst_shards = list(full.chunk(2))  # [5, 5]
        dst_buffers = [torch.zeros_like(s) for s in dst_shards]

        for t in plan.transfers:
            src_data = src_shards[t.src_rank][t.src_slices[0]]
            dst_buffers[t.dst_rank][t.dst_slices[0]] = src_data

        for i in range(2):
            assert torch.equal(dst_buffers[i], dst_shards[i]), (
                f"rank {i}: expected {dst_shards[i]}, got {dst_buffers[i]}"
            )


# ---------------------------------------------------------------------------
# ShardingDescriptor
# ---------------------------------------------------------------------------


class TestShardingDescriptor:
    def test_sharded(self):
        desc = ShardingDescriptor(
            mesh_shape=(4,),
            placements=(_Shard(0),),
            logical_shape=torch.Size([100]),
        )
        assert desc.mesh_shape == (4,)
        assert desc.logical_shape == torch.Size([100])
        assert desc.rank_map is None

    def test_replicated(self):
        desc = ShardingDescriptor(
            mesh_shape=(4,),
            placements=(_Replicate(),),
            logical_shape=torch.Size([100, 200]),
        )
        assert desc.mesh_shape == (4,)
        assert desc.logical_shape == torch.Size([100, 200])

    def test_with_rank_map(self):
        rank_map = {(0,): 10, (1,): 11}
        desc = ShardingDescriptor(
            mesh_shape=(2,),
            placements=(_Shard(0),),
            logical_shape=torch.Size([100]),
            rank_map=rank_map,
        )
        assert desc.rank_map == rank_map

    def test_frozen(self):
        desc = ShardingDescriptor(
            mesh_shape=(2,),
            placements=(_Shard(0),),
            logical_shape=torch.Size([50]),
        )
        with pytest.raises(AttributeError):
            desc.mesh_shape = (4,)

    def test_2d_mesh(self):
        desc = ShardingDescriptor(
            mesh_shape=(2, 4),
            placements=(_Shard(0), _Shard(1)),
            logical_shape=torch.Size([100, 200]),
        )
        assert desc.mesh_shape == (2, 4)
        assert len(desc.placements) == 2


# ---------------------------------------------------------------------------
# ChunkTransfer properties
# ---------------------------------------------------------------------------


class TestChunkTransferProperties:
    def test_numel_1d(self):
        t = _ChunkTransfer(
            src_rank=0,
            dst_rank=1,
            src_slices=(slice(0, 25),),
            dst_slices=(slice(0, 25),),
            global_slices=(slice(0, 25),),
        )
        assert t.numel == 25

    def test_numel_2d(self):
        t = _ChunkTransfer(
            src_rank=0,
            dst_rank=1,
            src_slices=(slice(0, 5), slice(0, 10)),
            dst_slices=(slice(0, 5), slice(0, 10)),
            global_slices=(slice(0, 5), slice(0, 10)),
        )
        assert t.numel == 50

    def test_nbytes(self):
        t = _ChunkTransfer(
            src_rank=0,
            dst_rank=1,
            src_slices=(slice(0, 10),),
            dst_slices=(slice(0, 10),),
            global_slices=(slice(0, 10),),
        )
        assert t.nbytes(itemsize=4) == 40

    def test_repr(self):
        t = _ChunkTransfer(
            src_rank=0,
            dst_rank=1,
            src_slices=(slice(0, 25),),
            dst_slices=(slice(0, 25),),
            global_slices=(slice(0, 25),),
        )
        r = repr(t)
        assert "src=0" in r
        assert "dst=1" in r


# ---------------------------------------------------------------------------
# execute_transfer_plan (simulated with mock backend)
# ---------------------------------------------------------------------------


class _MockBackend:
    """In-process mock backend that routes sends/recvs through a shared dict.

    All ranks run in a single process. Call ``run_rank(rank, fn)`` for each
    simulated rank, then call ``drain()`` to execute them sequentially.
    """

    def __init__(self):
        self._mailbox: dict[tuple[int, int, int], torch.Tensor] = {}
        self._tag_counter: dict[int, int] = {}

    def for_rank(self, rank: int) -> "_RankView":
        return _RankView(self, rank)


class _RankView:
    """Per-rank view of the mock backend."""

    def __init__(self, parent: _MockBackend, rank: int):
        self._parent = parent
        self._rank = rank
        self._send_seq: dict[int, int] = {}
        self._recv_seq: dict[int, int] = {}

    def send_tensor(self, tensor: torch.Tensor, dst: int, *, tag: int = 0) -> None:
        seq = self._send_seq.get(dst, 0)
        key = (self._rank, dst, seq)
        self._send_seq[dst] = seq + 1
        self._parent._mailbox[key] = tensor.clone()

    def recv_tensor(self, tensor: torch.Tensor, src: int, *, tag: int = 0) -> None:
        seq = self._recv_seq.get(src, 0)
        key = (src, self._rank, seq)
        self._recv_seq[src] = seq + 1
        tensor.copy_(self._parent._mailbox[key])

    def send_object(self, obj, dst: int) -> None:
        raise NotImplementedError

    def recv_object(self, src: int):
        raise NotImplementedError


class TestExecuteTransferPlan:
    """Test execute_transfer_plan with the mock backend."""

    def test_shard4_to_shard2(self):
        """Shard(0) on 4 src ranks -> Shard(0) on 2 dst ranks."""
        full = torch.arange(100, dtype=torch.float32)
        src_shards = list(full.chunk(4))

        plan = _compute_transfer_plan(
            global_shape=[100],
            src_mesh_shape=[4],
            src_placements=[_Shard(0)],
            dst_mesh_shape=[2],
            dst_placements=[_Shard(0)],
        )

        mock = _MockBackend()

        # Run senders first (they populate the mailbox)
        for rank in range(4):
            execute_transfer_plan(
                plan, src_shards[rank], None, rank, mock.for_rank(rank)
            )

        # Run receivers
        dst_buffers = [torch.zeros(50) for _ in range(2)]
        for rank in range(2):
            execute_transfer_plan(
                plan, None, dst_buffers[rank], rank, mock.for_rank(rank)
            )

        expected = list(full.chunk(2))
        for i in range(2):
            assert torch.equal(dst_buffers[i], expected[i])

    def test_2d_to_1d(self):
        """[Shard(0), Shard(1)] on 2x2 -> [Shard(0)] on 2."""
        full = torch.arange(200, dtype=torch.float32).reshape(10, 20)

        src_shards = {
            0: full[:5, :10].contiguous(),
            1: full[:5, 10:].contiguous(),
            2: full[5:, :10].contiguous(),
            3: full[5:, 10:].contiguous(),
        }

        plan = _compute_transfer_plan(
            global_shape=[10, 20],
            src_mesh_shape=[2, 2],
            src_placements=[_Shard(0), _Shard(1)],
            dst_mesh_shape=[2],
            dst_placements=[_Shard(0)],
        )

        mock = _MockBackend()

        for rank in range(4):
            execute_transfer_plan(
                plan, src_shards[rank], None, rank, mock.for_rank(rank)
            )

        dst_buffers = {
            0: torch.zeros(5, 20),
            1: torch.zeros(5, 20),
        }
        for rank in range(2):
            execute_transfer_plan(
                plan, None, dst_buffers[rank], rank, mock.for_rank(rank)
            )

        assert torch.equal(dst_buffers[0], full[:5])
        assert torch.equal(dst_buffers[1], full[5:])

    def test_replicate_to_shard(self):
        """Replicate on 2 -> Shard(0) on 4."""
        full = torch.arange(100, dtype=torch.float32)

        plan = _compute_transfer_plan(
            global_shape=[100],
            src_mesh_shape=[2],
            src_placements=[_Replicate()],
            dst_mesh_shape=[4],
            dst_placements=[_Shard(0)],
        )

        mock = _MockBackend()

        # Only rank 0 sends (dedup)
        for rank in range(2):
            execute_transfer_plan(
                plan, full.clone(), None, rank, mock.for_rank(rank)
            )

        dst_buffers = [torch.zeros(25) for _ in range(4)]
        for rank in range(4):
            execute_transfer_plan(
                plan, None, dst_buffers[rank], rank, mock.for_rank(rank)
            )

        expected = list(full.chunk(4))
        for i in range(4):
            assert torch.equal(dst_buffers[i], expected[i])


# ---------------------------------------------------------------------------
# ModelTransferPlan
# ---------------------------------------------------------------------------


class TestModelTransferPlan:
    def _make_descriptors(self, name, shape, src_placements, dst_placements,
                          src_mesh_shape, dst_mesh_shape):
        """Helper to build src/dst descriptors for a single param."""
        src_desc = ShardingDescriptor(
            mesh_shape=src_mesh_shape,
            placements=tuple(src_placements),
            logical_shape=torch.Size(shape),
        )
        dst_desc = ShardingDescriptor(
            mesh_shape=dst_mesh_shape,
            placements=tuple(dst_placements),
            logical_shape=torch.Size(shape),
        )
        return {name: src_desc}, {name: dst_desc}

    def test_build_single_param(self):
        """Build a plan for one parameter."""
        src_params, dst_params = self._make_descriptors(
            "weight", [100], [_Shard(0)], [_Shard(0)],
            src_mesh_shape=(4,), dst_mesh_shape=(2,),
        )
        plan = ModelTransferPlan.build(src_params, dst_params)
        assert len(plan.param_plans) == 1
        pp = plan.param_plans[0]
        assert pp.src_name == "weight"
        assert pp.dst_name == "weight"
        assert pp.strategy == "optimal"
        assert len(pp.plan.transfers) == 4

    def test_build_same_sharding(self):
        """Same sharding on both sides -> direct_copy strategy."""
        src_params, dst_params = self._make_descriptors(
            "weight", [100], [_Shard(0)], [_Shard(0)],
            src_mesh_shape=(2,), dst_mesh_shape=(2,),
        )
        plan = ModelTransferPlan.build(src_params, dst_params)
        assert plan.param_plans[0].strategy == "direct_copy"

    def test_build_with_transform(self):
        """Parameter with transform -> materialize strategy."""
        src_params, dst_params = self._make_descriptors(
            "weight", [100], [_Shard(0)], [_Shard(0)],
            src_mesh_shape=(4,), dst_mesh_shape=(2,),
        )
        plan = ModelTransferPlan.build(
            src_params, dst_params,
            transforms={"weight": lambda t: t[:90]},
        )
        assert plan.param_plans[0].strategy == "materialize"

    def test_name_mapping_dict(self):
        """Name mapping via dict."""
        src = ShardingDescriptor(
            mesh_shape=(2,), placements=(_Shard(0),),
            logical_shape=torch.Size([100]),
        )
        dst = ShardingDescriptor(
            mesh_shape=(2,), placements=(_Shard(0),),
            logical_shape=torch.Size([100]),
        )
        plan = ModelTransferPlan.build(
            {"megatron.weight": src},
            {"hf.weight": dst},
            name_mapping={"megatron.weight": "hf.weight"},
        )
        pp = plan.param_plans[0]
        assert pp.src_name == "megatron.weight"
        assert pp.dst_name == "hf.weight"

    def test_name_mapping_callable(self):
        """Name mapping via callable."""
        src = ShardingDescriptor(
            mesh_shape=(2,), placements=(_Shard(0),),
            logical_shape=torch.Size([100]),
        )
        dst = ShardingDescriptor(
            mesh_shape=(2,), placements=(_Shard(0),),
            logical_shape=torch.Size([100]),
        )
        plan = ModelTransferPlan.build(
            {"module.weight": src},
            {"model.weight": dst},
            name_mapping=lambda n: n.replace("module.", "model."),
        )
        assert plan.param_plans[0].dst_name == "model.weight"

    def test_batching(self):
        """Small buffer_size forces multiple batches."""
        src_params = {}
        dst_params = {}
        for i in range(5):
            name = f"layer.{i}.weight"
            desc = ShardingDescriptor(
                mesh_shape=(2,), placements=(_Shard(0),),
                logical_shape=torch.Size([100]),
            )
            src_params[name] = desc
            dst_params[name] = desc

        # 100 float32 elements = 400 bytes per param
        # buffer_size=800 -> 2 params per batch -> 3 batches for 5 params
        plan = ModelTransferPlan.build(
            src_params, dst_params, buffer_size=800,
        )
        assert len(plan.batches) == 3
        assert len(plan.batches[0]) == 2
        assert len(plan.batches[1]) == 2
        assert len(plan.batches[2]) == 1

    def test_total_bytes(self):
        """total_bytes computes sum of transfer sizes."""
        src_params, dst_params = self._make_descriptors(
            "weight", [100], [_Shard(0)], [_Shard(0)],
            src_mesh_shape=(4,), dst_mesh_shape=(2,),
        )
        plan = ModelTransferPlan.build(src_params, dst_params)
        # 4 transfers of 25 elements each -> 100 elements * 4 bytes
        assert plan.total_bytes == 400

    def test_per_rank_bytes(self):
        """per_rank_bytes tracks bytes per rank."""
        src_params, dst_params = self._make_descriptors(
            "weight", [100], [_Shard(0)], [_Shard(0)],
            src_mesh_shape=(4,), dst_mesh_shape=(2,),
        )
        plan = ModelTransferPlan.build(src_params, dst_params)
        prb = plan.per_rank_bytes
        # src ranks 0-3 each send 25 elements = 100 bytes
        for r in range(4):
            assert r in prb

    def test_summary(self):
        """summary() returns a human-readable string."""
        src_params, dst_params = self._make_descriptors(
            "weight", [100], [_Shard(0)], [_Shard(0)],
            src_mesh_shape=(4,), dst_mesh_shape=(2,),
        )
        plan = ModelTransferPlan.build(src_params, dst_params)
        s = plan.summary()
        assert "ModelTransferPlan" in s
        assert "1 parameters" in s
        assert "optimal" in s.lower() or "Strategy C" in s

    def test_execute_shard4_to_shard2(self):
        """End-to-end execute with mock backend, non-overlapping rank maps."""
        full = torch.arange(100, dtype=torch.float32)
        src_shards = list(full.chunk(4))

        src_rank_map = {(0,): 0, (1,): 1, (2,): 2, (3,): 3}
        dst_rank_map = {(0,): 10, (1,): 11}

        src_params = {
            "w": ShardingDescriptor(
                mesh_shape=(4,), placements=(_Shard(0),),
                logical_shape=torch.Size([100]),
                rank_map=src_rank_map,
            )
        }
        dst_params = {
            "w": ShardingDescriptor(
                mesh_shape=(2,), placements=(_Shard(0),),
                logical_shape=torch.Size([100]),
                rank_map=dst_rank_map,
            )
        }
        mtp = ModelTransferPlan.build(src_params, dst_params)
        mock = _MockBackend()

        # Senders (src ranks 0-3)
        for rank in range(4):
            mtp.execute(
                {"w": src_shards[rank]}, rank=rank, backend=mock.for_rank(rank)
            )

        # Receivers (dst ranks 10-11)
        results = {}
        for i, rank in enumerate([10, 11]):
            r = mtp.execute({}, rank=rank, backend=mock.for_rank(rank))
            results[i] = r["w"]

        expected = list(full.chunk(2))
        for i in range(2):
            assert torch.equal(results[i], expected[i])

    def test_execute_with_transform(self):
        """Execute with a per-parameter transform."""
        full = torch.arange(100, dtype=torch.float32)
        src_shards = list(full.chunk(2))

        src_rank_map = {(0,): 0, (1,): 1}
        dst_rank_map = {(0,): 10, (1,): 11}

        src_params = {
            "w": ShardingDescriptor(
                mesh_shape=(2,), placements=(_Shard(0),),
                logical_shape=torch.Size([100]),
                rank_map=src_rank_map,
            )
        }
        dst_params = {
            "w": ShardingDescriptor(
                mesh_shape=(2,), placements=(_Shard(0),),
                logical_shape=torch.Size([100]),
                rank_map=dst_rank_map,
            )
        }
        transform_called = []

        def truncate(t):
            transform_called.append(True)
            return t

        mtp = ModelTransferPlan.build(
            src_params, dst_params,
            transforms={"w": truncate},
        )
        assert mtp.param_plans[0].strategy == "materialize"

        mock = _MockBackend()
        for rank in range(2):
            mtp.execute(
                {"w": src_shards[rank]}, rank=rank, backend=mock.for_rank(rank)
            )
        assert len(transform_called) == 2

    def test_multiple_params(self):
        """Build and execute with multiple parameters."""
        src_rank_map = {(0,): 0, (1,): 1}
        dst_rank_map = {(0,): 10, (1,): 11}

        src_params = {
            "weight": ShardingDescriptor(
                mesh_shape=(2,), placements=(_Shard(0),),
                logical_shape=torch.Size([100]),
                rank_map=src_rank_map,
            ),
            "bias": ShardingDescriptor(
                mesh_shape=(2,), placements=(_Shard(0),),
                logical_shape=torch.Size([20]),
                rank_map=src_rank_map,
            ),
        }
        dst_params = {
            "weight": ShardingDescriptor(
                mesh_shape=(2,), placements=(_Shard(0),),
                logical_shape=torch.Size([100]),
                rank_map=dst_rank_map,
            ),
            "bias": ShardingDescriptor(
                mesh_shape=(2,), placements=(_Shard(0),),
                logical_shape=torch.Size([20]),
                rank_map=dst_rank_map,
            ),
        }
        plan = ModelTransferPlan.build(src_params, dst_params)
        assert len(plan.param_plans) == 2

        mock = _MockBackend()
        full_w = torch.arange(100, dtype=torch.float32)
        full_b = torch.arange(20, dtype=torch.float32)

        # Send phase (src ranks 0-1)
        for rank in range(2):
            plan.execute(
                {
                    "weight": list(full_w.chunk(2))[rank],
                    "bias": list(full_b.chunk(2))[rank],
                },
                rank=rank,
                backend=mock.for_rank(rank),
            )

        # Recv phase (dst ranks 10-11)
        for i, rank in enumerate([10, 11]):
            result = plan.execute({}, rank=rank, backend=mock.for_rank(rank))
            assert "weight" in result
            assert "bias" in result
            assert torch.equal(result["weight"], list(full_w.chunk(2))[i])
            assert torch.equal(result["bias"], list(full_b.chunk(2))[i])


# ---------------------------------------------------------------------------
# Framework adapter tests (with mock parameters)
# ---------------------------------------------------------------------------


class _MockParam:
    """Mock parameter for adapter tests."""

    def __init__(self, shape, **attrs):
        self.shape = shape
        self.data = torch.zeros(shape)
        for k, v in attrs.items():
            setattr(self, k, v)

    def named_parameters(self):
        raise NotImplementedError


class _MockModel:
    """Mock model that yields named parameters."""

    def __init__(self, params: dict):
        self._params = params

    def named_parameters(self):
        return iter(self._params.items())


class TestVLLMSharding:
    def test_column_parallel(self):
        from tensordict.dtensor_adapters.vllm import VLLMSharding

        adapter = VLLMSharding(tp_size=4, tp_rank=0)
        param = _MockParam([256, 1024], output_dim=0)
        desc = adapter.describe("linear.weight", param)
        assert desc.mesh_shape == (4,)
        assert desc.logical_shape == torch.Size([1024, 1024])

    def test_row_parallel(self):
        from tensordict.dtensor_adapters.vllm import VLLMSharding

        adapter = VLLMSharding(tp_size=2, tp_rank=0)
        param = _MockParam([1024, 512], input_dim=1)
        desc = adapter.describe("linear.weight", param)
        assert desc.mesh_shape == (2,)
        assert desc.logical_shape == torch.Size([1024, 1024])

    def test_replicated(self):
        from tensordict.dtensor_adapters.vllm import VLLMSharding

        adapter = VLLMSharding(tp_size=4, tp_rank=0)
        param = _MockParam([100])
        desc = adapter.describe("norm.weight", param)
        assert desc.mesh_shape == (4,)
        assert desc.logical_shape == torch.Size([100])

    def test_describe_model(self):
        from tensordict.dtensor_adapters.vllm import VLLMSharding

        adapter = VLLMSharding(tp_size=2, tp_rank=0)
        model = _MockModel({
            "qkv.weight": _MockParam([384, 1024], output_dim=0),
            "norm.weight": _MockParam([1024]),
        })
        descs = adapter.describe_model(model)
        assert len(descs) == 2
        assert descs["qkv.weight"].logical_shape == torch.Size([768, 1024])
        assert descs["norm.weight"].logical_shape == torch.Size([1024])

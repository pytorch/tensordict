# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Public API for cross-mesh DTensor transfer plan computation.

This module re-exports the core types and functions from the private
``tensordict._dtensor`` module under stable, public names.

The transfer plan computation is **pure math** -- no GPUs, no
``torch.distributed``, no actual tensors required. It takes shapes
and placement descriptors and produces a list of P2P transfer operations.

Example::

    from tensordict.dtensor import compute_transfer_plan, ShardingDescriptor
    from torch.distributed.tensor.placement_types import Shard

    plan = compute_transfer_plan(
        global_shape=[100],
        src_mesh_shape=[4],
        src_placements=[Shard(0)],
        dst_mesh_shape=[2],
        dst_placements=[Shard(0)],
    )
    print(plan)

"""

from tensordict._dtensor import (
    _chunk_slice as chunk_slice,
    _ChunkTransfer as ChunkTransfer,
    _compute_all_local_slices as compute_all_local_slices,
    _compute_local_slices as compute_local_slices,
    _compute_transfer_plan as compute_transfer_plan,
    _intersect_1d as intersect_1d,
    _intersect_slices as intersect_slices,
    _ShardSpec as ShardSpec,
    _slice_relative_to as slice_relative_to,
    _TransferPlan as TransferPlan,
    execute_transfer_plan,
    ShardingDescriptor,
)

__all__ = [
    "ChunkTransfer",
    "ShardingDescriptor",
    "ShardSpec",
    "TransferPlan",
    "chunk_slice",
    "compute_all_local_slices",
    "compute_local_slices",
    "compute_transfer_plan",
    "execute_transfer_plan",
    "intersect_1d",
    "intersect_slices",
    "slice_relative_to",
]

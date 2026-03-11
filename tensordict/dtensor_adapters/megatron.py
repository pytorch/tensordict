# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Adapter for building ShardingDescriptors from Megatron-LM parameters.

Megatron parameters carry sharding metadata as custom attributes:
- ``tensor_model_parallel``: bool
- ``partition_dim``: int (0 for column-parallel, 1 for row-parallel)
- ``partition_stride``: int (always 1 in practice)
- ``parallel_mode``: str | None ("duplicated", etc.)

For MoE models, Megatron also uses expert parallelism (EP) where experts
are split across EP ranks, creating a 2D sharding (EP x TP).
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch

from tensordict._dtensor import ShardingDescriptor


class MegatronSharding:
    """Build ShardingDescriptors from Megatron parameter metadata.

    Args:
        tp_group: the tensor-parallel process group. Used to determine
            TP world size and rank-to-global-rank mapping.
        ep_group: optional expert-parallel process group for MoE models.
        pp_rank: this rank's pipeline-parallel rank (default 0).
        pp_size: total number of pipeline-parallel stages (default 1).
    """

    def __init__(
        self,
        tp_group,
        ep_group=None,
        pp_rank: int = 0,
        pp_size: int = 1,
    ):
        from torch import distributed as dist

        self._tp_size = dist.get_world_size(tp_group)
        self._tp_group = tp_group
        self._tp_ranks = list(range(dist.get_world_size(tp_group)))
        self._tp_rank_map = {(i,): dist.get_global_rank(tp_group, i) for i in self._tp_ranks}

        self._ep_group = ep_group
        self._ep_size = dist.get_world_size(ep_group) if ep_group is not None else 1
        self._pp_rank = pp_rank
        self._pp_size = pp_size

    def describe(self, name: str, param) -> ShardingDescriptor:
        """Create a ShardingDescriptor from a Megatron parameter.

        Args:
            name: parameter name (for diagnostics).
            param: the parameter tensor. Expected to have Megatron's
                custom attributes (``tensor_model_parallel``,
                ``partition_dim``, etc.).
        """
        from torch.distributed.tensor.placement_types import Replicate, Shard

        is_tp = getattr(param, "tensor_model_parallel", False)
        partition_dim = getattr(param, "partition_dim", 0)

        if not is_tp:
            return ShardingDescriptor(
                mesh_shape=(self._tp_size,),
                placements=(Replicate(),),
                logical_shape=torch.Size(param.shape),
                rank_map=self._tp_rank_map,
            )

        # Compute the logical (unsharded) shape from the local shard
        local_shape = list(param.shape)
        logical_shape = list(local_shape)
        logical_shape[partition_dim] = local_shape[partition_dim] * self._tp_size

        return ShardingDescriptor(
            mesh_shape=(self._tp_size,),
            placements=(Shard(partition_dim),),
            logical_shape=torch.Size(logical_shape),
            rank_map=self._tp_rank_map,
        )

    def describe_model(
        self,
        model,
        named_parameters_fn: Callable | None = None,
    ) -> dict[str, ShardingDescriptor]:
        """Describe all parameters in a Megatron model.

        Args:
            model: the Megatron model.
            named_parameters_fn: optional callable that returns an iterator
                of ``(name, param)`` pairs. Defaults to
                ``model.named_parameters()``.
        """
        params_iter = (
            named_parameters_fn(model)
            if named_parameters_fn is not None
            else model.named_parameters()
        )
        return {name: self.describe(name, param) for name, param in params_iter}

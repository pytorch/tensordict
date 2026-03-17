# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Adapter for building ShardingDescriptors from vLLM parameters.

vLLM uses column-parallel (Shard(0)) and row-parallel (Shard(1))
for linear layers. QKV projections are often merged into a single
parameter. The model's ``BasevLLMParameter`` stores ``tp_rank`` and
``tp_size`` on each parameter.
"""

from __future__ import annotations

import torch

from tensordict._dtensor import ShardingDescriptor


class VLLMSharding:
    """Build ShardingDescriptors from vLLM parameter metadata.

    Args:
        tp_size: tensor-parallel world size.
        tp_rank: this worker's tensor-parallel rank.
        rank_map: optional mapping from mesh coords to global ranks.
            If ``None``, ranks are ``0..tp_size-1``.
    """

    def __init__(
        self,
        tp_size: int,
        tp_rank: int,
        rank_map: dict[tuple[int, ...], int] | None = None,
    ):
        self._tp_size = tp_size
        self._tp_rank = tp_rank
        self._rank_map = rank_map

    def describe(self, name: str, param) -> ShardingDescriptor:
        """Create a ShardingDescriptor from a vLLM parameter.

        vLLM parameters typically have ``output_dim`` (for column-parallel)
        or ``input_dim`` (for row-parallel) attributes that indicate which
        dimension is sharded.

        Args:
            name: parameter name.
            param: the parameter tensor.
        """
        from torch.distributed.tensor.placement_types import Replicate, Shard

        output_dim = getattr(param, "output_dim", None)
        input_dim = getattr(param, "input_dim", None)

        if output_dim is not None:
            shard_dim = output_dim
        elif input_dim is not None:
            shard_dim = input_dim
        else:
            return ShardingDescriptor(
                mesh_shape=(self._tp_size,),
                placements=(Replicate(),),
                logical_shape=torch.Size(param.shape),
                rank_map=self._rank_map,
            )

        local_shape = list(param.shape)
        logical_shape = list(local_shape)
        logical_shape[shard_dim] = local_shape[shard_dim] * self._tp_size

        return ShardingDescriptor(
            mesh_shape=(self._tp_size,),
            placements=(Shard(shard_dim),),
            logical_shape=torch.Size(logical_shape),
            rank_map=self._rank_map,
        )

    def describe_model(self, model) -> dict[str, ShardingDescriptor]:
        """Describe all parameters in a vLLM model."""
        return {
            name: self.describe(name, param) for name, param in model.named_parameters()
        }

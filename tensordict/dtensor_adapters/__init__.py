# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Framework adapters for constructing ShardingDescriptors.

These adapters extract sharding metadata from framework-specific
parameter objects and produce :class:`~tensordict._dtensor.ShardingDescriptor`
instances suitable for :class:`~tensordict._dtensor.ModelTransferPlan`.

Each adapter is in its own submodule to avoid hard dependencies on the
respective framework.
"""

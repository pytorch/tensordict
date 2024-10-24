# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from tensordict.tensorclass import tensorclass
from tensordict.tensordict import TensorDict


@tensorclass
class min:
    """A `min` tensorclass to be used as a result for :meth:`~tensordict.TensorDict.min` operations."""

    vals: TensorDict
    indices: TensorDict


@tensorclass
class max:
    """A `max` tensorclass to be used as a result for :meth:`~tensordict.TensorDict.max` operations."""

    vals: TensorDict
    indices: TensorDict


@tensorclass
class cummin:
    """A `cummin` tensorclass to be used as a result for :meth:`~tensordict.TensorDict.cummin` operations."""

    vals: TensorDict
    indices: TensorDict


@tensorclass
class cummax:
    """A `cummax` tensorclass to be used as a result for :meth:`~tensordict.TensorDict.cummax` operations."""

    vals: TensorDict
    indices: TensorDict

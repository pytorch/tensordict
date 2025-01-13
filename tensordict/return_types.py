# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

from tensordict.tensorclass import tensorclass
from tensordict.tensordict import TensorDict


@tensorclass
class min:
    """A `min` tensorclass to be used as a result for :meth:`~tensordict.TensorDict.min` operations."""

    vals: TensorDict
    indices: TensorDict

    def __post_init__(self):
        warnings.warn(
            f"{type(self)}.min is deprecated and will be removed in v0.9. "
            f"Use torch.return_types.min instead.",
            category=DeprecationWarning,
        )


@tensorclass
class max:
    """A `max` tensorclass to be used as a result for :meth:`~tensordict.TensorDict.max` operations."""

    vals: TensorDict
    indices: TensorDict

    def __post_init__(self):
        warnings.warn(
            f"{type(self)}.max is deprecated and will be removed in v0.9. "
            f"Use torch.return_types.max instead.",
            category=DeprecationWarning,
        )


@tensorclass
class cummin:
    """A `cummin` tensorclass to be used as a result for :meth:`~tensordict.TensorDict.cummin` operations."""

    vals: TensorDict
    indices: TensorDict

    def __post_init__(self):
        warnings.warn(
            f"{type(self)}.cummin is deprecated and will be removed in v0.9. "
            f"Use torch.return_types.cummin instead.",
            category=DeprecationWarning,
        )


@tensorclass
class cummax:
    """A `cummax` tensorclass to be used as a result for :meth:`~tensordict.TensorDict.cummax` operations."""

    vals: TensorDict
    indices: TensorDict

    def __post_init__(self):
        warnings.warn(
            f"{type(self)}.cummax is deprecated and will be removed in v0.9. "
            f"Use torch.return_types.cummax instead.",
            category=DeprecationWarning,
        )

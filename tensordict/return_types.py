# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from tensordict.tensorclass import tensorclass
from tensordict.tensordict import TensorDict


@tensorclass(shadow=True)
class min:
    """A `min` tensorclass to be used as a result for :meth:`~tensordict.TensorDict.min` operations."""

    values: TensorDict
    indices: TensorDict

    def __getitem__(self, item):
        try:
            return (self.values, self.indices)[item]
        except IndexError:
            raise IndexError(
                f"Indexing a {type(self)} element follows the torch.return_types.{type(self).__name__}'s "
                f"__getitem__ method API."
            )


@tensorclass(shadow=True)
class max:
    """A `max` tensorclass to be used as a result for :meth:`~tensordict.TensorDict.max` operations."""

    values: TensorDict
    indices: TensorDict

    def __getitem__(self, item):
        try:
            return (self.values, self.indices)[item]
        except IndexError:
            raise IndexError(
                f"Indexing a {type(self)} element follows the torch.return_types.{type(self).__name__}'s "
                f"__getitem__ method API."
            )


@tensorclass(shadow=True)
class cummin:
    """A `cummin` tensorclass to be used as a result for :meth:`~tensordict.TensorDict.cummin` operations."""

    values: TensorDict
    indices: TensorDict

    def __getitem__(self, item):
        try:
            return (self.values, self.indices)[item]
        except IndexError:
            raise IndexError(
                f"Indexing a {type(self)} element follows the torch.return_types.{type(self).__name__}'s "
                f"__getitem__ method API."
            )


@tensorclass(shadow=True)
class cummax:
    """A `cummax` tensorclass to be used as a result for :meth:`~tensordict.TensorDict.cummax` operations."""

    values: TensorDict
    indices: TensorDict

    def __getitem__(self, item):
        try:
            return (self.values, self.indices)[item]
        except IndexError:
            raise IndexError(
                f"Indexing a {type(self)} element follows the torch.return_types.{type(self).__name__}'s "
                f"__getitem__ method API."
            )

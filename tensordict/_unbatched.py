# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from functools import wraps
from typing import Any, Callable, TYPE_CHECKING

import torch
from tensordict._tensorcollection import TensorCollection
from tensordict.base import TensorDictBase

from tensordict.tensorclass import (
    _arg_to_tensordict,
    _from_tensordict_with_copy,
    _TD_PASS_THROUGH,
    TD_HANDLED_FUNCTIONS,
    TensorClass,
)
from tensordict.utils import (
    _getitem_batch_size,
    _is_tensorclass,
    _maybe_correct_neg_dim,
    IndexType,
    unravel_key,
)
from torch import Tensor

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any


def _arg_to_tensordict_unbatched(arg, batch_size):
    if _is_tensorclass(type(arg)):
        arg = arg._tensordict.empty()
        arg.batch_size = batch_size
        return arg
    elif isinstance(arg, (tuple, list)) and all(
        _is_tensorclass(type(item)) for item in arg
    ):
        arg_list = []
        for item in arg:
            item = item._tensordict.empty()
            item.batch_size = batch_size
            arg_list.append(item)

        return type(arg)(arg_list)
    return arg


def _bypass(func):
    @wraps(func)
    def bypassed_func(self, *args, **kwargs):
        meta_tensor = torch.zeros(
            self.batch_size, dtype=self.dtype, device=torch.device("meta")
        )
        name = func.__name__
        r = getattr(meta_tensor, name)(*args, **kwargs)
        self_copy = self.copy()
        self_copy.batch_size = r.shape
        return self_copy

    return bypassed_func


_TORCH_SHAPE_OPS = (
    torch.gather,
    torch.unbind,
    torch.cat,
    torch.stack,
    torch.unflatten,
    torch.flatten,
    torch.split,
    torch.squeeze,
    torch.unsqueeze,
)


class UnbatchedTensor(TensorClass):
    """A TensorClass that represents a tensor whose shape is ignored during shape operations.

    This class allows tensors to be stored in a TensorDict without enforcing batch size consistency.
    Shape operations (e.g., reshape, unsqueeze, squeeze) on the TensorDict will return the same UnbatchedTensor instance,
    while other operations (e.g., apply, key manipulation, pointwise arithmetic) may modify the underlying tensor content.

    Example:
        >>> td = TensorDict(a=UnbatchedTensor(torch.randn(3, 4)), b=torch.randn(2, 3), batch_size=(2,))
        >>> td_reshaped = td.reshape((1, 2))
        >>> td_reshaped["a"] is td["a"]
        True

    Note that accessing an UnbatchedTensor using `get()` and `__getitem__()` will return different results.
    `get()` returns the UnbatchedTensor instance, while `__getitem__()` returns the underlying tensor content.

    Example:
        >>> td.get("a")
        <UnbatchedTensor ...>
        >>> td["a"]
        tensor([[...]])

    """

    data: torch.Tensor | TensorDictBase
    _pass_through = True

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Callable:
        if func not in _TD_PASS_THROUGH or not all(
            issubclass(t, (Tensor, cls, TensorDictBase)) for t in types
        ):
            return NotImplemented

        if kwargs is None:
            kwargs = {}

        # get the output type from the arguments / keyword arguments
        if len(args) > 0:
            tensorclass_instance = args[0]
        else:
            tensorclass_instance = kwargs.get("input", kwargs["tensors"])
        if isinstance(tensorclass_instance, (tuple, list)):
            tensorclass_instance = tensorclass_instance[0]

        if func not in _TORCH_SHAPE_OPS:
            args = tuple(_arg_to_tensordict(arg) for arg in args)
            kwargs = {key: _arg_to_tensordict(value) for key, value in kwargs.items()}
            result = TD_HANDLED_FUNCTIONS[func](*args, **kwargs)
        else:
            # Get a brute force batch size
            args = tuple(
                _arg_to_tensordict_unbatched(arg, tensorclass_instance.batch_size)
                for arg in args
            )
            kwargs = {
                key: _arg_to_tensordict_unbatched(
                    value, tensorclass_instance.batch_size
                )
                for key, value in kwargs.items()
            }
            example_td = TD_HANDLED_FUNCTIONS[func](*args, **kwargs)
            result = tensorclass_instance.copy()
            result.batch_size = example_td.batch_size
            return result

        if isinstance(result, (list, tuple)):
            return type(result)(
                _from_tensordict_with_copy(tensorclass_instance, tensordict_result)
                for tensordict_result in result
            )
        return _from_tensordict_with_copy(tensorclass_instance, result)

    def chunk(self, chunks: int, dim: int | None = None):
        self_copy = self.copy()
        if dim is None:
            dim = 0
        dim = _maybe_correct_neg_dim(dim, self.batch_size)
        self_copy.batch_size = (
            self.batch_size[:dim]
            + (self.batch_size[dim] // chunks,)
            + self.batch_size[dim + 1 :]
        )
        return self_copy

    def split(self, split_size: int | list[int], dim: int | None = None):
        self_copy = self.copy()
        if dim is None:
            dim = 0
        dim = _maybe_correct_neg_dim(dim, self.batch_size)
        chunks = (
            len(split_size)
            if isinstance(split_size, (list, tuple))
            else -(self.batch_size[dim] // -split_size)
        )
        self_copy.batch_size = (
            self.batch_size[:dim]
            + (self.batch_size[dim] // chunks,)
            + self.batch_size[dim + 1 :]
        )
        return self_copy

    def __getitem__(self, index: IndexType) -> Self | Tensor | TensorCollection | Any:
        if isinstance(index, (tuple, str)) and unravel_key(index):
            raise ValueError(
                "TensorClass fields must be accessed as attributes, not items."
            )
        self_copy = self.copy()
        self_copy.batch_size = _getitem_batch_size(self.batch_size, index)
        return self_copy

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self.__dict__["_batch_size"] = torch.Size(batch_size)

    shape = batch_size

    def unbind(self, dim: int):
        return tuple(
            self[(slice(None),) * dim + (0,)] for _ in range(self.batch_size[dim])
        )

    @_bypass
    def reshape(self, *shape): ...

    @_bypass
    def view(self, *shape): ...

    def unsqueeze(self, dim: int):
        shape = list(self.batch_size)
        shape.insert(dim, 0)
        self_copy = self.copy()
        self_copy.batch_size = shape
        return self_copy

    def transpose(self, dim0, dim1):
        batch_size = list(self.batch_size)
        batch_size[dim1], batch_size[dim0] = batch_size[dim0], batch_size[dim1]
        self_copy = self.copy()
        self_copy.batch_size = batch_size
        return self_copy

    def permute(self, *dims):
        if len(dims) == 1 and not isinstance(dims[0], int):
            return self.permute(*dims[0])
        batch_size = list(self.batch_size)
        batch_size = [batch_size[d] for d in dims]
        self_copy = self.copy()
        self_copy.batch_size = batch_size
        return self_copy

    @classmethod
    def _stack_non_tensor(
        cls, list_of_non_tensor, dim: int = 0, raise_if_non_unique=False
    ):
        result = list_of_non_tensor[0].copy()
        batch_size = list(result.batch_size)
        batch_size.insert(dim, len(list_of_non_tensor))
        result.batch_size = torch.Size(batch_size)
        return result

    @_bypass
    def unflatten(self, dim, unflattened_size): ...

    @_bypass
    def flatten(self, start_dim: int = 0, end_dim=-1): ...

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

    **Indexed Assignment Behavior:**

    UnbatchedTensor is not affected by indexed assignment operations on the parent TensorDict.
    Since UnbatchedTensor does not follow batch dimensions, operations like ``td[:, :2] = other``
    will skip the UnbatchedTensor entries entirely.

    Example:
        >>> td = TensorDict(a=torch.ones(4, 3), unbatched=UnbatchedTensor(torch.ones(7, 11)), batch_size=(4, 3))
        >>> td[:, :2] = TensorDict(a=torch.zeros(4, 2), batch_size=(4, 2))
        >>> td["a"][:, :2]  # Regular tensor is modified
        tensor([[0., 0.], ...])
        >>> td["unbatched"]  # UnbatchedTensor is unchanged
        tensor([[1., 1., ...], ...])

    **Batch Size Computation:**

    UnbatchedTensor is excluded from ``auto_batch_size_()`` computation. The batch size is determined
    solely by regular tensors, and UnbatchedTensor simply adopts the resulting batch size.

    .. note::
        Pointwise arithmetic (``+``, ``-``, ``*``, ``/``) on TensorDicts containing UnbatchedTensor
        is supported. The operation is applied to the UnbatchedTensor's data while preserving its
        batch_size property.

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
            # Find the first UnbatchedTensor in the list
            for item in tensorclass_instance:
                if isinstance(item, cls):
                    tensorclass_instance = item
                    break
            else:
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

        # Preserve batch_size for non-shape operations
        source_batch_size = tensorclass_instance.batch_size

        if isinstance(result, (list, tuple)):
            out = []
            for tensordict_result in result:
                item = _from_tensordict_with_copy(tensorclass_instance, tensordict_result)
                item.batch_size = source_batch_size
                out.append(item)
            return type(result)(out)
        out = _from_tensordict_with_copy(tensorclass_instance, result)
        out.batch_size = source_batch_size
        return out

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

    @property
    def shape(self):
        return self.data.shape

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

    def _wrap_result(self, result):
        """Wrap a tensor result back into UnbatchedTensor preserving batch_size."""
        if isinstance(result, torch.Tensor) and not isinstance(result, UnbatchedTensor):
            out = UnbatchedTensor(result)
            out.batch_size = self.batch_size
            return out
        return result

    def __add__(self, other):
        return self._wrap_result(self.data + (other.data if isinstance(other, UnbatchedTensor) else other))

    def __radd__(self, other):
        return self._wrap_result(other + self.data)

    def __sub__(self, other):
        return self._wrap_result(self.data - (other.data if isinstance(other, UnbatchedTensor) else other))

    def __rsub__(self, other):
        return self._wrap_result(other - self.data)

    def __mul__(self, other):
        return self._wrap_result(self.data * (other.data if isinstance(other, UnbatchedTensor) else other))

    def __rmul__(self, other):
        return self._wrap_result(other * self.data)

    def __truediv__(self, other):
        return self._wrap_result(self.data / (other.data if isinstance(other, UnbatchedTensor) else other))

    def __rtruediv__(self, other):
        return self._wrap_result(other / self.data)

    def __floordiv__(self, other):
        return self._wrap_result(self.data // (other.data if isinstance(other, UnbatchedTensor) else other))

    def __rfloordiv__(self, other):
        return self._wrap_result(other // self.data)

    def __pow__(self, other):
        return self._wrap_result(self.data ** (other.data if isinstance(other, UnbatchedTensor) else other))

    def __rpow__(self, other):
        return self._wrap_result(other ** self.data)

    def __neg__(self):
        return self._wrap_result(-self.data)

    def __pos__(self):
        return self._wrap_result(+self.data)

    def __abs__(self):
        return self._wrap_result(abs(self.data))

    def add(self, other, *, alpha=None):
        if alpha is not None:
            return self._wrap_result(self.data.add(other.data if isinstance(other, UnbatchedTensor) else other, alpha=alpha))
        return self + other

    def sub(self, other, *, alpha=None):
        if alpha is not None:
            return self._wrap_result(self.data.sub(other.data if isinstance(other, UnbatchedTensor) else other, alpha=alpha))
        return self - other

    def mul(self, other):
        return self * other

    def div(self, other):
        return self / other

    def to(self, *args, **kwargs):
        """Casts the UnbatchedTensor to the specified device/dtype, preserving batch_size."""
        result = UnbatchedTensor(self.data.to(*args, **kwargs))
        result.batch_size = self.batch_size
        return result

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

    def clone(self, recurse: bool = True):
        """Clones the UnbatchedTensor, preserving the batch_size."""
        if recurse:
            result = type(self)(self.data.clone())
        else:
            result = type(self)(self.data)
        result.batch_size = self.batch_size
        return result

    def _clone(self, recurse: bool = True):
        """Internal clone method, preserving the batch_size."""
        return self.clone(recurse=recurse)

    def backward(self, *args, **kwargs):
        """Delegates backward to the underlying data tensor."""
        return self.data.backward(*args, **kwargs)

    @property
    def grad(self):
        """Returns the gradient of the underlying data tensor, wrapped in UnbatchedTensor."""
        data_grad = self.data.grad
        if data_grad is None:
            return None
        result = UnbatchedTensor(data_grad)
        result.batch_size = self.batch_size
        return result

    @grad.setter
    def grad(self, value):
        """Sets the gradient of the underlying data tensor."""
        if isinstance(value, UnbatchedTensor):
            self.data.grad = value.data
        else:
            self.data.grad = value

    def untyped_storage(self):
        """Returns the untyped storage of the underlying data tensor."""
        return self.data.untyped_storage()

    def data_ptr(self):
        """Returns the data pointer of the underlying data tensor."""
        return self.data.data_ptr()

    @property
    def device(self):
        """Returns the device of the underlying data tensor."""
        return self.data.device

    @property
    def dtype(self):
        """Returns the dtype of the underlying data tensor."""
        return self.data.dtype

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Returns a state dict containing the data and batch_size."""
        import collections

        out = collections.OrderedDict()
        if not keep_vars:
            out[prefix + "data"] = self.data.detach().clone()
        else:
            out[prefix + "data"] = self.data
        out[prefix + "__batch_size"] = self.batch_size
        out[prefix + "__is_unbatched"] = True
        if destination is not None:
            destination.update(out)
            return destination
        return out

    @classmethod
    def from_state_dict(cls, state_dict, prefix=""):
        """Creates an UnbatchedTensor from a state dict."""
        data = state_dict[prefix + "data"]
        batch_size = state_dict[prefix + "__batch_size"]
        result = cls(data)
        result.batch_size = batch_size
        return result

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Loads a state dict into the UnbatchedTensor."""
        data = state_dict.get("data")
        batch_size = state_dict.get("__batch_size", self.batch_size)
        if data is not None:
            if assign:
                self._tensordict.set("data", data)
            else:
                self.data.copy_(data)
        self.batch_size = batch_size
        return self

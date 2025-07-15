# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import torch
from tensordict.base import TensorDictBase

from tensordict.tensorclass import TensorClass
from tensordict.utils import (
    _create_segments_from_int,
    _create_segments_from_list,
    _getitem_batch_size,
    _is_tensorclass,
    _maybe_correct_neg_dim,
    unravel_key,
)
from torch import Tensor


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

    def _passthrough_pre(self):
        copy = self.copy()
        copy.data = None
        return copy

    def reshape(self, *shape, **kwargs):
        copy = self._passthrough_pre()
        copy = super(UnbatchedTensor, copy).reshape(*shape, **kwargs)
        copy.data = self.data
        return copy

    def view(self, *shape, **kwargs):
        copy = self._passthrough_pre()
        copy = super(UnbatchedTensor, copy).view(*shape, **kwargs)
        copy.data = self.data
        return copy

    def squeeze(self, *args, **kwargs):
        copy = self._passthrough_pre()
        copy = super(UnbatchedTensor, copy).squeeze(*args, **kwargs)
        copy.data = self.data
        return copy

    def flatten(self, *args, **kwargs):
        copy = self._passthrough_pre()
        copy = super(UnbatchedTensor, copy).flatten(*args, **kwargs)
        copy.data = self.data
        return copy

    def unflatten(self, *args, **kwargs):
        copy = self._passthrough_pre()
        copy = super(UnbatchedTensor, copy).unflatten(*args, **kwargs)
        copy.data = self.data
        return copy

    def permute(self, *args, **kwargs):
        copy = self._passthrough_pre()
        copy = super(UnbatchedTensor, copy).permute(*args, **kwargs)
        copy.data = self.data
        return copy

    def transpose(self, *args, **kwargs):
        copy = self._passthrough_pre()
        copy = super(UnbatchedTensor, copy).transpose(*args, **kwargs)
        copy.data = self.data
        return copy

    def repeat(self, *args, **kwargs):
        copy = self._passthrough_pre()
        copy = super(UnbatchedTensor, copy).repeat(*args, **kwargs)
        copy.data = self.data
        return copy

    def expand(self, *args, **kwargs):
        copy = self._passthrough_pre()
        copy = super(UnbatchedTensor, copy).expand(*args, **kwargs)
        copy.data = self.data
        return copy

    def split(self, split_size, dim=0):
        copy = self._passthrough_pre()
        copies = super(UnbatchedTensor, copy).split(split_size, dim)
        for copy in copies:
            copy.data = self.data
        return copies

    def chunk(self, *args, **kwargs): 
        copy = self._passthrough_pre()
        copies = self._passthrough_pre(UnbatchedTensor, copy).chunk(*args, **kwargs)
        for copy in copies:
            copy.data = self.data
        return copies
    def unbind(self, dim=0):  # type: ignore[override]
        copy = self._passthrough_pre()
        copies = self._passthrough_pre(UnbatchedTensor, copy).unbind(*args, **kwargs)
        for copy in copies:
            copy.data = self.data
        return copies
    def _wrap_result(self, result):
        # Only wrap if result is a torch.Tensor, else raise
        if not isinstance(result, torch.Tensor):
            raise NotImplementedError("UnbatchedTensor only supports torch.Tensor arithmetic/device ops.")
        return type(self)(result, batch_size=self.batch_size)
    def __add__(self, other):  # type: ignore[override]
        return self._wrap_result(self._tensordict.get("data") + self._get_data(other))
    def __sub__(self, other):  # type: ignore[override]
        return self._wrap_result(self._tensordict.get("data") - self._get_data(other))
    def __mul__(self, other):  # type: ignore[override]
        return self._wrap_result(self._tensordict.get("data") * self._get_data(other))
    def __truediv__(self, other):  # type: ignore[override]
        return self._wrap_result(self._tensordict.get("data") / self._get_data(other))
    def __floordiv__(self, other):  # type: ignore[override]
        left = self._tensordict.get("data")
        right = self._get_data(other)
        if not (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)):
            raise NotImplementedError("UnbatchedTensor floordiv only supports torch.Tensor operands.")
        return self._wrap_result(left // right)
    def __pow__(self, other):  # type: ignore[override]
        return self._wrap_result(self._tensordict.get("data") ** self._get_data(other))
    def __neg__(self):  # type: ignore[override]
        return self._wrap_result(-self._tensordict.get("data"))
    def __abs__(self):  # type: ignore[override]
        return self._wrap_result(abs(self._tensordict.get("data")))
    def __radd__(self, other):  # type: ignore[override]
        return self._wrap_result(self._get_data(other) + self._tensordict.get("data"))
    def __rsub__(self, other):  # type: ignore[override]
        return self._wrap_result(self._get_data(other) - self._tensordict.get("data"))
    def __rmul__(self, other):  # type: ignore[override]
        return self._wrap_result(self._get_data(other) * self._tensordict.get("data"))
    def __rtruediv__(self, other):  # type: ignore[override]
        return self._wrap_result(self._get_data(other) / self._tensordict.get("data"))
    def __rfloordiv__(self, other):  # type: ignore[override]
        left = self._get_data(other)
        right = self._tensordict.get("data")
        if not (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)):
            raise NotImplementedError("UnbatchedTensor floordiv only supports torch.Tensor operands.")
        return self._wrap_result(left // right)
    def __rpow__(self, other):  # type: ignore[override]
        return self._wrap_result(self._get_data(other) ** self._tensordict.get("data"))
    def _get_data(self, other):
        if isinstance(other, UnbatchedTensor):
            return other._tensordict.get("data")
        if isinstance(other, torch.Tensor):
            return other
        if isinstance(other, (int, float)):
            # Convert scalars to tensors for arithmetic operations
            return torch.tensor(other, dtype=self._tensordict.get("data").dtype, device=self._tensordict.get("data").device)
        # Do not allow TensorDictBase or anything else
        raise NotImplementedError("UnbatchedTensor arithmetic only supports torch.Tensor, UnbatchedTensor, or scalar operands.")

    # --- Device/dtype ops ---
    def to(self, *args, **kwargs):  # type: ignore[override]
        return self._wrap_result(self._tensordict.get("data").to(*args, **kwargs))
    def cpu(self, *args, **kwargs):  # type: ignore[override]
        return self._wrap_result(self._tensordict.get("data").cpu(*args, **kwargs))
    def cuda(self, *args, **kwargs):  # type: ignore[override]
        return self._wrap_result(self._tensordict.get("data").cuda(*args, **kwargs))
    def float(self, *args, **kwargs):
        return self._wrap_result(self._tensordict.get("data").float(*args, **kwargs))
    def double(self, *args, **kwargs):
        return self._wrap_result(self._tensordict.get("data").double(*args, **kwargs))
    def int(self, *args, **kwargs):
        return self._wrap_result(self._tensordict.get("data").int(*args, **kwargs))
    def long(self, *args, **kwargs):
        if not isinstance(self._tensordict.get("data"), torch.Tensor):
            raise NotImplementedError("UnbatchedTensor.long only supports torch.Tensor data.")
        return self._wrap_result(self._tensordict.get("data").long(*args, **kwargs))
    def half(self, *args, **kwargs):
        if not isinstance(self._tensordict.get("data"), torch.Tensor):
            raise NotImplementedError("UnbatchedTensor.half only supports torch.Tensor data.")
        return self._wrap_result(self._tensordict.get("data").half(*args, **kwargs))
    def bfloat16(self, *args, **kwargs):  # type: ignore[override]
        return self._wrap_result(self._tensordict.get("data").bfloat16(*args, **kwargs))
    def type(self, *args, **kwargs):
        return self._wrap_result(self._tensordict.get("data").type(*args, **kwargs))
    @property
    def device(self):
        return self._tensordict.get("data").device
    @device.setter
    def device(self, value):
        # Accepts DeviceType, returns None, matches base class signature
        raise NotImplementedError("Setting device is not supported for UnbatchedTensor.")
    @property
    def dtype(self):
        return self._tensordict.get("data").dtype

    # --- __torch_function__ for torch ops ---
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        if kwargs is None:
            kwargs = {}
        shape_ops = {
            torch.reshape, torch.squeeze, torch.unsqueeze, torch.flatten, torch.unflatten,
            torch.permute, torch.transpose, torch.repeat_interleave,
            torch.cat, torch.stack, torch.split, torch.chunk, torch.unbind,
        }
        if func in shape_ops:
            instance = args[0] if args else kwargs.get('input', None)
            if instance is not None and isinstance(instance, UnbatchedTensor):
                return instance._shape_passthrough(*args[1:], **kwargs)
        arithmetic_ops = {
            torch.add, torch.sub, torch.mul, torch.div, torch.true_divide, torch.floor_divide,
            torch.pow, torch.neg, torch.abs,
        }
        device_ops = {torch.float, torch.double, torch.int, torch.long, torch.half, torch.bfloat16, torch.cuda, torch.cpu}
        if func in arithmetic_ops or func in device_ops:
            instance = args[0] if args else kwargs.get('input', None)
            if instance is not None and isinstance(instance, UnbatchedTensor):
                new_args = tuple(a._tensordict.get("data") if isinstance(a, UnbatchedTensor) else a for a in args[1:])
                result = func(instance._tensordict.get("data"), *new_args, **kwargs)
                return instance._wrap_result(result)
        # If we don't recognize the function, return NotImplemented to let Python fall back
        return NotImplemented

    # --- copy method ---
    def copy(self):
        # Create a new UnbatchedTensor with the same data (not cloned)
        return type(self)(self._tensordict.get("data"), batch_size=self.batch_size)

    # --- __getitem__ returns self (same instance) for all indexing operations ---
    def __getitem__(self, index):
        return self

    def _getitem(self, item):
        return self

    # --- repr for debugging ---
    def __repr__(self):
        return f"UnbatchedTensor(data={self._tensordict.get('data')!r}, batch_size={self.batch_size})"

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import collections
import math
import typing
from numbers import Number
from typing import Tuple, List, Union, Any, Optional

import numpy as np
import torch
from torch import Tensor

try:
    try:
        from functorch._C import is_batchedtensor, get_unwrapped
    except ImportError:
        from torch._C._functorch import is_batchedtensor, get_unwrapped

except ImportError:
    pass

INDEX_TYPING = Union[None, int, slice, str, Tensor, List[Any], Tuple[Any, ...]]
DEVICE_TYPING = Union[torch.device, str, int]
if hasattr(typing, "get_args"):
    DEVICE_TYPING_ARGS = typing.get_args(DEVICE_TYPING)
else:
    DEVICE_TYPING_ARGS = (torch.device, str, int)

NESTED_KEY = Union[str, Tuple[str, ...]]


def _sub_index(tensor: torch.Tensor, idx: INDEX_TYPING) -> torch.Tensor:
    """Allows indexing of tensors with nested tuples.

     >>> sub_tensor1 = tensor[tuple1][tuple2]
     >>> sub_tensor2 = _sub_index(tensor, (tuple1, tuple2))
     >>> assert torch.allclose(sub_tensor1, sub_tensor2)

    Args:
        tensor (torch.Tensor): tensor to be indexed.
        idx (tuple of indices): indices sequence to be used.

    """
    if isinstance(idx, tuple) and len(idx) and isinstance(idx[0], tuple):
        idx0 = idx[0]
        idx1 = idx[1:]
        return _sub_index(_sub_index(tensor, idx0), idx1)
    return tensor[idx]


def _getitem_batch_size(
    shape: torch.Size,
    items: INDEX_TYPING,
) -> torch.Size:
    """Given an input shape and an index, returns the size of the resulting indexed tensor.

    This function is aimed to be used when indexing is an
    expensive operation.
    Args:
        shape (torch.Size): Input shape
        items (index): Index of the hypothetical tensor

    Returns:
        Size of the resulting object (tensor or tensordict)
    """
    # let's start with simple cases
    if isinstance(items, tuple) and len(items) == 1:
        items = items[0]
    if isinstance(items, int):
        return shape[1:]
    if isinstance(items, torch.Tensor) and items.dtype is torch.bool:
        return torch.Size([items.sum(), *shape[items.ndimension() :]])
    if (
        isinstance(items, (torch.Tensor, np.ndarray)) and len(items.shape) <= 1
    ) or isinstance(items, list):
        if len(items):
            return torch.Size([len(items), *shape[1:]])
        else:
            return shape[1:]

    if not isinstance(items, tuple):
        items = (items,)
    bs = []
    iter_bs = iter(shape)
    if all(isinstance(_item, torch.Tensor) for _item in items) and len(items) == len(
        shape
    ):
        shape0 = items[0].shape
        for _item in items[1:]:
            if _item.shape != shape0:
                raise RuntimeError(
                    f"all tensor indices must have the same shape, "
                    f"got {_item.shape} and {shape0}"
                )
        return shape0

    for _item in items:
        if isinstance(_item, slice):
            batch = next(iter_bs)
            v = len(range(*_item.indices(batch)))
        elif isinstance(_item, (list, torch.Tensor, np.ndarray)):
            batch = next(iter_bs)
            if isinstance(_item, torch.Tensor) and _item.dtype is torch.bool:
                v = _item.sum()
            else:
                v = len(_item)
        elif _item is None:
            v = 1
        elif isinstance(_item, Number):
            try:
                batch = next(iter_bs)
            except StopIteration:
                raise RuntimeError(
                    f"The shape {shape} is incompatible with " f"the index {items}."
                )
            continue
        else:
            raise NotImplementedError(
                f"batch dim cannot be computed for type {type(_item)}"
            )
        bs.append(v)
    list_iter_bs = list(iter_bs)
    bs += list_iter_bs
    return torch.Size(bs)


def convert_ellipsis_to_idx(idx: Union[Tuple, Ellipsis], batch_size: List[int]):
    """Given an index containing an ellipsis or just an ellipsis, converts any ellipsis to slice(None).

    Example:
        >>> idx = (..., 0)
        >>> batch_size = [1,2,3]
        >>> new_index = convert_ellipsis_to_idx(idx, batch_size)
        >>> print(new_index)
        (slice(None, None, None), slice(None, None, None), 0)

    Args:
        idx (tuple, Ellipsis): Input index
        batch_size (list): Shape of tensor to be indexed

    Returns:
        new_index (tuple): Output index
    """
    new_index = ()
    num_dims = len(batch_size)

    if idx is Ellipsis:
        idx = (...,)
    num_ellipsis = sum(_idx is Ellipsis for _idx in idx)
    if num_dims < (len(idx) - num_ellipsis):
        raise RuntimeError("Not enough dimensions in TensorDict for index provided.")

    start_pos, after_ellipsis_length = None, 0
    for i, item in enumerate(idx):
        if item is Ellipsis:
            if start_pos is not None:
                raise RuntimeError("An index can only have one ellipsis at most.")
            else:
                start_pos = i
        if item is not Ellipsis and start_pos is not None:
            after_ellipsis_length += 1

    before_ellipsis_length = start_pos
    ellipsis_length = num_dims - after_ellipsis_length - before_ellipsis_length

    new_index += idx[:start_pos]

    ellipsis_start = start_pos
    ellipsis_end = start_pos + ellipsis_length
    new_index += (slice(None),) * (ellipsis_end - ellipsis_start)

    new_index += idx[start_pos + 1 : start_pos + 1 + after_ellipsis_length]

    if len(new_index) != num_dims:
        raise RuntimeError(
            f"The new index {new_index} is incompatible with the dimensions of the batch size {num_dims}."
        )

    return new_index


def _copy(self: List[int]):
    out: List[int] = []
    for elem in self:
        out.append(elem)
    return out


def infer_size_impl(shape: List[int], numel: int) -> List[int]:
    """Infers the shape of an expanded tensor whose number of elements is indicated by :obj:`numel`.

    Copied from pytorch for compatibility issues (See #386).
    See https://github.com/pytorch/pytorch/blob/35d4fa444b67cbcbe34a862782ddf2d92f5b1ce7/torch/jit/_shape_functions.py
    for the original copy.

    """
    newsize = 1
    infer_dim: Optional[int] = None
    for dim in range(len(shape)):
        if shape[dim] == -1:
            if infer_dim is not None:
                raise AssertionError("only one dimension can be inferred")
            infer_dim = dim
        elif shape[dim] >= 0:
            newsize *= shape[dim]
        else:
            raise AssertionError("invalid shape dimensions")
    if not (
        numel == newsize
        or (infer_dim is not None and newsize > 0 and numel % newsize == 0)
    ):
        raise AssertionError("invalid shape")
    out = _copy(shape)
    if infer_dim is not None:
        out[infer_dim] = numel // newsize
    return out


def _get_shape(value):
    # we call it "legacy code"
    return value.shape


def _unwrap_value(value):
    # batch_dims = value.ndimension()
    if not isinstance(value, torch.Tensor):
        out = value
    elif is_batchedtensor(value):
        out = get_unwrapped(value)
    else:
        out = value
    return out
    # batch_dims = out.ndimension() - batch_dims
    # batch_size = out.shape[:batch_dims]
    # return out, batch_size


class KeyDependentDefaultDict(collections.defaultdict):
    """A key-dependent default dict.

    Examples:
        >>> my_dict = KeyDependentDefaultDict(lambda key: "foo_" + key)
        >>> print(my_dict["bar"])
        foo_bar
    """

    def __init__(self, fun):
        self.fun = fun
        super().__init__()

    def __missing__(self, key):
        value = self.fun(key)
        self[key] = value
        return value


if hasattr(math, "prod"):

    def prod(sequence):
        """General prod function, that generalised usage across math and np.

        Created for multiple python versions compatibility).

        """
        return math.prod(sequence)


else:

    def prod(sequence):
        """General prod function, that generalised usage across math and np.

        Created for multiple python versions compatibility).

        """
        return int(np.prod(sequence))


def expand_as_right(
    tensor: Union[torch.Tensor, "MemmapTensor", "TensorDictBase"],  # noqa: F821
    dest: Union[torch.Tensor, "MemmapTensor", "TensorDictBase"],  # noqa: F821
):
    """Expand a tensor on the right to match another tensor shape.

    Args:
        tensor: tensor to be expanded
        dest: tensor providing the target shape

    Returns:
         a tensor with shape matching the dest input tensor shape.

    Examples:
        >>> tensor = torch.zeros(3,4)
        >>> dest = torch.zeros(3,4,5)
        >>> print(expand_as_right(tensor, dest).shape)
        torch.Size([3,4,5])

    """
    if dest.ndimension() < tensor.ndimension():
        raise RuntimeError(
            "expand_as_right requires the destination tensor to have less "
            f"dimensions than the input tensor, got"
            f" tensor.ndimension()={tensor.ndimension()} and "
            f"dest.ndimension()={dest.ndimension()}"
        )
    if not (tensor.shape == dest.shape[: tensor.ndimension()]):
        raise RuntimeError(
            f"tensor shape is incompatible with dest shape, "
            f"got: tensor.shape={tensor.shape}, dest={dest.shape}"
        )
    for _ in range(dest.ndimension() - tensor.ndimension()):
        tensor = tensor.unsqueeze(-1)
    return tensor.expand(dest.shape)


def expand_right(
    tensor: Union[torch.Tensor, "MemmapTensor"], shape: Sequence[int]  # noqa: F821
) -> torch.Tensor:
    """Expand a tensor on the right to match a desired shape.

    Args:
        tensor: tensor to be expanded
        shape: target shape

    Returns:
         a tensor with shape matching the target shape.

    Examples:
        >>> tensor = torch.zeros(3,4)
        >>> shape = (3,4,5)
        >>> print(expand_right(tensor, shape).shape)
        torch.Size([3,4,5])

    """
    tensor_expand = tensor
    while tensor_expand.ndimension() < len(shape):
        tensor_expand = tensor_expand.unsqueeze(-1)
    tensor_expand = tensor_expand.expand(*shape)
    return tensor_expand


numpy_to_torch_dtype_dict = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
}
torch_to_numpy_dtype_dict = {
    value: key for key, value in numpy_to_torch_dtype_dict.items()
}


def _nested_key_type_check(key):
    is_tuple = isinstance(key, tuple)
    if not (
        isinstance(key, str)
        or (
            is_tuple and len(key) > 0 and all(isinstance(subkey, str) for subkey in key)
        )
    ):
        key_repr = (
            f"tuple({', '.join(str(type(i)) for i in key)})" if is_tuple else type(key)
        )
        raise TypeError(
            "Expected key to be a string or non-empty tuple of strings, but found "
            f"{key_repr}"
        )


def _normalize_key(key: NESTED_KEY) -> NESTED_KEY:
    # normalises tuples of length one to their string contents
    return key if not isinstance(key, tuple) or len(key) > 1 else key[0]

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

DIM_TYPING = Any
LEVELS_TYPING = Any
try:
    try:
        from functorch._C import is_batchedtensor, get_unwrapped
    except ImportError:
        from torch._C._functorch import is_batchedtensor, get_unwrapped

    try:
        import functorch.dim

        LEVELS_TYPING = Tuple[Union[int, functorch.dim.Dim], ...]
        DIM_TYPING = functorch.dim.Dim
        _has_functorch_dim = True
    except ImportError:
        _has_functorch_dim = False
except ImportError:
    pass

INDEX_TYPING = Union[None, int, slice, str, Tensor, List[Any], Tuple[Any, ...]]
DEVICE_TYPING = Union[torch.device, str, int]
if hasattr(typing, "get_args"):
    DEVICE_TYPING_ARGS = typing.get_args(DEVICE_TYPING)
else:
    DEVICE_TYPING_ARGS = (torch.device, str, int)


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
        elif _has_functorch_dim and _is_first_class_dim(_item):
            # any first class dimensions are ommited from the new batch size
            next(iter_bs)
            continue
        else:
            raise NotImplementedError(
                f"batch dim cannot be computed for type {type(_item)}"
            )
        bs.append(v)
    list_iter_bs = list(iter_bs)
    bs += list_iter_bs
    return torch.Size(bs)


def _is_first_class_dim(idx: INDEX_TYPING) -> bool:
    # return True if idx is a first-class dimension or a tuple of first-class dimensions
    return isinstance(idx, functorch.dim.Dim) or (
        isinstance(idx, tuple)
        and any(isinstance(item, functorch.dim.Dim) for item in idx)
    )


def _get_indexed_dims(
    idx: Tuple[INDEX_TYPING, ...], dims: Tuple[DIM_TYPING, ...], shape
) -> Tuple[DIM_TYPING, ...]:
    new_dims = []

    for item, size in zip(idx, shape):
        if isinstance(item, functorch.dim.Dim):
            # bind dimensions to the size of the dimensions they are indexing
            item.size = size
            new_dims.append(item)
        elif isinstance(item, tuple) and any(
            isinstance(d, functorch.dim.Dim) for d in item
        ):
            n_unbound = sum(not d.is_bound for d in item)
            if n_unbound > 1:
                raise functorch.dim.DimensionBindError(
                    f"cannot infer the sizes of {n_unbound} dimensions at once "
                    f"{tuple(d for d in item if not d.is_bound)}"
                )
            elif n_unbound == 1:
                d_size, rem = divmod(size, prod([d.size for d in item if d.is_bound]))
                if rem != 0:
                    raise functorch.dim.DimensionBindError(
                        "inferred dimension does not evenly fit into larger dimension: "
                        f"{size} vs "
                        f"{tuple(d.size if d.is_bound else '?' for d in item)}"
                    )

                (unbound_dim,) = tuple(d for d in item if not d.is_bound)
                unbound_dim.size = d_size
            else:
                size_prod = prod([d.size for d in item])
                if size_prod != size:
                    raise functorch.dim.DimensionBindError(
                        f"Dimension sizes do not match ({size} != {size_prod}) when "
                        f"matching dimension pack {item}"
                    )

            new_dims.extend(item)

    dims = dims + tuple(new_dims)

    # remove duplicate first-class dims. when first-class dims are reused the diagonal
    # entries are extracted and that dim only appears once in the resulting dims
    dims_out = []
    seen = set()
    for dim in dims:
        if dim not in seen:
            dims_out.append(dim)
        seen.add(dim)

    return tuple(dims_out)


def _is_in(d: DIM_TYPING, args: LEVELS_TYPING, nested: bool = False) -> bool:
    # FIXME: this is a workaround since equality comparisons with unbound
    # functorch.dim.Tensors results in a ValueError. Same as `d in args`
    if any(d is item for item in args):
        return True
    elif nested and any(isinstance(item, tuple) and _is_in(d, item) for item in args):
        return True
    return False


def _get_ordered_dims(
    dims: Tuple[DIM_TYPING, ...], args: LEVELS_TYPING
) -> Tuple[DIM_TYPING, ...]:
    return tuple(d for d in dims if not _is_in(d, args, nested=True))


def _get_ordered_shape(batch_size, args):
    def _parse_size(dim):
        if isinstance(dim, functorch.dim.Dim):
            return dim.size
        elif isinstance(dim, tuple):
            return prod([_parse_size(d) for d in dim])
        return batch_size[dim]

    # place all ordered dimensions at the front, dropping any re-ordered positional
    # arguments from the existing batch_size.
    positional_args = {arg for arg in args if isinstance(arg, int)}
    for arg in args:
        if isinstance(arg, tuple):
            positional_args.update({d for d in arg if isinstance(d, int)})
    return torch.Size(
        [_parse_size(d) for d in args]
        + [size for i, size in enumerate(batch_size) if i not in positional_args]
    )


def _reslice_without_first_class_dims(
    idx: INDEX_TYPING,
) -> Tuple[INDEX_TYPING, INDEX_TYPING]:
    # separates an index into parts with first-class dimensions and parts without
    if not isinstance(idx, tuple):
        idx = (idx,)

    idx_with = tuple(item if _is_first_class_dim(item) else slice(None) for item in idx)

    trim = 0
    for item in reversed(idx_with):
        if _is_first_class_dim(item):
            break
        trim += 1

    idx_without = tuple(item for item in idx if not _is_first_class_dim(item))

    if trim > 0:
        return idx_with[:-trim], idx_without

    return idx, idx_without


def _dims_are_compatible(dims1, dims2):
    return (
        len(dims1) == len(dims2)
        and all(_is_in(d, dims1) for d in dims2)
        and all(_is_in(d, dims2) for d in dims1)
    )


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

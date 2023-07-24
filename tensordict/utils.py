# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import inspect
import math
import time

import warnings
from collections import defaultdict
from collections.abc import KeysView
from copy import copy
from functools import wraps
from importlib import import_module
from typing import Any, Callable, List, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np
import torch

from packaging.version import parse
from tensordict._tensordict import (  # noqa: F401
    _unravel_key_to_tuple,  # noqa: F401
    unravel_key,  # noqa: F401
    unravel_key_list,  # noqa: F401
    unravel_keys,  # noqa: F401
)
from torch import Tensor

if TYPE_CHECKING:
    from tensordict.memmap import MemmapTensor
    from tensordict.tensordict import TensorDictBase

try:
    try:
        from functorch._C import get_unwrapped, is_batchedtensor
    except ImportError:
        from torch._C._functorch import get_unwrapped, is_batchedtensor
except ImportError:
    pass

try:
    from torchrec import KeyedJaggedTensor

    _has_torchrec = True
except ImportError as err:
    _has_torchrec = False

    class KeyedJaggedTensor:  # noqa: D101, D102
        pass

    TORCHREC_ERR = str(err)


IndexType = Union[None, int, slice, str, Tensor, List[Any], Tuple[Any, ...]]
DeviceType = Union[torch.device, str, int]
NestedKey = Union[str, Tuple[str, ...]]


def _sub_index(tensor: torch.Tensor, idx: IndexType) -> torch.Tensor:
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


def _getitem_batch_size(batch_size, index):
    """Given an input shape and an index, returns the size of the resulting indexed tensor.

    This function is aimed to be used when indexing is an
    expensive operation.
    Args:
        shape (torch.Size): Input shape
        items (index): Index of the hypothetical tensor

    Returns:
        Size of the resulting object (tensor or tensordict)

    Examples:
        >>> idx = (None, ..., None)
        >>> torch.zeros(4, 3, 2, 1)[idx].shape
        torch.Size([1, 4, 3, 2, 1, 1])
        >>> _getitem_batch_size([4, 3, 2, 1], idx)
        torch.Size([1, 4, 3, 2, 1, 1])
    """
    if not isinstance(index, tuple):
        if isinstance(index, int):
            return batch_size[1:]
        if isinstance(index, slice) and index == slice(None):
            return batch_size
        index = (index,)
    # index = convert_ellipsis_to_idx(index, batch_size)
    # broadcast shapes
    shapes_dict = {}
    look_for_disjoint = False
    disjoint = False
    bools = []
    for i, idx in enumerate(index):
        boolean = False
        if isinstance(idx, (range, list)):
            shape = len(idx)
        elif isinstance(idx, (torch.Tensor, np.ndarray)):
            if idx.dtype == torch.bool or idx.dtype == np.dtype("bool"):
                shape = torch.Size([idx.sum()])
                boolean = True
            else:
                shape = idx.shape
        elif isinstance(idx, slice):
            look_for_disjoint = not disjoint and (len(shapes_dict) > 0)
            shape = None
        else:
            shape = None
        if shape is not None:
            if look_for_disjoint:
                disjoint = True
            shapes_dict[i] = shape
        bools.append(boolean)
    bs_shape = None
    if shapes_dict:
        bs_shape = torch.broadcast_shapes(*shapes_dict.values())
    out = []
    count = -1
    for i, idx in enumerate(index):
        if idx is None:
            out.append(1)
            continue
        count += 1 if not bools[i] else idx.ndim
        if i in shapes_dict:
            if bs_shape is not None:
                if disjoint:
                    # the indices will be put at the beginning
                    out = list(bs_shape) + out
                else:
                    # if there is a single tensor or similar, we just extend
                    out.extend(bs_shape)
                bs_shape = None
            continue
        elif isinstance(idx, int):
            # could be spared for efficiency
            continue
        elif isinstance(idx, slice):
            batch = batch_size[count]
            out.append(len(range(*idx.indices(batch))))
    count += 1
    if batch_size[count:]:
        out.extend(batch_size[count:])
    return torch.Size(out)


# def _getitem_batch_size(shape: torch.Size, items: IndexType) -> torch.Size:
#     """Given an input shape and an index, returns the size of the resulting indexed tensor.
#
#     This function is aimed to be used when indexing is an
#     expensive operation.
#     Args:
#         shape (torch.Size): Input shape
#         items (index): Index of the hypothetical tensor
#
#     Returns:
#         Size of the resulting object (tensor or tensordict)
#
#     Examples:
#         >>> idx = (None, ..., None)
#         >>> torch.zeros(4, 3, 2, 1)[idx].shape
#         torch.Size([1, 4, 3, 2, 1, 1])
#         >>> _getitem_batch_size([4, 3, 2, 1], idx)
#         torch.Size([1, 4, 3, 2, 1, 1])
#     """
#     # let's start with simple cases
#     if isinstance(items, tuple) and len(items) == 1:
#         items = items[0]
#     if isinstance(items, int):
#         return shape[1:]
#     if isinstance(items, torch.Tensor) and items.dtype is torch.bool:
#         return torch.Size([items.sum(), *shape[items.ndimension() :]])
#     if (
#         isinstance(items, (torch.Tensor, np.ndarray)) and len(items.shape) <= 1
#     ) or isinstance(items, list):
#         if isinstance(items, torch.Tensor) and not items.shape:
#             return shape[1:]
#         if _is_lis_of_list_of_bools(items):
#             warnings.warn(
#                 "Got a list of list of bools: this indexing behaviour will be deprecated soon.",
#                 category=DeprecationWarning,
#             )
#             items = torch.tensor(items)
#             return torch.Size([items.sum(), *shape[items.ndimension() :]])
#         if len(items):
#             return torch.Size([len(items), *shape[1:]])
#         else:
#             return shape[1:]
#
#     if not isinstance(items, tuple):
#         items = (items,)
#
#     if any(item is Ellipsis for item in items):
#         items = convert_ellipsis_to_idx(items, shape)
#
#     sanitized_items = []
#     shapes = []
#     for _item in items:
#         if isinstance(_item, (list, range, )):
#             # _item = torch.tensor(_item)
#             shapes.append(torch.Size([len(_item)]))
#         elif isinstance(_item, np.ndarray):
#             shapes.append(torch.Size(np.shape))
#         elif isinstance(_item, torch.Tensor):
#             shapes.append(_item.shape)
#         else:
#             shapes.append(None)
#         if isinstance(_item, torch.Tensor) and _item.dtype is torch.bool:
#             # when using NumPy's advanced indexing patterns, any index containing a
#             # boolean array can be equivalently replaced with index.nonzero()
#             # note we add unbind(-1) since behaviour of numpy.ndarray.nonzero returns
#             # tuples of arrays whereas torch.Tensor.nonzero returns a single tensor
#             # https://numpy.org/doc/stable/user/basics.indexing.html#boolean-array-indexing
#             sanitized_items.extend(_item.nonzero().unbind(-1))
#         else:
#             sanitized_items.append(_item)
#
#     # when multiple tensor-like indices are present, they must be broadcastable onto a
#     # common shape. if this is satisfied then they are broadcast to that shape, and used
#     # to extract diagonal entries of the array.
#     # if the tensor indices are contiguous, or separated by scalars, they are replaced
#     # in-place by the broadcast shape. if they are separated by non-scalar indices, the
#     # broadcast shape is prepended to the new batch size
#     # https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
#     tensor_indices = []
#     contiguous, prev = True, None
#     for i, (_item, _shape) in enumerate(zip(sanitized_items, shapes)):
#         if isinstance(_item, (torch.Tensor, range, list, np.ndarray)):
#             tensor_indices.append(_item)
#             if prev is not None and i != prev + 1:
#                 contiguous = False
#             prev = i
#         elif isinstance(_item, Number) and prev is not None and i == prev + 1:
#             prev = i
#
#     bs = []
#     if tensor_indices:
#         try:
#             b_shape = torch.broadcast_shapes(*[shape for shape in shapes if shape])
#         except ValueError as err:
#             raise ValueError(
#                 "When indexing with tensor-like indices, each of those indices must be "
#                 f"broadcastable to a common shape. Got indices: {tensor_indices}."
#             ) from err
#         if not contiguous:
#             bs.extend(b_shape)
#
#     iter_bs = iter(shape)
#
#     cursor = -1
#     for _item in sanitized_items:
#         cursor += 1
#         if isinstance(_item, slice):
#             batch = next(iter_bs)
#             bs.append(len(range(*_item.indices(batch))))
#         elif isinstance(_item, range):
#             batch = next(iter_bs)
#             bs.append(min(batch, len(_item)))
#         elif isinstance(_item, (list, torch.Tensor, np.ndarray)):
#             batch = next(iter_bs)
#             if b is not None:
#                 # we haven't yet accounted for tensor indices, so we insert in-place
#                 bs.extend(b.shape)
#                 b = None
#         elif _item is None:
#             bs.append(1)
#         elif isinstance(_item, Number):
#             try:
#                 batch = next(iter_bs)
#             except StopIteration:
#                 raise RuntimeError(
#                     f"The shape {shape} is incompatible with " f"the index {items}."
#                 )
#             continue
#         elif _item is Ellipsis:
#             if cursor == len(sanitized_items) - 1:
#                 # then we can just skip
#                 continue
#             n_upcoming = len(sanitized_items) - cursor - 1
#             while cursor < len(shape) - n_upcoming:
#                 batch = next(iter_bs)
#                 bs.append(batch)
#                 cursor += 1
#         else:
#             raise NotImplementedError(
#                 f"batch dim cannot be computed for type {type(_item)}"
#             )
#
#     list_iter_bs = list(iter_bs)
#     bs += list_iter_bs
#     return torch.Size(bs)


def convert_ellipsis_to_idx(
    idx: tuple[int | Ellipsis] | Ellipsis, batch_size: list[int]
) -> tuple[int, ...]:
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
    istuple = isinstance(idx, tuple)
    if (not istuple and idx is not Ellipsis) or (
        istuple and all(_idx is not Ellipsis for _idx in idx)
    ):
        return idx
    new_index = ()
    num_dims = len(batch_size)

    if idx is Ellipsis:
        idx = (...,)

    num_ellipsis = sum(_idx is Ellipsis for _idx in idx)
    if num_dims < (len(idx) - num_ellipsis - sum(item is None for item in idx)):
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
        if item is None:
            # unsqueeze
            num_dims += 1

    before_ellipsis_length = start_pos
    if start_pos is None:
        return idx
    else:
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


def _copy(self: list[int]) -> list[int]:
    return list(self)


def infer_size_impl(shape: list[int], numel: int) -> list[int]:
    """Infers the shape of an expanded tensor whose number of elements is indicated by :obj:`numel`.

    Copied from pytorch for compatibility issues (See #386).
    See https://github.com/pytorch/pytorch/blob/35d4fa444b67cbcbe34a862782ddf2d92f5b1ce7/torch/jit/_shape_functions.py
    for the original copy.

    """
    newsize = 1
    infer_dim: int | None = None
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


def _unwrap_value(value: torch.Tensor) -> torch.Tensor:
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


if hasattr(math, "prod"):  # Python 3.8+

    def prod(sequence):
        """General prod function, that generalised usage across math and np.

        Created for multiple python versions compatibility.

        """
        return math.prod(sequence)

else:

    def prod(sequence):
        """General prod function, that generalised usage across math and np.

        Created for multiple python versions compatibility.

        """
        return int(np.prod(sequence))


def expand_as_right(
    tensor: torch.Tensor | MemmapTensor | TensorDictBase,
    dest: torch.Tensor | MemmapTensor | TensorDictBase,
) -> torch.Tensor | MemmapTensor | TensorDictBase:
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
    tensor: torch.Tensor | MemmapTensor, shape: Sequence[int]
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


NUMPY_TO_TORCH_DTYPE_DICT = {
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
TORCH_TO_NUMPY_DTYPE_DICT = {
    value: key for key, value in NUMPY_TO_TORCH_DTYPE_DICT.items()
}


def is_nested_key(key: NestedKey) -> bool:
    """Returns True if key is a NestedKey."""
    if isinstance(key, str):
        return True
    if key and isinstance(key, (list, tuple)):
        return all(isinstance(subkey, str) for subkey in key)
    return False


def is_seq_of_nested_key(seq: Sequence[NestedKey]) -> bool:
    """Returns True if seq is a Sequence[NestedKey]."""
    if seq and isinstance(seq, Sequence):
        return all(is_nested_key(k) for k in seq)
    elif isinstance(seq, Sequence):
        # we allow empty inputs
        return True
    return False


def index_keyedjaggedtensor(
    kjt: KeyedJaggedTensor, index: slice | range | list | torch.Tensor | np.ndarray
) -> KeyedJaggedTensor:
    """Indexes a KeyedJaggedTensor along the batch dimension.

    Args:
        kjt (KeyedJaggedTensor): a KeyedJaggedTensor to index
        index (torch.Tensor or other indexing type): batch index to use.
            Indexing with an integer will result in an error.

    Examples:
        >>> values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        >>> weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0])
        >>> keys = ["index_0", "index_1", "index_2"]
        >>> offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8, 9, 10, 11])
        >>>
        >>> jag_tensor = KeyedJaggedTensor(
        ...     values=values,
        ...     keys=keys,
        ...     offsets=offsets,
        ...     weights=weights,
        ... )
        >>> ikjt = index_keyedjaggedtensor(jag_tensor, [0, 2])
        >>> print(ikjt["index_0"].to_padded_dense(), j0.to_padded_dense())

    """
    if not _has_torchrec:
        raise ImportError(TORCHREC_ERR)
    if isinstance(index, (int,)):
        raise ValueError(
            "Indexing KeyedJaggedTensor instances with an integer is prohibited, "
            "as this would result in a KeyedJaggedTensor without batch size. "
            "If you want to get a single element from a KeyedJaggedTensor, "
            "call `index_keyedjaggedtensor(kjt, torch.tensor([index]))` instead."
        )
    lengths = kjt.lengths()
    keys = kjt.keys()
    numel = len(lengths) // len(keys)
    offsets = kjt.offsets()

    _offsets1 = offsets[:-1].view(len(keys), numel)[:, index]
    _offsets2 = offsets[1:].view(len(keys), numel)[:, index]
    lengths = lengths.view(len(keys), numel)[:, index].reshape(-1)

    full_index = torch.arange(offsets[-1]).view(1, 1, -1)
    sel = (full_index >= _offsets1.unsqueeze(-1)) & (
        full_index < _offsets2.unsqueeze(-1)
    )
    sel = sel.any(0).any(0)
    full_index = full_index.squeeze()[sel]
    values = kjt._values[full_index]
    weights = kjt._weights[full_index]
    return KeyedJaggedTensor(
        values=values, keys=kjt.keys(), weights=weights, lengths=lengths
    )


def setitem_keyedjaggedtensor(
    orig_tensor: KeyedJaggedTensor,
    index: slice | range | list | torch.Tensor | np.ndarray,
    other: KeyedJaggedTensor,
) -> KeyedJaggedTensor:
    """Equivalent of `tensor[index] = other` for KeyedJaggedTensors indexed along the batch dimension.

    Args:
        orig_tensor (torchrec.KeyedJaggedTensor): KeyedJaggedTensor to be updated.
        index (list or equivalent index): batch index to be written.
        other (torchrec.KeyedJaggedTensor): KeyedJaggedTensor to be written at
            the batch locations.

    Examples:
        >>> values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        >>> weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0])
        >>> keys = ["index_0", "index_1", "index_2"]
        >>> offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8, 9, 10, 11])
        >>> jag_tensor = KeyedJaggedTensor(
        ...    values=values,
        ...    keys=keys,
        ...    offsets=offsets,
        ...    weights=weights,
        ... )
        >>> keys = ["index_0", "index_1", "index_2"]
        >>> lengths2 = torch.IntTensor([2, 4, 6, 4, 2, 1])
        >>> values2 = torch.zeros(
        ...     lengths2.sum(),
        ... )
        >>> weights2 = -torch.ones(
        ...     lengths2.sum(),
        ... )
        >>> sub_jag_tensor = KeyedJaggedTensor(
        ...     values=values2,
        ...     keys=keys,
        ...     lengths=lengths2,
        ...     weights=weights2,
        ... )
        >>> setitem_keyedjaggedtensor(jag_tensor, [0, 2], sub_jag_tensor)
    """
    #     if not _has_torchrec:
    #         raise ImportError(TORCHREC_ERR)

    orig_tensor_lengths = orig_tensor.lengths()
    orig_tensor_keys = orig_tensor.keys()
    orig_tensor_numel = len(orig_tensor_lengths) // len(orig_tensor_keys)
    orig_tensor_offsets = orig_tensor.offsets()

    other_lengths = other.lengths()
    other_keys = other.keys()
    other_numel = len(other_lengths) // len(other_keys)
    # other_offsets = other.offsets()

    if not other_keys == orig_tensor_keys:
        raise KeyError("Mismatch in orig_tensor and other keys.")
    #     if other_numel - len(index) != orig_tensor_numel:
    #         raise RuntimeError("orig_tensor and otherination batch differ.")

    _offsets1 = orig_tensor_offsets[:-1]
    _offsets2 = orig_tensor_offsets[1:]
    _orig_tensor_shape = len(orig_tensor_keys), orig_tensor_numel

    _lengths_out = orig_tensor_lengths.view(_orig_tensor_shape).clone()
    _lengths_out[:, index] = other_lengths.view(len(orig_tensor_keys), other_numel)
    _lengths_out = _lengths_out.view(-1)

    # get the values of orig_tensor that we'll be keeping
    full_index = torch.arange(orig_tensor_offsets[-1]).view(1, 1, -1)
    sel = (full_index >= _offsets1.view(_orig_tensor_shape)[:, index].unsqueeze(-1)) & (
        full_index < _offsets2.view(_orig_tensor_shape)[:, index].unsqueeze(-1)
    )
    sel = (~sel).all(0).all(0)
    index_to_keep = full_index.squeeze()[sel]
    values_to_keep = orig_tensor._values[index_to_keep]
    new_values = other._values
    weights_to_keep = orig_tensor._weights[index_to_keep]
    new_weights = other._weights

    # compute new offsets
    _offsets = torch.cat([_lengths_out[:1] * 0, _lengths_out], 0)
    _offsets = _offsets.cumsum(0)

    # get indices of offsets for new elts
    _offsets1 = _offsets[:-1]
    _offsets2 = _offsets[1:]
    full_index = torch.arange(_offsets[-1]).view(1, 1, -1)
    sel = (full_index >= _offsets1.view(_orig_tensor_shape)[:, index].unsqueeze(-1)) & (
        full_index < _offsets2.view(_orig_tensor_shape)[:, index].unsqueeze(-1)
    )
    sel = sel.any(0).any(0)
    new_index_new_elts = full_index.squeeze()[sel]
    sel = (full_index >= _offsets1.view(_orig_tensor_shape)[:, index].unsqueeze(-1)) & (
        full_index < _offsets2.view(_orig_tensor_shape)[:, index].unsqueeze(-1)
    )
    sel = (~sel).all(0).all(0)
    new_index_to_keep = full_index.squeeze()[sel]

    # create an empty values tensor
    values_numel = values_to_keep.shape[0] + other._values.shape[0]
    tensor = torch.empty(
        [values_numel, *values_to_keep.shape[1:]],
        dtype=values_to_keep.dtype,
        device=values_to_keep.device,
    )
    tensor_weights = torch.empty(
        [values_numel, *values_to_keep.shape[1:]],
        dtype=weights_to_keep.dtype,
        device=weights_to_keep.device,
    )
    tensor[new_index_to_keep] = values_to_keep
    tensor[new_index_new_elts] = new_values
    tensor_weights[new_index_to_keep] = weights_to_keep
    tensor_weights[new_index_new_elts] = new_weights

    kjt = KeyedJaggedTensor(
        values=tensor,
        keys=orig_tensor_keys,
        weights=tensor_weights,
        lengths=_lengths_out,
    )
    for k, item in kjt.__dict__.items():
        orig_tensor.__dict__[k] = item
    return orig_tensor


def _ndimension(tensor: torch.Tensor) -> int:
    if isinstance(tensor, torch.Tensor):
        return tensor.ndimension()
    elif isinstance(tensor, KeyedJaggedTensor):
        return 1
    else:
        return tensor.ndimension()


def _shape(tensor: torch.Tensor) -> torch.Size:
    try:
        return tensor.shape
    except AttributeError as err:
        if type(tensor) is KeyedJaggedTensor:
            return torch.Size([len(tensor.lengths()) // len(tensor.keys())])
        raise err


def _device(tensor: torch.Tensor) -> torch.device:
    if isinstance(tensor, torch.Tensor):
        return tensor.device
    elif isinstance(tensor, KeyedJaggedTensor):
        return tensor.device()
    else:
        return tensor.device


def _is_shared(tensor: torch.Tensor) -> bool:
    if isinstance(tensor, torch.Tensor):
        if torch._C._functorch.is_batchedtensor(tensor):
            return None
        return tensor.is_shared()
    elif isinstance(tensor, KeyedJaggedTensor):
        return False
    else:
        return tensor.is_shared()


def _is_meta(tensor: torch.Tensor) -> bool:
    if isinstance(tensor, torch.Tensor):
        return tensor.is_meta
    elif isinstance(tensor, KeyedJaggedTensor):
        return False
    else:
        return tensor.is_meta


def _dtype(tensor: torch.Tensor) -> torch.dtype:
    if isinstance(tensor, torch.Tensor):
        return tensor.dtype
    elif isinstance(tensor, KeyedJaggedTensor):
        return tensor._values.dtype
    else:
        return tensor.dtype


def _get_item(tensor: torch.Tensor, index: IndexType) -> torch.Tensor:
    if isinstance(tensor, torch.Tensor):
        try:
            return tensor[index]
        except IndexError as err:
            # try to map list index to tensor, and assess type. If bool, we
            # likely have a nested list of booleans which is not supported by pytorch
            if _is_lis_of_list_of_bools(index):
                index = torch.tensor(index, device=tensor.device)
                if index.dtype is torch.bool:
                    warnings.warn(
                        "Indexing a tensor with a nested list of boolean values is "
                        "going to be deprecated as this functionality is not supported "
                        f"by PyTorch. (follows error: {err})",
                        category=DeprecationWarning,
                    )
                return tensor[index]
            raise err
    elif isinstance(tensor, KeyedJaggedTensor):
        return index_keyedjaggedtensor(tensor, index)
    else:
        return tensor[index]


def _set_item(
    tensor: torch.Tensor, index: IndexType, value: torch.Tensor, *, validated
) -> torch.Tensor:
    # the tensor must be validated
    if not validated:
        raise RuntimeError
    if isinstance(tensor, torch.Tensor):
        tensor[index] = value
        return tensor
    elif isinstance(tensor, KeyedJaggedTensor):
        tensor = setitem_keyedjaggedtensor(tensor, index, value)
        return tensor
    else:
        tensor[index] = value
        return tensor


def _requires_grad(tensor: torch.Tensor) -> bool:
    if isinstance(tensor, torch.Tensor):
        return tensor.requires_grad
    elif isinstance(tensor, KeyedJaggedTensor):
        return tensor._values.requires_grad
    else:
        return tensor.requires_grad


class timeit:
    """A dirty but easy to use decorator for profiling code."""

    _REG = {}

    def __init__(self, name) -> None:
        self.name = name

    def __call__(self, fn):
        @wraps(fn)
        def decorated_fn(*args, **kwargs):
            with self:
                out = fn(*args, **kwargs)
                return out

        return decorated_fn

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        t = time.time() - self.t0
        val = self._REG.setdefault(self.name, [0.0, 0.0, 0])

        count = val[2]
        N = count + 1
        val[0] = val[0] * (count / N) + t / N
        val[1] += t
        val[2] = N

    @staticmethod
    def print(prefix=None):
        keys = list(timeit._REG)
        keys.sort()
        for name in keys:
            strings = []
            if prefix:
                strings.append(prefix)
            strings.append(
                f"{name} took {timeit._REG[name][0] * 1000:4.4} msec (total = {timeit._REG[name][1]} sec)"
            )
            print(" -- ".join(strings))

    @staticmethod
    def erase():
        for k in timeit._REG:
            timeit._REG[k] = [0.0, 0.0, 0]


def int_generator(seed):
    """A pseudo-random chaing generator.

    To be used to produce deterministic integer sequences

    Examples:
        >>> for _ in range(2):
        ...     init_int = 10
        ...     for _ in range(10):
        ...        init_int = int_generator(init_int)
        ...        print(init_int, end=", ")
        ...     print("")
        6756, 1717, 4410, 9740, 9611, 9716, 5397, 7745, 4521, 7523,
        6756, 1717, 4410, 9740, 9611, 9716, 5397, 7745, 4521, 7523,
    """
    max_seed_val = 10_000
    rng = np.random.default_rng(seed)
    seed = int.from_bytes(rng.bytes(8), "big")
    return seed % max_seed_val


def _is_lis_of_list_of_bools(index, first_level=True):
    # determines if an index is a list of list of bools.
    # this is aimed at catching a deprecation feature where list of list
    # of bools are valid indices
    if first_level:
        if not isinstance(index, list):
            return False
        if not len(index):
            return False
        if isinstance(index[0], list):
            return _is_lis_of_list_of_bools(index[0], False)
        return False
    # then we know it is a list of lists
    if isinstance(index[0], bool):
        return True
    if isinstance(index[0], list):
        return _is_lis_of_list_of_bools(index[0], False)
    return False


def is_tensorclass(obj: type | Any) -> bool:
    """Returns True if obj is either a tensorclass or an instance of a tensorclass."""
    cls = obj if isinstance(obj, type) else type(obj)
    return _is_tensorclass(cls)


def _is_tensorclass(cls) -> bool:
    return (
        dataclasses.is_dataclass(cls)
        and "to_tensordict" in cls.__dict__
        and "_from_tensordict" in cls.__dict__
    )


class implement_for:
    """A version decorator that checks the version in the environment and implements a function with the fitting one.

    If specified module is missing or there is no fitting implementation, call of the decorated function
    will lead to the explicit error.
    In case of intersected ranges, last fitting implementation is used.

    Args:
        module_name (str or callable): version is checked for the module with this
            name (e.g. "gym"). If a callable is provided, it should return the
            module.
        from_version: version from which implementation is compatible. Can be open (None).
        to_version: version from which implementation is no longer compatible. Can be open (None).

    Examples:
        >>> @implement_for("torch", None, "1.13")
        >>> def fun(self, x):
        ...     # Older torch versions will return x + 1
        ...     return x + 1
        ...
        >>> @implement_for("torch", "0.13", "2.0")
        >>> def fun(self, x):
        ...     # More recent torch versions will return x + 2
        ...     return x + 2
        ...
        >>> @implement_for(lambda: import_module("torch"), "0.", None)
        >>> def fun(self, x):
        ...     # More recent gym versions will return x + 2
        ...     return x + 2
        ...
        >>> @implement_for("gymnasium", "0.27", None)
        >>> def fun(self, x):
        ...     # If gymnasium is to be used instead of gym, x+3 will be returned
        ...     return x + 3
        ...

        This indicates that the function is compatible with gym 0.13+, but doesn't with gym 0.14+.
    """

    # Stores pointers to fitting implementations: dict[func_name] = func_pointer
    _implementations = {}
    _setters = []

    def __init__(
        self,
        module_name: Union[str, Callable],
        from_version: str = None,
        to_version: str = None,
    ):
        self.module_name = module_name
        self.from_version = from_version
        self.to_version = to_version
        implement_for._setters.append(self)

    @staticmethod
    def check_version(version, from_version, to_version):
        return (from_version is None or parse(version) >= parse(from_version)) and (
            to_version is None or parse(version) < parse(to_version)
        )

    @staticmethod
    def get_class_that_defined_method(f):
        """Returns the class of a method, if it is defined, and None otherwise."""
        return f.__globals__.get(f.__qualname__.split(".")[0], None)

    @property
    def func_name(self):
        return self.fn.__name__

    def module_set(self):
        """Sets the function in its module, if it exists already."""
        cls = self.get_class_that_defined_method(self.fn)
        if cls is None:
            # class not yet defined
            return
        if cls.__class__.__name__ == "function":
            cls = inspect.getmodule(self.fn)
        setattr(cls, self.fn.__name__, self.fn)

    @staticmethod
    def import_module(module_name: Union[Callable, str]) -> str:
        """Imports module and returns its version."""
        if not callable(module_name):
            module = import_module(module_name)
        else:
            module = module_name()
        return module.__version__

    def __call__(self, fn):
        self.fn = fn

        # If the module is missing replace the function with the mock.
        func_name = self.func_name
        implementations = implement_for._implementations

        @wraps(fn)
        def unsupported(*args, **kwargs):
            raise ModuleNotFoundError(
                f"Supported version of '{func_name}' has not been found."
            )

        do_set = False
        # Return fitting implementation if it was encountered before.
        if func_name in implementations:
            try:
                # check that backends don't conflict
                version = self.import_module(self.module_name)
                if self.check_version(version, self.from_version, self.to_version):
                    do_set = True
                if not do_set:
                    return implementations[func_name]
            except ModuleNotFoundError:
                # then it's ok, there is no conflict
                return implementations[func_name]
        else:
            try:
                version = self.import_module(self.module_name)
                if self.check_version(version, self.from_version, self.to_version):
                    do_set = True
            except ModuleNotFoundError:
                return unsupported
        if do_set:
            implementations[func_name] = fn
            self.module_set()
            return fn
        return unsupported

    @classmethod
    def reset(cls, setters=None):
        if setters is None:
            setters = copy(cls._setters)
        cls._setters = []
        cls._implementations = {}
        for setter in setters:
            setter(setter.fn)
            cls._setters.append(setter)


def _unfold_sequence(seq):
    for item in seq:
        if isinstance(item, (list, tuple)):
            yield tuple(_unfold_sequence(item))
        else:
            if isinstance(item, (str, int, slice)) or item is Ellipsis:
                yield item
            else:
                yield id(item)


def _make_cache_key(args, kwargs):
    """Creats a key for the cache such that memory footprint is minimized."""
    return (
        tuple(_unfold_sequence(args)),
        tuple(_unfold_sequence(sorted(kwargs.items()))),
    )


def cache(fun):
    """A cache for TensorDictBase subclasses.

    This decorator will cache the values returned by a method as long as the
    input arguments match.
    Leaves (tensors and such) are not cached.
    The cache is stored within the tensordict such that it can be erased at any
    point in time.

    Examples:
        >>> import timeit
        >>> from tensordict import TensorDict
        >>> class SomeOtherTd(TensorDict):
        ...     @cache
        ...     def all_keys(self):
        ...         return set(self.keys(include_nested=True))
        >>> td = SomeOtherTd({("a", "b", "c", "d", "e", "f", "g"): 1.0}, [])
        >>> td.lock_()
        >>> print(timeit.timeit("set(td.keys(True))", globals={'td': td}))
        11.057
        >>> print(timeit.timeit("set(td.all_keys())", globals={'td': td}))
        0.88
    """
    from tensordict.memmap import MemmapTensor

    @wraps(fun)
    def newfun(_self: "TensorDictBase", *args, **kwargs):
        if not _self.is_locked:
            return fun(_self, *args, **kwargs)
        cache = _self._cache
        if cache is None:
            cache = _self._cache = defaultdict(dict)
        cache = cache[fun.__name__]
        key = _make_cache_key(args, kwargs)
        if key not in cache:
            out = fun(_self, *args, **kwargs)
            if not isinstance(out, (Tensor, MemmapTensor, KeyedJaggedTensor)):
                # we don't cache tensors to avoid filling the mem and / or
                # stacking them from their origin
                cache[key] = out
        else:
            out = cache[key]
        return out

    return newfun


def erase_cache(fun):
    """A decorator to erase the cache at each call."""

    @wraps(fun)
    def new_fun(self, *args, **kwargs):
        self._erase_cache()
        return fun(self, *args, **kwargs)

    return new_fun


_NON_STR_KEY_TUPLE_ERR = "Nested membership checks with tuples of strings is only supported when setting `include_nested=True`."
_NON_STR_KEY_ERR = "TensorDict keys are always strings. Membership checks are only supported for strings or non-empty tuples of strings (for nested TensorDicts)"
_GENERIC_NESTED_ERR = "Only NestedKeys are supported."


class _StringKeys(KeysView):
    """A key view where contains is restricted to strings."""

    def __contains__(self, item):
        if not isinstance(item, str):
            try:
                unravel_item = _unravel_key_to_tuple(item)
                if not unravel_item:  # catch errors during unravel
                    raise TypeError
            except Exception:
                raise TypeError(_NON_STR_KEY_ERR)
            if len(unravel_item) > 1:
                raise TypeError(_NON_STR_KEY_TUPLE_ERR)
            else:
                item = unravel_item[0]
        return super().__contains__(item)


class _StringOnlyDict(dict):
    """A dict class where contains is restricted to strings."""

    # kept here for debugging
    # def __setitem__(self, key, value):
    #     if not isinstance(key, str):
    #         raise RuntimeError
    #     return super().__setitem__(key, value)

    def __contains__(self, item):
        if not isinstance(item, str):
            try:
                unravel_item = _unravel_key_to_tuple(item)
                if not unravel_item:  # catch errors during unravel
                    raise TypeError
            except Exception:
                raise TypeError(_NON_STR_KEY_ERR)
            if len(unravel_item) > 1:
                raise TypeError(_NON_STR_KEY_TUPLE_ERR)
            else:
                item = unravel_item[0]
        return super().__contains__(item)

    def keys(self):
        return _StringKeys(self)


def lock_blocked(func):
    """Checks that the tensordict is unlocked before executing a function."""

    @wraps(func)
    def new_func(self, *args, **kwargs):
        if self.is_locked:
            raise RuntimeError(self.LOCK_ERROR)
        return func(self, *args, **kwargs)

    return new_func


class as_decorator:
    """Converts a method to a decorator.

    Examples:
        >>> from tensordict import TensorDict
        >>> data = TensorDict({}, [])
        >>> with data.lock_(): # lock_ is decorated
        ...     assert data.is_locked
        >>> assert not data.is_locked
    """

    def __init__(self, attr):
        self.attr = attr

    def __call__(self, func):
        @wraps(func)
        def new_func(_self, *args, **kwargs):
            _attr_pre = getattr(_self, self.attr)
            out = func(_self, *args, **kwargs)
            _attr_post = getattr(_self, self.attr)
            if _attr_post is not _attr_pre:
                _self._last_op = (new_func.__name__, (args, kwargs))
            else:
                _self._last_op = None
            return out

        return new_func

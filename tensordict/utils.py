# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import collections
import concurrent.futures
import dataclasses
import inspect
import logging

import math
import os

import sys
import time

import warnings
from collections import defaultdict, OrderedDict
from collections.abc import KeysView
from copy import copy
from distutils.util import strtobool
from functools import wraps
from importlib import import_module
from numbers import Number
from textwrap import indent
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import numpy as np
import torch
from functorch import dim as ftdim
from packaging.version import parse
from tensordict._contextlib import _DecoratorContextManager
from tensordict._tensordict import (  # noqa: F401
    _unravel_key_to_tuple,
    unravel_key,
    unravel_key_list,
    unravel_keys,
)

from torch import Tensor
from torch._C import _disabled_torch_function_impl
from torch.nn.parameter import (
    _ParameterMeta,
    UninitializedBuffer,
    UninitializedParameter,
    UninitializedTensorMixin,
)
from torch.utils.data._utils.worker import _generate_state

if TYPE_CHECKING:
    from tensordict.memmap_deprec import MemmapTensor as _MemmapTensor
    from tensordict.tensordict import TensorDictBase

try:
    try:
        from functorch._C import get_unwrapped, is_batchedtensor
    except ImportError:
        from torch._C._functorch import get_unwrapped, is_batchedtensor
except ImportError:
    pass

TORCHREC_ERR = None
try:
    from torchrec import KeyedJaggedTensor

    _has_torchrec = True
except ImportError as err:
    _has_torchrec = False

    class KeyedJaggedTensor:  # noqa: D103, D101
        pass

    TORCHREC_ERR = err

T = TypeVar("T", bound="TensorDictBase")

_STRDTYPE2DTYPE = {
    str(dtype): dtype
    for dtype in (
        torch.float32,
        torch.float64,
        torch.float16,
        torch.bfloat16,
        torch.complex32,
        torch.complex64,
        torch.complex128,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.quint8,
        torch.qint8,
        torch.qint32,
        torch.quint4x2,
    )
}

IndexType = Union[None, int, slice, str, Tensor, List[Any], Tuple[Any, ...]]
DeviceType = Union[torch.device, str, int]
NestedKey = Union[str, Tuple[str, ...]]

_KEY_ERROR = 'key "{}" not found in {} with ' "keys {}"
_LOCK_ERROR = (
    "Cannot modify locked TensorDict. For in-place modification, consider "
    "using the `set_()` method and make sure the key is present."
)


def _sub_index(tensor: Tensor, idx: IndexType) -> Tensor:
    """Allows indexing of tensors with nested tuples.

     >>> sub_tensor1 = tensor[tuple1][tuple2]
     >>> sub_tensor2 = _sub_index(tensor, (tuple1, tuple2))
     >>> assert torch.allclose(sub_tensor1, sub_tensor2)

    Args:
        tensor (Tensor): tensor to be indexed.
        idx (tuple of indices): indices sequence to be used.

    """
    if isinstance(idx, tuple) and len(idx) and isinstance(idx[0], tuple):
        idx0 = idx[0]
        idx1 = idx[1:]
        return _sub_index(_sub_index(tensor, idx0), idx1)
    return tensor[idx]


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


def _unwrap_value(value: Tensor) -> Tensor:
    # batch_dims = value.ndimension()
    if not isinstance(value, Tensor):
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
    tensor: torch.Tensor | _MemmapTensor | TensorDictBase,
    dest: torch.Tensor | _MemmapTensor | TensorDictBase,
) -> torch.Tensor | _MemmapTensor | TensorDictBase:
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
    if any(
        tensor.shape[i] != dest.shape[i] and tensor.shape[i] != 1
        for i in range(tensor.ndimension())
    ):
        raise RuntimeError(
            f"tensor shape is incompatible with dest shape, "
            f"got: tensor.shape={tensor.shape}, dest={dest.shape}"
        )
    for _ in range(dest.ndimension() - tensor.ndimension()):
        tensor = tensor.unsqueeze(-1)
    return tensor.expand(dest.shape)


def expand_right(tensor: Tensor, shape: Sequence[int]) -> Tensor:
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
    tensor_expand = tensor_expand.expand(shape)
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
        raise TORCHREC_ERR
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


def _ndimension(tensor: Tensor) -> int:
    if isinstance(tensor, Tensor):
        return tensor.ndimension()
    elif isinstance(tensor, KeyedJaggedTensor):
        return 1
    else:
        return tensor.ndimension()


def _shape(tensor: Tensor) -> torch.Size:
    if isinstance(tensor, UninitializedTensorMixin):
        return torch.Size([*getattr(tensor, "batch_size", ()), -1])
    elif not isinstance(tensor, Tensor):
        if type(tensor) is KeyedJaggedTensor:
            return torch.Size([len(tensor.lengths()) // len(tensor.keys())])
        return tensor.shape
    if tensor.is_nested:
        shape = []
        for i in range(tensor.ndim):
            try:
                shape.append(tensor.size(i))
            except RuntimeError:
                shape.append(-1)
        return torch.Size(shape)
    return tensor.shape


def _device(tensor: Tensor) -> torch.device:
    if isinstance(tensor, Tensor):
        return tensor.device
    elif isinstance(tensor, KeyedJaggedTensor):
        return tensor.device()
    else:
        return tensor.device


def _is_shared(tensor: Tensor) -> bool:
    if isinstance(tensor, Tensor):
        if torch._C._functorch.is_batchedtensor(tensor):
            return None
        return tensor.is_shared()
    if isinstance(tensor, ftdim.Tensor):
        return None
    elif isinstance(tensor, KeyedJaggedTensor):
        return False
    else:
        return tensor.is_shared()


def _is_meta(tensor: Tensor) -> bool:
    if isinstance(tensor, Tensor):
        return tensor.is_meta
    elif isinstance(tensor, KeyedJaggedTensor):
        return False
    else:
        return tensor.is_meta


def _dtype(tensor: Tensor) -> torch.dtype:
    if isinstance(tensor, Tensor):
        return tensor.dtype
    elif isinstance(tensor, KeyedJaggedTensor):
        return tensor._values.dtype
    else:
        return tensor.dtype


def _get_item(tensor: Tensor, index: IndexType) -> Tensor:
    if isinstance(tensor, Tensor):
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


def _set_item(tensor: Tensor, index: IndexType, value: Tensor, *, validated) -> Tensor:
    # the tensor must be validated
    if not validated:
        raise RuntimeError
    if isinstance(tensor, Tensor):
        tensor[index] = value
        return tensor
    elif isinstance(tensor, KeyedJaggedTensor):
        tensor = setitem_keyedjaggedtensor(tensor, index, value)
        return tensor
    from tensordict.tensorclass import NonTensorData, NonTensorStack

    if is_non_tensor(tensor):
        if (
            isinstance(value, NonTensorData)
            and isinstance(tensor, NonTensorData)
            and tensor.data == value.data
        ):
            return tensor
        elif isinstance(tensor, NonTensorData):
            tensor = NonTensorStack.from_nontensordata(tensor)
        if tensor.stack_dim != 0:
            tensor = NonTensorStack(*tensor.unbind(0), stack_dim=0)
        tensor[index] = value
        return tensor
    else:
        tensor[index] = value
        return tensor


def _requires_grad(tensor: Tensor) -> bool:
    if isinstance(tensor, Tensor):
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
    def print(prefix=None):  # noqa: T202
        keys = list(timeit._REG)
        keys.sort()
        for name in keys:
            strings = []
            if prefix:
                strings.append(prefix)
            strings.append(
                f"{name} took {timeit._REG[name][0] * 1000:4.4} msec (total = {timeit._REG[name][1]} sec)"
            )
            logging.info(" -- ".join(strings))

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
    _cache_modules = {}

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
    def check_version(version: str, from_version: str | None, to_version: str | None):
        version = parse(".".join([str(v) for v in parse(version).release]))
        return (from_version is None or version >= parse(from_version)) and (
            to_version is None or version < parse(to_version)
        )

    @staticmethod
    def get_class_that_defined_method(f):
        """Returns the class of a method, if it is defined, and None otherwise."""
        return f.__globals__.get(f.__qualname__.split(".")[0], None)

    @classmethod
    def get_func_name(cls, fn):
        # produces a name like torchrl.module.Class.method or torchrl.module.function
        first = str(fn).split(".")[0][len("<function ") :]
        last = str(fn).split(".")[1:]
        if last:
            first = [first]
            last[-1] = last[-1].split(" ")[0]
        else:
            last = [first.split(" ")[0]]
            first = []
        return ".".join([fn.__module__] + first + last)

    def _get_cls(self, fn):
        cls = self.get_class_that_defined_method(fn)
        if cls is None:
            # class not yet defined
            return
        if cls.__class__.__name__ == "function":
            cls = inspect.getmodule(fn)
        return cls

    def module_set(self):
        """Sets the function in its module, if it exists already."""
        prev_setter = type(self)._implementations.get(self.get_func_name(self.fn), None)
        if prev_setter is not None:
            prev_setter.do_set = False
        type(self)._implementations[self.get_func_name(self.fn)] = self
        cls = self.get_class_that_defined_method(self.fn)
        if cls is not None:
            if cls.__class__.__name__ == "function":
                cls = inspect.getmodule(self.fn)
        else:
            # class not yet defined
            return
        setattr(cls, self.fn.__name__, self.fn)

    @classmethod
    def import_module(cls, module_name: Union[Callable, str]) -> str:
        """Imports module and returns its version."""
        if not callable(module_name):
            module = cls._cache_modules.get(module_name, None)
            if module is None:
                if module_name in sys.modules:
                    sys.modules[module_name] = module = import_module(module_name)
                else:
                    cls._cache_modules[module_name] = module = import_module(
                        module_name
                    )
        else:
            module = module_name()
        return module.__version__

    _lazy_impl = collections.defaultdict(list)

    def _delazify(self, func_name):
        for local_call in implement_for._lazy_impl[func_name]:
            out = local_call()
        return out

    def __call__(self, fn):
        # function names are unique
        self.func_name = self.get_func_name(fn)
        self.fn = fn
        implement_for._lazy_impl[self.func_name].append(self._call)

        @wraps(fn)
        def _lazy_call_fn(*args, **kwargs):
            # first time we call the function, we also do the replacement.
            # This will cause the imports to occur only during the first call to fn
            return self._delazify(self.func_name)(*args, **kwargs)

        return _lazy_call_fn

    def _call(self):

        # If the module is missing replace the function with the mock.
        fn = self.fn
        func_name = self.func_name
        implementations = implement_for._implementations

        @wraps(fn)
        def unsupported(*args, **kwargs):
            raise ModuleNotFoundError(
                f"Supported version of '{func_name}' has not been found."
            )

        self.do_set = False
        # Return fitting implementation if it was encountered before.
        if func_name in implementations:
            try:
                # check that backends don't conflict
                version = self.import_module(self.module_name)
                if self.check_version(version, self.from_version, self.to_version):
                    self.do_set = True
                if not self.do_set:
                    return implementations[func_name].fn
            except ModuleNotFoundError:
                # then it's ok, there is no conflict
                return implementations[func_name].fn
        else:
            try:
                version = self.import_module(self.module_name)
                if self.check_version(version, self.from_version, self.to_version):
                    self.do_set = True
            except ModuleNotFoundError:
                return unsupported
        if self.do_set:
            self.module_set()
            return fn
        return unsupported

    @classmethod
    def reset(cls, setters_dict: Dict[str, implement_for] = None):
        """Resets the setters in setter_dict.

        ``setter_dict`` is a copy of implementations. We just need to iterate through its
        values and call :meth:`~.module_set` for each.

        """
        if setters_dict is None:
            setters_dict = copy(cls._implementations)
        for setter in setters_dict.values():
            setter.module_set()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"module_name={self.module_name}({self.from_version, self.to_version}), "
            f"fn_name={self.fn.__name__}, cls={self._get_cls(self.fn)}, is_set={self.do_set})"
        )


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
    from tensordict.memmap_deprec import MemmapTensor as _MemmapTensor

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
            if not isinstance(out, (Tensor, _MemmapTensor, KeyedJaggedTensor)):
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
_GENERIC_NESTED_ERR = "Only NestedKeys are supported. Got key {}."


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
            raise RuntimeError(_LOCK_ERROR)
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

    def __init__(self, attr=None):
        self.attr = attr

    def __call__(self, func):
        if self.attr is not None:

            @wraps(func)
            def new_func(_self, *args, **kwargs):
                _attr_pre = getattr(_self, self.attr)
                out = func(_self, *args, **kwargs)
                _attr_post = getattr(_self, self.attr)
                if out is not None:
                    if _attr_post is not _attr_pre:
                        out._last_op = (new_func.__name__, (args, kwargs, _self))
                    else:
                        out._last_op = None
                return out

        else:

            @wraps(func)
            def new_func(_self, *args, **kwargs):
                out = func(_self, *args, **kwargs)
                if out is not None:
                    out._last_op = (new_func.__name__, (args, kwargs, _self))
                return out

        return new_func


def _split_tensordict(
    td,
    chunksize,
    num_chunks,
    num_workers,
    dim,
    use_generator=False,
    to_tensordict=False,
):
    if chunksize is None and num_chunks is None:
        num_chunks = num_workers
    if chunksize is not None and num_chunks is not None:
        raise ValueError(
            "Either chunksize or num_chunks must be provided, but not both."
        )
    if num_chunks is not None:
        num_chunks = min(td.shape[dim], num_chunks)
        if use_generator:

            def _chunk_generator():
                chunksize = -(td.shape[dim] // -num_chunks)
                idx_start = 0
                base = (slice(None),) * dim
                for _ in range(num_chunks):
                    idx_end = idx_start + chunksize
                    out = td[base + (slice(idx_start, idx_end),)]
                    if to_tensordict:
                        out = out.to_tensordict()
                    yield out
                    idx_start = idx_end

            return _chunk_generator()
        return td.chunk(num_chunks, dim=dim)
    else:
        if chunksize == 0:
            if use_generator:

                def _unbind_generator():
                    base = (slice(None),) * dim
                    for i in range(td.shape[dim]):
                        out = td[base + (i,)]
                        if to_tensordict:
                            out = out.to_tensordict()
                        yield out

                return _unbind_generator()
            return td.unbind(dim=dim)
        if use_generator:

            def _split_generator():
                idx_start = 0
                base = (slice(None),) * dim
                for _ in range(num_chunks):
                    idx_end = idx_start + chunksize
                    out = td[base + (slice(idx_start, idx_end),)]
                    if to_tensordict:
                        out = out.to_tensordict()
                    yield out
                    idx_start = idx_end

            return _split_generator()
        chunksize = min(td.shape[dim], chunksize)
        return td.split(chunksize, dim=dim)


def _parse_to(*args, **kwargs):
    batch_size = kwargs.pop("batch_size", None)
    other = kwargs.pop("other", None)
    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
        *args, **kwargs
    )
    if other is not None:
        if device is not None and device != other.device:
            raise ValueError("other and device cannot be both passed")
        device = other.device
        dtypes = {val.dtype for val in other.values(True, True)}
        if len(dtypes) > 1 or len(dtypes) == 0:
            dtype = None
        elif len(dtypes) == 1:
            dtype = list(dtypes)[0]
    return device, dtype, non_blocking, convert_to_format, batch_size


class _ErrorInteceptor:
    """Context manager for catching errors and modifying message.

    Intended for use with stacking / concatenation operations applied to TensorDicts.

    """

    DEFAULT_EXC_MSG = "Expected all tensors to be on the same device"

    def __init__(
        self,
        key: NestedKey,
        prefix: str,
        exc_msg: str | None = None,
        exc_type: type[Exception] | None = None,
    ) -> None:
        self.exc_type = exc_type if exc_type is not None else RuntimeError
        self.exc_msg = exc_msg if exc_msg is not None else self.DEFAULT_EXC_MSG
        self.prefix = prefix
        self.key = key

    def _add_key_to_error_msg(self, msg: str) -> str:
        if msg.startswith(self.prefix):
            return f'{self.prefix} "{self.key}" /{msg[len(self.prefix):]}'
        return f'{self.prefix} "{self.key}". {msg}'

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, _):
        if exc_type is self.exc_type and (
            self.exc_msg is None or self.exc_msg in str(exc_value)
        ):
            exc_value.args = (self._add_key_to_error_msg(str(exc_value)),)


def _nested_keys_to_dict(keys: Iterator[NestedKey]) -> dict[str, Any]:
    nested_keys = {}
    for key in keys:
        if isinstance(key, str):
            nested_keys.setdefault(key, {})
        else:
            d = nested_keys
            for subkey in key:
                d = d.setdefault(subkey, {})
    return nested_keys


def _dict_to_nested_keys(
    nested_keys: dict[NestedKey, NestedKey], prefix: tuple[str, ...] = ()
) -> tuple[str, ...]:
    for key, subkeys in nested_keys.items():
        if subkeys:
            yield from _dict_to_nested_keys(subkeys, prefix=(*prefix, key))
        elif prefix:
            yield (*prefix, key)
        else:
            yield key


def _default_hook(td: T, key: tuple[str, ...]) -> None:
    """Used to populate a tensordict.

    For example, ``td.set(("a", "b"))`` may require to create ``"a"``.

    """
    out = td.get(key[0], None)
    if out is None:
        td._create_nested_str(key[0])
        out = td._get_str(key[0], None)
    return out


def _get_leaf_tensordict(
    tensordict: T, key: tuple[str, ...], hook: Callable = None
) -> tuple[TensorDictBase, str]:
    # utility function for traversing nested tensordicts
    # hook should return the default value for tensordit.get(key)
    while len(key) > 1:
        if hook is not None:
            tensordict = hook(tensordict, key)
        else:
            tensordict = tensordict.get(key[0])
        key = key[1:]
    return tensordict, key[0]


def assert_allclose_td(
    actual: T,
    expected: T,
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = True,
    msg: str = "",
) -> bool:
    """Compares two tensordicts and raise an exception if their content does not match exactly."""
    from tensordict.base import _is_tensor_collection

    if not _is_tensor_collection(actual.__class__) or not _is_tensor_collection(
        expected.__class__
    ):
        raise TypeError("assert_allclose inputs must be of TensorDict type")

    from tensordict._lazy import LazyStackedTensorDict

    if isinstance(actual, LazyStackedTensorDict) and isinstance(
        expected, LazyStackedTensorDict
    ):
        for sub_actual, sub_expected in zip(actual.tensordicts, expected.tensordicts):
            assert_allclose_td(sub_actual, sub_expected, rtol=rtol, atol=atol)
        return True

    set1 = set(actual.keys())
    set2 = set(expected.keys())
    if not (len(set1.difference(set2)) == 0 and len(set2) == len(set1)):
        raise KeyError(
            "actual and expected tensordict keys mismatch, "
            f"keys {(set1 - set2).union(set2 - set1)} appear in one but not "
            f"the other."
        )
    keys = sorted(actual.keys(), key=str)
    for key in keys:
        input1 = actual.get(key)
        input2 = expected.get(key)
        if _is_tensor_collection(input1.__class__):
            assert_allclose_td(input1, input2, rtol=rtol, atol=atol)
            continue

        mse = (input1.to(torch.float) - input2.to(torch.float)).pow(2).sum()
        mse = mse.div(input1.numel()).sqrt().item()

        default_msg = f"key {key} does not match, got mse = {mse:4.4f}"
        msg = "\t".join([default_msg, msg]) if len(msg) else default_msg
        torch.testing.assert_close(
            input1, input2, rtol=rtol, atol=atol, equal_nan=equal_nan, msg=msg
        )
    return True


def _get_repr(tensor: Tensor) -> str:
    s = ", ".join(
        [
            f"shape={_shape(tensor)}",
            f"device={_device(tensor)}",
            f"dtype={_dtype(tensor)}",
            f"is_shared={_is_shared(tensor)}",
        ]
    )
    return f"{tensor.__class__.__name__}({s})"


def _get_repr_custom(cls, shape, device, dtype, is_shared) -> str:
    s = ", ".join(
        [
            f"shape={shape}",
            f"device={device}",
            f"dtype={dtype}",
            f"is_shared={is_shared}",
        ]
    )
    return f"{cls.__name__}({s})"


def _make_repr(key: NestedKey, item, tensordict: T) -> str:
    from tensordict.base import _is_tensor_collection

    if _is_tensor_collection(type(item)):
        return f"{key}: {repr(tensordict.get(key))}"
    return f"{key}: {_get_repr(item)}"


def _td_fields(td: T, keys=None) -> str:
    strs = []
    if keys is None:
        keys = td.keys()
    for key in keys:
        shape = td.get_item_shape(key)
        if -1 not in shape:
            item = td.get(key)
            strs.append(_make_repr(key, item, td))
        else:
            # we know td is lazy stacked and the key is a leaf
            # so we can get the shape and escape the error
            temp_td = td
            from tensordict import LazyStackedTensorDict, TensorDictBase

            while isinstance(
                temp_td, LazyStackedTensorDict
            ):  # we need to grab the het tensor from the inner nesting level
                temp_td = temp_td.tensordicts[0]
            tensor = temp_td.get(key)

            if isinstance(tensor, TensorDictBase):
                substr = _td_fields(tensor)
            else:
                is_shared = (
                    tensor.is_shared()
                    if not isinstance(tensor, UninitializedTensorMixin)
                    else None
                )
                substr = _get_repr_custom(
                    tensor.__class__,
                    shape=shape,
                    device=tensor.device,
                    dtype=tensor.dtype,
                    is_shared=is_shared,
                )
            strs.append(f"{key}: {substr}")

    return indent(
        "\n" + ",\n".join(sorted(strs)),
        4 * " ",
    )


def _check_keys(
    list_of_tensordicts: Sequence[TensorDictBase],
    strict: bool = False,
    include_nested: bool = False,
    leaves_only: bool = False,
) -> set[str]:
    if not len(list_of_tensordicts):
        return set()
    keys: set[str] = set(
        list_of_tensordicts[0].keys(
            include_nested=include_nested, leaves_only=leaves_only
        )
    )
    for td in list_of_tensordicts[1:]:
        k = td.keys(include_nested=include_nested, leaves_only=leaves_only)
        if not strict:
            keys = keys.intersection(k)
        else:
            if set(k) != keys:
                raise KeyError(
                    f"got keys {keys} and {set(td.keys())} which are incompatible"
                )
    return keys


def _expand_to_match_shape(
    parent_batch_size: torch.Size,
    tensor: Tensor,
    self_batch_dims: int,
    self_device: DeviceType,
) -> Tensor | TensorDictBase:
    if hasattr(tensor, "dtype"):
        return torch.zeros(
            (
                *parent_batch_size,
                *_shape(tensor)[self_batch_dims:],
            ),
            dtype=tensor.dtype,
            device=self_device,
        )
    else:
        # tensordict
        out = tensor.empty()
        out.batch_size = torch.Size(
            [*parent_batch_size, *_shape(tensor)[self_batch_dims:]]
        )
        return out


def _set_max_batch_size(source: T, batch_dims=None):
    """Updates a tensordict with its maximium batch size."""
    tensor_data = [val for val in source.values() if not is_non_tensor(val)]

    for val in tensor_data:
        from tensordict.base import _is_tensor_collection

        if _is_tensor_collection(val.__class__):
            _set_max_batch_size(val, batch_dims=batch_dims)

    batch_size = []
    if not tensor_data:  # when source is empty
        if batch_dims:
            source.batch_size = source.batch_size[:batch_dims]
            return source
        else:
            return source

    curr_dim = 0
    while True:
        if tensor_data[0].dim() > curr_dim:
            curr_dim_size = tensor_data[0].size(curr_dim)
        else:
            source.batch_size = batch_size
            return
        for leaf in tensor_data[1:]:
            # if we have a nested empty tensordict we can modify its batch size at will
            if _is_tensor_collection(type(leaf)) and leaf.is_empty():
                continue
            if (leaf.dim() <= curr_dim) or (leaf.size(curr_dim) != curr_dim_size):
                source.batch_size = batch_size
                return
        if batch_dims is None or len(batch_size) < batch_dims:
            batch_size.append(curr_dim_size)
        curr_dim += 1


def _clone_value(value, recurse: bool):
    from tensordict.base import _is_tensor_collection

    if recurse:
        # this is not a problem for locked tds as we will not lock it
        return value.clone()
    elif _is_tensor_collection(value.__class__):
        return value._clone(recurse=False)
    else:
        return value


def _is_number(item):
    if isinstance(item, (Number, ftdim.Dim)):
        return True
    if isinstance(item, Tensor) and item.ndim == 0:
        return True
    if isinstance(item, np.ndarray) and item.ndim == 0:
        return True
    return False


def _expand_index(index, batch_size):
    len_index = sum(True for idx in index if idx is not None)
    if len_index > len(batch_size):
        raise ValueError
    if len_index < len(batch_size):
        index = index + (slice(None),) * (len(batch_size) - len_index)
    return index


def _renamed_inplace_method(fn):
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{fn.__name__.rstrip('_')} has been deprecated, use {fn.__name__} instead"
        )
        return fn(*args, **kwargs)

    return wrapper


def _broadcast_tensors(index):
    # tensors and range need to be broadcast
    tensors = {
        i: torch.as_tensor(tensor)
        for i, tensor in enumerate(index)
        if isinstance(tensor, (range, list, np.ndarray, Tensor))
    }
    if tensors:
        shape = torch.broadcast_shapes(*[tensor.shape for tensor in tensors.values()])
        tensors = {i: tensor.expand(shape) for i, tensor in tensors.items()}
        index = tuple(
            idx if i not in tensors else tensors[i] for i, idx in enumerate(index)
        )
    return index


def _reduce_index(index):
    if all(
        idx is Ellipsis or (isinstance(idx, slice) and idx == slice(None))
        for idx in index
    ):
        index = ()
    return index


def _get_shape_from_args(*args, kwarg_name="size", **kwargs):
    if not args and not kwargs:
        return ()
    if args:
        if len(args) > 1 or isinstance(args[0], Number):
            size = args
        else:
            size = args[0]
        if len(kwargs):
            raise TypeError(
                f"Either the kwarg `{kwarg_name}`, a single shape argument or a sequence of integers can be passed. Got args={args} and kwargs={kwargs}."
            )
    else:
        size = kwargs.pop(kwarg_name, None)
        if size is None:
            raise TypeError(
                f"Either the kwarg `{kwarg_name}`, a single shape argument or a sequence of integers can be passed. Got args={args} and kwargs={kwargs}."
            )
    return size


class Buffer(Tensor, metaclass=_ParameterMeta):
    r"""A kind of Tensor that is to be considered a module buffer.

    Args:
        data (Tensor): buffer tensor.
        requires_grad (bool, optional): if the buffer requires gradient. See
            :ref:`locally-disable-grad-doc` for more details. Default: `False`
    """

    def __new__(cls, data=None, requires_grad=False):
        if data is None:
            data = torch.empty(0)
        if type(data) is Tensor or type(data) is Buffer:
            # For ease of BC maintenance, keep this path for standard Tensor.
            # Eventually (tm), we should change the behavior for standard Tensor to match.
            return Tensor._make_subclass(cls, data, requires_grad)

        # Path for custom tensors: set a flag on the instance to indicate parameter-ness.
        t = data.detach().requires_grad_(requires_grad)
        t._is_buffer = True
        return t

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(
                self.data.clone(memory_format=torch.preserve_format), self.requires_grad
            )
            memo[id(self)] = result
            return result

    def __repr__(self):
        return "Buffer containing:\n" + super(Buffer, self).__repr__()

    def __reduce_ex__(self, proto):
        # See Note [Don't serialize hooks]
        return (
            torch._utils._rebuild_parameter,
            (self.data, self.requires_grad, OrderedDict()),
        )

    __torch_function__ = _disabled_torch_function_impl


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
        elif isinstance(idx, (int, ftdim.Dim)):
            # could be spared for efficiency
            continue
        elif isinstance(idx, slice):
            batch = batch_size[count]
            out.append(len(range(*idx.indices(batch))))
    count += 1
    if batch_size[count:]:
        out.extend(batch_size[count:])
    return torch.Size(out)


# Lazy classes control (legacy feature)
_DEFAULT_LAZY_OP = False
_LAZY_OP = os.environ.get("LAZY_LEGACY_OP", None)


class set_lazy_legacy(_DecoratorContextManager):
    """Sets the behaviour of some methods to a lazy transform.

    These methods include :meth:`~tensordict.TensorDict.view`, :meth:`~tensordict.TensorDict.permute`,
    :meth:`~tensordict.TensorDict.transpose`, :meth:`~tensordict.TensorDict.squeeze`
    and :meth:`~tensordict.TensorDict.unsqueeze`.

    This property is dynamic, ie. it can be changed during the code execution, but
    it won't propagate to sub-processes unless it has been called before the process
    has been created.

    """

    def __init__(self, mode: bool) -> None:
        super().__init__()
        self.mode = mode

    def clone(self) -> set_lazy_legacy:
        # override this method if your children class takes __init__ parameters
        return self.__class__(self.mode)

    def __enter__(self) -> None:
        self.set()

    def set(self) -> None:
        global _LAZY_OP
        self._old_mode = _LAZY_OP
        _LAZY_OP = bool(self.mode)
        # we do this such that sub-processes see the same lazy op than the main one
        os.environ["LAZY_LEGACY_OP"] = str(_LAZY_OP)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _LAZY_OP
        _LAZY_OP = bool(self._old_mode)
        os.environ["LAZY_LEGACY_OP"] = str(_LAZY_OP)


def lazy_legacy(allow_none=False):
    """Returns `True` if lazy representations will be used for selected methods."""
    global _LAZY_OP
    if _LAZY_OP is None and allow_none:
        return None
    elif _LAZY_OP is None:
        return _DEFAULT_LAZY_OP
    return strtobool(_LAZY_OP) if isinstance(_LAZY_OP, str) else _LAZY_OP


def _legacy_lazy(func):
    if not func.__name__.startswith("_legacy_"):
        raise NameError(
            f"The function name {func.__name__} must start with _legacy_ if it's decorated with _legacy_lazy."
        )
    func.LEGACY = True
    return func


# Process initializer for map
def _proc_init(base_seed, queue, num_threads):
    worker_id = queue.get(timeout=120)
    seed = base_seed + worker_id
    torch.manual_seed(seed)
    np_seed = _generate_state(base_seed, worker_id)
    np.random.seed(np_seed)
    torch.set_num_threads(num_threads)


def _prune_selected_keys(keys_to_update, prefix):
    if keys_to_update is None:
        return None
    return tuple(
        key[1:] for key in keys_to_update if isinstance(key, tuple) and key[0] == prefix
    )


class TensorDictFuture:
    """A custom future class for TensorDict multithreaded operations.

    Args:
        futures (list of futures): a list of concurrent.futures.Future objects to wait for.
        resulting_td (TensorDictBase): instance that will result from the futures
            completing.

    """

    def __init__(self, futures, resulting_td):
        self.futures = futures
        self.resulting_td = resulting_td

    def result(self):
        """Wait and returns the resulting tensordict."""
        concurrent.futures.wait(self.futures)
        return self.resulting_td


def _is_json_serializable(item):
    if isinstance(item, dict):
        for key, val in item.items():
            # Per se, int, float and bool are serializable but not recoverable
            # as such
            if not isinstance(key, (str,)) or not _is_json_serializable(val):
                return False
        else:
            return True
    if isinstance(item, (list, tuple, set)):
        for val in item:
            if not _is_json_serializable(val):
                return False
        else:
            return True
    return isinstance(item, (str, int, float, bool)) or item is None


def print_directory_tree(path, indent="", display_metadata=True):
    """Prints the directory tree starting from the specified path.

    Args:
        path (str): The path of the directory to print.
        indent (str): The current indentation level for formatting.
        display_metadata (bool): if ``True``, metadata of the dir will be
            displayed too.

    """
    if display_metadata:

        def get_directory_size(path="."):
            total_size = 0

            for dirpath, _, filenames in os.walk(path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(file_path)

            return total_size

        def format_size(size):
            # Convert size to a human-readable format
            for unit in ["B", "KB", "MB", "GB", "TB"]:
                if size < 1024.0:
                    return f"{size:.2f} {unit}"
                size /= 1024.0

        total_size_bytes = get_directory_size(path)
        formatted_size = format_size(total_size_bytes)
        logging.info(f"Directory size: {formatted_size}")

    if os.path.isdir(path):
        logging.info(indent + os.path.basename(path) + "/")
        indent += "    "
        for item in os.listdir(path):
            print_directory_tree(
                os.path.join(path, item), indent=indent, display_metadata=False
            )
    else:
        logging.info(indent + os.path.basename(path))


def isin(
    input: TensorDictBase,
    reference: TensorDictBase,
    key: NestedKey,
    dim: int = 0,
) -> Tensor:
    """Tests if each element of ``key`` in input ``dim`` is also present in the reference.

    This function returns a boolean tensor of length  ``input.batch_size[dim]`` that is ``True`` for elements in
    the entry ``key`` that are also present in the ``reference``. This function assumes that both ``input`` and
    ``reference`` have the same batch size and contain the specified entry, otherwise an error will be raised.

    Args:
        input (TensorDictBase): Input TensorDict.
        reference (TensorDictBase): Target TensorDict against which to test.
        key (Nestedkey): The key to test.
        dim (int, optional): The dimension along which to test. Defaults to ``0``.

    Returns:
        out (Tensor): A boolean tensor of length ``input.batch_size[dim]`` that is ``True`` for elements in
            the ``input`` ``key`` tensor that are also present in the ``reference``.

    Examples:
        >>> td = TensorDict(
        ...     {
        ...         "tensor1": torch.tensor([[1, 2, 3], [4, 5, 6], [1, 2, 3], [7, 8, 9]]),
        ...         "tensor2": torch.tensor([[10, 20], [30, 40], [40, 50], [50, 60]]),
        ...     },
        ...     batch_size=[4],
        ... )
        >>> td_ref = TensorDict(
        ...     {
        ...         "tensor1": torch.tensor([[1, 2, 3], [4, 5, 6], [10, 11, 12]]),
        ...         "tensor2": torch.tensor([[10, 20], [30, 40], [50, 60]]),
        ...     },
        ...     batch_size=[3],
        ... )
        >>> in_reference = isin(td, td_ref, key="tensor1")
        >>> expected_in_reference = torch.tensor([True, True, True, False])
        >>> torch.testing.assert_close(in_reference, expected_in_reference)
    """
    # Get the data
    reference_tensor = reference.get(key, default=None)
    target_tensor = input.get(key, default=None)

    # Check key is present in both tensordict and reference_tensordict
    if not isinstance(target_tensor, torch.Tensor):
        raise KeyError(f"Key '{key}' not found in input or not a tensor.")
    if not isinstance(reference_tensor, torch.Tensor):
        raise KeyError(f"Key '{key}' not found in reference or not a tensor.")

    # Check that both TensorDicts have the same number of dimensions
    if len(input.batch_size) != len(reference.batch_size):
        raise ValueError(
            "The number of dimensions in the batch size of the input and reference must be the same."
        )

    # Check dim is valid
    batch_dims = input.ndim
    if dim >= batch_dims or dim < -batch_dims or batch_dims == 0:
        raise ValueError(
            f"The specified dimension '{dim}' is invalid for an input TensorDict with batch size '{input.batch_size}'."
        )

    # Convert negative dimension to its positive equivalent
    if dim < 0:
        dim = batch_dims + dim

    # Find the common indices
    N = reference_tensor.shape[dim]
    cat_data = torch.cat([reference_tensor, target_tensor], dim=dim)
    _, unique_indices = torch.unique(
        cat_data, dim=dim, sorted=True, return_inverse=True
    )
    out = torch.isin(unique_indices[N:], unique_indices[:N], assume_unique=True)

    return out


def _index_preserve_data_ptr(index):
    if isinstance(index, tuple):
        return all(_index_preserve_data_ptr(idx) for idx in index)
    # we can't use a list comprehension here because it fails with tensor indices
    if index is None or index is Ellipsis:
        return True
    if isinstance(index, int):
        return True
    if isinstance(index, slice) and (index.start == 0 or index.start is None):
        return True
    return False


def remove_duplicates(
    input: TensorDictBase,
    key: NestedKey,
    dim: int = 0,
    *,
    return_indices: bool = False,
) -> TensorDictBase:
    """Removes indices duplicated in `key` along the specified dimension.

    This method detects duplicate elements in the tensor associated with the specified `key` along the specified
    `dim` and removes elements in the same indices in all other tensors within the TensorDict. It is expected for
    `dim` to be one of the dimensions within the batch size of the input TensorDict to ensure consistency in all
    tensors. Otherwise, an error will be raised.

    Args:
        input (TensorDictBase): The TensorDict containing potentially duplicate elements.
        key (NestedKey): The key of the tensor along which duplicate elements should be identified and removed. It
            must be one of the leaf keys within the TensorDict, pointing to a tensor and not to another TensorDict.
        dim (int, optional): The dimension along which duplicate elements should be identified and removed. It must be one of
            the dimensions within the batch size of the input TensorDict. Defaults to ``0``.
        return_indices (bool, optional): If ``True``, the indices of the unique elements in the input tensor will be
            returned as well. Defaults to ``False``.

    Returns:
        output (TensorDictBase): input tensordict with the indices corrsponding to duplicated elements
            in tensor `key` along dimension `dim` removed.
        unique_indices (torch.Tensor, optional): The indices of the first occurrences of the unique elements in the
            input tensordict for the specified `key` along the specified `dim`. Only provided if return_index is True.

    Example:
        >>> td = TensorDict(
        ...     {
        ...         "tensor1": torch.tensor([[1, 2, 3], [4, 5, 6], [1, 2, 3], [7, 8, 9]]),
        ...         "tensor2": torch.tensor([[10, 20], [30, 40], [40, 50], [50, 60]]),
        ...     }
        ...     batch_size=[4],
        ... )
        >>> output_tensordict = remove_duplicate_elements(td, key="tensor1", dim=0)
        >>> expected_output = TensorDict(
        ...     {
        ...         "tensor1": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        ...         "tensor2": torch.tensor([[10, 20], [30, 40], [50, 60]]),
        ...     },
        ...     batch_size=[3],
        ... )
        >>> assert (td == expected_output).all()
    """
    tensor = input.get(key, default=None)

    # Check if the key is a TensorDict
    if tensor is None:
        raise KeyError(f"The key '{key}' does not exist in the TensorDict.")

    # Check that the key points to a tensor
    if not isinstance(tensor, torch.Tensor):
        raise KeyError(f"The key '{key}' does not point to a tensor in the TensorDict.")

    # Check dim is valid
    batch_dims = input.ndim
    if dim >= batch_dims or dim < -batch_dims or batch_dims == 0:
        raise ValueError(
            f"The specified dimension '{dim}' is invalid for a TensorDict with batch size '{input.batch_size}'."
        )

    # Convert negative dimension to its positive equivalent
    if dim < 0:
        dim = batch_dims + dim

    # Get indices of unique elements (e.g. [0, 1, 0, 2])
    _, unique_indices, counts = torch.unique(
        tensor, dim=dim, sorted=True, return_inverse=True, return_counts=True
    )

    # Find first occurrence of each index  (e.g. [0, 1, 3])
    _, unique_indices_sorted = torch.sort(unique_indices, stable=True)
    cum_sum = counts.cumsum(0, dtype=torch.long)
    cum_sum = torch.cat(
        (torch.zeros(1, device=input.device, dtype=torch.long), cum_sum[:-1])
    )
    first_indices = unique_indices_sorted[cum_sum]

    # Remove duplicate elements in the TensorDict
    output = input[(slice(None),) * dim + (first_indices,)]

    if return_indices:
        return output, unique_indices

    return output


class _CloudpickleWrapper(object):
    def __init__(self, fn):
        self.fn = fn

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.fn)

    def __setstate__(self, ob: bytes):
        import pickle

        self.fn = pickle.loads(ob)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class _BatchedUninitializedParameter(UninitializedParameter):
    batch_size: torch.Size
    in_dim: int | None = None
    vmap_level: int | None = None

    def materialize(self, shape, device=None, dtype=None):
        UninitializedParameter.materialize(
            self, (*self.batch_size, *shape), device=device, dtype=dtype
        )


class _BatchedUninitializedBuffer(UninitializedBuffer):
    batch_size: torch.Size
    in_dim: int | None = None
    vmap_level: int | None = None

    def materialize(self, shape, device=None, dtype=None):
        UninitializedBuffer.materialize(
            self, (*self.batch_size, *shape), device=device, dtype=dtype
        )


class _add_batch_dim_pre_hook:
    def __call__(self, mod: torch.nn.Module, args, kwargs):
        for name, param in list(mod.named_parameters(recurse=False)):
            if hasattr(param, "in_dim") and hasattr(param, "vmap_level"):
                from torch._C._functorch import _add_batch_dim

                param = _add_batch_dim(param, param.in_dim, param.vmap_level)
                delattr(mod, name)
                setattr(mod, name, param)
        for key, val in list(mod._forward_pre_hooks.items()):
            if val is self:
                del mod._forward_pre_hooks[key]
                return
        else:
            raise RuntimeError("did not find pre-hook")


def is_non_tensor(data):
    """Checks if an item is a non-tensor."""
    return type(data).__dict__.get("_non_tensor", False)

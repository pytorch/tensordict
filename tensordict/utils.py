# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
import time
from functools import wraps
from numbers import Number
from typing import Any, List, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np
import torch
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

    class KeyedJaggedTensor:
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


def _getitem_batch_size(shape: torch.Size, items: IndexType) -> torch.Size:
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

    sanitized_items = []
    for _item in items:
        if isinstance(_item, (list, np.ndarray)):
            _item = torch.tensor(_item)
        elif isinstance(_item, torch.Tensor):
            # np.broadcast will complain if we give it CUDA tensors
            _item = _item.cpu()
        if isinstance(_item, torch.Tensor) and _item.dtype is torch.bool:
            # when using NumPy's advanced indexing patterns, any index containing a
            # boolean array can be equivalently replaced with index.nonzero()
            # note we add unbind(-1) since behaviour of numpy.ndarray.nonzero returns
            # tuples of arrays whereas torch.Tensor.nonzero returns a single tensor
            # https://numpy.org/doc/stable/user/basics.indexing.html#boolean-array-indexing
            sanitized_items.extend(_item.nonzero().unbind(-1))
        else:
            sanitized_items.append(_item)

    # when multiple tensor-like indices are present, they must be broadcastable onto a
    # common shape. if this is satisfied then they are broadcast to that shape, and used
    # to extract diagonal entries of the array.
    # if the tensor indices are contiguous, or separated by scalars, they are replaced
    # in-place by the broadcast shape. if they are separated by non-scalar indices, the
    # broadcast shape is prepended to the new batch size
    # https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
    tensor_indices = []
    contiguous, prev = True, None

    for i, _item in enumerate(sanitized_items):
        if isinstance(_item, torch.Tensor):
            tensor_indices.append(_item)
            if prev is not None and i != prev + 1:
                contiguous = False
            prev = i
        elif isinstance(_item, Number) and prev is not None and i == prev + 1:
            prev = i

    bs = []
    if tensor_indices:
        try:
            b = np.broadcast(*tensor_indices)
        except ValueError:
            raise ValueError(
                "When indexing with tensor-like indices, each of those indices must be "
                "broadcastable to a common shape."
            )
        if not contiguous:
            bs.extend(b.shape)
            b = None
    else:
        b = None

    iter_bs = iter(shape)

    for _item in sanitized_items:
        if isinstance(_item, slice):
            batch = next(iter_bs)
            bs.append(len(range(*_item.indices(batch))))
        elif isinstance(_item, (list, torch.Tensor, np.ndarray)):
            batch = next(iter_bs)
            if b is not None:
                # we haven't yet accounted for tensor indices, so we insert in-place
                bs.extend(b.shape)
                b = None
        elif _item is None:
            bs.append(1)
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

    list_iter_bs = list(iter_bs)
    bs += list_iter_bs
    return torch.Size(bs)


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


def _nested_key_type_check(key: NestedKey) -> None:
    msg = "Expected key to be a string or non-empty tuple of strings, but found {}."
    if type(key) is str:
        return
    is_tuple = type(key) is tuple
    if not is_tuple:
        raise TypeError(msg.format(type(key)))
    else:
        for subkey in key:
            if type(subkey) is not str:
                raise TypeError(msg.format(type(subkey)))


def _normalize_key(key: NestedKey) -> NestedKey:
    # normalises tuples of length one to their string contents
    return key if not isinstance(key, tuple) or len(key) > 1 else key[0]


def index_keyedjaggedtensor(
    kjt: KeyedJaggedTensor,
    index: slice | range | list | torch.Tensor | np.ndarray,
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
        values=values,
        keys=kjt.keys(),
        weights=weights,
        lengths=lengths,
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
        return tensor[index]
    elif isinstance(tensor, KeyedJaggedTensor):
        return index_keyedjaggedtensor(tensor, index)
    else:
        return tensor[index]


def _set_item(
    tensor: torch.Tensor, index: IndexType, value: torch.Tensor
) -> torch.Tensor:
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

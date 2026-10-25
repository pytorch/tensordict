# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import collections
import concurrent.futures
import contextvars
import functools
import html
import itertools
import logging
import math
import os
import queue
import re
import sys
import threading
import time
import warnings
import weakref
from collections import defaultdict
from collections.abc import KeysView
from dataclasses import is_dataclass
from functools import wraps
from numbers import Number
from textwrap import indent
from typing import (
    Any,
    Callable,
    List,
    NamedTuple,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import numpy as np
import pyvers
import torch
from tensordict import _indexing, _utils_key_json
from tensordict._deprecation import (
    deprecated_attributes,
    deprecation_message,
    warn_deprecated,
)
from tensordict._indexing import _getitem_batch_size  # noqa: F401
from tensordict._nestedkey import NestedKey
from torch import Tensor
from torch._C._functorch import (  # noqa: F401  # @manual=fbcode//caffe2:torch
    is_batchedtensor,
)
from torch.compiler import allow_in_graph, assume_constant_result, is_compiling
from torch.nn.parameter import (
    Buffer,
    UninitializedBuffer,
    UninitializedParameter,
    UninitializedTensorMixin,
)
from torch.utils.data._utils.worker import _generate_state
from torch.utils.weak import WeakIdKeyDictionary

try:
    from functorch import dim as ftdim

    _has_funcdim = True
except ImportError:
    _has_funcdim = False

if TYPE_CHECKING:
    from tensordict.base import TensorDictBase
    from tensordict.tensorclass import NonTensorStack

__all__ = [
    # Types
    "DeviceType",
    "IndexType",
    "NestedKey",
    # Constants
    "KEY_SEPARATOR",
    # Classes
    "Buffer",
    "LinkedList",
    "TensorDictFuture",
    "timeit",
    # Functions
    "assert_allclose_td",
    "assert_close",
    "expand_as_right",
    "expand_right",
    "is_bool",
    "is_floating",
    "is_inexact",
    "is_integer",
    "is_non_tensor",
    "is_tensorclass",
    "isin",
    "parse_tensor_dict_string",
    "print_directory_tree",
    "remove_duplicates",
    "unravel_key",
    "unravel_key_list",
    "unravel_key_to_tuple",
    # Configuration
    "capture_non_tensor_stack",
    "get_printoptions",
    "lazy_legacy",
    "list_to_stack",
    "set_capture_non_tensor_stack",
    "set_lazy_legacy",
    "set_list_to_stack",
    "set_printoptions",
    # Logging
    "logger",
]


# Dynamo cannot trace these C++ functions: allow_in_graph lets torch.compile call them
_add_batch_dim = allow_in_graph(torch._C._functorch._add_batch_dim)
_remove_batch_dim = allow_in_graph(torch._C._functorch._remove_batch_dim)


if not _has_funcdim:

    class _ftdim_mock:
        class Dim:
            pass

        class Tensor:
            pass

        def dims(self, *args, **kwargs):
            raise ImportError("functorch.dim not found")

    ftdim = _ftdim_mock  # noqa: F811

T = TypeVar("T", bound="TensorDictBase")

_PIN_MEM_TIMEOUT = 10
_TORCH_DTYPES = (
    torch.bfloat16,
    torch.bool,
    torch.complex128,
    torch.complex32,
    torch.complex64,
    torch.float16,
    torch.float32,
    torch.float64,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.int8,
    torch.qint32,
    torch.qint8,
    torch.quint4x2,
    torch.quint8,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
)
_STR_DTYPE_TO_DTYPE = {str(dtype): dtype for dtype in _TORCH_DTYPES}
_DTYPE_TO_STR_DTYPE = {
    dtype: str_dtype for str_dtype, dtype in _STR_DTYPE_TO_DTYPE.items()
}

#: The type of the indices that a tensordict accepts in ``td[index]``.
#:
#: An int, a slice, ``None``, a tensor (of indices or a boolean mask), a list,
#: or a tuple of these, as for a tensor; a str or a tuple of str is a key.
IndexType = Union[None, int, slice, str, Tensor, List[Any], Tuple[Any, ...]]

#: The type of the ``device`` arguments.
#:
#: A :class:`torch.device`, a str such as ``"cpu"`` or ``"cuda:0"``, or the index
#: of a CUDA device.
DeviceType = Union[torch.device, str, int]

# The default separator of flatten_keys and unflatten_keys. The defaults are
# bound when the methods are defined, so reassigning it changes nothing.
KEY_SEPARATOR: str = "."


_KEY_ERROR = 'key "{}" not found in {} with keys {}'
_LOCK_ERROR = (
    "Cannot modify locked TensorDict. For in-place modification, consider "
    "using the `set_()` method and make sure the key is present."
)


LOGGING_LEVEL = os.environ.get("TD_LOGGING_LEVEL", "INFO")
logger = logging.getLogger("tensordict")
logger.setLevel(LOGGING_LEVEL.upper())
# Disable propagation to the root logger
logger.propagate = False
# Remove all attached handlers
while logger.hasHandlers():
    logger.removeHandler(logger.handlers[0])
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(name)s][%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def _sub_index(tensor: Tensor, idx: IndexType) -> Tensor:
    """Allows indexing of tensors with nested tuples.

     >>> tensor = torch.arange(24).reshape(2, 3, 4)
     >>> tuple1, tuple2 = (0, slice(1, None)), (slice(None), 2)
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


def _copy(self: list[int]) -> list[int]:
    return list(self)


# The deprecated tensordict.utils.prod, which, unlike math.prod, takes no start.
def _prod(sequence):
    return math.prod(sequence)


def expand_as_right(
    tensor: torch.Tensor | TensorDictBase,
    dest: torch.Tensor | TensorDictBase,
) -> torch.Tensor | TensorDictBase:
    """Expands a tensor on the right to the shape of ``dest``.

    Broadcasting adds dimensions on the left; this function adds them on the
    right instead, then expands every dimension of size one. The leading
    dimensions of ``tensor`` must match those of ``dest`` or be of size one.
    The result is a view of ``tensor``, as with :meth:`torch.Tensor.expand`.

    Args:
        tensor (torch.Tensor or TensorDictBase): the tensor to expand.
        dest (torch.Tensor or TensorDictBase): the tensor whose shape the
            result takes. For a tensordict, the shape is its batch size.

    Returns:
        A view of ``tensor`` with the shape of ``dest``.

    Raises:
        RuntimeError: if ``dest`` has fewer dimensions than ``tensor``, or if
            their leading dimensions do not match.

    Examples:
        >>> import torch
        >>> from tensordict.utils import expand_as_right
        >>> x = torch.arange(3)
        >>> expand_as_right(x, torch.zeros(3, 2))
        tensor([[0, 0],
                [1, 1],
                [2, 2]])

    """
    if dest.ndimension() < tensor.ndimension():
        raise RuntimeError(
            "expand_as_right requires the destination tensor to have at least "
            f"as many dimensions as the input tensor, got"
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
    """Expands a tensor on the right to ``shape``.

    Broadcasting adds dimensions on the left; this function adds them on the
    right instead, then expands every dimension of size one. The leading
    dimensions of ``tensor`` must match ``shape`` or be of size one. The
    result is a view of ``tensor``, as with :meth:`torch.Tensor.expand`.

    Args:
        tensor (torch.Tensor): the tensor to expand.
        shape (torch.Size or sequence of int): the shape of the result.

    Returns:
        A view of ``tensor`` with shape ``shape``.

    Examples:
        >>> import torch
        >>> from tensordict.utils import expand_right
        >>> x = torch.arange(3)
        >>> expand_right(x, (3, 2))
        tensor([[0, 0],
                [1, 1],
                [2, 2]])
        >>> expand_right(torch.zeros(3, 4), (3, 4, 5)).shape
        torch.Size([3, 4, 5])

    """
    tensor_expand = tensor
    while tensor_expand.ndimension() < len(shape):
        tensor_expand = tensor_expand.unsqueeze(-1)
    tensor_expand = tensor_expand.expand(shape)
    return tensor_expand


def _populate_np_dtypes():
    d = {}
    for dtype in _TORCH_DTYPES:
        dtype_str = str(dtype).split(".")[-1]
        try:
            d[np.dtype(dtype_str)] = dtype
        except TypeError:
            continue
    return d


_NUMPY_TO_TORCH_DTYPE_DICT = _populate_np_dtypes()

_TORCH_TO_NUMPY_DTYPE_DICT = {
    value: key for key, value in _NUMPY_TO_TORCH_DTYPE_DICT.items()
}


def _is_nested_key(key: NestedKey) -> bool:
    """Returns True if key is a NestedKey."""
    if isinstance(key, str):
        return True
    if key and isinstance(key, (list, tuple)):
        return all(isinstance(subkey, str) for subkey in key)
    return False


def _is_seq_of_nested_key(seq: Sequence[NestedKey]) -> bool:
    """Returns True if seq is a Sequence[NestedKey]."""
    if seq and isinstance(seq, Sequence):
        return all(_is_nested_key(k) for k in seq)
    elif isinstance(seq, Sequence):
        # we allow empty inputs
        return True
    return False


def _shape(tensor: Tensor, nested_shape=False) -> torch.Size:
    if isinstance(tensor, UninitializedTensorMixin):
        return torch.Size([*getattr(tensor, "batch_size", ()), -1])
    elif not isinstance(tensor, Tensor):
        return tensor.shape
    if tensor.is_nested:
        if nested_shape:
            return tensor._nested_tensor_size()
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
    else:
        return tensor.device


def _canonical_stride(shape: Sequence[int]) -> tuple[int, ...]:
    """Returns the C-row-major (canonical) stride for a given shape."""
    stride: list[int] = []
    running = 1
    for size in reversed(tuple(shape)):
        stride.append(running)
        running *= int(size)
    stride.reverse()
    return tuple(stride)


def _canonicalize_tensor(tensor: Tensor) -> Tensor:
    """Materializes a tensor with canonical (C-row-major) strides if needed.

    Tensors whose strides already match the canonical stride for their shape
    are returned unchanged. Otherwise a fresh contiguous copy is allocated and
    the values are copied in.
    """
    if not isinstance(tensor, Tensor):
        return tensor
    if tensor.is_nested or tensor.layout != torch.strided:
        return tensor
    if tuple(tensor.stride()) == _canonical_stride(tensor.shape):
        return tensor
    out = torch.empty_like(tensor, memory_format=torch.contiguous_format)
    out.copy_(tensor)
    return out


def _is_shared(tensor: Tensor) -> bool:
    if isinstance(tensor, Tensor):
        if torch._C._functorch.is_batchedtensor(tensor):
            return None
        return tensor.is_shared()
    if isinstance(tensor, ftdim.Tensor):
        return None
    else:
        return tensor.is_shared()


def _dtype(tensor: Tensor) -> torch.dtype:
    if isinstance(tensor, Tensor):
        return tensor.dtype
    else:
        return tensor.dtype


def _get_item(tensor: Tensor, index: IndexType) -> Tensor:
    try:
        return tensor[index]
    except IndexError as err:
        # try to map list index to tensor, and assess type. If bool, we
        # likely have a nested list of booleans which is not supported by pytorch
        if _is_lis_of_list_of_bools(index):
            index = torch.tensor(index, device=tensor.device)
            if index.dtype is torch.bool:
                raise RuntimeError(
                    "Indexing a tensor with a nested list of boolean values is "
                    "not supported by PyTorch.",
                )
            return tensor[index]
        raise err
    except (RuntimeError, NotImplementedError, ValueError):
        # torch does not index a jagged nested tensor along its batch dim
        if isinstance(tensor, Tensor) and tensor.layout is torch.jagged:
            result = _jagged_getitem(tensor, index)
            if result is not None:
                return result
        raise


def _jagged_sizes(lens: Tensor, *sizes: Tensor) -> list[int | None]:
    """Reads the 0-d tensors ``sizes`` and the min and max of ``lens`` in one device sync.

    Returns ``[*sizes, min(lens), max(lens)]`` as ints, with ``None`` for the
    min and max of an empty ``lens``.
    """
    if lens.numel():
        return torch.stack([*sizes, lens.min(), lens.max()]).tolist()
    return [*(torch.stack(sizes).tolist() if sizes else ()), None, None]


# For each offsets tensor, the last batch-dim op read from it: its key and the
# new offsets and lengths it made. Jagged leaves that share offsets (as the
# token ids and the attention mask of a batch of sequences often do) then
# share the new ones, so that pointwise ops between them still work. torch
# treats offsets and lengths as immutable: it caches the sequence lengths it
# reads from them.
_JAGGED_STRUCTURES = WeakIdKeyDictionary()


def _jagged_structure(tensors: Sequence[Tensor], key, compute: Callable):
    """Returns ``compute()``, or what it returned for the same ``key`` and the same offsets and lengths of ``tensors``.

    ``key`` is a tuple, or a tensor compared by value.
    """
    sources = [(t.offsets(), t.lengths()) for t in tensors]
    entry = _JAGGED_STRUCTURES.get(sources[0][0])
    if entry is not None:
        old_sources, old_key, result = entry
        if (
            (
                old_key == key
                if type(key) is tuple
                else type(old_key) is not tuple
                and old_key.shape == key.shape
                and old_key.device == key.device
                and torch.equal(old_key, key)
            )
            and len(old_sources) == len(sources)
            and all(
                old_offsets() is offsets
                and (old_lengths() if old_lengths is not None else None) is lengths
                for (old_offsets, old_lengths), (offsets, lengths) in zip(
                    old_sources, sources
                )
            )
        ):
            return result
    result = compute()
    # weak references: the entry must not keep its key alive
    _JAGGED_STRUCTURES[sources[0][0]] = (
        [
            (weakref.ref(offsets), None if lengths is None else weakref.ref(lengths))
            for offsets, lengths in sources
        ],
        key,
        result,
    )
    return result


def _jagged_slice_structure(offsets: Tensor, lengths: Tensor | None, start, stop):
    """The offsets, lengths, values range and min and max sequence lengths of the rows ``start:stop``."""
    lens = (offsets.diff() if lengths is None else lengths)[start:stop]
    if lengths is not None:
        # a copy: a view would keep the offsets alive in _JAGGED_STRUCTURES
        return (offsets[start : stop + 1].clone(), lens, None, *_jagged_sizes(lens))
    begin, end, min_seqlen, max_seqlen = _jagged_sizes(
        lens, offsets[start], offsets[stop]
    )
    return (
        offsets[start : stop + 1] - begin,
        None,
        (begin, end - begin),
        min_seqlen,
        max_seqlen,
    )


def _jagged_narrow(tensor: Tensor, structure) -> Tensor:
    """Builds the rows of ``tensor`` that ``_jagged_slice_structure`` describes, as a view."""
    offsets, lengths, values_range, min_seqlen, max_seqlen = structure
    values = tensor.values()
    if values_range is not None:
        values = values.narrow(tensor._ragged_idx - 1, *values_range)
    return torch.nested.nested_tensor_from_jagged(
        values,
        offsets,
        lengths,
        jagged_dim=tensor._ragged_idx,
        min_seqlen=min_seqlen,
        max_seqlen=max_seqlen,
    )


def _jagged_getitem(tensor: Tensor, index: IndexType) -> Tensor | None:
    """Indexes a jagged nested tensor along its batch dim (dim 0), which torch does not do.

    A slice of step 1 returns a view; any other index of the batch dim
    copies the selected rows. The result keeps the ragged dim and has its min
    and max sequence lengths set. Returns ``None`` for the indices it does
    not handle (a ``None``, or a tensor index of the other dims), so that the
    caller raises the error of torch.
    """
    if not isinstance(index, tuple):
        index = (index,)
    first, rest = index[0], index[1:]
    if (
        first is None
        or isinstance(first, bool)
        or not all(type(idx) in (int, slice) for idx in rest)
    ):
        return None
    offsets = tensor.offsets()
    lengths = tensor.lengths()
    num_rows = offsets.numel() - 1
    if isinstance(first, slice) and first.step in (None, 1):
        start, stop, _ = first.indices(num_rows)
        stop = max(start, stop)
        structure = _jagged_structure(
            [tensor],
            ("slice", start, stop),
            lambda: _jagged_slice_structure(offsets, lengths, start, stop),
        )
        result = _jagged_narrow(tensor, structure)
    else:
        rows = torch.arange(num_rows, device=offsets.device)[first]
        if rows.ndim == 0:
            result = tensor[int(rows)]
            return result[rest] if rest else result
        if rows.ndim != 1:
            return None

        def compute():
            lens = (offsets.diff() if lengths is None else lengths)[rows]
            new_offsets = torch.nn.functional.pad(
                lens.cumsum(0, dtype=offsets.dtype), (1, 0)
            )
            return (new_offsets, lens, *_jagged_sizes(lens, new_offsets[-1]))

        new_offsets, lens, total, min_seqlen, max_seqlen = _jagged_structure(
            [tensor], rows, compute
        )
        # the position in values of each element of the selected rows
        positions = torch.repeat_interleave(
            offsets[:-1][rows] - new_offsets[:-1], lens, output_size=total
        ) + torch.arange(total, device=offsets.device)
        result = torch.nested.nested_tensor_from_jagged(
            tensor.values().index_select(tensor._ragged_idx - 1, positions),
            new_offsets,
            jagged_dim=tensor._ragged_idx,
            min_seqlen=min_seqlen,
            max_seqlen=max_seqlen,
        )
    return result[(slice(None), *rest)] if rest else result


def _jagged_split(tensor: Tensor, split_sizes: Sequence[int]) -> tuple[Tensor, ...]:
    """Splits a jagged nested tensor along its batch dim into views, as ``Tensor.split`` does."""
    offsets = tensor.offsets()
    lengths = tensor.lengths()

    def compute():
        structures = []
        start = 0
        for size in split_sizes:
            structures.append(
                _jagged_slice_structure(offsets, lengths, start, start + size)
            )
            start += size
        return structures

    structures = _jagged_structure([tensor], ("split", *split_sizes), compute)
    return tuple(_jagged_narrow(tensor, structure) for structure in structures)


def _jagged_cat(tensors: Sequence[Tensor]) -> Tensor | None:
    """Concatenates jagged nested tensors along their batch dim, which ``torch.cat`` does not do.

    Returns ``None`` unless every tensor is a jagged nested tensor with the
    same ragged dim, so that the caller raises the error of torch.
    """
    if (
        not all(isinstance(t, Tensor) and t.layout is torch.jagged for t in tensors)
        or len({t._ragged_idx for t in tensors}) != 1
    ):
        return None
    ragged_idx = tensors[0]._ragged_idx
    vdim = ragged_idx - 1
    values = [t.values() for t in tensors]
    sizes = [v.shape[vdim] for v in values]

    def compute():
        starts, lens, spans = [], [], []
        shift = 0
        for t, size in zip(tensors, sizes):
            offsets = t.offsets()
            lengths = t.lengths()
            starts.append(offsets[:-1] + shift)
            if lengths is None:
                lens.append(offsets.diff())
                # offsets that do not span the values leave holes, as lengths do
                spans.append((offsets[0] == 0) & (offsets[-1] == size))
            else:
                lens.append(lengths)
            shift += size
        starts.append(starts[-1].new_tensor([shift]))
        lens = torch.cat(lens)
        has_holes = len(spans) < len(tensors)
        if not has_holes:
            spans_all, min_seqlen, max_seqlen = _jagged_sizes(
                lens, torch.stack(spans).all().to(lens.dtype)
            )
            has_holes = not spans_all
        else:
            min_seqlen, max_seqlen = _jagged_sizes(lens)
        return (
            torch.cat(starts),
            lens if has_holes else None,
            min_seqlen,
            max_seqlen,
        )

    offsets, lengths, min_seqlen, max_seqlen = _jagged_structure(
        tensors, ("cat", *sizes), compute
    )
    return torch.nested.nested_tensor_from_jagged(
        torch.cat(values, vdim),
        offsets,
        lengths,
        jagged_dim=ragged_idx,
        min_seqlen=min_seqlen,
        max_seqlen=max_seqlen,
    )


def _cast_scalar(value, tensor):
    """Return a Python scalar ``value`` as torch writes it into ``tensor``: in its dtype and on its device."""
    if isinstance(value, Number) and isinstance(tensor, Tensor):
        return torch.tensor(value, dtype=tensor.dtype, device=tensor.device)
    return value


def _set_item(
    tensor: Tensor, index: IndexType, value: Tensor, *, validated, non_blocking
) -> Tensor:
    # the tensor must be validated
    if not validated:
        raise RuntimeError
    if isinstance(tensor, Tensor):
        if (
            _is_unbatched(value)
            and _is_unbatched(tensor)
            and value.shape == tensor.shape
        ):
            # the whole batch shares the payload: write it whole, as td[:] does
            tensor.copy_(value, non_blocking=non_blocking)
        else:
            tensor[index] = value
        return tensor
    from tensordict.tensorclass import NonTensorData, NonTensorStack

    if is_non_tensor(tensor):
        if isinstance(value, NonTensorData) and isinstance(tensor, NonTensorData):
            try:
                is_same = bool(tensor.data == value.data)
            except Exception:
                # Arrays, tensors and data frames compare elementwise, and the
                # result has no single truth value: write the value anyway.
                is_same = False
            if is_same:
                return tensor
        if isinstance(tensor, NonTensorData):
            tensor = NonTensorStack._from_nontensordata(tensor)
            if (
                isinstance(tensor, NonTensorData)
                and isinstance(value, Tensor)
                and (index is Ellipsis or (isinstance(index, tuple) and not index))
            ):
                # Without batch dims, the index selects the whole entry: the
                # value replaces it, as NonTensorStack.__setitem__ writes an
                # element of a stack
                return type(tensor)(
                    data=value,
                    batch_size=tensor.batch_size,
                    names=tensor._maybe_names(),
                    device=tensor.device,
                )
        # a MetaData is no stack: its __setitem__ leaves it unchanged for a
        # number or a tensor
        if isinstance(tensor, NonTensorStack) and tensor.stack_dim != 0:
            tensor = NonTensorStack(*tensor.unbind(0), stack_dim=0)
        tensor[index] = value
        return tensor
    else:
        tensor[index] = value
        return tensor


class timeit:
    """Measures the time spent in a function or a block of code, by name.

    Use it as a decorator or as a context manager. Each name accumulates
    its call count, its total time and its mean time.
    :meth:`timeit.print` logs them with the tensordict logger, and
    :meth:`timeit.erase` resets them.

    Args:
        name (str): the name under which the time is recorded.

    Examples:
        >>> from tensordict.utils import timeit
        >>> @timeit("step")
        ... def step():
        ...     return 1
        >>> _ = step()
        >>> with timeit("block"):
        ...     _ = step()
        >>> timeit.print()  # logs "block took ... msec" and "step took ... msec"
        >>> timeit.erase()

    """

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
        """Logs the mean and total time of each name, at the ``INFO`` level.

        Args:
            prefix (str, optional): a string put before each line. Defaults
                to ``None``.
        """
        keys = list(timeit._REG)
        keys.sort()
        for name in keys:
            strings = []
            if prefix:
                strings.append(prefix)
            strings.append(
                f"{name} took {timeit._REG[name][0] * 1000:4.4} msec (total = {timeit._REG[name][1]} sec)"
            )
            logger.info(" -- ".join(strings))

    @staticmethod
    def erase():
        """Resets the count and the times of every name to zero."""
        for k in timeit._REG:
            timeit._REG[k] = [0.0, 0.0, 0]


def _int_generator(seed):
    """A pseudo-random chain generator.

    To be used to produce deterministic integer sequences

    Examples:
        >>> for _ in range(2):
        ...     init_int = 10
        ...     ints = []
        ...     for _ in range(10):
        ...        init_int = _int_generator(init_int)
        ...        ints.append(init_int)
        ...     print(ints)
        [6756, 1717, 4410, 9740, 9611, 9716, 5397, 7745, 4521, 7523]
        [6756, 1717, 4410, 9740, 9611, 9716, 5397, 7745, 4521, 7523]
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
    """Returns ``True`` if ``obj`` is a tensorclass type or an instance of one.

    Args:
        obj (type or object): the class or the object to check.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict, is_tensorclass, tensorclass
        >>> @tensorclass
        ... class Data:
        ...     x: torch.Tensor
        >>> is_tensorclass(Data), is_tensorclass(Data(x=torch.zeros(2)))
        (True, True)
        >>> is_tensorclass(TensorDict())
        False
    """
    cls = obj if isinstance(obj, type) else type(obj)
    return _is_tensorclass(cls)


_TENSORCLASS_MEMO = {}

# Set while the archive writer stages placeholder leaves (see
# ``_write_archive_from_td``): tensorclass ``__post_init__`` must not run on
# that data, which is uninitialized or zero until the real bytes are written.
_SKIP_TENSORCLASS_POST_INIT = contextvars.ContextVar(
    "_SKIP_TENSORCLASS_POST_INIT", default=False
)


def _is_tensorclass(cls: type) -> bool:
    # Under Dynamo the memo is not read: a lookup that misses guards on all the
    # keys of the memo, so any class memoized later in eager would recompile.
    is_dynamo = is_compiling()
    out = None
    if not is_dynamo:
        out = _TENSORCLASS_MEMO.get(cls)
    if out is None:
        out = getattr(cls, "_is_tensorclass", False)
        if not is_dynamo:
            _TENSORCLASS_MEMO[cls] = out
    return out


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
    """Creates a key for the cache such that memory footprint is minimized."""
    # Fast path for the common args
    if not args and not kwargs:
        return ((), ())
    elif not kwargs and len(args) == 1 and type(args[0]) is str:
        return (args, ())
    else:
        return (
            tuple(_unfold_sequence(args)),
            tuple(_unfold_sequence(sorted(kwargs.items()))),
        )


def _cache_while_locked(fun):
    """A cache for TensorDictBase subclasses.

    This decorator will cache the values returned by a method as long as the
    input arguments match.
    Leaves (tensors and such) are not cached.
    The cache is stored within the tensordict such that it can be erased at any
    point in time.

    Examples:
        >>> from tensordict import TensorDict
        >>> class SomeOtherTd(TensorDict):
        ...     @_cache_while_locked
        ...     def all_keys(self):
        ...         return set(self.keys(include_nested=True))
        >>> td = SomeOtherTd({("a", "b", "c", "d", "e", "f", "g"): 1.0}, [])
        >>> td.all_keys() is td.all_keys()
        False
        >>> td = td.lock_()
        >>> td.all_keys() is td.all_keys()
        True
    """

    @wraps(fun)
    def newfun(_self: "TensorDictBase", *args, **kwargs):
        if not _self.is_locked or is_compiling():
            return fun(_self, *args, **kwargs)
        cache = _self._cache
        if cache is None:
            cache = _self._cache = defaultdict(dict)
        cache = cache[fun.__name__]
        key = _make_cache_key(args, kwargs)
        if key not in cache:
            out = fun(_self, *args, **kwargs)
            if not isinstance(out, Tensor):
                # we don't cache tensors to avoid filling the mem and / or
                # stacking them from their origin
                cache[key] = out
        else:
            out = cache[key]
        return out

    return newfun


def _copy_cached_result(fun):
    """Returns a shallow copy of the tensordict cached by ``_cache_while_locked``.

    Put it above ``@_cache_while_locked`` on methods that return a new, unlocked
    tensordict, so that a write into the result does not change the cache and
    later calls.
    """

    @wraps(fun)
    def newfun(_self: "TensorDictBase", *args, **kwargs):
        out = fun(_self, *args, **kwargs)
        if _self.is_locked and not is_compiling() and not out.is_locked:
            out = out.copy()
        return out

    return newfun


def _erase_cache_first(fun):
    """A decorator to erase the cache at each call."""

    @wraps(fun)
    def new_fun(self, *args, **kwargs):
        self._erase_cache()
        return fun(self, *args, **kwargs)

    return new_fun


_NON_STR_KEY_ERR = "TensorDict keys are always strings. Membership checks are only supported for strings or non-empty tuples of strings (for nested TensorDicts)"
_GENERIC_NESTED_ERR = "Only NestedKeys are supported: a key must be a string or a non-empty, possibly nested tuple of strings. Got key {!r}."


def _warn_nested_key_in_flat_keys(key: tuple[str, ...], tensordict) -> None:
    """Warns that ``key in tensordict.keys()`` finds the nested ``key`` today, and returns False in 0.17."""
    warnings.warn(
        deprecation_message(
            f"Finding the nested key {key} in {type(tensordict).__name__}.keys() "
            "without include_nested=True",
            removal="0.17",
        )
        + " Starting with TensorDict 0.17, the check returns False, as it does "
        "for TensorDict. Use keys(include_nested=True) to look up a nested key.",
        FutureWarning,
        # point to the code that runs `key in view`
        stacklevel=3,
    )


class _StringKeys(KeysView):
    """A key view where contains is restricted to strings.

    Saving the keys as an attribute is 25% faster than just subclassing KeysView.

    """

    def __init__(self, keys):
        self.keys = keys

    def __getitem__(self, key: str) -> Any:
        return self.keys.__getitem__(key)

    def __iter__(self):
        return iter(self.keys)

    def __repr__(self):
        return f"{type(self).__name__}({self.keys})"

    def __len__(self):
        return len(self.keys)

    def __contains__(self, item):
        if not isinstance(item, str):
            try:
                unravel_item = _unravel_key_to_tuple(item)
                if not unravel_item:  # catch errors during unravel
                    raise TypeError
            except Exception:
                raise TypeError(_NON_STR_KEY_ERR)
            if len(unravel_item) > 1:
                # A nested key is not a top-level key
                return False
            else:
                item = unravel_item[0]
        return self.keys.__contains__(item)


_StringOnlyDict = dict


class _LockedSchema(NamedTuple):
    """Immutable per-TD snapshot consulted by locked-fast-paths.

    Built once when a :class:`TensorDict` is locked, dropped when it is
    unlocked. Holds the bits Dynamo would otherwise re-derive (and
    re-guard on) by iterating ``self._tensordict`` on every call:

    Attributes:
        keys: tuple of the top-level keys, in the order they were stored
            when the lock was taken. Used by locked-fast-paths to walk
            the entry set without emitting ``DICT_KEYS_MATCH`` guards on
            the underlying Python dict.

    Locked TDs already disallow key insertion/removal, so this snapshot
    cannot go stale while the TD stays locked.
    """

    keys: tuple


def _lock_blocked(func):
    """Checks that the tensordict is unlocked before executing a function."""

    @wraps(func)
    def new_func(self, *args, **kwargs):
        if (
            not kwargs.get("ignore_lock", False)
            and self.is_locked
            and not kwargs.get("inplace")
        ):
            raise RuntimeError(_LOCK_ERROR)
        return func(self, *args, **kwargs)

    return new_func


class _UpdateIgnoreLockWarned:
    """The type of :data:`_UPDATE_IGNORE_LOCK_WARNED`."""

    __slots__ = ()

    def __bool__(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "_UPDATE_IGNORE_LOCK_WARNED"


# A truthy ``ignore_lock`` value. A public ``update`` method that warns about
# ``ignore_lock=True`` passes it down instead of ``True``, so that the nested
# ``update`` calls behave the same but do not warn again.
_UPDATE_IGNORE_LOCK_WARNED = _UpdateIgnoreLockWarned()


def _warn_update_ignore_lock(stacklevel: int = 2) -> _UpdateIgnoreLockWarned:
    """Warns that ``update(ignore_lock=True)`` is deprecated.

    Call it from a public ``update`` method only when ``ignore_lock`` is truthy
    and is not :data:`_UPDATE_IGNORE_LOCK_WARNED`, and pass the returned value
    down as ``ignore_lock``. ``stacklevel`` counts as in :func:`warnings.warn`
    from that ``update`` method: ``2`` points to its caller, ``3`` to the
    caller of a decorated ``update``.
    """
    warn_deprecated(
        "update(ignore_lock=True)",
        removal="0.17",
        replacement="'with td.unlock_(): td.update(...)'",
        stacklevel=stacklevel + 1,
    )
    return _UPDATE_IGNORE_LOCK_WARNED


def _strong_ref(obj):
    """Return a zero-arg callable equivalent to ``weakref.ref(obj)`` but holding a strong reference.

    Used under ``torch.compile`` where ``weakref.ref`` cannot be traced
    cleanly (Dynamo would emit a ``WeakRefVariable`` that fails to be
    reconstructed when the holding TD is returned from the compiled
    region). A strong reference is safe here because the compiled graph
    is transient: the closure is dropped as soon as the outer call ends,
    so it cannot create a real lifecycle cycle.
    """
    return lambda: obj


def _as_context_manager(attr=None):
    """Converts a method to a decorator.

    Examples:
        >>> from tensordict import TensorDict
        >>> data = TensorDict()
        >>> with data.lock_(): # lock_ is decorated
        ...     assert data.is_locked
        >>> assert not data.is_locked
    """

    def __call__(func):
        if attr is not None:

            @wraps(func)
            def func_as_decorator(_self, *args, **kwargs):
                # Under torch.compile, ``weakref.ref`` cannot be traced
                # cleanly (Dynamo emits a ``WeakRefVariable`` that fails
                # reconstruction when the holding TD is returned from the
                # compiled region). Fall back to a strong-ref closure so
                # the context-manager semantics (``with td.lock_(): ...``)
                # still work, but the resulting object is a regular
                # Python callable that Dynamo can handle.
                _make_ref = _strong_ref if is_compiling() else weakref.ref
                _attr_pre = getattr(_self, attr)
                out = func(_self, *args, **kwargs)
                _attr_post = getattr(_self, attr)
                if out is not None:
                    if _attr_post is not _attr_pre:
                        ref = _make_ref(_self)
                        out_lo = out
                        if is_tensorclass(out_lo):
                            # We write in the tensordict but the ref is still to self (the tensorclass object)
                            #  we do this because we don't want to call the __setattr__ of the tensorclass
                            out_lo = out_lo._tensordict
                        out_lo._last_op = (
                            func.__name__,
                            (
                                args,
                                kwargs,
                                ref,
                            ),
                        )
                    else:
                        out._last_op = None
                return out

        else:

            @wraps(func)
            def func_as_decorator(_self, *args, **kwargs):
                # See note in the attr-aware branch above.
                _make_ref = _strong_ref if is_compiling() else weakref.ref
                out = func(_self, *args, **kwargs)
                if out is not None:
                    ref = _make_ref(_self)
                    out_lo = out
                    if is_tensorclass(out_lo):
                        # We write in the tensordict but the ref is still to self (the tensorclass object)
                        #  we do this because we don't want to call the __setattr__ of the tensorclass
                        out_lo = out_lo._tensordict

                    out_lo._last_op = (func.__name__, (args, kwargs, ref))
                return out

        return func_as_decorator

    return __call__


def _find_smallest_uint(N):
    if N < 0:
        raise ValueError("N must be a non-negative integer")

    int8_max = 127
    int16_max = 32767
    int32_max = 2147483647
    int64_max = 9223372036854775807
    if N <= int8_max:
        return torch.int8
    elif N <= int16_max:
        return torch.int16
    elif N <= int32_max:
        return torch.int32
    elif N <= int64_max:
        return torch.int64
    else:
        return "uint is too large to be represented by uint64"


def _split_tensordict(
    td,
    chunksize,
    num_chunks,
    num_workers,
    dim,
    use_generator=False,
    to_tensordict=False,
    shuffle=False,
):
    if shuffle and not use_generator:
        raise RuntimeError(
            "Shuffling is not permitted unless use_generator is set to ``True`` for efficiency purposes."
        )
    if chunksize is None and num_chunks is None:
        num_chunks = num_workers
    if chunksize is not None and num_chunks is not None:
        raise ValueError(
            "Either chunksize or num_chunks must be provided, but not both."
        )
    if num_chunks is not None:
        num_chunks = min(td.shape[dim], num_chunks)
        if use_generator:

            def next_index(td=td, dim=dim, num_chunks=num_chunks):
                idx_start = 0
                n = td.shape[dim]
                chunksize = -(n // -num_chunks)
                idx_end = chunksize
                while idx_start < n:
                    yield slice(idx_start, idx_end)
                    idx_start = idx_end
                    idx_end += chunksize

        else:
            return td.chunk(num_chunks, dim=dim)
    else:
        if chunksize == 0:
            if use_generator:

                def next_index(td=td, dim=dim):
                    yield from range(td.shape[dim])

            else:
                return td.unbind(dim=dim)
        else:
            if use_generator:

                def next_index(td=td, dim=dim, chunksize=chunksize):
                    idx_start = 0
                    idx_end = chunksize
                    n = td.shape[dim]
                    while idx_start < n:
                        yield slice(idx_start, idx_end)
                        idx_start = idx_end
                        idx_end += chunksize

            else:
                chunksize = min(td.shape[dim], chunksize)
                return td.split(chunksize, dim=dim)
    # end up here only when use_generator = True
    if shuffle:

        def next_index_shuffle(next_index=next_index):
            n = td.shape[dim]
            device = td.device
            rp = torch.randperm(n, dtype=_find_smallest_uint(n), device=device)
            for idx in next_index():
                yield rp[idx].long()

        next_index = next_index_shuffle

    def _split_generator():
        base = (slice(None),) * dim
        for idx in next_index():
            out = td[base + (idx,)]
            if to_tensordict:
                out = out.to_tensordict()
            yield out

    return _split_generator()


def _parse_to(*args, **kwargs):
    batch_size = kwargs.pop("batch_size", None)
    non_blocking_pin = kwargs.pop("non_blocking_pin", False)
    num_threads = kwargs.pop("num_threads", None)
    other = kwargs.pop("other", None)
    inplace = kwargs.pop("inplace", False)
    # _to reads cast_filter from the kwargs that to() received
    kwargs.pop("cast_filter", None)
    if not is_compiling():
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
    else:
        non_blocking = kwargs.get("non_blocking", False)
        convert_to_format = kwargs.get("memory_format")
        if len(args) > 0 and isinstance(args[0], torch.dtype):
            # td.to(dtype)
            device = kwargs.get("device")
            dtype = args[0]
        elif len(args) > 0 and isinstance(args[0], torch.Tensor):
            # td.to(tensor)
            device = args[0].device
            dtype = args[0].dtype
        elif len(args) > 0:
            device = torch.device(args[0])
            if len(args) > 1:
                dtype = args[1]
            else:
                dtype = kwargs.get("dtype")
        else:
            device = kwargs.get("device")
            dtype = kwargs.get("dtype")
        if device is not None:
            device = torch.device(device)

    if device and device.type == "cuda" and device.index is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    if other is not None:
        if device is not None and device != other.device:
            raise ValueError("other and device cannot be both passed")
        device = other.device
        dtypes = {val.dtype for val in other.values(True, True)}
        if len(dtypes) > 1 or len(dtypes) == 0:
            dtype = None
        elif len(dtypes) == 1:
            dtype = list(dtypes)[0]
    return (
        device,
        dtype,
        non_blocking,
        convert_to_format,
        batch_size,
        non_blocking_pin,
        num_threads,
        inplace,
    )


def _positional_non_blocking(args):
    # The non_blocking of to(device, dtype, non_blocking), to(dtype, non_blocking)
    # and to(tensor, non_blocking), when it is not passed by keyword. A bool
    # first is not non_blocking: torch.Tensor.to does not read it as one
    for arg in args[1:]:
        if type(arg) is bool:
            if not arg:
                return False
            if not is_compiling():
                warnings.warn(
                    "to() ignores a positional non_blocking=True: it copies as "
                    "for the default non_blocking=None, which synchronizes "
                    "after a copy to a device other than CUDA. Starting with "
                    "TensorDict 0.17, the default will change to honoring it "
                    "as the keyword non_blocking=True is, without that "
                    "synchronization. Pass non_blocking=True by keyword to get "
                    "the new behaviour now, or drop it to keep the current one.",
                    FutureWarning,
                    stacklevel=_user_stacklevel(),
                )
            return None
    return None


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
            return f'{self.prefix} "{self.key}" /{msg[len(self.prefix) :]}'
        return f'{self.prefix} "{self.key}". {msg}'

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, _):
        if exc_type is self.exc_type and (
            self.exc_msg is None or self.exc_msg in str(exc_value)
        ):
            exc_value.args = (self._add_key_to_error_msg(str(exc_value)),)


class _TensorDictPropertyError(RuntimeError):
    """Raised when recursive TensorDict metadata validation fails."""

    pass


def _check_recursive_properties(
    tensordict: T,
    *,
    batch_size: bool = True,
    shape: bool = True,
    device: bool = True,
    lock: bool = True,
    names: bool = True,
    requires_grad: bool | None = None,
    is_shared: bool | None = None,
    is_memmap: bool | None = None,
    allow_none: bool = False,
    include_non_tensor: bool = True,
    include_unbatched: bool = True,
    strict: bool = True,
    raise_exception: bool = True,
) -> bool:
    """Recursively validates TensorDict metadata.

    This private checker is intended for tests and debugging. It walks the
    logical TensorDict tree and verifies that node metadata and leaf metadata
    are consistent with their parent context. The function is deliberately
    conservative: callers can disable individual checks when a test is about a
    known special case.
    """
    from tensordict.base import _is_tensor_collection

    if not _is_tensor_collection(type(tensordict)):
        raise TypeError(
            "Expected a TensorDictBase or tensorclass instance, "
            f"got {type(tensordict)}."
        )

    _MISSING = object()
    errors = []

    def _path_str(path):
        if not path:
            return "<root>"
        return repr(path)

    def _type_name(obj):
        return type(obj).__name__

    def _append_key(path, key):
        key = _unravel_key_to_tuple(key)
        if key:
            return path + key
        return path + (key,)

    def _add_error(path, message):
        errors.append(f"{message} at key {_path_str(path)}.")

    def _read_attr(obj, attr, path):
        try:
            value = getattr(obj, attr)
        except Exception as err:
            _add_error(
                path,
                f"Could not read {attr} from {_type_name(obj)}: {err}",
            )
            return _MISSING
        return value

    def _read_call(obj, name, path):
        try:
            value = getattr(obj, name)()
        except Exception as err:
            _add_error(
                path,
                f"Could not call {name} on {_type_name(obj)}: {err}",
            )
            return _MISSING
        return value

    def _as_size(value, path, label):
        if value is _MISSING:
            return _MISSING
        if value is None and allow_none:
            return value
        try:
            return torch.Size(value)
        except Exception as err:
            _add_error(
                path, f"Could not convert {label}={value!r} to torch.Size: {err}"
            )
            return _MISSING

    def _same_size(first, second):
        return tuple(first) == tuple(second)

    def _has_batch_prefix(child_size, parent_size):
        if len(child_size) < len(parent_size):
            return False
        return _same_size(child_size[: len(parent_size)], parent_size)

    def _check_expected_bool(value, expected, path, label):
        if value is _MISSING:
            return
        if value is None and allow_none:
            return
        if value is None:
            _add_error(path, f"{label} mismatch: expected {expected}, got None")
            return
        if bool(value) != expected:
            _add_error(path, f"{label} mismatch: expected {expected}, got {value}")

    def _leaf_is_memmap(value):
        try:
            from tensordict.memmap import MemoryMappedTensor
        except ImportError:
            return False
        return isinstance(value, MemoryMappedTensor)

    def _check_node_metadata(node, path, parent_metadata):
        node_batch_size = _MISSING
        node_device = _MISSING
        node_lock = _MISSING
        node_names = _MISSING

        if batch_size or shape or names:
            node_batch_size = _as_size(
                _read_attr(node, "batch_size", path), path, "batch_size"
            )
            if (
                batch_size
                and node_batch_size is not _MISSING
                and node_batch_size is not None
            ):
                batch_dims = _read_attr(node, "batch_dims", path)
                if batch_dims is not _MISSING and batch_dims is not None:
                    if batch_dims != len(node_batch_size):
                        _add_error(
                            path,
                            "Batch-dims mismatch: "
                            f"batch_dims={batch_dims}, batch_size={node_batch_size}",
                        )

        if shape:
            node_shape = _as_size(_read_attr(node, "shape", path), path, "shape")
            if (
                node_shape is not _MISSING
                and node_shape is not None
                and node_batch_size is not _MISSING
                and node_batch_size is not None
                and not _same_size(node_shape, node_batch_size)
            ):
                _add_error(
                    path,
                    f"Shape mismatch: shape={node_shape}, batch_size={node_batch_size}",
                )

        if device:
            node_device = _read_attr(node, "device", path)
            parent_device = parent_metadata.get("device")
            if (
                parent_device not in (None, _MISSING)
                and node_device not in (None, _MISSING)
                and torch.device(node_device) != torch.device(parent_device)
            ):
                _add_error(
                    path,
                    "Device mismatch: "
                    f"parent device={parent_device}, child device={node_device}",
                )

        if lock:
            node_lock = _read_attr(node, "is_locked", path)
            parent_lock = parent_metadata.get("lock")
            if (
                parent_lock is not _MISSING
                and node_lock is not _MISSING
                and parent_lock is not None
                and node_lock is not None
            ):
                if bool(parent_lock) and not bool(node_lock):
                    _add_error(
                        path,
                        "Lock mismatch: "
                        f"parent is_locked={parent_lock}, child is_locked={node_lock}",
                    )
                elif strict and bool(node_lock) != bool(parent_lock):
                    _add_error(
                        path,
                        "Lock mismatch: "
                        f"parent is_locked={parent_lock}, child is_locked={node_lock}",
                    )

        if names:
            node_names = _read_attr(node, "names", path)
            if (
                node_names is not _MISSING
                and node_names is not None
                and node_batch_size is not _MISSING
                and node_batch_size is not None
            ):
                if len(node_names) != len(node_batch_size):
                    _add_error(
                        path,
                        "Names mismatch: "
                        f"batch_size={node_batch_size}, names={node_names}",
                    )
                parent_names = parent_metadata.get("names")
                parent_batch_size = parent_metadata.get("batch_size")
                if (
                    parent_names not in (None, _MISSING)
                    and parent_batch_size not in (None, _MISSING)
                    and len(node_names) >= len(parent_names)
                ):
                    node_prefix = tuple(node_names[: len(parent_names)])
                    parent_prefix = tuple(parent_names)
                    if strict:
                        names_match = node_prefix == parent_prefix
                    else:
                        names_match = all(
                            parent_name is None or child_name == parent_name
                            for child_name, parent_name in _zip_strict(
                                node_prefix, parent_prefix
                            )
                        )
                    if not names_match:
                        _add_error(
                            path,
                            "Names mismatch: "
                            f"parent names={parent_names}, child names={node_names}",
                        )

        if parent_metadata.get("batch_size") not in (None, _MISSING):
            parent_batch_size = parent_metadata["batch_size"]
            if (
                batch_size
                and node_batch_size is not _MISSING
                and node_batch_size is not None
                and not _has_batch_prefix(node_batch_size, parent_batch_size)
            ):
                _add_error(
                    path,
                    "Batch-size mismatch: "
                    f"parent batch_size={parent_batch_size}, "
                    f"child batch_size={node_batch_size}, "
                    f"child type={_type_name(node)}",
                )

        if is_shared is not None:
            _check_expected_bool(
                _read_call(node, "is_shared", path),
                is_shared,
                path,
                "is_shared",
            )
        if is_memmap is not None:
            _check_expected_bool(
                _read_call(node, "is_memmap", path),
                is_memmap,
                path,
                "is_memmap",
            )

        return {
            "batch_size": node_batch_size,
            "device": node_device,
            "lock": node_lock,
            "names": node_names,
        }

    def _check_leaf_common(value, path, parent_metadata):
        parent_device = parent_metadata.get("device")
        if device and parent_device not in (None, _MISSING):
            leaf_device = _read_attr(value, "device", path)
            if leaf_device not in (None, _MISSING) and torch.device(
                leaf_device
            ) != torch.device(parent_device):
                _add_error(
                    path,
                    "Device mismatch: "
                    f"parent device={parent_device}, leaf device={leaf_device}",
                )

        if requires_grad is not None:
            grad = _read_attr(value, "requires_grad", path)
            _check_expected_bool(grad, requires_grad, path, "requires_grad")

        if is_shared is not None:
            try:
                shared = _is_shared(value)
            except Exception as err:
                _add_error(
                    path,
                    f"Could not read is_shared from {_type_name(value)}: {err}",
                )
            else:
                _check_expected_bool(shared, is_shared, path, "is_shared")

        if is_memmap is not None:
            _check_expected_bool(_leaf_is_memmap(value), is_memmap, path, "is_memmap")

    def _check_regular_tensor_leaf(value, path, parent_metadata):
        parent_batch_size = parent_metadata.get("batch_size")
        if (batch_size or shape) and parent_batch_size not in (None, _MISSING):
            leaf_shape = _as_size(_read_attr(value, "shape", path), path, "shape")
            if (
                leaf_shape is not _MISSING
                and leaf_shape is not None
                and not _has_batch_prefix(leaf_shape, parent_batch_size)
            ):
                _add_error(
                    path,
                    "Shape mismatch: "
                    f"parent batch_size={parent_batch_size}, leaf shape={leaf_shape}, "
                    f"leaf type={_type_name(value)}",
                )
        _check_leaf_common(value, path, parent_metadata)

    def _check_unbatched_leaf(value, path, parent_metadata):
        if not include_unbatched:
            return
        parent_batch_size = parent_metadata.get("batch_size")
        leaf_batch_size = _as_size(
            _read_attr(value, "batch_size", path), path, "batch_size"
        )
        if (
            batch_size
            and parent_batch_size not in (None, _MISSING)
            and leaf_batch_size not in (None, _MISSING)
            and not _same_size(leaf_batch_size, parent_batch_size)
            and not (allow_none and len(leaf_batch_size) == 0)
        ):
            _add_error(
                path,
                "Batch-size mismatch: "
                f"parent batch_size={parent_batch_size}, "
                f"leaf batch_size={leaf_batch_size}, "
                f"leaf type={_type_name(value)}",
            )
        _check_leaf_common(value, path, parent_metadata)

    def _check_non_tensor_leaf(value, path, parent_metadata):
        if not include_non_tensor:
            return
        parent_batch_size = parent_metadata.get("batch_size")
        leaf_batch_size = _as_size(
            _read_attr(value, "batch_size", path), path, "batch_size"
        )
        if (
            batch_size
            and parent_batch_size not in (None, _MISSING)
            and leaf_batch_size not in (None, _MISSING)
            and not _same_size(leaf_batch_size, parent_batch_size)
            and not (allow_none and len(leaf_batch_size) == 0)
        ):
            _add_error(
                path,
                "Batch-size mismatch: "
                f"parent batch_size={parent_batch_size}, "
                f"leaf batch_size={leaf_batch_size}, "
                f"leaf type={_type_name(value)}",
            )
        if shape:
            leaf_shape = _as_size(_read_attr(value, "shape", path), path, "shape")
            if (
                leaf_shape not in (None, _MISSING)
                and leaf_batch_size not in (None, _MISSING)
                and not _same_size(leaf_shape, leaf_batch_size)
            ):
                _add_error(
                    path,
                    f"Shape mismatch: shape={leaf_shape}, batch_size={leaf_batch_size}",
                )
        _check_leaf_common(value, path, parent_metadata)

    def _visit_leaf(value, path, parent_metadata):
        if _is_unbatched(value):
            _check_unbatched_leaf(value, path, parent_metadata)
        elif is_non_tensor(value):
            _check_non_tensor_leaf(value, path, parent_metadata)
        elif isinstance(value, torch.Tensor):
            _check_regular_tensor_leaf(value, path, parent_metadata)
        elif strict:
            _add_error(path, f"Unsupported leaf type {_type_name(value)}")

    def _visit_node(node, path, parent_metadata):
        node_metadata = _check_node_metadata(node, path, parent_metadata)
        try:
            items = tuple(node.items(include_nested=False))
        except Exception as err:
            _add_error(path, f"Could not iterate over {_type_name(node)} items: {err}")
            return
        for key, value in items:
            child_path = _append_key(path, key)
            if is_non_tensor(value):
                _check_non_tensor_leaf(value, child_path, node_metadata)
            elif _is_tensor_collection(type(value)):
                _visit_node(value, child_path, node_metadata)
            else:
                _visit_leaf(value, child_path, node_metadata)

    _visit_node(
        tensordict,
        (),
        {
            "batch_size": _MISSING,
            "device": _MISSING,
            "lock": _MISSING,
            "names": _MISSING,
        },
    )
    if errors:
        if raise_exception:
            formatted_errors = "\n".join(
                f"{idx}. {error}" for idx, error in enumerate(errors, 1)
            )
            raise _TensorDictPropertyError(
                "TensorDict recursive property check failed with "
                f"{len(errors)} error(s):\n{formatted_errors}"
            )
        return False
    return True


def assert_close(
    actual: T,
    expected: T,
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = True,
    intersection: bool = False,
    msg: str = "",
    prefix: NestedKey = (),
) -> bool:
    """Asserts that two tensordicts have the same keys and close values.

    Each pair of tensors is compared with :func:`torch.testing.assert_close`,
    which also checks that the dtypes, shapes and devices match. Nested
    tensordicts are compared entry by entry, and two lazy stacks are compared
    stack by stack. Non-tensor data is not compared. Tensorclasses are
    accepted.

    ``assert_allclose_td`` is another name for this function.

    Args:
        actual (TensorDictBase): the tensordict to check.
        expected (TensorDictBase): the tensordict with the expected values.
        rtol (float, optional): the relative tolerance. Defaults to ``None``,
            which uses the default of :func:`torch.testing.assert_close` for
            the dtype.
        atol (float, optional): the absolute tolerance. Defaults to ``None``,
            which uses the default of :func:`torch.testing.assert_close` for
            the dtype.
        equal_nan (bool, optional): if ``True``, two ``NaN`` values are equal.
            Defaults to ``True``.
        intersection (bool, optional): if ``True``, compares only the keys
            that both tensordicts have. If ``False``, different key sets
            raise a ``KeyError``. Defaults to ``False``.
        msg (str, optional): a message appended to the error message.
            Defaults to ``""``.
        prefix (NestedKey, optional): a key prepended to the keys in the error
            messages. Defaults to ``()``.

    Returns:
        ``True`` if the check passes.

    Raises:
        TypeError: if ``actual`` or ``expected`` is not a tensordict or a
            tensorclass.
        KeyError: if the keys differ and ``intersection=False``.
        AssertionError: if two values are not close. The message names the
            key and the root mean squared error, followed by
            :func:`torch.testing.assert_close`'s description of the mismatch.
        RuntimeError: if the shapes of two tensors cannot be broadcast
            together.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict, assert_close
        >>> a = TensorDict(x=torch.zeros(3), y=torch.ones(3))
        >>> assert_close(a, a.clone())
        True
        >>> b = TensorDict(x=torch.full((3,), 0.01), y=torch.ones(3))
        >>> assert_close(a, b, atol=0.1, rtol=0)
        True
        >>> assert_close(a, b)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        AssertionError: key ('x',) does not match, got mse = 0.0100
        Tensor-likes are not close!
        ...

    """
    from tensordict.base import _is_tensor_collection

    if not _is_tensor_collection(type(actual)) or not _is_tensor_collection(
        type(expected)
    ):
        raise TypeError(
            f"assert_allclose inputs must be of TensorDict type, got {type(actual)} and {type(expected)}"
        )

    from tensordict._lazy import LazyStackedTensorDict

    if is_tensorclass(actual):
        actual = actual._tensordict
    if is_tensorclass(expected):
        expected = expected._tensordict

    if isinstance(actual, LazyStackedTensorDict) and isinstance(
        expected, LazyStackedTensorDict
    ):
        if expected.stack_dim != actual.stack_dim:
            # turn expected in actual stack dim
            expected = expected.to_lazystack(actual.stack_dim)

        for sub_actual, sub_expected in _zip_strict(
            actual.tensordicts, expected.tensordicts
        ):
            assert_close(
                sub_actual,
                sub_expected,
                rtol=rtol,
                atol=atol,
                msg=msg,
                intersection=intersection,
                equal_nan=equal_nan,
                prefix=prefix,
            )
        return True

    try:
        set1 = set(actual.keys())
        set2 = set(expected.keys())
    except ValueError:
        # Persistent tensordicts do not work with is_leaf
        def istensor(cls):
            return issubclass(cls, torch.Tensor)

        set1 = set(actual.keys(is_leaf=istensor))
        set2 = set(expected.keys(is_leaf=istensor))
    if not intersection and (
        not (len(set1.difference(set2)) == 0 and len(set2) == len(set1))
    ):
        _mismatch_keys(set1, set2)
    elif intersection and set1 != set2:
        actual = actual.select(*set2, strict=False)
        expected = expected.select(*set1, strict=False)

    keys = sorted(actual.keys(), key=str)
    for key in keys:
        input1 = actual.get(key)
        input2 = expected.get(key)
        if _is_tensor_collection(type(input1)):
            if is_non_tensor(input1):
                # We skip non-tensor data
                continue
            assert_close(
                input1,
                input2,
                rtol=rtol,
                atol=atol,
                msg=msg,
                intersection=intersection,
                equal_nan=equal_nan,
                prefix=prefix + (key,),
            )
            continue
        elif not isinstance(input1, torch.Tensor):
            continue
        try:
            if input1.is_nested:
                input1v = input1.values()
                input2v = input2.values()
                mse = (input1v.to(torch.float) - input2v.to(torch.float)).pow(2).sum()
                input1o = input1.offsets()
                input2o = input2.offsets()
                mse = (
                    mse
                    + (input1o.to(torch.float) - input2o.to(torch.float)).pow(2).sum()
                )
            else:
                mse = (input1.to(torch.float) - input2.to(torch.float)).pow(2).sum()
        except Exception as err:
            raise RuntimeError(
                f"Failed to compare key {prefix + (key,)}. Scroll up for more details."
            ) from err
        mse = mse.data.div(input1.numel()).sqrt().item()

        local_msg = f"key {prefix + (key,)} does not match, got mse = {mse:4.4f}"
        new_msg = ",\t".join([local_msg, msg]) if len(msg) else local_msg
        if input1.is_nested:
            torch.testing.assert_close(
                input1v.data,
                input2v.data,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
                msg=lambda torch_msg, _msg=new_msg: f"{_msg}\n{torch_msg}",
            )
        else:
            torch.testing.assert_close(
                input1.data,
                input2.data,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
                msg=lambda torch_msg, _msg=new_msg: f"{_msg}\n{torch_msg}",
            )
        local_msg = f"key {prefix + (key,)} matches"
        msg = "\t".join([local_msg, msg]) if len(msg) else local_msg

    return True


def _plain_summary(tensor: Tensor) -> str:
    """Short value summary for ``plain`` print mode."""
    if isinstance(tensor, UninitializedTensorMixin) or tensor.numel() == 0:
        return "[]"
    if tensor.is_nested:
        # The jagged layout has no std over all elements (#3259).
        tensor = torch.cat([t.reshape(-1) for t in tensor.unbind()])
    if tensor.is_floating_point() or tensor.is_complex():
        # casting complex to real drops the imaginary part and warns (#3286).
        tensor = tensor.to(torch.complex128 if tensor.is_complex() else torch.float64)
        # std of one element is nan, and torch warns when computing it (#3259).
        std = tensor.std().item() if tensor.numel() > 1 else float("nan")
        return f"mean={tensor.mean().item():.4g}, std={std:.4g}"
    return f"min={tensor.min().item()}, max={tensor.max().item()}"


def _tensor_repr_fields(tensor: Tensor) -> list[str]:
    """Build the list of ``key=value`` strings for a single tensor descriptor."""
    opts = _REPR_OPTIONS
    parts: list[str] = []
    if opts["show_shape"]:
        parts.append(f"shape={_shape(tensor)}")
    if opts["show_field_device"]:
        parts.append(f"device={_device(tensor)}")
    if opts["show_dtype"]:
        parts.append(f"dtype={_dtype(tensor)}")
    if opts["show_field_is_shared"]:
        parts.append(f"is_shared={_is_shared(tensor)}")
    if opts["show_grad"]:
        parts.append(f"requires_grad={tensor.requires_grad}")
    if opts["show_is_contiguous"]:
        parts.append(f"is_contiguous={tensor.is_contiguous()}")
    if opts["show_is_view"]:
        parts.append(f"is_view={tensor._base is not None}")
    if opts["show_storage_size"]:
        parts.append(f"storage_size={tensor.untyped_storage().nbytes()}")
    if opts["plain"]:
        parts.append(_plain_summary(tensor))
    return parts


def _get_repr(tensor: Tensor) -> str:
    s = ", ".join(_tensor_repr_fields(tensor))
    return f"{type(tensor).__name__}({s})"


def _tensor_repr_fields_custom(shape, device, dtype, is_shared, **flags) -> list[str]:
    """Build the list of ``key=value`` strings for a tensor descriptor with explicit values.

    The keyword arguments ``requires_grad``, ``is_contiguous``, ``is_view`` and
    ``storage_size`` are shown when given and their print option is set.
    """
    opts = _REPR_OPTIONS
    parts: list[str] = []
    if opts["show_shape"]:
        parts.append(f"shape={shape}")
    if opts["show_field_device"]:
        parts.append(f"device={device}")
    if opts["show_dtype"]:
        parts.append(f"dtype={dtype}")
    if opts["show_field_is_shared"]:
        parts.append(f"is_shared={is_shared}")
    for name, option in (
        ("requires_grad", "show_grad"),
        ("is_contiguous", "show_is_contiguous"),
        ("is_view", "show_is_view"),
        ("storage_size", "show_storage_size"),
    ):
        if name in flags and opts[option]:
            parts.append(f"{name}={flags[name]}")
    return parts


def _get_repr_custom(cls, shape, device, dtype, is_shared, **flags) -> str:
    s = ", ".join(_tensor_repr_fields_custom(shape, device, dtype, is_shared, **flags))
    return f"{cls.__name__}({s})"


def _make_repr(key: NestedKey, item, tensordict: T, sep) -> str:
    from tensordict.base import _is_tensor_collection

    if _is_tensor_collection(type(item)):
        return sep.join([key, repr(tensordict.get(key))])
    return sep.join([key, _get_repr(item)])


# The dtype, device and requires_grad of a tensor that a file holds, which
# reading it would give.
_FileLeaf = collections.namedtuple("_FileLeaf", ["dtype", "device", "requires_grad"])


def _stacked_repr(shape, tensors) -> str | None:
    """The repr of ``torch.stack(tensors)``, built without stacking them.

    ``tensors`` may hold :class:`_FileLeaf` descriptions, and a single one
    gives the repr of the tensor that reading the file returns.

    Returns ``None`` if the stack would fail, under ``plain=True`` (which
    prints statistics of the values), or under ``show_is_contiguous=True``
    when a member is not contiguous.
    """
    if -1 in shape or _REPR_OPTIONS["plain"]:
        return None
    # Stacking contiguous tensors gives a contiguous tensor, but stacking
    # channels_last ones keeps their memory format.
    if _REPR_OPTIONS["show_is_contiguous"] and not all(
        isinstance(t, _FileLeaf) or t.is_contiguous() for t in tensors
    ):
        return None
    device = tensors[0].device
    if any(tensor.device != device for tensor in tensors):
        return None
    try:
        dtype = functools.reduce(torch.promote_types, [t.dtype for t in tensors])
    except RuntimeError:
        return None
    # torch.stack returns a new tensor, which is_shared() reports as shared
    # only on CUDA.
    return _get_repr_custom(
        Tensor,
        shape,
        device,
        dtype,
        device.type == "cuda",
        requires_grad=torch.is_grad_enabled() and any(t.requires_grad for t in tensors),
        is_contiguous=True,
        is_view=False,
        storage_size=shape.numel() * dtype.itemsize,
    )


def _file_leaf(td, key):
    """The shape of tensor entry ``key`` of the h5 or zarr file ``td``, and its :class:`_FileLeaf`, read from the metadata.

    Returns ``None`` if the entry is missing or is not a tensor.
    """
    try:
        meta = td._get_metadata(key)
    except KeyError:
        return None
    if not meta.get("array"):
        return None
    device = td.device if td.device is not None else torch.device("cpu")
    return meta["shape"], [_FileLeaf(meta["dtype"], device, False)]


def _stacked_leaf(td, key):
    """``LazyStackedTensorDict._plain_leaf`` for printing, which reads no h5 or zarr member.

    The members that are h5 or zarr files give a :class:`_FileLeaf`.
    """
    from tensordict._lazy import _is_plain_tensor, LazyStackedTensorDict
    from tensordict.persistent import PersistentTensorDict

    if (
        not isinstance(key, str)
        or not td.tensordicts
        or td._hook_out is not None
        or td._is_vmapped
    ):
        return None
    shapes, tensors = [], []
    for member in td.tensordicts:
        if isinstance(member, LazyStackedTensorDict):
            leaf = _stacked_leaf(member, key)
        elif isinstance(member, PersistentTensorDict):
            leaf = _file_leaf(member, key)
        else:
            tensor = member._get_str(key, None)
            leaf = (tensor.shape, [tensor]) if _is_plain_tensor(tensor) else None
        if leaf is None:
            return None
        shapes.append(leaf[0])
        tensors.extend(leaf[1])
    shape = list(shapes[0])
    for other in shapes[1:]:
        if len(shape) != len(other):
            return torch.Size([-1]), tensors
        shape = [s1 if s1 == s2 else -1 for (s1, s2) in zip(shape, other)]
    shape.insert(td.stack_dim, len(td.tensordicts))
    return torch.Size(shape), tensors


def _unbatched_leaf(td, key):
    """The ``UnbatchedTensor`` of entry ``key`` of the first member of lazy stack ``td``.

    Stacking keeps only that one. Returns ``None`` unless every member holds
    an ``UnbatchedTensor`` there.
    """
    from tensordict._lazy import LazyStackedTensorDict
    from tensordict.persistent import PersistentTensorDict

    if (
        not isinstance(key, str)
        or not td.tensordicts
        or td._hook_out is not None
        or td._is_vmapped
    ):
        return None
    leaves = []
    for member in td.tensordicts:
        if isinstance(member, PersistentTensorDict):
            # a file holds no UnbatchedTensor, and _get_str would read it
            return None
        if isinstance(member, LazyStackedTensorDict):
            leaf = _unbatched_leaf(member, key)
        else:
            leaf = member._get_str(key, None)
        if not _is_unbatched(leaf):
            return None
        leaves.append(leaf)
    return leaves[0]


def _mixed_repr(td, key) -> str | None:
    """The types of the members' values of entry ``key`` of lazy stack ``td``, as ``[NonTensorData, Tensor]``.

    Returns ``None`` unless the members hold different kinds of values there
    (tensors, non-tensor data, tensordicts), which cannot be stacked (#3258).
    """
    from tensordict._lazy import LazyStackedTensorDict
    from tensordict.base import _is_tensor_collection

    def values(lazy):
        for member in lazy.tensordicts:
            if isinstance(member, LazyStackedTensorDict):
                yield from values(member)
            else:
                yield member._get_str(key, None)

    if not isinstance(key, str):
        return None
    items = list(values(td))
    if any(item is None for item in items):
        return None
    kinds = {
        "non-tensor"
        if is_non_tensor(item)
        else "collection"
        if _is_tensor_collection(type(item))
        else "tensor"
        for item in items
    }
    if len(kinds) < 2:
        return None
    names = dict.fromkeys(type(item).__name__ for item in items)
    return f"[{', '.join(names)}]"


def _td_fields(td: T, keys=None, sep=": ") -> str:
    from tensordict._lazy import LazyStackedTensorDict
    from tensordict.persistent import PersistentTensorDict

    # A TypedTensorDict reads its entries from the tensordict it wraps.
    source = td
    while type(source)._wrapped_attr is not None:
        source = getattr(source, type(source)._wrapped_attr)
    is_lazy = isinstance(source, LazyStackedTensorDict)
    is_file = isinstance(source, PersistentTensorDict)
    strs = []
    if keys is None:
        keys = td.keys()
    for key in keys:
        # The tensor entries of a lazy stack, and of an h5 or zarr file, are
        # printed from the members and from the metadata (#3257).
        if is_lazy:
            leaf = _stacked_leaf(source, key)
            unbatched = _unbatched_leaf(source, key) if leaf is None else None
            if unbatched is not None:
                # stacking returns an alias of the first member's tensor
                alias = torch.ops.aten.alias.default(unbatched)
                strs.append(sep.join([key, _get_repr(alias)]))
                continue
        elif is_file:
            leaf = _file_leaf(source, key)
        else:
            leaf = None
        if leaf is not None:
            shape = leaf[0]
            substr = _stacked_repr(*leaf)
            if substr is not None:
                strs.append(sep.join([key, substr]))
                continue
        else:
            try:
                shape = td.get_item_shape(key)
            except Exception:
                # The members of a lazy stack hold values that cannot be
                # stacked, such as a string and a tensor (#3258).
                substr = _mixed_repr(source, key) if is_lazy else None
                if substr is None:
                    raise
                strs.append(sep.join([key, substr]))
                continue
        if -1 not in shape:
            item = td.get(key)
            strs.append(_make_repr(key, item, td, sep=sep))
        else:
            # we know td is lazy stacked and the key is a leaf
            # so we can get the shape and escape the error
            temp_td = source
            from tensordict import (
                is_tensor_collection,
                LazyStackedTensorDict,
                TensorDictBase,
            )

            first = leaf[1][0] if leaf is not None else None
            if isinstance(first, _FileLeaf):
                # the first member is a file: print it from the metadata
                substr = _get_repr_custom(
                    Tensor,
                    shape,
                    first.device,
                    first.dtype,
                    first.device.type == "cuda",
                )
                strs.append(sep.join([key, substr]))
                continue
            while isinstance(
                temp_td, LazyStackedTensorDict
            ):  # we need to grab the heterogeneous tensor from the inner nesting level
                temp_td = temp_td.tensordicts[0]
            tensor = temp_td.get(key)
            if is_tensor_collection(tensor):
                tensor = td.get(key)
                strs.append(_make_repr(key, tensor, td, sep=sep))
                continue

            if isinstance(tensor, TensorDictBase):
                substr = _td_fields(tensor)
            else:
                is_shared = (
                    tensor.is_shared()
                    if not isinstance(tensor, UninitializedTensorMixin)
                    else None
                )
                substr = _get_repr_custom(
                    type(tensor),
                    shape=shape,
                    device=tensor.device,
                    dtype=tensor.dtype,
                    is_shared=is_shared,
                )
            strs.append(sep.join([key, substr]))

    sort_keys = _REPR_OPTIONS["sort_keys"]
    if sort_keys == "alphabetical":
        strs = sorted(strs)
    elif callable(sort_keys):
        strs = sorted(strs, key=sort_keys)

    return indent(
        "\n" + ",\n".join(strs),
        4 * " ",
    )


# set_printoptions(compact=True): the repr is built from metadata only (shapes,
# dtypes and devices of the leaves), so that it reads no data, syncs no device
# and stacks no leaf of a lazy stack. A container is a (prefix, entries) pair,
# an entry a (key, token or container) pair.
_COMPACT_WIDTH = 80
_CPU = torch.device("cpu")


@functools.cache
def _compact_dtype(dtype: torch.dtype) -> str:
    # f32, i64, u8, c64, bf16, bool, f8_e4m3fn
    name = str(dtype).replace("torch.", "")
    for long, short in (("float", "f"), ("uint", "u"), ("int", "i"), ("complex", "c")):
        name = name.replace(long, short)
    return name


def _compact_leaf(td, key):
    # Returns the shape after the batch dims, the dtype and the set of devices
    # of a tensor entry. In a lazy stack, a dim that differs between members
    # is -1 and the dtype is the promoted one, as the stacked tensor has. If a
    # member holds a non-tensor value, returns the entry of that member, as
    # when it comes first.
    from tensordict import LazyStackedTensorDict, TensorDict
    from tensordict.persistent import PersistentTensorDict
    from tensordict.typedtensordict import TypedTensorDict

    if isinstance(td, TypedTensorDict):
        td = td._source
    if isinstance(td, LazyStackedTensorDict):
        # The members of nested lazy stacks are walked in one loop, and the
        # leaves of plain TensorDict members read directly: a stack can have
        # thousands of members.
        shapes, dtypes, devices = set(), set(), set()
        members = list(td.tensordicts)
        while members:
            sub = members.pop()
            if type(sub) is TensorDict:
                tensor = sub._get_str(key, None)
                if is_non_tensor(tensor):
                    return _compact_entry(sub, key)
                shapes.add(_compact_shape(sub, tensor))
                dtypes.add(tensor.dtype)
                devices.add(tensor.device)
            elif isinstance(sub, LazyStackedTensorDict):
                members.extend(sub.tensordicts)
            else:
                leaf = _compact_leaf(sub, key)
                if isinstance(leaf, str):
                    return leaf
                shape, dtype, sub_devices = leaf
                shapes.add(tuple(shape))
                dtypes.add(dtype)
                devices |= sub_devices
        shapes = iter(shapes)
        shape = list(next(shapes))
        for other in shapes:
            if len(other) != len(shape):
                shape = [-1]
                break
            shape = [s if s == o else -1 for s, o in zip(shape, other)]
        return shape, functools.reduce(torch.promote_types, dtypes), devices
    if isinstance(td, PersistentTensorDict):
        meta = td._get_metadata(key)
        device = td.device if td.device is not None else torch.device("cpu")
        return list(meta["shape"][td.batch_dims :]), meta["dtype"], {device}
    tensor = td._get_str(key, None)
    if is_non_tensor(tensor):
        return _compact_entry(td, key)
    return _compact_shape(td, tensor), tensor.dtype, {tensor.device}


def _compact_shape(td, tensor) -> tuple:
    # an unbatched tensor does not share the batch dims: its full shape
    shape = _shape(tensor)
    return tuple(shape if _is_unbatched(tensor) else shape[td.batch_dims :])


def _compact_token(td, shape, dtype, devices) -> str:
    token = f"[{','.join(map(str, shape))}]"
    if _REPR_OPTIONS["show_dtype"]:
        token = _compact_dtype(dtype) + token
    if devices != {_CPU} and td.device is None:
        token += "@" + ",".join(sorted(str(device) for device in devices))
    return token


def _compact_entry(td, key):
    from tensordict import LazyStackedTensorDict, NonTensorData, TensorDict
    from tensordict.base import _is_tensor_collection
    from tensordict.persistent import PersistentTensorDict
    from tensordict.typedtensordict import TypedTensorDict

    if type(td) is TensorDict:
        item = td._get_str(key, None)
        if isinstance(item, Tensor) and not _is_tensor_collection(type(item)):
            shape = _compact_shape(td, item)
            return _compact_token(td, shape, item.dtype, {item.device})
    # the entry of the first member decides what the entry of a lazy stack is
    first = td
    while isinstance(first, TypedTensorDict) or (
        isinstance(first, LazyStackedTensorDict) and first.tensordicts
    ):
        if isinstance(first, TypedTensorDict):
            first = first._source
        else:
            first = first.tensordicts[0]
    if isinstance(first, PersistentTensorDict):
        meta = first._get_metadata(key)
        if meta.get("non_tensor"):
            return "NonTensorStack" if "shape" in meta else "NonTensorData"
        item = None if meta.get("array") else first._get_str(key, None)
    else:
        item = first._get_str(key, None)
    if item is None or not _is_tensor_collection(type(item)):
        leaf = _compact_leaf(td, key)
        return leaf if isinstance(leaf, str) else _compact_token(td, *leaf)
    if isinstance(item, NonTensorData):
        return type(item.data).__name__
    if is_non_tensor(item) or (not is_tensorclass(item) and _keeps_own_repr(item)):
        # a nested store shows its class name only: its leaves are on a server
        return type(item).__name__
    sub = item if first is td else td._get_str(key, None)
    prefix = "" if type(item) is TensorDict else type(item).__name__
    extra = sub.batch_size[td.batch_dims :]
    if extra:
        prefix += f"[{','.join(map(str, extra))}]"
    return prefix, _compact_body(sub)


def _compact_entries(td, keys) -> list:
    entries = [(key, _compact_entry(td, key)) for key in keys]
    sort_keys = _REPR_OPTIONS["sort_keys"]
    if sort_keys == "alphabetical":
        entries.sort(key=lambda entry: f"{entry[0]}: {_compact_flat(entry[1])}")
    elif callable(sort_keys):
        entries.sort(
            key=lambda entry: sort_keys(f"{entry[0]}: {_compact_flat(entry[1])}")
        )
    return entries


def _compact_body(td) -> list:
    if not is_tensorclass(td):
        return _compact_entries(td, td.keys())
    from tensordict.tensorclass import _none_fields, _repr_hidden_fields

    # the fields declared with field(repr=False) are left out
    hidden = _repr_hidden_fields(type(td))
    keys = td._tensordict.keys()
    if hidden:
        keys = [key for key in keys if key not in hidden]
    entries = _compact_entries(td._tensordict, keys)
    # the fields that hold no tensor: by type name, as non-tensor entries
    others = [
        (key, "None" if value is None else type(value).__name__)
        for key, value in td._non_tensordict.items()
        if key not in hidden
    ]
    if not type(td)._is_non_tensor:
        others += [(key, "None") for key in _none_fields(td) if key not in hidden]
    return entries + others


def _compact_flat(node) -> str:
    if isinstance(node, str):
        return node
    prefix, entries = node
    return f"{prefix}{{{', '.join(f'{k}: {_compact_flat(v)}' for k, v in entries)}}}"


def _compact_render(node, indent: str, used: int, tail: int) -> str:
    # One line if it fits in _COMPACT_WIDTH columns, given the ``used``
    # columns before it and the ``tail`` ones after it; otherwise one entry
    # per line, each rendered by the same rule.
    flat = _compact_flat(node)
    if (
        isinstance(node, str)
        or not node[1]
        or used + len(flat) + tail <= _COMPACT_WIDTH
    ):
        return flat
    prefix, entries = node
    pad = indent + 4 * " "
    lines = []
    for i, (key, value) in enumerate(entries):
        value_tail = tail + 1 if i == len(entries) - 1 else 1
        value = _compact_render(value, pad, len(pad) + len(key) + 2, value_tail)
        lines.append(f"{pad}{key}: {value}")
    return prefix + "{\n" + ",\n".join(lines) + "}"


def _keeps_own_repr(td) -> bool:
    # The stores list their key names and a non-tensor stack its values: a
    # class with its own _repr_parts keeps its repr.
    from tensordict import LazyStackedTensorDict, TensorDictBase

    return type(td)._repr_parts not in (
        TensorDictBase._repr_parts,
        LazyStackedTensorDict._repr_parts,
    )


def _compact_parts(obj):
    # Returns the header items, the entries and the exclusive fields of the
    # compact repr, or None for a class that keeps its own repr.
    from tensordict import LazyStackedTensorDict

    header = []
    if is_tensorclass(obj):
        expected = type(obj).__expected_keys__
        if _REPR_OPTIONS["show_batch_size"] and "batch_size" not in expected:
            header.append(f"batch_size=[{','.join(map(str, obj.batch_size))}]")
        if _REPR_OPTIONS["show_device"] and "device" not in expected:
            if obj.device is not None:
                header.append(f"device={obj.device}")
    else:
        if _keeps_own_repr(obj):
            return None
        if _REPR_OPTIONS["show_batch_size"]:
            header.append(f"batch_size=[{','.join(map(str, obj.batch_size))}]")
        if _REPR_OPTIONS["show_device"] and obj.device is not None:
            header.append(f"device={obj.device}")
    exclusive = []
    if isinstance(obj, LazyStackedTensorDict):
        header.append(f"stack_dim={obj.stack_dim}")
        keys = set(obj.keys())
        for i, member in enumerate(obj.tensordicts):
            own = [key for key in member.keys() if key not in keys]
            if own:
                exclusive.append((str(i), ("", _compact_entries(member, own))))
    return header, _compact_body(obj), exclusive


def _compact_repr(obj) -> str | None:
    """Returns the repr of ``set_printoptions(compact=True)``, or ``None`` for a class that keeps its own."""
    parts = _compact_parts(obj)
    if parts is None:
        return None
    header, body, exclusive = parts
    head = f"{type(obj).__name__}(" + "".join(f"{h}, " for h in header) + "fields="
    # on the last line of the fields come the exclusive fields, if any
    tail = 1 + len(", exclusive_fields={") if exclusive else 1
    out = head + _compact_render(("", body), "", len(head), tail)
    if exclusive:
        used = len(out.rsplit("\n", 1)[-1]) + len(", exclusive_fields=")
        out += ", exclusive_fields=" + _compact_render(("", exclusive), "", used, 1)
    return out + ")"


# The HTML repr of notebooks (_repr_html_): the entries of the compact repr in
# nested <details> blocks, with inherited colors so that it reads in light and
# dark themes. It shows at most _HTML_MAX_ENTRIES entries.
_HTML_MAX_ENTRIES = 200
_HTML_LIST = "list-style:none;margin:0;padding-left:1.5em"
_HTML_DIM = "opacity:0.7"


def _html_entries(entries, budget: list) -> str:
    items = []
    for i, (key, value) in enumerate(entries):
        if budget[0] <= 0:
            items.append(f"<li>... and {len(entries) - i} more</li>")
            break
        budget[0] -= 1
        key = html.escape(str(key))
        if isinstance(value, str):
            value = html.escape(value)
            items.append(f'<li>{key}: <span style="{_HTML_DIM}">{value}</span></li>')
            continue
        prefix, sub = value
        label = html.escape(f"{prefix} " if prefix else "")
        label += f"{len(sub)} {'entry' if len(sub) == 1 else 'entries'}"
        items.append(
            f'<li><details open><summary>{key}: <span style="{_HTML_DIM}">{label}'
            f"</span></summary>{_html_entries(sub, budget)}</details></li>"
        )
    return f'<ul style="{_HTML_LIST}">' + "".join(items) + "</ul>"


def _html_repr(obj) -> str | None:
    """Returns the ``_repr_html_`` of a tensordict or a tensorclass.

    It reads what the compact repr reads, and falls back to the text repr in a
    ``<pre>`` block for the classes that keep their own repr, or if it fails.
    """
    from tensordict import TensorDictBase
    from tensordict.tensorclass import _repr

    if not _REPR_OPTIONS["html"]:
        return None
    if not _REPR_OPTIONS["compact"]:
        # with compact=False (or a detailed-only option), the notebook shows
        # the detailed text repr, with the options that it applies
        return f"<pre>{html.escape(repr(obj))}</pre>"
    try:
        # a class with its own repr (a non-tensor class among them) keeps it
        if is_tensorclass(obj):
            parts = _compact_parts(obj) if type(obj).__repr__ is _repr else None
        elif type(obj).__repr__ is TensorDictBase.__repr__:
            parts = _compact_parts(obj)
        else:
            parts = None
        if parts is not None:
            header, body, exclusive = parts
            names = obj._maybe_names() if _REPR_OPTIONS["show_batch_size"] else None
            if names is not None and any(name is not None for name in names):
                at = 1 if header and header[0].startswith("batch_size=") else 0
                header.insert(at, f"names=[{','.join(map(str, names))}]")
            budget = [_HTML_MAX_ENTRIES]
            out = (
                '<div style="font-family:monospace;line-height:1.4">'
                f"<details open><summary><b>{html.escape(type(obj).__name__)}</b> "
                f'<span style="{_HTML_DIM}">{html.escape(", ".join(header))}</span>'
                f"</summary>{_html_entries(body, budget)}"
            )
            if exclusive:
                out += (
                    '<details open style="padding-left:1.5em"><summary>'
                    f'<i style="{_HTML_DIM}">exclusive_fields</i></summary>'
                    f"{_html_entries(exclusive, budget)}</details>"
                )
            return out + "</details></div>"
    except Exception:
        pass
    return f"<pre>{html.escape(repr(obj))}</pre>"


def _check_keys(
    list_of_tensordicts: Sequence[TensorDictBase],
    strict: bool = False,
    include_nested: bool = False,
    leaves_only: bool = False,
) -> set[str] | list[str]:
    from tensordict.base import _is_leaf_nontensor

    if not len(list_of_tensordicts):
        return set()
    # is_leaf only filters nested or leaf keys: without it, TensorDict.keys()
    # returns its cheap view of the dict keys.
    is_leaf = _is_leaf_nontensor if include_nested or leaves_only else None
    keys = list_of_tensordicts[0].keys(
        include_nested=include_nested,
        leaves_only=leaves_only,
        is_leaf=is_leaf,
    )
    # TODO: compile doesn't like set() over an arbitrary object
    is_comp = is_compiling()
    if is_comp:
        keys_set = {k for k in keys}  # noqa: C416
    else:
        keys_set: set[str] = set(keys)
    for i, td in enumerate(list_of_tensordicts[1:], 1):
        k = td.keys(
            include_nested=include_nested,
            leaves_only=leaves_only,
            is_leaf=is_leaf,
        )
        if not strict:
            keys_set = keys_set.intersection(k)
        else:
            if is_comp:
                k = {v for v in k}  # noqa: C416
            else:
                k = set(k)
            if k != keys_set:
                raise KeyError(
                    f"tensordict {i} has keys {sorted(k, key=str)} but tensordict 0 "
                    f"has keys {sorted(keys_set, key=str)}; all tensordicts must have "
                    f"the same keys"
                )
    if strict:
        if is_comp:
            return [key for key in keys]  # noqa: C416
        else:
            return list(keys)
    return keys_set


def _set_max_batch_size(
    source: T, batch_dims: int | None = None, keep_compliant_size: bool = False
):
    """Updates a tensordict with its maximum batch size."""
    from tensordict.base import _is_tensor_collection

    tensor_data = [
        val
        for val in source.values()
        if not _is_unbatched(val) and not (_pass_through(val) and not val.batch_size)
    ]

    for val in tensor_data:
        if _is_tensor_collection(type(val)):
            # A lazy tensordict (such as a lazy stack or a NonTensorStack) gets
            # its batch size from its members and cannot be resized.
            if val._lazy:
                continue
            if (
                batch_dims is not None
                and keep_compliant_size
                and val.batch_dims >= batch_dims
            ):
                continue
            _set_max_batch_size(val, batch_dims=batch_dims)

    batch_size = []
    if not tensor_data:  # when source is empty
        if batch_dims:
            source.batch_size = source.batch_size[:batch_dims]
            return source
        else:
            return source

    curr_dim = 0
    tensor_shapes = [_shape(_tensor_data) for _tensor_data in tensor_data]

    while True:
        if len(tensor_shapes[0]) > curr_dim:
            curr_dim_size = tensor_shapes[0][curr_dim]
        else:
            source.batch_size = batch_size
            return
        # A ragged dim (-1) cannot be part of a dense batch size
        if curr_dim_size == -1:
            source.batch_size = batch_size
            return
        for leaf, shape in _zip_strict(tensor_data[1:], tensor_shapes[1:]):
            # if we have a nested empty tensordict we can modify its batch size at will
            if _is_tensor_collection(type(leaf)) and leaf.is_empty():
                continue
            if (len(shape) <= curr_dim) or (shape[curr_dim] != curr_dim_size):
                source.batch_size = batch_size
                return
        if batch_dims is None or len(batch_size) < batch_dims:
            batch_size.append(curr_dim_size)
        curr_dim += 1


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


def _expands_to(batch_size: torch.Size, shape: Sequence[int]) -> bool:
    """Whether :meth:`torch.Tensor.expand` accepts ``shape`` for a tensor of size ``batch_size``.

    ``shape`` needs at least as many sizes as ``batch_size`` has dims, and its trailing
    sizes must match ``batch_size`` except where ``batch_size`` is 1.
    """
    ndim = len(batch_size)
    return len(shape) >= ndim and all(
        old == 1 or new == old
        for old, new in zip(batch_size, shape[len(shape) - ndim :])
    )


def _resolve_expand_shape(shape: Sequence[int], batch_size: torch.Size) -> torch.Size:
    """Replaces the ``-1`` sizes of an ``expand`` shape and checks the shape, as :meth:`torch.Tensor.expand` does.

    The shapes are aligned on the right. A ``-1`` keeps the size of the matching dim of
    ``batch_size``. The other sizes must match the existing size, or expand a dim of size
    1. The sizes of new leading dims are left as they are.
    """
    num_new_dims = len(shape) - len(batch_size)
    if num_new_dims < 0:
        # too few sizes: left to the caller, which raises
        return shape
    resolved = list(shape)
    for i, size in enumerate(shape):
        if i < num_new_dims:
            continue
        existing = batch_size[i - num_new_dims]
        if size == -1:
            resolved[i] = existing
        elif size != existing and existing != 1:
            raise RuntimeError(
                f"The expanded size of the tensor ({size}) must match the existing size "
                f"({existing}) at non-singleton dimension {i}.  Target sizes: "
                f"{list(shape)}.  Tensor sizes: {list(batch_size)}"
            )
    return torch.Size(resolved)


class _BufferLegacy(torch.nn.Buffer):
    """A buffer subclass that keeps the grad fn history."""

    def __new__(cls, data=None, *, persistent=True):
        if data is None:
            data = torch.empty(0)

        t = data
        t.persistent = persistent
        t._is_buffer = True
        return t


from tensordict._utils_options import (  # noqa: F401
    _REPR_OPTIONS,
    _strtobool,
    capture_non_tensor_stack,
    get_printoptions,
    lazy_legacy,
    list_to_stack,
    set_capture_non_tensor_stack,
    set_lazy_legacy,
    set_list_to_stack,
    set_printoptions,
)


def _convert_list_to_stack(
    a_list: list[Any],
    *,
    wrap_ragged: bool = True,
) -> tuple[torch.Tensor | TensorDictBase | NonTensorStack | list[Any], bool]:  # noqa
    if not a_list:
        return torch.tensor(a_list), False
    # First, check elements and determine if there are lists within
    nontensor = True
    if all(isinstance(elt, list) for elt in a_list):
        a_list, nontensor = zip(
            *[_convert_list_to_stack(elt, wrap_ragged=wrap_ragged) for elt in a_list]
        )
        nontensor = any(nontensor)
    # FIXME: we should check that the type is unique
    all_castable = all(isinstance(elt, (bool, int, float)) for elt in a_list)
    if all_castable:
        return torch.tensor(a_list), False
    all_tensors = all(isinstance(elt, torch.Tensor) for elt in a_list)
    if not nontensor or all_tensors:
        # should we stack?
        if all_tensors and len({x.shape for x in a_list}) == 1:
            # FIXME: this may lead to some weird behaviours if we have nested lists and by chance one of them has
            #  things that can be stacked, and others don't.
            return torch.stack(a_list), False
        if all_tensors and wrap_ragged:
            # No common shape to stack. A list has no .shape for the caller to read.
            from tensordict import NonTensorStack

            return NonTensorStack(*a_list), True
        # TODO: check that LazyStack understands that a list is a bunch of elements to write in separate tds
        return list(a_list), False
    from tensordict.base import _is_tensor_collection

    if all(_is_tensor_collection(type(elt)) for elt in a_list):
        return torch.stack(a_list), False
    from tensordict import NonTensorStack

    return NonTensorStack(*a_list), True


def _recursive_unbind_list(a_list, dim):
    if dim == 0:
        return list(a_list)
    try:
        return map(
            list, _zip_strict(*[_recursive_unbind_list(elt, dim - 1) for elt in a_list])
        )
    except Exception:
        raise ValueError("lengths of nested lists differed.")


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
    sub_keys = tuple(
        key[1:] for key in keys_to_update if isinstance(key, tuple) and key[0] == prefix
    )
    if not sub_keys and prefix in keys_to_update:
        # the key names the sub-tensordict: all its entries, as select gives
        return None
    return sub_keys


def _selection_trie(keys_to_update):
    """The unravelled ``keys_to_update`` as a trie, for :func:`_is_key_selected` (#3216).

    Each node maps a key part to its child node; the ``None`` entry marks a
    listed key. ``update`` builds it once per call.
    """
    trie = {}
    for key in keys_to_update:
        node = trie
        for part in (key,) if isinstance(key, str) else key:
            node = node.setdefault(part, {})
        node[None] = True
    return trie


def _is_key_selected(key, trie):
    """Whether ``update`` writes the entry at ``key`` (a tuple), given ``_selection_trie(keys_to_update)``.

    Follows the rule of :func:`_prune_selected_keys`: a listed key selects every
    entry under it, unless deeper keys are listed under it, which narrow it. So
    a flat dict source with tuple keys gets the entries that a tensordict
    source gets (#3216).
    """
    node = trie
    for part in key[:-1]:
        node = node.get(part)
        if node is None:
            return False
        if len(node) == 1 and None in node:
            # a listed key with nothing listed under it: all its entries
            return True
    return key[-1] in node


class TensorDictFuture:
    """A tensordict that is still being written by background threads.

    :meth:`~tensordict.TensorDictBase.memmap`,
    :meth:`~tensordict.TensorDictBase.memmap_`,
    :meth:`~tensordict.TensorDictBase.memmap_like`,
    :meth:`~tensordict.TensorDictBase.save` and
    :meth:`~tensordict.TensorDictBase.consolidate` return one when called
    with ``return_early=True`` and ``num_threads > 1``. Call :meth:`result`
    to wait for the writes and get the tensordict. You do not build one
    yourself.

    Args:
        futures (list of concurrent.futures.Future): the writes to wait for.
        resulting_td (TensorDictBase): the tensordict that :meth:`result`
            returns.

    Examples:
        >>> import tempfile
        >>> import torch
        >>> from tensordict import TensorDict
        >>> td = TensorDict(a=torch.zeros(3), b=torch.ones(3), batch_size=[3])
        >>> with tempfile.TemporaryDirectory() as path:
        ...     future = td.memmap(path, num_threads=2, return_early=True)
        ...     result = future.result()
        ...     print(result.is_memmap(), result["b"].tolist())
        True [1.0, 1.0, 1.0]

    """

    def __init__(self, futures, resulting_td):
        self.futures = futures
        self.resulting_td = resulting_td

    def result(self):
        """Waits for the writes to finish and returns the tensordict.

        If a write failed, raises the first error of the writes.
        """
        concurrent.futures.wait(self.futures)
        for future in self.futures:
            future.result()
        return self.resulting_td


_SHARED_EXECUTORS: dict[int, concurrent.futures.ThreadPoolExecutor] = {}
_SHARED_EXECUTORS_LOCK = threading.Lock()


def _get_shared_executor(num_threads: int) -> concurrent.futures.ThreadPoolExecutor:
    """Returns a process-wide executor with ``num_threads`` workers.

    Repeated serialization calls (e.g. checkpoint loops) reuse the pool
    instead of spawning and tearing down threads on every call. The tasks
    submitted to it must not themselves submit to (and wait on) the shared
    pool, or they may deadlock.
    """
    with _SHARED_EXECUTORS_LOCK:
        executor = _SHARED_EXECUTORS.get(num_threads)
        if executor is None:
            executor = _SHARED_EXECUTORS[num_threads] = (
                concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
            )
        return executor


def _reset_shared_executors_after_fork() -> None:
    # the workers don't survive a fork: drop the executors in the child so that
    # they are recreated on first use. The lock is replaced too, as the child
    # inherits it locked if another thread held it at the fork.
    global _SHARED_EXECUTORS_LOCK
    _SHARED_EXECUTORS_LOCK = threading.Lock()
    _SHARED_EXECUTORS.clear()


if hasattr(os, "register_at_fork"):  # not available on Windows, which cannot fork
    os.register_at_fork(after_in_child=_reset_shared_executors_after_fork)


def _is_json_serializable(item):
    if isinstance(item, dict):
        for key, val in item.items():
            # Per se, int, float and bool are serializable but not recoverable
            # as such
            if not isinstance(key, (str,)) or not _is_json_serializable(val):
                return False
        else:
            return True
    # a set is not a JSON type: leave it to pickle
    if isinstance(item, (list, tuple)):
        for val in item:
            if not _is_json_serializable(val):
                return False
        else:
            return True
    return isinstance(item, (str, int, float, bool)) or item is None


def print_directory_tree(path, indent="", display_metadata=True) -> str:
    """Logs the tree of files and directories under ``path``.

    Each line is logged at the ``INFO`` level with the tensordict logger
    (:data:`tensordict.utils.logger`) and the tree is returned as a string.
    The entries of a directory are in the order of :func:`os.listdir`.

    Args:
        path (str): the directory (or file) to show.
        indent (str, optional): the string put before each line of the tree;
            each level adds four spaces to it. Defaults to ``""``.
        display_metadata (bool, optional): if ``True``, the first line gives
            the total size of the files under ``path``. Defaults to ``True``.

    Returns:
        The lines that were logged, joined by newlines.

    Examples:
        >>> import tempfile
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.utils import print_directory_tree
        >>> td = TensorDict(a=torch.zeros(3), b=torch.ones(3), batch_size=[3])
        >>> with tempfile.TemporaryDirectory() as path:
        ...     _ = td.memmap(path)
        ...     tree = print_directory_tree(path, display_metadata=False)
        >>> sorted(line.strip() for line in tree.splitlines()[1:])
        ['a.memmap', 'b.memmap', 'meta.json']

    """
    string = []
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
        string.append(f"Directory size: {formatted_size}")
        logger.info(string[-1])

    if os.path.isdir(path):
        string.append(indent + os.path.basename(path) + "/")
        logger.info(string[-1])
        indent += "    "
        for item in os.listdir(path):
            string.append(
                print_directory_tree(
                    os.path.join(path, item), indent=indent, display_metadata=False
                )
            )
    else:
        string.append(indent + os.path.basename(path))
        logger.info(string[-1])
    return "\n".join(string)


def isin(
    input: TensorDictBase,
    reference: TensorDictBase,
    key: NestedKey,
    dim: int = 0,
) -> Tensor:
    """Returns which elements of ``input`` along ``dim`` are also in ``reference``.

    The entry ``key`` is cut into slices along ``dim``. An element of
    ``input`` is in ``reference`` if its slice is equal to one of the slices
    of ``reference[key]``. The two tensordicts must have the same number of
    batch dimensions, and the shapes of the two entries must match except
    along ``dim``.

    Args:
        input (TensorDictBase): the tensordict to test.
        reference (TensorDictBase): the tensordict to look in.
        key (NestedKey): the key of the tensor to compare. It must hold a
            tensor in both tensordicts.
        dim (int, optional): the batch dimension along which to compare.
            Defaults to ``0``.

    Returns:
        A boolean tensor of length ``input.batch_size[dim]``.

    Raises:
        KeyError: if ``key`` is missing or is not a tensor in either
            tensordict.
        ValueError: if the numbers of batch dimensions differ, or if ``dim``
            is not a batch dimension.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.utils import isin
        >>> td = TensorDict(
        ...     x=torch.tensor([[1, 2], [3, 4], [1, 2], [5, 6]]),
        ...     y=torch.arange(4),
        ...     batch_size=[4],
        ... )
        >>> reference = TensorDict(x=torch.tensor([[1, 2], [5, 6]]), batch_size=[2])
        >>> isin(td, reference, key="x")
        tensor([ True, False,  True,  True])
    """
    # Get the data
    reference_tensor = reference.get(key)
    target_tensor = input.get(key)

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
    # slices first: under torch.compile, ``index is None`` on a slice fixes the
    # SymInt bounds it holds (gh-2899)
    if isinstance(index, slice):
        return index.start is None or index.start == 0
    # we can't use a list comprehension here because it fails with tensor indices
    if index is None or index is Ellipsis:
        return True
    # torch reads a bool as a 0-d mask, which copies
    if isinstance(index, int) and not isinstance(index, bool):
        return True
    return False


def remove_duplicates(
    input: TensorDictBase,
    key: NestedKey,
    dim: int = 0,
    *,
    return_indices: bool = False,
) -> TensorDictBase:
    """Returns a tensordict without the elements whose ``key`` value repeats an earlier one.

    The entry ``key`` is cut into slices along ``dim``. For each group of
    equal slices, the first element is kept, and the same indices are kept in
    every other entry. The result is sorted by the value of ``key``, not in
    the original order, and its entries are new tensors.

    Args:
        input (TensorDictBase): the tensordict to deduplicate.
        key (NestedKey): the key of the tensor that defines duplicates. It
            must hold a tensor, not a tensordict.
        dim (int, optional): the batch dimension along which to look for
            duplicates. Defaults to ``0``.

    Keyword Args:
        return_indices (bool, optional): if ``True``, also returns, for each
            element of ``input`` along ``dim``, the index of its value in the
            result. Defaults to ``False``.

    Returns:
        The deduplicated tensordict, or a tuple of the tensordict and the
        indices if ``return_indices=True``.

    Raises:
        KeyError: if ``key`` is missing or does not hold a tensor.
        ValueError: if ``dim`` is not a batch dimension.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.utils import remove_duplicates
        >>> td = TensorDict(x=torch.tensor([7, 1, 7, 3]), y=torch.arange(4), batch_size=[4])
        >>> out, indices = remove_duplicates(td, key="x", return_indices=True)
        >>> out["x"], out["y"]
        (tensor([1, 3, 7]), tensor([1, 3, 0]))
        >>> indices
        tensor([2, 0, 2, 1])
    """
    tensor = input.get(key)

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
                param = _add_batch_dim(param, param.in_dim, param.vmap_level)
                delattr(mod, name)
                setattr(mod, name, param)
        for key, val in list(mod._forward_pre_hooks.items()):
            if val is self:
                del mod._forward_pre_hooks[key]
                return
        else:
            raise RuntimeError("did not find pre-hook")


def is_non_tensor(data) -> bool:
    """Returns ``True`` if ``data`` is non-tensor data or a stack of non-tensor data.

    Args:
        data (object): the object to check, for example a value read from a
            tensordict with :meth:`~tensordict.TensorDictBase.get`.

    Examples:
        >>> import torch
        >>> from tensordict import NonTensorData, TensorDict, is_non_tensor
        >>> td = TensorDict(x=torch.zeros(2), name=NonTensorData("a"))
        >>> is_non_tensor(td.get("name")), is_non_tensor(td.get("x"))
        (True, False)
        >>> td["name"]
        'a'
    """
    return _is_non_tensor(type(data))


def _pass_through(data) -> bool:
    return _pass_through_cls(type(data))


_UNBATCHED_MEMO = {}


def _is_unbatched(data) -> bool:
    """Returns True only for values with ``_pass_through = True`` (e.g., UnbatchedTensor).

    Unlike ``_pass_through`` which also matches ``_is_non_tensor`` (NonTensorData, MetaData),
    this only matches types that explicitly set ``_pass_through = True``.
    """
    cls = type(data)
    if cls is Tensor:
        return False
    if is_compiling():
        return bool(getattr(cls, "_pass_through", False))
    out = _UNBATCHED_MEMO.get(cls)
    if out is not None:
        return out
    out = bool(getattr(cls, "_pass_through", False))
    _UNBATCHED_MEMO[cls] = out
    return out


# The _foreach ops whose per-tensor op is not ``torch.Tensor.<name>``:
# _foreach_maximum and _foreach_minimum dispatch to clamp_min and clamp_max
# (torch.maximum takes no Python scalar), and _foreach_norm takes the
# arguments of torch.linalg.vector_norm.
_FOREACH_PER_TENSOR_OPS = {
    "maximum": torch.clamp_min,
    "maximum_": torch.Tensor.clamp_min_,
    "minimum": torch.clamp_max,
    "minimum_": torch.Tensor.clamp_max_,
    "norm": torch.linalg.vector_norm,
}


def _any_unbatched(tensors) -> bool:
    # A loop is faster than any() over a generator for short lists. The class
    # attribute is read directly, as in _is_unbatched, which is slower per call
    # for the nn.Parameter leaves of TensorDictParams.
    for tensor in tensors:
        cls = type(tensor)
        if cls is not Tensor and getattr(cls, "_pass_through", False):
            return True
    return False


# Python picks the overload of torch._foreach_add and torch._foreach_sub that
# takes a number for a 0-dim tensor, and reading the number of a CUDA tensor
# synchronizes the device. The overload of _foreach_add that takes a 0-dim
# tensor does not, and computes ``tensor + alpha * other`` in one kernel.
# Passing alpha makes Python pick it, since the number overload has no alpha.
_FOREACH_ADD_TENSOR = frozenset({"add", "add_", "sub", "sub_"})
_FOREACH_SUB = frozenset({"sub", "sub_"})
_FLOAT32_64 = (torch.float32, torch.float64)
# A subclass with __torch_function__ would see get_device and, for a sub,
# _foreach_add
_PLAIN_TENSOR_TYPES = (Tensor, torch.nn.Parameter)
# The check costs about 0.2 us per tensor, about a tenth of what add_ costs per
# leaf; the cap keeps it to about 25 us, the order of the sync it saves.
_FOREACH_ADD_TENSOR_MAX = 128


def _is_cuda_scalar_of(other, tensors) -> bool:
    # The overload that takes a 0-dim tensor casts it to the dtype of each
    # tensor first, the one that takes a number does not: integer tensors with
    # a float64 0-dim tensor give float64 instead of float32, and float16 or
    # bfloat16 tensors with a wider or integer 0-dim tensor round differently.
    # Both give the same dtypes and values for float32 and float64 tensors, and
    # for tensors of the dtype of a floating-point 0-dim tensor. A subclass
    # of Tensor would turn the results into that subclass. Under a torch
    # function mode, every attribute read below would go through the mode.
    if torch._C._is_torch_function_mode_enabled():
        return False
    if not (
        type(other) is Tensor
        and other.is_cuda
        and not other.ndim
        and other.is_floating_point()
        and not other.requires_grad
    ):
        return False
    # get_device() is cheaper than building a torch.device for every tensor
    device, dtype = other.get_device(), other.dtype
    for tensor in tensors:
        if (
            type(tensor) not in _PLAIN_TENSOR_TYPES
            or tensor.get_device() != device
            or (tensor.dtype not in _FLOAT32_64 and tensor.dtype is not dtype)
        ):
            return False
    return True


def _foreach(name: str, tensors, *args, **kwargs):
    """Calls ``torch._foreach_<name>(tensors, *args, **kwargs)``, one tensor at a time under torch.func or with an unbatched tensor.

    functorch has no batching rule for the ``_foreach`` ops, so inside any
    ``torch.func`` transform (``vmap``, ``grad``, ``jacrev``, ``functionalize``,
    ...) each tensor goes through ``torch.Tensor.<name>``. Under a transform
    without ``vmap``, such as ``grad`` alone, the ``_foreach`` ops would work,
    but they also run one op per tensor. A list or tuple in ``args`` gives one
    value per tensor; any other value is used for every tensor.

    The same per-tensor path is taken when ``tensors`` or a list in ``args``
    holds an unbatched tensor (e.g. :class:`~tensordict.UnbatchedTensor`): its
    ``__torch_dispatch__`` would handle the whole ``_foreach`` call and wrap
    every output in an unbatched tensor of one batch size. It is also taken for
    ``sub`` and ``sub_`` with one tensor and ``alpha``, for which
    ``torch._foreach_sub`` has no overload.
    """
    # Dynamo folds this check to a constant when it traces, also inside a
    # traced vmap. Other checks fail on some supported torch: Dynamo traces
    # ``get_dynamic_layer_stack_depth()`` only from torch 2.13, and evaluates
    # ``peek_interpreter_stack() is None`` as False.
    if not torch._C._are_functorch_transforms_active() and not (
        _any_unbatched(tensors)
        or any(isinstance(arg, (list, tuple)) and _any_unbatched(arg) for arg in args)
    ):
        if (
            name in _FOREACH_ADD_TENSOR
            and len(tensors) <= _FOREACH_ADD_TENSOR_MAX
            # sign * True would turn a bool alpha, which torch rejects for
            # floating-point results, into an int
            and type(kwargs.get("alpha")) is not bool
            and _is_cuda_scalar_of(args[0], tensors)
        ):
            # Pass a copy, read once as the number would be: the in-place kernel
            # can read other after writing a tensor that holds it, and autograd
            # saves other for backward, which fails if other was made in
            # inference mode or is changed in place before backward. Without
            # alpha, the number also dropped the forward-mode tangent of other.
            other = args[0].clone() if "alpha" in kwargs else args[0].detach().clone()
            op = torch._foreach_add_ if name.endswith("_") else torch._foreach_add
            sign = -1 if name.startswith("sub") else 1
            return op(tensors, other, alpha=sign * kwargs.get("alpha", 1))
        if not (
            name in _FOREACH_SUB
            and args
            and isinstance(args[0], Tensor)
            and "alpha" in kwargs
            and type(kwargs["alpha"]) is not bool
        ):
            return getattr(torch, "_foreach_" + name)(tensors, *args, **kwargs)
        # torch._foreach_sub has no overload for one tensor with alpha
        # (_foreach_add has): subtract it from one tensor at a time
    op = _FOREACH_PER_TENSOR_OPS.get(name) or getattr(torch.Tensor, name)
    per_tensor_args = [
        arg if isinstance(arg, (list, tuple)) else [arg] * len(tensors) for arg in args
    ]
    results = [
        op(tensor, *tensor_args, **kwargs)
        for tensor, *tensor_args in _zip_strict(tensors, *per_tensor_args)
    ]
    return None if name.endswith("_") else results


# Set the TD_CHECK_INVARIANTS environment variable (as the CI does) to check the
# tensordicts that are built without validation, see _check_invariants.
_CHECK_INVARIANTS = bool(_strtobool(os.environ.get("TD_CHECK_INVARIANTS", "0")))


def _check_invariants(td) -> None:
    """Raise an ``AssertionError`` if the batch size, entries or dim names of ``td`` disagree.

    ``TensorDict._new_unsafe`` does not validate its inputs. When
    ``TD_CHECK_INVARIANTS`` is set, it calls this function, so that code that
    builds a tensordict with a wrong batch size or wrong dim names fails where
    the error is made. The ``except RuntimeError`` fallbacks of the library do
    not catch an ``AssertionError``.
    """
    from tensordict.base import _is_tensor_collection

    batch_size = td._batch_size
    if type(batch_size) is not torch.Size:
        raise AssertionError(
            f"The batch size {batch_size!r} is a {type(batch_size).__name__}, "
            f"not a torch.Size."
        )
    batch_dims = len(batch_size)
    for key, value in td._tensordict.items():
        if not isinstance(value, Tensor) and not _is_tensor_collection(type(value)):
            # other values have no batch dims (e.g. the tuples that torch.func
            # puts in a pytree)
            continue
        if _is_unbatched(value):
            # unbatched entries do not follow the batch size
            continue
        shape = _shape(value)
        if shape[:batch_dims] != batch_size:
            raise AssertionError(
                f"The entry {key!r} has shape {shape}, which does not start with "
                f"the batch size {batch_size}."
            )
    names = td._td_dim_names
    if names is not None and len(names) != batch_dims:
        raise AssertionError(
            f"The dim names {names} do not match the batch size {batch_size}."
        )


_NON_TENSOR_MEMO = {}


def _is_non_tensor(cls: type):
    if cls is Tensor:
        # The most common entry type: skip is_compiling() and the memo lookup.
        return False
    if is_compiling():
        return bool(getattr(cls, "_is_non_tensor", False))
    out = _NON_TENSOR_MEMO.get(cls)
    if out is None:
        out = _NON_TENSOR_MEMO[cls] = bool(getattr(cls, "_is_non_tensor", False))
    return out


_PASSTHROUGH_MEMO = {}


def _pass_through_cls(cls: type):
    if cls is Tensor:
        # The most common entry type: skip is_compiling() and the memo lookup.
        return False
    if is_compiling():
        return bool(getattr(cls, "_is_non_tensor", False)) or getattr(
            cls, "_pass_through", False
        )
    out = _PASSTHROUGH_MEMO.get(cls)
    if out is None:
        out = _PASSTHROUGH_MEMO[cls] = bool(
            getattr(cls, "_is_non_tensor", False)
        ) or getattr(cls, "_pass_through", False)
    return out


class _KeyDependentDefaultDict(collections.defaultdict):
    """A key-dependent default dict.

    Examples:
        >>> my_dict = _KeyDependentDefaultDict(lambda key: "foo_" + key)
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


def _is_namedtuple(obj):
    """Check if obj is a namedtuple."""
    return isinstance(obj, tuple) and hasattr(obj, "_fields")


def _is_namedtuple_class(cls):
    """Check if a class is a namedtuple class."""
    base_attrs = {"_fields", "_replace", "_asdict"}
    return all(hasattr(cls, attr) for attr in base_attrs)


def _make_dtype_promotion(func):
    dtype = getattr(torch, func.__name__)

    @wraps(func)
    def new_func(self, *, cast_filter=None):
        def todtype(x):
            return x.to(dtype)

        if cast_filter is None:
            return self._fast_apply(todtype, propagate_lock=True)
        return _cast_filtered(self, todtype, cast_filter, propagate_lock=True)

    new_func.__doc__ = rf"""Returns a tensordict whose tensor leaves are cast to ``{str(dtype)}``.
{func.__doc__ or ""}
        Keyword Args:
            cast_filter (Callable, optional): a predicate
                ``fn(key, value) -> bool`` that selects the tensor leaves to
                cast, as in :meth:`to`. Defaults to ``None``, which casts every
                tensor leaf (a tensorclass uses its own ``cast_filter``, if it
                declares one).
        """
    return new_func


def _cast_filtered(td, cast, cast_filter, **kwargs):
    """Applies ``cast`` to the tensor leaves of ``td`` that ``cast_filter`` selects.

    ``cast_filter`` gets the key of each leaf as a tuple of str, and the leaf.
    The other leaves are kept as they are.
    """

    def cast_if(key, tensor):
        if cast_filter(key if type(key) is tuple else (key,), tensor):
            return cast(tensor)
        return tensor

    return td._fast_apply(cast_if, named=True, nested_keys=True, **kwargs)


def is_inexact(*args) -> bool:
    """Whether the last argument is a floating-point or complex tensor.

    The predicate tests its last argument only, so it works both as
    ``fn(value)``, for example inside a function given to
    :meth:`~tensordict.TensorDictBase.apply`, and as ``fn(key, value)``, for
    example as the ``cast_filter`` of :meth:`~tensordict.TensorDictBase.to`.

    Args:
        *args: the value to test is the last one. A key may come before it.

    Returns:
        ``True`` if the dtype of the value is a floating-point or complex dtype.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict, is_inexact
        >>> is_inexact(torch.zeros(3)), is_inexact("a", torch.zeros(3, dtype=torch.long))
        (True, False)
        >>> td = TensorDict(x=torch.zeros(3), i=torch.zeros(3, dtype=torch.long))
        >>> td_bf16 = td.bfloat16(cast_filter=is_inexact)
        >>> td_bf16["x"].dtype, td_bf16["i"].dtype
        (torch.bfloat16, torch.int64)
    """
    dtype = args[-1].dtype
    return dtype.is_floating_point or dtype.is_complex


def is_floating(*args) -> bool:
    """Whether the last argument is a floating-point tensor.

    The predicate tests its last argument only, so it can be passed as
    ``fn(value)`` and as ``fn(key, value)``: see :func:`~tensordict.is_inexact`.

    Args:
        *args: the value to test is the last one. A key may come before it.

    Returns:
        ``True`` if the dtype of the value is a floating-point dtype.

    Examples:
        >>> import torch
        >>> from tensordict import is_floating
        >>> is_floating(torch.zeros(3)), is_floating(torch.zeros(3, dtype=torch.complex64))
        (True, False)
    """
    return args[-1].dtype.is_floating_point


def is_integer(*args) -> bool:
    """Whether the last argument is an integer tensor (signed or unsigned, not bool).

    The predicate tests its last argument only, so it can be passed as
    ``fn(value)`` and as ``fn(key, value)``: see :func:`~tensordict.is_inexact`.

    Args:
        *args: the value to test is the last one. A key may come before it.

    Returns:
        ``True`` if the dtype of the value is neither floating-point, complex
        nor ``torch.bool``.

    Examples:
        >>> import torch
        >>> from tensordict import is_integer
        >>> is_integer(torch.zeros(3, dtype=torch.uint8)), is_integer(torch.zeros(3, dtype=torch.bool))
        (True, False)
    """
    dtype = args[-1].dtype
    return not (dtype.is_floating_point or dtype.is_complex or dtype is torch.bool)


def is_bool(*args) -> bool:
    """Whether the last argument is a boolean tensor.

    The predicate tests its last argument only, so it can be passed as
    ``fn(value)`` and as ``fn(key, value)``: see :func:`~tensordict.is_inexact`.

    Args:
        *args: the value to test is the last one. A key may come before it.

    Returns:
        ``True`` if the dtype of the value is ``torch.bool``.

    Examples:
        >>> import torch
        >>> from tensordict import is_bool
        >>> is_bool(torch.zeros(3, dtype=torch.bool)), is_bool("mask", torch.zeros(3))
        (True, False)
    """
    return args[-1].dtype is torch.bool


def unravel_key_to_tuple(key: NestedKey) -> tuple[str, ...]:
    """Unravels a nested key into a flat tuple of str.

    Unlike :func:`~tensordict.unravel_key`, a str key gives a one-element
    tuple, and an invalid key gives ``()`` instead of raising.

    Args:
        key (NestedKey): the key to unravel.

    Returns:
        A flat tuple of str, or ``()`` if ``key`` is not a valid nested key.

    Examples:
        >>> from tensordict import unravel_key_to_tuple
        >>> unravel_key_to_tuple("a")
        ('a',)
        >>> unravel_key_to_tuple(("a", ("b", ("c",))))
        ('a', 'b', 'c')
        >>> unravel_key_to_tuple(("a", 1))
        ()
        >>> unravel_key_to_tuple(1)
        ()

    """
    if isinstance(key, str):
        return (key,)
    if not isinstance(key, tuple):
        return ()
    for subkey in key:
        if not isinstance(subkey, str):
            break
    else:
        # A flat tuple of str, the most common nested key.
        return key if type(key) is tuple else tuple(key)
    # _append_key_parts(key, result), inline to save a call.
    result = []
    for subkey in key:
        if isinstance(subkey, str):
            result.append(subkey)
        elif (
            not isinstance(subkey, tuple)
            or not subkey
            or not _append_key_parts(subkey, result)
        ):
            return ()
    return tuple(result)


# The name TorchRL and other callers import.
_unravel_key_to_tuple = unravel_key_to_tuple


def _append_key_parts(key: tuple, result: list) -> bool:
    """Append the str parts of the nested ``key`` to ``result``.

    Returns ``False`` if a part is neither a str nor a non-empty tuple of keys.
    """
    for subkey in key:
        if isinstance(subkey, str):
            result.append(subkey)
        elif (
            not isinstance(subkey, tuple)
            or not subkey
            or not _append_key_parts(subkey, result)
        ):
            return False
    return True


def unravel_key(key):
    """Unravels a nested key into a str or a flat tuple of str.

    Nested tuples are flattened, and a tuple with a single str gives that
    str, so equal keys compare equal: ``unravel_key(("a",)) == "a"``. A tuple
    with a part that is neither a str nor a tuple of str unravels to ``()``.
    See :ref:`keys`.

    Args:
        key (NestedKey): the key to unravel.

    Returns:
        A str, a flat tuple of at least two str, or ``()``.

    Raises:
        RuntimeError: if ``key`` is neither a str nor a tuple.

    Examples:
        >>> from tensordict import unravel_key
        >>> unravel_key("a")
        'a'
        >>> unravel_key(("a",))
        'a'
        >>> unravel_key((("a", ("b",))))
        ('a', 'b')
        >>> unravel_key(("a", 1))
        ()

    """
    if isinstance(key, str):
        return key
    if not isinstance(key, tuple):
        raise RuntimeError("key should be a Sequence<NestedKey>")
    # The code of _unravel_key_to_tuple, inline to save a call.
    for subkey in key:
        if not isinstance(subkey, str):
            break
    else:
        if type(key) is not tuple:
            key = tuple(key)
        return key[0] if len(key) == 1 else key
    result = []
    for subkey in key:
        if isinstance(subkey, str):
            result.append(subkey)
        elif (
            not isinstance(subkey, tuple)
            or not subkey
            or not _append_key_parts(subkey, result)
        ):
            return ()
    return result[0] if len(result) == 1 else tuple(result)


def _unravel_keys(*keys):
    """Unravels a sequence of keys."""
    if len(keys) == 1:
        return unravel_key(keys[0])
    return tuple(unravel_key(key) for key in keys)


def unravel_key_list(keys):
    """Unravels every key of a list, as :func:`~tensordict.unravel_key` does.

    Args:
        keys (list or tuple of NestedKey): the keys to unravel.

    Returns:
        A list of the unraveled keys (strings, or flat tuples of strings).

    Raises:
        TypeError: if ``keys`` is not a list or a tuple.
        RuntimeError: if a key is not a valid nested key.

    Examples:
        >>> from tensordict import unravel_key_list
        >>> unravel_key_list(["a", ("b",), ("c", ("d",))])
        ['a', 'b', ('c', 'd')]

    """
    if not isinstance(keys, (list, tuple)):
        # pybind11's wording when this was a C++ function, which TorchRL's
        # tests match.
        raise TypeError(
            "unravel_key_list(): incompatible function arguments. Expected a "
            f"list or a tuple of keys, got {type(keys)}."
        )
    result = []
    for key in keys:
        if isinstance(key, str):
            result.append(key)
            continue
        if type(key) is tuple:
            # A flat tuple of str unravels to itself, or to its only part. A
            # single str subclass part takes the path below, whose comparison
            # with () calls the subclass's __eq__.
            for subkey in key:
                if not isinstance(subkey, str):
                    break
            else:
                if len(key) > 1:
                    result.append(key)
                    continue
                if key and type(key[0]) is str:
                    result.append(key[0])
                    continue
        key = unravel_key(key)
        if key == ():
            raise RuntimeError("key should be a Sequence<NestedKey>")
        result.append(key)
    return result


from tensordict._utils_key_json import (  # noqa: F401
    _decode_key_from_filesystem,
    _encode_key_for_filesystem,
    _get_robust_key_setting_with_warning,
    _is_safe_legacy_key,
)

assert_allclose_td = assert_close


def _prefix_last_key(key, prefix):
    if isinstance(key, str):
        return prefix + key
    if len(key) == 1:
        return (_prefix_last_key(key[0], prefix),)
    return key[:-1] + (_prefix_last_key(key[-1], prefix),)


_NESTED_TENSOR_ERR = (
    "The PyTorch version isn't compatible with "
    "nested tensors. Please upgrade to a more recent "
    "version."
)


def _lock_warn():
    warnings.warn(
        "Using lock_() inside a compiled graph: the parent/child unlock "
        "enforcement (the 'cannot unlock a child while a parent is locked' "
        "invariant) relies on weakrefs and is skipped under torch.compile. "
        "If you need that invariant to be enforced, lock_() in eager mode "
        "before entering the compiled region.",
        category=UserWarning,
    )


_lock_warn = assume_constant_result(_lock_warn)


_zip_strict = functools.partial(zip, strict=True)


def _pin_mem(q_in, q_out):
    # All the inputs are queued before the threads start, so an empty queue means
    # that the other threads took the rest. Checking q_in.empty() before a
    # blocking get() would race with them: two threads could see the last input,
    # and the one that lost would raise queue.Empty after _PIN_MEM_TIMEOUT.
    while True:
        try:
            input = q_in.get_nowait()
        except queue.Empty:
            return
        try:
            key, val = input[0], input[1].pin_memory()
        except Exception as err:
            # Surface the exception
            q_out.put(err)
            return
        q_out.put((key, val))


def _infer_size_impl(shape: List[int], numel: int) -> List[int]:
    # A local copy of torch.jit._shape_functions.infer_size_impl, which is skipped
    # by torch.compile. It raises RuntimeError, as torch.Tensor.view does, where
    # the original raises AssertionError.
    newsize = 1
    infer_dim: int | None = None
    for dim in range(len(shape)):
        if shape[dim] == -1:
            if infer_dim is not None:
                raise RuntimeError("only one dimension can be inferred")
            infer_dim = dim
        elif shape[dim] >= 0:
            newsize *= shape[dim]
        else:
            raise RuntimeError(
                f"invalid shape dimensions in {list(shape)}: sizes must be non-negative or -1"
            )
    if not (
        (infer_dim is None and numel == newsize)
        or (infer_dim is not None and newsize > 0 and numel % newsize == 0)
    ):
        raise RuntimeError(
            f"invalid shape {list(shape)} for a batch of {numel} elements"
        )
    out = _copy(shape)
    if infer_dim is not None:
        out[infer_dim] = numel // newsize
    return out


def parse_tensor_dict_string(s: str):
    """Builds a :class:`~tensordict.TensorDict` from its printed form.

    It is a debugging tool: it rebuilds a tensordict with the keys, shapes,
    dtypes, devices and batch sizes of a printed one, so that you can
    reproduce a tensordict from a log. The values are not in the repr, so
    every tensor of the result is filled with zeros.

    It reads both the compact repr, the default, and the one printed with
    ``set_printoptions(compact=False)``.

    Args:
        s (str): the repr of a tensordict, as printed with the default print
            options or with ``compact=False`` (see
            :func:`~tensordict.set_printoptions`).

    Returns:
        A new :class:`~tensordict.TensorDict`.

    Raises:
        ValueError: if a field is not a ``Tensor`` or a ``TensorDict`` printed
            with the default print options, or if ``s`` is the repr of a lazy
            stack with exclusive keys.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict, parse_tensor_dict_string
        >>> td = TensorDict(
        ...     obs=torch.ones(3, 4),
        ...     next=TensorDict(done=torch.ones(3, 1, dtype=torch.bool)),
        ...     batch_size=[3],
        ... )
        >>> rebuilt = parse_tensor_dict_string(repr(td))
        >>> rebuilt["next", "done"].dtype, rebuilt.batch_size
        (torch.bool, torch.Size([3]))
        >>> rebuilt["obs"].sum()
        tensor(0.)

    """
    from tensordict import TensorDict

    header = _COMPACT_HEADER.search(s)
    if header is not None:
        return _parse_compact_repr(s, header)

    # Regular expression patterns. A field line is "<indent><key>: <value>", and a
    # key can contain any character.
    field_line_pattern = re.compile(r"^[ \t]*(.*?): \w+\(", re.MULTILINE)
    # a tensorclass prints its fields as "<key>=Tensor(...)" and "<key>=TensorDict("
    field_pattern = re.compile(
        r"^[ \t]*(.*?)(?:: |=)Tensor\(shape=torch.Size\((\[(.*?)\])\), device=(\w+(?::\d+)?), dtype=torch.(\w+), is_shared=(\w+)\)",
        re.MULTILINE,
    )
    nested_field_pattern = re.compile(r"^[ \t]*(.*?)(?:: |=)TensorDict\(", re.MULTILINE)
    # A lazy stack prints its exclusive keys in a block that is empty otherwise
    exclusive_fields_pattern = re.compile(
        r"^[ \t]*exclusive_fields=\{\n(?![ \t]*\})", re.MULTILINE
    )
    batch_size_pattern = r"batch_size=torch.Size\((\[(.*?)\])\)"
    device_pattern = r"device=(\w+(?::\d+)?)(?=,|$)"

    # Find the nested TensorDicts of this level first. A match inside a nested
    # TensorDict already found is a field of that TensorDict.
    nested_dicts = []
    for match in nested_field_pattern.finditer(s):
        start_idx = match.start(1)
        if any(start <= start_idx <= end for _, start, _, end in nested_dicts):
            continue
        depth = 1
        for i in range(match.end(), len(s)):
            if s[i] == "(":
                depth += 1
            elif s[i] == ")":
                depth -= 1
            if depth == 0:
                end_idx = i
                break
        nested_dicts.append((match.group(1), start_idx, match.end(), end_idx))
    nested_dict_ranges = [(start, end) for _, start, _, end in nested_dicts]

    # Find all fields in the string that are not part of a nested TensorDict
    fields = {}
    tensor_field_starts = set()
    for match in field_pattern.finditer(s):
        name, _, shape, device, dtype, is_shared = match.groups()
        field_start = match.start(1)
        field_end = match.end()
        if any(
            field_start >= start and field_end <= end
            for start, end in nested_dict_ranges
        ):
            continue  # skip if this field is inside a nested TensorDict
        tensor_field_starts.add(field_start)
        shape = [int(x) for x in shape.split(", ")] if shape else []
        fields[name] = torch.zeros(
            tuple(shape), device=torch.device(device), dtype=getattr(torch, dtype)
        )

    # Every other field line of this level is a field that cannot be parsed
    for match in field_line_pattern.finditer(s):
        line_start = match.start(1)
        if line_start in tensor_field_starts or any(
            start <= line_start <= end for start, end in nested_dict_ranges
        ):
            continue
        line = s[line_start:].split("\n", 1)[0].rstrip(",}")
        raise ValueError(
            f"Cannot parse the field {line!r}: parse_tensor_dict_string reads "
            "Tensor and TensorDict fields printed with the default print options."
        )
    for match in exclusive_fields_pattern.finditer(s):
        if any(start <= match.start() <= end for start, end in nested_dict_ranges):
            continue
        raise ValueError(
            "Cannot parse the exclusive_fields of a lazy stack: "
            "parse_tensor_dict_string does not rebuild lazy stacks with exclusive keys."
        )

    # Now add the nested TensorDicts to the fields
    for name, _, content_start, end_idx in nested_dicts:
        content = s[content_start:end_idx]
        nested_fields = parse_tensor_dict_string(f"TensorDict({content})")
        fields[name] = nested_fields

    # Parse batch size
    batch_size_matches = re.findall(batch_size_pattern, s)
    if batch_size_matches:
        batch_size_match = batch_size_matches[-1]  # Take the last match
        if batch_size_match[1]:
            batch_size = [int(x) for x in batch_size_match[1].split(", ")]
        else:
            batch_size = []
    else:
        raise ValueError("Batch size not found in the string")
    # Parse device
    device_matches = re.findall(device_pattern, s)
    if device_matches:
        device = device_matches[-1]  # Take the last match
        if isinstance(device, str) and device.lower() == "none":
            device = None
        else:
            device = torch.device(device)
    else:
        raise ValueError("Device not found in the string")
    tensor_dict = TensorDict(fields, batch_size=torch.Size(batch_size), device=device)
    return tensor_dict


# The compact repr: "Name(batch_size=[3], device=cpu, fields={a: f32[4], b: {...}})"
_COMPACT_HEADER = re.compile(
    r"\w+\((?:batch_size=\[([\d,]*)\], )?(?:device=([\w:]+), )?(?:stack_dim=\d+, )?fields=(?=\{)"
)
# "f32[4]", "f32[4]@cuda:0", "[5]" (a nested batch dim), "Mesh", "str"
_COMPACT_VALUE = re.compile(r"(\w*)(?:\[([-\d,]*)\])?(?:@([\w:]+(?:,[\w:]+)*))?")


def _compact_dims(dims: str) -> list[int]:
    return [int(x) for x in dims.split(",")] if dims else []


def _parse_compact_repr(s: str, header: re.Match):
    batch_size, device = header.groups()
    if batch_size is None:
        raise ValueError("Batch size not found in the string")
    device = torch.device(device) if device is not None else None
    td, pos = _parse_compact_fields(s, header.end(), _compact_dims(batch_size), device)
    if s.startswith(", exclusive_fields=", pos):
        raise ValueError(
            "Cannot parse the exclusive_fields of a lazy stack: "
            "parse_tensor_dict_string does not rebuild lazy stacks with exclusive keys."
        )
    return td


def _parse_compact_fields(s: str, pos: int, batch_size: list, device):
    # Reads the "{key: value, ...}" block at s[pos] and returns the TensorDict
    # it describes and the position after it.
    from tensordict import TensorDict

    dtypes = {
        _compact_dtype(v): v for v in vars(torch).values() if isinstance(v, torch.dtype)
    }
    fields = {}
    pos = len(s) - len(s[pos + 1 :].lstrip())
    while not s.startswith("}", pos):
        colon = s.find(": ", pos)
        if colon < 0:
            raise ValueError(f"Cannot parse the fields {s[pos:]!r}.")
        key, key_start = s[pos:colon], pos
        value = _COMPACT_VALUE.match(s, colon + 2)
        name, dims, devices = value.groups()
        pos = value.end()
        if s.startswith("{", pos) and not name and not devices:
            sub_batch_size = batch_size + _compact_dims(dims)
            fields[key], pos = _parse_compact_fields(s, pos, sub_batch_size, device)
        elif (
            dims is None
            or "-1" in dims.split(",")
            or name not in dtypes
            or "," in (devices or "")
        ):
            raise ValueError(
                f"Cannot parse the field {s[key_start:pos]!r}: "
                "parse_tensor_dict_string reads Tensor and TensorDict fields "
                "printed with the default print options."
            )
        else:
            fields[key] = torch.zeros(
                batch_size + _compact_dims(dims),
                dtype=dtypes[name],
                device=torch.device(devices or device or "cpu"),
            )
        if s.startswith(",", pos):
            pos = len(s) - len(s[pos + 1 :].lstrip())
        elif not s.startswith("}", pos):
            raise ValueError(f"Cannot parse the fields {s[key_start:]!r}.")
    return TensorDict(fields, batch_size=batch_size, device=device), pos + 1


def _rebuild_njt_from_njt(x, values, offsets, lengths):
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import FunctionalTensor
    from torch.nested._internal.nested_tensor import (
        _tensor_symint_registry,
        NestedTensor,
    )
    from torch.nested._internal.ops import extract_kwargs

    kwargs = extract_kwargs(x)
    kwargs["offsets"] = offsets
    if x._lengths is not None:
        kwargs["lengths"] = lengths
        ragged_source = x._lengths
    else:
        ragged_source = x._offsets
    new_thing = kwargs.get("lengths", kwargs.get("offsets"))
    if isinstance(new_thing, (FakeTensor, FunctionalTensor)):
        from torch._subclasses.functional_tensor import mb_unwrap_functional_tensor

        # Temporary hack until we have the union find
        tgt = mb_unwrap_functional_tensor(new_thing)
        src = mb_unwrap_functional_tensor(ragged_source)
        tgt.nested_int_memo = src.nested_int_memo
    elif new_thing is not None:
        _tensor_symint_registry[new_thing] = _tensor_symint_registry[ragged_source]

    return NestedTensor(
        values,
        **kwargs,
    )


def _mismatch_keys(keys1, keys2):
    def keyfunc(key):
        return "".join(key) if isinstance(key, tuple) else key

    keys1 = sorted(
        keys1,
        key=keyfunc,
    )
    keys2 = sorted(
        keys2,
        key=keyfunc,
    )
    if set(keys1) - set(keys2):
        sub1 = rf"The first TD has keys {set(keys1) - set(keys2)} that the second does not have."
    else:
        sub1 = None
    if set(keys2) - set(keys1):
        sub2 = rf"The second TD has keys {set(keys2) - set(keys1)} that the first does not have."
    else:
        sub2 = None
    main = [r"keys in tensordicts mismatch."]
    if sub1 is not None:
        main.append(sub1)
    if sub2 is not None:
        main.append(sub2)
    raise KeyError(r" ".join(main))


def _is_dataclass(obj):
    """Check if an object is a dataclass."""
    return is_dataclass(obj)


def _is_list_tensor_compatible(t) -> Tuple[bool, tuple | None, type | None]:
    length_t = len(t)
    dtypes = set()
    sizes = set()
    for i in t:
        if isinstance(i, (float, int, torch.SymInt, Number)):
            dtypes.add(type(i))
            if len(dtypes) > 1:
                return False, None, None
            continue
        elif isinstance(i, list):
            is_compat, size_i, dtype = _is_list_tensor_compatible(i)
            if not is_compat:
                return False, None, None
            if dtype is not None:
                dtypes.add(dtype)
            if len(dtypes) > 1:
                return False, None, None
            sizes.add(size_i)
            if len(sizes) > 1:
                return False, None, None
            continue
        return False, None, None
    else:
        if len(dtypes):
            dtype = list(dtypes)[0]
        else:
            dtype = None
        if len(sizes):
            return True, (length_t, *list(sizes)[0]), dtype
        return True, (length_t,), dtype


# The per-thread mode of a thread that set none, which reads the process-wide one
_UNSET_MODE = object()


class _ThreadMode(threading.local):
    # threading.local runs __init__ in each thread on its first access, so
    # reads never miss the attribute (a miss costs an AttributeError)
    def __init__(self):
        self.mode = _UNSET_MODE


class _ContextManager:
    # Reading or writing one attribute is atomic, so no lock is needed: the
    # mode is read on every call of a tensordict.nn module.
    def __init__(self, default=None):
        self._mode: Any | None = default
        self._local = _ThreadMode()
        # Until a thread sets its own mode, reads skip the slower per-thread slot
        self._has_thread_modes = False

    def get_mode(self) -> Any | None:
        if self._has_thread_modes:
            mode = self._local.mode
            if mode is not _UNSET_MODE:
                return mode
        return self._mode

    def set_mode(self, type: Any | None) -> None:
        # In a thread that set its own mode, a change stays in that thread
        if self._has_thread_modes and self._local.mode is not _UNSET_MODE:
            self._local.mode = type
        else:
            self._mode = type

    def get_thread_mode(self) -> Any:
        return self._local.mode

    def set_thread_mode(self, type: Any) -> None:
        self._has_thread_modes = True
        self._local.mode = type


def _maybe_correct_neg_dim(
    dim: int, shape: torch.Size | None, ndim: int | None = None
) -> int:
    """Corrects neg dim to pos."""
    if ndim is None:
        ndim = len(shape)
    if dim < 0:
        new_dim = ndim + dim
    else:
        new_dim = dim
    if new_dim < 0 or new_dim >= ndim:
        if shape is not None:
            raise IndexError(
                f"Incompatible dim {new_dim} for tensordict with shape {shape}."
            )
        raise IndexError(
            f"Incompatible dim {new_dim} for tensordict with batch dims {ndim}."
        )
    return new_dim


def _check_unflatten_size(
    batch_size: torch.Size, dim: int, unflattened_size: Sequence[int]
) -> None:
    """Raises as torch.unflatten does when the sizes don't multiply up to the dim size."""
    if (
        -1 not in unflattened_size
        and -1 not in batch_size
        and math.prod(unflattened_size) != batch_size[dim]
    ):
        raise RuntimeError(
            f"unflatten: Provided sizes {list(unflattened_size)} don't multiply up "
            f"to the size of dim {dim} ({batch_size[dim]}) in the input tensor"
        )


# tensordict, and torch, whose torch function modes (such as the one of
# torch.set_default_device) can call tensordict's torch.stack and torch.cat
_LIBRARY_DIRS = (
    os.path.dirname(os.path.abspath(__file__)) + os.sep,
    os.path.dirname(os.path.abspath(torch.__file__)) + os.sep,
)


def _user_stacklevel() -> int:
    """The ``stacklevel`` of the first frame outside tensordict and torch, for a warning emitted by the caller."""
    frame, stacklevel = sys._getframe(2), 2
    while frame is not None and frame.f_code.co_filename.startswith(_LIBRARY_DIRS):
        frame, stacklevel = frame.f_back, stacklevel + 1
    return stacklevel


# Set while to_h5/to_zarr write a tensordict: True once the save has warned, so
# a save with several UnbatchedTensor leaves warns once. Python's per-call-site
# dedup can't be relied on: a library that changes the warnings filters during
# the save (zarr 3.0) resets it.
_UNBATCHED_SAVE = threading.local()


def _warn_unbatched_saved_as_plain() -> None:
    """Warns that an h5, zarr or ``.tdz`` save writes an UnbatchedTensor leaf as a plain tensor (#3264)."""
    state = getattr(_UNBATCHED_SAVE, "warned", None)
    if state:
        return
    if state is False:
        _UNBATCHED_SAVE.warned = True
    warnings.warn(
        "UnbatchedTensor leaves are saved as plain tensors: they are read back "
        "as plain tensors, or fail to load when their shape does not fit the "
        "batch size. Exclude them before saving.",
        UserWarning,
        stacklevel=_user_stacklevel(),
    )


def _warn_dim_past_batch_dims(op: str) -> None:
    """Warns that ``op`` got a dim outside the batch dims, which raises from 0.17 (#2148)."""
    warnings.warn(
        f"{op} got a dim outside the batch dims, and uses it on the dims of "
        "the leaves instead. Starting with TensorDict 0.17, the default will "
        "change to raising an IndexError, as torch does for a tensor of the "
        "batch size.",
        FutureWarning,
        stacklevel=_user_stacklevel(),
    )


def _warn_batch_dims_ignored(op: str, result, batch_dims, batch_size) -> None:
    """Warns that ``op`` ignored ``batch_dims`` without ``auto_batch_size``, where turning it on, as from 0.17, changes ``result`` (#2979)."""
    from tensordict.base import _is_tensor_collection

    if batch_size is None:
        # with a batch_size, auto_batch_size raises: the result changes
        try:
            expected = result.copy().auto_batch_size_(batch_dims=batch_dims)
        except Exception:
            expected = None
        if expected is not None and (
            expected.batch_size == result.batch_size
            and all(
                expected.get(key).batch_size == value.batch_size
                for key, value in result.items(include_nested=True)
                if _is_tensor_collection(type(value))
            )
        ):
            return
    # with a batch_size, auto_batch_size=True raises: advise it only without
    advice = (
        "Drop batch_dims to keep today's result."
        if batch_size is not None
        else "Pass auto_batch_size=True to get that result now, or drop batch_dims."
    )
    warnings.warn(
        f"{op} ignores batch_dims={batch_dims} without auto_batch_size=True. "
        "Starting with TensorDict 0.17, the default will change to turning "
        f"auto_batch_size on when batch_dims is given, as from_dict does. {advice}",
        FutureWarning,
        stacklevel=_user_stacklevel(),
    )


def _warn_lazy_shape_args(op: str, what: str) -> None:
    """Warns that a lazy stack accepted arguments of ``op`` that ``TensorDict`` rejects, which raises from 0.17.

    ``what`` describes the arguments in a fixed text, without their values.
    """
    warnings.warn(
        f"LazyStackedTensorDict.{op} got {what}, which TensorDict.{op} "
        "rejects, and returns the result of its members. Starting with "
        "TensorDict 0.17, the default will change to raising a ValueError, "
        f"as TensorDict.{op} does.",
        FutureWarning,
        stacklevel=_user_stacklevel(),
    )


# Check if the new shape is a flatten / unflatten version of the current one
def _check_is_flatten(new_shape, old_shape, return_flatten_dim=False):
    if not new_shape or not old_shape:
        if return_flatten_dim:
            return False, (-1, -1)
        return False
    if new_shape.numel() != old_shape.numel():
        if return_flatten_dim:
            return False, (-1, -1)
        return False
    # a shape is a flatten version of another if all the first sizes and/or all the last sizes match
    for i, (first_new, first_old) in enumerate(zip(new_shape, old_shape)):  # noqa: B007
        if first_new != first_old:
            break
    # 'i' must be the result of the flatten op
    for j, (last_new, last_old) in enumerate(  # noqa: B007
        zip(reversed(new_shape), reversed(old_shape))
    ):
        if last_new != last_old:
            break
    # j is also the result of the flatten, so if j and i match this is the result of a flatten
    if i == len(new_shape) - j - 1:
        if return_flatten_dim:
            j = len(old_shape) - j - 1
            return True, (i, j)
        return True
    if return_flatten_dim:
        return False, (-1, -1)
    return False


def _check_is_unflatten(new_shape, old_shape, return_flatten_dim=False):
    out = _check_is_flatten(old_shape, new_shape, return_flatten_dim=return_flatten_dim)
    if return_flatten_dim:
        out, (i, j) = out
        # if out:
        #     j = len(new_shape) - j - 1
        return out, (i, j)
    return out


def _create_segments_from_int(split_size, max_size):
    if split_size <= 0:
        raise RuntimeError(
            f"split_size must be a positive integer, but got {split_size}."
        )
    splits = [
        (start, min(start + split_size, max_size))
        for start in range(0, max_size, split_size)
    ]
    return splits


def _create_segments_from_list(
    split_size: list[int] | tuple[int],
    max_size: int,
):
    splits = [
        (start, min(start + size, max_size))
        for start, size in zip(
            [0] + list(itertools.accumulate(split_size[:-1])),
            split_size,
        )
    ]
    total_split_size = sum(split_size)
    if total_split_size != max_size:
        raise RuntimeError(
            f"Split method expects split_size to sum exactly to {max_size}, "
            f"but got sum({split_size}) = {total_split_size}"
        )

    return splits


from tensordict._utils_key_json import _json_dumps  # noqa: F401


class LinkedList(list):
    """A list of the values of a non-tensor entry that writes item assignments back to the tensordict.

    ``td[key]`` returns a :class:`LinkedList` when the entry is a
    :class:`~tensordict.NonTensorStack`, and so does
    :meth:`~tensordict.TensorDictBase.tolist` with ``as_linked_list=True``.
    Setting an item (``values[0] = "z"``) writes the value to the tensordict.
    The methods that change the length of the list (``append``, ``extend``,
    ``insert``, ``pop``, ``remove``, ``clear`` and ``del``) change the list
    only, and unlink it from the tensordict: later item assignments no longer
    write back. A :class:`LinkedList` compares equal to a list with the same
    items. If the write to the tensordict raises, the list is unchanged.
    :meth:`tolist` returns a plain :class:`list` copy that is not linked to
    the tensordict, as :meth:`~tensordict.NonTensorStack.tolist` does.
    ``LinkedList(iterable)``, without ``td``, builds a list that is not
    linked to any tensordict.

    Examples:
        >>> from tensordict import TensorDict
        >>> td = TensorDict(batch_size=[3])
        >>> td["names"] = ["a", "b", "c"]
        >>> values = td["names"]
        >>> values[0] = "z"
        >>> td.get("names").tolist()
        ['z', 'b', 'c']
        >>> values.tolist()
        ['z', 'b', 'c']

    """

    def __init__(self, *args, td: TensorDictBase | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Without td, as in type(values)(iterable), which code that rebuilds a
        # sequence of the same type calls, the list is not linked.
        self._td = weakref.ref(td) if td is not None else (lambda: None)

    def __setitem__(self, key, value):
        td = self._td()
        if td is None:
            super().__setitem__(key, value)
            return
        # Keep the old items, to put them back if the write to the tensordict
        # fails.
        is_slice = isinstance(key, slice)
        old = list(self) if is_slice else list.__getitem__(self, key)
        super().__setitem__(key, value)
        try:
            td[key] = value
        except Exception:
            super().__setitem__(slice(None) if is_slice else key, old)
            raise

    def tolist(self) -> list:
        """Returns a plain :class:`list` copy of the items, not linked to the tensordict.

        Nested :class:`LinkedList` items, as ``td[key]`` returns for a
        non-tensor entry with more than one batch dimension, become nested
        lists, as with :meth:`~tensordict.NonTensorStack.tolist`. The other
        items are not copied.

        Examples:
            >>> from tensordict import TensorDict
            >>> td = TensorDict(batch_size=[2, 2])
            >>> td["names"] = [["a", "b"], ["c", "d"]]
            >>> td["names"].tolist()
            [['a', 'b'], ['c', 'd']]

        """
        return [
            item.tolist() if isinstance(item, LinkedList) else item for item in self
        ]

    def append(self, object: Any) -> None:
        """Appends ``object`` to the list and unlinks the list from the tensordict."""
        self._td = lambda: None
        return super().append(object)

    def extend(self, object: Any) -> None:
        """Extends the list with the items of ``object`` and unlinks it from the tensordict."""
        self._td = lambda: None
        return super().extend(object)

    def insert(self, index: int, object: Any) -> None:
        """Inserts ``object`` before ``index`` and unlinks the list from the tensordict."""
        self._td = lambda: None
        return super().insert(index, object)

    def pop(self, index: int = -1) -> Any:
        """Removes and returns the item at ``index`` and unlinks the list from the tensordict."""
        self._td = lambda: None
        return super().pop(index)

    def remove(self, value: Any) -> None:
        """Removes the first item equal to ``value`` and unlinks the list from the tensordict."""
        self._td = lambda: None
        return super().remove(value)

    def clear(self) -> None:
        """Removes every item and unlinks the list from the tensordict."""
        self._td = lambda: None
        return super().clear()

    def __delitem__(self, index: int) -> None:
        self._td = lambda: None
        return super().__delitem__(index)

    def __repr__(self) -> str:
        return f"LinkedList({super().__repr__()})"

    # No __str__ override: list has no __str__ of its own, so a
    # super().__str__() call would dispatch back to this class's __repr__
    # and print a doubly-wrapped "LinkedList(LinkedList([...]))".

    def __eq__(self, other: Any) -> bool:
        return list(self) == other

    def __ne__(self, other: Any) -> bool:
        return list(self) != other

    def __reduce__(self):
        # A copy is not linked to the tensordict (a weakref does not pickle).
        # Python 3.10 has no object.__getstate__, which a list subclass would
        # need for the default reduction.
        return type(self), (list(self),)

    def __iadd__(self, other: Any) -> Any:
        self._td = lambda: None
        return super().__iadd__(other)

    def __imul__(self, other: Any) -> Any:
        self._td = lambda: None
        return super().__imul__(other)


# register LinkedList in PyTree
torch.utils._pytree.register_pytree_node(
    LinkedList,
    torch.utils._pytree._list_flatten,
    torch.utils._pytree._list_unflatten,
    serialized_type_name="tensordict.utils.LinkedList",
    flatten_with_keys_fn=torch.utils._pytree._list_flatten_with_keys,
)


__getattr__ = deprecated_attributes(
    __name__,
    {
        "BufferLegacy": (_BufferLegacy, None),
        "KeyDependentDefaultDict": (_KeyDependentDefaultDict, None),
        "NESTED_TENSOR_ERR": (_NESTED_TENSOR_ERR, None),
        "NUMPY_TO_TORCH_DTYPE_DICT": (_NUMPY_TO_TORCH_DTYPE_DICT, None),
        "TORCH_TO_NUMPY_DTYPE_DICT": (_TORCH_TO_NUMPY_DTYPE_DICT, None),
        "cache": (_cache_while_locked, None),
        "convert_ellipsis_to_idx": (_indexing.convert_ellipsis_to_idx, None),
        "erase_cache": (_erase_cache_first, None),
        "get_json_backend": (_utils_key_json.get_json_backend, None),
        # tensordict depends on pyvers only for this alias and for the
        # implement_for aliases of tensordict and tensordict.tensordict: drop
        # pyvers from pyproject.toml with them.
        "implement_for": (pyvers.implement_for, "pyvers.implement_for"),
        "infer_size_impl": (_infer_size_impl, None),
        "int_generator": (_int_generator, None),
        "is_nested_key": (
            _is_nested_key,
            "isinstance(key, tensordict.NestedKey) (which rejects lists and "
            "accepts nested tuples)",
        ),
        "is_namedtuple": (_is_namedtuple, None),
        "is_namedtuple_class": (_is_namedtuple_class, None),
        "is_seq_of_nested_key": (
            _is_seq_of_nested_key,
            "isinstance(key, tensordict.NestedKey) on each key (which rejects "
            "lists and accepts nested tuples)",
        ),
        "json_dumps": (_utils_key_json.json_dumps, "json.dumps or orjson.dumps"),
        "lock_blocked": (_lock_blocked, None),
        "prod": (_prod, "math.prod"),
        "set_json_backend": (_utils_key_json.set_json_backend, None),
        "strtobool": (_strtobool, None),
        "unravel_keys": (_unravel_keys, "unravel_key or unravel_key_list"),
    },
    removal="0.17",
)

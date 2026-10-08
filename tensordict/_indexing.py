# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""How torch reads an index.

Torch indexes the entries of a tensordict, so what a tensordict computes from
an index without indexing a tensor, such as its batch size, must follow
torch's rules. This module reads an index as torch does, so that the rest of
the library does it in one place.

Torch reads the elements of an index from the first dim on:

* An int, or anything that torch reads as one (a NumPy integer, a 0-d integer
  tensor or ndarray), selects one position and removes its dim.
* A slice keeps its dim.
* ``None`` adds a dim of size 1.
* ``...`` stands for as many ``:`` as there are dims that the other elements
  do not use.
* The other elements are advanced indices. An integer tensor, ndarray, list or
  range uses one dim, a k-D boolean mask uses k dims, and a scalar bool uses
  none. The advanced indices broadcast together into one block of dims. The
  block takes the place of the first advanced index, unless a slice or
  ``None`` separates two advanced indices, in which case it goes first.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    from torch.compiler import is_compiling
except ImportError:  # torch 2.0
    from torch._dynamo import is_compiling

# The kinds of index elements, see _read_element
_INT = 0
_SLICE = 1
_NONE = 2
_ELLIPSIS = 3
_INDEX = 4
_MASK = 5
_BOOL = 6


def _is_list_of_bools(index) -> bool:
    """Whether ``index`` is a non-empty list of bools (a boolean mask for torch)."""
    return (
        isinstance(index, list)
        and bool(index)
        and all(isinstance(elt, (bool, np.bool_)) for elt in index)
    )


def _nested_list_to_tensor(index):
    """Converts a nested list index to a tensor, which is how torch reads it."""
    if isinstance(index, list) and index and isinstance(index[0], list):
        return torch.tensor(index)
    return index


def _as_tuple(index) -> tuple:
    """Return ``index`` as a tuple: tensordict reads a bare index as a 1-tuple.

    torch reads a bare nested list, or a list with a slice or a tensor in it,
    as a tuple of indices instead (deprecated).
    """
    return index if isinstance(index, tuple) else (index,)


def _read_element(element):
    """Read an element of an index as torch does.

    Returns ``(kind, num_dims, element)``, where ``num_dims`` is the number of
    dims that the element uses, and ``kind`` is one of:

    * ``_INT``: an int, or anything that torch reads as one;
    * ``_SLICE``, ``_NONE`` and ``_ELLIPSIS``;
    * ``_INDEX``: an integer tensor, ndarray, list or range with one or more
      dims;
    * ``_MASK``: a boolean mask with one or more dims;
    * ``_BOOL``: a scalar bool, which torch reads as a 0-d mask.

    The returned element is the given one, except that a list of bools or a
    nested list is returned as a tensor and a uint8 tensor as a boolean mask,
    as torch reads them.
    """
    if element is None:
        return _NONE, 0, element
    if element is Ellipsis:
        return _ELLIPSIS, 0, element
    if isinstance(element, slice):
        return _SLICE, 1, element
    if isinstance(element, (bool, np.bool_)):
        # NumPy bools are read as by NumPy: torch reads them as ints with
        # NumPy 1 and rejects them with NumPy 2
        return _BOOL, 0, element
    if isinstance(element, range):
        return _INDEX, 1, element
    if isinstance(element, list):
        if _is_list_of_bools(element):
            element = torch.tensor(element)
        else:
            element = _nested_list_to_tensor(element)
            if isinstance(element, list):
                return _INDEX, 1, element
    if isinstance(element, torch.Tensor):
        if element.dtype == torch.uint8:
            # torch reads a uint8 tensor as a mask (deprecated)
            element = element.bool()
        is_mask = element.dtype == torch.bool
    elif isinstance(element, np.ndarray):
        is_mask = element.dtype == np.dtype("bool")
    else:
        # an int, or anything else that torch reads as one
        return _INT, 1, element
    if is_mask:
        if element.ndim:
            return _MASK, element.ndim, element
        return _BOOL, 0, element
    if element.ndim:
        return _INDEX, 1, element
    # torch reads a 0-d integer index as an int
    return _INT, 1, element


def _num_indexed_dims(index) -> int:
    """Number of dims that an element of an index uses."""
    return _read_element(index)[1]


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
    if idx is Ellipsis:
        idx = (idx,)
    elif not isinstance(idx, tuple):
        return idx
    position = None
    for i, element in enumerate(idx):
        if element is Ellipsis:
            if position is not None:
                raise RuntimeError("An index can only have one ellipsis at most.")
            position = i
    if position is None:
        return idx
    # the ellipsis covers the dims that the other index elements do not use
    ellipsis_length = len(batch_size) - sum(
        _num_indexed_dims(element) for element in idx if element is not Ellipsis
    )
    if ellipsis_length < 0:
        raise RuntimeError("Not enough dimensions in TensorDict for index provided.")
    return idx[:position] + (slice(None),) * ellipsis_length + idx[position + 1 :]


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
        if isinstance(index, int) and not isinstance(index, bool):
            return batch_size[1:]
        if isinstance(index, slice) and index == slice(None):
            return batch_size
        index = (index,)
    index = convert_ellipsis_to_idx(index, batch_size)
    out = []
    dim = 0
    # the shapes of the advanced indices, and where their block goes
    advanced = []
    position = None
    after_gap = separated = False
    for element in index:
        kind, num_dims, element = _read_element(element)
        if kind == _INT:
            dim += 1
        elif kind == _SLICE:
            out.append(_slice_length(element, batch_size[dim]))
            dim += 1
            after_gap = position is not None
        elif kind == _NONE:
            out.append(1)
            after_gap = position is not None
        else:
            if position is None:
                position = len(out)
            elif after_gap:
                separated = True
            if kind == _MASK:
                # int() graph-breaks on the data-dependent size under compile
                advanced.append((int(element.sum()),))
            elif kind == _BOOL:
                advanced.append((int(element),))
            elif isinstance(element, (list, range)):
                advanced.append((len(element),))
            else:
                advanced.append(element.shape)
            dim += num_dims
    out.extend(batch_size[dim:])
    if advanced:
        if separated:
            position = 0
        out[position:position] = (
            advanced[0] if len(advanced) == 1 else torch.broadcast_shapes(*advanced)
        )
    return torch.Size(out)


def _slice_length(index: slice, size: int) -> int:
    """The length of ``range(size)[index]``."""
    if is_compiling():
        # torch.compile cannot trace slice.indices before torch 2.13
        return _traceable_slice_length(index, size)
    return len(range(*index.indices(size)))


def _traceable_slice_length(index: slice, size: int) -> int:
    """``len(range(*index.indices(size)))``, computed as CPython does."""
    step = 1 if index.step is None else index.step
    if step == 0:
        raise ValueError("slice step cannot be zero")
    if step > 0:
        lower, upper = 0, size
    else:
        lower, upper = -1, size - 1
    start, stop = index.start, index.stop
    if start is None:
        start = lower if step > 0 else upper
    elif start < 0:
        start = max(start + size, lower)
    else:
        start = min(start, upper)
    if stop is None:
        stop = upper if step > 0 else lower
    elif stop < 0:
        stop = max(stop + size, lower)
    else:
        stop = min(stop, upper)
    if step > 0:
        return max(0, (stop - start + step - 1) // step)
    return max(0, (start - stop - step - 1) // -step)

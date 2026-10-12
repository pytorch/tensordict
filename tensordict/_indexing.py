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


def _entry_index(index):
    """Return ``index`` as tensordict reads it, for an entry indexed with it.

    tensordict reads a bare index as a 1-tuple, where torch reads a bare list
    as a tuple of indices (deprecated). It also rejects NumPy bool scalars, see
    :func:`_read_element`.
    """
    if isinstance(index, (int, slice, torch.Tensor)):
        # the most common indices, which need no change
        return index
    if isinstance(index, list):
        return (index,)
    for element in _as_tuple(index):
        _read_element(element)
    return index


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
    as torch reads them. A NumPy bool scalar raises an ``IndexError``, as in
    torch with NumPy 2.3 and later.
    """
    if element is None:
        return _NONE, 0, element
    if element is Ellipsis:
        return _ELLIPSIS, 0, element
    if isinstance(element, slice):
        return _SLICE, 1, element
    if isinstance(element, bool):
        return _BOOL, 0, element
    if isinstance(element, np.bool_):
        # torch reads a NumPy bool as an int before NumPy 2.3, and rejects it
        # from NumPy 2.3 on: reject it with any NumPy
        raise IndexError(
            f"A NumPy bool scalar is not a valid index, got {element!r}. Use a "
            "Python bool or a boolean tensor instead."
        )
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


def _is_new_dim_index(index) -> bool:
    """Whether ``index`` is ``None`` or a true scalar bool, which add a dim of size 1 and select all of it."""
    if index is None or index is True:
        return True
    return (
        isinstance(index, torch.Tensor)
        and index.shape == ()
        and index.dtype == torch.bool
        and bool(index)
    )


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
    return _expand_ellipsis(idx, len(batch_size))


def _expand_ellipsis(index: tuple, ndim: int) -> tuple:
    """Replace the ``Ellipsis`` of ``index`` by the ``:`` that it stands for."""
    position = None
    for i, element in enumerate(index):
        if element is Ellipsis:
            if position is not None:
                raise RuntimeError("An index can only have one ellipsis at most.")
            position = i
    if position is None:
        return index
    # the ellipsis covers the dims that the other index elements do not use
    ellipsis_length = ndim - sum(
        _num_indexed_dims(element) for element in index if element is not Ellipsis
    )
    if ellipsis_length < 0:
        raise RuntimeError("Not enough dimensions in TensorDict for index provided.")
    return index[:position] + (slice(None),) * ellipsis_length + index[position + 1 :]


def _read_index(index, ndim, sizes=None):
    """Read ``index`` as torch does for a tensor with ``ndim`` dims.

    If the ``sizes`` of the dims are given, the index is checked against them
    as torch checks it: too many indices, ints out of range and masks of
    another shape raise the ``IndexError`` that torch raises. Index tensors
    are not read, so the values out of range that they may hold are not found.

    Returns ``(dims, advanced, position, rest)``:

    * ``dims`` describes the dims that the slices and ``None`` give, in order,
      as ``(input_dim, element)``: ``element`` is the slice of ``input_dim``,
      and ``input_dim`` and ``element`` are ``None`` for a dim that ``None``
      adds.
    * ``advanced`` has a ``(kind, element, input_dim, num_dims)`` for each
      advanced index, which uses the ``num_dims`` dims from ``input_dim``.
    * ``position`` is where the block of the advanced dims goes in ``dims``.
    * The input dims from ``rest`` on are not indexed, and follow ``dims``.
    """
    if not isinstance(index, tuple):
        index = (index,)
    given = index
    try:
        index = _expand_ellipsis(index, ndim)
    except RuntimeError:
        # too many indices for the dims, as torch says
        if (
            sizes is not None
            and sum(
                _num_indexed_dims(element)
                for element in given
                if element is not Ellipsis
            )
            > ndim
        ):
            _raise_index_error(
                sizes, given, f"too many indices for tensor of dimension {ndim}"
            )
        raise
    dims = []
    advanced = []
    position = None
    after_gap = separated = False
    dim = 0
    error = None
    for element in index:
        kind, num_dims, element = _read_element(element)
        if sizes is not None and error is None:
            # the conditions are checked here, and the message is only built
            # for an element that fails them
            if kind == _INT:
                if isinstance(element, (int, np.integer)) and not (
                    dim < ndim and -sizes[dim] <= element < sizes[dim]
                ):
                    error = _element_error(kind, element, dim, num_dims, sizes)
            elif kind == _MASK and element.shape != sizes[dim : dim + num_dims]:
                error = _element_error(kind, element, dim, num_dims, sizes)
        if kind == _INT:
            dim += 1
        elif kind == _SLICE:
            dims.append((dim, element))
            dim += 1
            after_gap = position is not None
        elif kind == _NONE:
            dims.append((None, None))
            after_gap = position is not None
        else:
            if position is None:
                position = len(dims)
            elif after_gap:
                separated = True
            advanced.append((kind, element, dim, num_dims))
            dim += num_dims
    if separated:
        position = 0
    if sizes is not None:
        # torch counts the indexed dims first
        if dim > ndim:
            error = f"too many indices for tensor of dimension {ndim}"
        if error is not None:
            _raise_index_error(sizes, given, error)
    return dims, advanced, position, dim


def _element_error(kind, element, dim, num_dims, sizes):
    """The error that torch finds in an element of an index that uses the dims from ``dim`` on, or ``None``."""
    if kind == _INT and isinstance(element, (int, np.integer)):
        if dim < len(sizes) and not -sizes[dim] <= element < sizes[dim]:
            return (
                f"index {element} is out of bounds for dimension {dim} with size "
                f"{sizes[dim]}"
            )
    elif kind == _MASK and element.shape != sizes[dim : dim + num_dims]:
        for i, (size, expected) in enumerate(zip(element.shape, sizes[dim:])):
            if size != expected:
                return (
                    f"The shape of the mask {list(element.shape)} at index {i} does "
                    f"not match the shape of the indexed tensor {list(sizes)} at "
                    f"index {dim + i}"
                )
    return None


def _raise_index_error(sizes, index, message):
    """Raise the ``IndexError`` that torch raises for ``index`` on a tensor of shape ``sizes``.

    Torch writes its message from the part of the tensor that it has indexed
    when it finds the error, so ``index`` is read on a tensor of that shape,
    which a single expanded element backs. ``message`` is used if that tensor
    does not give an ``IndexError``, which happens with index tensors on
    another device than the CPU.
    """
    try:
        torch.zeros(()).expand(sizes)[index]
    except IndexError:
        raise
    except RuntimeError:
        pass
    raise IndexError(message)


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
            if not batch_size or not -batch_size[0] <= index < batch_size[0]:
                _read_index(index, len(batch_size), batch_size)
            return batch_size[1:]
        if isinstance(index, slice) and index == slice(None):
            return batch_size
    elif (
        index
        and type(index[0]) is int
        and type(index[-1]) is int
        and len(index) <= len(batch_size)
    ):
        # A tuple of ints in range removes the dims that it indexes. Its first
        # and last elements are read first, so that the other tuples, such as
        # (int, slice) or the (tensor,) of td[tensor], skip the loop.
        for size, element in zip(batch_size, index):
            if type(element) is not int or not -size <= element < size:
                break
        else:
            return torch.Size(batch_size[len(index) :])
    dims, advanced, position, rest = _read_index(index, len(batch_size), batch_size)
    out = [
        1 if dim is None else _slice_length(element, batch_size[dim])
        for dim, element in dims
    ]
    out.extend(batch_size[rest:])
    if advanced:
        shapes = [_advanced_shape(kind, element) for kind, element, _, _ in advanced]
        if len(shapes) == 1:
            shape = shapes[0]
        else:
            try:
                shape = torch.broadcast_shapes(*shapes)
            except RuntimeError:
                _raise_index_error(
                    batch_size,
                    index,
                    "shape mismatch: indexing tensors could not be broadcast "
                    f"together with shapes {', '.join(str(list(s)) for s in shapes)}",
                )
        out[position:position] = shape
    return torch.Size(out)


def _getitem_names(names, index):
    """Return the dim names that indexing a tensor with dim names ``names`` with ``index`` gives.

    A dim of the result keeps the name of the input dim that it comes from,
    if it comes from exactly one: the dim of a slice or a kept dim, or a dim of
    the advanced block along which only advanced indices that use the same
    single input dim vary. The other dims are unnamed: the dims that ``None``
    adds, and the advanced dims that come from several input dims, such as
    the dim of an N-D mask or of advanced indices that broadcast together.
    Names are unique, so an N-D index of one input dim names at most one dim
    of the block after it: the only one along which it varies.
    """
    dims, advanced, position, rest = _read_index(index, len(names))
    out = [None if dim is None else names[dim] for dim, _ in dims]
    out.extend(names[rest:])
    if advanced:
        out[position:position] = _block_names(names, advanced)
    return out


def _block_names(names, advanced):
    """Return the names of the dims of the advanced block, see :func:`_getitem_names`."""
    shapes = []
    for kind, element, first, num_dims in advanced:
        if kind == _BOOL:
            # a scalar bool uses no dim
            continue
        if kind == _MASK:
            # the number of selected elements, unknown without reading the mask
            shape = (None,)
        elif isinstance(element, (list, range)):
            shape = (len(element),)
        else:
            shape = tuple(element.shape)
        shapes.append((shape, range(first, first + num_dims)))
    ndim = max(_advanced_ndim(kind, element) for kind, element, _, _ in advanced)
    out = []
    for dim in range(-ndim, 0):
        # the input dims of the advanced indices that vary along this dim of
        # the block, or of all those that have it if none varies along it
        having = [(shape[dim], used) for shape, used in shapes if len(shape) >= -dim]
        varying = [used for size, used in having if size != 1]
        sources = set().union(*(varying or [used for _, used in having]))
        name = names[sources.pop()] if len(sources) == 1 else None
        out.append((name, bool(varying)))
    # a name that several dims of the block would take stays only on the one
    # along which its index varies, if there is exactly one
    return [
        name
        if name is None
        or [other for other, _ in out].count(name) == 1
        or (varies and [v for other, v in out if other == name].count(True) == 1)
        else None
        for name, varies in out
    ]


def _advanced_shape(kind, element):
    """The shape that an advanced index contributes to the advanced block."""
    if kind == _MASK:
        # int() graph-breaks on the data-dependent size under compile
        return (int(element.sum()),)
    if kind == _BOOL:
        return (int(element),)
    if isinstance(element, (list, range)):
        return (len(element),)
    return element.shape


def _advanced_ndim(kind, element):
    """The number of dims that an advanced index contributes to the advanced block."""
    if kind in (_MASK, _BOOL) or isinstance(element, (list, range)):
        return 1
    return element.ndim


def _slice_length(index: slice, size: int) -> int:
    """The length of ``range(size)[index]``."""
    return len(range(*index.indices(size)))

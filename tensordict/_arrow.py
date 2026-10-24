# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Conversion between Arrow tables and tensordicts, with pyarrow as an optional dependency."""

from __future__ import annotations

import importlib.util
import warnings
from typing import Any, Literal, TYPE_CHECKING

import torch
from tensordict._td import TensorDict
from tensordict.base import is_tensor_collection, TensorDictBase
from tensordict.tensorclass import NonTensorStack
from tensordict.utils import _user_stacklevel, is_non_tensor, NestedKey, unravel_key

if TYPE_CHECKING:
    import pyarrow

_has_pyarrow = importlib.util.find_spec("pyarrow") is not None

__all__ = ["from_arrow", "to_arrow"]

_NULLS = ("raise", "fill", "mask")
# The signed dtypes of the same size as the unsigned ones that masked_fill lacks.
_SIGNED = {
    torch.uint16: torch.int16,
    torch.uint32: torch.int32,
    torch.uint64: torch.int64,
}


def _import_pyarrow():
    if not _has_pyarrow:
        raise ImportError(
            "from_arrow and to_arrow need pyarrow 15 or later: pip install pyarrow"
        )
    import pyarrow as pa

    return pa


def _check_pyarrow_version(pa, name: str) -> None:
    """Raises an ImportError naming the minimum version on pyarrow older than 15.

    It runs before ``name`` uses any pyarrow type: pyarrow 10 to 14 lack some
    of them, and give tensors that ``torch.from_dlpack`` rejects.
    """
    if int(pa.__version__.split(".")[0]) < 15:
        raise ImportError(
            f"{name} needs pyarrow 15 or later, got pyarrow=={pa.__version__}. "
            "Upgrade with `pip install 'pyarrow>=15'`."
        )


def _is_tensor_type(pa, typ) -> bool:
    """Whether an Arrow type converts to a dense tensor."""
    if isinstance(typ, pa.FixedShapeTensorType):
        return _is_tensor_type(pa, typ.storage_type)
    if pa.types.is_fixed_size_list(typ):
        return _is_tensor_type(pa, typ.value_type)
    return (
        pa.types.is_integer(typ)
        or pa.types.is_floating(typ)
        or pa.types.is_boolean(typ)
    )


def _check_nulls(arr, nulls: str, key: tuple) -> None:
    if arr.null_count and nulls == "raise":
        raise ValueError(
            f"The Arrow column {key} has {arr.null_count} null value(s). Pass "
            "nulls='fill' to fill them, or nulls='mask' to also get a mask entry."
        )


def _check_inner_nulls(pa, arr, key: tuple) -> None:
    """Raises for a null inside the rows of a list or a struct, at any depth."""
    typ = arr.type
    if pa.types.is_struct(typ):
        children = arr.flatten()
    elif (
        pa.types.is_list(typ)
        or pa.types.is_large_list(typ)
        or pa.types.is_fixed_size_list(typ)
    ):
        children = [arr.flatten()]
    else:
        return
    for child in children:
        _check_nulls(child, "raise", key)
        _check_inner_nulls(pa, child, key)


def _is_unchecked(pa, typ) -> bool:
    """Whether ``_check_inner_nulls`` does not look inside an Arrow type: a map, a list view or a dictionary."""
    return (
        pa.types.is_map(typ)
        or pa.types.is_dictionary(typ)
        # pyarrow 15 has no list views.
        or (
            hasattr(pa.types, "is_list_view")
            and (pa.types.is_list_view(typ) or pa.types.is_large_list_view(typ))
        )
    )


def _has_unchecked_type(pa, typ) -> bool:
    """Whether an Arrow type has a map, a list view or a dictionary, at any depth outside unions."""
    if _is_unchecked(pa, typ):
        return True
    if pa.types.is_struct(typ):
        return any(_has_unchecked_type(pa, field.type) for field in typ)
    if (
        pa.types.is_list(typ)
        or pa.types.is_large_list(typ)
        or pa.types.is_fixed_size_list(typ)
    ):
        return _has_unchecked_type(pa, typ.value_type)
    return False


def _has_unchecked_nulls(pa, arr, inside: bool = False) -> bool:
    """Whether an Arrow array has a null that ``_check_inner_nulls`` lets through.

    Such a null is inside a map or a list view, or in the dictionary values
    that the indices select. Unions are not looked into.
    """
    if inside and arr.null_count:
        return True
    typ = arr.type
    if not inside and not _has_unchecked_type(pa, typ):
        return False
    if pa.types.is_dictionary(typ):
        if not _has_unchecked_nulls(pa, arr.dictionary, True):
            return False
        # Only the values that the indices select count.
        children = [arr.dictionary_decode()]
    elif pa.types.is_struct(typ):
        children = arr.flatten()
    elif pa.types.is_map(typ):
        # MapArray.flatten() raises, and keys and items ignore the offset of
        # arr: these are the key-value structs of its rows.
        children = [pa.ListArray.from_arrays(arr.offsets, arr.values).flatten()]
    elif (
        pa.types.is_list(typ)
        or pa.types.is_large_list(typ)
        or pa.types.is_fixed_size_list(typ)
        or _is_unchecked(pa, typ)
    ):
        # Lists and list views: flatten() honours the offset of arr.
        children = [arr.flatten()]
    else:
        return False
    inside = inside or _is_unchecked(pa, typ)
    return any(_has_unchecked_nulls(pa, child, inside) for child in children)


def _primitive(pa, arr, copy: bool | None) -> torch.Tensor:
    """Views (or copies) a primitive Arrow array without nulls as a 1-D tensor."""
    if pa.types.is_boolean(arr.type):
        # Arrow packs booleans as bits: this copies them to bytes.
        return torch.from_numpy(arr.to_numpy(zero_copy_only=False))
    # DLPack honours arr.offset and needs no writable buffer.
    tensor = torch.from_dlpack(arr)
    buffer = arr.buffers()[1]
    # torch has no read-only tensors: an in-place write into a view of an
    # immutable (for example memory-mapped) buffer kills the process.
    if copy or (copy is None and buffer is not None and not buffer.is_mutable):
        tensor = tensor.clone()
    return tensor


def _to_tensor(pa, arr, copy, nulls, key) -> torch.Tensor:
    """Converts an Arrow array of a tensor type (see ``_is_tensor_type``) to a dense tensor."""
    typ = arr.type
    _check_nulls(arr, nulls, key)
    n = len(arr)
    if isinstance(typ, pa.FixedShapeTensorType):
        # ``shape`` is the physical shape, ``permutation`` the logical order.
        tensor = _to_tensor(pa, arr.storage, copy, nulls, key).view(n, *typ.shape)
        if typ.permutation is not None:
            tensor = tensor.permute(0, *[1 + dim for dim in typ.permutation])
        return tensor
    if pa.types.is_fixed_size_list(typ):
        # ``values`` ignores the offset and the nulls of ``arr``, which
        # ``flatten`` would drop.
        size = typ.list_size
        values = arr.values.slice(arr.offset * size, n * size)
        tensor = _to_tensor(pa, values, copy, nulls, key)
        tensor = tensor.view(n, size, *tensor.shape[1:])
    elif arr.null_count:
        # Drop the validity bitmap to view the values, then fill the nulls.
        values = pa.Array.from_buffers(
            typ, n, [None, arr.buffers()[1]], null_count=0, offset=arr.offset
        )
        tensor = _primitive(pa, values, copy)
    else:
        return _primitive(pa, arr, copy)
    if arr.null_count:
        valid = torch.from_numpy(arr.is_valid().to_numpy(zero_copy_only=False))
        # torch has no masked_fill for uint16, uint32 and uint64: fill a
        # signed view, in which 0 has the same bits.
        dtype = tensor.dtype
        tensor = (
            tensor.view(_SIGNED.get(dtype, dtype))
            .masked_fill(
                ~valid.view(n, *[1] * (tensor.ndim - 1)),
                float("nan") if tensor.is_floating_point() else 0,
            )
            .view(dtype)
        )
    return tensor


def _to_jagged(pa, arr, copy, nulls, key) -> torch.Tensor:
    """Converts a list array of a tensor type to a jagged nested tensor."""
    if arr.null_count:
        arr = arr.fill_null(pa.scalar([], type=arr.type))
    offsets = _primitive(pa, arr.offsets, copy)
    start, stop = int(offsets[0]), int(offsets[-1])
    if start:
        # A sliced array: its offsets do not start at 0.
        offsets = offsets - start
    values = arr.values.slice(start, stop - start)
    return torch.nested.nested_tensor_from_jagged(
        _to_tensor(pa, values, copy, nulls, key), offsets.to(torch.int64)
    )


def _from_array(pa, arr, copy, nulls, key, masks):
    """Converts one Arrow column, or one field of a struct column, to an entry."""
    typ = arr.type
    if pa.types.is_struct(typ):
        # flatten() honours the offset of arr, and gives the fields its nulls.
        return TensorDict(
            {
                field.name: _from_array(
                    pa, child, copy, nulls, (*key, field.name), masks
                )
                for field, child in zip(typ, arr.flatten())
            },
            batch_size=[len(arr)],
        )
    _check_nulls(arr, nulls, key)
    if arr.null_count and nulls == "mask":
        masks[key] = torch.from_numpy(arr.is_valid().to_numpy(zero_copy_only=False))
    if _is_tensor_type(pa, typ):
        return _to_tensor(pa, arr, copy, nulls, key)
    if (pa.types.is_list(typ) or pa.types.is_large_list(typ)) and _is_tensor_type(
        pa, typ.value_type
    ):
        return _to_jagged(pa, arr, copy, nulls, key)
    if nulls == "raise":
        _check_inner_nulls(pa, arr, key)
    if not len(arr):
        # NonTensorStack(batch_size=[0]) has the batch size [0, 0].
        return NonTensorStack()
    return NonTensorStack(*arr.to_pylist(), batch_size=[len(arr)])


def from_arrow(
    table: Any,
    *,
    copy: bool | None = None,
    nulls: Literal["raise", "fill", "mask"] = "raise",
    mask_key: NestedKey | None = None,
    device: torch.device | str | int | None = None,
) -> TensorDict:
    """Converts an Arrow table to a tensordict, with one entry per column, viewing the data where it can.

    Requires pyarrow 15 or later. The columns convert as follows:

    - integer and floating-point columns become tensors that view the Arrow
      data through DLPack. Boolean columns are copied, since Arrow packs them
      as bits.
    - ``list`` and ``large_list`` columns of one of these types become jagged
      nested tensors (``layout=torch.jagged``) that view the values.
    - ``fixed_size_list`` and ``arrow.fixed_shape_tensor`` columns become
      tensors with trailing dimensions.
    - ``struct`` columns become nested tensordicts.
    - all the other columns, such as strings, binary, dictionaries,
      timestamps, and lists of lists or of strings, become a
      :class:`~tensordict.NonTensorStack` of the Python objects of
      ``pyarrow.Array.to_pylist``.

    A column made of more than one chunk is first concatenated with
    ``combine_chunks``, which copies it. The tensors that view the Arrow data
    keep its buffers alive, and so the whole table that they come from.

    Args:
        table (pyarrow.Table or pyarrow.RecordBatch): the table. Other objects
            go through :func:`pyarrow.table` first: for example a Polars
            DataFrame, or any object that has an ``__arrow_c_stream__``
            method. A Hugging Face dataset holds its table in
            ``dataset.data.table``: after ``select``, ``shuffle`` or
            ``filter``, call ``dataset = dataset.flatten_indices()`` first,
            since this table holds all the rows of the original dataset.

    Keyword Args:
        copy (bool, optional): whether to copy the data. ``None`` views the
            buffers that Arrow marks as mutable, and copies the immutable
            ones, such as those of a memory-mapped file (for example a Hugging
            Face dataset loaded from disk). ``True`` copies every column.
            ``False`` views every column that it can: an in-place write into a
            view of an immutable buffer then crashes the process, since torch
            has no read-only tensors. Defaults to ``None``.
        nulls (str, optional): what to do with null values. ``"raise"``
            raises a ``ValueError`` for a null row, and for a null inside the
            rows of a list or a struct. A null inside a map or a list view,
            or in the dictionary values that a dictionary-encoded column
            selects, comes back as ``None`` with a ``FutureWarning``:
            starting with TensorDict 0.18, the default will change to raising
            a ``ValueError`` for these nulls too. Nulls inside a union or a
            run-end encoded column are not checked, and come back as
            ``None``. ``"fill"`` fills them with ``NaN`` in floating-point
            tensors, ``0`` (``False``) in the other tensors, an
            empty list in jagged tensors, and ``None`` in a
            :class:`~tensordict.NonTensorStack`. A null struct makes each of
            its fields null. ``"mask"`` fills them as ``"fill"`` does, and adds
            the boolean entry ``(*mask_key, *key)`` for each entry ``key``
            that has nulls, ``True`` where the row is not null. Filling copies
            the column. Defaults to ``"raise"``.
        mask_key (NestedKey, optional): the key of the masks that
            ``nulls="mask"`` adds. It differs from the ``"masks"`` key of
            :func:`~tensordict.pad_sequence` with ``return_mask=True``, so
            that the result can be padded with masks too. Passing it with
            another ``nulls`` raises a ``ValueError``. Defaults to
            ``"_mask"``.
        device (torch.device, str or int, optional): the device of the
            result. Defaults to ``None``, which keeps the tensors on the CPU
            and gives a tensordict without a device.

    Returns:
        A :class:`~tensordict.TensorDict` with the batch size ``[num_rows]``.

    .. seealso:: :func:`~tensordict.to_arrow`, the reverse conversion.

    Examples:
        >>> import pyarrow as pa
        >>> from tensordict import from_arrow
        >>> table = pa.table(
        ...     {
        ...         "x": pa.array([1.0, 2.0, 3.0]),
        ...         "ids": pa.array([[1, 2], [3], []]),
        ...         "text": pa.array(["a", "b", "c"]),
        ...     }
        ... )
        >>> td = from_arrow(table)
        >>> td["x"]
        tensor([1., 2., 3.], dtype=torch.float64)
        >>> td["ids"].unbind(0)
        (tensor([1, 2]), tensor([3]), tensor([], dtype=torch.int64))
        >>> td.get("text").tolist()
        ['a', 'b', 'c']
        >>> td = from_arrow(pa.table({"y": pa.array([1.0, None])}), nulls="mask")
        >>> td["y"], td["_mask", "y"]
        (tensor([1., nan], dtype=torch.float64), tensor([ True, False]))
    """
    pa = _import_pyarrow()
    _check_pyarrow_version(pa, "from_arrow")
    if nulls not in _NULLS:
        raise ValueError(f"nulls must be one of {_NULLS}, got {nulls!r}.")
    if mask_key is not None and nulls != "mask":
        raise ValueError(f"from_arrow got a mask_key but nulls={nulls!r}, not 'mask'.")
    mask_key = unravel_key("_mask" if mask_key is None else mask_key)
    first = mask_key if isinstance(mask_key, str) else mask_key[0]
    if not isinstance(table, (pa.Table, pa.RecordBatch)):
        table = pa.table(table)
    if nulls == "mask" and first in table.column_names:
        raise ValueError(
            f"nulls='mask' writes the masks under the key {first!r}, which is a column "
            "of the table."
        )
    masks = {}
    data = {}
    unchecked_nulls = False
    for name, column in zip(table.column_names, table.columns):
        if isinstance(column, pa.ChunkedArray):
            column = (
                column.chunk(0) if column.num_chunks == 1 else column.combine_chunks()
            )
        data[name] = _from_array(pa, column, copy, nulls, (name,), masks)
        unchecked_nulls = unchecked_nulls or (
            nulls == "raise"
            # Cheap gate, so numeric columns skip the walk over their type.
            and (pa.types.is_nested(column.type) or pa.types.is_dictionary(column.type))
            and _has_unchecked_nulls(pa, column)
        )
    if unchecked_nulls:
        # #3312: warn once all the columns converted, so a call that raises does not warn.
        warnings.warn(
            "from_arrow got a null inside a map, a list view or the values of a "
            "dictionary-encoded column, and returns it as None. Starting with "
            "TensorDict 0.18, the default will change to raising a ValueError for "
            "these nulls, as for nulls inside lists. Pass nulls='fill' to keep "
            "this result without the warning. nulls='mask' gives the same result: "
            "its masks only mark null rows.",
            FutureWarning,
            stacklevel=_user_stacklevel(),
        )
    result = TensorDict(data, batch_size=[table.num_rows])
    for key, mask in masks.items():
        result.set(unravel_key((mask_key, *key)), mask)
    if device is not None:
        result = result.to(device)
    return result


def _to_array(pa, value, key):
    """Converts the entry ``key`` of a tensordict to an Arrow array."""
    if is_non_tensor(value):
        return pa.array(value.tolist())
    if is_tensor_collection(value):
        items = list(value.items())
        if not items:
            return pa.array([{}] * value.shape[0], type=pa.struct([]))
        return pa.StructArray.from_arrays(
            [_to_array(pa, item, unravel_key((key, k))) for k, item in items],
            names=[k for k, _ in items],
        )
    tensor = value.detach().cpu().contiguous()
    if tensor.is_nested:
        if tensor.layout is not torch.jagged:
            raise ValueError(
                "to_arrow converts the nested tensors of layout torch.jagged only, "
                f"got {tensor.layout}."
            )
        if tensor._ragged_idx != 1:
            raise ValueError(
                f"to_arrow needs the ragged dim of the jagged tensor {key!r} at "
                f"position 1, got {tensor._ragged_idx}. Move it there first, for "
                f"example with .transpose(1, {tensor._ragged_idx})."
            )
        return pa.LargeListArray.from_arrays(
            pa.array(tensor.offsets().numpy()), _to_array(pa, tensor.values(), key)
        )
    if tensor.ndim == 1:
        return pa.array(tensor.numpy())
    if not len(tensor):
        # from_numpy_ndarray raises for an empty array.
        typ = pa.fixed_shape_tensor(
            pa.from_numpy_dtype(tensor.numpy().dtype), tensor.shape[1:]
        )
        return pa.ExtensionArray.from_storage(typ, pa.array([], typ.storage_type))
    return pa.FixedShapeTensorArray.from_numpy_ndarray(tensor.numpy())


def to_arrow(td: TensorDictBase) -> pyarrow.Table:
    """Converts a tensordict with one batch dimension to an Arrow table, with one column per entry.

    Requires pyarrow 15 or later. This is the reverse of
    :func:`~tensordict.from_arrow`. The entries convert as follows:

    - 1-D tensors become primitive columns. The columns of contiguous numeric
      tensors on the CPU view their data. Boolean tensors are copied.
    - tensors with more dimensions become ``arrow.fixed_shape_tensor`` columns.
    - jagged nested tensors (``layout=torch.jagged``) with their ragged dim
      at position 1 become ``large_list`` columns; another ragged dim raises a
      ``ValueError``.
    - nested tensordicts become ``struct`` columns.
    - non-tensor entries, such as a :class:`~tensordict.NonTensorStack` of
      strings, become columns of the type that :func:`pyarrow.array` infers
      from their values.

    A :class:`~tensordict.LazyStackedTensorDict` with entries of different
    shapes converts after
    ``densify(layout=torch.jagged)``.

    Args:
        td (TensorDictBase): the tensordict. Its batch size must have one
            dimension: call ``td.reshape(-1)`` first to flatten a batch with
            more dimensions.

    Returns:
        A ``pyarrow.Table`` with ``td.batch_size[0]`` rows, or no rows if
        ``td`` has no entries.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict, to_arrow
        >>> td = TensorDict(
        ...     x=torch.tensor([1.0, 2.0]),
        ...     image=torch.zeros(2, 3, 4),
        ...     obs=TensorDict(y=torch.tensor([3, 4]), batch_size=[2]),
        ...     batch_size=[2],
        ... )
        >>> to_arrow(td).schema
        x: float
        image: extension<arrow.fixed_shape_tensor[value_type=float, shape=[3,4], permutation=[0,1]]>
        obs: struct<y: int64>
          child 0, y: int64
    """
    pa = _import_pyarrow()
    _check_pyarrow_version(pa, "to_arrow")
    if td.batch_dims != 1:
        raise ValueError(
            "to_arrow needs a tensordict with one batch dimension, got the batch "
            f"size {td.batch_size}. Call td.reshape(-1) first to flatten it."
        )
    return pa.table({key: _to_array(pa, value, key) for key, value in td.items()})

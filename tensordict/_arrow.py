# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Conversion between Arrow tables and tensordicts, with pyarrow as an optional dependency."""

from __future__ import annotations

import importlib.util
from typing import Any, Literal, TYPE_CHECKING

import torch
from tensordict._td import TensorDict
from tensordict.base import is_tensor_collection, TensorDictBase
from tensordict.tensorclass import NonTensorStack
from tensordict.utils import is_non_tensor

if TYPE_CHECKING:
    import pyarrow

_has_pyarrow = importlib.util.find_spec("pyarrow") is not None

__all__ = ["from_arrow", "to_arrow"]

_NULLS = ("raise", "fill", "mask")


def _import_pyarrow():
    if not _has_pyarrow:
        raise ImportError(
            "from_arrow and to_arrow need pyarrow 15 or later: pip install pyarrow"
        )
    import pyarrow as pa

    return pa


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
        tensor = tensor.masked_fill(
            ~valid.view(n, *[1] * (tensor.ndim - 1)),
            float("nan") if tensor.is_floating_point() else 0,
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
    return NonTensorStack(*arr.to_pylist(), batch_size=[len(arr)])


def from_arrow(
    table: Any,
    *,
    copy: bool | None = None,
    nulls: Literal["raise", "fill", "mask"] = "raise",
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
            ``dataset.data.table``.

    Keyword Args:
        copy (bool, optional): whether to copy the data. ``None`` views the
            buffers that Arrow marks as mutable, and copies the immutable
            ones, such as those of a memory-mapped file (for example a Hugging
            Face dataset loaded from disk). ``True`` copies every column.
            ``False`` views every column that it can: an in-place write into a
            view of an immutable buffer then crashes the process, since torch
            has no read-only tensors. Defaults to ``None``.
        nulls (str, optional): what to do with null values. ``"raise"``
            raises a ``ValueError``. ``"fill"`` fills them with ``NaN`` in
            floating-point tensors, ``0`` (``False``) in the other tensors, an
            empty list in jagged tensors, and ``None`` in a
            :class:`~tensordict.NonTensorStack`. A null struct makes each of
            its fields null. ``"mask"`` fills them as ``"fill"`` does, and adds
            the boolean entry ``("_mask", *key)`` for each entry ``key`` that
            has nulls, ``True`` where the row is not null. Filling copies the
            column. Defaults to ``"raise"``.
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
    if nulls not in _NULLS:
        raise ValueError(f"nulls must be one of {_NULLS}, got {nulls!r}.")
    if not isinstance(table, (pa.Table, pa.RecordBatch)):
        table = pa.table(table)
    if nulls == "mask" and "_mask" in table.column_names:
        raise ValueError(
            "nulls='mask' writes the masks under the key '_mask', which is a column "
            "of the table."
        )
    masks = {}
    data = {}
    for name, column in zip(table.column_names, table.columns):
        if isinstance(column, pa.ChunkedArray):
            column = (
                column.chunk(0) if column.num_chunks == 1 else column.combine_chunks()
            )
        data[name] = _from_array(pa, column, copy, nulls, (name,), masks)
    result = TensorDict(data, batch_size=[table.num_rows])
    for key, mask in masks.items():
        result.set(("_mask", *key), mask)
    if device is not None:
        result = result.to(device)
    return result


def _to_array(pa, value):
    """Converts one entry of a tensordict to an Arrow array."""
    if is_non_tensor(value):
        return pa.array(value.tolist())
    if is_tensor_collection(value):
        keys = list(value.keys())
        if not keys:
            return pa.array([{}] * value.shape[0], type=pa.struct([]))
        return pa.StructArray.from_arrays(
            [_to_array(pa, value.get(key)) for key in keys], names=keys
        )
    tensor = value.detach().cpu().contiguous()
    if tensor.is_nested:
        if tensor.layout is not torch.jagged:
            raise ValueError(
                "to_arrow converts the nested tensors of layout torch.jagged only, "
                f"got {tensor.layout}."
            )
        return pa.LargeListArray.from_arrays(
            pa.array(tensor.offsets().numpy()), _to_array(pa, tensor.values())
        )
    if tensor.ndim == 1:
        return pa.array(tensor.numpy())
    return pa.FixedShapeTensorArray.from_numpy_ndarray(tensor.numpy())


def to_arrow(td: TensorDictBase) -> pyarrow.Table:
    """Converts a tensordict with one batch dimension to an Arrow table, with one column per entry.

    Requires pyarrow 15 or later. This is the reverse of
    :func:`~tensordict.from_arrow`. The entries convert as follows:

    - 1-D tensors become primitive columns. The columns of contiguous numeric
      tensors on the CPU view their data. Boolean tensors are copied.
    - tensors with more dimensions become ``arrow.fixed_shape_tensor`` columns.
    - jagged nested tensors (``layout=torch.jagged``) become ``large_list``
      columns.
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
        A ``pyarrow.Table`` with ``td.batch_size[0]`` rows.

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
    if td.batch_dims != 1:
        raise ValueError(
            "to_arrow needs a tensordict with one batch dimension, got the batch "
            f"size {td.batch_size}. Call td.reshape(-1) first to flatten it."
        )
    return pa.table({key: _to_array(pa, value) for key, value in td.items()})

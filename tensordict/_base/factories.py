# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Sequence

import torch
from tensordict.base import TensorDictBase

__all__ = [
    "from_any",
    "from_csv",
    "from_dict",
    "from_h5",
    "from_json",
    "from_namedtuple",
    "from_pandas",
    "from_parquet",
    "from_struct_array",
    "from_tuple",
    "from_zarr",
]


def from_any(
    obj,
    *,
    auto_batch_size: bool = False,
    batch_dims: int | None = None,
    device: torch.device | None = None,
    batch_size: torch.Size | None = None,
):
    """Converts a dict, tuple, dataclass, array or file to a tensordict, recursively.

    The conversion depends on the type of ``obj``: see
    :meth:`TensorDictBase.from_any <tensordict.TensorDictBase.from_any>` for
    the list. A tensor or a tensor collection is returned as it is.

    Args:
        obj: the object to convert.

    Keyword Args:
        auto_batch_size (bool, optional): if ``True``, sets the batch size to
            the largest leading shape that all the entries share. Exclusive
            with ``batch_size``, except for a dict, where ``batch_size``
            wins, and for an :class:`h5py.File`, where ``auto_batch_size``
            wins. Defaults to ``False``.
        batch_dims (int, optional): with ``auto_batch_size``, the maximum
            number of batch dimensions. Ignored without
            ``auto_batch_size``, with a ``FutureWarning`` when it would
            change the result. Starting with TensorDict 0.17, the default
            will change to turning on ``auto_batch_size`` when
            ``batch_dims`` is given, as ``from_dict`` does. Defaults to
            ``None``.
        device (torch.device, str or int, optional): the device of the
            result. Leaves on another device are moved to it. ``None`` leaves
            each leaf where it is, and the result has no device. Defaults to
            ``None``.
        batch_size (torch.Size or list of int, optional): the batch size of
            the result: the leading dimensions that all its entries share.
            Exclusive with ``auto_batch_size``, except for a dict, where
            ``batch_size`` wins and no batch size is computed, and for an
            :class:`h5py.File`, where ``auto_batch_size`` wins and
            ``batch_size`` is ignored. Defaults to ``None``.

    Returns:
        A tensordict (a :class:`~tensordict.PersistentTensorDict` for an HDF5
        file), a tensor or a :class:`~tensordict.NonTensorStack` for a list,
        or ``obj`` itself for the other types.

    Examples:
        >>> import torch
        >>> from tensordict import from_any
        >>> td = from_any({"obs": torch.zeros(3, 4), "pair": (torch.zeros(3), 1)})
        >>> sorted(td.keys(include_nested=True, leaves_only=True), key=str)
        [('pair', '0'), ('pair', '1'), 'obs']
    """
    from tensordict import TensorDict

    return TensorDict.from_any(
        obj,
        auto_batch_size=auto_batch_size,
        batch_dims=batch_dims,
        device=device,
        batch_size=batch_size,
    )


def from_tuple(
    obj,
    *,
    auto_batch_size: bool = False,
    batch_dims: int | None = None,
    device: torch.device | None = None,
    batch_size: torch.Size | None = None,
) -> "TensorDictBase":
    """Converts a tuple to a tensordict with the keys ``"0"``, ``"1"``, ...

    Each element is converted with :func:`from_any`. See
    :meth:`TensorDictBase.from_tuple <tensordict.TensorDictBase.from_tuple>`.

    Args:
        obj (tuple): the tuple to convert.

    Keyword Args:
        auto_batch_size (bool, optional): if ``True``, sets the batch size to
            the largest leading shape that all the entries share. Exclusive
            with ``batch_size``. Defaults to ``False``.
        batch_dims (int, optional): with ``auto_batch_size``, the maximum
            number of batch dimensions. Ignored without
            ``auto_batch_size``, with a ``FutureWarning`` when it would
            change the result. Starting with TensorDict 0.17, the default
            will change to turning on ``auto_batch_size`` when
            ``batch_dims`` is given, as ``from_dict`` does. Defaults to
            ``None``.
        device (torch.device, str or int, optional): the device of the
            result. Leaves on another device are moved to it. ``None`` leaves
            each leaf where it is, and the result has no device. Defaults to
            ``None``.
        batch_size (torch.Size or list of int, optional): the batch size of
            the result: the leading dimensions that all its entries share.
            Exclusive with ``auto_batch_size``. Defaults to ``None``, an empty
            batch size.

    Returns:
        A :class:`~tensordict.TensorDict`.

    Examples:
        >>> import torch
        >>> from tensordict import from_tuple
        >>> td = from_tuple((torch.zeros(3, 4), torch.ones(3)), auto_batch_size=True)
        >>> list(td.keys()), td.batch_size
        (['0', '1'], torch.Size([3]))
    """
    from tensordict import TensorDict

    return TensorDict.from_tuple(
        obj,
        auto_batch_size=auto_batch_size,
        batch_dims=batch_dims,
        device=device,
        batch_size=batch_size,
    )


def from_namedtuple(
    named_tuple,
    *,
    auto_batch_size: bool = False,
    batch_dims: int | None = None,
    device: torch.device | None = None,
    batch_size: torch.Size | None = None,
) -> "TensorDictBase":
    """Converts a named tuple to a tensordict whose keys are its field names.

    Each field value is converted with :func:`from_any`. See
    :meth:`TensorDictBase.from_namedtuple
    <tensordict.TensorDictBase.from_namedtuple>`.

    Args:
        named_tuple (NamedTuple): the named tuple to convert.

    Keyword Args:
        auto_batch_size (bool, optional): if ``True``, sets the batch size to
            the largest leading shape that all the entries share. Exclusive
            with ``batch_size``. Defaults to ``False``.
        batch_dims (int, optional): with ``auto_batch_size``, the maximum
            number of batch dimensions. Ignored without
            ``auto_batch_size``, with a ``FutureWarning`` when it would
            change the result. Starting with TensorDict 0.17, the default
            will change to turning on ``auto_batch_size`` when
            ``batch_dims`` is given, as ``from_dict`` does. Defaults to
            ``None``.
        device (torch.device, str or int, optional): the device of the
            result. Leaves on another device are moved to it. ``None`` leaves
            each leaf where it is, and the result has no device. Defaults to
            ``None``.
        batch_size (torch.Size or list of int, optional): the batch size of
            the result: the leading dimensions that all its entries share.
            Exclusive with ``auto_batch_size``. Defaults to ``None``, an empty
            batch size.

    Returns:
        A :class:`~tensordict.TensorDict`.

    Examples:
        >>> import torch
        >>> from tensordict import from_namedtuple
        >>> td = from_namedtuple(torch.tensor([[2, 0, 1]]).max(dim=1))
        >>> td["values"], td["indices"]
        (tensor([2]), tensor([0]))
    """
    from tensordict import TensorDict

    return TensorDict.from_namedtuple(
        named_tuple,
        auto_batch_size=auto_batch_size,
        batch_dims=batch_dims,
        device=device,
        batch_size=batch_size,
    )


def from_struct_array(
    struct_array,
    *,
    auto_batch_size: bool = False,
    batch_dims: int | None = None,
    device: torch.device | None = None,
    batch_size: torch.Size | None = None,
) -> "TensorDictBase":
    """Converts a NumPy structured array to a tensordict, with one entry per field.

    The tensors share memory with the array. See
    :meth:`TensorDictBase.from_struct_array
    <tensordict.TensorDictBase.from_struct_array>`.

    Args:
        struct_array (np.ndarray): the structured array to convert.

    Keyword Args:
        auto_batch_size (bool, optional): if ``True``, sets the batch size to
            the largest leading shape that all the entries share. Exclusive
            with ``batch_size``. Defaults to ``False``.
        batch_dims (int, optional): with ``auto_batch_size``, the maximum
            number of batch dimensions. Ignored without
            ``auto_batch_size``, with a ``FutureWarning`` when it would
            change the result. Starting with TensorDict 0.17, the default
            will change to turning on ``auto_batch_size`` when
            ``batch_dims`` is given, as ``from_dict`` does. Defaults to
            ``None``.
        device (torch.device, str or int, optional): the device of the
            result. Leaves on another device are moved to it. ``None`` leaves
            each leaf where it is, and the result has no device. Defaults to
            ``None``.
        batch_size (torch.Size or list of int, optional): the batch size of
            the result: the leading dimensions that all its entries share.
            Exclusive with ``auto_batch_size``. Defaults to ``None``, which
            uses the shape of the array.

    Returns:
        A :class:`~tensordict.TensorDict`.

    Examples:
        >>> import numpy as np
        >>> from tensordict import from_struct_array
        >>> x = np.array(
        ...     [("Rex", 9, 81.0), ("Fido", 3, 27.0)],
        ...     dtype=[("name", "U10"), ("age", "i4"), ("weight", "f4")],
        ... )
        >>> td = from_struct_array(x)
        >>> td.batch_size, td["age"]
        (torch.Size([2]), tensor([9, 3], dtype=torch.int32))
        >>> x["age"] += 1  # the tensordict shares memory with the array
        >>> td["age"]
        tensor([10,  4], dtype=torch.int32)
        >>> bool((td.to_struct_array() == x).all())
        True

    """
    return TensorDictBase.from_struct_array(
        struct_array,
        auto_batch_size=auto_batch_size,
        batch_dims=batch_dims,
        device=device,
        batch_size=batch_size,
    )


def from_dict(
    d,
    *,
    auto_batch_size: bool | None = None,
    batch_dims: int | None = None,
    device: torch.device | None = None,
    batch_size: torch.Size | None = None,
    names: Sequence[str] | None = None,
) -> "TensorDictBase":
    """Converts a dictionary to a tensordict, recursively.

    Nested dicts become nested tensordicts, and the values are converted with
    :func:`from_any`. Unlike the :class:`~tensordict.TensorDict` constructor,
    it can compute the batch size. See
    :meth:`TensorDict.from_dict <tensordict.TensorDict.from_dict>`.

    Args:
        d (dict): the dictionary to convert.

    Keyword Args:
        auto_batch_size (bool, optional): if ``True``, sets the batch size to
            the largest leading shape that all the entries share, for the
            result and for each nested tensordict. Defaults to ``None``,
            which behaves like ``False`` unless ``batch_dims`` is given.
        batch_dims (int, optional): with ``auto_batch_size``, the maximum
            number of batch dimensions. Turns on ``auto_batch_size`` when that
            is ``None``. Exclusive with ``batch_size``. Defaults to ``None``.
        device (torch.device, str or int, optional): the device of the
            result. Leaves on another device are moved to it. ``None`` leaves
            each leaf where it is, and the result has no device. Defaults to
            ``None``.
        batch_size (torch.Size or list of int, optional): the batch size of
            the result: the leading dimensions that all its entries share.
            Takes precedence over ``auto_batch_size``. Defaults to ``None``,
            an empty batch size unless it is computed.
        names (list of str, optional): the names of the batch dimensions, one
            per dimension. Used only with ``batch_size``: when the batch size
            is empty or computed, the names are dropped. Defaults to ``None``.

    Returns:
        A :class:`~tensordict.TensorDict`.

    Raises:
        ValueError: if both ``batch_size`` and ``batch_dims`` are given.

    Examples:
        >>> import torch
        >>> from tensordict import from_dict
        >>> input_dict = {"a": torch.zeros(3, 4), "b": {"c": torch.zeros(3, 4, 5)}}
        >>> from_dict(input_dict).batch_size
        torch.Size([])
        >>> td = from_dict(input_dict, auto_batch_size=True)
        >>> td.batch_size, td["b"].batch_size
        (torch.Size([3, 4]), torch.Size([3, 4, 5]))
        >>> from_dict(input_dict, batch_dims=1).batch_size
        torch.Size([3])
        >>> from_dict(input_dict, batch_size=[3], names=["n"]).names
        ['n']

    """
    from tensordict import TensorDict

    return TensorDict.from_dict(
        d,
        auto_batch_size=auto_batch_size,
        batch_dims=batch_dims,
        device=device,
        batch_size=batch_size,
        names=names,
    )


def from_h5(
    h5_file,
    *,
    mode: str = "r",
    auto_batch_size: bool = False,
    batch_dims: int | None = None,
    batch_size: torch.Size | None = None,
    non_tensor_as_stack: bool = False,
) -> "TensorDictBase":
    """Opens an HDF5 file as a :class:`~tensordict.PersistentTensorDict`.

    See :meth:`TensorDictBase.from_h5 <tensordict.TensorDictBase.from_h5>`.

    Args:
        h5_file (str or path): the path to the HDF5 file.

    Keyword Args:
        mode (str, optional): how the file is opened, as in
            :class:`h5py.File`: ``"r"`` (read only), ``"r+"`` (read and
            write), and so on. Defaults to ``"r"``.
        auto_batch_size (bool, optional): if ``True``, sets the batch size to
            the largest leading shape that all the entries share. Exclusive
            with ``batch_size``. Defaults to ``False``.
        batch_dims (int, optional): with ``auto_batch_size``, the maximum
            number of batch dimensions. Ignored without
            ``auto_batch_size``, with a ``FutureWarning`` when it would
            change the result. Starting with TensorDict 0.17, the default
            will change to turning on ``auto_batch_size`` when
            ``batch_dims`` is given, as ``from_dict`` does. Defaults to
            ``None``.
        batch_size (torch.Size or list of int, optional): the batch size of
            the result: the leading dimensions that all its entries share.
            Exclusive with ``auto_batch_size``. Defaults to ``None``, which
            gives each nested tensordict the largest batch size that its
            entries share.
        non_tensor_as_stack (bool, optional): if ``True``, a non-tensor
            dataset that tensordict did not write reads as a
            :class:`~tensordict.NonTensorStack` of its rows when its leading
            dimensions equal the batch size. Defaults to ``False``.

    Returns:
        A :class:`~tensordict.PersistentTensorDict` backed by the file.
    """
    from tensordict import TensorDict

    return TensorDict.from_h5(
        h5_file,
        mode=mode,
        auto_batch_size=auto_batch_size,
        batch_dims=batch_dims,
        batch_size=batch_size,
        non_tensor_as_stack=non_tensor_as_stack,
    )


def from_zarr(
    zarr_file,
    *,
    mode: str = "r",
    auto_batch_size: bool = False,
    batch_dims: int | None = None,
    batch_size: torch.Size | None = None,
    non_tensor_as_stack: bool = False,
    allow_pickle: bool | None = None,
) -> "TensorDictBase":
    """Opens a zarr store as a :class:`~tensordict.PersistentTensorDict`.

    Requires ``zarr>=3.1.1``. See
    :meth:`TensorDictBase.from_zarr <tensordict.TensorDictBase.from_zarr>`.

    Args:
        zarr_file (str, path or zarr store): the path to the zarr store (a
            directory), or a ``zarr.abc.store.Store`` instance.

    Keyword Args:
        mode (str, optional): how the store is opened: ``"r"`` (read only),
            ``"r+"`` (read and write), and so on. Defaults to ``"r"``.
        auto_batch_size (bool, optional): if ``True``, sets the batch size to
            the largest leading shape that all the entries share. Exclusive
            with ``batch_size``. Defaults to ``False``.
        batch_dims (int, optional): with ``auto_batch_size``, the maximum
            number of batch dimensions. Ignored without
            ``auto_batch_size``, with a ``FutureWarning`` when it would
            change the result. Starting with TensorDict 0.17, the default
            will change to turning on ``auto_batch_size`` when
            ``batch_dims`` is given, as ``from_dict`` does. Defaults to
            ``None``.
        batch_size (torch.Size or list of int, optional): the batch size of
            the result: the leading dimensions that all its entries share.
            Exclusive with ``auto_batch_size``. Defaults to ``None``, which
            reads the batch size that :meth:`~tensordict.TensorDictBase.to_zarr`
            stored, or computes it for a store written by another tool.
        non_tensor_as_stack (bool, optional): if ``True``, a non-tensor
            array that tensordict did not write reads as a
            :class:`~tensordict.NonTensorStack` of its rows when its leading
            dimensions equal the batch size. Defaults to ``False``.
        allow_pickle (bool, optional): whether to decode pickled non-tensor
            values. ``False`` rejects them. ``True`` permits them, only for
            data you trust. ``None`` permits them with a ``FutureWarning``.
            Starting with TensorDict 0.17, the default will change to
            ``allow_pickle=False``. Defaults to ``None``.

    Returns:
        A :class:`~tensordict.PersistentTensorDict` backed by the store.
    """
    from tensordict import TensorDict

    return TensorDict.from_zarr(
        zarr_file,
        mode=mode,
        auto_batch_size=auto_batch_size,
        batch_dims=batch_dims,
        batch_size=batch_size,
        non_tensor_as_stack=non_tensor_as_stack,
        allow_pickle=allow_pickle,
    )


def from_pandas(
    dataframe,
    *,
    auto_batch_size: bool = False,
    batch_dims: int | None = None,
    device: torch.device | None = None,
    batch_size: torch.Size | None = None,
    separator: str | None = None,
    dtype: torch.dtype | None = None,
) -> "TensorDictBase":
    """Converts a pandas DataFrame to a tensordict, with one entry per column.

    See :meth:`TensorDictBase.from_pandas
    <tensordict.TensorDictBase.from_pandas>` for how each column type is
    converted.

    Args:
        dataframe (pd.DataFrame): the DataFrame to convert.

    Keyword Args:
        auto_batch_size (bool, optional): if ``True``, sets the batch size to
            the largest leading shape that all the entries share. Exclusive
            with ``batch_size``. Defaults to ``False``.
        batch_dims (int, optional): with ``auto_batch_size``, the maximum
            number of batch dimensions. Ignored without
            ``auto_batch_size``, with a ``FutureWarning`` when it would
            change the result. Starting with TensorDict 0.17, the default
            will change to turning on ``auto_batch_size`` when
            ``batch_dims`` is given, as ``from_dict`` does. Defaults to
            ``None``.
        device (torch.device, str or int, optional): the device of the
            result. Leaves on another device are moved to it. ``None`` leaves
            each leaf where it is, and the result has no device. Defaults to
            ``None``.
        batch_size (torch.Size or list of int, optional): the batch size of
            the result: the leading dimensions that all its entries share.
            Exclusive with ``auto_batch_size``. Defaults to ``None``, which
            gives ``[num_rows]``.
        separator (str, optional): if given, splits the column names on it
            to build nested entries: with ``separator="."``, the column
            ``"obs.x"`` becomes the entry ``("obs", "x")``. Defaults to
            ``None``.
        dtype (torch.dtype, optional): if given, casts the numeric and
            boolean columns to it. Defaults to ``None``.

    Returns:
        A :class:`~tensordict.TensorDict`.

    Examples:
        >>> import pandas as pd
        >>> from tensordict import from_pandas
        >>> df = pd.DataFrame({"obs.x": [1.0, 2.0], "obs.y": [3.0, 4.0], "done": [False, True]})
        >>> td = from_pandas(df, separator=".")
        >>> td["obs", "x"], td.batch_size
        (tensor([1., 2.], dtype=torch.float64), torch.Size([2]))
    """
    return TensorDictBase.from_pandas(
        dataframe,
        auto_batch_size=auto_batch_size,
        batch_dims=batch_dims,
        device=device,
        batch_size=batch_size,
        separator=separator,
        dtype=dtype,
    )


def from_csv(
    path,
    *,
    auto_batch_size: bool = False,
    batch_dims: int | None = None,
    device: torch.device | None = None,
    batch_size: torch.Size | None = None,
    separator: str | None = None,
    dtype: torch.dtype | None = None,
    **kwargs,
) -> "TensorDictBase":
    """Reads a CSV file into a tensordict, with one entry per column.

    Requires pandas or pyarrow. See :meth:`TensorDictBase.from_csv
    <tensordict.TensorDictBase.from_csv>`.

    Args:
        path (str or Path): the path of the CSV file.

    Keyword Args:
        auto_batch_size (bool, optional): if ``True``, sets the batch size to
            the largest leading shape that all the entries share. Exclusive
            with ``batch_size``. Defaults to ``False``.
        batch_dims (int, optional): with ``auto_batch_size``, the maximum
            number of batch dimensions. Ignored without
            ``auto_batch_size``, with a ``FutureWarning`` when it would
            change the result. Starting with TensorDict 0.17, the default
            will change to turning on ``auto_batch_size`` when
            ``batch_dims`` is given, as ``from_dict`` does. Defaults to
            ``None``.
        device (torch.device, str or int, optional): the device of the
            result. Leaves on another device are moved to it. ``None`` leaves
            each leaf where it is, and the result has no device. Defaults to
            ``None``.
        batch_size (torch.Size or list of int, optional): the batch size of
            the result: the leading dimensions that all its entries share.
            Exclusive with ``auto_batch_size``. Defaults to ``None``, which
            gives ``[num_rows]``.
        separator (str, optional): if given, splits the column names on it
            to build nested entries: with ``separator="."``, the column
            ``"obs.x"`` becomes the entry ``("obs", "x")``. Defaults to
            ``None``.
        dtype (torch.dtype, optional): if given, casts the numeric and
            boolean columns to it. Defaults to ``None``.
        **kwargs: passed to the CSV reader.

    Returns:
        A :class:`~tensordict.TensorDict`.
    """
    return TensorDictBase.from_csv(
        path,
        auto_batch_size=auto_batch_size,
        batch_dims=batch_dims,
        device=device,
        batch_size=batch_size,
        separator=separator,
        dtype=dtype,
        **kwargs,
    )


def from_parquet(
    path,
    *,
    auto_batch_size: bool = False,
    batch_dims: int | None = None,
    device: torch.device | None = None,
    batch_size: torch.Size | None = None,
    separator: str | None = None,
    dtype: torch.dtype | None = None,
    columns: list[str] | None = None,
    **kwargs,
) -> "TensorDictBase":
    """Reads a Parquet file into a tensordict, with one entry per column.

    Requires pyarrow or pandas. See :meth:`TensorDictBase.from_parquet
    <tensordict.TensorDictBase.from_parquet>`.

    Args:
        path (str or Path): the path of the Parquet file.

    Keyword Args:
        auto_batch_size (bool, optional): if ``True``, sets the batch size to
            the largest leading shape that all the entries share. Exclusive
            with ``batch_size``. Defaults to ``False``.
        batch_dims (int, optional): with ``auto_batch_size``, the maximum
            number of batch dimensions. Ignored without
            ``auto_batch_size``, with a ``FutureWarning`` when it would
            change the result. Starting with TensorDict 0.17, the default
            will change to turning on ``auto_batch_size`` when
            ``batch_dims`` is given, as ``from_dict`` does. Defaults to
            ``None``.
        device (torch.device, str or int, optional): the device of the
            result. Leaves on another device are moved to it. ``None`` leaves
            each leaf where it is, and the result has no device. Defaults to
            ``None``.
        batch_size (torch.Size or list of int, optional): the batch size of
            the result: the leading dimensions that all its entries share.
            Exclusive with ``auto_batch_size``. Defaults to ``None``, which
            gives ``[num_rows]``.
        separator (str, optional): if given, splits the column names on it
            to build nested entries: with ``separator="."``, the column
            ``"obs.x"`` becomes the entry ``("obs", "x")``. Defaults to
            ``None``.
        dtype (torch.dtype, optional): if given, casts the numeric and
            boolean columns to it. Defaults to ``None``.
        columns (list of str, optional): the columns to read. Defaults to
            ``None``, which reads all the columns.
        **kwargs: passed to the Parquet reader.

    Returns:
        A :class:`~tensordict.TensorDict`.
    """
    return TensorDictBase.from_parquet(
        path,
        auto_batch_size=auto_batch_size,
        batch_dims=batch_dims,
        device=device,
        batch_size=batch_size,
        separator=separator,
        dtype=dtype,
        columns=columns,
        **kwargs,
    )


def from_json(
    path,
    *,
    auto_batch_size: bool = False,
    batch_dims: int | None = None,
    device: torch.device | None = None,
    batch_size: torch.Size | None = None,
    separator: str | None = None,
    dtype: torch.dtype | None = None,
    lines: bool = False,
    **kwargs,
) -> "TensorDictBase":
    """Reads a JSON file of records into a tensordict, with one entry per field.

    The file holds an array of records, or one record per line with
    ``lines=True``. See :meth:`TensorDictBase.from_json
    <tensordict.TensorDictBase.from_json>`.

    Args:
        path (str or Path): the path of the JSON file.

    Keyword Args:
        auto_batch_size (bool, optional): if ``True``, sets the batch size to
            the largest leading shape that all the entries share. Exclusive
            with ``batch_size``. Defaults to ``False``.
        batch_dims (int, optional): with ``auto_batch_size``, the maximum
            number of batch dimensions. Ignored without
            ``auto_batch_size``, with a ``FutureWarning`` when it would
            change the result. Starting with TensorDict 0.17, the default
            will change to turning on ``auto_batch_size`` when
            ``batch_dims`` is given, as ``from_dict`` does. Defaults to
            ``None``.
        device (torch.device, str or int, optional): the device of the
            result. Leaves on another device are moved to it. ``None`` leaves
            each leaf where it is, and the result has no device. Defaults to
            ``None``.
        batch_size (torch.Size or list of int, optional): the batch size of
            the result: the leading dimensions that all its entries share.
            Exclusive with ``auto_batch_size``. Defaults to ``None``, which
            gives ``[num_rows]``.
        separator (str, optional): if given, splits the column names on it
            to build nested entries: with ``separator="."``, the column
            ``"obs.x"`` becomes the entry ``("obs", "x")``. Defaults to
            ``None``.
        dtype (torch.dtype, optional): if given, casts the numeric and
            boolean columns to it. Defaults to ``None``.
        lines (bool, optional): if ``True``, reads the file as JSON Lines,
            one record per line. Defaults to ``False``.
        **kwargs: passed to :func:`pandas.read_json`. Ignored without pandas.

    Returns:
        A :class:`~tensordict.TensorDict`.
    """
    return TensorDictBase.from_json(
        path,
        auto_batch_size=auto_batch_size,
        batch_dims=batch_dims,
        device=device,
        batch_size=batch_size,
        separator=separator,
        dtype=dtype,
        lines=lines,
        **kwargs,
    )


for _name in __all__:
    globals()[_name].__module__ = "tensordict.base"

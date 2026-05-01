# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping
from typing import Type

import torch
from tensordict._tensorcollection import TensorCollection
from tensordict.base import TensorDictBase

__all__ = [
    "from_any",
    "from_csv",
    "from_dict",
    "from_h5",
    "from_json",
    "from_list",
    "from_namedtuple",
    "from_pandas",
    "from_parquet",
    "from_struct_array",
    "from_tuple",
]


def from_any(
    obj,
    *,
    auto_batch_size: bool = False,
    batch_dims: int | None = None,
    device: torch.device | None = None,
    batch_size: torch.Size | None = None,
):
    """Converts any object to a TensorDict.

    .. seealso:: :meth:`~tensordict.TensorDictBase.from_any` for more information.
    """
    return TensorDictBase.from_any(
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
    """Converts a tuple to a TensorDict.

    .. seealso:: :meth:`TensorDictBase.from_tuple` for more information.
    """
    return TensorDictBase.from_tuple(
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
    """Converts a namedtuple to a TensorDict.

    .. seealso:: :meth:`TensorDictBase.from_namedtuple` for more information.
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
    """Converts a structured numpy array to a TensorDict.

    .. seealso:: :meth:`TensorDictBase.from_struct_array` for more information.

    Examples:
        >>> x = np.array(
        ...     [("Rex", 9, 81.0), ("Fido", 3, 27.0)],
        ...     dtype=[("name", "U10"), ("age", "i4"), ("weight", "f4")],
        ... )
        >>> td = from_struct_array(x)
        >>> x_recon = td.to_struct_array()
        >>> assert (x_recon == x).all()
        >>> assert x_recon.shape == x.shape
        >>> # Try modifying x age field and check effect on td
        >>> x["age"] += 1
        >>> assert (td["age"] == np.array([10, 4])).all()

    """
    return TensorDictBase.from_struct_array(
        struct_array,
        auto_batch_size=auto_batch_size,
        batch_dims=batch_dims,
        device=device,
        batch_size=batch_size,
    )


def from_list(
    input: list[TensorCollection | Mapping],
    *,
    auto_batch_size: bool = False,
    batch_dims: int | None = None,
    device: torch.device | None = None,
    batch_size: torch.Size | None = None,
    cls: Type | None = None,
    lazy_stack: bool = None,
) -> TensorCollection:
    """Converts a list of dictionaries or TensorDicts to a TensorDict.

    .. seealso:: :meth:`TensorDictBase.from_dict` for more information.
    """
    if cls is not None:
        cls = TensorDictBase
    return cls.from_list(
        input,
        auto_batch_size=auto_batch_size,
        batch_dims=batch_dims,
        device=device,
        batch_size=batch_size,
        type=type,
        lazy_stack=lazy_stack,
    )


def from_dict(
    d,
    *,
    auto_batch_size: bool = False,
    batch_dims: int | None = None,
    device: torch.device | None = None,
    batch_size: torch.Size | None = None,
) -> "TensorDictBase":
    """Converts a dictionary to a TensorDict.

    .. seealso:: :meth:`TensorDictBase.from_dict` for more information.


    Examples:
        >>> input_dict = {"a": torch.randn(3, 4), "b": torch.randn(3)}
        >>> print(from_dict(input_dict))
        TensorDict(
            fields={
                a: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                b: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> # nested dict: the nested TensorDict can have a different batch-size
        >>> # as long as its leading dims match.
        >>> input_dict = {"a": torch.randn(3), "b": {"c": torch.randn(3, 4)}}
        >>> print(from_dict(input_dict))
        TensorDict(
            fields={
                a: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                b: TensorDict(
                    fields={
                        c: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([3, 4]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> # we can also use this to work out the batch sie of a tensordict
        >>> input_td = TensorDict({"a": torch.randn(3), "b": {"c": torch.randn(3, 4)}}, [])
        >>> print(
        from_dict(input_td))
        TensorDict(
            fields={
                a: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                b: TensorDict(
                    fields={
                        c: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([3, 4]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)

    """
    from tensordict import TensorDict

    return TensorDict.from_dict(
        d,
        auto_batch_size=auto_batch_size,
        batch_dims=batch_dims,
        device=device,
        batch_size=batch_size,
    )


def from_h5(
    h5_file,
    *,
    auto_batch_size: bool = False,
    batch_dims: int | None = None,
    device: torch.device | None = None,
    batch_size: torch.Size | None = None,
) -> "TensorDictBase":
    """Converts an HDF5 file to a TensorDict.

    .. seealso:: :meth:`TensorDictBase.from_h5` for more information.
    """
    from tensordict import TensorDict

    return TensorDict.from_h5(
        h5_file,
        auto_batch_size=auto_batch_size,
        batch_dims=batch_dims,
        device=device,
        batch_size=batch_size,
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
    """Converts a pandas DataFrame to a TensorDict.

    .. seealso:: :meth:`TensorDictBase.from_pandas` for more information.
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
    """Creates a TensorDict from a CSV file.

    .. seealso:: :meth:`TensorDictBase.from_csv` for more information.
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
    """Creates a TensorDict from a Parquet file.

    .. seealso:: :meth:`TensorDictBase.from_parquet` for more information.
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
    """Creates a TensorDict from a JSON file.

    .. seealso:: :meth:`TensorDictBase.from_json` for more information.
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

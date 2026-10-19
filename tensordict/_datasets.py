# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import warnings
from typing import Literal

import numpy as np
import torch
from tensordict._tensorcollection import TensorCollection
from tensordict.utils import _user_stacklevel, is_non_tensor


def to_mds(
    self: TensorCollection,
    *,
    columns: dict[str, str] | None = None,
    out: str | tuple[str, str],
    keep_local: bool = False,
    compression: str | None = None,
    hashes: list[str] | None = None,
    size_limit: int | str | None = 1 << 26,
    writer: "streaming.MDSWriter" | None = None,  # noqa # type-ignore
    finish_writer: bool | None = None,
    **kwargs,
) -> None:
    """Writes the tensordict to a MosaicML Streaming (MDS) dataset.

    Each element along the first batch dimension is one sample of the
    dataset; a tensordict without batch dimensions is one sample. Tensors are
    written as NumPy arrays, and zero-dimensional ones as Python numbers.
    Read the dataset back with ``streaming.StreamingDataset``. See the
    `streaming documentation <https://docs.mosaicml.com/projects/streaming>`_.

    The MDS writer has limited support for nested dictionaries: call
    :meth:`~tensordict.TensorDictBase.flatten_keys` before writing, and
    :meth:`~tensordict.TensorDictBase.unflatten_keys` after reading.

    The type of a non-tensor column is read from its first sample, which
    must not be ``None``; the other samples must have the same type, or be
    ``None``.

    Keyword Args:
        columns (dict of str to str, optional): the MDS encoding of each
            column, for example ``{"obs": "ndarray", "name": "str"}``.
            Defaults to ``None``, which infers them from the first sample.
        out (str or tuple of str): where to write the shard files:

            - a local directory: the shards are written there;
            - a remote location: the shards are written to a local temporary
              directory, uploaded, and deleted;
            - a ``(local_dir, remote_dir)`` tuple: the shards are written to
              ``local_dir`` and uploaded to ``remote_dir``.

        keep_local (bool, optional): if ``True`` and the dataset is uploaded,
            keeps the local copy. Defaults to ``False``.
        compression (str, optional): the compression, as ``"name"`` or
            ``"name:level"``. Defaults to ``None``.
        hashes (list of str, optional): the hash algorithms to apply to the
            shard files. Defaults to ``None``.
        size_limit (int or str, optional): the size after which a new shard
            starts, in bytes or as a string such as ``"100kb"``. ``None``
            writes one shard. Ignored if ``writer`` is passed. Defaults to
            ``1 << 26`` (64 MiB).
        writer (streaming.MDSWriter, optional): the writer to use. Defaults to
            ``None``, which creates one from ``out`` and the other keyword
            arguments, and finishes it after writing.
        finish_writer (bool, optional): whether to finish a writer passed as
            ``writer`` after writing, which writes the dataset index. Pass
            ``False`` to write more samples through it, and finish it
            yourself afterwards, once. A writer that ``to_mds`` creates is
            always finished. Defaults to ``None``, which leaves the writer
            open for a tensordict without batch dimensions, and finishes it,
            with a ``FutureWarning``, for a tensordict with batch dimensions.
            Starting with TensorDict 0.17, the default will change to leaving
            the writer open in both cases.
        **kwargs: passed to ``streaming.MDSWriter``.

    Raises:
        ImportError: if ``mosaicml-streaming`` is not installed.
        ValueError: if the tensordict is empty.

    Examples:
        The example needs ``mosaicml-streaming`` (``pip install mosaicml-streaming``).

        >>> import tempfile
        >>> import torch
        >>> from tensordict import LazyStackedTensorDict, TensorDict
        >>> td = LazyStackedTensorDict(
        ...     TensorDict(a=0, c=torch.zeros(2), d="a string"),
        ...     TensorDict(a=1, c=torch.zeros(3), d="another string"),
        ... )
        >>> from streaming import StreamingDataset  # doctest: +SKIP
        >>> with tempfile.TemporaryDirectory() as tmpdir:  # doctest: +SKIP
        ...     td.to_mds(out=tmpdir)
        ...     dataset = StreamingDataset(local=tmpdir, remote=None, batch_size=2)
        ...     loader = torch.utils.data.DataLoader(
        ...         dataset, batch_size=2, collate_fn=LazyStackedTensorDict.from_list
        ...     )
        ...     batch = next(iter(loader))  # a lazy stack of the two samples

    """
    try:
        from streaming import MDSWriter
    except ImportError:
        raise ImportError(
            "Failed to load MDSWriter from streaming. Check that mosaicml-streaming is installed."
        )

    if not self.numel():
        raise ValueError("Cannot write an empty TensorDict to a streaming dataset.")
    if self.ndim:
        if writer is None:
            if columns is None:
                columns = _columns(self[0])
            writer = MDSWriter(
                out=out,
                columns=columns,
                keep_local=keep_local,
                compression=compression,
                hashes=hashes,
                size_limit=size_limit,
                **kwargs,
            )
            finish_writer = True
        elif finish_writer is None:
            warnings.warn(
                "to_mds finished the writer passed by the caller after a "
                "tensordict with batch dims. Starting with TensorDict 0.17, the "
                "default will change to leaving it open, as for a tensordict "
                "without batch dims. Pass finish_writer=True or False to choose.",
                FutureWarning,
                stacklevel=_user_stacklevel(),
            )
            finish_writer = True
        td_dict = self.tolist(convert_tensors="numpy")
        if not isinstance(td_dict, list):
            raise ValueError(f"Expected a list of dictionarie, got {type(td_dict)}.")
        # a writer is a context manager that finishes it on exit
        with writer if finish_writer else contextlib.nullcontext(writer) as w:
            for td_i in td_dict:
                for k, v in td_i.items():
                    if isinstance(v, np.ndarray) and v.shape == ():
                        td_i[k] = v.item()
                w.write(td_i)
            return

    # a writer passed by the caller is left open unless finish_writer=True:
    # they may write more samples
    created = writer is None
    if created:
        if columns is None:
            columns = _columns(self)
        writer = MDSWriter(
            out=out,
            columns=columns,
            keep_local=keep_local,
            compression=compression,
            hashes=hashes,
            size_limit=size_limit,
            **kwargs,
        )
    names = getattr(writer, "column_names", None)
    encodings = getattr(writer, "column_encodings", None)
    if names is None or encodings is None:
        # a writer without column encodings gets the tensordict, as before
        td_dict = self
    else:
        # only the writer's columns are converted; the pkl columns of a
        # caller's writer store the entry itself, as before
        pkl = () if created else {n for n, e in zip(names, encodings) if e == "pkl"}
        td_dict = self.select(*(n for n in names if n not in pkl), strict=False)
        td_dict = td_dict.tolist(convert_tensors="numpy")
        for k, v in td_dict.items():
            if isinstance(v, np.ndarray) and v.shape == ():
                td_dict[k] = v.item()
        for k in pkl:
            if k in self.keys():
                td_dict[k] = self[k]
    if created or finish_writer:
        with writer as w:
            w.write(td_dict)
    else:
        writer.write(td_dict)


def _columns(self: TensorCollection) -> dict[str, str]:
    return {k: _get_elt_type(v) for k, v in self.items()}


def _get_elt_type(
    elt,
) -> Literal[
    "bytes",
    "float16",
    "float32",
    "float64",
    "int",
    "int16",
    "int32",
    "int64",
    "int8",
    "jpeg",
    "jpeg_array",
    "jpegarray",
    "json",
    "list[jpeg]",
    "list[pil]",
    "list[png]",
    "ndarray",
    "pil",
    "pkl",
    "png",
    "str",
    "str_decimal",
    "str_float",
    "str_int",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
]:
    if is_non_tensor(elt):
        from tensordict._lazy import LazyStackedTensorDict

        if isinstance(elt, LazyStackedTensorDict):
            return _get_elt_type(elt.tolist())
        return _get_elt_type(elt.data)
    if isinstance(elt, list):
        return "json"
    if isinstance(elt, bytes):
        return "bytes"
    elif isinstance(elt, (torch.Tensor, np.ndarray)):
        if not elt.shape:
            # get the dtype and return
            if isinstance(elt, torch.Tensor):
                dtype = elt.dtype
                if dtype.is_floating_point:
                    # MDS has no "float" encoding, only one per dtype
                    return str(dtype).removeprefix("torch.")
                else:
                    return "int"
            else:
                dtype = elt.dtype
                if dtype.kind == "f":
                    return dtype.name
                elif dtype.kind in ("i", "b"):
                    return "int"
                else:
                    return "str"
        return "ndarray"
    elif isinstance(elt, float):
        return "float64"
    elif isinstance(elt, (int, bool)):
        return "int"
    elif isinstance(elt, str):
        return "str"
    else:
        raise ValueError(f"Unknown type {type(elt)}")

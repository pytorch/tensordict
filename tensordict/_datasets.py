# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from tensordict._tensorcollection import TensorCollection
from tensordict.utils import is_non_tensor


def to_mds(
    self: TensorCollection,
    *,
    columns: dict[str, str] | None = None,
    out: str | tuple[str, str],
    keep_local: bool = False,
    compression: str | None = None,
    hashes: list[str] | None = None,
    size_limit: int | str | None = 1 << 26,
    writer: "streaming.MDSWriter" | None = None,  # noqa # type-ignore
    **kwargs,
) -> None:
    """Writes the content of a TensorCollection to a streaming dataset.

    Keyword Args:
        out (str | Tuple[str, str]): Output dataset directory to save shard files.

            1. If ``out`` is a local directory, shard files are saved locally.
            2. If ``out`` is a remote directory, a local temporary directory is created to
                cache the shard files and then the shard files are uploaded to a remote
                location. At the end, the temp directory is deleted once shards are uploaded.
            3. If ``out`` is a tuple of ``(local_dir, remote_dir)``, shard files are saved in the
                `local_dir` and also uploaded to a remote location.
        columns (dict[str, str]): an optional dict of columns. Will be automatically inferred from
            the tensor collection if not provided.
        keep_local (bool): If the dataset is uploaded, whether to keep the local dataset directory
            or remove it after uploading. Defaults to ``False``.
        compression (str, optional): Optional compression or compression:level. Defaults to
            ``None``.
        hashes (List[str], optional): Optional list of hash algorithms to apply to shard files.
            Defaults to ``None``.
        size_limit (Union[int, str], optional): Optional shard size limit, after which point to
            start a new shard. If ``None``, puts everything in one shard. Can specify bytes
            human-readable format as well, for example ``"100kb"`` for 100 kilobyte
            (100*1024) and so on. Defaults to ``1 << 26``.
            Ignored if
        writer: (MDSWriter, optional): the write to use. Will be created from the `out` kwarg as well as other
            input kwargs.
        **kwargs (Any): Additional settings for the Writer.

    .. note:: The MDSWriter has limited support for nested dictionaries. The proper way to handle nested tensordicts
        is to use the :meth:`~tensordict.TensorDictBase.flatten_keys` method before writing, and :meth:`~tensordict.TensorDictBase.unflatten_keys` method after reading.

    .. warning::
        This method requires `mosaicml-streaming` to be installed.

    .. warning::
        For non-tensor data, the type of the data must be fixed. The way tensordict recovers the data type is by
        looking at the first element of the list. If it's `None` an error will be thrown. Otherwise all the
        data in the list must have the same type (or be None if missing).

    .. seealso:: See the Mosaic streaming library API at `<https://docs.mosaicml.com/projects/streaming>`_

    The following example shows and end-to-end example of how to create a dataset and load it in a
    PyTorch dataloader.

    Examples:
        >>> import tempfile
        >>> from typing import Any
        >>> from tensordict import TensorDict, LazyStackedTensorDict
        >>> import torch
        >>>
        >>>
        >>> td = LazyStackedTensorDict(
        ...     TensorDict(a=0, b=1, c=torch.randn(2), d="a string"),
        ...     TensorDict(a=0, b=1, c=torch.randn(3), d="another string"),
        ...     TensorDict(a=0, b=1, c=torch.randn(3), d="yet another string"),
        ... )
        >>>
        >>> with tempfile.TemporaryDirectory() as tempdir:
        ...     # Create a dataset on one process / thread / node...
        ...     td.to_mds(out=tempdir)
        ...
        ...     # Create a dataloader
        ...     from streaming import StreamingDataset
        ...
        ...     # Load the dataset on another thread / process / node...
        ...     dataset = StreamingDataset(local=tempdir, remote=None, batch_size=2)
        ...
        ...     # Use the class `from_list` method as a collate_fn
        ...     dl = torch.utils.data.DataLoader(dataset=dataset, batch_size=2, collate_fn=LazyStackedTensorDict.from_list)
        ...     for batch in dl:
        ...         print("batch", batch)

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
        td_dict = self.tolist(convert_tensors="numpy")
        if not isinstance(td_dict, list):
            raise ValueError(f"Expected a list of dictionarie, got {type(td_dict)}.")
        with writer as w:
            for td_i in td_dict:
                for k, v in td_i.items():
                    if isinstance(v, np.ndarray) and v.shape == ():
                        td_i[k] = v.item()
                w.write(td_i)
            return

    if writer is None:
        if columns is None:
            columns = _columns(self)
        writer = MDSWriter(
            out=out,
            columns=columns,
            keep_local=keep_local,
            compression=compression,
            hashes=hashes,
            size_limit=size_limit,
        )
    writer.write(self)


def _columns(self: TensorCollection) -> dict[str, str]:
    return {k: _get_elt_type(v) for k, v in self.items()}


def _get_elt_type(
    elt,
) -> Literal[
    "bytes",
    "float",
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
                    return "float"
                else:
                    return "int"
            else:
                dtype = elt.dtype
                if dtype.kind == "f":
                    return "float"
                elif dtype.kind in ("i", "b"):
                    return "int"
                else:
                    return "str"
        return "ndarray"
    elif isinstance(elt, float):
        return "float"
    elif isinstance(elt, (int, bool)):
        return "int"
    elif isinstance(elt, str):
        return "str"
    else:
        raise ValueError(f"Unknown type {type(elt)}")

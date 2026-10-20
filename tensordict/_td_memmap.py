# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch
from tensordict._tensorcollection import TensorCollection
from tensordict.memmap import _raise_if_jagged, MemoryMappedTensor

__all__ = [
    "_format_metadata",
    "_memmap_filename",
    "_populate_empty",
    "_populate_memmap",
    "_populate_storage",
    "_save_metadata",
    "_update_metadata",
    "_warn_if_existing_file_smaller",
]


# The version of the layout that memmap writes. A meta.json without a
# "_format_version" entry was written before the format had a version: version 0.
_FORMAT_VERSION = 1


def _class_id(cls: type) -> str:
    """Returns the identifier of ``cls`` that a meta.json stores under ``"_class"``.

    The identifier is ``"module:qualname"``, with the public ``tensordict``
    module for the classes that ``tensordict`` exports, so that it does not
    depend on the private module that defines them.
    """
    import tensordict

    module = cls.__module__
    if getattr(tensordict, cls.__qualname__, None) is cls:
        module = "tensordict"
    return f"{module}:{cls.__qualname__}"


def _format_metadata(cls: type) -> dict:
    """Returns the format version and the class identifier of a meta.json for ``cls``."""
    return {"_format_version": _FORMAT_VERSION, "_class": _class_id(cls)}


def _save_metadata(data: TensorCollection, prefix: Path, metadata=None) -> None:
    """Saves the metadata of a memmap tensordict on disk."""
    filepath = prefix / "meta.json"
    if metadata is None:
        metadata = {}
    metadata.update(
        {
            "shape": list(data.shape),
            "device": str(data.device),
            "_type": str(type(data)),
        }
    )
    # Keep the entry of a key named like a format entry: the file then reads
    # as version 0, or by _type.
    for name, value in _format_metadata(type(data)).items():
        metadata.setdefault(name, value)
    with open(filepath, "wb") as json_metadata:
        from tensordict._utils_key_json import json_dumps

        json_str = json_dumps(metadata)
        # Ensure we write bytes to the binary file.
        if isinstance(json_str, str):
            json_metadata.write(json_str.encode("utf-8"))
        else:
            json_metadata.write(json_str)


def _memmap_filename(prefix, key, robust_key) -> str | None:
    """Returns the path of the file that holds the leaf ``key`` under ``prefix``."""
    from .utils import _encode_key_for_filesystem, _get_robust_key_setting_with_warning

    if prefix is None:
        return None
    effective_robust_key = _get_robust_key_setting_with_warning(key, robust_key)
    safe_key = _encode_key_for_filesystem(key, robust=effective_robust_key)
    return str(prefix / f"{safe_key}.memmap")


def _warn_if_existing_file_smaller(value, key, filename) -> None:
    """Warns that ``memmap_like`` will grow an existing file smaller than ``value``.

    Call it on the thread of the user's call, before the file is mapped. A
    warning emitted in an executor thread is not seen by the caller's
    ``warnings.catch_warnings`` on free-threaded builds, and its stacklevel
    would point at the thread pool (#2968).
    """
    from .utils import _user_stacklevel

    if os.path.islink(filename):
        # Mapping through a symlink raises anyway.
        return
    try:
        file_size = os.path.getsize(filename)
    except OSError:
        return
    # The size MemoryMappedTensor.from_tensor maps for this leaf.
    if value.is_nested:
        numel = int(value._nested_tensor_size().prod(-1).sum())
    else:
        numel = value.numel()
    nbytes = value.dtype.itemsize * numel
    if file_size < nbytes:
        warnings.warn(
            f"memmap_like grew the existing file {filename} from "
            f"{file_size} to {nbytes} bytes for the leaf {key!r} of dtype "
            f"{value.dtype}, so the leaf mixes the old values and zeros. "
            "Starting with TensorDict 0.17, the default will change to "
            "raising a RuntimeError when an existing file is smaller than "
            "the leaf. Remove the file, or use a new prefix, to get a "
            "file filled with zeros.",
            FutureWarning,
            stacklevel=_user_stacklevel(),
        )


def _populate_memmap(
    *,
    dest,
    value,
    key,
    copy_existing,
    prefix,
    like,
    existsok,
    robust_key,
    warn_existing_size=True,
):
    """Populates ``dest`` with a memory-mapped copy of ``value``.

    With ``like=True`` and ``warn_existing_size=True`` (as in ``memmap_like``),
    an existing file smaller than the leaf is grown and mapped with a
    FutureWarning (#2968). A caller that runs this in an executor passes
    ``warn_existing_size=False`` and calls
    :func:`_warn_if_existing_file_smaller` itself first.
    """
    _raise_if_jagged(value, key)
    filename = _memmap_filename(prefix, key, robust_key)
    if value.is_nested:
        shape = value._nested_tensor_size()
        if prefix is not None and filename is not None:
            shape_filename = Path(filename)
            shape_filename = shape_filename.with_suffix(".shape.memmap")
            MemoryMappedTensor.from_tensor(
                shape,
                filename=shape_filename,
                copy_existing=copy_existing,
                existsok=existsok,
                copy_data=True,
            )
    else:
        shape = None
    if like and warn_existing_size and existsok and filename is not None:
        _warn_if_existing_file_smaller(value, key, filename)
    memmap_tensor = MemoryMappedTensor.from_tensor(
        value.data if value.requires_grad else value,
        filename=filename,
        copy_existing=copy_existing,
        copy_data=not like,
        shape=shape,
        existsok=existsok,
    )
    dest._tensordict[key] = memmap_tensor
    return memmap_tensor


def _populate_empty(
    *,
    dest,
    key,
    shape,
    dtype,
    prefix,
    robust_key,
):
    """Populates ``dest`` with an empty memory-mapped tensor."""
    from .utils import _encode_key_for_filesystem, _get_robust_key_setting_with_warning

    if prefix is None:
        filename = None
    else:
        effective_robust_key = _get_robust_key_setting_with_warning(key, robust_key)
        safe_key = _encode_key_for_filesystem(key, robust=effective_robust_key)
        filename = str(prefix / f"{safe_key}.memmap")
    if isinstance(shape, torch.Tensor):
        if prefix is not None:
            shape_filename = Path(filename)
            shape_filename = shape_filename.with_suffix(".shape.memmap")
            MemoryMappedTensor.from_tensor(
                shape,
                filename=shape_filename,
                existsok=True,
                copy_data=True,
            )
    memmap_tensor = MemoryMappedTensor.empty(
        shape=shape,
        dtype=dtype,
        filename=filename,
        existsok=True,
    )
    dest._tensordict[key] = memmap_tensor
    return memmap_tensor


def _populate_storage(
    *,
    dest,
    key,
    shape,
    dtype,
    prefix,
    storage,
    robust_key,
):
    """Populates ``dest`` with a memory-mapped tensor backed by ``storage``."""
    from .utils import _encode_key_for_filesystem, _get_robust_key_setting_with_warning

    if prefix is None:
        filename = None
    else:
        effective_robust_key = _get_robust_key_setting_with_warning(key, robust_key)
        safe_key = _encode_key_for_filesystem(key, robust=effective_robust_key)
        filename = str(prefix / f"{safe_key}.memmap")
    if isinstance(shape, torch.Tensor):
        if prefix is not None:
            shape_filename = Path(filename)
            shape_filename = shape_filename.with_suffix(".shape.memmap")
            MemoryMappedTensor.from_tensor(
                shape,
                filename=shape_filename,
                existsok=True,
                copy_data=True,
            )
    memmap_tensor = MemoryMappedTensor.from_storage(
        storage=storage,
        shape=shape,
        dtype=dtype,
        filename=filename,
    )
    dest._tensordict[key] = memmap_tensor
    return memmap_tensor


def _update_metadata(*, metadata, key, value, is_collection):
    """Records the metadata of a leaf or of a nested tensordict in ``metadata[key]``."""
    if not is_collection:
        metadata[key] = {
            "device": str(value.device),
            "shape": (
                list(value.shape)
                if not value.is_nested
                else list(value._nested_tensor_size().shape)
            ),
            "dtype": str(value.dtype),
            "is_nested": value.is_nested,
        }
    else:
        metadata[key] = {
            "type": type(value).__name__,
        }

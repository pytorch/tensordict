# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

import torch
from tensordict._tensorcollection import TensorCollection
from tensordict.memmap import MemoryMappedTensor

__all__ = [
    "_populate_empty",
    "_populate_memmap",
    "_populate_storage",
    "_save_metadata",
    "_update_metadata",
]


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
    with open(filepath, "wb") as json_metadata:
        from tensordict.utils import json_dumps

        json_str = json_dumps(metadata)
        # Ensure we write bytes to the binary file.
        if isinstance(json_str, str):
            json_metadata.write(json_str.encode("utf-8"))
        else:
            json_metadata.write(json_str)


def _populate_memmap(
    *, dest, value, key, copy_existing, prefix, like, existsok, robust_key
):
    """Populate ``dest`` with a memory-mapped copy of ``value``."""
    from .utils import _encode_key_for_filesystem, _get_robust_key_setting_with_warning

    if prefix is None:
        filename = None
    else:
        effective_robust_key = _get_robust_key_setting_with_warning(key, robust_key)
        safe_key = _encode_key_for_filesystem(key, robust=effective_robust_key)
        filename = str(prefix / f"{safe_key}.memmap")
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
    """Populate ``dest`` with an empty memory-mapped tensor."""
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
    """Populate ``dest`` with a memory-mapped tensor backed by ``storage``."""
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
    """Update memmap metadata for a tensor or nested tensor collection."""
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

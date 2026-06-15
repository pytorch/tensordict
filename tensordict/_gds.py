# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""GPUDirect Storage (cuFile) helpers for consolidated TensorDicts.

This module provides the opt-in load and save paths used by
``TensorDictBase.from_consolidated(..., use_gds=True)`` and
``TensorDictBase.consolidate(filename=..., use_gds=True)``.

Nothing here is imported at top level by the rest of the package. The
public API on ``TensorDictBase`` lazily imports the helpers below only
when ``use_gds=True`` is requested, so importing :mod:`tensordict` stays
free on installs without ``torch.cuda.gds`` (PyTorch < 2.7) or without
CUDA.
"""
from __future__ import annotations

import json
import os
import weakref
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensordict.base import TensorDictBase


# Capability probe. Cheap, no side effects. ``torch.cuda.gds`` is a
# regular submodule on torch >= 2.7 but the underlying C++ bindings can
# be absent on builds without cuFile support, in which case calling any
# function in it raises RuntimeError.
_has_torch_gds = hasattr(getattr(torch.cuda, "gds", None), "GdsFile")


def _require_gds(device: torch.device) -> None:
    """Validate that GDS is usable for ``device``; raise otherwise.

    This only checks the API surface and the device type. Filesystem
    support (nvidia-fs kernel module, supported FS such as ext4 with
    direct I/O, GPFS, Lustre, WekaFS) cannot be probed without actually
    issuing a cuFile call; the caller must catch ``RuntimeError`` from
    ``load_storage`` / ``save_storage`` and surface it.
    """
    if not _has_torch_gds:
        raise RuntimeError(
            "use_gds=True requires torch.cuda.gds, available in PyTorch "
            ">= 2.7. Installed torch does not expose it."
        )
    if not hasattr(torch._C, "_gds_register_buffer"):
        raise RuntimeError(
            "use_gds=True requires the cuFile C++ bindings, which are "
            "missing from this PyTorch build."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("use_gds=True requires a CUDA-capable PyTorch build.")
    if device.type != "cuda":
        raise RuntimeError(
            f"use_gds=True requires a CUDA device, got device={device!r}."
        )


def _safe_deregister(storage) -> None:
    """Deregister a cuFile buffer, swallowing errors during teardown."""
    try:
        from torch.cuda.gds import gds_deregister_buffer

        gds_deregister_buffer(storage)
    except Exception:
        # GC-time teardown; never let this escape and crash the
        # interpreter shutdown path.
        pass


def _gds_unavailable_hint() -> str:
    return (
        " Hint: ensure the nvidia-fs kernel module is loaded and the "
        "file lives on a GDS-supported filesystem (ext4 with direct "
        "I/O, GPFS, Lustre, WekaFS, etc.)."
    )


def _load_consolidated_gds(
    filename: str | os.PathLike, device: torch.device
) -> "TensorDictBase":
    """Load a consolidated TensorDict file directly into CUDA memory.

    The file format is the one produced by
    ``TensorDictBase.consolidate(filename=...)``: a contiguous data
    region followed by a JSON metadata blob and an int64 length suffix.
    Only the data region goes through cuFile; the small trailing JSON is
    parsed on CPU.
    """
    from torch.cuda.gds import gds_register_buffer, GdsFile

    from tensordict._reductions import _rebuild_tensordict_files_consolidated

    _require_gds(device)

    filename = str(Path(filename))
    total_size = os.path.getsize(filename)

    # Parse the trailer on CPU. It is tiny (a JSON blob plus 8 bytes).
    file_cpu = torch.from_file(
        filename,
        dtype=torch.uint8,
        size=total_size,
        device=torch.device("cpu"),
    )
    metadata_size = int(file_cpu[-8:].clone().view(torch.int64).item())
    metadata_bytes = bytes(file_cpu[-metadata_size - 8 : -8].tolist())
    metadata = json.loads(metadata_bytes)
    data_region_size = total_size - metadata_size - 8
    del file_cpu

    # Allocate the consolidated CUDA buffer and register it.
    data_tensor = torch.empty(data_region_size, dtype=torch.uint8, device=device)
    storage = data_tensor.untyped_storage()
    gds_register_buffer(storage)

    gf = None
    try:
        gf = GdsFile(filename, os.O_RDONLY)
        try:
            gf.load_storage(storage, offset=0)
        finally:
            del gf
            gf = None
    except RuntimeError as exc:
        _safe_deregister(storage)
        raise RuntimeError(
            f"cuFile read failed for {filename!r}." + _gds_unavailable_hint()
        ) from exc
    except BaseException:
        # Make sure we don't leak a registration on any failure path,
        # including KeyboardInterrupt.
        if gf is not None:
            del gf
        _safe_deregister(storage)
        raise

    result = _rebuild_tensordict_files_consolidated(metadata, data_tensor)

    # Keep the buffer registered for the lifetime of the TensorDict.
    # ``gds_deregister_buffer`` only unpins the cuFile mapping; the
    # CUDA allocation itself stays valid until the storage's refcount
    # drops.
    weakref.finalize(result, _safe_deregister, storage)
    return result


def _save_consolidated_gds(
    filename: str | os.PathLike,
    storage,
    metadata_bytes: bytes,
    len_bytes: bytes,
) -> None:
    """Write a consolidated CUDA storage to ``filename`` via cuFile.

    The data region is DMA'd out via ``cuFileWrite``; the small JSON
    metadata trailer is appended with a plain ``open()`` write.

    ``metadata_bytes`` is the UTF-8 JSON blob; ``len_bytes`` is its
    length encoded as an int64 (8 bytes), matching the existing
    consolidated file layout.
    """
    from torch.cuda.gds import gds_register_buffer, GdsFile

    # The device of the storage cannot be queried via the untyped
    # storage on all torch versions in a stable way; the caller passes
    # a CUDA storage so we cross-check via ``storage.device``.
    device = torch.device(storage.device)
    _require_gds(device)

    filename = str(Path(filename))

    gds_register_buffer(storage)
    gf = None
    try:
        gf = GdsFile(filename, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        try:
            gf.save_storage(storage, offset=0)
        finally:
            del gf
            gf = None
    except RuntimeError as exc:
        _safe_deregister(storage)
        raise RuntimeError(
            f"cuFile write failed for {filename!r}." + _gds_unavailable_hint()
        ) from exc
    except BaseException:
        if gf is not None:
            del gf
        _safe_deregister(storage)
        raise
    else:
        _safe_deregister(storage)

    # Trailer: small CPU-side append; cuFile is not appropriate here.
    with open(filename, "ab") as f:
        f.write(metadata_bytes)
        f.write(len_bytes)

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Persistent tensordicts (H5, zarr and others)."""

from __future__ import annotations

import importlib
import json
import math
import os
import pickle
import sys
import tempfile
import warnings
import weakref
from numbers import Number
from pathlib import Path
from typing import Any, Callable, overload, Type, TYPE_CHECKING

import numpy as np
import torch
from tensordict._base.unsupported import (
    _NoSelectExclude,
    _NoSetNonTensor,
    _NoShapeOps,
    _NoVmap,
)
from tensordict._deprecation import deprecated
from tensordict._td import (
    _TensorDictKeysView,
    _unravel_key_to_tuple,
    NO_DEFAULT,
    TensorDict,
)
from tensordict._td_memmap import _format_metadata
from tensordict._tensorcollection import TensorCollection
from tensordict.base import (
    _default_is_leaf,
    _is_leaf_nontensor,
    _normalize_from_dict_args,
    _PACKAGE_DIR,
    _register_tensor_class,
    BEST_ATTEMPT_INPLACE,
    is_tensor_collection,
    T,
    TensorDictBase,
)
from tensordict.memmap import MemoryMappedTensor
from tensordict.utils import (
    _as_context_manager,
    _cache_while_locked,
    _is_unbatched,
    _LOCK_ERROR,
    _NUMPY_TO_TORCH_DTYPE_DICT,
    _UNBATCHED_SAVE,
    _warn_unbatched_saved_as_plain,
    expand_right,
    IndexType,
    is_non_tensor,
    KEY_SEPARATOR,
    NestedKey,
    unravel_key,
)
from torch import Tensor
from torch.utils.data import get_worker_info

_has_h5 = importlib.util.find_spec("h5py", None) is not None
_has_zarr = importlib.util.find_spec("zarr", None) is not None

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any

__all__ = ["PersistentTensorDict"]


class _Visitor:
    def __init__(self, fun=None):
        self.elts = []
        self.fun = fun

    def __call__(self, name):
        self.elts.append(name)

    def __iter__(self):
        if self.fun is None:
            yield from self.elts
        else:
            for elt in self.elts:
                yield self.fun(elt)


class _PersistentTDKeysView(_TensorDictKeysView):
    def __iter__(self):
        td = self.tensordict
        if self.include_nested:
            visitor = (
                unravel_key(tuple(key.split("/")))
                for key in td._backend.walk_keys(td.file)
            )
        else:
            visitor = td._backend.keys(td.file)
        for key in visitor:
            if self._accepts(key, td._get_metadata(key)):
                yield key

    def _accepts(self, key, metadata) -> bool:
        if metadata.get("non_tensor"):
            # For consistency with tensordict where currently a non-tensor is stored in a
            # tensorclass and hence can be seen as a nested tensordict
            # that situation should be clarified
            return self.is_leaf is _is_leaf_nontensor or not self.leaves_only
        if metadata.get("array"):
            return True
        return not self.leaves_only and (
            not isinstance(key, tuple) or self.include_nested
        )

    def __contains__(self, key):
        key = unravel_key(key)
        key_tuple = _unravel_key_to_tuple(key)
        if len(key_tuple) > 1 and not self.include_nested:
            return False
        # "/" is the storage path separator and "", "." and ".." are resolved as
        # paths by the storage library: such keys cannot be stored entries
        if any("/" in subkey or subkey in ("", ".", "..") for subkey in key_tuple):
            return False
        try:
            metadata = self.tensordict._get_metadata(key)
        except (KeyError, ValueError):
            return False
        return self._accepts(key, metadata)


# The largest block that _H5Backend.write_full allocates, unless one row of
# the dataset is larger.
_WRITE_FULL_BLOCK_BYTES = 64 << 20


class _PersistentBackend:
    """Internal adapter encapsulating the storage dialect of a :class:`PersistentTensorDict`.

    Each backend translates the generic node/group operations used by
    :class:`PersistentTensorDict` into calls of the underlying storage library
    (h5py, zarr, ...). All TensorDict semantics (locking, nesting, batch-size
    handling) live in :class:`PersistentTensorDict`; only I/O specifics live here.
    """

    name: str
    # set while from_dict writes data with a known batch size: groups then get
    # their metadata when they are created
    metadata_on_create = False
    # Pickles made before the zarr read policy retain the warning default.
    allow_pickle: bool | None = None

    def __init__(self, locking=None):
        self.locking = locking

    @staticmethod
    def available() -> bool:
        raise NotImplementedError

    def check_available(self) -> None:
        raise NotImplementedError

    # -- file / group lifecycle
    def open_file(self, filename, mode):
        raise NotImplementedError

    def open_new_file(self, filename, metadata: dict | None = None):
        raise NotImplementedError

    def close(self, file) -> None:
        raise NotImplementedError

    # -- node classification
    def is_array(self, node) -> bool:
        raise NotImplementedError

    def is_non_tensor(self, node) -> bool:
        raise NotImplementedError

    def is_non_tensor_meta(self, node) -> bool:
        raise NotImplementedError

    def node_dtype(self, node):
        raise NotImplementedError

    # -- reads
    def read_full(self, node):
        raise NotImplementedError

    def read_non_tensor(self, node):
        raise NotImplementedError

    def read_at(self, node, idx, device):
        raise NotImplementedError

    # -- writes
    def write_at(self, node, idx, value) -> None:
        raise NotImplementedError

    def write_masked(self, node, mask, value) -> None:
        raise NotImplementedError

    def write_full(self, node, value) -> None:
        """Writes the scalar ``value`` into every element of the array ``node``."""
        raise NotImplementedError

    def is_read_only(self, file) -> bool:
        raise NotImplementedError

    def try_create_dataset(self, file, key, value, kwargs) -> bool:
        raise NotImplementedError

    def copy_dataset(self, dest_file, key, src_node, kwargs) -> None:
        raise NotImplementedError

    def create_group(self, file, key, metadata: dict | None = None) -> None:
        raise NotImplementedError

    def encode_non_tensor(self, value):
        raise NotImplementedError

    def encode_non_tensor_stack(self, values, shape):
        raise NotImplementedError

    def read_non_tensor_stack(self, node):
        raise NotImplementedError

    def non_tensor_stack_shape(self, node):
        """The shape of the NonTensorStack that ``node`` holds, or ``None`` if it holds no stack."""
        shape = node.attrs.get(_STACK_ATTR)
        return None if shape is None else torch.Size(shape)

    def holds_rows(self, node, batch_size) -> bool:
        """Whether the unmarked non-tensor ``node`` has one element per row of ``batch_size``."""
        return bool(batch_size) and node.shape[: len(batch_size)] == tuple(batch_size)

    # -- structure
    def keys(self, file):
        raise NotImplementedError

    def walk_keys(self, file):
        raise NotImplementedError

    def contains(self, file, key) -> bool:
        raise NotImplementedError

    def delete(self, file, key) -> None:
        raise NotImplementedError

    def move(self, file, old_key, new_key) -> None:
        raise NotImplementedError

    # -- pickling
    def getstate(self, file) -> dict:
        raise NotImplementedError

    def setstate(self, state):
        raise NotImplementedError

    # -- optional tensordict metadata persisted alongside the data
    def read_attrs_metadata(self, group) -> dict | None:
        return None

    def write_attrs_metadata(self, group, metadata: dict) -> None:
        return None

    def finalize(self, file) -> None:
        """Hook called when a bulk write to the store is complete."""
        return None


# marks the h5 string datasets that tensordict wrote
_H5_STR_ATTR = "__tensordict_str__"
# marks the arrays that hold the elements of a NonTensorStack, with its shape
_STACK_ATTR = "__tensordict_stack__"


class _NonTensorStackArray:
    """The elements of a NonTensorStack, to be written as one array."""

    __slots__ = ("array", "shape")

    def __init__(self, array, shape):
        self.array = array
        self.shape = shape


def _all_of_type(values, ndim, cls) -> bool:
    """Whether every element of the ``ndim``-deep nested list ``values`` is exactly of type ``cls``."""
    if ndim == 0:
        return type(values) is cls
    return all(_all_of_type(value, ndim - 1, cls) for value in values)


class _H5Backend(_PersistentBackend):
    """h5py storage dialect."""

    name = "h5"

    @staticmethod
    def available() -> bool:
        return _has_h5

    def check_available(self) -> None:
        if not _has_h5:
            raise ModuleNotFoundError("Could not load h5py.")

    def open_file(self, filename, mode):
        import h5py

        return h5py.File(filename, mode, locking=self.locking)

    def open_new_file(self, filename, metadata=None):
        import h5py

        return h5py.File(filename, "w", locking=self.locking)

    def close(self, file) -> None:
        file.close()

    def is_array(self, node) -> bool:
        import h5py

        return isinstance(node, h5py.Dataset)

    def is_non_tensor(self, node) -> bool:
        return _is_non_tensor_h5(node)

    def is_non_tensor_meta(self, node) -> bool:
        return node.dtype not in _NUMPY_TO_TORCH_DTYPE_DICT

    def node_dtype(self, node):
        return node.dtype

    def read_full(self, node):
        return node[()]

    def read_non_tensor(self, node):
        value = node[()]
        if isinstance(value, bytes) and node.attrs.get(_H5_STR_ATTR, False):
            # point to the first frame outside tensordict: the user's call
            frame, stacklevel = sys._getframe(1), 2
            while frame is not None and frame.f_code.co_filename.startswith(
                _PACKAGE_DIR
            ):
                frame, stacklevel = frame.f_back, stacklevel + 1
            warnings.warn(
                f"The h5 backend of PersistentTensorDict reads the string "
                f"{node.name!r}, which tensordict wrote, as bytes. Starting with "
                "TensorDict 0.17, the default will change to reading it as str, "
                "as the other backends do. To support both, decode only bytes: "
                "`value.decode() if isinstance(value, bytes) else value`.",
                FutureWarning,
                stacklevel=stacklevel,
            )
        return value

    def read_at(self, node, idx, device):
        try:
            return torch.as_tensor(node[idx], device=device)
        except TypeError as err:
            if "Boolean indexing array has incompatible shape" in str(err):
                # Known bug in h5py: cannot broadcast boolean mask on the right as
                # done in np and torch. Therefore we put a performance warning
                # and convert to torch tensor first.
                warnings.warn(
                    "Indexing an h5py.Dataset object with a boolean mask "
                    "that needs broadcasting does not work directly. "
                    "tensordict will cast the entire array in memory and index it using the mask. "
                    "This is suboptimal and may lead to performance issue."
                )
                return torch.as_tensor(np.asarray(node), device=device)[idx]
            else:
                raise err

    def write_at(self, node, idx, value) -> None:
        try:
            node[idx] = value
        except TypeError as err:
            if "Boolean indexing array has incompatible shape" in str(err):
                # Known bug in h5py: cannot broadcast boolean mask on the right as
                # done in np and torch. Therefore we put a performance warning
                # and convert to torch tensor first.
                warnings.warn(
                    "Indexing an h5py.Dataset object with a boolean mask "
                    "that needs broadcasting does not work directly. "
                    "tensordict will cast the entire array in memory and index it using the mask. "
                    "This is suboptimal and may lead to performance issue."
                )
                idx = tuple(
                    (
                        expand_right(torch.as_tensor(_idx), node.shape).numpy()
                        if _idx.dtype == np.dtype("bool")
                        else _idx
                    )
                    for _idx in idx
                )
                node[idx] = torch.as_tensor(value)
            else:
                raise err

    def write_masked(self, node, mask, value) -> None:
        node[mask] = value

    def write_full(self, node, value) -> None:
        # h5py writes a broadcast scalar much more slowly than a full array:
        # write blocks of rows of at most _WRITE_FULL_BLOCK_BYTES (or one row,
        # if a row is larger), so that a dataset larger than memory can be
        # filled too
        if not node.shape:
            node[()] = value
            return
        if np.ndim(value):
            # not a scalar: h5py broadcasts it into the whole dataset, as before
            node[:] = value
            return
        # cast as h5py casts a scalar: a complex value raises for a real dataset
        value = np.asarray(value, dtype=node.dtype)
        if not node.size:
            return
        row_bytes = node.dtype.itemsize * math.prod(node.shape[1:])
        rows = min(node.shape[0], max(1, _WRITE_FULL_BLOCK_BYTES // max(row_bytes, 1)))
        block = np.full((rows, *node.shape[1:]), value, dtype=node.dtype)
        for start in range(0, node.shape[0], rows):
            stop = min(start + rows, node.shape[0])
            node[start:stop] = block[: stop - start]

    def try_create_dataset(self, file, key, value, kwargs) -> bool:
        stack_shape = None
        if isinstance(value, _NonTensorStackArray):
            value, stack_shape = value.array, value.shape
        try:
            node = file.create_dataset(key, data=value, **kwargs)
            if isinstance(value, str):
                # marks the strings that tensordict wrote, to read them as str
                node.attrs[_H5_STR_ATTR] = True
            elif stack_shape is not None:
                node.attrs[_STACK_ATTR] = list(stack_shape)
            return True
        except (ValueError, OSError) as err:
            if "name already exists" in str(err):
                return False
            # e.g. a closed or read-only file: the write cannot be made
            raise

    def copy_dataset(self, dest_file, key, src_node, kwargs) -> None:
        node = dest_file.create_dataset(key, data=src_node, **kwargs)
        for attr in (_H5_STR_ATTR, _STACK_ATTR):
            if attr in src_node.attrs:
                node.attrs[attr] = src_node.attrs[attr]

    def create_group(self, file, key, metadata=None) -> None:
        file.create_group(key)

    def encode_non_tensor(self, value):
        if isinstance(value, str):
            return value
        import h5py

        out = np.array(value)
        return out.astype(h5py.opaque_dtype(out.dtype))

    def encode_non_tensor_stack(self, values, shape):
        import h5py

        if _all_of_type(values, len(shape), str):
            array = np.array(values, dtype=h5py.string_dtype())
            if any("\x00" in value for value in array.ravel()):
                # h5py would raise, and the dataset would not be written
                raise ValueError(
                    "Cannot write a NonTensorStack of str that contains a NUL "
                    "character to h5: h5 strings cannot hold it. Use bytes, zarr, "
                    "or a TensorDict saved with memmap."
                )
        elif any(
            _all_of_type(values, len(shape), cls) for cls in (bytes, bool, int, float)
        ):
            # the opaque encoding keeps these values and their type
            array = self.encode_non_tensor(values)
        else:
            raise TypeError(
                "Cannot write a NonTensorStack to h5 unless its elements are all "
                "str, bytes, bool, int or float, of a single type. Use zarr, or a "
                "TensorDict saved with memmap, for other values."
            )
        return _NonTensorStackArray(array, shape)

    def read_non_tensor_stack(self, node):
        import h5py

        string_info = h5py.check_string_dtype(node.dtype)
        if string_info is not None and string_info.length is None:
            # the variable-length strings written for a stack of str
            return node.asstr()[()].tolist()
        return node[()].tolist()

    def holds_rows(self, node, batch_size) -> bool:
        import h5py

        # an opaque node holds a single value that tensordict wrote
        return not h5py.check_opaque_dtype(node.dtype) and super().holds_rows(
            node, batch_size
        )

    def keys(self, file):
        return file.keys()

    def walk_keys(self, file):
        visitor = _Visitor()
        file.visit(visitor)
        yield from visitor

    def contains(self, file, key) -> bool:
        return key in file

    def delete(self, file, key) -> None:
        del file[key]

    def move(self, file, old_key, new_key) -> None:
        try:
            file.move(old_key, new_key)
        except ValueError as err:
            # h5py raises ValueError for a missing source too; an existing
            # destination is reported first, as in _ZarrBackend.move
            if old_key not in file and new_key not in file:
                raise KeyError(f"key {old_key} not found in TensorDict.") from err
            raise KeyError(f"key {new_key} already present in TensorDict.") from err

    def is_read_only(self, file) -> bool:
        return file.file.mode == "r"

    def getstate(self, file) -> dict:
        return {"filename": file.file.filename, "group_name": file.name}

    def setstate(self, state):
        import h5py

        file = h5py.File(state["filename"], mode=state["mode"], locking=self.locking)
        if state["group_name"] != "/":
            file = file[state["group_name"]]
        return file


class _ZarrNonTensorPayload:
    """Serialized non-tensor payload to be written as a 1D uint8 zarr array."""

    __slots__ = ("payload", "encoding")

    def __init__(self, payload: bytes, encoding: str):
        self.payload = payload
        self.encoding = encoding


_ZARR_NON_TENSOR_ATTR = "__tensordict_non_tensor__"
_ZARR_METADATA_ATTR = "__tensordict__"


class _ZarrBackend(_PersistentBackend):
    """zarr (>= 3.0) storage dialect.

    Tensors are stored as zarr arrays (one chunk per leaf and no compressor
    unless specified otherwise through the ``PersistentTensorDict`` kwargs),
    nested tensordicts as zarr groups. Non-tensor entries are stored as 1D
    ``uint8`` arrays holding a JSON or pickle payload, marked with a
    ``"__tensordict_non_tensor__"`` attribute. The tensordict batch size (and
    dimension names) are persisted in a ``"__tensordict__"`` group attribute.
    """

    name = "zarr"

    def __init__(self, locking=None):
        super().__init__(locking=locking)
        # whether the store carries consolidated metadata (None = unknown yet)
        self._store_consolidated = None

    @staticmethod
    def available() -> bool:
        return _has_zarr

    def check_available(self) -> None:
        if not _has_zarr:
            raise ModuleNotFoundError(
                "Could not load zarr. Install it with `pip install 'zarr>=3.0'`."
            )
        import zarr

        if int(zarr.__version__.split(".")[0]) < 3:
            raise ImportError(
                f"tensordict's zarr backend requires zarr>=3.0, got zarr=={zarr.__version__}. "
                "Upgrade with `pip install 'zarr>=3.0'`."
            )

    @staticmethod
    def _zarr_mode(mode):
        # h5py's "x" spelling maps onto zarr's "w-"
        return "w-" if mode == "x" else mode

    def open_file(self, filename, mode):
        # zarr 2 has no use_consolidated: name the version needed (#3298)
        self.check_available()
        import zarr

        if isinstance(filename, Path):
            filename = str(filename)
        mode = self._zarr_mode(mode)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if mode == "r":
                # read-only: use consolidated metadata when present so that
                # key listing and metadata reads don't hit the store per node
                return zarr.open_group(filename, mode=mode)
            # writable: consolidated metadata would go stale on mutation
            return zarr.open_group(filename, mode=mode, use_consolidated=False)

    def open_new_file(self, filename, metadata=None):
        self.check_available()
        import zarr

        if isinstance(filename, Path):
            filename = str(filename)
        self._store_consolidated = False
        return zarr.open_group(
            filename,
            mode="w",
            use_consolidated=False,
            attributes=_zarr_metadata_attributes(metadata),
        )

    def finalize(self, file) -> None:
        """Consolidates the store metadata once a bulk write is complete."""
        import zarr

        if not getattr(file.store, "supports_deletes", True):
            # append-only stores (e.g. ZipStore): rewriting metadata appends
            # duplicate entries, and their metadata reads are cheap anyway
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zarr.consolidate_metadata(file.store)
        self._store_consolidated = True

    def _after_structural_change(self, file) -> None:
        # keep on-disk consolidated metadata (if any) in sync with the change
        if self._store_consolidated is None:
            import zarr

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._store_consolidated = (
                        zarr.open_group(
                            file.store, mode="r"
                        ).metadata.consolidated_metadata
                        is not None
                    )
            except Exception:
                self._store_consolidated = False
        if self._store_consolidated:
            self.finalize(file)

    def close(self, file) -> None:
        store_close = getattr(file.store, "close", None)
        if store_close is not None:
            store_close()

    def is_array(self, node) -> bool:
        import zarr

        return isinstance(node, zarr.Array)

    def holds_rows(self, node, batch_size) -> bool:
        # a marked node holds the bytes of a single value
        return self._non_tensor_marker(node) is None and super().holds_rows(
            node, batch_size
        )

    def _non_tensor_marker(self, node):
        try:
            return node.attrs.get(_ZARR_NON_TENSOR_ATTR)
        except Exception:
            return None

    def is_non_tensor(self, node) -> bool:
        if self._non_tensor_marker(node) is not None:
            return True
        return self.node_dtype(node) not in _NUMPY_TO_TORCH_DTYPE_DICT

    def is_non_tensor_meta(self, node) -> bool:
        return self.is_non_tensor(node)

    def node_dtype(self, node):
        return np.dtype(node.dtype)

    def read_full(self, node):
        return node[...]

    def read_non_tensor(self, node):
        marker = self._non_tensor_marker(node)
        if marker is None:
            # foreign zarr store: return the raw payload
            return node[()]
        encoding = marker.get("encoding") if isinstance(marker, dict) else None
        if encoding == "json":
            return json.loads(node[...].tobytes().decode("utf-8"))
        if encoding != "pickle":
            raise ValueError(f"Unknown zarr non-tensor encoding: {encoding!r}.")
        if self.allow_pickle is False:
            raise RuntimeError(
                "Refusing to load pickled non-tensor data from a zarr TensorDict. "
                "Pickle can execute arbitrary code. Pass allow_pickle=True only "
                "when the saved data is trusted."
            )
        if self.allow_pickle is None:
            # point to the first frame outside tensordict: the user's call
            frame, stacklevel = sys._getframe(1), 2
            while frame is not None and frame.f_code.co_filename.startswith(
                _PACKAGE_DIR
            ):
                frame, stacklevel = frame.f_back, stacklevel + 1
            warnings.warn(
                "PersistentTensorDict is loading a pickled non-tensor zarr field. "
                "This is only safe for trusted data. Starting with TensorDict "
                "0.17, the default will change to allow_pickle=False. Pass "
                "allow_pickle=True to acknowledge trusted data or "
                "allow_pickle=False to reject pickle now.",
                FutureWarning,
                stacklevel=stacklevel,
            )
        return pickle.loads(node[...].tobytes())

    @staticmethod
    def _normalize_index(idx):
        if isinstance(idx, tuple) and len(idx) == 0:
            return Ellipsis
        return idx

    @staticmethod
    def _as_dim0_bool_mask(idx):
        """Returns a 1D boolean mask over dim 0 if idx is one, else None."""
        if isinstance(idx, tuple):
            if len(idx) != 1:
                return None
            idx = idx[0]
        if (
            isinstance(idx, np.ndarray)
            and idx.dtype == np.dtype("bool")
            and idx.ndim == 1
        ):
            return idx
        return None

    def read_at(self, node, idx, device):
        idx = self._normalize_index(idx)
        try:
            return torch.as_tensor(node[idx], device=device)
        except (TypeError, IndexError, ValueError) as err:
            mask = self._as_dim0_bool_mask(idx)
            if mask is not None:
                return torch.as_tensor(node.oindex[mask], device=device)
            warnings.warn(
                "Indexing a zarr.Array with this index type is not supported "
                "out-of-core. tensordict will load the entire array in memory "
                "and index it. This is suboptimal and may lead to performance issues."
            )
            try:
                return torch.as_tensor(node[...], device=device)[idx]
            except Exception:
                raise err

    def write_at(self, node, idx, value) -> None:
        idx = self._normalize_index(idx)
        try:
            node[idx] = value
        except (TypeError, IndexError, ValueError) as err:
            mask = self._as_dim0_bool_mask(idx)
            if mask is not None:
                node.oindex[mask] = value
                return
            warnings.warn(
                "Indexing a zarr.Array with this index type is not supported "
                "out-of-core. tensordict will load the entire array in memory, "
                "index it and write it back. This is suboptimal and may lead to "
                "performance issues."
            )
            try:
                full = node[...]
                full[idx] = value
                node[...] = full
            except Exception:
                raise err

    def write_masked(self, node, mask, value) -> None:
        node[mask] = value

    def write_full(self, node, value) -> None:
        node[...] = value

    def _parent_and_name(self, file, key):
        if "/" in key:
            path, name = key.rsplit("/", 1)
            return file.require_group(path), name
        return file, key

    def _create_kwargs(self, shape, kwargs):
        create_kwargs = dict(kwargs)
        if "compressors" not in create_kwargs:
            create_kwargs["compressors"] = None
        chunks = create_kwargs.pop("chunks", None)
        if len(shape):
            if chunks is None:
                # one chunk per leaf by default: this is the checkpoint-friendly
                # layout (single read per tensor); pass chunks=... for datasets
                chunks = shape
            else:
                # a single chunks spec applies to every leaf regardless of rank:
                # it constrains the leading dimensions, trailing ones stay whole
                if isinstance(chunks, int):
                    chunks = (chunks,)
                chunks = tuple(chunks)[: len(shape)]
                chunks = chunks + tuple(shape[len(chunks) :])
            create_kwargs["chunks"] = tuple(max(1, int(c)) for c in chunks)
        return create_kwargs

    def try_create_dataset(self, file, key, value, kwargs) -> bool:
        from zarr.errors import ContainsArrayError, ContainsGroupError

        attributes = {}
        if isinstance(value, _NonTensorStackArray):
            attributes[_STACK_ATTR] = list(value.shape)
            value = value.array
        if isinstance(value, _ZarrNonTensorPayload):
            # payloads are small metadata blobs: always use the default layout.
            # The marker is set with the array so that its zarr.json is written
            # once: an append-only store (e.g. ZipStore) cannot replace an entry
            kwargs = {
                "attributes": {
                    **attributes,
                    _ZARR_NON_TENSOR_ATTR: {"encoding": value.encoding},
                }
            }
            value = np.frombuffer(value.payload, dtype=np.uint8)
            dtype = value.dtype
        else:
            value = np.asarray(value)
            dtype = value.dtype
            if attributes:
                # the strings of a stack, in zarr's variable-length string
                # type: an element can then be replaced by a longer string
                attributes = {**(kwargs.get("attributes") or {}), **attributes}
                kwargs = {**kwargs, "attributes": attributes}
                dtype = str
        parent, name = self._parent_and_name(file, key)
        try:
            array = parent.create_array(
                name,
                shape=value.shape,
                dtype=dtype,
                **self._create_kwargs(value.shape, kwargs),
            )
        except (ContainsArrayError, ContainsGroupError):
            return False
        if value.size:
            array[...] = value
        self._after_structural_change(file)
        return True

    def copy_dataset(self, dest_file, key, src_node, kwargs) -> None:
        parent, name = self._parent_and_name(dest_file, key)
        # the source attributes take precedence over user-passed ones
        attributes = {**(kwargs.get("attributes") or {}), **src_node.attrs}
        array = parent.create_array(
            name,
            shape=src_node.shape,
            dtype=src_node.dtype,
            **self._create_kwargs(src_node.shape, {**kwargs, "attributes": attributes}),
        )
        if src_node.size:
            array[...] = src_node[...]

    def create_group(self, file, key, metadata=None) -> None:
        # the metadata is written with the group so that its zarr.json is
        # written once: an append-only store (e.g. ZipStore) cannot replace it
        file.create_group(key, attributes=_zarr_metadata_attributes(metadata))
        self._after_structural_change(file)

    def encode_non_tensor(self, value):
        try:
            payload = json.dumps(value)
            if json.loads(payload) == value:
                return _ZarrNonTensorPayload(payload.encode("utf-8"), "json")
        except (TypeError, ValueError):
            pass
        return _ZarrNonTensorPayload(pickle.dumps(value), "pickle")

    def encode_non_tensor_stack(self, values, shape):
        if _all_of_type(values, len(shape), str):
            array = np.array(values)
        else:
            array = self.encode_non_tensor(values)
        return _NonTensorStackArray(array, shape)

    def read_non_tensor_stack(self, node):
        if self._non_tensor_marker(node) is not None:
            return self.read_non_tensor(node)
        return node[...].tolist()

    def keys(self, file):
        return file.keys()

    def walk_keys(self, file):
        for name, _ in file.members(max_depth=None):
            yield name

    def contains(self, file, key) -> bool:
        return key in file

    def delete(self, file, key) -> None:
        del file[key]
        self._after_structural_change(file)

    def move(self, file, old_key, new_key) -> None:
        if self.contains(file, new_key):
            raise KeyError(f"key {new_key} already present in TensorDict.")
        try:
            file.move(old_key, new_key)
            self._after_structural_change(file)
            return
        except NotImplementedError:
            pass
        # zarr-python does not implement Group.move yet: copy then delete.
        self._copy_node(file, new_key, file[old_key])
        self.delete(file, old_key)

    def _copy_node(self, file, new_key, node) -> None:
        if self.is_array(node):
            parent, name = self._parent_and_name(file, new_key)
            array = parent.create_array(
                name,
                shape=node.shape,
                dtype=node.dtype,
                chunks=node.chunks,
                attributes=dict(node.attrs),
            )
            if node.size:
                array[...] = node[...]
        else:
            group = file.require_group(new_key)
            attrs = dict(node.attrs)
            if attrs:
                group.attrs.update(attrs)
            for subkey in node.keys():
                self._copy_node(file, f"{new_key}/{subkey}", node[subkey])

    def is_read_only(self, file) -> bool:
        return file.read_only

    def getstate(self, file) -> dict:
        # zarr v3 groups (and their stores) are natively picklable
        return {"zarr_group": file}

    def setstate(self, state):
        return state.pop("zarr_group")

    def read_attrs_metadata(self, group) -> dict | None:
        try:
            metadata = group.attrs.get(_ZARR_METADATA_ATTR)
        except Exception:
            return None
        if metadata is None:
            return None
        return dict(metadata)

    def write_attrs_metadata(self, group, metadata: dict) -> None:
        try:
            if group.attrs.get(_ZARR_METADATA_ATTR) == metadata:
                # written when the group was created
                return
            group.attrs[_ZARR_METADATA_ATTR] = metadata
        except Exception:
            # best effort: read-only stores cannot persist metadata
            pass


def _zarr_metadata_attributes(metadata: dict | None) -> dict | None:
    if metadata is None:
        return None
    return {_ZARR_METADATA_ATTR: metadata}


def _attrs_metadata(batch_size, names) -> dict:
    """The batch size and names that a group persists in its attributes."""
    metadata = {"batch_size": list(batch_size), "version": 1}
    if names is not None:
        metadata["names"] = list(names)
    return metadata


_BACKENDS: dict[str, Type[_PersistentBackend]] = {
    "h5": _H5Backend,
    "zarr": _ZarrBackend,
}


class PersistentTensorDict(
    _NoShapeOps, _NoVmap, _NoSetNonTensor, _NoSelectExclude, TensorDictBase
):
    """A tensordict whose entries live in an HDF5 file or a Zarr store.

    The HDF5 backend (``backend="h5"``) needs ``h5py``. The Zarr backend
    (``backend="zarr"``) needs ``zarr>=3.0``. Each read of an entry loads it
    from the file into a new tensor, so ``ptd["a"] is ptd["a"]`` is
    ``False``, and in-place changes to the tensor that you read do not reach
    the file. Writes (``ptd["a"] = value``) go to the file, and so do the
    in-place methods of the tensordict, such as ``ptd.add_(1)`` and
    ``ptd.update_(other)``.

    Use it to read or write existing HDF5 or Zarr data. To keep tensordict
    data on disk for fast reloads, use :meth:`~tensordict.TensorDictBase.memmap`
    instead. :ref:`storage` compares the storage backends, and
    :ref:`choosing-a-container` compares the containers.

    Open a file with :meth:`from_h5` or :meth:`from_zarr`. Create one from
    data with :meth:`from_dict`. Call :meth:`close` when you are done: the
    file stays open when the object goes out of scope. To load every entry
    into memory, call :meth:`~tensordict.TensorDictBase.to_tensordict`.

    Like other :class:`~tensordict.TensorDictBase` subclasses,
    :class:`PersistentTensorDict` has a ``device`` attribute. This does not
    mean that the data is stored on that device: the entries are cast to it
    when they are read.

    With the h5 backend, a string comes back as ``bytes``. Starting with
    TensorDict 0.17, the default will change to ``str`` for the strings that
    tensordict wrote, which it marks with an attribute; reading one as
    ``bytes`` warns until then. Strings that other tools wrote stay ``bytes``.

    Keyword Args:
        batch_size (torch.Size or list of int, optional): the batch size of the
            tensordict: the leading dimensions that all its entries share.
            Defaults to ``None``, which gives ``[]``.
        filename (str, optional): the path to the h5 file or zarr store (for the
            zarr backend, a ``zarr.abc.store.Store`` instance is also accepted).
            Exclusive with ``group``. Defaults to ``None``.
        group (h5py.Group or zarr.Group, optional): an open file or group
            that holds the data. Exclusive with ``filename``. Defaults to
            ``None``.
        mode (str, optional): how the file is opened, as in :class:`h5py.File`:
            ``"r"`` (read only), ``"r+"`` (read and write, the file must
            exist), ``"w"`` (create, or truncate an existing file), ``"w-"``
            or ``"x"`` (create, fail if the file exists), ``"a"`` (read and
            write, create the file if needed). Defaults to ``"r"``.
        backend (str, optional): the storage backend, ``"h5"`` (requires
            ``h5py``) or ``"zarr"`` (requires ``zarr>=3.0``). Defaults to
            ``"h5"``.
        device (torch.device, str or int, optional): the device that the
            entries are cast to when they are read. Defaults to ``None``: the
            entries are read on CPU and ``ptd.device`` is ``None``.
        validate_batch_size (bool, optional): if ``True``, the constructor
            checks the batch size against the shapes of the stored arrays,
            which reads the metadata of the whole store. Defaults to ``True``.
        non_tensor_as_stack (bool, optional): if ``True``, a non-tensor array
            that tensordict did not write, such as a string column written by
            another tool, reads as a
            :class:`~tensordict.NonTensorStack` of its rows when its leading
            dims equal the batch size, so that indexing the tensordict selects
            its rows. Otherwise it reads as one
            :class:`~tensordict.NonTensorData` that holds the whole array.
            Defaults to ``False``.
        allow_pickle (bool, optional): with the zarr backend, whether
            pickled non-tensor entries can be read. Pickle can run arbitrary
            code: pass ``True`` only for trusted data. ``False`` raises
            before decoding a pickle. Defaults to ``None``, which reads them
            with a ``FutureWarning``. Starting with TensorDict 0.17, the
            default will change to ``allow_pickle=False``. Tensor and JSON
            entries, and the h5 backend, are not affected.
        **kwargs: passed to :meth:`h5py.Group.create_dataset` (h5 backend) or
            :meth:`zarr.Group.create_array` (zarr backend) when an entry is
            written. With the zarr backend, arrays are stored as a single
            chunk without compression by default; pass ``chunks=...`` or
            ``compressors=...`` to change it.

    Examples:
        >>> import os
        >>> import tempfile
        >>> import torch
        >>> from tensordict import PersistentTensorDict
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = os.path.join(tmpdir, "data.h5")
        ...     data = PersistentTensorDict(filename=path, batch_size=[3], mode="w")
        ...     data["a", "b"] = torch.zeros(3, 4)
        ...     data["a", "b"] += 1
        ...     print(type(data["a"]).__name__, data["a", "b"][0])
        ...     data.close()
        PersistentTensorDict tensor([1., 1., 1., 1.])

    """

    _td_dim_names = None
    _masking_on_dense_copy = True
    _to_writes_back_in_place = True
    _rename_key_lenient = True
    LOCKING = None
    # each read loads the entry from the file into a new tensor
    _reads_return_copies = True
    _leaves_built_on_read = True
    _pin_memory_unsupported = True
    _storage_holds_no_gradients = True
    _copy_shares_storage = True
    _fills_without_reading = True
    # td[key] = value writes into an existing array of the file
    _setitem_key_inplace = BEST_ATTEMPT_INPLACE
    # _change_batch_size raises, so an in-place shape method that changes the
    # batch size raises before it rewrites the datasets
    _inplace_batch_size_change = False
    # an open file cannot be sent to the workers of map
    _map_on_dense_copies = True
    # for the pickles made before non_tensor_as_stack
    _non_tensor_as_stack = False
    # (weakref to the tensordict that caches this one, key in it), or None
    _nested_parent = None

    @property
    @deprecated("PersistentTensorDict.kwargs", removal="0.17")
    def kwargs(self) -> dict[str, Any]:
        """The keyword arguments that the tensordict passes to the backend when it creates a dataset.

        .. deprecated:: 0.15
            This is internal and has no replacement.
        """
        return self._dataset_kwargs

    def __init__(
        self,
        *,
        batch_size=None,
        filename=None,
        group=None,
        mode="r",
        backend="h5",
        device=None,
        validate_batch_size: bool = True,
        non_tensor_as_stack: bool = False,
        allow_pickle: bool | None = None,
        **kwargs,
    ):
        if allow_pickle is not None and not isinstance(allow_pickle, bool):
            raise TypeError("allow_pickle must be a bool or None.")
        if batch_size is None:
            batch_size = torch.Size(())
        self._locked_tensordicts = []
        self._lock_id = set()
        backend_cls = _BACKENDS.get(backend)
        if backend_cls is None:
            raise NotImplementedError(
                f"Unknown backend {backend!r}. Supported backends are {sorted(_BACKENDS)}."
            )
        self.backend = backend
        self._backend = backend_cls(locking=self.LOCKING)
        self._backend.allow_pickle = allow_pickle
        self._backend.check_available()

        self.filename = filename
        self.mode = mode
        if filename is not None and group is None:
            self.file = self._backend.open_file(filename, mode)
        elif group is not None:
            self.file = group
        else:
            raise RuntimeError(
                f"Either group or filename must be provided, and not both. Got group={group} and filename={filename}."
            )
        self._batch_size = torch.Size(batch_size)
        self._device = torch.device(device) if device is not None else None
        self._is_shared = False
        self._is_memmap = False
        self._dataset_kwargs = kwargs

        # we use this to allow nested tensordicts to have a different batch-size
        self._nested_tensordicts = {}
        self._pin_mem = False
        self._non_tensor_as_stack = non_tensor_as_stack

        # this must be kept last
        if validate_batch_size:
            self._check_batch_size(self._batch_size)

    @classmethod
    def _open_h5(
        cls,
        filename,
        *,
        mode="r",
        batch_size: torch.Size | None = None,
        non_tensor_as_stack: bool = False,
    ):
        """Opens an h5 file; the public :meth:`~.from_h5` calls it.

        Without ``batch_size``, each nested tensordict gets the largest batch
        size that its entries share.
        """
        out = cls(
            filename=filename,
            mode=mode,
            batch_size=batch_size,
            non_tensor_as_stack=non_tensor_as_stack,
        )
        if batch_size is None:
            # determine batch size
            _set_max_batch_size(out)
        return out

    @classmethod
    def _open_zarr(
        cls,
        filename,
        *,
        mode="r",
        batch_size: torch.Size | None = None,
        non_tensor_as_stack: bool = False,
        allow_pickle: bool | None = None,
    ):
        """Opens a zarr store; the public :meth:`~.from_zarr` calls it.

        Without ``batch_size``, the batch size (and the dimension names) are
        read from the ``"__tensordict__"`` attribute that :meth:`~.from_dict`
        and :meth:`~tensordict.TensorDictBase.to_zarr` write, or else each
        nested tensordict gets the largest batch size that its entries share.
        """
        out = cls(
            filename=filename,
            mode=mode,
            batch_size=batch_size,
            backend="zarr",
            non_tensor_as_stack=non_tensor_as_stack,
            allow_pickle=allow_pickle,
        )
        if batch_size is None:
            metadata = out._backend.read_attrs_metadata(out.file)
            if metadata is not None and metadata.get("batch_size") is not None:
                batch_size = torch.Size(metadata["batch_size"])
                out._check_batch_size(batch_size)
                out._batch_size = batch_size
                names = metadata.get("names")
                if names:
                    out.names = names
            else:
                # foreign zarr store: determine batch size
                _set_max_batch_size(out)
        return out

    @classmethod
    def from_dict(
        cls,
        input_dict,
        filename,
        *,
        auto_batch_size: bool = False,
        batch_size=None,
        device=None,
        backend: str = "h5",
        batch_dims: int | None = None,
        allow_pickle: bool | None = None,
        **kwargs,
    ):
        """Writes a dict or a tensordict to a new HDF5 file or Zarr store and returns it opened.

        The file is created, or truncated if it exists, and the result is open
        for reading and writing.

        Args:
            input_dict (dict or TensorDictBase): the data to write.
            filename (str or Path): the path of the h5 file or zarr store.

        Keyword Args:
            auto_batch_size (bool, optional): not used. The batch size comes
                from ``batch_size``, or from the input as described there,
                whatever the value of this argument
                (`#2247 <https://github.com/pytorch/tensordict/issues/2247>`_).
                Defaults to ``False``.
            batch_size (torch.Size or list of int, optional): the batch size of
                the result. Defaults to ``None``: a tensordict input keeps its
                batch size, and a dict input gets the largest leading shape
                that all its entries share.
            device (torch.device, str or int, optional): the device that the
                entries are cast to when they are read. Defaults to ``None``,
                which reads them on the CPU.
            backend (str, optional): the storage backend, ``"h5"`` or
                ``"zarr"``. Defaults to ``"h5"``.
            batch_dims (int, optional): for a dict input without
                ``batch_size``, the number of leading dimensions of the
                inferred batch size. Exclusive with ``batch_size``. Defaults to
                ``None``.
            allow_pickle (bool, optional): with the zarr backend, whether the
                returned tensordict can read pickled non-tensor entries.
                ``False`` raises on them. Defaults to ``None``, which reads
                them, as ``True`` does, since they were written by this call.
                Reopening the store uses the policy of the new call.
            **kwargs: passed to :meth:`h5py.Group.create_dataset` (h5 backend)
                or :meth:`zarr.Group.create_array` (zarr backend).

        Returns:
            A :class:`PersistentTensorDict` instance linked to the newly created file.

        Examples:
            >>> import os
            >>> import tempfile
            >>> import torch
            >>> from tensordict import PersistentTensorDict
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     ptd = PersistentTensorDict.from_dict(
            ...         {"a": torch.zeros(3, 4), "b": {"c": torch.ones(3)}},
            ...         os.path.join(tmpdir, "data.h5"),
            ...     )
            ...     print(ptd.batch_size, ptd["b", "c"])
            ...     ptd.close()
            torch.Size([3]) tensor([1., 1., 1.])

        """
        return cls._from_dict(
            input_dict,
            filename,
            batch_size=_normalize_from_dict_args(
                input_dict, batch_size=batch_size, batch_dims=batch_dims
            ),
            device=device,
            backend=backend,
            batch_dims=batch_dims,
            allow_pickle=allow_pickle,
            **kwargs,
        )

    @classmethod
    def _from_dict(
        cls,
        input_dict,
        filename,
        *,
        batch_size=None,
        device=None,
        backend: str = "h5",
        names=None,
        batch_dims=None,
        allow_pickle: bool | None = None,
        **kwargs,
    ):
        if allow_pickle is not None and not isinstance(allow_pickle, bool):
            raise TypeError("allow_pickle must be a bool or None.")
        # Only this writer truncates the store and knows all its values.
        if backend == "zarr" and allow_pickle is None:
            allow_pickle = True
        # from_dict, which also sets the dimension names before writing the
        # metadata, so that a store that cannot replace an entry gets it once
        backend_cls = _BACKENDS.get(backend)
        if backend_cls is None:
            raise NotImplementedError(
                f"Unknown backend {backend!r}. Supported backends are {sorted(_BACKENDS)}."
            )
        _has_batch_size = True
        if batch_size is None:
            if is_tensor_collection(input_dict):
                batch_size = input_dict.batch_size
            else:
                _has_batch_size = False
                batch_size = torch.Size([])
        metadata = None
        if _has_batch_size:
            metadata = _attrs_metadata(batch_size, _source_names(input_dict, names))
        file = backend_cls(locking=cls.LOCKING).open_new_file(filename, metadata)

        # let's make a tensordict first
        out = cls(
            group=file,
            batch_size=batch_size,
            device=device,
            backend=backend,
            allow_pickle=allow_pickle,
            **kwargs,
        )
        out._backend.metadata_on_create = _has_batch_size
        # one UnbatchedTensor warning per save, however many leaves (#3264)
        outer = getattr(_UNBATCHED_SAVE, "warned", None)
        if outer is None:
            _UNBATCHED_SAVE.warned = False
        try:
            if is_tensor_collection(input_dict):
                out.update(input_dict)
            else:
                out.update(TensorDict(input_dict, batch_size=batch_size))
        finally:
            if outer is None:
                _UNBATCHED_SAVE.warned = None
        out._backend.metadata_on_create = False
        if not _has_batch_size:
            _set_max_batch_size(out)
            if batch_dims is not None:
                out.auto_batch_size_(batch_dims=batch_dims)
        if names is not None:
            out.names = names
        out._write_attrs_metadata()
        out._backend.finalize(out.file)
        return out

    def _write_attrs_metadata(self):
        """Persists batch size and names in the storage attributes (best effort)."""
        metadata = _attrs_metadata(self.batch_size, self._td_dim_names)
        self._backend.write_attrs_metadata(self.file, metadata)
        for td in self._nested_tensordicts.values():
            td._write_attrs_metadata()

    def _nested_batch_size_from_attrs(self, group):
        """Batch size persisted in a nested group's metadata, or ``None``."""
        metadata = self._backend.read_attrs_metadata(group)
        if metadata is not None and metadata.get("batch_size") is not None:
            return torch.Size(metadata["batch_size"])
        return None

    def _make_nested(self, key, group):
        """Builds (and caches the backend of) a nested PersistentTensorDict."""
        nested_batch_size = self._nested_batch_size_from_attrs(group)
        out = self._nested_tensordicts[key] = PersistentTensorDict(
            group=group,
            batch_size=(
                nested_batch_size if nested_batch_size is not None else self.batch_size
            ),
            device=self.device,
            backend=self.backend,
            # a persisted batch size was already validated at write time
            validate_batch_size=nested_batch_size is None,
            non_tensor_as_stack=self._non_tensor_as_stack,
            allow_pickle=self._backend.allow_pickle,
            **self._dataset_kwargs,
        )
        # share the backend instance so per-store state (e.g. the consolidated
        # metadata flag of the zarr backend) is tracked once per store
        out._backend = self._backend
        out._nested_parent = (weakref.ref(self), key)
        return out

    def _link_nested(self):
        """Records ``self`` as the parent of its cached nested tensordicts."""
        for key, td in self._nested_tensordicts.items():
            td._nested_parent = (weakref.ref(self), key)

    def close(self):
        """Closes the file or store of the tensordict.

        The file is not closed when the tensordict goes out of scope. After
        ``close()``, reading an entry raises a ``KeyError``. Writing one to
        an HDF5 file raises; whether a zarr store still accepts writes depends
        on the store (a local directory store does).
        """
        self._backend.close(self.file)

    def _process_key(self, key):
        key = _unravel_key_to_tuple(key)
        return "/".join(key)

    def _check_batch_size(self, batch_size) -> None:
        if not len(batch_size):
            # an empty batch size cannot mismatch any array
            return
        for key, metadata in self._items_metadata(
            include_nested=True, leaves_only=True
        ):
            if not metadata.get("array"):
                continue
            size = metadata["shape"]
            if torch.Size(size[: len(batch_size)]) != batch_size:
                # metadata alone cannot tell opaque-encoded non-tensor payloads
                # apart from genuine arrays (h5); fetch the node to double check
                if self._backend.is_non_tensor(self._get_array(key)):
                    continue
                raise ValueError(
                    f"batch size and array size mismatch: array.shape={tuple(size)}, batch_size={batch_size}."
                )

    def _get_array(self, key, default=NO_DEFAULT):
        try:
            key = self._process_key(key)
            array = self.file[key]
            return array
        except KeyError:
            if default is not NO_DEFAULT:
                return default
            # name the file, not the tensordict: its repr reads every dataset
            raise KeyError(f"key {key} not found in PersistentTensorDict {self.file}")

    def _process_array(self, key, array):
        if self._backend.is_array(array):
            if self.device is not None:
                device = self.device
            else:
                device = torch.device("cpu")
            # we convert to an array first to avoid "Creating a tensor from a list of numpy.ndarrays is extremely slow."
            if not self._backend.is_non_tensor(array):
                array = self._backend.read_full(array)
                out = torch.as_tensor(array, device=device)
                if self._pin_mem:
                    out = out.pin_memory()
            else:
                from tensordict.tensorclass import NonTensorData, NonTensorStack

                stack_shape = self._backend.non_tensor_stack_shape(array)
                if stack_shape is not None:
                    out = NonTensorStack._from_list(
                        self._backend.read_non_tensor_stack(array),
                        device=device,
                        ndim=len(stack_shape),
                    )
                elif (rows := self._non_tensor_rows(key, array)) is not None:
                    # one element per row, decoded as a NonTensorData would be
                    out = NonTensorStack._from_list(
                        self._backend.read_non_tensor(array).tolist(),
                        device=device,
                        ndim=len(rows),
                    )
                else:
                    array = self._backend.read_non_tensor(array)
                    out = NonTensorData(
                        data=array, device=device, batch_size=self.batch_size
                    )
            return out
        else:
            out = self._nested_tensordicts.get(key)
            if out is None:
                out = self._make_nested(key, array)
            return out

    def _non_tensor_rows(self, key, array):
        """The batch size of the rows that the unmarked non-tensor ``array`` reads as, or ``None``."""
        if not self._non_tensor_as_stack:
            return None
        # the rows are those of the tensordict that holds the entry
        owner = self
        for subkey in _unravel_key_to_tuple(key)[:-1]:
            owner = owner._get_str(subkey, NO_DEFAULT)
        if self._backend.holds_rows(array, owner.batch_size):
            return owner.batch_size
        return None

    @_cache_while_locked  # noqa: B019
    def _get_str(self, key: NestedKey, default, **kwargs):
        key = _unravel_key_to_tuple(key)
        array = self._get_array(key, default)
        if array is default:
            return array
        return self._process_array(key, array)

    _get_tuple = _get_str

    def _read_array_at(self, array, idx):
        """Reads ``array[idx]``, loading only the required part of the array when possible.

        Only indices that the storage library reads with the same semantics as
        torch, and efficiently, are delegated to it. Other indices are read in
        full and indexed in memory.
        """
        if self.device is not None:
            device = self.device
        else:
            device = torch.device("cpu")
        out = None
        if _is_basic_index(idx):
            try:
                out = self._backend.read_at(array, idx, device)
            except (TypeError, ValueError, IndexError, OSError):
                # e.g. out of bounds: torch raises the error below
                pass
        elif len(array.shape):
            span = _index_span(idx, array.shape[0])
            if span is not None:
                # Read the rows between the first and last selected ones as a
                # slice: point selection is slow in h5py and requires sorted,
                # unique indices.
                start, stop, local_idx = span
                out = self._backend.read_at(array, slice(start, stop), device)
                out = out[local_idx.to(out.device)]
        if out is None:
            out = torch.as_tensor(self._backend.read_full(array), device=device)[idx]
        if self._pin_mem:
            return out.pin_memory()
        return out

    def _get_at_str(self, key, idx, default, **kwargs):
        array = self._get_array(key, default)
        if array is default:
            return default
        if self._backend.is_array(array):
            if not self._backend.is_non_tensor(array):
                return self._read_array_at(array, idx)
            # a locked td indexes the stack that _get_str caches instead
            if not self.is_locked:
                out = self._read_non_tensor_rows_at(key, array, idx)
                if out is not None:
                    return out
        return super()._get_at_str(key, idx, default, **kwargs)

    _get_at_tuple = _get_at_str

    def _read_non_tensor_rows_at(self, key, array, idx):
        """Reads ``idx`` of a non-tensor ``array`` read as rows, or returns ``None``.

        Only the indexed rows are read, so that the cost does not grow with
        the length of the array. ``None`` is returned for an array that is not
        read as rows, and for an index that is not supported here.
        """
        if self._backend.non_tensor_stack_shape(array) is not None:
            return None
        rows = self._non_tensor_rows(key, array)
        if rows is None:
            return None
        try:
            if _is_basic_index(idx):
                items = idx if isinstance(idx, tuple) else (idx,)
                # an Ellipsis would reach the dimensions after the rows
                if Ellipsis in items or len(items) > len(rows):
                    return None
                values = array[idx]
            else:
                span = _index_span(idx, rows[0])
                if span is None:
                    return None
                start, stop, local_idx = span
                values = np.asarray(array[start:stop])[local_idx.numpy()]
        except (TypeError, ValueError, IndexError, OSError):
            # e.g. out of bounds: the full read below raises as torch does
            return None
        from tensordict.tensorclass import NonTensorData, NonTensorStack

        device = self.device if self.device is not None else torch.device("cpu")
        shape = np.shape(values)
        ndim = len(shape) - (array.ndim - len(rows))
        if not math.prod(shape[:ndim]):
            # no row: leave the empty stack to the full read
            return None
        values = values.tolist() if hasattr(values, "tolist") else values
        if not ndim:
            return NonTensorData(data=values, device=device, batch_size=torch.Size([]))
        return NonTensorStack._from_list(values, device=device, ndim=ndim)

    def _get_metadata(self, key):
        """Gets the metadata for an entry.

        This method avoids creating a tensor from scratch, and just reads the metadata of the array.
        """
        array = self._get_array(key)
        if self._backend.is_array(array) and not self._backend.is_non_tensor_meta(
            array
        ):
            shape = torch.Size(array.shape)
            return {
                "dtype": _NUMPY_TO_TORCH_DTYPE_DICT[self._backend.node_dtype(array)],
                "shape": shape,
                "dim": len(shape),
                "array": True,
            }
        elif self._backend.is_array(array):
            stack_shape = self._backend.non_tensor_stack_shape(array)
            if stack_shape is not None:
                # a stack has a shape, which counts towards the batch size
                return {
                    "non_tensor": True,
                    "shape": stack_shape,
                    "dim": len(stack_shape),
                }
            return {"non_tensor": True}
        else:
            val = self.get(key)
            shape = val.shape
            return {
                "dtype": None,
                "shape": shape,
                "dim": len(shape),
                "array": False,
            }

    @classmethod
    def _process_index(cls, idx, array=None):
        if isinstance(idx, tuple):
            return tuple(cls._process_index(_idx, array) for _idx in idx)
        if isinstance(idx, torch.Tensor):
            return idx.cpu().detach().numpy()
        if isinstance(idx, (range, list)):
            return np.asarray(idx)
        return idx

    @overload
    def __getitem__(self, item: str | tuple[str, ...]) -> Any: ...

    @overload
    def __getitem__(self, item: IndexType) -> TensorDictBase: ...

    # Override: an index gives a writable view of the file.
    def __getitem__(self, item: IndexType) -> Self | Tensor | TensorCollection | Any:
        if isinstance(item, str) or (
            isinstance(item, tuple) and _unravel_key_to_tuple(item)
        ):
            result = self.get(item, default=NO_DEFAULT)
            if is_non_tensor(result):
                result_data = getattr(result, "data", NO_DEFAULT)
                if result_data is NO_DEFAULT:
                    return result.tolist(as_linked_list=True)
                return result_data
            return result
        if isinstance(item, list):
            # convert to tensor
            item = torch.tensor(item)
        return self._get_sub_tensordict(item)

    # Override: a view of the file, as an index gives, except in a DataLoader
    # worker.
    def __getitems__(self, indices: IndexType) -> TensorDictBase:
        sub_td = self[indices]
        if get_worker_info() is None or not is_tensor_collection(sub_td):
            return sub_td
        # In a DataLoader worker, read the rows here: the lazy view pickles as
        # the file name, so the main process would read every batch (#3218).
        return _read_rows(sub_td)

    # Override: writes through a sub-tensordict of the file, and a number into
    # the tensor entries only.
    def _setitem_index(self, index: IndexType, value: Any) -> None:
        if isinstance(index, list):
            # convert to tensor
            index = torch.tensor(index)
        sub_td = self._get_sub_tensordict(index)
        if isinstance(value, Number):
            # write the scalar into every tensor entry, in the dtype of the
            # entry and expanded to the indexed shape, which h5py needs for an
            # index tensor. As in a TensorDictStore, the non-tensor entries are
            # left as they are.
            entries = {}
            for key, metadata in self._items_metadata(
                include_nested=True, leaves_only=True
            ):
                # with h5, the leaves include non-tensor entries stored as numbers
                if self.entry_class(key) is torch.Tensor:
                    shape = sub_td.batch_size + metadata["shape"][self.batch_dims :]
                    entries[key] = torch.as_tensor(
                        value, dtype=metadata["dtype"]
                    ).expand(shape)
            # cast first: a scalar that a dtype cannot hold raises before any write
            for key, item in entries.items():
                sub_td.set_(key, item)
            return
        err_set_batch_size = None
        if not isinstance(value, TensorDictBase):
            value = TensorDict.from_dict(value, batch_size=[])
            # try to assign the current shape. If that does not work, we can
            # try to expand
            try:
                value.batch_size = sub_td.batch_size
            except RuntimeError as err0:
                err_set_batch_size = err0
        if value.shape != sub_td.shape:
            try:
                value = value.expand(sub_td.shape)
            except RuntimeError as err:
                if err_set_batch_size is not None:
                    raise err from err_set_batch_size
                raise RuntimeError(
                    f"Cannot broadcast the tensordict {value} to the shape of the indexed persistent tensordict {self}[{index}]."
                ) from err
        sub_td.update(value, inplace=True)

    # @_cache_while_locked  # noqa: B019
    def keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
        *,
        sort: bool = False,
    ) -> _PersistentTDKeysView:
        if is_leaf not in (None, _default_is_leaf, _is_leaf_nontensor):
            raise ValueError(
                f"is_leaf {is_leaf} is not supported within tensordicts of type {type(self).__name__}."
            )
        return _PersistentTDKeysView(
            tensordict=self,
            include_nested=include_nested,
            leaves_only=leaves_only,
            is_leaf=is_leaf,
            sort=sort,
        )

    def _items_metadata(self, include_nested=False, leaves_only=False):
        """Iterates over the metadata of the PersistentTensorDict."""
        for key in self.keys(include_nested, leaves_only):
            yield (key, self._get_metadata(key))

    # Override: the batch size changes only through the batch_size setter,
    # which checks it against the file.
    def _change_batch_size(self, value):
        raise NotImplementedError

    # Override: checks the batch size against the file; if it does not fit,
    # keeps the old one and raises, like TensorDict.
    @TensorDictBase.batch_size.setter
    def batch_size(self, value):
        _batch_size = self._batch_size
        try:
            self._batch_size = torch.Size(value)
            self._check_batch_size(self._batch_size)
        except ValueError as err:
            self._batch_size = _batch_size
            raise RuntimeError(str(err)) from err

    # Override: a nested name that clashes with the new names warns and is
    # dropped until 0.17, where it raises as on TensorDict.
    def _rename_subtds(self, names):
        if names is None:
            names = [None] * self.ndim
        for key, item in self._nested_tensordicts.items():
            if is_tensor_collection(item):
                td_names = list(names) + list(item.names)[self.ndim :]
                clash = [
                    n for n in td_names[self.ndim :] if n is not None and n in names
                ]
                if clash:
                    # nested names were dropped before #2252, so this did not raise
                    warnings.warn(
                        f"The new names {list(names)} clash with the names "
                        f"{clash} of the nested tensordict at key {key!r}, so "
                        "those dims are left unnamed. Starting with TensorDict "
                        "0.17, the default will change to raising a "
                        "ValueError, as TensorDict does.",
                        FutureWarning,
                        stacklevel=4,
                    )
                    td_names = list(names) + [
                        None if n in clash else n for n in td_names[self.ndim :]
                    ]
                if (
                    item.batch_dims < self.ndim
                    and sum(n is None for n in td_names) == item.batch_dims
                ):
                    # a group that reopens with fewer batch dims (a zip
                    # store) drops these names: drop them without the
                    # warning for a list of the wrong length
                    td_names = [None] * item.batch_dims
                item.rename_(*td_names)

    # Override: deletes the node in the file.
    def _del(self, key: tuple[str, ...]) -> None:
        path = self._process_key(key)
        # h5py raises a KeyError for a missing key, zarr does not
        missing = self.backend == "zarr" and not self._backend.contains(self.file, path)
        self._backend.delete(self.file, path)
        # a sub-tensordict read before keeps the deleted group: drop it, so that
        # a new entry at this key gets a fresh one
        self._forget_nested(key)
        if missing:
            # stacklevel 4: skip del_ and its _lock_blocked wrapper
            warnings.warn(
                f"{type(self).__name__}.del_ of a missing key "
                f"{key[0] if len(key) == 1 else key!r} does "
                "nothing today with the zarr backend. Starting with TensorDict "
                "0.17, the default will change to raising a KeyError, as "
                "TensorDict does.",
                FutureWarning,
                stacklevel=4,
            )

    # Override: a TensorDict skeleton, built without reading the tensors: only
    # the sub-tensordicts and the non-tensor entries are read.
    def _empty(self, recurse, batch_size, device, names):
        out = TensorDict(
            {},
            device=self.device if device is NO_DEFAULT else device,
            batch_size=self.batch_size if batch_size is None else batch_size,
            names=self.names if names is None and self._has_names() else names,
        )
        if recurse:
            for key in self.keys():
                if self.entry_class(key) is torch.Tensor:
                    continue
                val = self._get_str(key, NO_DEFAULT)
                if is_tensor_collection(val):
                    out._set_str(
                        key,
                        val.empty(
                            recurse=True,
                            batch_size=batch_size,
                            device=device,
                            names=names,
                        ),
                        inplace=False,
                        validated=True,
                        non_blocking=False,
                    )
        return out

    # Override: walk the cached sub-tensordicts, without reading the file.
    def _lock_sub_tensordicts(self):
        return list(self._nested_tensordicts.values())

    def _storage_is_read_only(self) -> bool:
        return self._backend.is_read_only(self.file)

    def _write_back_to(self, result: TensorDictBase) -> None:
        """Writes the dense copy of a ``with self.to(...)`` block back into the file, on exit.

        ``result`` is the copy that ``to()`` returned, as the block left it.
        The caller has checked that it is not ``self`` and that the file is
        writable. A copy on ``meta`` holds no data: nothing is written.
        Otherwise each leaf that the file and ``result`` both have, with the
        same shape, is written into its dataset in place: the file casts it
        to the dtype of the dataset, keeps its layout (compression, chunks),
        and reads nothing. The other entries of ``result`` (new keys, entries
        whose shape the block changed, a group it replaced with a tensor) go
        through ``update()``, which replaces or creates them without looking
        them up in the file; a tensor replaced with a tensordict makes it
        raise, as on a ``TensorDict``. As on a ``TensorDict``, the non-tensor
        entries that the file has stay unchanged.
        """
        if result.device is not None and result.device.type == "meta":
            return
        keys = set(self.keys(True, True, is_leaf=_is_leaf_nontensor))
        keys &= set(result.keys(True, True, is_leaf=_is_leaf_nontensor))
        for key, value in result.items(True, True):
            if key not in keys:
                continue
            if self._get_metadata(key).get("shape") == value.shape:
                self.set_(key, value)
            else:
                # the block changed its shape: update() replaces it
                keys.discard(key)
        self.update(result.exclude(*keys))

    def entry_class(self, key: NestedKey) -> type:
        # Inspects the node only: no data is read from storage.
        array = self._get_array(key)
        if not self._backend.is_array(array):
            return PersistentTensorDict
        if self._backend.is_non_tensor(array):
            from tensordict.tensorclass import NonTensorData, NonTensorStack

            if (
                self._backend.non_tensor_stack_shape(array) is not None
                or self._non_tensor_rows(key, array) is not None
            ):
                return NonTensorStack
            return NonTensorData
        return torch.Tensor

    # Override: the file reads give copies, so the fill is written to the
    # arrays directly, without reading them.
    def _masked_fill_(self, mask, value):
        for key in self.keys(include_nested=True, leaves_only=True):
            array = self._get_array(key)
            self._backend.write_masked(
                array, expand_right(mask, array.shape).cpu().numpy(), value
            )
        return self

    def _memmap_(
        self,
        *,
        prefix: str | None,
        copy_existing: bool,
        executor,
        futures,
        inplace,
        like,
        share_non_tensor,
        existsok,
        robust_key,
    ) -> T:
        if inplace:
            raise RuntimeError(
                "Cannot build a memmap TensorDict in-place from a PersistentTensorDict. "
                "Use `td.memmap()` instead."
            )

        # re-implements this to make it faster using the meta-data
        def save_metadata(data: TensorDictBase, filepath, metadata=None):
            if metadata is None:
                metadata = {}
            metadata.update(
                {
                    "shape": list(data.shape),
                    "device": str(data.device),
                    "_type": str(type(self)),
                }
            )
            for name, value in _format_metadata(type(self)).items():
                metadata.setdefault(name, value)
            with open(filepath, "wb") as json_metadata:
                from tensordict._utils_key_json import json_dumps

                json_str = json_dumps(metadata)
                # Ensure we write bytes to the binary file
                if isinstance(json_str, str):
                    json_metadata.write(json_str.encode("utf-8"))
                else:
                    json_metadata.write(json_str)

        if prefix is not None:
            prefix = Path(prefix)
            if not prefix.exists():
                os.makedirs(prefix, exist_ok=True)
            metadata = {}
        if not self.keys():
            raise Exception(
                "memmap_like() must be called when the TensorDict is (partially) "
                "populated. Set a tensor first."
            )
        dest = TensorDict(
            {},
            batch_size=self.batch_size,
            names=self.names if self._has_names() else None,
            device=torch.device("cpu"),
        )
        dest._is_memmap = True
        for key, value in self._items_metadata():
            # an h5 enum has a torch dtype in the metadata but reads as a
            # non-tensor
            if not value.get("array") or self._backend.is_non_tensor(
                self._get_array(key)
            ):
                value = self._get_str(key, default=NO_DEFAULT)
                dest._set_str(
                    key,
                    value._memmap_(
                        prefix=prefix / key if prefix is not None else None,
                        executor=executor,
                        like=like,
                        copy_existing=copy_existing,
                        futures=futures,
                        inplace=inplace,
                        share_non_tensor=share_non_tensor,
                        existsok=existsok,
                        robust_key=robust_key,
                    ),
                    inplace=False,
                    validated=True,
                    non_blocking=False,
                )
                if prefix is not None:
                    metadata[key] = {
                        "type": type(value).__name__,
                    }
                continue
            else:
                value = self._get_str(key, default=NO_DEFAULT)
                if prefix is not None:
                    metadata[key] = {
                        "dtype": str(value.dtype),
                        "shape": list(value.shape),
                        "device": str(value.device),
                    }

                def _populate(
                    tensordict=dest, key=key, value=value, prefix=prefix, like=like
                ):
                    val = MemoryMappedTensor.from_tensor(
                        value,
                        filename=(
                            str(prefix / f"{key}.memmap")
                            if prefix is not None
                            else None
                        ),
                        copy_data=not like,
                        copy_existing=copy_existing,
                        existsok=existsok,
                    )
                    tensordict._set_str(
                        key,
                        val,
                        inplace=False,
                        validated=True,
                        non_blocking=False,
                    )

                if executor is None:
                    _populate()
                else:
                    futures.append(executor.submit(_populate))

        if prefix is not None:
            if executor is None:
                save_metadata(dest, prefix / "meta.json", metadata)
            else:
                futures.append(
                    executor.submit(save_metadata, dest, prefix / "meta.json", metadata)
                )
        return dest

    def _popped_value(self, value):
        # Override: a zarr group reads from the store: copy it before deleting it
        if (
            self.backend == "zarr"
            and not self.is_locked
            and isinstance(value, PersistentTensorDict)
        ):
            return value.to_tensordict()
        return value

    # Override: moves the node in the file, without reading the data.
    def _rename_key(self, old_key: NestedKey, new_key: NestedKey) -> None:
        old = _unravel_key_to_tuple(old_key)
        new = _unravel_key_to_tuple(new_key)
        old_path = self._process_key(old)
        new_path = self._process_key(new)
        if (
            old
            and new
            and all(old)
            and all(new)
            # neither key lies under the other
            and old[: len(new)] != new
            and new[: len(old)] != old
            and self._backend.contains(self.file, new_path)
            and self._backend.contains(self.file, old_path)
        ):
            # the backends refuse to move onto an existing node (rename_key_
            # has checked safe): overwrite it, as TensorDict does
            self._backend.delete(self.file, new_path)
            self._forget_nested(new)
        if (
            new
            and len(old) > len(new)
            and old[: len(new)] == new
            and self._backend.contains(self.file, old_path)
        ):
            # the entry replaces the key that holds it, as in TensorDict: move
            # it out of that key before the key is deleted
            moved_path = new_path + "_"
            while self._backend.contains(self.file, moved_path):
                moved_path += "_"
            self._backend.move(self.file, old_path, moved_path)
            self._backend.delete(self.file, new_path)
            self._forget_nested(new)
            old_path = moved_path
        if old and len(new) > len(old) and new[: len(old)] == old:
            # the backends cannot move a node inside itself: move it aside
            # first, as TensorDict pops the entry before it sets it
            moved_path = old_path + "_"
            while self._backend.contains(self.file, moved_path):
                moved_path += "_"
            self._backend.move(self.file, old_path, moved_path)
            self._forget_nested(old)
            old_path = moved_path
        self._backend.move(self.file, old_path, new_path)
        if old:
            self._forget_nested(old)

    def _forget_nested(self, key: tuple) -> None:
        """Drops the cached nested tensordicts at or under ``key``, whose node has moved or gone.

        The tensordicts that hold this one cache their own copies of these
        nodes (#3181): they are dropped from the outermost one down.
        """
        td = self
        while td._nested_parent is not None:
            parent = td._nested_parent[0]()
            if parent is None:
                break
            key = td._nested_parent[1] + key
            td = parent
        td._drop_nested(key)

    def _drop_nested(self, key: tuple) -> None:
        for cached in list(self._nested_tensordicts):
            if cached[: len(key)] == key:
                del self._nested_tensordicts[cached]
            elif key[: len(cached)] == cached:
                self._nested_tensordicts[cached]._drop_nested(key[len(cached) :])

    # Override: writes the arrays without reading them.
    def _fill_(self, key: tuple[str, ...], value: float | bool) -> None:
        md = self._get_metadata(key)
        if md.get("array"):
            self._backend.write_full(self._get_array(key), value)
        else:
            nested = self.get(key)
            for subkey in nested.keys():
                nested._fill_((subkey,), value)

    def _create_nested_str(self, key):
        # The existing entry is replaced. On a read-only file, create_group
        # raises the ValueError it raised before entries were replaced (a
        # delete first would raise a KeyError instead).
        if not self._storage_is_read_only() and self._backend.contains(self.file, key):
            self._del((key,))
        self._backend.create_group(self.file, key)
        target_td = self._get_str(key, default=NO_DEFAULT)
        return target_td

    # Override: an existing group on the path keeps its entries.
    def _create_nested_tuple(self, key):
        if self._storage_is_read_only():
            # a read-only file raises from creating the first level, as
            # before, and keeps its cached sub-tensordicts
            self._create_nested_str(key[0])
            return
        if len(key) > 1:
            group = self._get_array(key[0], None)
            if group is not None and not self._backend.is_array(group):
                cached = self._nested_tensordicts.get((key[0],))
                if cached is not None and cached.is_locked:
                    # a locked sub-tensordict is not written into: creating
                    # the group raises for the existing name, as before
                    self._backend.create_group(self.file, key[0])
                # a sub-tensordict read before under this key would keep a
                # replaced group
                self._forget_nested(key)
                td = self._get_str(key[0], NO_DEFAULT)
            else:
                self._forget_nested(key)
                td = self._create_nested_str(key[0])
            td._create_nested_tuple(key[1:])
        else:
            self._forget_nested(key)
            self._create_nested_str(key[0])

    # Override: one to_tensordict() read, outside the cache that the base
    # keeps while locked: a locked file would keep a copy of its data, and
    # later calls would miss writes from other processes. inplace=True raises.
    @_as_context_manager()
    def flatten_keys(
        self,
        separator: str = KEY_SEPARATOR,
        inplace: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
    ) -> T:
        """Returns an in-memory copy where the nested keys are joined by ``separator``.

        The entries are first loaded with
        :meth:`~tensordict.TensorDictBase.to_tensordict`, so the result is a
        :class:`~tensordict.TensorDict` and the file is not changed. See
        :meth:`~tensordict.TensorDictBase.flatten_keys`.

        Args:
            separator (str, optional): the separator between the parts of a
                nested key. Defaults to ``"."``.
            inplace (bool, optional): must be ``False``: ``True`` raises a
                ``ValueError``. Defaults to ``False``.
            is_leaf (Callable, optional): see
                :meth:`~tensordict.TensorDictBase.flatten_keys`. Defaults to
                ``None``.
        """
        if inplace:
            raise ValueError(
                "Cannot call flatten_keys in_place with a PersistentTensorDict."
            )
        return self.to_tensordict().flatten_keys(separator=separator, is_leaf=is_leaf)

    # Override: one to_tensordict() read, outside the cache that the base
    # keeps while locked: a locked file would keep a copy of its data, and
    # later calls would miss writes from other processes. inplace=True raises.
    @_as_context_manager()
    def unflatten_keys(
        self, separator: str = KEY_SEPARATOR, inplace: bool = False
    ) -> T:
        if inplace:
            raise ValueError(
                "Cannot call unflatten_keys in_place with a PersistentTensorDict."
            )
        return self.to_tensordict().unflatten_keys(separator=separator)

    # Override: changes the device without reading the file, and rejects the
    # in-place and pinned copies, which the file cannot hold.
    def _to(
        self,
        device,
        dtype,
        non_blocking,
        convert_to_format,
        batch_size,
        non_blocking_pin,
        num_threads,
        inplace,
        args,
        kwargs,
    ):
        if inplace:
            raise TypeError(f"Cannot use inplace=True with {type(self).__name__}.to().")
        if non_blocking_pin:
            raise RuntimeError(
                f"Cannot use non_blocking_pin=True {type(self).__name__}.to(). Call "
                f"`to_tensordict()` before executing this code."
            )
        if convert_to_format is not None and dtype is None:
            warnings.warn(
                f"{type(self).__name__}.to(memory_format=...) without a dtype "
                "ignores the memory format. Starting with TensorDict 0.17, the "
                "default will change to returning a TensorDict copy in that "
                "memory format, as to(dtype, memory_format=...) does, which "
                "raises where torch.Tensor.to raises. Call "
                "to_tensordict().to(memory_format=...) to get the new "
                "behaviour now.",
                FutureWarning,
                # skip to() and the _as_context_manager wrapper
                stacklevel=4,
            )
            convert_to_format = None
        if (
            device is not None
            and dtype is None
            and convert_to_format is None
            and device != self.device
        ):
            result = self.clone(False)
            result._device = device
            for key, nested in list(result._nested_tensordicts.items()):
                result._nested_tensordicts[key] = nested.to(device)
            result._link_nested()
            if batch_size is not None:
                result.batch_size = batch_size
            return result
        return super()._to(
            device,
            dtype,
            non_blocking,
            convert_to_format,
            batch_size,
            non_blocking_pin,
            num_threads,
            inplace,
            args,
            kwargs,
        )

    def _to_numpy(self, value):
        if hasattr(value, "requires_grad") and value.requires_grad:
            raise RuntimeError("Cannot set a tensor that has requires_grad=True.")
        if isinstance(value, torch.Tensor):
            if _is_unbatched(value):
                # the file has no place for the class (#3264)
                _warn_unbatched_saved_as_plain()
            out = value.cpu().detach().numpy()
        elif isinstance(value, dict):
            out = TensorDict(value, [])
        elif is_non_tensor(value):
            try:
                data = value.data
            except AttributeError:
                # a stack of different values: write one per element
                out = self._backend.encode_non_tensor_stack(
                    value.tolist(), value.batch_size
                )
            else:
                out = self._backend.encode_non_tensor(data)
        elif is_tensor_collection(value):
            out = value
        elif isinstance(value, (np.ndarray,)):
            out = value
        else:
            raise NotImplementedError(
                f"Cannot set values of type {value} in a PersistentTensorDict."
            )
        return out

    def _set(
        self,
        key: NestedKey,
        value: Any,
        *,
        inplace: bool = False,
        idx=None,
        validated: bool = False,
        ignore_lock: bool = False,
        non_blocking: bool = False,
    ) -> PersistentTensorDict:
        if not validated:
            value = self._validate_value(value, check_shape=idx is None, key=key)
        non_tensor = is_non_tensor(value)
        value = self._to_numpy(value)
        if not inplace:
            if idx is not None:
                raise RuntimeError("Cannot pass an index to _set when inplace=False.")
            elif self.is_locked and not ignore_lock:
                raise RuntimeError(_LOCK_ERROR)
        # shortcut set if we're placing a tensordict
        key = _unravel_key_to_tuple(key)
        first_key, subkey = key[0], key[1:]
        if is_tensor_collection(value):
            target_td = self._get_str(first_key, default=None)
            if target_td is None:
                metadata = None
                if self._backend.metadata_on_create:
                    metadata = _attrs_metadata(value.batch_size, _source_names(value))
                self._backend.create_group(self.file, first_key, metadata)
                target_td = self._get_str(first_key, default=NO_DEFAULT)
                target_td.batch_size = value.batch_size
                if value._has_names():
                    target_td.names = value.names
            elif not is_tensor_collection(target_td):
                raise RuntimeError(
                    f"cannot set a tensor collection in place of a non-tensor collection in {type(self).__name__}. "
                    f"Got self.get({first_key})={target_td} and value={value}."
                )
            if idx is None:
                if len(subkey):
                    target_td.set(subkey, value, inplace=inplace)
                else:
                    target_td.update(value, inplace=inplace)
            else:
                if len(subkey):
                    target_td.set_at_(subkey, value, idx=idx)
                else:
                    target_td.update_at_(value, idx=idx)

            return self

        if inplace:
            # could be called before but will go under further refactoring of set
            key = self._process_key(key)
            array = self.file[key]
            stack = non_tensor and (
                isinstance(value, _NonTensorStackArray)
                or self._backend.non_tensor_stack_shape(array) is not None
            )
            if stack and idx is not None:
                raise RuntimeError(
                    f"Cannot write {key!r} in place: a {type(self).__name__} "
                    "writes a stack of different non-tensor values as a whole. "
                    "Set the entry with inplace=False instead."
                )
            elif stack:
                # a stack is written as a whole: replace the array
                self._backend.delete(self.file, key)
                self._backend.try_create_dataset(
                    self.file, key, value, self._dataset_kwargs
                )
            else:
                if idx is None:
                    idx = ()
                else:
                    idx = self._process_index(idx, array)
                self._backend.write_at(array, idx, value)
        else:
            key = self._process_key(key)
            if not self._backend.try_create_dataset(
                self.file, key, value, self._dataset_kwargs
            ):
                warnings.warn(
                    "Replacing an array with another one is inefficient. "
                    "Consider using different names or populating in-place using `inplace=True`."
                )
                self._backend.delete(self.file, key)
                self._backend.try_create_dataset(
                    self.file, key, value, self._dataset_kwargs
                )
            # If we have a nested key, let's make sure we have the corresponding TD registered
            if subkey:
                self._get_tuple((first_key, *subkey[:-1]), default=NO_DEFAULT)
        return self

    # Override: asks the file, without listing the keys.
    def _contains_str(self, key):
        return self._backend.contains(self.file, self._process_key(key))

    def _set_str(
        self,
        key: str,
        value: Any,
        *,
        inplace: bool,
        validated: bool,
        ignore_lock: bool = False,
        non_blocking: bool = False,
    ):
        inplace = self._convert_inplace(inplace, key)
        return self._set(
            key,
            value,
            inplace=inplace,
            validated=validated,
            ignore_lock=ignore_lock,
            non_blocking=non_blocking,
        )

    def _set_tuple(self, key, value, *, inplace, validated, non_blocking):
        key = _unravel_key_to_tuple(key)
        if len(key) == 1:
            return self._set_str(
                key[0],
                value,
                inplace=inplace,
                validated=validated,
                non_blocking=non_blocking,
            )
        elif key[0] in self.keys():
            return self._get_str(key[0], NO_DEFAULT)._set_tuple(
                key[1:],
                value,
                inplace=inplace,
                validated=validated,
                non_blocking=non_blocking,
            )
        # the path in the file, which a KeyError shows
        inplace = self._convert_inplace(inplace, self._process_key(key))
        return self._set(
            key, value, inplace=inplace, validated=validated, non_blocking=non_blocking
        )

    def _set_at_str(self, key, value, idx, *, validated, non_blocking):
        return self._set(
            key,
            value,
            inplace=True,
            idx=idx,
            validated=validated,
            non_blocking=non_blocking,
        )

    def _set_at_tuple(self, key, value, idx, *, validated, non_blocking):
        return self._set(
            key,
            value,
            inplace=True,
            idx=idx,
            validated=validated,
            non_blocking=non_blocking,
        )

    def _set_metadata(self, orig_metadata_container: PersistentTensorDict):
        for key, td in orig_metadata_container._nested_tensordicts.items():
            array = self._get_array(key)
            self._nested_tensordicts[key] = PersistentTensorDict(
                group=array,
                batch_size=td.batch_size,
                device=td.device,
                backend=self.backend,
                non_tensor_as_stack=self._non_tensor_as_stack,
                allow_pickle=self._backend.allow_pickle,
                **self._dataset_kwargs,
            )
            self._nested_tensordicts[key]._backend = self._backend
            self._nested_tensordicts[key].names = td._td_dim_names
            self._nested_tensordicts[key]._set_metadata(td)
        self._link_nested()

    def _clone(self, recurse: bool = True, newfile=None) -> PersistentTensorDict:
        if recurse:
            # this should clone the data to a new location indicated by newfile
            if newfile is None:
                warnings.warn(
                    "A destination should be provided when cloning a "
                    "PersistentTensorDict. A temporary file will be used "
                    "instead. Use `recurse=False` to keep track of the original data "
                    "with a new PersistentTensorDict instance."
                )
                tmpfile = tempfile.NamedTemporaryFile(delete=False)
                newfile = tmpfile.name
                tmpfile.close()  # Close file handle before the backend opens it
            f_dest = self._backend.open_new_file(newfile)
            f_src = self.file
            # is_leaf=_is_leaf_nontensor ensures non-tensor entries are copied too
            for key in self.keys(
                include_nested=True, leaves_only=True, is_leaf=_is_leaf_nontensor
            ):
                key = self._process_key(key)
                self._backend.copy_dataset(
                    f_dest, key, f_src[key], self._dataset_kwargs
                )
            # create a non-recursive copy and update the file
            # this way, we can keep the batch-size of every nested tensordict
            clone = self.clone(False)
            clone.file = f_dest
            clone.filename = newfile
            clone._pin_mem = False
            names = self._td_dim_names
            if names is not None:
                names = list(names)
            clone.names = names
            clone._nested_tensordicts = {}
            clone._set_metadata(self)
            clone._write_attrs_metadata()
            clone._backend.finalize(clone.file)
            return clone
        else:
            # we need to keep the batch-size of nested tds, which we do manually
            nested_tds = {
                key: td.clone(False) for key, td in self._nested_tensordicts.items()
            }
            filename = self.filename
            file = self.file if filename is None else None
            clone = PersistentTensorDict(
                filename=filename,
                group=file,
                mode=self.mode,
                backend=self.backend,
                device=self.device,
                batch_size=self.batch_size,
                non_tensor_as_stack=self._non_tensor_as_stack,
                allow_pickle=self._backend.allow_pickle,
                **self._dataset_kwargs,
            )
            clone._nested_tensordicts = nested_tds
            clone._link_nested()
            clone._pin_mem = False
            clone.names = self._td_dim_names
            return clone

    # Override: replaces the open file with what its backend can pickle.
    def __getstate__(self):
        state = self.__dict__.copy()
        backend_state = self._backend.getstate(state["file"])
        state["file"] = None
        state.update(backend_state)
        # a weakref cannot be pickled: the parent links its children back
        state["_nested_parent"] = None
        state["__lock_parents_weakrefs"] = None
        # as TensorDictBase.__getstate__ does: the last op can hold weakrefs,
        # and the cache is rebuilt on demand (gh-2920)
        state["_last_op"] = None
        state["_cache"] = None
        return state

    # Override: reopens the file, and reads the pickles of TensorDict 0.14.
    def __setstate__(self, state):
        backend = state["_backend"]
        state["file"] = backend.setstate(state)
        state.pop("group_name", None)
        # Pickles made with TensorDict 0.14 and earlier store the dataset
        # options under "kwargs".
        if "kwargs" in state:
            state["_dataset_kwargs"] = state.pop("kwargs")
        self.__dict__.update(state)
        self._link_nested()
        if self._is_locked:
            # this can cause avoidable overhead, as we will be locking the leaves
            # then locking their parent, and the parent of the parent, every
            # time re-locking tensordicts that have already been locked.
            # To avoid this, we should lock only at the root, but it isn't easy
            # to spot what the root is...
            self._is_locked = False
            self.lock_()


_register_tensor_class(PersistentTensorDict)


def _set_max_batch_size(source: PersistentTensorDict):
    """Updates a tensordict with its maximium batch size."""
    tensor_data = list(source._items_metadata())
    for key, val in tensor_data:
        if not val.get("non_tensor", None) and not val.get("array", None):
            _set_max_batch_size(source.get(key, None))

    batch_size = []
    if not tensor_data:  # when source is empty
        source.batch_size = batch_size
        return

    curr_dim = 0
    # We need to reload this list because the value have changed
    tensor_data = list(source._items_metadata())
    tensor_keys, tensor_data = zip(*tensor_data)
    # Filter out the non-tensor data, except the stacks, which have a shape
    tensor_data = [
        data for data in tensor_data if not data.get("non_tensor") or "shape" in data
    ]
    while True:
        if tensor_data[0]["dim"] > curr_dim:
            curr_dim_size = tensor_data[0]["shape"][curr_dim]
        else:
            source.batch_size = batch_size
            return
        for tensor in tensor_data[1:]:
            if tensor["dim"] <= curr_dim or tensor["shape"][curr_dim] != curr_dim_size:
                source.batch_size = batch_size
                return
        batch_size.append(curr_dim_size)
        curr_dim += 1


def _source_names(value, names=None):
    """The names that a group written from ``value`` is expected to persist."""
    if names is None and is_tensor_collection(value) and value._has_names():
        names = value.names
    return names


def _read_rows(sub_td: TensorDictBase) -> TensorDict:
    """Reads an index of a PersistentTensorDict into a TensorDict.

    Gives what ``sub_td.to_tensordict()`` gives, without copying the tensors
    that were just read from the file.
    """
    return TensorDict(
        {
            key: (
                _read_rows(value)
                if is_tensor_collection(value) and not is_non_tensor(value)
                else value
            )
            for key, value in sub_td.items(is_leaf=_is_leaf_nontensor)
        },
        device=sub_td.device,
        batch_size=sub_td.batch_size,
        names=sub_td._maybe_names(),
    )


def _is_non_tensor_h5(val):
    import h5py

    dt = val.dtype
    if (
        h5py.check_string_dtype(dt)
        or h5py.check_vlen_dtype(dt)
        or h5py.check_enum_dtype(dt)
        or h5py.check_opaque_dtype(dt)
    ):
        return True
    return False


def _is_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_basic_index(idx) -> bool:
    """Whether ``idx`` only holds integers, positive-step slices and at most one Ellipsis.

    Storage libraries read such indices as hyperslabs, with the same result as torch.
    """
    items = idx if isinstance(idx, tuple) else (idx,)
    num_ellipsis = 0
    for item in items:
        if item is Ellipsis:
            num_ellipsis += 1
        elif isinstance(item, slice):
            if not all(
                bound is None or _is_int(bound)
                for bound in (item.start, item.stop, item.step)
            ):
                return False
            if item.step is not None and item.step <= 0:
                return False
        elif not _is_int(item):
            return False
    return num_ellipsis <= 1


def _index_span(idx, size: int) -> tuple[int, int, Tensor] | None:
    """Returns ``(start, stop, local_idx)`` such that ``array[start:stop][local_idx]`` equals ``array[idx]``.

    Only 1-D integer indices and 1-D boolean masks along the first dimension are
    supported; ``None`` is returned for any other index, or an out-of-bounds one.
    """
    if isinstance(idx, tuple) and len(idx) == 1:
        # sub-tensordicts store their index as a tuple
        idx = idx[0]
    if isinstance(idx, (list, range)):
        try:
            idx = torch.as_tensor(idx)
        except (TypeError, ValueError, RuntimeError):
            return None
    elif isinstance(idx, np.ndarray):
        idx = torch.from_numpy(idx)
    elif not isinstance(idx, Tensor):
        return None
    if idx.ndim != 1 or not idx.numel():
        return None
    if idx.dtype is torch.bool:
        if idx.shape[0] != size:
            return None
        idx = idx.nonzero().squeeze(-1)
        if not idx.numel():
            return None
    elif idx.is_floating_point() or idx.is_complex() or idx.dtype is torch.uint8:
        # uint8 tensors are masks for torch
        return None
    idx = idx.to(device="cpu", dtype=torch.long)
    if idx.min() < -size or idx.max() >= size:
        return None
    idx = torch.where(idx < 0, idx + size, idx)
    start = int(idx.min())
    stop = int(idx.max()) + 1
    return start, stop, idx - start

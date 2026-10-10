# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The lazily stacked store: :class:`LazyStackedTensorDictStore` and its element view.

The dense store, :class:`TensorDictStore`, is in :mod:`tensordict.store._store`,
which re-exports the classes of this module.
"""

from __future__ import annotations

import asyncio
import json
import struct
import threading
import uuid
import weakref
from numbers import Number
from typing import Any, Callable, Sequence, Tuple, Type

import torch
from tensordict._deprecation import deprecated
from tensordict._indexing import _as_tuple, _getitem_batch_size, convert_ellipsis_to_idx
from tensordict._td import (
    _TensorDictKeysView,
    _unravel_key_to_tuple,
    CompatibleType,
    NO_DEFAULT,
    TensorDict,
)
from tensordict.base import (
    _register_tensor_class,
    is_tensor_collection,
    T,
    TensorDictBase,
)
from tensordict.store._store import _has_redis, _KEY_SEP, Self, STORE_BACKENDS
from tensordict.store._utils import (
    _bytes_to_tensor,
    _check_indexed_value,
    _compute_byte_ranges,
    _compute_covering_range,
    _decode_meta,
    _dtype_to_str,
    _get_local_idx,
    _getitem_result_shape,
    _is_scattered_index,
    _LUA_GETRANGES,
    _LUA_SETRANGES,
    _normalize_index,
    _prepare_indexed_value,
    _str_to_dtype,
    _tensor_to_bytes,
)
from tensordict.utils import (
    _as_context_manager,
    _erase_cache_first,
    _KEY_ERROR,
    _lock_blocked,
    _LOCK_ERROR,
    expand_as_right,
    NestedKey,
    unravel_key,
)

# ---------------------------------------------------------------------------
# _StoreStackElementView — write-through view for a single stack element
# ---------------------------------------------------------------------------


class _StoreStackElementView(TensorDictBase):
    """Write-through view of one element in a :class:`LazyStackedTensorDictStore`.

    Returned by ``redis_lazy_stack[int]``.  All reads and writes go through
    the parent's Redis connection so that mutations propagate.  This class is
    private — users interact with it via the normal ``TensorDictBase`` API.
    """

    _td_dim_names = None

    def __init__(self, parent, element_idx: int):
        self._parent = parent
        self._element_idx = _normalize_index(element_idx, parent._count)

        self._locked_tensordicts = []
        self._lock_id = set()
        self._is_shared = False
        self._is_memmap = False

        self._batch_size = parent._inner_batch_size
        self._device = parent._device

    # ---- helpers that delegate to the parent ----

    def _run_sync(self, coro):
        return self._parent._run_sync(coro)

    def _get_all_keys(self) -> set[str]:
        return self._parent._get_all_keys()

    # ---- TensorDictBase interface ----

    @property
    def batch_size(self) -> torch.Size:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = torch.Size(value)

    @property
    def device(self) -> torch.device | None:
        return self._device

    @device.setter
    def device(self, value):
        self._device = torch.device(value) if value is not None else None

    def _rename_subtds(self, names):
        pass

    # ---- reads ----

    def _get_str(self, key, default=NO_DEFAULT, **kwargs):
        all_keys = self._get_all_keys()
        key_path = key

        # Nested prefix
        prefix_check = key_path + _KEY_SEP
        nested_keys = [k for k in all_keys if k.startswith(prefix_check)]
        if nested_keys:
            result = self._run_sync(
                self._parent._abatch_get_element_keys(self._element_idx, nested_keys)
            )
            source: dict = {}
            for kp, tensor in result.items():
                rel = kp[len(prefix_check) :]
                parts = rel.split(_KEY_SEP)
                d = source
                for part in parts[:-1]:
                    d = d.setdefault(part, {})
                d[parts[-1]] = tensor
            return TensorDict(source, batch_size=self._batch_size, device=self._device)

        if key_path in all_keys:
            result = self._run_sync(
                self._parent._abatch_get_element_keys(self._element_idx, [key_path])
            )
            t = result.get(key_path)
            if t is not None:
                return t

        if default is not NO_DEFAULT:
            return default
        raise KeyError(f"key {key} not found in {type(self).__name__}")

    def _get_at_str(self, key, idx, default=NO_DEFAULT, **kwargs):
        tensor = self._get_str(key, default=default, **kwargs)
        if tensor is default:
            return default
        return tensor[idx]

    def _get_at_tuple(self, key, idx, default=NO_DEFAULT, **kwargs):
        key = _unravel_key_to_tuple(key)
        if len(key) == 1:
            return self._get_at_str(key[0], idx, default=default, **kwargs)
        first = self._get_str(key[0], default, **kwargs)
        if first is default:
            return default
        return first._get_at_tuple(key[1:], idx, default=default, **kwargs)

    # ---- writes ----

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
        if not validated:
            value = self._validate_value(value, check_shape=True, key=key)
        if self.is_locked and not ignore_lock and not inplace:
            raise RuntimeError(_LOCK_ERROR)

        if is_tensor_collection(value):
            for sub_key in value.keys(include_nested=True, leaves_only=True):
                sub_tuple = _unravel_key_to_tuple(sub_key)
                full_kp = key + _KEY_SEP + _KEY_SEP.join(sub_tuple)
                self._run_sync(
                    self._parent._aset_element_key(
                        self._element_idx, full_kp, value.get(sub_key)
                    )
                )
            return self

        if isinstance(value, torch.Tensor):
            self._run_sync(
                self._parent._aset_element_key(self._element_idx, key, value)
            )
            return self

        try:
            value = torch.as_tensor(value)
            self._run_sync(
                self._parent._aset_element_key(self._element_idx, key, value)
            )
        except (ValueError, TypeError):
            raise TypeError(
                f"{type(self).__name__} only supports tensor values, got {type(value)}"
            )
        return self

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
        key_path = _KEY_SEP.join(key)
        if not validated:
            value = self._validate_value(value, check_shape=True, key=key)
        if self.is_locked and not inplace:
            raise RuntimeError(_LOCK_ERROR)
        if isinstance(value, torch.Tensor):
            self._run_sync(
                self._parent._aset_element_key(self._element_idx, key_path, value)
            )
        elif is_tensor_collection(value):
            for sub_key in value.keys(include_nested=True, leaves_only=True):
                sub_tuple = _unravel_key_to_tuple(sub_key)
                full_kp = key_path + _KEY_SEP + _KEY_SEP.join(sub_tuple)
                self._run_sync(
                    self._parent._aset_element_key(
                        self._element_idx, full_kp, value.get(sub_key)
                    )
                )
        else:
            self._run_sync(
                self._parent._aset_element_key(
                    self._element_idx, key_path, torch.as_tensor(value)
                )
            )
        return self

    def __setitem__(self, index, value):
        index_unravel = _unravel_key_to_tuple(index)
        if index_unravel:
            return self.set(index_unravel, value, inplace=True)

        _check_indexed_value(self, value)
        if isinstance(value, Number):
            # a scalar is written to every entry, which casts it to its dtype
            items = [
                (key, value) for key in self.keys(include_nested=True, leaves_only=True)
            ]
        else:
            if not isinstance(value, TensorDictBase):
                value = TensorDict.from_dict(value, batch_size=[])
            items = [
                (key, value.get(key))
                for key in value.keys(include_nested=True, leaves_only=True)
            ]

        for key, item in items:
            key_tuple = _unravel_key_to_tuple(key)
            key_path = _KEY_SEP.join(key_tuple)
            existing = (
                self._get_str(key_tuple[0]) if len(key_tuple) == 1 else self.get(key)
            )
            existing[index] = item
            self._run_sync(
                self._parent._aset_element_key(self._element_idx, key_path, existing)
            )

    def _index_tensordict(self, index):
        return self.to_tensordict()[index]

    def _set_at_str(self, key, value, idx, *, validated, non_blocking):
        # Read full element tensor, patch locally, write back
        tensor = self._get_str(key)
        tensor[idx] = value
        self._run_sync(self._parent._aset_element_key(self._element_idx, key, tensor))
        return self

    def _set_at_tuple(self, key, value, idx, *, validated, non_blocking):
        key = _unravel_key_to_tuple(key)
        if len(key) == 1:
            return self._set_at_str(
                key[0], value, idx, validated=validated, non_blocking=non_blocking
            )
        key_path = _KEY_SEP.join(key)
        tensor = self._get_str(key[0])
        if is_tensor_collection(tensor):
            tensor._set_at_tuple(
                key[1:], value, idx, validated=validated, non_blocking=non_blocking
            )
            return self
        tensor[idx] = value
        self._run_sync(
            self._parent._aset_element_key(self._element_idx, key_path, tensor)
        )
        return self

    def _convert_inplace(self, inplace, key):
        if inplace is not False:
            all_keys = self._get_all_keys()
            has_key = key in all_keys or any(
                k.startswith(key + _KEY_SEP) for k in all_keys
            )
            if inplace is True and not has_key:
                raise KeyError(
                    _KEY_ERROR.format(key, type(self).__name__, sorted(self.keys()))
                )
            inplace = has_key
        return inplace

    # ---- keys ----

    def keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
        *,
        sort: bool = False,
    ) -> _LazyStackedStoreKeysView:
        return _LazyStackedStoreKeysView(
            tensordict=self,
            include_nested=include_nested,
            leaves_only=leaves_only,
            is_leaf=is_leaf,
            sort=sort,
        )

    @_lock_blocked
    def del_(self, key: NestedKey) -> _StoreStackElementView:
        raise RuntimeError(
            "Cannot delete keys from a stack element view. "
            "Delete from the parent LazyStackedTensorDictStore instead."
        )

    def rename_key_(self, old_key, new_key, safe=False):
        raise RuntimeError(
            "Cannot rename keys on a stack element view. "
            "Rename on the parent LazyStackedTensorDictStore instead."
        )

    def entry_class(self, key: NestedKey) -> type:
        return self._parent.entry_class(key)

    # ---- locking ----

    def _propagate_lock(self, lock_parents_weakrefs=None, *, is_compiling):
        self._is_locked = True

    @_erase_cache_first
    def _propagate_unlock(self):
        self._is_locked = False
        self._is_shared = False
        self._is_memmap = False
        return []

    # ---- materialization ----

    def to_tensordict(self, *, retain_none: bool | None = None) -> TensorDict:
        result_map = self._run_sync(self._parent._abatch_get_element(self._element_idx))
        source: dict = {}
        for kp, tensor in result_map.items():
            parts = kp.split(_KEY_SEP)
            d = source
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = tensor
        return TensorDict(source, batch_size=self._batch_size, device=self._device)

    @deprecated(
        "LazyStackedTensorDictStore[i].to_local()",
        removal="0.17",
        replacement="to_tensordict()",
    )
    def to_local(self) -> TensorDict:
        return self.to_tensordict()

    def contiguous(self, *, canonical: bool = False) -> TensorDict:
        out = self.to_tensordict()
        if canonical:
            out = out.contiguous(canonical=True)
        return out

    def is_contiguous(self) -> bool:
        return False

    def detach_(self) -> Self:
        return self

    @_lock_blocked
    def popitem(self) -> Tuple[NestedKey, CompatibleType]:
        raise RuntimeError("Cannot popitem from a stack element view.")

    def _change_batch_size(self, new_size: torch.Size) -> None:
        self._batch_size = new_size

    def zero_(self) -> Self:
        for key in self.keys():
            self.fill_(key, 0)
        return self

    def fill_(self, key: NestedKey, value: float | bool) -> TensorDictBase:
        existing = self.get(key)
        if is_tensor_collection(existing):
            for subkey in existing.keys():
                existing.fill_(subkey, value)
        else:
            existing = existing.fill_(value)
            self.set_(key, existing)
        return self

    def empty(
        self, recurse=False, *, batch_size=None, device=NO_DEFAULT, names=None
    ) -> T:
        return TensorDict(
            {},
            device=self.device if device is NO_DEFAULT else device,
            batch_size=self.batch_size if batch_size is None else batch_size,
            names=self.names if names is None and self._has_names() else names,
        )

    def masked_fill(self, mask, value):
        return self.to_tensordict().masked_fill(mask, value)

    def masked_fill_(self, mask, value):
        for key in self.keys(include_nested=True, leaves_only=True):
            tensor = self.get(key)
            tensor = tensor.masked_fill(expand_as_right(mask, tensor), value)
            self.set_(key, tensor)
        return self

    def masked_select(self, mask):
        return self.to_tensordict().masked_select(mask)

    def where(self, condition, other, *, out=None, pad=None, update_batch_size=False):
        return self.to_tensordict().where(
            condition=condition,
            other=other,
            out=out,
            pad=pad,
            update_batch_size=update_batch_size,
        )

    # ---- pickling: materialize, don't try to serialize the view ----

    def __reduce__(self):
        return (TensorDict, (), self.to_tensordict().__getstate__())

    def __repr__(self):
        keys_str = list(self.keys())
        return (
            f"_StoreStackElementView(\n"
            f"    parent_td_id={self._parent._td_id!r},\n"
            f"    element_idx={self._element_idx},\n"
            f"    keys={keys_str},\n"
            f"    batch_size={self.batch_size},\n"
            f"    device={self.device})"
        )

    # ---- not-supported stubs (same as parent) ----

    @classmethod
    def from_dict(cls, *args, **kwargs):
        raise NotImplementedError(f"{cls.__name__} cannot be created from a dict.")

    def _clone(self, recurse=True):
        return self.to_tensordict()

    def _view(self, *a, **kw):
        raise RuntimeError(f"Cannot call `view` on a {type(self).__name__}.")

    def _transpose(self, dim0, dim1):
        raise RuntimeError(f"Cannot call `transpose` on a {type(self).__name__}.")

    def _permute(self, *a, **kw):
        raise RuntimeError(f"Cannot call `permute` on a {type(self).__name__}.")

    def _squeeze(self, dim=None):
        raise RuntimeError(f"Cannot call `squeeze` on a {type(self).__name__}.")

    def _unsqueeze(self, dim):
        raise RuntimeError(f"Cannot call `unsqueeze` on a {type(self).__name__}.")

    def chunk(self, chunks, dim=0):
        return self.to_tensordict().chunk(chunks, dim)

    def share_memory_(self):
        raise NotImplementedError(
            f"Cannot call share_memory_ on a {type(self).__name__}."
        )

    def _memmap_(self, **kw):
        raise RuntimeError(f"Cannot call memmap on a {type(self).__name__}.")

    def make_memmap(self, key, shape, *, dtype=None, robust_key=True):
        raise RuntimeError(f"Cannot make memmap on a {type(self).__name__}.")

    def make_memmap_from_storage(
        self, key, storage, shape, *, dtype=None, robust_key=True
    ):
        raise RuntimeError(f"Cannot make memmap on a {type(self).__name__}.")

    def make_memmap_from_tensor(self, key, tensor, *, copy_data=True, robust_key=True):
        raise RuntimeError(f"Cannot make memmap on a {type(self).__name__}.")

    def memmap_(self, prefix=None, copy_existing=False, num_threads=0):
        """Raises a ``RuntimeError``: a stack element view cannot be memory-mapped in-place.

        Args:
            prefix (str, optional): unused.
            copy_existing (bool, optional): unused.
            num_threads (int, optional): unused.

        See :meth:`~tensordict.TensorDictBase.memmap_`.
        """
        raise RuntimeError(f"Cannot call memmap_ on a {type(self).__name__}.")

    def pin_memory(self, *a, **kw):
        raise RuntimeError(f"Cannot pin memory of a {type(self).__name__}.")

    def _add_batch_dim(self, *, in_dim, vmap_level):
        raise RuntimeError(f"{type(self).__name__} cannot be used with vmap.")

    def _remove_batch_dim(self, vmap_level, batch_size, out_dim): ...

    def _maybe_remove_batch_dim(self, funcname, vmap_level, batch_size, out_dim): ...

    def _select(self, *keys, inplace=False, strict=True, set_shared=True):
        raise NotImplementedError(f"Cannot call select on a {type(self).__name__}.")

    def _exclude(self, *keys, inplace=False, set_shared=True):
        raise NotImplementedError(f"Cannot call exclude on a {type(self).__name__}.")

    @_as_context_manager()
    def flatten_keys(self, separator=".", inplace=False):
        """Returns an in-memory copy where the nested keys are joined by ``separator``.

        The content is first loaded with :meth:`~tensordict.TensorDictBase.to_tensordict`.

        Args:
            separator (str, optional): the separator between the nested items. Defaults to ``"."``.
            inplace (bool, optional): ignored; a new tensordict is always returned.
                Defaults to ``False``.

        See :meth:`~tensordict.TensorDictBase.flatten_keys`.
        """
        return self.to_tensordict().flatten_keys(separator=separator)

    @_as_context_manager()
    def unflatten_keys(self, separator=".", inplace=False):
        return self.to_tensordict().unflatten_keys(separator=separator)

    _load_memmap = TensorDict._load_memmap

    def _set_non_tensor(self, key, value):
        raise NotImplementedError(
            f"set_non_tensor is not compatible with {type(self).__name__}."
        )

    def _stack_onto_(self, list_item, dim):
        raise RuntimeError(f"Cannot call _stack_onto_ on a {type(self).__name__}.")


# ---------------------------------------------------------------------------
# LazyStackedTensorDictStore — lazy-stack storage in Redis
# ---------------------------------------------------------------------------

# Upload chunk size: number of stack elements processed per pipeline command.
_UPLOAD_CHUNK = 10_000


class _LazyStackedStoreKeysView(_TensorDictKeysView):
    """Keys view for LazyStackedTensorDictStore."""

    def __iter__(self):
        td = self.tensordict
        all_keys = td._get_all_keys()
        seen = set()
        for full_key in all_keys:
            parts = full_key.split(_KEY_SEP)
            if self.include_nested:
                key = tuple(parts) if len(parts) > 1 else parts[0]
                if self.leaves_only and len(parts) > 1:
                    if key not in seen:
                        seen.add(key)
                        yield key
                elif not self.leaves_only or len(parts) == 1:
                    if key not in seen:
                        seen.add(key)
                        yield key
            else:
                top_key = parts[0]
                if top_key in seen:
                    continue
                seen.add(top_key)
                is_leaf_key = len(parts) == 1
                if self.leaves_only and not is_leaf_key:
                    continue
                yield top_key

    def __contains__(self, key):
        key = unravel_key(key)
        td = self.tensordict
        if isinstance(key, str):
            full_key = key
        else:
            full_key = _KEY_SEP.join(key)
        all_keys = td._get_all_keys()
        if full_key in all_keys:
            return True
        prefix_check = full_key + _KEY_SEP
        return any(k.startswith(prefix_check) for k in all_keys)

    def __len__(self):
        return sum(1 for _ in self)


class LazyStackedTensorDictStore(TensorDictBase):
    """A LazyStackedTensorDict backed by a key-value store.

    Supports `Redis <https://redis.io>`_, `Dragonfly <https://dragonflydb.io>`_,
    `KeyDB <https://docs.keydb.dev>`_, and any other Redis-wire-compatible
    server.

    Stores each leaf key as a **single concatenated blob** in the store,
    regardless of how many stack elements there are.  For *N* elements and
    *K* leaf keys this uses only *O(K)* keys (plus offset tables for
    heterogeneous shapes).

    Two storage modes are supported:

    * **Homogeneous** — all stack elements have the same shape per key.
      Byte offsets are computed arithmetically (no offset table stored).
    * **Heterogeneous** — element shapes may differ per key.  An offset
      table (packed int64 array of *N+1* byte offsets) is stored alongside
      the data blob, and per-element shapes are persisted in the metadata
      hash.

    Keyword Args:
        backend (str): Store backend (``"redis"``, ``"dragonfly"``, etc.).
        host (str): Server hostname.  Defaults to ``"localhost"``.
        port (int): Server port.  Defaults to ``6379``.
        db (int): Database number.  Defaults to ``0``.
        unix_socket_path (str, optional): Unix domain socket path.
        prefix (str): Key namespace.  Defaults to ``"tensordict"``.
        count (int): Number of stack elements (*N*).
        stack_dim (int): Dimension along which the stack was performed.
        inner_batch_size (Sequence[int]): Batch size of each element.
        device (torch.device, optional): Device for retrieved tensors.
        client: Existing ``redis.asyncio.Redis`` client.
        td_id (str, optional): UUID for reconnecting to existing data.
        cache_metadata (bool): Cache (shape, dtype) locally.
        **redis_kwargs: Extra connection keyword arguments.
    """

    _td_dim_names = None

    def __init__(
        self,
        *,
        backend: STORE_BACKENDS = "redis",
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        unix_socket_path: str | None = None,
        prefix: str = "tensordict",
        count: int,
        stack_dim: int = 0,
        inner_batch_size: Sequence[int],
        device=None,
        client=None,
        td_id: str | None = None,
        cache_metadata: bool = True,
        **redis_kwargs,
    ):
        if not _has_redis:
            raise ModuleNotFoundError(
                "Could not import redis. Install it with: pip install redis"
            )
        import redis.asyncio as aioredis

        self._locked_tensordicts = []
        self._lock_id = set()
        self._is_shared = False
        self._is_memmap = False

        self._backend = backend

        self._cache_metadata = cache_metadata
        self._meta_cache: dict[str, tuple[list[int], torch.dtype]] | None = (
            {} if cache_metadata else None
        )

        self._td_id = td_id or str(uuid.uuid4())
        self._namespace = prefix

        self._count = count
        self._stack_dim = stack_dim
        self._inner_batch_size = torch.Size(inner_batch_size)

        bs = list(inner_batch_size)
        bs.insert(stack_dim, count)
        self._batch_size = torch.Size(bs)

        self._host = host
        self._port = port
        self._db = db
        self._unix_socket_path = unix_socket_path
        self._redis_kwargs = redis_kwargs

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._owns_loop = True

        if client is not None:
            self._client = client
        else:
            connect_kwargs = dict(redis_kwargs)
            if unix_socket_path is not None:
                connect_kwargs["unix_socket_path"] = unix_socket_path
            else:
                connect_kwargs["host"] = host
                connect_kwargs["port"] = port
            connect_kwargs["db"] = db
            self._client = aioredis.Redis(**connect_kwargs)

        self._device = torch.device(device) if device is not None else None

        self._run_sync(self._apersist_global_metadata())

    # ---- sync bridge ----

    def _run_sync(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    # ---- Redis key helpers ----

    def _redis_key(self, suffix: str) -> str:
        return f"{self._namespace}:{{{self._td_id}}}:{suffix}"

    def _data_key(self, key_path: str) -> str:
        return self._redis_key(f"d:{key_path}")

    def _idx_key(self, key_path: str) -> str:
        """Redis key for the offset table of a leaf key."""
        return self._redis_key(f"d:{key_path}:idx")

    def _meta_key(self, key_path: str) -> str:
        return self._redis_key(f"m:{key_path}")

    @property
    def _keys_registry_key(self) -> str:
        return self._redis_key("__keys__")

    # ---- async metadata persistence ----

    async def _apersist_global_metadata(self):
        pipe = self._client.pipeline()
        pipe.set(self._redis_key("__type__"), "lazy_stack")
        pipe.set(self._redis_key("__count__"), str(self._count))
        pipe.set(self._redis_key("__stack_dim__"), str(self._stack_dim))
        pipe.set(
            self._redis_key("__inner_batch_size__"),
            json.dumps(list(self._inner_batch_size)),
        )
        device_str = str(self._device) if self._device is not None else ""
        pipe.set(self._redis_key("__device__"), device_str)
        await pipe.execute()

    async def _aget_all_keys(self) -> set[str]:
        raw = await self._client.smembers(self._keys_registry_key)
        return {k.decode() if isinstance(k, bytes) else k for k in raw}

    def _get_all_keys(self) -> set[str]:
        return self._run_sync(self._aget_all_keys())

    async def _aget_metadata_batch(
        self, key_paths: list[str]
    ) -> dict[str, tuple[list[int], torch.dtype]]:
        result: dict[str, tuple[list[int], torch.dtype]] = {}
        uncached: list[str] = []
        for kp in key_paths:
            if self._meta_cache is not None and kp in self._meta_cache:
                result[kp] = self._meta_cache[kp]
            else:
                uncached.append(kp)
        if uncached:
            pipe = self._client.pipeline()
            for kp in uncached:
                pipe.hgetall(self._meta_key(kp))
            raw_metas = await pipe.execute()
            for kp, raw_meta in zip(uncached, raw_metas):
                meta = _decode_meta(raw_meta)
                shape = json.loads(meta["shape"])
                dtype = _str_to_dtype(meta["dtype"])
                result[kp] = (shape, dtype)
                if self._meta_cache is not None:
                    self._meta_cache[kp] = (shape, dtype)
        return result

    # ---- key-level homogeneity check helpers ----

    def _is_key_homogeneous(self, meta: dict) -> bool:
        return meta.get("homogeneous", "1") == "1"

    def _row_bytes(self, shape: list[int], dtype: torch.dtype) -> int:
        """Byte size of one stack element for a homogeneous key."""
        elem_size = torch.tensor([], dtype=dtype).element_size()
        nbytes = elem_size
        for s in shape:
            nbytes *= s
        return nbytes

    # ---- streaming upload ----

    async def _aupload_lazy_stack(self, lazy_td):
        """Stream data from a LazyStackedTensorDict into Redis."""
        tds = lazy_td.tensordicts

        leaf_keys = sorted(lazy_td.keys(include_nested=True, leaves_only=True), key=str)

        for raw_key in leaf_keys:
            key_tuple = _unravel_key_to_tuple(raw_key)
            key_path = _KEY_SEP.join(key_tuple)

            # Gather per-element info to decide homogeneous vs heterogeneous
            first_tensor = tds[0].get(raw_key)
            first_shape = list(first_tensor.shape)
            first_dtype = first_tensor.dtype

            homogeneous = True
            for td in tds[1:]:
                t = td.get(raw_key)
                if list(t.shape) != first_shape or t.dtype != first_dtype:
                    homogeneous = False
                    break

            if homogeneous:
                await self._aupload_homogeneous_key(
                    key_path, raw_key, tds, first_shape, first_dtype
                )
            else:
                await self._aupload_heterogeneous_key(
                    key_path, raw_key, tds, first_dtype
                )

    async def _aupload_homogeneous_key(
        self,
        key_path: str,
        raw_key,
        tds: list,
        shape: list[int],
        dtype: torch.dtype,
    ):
        """Upload a single homogeneous leaf key with chunked SETRANGE."""
        row_bytes = self._row_bytes(shape, dtype)
        full_shape = [self._count] + shape
        N = len(tds)

        # Metadata
        meta = {
            "shape": json.dumps(full_shape),
            "dtype": _dtype_to_str(dtype),
            "homogeneous": "1",
        }
        pipe = self._client.pipeline()
        pipe.hset(self._meta_key(key_path), mapping=meta)
        pipe.sadd(self._keys_registry_key, key_path)
        await pipe.execute()

        if self._meta_cache is not None:
            self._meta_cache[key_path] = (full_shape, dtype)

        # Stream data in chunks
        for chunk_start in range(0, N, _UPLOAD_CHUNK):
            chunk_end = min(chunk_start + _UPLOAD_CHUNK, N)
            parts = []
            for i in range(chunk_start, chunk_end):
                t = tds[i].get(raw_key)
                parts.append(_tensor_to_bytes(t))
            data = b"".join(parts)
            offset = chunk_start * row_bytes
            pipe = self._client.pipeline()
            pipe.setrange(self._data_key(key_path), offset, data)
            await pipe.execute()

    async def _aupload_heterogeneous_key(
        self,
        key_path: str,
        raw_key,
        tds: list,
        dtype: torch.dtype,
    ):
        """Upload a single heterogeneous leaf key with offset table."""
        N = len(tds)
        offsets = [0]
        shapes: list[list[int]] = []
        byte_offset = 0

        # Stream data in chunks and build offset table
        for chunk_start in range(0, N, _UPLOAD_CHUNK):
            chunk_end = min(chunk_start + _UPLOAD_CHUNK, N)
            parts = []
            for i in range(chunk_start, chunk_end):
                t = tds[i].get(raw_key)
                data_bytes = _tensor_to_bytes(t)
                parts.append(data_bytes)
                shapes.append(list(t.shape))
                byte_offset += len(data_bytes)
                offsets.append(byte_offset)
            data = b"".join(parts)
            file_offset = offsets[chunk_start]
            pipe = self._client.pipeline()
            pipe.setrange(self._data_key(key_path), file_offset, data)
            await pipe.execute()

        # Store offset table (N+1 packed int64s)
        offset_bytes = struct.pack(f"<{len(offsets)}q", *offsets)

        # Full stacked shape uses first element's shape as representative
        # (the actual per-element shapes are in the `shapes` metadata field)
        full_shape = [N] + shapes[0]

        meta = {
            "shape": json.dumps(full_shape),
            "dtype": _dtype_to_str(dtype),
            "homogeneous": "0",
            "shapes": json.dumps(shapes),
        }
        pipe = self._client.pipeline()
        pipe.set(self._idx_key(key_path), offset_bytes)
        pipe.hset(self._meta_key(key_path), mapping=meta)
        pipe.sadd(self._keys_registry_key, key_path)
        await pipe.execute()

        if self._meta_cache is not None:
            self._meta_cache[key_path] = (full_shape, dtype)

    # ---- element access (reads) ----

    async def _abatch_get_element(self, element_idx: int) -> dict[str, torch.Tensor]:
        """Pipelined fetch of all keys for a single stack element."""
        all_keys = sorted(await self._aget_all_keys())
        if not all_keys:
            return {}

        # Fetch metadata for all keys
        pipe = self._client.pipeline()
        for kp in all_keys:
            pipe.hgetall(self._meta_key(kp))
        raw_metas = await pipe.execute()

        pos = _normalize_index(element_idx, self._count)

        # Prepare data fetches
        pipe = self._client.pipeline()
        key_info: list[tuple[str, list[int], torch.dtype, bool]] = []

        for kp, raw_meta in zip(all_keys, raw_metas):
            meta = _decode_meta(raw_meta)
            dtype = _str_to_dtype(meta["dtype"])
            homogeneous = self._is_key_homogeneous(meta)

            if homogeneous:
                full_shape = json.loads(meta["shape"])
                elem_shape = full_shape[1:]
                row_bytes = self._row_bytes(elem_shape, dtype)
                offset = pos * row_bytes
                pipe.getrange(self._data_key(kp), offset, offset + row_bytes - 1)
                key_info.append((kp, elem_shape, dtype, True))
            else:
                # Need offset table lookup first
                pipe.getrange(self._idx_key(kp), pos * 8, (pos + 2) * 8 - 1)
                key_info.append((kp, [], dtype, False))

        results = await pipe.execute()

        # Second pass for heterogeneous keys that needed offset lookup.
        # Reuse raw_metas from the first pipeline (no extra round-trip).
        hetero_kps: list[tuple[int, str, int, int, list[int]]] = []
        for ri, (kp, _elem_shape, _dtype, is_homo) in enumerate(key_info):
            if not is_homo:
                off_data = results[ri]
                start, end = struct.unpack("<2q", off_data)
                meta = _decode_meta(raw_metas[ri])
                shapes = json.loads(meta["shapes"])
                hetero_kps.append((ri, kp, start, end, shapes[pos]))

        if hetero_kps:
            pipe = self._client.pipeline()
            for _, kp, _start, _end, _ in hetero_kps:
                pipe.getrange(self._data_key(kp), _start, _end - 1)
            hetero_data = await pipe.execute()
            for (ri, kp, _start, _end, shape), data in zip(hetero_kps, hetero_data):
                results[ri] = data
                key_info[ri] = (kp, shape, key_info[ri][2], True)

        # Reconstruct tensors
        out: dict[str, torch.Tensor] = {}
        for ri, (kp, elem_shape, dtype, _) in enumerate(key_info):
            data = results[ri]
            tensor = _bytes_to_tensor(data, elem_shape, dtype)
            if self._device is not None:
                tensor = tensor.to(self._device)
            out[kp] = tensor

        return out

    async def _abatch_get_element_keys(
        self, element_idx: int, key_paths: list[str]
    ) -> dict[str, torch.Tensor]:
        """Pipelined fetch of *specific* keys for a single stack element."""
        if not key_paths:
            return {}

        pos = _normalize_index(element_idx, self._count)

        # Fetch metadata
        pipe = self._client.pipeline()
        for kp in key_paths:
            pipe.hgetall(self._meta_key(kp))
        raw_metas = await pipe.execute()

        # Prepare data fetches
        pipe = self._client.pipeline()
        key_info: list[tuple[str, list[int], torch.dtype, bool]] = []

        for kp, raw_meta in zip(key_paths, raw_metas):
            meta = _decode_meta(raw_meta)
            dtype = _str_to_dtype(meta["dtype"])
            homogeneous = self._is_key_homogeneous(meta)

            if homogeneous:
                full_shape = json.loads(meta["shape"])
                elem_shape = full_shape[1:]
                row_bytes = self._row_bytes(elem_shape, dtype)
                offset = pos * row_bytes
                pipe.getrange(self._data_key(kp), offset, offset + row_bytes - 1)
                key_info.append((kp, elem_shape, dtype, True))
            else:
                pipe.getrange(self._idx_key(kp), pos * 8, (pos + 2) * 8 - 1)
                key_info.append((kp, [], dtype, False))

        results = await pipe.execute()

        # Second pass for heterogeneous keys (reuse raw_metas, no extra fetch)
        hetero_kps: list[tuple[int, str, int, int, list[int]]] = []
        for ri, (kp, _elem_shape, _dtype, is_homo) in enumerate(key_info):
            if not is_homo:
                off_data = results[ri]
                _start, _end = struct.unpack("<2q", off_data)
                meta = _decode_meta(raw_metas[ri])
                shapes = json.loads(meta["shapes"])
                hetero_kps.append((ri, kp, _start, _end, shapes[pos]))

        if hetero_kps:
            pipe = self._client.pipeline()
            for _, kp, _start, _end, _ in hetero_kps:
                pipe.getrange(self._data_key(kp), _start, _end - 1)
            hetero_data = await pipe.execute()
            for (ri, kp, _start, _end, shape), data in zip(hetero_kps, hetero_data):
                results[ri] = data
                key_info[ri] = (kp, shape, key_info[ri][2], True)

        out: dict[str, torch.Tensor] = {}
        for ri, (kp, elem_shape, dtype, _) in enumerate(key_info):
            data = results[ri]
            tensor = _bytes_to_tensor(data, elem_shape, dtype)
            if self._device is not None:
                tensor = tensor.to(self._device)
            out[kp] = tensor

        return out

    async def _aset_element_key(
        self, element_idx: int, key_path: str, value: torch.Tensor
    ):
        """Write a single key for one stack element via SETRANGE."""
        pos = _normalize_index(element_idx, self._count)
        raw_meta = _decode_meta(await self._client.hgetall(self._meta_key(key_path)))

        # Key doesn't exist yet — need to register and upload
        if not raw_meta:
            await self._client.sadd(self._keys_registry_key, key_path)
            # Create as homogeneous with this single element
            value = value.contiguous().cpu()
            raw_bytes = _tensor_to_bytes(value)
            elem_shape = list(value.shape)
            full_shape = [self._count] + elem_shape
            row_bytes = len(raw_bytes)

            pipe = self._client.pipeline()
            # Write the element at the right offset (zero-fill for other elements)
            pipe.setrange(self._data_key(key_path), pos * row_bytes, raw_bytes)
            # Ensure the full blob is allocated
            pipe.setrange(
                self._data_key(key_path),
                self._count * row_bytes - 1,
                b"\x00",
            )
            pipe.hset(
                self._meta_key(key_path),
                mapping={
                    "shape": json.dumps(full_shape),
                    "dtype": str(value.dtype),
                    "homogeneous": "1",
                },
            )
            await pipe.execute()
            if self._meta_cache is not None:
                self._meta_cache[key_path] = (full_shape, value.dtype)
            return

        dtype = _str_to_dtype(raw_meta["dtype"])
        homogeneous = self._is_key_homogeneous(raw_meta)

        shape = (
            json.loads(raw_meta["shape"])
            if homogeneous
            else [self._count] + json.loads(raw_meta["shapes"])[pos]
        )
        value = _prepare_indexed_value(value, shape, dtype, pos)
        raw_bytes = _tensor_to_bytes(value)

        if homogeneous:
            offset = pos * len(raw_bytes)
            await self._client.setrange(self._data_key(key_path), offset, raw_bytes)
        else:
            off_data = await self._client.getrange(
                self._idx_key(key_path), pos * 8, (pos + 2) * 8 - 1
            )
            start, end = struct.unpack("<2q", off_data)
            if len(raw_bytes) != end - start:
                raise ValueError(
                    f"Size mismatch for heterogeneous key {key_path!r} "
                    f"element {pos}: expected {end - start} bytes, "
                    f"got {len(raw_bytes)} bytes. "
                    f"Shape changes on individual elements of a heterogeneous "
                    f"key are not supported."
                )
            await self._client.setrange(self._data_key(key_path), start, raw_bytes)

    async def _abatch_get_at(
        self, key_paths: list[str], idx
    ) -> dict[str, torch.Tensor]:
        """Batch fetch indexed slices of multiple keys using GETRANGE."""
        if not key_paths:
            return {}

        meta_map = await self._aget_metadata_batch(key_paths)

        pipe = self._client.pipeline()
        plan: list[tuple[str, list[int], torch.dtype, object, bool]] = []
        fallback_kps: list[str] = []

        scattered = _is_scattered_index(idx)

        for kp in key_paths:
            full_shape, dtype = meta_map[kp]
            result_shape = _getitem_result_shape(full_shape, idx)
            local_idx = _get_local_idx(idx, full_shape[0])

            if scattered:
                ranges = _compute_byte_ranges(full_shape, dtype, idx)
                if ranges is None:
                    fallback_kps.append(kp)
                    continue
                if not ranges:
                    plan.append((kp, result_shape, dtype, None, False))
                    continue
                argv: list = []
                for byte_offset, byte_length in ranges:
                    argv.append(byte_offset)
                    argv.append(byte_length)
                pipe.eval(_LUA_GETRANGES, 1, self._data_key(kp), *argv)
            else:
                cr = _compute_covering_range(full_shape, dtype, idx)
                if cr is None:
                    fallback_kps.append(kp)
                    continue
                byte_offset, byte_length = cr
                if byte_length == 0:
                    plan.append((kp, result_shape, dtype, None, False))
                    continue
                pipe.getrange(
                    self._data_key(kp),
                    byte_offset,
                    byte_offset + byte_length - 1,
                )
            plan.append((kp, result_shape, dtype, local_idx, True))

        # Other indices (e.g. multi-element tuples or None): fetch the whole
        # tensor in the same pipeline and index it locally.
        for kp in fallback_kps:
            pipe.get(self._data_key(kp))

        has_cmds = any(has_cmd for _, _, _, _, has_cmd in plan) or fallback_kps
        raw_results = await pipe.execute() if has_cmds else []

        result: dict[str, torch.Tensor] = {}
        ri = 0
        for kp, result_shape, dtype, local_idx, has_cmd in plan:
            if not has_cmd:
                result[kp] = torch.empty(result_shape, dtype=dtype)
                continue
            data = raw_results[ri]
            ri += 1
            tensor = _bytes_to_tensor(
                data,
                result_shape if local_idx is None else [-1] + list(meta_map[kp][0][1:]),
                dtype,
            )
            if local_idx is not None:
                tensor = tensor[local_idx]
                tensor = tensor.reshape(result_shape)
            if self._device is not None:
                tensor = tensor.to(self._device)
            result[kp] = tensor

        for kp in fallback_kps:
            full_shape, dtype = meta_map[kp]
            tensor = _bytes_to_tensor(raw_results[ri], full_shape, dtype)[
                _as_tuple(idx)
            ]
            ri += 1
            if self._device is not None:
                tensor = tensor.to(self._device)
            result[kp] = tensor

        return result

    # ---- element access (writes) ----

    async def _aset_element(self, element_idx: int, value_td: TensorDictBase):
        """Write all keys for a single stack element via pipelined SETRANGE."""
        all_keys = sorted(await self._aget_all_keys())
        pos = _normalize_index(element_idx, self._count)

        # Pipeline: fetch all metadata + offset tables in one round-trip
        meta_pipe = self._client.pipeline()
        for kp in all_keys:
            meta_pipe.hgetall(self._meta_key(kp))
        raw_metas = await meta_pipe.execute()

        # Classify keys and prepare offset fetches for heterogeneous keys
        parsed_metas: list[tuple[str, dict, torch.dtype, bool]] = []
        offset_pipe = self._client.pipeline()
        hetero_indices: list[int] = []
        for i, (kp, raw_meta) in enumerate(zip(all_keys, raw_metas)):
            meta = _decode_meta(raw_meta)
            dtype = _str_to_dtype(meta["dtype"])
            homogeneous = self._is_key_homogeneous(meta)
            parsed_metas.append((kp, meta, dtype, homogeneous))
            if not homogeneous:
                offset_pipe.getrange(self._idx_key(kp), pos * 8, (pos + 2) * 8 - 1)
                hetero_indices.append(i)

        hetero_offsets = await offset_pipe.execute() if hetero_indices else []

        # Build the write pipeline
        write_pipe = self._client.pipeline()
        hi = 0
        for kp, meta, dtype, homogeneous in parsed_metas:
            key_parts = kp.split(_KEY_SEP)
            raw_key = tuple(key_parts) if len(key_parts) > 1 else key_parts[0]
            value = value_td.get(raw_key)
            shape = (
                json.loads(meta["shape"])
                if homogeneous
                else [self._count] + json.loads(meta["shapes"])[pos]
            )
            value = _prepare_indexed_value(value, shape, dtype, pos)

            if homogeneous:
                full_shape = json.loads(meta["shape"])
                elem_shape = full_shape[1:]
                row_bytes = self._row_bytes(elem_shape, dtype)
                offset = pos * row_bytes
                write_pipe.setrange(self._data_key(kp), offset, _tensor_to_bytes(value))
            else:
                off_data = hetero_offsets[hi]
                hi += 1
                start, end = struct.unpack("<2q", off_data)
                new_bytes = _tensor_to_bytes(value)
                if len(new_bytes) != end - start:
                    raise ValueError(
                        f"Cannot write element {element_idx} for key {kp!r}: "
                        f"new size {len(new_bytes)} != existing size "
                        f"{end - start}. Resizing heterogeneous elements "
                        f"in-place is not supported."
                    )
                write_pipe.setrange(self._data_key(kp), start, new_bytes)
        await write_pipe.execute()

    async def _abatch_set_at(self, items: dict[str, tuple[torch.Tensor, object]]):
        """Batch-write indexed slices using SETRANGE / Lua."""
        if not items:
            return

        key_paths = list(items.keys())
        meta_map = await self._aget_metadata_batch(key_paths)

        direct_kps: list[str] = []
        lua_kps: list[str] = []
        rmw_kps: list[str] = []

        for kp in key_paths:
            _, idx = items[kp]
            shape, dtype = meta_map[kp]
            ranges = _compute_byte_ranges(shape, dtype, idx)
            if ranges is None:
                rmw_kps.append(kp)
            elif not ranges:
                pass
            elif len(ranges) == 1:
                direct_kps.append(kp)
            elif _is_scattered_index(idx):
                lua_kps.append(kp)
            else:
                rmw_kps.append(kp)

        pipe = self._client.pipeline()
        has_pipe_cmds = False

        for kp in direct_kps:
            value, idx = items[kp]
            shape, dtype = meta_map[kp]
            value = _prepare_indexed_value(value, shape, dtype, idx)
            ranges = _compute_byte_ranges(shape, dtype, idx)
            byte_offset, _ = ranges[0]
            pipe.setrange(
                self._data_key(kp),
                byte_offset,
                _tensor_to_bytes(value.contiguous()),
            )
            has_pipe_cmds = True

        for kp in lua_kps:
            value, idx = items[kp]
            shape, dtype = meta_map[kp]
            value = _prepare_indexed_value(value, shape, dtype, idx)
            ranges = _compute_byte_ranges(shape, dtype, idx)
            value_bytes = _tensor_to_bytes(value.contiguous())
            argv: list = []
            offset = 0
            for byte_offset, byte_length in ranges:
                argv.append(byte_offset)
                argv.append(value_bytes[offset : offset + byte_length])
                offset += byte_length
            pipe.eval(_LUA_SETRANGES, 1, self._data_key(kp), *argv)
            has_pipe_cmds = True

        if has_pipe_cmds:
            await pipe.execute()

        # Covering-range RMW for step>1 slices or unsupported
        if rmw_kps:
            pipe = self._client.pipeline()
            cr_data: list[tuple[str, int, int, list[int], torch.dtype]] = []
            for kp in rmw_kps:
                shape, dtype = meta_map[kp]
                _, idx = items[kp]
                cr = _compute_covering_range(shape, dtype, idx)
                if cr is None:
                    # Full read-modify-write
                    data = await self._client.get(self._data_key(kp))
                    tensor = _bytes_to_tensor(data, shape, dtype)
                    value, idx = items[kp]
                    tensor[_as_tuple(idx)] = value
                    await self._client.set(self._data_key(kp), _tensor_to_bytes(tensor))
                    continue
                byte_offset, byte_length = cr
                pipe.getrange(
                    self._data_key(kp),
                    byte_offset,
                    byte_offset + byte_length - 1,
                )
                cr_data.append((kp, byte_offset, byte_length, shape, dtype))
            if cr_data:
                raw_covers = await pipe.execute()
                pipe = self._client.pipeline()
                for (kp, byte_offset, byte_length, shape, dtype), data in zip(
                    cr_data, raw_covers
                ):
                    rest = shape[1:]
                    elem_size = torch.tensor([], dtype=dtype).element_size()
                    row_bytes = elem_size * (
                        int(torch.tensor(rest).prod().item()) if rest else 1
                    )
                    covering_rows = byte_length // row_bytes if row_bytes > 0 else 0
                    covering_shape = [covering_rows] + rest
                    covering_tensor = _bytes_to_tensor(data, covering_shape, dtype)
                    value, idx = items[kp]
                    local_idx = _get_local_idx(idx, shape[0])
                    covering_tensor[local_idx] = value
                    pipe.setrange(
                        self._data_key(kp),
                        byte_offset,
                        _tensor_to_bytes(covering_tensor.contiguous()),
                    )
                await pipe.execute()

    # ---- TensorDictBase interface ----

    @property
    def batch_size(self) -> torch.Size:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = torch.Size(value)

    @property
    def device(self) -> torch.device | None:
        return self._device

    @device.setter
    def device(self, value):
        self._device = torch.device(value) if value is not None else None

    def _rename_subtds(self, names):
        pass

    # ---- Key access ----

    def _index_tensordict(self, index, new_batch_size=None, names=None):
        """Eagerly fetch all leaf tensors for the given index in one pipeline."""
        batch_size = self.batch_size
        if new_batch_size is None:
            new_batch_size = _getitem_batch_size(batch_size, index)
        if names is None:
            names = self._get_names_idx(index)

        all_keys = sorted(self._get_all_keys())
        result_map = self._run_sync(self._abatch_get_at(all_keys, index))

        source: dict = {}
        for kp, value in result_map.items():
            parts = kp.split(_KEY_SEP)
            d = source
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value

        def _build(d, bs):
            for k, v in d.items():
                if isinstance(v, dict):
                    d[k] = _build(v, bs)
            return TensorDict._new_unsafe(
                source=d,
                batch_size=bs,
                device=self._device,
                names=names,
            )

        return _build(source, new_batch_size)

    def __getitem__(self, index):
        index_unravel = _unravel_key_to_tuple(index)
        if index_unravel:
            return self._get_tuple(index_unravel, NO_DEFAULT)

        # Integer index on the stack dim: return write-through view
        if (
            isinstance(index, int)
            and not isinstance(index, bool)
            and self._stack_dim == 0
        ):
            return _StoreStackElementView(self, index)

        # As in TensorDictBase.__getitem__, an Ellipsis stands for the
        # remaining batch dimensions, not the leaf dimensions.
        if isinstance(index, tuple) and any(idx is Ellipsis for idx in index):
            index = convert_ellipsis_to_idx(index, self.batch_size)

        # General indexing via _index_tensordict
        return self._index_tensordict(index)

    def __setitem__(self, index, value):
        index_unravel = _unravel_key_to_tuple(index)
        if index_unravel:
            return self.set(index_unravel, value, inplace=True)

        if isinstance(index, list):
            index = torch.tensor(index)

        _check_indexed_value(self, value)
        if isinstance(value, Number):
            # a scalar is written to every entry, in its dtype
            self._run_sync(
                self._abatch_set_at({kp: (value, index) for kp in self._get_all_keys()})
            )
            return

        # Integer assignment on stack dim: write element
        if (
            isinstance(index, int)
            and not isinstance(index, bool)
            and self._stack_dim == 0
        ):
            if not isinstance(value, TensorDictBase):
                value = TensorDict.from_dict(value, batch_size=[])
            self._run_sync(self._aset_element(index, value))
            return

        if not isinstance(value, TensorDictBase):
            value = TensorDict.from_dict(value, batch_size=[])

        items: dict[str, tuple[torch.Tensor, object]] = {}
        for key in value.keys(include_nested=True, leaves_only=True):
            key_tuple = _unravel_key_to_tuple(key)
            key_path = _KEY_SEP.join(key_tuple)
            items[key_path] = (value.get(key), index)

        self._run_sync(self._abatch_set_at(items))

    def _get_str(self, key, default=NO_DEFAULT, **kwargs):
        key_path = key
        all_keys = self._get_all_keys()

        # Check nested
        prefix_check = key_path + _KEY_SEP
        nested_keys = [k for k in all_keys if k.startswith(prefix_check)]
        if nested_keys:
            # Return full stacked tensor for each nested leaf, build TD
            result = self._run_sync(self._abatch_get_at(nested_keys, slice(None)))
            source: dict = {}
            for kp, tensor in result.items():
                rel = kp[len(prefix_check) :]
                parts = rel.split(_KEY_SEP)
                d = source
                for part in parts[:-1]:
                    d = d.setdefault(part, {})
                d[parts[-1]] = tensor
            return TensorDict(source, batch_size=self.batch_size, device=self._device)

        if key_path in all_keys:
            result = self._run_sync(self._abatch_get_at([key_path], slice(None)))
            t = result.get(key_path)
            if t is not None:
                return t

        if default is not NO_DEFAULT:
            return default
        raise KeyError(f"key {key} not found in {type(self).__name__}")

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
        if not validated:
            value = self._validate_value(value, check_shape=True, key=key)
        if self.is_locked and not ignore_lock and not inplace:
            raise RuntimeError(_LOCK_ERROR)

        key_path = key

        if is_tensor_collection(value):
            for sub_key in value.keys(include_nested=True, leaves_only=True):
                sub_tuple = _unravel_key_to_tuple(sub_key)
                full_kp = key_path + _KEY_SEP + _KEY_SEP.join(sub_tuple)
                tensor = value.get(sub_key)
                self._run_sync(self._aset_full_tensor(full_kp, tensor))
            return self

        if isinstance(value, torch.Tensor):
            self._run_sync(self._aset_full_tensor(key_path, value))
            return self

        try:
            value = torch.as_tensor(value)
            self._run_sync(self._aset_full_tensor(key_path, value))
        except (ValueError, TypeError):
            raise TypeError(
                f"LazyStackedTensorDictStore only supports tensor values, got {type(value)}"
            )
        return self

    async def _aset_full_tensor(self, key_path: str, tensor: torch.Tensor):
        """Set a full (stacked) tensor for a key."""
        data = _tensor_to_bytes(tensor)
        shape = list(tensor.shape)
        dtype = tensor.dtype
        meta = {
            "shape": json.dumps(shape),
            "dtype": _dtype_to_str(dtype),
            "homogeneous": "1",
        }
        pipe = self._client.pipeline()
        pipe.set(self._data_key(key_path), data)
        pipe.hset(self._meta_key(key_path), mapping=meta)
        pipe.sadd(self._keys_registry_key, key_path)
        await pipe.execute()
        if self._meta_cache is not None:
            self._meta_cache[key_path] = (shape, dtype)

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
        key_path = _KEY_SEP.join(key)
        if not validated:
            value = self._validate_value(value, check_shape=True, key=key)
        if self.is_locked and not inplace:
            raise RuntimeError(_LOCK_ERROR)
        if isinstance(value, torch.Tensor):
            self._run_sync(self._aset_full_tensor(key_path, value))
        elif is_tensor_collection(value):
            for sub_key in value.keys(include_nested=True, leaves_only=True):
                sub_tuple = _unravel_key_to_tuple(sub_key)
                full_kp = key_path + _KEY_SEP + _KEY_SEP.join(sub_tuple)
                self._run_sync(self._aset_full_tensor(full_kp, value.get(sub_key)))
        else:
            self._run_sync(self._aset_full_tensor(key_path, torch.as_tensor(value)))
        return self

    def _set_at_str(self, key, value, idx, *, validated, non_blocking):
        items = {key: (value, idx)}
        self._run_sync(self._abatch_set_at(items))
        return self

    def _set_at_tuple(self, key, value, idx, *, validated, non_blocking):
        key = _unravel_key_to_tuple(key)
        if len(key) == 1:
            return self._set_at_str(
                key[0], value, idx, validated=validated, non_blocking=non_blocking
            )
        key_path = _KEY_SEP.join(key)
        items = {key_path: (value, idx)}
        self._run_sync(self._abatch_set_at(items))
        return self

    def _get_at_str(self, key, idx, default=NO_DEFAULT, **kwargs):
        result = self._run_sync(self._abatch_get_at([key], idx))
        tensor = result.get(key)
        if tensor is None:
            if default is not NO_DEFAULT:
                return default
            raise KeyError(f"key {key} not found in {type(self).__name__}")
        return tensor

    def _get_at_tuple(self, key, idx, default=NO_DEFAULT, **kwargs):
        key = _unravel_key_to_tuple(key)
        if len(key) == 1:
            return self._get_at_str(key[0], idx, default=default, **kwargs)
        key_path = _KEY_SEP.join(key)
        result = self._run_sync(self._abatch_get_at([key_path], idx))
        tensor = result.get(key_path)
        if tensor is None:
            if default is not NO_DEFAULT:
                return default
            raise KeyError(f"key {key} not found in {type(self).__name__}")
        return tensor

    def _convert_inplace(self, inplace, key):
        if inplace is not False:
            all_keys = self._get_all_keys()
            has_key = key in all_keys or any(
                k.startswith(key + _KEY_SEP) for k in all_keys
            )
            if inplace is True and not has_key:
                raise KeyError(
                    _KEY_ERROR.format(key, type(self).__name__, sorted(self.keys()))
                )
            inplace = has_key
        return inplace

    def keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
        *,
        sort: bool = False,
    ) -> _LazyStackedStoreKeysView:
        return _LazyStackedStoreKeysView(
            tensordict=self,
            include_nested=include_nested,
            leaves_only=leaves_only,
            is_leaf=is_leaf,
            sort=sort,
        )

    @_lock_blocked
    def del_(self, key: NestedKey) -> LazyStackedTensorDictStore:
        if isinstance(key, str):
            key_path = key
        else:
            key = _unravel_key_to_tuple(key)
            key_path = _KEY_SEP.join(key)

        all_keys = self._get_all_keys()
        if key_path in all_keys:
            self._run_sync(self._adel_key(key_path))
        prefix_check = key_path + _KEY_SEP
        for k in all_keys:
            if k.startswith(prefix_check):
                self._run_sync(self._adel_key(k))
        return self

    async def _adel_key(self, key_path: str):
        pipe = self._client.pipeline()
        pipe.delete(self._data_key(key_path))
        pipe.delete(self._idx_key(key_path))
        pipe.delete(self._meta_key(key_path))
        pipe.srem(self._keys_registry_key, key_path)
        await pipe.execute()
        if self._meta_cache is not None:
            self._meta_cache.pop(key_path, None)

    def rename_key_(
        self, old_key: NestedKey, new_key: NestedKey, safe: bool = False
    ) -> LazyStackedTensorDictStore:
        if isinstance(old_key, str):
            old_path = old_key
        else:
            old_path = _KEY_SEP.join(_unravel_key_to_tuple(old_key))
        if isinstance(new_key, str):
            new_path = new_key
        else:
            new_path = _KEY_SEP.join(_unravel_key_to_tuple(new_key))

        if safe and new_path in self._get_all_keys():
            raise KeyError(f"key {new_key} already present in {type(self).__name__}.")

        async def _arename():
            all_keys = await self._aget_all_keys()
            if old_path in all_keys:
                pipe = self._client.pipeline()
                pipe.rename(self._data_key(old_path), self._data_key(new_path))
                pipe.rename(self._meta_key(old_path), self._meta_key(new_path))
                pipe.srem(self._keys_registry_key, old_path)
                pipe.sadd(self._keys_registry_key, new_path)
                # Try renaming idx key (may not exist for homogeneous)
                try:
                    pipe.rename(self._idx_key(old_path), self._idx_key(new_path))
                except Exception:
                    pass
                await pipe.execute()

        self._run_sync(_arename())
        return self

    def entry_class(self, key: NestedKey) -> type:
        if isinstance(key, str):
            key_path = key
        else:
            key_path = _KEY_SEP.join(_unravel_key_to_tuple(key))
        all_keys = self._get_all_keys()
        if key_path in all_keys:
            return torch.Tensor
        prefix_check = key_path + _KEY_SEP
        if any(k.startswith(prefix_check) for k in all_keys):
            return LazyStackedTensorDictStore
        raise KeyError(f"key {key} not found in {type(self).__name__}")

    # ---- Locking ----

    def _propagate_lock(self, lock_parents_weakrefs=None, *, is_compiling):
        self._is_locked = True
        if not is_compiling:
            if lock_parents_weakrefs is None:
                lock_parents_weakrefs = []
            else:
                self._lock_parents_weakrefs = (
                    self._lock_parents_weakrefs + lock_parents_weakrefs
                )
            lock_parents_weakrefs = list(lock_parents_weakrefs)
            lock_parents_weakrefs.append(weakref.ref(self))

    @_erase_cache_first
    def _propagate_unlock(self):
        self._is_locked = False
        self._is_shared = False
        self._is_memmap = False
        return []

    # ---- Materialization ----

    @deprecated(
        "LazyStackedTensorDictStore.to_local()",
        removal="0.17",
        replacement="to_tensordict()",
    )
    def to_local(self) -> TensorDict:
        """Pulls the whole store into a local ``TensorDict``.

        .. deprecated:: 0.15
            Use :meth:`to_tensordict` instead.
        """
        return self.to_tensordict()

    def contiguous(self, *, canonical: bool = False) -> TensorDict:
        out = self.to_tensordict()
        if canonical:
            out = out.contiguous(canonical=True)
        return out

    # ---- Construction ----

    @classmethod
    def from_lazy_stack(
        cls,
        lazy_td,
        *,
        backend: STORE_BACKENDS = "redis",
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        unix_socket_path: str | None = None,
        prefix: str = "tensordict",
        device=None,
        **kwargs,
    ) -> LazyStackedTensorDictStore:
        """Upload a :class:`LazyStackedTensorDict` to a key-value store.

        Data is streamed in chunks of ``_UPLOAD_CHUNK`` elements to avoid
        materialising the full stack in memory.

        Args:
            lazy_td (LazyStackedTensorDict): The source lazy stack.

        Keyword Args:
            backend: Store backend (``"redis"``, ``"dragonfly"``, etc.).
            host, port, db, unix_socket_path, prefix: Connection params.
            device: Device override for retrieved tensors.
            **kwargs: Extra connection keyword arguments.

        Returns:
            A :class:`LazyStackedTensorDictStore` backed by the uploaded data.
        """
        from tensordict._lazy import LazyStackedTensorDict

        if not isinstance(lazy_td, LazyStackedTensorDict):
            raise TypeError(f"Expected LazyStackedTensorDict, got {type(lazy_td)}")

        if device is None:
            device = lazy_td.device

        count = len(lazy_td.tensordicts)
        stack_dim = lazy_td.stack_dim
        inner_batch_size = lazy_td.tensordicts[0].batch_size

        connect_kwargs = {}
        if unix_socket_path is not None:
            connect_kwargs["unix_socket_path"] = unix_socket_path
        else:
            connect_kwargs["host"] = host
            connect_kwargs["port"] = port
        connect_kwargs["db"] = db

        out = cls(
            backend=backend,
            count=count,
            stack_dim=stack_dim,
            inner_batch_size=inner_batch_size,
            device=device,
            prefix=prefix,
            **connect_kwargs,
            **kwargs,
        )
        out._run_sync(out._aupload_lazy_stack(lazy_td))
        return out

    @classmethod
    def from_store(
        cls,
        *,
        backend: STORE_BACKENDS = "redis",
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        unix_socket_path: str | None = None,
        prefix: str = "tensordict",
        td_id: str,
        device=None,
        **kwargs,
    ) -> LazyStackedTensorDictStore:
        """Reconnect to an existing LazyStackedTensorDictStore on a server."""
        import redis.asyncio as aioredis

        connect_kwargs = dict(kwargs)
        if unix_socket_path is not None:
            connect_kwargs["unix_socket_path"] = unix_socket_path
        else:
            connect_kwargs["host"] = host
            connect_kwargs["port"] = port
        connect_kwargs["db"] = db

        loop = asyncio.new_event_loop()
        client = aioredis.Redis(**connect_kwargs)

        async def _read_meta():
            pipe = client.pipeline()
            pipe.get(f"{prefix}:{{{td_id}}}:__type__")
            pipe.get(f"{prefix}:{{{td_id}}}:__count__")
            pipe.get(f"{prefix}:{{{td_id}}}:__stack_dim__")
            pipe.get(f"{prefix}:{{{td_id}}}:__inner_batch_size__")
            pipe.get(f"{prefix}:{{{td_id}}}:__device__")
            results = await pipe.execute()
            await client.aclose()
            return results

        raw_type, raw_count, raw_sd, raw_ibs, raw_dev = loop.run_until_complete(
            _read_meta()
        )
        loop.close()

        if raw_type is None:
            raise KeyError(f"No LazyStackedTensorDictStore with td_id={td_id!r} found.")

        count = int(raw_count)
        stack_dim = int(raw_sd)
        inner_batch_size = json.loads(raw_ibs)

        if device is None:
            dev_str = raw_dev.decode() if isinstance(raw_dev, bytes) else raw_dev
            device = torch.device(dev_str) if dev_str else None

        return cls(
            backend=backend,
            host=host,
            port=port,
            db=db,
            unix_socket_path=unix_socket_path,
            prefix=prefix,
            count=count,
            stack_dim=stack_dim,
            inner_batch_size=inner_batch_size,
            device=device,
            td_id=td_id,
            **kwargs,
        )

    @classmethod
    def from_dict(
        cls,
        input_dict,
        *,
        auto_batch_size: bool = False,
        batch_size=None,
        device=None,
        **kwargs,
    ):
        """Not directly supported — use :meth:`from_lazy_stack` instead."""
        raise NotImplementedError(
            f"{cls.__name__}.from_dict is not supported. "
            "Use LazyStackedTensorDictStore.from_lazy_stack(lazy_td, ...) instead."
        )

    # ---- Cloning ----

    def _clone(self, recurse: bool = True) -> LazyStackedTensorDictStore:
        if recurse:
            new_td = LazyStackedTensorDictStore(
                host=self._host,
                port=self._port,
                db=self._db,
                unix_socket_path=self._unix_socket_path,
                prefix=self._namespace,
                count=self._count,
                stack_dim=self._stack_dim,
                inner_batch_size=self._inner_batch_size,
                device=self._device,
            )
            new_td.update(self.to_tensordict())
            return new_td
        # Shallow: same data, new wrapper
        return LazyStackedTensorDictStore(
            host=self._host,
            port=self._port,
            db=self._db,
            unix_socket_path=self._unix_socket_path,
            prefix=self._namespace,
            count=self._count,
            stack_dim=self._stack_dim,
            inner_batch_size=self._inner_batch_size,
            device=self._device,
            td_id=self._td_id,
        )

    # ---- Misc required overrides ----

    def is_contiguous(self) -> bool:
        return False

    def detach_(self) -> Self:
        return self

    @_lock_blocked
    def popitem(self) -> Tuple[NestedKey, CompatibleType]:
        keys_list = list(self.keys())
        if not keys_list:
            raise KeyError(f"popitem(): {type(self).__name__} is empty")
        key = keys_list[-1]
        value = self.get(key)
        self.del_(key)
        return key, value

    def _change_batch_size(self, new_size: torch.Size) -> None:
        self._batch_size = new_size

    def zero_(self) -> Self:
        for key in self.keys():
            self.fill_(key, 0)
        return self

    def fill_(self, key: NestedKey, value: float | bool) -> TensorDictBase:
        existing = self.get(key)
        if is_tensor_collection(existing):
            for subkey in existing.keys():
                existing.fill_(subkey, value)
        else:
            existing = existing.fill_(value)
            self.set_(key, existing)
        return self

    def empty(
        self, recurse=False, *, batch_size=None, device=NO_DEFAULT, names=None
    ) -> T:
        return TensorDict(
            {},
            device=self.device if device is NO_DEFAULT else device,
            batch_size=self.batch_size if batch_size is None else batch_size,
            names=self.names if names is None and self._has_names() else names,
        )

    def masked_fill(self, mask, value):
        return self.to_tensordict().masked_fill(mask, value)

    def masked_fill_(self, mask, value):
        for key in self.keys(include_nested=True, leaves_only=True):
            tensor = self.get(key)
            tensor = tensor.masked_fill(mask, value)
            self.set_(key, tensor)
        return self

    def masked_select(self, mask):
        return self.to_tensordict().masked_select(mask)

    def where(self, condition, other, *, out=None, pad=None, update_batch_size=False):
        return self.to_tensordict().where(
            condition=condition,
            other=other,
            out=out,
            pad=pad,
            update_batch_size=update_batch_size,
        )

    # ---- Pickling ----

    def __getstate__(self):
        return {
            "_backend": self._backend,
            "_host": self._host,
            "_port": self._port,
            "_db": self._db,
            "_unix_socket_path": self._unix_socket_path,
            "_namespace": self._namespace,
            "_td_id": self._td_id,
            "_count": self._count,
            "_stack_dim": self._stack_dim,
            "_inner_batch_size": self._inner_batch_size,
            "_batch_size": self._batch_size,
            "_device": self._device,
            "_redis_kwargs": self._redis_kwargs,
            "_is_locked": self._is_locked,
            "_td_dim_names": self._td_dim_names,
            "_cache_metadata": self._cache_metadata,
        }

    def __setstate__(self, state):
        import redis.asyncio as aioredis

        self._backend = state.get("_backend", "redis")
        self._host = state["_host"]
        self._port = state["_port"]
        self._db = state["_db"]
        self._unix_socket_path = state["_unix_socket_path"]
        self._namespace = state["_namespace"]
        self._td_id = state["_td_id"]
        self._count = state["_count"]
        self._stack_dim = state["_stack_dim"]
        self._inner_batch_size = state["_inner_batch_size"]
        self._batch_size = state["_batch_size"]
        self._device = state["_device"]
        self._redis_kwargs = state["_redis_kwargs"]
        self._td_dim_names = state["_td_dim_names"]

        self._locked_tensordicts = []
        self._lock_id = set()
        self._is_shared = False
        self._is_memmap = False
        self._cache_metadata = state.get("_cache_metadata", True)
        self._meta_cache = {} if self._cache_metadata else None

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._owns_loop = True

        connect_kwargs = dict(self._redis_kwargs)
        if self._unix_socket_path is not None:
            connect_kwargs["unix_socket_path"] = self._unix_socket_path
        else:
            connect_kwargs["host"] = self._host
            connect_kwargs["port"] = self._port
        connect_kwargs["db"] = self._db
        self._client = aioredis.Redis(**connect_kwargs)

        was_locked = state.get("_is_locked", False)
        self._is_locked = False
        if was_locked:
            self.lock_()

    # ---- Cleanup ----

    def close(self):
        if hasattr(self, "_owns_loop") and self._owns_loop:
            if (
                hasattr(self, "_client")
                and hasattr(self, "_loop")
                and self._loop.is_running()
            ):
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self._client.aclose(), self._loop
                    )
                    future.result(timeout=2)
                except Exception:
                    pass
            if hasattr(self, "_loop") and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            if hasattr(self, "_thread") and self._thread.is_alive():
                self._thread.join(timeout=2)
            if hasattr(self, "_loop") and not self._loop.is_running():
                # releases the selector and the self-pipe sockets of the loop
                self._loop.close()
            self._owns_loop = False

    def clear_redis(self):
        """Delete all keys associated with this TensorDict from Redis."""

        async def _aclear():
            all_keys = await self._aget_all_keys()
            pipe = self._client.pipeline()
            for key_path in all_keys:
                pipe.delete(self._data_key(key_path))
                pipe.delete(self._idx_key(key_path))
                pipe.delete(self._meta_key(key_path))
            pipe.delete(self._keys_registry_key)
            pipe.delete(self._redis_key("__type__"))
            pipe.delete(self._redis_key("__count__"))
            pipe.delete(self._redis_key("__stack_dim__"))
            pipe.delete(self._redis_key("__inner_batch_size__"))
            pipe.delete(self._redis_key("__device__"))
            await pipe.execute()

        self._run_sync(_aclear())

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        keys_str = list(self.keys())
        return (
            f"LazyStackedTensorDictStore(\n"
            f"    keys={keys_str},\n"
            f"    batch_size={self.batch_size},\n"
            f"    count={self._count},\n"
            f"    stack_dim={self._stack_dim},\n"
            f"    device={self.device},\n"
            f"    backend={self._backend!r},\n"
            f"    td_id={self._td_id!r})"
        )

    # ---- Delegated ops ----

    # ---- Shape ops: not supported ----

    def _view(self, *args, **kwargs):
        raise RuntimeError(
            f"Cannot call `view` on a {type(self).__name__}. "
            "Call `to_tensordict()` first."
        )

    def _transpose(self, dim0, dim1):
        raise RuntimeError(
            f"Cannot call `transpose` on a {type(self).__name__}. "
            "Call `to_tensordict()` first."
        )

    def _permute(self, *args, **kwargs):
        raise RuntimeError(
            f"Cannot call `permute` on a {type(self).__name__}. "
            "Call `to_tensordict()` first."
        )

    def _squeeze(self, dim=None):
        raise RuntimeError(
            f"Cannot call `squeeze` on a {type(self).__name__}. "
            "Call `to_tensordict()` first."
        )

    def _unsqueeze(self, dim: int):
        raise RuntimeError(
            f"Cannot call `unsqueeze` on a {type(self).__name__}. "
            "Call `to_tensordict()` first."
        )

    def chunk(self, chunks: int, dim: int = 0) -> tuple[TensorDictBase, ...]:
        splits = -(self.batch_size[dim] // -chunks)
        return self.split(splits, dim)

    # ---- Memory ops: not supported ----

    def share_memory_(self):
        raise NotImplementedError(
            f"Cannot call share_memory_ on a {type(self).__name__}."
        )

    def _memmap_(
        self,
        *,
        prefix,
        copy_existing,
        executor,
        futures,
        inplace,
        like,
        share_non_tensor,
        existsok,
        robust_key,
    ):
        raise RuntimeError(f"Cannot call memmap on a {type(self).__name__} in-place.")

    def make_memmap(self, key, shape, *, dtype=None, robust_key=True):
        raise RuntimeError(
            f"Cannot make memory-mapped tensor on a {type(self).__name__}."
        )

    def make_memmap_from_storage(
        self, key, storage, shape, *, dtype=None, robust_key=True
    ):
        raise RuntimeError(
            f"Cannot make memory-mapped tensor on a {type(self).__name__}."
        )

    def make_memmap_from_tensor(self, key, tensor, *, copy_data=True, robust_key=True):
        raise RuntimeError(
            f"Cannot make memory-mapped tensor on a {type(self).__name__}."
        )

    def memmap_(self, prefix=None, copy_existing=False, num_threads=0):
        """Raises a ``RuntimeError``: a LazyStackedTensorDictStore cannot be memory-mapped in-place.

        Args:
            prefix (str, optional): unused.
            copy_existing (bool, optional): unused.
            num_threads (int, optional): unused.

        See :meth:`~tensordict.TensorDictBase.memmap_`.
        """
        raise RuntimeError(
            f"Cannot build a memmap TensorDict in-place from a {type(self).__name__}."
        )

    def pin_memory(self, *args, **kwargs):
        raise RuntimeError(f"Cannot pin memory of a {type(self).__name__}.")

    def _add_batch_dim(self, *, in_dim, vmap_level):
        raise RuntimeError(f"{type(self).__name__} cannot be used with vmap.")

    def _remove_batch_dim(self, vmap_level, batch_size, out_dim): ...

    def _maybe_remove_batch_dim(self, funcname, vmap_level, batch_size, out_dim): ...

    def _select(self, *keys, inplace=False, strict=True, set_shared=True):
        raise NotImplementedError(f"Cannot call select on a {type(self).__name__}.")

    def _exclude(self, *keys, inplace=False, set_shared=True):
        raise NotImplementedError(f"Cannot call exclude on a {type(self).__name__}.")

    @_as_context_manager()
    def flatten_keys(self, separator=".", inplace=False):
        """Returns an in-memory copy where the nested keys are joined by ``separator``.

        The content is first loaded with :meth:`~tensordict.TensorDictBase.to_tensordict`.

        Args:
            separator (str, optional): the separator between the nested items. Defaults to ``"."``.
            inplace (bool, optional): must be ``False``; ``True`` raises a ``ValueError``.
                Defaults to ``False``.

        See :meth:`~tensordict.TensorDictBase.flatten_keys`.
        """
        if inplace:
            raise ValueError(
                f"Cannot call flatten_keys in_place with a {type(self).__name__}."
            )
        return self.to_tensordict().flatten_keys(separator=separator)

    @_as_context_manager()
    def unflatten_keys(self, separator=".", inplace=False):
        if inplace:
            raise ValueError(
                f"Cannot call unflatten_keys in_place with a {type(self).__name__}."
            )
        return self.to_tensordict().unflatten_keys(separator=separator)

    _load_memmap = TensorDict._load_memmap

    def _set_non_tensor(self, key: NestedKey, value: Any):
        raise NotImplementedError(
            f"set_non_tensor is not compatible with {type(self).__name__}."
        )

    def _stack_onto_(self, list_item, dim):
        raise RuntimeError(f"Cannot call _stack_onto_ on a {type(self).__name__}.")


_register_tensor_class(LazyStackedTensorDictStore)

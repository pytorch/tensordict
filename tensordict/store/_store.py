# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Key-value store backed TensorDict implementations (Redis, Dragonfly, etc.)."""

from __future__ import annotations

import asyncio
import importlib
import json
import pickle
import struct
import threading
import uuid
import weakref
from typing import Any, Callable, Literal, Sequence, Tuple, Type, TYPE_CHECKING

import torch
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
from tensordict.utils import (
    _as_context_manager,
    _getitem_batch_size,
    _is_tensorclass,
    _KEY_ERROR,
    _LOCK_ERROR,
    erase_cache,
    is_non_tensor,
    lock_blocked,
    NestedKey,
    unravel_key,
)

_has_redis = importlib.util.find_spec("redis", None) is not None

STORE_BACKENDS = Literal["redis", "dragonfly"]

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any

# Separator used for nested key paths in Redis keys
_KEY_SEP = "."

# Lua scripts executed server-side to batch byte-range operations into a
# single round-trip per key.  Each script takes ONE Redis key and a flat
# argument list encoding the byte ranges.

# GETRANGES: ARGV = [offset1, length1, offset2, length2, ...]
# Returns the *concatenated* bytes from all ranges.
_LUA_GETRANGES = """\
local key = KEYS[1]
local parts = {}
for i = 1, #ARGV, 2 do
    local off = tonumber(ARGV[i])
    local len = tonumber(ARGV[i + 1])
    parts[#parts + 1] = redis.call('GETRANGE', key, off, off + len - 1)
end
return table.concat(parts)
"""

# SETRANGES: ARGV = [offset1, data1, offset2, data2, ...]
# Applies all SETRANGEs atomically; returns "OK".
_LUA_SETRANGES = """\
local key = KEYS[1]
for i = 1, #ARGV, 2 do
    redis.call('SETRANGE', key, tonumber(ARGV[i]), ARGV[i + 1])
end
return redis.status_reply('OK')
"""


def _dtype_to_str(dtype: torch.dtype) -> str:
    """Convert a torch.dtype to its string representation."""
    return str(dtype)


def _str_to_dtype(s: str) -> torch.dtype:
    """Convert a string representation back to a torch.dtype."""
    # e.g. "torch.float32" -> torch.float32
    return getattr(torch, s.split(".")[-1])


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize a tensor to raw bytes.

    The tensor is made contiguous and moved to CPU before serialization.
    Uses NumPy's buffer protocol for fast zero-overhead memcpy.
    """
    return tensor.detach().contiguous().cpu().numpy().tobytes()


def _bytes_to_tensor(
    data: bytes,
    shape: list[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Deserialize raw bytes to a tensor using torch.frombuffer.

    Returns a writable tensor (backed by a bytearray copy of the data).
    """
    buf = bytearray(data)
    return torch.frombuffer(buf, dtype=dtype).reshape(shape)


def _decode_meta(raw_meta: dict) -> dict[str, str]:
    """Decode a Redis hash response (bytes keys/values) to a ``{str: str}`` dict."""
    return {
        k.decode() if isinstance(k, bytes) else k: (
            v.decode() if isinstance(v, bytes) else v
        )
        for k, v in raw_meta.items()
    }


def _resolve_tensorclass(
    explicit: type | str | None,
    raw_stored: bytes | None,
) -> type | None:
    """Resolve a TensorClass type from an explicit argument or a stored path.

    Args:
        explicit: A class, a ``"module.ClassName"`` string, or ``None``.
        raw_stored: The raw bytes read from the ``__tensorclass__`` Redis key
            (may be ``None`` if nothing was stored).

    Returns:
        The resolved TensorClass type, or ``None`` if there is nothing to
        resolve.
    """
    if isinstance(explicit, type):
        return explicit
    class_path: str | None = explicit
    if class_path is None and raw_stored is not None:
        class_path = (
            raw_stored.decode() if isinstance(raw_stored, bytes) else raw_stored
        )
    if class_path is None:
        return None
    module_path, _, class_name = class_path.rpartition(".")
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _compute_byte_ranges(
    shape: list[int],
    dtype: torch.dtype,
    idx,
) -> list[tuple[int, int]] | None:
    """Compute per-row ``(byte_offset, byte_length)`` pairs for the write path.

    Returns ``None`` when *idx* is an unsupported type, signalling that the
    caller should fall back to a full read-modify-write.

    Every index type (int, slice, list, tensor, bool) returns one
    ``(offset, row_size)`` entry per selected row.  This is the canonical
    decomposition used by the ``SETRANGES`` Lua script.
    """
    # Unwrap 1-element tuples produced by _SubTensorDict
    if isinstance(idx, tuple):
        if len(idx) == 1:
            idx = idx[0]
        else:
            return None  # multi-dim tuple: fall back

    # Ellipsis selects everything along the first dim
    if idx is Ellipsis:
        idx = slice(None)

    elem_size = torch.tensor([], dtype=dtype).element_size()
    row_size = elem_size
    for s in shape[1:]:
        row_size *= s

    if isinstance(idx, int):
        pos = idx % shape[0]
        return [(pos * row_size, row_size)]

    if isinstance(idx, (slice, range)):
        positions = range(*idx.indices(shape[0])) if isinstance(idx, slice) else idx
        if len(positions) == 0:
            return []
        # Contiguous step-1: single covering range
        if positions.step == 1:
            return [(positions[0] * row_size, len(positions) * row_size)]
        return [(p * row_size, row_size) for p in positions]

    if isinstance(idx, list):
        return [(int(p) * row_size, row_size) for p in idx]

    if isinstance(idx, torch.Tensor):
        if idx.dtype == torch.bool:
            positions = idx.nonzero(as_tuple=False).squeeze(-1).tolist()
        else:
            positions = idx.reshape(-1).tolist()
        return [(int(p) * row_size, row_size) for p in positions]

    return None  # unsupported index type


def _compute_covering_range(
    shape: list[int],
    dtype: torch.dtype,
    idx,
) -> tuple[int, int] | None:
    """Compute a single ``(byte_offset, byte_length)`` for the read path.

    For **int** and **slice** (any step), returns one covering range.
    Returns ``None`` for all other index types (scatter, unsupported).
    """
    if isinstance(idx, tuple):
        idx = idx[0] if len(idx) == 1 else None
    if idx is None:
        return None
    if idx is Ellipsis:
        idx = slice(None)

    elem_size = torch.tensor([], dtype=dtype).element_size()
    row_size = elem_size
    for s in shape[1:]:
        row_size *= s

    if isinstance(idx, int):
        pos = idx % shape[0]
        return (pos * row_size, row_size)

    if isinstance(idx, (slice, range)):
        positions = range(*idx.indices(shape[0])) if isinstance(idx, slice) else idx
        if len(positions) == 0:
            return (0, 0)
        start = positions[0]
        stop = positions[-1] + 1
        return (start * row_size, (stop - start) * row_size)

    return None


def _get_local_idx(idx, shape_0: int):
    """Return a local post-index to apply after fetching a covering range.

    For **int** and **step-1 slices** returns ``None`` (the fetched bytes
    match the result shape exactly).

    For **step > 1 slices** returns a ``slice(None, None, step)`` to stride
    the covering range.

    For tensor / list / bool returns ``None`` — the Lua path fetches exactly
    the needed rows so no post-processing is required.
    """
    if isinstance(idx, tuple):
        idx = idx[0] if len(idx) == 1 else idx
    if idx is Ellipsis:
        return None
    if isinstance(idx, int):
        return None
    if isinstance(idx, (slice, range)):
        positions = range(*idx.indices(shape_0)) if isinstance(idx, slice) else idx
        if len(positions) == 0 or positions.step == 1:
            return None
        return slice(None, None, positions.step)
    # tensor / list / bool — Lua fetches exact rows
    return None


def _is_scattered_index(idx) -> bool:
    """Return True when *idx* is tensor / list / bool (scattered, use Lua)."""
    if isinstance(idx, tuple):
        idx = idx[0] if len(idx) == 1 else idx
    if idx is Ellipsis:
        return False
    if isinstance(idx, (int, slice, range)):
        return False
    if isinstance(idx, (list, torch.Tensor)):
        return True
    return False


def _getitem_result_shape(
    shape: list[int],
    idx,
) -> list[int]:
    """Compute the result shape of ``tensor[idx]`` without creating a tensor.

    Only handles first-dimension indexing (same cases as ``_compute_byte_ranges``).
    """
    if isinstance(idx, tuple):
        if len(idx) == 1:
            idx = idx[0]
        else:
            # Multi-dim: not handled, let caller use torch logic
            return list(torch.zeros(shape)[idx].shape)

    if idx is Ellipsis:
        idx = slice(None)

    rest = list(shape[1:])

    if isinstance(idx, int):
        return rest

    if isinstance(idx, slice):
        n = len(range(*idx.indices(shape[0])))
        return [n] + rest

    if isinstance(idx, range):
        return [len(idx)] + rest

    if isinstance(idx, list):
        return [len(idx)] + rest

    if isinstance(idx, torch.Tensor):
        if idx.dtype == torch.bool:
            n = int(idx.sum().item())
        else:
            n = idx.numel()
        return [n] + rest

    # Fallback
    return list(torch.zeros(shape)[idx].shape)


class _StoreTDKeysView(_TensorDictKeysView):
    """Keys view for TensorDictStore backed by a Redis key registry."""

    def __iter__(self):
        td = self.tensordict
        # Get all registered keys from Redis
        all_keys = td._get_all_keys()
        prefix = td._prefix
        prefix_dot = prefix + _KEY_SEP if prefix else ""

        seen = set()
        for full_key in all_keys:
            # Filter keys belonging to this prefix level
            if prefix:
                if not full_key.startswith(prefix_dot):
                    continue
                relative = full_key[len(prefix_dot) :]
            else:
                relative = full_key

            parts = relative.split(_KEY_SEP)

            if self.include_nested:
                # Yield full nested tuple
                key = tuple(parts) if len(parts) > 1 else parts[0]
                if self.leaves_only and len(parts) > 1:
                    # Only yield if this is a leaf (no further nesting)
                    if key not in seen:
                        seen.add(key)
                        yield key
                elif not self.leaves_only or len(parts) == 1:
                    if key not in seen:
                        seen.add(key)
                        yield key
            else:
                # Only yield top-level keys
                top_key = parts[0]
                if top_key in seen:
                    continue
                seen.add(top_key)

                # Check if this is a leaf or a nested td
                is_leaf_key = len(parts) == 1
                if self.leaves_only and not is_leaf_key:
                    continue
                yield top_key

    def __contains__(self, key):
        key = unravel_key(key)
        td = self.tensordict
        prefix = td._prefix
        if isinstance(key, str):
            full_key = (prefix + _KEY_SEP + key) if prefix else key
        else:
            # tuple key
            full_key = (
                (prefix + _KEY_SEP + _KEY_SEP.join(key))
                if prefix
                else _KEY_SEP.join(key)
            )
        all_keys = td._get_all_keys()
        # Exact leaf match
        if full_key in all_keys:
            return True
        # Prefix match (nested tensordict)
        prefix_check = full_key + _KEY_SEP
        return any(k.startswith(prefix_check) for k in all_keys)

    def __len__(self):
        return sum(1 for _ in self)


class TensorDictStore(TensorDictBase):
    """A TensorDict backed by an in-memory key-value store for out-of-core storage.

    Supports `Redis <https://redis.io>`_, `Dragonfly <https://dragonflydb.io>`_,
    `KeyDB <https://docs.keydb.dev>`_, and any other Redis-wire-compatible server.

    Tensors are stored as raw bytes using ``torch.frombuffer``-compatible
    serialization.  Metadata (shape, dtype) is stored in server-side Hashes
    for fast introspection without downloading tensor data.

    All I/O is performed via an async ``redis.asyncio.Redis`` client running on
    a dedicated background event loop, ensuring non-blocking operation from the
    caller's perspective.

    Keyword Args:
        backend (str): Name of the server backend.  Accepted values include
            ``"redis"`` (default), ``"dragonfly"``, ``"keydb"``, or any
            other string (stored for documentation; all use the same wire
            protocol).
        host (str): Server hostname. Defaults to ``"localhost"``.
        port (int): Server port. Defaults to ``6379``.
        db (int): Database number. Defaults to ``0``.
        unix_socket_path (str, optional): Path to a Unix domain socket for
            local high-performance connections. Mutually exclusive with
            ``host``/``port``.
        prefix (str, optional): A namespace prefix for all keys.
            Defaults to ``"tensordict"``.
        batch_size (torch.Size or compatible): The TensorDict batch size.
            Defaults to ``torch.Size(())``.
        device (torch.device or compatible, optional): Target device for
            retrieved tensors. Defaults to ``None`` (CPU).
        client: An existing ``redis.asyncio.Redis`` client instance. If
            provided, ``host``/``port``/``db``/``unix_socket_path`` are ignored.
        td_id (str, optional): A unique identifier for this TensorDict in the
            store.  Defaults to a new UUID.  Pass an existing ID to reconnect
            to previously stored data.
        cache_metadata (bool): If ``True`` (default), cache tensor metadata
            locally to reduce round-trips.
        **redis_kwargs: Additional keyword arguments passed to
            ``redis.asyncio.Redis``.

    Examples:
        >>> from tensordict.store import TensorDictStore
        >>> # Redis backend (default)
        >>> td = TensorDictStore(batch_size=[100])
        >>> td["obs"] = torch.randn(100, 84)
        >>> td["obs"]  # fetched from the store
        >>> local_td = td.to_local()
        >>>
        >>> # Dragonfly backend (same wire protocol, up to 25x faster)
        >>> td = TensorDictStore(backend="dragonfly", host="dragonfly-host", batch_size=[100])

    .. note::
        Requires ``redis`` package: ``pip install redis``.

    .. note::
        Tensors are stored as CPU contiguous bytes.  The ``device`` attribute
        controls the device tensors are cast to upon retrieval.
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
        batch_size=None,
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

        if batch_size is None:
            batch_size = torch.Size(())

        self._locked_tensordicts = []
        self._lock_id = set()
        self._is_shared = False
        self._is_memmap = False

        self._backend = backend

        # Nested TensorDict cache
        self._nested_tensordicts: dict[str, TensorDictStore] = {}

        # Metadata cache: (shape, dtype) per key_path, shared with nested views
        self._cache_metadata = cache_metadata
        self._meta_cache: dict[str, tuple[list[int], torch.dtype]] | None = (
            {} if cache_metadata else None
        )

        # Keys cache: shared mutable container ``[set | None]`` holding the set
        # of all registered leaf key paths.  The list wrapper ensures that
        # nested views (which share the same list object) observe invalidation
        # (``_keys_cache[0] = None``) immediately without needing their own
        # attribute reassignment.
        self._keys_cache: list[set[str] | None] = [None]

        # Unique identifier for this TensorDict instance in the store
        self._td_id = td_id or str(uuid.uuid4())

        # Key prefix within this tensordict (for nested views)
        self._prefix = ""

        # Namespace prefix for store keys
        self._namespace = prefix

        # Connection parameters (for pickling/reconnection)
        self._host = host
        self._port = port
        self._db = db
        self._unix_socket_path = unix_socket_path
        self._redis_kwargs = redis_kwargs

        # Create a new event loop in a background thread
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

        self._batch_size = torch.Size(batch_size)
        self._device = torch.device(device) if device is not None else None

        # Fully-qualified class path for TensorClass round-trips (optional)
        self._tensorclass_cls: str | None = None

        # Persist batch_size and device to the store
        self._run_sync(self._apersist_metadata())

    @classmethod
    def _new_nested(cls, *, parent: TensorDictStore, key_prefix: str, batch_size=None):
        """Create a nested view sharing the parent's Redis client and event loop.

        This is an internal factory used for nested TensorDict access. The
        returned instance shares the same connection and TD identity but
        operates under a different key prefix.
        """
        obj = cls.__new__(cls)
        obj._locked_tensordicts = []
        obj._lock_id = set()
        obj._is_shared = False
        obj._is_memmap = False
        obj._nested_tensordicts = {}
        obj._td_dim_names = None

        obj._backend = parent._backend
        obj._td_id = parent._td_id
        obj._prefix = key_prefix
        obj._namespace = parent._namespace
        obj._cache_metadata = parent._cache_metadata
        obj._meta_cache = parent._meta_cache
        obj._keys_cache = parent._keys_cache

        obj._host = parent._host
        obj._port = parent._port
        obj._db = parent._db
        obj._unix_socket_path = parent._unix_socket_path
        obj._redis_kwargs = parent._redis_kwargs

        obj._client = parent._client
        obj._loop = parent._loop
        obj._thread = parent._thread
        obj._owns_loop = False

        obj._batch_size = (
            torch.Size(batch_size) if batch_size is not None else parent._batch_size
        )
        obj._device = parent._device
        obj._tensorclass_cls = None
        return obj

    def _run_sync(self, coro):
        """Run an async coroutine synchronously via the background event loop."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    # ---- Redis key construction ----

    def _redis_key(self, suffix: str) -> str:
        """Build a Redis key with hash tag for cluster compatibility.

        Format: {namespace}:{td_id}:{suffix}
        The {td_id} is wrapped in Redis hash tags so all keys for this
        TensorDict land on the same shard.
        """
        return f"{self._namespace}:{{{self._td_id}}}:{suffix}"

    def _data_key(self, key_path: str) -> str:
        """Redis key for tensor data."""
        return self._redis_key(f"d:{key_path}")

    def _meta_key(self, key_path: str) -> str:
        """Redis key for tensor metadata hash."""
        return self._redis_key(f"m:{key_path}")

    @property
    def _keys_registry_key(self) -> str:
        """Redis key for the Set of all leaf key paths."""
        return self._redis_key("__keys__")

    @property
    def _batch_size_key(self) -> str:
        """Redis key for stored batch size."""
        return self._redis_key("__batch_size__")

    @property
    def _device_key(self) -> str:
        """Redis key for stored device."""
        return self._redis_key("__device__")

    @property
    def _tensorclass_key(self) -> str:
        """Redis key for the stored TensorClass fully-qualified class path."""
        return self._redis_key("__tensorclass__")

    def _full_key_path(self, key: str) -> str:
        """Build the full dot-separated key path including prefix."""
        if self._prefix:
            return self._prefix + _KEY_SEP + key
        return key

    # ---- Async internal methods ----

    async def _apersist_metadata(self):
        """Persist batch_size, device, and optional tensorclass path to the store."""
        pipe = self._client.pipeline()
        pipe.set(self._batch_size_key, json.dumps(list(self._batch_size)))
        device_str = str(self._device) if self._device is not None else ""
        pipe.set(self._device_key, device_str)
        if self._tensorclass_cls is not None:
            pipe.set(self._tensorclass_key, self._tensorclass_cls)
        await pipe.execute()

    async def _aset_tensor(self, key_path: str, tensor: torch.Tensor):
        """Store a tensor's data and metadata in Redis."""
        data = _tensor_to_bytes(tensor)
        shape = list(tensor.shape)
        dtype = tensor.dtype
        meta = {
            "shape": json.dumps(shape),
            "dtype": _dtype_to_str(dtype),
        }
        pipe = self._client.pipeline()
        pipe.set(self._data_key(key_path), data)
        pipe.hset(self._meta_key(key_path), mapping=meta)
        pipe.sadd(self._keys_registry_key, key_path)
        await pipe.execute()
        if self._meta_cache is not None:
            self._meta_cache[key_path] = (shape, dtype)
        if self._keys_cache[0] is not None:
            self._keys_cache[0].add(key_path)

    async def _aset_non_tensor(self, key_path: str, value: Any):
        """Store a non-tensor value in Redis."""
        # Try JSON first, fall back to pickle
        try:
            serialized = json.dumps(value)
            encoding = "json"
        except (TypeError, ValueError):
            serialized = pickle.dumps(value)
            encoding = "pickle"

        meta = {
            "is_non_tensor": "1",
            "encoding": encoding,
        }
        data = (
            serialized if isinstance(serialized, bytes) else serialized.encode("utf-8")
        )
        pipe = self._client.pipeline()
        pipe.set(self._data_key(key_path), data)
        pipe.hset(self._meta_key(key_path), mapping=meta)
        pipe.sadd(self._keys_registry_key, key_path)
        await pipe.execute()
        if self._keys_cache[0] is not None:
            self._keys_cache[0].add(key_path)

    async def _aset_non_tensor_at(self, key_path: str, value: Any, idx: int | slice):
        """Read-modify-write a single element of a batched non-tensor key.

        On the first per-element write the storage format is promoted from a
        single-blob (``json`` / ``pickle``) to a ``json_array`` of length
        ``batch_size[0]``, enabling future per-element reads and writes.
        """
        pipe = self._client.pipeline()
        pipe.get(self._data_key(key_path))
        pipe.hgetall(self._meta_key(key_path))
        data, raw_meta = await pipe.execute()

        meta = _decode_meta(raw_meta) if raw_meta else {}
        batch_dim = self._batch_size[0] if self._batch_size else 1

        if data is not None and meta.get("encoding") == "json_array":
            text = data.decode() if isinstance(data, bytes) else data
            array = json.loads(text)
        elif data is not None:
            # Promote single-blob → json_array by broadcasting
            blob_val = self._deserialize_non_tensor(data, meta)
            array = [blob_val] * batch_dim
        else:
            # Key doesn't exist yet — create one filled with None
            array = [None] * batch_dim

        if isinstance(idx, int):
            array[idx] = value
        elif isinstance(idx, slice):
            indices = range(*idx.indices(len(array)))
            if not isinstance(value, (list, tuple)):
                value = [value] * len(indices)
            for i, v in zip(indices, value):
                array[i] = v
        else:
            raise TypeError(
                f"Non-tensor indexed write supports int/slice, got {type(idx)}"
            )

        serialized = json.dumps(array).encode("utf-8")
        new_meta = {"is_non_tensor": "1", "encoding": "json_array"}
        pipe = self._client.pipeline()
        pipe.set(self._data_key(key_path), serialized)
        pipe.hset(self._meta_key(key_path), mapping=new_meta)
        pipe.sadd(self._keys_registry_key, key_path)
        await pipe.execute()
        if self._keys_cache[0] is not None:
            self._keys_cache[0].add(key_path)

    async def _aget_tensor(self, key_path: str) -> torch.Tensor | None:
        """Retrieve a tensor from Redis. Returns None if not found.

        When the metadata cache already contains ``(shape, dtype)`` for this
        key, only a single ``GET`` is issued (no ``HGETALL``).
        """
        cached = (
            self._meta_cache.get(key_path) if self._meta_cache is not None else None
        )
        if cached is not None:
            # Fast path: metadata cached — single GET
            data = await self._client.get(self._data_key(key_path))
            if data is None:
                return None
            shape, dtype = cached
            tensor = _bytes_to_tensor(data, shape, dtype)
            if self._device is not None:
                tensor = tensor.to(self._device)
            return tensor

        # Slow path: pipeline GET + HGETALL
        pipe = self._client.pipeline()
        pipe.get(self._data_key(key_path))
        pipe.hgetall(self._meta_key(key_path))
        data, raw_meta = await pipe.execute()

        if data is None:
            return None

        meta = _decode_meta(raw_meta)

        if meta.get("is_non_tensor") == "1":
            return self._deserialize_non_tensor(data, meta)

        shape = json.loads(meta["shape"])
        dtype = _str_to_dtype(meta["dtype"])
        if self._meta_cache is not None:
            self._meta_cache[key_path] = (shape, dtype)
        tensor = _bytes_to_tensor(data, shape, dtype)
        if self._device is not None:
            tensor = tensor.to(self._device)
        return tensor

    async def _aget_metadata(self, key_path: str) -> dict:
        """Retrieve metadata for a key without downloading tensor data."""
        raw = await self._client.hgetall(self._meta_key(key_path))
        return _decode_meta(raw)

    async def _adel_key(self, key_path: str):
        """Delete a key's data, metadata, and registry entry."""
        pipe = self._client.pipeline()
        pipe.delete(self._data_key(key_path))
        pipe.delete(self._meta_key(key_path))
        pipe.srem(self._keys_registry_key, key_path)
        await pipe.execute()
        if self._meta_cache is not None:
            self._meta_cache.pop(key_path, None)
        if self._keys_cache[0] is not None:
            self._keys_cache[0].discard(key_path)

    async def _aget_all_keys(self) -> set[str]:
        """Get all registered key paths from Redis."""
        raw = await self._client.smembers(self._keys_registry_key)
        return {k.decode() if isinstance(k, bytes) else k for k in raw}

    async def _aget_metadata_batch(
        self, key_paths: list[str]
    ) -> dict[str, tuple[list[int], torch.dtype]]:
        """Get ``(shape, dtype)`` for multiple keys, using the local cache when available.

        Cache misses are fetched via a single Redis pipeline and stored back
        into ``_meta_cache`` (when caching is enabled).
        """
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

    async def _aget_batch_tensors(
        self, key_paths: list[str]
    ) -> dict[str, torch.Tensor]:
        """Batch-fetch multiple tensors using a Redis pipeline.

        Metadata is looked up from ``_meta_cache`` when available; only
        cache-missing keys trigger an ``HGETALL`` pipeline.  Data is always
        fetched via a single ``GET`` pipeline.
        """
        if not key_paths:
            return {}

        # -- split cached / uncached metadata --
        cached_meta: dict[str, tuple[list[int], torch.dtype]] = {}
        uncached_kps: list[str] = []
        for kp in key_paths:
            if self._meta_cache is not None and kp in self._meta_cache:
                cached_meta[kp] = self._meta_cache[kp]
            else:
                uncached_kps.append(kp)

        # -- single pipeline: data GETs + metadata HGETALLs for uncached --
        pipe = self._client.pipeline()
        for kp in key_paths:
            pipe.get(self._data_key(kp))
        for kp in uncached_kps:
            pipe.hgetall(self._meta_key(kp))
        raw = await pipe.execute()

        data_list = raw[: len(key_paths)]
        meta_list = raw[len(key_paths) :]

        # parse uncached metadata
        uncached_meta: dict[str, dict] = {}
        for kp, raw_meta in zip(uncached_kps, meta_list):
            uncached_meta[kp] = _decode_meta(raw_meta)

        result: dict[str, torch.Tensor] = {}
        for kp, data in zip(key_paths, data_list):
            if data is None:
                continue
            if kp in cached_meta:
                shape, dtype = cached_meta[kp]
                tensor = _bytes_to_tensor(data, shape, dtype)
            else:
                meta = uncached_meta[kp]
                if meta.get("is_non_tensor") == "1":
                    result[kp] = self._deserialize_non_tensor(data, meta)
                    continue
                shape = json.loads(meta["shape"])
                dtype = _str_to_dtype(meta["dtype"])
                if self._meta_cache is not None:
                    self._meta_cache[kp] = (shape, dtype)
                tensor = _bytes_to_tensor(data, shape, dtype)
            if self._device is not None:
                tensor = tensor.to(self._device)
            result[kp] = tensor
        return result

    # ---- Byte-range batch operations ----

    async def _abatch_get_at(
        self, key_paths: list[str], idx
    ) -> dict[str, torch.Tensor]:
        """Batch-fetch indexed slices of multiple tensors.

        Two strategies per key (always exactly **K** pipeline commands for
        **K** keys):

        * **int / slice (any step)** — a single ``GETRANGE`` fetches the
          covering range.  A local post-index (``[::step]``) is applied when
          the step is > 1.
        * **list / tensor / bool mask** — the ``GETRANGES`` Lua script
          executes all per-row ``GETRANGE`` calls server-side in one ``EVAL``,
          returning concatenated bytes.

        Falls back to full ``GET`` + local indexing for unsupported index types.
        """
        if not key_paths:
            return {}

        meta_map = await self._aget_metadata_batch(key_paths)
        scattered = _is_scattered_index(idx)

        pipe = self._client.pipeline()
        # (key_path, result_shape, dtype, local_idx, has_cmd)
        plan: list[tuple[str, list[int], torch.dtype, object, bool]] = []
        fallback_kps: list[str] = []

        for kp in key_paths:
            shape, dtype = meta_map[kp]
            result_shape = _getitem_result_shape(shape, idx)
            local_idx = _get_local_idx(idx, shape[0])

            if scattered:
                ranges = _compute_byte_ranges(shape, dtype, idx)
                if ranges is None:
                    fallback_kps.append(kp)
                    continue
                if not ranges:
                    plan.append((kp, result_shape, dtype, None, False))
                    continue
                argv: list[int] = []
                for byte_offset, byte_length in ranges:
                    argv.append(byte_offset)
                    argv.append(byte_length)
                pipe.eval(_LUA_GETRANGES, 1, self._data_key(kp), *argv)
            else:
                cr = _compute_covering_range(shape, dtype, idx)
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

        has_cmds = any(has_cmd for _, _, _, _, has_cmd in plan)
        raw_results = await pipe.execute() if has_cmds else []

        result: dict[str, torch.Tensor] = {}
        ri = 0
        for kp, result_shape, dtype, local_idx, has_cmd in plan:
            if not has_cmd:
                result[kp] = torch.empty(result_shape, dtype=dtype)
                continue
            data = raw_results[ri]
            ri += 1
            # For Lua path: data is already exactly the needed rows.
            # For GETRANGE path: data may be a covering range needing post-index.
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

        # Fallback: full GET + local index
        if fallback_kps:
            tensors = await self._aget_batch_tensors(fallback_kps)
            for kp, tensor in tensors.items():
                result[kp] = tensor[idx]

        return result

    async def _abatch_index(
        self, key_paths: list[str], idx
    ) -> dict[str, torch.Tensor | Any]:
        """Batch-fetch indexed slices of all leaf tensors in one pipeline.

        Combines metadata lookup, byte-range computation, and GETRANGE for
        tensor keys into a single round-trip. Non-tensor keys are handled
        separately via a full GET pipeline.

        Returns a flat ``{key_path: indexed_value}`` dict.
        """
        if not key_paths:
            return {}

        # -- Stage 1: fetch metadata for all keys --------------------------
        # We need to distinguish tensors from non-tensors, so we always
        # fetch raw metadata for uncached keys.
        cached_meta: dict[str, tuple[list[int], torch.dtype]] = {}
        uncached_kps: list[str] = []
        for kp in key_paths:
            if self._meta_cache is not None and kp in self._meta_cache:
                cached_meta[kp] = self._meta_cache[kp]
            else:
                uncached_kps.append(kp)

        # Pipeline HGETALL for uncached + GET for any non-tensor keys we
        # discover.  We don't yet know which are non-tensors, so we start
        # with metadata only.
        raw_metas: dict[str, dict] = {}
        if uncached_kps:
            pipe = self._client.pipeline()
            for kp in uncached_kps:
                pipe.hgetall(self._meta_key(kp))
            meta_results = await pipe.execute()
            for kp, raw_meta in zip(uncached_kps, meta_results):
                raw_metas[kp] = _decode_meta(raw_meta)

        # Classify keys
        tensor_kps: list[str] = []
        non_tensor_kps: list[str] = []
        all_meta: dict[str, tuple[list[int], torch.dtype]] = dict(cached_meta)
        for kp, meta in raw_metas.items():
            if meta.get("is_non_tensor") == "1":
                non_tensor_kps.append(kp)
            else:
                shape = json.loads(meta["shape"])
                dtype = _str_to_dtype(meta["dtype"])
                all_meta[kp] = (shape, dtype)
                if self._meta_cache is not None:
                    self._meta_cache[kp] = (shape, dtype)
                tensor_kps.append(kp)
        # Cached keys are always tensors (non-tensors are never cached).
        for kp in cached_meta:
            tensor_kps.append(kp)

        # -- Stage 2: pipelined byte-range reads (hybrid) ---------------------
        pipe = self._client.pipeline()
        scattered = _is_scattered_index(idx)

        # (kp, result_shape, dtype, local_idx, has_cmd)
        plan: list[tuple[str, list[int], torch.dtype, object, bool]] = []
        fallback_tensor_kps: list[str] = []

        for kp in tensor_kps:
            shape, dtype = all_meta[kp]
            result_shape = _getitem_result_shape(shape, idx)
            local_idx = _get_local_idx(idx, shape[0])

            if scattered:
                ranges = _compute_byte_ranges(shape, dtype, idx)
                if ranges is None:
                    fallback_tensor_kps.append(kp)
                    continue
                if not ranges:
                    plan.append((kp, result_shape, dtype, None, False))
                    continue
                argv: list[int] = []
                for byte_offset, byte_length in ranges:
                    argv.append(byte_offset)
                    argv.append(byte_length)
                pipe.eval(_LUA_GETRANGES, 1, self._data_key(kp), *argv)
            else:
                cr = _compute_covering_range(shape, dtype, idx)
                if cr is None:
                    fallback_tensor_kps.append(kp)
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

        # Full GET for fallback tensors
        for kp in fallback_tensor_kps:
            pipe.get(self._data_key(kp))
            pipe.hgetall(self._meta_key(kp))

        # Full GET for non-tensor keys
        for kp in non_tensor_kps:
            pipe.get(self._data_key(kp))

        has_cmds = (
            any(has_cmd for _, _, _, _, has_cmd in plan)
            or fallback_tensor_kps
            or non_tensor_kps
        )
        all_results = await pipe.execute() if has_cmds else []

        # -- Stage 3: reassemble results -----------------------------------
        result: dict[str, torch.Tensor | Any] = {}

        flat_idx = 0
        for kp, result_shape, dtype, local_idx, has_cmd in plan:
            if not has_cmd:
                result[kp] = torch.empty(result_shape, dtype=dtype)
                continue
            data = all_results[flat_idx]
            flat_idx += 1
            tensor = _bytes_to_tensor(
                data,
                result_shape if local_idx is None else [-1] + list(all_meta[kp][0][1:]),
                dtype,
            )
            if local_idx is not None:
                tensor = tensor[local_idx]
                tensor = tensor.reshape(result_shape)
            if self._device is not None:
                tensor = tensor.to(self._device)
            result[kp] = tensor

        # Fallback tensor results (full GET + index locally)
        for kp in fallback_tensor_kps:
            data = all_results[flat_idx]
            raw_meta = all_results[flat_idx + 1]
            flat_idx += 2
            meta = _decode_meta(raw_meta)
            shape = json.loads(meta["shape"])
            dtype = _str_to_dtype(meta["dtype"])
            tensor = _bytes_to_tensor(data, shape, dtype)
            if self._device is not None:
                tensor = tensor.to(self._device)
            result[kp] = tensor[idx]

        # Non-tensor results — index into json_array when applicable
        for kp in non_tensor_kps:
            data = all_results[flat_idx]
            flat_idx += 1
            result[kp] = self._deserialize_non_tensor(data, raw_metas[kp], idx=idx)

        return result

    async def _abatch_set_at(self, items: dict[str, tuple[torch.Tensor, object]]):
        """Batch-write slices of multiple tensors.

        *items* maps ``key_path`` to ``(value_tensor, idx)``.

        Three strategies, chosen per-key:

        1. **Direct SETRANGE** (int, step-1 slice) — value bytes match the
           covering range exactly.
        2. **Lua SETRANGES** (list / tensor / bool) — per-row
           ``(offset, data)`` pairs executed server-side in one ``EVAL``.
        3. **Covering-range RMW** (step>1 slice) — fetch the covering
           range, patch locally, write back.  Two commands per key but avoids
           sending thousands of Lua arguments.
        4. **Full RMW** fallback for unsupported indices.

        If a key has no metadata yet (first write to an empty store via
        index), a zero-initialised tensor is created with shape
        ``[batch_size[0], *value.shape]`` before proceeding.
        """
        if not items:
            return

        key_paths = list(items.keys())

        # Create new keys that have no metadata yet (first indexed write).
        all_keys = await self._aget_all_keys()
        new_kps = [kp for kp in key_paths if kp not in all_keys]
        if new_kps:
            batch_dim = self._batch_size[0] if self._batch_size else 0
            for kp in new_kps:
                value, _ = items[kp]
                full_shape = [batch_dim] + list(value.shape)
                full_tensor = torch.zeros(full_shape, dtype=value.dtype)
                await self._aset_tensor(kp, full_tensor)

        meta_map = await self._aget_metadata_batch(key_paths)

        # ---- Phase 1: classify each key --------------------------------
        direct_kps: list[str] = []  # int, step-1 slice
        lua_kps: list[str] = []  # tensor / list / bool
        rmw_kps: list[str] = []  # step>1 slice (covering RMW)
        fallback_kps: list[str] = []

        for kp in key_paths:
            _, idx = items[kp]
            shape, dtype = meta_map[kp]
            ranges = _compute_byte_ranges(shape, dtype, idx)
            if ranges is None:
                fallback_kps.append(kp)
            elif not ranges:
                pass  # empty index, nothing to write
            elif len(ranges) == 1:
                direct_kps.append(kp)
            elif _is_scattered_index(idx):
                lua_kps.append(kp)
            else:
                # step>1 slice — many ranges, but a contiguous covering range
                rmw_kps.append(kp)

        # ---- Phase 2: single pipeline for direct + lua -------------------
        pipe = self._client.pipeline()
        has_pipe_cmds = False

        for kp in direct_kps:
            value, idx = items[kp]
            shape, dtype = meta_map[kp]
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

        # ---- Phase 3: covering-range RMW for step>1 slices ---------------
        if rmw_kps:
            # Pipeline 1: GETRANGE covering ranges
            pipe = self._client.pipeline()
            cr_data: list[tuple[str, int, int, list[int], torch.dtype]] = []
            for kp in rmw_kps:
                shape, dtype = meta_map[kp]
                _, idx = items[kp]
                cr = _compute_covering_range(shape, dtype, idx)
                byte_offset, byte_length = cr
                pipe.getrange(
                    self._data_key(kp),
                    byte_offset,
                    byte_offset + byte_length - 1,
                )
                cr_data.append((kp, byte_offset, byte_length, shape, dtype))
            raw_covers = await pipe.execute()

            # Patch locally, then Pipeline 2: SETRANGE
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

        # ---- Phase 4: full RMW fallback ----------------------------------
        if fallback_kps:
            await self._abatch_read_modify_write(
                fallback_kps, {kp: items[kp] for kp in fallback_kps}
            )

    async def _abatch_read_modify_write(
        self, key_paths: list[str], items: dict[str, tuple[torch.Tensor, object]]
    ):
        """Pipelined full read-modify-write fallback for exotic indices.

        Downloads complete tensors for all *key_paths* in one pipeline,
        patches them in memory, and re-uploads in a second pipeline.
        """
        # Pipeline GET data + metadata
        pipe = self._client.pipeline()
        for kp in key_paths:
            pipe.get(self._data_key(kp))
            pipe.hgetall(self._meta_key(kp))
        raw_results = await pipe.execute()

        # Patch in memory, prepare SET pipeline
        pipe = self._client.pipeline()
        for i, kp in enumerate(key_paths):
            data, raw_meta = raw_results[2 * i], raw_results[2 * i + 1]
            meta = _decode_meta(raw_meta)
            shape = json.loads(meta["shape"])
            dtype = _str_to_dtype(meta["dtype"])
            existing = _bytes_to_tensor(data, shape, dtype)
            value, idx = items[kp]
            existing[idx] = value
            pipe.set(self._data_key(kp), _tensor_to_bytes(existing))
        await pipe.execute()

    # ---- Sync helpers ----

    def _get_all_keys(self) -> set[str]:
        """Get all registered key paths (sync wrapper).

        Uses the local ``_keys_cache`` when available, falling back to a
        Redis ``SMEMBERS`` round-trip and populating the cache for next time.
        """
        if self._keys_cache[0] is not None:
            return self._keys_cache[0]
        keys = self._run_sync(self._aget_all_keys())
        if self._cache_metadata:
            self._keys_cache[0] = keys
        return keys

    @staticmethod
    def _deserialize_non_tensor(data: bytes, meta: dict, idx=None) -> Any:
        """Deserialize a non-tensor value from Redis.

        When *idx* is provided the element(s) at *idx* are returned.
        For ``json_array`` encoding this indexes directly into the stored
        array.  For scalar encodings (``json`` / ``pickle``) the single
        stored value is returned as-is (broadcast semantics).
        """
        encoding = meta.get("encoding", "json")
        text = data.decode() if isinstance(data, bytes) else data

        if encoding == "json_array":
            array = json.loads(text)
            if idx is not None:
                if isinstance(idx, int):
                    return array[idx]
                if isinstance(idx, slice):
                    return array[idx]
                if isinstance(idx, (list, torch.Tensor)):
                    indices = idx
                    if isinstance(indices, torch.Tensor):
                        indices = indices.tolist()
                    return [array[i] for i in indices]
            # Full-batch read: wrap in NonTensorStack for TensorClass compat
            from tensordict._lazy import LazyStackedTensorDict
            from tensordict.tensorclass import NonTensorData

            return LazyStackedTensorDict(
                *[NonTensorData(data=v, batch_size=[]) for v in array]
            )

        # Scalar encoding — return the single stored value (broadcast).
        if encoding == "json":
            return json.loads(text)
        return pickle.loads(data)

    # ---- TensorDictBase interface: batch_size / device / names ----

    @property
    def batch_size(self) -> torch.Size:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        old = self._batch_size
        try:
            self._batch_size = torch.Size(value)
            self._check_batch_size(self._batch_size)
            self._run_sync(self._apersist_metadata())
        except ValueError:
            self._batch_size = old

    @property
    def device(self) -> torch.device | None:
        return self._device

    @device.setter
    def device(self, value):
        self._device = torch.device(value) if value is not None else None
        self._run_sync(self._apersist_metadata())

    _erase_names = TensorDict._erase_names
    _has_names = TensorDict._has_names
    _set_names = TensorDict._set_names
    names = TensorDict.names

    def _rename_subtds(self, names):
        if names is None:
            names = [None] * self.ndim
        for item in self._nested_tensordicts.values():
            if is_tensor_collection(item):
                td_names = list(names) + [None] * (item.ndim - self.ndim)
                item.rename_(*td_names)

    # ---- Key access / mutation ----

    def _index_tensordict(self, index, new_batch_size=None, names=None):
        """Eagerly fetch all leaf tensors for the given index in a single pipeline.

        Overrides the default ``TensorDict._index_tensordict`` (which iterates
        ``self.items()`` -- one Redis round-trip per key) with a batched
        approach that issues a single pipeline of ``GETRANGE`` commands across
        all keys.
        """
        batch_size = self.batch_size

        if new_batch_size is None:
            new_batch_size = _getitem_batch_size(batch_size, index)
        if names is None:
            names = self._get_names_idx(index)

        # Collect all leaf key paths under our prefix
        all_keys = self._get_all_keys()
        prefix = (self._prefix + _KEY_SEP) if self._prefix else ""
        leaf_kps = sorted(k for k in all_keys if k.startswith(prefix) or not prefix)

        # Single batched pipeline for all keys
        result_map = self._run_sync(self._abatch_index(leaf_kps, index))

        # Build nested source dict from flat key paths
        prefix_len = len(prefix)
        source: dict = {}
        for kp, value in result_map.items():
            rel_key = kp[prefix_len:] if prefix else kp
            parts = rel_key.split(_KEY_SEP)
            d = source
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value

        # Recursively convert nested dicts into TensorDicts
        def _build(d, bs):
            for k, v in d.items():
                if isinstance(v, dict):
                    # Nested TD: batch_size = new_batch_size + extra dims
                    d[k] = _build(v, bs)
            return TensorDict._new_unsafe(
                source=d,
                batch_size=bs,
                device=self._device,
                names=names,
            )

        return _build(source, new_batch_size)

    def __setitem__(self, index, value):
        index_unravel = _unravel_key_to_tuple(index)
        if index_unravel:
            return self.set(index_unravel, value, inplace=True)

        # Index-based assignment: bypass _SubTensorDict, batch SETRANGE
        if isinstance(index, list):
            index = torch.tensor(index)

        if not isinstance(value, TensorDictBase):
            value = TensorDict.from_dict(value, batch_size=[])

        # Collect all leaf items, separating tensors from non-tensors.
        # ``leaves_only=True`` hides non-tensor keys (NonTensorData is a
        # TensorDictBase subclass), so we iterate with ``leaves_only=False``
        # and classify each value ourselves.
        from tensordict.tensorclass import NonTensorData

        tensor_items: dict[str, tuple[torch.Tensor, object]] = {}
        non_tensor_items: list[tuple[str, Any]] = []
        for key in value.keys(include_nested=True, leaves_only=False):
            key_tuple = _unravel_key_to_tuple(key)
            key_path = self._full_key_path(_KEY_SEP.join(key_tuple))
            val = value.get(key)
            if isinstance(val, NonTensorData):
                non_tensor_items.append((key_path, val.data))
            elif is_non_tensor(val):
                non_tensor_items.append((key_path, val.tolist()))
            elif isinstance(val, torch.Tensor):
                tensor_items[key_path] = (val, index)
            elif not is_tensor_collection(val):
                non_tensor_items.append((key_path, val))

        if tensor_items:
            self._run_sync(self._abatch_set_at(tensor_items))
        for key_path, raw_val in non_tensor_items:
            self._run_sync(self._aset_non_tensor_at(key_path, raw_val, index))

    def _get_str(self, key, default=NO_DEFAULT, **kwargs):
        key_path = self._full_key_path(key)
        all_keys = self._get_all_keys()

        # Check if it's a nested tensordict (key is a prefix of other keys)
        prefix_check = key_path + _KEY_SEP
        nested_keys = [k for k in all_keys if k.startswith(prefix_check)]
        if nested_keys:
            # Return a nested TensorDictStore view
            nested = self._nested_tensordicts.get(key)
            if nested is None:
                nested = TensorDictStore._new_nested(
                    parent=self,
                    key_prefix=key_path,
                )
                self._nested_tensordicts[key] = nested
            return nested

        # Check if it's a direct leaf key
        if key_path in all_keys:
            result = self._run_sync(self._aget_tensor(key_path))
            if result is not None:
                return result

        if default is not NO_DEFAULT:
            return default
        raise KeyError(f"key {key} not found in {type(self).__name__}")

    def _get_tuple(self, key, default=NO_DEFAULT, **kwargs):
        """Resolve a nested key tuple directly without intermediate views.

        For ``td["obs", "deep", "z"]``, join to ``"obs.deep.z"`` and check
        the keys set in one shot, avoiding per-level ``_get_str`` calls and
        intermediate ``TensorDictStore`` creation.
        """
        if len(key) == 1:
            return self._get_str(key[0], default, **kwargs)

        full_path = self._full_key_path(_KEY_SEP.join(key))
        all_keys = self._get_all_keys()

        # Direct leaf hit — single fetch
        if full_path in all_keys:
            result = self._run_sync(self._aget_tensor(full_path))
            if result is not None:
                return result

        # Check if it's a nested prefix (e.g. ("obs",) when keys contain "obs.x")
        prefix_check = full_path + _KEY_SEP
        if any(k.startswith(prefix_check) for k in all_keys):
            nested = self._nested_tensordicts.get(key[0])
            if nested is None:
                nested = TensorDictStore._new_nested(
                    parent=self,
                    key_prefix=self._full_key_path(key[0]),
                )
                self._nested_tensordicts[key[0]] = nested
            return nested._get_tuple(key[1:], default, **kwargs)

        if default is not NO_DEFAULT:
            return default
        raise KeyError(f"key {key} not found in {type(self).__name__}")

    def _get_at_str(self, key, idx, default=NO_DEFAULT, **kwargs):
        """Retrieve an indexed slice of a single tensor via ``GETRANGE``."""
        key_path = self._full_key_path(key)
        all_keys = self._get_all_keys()

        # Nested tensordict: delegate
        prefix_check = key_path + _KEY_SEP
        if any(k.startswith(prefix_check) for k in all_keys):
            td = self._get_str(key, default, **kwargs)
            if td is default:
                return td
            return td[idx]

        if key_path not in all_keys:
            if default is not NO_DEFAULT:
                return default
            raise KeyError(f"key {key} not found in {type(self).__name__}")

        result = self._run_sync(self._abatch_get_at([key_path], idx))
        tensor = result.get(key_path)
        if tensor is None:
            if default is not NO_DEFAULT:
                return default
            raise KeyError(key)
        return tensor

    def _get_at_tuple(self, key, idx, default=NO_DEFAULT, **kwargs):
        """Retrieve an indexed slice via nested key tuple."""
        key = _unravel_key_to_tuple(key)
        if len(key) == 1:
            return self._get_at_str(key[0], idx, default=default, **kwargs)

        # Direct resolution: join tuple to full path and check for leaf key
        full_path = self._full_key_path(_KEY_SEP.join(key))
        all_keys = self._get_all_keys()

        if full_path in all_keys:
            result = self._run_sync(self._abatch_get_at([full_path], idx))
            tensor = result.get(full_path)
            if tensor is not None:
                return tensor

        # Fall back to per-level navigation (nested prefix case)
        first = self._get_str(key[0], default, **kwargs)
        if first is default:
            return default
        return first._get_at_tuple(key[1:], idx, default=default, **kwargs)

    def _convert_inplace(self, inplace, key):
        """Convert BEST_ATTEMPT_INPLACE sentinel to a bool."""
        if inplace is not False:
            key_path = self._full_key_path(key)
            all_keys = self._get_all_keys()
            has_key = key_path in all_keys
            # Also check if it's a nested key prefix
            if not has_key:
                prefix_check = key_path + _KEY_SEP
                has_key = any(k.startswith(prefix_check) for k in all_keys)
            if inplace is True and not has_key:
                raise KeyError(
                    _KEY_ERROR.format(key, type(self).__name__, sorted(self.keys()))
                )
            inplace = has_key
        return inplace

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
        if not validated:
            value = self._validate_value(value, check_shape=True)
        if self.is_locked and not ignore_lock:
            if not inplace:
                raise RuntimeError(_LOCK_ERROR)

        key_path = self._full_key_path(key)

        if is_non_tensor(value):
            from tensordict.tensorclass import NonTensorData

            if isinstance(value, NonTensorData):
                raw_value = value.data
            else:
                raw_value = value.tolist()
            self._run_sync(self._aset_non_tensor(key_path, raw_value))
            return self

        if is_tensor_collection(value):
            # Create nested tensordict and populate it
            target_td = self._nested_tensordicts.get(key)
            if target_td is None:
                target_td = TensorDictStore._new_nested(
                    parent=self,
                    key_prefix=key_path,
                    batch_size=value.batch_size,
                )
                self._nested_tensordicts[key] = target_td
            target_td.update(value, inplace=inplace)
            return self

        if isinstance(value, torch.Tensor):
            self._run_sync(self._aset_tensor(key_path, value))
            return self

        # Fallback: try to convert to tensor
        try:
            value = torch.as_tensor(value)
            self._run_sync(self._aset_tensor(key_path, value))
        except (ValueError, TypeError):
            self._run_sync(self._aset_non_tensor(key_path, value))
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
        elif key[0] in self.keys():
            return self._get_str(key[0], NO_DEFAULT)._set_tuple(
                key[1:],
                value,
                inplace=inplace,
                validated=validated,
                non_blocking=non_blocking,
            )
        # Direct set with full key path
        key_path = self._full_key_path(_KEY_SEP.join(key))
        if not validated:
            value = self._validate_value(value, check_shape=True)
        if self.is_locked and not inplace:
            raise RuntimeError(_LOCK_ERROR)

        if isinstance(value, torch.Tensor):
            self._run_sync(self._aset_tensor(key_path, value))
        elif is_non_tensor(value):
            from tensordict.tensorclass import NonTensorData

            raw_value = (
                value.data if isinstance(value, NonTensorData) else value.tolist()
            )
            self._run_sync(self._aset_non_tensor(key_path, raw_value))
        elif is_tensor_collection(value):
            nested_prefix = self._full_key_path(_KEY_SEP.join(key))
            nested = TensorDictStore._new_nested(
                parent=self,
                key_prefix=nested_prefix,
                batch_size=value.batch_size,
            )
            nested.update(value, inplace=inplace)
        else:
            self._run_sync(self._aset_non_tensor(key_path, value))
        return self

    def _set_at_str(self, key, value, idx, *, validated, non_blocking):
        key_path = self._full_key_path(key)
        if not isinstance(value, torch.Tensor):
            # Non-tensor indexed write: RMW on the JSON array
            self._run_sync(self._aset_non_tensor_at(key_path, value, idx))
            return self
        items = {key_path: (value, idx)}
        self._run_sync(self._abatch_set_at(items))
        return self

    def _set_at_tuple(self, key, value, idx, *, validated, non_blocking):
        key = _unravel_key_to_tuple(key)
        if len(key) == 1:
            return self._set_at_str(
                key[0], value, idx, validated=validated, non_blocking=non_blocking
            )
        td = self._get_str(key[0], NO_DEFAULT)
        return td._set_at_tuple(
            key[1:], value, idx, validated=validated, non_blocking=non_blocking
        )

    # ---- Keys / structure ----

    def keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
        *,
        sort: bool = False,
    ) -> _StoreTDKeysView:
        return _StoreTDKeysView(
            tensordict=self,
            include_nested=include_nested,
            leaves_only=leaves_only,
            is_leaf=is_leaf,
            sort=sort,
        )

    @lock_blocked
    def del_(self, key: NestedKey) -> TensorDictStore:
        if isinstance(key, str):
            key_path = self._full_key_path(key)
        else:
            key = _unravel_key_to_tuple(key)
            key_path = self._full_key_path(_KEY_SEP.join(key))

        all_keys = self._get_all_keys()
        # Delete exact key
        if key_path in all_keys:
            self._run_sync(self._adel_key(key_path))

        # Delete nested keys (prefix match)
        prefix_check = key_path + _KEY_SEP
        nested_to_delete = [k for k in all_keys if k.startswith(prefix_check)]
        for nested_key in nested_to_delete:
            self._run_sync(self._adel_key(nested_key))

        # Clean up nested cache
        cache_key = key if isinstance(key, str) else key[0]
        self._nested_tensordicts.pop(cache_key, None)

        return self

    def rename_key_(
        self, old_key: NestedKey, new_key: NestedKey, safe: bool = False
    ) -> TensorDictStore:
        if isinstance(old_key, str):
            old_path = self._full_key_path(old_key)
        else:
            old_key_tuple = _unravel_key_to_tuple(old_key)
            old_path = self._full_key_path(_KEY_SEP.join(old_key_tuple))

        if isinstance(new_key, str):
            new_path = self._full_key_path(new_key)
        else:
            new_key_tuple = _unravel_key_to_tuple(new_key)
            new_path = self._full_key_path(_KEY_SEP.join(new_key_tuple))

        if safe and new_path in self._get_all_keys():
            raise KeyError(f"key {new_key} already present in {type(self).__name__}.")

        async def _arename(old_path, new_path):
            all_keys = await self._aget_all_keys()
            # Rename exact match
            if old_path in all_keys:
                pipe = self._client.pipeline()
                pipe.rename(self._data_key(old_path), self._data_key(new_path))
                pipe.rename(self._meta_key(old_path), self._meta_key(new_path))
                pipe.srem(self._keys_registry_key, old_path)
                pipe.sadd(self._keys_registry_key, new_path)
                await pipe.execute()
            # Rename nested keys
            old_prefix = old_path + _KEY_SEP
            new_prefix = new_path + _KEY_SEP
            nested = [k for k in all_keys if k.startswith(old_prefix)]
            for k in nested:
                new_k = new_prefix + k[len(old_prefix) :]
                pipe = self._client.pipeline()
                pipe.rename(self._data_key(k), self._data_key(new_k))
                pipe.rename(self._meta_key(k), self._meta_key(new_k))
                pipe.srem(self._keys_registry_key, k)
                pipe.sadd(self._keys_registry_key, new_k)
                await pipe.execute()

        self._run_sync(_arename(old_path, new_path))
        # Invalidate keys cache — rename changes the set of keys
        self._keys_cache[0] = None
        if self._meta_cache is not None:
            # Transfer cached metadata from old to new path
            old_meta = self._meta_cache.pop(old_path, None)
            if old_meta is not None:
                self._meta_cache[new_path] = old_meta
        return self

    def entry_class(self, key: NestedKey) -> type:
        if isinstance(key, str):
            key_path = self._full_key_path(key)
        else:
            key = _unravel_key_to_tuple(key)
            key_path = self._full_key_path(_KEY_SEP.join(key))

        all_keys = self._get_all_keys()
        # Direct leaf
        if key_path in all_keys:
            meta = self._run_sync(self._aget_metadata(key_path))
            if meta.get("is_non_tensor") == "1":
                from tensordict.tensorclass import NonTensorData

                return NonTensorData
            return torch.Tensor

        # Nested TD
        prefix_check = key_path + _KEY_SEP
        if any(k.startswith(prefix_check) for k in all_keys):
            return TensorDictStore

        raise KeyError(f"key {key} not found in {type(self).__name__}")

    # ---- Locking ----

    def _propagate_lock(self, lock_parents_weakrefs=None, *, is_compiling):
        self._is_locked = True
        if lock_parents_weakrefs is not None:
            lock_parents_weakrefs = [
                ref
                for ref in lock_parents_weakrefs
                if not any(refref is ref for refref in self._lock_parents_weakrefs)
            ]
        if not is_compiling:
            is_root = lock_parents_weakrefs is None
            if is_root:
                lock_parents_weakrefs = []
            else:
                self._lock_parents_weakrefs = (
                    self._lock_parents_weakrefs + lock_parents_weakrefs
                )
            lock_parents_weakrefs = list(lock_parents_weakrefs)
            lock_parents_weakrefs.append(weakref.ref(self))
        for _td in self._nested_tensordicts.values():
            _td._propagate_lock(lock_parents_weakrefs, is_compiling=is_compiling)

    @erase_cache
    def _propagate_unlock(self):
        self._is_locked = False
        self._is_shared = False
        self._is_memmap = False
        sub_tds = []
        for _td in self._nested_tensordicts.values():
            sub_tds.extend(_td._propagate_unlock())
            sub_tds.append(_td)
        return sub_tds

    # ---- Materialization ----

    def to_tensordict(self, *, retain_none: bool | None = None) -> TensorDict:
        """Materialize all leaf tensors into a local ``TensorDict``.

        Uses a single Redis pipeline to fetch all data in one round-trip,
        avoiding per-key ``_get_str`` overhead.
        """
        all_keys = self._get_all_keys()
        # Filter to keys under our prefix
        if self._prefix:
            prefix_check = self._prefix + _KEY_SEP
            leaf_kps = sorted(k for k in all_keys if k.startswith(prefix_check))
        else:
            leaf_kps = sorted(all_keys)

        if not leaf_kps:
            return TensorDict({}, batch_size=self.batch_size, device=self.device)

        result_map = self._run_sync(self._aget_batch_tensors(leaf_kps))

        # Build nested dict structure
        strip = len(self._prefix) + len(_KEY_SEP) if self._prefix else 0
        source: dict = {}
        for kp in leaf_kps:
            value = result_map.get(kp)
            if value is None:
                continue
            rel = kp[strip:]
            parts = rel.split(_KEY_SEP)
            d = source
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value
        return TensorDict(
            source,
            batch_size=self.batch_size,
            device=self.device,
            names=self._maybe_names(),
        )

    def to_local(self) -> TensorDict:
        """Pull the entire Redis-backed dict into local RAM.

        Alias for :meth:`to_tensordict`.
        """
        return self.to_tensordict()

    def contiguous(self) -> TensorDict:
        """Materialize into a regular TensorDict."""
        return self.to_tensordict()

    # ---- Construction ----

    @classmethod
    def from_dict(
        cls,
        input_dict,
        *,
        backend: STORE_BACKENDS = "redis",
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        unix_socket_path: str | None = None,
        prefix: str = "tensordict",
        auto_batch_size: bool = False,
        batch_size=None,
        device=None,
        **kwargs,
    ):
        """Create a TensorDictStore from a dictionary or TensorDict.

        Args:
            input_dict: A dictionary or TensorDict to store.

        Keyword Args:
            backend: Store backend (``"redis"``, ``"dragonfly"``, etc.).
            host, port, db, unix_socket_path, prefix: Connection params.
            auto_batch_size: If True, infer batch_size from data.
            batch_size: Explicit batch_size.
            device: Target device for tensor retrieval.
            **kwargs: Additional connection kwargs.

        Returns:
            A new TensorDictStore.
        """
        if batch_size is None:
            if is_tensor_collection(input_dict):
                batch_size = input_dict.batch_size
            else:
                batch_size = torch.Size([])

        connect_kwargs = {}
        if unix_socket_path is not None:
            connect_kwargs["unix_socket_path"] = unix_socket_path
        else:
            connect_kwargs["host"] = host
            connect_kwargs["port"] = port
        connect_kwargs["db"] = db

        out = cls(
            backend=backend,
            batch_size=batch_size,
            device=device,
            prefix=prefix,
            **connect_kwargs,
            **kwargs,
        )
        if is_tensor_collection(input_dict):
            out.update(input_dict)
        else:
            out.update(TensorDict(input_dict, batch_size=batch_size))
        return out

    @classmethod
    def from_tensordict(
        cls,
        td: TensorDictBase,
        *,
        backend: STORE_BACKENDS = "redis",
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        unix_socket_path: str | None = None,
        prefix: str = "tensordict",
        device=None,
        **kwargs,
    ) -> TensorDictStore:
        """Upload a TensorDict to a key-value store.

        Creates a new :class:`TensorDictStore` and copies all tensor data from
        the provided TensorDict into the store.

        Args:
            td (TensorDictBase): The source TensorDict whose data will be
                stored.

        Keyword Args:
            backend (str): Store backend (``"redis"``, ``"dragonfly"``, etc.).
            host (str): Server hostname. Defaults to ``"localhost"``.
            port (int): Server port. Defaults to ``6379``.
            db (int): Database number. Defaults to ``0``.
            unix_socket_path (str, optional): Unix socket path.
            prefix (str): Key namespace. Defaults to ``"tensordict"``.
            device (torch.device, optional): Device override for retrieved
                tensors. If ``None``, uses the source TensorDict's device.
            **kwargs: Extra connection kwargs.

        Returns:
            A new TensorDictStore backed by the uploaded data.

        Examples:
            >>> local = TensorDict({"obs": torch.randn(10, 84)}, [10])
            >>> remote = TensorDictStore.from_tensordict(local, host="my-server")
            >>> remote["obs"].shape
            torch.Size([10, 84])
        """
        if device is None:
            device = td.device

        connect_kwargs = {}
        if unix_socket_path is not None:
            connect_kwargs["unix_socket_path"] = unix_socket_path
        else:
            connect_kwargs["host"] = host
            connect_kwargs["port"] = port
        connect_kwargs["db"] = db

        from tensordict._lazy import LazyStackedTensorDict

        if isinstance(td, LazyStackedTensorDict):
            return LazyStackedTensorDictStore.from_lazy_stack(
                td,
                backend=backend,
                host=host,
                port=port,
                db=db,
                unix_socket_path=unix_socket_path,
                prefix=prefix,
                device=device,
                **kwargs,
            )

        # Detect TensorClass and record its fully-qualified class path
        td_type = type(td)
        tensorclass_cls: str | None = None
        if _is_tensorclass(td_type):
            tensorclass_cls = f"{td_type.__module__}.{td_type.__qualname__}"

        out = cls(
            backend=backend,
            batch_size=td.batch_size,
            device=device,
            prefix=prefix,
            **connect_kwargs,
            **kwargs,
        )
        if tensorclass_cls is not None:
            out._tensorclass_cls = tensorclass_cls
            out._run_sync(out._apersist_metadata())
        out.update(td)
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
        tensorclass_cls: type | str | None = None,
        **kwargs,
    ) -> TensorDictStore:
        """Connect to an existing TensorDictStore on a remote server.

        This is the cross-node entry point: one process stores data with
        :meth:`from_tensordict` (or regular ``__setitem__``), and another
        process on any machine that can reach the same server reconstructs
        the handle by passing the same ``td_id``.

        Batch size and device are read from the metadata already persisted
        in the store by the original writer.

        If the original writer was a TensorClass instance, the class path
        is stored automatically. On retrieval, the TensorClass is
        reconstructed by importing the class. You can also pass
        ``tensorclass_cls`` explicitly to override or skip the auto-import.

        Keyword Args:
            backend (str): Store backend (``"redis"``, ``"dragonfly"``, etc.).
            host (str): Server hostname. Defaults to ``"localhost"``.
            port (int): Server port. Defaults to ``6379``.
            db (int): Database number. Defaults to ``0``.
            unix_socket_path (str, optional): Unix socket path.
            prefix (str): Key namespace. Defaults to ``"tensordict"``.
            td_id (str): The unique identifier of the TensorDict to reconnect
                to. Obtain this from a previously created instance via
                ``td._td_id``.
            device (torch.device, optional): Device override for retrieved
                tensors. If ``None``, uses the device stored in the server.
            tensorclass_cls (type | str | None): TensorClass to wrap the
                result with.  If a class, used directly via
                ``cls._from_tensordict``.  If a string, interpreted as a
                fully-qualified class path (``"module.ClassName"``) and
                imported.  If ``None`` (default), the class path stored in
                the server (if any) is used automatically.
            **kwargs: Extra connection kwargs.

        Returns:
            A TensorDictStore (or TensorClass wrapping one) connected to the
            existing data.

        Examples:
            On node A (writer)::

                td = TensorDictStore(host="my-server", batch_size=[100])
                td["obs"] = torch.randn(100, 84)
                print(td._td_id)  # e.g. "a1b2c3d4-..."

            On node B (reader)::

                td = TensorDictStore.from_store(
                    host="my-server",
                    td_id="a1b2c3d4-...",
                )
                td["obs"]  # fetched from the shared store
        """
        import redis.asyncio as aioredis

        connect_kwargs = dict(kwargs)
        if unix_socket_path is not None:
            connect_kwargs["unix_socket_path"] = unix_socket_path
        else:
            connect_kwargs["host"] = host
            connect_kwargs["port"] = port
        connect_kwargs["db"] = db

        # Temporarily create an async client to read stored metadata
        loop = asyncio.new_event_loop()
        client = aioredis.Redis(**connect_kwargs)

        async def _read_meta():
            batch_size_key = f"{prefix}:{{{td_id}}}:__batch_size__"
            device_key = f"{prefix}:{{{td_id}}}:__device__"
            tc_key = f"{prefix}:{{{td_id}}}:__tensorclass__"
            pipe = client.pipeline()
            pipe.get(batch_size_key)
            pipe.get(device_key)
            pipe.get(tc_key)
            raw_bs, raw_dev, raw_tc = await pipe.execute()
            await client.aclose()
            return raw_bs, raw_dev, raw_tc

        raw_bs, raw_dev, raw_tc = loop.run_until_complete(_read_meta())
        loop.close()

        if raw_bs is None:
            raise KeyError(
                f"No TensorDictStore with td_id={td_id!r} found at "
                f"{host}:{port} db={db} (prefix={prefix!r})."
            )

        batch_size = torch.Size(json.loads(raw_bs))
        if device is None:
            dev_str = raw_dev.decode() if isinstance(raw_dev, bytes) else raw_dev
            device = torch.device(dev_str) if dev_str else None

        store = cls(
            backend=backend,
            host=host,
            port=port,
            db=db,
            unix_socket_path=unix_socket_path,
            prefix=prefix,
            batch_size=batch_size,
            device=device,
            td_id=td_id,
            **kwargs,
        )

        # Resolve TensorClass wrapping
        tc_type = _resolve_tensorclass(tensorclass_cls, raw_tc)
        if tc_type is not None:
            store._tensorclass_cls = f"{tc_type.__module__}.{tc_type.__qualname__}"
            return tc_type._from_tensordict(store)
        return store

    from_dict_instance = TensorDict.from_dict_instance

    # ---- Cloning ----

    def _clone(self, recurse: bool = True) -> TensorDictStore:
        if recurse:
            # Deep clone: new UUID, copies all data
            new_td = TensorDictStore(
                backend=self._backend,
                host=self._host,
                port=self._port,
                db=self._db,
                unix_socket_path=self._unix_socket_path,
                prefix=self._namespace,
                batch_size=self._batch_size,
                device=self._device,
            )
            new_td.update(self.to_tensordict())
            return new_td
        else:
            # Shallow clone: same Redis data, new Python wrapper
            return TensorDictStore._new_nested(
                parent=self,
                key_prefix=self._prefix,
            )

    # ---- Misc required overrides ----

    def is_contiguous(self) -> bool:
        return False

    def detach_(self) -> Self:
        return self

    @lock_blocked
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
        self._run_sync(self._apersist_metadata())

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
        if recurse:
            out = self.empty(
                recurse=False, batch_size=batch_size, device=device, names=names
            )
            for key, val in self.items():
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
        state = {
            "_backend": self._backend,
            "_host": self._host,
            "_port": self._port,
            "_db": self._db,
            "_unix_socket_path": self._unix_socket_path,
            "_namespace": self._namespace,
            "_td_id": self._td_id,
            "_prefix": self._prefix,
            "_batch_size": self._batch_size,
            "_device": self._device,
            "_redis_kwargs": self._redis_kwargs,
            "_is_locked": self._is_locked,
            "_td_dim_names": self._td_dim_names,
            "_cache_metadata": self._cache_metadata,
            "_tensorclass_cls": self._tensorclass_cls,
        }
        return state

    def __setstate__(self, state):
        import redis.asyncio as aioredis

        self._backend = state.get("_backend", "redis")
        self._host = state["_host"]
        self._port = state["_port"]
        self._db = state["_db"]
        self._unix_socket_path = state["_unix_socket_path"]
        self._namespace = state["_namespace"]
        self._td_id = state["_td_id"]
        self._prefix = state["_prefix"]
        self._batch_size = state["_batch_size"]
        self._device = state["_device"]
        self._redis_kwargs = state["_redis_kwargs"]
        self._td_dim_names = state["_td_dim_names"]
        self._tensorclass_cls = state.get("_tensorclass_cls")

        self._locked_tensordicts = []
        self._lock_id = set()
        self._is_shared = False
        self._is_memmap = False
        self._nested_tensordicts = {}
        self._cache_metadata = state.get("_cache_metadata", True)
        self._meta_cache = {} if self._cache_metadata else None
        self._keys_cache = [None]

        # Recreate event loop and client
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
        """Close the Redis connection and stop the background event loop."""
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
            self._owns_loop = False

    def clear_redis(self):
        """Delete all keys associated with this TensorDict from Redis.

        This removes all tensor data, metadata, and the key registry.
        """

        async def _aclear():
            all_keys = await self._aget_all_keys()
            if not all_keys:
                pipe = self._client.pipeline()
                pipe.delete(self._keys_registry_key)
                pipe.delete(self._batch_size_key)
                pipe.delete(self._device_key)
                await pipe.execute()
                return
            pipe = self._client.pipeline()
            for key_path in all_keys:
                pipe.delete(self._data_key(key_path))
                pipe.delete(self._meta_key(key_path))
            pipe.delete(self._keys_registry_key)
            pipe.delete(self._batch_size_key)
            pipe.delete(self._device_key)
            await pipe.execute()

        self._run_sync(_aclear())
        if self._meta_cache is not None:
            self._meta_cache.clear()
        self._keys_cache[0] = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        keys_str = list(self.keys())
        device = self.device
        batch_size = self.batch_size
        return (
            f"TensorDictStore(\n"
            f"    keys={keys_str},\n"
            f"    batch_size={batch_size},\n"
            f"    device={device},\n"
            f"    backend={self._backend!r},\n"
            f"    td_id={self._td_id!r})"
        )

    # ---- Delegated to TensorDict (same pattern as PersistentTensorDict) ----

    __eq__ = TensorDict.__eq__
    __ne__ = TensorDict.__ne__
    __xor__ = TensorDict.__xor__
    __or__ = TensorDict.__or__
    __ge__ = TensorDict.__ge__
    __gt__ = TensorDict.__gt__
    __le__ = TensorDict.__le__
    __lt__ = TensorDict.__lt__

    _apply_nest = TensorDict._apply_nest
    _cast_reduction = TensorDict._cast_reduction
    _check_device = TensorDict._check_device
    _check_is_shared = TensorDict._check_is_shared
    _convert_to_tensordict = TensorDict._convert_to_tensordict
    _get_names_idx = TensorDict._get_names_idx
    _multithread_apply_flat = TensorDict._multithread_apply_flat
    _multithread_rebuild = TensorDict._multithread_rebuild
    _to_module = TensorDict._to_module
    _unbind = TensorDict._unbind
    all = TensorDict.all
    any = TensorDict.any
    expand = TensorDict.expand
    _repeat = TensorDict._repeat
    repeat_interleave = TensorDict.repeat_interleave
    reshape = TensorDict.reshape
    split = TensorDict.split

    # ---- Shape ops: raise NotImplementedError ----

    def _view(self, *args, **kwargs):
        raise RuntimeError(
            f"Cannot call `view` on a {type(self).__name__}. "
            "Call `to_tensordict()` or `to_local()` first."
        )

    def _transpose(self, dim0, dim1):
        raise RuntimeError(
            f"Cannot call `transpose` on a {type(self).__name__}. "
            "Call `to_tensordict()` or `to_local()` first."
        )

    def _permute(self, *args, **kwargs):
        raise RuntimeError(
            f"Cannot call `permute` on a {type(self).__name__}. "
            "Call `to_tensordict()` or `to_local()` first."
        )

    def _squeeze(self, dim=None):
        raise RuntimeError(
            f"Cannot call `squeeze` on a {type(self).__name__}. "
            "Call `to_tensordict()` or `to_local()` first."
        )

    def _unsqueeze(self, dim: int):
        raise RuntimeError(
            f"Cannot call `unsqueeze` on a {type(self).__name__}. "
            "Call `to_tensordict()` or `to_local()` first."
        )

    def chunk(self, chunks: int, dim: int = 0) -> tuple[TensorDictBase, ...]:
        splits = -(self.batch_size[dim] // -chunks)
        return self.split(splits, dim)

    # ---- Memory ops: not supported ----

    def share_memory_(self):
        raise NotImplementedError(
            f"Cannot call share_memory_ on a {type(self).__name__}. "
            "Call `to_tensordict()` or `to_local()` first."
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
        raise RuntimeError(
            f"Cannot call memmap on a {type(self).__name__} in-place. "
            "Call `to_tensordict()` or `to_local()` first."
        )

    def make_memmap(self, key, shape, *, dtype=None, robust_key=None):
        raise RuntimeError(
            f"Cannot make memory-mapped tensor on a {type(self).__name__}."
        )

    def make_memmap_from_storage(
        self, key, storage, shape, *, dtype=None, robust_key=None
    ):
        raise RuntimeError(
            f"Cannot make memory-mapped tensor on a {type(self).__name__}."
        )

    def make_memmap_from_tensor(self, key, tensor, *, copy_data=True, robust_key=None):
        raise RuntimeError(
            f"Cannot make memory-mapped tensor on a {type(self).__name__}."
        )

    def memmap_(self, prefix=None, copy_existing=False, num_threads=0):
        raise RuntimeError(
            f"Cannot build a memmap TensorDict in-place from a {type(self).__name__}. "
            "Use `td.memmap()` instead."
        )

    def pin_memory(self, *args, **kwargs):
        raise RuntimeError(
            f"Cannot pin memory of a {type(self).__name__}. "
            "Call `to_tensordict()` or `to_local()` before making this call."
        )

    def _add_batch_dim(self, *, in_dim, vmap_level):
        raise RuntimeError(f"{type(self).__name__} cannot be used with vmap.")

    def _remove_batch_dim(self, vmap_level, batch_size, out_dim): ...

    def _maybe_remove_batch_dim(self, funcname, vmap_level, batch_size, out_dim): ...

    def _select(self, *keys, inplace=False, strict=True, set_shared=True):
        raise NotImplementedError(
            f"Cannot call select on a {type(self).__name__}. "
            "Call `to_tensordict()` or `to_local()` first."
        )

    def _exclude(self, *keys, inplace=False, set_shared=True):
        raise NotImplementedError(
            f"Cannot call exclude on a {type(self).__name__}. "
            "Call `to_tensordict()` or `to_local()` first."
        )

    @_as_context_manager()
    def flatten_keys(self, separator=".", inplace=False):
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
            f"set_non_tensor is not compatible with the tensordict type {type(self).__name__}."
        )

    def _stack_onto_(self, list_item, dim):
        raise RuntimeError(
            f"Cannot call _stack_onto_ on a {type(self).__name__}. "
            "Call `to_tensordict()` or `to_local()` first."
        )


_register_tensor_class(TensorDictStore)


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
        self._element_idx = element_idx % parent._count

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

    _erase_names = TensorDict._erase_names
    _has_names = TensorDict._has_names
    _set_names = TensorDict._set_names
    names = TensorDict.names

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

    _get_tuple = TensorDict._get_tuple

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
            value = self._validate_value(value, check_shape=True)
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
                f"{type(self).__name__} only supports tensor values, "
                f"got {type(value)}"
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
            value = self._validate_value(value, check_shape=True)
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

        if not isinstance(value, TensorDictBase):
            value = TensorDict.from_dict(value, batch_size=[])

        for key in value.keys(include_nested=True, leaves_only=True):
            key_tuple = _unravel_key_to_tuple(key)
            key_path = _KEY_SEP.join(key_tuple)
            existing = (
                self._get_str(key_tuple[0]) if len(key_tuple) == 1 else self.get(key)
            )
            existing[index] = value.get(key)
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

    @lock_blocked
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

    @erase_cache
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

    def to_local(self) -> TensorDict:
        return self.to_tensordict()

    def contiguous(self) -> TensorDict:
        return self.to_tensordict()

    def is_contiguous(self) -> bool:
        return False

    def detach_(self) -> Self:
        return self

    @lock_blocked
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

    from_dict_instance = TensorDict.from_dict_instance

    __eq__ = TensorDict.__eq__
    __ne__ = TensorDict.__ne__
    __xor__ = TensorDict.__xor__
    __or__ = TensorDict.__or__
    __ge__ = TensorDict.__ge__
    __gt__ = TensorDict.__gt__
    __le__ = TensorDict.__le__
    __lt__ = TensorDict.__lt__

    _apply_nest = TensorDict._apply_nest
    _cast_reduction = TensorDict._cast_reduction
    _check_device = TensorDict._check_device
    _check_is_shared = TensorDict._check_is_shared
    _convert_to_tensordict = TensorDict._convert_to_tensordict
    _get_names_idx = TensorDict._get_names_idx
    _multithread_apply_flat = TensorDict._multithread_apply_flat
    _multithread_rebuild = TensorDict._multithread_rebuild
    _to_module = TensorDict._to_module
    _unbind = TensorDict._unbind
    all = TensorDict.all
    any = TensorDict.any
    expand = TensorDict.expand
    _repeat = TensorDict._repeat
    repeat_interleave = TensorDict.repeat_interleave
    reshape = TensorDict.reshape
    split = TensorDict.split

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

    def make_memmap(self, key, shape, *, dtype=None, robust_key=None):
        raise RuntimeError(f"Cannot make memmap on a {type(self).__name__}.")

    def make_memmap_from_storage(
        self, key, storage, shape, *, dtype=None, robust_key=None
    ):
        raise RuntimeError(f"Cannot make memmap on a {type(self).__name__}.")

    def make_memmap_from_tensor(self, key, tensor, *, copy_data=True, robust_key=None):
        raise RuntimeError(f"Cannot make memmap on a {type(self).__name__}.")

    def memmap_(self, prefix=None, copy_existing=False, num_threads=0):
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

    async def _aget_key_meta(self, key_path: str) -> dict[str, str]:
        raw = await self._client.hgetall(self._meta_key(key_path))
        return _decode_meta(raw)

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

    async def _aget_element_tensor(
        self, key_path: str, element_idx: int
    ) -> torch.Tensor:
        """Fetch a single stack element's tensor for one key."""
        meta = await self._aget_key_meta(key_path)
        dtype = _str_to_dtype(meta["dtype"])
        homogeneous = self._is_key_homogeneous(meta)

        if homogeneous:
            full_shape = json.loads(meta["shape"])
            elem_shape = full_shape[1:]
            row_bytes = self._row_bytes(elem_shape, dtype)
            pos = element_idx % self._count
            offset = pos * row_bytes
            data = await self._client.getrange(
                self._data_key(key_path), offset, offset + row_bytes - 1
            )
            tensor = _bytes_to_tensor(data, elem_shape, dtype)
        else:
            # Read offsets
            pos = element_idx % self._count
            off_data = await self._client.getrange(
                self._idx_key(key_path), pos * 8, (pos + 2) * 8 - 1
            )
            start, end = struct.unpack("<2q", off_data)
            data = await self._client.getrange(self._data_key(key_path), start, end - 1)
            shapes = json.loads(meta["shapes"])
            elem_shape = shapes[pos]
            tensor = _bytes_to_tensor(data, elem_shape, dtype)

        if self._device is not None:
            tensor = tensor.to(self._device)
        return tensor

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

        pos = element_idx % self._count

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

        pos = element_idx % self._count

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
        pos = element_idx % self._count
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

        value = value.contiguous().cpu()
        raw_bytes = _tensor_to_bytes(value)

        if homogeneous:
            full_shape = json.loads(raw_meta["shape"])
            elem_shape = full_shape[1:]
            row_bytes = self._row_bytes(elem_shape, dtype)
            if len(raw_bytes) != row_bytes:
                raise ValueError(
                    f"Shape mismatch for homogeneous key {key_path!r}: "
                    f"expected {row_bytes} bytes (shape {elem_shape}), "
                    f"got {len(raw_bytes)} bytes (shape {list(value.shape)}). "
                    f"To change the shape of a homogeneous key, reassign the "
                    f"full stacked tensor via the parent."
                )
            offset = pos * row_bytes
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

        scattered = _is_scattered_index(idx)

        for kp in key_paths:
            full_shape, dtype = meta_map[kp]
            result_shape = _getitem_result_shape(full_shape, idx)
            local_idx = _get_local_idx(idx, full_shape[0])

            if scattered:
                ranges = _compute_byte_ranges(full_shape, dtype, idx)
                if ranges is None or not ranges:
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
                    plan.append((kp, result_shape, dtype, None, False))
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

        has_cmds = any(has_cmd for _, _, _, _, has_cmd in plan)
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

        return result

    # ---- element access (writes) ----

    async def _aset_element(self, element_idx: int, value_td: TensorDictBase):
        """Write all keys for a single stack element via pipelined SETRANGE."""
        all_keys = sorted(await self._aget_all_keys())
        pos = element_idx % self._count

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
                    tensor[idx] = value
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

    _erase_names = TensorDict._erase_names
    _has_names = TensorDict._has_names
    _set_names = TensorDict._set_names
    names = TensorDict.names

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
        if isinstance(index, int) and self._stack_dim == 0:
            return _StoreStackElementView(self, index)

        # General indexing via _index_tensordict
        return self._index_tensordict(index)

    def __setitem__(self, index, value):
        index_unravel = _unravel_key_to_tuple(index)
        if index_unravel:
            return self.set(index_unravel, value, inplace=True)

        if isinstance(index, list):
            index = torch.tensor(index)

        # Integer assignment on stack dim: write element
        if isinstance(index, int) and self._stack_dim == 0:
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

    _get_tuple = TensorDict._get_tuple

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
            value = self._validate_value(value, check_shape=True)
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
            value = self._validate_value(value, check_shape=True)
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

    @lock_blocked
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

    @erase_cache
    def _propagate_unlock(self):
        self._is_locked = False
        self._is_shared = False
        self._is_memmap = False
        return []

    # ---- Materialization ----

    def to_local(self) -> TensorDict:
        return self.to_tensordict()

    def contiguous(self) -> TensorDict:
        return self.to_tensordict()

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

    from_dict_instance = TensorDict.from_dict_instance

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

    @lock_blocked
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

    __eq__ = TensorDict.__eq__
    __ne__ = TensorDict.__ne__
    __xor__ = TensorDict.__xor__
    __or__ = TensorDict.__or__
    __ge__ = TensorDict.__ge__
    __gt__ = TensorDict.__gt__
    __le__ = TensorDict.__le__
    __lt__ = TensorDict.__lt__

    _apply_nest = TensorDict._apply_nest
    _cast_reduction = TensorDict._cast_reduction
    _check_device = TensorDict._check_device
    _check_is_shared = TensorDict._check_is_shared
    _convert_to_tensordict = TensorDict._convert_to_tensordict
    _get_names_idx = TensorDict._get_names_idx
    _multithread_apply_flat = TensorDict._multithread_apply_flat
    _multithread_rebuild = TensorDict._multithread_rebuild
    _to_module = TensorDict._to_module
    _unbind = TensorDict._unbind
    all = TensorDict.all
    any = TensorDict.any
    expand = TensorDict.expand
    _repeat = TensorDict._repeat
    repeat_interleave = TensorDict.repeat_interleave
    reshape = TensorDict.reshape
    split = TensorDict.split

    # ---- Shape ops: not supported ----

    def _view(self, *args, **kwargs):
        raise RuntimeError(
            f"Cannot call `view` on a {type(self).__name__}. "
            "Call `to_tensordict()` or `to_local()` first."
        )

    def _transpose(self, dim0, dim1):
        raise RuntimeError(
            f"Cannot call `transpose` on a {type(self).__name__}. "
            "Call `to_tensordict()` or `to_local()` first."
        )

    def _permute(self, *args, **kwargs):
        raise RuntimeError(
            f"Cannot call `permute` on a {type(self).__name__}. "
            "Call `to_tensordict()` or `to_local()` first."
        )

    def _squeeze(self, dim=None):
        raise RuntimeError(
            f"Cannot call `squeeze` on a {type(self).__name__}. "
            "Call `to_tensordict()` or `to_local()` first."
        )

    def _unsqueeze(self, dim: int):
        raise RuntimeError(
            f"Cannot call `unsqueeze` on a {type(self).__name__}. "
            "Call `to_tensordict()` or `to_local()` first."
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

    def make_memmap(self, key, shape, *, dtype=None, robust_key=None):
        raise RuntimeError(
            f"Cannot make memory-mapped tensor on a {type(self).__name__}."
        )

    def make_memmap_from_storage(
        self, key, storage, shape, *, dtype=None, robust_key=None
    ):
        raise RuntimeError(
            f"Cannot make memory-mapped tensor on a {type(self).__name__}."
        )

    def make_memmap_from_tensor(self, key, tensor, *, copy_data=True, robust_key=None):
        raise RuntimeError(
            f"Cannot make memory-mapped tensor on a {type(self).__name__}."
        )

    def memmap_(self, prefix=None, copy_existing=False, num_threads=0):
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

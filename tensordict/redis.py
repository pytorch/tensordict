# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Redis-backed TensorDict implementation for out-of-core tensor storage."""

from __future__ import annotations

import asyncio
import importlib
import json
import pickle
import threading
import uuid
import weakref
from typing import Any, Callable, Tuple, Type, TYPE_CHECKING

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
    _KEY_ERROR,
    _LOCK_ERROR,
    erase_cache,
    is_non_tensor,
    lock_blocked,
    NestedKey,
    unravel_key,
)

_has_redis = importlib.util.find_spec("redis", None) is not None

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any

# Separator used for nested key paths in Redis keys
_KEY_SEP = "."


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


def _compute_byte_ranges(
    shape: list[int],
    dtype: torch.dtype,
    idx,
) -> list[tuple[int, int]] | None:
    """Compute ``(byte_offset, byte_length)`` pairs for indexing ``shape[0]`` with *idx*.

    Returns ``None`` when *idx* is an unsupported type, signalling that the
    caller should fall back to a full read-modify-write.
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
        # Contiguous with step 1: return a single range
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


def _compute_covering_range(
    shape: list[int],
    dtype: torch.dtype,
    idx,
) -> tuple[int, int, object] | None:
    """Compute a single covering byte range for indexed reads.

    Returns ``(byte_offset, byte_length, local_idx)`` or ``None``.
    - **byte_offset / byte_length**: a single contiguous GETRANGE span.
    - **local_idx**: ``None`` when the fetched bytes are already the result
      (int index, step-1 slice), or an index to apply on the covering tensor's
      first dimension to extract the requested rows.

    This always emits at most **one** GETRANGE per key regardless of index type.
    """
    # Unwrap 1-element tuples
    if isinstance(idx, tuple):
        if len(idx) == 1:
            idx = idx[0]
        else:
            return None

    if idx is Ellipsis:
        idx = slice(None)

    elem_size = torch.tensor([], dtype=dtype).element_size()
    row_size = elem_size
    for s in shape[1:]:
        row_size *= s

    if isinstance(idx, int):
        pos = idx % shape[0]
        return (pos * row_size, row_size, None)

    if isinstance(idx, (slice, range)):
        positions = range(*idx.indices(shape[0])) if isinstance(idx, slice) else idx
        if len(positions) == 0:
            return (0, 0, None)
        start = positions[0]
        stop = positions[-1] + 1  # make exclusive
        covering_rows = stop - start
        if positions.step == 1:
            return (start * row_size, covering_rows * row_size, None)
        # Step > 1: fetch covering range, stride locally
        local_idx = slice(None, None, positions.step)
        return (start * row_size, covering_rows * row_size, local_idx)

    if isinstance(idx, list):
        idx = torch.tensor(idx)

    if isinstance(idx, torch.Tensor):
        if idx.dtype == torch.bool:
            positions = idx.nonzero(as_tuple=False).squeeze(-1)
        else:
            positions = idx.reshape(-1)
        if positions.numel() == 0:
            return (0, 0, None)
        min_pos = int(positions.min().item())
        max_pos = int(positions.max().item())
        covering_rows = max_pos - min_pos + 1
        # Shift indices relative to covering range start
        local_idx = positions - min_pos
        return (min_pos * row_size, covering_rows * row_size, local_idx)

    return None


class _RedisTDKeysView(_TensorDictKeysView):
    """Keys view for RedisTensorDict backed by a Redis key registry."""

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


class RedisTensorDict(TensorDictBase):
    """A TensorDict backed by a Redis instance for out-of-core storage.

    Tensors are stored as raw bytes in Redis using ``torch.frombuffer``-compatible
    serialization. Metadata (shape, dtype) is stored in Redis Hashes for fast
    introspection without downloading tensor data.

    All Redis I/O is performed via an async ``redis.asyncio.Redis`` client
    running on a dedicated background event loop, ensuring non-blocking
    operation from the caller's perspective.

    Keyword Args:
        host (str): Redis server hostname. Defaults to ``"localhost"``.
        port (int): Redis server port. Defaults to ``6379``.
        db (int): Redis database number. Defaults to ``0``.
        unix_socket_path (str, optional): Path to a Unix domain socket for
            local high-performance connections. Mutually exclusive with
            ``host``/``port``.
        prefix (str, optional): A namespace prefix for all Redis keys.
            Defaults to ``"tensordict"``.
        batch_size (torch.Size or compatible): The TensorDict batch size.
            Defaults to ``torch.Size(())``.
        device (torch.device or compatible, optional): Target device for
            retrieved tensors. Defaults to ``None`` (CPU).
        client: An existing ``redis.asyncio.Redis`` client instance. If
            provided, ``host``/``port``/``db``/``unix_socket_path`` are ignored.
        td_id (str, optional): A unique identifier for this TensorDict in Redis.
            Defaults to a new UUID. Pass an existing ID to reconnect to
            previously stored data.
        **redis_kwargs: Additional keyword arguments passed to
            ``redis.asyncio.Redis``.

    Examples:
        >>> from tensordict.redis import RedisTensorDict
        >>> td = RedisTensorDict(batch_size=[100])
        >>> td["obs"] = torch.randn(100, 84)
        >>> td["obs"]  # fetched from Redis
        >>> local_td = td.to_local()  # materialize everything into RAM

    .. note::
        Requires ``redis`` package: ``pip install redis``.

    .. note::
        Tensors are stored as CPU contiguous bytes. The ``device`` attribute
        controls the device tensors are cast to upon retrieval.
    """

    _td_dim_names = None

    def __init__(
        self,
        *,
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

        # Nested TensorDict cache
        self._nested_tensordicts: dict[str, RedisTensorDict] = {}

        # Metadata cache: (shape, dtype) per key_path, shared with nested views
        self._cache_metadata = cache_metadata
        self._meta_cache: dict[str, tuple[list[int], torch.dtype]] | None = (
            {} if cache_metadata else None
        )

        # Unique identifier for this TensorDict instance in Redis
        self._td_id = td_id or str(uuid.uuid4())

        # Key prefix within this tensordict (for nested views)
        self._prefix = ""

        # Namespace prefix for Redis keys
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

        # Persist batch_size and device to Redis
        self._run_sync(self._apersist_metadata())

    @classmethod
    def _new_nested(cls, *, parent: RedisTensorDict, key_prefix: str, batch_size=None):
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

        obj._td_id = parent._td_id
        obj._prefix = key_prefix
        obj._namespace = parent._namespace
        obj._cache_metadata = parent._cache_metadata
        obj._meta_cache = parent._meta_cache

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

    def _full_key_path(self, key: str) -> str:
        """Build the full dot-separated key path including prefix."""
        if self._prefix:
            return self._prefix + _KEY_SEP + key
        return key

    # ---- Async internal methods ----

    async def _apersist_metadata(self):
        """Persist batch_size and device to Redis."""
        pipe = self._client.pipeline()
        pipe.set(self._batch_size_key, json.dumps(list(self._batch_size)))
        device_str = str(self._device) if self._device is not None else ""
        pipe.set(self._device_key, device_str)
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

    async def _aget_tensor(self, key_path: str) -> torch.Tensor | None:
        """Retrieve a tensor from Redis. Returns None if not found."""
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
        """Batch-fetch multiple tensors using a Redis pipeline."""
        if not key_paths:
            return {}

        # Batch get data
        pipe = self._client.pipeline()
        for kp in key_paths:
            pipe.get(self._data_key(kp))
        data_list = await pipe.execute()

        # Batch get metadata
        pipe = self._client.pipeline()
        for kp in key_paths:
            pipe.hgetall(self._meta_key(kp))
        meta_list = await pipe.execute()

        result = {}
        for kp, data, raw_meta in zip(key_paths, data_list, meta_list):
            if data is None:
                continue
            meta = _decode_meta(raw_meta)
            if meta.get("is_non_tensor") == "1":
                result[kp] = self._deserialize_non_tensor(data, meta)
            else:
                shape = json.loads(meta["shape"])
                dtype = _str_to_dtype(meta["dtype"])
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

        Uses :func:`_compute_covering_range` so that every key emits **at most
        one** ``GETRANGE`` command, regardless of whether the index is an int,
        slice-with-step, tensor, or boolean mask.  A local post-index is
        applied when the covering range is larger than the result.

        Falls back to full ``GET`` + local indexing for unsupported index types.
        """
        if not key_paths:
            return {}

        meta_map = await self._aget_metadata_batch(key_paths)

        pipe = self._client.pipeline()
        # (key_path, covering_shape, dtype, local_idx, has_data)
        plan: list[tuple[str, list[int], torch.dtype, object, bool]] = []
        fallback_kps: list[str] = []

        for kp in key_paths:
            shape, dtype = meta_map[kp]
            cr = _compute_covering_range(shape, dtype, idx)
            if cr is None:
                fallback_kps.append(kp)
                continue
            byte_offset, byte_length, local_idx = cr
            elem_size = torch.tensor([], dtype=dtype).element_size()
            rest = shape[1:]
            row_bytes = elem_size * (
                int(torch.tensor(rest).prod().item()) if rest else 1
            )
            covering_rows = byte_length // row_bytes if row_bytes > 0 else 0
            covering_shape = [covering_rows] + rest
            has_data = byte_length > 0
            if has_data:
                pipe.getrange(
                    self._data_key(kp),
                    byte_offset,
                    byte_offset + byte_length - 1,
                )
            plan.append((kp, covering_shape, dtype, local_idx, has_data))

        raw_results = await pipe.execute() if any(p[4] for p in plan) else []

        result: dict[str, torch.Tensor] = {}
        ri = 0
        for kp, covering_shape, dtype, local_idx, has_data in plan:
            result_shape = _getitem_result_shape(meta_map[kp][0], idx)
            if not has_data:
                result[kp] = torch.empty(result_shape, dtype=dtype)
                continue
            data = raw_results[ri]
            ri += 1
            tensor = _bytes_to_tensor(data, covering_shape, dtype)
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

        # -- Stage 2: build single-GETRANGE pipeline -------------------------
        pipe = self._client.pipeline()

        # (kp, covering_shape, dtype, local_idx, has_data)
        plan: list[tuple[str, list[int], torch.dtype, object, bool]] = []
        fallback_tensor_kps: list[str] = []

        for kp in tensor_kps:
            shape, dtype = all_meta[kp]
            cr = _compute_covering_range(shape, dtype, idx)
            if cr is None:
                fallback_tensor_kps.append(kp)
                continue
            byte_offset, byte_length, local_idx = cr
            elem_size = torch.tensor([], dtype=dtype).element_size()
            rest = shape[1:]
            row_bytes = elem_size * (
                int(torch.tensor(rest).prod().item()) if rest else 1
            )
            covering_rows = byte_length // row_bytes if row_bytes > 0 else 0
            covering_shape = [covering_rows] + rest
            has_data = byte_length > 0
            if has_data:
                pipe.getrange(
                    self._data_key(kp),
                    byte_offset,
                    byte_offset + byte_length - 1,
                )
            plan.append((kp, covering_shape, dtype, local_idx, has_data))

        # Full GET for fallback tensors
        for kp in fallback_tensor_kps:
            pipe.get(self._data_key(kp))
            pipe.hgetall(self._meta_key(kp))

        # Full GET for non-tensor keys
        for kp in non_tensor_kps:
            pipe.get(self._data_key(kp))

        all_results = await pipe.execute()

        # -- Stage 3: reassemble results -----------------------------------
        result: dict[str, torch.Tensor | Any] = {}

        flat_idx = 0
        for kp, covering_shape, dtype, local_idx, has_data in plan:
            result_shape = _getitem_result_shape(all_meta[kp][0], idx)
            if not has_data:
                result[kp] = torch.empty(result_shape, dtype=dtype)
                continue
            data = all_results[flat_idx]
            flat_idx += 1
            tensor = _bytes_to_tensor(data, covering_shape, dtype)
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

        # Non-tensor results (not indexable, returned as-is)
        for kp in non_tensor_kps:
            data = all_results[flat_idx]
            flat_idx += 1
            result[kp] = self._deserialize_non_tensor(data, raw_metas[kp])

        return result

    async def _abatch_set_at(self, items: dict[str, tuple[torch.Tensor, object]]):
        """Batch-write slices of multiple tensors.

        *items* maps ``key_path`` to ``(value_tensor, idx)``.

        Three strategies, chosen per-key:

        1. **Direct SETRANGE** (int, step-1 slice): value bytes are written in
           a single ``SETRANGE`` without reading first.
        2. **Partial read-modify-write** (step>1, tensor, bool mask): fetch the
           *covering range* via ``GETRANGE``, patch in memory, write back with
           ``SETRANGE``.  2 commands per key across 2 pipelines.
        3. **Full read-modify-write** fallback for unsupported indices.
        """
        if not items:
            return

        key_paths = list(items.keys())
        meta_map = await self._aget_metadata_batch(key_paths)

        # Classify each key-path into one of three strategies.
        # Store (kp, byte_offset, byte_length, local_idx) for covering-range keys.
        direct_kps: list[tuple[str, int]] = []  # (kp, byte_offset)
        partial_kps: list[tuple[str, int, int, object, list[int], torch.dtype]] = []
        fallback_kps: list[str] = []

        for kp in key_paths:
            shape, dtype = meta_map[kp]
            _, idx = items[kp]
            cr = _compute_covering_range(shape, dtype, idx)
            if cr is None:
                fallback_kps.append(kp)
                continue
            byte_offset, byte_length, local_idx = cr
            if local_idx is None:
                direct_kps.append((kp, byte_offset))
            else:
                partial_kps.append(
                    (kp, byte_offset, byte_length, local_idx, shape, dtype)
                )

        # --- Strategy 1: direct SETRANGE (single pipeline) -----------------
        if direct_kps:
            pipe = self._client.pipeline()
            for kp, byte_offset in direct_kps:
                value, _ = items[kp]
                pipe.setrange(
                    self._data_key(kp),
                    byte_offset,
                    _tensor_to_bytes(value.contiguous()),
                )
            await pipe.execute()

        # --- Strategy 2: partial covering-range RMW (two pipelines) --------
        if partial_kps:
            # Pipeline 1: GETRANGE covering ranges
            pipe = self._client.pipeline()
            for kp, byte_offset, byte_length, _, _, _ in partial_kps:
                pipe.getrange(
                    self._data_key(kp),
                    byte_offset,
                    byte_offset + byte_length - 1,
                )
            raw_covers = await pipe.execute()

            # Patch in memory and pipeline 2: SETRANGE
            pipe = self._client.pipeline()
            for (kp, byte_offset, byte_length, local_idx, shape, dtype), data in zip(
                partial_kps, raw_covers
            ):
                rest = shape[1:]
                elem_size = torch.tensor([], dtype=dtype).element_size()
                row_bytes = elem_size * (
                    int(torch.tensor(rest).prod().item()) if rest else 1
                )
                covering_rows = byte_length // row_bytes if row_bytes > 0 else 0
                covering_shape = [covering_rows] + rest
                covering_tensor = _bytes_to_tensor(data, covering_shape, dtype)
                value, _ = items[kp]
                covering_tensor[local_idx] = value
                pipe.setrange(
                    self._data_key(kp),
                    byte_offset,
                    _tensor_to_bytes(covering_tensor.contiguous()),
                )
            await pipe.execute()

        # --- Strategy 3: full RMW fallback ---------------------------------
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
        """Get all registered key paths (sync wrapper)."""
        return self._run_sync(self._aget_all_keys())

    @staticmethod
    def _deserialize_non_tensor(data: bytes, meta: dict) -> Any:
        """Deserialize a non-tensor value from Redis."""
        encoding = meta.get("encoding", "json")
        if encoding == "json":
            return json.loads(data.decode() if isinstance(data, bytes) else data)
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

        # Collect all leaf items
        items: dict[str, tuple[torch.Tensor, object]] = {}
        for key in value.keys(include_nested=True, leaves_only=True):
            key_tuple = _unravel_key_to_tuple(key)
            key_path = self._full_key_path(_KEY_SEP.join(key_tuple))
            items[key_path] = (value.get(key), index)

        self._run_sync(self._abatch_set_at(items))

    def _get_str(self, key, default=NO_DEFAULT, **kwargs):
        key_path = self._full_key_path(key)
        all_keys = self._get_all_keys()

        # Check if it's a nested tensordict (key is a prefix of other keys)
        prefix_check = key_path + _KEY_SEP
        nested_keys = [k for k in all_keys if k.startswith(prefix_check)]
        if nested_keys:
            # Return a nested RedisTensorDict view
            nested = self._nested_tensordicts.get(key)
            if nested is None:
                nested = RedisTensorDict._new_nested(
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

    _get_tuple = TensorDict._get_tuple

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
                raw_value = value
            self._run_sync(self._aset_non_tensor(key_path, raw_value))
            return self

        if is_tensor_collection(value):
            # Create nested tensordict and populate it
            target_td = self._nested_tensordicts.get(key)
            if target_td is None:
                target_td = RedisTensorDict._new_nested(
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

            raw_value = value.data if isinstance(value, NonTensorData) else value
            self._run_sync(self._aset_non_tensor(key_path, raw_value))
        elif is_tensor_collection(value):
            nested_prefix = self._full_key_path(_KEY_SEP.join(key))
            nested = RedisTensorDict._new_nested(
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
    ) -> _RedisTDKeysView:
        return _RedisTDKeysView(
            tensordict=self,
            include_nested=include_nested,
            leaves_only=leaves_only,
            is_leaf=is_leaf,
            sort=sort,
        )

    @lock_blocked
    def del_(self, key: NestedKey) -> RedisTensorDict:
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
    ) -> RedisTensorDict:
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
            return RedisTensorDict

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
        """Create a RedisTensorDict from a dictionary or TensorDict.

        Args:
            input_dict: A dictionary or TensorDict to store in Redis.

        Keyword Args:
            host, port, db, unix_socket_path, prefix: Redis connection params.
            auto_batch_size: If True, infer batch_size from data.
            batch_size: Explicit batch_size.
            device: Target device for tensor retrieval.
            **kwargs: Additional Redis connection kwargs.

        Returns:
            A new RedisTensorDict.
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
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        unix_socket_path: str | None = None,
        prefix: str = "tensordict",
        device=None,
        **kwargs,
    ) -> RedisTensorDict:
        """Upload a TensorDict to Redis.

        Creates a new :class:`RedisTensorDict` and copies all tensor data from
        the provided TensorDict into Redis.

        Args:
            td (TensorDictBase): The source TensorDict whose data will be
                stored in Redis.

        Keyword Args:
            host (str): Redis hostname. Defaults to ``"localhost"``.
            port (int): Redis port. Defaults to ``6379``.
            db (int): Redis database. Defaults to ``0``.
            unix_socket_path (str, optional): Unix socket path.
            prefix (str): Redis key namespace. Defaults to ``"tensordict"``.
            device (torch.device, optional): Device override for retrieved
                tensors. If ``None``, uses the source TensorDict's device.
            **kwargs: Extra Redis connection kwargs.

        Returns:
            A new RedisTensorDict backed by the uploaded data.

        Examples:
            >>> local = TensorDict({"obs": torch.randn(10, 84)}, [10])
            >>> remote = RedisTensorDict.from_tensordict(local, host="my-redis")
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

        out = cls(
            batch_size=td.batch_size,
            device=device,
            prefix=prefix,
            **connect_kwargs,
            **kwargs,
        )
        out.update(td)
        return out

    @classmethod
    def from_redis(
        cls,
        *,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        unix_socket_path: str | None = None,
        prefix: str = "tensordict",
        td_id: str,
        device=None,
        **kwargs,
    ) -> RedisTensorDict:
        """Connect to an existing RedisTensorDict stored on a Redis server.

        This is the cross-node entry point: one process stores data with
        :meth:`from_tensordict` (or regular ``__setitem__``), and another
        process on any machine that can reach the same Redis server
        reconstructs the handle by passing the same ``td_id``.

        Batch size and device are read from the metadata already persisted
        in Redis by the original writer.

        Keyword Args:
            host (str): Redis hostname. Defaults to ``"localhost"``.
            port (int): Redis port. Defaults to ``6379``.
            db (int): Redis database. Defaults to ``0``.
            unix_socket_path (str, optional): Unix socket path.
            prefix (str): Redis key namespace. Defaults to ``"tensordict"``.
            td_id (str): The unique identifier of the TensorDict to reconnect
                to. Obtain this from a previously created instance via
                ``td._td_id``.
            device (torch.device, optional): Device override for retrieved
                tensors. If ``None``, uses the device stored in Redis.
            **kwargs: Extra Redis connection kwargs.

        Returns:
            A RedisTensorDict connected to the existing data.

        Examples:
            On node A (writer)::

                td = RedisTensorDict(host="shared-redis", batch_size=[100])
                td["obs"] = torch.randn(100, 84)
                print(td._td_id)  # e.g. "a1b2c3d4-..."

            On node B (reader)::

                td = RedisTensorDict.from_redis(
                    host="shared-redis",
                    td_id="a1b2c3d4-...",
                )
                td["obs"]  # fetched from the shared Redis
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
            pipe = client.pipeline()
            pipe.get(batch_size_key)
            pipe.get(device_key)
            raw_bs, raw_dev = await pipe.execute()
            await client.aclose()
            return raw_bs, raw_dev

        raw_bs, raw_dev = loop.run_until_complete(_read_meta())
        loop.close()

        if raw_bs is None:
            raise KeyError(
                f"No RedisTensorDict with td_id={td_id!r} found at "
                f"{host}:{port} db={db} (prefix={prefix!r})."
            )

        batch_size = torch.Size(json.loads(raw_bs))
        if device is None:
            dev_str = raw_dev.decode() if isinstance(raw_dev, bytes) else raw_dev
            device = torch.device(dev_str) if dev_str else None

        return cls(
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

    from_dict_instance = TensorDict.from_dict_instance

    # ---- Cloning ----

    def _clone(self, recurse: bool = True) -> RedisTensorDict:
        if recurse:
            # Deep clone: new UUID, copies all data
            new_td = RedisTensorDict(
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
            return RedisTensorDict._new_nested(
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
        }
        return state

    def __setstate__(self, state):
        import redis.asyncio as aioredis

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

        self._locked_tensordicts = []
        self._lock_id = set()
        self._is_shared = False
        self._is_memmap = False
        self._nested_tensordicts = {}
        self._cache_metadata = state.get("_cache_metadata", True)
        self._meta_cache = {} if self._cache_metadata else None

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
            f"RedisTensorDict(\n"
            f"    keys={keys_str},\n"
            f"    batch_size={batch_size},\n"
            f"    device={device},\n"
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


_register_tensor_class(RedisTensorDict)

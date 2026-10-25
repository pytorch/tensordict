# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The connection, the server keys and the lifecycle that the stores share.

:class:`_StoreConnection` is mixed into :class:`~tensordict.store.TensorDictStore`
and :class:`~tensordict.store.LazyStackedTensorDictStore`.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import threading
from typing import Any

import torch
from tensordict.store._utils import (
    _bytes_to_tensor,
    _compute_byte_ranges,
    _compute_covering_range,
    _decode_meta,
    _get_local_idx,
    _getitem_result_shape,
    _is_scattered_index,
    _LUA_GETRANGES,
    _LUA_SETRANGES,
    _prepare_indexed_value,
    _row_size,
    _str_to_dtype,
    _tensor_to_bytes,
)

_has_redis = importlib.util.find_spec("redis") is not None

# Separator used for nested key paths in Redis keys
_KEY_SEP = "."


def _server_key(namespace: str, td_id: str, suffix: str) -> str:
    """Returns the server key ``{namespace}:{td_id}:{suffix}``.

    The braces make ``td_id`` a hash tag, so that all the keys of a store land
    on the same shard of a Redis cluster.
    """
    return f"{namespace}:{{{td_id}}}:{suffix}"


def _connect_kwargs(
    host: str,
    port: int,
    db: int,
    unix_socket_path: str | None,
    redis_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Returns the keyword arguments of ``redis.asyncio.Redis`` for a server.

    With a ``unix_socket_path``, ``host`` and ``port`` are left out.
    """
    connect_kwargs = dict(redis_kwargs)
    if unix_socket_path is not None:
        connect_kwargs["unix_socket_path"] = unix_socket_path
    else:
        connect_kwargs["host"] = host
        connect_kwargs["port"] = port
    connect_kwargs["db"] = db
    return connect_kwargs


def _client_settings(client) -> tuple | None:
    """Reads the server of a ``redis.asyncio.Redis`` client.

    Returns ``(host, port, db, unix_socket_path, options)``, where ``options``
    holds credentials and standard Redis TLS settings. Copies (pickled or
    cloned) and metadata readers connect with these. Custom connection-class
    behavior beyond these settings is not reproduced. Returns ``None`` if the
    client does not expose ``connection_pool.connection_kwargs``.
    """
    try:
        kwargs = client.connection_pool.connection_kwargs
    except AttributeError:
        return None
    options = {
        name: kwargs[name]
        for name in ("username", "password")
        if kwargs.get(name) is not None
    }
    if _has_redis:
        import redis.asyncio as aioredis

        connection_class = getattr(client.connection_pool, "connection_class", None)
        if isinstance(connection_class, type) and issubclass(
            connection_class, aioredis.SSLConnection
        ):
            # Redis selects SSLConnection via ssl=True, not connection_class.
            # Pool/URL clients may omit defaults; keep those too, so caller
            # kwargs cannot replace the client's verification policy.
            options["ssl"] = True
            options.update(
                (name, kwargs.get(name, parameter.default))
                for name, parameter in inspect.signature(
                    aioredis.SSLConnection
                ).parameters.items()
                if name.startswith("ssl_")
            )
            options.update(
                (name, value)
                for name, value in kwargs.items()
                if name.startswith("ssl_")
            )
            unsupported = (
                options.keys() - inspect.signature(aioredis.Redis).parameters.keys()
            )
            if unsupported:
                raise RuntimeError(
                    "Cannot reproduce Redis TLS settings: "
                    + ", ".join(sorted(unsupported))
                )
    return (
        kwargs.get("host", "localhost"),
        kwargs.get("port", 6379),
        kwargs.get("db", 0),
        kwargs.get("path"),
        options,
    )


def _read_stored_metadata(
    connect_kwargs: dict[str, Any],
    namespace: str,
    td_id: str,
    suffixes: tuple[str, ...],
) -> list:
    """Reads the global keys ``suffixes`` of the store ``td_id``.

    ``from_store`` calls it before the store exists, with a temporary client and
    event loop: the constructor writes these keys.
    """
    import redis.asyncio as aioredis

    client = aioredis.Redis(**connect_kwargs)

    async def _read():
        try:
            pipe = client.pipeline()
            for suffix in suffixes:
                pipe.get(_server_key(namespace, td_id, suffix))
            return await pipe.execute()
        finally:
            # also when the read fails: the loop closes after this coroutine
            await client.aclose()

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_read())
    finally:
        loop.close()


class _StoreConnection:
    """The connection, the server keys, pickling and cleanup of a store.

    The mixin owns the background event loop, the ``redis.asyncio`` client and
    the names of the server keys. A store describes its layout with the class
    attributes and the two hooks below (:meth:`_entry_server_keys`,
    :meth:`_init_local_state`). Its constructor sets ``_cache_metadata`` and
    calls :meth:`_init_local_state`, sets the other attributes of
    ``_pickled_attrs``, and calls :meth:`_open_connection`.
    """

    #: Suffixes of the server keys that describe the whole store (batch size,
    #: device, ...). ``clear_redis`` deletes them.
    _global_key_suffixes: tuple[str, ...] = ()

    #: Instance attributes that ``__getstate__`` saves. ``__setstate__`` restores
    #: them, and takes the value of ``_pickle_defaults`` for a name that an
    #: older pickle lacks.
    _pickled_attrs: tuple[str, ...] = (
        "_backend",
        "_host",
        "_port",
        "_db",
        "_unix_socket_path",
        "_namespace",
        "_td_id",
        "_batch_size",
        "_device",
        "_redis_kwargs",
        "_is_locked",
        "_td_dim_names",
        "_cache_metadata",
    )
    _pickle_defaults: dict[str, Any] = {
        "_backend": "redis",
        "_is_locked": False,
        "_cache_metadata": True,
    }

    #: Whether ``_get_all_keys`` keeps the key registry in ``_keys_cache`` when
    #: ``cache_metadata`` is set. Keys that another handle adds later are then
    #: invisible until the store is reopened.
    _caches_keys: bool = True

    #: Key path of a nested view of a ``TensorDictStore``, empty for a store.
    #: ``clear_redis`` on a nested view deletes only the entries under it.
    _prefix: str = ""

    # ---- hooks ----

    def _entry_server_keys(self, key_path: str) -> tuple[str, ...]:
        """Returns the server keys that hold the entry ``key_path``.

        ``_adel_key`` and ``clear_redis`` delete them. The default returns the
        data and the metadata keys.
        """
        return (self._data_key(key_path), self._meta_key(key_path))

    def _init_local_state(self) -> None:
        """Resets the state that the server does not hold: lock bookkeeping and caches.

        The constructor and ``__setstate__`` call it once ``_cache_metadata`` is
        set. A store with more per-process state extends it.
        """
        self._lock_id = set()
        self._is_shared = False
        self._is_memmap = False
        # (shape, dtype) per key path, shared with the nested views
        self._meta_cache: dict[str, tuple[list[int] | None, torch.dtype]] | None = (
            {} if self._cache_metadata else None
        )
        # Shared mutable container ``[set | None]`` holding the set of all
        # registered leaf key paths. The list wrapper ensures that nested views
        # (which share the same list object) observe invalidation
        # (``_keys_cache[0] = None``) immediately.
        self._keys_cache: list[set[str] | None] = [None]

    # ---- connection ----

    def _connection_kwargs(self) -> dict[str, Any]:
        """Returns the keyword arguments of ``redis.asyncio.Redis`` for the server of the store.

        Raises a ``RuntimeError`` if the store was built with a client whose
        server cannot be read.
        """
        if self._redis_kwargs is None:
            raise RuntimeError(
                f"Cannot connect a copy of this {type(self).__name__}: it was "
                "built with a client that does not expose its server in "
                "client.connection_pool.connection_kwargs. Build the store "
                "with host, port and db instead."
            )
        return _connect_kwargs(
            self._host, self._port, self._db, self._unix_socket_path, self._redis_kwargs
        )

    def _open_connection(self, client=None) -> None:
        """Starts the background event loop and sets the client.

        Without ``client``, a new client connects with :meth:`_connection_kwargs`.
        With ``client``, the server of the client replaces the connection
        attributes, so that the copies of the store connect to it. If it
        cannot be read, ``_redis_kwargs`` is ``None`` and the store cannot be
        copied.
        """
        import redis.asyncio as aioredis

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._owns_loop = True
        if client is None:
            client = aioredis.Redis(**self._connection_kwargs())
        else:
            settings = _client_settings(client)
            if settings is None:
                self._redis_kwargs = None
            else:
                (
                    self._host,
                    self._port,
                    self._db,
                    self._unix_socket_path,
                    options,
                ) = settings
                self._redis_kwargs = {**self._redis_kwargs, **options}
        self._client = client

    def _run_sync(self, coro):
        """Run an async coroutine synchronously via the background event loop."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    # ---- server keys ----

    def _redis_key(self, suffix: str) -> str:
        """Build a Redis key with hash tag for cluster compatibility.

        Format: {namespace}:{td_id}:{suffix}
        The {td_id} is wrapped in Redis hash tags so all keys for this
        TensorDict land on the same shard.
        """
        return _server_key(self._namespace, self._td_id, suffix)

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

    # ---- keys and metadata ----

    async def _aget_all_keys(self) -> set[str]:
        """Get all registered key paths from Redis."""
        raw = await self._client.smembers(self._keys_registry_key)
        return {k.decode() if isinstance(k, bytes) else k for k in raw}

    def _get_all_keys(self) -> set[str]:
        """Get all registered key paths (sync wrapper).

        Uses the local ``_keys_cache`` when available, falling back to a
        Redis ``SMEMBERS`` round-trip. The result is cached for next time if
        the store caches the registry (``_caches_keys``) and its metadata.
        """
        if self._keys_cache[0] is not None:
            return self._keys_cache[0]
        keys = self._run_sync(self._aget_all_keys())
        if self._caches_keys and self._cache_metadata:
            self._keys_cache[0] = keys
        return keys

    @staticmethod
    def _is_key_homogeneous(meta: dict) -> bool:
        """Whether the elements of a lazy-stack entry have one shape and dtype.

        Only ``LazyStackedTensorDictStore`` writes the ``homogeneous`` flag, so
        every other entry is homogeneous.
        """
        return meta.get("homogeneous", "1") == "1"

    async def _aget_metadata_batch(
        self,
        key_paths: list[str],
        non_tensor: dict | None = None,
        layouts: dict | None = None,
    ) -> dict[str, tuple[list[int] | None, torch.dtype]]:
        """Get ``(shape, dtype)`` for multiple keys, using the local cache when available.

        Cache misses are fetched via a single Redis pipeline and stored back
        into ``_meta_cache`` (when caching is enabled). The shape is ``None``
        for a key whose elements differ in shape or dtype. When *non_tensor*
        is given, the metadata of non-tensor keys is stored there instead.
        When *layouts* is given, the ``layout`` field of each fetched entry, or
        ``None``, is stored there.
        """
        result: dict[str, tuple[list[int] | None, torch.dtype]] = {}
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
                if not raw_meta:
                    # deleted by another handle, or not a leaf
                    raise KeyError(f"key {kp} not found in {type(self).__name__}")
                meta = _decode_meta(raw_meta)
                if non_tensor is not None and meta.get("is_non_tensor") == "1":
                    non_tensor[kp] = meta
                    continue
                shape = (
                    json.loads(meta["shape"])
                    if self._is_key_homogeneous(meta)
                    else None
                )
                dtype = _str_to_dtype(meta["dtype"])
                result[kp] = (shape, dtype)
                if layouts is not None:
                    layouts[kp] = meta.get("layout")
                if self._meta_cache is not None:
                    self._meta_cache[kp] = (shape, dtype)

        return result

    async def _adel_key(self, key_path: str):
        """Delete a key's data, metadata, and registry entry."""
        pipe = self._client.pipeline()
        for server_key in self._entry_server_keys(key_path):
            pipe.delete(server_key)
        pipe.srem(self._keys_registry_key, key_path)
        await pipe.execute()
        if self._meta_cache is not None:
            self._meta_cache.pop(key_path, None)
        if self._keys_cache[0] is not None:
            self._keys_cache[0].discard(key_path)

    # ---- byte-range reads and writes ----

    def _queue_indexed_reads(
        self, pipe, key_paths: list[str], meta_map: dict, idx
    ) -> tuple[list[tuple[str, list[int], torch.dtype, object, bool]], list[str]]:
        """Queue on *pipe* the byte-range reads of ``tensor[idx]`` for each key of *key_paths*.

        A scattered index (integer index or mask) is read with one
        ``GETRANGES`` ``EVAL``, any other with a ``GETRANGE`` of the covering
        range. Returns ``(plan, fallback_kps)``: *plan* holds a
        ``(key_path, result_shape, dtype, local_idx, has_cmd)`` entry per read
        key, for :meth:`_assemble_indexed_reads`, and *fallback_kps* the keys
        whose index cannot be read by byte ranges.
        """
        scattered = _is_scattered_index(idx)
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
        return plan, fallback_kps

    def _assemble_indexed_reads(
        self, plan: list, raw_results: list, meta_map: dict, result: dict
    ) -> int:
        """Store in *result* the tensors that the reads of *plan* returned in *raw_results*.

        Returns the number of entries of *raw_results* that *plan* used.
        """
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
        return ri

    def _queue_range_writes(
        self,
        pipe,
        direct_kps: list[str],
        lua_kps: list[str],
        items: dict[str, tuple[torch.Tensor, object]],
        meta_map: dict,
    ) -> None:
        """Queue on *pipe* the writes of *items* for the keys that one ``SETRANGE`` or one ``SETRANGES`` ``EVAL`` writes.

        *direct_kps* are the keys whose index selects one byte range,
        *lua_kps* those whose scattered index selects several.
        """
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

    def _queue_covering_patches(
        self,
        pipe,
        cr_data: list[tuple[str, int, int, list[int], torch.dtype]],
        raw_covers: list,
        items: dict[str, tuple[torch.Tensor, object]],
    ) -> None:
        """Patch with *items* the covering ranges read in *raw_covers* and queue their ``SETRANGE`` on *pipe*.

        *cr_data* holds a ``(key_path, byte_offset, byte_length, shape, dtype)``
        entry per covering range.
        """
        for (kp, byte_offset, byte_length, shape, dtype), data in zip(
            cr_data, raw_covers
        ):
            rest = shape[1:]
            row_bytes = _row_size(shape, dtype)
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

    # ---- lifecycle ----

    def close(self):
        """Closes the connection to the server and stops the background event loop.

        The data stays on the server: reconnect with :meth:`from_store`, or
        delete it with :meth:`clear_redis` before you close. After
        ``close()``, this object cannot read or write. Garbage collection
        calls ``close()``.
        """
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
        """Deletes all the keys of this store from the server.

        This removes all tensor data, metadata, the key registry and the
        metadata of the store, such as its batch size and the stored
        TensorClass path. On a nested view, it removes only the entries
        under the view. The connection stays open, so you can write new
        entries. Call :meth:`close` to close it.
        """

        async def _aclear():
            key_paths = await self._aget_all_keys()
            if self._prefix:
                prefix = self._prefix + _KEY_SEP
                key_paths = [kp for kp in key_paths if kp.startswith(prefix)]
            pipe = self._client.pipeline()
            for key_path in key_paths:
                for server_key in self._entry_server_keys(key_path):
                    pipe.delete(server_key)
            if not self._prefix:
                pipe.delete(self._keys_registry_key)
                for suffix in self._global_key_suffixes:
                    pipe.delete(self._redis_key(suffix))
            elif key_paths:
                pipe.srem(self._keys_registry_key, *key_paths)
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

    # ---- pickling ----

    # Override: the event loop, its thread and the client cannot be pickled.
    # The copy opens its own connection to the same server.
    def __getstate__(self) -> dict[str, Any]:
        self._connection_kwargs()  # raises if the copy cannot connect
        return {name: getattr(self, name) for name in self._pickled_attrs}

    # Override: see __getstate__.
    def __setstate__(self, state: dict[str, Any]) -> None:
        defaults = self._pickle_defaults
        for name in self._pickled_attrs:
            value = state.get(name, defaults[name]) if name in defaults else state[name]
            setattr(self, name, value)
        self._init_local_state()
        self._open_connection()
        was_locked = self._is_locked
        self._is_locked = False
        if was_locked:
            self.lock_()

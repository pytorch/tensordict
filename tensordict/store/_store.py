# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Key-value store backed TensorDict implementations (Redis, Dragonfly, etc.)."""

from __future__ import annotations

import importlib
import json
import pickle
import uuid
from numbers import Number
from typing import Any, Callable, Literal, Sequence, Type, TYPE_CHECKING

import torch
from tensordict._base.unsupported import _StoreUnsupported
from tensordict._deprecation import deprecated
from tensordict._indexing import _as_tuple, _getitem_batch_size, convert_ellipsis_to_idx
from tensordict._td import (
    _TensorDictKeysView,
    _unravel_key_to_tuple,
    NO_DEFAULT,
    TensorDict,
)
from tensordict.base import (
    _has_lazy_stack,
    _is_leaf_nontensor,
    _is_tensor_collection,
    _normalize_from_dict_args,
    _register_tensor_class,
    BEST_ATTEMPT_INPLACE,
    is_tensor_collection,
    TensorDictBase,
)
from tensordict.utils import (
    _as_context_manager,
    _is_tensorclass,
    _LOCK_ERROR,
    _warn_nested_key_in_flat_keys,
    is_non_tensor,
    KEY_SEPARATOR,
    NestedKey,
    unravel_key,
)

_has_redis = importlib.util.find_spec("redis", None) is not None

#: The type of the ``backend`` argument of the stores, ``"redis"`` or ``"dragonfly"``.
#:
#: The value is a label, kept in the store and shown by its ``repr``. Every
#: server speaks the Redis protocol, so it does not change how the store
#: connects.
STORE_BACKENDS = Literal["redis", "dragonfly"]

if TYPE_CHECKING:
    from typing_extensions import Self
else:
    Self = Any

from tensordict.store._connection import (
    _client_settings,
    _connect_kwargs,
    _KEY_SEP,
    _read_stored_metadata,
    _StoreConnection,
)

# Re-export every helper of _utils, including those that only
# tensordict.store._lazy uses.
from tensordict.store._utils import (  # noqa: F401
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
    _non_tensor_positions,
    _non_tensor_write_positions,
    _normalize_index,
    _prepare_indexed_value,
    _str_to_dtype,
    _tensor_to_bytes,
    _warn_missing_key,
)


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
        resolved = explicit
    else:
        class_path: str | None = explicit
        if class_path is None and raw_stored is not None:
            class_path = (
                raw_stored.decode() if isinstance(raw_stored, bytes) else raw_stored
            )
        if class_path is None:
            return None
        module_path, _, class_name = class_path.rpartition(".")
        mod = importlib.import_module(module_path)
        resolved = getattr(mod, class_name)
    if not isinstance(resolved, type) or not _is_tensorclass(resolved):
        raise TypeError(f"Expected a TensorClass type, got {resolved!r}")
    return resolved


def _iter_registry_keys(paths, include_nested, leaves_only):
    """Yields the keys of a store from its sorted leaf paths.

    The registry holds the leaf paths only. With ``include_nested=True`` and
    ``leaves_only=False`` the keys of the nested tensordicts are rebuilt from
    the path prefixes and yielded after their content, as
    ``TensorDict.keys(True)`` does. The paths of a nested tensordict are
    contiguous once sorted, so each prefix is yielded once.
    """
    if not include_nested:
        seen = set()
        for path in paths:
            parts = path.split(_KEY_SEP)
            top_key = parts[0]
            if top_key in seen:
                continue
            seen.add(top_key)
            if leaves_only and len(parts) > 1:
                continue
            yield top_key
        return
    if leaves_only:
        for path in paths:
            parts = path.split(_KEY_SEP)
            yield tuple(parts) if len(parts) > 1 else parts[0]
        return
    # The parent of the previous path: its prefixes are not yielded yet.
    open_prefix: tuple[str, ...] = ()
    for path in paths:
        parts = tuple(path.split(_KEY_SEP))
        prefix = parts[:-1]
        if prefix != open_prefix:
            # yield the prefixes this path leaves, deepest first
            common = 0
            for a, b in zip(open_prefix, prefix):
                if a != b:
                    break
                common += 1
            for depth in range(len(open_prefix), common, -1):
                yield open_prefix[:depth] if depth > 1 else open_prefix[0]
            open_prefix = prefix
        yield parts if len(parts) > 1 else parts[0]
    for depth in range(len(open_prefix), 0, -1):
        yield open_prefix[:depth] if depth > 1 else open_prefix[0]


def _rename_plan(all_keys, old_path, new_path, old_key, new_key):
    """Plans the move of the entry at ``old_path`` to ``new_path`` in a store.

    Returns ``None`` if nothing is stored at or below ``old_path``. Otherwise
    returns ``(renames, deletes)``: the ``(source, destination)`` pairs of
    leaf paths, ordered so that no leaf is overwritten before it moves, and
    the leaves at or below ``new_path`` that the entry replaces, as
    ``TensorDict`` does. A destination below a leaf raises ``KeyError``,
    unless that leaf moves.
    """
    old_prefix = old_path + _KEY_SEP
    new_prefix = new_path + _KEY_SEP
    if old_path in all_keys:
        renames = [(old_path, new_path)]
    else:
        renames = [
            (k, new_prefix + k[len(old_prefix) :])
            for k in all_keys
            if k.startswith(old_prefix)
        ]
    if not renames:
        return None
    if old_path == new_path:
        return [], []
    parts = new_path.split(_KEY_SEP)
    for i in range(1, len(parts)):
        parent = _KEY_SEP.join(parts[:i])
        # a leaf at or below old_path moves away
        moves = parent == old_path or parent.startswith(old_prefix)
        if not moves and parent in all_keys:
            raise KeyError(
                f"Cannot rename {old_key!r} to {new_key!r}: the new key is "
                "below a leaf."
            )
    sources = {src for src, _ in renames}
    deletes = [
        k
        for k in all_keys
        if (k == new_path or k.startswith(new_prefix)) and k not in sources
    ]
    # A destination can be the source of another move: a shorter one when
    # new_path holds old_path, a longer one when old_path holds new_path.
    # Move that source first.
    renames.sort(key=lambda pair: len(pair[0]), reverse=new_path.startswith(old_prefix))
    return renames, deletes


class _StoreTDKeysView(_TensorDictKeysView):
    """Keys view for TensorDictStore backed by a Redis key registry."""

    def __iter__(self):
        td = self.tensordict
        # Get all registered keys from Redis
        all_keys = td._get_all_keys()
        prefix = td._prefix
        prefix_dot = prefix + _KEY_SEP if prefix else ""

        # The registry is a set: sort the paths of this level, as
        # to_tensordict() does, so that the keys come in one order. The paths
        # of a nested tensordict are contiguous once sorted, so values() and
        # items(), which recurse into each nested tensordict, follow this
        # order too.
        if prefix:
            start = len(prefix_dot)
            paths = sorted(k[start:] for k in all_keys if k.startswith(prefix_dot))
        else:
            paths = sorted(all_keys)
        return _iter_registry_keys(paths, self.include_nested, self.leaves_only)

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
        # Exact leaf match, or prefix match (nested tensordict)
        prefix_check = full_key + _KEY_SEP
        found = full_key in all_keys or any(
            k.startswith(prefix_check) for k in all_keys
        )
        if found and not isinstance(key, str) and not self.include_nested:
            _warn_nested_key_in_flat_keys(key, td)
        return found

    def __len__(self):
        return sum(1 for _ in self)


class TensorDictStore(_StoreConnection, _StoreUnsupported, TensorDictBase):
    """A tensordict whose entries live on a Redis-compatible server.

    Supports `Redis <https://redis.io>`_, `Dragonfly <https://dragonflydb.io>`_,
    `KeyDB <https://docs.keydb.dev>`_, and any other Redis-wire-compatible server.

    Use it to share data between processes or machines, or to keep data out of
    the RAM of the training process. Each tensor entry is stored as one string
    of raw bytes on the server. Its shape and dtype are stored next to it, so
    that ``keys()`` and ``shape`` do not download the data. See
    :ref:`storage` for a guide.

    Each read or write blocks until the server answers. One call sends its
    commands to the server in one round-trip.

    .. rubric:: Lifecycle and limits

    - The store needs a running server. By default it connects to
      ``localhost:6379``. To start a local Redis server, run
      ``docker run -d -p 6379:6379 redis``.
    - The data stays on the server after :meth:`close`, after the object is
      garbage collected, and after the process ends. The server keeps it in
      its memory until you delete it.
    - To reconnect from another process, pass the ``td_id`` of the store to
      :meth:`from_store`, which also reads the batch size and the device. The
      id is in the private attribute ``_td_id``, and is shown by ``repr``.
      There is no public accessor.
    - :meth:`close` closes the connection and stops the background loop. The
      object cannot read or write after that, and the data stays on the
      server. :meth:`clear_redis` deletes the data and keeps the connection
      open.
    - Each tensor entry is one string on the server. Redis limits a string to
      512 MB by default (``proto-max-bulk-len``), and a larger write raises
      ``redis.exceptions.ResponseError``. For larger entries, raise the limit
      on the server, or split the entry.
    - Non-tensor values are stored as JSON. A value that JSON cannot encode
      is stored with :mod:`pickle`, and each read unpickles it. Any client
      that can write to the server can then run code in the readers, so
      connect only to a server that you trust.

    Keyword Args:
        backend (str, optional): a label for the kind of server, kept in the
            store and shown by its ``repr``, for example ``"redis"``,
            ``"dragonfly"`` or ``"keydb"``. Every server speaks the same
            protocol, so the value does not change how the store connects.
            Defaults to ``"redis"``.
        host (str, optional): the host name of the server. Defaults to
            ``"localhost"``.
        port (int, optional): the port of the server. Defaults to ``6379``.
        db (int, optional): the database number on the server. Defaults to
            ``0``.
        unix_socket_path (str, optional): the path of a Unix socket, for a
            server on the same machine. If set, ``host`` and ``port`` are
            ignored. Defaults to ``None``.
        prefix (str, optional): the namespace of the keys of the store on the
            server. Defaults to ``"tensordict"``.
        batch_size (torch.Size or list of int, optional): the batch size of
            the store: the leading dimensions that all its entries share.
            Defaults to ``None``, which gives ``torch.Size([])``.
        device (torch.device, str or int, optional): the device of the
            tensors that reads return. The server holds CPU bytes. Defaults to
            ``None``: reads return CPU tensors, and the store has no device.
        client (redis.asyncio.Redis, optional): an existing client. If given,
            ``host``, ``port``, ``db`` and ``unix_socket_path`` are ignored.
            Defaults to ``None``, which creates a client.
        td_id (str, optional): the identifier of the store on the server.
            Defaults to ``None``, which creates a new UUID. The constructor
            does not read the batch size of an existing store: use
            :meth:`from_store` to reconnect.
        cache_metadata (bool, optional): if ``True``, keeps the shape and
            dtype of each entry in the process, to save round-trips to the
            server. Defaults to ``True``.
        **redis_kwargs: passed to ``redis.asyncio.Redis``.

    Raises:
        ModuleNotFoundError: if the ``redis`` package is not installed
            (``pip install redis``).

    Examples:
        The example needs a Redis server on ``localhost:6379``.

        >>> import torch
        >>> from tensordict.store import TensorDictStore
        >>> td = TensorDictStore(batch_size=[100])  # doctest: +SKIP
        >>> td["obs"] = torch.zeros(100, 84)  # doctest: +SKIP
        >>> td["obs"].shape  # doctest: +SKIP
        torch.Size([100, 84])
        >>> type(td.to_tensordict()).__name__  # doctest: +SKIP
        'TensorDict'
        >>> td.clear_redis()  # doctest: +SKIP
        >>> td.close()  # doctest: +SKIP

        To connect to a Dragonfly server on another host, pass its address,
        for example
        ``TensorDictStore(backend="dragonfly", host="dragonfly-host", batch_size=[100])``.
    """

    _td_dim_names = None
    _masking_on_dense_copy = True
    _masked_select_on_dense_copy = True
    # each read fetches the entry from the server into a new tensor
    _reads_return_copies = True
    # update({key: self}) writes a copy of the data on the server
    _update_stores_copy_of_self = True
    _leaves_built_on_read = True
    _contiguous_via_to_tensordict = True
    _pin_memory_unsupported = True
    _storage_holds_no_gradients = True
    _copy_shares_storage = True
    _rename_key_lenient = True
    # td[key] = value writes into an existing entry on the server
    _setitem_key_inplace = BEST_ATTEMPT_INPLACE
    # Set on the shallow copies that _clone(recurse=False) returns: they share
    # the data and the metadata keys of the store, but not its batch size.
    _is_shallow_copy = False

    _global_key_suffixes = ("__batch_size__", "__device__", "__tensorclass__")
    _pickled_attrs = _StoreConnection._pickled_attrs + (
        "_prefix",
        "_tensorclass_cls",
        "_is_shallow_copy",
    )
    _pickle_defaults = {
        **_StoreConnection._pickle_defaults,
        "_tensorclass_cls": None,
        "_is_shallow_copy": False,
    }

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

        if batch_size is None:
            batch_size = torch.Size(())

        self._backend = backend
        self._cache_metadata = cache_metadata
        self._init_local_state()

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
        self._open_connection(client)

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
        obj._cache_metadata = parent._cache_metadata
        obj._init_local_state()
        obj._td_dim_names = None

        obj._backend = parent._backend
        obj._td_id = parent._td_id
        obj._prefix = key_prefix
        obj._namespace = parent._namespace
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

    # Override: also resets the cache of the nested views.
    def _init_local_state(self) -> None:
        super()._init_local_state()
        # Nested TensorDict cache
        self._nested_tensordicts: dict[str, TensorDictStore] = {}

    # ---- Redis key construction ----

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

    async def _aset_non_tensor(self, key_path: str, value: Any, array: bool = False):
        """Store a non-tensor value in Redis.

        With ``array=True``, *value* is the list of the elements along dim 0,
        stored as a ``json_array`` as indexed writes store it. A value that
        JSON cannot encode is pickled either way.
        """
        # Try JSON first, fall back to pickle
        try:
            serialized = json.dumps(value)
            encoding = "json_array" if array else "json"
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

    async def _aset_non_tensor_at(
        self,
        key_path: str,
        value: Any,
        idx: int | slice | torch.Tensor | list | range,
    ):
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

        positions = _non_tensor_write_positions(idx, len(array))
        if isinstance(positions, int):
            array[positions] = value
        else:
            if not isinstance(value, (list, tuple)):
                value = [value] * len(positions)
            for i, v in zip(positions, value):
                array[i] = v

        serialized = json.dumps(array).encode("utf-8")
        new_meta = {"is_non_tensor": "1", "encoding": "json_array"}
        pipe = self._client.pipeline()
        pipe.set(self._data_key(key_path), serialized)
        pipe.hset(self._meta_key(key_path), mapping=new_meta)
        pipe.sadd(self._keys_registry_key, key_path)
        await pipe.execute()
        if self._keys_cache[0] is not None:
            self._keys_cache[0].add(key_path)

    async def _abatch_set_non_tensor_at(
        self,
        items: list[tuple[str, Any]],
        idx: int | slice | torch.Tensor | list | range,
    ):
        """Batch read-modify-write for multiple non-tensor keys at once.

        Equivalent to calling :meth:`_aset_non_tensor_at` for each item, but
        uses a single pipeline for the reads and a single pipeline for the
        writes, avoiding per-key ``_run_sync`` overhead.
        """
        if not items:
            return

        # Phase 1: pipeline GET all keys + metadata
        pipe = self._client.pipeline()
        for key_path, _ in items:
            pipe.get(self._data_key(key_path))
            pipe.hgetall(self._meta_key(key_path))
        raw_results = await pipe.execute()

        batch_dim = self._batch_size[0] if self._batch_size else 1

        # Phase 2: patch arrays locally
        write_pipe = self._client.pipeline()
        for i, (key_path, value) in enumerate(items):
            data = raw_results[2 * i]
            raw_meta = raw_results[2 * i + 1]
            meta = _decode_meta(raw_meta) if raw_meta else {}

            if data is not None and meta.get("encoding") == "json_array":
                text = data.decode() if isinstance(data, bytes) else data
                array = json.loads(text)
            elif data is not None:
                blob_val = self._deserialize_non_tensor(data, meta)
                array = [blob_val] * batch_dim
            else:
                array = [None] * batch_dim

            positions = _non_tensor_write_positions(idx, len(array))
            if isinstance(positions, int):
                array[positions] = value
            else:
                if not isinstance(value, (list, tuple)):
                    value = [value] * len(positions)
                for j, v in zip(positions, value):
                    array[j] = v

            serialized = json.dumps(array).encode("utf-8")
            new_meta = {"is_non_tensor": "1", "encoding": "json_array"}
            write_pipe.set(self._data_key(key_path), serialized)
            write_pipe.hset(self._meta_key(key_path), mapping=new_meta)
            write_pipe.sadd(self._keys_registry_key, key_path)

        # Phase 3: single write pipeline
        await write_pipe.execute()
        if self._keys_cache[0] is not None:
            for key_path, _ in items:
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

    def _is_tensor_entry(self, key_path: str) -> bool:
        """Whether the entry at ``key_path`` exists and holds a tensor."""
        if self._meta_cache is not None and key_path in self._meta_cache:
            return True
        meta = self._run_sync(self._aget_metadata(key_path))
        return bool(meta) and meta.get("is_non_tensor") != "1"

    async def _aget_metadata(self, key_path: str) -> dict:
        """Retrieve metadata for a key without downloading tensor data."""
        raw = await self._client.hgetall(self._meta_key(key_path))
        return _decode_meta(raw)

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
        Non-tensor keys are read in full and indexed as in ``store[idx]``.
        """
        if not key_paths:
            return {}

        non_tensor_metas: dict[str, dict] = {}
        meta_map = await self._aget_metadata_batch(
            key_paths, non_tensor=non_tensor_metas
        )
        result: dict[str, torch.Tensor | Any] = {}
        if non_tensor_metas:
            pipe = self._client.pipeline()
            for kp in non_tensor_metas:
                pipe.get(self._data_key(kp))
            for (kp, meta), data in zip(non_tensor_metas.items(), await pipe.execute()):
                if data is not None:
                    result[kp] = self._deserialize_non_tensor(data, meta, idx=idx)
            key_paths = [kp for kp in key_paths if kp not in non_tensor_metas]
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
                result[kp] = tensor[_as_tuple(idx)]

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
            result[kp] = tensor[_as_tuple(idx)]

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
        # Instead of creating a full zero tensor and sending it over the wire,
        # we pre-extend the Redis string to the right length using SETRANGE
        # with a single null byte at the end (Redis auto-fills with \0) and
        # only write metadata.  The actual data is written by the normal
        # SETRANGE / Lua path below.
        all_keys = await self._aget_all_keys()
        new_kps = [kp for kp in key_paths if kp not in all_keys]
        if new_kps:
            batch_dim = self._batch_size[0] if self._batch_size else 0
            pipe = self._client.pipeline()
            for kp in new_kps:
                value, idx = items[kp]
                # For int index value is one element (shape = elem_shape);
                # for slice/tensor/list the first dim is the indexed dim.
                if isinstance(idx, int) and not isinstance(idx, bool):
                    elem_shape = list(value.shape)
                else:
                    elem_shape = list(value.shape[1:]) if value.ndim > 0 else []
                full_shape = [batch_dim] + elem_shape
                _compute_byte_ranges(full_shape, value.dtype, idx)
                elem_size = value.element_size()
                numel = batch_dim * (
                    int(torch.tensor(elem_shape).prod().item()) if elem_shape else 1
                )
                total_bytes = numel * elem_size
                if total_bytes > 0:
                    pipe.setrange(self._data_key(kp), total_bytes - 1, b"\x00")
                meta = {
                    "shape": json.dumps(full_shape),
                    "dtype": _dtype_to_str(value.dtype),
                }
                pipe.hset(self._meta_key(kp), mapping=meta)
                pipe.sadd(self._keys_registry_key, kp)
                if self._meta_cache is not None:
                    self._meta_cache[kp] = (full_shape, value.dtype)
                if self._keys_cache[0] is not None:
                    self._keys_cache[0].add(kp)
            await pipe.execute()

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
            existing[_as_tuple(idx)] = value
            pipe.set(self._data_key(kp), _tensor_to_bytes(existing))
        await pipe.execute()

    @staticmethod
    def _deserialize_non_tensor(data: bytes, meta: dict, idx=None) -> Any:
        """Deserialize a non-tensor value from Redis.

        When *idx* is provided the element(s) at *idx* are returned.
        For ``json_array`` encoding this indexes directly into the stored
        array.  For scalar encodings (``json`` / ``pickle``) the single
        stored value is returned as-is (broadcast semantics).
        """
        encoding = meta.get("encoding", "json")
        if encoding not in ("json", "json_array"):
            # pickle bytes are not UTF-8: only JSON is decoded
            return pickle.loads(data)
        text = data.decode() if isinstance(data, bytes) else data

        if encoding == "json_array":
            array = json.loads(text)
            positions = None if idx is None else _non_tensor_positions(idx, len(array))
            if isinstance(positions, int):
                return array[positions]
            if positions is not None:
                array = [array[p] for p in positions]
            # Wrap in NonTensorStack for TensorClass compat
            from tensordict.tensorclass import NonTensorData, NonTensorStack

            stack = NonTensorStack(
                *[NonTensorData(data=v, batch_size=[]) for v in array]
            )
            if idx is not None and positions is None:
                # other indices, e.g. multi-dimensional ones: index locally
                return stack[idx]
            return stack

        # Scalar encoding — return the single stored value (broadcast).
        return json.loads(text)

    # ---- TensorDictBase interface: batch_size / device / names ----

    # Override: the device can be set, and is persisted on the server.
    @TensorDictBase.device.setter
    def device(self, value):
        self._device = torch.device(value) if value is not None else None
        self._run_sync(self._apersist_metadata())

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

        # Single batched pipeline for all keys.
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

    # Override: writes the indexed entries on the server in one batch (SETRANGE),
    # without a _SubTensorDict.
    def _setitem_index(self, index, value):
        if isinstance(index, list):
            index = torch.tensor(index)
        _check_indexed_value(self, value)
        if isinstance(value, Number):
            # a scalar is written to every tensor entry, in its dtype; the
            # non-tensor entries are left as they are
            key_paths = [
                self._full_key_path(_KEY_SEP.join(_unravel_key_to_tuple(key)))
                for key in self.keys(include_nested=True, leaves_only=True)
            ]
            non_tensor = {}
            tensor_paths = self._run_sync(
                self._aget_metadata_batch(key_paths, non_tensor=non_tensor)
            )
            self._run_sync(
                self._abatch_set_at({kp: (value, index) for kp in tensor_paths})
            )
            return

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

        if non_tensor_items:
            # check the index before writing anything, to avoid a partial write
            _non_tensor_write_positions(
                index, self._batch_size[0] if self._batch_size else 1
            )
        if tensor_items:
            self._run_sync(self._abatch_set_at(tensor_items))
        if non_tensor_items:
            self._run_sync(self._abatch_set_non_tensor_at(non_tensor_items, index))

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
        if not _is_tensor_collection(type(first)):
            raise ValueError(
                f"{key[0]!r} is a {type(first).__name__}, not a tensordict, "
                f"so it has no entry {key[1] if len(key) == 2 else key[1:]!r}."
            )
        return first._get_at_tuple(key[1:], idx, default=default, **kwargs)

    # Override: looks the key up in the key registry.
    def _contains_str(self, key):
        key_path = self._full_key_path(key)
        all_keys = self._get_all_keys()
        if key_path in all_keys:
            return True
        # Also check if it's a nested key prefix
        prefix_check = key_path + _KEY_SEP
        return any(k.startswith(prefix_check) for k in all_keys)

    def _write_non_tensor(self, key_path: str, value: Any) -> None:
        """Write a ``NonTensorData`` or ``NonTensorStack`` value."""
        from tensordict.tensorclass import NonTensorData, NonTensorStack

        if isinstance(value, NonTensorData):
            self._run_sync(self._aset_non_tensor(key_path, value.data))
            return
        # A stack along a 1-D batch holds one value per element: store them as
        # indexed writes do, so that indexed reads follow the index. Other
        # stacks stay one blob, as json_array entries are read as 1-D stacks.
        array = (
            isinstance(value, NonTensorStack)
            and self.batch_dims == 1
            and value.batch_size == self.batch_size
        )
        self._run_sync(self._aset_non_tensor(key_path, value.tolist(), array=array))

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
            value = self._validate_value(value, check_shape=True, key=key)
        if self.is_locked and not ignore_lock:
            if not inplace:
                raise RuntimeError(_LOCK_ERROR)

        key_path = self._full_key_path(key)

        if is_non_tensor(value):
            self._write_non_tensor(key_path, value)
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
            value = self._validate_value(value, check_shape=True, key=key)
        if self.is_locked and not inplace:
            raise RuntimeError(_LOCK_ERROR)

        if isinstance(value, torch.Tensor):
            self._run_sync(self._aset_tensor(key_path, value))
        elif is_non_tensor(value):
            self._write_non_tensor(key_path, value)
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
        nested = False
        if is_tensor_collection(value) and not is_non_tensor(value):
            all_keys = self._get_all_keys()
            nested = any(k.startswith(key_path + _KEY_SEP) for k in all_keys)
        if nested:
            # The store holds a tensordict at key: write each leaf of value under
            # its full key, as store[key][idx] = value does. That would create
            # the leaves that the store lacks, so check first that it has them.
            for subkey in value.keys(True, True, is_leaf=_is_leaf_nontensor):
                subkey = _unravel_key_to_tuple(subkey)
                if _KEY_SEP.join((key_path, *subkey)) not in all_keys:
                    raise KeyError(
                        f"key {(key, *subkey)} not found in {type(self).__name__}"
                    )
            if isinstance(idx, tuple) and any(i is Ellipsis for i in idx):
                # an Ellipsis stands for the batch dims, as on a TensorDict;
                # store[key][idx] = value would read it on the dims of each leaf
                idx = convert_ellipsis_to_idx(idx, self.batch_size)
            TensorDictStore._new_nested(parent=self, key_prefix=key_path)[idx] = value
            return self
        if not isinstance(value, torch.Tensor) and not self._is_tensor_entry(key_path):
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

    # Override: deletes the keys on the server.
    def _del(self, key: tuple[str, ...]) -> None:
        key_path = self._full_key_path(_KEY_SEP.join(key))

        all_keys = self._get_all_keys()
        # Delete exact key
        found = key_path in all_keys
        if found:
            self._run_sync(self._adel_key(key_path))

        # Delete nested keys (prefix match)
        prefix_check = key_path + _KEY_SEP
        nested_to_delete = [k for k in all_keys if k.startswith(prefix_check)]
        for nested_key in nested_to_delete:
            self._run_sync(self._adel_key(nested_key))

        # Clean up nested cache
        self._nested_tensordicts.pop(key[0], None)

        if not found and not nested_to_delete:
            # stacklevel 4: skip del_ and its _lock_blocked wrapper
            _warn_missing_key(
                self, "del_", key[0] if len(key) == 1 else key, stacklevel=4
            )

    def _popped_value(self, value):
        # Override: a nested entry is a view that reads from the store, and
        # del_ deletes its data: return a copy of it.
        if isinstance(value, TensorDictStore):
            return value.to_tensordict()
        return value

    # Override: renames the keys on the server, without reading the data.
    def _rename_key(self, old_key: NestedKey, new_key: NestedKey) -> None:
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

        async def _arename(old_path, new_path):
            all_keys = await self._aget_all_keys()
            plan = _rename_plan(all_keys, old_path, new_path, old_key, new_key)
            if plan is None:
                return None
            renames, deletes = plan
            pipe = self._client.pipeline()
            # the entry replaces what is stored at new_path, as in TensorDict
            for k in deletes:
                pipe.delete(self._data_key(k))
                pipe.delete(self._meta_key(k))
                pipe.srem(self._keys_registry_key, k)
            for src, dst in renames:
                pipe.rename(self._data_key(src), self._data_key(dst))
                pipe.rename(self._meta_key(src), self._meta_key(dst))
                pipe.srem(self._keys_registry_key, src)
                pipe.sadd(self._keys_registry_key, dst)
            await pipe.execute()
            return plan

        plan = self._run_sync(_arename(old_path, new_path))
        # Invalidate keys cache — rename changes the set of keys
        self._keys_cache[0] = None
        if plan is not None and self._meta_cache is not None:
            renames, deletes = plan
            for key_path in deletes:
                self._meta_cache.pop(key_path, None)
            for src, dst in renames:
                self._meta_cache.pop(src, None)
                self._meta_cache.pop(dst, None)
        if plan is None:
            # stacklevel 3: skip rename_key_
            _warn_missing_key(self, "rename_key_", old_key, stacklevel=3)

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

    # Override: walk the cached sub-tensordicts, without a server read.
    def _lock_sub_tensordicts(self):
        return list(self._nested_tensordicts.values())

    # ---- Materialization ----

    # Override: reads all the leaves in one round trip (retain_none has no
    # effect on a store).
    def _to_tensordict(self, *, retain_none: bool | None) -> TensorDict:
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

    # Override: deprecated alias of to_tensordict(), removed in 0.17.
    @deprecated(
        "TensorDictStore.to_local()", removal="0.17", replacement="to_tensordict()"
    )
    def to_local(self) -> TensorDict:
        """Returns a :class:`~tensordict.TensorDict` with a copy of the data of the store.

        .. deprecated:: 0.15
            Use :meth:`to_tensordict` instead.
        """
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
        batch_dims: int | None = None,
        names: Sequence[str] | None = None,
        **kwargs,
    ):
        """Returns a new store that holds the entries of a dict or a tensordict.

        Args:
            input_dict (dict or TensorDictBase): the entries to write to the
                server.

        Keyword Args:
            backend (str, optional): a label for the kind of server, as in
                :class:`TensorDictStore`. Defaults to ``"redis"``.
            host (str, optional): the host name of the server. Defaults to
                ``"localhost"``.
            port (int, optional): the port of the server. Defaults to ``6379``.
            db (int, optional): the database number on the server. Defaults
                to ``0``.
            unix_socket_path (str, optional): the path of a Unix socket, for a
                server on the same machine. If set, ``host`` and ``port`` are
                ignored. Defaults to ``None``.
            prefix (str, optional): the namespace of the keys of the store on
                the server. Defaults to ``"tensordict"``.
            auto_batch_size (bool, optional): if ``True`` and ``input_dict``
                is a dict without ``batch_size``, sets the batch size to the
                largest leading shape that all the entries share. Defaults to
                ``False``.
            batch_size (torch.Size or list of int, optional): the batch size
                of the store. Defaults to ``None``, which takes the batch size
                of a tensordict ``input_dict``, and ``[]`` (or the inferred
                batch size with ``auto_batch_size=True``) for a dict.
            device (torch.device, str or int, optional): the device of the
                tensors that reads return, as in :class:`TensorDictStore`.
                Defaults to ``None``.
            batch_dims (int, optional): with ``auto_batch_size``, the maximum
                number of batch dimensions; giving it turns
                ``auto_batch_size`` on. Exclusive with ``batch_size``.
                Defaults to ``None``.
            names (list of str, optional): the names of the batch dimensions,
                one per dimension. Defaults to ``None``.
            **kwargs: passed to the :class:`TensorDictStore` constructor, such
                as ``client`` or ``td_id``, and options of the Redis client.

        Returns:
            A new TensorDictStore.

        Examples:
            The example needs a Redis server on ``localhost:6379``.

            >>> import torch
            >>> from tensordict.store import TensorDictStore
            >>> store = TensorDictStore.from_dict({"a": torch.zeros(5, 3)}, batch_size=[5])  # doctest: +SKIP
            >>> store.batch_size  # doctest: +SKIP
            torch.Size([5])
            >>> store.clear_redis()  # doctest: +SKIP
            >>> store.close()  # doctest: +SKIP
        """
        batch_size = _normalize_from_dict_args(
            input_dict, batch_size=batch_size, batch_dims=batch_dims
        )
        if batch_size is None:
            batch_size = torch.Size([])
            if auto_batch_size or batch_dims is not None:
                input_dict = TensorDict(
                    input_dict, batch_size=batch_size
                ).auto_batch_size_(batch_dims=batch_dims)
                batch_size = input_dict.batch_size

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
        if names is not None:
            out.names = names
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
        """Returns a new store that holds a copy of the data of ``td``.

        A :class:`~tensordict.LazyStackedTensorDict` gives a
        :class:`~tensordict.store.LazyStackedTensorDictStore`, which keeps
        entries of different shapes. For a tensorclass, the store records its
        class, so that :meth:`from_store` returns an instance of it.

        Args:
            td (TensorDictBase): the tensordict or tensorclass to copy to the
                server.

        Keyword Args:
            backend (str, optional): a label for the kind of server, as in
                :class:`TensorDictStore`. Defaults to ``"redis"``.
            host (str, optional): the host name of the server. Defaults to
                ``"localhost"``.
            port (int, optional): the port of the server. Defaults to ``6379``.
            db (int, optional): the database number on the server. Defaults
                to ``0``.
            unix_socket_path (str, optional): the path of a Unix socket, for a
                server on the same machine. If set, ``host`` and ``port`` are
                ignored. Defaults to ``None``.
            prefix (str, optional): the namespace of the keys of the store on
                the server. Defaults to ``"tensordict"``.
            device (torch.device, str or int, optional): the device of the
                tensors that reads return. Defaults to ``None``, which takes
                the device of ``td``.
            **kwargs: passed to the :class:`TensorDictStore` constructor, such
                as ``client`` or ``td_id``, and options of the Redis client.

        Returns:
            A :class:`TensorDictStore`, or a
            :class:`~tensordict.store.LazyStackedTensorDictStore` for a lazy
            stack.

        Examples:
            The example needs a Redis server on ``localhost:6379``. For a
            server on another machine, pass ``host``.

            >>> import torch
            >>> from tensordict import TensorDict
            >>> from tensordict.store import TensorDictStore
            >>> local = TensorDict(obs=torch.zeros(10, 84), batch_size=[10])
            >>> remote = TensorDictStore.from_tensordict(local)  # doctest: +SKIP
            >>> remote["obs"].shape  # doctest: +SKIP
            torch.Size([10, 84])
            >>> remote.clear_redis()  # doctest: +SKIP
            >>> remote.close()  # doctest: +SKIP
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

    # Override: takes the connection arguments of the store.
    @classmethod
    def from_schema(
        cls,
        schema: dict[str, tuple[list[int] | torch.Size, torch.dtype]],
        *,
        batch_size: Sequence[int] | torch.Size | None = None,
        backend: STORE_BACKENDS = "redis",
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        unix_socket_path: str | None = None,
        prefix: str = "tensordict",
        device=None,
        **kwargs,
    ) -> TensorDictStore:
        """Returns a new store with zero-filled entries of the given shapes and dtypes.

        Each key of ``schema`` is registered with its shape and dtype. The
        server allocates each entry with ``SETRANGE`` and fills it with zeros.
        No tensor data goes through the Python process, but the server
        allocates the full size in its memory at once.

        Use it for a replay buffer or another shared buffer whose schema you
        know before you fill it.

        Each entry is one string on the server. Redis limits a string to 512
        MB by default (``proto-max-bulk-len``). A larger entry raises
        ``redis.exceptions.ResponseError``: raise the limit on the server, or
        split the entry.

        Args:
            schema (dict): maps each key (a str, or a tuple of str for a
                nested entry) to ``(feature_shape, dtype)``, where
                ``feature_shape`` is the shape after the batch dimensions. The
                entry has shape ``[*batch_size, *feature_shape]``.

        Keyword Args:
            batch_size (torch.Size or list of int, optional): the batch size
                of the store. Defaults to ``None``, which gives
                ``torch.Size([])``.
            backend (str, optional): a label for the kind of server, as in
                :class:`TensorDictStore`. Defaults to ``"redis"``.
            host (str, optional): the host name of the server. Defaults to
                ``"localhost"``.
            port (int, optional): the port of the server. Defaults to ``6379``.
            db (int, optional): the database number on the server. Defaults
                to ``0``.
            unix_socket_path (str, optional): the path of a Unix socket, for a
                server on the same machine. If set, ``host`` and ``port`` are
                ignored. Defaults to ``None``.
            prefix (str, optional): the namespace of the keys of the store on
                the server. Defaults to ``"tensordict"``.
            device (torch.device, str or int, optional): the device of the
                tensors that reads return, as in :class:`TensorDictStore`.
                Defaults to ``None``.
            **kwargs: passed to the :class:`TensorDictStore` constructor, such
                as ``client`` or ``td_id``, and options of the Redis client.

        Returns:
            A new :class:`TensorDictStore` whose entries are zeros.

        Examples:
            The example needs a Redis server on ``localhost:6379``. The
            ``obs`` entry takes 21 MB on the server.

            >>> import torch
            >>> from tensordict import TensorDict
            >>> from tensordict.store import TensorDictStore
            >>> store = TensorDictStore.from_schema(  # doctest: +SKIP
            ...     {"obs": ([84, 84, 3], torch.uint8),
            ...      "action": ([4], torch.float32),
            ...      "reward": ([], torch.float32)},
            ...     batch_size=[1000],
            ... )
            >>> store["obs"].shape  # doctest: +SKIP
            torch.Size([1000, 84, 84, 3])
            >>> store[0] = TensorDict(  # doctest: +SKIP
            ...     obs=torch.ones(84, 84, 3, dtype=torch.uint8),
            ...     action=torch.ones(4),
            ...     reward=torch.ones(()),
            ...     batch_size=[],
            ... )
            >>> store["action"][:2]  # doctest: +SKIP
            tensor([[1., 1., 1., 1.],
                    [0., 0., 0., 0.]])
            >>> store.clear_redis()  # doctest: +SKIP
            >>> store.close()  # doctest: +SKIP
        """
        # Read the schema before the store writes anything, so that an invalid
        # entry leaves nothing on the server. A nested key is stored under its
        # dot-joined path.
        entries = []
        for key, (elem_shape, dtype) in schema.items():
            if isinstance(key, tuple):
                key = _KEY_SEP.join(_unravel_key_to_tuple(key))
            elem_size = torch.tensor([], dtype=dtype).element_size()
            entries.append((key, list(elem_shape), dtype, elem_size))

        connect_kwargs = dict(kwargs)
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
        )

        async def _preallocate():
            pipe = out._client.pipeline()
            for key, elem_shape, dtype, elem_size in entries:
                full_shape = list(out._batch_size) + elem_shape
                numel = 1
                for s in full_shape:
                    numel *= s
                total_bytes = numel * elem_size

                key_path = out._full_key_path(key)
                if total_bytes > 0:
                    pipe.setrange(out._data_key(key_path), total_bytes - 1, b"\x00")
                meta = {
                    "shape": json.dumps(full_shape),
                    "dtype": _dtype_to_str(dtype),
                }
                pipe.hset(out._meta_key(key_path), mapping=meta)
                pipe.sadd(out._keys_registry_key, key_path)
                if out._meta_cache is not None:
                    out._meta_cache[key_path] = (full_shape, dtype)
                if out._keys_cache[0] is not None:
                    out._keys_cache[0].add(key_path)
            await pipe.execute()

        out._run_sync(_preallocate())
        # Invalidate keys cache so next keys() call fetches fresh data
        out._keys_cache[0] = None
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
        """Returns a store connected to data that another store wrote to the server.

        One process writes the data, with :meth:`from_tensordict` or with
        ``td[key] = value``; another process, on any machine that reaches the
        server, passes the same ``td_id`` to connect to it. The batch size and
        the device are read from the server.

        If the writer was a tensorclass, the server holds the path of its
        class, and the result is an instance of that class, which must be
        importable. ``tensorclass_cls`` overrides it.

        Keyword Args:
            backend (str, optional): a label for the kind of server, as in
                :class:`TensorDictStore`. Defaults to ``"redis"``.
            host (str, optional): the host name of the server. Defaults to
                ``"localhost"``.
            port (int, optional): the port of the server. Defaults to ``6379``.
            db (int, optional): the database number on the server. Defaults
                to ``0``.
            unix_socket_path (str, optional): the path of a Unix socket, for a
                server on the same machine. If set, ``host`` and ``port`` are
                ignored. Defaults to ``None``.
            prefix (str, optional): the namespace of the keys of the store on
                the server. Defaults to ``"tensordict"``.
            td_id (str): the identifier of the store to connect to. A store
                shows it in its ``repr``, and holds it in the private attribute
                ``_td_id``; there is no public accessor.
            device (torch.device, str or int, optional): the device of the
                tensors that reads return. Defaults to ``None``, which uses the
                device stored on the server.
            tensorclass_cls (type or str, optional): the tensorclass to wrap
                the result in, as a class or as a path ``"module.ClassName"``
                to import. Defaults to ``None``, which uses the class stored on
                the server, if any.
            **kwargs: passed to the :class:`TensorDictStore` constructor, such
                as ``client`` or ``cache_metadata``, and options of the Redis
                client.

        Returns:
            A :class:`TensorDictStore`, or a tensorclass that wraps one.

        Raises:
            KeyError: if the server has no store with this ``td_id`` under
                ``prefix``.

        Examples:
            The example needs a Redis server on ``localhost:6379``. The
            second store could be in another process.

            >>> import torch
            >>> from tensordict.store import TensorDictStore
            >>> writer = TensorDictStore(batch_size=[100])  # doctest: +SKIP
            >>> writer["obs"] = torch.zeros(100, 84)  # doctest: +SKIP
            >>> reader = TensorDictStore.from_store(td_id=writer._td_id)  # doctest: +SKIP
            >>> reader.batch_size, reader["obs"].shape  # doctest: +SKIP
            (torch.Size([100]), torch.Size([100, 84]))
            >>> writer.clear_redis()  # doctest: +SKIP
            >>> writer.close()  # doctest: +SKIP
            >>> reader.close()  # doctest: +SKIP
        """
        # ``client`` and ``cache_metadata`` go to the constructor only.
        redis_kwargs = {
            k: v for k, v in kwargs.items() if k not in ("client", "cache_metadata")
        }
        settings = _client_settings(kwargs.get("client"))
        if settings is not None:
            # the store is on the server of the client
            host, port, db, unix_socket_path, options = settings
            redis_kwargs = {**redis_kwargs, **options}
        raw_bs, raw_dev, raw_tc = _read_stored_metadata(
            _connect_kwargs(host, port, db, unix_socket_path, redis_kwargs),
            prefix,
            td_id,
            ("__batch_size__", "__device__", "__tensorclass__"),
        )

        if raw_bs is None:
            server = unix_socket_path or f"{host}:{port}"
            raise KeyError(
                f"No TensorDictStore with td_id={td_id!r} found at "
                f"{server} db={db} (prefix={prefix!r})."
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
            # Like a load: the data comes from outside this process
            return tc_type._from_tensordict(store, rebuild=False)
        return store

    # ---- Cloning ----

    def _clone(self, recurse: bool = True) -> TensorDictStore:
        if recurse:
            # Deep clone: new UUID, copies all data
            new_td = TensorDictStore(
                backend=self._backend,
                **self._connection_kwargs(),
                prefix=self._namespace,
                batch_size=self._batch_size,
                device=self._device,
            )
            new_td.update(self.to_tensordict())
            return new_td
        else:
            # Shallow clone: same Redis data, new Python wrapper
            out = TensorDictStore._new_nested(
                parent=self,
                key_prefix=self._prefix,
            )
            out._is_shallow_copy = True
            return out

    # ---- Misc required overrides ----

    def _check_new_batch_size(self, new_size: torch.Size) -> None:
        # Check the stored shapes of the tensors instead of fetching them. A
        # non-tensor value stored once fits any batch size; one written per
        # element holds a value per element of the first batch dim.
        prefix = self._prefix + _KEY_SEP if self._prefix else ""
        key_paths = [k for k in self._get_all_keys() if k.startswith(prefix)]
        non_tensor = {}
        shapes = self._run_sync(
            self._aget_metadata_batch(key_paths, non_tensor=non_tensor)
        )
        for key_path, meta in non_tensor.items():
            if meta.get("encoding") == "json_array":
                value = self._run_sync(self._aget_tensor(key_path))
                shapes[key_path] = (value.shape, None)
        for key_path, (shape, _) in shapes.items():
            if torch.Size(shape[: len(new_size)]) != new_size:
                key = unravel_key(tuple(key_path[len(prefix) :].split(_KEY_SEP)))
                raise RuntimeError(
                    f"the entry {key} has shape {torch.Size(shape)} which "
                    f"is incompatible with the batch-size {new_size}."
                )

    # Override: persists the batch size on the server.
    def _change_batch_size(self, new_size: torch.Size) -> None:
        self._batch_size = new_size
        # Nested views and shallow copies share the metadata keys of the store:
        # only the store's own handle persists its batch size.
        if not self._prefix and not self._is_shallow_copy:
            self._run_sync(self._apersist_metadata())

    # Override: a lazy stack in the input is read as one TensorDict first.
    def _update_(
        self,
        input_dict_or_td: dict | TensorDictBase,
        *,
        clone: bool,
        non_blocking: bool,
        keys_to_update: list[tuple[str, ...]] | None,
    ) -> Self:
        # TensorDictBase._update_ splits self into views along the stack dim
        # of a lazy stack in the input, and changes their copies without
        # writing them (#2593).
        if is_tensor_collection(input_dict_or_td) and _has_lazy_stack(input_dict_or_td):
            if keys_to_update is not None:
                # the other keys may have elements that differ in shape
                input_dict_or_td = input_dict_or_td.select(
                    *keys_to_update, strict=False
                )
            input_dict_or_td = input_dict_or_td.to_tensordict()
        # not super(): LazyStackedTensorDictStore[i] shares this method
        return TensorDictBase._update_(
            self,
            input_dict_or_td,
            clone=clone,
            non_blocking=non_blocking,
            keys_to_update=keys_to_update,
        )

    # Override: a TensorDict skeleton of the sub-tensordicts, found in the key
    # registry: no data is read. Each sub-tensordict keeps its own batch size.
    def _empty(self, recurse, batch_size, device, names):
        out = TensorDict(
            {},
            device=self.device if device is NO_DEFAULT else device,
            batch_size=self.batch_size if batch_size is None else batch_size,
            names=self.names if names is None and self._has_names() else names,
        )
        if recurse:
            nested = {
                key[0]
                for key in self.keys(include_nested=True, leaves_only=True)
                if isinstance(key, tuple)
            }
            for key in self.keys():
                if key in nested:
                    out._set_str(
                        key,
                        self._get_str(key).empty(
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

    # Override: lists the key names, without reading the entries.
    def _repr_parts(self) -> list[str]:
        return [
            f"    keys={list(self.keys())}",
            f"    batch_size={self.batch_size}",
            f"    device={self.device}",
            f"    backend={self._backend!r}",
            f"    td_id={self._td_id!r}",
        ]

    # Override: one to_tensordict() read, outside the cache that the base
    # keeps while locked: a locked store would keep a copy of its data, and
    # later calls would miss writes from other processes. inplace=True raises.
    @_as_context_manager()
    def flatten_keys(self, separator=KEY_SEPARATOR, inplace=False, is_leaf=None):
        """Returns a local copy of the store whose nested keys are joined by ``separator``.

        The data is read from the server with :meth:`to_tensordict`, and the
        store is not changed.

        Args:
            separator (str, optional): the string that joins the parts of a
                nested key. Defaults to ``"."``.
            inplace (bool, optional): must be ``False``; ``True`` raises a
                ``ValueError``. Defaults to ``False``.
            is_leaf (Callable, optional): see
                :meth:`~tensordict.TensorDictBase.flatten_keys`. Defaults to
                ``None``.

        See :meth:`~tensordict.TensorDictBase.flatten_keys`.
        """
        if inplace:
            raise ValueError(
                f"Cannot call flatten_keys in_place with a {type(self).__name__}."
            )
        return self.to_tensordict().flatten_keys(separator=separator, is_leaf=is_leaf)

    # Override: one to_tensordict() read, outside the cache that the base
    # keeps while locked: a locked store would keep a copy of its data, and
    # later calls would miss writes from other processes. inplace=True raises.
    @_as_context_manager()
    def unflatten_keys(self, separator=KEY_SEPARATOR, inplace=False):
        if inplace:
            raise ValueError(
                f"Cannot call unflatten_keys in_place with a {type(self).__name__}."
            )
        return self.to_tensordict().unflatten_keys(separator=separator)


_register_tensor_class(TensorDictStore)


# The lazily stacked store lives in tensordict.store._lazy, which imports
# from this module, so it is imported at the end. TensorDictStore uses
# LazyStackedTensorDictStore to store a lazy stack, and imports and
# pickles that name this module must keep working.
from tensordict.store._lazy import (  # noqa: E402, F401
    _LazyStackedStoreKeysView,
    _StoreStackElementView,
    _UPLOAD_CHUNK,
    LazyStackedTensorDictStore,
)

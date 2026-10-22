# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The lazily stacked store: :class:`LazyStackedTensorDictStore` and its element view.

The dense store, :class:`TensorDictStore`, is in :mod:`tensordict.store._store`,
which re-exports the classes of this module.
"""

from __future__ import annotations

import json
import math
import struct
import uuid
import warnings
from numbers import Number
from typing import Any, Callable, Sequence, Type

import torch
from tensordict._base.unsupported import _StoreUnsupported
from tensordict._deprecation import deprecated, deprecation_message, warn_deprecated
from tensordict._indexing import _as_tuple, _getitem_batch_size, convert_ellipsis_to_idx
from tensordict._td import (
    _TensorDictKeysView,
    _unravel_key_to_tuple,
    NO_DEFAULT,
    TensorDict,
)
from tensordict._td_functions import lazy_stack
from tensordict.base import (
    _has_lazy_stack,
    _is_leaf_nontensor,
    _is_tensor_collection,
    _KeyThroughLeafError,
    _register_tensor_class,
    BEST_ATTEMPT_INPLACE,
    is_tensor_collection,
    TensorDictBase,
)
from tensordict.store._connection import (
    _client_settings,
    _connect_kwargs,
    _read_stored_metadata,
    _StoreConnection,
)
from tensordict.store._store import (
    _has_redis,
    _iter_registry_keys,
    _KEY_SEP,
    _rename_plan,
    Self,
    STORE_BACKENDS,
    TensorDictStore,
)
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
    _warn_missing_key,
)
from tensordict.utils import (
    _as_context_manager,
    _LOCK_ERROR,
    _warn_nested_key_in_flat_keys,
    _zip_strict,
    is_non_tensor,
    KEY_SEPARATOR,
    NestedKey,
    unravel_key,
)


def _empty_structure(key_paths: set[str]) -> dict:
    """Returns the sub-tensordicts of a store as nested empty dicts.

    ``key_paths`` are the leaf paths of the key registry (``"b.c"`` for
    ``("b", "c")``), so no data is read.
    """
    structure: dict = {}
    for key_path in sorted(key_paths):
        d = structure
        for part in key_path.split(_KEY_SEP)[:-1]:
            d = d.setdefault(part, {})
    return structure


def _empty_from_key_registry(td, recurse, batch_size, device, names):
    """The ``_empty`` of a ``LazyStackedTensorDictStore`` and of its element views.

    The sub-tensordicts of the result come from the key registry, so no data
    is read.
    """
    return TensorDict(
        _empty_structure(td._get_all_keys()) if recurse else {},
        device=td.device if device is NO_DEFAULT else device,
        batch_size=td.batch_size if batch_size is None else batch_size,
        names=td.names if names is None and td._has_names() else names,
    )


def _writes_existing_leaves(key_paths: set[str], key_path: str, value: Any) -> bool:
    """Returns whether writing ``value`` at ``key_path`` only writes leaves in ``key_paths``.

    A tensordict value is written leaf by leaf under ``key_path``, so each of
    its leaves must be in the key registry; any other value is written as the
    leaf ``key_path``.
    """
    if is_tensor_collection(value):
        prefix = key_path + _KEY_SEP
        return all(
            prefix + _KEY_SEP.join(_unravel_key_to_tuple(sub_key)) in key_paths
            for sub_key in value.keys(include_nested=True, leaves_only=True)
        )
    return key_path in key_paths


class _HeterogeneousKeyError(RuntimeError):
    """Raised when a key whose elements differ in shape or dtype is read or written as one tensor."""

    def __init__(self, key_path: str):
        self.key_path = key_path
        super().__init__(
            f"Failed to stack the elements of the entry {key_path!r}: they differ "
            "in shape or dtype. Read or write each element instead, for example "
            "store[i][key]."
        )


def _element_index(idx, count: int) -> tuple[int, tuple] | None:
    """Splits an index that picks one stack element into that element and the index into it.

    Returns None for an index that picks several elements or none.
    """
    first, rest = (idx[0], idx[1:]) if isinstance(idx, tuple) and idx else (idx, ())
    if isinstance(first, slice):
        picked = range(count)[first]
        if len(picked) == 1:
            # the slice keeps the stack dim, of size 1
            return picked[0], (None, *rest)
    elif isinstance(first, int) and not isinstance(first, bool):
        return first, rest
    return None


def _is_scalar(value) -> bool:
    """Whether ``value`` writes one value at every position it is written to."""
    return isinstance(value, Number) or (
        isinstance(value, torch.Tensor) and value.numel() == 1
    )


def _getranges(pipe, key: str, ranges: list[tuple[int, int]]) -> None:
    """Queues on ``pipe`` the read of the concatenated byte ranges of ``key``."""
    if len(ranges) == 1:
        (offset, length), *_ = ranges
        pipe.getrange(key, offset, offset + length - 1)
    else:
        pipe.eval(_LUA_GETRANGES, 1, key, *[n for r in ranges for n in r])


# ---------------------------------------------------------------------------
# _StoreStackElementView — write-through view for a single stack element
# ---------------------------------------------------------------------------


def _check_shape_fits(key_path: str, shape: list[int], batch_size: torch.Size) -> None:
    """Raises if ``batch_size`` is not the leading dims of the entry ``shape``."""
    if torch.Size(shape[: len(batch_size)]) != batch_size:
        key = unravel_key(tuple(key_path.split(_KEY_SEP)))
        raise RuntimeError(
            f"the entry {key} has shape {torch.Size(shape)} which is "
            f"incompatible with the batch-size {batch_size}."
        )


class _StoreStackElementView(_StoreUnsupported, TensorDictBase):
    """Write-through view of one element in a :class:`LazyStackedTensorDictStore`.

    Returned by ``redis_lazy_stack[int]``.  All reads and writes go through
    the parent's Redis connection so that mutations propagate.  This class is
    private — users interact with it via the normal ``TensorDictBase`` API.
    """

    _td_dim_names = None
    _masking_on_dense_copy = True
    _masked_select_on_dense_copy = True
    # masked_select(inplace=True) cannot change the batch size of the stored
    # members
    _masked_select_inplace_unsupported = True
    # each read fetches the entry from the server into a new tensor
    _reads_return_copies = True
    # and each sub-tensordict is a new TensorDict
    _nested_reads_return_copies = True
    # update({key: self}) writes a copy of the data on the server
    _update_stores_copy_of_self = True
    _leaves_built_on_read = True
    _contiguous_via_to_tensordict = True
    _pin_memory_unsupported = True
    _storage_holds_no_gradients = True
    # td[key] = value writes into an existing entry of the element
    _setitem_key_inplace = BEST_ATTEMPT_INPLACE

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

    # Override: checks the new batch size against the shapes in the metadata,
    # without reading the entries.
    def _batch_size_setter(self, new_batch_size: torch.Size) -> None:
        new_batch_size = torch.Size(new_batch_size)
        if new_batch_size != self._batch_size:
            self._check_new_batch_size(new_batch_size)
            self._batch_size = new_batch_size

    # Override: the shapes of the element in the metadata.
    def _check_new_batch_size(self, new_size: torch.Size) -> None:
        parent = self._parent
        for key_path, meta in parent._run_sync(parent._aget_metas()).items():
            if parent._is_key_homogeneous(meta):
                shape = json.loads(meta["shape"])[1:]
            else:
                shape = json.loads(meta["shapes"])[self._element_idx]
            _check_shape_fits(key_path, shape, new_size)

    # Override: the device can be set, but leaf reads ignore it (a
    # FutureWarning until 0.17, where it raises as on TensorDict).
    @TensorDictBase.device.setter
    def device(self, value):
        warnings.warn(
            "Setting the device of LazyStackedTensorDictStore[i] is applied only "
            "in part: the leaves that it reads stay on the device of the store, "
            "while its nested tensordicts, to_tensordict() and contiguous() use "
            "the new device. Starting with TensorDict 0.17, the default will "
            "change to raising a RuntimeError, as TensorDict does. Call "
            ".to(device) instead.",
            FutureWarning,
            stacklevel=2,
        )
        self._device = torch.device(value) if value is not None else None

    # ---- reads ----

    def _get_str(self, key, default=NO_DEFAULT, **kwargs):
        all_keys = self._get_all_keys()
        key_path = key

        # Nested prefix, in the order of the keys view
        prefix_check = key_path + _KEY_SEP
        nested_keys = sorted(k for k in all_keys if k.startswith(prefix_check))
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
        if not _is_tensor_collection(type(first)):
            raise ValueError(
                f"{key[0]!r} is a {type(first).__name__}, not a tensordict, "
                f"so it has no entry {key[1] if len(key) == 2 else key[1:]!r}."
            )
        return first._get_at_tuple(key[1:], idx, default=default, **kwargs)

    # ---- writes ----

    # Override: a lazy stack in the input is read as one TensorDict first,
    # as on TensorDictStore.
    _update_ = TensorDictStore._update_

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
        # the view of a locked store writes the keys the store has, also
        # without inplace, and raises for a new key (#2404)
        if (
            (self.is_locked or self._parent.is_locked)
            and not ignore_lock
            and inplace is not True
            and (
                (self.is_locked and inplace is False)
                or not _writes_existing_leaves(self._get_all_keys(), key, value)
            )
        ):
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
        if (
            (self.is_locked or self._parent.is_locked)
            and inplace is not True
            and (
                (self.is_locked and inplace is False)
                or not _writes_existing_leaves(self._get_all_keys(), key_path, value)
            )
        ):
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

    # Override: writes each entry of the element back to the server.
    def _setitem_index(self, index, value):
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
        # Read the element entry, patch it locally, write it back. ``key`` is a
        # key path: a nested entry is read as a new TensorDict, so its leaves
        # are written back under their full keys.
        existing = self._get_str(key)
        if not validated and _KEY_SEP in key and isinstance(value, dict):
            # nested keys keep the TensorDict.set_at_ handling of a dict
            # value, which reads it at the batch size of the element
            value = self._validate_value(value, check_shape=False)
        existing[idx] = value
        if is_tensor_collection(existing):
            self._set_str(key, existing, inplace=True, validated=True, ignore_lock=True)
        else:
            self._run_sync(
                self._parent._aset_element_key(self._element_idx, key, existing)
            )
        return self

    def _set_at_tuple(self, key, value, idx, *, validated, non_blocking):
        key = _unravel_key_to_tuple(key)
        return self._set_at_str(
            _KEY_SEP.join(key),
            value,
            idx,
            validated=validated,
            non_blocking=non_blocking,
        )

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

    # Override: a stack element cannot delete the keys it shares with the stack.
    def _del(self, key: tuple[str, ...]) -> None:
        raise RuntimeError(
            "Cannot delete keys from a stack element view. "
            "Delete from the parent LazyStackedTensorDictStore instead."
        )

    # Override: a stack element cannot rename the keys it shares with the stack.
    def _rename_key(self, old_key: NestedKey, new_key: NestedKey) -> None:
        raise RuntimeError(
            "Cannot rename keys on a stack element view. "
            "Rename on the parent LazyStackedTensorDictStore instead."
        )

    def entry_class(self, key: NestedKey) -> type:
        return self._parent.entry_class(key)

    # ---- locking ----

    # Override: a view holds no sub-tensordicts to lock.
    def _lock_sub_tensordicts(self):
        return []

    # ---- materialization ----

    # Override: reads all the leaves of the element in one round trip.
    def _to_tensordict(self, *, retain_none: bool | None) -> TensorDict:
        result_map = self._run_sync(self._parent._abatch_get_element(self._element_idx))
        source: dict = {}
        for kp, tensor in result_map.items():
            parts = kp.split(_KEY_SEP)
            d = source
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = tensor
        return TensorDict(source, batch_size=self._batch_size, device=self._device)

    # Override: deprecated alias of to_tensordict(), removed in 0.17.
    @deprecated(
        "LazyStackedTensorDictStore[i].to_local()",
        removal="0.17",
        replacement="to_tensordict()",
    )
    def to_local(self) -> TensorDict:
        return self.to_tensordict()

    # Override: a TensorDict skeleton, built from the stored keys without reading data.
    _empty = _empty_from_key_registry

    # ---- pickling: materialize, don't try to serialize the view ----

    def __reduce__(self):
        return (TensorDict, (), self.to_tensordict().__getstate__())

    # Override: lists the key names, without reading the entries.
    def _repr_parts(self) -> list[str]:
        return [
            f"    parent_td_id={self._parent._td_id!r}",
            f"    element_idx={self._element_idx}",
            f"    keys={list(self.keys())}",
            f"    batch_size={self.batch_size}",
            f"    device={self.device}",
        ]

    # ---- not-supported stubs (same as parent) ----

    # Override: an element of a store cannot be built from a dict.
    @classmethod
    def from_dict(cls, *args, **kwargs):
        raise NotImplementedError(f"{cls.__name__} cannot be created from a dict.")

    def _clone(self, recurse=True):
        return self.to_tensordict()

    # Override: one batched read of the element, where the generic split reads
    # one key at a time
    def _chunk(self, chunks, dim):
        return self.to_tensordict()._chunk(chunks, dim)

    # Override: works on an in-memory copy, also with inplace=True, which
    # raises in 0.17 as on the parent store.
    @_as_context_manager()
    def flatten_keys(self, separator=KEY_SEPARATOR, inplace=False, is_leaf=None):
        """Returns a local copy of the element whose nested keys are joined by ``separator``.

        The data is read from the server with :meth:`~tensordict.TensorDictBase.to_tensordict`.

        Args:
            separator (str, optional): the string that joins the parts of a
                nested key. Defaults to ``"."``.
            inplace (bool, optional): ignored; a new tensordict is always returned.
                ``True`` is deprecated, and will raise a ``ValueError`` in 0.17,
                as on the parent store. Defaults to ``False``.
            is_leaf (Callable, optional): see
                :meth:`~tensordict.TensorDictBase.flatten_keys`. Defaults to
                ``None``.

        See :meth:`~tensordict.TensorDictBase.flatten_keys`.
        """
        if inplace:
            warn_deprecated(
                "LazyStackedTensorDictStore[i].flatten_keys(inplace=True)",
                removal="0.17",
                replacement="flatten_keys()",
                stacklevel=3,
            )
        return self.to_tensordict().flatten_keys(separator=separator, is_leaf=is_leaf)

    # Override: works on an in-memory copy, as flatten_keys does.
    @_as_context_manager()
    def unflatten_keys(self, separator=KEY_SEPARATOR, inplace=False):
        if inplace:
            warn_deprecated(
                "LazyStackedTensorDictStore[i].unflatten_keys(inplace=True)",
                removal="0.17",
                replacement="unflatten_keys()",
                stacklevel=3,
            )
        return self.to_tensordict().unflatten_keys(separator=separator)


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
        # Sorted as in _StoreTDKeysView. _get_str sorts the paths of a nested
        # tensordict too, so that values() and items() follow this order.
        return _iter_registry_keys(
            sorted(all_keys), self.include_nested, self.leaves_only
        )

    def __contains__(self, key):
        key = unravel_key(key)
        td = self.tensordict
        if isinstance(key, str):
            full_key = key
        else:
            full_key = _KEY_SEP.join(key)
        all_keys = td._get_all_keys()
        prefix_check = full_key + _KEY_SEP
        found = full_key in all_keys or any(
            k.startswith(prefix_check) for k in all_keys
        )
        if found and not isinstance(key, str) and not self.include_nested:
            _warn_nested_key_in_flat_keys(key, td)
        return found

    def __len__(self):
        return sum(1 for _ in self)


class LazyStackedTensorDictStore(_StoreConnection, _StoreUnsupported, TensorDictBase):
    """A lazy stack of tensordicts whose entries live on a Redis-compatible server.

    .. warning::
        ``LazyStackedTensorDictStore`` is experimental: its API can change in a
        minor release without a deprecation cycle.

    :meth:`TensorDictStore.from_tensordict <tensordict.store.TensorDictStore.from_tensordict>`
    returns this class when you pass a :class:`~tensordict.LazyStackedTensorDict`,
    so that the elements of the stack can have entries of different shapes.
    You can also call :meth:`from_lazy_stack`, and :meth:`from_store` to
    reconnect. The class behaves like a lazy stack, but it is a
    :class:`~tensordict.TensorDictBase` subclass, not a
    :class:`~tensordict.LazyStackedTensorDict` subclass.

    Supports `Redis <https://redis.io>`_, `Dragonfly <https://dragonflydb.io>`_,
    `KeyDB <https://docs.keydb.dev>`_, and any other Redis-wire-compatible
    server. The data lifetime, ``close`` and ``clear_redis`` work as for
    :class:`~tensordict.store.TensorDictStore` (see its "Lifecycle and
    limits" section).

    .. warning::
        ``store["nested"]`` is an in-memory copy of the nested entry: writes
        through it, for example ``store["nested"].set_(...)`` or
        ``store["nested"].zero_()``, change the copy only, not the store.
        Write with the full key, ``store.set(("nested", "b"), ...)``, or with
        :meth:`update`.

    ``store[i]`` is a view of element ``i`` that reads and writes on the
    server. On a locked store, the view follows the lock of the store: it
    writes the keys that the store has, with or without ``inplace``, and a
    write of a new key raises the lock error. Unlike ``stack[i]`` of a locked
    lazy stack, the view itself is not locked: its ``is_locked`` is
    ``False``, and the results of operations on it, such as ``store[i] + 1``
    or ``store[i].to("cpu")``, are unlocked tensordicts.

    Each leaf key is one string on the server for the whole stack, with a
    table of offsets when the shapes of the elements differ, so the number of
    keys on the server does not grow with the number of elements.

    The constructor makes an empty stack of ``count`` elements of batch size
    ``inner_batch_size``. To copy a lazy stack to the server, use
    :meth:`from_lazy_stack`.

    Keyword Args:
        backend (str, optional): a label for the kind of server, as in
            :class:`~tensordict.store.TensorDictStore`. Defaults to
            ``"redis"``.
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
        count (int): the number of elements of the stack.
        stack_dim (int, optional): the dimension of the stack in the batch
            size. Defaults to ``0``.
        inner_batch_size (sequence of int): the batch size of each element.
        device (torch.device, str or int, optional): the device of the tensors
            that reads return. Defaults to ``None``: reads return CPU tensors,
            and the store has no device.
        client (redis.asyncio.Redis, optional): an existing client. If given,
            ``host``, ``port``, ``db`` and ``unix_socket_path`` are ignored.
            Defaults to ``None``, which creates a client.
        td_id (str, optional): the identifier of the store on the server.
            Defaults to ``None``, which creates a new UUID. To reconnect to an
            existing store, use :meth:`from_store`.
        cache_metadata (bool, optional): if ``True``, keeps the shape and
            dtype of each entry in the process, to save round-trips to the
            server. Defaults to ``True``.
        **redis_kwargs: passed to ``redis.asyncio.Redis``.

    Examples:
        The example needs a Redis server on ``localhost:6379``.

        >>> import torch
        >>> from tensordict import TensorDict, lazy_stack
        >>> from tensordict.store import TensorDictStore
        >>> local = lazy_stack([TensorDict(x=torch.zeros(3)), TensorDict(x=torch.zeros(5))])
        >>> store = TensorDictStore.from_tensordict(local)  # doctest: +SKIP
        >>> type(store).__name__  # doctest: +SKIP
        'LazyStackedTensorDictStore'
        >>> store[0]["x"].shape, store[1]["x"].shape  # doctest: +SKIP
        (torch.Size([3]), torch.Size([5]))
        >>> store[1] = TensorDict(x=torch.ones(5))  # doctest: +SKIP
        >>> store[1]["x"]  # doctest: +SKIP
        tensor([1., 1., 1., 1., 1.])
        >>> store.is_locked = True  # doctest: +SKIP
        >>> store.is_locked, store[1].is_locked  # doctest: +SKIP
        (True, False)
        >>> store[1]["x"] = torch.zeros(5)  # doctest: +SKIP
        >>> store[1]["y"] = torch.zeros(5)  # doctest: +SKIP
        Traceback (most recent call last):
        ...
        RuntimeError: Cannot modify locked TensorDict. ...
        >>> store.is_locked = False  # doctest: +SKIP
        >>> store.clear_redis()  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
    """

    _td_dim_names = None
    _masking_on_dense_copy = True
    _masked_select_on_dense_copy = True
    # masked_select(inplace=True) cannot change the batch size of the stored
    # members
    _masked_select_inplace_unsupported = True
    # each read fetches the entry from the server into a new tensor
    _reads_return_copies = True
    # and each sub-tensordict is a new TensorDict
    _nested_reads_return_copies = True
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

    _global_key_suffixes = (
        "__type__",
        "__count__",
        "__stack_dim__",
        "__inner_batch_size__",
        "__device__",
    )
    _pickled_attrs = _StoreConnection._pickled_attrs + (
        "_count",
        "_stack_dim",
        "_inner_batch_size",
    )
    # each call reads the key registry from the server, so that keys that
    # another handle adds are visible
    _caches_keys = False

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

        self._backend = backend
        self._cache_metadata = cache_metadata
        self._init_local_state()

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
        self._open_connection(client)

        self._device = torch.device(device) if device is not None else None

        self._run_sync(self._apersist_global_metadata())

    # ---- Redis key helpers ----

    def _idx_key(self, key_path: str) -> str:
        """Redis key for the offset table of a leaf key."""
        return self._redis_key(f"d:{key_path}:idx")

    # Override: a key whose elements differ in shape also has an offset table.
    def _entry_server_keys(self, key_path: str) -> tuple[str, ...]:
        return (
            self._data_key(key_path),
            self._idx_key(key_path),
            self._meta_key(key_path),
        )

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

    # ---- key-level homogeneity check helpers ----

    async def _ais_homogeneous(self, key_paths: list[str]) -> list[bool]:
        """Reads from the metadata which keys hold elements of one shape and dtype."""
        cache = self._meta_cache or {}
        uncached = [kp for kp in key_paths if kp not in cache]
        pipe = self._client.pipeline()
        for kp in uncached:
            pipe.hget(self._meta_key(kp), "homogeneous")
        flags = dict(zip(uncached, await pipe.execute() if uncached else []))
        # as in _is_key_homogeneous, a key without the flag is homogeneous
        return [
            cache[kp][0] is not None if kp in cache else flags[kp] in (None, b"1", "1")
            for kp in key_paths
        ]

    def _init_local_state(self) -> None:
        super()._init_local_state()
        # the layout field of the metadata of each entry, or None
        self._layouts: dict[str, str | None] = {}

    def _stack_first(self, shape: list[int], layout: str | None) -> bool | None:
        """Whether a homogeneous entry of stored ``shape`` holds its elements stack dim first.

        ``from_lazy_stack`` and the element writes store an entry stack dim
        first, and ``set`` and ``update`` in the batch layout; each also
        writes ``layout`` (``"stack"`` or ``"batch"``) in the metadata. The
        shape tells the layouts apart unless it fits both, and the field
        decides then. The shape comes first because a version without the
        field leaves it unchanged when it rewrites the entry. ``None`` when
        neither decides, as for an entry written before the field existed:
        each reader then keeps its behaviour from before, the indexed reads
        and writes taking the batch layout and the element views stack dim
        first.
        """
        if not self._stack_pos:
            # the two layouts are the same
            return None
        prefix = list(shape[: len(self._batch_size)])
        stack = prefix == [self._count, *self._inner_batch_size]
        batch = prefix == list(self._batch_size)
        if stack != batch:
            return stack
        if stack and layout is not None:
            return layout == "stack"
        return None

    def _same_in_both_layouts(self, shape: list[int], idx, scalar: bool) -> bool:
        """Whether ``idx``, in the batch layout, picks the same bytes of a stack-first entry of ``shape`` as if the entry were in the batch layout.

        True when the index picks the same position of each dim up to the
        stack dim either way or, for a write of one ``scalar`` value, the
        same positions in any order: the byte ranges of the batch layout are
        then right.
        """

        def positions(i, size):
            if isinstance(i, slice):
                picked = range(size)[i]
                return picked[::-1] if picked.step < 0 else picked
            if isinstance(i, int) and not isinstance(i, bool) and -size <= i < size:
                return range(i % size, i % size + 1)
            return None

        sd = self._stack_pos
        lead = _as_tuple(idx)[: sd + 1]
        lead = (*lead, *[slice(None)] * (sd + 1 - len(lead)))
        for size, i, j in zip(shape, lead, (lead[sd], *lead[:sd])):
            picked = positions(i, size)
            if picked is None or picked != positions(j, size):
                return False
        return scalar or not any(isinstance(i, slice) for i in lead)

    @property
    def _stack_pos(self) -> int:
        # the position of the stack dim in the batch size, as __init__ inserts it
        if self._stack_dim >= 0:
            return self._stack_dim
        return max(len(self._inner_batch_size) + self._stack_dim, 0)

    def _element_ranges(
        self, shape: list[int], dtype: torch.dtype, layout: str | None, pos: int
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """The shape of element ``pos`` of a homogeneous entry, and the byte ranges that hold it."""
        if self._stack_first(shape, layout) is False:
            # batch layout: the element is one block of the dims after the
            # stack dim in each row of the dims before it
            sd = self._stack_pos
            block = self._row_bytes(shape[sd + 1 :], dtype)
            rows = range(math.prod(shape[:sd]))
            return shape[:sd] + shape[sd + 1 :], [
                ((row * self._count + pos) * block, block) for row in rows
            ]
        row_bytes = self._row_bytes(shape[1:], dtype)
        return shape[1:], [(pos * row_bytes, row_bytes)]

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
            "layout": "stack",
        }
        pipe = self._client.pipeline()
        pipe.hset(self._meta_key(key_path), mapping=meta)
        pipe.sadd(self._keys_registry_key, key_path)
        await pipe.execute()

        self._layouts[key_path] = "stack"
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
            "layout": "stack",
        }
        pipe = self._client.pipeline()
        pipe.set(self._idx_key(key_path), offset_bytes)
        pipe.hset(self._meta_key(key_path), mapping=meta)
        pipe.sadd(self._keys_registry_key, key_path)
        await pipe.execute()

        self._layouts[key_path] = "stack"
        if self._meta_cache is not None:
            self._meta_cache[key_path] = (None, dtype)

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
                elem_shape, ranges = self._element_ranges(
                    json.loads(meta["shape"]), dtype, meta.get("layout"), pos
                )
                _getranges(pipe, self._data_key(kp), ranges)
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
                elem_shape, ranges = self._element_ranges(
                    json.loads(meta["shape"]), dtype, meta.get("layout"), pos
                )
                _getranges(pipe, self._data_key(kp), ranges)
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
            # Ensure the full blob is allocated. This comes first: the byte it
            # writes is the last byte of the last element.
            pipe.setrange(
                self._data_key(key_path),
                self._count * row_bytes - 1,
                b"\x00",
            )
            # Write the element at the right offset (zero-fill for other elements)
            pipe.setrange(self._data_key(key_path), pos * row_bytes, raw_bytes)
            pipe.hset(
                self._meta_key(key_path),
                mapping={
                    "shape": json.dumps(full_shape),
                    "dtype": str(value.dtype),
                    "homogeneous": "1",
                    "layout": "stack",
                },
            )
            await pipe.execute()
            self._layouts[key_path] = "stack"
            if self._meta_cache is not None:
                self._meta_cache[key_path] = (full_shape, value.dtype)
            return

        dtype = _str_to_dtype(raw_meta["dtype"])
        homogeneous = self._is_key_homogeneous(raw_meta)

        if homogeneous:
            elem_shape, ranges = self._element_ranges(
                json.loads(raw_meta["shape"]), dtype, raw_meta.get("layout"), pos
            )
            shape = [self._count] + elem_shape
        else:
            shape = [self._count] + json.loads(raw_meta["shapes"])[pos]
        value = _prepare_indexed_value(value, shape, dtype, pos)
        raw_bytes = _tensor_to_bytes(value)

        if homogeneous and len(ranges) > 1:
            argv: list = []
            for i, (offset, length) in enumerate(ranges):
                argv += [offset, raw_bytes[i * length : (i + 1) * length]]
            await self._client.eval(_LUA_SETRANGES, 1, self._data_key(key_path), *argv)
        elif homogeneous:
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
        """Batch fetch indexed slices of multiple keys using GETRANGE.

        ``idx`` is in the batch layout. An entry stored in it (written by
        ``set``) is read by byte ranges; an entry stored stack dim first
        (written by ``from_lazy_stack``), when the stack dim is not 0, is read
        whole and moved into the batch layout, unless ``idx`` picks the same
        bytes in both layouts. See ``_stack_first``.
        """
        if not key_paths:
            return {}

        meta_map = await self._aget_metadata_batch(key_paths, layouts=self._layouts)

        result: dict[str, torch.Tensor] = {}
        heterogeneous = [kp for kp in key_paths if meta_map[kp][0] is None]
        if heterogeneous:
            # one element of these keys is read with its own shape, as on a
            # lazy stack; several elements cannot be stacked
            element = _element_index(idx, self._count) if self._stack_dim == 0 else None
            if element is None:
                raise _HeterogeneousKeyError(heterogeneous[0])
            pos, rest = element
            elements = await self._abatch_get_element_keys(pos, heterogeneous)
            result.update({kp: tensor[rest] for kp, tensor in elements.items()})
            key_paths = [kp for kp in key_paths if kp not in elements]

        pipe = self._client.pipeline()
        plan: list[tuple[str, list[int], torch.dtype, object, bool]] = []
        fallback_kps: list[str] = []

        scattered = _is_scattered_index(idx)

        stack_first = {
            kp
            for kp in key_paths
            if self._stack_first(meta_map[kp][0], self._layouts.get(kp))
            and not self._same_in_both_layouts(meta_map[kp][0], idx, False)
        }
        for kp in key_paths:
            full_shape, dtype = meta_map[kp]
            if kp in stack_first:
                fallback_kps.append(kp)
                continue
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
            tensor = _bytes_to_tensor(raw_results[ri], full_shape, dtype)
            if kp in stack_first:
                tensor = tensor.movedim(0, self._stack_pos)
            tensor = tensor[_as_tuple(idx)]
            ri += 1
            if self._device is not None:
                tensor = tensor.to(self._device)
            result[kp] = tensor

        return result

    # ---- element access (writes) ----

    async def _aset_element(self, element_idx: int, value_td: TensorDictBase):
        """Write the keys of ``value_td`` for a single stack element via pipelined SETRANGE."""
        value_paths = {
            _KEY_SEP.join(_unravel_key_to_tuple(key))
            for key in value_td.keys(
                include_nested=True, leaves_only=True, is_leaf=_is_leaf_nontensor
            )
        }
        missing = value_paths.difference(await self._aget_all_keys())
        if missing:
            # as the writes at other indices, an element cannot take a new key
            raise KeyError(f"key {min(missing)} not found in {type(self).__name__}")
        all_keys = sorted(value_paths)
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
        """Batch-write indexed slices using SETRANGE / Lua.

        Each index is in the batch layout. An entry stored in it (written by
        ``set``) is written by byte ranges; an entry stored stack dim first
        (written by ``from_lazy_stack``), when the stack dim is not 0, by a
        read-modify-write of the whole entry, which keeps its layout, unless
        the write gives the same bytes in both layouts. See ``_stack_first``.
        """
        if not items:
            return

        key_paths = list(items.keys())
        meta_map = await self._aget_metadata_batch(key_paths, layouts=self._layouts)

        heterogeneous = [kp for kp in key_paths if meta_map[kp][0] is None]
        elements = {}
        for kp in heterogeneous:
            # one element of these keys is written with its own shape, as on
            # a lazy stack; several elements cannot be written as one tensor
            elements[kp] = (
                _element_index(items[kp][1], self._count)
                if self._stack_dim == 0
                else None
            )
            if elements[kp] is None:
                raise _HeterogeneousKeyError(kp)
        for kp, (pos, rest) in elements.items():
            existing = (await self._abatch_get_element_keys(pos, [kp]))[kp]
            existing[rest] = items[kp][0]
            await self._aset_element_key(pos, kp, existing)
        key_paths = [kp for kp in key_paths if kp not in elements]

        direct_kps: list[str] = []
        lua_kps: list[str] = []
        rmw_kps: list[str] = []

        stack_first = [
            kp
            for kp in key_paths
            if self._stack_first(meta_map[kp][0], self._layouts.get(kp))
            and not self._same_in_both_layouts(
                meta_map[kp][0], items[kp][1], _is_scalar(items[kp][0])
            )
        ]
        for kp in key_paths:
            if kp in stack_first:
                continue
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

        # an entry stored stack dim first is read whole in the same pipeline,
        # written in the batch layout and stored back in its own
        for kp in stack_first:
            pipe.get(self._data_key(kp))
            has_pipe_cmds = True

        if has_pipe_cmds:
            raw = await pipe.execute()

        if stack_first:
            pipe = self._client.pipeline()
            for kp, data in zip(stack_first, raw[-len(stack_first) :]):
                shape, dtype = meta_map[kp]
                value, idx = items[kp]
                tensor = _bytes_to_tensor(data, shape, dtype)
                tensor.movedim(0, self._stack_pos)[_as_tuple(idx)] = value
                pipe.set(self._data_key(kp), _tensor_to_bytes(tensor))
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

    # Override: checks the new batch size against the shapes in the metadata,
    # without reading the entries, like a stack element.
    _batch_size_setter = _StoreStackElementView._batch_size_setter

    # Override: the shapes of the elements in the metadata; a dim along which
    # the elements differ fits no batch size.
    def _check_new_batch_size(self, new_size: torch.Size) -> None:
        for key_path, meta in self._run_sync(self._aget_metas()).items():
            if self._is_key_homogeneous(meta):
                shape = json.loads(meta["shape"])[1:]
            else:
                shape = [
                    sizes[0] if len(set(sizes)) == 1 else -1
                    for sizes in zip(*json.loads(meta["shapes"]))
                ]
            shape.insert(self._stack_dim, self._count)
            _check_shape_fits(key_path, shape, new_size)

    async def _aget_metas(self) -> dict[str, dict]:
        """Reads the metadata of every entry, without its data."""
        key_paths = sorted(await self._aget_all_keys())
        if not key_paths:
            return {}
        pipe = self._client.pipeline()
        for kp in key_paths:
            pipe.hgetall(self._meta_key(kp))
        raw_metas = await pipe.execute()
        return {kp: _decode_meta(meta) for kp, meta in zip(key_paths, raw_metas)}

    # Override: the device can be set; reads move the entries to it.
    @TensorDictBase.device.setter
    def device(self, value):
        self._device = torch.device(value) if value is not None else None

    # ---- Key access ----

    def _index_tensordict(self, index, new_batch_size=None, names=None):
        """Eagerly fetch all leaf tensors for the given index in one pipeline."""
        batch_size = self.batch_size
        if new_batch_size is None:
            new_batch_size = _getitem_batch_size(batch_size, index)
        if names is None:
            names = self._get_names_idx(index)

        all_keys = sorted(self._get_all_keys())
        try:
            result_map = self._run_sync(self._abatch_get_at(all_keys, index))
        except _HeterogeneousKeyError:
            # as on a lazy stack, a slice of the stack dim gives the lazy
            # stack of the elements it picks, each with its own shapes
            first, rest = (
                (index[0], index[1:])
                if isinstance(index, tuple) and index
                else (index, ())
            )
            if (
                self._stack_dim
                or not isinstance(first, slice)
                or not range(self._count)[first]
                or not all(i is None or isinstance(i, (int, slice)) for i in rest)
            ):
                raise
            return lazy_stack(
                [self._index_tensordict((i, *rest)) for i in range(self._count)[first]]
            )

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

    # Override: an integer on the stack dim gives a view that writes through.
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

    # Override: writes the indexed entries on the server in one batch
    # (SETRANGE), and an element at an int index along the stack dim.
    def _setitem_index(self, index, value):
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

        # Check nested, in the order of the keys view
        prefix_check = key_path + _KEY_SEP
        nested_keys = sorted(k for k in all_keys if k.startswith(prefix_check))
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

    def _get_tuple(self, key, default=NO_DEFAULT, **kwargs):
        # Override: a nested entry is read alone, not with the whole
        # sub-tensordict of key[0], whose other entries may have elements of
        # different shapes
        if len(key) > 1:
            try:
                return self._get_str(_KEY_SEP.join(key), NO_DEFAULT, **kwargs)
            except KeyError:
                all_keys = self._get_all_keys()
                for i in range(1, len(key)):
                    if _KEY_SEP.join(key[:i]) in all_keys:
                        rest = key[i] if len(key) == i + 1 else key[i:]
                        raise _KeyThroughLeafError(
                            f"{key[i - 1]!r} is a Tensor, not a tensordict, "
                            f"so it has no entry {rest!r}."
                        )
                if default is not NO_DEFAULT:
                    return default
                raise
        return super()._get_tuple(key, default, **kwargs)

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
        if (
            self.is_locked
            and not ignore_lock
            and inplace is not True
            and (
                inplace is False
                or not _writes_existing_leaves(self._get_all_keys(), key, value)
            )
        ):
            raise RuntimeError(_LOCK_ERROR)

        key_path = key

        if is_tensor_collection(value):
            for sub_key in value.keys(include_nested=True, leaves_only=True):
                sub_tuple = _unravel_key_to_tuple(sub_key)
                full_kp = key_path + _KEY_SEP + _KEY_SEP.join(sub_tuple)
                tensor = value.get(sub_key)
                self._run_sync(self._aset_full_tensor(full_kp, tensor, inplace))
            return self

        if isinstance(value, torch.Tensor):
            self._run_sync(self._aset_full_tensor(key_path, value, inplace))
            return self

        try:
            value = torch.as_tensor(value)
            self._run_sync(self._aset_full_tensor(key_path, value, inplace))
        except (ValueError, TypeError):
            raise TypeError(
                f"LazyStackedTensorDictStore only supports tensor values, got {type(value)}"
            )
        return self

    async def _aset_full_tensor(
        self, key_path: str, tensor: torch.Tensor, inplace: bool = False
    ):
        """Set a full (stacked) tensor for a key."""
        if inplace is True and not (await self._ais_homogeneous([key_path]))[0]:
            # as on a lazy stack, set_ cannot write one tensor to elements
            # of different shapes
            raise _HeterogeneousKeyError(key_path)
        data = _tensor_to_bytes(tensor)
        shape = list(tensor.shape)
        dtype = tensor.dtype
        meta = {
            "shape": json.dumps(shape),
            "dtype": _dtype_to_str(dtype),
            "homogeneous": "1",
            "layout": "batch",
        }
        pipe = self._client.pipeline()
        pipe.set(self._data_key(key_path), data)
        pipe.hset(self._meta_key(key_path), mapping=meta)
        pipe.sadd(self._keys_registry_key, key_path)
        await pipe.execute()
        self._layouts[key_path] = "batch"
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
        if (
            self.is_locked
            and inplace is not True
            and (
                inplace is False
                or not _writes_existing_leaves(self._get_all_keys(), key_path, value)
            )
        ):
            raise RuntimeError(_LOCK_ERROR)
        if isinstance(value, torch.Tensor):
            self._run_sync(self._aset_full_tensor(key_path, value, inplace))
        elif is_tensor_collection(value):
            for sub_key in value.keys(include_nested=True, leaves_only=True):
                sub_tuple = _unravel_key_to_tuple(sub_key)
                full_kp = key_path + _KEY_SEP + _KEY_SEP.join(sub_tuple)
                self._run_sync(
                    self._aset_full_tensor(full_kp, value.get(sub_key), inplace)
                )
        else:
            self._run_sync(
                self._aset_full_tensor(key_path, torch.as_tensor(value), inplace)
            )
        return self

    def _set_at_items(self, key_path: str, value, idx) -> dict:
        """Return the items of :meth:`_abatch_set_at` that write ``value`` at ``idx`` under ``key_path``.

        If the store holds a tensordict at ``key_path``, the leaves of a
        tensordict value are written under their full keys, and an Ellipsis in
        ``idx`` stands for the batch dims, as on a tensordict. ``_abatch_set_at``
        reads the metadata of all the keys first, so a leaf that the store lacks
        raises a ``KeyError`` before anything is written.
        """
        prefix = key_path + _KEY_SEP
        if (
            is_tensor_collection(value)
            and not is_non_tensor(value)
            and any(k.startswith(prefix) for k in self._get_all_keys())
        ):
            if isinstance(idx, tuple) and any(i is Ellipsis for i in idx):
                idx = convert_ellipsis_to_idx(idx, self.batch_size)
            return {
                prefix + _KEY_SEP.join(_unravel_key_to_tuple(subkey)): (leaf, idx)
                for subkey, leaf in value.items(True, True, is_leaf=_is_leaf_nontensor)
            }
        return {key_path: (value, idx)}

    def _set_at_str(self, key, value, idx, *, validated, non_blocking):
        items = self._set_at_items(key, value, idx)
        self._run_sync(self._abatch_set_at(items))
        return self

    def _set_at_tuple(self, key, value, idx, *, validated, non_blocking):
        key = _unravel_key_to_tuple(key)
        if len(key) == 1:
            return self._set_at_str(
                key[0], value, idx, validated=validated, non_blocking=non_blocking
            )
        key_path = _KEY_SEP.join(key)
        items = self._set_at_items(key_path, value, idx)
        self._run_sync(self._abatch_set_at(items))
        return self

    def _get_at_str(self, key, idx, default=NO_DEFAULT, **kwargs):
        try:
            result = self._run_sync(self._abatch_get_at([key], idx))
        except KeyError:
            # not a leaf: a sub-tensordict, or a missing key
            value = self._get_str(key, default, **kwargs)
            return value if value is default else value[idx]
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
        try:
            result = self._run_sync(self._abatch_get_at([key_path], idx))
        except KeyError:
            # not a leaf: a sub-tensordict, a path through a tensor, or a
            # missing key
            value = self._get_tuple(key, default, **kwargs)
            return value if value is default else value[idx]
        tensor = result.get(key_path)
        if tensor is None:
            if default is not NO_DEFAULT:
                return default
            raise KeyError(f"key {key} not found in {type(self).__name__}")
        return tensor

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

    # Override: deletes the keys on the server.
    def _del(self, key: tuple[str, ...]) -> None:
        key_path = _KEY_SEP.join(key)

        all_keys = self._get_all_keys()
        found = key_path in all_keys
        if found:
            self._run_sync(self._adel_key(key_path))
        prefix_check = key_path + _KEY_SEP
        for k in all_keys:
            if k.startswith(prefix_check):
                found = True
                self._run_sync(self._adel_key(k))
        if not found:
            # stacklevel 4: skip del_ and its _lock_blocked wrapper
            _warn_missing_key(
                self, "del_", key[0] if len(key) == 1 else key, stacklevel=4
            )

    # Override: renames the keys on the server, without reading the data.
    def _rename_key(self, old_key: NestedKey, new_key: NestedKey) -> None:
        if isinstance(old_key, str):
            old_path = old_key
        else:
            old_path = _KEY_SEP.join(_unravel_key_to_tuple(old_key))
        if isinstance(new_key, str):
            new_path = new_key
        else:
            new_path = _KEY_SEP.join(_unravel_key_to_tuple(new_key))

        async def _arename():
            all_keys = await self._aget_all_keys()
            plan = _rename_plan(all_keys, old_path, new_path, old_key, new_key)
            if plan is None:
                return None
            renames, deletes = plan
            moves = []
            for src_kp, dst_kp in renames:
                moves += [
                    (self._data_key(src_kp), self._data_key(dst_kp)),
                    (self._meta_key(src_kp), self._meta_key(dst_kp)),
                    (self._idx_key(src_kp), self._idx_key(dst_kp)),
                ]
            # Only leaves whose elements differ in shape or dtype have an
            # offset table, and empty leaves may have no data. RENAME fails
            # on a missing key, so a key that the leaf lacks is deleted at
            # the destination instead.
            pipe = self._client.pipeline()
            for src, _ in moves:
                pipe.exists(src)
            found = await pipe.execute()
            pipe = self._client.pipeline()
            # the entry replaces what is stored at new_path, as in TensorDict
            for kp in deletes:
                pipe.delete(self._data_key(kp))
                pipe.delete(self._idx_key(kp))
                pipe.delete(self._meta_key(kp))
                pipe.srem(self._keys_registry_key, kp)
            for (src, dst), src_found in zip(moves, found):
                if src_found:
                    pipe.rename(src, dst)
                else:
                    pipe.delete(dst)
            for src_kp, dst_kp in renames:
                pipe.srem(self._keys_registry_key, src_kp)
                pipe.sadd(self._keys_registry_key, dst_kp)
            await pipe.execute()
            return plan

        plan = self._run_sync(_arename())
        if plan is not None and self._meta_cache is not None:
            # the old paths are gone, and the leaves at new_path may have been
            # replaced
            renames, deletes = plan
            for kp in deletes:
                self._meta_cache.pop(kp, None)
            for src_kp, dst_kp in renames:
                self._meta_cache.pop(src_kp, None)
                self._meta_cache.pop(dst_kp, None)
        if plan is None:
            # stacklevel 3: skip rename_key_
            _warn_missing_key(self, "rename_key_", old_key, stacklevel=3)

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
            # get() builds a sub-tensordict as a TensorDict
            return TensorDict
        raise KeyError(f"key {key} not found in {type(self).__name__}")

    # ---- Locking ----

    # Override: the stack holds no sub-tensordicts to lock.
    def _lock_sub_tensordicts(self):
        return []

    # ---- Materialization ----

    # Override: deprecated alias of to_tensordict(), removed in 0.17.
    @deprecated(
        "LazyStackedTensorDictStore.to_local()",
        removal="0.17",
        replacement="to_tensordict()",
    )
    def to_local(self) -> TensorDict:
        """Returns a local copy of the data of the store.

        .. deprecated:: 0.15
            Use :meth:`to_tensordict` instead.
        """
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
        """Returns a new store that holds a copy of the data of a lazy stack.

        The data is sent in chunks of 10,000 elements, so that the full stack
        is not built in memory at once.

        Args:
            lazy_td (LazyStackedTensorDict): the lazy stack to copy to the
                server.

        Keyword Args:
            backend (str, optional): a label for the kind of server, as in
                :class:`~tensordict.store.TensorDictStore`. Defaults to
                ``"redis"``.
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
                the device of ``lazy_td``.
            **kwargs: passed to the constructor, such as ``client`` or
                ``td_id``, and options of the Redis client.

        Raises:
            TypeError: if ``lazy_td`` is not a
                :class:`~tensordict.LazyStackedTensorDict`.

        Examples:
            The example needs a Redis server on ``localhost:6379``.

            >>> import torch
            >>> from tensordict import TensorDict, lazy_stack
            >>> from tensordict.store import LazyStackedTensorDictStore
            >>> local = lazy_stack([TensorDict(x=torch.zeros(2)), TensorDict(x=torch.zeros(4))])
            >>> store = LazyStackedTensorDictStore.from_lazy_stack(local)  # doctest: +SKIP
            >>> store.batch_size, store[1]["x"].shape  # doctest: +SKIP
            (torch.Size([2]), torch.Size([4]))
            >>> store.clear_redis()  # doctest: +SKIP
            >>> store.close()  # doctest: +SKIP
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
        """Returns a store connected to a lazy stack that another store wrote to the server.

        Reads the number of elements, the stack dimension, the batch size of
        the elements and the device from the server.

        Keyword Args:
            backend (str, optional): a label for the kind of server, as in
                :class:`~tensordict.store.TensorDictStore`. Defaults to
                ``"redis"``.
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
            **kwargs: passed to the constructor, such as ``client`` or
                ``cache_metadata``, and options of the Redis client.

        Returns:
            A :class:`LazyStackedTensorDictStore` connected to the existing data.

        Raises:
            KeyError: if the server has no lazy stack with this ``td_id``.

        Examples:
            The example needs a Redis server on ``localhost:6379``. The
            second store could be in another process.

            >>> import torch
            >>> from tensordict import TensorDict, lazy_stack
            >>> from tensordict.store import LazyStackedTensorDictStore
            >>> local = lazy_stack([TensorDict(x=torch.zeros(2)), TensorDict(x=torch.zeros(4))])
            >>> writer = LazyStackedTensorDictStore.from_lazy_stack(local)  # doctest: +SKIP
            >>> reader = LazyStackedTensorDictStore.from_store(td_id=writer._td_id)  # doctest: +SKIP
            >>> reader[1]["x"].shape  # doctest: +SKIP
            torch.Size([4])
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
        raw_type, raw_count, raw_sd, raw_ibs, raw_dev = _read_stored_metadata(
            _connect_kwargs(host, port, db, unix_socket_path, redis_kwargs),
            prefix,
            td_id,
            (
                "__type__",
                "__count__",
                "__stack_dim__",
                "__inner_batch_size__",
                "__device__",
            ),
        )

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

    # Override: raises; a lazy store is built with from_lazy_stack.
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
        """Raises a ``NotImplementedError``: build the store with :meth:`from_lazy_stack`.

        The arguments are those of :meth:`~tensordict.TensorDictBase.from_dict`,
        and are not used.
        """
        raise NotImplementedError(
            f"{cls.__name__}.from_dict is not supported. "
            "Use LazyStackedTensorDictStore.from_lazy_stack(lazy_td, ...) instead."
        )

    # ---- Cloning ----

    def _clone(self, recurse: bool = True) -> LazyStackedTensorDictStore:
        if recurse:
            new_td = LazyStackedTensorDictStore(
                **self._connection_kwargs(),
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
            **self._connection_kwargs(),
            prefix=self._namespace,
            count=self._count,
            stack_dim=self._stack_dim,
            inner_batch_size=self._inner_batch_size,
            device=self._device,
            td_id=self._td_id,
        )

    # ---- Misc required overrides ----

    def _inplace_on_copies(self, method: Callable, *args, **kwargs) -> Self:
        # Override: a key whose elements differ in shape cannot be read as one
        # tensor. The method then runs on each element, as on a lazy stack.
        try:
            return super()._inplace_on_copies(method, *args, **kwargs)
        except _HeterogeneousKeyError as err:
            error = err

        # the public name of a hook such as _masked_fill_
        name = method.__name__.lstrip("_")

        def split(arg):
            if is_tensor_collection(arg) and arg.batch_size == self.batch_size:
                return arg.unbind(self._stack_dim)
            if (
                name == "masked_fill_"
                and isinstance(arg, torch.Tensor)
                and arg.shape == self.batch_size
            ):
                # the mask: each element is filled with its slice, as on a
                # lazy stack
                return arg.unbind(self._stack_dim)
            if is_tensor_collection(arg) or (
                isinstance(arg, torch.Tensor) and arg.ndim
            ):
                raise RuntimeError(
                    f"{name} cannot split this argument over the "
                    f"elements of the entry {error.key_path!r}, which differ in "
                    "shape. Call it on each element instead, for example "
                    f"store[i].{name}(...)."
                ) from error
            return [arg] * self._count

        args = [split(arg) for arg in args]
        kwargs = {key: split(arg) for key, arg in kwargs.items()}
        for i in range(self._count):
            _StoreStackElementView(self, i)._inplace_on_copies(
                method,
                *[arg[i] for arg in args],
                **{key: arg[i] for key, arg in kwargs.items()},
            )
        return self

    # Override: a sub-tensordict with a leaf whose elements differ in shape is
    # written leaf by leaf.
    def _update_entries(
        self,
        input_dict_or_td: dict | TensorDictBase,
        *,
        clone: bool,
        inplace: bool,
        non_blocking: bool,
        keys_to_update: list[NestedKey] | None,
        is_leaf: Callable[[Type], bool],
        update_batch_size: bool,
        ignore_lock: bool,
    ) -> Self:
        # TensorDictBase._update_entries reads each sub-tensordict of self
        # that the input writes into, and a sub-tensordict with a leaf whose
        # elements differ in shape cannot be read as one tensordict. Each leaf
        # of the input is then written under its full key, which merges the
        # sub-tensordicts as update does.
        try:
            return super()._update_entries(
                input_dict_or_td,
                clone=clone,
                inplace=inplace,
                non_blocking=non_blocking,
                keys_to_update=keys_to_update,
                is_leaf=is_leaf,
                update_batch_size=update_batch_size,
                ignore_lock=ignore_lock,
            )
        except _HeterogeneousKeyError:
            if is_leaf is not _is_leaf_nontensor or update_batch_size or ignore_lock:
                raise
        if keys_to_update is not None:
            keys_to_update = [_unravel_key_to_tuple(key) for key in keys_to_update]
        all_keys = self._get_all_keys()
        if not is_tensor_collection(input_dict_or_td):
            input_dict_or_td = TensorDict.from_dict(
                input_dict_or_td, batch_dims=self.batch_dims
            )
        for key, value in input_dict_or_td.items(True, True):
            key = _unravel_key_to_tuple(key)
            if keys_to_update is not None and not any(
                key[: len(k)] == k for k in keys_to_update
            ):
                continue
            if clone:
                value = value.clone()
            if inplace and _KEY_SEP.join(key) in all_keys:
                self.set_(key, value, non_blocking=non_blocking)
            else:
                # a locked store raises here, as update does
                self.set(key, value, non_blocking=non_blocking)
        return self

    # Override: a key whose elements differ in shape, or a lazy stack in the
    # input, is written leaf by leaf.
    def _update_(
        self,
        input_dict_or_td: dict | TensorDictBase,
        *,
        clone: bool,
        non_blocking: bool,
        keys_to_update: list[tuple[str, ...]] | None,
    ) -> Self:
        # Without keys_to_update, TensorDictBase._update_ reads every entry of
        # self, and a key whose elements differ in shape cannot be read as one
        # tensor, and with keys_to_update it reads the whole sub-tensordicts
        # of self. A lazy stack in the input splits self into element views
        # there, whose copies it changes without writing them (#2593). Each
        # leaf of the input is then written alone, and the leaves that self
        # lacks are handled as in TensorDictBase._update_.
        if not is_tensor_collection(input_dict_or_td):
            input_dict_or_td = TensorDict.from_dict(
                input_dict_or_td, batch_dims=self.batch_dims
            )
        lazy = _has_lazy_stack(input_dict_or_td)
        if not lazy:
            try:
                return super()._update_(
                    input_dict_or_td,
                    clone=clone,
                    non_blocking=non_blocking,
                    keys_to_update=keys_to_update,
                )
            except _HeterogeneousKeyError:
                pass
        all_keys = self._get_all_keys()
        missing = None
        written = False
        elements = None
        for key in input_dict_or_td.keys(True, True):
            key = _unravel_key_to_tuple(key)
            if keys_to_update is not None and not any(
                key[: len(k)] == k for k in keys_to_update
            ):
                continue
            key_path = _KEY_SEP.join(key)
            if key_path in all_keys:
                if lazy and not self._run_sync(self._ais_homogeneous([key_path]))[0]:
                    # the elements of the input, which a lazy stack does not
                    # stack into one tensor, are written one by one
                    if elements is None:
                        elements = input_dict_or_td.unbind(self._stack_dim)
                    for i, element in _zip_strict(range(self._count), elements):
                        _StoreStackElementView(self, i).set_(
                            key, element.get(key), non_blocking=non_blocking
                        )
                else:
                    self.set_(key, input_dict_or_td.get(key), non_blocking=non_blocking)
                written = True
            elif keys_to_update is not None:
                raise KeyError(f"The key {key} was not found in the dest tensordict.")
            elif missing is None:
                missing = key
        if missing is not None and not written:
            raise KeyError(f"The key {missing} was not found in the dest tensordict.")
        return self

    # Override: a TensorDict skeleton, built from the stored keys without reading data.
    _empty = _empty_from_key_registry

    # Override: until 0.17, a mask that is not batch-shaped broadcasts from
    # the right, as before 0.15.
    def _masked_fill_(self, mask, value):
        if mask.shape == self.batch_size or not mask.ndim:
            return super()._masked_fill_(mask, value)
        # an entry with as many dims as the mask is filled the same way when the
        # mask is aligned with its leading dims; the warning comes before any
        # write
        entries = [
            (key, self.get(key))
            for key in self.keys(include_nested=True, leaves_only=True)
        ]
        if any(tensor.ndim != mask.ndim for _, tensor in entries):
            warnings.warn(
                deprecation_message(
                    f"Calling {type(self).__name__}.masked_fill_ with a mask of "
                    "shape other than the batch size broadcasts the mask from the right "
                    "against each entry. This",
                    removal="0.17",
                )
                + " From then on, the mask is aligned with the leading batch dims, "
                "as in TensorDict.masked_fill_. Expand the mask to the shape of "
                "the entries to keep today's result.",
                FutureWarning,
                stacklevel=3,
            )
        for key, tensor in entries:
            self.set_(key, tensor.masked_fill(mask, value))
        return self

    # Override: lists the key names, without reading the entries.
    def _repr_parts(self) -> list[str]:
        return [
            f"    keys={list(self.keys())}",
            f"    batch_size={self.batch_size}",
            f"    count={self._count}",
            f"    stack_dim={self._stack_dim}",
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


_register_tensor_class(LazyStackedTensorDictStore)

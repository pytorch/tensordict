# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import collections
import enum
import gc
import importlib
import importlib.util

# JSON backend is now handled by _utils_key_json.json_dumps
import json
import os.path
import sys
import warnings
import weakref
from collections.abc import Iterable, Mapping, MutableMapping
from concurrent.futures import Future, ThreadPoolExecutor, wait
from functools import wraps
from pathlib import Path
from textwrap import indent
from threading import local
from types import EllipsisType
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    overload,
    Sequence,
    Tuple,
    Type,
    TYPE_CHECKING,
    TypeVar,
)

import numpy as np
import torch
from tensordict._base.unsupported import _unsupported
from tensordict._contextlib import LAST_OP_MAPS
from tensordict._deprecation import (
    deprecated,
    deprecation_message,
    warn_deprecated,
    warn_deprecated_env_var,
)
from tensordict._indexing import (
    _entry_index,
    _expand_batch_ellipsis,
    _getitem_batch_size,
    _getitem_names,
    _is_new_dim_index,
    _mask_to_indices,
    _read_element,
    convert_ellipsis_to_idx,
)
from tensordict._nestedkey import NestedKey
from tensordict._tensorcollection import TensorCollection
from tensordict._unbatched import _unbatched_with_batch_size
from tensordict.memmap import MemoryMappedTensor
from tensordict.utils import (
    _add_batch_dim_pre_hook,
    _any_unbatched,
    _as_context_manager,
    _cache_while_locked,
    _canonicalize_tensor,
    _CloudpickleWrapper,
    _compact_repr,
    _convert_list_to_stack,
    _copy_cached_result,
    _erase_cache_first,
    _foreach,
    _GENERIC_NESTED_ERR,
    _get_item,
    _index_preserve_data_ptr,
    _is_non_tensor,
    _is_tensorclass,
    _is_unbatched,
    _KEY_ERROR,
    _LIBRARY_DIRS,
    _lock_blocked,
    _LOCK_ERROR,
    _lock_warn,
    _maybe_correct_neg_dim,
    _mismatch_keys,
    _pass_through,
    _pass_through_cls,
    _proc_init,
    _prune_selected_keys,
    _REPR_OPTIONS,
    _set_max_batch_size,
    _shape,
    _SKIP_TENSORCLASS_POST_INIT,
    _split_tensordict,
    _strtobool,
    _td_fields,
    _unravel_key_to_tuple,
    _UPDATE_IGNORE_LOCK_WARNED,
    _user_stacklevel,
    _warn_update_ignore_lock,
    _zip_strict,
    capture_non_tensor_stack,
    DeviceType,
    expand_as_right,
    ftdim,
    IndexType,
    is_batchedtensor,
    is_non_tensor,
    is_tensorclass,
    KEY_SEPARATOR,
    list_to_stack,
    set_capture_non_tensor_stack,
    unravel_key,
    unravel_key_list,
)
from torch import _foreach_copy_, multiprocessing as mp, nn, Tensor
from torch.autograd import forward_ad
from torch.compiler import allow_in_graph, is_compiling
from torch.nn.parameter import Buffer, UninitializedTensorMixin
from torch.utils._pytree import tree_map

__all__ = [
    "NO_DEFAULT",
    "TensorDictBase",
    "from_any",
    "from_csv",
    "from_dict",
    "from_h5",
    "from_json",
    "from_namedtuple",
    "from_pandas",
    "from_parquet",
    "from_struct_array",
    "from_tuple",
    "from_zarr",
    "get_defaults_to_none",
    "is_tensor_collection",
    "set_get_defaults_to_none",
]

_foreach_copy_compiled = allow_in_graph(_foreach_copy_)

_has_h5 = importlib.util.find_spec("h5py") is not None


def _sync_cuda_transfer(stream=None):
    """Wait only for CUDA work up to the current point (e.g. queued D2H copies).

    Uses a CUDA event instead of torch.cuda.synchronize() so we do not wait
    for all pending GPU work, only for work recorded up to the event.
    """
    event = torch.cuda.Event()
    if stream is not None:
        event.record(stream)
    else:
        event.record()
    event.synchronize()


# NO_DEFAULT is a sentinel value used to detect when a default argument was not provided.
# Using None is not an option since `td.get(key)` (returning None) is a valid usage.
# When passed to methods like `get()`, it signals "raise KeyError if key is missing"
# rather than returning a default value.
class _NoDefault(enum.IntEnum):
    ZERO = 0


NO_DEFAULT = _NoDefault.ZERO

# _UNSET is a sentinel used in pop() to detect if a key exists without try/except.
# Unlike NO_DEFAULT (which triggers KeyError in get()), _UNSET can be returned by get()
# to indicate a missing key. This makes pop() compatible with torch.compile.
_UNSET = object()

T = TypeVar("T", bound="TensorCollection")


if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any


class _BEST_ATTEMPT_INPLACE:
    def __bool__(self):
        # we use an exception to exit when running `inplace = BEST_ATTEMPT_INPLACE if inplace else False`
        # more than once
        raise NotImplementedError


BEST_ATTEMPT_INPLACE = _BEST_ATTEMPT_INPLACE()

CompatibleType = Tensor | TensorCollection

_STR_MIXED_INDEX_ERROR = "Received a mixed string-non string index. Only string-only or string-free indices are supported."

_SELF_NESTING_ERROR = (
    "Cannot set a tensordict inside itself (key {!r}): tensordicts that "
    "contain themselves are not supported."
)

_HEURISTIC_EXCLUDED = (Tensor, tuple, list, set, dict, np.ndarray)

# Context-manager ops whose inverse __exit__ runs also when the body raises
_UNDO_ON_ERROR_OPS = frozenset({"to_module", "lock_", "unlock_"})

_GET_DEFAULTS_TO_NONE_REPLACEMENT = "td[key] to raise a KeyError for a missing key"

if "TD_GET_DEFAULTS_TO_NONE" in os.environ:
    _GET_DEFAULTS_TO_NONE = _strtobool(os.environ["TD_GET_DEFAULTS_TO_NONE"])
else:
    _GET_DEFAULTS_TO_NONE = True

if not _GET_DEFAULTS_TO_NONE:
    warn_deprecated_env_var(
        "TD_GET_DEFAULTS_TO_NONE",
        os.environ["TD_GET_DEFAULTS_TO_NONE"],
        removal="0.17",
        replacement=_GET_DEFAULTS_TO_NONE_REPLACEMENT,
    )


def _set_get_defaults_to_none(set_to_none: bool = True) -> None:
    global _GET_DEFAULTS_TO_NONE
    _GET_DEFAULTS_TO_NONE = bool(set_to_none)


def _get_defaults_to_none() -> bool:
    return _GET_DEFAULTS_TO_NONE


@deprecated(
    "set_get_defaults_to_none()",
    removal="0.17",
    replacement=_GET_DEFAULTS_TO_NONE_REPLACEMENT,
)
def set_get_defaults_to_none(set_to_none: bool = True):
    """Sets whether :meth:`~tensordict.TensorDictBase.get` returns ``None`` or raises for a missing key.

    The setting applies to every call of ``get`` and ``get_at`` that passes no
    default. The environment variable ``TD_GET_DEFAULTS_TO_NONE`` sets its
    initial value.

    .. deprecated:: 0.15
        Since v0.7, :meth:`~tensordict.TensorDictBase.get` returns ``None`` for a
        missing key by default. This function, and setting ``TD_GET_DEFAULTS_TO_NONE``
        to a false value, are deprecated and will be removed in TensorDict 0.17.
        Use ``td[key]`` to raise a ``KeyError`` for a missing key.

    Args:
        set_to_none (bool, optional): if ``True``, ``get`` returns ``None`` for
            a missing key when no default is passed. If ``False``, it raises a
            ``KeyError``. Defaults to ``True``.
    """
    _set_get_defaults_to_none(set_to_none)


@deprecated(
    "get_defaults_to_none()",
    removal="0.17",
    replacement=_GET_DEFAULTS_TO_NONE_REPLACEMENT,
)
def get_defaults_to_none(set_to_none: bool = True):
    """Returns whether :meth:`~tensordict.TensorDictBase.get` returns ``None`` for a missing key.

    ``False`` means that ``get`` raises a ``KeyError`` when no default is
    passed. See :func:`~tensordict.set_get_defaults_to_none`.

    .. deprecated:: 0.15
        Since v0.7, :meth:`~tensordict.TensorDictBase.get` returns ``None`` for a
        missing key by default. This function is deprecated and will be removed in
        TensorDict 0.17. Use ``td[key]`` to raise a ``KeyError`` for a missing key.

    Args:
        set_to_none (bool, optional): ignored. Defaults to ``True``.
    """
    return _get_defaults_to_none()


class _RecorderState(local):
    """Thread-safe, Per-thread state for :class:`_RecordDeviceTransfer`.

    Subclassing :class:`threading.local` is the documented way to get
    per-thread default attributes: ``__init__`` is invoked once for each
    thread that first touches the instance, so every thread sees
    ``marked = False`` / ``has_transfer = False`` initially.
    """

    def __init__(self):
        self.marked = False
        self.has_transfer = False


class _RecordDeviceTransfer:
    """A class that records the device transfers during a TensorDict initialization.

    State is held per-thread so concurrent ``TensorDict`` constructions
    on different threads do not race on the ``mark()`` guard.
    Single-threaded behaviour is unchanged.
    """

    def __init__(self):
        self._state = _RecorderState()

    @property
    def marked(self) -> bool:
        return self._state.marked

    def mark(self):
        if self._state.marked:
            raise RuntimeError("Can only mark one TensorDict at a time.")
        self._state.marked = True
        self._state.has_transfer = False

    def unmark(self):
        self._state.marked = False
        self._state.has_transfer = False

    def record_transfer(self, device):
        # Adds a device to all markers
        self._state.has_transfer = True

    def has_transfer(self):
        return self._state.has_transfer


_device_recorder = _RecordDeviceTransfer()


class _InplaceShapeState(local):
    """Per-thread state of the in-place shape ops.

    ``nested`` is ``True`` while ``_inplace_rebind_leaves`` runs the op on a
    nested tensordict. The top-level call has then checked the whole tree with
    ``_raise_if_holds_lazy_stack``, so the nested calls skip that walk.
    """

    def __init__(self):
        self.nested = False


_inplace_shape_state = _InplaceShapeState()


class _KeyThroughLeafError(ValueError):
    """The ValueError raised when a nested key runs through a leaf.

    :meth:`~TensorDictBase.get` called with a default returns the default
    instead.
    """


class _InplaceWriteError(ValueError, RuntimeError):
    """The error raised when an in-place write of an entry fails.

    Subclasses ValueError, the type that :meth:`~TensorDictBase.set_` raised
    before, and RuntimeError, which torch raises for the same write in
    :meth:`~TensorDictBase.update_` and :meth:`~TensorDictBase.copy_`.
    """


def _key_through_leaf_error(key: tuple, i: int, value: Any) -> _KeyThroughLeafError:
    """The error for reading ``key[i:]`` in ``value``, the leaf at ``key[i - 1]``."""
    rest = key[i] if len(key) == i + 1 else key[i:]
    return _KeyThroughLeafError(
        f"{key[i - 1]!r} is a {type(value).__name__}, not a tensordict, "
        f"so it has no entry {rest!r}."
    )


def _raise_if_not_a_tensordict(
    err: AttributeError, key: tuple, i: int, value: Any
) -> None:
    """Raises a ValueError if ``err`` comes from reading ``key[i:]`` in a leaf.

    ``value`` is the entry at ``key[i - 1]`` and ``err`` the AttributeError
    that ``value._get_tuple(key[i:], ...)`` raised. Other AttributeErrors are
    left to the caller, which returns ``None``.
    """
    if "has no attribute" in str(err):
        raise _key_through_leaf_error(key, i, value)


def _is_inside(key: NestedKey, other: NestedKey) -> bool:
    """Returns whether the nested key ``key`` lies strictly inside ``other``."""
    key = (key,) if isinstance(key, str) else key
    other = (other,) if isinstance(other, str) else other
    return len(key) > len(other) and key[: len(other)] == other


def _rename_keys_pairs(
    mapping: Mapping[NestedKey, NestedKey],
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """The pairs of tuple keys of :meth:`~TensorDictBase.rename_keys`, after the checks that read no data."""
    pairs = []
    for src, dst in mapping.items():
        pair = []
        for key in (src, dst):
            if not isinstance(key, (str, tuple)):
                raise TypeError(
                    f"Expected a key to be a string or a tuple of strings but found {type(key)}"
                )
            key_tuple = _unravel_key_to_tuple(key)
            if not key_tuple:
                raise KeyError(_GENERIC_NESTED_ERR.format(key))
            pair.append(key_tuple)
        pairs.append(tuple(pair))
    for side, keys in (
        ("source", [src for src, _ in pairs]),
        ("destination", [dst for _, dst in pairs]),
    ):
        for i, key in enumerate(keys):
            for other in keys[:i]:
                if key == other:
                    raise ValueError(f"The key {key} is a {side} more than once.")
                if _is_inside(key, other) or _is_inside(other, key):
                    raise ValueError(
                        f"The {side}s {other} and {key} lie inside one another, so "
                        "the result would depend on the order of the mapping."
                    )
    return pairs


def _keep(value):
    return value


def _rename_entries(
    tds: list[TensorDictBase],
    pairs: list[tuple[tuple[str, ...], tuple[str, ...]]],
    *,
    strict: bool,
    prune_root: bool,
) -> None:
    """Moves the entries of ``pairs`` in each tensordict of ``tds``: the end of :meth:`~TensorDictBase.rename_keys`.

    ``tds`` are private copies, so a call that raises halfway changes nothing
    the user holds. Every source is read before any is deleted, and the
    collisions are checked after the deletions, so swaps and chains work. A
    sub-tensordict that the move empties is deleted only if it is empty in
    every tensordict of ``tds``, and never when it is a field of a
    tensorclass (``prune_root=False`` for the fields of the root).
    """
    moves = []
    for td in tds:
        keys = td.keys(include_nested=True)
        present = []
        for src, dst in pairs:
            if src not in keys:
                if strict:
                    raise KeyError(
                        _KEY_ERROR.format(
                            src[0] if len(src) == 1 else src,
                            type(td).__name__,
                            sorted(keys, key=str),
                        )
                    )
                continue
            if src != dst:
                present.append((src, dst))
        moves.append(present)
    values = [
        [td._get_tuple(src, NO_DEFAULT) for src, _ in present]
        for td, present in zip(tds, moves)
    ]
    for td, present in zip(tds, moves):
        for src, _ in present:
            td.del_(src)
    parents = {
        src[:i]
        for present in moves
        for src, _ in present
        for i in range(1 if prune_root else 2, len(src))
    }
    for parent in sorted(parents, key=len, reverse=True):
        holders = []
        for td in tds:
            group = td._get_tuple(parent, None)
            if group is None:
                continue
            container = td._get_tuple(parent[:-1], None) if len(parent) > 1 else td
            if (
                _is_tensorclass(type(container))
                or not isinstance(group, TensorDictBase)
                or group._has_exclusive_keys
                or list(group.keys())
            ):
                holders = None
                break
            holders.append(td)
        for td in holders or ():
            td.del_(parent)
    for td, present in zip(tds, moves):
        keys = td.keys(include_nested=True)
        for _, dst in present:
            if dst in keys:
                raise ValueError(
                    f"The destination {dst} is already present and is not renamed away."
                )
            for i in range(1, len(dst)):
                value = td._get_tuple(dst[:i], None)
                if value is None:
                    break
                if _is_leaf_nontensor(type(value)):
                    raise ValueError(
                        f"The destination {dst} runs through the entry {dst[:i]}, "
                        "which is not a tensordict."
                    )
    for td, present, vals in zip(tds, moves, values):
        for (_, dst), value in zip(present, vals):
            td._set_tuple(dst, value, inplace=False, validated=True, non_blocking=False)


def _holds_its_tensors(td: TensorDictBase) -> bool:
    """Returns whether ``td`` reads its entries from a ``TensorDict``, which holds its tensors.

    ``TensorDictParams``, ``TypedTensorDict`` and tensorclasses read through the
    container that they wrap. h5 files, stores, lazy stacks, sub-tensordicts
    and lazy views build their entries again on each read.
    """
    from tensordict._td import TensorDict

    while not isinstance(td, TensorDict):
        from tensordict.nn.params import TensorDictParams
        from tensordict.typedtensordict import TypedTensorDict

        if isinstance(td, TensorDictParams):
            td = td._param_td
        elif isinstance(td, TypedTensorDict):
            td = td._source
        elif _is_tensorclass(type(td)):
            td = td._tensordict
        else:
            return False
    return True


def _holds_leaves_of(td: TensorDictBase, other: TensorDictBase) -> bool:
    """Returns whether each leaf of ``other`` is the tensor that ``td`` holds under the same key.

    Only containers that hold their tensors are read (see ``_holds_its_tensors``).
    For the others, this returns ``False`` without reading any entry.
    """
    if not _holds_its_tensors(td):
        # A NonTensorStack holds no tensor for the op to write into
        return is_non_tensor(td)
    for key, value in other.items():
        held = td._get_str(key, None)
        if value is held:
            continue
        if not (
            _is_tensor_collection(type(value))
            and _is_tensor_collection(type(held))
            and _holds_leaves_of(held, value)
        ):
            return False
    return True


def _leaves_on(td: TensorDictBase, device: torch.device) -> bool:
    """Returns whether each tensor of the ``TensorDict`` ``td`` is on ``device`` already.

    So are the tensors of its nested ``TensorDict`` instances; non-tensor data
    holds no tensor. Any other nested collection gives ``False``. A tensor on
    another device is left to ``td.to(device)``, which copies all the leaves
    before anything is written: on CUDA, copies from pageable memory issued
    between the writes would wait for each write.
    """
    for item in td._tensordict.values():
        if isinstance(item, Tensor):
            if item.device != device:
                return False
        elif type(item) is type(td):
            if not _leaves_on(item, device):
                return False
        elif not is_non_tensor(item):
            return False
    return True


def _pair_dense_leaves(
    dests: dict, sources: dict, vals: list, other_vals: list
) -> bool:
    """Pairs the leaves of two trees of exact ``TensorDict`` nodes that have the same keys.

    ``dests`` and ``sources`` are the ``_tensordict`` dicts of two such nodes.
    Their tensors are appended to ``vals`` and ``other_vals``, depth first in
    the key order of ``dests``: the order of ``_items_list(True, True)``.
    Returns ``False`` at the first key that ``sources`` lacks, or whose entries
    are not both plain tensors or both exact ``TensorDict`` nodes, and when the
    nodes have different numbers of keys: the caller then takes the general
    route of ``_update_plan``.
    """
    if len(dests) != len(sources):
        return False
    for key, val in dests.items():
        other = sources.get(key)
        cls = type(val)
        if cls is Tensor:
            if type(other) is not Tensor:
                return False
            vals.append(val)
            other_vals.append(other)
        elif not (
            cls is _TENSORDICT_TYPE
            and type(other) is cls
            and _pair_dense_leaves(val._tensordict, other._tensordict, vals, other_vals)
        ):
            return False
    return True


def _mixes_lazy_stacks(td: TensorDictBase, others: tuple) -> bool:
    """Returns whether some of ``td`` and the tensordicts in ``others`` are lazy stacks and some are not.

    The pointwise ops flatten a lazy stack into one key per member, such as
    ``("0", "a")``, which a dense operand does not have (#1932).
    """
    # a cheap filter first: lazy stacks set _lazy, numbers and tensors do not have it
    lazy = td._lazy
    for other in others:
        other_lazy = getattr(type(other), "_lazy", None)
        if other_lazy is None and _is_tensorclass(type(other)):
            # a tensorclass counts as the tensordict it wraps
            other = other._tensordict
            other_lazy = other._lazy
        if other_lazy is False and type(other)._wrapped_attr is not None:
            # a TypedTensorDict counts as the tensordict it wraps
            other_lazy = getattr(other, type(other)._wrapped_attr)._lazy
        lazy = lazy or other_lazy
    if not lazy:
        return False
    from tensordict._lazy import LazyStackedTensorDict

    is_stack = isinstance(td, LazyStackedTensorDict)
    for other in map(_unwrap_tensorclass, others):
        if not isinstance(other, TensorDictBase):
            continue
        if not is_stack and type(other)._wrapped_attr is not None:
            # a lazy stack pairs a typed operand's leaves by key already
            other = getattr(other, type(other)._wrapped_attr)
        if isinstance(other, LazyStackedTensorDict) is not is_stack:
            return True
    return False


def _unwrap_tensorclass(other: Any) -> Any:
    """Returns the tensordict behind a tensorclass ``other``, and any other ``other`` as it is."""
    return other._tensordict if _is_tensorclass(type(other)) else other


def _same_leaf_keys(td: TensorDictBase, others: tuple) -> bool:
    """Returns whether ``others`` holds tensordicts, all with the leaf keys of ``td``."""
    tds = [
        other
        for other in map(_unwrap_tensorclass, others)
        if isinstance(other, TensorDictBase)
    ]
    keys = set(td.keys(True, True))
    return bool(tds) and all(set(other.keys(True, True)) == keys for other in tds)


def _apply_pointwise(
    td: TensorDictBase, fn: Callable, others: tuple, inplace: bool
) -> TensorDictBase:
    """Computes ``fn(leaf, *others)`` leaf by leaf, pairing the leaves of the tensordicts in ``others`` by key."""
    others = tuple(map(_unwrap_tensorclass, others))
    is_td = [isinstance(other, TensorDictBase) for other in others]

    def leaf_fn(x, *leaves):
        leaves = iter(leaves)
        return fn(
            x, *[next(leaves) if flag else other for flag, other in zip(is_td, others)]
        )

    tds = [other for flag, other in zip(is_td, others) if flag]
    if inplace:
        return td.apply_(leaf_fn, *tds)
    # apply ignores the keys that only the others have; the dense ops raise
    keys = set(td.keys(True, True))
    for other in tds:
        other_keys = set(other.keys(True, True))
        if other_keys != keys:
            _mismatch_keys(keys, other_keys)
    return td.apply(leaf_fn, *tds, propagate_lock=True)


def _maybe_broadcast_other(op: str, n_other: int = 1) -> Callable[[Callable], Callable]:
    """Ensures that elementwise ops are broadcast when an nd tensor is passed."""
    # add_, mul_, ... are in-place; __eq__, __lt__, ... also end with "_".
    inplace = op.endswith("_") and not op.endswith("__")
    # torch.Tensor has no rsub method: call torch.rsub on the leaves instead.
    torch_op = None if hasattr(torch.Tensor, op) else getattr(torch, op)

    def leaf_op(x, *args, **kwargs):
        if torch_op is None:
            return getattr(x, op)(*args, **kwargs)
        return torch_op(x, *args, **kwargs)

    def pointwise(self, others, args, kwargs):
        # alpha=None and default=None are the defaults of the ops
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return _apply_pointwise(
            self, lambda x, *xs: leaf_op(x, *xs, *args, **kwargs), others, inplace
        )

    def wrap_func(func):
        @wraps(func)
        def new_func(self, *others, **kwargs):
            others, args = others[:n_other], others[n_other:]
            need_broadcast = False
            for other in others:
                if other is None:
                    continue
                if (isinstance(other, torch.Tensor) and other.ndim) or (
                    _is_tensor_collection(type(other))
                    and other.ndim
                    and other.shape != self.shape
                ):
                    need_broadcast = True
                    break
            if not need_broadcast:
                if kwargs.get("default") is None and _mixes_lazy_stacks(self, others):
                    return pointwise(self, others, args, kwargs)
                try:
                    return func(self, *others, *args, **kwargs)
                except KeyError:
                    # A lazy stack nested in a dense tensordict gives keys such
                    # as ("n", "0", "a"), which a dense operand lacks (#1932).
                    if kwargs.get("default") is not None or not _same_leaf_keys(
                        self, others
                    ):
                        raise
                    if inplace:
                        # apply_ writes as it goes: raise for a leaf that only
                        # some members of a lazy stack have before any write
                        _apply_pointwise(self, lambda *xs: None, others, False)
                    return pointwise(self, others, args, kwargs)
            others_map = []
            shape = self.shape
            self_expand = self
            # None and numbers, such as the weight of lerp, are passed as they are
            is_operand = [
                isinstance(other, torch.Tensor) or _is_tensor_collection(type(other))
                for other in others
            ]
            shapes = [shape, *[o.shape for o, f in zip(others, is_operand) if f]]
            shape = torch.broadcast_shapes(*shapes)
            if shape != self_expand.shape:
                self_expand = self_expand.expand(shape)
            for other, flag in zip(others, is_operand):
                if not flag:
                    others_map.append(other)
                    continue
                # broadcast dims
                if shape != other.shape:
                    other = other.expand(shape)
                others_map.append(other)
            if any(isinstance(other, torch.Tensor) for other in others_map):
                from tensordict._lazy import LazyStackedTensorDict

                if isinstance(self_expand, LazyStackedTensorDict) and not is_non_tensor(
                    self_expand
                ):
                    # The leaves of each member lack the stack dim: give each
                    # member its slice of the operands.
                    stack_dim = self_expand.stack_dim
                    others_split = [
                        other.unbind(stack_dim)
                        if flag
                        else [other] * self_expand.shape[stack_dim]
                        for other, flag in zip(others_map, is_operand)
                    ]
                    results = [
                        getattr(td, op)(*member_others, *args, **kwargs)
                        for td, *member_others in zip(
                            self_expand.tensordicts, *others_split
                        )
                    ]
                    if inplace and self_expand is self:
                        return self
                    return self_expand._new_lazy_unsafe(
                        *results,
                        stack_dim=stack_dim,
                        stack_dim_name=self_expand._td_dim_name,
                    )
                if (
                    n_other > 1
                    and any(_is_tensor_collection(type(o)) for o in others_map)
                    # an empty tensordict returns an empty result, as before
                    and not self_expand.is_empty()
                    # the leaves of a nested lazy stack lack the stack dim, so
                    # expand_as_right would misalign the tensors: raise as before
                    and not any(
                        _has_nested_lazy_stack(o)
                        for o in (self_expand, *others_map)
                        if _is_tensor_collection(type(o))
                    )
                ):
                    if op == "clamp":
                        # clamp rejects a tensordict bound next to a tensor one
                        return func(self, *others, *args, **kwargs)
                    # pair the tensordict operands by key, expand the tensors
                    is_tensor = [
                        isinstance(other, torch.Tensor) for other in others_map
                    ]
                    return _apply_pointwise(
                        self_expand,
                        lambda x, *xs: leaf_op(
                            x,
                            *[
                                expand_as_right(other, x) if flag else other
                                for other, flag in zip(xs, is_tensor)
                            ],
                            *args,
                            **kwargs,
                        ),
                        others_map,
                        inplace,
                    )
                result = self_expand._fast_apply(
                    lambda x: leaf_op(
                        x,
                        *[
                            expand_as_right(other, x) if flag else other
                            for other, flag in zip(others_map, is_operand)
                        ],
                        *args,
                        **kwargs,
                    )
                )
                # Like torch, an in-place op returns self once it has written into
                # the tensors that self holds. Containers that build their entries
                # on each read (lazy stacks, views) still return the new tensordict.
                if inplace and self_expand is self and _holds_leaves_of(self, result):
                    return self
                return result
            return getattr(self_expand, op)(*others_map, *args, **kwargs)

        return new_func

    return wrap_func


__base__setattr__ = torch.nn.Module.__setattr__


def _maybe_preserve_module_state(
    module: torch.nn.Module,
    name: str,
    tensor: torch.Tensor,
    *,
    preserve_module_state: bool,
) -> torch.Tensor:
    if not preserve_module_state:
        return tensor
    try:
        param = module._parameters.get(name, NO_DEFAULT)
    except AttributeError:
        param = NO_DEFAULT
    if (
        param is not NO_DEFAULT
        and param is not None
        and isinstance(tensor, torch.Tensor)
    ):
        if (
            not isinstance(tensor, torch.nn.Parameter)
            or tensor.requires_grad != param.requires_grad
        ):
            if _tracks_gradients(tensor):
                # swap_tensor writes it to module._parameters as it is
                return tensor
            return _new_parameter(tensor, requires_grad=param.requires_grad)
    elif (
        preserve_module_state
        and isinstance(tensor, torch.nn.Parameter)
        and name in getattr(module, "_buffers", {})
    ):
        persistent = name not in module._non_persistent_buffers_set
        return Buffer(tensor, persistent=persistent)
    return tensor


def _set_tensor_dict(
    __dict__,
    _parameters,
    _buffers,
    hooks,
    module: torch.nn.Module,
    name: str,
    tensor: torch.Tensor,
    inplace: bool,
    *,
    return_swap: bool,
    preserve_module_state: bool,
) -> None:
    """Simplified version of torch.nn.utils._named_member_accessor."""
    if (
        not inplace
        and not hooks
        and type(_parameters) is dict
        and type(_buffers) is dict
    ):
        if type(tensor) is nn.Parameter:
            out = _parameters.get(name, NO_DEFAULT)
            if (type(out) is nn.Parameter or out is None) and (
                not preserve_module_state
                or out is None
                or tensor.requires_grad == out.requires_grad
            ):
                # Pop before setting to retain the registration order of the
                # general path, including when updating only part of a module.
                del _parameters[name]
                _parameters[name] = tensor
                return out
        elif (
            type(tensor) is Tensor
            and not getattr(tensor, "_is_param", False)
            and name not in _parameters
            and name in _buffers
        ):
            out = _buffers.pop(name)
            _buffers[name] = tensor
            return out
    was_buffer = False
    keep_parameter_slot = False
    # Build the new value before removing the old one, so that the module
    # keeps its entry if building the new value raises.
    out = _parameters.get(name, NO_DEFAULT)  # type: ignore[assignment]
    was_parameter = out is not NO_DEFAULT
    if out is NO_DEFAULT:
        out = _buffers.get(name, NO_DEFAULT)
        was_buffer = out is not NO_DEFAULT
    if out is NO_DEFAULT:
        out = __dict__[name]
    if inplace:
        # swap tensor and out after updating out
        out_tmp = out.clone() if return_swap else out
        out.data.copy_(tensor.data)
        tensor = out
        out = out_tmp
    elif (
        preserve_module_state
        and was_parameter
        and out is not None
        and isinstance(tensor, torch.Tensor)
    ):
        if (
            not isinstance(tensor, torch.nn.Parameter)
            or tensor.requires_grad != out.requires_grad
        ):
            if _tracks_gradients(tensor):
                keep_parameter_slot = True
            else:
                tensor = _new_parameter(tensor, requires_grad=out.requires_grad)
    elif (
        preserve_module_state is True
        and was_buffer
        and isinstance(tensor, torch.nn.Parameter)
    ):
        persistent = name not in module._non_persistent_buffers_set
        tensor = Buffer(tensor, persistent=persistent)

    if was_parameter:
        del _parameters[name]
    elif was_buffer:
        del _buffers[name]
    else:
        del __dict__[name]
    if isinstance(tensor, torch.nn.Parameter):
        for hook in hooks:
            output = hook(module, name, tensor)
            if output is not None:
                tensor = output
        _parameters[name] = tensor

        if isinstance(tensor, UninitializedTensorMixin):
            module.register_forward_pre_hook(
                _add_batch_dim_pre_hook(), with_kwargs=True
            )

    elif keep_parameter_slot:
        # keep the registration without making a new leaf, as
        # torch.func.functional_call does
        _parameters[name] = tensor
    elif was_buffer and isinstance(tensor, torch.Tensor):
        _buffers[name] = tensor
    else:
        __dict__[name] = tensor
    return out


def _wrap_extra_state(state: Any) -> Any:
    # Wrap the non-tensor objects that get_extra_state() returns, also inside
    # dicts, in NonTensorData: the tensordict stays valid and to_dict() gives
    # them back unchanged.
    from tensordict.tensorclass import NonTensorData

    if isinstance(state, dict):
        return {key: _wrap_extra_state(val) for key, val in state.items()}
    if isinstance(state, Tensor) or _is_tensor_collection(type(state)):
        return state
    return NonTensorData(state)


def _new_parameter(tensor: torch.Tensor, *, requires_grad: bool) -> torch.nn.Parameter:
    """Wraps ``tensor`` in an ``nn.Parameter`` that shares its storage.

    ``nn.Parameter`` rejects a ``MemoryMappedTensor``, whose ``detach()``
    returns a plain ``Tensor``, so it wraps a plain view of the same storage.
    """
    if isinstance(tensor, MemoryMappedTensor):
        tensor = tensor.as_subclass(torch.Tensor)
    return torch.nn.Parameter(tensor, requires_grad=requires_grad)


def _tracks_gradients(tensor: torch.Tensor) -> bool:
    """Whether wrapping ``tensor`` in a new ``nn.Parameter`` would stop its gradients.

    A new ``nn.Parameter`` is a new leaf, so gradients would no longer reach a
    tensor that requires grad (a leaf or a computed tensor) or a ``vmap`` slice.
    """
    return not isinstance(tensor, torch.nn.Parameter) and (
        tensor.requires_grad or is_batchedtensor(tensor)
    )


def _check_p2p_peer(
    peer: int | None,
    group_peer: int | None,
    group: "torch.distributed.ProcessGroup" | None,
    peer_name: str,
    group_peer_name: str,
) -> None:
    """Validates the global-rank / group-rank peer arguments of the p2p methods.

    Mirrors the torch functional API contract: the peer is specified either
    globally (``dst``/``src``) or relative to ``group``
    (``group_dst``/``group_src``), never both.
    """
    if group_peer is not None:
        if group is None:
            raise ValueError(f"`{group_peer_name}` requires `group` to be passed.")
        if peer is not None:
            raise ValueError(
                f"`{peer_name}` and `{group_peer_name}` are mutually exclusive."
            )
    elif peer is None:
        raise ValueError(
            f"Exactly one of `{peer_name}` and `{group_peer_name}` must be provided."
        )


def _resolve_tensorclass_type(type_str: str):
    """Import and return a tensorclass from its fully qualified name.

    Args:
        type_str: A dotted path like ``"tensordict.testing.MyData"``.
    """
    import importlib

    module_path, class_name = type_str.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# TensorDict, which _td.py sets once it defines the class: __getitem__ reads a
# str key of an exact TensorDict straight from its dict.
_TENSORDICT_TYPE = None


def _set_tensordict_type(cls):
    global _TENSORDICT_TYPE
    _TENSORDICT_TYPE = cls


# _SubTensorDict, which _td.py sets once it defines the class: _apply_nest
# reads it without a function-level import on every call.
_SUBTENSORDICT_TYPE = None


def _set_subtensordict_type(cls):
    global _SUBTENSORDICT_TYPE
    _SUBTENSORDICT_TYPE = cls


# The torch functions that TensorDictBase.__torch_function__ handles, which
# _torch_func.py registers (as its TD_HANDLED_FUNCTIONS, the same dict):
# __torch_function__ reads it without a function-level import on every call.
_TD_HANDLED_FUNCTIONS: dict[Callable, Callable] = {}


def _register_tensor_class(cls):
    global _ACCEPTED_CLASSES
    _ACCEPTED_CLASSES = set(_ACCEPTED_CLASSES)
    _ACCEPTED_CLASSES.add(cls)
    _ACCEPTED_CLASSES = tuple(_ACCEPTED_CLASSES)


def _is_accepted_class(cls: type) -> bool:
    """Returns True if ``cls`` is a subclass of one of the ``_ACCEPTED_CLASSES``."""
    if is_compiling():
        # Dynamo guards on the length of _ACCEPTED_CLASSES, which
        # _register_tensor_class rebinds each time a tensorclass is defined,
        # so the frame would recompile. Every registered class is a subclass
        # of one of these three, or a tensorclass.
        return issubclass(
            cls, (Tensor, TensorDictBase, ftdim.Tensor)
        ) or _is_tensorclass(cls)
    return issubclass(cls, _ACCEPTED_CLASSES)


_TENSOR_COLLECTION_MEMO = {}


def _unflatten_state_dict(flat_sd):
    """Convert a flat state_dict (dot-separated keys with _metadata) to nested OrderedDicts.

    Creates intermediate nodes from both data keys and _metadata keys, so
    that tensor-collection nodes that carry only metadata (e.g. NonTensorData)
    are preserved in the nested structure.
    """
    _metadata = getattr(flat_sd, "_metadata", None)
    root = collections.OrderedDict()
    root._metadata = collections.OrderedDict()

    def _ensure_nested(parent, part):
        if part not in parent:
            nested = collections.OrderedDict()
            nested._metadata = collections.OrderedDict()
            parent[part] = nested
        return parent[part]

    for flat_key, value in flat_sd.items():
        parts = flat_key.split(".")
        current = root
        for part in parts[:-1]:
            current = _ensure_nested(current, part)
        current[parts[-1]] = value

    if _metadata is not None:
        for meta_key, meta_value in _metadata.items():
            if meta_key == "":
                root._metadata[""] = meta_value
            else:
                parts = meta_key.split(".")
                current = root
                for part in parts:
                    current = _ensure_nested(current, part)
                current._metadata[""] = meta_value

    return root


def _is_tensor_collection(datatype: type) -> bool:
    if datatype is Tensor:
        # The most common entry type: skip is_compiling() and the memo lookup.
        return False
    if is_compiling():
        return issubclass(datatype, TensorDictBase) or _is_tensorclass(datatype)
    out = _TENSOR_COLLECTION_MEMO.get(datatype)
    if out is None:
        out = issubclass(datatype, TensorDictBase) or _is_tensorclass(datatype)
        _TENSOR_COLLECTION_MEMO[datatype] = out
    return out


def is_tensor_collection(datatype: type | Any) -> bool:
    """Returns ``True`` if an object or a type is a tensordict or a tensorclass.

    A tensor, a dict and any other object return ``False``.

    Args:
        datatype (type or object): the type to check, or an object whose type
            is checked.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict, TensorDictBase, is_tensor_collection, tensorclass
        >>> is_tensor_collection(TensorDictBase)
        True
        >>> is_tensor_collection(TensorDict())
        True
        >>> @tensorclass
        ... class MyClass:
        ...     pass
        ...
        >>> is_tensor_collection(MyClass)
        True
        >>> is_tensor_collection(MyClass(batch_size=[]))
        True
        >>> is_tensor_collection(torch.zeros(3)), is_tensor_collection({"a": 0})
        (False, False)

    """
    # memoizing is 2x faster
    if not isinstance(datatype, type):
        datatype = type(datatype)
    return _is_tensor_collection(datatype)


def _non_tensor_content(value):
    # What td[key] returns for a non-tensor entry: its content, or a list for
    # a stack of non-tensor data. Shared by TensorDictBase.__getitem__ and
    # _get_tuple_maybe_non_tensor.
    if isinstance(value, TensorDictBase):
        return value.tolist(as_linked_list=True)
    return value.data


def _default_is_leaf(cls: Type) -> bool:
    """Returns ``True`` if a type is a leaf: anything but a tensordict or a tensorclass.

    This is the default ``is_leaf`` of :meth:`~tensordict.TensorDictBase.keys`
    and the methods that walk the nested entries. A
    :class:`~tensordict.UnbatchedTensor` is a leaf too. A
    :class:`~tensordict.NonTensorData` is not, so ``leaves_only=True`` skips
    the non-tensor entries. See :ref:`keys`.

    Args:
        cls (type): the type of an entry, as returned by
            :meth:`~tensordict.TensorDictBase.entry_class`.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict, NonTensorData, default_is_leaf
        >>> default_is_leaf(torch.Tensor), default_is_leaf(NonTensorData), default_is_leaf(TensorDict)
        (True, False, False)
        >>> td = TensorDict(a={}, b="a string!", c=torch.zeros(()))
        >>> list(td.keys(leaves_only=True, is_leaf=default_is_leaf))
        ['c']

    .. seealso:: :func:`~tensordict.is_leaf_nontensor`.
    """
    # Only check for _pass_through attribute, not _is_non_tensor
    # This ensures NonTensorData is NOT considered a leaf (preserving original behavior)
    # while UnbatchedTensor IS considered a leaf
    if cls is Tensor:
        return True
    return not _is_tensor_collection(cls) or getattr(cls, "_pass_through", False)


def _flatten_keys_leaves(
    td: TensorCollection, is_leaf: Callable[[Type], bool]
) -> list[tuple[NestedKey, Any]]:
    """Returns the ``(key, value)`` pairs that ``flatten_keys`` keeps.

    Unlike ``items(True, True, is_leaf)``, a tensor collection whose class
    ``is_leaf`` accepts is kept whole: its entries are not listed.
    """
    result = []
    for key, val in td.items():
        cls = type(val)
        if is_leaf(cls):
            result.append(((key,), val))
        elif _is_tensor_collection(cls) and not _pass_through(val):
            result.extend(
                ((key, *subkey), subval)
                for subkey, subval in _flatten_keys_leaves(val, is_leaf)
            )
    return result


def _is_leaf_nontensor(cls: Type) -> bool:
    """Returns ``True`` if a type is a tensor or non-tensor data.

    Pass it as ``is_leaf`` to :meth:`~tensordict.TensorDictBase.keys` and the
    methods that walk the nested entries, to list the non-tensor entries
    (:class:`~tensordict.NonTensorData`, :class:`~tensordict.NonTensorStack`)
    with the tensors. A :class:`~tensordict.UnbatchedTensor` is a leaf too.
    Other tensordicts and tensorclasses are not leaves, and nor is a type that
    is neither a tensor nor a tensor collection. See :ref:`non-tensor-data`.

    Args:
        cls (type): the type of an entry, as returned by
            :meth:`~tensordict.TensorDictBase.entry_class`.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict, NonTensorData, is_leaf_nontensor
        >>> is_leaf_nontensor(torch.Tensor), is_leaf_nontensor(NonTensorData), is_leaf_nontensor(TensorDict)
        (True, True, False)
        >>> td = TensorDict(a={}, b="a string!", c=torch.zeros(()))
        >>> list(td.keys(leaves_only=True, is_leaf=is_leaf_nontensor))
        ['b', 'c']

    .. seealso:: :func:`~tensordict.default_is_leaf`.
    """
    if cls is Tensor:
        return True
    if _is_tensor_collection(cls):
        return _pass_through_cls(cls)
    return issubclass(cls, torch.Tensor)


def _load_metadata(prefix: Path):
    filepath = prefix / "meta.json"
    # `open` as a method so that archive paths (zip entries) can be read
    # through the same code path as regular files.
    with filepath.open("rb") as json_metadata:
        metadata = json.loads(json_metadata.read())
    return metadata


class _NestedTensorsAsLists:
    """Class used to iterate over leaves of lazily stacked tensordicts."""

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(cls, cls).__new__(cls)
        return cls.instance

    def __bool__(self):
        return False

    def __call__(self, val):
        return _default_is_leaf(val)


class _NestedTensorsAsListsNonTensor:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(cls, cls).__new__(cls)
        return cls.instance

    def __bool__(self):
        return False

    def __call__(self, val):
        return _is_leaf_nontensor(val)


_NESTED_TENSORS_AS_LISTS = _NestedTensorsAsLists()


_NESTED_TENSORS_AS_LISTS_NONTENSOR = _NestedTensorsAsListsNonTensor()

# The is_leaf functions that accept no tensor collection with entries: for
# them, items(True, True, is_leaf) lists the leaves that flatten_keys keeps.
_IS_LEAF_WITHOUT_NODES = (
    _default_is_leaf,
    _NESTED_TENSORS_AS_LISTS,
    _NESTED_TENSORS_AS_LISTS_NONTENSOR,
    _is_leaf_nontensor,
    _is_non_tensor,
)


def _leaf_data(leaf):
    """The ``data`` of a leaf: the function that ``TensorDictBase.data`` applies."""
    return leaf.data


# Set while an in-place method runs on a tensordict whose file is opened
# read-only, so that the nested calls it makes do not warn again.
_READ_ONLY_CALL = local()
# Set while empty() builds a result for the names it was given, so that the
# empty() calls it makes do not warn of the names again.
_EMPTY_NAMES_CALL = local()
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _normalize_from_dict_args(input_dict, *, batch_size, batch_dims):
    """Checks the batch-size arguments of a ``from_dict`` that writes to a file or a store.

    Raises if both ``batch_size`` and ``batch_dims`` are given, as
    ``TensorDict.from_dict`` does. Returns ``batch_size``, else the batch size
    of a tensor collection ``input_dict``, else ``None``: the caller then
    infers it from the entries.
    """
    if batch_dims is not None and batch_size is not None:
        raise ValueError("Cannot pass both batch_size and batch_dims to `from_dict`.")
    if batch_size is None and _is_tensor_collection(type(input_dict)):
        return input_dict.batch_size
    return batch_size


def _check_member_as_lists(td: TensorDictBase, is_leaf) -> None:
    """Raises if an in-place method walks the copies that ``td`` reads (#2592).

    ``td`` is a member of a lazy stack, or a tensordict that reads copies,
    such as a file, which a tensordict holds (#2775).
    """
    if td._reads_return_copies and is_leaf is _NESTED_TENSORS_AS_LISTS:
        _raise_if_inplace_call()


def _raise_on_exclusive_keys(method: str) -> None:
    """Raises because ``method`` would drop the keys that some members of a lazy stack lack."""
    raise RuntimeError(
        f"the method {method} cannot complete when there are exclusive keys."
    )


def _has_top_level_exclusive_keys(td: TensorDictBase) -> bool:
    """Whether the members of the lazy stack ``td`` differ in their top-level keys.

    Call it only when ``td._has_exclusive_keys`` is set, that is on a lazy stack.
    """
    keys = set(td.keys())
    return any(set(member.keys()) != keys for member in td.tensordicts)


def _warn_read_only(method: str) -> None:
    """Warns that the in-place ``method`` leaves a file opened read-only unchanged."""
    if getattr(_READ_ONLY_CALL, "running", False):
        return
    # point to the first frame outside tensordict: the user's call
    frame, stacklevel = sys._getframe(1), 2
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame, stacklevel = frame.f_back, stacklevel + 1
    warnings.warn(
        deprecation_message(
            f"Calling {method}() on a tensordict stored in a file opened "
            "read-only, which leaves the file unchanged,",
            removal="0.17",
        )
        + " From then on, the call raises. Open the file with mode='r+' to "
        "write the result, or call to_tensordict() first to change a copy in "
        "memory.",
        FutureWarning,
        stacklevel=stacklevel,
    )


def _warn_names_length(names: list, batch_dims: int) -> None:
    """Warns that ``names``, of the wrong length, is accepted because of its None entries, which raises from 0.17 (#2600)."""
    # point to the first frame outside tensordict: the user's call
    frame, stacklevel = sys._getframe(1), 2
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame, stacklevel = frame.f_back, stacklevel + 1
    warnings.warn(
        f"Got the dimension names {names} for {batch_dims} batch dims. The "
        "list has the wrong length, but is accepted today because of its None "
        "entries, and its names are dropped. Starting with TensorDict 0.17, "
        "the default will change to raising a ValueError, as for any other "
        "list of the wrong length. Pass one name per batch dim.",
        FutureWarning,
        stacklevel=stacklevel,
    )


def _warn_stack_dim_name_taken(name: str) -> None:
    """Warns that the stack dim ``name`` of a lazy stack is also a dim name of its members, which raises from 0.17 (#2690)."""
    # point to the first frame outside tensordict: the user's call
    frame, stacklevel = sys._getframe(1), 2
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame, stacklevel = frame.f_back, stacklevel + 1
    warnings.warn(
        f"The stack dim name {name!r} is also the name of a dim of the "
        "stacked tensordicts, so the lazy stack has non-unique dimension "
        "names. This is accepted today. Starting with TensorDict 0.17, the "
        "default will change to raising a ValueError, as on a TensorDict. "
        "Give the stack dim a name the stacked tensordicts do not use.",
        FutureWarning,
        stacklevel=stacklevel,
    )


# the scalars that _warn_scalar_over_non_tensor warns about
_SCALAR_TYPES = (bool, int, float, np.bool_, np.integer, np.floating)


def _warn_scalar_over_non_tensor(
    key: str, value: bool | int | float | np.bool_ | np.integer | np.floating
) -> None:
    """Warns that ``value``, written over the non-tensor entry ``key``, becomes a tensor today."""
    # point to the first frame outside tensordict: the user's call
    frame, stacklevel = sys._getframe(1), 2
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame, stacklevel = frame.f_back, stacklevel + 1
    warnings.warn(
        f"The {type(value).__name__} written over the NonTensorData entry "
        f"{key!r} is stored as a 0-d tensor today. Starting with TensorDict "
        "0.17, the default will change to storing it as NonTensorData, like "
        "the entry it replaces. Write torch.tensor(value) to keep the tensor, "
        "or NonTensorData(value) to keep the non-tensor entry now.",
        FutureWarning,
        stacklevel=stacklevel,
    )


# the numbers that _warn_tuple_dtype warns about, in tuples and nested tuples;
# exact types, since NumPy scalars such as np.float64 subclass float
_NUMBER_TYPES = frozenset((bool, int, float, complex))


def _is_tuple_of_numbers(value: tuple) -> bool:
    # a flat tuple is checked at C speed; nested tuples item by item
    types = set(map(type, value))
    if types <= _NUMBER_TYPES:
        return True
    if not any(issubclass(cls, tuple) for cls in types):
        return False
    return all(
        type(item) in _NUMBER_TYPES
        or (isinstance(item, tuple) and _is_nested_tuple_of_numbers(item))
        for item in value
    )


def _is_nested_tuple_of_numbers(value: tuple) -> bool:
    return all(
        type(item) in _NUMBER_TYPES
        or (isinstance(item, tuple) and _is_nested_tuple_of_numbers(item))
        for item in value
    )


def _warn_tuple_dtype(value: tuple, array: np.ndarray) -> None:
    """Warns that the tuple of numbers ``value``, stored as ``array``, will get the dtype of ``torch.tensor(value)``.

    NumPy stores a float or complex tuple of Python numbers as float64 or
    complex128, and ``torch.tensor`` as the default dtype or its complex
    counterpart: they differ exactly when the default dtype is not float64.
    """
    if is_compiling():
        # Dynamo can trace neither the frame walk nor warnings.warn
        return
    new = torch.get_default_dtype()
    if new == torch.float64 or not _is_tuple_of_numbers(value):
        return
    old = torch.as_tensor(array).dtype
    if old.is_complex:
        # torch has no complex dtype for a bfloat16 default
        new = {torch.float16: torch.complex32, torch.float32: torch.complex64}.get(new)
        if new is None:
            return
    # point to the first frame outside tensordict: the user's call
    frame, stacklevel = sys._getframe(1), 2
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame, stacklevel = frame.f_back, stacklevel + 1
    warnings.warn(
        f"A tuple of numbers is stored as a {old} tensor today. Starting with "
        "TensorDict 0.16, the default will change to the tensor that "
        f"torch.tensor(value) gives ({new}), as for a list of numbers. Pass "
        f"torch.tensor(value) to get {new} now, or "
        f"torch.tensor(value, dtype={old}) or a NumPy array to keep {old}.",
        FutureWarning,
        stacklevel=stacklevel,
    )


# the values that set_at_ casts to a tensor in a non-tensor entry, and warns about
_SET_AT_CAST_TYPES = (list, *_SCALAR_TYPES)


def _warn_set_at_cast(
    key: str | None,
    value: list | bool | int | float | np.bool_ | np.integer | np.floating,
) -> None:
    """Warns that ``value``, written at an index of a non-tensor entry, becomes a tensor today.

    ``key`` is the entry for a direct ``set_at_``, and ``None`` for
    ``td[index] = value``, which announces its own default change.
    """
    if is_compiling():
        # Dynamo can trace neither the frame walk nor warnings.warn
        return
    # point to the first frame outside tensordict: the user's call
    frame, stacklevel = sys._getframe(1), 2
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame, stacklevel = frame.f_back, stacklevel + 1
    if key is None:
        message = (
            f"The {type(value).__name__} written with [index] = value is cast "
            "to a tensor in the non-tensor entries today. Starting with "
            "TensorDict 0.17, the default will change to leaving the non-tensor "
            "entries unchanged, as a TensorDictStore does for a number. Write "
            "into td.exclude(<non-tensor keys>)[index] to leave them unchanged "
            "now."
        )
    else:
        message = (
            f"The {type(value).__name__} written at an index of the non-tensor "
            f"entry {key!r} is cast to a tensor today. Starting with TensorDict "
            "0.17, the default will change to storing it as non-tensor data. "
            "Write torch.tensor(value) to keep the tensor, or "
            "NonTensorData(value) to store it as non-tensor data now."
        )
    warnings.warn(message, FutureWarning, stacklevel=stacklevel)


def _set_at_each_key(
    td: TensorDictBase, value: Any, index: IndexType, *, warn_cast: bool = True
) -> None:
    """Writes ``value`` at ``index`` of every entry of ``td``: the write step of ``td[index] = value``.

    ``value`` is not a tensor collection, and ``index`` has been read by
    ``__setitem__`` (no Ellipsis left; ``None`` may remain). Every entry is
    written: a tensor entry casts the value to its dtype, a nested tensordict
    writes it into each of its entries, and a non-tensor entry casts a list or
    a number to a tensor. No entry is skipped.

    When ``warn_cast`` is ``True`` and the call casts a list or a number into a
    non-tensor entry, at any depth, it emits one :class:`FutureWarning` at the
    user's line, before the writes: from TensorDict 0.17 the non-tensor entries
    are left unchanged. It decides from the type of ``value`` and the types of
    the entries, never from tensor data. It does not warn under
    ``torch.compile``, nor for a shared or memmap container, which refuses
    these writes. The writes of the entries do not warn: the nested calls get
    ``warn_cast=False``. Containers whose ``_set_at_str`` does not take
    ``warn_cast`` (``_set_at_warns_cast`` is ``False``) write each entry with
    ``set_at_`` and do not warn.
    """
    if not type(td)._set_at_warns_cast:
        # these containers never warn of the cast
        for key in td.keys():
            td.set_at_(key, value, index)
        return
    if (
        warn_cast
        and isinstance(value, _SET_AT_CAST_TYPES)
        and td._has_non_tensor
        and _set_at_casts(td, value)
    ):
        _warn_set_at_cast(None, value)
    if not isinstance(index, (int, slice, Tensor)):
        # a lazy stack passes the index on as it is
        index = _entry_index(index)
    for key in td.keys():
        td._set_at_str(
            key, value, index, validated=False, non_blocking=False, warn_cast=False
        )


def _empty_misfit(
    td: TensorDictBase, batch_size: torch.Size
) -> tuple[NestedKey, TensorDictBase] | None:
    """The first ``(key, entry)`` of ``td`` whose batch size ``empty(recurse=True, batch_size=batch_size)`` cannot set, or ``None``.

    The batch_size setter gives the new batch size to the shorter
    sub-tensordicts and skips the empty ones of other shapes: an entry fits if
    its batch size and ``batch_size`` agree on the dims that they both have.
    """
    for key, value in td.items(True):
        ndim = min(value.batch_dims, len(batch_size))
        if value.batch_size[:ndim] != batch_size[:ndim]:
            return key, value
    return None


def _empty_leaves(td: TensorDictBase) -> list[NestedKey]:
    """The keys of the leaves that ``empty(recurse=True)`` drops from ``td``.

    They are those of ``keys(True, True)``, and in a nested lazy stack those of
    every member: ``keys()`` lists only the keys that all members hold (#2840).
    """
    leaves = {}
    _add_empty_leaves(td, None, leaves)
    return list(leaves)


def _add_empty_leaves(td: TensorDictBase, prefix: tuple | None, leaves: dict) -> None:
    """Adds the keys of the leaves of ``td`` under ``prefix`` to ``leaves``, for :func:`_empty_leaves`."""
    # A module-level function, not a closure: a recursive closure refers to
    # itself through its cell, a reference cycle that every call would leave
    # for the garbage collector.
    from tensordict._td import TensorDict

    if _is_tensorclass(type(td)):
        td = td._tensordict
    if type(td) is not TensorDict:
        from tensordict._lazy import LazyStackedTensorDict
        from tensordict.typedtensordict import TypedTensorDict

        # these hold their entries in the tensordict that they wrap
        while isinstance(td, TypedTensorDict) or _is_tensorclass(type(td)):
            td = td._source if isinstance(td, TypedTensorDict) else td._tensordict
        if isinstance(td, LazyStackedTensorDict):
            for member in td.tensordicts:
                _add_empty_leaves(member, prefix, leaves)
            return
        if not isinstance(td, TensorDict):
            for key in td.keys(True, True):
                if prefix is not None:
                    key = prefix + (key if isinstance(key, tuple) else (key,))
                leaves[key] = None
            return
    # as keys(True, True)
    for key, value in td._tensordict.items():
        cls = type(value)
        full_key = key if prefix is None else prefix + (key,)
        if cls is Tensor:
            leaves[full_key] = None
            continue
        if cls is TensorDict or (
            _is_tensor_collection(cls) and not _pass_through(value)
        ):
            _add_empty_leaves(value, (key,) if prefix is None else full_key, leaves)
        elif _default_is_leaf(cls):
            leaves[full_key] = None


def _warn_empty_names(
    names: Sequence[str | None],
    batch_size: Sequence[int],
    result_names: list,
    result_batch_size: Sequence[int] | None = None,
) -> None:
    """Warns that ``empty(batch_size=batch_size, names=names)``, which returns ``result_names`` and ``result_batch_size``, changes in 0.17 (#2711).

    From 0.17, ``empty()`` applies ``names`` as the names setter does, on a
    result of batch size ``batch_size``. This warns when the setter raises for
    them, or gives other names than ``result_names``, or when the result has
    kept another batch size for them. ``result_batch_size=None`` means
    ``batch_size``.
    """
    if is_compiling() or getattr(_EMPTY_NAMES_CALL, "running", False):
        return
    # the checks of the names setter, in its order
    value = list(names)
    batch_dims = len(batch_size)
    # a lazy stack keeps the batch size of its members for names of another
    # length; for names of the right length, its own warning tells why
    resized = (
        result_batch_size is not None
        and list(result_batch_size) != list(batch_size)
        and len(value) != batch_dims
    )
    num_none = 0
    for v in value:
        num_none += v is None
    if num_none == batch_dims:
        if list(result_names) == [None] * batch_dims and not resized:
            return
        future = "drops these names"
    elif len(value) == batch_dims and len(set(value)) == len(value) - max(
        num_none - 1, 0
    ):
        return
    else:
        future = "raises a ValueError for these names"
    # point to the first frame outside tensordict: the user's call
    frame, stacklevel = sys._getframe(1), 2
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame, stacklevel = frame.f_back, stacklevel + 1
    result = f"the names {list(result_names)}"
    future_size = ""
    if resized:
        result = f"the batch size {list(result_batch_size)} and {result}"
        future_size = f"returning the batch size {list(batch_size)} and "
    warnings.warn(
        f"empty(batch_size={list(batch_size)}, names={value}) returns a "
        f"tensordict with {result}. Starting with TensorDict 0.17, the default "
        f"will change to {future_size}applying the names as the names setter "
        f"does, which {future}. Pass one name per batch dim, "
        "with no name used twice.",
        FutureWarning,
        stacklevel=stacklevel,
    )


def _set_at_casts(td: TensorDictBase, value: Any) -> bool:
    """Whether writing the list or number ``value`` at an index of every entry of ``td``, which has a non-tensor entry, casts it to a tensor there."""
    if is_compiling() or td._is_shared or td._is_memmap:
        return False
    if isinstance(value, list):
        # a list of strings, for instance, is stored as non-tensor data
        return not is_tensor_collection(td._validate_value(value, check_shape=False))
    return True


def _members_hold_leaf(members: list[TensorDictBase]) -> bool:
    """Whether a member of a lazy stack holds a leaf, counting non-tensor data and the members of the lazy stacks it holds (#2807)."""
    for member in members:
        for _ in member.values(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS_NONTENSOR):
            return True
    return False


def _in_dense_is_empty() -> bool:
    """Whether the calling ``is_empty()`` runs for a ``TensorDict.is_empty`` call, which warns once, if its own result is ``True`` (#2807)."""
    frame = sys._getframe(2)
    while frame is not None and frame.f_code.co_filename.startswith(_LIBRARY_DIRS):
        if (
            frame.f_code.co_name == "is_empty"
            and frame.f_globals.get("__name__") == "tensordict._td"
        ):
            return True
        frame = frame.f_back
    return False


def _warn_lazy_is_empty() -> None:
    """Warns that ``is_empty()`` returns ``True`` for a lazy stack whose members hold leaves, until 0.17 (#2807)."""
    warnings.warn(
        "is_empty() returns True for a lazy stack that lists no leaf, while "
        "some of its members hold leaves. Starting with TensorDict 0.17, the "
        "default will change to returning True only if every member is empty.",
        FutureWarning,
        stacklevel=_user_stacklevel(),
    )


# Keywords that apply() and named_apply() take through **constructor_kwargs
# and do not warn about: _apply_nest honours the first four, and checked (like
# named in named_apply) collides with the value they pass themselves, so the
# call raises a TypeError.
_APPLY_NEST_KWARGS = frozenset(
    {"named", "nested_keys", "prefix", "call_on_unbatched", "checked"}
)


def _warn_ignored_kwargs(method: str, kwargs: Iterable[str]) -> None:
    """Warns that ``method`` ignores the keyword arguments ``kwargs``, which raise a TypeError from 0.17 (#2979)."""
    names = ", ".join(repr(name) for name in kwargs)
    warnings.warn(
        f"{method}() ignores the keyword argument(s) {names}. Starting with "
        "TensorDict 0.17, the default will change to raising a TypeError for "
        "unknown keyword arguments.",
        FutureWarning,
        stacklevel=_user_stacklevel(),
    )


def _warn_as_tensor_memmap() -> None:
    """Warns that ``as_tensor()`` keeps :class:`~tensordict.MemoryMappedTensor` leaves, until 0.17 (#2979)."""
    if is_compiling():
        return
    warnings.warn(
        "as_tensor() keeps the MemoryMappedTensor leaves of the tensordict as "
        "they are. Starting with TensorDict 0.17, the default will change to "
        "returning them as plain torch.Tensor over the same storage, with "
        "as_subclass(torch.Tensor).",
        FutureWarning,
        stacklevel=_user_stacklevel(),
    )


def _warn_split_keys_kept_in_last() -> None:
    """Warns that ``split_keys(strict=False)`` keeps a named entry that only some members of a lazy stack hold in the last part, until 0.17 (#2827)."""
    if is_compiling():
        return
    warnings.warn(
        "split_keys(strict=False) leaves an entry that only some members of a "
        "lazy stack hold in the last part, although a key set names it. "
        "Starting with TensorDict 0.17, the default will change to putting it "
        "in the part of that key set, in the members that hold it, as with "
        "strict=True.",
        FutureWarning,
        stacklevel=_user_stacklevel(),
    )


def _warn_update_batch_size_prefix(dest_batch_size, source_batch_size) -> None:
    """Warns that ``update(update_batch_size=True)`` keeps a batch size that is a prefix of the source's, until 0.17 (#2957)."""
    if is_compiling():
        return
    warnings.warn(
        "update(update_batch_size=True) got a source with batch size "
        f"{source_batch_size} for a destination with batch size "
        f"{dest_batch_size}, a prefix of it, and keeps the destination's batch "
        "size. Starting with TensorDict 0.17, the default will change to "
        "giving the destination the source's batch size, as for any other "
        "mismatch, which raises if the destination has keys that the source "
        "lacks, or with inplace=True or keys_to_update.",
        FutureWarning,
        stacklevel=_user_stacklevel(),
    )


def _warn_get_kwargs(
    td: TensorDictBase, kwargs: dict[str, Any], result: Any, default: Any
) -> None:
    """Warns that ``td`` ignored ``as_list=True`` or ``as_nested_tensor=True``, which a lazy stack honors."""
    if (
        isinstance(result, torch.Tensor)
        and result is not default
        and td.batch_dims
        and not (kwargs.get("as_nested_tensor") and result.is_nested)
    ):
        if kwargs.get("as_list"):
            name, future = "as_list", "list(tensor.unbind(0))"
        else:
            name, future = (
                "as_nested_tensor",
                "torch.nested.as_nested_tensor(list(tensor.unbind(0)), layout=layout)",
            )
        message = (
            f"{type(td).__name__}.get(key, {name}=True) returns the stored "
            "tensor today. Starting with TensorDict 0.17, the default will "
            f"change to returning {future}, as a LazyStackedTensorDict does. "
            f"Drop {name}=True to keep the tensor."
        )
    else:
        return
    # point to the first frame outside tensordict: the user's call
    frame, stacklevel = sys._getframe(1), 2
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame, stacklevel = frame.f_back, stacklevel + 1
    warnings.warn(message, FutureWarning, stacklevel=stacklevel)


def _unwrap(td: TensorDictBase) -> TensorDictBase:
    """The tensordict that ``td`` holds its entries in: the one that a tensorclass or a TypedTensorDict wraps, else ``td``."""
    while True:
        if _is_tensorclass(type(td)):
            td = td._tensordict
        elif getattr(type(td), "_wrapped_attr", None) is not None:
            # a TypedTensorDict
            td = getattr(td, type(td)._wrapped_attr)
        else:
            return td


def _has_lazy_stack(td: TensorDictBase) -> bool:
    """Whether ``td`` or one of its nested tensordicts is a lazy stack, which ``_apply_nest`` splits.

    No data of a file or a store is read: they cannot hold a lazy stack.
    """
    from tensordict._lazy import LazyStackedTensorDict

    # these apply their functions through the tensordict that they wrap
    td = _unwrap(td)
    if isinstance(td, LazyStackedTensorDict):
        return True
    if td._reads_return_copies:
        return False
    return any(
        _has_lazy_stack(value)
        for value in td.values()
        if _is_tensor_collection(type(value)) and not _is_non_tensor(type(value))
    )


def _has_nested_lazy_stack(td: TensorDictBase) -> bool:
    """Whether a tensordict below the root of ``td``, or below the members of a lazy stack ``td``, is a lazy stack."""
    from tensordict._lazy import LazyStackedTensorDict

    td = _unwrap(td)
    if isinstance(td, LazyStackedTensorDict):
        return any(_has_lazy_stack(member) for member in td.tensordicts)
    if td._reads_return_copies:
        return False
    return any(
        _has_lazy_stack(value)
        for value in td.values()
        if _is_tensor_collection(type(value)) and not _is_non_tensor(type(value))
    )


def _all_leaf_keys(td: TensorDictBase) -> list:
    """The keys of the leaves of ``td``, non-tensor ones included, as tuples.

    Unlike ``keys(True, True, is_leaf=_is_leaf_nontensor)``, they include the
    leaves that only some members of a lazy stack hold (#2803).
    """
    from tensordict._lazy import LazyStackedTensorDict
    from tensordict._td import TensorDict

    keys = {}
    nodes = [((), td)]
    while nodes:
        prefix, td = nodes.pop()
        if type(td) is TensorDict:
            # the common case, read without the checks below
            items = td._tensordict.items()
        else:
            # these hold their entries in the tensordict that they wrap
            td = _unwrap(td)
            if isinstance(td, LazyStackedTensorDict):
                nodes.extend((prefix, member) for member in td.tensordicts)
                continue
            items = td.items()
        for key, value in items:
            cls = type(value)
            if cls is TensorDict or (
                _is_tensor_collection(cls) and not _is_non_tensor(cls)
            ):
                nodes.append(((*prefix, key), value))
            else:
                keys[(*prefix, key)] = None
    return list(keys)


def _lazy_on_path(td: TensorDictBase, key: tuple) -> bool:
    """Whether ``td`` or a tensordict that ``key`` runs through in it is lazy, as a lazy stack is: a cheap test before ``_held_by_member``."""
    if type(td)._wrapped_attr is not None:
        td = _unwrap(td)
    for name in key[:-1]:
        if td._lazy:
            return True
        td = td._get_str(name, None)
        if not _is_tensor_collection(type(td)) or _is_non_tensor(type(td)):
            return False
        td = _unwrap(td)
    return td._lazy


def _held_by_member(td: TensorDictBase, key: tuple) -> bool:
    """Whether ``td`` holds ``key``, in some member of each lazy stack that ``key`` runs through."""
    from tensordict._lazy import LazyStackedTensorDict
    from tensordict._td import TensorDict

    # each node is a tensordict that holds key[:i], and the next name
    nodes = [(0, td)]
    while nodes:
        i, td = nodes.pop()
        if i == len(key):
            return True
        if type(td) is TensorDict:
            # the common case, read without the checks below
            value = td._tensordict.get(key[i])
        elif not _is_tensor_collection(type(td)) or _is_non_tensor(type(td)):
            continue
        else:
            td = _unwrap(td)
            if isinstance(td, LazyStackedTensorDict):
                nodes.extend((i, member) for member in td.tensordicts)
                continue
            value = td._get_str(key[i], None)
        if value is not None:
            nodes.append((i + 1, value))
    return False


def _split_keys_by_member(
    td: TensorDictBase, key_sets: list, inplace: bool, reproduce_struct: bool
) -> list:
    """``td.split_keys(*key_sets)``, splitting each member of a lazy stack in ``td`` on its own (#2827).

    An entry that only some members hold goes to the part whose key set
    names it, in those members. ``key_sets`` are dicts of key tuples.
    """
    from tensordict._lazy import LazyStackedTensorDict
    from tensordict.typedtensordict import TypedTensorDict

    if _is_tensorclass(type(td)) or isinstance(td, TypedTensorDict):
        # split the tensordict that it wraps and wrap the parts, as
        # td.split_keys does
        inner = td._source if isinstance(td, TypedTensorDict) else td._tensordict
        parts = _split_keys_by_member(inner, key_sets, inplace, reproduce_struct)
        if isinstance(td, TypedTensorDict):
            parts = [type(td)._wrap_td(part) for part in parts]
        else:
            parts = [
                td._from_tensordict(part, dict(td._non_tensordict), safe=False)
                for part in parts
            ]
        if inplace:
            parts[-1] = td
        return parts
    if isinstance(td, LazyStackedTensorDict):
        members = zip(
            *(
                _split_keys_by_member(member, key_sets, inplace, reproduce_struct)
                for member in td.tensordicts
            )
        )
        parts = [LazyStackedTensorDict(*m, stack_dim=td.stack_dim) for m in members]
        if inplace:
            parts[-1] = td
        return parts
    # a key in a tensordict that holds a lazy stack is split in it
    own_sets, nested = [], {}
    for i, key_set in enumerate(key_sets):
        own_sets.append({})
        for key, out_key in key_set.items():
            value = td._get_str(key[0], None) if len(key) > 1 else None
            if not (_is_tensor_collection(type(value)) and _has_lazy_stack(value)):
                own_sets[-1][key] = out_key
            elif out_key[0] == key[0] and len(out_key) > 1:
                sets = nested.setdefault(key[0], [{} for _ in key_sets])
                sets[i][key[1:]] = out_key[1:]
            elif _held_by_member(value, key[1:]) and td.get(key, None) is None:
                raise KeyError(
                    f"split_keys cannot move {key}, which only some members of "
                    f"a lazy stack hold, to {out_key}, outside that stack."
                )
            else:
                own_sets[-1][key] = out_key
    parts = td.split_keys(
        *own_sets, inplace=inplace, strict=False, reproduce_struct=reproduce_struct
    )
    for name, sets in nested.items():
        # the last part lacks the keys that own_sets moved out of name
        sub_parts = _split_keys_by_member(
            parts[-1].get(name), sets, inplace, reproduce_struct
        )
        for part, sub_part in zip(parts, sub_parts):
            if part.get(name, None) is sub_part:
                # inplace: the last part is td, split in place
                pass
            elif reproduce_struct or _all_leaf_keys(sub_part):
                part.set(name, sub_part)
            else:
                part.pop(name, None)
    if inplace and not reproduce_struct:
        parts[-1].filter_empty_()
    return list(parts)


def _has_unmatched_member_key(keys: list, source: dict) -> bool:
    """Whether a key of ``keys`` that ``source`` lacks may pass through a member of a lazy stack.

    Such a key has a member index, as ``("n", "0", "x")`` (#2775).
    """
    for key in keys:
        if type(key) is tuple and key not in source and any(map(str.isdigit, key)):
            return True
    return False


def _read_at_members(td: TensorDictBase, source: TensorDictBase, key: tuple) -> Any:
    """The tensor of ``source`` for the key of ``td`` that passes through members of lazy stacks, or ``None``.

    ``key`` is listed by ``_NESTED_TENSORS_AS_LISTS``, as ``("n", "0", "x")``
    for the entry ``"x"`` of member ``0`` of the lazy stack ``td["n"]``.
    ``source`` has no such keys: its tensor at ``("n", "x")`` is read at the
    index of the member along the stack dim (#2775). A tensor that lacks the
    stack dim, or has it with another size, is broadcast to the stacked leaf
    as ``copy_`` would, and raises as that ``copy_`` does where it cannot
    (#2834). Each member holds the whole of an unbatched leaf, so such a
    tensor is broadcast to the leaf of the member. ``None`` where ``source``
    lacks the entry.
    """
    from tensordict._lazy import LazyStackedTensorDict
    from tensordict.typedtensordict import TypedTensorDict

    members = []
    for part in key:
        if _is_tensorclass(type(td)):
            # a lazy stack of tensorclasses is a tensorclass over the stack
            td = td._tensordict
        if isinstance(td, TypedTensorDict):
            # a TypedTensorDict over a lazy stack lists the leaves of its members
            td = td._source
        if isinstance(td, LazyStackedTensorDict):
            members.append((td.stack_dim, len(td.tensordicts), int(part)))
            td = td.tensordicts[int(part)]
            continue
        if not _is_tensor_collection(type(source)):
            return None
        td, source = td._get_str(part, None), source._get_str(part, None)
        if td is None:
            return None
    if not isinstance(source, Tensor):
        return None
    for i, (dim, size, index) in enumerate(members):
        if source.ndim <= dim or source.shape[dim] != size:
            # the shape of the leaf that this stack reads, as a dense
            # tensordict holds it; an unbatched leaf has no stack dim
            shape = list(td.shape)
            unbatched = _is_unbatched(td)
            if not unbatched:
                for inner_dim, inner_size, _ in reversed(members[i:]):
                    shape.insert(inner_dim, inner_size)
            try:
                source = source.expand(shape)
            except RuntimeError:
                # raise as copy_ into a dense leaf of that shape does
                source.new_empty(shape, device="meta").copy_(source.to("meta"))
                raise
            if unbatched:
                return source
        source = source.select(dim, index)
    return source


def _reads_copies_under(td: TensorDictBase, keys: list) -> bool:
    """Whether a key of ``keys`` passes through or names a nested tensordict of ``td`` whose leaves are read as copies.

    Such as a lazy stack, which stacks the leaves of its members, or an h5
    file (#2780). Only the entries on the path of each key are read, and the
    tree under a key that names a tensordict.
    """
    tensordict_type = _TENSORDICT_TYPE
    # the entries that keys name: the leaves under them are written too
    named = []
    for key in keys:
        node = td
        for part in key:
            if type(node) is tensordict_type:
                node = node._tensordict.get(part)
                continue
            if node is not td:
                # td itself writes back the copies that it reads
                copies = _entry_reads_copies(node)
                if copies:
                    return True
                if copies is None:
                    break
            node = node._get_str(part, None)
        else:
            if type(node) is not Tensor:
                named.append(node)
    while named:
        node = named.pop()
        if type(node) is not tensordict_type:
            copies = _entry_reads_copies(node)
            if copies:
                return True
            if copies is None:
                continue
        for value in node.values():
            if type(value) is not Tensor:
                named.append(value)
    return False


def _entry_reads_copies(value: Any) -> bool | None:
    """Whether the tensordict ``value`` reads its leaves as copies, or ``None`` for an entry that is not a tensordict.

    Called by ``_reads_copies_under``.
    """
    from tensordict._lazy import LazyStackedTensorDict

    cls = type(value)
    if not _is_tensor_collection(cls) or _is_non_tensor(cls):
        return None
    while _is_tensorclass(type(value)):
        value = value._tensordict
    return isinstance(value, LazyStackedTensorDict) or value._reads_return_copies


def _has_typed_tensordict(td: TensorDictBase) -> bool:
    """Whether ``td`` or one of its nested tensordicts is a ``TypedTensorDict``, which lists the leaves of the members of a lazy source.

    No data of a file or a store is read: they cannot hold one.
    """
    from tensordict.typedtensordict import TypedTensorDict

    while _is_tensorclass(type(td)):
        td = td._tensordict
    if isinstance(td, TypedTensorDict):
        return True
    if td._reads_return_copies:
        return False
    return any(
        _has_typed_tensordict(value)
        for value in td.values()
        if _is_tensor_collection(type(value)) and not _is_non_tensor(type(value))
    )


def _keys_update_pairs(
    source: TensorDictBase,
    dest: TensorDictBase | None,
    prefix: tuple,
    selected: bool,
    wanted: set,
    grad: bool,
    dests: list[Tensor],
    sources: list[Tensor],
    entries: list[tuple],
) -> int:
    """Pairs the tensors that ``update_(keys_to_update=...)`` copies from the exact ``TensorDict`` ``source`` into ``dest``.

    Walks ``source`` in the order of the ``_apply_nest`` route of
    :meth:`~TensorDictBase._update_`, and selects an entry as it does: an
    entry is selected if its key, or the key of a tensordict that holds it,
    is in ``wanted``, a set of key tuples. ``dest`` is an exact
    ``TensorDict``, or ``None`` where it lacks the node. Appends each selected
    exact tensor and the exact tensor of ``dest`` at its key to ``sources``
    and ``dests``, unless ``grad`` is set and one of them requires grad:
    ``_foreach_copy_`` records no autograd graph. Each other selected leaf,
    and each nested tensordict that is not an exact ``TensorDict`` on both
    sides, goes in ``entries`` as ``(len(dests), key, source, dest, nested)``,
    for the ``_apply_nest`` route to write with the same call, at its place
    in the order. Writes nothing. Returns the number of keys of ``wanted``
    that name an entry of ``dest`` that is not a tensordict, read through
    exact ``TensorDict`` nodes, or ``-1`` where a selected leaf, or a nested
    tensordict that is not an exact ``TensorDict``, has no destination: the
    general route then writes the whole update and raises as it does.
    """
    tensordict_type = _TENSORDICT_TYPE
    dest_entries = None if dest is None else dest._tensordict
    named = 0
    for key, value in source._tensordict.items():
        cls = type(value)
        name = (*prefix, key)
        is_named = name in wanted
        is_selected = selected or is_named
        if cls is Tensor or (cls is not tensordict_type and _is_leaf_nontensor(cls)):
            if not is_selected:
                continue
            other = None if dest_entries is None else dest_entries.get(key)
            if (
                type(other) is Tensor
                and cls is Tensor
                and not (grad and (value.requires_grad or other.requires_grad))
            ):
                dests.append(other)
                sources.append(value)
                named += is_named
                continue
            if other is None:
                return -1
            entries.append((len(dests), name, value, other, False))
            named += is_named and _is_leaf_nontensor(type(other))
            continue
        other = None if dest_entries is None else dest_entries.get(key)
        if cls is tensordict_type and (other is None or type(other) is cls):
            nested = _keys_update_pairs(
                value, other, name, is_selected, wanted, grad, dests, sources, entries
            )
            if nested < 0:
                return -1
            named += nested
        elif other is None:
            return -1
        else:
            entries.append((len(dests), name, value, other, True))
    return named


def _any_requires_grad(tensors: list[Tensor]) -> bool:
    # A loop is faster than any() over a generator for short lists.
    for tensor in tensors:
        if tensor.requires_grad:
            return True
    return False


def _copy_leaves(
    dests: list[Tensor], sources: list[Tensor], non_blocking: bool, exact: bool = False
) -> None:
    """Copies each tensor of ``sources`` into the tensor of ``dests`` at the same position, as ``update_`` does.

    ``exact`` tells that every tensor is an exact ``Tensor``: none is
    unbatched. A single pair is then copied with ``copy_``.

    ``torch._foreach_copy_`` records no autograd graph, and marks every
    ``dest`` as requiring grad: where grad mode is on and a tensor requires
    grad, each pair is copied with ``copy_``, as on the other routes of
    ``update_`` (#2863). So is each pair inside a forward-AD dual level, as
    ``_foreach_copy_`` has no forward derivative (#2886). A ``dest`` that is
    a leaf that requires grad is left to ``_foreach_copy_``, or to ``copy_``
    for a single exact pair: both raise for it before they copy anything.
    """
    if exact and len(dests) == 1:
        # one copy_ costs less than the checks below and a _foreach_copy_
        dests[0].copy_(sources[0], non_blocking=non_blocking)
    # _foreach_copy_compiled cannot run under a torch.func transform
    # (vmap), _foreach can. Same check as in _foreach, which explains it.
    elif torch._C._are_functorch_transforms_active():
        _foreach("copy_", dests, sources, non_blocking=non_blocking)
    elif (
        torch.is_grad_enabled()
        and (_any_requires_grad(dests) or _any_requires_grad(sources))
        and not any(dest.requires_grad and dest.is_leaf for dest in dests)
    ):
        for dest, source in zip(dests, sources):
            dest.copy_(source, non_blocking=non_blocking)
    elif is_compiling():
        _foreach_copy_compiled(dests, sources, non_blocking=non_blocking)
    elif forward_ad._current_level >= 0 and not any(
        dest.requires_grad and dest.is_leaf for dest in dests
    ):
        # inside a dual level (after the compile test: no guard on it)
        for dest, source in zip(dests, sources):
            dest.copy_(source, non_blocking=non_blocking)
    elif not exact and (_any_unbatched(dests) or _any_unbatched(sources)):
        _foreach("copy_", dests, sources, non_blocking=non_blocking)
    else:
        # what _foreach does for copy_ outside of a torch.func
        # transform and without unbatched tensors, without its
        # overhead (about 2 µs, as much as the copy of 2 leaves)
        torch._foreach_copy_(dests, sources, non_blocking=non_blocking)


def _warn_update_skipped(skipped: list) -> None:
    """Warns that ``update_`` skipped the non-tensor entries under the keys ``skipped``."""
    # point to the first frame outside tensordict: the user's call
    frame, stacklevel = sys._getframe(1), 2
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame, stacklevel = frame.f_back, stacklevel + 1
    warnings.warn(
        f"update_() skipped the non-tensor entries {skipped} of the source: "
        "the shared or memory-mapped destination cannot store their values.",
        UserWarning,
        stacklevel=stacklevel,
    )


class _NestedReadsCopies(Exception):
    """Raised by a lazy stack whose members read copies when an in-place method lists the leaves of a tensordict that holds it.

    See ``_raise_if_inplace_call``.
    """


# Set while an in-place method runs on copies that are written back, so that
# the lazy stacks it lists do not raise _NestedReadsCopies.
_ON_COPIES_CALL = local()


def _on_copies(fn: Callable, *args, **kwargs) -> Any:
    """Calls ``fn``, with no ``_NestedReadsCopies`` raised by the lazy stacks that it lists."""
    running = getattr(_ON_COPIES_CALL, "running", False)
    _ON_COPIES_CALL.running = True
    try:
        return fn(*args, **kwargs)
    finally:
        _ON_COPIES_CALL.running = running


def _raise_if_inplace_call() -> None:
    """Raises ``_NestedReadsCopies`` if an in-place method changes the leaves being listed.

    Called by a lazy stack whose members read copies, such as h5 files, when
    a tensordict that holds it lists its leaves (#2592), and by a tensordict
    that reads copies, such as a file, which a tensordict holds (#2775). The
    in-place method would change copies: ``_write_back_inplace`` catches the
    error and runs the method again through ``_inplace_on_nested_copies``, as
    ``TensorDictBase._update_`` does. The frames are only searched here, so
    that the in-place methods of the tensordicts that hold no such stack pay
    nothing.
    """
    if getattr(_ON_COPIES_CALL, "running", False):
        return
    frame = sys._getframe(1)
    while frame is not None:
        code = frame.f_code
        if (
            code is _WRITE_BACK_INPLACE_CODE
            or code is _UPDATE_INPLACE_CODE
            or code is _UPDATE_PLAN_CODE
        ):
            raise _NestedReadsCopies
        frame = frame.f_back


def _write_back_inplace(method: Callable) -> Callable:
    """Makes an in-place method also write its result where reads return copies.

    The in-place methods change the tensor leaves that ``_items_list(True,
    True)`` returns. On a class that sets ``_reads_return_copies``, such as a
    file or a store, these leaves are copies of the stored data, and the
    method runs through ``_inplace_on_copies`` instead. Where only a nested
    lazy stack reads copies, it raises ``_NestedReadsCopies`` before the
    method changes anything, and the method runs through
    ``_inplace_on_nested_copies``.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self._reads_return_copies:
            try:
                return method(self, *args, **kwargs)
            except _NestedReadsCopies:
                pass
            return _on_copies(self._inplace_on_nested_copies, method, *args, **kwargs)
        return _on_copies(self._inplace_on_copies, method, *args, **kwargs)

    return wrapper


# the frame of an in-place method, which _raise_if_inplace_call looks for
_WRITE_BACK_INPLACE_CODE = _write_back_inplace(lambda self: self).__code__


def _write_back_members(written: list, method: str) -> None:
    """Writes back the copies of the leaves that the in-place ``method`` changed.

    ``written`` holds ``(td, leaves)`` pairs: ``leaves`` is the changed copy
    of the leaves of ``td``, which reads copies. A ``td`` opened read-only is
    left unchanged, with one warning for the call, as for one file.
    """
    read_only = False
    for td, leaves in written:
        if td._storage_is_read_only():
            read_only = True
        else:
            td._write_back(*leaves._items_list(True, True), method)
    if read_only:
        _warn_read_only(method)


def _expand_to_match_shape(
    parent_batch_size: torch.Size,
    data: Tensor | TensorDictBase,
    self_batch_dims: int,
    self_device: DeviceType,
    index: Any = None,
) -> Tensor | TensorDictBase:
    """Creates and empty tensor / tensordict that can host values.

    Given a tensordict with shape ``parent_batch_size``, this function creates an expanded version
    of ``data`` such that ``data_expand[index].shape == data.shape``.

    """
    if not parent_batch_size and self_batch_dims == 1:
        # This is what happens when indexing an empty tensor with a bool:
        #  torch.zeros(())[True].shape == torch.Size((1,))
        return data.new_zeros(data.shape[1:])
    if not _is_tensor_collection(type(data)):
        result = torch.zeros(
            (
                *parent_batch_size,
                *_shape(data)[self_batch_dims:],
            ),
            dtype=data.dtype,
            device=self_device,
        )
    else:
        # tensordict
        batch_size = torch.Size([*parent_batch_size, *_shape(data)[self_batch_dims:]])
        if is_non_tensor(data):
            from tensordict.tensorclass import NonTensorData, NonTensorStack

            if isinstance(data, NonTensorStack):
                # The rows that are not written hold None, as a tensor gets
                #  zeros (a NonTensorData keeps its one value for all rows)
                return NonTensorData(None, batch_size=batch_size, device=self_device)
        result = data.empty(batch_size=batch_size)
    return result


def _batch_mismatch_error(
    batch_size: torch.Size, value: Any, key: NestedKey | None
) -> RuntimeError:
    """Builds the error raised when a value does not start with the batch size."""
    shape = _shape(value)
    msg = (
        f"batch dimension mismatch, got self.batch_size={batch_size} and "
        f"value.shape={shape}"
    )
    if key is not None:
        msg += f" for key {key!r}"
    msg += ". The leading dimensions of a value must match the batch size."
    if not shape:
        msg += (
            " Python scalars are stored as 0-dim tensors: to store a scalar in a "
            "tensordict with a non-empty batch size, expand it to the batch size or "
            "wrap it in NonTensorData (in a tensorclass, the nocast option stores "
            "scalars as they are)."
        )
    return RuntimeError(msg)


# TensorDictBase's methods are grouped by area of the API into mixins under
# tensordict/_base/. Those modules import the helpers above from this module,
# so they are imported here, after the helpers and before the class. Hence:
# - A helper that a mixin imports must be defined above this point.
# - In a mixin, TensorDictBase is imported for type checking only. Once the
#   class exists, this module binds it into each mixin module, so that
#   annotations that name it resolve. Code that runs at import time, such as
#   decorators and default values, cannot use it. A method that uses it at
#   run time imports it locally (ruff's TC004 flags a run-time use of the
#   TYPE_CHECKING import).
# - Globals that this module rebinds at run time, such as
#   _GET_DEFAULTS_TO_NONE and _ACCEPTED_CLASSES, must be read as
#   tensordict.base.<name>: a mixin that imports one keeps its first value.
# - Abstract methods, properties, implement_for overloads and methods that
#   read a rebound global stay in the class below.
from tensordict._base.convert import _Conversion  # noqa: E402
from tensordict._base.device import _DeviceOps  # noqa: E402
from tensordict._base.distributed import _Distributed  # noqa: E402
from tensordict._base.pointwise import _PointwiseOps  # noqa: E402
from tensordict._base.reductions import _Reductions  # noqa: E402
from tensordict._base.serialization import _Serialization  # noqa: E402
from tensordict._base.shape import _ShapeOps  # noqa: E402

_TENSORDICTBASE_MIXINS = (
    _PointwiseOps,
    _Reductions,
    _ShapeOps,
    _DeviceOps,
    _Serialization,
    _Conversion,
    _Distributed,
)


def _value_at_new_dim(td: TensorDictBase, value):
    """Return what ``td[None] = value`` and ``td[True] = value`` write to every element of ``td``.

    ``None`` and ``True`` add a dim of size 1 in front of the batch dims. The
    value is broadcast to that batch size, and its only element is written.
    """
    batch_size = torch.Size([1, *td.batch_size])
    if isinstance(value, dict):
        value = td.from_dict_instance(value, batch_size=batch_size, device=td.device)
    return value.expand(batch_size)[0]


def _flatten_nested_dicts(source: dict) -> dict:
    """Return ``source`` with the entries of its nested dicts under nested keys."""
    result = {}
    for key, value in source.items():
        if isinstance(value, dict):
            for subkey, subvalue in _flatten_nested_dicts(value).items():
                result[(key, subkey)] = subvalue
        else:
            result[key] = value
    return result


def _select_entry(key: tuple, value: Any, keys_to_update: list[tuple]) -> Any:
    """Return the part of the entry ``key: value`` that ``keys_to_update`` selects, or ``None``.

    As in ``update_``, the leaves to update are those whose key starts with one
    of ``keys_to_update``. The entry is kept whole if one of them is a prefix of
    ``key``, and a tensor collection keeps only the leaves to update.
    """
    subkeys = []
    for key_to_update in keys_to_update:
        if key[: len(key_to_update)] == key_to_update:
            return value
        if key_to_update[: len(key)] == key:
            subkeys.append(key_to_update[len(key) :])
    if not subkeys or not _is_tensor_collection(type(value)):
        return None
    leaves = [
        leaf
        for leaf in value.keys(True, True, is_leaf=_is_leaf_nontensor)
        if any(
            _unravel_key_to_tuple(leaf)[: len(subkey)] == subkey for subkey in subkeys
        )
    ]
    if not leaves:
        return None
    try:
        return value.select(*leaves)
    except NotImplementedError:
        # PersistentTensorDict and the stores have no select
        return value.to_tensordict().select(*leaves)


def _stored_value(source, dest):
    """The source that ``dest.copy_`` writes for ``source``, or ``None`` if ``dest`` is a shared or memory-mapped non-tensor leaf that cannot store its values."""
    if not (_is_non_tensor(type(dest)) and (dest._is_shared or dest._is_memmap)):
        return source
    from tensordict.tensorclass import _stored_non_tensor

    return _stored_non_tensor(dest, source)


def _update_at_aliases(update_at_: Callable) -> Callable:
    """Accepts the argument names that the overrides of ``update_at_`` took before 0.15.

    ``LazyStackedTensorDict``, ``NonTensorStack``, ``NonTensorData`` and the
    tensorclass ``update_at_`` called the index ``index``. ``_SubTensorDict``
    called the input ``input_dict``, and took
    ``discard_idx_attr=True`` to write into its source at an index of the
    source: classes with a ``_update_source_at_`` method still take it.
    Each of these names warns, and goes in 0.17.
    """

    @wraps(update_at_)
    def wrapper(self, *args, **kwargs):
        if kwargs:
            if "discard_idx_attr" in kwargs and hasattr(
                type(self), "_update_source_at_"
            ):
                warn_deprecated(
                    "update_at_(discard_idx_attr=...)",
                    removal="0.17",
                    replacement="get_parent_tensordict().update_at_(...)",
                )
                if kwargs.pop("discard_idx_attr"):
                    return self._update_source_at_(*args, **kwargs)
            if "index" in kwargs and len(args) < 2:
                warn_deprecated(
                    "update_at_(index=...)",
                    removal="0.17",
                    replacement="update_at_(idx=...)",
                )
                kwargs["idx"] = kwargs.pop("index")
            if "input_dict" in kwargs and not args:
                warn_deprecated(
                    "update_at_(input_dict=...)",
                    removal="0.17",
                    replacement="update_at_(input_dict_or_td=...)",
                )
                kwargs["input_dict_or_td"] = kwargs.pop("input_dict")
        return update_at_(self, *args, **kwargs)

    return wrapper


class TensorDictBase(*_TENSORDICTBASE_MIXINS, MutableMapping, TensorCollection):
    """The abstract base class of every tensordict.

    :class:`~tensordict.TensorDict`, :class:`~tensordict.LazyStackedTensorDict`,
    :class:`~tensordict.PersistentTensorDict` and the other tensordict classes
    subclass it, and tensorclasses have the same methods. You do not create a
    ``TensorDictBase``: use it in type hints and in ``isinstance`` checks. A
    tensorclass is not an instance of it: use
    :func:`~tensordict.is_tensor_collection` to accept both. The methods are
    listed by task in :ref:`api-tensordict-methods`.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict, TensorDictBase, lazy_stack
        >>> td = TensorDict(a=torch.zeros(3), batch_size=[3])
        >>> isinstance(td, TensorDictBase), isinstance(lazy_stack([td, td]), TensorDictBase)
        (True, True)
    """

    _safe: bool = False
    _lazy: bool = False
    _inplace_set: bool = False
    _is_locked: bool = False
    _is_shared: bool = False
    _is_memmap: bool = False
    # whether _set_at_str takes warn_cast and may warn of a list or a number
    # cast to a tensor in a non-tensor entry (gh-2522)
    _set_at_warns_cast: bool = False
    _cache: bool | None = None
    _is_non_tensor: bool = False
    _memmap_prefix = None
    _stream: torch.cuda.Stream | None = None
    # Class-level default so `_last_op` is always readable, even on a TD
    # that has never been passed through a ``_as_context_manager``-wrapped
    # method (notably under ``torch.compile`` where that decorator
    # short-circuits and never writes the attribute).
    _last_op = None
    # Set by flatten(inplace=True): (start_dim, merged sizes), which a
    # with-block around the call reads on exit.
    _inplace_flatten_sizes = None
    # Set by flatten_keys(inplace=True): whether unflatten_keys undoes it,
    # that is, no key held the separator before. A with-block reads it on exit.
    _inplace_flatten_keys_exact = None
    # Class-level default; instances populate this on lock_() so locked-
    # fast-paths can avoid iterating ``_tensordict`` (which would emit
    # DICT_KEYS_MATCH guards under Dynamo). Cleared on unlock_().
    _locked_schema = None
    # Class-level default for the dim-name methods: no names until
    # _set_names stores them on the instance.
    _td_dim_names = None
    # Set by the backends on which ``where`` and ``masked_fill`` work on a
    # ``to_tensordict()`` copy: the stores read it in one round trip, and on
    # all of them the result is a new, unlocked TensorDict.
    _masking_on_dense_copy: bool = False
    # Set by the files, into which the exit of a ``with td.to(...)`` block
    # writes the dense copy in place, with _write_back_to().
    _to_writes_back_in_place: bool = False
    # Set by the stores, on which ``masked_select`` masks a
    # ``to_tensordict()`` copy, read in one round trip. On the views and h5
    # files, that is slower than masking each entry.
    _masked_select_on_dense_copy: bool = False
    # Set by the backends on which ``masked_select(inplace=True)`` raises (the
    # lazy stacks raise too).
    _masked_select_inplace_unsupported: bool = False
    # Whether the tensors that items() returns are copies of the stored data,
    # as for an h5 or zarr file or a store. The in-place methods then write
    # their results back (see _write_back_inplace and _write_back).
    _reads_return_copies = False
    # Whether a sub-tensordict read with _get_str is a new TensorDict filled
    # from the stored data, as in the lazy stores. update() then writes the
    # leaves it merges into the sub-tensordict back (see _update_write_back).
    _nested_reads_return_copies = False
    # Whether update({key: self}) writes a copy of the data, as the stores do:
    # they write what the source holds on the server, not the Python object,
    # so no cycle forms and update() does not raise (#2741).
    _update_stores_copy_of_self = False
    # Whether the in-place shape methods (reshape, flatten, repeat, ... with
    # inplace=True) can change the batch size after they rewrite the leaves.
    # If not, _inplace_rebind_leaves raises before it writes anything.
    _inplace_batch_size_change = True
    # Set by PersistentTensorDict and the stores, whose rename_key_ did not
    # check the lock or the type of the old key before 0.15: until 0.17 they
    # rename in a locked tensordict with a warning, and pass an old key that is
    # not a str or tuple to _rename_key (which raises or warns as before).
    _rename_key_lenient: bool = False
    # Set by PersistentTensorDict, whose file handles cannot be sent to another
    # process: map sends a to_tensordict() copy of each chunk to the workers.
    _map_on_dense_copies: bool = False
    # Whether the members of a lazy stack have different keys, which to_dict
    # and state_dict would silently drop (a property on LazyStackedTensorDict).
    _has_exclusive_keys: bool = False
    # Set by TypedTensorDict to the name of the attribute that holds the
    # tensordict it wraps, "_source": the pointwise ops see that a typed
    # operand over a lazy stack mixes with a dense receiver (#2759).
    _wrapped_attr: str | None = None
    # Set by _SubTensorDict, whose memmap_ would memory-map its whole source:
    # memmap_ raises.
    _memmap_inplace_unsupported: bool = False
    # Whether each read of a leaf builds a new tensor: a lazy stack stacks its
    # members, a file or a store reads its data. is_contiguous() then returns
    # False without reading the leaves.
    _leaves_built_on_read: bool = False
    # Set by the stores, on which contiguous() returns to_tensordict(), read
    # in one round trip, instead of reading the leaves one at a time.
    _contiguous_via_to_tensordict: bool = False
    # Set by the views and the backends whose leaves cannot be pinned in
    # place: pin_memory raises.
    _pin_memory_unsupported: bool = False
    # Set by LazyStackedTensorDict, whose leaves belong to its members:
    # reshape and repeat_interleave with inplace=True raise
    # NotImplementedError before they change anything.
    _inplace_shape_ops_unsupported: bool = False
    # Set by LazyStackedTensorDict, whose flatten regroups its members rather
    # than flattening leaves: flatten(d, d) returns self, and for
    # start_dim > end_dim it inserted a dim of size 1 before 0.15: until 0.17
    # it still does, with a FutureWarning, where the others raise a ValueError.
    _flatten_lenient: bool = False
    # Set by a TypedTensorDict over a lazy stack: gather(inplace=True) with an
    # index of the batch shape writes into the members there, and until 0.17
    # it warns, where the lazy stacks raise.
    _gather_inplace_warns: bool = False
    # Whether get() ignores as_list and as_nested_tensor, and warns about them
    # (see _warn_get_kwargs). False for lazy stacks, which honor them.
    _get_ignores_kwargs: bool = True
    # Set by the files and stores, which hold no autograd history: detach_
    # returns without reading their entries.
    _storage_holds_no_gradients: bool = False
    # Set by PersistentTensorDict, which writes a constant without reading the
    # array: zero_ fills each entry through _fill_ instead of reading copies.
    _fills_without_reading: bool = False
    # The inplace argument with which td[key] = value writes the key: False
    # replaces the entry. BEST_ATTEMPT_INPLACE (_SubTensorDict, the files and
    # stores) writes into an existing entry and adds a new one.
    _setitem_key_inplace = False

    @classmethod
    def _new_unsafe(cls, *args, **kwargs) -> "TensorDictBase":
        """Builds a tensordict from ``source`` without validating it.

        Every subclass accepts the arguments of
        :meth:`TensorDict._new_unsafe <tensordict.TensorDict._new_unsafe>`:
        ``source=None, batch_size=None, device=None, names=None,
        non_blocking=None, lock=False, nested=True, is_shared=False,
        is_memmap=False``, and passes the keywords it does not use to it. The
        base code calls it on ``self`` with a dict ``source`` for the results
        of :meth:`split`, :meth:`contiguous`, :meth:`empty`, ``_unbind`` and
        ``_index_tensordict``. The result is an instance of the class when the
        class can hold ``source``: the default returns a
        :class:`~tensordict.TensorDict`, and so does
        :class:`~tensordict.nn.TensorDictParams` for a dict.
        """
        from tensordict._td import TensorDict

        return TensorDict._new_unsafe(*args, **kwargs)

    def __bool__(self) -> bool:
        raise RuntimeError(
            "Converting a tensordict to boolean value is not permitted. "
            "Use len(td.keys()) to test for keys, td.is_empty() to test for "
            "leaves, or (a == b).all() / .any() to reduce a comparison."
        )

    def __repr__(self) -> str:
        try:
            if _REPR_OPTIONS["compact"]:
                compact = _compact_repr(self)
                if compact is not None:
                    return compact
            string = ",\n".join(self._repr_parts())
        except AttributeError:
            # When using torch.compile, an exception may be raised with a tensordict object
            #  that has no attribute (no _tensordict or no _batch_size).
            #  To get the proper erro message and not an attribute error raised during __repr__,
            #  we simply default to '...' when trying to print the TD content.
            string = "..."
        return f"{type(self).__name__}(\n{string})"

    def _repr_parts(self) -> list[str]:
        """Returns the lines that :meth:`__repr__` joins inside ``ClassName(...)``.

        Each part is indented by four spaces; ``__repr__`` separates them with
        a comma and a line break. The base implementation lists the fields, then the batch
        size, the device and ``is_shared`` as :class:`~tensordict.set_printoptions`
        selects. A class overrides it to add lines (a lazy stack's exclusive
        fields and stack dim), or so that the repr reads no data (the stores
        list their key names).
        """
        fields = _td_fields(self)
        parts = [indent(f"fields={{{fields}}}", 4 * " ")]
        if _REPR_OPTIONS["show_batch_size"]:
            parts.append(indent(f"batch_size={self.batch_size}", 4 * " "))
        if _REPR_OPTIONS["show_device"]:
            parts.append(indent(f"device={self.device}", 4 * " "))
        if _REPR_OPTIONS["show_is_shared"]:
            parts.append(indent(f"is_shared={self.is_shared()}", 4 * " "))
        return parts

    def __iter__(self) -> Iterator[TensorDictBase]:
        """Iterates over the first batch dimension of the tensordict.

        Each item is a tensordict with the remaining batch dimensions, as
        returned by ``td.unbind(0)``. To iterate over the keys, use
        :meth:`~.keys`.

        Raises:
            TypeError: if the tensordict has no batch dimensions, as ``iter()``
                does for a 0-d tensor.
        """
        # Not a generator function, so that iter(td) raises at once, as
        # Tensor.__iter__ does.
        if not self.batch_dims:
            raise TypeError(
                "iteration over a 0-d tensordict. Use keys(), values() or items() "
                "to iterate over its entries."
            )
        return iter(self.unbind(0))

    def __len__(self) -> int:
        """Returns the size of the first batch dimension, or 0 if the tensordict has no batch dimensions."""
        batch_size = self.batch_size
        if not batch_size:
            return 0
        return batch_size[0]

    def __deepcopy__(self, memo: Dict[Any, Any]) -> "tensordict.TensorDict":  # noqa  # type: ignore
        return self.clone()

    def __contains__(self, key: NestedKey) -> bool:  # type: ignore
        if isinstance(key, str):
            return self._contains_key(key)
        if isinstance(key, tuple):
            key = unravel_key(key)
            if not key:
                raise RuntimeError(
                    "key must be a NestedKey (a str or a possibly tuple of str)."
                )
            return self._contains_key(key)
        return self._contains_tensordict(key)

    def _contains_key(self, key: NestedKey) -> bool:
        """Answers ``key in self`` for a str or an unravelled tuple key.

        Called by ``__contains__`` with a str key, or with the unravelled form
        of a tuple key: a str for a tuple of one str, a tuple of two str or
        more otherwise (an empty tuple raised before the call). Returns whether
        ``key`` is in ``self.keys(True, is_leaf=_is_leaf_nontensor)``, without
        reading the leaves; the default asks that keys view (``self.keys()``
        for a str).
        """
        if isinstance(key, str):
            return key in self.keys()
        return key in self.keys(True, is_leaf=_is_leaf_nontensor)

    def _contains_tensordict(self, item: Any) -> bool:
        """Answers ``item in self`` for an ``item`` that is not a str or a tuple.

        Called by ``__contains__`` after the str and tuple checks. The default
        raises the NestedKey ``RuntimeError``; a lazy stack reports whether a
        tensordict ``item`` is one of its members (identity).
        """
        raise RuntimeError(
            "key must be a NestedKey (a str or a possibly tuple of str)."
        )

    @overload
    def __getitem__(self, index: str | tuple[str, ...]) -> Any: ...

    @overload
    def __getitem__(self, index: IndexType) -> Self: ...

    def __getitem__(self, index: IndexType) -> Self | Tensor | TensorCollection | Any:
        """Returns the entry at a key, or the tensordict indexed along its batch dimensions.

        A string or a tuple of strings reads an entry, as :meth:`~.get` does,
        except that a missing key raises ``KeyError``. A non-tensor entry
        returns its content (the string, not the
        :class:`~tensordict.NonTensorData`), and a stack of non-tensor data
        returns a list. See :ref:`keys`.

        Any other index (an int, a slice, a tensor, a boolean mask, ``...``,
        ``None``, or a tuple of these) applies to the batch dimensions of every
        entry, as it would to a tensor of shape :attr:`~.batch_size`, and
        returns a tensordict. See :ref:`batch-size`.

        Raises:
            KeyError: if the key is missing.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"root": torch.arange(2), ("nested", "entry"): torch.arange(2)}, [2])
            >>> td["root"]
            tensor([0, 1])
            >>> td["nested", "entry"]
            tensor([0, 1])
            >>> td[:1].batch_size, td[:1]["nested", "entry"]
            (torch.Size([1]), tensor([0]))
            >>> td[torch.tensor([True, False])]["root"]
            tensor([0])
        """
        istuple = isinstance(index, tuple)
        if istuple or isinstance(index, str):
            if not istuple and type(self) is _TENSORDICT_TYPE:
                # An exact TensorDict reads a str key straight from its dict,
                # as TensorDict._get_str does, and non-tensor entries as
                # _get_tuple_maybe_non_tensor returns them.
                try:
                    result = self._tensordict[index]
                except KeyError:
                    return self._default_get(index, NO_DEFAULT)
                cls = type(result)
                if cls is Tensor or cls is _TENSORDICT_TYPE:
                    return result
                if _pass_through(result) and not _is_unbatched(result):
                    return _non_tensor_content(result)
                return result
            # _unravel_key_to_tuple will return an empty tuple if the index isn't a NestedKey
            idx_unravel = _unravel_key_to_tuple(index)
            if idx_unravel:
                return self._get_tuple_maybe_non_tensor(idx_unravel, NO_DEFAULT)

        # Ellipsis is found by its type, and an empty tuple by comparison:
        # under torch.compile, an ``is`` test on a slice, or the truth value of
        # a tuple holding one, fixes the bounds of the slice.
        if (istuple and index == ()) or (not istuple and type(index) is EllipsisType):
            # empty tuple returns self
            return self
        if not istuple:
            if isinstance(index, int):
                return self._index_tensordict(index)
            # we only want tuple indices
            index = (index,)
        # # convert range/np.ndarray to tensor: this is not cheap
        # index = tuple(
        #     torch.tensor(idx) if isinstance(idx, (np.ndarray, range)) else idx
        #     for idx in index
        # )
        if istuple and any(type(idx) is EllipsisType for idx in index):
            index = convert_ellipsis_to_idx(index, self.batch_size)
        # The bounds are tested one by one: the Dynamo of torch 2.11 and 2.12
        # fixes the bounds of a slice that it compares with slice(None).
        if len(index) <= len(self.batch_size) and all(
            isinstance(idx, slice)
            and idx.start is None
            and idx.stop is None
            and idx.step is None
            for idx in index
        ):
            return self

        return self._index_tensordict(index)

    # this is necessary for data collectors for instance, otherwise indexing
    # will always be achieved one element at a time.
    __getitems__ = __getitem__

    def _get_sub_tensordict(self, idx: IndexType) -> Self:
        """Returns a _SubTensorDict with the desired index."""
        from tensordict._td import _SubTensorDict

        return _SubTensorDict(source=self, idx=idx)

    def __setitem__(
        self,
        index: IndexType,
        value: Any,
    ) -> None:
        """Writes ``value`` at a key, or at an index of every entry.

        A list or a number written at an index, as in ``td[mask] = 0``, goes
        into every entry, and is cast to a tensor in the non-tensor entries
        too. That write emits a :class:`FutureWarning`: starting with
        TensorDict 0.17, the default will change to leaving the non-tensor
        entries unchanged, as a :class:`~tensordict.store.TensorDictStore`
        does for a number. To leave them unchanged now, write into
        ``td.exclude(<non-tensor keys>)[index]``, which shares the tensors of
        ``td``.

        On a dense :class:`~tensordict.TensorDict`, when ``td[index] = src``
        brings a new non-tensor key, a :class:`~tensordict.NonTensorData`
        (one value for the whole batch) fills every row of ``td``, written or
        not, while a :class:`~tensordict.NonTensorStack` (one value per row)
        leaves ``None`` in the rows that ``index`` does not select (see
        :ref:`non-tensor-data`). A
        :class:`~tensordict.LazyStackedTensorDict` adds the key only to the
        members that ``index`` selects.
        """
        istuple = isinstance(index, tuple)
        if istuple or isinstance(index, str):
            index_unravel = _unravel_key_to_tuple(index)
            if index_unravel:
                if value is self:
                    raise ValueError(_SELF_NESTING_ERROR.format(index))
                self._set_tuple(
                    index_unravel,
                    value,
                    inplace=self._setitem_key_inplace,
                    validated=False,
                    non_blocking=False,
                )
                return
        # we must use any and because using Ellipsis in index can break with some indices
        if index is Ellipsis or (istuple and any(idx is Ellipsis for idx in index)):
            index = _expand_batch_ellipsis(index, self.batch_size)
        self._setitem_index(index, value)

    def _setitem_index(self, index: IndexType, value: Any) -> None:
        """Writes ``value`` at the batch index ``index`` of every entry, for :meth:`__setitem__`.

        ``__setitem__`` has written the keys (a str or a tuple of str), and has
        replaced an ``Ellipsis`` in ``index`` by the batch dims that it covers
        (see :func:`~tensordict._indexing._expand_batch_ellipsis`). Any other
        index, a list included, is passed as given (the h5 file and the stores
        convert a list to a tensor). ``value`` is passed as given too: a tensor
        collection (a tensorclass is not unwrapped: the base hook does it), a
        dict, a tensor or a scalar. The return value is ignored.

        The base reads the index as ``__getitem__`` does, writes the entries of
        a tensor collection or a dict key by key (a new key through a
        sub-tensordict of ``self``), and writes any other value into every
        entry.
        """
        if (
            # is_tensorclass reads this class attribute: skip the call for the
            # values whose class does not have it, such as tensordicts
            (isinstance(value, type) or getattr(type(value), "_is_tensorclass", False))
            and is_tensorclass(value)
            and not is_non_tensor(value)
        ):
            # write the fields key by key, as a tensorclass __setitem__ does;
            # NonTensorData keeps the leaf path (its _tensordict is empty)
            value = value._tensordict

        if isinstance(index, tuple) and len(index) == 1:
            index = index[0]
        if (
            # skip the call for the indices that _is_new_dim_index rejects first
            index is None or index is True or isinstance(index, torch.Tensor)
        ) and _is_new_dim_index(index):
            # None and True add a dim of size 1, and the value is written to it
            # (False selects nothing, as a 0-d False mask does)
            if not self.batch_dims:
                # the entries of a tensordict without batch dims may not take an
                # index (e.g. NonTensorData), so write through a dim of size 1
                with self.unsqueeze(0) as td_unsqueezed:
                    td_unsqueezed[:] = value
                return
            if isinstance(value, (TensorDictBase, dict)):
                # torch reads the other values with None and True itself
                value = _value_at_new_dim(self, value)
                # a single slice, as an UnbatchedTensor entry may have fewer
                # dims than the batch dims
                index = slice(None)
        if isinstance(index, list):
            # Index with (list,), as __getitem__ does: torch reads a bare nested
            # list, and _SubTensorDict any bare list, as per-dim indices
            index = (index,)

        if isinstance(value, (TensorDictBase, dict)):
            indexed_bs = _getitem_batch_size(self.batch_size, index)
            # the device the entries of value are moved to as they are written
            move_to = None
            if isinstance(value, dict):
                value = self.from_dict_instance(
                    value, batch_size=indexed_bs, device=self.device
                )
            elif value.device != self.device:
                from tensordict._td import TensorDict

                if (
                    value.device is None
                    and type(value) is TensorDict
                    and _leaves_on(value, self.device)
                ):
                    # the tensors of a value without a device are often on
                    # self.device already: write them as they are, rather
                    # than a moved copy of the whole tree
                    move_to = self.device
                else:
                    value = value.to(self.device)
                # value = self.empty(recurse=True)[index].update(value)
            if value.batch_size != indexed_bs:
                if value.shape == indexed_bs[-len(value.shape) :]:
                    # try to expand on the left (broadcasting)
                    value = value.expand(indexed_bs)
                else:
                    try:
                        # copy and change batch_size if can't be expanded
                        value = value.copy()
                        value.batch_size = indexed_bs
                    except RuntimeError as err:
                        raise RuntimeError(
                            f"indexed destination TensorDict batch size is {indexed_bs} "
                            f"(batch_size = {self.batch_size}, index={index}), "
                            f"which differs from the source batch size {value.batch_size}"
                        ) from err

            keys = set(self.keys())
            subtd = None
            for value_key, item in value.items():
                if (
                    move_to is not None
                    and not isinstance(item, Tensor)
                    and (type(item) is not TensorDict or value_key not in keys)
                ):
                    # non-tensor data and a nested TensorDict under a new key
                    # get the device of self, as in value.to(move_to); their
                    # tensors are on it already. A nested TensorDict at an
                    # existing key gets it from the entry it is written into.
                    item = item.to(move_to)
                if value_key in keys:
                    self._set_at_str(
                        value_key, item, index, validated=True, non_blocking=False
                    )
                else:
                    if subtd is None:
                        subtd = self._get_sub_tensordict(index)
                    subtd.set(value_key, item, inplace=True, non_blocking=False)
        else:
            # read the index as getitem does, also if there is no key to write
            index = _entry_index(index)
            _set_at_each_key(self, value, index)

    def __delitem__(self, key: NestedKey) -> Self:
        return self.del_(key)

    def __getstate__(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        for key in (
            "_last_op",
            "_cache",
            "__lock_parents_weakrefs",
        ):
            result.pop(key, None)
        return result

    def __setstate__(self, state: dict[str, Any]) -> None:
        for key, value in state.items():
            setattr(self, key, value)
        self._cache = None
        self._last_op = None
        if self._is_locked:
            # this can cause avoidable overhead, as we will be locking the leaves
            # then locking their parent, and the parent of the parent, every
            # time re-locking tensordicts that have already been locked.
            # To avoid this, we should lock only at the root, but it isn't easy
            # to spot what the root is...
            self._is_locked = False
            self.lock_()

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Callable:
        if kwargs is None:
            kwargs = {}
        if func not in _TD_HANDLED_FUNCTIONS or not all(
            issubclass(t, (Tensor, TensorDictBase)) or _is_tensorclass(t) for t in types
        ):
            from torch._ops import HigherOrderOperator

            if isinstance(func, HigherOrderOperator):
                with torch._C.DisableTorchFunctionSubclass():
                    return func(*args, **kwargs)
            return NotImplemented
        return _TD_HANDLED_FUNCTIONS[func](*args, **kwargs)

    def auto_batch_size_(
        self, batch_dims: int | None = None, keep_compliant_size: bool = False
    ) -> Self:
        """Sets the batch size to the largest leading shape that all the entries share.

        The nested tensordicts get their own largest batch size first. Nested
        lazy stacks keep their batch size, and a lazy stack itself raises a
        ``RuntimeError`` if its batch size would change. See
        :ref:`batch-size`.

        Args:
            batch_dims (int, optional): the maximum number of batch dimensions,
                for the tensordict and for each nested tensordict. Defaults to
                ``None``, which sets no maximum.
            keep_compliant_size (bool, optional): if ``True`` and ``batch_dims``
                is passed, a nested tensordict that already has ``batch_dims``
                batch dimensions or more keeps its batch size. Defaults to
                ``False``.

        Returns:
            ``self``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.zeros(3, 4, 5), "b": {"c": torch.zeros(3, 4, 6)}}, batch_size=[])
            >>> _ = td.auto_batch_size_()
            >>> td.batch_size, td["b"].batch_size
            (torch.Size([3, 4]), torch.Size([3, 4, 6]))
            >>> _ = td.auto_batch_size_(batch_dims=1)
            >>> td.batch_size, td["b"].batch_size
            (torch.Size([3]), torch.Size([3]))

        """
        _set_max_batch_size(self, batch_dims, keep_compliant_size=keep_compliant_size)
        return self

    @classmethod
    @abc.abstractmethod
    def from_dict(
        cls,
        input_dict,
        *,
        auto_batch_size: bool | None = None,
        batch_size: torch.Size | None = None,
        device: torch.device | None = None,
        batch_dims: int | None = None,
        names: List[str] | None = None,
    ):
        """Returns a tensordict built from a dictionary, which can be nested.

        The nested dictionaries become nested tensordicts. Without
        ``batch_size``, ``auto_batch_size`` or ``batch_dims``, the batch size
        is ``[]``. With ``auto_batch_size=True``, each nested tensordict gets
        its own largest batch size, as with :meth:`~.auto_batch_size_`.
        :class:`~tensordict.TensorDict` returns a tensordict passed as
        ``input_dict`` unchanged, and ignores the keyword arguments: call
        :meth:`~.auto_batch_size_` on it instead. See :ref:`batch-size`.

        Args:
            input_dict (dict): the entries. A key can be a string or a tuple
                of strings, and a value a tensor, a dictionary, or any object
                that :class:`~tensordict.TensorDict` accepts.

        Keyword Args:
            auto_batch_size (bool, optional): if ``True``, sets the batch size
                to the largest leading shape that all the entries share.
                Defaults to ``None``, which is ``True`` if ``batch_dims`` is
                passed and ``False`` otherwise.
            batch_size (torch.Size or list of int, optional): the batch size of
                the result: the leading dimensions that all its entries share.
                Exclusive with ``batch_dims``. Defaults to ``None``.
            device (torch.device, str or int, optional): the device of the
                result. Leaves on another device are moved to it. ``None``
                leaves each leaf where it is, and the result has no device.
                Defaults to ``None``.
            batch_dims (int, optional): the maximum number of batch dimensions
                of the computed batch size. Turns on ``auto_batch_size``.
                Exclusive with ``batch_size``. Defaults to ``None``.
            names (list of str, optional): the names of the batch dimensions,
                one per dimension. Defaults to ``None``.

        Raises:
            ValueError: if both ``batch_size`` and ``batch_dims`` are passed.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> data = {"a": torch.zeros(3, 4), "b": {"c": torch.zeros(3, 4, 5)}}
            >>> TensorDict.from_dict(data).batch_size
            torch.Size([])
            >>> td = TensorDict.from_dict(data, auto_batch_size=True)
            >>> td.batch_size, td["b"].batch_size
            (torch.Size([3, 4]), torch.Size([3, 4, 5]))
            >>> TensorDict.from_dict(data, batch_dims=1).batch_size
            torch.Size([3])
            >>> TensorDict.from_dict(data, batch_size=[3]).batch_size
            torch.Size([3])
        """
        raise NotImplementedError

    @classmethod
    def _from_dict_validated(cls, *args, **kwargs):
        """A faster version of from_dict when the values have been validated.

        By default, falls back on :meth:`~.from_dict`.
        """
        return cls.from_dict(*args, **kwargs)

    # Shape functionality
    @property
    def shape(self) -> torch.Size:
        """The batch size of the tensordict, as a :class:`torch.Size`: the same as :attr:`~.batch_size`.

        Assigning it sets the batch size.
        """
        return self.batch_size

    @shape.setter
    def shape(self, value):
        self.batch_size = value

    @property
    def batch_size(self) -> torch.Size:
        """The batch size of the tensordict, as a :class:`torch.Size`. ``shape`` is the same attribute.

        The batch size is the shape of the leading dimensions that all tensors
        in the tensordict share. The other dimensions of each tensor are its
        feature dimensions. For example, a batch of videos of shape
        ``[B, T, C, W, H]`` can have batch size ``[B, T]`` (batch and time)
        and feature dimensions ``[C, W, H]`` (channels and space).

        You set the batch size when you create the tensordict. The batch size
        is not inferred from the tensors unless you ask for it, with
        ``TensorDict.from_dict(..., auto_batch_size=True)`` or
        :meth:`~.auto_batch_size_`.

        You can assign a new batch size if every tensor has the new leading
        dimensions. Otherwise, the assignment raises ``RuntimeError``. An empty
        batch size ``()`` always fits a :class:`~tensordict.TensorDict`. A
        :class:`~tensordict.LazyStackedTensorDict` raises ``RuntimeError`` for
        any new batch size. A nested tensordict with fewer batch dimensions than
        the new batch size gets the new batch size. The other nested tensordicts
        keep theirs. See :ref:`batch-size`.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> data = TensorDict({
            ...     "key 0": torch.zeros(3, 4),
            ...     "key 1": torch.zeros(3, 5),
            ...     "nested": TensorDict({"key 0": torch.zeros(3, 4)}, batch_size=[3, 4])},
            ...     batch_size=[3])
            >>> data.batch_size
            torch.Size([3])
            >>> data.batch_size = ()  # an empty batch size always fits
            >>> data.batch_size, data["nested"].batch_size
            (torch.Size([]), torch.Size([3, 4]))
            >>> data.batch_size = [4]
            Traceback (most recent call last):
            ...
            RuntimeError: the Tensor key 0 has shape torch.Size([3, 4]) which is incompatible with the batch-size torch.Size([4]).
        """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_size: torch.Size) -> None:
        self._batch_size_setter(new_size)
        self._update_consolidated_metadata()

    def size(self, dim: int | None = None) -> torch.Size | int:
        """Returns the batch size, or the size of one batch dimension.

        Args:
            dim (int, optional): a batch dimension. If ``None``, the whole
                :attr:`~.batch_size` is returned. Defaults to ``None``.

        Examples:
            >>> from tensordict import TensorDict
            >>> td = TensorDict(batch_size=[3, 4])
            >>> td.size()
            torch.Size([3, 4])
            >>> td.size(-1)
            4
        """
        if dim is None:
            return self.batch_size
        return self.batch_size[dim]

    @property
    def data(self) -> Self:
        """A tensordict of the ``.data`` of every leaf: the same storage, without autograd.

        The leaves of the result share storage with the leaves of the
        tensordict, and do not require gradients. An in-place change to the
        result changes the tensordict.

        Assigning ``td.data = other`` rebinds the ``.data`` of each leaf of
        the tensordict to the matching leaf of ``other``, as :attr:`torch.Tensor.data`
        does. The leaves then share storage with ``other``: memory-mapped or
        shared-memory leaves stop writing to their file or shared memory.
        ``other`` and each of its nested tensordicts must have the batch size
        of the matching tensordict, and each key of ``other`` must be in the
        tensordict.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(x=torch.zeros(3, requires_grad=True), batch_size=[3])
            >>> data = td.data
            >>> data["x"].requires_grad, data["x"].data_ptr() == td["x"].data_ptr()
            (False, True)
        """
        return self._data()

    @data.setter
    def data(self, value: Self):
        self._data_setter(value)

    @property
    def grad(self) -> Self:
        """A tensordict of the ``.grad`` of every leaf, or ``None`` if no leaf has one.

        Leaves whose ``.grad`` is ``None`` are left out. Assigning ``td.grad = other`` sets the ``.grad`` of each leaf from the
        matching leaf of ``other``, or copies into it if it exists.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(x=torch.ones(3, requires_grad=True), y=torch.ones(3, requires_grad=True), batch_size=[3])
            >>> (td["x"] * 2).sum().backward()
            >>> td.grad["x"], "y" in td.grad.keys()
            (tensor([2., 2., 2.]), False)
        """
        return self._grad()

    @grad.setter
    def grad(self, grad):
        def set_grad(x, grad):
            if x.grad is None:
                x.grad = grad
            else:
                x.grad.copy_(grad)

        self._fast_apply(set_grad, grad)

    def zero_grad(self, set_to_none: bool = True) -> Self:
        """Resets the ``.grad`` of every leaf, nested ones included, and returns ``self``.

        Args:
            set_to_none (bool, optional): if ``True``, sets each ``.grad`` to
                ``None``. If ``False``, fills each existing ``.grad`` with zeros in
                place. Defaults to ``True``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(x=torch.ones(3, requires_grad=True), batch_size=[3])
            >>> td["x"].sum().backward()
            >>> print(td.zero_grad()["x"].grad)
            None
        """
        if set_to_none:
            for val in self._values_list(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS):
                val.grad = None
            return self
        for val in self._values_list(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS):
            grad = val.grad
            if grad is not None:
                grad.zero_()
        return self

    @_cache_while_locked  # noqa
    def _dtype(self):
        dtype = None
        for val in self.values(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS):
            val_dtype = getattr(val, "dtype", None)
            if dtype is None and val_dtype is not None:
                dtype = val_dtype
            elif dtype is not None and val_dtype is not None and dtype != val_dtype:
                return None
        return dtype

    @property
    def dtype(self):
        """The dtype of the leaves, or ``None`` if they have different dtypes or there is no leaf."""
        return self._dtype()

    def _batch_size_setter(self, new_batch_size: torch.Size) -> None:
        if new_batch_size == self.batch_size or (
            # a list equal to the batch size (a tuple compares equal above)
            type(new_batch_size) is list and new_batch_size == list(self.batch_size)
        ):
            return
        if self._lazy:
            raise RuntimeError(
                f"Received a new batch size {new_batch_size} with an existing batch_size {self.batch_size} in a lazy TD. "
                "Modifying the batch size of a lazy representation of a "
                "tensordict is not permitted. Consider instantiating the "
                "tensordict first by calling `td = td.to_tensordict()` before "
                "resetting the batch size."
            )
        if not isinstance(new_batch_size, torch.Size):
            new_batch_size = torch.Size(new_batch_size)
        for key, value in self.items():
            if _is_tensor_collection(type(value)):
                if len(value.batch_size) < len(new_batch_size):
                    # document as edge case
                    value.batch_size = new_batch_size
                    self._set_str(
                        key, value, inplace=True, validated=True, non_blocking=False
                    )
        self._check_new_batch_size(new_batch_size)
        has_names = self._has_names()
        if has_names:
            # if the tensordict has dim names and the new batch-size has more dims,
            # we can simply add empty names after the current ones.
            # Otherwise, we discard the extra existing names.
            names = self.names
            self._erase_names()
        self._change_batch_size(new_batch_size)
        if has_names:
            # if the tensordict has dim names and the new batch-size has more dims,
            # we can simply add empty names after the current ones.
            # Otherwise, we discard the extra existing names.
            if len(names) < len(new_batch_size):
                self._set_names(names + [None] * (len(new_batch_size) - len(names)))
            else:
                self._set_names(names[: self.batch_dims])

    def _set_names(self, names: Sequence[str] | None):
        # we don't run checks on types for efficiency purposes
        if names is None:
            self._rename_subtds(names)
            self._erase_names()
            return
        value = list(names)
        # Faster but incompatible with dynamo
        # num_none = sum(v is None for v in value)
        num_none = 0
        for v in value:
            num_none += v is None
        batch_dims = self.batch_dims
        if num_none == batch_dims:
            if len(value) != batch_dims:
                _warn_names_length(value, batch_dims)
            self._set_names(None)
            return
        if num_none:
            num_none -= 1
        if len(set(value)) != len(value) - num_none:
            raise ValueError(f"Some dimension names are non-unique: {value}.")
        if len(value) != self.batch_dims:
            raise ValueError(
                "the length of the dimension names must equate the tensordict batch_dims attribute. "
                f"Got {value} for batch_dims {self.batch_dims}."
            )
        self._rename_subtds(value)
        self._td_dim_names = list(value)

    @property
    def batch_dims(self) -> int:
        """The number of batch dimensions, that is ``len(self.batch_size)``.

        ``ndim``, ``dim()`` and ``ndimension()`` return the same value.

        Examples:
            >>> from tensordict import TensorDict
            >>> TensorDict(batch_size=[3, 4]).batch_dims
            2
        """
        return len(self.batch_size)

    def ndimension(self) -> int:
        """Returns the number of batch dimensions, the same as :attr:`~.batch_dims`."""
        return self.batch_dims

    @property
    def ndim(self) -> int:
        """The number of batch dimensions, the same as :attr:`~.batch_dims`."""
        return self.batch_dims

    def dim(self) -> int:
        """Returns the number of batch dimensions, the same as :attr:`~.batch_dims`."""
        return self.batch_dims

    def numel(self) -> int:
        """Returns the number of elements in the batch, which is the product of ``batch_size``.

        The feature dimensions of the leaves do not count. The minimum is 1: a
        tensordict with batch size ``[]`` or ``[0]`` returns 1, unlike
        :meth:`torch.Tensor.numel`, which returns 0 for a size of 0.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> TensorDict(a=torch.zeros(2, 3, 10), batch_size=[2, 3]).numel()
            6
            >>> TensorDict(batch_size=[]).numel()
            1
        """
        return max(1, self.batch_size.numel())

    @property
    def depth(self) -> int:
        """The depth of the tensordict: the maximum number of levels of nested tensordicts.

        A tensordict with no nested tensordict has depth 0. Empty nested
        tensordicts do not count.

        Examples:
            >>> from tensordict import TensorDict
            >>> TensorDict({"a": 0}).depth
            0
            >>> TensorDict({"a": {"b": {"c": 0}}}).depth
            2
        """
        return self._depth()

    @_cache_while_locked  # noqa: B019
    def _depth(self):
        depth = 0
        for key in self.keys(True, True, is_leaf=_is_leaf_nontensor):
            if isinstance(key, tuple):
                depth = max(depth, len(key) - 1)
        return depth

    def new_zeros(
        self,
        *size: torch.Size,
        dtype: torch.dtype = None,
        device: DeviceType = NO_DEFAULT,
        requires_grad: bool = False,
        layout: torch.layout = torch.strided,
        pin_memory: bool | None = None,
        empty_lazy: bool = False,
    ):  # noqa: D417
        """Returns a tensordict of batch size ``size`` with every leaf filled with 0.

        Each leaf keeps its feature dimensions: a leaf of shape
        ``(*self.batch_size, *f)`` becomes ``(*size, *f)``. By default, each
        leaf keeps its dtype and device.

        Args:
            *size (int or sequence of int): the batch size of the result, as
                integers or as one list, tuple or ``torch.Size``.

        Keyword Args:
            dtype (torch.dtype, optional): the dtype of every leaf of the
                result. Defaults to ``None``: each leaf keeps its dtype.
            device (torch.device, str or int, optional): the device of the
                result and of its leaves. If not passed, each leaf keeps its
                device, and the result keeps the device of the tensordict.
            requires_grad (bool, optional): if ``True``, autograd records the
                operations on the new leaves. Defaults to ``False``.
            layout (torch.layout, optional): the layout of the new leaves.
                Defaults to ``torch.strided``.
            pin_memory (bool, optional): if ``True``, the new leaves are
                allocated in pinned memory. Works only for CPU tensors.
                Defaults to ``False``.
            empty_lazy (bool, optional): if ``True``, the lazy stacks of the
                result are emptied of their content. Use it when the content
                of a lazy stack is likely to change while you fill the new
                tensordict. It applies to the sub-tensordicts too. Defaults to
                ``False``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3, 4), b=torch.zeros(3, dtype=torch.int64), batch_size=[3])
            >>> out = td.new_zeros(2, 2)
            >>> out.batch_size, out["a"].shape, out["b"].dtype
            (torch.Size([2, 2]), torch.Size([2, 2, 4]), torch.int64)

        """
        kwargs = {}
        if pin_memory is not None:
            kwargs = {"pin_memory": pin_memory}
        return self._new_impl(
            size,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            layout=layout,
            funcname="new_zeros",
            empty_lazy=empty_lazy,
            **kwargs,
        )

    def new_ones(
        self,
        *size: torch.Size,
        dtype: torch.dtype = None,
        device: DeviceType = NO_DEFAULT,
        requires_grad: bool = False,
        layout: torch.layout = torch.strided,
        pin_memory: bool | None = None,
        empty_lazy: bool = False,
    ):  # noqa: D417
        """Returns a tensordict of batch size ``size`` with every leaf filled with 1.

        Each leaf keeps its feature dimensions: a leaf of shape
        ``(*self.batch_size, *f)`` becomes ``(*size, *f)``. By default, each
        leaf keeps its dtype and device.

        Args:
            *size (int or sequence of int): the batch size of the result, as
                integers or as one list, tuple or ``torch.Size``.

        Keyword Args:
            dtype (torch.dtype, optional): the dtype of every leaf of the
                result. Defaults to ``None``: each leaf keeps its dtype.
            device (torch.device, str or int, optional): the device of the
                result and of its leaves. If not passed, each leaf keeps its
                device, and the result keeps the device of the tensordict.
            requires_grad (bool, optional): if ``True``, autograd records the
                operations on the new leaves. Defaults to ``False``.
            layout (torch.layout, optional): the layout of the new leaves.
                Defaults to ``torch.strided``.
            pin_memory (bool, optional): if ``True``, the new leaves are
                allocated in pinned memory. Works only for CPU tensors.
                Defaults to ``False``.
            empty_lazy (bool, optional): if ``True``, the lazy stacks of the
                result are emptied of their content. Use it when the content
                of a lazy stack is likely to change while you fill the new
                tensordict. It applies to the sub-tensordicts too. Defaults to
                ``False``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3, 4), b=torch.zeros(3, dtype=torch.int64), batch_size=[3])
            >>> out = td.new_ones(2)
            >>> out.batch_size, out["a"].shape
            (torch.Size([2]), torch.Size([2, 4]))

        """
        kwargs = {}
        if pin_memory is not None:
            kwargs = {"pin_memory": pin_memory}
        return self._new_impl(
            size,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            layout=layout,
            funcname="new_ones",
            empty_lazy=empty_lazy,
            **kwargs,
        )

    def new_empty(
        self,
        *size: torch.Size,
        dtype: torch.dtype = None,
        device: DeviceType = NO_DEFAULT,
        requires_grad: bool = False,
        layout: torch.layout = torch.strided,
        pin_memory: bool | None = None,
        empty_lazy: bool = False,
    ):  # noqa: D417
        """Returns a tensordict of batch size ``size`` with uninitialized leaves.

        The values of the leaves are not set, as with :func:`torch.empty`.

        Each leaf keeps its feature dimensions: a leaf of shape
        ``(*self.batch_size, *f)`` becomes ``(*size, *f)``. By default, each
        leaf keeps its dtype and device.

        Args:
            *size (int or sequence of int): the batch size of the result, as
                integers or as one list, tuple or ``torch.Size``.

        Keyword Args:
            dtype (torch.dtype, optional): the dtype of every leaf of the
                result. Defaults to ``None``: each leaf keeps its dtype.
            device (torch.device, str or int, optional): the device of the
                result and of its leaves. If not passed, each leaf keeps its
                device, and the result keeps the device of the tensordict.
            requires_grad (bool, optional): if ``True``, autograd records the
                operations on the new leaves. Defaults to ``False``.
            layout (torch.layout, optional): the layout of the new leaves.
                Defaults to ``torch.strided``.
            pin_memory (bool, optional): if ``True``, the new leaves are
                allocated in pinned memory. Works only for CPU tensors.
                Defaults to ``False``.
            empty_lazy (bool, optional): if ``True``, the lazy stacks of the
                result are emptied of their content. Use it when the content
                of a lazy stack is likely to change while you fill the new
                tensordict. It applies to the sub-tensordicts too. Defaults to
                ``False``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3, 4), b=torch.zeros(3, dtype=torch.int64), batch_size=[3])
            >>> out = td.new_empty(5)
            >>> out.batch_size, out["a"].shape
            (torch.Size([5]), torch.Size([5, 4]))

        """
        kwargs = {}
        if pin_memory is not None:
            kwargs = {"pin_memory": pin_memory}
        return self._new_impl(
            size,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            layout=layout,
            funcname="new_empty",
            empty_lazy=empty_lazy,
            **kwargs,
        )

    def new_full(
        self,
        size: torch.Size,
        fill_value,
        *,
        dtype: torch.dtype = None,
        device: DeviceType = NO_DEFAULT,
        requires_grad: bool = False,
        layout: torch.layout = torch.strided,
        pin_memory: bool | None = None,
        empty_lazy: bool = False,
    ):  # noqa: D417
        """Returns a tensordict of batch size ``size`` with every leaf filled with ``fill_value``.

        Each leaf keeps its feature dimensions: a leaf of shape
        ``(*self.batch_size, *f)`` becomes ``(*size, *f)``. By default, each
        leaf keeps its dtype and device.

        Args:
            size (int or sequence of int): the batch size of the result, as an
                int, or as a list, tuple or ``torch.Size``.
            fill_value (scalar): the value of every element of the result.

        Keyword Args:
            dtype (torch.dtype, optional): the dtype of every leaf of the
                result. Defaults to ``None``: each leaf keeps its dtype.
            device (torch.device, str or int, optional): the device of the
                result and of its leaves. If not passed, each leaf keeps its
                device, and the result keeps the device of the tensordict.
            requires_grad (bool, optional): if ``True``, autograd records the
                operations on the new leaves. Defaults to ``False``.
            layout (torch.layout, optional): the layout of the new leaves.
                Defaults to ``torch.strided``.
            pin_memory (bool, optional): if ``True``, the new leaves are
                allocated in pinned memory. Works only for CPU tensors.
                Defaults to ``False``.
            empty_lazy (bool, optional): if ``True``, the lazy stacks of the
                result are emptied of their content. Use it when the content
                of a lazy stack is likely to change while you fill the new
                tensordict. It applies to the sub-tensordicts too. Defaults to
                ``False``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3, 4), b=torch.zeros(3, dtype=torch.int64), batch_size=[3])
            >>> out = td.new_full((2,), 7.0)
            >>> out["a"][0], out["b"]
            (tensor([7., 7., 7., 7.]), tensor([7, 7]))

        """
        kwargs = {}
        if pin_memory is not None:
            kwargs = {"pin_memory": pin_memory}
        return self._new_impl(
            size,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            layout=layout,
            funcname="new_full",
            fill_value=fill_value,
            empty_lazy=empty_lazy,
            **kwargs,
        )

    def _new_impl(
        self,
        size: torch.Size,
        *,
        dtype: torch.dtype = None,
        device: DeviceType = NO_DEFAULT,
        requires_grad: bool = False,
        layout: torch.layout = torch.strided,
        funcname: str,
        empty_lazy: bool = False,
        **kwargs,
    ):
        if isinstance(size, int):
            size = (size,)
        elif len(size) == 1 and not isinstance(size[0], int):
            size = size[0]

        ndim = self.ndim
        if device is not NO_DEFAULT:
            kwargs["device"] = device

        def func(tensor, size=size):
            feature_shape = tensor.shape[ndim:]
            size = torch.Size((*size, *feature_shape))
            kwargs_copy = kwargs
            if empty_lazy and is_tensor_collection(tensor):
                kwargs_copy = dict(kwargs)
                kwargs_copy["empty_lazy"] = empty_lazy
            return getattr(tensor, funcname)(
                size,
                dtype=dtype,
                requires_grad=requires_grad,
                layout=layout,
                **kwargs_copy,
            )

        names = self._maybe_names()
        if names:
            if len(size) > self.ndim:
                names = [None] * (len(size) - self.ndim) + list(names)
            elif self.ndim > len(size):
                names = names[-len(size) :]
        return self._fast_apply(
            func,
            call_on_nested=True,
            device=device,
            batch_size=size,
            names=names,
        )

    def new_tensor(
        self,
        data: torch.Tensor | TensorCollection | np.ndarray | Sequence | float,
        *,
        dtype: torch.dtype = None,
        device: DeviceType = NO_DEFAULT,
        requires_grad: bool = False,
        pin_memory: bool | None = None,
    ) -> Self:  # noqa: D417
        """Returns a tensordict whose leaves are copies of ``data``, each with the dtype of its leaf.

        Each leaf of the tensordict gives a leaf of the result, built as
        ``leaf.new_tensor(data)``: the batch size of the result and the shape
        of every leaf are the shape of ``data``. If ``data`` is a tensordict or
        a tensorclass, each leaf is built from the matching leaf of ``data``.

        Args:
            data (torch.Tensor, TensorDictBase or array_like): the data to
                copy. As for :meth:`torch.Tensor.new_tensor`, it can also be a
                list, a tuple, a NumPy array or a scalar.

        Keyword Args:
            dtype (torch.dtype, optional): the dtype of every leaf of the
                result. Defaults to ``None``: each leaf keeps the dtype of the
                matching leaf of the tensordict.
            device (torch.device, str or int, optional): the device of the
                result and of its leaves. If not passed, the device of
                ``data``: the default device for a list, a tuple, a NumPy array
                or a scalar.
            requires_grad (bool, optional): if ``True``, autograd records the
                operations on the new leaves. Defaults to ``False``.
            pin_memory (bool, optional): if ``True``, the new leaves are
                allocated in pinned memory. Works only for CPU tensors.
                Defaults to ``False``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(
            ...     a=torch.zeros(3), b=torch.zeros(3, dtype=torch.int64), batch_size=[3]
            ... )
            >>> out = td.new_tensor([1, 2])
            >>> out.batch_size
            torch.Size([2])
            >>> out["a"], out["b"]
            (tensor([1., 2.]), tensor([1, 2]))

        """
        if isinstance(data, torch.Tensor) or _is_tensor_collection(type(data)):
            shape = data.shape
            data_device = data.device
        else:
            # An array_like, as torch.Tensor.new_tensor takes. The entries read
            # ``data`` as it is, so torch does not warn about copying a tensor.
            as_tensor = torch.as_tensor(data)
            shape = as_tensor.shape
            data_device = as_tensor.device
        kwargs = {}
        if device is not NO_DEFAULT:
            kwargs["device"] = device
        else:
            device = kwargs["device"] = data_device
        if pin_memory is not None:
            kwargs["pin_memory"] = pin_memory
        if dtype is not None:
            # Under torch.compile, Tensor.new_tensor(data, dtype=None) takes the
            # dtype of data, not of the tensor, so pass dtype only when it is set.
            kwargs["dtype"] = dtype

        def func(x, tensor_b=None):
            if tensor_b is None:
                tensor_b = data
            return x.new_tensor(
                tensor_b,
                requires_grad=requires_grad,
                **kwargs,
            )

        if _is_tensor_collection(type(data)):
            return self._fast_apply(
                func,
                data,
                call_on_nested=True,
                device=device,
                batch_size=shape,
            )
        return self._fast_apply(
            func,
            call_on_nested=True,
            device=device,
            batch_size=shape,
        )

    def _unbind(self, dim: int):
        batch_size = torch.Size([s for i, s in enumerate(self.batch_size) if i != dim])
        names = None
        if self._has_names():
            names = [name for i, name in enumerate(self.names) if i != dim]
            # We could use any() but dynamo doesn't like generators
            for name in names:
                if name is not None:
                    break
            else:
                names = None
        device = self.device

        is_shared = self._is_shared
        is_memmap = self._is_memmap

        def empty(
            batch_size=batch_size,
            names=names,
            device=device,
            is_shared=is_shared,
            is_memmap=is_memmap,
        ):
            result = self._new_unsafe(
                {}, batch_size=batch_size, names=names, device=device
            )
            result._is_shared = is_shared
            result._is_memmap = is_memmap
            return result

        tds = tuple(empty() for _ in range(self.batch_size[dim]))

        def unbind(key, val, tds=tds):
            if _is_unbatched(val):
                for td in tds:
                    td._set_str(
                        key,
                        _unbatched_with_batch_size(val, batch_size),
                        validated=True,
                        inplace=False,
                        non_blocking=False,
                    )
                return
            unbound = (
                val.unbind(dim)
                if not isinstance(val, TensorDictBase)
                # tensorclass is also unbound using plain unbind
                else val._unbind(dim)
            )
            for td, _val in _zip_strict(tds, unbound):
                td._set_str(
                    key, _val, validated=True, inplace=False, non_blocking=False
                )

        for key, val in self.items():
            unbind(key, val)
        return tds

    @abc.abstractmethod
    def _unsqueeze(self, dim: int):
        """Inserts a batch dim of size 1 at ``dim``.

        Called by :meth:`unsqueeze`, which has checked that ``dim`` is in
        ``[0, batch_dims]``. The leaves of the result are views of the leaves
        of ``self``. :meth:`unsqueeze` locks a shared or memory-mapped result.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _squeeze(self, dim=None):
        """Removes the batch dim ``dim``, or every batch dim of size 1 when ``dim`` is ``None``.

        Called by :meth:`squeeze`, which has checked that ``dim`` is in
        ``[0, batch_dims)`` and that its size is 1 (it returns ``self``
        otherwise). The leaves of the result are views of the leaves of
        ``self``. :meth:`squeeze` locks a shared or memory-mapped result.
        """
        raise NotImplementedError

    def copy(self) -> Self:
        """Returns a shallow copy of the tensordict: a new structure that holds the same leaves.

        The same as ``td.clone(recurse=False)``. The nested tensordicts are new
        objects, but the leaves are the same tensors: an in-place change to a
        leaf shows in both.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3), b={"c": torch.zeros(3)}, batch_size=[3])
            >>> c = td.copy()
            >>> c["a"] is td["a"], c["b"] is td["b"]
            (True, False)
        """
        return self.clone(recurse=False)

    @abc.abstractmethod
    def _view(self, shape: torch.Size) -> Self:
        """Returns a tensordict whose batch size is ``shape``, with views of the leaves of ``self``.

        Called by :meth:`view` with a shape that is not the batch size of
        ``self``. ``shape`` can hold one ``-1``, which the hook infers from
        the number of elements of the batch. A class that cannot give views
        raises. :meth:`view` locks a shared or memory-mapped result.
        """
        raise NotImplementedError

    def _view_dtype(self, *, dtype, batch_size):
        # We use apply because we want to check the shapes
        def view(x):
            return x.view(dtype)

        return self.apply(view, batch_size=batch_size)

    @abc.abstractmethod
    def _transpose(self, dim0, dim1):
        raise NotImplementedError

    # Alias for swapaxes (matching torch.swapdims)

    @abc.abstractmethod
    def _permute(self, dims: list[int]):
        """Returns a tensordict whose batch dims are the batch dims ``dims`` of ``self``, in that order.

        Called by :meth:`permute`, which has made the negative dims of
        ``dims`` positive. ``dims`` can be shorter than the batch dims, if it
        is a permutation of the first ``len(dims)`` dims: the other dims stay
        in place. The hook checks ``dims``, because the check depends on the
        class: ``TensorDict`` raises a ``ValueError`` for dims that are not
        such a permutation, and ``LazyStackedTensorDict``, which permutes its
        members, warns until TensorDict 0.17. ``TensorDict`` returns ``self``
        for the identity. :meth:`permute` locks a shared or memory-mapped result.
        """
        raise NotImplementedError

    # Alias for movedim (matching torch.moveaxis)

    # Cache functionality
    def _erase_cache(self):
        self._cache = None

    # Dim names functionality
    @property
    def names(self):
        """The dimension names of the tensordict, one per batch dimension.

        A dimension without a name is ``None``. The names can be set at
        construction time using the ``names`` argument, or by assignment
        (``td.names = [...]``). The names must be unique. Shape operations and
        indexing keep the names of the dimensions that remain. Assigning
        ``None`` removes the names. See :ref:`batch-size`.

        .. seealso:: :meth:`~.refine_names` and :meth:`~.rename_`.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3, 4), batch_size=[3, 4], names=["time", None])
            >>> td.names
            ['time', None]
            >>> td.names = ["time", "batch"]
            >>> td[0].names
            ['batch']
        """
        names = self._td_dim_names
        if names is None:
            return [None for _ in range(self.batch_dims)]
        # assert len(names) == self.batch_dims, (names, self.batch_dims)
        # Return a copy but don't use copy to make dynamo happy
        return list(names)

    @names.setter
    def names(self, value):
        self._set_names(value)
        self._update_consolidated_metadata()

    def _get_names_idx(self, idx):
        if not self._has_names():
            return None
        names = _getitem_names(self.names, idx)
        if all(name is None for name in names):
            return None
        return names

    def _erase_names(self):
        """Erases the dimension names from a tensordict."""
        self._td_dim_names = None

    def _rename_subtds(self, value):
        """Gives the sub-tensordicts the names in value for the dims they share with self.

        The dims a sub-tensordict has beyond ``self.batch_dims`` keep their
        names. ``value=None`` clears the names of the shared dims.

        The default renames the sub-tensordicts that
        :meth:`_lock_sub_tensordicts` returns, with ``rename_``.
        """
        if value is None:
            value = [None] * self.ndim
        for item in self._lock_sub_tensordicts():
            td_names = list(value) + list(item.names)[self.ndim :]
            item.rename_(*td_names)

    def _check_dim_name(self, name):
        if name is None:
            return False
        if self._has_names() and name in self.names:
            return True
        for key in self.keys():
            if _is_tensor_collection(self.entry_class(key)):
                if self._get_str(key, NO_DEFAULT)._check_dim_name(name):
                    return True
        else:
            return False

    def refine_names(self, *names) -> Self:
        """Names the unnamed batch dimensions, and checks the named ones, in place.

        Each entry of ``names`` goes with one batch dimension. An unnamed
        dimension takes the new name, or stays unnamed for ``None``. A named
        dimension must get its own name. ``names`` can hold one ``...``, which
        keeps the names of the dimensions that it covers. This follows
        :meth:`torch.Tensor.refine_names`.

        Args:
            *names (str or None): one name per batch dimension, or fewer with
                one ``...``.

        Returns:
            ``self``.

        Raises:
            RuntimeError: if a named dimension gets another name, or ``None``.

        Examples:
            >>> from tensordict import TensorDict
            >>> td = TensorDict(batch_size=[3, 4, 5], names=["a", None, None])
            >>> td.refine_names("a", None, "c").names
            ['a', None, 'c']
            >>> td.refine_names(..., "b", "c").names
            ['a', 'b', 'c']

        """
        # replace ellipsis if any
        names_copy = list(names)
        if any(name is Ellipsis for name in names):
            ellipsis_name = [NO_DEFAULT for _ in range(self.ndim - len(names) + 1)]
            names = []
            for name in names_copy:
                if name is Ellipsis:
                    names += ellipsis_name
                else:
                    names.append(name)
        # check that the names that are set are either None or identical
        curr_names = self.names
        for i, name in enumerate(names):
            if name is NO_DEFAULT:
                # whatever value is ok
                names[i] = curr_names[i]
                continue
            else:
                if curr_names[i] is None:
                    continue
                if self.names[i] == name:
                    continue
                else:
                    raise RuntimeError(
                        f"refine_names: cannot coerce TensorDict names {self.names} with {names_copy}."
                    )
        self._set_names(names)
        # we also need to rename the sub-tensordicts
        # self._rename_subtds(self.names)
        return self

    def rename(self, *names, **rename_map):
        """Returns a shallow copy of the tensordict with new names for its batch dimensions.

        Pass either the new names, one per batch dimension, or keyword
        arguments that map old names to new ones. The leaves are not copied
        (see :meth:`~.copy`), and the names of the tensordict do not change.
        The nested tensordicts of the copy get the new names for the batch
        dimensions they share with it.

        Args:
            *names (str or None): the new names, one per batch dimension, or
                a single ``None`` to clear every name. Exclusive with
                ``**rename_map``.
            **rename_map (str or None): new names by old name, as in
                ``rename(old="new")``. Exclusive with ``*names``.

        Raises:
            ValueError: if both or neither of ``names`` and ``rename_map`` are
                passed, or if a key of ``rename_map`` is not a name of the
                tensordict.

        Examples:
            >>> from tensordict import TensorDict
            >>> td = TensorDict(batch_size=[1, 2, 3, 4], names=list("abcd"))
            >>> td.rename(c="g").names
            ['a', 'b', 'g', 'd']
            >>> td.rename("w", "x", "y", "z").names
            ['w', 'x', 'y', 'z']
            >>> td.rename(None).names
            [None, None, None, None]
            >>> td.names
            ['a', 'b', 'c', 'd']

        """
        clone = self.clone(recurse=False)
        if rename_map and names:
            raise ValueError(
                "Passed both a name map and a name list. Only one is accepted."
            )
        elif not rename_map and not names:
            raise ValueError(
                "Neither a name map nor a name list was passed. Only one is accepted."
            )
        elif rename_map:
            cnames = list(clone.names)
            for i, name in enumerate(cnames):
                new_name = rename_map.pop(name, NO_DEFAULT)
                if new_name is not NO_DEFAULT:
                    cnames[i] = new_name
            clone.names = cnames
            if rename_map:
                raise ValueError(
                    f"Some names to be renamed were not part of the tensordict names: {rename_map.keys()} vs {self.names}."
                )
        elif len(names) == 1 and names[0] is None:
            clone.names = None
        else:
            clone.names = names
        return clone

    def rename_(self, *names, **rename_map):
        """Renames the batch dimensions of the tensordict in place and returns ``self``.

        Pass either the new names, one per batch dimension, or keyword
        arguments that map old names to new ones. The nested tensordicts get
        the new names for the batch dimensions they share with the tensordict.

        Args:
            *names (str or None): the new names, one per batch dimension, or
                a single ``None`` to clear every name. Exclusive with
                ``**rename_map``.
            **rename_map (str or None): new names by old name, as in
                ``rename_(old="new")``. Exclusive with ``*names``.

        Raises:
            ValueError: if both ``names`` and ``rename_map`` are passed, or
                neither (except on a tensordict with no batch dimension), or if
                a key of ``rename_map`` is not a name of the tensordict.

        .. seealso:: :meth:`~.rename`, which returns a renamed copy.

        Examples:
            >>> from tensordict import TensorDict
            >>> td = TensorDict(batch_size=[1, 2, 3, 4], names=list("abcd"))
            >>> td.rename_(c="g") is td
            True
            >>> td.names
            ['a', 'b', 'g', 'd']
            >>> td.rename_(None).names
            [None, None, None, None]
        """
        if rename_map and names:
            raise ValueError(
                "Passed both a name map and a name list. Only one is accepted."
            )
        elif not rename_map and not names and self.batch_dims:
            raise ValueError(
                "Neither a name map nor a name list was passed. Only one is accepted."
            )
        elif rename_map:
            cnames = list(self.names)
            for i, name in enumerate(cnames):
                new_name = rename_map.pop(name, NO_DEFAULT)
                if new_name is not NO_DEFAULT:
                    cnames[i] = new_name
            if rename_map:
                raise ValueError(
                    f"Some names to be renamed were not part of the tensordict names: {rename_map.keys()} vs {self.names}."
                )
            self._set_names(cnames)
        elif len(names) == 1 and names[0] is None:
            self._set_names(None)
        else:
            self._set_names(names)
        return self

    def _has_names(self):
        return self._td_dim_names is not None

    def _maybe_names(self) -> Sequence[str] | None:
        if self._has_names():
            return self.names
        return None

    @property
    def _has_non_tensor(self):
        """Checks if the tensordict has non-tensor data."""
        for value in self.values(True, True, is_leaf=_is_leaf_nontensor):
            if _is_non_tensor(type(value)):
                return True
        return False

    # Device functionality: device is optional. If provided, it will enforce
    # all data is on the same device
    @property
    def device(self) -> torch.device | None:
        """The device of the tensordict, or ``None`` if it has none.

        If the tensordict has a device, all its leaves, nested ones included,
        are on that device: the tensors that you pass to the constructor or
        set later are moved to it. If the device is ``None``, the leaves can be
        on different devices. The device is not inferred from the tensors.
        Assigning it raises a ``RuntimeError``: use :meth:`~.to` to get a
        tensordict on another device. See :ref:`devices`.

        Examples:
            The examples use the ``"meta"`` device so that they run on CPU.
            A CUDA device works the same way.

            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({
            ...     "cpu": torch.randn(3, device='cpu'),
            ...     "meta": torch.randn(3, device='meta'),
            ... }, batch_size=[], device=None)
            >>> td['cpu'].device
            device(type='cpu')
            >>> td['meta'].device
            device(type='meta')
            >>> print(td.device)
            None
            >>> td = TensorDict({
            ...     "x": torch.randn(3, device='cpu'),
            ...     "y": TensorDict({'z': torch.randn(3, device='cpu')}, batch_size=[], device=None),
            ... }, batch_size=[], device='meta')
            >>> td['x'].device
            device(type='meta')
            >>> td['y'].device # nested tensordicts are also mapped onto the appropriate device.
            device(type='meta')
            >>> td['y', 'z'].device
            device(type='meta')
            >>> td['w'] = torch.zeros(3)  # a new entry is moved to the device too
            >>> td['w'].device
            device(type='meta')

        """
        return self._device

    @device.setter
    def device(self, value: DeviceType) -> None:
        raise RuntimeError(
            "device cannot be set using tensordict.device = device, "
            "because device cannot be updated in-place. To update device, use "
            "tensordict.to(new_device), which will return a new tensordict "
            "on the new device. You can also use tensordict.to(new_device) as a "
            "context manager for temporary device changes."
        )

    @_lock_blocked
    def clear(self) -> Self:
        """Removes every entry of the tensordict and returns ``self``.

        Raises:
            RuntimeError: if the tensordict is locked.
        """
        for key in list(self.keys()):
            del self[key]
        return self

    @_lock_blocked
    def popitem(self) -> Tuple[NestedKey, CompatibleType]:
        """Removes the last top-level entry and returns it as a ``(key, value)`` pair.

        The last entry is the last key of :meth:`~.keys`, which is the last
        one inserted for a :class:`~tensordict.TensorDict`. A nested
        tensordict is returned whole.

        Raises:
            KeyError: if the tensordict is empty.
            RuntimeError: if the tensordict is locked.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3), b=torch.ones(3), batch_size=[3])
            >>> td.popitem()
            ('b', tensor([1., 1., 1.]))
            >>> list(td.keys())
            ['a']
        """
        # Generic: pop the last key of keys(). A comprehension walks the keys
        # once: list() would call the view's __len__, which walks them too.
        keys = [key for key in self.keys()]  # noqa: C416
        if not keys:
            raise KeyError(f"popitem(): {type(self).__name__} is empty")
        key = keys[-1]
        return key, self.pop(key)

    @property
    def is_cuda(self):
        """Whether the tensordict has a CUDA device (``False`` if it has no device)."""
        return self.device is not None and self.device.type == "cuda"

    @property
    def is_cpu(self):
        """Whether the tensordict has a CPU device (``False`` if it has no device)."""
        return self.device is not None and self.device.type == "cpu"

    @property
    def is_meta(self) -> bool:
        """Whether the tensordict has a meta device (``False`` if it has no device)."""
        return self.device is not None and self.device.type == "meta"

    def share_memory_(self) -> Self:
        """Moves the leaves to shared memory, locks the tensordict, and returns ``self``.

        Processes that receive the tensordict, for example through
        :mod:`torch.multiprocessing`, then read and write the same tensors.
        The lock forbids adding, removing or replacing entries, but in-place
        writes work (see :ref:`keys-locking`). On a tensordict whose device is
        CUDA, the call does nothing: CUDA tensors are shared already, and the
        tensordict is not locked. After
        :meth:`~.unlock_`, :meth:`~.is_shared` returns ``False``, although the
        tensors stay in shared memory.

        Raises:
            RuntimeError: if the tensordict is memory-mapped.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3), batch_size=[3])
            >>> td.share_memory_().is_shared(), td.is_locked, td["a"].is_shared()
            (True, True, True)
        """
        self._share_memory_()
        return self

    def _share_memory_(self) -> None:
        """Places the leaves in shared memory and locks the tensordict.

        The storage part of :meth:`share_memory_`, which returns ``self``. The
        default raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"Cannot call share_memory_ on a {type(self).__name__}. "
            "Call `to_tensordict()` first."
        )

    @abc.abstractmethod
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
    ) -> Self:
        raise NotImplementedError

    @property
    def saved_path(self):
        """The directory where a memory-mapped tensordict stores its files, as a :class:`~pathlib.Path`.

        Raises:
            AttributeError: if :meth:`~.is_memmap` returns ``False``, for
                example after :meth:`~.unlock_`.
        """
        if self.is_memmap():
            path = self._memmap_prefix
            return path
        raise AttributeError(
            f"The tensordict has no saved path (memmap={self.is_memmap()}, path={self._memmap_prefix})."
        )

    def make_memmap(
        self,
        key: NestedKey,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        robust_key: bool | None = True,
    ) -> MemoryMappedTensor:
        """Adds an uninitialized memory-mapped tensor of shape ``shape`` at ``key``, and returns it.

        The tensordict must be memory-mapped (see :meth:`~.memmap`), and the
        file of the new tensor goes in its directory.

        The entry is written even though a memory-mapped tensordict is
        locked. Other tensordicts that share its files, for example in other
        processes, see it only after
        :meth:`~tensordict.TensorDictBase.memmap_refresh_`.

        Args:
            key (NestedKey): the key of the new entry. A tuple key creates
                the missing nested tensordicts.
            shape (torch.Size or torch.Tensor): the shape of the new tensor,
                or a tensor of shapes for a nested tensor.

        Keyword Args:
            dtype (torch.dtype, optional): the dtype of the new tensor.
                Defaults to ``None``, which uses
                :func:`torch.get_default_dtype`.
            robust_key (bool, optional): if ``True``, keys are percent-encoded
                in file names, so that every file stays inside the directory of
                the tensordict. If ``False``, keys are used as file names
                unchanged. ``None`` is the same as ``True``. Defaults to
                ``True``.

        Returns:
            The new :class:`~tensordict.MemoryMappedTensor`.

        Raises:
            RuntimeError: if the tensordict is not memory-mapped, or if the key
                exists.

        Examples:
            >>> import tempfile
            >>> import torch
            >>> from tensordict import TensorDict
            >>> with tempfile.TemporaryDirectory() as path:
            ...     td = TensorDict(a=torch.zeros(3), batch_size=[3]).memmap(path)
            ...     x = td.make_memmap(("nested", "b"), (3, 4))
            ...     print(td["nested", "b"].shape, td["nested", "b"] is x)
            torch.Size([3, 4]) True
        """
        raise RuntimeError(
            _unsupported(
                self, "make_memmap", "Only a memory-mapped TensorDict supports it."
            )
        )

    def make_memmap_from_storage(
        self,
        key: NestedKey,
        storage: torch.UntypedStorage,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        robust_key: bool | None = True,
    ) -> MemoryMappedTensor:
        """Adds a memory-mapped tensor at ``key`` that reads its data from ``storage``, and returns it.

        The tensordict must be memory-mapped (see :meth:`~.memmap`).

        The entry is written even though a memory-mapped tensordict is
        locked. Other tensordicts that share its files, for example in other
        processes, see it only after
        :meth:`~tensordict.TensorDictBase.memmap_refresh_`.

        If the storage has a file, it must be the file of the new entry in the
        directory of the tensordict, or the call raises a ``RuntimeError``. A
        storage without a file is used as it is: its data is not written to
        the directory. If the tensordict has a directory and the storage is
        not empty, :meth:`~.load_memmap` and
        :meth:`~tensordict.TensorDictBase.memmap_refresh_` drop the entry, and
        the call emits a ``FutureWarning``: starting with TensorDict 0.17, the
        default will change to raising a ``RuntimeError``. Use
        :meth:`~.make_memmap_from_tensor` to copy the data into the directory.

        Args:
            key (NestedKey): the key of the new entry. A tuple key creates
                the missing nested tensordicts.
            storage (torch.UntypedStorage): the storage of the new tensor. It
                must be in physical memory.
            shape (torch.Size or torch.Tensor): the shape of the new tensor,
                or a tensor of shapes for a nested tensor.

        Keyword Args:
            dtype (torch.dtype, optional): the dtype of the new tensor.
                Defaults to ``None``, which uses
                :func:`torch.get_default_dtype`.
            robust_key (bool, optional): if ``True``, keys are percent-encoded
                in file names, so that every file stays inside the directory of
                the tensordict. If ``False``, keys are used as file names
                unchanged. ``None`` is the same as ``True``. Defaults to
                ``True``.

        Returns:
            The new :class:`~tensordict.MemoryMappedTensor`.

        Raises:
            RuntimeError: if the tensordict is not memory-mapped, if the key
                exists, or if the file of the storage is not the file of the
                new entry.
        """
        raise RuntimeError(
            _unsupported(
                self,
                "make_memmap_from_storage",
                "Only a memory-mapped TensorDict supports it.",
            )
        )

    def make_memmap_from_tensor(
        self,
        key: NestedKey,
        tensor: torch.Tensor,
        *,
        copy_data: bool = True,
        robust_key: bool | None = True,
    ) -> MemoryMappedTensor:
        """Adds a memory-mapped copy of ``tensor`` at ``key``, and returns it.

        The tensordict must be memory-mapped (see :meth:`~.memmap`), and the
        file of the new tensor goes in its directory. The new tensor does not
        share storage with ``tensor``.

        The entry is written even though a memory-mapped tensordict is
        locked. Other tensordicts that share its files, for example in other
        processes, see it only after
        :meth:`~tensordict.TensorDictBase.memmap_refresh_`.

        Args:
            key (NestedKey): the key of the new entry. A tuple key creates
                the missing nested tensordicts.
            tensor (torch.Tensor): the tensor to copy.

        Keyword Args:
            copy_data (bool, optional): if ``False``, the new tensor has the
                shape and dtype of ``tensor`` but its content is not
                initialized. Defaults to ``True``.
            robust_key (bool, optional): if ``True``, keys are percent-encoded
                in file names, so that every file stays inside the directory of
                the tensordict. If ``False``, keys are used as file names
                unchanged. ``None`` is the same as ``True``. Defaults to
                ``True``.

        Returns:
            The new :class:`~tensordict.MemoryMappedTensor`.

        Raises:
            RuntimeError: if the tensordict is not memory-mapped, or if the key
                exists.

        Examples:
            >>> import tempfile
            >>> import torch
            >>> from tensordict import TensorDict
            >>> with tempfile.TemporaryDirectory() as path:
            ...     td = TensorDict(a=torch.zeros(3), batch_size=[3]).memmap(path)
            ...     x = td.make_memmap_from_tensor("b", torch.ones(3, 2))
            ...     print(td["b"].sum())
            tensor(6.)
        """
        raise RuntimeError(
            _unsupported(
                self,
                "make_memmap_from_tensor",
                "Only a memory-mapped TensorDict supports it.",
            )
        )

    # Key functionality: set, get, set_, set_at_, update, update_
    @abc.abstractmethod
    def entry_class(self, key: NestedKey) -> type:
        """Returns the class of an entry, possibly without reading the entry.

        It is the same as ``type(td.get(key))``. Prefer it to that expression
        when :meth:`~.get` is expensive, for example on a lazy stack or a
        tensordict stored on disk.

        Args:
            key (NestedKey): the key of the entry.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3), b=TensorDict(c=torch.zeros(3)))
            >>> td.entry_class("a")
            <class 'torch.Tensor'>
            >>> td.entry_class("b")
            <class 'tensordict._td.TensorDict'>

        """
        raise NotImplementedError

    def set(
        self,
        key: NestedKey,
        item: CompatibleType,
        inplace: bool = False,
        *,
        non_blocking: bool = False,
        **kwargs: Any,
    ) -> Self:
        """Sets an entry, and returns the tensordict.

        ``td.set(key, value)`` is the same as ``td[key] = value``. By default,
        the value replaces the existing entry (no data is copied). A tuple key
        sets a nested entry, and creates the missing sub-tensordicts. The
        leading dimensions of the value must match :attr:`~.batch_size`.
        Arrays and numbers are converted to tensors. Strings and other
        objects are stored as :class:`~tensordict.NonTensorData`.

        A ``bool``, ``int`` or ``float``, or a NumPy scalar of one of these
        kinds, written over an existing :class:`~tensordict.NonTensorData`
        entry is stored as a 0-d tensor, and the write emits a
        :class:`FutureWarning`. Starting with TensorDict 0.17, the default will
        change to storing it as :class:`~tensordict.NonTensorData`, like the
        entry it replaces.

        Args:
            key (NestedKey): the key to set: a string, or a tuple of strings
                for a nested entry. See :ref:`keys`.
            item (torch.Tensor, TensorDictBase or any object): the value to
                store.
            inplace (bool, optional): if ``True`` and the key exists, the value
                is copied into the existing tensor, which keeps its storage. If
                the key does not exist, the entry is added. For a write that
                raises on a missing key, use :meth:`~.set_`. Defaults to
                ``False``. See :ref:`inplace-conventions`.

        Keyword Args:
            non_blocking (bool, optional): if ``True``, copies between devices
                can run asynchronously with respect to the host. Defaults to
                ``False``.
            **kwargs: ignored, with a :class:`FutureWarning`. Starting with
                TensorDict 0.17, the default will change to raising a
                ``TypeError`` for them, as :meth:`~.set_` does.

        Returns:
            ``self``.

        Raises:
            RuntimeError: if the tensordict is locked (see :ref:`keys-locking`)
                and the call adds or replaces an entry, or if the leading
                dimensions of the value do not match the batch size.
            ValueError: if ``inplace=True`` and the value does not fit the
                shape of the existing entry. The error is also a
                ``RuntimeError``, the type that :meth:`~.update_` raises.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({}, batch_size=[3, 4])
            >>> td = td.set("x", torch.randn(3, 4))
            >>> y = torch.randn(3, 4, 5)
            >>> td = td.set("y", y, inplace=True)  # works, even if 'y' is not present yet
            >>> td = td.set("y", torch.zeros_like(y), inplace=True)
            >>> assert (y==0).all()  # y values are overwritten
            >>> td = td.set(("nested", "z"), torch.zeros(3, 4, 2))  # creates td["nested"]
            >>> td["nested"].batch_size
            torch.Size([3, 4])
            >>> td.set("y", torch.ones(5), inplace=True)  # the shape does not match
            Traceback (most recent call last):
            ...
            RuntimeError: batch dimension mismatch, got self.batch_size=torch.Size([3, 4]) and value.shape=torch.Size([5]) ...

        """
        if item is self:
            raise ValueError(_SELF_NESTING_ERROR.format(key))
        key_tuple = _unravel_key_to_tuple(key)
        if not key_tuple:
            raise KeyError(_GENERIC_NESTED_ERR.format(key))
        if kwargs:
            _warn_ignored_kwargs("set", kwargs)
        # inplace is loose here, but for set_ it is constraining. We translate it
        # to None to tell _set_str and others to drop it if the key isn't found
        inplace = BEST_ATTEMPT_INPLACE if inplace else False
        return self._set_tuple(
            key_tuple, item, inplace=inplace, validated=False, non_blocking=non_blocking
        )

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
    def _set_tuple(self, key, value, *, inplace, validated, non_blocking: bool):
        raise NotImplementedError

    @_lock_blocked
    def set_non_tensor(self, key: NestedKey, value: Any):
        """Stores ``value`` as non-tensor data at ``key``, and returns ``self``.

        The value is wrapped in a :class:`~tensordict.NonTensorData` with the
        batch size, device and names of the tensordict that holds it. A tuple
        key creates the missing nested tensordicts. Unlike :meth:`~.set`, a
        number is stored as non-tensor data too. ``td[key]`` and
        :meth:`~.get_non_tensor` return ``value``, and :meth:`~.get` returns
        the ``NonTensorData``. See :ref:`non-tensor-data`.

        Args:
            key (NestedKey): the key of the entry.
            value (Any): the value to store.

        Raises:
            RuntimeError: if the tensordict is locked.

        Examples:
            >>> from tensordict import TensorDict
            >>> data = TensorDict({}, batch_size=[])
            >>> data = data.set_non_tensor(("nested", "the string"), "a string!")
            >>> data["nested", "the string"]
            'a string!'
            >>> data.get(("nested", "the string"))
            NonTensorData(data=a string!, batch_size=torch.Size([]), device=None)

        """
        key = unravel_key(key)
        return self._set_non_tensor(key, value)

    def _set_non_tensor(self, key: NestedKey, value: Any):
        if isinstance(key, tuple):
            if len(key) == 1:
                return self._set_non_tensor(key[0], value)
            sub_td = self._get_str(key[0], None)
            if sub_td is None:
                sub_td = self._create_nested_str(key[0])
            sub_td._set_non_tensor(key[1:], value)
            return self
        from tensordict.tensorclass import NonTensorData

        self._set_str(
            key,
            NonTensorData(
                data=value,
                batch_size=self.batch_size,
                device=self.device,
                names=self._maybe_names(),
            ),
            validated=True,
            inplace=False,
            non_blocking=False,
        )
        return self

    def get_non_tensor(self, key: NestedKey, default=NO_DEFAULT):
        """Returns the content of the non-tensor entry at ``key``.

        A :class:`~tensordict.NonTensorData` returns its data, and a stack of
        non-tensor data returns a list. An entry that is not non-tensor data
        is returned as it is. See :ref:`non-tensor-data`.

        Args:
            key (NestedKey): the key of the entry.
            default (Any, optional): the value returned if the key is missing.
                If not passed, a missing key raises ``KeyError``.

        Raises:
            KeyError: if the key is missing and no default is passed.

        .. seealso:: :meth:`~.set_non_tensor`.

        Examples:
            >>> from tensordict import TensorDict
            >>> data = TensorDict({}, batch_size=[])
            >>> data = data.set_non_tensor(("nested", "the string"), "a string!")
            >>> data.get_non_tensor(("nested", "the string"))
            'a string!'
            >>> data.get_non_tensor("missing", "default")
            'default'

        """
        key = unravel_key(key)
        return self._get_non_tensor(key, default=default)

    def _get_non_tensor(self, key: NestedKey, default=NO_DEFAULT):
        if isinstance(key, tuple):
            if len(key) == 1:
                return self._get_non_tensor(key[0], default=default)
            subtd = self._get_str(key[0], default=default)
            if subtd is default:
                return subtd
            return subtd._get_non_tensor(key[1:], default=default)
        value = self._get_str(key, default=default)

        if is_non_tensor(value):
            from tensordict import NonTensorStack

            if isinstance(value, NonTensorStack) and not capture_non_tensor_stack():
                return value.tolist(as_linked_list=True)
            data = getattr(value, "data", None)
            if data is None:
                return value.tolist(as_linked_list=True)
            return data
        return value

    def is_non_tensor(self, key: NestedKey) -> bool:
        """Returns ``True`` if the entry at ``key`` is non-tensor data.

        Non-tensor data is a :class:`~tensordict.NonTensorData` or a
        :class:`~tensordict.NonTensorStack`. Unlike the function
        :func:`~tensordict.is_non_tensor`, which checks a value, this method
        checks the stored entry. This matters for a tensorclass, whose
        ``get`` returns the content of a non-tensor entry, not the
        ``NonTensorData``.

        Args:
            key (NestedKey): the key to check.

        Raises:
            KeyError: if the key is missing.

        .. seealso:: :meth:`~.set_non_tensor` and :meth:`~.get_non_tensor`.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"obs": torch.zeros(4)}, [])
            >>> td = td.set_non_tensor("label", "cat")
            >>> td.is_non_tensor("label")
            True
            >>> td.is_non_tensor("obs")
            False
        """
        key = unravel_key(key)
        return self._is_non_tensor_leaf(key)

    def _is_non_tensor_leaf(self, key: NestedKey):
        if isinstance(key, tuple):
            if len(key) == 1:
                return self._is_non_tensor_leaf(key[0])
            subtd = self._get_str(key[0], NO_DEFAULT)
            return subtd._is_non_tensor_leaf(key[1:])
        val = self._get_str(key, NO_DEFAULT)
        return _is_non_tensor(type(val))

    def filter_non_tensor_data(self) -> Self:
        """Returns a tensordict without the non-tensor entries, nested ones included.

        The other leaves are not copied.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3), b="text", n={"c": "text", "d": torch.zeros(3)}, batch_size=[3])
            >>> list(td.filter_non_tensor_data().keys(include_nested=True, leaves_only=True))
            ['a', ('n', 'd')]
        """

        def _filter(x):
            if not is_non_tensor(x):
                if is_tensor_collection(x):
                    return x.filter_non_tensor_data()
                return x

        return self._apply_nest(_filter, call_on_nested=True, filter_empty=False)

    def filter_empty_(self):
        """Removes the nested tensordicts that have no leaf, in place, and returns ``self``.

        A nested tensordict that holds only empty tensordicts is removed too.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3), b={}, c={"d": {}}, batch_size=[3])
            >>> list(td.filter_empty_().keys())
            ['a']
        """

        # A lazy stack filters each member, and a sub-tensordict is filtered
        # before it is checked: a nested key that names a member, such as
        # ("0", "a"), cannot be deleted from the stack (#2796). A lazy stack
        # is empty if all its members are: is_empty() only sees the keys that
        # all members share.
        def filter_td(td) -> bool:
            # Filters td in place and returns whether it is then empty.
            members = td._members_as_lists()
            if members is not None:
                return all([filter_td(member) for member in members])
            kept = False
            for key, val in list(td.items()):
                cls = type(val)
                if not _is_tensor_collection(cls):
                    continue
                if _pass_through(val):
                    if val.is_empty():
                        del td[key]
                elif filter_td(val._tensordict if _is_tensorclass(cls) else val):
                    del td[key]
                else:
                    kept = True
            return not kept and td.is_empty()

        filter_td(self)
        return self

    def _contains_str(self, key: str) -> bool:
        """Returns whether ``key`` is a key of ``self`` (not of a sub-tensordict), without reading data.

        :meth:`_convert_inplace` calls it with the str key of a ``_set_str``
        call. The base checks ``key in self.keys()``; a file or a store asks
        its storage.
        """
        return key in self.keys()

    def _convert_inplace(self, inplace, key):
        if inplace is not False:
            has_key = self._contains_str(key)
            if inplace is True and not has_key:  # inplace could be None
                raise KeyError(
                    _KEY_ERROR.format(key, type(self).__name__, sorted(self.keys()))
                )
            inplace = has_key
        return inplace

    def set_at_(
        self,
        key: NestedKey,
        value: CompatibleType,
        index: IndexType,
        *,
        non_blocking: bool = False,
    ) -> Self:
        """Writes ``value`` into the existing entry ``key`` at the index ``index``, in place.

        ``td.set_at_(key, value, index)`` is the same as ``td[key][index] = value``.

        A list, or a ``bool``, ``int`` or ``float`` (also as a NumPy scalar),
        written into a non-tensor entry is cast to a tensor, and the write
        emits a :class:`FutureWarning`. Starting with TensorDict 0.17, the
        default will change to storing it as non-tensor data. Wrap the value
        in :class:`~tensordict.NonTensorData` to store it as non-tensor data
        now, or pass a tensor to keep the tensor. ``td[index] = value`` writes
        through ``set_at_`` too, but announces another change: see
        :meth:`__setitem__`.

        The entry keeps its storage, so ``set_at_`` works on a locked
        tensordict.

        Args:
            key (NestedKey): the key of the entry. The key must exist.
            value (torch.Tensor or any compatible value): the value to write.
                It must fit ``td[key][index]``.
            index (int, slice, torch.Tensor or tuple): the index to write at,
                along the batch dimensions.

        Keyword Args:
            non_blocking (bool, optional): if ``True``, copies between devices
                can run asynchronously with respect to the host. Defaults to
                ``False``.

        Returns:
            ``self``.

        Raises:
            KeyError: if the key is missing.
            RuntimeError: if the value does not fit ``td[key][index]``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> x = torch.zeros(3, 4)
            >>> td = TensorDict(x=x, batch_size=[3, 4])
            >>> td = td.set_at_("x", value=torch.ones(1, 4), index=slice(1))
            >>> x[:, 0]
            tensor([1., 0., 0.])
        """
        key = _unravel_key_to_tuple(key)
        return self._set_at_tuple(
            key, value, _entry_index(index), validated=False, non_blocking=non_blocking
        )

    @abc.abstractmethod
    def _set_at_str(self, key, value, idx, *, validated, non_blocking: bool):
        raise NotImplementedError

    @abc.abstractmethod
    def _set_at_tuple(self, key, value, idx, *, validated, non_blocking: bool):
        raise NotImplementedError

    def set_(
        self,
        key: NestedKey,
        item: CompatibleType,
        *,
        non_blocking: bool = False,
    ) -> Self:
        """Copies a value into an existing entry, which keeps its storage, and returns ``self``.

        The existing tensor keeps its ``data_ptr()``, so other references to
        it see the new values. ``set_`` works on a locked tensordict. See
        :ref:`inplace-conventions`.

        Args:
            key (NestedKey): the key of the entry. The key must exist.
            item (torch.Tensor, TensorDictBase or any compatible value): the
                value to copy into the entry. Its leading dimensions must match
                the batch size, and it is broadcast to the shape of the entry.

        Keyword Args:
            non_blocking (bool, optional): if ``True``, copies between devices
                can run asynchronously with respect to the host. Defaults to
                ``False``.

        Returns:
            ``self``.

        Raises:
            KeyError: if the key is missing.
            RuntimeError: if the leading dimensions of the value do not match
                the batch size.
            ValueError: if the value does not fit the shape of the entry.
                The error is also a ``RuntimeError``, the type that
                :meth:`~.update_` raises.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> x = torch.ones(3, 4)
            >>> td = TensorDict(x=x, batch_size=[3, 4])
            >>> td = td.set_("x", torch.zeros(3, 4))
            >>> td["x"] is x, x.sum()
            (True, tensor(0.))

        """
        key_tuple = _unravel_key_to_tuple(key)
        if not key_tuple:
            raise KeyError(_GENERIC_NESTED_ERR.format(key))
        return self._set_tuple(
            key_tuple, item, inplace=True, validated=False, non_blocking=non_blocking
        )

    # Stack functionality
    def _stack_onto_(
        self,
        list_item: list[CompatibleType],
        dim: int,
    ) -> Self:
        """Stacks a list of tensordicts and writes the result in place into ``self``.

        This is the hook behind ``torch.stack(list_item, dim, out=self)``.

        Args:
            list_item (list of TensorDictBase): the tensordicts to stack.
            dim (int): dimension along which the tensordicts are stacked.

        Returns:
            self

        """
        for key in self.keys():
            vals = [td._get_str(key, None) for td in list_item]
            if all(v is None for v in vals):
                continue
            self.set_(key, torch.stack(vals, dim=dim))
        return self

    def _stack_onto_at_(
        self,
        list_item: list[CompatibleType],
        dim: int,
        idx: IndexType,
    ) -> Self:
        """Similar to _stack_onto_ but on a specific index. Only works with regular TensorDicts."""
        raise RuntimeError(
            f"Cannot call _stack_onto_at_ with {type(self).__name__}. "
            "Make sure your sub-classed tensordicts are turned into regular tensordicts by calling to_tensordict() "
            "before calling __getindex__ and stack."
        )

    def _default_get(self, key: NestedKey, default: Any = NO_DEFAULT) -> CompatibleType:
        if default is not NO_DEFAULT:
            return default
        else:
            # raise KeyError
            raise KeyError(
                _KEY_ERROR.format(key, type(self).__name__, sorted(self.keys()))
            )

    @overload
    def get(self, key): ...
    @overload
    def get(self, key, default): ...

    def get(self, key: NestedKey, *args, **kwargs) -> CompatibleType:
        """Returns the value stored at ``key``, or ``default`` if the key is missing.

        ``td.get(key)`` reads the same entry as ``td[key]``, with two
        differences: a missing key returns ``default`` where ``td[key]``
        raises ``KeyError``, and a non-tensor entry returns its
        :class:`~tensordict.NonTensorData` where ``td[key]`` returns the
        content.

        A lazy stack reads the entry of every stacked tensordict and stacks
        the results into a new tensor. If their shapes differ, this raises a
        ``RuntimeError``: pass ``as_list``, ``as_padded_tensor`` or
        ``as_nested_tensor`` to get a list, a padded tensor or a nested tensor
        instead. See :ref:`stacking-variable-size`.

        Args:
            key (NestedKey): the key to read. A tuple reads a nested entry:
                ``td.get(("a", "b"))`` is ``td["a"]["b"]``. See :ref:`keys`.
            default (Any, optional): the value returned if the key is missing.
                Defaults to ``None``, as for :meth:`dict.get`. A nested key
                that runs through a tensor is missing too: ``get`` returns
                the default passed to it, and without one raises a
                ``ValueError`` that names the tensor.

        Keyword Args:
            as_list (bool, optional): if ``True``, a lazy stack returns the
                entries as a list, one item per stacked tensordict.
                Exclusive with ``as_padded_tensor`` and ``as_nested_tensor``.
                Defaults to ``False``.
            as_padded_tensor (bool, optional): if ``True``, a lazy stack pads
                the entries to the largest size of each dimension, also across
                the levels of a lazy stack of lazy stacks, and stacks them
                into one tensor. Exclusive with ``as_list`` and
                ``as_nested_tensor``. Defaults to ``False``.
            as_nested_tensor (bool, optional): if ``True``, a lazy stack returns
                the entries as a nested tensor (:mod:`torch.nested`). A lazy
                stack of lazy stacks raises a ``ValueError``, as a nested
                tensor has a single nested level. Exclusive with ``as_list``
                and ``as_padded_tensor``. Defaults to ``False``.
            layout (torch.layout, optional): with ``as_nested_tensor=True``,
                the layout of the nested tensor: ``torch.jagged`` or
                ``torch.strided``. Defaults to ``torch.jagged``.
            padding_side (str, optional): with ``as_padded_tensor=True``, the
                side of every dimension that gets the padding: ``"left"`` or
                ``"right"``. Defaults to ``"right"``.
            padding_value (float, int or bool, optional): with
                ``as_padded_tensor=True``, the value of the padding. Defaults
                to ``0.0``.

        .. note:: A dense tensordict ignores the keyword arguments today and
            warns for ``as_list=True`` and ``as_nested_tensor=True``. Starting
            with TensorDict 0.17, the default will change to returning a list
            or a nested tensor, as a lazy stack does.

        .. versionchanged:: 0.7
            A missing key returns ``None`` instead of raising ``KeyError``.
            ``set_get_defaults_to_none(False)`` and ``TD_GET_DEFAULTS_TO_NONE=0``
            restore the old behavior. They are deprecated and will be removed
            in TensorDict 0.17.

        Examples:
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"x": 1, "nested": {"y": 2}}, batch_size=[])
            >>> td.get("x")
            tensor(1)
            >>> td.get(("nested", "y"))
            tensor(2)
            >>> print(td.get("z"))
            None
            >>> td.get("z", 0)
            0

            On a lazy stack of entries of different shapes:

            >>> import torch
            >>> from tensordict import lazy_stack
            >>> td = lazy_stack([
            ...     TensorDict({"x": torch.ones(1,)}),
            ...     TensorDict({"x": torch.ones(2,) * 2}),
            ... ])
            >>> td.get("x", as_list=True)
            [tensor([1.]), tensor([2., 2.])]
            >>> td.get("x", as_padded_tensor=True)
            tensor([[1., 0.],
                    [2., 2.]])
            >>> nested = td.get("x", as_nested_tensor=True)
            >>> nested.is_nested, nested.offsets()
            (True, tensor([0, 1, 3]))
        """
        key_tuple = _unravel_key_to_tuple(key)
        if not key_tuple:
            raise KeyError(_GENERIC_NESTED_ERR.format(key))
        # Find what the default is
        passed_default = True
        if args:
            default = args[0]
            if len(args) > 1:
                raise TypeError("Only one arg is allowed in TD.get.")
            elif "default" in kwargs:
                raise TypeError("'default' arg was passed twice.")
        elif "default" in kwargs:
            default = kwargs.pop("default")
            if args:
                raise TypeError("'default' arg was passed twice.")
        elif _GET_DEFAULTS_TO_NONE:
            default = None
            passed_default = False
        else:
            default = NO_DEFAULT
        try:
            result = self._get_tuple(key_tuple, default=default, **kwargs)
        except _KeyThroughLeafError:
            # A key that runs through a leaf is missing for a caller that
            # passed a default; without one, the error names the leaf.
            if not passed_default or default is NO_DEFAULT:
                raise
            return default
        if (
            kwargs
            and (
                ("as_list" in kwargs and kwargs["as_list"])
                or ("as_nested_tensor" in kwargs and kwargs["as_nested_tensor"])
            )
            and self._get_ignores_kwargs
            and not is_compiling()
        ):
            # the cheap checks come first: get is a hot path
            _warn_get_kwargs(self, kwargs, result, default)
        return result

    @abc.abstractmethod
    def _get_str(self, key, default, **kwargs):
        raise NotImplementedError

    def _get_tuple(self, key, default, **kwargs):
        """Reads the entry at ``key``: the storage part of :meth:`get`.

        :meth:`get` has unravelled ``key`` to a non-empty tuple of str and
        resolved ``default`` (``NO_DEFAULT`` when a missing key must raise
        ``KeyError``). A missing key returns ``default``; a path through an
        entry that is not a tensordict raises ``ValueError``. The h5 and zarr
        files and :class:`~tensordict.store.TensorDictStore` treat that path
        as a missing key, and return ``default``.
        """
        first = self._get_str(key[0], default, **kwargs)
        if len(key) == 1 or first is default:
            return first
        if isinstance(first, Tensor):
            # Tested first: under the Dynamo of torch 2.11, reading a missing
            # attribute of a tensor raises an internal error, not AttributeError.
            raise _key_through_leaf_error(key, 1, first)
        try:
            return first._get_tuple(key[1:], default=default, **kwargs)
        except AttributeError as err:
            _raise_if_not_a_tensordict(err, key, 1, first)

    def _get_tuple_maybe_non_tensor(self, key, default, **kwargs):
        result = self._get_tuple(key, default, **kwargs)
        if _pass_through(result) and not _is_unbatched(result):
            return _non_tensor_content(result)
        return result

    @overload
    def get_at(self, key, index): ...

    @overload
    def get_at(self, key, index, default): ...

    def get_at(
        self,
        key: NestedKey,
        *args,
        **kwargs,
    ) -> CompatibleType:
        """Returns the entry at ``key``, indexed with ``index``.

        ``td.get_at(key, index)`` is the same as ``td.get(key)[index]``, but a
        tensordict stored on disk or in a store reads only the indexed part.

        Args:
            key (NestedKey): the key to read.
            index (int, slice, torch.Tensor or tuple): the index to apply to
                the entry.
            default (Any, optional): the value returned if the key is missing.
                Defaults to ``None``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"x": torch.arange(3)}, batch_size=[3])
            >>> td.get_at("x", 1)
            tensor(1)
            >>> td.get_at("x", torch.tensor([True, False, True]))
            tensor([0, 2])
            >>> print(td.get_at("y", 1))
            None

        """
        # TODO: check that this works with masks, and add to docstring
        key_tuple = _unravel_key_to_tuple(key)
        if not key_tuple:
            raise KeyError(_GENERIC_NESTED_ERR.format(key))

        try:
            if len(args):
                index = args[0]
                args = args[1:]
            else:
                index = kwargs.pop("index")
        except KeyError:
            raise TypeError("index argument missing from get_at")

        # Find what the default is
        if args:
            default = args[0]
            if len(args) > 1:
                raise TypeError("only one (keyword) argument is allowed.")
        elif "default" in kwargs:
            default = kwargs.pop("default")
        elif _GET_DEFAULTS_TO_NONE:
            default = None
        else:
            default = NO_DEFAULT

        return self._get_at_tuple(key_tuple, _entry_index(index), default, **kwargs)

    def _get_at_str(self, key, idx, default, **kwargs):
        """Reads the entry ``key`` (a string) at ``idx``; see :meth:`_get_at_tuple`."""
        out = self._get_str(key, default, **kwargs)
        if out is default:
            return out
        return out[idx]

    def _get_at_tuple(self, key, idx, default, **kwargs):
        """Reads the entry ``key`` at ``idx``, for :meth:`get_at`.

        :meth:`get_at` has unravelled ``key`` into a non-empty tuple of
        strings, normalized ``idx`` with ``_entry_index`` and resolved
        ``default`` (``NO_DEFAULT`` when the key must exist); ``kwargs`` are
        the remaining keyword arguments of the call. The hook returns
        ``default`` itself when the entry is missing and ``default`` is not
        ``NO_DEFAULT``, raises ``KeyError`` when it is, and otherwise returns
        the entry indexed with ``idx``. The default reads the whole entry
        with :meth:`_get_tuple` and indexes it; a class that can read only the
        indexed part overrides it.
        """
        out = self._get_tuple(key, default, **kwargs)
        if out is default:
            return out
        return out[idx]

    def get_item_shape(self, key: NestedKey):
        """Returns the shape of the entry at ``key``, without reading the entry when possible.

        For a lazy stack, a dimension whose size differs between the stacked
        entries has size ``-1``.

        Args:
            key (NestedKey): the key of the entry.

        Raises:
            KeyError: if the tensordict has no entry at ``key``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict, lazy_stack
            >>> td = lazy_stack([TensorDict(x=torch.zeros(2)), TensorDict(x=torch.zeros(3))])
            >>> td.get_item_shape("x")
            torch.Size([2, -1])
        """
        return _shape(self.get(key, NO_DEFAULT))

    def update(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T | None = None,
        clone: bool = False,
        inplace: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
        is_leaf: Callable[[Type], bool] | None = None,
        update_batch_size: bool = False,
        ignore_lock: bool = False,
        **kwargs,
    ) -> Self:
        """Writes the entries of a dictionary or another tensordict into the tensordict, and returns ``self``.

        Each entry is written as with :meth:`~.set`: by reference, unless
        ``clone=True``. Nested tensordicts are merged, not replaced: the
        entries of the tensordict that the input does not have are kept.
        Keyword arguments are top-level entries: ``td.update(a=x)`` writes
        ``x`` at ``"a"``, and wins over an entry ``"a"`` of the input. A key
        named like an argument of ``update`` (``clone``, ``inplace``...) must
        go in the input dictionary.

        .. warning:: ``update`` is not atomic. If one entry fails (for
            example, its shape does not match the batch size), the entries
            before it are already written. Do not catch the exception and
            continue with the same tensordict. Update a copy, or make sure
            that the input is valid first.

        Args:
            input_dict_or_td (TensorDictBase or dict, optional): the entries to
                write. A dictionary can be nested and have tuple keys. If
                ``None``, only the keyword arguments are written. Defaults to
                ``None``.
            clone (bool, optional): if ``True``, clones the tensors of the
                input before writing them. Defaults to ``False``.
            inplace (bool, optional): if ``True``, a value whose key exists is
                copied into the existing tensor, which keeps its storage. A
                new key is added. Defaults to ``False``. See
                :ref:`inplace-conventions`.

        Keyword Args:
            non_blocking (bool, optional): if ``True``, copies between devices
                can run asynchronously with respect to the host. Defaults to
                ``False``.
            keys_to_update (sequence of NestedKey, optional): if passed, only
                these keys are written, without the cost of
                ``td.update(other.select(*keys_to_update))``. Defaults to
                ``None``, which writes every entry.
            is_leaf (Callable[[type], bool], optional): tells which types of
                values are written whole. The other values are merged key by
                key. Defaults to ``None``, which is
                :func:`~tensordict.is_leaf_nontensor`. See also
                :func:`~tensordict.default_is_leaf`.
            update_batch_size (bool, optional): if ``True``, sets the batch
                size of the tensordict to that of the input when they differ.
                A lazy stack is then refilled with copies of the tensordicts of
                the input. It needs ``keys_to_update=None`` and
                ``inplace=False``, and the keys of the tensordict must all be in
                the input. A batch size that is a prefix of the input's (``[]``
                for an input of batch size ``[2]``) is kept, with a
                :class:`FutureWarning`: starting with TensorDict 0.17, the
                default will change to replacing it as well. A batch size whose
                leading dims are those of the input (``[2, 3]`` for ``[2]``) is
                kept. Defaults to ``False``.
            ignore_lock (bool, optional): deprecated: ``True`` warns, and the
                argument will be removed in TensorDict 0.17. On a locked
                :class:`~tensordict.TensorDict`, it does not lift the lock: a
                new or replaced entry still raises ``RuntimeError``
                (`#2260 <https://github.com/pytorch/tensordict/issues/2260>`_).
                Use ``inplace=True``, or unlock the tensordict first, as in
                ``with td.unlock_(): td.update(other)``. Defaults to ``False``.
            **kwargs: entries to write at the top level, by key.

        Returns:
            ``self``.

        Raises:
            RuntimeError: if the tensordict is locked (see :ref:`keys-locking`)
                and the call adds or replaces an entry, or if the leading
                dimensions of a value do not match the batch size.

        .. note:: With ``update_batch_size=True``, a lazy stack of N
            tensordicts updated with a lazy stack of M > N tensordicts along
            the stack dimension gets copies of the M - N extra tensordicts
            appended, so ``update`` can grow a lazy stack. Without it, the
            extra tensordicts are left out.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3), batch_size=[3])
            >>> other = TensorDict(a=torch.ones(3), b=torch.ones(3, 4), batch_size=[3])
            >>> td = td.update(other)
            >>> td["a"] is other["a"]  # written by reference unless clone=True
            True
            >>> td = td.update(c=torch.zeros(3))  # keyword form
            >>> sorted(td.keys())
            ['a', 'b', 'c']

            Nested tensordicts are merged:

            >>> td = td.update({"nested": {"x": torch.zeros(3)}})
            >>> td = td.update({"nested": {"y": torch.ones(3)}})
            >>> sorted(td["nested"].keys())
            ['x', 'y']
        """
        if ignore_lock and ignore_lock is not _UPDATE_IGNORE_LOCK_WARNED:
            ignore_lock = _warn_update_ignore_lock()
        if (
            not (clone or inplace or update_batch_size or ignore_lock)
            and keys_to_update is None
            and (is_leaf is None or is_leaf is _is_leaf_nontensor)
            and (not kwargs or input_dict_or_td is None)
            and not is_compiling()
            and self._update_fast(
                kwargs if input_dict_or_td is None else input_dict_or_td
            )
        ):
            return self
        if not inplace and not ignore_lock and self.is_locked:
            raise RuntimeError(_LOCK_ERROR)
        if kwargs:
            if input_dict_or_td is None:
                input_dict_or_td = kwargs
            elif isinstance(input_dict_or_td, dict):
                input_dict_or_td = {**input_dict_or_td, **kwargs}
            else:
                self.update(
                    input_dict_or_td,
                    clone=clone,
                    inplace=inplace,
                    non_blocking=non_blocking,
                    keys_to_update=keys_to_update,
                    is_leaf=is_leaf,
                    update_batch_size=update_batch_size,
                    ignore_lock=ignore_lock,
                )
                input_dict_or_td = kwargs
        elif input_dict_or_td is None:
            return self
        if input_dict_or_td is self:
            # no op
            return self
        if is_leaf is None:
            is_leaf = _is_leaf_nontensor
        if keys_to_update is not None:
            if len(keys_to_update) == 0:
                return self
            keys_to_update = unravel_key_list(keys_to_update)
        return self._update_entries(
            input_dict_or_td,
            clone=clone,
            inplace=inplace,
            non_blocking=non_blocking,
            keys_to_update=keys_to_update,
            is_leaf=is_leaf,
            update_batch_size=update_batch_size,
            ignore_lock=ignore_lock,
        )

    def _update_fast(self, source: dict[str, CompatibleType] | T) -> bool:
        """Writes ``source`` into ``self`` without the checks of :meth:`update`, if that is safe, and returns whether it did.

        :meth:`update` calls it first, and only when it got default arguments
        (no ``clone``, ``inplace``, ``keys_to_update``, ``update_batch_size``,
        ``ignore_lock`` or custom ``is_leaf``, and no kwargs besides a missing
        source), outside of ``torch.compile``. ``source`` may be ``self``.
        The lock has not been checked yet: return ``False`` when ``self`` is
        locked.
        When it returns ``False``, :meth:`update` takes the general path, so a
        class that overrides it must write nothing before it returns
        ``False``. The base returns ``False``.
        """
        return False

    def _update_entries(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        *,
        clone: bool,
        inplace: bool,
        non_blocking: bool,
        keys_to_update: list[NestedKey] | None,
        is_leaf: Callable[[Type], bool],
        update_batch_size: bool,
        ignore_lock: bool,
    ) -> Self:
        """Writes the entries of ``input_dict_or_td`` into ``self``: the storage part of :meth:`update`.

        :meth:`update` has warned about ``ignore_lock``, checked the lock,
        merged the kwargs into ``input_dict_or_td`` and returned for ``None``,
        ``self`` and an empty ``keys_to_update``. ``input_dict_or_td`` is a
        dict or a tensor collection, ``is_leaf`` is set and ``keys_to_update``
        is ``None`` or the unravelled keys. The base checks the batch size,
        replaces it if ``update_batch_size`` asks for it, and writes each key
        with :meth:`_set_tuple`, merging the sub-tensordicts. Returns ``self``.
        """
        batch_size_changed = False
        is_collection = _is_tensor_collection(type(input_dict_or_td))
        if (
            is_collection
            # here, we do a loose check on the batch sizes: it could be that the source has batch_size (1,) and self (1, 2)
            # and that all the values have an appropriate shape for the new batch size.
            # What we want to catch is any batch size item that is obviously mismatching, like (1, 2, 3) and (1, 4, 3).
            and self.batch_size[: input_dict_or_td.batch_dims]
            != input_dict_or_td.batch_size[: self.batch_dims]
        ):
            if not update_batch_size:
                raise RuntimeError(
                    f"update() got a source with batch size {input_dict_or_td.batch_size} "
                    f"for a destination with batch size {self.batch_size}. "
                    "Pass update_batch_size=True to replace the batch size."
                )
            if inplace:
                raise RuntimeError(
                    "Source and destination tensor collection shapes mismatch, but "
                    "update was called with inplace=True, which cannot be achieved."
                )
            if keys_to_update is not None:
                raise RuntimeError(
                    "Updating tensordicts of different batch-size with keys_to_update "
                    "is currently not supported."
                )
            # This could be expensive but we must run it
            keys_source = set(input_dict_or_td.keys(True))
            keys_dest = set(self.keys(True))
            if not keys_dest.issubset(keys_source):
                raise RuntimeError(
                    "Some keys of the dest tensordict are not present in the source "
                    "during update with mismatching batch-size. "
                    f"batch_size of source={input_dict_or_td.batch_size}, batch_size of dest={self.batch_size}, "
                    f"keys in dest but not in source: {{{keys_dest - keys_source}}}."
                )
            # We can swap target with value if the batch sizes are incongruent. We must make sure the id of target
            # stays the same though
            self.batch_size = ()
            # Remove all leaves, and update
            ks = self.keys(True, True, is_leaf=is_leaf)
            self.exclude(*ks, inplace=True)
            self.batch_size = input_dict_or_td.batch_size
            self.update(input_dict_or_td, update_batch_size=True)
            return self
        if (
            update_batch_size
            and is_collection
            and self.batch_dims < input_dict_or_td.batch_dims
        ):
            # self's batch size is a prefix of the source's (#2957)
            _warn_update_batch_size_prefix(self.batch_size, input_dict_or_td.batch_size)

        for input_key, value in input_dict_or_td.items():
            key = _unravel_key_to_tuple(input_key)
            if not key:
                raise KeyError(_GENERIC_NESTED_ERR.format(input_key))
            firstkey, subkey = key[0], key[1:]
            if keys_to_update and not any(
                firstkey == ktu if isinstance(ktu, str) else firstkey == ktu[0]
                for ktu in keys_to_update
            ):
                continue
            target = self._get_str(firstkey, None)
            if clone and hasattr(value, "clone"):
                value = value.clone()
            elif clone:
                value = tree_map(torch.clone, value)
            elif (
                value is self
                and not self._update_stores_copy_of_self
                and (keys_to_update is None or target is None)
            ):
                # with keys_to_update, a merge of self into a sub-tensordict
                # can leave out the entry that holds the sub-tensordict
                raise ValueError(_SELF_NESTING_ERROR.format(input_key))
            # the key must be a string by now. Let's check if it is present
            if target is not None:
                if not is_leaf(type(target)) and not is_leaf(type(value)):
                    if subkey:
                        sub_keys_to_update = _prune_selected_keys(
                            keys_to_update, firstkey
                        )
                        target.update(
                            {subkey: value},
                            inplace=inplace,
                            clone=clone,
                            keys_to_update=sub_keys_to_update,
                            non_blocking=non_blocking,
                            update_batch_size=update_batch_size,
                            ignore_lock=ignore_lock,
                        )
                        if self._nested_reads_return_copies:
                            self._update_write_back(
                                firstkey,
                                target,
                                subkey,
                                value,
                                sub_keys_to_update,
                                is_leaf=is_leaf,
                                inplace=inplace,
                                non_blocking=non_blocking,
                            )
                        continue
                    elif isinstance(value, (dict,)) or _is_tensor_collection(
                        type(value)
                    ):
                        from tensordict._lazy import LazyStackedTensorDict

                        value_is_lazy_stack = isinstance(value, LazyStackedTensorDict)
                        target_is_lazy_stack = isinstance(target, LazyStackedTensorDict)
                        if value_is_lazy_stack and not target_is_lazy_stack:
                            sub_keys_to_update = _prune_selected_keys(
                                keys_to_update, firstkey
                            )
                            self._set_tuple(
                                key,
                                LazyStackedTensorDict(
                                    *target.unbind(value.stack_dim),
                                    stack_dim=value.stack_dim,
                                ).update(
                                    value,
                                    inplace=inplace,
                                    clone=clone,
                                    keys_to_update=sub_keys_to_update,
                                    non_blocking=non_blocking,
                                    update_batch_size=update_batch_size,
                                    ignore_lock=ignore_lock,
                                ),
                                validated=True,
                                inplace=False,
                                non_blocking=non_blocking,
                            )

                        else:
                            sub_keys_to_update = _prune_selected_keys(
                                keys_to_update, firstkey
                            )
                            target.update(
                                value,
                                inplace=inplace,
                                clone=clone,
                                non_blocking=non_blocking,
                                keys_to_update=sub_keys_to_update,
                                update_batch_size=update_batch_size,
                                ignore_lock=ignore_lock,
                            )
                            if self._nested_reads_return_copies:
                                self._update_write_back(
                                    firstkey,
                                    target,
                                    (),
                                    value,
                                    sub_keys_to_update,
                                    is_leaf=is_leaf,
                                    inplace=inplace,
                                    non_blocking=non_blocking,
                                )
                        continue
                # A tensor collection may still be a leaf so we need to duplicate the logic here
                if (
                    update_batch_size
                    and _is_tensor_collection(type(target))
                    and type(target) is type(value)
                    and target.shape != value.shape
                ):
                    batch_size_changed = True
                    from tensordict._lazy import LazyStackedTensorDict

                    # We can swap target with value if the batch sizes are incongruent. We must make sure the id of target
                    # stays the same though
                    if isinstance(target, LazyStackedTensorDict):
                        hook_out, hook_in = target._hook_out, target._hook_in
                        target.__init__(
                            *value.unbind(target.stack_dim),
                            stack_dim=target.stack_dim,
                            stack_dim_name=target._td_dim_name,
                        )
                        target._hook_out, target._hook_in = hook_out, hook_in
                    else:
                        target = target.exclude(
                            *target.keys(True, True, is_leaf=is_leaf), inplace=True
                        )
                        target.update(value, update_batch_size=update_batch_size)
                        target.batch_size = value.batch_size
                    continue

            self._set_tuple(
                key,
                value,
                inplace=BEST_ATTEMPT_INPLACE if inplace else False,
                validated=False,
                non_blocking=non_blocking,
            )
        if batch_size_changed:
            bd = self.batch_dims
            self.batch_size = ()
            # self.batch_size = ()
            self.auto_batch_size_(bd, keep_compliant_size=True)
        return self

    def _update_write_back(
        self,
        firstkey,
        target,
        subkey,
        value,
        keys_to_update,
        *,
        is_leaf,
        inplace,
        non_blocking,
    ):
        """Writes the leaves that update() merged into the copy ``target`` back to self.

        ``target`` is a copy of ``self[firstkey]`` and ``value`` was merged
        into it at ``subkey``. Only the leaves of ``value`` are written, so the
        other leaves of ``target``, which may have been read with the shape of
        the first element of a stack, are left as they are.
        """
        if not _is_tensor_collection(type(value)):
            from tensordict._td import TensorDict

            value = TensorDict(value, batch_size=[])
        keys_to_update = [_unravel_key_to_tuple(k) for k in keys_to_update or ()]
        for key in value.keys(True, True, is_leaf=is_leaf):
            key = (*subkey, *_unravel_key_to_tuple(key))
            if keys_to_update and not any(key[: len(k)] == k for k in keys_to_update):
                continue
            full_key = (firstkey, *key)
            self._set_tuple(
                full_key,
                target._get_tuple(key, NO_DEFAULT),
                # a locked tensordict takes no new keys, even in place
                inplace=inplace and (not self.is_locked or full_key in self.keys(True)),
                validated=True,
                non_blocking=non_blocking,
            )

    def update_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T | None = None,
        clone: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
        **kwargs,
    ) -> Self:
        """Copies the entries of a dictionary or tensordict into the existing entries, and returns ``self``.

        Each value is copied into the existing tensor, which keeps its
        storage, as with :meth:`~.set_`. ``update_`` therefore works on a
        locked tensordict. Every key of the input must exist in the tensordict.
        Keyword arguments are top-level entries, and win over an entry of the
        input with the same key. A key named like an argument of ``update_``
        must go in the input dictionary. See :ref:`inplace-conventions`.

        On a shared or memory-mapped tensordict, a non-tensor value that its
        shared buffer cannot store (for example ``None``, or an int where a
        string is shared) is skipped, with a :class:`UserWarning` that names
        its key.

        Args:
            input_dict_or_td (TensorDictBase or dict, optional): the entries to
                copy. If ``None``, only the keyword arguments are copied.
                Defaults to ``None``.
            clone (bool, optional): if ``True``, clones the tensors of the
                input before copying them. Defaults to ``False``.

        Keyword Args:
            non_blocking (bool, optional): if ``True``, copies between devices
                can run asynchronously with respect to the host. Defaults to
                ``False``.
            keys_to_update (sequence of NestedKey, optional): if passed, only
                these keys are copied, without the cost of
                ``td.update_(other.select(*keys_to_update))``. Defaults to
                ``None``, which copies every entry.
            **kwargs: entries to copy at the top level, by key.

        Returns:
            ``self``.

        Raises:
            KeyError: if a key of the input is missing from the tensordict.
            RuntimeError: if a value does not fit the shape of its entry.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> a = torch.ones(3)
            >>> td = TensorDict(a=a, b=torch.ones(3, 4), batch_size=[3])
            >>> _ = td.update_(TensorDict(a=torch.zeros(3), b=torch.zeros(3, 4), batch_size=[3]))
            >>> td["a"] is a, a.sum()
            (True, tensor(0.))
            >>> _ = td.update_(a=torch.ones(3))  # keyword form
            >>> a.sum()
            tensor(3.)

        """
        if (
            input_dict_or_td is None
            and "input_dict" in kwargs
            and "input_dict" not in self.keys()
        ):
            # the name of the argument in _SubTensorDict.update_ before 0.15
            warn_deprecated(
                "update_(input_dict=...)",
                removal="0.17",
                replacement="update_(input_dict_or_td=...)",
            )
            input_dict_or_td = kwargs.pop("input_dict")
        if kwargs:
            if input_dict_or_td is None:
                input_dict_or_td = kwargs
            elif isinstance(input_dict_or_td, dict):
                input_dict_or_td = {**input_dict_or_td, **kwargs}
            else:
                self.update_(
                    input_dict_or_td,
                    clone=clone,
                    non_blocking=non_blocking,
                    keys_to_update=keys_to_update,
                )
                input_dict_or_td = kwargs
        elif input_dict_or_td is None:
            return self
        if input_dict_or_td is self:
            # no op
            return self
        if keys_to_update is not None:
            if len(keys_to_update) == 0:
                return self
            keys_to_update = [_unravel_key_to_tuple(key) for key in keys_to_update]
        return self._update_(
            input_dict_or_td,
            clone=clone,
            non_blocking=non_blocking,
            keys_to_update=keys_to_update,
        )

    def _update_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        *,
        clone: bool,
        non_blocking: bool,
        keys_to_update: list[tuple[str, ...]] | None,
    ) -> Self:
        """Copies the leaves of ``input_dict_or_td`` into the leaves of ``self``: the storage part of :meth:`update_`.

        :meth:`update_` has merged the kwargs and returned for ``None``,
        ``self`` and an empty ``keys_to_update``. ``input_dict_or_td`` is a
        dict or a tensor collection, and ``keys_to_update`` is ``None`` or a
        list of key tuples. A key of the input that ``self`` lacks raises
        ``KeyError``; without ``keys_to_update``, the base ignores such keys
        when another key matches. A non-tensor value that a shared or
        memory-mapped leaf of ``self`` cannot store is skipped, and the base
        warns once with the keys it skipped. The base reads a dict as a
        ``TensorDict`` with ``batch_dims=self.batch_dims``. Returns ``self``.
        """
        if not _is_tensor_collection(type(input_dict_or_td)):
            from tensordict import TensorDict

            input_dict_or_td = TensorDict.from_dict(
                input_dict_or_td, batch_dims=self.batch_dims
            )
        entries = None
        if (
            keys_to_update is not None
            and type(self) is _TENSORDICT_TYPE
            and type(input_dict_or_td) is _TENSORDICT_TYPE
            and not is_compiling()
            and forward_ad._current_level < 0
        ):
            # Fast route for exact TensorDicts, in eager mode (compiled code
            # keeps the graphs and guards of the route below): pair the
            # selected tensors in the order of the _apply_nest route below,
            # and copy them with _foreach_copy_, which forward AD does not
            # support (hence no dual level). The route below writes each
            # other entry at its place in the order. Where each key names an
            # entry that is not a tensordict, the walk has read the path of
            # each key through exact TensorDicts: _reads_copies_under is False.
            wanted = set(keys_to_update)
            dests, sources, entries = [], [], []
            named = _keys_update_pairs(
                input_dict_or_td,
                self,
                (),
                () in wanted,
                wanted,
                torch.is_grad_enabled(),
                dests,
                sources,
                entries,
            )
            if named < 0 or (
                named != len(wanted) and _reads_copies_under(self, keys_to_update)
            ):
                entries = None
            elif not entries:
                if dests:
                    _copy_leaves(dests, sources, non_blocking, exact=True)
                return self
        if (
            keys_to_update is not None
            and entries is None
            and _reads_copies_under(self, keys_to_update)
        ):
            # the _apply_nest route below would write into the copies that a
            # nested lazy stack or file reads: write the listed entries on the
            # route without keys_to_update instead, which writes the stored
            # data (#2780). A listed entry that self lacks raises KeyError, as
            # on the _apply_nest route, but before anything is written.
            try:
                input_dict_or_td = input_dict_or_td.select(
                    *keys_to_update, strict=False
                )
            except NotImplementedError:
                # a file cannot select its entries
                input_dict_or_td = input_dict_or_td.to_tensordict().select(
                    *keys_to_update, strict=False
                )
            dest_keys = self.keys(True, True, is_leaf=_is_leaf_nontensor)
            for key in input_dict_or_td.keys(True, True, is_leaf=_is_leaf_nontensor):
                if key not in dest_keys:
                    raise KeyError(
                        f"The key {_unravel_key_to_tuple(key)} was not found in the dest tensordict."
                    )
            keys_to_update = None
        # The tensor leaves of self that the copy_ calls of _apply_nest below
        # change, to write back where they are copies of the stored data. A
        # lazy stack in the input splits self into its elements: those leaves
        # are not recorded, and stay unchanged.
        written = None
        # The keys of the non-tensor values that a shared or memory-mapped
        # leaf of self cannot store: they are skipped on every route (#2487).
        skipped = []

        if keys_to_update is not None:
            if self._reads_return_copies and not _has_lazy_stack(input_dict_or_td):
                written = []

            named = True

            def inplace_update(name, source, dest):
                if source is None:
                    return None
                name = _unravel_key_to_tuple(name)
                for key in keys_to_update:
                    if key == name[: len(key)]:
                        if dest is None:
                            raise KeyError(
                                f"The key {name} was not found in the dest tensordict."
                            )
                        if not isinstance(dest, Tensor):
                            source = _stored_value(source, dest)
                            if source is None:
                                # a str for a flat key, as on the other routes
                                skipped.append(name if len(name) > 1 else name[0])
                                return None
                        dest.copy_(source, non_blocking=non_blocking)
                        if written is not None and isinstance(dest, Tensor):
                            written.append((name, dest))

            if entries is not None:
                # the fast route: self is an exact TensorDict, which reads
                # no copies (written is None)
                start = 0
                for end, name, source, dest, nested in entries:
                    if start < end:
                        _copy_leaves(
                            dests[start:end],
                            sources[start:end],
                            non_blocking,
                            exact=True,
                        )
                        start = end
                    if not nested:
                        inplace_update(name, source, dest)
                    else:
                        # as TensorDict._apply_nest calls it on a nested node
                        source._apply_nest(
                            inplace_update,
                            dest,
                            inplace=False,
                            batch_size=None,
                            device=NO_DEFAULT,
                            checked=False,
                            named=True,
                            nested_keys=True,
                            default=None,
                            prefix=name,
                            filter_empty=True,
                            is_leaf=_is_leaf_nontensor,
                            out=None,
                            call_on_unbatched=False,
                        )
                if start < len(dests):
                    _copy_leaves(dests[start:], sources[start:], non_blocking, True)
                if skipped:
                    _warn_update_skipped(skipped)
                return self

        else:
            # Fastest route using _foreach_copy_
            try:
                plan = self._update_plan(input_dict_or_td, non_blocking=non_blocking)
            except _NestedReadsCopies:
                # a nested tensordict reads copies, such as an h5 file: update
                # a copy of self in which it is a copy of its leaves, written
                # back as by the in-place methods (#2775)
                return _on_copies(
                    self._inplace_on_nested_copies,
                    TensorDictBase._update_,
                    input_dict_or_td,
                    clone=clone,
                    non_blocking=non_blocking,
                    keys_to_update=keys_to_update,
                )
            if plan is not None:
                vals, other_val, finish = plan
                _copy_leaves(vals, other_val, non_blocking)
                if finish is not None:
                    finish()
                return self
            if self._reads_return_copies and not _has_lazy_stack(input_dict_or_td):
                written = []

            def inplace_update(name, source, dest):
                if source is None:
                    return None
                if dest is None:
                    raise KeyError(
                        f"The key {name} was not found in the dest tensordict."
                    )
                if not isinstance(dest, Tensor):
                    source = _stored_value(source, dest)
                    if source is None:
                        skipped.append(name)
                        return None
                dest.copy_(source, non_blocking=non_blocking)
                if written is not None and isinstance(dest, Tensor):
                    written.append((name, dest))

            named = True

        input_dict_or_td._apply_nest(
            inplace_update,
            self,
            nested_keys=True,
            default=None,
            filter_empty=True,
            named=named,
            is_leaf=_is_leaf_nontensor,
        )
        if written:
            self._write_back(*zip(*written), "update_")
        if skipped:
            _warn_update_skipped(skipped)
        return self

    def _update_plan(
        self, input_dict_or_td: TensorDictBase, *, non_blocking: bool
    ) -> tuple[list[Tensor], list[Tensor], Callable[[], None] | None] | None:
        """Pairs the tensor leaves that the fast route of :meth:`_update_` copies, and writes nothing.

        Returns ``(dests, sources, finish)``: the update copies each tensor of
        ``sources`` into the leaf of ``self`` at the same position in
        ``dests``, then calls ``finish`` unless it is ``None``. ``finish``
        writes the leaves back where ``self`` reads copies, and copies the
        non-tensor leaves. Returns ``None`` where no tensor leaf of
        ``input_dict_or_td``, a tensor collection, matches one of ``self``,
        and raises ``_NestedReadsCopies`` as ``_items_list`` does. A lazy
        stack overrides it with the plans of its members, which it checks
        before it writes any member (#2776).
        """
        if type(input_dict_or_td) is type(self) and self._is_dense_node():
            # two trees of exact TensorDicts with the same keys and only plain
            # tensor leaves: pair the leaves in the order of _items_list below,
            # in one walk. Any other entry takes the general route (#2859).
            dests = self._tensordict
            sources = input_dict_or_td._tensordict
            if dests and len(dests) == len(sources):
                vals = []
                other_val = []
                for key, val in dests.items():
                    other = sources.get(key)
                    cls = type(val)
                    if cls is Tensor:
                        if type(other) is not Tensor:
                            break
                        vals.append(val)
                        other_val.append(other)
                    elif not (
                        cls is _TENSORDICT_TYPE
                        and type(other) is cls
                        and _pair_dense_leaves(
                            val._tensordict, other._tensordict, vals, other_val
                        )
                    ):
                        break
                else:
                    if vals:
                        return vals, other_val, None
        keys, vals = self._items_list(True, True)
        source = dict(zip(*input_dict_or_td._items_list(True, True)))
        new_keys = [key for key in keys if key in source]
        unmatched = len(new_keys) < min(len(keys), len(source))
        if (
            unmatched
            and (new_keys or _has_typed_tensordict(input_dict_or_td))
            and _has_lazy_stack(input_dict_or_td)
        ):
            # a nested lazy stack lists the leaves of each member, under
            # keys that self lacks, which leaves keys of self unmatched,
            # and a TypedTensorDict over a lazy stack, also a nested one,
            # lists all its leaves so (#2772): read those keys stacked
            # instead (#2768)
            for key in keys:
                if key not in source:
                    val = input_dict_or_td._get_tuple(_unravel_key_to_tuple(key), None)
                    if isinstance(val, Tensor):
                        source[key] = val
            new_keys = [key for key in keys if key in source]
        elif (
            unmatched
            # the dense tensordicts skip the walk of _has_lazy_stack
            and _has_unmatched_member_key(keys, source)
            and _has_lazy_stack(self)
        ):
            # a lazy stack of self lists the leaves of each member, under
            # keys that the input lacks: read those keys at the index of
            # each member instead (#2775)
            for key in keys:
                if key not in source:
                    val = _read_at_members(
                        self, input_dict_or_td, _unravel_key_to_tuple(key)
                    )
                    if val is not None:
                        source[key] = val
            new_keys = [key for key in keys if key in source]
        other_val = [source[key] for key in new_keys]
        if not new_keys:
            return None
        if len(other_val) != len(vals):
            vals = dict(zip(keys, vals))
            vals = [vals[k] for k in new_keys]
        has_non_tensor = input_dict_or_td._has_non_tensor
        if not self._reads_return_copies and not has_non_tensor:
            return vals, other_val, None

        def finish():
            if self._reads_return_copies:
                self._write_back(new_keys, vals, "update_")
            if not has_non_tensor:
                return
            from tensordict import NonTensorStack

            # _items_list leaves out the non-tensor leaves: copy them here.
            # Like the tensor leaves, keys missing in self are skipped.
            # copy_ raises for a NonTensorStack into a NonTensorData:
            # these leaves are skipped too.
            skipped = []

            def inplace_update(name, source, dest):
                if (
                    _is_non_tensor(type(source))
                    and _is_non_tensor(type(dest))
                    and (
                        not isinstance(source, NonTensorStack)
                        or isinstance(dest, NonTensorStack)
                    )
                ):
                    source = _stored_value(source, dest)
                    if source is None:
                        skipped.append(name)
                    else:
                        dest.copy_(source, non_blocking=non_blocking)

            input_dict_or_td._apply_nest(
                inplace_update,
                self,
                nested_keys=True,
                default=None,
                filter_empty=True,
                named=True,
                is_leaf=_is_leaf_nontensor,
            )
            if skipped:
                _warn_update_skipped(skipped)

        return vals, other_val, finish

    @_update_at_aliases
    def update_at_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        idx: IndexType,
        clone: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
    ) -> Self:
        """Copies the entries of a dictionary or tensordict into the existing entries at ``idx``.

        ``td.update_at_(other, idx)`` writes each entry as
        ``td.set_at_(key, other[key], idx)``: the existing tensors keep their
        storage, so ``update_at_`` works on a locked tensordict. Every key of
        the input must exist in the tensordict. Tensor and non-tensor leaves
        are supported. For a faster copy of tensors only, use
        :meth:`~.copy_at_`.

        Args:
            input_dict_or_td (TensorDictBase or dict): the entries to copy.
                Their leading dimensions are those of ``td[idx]``.
            idx (int, slice, torch.Tensor or tuple): the index to write at,
                along the batch dimensions.
            clone (bool, optional): if ``True``, clones the tensors of the
                input before copying them. Defaults to ``False``.

        Keyword Args:
            non_blocking (bool, optional): if ``True``, copies between devices
                can run asynchronously with respect to the host. Defaults to
                ``False``.
            keys_to_update (sequence of NestedKey, optional): if passed, only
                these keys are copied. Defaults to ``None``, which copies every
                entry.

        Returns:
            ``self``.

        Raises:
            KeyError: if a key of the input is missing from the tensordict.
            RuntimeError: if a value does not fit the shape of its entry.

        .. deprecated:: 0.15
            The keyword names ``index`` (for ``idx``) and ``input_dict`` (for
            ``input_dict_or_td``) warn, and will be removed in TensorDict 0.17.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3, 4, 5), b=torch.zeros(3, 4, 10), batch_size=[3, 4])
            >>> other = TensorDict(a=torch.ones(1, 4, 5), b=torch.ones(1, 4, 10), batch_size=[1, 4])
            >>> _ = td.update_at_(other, slice(1, 2))
            >>> td["a"].sum(dim=(1, 2))
            tensor([ 0., 20.,  0.])

        """
        if isinstance(idx, tuple) and not idx:
            return self.update_(
                input_dict_or_td=input_dict_or_td,
                keys_to_update=keys_to_update,
                clone=clone,
                non_blocking=non_blocking,
            )
        if keys_to_update is not None:
            if len(keys_to_update) == 0:
                return self
            keys_to_update = [
                _unravel_key_to_tuple(key) for key in unravel_key_list(keys_to_update)
            ]
        return self._update_at_(
            input_dict_or_td,
            _entry_index(idx),
            clone=clone,
            non_blocking=non_blocking,
            keys_to_update=keys_to_update,
        )

    def _update_at_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        idx: IndexType,
        *,
        clone: bool,
        non_blocking: bool,
        keys_to_update: list[tuple[str, ...]] | None,
    ) -> Self:
        """Copies the leaves of ``input_dict_or_td`` into ``self`` at ``idx``: the storage part of :meth:`update_at_`.

        :meth:`update_at_` has sent ``idx == ()`` to :meth:`update_`,
        returned for an empty ``keys_to_update`` and passed ``idx`` through
        ``_entry_index``. ``input_dict_or_td`` is a dict or a tensor
        collection, and ``keys_to_update`` is ``None`` or a list of key
        tuples. The base writes each leaf as :meth:`set_at_` does. Returns
        ``self``.
        """
        if isinstance(input_dict_or_td, dict):
            input_dict_or_td = _flatten_nested_dicts(input_dict_or_td)
        for key, value in input_dict_or_td.items():
            firstkey, *nextkeys = _unravel_key_to_tuple(key)
            if keys_to_update:
                value = _select_entry((firstkey, *nextkeys), value, keys_to_update)
                if value is None:
                    continue
            if not _is_accepted_class(type(value)):
                raise TypeError(
                    f"Expected value to be one of types {_ACCEPTED_CLASSES} "
                    f"but got {type(value)}"
                )
            if clone:
                value = value.clone()
            # as set_at_ does: update_at_ has passed idx through _entry_index
            self._set_at_tuple(
                (firstkey, *nextkeys),
                value,
                idx,
                validated=False,
                non_blocking=non_blocking,
            )
        return self

    def _update_at_fast(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        idx: IndexType,
        clone: bool,
        *,
        non_blocking: bool,
        keys_to_update: Sequence[NestedKey] | None,
    ) -> Any:
        return NotImplemented

    def replace(self, *args, **kwargs):
        """Returns a shallow copy of the tensordict with some entries replaced.

        The copy is made with :meth:`~.copy`, and the new entries are written
        into it with :meth:`~.update`: nested tensordicts are merged, and the
        tensordict does not change. With no entry to write, ``replace``
        returns ``self``.

        When both a positional argument and ``**kwargs`` are given, the
        keyword entries are merged into a copy of the positional argument
        (with ``dict`` unpacking for a dictionary, with :meth:`~.copy` and
        :meth:`~.update` for a tensordict), so the argument does not change.
        A tensordict backed by a file or a store, such as a
        :class:`~tensordict.PersistentTensorDict`, shares that storage with
        its copy and still receives the keyword entries.

        Args:
            *args (dict or TensorDictBase): at most one, the entries to write.
            **kwargs: top-level entries to write, by key. They take
                precedence over the entries of the positional argument.

        Returns:
            A shallow copy of the tensordict, or ``self`` if there is no entry
            to write.

        Raises:
            RuntimeError: if more than one positional argument is passed, or if
                it is not a dictionary or a tensordict.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3), b=torch.zeros(3), batch_size=[3])
            >>> new = td.replace(a=torch.ones(3))
            >>> new["a"], td["a"], new["b"] is td["b"]
            (tensor([1., 1., 1.]), tensor([0., 0., 0.]), True)

        """
        if is_compiling() and not args:
            if not kwargs:
                return self
            result = self.copy()
            for k, v in kwargs.items():
                result[k] = v
            return result
        if args:
            if len(args) > 1:
                raise RuntimeError(
                    "Only a single argument containing a dictionary-like "
                    f"structure of entries to replace can be passed to replace. Received {len(args)} "
                    f"arguments instead."
                )
            dict_to_replace = args[0]
            # Merge kwargs into a copy: the caller's argument must not change.
            if kwargs:
                if isinstance(dict_to_replace, dict):
                    dict_to_replace = {**dict_to_replace, **kwargs}
                elif is_tensor_collection(dict_to_replace):
                    dict_to_replace = dict_to_replace.copy()
                    dict_to_replace.update(kwargs)
        else:
            dict_to_replace = kwargs
        is_dict = isinstance(dict_to_replace, dict)
        if is_dict:
            if not dict_to_replace:
                return self
        else:
            if not is_tensor_collection(dict_to_replace):
                raise RuntimeError(
                    f"Cannot use object type {type(dict_to_replace)} to update values in tensordict."
                )
            if dict_to_replace.is_empty():
                return self
        result = self.copy()
        # using update makes sure that any optimization (e.g. for lazy stacks) is done properly
        result.update(dict_to_replace)
        return result

    @_lock_blocked
    def create_nested(self, key):
        """Adds an empty nested tensordict at ``key``, and returns ``self``.

        The new tensordict has the batch size, device and names of the
        tensordict it is added to. An existing entry at ``key`` is replaced.
        For a tuple key, a plain :class:`~tensordict.TensorDict` keeps a
        plain, unlocked ``TensorDict`` already on the path with its entries:
        ``("a", "b")`` adds ``"b"`` to an existing ``td["a"]``. A lazy stack of
        plain ``TensorDict`` instances does this in each member, and a
        tensorclass or a :class:`~tensordict.nn.TensorDictParams` follows the
        rule of the tensordict it wraps. Any other entry on the path, such as a
        tensor, a lazy stack, a tensorclass, a typed tensordict or a locked
        tensordict, is replaced by a new empty tensordict, and so is every
        level of the path when the tensordict is of any other type, such as a
        view, a typed tensordict or a file.

        Args:
            key (NestedKey): the key of the new tensordict.

        Raises:
            RuntimeError: if the tensordict is locked.

        Examples:
            >>> from tensordict import TensorDict
            >>> data = TensorDict({}, [3, 4, 5])
            >>> _ = data.create_nested(("some", "nested", "value"))
            >>> data["some", "nested", "value"].batch_size
            torch.Size([3, 4, 5])
            >>> list(data.keys(include_nested=True))
            [('some', 'nested', 'value'), ('some', 'nested'), 'some']
            >>> _ = data.create_nested(("some", "other"))
            >>> list(data.keys(include_nested=True))
            [('some', 'nested', 'value'), ('some', 'nested'), ('some', 'other'), 'some']
        """
        key = _unravel_key_to_tuple(key)
        self._create_nested_tuple(key)
        return self

    def _create_nested_str(self, key):
        out = self.empty()
        self._set_str(key, out, inplace=False, validated=True, non_blocking=False)
        return out

    def _create_nested_tuple(self, key):
        """Adds an empty nested tensordict at the tuple ``key``.

        ``create_nested`` calls it with the key unravelled to a non-empty tuple.
        The hook puts a new empty tensordict at the last level, and returns
        nothing. A plain ``TensorDict`` goes down into a plain, unlocked
        ``TensorDict`` already on the path, so it keeps its entries, and
        replaces any other entry with a new empty tensordict. Every other
        class replaces each level of the path. ``LazyStackedTensorDict``
        overrides it: a stack of plain ``TensorDict`` instances hands the key
        to each member.
        """
        from tensordict._td import TensorDict

        if len(key) > 1:
            # Only a plain TensorDict keeps a plain, unlocked TensorDict on
            # the path. Anything else is replaced, level by level.
            td = self._get_str(key[0], None) if type(self) is TensorDict else None
            if type(td) is not TensorDict or td.is_locked:
                td = self._create_nested_str(key[0])
            td._create_nested_tuple(key[1:])
        else:
            self._create_nested_str(key[0])

    def copy_(self, tensordict: T, non_blocking: bool = False) -> Self:
        """Copies the entries of ``tensordict`` into the existing entries, and returns ``self``.

        The same as :meth:`~.update_`, under the name of
        :meth:`torch.Tensor.copy_`: the existing tensors keep their storage,
        and every key of ``tensordict`` must exist in the tensordict.

        Args:
            tensordict (TensorDictBase or dict): the entries to copy.
            non_blocking (bool, optional): if ``True``, copies between devices
                can run asynchronously with respect to the host. Defaults to
                ``False``.

        Raises:
            KeyError: if a key of ``tensordict`` is missing from the
                tensordict.
            RuntimeError: if a value does not fit the shape of its entry.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> x = torch.zeros(3)
            >>> td = TensorDict(x=x, batch_size=[3])
            >>> _ = td.copy_(TensorDict(x=torch.ones(3), batch_size=[3]))
            >>> x
            tensor([1., 1., 1.])
        """
        return self.update_(tensordict, non_blocking=non_blocking)

    def copy_at_(
        self,
        tensordict: T,
        idx: IndexType,
        non_blocking: bool = False,
        *,
        fast: bool = True,
    ) -> Self:
        """Copies the tensors of ``tensordict`` into the existing entries at ``idx``, and returns ``self``.

        ``copy_at_`` is a faster :meth:`~.update_at_` for the hot paths where
        every key of ``tensordict`` is a tensor entry of the tensordict. With
        ``fast=True``, a ``tensordict`` with only non-tensor entries raises a
        ``RuntimeError``. If it also holds tensors, its non-tensor entries are
        skipped, not copied, with a ``FutureWarning``: starting with
        TensorDict 0.17, the default will change to raising the same
        ``RuntimeError``. Use ``fast=False`` or :meth:`~.update_at_` to write
        them.

        Args:
            tensordict (TensorDictBase): the entries to copy. Their leading
                dimensions are those of ``td[idx]``.
            idx (int, slice, torch.Tensor or tuple): the index to write at,
                along the batch dimensions.
            non_blocking (bool, optional): if ``True``, copies between devices
                can run asynchronously with respect to the host. Defaults to
                ``False``.

        Keyword Args:
            fast (bool, optional): if ``True``, uses only the tensor-only copy,
                and raises a ``RuntimeError`` if the source, the destination or
                the index does not allow it (for example a dictionary source, a
                tensordict that is not a :class:`~tensordict.TensorDict`, a
                tensor index, a value that needs broadcasting, or a key missing
                from the tensordict). If ``False``, calls
                :meth:`~.update_at_`. Defaults to ``True``.

                .. versionchanged:: 0.15
                    ``fast=None``, deprecated in 0.14, is no longer accepted.

        Returns:
            ``self``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(x=torch.zeros(3, 2), batch_size=[3])
            >>> _ = td.copy_at_(TensorDict(x=torch.ones(2), batch_size=[]), 1)
            >>> td["x"][:, 0]
            tensor([0., 1., 0.])
        """
        if fast is None:
            raise TypeError(
                "copy_at_(..., fast=None) is no longer supported. Pass "
                "fast=True to require the optimized tensor-only copy path, or "
                "fast=False for the general update_at_ semantics."
            )
        elif fast is False:
            return self.update_at_(tensordict, idx, non_blocking=non_blocking)
        result = self._update_at_fast(
            input_dict_or_td=tensordict,
            idx=idx,
            clone=False,
            non_blocking=non_blocking,
            keys_to_update=None,
        )
        if result is not NotImplemented:
            return result
        if fast:
            raise RuntimeError(
                "copy_at_(..., fast=True) requires the optimized tensor-only "
                "copy path, but the source, destination, or index is not "
                "compatible. Use fast=False or update_at_ for the general "
                "update semantics."
            )
        return self.update_at_(tensordict, idx, non_blocking=non_blocking)

    def is_empty(self) -> bool:
        """Returns ``True`` if the tensordict has no leaf.

        Empty sub-tensordicts do not count as leaves. A lazy stack counts only
        the tensors that all its members share: it returns ``True`` when they
        share none, and warns when some member holds a leaf. Starting with
        TensorDict 0.17, the default will change to returning ``True`` only if
        every member is empty.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> TensorDict().is_empty()
            True
            >>> TensorDict({"nested": TensorDict()}).is_empty()
            True
            >>> TensorDict({"a": torch.zeros(())}).is_empty()
            False
        """
        if is_compiling():
            # Dynamo cannot close a key generator that is left before its end.
            return not list(self.keys(True, True))
        for _ in self.keys(True, True):
            return False
        members = self._members_as_lists()
        if (
            members is not None
            and _members_hold_leaf(members)
            and not _in_dense_is_empty()
        ):
            _warn_lazy_is_empty()
        return True

    # Dict features: setdefault, items, values, keys, ...
    def setdefault(
        self, key: NestedKey, default: CompatibleType | Any, inplace: bool = False
    ) -> CompatibleType:
        """Returns the entry at ``key``, after writing ``default`` there if the key is missing.

        As for :meth:`dict.setdefault`, an existing entry is not changed. The
        result is read with :meth:`~.get`, so a non-tensor entry returns its
        :class:`~tensordict.NonTensorData`.

        Args:
            key (NestedKey): the key of the entry.
            default (torch.Tensor, TensorDictBase or any object): the value
                written if the key is missing.
            inplace (bool, optional): passed to :meth:`~.set` when ``default``
                is written. Defaults to ``False``.

        Raises:
            RuntimeError: if the key is missing and the tensordict is locked,
                or if the leading dimensions of ``default`` do not match the
                batch size.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({}, batch_size=[3])
            >>> td.setdefault("a", torch.zeros(3))
            tensor([0., 0., 0.])
            >>> td.setdefault("a", torch.ones(3))  # "a" exists: not changed
            tensor([0., 0., 0.])

        """
        if key not in self.keys(include_nested=isinstance(key, tuple)):
            self.set(key, default, inplace=inplace)
        return self.get(key)

    def items(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf=None,
        *,
        sort: bool = False,
    ) -> Iterator[tuple[str, CompatibleType]]:  # noqa: D417
        """Returns a generator of the ``(key, value)`` pairs of the tensordict.

        By default, only the top-level entries are returned, and a nested
        tensordict is returned as one value. Non-tensor data comes as a
        :class:`~tensordict.NonTensorData`, not as the object that
        ``td[key]`` returns. See :ref:`keys` for nested keys and leaves.

        Args:
            include_nested (bool, optional): if ``True``, the entries of the
                nested tensordicts are also returned, under nested keys (tuples
                of strings). Defaults to ``False``.
            leaves_only (bool, optional): if ``True``, only the leaves are
                returned, and nested tensordicts are skipped. Defaults to
                ``False``.
            is_leaf (Callable, optional): a function that takes a class and
                returns ``True`` if values of that class count as leaves. It
                filters the result only when ``leaves_only=True``. It does not
                stop the recursion: with ``include_nested=True``, a nested
                tensordict is still searched even if ``is_leaf`` returns
                ``True`` for it. Defaults to ``None``, which uses
                :func:`~tensordict.default_is_leaf`: tensordicts,
                tensorclasses and non-tensor data are not leaves. Pass
                :func:`~tensordict.is_leaf_nontensor` to count non-tensor data
                as leaves.

        Keyword Args:
            sort (bool, optional): if ``True``, the keys are sorted. Nested keys
                are sorted by their joined name: ``("a", "key")`` sorts as
                ``"a.key"``. Sorting takes extra time on large tensordicts.
                Defaults to ``False``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(
            ...     a=torch.zeros(2),
            ...     nested=TensorDict(b=torch.ones(2, 3), batch_size=[2]),
            ...     batch_size=[2],
            ... )
            >>> for key, value in td.items(include_nested=True, leaves_only=True):
            ...     print(key, value.shape)
            a torch.Size([2])
            ('nested', 'b') torch.Size([2, 3])

        """
        if (
            is_leaf is _NESTED_TENSORS_AS_LISTS
            or is_leaf is _NESTED_TENSORS_AS_LISTS_NONTENSOR
        ):
            members = self._members_as_lists()
            if members is not None:
                for i, td in enumerate(members):
                    _check_member_as_lists(td, is_leaf)
                    for key, val in td.items(
                        include_nested=include_nested,
                        leaves_only=leaves_only,
                        is_leaf=is_leaf,
                        sort=sort,
                    ):
                        if isinstance(key, str):
                            yield (str(i), key), val
                        else:
                            yield (str(i), *key), val
                return
            # a file nested in a tensordict, also in a member of a lazy stack
            _check_member_as_lists(self, is_leaf)
        if sort:

            def keyfunc(item):
                return item[0] if isinstance(item[0], str) else ".".join(item[0])

            yield from sorted(
                self.items(
                    include_nested=include_nested,
                    leaves_only=leaves_only,
                    is_leaf=is_leaf,
                ),
                key=keyfunc,
            )
        else:
            if is_leaf is None:
                is_leaf = _default_is_leaf

            if include_nested:
                # check the conditions once only
                for k in self.keys():
                    val = self._get_str(k, NO_DEFAULT)
                    cls = type(val)
                    if not leaves_only or is_leaf(cls):
                        yield k, val
                    if _is_tensor_collection(cls):
                        # Don't recurse into pass-through values (e.g., UnbatchedTensor)
                        if not _pass_through(val):
                            # _unravel_key_to_tuple((k, _key)), with less work:
                            # no call for a str _key, and a tuple _key is spliced
                            # in, so that a flat key takes the fast path instead
                            # of being rebuilt part by part on every level.
                            k_is_str = isinstance(k, str)
                            yield from (
                                (
                                    (
                                        (k, _key)
                                        if k_is_str and isinstance(_key, str)
                                        else _unravel_key_to_tuple(
                                            (k, *_key)
                                            if isinstance(_key, tuple) and _key
                                            else (k, _key)
                                        )
                                    ),
                                    _val,
                                )
                                for _key, _val in val.items(
                                    include_nested=include_nested,
                                    leaves_only=leaves_only,
                                    is_leaf=is_leaf,
                                )
                            )
            elif leaves_only:
                for k in self.keys():
                    val = self._get_str(k, NO_DEFAULT)
                    if is_leaf(type(val)):
                        yield k, val
            else:
                for k in self.keys():
                    yield k, self._get_str(k, NO_DEFAULT)

    def _members_as_lists(self) -> list[TensorDictBase] | None:
        """The tensordicts that the ``_NESTED_TENSORS_AS_LISTS`` traversals walk one by one.

        :meth:`items` and :meth:`values` call it when ``is_leaf`` is
        ``_NESTED_TENSORS_AS_LISTS`` or ``_NESTED_TENSORS_AS_LISTS_NONTENSOR``.
        ``None`` (the default) takes the generic traversal, with ``is_leaf`` as
        the leaf predicate. A lazy stack returns its members: :meth:`values`
        then lists the values of each member in turn, and :meth:`items`
        prefixes each key with the index of its member:
        ``("0", "a"), ("1", "a")``.
        """
        return None

    def non_tensor_items(self, include_nested: bool = False):
        """Returns a tuple of the ``(key, value)`` pairs of the non-tensor data of the tensordict.

        The values are :class:`~tensordict.NonTensorData` (or
        :class:`~tensordict.NonTensorStack`) instances. See
        :ref:`non-tensor-data`.

        Args:
            include_nested (bool, optional): if ``True``, the non-tensor data of
                the nested tensordicts is also returned, under nested keys.
                Defaults to ``False``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(2), s="text", nested=TensorDict(t="more"))
            >>> [key for key, _ in td.non_tensor_items(include_nested=True)]
            ['s', ('nested', 't')]
            >>> td.non_tensor_items()[0][1].data
            'text'
        """
        return tuple(
            self.items(
                include_nested,
                leaves_only=True,
                is_leaf=_is_non_tensor,
            )
        )

    def values(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf=None,
        *,
        sort: bool = False,
    ) -> Iterator[CompatibleType]:  # noqa: D417
        """Returns a generator of the values of the tensordict.

        By default, only the top-level values are returned, and a nested
        tensordict is returned as one value. Non-tensor data comes as a
        :class:`~tensordict.NonTensorData`, not as the object that
        ``td[key]`` returns. See :ref:`keys` for nested keys and leaves.

        Args:
            include_nested (bool, optional): if ``True``, the entries of the
                nested tensordicts are also returned, under nested keys (tuples
                of strings). Defaults to ``False``.
            leaves_only (bool, optional): if ``True``, only the leaves are
                returned, and nested tensordicts are skipped. Defaults to
                ``False``.
            is_leaf (Callable, optional): a function that takes a class and
                returns ``True`` if values of that class count as leaves. It
                filters the result only when ``leaves_only=True``. It does not
                stop the recursion: with ``include_nested=True``, a nested
                tensordict is still searched even if ``is_leaf`` returns
                ``True`` for it. Defaults to ``None``, which uses
                :func:`~tensordict.default_is_leaf`: tensordicts,
                tensorclasses and non-tensor data are not leaves. Pass
                :func:`~tensordict.is_leaf_nontensor` to count non-tensor data
                as leaves.

        Keyword Args:
            sort (bool, optional): if ``True``, the keys are sorted. Nested keys
                are sorted by their joined name: ``("a", "key")`` sorts as
                ``"a.key"``. Sorting takes extra time on large tensordicts.
                Defaults to ``False``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(
            ...     a=torch.zeros(2),
            ...     nested=TensorDict(b=torch.ones(2, 3), batch_size=[2]),
            ...     batch_size=[2],
            ... )
            >>> [type(value).__name__ for value in td.values()]
            ['Tensor', 'TensorDict']
            >>> [value.shape for value in td.values(include_nested=True, leaves_only=True)]
            [torch.Size([2]), torch.Size([2, 3])]

        """
        if (
            is_leaf is _NESTED_TENSORS_AS_LISTS
            or is_leaf is _NESTED_TENSORS_AS_LISTS_NONTENSOR
        ):
            members = self._members_as_lists()
            if members is not None:
                for td in members:
                    _check_member_as_lists(td, is_leaf)
                    yield from td.values(
                        include_nested=include_nested,
                        leaves_only=leaves_only,
                        is_leaf=is_leaf,
                        sort=sort,
                    )
                return
            # a file nested in a tensordict, also in a member of a lazy stack
            _check_member_as_lists(self, is_leaf)
        if sort:
            for _, value in self.items(include_nested, leaves_only, is_leaf, sort=sort):
                yield value
        else:
            if is_leaf is None:
                is_leaf = _default_is_leaf

            # check the conditions once only
            if include_nested:
                for k in self.keys():
                    val = self._get_str(k, NO_DEFAULT)
                    cls = type(val)
                    if not leaves_only or is_leaf(cls):
                        yield val
                    if include_nested and _is_tensor_collection(cls):
                        # Don't recurse into pass-through values (e.g., UnbatchedTensor)
                        if not _pass_through(val):
                            yield from val.values(
                                include_nested=include_nested,
                                leaves_only=leaves_only,
                                is_leaf=is_leaf,
                            )
            elif leaves_only:
                for k in self.keys(sort=sort):
                    val = self._get_str(k, NO_DEFAULT)
                    if is_leaf(type(val)):
                        yield val
            else:
                for k in self.keys(sort=sort):
                    yield self._get_str(k, NO_DEFAULT)

    def _inplace_on_copies(self, method: Callable, *args, **kwargs) -> Self:
        """Runs the in-place ``method`` on a copy of the tensor leaves, and writes them back.

        Called by the in-place methods when ``_reads_return_copies`` is set.
        """
        # the public name of a hook such as _masked_fill_
        name = method.__name__.lstrip("_")
        if self._storage_is_read_only():
            # the method runs on self, which writes nothing
            running = getattr(_READ_ONLY_CALL, "running", False)
            _READ_ONLY_CALL.running = True
            try:
                result = method(self, *args, **kwargs)
            finally:
                _READ_ONLY_CALL.running = running
            _warn_read_only(name)
            return result
        try:
            leaves = self._copy_of_leaves()
        except RuntimeError:
            # The members of a lazy stack that differ in shape cannot be read
            # as one tensor: the method runs on self, which writes nothing
            # unless it broadcasts self over its members.
            return method(self, *args, **kwargs)
        result = method(leaves, *args, **kwargs)
        if result is not leaves:
            # broadcasting self returns a new tensordict and writes nothing
            return result
        keys, values = leaves._items_list(True, True)
        self._write_back(keys, values, name)
        return self

    def _inplace_on_nested_copies(self, method: Callable, *args, **kwargs) -> Self:
        """Runs the in-place ``method`` where a nested tensordict reads copies, and writes them back.

        Called by the in-place methods when a tensordict that ``self`` holds
        raises ``_NestedReadsCopies``: a lazy stack whose members read
        copies, or a file. ``TensorDict`` and ``LazyStackedTensorDict``
        override it. Here the method runs on ``self``, which leaves the
        copies unchanged.
        """
        return method(self, *args, **kwargs)

    def _copy_of_leaves(self) -> TensorDictBase:
        """A ``TensorDict`` of the tensor leaves that ``items`` reads, for an in-place method to change.

        Only called when ``_reads_return_copies`` is set, by
        ``_inplace_on_copies``.
        """
        from tensordict._td import TensorDict

        items = list(self.items(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS))
        # The leaves are set in the order of self, as in to_tensordict():
        # lerp_, addcdiv_ and addcmul_ pair their operands by position.
        leaves = TensorDict(batch_size=self.batch_size, device=self.device)
        for key, value in items:
            if not isinstance(value, Tensor):
                # a non-tensor leaf, such as the str that a store reads: the
                # in-place methods leave it unchanged
                continue
            leaves._set_tuple(
                _unravel_key_to_tuple(key),
                value,
                inplace=False,
                validated=True,
                non_blocking=False,
            )
        return leaves

    def _write_back(self, keys, values, method: str) -> None:
        """Writes the leaves that the in-place method ``method`` changed back to storage.

        Only called when ``_reads_return_copies`` is set. ``values`` are the
        changed copies of the leaves under ``keys``.
        """
        if not keys:
            return
        if self._storage_is_read_only():
            _warn_read_only(method)
            return
        for key, value in zip(keys, values):
            self._set_tuple(
                _unravel_key_to_tuple(key),
                value,
                inplace=True,
                validated=True,
                non_blocking=False,
            )

    def _storage_is_read_only(self) -> bool:
        """Whether the storage of a tensordict that reads copies cannot be written."""
        return False

    @_cache_while_locked  # noqa: B019
    def _values_list(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        *,
        collapse: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
        sorting_keys: List[NestedKey] | None = None,
    ) -> List:
        if sorting_keys is None:
            return list(
                self.values(
                    include_nested=include_nested,
                    leaves_only=leaves_only,
                    is_leaf=_NESTED_TENSORS_AS_LISTS if not collapse else is_leaf,
                )
            )
        else:
            keys, vals = self._items_list(
                include_nested=include_nested,
                leaves_only=leaves_only,
                is_leaf=is_leaf,
                collapse=collapse,
            )
            if is_compiling():
                key_to_index = {key: i for i, key in enumerate(keys)}
                return [vals[key_to_index[key]] for key in sorting_keys]
            else:
                source = dict(zip(keys, vals))
                return [source[key] for key in sorting_keys]

    @_cache_while_locked  # noqa: B019
    def _items_list(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        *,
        collapse: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
        sorting_keys: List[NestedKey] | None = None,
        default: str | CompatibleType | None = None,
    ) -> Tuple[List, List]:
        items = self.items(
            include_nested=include_nested,
            leaves_only=leaves_only,
            is_leaf=_NESTED_TENSORS_AS_LISTS if not collapse else None,
        )
        keys_vals = tuple(zip(*items))
        if not keys_vals:
            return (), ()
        keys, vals = keys_vals
        if sorting_keys is None:
            return list(keys), list(vals)
        if default is None:
            # TODO: check that lists are identical
            if is_compiling():
                key_to_index = {key: i for i, key in enumerate(keys)}
                new_vals = [vals[key_to_index[key]] for key in sorting_keys]
                if len(new_vals) < len(vals):
                    raise KeyError(
                        f"Some keys were not found: {set(sorting_keys).symmetric_difference(keys)}."
                    )
            else:
                source = dict(zip(keys, vals))
                new_vals = [source[key] for key in sorting_keys]
            if len(new_vals) < len(vals):
                raise KeyError(
                    f"Some keys were not found: {set(sorting_keys).symmetric_difference(keys)}."
                )
            return sorting_keys, new_vals
        source = dict(zip(keys, vals))
        if isinstance(default, str) and default == "intersection":
            new_keys = [key for key in sorting_keys if key in source]
        else:
            new_keys = list(set(sorting_keys).union(keys))
        vals = [source.get(key, default) for key in new_keys]
        return new_keys, vals

    def _grad(self):
        # We can't cache this because zero_grad can be called outside (eg from optimizer) and we want the tensors
        # to clear out when that is done.
        keys, vals = self._items_list(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS)
        grads = [val.grad for val in vals]
        items = dict(zip(keys, grads))

        def get(name, val):
            return items[name]

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            propagate_lock=True,
            filter_empty=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
        )

    def _data(self):
        return self._fast_apply(
            _leaf_data, propagate_lock=True, is_leaf=_NESTED_TENSORS_AS_LISTS
        )

    def _data_setter(self, value: Self):
        """Sets the ``data`` of each tensor leaf of ``self`` to the leaf of ``value`` under the same key: the storage part of the ``data`` setter.

        The setter passes ``value`` unchanged. The hook raises
        ``RuntimeError`` if the batch sizes differ, and ``KeyError``, before
        it sets anything, if ``value`` has a key that ``self`` lacks. Returns
        ``None``.
        """
        if value.batch_size != self.batch_size:
            raise RuntimeError(
                f"Cannot assign .data from a source with batch size {value.batch_size} "
                f"to a destination with batch size {self.batch_size}."
            )
        if (
            getattr(type(value), "_wrapped_attr", None)
            and self._members_as_lists() is None
            and _has_lazy_stack(value)
        ):
            # a TypedTensorDict over a lazy stack lists the leaves of each
            # member (#2772): read them stacked, as self lists them
            value = value.to_tensordict()
        nodes_self = dict(
            zip(*self._items_list(True, False, is_leaf=_NESTED_TENSORS_AS_LISTS))
        )
        for key, node in zip(
            *value._items_list(True, False, is_leaf=_NESTED_TENSORS_AS_LISTS)
        ):
            node_self = nodes_self.get(key)
            if (
                _is_tensor_collection(type(node))
                and _is_tensor_collection(type(node_self))
                and node.batch_size != node_self.batch_size
            ):
                raise RuntimeError(
                    f"Cannot assign .data from a source with batch size {node.batch_size} "
                    f"to a destination with batch size {node_self.batch_size} at key {key!r}."
                )
        keys, vals = value._items_list(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS)
        keys_self, vals_self = self._items_list(
            True, True, is_leaf=_NESTED_TENSORS_AS_LISTS
        )
        self_dict = dict(zip(keys_self, vals_self))
        missing = [key for key in keys if key not in self_dict]
        if missing:
            raise KeyError(
                f"Cannot assign .data: the keys {missing} of the source are not in "
                "the destination."
            )
        for key, val in zip(keys, vals):
            val_self = self_dict[key]
            if hasattr(val_self, "data"):
                val_self.data = val

    @abc.abstractmethod
    def keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
        *,
        sort: bool = False,
    ):
        """Returns a view of the keys of the tensordict.

        By default, only the top-level keys are returned. A nested key is a
        tuple of strings, for example ``("nested", "b")``. See :ref:`keys`.

        The view is lazy: if you modify the tensordict after you call
        ``keys()`` and before you iterate over the result, the iteration
        returns the keys after the change.

        .. note::
            On a :class:`~tensordict.TensorDict`, and on most other types,
            ``key in td.keys(include_nested=True, leaves_only=True)`` checks
            that ``key`` exists, not that it is a leaf: it is ``True`` for a
            nested tensordict, although iterating the view does not list it. To
            test that a key is a leaf, call ``is_leaf(td.entry_class(key))``,
            with :func:`~tensordict.default_is_leaf` as the default
            ``is_leaf``. A nested key is never in a view built with
            ``include_nested=False``: ``("nested", "b") in td.keys()`` is
            ``False``. On a :class:`~tensordict.LazyStackedTensorDict`, a
            :class:`~tensordict.store.TensorDictStore` and a
            :class:`~tensordict.store.LazyStackedTensorDictStore`, it is still
            ``True``, with a ``FutureWarning``, until TensorDict 0.17.

        Args:
            include_nested (bool, optional): if ``True``, the entries of the
                nested tensordicts are also returned, under nested keys (tuples
                of strings). Defaults to ``False``.
            leaves_only (bool, optional): if ``True``, only the leaves are
                returned, and nested tensordicts are skipped. Defaults to
                ``False``.
            is_leaf (Callable, optional): a function that takes a class and
                returns ``True`` if values of that class count as leaves. It
                filters the result only when ``leaves_only=True``. It does not
                stop the recursion: with ``include_nested=True``, a nested
                tensordict is still searched even if ``is_leaf`` returns
                ``True`` for it. Defaults to ``None``, which uses
                :func:`~tensordict.default_is_leaf`: tensordicts,
                tensorclasses and non-tensor data are not leaves. Pass
                :func:`~tensordict.is_leaf_nontensor` to count non-tensor data
                as leaves.

        Keyword Args:
            sort (bool, optional): if ``True``, the keys are sorted. Nested keys
                are sorted by their joined name: ``("a", "key")`` sorts as
                ``"a.key"``. Sorting takes extra time on large tensordicts.
                Defaults to ``False``.

        Examples:
            >>> from tensordict import TensorDict
            >>> data = TensorDict({"0": 0, "1": {"2": 2}}, batch_size=[])
            >>> list(data.keys())
            ['0', '1']
            >>> list(data.keys(leaves_only=True))
            ['0']
            >>> list(data.keys(include_nested=True, leaves_only=True))
            ['0', ('1', '2')]
        """
        raise NotImplementedError

    def pop(self, key: NestedKey, default: Any = NO_DEFAULT) -> CompatibleType:
        """Removes an entry from the tensordict and returns its value.

        A nested key removes one entry of a nested tensordict and keeps the
        nested tensordict. A locked tensordict raises ``RuntimeError`` when
        the key is present (see :meth:`~.lock_`).

        Args:
            key (NestedKey): the key of the entry to remove.
            default (Any, optional): the value returned when the key is
                missing, including a nested key that runs through a tensor.
                Without it, a missing key raises ``KeyError`` and a nested key
                that runs through a tensor raises ``ValueError``.

        Returns:
            The value of the removed entry, or ``default``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3), b=TensorDict(c=torch.ones(3)))
            >>> td.pop(("b", "c"))
            tensor([1., 1., 1.])
            >>> sorted(td.keys(include_nested=True))
            ['a', 'b']
            >>> td.pop("missing", None) is None
            True
        """
        key_tuple = _unravel_key_to_tuple(key)
        if not key_tuple:
            raise KeyError(_GENERIC_NESTED_ERR.format(key))
        # Use _UNSET sentinel to detect if key exists without try/except (compile-friendly)
        out = self.get(key_tuple, _UNSET)
        if out is _UNSET:
            # Key not found
            if default is NO_DEFAULT:
                if len(key_tuple) > 1:
                    # a key that runs through a leaf raises get's ValueError
                    try:
                        self.get(key_tuple, NO_DEFAULT)
                    except KeyError:
                        pass
                raise KeyError(
                    _KEY_ERROR.format(
                        key_tuple[0] if len(key_tuple) == 1 else key_tuple,
                        type(self).__name__,
                        sorted(self.keys(include_nested=len(key_tuple) > 1), key=str),
                    )
                )
            return default
        out = self._popped_value(out)
        self.del_(key_tuple)
        return out

    def _popped_value(self, value: Any) -> Any:
        """Returns what pop returns for ``value``, an entry that del_ will delete.

        A class whose sub-tensordicts are views over storage that del_ deletes
        overrides this to return a copy.
        """
        return value

    @property
    def sorted_keys(self) -> list[NestedKey]:
        """The top-level keys of the tensordict, as a list sorted in alphabetical order.

        For nested keys, call ``keys(include_nested=True, sort=True)``. On a
        locked tensordict, the sorted keys are cached until the tensordict is
        unlocked.

        Examples:
            >>> from tensordict import TensorDict
            >>> td = TensorDict(b=0, a=1, c=TensorDict(d=2))
            >>> td.sorted_keys
            ['a', 'b', 'c']
        """
        return list(self._sorted_keys())

    @_cache_while_locked  # noqa: B019
    def _sorted_keys(self) -> tuple[NestedKey, ...]:
        # The lock cache stores a tuple so that sorted_keys can return a new
        # list: a caller that changes that list cannot change the cache.
        return tuple(sorted(self.keys()))

    def _transform_keys(self, key_transform: Callable[[NestedKey], NestedKey]) -> Self:
        """Returns a new tensordict with every key transformed by ``key_transform``.

        ``key_transform`` is called on the string key of each entry, level by
        level: ``("a", "b")`` becomes ``(key_transform("a"), key_transform("b"))``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({("a", "b"): torch.zeros(3)}, [3])
            >>> list(td._transform_keys(lambda key: f"avg_{key}").keys(True, True))
            [('avg_a', 'avg_b')]
        """
        new_td = self.empty()
        for key, value in self.items():
            new_key = key_transform(key)
            # If the value is a TensorDict, recursively transform its keys
            if hasattr(value, "_transform_keys"):
                value = value._transform_keys(key_transform)
            new_td[new_key] = value
        return new_td

    def rename_key_(
        self, old_key: NestedKey, new_key: NestedKey, safe: bool = False
    ) -> Self:
        """Renames an entry of the tensordict in place and returns the tensordict.

        The value is not copied. A locked tensordict raises ``RuntimeError``
        (see :meth:`~.lock_`). Until TensorDict 0.17, a locked
        :class:`~tensordict.PersistentTensorDict` or store renames the key and
        warns instead.

        Args:
            old_key (NestedKey): the key to rename. A missing key raises
                ``KeyError``.
            new_key (NestedKey): the new key of the entry. An empty key raises
                ``KeyError``. A key inside ``old_key``, such as ``("a", "b")``
                for ``"a"``, moves the entry there, as
                ``td.set(new_key, td.pop(old_key))`` does.
            safe (bool, optional): if ``True``, a ``KeyError`` is raised when
                ``new_key`` is already present in the tensordict, unless it
                lies inside ``old_key``, which moves with the entry. If
                ``False``, the existing entry is overwritten. Defaults to
                ``False``.

        Returns:
            ``self``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.zeros(3), "b": {"c": torch.ones(3)}}, [3])
            >>> td = td.rename_key_("a", "x")
            >>> td = td.rename_key_(("b", "c"), ("b", "y"))
            >>> list(td.keys(include_nested=True, leaves_only=True))
            [('b', 'y'), 'x']

        """
        lenient = self._rename_key_lenient
        if self.is_locked:
            if not lenient:
                raise RuntimeError(_LOCK_ERROR)
            warnings.warn(
                f"{type(self).__name__}.rename_key_ renames keys of a locked "
                "tensordict today. Starting with TensorDict 0.17, the default "
                "will change to raising a RuntimeError, as TensorDict does.",
                FutureWarning,
                stacklevel=2,
            )
        # these checks are not perfect, tuples that are not tuples of strings or empty
        # tuples could go through but (1) it will raise an error anyway and (2)
        # those checks are expensive when repeated often.
        if not isinstance(old_key, (str, tuple)):
            if lenient:
                self._rename_key(old_key, new_key)
                return self
            raise TypeError(
                f"Expected old_name to be a string or a tuple of strings but found {type(old_key)}"
            )
        if not isinstance(new_key, (str, tuple)):
            raise TypeError(
                f"Expected new_name to be a string or a tuple of strings but found {type(new_key)}"
            )
        old_key = unravel_key(old_key)
        new_key = unravel_key(new_key)
        if new_key == ():
            raise KeyError(_GENERIC_NESTED_ERR.format(new_key))
        if (
            safe
            and old_key != new_key
            and new_key in self.keys(include_nested=True)
            # a new key inside the old one moves with the entry
            and not _is_inside(new_key, old_key)
        ):
            raise KeyError(f"key {new_key} already present in TensorDict.")
        self._rename_key(old_key, new_key)
        return self

    def _rename_key(self, old_key: NestedKey, new_key: NestedKey) -> None:
        """Moves the entry at ``old_key`` to ``new_key``: the storage part of :meth:`rename_key_`.

        :meth:`rename_key_` has checked the lock, the key types and ``safe``,
        and unravelled both keys (see ``_rename_key_lenient`` for the
        exceptions). ``old_key == new_key`` only checks that the key exists.
        """
        if old_key == new_key:
            if old_key not in self.keys(include_nested=isinstance(old_key, tuple)):
                raise KeyError(f"Key {old_key} not found in tensordict.")
            return
        value = self.get(old_key, default=NO_DEFAULT)
        # only a tuple can lie inside another key; skipping the check for a
        # str keeps the common renames as fast as before
        inside = isinstance(new_key, tuple) and _is_inside(new_key, old_key)
        if inside:
            # deleting the old entry after the move would delete the moved
            # entry with it
            self.del_(old_key)
        if isinstance(new_key, str):
            self._set_str(
                new_key, value, inplace=False, validated=True, non_blocking=False
            )
        else:
            self._set_tuple(
                new_key, value, inplace=False, validated=True, non_blocking=False
            )
        new_key_tuple = (new_key,) if isinstance(new_key, str) else new_key
        if not inside and not (
            isinstance(old_key, tuple)
            and old_key[: len(new_key_tuple)] == new_key_tuple
        ):
            # eg: td.rename_(("0", "1", "2"), ("0", "1")), then ("0", "1", "2") should not be deleted
            self.del_(old_key)

    def rename_keys(
        self, mapping: Mapping[NestedKey, NestedKey], *, strict: bool = True
    ) -> Self:
        """Returns a tensordict with several keys renamed at once.

        The tensordict is not modified. The result has a new tree of nested
        tensordicts and shares every leaf with the tensordict, tensors and
        non-tensor data alike. Its batch size, device and dimension names are
        those of the tensordict, and it is not locked.

        Every source is read before any destination is written, so swaps
        (``{"a": "b", "b": "a"}``) and chains (``{"a": "b", "b": "c"}``) work.
        A nested tensordict that the renaming leaves empty is removed. Other
        empty nested tensordicts are kept.

        A lazy stack renames the keys in each of its tensordicts, and removes a
        nested tensordict only if the renaming empties it in all of them. A
        tensorclass returns the same class, its destinations must be fields,
        and its fields are never removed.
        :class:`~tensordict.PersistentTensorDict` and the stores raise
        ``NotImplementedError``: call :meth:`~.to_tensordict` first.

        To rename a single key in place, see :meth:`~.rename_key_`.

        Args:
            mapping (Mapping[NestedKey, NestedKey]): the old key of each entry
                mapped to its new key. Both sides can be nested keys, so an
                entry can move to another nested tensordict and a whole nested
                tensordict can be renamed.

        Keyword Args:
            strict (bool, optional): if ``True``, a source missing from the
                tensordict raises ``KeyError``. If ``False``, it is skipped.
                Defaults to ``True``.

        Returns:
            A new tensordict of the same type as ``self``, except that a
            sub-tensordict view and a :class:`~tensordict.nn.TensorDictParams`
            return a :class:`~tensordict.TensorDict`, as :meth:`exclude` does.

        Raises:
            KeyError: if ``strict=True`` and a source is missing, or if a key is
                empty.
            ValueError: if two sources have the same destination, if a source
                lies inside another source or a destination inside another
                destination, or if a destination is already present and is not
                renamed away.
            AttributeError: if ``self`` is a tensorclass and a destination is
                not one of its fields.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(()), b=torch.ones(()))
            >>> out = td.rename_keys({"a": "b", "b": "a"})
            >>> out["a"], td["a"]
            (tensor(1.), tensor(0.))
            >>> td = TensorDict(raw=TensorDict(p=torch.zeros(3), u=torch.ones(3)))
            >>> out = td.rename_keys({("raw", "p"): "pressure", ("raw", "u"): "velocity"})
            >>> sorted(out.keys(include_nested=True))  # "raw" was emptied and removed
            ['pressure', 'velocity']
            >>> out = td.rename_keys({"raw": "solution"})
            >>> sorted(out.keys(include_nested=True, leaves_only=True))
            [('solution', 'p'), ('solution', 'u')]
            >>> td.rename_keys({("raw", "zz"): "z", ("raw", "p"): "p"}, strict=False)["p"]
            tensor([0., 0., 0.])
        """
        return self._rename_keys(
            _rename_keys_pairs(mapping), strict=strict, prune_root=True
        )

    def _rename_keys(
        self,
        pairs: list[tuple[tuple[str, ...], tuple[str, ...]]],
        *,
        strict: bool,
        prune_root: bool,
    ) -> Self:
        """Returns a copy of ``self`` with the keys renamed: the storage part of :meth:`rename_keys`.

        :meth:`rename_keys` has turned the mapping into ``pairs`` of tuple keys
        and checked that they do not overlap. The hook copies the tree of
        sub-tensordicts, shares every leaf, and moves the entries of the copy
        with ``_rename_entries``. ``prune_root=False`` keeps the top-level
        sub-tensordicts that the move empties (the fields of a tensorclass).
        """
        out = self._fast_apply(_keep, is_leaf=_is_leaf_nontensor)
        _rename_entries([out], pairs, strict=strict, prune_root=prune_root)
        return out

    @_lock_blocked
    def del_(self, key: NestedKey) -> Self:
        """Deletes an entry of the tensordict in place and returns the tensordict.

        A locked tensordict raises ``RuntimeError`` (see :meth:`~.lock_`).
        ``del td[key]`` does the same.

        Args:
            key (NestedKey): the key to delete. A missing key raises
                ``KeyError``, and a key whose path runs through an entry that
                is not a tensordict raises ``ValueError``. Until TensorDict
                0.17, the stores and the zarr backend of
                :class:`~tensordict.PersistentTensorDict` warn and do nothing
                for a missing key.

        Returns:
            ``self``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3), b=TensorDict(c=torch.ones(3)))
            >>> list(td.del_(("b", "c")).keys(True))
            ['a', 'b']

        """
        self._del(_unravel_key_to_tuple(key))
        return self

    def _del(self, key: tuple[str, ...]) -> None:
        """Deletes the entry at ``key``: the storage part of :meth:`del_`.

        :meth:`del_` has checked the lock and unravelled ``key`` to a tuple of
        str, and returns ``self``. It does not check for an empty key. An empty
        or missing key raises ``KeyError``; a key whose path runs through an
        entry that is not a tensordict raises ``ValueError``. Until 0.17, the
        backends keep their former errors: for ``()``,
        :class:`~tensordict.PersistentTensorDict` raises ``ValueError``,
        :class:`~tensordict.store.TensorDictStore` raises ``IndexError`` and
        :class:`~tensordict.store.LazyStackedTensorDictStore` warns; for a
        missing key, the stores and the zarr backend warn and return.
        """
        raise NotImplementedError(f"del_ is not implemented for {type(self).__name__}.")

    # Apply and map functionality
    def apply_(self, fn: Callable, *others, **kwargs) -> Self:
        """Applies a callable to every leaf and writes the results into the existing leaves.

        Same as :meth:`~.apply` with ``inplace=True``: each result is copied
        into the leaf it comes from (as :meth:`~.set_` does), so the leaves
        keep their storage.

        Args:
            fn (Callable): the function called on each tensor leaf. It returns
                a tensor that can be copied into the leaf.
            *others (TensorDictBase): other tensordicts with the same
                structure. ``fn`` receives the leaf of each, after the leaf of
                the tensordict, under the same key.
            **kwargs: the keyword arguments of :meth:`~.apply`.

        Returns:
            ``self``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3), b=TensorDict(c=torch.ones(3)))
            >>> a = td["a"]
            >>> _ = td.apply_(lambda x: x + 1)
            >>> a
            tensor([1., 1., 1.])
            >>> td["b", "c"]
            tensor([2., 2., 2.])
        """
        self._apply_(fn, *others, **kwargs)
        return self

    def _apply_(self, fn: Callable, *others, **kwargs) -> None:
        """Applies ``fn`` to the leaves and writes each result into its leaf.

        Called by :meth:`apply_` with its arguments, which returns ``self``
        afterwards. The base implementation calls :meth:`apply` with
        ``inplace=True``. Override it when the class can write the results
        faster than through :meth:`apply` (a lazy stack applies ``fn`` to each
        member).
        """
        self.apply(fn, *others, inplace=True, **kwargs)

    def apply(
        self,
        fn: Callable,
        *others: T,
        batch_size: Sequence[int] | None = None,
        device: torch.device | None = NO_DEFAULT,
        names: Sequence[str] | None = NO_DEFAULT,
        inplace: bool = False,
        default: Any = NO_DEFAULT,
        filter_empty: bool | None = None,
        propagate_lock: bool = False,
        call_on_nested: bool = False,
        is_leaf: Callable[[type], bool] | None = None,
        out: TensorDictBase | None = None,
        **constructor_kwargs,
    ) -> Self | None:
        """Applies a callable to every leaf and returns a new tensordict with the results.

        ``fn`` is called on each tensor leaf, in the nested tensordicts too, and
        its result is stored under the same key. If ``fn`` returns ``None``, the
        entry is left out of the result, which can be used to filter entries.
        With ``others``, ``fn`` receives the leaf of each tensordict under the
        same key. For a callable that also receives the key, see
        :meth:`~.named_apply`.

        Args:
            fn (Callable): the function called on each leaf, as
                ``fn(leaf, *other_leaves)``. It returns a tensor, a tensordict,
                or ``None`` to leave the entry out.
            *others (TensorDictBase): other tensordicts with the same structure
                as the tensordict. An entry that one of them lacks raises
                ``KeyError``, unless ``default`` is given.

        Keyword Args:
            batch_size (torch.Size or list of int, optional): the batch size of
                the result, for a ``fn`` that changes the shape of the leaves:
                the results must have it as their leading dimensions. ``fn`` is
                then not called on :class:`~tensordict.UnbatchedTensor` entries,
                which keep their data and take the new batch size. Defaults to
                ``None``, which keeps the batch size of the tensordict.
            device (torch.device, str or int, optional): the device of the
                result. The results of ``fn`` are moved to it. ``None`` gives a
                result with no device, which leaves each result where ``fn``
                put it. By default, the result has the device of the
                tensordict.
            names (list of str, optional): the names of the batch dimensions of
                the result, one per dimension. By default, the names of the
                tensordict are kept, or dropped if ``batch_size`` is given.
            inplace (bool, optional): if ``True``, the results are written into
                the existing leaves of the tensordict (as :meth:`~.set_` does),
                and ``self`` is returned. See :ref:`inplace-conventions`.
                Defaults to ``False``.
            default (Any, optional): the value passed to ``fn`` for an entry
                that one of ``others`` lacks. By default, a missing entry raises
                ``KeyError``.
            filter_empty (bool, optional): if ``True``, a nested tensordict
                with no entry in the result (because ``fn`` returned ``None``
                for all its leaves, or because it was already empty) is left
                out. If ``False``, it is kept, empty. Defaults to ``None``,
                which leaves out the nested tensordicts that ``fn`` emptied and
                keeps those that were already empty. Non-tensor data that
                ``fn`` is not called on is kept in all cases.
            propagate_lock (bool, optional): if ``True``, the result of a locked
                tensordict is locked too. Defaults to ``False``.
            call_on_nested (bool, optional): if ``True``, ``fn`` is called on the
                top-level entries, nested tensordicts and tensorclasses
                included, and is responsible for going into them. If ``False``,
                ``fn`` is called on the leaves only. Defaults to ``False``.
            is_leaf (Callable, optional): called with the type of each entry:
                ``fn`` is called on the entries for which it returns ``True``,
                and ``apply`` goes into the others. Defaults to ``None``, which
                uses :func:`~tensordict.default_is_leaf`: ``fn`` is called on
                tensors, and non-tensor data is kept unchanged. Pass
                :func:`~tensordict.is_leaf_nontensor` to call ``fn`` on
                non-tensor data too.
            out (TensorDictBase, optional): a tensordict to write the results
                into, instead of a new one. When ``batch_size`` or ``device`` is
                given, ``out`` must have it. For a lazy stack, ``out`` must be a
                lazy stack too, stacked along any dimension. See the warning
                below about ``out=self``. Defaults to ``None``.
            **constructor_kwargs: ignored, with a :class:`FutureWarning`.
                Starting with TensorDict 0.17, the default will change to
                raising a ``TypeError`` for them.

        Returns:
            A new tensordict, ``out`` if it is given, or ``self`` if
            ``inplace=True``. ``None`` if ``fn`` returns ``None`` for every
            leaf and ``filter_empty`` is not ``False``. With
            ``filter_empty=None``, a tensordict that is already empty gives an
            empty tensordict.

            The result has the type of the tensordict for a
            :class:`~tensordict.TensorDict`, a lazy stack and a tensorclass.
            Other types, for example :class:`~tensordict.PersistentTensorDict`
            or a memory-mapped tensordict, give a :class:`~tensordict.TensorDict`
            that lives in memory. A locked tensordict gives an unlocked result
            unless ``propagate_lock=True``.

        .. warning:: With ``out=self``, the results are silently wrong if
            ``fn`` reads other entries of the tensordict, which may already be
            overwritten.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(
            ...     a=-torch.ones(3),
            ...     b=TensorDict(c=torch.ones(3), batch_size=[3]),
            ...     batch_size=[3],
            ... )
            >>> td.apply(lambda x: x + 1)["b", "c"]
            tensor([2., 2., 2.])

            With another tensordict, ``fn`` receives one leaf of each:

            >>> td.apply(lambda x, y: x + y, td)["a"]
            tensor([-2., -2., -2.])

            Returning ``None`` leaves the entry out:

            >>> out = td.apply(lambda x: x if x.sum() > 0 else None)
            >>> list(out.keys(include_nested=True, leaves_only=True))
            [('b', 'c')]

            A ``fn`` that changes the shape of the leaves needs a ``batch_size``:

            >>> td.apply(lambda x: x.unsqueeze(-1), batch_size=[3, 1]).batch_size
            torch.Size([3, 1])

            By default, ``fn`` is not called on non-tensor data:

            >>> from tensordict import is_leaf_nontensor
            >>> td = TensorDict(a=torch.zeros(3), b="a string", batch_size=[3])
            >>> seen = []
            >>> _ = td.apply(lambda x: seen.append(type(x).__name__))
            >>> seen
            ['Tensor']
            >>> seen = []
            >>> _ = td.apply(lambda x: seen.append(type(x).__name__), is_leaf=is_leaf_nontensor)
            >>> seen
            ['Tensor', 'NonTensorData']
        """
        if constructor_kwargs:
            ignored = [k for k in constructor_kwargs if k not in _APPLY_NEST_KWARGS]
            if ignored:
                _warn_ignored_kwargs("apply", ignored)
        result = self._apply_nest(
            fn,
            *others,
            batch_size=batch_size,
            device=device,
            names=names,
            inplace=inplace,
            checked=False,
            default=default,
            filter_empty=filter_empty,
            call_on_nested=call_on_nested,
            is_leaf=is_leaf,
            out=out,
            **constructor_kwargs,
        )
        if propagate_lock and not inplace and self.is_locked and result is not None:
            result.lock_()
        return result

    def named_apply(
        self,
        fn: Callable,
        *others: T,
        nested_keys: bool = False,
        batch_size: Sequence[int] | None = None,
        device: torch.device | None = NO_DEFAULT,
        names: Sequence[str] | None = NO_DEFAULT,
        inplace: bool = False,
        default: Any = NO_DEFAULT,
        filter_empty: bool | None = None,
        propagate_lock: bool = False,
        call_on_nested: bool = False,
        is_leaf: Callable[[type], bool] | None = None,
        out: TensorDictBase | None = None,
        **constructor_kwargs,
    ) -> Self | None:
        """Applies a callable to every leaf and its key, and returns a new tensordict with the results.

        Same as :meth:`~.apply`, except that ``fn`` also receives the key of
        each leaf: the last string of its nested key, or the whole nested key
        with ``nested_keys=True``. If ``fn`` returns ``None``, the entry is left
        out of the result, which can be used to filter entries by key.

        Args:
            fn (Callable): the function called on each leaf, as
                ``fn(key, leaf, *other_leaves)``. It returns a tensor, a
                tensordict, or ``None`` to leave the entry out.
            *others (TensorDictBase): other tensordicts with the same structure
                as the tensordict. An entry that one of them lacks raises
                ``KeyError``, unless ``default`` is given.

        Keyword Args:
            nested_keys (bool, optional): if ``True``, ``fn`` receives the whole
                nested key of each leaf, a tuple of strings for an entry of a
                nested tensordict. If ``False``, it receives the last string of
                the key. Defaults to ``False``.
            batch_size (torch.Size or list of int, optional): the batch size of
                the result, for a ``fn`` that changes the shape of the leaves:
                the results must have it as their leading dimensions. ``fn`` is
                then not called on :class:`~tensordict.UnbatchedTensor` entries,
                which keep their data and take the new batch size. Defaults to
                ``None``, which keeps the batch size of the tensordict.
            device (torch.device, str or int, optional): the device of the
                result. The results of ``fn`` are moved to it. ``None`` gives a
                result with no device, which leaves each result where ``fn``
                put it. By default, the result has the device of the
                tensordict.
            names (list of str, optional): the names of the batch dimensions of
                the result, one per dimension. By default, the names of the
                tensordict are kept, or dropped if ``batch_size`` is given.
            inplace (bool, optional): if ``True``, the results are written into
                the existing leaves of the tensordict (as :meth:`~.set_` does),
                and ``self`` is returned. See :ref:`inplace-conventions`.
                Defaults to ``False``.
            default (Any, optional): the value passed to ``fn`` for an entry
                that one of ``others`` lacks. By default, a missing entry raises
                ``KeyError``.
            filter_empty (bool, optional): if ``True``, a nested tensordict
                with no entry in the result (because ``fn`` returned ``None``
                for all its leaves, or because it was already empty) is left
                out. If ``False``, it is kept, empty. Defaults to ``None``,
                which leaves out the nested tensordicts that ``fn`` emptied and
                keeps those that were already empty. Non-tensor data that
                ``fn`` is not called on is kept in all cases.
            propagate_lock (bool, optional): if ``True``, the result of a locked
                tensordict is locked too. Defaults to ``False``.
            call_on_nested (bool, optional): if ``True``, ``fn`` is called on the
                top-level entries, nested tensordicts and tensorclasses
                included, and is responsible for going into them. If ``False``,
                ``fn`` is called on the leaves only. Defaults to ``False``.
            is_leaf (Callable, optional): called with the type of each entry:
                ``fn`` is called on the entries for which it returns ``True``,
                and ``named_apply`` goes into the others. Defaults to ``None``, which
                uses :func:`~tensordict.default_is_leaf`: ``fn`` is called on
                tensors, and non-tensor data is kept unchanged. Pass
                :func:`~tensordict.is_leaf_nontensor` to call ``fn`` on
                non-tensor data too.
            out (TensorDictBase, optional): a tensordict to write the results
                into, instead of a new one. When ``batch_size`` or ``device`` is
                given, ``out`` must have it. For a lazy stack, ``out`` must be a
                lazy stack too, stacked along any dimension. See the warning
                below about ``out=self``. Defaults to ``None``.
            **constructor_kwargs: ignored, with a :class:`FutureWarning`.
                Starting with TensorDict 0.17, the default will change to
                raising a ``TypeError`` for them.

        Returns:
            A new tensordict, ``out`` if it is given, or ``self`` if
            ``inplace=True``. ``None`` if ``fn`` returns ``None`` for every
            leaf and ``filter_empty`` is not ``False``. With
            ``filter_empty=None``, a tensordict that is already empty gives an
            empty tensordict.

            The result has the type of the tensordict for a
            :class:`~tensordict.TensorDict`, a lazy stack and a tensorclass.
            Other types, for example :class:`~tensordict.PersistentTensorDict`
            or a memory-mapped tensordict, give a :class:`~tensordict.TensorDict`
            that lives in memory. A locked tensordict gives an unlocked result
            unless ``propagate_lock=True``.

        .. warning:: With ``out=self``, the results are silently wrong if
            ``fn`` reads other entries of the tensordict, which may already be
            overwritten.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(
            ...     a=torch.zeros(3),
            ...     nested=TensorDict(a=torch.ones(3), b=torch.ones(3), batch_size=[3]),
            ...     batch_size=[3],
            ... )
            >>> out = td.named_apply(lambda key, x: x if key == "a" else None)
            >>> list(out.keys(include_nested=True, leaves_only=True))
            ['a', ('nested', 'a')]
            >>> out = td.named_apply(
            ...     lambda key, x: x + 1 if key == ("nested", "b") else x,
            ...     nested_keys=True,
            ... )
            >>> out["nested", "b"], out["nested", "a"]
            (tensor([2., 2., 2.]), tensor([1., 1., 1.]))

            With another tensordict, ``fn`` receives one leaf of each:

            >>> td.named_apply(lambda key, x, y: x + y, td)["nested", "a"]
            tensor([2., 2., 2.])
        """
        if constructor_kwargs:
            ignored = [k for k in constructor_kwargs if k not in _APPLY_NEST_KWARGS]
            if ignored:
                _warn_ignored_kwargs("named_apply", ignored)
        result = self._apply_nest(
            fn,
            *others,
            batch_size=batch_size,
            device=device,
            names=names,
            inplace=inplace,
            checked=False,
            default=default,
            named=True,
            nested_keys=nested_keys,
            filter_empty=filter_empty,
            call_on_nested=call_on_nested,
            out=out,
            is_leaf=is_leaf,
            **constructor_kwargs,
        )
        if propagate_lock and not inplace and self.is_locked and result is not None:
            result.lock_()
        return result

    def _multithread_apply_flat(
        self,
        fn: Callable,
        *others: T,
        call_on_nested: bool = False,
        default: Any = NO_DEFAULT,
        named: bool = False,
        nested_keys: bool = False,
        prefix: tuple = (),
        is_leaf: Callable[[Type], bool] | None = None,
        executor: ThreadPoolExecutor,
        futures: List[Future],
        local_futures: List,
    ) -> None:
        if is_leaf is None:
            is_leaf = _default_is_leaf
        for key, item in self.items():
            if (
                not call_on_nested and not is_leaf(type(item))
                # and not is_non_tensor(item)
            ):
                if default is not NO_DEFAULT:
                    _others = [_other._get_str(key, default=None) for _other in others]
                    _others = [
                        self.empty(recurse=True) if _other is None else _other
                        for _other in _others
                    ]
                else:
                    _others = [
                        _other._get_str(key, default=NO_DEFAULT) for _other in others
                    ]
                local_futures.append([])
                item._multithread_apply_flat(
                    fn,
                    *_others,
                    named=named,
                    nested_keys=nested_keys,
                    prefix=prefix + (key,),
                    is_leaf=is_leaf,
                    executor=executor,
                    futures=futures,
                    local_futures=local_futures[-1],
                )
            else:
                _others = [_other._get_str(key, default=default) for _other in others]
                if named:
                    if nested_keys:
                        future = executor.submit(
                            fn, prefix + (key,) if prefix != () else key, item, *_others
                        )
                    else:
                        future = executor.submit(fn, key, item, *_others)
                else:
                    future = executor.submit(fn, item, *_others)
                futures.append(future)
                local_futures.append(future)

    def _multithread_rebuild(
        self,
        *,
        batch_size: Sequence[int] | None = None,
        device: torch.device | None = NO_DEFAULT,
        names: Sequence[str] | None = NO_DEFAULT,
        inplace: bool = False,
        checked: bool = False,
        out: TensorCollection | None = None,
        filter_empty: bool = False,
        executor: ThreadPoolExecutor,
        futures: List[Future],
        local_futures: List,
        subs_results: Dict[Future, Any] | None = None,
        multithread_set: bool = False,  # Experimental
        **constructor_kwargs,
    ) -> None:
        from tensordict._td import _SubTensorDict, TensorDict

        if constructor_kwargs:
            raise RuntimeError(
                f"constructor_kwargs not supported for class {type(self).__name__}."
            )
        # Rebuilds a tensordict from the futures of its leaves
        if inplace:
            result = self
            is_locked = result.is_locked
        elif out is not None:
            result = out
            if out.is_locked:
                raise RuntimeError(_LOCK_ERROR)
            is_locked = False
            if batch_size is not None and torch.Size(batch_size) != out.batch_size:
                raise RuntimeError(
                    "batch_size and out.batch_size must be equal when both are provided."
                )
            if device is not NO_DEFAULT and device != out.device:
                raise RuntimeError(
                    "device and out.device must be equal when both are provided."
                )
        else:

            def make_result(names=names, batch_size=batch_size):
                if names is NO_DEFAULT:
                    if batch_size is not None:
                        # erase names
                        names = None
                    elif batch_size is None:
                        names = self.names if self._has_names() else None
                return self.empty(batch_size=batch_size, device=device, names=names)

            result = make_result()
            is_locked = False

        any_set = set()

        if isinstance(result, _SubTensorDict):

            def setter(
                item_trsf,
                key,
                inplace=inplace,
                result=result,
            ):
                set_item = item_trsf is not None
                any_set.add(set_item)
                if not set_item:
                    return
                result.set(key, item_trsf, inplace=inplace)

        elif checked and isinstance(result, TensorDict) and (inplace is not True):

            def setter(
                item_trsf,
                key,
                result=result,
            ):
                set_item = item_trsf is not None
                any_set.add(set_item)
                if not set_item:
                    return
                result._tensordict[key] = item_trsf

        else:
            local_inplace = BEST_ATTEMPT_INPLACE if inplace else False

            def setter(
                item_trsf,
                key,
                result=result,
                checked=checked,
            ):
                set_item = item_trsf is not None
                any_set.add(set_item)
                if not set_item:
                    return

                result._set_str(
                    key,
                    item_trsf,
                    inplace=local_inplace,
                    validated=checked,
                    non_blocking=False,
                )

        for i, (key, local_future) in enumerate(
            _zip_strict(self.keys(), local_futures)
        ):
            if isinstance(local_future, list):
                # We can't make this a future as it could cause deadlocks:
                #  If we put a future over the root and this triggers another
                #  call on the leaves, the root will occupy a spot in the execution queue
                #  and wait for completion, potentially preventing the leaf of
                #  getting in the execution queue at all.
                td = self._get_str(key, default=None)
                item_trsf = td._multithread_rebuild(
                    batch_size=batch_size,
                    device=device,
                    names=names,
                    inplace=inplace,
                    checked=checked,
                    out=out._get_str(key, default=None) if out is not None else None,
                    filter_empty=filter_empty,
                    executor=executor,
                    futures=futures,
                    local_futures=local_future,
                    subs_results=subs_results,
                    multithread_set=multithread_set,
                    **constructor_kwargs,
                )
                if multithread_set:
                    local_future = executor.submit(setter, item_trsf=item_trsf, key=key)
                    local_futures[i] = local_future
                    futures.append(local_future)
                else:
                    setter(item_trsf=item_trsf, key=key)
            else:
                if multithread_set:
                    if subs_results is not None:
                        local_result = subs_results[local_future]
                    else:
                        # TODO: check if add_done_callback can safely be used here
                        #  The issue is that it does not raises an exception encountered during the
                        #  execution, resulting in UBs.
                        local_result = local_future.result()
                    local_future = executor.submit(
                        setter, item_trsf=local_result, key=key
                    )
                    futures.append(local_future)
                    local_futures[i] = local_future
                else:
                    local_result = local_future.result()
                    setter(item_trsf=local_result, key=key)

        if multithread_set:
            wait(local_futures)
        any_set = True in any_set or is_non_tensor(self)

        if filter_empty and not any_set:
            return
        elif not filter_empty and not inplace and is_locked:
            result.lock_()
        return result

    def _multithread_apply_nest(
        self,
        fn: Callable,
        *others: T,
        batch_size: Sequence[int] | None = None,
        device: torch.device | None = NO_DEFAULT,
        names: Sequence[str] | None = NO_DEFAULT,
        inplace: bool = False,
        checked: bool = False,
        call_on_nested: bool = False,
        default: Any = NO_DEFAULT,
        named: bool = False,
        nested_keys: bool = False,
        prefix: tuple = (),
        filter_empty: bool | None = None,
        is_leaf: Callable[[Type], bool] | None = None,
        out: TensorDictBase | None = None,
        num_threads: int,
        call_when_done: Callable | None = None,
        **constructor_kwargs,
    ) -> Self | None:
        """A deadlock-safe multithread wrapper around TD.apply.

        First launches fn for all the leaves, then rebuilds the tensordicts out of them.

        An optional ``call_when_done`` function can be passed to execute a method on the main thread
        after a future is completed.

        """
        if call_on_nested:
            warnings.warn(
                "Multithreaded apply with call_on_nested=True should not be used for deep TensorDicts. "
                "In the best cases, it will be inefficient, in the worst an arbitrary large number of "
                "threads will be spawn."
            )
        # We create 2 structures that will have the same elements within:
        #  futures is a flat list of all the futures we need to wait for,
        #  local_futures is a nested representation of this flat structure.
        #  In local_futures, the order of the items can be used to link the items to their key.
        futures = []
        local_futures = []
        executor = ThreadPoolExecutor(max_workers=num_threads)
        self._multithread_apply_flat(
            fn,
            *others,
            call_on_nested=call_on_nested,
            default=default,
            named=named,
            nested_keys=nested_keys,
            prefix=prefix,
            is_leaf=is_leaf,
            executor=executor,
            futures=futures,
            local_futures=local_futures,
        )
        if call_when_done is not None:
            subs_results = {}

            def cb(fut):
                fut._result = call_when_done(fut.result())
                return fut

            for fut in futures:
                fut.add_done_callback(cb)
            # wait(futures)
            # for fut in as_completed(futures):
            #     subs_results[fut] = call_when_done(fut.result())
            #     fut._result = None
            #     # futures.remove(fut)
            #     # del fut
        else:
            subs_results = None
        return self._multithread_rebuild(
            batch_size=batch_size,
            device=device,
            names=names,
            inplace=inplace,
            checked=checked,
            out=out,
            filter_empty=filter_empty,
            executor=executor,
            futures=futures,
            local_futures=local_futures,
            subs_results=subs_results,
            **constructor_kwargs,
        )

    def _apply_nest(
        self,
        fn: Callable,
        *others: T,
        batch_size: Sequence[int] | None = None,
        device: torch.device | None = NO_DEFAULT,
        names: Sequence[str] | None = NO_DEFAULT,
        inplace: bool = False,
        checked: bool = False,
        call_on_nested: bool = False,
        default: Any = NO_DEFAULT,
        named: bool = False,
        nested_keys: bool = False,
        prefix: tuple = (),
        filter_empty: bool | None = None,
        is_leaf: Callable[[Type], bool] | None = None,
        out: TensorDictBase | None = None,
        call_on_unbatched: bool = False,
        **constructor_kwargs,
    ) -> Self | None:
        if inplace:
            result = self
            is_locked = result.is_locked
        elif out is not None:
            result = out
            if out.is_locked:
                raise RuntimeError(_LOCK_ERROR)
            is_locked = False
            if batch_size is not None and torch.Size(batch_size) != out.batch_size:
                raise RuntimeError(
                    "batch_size and out.batch_size must be equal when both are provided."
                )
            if device is not NO_DEFAULT and device != out.device:
                if not checked:
                    raise RuntimeError(
                        f"device and out.device must be equal when both are provided. Got device={device} and out.device={out.device}."
                    )
                else:
                    device = torch.device(device)
                    out._device = device
                    for node in out.values(True, True, is_leaf=_is_tensor_collection):
                        if is_tensorclass(node):
                            node._tensordict._device = device
                        else:
                            node._device = device
        else:

            def make_result(names=names, batch_size=batch_size):
                if names is NO_DEFAULT:
                    if batch_size is not None:
                        # erase names
                        names = None
                    else:
                        names = self.names if self._has_names() else None
                return self.empty(batch_size=batch_size, device=device, names=names)

            result = None
            is_locked = False

        any_set = False
        if is_leaf is None:
            is_leaf = _default_is_leaf
        is_sub_td = isinstance(self, _SUBTENSORDICT_TYPE)

        for key, item in self.items():
            if (
                not call_on_nested and not is_leaf(type(item))
                # and not is_non_tensor(item)
            ):
                if default is not NO_DEFAULT:
                    _others = [_other._get_str(key, default=None) for _other in others]
                    _others = [
                        self.empty(recurse=True) if _other is None else _other
                        for _other in _others
                    ]
                else:
                    _others = [
                        _other._get_str(key, default=NO_DEFAULT) for _other in others
                    ]

                item_trsf = item._apply_nest(
                    fn,
                    *_others,
                    inplace=inplace,
                    batch_size=batch_size,
                    device=device,
                    checked=checked,
                    named=named,
                    nested_keys=nested_keys,
                    default=default,
                    prefix=prefix + (key,),
                    filter_empty=filter_empty,
                    is_leaf=is_leaf,
                    out=out._get_str(key, default=None) if out is not None else None,
                    call_on_unbatched=call_on_unbatched,
                    **constructor_kwargs,
                )
            else:
                # Pass-through values (e.g., UnbatchedTensor) with shape-changing ops
                # (indicated by batch_size being set) keep their payload unchanged
                # but must expose the new TensorDict-facing batch metadata.
                # For other ops (data ops like zero_, and to() with a
                # batch_size, which passes call_on_unbatched=True), apply the
                # function normally.
                if (
                    batch_size is not None
                    and not call_on_unbatched
                    and _is_unbatched(item)
                ):
                    item_trsf = _unbatched_with_batch_size(item, batch_size)
                else:
                    _others = [
                        _other._get_str(key, default=default) for _other in others
                    ]
                    if named:
                        if nested_keys:
                            item_trsf = fn(
                                prefix + (key,) if prefix != () else key, item, *_others
                            )
                        else:
                            item_trsf = fn(key, item, *_others)
                    else:
                        item_trsf = fn(item, *_others)
                    if (
                        call_on_unbatched
                        and batch_size is not None
                        and _is_unbatched(item)
                        and _is_unbatched(item_trsf)
                    ):
                        item_trsf = _unbatched_with_batch_size(item_trsf, batch_size)
            if item_trsf is not None:
                if not any_set:
                    if result is None:
                        result = make_result()
                    any_set = True
                if is_sub_td:
                    result.set(key, item_trsf, inplace=inplace)
                else:
                    result._set_str(
                        key,
                        item_trsf,
                        inplace=BEST_ATTEMPT_INPLACE if inplace else False,
                        validated=checked,
                        non_blocking=False,
                    )

        if filter_empty and not any_set:
            return
        elif filter_empty is None and not any_set and not self.is_empty():
            # we raise the deprecation warning only if the tensordict wasn't already empty.
            # After we introduce the new behaviour, we will have to consider what happens
            # to empty tensordicts by default: will they disappear or stay?
            return
        if result is None:
            result = make_result()

        if not inplace and is_locked:
            result.lock_()
        return result

    def _is_dense_node(self) -> bool:
        """Whether :meth:`_fast_apply` can use ``_dense_apply``: see ``TensorDict._is_dense_node``."""
        return False

    def _fast_apply(
        self,
        fn: Callable,
        *others: T,
        batch_size: Sequence[int] | None = None,
        device: torch.device | None = NO_DEFAULT,
        names: Sequence[str] | None = NO_DEFAULT,
        inplace: bool = False,
        call_on_nested: bool = False,
        default: Any = NO_DEFAULT,
        named: bool = False,
        nested_keys: bool = False,
        # filter_empty must be False because we use _fast_apply for all sorts of ops like expand etc
        # and non-tensor data will disappear if we use True by default.
        filter_empty: bool | None = False,
        is_leaf: Callable[[Type], bool] | None = None,
        propagate_lock: bool = False,
        out: TensorDictBase | None = None,
        num_threads: int = 0,
        checked: bool = True,
        **constructor_kwargs,
    ) -> Self | None:
        """A faster apply method.

        This method does not check the results of ``fn``: the caller must make
        sure that their metadata (device, shape etc.) match what
        :meth:`~.apply` would accept.

        """
        if num_threads:

            def func(*args, **kwargs):
                return self._multithread_apply_nest(
                    *args, **kwargs, num_threads=num_threads
                )

        elif (
            not others
            and out is None
            and not inplace
            and batch_size is None
            and checked
            and not call_on_nested
            # to() passes call_on_unbatched, which _dense_apply takes
            and (
                not constructor_kwargs
                or list(constructor_kwargs) == ["call_on_unbatched"]
            )
            and not is_compiling()
            and names is NO_DEFAULT
            and (
                is_leaf is None
                or is_leaf is _default_is_leaf
                or is_leaf is _NESTED_TENSORS_AS_LISTS
            )
            and self._is_dense_node()
        ):
            # An exact TensorDict without dim names: build the result
            # directly, as _apply_nest would, without its generic steps.
            result = self._dense_apply(
                fn,
                named,
                nested_keys,
                (),
                filter_empty,
                _default_is_leaf if is_leaf is None else is_leaf,
                device,
                constructor_kwargs.get("call_on_unbatched", False),
            )
            if propagate_lock and self.is_locked and result is not None:
                result.lock_()
            return result
        else:
            func = self._apply_nest
        result = func(
            fn,
            *others,
            batch_size=batch_size,
            device=device,
            names=names,
            inplace=inplace,
            checked=checked,
            call_on_nested=call_on_nested,
            named=named,
            default=default,
            nested_keys=nested_keys,
            filter_empty=filter_empty,
            is_leaf=is_leaf,
            out=out,
            **constructor_kwargs,
        )
        if propagate_lock and not inplace and self.is_locked and result is not None:
            result.lock_()
        return result

    def _inplace_rebind_leaves(
        self,
        leaf_fn: Callable[[Any], Any],
        nested_fn: Callable[[T], None],
        new_batch_size: torch.Size | None = None,
    ) -> Self:
        """Replace every tensor leaf in this tensordict in place.

        Shared building block for shape-changing inplace ops (repeat, gather,
        reshape, ...). Iterates over keys, applies ``leaf_fn`` to each tensor
        leaf and uses ``_set_str`` to swap the entry; the function-local
        reference to the old leaf is dropped before the swap so its storage
        refcount falls to zero and the allocator can reuse the block for the
        next leaf. Nested tensor collections are handled by ``nested_fn``,
        which is expected to mutate the child in place. ``UnbatchedTensor`` /
        ``_pass_through`` leaves are left untouched.

        After all leaves are processed, the dict's advertised ``batch_size``
        is flipped via ``_change_batch_size`` when ``new_batch_size`` is
        provided and differs from the current one — bypassing the redundant
        shape check in ``_batch_size_setter``.
        """
        if (
            not self._inplace_batch_size_change
            and new_batch_size is not None
            and tuple(new_batch_size) != tuple(self.batch_size)
        ):
            raise NotImplementedError(
                "An in-place shape operation that changes the batch size is not "
                f"supported on {type(self).__name__}: call to_tensordict() first, "
                "or use inplace=False."
            )
        keys = list(self.keys())
        for key in keys:
            leaf = self._get_str(key, default=None)
            if leaf is None:
                continue
            if _is_unbatched(leaf):
                if new_batch_size is not None and leaf.batch_size != new_batch_size:
                    self._set_str(
                        key,
                        _unbatched_with_batch_size(leaf, new_batch_size),
                        validated=True,
                        inplace=False,
                    )
                continue
            if _is_tensor_collection(type(leaf)):
                nested = _inplace_shape_state.nested
                _inplace_shape_state.nested = True
                try:
                    nested_fn(leaf)
                finally:
                    _inplace_shape_state.nested = nested
                continue
            new_leaf = leaf_fn(leaf)
            if new_leaf is leaf:
                # nothing to rebind, which a locked tensordict allows
                continue
            del leaf
            self._set_str(key, new_leaf, validated=True, inplace=False)
        if new_batch_size is not None and tuple(new_batch_size) != tuple(
            self.batch_size
        ):
            self._change_batch_size(torch.Size(new_batch_size))
        return self

    def _raise_if_holds_lazy_stack(
        self,
        op_name: str,
        prefix: tuple[str, ...] = (),
        *,
        new_batch_size: torch.Size | None = None,
    ) -> None:
        """Raise if any nested entry is lazy, before an in-place shape op.

        ``_inplace_rebind_leaves`` rewrites the leaves in key order, and the
        ``LazyStackedTensorDict`` (and ``NonTensorStack``) versions of
        ``reshape``, ``flatten``, ``unflatten``, ``repeat``,
        ``repeat_interleave``, ``gather`` and ``masked_select`` refuse
        ``inplace=True``. Finding a lazy stack halfway through would leave the
        earlier leaves rewritten while ``batch_size`` is unchanged, so the
        whole tree is checked first. The check tests ``_lazy``, so it also
        catches ``_SubTensorDict``, which refuses these ops too. The legacy
        lazy views (``_CustomOpTensorDict``) refuse ``flatten``, ``unflatten``,
        ``repeat``, ``gather`` and ``masked_select`` but not ``reshape`` and
        ``repeat_interleave``, so the check lets them through for those two.
        The calls on nested tensordicts skip the walk, which the top-level
        call has done.

        ``new_batch_size`` is the batch size that the op gives ``self``. When
        it differs from the batch size, a nested tensordict whose
        ``_inplace_batch_size_change`` is False (a file, a ``TypedTensorDict``
        over a lazy stack) raises too, before any leaf is rewritten. The
        top-level call sets it to None when the batch size stays; the nested
        calls receive it only as a sign that the batch size changes (it is not
        their batch size).
        """
        if not prefix:
            if _inplace_shape_state.nested:
                return
            if new_batch_size is not None and (
                tuple(new_batch_size) == tuple(self.batch_size)
                or not self._inplace_batch_size_change
            ):
                # the batch size stays, or self raises before any write
                new_batch_size = None
        for key, value in self.items():
            cls = type(value)
            if cls is Tensor:
                continue
            if not isinstance(value, TensorDictBase):
                if not _is_tensorclass(cls) or _is_non_tensor(cls):
                    continue
                value = value._tensordict
            nested_key = prefix + (key,)
            if value._lazy:
                from tensordict._lazy import _CustomOpTensorDict

                if not (
                    op_name in ("reshape", "repeat_interleave")
                    and isinstance(value, _CustomOpTensorDict)
                ):
                    raise NotImplementedError(
                        f"{op_name}(inplace=True) is not supported on a tensordict "
                        f"that holds a lazy tensordict (such as a "
                        f"LazyStackedTensorDict or a NonTensorStack); found a "
                        f"{type(value).__name__} at key "
                        f"{nested_key[0] if len(nested_key) == 1 else nested_key!r}. "
                        f"Use inplace=False instead."
                    )
            if new_batch_size is not None and not value._inplace_batch_size_change:
                raise NotImplementedError(
                    f"{op_name}(inplace=True) changes the batch size, which a "
                    f"{type(value).__name__} at key "
                    f"{nested_key[0] if len(nested_key) == 1 else nested_key!r} "
                    "cannot do in place. Use inplace=False instead."
                )
            value._raise_if_holds_lazy_stack(
                op_name, nested_key, new_batch_size=new_batch_size
            )

    def map(
        self,
        fn: Callable[[TensorCollection], TensorCollection | None],
        dim: int = 0,
        num_workers: int | None = None,
        *,
        out: TensorCollection | None = None,
        chunksize: int | None = None,
        num_chunks: int | None = None,
        pool: mp.Pool | None = None,
        generator: torch.Generator | None = None,
        max_tasks_per_child: int | None = None,
        worker_threads: int = 1,
        index_with_generator: bool = False,
        pbar: bool = False,
        mp_start_method: str | None = None,
    ) -> Self:
        """Applies a function to chunks of the tensordict in worker processes, and gathers the results.

        The tensordict is split along ``dim`` into chunks, ``fn`` is called on
        each chunk in a pool of worker processes, and the results are joined
        along ``dim``: concatenated with :func:`torch.cat`, or stacked when
        ``chunksize=0``. For an iterator over the results, see
        :meth:`~.map_iter`.

        The chunks are slices of the tensordict, not copies. On a memory-mapped
        tensordict (see :meth:`~.memmap_`), the workers read them from the
        files without a copy, which makes it possible to process data that is
        larger than memory.

        Args:
            fn (Callable): the function called on each chunk, a tensordict
                that holds a slice of the tensordict along ``dim``. It returns
                a tensordict, a tensor, or ``None``. It must be picklable: define
                it at the top level of a module or script, not as a lambda.
            dim (int, optional): the dimension along which the tensordict is
                split into chunks. Defaults to ``0``.
            num_workers (int, optional): the number of worker processes.
                Ignored when ``pool`` is given. Defaults to ``None``, which uses
                one worker per CPU.

        Keyword Args:
            out (TensorDictBase, optional): a tensordict to write the results
                into, with the batch size of the result. If it is in shared
                memory or memory-mapped (:meth:`~.is_shared` or
                :meth:`~.is_memmap` is ``True``), the workers write into it
                directly; otherwise the results are sent back and written into
                it in the main process. Defaults to ``None``.
            chunksize (int, optional): the size of each chunk along ``dim``.
                ``0`` passes the elements one by one, unbound along ``dim``.
                Exclusive with ``num_chunks``. Defaults to ``None``.
            num_chunks (int, optional): the number of chunks. Exclusive with
                ``chunksize``. Defaults to ``None``: if neither is given, the
                tensordict is split into one chunk per worker. Use more chunks
                when a chunk per worker does not fit in memory.
            pool (multiprocessing.Pool, optional): a pool of processes to run
                the job. When it is given, ``num_workers``, ``generator``,
                ``max_tasks_per_child``, ``worker_threads`` and
                ``mp_start_method`` are ignored. Defaults to ``None``, which
                creates a pool for the call and closes it afterwards.
            generator (torch.Generator, optional): the generator that draws the
                base seed of the workers. Worker ``i`` seeds torch with the
                base seed plus ``i``, and NumPy with a seed derived from both,
                so that the workers draw different random numbers. This makes the workers independent,
                not the results reproducible: which worker gets which chunk
                changes from call to call. Defaults to ``None``, which uses a
                new :class:`torch.Generator`, so the base seed is the same on
                every call. To leave the workers unseeded, pass a ``pool``.
            max_tasks_per_child (int, optional): the number of chunks a worker
                process handles before it is replaced by a new one. Defaults to
                ``None``, which keeps the workers for the whole call.
            worker_threads (int, optional): the number of torch threads in each
                worker. Defaults to ``1``.
            index_with_generator (bool, optional): if ``True``, each chunk is
                indexed when a worker asks for it, instead of all at once
                before the job starts. This starts faster, but indexing is
                slower than :meth:`~.split` and :meth:`~.chunk`, so the whole
                call may take longer. Defaults to ``False``.
            pbar (bool, optional): if ``True``, shows a progress bar. Needs
                ``tqdm``; without it, no bar is shown. Defaults to ``False``.
            mp_start_method (str, optional): the start method of the worker
                processes, for example ``"fork"`` or ``"spawn"``. CUDA tensors
                cannot be shared with processes started with ``"fork"``.
                Defaults to ``None``, which uses the default start method of
                :mod:`torch.multiprocessing`.

        Returns:
            The results joined along ``dim``, a tensordict or a tensor
            depending on what ``fn`` returns. When ``out`` is given, ``out``.
            Otherwise, ``None`` if ``fn`` returns ``None`` for every chunk.

        Examples:
            Run this example as a script, not in a REPL: the worker processes
            must import ``process_data``, and the ``if __name__ == "__main__"``
            block keeps them from running the main code again.

            .. code-block:: python

                import torch
                from tensordict import TensorDict

                def process_data(data):
                    # The chunks of a memory-mapped tensordict are locked:
                    # return a new tensordict instead of adding a key to data.
                    return data.empty().set("y", data["x"] + 1)

                if __name__ == "__main__":
                    data = TensorDict({"x": torch.zeros(1, 1_000)}, [1, 1_000]).memmap_()
                    data = data.map(process_data, dim=1, num_workers=2)
                    print(data["y"][:, :10])
                    # tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
        """
        from torch import multiprocessing as mp

        if pool is None:
            if num_workers is None:
                num_workers = mp.cpu_count()  # Get the number of CPU cores
            if generator is None:
                generator = torch.Generator()
            seed = (
                torch.empty((), dtype=torch.int64).random_(generator=generator).item()
            )
            if mp_start_method is not None:
                ctx = mp.get_context(mp_start_method)
            else:
                ctx = mp.get_context()

            queue = ctx.Queue(maxsize=num_workers)
            for i in range(num_workers):
                queue.put(i)
            with ctx.Pool(
                processes=num_workers,
                initializer=_proc_init,
                initargs=(seed, queue, worker_threads),
                maxtasksperchild=max_tasks_per_child,
            ) as pool:
                return self._map(
                    fn=fn,
                    dim=dim,
                    chunksize=chunksize,
                    num_chunks=num_chunks,
                    pool=pool,
                    pbar=pbar,
                    out=out,
                    index_with_generator=index_with_generator,
                    iterable=False,
                    shuffle=False,
                )
        else:
            return self._map(
                fn=fn,
                dim=dim,
                chunksize=chunksize,
                num_chunks=num_chunks,
                pool=pool,
                pbar=pbar,
                out=out,
                index_with_generator=index_with_generator,
                iterable=False,
                shuffle=False,
            )

    def map_iter(
        self,
        fn: Callable[[TensorCollection], TensorCollection | None],
        dim: int = 0,
        num_workers: int | None = None,
        *,
        shuffle: bool = False,
        chunksize: int | None = None,
        num_chunks: int | None = None,
        pool: mp.Pool | None = None,
        generator: torch.Generator | None = None,
        max_tasks_per_child: int | None = None,
        worker_threads: int = 1,
        index_with_generator: bool = True,
        pbar: bool = False,
        mp_start_method: str | None = None,
    ) -> Iterator[T]:
        """Applies a function to chunks of the tensordict in worker processes, and yields the results.

        The iterator version of :meth:`~.map`: the tensordict is split along
        ``dim`` into chunks, ``fn`` is called on each chunk in a pool of worker
        processes, and the result for each chunk is yielded as soon as it is
        ready, in the order of the chunks (unless ``shuffle=True``). This can
        serve as a data loader over a tensordict.

        The chunks are slices of the tensordict, not copies. On a memory-mapped
        tensordict (see :meth:`~.memmap_`), the workers read them from the
        files without a copy, which makes it possible to process data that is
        larger than memory.

        Args:
            fn (Callable): the function called on each chunk, a tensordict
                that holds a slice of the tensordict along ``dim``. It returns
                a tensordict, a tensor, or ``None``. It must be picklable: define
                it at the top level of a module or script, not as a lambda.
            dim (int, optional): the dimension along which the tensordict is
                split into chunks. Defaults to ``0``.
            num_workers (int, optional): the number of worker processes.
                Ignored when ``pool`` is given. Defaults to ``None``, which uses
                one worker per CPU.

        Keyword Args:
            shuffle (bool, optional): if ``True``, the chunks take their
                elements from a random permutation of the tensordict along
                ``dim``, so a chunk holds elements that are not contiguous, and
                the results are yielded in the order the workers finish them.
                Needs ``index_with_generator=True``: otherwise it raises
                ``RuntimeError``. Defaults to ``False``.
            chunksize (int, optional): the size of each chunk along ``dim``.
                ``0`` passes the elements one by one, unbound along ``dim``.
                Exclusive with ``num_chunks``. Defaults to ``None``.
            num_chunks (int, optional): the number of chunks. Exclusive with
                ``chunksize``. Defaults to ``None``: if neither is given, the
                tensordict is split into one chunk per worker. Use more chunks
                when a chunk per worker does not fit in memory.
            pool (multiprocessing.Pool, optional): a pool of processes to run
                the job. When it is given, ``num_workers``, ``generator``,
                ``max_tasks_per_child``, ``worker_threads`` and
                ``mp_start_method`` are ignored. Defaults to ``None``, which
                creates a pool for the call and closes it afterwards.
            generator (torch.Generator, optional): the generator that draws the
                base seed of the workers. Worker ``i`` seeds torch with the
                base seed plus ``i``, and NumPy with a seed derived from both,
                so that the workers draw different random numbers. This makes the workers independent,
                not the results reproducible: which worker gets which chunk
                changes from call to call. Defaults to ``None``, which uses a
                new :class:`torch.Generator`, so the base seed is the same on
                every call. To leave the workers unseeded, pass a ``pool``.
            max_tasks_per_child (int, optional): the number of chunks a worker
                process handles before it is replaced by a new one. Defaults to
                ``None``, which keeps the workers for the whole call.
            worker_threads (int, optional): the number of torch threads in each
                worker. Defaults to ``1``.
            index_with_generator (bool, optional): if ``True``, each chunk is
                indexed when a worker asks for it, instead of all at once
                before the job starts. This saves the memory and the time of
                splitting the whole tensordict upfront, but indexing is slower
                than :meth:`~.split` and :meth:`~.chunk`. Defaults to ``True``,
                unlike :meth:`~.map`.
            pbar (bool, optional): if ``True``, shows a progress bar. Needs
                ``tqdm``; without it, no bar is shown. Defaults to ``False``.
            mp_start_method (str, optional): the start method of the worker
                processes, for example ``"fork"`` or ``"spawn"``. CUDA tensors
                cannot be shared with processes started with ``"fork"``.
                Defaults to ``None``, which uses the default start method of
                :mod:`torch.multiprocessing`.

        Yields:
            The result of ``fn`` for each chunk.

        Examples:
            Run this example as a script, not in a REPL: the worker processes
            must import ``process_data``, and the ``if __name__ == "__main__"``
            block keeps them from running the main code again.

            .. code-block:: python

                import torch
                from tensordict import TensorDict

                def process_data(data):
                    # The chunks of a memory-mapped tensordict are locked:
                    # return a new tensordict instead of adding a key to data.
                    return data.empty().set("y", data["x"] + 1)

                if __name__ == "__main__":
                    data = TensorDict({"x": torch.zeros(1, 1_000)}, [1, 1_000]).memmap_()
                    for sample in data.map_iter(process_data, dim=1, num_workers=2, chunksize=5):
                        print(sample["y"])
                        # tensor([[1., 1., 1., 1., 1.]])
                        break
        """
        from torch import multiprocessing as mp

        if pool is None:
            if num_workers is None:
                num_workers = mp.cpu_count()  # Get the number of CPU cores
            if generator is None:
                generator = torch.Generator()
            seed = (
                torch.empty((), dtype=torch.int64).random_(generator=generator).item()
            )
            if mp_start_method is not None:
                ctx = mp.get_context(mp_start_method)
            else:
                ctx = mp.get_context()

            queue = ctx.Queue(maxsize=num_workers)
            for i in range(num_workers):
                queue.put(i)
            pool = ctx.Pool(
                processes=num_workers,
                initializer=_proc_init,
                initargs=(seed, queue, worker_threads),
                maxtasksperchild=max_tasks_per_child,
            )
            try:
                yield from self._map(
                    fn=fn,
                    dim=dim,
                    chunksize=chunksize,
                    num_chunks=num_chunks,
                    pool=pool,
                    pbar=pbar,
                    out=None,
                    index_with_generator=index_with_generator,
                    iterable=True,
                    shuffle=shuffle,
                )
            finally:
                try:
                    pool.close()
                    pool.join()
                except Exception:
                    pool.terminate()
        else:
            yield from self._map(
                fn=fn,
                dim=dim,
                chunksize=chunksize,
                num_chunks=num_chunks,
                pool=pool,
                pbar=pbar,
                out=None,
                index_with_generator=index_with_generator,
                iterable=True,
                shuffle=shuffle,
            )

    def _map(
        self,
        fn: Callable[[TensorDictBase], TensorDictBase | None],
        dim: int = 0,
        *,
        shuffle: bool = False,
        out: TensorDictBase | None = None,
        chunksize: int | None = None,
        num_chunks: int | None = None,
        pool: mp.Pool | None = None,
        index_with_generator: bool = False,
        pbar: bool = False,
        iterable: bool,
    ):
        num_workers = pool._processes
        dim = _maybe_correct_neg_dim(dim, self.batch_size)

        self_split = _split_tensordict(
            self,
            chunksize,
            num_chunks,
            num_workers,
            dim,
            shuffle=shuffle,
            use_generator=index_with_generator,
            to_tensordict=self._map_on_dense_copies,
        )
        if not index_with_generator:
            length = len(self_split)
            if self._map_on_dense_copies:
                self_split = tuple(split.to_tensordict() for split in self_split)
        else:
            length = None
        call_chunksize = 1

        if out is not None and (out.is_shared() or out.is_memmap()):

            def wrap_fn_with_out(fn, out):
                @wraps(fn)
                def newfn(item_and_out):
                    item, out = item_and_out
                    result = fn(item)
                    out.update_(result)
                    return

                out_split = _split_tensordict(
                    out,
                    chunksize,
                    num_chunks,
                    num_workers,
                    dim,
                    shuffle=shuffle,
                    use_generator=index_with_generator,
                )
                return _CloudpickleWrapper(newfn), _zip_strict(self_split, out_split)

            # The workers fill out and return None, so the loop below leaves
            # out alone and returns it.
            fn, self_split = wrap_fn_with_out(fn, out)

        imap_fn = pool.imap if not shuffle else pool.imap_unordered
        imap = imap_fn(fn, self_split, call_chunksize)

        if pbar and importlib.util.find_spec("tqdm", None) is not None:
            import tqdm

            imap = tqdm.tqdm(imap, total=length)

        if iterable:
            return imap
        else:
            imaplist = []
            start = 0
            base_index = (slice(None),) * dim
            for item in imap:
                if item is not None:
                    if out is not None:
                        if chunksize == 0:
                            out[base_index + (start,)].update_(item)
                            start += 1
                        else:
                            end = start + item.shape[dim]
                            chunk = base_index + (slice(start, end),)
                            out[chunk].update_(item)
                            start = end
                    else:
                        imaplist.append(item)
            del imap

            # support inplace modif
            if imaplist:
                if chunksize == 0:
                    from tensordict._lazy import LazyStackedTensorDict

                    # We want to be able to return whichever data structure
                    with set_capture_non_tensor_stack(False):
                        out = LazyStackedTensorDict.maybe_dense_stack(imaplist, dim)
                else:
                    out = torch.cat(imaplist, dim)
            return out

    def __copy__(self):
        """Returns a shallow copy of the tensordict.

        The nested tensordicts are new objects, and the leaves are shared.
        Called by :func:`copy.copy`. Same as :meth:`~.copy`.
        """
        return self.copy()

    def norm(
        self,
        *,
        out=None,
        dtype: torch.dtype | None = None,
        key_transform: Callable[[NestedKey], NestedKey] | None = None,
    ) -> Self:
        """Returns a tensordict with the norm of each leaf.

        Each tensor leaf is replaced by its 2-norm, computed over all its
        elements, so the result has an empty batch size. Non-tensor data is
        kept as it is.

        Keyword Args:
            out (TensorDictBase, optional): a tensordict to write the norms
                into, with the keys of the tensordict and an empty batch size.
                Defaults to ``None``.
            dtype (torch.dtype, optional): the dtype of the norms. Defaults to
                ``None``, which keeps the dtype of each leaf.
            key_transform (Callable, optional): a function that renames the
                keys of the result. It is called on each string key, level by
                level: ``("a", "b")`` becomes
                ``(key_transform("a"), key_transform("b"))``. Defaults to
                ``None``, which keeps the keys.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.full((3, 4), 2.0), b=TensorDict(c=torch.ones(3)), batch_size=[3])
            >>> norms = td.norm()
            >>> norms["a"], norms["b", "c"], norms.batch_size
            (tensor(6.9282), tensor(1.7321), torch.Size([]))
            >>> list(td.norm(key_transform=lambda key: f"{key}_norm").keys(True, True))
            ['a_norm', ('b_norm', 'c_norm')]
        """
        keys, vals = self._items_list(True, True, collapse=True)
        if vals:
            vals = _foreach("norm", vals, dtype=dtype)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        result = self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            batch_size=torch.Size([]),
            propagate_lock=True,
            out=out,
        )
        if key_transform is not None:
            result = result._transform_keys(key_transform)
        return result

    def _clone_recurse(self) -> Self:  # noqa: D417
        keys, vals = self._items_list(True, True)
        foreach_vals = {}
        iter_vals = {}
        for key, val in zip(keys, vals):
            if (
                type(val) is torch.Tensor
                and not val.requires_grad
                and val.dtype not in (torch.bool,)
            ):
                foreach_vals[key] = val
            else:
                iter_vals[key] = val
        if foreach_vals:
            foreach_vals = dict(
                _zip_strict(
                    foreach_vals.keys(),
                    _foreach("add", tuple(foreach_vals.values()), 0),
                )
            )
        if iter_vals:
            iter_vals = dict(
                _zip_strict(
                    iter_vals.keys(),
                    (
                        val.clone() if hasattr(val, "clone") else val
                        for val in iter_vals.values()
                    ),
                )
            )

        items = foreach_vals
        items.update(iter_vals)

        def pop(name, val):
            return items.pop(name, None)

        result = self._fast_apply(
            pop,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=False,
            filter_empty=False,
            default=None,
        )
        if items:
            result.update(items)
        return result

    # Functorch compatibility
    @abc.abstractmethod
    @_cache_while_locked  # noqa: B019
    def _add_batch_dim(self, *, in_dim: int, vmap_level: int) -> Self:
        raise NotImplementedError

    @abc.abstractmethod
    @_cache_while_locked  # noqa: B019
    def _remove_batch_dim(self, vmap_level: int, batch_size: int, out_dim: int) -> Self:
        raise NotImplementedError

    @abc.abstractmethod
    @_cache_while_locked  # noqa: B019
    def _maybe_remove_batch_dim(
        self, funcname: str, vmap_level: int, batch_size: int, out_dim: int
    ) -> Self:
        raise NotImplementedError

    # Validation and checks
    def _convert_to_tensor(
        self, array: Any
    ) -> Tensor | "NonTensorData" | TensorDictBase:  # noqa: F821
        # We are sure that array is not a dict or anything in _ACCEPTED_CLASSES
        castable = None
        if isinstance(array, (float, int, bool)):
            castable = True
        elif isinstance(array, list) and list_to_stack():
            return _convert_list_to_stack(array)[0]
        elif isinstance(array, np.bool_):
            castable = True
            array = array.item()
        elif isinstance(array, (np.ndarray, np.number)):
            if array.dtype.names is not None:
                return TensorDictBase.from_struct_array(array, device=self.device)
            castable = array.dtype.kind in ("c", "i", "f", "b", "u")
        elif isinstance(array, (list, tuple)):
            value = array
            array = np.asarray(array)
            castable = array.dtype.kind in ("c", "i", "f", "b", "u")
            if array.dtype.kind in ("c", "f") and isinstance(value, tuple):
                _warn_tuple_dtype(value, array)
        elif hasattr(array, "numpy"):
            # tf.Tensor with no shape can't be converted otherwise
            array = array.numpy()
            castable = array.dtype.kind in ("c", "i", "f", "b", "u")
        if castable:
            # torch cannot share the memory of a read-only array: copy it.
            if hasattr(array, "flags") and not array.flags.writeable:
                array = array.copy()
            try:
                return torch.as_tensor(array, device=self.device)
            except ValueError:
                if not isinstance(array, np.ndarray):
                    raise
            # torch cannot view an array whose strides are negative or not a
            # multiple of the item size (e.g. a field of a packed structured
            # array), or whose byte order is not native: copy it to a native,
            # C-contiguous array. Any other error is raised again here.
            array = array.astype(array.dtype.newbyteorder("="), order="C")
            return torch.as_tensor(array, device=self.device)
        else:
            from tensordict.tensorclass import NonTensorData

            return NonTensorData(
                data=array,
                batch_size=self.batch_size,
                device=self.device,
                names=self._maybe_names(),
            )

    def _check_batch_size(self, *, raise_exception: bool = True) -> None | bool:
        batch_dims = self.batch_dims
        val = True
        for value in self.values():
            if _is_tensor_collection(type(value)):
                val &= value._check_batch_size(raise_exception=raise_exception)
                if not val:
                    return False
            val &= _shape(value)[:batch_dims] == self.batch_size
            if not val:
                if raise_exception:
                    raise RuntimeError(
                        f"batch_size are incongruent, got value with shape {_shape(value)}, "
                        f"-- expected {self.batch_size}"
                    )
                return False
        return val

    def _check_new_batch_size(self, new_size: torch.Size) -> None:
        batch_dims = len(new_size)
        for key, tensor in self.items():
            if _is_unbatched(tensor):
                continue
            if _shape(tensor)[:batch_dims] != new_size and not (
                _is_tensor_collection(type(tensor)) and tensor.is_empty()
            ):
                raise RuntimeError(
                    f"the {type(tensor).__name__} {key} has shape {_shape(tensor)} which "
                    f"is incompatible with the batch-size {new_size}."
                )

    @property
    def _validate_value(self):
        if is_compiling():
            return self._validate_value_generic
        if self.device:
            if self.batch_size:
                method_name = "_validate_value_generic"
            else:
                method_name = "_validate_value_batchfree"
        else:
            if self.batch_size:
                method_name = "_validate_value_devicefree"
            else:
                method_name = "_validate_value_batchfree_devicefree"
        return getattr(self, method_name)

    def _validate_value_generic(
        self,
        value: CompatibleType | dict[str, CompatibleType],
        non_blocking: bool = False,
        *,
        check_shape: bool = True,
        key: NestedKey | None = None,
    ) -> CompatibleType | dict[str, CompatibleType]:
        cls = type(value)
        if issubclass(cls, torch.Tensor):
            is_tc = False
        elif _is_tensor_collection(cls):
            is_tc = True
        elif issubclass(cls, dict):
            # We use non-blocking if someone's watching or if non-blocking is explicitly passed
            value = self._convert_to_tensordict(
                value, non_blocking=_device_recorder.marked or non_blocking
            )
            is_tc = True
        elif not _is_accepted_class(cls):
            # If cls is not a tensor
            try:
                value = self._convert_to_tensor(value)
            except ValueError as err:
                raise ValueError(
                    f"TensorDict conversion only supports tensorclasses, tensordicts,"
                    f" numeric scalars and tensors. Got {type(value)}"
                ) from err
            is_tc = _is_tensor_collection(cls)
        if _is_unbatched(value):
            device = self.device
            if device is not None and value.device != device:
                if _device_recorder.marked and device.type != "cuda":
                    _device_recorder.record_transfer(device)
                value = value.to(device, non_blocking=non_blocking)
            if value.batch_size != self.batch_size:
                value = _unbatched_with_batch_size(value, self.batch_size)
            return value
        batch_size = self.batch_size
        # A tensor collection's shape is its batch size
        if (
            check_shape
            and (value.batch_size if is_tc else _shape(value))[: len(batch_size)]
            != batch_size
        ):
            # if TensorDict, let's try to map it to the desired shape
            if is_tc:
                # we must clone the value before not to corrupt the data passed to set()
                value = value.clone(recurse=False)
                value.batch_size = self.batch_size
            else:
                raise _batch_mismatch_error(self.batch_size, value, key)
        device = self.device
        if device is not None and value.device != device:
            if _device_recorder.marked and device.type != "cuda":
                _device_recorder.record_transfer(device)
            value = value.to(device, non_blocking=non_blocking)
        if check_shape:
            if not is_tc:
                return value
            has_names = self._has_names()
            # we do our best to match the dim names of the value and the
            # container.
            if has_names:
                if value.names[: self.batch_dims] != self.names:
                    # we clone not to corrupt the value
                    value = value.clone(False).refine_names(*self.names)
            else:
                if value._has_names():
                    names = value.names[: self.batch_dims]
                    # an all-None prefix would re-walk every child for nothing
                    if any(name is not None for name in names):
                        self._set_names(names)
        return value

    def _validate_value_batchfree(
        self,
        value: CompatibleType | dict[str, CompatibleType],
        non_blocking: bool = False,
        *,
        check_shape: bool = True,
        key: NestedKey | None = None,
    ) -> CompatibleType | dict[str, CompatibleType]:
        cls = type(value)
        if issubclass(cls, torch.Tensor) or _is_tensor_collection(cls):
            pass
        elif issubclass(cls, dict):
            # We use non-blocking if someone's watching or if non-blocking is explicitly passed
            value = self._convert_to_tensordict(
                value, non_blocking=_device_recorder.marked or non_blocking
            )
        elif not _is_accepted_class(cls):
            # If cls is not a tensor
            try:
                value = self._convert_to_tensor(value)
            except ValueError as err:
                raise ValueError(
                    f"TensorDict conversion only supports tensorclasses, tensordicts,"
                    f" numeric scalars and tensors. Got {type(value)}"
                ) from err
        device = self.device
        if device is not None and value.device != device:
            if _device_recorder.marked and device.type != "cuda":
                _device_recorder.record_transfer(device)
            value = value.to(device, non_blocking=non_blocking)
        if _is_unbatched(value):
            if value.batch_size != self.batch_size:
                value = _unbatched_with_batch_size(value, self.batch_size)
            return value
        return value

    def _validate_value_devicefree(
        self,
        value: CompatibleType | dict[str, CompatibleType],
        non_blocking: bool = False,
        *,
        check_shape: bool = True,
        key: NestedKey | None = None,
    ) -> CompatibleType | dict[str, CompatibleType]:
        cls = type(value)
        if issubclass(cls, torch.Tensor):
            is_tc = False
        elif _is_tensor_collection(cls):
            is_tc = True
        elif issubclass(cls, dict):
            # We use non-blocking if someone's watching or if non-blocking is explicitly passed
            value = self._convert_to_tensordict(
                value, non_blocking=_device_recorder.marked or non_blocking
            )
            is_tc = True
        elif not _is_accepted_class(cls):
            # If cls is not a tensor
            try:
                value = self._convert_to_tensor(value)
            except ValueError as err:
                raise ValueError(
                    f"TensorDict conversion only supports tensorclasses, tensordicts,"
                    f" numeric scalars and tensors. Got {type(value)}"
                ) from err
            is_tc = _is_tensor_collection(cls)
        if _is_unbatched(value):
            if value.batch_size != self.batch_size:
                value = _unbatched_with_batch_size(value, self.batch_size)
            return value

        batch_size = self.batch_size
        # A tensor collection's shape is its batch size
        if (
            check_shape
            and (value.batch_size if is_tc else _shape(value))[: len(batch_size)]
            != batch_size
        ):
            # if TensorDict, let's try to map it to the desired shape
            if is_tc:
                # we must clone the value before not to corrupt the data passed to set()
                value = value.clone(recurse=False)
                value.batch_size = self.batch_size
            else:
                raise _batch_mismatch_error(self.batch_size, value, key)
        if check_shape:
            if not is_tc:
                return value
            has_names = self._has_names()
            # we do our best to match the dim names of the value and the
            # container.
            if has_names:
                if value.names[: self.batch_dims] != self.names:
                    # we clone not to corrupt the value
                    value = value.clone(False).refine_names(
                        *(self.names + value.names[self.batch_dims :])
                    )
            else:
                if value._has_names():
                    names = value.names[: self.batch_dims]
                    # an all-None prefix would re-walk every child for nothing
                    if any(name is not None for name in names):
                        self._set_names(names)
        return value

    def _validate_value_batchfree_devicefree(
        self,
        value: CompatibleType | dict[str, CompatibleType],
        non_blocking: bool = False,
        *,
        check_shape: bool = True,
        key: NestedKey | None = None,
    ) -> CompatibleType | dict[str, CompatibleType]:
        cls = type(value)
        if issubclass(cls, torch.Tensor) or _is_tensor_collection(cls):
            pass
        elif issubclass(cls, dict):
            # We use non-blocking if someone's watching or if non-blocking is explicitly passed
            value = self._convert_to_tensordict(
                value, non_blocking=_device_recorder.marked or non_blocking
            )
        elif not _is_accepted_class(cls):
            # If cls is not a tensor
            try:
                value = self._convert_to_tensor(value)
            except ValueError as err:
                raise ValueError(
                    f"TensorDict conversion only supports tensorclasses, tensordicts,"
                    f" numeric scalars and tensors. Got {type(value)}"
                ) from err
        if _is_unbatched(value):
            if value.batch_size != self.batch_size:
                value = _unbatched_with_batch_size(value, self.batch_size)
            return value
        return value

    def __enter__(self):
        is_tc = _is_tensorclass(type(self))
        if not hasattr(self, "_last_op_queue"):
            if is_tc:
                _last_op_queue = self._tensordict._last_op_queue = collections.deque()
            else:
                _last_op_queue = self._last_op_queue = collections.deque()
        else:
            _last_op_queue = (
                self._last_op_queue
                if not _is_tensorclass(type(self))
                else self._tensordict._last_op_queue
            )
        if is_tc:
            # get last-op from tensordict - that's where it's written
            _last_op = self._tensordict._last_op
        else:
            _last_op = self._last_op
        _last_op_queue.append(_last_op)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # During exit, updates mustn't be made in-place as the source and dest
        # storage location can be identical, resulting in a RuntimeError
        if is_compiling():
            self.clear_refs_for_compile_()
        is_tc = _is_tensorclass(type(self))
        _last_op = (
            self._last_op_queue.pop()
            if not is_tc
            else self._tensordict._last_op_queue.pop()
        )
        if exc_type is not None and issubclass(exc_type, Exception):
            # to_module, lock_ and unlock_ change the original object on entry:
            # undo the change. The other ops do not write back after an error.
            if _last_op is not None and _last_op[0] in _UNDO_ON_ERROR_OPS:
                last_op, (args, kwargs, out_wr) = _last_op
                LAST_OP_MAPS[last_op](self, args, kwargs, out_wr())
            return False
        if _last_op is not None:
            last_op, (args, kwargs, out_wr) = _last_op
            # TODO: transpose, flatten etc. as decorator should lock the content to make sure that no key is
            #  added or deleted
            _inv_caller = LAST_OP_MAPS.get(last_op)
            if _inv_caller is not None:
                prev_ref = out_wr()
                result = _inv_caller(self, args, kwargs, prev_ref)
                return result
            else:
                raise NotImplementedError(f"Unrecognised function {last_op}.")
        return self

    def clear_refs_for_compile_(self) -> Self:
        """Clears the record of the last operation of the tensordict and its nested tensordicts.

        The tensordict records its last operation, with a weak reference to its
        result, for the ``with`` form of methods such as :meth:`~.lock_`. Call
        this method on a tensordict before you return it from a function
        compiled with :func:`torch.compile` when compiling fails with
        ``torch._dynamo.exc.Unsupported: reconstruct: WeakRefVariable()``.
        See :ref:`compile`.

        Returns:
            ``self``.
        """
        self._last_op = None
        for v in self.values(True, True, is_leaf=_is_tensor_collection):
            if _is_tensorclass(type(v)):
                v = v._tensordict
            v._last_op = None
        return self

    # Clone, select, exclude, empty
    def select(
        self,
        *keys: NestedKey,
        inplace: bool = False,
        strict: bool = True,
        as_tensordict: bool = False,
    ) -> Self:
        """Returns a tensordict that holds only the entries of ``keys``.

        The values are not copied: the result holds the same tensors as the
        tensordict, so an in-place change to a tensor shows in both. A nested
        key such as ``("next", "obs")`` keeps one entry of a nested tensordict,
        and the key of a nested tensordict keeps all its entries. The result is
        not locked. See :ref:`keys`.

        Args:
            *keys (NestedKey): the keys to keep.

        Keyword Args:
            inplace (bool, optional): if ``True``, removes the other entries from
                the tensordict and returns it. Defaults to ``False``.
            strict (bool, optional): if ``True``, a missing key raises a
                ``KeyError``. If ``False``, missing keys are ignored. Defaults to
                ``True``.
            as_tensordict (bool, optional): if ``True``, a tensorclass returns
                a :class:`~tensordict.TensorDict` instead of an instance of its
                class, whose other fields are ``None``. Has no effect on other
                tensordicts. Defaults to ``False``.

        Returns:
            A tensordict of the same type, or ``self`` if ``inplace=True``.

        .. seealso:: :meth:`~.exclude` keeps the other entries, and
            :meth:`~.split_keys` returns both parts.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(
            ...     obs=torch.zeros(3, 4),
            ...     action=torch.zeros(3, 2),
            ...     next=TensorDict(obs=torch.zeros(3, 4), reward=torch.zeros(3, 1)),
            ...     batch_size=[3],
            ... )
            >>> sub = td.select("obs", ("next", "reward"))
            >>> list(sub.keys(include_nested=True, leaves_only=True))
            ['obs', ('next', 'reward')]
            >>> td.select("obs")["obs"] is td["obs"]
            True
            >>> list(td.select("obs", "missing", strict=False).keys())
            ['obs']
        """
        keys = unravel_key_list(keys)
        result = self._select(*keys, inplace=inplace, strict=strict)
        if not inplace and (result._is_memmap or result._is_shared):
            result.lock_()
        return result

    def _select(
        self,
        *keys: NestedKey,
        inplace: bool = False,
        strict: bool = True,
        set_shared: bool = True,
    ) -> Self:
        # Generic: select in a TensorDict copy (see to_tensordict). Only the
        # entries that the keys name are copied, unless one is missing and
        # the KeyError must list all the keys.
        if inplace:
            raise RuntimeError("Cannot call select inplace on a lazy tensordict.")
        roots = dict.fromkeys(key if isinstance(key, str) else key[0] for key in keys)
        present = [root for root in roots if root in self.keys()]
        if strict and len(present) < len(roots):
            present = None
        return self._copy_entries(present)._select(
            *keys, inplace=False, strict=strict, set_shared=set_shared
        )

    def _copy_entries(self, keys: list[str] | None):
        """A to_tensordict() copy of the top-level entries ``keys`` (all if ``None``)."""
        if keys is None:
            return self.to_tensordict()
        from tensordict._td import TensorDict

        return TensorDict._new_unsafe(
            {key: self._get_str(key, NO_DEFAULT) for key in keys},
            batch_size=self.batch_size,
            device=self.device,
            names=self._maybe_names(),
        ).to_tensordict()

    def exclude(self, *keys: NestedKey, inplace: bool = False) -> Self:
        """Returns a tensordict without the entries of ``keys``.

        The values are not copied: the result holds the same tensors as the
        tensordict, so an in-place change to a tensor shows in both. A nested
        key removes one entry of a nested tensordict, and the key of a nested
        tensordict removes all its entries. Unlike :meth:`~.select`, a missing
        key is ignored. The result is not locked. See :ref:`keys`.

        Args:
            *keys (NestedKey): the keys to remove.

        Keyword Args:
            inplace (bool, optional): if ``True``, removes the entries from the
                tensordict and returns it. Defaults to ``False``.

        Returns:
            A tensordict of the same type, or ``self`` if ``inplace=True``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(
            ...     obs=torch.zeros(3, 4),
            ...     action=torch.zeros(3, 2),
            ...     next=TensorDict(obs=torch.zeros(3, 4), reward=torch.zeros(3, 1)),
            ...     batch_size=[3],
            ... )
            >>> sub = td.exclude("action", ("next", "obs"))
            >>> list(sub.keys(include_nested=True, leaves_only=True))
            ['obs', ('next', 'reward')]
            >>> sub["obs"] is td["obs"]
            True
            >>> sorted(td.exclude("missing").keys())
            ['action', 'next', 'obs']
        """
        keys = unravel_key_list(keys)
        result = self._exclude(*keys, inplace=inplace)
        if not inplace and (result._is_memmap or result._is_shared):
            result.lock_()
        return result

    def _exclude(
        self,
        *keys: NestedKey,
        inplace: bool = False,
        set_shared: bool = True,
    ) -> Self:
        # Generic: exclude in a TensorDict copy of the entries that are kept
        # (see _select).
        if inplace:
            raise RuntimeError("Cannot call exclude inplace on a lazy tensordict.")
        excluded = {key for key in keys if isinstance(key, str)}
        kept = [key for key in self.keys() if key not in excluded]
        return self._copy_entries(kept)._exclude(
            *keys, inplace=False, set_shared=set_shared
        )

    def _maybe_set_shared_attributes(self, result, lock=False):
        # We must use _is_shared to avoid having issues with CUDA tensordicts
        if self._is_shared:
            result._is_shared = True
            if lock:
                result.lock_()
        elif self._is_memmap:
            result._is_memmap = True
            if lock:
                result.lock_()

    def clone(self, recurse: bool = True, **kwargs) -> Self:
        """Returns a copy of the tensordict, of the same type.

        With ``recurse=True``, every tensor leaf is copied. The result is not
        locked, even if the tensordict is: a clone is usually made to be
        modified. To get a :class:`~tensordict.TensorDict` from any other type,
        call :meth:`~.to_tensordict` instead.

        Args:
            recurse (bool, optional): if ``True``, the leaves are copied too.
                If ``False``, only the tree of nested tensordicts is copied:
                the nested tensordicts are new objects, and the leaves are the
                same tensors as in the tensordict. Defaults to ``True``.
            **kwargs: options of the class, for example ``newfile``, the
                destination file of a :class:`~tensordict.PersistentTensorDict`
                cloned with ``recurse=True``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.zeros(3), "b": {"c": torch.ones(3)}}, [3])
            >>> td_clone = td.clone()
            >>> td_clone["a"] is td["a"]
            False
            >>> td_shallow = td.clone(recurse=False)
            >>> td_shallow["a"] is td["a"], td_shallow["b"] is td["b"]
            (True, False)
            >>> td.lock_().clone().is_locked
            False
        """
        result = self._clone(recurse=recurse, **kwargs)
        if not recurse and (result._is_shared or result._is_memmap):
            result.lock_()
        return result

    @abc.abstractmethod
    def _clone(self, recurse: bool = False):
        raise NotImplementedError

    def as_tensor(self) -> Self:
        """Returns a tensordict whose leaves are converted with their ``as_tensor()`` method, if they have one.

        The leaves without an ``as_tensor()`` method, :class:`torch.Tensor`
        and :class:`~tensordict.MemoryMappedTensor` included, are kept as they
        are. No tensordict leaf type defines ``as_tensor()``, so the result
        holds the same leaves as the tensordict. A locked tensordict gives a
        locked result.

        A tensordict with :class:`~tensordict.MemoryMappedTensor` leaves also
        emits a :class:`FutureWarning`. Starting with TensorDict 0.17, the
        default will change to returning those leaves as plain
        :class:`torch.Tensor` over the same storage, with
        ``as_subclass(torch.Tensor)``, without a copy.
        """
        return self._as_tensor(warn=True)

    def _as_tensor(self, *, warn: bool) -> Self:
        has_memmap = False

        def as_tensor(tensor):
            nonlocal has_memmap
            if type(tensor) is torch.Tensor:
                # torch.Tensor has no as_tensor() method.
                return tensor
            if isinstance(tensor, MemoryMappedTensor):
                has_memmap = True
            try:
                return tensor.as_tensor()
            except AttributeError:
                return tensor

        result = self._fast_apply(as_tensor, propagate_lock=True)
        if warn and has_memmap:
            _warn_as_tensor_memmap()
        return result

    _CONFLICTING_BATCH_SIZES = "Conflicting batch sizes in {}: batch_size and auto_batch_size cannot be both specified."

    def empty(
        self, recurse=False, *, batch_size=None, device=NO_DEFAULT, names=None
    ) -> Self:  # noqa: D417
        """Returns a new tensordict of the same type, with no tensor leaves.

        The result has the batch size, device and dimension names of the
        tensordict, unless the keyword arguments change them. It is not locked,
        even if the tensordict is.

        Args:
            recurse (bool, optional): if ``True``, the result has the nested
                tensordicts and the non-tensor data of the tensordict, without
                its tensor leaves. If ``False``, the result has no entries.
                Defaults to ``False``.

        Keyword Args:
            batch_size (torch.Size or list of int, optional): the batch size of
                the result. Defaults to ``None``, which keeps the batch size of
                the tensordict.
                With ``recurse=True``, a nested tensordict whose batch size is the
                start of the new one gets the new one, a nested tensordict whose
                batch size starts with the new one keeps it, and any other
                nested tensordict raises a ``RuntimeError``. A
                :class:`~tensordict.LazyStackedTensorDict` whose members took
                the batch size before 0.15 (one stacked along dim 0, for
                instance) keeps that result, the empty stack of its members,
                with a :class:`FutureWarning`, when a nested tensordict does not
                fit the new batch size or ``names`` has another length.
                Starting with TensorDict 0.17, the default will change to
                raising a ``RuntimeError`` or a ``ValueError``, as
                ``TensorDict.empty`` and the names setter do.
            device (torch.device, str or int, optional): the device of the
                result. ``None`` gives a result with no device. By default, the
                result has the device of the tensordict.
            names (list of str, optional): the names of the batch dimensions
                of the result, one per dimension. Defaults to ``None``, which
                keeps the names of the tensordict for the dimensions that
                remain. Where ``empty()`` keeps names that the names
                setter would reject or drop (a list of another length, or a
                name used twice), it emits a :class:`FutureWarning`. Starting
                with TensorDict 0.17, the default will change to applying them
                as the names setter does, with or without ``recurse``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.zeros(3), "b": {"c": torch.ones(3)}}, [3])
            >>> empty = td.empty()
            >>> list(empty.keys()), empty.batch_size
            ([], torch.Size([3]))
            >>> list(td.empty(recurse=True).keys(include_nested=True))
            ['b']
            >>> td.empty(batch_size=[]).batch_size
            torch.Size([])
        """
        if (
            names is None
            or (
                len(names) == len(self.batch_size if batch_size is None else batch_size)
                and len(set(names)) == len(names)
            )
            or is_compiling()
        ):
            return self._empty(recurse, batch_size, device, names)
        # names that the names setter may reject or drop: the empty() calls
        # that build the result do not warn of them, this one does
        running = getattr(_EMPTY_NAMES_CALL, "running", False)
        _EMPTY_NAMES_CALL.running = True
        try:
            result = self._empty(recurse, batch_size, device, names)
        finally:
            _EMPTY_NAMES_CALL.running = running
        _warn_empty_names(
            names,
            self.batch_size if batch_size is None else batch_size,
            result.names,
            result.batch_size,
        )
        return result

    def _empty(self, recurse, batch_size, device, names) -> TensorDictBase:
        """Builds the result of :meth:`empty`.

        The arguments are those of :meth:`empty`, as given: ``batch_size`` is
        ``None`` or the new batch size, ``device`` is ``NO_DEFAULT`` to keep the
        device, and ``names=None`` keeps the names. Returns a new, unlocked
        tensordict without tensor leaves (with ``recurse=True``, with the
        sub-tensordicts of ``self``, without their tensor leaves; a dense
        tensordict and an h5 file keep their non-tensor entries). A file or a
        store builds it without reading its tensors.
        """
        if not recurse:
            result = self._select(set_shared=False)
        else:
            # simply exclude the leaves
            leaves = _empty_leaves(self)
            # The tensorclass entries become shells that the caller fills:
            # __post_init__ would see no fields, here or when to() rebuilds
            # them. Dynamo cannot set a ContextVar, so compiled code still
            # runs it.
            token = None if is_compiling() else _SKIP_TENSORCLASS_POST_INIT.set(True)
            try:
                result = self._exclude(*leaves, set_shared=False)
                if device is not NO_DEFAULT and device is not None:
                    result = result.to(device)
                    device = NO_DEFAULT
            finally:
                if token is not None:
                    _SKIP_TENSORCLASS_POST_INIT.reset(token)
            if batch_size is not None:
                batch_size = torch.Size(batch_size)
                misfit = _empty_misfit(result, batch_size)
                if misfit is not None:
                    key, value = misfit
                    raise RuntimeError(
                        f"empty(recurse=True) cannot give the batch size "
                        f"{batch_size} to the entry {key!r}, whose batch size "
                        f"{value.batch_size} does not start with it."
                    )
        if batch_size is not None:
            result.batch_size = batch_size
        if device is not NO_DEFAULT:
            if device is None:
                result.clear_device_()
            else:
                result = result.to(device)
        if names is not None:
            result.names = names
        return result

    # Masking
    def masked_fill_(self, mask: Tensor, value: float | bool) -> Self:
        """Writes ``value`` into every leaf where ``mask`` is ``True``, in place.

        The leaves keep their storage, and the call works on a locked
        tensordict. See :ref:`inplace-conventions`.

        Args:
            mask (torch.Tensor): a boolean mask over the leading batch
                dimensions: its shape is the batch size, or its first
                dimensions. It is expanded to the shape of each leaf.
            value (float or bool): the value to write.

        Returns:
            ``self``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(source={'a': torch.zeros(3, 4)},
            ...     batch_size=[3])
            >>> mask = torch.tensor([True, False, False])
            >>> _ = td.masked_fill_(mask, 1.0)
            >>> td.get("a")
            tensor([[1., 1., 1., 1.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.]])
        """
        self._masked_fill_(mask, value)
        return self

    @_write_back_inplace
    def _masked_fill_(self, mask: Tensor, value: float | bool) -> Self:
        """Writes ``value`` into each tensor leaf where ``mask`` is True.

        Called by :meth:`masked_fill_`, which returns ``self`` afterwards.
        ``mask`` is aligned with the leading dims of each leaf
        (:func:`~tensordict.utils.expand_as_right`). The base implementation
        fills the leaves in place and must return ``self``; through
        ``_write_back_inplace`` the classes whose reads return copies run it
        on the copies and write them back. Override it when the storage can
        fill without reading the leaves (a file) or must split the mask over
        its members (a lazy stack).
        """
        for item in self.values():
            mask_expand = expand_as_right(mask, item)
            item.masked_fill_(mask_expand, value)
        return self

    def masked_fill(self, mask: Tensor, value: float | bool) -> Self:
        """Returns a copy of the tensordict with ``value`` written into every leaf where ``mask`` is ``True``.

        The out-of-place version of :meth:`~.masked_fill_`: the tensordict is
        not modified.

        Args:
            mask (torch.Tensor): a boolean mask over the leading batch
                dimensions: its shape is the batch size, or its first
                dimensions. It is expanded to the shape of each leaf.
            value (float or bool): the value to write.

        Returns:
            A new tensordict.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(source={'a': torch.zeros(3, 4)},
            ...     batch_size=[3])
            >>> mask = torch.tensor([True, False, False])
            >>> td1 = td.masked_fill(mask, 1.0)
            >>> td1.get("a")
            tensor([[1., 1., 1., 1.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.]])
        """
        if self._masking_on_dense_copy:
            return self.to_tensordict().masked_fill_(mask, value)
        return self._masked_fill(mask, value)

    def _masked_fill(self, mask: Tensor, value: float | bool) -> Self:
        """Storage-specific part of :meth:`masked_fill`, called when ``_masking_on_dense_copy`` is False.

        Takes the arguments of :meth:`masked_fill` as they are and returns a
        filled copy, leaving ``self`` untouched. The base fills a clone in
        place. :class:`~tensordict.LazyStackedTensorDict` splits ``mask``
        along the stack dim, runs :meth:`masked_fill` on each member and
        stacks the results lazily.
        """
        result = self._clone(recurse=True)
        result._masked_fill_(mask, value)
        return result

    def _change_batch_size(self, new_size: torch.Size) -> None:
        self._batch_size = new_size

    def is_contiguous(self) -> bool:
        """``True`` if all the leaves of the tensordict, nested ones included, are contiguous.

        A lazy stack and a :class:`~tensordict.PersistentTensorDict`, which
        build their leaves when they are read, always return ``False``. See
        :meth:`~.contiguous`.
        """
        if self._leaves_built_on_read:
            return False
        return all([value.is_contiguous() for value in self.values()])

    def contiguous(self, *, canonical: bool = False, inplace: bool = False) -> Self:
        """Returns a new tensordict with contiguous leaves.

        Each leaf is replaced by :meth:`torch.Tensor.contiguous`: a leaf that
        is already contiguous is shared with the tensordict, the others are
        copied.

        Keyword Args:
            canonical (bool, optional): if ``True``, a leaf whose strides are
                not the row-major strides of its shape is copied too, even if
                :meth:`torch.Tensor.is_contiguous` returns ``True`` for it (as
                it can with dimensions of size 1). Defaults to ``False``.
            inplace (bool, optional): if ``True``, the tensordict keeps its
                identity and keys, each leaf that is not contiguous is replaced
                by a contiguous copy, and ``self`` is returned. See
                :ref:`inplace-conventions`. Defaults to ``False``.

        .. note::
            On a :class:`~tensordict.LazyStackedTensorDict`, a
            :class:`~tensordict.PersistentTensorDict` (h5, zarr) or a store,
            the result is a :class:`~tensordict.TensorDict`. If the tensordicts
            of a lazy stack have no names, the result has no names either,
            even when the stack dimension has a name (from ``stack_dim_name``
            or the ``names`` setter). :meth:`~tensordict.LazyStackedTensorDict.densify` and
            :meth:`clone` keep that name. To keep it here, name the dimension
            again with :meth:`refine_names`.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict, lazy_stack
            >>> td = TensorDict(a=torch.zeros(4, 3).T, batch_size=[3])
            >>> td.is_contiguous(), td.contiguous().is_contiguous()
            (False, True)

            On a lazy stack, the result loses the name of the stack dimension:

            >>> stack = lazy_stack(
            ...     [TensorDict(a=torch.zeros(3), batch_size=[3]) for _ in range(2)],
            ...     stack_dim_name="s",
            ... )
            >>> stack.names, stack.contiguous().names
            (['s', None], [None, None])
            >>> stack.contiguous().refine_names("s", ...).names
            ['s', None]
            >>> stack.densify().names
            ['s', None]

        """
        if inplace:
            return self._contiguous_inplace(canonical)
        if self._contiguous_via_to_tensordict:
            result = self.to_tensordict()
            return result.contiguous(canonical=True) if canonical else result
        if canonical:
            source = {
                key: (
                    value.contiguous(canonical=True)
                    if is_tensor_collection(value)
                    else _canonicalize_tensor(value.contiguous())
                )
                for key, value in self.items()
            }
        else:
            source = {key: value.contiguous() for key, value in self.items()}
        return self._new_unsafe(
            source=source,
            batch_size=self.batch_size,
            device=self.device,
            names=self.names if self._has_names() else None,
        )

    def _contiguous_inplace(self, canonical: bool) -> Self:
        """Makes the leaves contiguous in place, for ``contiguous(inplace=True)``.

        :meth:`contiguous` calls it with ``inplace=True`` and its ``canonical``
        argument, and checks nothing before. The hook replaces each leaf with
        its contiguous copy (rematerialized with ``torch.contiguous_format``
        when ``canonical`` is ``True``), recursing into the sub-tensordicts,
        and returns ``self``. The default raises ``NotImplementedError``: only
        the classes that can rebind their leaves override it.
        """
        raise NotImplementedError(
            f"contiguous(inplace=True) is not supported on {type(self).__name__}. "
            "Call .to_tensordict() first or use inplace=False."
        )

    @_as_context_manager()
    @_copy_cached_result
    @_cache_while_locked  # noqa: B019
    def flatten_keys(
        self,
        separator: str = KEY_SEPARATOR,
        inplace: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
    ) -> Self:
        """Returns a tensordict with the nested entries moved to the top level, under joined keys.

        The leaf at ``("a", "b", "c")`` gets the key ``"a.b.c"`` with the
        default separator. The leaves are not copied. The result has the batch
        size, device and dimension names of the tensordict, and the type of
        :meth:`~.empty`: a :class:`~tensordict.TensorDict` for most types, a
        lazy stack for a lazy stack. Two keys that become the same once joined
        raise a ``KeyError``. :meth:`~.unflatten_keys` does the reverse.

        It can be used as a context manager: the changes made to the flat
        tensordict in the ``with`` block are written back to the tensordict
        when the block exits.

        Args:
            separator (str, optional): the string that joins the parts of a
                nested key. Defaults to ``"."``.
            inplace (bool, optional): if ``True``, the entries are moved in
                the tensordict itself, which is returned. Defaults to ``False``.
            is_leaf (Callable, optional): a function that takes a class and
                returns ``True`` if values of that class count as leaves. Here
                it stops the flattening: a nested tensordict whose class is a
                leaf is kept as one value under its key. Defaults to ``None``,
                which uses :func:`~tensordict.is_leaf_nontensor`: tensors and
                non-tensor data are leaves, tensordicts and tensorclasses are
                flattened.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": 1, ("b", "c"): 2, ("e", "f", "g"): 3})
            >>> list(td.flatten_keys().keys())
            ['a', 'b.c', 'e.f.g']
            >>> list(td.flatten_keys(separator="/").keys())
            ['a', 'b/c', 'e/f/g']
            >>> with td.flatten_keys() as flat:
            ...     flat["b.c"] += 1
            >>> td["b", "c"]
            tensor(3)

            With :meth:`~.unflatten_keys`, it converts a state dict to a
            nested tensordict and back:

            >>> model = torch.nn.Sequential(torch.nn.Linear(3, 4))
            >>> params = TensorDict(model.state_dict()).unflatten_keys(".")
            >>> sorted(params.keys(include_nested=True, leaves_only=True))
            [('0', 'bias'), ('0', 'weight')]
            >>> _ = model.load_state_dict(params.flatten_keys(".").to_dict())
        """
        if self._has_exclusive_keys:
            _raise_on_exclusive_keys("flatten_keys")
        if inplace:
            return self._flatten_keys_inplace(separator=separator, is_leaf=is_leaf)
        return self._flatten_keys_outplace(separator=separator, is_leaf=is_leaf)

    def _flatten_keys_outplace(self, separator, is_leaf):
        """Returns a new tensordict with the leaves of ``self`` under flat keys.

        :meth:`flatten_keys` calls it without ``inplace``, after it has checked
        that a lazy stack has no exclusive keys. ``is_leaf`` may be ``None``.
        The hook raises a ``KeyError`` when two keys collide once flattened.
        """
        if is_leaf is None:
            is_leaf = _is_leaf_nontensor
        if is_leaf in _IS_LEAF_WITHOUT_NODES:
            # items() lists the same leaves, faster
            leaves = self.items(include_nested=True, leaves_only=True, is_leaf=is_leaf)
        else:
            leaves = _flatten_keys_leaves(self, is_leaf)
        all_leaves_all_vals = zip(*leaves)
        try:
            all_leaves, all_vals = all_leaves_all_vals
        except ValueError:
            return self.empty()
        all_leaves_flat = [
            key if isinstance(key, str) else separator.join(key) for key in all_leaves
        ]

        if len(set(all_leaves_flat)) < len(all_leaves_flat):
            # find duplicates
            seen = set()
            conflicts = []
            for leaf, leaf_flat in zip(all_leaves, all_leaves_flat):
                if leaf_flat in seen:
                    conflicts.append(leaf)
                else:
                    seen.add(leaf_flat)
            raise KeyError(
                f"Flattening keys in tensordict causes keys {conflicts} to collide."
            )
        result = self.empty()
        _set_dict = getattr(result, "_set_dict", None)
        if _set_dict is not None:
            _set_dict(
                dict(zip(all_leaves_flat, all_vals)),
                validated=True,
            )
        else:
            for val, leaf_flat in zip(all_vals, all_leaves_flat):
                result._set_str(
                    leaf_flat,
                    val,
                    validated=True,
                    inplace=False,
                    non_blocking=False,
                )
        # Uncomment if you want key operations to propagate the shared status
        # self._maybe_set_shared_attributes(result)
        # if result._is_shared or result._is_memmap:
        #     result.lock_()
        return result

    def _flatten_keys_inplace(self, separator, is_leaf):
        if is_leaf is None:
            is_leaf = _is_leaf_nontensor
        if is_leaf in _IS_LEAF_WITHOUT_NODES:
            # keys() lists the same leaves, faster
            leaves = self.keys(include_nested=True, leaves_only=True, is_leaf=is_leaf)
        else:
            leaves = (key for key, _ in _flatten_keys_leaves(self, is_leaf))
        all_leaves = [_unravel_key_to_tuple(key) for key in leaves]
        all_leaves_flat = [separator.join(key) for key in all_leaves]
        if len(set(all_leaves_flat)) < len(set(all_leaves)):
            # find duplicates
            seen = set()
            conflicts = []
            for leaf, leaf_flat in zip(all_leaves, all_leaves_flat):
                if leaf_flat in seen:
                    conflicts.append(leaf)
                else:
                    seen.add(leaf_flat)
            raise KeyError(
                f"Flattening keys in tensordict causes keys {conflicts} to collide."
            )
        self._inplace_flatten_keys_exact = not any(
            separator in part for key in all_leaves for part in key
        )
        # we will need to remove the empty tensordicts later on, but not the
        # leaves that sit at the root
        root_keys = set(self.keys())
        for leaf, leaf_flat in zip(all_leaves, all_leaves_flat):
            self.rename_key_(leaf, leaf_flat)
            if len(leaf) == 1:
                root_keys.discard(leaf[0])
        self.exclude(*root_keys, inplace=True)
        return self

    def _unflatten_keys_fast(self, separator: str) -> bool:
        """Unflattens the keys of a fresh shallow copy in one pass, if it can.

        Returns ``False``, leaving ``self`` unchanged, when :meth:`unflatten_keys`
        must rename the keys one by one. :class:`~tensordict.TensorDict`
        overrides it.
        """
        return False

    @_as_context_manager()
    @_copy_cached_result
    @_cache_while_locked  # noqa: B019
    def unflatten_keys(
        self, separator: str = KEY_SEPARATOR, inplace: bool = False
    ) -> Self:
        """Returns a tensordict with the keys that hold the separator split into nested keys.

        The leaf at ``"a.b.c"`` moves to ``("a", "b", "c")`` with the default
        separator. The leaves are not copied. The new nested tensordicts take
        the batch size, dimension names and device of the tensordict. The
        result has the type of the tensordict for a
        :class:`~tensordict.TensorDict` and a lazy stack. A split key that
        would overwrite an existing entry raises a ``KeyError``.
        :meth:`~.flatten_keys` does the reverse.

        It can be used as a context manager: the changes made to the nested
        tensordict in the ``with`` block are written back to the tensordict
        when the block exits.

        Args:
            separator (str, optional): the string that separates the parts of
                a nested key. Defaults to ``"."``.
            inplace (bool, optional): if ``True``, the entries are moved in
                the tensordict itself, which is returned. Defaults to ``False``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": 1, "b.c": 2, "e.f.g": 3})
            >>> nested = td.unflatten_keys()
            >>> list(nested.keys(include_nested=True, leaves_only=True, sort=True))
            ['a', ('b', 'c'), ('e', 'f', 'g')]
            >>> nested["e", "f", "g"]
            tensor(3)
            >>> td = TensorDict({"a/b": torch.zeros(3)}, batch_size=[3])
            >>> td.unflatten_keys("/")["a"].batch_size
            torch.Size([3])

            With :meth:`~.flatten_keys`, it converts a state dict to a nested
            tensordict and back:

            >>> model = torch.nn.Sequential(torch.nn.Linear(3, 4))
            >>> params = TensorDict(model.state_dict()).unflatten_keys(".")
            >>> sorted(params.keys(include_nested=True, leaves_only=True))
            [('0', 'bias'), ('0', 'weight')]
            >>> _ = model.load_state_dict(params.flatten_keys(".").to_dict())
        """
        if self._has_exclusive_keys:
            _raise_on_exclusive_keys("unflatten_keys")
        if not inplace:
            result = self._clone(recurse=False)
            if not result._unflatten_keys_fast(separator):
                result = result.unflatten_keys(separator=separator, inplace=True)
            if result._is_shared or result._is_memmap:
                result.lock_()
            return result
        else:
            # a comprehension, which also traces under torch.compile
            key_list = [k for k in self.keys()]  # noqa: C416
            for key in key_list:
                if separator in key:
                    new_key = tuple(key.split(separator))
                    try:
                        self.rename_key_(key, new_key, safe=True)
                    except KeyError:
                        raise KeyError(
                            f"Unflattening key(s) in tensordict will override an existing for unflattened key {new_key}."
                        )
            return self

    def split_keys(
        self,
        *key_sets,
        inplace=False,
        default: Any = NO_DEFAULT,
        strict: bool = True,
        reproduce_struct: bool = False,
    ) -> Tuple[T, ...]:
        """Splits the tensordict into one tensordict per set of keys, plus one for the remaining entries.

        Returns ``N + 1`` tensordicts for ``N`` key sets: the last one holds
        the entries that no set took. The values are not copied. Each set
        takes its entries from what the earlier sets left. Empty nested
        tensordicts are removed from the last tensordict, unless
        ``reproduce_struct=True``.

        Args:
            *key_sets (list of NestedKey, dict or Callable): the sets of keys,
                applied in order. A list selects keys, and a dict selects its
                keys and renames each to its value. A callable
                ``fn(key, value) -> bool`` selects every tensor leaf that no
                earlier set took and for which ``fn`` returns ``True``. ``key``
                is the nested key of the leaf as a tuple of str, for example
                ``("observation",)`` or ``("next", "done")``, and ``value`` is
                the leaf as ``get`` returns it. ``fn`` is not called on nested
                tensordicts or non-tensor data: select a whole nested
                tensordict with its key in a list, and non-tensor entries stay
                in the last tensordict unless a list names them. ``fn`` is
                called once per leaf, outside any graph, so it should test
                metadata (key, dtype, shape, device) rather than tensor values.

        Keyword Args:
            inplace (bool, optional): if ``True``, the selected entries are
                removed from the tensordict, which is returned as the last
                tensordict. Defaults to ``False``.
            default (Any, optional): with ``strict=False``, the value put in
                the output for a missing key. By default, a missing key is
                skipped. A ``default`` of ``None`` also skips the key. Applies to the keys in lists and dicts only: a
                callable never selects a missing key.
            strict (bool, optional): if ``True``, a missing key raises a
                ``KeyError``, and ``default`` is ignored. Applies to the keys in
                lists and dicts only. Defaults to ``True``.
            reproduce_struct (bool, optional): if ``True``, every returned
                tensordict has the tree of nested tensordicts of the
                tensordict, even where a nested tensordict holds no leaves.
                Non-tensor entries are not copied: each one goes to the output
                whose key set names it, or to the last one, as with
                ``reproduce_struct=False``. Defaults to ``False``.

        Returns:
            A tuple of ``N + 1`` tensordicts.

        .. note:: Duplicate keys in the sets are not checked.

        Examples:
            >>> from tensordict import TensorDict
            >>> td = TensorDict(
            ...     a=0,
            ...     b=0,
            ...     c=0,
            ...     d=0,
            ... )
            >>> td_a, td_bc, td_d = td.split_keys(["a"], ["b", "c"])
            >>> sorted(td_a.keys()), sorted(td_bc.keys()), sorted(td_d.keys())
            (['a'], ['b', 'c'], ['d'])
            >>> # with inplace=True, the last tensordict is td itself
            >>> td_a, rest = td.split_keys(["a"], inplace=True)
            >>> rest is td, sorted(td.keys())
            (True, ['b', 'c', 'd'])
            >>> # a callable selects leaves by key and value; the first match wins
            >>> import torch
            >>> td = TensorDict(
            ...     s=torch.zeros(10),
            ...     v=torch.zeros(10, 3),
            ...     m=torch.zeros(10, 4, 4),
            ...     nested=TensorDict(t=torch.zeros(10, 3), batch_size=[10]),
            ...     batch_size=[10],
            ... )
            >>> nb = td.batch_dims
            >>> scalars, vectors, rest = td.split_keys(
            ...     lambda k, v: v.ndim == nb, lambda k, v: v.ndim == nb + 1
            ... )
            >>> sorted(scalars.keys(True, True), key=str)
            ['s']
            >>> sorted(vectors.keys(True, True), key=str)
            [('nested', 't'), 'v']
            >>> sorted(rest.keys(True, True), key=str)
            ['m']
            >>> # lists and callables mix: the nested key goes first
            >>> nested, vectors, rest = td.split_keys(
            ...     ["nested"], lambda k, v: v.ndim == nb + 1
            ... )
            >>> sorted(vectors.keys())
            ['v']
        """
        from tensordict import PersistentTensorDict

        if isinstance(self, PersistentTensorDict):
            last_out = self.to_tensordict()
        else:
            last_out = self.copy()
        if strict:
            default = NO_DEFAULT
        elif default is NO_DEFAULT:
            default = None
        outs = []
        if inplace:
            keys_to_del = set()
        leaf_keys = None
        warn = True
        dict_sets = []
        for key_set in key_sets:
            out = self.empty(recurse=reproduce_struct)
            if reproduce_struct:
                # empty(recurse=True) keeps the non-tensor leaves, and in a
                # lazy stack inside a tensordict the tensors that only some
                # members hold: they belong to the output whose key set names
                # them, or to the last one. Every output has the same ones:
                # find them once, in every member of a lazy stack (#2803).
                if leaf_keys is None:
                    leaf_keys = _all_leaf_keys(out)
                if leaf_keys:
                    out = out.exclude(*leaf_keys)
            outs.append(out)
            # A key set that is both callable and iterable (an Enum class, for
            # example) is iterated, as it was before callables were accepted.
            if callable(key_set) and not isinstance(key_set, Iterable):
                key_set = [
                    key
                    for key, val in last_out.items(True, True)
                    if key_set((key,) if isinstance(key, str) else key, val)
                ]
            if not isinstance(key_set, dict):
                key_set = {key: key for key in key_set}
            # a key set may be an iterator, read once
            dict_sets.append(key_set)
            for key in key_set:
                try:
                    val = last_out.pop(key, default)
                except KeyError:
                    if not _has_lazy_stack(self):
                        raise
                    key_sets = [
                        {
                            _unravel_key_to_tuple(k): _unravel_key_to_tuple(
                                ks[k] if isinstance(ks, dict) else k
                            )
                            for k in ks
                        }
                        for ks in (*dict_sets, *key_sets[len(dict_sets) :])
                    ]
                    for k in (k for ks in key_sets for k in ks):
                        if not _held_by_member(self, k):
                            # no member holds k: the KeyError of pop
                            raise
                    return tuple(
                        _split_keys_by_member(self, key_sets, inplace, reproduce_struct)
                    )
                if (
                    val is default
                    and not strict
                    and warn
                    and _lazy_on_path(self, _unravel_key_to_tuple(key))
                    and _held_by_member(self, _unravel_key_to_tuple(key))
                ):
                    # once per call, for a key that only some members of a
                    # lazy stack hold (#2827)
                    warn = False
                    _warn_split_keys_kept_in_last()
                if val is not None:
                    outs[-1].set(key_set[key], val)
                if inplace:
                    keys_to_del.add(key)
        if inplace:
            # We update self here because doing it in the loop would
            #  possibly break people's code when doing a try/except KeyError
            #  around this method
            for key in keys_to_del:
                try:
                    self.pop(key, default=default)
                except KeyError:
                    # We're good if strict is False
                    if strict:
                        raise
            last_out = self
        if not reproduce_struct:
            last_out.filter_empty_()
        outs.append(last_out)
        return tuple(outs)

    def separates(
        self,
        *keys: NestedKey,
        default: Any = NO_DEFAULT,
        strict: bool = True,
        filter_empty: bool = True,
    ) -> Self:
        """Removes the entries of ``keys`` from the tensordict in place, and returns them in a new tensordict.

        The values are not copied. The result keeps the nested tensordicts
        that hold the removed entries, with the batch size of the tensordict.
        This is :meth:`~.split_keys` with ``inplace=True`` and one key set,
        except that only the separated part is returned: the tensordict holds
        the rest.

        Args:
            *keys (NestedKey): the keys to remove.

        Keyword Args:
            default (Any, optional): if given, a missing key does not raise.
                The value of ``default`` is not used: the result holds a
                ``None`` entry under the missing key.
            strict (bool, optional): if ``True``, a missing key raises a
                ``KeyError``. If ``False``, the result holds a ``None`` entry
                under it. Defaults to ``True``.
            filter_empty (bool, optional): if ``True``, the nested tensordicts
                of the tensordict that the removal leaves empty are removed
                too. Defaults to ``True``.

        Returns:
            A new tensordict with the removed entries.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(
            ...     a=torch.zeros(3),
            ...     b=torch.zeros(3),
            ...     nested=TensorDict(c=torch.zeros(3), batch_size=[3]),
            ...     batch_size=[3],
            ... )
            >>> out = td.separates("a", ("nested", "c"))
            >>> sorted(out.keys(include_nested=True, leaves_only=True), key=str)
            [('nested', 'c'), 'a']
            >>> list(td.keys())
            ['b']
        """
        from tensordict import PersistentTensorDict

        if isinstance(self, PersistentTensorDict):
            last_out = self.to_tensordict()
        else:
            last_out = self
        strict = strict and default is NO_DEFAULT
        if strict:
            default = NO_DEFAULT
        else:
            default = None
        key_set = keys

        # We want to keep the metadata such as batch-size etc so we call with recurse=True
        out = self.empty(recurse=True)
        for key in key_set:
            val = last_out.pop(key, default)
            out.set(key, val)
        out.filter_empty_()
        if filter_empty:
            self.filter_empty_()
        return out

    def _index_tensordict(
        self,
        index: IndexType,
        new_batch_size: torch.Size | None = None,
        names: List[str] | None = None,
    ) -> Self:
        TensorDict = _TENSORDICT_TYPE
        batch_size = self.batch_size
        batch_dims = len(batch_size)

        def _check_for_invalid_index(index):
            if batch_size:
                return
            if index is None or isinstance(index, (bool, np.bool_)):
                return
            if (
                isinstance(index, torch.Tensor)
                and index.dtype == torch.bool
                and not index.ndim
            ):
                return
            if isinstance(index, tuple):
                if len(index) == 1:
                    return _check_for_invalid_index(index[0])
                # None and the scalar bools use no dim
                elif all(_read_element(idx)[1] == 0 for idx in index):
                    return
            raise RuntimeError(
                f"indexing a tensordict with td.batch_dims==0 is not permitted. Got index {index}."
            )

        _check_for_invalid_index(index)

        if new_batch_size is not None:
            batch_size = new_batch_size
        else:
            if (
                type(index) is tuple
                and len(index) == 1
                and type(index[0]) is Tensor
                and index[0].dtype is torch.bool
                and not is_compiling()
            ):
                index = _mask_to_indices(index, batch_size, self.device)
            batch_size = _getitem_batch_size(batch_size, index)

        if names is None:
            names = self._get_names_idx(index)

        source = {}
        for key, item in self.items():
            if _is_unbatched(item):
                source[key] = _unbatched_with_batch_size(item, batch_size)
            elif isinstance(item, TensorDict):
                # this is the simplest case, we can pre-compute the batch size easily
                new_batch_size = batch_size + item.batch_size[batch_dims:]
                source[key] = item._index_tensordict(
                    index, new_batch_size=new_batch_size
                )
            else:
                source[key] = _get_item(item, index)
        result = self._new_unsafe(
            source=source,
            batch_size=batch_size,
            device=self.device,
            names=names,
            # lock=self.is_locked,
        )
        if self._is_memmap and _index_preserve_data_ptr(index):
            result._is_memmap = True
            result.lock_()
        elif self._is_shared and _index_preserve_data_ptr(index):
            result._is_shared = True
            result.lock_()
        return result

    # Locking functionality
    @property
    def is_locked(self) -> bool:
        """``True`` if the tensordict is locked.

        A locked tensordict raises ``RuntimeError`` on any operation that adds,
        removes or renames an entry, or that replaces a leaf with a new tensor.
        Writes into the existing leaves still work. See :meth:`~.lock_`.

        Setting ``td.is_locked = True`` calls :meth:`~.lock_`, and
        ``td.is_locked = False`` calls :meth:`~.unlock_`.

        Examples:
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": 0})
            >>> td.is_locked
            False
            >>> td.is_locked = True
            >>> td.is_locked
            True
        """
        return self._is_locked

    @is_locked.setter
    def is_locked(self, value: bool) -> None:
        if value:
            self.lock_()
        else:
            self.unlock_()

    def _propagate_lock(self, lock_parents_weakrefs=None, *, is_compiling):
        """Locks this tensordict and the tensor collections it holds, and registers the parents that hold the lock.

        ``lock_()`` calls it on an unlocked root, with no
        ``lock_parents_weakrefs``; each node calls it on its sub-tensordicts
        with the weakrefs of the parents above them. The hook sets
        ``_is_locked`` and, out of compile, adds the parents it does not hold
        yet to ``_lock_parents_weakrefs`` (which ``unlock_`` checks) and builds
        the locked schema. Returns ``None``.
        """
        self._is_locked = True
        if lock_parents_weakrefs is not None:
            existing = self._lock_parents_weakrefs
            # Most tensordicts have no lock parent yet: skip the filter then.
            if existing:
                lock_parents_weakrefs = [
                    ref
                    for ref in lock_parents_weakrefs
                    if not any(refref is ref for refref in existing)
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
            # Build a per-TD locked schema so locked-fast-paths can walk
            # an immutable key tuple instead of iterating ``_tensordict``
            # under Dynamo (which emits DICT_KEYS_MATCH guards).
            # Skip under compile: building the schema would itself iterate
            # the backing dict and add the very guard we're trying to
            # avoid. Users who want the fast path must lock_() in eager
            # mode before entering the compiled region.
            self._build_lock_schema()

        for value in self._lock_sub_tensordicts():
            value._propagate_lock(lock_parents_weakrefs, is_compiling=is_compiling)

    def _lock_sub_tensordicts(self):
        """The sub-tensordicts that lock_(), unlock_() and the default _rename_subtds reach from this one."""
        return [value for value in self.values() if _is_tensor_collection(type(value))]

    def _build_lock_schema(self) -> None:
        """Build the immutable locked schema for this TD.

        Default no-op. Overridden on :class:`TensorDict` (and any other
        leaf type that benefits from the locked-fast-path).
        """
        return

    @property
    def _lock_parents_weakrefs(self):
        _lock_parents_weakrefs = self.__dict__.get("__lock_parents_weakrefs")
        if _lock_parents_weakrefs is None:
            self.__dict__["__lock_parents_weakrefs"] = []
            _lock_parents_weakrefs = self.__dict__["__lock_parents_weakrefs"]
        return _lock_parents_weakrefs

    @_lock_parents_weakrefs.setter
    def _lock_parents_weakrefs(self, value: list):
        self.__dict__["__lock_parents_weakrefs"] = value

    @_as_context_manager("is_locked")
    def lock_(self) -> Self:
        """Locks the tensordict and its nested tensordicts, and returns it.

        A locked tensordict raises ``RuntimeError`` on any operation that adds,
        removes or renames an entry, or that replaces a leaf with a new tensor:
        :meth:`~.set`, ``__setitem__``, :meth:`~.update`, :meth:`~.pop`,
        :meth:`~.rename_key_`, ``del``. Writes into the existing leaves still
        work: :meth:`~.set_`, :meth:`~.update_`, and ``set(..., inplace=True)``.
        Check the state with :attr:`~.is_locked`. See :ref:`keys-locking`.

        Can be used as a context manager: the tensordict is unlocked when the
        ``with`` block exits, also if the block raises an exception.

        Examples:
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": 1, "b": 2, "c": 3}, batch_size=[])
            >>> with td.lock_():
            ...     assert td.is_locked
            ...     try:
            ...         td.set("d", 0)  # adds a key: error
            ...     except RuntimeError:
            ...         print("td is locked!")
            ...     try:
            ...         del td["a"]  # removes a key: error
            ...     except RuntimeError:
            ...         print("td is locked!")
            ...     try:
            ...         td.rename_key_("a", "d")  # renames a key: error
            ...     except RuntimeError:
            ...         print("td is locked!")
            ...     # These calls write into the existing leaves, so they work:
            ...     _ = td.set("a", 0, inplace=True)
            ...     _ = td.set_("a", 0)
            ...     _ = td.update({"a": 0}, inplace=True)
            ...     _ = td.update_({"a": 0})
            td is locked!
            td is locked!
            td is locked!
            >>> td.is_locked
            False
        """
        if self.is_locked:
            return self
        is_comp = is_compiling()
        if is_comp:
            _lock_warn()
        self._propagate_lock(is_compiling=is_comp)
        return self

    @_erase_cache_first
    def _propagate_unlock(self):
        # if we end up here, we can clear the graph associated with this td
        self._is_locked = False

        self._is_shared = False
        self._is_memmap = False
        # Drop the locked schema; it would go stale once keys can be added
        # or removed again.
        if "_locked_schema" in self.__dict__:
            self.__dict__["_locked_schema"] = None

        # Remove consolidated metadata when unlocking to prevent silent errors
        if hasattr(self, "_consolidated"):
            delattr(self, "_consolidated")

        sub_tds = []
        for value in self._lock_sub_tensordicts():
            sub_tds.extend(value._propagate_unlock())
            sub_tds.append(value)
        return sub_tds

    def _check_unlock(self, first_attempt=True):
        if not first_attempt:
            gc.collect()
        obj = None
        for ref in self._lock_parents_weakrefs:
            obj = ref()
            # check if the locked parent exists and if it's locked
            # we check _is_locked because it can be False or None in the case of Lazy stacks,
            # but if we check obj.is_locked it will be True for this class.
            if obj is not None and obj._is_locked:
                break

        else:
            try:
                self._lock_parents_weakrefs = []
            except AttributeError:
                # Some tds (eg, LazyStack) have an automated way of creating the _lock_parents_weakref
                pass
            return

        if first_attempt:
            del obj
            return self._check_unlock(False)
        raise RuntimeError(
            "Cannot unlock a tensordict that is part of a locked graph. "
            "Unlock the root tensordict first. If the tensordict is part of multiple graphs, "
            "group the graphs under a common tensordict an unlock this root. "
            f"self: {self}, obj: {obj}"
        )

    @_as_context_manager("is_locked")
    def unlock_(self) -> Self:
        """Unlocks the tensordict and its nested tensordicts, and returns it.

        After this call, entries can be added, removed and renamed again. See
        :meth:`~.lock_` for what a lock blocks.

        Can be used as a context manager: the tensordict is locked again when
        the ``with`` block exits, also if the block raises an exception.

        Raises:
            RuntimeError: if the tensordict, or one of its nested tensordicts,
                is also an entry of another locked tensordict. Unlock that
                parent first. The tensordict stays locked.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3), batch_size=[3], lock=True)
            >>> with td.unlock_():
            ...     td["b"] = torch.ones(3)
            >>> td.is_locked
            True
            >>> sorted(td.keys())
            ['a', 'b']
        """
        # Raise before _propagate_unlock clears the storage flags of the nodes.
        if self._is_locked:
            self._check_unlock()
        try:
            sub_tds = self._propagate_unlock()
            for sub_td in sub_tds:
                sub_td._check_unlock()

            self._check_unlock()
        except RuntimeError as err:
            self.lock_()
            raise err
        return self

    def attrs(
        self,
        *,
        fields: Sequence[str] = ("device", "dtype", "shape"),
        num_threads: int | None = None,
    ) -> Self:
        """Returns a tensordict that describes each tensor leaf with a :class:`~tensordict.TensorAttrs`.

        Each tensor leaf is replaced by a :class:`~tensordict.TensorAttrs` that
        records its device, dtype and shape. The result has an empty batch size
        and no device. Pass it to :meth:`~.to` to cast another tensordict leaf
        by leaf, to the device and dtype of the matching leaf, when the leaves
        do not all share one device or dtype.

        Keyword Args:
            fields (sequence of str, optional): the attributes to record, any
                subset of ``("device", "dtype", "shape")``. The others are
                ``None``. Defaults to ``("device", "dtype", "shape")``.
            num_threads (int, optional): the number of threads that build the
                result. Defaults to ``None``, which uses the calling thread
                only.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(
            ...     {"a": torch.zeros(3, device="cpu"),
            ...      "b": torch.zeros(3, dtype=torch.int32)},
            ...     batch_size=[3],
            ... )
            >>> attrs = td.attrs(fields=("device", "dtype"))
            >>> target = TensorDict(a=torch.zeros(3, device="cpu"), b=torch.zeros(3), batch_size=[3])
            >>> out = target.to(attrs)  # casts "b" to int32
            >>> out["b"].dtype
            torch.int32
        """
        from tensordict.tensorclass import TensorAttrs

        def _to_attrs(t):
            return TensorAttrs.from_tensor(t, fields=fields)

        return self._fast_apply(
            _to_attrs,
            batch_size=(),
            device=None,
            propagate_lock=False,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            num_threads=num_threads if num_threads is not None else 0,
        )

    def _to_per_leaf(
        self,
        attrs_td: TensorDictBase,
        *,
        non_blocking: bool | None = None,
        non_blocking_pin: bool = False,
        num_threads: int | None = None,
        inplace: bool = False,
    ) -> Self:
        """Cast each leaf of ``self`` to the attributes recorded in ``attrs_td``.

        Leaves absent from ``attrs_td`` (or whose attrs have both ``tgt_device=None``
        and ``tgt_dtype=None``) are passed through unchanged.

        When the caller does not pass ``non_blocking`` explicitly, per-leaf copies
        are issued asynchronously. A single :meth:`_sync_all` is invoked at the end
        only if at least one leaf went D2H (source on a CUDA/XPU/etc. device, target
        on CPU). Async H2D copies do not need an explicit sync — subsequent kernels
        on the destination device serialize on the same CUDA stream that queued the
        copy, so the dependency is already honored. Only the D2H direction needs the
        barrier because host reads are not coordinated by the device's stream scheduler.
        """
        from tensordict.tensorclass import TensorAttrs

        if non_blocking_pin:
            raise NotImplementedError(
                "non_blocking_pin is not yet supported when an attrs tensordict is passed to `to()`."
            )

        if non_blocking is None:
            sub_non_blocking = True
            do_sync = True
        else:
            sub_non_blocking = non_blocking
            do_sync = not non_blocking

        def _is_attrs_leaf(cls):
            return issubclass(cls, TensorAttrs) or _default_is_leaf(cls)

        # _flatten_keys_leaves keeps each TensorAttrs whole; items() would also
        # search it for entries
        spec: dict = {}
        for key, val in _flatten_keys_leaves(attrs_td, _is_attrs_leaf):
            if isinstance(val, TensorAttrs):
                spec[unravel_key(key)] = val

        # D2H (device-to-host) is the only transfer direction that needs an explicit
        # sync after an async copy: host memory is outside the source device's stream
        # scheduler, so reads after the copy call returns may observe stale data. H2D
        # and cross-device D2D are fine — the destination's stream already serializes
        # on the enqueued copy.
        needs_d2h_sync = False

        def _cast(name, tensor):
            attrs = spec.get(name)
            if attrs is None:
                return tensor
            target_device = attrs.tgt_device
            target_dtype = attrs.tgt_dtype
            if target_device is None and target_dtype is None:
                return tensor
            if (
                target_device is not None
                and target_device.type == "cpu"
                and tensor.device.type != "cpu"
            ):
                nonlocal needs_d2h_sync
                needs_d2h_sync = True
            return tensor.to(
                device=target_device,
                dtype=target_dtype,
                non_blocking=sub_non_blocking,
            )

        result = self._fast_apply(
            _cast,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            out=self if inplace else None,
            checked=True,
            device=None,
            num_threads=num_threads if num_threads is not None else 0,
        )

        if needs_d2h_sync and do_sync:
            self._sync_all()

        return result

    @property
    def _has_cuda(self):
        val = self.__dict__.get("_has_cuda_val")
        if val is None:
            # Cache this value
            val = torch.cuda.is_available()
            self.__dict__["_has_cuda_val"] = val
        return val

    @property
    def _has_mps(self):
        val = self.__dict__.get("_has_mps_val")
        if val is None:
            # Cache this value
            val = torch.backends.mps.is_available()
            self.__dict__["_has_mps_val"] = val
        return val

    def is_floating_point(self) -> bool:
        """``True`` if every tensor leaf of the tensordict, nested ones included, has a floating-point dtype.

        Non-tensor data is not checked. An empty tensordict returns ``True``.
        """
        for item in self.values(include_nested=True, leaves_only=True):
            if not item.is_floating_point():
                return False
        else:
            return True

    # Gradient compatibility
    @property
    def requires_grad(self) -> bool:
        """``True`` if any leaf of the tensordict, nested ones included, requires gradients."""
        return any(v.requires_grad for v in self.values())

    def requires_grad_(self, requires_grad=True) -> Self:
        """Sets ``requires_grad`` on every tensor leaf in place, and returns the tensordict.

        Args:
            requires_grad (bool, optional): whether autograd records the
                operations on the leaves. Defaults to ``True``.

        Returns:
            ``self``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3), b=TensorDict(c=torch.zeros(3)))
            >>> td.requires_grad
            False
            >>> td.requires_grad_()["b", "c"].requires_grad
            True
            >>> td.requires_grad
            True
        """
        for val in self._values_list(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS):
            val.requires_grad_(requires_grad)
        return self

    def backward(
        self,
        gradient: TensorDictBase | None = None,
        retain_graph: bool | None = None,
        create_graph: bool = False,
        inputs: TensorDictBase | Sequence[torch.Tensor] | None = None,
    ) -> None:
        """Computes the gradients of the leaves of the tensordict that require gradients.

        Like :meth:`torch.Tensor.backward` for every leaf at once: the leaves
        that require gradients, nested ones included, are differentiated in a
        single :func:`torch.autograd.backward` call, and the others are
        ignored. The gradients are accumulated in the ``.grad`` of the graph
        leaves, as usual. Returns ``None``.

        Without ``gradient``, every leaf that requires gradients must have a
        single element, and its gradient is ``1``: ``td.backward()`` is then
        the same as ``td.sum(reduce=True).backward()``.

        Args:
            gradient (TensorDictBase, optional): the gradient of each leaf, under
                the same nested key and with the same shape. The leaves can
                differ in shape, dtype and device. Defaults to ``None``, which
                needs single-element leaves.
            retain_graph (bool, optional): if ``False``, the graph used to
                compute the gradients is freed. Defaults to ``None``, which
                takes the value of ``create_graph``.
            create_graph (bool, optional): if ``True``, the graph of the
                derivative is built, for higher-order derivatives. Defaults to
                ``False``.
            inputs (TensorDictBase or sequence of torch.Tensor, optional): the
                tensors whose ``.grad`` receives the gradients; the others are
                ignored. Defaults to ``None``, which accumulates into all the
                graph leaves used to compute the leaves of the tensordict.

        Raises:
            RuntimeError: if no leaf requires gradients, if ``gradient`` is not
                given and a leaf has more than one element, or if a gradient
                has the wrong shape.
            KeyError: if ``gradient`` has no entry for a leaf that requires
                gradients.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> x = torch.ones(3, requires_grad=True)
            >>> losses = TensorDict(actor_loss=x.sum(), critic_loss=x.pow(2).sum())
            >>> losses.backward()  # a gradient of 1 for each single-element leaf
            >>> x.grad
            tensor([3., 3., 3.])

            Weight the losses with a gradient tensordict:

            >>> x.grad = None
            >>> losses = TensorDict(actor_loss=x.sum(), critic_loss=x.pow(2).sum())
            >>> weights = TensorDict(actor_loss=torch.tensor(0.5), critic_loss=torch.tensor(1.0))
            >>> losses.backward(weights)
            >>> x.grad
            tensor([2.5000, 2.5000, 2.5000])
        """
        keys = []
        tensors = []
        for key, value in self.items(True, True, is_leaf=_is_leaf_nontensor):
            if isinstance(value, torch.Tensor) and value.requires_grad:
                keys.append(key)
                tensors.append(value)
        if not tensors:
            raise RuntimeError(
                "backward() cannot be called on a tensordict that has no leaf tensor "
                "requiring gradients."
            )
        if gradient is None:
            grad_tensors = None
            for key, tensor in zip(keys, tensors):
                if tensor.numel() != 1:
                    raise RuntimeError(
                        "grad can be implicitly created only for scalar outputs: the "
                        f"leaf at key {key!r} has shape {tuple(tensor.shape)}. Pass a "
                        "gradient tensordict matching the structure of this tensordict "
                        "to backward()."
                    )
        else:
            if not isinstance(gradient, TensorDictBase):
                raise TypeError(
                    "gradient must be a TensorDictBase instance with entries matching "
                    "the differentiable leaves of the tensordict, got "
                    f"{type(gradient).__name__} instead."
                )
            grad_tensors = []
            for key, tensor in zip(keys, tensors):
                grad = gradient.get(key, default=None)
                if grad is None:
                    raise KeyError(
                        f"Missing gradient entry for key {key!r} in the gradient "
                        "tensordict passed to backward()."
                    )
                if grad.shape != tensor.shape:
                    raise RuntimeError(
                        f"Mismatch in shape: the gradient at key {key!r} has shape "
                        f"{tuple(grad.shape)} but the corresponding leaf has shape "
                        f"{tuple(tensor.shape)}."
                    )
                grad_tensors.append(grad)
        if isinstance(inputs, TensorDictBase):
            inputs = tuple(
                value
                for value in inputs.values(True, True, is_leaf=_is_leaf_nontensor)
                if isinstance(value, torch.Tensor)
            )
        torch.autograd.backward(
            tensors,
            grad_tensors=grad_tensors,
            retain_graph=retain_graph,
            create_graph=create_graph,
            inputs=inputs,
        )

    def detach_(self) -> Self:
        """Detaches every tensor leaf from the autograd graph in place, and returns the tensordict.

        Each leaf is detached with :meth:`torch.Tensor.detach_`: it keeps its
        storage and no longer requires gradients.

        Returns:
            ``self``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3, requires_grad=True) + 1)
            >>> a = td["a"]
            >>> _ = td.detach_()
            >>> a.requires_grad, td["a"] is a
            (False, True)
        """
        if not self._storage_holds_no_gradients:
            self._detach_()
        return self

    def _detach_(self) -> None:
        """Detaches the tensor leaves in place.

        Called by :meth:`detach_`, which returns ``self`` afterwards, unless
        ``_storage_holds_no_gradients`` is set. The base implementation calls
        ``detach_`` on each value, and replaces each
        :class:`~tensordict.UnbatchedTensor` with its ``detach()``, which
        shares its storage. Override it when the values are not the stored
        tensors (a lazy stack detaches each member) or when the leaves cannot
        be detached in place (a sub-tensordict raises).
        """
        compiling = is_compiling()
        for key, value in self.items():
            if _is_unbatched(value):
                # A tensordict stores an UnbatchedTensor as an alias that
                # keeps its batch size, and torch cannot detach a view in place.
                self._set_str(
                    key, value.detach(), inplace=False, validated=True, ignore_lock=True
                )
            elif (
                compiling
                and isinstance(value, torch.Tensor)
                and not value.requires_grad
                and value._base is None
            ):
                # detach_() does nothing on such a leaf, and Dynamo cannot
                # trace an in-place view op on a graph input (#2792).
                continue
            else:
                value.detach_()

    @_cache_while_locked  # noqa: B019
    def detach(self) -> Self:
        """Returns a new tensordict with every tensor leaf detached from the autograd graph.

        The leaves are :meth:`torch.Tensor.detach` views: they share their
        storage with the leaves of the tensordict, and do not require
        gradients. A locked tensordict gives a locked result.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3, requires_grad=True) + 1)
            >>> out = td.detach()
            >>> out["a"].requires_grad, td["a"].requires_grad
            (False, True)
            >>> out["a"].data_ptr() == td["a"].data_ptr()
            True
        """

        def detach(x):
            return x.detach()

        return self._fast_apply(
            detach,
            propagate_lock=True,
        )


# The mixin modules import TensorDictBase for type checking only, so their
# string annotations that name it (e.g. ``other: TensorDictBase | torch.Tensor``)
# cannot be resolved at run time. Binding the class in each module lets
# typing.get_type_hints and inspect.signature(eval_str=True) resolve them.
for _mixin in _TENSORDICTBASE_MIXINS:
    sys.modules[_mixin.__module__].TensorDictBase = TensorDictBase
del _mixin

_ACCEPTED_CLASSES = (
    Tensor,
    TensorDictBase,
)

# the frame of update_, which _raise_if_inplace_call looks for too
_UPDATE_INPLACE_CODE = TensorDictBase._update_.__code__
# the frame of the plan of update_ for a member of a lazy stack (#2776)
_UPDATE_PLAN_CODE = TensorDictBase._update_plan.__code__


from tensordict._base.factories import (  # noqa: F401
    from_any,
    from_csv,
    from_dict,
    from_h5,
    from_json,
    from_namedtuple,
    from_pandas,
    from_parquet,
    from_struct_array,
    from_tuple,
    from_zarr,
)

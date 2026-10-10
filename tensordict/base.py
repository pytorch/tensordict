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
from collections.abc import MutableMapping
from concurrent.futures import Future, ThreadPoolExecutor, wait
from functools import wraps
from pathlib import Path
from textwrap import indent
from threading import local
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
from warnings import warn

import numpy as np
import torch
from tensordict._contextlib import LAST_OP_MAPS
from tensordict._deprecation import deprecated, warn_deprecated
from tensordict._indexing import (
    _entry_index,
    _getitem_batch_size,
    _getitem_names,
    _is_new_dim_index,
    _read_element,
    convert_ellipsis_to_idx,
)
from tensordict._nestedkey import NestedKey
from tensordict._tensorcollection import TensorCollection
from tensordict.memmap import MemoryMappedTensor
from tensordict.utils import (
    _add_batch_dim_pre_hook,
    _as_context_manager,
    _cache_while_locked,
    _CloudpickleWrapper,
    _convert_list_to_stack,
    _erase_cache_first,
    _GENERIC_NESTED_ERR,
    _get_item,
    _index_preserve_data_ptr,
    _is_non_tensor,
    _is_tensorclass,
    _is_unbatched,
    _KEY_ERROR,
    _lock_blocked,
    _LOCK_ERROR,
    _lock_warn,
    _maybe_correct_neg_dim,
    _pass_through,
    _pass_through_cls,
    _proc_init,
    _prune_selected_keys,
    _REPR_OPTIONS,
    _set_max_batch_size,
    _shape,
    _split_tensordict,
    _strtobool,
    _td_fields,
    _unravel_key_to_tuple,
    _zip_strict,
    capture_non_tensor_stack,
    DeviceType,
    expand_as_right,
    ftdim,
    IndexType,
    is_batchedtensor,
    is_non_tensor,
    is_tensorclass,
    list_to_stack,
    set_capture_non_tensor_stack,
    unravel_key,
    unravel_key_list,
)
from torch import _foreach_copy_, multiprocessing as mp, nn, Tensor
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

_GET_DEFAULTS_TO_NONE_REPLACEMENT = "td[key] to raise a KeyError for a missing key"

if "TD_GET_DEFAULTS_TO_NONE" in os.environ:
    _GET_DEFAULTS_TO_NONE = _strtobool(os.environ["TD_GET_DEFAULTS_TO_NONE"])
else:
    _GET_DEFAULTS_TO_NONE = True

if not _GET_DEFAULTS_TO_NONE:
    warn_deprecated(
        f"TD_GET_DEFAULTS_TO_NONE={os.environ['TD_GET_DEFAULTS_TO_NONE']}",
        removal="0.17",
        replacement=_GET_DEFAULTS_TO_NONE_REPLACEMENT,
        stacklevel=1,
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
    """Sets the default of `get` to `None` and silences deprecation warnings during calls to `get` that result in a `KeyError`.

    This can also be controlled via the environment variable ``TD_GET_DEFAULTS_TO_NONE``.

    .. deprecated:: 0.15
        Since v0.7, :meth:`~tensordict.TensorDictBase.get` returns ``None`` for a
        missing key by default. This function, and setting ``TD_GET_DEFAULTS_TO_NONE``
        to a false value, are deprecated and will be removed in TensorDict 0.17.
        Use ``td[key]`` to raise a ``KeyError`` for a missing key.

    Args:
        set_to_none (bool): whether the default of `get` should be `None`, or should `get` raise a `KeyError` if
            no `default` is passed and the key is absent from the `TensorDict`.
            Defaults to `True`.

    """
    _set_get_defaults_to_none(set_to_none)


@deprecated(
    "get_defaults_to_none()",
    removal="0.17",
    replacement=_GET_DEFAULTS_TO_NONE_REPLACEMENT,
)
def get_defaults_to_none(set_to_none: bool = True):
    """Returns the status of `get` default value.

    .. deprecated:: 0.15
        Since v0.7, :meth:`~tensordict.TensorDictBase.get` returns ``None`` for a
        missing key by default. This function is deprecated and will be removed in
        TensorDict 0.17. Use ``td[key]`` to raise a ``KeyError`` for a missing key.
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


def _maybe_broadcast_other(op: str, n_other: int = 1) -> Callable[[Callable], Callable]:
    """Ensures that elementwise ops are broadcast when an nd tensor is passed."""

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
                return func(self, *others, *args, **kwargs)
            others_map = []
            shape = self.shape
            self_expand = self
            shapes = [shape, *[other.shape for other in others if other is not None]]
            shape = torch.broadcast_shapes(*shapes)
            if shape != self_expand.shape:
                self_expand = self_expand.expand(shape)
            for other in others:
                if other is None:
                    others_map.append(other)
                    continue
                # broadcast dims
                if shape != other.shape:
                    other = other.expand(shape)
                others_map.append(other)
            if any(isinstance(other, torch.Tensor) for other in others_map):
                return self_expand._fast_apply(
                    lambda x: getattr(x, op)(
                        *[
                            expand_as_right(other, x) if other is not None else None
                            for other in others_map
                        ],
                        *args,
                        **kwargs,
                    )
                )
            return getattr(self_expand, op)(*others_map, *args, **kwargs)

        return new_func

    return wrap_func


__base__setattr__ = torch.nn.Module.__setattr__


_TO_MODULE_PRESERVE_MODULE_STATE_WARNING = (
    "TensorDict.to_module() is replacing an existing nn.Parameter in the "
    "destination module with a tensor leaf that is not an nn.Parameter. This "
    "historical behavior can remove the key from module.state_dict(). Starting "
    "with TensorDict 0.14, to_module() preserves existing module parameter and "
    "buffer registrations by default. Passing preserve_module_state=None is "
    "deprecated; pass False to request the historical replacement behavior or "
    "True to preserve registrations explicitly. Support for None will be "
    "removed in TensorDict 0.15."
)


def _warn_to_module_preserve_module_state(memo) -> None:
    if memo is None:
        warn(
            _TO_MODULE_PRESERVE_MODULE_STATE_WARNING,
            FutureWarning,
            stacklevel=3,
        )
        return
    if memo.get("preserve_module_state_warned", False):
        return
    memo["preserve_module_state_warned"] = True
    warn(
        _TO_MODULE_PRESERVE_MODULE_STATE_WARNING,
        FutureWarning,
        stacklevel=3,
    )


def _maybe_preserve_module_state(
    module: torch.nn.Module,
    name: str,
    tensor: torch.Tensor,
    *,
    preserve_module_state: bool | None,
    memo,
) -> torch.Tensor:
    if preserve_module_state is False or (
        preserve_module_state is None and isinstance(tensor, torch.nn.Parameter)
    ):
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
        if preserve_module_state is None and not isinstance(tensor, torch.nn.Parameter):
            _warn_to_module_preserve_module_state(memo)
        elif preserve_module_state and (
            not isinstance(tensor, torch.nn.Parameter)
            or tensor.requires_grad != param.requires_grad
        ):
            if _tracks_gradients(tensor):
                # swap_tensor writes it to module._parameters as it is
                return tensor
            return torch.nn.Parameter(tensor, requires_grad=param.requires_grad)
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
    preserve_module_state: bool | None,
    memo,
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
    out = _parameters.pop(name, NO_DEFAULT)  # type: ignore[assignment]
    was_parameter = out is not NO_DEFAULT
    if out is NO_DEFAULT:
        out = _buffers.pop(name, NO_DEFAULT)
        was_buffer = out is not NO_DEFAULT
    if out is NO_DEFAULT:
        # dynamo doesn't like pop...
        out = __dict__.pop(name)
    if inplace:
        # swap tensor and out after updating out
        out_tmp = out.clone() if return_swap else out
        out.data.copy_(tensor.data)
        tensor = out
        out = out_tmp
    elif (
        preserve_module_state is not False
        and was_parameter
        and out is not None
        and isinstance(tensor, torch.Tensor)
    ):
        if preserve_module_state is None and not isinstance(tensor, torch.nn.Parameter):
            _warn_to_module_preserve_module_state(memo)
        elif preserve_module_state and (
            not isinstance(tensor, torch.nn.Parameter)
            or tensor.requires_grad != out.requires_grad
        ):
            if _tracks_gradients(tensor):
                keep_parameter_slot = True
            else:
                tensor = torch.nn.Parameter(tensor, requires_grad=out.requires_grad)
    elif (
        preserve_module_state is True
        and was_buffer
        and isinstance(tensor, torch.nn.Parameter)
    ):
        persistent = name not in module._non_persistent_buffers_set
        tensor = Buffer(tensor, persistent=persistent)

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
    is_dynamo = is_compiling()
    out = None
    if not is_dynamo:
        out = _TENSOR_COLLECTION_MEMO.get(datatype)

    if out is None:
        out = issubclass(datatype, TensorDictBase) or _is_tensorclass(datatype)
        if not is_dynamo:
            _TENSOR_COLLECTION_MEMO[datatype] = out
    return out


def is_tensor_collection(datatype: type | Any) -> bool:
    """Checks if a data object or a type is a tensor container from the tensordict lib.

    Returns:
        ``True`` if the input is a TensorDictBase subclass, a tensorclass or an istance of these.
        ``False`` otherwise.

    Examples:
        >>> is_tensor_collection(TensorDictBase)  # True
        >>> is_tensor_collection(TensorDict())  # True
        >>> @tensorclass
        ... class MyClass:
        ...     pass
        ...
        >>> is_tensor_collection(MyClass)  # True
        >>> is_tensor_collection(MyClass(batch_size=[]))  # True

    """
    # memoizing is 2x faster
    if not isinstance(datatype, type):
        datatype = type(datatype)
    return _is_tensor_collection(datatype)


def _default_is_leaf(cls: Type) -> bool:
    """Returns ``True`` if a type is not a tensor collection (tensordict or tensorclass), or is a pass-through type.

    Pass-through types (like UnbatchedTensor) have ``_pass_through=True`` and are considered leaves
    because their shape doesn't conform to batch dimensions.

    Note: NonTensorData types are NOT considered leaves here (they have ``_is_non_tensor=True``
    but not ``_pass_through=True``), so they are excluded from leaves when ``leaves_only=True``.

    Examples:
        >>> from tensordict import TensorDict, default_is_leaf
        >>> import torch
        >>> td = TensorDict(a={}, b="a string!", c=torch.randn(()))
        >>> print(td.keys(leaves_only=True, is_leaf=default_is_leaf))
        _TensorDictKeysView(['c'],
            include_nested=False,
            leaves_only=True)

    .. seealso:: :meth:`~tensordict.is_leaf_nontensor`.
    """
    # Only check for _pass_through attribute, not _is_non_tensor
    # This ensures NonTensorData is NOT considered a leaf (preserving original behavior)
    # while UnbatchedTensor IS considered a leaf
    if cls is Tensor:
        return True
    return not _is_tensor_collection(cls) or getattr(cls, "_pass_through", False)


def _is_leaf_nontensor(cls: Type) -> bool:
    """Returns ``True`` if a type is not a tensor collection (tensordict or tensorclass) or is a non-tensor.

    Examples:
        >>> from tensordict import TensorDict, is_leaf_nontensor
        >>> import torch
        >>> td = TensorDict(a={}, b="a string!", c=torch.randn(()))
        >>> print(td.keys(leaves_only=True, is_leaf=is_leaf_nontensor))
        _TensorDictKeysView(['b', 'c'],
            include_nested=False,
            leaves_only=True)

    .. seealso:: :meth:`~tensordict.default_is_leaf`.
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


class TensorDictBase(*_TENSORDICTBASE_MIXINS, MutableMapping, TensorCollection):
    """TensorDictBase is an abstract parent class for TensorDicts, a torch.Tensor data container."""

    _safe: bool = False
    _lazy: bool = False
    _inplace_set: bool = False
    is_meta: bool = False
    _is_locked: bool = False
    _cache: bool | None = None
    _is_non_tensor: bool = False
    _memmap_prefix = None
    _stream: torch.cuda.Stream | None = None
    # Class-level default so `_last_op` is always readable, even on a TD
    # that has never been passed through a ``_as_context_manager``-wrapped
    # method (notably under ``torch.compile`` where that decorator
    # short-circuits and never writes the attribute).
    _last_op = None
    # Class-level default; instances populate this on lock_() so locked-
    # fast-paths can avoid iterating ``_tensordict`` (which would emit
    # DICT_KEYS_MATCH guards under Dynamo). Cleared on unlock_().
    _locked_schema = None
    # Class-level default for the dim-name methods: no names until
    # _set_names stores them on the instance.
    _td_dim_names = None

    @classmethod
    def _new_unsafe(cls, *args, **kwargs) -> "TensorDictBase":
        # This to make sure all TensorDictBase subclasses have a proper fallback if they don't have a _new_unsafe
        # In other words, only TensorDict subclasses will have their type preserved, others will become TensorDict
        # instances (note that TensorDictBase should not be directly subclassed outside of this codebase, as it is
        # highly abstract).
        from tensordict._td import TensorDict

        return TensorDict._new_unsafe(*args, **kwargs)

    def __bool__(self) -> bool:
        raise RuntimeError("Converting a tensordict to boolean value is not permitted")

    def __repr__(self) -> str:
        try:
            fields = _td_fields(self)
            parts = [indent(f"fields={{{fields}}}", 4 * " ")]
            if _REPR_OPTIONS["show_batch_size"]:
                parts.append(indent(f"batch_size={self.batch_size}", 4 * " "))
            if _REPR_OPTIONS["show_device"]:
                parts.append(indent(f"device={self.device}", 4 * " "))
            if _REPR_OPTIONS["show_is_shared"]:
                parts.append(indent(f"is_shared={self.is_shared()}", 4 * " "))
            string = ",\n".join(parts)
        except AttributeError:
            # When using torch.compile, an exception may be raised with a tensordict object
            #  that has no attribute (no _tensordict or no _batch_size).
            #  To get the proper erro message and not an attribute error raised during __repr__,
            #  we simply default to '...' when trying to print the TD content.
            string = "..."
        return f"{type(self).__name__}(\n{string})"

    def __iter__(self) -> Iterator:
        """Iterates over the first batch dimension of the tensordict.

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
        """Returns the length of first dimension, if there is, otherwise 0."""
        batch_size = self.batch_size
        if not batch_size:
            return 0
        return batch_size[0]

    def __deepcopy__(self, memo: Dict[Any, Any]) -> "tensordict.TensorDict":  # noqa  # type: ignore
        return self.clone()

    def __contains__(self, key: NestedKey) -> bool:  # type: ignore
        if isinstance(key, str):
            return key in self.keys()
        if isinstance(key, tuple):
            key = unravel_key(key)
            if not key:
                raise RuntimeError(
                    "key must be a NestedKey (a str or a possibly tuple of str)."
                )
            return key in self.keys(True, is_leaf=_is_leaf_nontensor)
        raise RuntimeError(
            "key must be a NestedKey (a str or a possibly tuple of str)."
        )

    def __getitem__(self, index: IndexType) -> Self | Tensor | TensorCollection | Any:
        """Indexes all tensors according to the provided index.

        The index can be a (nested) key or any valid shape index given the
        tensordict batch size.

        If the index is a nested key and the result is a :class:`~tensordict.NonTensorData`
        object, the content of the non-tensor is returned.

        Examples:
            >>> td = TensorDict({"root": torch.arange(2), ("nested", "entry"): torch.arange(2)}, [2])
            >>> td["root"]
            tensor([0, 1])
            >>> td["nested", "entry"]
            tensor([0, 1])
            >>> td[:1]
            TensorDict(
                fields={
                    nested: TensorDict(
                        fields={
                            entry: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.int64, is_shared=False)},
                        batch_size=torch.Size([1]),
                        device=None,
                        is_shared=False),
                    root: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.int64, is_shared=False)},
                batch_size=torch.Size([1]),
                device=None,
                is_shared=False)
        """
        istuple = isinstance(index, tuple)
        if istuple or isinstance(index, str):
            # _unravel_key_to_tuple will return an empty tuple if the index isn't a NestedKey
            idx_unravel = _unravel_key_to_tuple(index)
            if idx_unravel:
                return self._get_tuple_maybe_non_tensor(idx_unravel, NO_DEFAULT)

        if (istuple and not index) or (not istuple and index is Ellipsis):
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
        if istuple and any(idx is Ellipsis for idx in index):
            index = convert_ellipsis_to_idx(index, self.batch_size)
        if len(index) <= self.batch_dims and all(
            isinstance(idx, slice) and idx == slice(None) for idx in index
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
        from tensordict._td import _SubTensorDict

        istuple = isinstance(index, tuple)
        if istuple or isinstance(index, str):
            # try:
            index_unravel = _unravel_key_to_tuple(index)
            if index_unravel:
                if value is self:
                    raise ValueError(_SELF_NESTING_ERROR.format(index))
                self._set_tuple(
                    index_unravel,
                    value,
                    inplace=(
                        BEST_ATTEMPT_INPLACE
                        if isinstance(self, _SubTensorDict)
                        else False
                    ),
                    validated=False,
                    non_blocking=False,
                )
                return

        # we must use any and because using Ellipsis in index can break with some indices
        if index is Ellipsis or (
            isinstance(index, tuple) and any(idx is Ellipsis for idx in index)
        ):
            index = convert_ellipsis_to_idx(index, self.batch_size)
        if isinstance(index, tuple) and len(index) == 1:
            index = index[0]
        if _is_new_dim_index(index):
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
            if isinstance(value, dict):
                value = self.from_dict_instance(
                    value, batch_size=indexed_bs, device=self.device
                )
            elif value.device != self.device:
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
            for key in self.keys():
                self.set_at_(key, value, index)

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
        from tensordict._torch_func import TD_HANDLED_FUNCTIONS

        if kwargs is None:
            kwargs = {}
        if func not in TD_HANDLED_FUNCTIONS or not all(
            issubclass(t, (Tensor, TensorDictBase)) or _is_tensorclass(t) for t in types
        ):
            from torch._ops import HigherOrderOperator

            if isinstance(func, HigherOrderOperator):
                with torch._C.DisableTorchFunctionSubclass():
                    return func(*args, **kwargs)
            return NotImplemented
        return TD_HANDLED_FUNCTIONS[func](*args, **kwargs)

    def auto_batch_size_(
        self, batch_dims: int | None = None, keep_compliant_size: bool = False
    ) -> Self:
        """Sets the maximum batch-size for the tensordict, up to an optional batch_dims.

        Args:
            batch_dims (int, optional): if provided, the batch-size will be at
                most ``batch_dims`` long.
            keep_compliant_size (bool, optional): if `True`, a sub-tensordict with a compliant
                size will not see its shape be changed in case `batch_dims` is passed.
                If `False`, all contained tensordicts will have a `batch_dims` that matches
                `batch_dims`.
                Defaults to `False`.

        Returns:
            self

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> td = TensorDict({"a": torch.randn(3, 4, 5), "b": {"c": torch.randn(3, 4, 6)}}, batch_size=[])
            >>> td.auto_batch_size_()
            >>> print(td.batch_size)
            torch.Size([3, 4])
            >>> td.auto_batch_size_(batch_dims=1)
            >>> print(td.batch_size)
            torch.Size([3])

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
        """Returns a TensorDict created from a dictionary or another :class:`~.tensordict.TensorDict`.

        If ``batch_size`` is not specified and ``auto_batch_size=True``, the maximum batch size possible is used.

        This function works on nested dictionaries too. For :class:`~tensordict.TensorDict`, a tensor
        collection passed as ``input_dict`` is currently returned as is, and the keyword arguments are
        not applied to it: use :meth:`~.auto_batch_size_` to compute the batch size of an existing
        tensordict.

        Args:
            input_dict (dictionary, optional): a dictionary to use as a data source
                (nested keys compatible).

        Keyword Args:
            auto_batch_size (bool, optional): if ``True``, the batch size will be computed automatically.
                Defaults to ``False``.
            batch_size (iterable of int, optional): a batch size for the tensordict.
            device (torch.device or compatible type, optional): a device for the TensorDict.
            batch_dims (int, optional): the ``batch_dims`` (ie number of leading dimensions
                to be considered for ``batch_size``). Exclusinve with ``batch_size``.
                Note that this is the __maximum__ number of batch dims of the tensordict,
                a smaller number is tolerated.
            names (list of str, optional): the dimension names of the tensordict.

        Examples:
            >>> input_dict = {"a": torch.randn(3, 4), "b": torch.randn(3)}
            >>> print(TensorDict.from_dict(input_dict, auto_batch_size=True))
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([3]),
                device=None,
                is_shared=False)
            >>> # nested dict: the nested TensorDict can have a different batch-size
            >>> # as long as its leading dims match.
            >>> input_dict = {"a": torch.randn(3), "b": {"c": torch.randn(3, 4)}}
            >>> print(TensorDict.from_dict(input_dict, auto_batch_size=True))
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([3, 4]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3]),
                device=None,
                is_shared=False)
            >>> # to work out the batch size of an existing tensordict, use auto_batch_size_
            >>> input_td = TensorDict({"a": torch.randn(3), "b": {"c": torch.randn(3, 4)}}, [])
            >>> print(input_td.auto_batch_size_())
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([3, 4]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3]),
                device=None,
                is_shared=False)

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
        """See :obj:`~tensordict.TensorDictBase.batch_size`."""
        return self.batch_size

    @shape.setter
    def shape(self, value):
        self.batch_size = value

    @property
    @abc.abstractmethod
    def batch_size(self) -> torch.Size:
        """Shape (or batch_size) of a TensorDict.

        The shape of a tensordict corresponds to the common first ``N``
        dimensions of the tensors it contains, where ``N`` is an arbitrary
        number. The batch-size contrasts with the "feature size" which repesents
        the semantically relevant shapes of a tensor. For instance, a batch of videos
        may have shape ``[B, T, C, W, H]``, where ``[B, T]`` is the batch-size (batch and
        time dimensions) and ``[C, W, H]`` are the feature dimensions (channels and spacial
        dimensions).

        The ``TensorDict`` shape is controlled by the user upon
        initialization (ie, it is not inferred from the tensor shapes).

        The ``batch_size`` can be edited dynamically if the new size is compatible
        with the TensorDict content. For instance, setting the batch size to
        an empty value is always allowed.

        Returns:
            a :obj:`~torch.Size` object describing the TensorDict batch size.

        Examples:
            >>> data = TensorDict({
            ...     "key 0": torch.randn(3, 4),
            ...     "key 1": torch.randn(3, 5),
            ...     "nested": TensorDict({"key 0": torch.randn(3, 4)}, batch_size=[3, 4])},
            ...     batch_size=[3])
            >>> data.batch_size = () # resets the batch-size to an empty value
        """
        raise NotImplementedError

    def size(self, dim: int | None = None) -> torch.Size | int:
        """Returns the size of the dimension indicated by ``dim``.

        If ``dim`` is not specified, returns the ``batch_size`` attribute of the TensorDict.

        """
        if dim is None:
            return self.batch_size
        return self.batch_size[dim]

    @property
    def data(self) -> Self:
        """Returns a tensordict containing the .data attributes of the leaf tensors."""
        return self._data()

    @data.setter
    def data(self, value: Self):
        self._data_setter(value)

    @property
    def grad(self) -> Self:
        """Returns a tensordict containing the .grad attributes of the leaf tensors."""
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
        """Zeros all the gradients of the TensorDict recursively.

        Args:
            set_to_none (bool, optional): if ``True``, tensor.grad will be ``None``,
                otherwise ``0``.
                Defaults to ``True``.

        """
        if set_to_none:
            for val in self._values_list(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS):
                val.grad = None
            return self
        for val in self._values_list(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS):
            val.grad.zero_()
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
        """Returns the dtype of the values in the tensordict, if it is unique."""
        return self._dtype()

    def _batch_size_setter(self, new_batch_size: torch.Size) -> None:
        if new_batch_size == self.batch_size:
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
        if num_none == self.batch_dims:
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
        """Length of the tensordict batch size.

        Returns:
            int describing the number of dimensions of the tensordict.

        """
        return len(self.batch_size)

    def ndimension(self) -> int:
        """See :meth:`~.batch_dims`."""
        return self.batch_dims

    @property
    def ndim(self) -> int:
        """See :meth:`~.batch_dims`."""
        return self.batch_dims

    def dim(self) -> int:
        """See :meth:`~.batch_dims`."""
        return self.batch_dims

    def numel(self) -> int:
        """Total number of elements in the batch.

        Lower-bounded to 1, as a stack of two tensordict with empty shape will
        have two elements, therefore we consider that a tensordict is at least
        1-element big.
        """
        return max(1, self.batch_size.numel())

    @property
    def depth(self) -> int:
        """Returns the depth - maximum number of levels - of a tensordict.

        The minimum depth is 0 (no nested tensordict).
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
        """Returns a TensorDict of size ``size`` filled with 0.

        By default, the returned TensorDict has the same ``torch.dtype`` and ``torch.device`` as this tensordict.

        Args:
            size (int...): a list, tuple, or torch.Size of integers defining the shape of the output tensor.

        Keyword Args:
            dtype (torch.dtype, optional): the desired type of returned tensordict.
                Default: if ``None``, the `torch.dtype` will be unchanged.
            device (torch.device, optional): the desired device of returned tensordict.
                Default: if ``None``, the ``torch.device`` will be unchanged.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensors. Default: ``False``.
            layout (torch.layout, optional): the desired layout of returned TensorDict values.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in the
                pinned memory. Works only for CPU tensors. Default: ``False``.
            empty_lazy (bool, optional): If `True`, lazy stacks will be emptied of their content.
                This can be useful whenever the content of a lazy stack is likely to change
                during filling of the new tensordict. This argument is propagated to sub-tensordicts.
                Defaults to ``False``.

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
        """Returns a TensorDict of size ``size`` filled with 1.

        By default, the returned TensorDict has the same ``torch.dtype`` and ``torch.device`` as this tensordict.

        Args:
            size (int...): a list, tuple, or torch.Size of integers defining the shape of the output tensor.

        Keyword Args:
            dtype (torch.dtype, optional): the desired type of returned tensordict.
                Default: if ``None``, the `torch.dtype` will be unchanged.
            device (torch.device, optional): the desired device of returned tensordict.
                Default: if ``None``, the ``torch.device`` will be unchanged.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensors. Default: ``False``.
            layout (torch.layout, optional): the desired layout of returned TensorDict values.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in the
                pinned memory. Works only for CPU tensors. Default: ``False``.
            empty_lazy (bool, optional): If `True`, lazy stacks will be emptied of their content.
                This can be useful whenever the content of a lazy stack is likely to change
                during filling of the new tensordict. This argument is propagated to sub-tensordicts.
                Defaults to ``False``.

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
        """Returns a TensorDict of size ``size`` with emtpy tensors.

        By default, the returned TensorDict has the same ``torch.dtype`` and ``torch.device`` as this tensordict.

        Args:
            size (int...): a list, tuple, or torch.Size of integers defining the shape of the output tensor.

        Keyword Args:
            dtype (torch.dtype, optional): the desired type of returned tensordict.
                Default: if ``None``, the `torch.dtype` will be unchanged.
            device (torch.device, optional): the desired device of returned tensordict.
                Default: if ``None``, the ``torch.device`` will be unchanged.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensors. Default: ``False``.
            layout (torch.layout, optional): the desired layout of returned TensorDict values.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in the
                pinned memory. Works only for CPU tensors. Default: ``False``.
            empty_lazy (bool, optional): If `True`, lazy stacks will be emptied of their content.
                This can be useful whenever the content of a lazy stack is likely to change
                during filling of the new tensordict. This argument is propagated to sub-tensordicts.
                Defaults to ``False``.

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
        """Returns a TensorDict of size ``size`` filled with 1.

        By default, the returned TensorDict has the same ``torch.dtype`` and ``torch.device`` as this tensordict.

        Args:
            size (sequence of int): a list, tuple, or torch.Size of integers defining the shape of the output tensor.
            fill_value (scalar): the number to fill the output tensor with.

        Keyword Args:
            dtype (torch.dtype, optional): the desired type of returned tensordict.
                Default: if ``None``, the `torch.dtype` will be unchanged.
            device (torch.device, optional): the desired device of returned tensordict.
                Default: if ``None``, the ``torch.device`` will be unchanged.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensors. Default: ``False``.
            layout (torch.layout, optional): the desired layout of returned TensorDict values.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in the
                pinned memory. Works only for CPU tensors. Default: ``False``.
            empty_lazy (bool, optional): If `True`, lazy stacks will be emptied of their content.
                This can be useful whenever the content of a lazy stack is likely to change
                during filling of the new tensordict. This argument is propagated to sub-tensordicts.
                Defaults to ``False``.

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
        data: torch.Tensor | TensorCollection,
        *,
        dtype: torch.dtype = None,
        device: DeviceType = NO_DEFAULT,
        requires_grad: bool = False,
        pin_memory: bool | None = None,
    ) -> Self:  # noqa: D417
        """Returns a new TensorDict with data as the tensor ``data``.

        By default, the returned TensorDict values have the same ``torch.dtype`` and ``torch.device`` as this tensor.

        The ``data`` can also be a tensor collection (``TensorDict`` or ``tensorclass``), in which case
        the ``new_tensor`` method iterates over the tensor pairs of ``self`` and ``data``.

        Args:
            data (torch.Tensor or TensorDictBase): the data to be copied.

        Keyword Args:
            dtype (torch.dtype, optional): the desired type of returned tensordict.
                Default: if ``None``, the `torch.dtype` will be unchanged.
            device (torch.device, optional): the desired device of returned tensordict.
                Default: if ``None``, the ``torch.device`` will be unchanged.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensors. Default: ``False``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in the
                pinned memory. Works only for CPU tensors. Default: ``False``.

        """
        kwargs = {}
        if device is not NO_DEFAULT:
            kwargs["device"] = device
        else:
            device = kwargs["device"] = data.device
        if pin_memory is not None:
            kwargs["pin_memory"] = pin_memory

        def func(x, tensor_b=None):
            if tensor_b is None:
                tensor_b = data
            return x.new_tensor(
                tensor_b,
                dtype=dtype,
                requires_grad=requires_grad,
                **kwargs,
            )

        if _is_tensor_collection(type(data)):
            return self._fast_apply(
                func,
                data,
                call_on_nested=True,
                device=device,
                batch_size=data.shape,
            )
        return self._fast_apply(
            func,
            call_on_nested=True,
            device=device,
            batch_size=data.shape,
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
                        val._with_batch_size(batch_size),
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
    def chunk(self, chunks: int, dim: int = 0) -> tuple[TensorCollection, ...]:
        """Splits a tensordict into the specified number of chunks, if possible.

        Each chunk is a view of the input tensordict.

        Args:
            chunks (int): number of chunks to return
            dim (int, optional): dimension along which to split the
                tensordict. Default is 0.

        Examples:
            >>> td = TensorDict({
            ...     'x': torch.arange(24).reshape(3, 4, 2),
            ... }, batch_size=[3, 4])
            >>> td0, td1 = td.chunk(dim=-1, chunks=2)
            >>> td0['x']
            tensor([[[ 0,  1],
                     [ 2,  3]],
                    [[ 8,  9],
                     [10, 11]],
                    [[16, 17],
                     [18, 19]]])

        """
        raise NotImplementedError

    @abc.abstractmethod
    def _unsqueeze(self, dim: int):
        raise NotImplementedError

    @abc.abstractmethod
    def _squeeze(self, dim=None):
        raise NotImplementedError

    def copy(self) -> Self:
        """Return a shallow copy of the tensordict (ie, copies the structure but not the data).

        Equivalent to `TensorDictBase.clone(recurse=False)`
        """
        return self.clone(recurse=False)

    @abc.abstractmethod
    def _view(
        self,
        *args,
        **kwargs,
    ) -> Self:
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
    def _permute(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError

    # Alias for movedim (matching torch.moveaxis)

    # Cache functionality
    def _erase_cache(self):
        self._cache = None

    # Dim names functionality
    @property
    def names(self):
        """The dimension names of the tensordict.

        The names can be set at construction time using the ``names`` argument.

        See also :meth:`~.refine_names` for details on how to set the names after
        construction.
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

    @abc.abstractmethod
    def _rename_subtds(self, value):
        """Gives the sub-tensordicts the names in value for the dims they share with self.

        The dims a sub-tensordict has beyond ``self.batch_dims`` keep their
        names. ``value=None`` clears the names of the shared dims.
        """
        raise NotImplementedError

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
        """Refines the dimension names of self according to names.

        Refining is a special case of renaming that "lifts" unnamed dimensions.
        A None dim can be refined to have any name; a named dim can only be
        refined to have the same name.

        Because named tensors can coexist with unnamed tensors, refining names
        gives a nice way to write named-tensor-aware code that works with both
        named and unnamed tensors.

        names may contain up to one Ellipsis (...). The Ellipsis is expanded
        greedily; it is expanded in-place to fill names to the same length as
        self.dim() using names from the corresponding indices of self.names.

        Returns: the same tensordict with dimensions named according to the input.

        Examples:
            >>> td = TensorDict({}, batch_size=[3, 4, 5, 6])
            >>> tdr = td.refine_names(None, None, None, "d")
            >>> assert tdr.names == [None, None, None, "d"]
            >>> tdr = td.refine_names("a", None, None, "d")
            >>> assert tdr.names == ["a", None, None, "d"]

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
        """Returns a clone of the tensordict with dimensions renamed.

        Examples:
            >>> td = TensorDict({}, batch_size=[1, 2, 3 ,4])
            >>> td.names = list("abcd")
            >>> td_rename = td.rename(c="g")
            >>> assert td_rename.names == list("abgd")

        """
        clone = self.clone(recurse=False)
        if len(names) == 1 and names[0] is None:
            clone.names = None
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
        else:
            clone.names = names
        return clone

    def rename_(self, *names, **rename_map):
        """Same as :meth:`~.rename`, but executes the renaming in-place.

        Examples:
            >>> td = TensorDict({}, batch_size=[1, 2, 3 ,4])
            >>> td.names = list("abcd")
            >>> td_renamed = td.rename_(c="g")
            >>> assert td_renamed is td
            >>> assert td.names == list("abgd")
        """
        if len(names) == 1 and names[0] is None:
            self._set_names(None)
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
    @abc.abstractmethod
    def device(self) -> torch.device | None:
        """Device of a TensorDict.

        If the TensorDict has a specified device, all
        its tensors (incl. nested ones) must live on the same device.
        If the TensorDict device is ``None``, different values can be located
        on different devices.

        Returns:
            torch.device object indicating the device where the tensors
            are placed, or None if TensorDict does not have a device.

        Examples:
            >>> td = TensorDict({
            ...     "cpu": torch.randn(3, device='cpu'),
            ...     "cuda": torch.randn(3, device='cuda'),
            ... }, batch_size=[], device=None)
            >>> td['cpu'].device
            device(type='cpu')
            >>> td['cuda'].device
            device(type='cuda')
            >>> td = TensorDict({
            ...     "x": torch.randn(3, device='cpu'),
            ...     "y": torch.randn(3, device='cuda'),
            ... }, batch_size=[], device='cuda')
            >>> td['x'].device
            device(type='cuda')
            >>> td['y'].device
            device(type='cuda')
            >>> td = TensorDict({
            ...     "x": torch.randn(3, device='cpu'),
            ...     "y": TensorDict({'z': torch.randn(3, device='cpu')}, batch_size=[], device=None),
            ... }, batch_size=[], device='cuda')
            >>> td['x'].device
            device(type='cuda')
            >>> td['y'].device # nested tensordicts are also mapped onto the appropriate device.
            device(type='cuda')
            >>> td['y', 'x'].device
            device(type='cuda')

        """
        raise NotImplementedError

    @device.setter
    @abc.abstractmethod
    def device(self, value: DeviceType) -> None:
        raise NotImplementedError

    @_lock_blocked
    def clear(self) -> Self:
        """Erases the content of the tensordict."""
        for key in list(self.keys()):
            del self[key]
        return self

    @abc.abstractmethod
    def popitem(self) -> Tuple[NestedKey, CompatibleType]:
        """Removes the item that was last inserted into the TensorDict.

        ``popitem`` will only return non-nested values.
        """
        raise NotImplementedError

    @property
    def is_cuda(self):
        return self.device is not None and self.device.type == "cuda"

    @property
    def is_cpu(self):
        return self.device is not None and self.device.type == "cpu"

    @abc.abstractmethod
    def share_memory_(self) -> Self:
        """Places all the tensors in shared memory.

        The TensorDict is then locked, meaning that any writing operations that
        isn't in-place will throw an exception (eg, rename, set or remove an
        entry).
        Conversely, once the tensordict is unlocked, the share_memory attribute
        is turned to ``False``, because cross-process identity is not
        guaranteed anymore.

        Returns:
            self

        """
        raise NotImplementedError

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
        """Returns the path where a memmap saved TensorDict is being stored.

        This argument valishes as soon as is_memmap() returns ``False`` (e.g., when the tensordict is unlocked).
        """
        if self.is_memmap():
            path = self._memmap_prefix
            return path
        raise AttributeError(
            f"The tensordict has no saved path (memmap={self.is_memmap()}, path={self._memmap_prefix})."
        )

    @abc.abstractmethod
    def make_memmap(
        self,
        key: NestedKey,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        robust_key: bool | None = True,
    ) -> MemoryMappedTensor:
        """Creates an empty memory-mapped tensor given a shape and possibly a dtype.

        .. warning::
            This method is not lock-safe by design. A memory-mapped TensorDict instance present on multiple nodes
            will need to be updated using the method :meth:`~tensordict.TensorDictBase.memmap_refresh_`.

        Writing an existing entry will result in an error.

        Args:
            key (NestedKey): the key of the new entry to write. If the key is already present in the tensordict, an
                exception is raised.
            shape (torch.Size or equivalent, torch.Tensor for nested tensors): the shape of the tensor to write.

        Keyword Arguments:
            dtype (torch.dtype, optional): the dtype of the new tensor.
            robust_key (bool, optional): if ``True`` (default), uses robust key encoding that safely
                handles keys with path separators and special characters. If ``False``,
                uses legacy behavior (keys used as-is). If ``None``, uses the default
                robust behavior.

        Returns:
            A new memory mapped tensor.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def make_memmap_from_storage(
        self,
        key: NestedKey,
        storage: torch.UntypedStorage,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        robust_key: bool | None = True,
    ) -> MemoryMappedTensor:
        """Creates an empty memory-mapped tensor given a storage, a shape and possibly a dtype.

        .. warning::
            This method is not lock-safe by design. A memory-mapped TensorDict instance present on multiple nodes
            will need to be updated using the method :meth:`~tensordict.TensorDictBase.memmap_refresh_`.

        .. note::
            If the storage has a filename associated, it must match the new filename for the file.
            If it has not a filename associated but the tensordict has an associated path, this will result in an
            exception.

        Args:
            key (NestedKey): the key of the new entry to write. If the key is already present in the tensordict, an
                exception is raised.
            storage (torch.UntypedStorage): the storage to use for the new MemoryMappedTensor. Must be a physical memory
                storage.
            shape (torch.Size or equivalent, torch.Tensor for nested tensors): the shape of the tensor to write.

        Keyword Arguments:
            dtype (torch.dtype, optional): the dtype of the new tensor.
            robust_key (bool, optional): if ``True`` (default), uses robust key encoding that safely
                handles keys with path separators and special characters. If ``False``,
                uses legacy behavior (keys used as-is). If ``None``, uses the default
                robust behavior.

        Returns:
            A new memory mapped tensor with the given storage.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def make_memmap_from_tensor(
        self,
        key: NestedKey,
        tensor: torch.Tensor,
        *,
        copy_data: bool = True,
        robust_key: bool | None = True,
    ) -> MemoryMappedTensor:
        """Creates an empty memory-mapped tensor given a tensor.

        .. warning::
            This method is not lock-safe by design. A memory-mapped TensorDict instance present on multiple nodes
            will need to be updated using the method :meth:`~tensordict.TensorDictBase.memmap_refresh_`.

        This method always copies the storage content if ``copy_data`` is ``True`` (i.e., the storage is not shared).

        Args:
            key (NestedKey): the key of the new entry to write. If the key is already present in the tensordict, an
                exception is raised.
            tensor (torch.Tensor): the tensor to replicate on physical memory.

        Keyword Arguments:
            copy_data (bool, optionaL): if ``False``, the new tensor will share the metadata of the input such as
                shape and dtype, but the content will be empty. Defaults to ``True``.
            robust_key (bool, optional): if ``True`` (default), uses robust key encoding that safely
                handles keys with path separators and special characters. If ``False``,
                uses legacy behavior (keys used as-is). If ``None``, uses the default
                robust behavior.

        Returns:
            A new memory mapped tensor with the given storage.

        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def _load_memmap(
        cls,
        prefix: Path,
        metadata: dict,
        device: torch.device | None = None,
        *,
        robust_key,
        out=None,
        allow_pickle: bool | None = None,
        mode: str | None = None,
    ):
        raise NotImplementedError

    # Key functionality: set, get, set_, set_at_, update, update_
    @abc.abstractmethod
    def entry_class(self, key: NestedKey) -> type:
        """Returns the class of an entry, possibly avoiding a call to `isinstance(td.get(key), type)`.

        This method should be preferred to ``tensordict.get(key).shape`` whenever
        :meth:`.get` can be expensive to execute.

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
        """Sets a new key-value pair.

        Args:
            key (str, tuple of str): name of the key to be set.
            item (torch.Tensor or equivalent, TensorDictBase instance): value
                to be stored in the tensordict.
            inplace (bool, optional): if ``True`` and if a key matches an existing
                key in the tensordict, then the update will occur in-place
                for that key-value pair. If inplace is ``True`` and
                the entry cannot be found, it will be added. For a more restrictive
                in-place operation, use :meth:`~.set_` instead.
                Defaults to ``False``.

        Keyword Args:
            non_blocking (bool, optional): if ``True`` and this copy is between
                different devices, the copy may occur asynchronously with respect
                to the host.

        Returns:
            self

        Examples:
            >>> td = TensorDict({}, batch_size=[3, 4])
            >>> td.set("x", torch.randn(3, 4))
            >>> y = torch.randn(3, 4, 5)
            >>> td.set("y", y, inplace=True) # works, even if 'y' is not present yet
            >>> td.set("y", torch.zeros_like(y), inplace=True)
            >>> assert (y==0).all() # y values are overwritten
            >>> td.set("y", torch.ones(5), inplace=True) # raises an exception as shapes mismatch

        """
        if item is self:
            raise ValueError(_SELF_NESTING_ERROR.format(key))
        key_tuple = _unravel_key_to_tuple(key)
        if not key_tuple:
            raise KeyError(_GENERIC_NESTED_ERR.format(key))
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
        """Registers a non-tensor value in the tensordict using :class:`tensordict.tensorclass.NonTensorData`.

        The value can be retrieved using :meth:`TensorDictBase.get_non_tensor`
        or directly using `get`, which will return the :class:`tensordict.tensorclass.NonTensorData`
        object.

        return: self

        Examples:
            >>> data = TensorDict({}, batch_size=[])
            >>> data.set_non_tensor(("nested", "the string"), "a string!")
            >>> assert data.get_non_tensor(("nested", "the string")) == "a string!"
            >>> # regular `get` works but returns a NonTensorData object
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
        """Gets a non-tensor value, if it exists, or `default` if the non-tensor value is not found.

        This method is robust to tensor/TensorDict values, meaning that if the
        value gathered is a regular tensor it will be returned too (although
        this method comes with some overhead and should not be used out of its
        natural scope).

        See :meth:`~tensordict.TensorDictBase.set_non_tensor` for more information
        on how to set non-tensor values in a tensordict.

        Args:
            key (NestedKey): the location of the NonTensorData object.
            default (Any, optional): the value to be returned if the key cannot
                be found.

        Returns: the content of the :class:`tensordict.tensorclass.NonTensorData`,
            or the entry corresponding to the ``key`` if it isn't a
            :class:`tensordict.tensorclass.NonTensorData` (or ``default`` if the
            entry cannot be found).

        Examples:
            >>> data = TensorDict({}, batch_size=[])
            >>> data.set_non_tensor(("nested", "the string"), "a string!")
            >>> assert data.get_non_tensor(("nested", "the string")) == "a string!"
            >>> # regular `get` works but returns a NonTensorData object
            >>> data.get(("nested", "the string"))
            NonTensorData(data=a string!, batch_size=torch.Size([]), device=None)

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
        """Checks if the value associated with a key is a non-tensor (:class:`~tensordict.NonTensorData`).

        Unlike the standalone :func:`~tensordict.is_non_tensor` function which checks
        a value directly, this method checks the **internal representation** of the
        stored value, making it reliable for both :class:`~tensordict.TensorDict` and
        :class:`~tensordict.TensorClass` instances.

        This is useful because :class:`~tensordict.TensorClass` ``get()`` unwraps
        :class:`~tensordict.NonTensorData` to plain values, so calling
        ``is_non_tensor(tc.get(key))`` on the result would return ``False``.
        This method always checks the wrapped storage.

        Args:
            key (NestedKey): the key to check.

        Returns:
            ``True`` if the value stored at ``key`` is a
            :class:`~tensordict.NonTensorData` or :class:`~tensordict.NonTensorStack`,
            ``False`` otherwise.

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> td = TensorDict({"obs": torch.randn(4)}, [])
            >>> td.set_non_tensor("label", "cat")
            >>> td.is_non_tensor("label")
            True
            >>> td.is_non_tensor("obs")
            False

        See Also:
            :meth:`~tensordict.TensorDictBase.set_non_tensor`,
            :meth:`~tensordict.TensorDictBase.get_non_tensor`.
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
        """Filters out all non-tensor-data."""

        def _filter(x):
            if not is_non_tensor(x):
                if is_tensor_collection(x):
                    return x.filter_non_tensor_data()
                return x

        return self._apply_nest(_filter, call_on_nested=True, filter_empty=False)

    def filter_empty_(self):
        """Filters out all empty tensordicts in-place."""
        for key, val in reversed(
            list(self.items(True, is_leaf=_NESTED_TENSORS_AS_LISTS, sort=True))
        ):
            if _is_tensor_collection(type(val)) and val.is_empty():
                del self[key]
        return self

    def _convert_inplace(self, inplace, key):
        if inplace is not False:
            has_key = key in self.keys()
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
        """Sets the values in-place at the index indicated by ``index``.

        Args:
            key (str, tuple of str): key to be modified.
            value (torch.Tensor): value to be set at the index `index`
            index (int, tensor or tuple): index where to write the values.

        Keyword Args:
            non_blocking (bool, optional): if ``True`` and this copy is between
                different devices, the copy may occur asynchronously with respect
                to the host.

        Returns:
            self

        Examples:
            >>> td = TensorDict({}, batch_size=[3, 4])
            >>> x = torch.randn(3, 4)
            >>> td.set("x", x)
            >>> td.set_at_("x", value=torch.ones(1, 4), index=slice(1))
            >>> assert (x[0] == 1).all()
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
        """Sets a value to an existing key while keeping the original storage.

        Args:
            key (str): name of the value
            item (torch.Tensor or compatible type, TensorDictBase): value to
                be stored in the tensordict

        Keyword Args:
            non_blocking (bool, optional): if ``True`` and this copy is between
                different devices, the copy may occur asynchronously with respect
                to the host.

        Returns:
            self

        Examples:
            >>> td = TensorDict({}, batch_size=[3, 4])
            >>> x = torch.randn(3, 4)
            >>> td.set("x", x)
            >>> td.set_("x", torch.zeros_like(x))
            >>> assert (x == 0).all()

        """
        key_tuple = _unravel_key_to_tuple(key)
        if not key_tuple:
            raise KeyError(_GENERIC_NESTED_ERR.format(key))
        return self._set_tuple(
            key_tuple, item, inplace=True, validated=False, non_blocking=non_blocking
        )

    # Stack functionality
    @abc.abstractmethod
    def _stack_onto_(
        self,
        list_item: list[CompatibleType],
        dim: int,
    ) -> Self:
        """Stacks a list of values onto an existing key while keeping the original storage.

        Args:
            key (str): name of the value
            list_item (list of torch.Tensor): value to be stacked and stored in the tensordict.
            dim (int): dimension along which the tensors should be stacked.

        Returns:
            self

        """
        raise NotImplementedError

    def _stack_onto_at_(
        self,
        key: NestedKey,
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
        """Gets the value stored with the input key.

        Args:
            key (str, tuple of str): key to be queried. If tuple of str it is
                equivalent to chained calls of getattr.
            default: default value if the key is not found in the tensordict. Defaults to ``None``.

                .. warning::
                    Previously, if a key was not present in the tensordict and no default
                    was passed, a `KeyError` was raised. From v0.7, this behaviour has been changed
                    and a `None` value is returned instead (in accordance with the what dict.get behavior).
                    Use ``td[key]`` to raise a `KeyError` for a missing key. Restoring the old behavior with
                    ``TD_GET_DEFAULTS_TO_NONE=0`` or ``set_get_defaults_to_none(False)`` is deprecated
                    and will be removed in TensorDict 0.17.

        .. note:: Keyword arguments can be passed to :meth:`~.get` when dealing with ragged tensors.
            See :meth:`~tensordict.LazyStackedTensorDict.get` for a complete overview.

        Examples:
            >>> td = TensorDict({"x": 1}, batch_size=[])
            >>> td.get("x")
            tensor(1)
            >>> td.get("y")
            None
        """
        key_tuple = _unravel_key_to_tuple(key)
        if not key_tuple:
            raise KeyError(_GENERIC_NESTED_ERR.format(key))
        # Find what the default is
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
        else:
            default = NO_DEFAULT
        return self._get_tuple(key_tuple, default=default, **kwargs)

    @abc.abstractmethod
    def _get_str(self, key, default, **kwargs):
        raise NotImplementedError

    def _get_tuple(self, key, default, **kwargs):
        first = self._get_str(key[0], default, **kwargs)
        if len(key) == 1 or first is default:
            return first
        try:
            return first._get_tuple(key[1:], default=default, **kwargs)
        except AttributeError as err:
            if "has no attribute" in str(err):
                rest = key[1] if len(key) == 2 else key[1:]
                raise ValueError(
                    f"{key[0]!r} is a {type(first).__name__}, not a tensordict, "
                    f"so it has no entry {rest!r}."
                )

    def _get_tuple_maybe_non_tensor(self, key, default, **kwargs):
        result = self._get_tuple(key, default, **kwargs)
        if _pass_through(result) and not _is_unbatched(result):
            if isinstance(result, TensorDictBase):
                return result.tolist(as_linked_list=True)
            return result.data
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
        """Get the value of a tensordict from the key `key` at the index `idx`.

        Args:
            key (str, tuple of str): key to be retrieved.
            index (int, slice, torch.Tensor, iterable): index of the tensor.
            default (torch.Tensor): default value to return if the key is
                not present in the tensordict.

        Returns:
            indexed tensor.

        Examples:
            >>> td = TensorDict({"x": torch.arange(3)}, batch_size=[])
            >>> td.get_at("x", index=1)
            tensor(1)

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
        out = self._get_str(key, default, **kwargs)
        if out is default:
            return out
        return out[idx]

    def _get_at_tuple(self, key, idx, default, **kwargs):
        out = self._get_tuple(key, default, **kwargs)
        if out is default:
            return out
        return out[idx]

    def get_item_shape(self, key: NestedKey):
        """Returns the shape of the entry, possibly avoiding recurring to :meth:`~.get`."""
        return _shape(self.get(key))

    @_lock_blocked
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
        """Updates the TensorDict with values from either a dictionary or another TensorDict.

        .. warning:: `update` will corrupt the data if called within a try/except block. Do not user this method within
            such blocks hoping to catch and patch errors that occur during the execution.

        Args:
            input_dict_or_td (TensorDictBase or dict, optional): input data to be written
                in self. If ``None``, only the keyword arguments are used.
            clone (bool, optional): whether the tensors in the input (
                tensor) dict should be cloned before being set.
                Defaults to ``False``.
            inplace (bool, optional): if ``True`` and if a key matches an existing
                key in the tensordict, then the update will occur in-place
                for that key-value pair. If the entry cannot be found, it will be
                added. Defaults to ``False``.

        Keyword Args:
            keys_to_update (sequence of NestedKeys, optional): if provided, only
                the list of keys in ``key_to_update`` will be updated.
                This is aimed at avoiding calls to
                ``data_dest.update(data_src.select(*keys_to_update))``.
            non_blocking (bool, optional): if ``True`` and this copy is between
                different devices, the copy may occur asynchronously with respect
                to the host.
            is_leaf (Callable[[Type], bool], optional): a callable that indicates
                whether an object type is to be considered a leaf and swapped
                or a tensor collection.

                .. seealso:: :meth:`~tensordict.is_leaf_nontensor` and :meth:`~tensordict.default_is_leaf`.

            update_batch_size (bool, optional): if ``True``, ``update`` will attempt to update the batch-size
                of the destination (`self`) if it mismatches the source's batch size. Defaults to ``False``.

                .. note:: In cases where the batch size does not match, :class:`~tensordict.LazyStackTensorDict`
                    instances will be emptied of their content and copies of the tensordicts from the source
                    will be used to repopulate the container.

                .. note:: This argument assumes that `keys_to_update` is left empty, and that `inplace=False`.
                    If the keys of the destination (`self`) is not a subset of the keys of the source,
                    an exception will be raised, as TensorDict will be unable to infer what to do with the extra
                    destination entries.

            ignore_lock (bool, optional): if ``True``, any tensordict can be updated regardless of its locked status.
                Defaults to `False`.

            **kwargs: additional key-value pairs to write into ``self`` as top-level entries. When both
                ``input_dict_or_td`` and ``kwargs`` are provided, kwargs win on key conflict.

        .. note:: When updating a :class:`~tensordict.LazyStackedTensorDict` with N elements with another
            :class:`~tensordict.LazyStackedTensorDict` with M elements, with M > N, along the stack dimension,
            the ``update`` method will append copies of the extra tensordicts to the dest (self) lazy stack.
            This allows users to rely on ``update`` to increment lazy stacks progressively.

        .. note:: Keyword arguments are treated as top-level keys only. Nested structures still work when
            the kwarg value is itself a dict or tensor collection (e.g. ``td.update(outer={"inner": val})``).
            For deeper tuple keys, use the positional dict form: ``td.update({("a", "b"): val})``.
            Keys whose names collide with reserved parameters (``clone``, ``inplace``, ``non_blocking``,
            ``keys_to_update``, ``is_leaf``, ``update_batch_size``, ``ignore_lock``) must be passed through
            the positional dict form.

        Returns:
            self

        Examples:
            >>> td = TensorDict({}, batch_size=[3])
            >>> a = torch.randn(3)
            >>> b = torch.randn(3, 4)
            >>> other_td = TensorDict({"a": a, "b": b}, batch_size=[])
            >>> td.update(other_td, inplace=True) # writes "a" and "b" even though they can't be found
            >>> assert td['a'] is other_td['a']
            >>> other_td = other_td.clone().zero_()
            >>> td.update(other_td)
            >>> assert td['a'] is other_td['a']  # entries are written by reference unless clone=True
            >>> # keyword form for top-level entries
            >>> td.update(monkey=torch.zeros(3))
            >>> assert (td["monkey"] == 0).all()

        """
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
        batch_size_changed = False
        if input_dict_or_td is self:
            # no op
            return self
        if is_leaf is None:
            is_leaf = _is_leaf_nontensor
        if keys_to_update is not None:
            if len(keys_to_update) == 0:
                return self
            keys_to_update = unravel_key_list(keys_to_update)

        if (
            _is_tensor_collection(type(input_dict_or_td))
            # here, we do a loose check on the batch sizes: it could be that the source has batch_size (1,) and self (1, 2)
            # and that all the values have an appropriate shape for the new batch size.
            # What we want to catch is any batch size item that is obviously mismatching, like (1, 2, 3) and (1, 4, 3).
            and self.batch_size[: input_dict_or_td.batch_dims]
            != input_dict_or_td.batch_size[: self.batch_dims]
        ):
            if not update_batch_size:
                raise RuntimeError(
                    "update_batch_size must be set to True to be able to update "
                    "tensordicts of different batch size. Got sizes {}"
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

    def update_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T | None = None,
        clone: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
        **kwargs,
    ) -> Self:
        """Updates the TensorDict in-place with values from either a dictionary or another TensorDict.

        Unlike :meth:`~.update`, this function will throw an error if the key is unknown to ``self``.

        Args:
            input_dict_or_td (TensorDictBase or dict, optional): input data to be written
                in self. If ``None``, only the keyword arguments are used.
            clone (bool, optional): whether the tensors in the input (
                tensor) dict should be cloned before being set. Defaults to ``False``.

        Keyword Args:
            keys_to_update (sequence of NestedKeys, optional): if provided, only
                the list of keys in ``key_to_update`` will be updated.
                This is aimed at avoiding calls to
                ``data_dest.update_(data_src.select(*keys_to_update))``.
            non_blocking (bool, optional): if ``True`` and this copy is between
                different devices, the copy may occur asynchronously with respect
                to the host.
            **kwargs: additional key-value pairs to write into ``self`` as top-level entries. As with the
                positional form, any key that does not already exist in ``self`` raises :class:`KeyError`.
                When both ``input_dict_or_td`` and ``kwargs`` are provided, kwargs win on key conflict.

        .. note:: Keyword arguments are treated as top-level keys only. Nested structures still work when
            the kwarg value is itself a dict or tensor collection. Keys whose names collide with reserved
            parameters (``clone``, ``non_blocking``, ``keys_to_update``) must be passed through the positional
            dict form.

        Returns:
            self

        Examples:
            >>> a = torch.randn(3)
            >>> b = torch.randn(3, 4)
            >>> td = TensorDict({"a": a, "b": b}, batch_size=[3])
            >>> other_td = TensorDict({"a": a*0, "b": b*0}, batch_size=[])
            >>> td.update_(other_td)
            >>> assert td['a'] is not other_td['a']
            >>> assert (td['a'] == other_td['a']).all()
            >>> assert (td['a'] == 0).all()
            >>> # keyword form for top-level entries (key must already exist)
            >>> td.update_(a=torch.ones(3))
            >>> assert (td["a"] == 1).all()

        """
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

        if not _is_tensor_collection(type(input_dict_or_td)):
            from tensordict import TensorDict

            input_dict_or_td = TensorDict.from_dict(
                input_dict_or_td, batch_dims=self.batch_dims
            )

        if keys_to_update is not None:
            if len(keys_to_update) == 0:
                return self
            keys_to_update = [_unravel_key_to_tuple(key) for key in keys_to_update]

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
                        dest.copy_(source, non_blocking=non_blocking)

        else:
            # Fastest route using _foreach_copy_
            keys, vals = self._items_list(True, True)
            new_keys, other_val = input_dict_or_td._items_list(
                True, True, sorting_keys=keys, default="intersection"
            )
            if len(new_keys):
                if len(other_val) != len(vals):
                    vals = dict(zip(keys, vals))
                    vals = [vals[k] for k in new_keys]
                copy_fn = _foreach_copy_compiled if is_compiling() else _foreach_copy_
                copy_fn(vals, other_val, non_blocking=non_blocking)
                return self
            named = True

            def inplace_update(name, source, dest):
                if source is None:
                    return None
                if dest is None:
                    raise KeyError(
                        f"The key {name} was not found in the dest tensordict."
                    )
                dest.copy_(source, non_blocking=non_blocking)

        input_dict_or_td._apply_nest(
            inplace_update,
            self,
            nested_keys=True,
            default=None,
            filter_empty=True,
            named=named,
            is_leaf=_is_leaf_nontensor,
        )
        return self

    def update_at_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        idx: IndexType,
        clone: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
    ) -> Self:
        """Updates the TensorDict in-place at the specified index with values from either a dictionary or another TensorDict.

        Unlike  TensorDict.update, this function will throw an error if the key is unknown to the TensorDict.
        This method keeps the general ``set_at_`` semantics and supports tensor
        and non-tensor leaves. For optimized tensor-only copies in hot paths,
        use :meth:`~tensordict.TensorDictBase.copy_at_` with ``fast=True``.

        Args:
            input_dict_or_td (TensorDictBase or dict): input data to be written
                in self.
            idx (int, torch.Tensor, iterable, slice): index of the tensordict
                where the update should occur.
            clone (bool, optional): whether the tensors in the input (
                tensor) dict should be cloned before being set. Default is
                `False`.

        Keyword Args:
            keys_to_update (sequence of NestedKeys, optional): if provided, only
                the list of keys in ``key_to_update`` will be updated.
            non_blocking (bool, optional): if ``True`` and this copy is between
                different devices, the copy may occur asynchronously with respect
                to the host.

        Returns:
            self

        Examples:
            >>> td = TensorDict({
            ...     'a': torch.zeros(3, 4, 5),
            ...     'b': torch.zeros(3, 4, 10)}, batch_size=[3, 4])
            >>> td.update_at_(
            ...     TensorDict({
            ...         'a': torch.ones(1, 4, 5),
            ...         'b': torch.ones(1, 4, 10)}, batch_size=[1, 4]),
            ...    slice(1, 2))
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: Tensor(shape=torch.Size([3, 4, 10]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([3, 4]),
                device=None,
                is_shared=False)
            >>> assert (td[1] == 1).all()

        """
        if idx == ():
            return self.update_(
                input_dict_or_td=input_dict_or_td,
                keys_to_update=keys_to_update,
                clone=clone,
                non_blocking=non_blocking,
            )
        if keys_to_update is not None:
            if len(keys_to_update) == 0:
                return self
            keys_to_update = unravel_key_list(keys_to_update)
        for key, value in input_dict_or_td.items():
            firstkey, *nextkeys = _unravel_key_to_tuple(key)
            if keys_to_update and not any(
                firstkey == ktu if isinstance(ktu, str) else firstkey == ktu[0]
                for ktu in keys_to_update
            ):
                continue
            if not _is_accepted_class(type(value)):
                raise TypeError(
                    f"Expected value to be one of types {_ACCEPTED_CLASSES} "
                    f"but got {type(value)}"
                )
            if clone:
                value = value.clone()
            self.set_at_((firstkey, *nextkeys), value, idx, non_blocking=non_blocking)
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
        """Creates a shallow copy of the tensordict where entries have been replaced.

        Accepts one unnamed argument which must be a dictionary of a :class:`~tensordict.TensorDictBase` subclass.
        Additionally, first-level entries can be updated with the named keyword arguments.

        Returns:
            a copy of ``self`` with updated entries if the input is non-empty. If an empty dict or no dict is provided
            and the kwargs are empty, ``self`` is returned.

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
        else:
            dict_to_replace = {}
        if kwargs:
            dict_to_replace.update(kwargs)
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
        """Creates a nested tensordict of the same shape, device and dim names as the current tensordict.

        If the value already exists, it will be overwritten by this operation.
        This operation is blocked in locked tensordicts.

        Examples:
            >>> data = TensorDict({}, [3, 4, 5])
            >>> data.create_nested("root")
            >>> data.create_nested(("some", "nested", "value"))
            >>> print(data)
            TensorDict(
                fields={
                    root: TensorDict(
                        fields={
                        },
                        batch_size=torch.Size([3, 4, 5]),
                        device=None,
                        is_shared=False),
                    some: TensorDict(
                        fields={
                            nested: TensorDict(
                                fields={
                                    value: TensorDict(
                                        fields={
                                        },
                                        batch_size=torch.Size([3, 4, 5]),
                                        device=None,
                                        is_shared=False)},
                                batch_size=torch.Size([3, 4, 5]),
                                device=None,
                                is_shared=False)},
                        batch_size=torch.Size([3, 4, 5]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3, 4, 5]),
                device=None,
                is_shared=False)
        """
        key = _unravel_key_to_tuple(key)
        self._create_nested_tuple(key)
        return self

    def _create_nested_str(self, key):
        out = self.empty()
        self._set_str(key, out, inplace=False, validated=True, non_blocking=False)
        return out

    def _create_nested_tuple(self, key):
        td = self._create_nested_str(key[0])
        if len(key) > 1:
            td._create_nested_tuple(key[1:])

    def copy_(self, tensordict: T, non_blocking: bool = False) -> Self:
        """See :obj:`TensorDictBase.update_`.

        The non-blocking argument will be ignored and is just present for
        compatibility with :func:`torch.Tensor.copy_`.
        """
        return self.update_(tensordict, non_blocking=non_blocking)

    def copy_at_(
        self,
        tensordict: T,
        idx: IndexType,
        non_blocking: bool = False,
        *,
        fast: bool | None = True,
    ) -> Self:
        """Copies values from ``tensordict`` into ``self`` at the specified index.

        ``copy_at_`` is an explicit copy-oriented variant of
        :meth:`~tensordict.TensorDictBase.update_at_`. Unlike ``update_at_``,
        it may use optimized tensor-only copy paths and is intended for hot
        paths where the source and destination structures are known to match.

        Args:
            tensordict (TensorDictBase): input data to be copied in ``self``.
            idx (int, torch.Tensor, iterable, slice): index of the tensordict
                where the copy should occur.
            non_blocking (bool, optional): if ``True`` and this copy is between
                different devices, the copy may occur asynchronously with respect
                to the host.

        Keyword Args:
            fast (bool or None, optional): controls whether ``copy_at_`` may
                fall back to :meth:`~tensordict.TensorDictBase.update_at_`.
                If ``True``, only the optimized tensor-only path is used and a
                ``RuntimeError`` is raised when the fast path is not available.
                If ``False``, this method delegates directly to ``update_at_``.
                If ``None``, the deprecated compatibility behavior warns and
                falls back to ``update_at_`` when the fast path is unavailable.
                Defaults to ``True``.

        Returns:
            self
        """
        if fast is None:
            warnings.warn(
                "copy_at_(..., fast=None) is deprecated and falls back to "
                "update_at_ when the optimized tensor-only copy path is "
                "unavailable. Starting with TensorDict 0.14, copy_at_ defaults "
                "to fast=True. Pass fast=False to request the general update "
                "semantics explicitly; support for fast=None will be removed "
                "in TensorDict 0.15.",
                FutureWarning,
                stacklevel=2,
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
        """Checks if the tensordict contains any leaf."""
        for _ in self.keys(True, True):
            return False
        return True

    # Dict features: setdefault, items, values, keys, ...
    def setdefault(
        self, key: NestedKey, default: CompatibleType | Any, inplace: bool = False
    ) -> CompatibleType:
        """Insert the ``key`` entry with a value of ``default`` if ``key`` is not in the tensordict.

        Return the value for ``key`` if ``key`` is in the tensordict, else ``default``.

        Args:
            key (str or nested key): the name of the value.
            default (torch.Tensor or compatible type, TensorDictBase): value
                to be stored in the tensordict if the key is not already present.

        Returns:
            The value of key in the tensordict. Will be default if the key was not
            previously set.

        Examples:
            >>> td = TensorDict({}, batch_size=[3, 4])
            >>> val = td.setdefault("a", torch.zeros(3, 4))
            >>> assert (val == 0).all()
            >>> val = td.setdefault("a", torch.ones(3, 4))
            >>> assert (val == 0).all() # output is still 0

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
        """Returns a generator of key-value pairs for the tensordict.

        Args:
            include_nested (bool, optional): if ``True``, nested values will be returned.
                Defaults to ``False``.
            leaves_only (bool, optional): if ``False``, only leaves will be
                returned. Defaults to ``False``.
            is_leaf (callable, optional): a callable over a class type returning
                a bool indicating if this class has to be considered as a leaf.

                .. note:: The purpose of `is_leaf` is not to prevent recursive calls into nested tensordicts, but
                    rather to mark certain types as "leaves" for the purpose of filtering when `leaves_only=True`.
                    Even if `is_leaf(cls)` returns `True`, the nested structure of the tensordict will still be
                    traversed if `include_nested=True`.
                    In other words, `is_leaf` does not control the recursion depth, but rather provides a way to filter
                    out certain types from the result when `leaves_only=True`. This means that a node in the tree can
                    be both a leaf and a node with children.
                    In practice, the default value of ``is_leaf`` does exclude tensordict and tensorclass instances
                    from the leaf set.

                .. seealso:: :meth:`~tensordict.is_leaf_nontensor` and :meth:`~tensordict.default_is_leaf`.

        Keyword Args:
            sort (bool, optional): whether the keys should be sorted. For nested keys,
                the keys are sorted according to their joined name (ie, ``("a", "key")`` will
                be counted as ``"a.key"`` for sorting). Be mindful that sorting may incur
                significant overhead when dealing with large tensordicts.
                Defaults to ``False``.

        """
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
                        if not is_non_tensor(cls) and not _pass_through(val):
                            yield from (
                                (_unravel_key_to_tuple((k, _key)), _val)
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

    def non_tensor_items(self, include_nested: bool = False):
        """Returns all non-tensor leaves, maybe recursively."""
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
        """Returns a generator representing the values for the tensordict.

        Args:
            include_nested (bool, optional): if ``True``, nested values will be returned.
                Defaults to ``False``.
            leaves_only (bool, optional): if ``False``, only leaves will be
                returned. Defaults to ``False``.
            is_leaf (callable, optional): a callable over a class type returning
                a bool indicating if this class has to be considered as a leaf.

                .. note:: The purpose of `is_leaf` is not to prevent recursive calls into nested tensordicts, but
                    rather to mark certain types as "leaves" for the purpose of filtering when `leaves_only=True`.
                    Even if `is_leaf(cls)` returns `True`, the nested structure of the tensordict will still be
                    traversed if `include_nested=True`.
                    In other words, `is_leaf` does not control the recursion depth, but rather provides a way to filter
                    out certain types from the result when `leaves_only=True`. This means that a node in the tree can
                    be both a leaf and a node with children.
                    In practice, the default value of ``is_leaf`` does exclude tensordict and tensorclass instances
                    from the leaf set.

                .. seealso:: :meth:`~tensordict.is_leaf_nontensor` and :meth:`~tensordict.default_is_leaf`.

        Keyword Args:
            sort (bool, optional): whether the keys should be sorted. For nested keys,
                the keys are sorted according to their joined name (ie, ``("a", "key")`` will
                be counted as ``"a.key"`` for sorting). Be mindful that sorting may incur
                significant overhead when dealing with large tensordicts.
                Defaults to ``False``.

        """
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
                        if not is_non_tensor(cls) and not _pass_through(val):
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
        keys, vals = self._items_list(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS)
        data = [val.data for val in vals]
        items = dict(zip(keys, data))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            propagate_lock=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
        )

    def _data_setter(self, value: Self):
        keys, vals = value._items_list(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS)
        keys_self, vals_self = value._items_list(
            True, True, is_leaf=_NESTED_TENSORS_AS_LISTS
        )
        self_dict = dict(zip(keys_self, vals_self))
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
        """Returns a generator of tensordict keys.

        .. warning::
            TensorDict ``keys()`` method returns a lazy view of the keys. If the ``keys``
            are queried but not iterated over and then the tensordict is modified, iterating over
            the keys later will return the new configuration of the keys.

        Args:
            include_nested (bool, optional): if ``True``, nested values will be returned.
                Defaults to ``False``.
            leaves_only (bool, optional): if ``False``, only leaves will be
                returned. Defaults to ``False``.
            is_leaf (callable, optional): a callable over a class type returning
                a bool indicating if this class has to be considered as a leaf.

                .. note:: The purpose of `is_leaf` is not to prevent recursive calls into nested tensordicts, but
                    rather to mark certain types as "leaves" for the purpose of filtering when `leaves_only=True`.
                    Even if `is_leaf(cls)` returns `True`, the nested structure of the tensordict will still be
                    traversed if `include_nested=True`.
                    In other words, `is_leaf` does not control the recursion depth, but rather provides a way to filter
                    out certain types from the result when `leaves_only=True`. This means that a node in the tree can
                    be both a leaf and a node with children.
                    In practice, the default value of ``is_leaf`` does exclude tensordict and tensorclass instances
                    from the leaf set.

                .. seealso:: :meth:`~tensordict.is_leaf_nontensor` and :meth:`~tensordict.default_is_leaf`.

        Keyword Args:
            sort (bool, optional): whether the keys shoulbe sorted. For nested keys,
                the keys are sorted according to their joined name (ie, ``("a", "key")`` will
                be counted as ``"a.key"`` for sorting). Be mindful that sorting may incur
                significant overhead when dealing with large tensordicts.
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
        """Removes and returns a value from a tensordict.

        If the value is not present and no default value is provided, a KeyError
        is thrown.

        Args:
            key (str or nested key): the entry to look for.
            default (Any, optional): the value to return if the key cannot be found.

        Examples:
            >>> td = TensorDict({"1": 1}, [])
            >>> one = td.pop("1")
            >>> assert one == 1
            >>> none = td.pop("1", default=None)
            >>> assert none is None
        """
        key_tuple = _unravel_key_to_tuple(key)
        if not key_tuple:
            raise KeyError(_GENERIC_NESTED_ERR.format(key))
        # Use _UNSET sentinel to detect if key exists without try/except (compile-friendly)
        out = self.get(key_tuple, _UNSET)
        if out is _UNSET:
            # Key not found
            if default is NO_DEFAULT:
                raise KeyError(
                    _KEY_ERROR.format(
                        key_tuple[0] if len(key_tuple) == 1 else key_tuple,
                        type(self).__name__,
                        sorted(self.keys(include_nested=len(key_tuple) > 1), key=str),
                    )
                )
            return default
        self.del_(key_tuple)
        return out

    @property
    @_cache_while_locked  # noqa: B019
    def sorted_keys(self) -> list[NestedKey]:
        """Returns the keys sorted in alphabetical order.

        Does not support extra arguments.

        If the TensorDict is locked, the keys are cached until the tensordict
        is unlocked for faster execution.

        """
        return sorted(self.keys())

    def _transform_keys(self, key_transform: Callable[[NestedKey], NestedKey]) -> Self:
        """Transform all keys using the provided function.

        Args:
            key_transform (Callable): A function that takes a NestedKey and returns a new NestedKey.
                For string keys, it receives a string. For tuple keys, it receives a tuple.

        Returns:
            A new TensorDict with transformed keys.

        Examples:
            >>> td = TensorDict({"a": torch.randn(3), "b": torch.randn(3)}, [3])
            >>> td_transformed = td._transform_keys(lambda key: f"avg_{key}")
            >>> print(td_transformed.keys())
            ["avg_a", "avg_b"]

            >>> td_nested = TensorDict({("a", "b"): torch.randn(3)}, [3])
            >>> td_transformed = td_nested._transform_keys(lambda key: tuple(f"avg_{k}" for k in key))
            >>> print(td_transformed.keys())
            [("avg_a", "avg_b")]

        """
        new_td = self.empty()
        for key, value in self.items():
            new_key = key_transform(key)
            # If the value is a TensorDict, recursively transform its keys
            if hasattr(value, "_transform_keys"):
                value = value._transform_keys(key_transform)
            new_td[new_key] = value
        return new_td

    @abc.abstractmethod
    def rename_key_(
        self, old_key: NestedKey, new_key: NestedKey, safe: bool = False
    ) -> Self:
        """Renames a key with a new string and returns the same tensordict with the updated key name.

        Args:
            old_key (str or nested key): key to be renamed.
            new_key (str or nested key): new name of the entry.
            safe (bool, optional): if ``True``, an error is thrown when the new
                key is already present in the TensorDict.

        Returns:
            self

        """
        raise NotImplementedError

    @abc.abstractmethod
    def del_(self, key: NestedKey) -> Self:
        """Deletes a key of the tensordict.

        Args:
            key (NestedKey): key to be deleted

        Returns:
            self

        """
        raise NotImplementedError

    # Apply and map functionality
    def apply_(self, fn: Callable, *others, **kwargs) -> Self:
        """Applies a callable to all values stored in the tensordict and re-writes them in-place.

        Args:
            fn (Callable): function to be applied to the tensors in the
                tensordict.
            *others (sequence of TensorDictBase, optional): the other
                tensordicts to be used.

        Keyword Args: See :meth:`~.apply`.

        Returns:
            self or a copy of self with the function applied

        """
        return self.apply(fn, *others, inplace=True, **kwargs)

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
        out: TensorDictBase | None = None,
        **constructor_kwargs,
    ) -> Self | None:
        """Applies a callable to all values stored in the tensordict and sets them in a new tensordict.

        The callable signature must be ``Callable[Tuple[Tensor, ...], Tensor | TensorDictBase | None]``.

        Args:
            fn (Callable): function to be applied to the tensors in the
                tensordict.
            *others (TensorDictBase instances, optional): if provided, these
                tensordict instances should have a structure matching the one
                of self. The ``fn`` argument should receive as many
                unnamed inputs as the number of tensordicts, including self.
                If other tensordicts have missing entries, a default value
                can be passed through the ``default`` keyword argument.

        Keyword Args:
            batch_size (sequence of int, optional): if provided,
                the resulting TensorDict will have the desired batch_size.
                The :obj:`batch_size` argument should match the batch_size after
                the transformation. This is a keyword only argument.
            device (torch.device, optional): the resulting device, if any.
            names (list of str, optional): the new dimension names, in case the
                batch_size is modified.
            inplace (bool, optional): if True, changes are made in-place.
                Default is False. This is a keyword only argument.
            default (Any, optional): default value for missing entries in the
                other tensordicts. If not provided, missing entries will
                raise a `KeyError`.
            filter_empty (bool, optional): if ``True``, empty tensordicts will be
                filtered out. This also comes with a lower computational cost as
                empty data structures won't be created and destroyed. Non-tensor data
                is considered as a leaf and thereby will be kept in the tensordict even
                if left untouched by the function. If ``False``, empty tensordicts are kept.
                Defaults to ``None``, which behaves like ``True`` (tensordicts left empty by
                ``fn`` are filtered out, and ``None`` is returned if ``fn`` returns no value at
                all) except that tensordicts that were already empty are kept.
            propagate_lock (bool, optional): if ``True``, a locked tensordict will produce
                another locked tensordict. Defaults to ``False``.
            call_on_nested (bool, optional): if ``True``, the function will be called on first-level tensors
                and containers (TensorDict or tensorclass). In this scenario, ``func`` is responsible of
                propagating its calls to nested levels. This allows a fine-grained behaviour
                when propagating the calls to nested tensordicts.
                If ``False``, the function will only be called on leaves, and ``apply`` will take care of dispatching
                the function to all leaves.

                    >>> td = TensorDict({"a": {"b": [0.0, 1.0]}, "c": [1.0, 2.0]})
                    >>> def mean_tensor_only(val):
                    ...     if is_tensor_collection(val):
                    ...         raise RuntimeError("Unexpected!")
                    ...     return val.mean()
                    >>> td_mean = td.apply(mean_tensor_only)
                    >>> def mean_any(val):
                    ...     if is_tensor_collection(val):
                    ...         # Recurse
                    ...         return val.apply(mean_any, call_on_nested=True)
                    ...     return val.mean()
                    >>> td_mean = td.apply(mean_any, call_on_nested=True)

            out (TensorDictBase, optional): a tensordict where to write the results. This can be used to avoid
                creating a new tensordict:

                    >>> td = TensorDict({"a": 0})
                    >>> td.apply(lambda x: x+1, out=td)
                    >>> assert (td==1).all()

                .. warning::
                    If the operation executed on the tensordict requires multiple keys to be accessed for
                    a single computation, providing an ``out`` argument equal to ``self`` can cause the operation
                    to provide silently wrong results.
                    For instance:

                        >>> td = TensorDict({"a": 1, "b": 1})
                        >>> td.apply(lambda x: x+td["a"])["b"] # Right!
                        tensor(2)
                        >>> td.apply(lambda x: x+td["a"], out=td)["b"] # Wrong!
                        tensor(3)

            **constructor_kwargs: additional keyword arguments to be passed to the
                TensorDict constructor.

        Returns:
            a new tensordict with transformed_in tensors.

        Example:
            >>> td = TensorDict({
            ...     "a": -torch.ones(3),
            ...     "b": {"c": torch.ones(3)}},
            ...     batch_size=[3])
            >>> td_1 = td.apply(lambda x: x+1)
            >>> assert (td_1["a"] == 0).all()
            >>> assert (td_1["b", "c"] == 2).all()
            >>> td_2 = td.apply(lambda x, y: x+y, td)
            >>> assert (td_2["a"] == -2).all()
            >>> assert (td_2["b", "c"] == 2).all()

        .. note::
            If ``None`` is returned by the function, the entry is ignored. This
            can be used to filter the data in the tensordict:

            >>> td = TensorDict({"1": 1, "2": 2, "b": {"2": 2, "1": 1}}, [])
            >>> def filter(tensor):
            ...     if tensor == 1:
            ...         return tensor
            >>> td.apply(filter)
            TensorDict(
                fields={
                    1: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                    b: TensorDict(
                        fields={
                            1: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)

        .. note::
            The apply method will return an :class:`~tensordict.TensorDict` instance,
            regardless of the input type. To keep the same type, one can execute

            >>> out = td.clone(False).update(td.apply(...))


        """
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
        out: TensorDictBase | None = None,
        **constructor_kwargs,
    ) -> Self | None:
        """Applies a key-conditioned callable to all values stored in the tensordict and sets them in a new atensordict.

        The callable signature must be ``Callable[Tuple[str, Tensor, ...], Tensor | TensorDictBase | None]``.

        Args:
            fn (Callable): function to be applied to the (name, tensor) pairs in the
                tensordict. For each leaf, only its leaf name will be used (not
                the full `NestedKey`).
            *others (TensorDictBase instances, optional): if provided, these
                tensordict instances should have a structure matching the one
                of self. The ``fn`` argument should receive as many
                unnamed inputs as the number of tensordicts, including self.
                If other tensordicts have missing entries, a default value
                can be passed through the ``default`` keyword argument.
            nested_keys (bool, optional): if ``True``, the complete path
                to the leaf will be used. Defaults to ``False``, i.e. only the last
                string is passed to the function.
            batch_size (sequence of int, optional): if provided,
                the resulting TensorDict will have the desired batch_size.
                The :obj:`batch_size` argument should match the batch_size after
                the transformation. This is a keyword only argument.
            device (torch.device, optional): the resulting device, if any.
            names (list of str, optional): the new dimension names, in case the
                batch_size is modified.
            inplace (bool, optional): if True, changes are made in-place.
                Default is False. This is a keyword only argument.
            default (Any, optional): default value for missing entries in the
                other tensordicts. If not provided, missing entries will
                raise a `KeyError`.
            filter_empty (bool, optional): if ``True``, empty tensordicts will be
                filtered out. This also comes with a lower computational cost as
                empty data structures won't be created and destroyed. If ``False``,
                empty tensordicts are kept. Defaults to ``None``, which behaves like
                ``True`` (tensordicts left empty by ``fn`` are filtered out, and ``None``
                is returned if ``fn`` returns no value at all) except that tensordicts
                that were already empty are kept.
            propagate_lock (bool, optional): if ``True``, a locked tensordict will produce
                another locked tensordict. Defaults to ``False``.
            call_on_nested (bool, optional): if ``True``, the function will be called on first-level tensors
                and containers (TensorDict or tensorclass). In this scenario, ``func`` is responsible of
                propagating its calls to nested levels. This allows a fine-grained behaviour
                when propagating the calls to nested tensordicts.
                If ``False``, the function will only be called on leaves, and ``apply`` will take care of dispatching
                the function to all leaves.

                    >>> td = TensorDict({"a": {"b": [0.0, 1.0]}, "c": [1.0, 2.0]})
                    >>> def mean_tensor_only(val):
                    ...     if is_tensor_collection(val):
                    ...         raise RuntimeError("Unexpected!")
                    ...     return val.mean()
                    >>> td_mean = td.apply(mean_tensor_only)
                    >>> def mean_any(val):
                    ...     if is_tensor_collection(val):
                    ...         # Recurse
                    ...         return val.apply(mean_any, call_on_nested=True)
                    ...     return val.mean()
                    >>> td_mean = td.apply(mean_any, call_on_nested=True)

            out (TensorDictBase, optional): a tensordict where to write the results. This can be used to avoid
                creating a new tensordict:

                    >>> td = TensorDict({"a": 0})
                    >>> td.apply(lambda x: x+1, out=td)
                    >>> assert (td==1).all()

                .. warning::
                    If the operation executed on the tensordict requires multiple keys to be accessed for
                    a single computation, providing an ``out`` argument equal to ``self`` can cause the operation
                    to provide silently wrong results.
                    For instance:

                        >>> td = TensorDict({"a": 1, "b": 1})
                        >>> td.apply(lambda x: x+td["a"])["b"] # Right!
                        tensor(2)
                        >>> td.apply(lambda x: x+td["a"], out=td)["b"] # Wrong!
                        tensor(3)

            **constructor_kwargs: additional keyword arguments to be passed to the
                TensorDict constructor.

        Returns:
            a new tensordict with transformed_in tensors.

        Example:
            >>> td = TensorDict({
            ...     "a": -torch.ones(3),
            ...     "nested": {"a": torch.ones(3), "b": torch.zeros(3)}},
            ...     batch_size=[3])
            >>> def name_filter(name, tensor):
            ...     if name == "a":
            ...         return tensor
            >>> td.named_apply(name_filter)
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                    nested: TensorDict(
                        fields={
                            a: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([3]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3]),
                device=None,
                is_shared=False)
            >>> def name_filter(name, *tensors):
            ...     if name == "a":
            ...         r = 0
            ...         for tensor in tensors:
            ...             r = r + tensor
            ...         return tensor
            >>> out = td.named_apply(name_filter, td)
            >>> print(out)
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                    nested: TensorDict(
                        fields={
                            a: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([3]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3]),
                device=None,
                is_shared=False)
            >>> print(out["a"])
            tensor([-1., -1., -1.])

        .. note::
            If ``None`` is returned by the function, the entry is ignored. This
            can be used to filter the data in the tensordict:

            >>> td = TensorDict({"1": 1, "2": 2, "b": {"2": 2, "1": 1}}, [])
            >>> def name_filter(name, tensor):
            ...     if name == "1":
            ...         return tensor
            >>> td.named_apply(name_filter)
            TensorDict(
                fields={
                    1: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                    b: TensorDict(
                        fields={
                            1: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)

        """
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
            if batch_size is not None and batch_size != out.batch_size:
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
                    out=out,
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
        **constructor_kwargs,
    ) -> Self | None:
        from tensordict._td import _SubTensorDict

        if inplace:
            result = self
            is_locked = result.is_locked
        elif out is not None:
            result = out
            if out.is_locked:
                raise RuntimeError(_LOCK_ERROR)
            is_locked = False
            if batch_size is not None and batch_size != out.batch_size:
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
        is_sub_td = isinstance(self, _SubTensorDict)

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
                    **constructor_kwargs,
                )
            else:
                # Pass-through values (e.g., UnbatchedTensor) with shape-changing ops
                # (indicated by batch_size being set) keep their payload unchanged
                # but must expose the new TensorDict-facing batch metadata.
                # For other ops (data ops like zero_), apply the function normally.
                if batch_size is not None and _is_unbatched(item):
                    item_trsf = item._with_batch_size(batch_size)
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

        This method does not run any check after performing the func. This
        means that one to make sure that the metadata of the resulting tensors
        (device, shape etc.) match the :meth:`~.apply` ones.

        """
        if num_threads:

            def func(*args, **kwargs):
                return self._multithread_apply_nest(
                    *args, **kwargs, num_threads=num_threads
                )

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
        keys = list(self.keys())
        for key in keys:
            leaf = self._get_str(key, default=None)
            if leaf is None:
                continue
            if _is_unbatched(leaf):
                if new_batch_size is not None and leaf.batch_size != new_batch_size:
                    self._set_str(
                        key,
                        leaf._with_batch_size(new_batch_size),
                        validated=True,
                        inplace=False,
                    )
                continue
            if _is_tensor_collection(type(leaf)):
                nested_fn(leaf)
                continue
            new_leaf = leaf_fn(leaf)
            del leaf
            self._set_str(key, new_leaf, validated=True, inplace=False)
        if new_batch_size is not None and tuple(new_batch_size) != tuple(
            self.batch_size
        ):
            self._change_batch_size(torch.Size(new_batch_size))
        return self

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
        """Maps a function to splits of the tensordict across one dimension.

        This method will apply a function to a tensordict instance by chunking
        it in tensordicts of equal size and dispatching the operations over the
        desired number of workers.

        The function signature should be ``Callabe[[TensorDict], TensorDict | Tensor]``.
        The output must support the :func:`torch.cat` operation. The function
        must be serializable.

        .. note::
            This method is particularly useful when working with large
            datasets stored on disk (e.g. memory-mapped tensordicts) where
            chunks will be zero-copied slices of the original data which can
            be passed to the processes with virtually zero-cost. This allows
            to tread very large datasets (eg. over a Tb big) to be processed
            at little cost.

        Args:
            fn (callable): function to apply to the tensordict.
                Signatures similar to ``Callabe[[TensorDict], TensorDict | Tensor]``
                are supported.
            dim (int, optional): the dim along which the tensordict will be chunked.
            num_workers (int, optional): the number of workers. Exclusive with ``pool``.
                If none is provided, the number of workers will be set to the
                number of cpus available.

        Keyword Args:
            out (TensorDictBase, optional): an optional container for the output.
                Its batch-size along the ``dim`` provided must match ``self.ndim``.
                If it is shared or memmap (:meth:`~.is_shared` or :meth:`~.is_memmap`
                returns ``True``) it will be populated within the remote processes,
                avoiding data inward transfers. Otherwise, the data from the ``self``
                slice will be sent to the process, collected on the current process
                and written inplace into ``out``.
            chunksize (int, optional): The size of each chunk of data.
                A ``chunksize`` of 0 will unbind the tensordict along the
                desired dimension and restack it after the function is applied,
                whereas ``chunksize>0`` will split the tensordict and call
                :func:`torch.cat` on the resulting list of tensordicts.
                If none is provided, the number of chunks will equate the number
                of workers. For very large tensordicts, such large chunks
                may not fit in memory for the operation to be done and
                more chunks may be needed to make the operation practically
                doable. This argument is exclusive with ``num_chunks``.
            num_chunks (int, optional): the number of chunks to split the tensordict
                into. If none is provided, the number of chunks will equate the number
                of workers. For very large tensordicts, such large chunks
                may not fit in memory for the operation to be done and
                more chunks may be needed to make the operation practically
                doable. This argument is exclusive with ``chunksize``.
            pool (mp.Pool, optional): a multiprocess Pool instance to use
                to execute the job. If none is provided, a pool will be created
                within the ``map`` method.
            generator (torch.Generator, optional): a generator to use for seeding.
                A base seed will be generated from it, and each worker
                of the pool will be seeded with the provided seed incremented
                by a unique integer from ``0`` to ``num_workers``. If no generator
                is provided, a random integer will be used as seed.
                To work with unseeded workers, a pool should be created separately
                and passed to :meth:`map` directly.

                .. note::
                  Caution should be taken when providing a low-valued seed as
                  this can cause autocorrelation between experiments, example:
                  if 8 workers are asked and the seed is 4, the workers seed will
                  range from 4 to 11. If the seed is 5, the workers seed will range
                  from 5 to 12. These two experiments will have an overlap of 7
                  seeds, which can have unexpected effects on the results.

                .. note::
                  The goal of seeding the workers is to have independent seed on
                  each worker, and NOT to have reproducible results across calls
                  of the `map` method. In other words, two experiments may and
                  probably will return different results as it is impossible to
                  know which worker will pick which job. However, we can make sure
                  that each worker has a different seed and that the pseudo-random
                  operations on each will be uncorrelated.

            max_tasks_per_child (int, optional): the maximum number of jobs picked
                by every child process. Defaults to ``None``, i.e., no restriction
                on the number of jobs.
            worker_threads (int, optional): the number of threads for the workers.
                Defaults to ``1``.
            index_with_generator (bool, optional): if ``True``, the splitting / chunking
                of the tensordict will be done during the query, sparing init time.
                Note that :meth:`~.chunk` and :meth:`~.split` are much more
                efficient than indexing (which is used within the generator)
                so a gain of processing time at init time may have a negative
                impact on the total runtime. Defaults to ``False``.
            pbar (bool, optional): if ``True``, a progress bar will be displayed.
                Requires tqdm to be available. Defaults to ``False``.
            mp_start_method (str, optional): the start method for multiprocessing.
                If not provided, the default start method will be used.
                Accepted strings are ``"fork"`` and ``"spawn"``. Keep in mind that
                ``"cuda"`` tensors cannot be shared between processes with the
                ``"fork"`` start method. This is without effect if the ``pool``
                is passed to the ``map`` method.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>>
            >>> def process_data(data):
            ...     data.set("y", data.get("x") + 1)
            ...     return data
            >>> if __name__ == "__main__":
            ...     data = TensorDict({"x": torch.zeros(1, 1_000_000)}, [1, 1_000_000]).memmap_()
            ...     data = data.map(process_data, dim=1)
            ...     print(data["y"][:, :10])
            ...
            tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
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
        """Maps a function to splits of the tensordict across one dimension iteratively.

        This is the iterable version of :meth:`~TensorDictBase.map`.

        This method will apply a function to a tensordict instance by chunking
        it in tensordicts of equal size and dispatching the operations over the
        desired number of workers. It will yield the results one at a time.

        The function signature should be ``Callabe[[TensorDict], TensorDict | Tensor]``.
        The function must be serializable.

        .. note::
            This method is particularly useful when working with large
            datasets stored on disk (e.g. memory-mapped tensordicts) where
            chunks will be zero-copied slices of the original data which can
            be passed to the processes with virtually zero-cost. This allows
            to tread very large datasets (eg. over a Tb big) to be processed
            at little cost.

        .. note::
            This function be used to represent a dataset and load from it,
            in a dataloader-like fashion.

        Args:
            fn (callable): function to apply to the tensordict.
                Signatures similar to ``Callabe[[TensorDict], TensorDict | Tensor]``
                are supported.
            dim (int, optional): the dim along which the tensordict will be chunked.
            num_workers (int, optional): the number of workers. Exclusive with ``pool``.
                If none is provided, the number of workers will be set to the
                number of cpus available.

        Keyword Args:
            shuffle (bool, optional): whether the indices should be globally shuffled.
                If ``True``, each batch will contain non-contiguous samples.
                If ``index_with_generator=False`` and `shuffle=True``, an error will be raised.
                Defaults to ``False``.
            chunksize (int, optional): The size of each chunk of data.
                A ``chunksize`` of 0 will unbind the tensordict along the
                desired dimension and restack it after the function is applied,
                whereas ``chunksize>0`` will split the tensordict and call
                :func:`torch.cat` on the resulting list of tensordicts.
                If none is provided, the number of chunks will equate the number
                of workers. For very large tensordicts, such large chunks
                may not fit in memory for the operation to be done and
                more chunks may be needed to make the operation practically
                doable. This argument is exclusive with ``num_chunks``.
            num_chunks (int, optional): the number of chunks to split the tensordict
                into. If none is provided, the number of chunks will equate the number
                of workers. For very large tensordicts, such large chunks
                may not fit in memory for the operation to be done and
                more chunks may be needed to make the operation practically
                doable. This argument is exclusive with ``chunksize``.
            pool (mp.Pool, optional): a multiprocess Pool instance to use
                to execute the job. If none is provided, a pool will be created
                within the ``map`` method.
            generator (torch.Generator, optional): a generator to use for seeding.
                A base seed will be generated from it, and each worker
                of the pool will be seeded with the provided seed incremented
                by a unique integer from ``0`` to ``num_workers``. If no generator
                is provided, a random integer will be used as seed.
                To work with unseeded workers, a pool should be created separately
                and passed to :meth:`map` directly.

                .. note::
                  Caution should be taken when providing a low-valued seed as
                  this can cause autocorrelation between experiments, example:
                  if 8 workers are asked and the seed is 4, the workers seed will
                  range from 4 to 11. If the seed is 5, the workers seed will range
                  from 5 to 12. These two experiments will have an overlap of 7
                  seeds, which can have unexpected effects on the results.

                .. note::
                  The goal of seeding the workers is to have independent seed on
                  each worker, and NOT to have reproducible results across calls
                  of the `map` method. In other words, two experiments may and
                  probably will return different results as it is impossible to
                  know which worker will pick which job. However, we can make sure
                  that each worker has a different seed and that the pseudo-random
                  operations on each will be uncorrelated.

            max_tasks_per_child (int, optional): the maximum number of jobs picked
                by every child process. Defaults to ``None``, i.e., no restriction
                on the number of jobs.
            worker_threads (int, optional): the number of threads for the workers.
                Defaults to ``1``.
            index_with_generator (bool, optional): if ``True``, the splitting / chunking
                of the tensordict will be done during the query, sparing init time.
                Note that :meth:`~.chunk` and :meth:`~.split` are much more
                efficient than indexing (which is used within the generator)
                so a gain of processing time at init time may have a negative
                impact on the total runtime. Defaults to ``True``.

                .. note::
                    The default value of ``index_with_generator`` differs for ``map_iter``
                    and ``map`` and the former assumes that it is prohibitively expensive to
                    store a split version of the TensorDict in memory.

            pbar (bool, optional): if ``True``, a progress bar will be displayed.
                Requires tqdm to be available. Defaults to ``False``.
            mp_start_method (str, optional): the start method for multiprocessing.
                If not provided, the default start method will be used.
                Accepted strings are ``"fork"`` and ``"spawn"``. Keep in mind that
                ``"cuda"`` tensors cannot be shared between processes with the
                ``"fork"`` start method. This is without effect if the ``pool``
                is passed to the ``map`` method.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>>
            >>> def process_data(data):
            ...     data.unlock_()
            ...     data.set("y", data.get("x") + 1)
            ...     return data
            >>> if __name__ == "__main__":
            ...     data = TensorDict({"x": torch.zeros(1, 1_000_000)}, [1, 1_000_000]).memmap_()
            ...     for sample in data.map_iter(process_data, dim=1, chunksize=5):
            ...         print(sample["y"])
            ...         break
            ...
            tensor([[1., 1., 1., 1., 1.]])

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
        )
        if not index_with_generator:
            length = len(self_split)
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

            fn, self_split = wrap_fn_with_out(fn, out)
            out = None

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
        """Copies the tensordict without cloning its tensors."""
        return self.copy()

    def norm(
        self,
        *,
        out=None,
        dtype: torch.dtype | None = None,
        key_transform: Callable[[NestedKey], NestedKey] | None = None,
    ) -> Self:
        """Computes the norm of each tensor in the tensordict.

        Keyword Args:
            out (TensorDict, optional): the output tensordict.
            dtype (torch.dtype, optional): the output dtype.
            key_transform (Callable[[NestedKey], NestedKey], optional): A function to transform key names.
                If provided, all keys in the result will be transformed using this function.
                For string keys, the function receives a string. For tuple keys, it receives a tuple.
                Default: ``None``.

        """
        keys, vals = self._items_list(True, True, collapse=True)
        vals = torch._foreach_norm(vals, dtype=dtype)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        result = self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            batch_size=[],
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
                    torch._foreach_add(tuple(foreach_vals.values()), 0),
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
            array = np.asarray(array)
            castable = array.dtype.kind in ("c", "i", "f", "b", "u")
        elif hasattr(array, "numpy"):
            # tf.Tensor with no shape can't be converted otherwise
            array = array.numpy()
            castable = array.dtype.kind in ("c", "i", "f", "b", "u")
        if castable:
            if hasattr(array, "flags") and not array.flags.writeable:
                array = array.copy()
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
                value = value._with_batch_size(self.batch_size)
            return value
        batch_size = self.batch_size
        if check_shape and _shape(value)[: self.batch_dims] != batch_size:
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
                value = value._with_batch_size(self.batch_size)
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
                value = value._with_batch_size(self.batch_size)
            return value

        batch_size = self.batch_size
        if check_shape and _shape(value)[: self.batch_dims] != batch_size:
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
                value = value._with_batch_size(self.batch_size)
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
        if exc_type is not None and issubclass(exc_type, Exception):
            return False
        is_tc = _is_tensorclass(type(self))
        _last_op = (
            self._last_op_queue.pop()
            if not is_tc
            else self._tensordict._last_op_queue.pop()
        )
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
        """Clears the weakrefs in order for the tensordict to get out of the compile region safely.

        Use this whenever you hit `torch._dynamo.exc.Unsupported: reconstruct: WeakRefVariable()`
        before returning a TensorDict.

        Returns: self
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
        """Selects the keys of the tensordict and returns a new tensordict with only the selected keys.

        The values are not copied: in-place modifications a tensor of either
        of the original or new tensordict will result in a change in both
        tensordicts.

        Args:
            *keys (str): keys to select
            inplace (bool): if True, the tensordict is pruned in place.
                Default is ``False``.
            strict (bool, optional): whether selecting a key that is not present
                will return an error or not. Default: :obj:`True`.
            as_tensordict (bool, optional): if ``True``, the result will be a
                plain :class:`~tensordict.TensorDict` even when called on a
                :class:`~tensordict.TensorClass` instance. This avoids the
                TensorClass wrapper that fills unselected fields with ``None``.
                For :class:`~tensordict.TensorDictBase` subclasses, this is a
                no-op. Default: ``False``.

        Returns:
            A new tensordict (or the same if ``inplace=True``) with the selected keys only.

        .. note::
            To select keys in a tensordict and return a version of this tensordict
            deprived of these keys, see the :meth:`~.split_keys` method.

        Examples:
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": 0, "b": {"c": 1, "d": 2}}, [])
            >>> td.select("a", ("b", "c"))
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> td.select("a", "b")
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                            d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> td.select("this key does not exist", strict=False)
            TensorDict(
                fields={
                },
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
        """
        keys = unravel_key_list(keys)
        result = self._select(*keys, inplace=inplace, strict=strict)
        if not inplace and (result._is_memmap or result._is_shared):
            result.lock_()
        return result

    @abc.abstractmethod
    def _select(
        self,
        *keys: NestedKey,
        inplace: bool = False,
        strict: bool = True,
        set_shared: bool = True,
    ) -> Self:
        raise NotImplementedError

    def exclude(self, *keys: NestedKey, inplace: bool = False) -> Self:
        """Excludes the keys of the tensordict and returns a new tensordict without these entries.

        The values are not copied: in-place modifications a tensor of either
        of the original or new tensordict will result in a change in both
        tensordicts.

        Args:
            *keys (str): keys to exclude.
            inplace (bool): if True, the tensordict is pruned in place.
                Default is ``False``.

        Returns:
            A new tensordict (or the same if ``inplace=True``) without the excluded entries.

        Examples:
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": 0, "b": {"c": 1, "d": 2}}, [])
            >>> td.exclude("a", ("b", "c"))
            TensorDict(
                fields={
                    b: TensorDict(
                        fields={
                            d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> td.exclude("a", "b")
            TensorDict(
                fields={
                },
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)

        """
        keys = unravel_key_list(keys)
        result = self._exclude(*keys, inplace=inplace)
        if not inplace and (result._is_memmap or result._is_shared):
            result.lock_()
        return result

    @abc.abstractmethod
    def _exclude(
        self,
        *keys: NestedKey,
        inplace: bool = False,
        set_shared: bool = True,
    ) -> Self:
        raise NotImplementedError

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
        """Clones a TensorDictBase subclass instance onto a new TensorDictBase subclass of the same type.

        To create a TensorDict instance from any other TensorDictBase subtype, call the :meth:`~.to_tensordict` method
        instead.

        Args:
            recurse (bool, optional): if ``True``, each tensor contained in the
                TensorDict will be copied too. Otherwise only the TensorDict
                tree structure will be copied. Defaults to ``True``.

        .. note::
            Unlike many other ops (pointwise arithmetic, shape operations, ...) ``clone`` does not inherit the
            original lock attribute. This design choice is made such that a clone can be created to be modified,
            which is the most frequent usage.

        """
        result = self._clone(recurse=recurse, **kwargs)
        if not recurse and (result._is_shared or result._is_memmap):
            result.lock_()
        return result

    @abc.abstractmethod
    def _clone(self, recurse: bool = False):
        raise NotImplementedError

    def as_tensor(self) -> Self:
        """Converts every leaf of a tensordict to a plain torch.Tensor."""

        def as_tensor(tensor):
            try:
                return tensor.as_tensor()
            except AttributeError:
                return tensor

        return self._fast_apply(as_tensor, propagate_lock=True)

    _CONFLICTING_BATCH_SIZES = "Conflicting batch sizes in {}: batch_size and auto_batch_size cannot be both specified."

    def empty(
        self, recurse=False, *, batch_size=None, device=NO_DEFAULT, names=None
    ) -> Self:  # noqa: D417
        """Returns a new, empty tensordict with the same device and batch size.

        Args:
            recurse (bool, optional): if ``True``, the entire structure of the
                ``TensorDict`` will be reproduced without content.
                Otherwise, only the root will be duplicated.
                Defaults to ``False``.

        Keyword Args:
            batch_size (torch.Size, optional): a new batch-size for the tensordict.
            device (torch.device, optional): a new device.
            names (list of str, optional): dimension names.

        """
        if not recurse:
            result = self._select(set_shared=False)
        else:
            # simply exclude the leaves. A comprehension walks the keys once:
            # unpacking the view would call its __len__, which walks them too.
            leaves = [key for key in self.keys(True, True)]  # noqa: C416
            result = self._exclude(*leaves, set_shared=False)
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
    @abc.abstractmethod
    def masked_fill_(self, mask: Tensor, value: float | bool) -> Self:
        """Fills the values corresponding to the mask with the desired value.

        Args:
            mask (boolean torch.Tensor): mask of values to be filled. Shape
                must match the tensordict batch-size.
            value: value to used to fill the tensors.

        Returns:
            self

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3, 4)},
            ...     batch_size=[3])
            >>> mask = torch.tensor([True, False, False])
            >>> td.masked_fill_(mask, 1.0)
            >>> td.get("a")
            tensor([[1., 1., 1., 1.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.]])
        """
        raise NotImplementedError

    @abc.abstractmethod
    def masked_fill(self, mask: Tensor, value: float | bool) -> Self:
        """Out-of-place version of masked_fill.

        Args:
            mask (boolean torch.Tensor): mask of values to be filled. Shape
                must match the tensordict batch-size.
            value: value to used to fill the tensors.

        Returns:
            self

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3, 4)},
            ...     batch_size=[3])
            >>> mask = torch.tensor([True, False, False])
            >>> td1 = td.masked_fill(mask, 1.0)
            >>> td1.get("a")
            tensor([[1., 1., 1., 1.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.]])
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _change_batch_size(self, new_size: torch.Size) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def is_contiguous(self) -> bool:
        """Returns a boolean indicating if all the tensors are contiguous."""
        raise NotImplementedError

    @abc.abstractmethod
    def contiguous(self, *, canonical: bool = False, inplace: bool = False) -> Self:
        """Returns a new tensordict of the same type with contiguous values (or self if values are already contiguous).

        Args:
            canonical (bool, optional): if ``True``, every dense tensor leaf
                whose strides do not match the canonical C-row-major strides
                for its shape is rematerialized into a freshly allocated
                contiguous tensor (using ``torch.contiguous_format``), even if
                :meth:`torch.Tensor.is_contiguous` returns ``True`` (e.g.
                because of size-1 dimensions). When ``False`` (the default),
                the historical behavior of :meth:`torch.Tensor.contiguous` is
                preserved. Defaults to ``False``.
            inplace (bool, optional): If ``True``, this tensordict's identity
                and key set are preserved; each non-contiguous leaf is
                replaced by its contiguous counterpart one at a time.
                ``Tensor.contiguous()`` returns ``self`` when a leaf is
                already contiguous, so for fully-contiguous tensordicts the
                operation is effectively a no-op aside from the key walk.
                Defaults to ``False``.

        """
        raise NotImplementedError

    @_cache_while_locked  # noqa: B019
    @_as_context_manager()
    def flatten_keys(
        self,
        separator: str = ".",
        inplace: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
    ) -> Self:
        """Converts a nested tensordict into a flat one, recursively.

        The TensorDict type will be lost and the result will be a simple TensorDict instance.

        Args:
            separator (str, optional): the separator between the nested items.
            inplace (bool, optional): if ``True``, the resulting tensordict will
                have the same identity as the one where the call has been made.
                Defaults to ``False``.
            is_leaf (callable, optional): a callable over a class type returning
                a bool indicating if this class has to be considered as a leaf.

                .. note:: The purpose of `is_leaf` is not to prevent recursive calls into nested tensordicts, but
                    rather to mark certain types as "leaves" for the purpose of filtering when `leaves_only=True`.
                    Even if `is_leaf(cls)` returns `True`, the nested structure of the tensordict will still be
                    traversed if `include_nested=True`.
                    In other words, `is_leaf` does not control the recursion depth, but rather provides a way to filter
                    out certain types from the result when `leaves_only=True`. This means that a node in the tree can
                    be both a leaf and a node with children.
                    In practice, the default value of ``is_leaf`` does exclude tensordict and tensorclass instances
                    from the leaf set.

                .. seealso:: :meth:`~tensordict.is_leaf_nontensor` and :meth:`~tensordict.default_is_leaf`.

        Examples:
            >>> data = TensorDict({"a": 1, ("b", "c"): 2, ("e", "f", "g"): 3}, batch_size=[])
            >>> data.flatten_keys(separator=" - ")
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                    b - c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                    e - f - g: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)

        This method and :meth:`~.unflatten_keys` are particularly useful when
        handling state-dicts, as they make it possible to seamlessly convert
        flat dictionaries into data structures that mimic the structure of the
        model.

        Examples:
            >>> model = torch.nn.Sequential(torch.nn.Linear(3 ,4))
            >>> ddp_model = torch.ao.quantization.QuantWrapper(model)
            >>> state_dict = TensorDict(ddp_model.state_dict(), batch_size=[]).unflatten_keys(".")
            >>> print(state_dict)
            TensorDict(
                fields={
                    module: TensorDict(
                        fields={
                            0: TensorDict(
                                fields={
                                    bias: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
                                    weight: Tensor(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
                                batch_size=torch.Size([]),
                                device=None,
                                is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> model_state_dict = state_dict.get("module")
            >>> print(model_state_dict)
            TensorDict(
                fields={
                    0: TensorDict(
                        fields={
                            bias: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
                            weight: Tensor(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> model.load_state_dict(dict(model_state_dict.flatten_keys(".")))
        """
        if inplace:
            return self._flatten_keys_inplace(separator=separator, is_leaf=is_leaf)
        return self._flatten_keys_outplace(separator=separator, is_leaf=is_leaf)

    def _flatten_keys_outplace(self, separator, is_leaf):
        if is_leaf is None:
            is_leaf = _is_leaf_nontensor
        all_leaves_all_vals = zip(
            *self.items(include_nested=True, leaves_only=True, is_leaf=is_leaf)
        )
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
        all_leaves = [
            _unravel_key_to_tuple(key)
            for key in self.keys(include_nested=True, leaves_only=True, is_leaf=is_leaf)
        ]
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
        # we will need to remove the empty tensordicts later on
        root_keys = set(self.keys())
        for leaf, leaf_flat in zip(all_leaves, all_leaves_flat):
            self.rename_key_(leaf, leaf_flat)
            if isinstance(leaf, str):
                root_keys.discard(leaf)
        self.exclude(*root_keys, inplace=True)
        return self

    @_cache_while_locked  # noqa: B019
    @_as_context_manager()
    def unflatten_keys(self, separator: str = ".", inplace: bool = False) -> Self:
        """Converts a flat tensordict into a nested one, recursively.

        The TensorDict type will be lost and the result will be a simple TensorDict instance.
        The metadata of the nested tensordicts will be inferred from the root:
        all instances across the data tree will share the same batch-size,
        dimension names and device.

        Args:
            separator (str, optional): the separator between the nested items.
            inplace (bool, optional): if ``True``, the resulting tensordict will
                have the same identity as the one where the call has been made.
                Defaults to ``False``.

        Examples:
            >>> data = TensorDict({"a": 1, "b - c": 2, "e - f - g": 3}, batch_size=[])
            >>> data.unflatten_keys(separator=" - ")
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False),
                    e: TensorDict(
                        fields={
                            f: TensorDict(
                                fields={
                                    g: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
                                batch_size=torch.Size([]),
                                device=None,
                                is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)

        This method and :meth:`~.unflatten_keys` are particularly useful when
        handling state-dicts, as they make it possible to seamlessly convert
        flat dictionaries into data structures that mimic the structure of the
        model.

        Examples:
            >>> model = torch.nn.Sequential(torch.nn.Linear(3 ,4))
            >>> ddp_model = torch.ao.quantization.QuantWrapper(model)
            >>> state_dict = TensorDict(ddp_model.state_dict(), batch_size=[]).unflatten_keys(".")
            >>> print(state_dict)
            TensorDict(
                fields={
                    module: TensorDict(
                        fields={
                            0: TensorDict(
                                fields={
                                    bias: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
                                    weight: Tensor(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
                                batch_size=torch.Size([]),
                                device=None,
                                is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> model_state_dict = state_dict.get("module")
            >>> print(model_state_dict)
            TensorDict(
                fields={
                    0: TensorDict(
                        fields={
                            bias: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
                            weight: Tensor(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> model.load_state_dict(dict(model_state_dict.flatten_keys(".")))

        """
        if not inplace:
            result = self._clone(recurse=False).unflatten_keys(
                separator=separator, inplace=True
            )
            if result._is_shared or result._is_memmap:
                result.lock_()
            return result
        else:
            if not is_compiling():
                key_list = list(self.keys())
            else:
                key_list = [k for k in self.keys()]  # noqa

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
        """Splits the tensordict in subsets given one or more set of keys.

        The method will return ``N+1`` tensordicts, where ``N`` is the number of
        the arguments provided.

        Args:
            key_sets (sequence of Dict[in_key, out_key] or list of keys): the various splits.
            inplace (bool, optional): if ``True``, the keys are removed from ``self``
                in-place. Defaults to ``False``.
            default (Any, optional): the value to be returned when a key is missing.
                If not specified and ``strict=True``, an exception is raised.
            strict (bool, optional): if ``True``, an exception is raised when a key
                is missing. Defaults to ``True``.
            reproduce_struct (bool, optional): if ``True``, all tensordict returned have
                the same tree structure as ``self``, even if some sub-tensordicts
                contain no leaves.

        .. note::
            ``None`` non-tensor values will be ignored and not returned.

        .. note::
            The method does not check for duplicates in the provided lists.

        Examples:
            >>> td = TensorDict(
            ...     a=0,
            ...     b=0,
            ...     c=0,
            ...     d=0,
            ... )
            >>> td_a, td_bc, td_d = td.split_keys(["a"], ["b", "c"])
            >>> print(td_bc)
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
        for key_set in key_sets:
            outs.append(self.empty(recurse=reproduce_struct))
            if not isinstance(key_set, dict):
                key_set = {key: key for key in key_set}
            for key in key_set:
                val = last_out.pop(key, default)
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
        """Separates the specified keys from the tensordict in-place.

        .. seealso:: This method is equivalent to calling :meth:`~tensordict.TensorDictBase.split_keys` with
            ``inplace=True`` on a single split.

        .. seealso:: This method is equivalent to calling :meth:`~tensordict.TensorDictBase.exclude` except that it
            returns the other split of the data.

        Args:
            keys (NestedKey): the keys to separate from the tensordict.
            default (Any, optional): the value to be returned when a key is missing.
                If not specified and ``strict=True``, an exception is raised. Otherwise, the default of any missing key
                will be ``None`` unless specified otherwise.
            strict (bool, optional): if ``True``, an exception is raised when a key
                is missing. Defaults to ``True``.
            filter_empty (bool, optional): if ``True``, empty tensordicts within ``self`` will be removed.
                Defaults to ``True``.

        Returns:
            T: the separated tensordict.

        Examples:
            >>> td = TensorDict(
            ...     a=0,
            ...     b=0,
            ...     c=0,
            ...     d=0,
            ... )
            >>> td_a_c = td.separates("a", "c")
            >>> print(td_a_c)
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                    c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> print(td)
            TensorDict(
                fields={
                    b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                    d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)

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
        from tensordict._td import TensorDict

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
            batch_size = _getitem_batch_size(batch_size, index)

        if names is None:
            names = self._get_names_idx(index)

        source = {}
        for key, item in self.items():
            if _is_unbatched(item):
                source[key] = item._with_batch_size(batch_size)
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
        return self._is_locked

    @is_locked.setter
    def is_locked(self, value: bool) -> None:
        if value:
            self.lock_()
        else:
            self.unlock_()

    def _propagate_lock(self, lock_parents_weakrefs=None, *, is_compiling):
        """Registers the parent tensordict that handles the lock."""
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
            # Build a per-TD locked schema so locked-fast-paths can walk
            # an immutable key tuple instead of iterating ``_tensordict``
            # under Dynamo (which emits DICT_KEYS_MATCH guards).
            # Skip under compile: building the schema would itself iterate
            # the backing dict and add the very guard we're trying to
            # avoid. Users who want the fast path must lock_() in eager
            # mode before entering the compiled region.
            self._build_lock_schema()

        for value in self.values():
            if _is_tensor_collection(type(value)):
                value._propagate_lock(lock_parents_weakrefs, is_compiling=is_compiling)

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
        """Locks a tensordict for non in-place operations.

        Functions such as :meth:`~.set`, :meth:`~.__setitem__`, :meth:`~.update`,
        :meth:`~.rename_key_` or other operations that add or remove entries
        will be blocked.

        This method can be used as a decorator.

        Example:
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": 1, "b": 2, "c": 3}, batch_size=[])
            >>> with td.lock_():
            ...     assert td.is_locked
            ...     try:
            ...         td.set("d", 0) # error!
            ...     except RuntimeError:
            ...         print("td is locked!")
            ...     try:
            ...         del td["d"]
            ...     except RuntimeError:
            ...         print("td is locked!")
            ...     try:
            ...         td.rename_key_("a", "d")
            ...     except RuntimeError:
            ...         print("td is locked!")
            ...     td.set("a", 0, inplace=True)  # No storage is added, moved or removed
            ...     td.set_("a", 0) # No storage is added, moved or removed
            ...     td.update({"a": 0}, inplace=True)  # No storage is added, moved or removed
            ...     td.update_({"a": 0})  # No storage is added, moved or removed
            >>> assert not td.is_locked
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
        for value in self.values():
            if _is_tensor_collection(type(value)):
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
        """Unlocks a tensordict for non in-place operations.

        Can be used as a decorator.

        See :meth:`~.lock_` for more details.
        """
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
        """Return a deviceless tensordict whose leaves are :class:`~tensordict.TensorAttrs`.

        Each tensor leaf of ``self`` is replaced by a :class:`~tensordict.TensorAttrs`
        describing the requested tensor attributes. The result is intended to be passed to
        :meth:`to` to drive per-leaf device/dtype casting when the source tensordict
        is heterogeneous.

        Keyword Args:
            fields (sequence of str, optional): which attributes to record on each
                :class:`~tensordict.TensorAttrs`. Accepts any subset of
                ``("device", "dtype", "shape")``. Attributes not listed remain ``None``.
                Defaults to ``("device", "dtype", "shape")``.
            num_threads (int or None, optional): number of threads to use when
                iterating leaves. Defaults to ``None`` (single-threaded). Construction
                of :class:`TensorAttrs` is Python-bound, so threading typically yields
                little; exposed for symmetry with :meth:`to`.

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
            >>> out = target.to(attrs)   # casts `b` to int32 per-leaf
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

        spec: dict = {}
        for key, val in attrs_td.items(
            include_nested=True, leaves_only=True, is_leaf=_is_attrs_leaf
        ):
            if isinstance(val, TensorAttrs):
                spec[key] = val

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
        """Checks if all tensors in the tensordict are floating point."""
        for item in self.values(include_nested=True, leaves_only=True):
            if not item.is_floating_point():
                return False
        else:
            return True

    # Gradient compatibility
    @property
    def requires_grad(self) -> bool:
        return any(v.requires_grad for v in self.values())

    def requires_grad_(self, requires_grad=True) -> Self:
        """Change if autograd should record operations on this tensor: sets this tensor's requires_grad attribute in-place.

        Returns this tensordict.

        Args:
            requires_grad (bool, optional): whether or not autograd should record operations on this tensordict.
                Defaults to ``True``.

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
        """Computes the gradient of the differentiable leaves of the tensordict w.r.t. graph leaves.

        This method mirrors :meth:`torch.Tensor.backward`: it recursively collects the
        leaf tensors that require gradients and differentiates the graph through a single
        :func:`torch.autograd.backward` call. Non-differentiable leaves are ignored.

        If ``gradient`` is ``None``, every differentiable leaf must be scalar (a single
        element) and an implicit gradient of ``1`` is used, such that for scalar leaves
        ``td.backward()`` is equivalent to ``td.sum(reduce=True).backward()``. Otherwise,
        ``gradient`` must be a tensordict whose entries match the differentiable leaves
        by nested key. Leaves may have heterogeneous shapes, dtypes and devices, as long
        as each gradient entry matches the leaf it is associated with.

        Args:
            gradient (TensorDictBase, optional): the gradient w.r.t. the tensordict.
                Entries are matched to the differentiable leaves by nested key and must
                have the same shape as the leaf they match. Can be omitted if all
                differentiable leaves are scalar. Defaults to ``None``.
            retain_graph (bool, optional): if ``False``, the graph used to compute the
                grads will be freed. Defaults to the value of ``create_graph``.
            create_graph (bool, optional): if ``True``, graph of the derivative will be
                constructed, allowing to compute higher order derivative products.
                Defaults to ``False``.
            inputs (TensorDictBase or sequence of Tensor, optional): inputs w.r.t. which
                the gradient will be accumulated in ``.grad``. All other tensors will be
                ignored. If not provided, the gradient is accumulated into all the leaf
                tensors that were used to compute the differentiated tensors.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> x = torch.randn(3, requires_grad=True)
            >>> loss_td = TensorDict(
            ...     actor_loss=x.sum(),
            ...     critic_loss=x.pow(2).sum(),
            ... )
            >>> loss_td.backward()  # implicit gradient of 1 for scalar leaves
            >>> assert x.grad is not None
            >>> # weighted losses through a matching gradient tensordict
            >>> x.grad = None
            >>> loss_td = TensorDict(
            ...     actor_loss=x.sum(),
            ...     critic_loss=x.pow(2).sum(),
            ... )
            >>> weights = TensorDict(
            ...     actor_loss=torch.tensor(0.5),
            ...     critic_loss=torch.tensor(1.0),
            ... )
            >>> loss_td.backward(weights)

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

    @abc.abstractmethod
    def detach_(self) -> Self:
        """Detach the tensors in the tensordict in-place.

        Returns:
            self.

        """
        raise NotImplementedError

    @_cache_while_locked  # noqa: B019
    def detach(self) -> Self:
        """Detach the tensors in the tensordict.

        Returns:
            a new tensordict with no tensor requiring gradient.

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

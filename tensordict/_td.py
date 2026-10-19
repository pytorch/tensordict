# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import weakref
from collections import defaultdict
from collections.abc import Mapping
from numbers import Number
from pathlib import Path
from textwrap import indent
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Sequence,
    Tuple,
    Type,
    TYPE_CHECKING,
)
from warnings import warn

import torch
from tensordict._archive import _memmap_tensor_from_path
from tensordict._base.unsupported import _NoShapeOps
from tensordict._indexing import (
    _entry_index,
    _getitem_batch_size,
    convert_ellipsis_to_idx,
)
from tensordict._nestedkey import NestedKey
from tensordict._tensorcollection import TensorCollection
from tensordict._unbatched import _unbatched_with_batch_size
from tensordict.base import (
    _default_is_leaf,
    _device_recorder,
    _expand_to_match_shape,
    _flatten_nested_dicts,
    _foreach_copy_,
    _foreach_copy_compiled,
    _has_lazy_stack,
    _holds_its_tensors,
    _in_dense_is_empty,
    _InplaceWriteError,
    _is_accepted_class,
    _is_leaf_nontensor,
    _IS_LEAF_WITHOUT_NODES,
    _is_tensor_collection,
    _key_through_leaf_error,
    _load_metadata,
    _members_hold_leaf,
    _NESTED_TENSORS_AS_LISTS,
    _raise_if_not_a_tensordict,
    _register_tensor_class,
    _SCALAR_TYPES,
    _select_entry,
    _SELF_NESTING_ERROR,
    _SET_AT_CAST_TYPES,
    _set_at_each_key,
    _set_subtensordict_type,
    _set_tensordict_type,
    _UNSET,
    _warn_empty_names,
    _warn_lazy_is_empty,
    _warn_scalar_over_non_tensor,
    _warn_set_at_cast,
    _wrap_extra_state,
    _write_back_members,
    BEST_ATTEMPT_INPLACE,
    CompatibleType,
    is_tensor_collection,
    NO_DEFAULT,
    T,
    TensorDictBase,
)
from tensordict.memmap import _raise_if_jagged, MemoryMappedTensor
from tensordict.utils import (
    _as_context_manager,
    _BatchedUninitializedBuffer,
    _BatchedUninitializedParameter,
    _cache_while_locked,
    _canonicalize_tensor,
    _cast_scalar,
    _CHECK_INVARIANTS,
    _check_invariants,
    _create_segments_from_int,
    _encode_key_for_filesystem,
    _GENERIC_NESTED_ERR,
    _get_robust_key_setting_with_warning,
    _import_and_wrap_functorch,
    _infer_size_impl,
    _is_non_tensor,
    _is_safe_legacy_key,
    _is_unbatched,
    _KEY_ERROR,
    _lock_blocked,
    _LOCK_ERROR,
    _LockedSchema,
    _NON_STR_KEY_ERR,
    _pass_through,
    _pass_through_cls,
    _prune_selected_keys,
    _set_item,
    _set_max_batch_size,
    _shape,
    _STR_DTYPE_TO_DTYPE,
    _StringKeys,
    _StringOnlyDict,
    _sub_index,
    _unravel_key_to_tuple,
    _zip_strict,
    DeviceType,
    expand_as_right,
    IndexType,
    is_non_tensor,
    is_tensorclass,
    unravel_key,
    unravel_key_list,
)
from torch import Tensor
from torch._functorch.vmap import _maybe_remove_batch_dim
from torch.compiler import is_compiling
from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX
from torch.utils._pytree import tree_map

try:
    from functorch import dim as ftdim

    _has_funcdim = True
except ImportError:
    from tensordict.utils import _ftdim_mock as ftdim

    _has_funcdim = False

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any

_register_tensor_class(ftdim.Tensor)


_add_batch_dim, _remove_batch_dim = _import_and_wrap_functorch(
    "_add_batch_dim", "_remove_batch_dim"
)

# Device inputs of TensorDict(device=...) -> (indexed device, is_cuda). Only
# inputs whose meaning does not depend on the current CUDA device are kept, and
# a process sees few distinct devices, so the dict stays small.
_PARSED_DEVICES: dict = {}
_PARSED_DEVICES_MAX = 64

# torch.backends.mps.is_available() does not change during a process; read on
# first eager use by _mps_available().
_MPS_AVAILABLE = None


def _parse_device(device: DeviceType, memoize: bool) -> tuple[torch.device, bool]:
    """Parses a ``device`` argument and gives it an index if it has none.

    Returns the device and whether it is a CUDA device. With ``memoize``, the
    result is remembered for ``str`` and ``torch.device`` inputs, except for a
    CUDA device without index, which means the current CUDA device.
    """
    memoize = memoize and (type(device) is str or type(device) is torch.device)
    if memoize:
        parsed = _PARSED_DEVICES.get(device)
        if parsed is not None:
            return parsed
    parsed_device = torch.device(device)
    device_type = parsed_device.type
    is_cuda = device_type == "cuda"
    if parsed_device.index is None:
        if is_cuda:
            return (
                torch.device(device_type, index=torch.cuda.current_device()),
                True,
            )
        if device_type not in ("cpu", "meta"):
            parsed_device = torch.device(device_type, index=0)
    parsed = (parsed_device, is_cuda)
    if memoize and len(_PARSED_DEVICES) < _PARSED_DEVICES_MAX:
        _PARSED_DEVICES[device] = parsed
    return parsed


def _mps_available() -> bool:
    global _MPS_AVAILABLE
    if _MPS_AVAILABLE is None:
        _MPS_AVAILABLE = torch.backends.mps.is_available()
    return _MPS_AVAILABLE


def _nested_members_hold_leaf(td: TensorDict) -> bool:
    """Whether a nested collection of ``td``, which lists no leaf, holds one in the members of a lazy stack (#2807)."""
    for item in td._tensordict.values():
        if type(item) is TensorDict:
            if _nested_members_hold_leaf(item):
                return True
        elif _members_hold_leaf([item]):
            return True
    return False


def _clone_value(value, recurse: bool):
    if recurse:
        # this is not a problem for locked tds as we will not lock it
        return value.clone()
    elif _is_tensor_collection(type(value)):
        return value._clone(recurse=False)
    else:
        return value


def _propagate_lock_dense(td, lock_parents_weakrefs):
    """The eager steps of ``TensorDictBase._propagate_lock`` for an exact TensorDict.

    Each node gets the same lock flag, parent weakrefs and locked schema as
    the generic method gives it. The exact TensorDict entries are locked by
    this function, the other tensor collections by their own
    ``_propagate_lock``, in the order of the entries and with the same list
    of parent weakrefs.
    """
    td._is_locked = True
    td_dict = td.__dict__
    if lock_parents_weakrefs is None:
        lock_parents_weakrefs = [weakref.ref(td)]
    else:
        existing = td_dict.get("__lock_parents_weakrefs")
        if existing:
            lock_parents_weakrefs = [
                ref
                for ref in lock_parents_weakrefs
                if not any(refref is ref for refref in existing)
            ]
            td_dict["__lock_parents_weakrefs"] = existing + lock_parents_weakrefs
        else:
            td_dict["__lock_parents_weakrefs"] = list(lock_parents_weakrefs)
        lock_parents_weakrefs = lock_parents_weakrefs + [weakref.ref(td)]
    entries = td._tensordict
    td_dict["_locked_schema"] = _LockedSchema(keys=tuple(entries))
    for value in entries.values():
        cls = type(value)
        if cls is TensorDict:
            _propagate_lock_dense(value, lock_parents_weakrefs)
        elif _is_tensor_collection(cls):
            value._propagate_lock(lock_parents_weakrefs, is_compiling=False)


class TensorDict(TensorDictBase):
    """A dictionary of tensors that share a batch size.

    The batch size is the leading part of the shape that every entry has in
    common. With ``batch_size=[3, 4]``, every entry has a shape that starts
    with ``(3, 4)``. Shape operations (indexing, ``reshape``, ``view``,
    ``unsqueeze``, ``permute``, masking) act on the batch dimensions of every
    entry at once. Pointwise operations (``td + other``, ``td * 2``),
    ``td.to(device)`` and ``td.to(dtype)`` apply to every entry. An entry can
    be another tensordict, so keys can be nested: ``td["a", "b"]`` reads the
    entry ``"b"`` of the nested tensordict ``"a"``. :ref:`batch-size` and
    :ref:`keys` explain these ideas.

    Use :class:`TensorDict` for most data. To stack tensordicts whose keys or
    entry shapes differ, use :class:`~tensordict.LazyStackedTensorDict`. To
    declare a fixed set of fields with types, use
    :class:`~tensordict.TensorClass` or :class:`~tensordict.TypedTensorDict`.
    :ref:`choosing-a-container` compares the containers.

    Build a tensordict from a dict, or from keyword arguments as with
    ``dict(...)``:

        >>> import torch
        >>> from tensordict import TensorDict
        >>> td = TensorDict(
        ...     {"obs": torch.zeros(3, 4, 5), "reward": torch.zeros(3, 4)},
        ...     batch_size=[3, 4],
        ... )
        >>> td[0]["obs"].shape
        torch.Size([4, 5])
        >>> td.view(-1)["obs"].shape
        torch.Size([12, 5])
        >>> (td + 1)["reward"].sum()
        tensor(12.)
        >>> td.to(torch.float64)["obs"].dtype
        torch.float64
        >>> td = TensorDict(a=torch.zeros(3), b={"c": torch.ones(3, 2)}, batch_size=[3])
        >>> td["b", "c"].shape
        torch.Size([3, 2])

    A tensordict has these constraints:

    - The batch size is not inferred from the data: it defaults to ``[]``.
      Call :meth:`~tensordict.TensorDictBase.auto_batch_size_` to set it to
      the largest leading shape that all entries share.
    - The shape of every entry must start with the batch size. If it does
      not, setting the entry raises a ``RuntimeError``.
    - If ``device`` is set, every entry is moved to that device. If it is
      not set, the entries can be on different devices. See :ref:`devices`.
    - A key is a string, or a tuple of strings for a nested entry. Other keys
      raise a ``KeyError``.

    The values of ``source`` are stored as given, not copied, except when
    they are moved to ``device``. A nested ``dict`` becomes a nested
    tensordict with the same batch size. Numbers and arrays become tensors,
    and other values such as strings are stored as non-tensor data (see
    :ref:`non-tensor-data`).

    Args:
        source (dict, Mapping or TensorDictBase, optional): the entries, read
            like a ``dict``: string or nested keys to tensors, tensordicts,
            nested dicts or other values. Exclusive with ``**kwargs``.
            Defaults to ``None``, which gives an empty tensordict that can be
            filled later.
        batch_size (torch.Size, list of int or int, optional): the batch size
            of the tensordict: the leading dimensions that all its entries
            share. It can be changed later if the entries allow it. Defaults
            to ``None``, which gives the batch size ``[]``.
        device (torch.device, str or int, optional): the device of the
            tensordict. Entries on another device are moved to it. Defaults to
            ``None``, which leaves each entry where it is, and the tensordict
            has no device.
        names (list of str, optional): the names of the batch dimensions, one
            per dimension. Defaults to ``None``, which leaves the dimensions
            unnamed.
        non_blocking (bool, optional): with ``device``, if ``True``, the
            entries are copied to the device asynchronously with respect to
            the host, with no synchronization: this is safe only from CPU to
            CUDA. If ``False``, every copy is synchronous. Defaults to
            ``None``, which copies asynchronously and synchronizes after the
            copies when needed.
        lock (bool, optional): if ``True``, the tensordict is locked: its
            entries cannot be added, removed or replaced. See
            :ref:`keys-locking`. Defaults to ``False``.
        **kwargs: the entries, as keyword arguments, instead of ``source``.

    Raises:
        ValueError: if both ``source`` and ``**kwargs`` are given, or if the
            length of ``names`` differs from that of ``batch_size``.
        RuntimeError: if the shape of an entry does not start with
            ``batch_size``.

    Examples:
        Name the batch dimensions. Shape operations keep the names:

        >>> import torch
        >>> from tensordict import TensorDict
        >>> td = TensorDict(
        ...     obs=torch.zeros(3, 4), action=torch.zeros(3, 2), batch_size=[3], names=["time"]
        ... )
        >>> td.names
        ['time']
        >>> td.unsqueeze(0).names
        [None, 'time']

        Set the batch size from the data:

        >>> td = TensorDict(obs=torch.zeros(3, 4))
        >>> td.batch_size
        torch.Size([])
        >>> td.auto_batch_size_().batch_size
        torch.Size([3, 4])

        Adding an entry to a locked tensordict raises an error:

        >>> td = TensorDict(obs=torch.zeros(3), batch_size=[3], lock=True)
        >>> td["action"] = torch.zeros(3)
        Traceback (most recent call last):
            ...
        RuntimeError: Cannot modify locked TensorDict. For in-place modification, consider using the `set_()` method and make sure the key is present.

    """

    _td_dim_names = None
    _set_at_warns_cast = True

    def __init__(
        self,
        source: T | Mapping[Any, Any] | None = None,
        batch_size: Sequence[int] | torch.Size | int | None = None,
        device: DeviceType | None = None,
        names: Sequence[str] | None = None,
        non_blocking: bool | None = None,
        lock: bool = False,
        **kwargs: Any,
    ) -> None:
        if (source is not None) and kwargs:
            raise ValueError(
                "Either a dictionary or a sequence of kwargs must be provided, not both."
            )
        source = source if not kwargs else kwargs

        self._tensordict = _StringOnlyDict()
        is_eager = not is_compiling()
        # Eager only: under compile, Dynamo does not guard on the emptiness
        # read here, and would reuse the empty-source graph for a full source.
        is_empty = is_eager and (
            source is None or (type(source) is dict and not source)
        )

        if is_empty and names is None and not lock and type(self) is TensorDict:
            # Nothing to validate, move, sync or rename: this ends in the
            # state the general path below gives an empty source.
            if device is not None:
                device = _parse_device(device, is_eager)[0]
            self._device = device
            self._batch_size = self._parse_batch_size(source, batch_size)
            self._td_dim_names = None
            return

        has_device = device is not None
        sub_non_blocking = False
        call_sync = False
        if has_device:
            if non_blocking is None:
                sub_non_blocking = True
            else:
                sub_non_blocking = non_blocking
            # Auto-index the device
            device, is_cuda = _parse_device(device, is_eager)
            if is_cuda:
                # CUDA does its sync by itself
                call_sync = False
            elif non_blocking is None and not is_eager:
                # Under compile, copy with blocking calls instead of marking
                # the device recorder and syncing: the sync breaks the graph,
                # and a mark nested in another TensorDict() raises.
                sub_non_blocking = False
            else:
                # An empty source has no transfer to record.
                call_sync = non_blocking is None and not is_empty
                if call_sync:
                    _device_recorder.mark()
        try:
            self._device = device

            if source is None:
                source = {}
            if type(source) is not dict and not isinstance(source, Mapping):
                raise ValueError(
                    "A TensorDict source is expected to be a TensorDictBase "
                    f"sub-type or a Mapping, found type(source)={type(source)}."
                )
            self._batch_size = self._parse_batch_size(source, batch_size)
            self._set_names(names)

            # Fast path: use dict.update() to establish all keys in one
            # bulk operation, then validate values individually. This
            # avoids per-key dict mutations that trigger Dynamo
            # DICT_KEYS_MATCH guards and graph breaks.
            _tensordict = self._tensordict
            # Read on first use: exact tensors that match need no validation.
            _validate_value = None
            _tensordict.update(source)
            # Exact tensors need neither conversion nor collection dispatch.
            # Keep subclasses (including TensorDict subclasses with custom
            # validation), nested tensors and device moves on the general path.
            validate_tensor = type(self) is TensorDict
            batch_size = self._batch_size
            batch_dims = len(batch_size)
            for key in list(_tensordict):
                if isinstance(key, str):
                    value = _tensordict[key]
                    if (
                        validate_tensor
                        and type(value) is torch.Tensor
                        and (
                            not batch_dims
                            or (
                                not value.is_nested
                                and value.shape[:batch_dims] == batch_size
                            )
                        )
                        and (device is None or value.device == device)
                    ):
                        continue
                    if _validate_value is None:
                        _validate_value = self._validate_value
                    _tensordict[key] = _validate_value(
                        value,
                        check_shape=True,
                        non_blocking=sub_non_blocking,
                        key=key,
                    )
                else:
                    # Tuple keys need nested TensorDict creation via the
                    # standard path.
                    value = _tensordict.pop(key)
                    self.set(key, value, non_blocking=sub_non_blocking)
            if call_sync:
                if _device_recorder.has_transfer():
                    self._sync_all()
                _device_recorder.unmark()
                call_sync = False

            if lock:
                self.lock_()
        finally:
            if call_sync:
                _device_recorder.unmark()

    @classmethod
    def _new_unsafe(
        cls,
        source: T | dict[NestedKey, CompatibleType] | None = None,
        batch_size: Sequence[int] | torch.Size | int | None = None,
        device: DeviceType | None = None,
        names: Sequence[str] | None = None,
        non_blocking: bool | None = None,
        lock: bool = False,
        nested: bool = True,
        is_shared: bool = False,
        is_memmap: bool = False,
        **kwargs: Any,
    ) -> TensorDict:
        is_eager = not is_compiling()
        if not is_eager and cls is TensorDict:
            # If the cls is not TensorDict, we must escape this to keep the same class.
            # That's unfortunate because as of now it graph breaks but that's the best we can do.
            td = TensorDict(
                source,
                batch_size=batch_size,
                device=device,
                names=names,
                non_blocking=non_blocking,
                lock=lock,
                **kwargs,
            )
            # The data is already in shared or memory-mapped storage: set the
            # flags as the eager branch does, without copying it.
            if is_shared:
                td._is_shared = True
            if is_memmap:
                td._is_memmap = True
            return td
        if kwargs and not source:
            source = kwargs
        self = cls.__new__(cls)
        sub_non_blocking = False
        if device is not None:
            if non_blocking is None:
                sub_non_blocking = True
                non_blocking = False
            else:
                sub_non_blocking = non_blocking
            if type(device) is not torch.device:
                device = torch.device(device)
            if _mps_available() if is_eager else self._has_mps:
                # With MPS, an explicit sync is required
                sub_non_blocking = True
        self._device = device
        self._tensordict = _tensordict = _StringOnlyDict()
        self._batch_size = batch_size
        if not nested and type(source) is dict:
            _tensordict.update(source)
        elif source:  # faster than calling items
            for key, value in source.items():
                if nested and isinstance(value, dict):
                    value = cls._new_unsafe(
                        source=value,
                        batch_size=self._batch_size,
                        device=self._device,
                        non_blocking=sub_non_blocking,
                    )
                _tensordict[key] = value
        self._td_dim_names = names
        if lock:
            self.lock_()
        if is_shared:
            self._is_shared = True
        if is_memmap:
            self._is_memmap = True
        if _CHECK_INVARIANTS and not is_compiling():
            _check_invariants(self)
        return self

    def _is_dense_node(self) -> bool:
        # An exact TensorDict without dim names: _dense_apply rebuilds it.
        return type(self) is TensorDict and self._td_dim_names is None

    def _dense_apply(
        self,
        fn,
        named,
        nested_keys,
        prefix,
        filter_empty,
        is_leaf,
        device=NO_DEFAULT,
        call_on_unbatched=False,
    ):
        # What _apply_nest returns for a node that passes _is_dense_node, with
        # checked=True and no others, out, inplace, batch size or names: each
        # node gets the batch size of its source node, and the given device
        # or, without one, the device of its source node. Entries other than
        # tensors and such nodes take the steps of _apply_nest for one entry.
        result = self._new_unsafe(
            batch_size=self._batch_size,
            device=self._device if device is NO_DEFAULT else device,
        )
        _tensordict = result._tensordict
        for key, item in self._tensordict.items():
            cls = type(item)
            if cls is TensorDict and item._td_dim_names is None:
                item_trsf = item._dense_apply(
                    fn,
                    named,
                    nested_keys,
                    prefix + (key,),
                    filter_empty,
                    is_leaf,
                    device,
                    call_on_unbatched,
                )
            elif cls is not Tensor and not is_leaf(cls):
                # the call _apply_nest makes for a nested collection
                item_trsf = item._apply_nest(
                    fn,
                    inplace=False,
                    batch_size=None,
                    device=device,
                    checked=True,
                    named=named,
                    nested_keys=nested_keys,
                    default=NO_DEFAULT,
                    prefix=prefix + (key,),
                    filter_empty=filter_empty,
                    is_leaf=is_leaf,
                    out=None,
                    call_on_unbatched=call_on_unbatched,
                )
            elif named:
                if nested_keys:
                    item_trsf = fn(prefix + (key,) if prefix != () else key, item)
                else:
                    item_trsf = fn(key, item)
            else:
                item_trsf = fn(item)
            if item_trsf is not None:
                _tensordict[key] = item_trsf
        if not _tensordict and (
            filter_empty or (filter_empty is None and not self.is_empty())
        ):
            return None
        return result

    # Override: the implementation, which TensorDictBase.from_module calls.
    @classmethod
    def from_module(
        cls,
        module: torch.nn.Module,
        as_module: bool = False,
        lock: bool = False,
        use_state_dict: bool = False,
        filter_empty: bool = True,
    ):
        """Returns a tensordict that holds the parameters and buffers of a module.

        The entries are the parameters and buffers of the module themselves,
        not copies. Submodules become nested tensordicts.

        .. note::
            This method does not lock the result by default (``lock=False``).
            The function :func:`tensordict.from_module` locks it by default
            (``lock=True``).

        Args:
            module (torch.nn.Module): the module to get the parameters from.
            as_module (bool, optional): if ``True``, a :class:`~tensordict.nn.TensorDictParams`
                instance is returned, which can be used to store parameters
                within a :class:`torch.nn.Module`. Defaults to ``False``.
            lock (bool, optional): if ``True``, the resulting tensordict is
                locked. Defaults to ``False``.
            use_state_dict (bool, optional): if ``True``, the state-dict from the
                module is used and unflattened into a TensorDict with
                the tree structure of the model. Use it when the module has
                state-dict hooks. Defaults to ``False``.
            filter_empty (bool, optional): if ``True``, submodules without
                parameters or buffers (``nn.ReLU``, for example) get no
                entry. If ``False``, they get an empty tensordict.
                Defaults to ``True``.

        Returns:
            A :class:`~tensordict.TensorDict`, or a
            :class:`~tensordict.nn.TensorDictParams` if ``as_module=True``.

        Examples:
            >>> from torch import nn
            >>> from tensordict import TensorDict
            >>> module = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
            >>> params = TensorDict.from_module(module)
            >>> params["0", "weight"] is module[0].weight
            True
            >>> list(params.keys())
            ['0']
            >>> params.is_locked
            False

        """
        result = cls._from_module(
            module=module,
            as_module=as_module,
            use_state_dict=use_state_dict,
            filter_empty=filter_empty,
        )
        if result is None:
            result = cls._new_unsafe({}, batch_size=torch.Size(()))
        if lock:
            result.lock_()
        return result

    @classmethod
    def _from_module(
        cls,
        module: torch.nn.Module,
        as_module: bool = False,
        use_state_dict: bool = False,
        prefix="",
        filter_empty: bool = True,
        _memo=None,
    ):
        if isinstance(module, TensorDictBase):
            from tensordict.nn import TensorDictParams

            if isinstance(module, TensorDictParams):
                if _memo is not None:
                    _memo["has_params"] = True
                return module
        if (
            as_module
            and cls is TensorDict
            and not use_state_dict
            and not is_compiling()
        ):
            from tensordict.nn.params import _maybe_make_param_or_buffer

            _memo = {"convert_buffers": _maybe_make_param_or_buffer}
        convert_buffers = _memo["convert_buffers"] if _memo is not None else None
        destination = {}
        if use_state_dict:
            keep_vars = False
            # do we need this feature atm?
            local_metadata = {}
            # if hasattr(destination, "_metadata"):
            #     destination._metadata[prefix[:-1]] = local_metadata
            for hook in module._state_dict_pre_hooks.values():
                hook(module, prefix, keep_vars)
            module._save_to_state_dict(destination, "", keep_vars)
        else:
            for name, param in module._parameters.items():
                if param is None:
                    continue
                destination[name] = param
            for name, buffer in module._buffers.items():
                if buffer is None:
                    continue
                if convert_buffers is not None:
                    buffer = convert_buffers(buffer)
                destination[name] = buffer

        if use_state_dict:
            for hook in module._state_dict_hooks.values():
                hook_result = hook(module, destination, prefix, local_metadata)
                if hook_result is not None:
                    destination = hook_result
            if _EXTRA_STATE_KEY_SUFFIX in destination:
                destination[_EXTRA_STATE_KEY_SUFFIX] = _wrap_extra_state(
                    destination[_EXTRA_STATE_KEY_SUFFIX]
                )
        if not filter_empty or destination:
            destination_set = True
            if cls is TensorDict and not use_state_dict:
                destination = cls._new_unsafe(
                    destination, batch_size=torch.Size(()), nested=False
                )
            else:
                destination = cls._new_unsafe(destination, batch_size=torch.Size(()))
        else:
            destination_set = False
        for name, submodule in module._modules.items():
            if submodule is not None:
                if cls is TensorDict:
                    subtd = cls._from_module(
                        module=submodule,
                        as_module=False,
                        use_state_dict=use_state_dict,
                        prefix=prefix + name + ".",
                        filter_empty=filter_empty,
                        _memo=_memo,
                    )
                else:
                    subtd = cls._from_module(
                        module=submodule,
                        as_module=False,
                        use_state_dict=use_state_dict,
                        prefix=prefix + name + ".",
                        filter_empty=filter_empty,
                    )
                if subtd is not None:
                    if not destination_set:
                        destination = cls._new_unsafe(batch_size=torch.Size(()))
                        destination_set = True
                    destination._set_str(
                        name, subtd, validated=True, inplace=False, non_blocking=False
                    )
        if not destination_set:
            return

        if as_module:
            from tensordict.nn.params import TensorDictParams

            if _memo is not None and not _memo.get("has_params", False):
                # The capture traversal already converted buffers. Avoid the
                # wrapper constructor's second conversion/unlock traversal.
                # Existing TensorDictParams children still need that traversal
                # to preserve its recursive unlock behavior.
                result = TensorDictParams(destination, no_convert="skip")
                result.no_convert = True
                return result
            return TensorDictParams(destination, no_convert=True)
        return destination

    @property
    def _has_non_tensor(self):
        # Faster than TensorDictBase._has_non_tensor: it checks the set of value
        # types and asks the nested collections, so that a nested lazy stack
        # never stacks its leaves (which raises for heterogeneous shapes).
        # update_ calls it once per member of a lazy stack: the first loop is
        # the fast path for a tensordict of plain tensors.
        values = self._tensordict.values()
        for value in values:
            cls = type(value)
            if cls is Tensor:
                continue
            if cls is TensorDict:
                # a nested TensorDict answers for itself, as below
                if value._has_non_tensor:
                    return True
                continue
            break
        else:
            return False
        for cls in set(map(type, values)):
            if issubclass(cls, Tensor):
                continue
            if cls is not TensorDict:
                if _is_non_tensor(cls):
                    return True
                if _pass_through_cls(cls):
                    continue
                if not _is_tensor_collection(cls):
                    return super()._has_non_tensor
            for value in values:
                if type(value) is cls and value._has_non_tensor:
                    return True
        return False

    # Override: dense fast path over the dict of entries.
    def is_empty(self):

        for item in self._tensordict.values():
            # we need to check if item is empty
            if _is_tensor_collection(type(item)):
                if not item.is_empty():
                    return False

                if is_non_tensor(item):
                    return False
            else:
                return False
        # A nested lazy stack does not warn: this call warns once, as its own
        # result is True (#2807).
        if (
            self._tensordict
            and not is_compiling()
            and _nested_members_hold_leaf(self)
            and not _in_dense_is_empty()
        ):
            _warn_lazy_is_empty()
        return True

    # Functorch compatibility
    @_cache_while_locked  # noqa: B019
    def _add_batch_dim(self, *, in_dim: int, vmap_level: int) -> Self:
        td = self

        def _add_batch_dim_wrapper(key: str, value: Any) -> Any:
            if _is_unbatched(value):
                return _unbatched_with_batch_size(value, td.batch_size)._add_batch_dim(
                    in_dim=in_dim, vmap_level=vmap_level
                )
            if is_tensor_collection(value):
                return value._add_batch_dim(in_dim=in_dim, vmap_level=vmap_level)

            if isinstance(
                value, (_BatchedUninitializedParameter, _BatchedUninitializedBuffer)
            ):
                value.in_dim = in_dim
                value.vmap_level = vmap_level
                return value
            return _add_batch_dim(value, in_dim, vmap_level)

        out = self._new_unsafe(
            {key: _add_batch_dim_wrapper(key, value) for key, value in td.items()},
            batch_size=torch.Size(
                [b for i, b in enumerate(td.batch_size) if i != in_dim]
            ),
            names=(
                [name for i, name in enumerate(td.names) if i != in_dim]
                if self._has_names()
                else None
            ),
            lock=self.is_locked,
        )
        return out

    @_cache_while_locked  # noqa: B019
    def _remove_batch_dim(self, vmap_level: int, batch_size: int, out_dim: int) -> Self:
        new_batch_size = list(self.batch_size)
        new_batch_size.insert(out_dim, batch_size)
        names = self._maybe_names()
        if names:
            new_names = list(names)
            new_names.insert(out_dim, None)
        else:
            new_names = None

        def _process_remove(value):
            if _is_unbatched(value):
                return _unbatched_with_batch_size(
                    value, self.batch_size
                )._remove_batch_dim(
                    vmap_level=vmap_level, batch_size=batch_size, out_dim=out_dim
                )
            if is_tensor_collection(value):
                return value._remove_batch_dim(
                    vmap_level=vmap_level, batch_size=batch_size, out_dim=out_dim
                )
            return _remove_batch_dim(value, vmap_level, batch_size, out_dim)

        out = TensorDict(
            {key: _process_remove(value) for key, value in self.items()},
            batch_size=new_batch_size,
            names=new_names,
            lock=self.is_locked,
        )
        return out

    @_cache_while_locked  # noqa: B019
    def _maybe_remove_batch_dim(
        self, funcname: str, vmap_level: int, batch_size: int, out_dim: int
    ) -> Self:
        new_batch_size = list(self.batch_size)
        new_batch_size.insert(out_dim, batch_size)
        names = self._maybe_names()
        if names:
            new_names = list(names)
            new_names.insert(out_dim, None)
        else:
            new_names = None

        def _process_value(value):
            if _is_unbatched(value):
                return _unbatched_with_batch_size(
                    value, self.batch_size
                )._maybe_remove_batch_dim(
                    funcname=funcname,
                    vmap_level=vmap_level,
                    batch_size=batch_size,
                    out_dim=out_dim,
                )
            if is_tensor_collection(value):
                return value._maybe_remove_batch_dim(
                    funcname=funcname,
                    vmap_level=vmap_level,
                    batch_size=batch_size,
                    out_dim=out_dim,
                )
            return _maybe_remove_batch_dim(
                funcname, value, vmap_level, batch_size, out_dim
            )

        out = TensorDict(
            {key: _process_value(value) for key, value in self.items()},
            batch_size=new_batch_size,
            names=new_names,
            lock=self.is_locked,
        )
        return out

    # Override: Tensor.chunk on each leaf is faster than the generic split
    def _chunk(self, chunks: int, dim: int) -> tuple[TensorCollection, ...]:
        batch_size = self.batch_size
        max_size = batch_size[dim]
        split_size = -(max_size // -chunks)
        segments = _create_segments_from_int(split_size, max_size)
        # The front has checked chunks and dim, and a sub-tensordict has
        # the same batch dim: call its hook directly. A call through the
        # front adds a Python frame per nesting level, which slows deep
        # tensordicts.
        splits = {
            k: (
                (v,) * len(segments)
                if _is_unbatched(v)
                else v.chunk(chunks, dim)
                if type(v) is Tensor or not isinstance(v, TensorDictBase)
                else v._chunk(chunks, dim)
            )
            for k, v in self.items()
        }
        names = self._maybe_names()
        batch_sizes = [
            torch.Size(
                tuple(d if i != dim else end - start for i, d in enumerate(batch_size))
            )
            for start, end in segments
        ]
        splits = [
            {k: v[ss] for k, v in splits.items()} for ss in range(len(batch_sizes))
        ]
        for split, bsz in _zip_strict(splits, batch_sizes):
            for key, value in split.items():
                if _is_unbatched(value):
                    split[key] = _unbatched_with_batch_size(value, bsz)
        device = self.device
        is_shared = self._is_shared
        is_memmap = self._is_memmap
        is_locked = self.is_locked
        result = tuple(
            self._new_unsafe(
                source=split,
                batch_size=bsz,
                names=names,
                device=device,
                lock=is_locked,
                is_shared=is_shared,
                is_memmap=is_memmap,
            )
            for split, bsz in _zip_strict(splits, batch_sizes)
        )
        return result

    def _view(self, shape: torch.Size) -> Self:
        shape = _infer_size_impl(shape, self.batch_size.numel())
        if torch.Size(shape) == self.shape:
            return self
        batch_dims = self.batch_dims

        def _view(tensor):
            return tensor.view((*shape, *tensor.shape[batch_dims:]))

        result = self._fast_apply(
            _view, batch_size=shape, call_on_nested=True, propagate_lock=True
        )
        self._maybe_set_shared_attributes(result)
        return result

    def _transpose(self, dim0, dim1):
        def _transpose(tensor):
            return tensor.transpose(dim0, dim1)

        batch_size = list(self.batch_size)
        v0 = batch_size[dim0]
        v1 = batch_size[dim1]
        batch_size[dim1] = v0
        batch_size[dim0] = v1
        if self._has_names():
            names = self.names
            names = [
                names[dim0] if i == dim1 else names[dim1] if i == dim0 else names[i]
                for i in range(self.ndim)
            ]
        else:
            names = None
        result = self._fast_apply(
            _transpose,
            batch_size=torch.Size(batch_size),
            call_on_nested=True,
            names=names,
            propagate_lock=True,
        )
        self._maybe_set_shared_attributes(result)
        return result

    def _permute(self, dims_list: list[int]):
        if any(dim < 0 or dim >= self.ndim for dim in dims_list):
            raise ValueError(
                "Received an permutation order incompatible with the tensordict shape."
            )
        # note: to allow this to work recursively, we must allow permutation order with fewer elements than dims,
        # as long as this list is complete.
        if sorted(dims_list) != list(range(len(dims_list))):
            raise ValueError(
                f"Cannot compute the permutation, got dims={dims_list} but expected a permutation of {list(range(len(dims_list)))}."
            )
        if not len(dims_list) and not self.batch_dims:
            return self
        if dims_list == list(range(len(dims_list))):
            return self

        def _permute(tensor):
            return tensor.permute(*dims_list, *range(len(dims_list), tensor.ndim))

        batch_size = self.batch_size
        batch_size = [batch_size[p] for p in dims_list] + list(
            batch_size[len(dims_list) :]
        )
        if self._has_names():
            names = self.names
            names = [names[i] for i in dims_list] + names[len(dims_list) :]
        else:
            names = None
        result = self._fast_apply(
            _permute,
            batch_size=batch_size,
            call_on_nested=True,
            names=names,
            propagate_lock=True,
        )
        self._maybe_set_shared_attributes(result)
        return result

    def _squeeze(self, dim=None):
        batch_size = self.batch_size
        if dim is None:
            names = list(self.names) if self._has_names() else None
            if names is not None:
                names = [
                    name for size, name in _zip_strict(batch_size, names) if size != 1
                ]
            batch_size = [size for size in batch_size if size != 1]
            batch_size = torch.Size(batch_size)
            if batch_size == self.batch_size:
                return self

            # we only want to squeeze dimensions lower than the batch dim, and view
            # is the perfect op for this
            def _squeeze(tensor):
                from tensordict import LazyStackedTensorDict

                if (
                    isinstance(tensor, LazyStackedTensorDict)
                    and tensor.shape[tensor.stack_dim] == 1
                ):
                    return tensor.tensordicts[0]
                return tensor.view(*batch_size, *tensor.shape[self.batch_dims :])

            return self._fast_apply(
                _squeeze,
                batch_size=batch_size,
                names=names,
                inplace=False,
                call_on_nested=True,
                propagate_lock=True,
            )
        batch_size = list(batch_size)
        batch_size.pop(dim)
        batch_size = list(batch_size)
        names = list(self.names) if self._has_names() else None
        if names:
            names.pop(dim)

        def squeeze(x):
            return x.squeeze(dim)

        result = self._fast_apply(
            squeeze,
            batch_size=batch_size,
            names=names,
            inplace=False,
            call_on_nested=True,
            propagate_lock=True,
        )
        self._maybe_set_shared_attributes(result)
        return result

    def _unsqueeze(self, dim: int):
        batch_size = list(self.batch_size)
        batch_size.insert(dim, 1)
        batch_size = torch.Size(batch_size)

        names = list(self.names) if self._has_names() else None
        if names is not None:
            names.insert(dim, None)

        def _unsqueeze(tensor):
            return tensor.unsqueeze(dim)

        result = self._fast_apply(
            _unsqueeze,
            batch_size=batch_size,
            names=names,
            inplace=False,
            call_on_nested=True,
            propagate_lock=True,
        )
        self._maybe_set_shared_attributes(result)
        return result

    @classmethod
    def from_dict(
        cls,
        input_dict: dict[Any, Any] | TensorDictBase,
        *,
        auto_batch_size: bool | None = None,
        batch_size: Sequence[int] | torch.Size | None = None,
        device: torch.device | None = None,
        batch_dims: int | None = None,
        names: Sequence[str] | None = None,
    ) -> Self:
        if _is_tensor_collection(type(input_dict)):
            return input_dict

        if batch_dims is not None and batch_size is not None:
            raise ValueError(
                "Cannot pass both batch_size and batch_dims to `from_dict`."
            )

        batch_size_set = torch.Size(()) if batch_size is None else batch_size
        input_dict = dict(input_dict)
        for key, value in list(input_dict.items()):
            # we don't know if another tensor of smaller size is coming
            # so we can't be sure that the batch-size will still be valid later
            input_dict[key] = TensorDict.from_any(
                value,
                auto_batch_size=False,
                device=device,
                batch_size=batch_size,
            )
        if batch_size is None and names is not None:
            names = list(names)
            if None not in names:
                # a 0-dim tensordict drops names without None entries: drop
                # them here, without the warning for a list of the wrong length
                names = None
        # regular __init__ breaks because a tensor may have the same batch-size as the tensordict
        out = cls(
            input_dict,
            batch_size=batch_size_set,
            device=device,
            names=names,
        )
        if batch_size is None:
            if auto_batch_size is None and batch_dims is None:
                auto_batch_size = False
            elif auto_batch_size is None:
                auto_batch_size = True
            if auto_batch_size:
                _set_max_batch_size(out, batch_dims)
        else:
            out.batch_size = batch_size
        return out

    @classmethod
    def _from_dict_validated(
        cls, input_dict, batch_size=None, device=None, batch_dims=None, names=None
    ):
        return cls._new_unsafe(
            input_dict,
            batch_size=torch.Size(batch_size),
            device=torch.device(device) if device is not None else device,
            names=names if any(name is not None for name in names) else None,
        )

    @staticmethod
    def _parse_batch_size(
        source: T | dict | None,
        batch_size: Sequence[int] | torch.Size | int | None = None,
    ) -> torch.Size:
        ERR = "batch size was not specified when creating the TensorDict instance and it could not be retrieved from source."

        if is_compiling():
            if isinstance(batch_size, torch.Size):
                return batch_size
            elif isinstance(batch_size, tuple):
                return torch.Size(batch_size)
            elif isinstance(batch_size, list):
                return torch.Size(tuple(batch_size))
            if batch_size is None:
                return torch.Size([])
            elif isinstance(batch_size, (Number, torch.SymInt)):
                # torch.SymInt does not subclass Number but must be handled the
                # same way during compilation (e.g. batch_size=x.shape[0]).
                # See: https://github.com/pytorch/tensordict/issues/1003
                return torch.Size([batch_size])
            elif isinstance(source, TensorDictBase):
                return source.batch_size
            raise ValueError()

        try:
            return torch.Size(batch_size)
        except Exception:
            if batch_size is None:
                return torch.Size([])
            elif isinstance(batch_size, Number):
                return torch.Size([batch_size])
            elif isinstance(source, TensorDictBase):
                return source.batch_size
            raise ValueError(ERR)

    def _rename_subtds(self, names):
        """Propagates names to nested tensordicts.

        A child may have more batch dims than self. Only the first
        self.batch_dims are shared, so names beyond that are the child's own
        and are left alone. ``names=None`` clears the shared dims.

        Every new name list is computed before any is set, so a clash
        anywhere in the tree leaves all names unchanged.
        """
        if not self._batch_size or not self._tensordict:
            # no dim is shared, or no child to rename
            return
        renames = []
        self._collect_subtd_names(names, (), renames)
        for item, td_names in renames:
            if isinstance(item, TensorDict):
                # For TensorDict items, we can directly set _td_dim_names
                item._td_dim_names = td_names
            else:
                # For other tensor collections (tensorclasses, NonTensorData, etc.),
                # use the public API which handles the renaming correctly
                item.rename_(*td_names)

    def _collect_subtd_names(self, names, prefix, renames):
        if names is None:
            names = [None] * self.batch_dims
        for key, item in self._tensordict.items():
            is_td = isinstance(item, TensorDict)
            if is_td:
                item_names = item._td_dim_names
            elif _is_tensor_collection(type(item)):
                item_names = item.names
            else:
                continue
            if item_names is None:
                # Extend names with None for the remaining dimensions
                td_names = list(names) + [None] * (item.batch_dims - len(names))
            else:
                td_names = list(names) + list(item_names)[len(names) :]
                # the shared prefix must not reuse a name the child owns,
                # mirroring the uniqueness check of _set_names
                named = [name for name in td_names if name is not None]
                if len(set(named)) != len(named):
                    raise ValueError(
                        f"Some dimension names are non-unique: {td_names} "
                        f"for the nested tensordict at key {prefix + (key,)}."
                    )
            if is_td:
                if all(name is None for name in td_names):
                    td_names = None
                renames.append((item, td_names))
                # Recursively collect the names of nested tensor collections
                item._collect_subtd_names(td_names, prefix + (key,), renames)
            else:
                renames.append((item, td_names))

    # Override: O(1) on the dict of entries, which keeps the insertion order.
    @_lock_blocked
    def popitem(self) -> Tuple[NestedKey, CompatibleType]:
        item = self._tensordict.popitem()
        if hasattr(self, "_consolidated"):
            self._mark_consolidated_stale()
        return item

    # Override: a dict of valid tensors goes into the dict of entries in one
    # call. With sub-tensordicts or dicts among them, see _update_fast_merge.
    def _update_fast(self, source: dict[str, CompatibleType] | T) -> bool:
        if type(self) is not TensorDict or self._is_locked:
            return False
        if source is self:
            return True
        batch_size = self._batch_size
        batch_dims = len(batch_size)
        device = self._device
        if type(source) is TensorDict:
            if batch_size[: source.batch_dims] == source.batch_size[:batch_dims]:
                source = source._tensordict
        if type(source) is dict:
            # Validate before writing so a fallback preserves the general
            # path's ordering of conversions, errors and partial updates.
            for key, value in source.items():
                if (
                    type(key) is not str
                    or type(value) is not Tensor
                    or (
                        batch_dims
                        and (value.is_nested or value.shape[:batch_dims] != batch_size)
                    )
                    or (device is not None and value.device != device)
                ):
                    break
            else:
                self._tensordict.update(source)
                if hasattr(self, "_consolidated"):
                    self._mark_consolidated_stale()
                return True
            if type(key) is str and (type(value) is TensorDict or type(value) is dict):
                self._update_fast_merge(source)
                return True
        return False

    def _update_fast_merge(self, source: dict[str, CompatibleType]) -> None:
        """Writes ``source`` into ``self`` key by key, in the order of the general path of :meth:`update`.

        :meth:`_update_fast` calls it for a ``source`` dict that holds a
        sub-tensordict or a dict. A valid tensor is written as the general
        path writes it, and a :class:`TensorDict` or dict whose key holds a
        :class:`TensorDict` in ``self`` is merged into it with the same
        ``update`` call. Any other entry goes through the general path, one
        key at a time, so the result, the errors and the entries written
        before an error are those of the general path.
        """
        batch_size = self._batch_size
        batch_dims = len(batch_size)
        device = self._device
        entries = self._tensordict
        consolidated = hasattr(self, "_consolidated")
        # New keys for the general path are collected in `pending` and
        # written with one _update_entries call, made before the next write
        # of any other kind and at the end. The general path writes the keys
        # of a dict in order, so the result and the entries written before an
        # error are those of one call per key, without the fixed cost of a
        # call per key. Writing a key that self does not hold yet cannot change
        # source or another entry, so deferring it to the next write changes
        # nothing else; keys that self holds keep their own call.
        pending = None
        # Iterate over source itself, as the general path does: a merge that
        # writes into source (if source is a sub-tensordict of self) shows.
        for key, value in source.items():
            # 0: the general path, 1: a valid tensor, 2: a merge into child
            kind = 0
            if type(key) is str:
                if type(value) is Tensor:
                    if not (
                        (
                            batch_dims
                            and (
                                value.is_nested
                                or value.shape[:batch_dims] != batch_size
                            )
                        )
                        or (device is not None and value.device != device)
                    ):
                        kind = 1
                elif (
                    type(value) is TensorDict or type(value) is dict
                ) and value is not self:
                    # self goes to the general path, which raises before it
                    # writes anything
                    child = entries.get(key)
                    if type(child) is TensorDict:
                        kind = 2
                if not kind and key not in entries:
                    if pending is None:
                        pending = {}
                    pending[key] = value
                    continue
            if pending:
                self._update_entries(
                    pending,
                    clone=False,
                    inplace=False,
                    non_blocking=False,
                    keys_to_update=None,
                    is_leaf=_is_leaf_nontensor,
                    update_batch_size=False,
                    ignore_lock=False,
                )
                pending = None
            if kind == 1:
                # a valid tensor: written as the general path writes it
                entries[key] = value
                if consolidated:
                    self._mark_consolidated_stale()
            elif kind == 2:
                child.update(value)
            else:
                self._update_entries(
                    {key: value},
                    clone=False,
                    inplace=False,
                    non_blocking=False,
                    keys_to_update=None,
                    is_leaf=_is_leaf_nontensor,
                    update_batch_size=False,
                    ignore_lock=False,
                )
        if pending:
            self._update_entries(
                pending,
                clone=False,
                inplace=False,
                non_blocking=False,
                keys_to_update=None,
                is_leaf=_is_leaf_nontensor,
                update_batch_size=False,
                ignore_lock=False,
            )

    # Override: dense fast path.
    def _contains_str(self, key: str) -> bool:
        return key in self._tensordict

    # Override: perf: looks the key up in the dicts of the exact TensorDicts on
    # its path, so that torch.compile guards on the key and not on the order of
    # all the keys (#2892), and eager mode builds no keys view.
    def _contains_key(self, key: NestedKey) -> bool:
        if isinstance(key, str):
            return key in self._tensordict
        root = self._tensordict.get(key[0])
        if root is None:
            return False
        if type(root) is not TensorDict:
            # a tensor, a lazy stack, a tensorclass...: the keys view answers
            return TensorDictBase._contains_key(self, key)
        # the steps of _TensorDictKeysView.__contains__
        leaf_td = root._get_tuple(key[1:-1], None) if len(key) > 2 else root
        if type(leaf_td) is TensorDict:
            return key[-1] in leaf_td._tensordict
        if leaf_td is None or not _is_tensor_collection(type(leaf_td)):
            return False
        return key[-1] in leaf_td.keys()

    def _set_str(
        self,
        key: NestedKey,
        value: dict[str, CompatibleType] | CompatibleType,
        *,
        inplace: bool,
        validated: bool,
        ignore_lock: bool = False,
        non_blocking: bool = False,
    ) -> Self:
        if inplace is not False:
            best_attempt = inplace is BEST_ATTEMPT_INPLACE
            inplace = self._convert_inplace(inplace, key)
        if not validated:
            if type(value) is Tensor and type(self) is TensorDict:
                batch_size = self._batch_size
                device = self._device
                # len(): `not` on a torch.Size makes Dynamo specialize its sizes
                validated = (
                    not len(batch_size)
                    or (
                        not value.is_nested
                        and value.shape[: len(batch_size)] == batch_size
                    )
                ) and (device is None or value.device == device)
            elif (
                type(value) is TensorDict
                and type(self) is TensorDict
                and not self._batch_size
                and not is_compiling()
            ):
                # A plain TensorDict stored in a TensorDict without batch dims:
                # the validator could only move it to self's device (there is no
                # batch size or names to match).
                device = self._device
                validated = device is None or value._device == device
            if not validated:
                original = value
                value = self._validate_value(
                    value, check_shape=True, non_blocking=non_blocking, key=key
                )
                # a scalar is converted to a tensor: check the type only when it
                # replaces a NonTensorData entry, not a NonTensorStack, and the
                # write below goes through (not in place, not locked)
                if (
                    not inplace
                    and original is not value
                    and not self._batch_size
                    and not (self._is_locked and not ignore_lock)
                ):
                    dest = self._tensordict.get(key)
                    if (
                        is_non_tensor(dest)
                        and is_tensorclass(dest)
                        and isinstance(original, _SCALAR_TYPES)
                    ):
                        _warn_scalar_over_non_tensor(key, original)
        if not inplace:
            if self._is_locked and not ignore_lock:
                raise RuntimeError(_LOCK_ERROR)
            self._tensordict[key] = value
            if hasattr(self, "_consolidated"):
                # An unpickled consolidated copy is not locked
                self._mark_consolidated_stale()
        else:
            try:
                dest = self._get_str(key, default=NO_DEFAULT)
                if best_attempt and _is_tensor_collection(type(dest)):
                    dest.update(
                        value,
                        inplace=True,
                        non_blocking=non_blocking,
                        ignore_lock=ignore_lock,
                    )
                else:
                    if dest is not value:
                        try:
                            dest.copy_(value, non_blocking=non_blocking)
                        except RuntimeError:
                            # if we're updating a param and the storages match, nothing needs to be done
                            if not (
                                isinstance(dest, torch.Tensor)
                                and dest.data.untyped_storage().data_ptr()
                                == value.data.untyped_storage().data_ptr()
                            ):
                                raise
                        if hasattr(self, "_consolidated") and is_non_tensor(dest):
                            # A non-tensor value is not in the consolidated storage
                            self._mark_consolidated_stale()
            except KeyError as err:
                raise err
            except Exception as err:
                raise _InplaceWriteError(
                    f"Failed to update '{key}' in tensordict {self}"
                ) from err
        return self

    def _set_dict(
        self,
        d: dict[str, CompatibleType],
        *,
        validated: bool,
    ):
        if not validated:
            raise RuntimeError("Not Implemented for non-validated inputs")
        self._tensordict = d

    def _set_tuple(
        self,
        key: NestedKey,
        value: dict[str, CompatibleType] | CompatibleType,
        *,
        inplace: bool,
        validated: bool,
        non_blocking: bool = False,
    ) -> Self:
        if len(key) == 1:
            return self._set_str(
                key[0],
                value,
                inplace=inplace,
                validated=validated,
                non_blocking=non_blocking,
            )
        td = self._get_str(key[0], None)
        if td is None:
            td = self._create_nested_str(key[0])
            inplace = False
        elif not _is_tensor_collection(type(td)):
            raise KeyError(
                f"The entry {key[0]} is already present in tensordict {self}."
            )
        td._set_tuple(
            key[1:],
            value,
            inplace=inplace,
            validated=validated,
            non_blocking=non_blocking,
        )
        return self

    _SHARED_INPLACE_ERROR = (
        "You're attempting to update a leaf in-place with a shared "
        "tensordict, but the new value does not match the previous. "
        "If you're using NonTensorData, see the class documentation "
        "to see how to properly pre-allocate memory in shared contexts."
    )

    def _set_at_str(
        self, key, value, idx, *, validated, non_blocking: bool, warn_cast: bool = True
    ):
        tensor_in = self._get_str(key, NO_DEFAULT)
        converted = False
        if not validated:
            if (
                isinstance(value, Number)
                and is_tensor_collection(tensor_in)
                and not is_non_tensor(tensor_in)
            ):
                # each entry of the nested tensordict casts the scalar
                nested = (
                    tensor_in._tensordict if is_tensorclass(tensor_in) else tensor_in
                )
                if warn_cast or not type(nested)._set_at_warns_cast:
                    tensor_in[idx] = value
                else:
                    # td[index] = value has decided whether to warn
                    _set_at_each_key(nested, value, idx, warn_cast=False)
                return self
            original = value
            value = _cast_scalar(value, tensor_in)
            converted = not is_tensor_collection(value)
            value = self._validate_value(
                value, check_shape=False, non_blocking=non_blocking
            )
            validated = True
            if (
                warn_cast
                and type(tensor_in) is not Tensor
                and isinstance(original, _SET_AT_CAST_TYPES)
                and not is_tensor_collection(value)
                # shared and memmap containers refuse the write below
                and not (self._is_shared or self._is_memmap)
            ):
                if getattr(type(tensor_in), "_is_non_tensor", False):
                    _warn_set_at_cast(key, original)
                elif is_tensor_collection(tensor_in) and tensor_in._has_non_tensor:
                    # written into each entry of the nested tensordict, as
                    # td[index] = value does
                    _warn_set_at_cast(None, original)

        if is_non_tensor(value) and not (self._is_shared or self._is_memmap):
            if isinstance(idx, tuple) and len(idx) == 1:
                idx = idx[0]
            dest = tensor_in
            if (
                isinstance(idx, tuple)
                and not idx
                and is_tensorclass(dest)
                and is_non_tensor(dest)
            ):
                # () selects the whole entry, but a NonTensorData cannot be
                # indexed with it: write the value into the entry, as set_
                # does, which keeps the batch size of the entry
                self._set_str(
                    key, value, inplace=True, validated=True, non_blocking=non_blocking
                )
                return self
            if (
                isinstance(idx, torch.Tensor)
                and idx.shape == ()
                and self.shape == ()
                and idx.dtype == torch.bool
                and idx
            ):
                # squeeze(0) raises IndexError on an entry with no batch dim,
                # unlike Tensor.squeeze. Skip it and let the write below run.
                # An entry with a batch dim is still squeezed. A tensor has no
                # batch size, so it is squeezed too.
                dest_batch_size = getattr(dest, "batch_size", None)
                if dest_batch_size is None or dest_batch_size:
                    self._set_str(
                        key,
                        dest.squeeze(0),
                        validated=True,
                        inplace=False,
                        ignore_lock=True,
                    )
            dest_data = dest[idx].tolist()
            value_data = value.tolist()
            try:
                is_diff = bool(dest_data != value_data)
            except Exception:
                # Arrays, tensors and data frames compare elementwise, and the
                # result has no single truth value: write the value anyway.
                is_diff = True
            if is_diff:
                if _is_stack_of_batchless_non_tensors(dest):
                    # Write into the stack: rebuilding it with
                    # maybe_to_stack() would cost O(size) per write
                    dest_val = dest
                else:
                    dest_val = dest.maybe_to_stack()
                if (
                    idx is Ellipsis
                    and dest_val is dest
                    and is_tensorclass(dest)
                    and is_tensorclass(value)
                    and is_non_tensor(value)
                ):
                    # A NonTensorData without batch dims: the Ellipsis selects
                    # the whole entry, and setitem on it does not write the
                    # data, so build the new entry instead
                    dest_val = type(dest)(
                        data=value.data,
                        batch_size=dest.batch_size,
                        names=dest._maybe_names(),
                        device=dest.device,
                    )
                else:
                    if converted and is_tensorclass(value):
                        # The data was converted to a NonTensorData with the
                        # batch size of self: drop its batch dims, so that it
                        # broadcasts to dest[idx]
                        value = value.empty(batch_size=())
                    dest_val[idx] = value
                if dest_val is not dest:
                    self._set_str(
                        key,
                        dest_val,
                        validated=True,
                        inplace=False,
                        ignore_lock=True,
                    )
            # TODO: ultimately, we want to get rid of the above logic
            # dest_val = dest.maybe_to_stack()
            # dest_val[idx] = value
            # if dest_val is not dest:
            #     self._set_str(
            #         key,
            #         dest_val,
            #         validated=True,
            #         inplace=False,
            #         ignore_lock=True,
            #     )
            return

        if isinstance(idx, tuple) and len(idx) and isinstance(idx[0], tuple):
            warn(
                "Multiple indexing can lead to unexpected behaviours when "
                "setting items, for instance `td[idx1][idx2] = other` may "
                "not write to the desired location if idx1 is a list/tensor."
            )
            tensor_in = _sub_index(tensor_in, idx)
            tensor_in.copy_(value, non_blocking=non_blocking)
        else:
            tensor_out = _set_item(
                tensor_in, idx, value, validated=validated, non_blocking=non_blocking
            )
            if tensor_in is not tensor_out:
                if self._is_shared or self._is_memmap:
                    raise RuntimeError(self._SHARED_INPLACE_ERROR)
                # this happens only when a NonTensorData becomes a NonTensorStack
                # so it is legitimate (there is no in-place modification of a tensor
                # that was expected to happen but didn't).
                # For this reason we can ignore the locked attribute of the td.
                self._set_str(
                    key,
                    tensor_out,
                    validated=True,
                    inplace=False,
                    ignore_lock=True,
                    non_blocking=non_blocking,
                )

        return self

    def _set_at_tuple(self, key, value, idx, *, validated, non_blocking: bool):
        if len(key) == 1:
            return self._set_at_str(
                key[0], value, idx, validated=validated, non_blocking=non_blocking
            )
        if key[0] not in self.keys():
            # this won't work
            raise KeyError(f"key {key} not found in set_at_ with tensordict {self}.")
        else:
            td = self._get_str(key[0], NO_DEFAULT)
        td._set_at_tuple(
            key[1:], value, idx, validated=validated, non_blocking=non_blocking
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
        if (
            clone
            or keys_to_update is not None
            or type(self) is not TensorDict
            or type(input_dict_or_td) is not TensorDict
        ):
            return NotImplemented
        source_keys, source_values = input_dict_or_td._items_list(True, True)
        if not source_keys:
            if input_dict_or_td.is_empty():
                return self
            return NotImplemented
        dest_values = []
        for key in source_keys:
            dest = self
            for subkey in _unravel_key_to_tuple(key):
                if type(dest) is not TensorDict or subkey not in dest._tensordict:
                    return NotImplemented
                dest = dest._tensordict[subkey]
            dest_values.append(dest)
        idx = _entry_index(idx)
        if not isinstance(idx, tuple):
            idx = (idx,)
        idx = convert_ellipsis_to_idx(idx, self.batch_size)
        indexed_dest_values = []
        for dest, source in zip(dest_values, source_values):
            if not isinstance(dest, Tensor) or not isinstance(source, Tensor):
                return NotImplemented
            try:
                dest_indexed = dest[idx]
            except (IndexError, RuntimeError, TypeError):
                return NotImplemented
            if dest_indexed.shape != source.shape:
                return NotImplemented
            if (
                dest_indexed is not dest
                and getattr(dest_indexed, "_base", None) is None
            ):
                return NotImplemented
            indexed_dest_values.append(dest_indexed)
        copy_fn = _foreach_copy_compiled if is_compiling() else _foreach_copy_
        copy_fn(indexed_dest_values, source_values, non_blocking=non_blocking)
        return self

    # Override: dense fast path: a top-level key is deleted without the call
    # to _del.
    @_lock_blocked
    def del_(self, key: NestedKey) -> Self:
        key = _unravel_key_to_tuple(key)
        if len(key) == 1:
            del self._tensordict[key[0]]
            if hasattr(self, "_consolidated"):
                self._mark_consolidated_stale()
            return self
        self._del(key)
        return self

    def _del(self, key: tuple[str, ...]) -> None:
        if len(key) == 1:
            del self._tensordict[key[0]]
            if hasattr(self, "_consolidated"):
                self._mark_consolidated_stale()
        elif not key:
            raise KeyError(_GENERIC_NESTED_ERR.format(key))
        else:
            try:
                parent = self._get_tuple(key[:-1], None)
            except ValueError:
                # a tensor before key[-2]: raise the error of the whole key,
                # which names the rest of the path
                self._get_tuple(key, None)
                raise
            if parent is None:
                # name the first missing sub-tensordict
                i = 1
                while self._get_tuple(key[:i], None) is not None:
                    i += 1
                raise KeyError(f"No sub-tensordict with key {key[i - 1]}.")
            if not _is_tensor_collection(type(parent)):
                raise ValueError(
                    f"{key[-2]!r} is a {type(parent).__name__}, not a tensordict, "
                    f"so it has no entry {key[-1]!r}."
                )
            parent.del_(key[-1])

    # Override: stacks into the existing leaf storages (torch.stack(out=)).
    def _stack_onto_(self, list_item: list[CompatibleType], dim: int) -> TensorDict:
        for key in self.keys():
            vals = [item._get_str(key, None) for item in list_item]
            if all(v is None for v in vals):
                continue
            dest = self._get_str(key, NO_DEFAULT)
            if _is_unbatched(dest):
                self._set_str(
                    key,
                    _unbatched_with_batch_size(dest, self.batch_size),
                    inplace=False,
                    validated=True,
                )
                continue
            new_dest = torch.stack(
                vals,
                dim=dim,
                out=dest,
            )
            if new_dest is not dest:
                self._set_str(key, new_dest, inplace=False, validated=True)
        return self

    def entry_class(self, key: NestedKey) -> type:
        return type(self.get(key))

    def _stack_onto_at_(
        self,
        list_item: list[CompatibleType],
        dim: int,
        idx: IndexType,
    ) -> TensorDict:
        if not isinstance(idx, tuple):
            idx = (idx,)
        idx = convert_ellipsis_to_idx(idx, self.batch_size)
        for key in self.keys():
            try:
                vals = [td._get_str(key, NO_DEFAULT) for td in list_item]
            except KeyError:
                if all(td._get_str(key, None) is None for td in list_item):
                    # as in _stack_onto_, keep an entry that no tensordict has
                    continue
                raise
            v = self._get_str(key, NO_DEFAULT)
            if _is_unbatched(v):
                continue
            if _is_tensor_collection(type(v)):
                if not isinstance(v, TensorDictBase):
                    raise RuntimeError(
                        f"Cannot call _stack_onto_at_ with an entry of type {type(v).__name__}."
                    )
                if not all(
                    _is_tensor_collection(type(val)) and not _is_non_tensor(type(val))
                    for val in vals
                ):
                    raise RuntimeError(
                        f"Cannot stack non-tensordict values into the tensordict entry {key!r}."
                    )
                v._stack_onto_at_(vals, dim=dim, idx=idx)
                continue
            v_idx = v[idx]
            # A copy (advanced index) is not a view: writing into it would be lost.
            # Empty and meta tensors have no data (data_ptr() is 0 for both).
            if not v_idx._is_view() and v_idx.data_ptr() != v.data_ptr():
                raise IndexError(
                    f"Index {idx} is incompatible with stack(..., out=data) as the storages of the indexed tensors differ."
                )
            # torch.stack would resize the view and write past the indexed region.
            val = vals[0]
            if isinstance(val, Tensor):
                shape = v_idx.shape
                stacked_shape = list(val.shape)
                stacked_shape.insert(dim if dim >= 0 else dim + len(shape), len(vals))
                if shape != tuple(stacked_shape):
                    raise RuntimeError(
                        f"Cannot stack values of shape {tuple(val.shape)} into the entry "
                        f"{key!r} of shape {tuple(shape)} with stack(..., out=data)."
                    )
            torch.stack(vals, dim=dim, out=v_idx)
        return self

    def _get_str(self, key, default, **kwargs):
        try:
            return self._tensordict[key]
        except KeyError:
            return self._default_get(key, default)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Subclasses keep the generic walk, which calls their _get_str and
        # _default_get, without going through the _get_tuple below.
        if cls._get_tuple is TensorDict._get_tuple:
            cls._get_tuple = TensorDictBase._get_tuple
        # and ask their keys view in `key in td`
        if cls._contains_key is TensorDict._contains_key:
            cls._contains_key = TensorDictBase._contains_key

    # Override: reads the levels that are exact TensorDicts from their dicts,
    # without a _get_str and a _get_tuple call per level.
    def _get_tuple(self, key, default, **kwargs):
        if type(self) is not TensorDict:
            # a subclass's own _get_tuple calling super()._get_tuple
            return TensorDictBase._get_tuple(self, key, default, **kwargs)
        try:
            value = self._tensordict[key[0]]
        except KeyError:
            return self._default_get(key[0], default)
        if len(key) == 1:
            return value
        for i in range(1, len(key)):
            if value is default:
                return value
            if type(value) is not TensorDict:
                # a lazy stack, a tensorclass, a tensor...: it reads the rest
                if isinstance(value, Tensor):
                    # as in TensorDictBase._get_tuple, for the Dynamo of torch 2.11
                    raise _key_through_leaf_error(key, i, value)
                try:
                    return value._get_tuple(key[i:], default=default, **kwargs)
                except AttributeError as err:
                    _raise_if_not_a_tensordict(err, key, i, value)
                    return None
            td = value
            value = td._tensordict.get(key[i], NO_DEFAULT)
            if value is NO_DEFAULT:
                return td._default_get(key[i], default)
        return value

    # Override: shares the dense leaves and marks the tensordict shared.
    def _share_memory_(self) -> None:
        if self.is_memmap():
            raise RuntimeError(
                "memmap and shared memory are mutually exclusive features."
            )
        if self.device is not None and self.device.type == "cuda":
            # cuda tensors are shared by default
            return
        for value in self.values():
            if (
                isinstance(value, Tensor)
                and value.device.type == "cpu"
                or _is_tensor_collection(type(value))
            ):
                value.share_memory_()
        self._is_shared = True
        self.lock_()

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
        for key, value in self.items():
            if _is_unbatched(value):
                # the leaf would be written and loaded back as a plain tensor
                raise NotImplementedError(
                    f"UnbatchedTensor entries cannot be memory-mapped yet, but "
                    f"{key!r} is one. Exclude it first, e.g. with td.exclude({key!r})."
                )
        if prefix is not None:
            prefix = Path(prefix)
            if not prefix.exists():
                os.makedirs(prefix, exist_ok=True)
            metadata = {}
        if inplace and self._is_shared:
            raise RuntimeError(
                "memmap and shared memory are mutually exclusive features."
            )

        dest = self if inplace else self.empty(device=torch.device("cpu"))

        # We must set these attributes before memmapping because we need the metadata
        # to match the tensordict content.
        if inplace:
            self._is_memmap = True
            self._is_shared = False  # since they are mutually exclusive
            self._device = torch.device("cpu")
            # The leaves leave the consolidated storage
            self._mark_consolidated_stale()
        else:
            dest._is_memmap = True
            dest._is_shared = False  # since they are mutually exclusive

        for key, value in self.items():
            type_value = type(value)
            if _is_tensor_collection(type_value):
                if prefix is not None:
                    effective_robust_key = _get_robust_key_setting_with_warning(
                        key, robust_key
                    )
                    safe_key = _encode_key_for_filesystem(
                        key, robust=effective_robust_key
                    )
                    value_prefix = prefix / safe_key
                    if value_prefix.is_symlink():
                        raise RuntimeError(
                            "Refusing to write a memory-mapped TensorDict through "
                            f"symlink {value_prefix}."
                        )
                else:
                    value_prefix = None
                dest._tensordict[key] = value._memmap_(
                    prefix=value_prefix,
                    copy_existing=copy_existing,
                    executor=executor,
                    futures=futures,
                    inplace=inplace,
                    like=like,
                    share_non_tensor=share_non_tensor,
                    existsok=existsok,
                    robust_key=robust_key,
                )
                if prefix is not None:
                    _update_metadata(
                        metadata=metadata, key=key, value=value, is_collection=True
                    )
                continue
            else:
                # Check on the calling thread: memmap() does not re-raise
                # errors from executor jobs.
                _raise_if_jagged(value, key)
                if executor is None:
                    _populate_memmap(
                        dest=dest,
                        value=value,
                        key=key,
                        copy_existing=copy_existing,
                        prefix=prefix,
                        like=like,
                        existsok=existsok,
                        robust_key=robust_key,
                    )
                else:
                    futures.append(
                        executor.submit(
                            _populate_memmap,
                            dest=dest,
                            value=value,
                            key=key,
                            copy_existing=copy_existing,
                            prefix=prefix,
                            like=like,
                            existsok=existsok,
                            robust_key=robust_key,
                        )
                    )
                if prefix is not None:
                    _update_metadata(
                        metadata=metadata, key=key, value=value, is_collection=False
                    )

        if prefix is not None:
            if executor is None:
                _save_metadata(
                    dest,
                    prefix,
                    metadata=metadata,
                )
            else:
                futures.append(executor.submit(_save_metadata, dest, prefix, metadata))
        dest._is_locked = True
        dest._memmap_prefix = prefix
        return dest

    @classmethod
    def _load_memmap(
        cls,
        prefix: str,
        metadata: dict,
        device: torch.device | None = None,
        out=None,
        *,
        robust_key,
        allow_pickle: bool | None = None,
        mode: str | None = None,
    ) -> Self:
        if metadata.get("device", "None") == "None":
            metadata["device"] = None
        else:
            metadata["device"] = torch.device(metadata["device"])
        metadata["shape"] = torch.Size(metadata.get("shape", ()))

        if out is None:
            result = cls(
                {},
                batch_size=metadata.pop("shape"),
                device=metadata.pop("device") if device is None else device,
            )
        else:
            if is_tensorclass(out):
                # A tensordict save, such as the ``_tensordict`` directory
                # refreshed by ``memmap_refresh_``, loads into the tensordict
                # of a tensorclass.
                out = out._tensordict
            result = out
            if out.device is not None:
                # An unindexed device (e.g. "cuda") matches any index of its type.
                if device is not None and (
                    device.type != out.device.type
                    or device.index is not None
                    and out.device.index is not None
                    and device.index != out.device.index
                ):
                    raise ValueError(
                        f"Cannot load a tensordict on device {device} into a "
                        f"tensordict on device {out.device}."
                    )
                device = out.device

        paths = []
        loaded_keys = set()
        for key, entry_metadata in metadata.items():
            if not isinstance(entry_metadata, dict):
                # there can be other metadata
                continue
            type_value = entry_metadata.get("type")
            if type_value is not None:
                paths.append(key)
                continue
            dtype = entry_metadata.get("dtype")
            shape = entry_metadata.get("shape")
            # Use smart warning for loading that only warns when encoding would differ
            effective_robust_key = _get_robust_key_setting_with_warning(key, robust_key)

            # Use encoded filename for file system operations
            safe_key = _encode_key_for_filesystem(key, robust=effective_robust_key)
            memmap_file = prefix / f"{safe_key}.memmap"
            memmap_file_exists = memmap_file.exists()

            # If robust encoding is requested but file doesn't exist, try legacy filename
            if (
                not memmap_file_exists
                and effective_robust_key
                and safe_key != key
                and _is_safe_legacy_key(key)
            ):
                legacy_key = _encode_key_for_filesystem(key, robust=False)
                legacy_file = prefix / f"{legacy_key}.memmap"
                if legacy_file.exists():
                    # Use legacy filename for backward compatibility
                    safe_key = legacy_key
                    memmap_file = legacy_file
                    memmap_file_exists = True

            if dtype is None or shape is None:
                # invalid dict means
                continue
            shape = torch.Size(shape)
            if not memmap_file_exists:
                # torch.from_file does not create a backing file for a tensor
                # with no elements. Such tensors can be reconstructed from
                # their metadata; a missing non-empty (or nested) tensor is an
                # incomplete entry and must not be synthesized.
                if entry_metadata.get("is_nested", False) or shape.numel():
                    continue
                tensor = torch.empty(
                    shape,
                    device=device,
                    dtype=_STR_DTYPE_TO_DTYPE[dtype],
                )
                result._set_str(
                    key,
                    tensor,
                    validated=True,
                    inplace=False,
                    non_blocking=False,
                )
                loaded_keys.add(key)
                continue
            is_fake = torch._guards.active_fake_mode()
            if (device is None or device != torch.device("meta")) and not is_fake:
                if entry_metadata.get("is_nested", False):
                    # The shape is the shape of the shape, get the shape from it
                    shape = _memmap_tensor_from_path(
                        memmap_file.with_suffix(".shape.memmap"),
                        shape=shape,
                        dtype=torch.long,
                        mode=mode,
                    )
                else:
                    shape = torch.Size(shape)
                tensor = _memmap_tensor_from_path(
                    memmap_file,
                    dtype=_STR_DTYPE_TO_DTYPE[dtype],
                    shape=shape,
                    mode=mode,
                )
                if device is not None:
                    tensor = tensor.to(device, non_blocking=True)
            else:
                tensor = torch.zeros(
                    shape,
                    device=device,
                    dtype=_STR_DTYPE_TO_DTYPE[dtype],
                )
            result._set_str(
                key,
                tensor,
                validated=True,
                inplace=False,
                non_blocking=False,
            )
            loaded_keys.add(key)
        # Load collection directories named by metadata. New saves use robust
        # encoding; safe single-component legacy names remain readable.
        for key in paths:
            effective_robust_key = _get_robust_key_setting_with_warning(key, robust_key)
            safe_key = _encode_key_for_filesystem(key, robust=effective_robust_key)
            path = prefix / safe_key
            path_is_dir = path.is_dir()
            if (
                not path_is_dir
                and effective_robust_key
                and safe_key != key
                and _is_safe_legacy_key(key, is_collection=True)
            ):
                legacy_path = prefix / key
                if legacy_path.is_dir():
                    path = legacy_path
                    path_is_dir = True
            if not path_is_dir:
                continue
            existing_elt = result._get_str(key, default=None)
            if existing_elt is not None and not _is_tensor_collection(
                type(existing_elt)
            ):
                existing_elt = None
            loaded = TensorDict.load_memmap(
                path,
                device=device,
                non_blocking=True,
                out=existing_elt,
                robust_key=robust_key,
                allow_pickle=allow_pickle,
                mode=mode,
            )
            if loaded is not existing_elt:
                result._set_str(key, loaded, inplace=False, validated=False)
            loaded_keys.add(key)
        if out is not None:
            # Keys of ``out`` that are absent from the saved data are stale.
            for key in list(result.keys()):
                if key not in loaded_keys:
                    result.del_(key)
        # Archive paths are read-only views inside a zip file, and directories
        # mapped copy-on-write can hold writes that their files lack: neither
        # can be used as a target for a subsequent memmap_()/refresh, so only
        # the other directories are recorded.
        result._memmap_prefix = (
            prefix if isinstance(prefix, Path) and mode != "r" else None
        )
        return result

    def _make_memmap_subtd(self, key, *, robust_key):
        """Creates a sub-tensordict given a tuple key."""
        result = self
        for key_str in key:
            result_tmp = result._get_str(key_str, default=None)
            if result_tmp is None:
                result_tmp = result.empty()
                if result._memmap_prefix is not None:
                    effective_robust_key = _get_robust_key_setting_with_warning(
                        key_str, robust_key
                    )
                    safe_key = _encode_key_for_filesystem(
                        key_str, robust=effective_robust_key
                    )
                    subtd_prefix = result._memmap_prefix / safe_key
                    if subtd_prefix.is_symlink():
                        raise RuntimeError(
                            "Refusing to write a memory-mapped TensorDict through "
                            f"symlink {subtd_prefix}."
                        )
                    result_tmp.memmap_(prefix=subtd_prefix)
                    metadata = _load_metadata(result._memmap_prefix)
                    _update_metadata(
                        metadata=metadata,
                        key=key_str,
                        value=result_tmp,
                        is_collection=True,
                    )
                    _save_metadata(
                        result, prefix=result._memmap_prefix, metadata=metadata
                    )
                result._tensordict[key_str] = result_tmp
            result = result_tmp
        return result

    # Override: only dense storage can hold the new memory-mapped tensor.
    def make_memmap(
        self,
        key: NestedKey,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        robust_key: bool | None = True,
    ) -> MemoryMappedTensor:
        if not self.is_memmap():
            raise RuntimeError(
                "Can only make a memmap tensor within a memory-mapped tensordict."
            )

        key = unravel_key(key)
        if isinstance(key, tuple):
            last_node = self._make_memmap_subtd(key[:-1], robust_key=robust_key)
            last_key = key[-1]
        else:
            last_node = self
            last_key = key
        if last_key in last_node.keys():
            raise RuntimeError(
                f"The key {last_key} already exists within the target tensordict. Delete that entry before "
                f"overwriting it."
            )
        if dtype is None:
            dtype = torch.get_default_dtype()
        if last_node._memmap_prefix is not None:
            metadata = _load_metadata(last_node._memmap_prefix)
            memmap_tensor = _populate_empty(
                key=last_key,
                dest=last_node,
                prefix=last_node._memmap_prefix,
                shape=shape,
                dtype=dtype,
                robust_key=robust_key,
            )
            _update_metadata(
                metadata=metadata,
                key=last_key,
                value=memmap_tensor,
                is_collection=False,
            )
            _save_metadata(
                last_node, prefix=last_node._memmap_prefix, metadata=metadata
            )
        else:
            memmap_tensor = MemoryMappedTensor.empty(shape=shape, dtype=dtype)

        last_node._set_str(
            last_key, memmap_tensor, validated=False, inplace=False, ignore_lock=True
        )

        return memmap_tensor

    # Override: only dense storage can hold the new memory-mapped tensor.
    def make_memmap_from_storage(
        self,
        key: NestedKey,
        storage: torch.UntypedStorage,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        robust_key: bool | None = True,
    ) -> MemoryMappedTensor:
        if not self.is_memmap():
            raise RuntimeError(
                "Can only make a memmap tensor within a memory-mapped tensordict."
            )

        key = unravel_key(key)
        if isinstance(key, tuple):
            last_node = self._make_memmap_subtd(key[:-1], robust_key=robust_key)
            last_key = key[-1]
        else:
            last_node = self
            last_key = key
        if last_key in last_node.keys():
            raise RuntimeError(
                f"The key {last_key} already exists within the target tensordict. Delete that entry before "
                f"overwriting it."
            )
        if dtype is None:
            dtype = torch.get_default_dtype()

        if last_node._memmap_prefix is not None:
            metadata = _load_metadata(last_node._memmap_prefix)
            memmap_tensor = _populate_storage(
                key=last_key,
                dest=last_node,
                prefix=last_node._memmap_prefix,
                storage=storage,
                shape=shape,
                dtype=dtype,
                robust_key=robust_key,
            )
            _update_metadata(
                metadata=metadata,
                key=last_key,
                value=memmap_tensor,
                is_collection=False,
            )
            _save_metadata(
                last_node, prefix=last_node._memmap_prefix, metadata=metadata
            )
        else:
            memmap_tensor = MemoryMappedTensor.from_storage(
                storage=storage, shape=shape, dtype=dtype
            )

        last_node._set_str(
            last_key, memmap_tensor, validated=False, inplace=False, ignore_lock=True
        )

        return memmap_tensor

    # Override: only dense storage can hold the new memory-mapped tensor.
    def make_memmap_from_tensor(
        self,
        key: NestedKey,
        tensor: torch.Tensor,
        *,
        copy_data: bool = True,
        existsok: bool = True,
        robust_key: bool | None = True,
    ) -> MemoryMappedTensor:
        if not self.is_memmap():
            raise RuntimeError(
                "Can only make a memmap tensor within a memory-mapped tensordict."
            )

        key = unravel_key(key)
        if isinstance(key, tuple):
            last_node = self._make_memmap_subtd(key[:-1], robust_key=robust_key)
            last_key = key[-1]
        else:
            last_node = self
            last_key = key
        if last_key in last_node.keys():
            raise RuntimeError(
                f"The key {last_key} already exists within the target tensordict. Delete that entry before "
                f"overwriting it."
            )

        if last_node._memmap_prefix is not None:
            metadata = _load_metadata(last_node._memmap_prefix)
            memmap_tensor = _populate_memmap(
                dest=last_node,
                value=tensor,
                key=last_key,
                copy_existing=True,
                prefix=last_node._memmap_prefix,
                like=not copy_data,
                existsok=existsok,
                robust_key=robust_key,
            )
            _update_metadata(
                metadata=metadata,
                key=last_key,
                value=memmap_tensor,
                is_collection=False,
            )
            _save_metadata(
                last_node, prefix=last_node._memmap_prefix, metadata=metadata
            )
        else:
            memmap_tensor = MemoryMappedTensor.from_tensor(tensor)

        last_node._set_str(
            last_key, memmap_tensor, validated=False, inplace=False, ignore_lock=True
        )

        return memmap_tensor

    def _build_lock_schema(self) -> None:
        # Snapshot the current top-level key set as a Python tuple so
        # locked-fast-paths can iterate it without hitting Dynamo
        # ``DICT_KEYS_MATCH`` guards on ``_tensordict``. Stored on the
        # instance dict to override the class-level default of ``None``.
        self.__dict__["_locked_schema"] = _LockedSchema(
            keys=tuple(self._tensordict),
        )

    # Override: perf: an exact TensorDict tree locks in one recursion, without
    # the lists and calls that the generic method makes for each node.
    def _propagate_lock(self, lock_parents_weakrefs=None, *, is_compiling):
        """Locks this tensordict and the tensor collections it holds.

        ``lock_()`` calls it on an unlocked root, with no
        ``lock_parents_weakrefs``; a locking parent calls it on each entry that
        is a tensor collection, with the weakrefs of the parents above that
        entry. It sets ``_is_locked`` and, out of compile, adds the new parents
        to ``_lock_parents_weakrefs`` and builds ``_locked_schema``, then locks
        the entries. It leaves the same state as
        ``TensorDictBase._propagate_lock``, which compiled calls and subclasses
        take. Returns ``None``.
        """
        if is_compiling or type(self) is not TensorDict:
            return TensorDictBase._propagate_lock(
                self, lock_parents_weakrefs, is_compiling=is_compiling
            )
        _propagate_lock_dense(self, lock_parents_weakrefs)

    def _clone(self, recurse: bool = True) -> Self:
        if recurse and self.device is not None:
            return self._clone_recurse()

        if not recurse and is_compiling():
            result = TensorDict(
                batch_size=self.batch_size,
                device=self.device,
                names=self._maybe_names(),
            )
            schema = self._locked_schema
            _src = self._tensordict
            _dst = result._tensordict
            if self._is_locked and schema is not None:
                # Locked-fast-path: walk the cached immutable key tuple
                # instead of ``result._tensordict.update(self._tensordict)``
                # which emits ``DICT_LENGTH`` and related guards on the
                # underlying Python dict. Reading via the schema replaces
                # those with cheap ``LENGTH_CHECK`` / ``EQUALS_MATCH``
                # guards on the immutable schema tuple, which Dynamo can
                # potentially share across TDs that lock to the same
                # key set.
                keys = schema.keys
            else:
                keys = _src
            for key in keys:
                _dst[key] = _clone_value(_src[key], recurse=False)
            return result

        result = self._new_unsafe(
            source={key: _clone_value(value, recurse) for key, value in self.items()},
            batch_size=self.batch_size,
            device=self.device,
            names=self._maybe_names(),
        )
        # If this is uncommented, a shallow copy of a shared/memmap will be shared and locked too
        # This may be undesirable, not sure if this should be the default behaviour
        # (one usually does a copy to modify it).
        # if not recurse:
        #     self._maybe_set_shared_attributes(result)
        return result

    # Override: rebinds each leaf to a contiguous copy.
    def _contiguous_inplace(self, canonical: bool) -> Self:
        if canonical:

            def leaf_fn(value):
                return _canonicalize_tensor(value.contiguous())

        else:

            def leaf_fn(value):
                return value.contiguous()

        def nested_fn(nested):
            nested.contiguous(canonical=canonical, inplace=True)

        return self._inplace_rebind_leaves(leaf_fn, nested_fn, None)

    # Override: fast path for the dense root, which many operations call; its
    # names default is NO_DEFAULT, as names=None drops the names.
    def empty(
        self, recurse=False, *, batch_size=None, device=NO_DEFAULT, names=NO_DEFAULT
    ) -> Self:
        if not recurse:
            if names is NO_DEFAULT:
                names = self._maybe_names()
                if names is not None and batch_size is not None:
                    # as the batch_size setter: keep the leading names, pad with None
                    names = (names + [None] * len(batch_size))[: len(batch_size)]
                    if all(name is None for name in names):
                        names = None
            elif names is not None and (
                len(names)
                != len(self._batch_size if batch_size is None else batch_size)
                or len(set(names)) != len(names)
            ):
                # names that the names setter may reject or drop: the result
                # keeps them as given
                _warn_empty_names(
                    names, self._batch_size if batch_size is None else batch_size, names
                )
            return self._new_unsafe(
                device=self._device if device is NO_DEFAULT else device,
                batch_size=(
                    self._batch_size if batch_size is None else torch.Size(batch_size)
                ),
                source={},
                names=names,
            )
        # names=None keeps the names, as in TensorDictBase.empty: lazy stacks
        # pass names=None to their members when no names are given.
        if names is NO_DEFAULT:
            names = None
        if batch_size is None and device is NO_DEFAULT and names is None:
            return super().empty(recurse=recurse)
        # super().empty() reuses the entries of self that hold no tensor
        # (non-tensor data, sub-tensordicts without leaves). Start from a
        # shallow copy so that the new batch size, device and names do not
        # reach self.
        return super(TensorDict, self.copy()).empty(
            recurse=recurse, batch_size=batch_size, device=device, names=names
        )

    # Override: builds each new sub-tensordict once instead of renaming the
    # keys one by one. It handles only keys that hold a plain tensor or
    # TensorDict and meet neither an existing key nor one another, on an
    # unlocked, unnamed TensorDict; rename_key_ handles (and raises for) the rest.
    def _unflatten_keys_fast(self, separator: str) -> bool:
        if (
            type(self) is not TensorDict
            or type(separator) is not str
            or not separator
            or self._is_locked
            or self._td_dim_names is not None
            or is_compiling()
        ):
            return False
        src = self._tensordict
        tree = {}
        moved = []
        for key, value in src.items():
            if separator not in key:
                continue
            cls = type(value)
            if cls is not Tensor and cls is not TensorDict:
                return False
            parts = key.split(separator)
            if "" in parts or parts[0] in src:
                return False
            node = tree
            for part in parts[:-1]:
                node = node.setdefault(part, {})
                if type(node) is not dict:
                    return False
            if parts[-1] in node:
                return False
            node[parts[-1]] = value
            moved.append(key)
        for key in moved:
            del src[key]
        # new groups go last, in the order of their first key, as with rename_key_
        for key, sub in tree.items():
            src[key] = self._new_unsafe(
                source=sub, batch_size=self._batch_size, device=self._device
            )
        return True

    # Override: dense fast path that shares the leaves and supports inplace.
    def _select(
        self,
        *keys: NestedKey,
        inplace: bool = False,
        strict: bool = True,
        set_shared: bool = True,
    ) -> Self:
        if inplace and self.is_locked:
            raise RuntimeError(_LOCK_ERROR)

        source = {}
        if len(keys):
            keys_to_select = None
            for key in keys:
                if isinstance(key, str):
                    subkey = []
                else:
                    key, subkey = key[0], key[1:]

                val = self._get_str(key, default=_UNSET if not strict else NO_DEFAULT)
                if val is _UNSET:
                    continue
                source[key] = val
                if len(subkey):
                    if keys_to_select is None:
                        # delay creation of defaultdict
                        keys_to_select = defaultdict(list)
                    keys_to_select[key].append(subkey)

            if keys_to_select is not None:
                for key, val in keys_to_select.items():
                    entry = source[key]
                    try:
                        source[key] = entry._select(
                            *val, strict=strict, inplace=inplace, set_shared=set_shared
                        )
                    except AttributeError:
                        # Checked only on failure, so that the common path pays
                        # nothing for it.
                        if _is_tensor_collection(type(entry)):
                            raise
                        # The path runs through a leaf: the nested key is missing.
                        if strict:
                            rest = val[0][0] if len(val[0]) == 1 else val[0]
                            raise KeyError(
                                f"{key!r} is a {type(entry).__name__}, not a "
                                f"tensordict, so it has no entry {rest!r}."
                            ) from None
                        # Nested calls pass the key alone as a 1-tuple.
                        if key not in keys and (key,) not in keys:
                            del source[key]

        names = self._td_dim_names
        if names is not None:
            names = list(names)

        result = self._new_unsafe(
            device=self.device,
            batch_size=self.batch_size,
            source=source,
            # names=self.names if self._has_names() else None,
            names=names,
        )
        if inplace:
            self._tensordict = result._tensordict
            if hasattr(self, "_consolidated"):
                self._mark_consolidated_stale()
            return self
        # If this is uncommented, a shallow copy of a shared/memmap will be shared and locked too
        # This may be undesirable, not sure if this should be the default behaviour
        # (one usually does a copy to modify it).
        # if set_shared:
        #     self._maybe_set_shared_attributes(result)
        return result

    # Override: dense fast path that shares the leaves and supports inplace.
    def _exclude(
        self, *keys: NestedKey, inplace: bool = False, set_shared: bool = True
    ) -> Self:
        # faster than Base.exclude
        if not len(keys):
            return self.copy() if not inplace else self
        if not inplace:
            _tensordict = dict(self._tensordict)
        else:
            _tensordict = self._tensordict
        keys_to_exclude = None
        # The keys are unravelled (exclude does it once): a str or a tuple of
        # two or more str. The sub-keys passed down keep that form.
        for key in keys:
            if isinstance(key, str):
                if inplace and self.is_locked and key in _tensordict:
                    raise RuntimeError(_LOCK_ERROR)
                _tensordict.pop(key, None)
            else:
                if keys_to_exclude is None:
                    # delay creation of defaultdict
                    keys_to_exclude = defaultdict(list)
                if key[0] in self._tensordict:
                    keys_to_exclude[key[0]].append(key[1] if len(key) == 2 else key[1:])
        if keys_to_exclude is not None:
            for key, cur_keys in keys_to_exclude.items():
                val = _tensordict.get(key)
                if val is not None:
                    try:
                        val = val._exclude(
                            *cur_keys, inplace=inplace, set_shared=set_shared
                        )
                    except AttributeError:
                        if _is_tensor_collection(type(val)):
                            raise
                        # A path that runs through a leaf is missing: keep the leaf.
                        continue
                    if not inplace:
                        _tensordict[key] = val
        if inplace:
            if hasattr(self, "_consolidated"):
                self._mark_consolidated_stale()
            return self
        result = self._new_unsafe(
            _tensordict,
            batch_size=self.batch_size,
            device=self.device,
            names=self.names if self._has_names() else None,
        )
        # If this is uncommented, a shallow copy of a shared/memmap will be shared and locked too
        # This may be undesirable, not sure if this should be the default behaviour
        # (one usually does a copy to modify it).
        # if set_shared:
        #     self._maybe_set_shared_attributes(result)
        return result

    # @_cache_while_locked
    def keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
        *,
        sort: bool = False,
    ) -> _TensorDictKeysView:
        if not include_nested and not leaves_only and is_leaf is None:
            if not sort:
                return _StringKeys(self._tensordict.keys())
            else:

                def keyfunc(x):
                    return ".".join(x) if isinstance(x, tuple) else x

                return sorted(
                    _StringKeys(self._tensordict.keys()),
                    key=keyfunc,
                )
        else:
            return self._nested_keys(
                include_nested=include_nested,
                leaves_only=leaves_only,
                is_leaf=is_leaf,
                sort=sort,
            )

    @_cache_while_locked  # noqa: B019
    def _nested_keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
        *,
        sort: bool = False,
    ) -> _TensorDictKeysView:
        return _TensorDictKeysView(
            self,
            include_nested=include_nested,
            leaves_only=leaves_only,
            is_leaf=is_leaf,
            sort=sort,
        )

    # Override: the method runs on a copy of self in which each nested
    # tensordict that reads copies, such as an h5 file, is replaced by a copy
    # of its leaves, and each nested lazy stack whose members read copies by a
    # stack of copies of their leaves, and each such tensordict or member
    # writes its copy back, as the in-place methods of the lazy stack do
    # (#2592, #2775).
    def _inplace_on_nested_copies(self, method: Callable, *args, **kwargs) -> Self:
        # the public name of a hook such as _zero_
        name = method.__name__.lstrip("_")
        written = []
        copies = self._tree_of_copies(written)
        result = method(copies, *args, **kwargs)
        if result is not copies:
            # broadcasting self returns a new tensordict and writes nothing
            return result
        _write_back_members(written, name)
        return self

    def _tree_of_copies(self, written: list) -> TensorDict:
        """A shallow copy of ``self``, with each nested tensordict that reads copies replaced by a copy of its leaves.

        A nested lazy stack whose members read copies is replaced by a stack
        of copies. Called by ``_inplace_on_nested_copies``. Appends each such
        tensordict or member and its copy to ``written``, for it to write the
        copy back.
        """
        from tensordict._lazy import LazyStackedTensorDict

        source = {}
        for key, value in self._tensordict.items():
            if isinstance(value, TensorDict):
                value = value._tree_of_copies(written)
            elif isinstance(value, LazyStackedTensorDict):
                if not _is_non_tensor(type(value)):
                    # also for members that hold files
                    value = value._stack_of_copies(written)
            elif (
                isinstance(value, TensorDictBase)
                and value._reads_return_copies
                and not _has_lazy_stack(value)
            ):
                # a TypedTensorDict over a lazy stack, whose copy would split
                # the keys over the members of that stack, is left out, as in
                # _stack_of_copies
                leaves = value._copy_of_leaves()
                written.append((value, leaves))
                value = leaves
            source[key] = value
        return self._new_unsafe(
            source=source,
            batch_size=self.batch_size,
            device=self.device,
            names=self._maybe_names(),
        )

    # some custom methods for efficiency
    # Override: dense fast path over the dict of entries.
    def items(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
        *,
        sort: bool = False,
    ) -> Iterator[tuple[str, CompatibleType]]:
        if not include_nested and not leaves_only:
            if not sort:
                return self._tensordict.items()

            def keyfunc(x):
                return x[0]

            return sorted(self._tensordict.items(), key=keyfunc)
        elif include_nested and leaves_only and not sort:
            is_leaf = _default_is_leaf if is_leaf is None else is_leaf
            # is_leaf only filters: the entries of a tensor collection that it
            # accepts are listed too
            accepts_nodes = is_leaf not in _IS_LEAF_WITHOUT_NODES
            result = []

            def fast_iter():
                for key, val in self._tensordict.items():
                    # We could easily make this faster, here we're iterating twice over the keys,
                    #  but we could iterate just once.
                    #  Ideally we should make a "dirty" list of items then call unravel_key on all of them.
                    if is_leaf(type(val)):
                        result.append((key, val))
                        # Recurse as TensorDictBase.items does: not into
                        # non-tensor or pass-through values.
                        if not (
                            accepts_nodes
                            and _is_tensor_collection(type(val))
                            and not is_non_tensor(val)
                            and not _pass_through(val)
                        ):
                            continue
                    elif (
                        # Skip the values without entries, such as the tensors in
                        # non_tensor_items. The cheap checks come first: a TensorDict
                        # has entries, and the default is_leaf and
                        # _NESTED_TENSORS_AS_LISTS only reject tensor collections.
                        type(val) is not TensorDict
                        and is_leaf is not _default_is_leaf
                        and is_leaf is not _NESTED_TENSORS_AS_LISTS
                        and not _is_tensor_collection(type(val))
                    ):
                        continue
                    for _key, _val in val.items(
                        include_nested=include_nested,
                        leaves_only=leaves_only,
                        is_leaf=is_leaf,
                    ):
                        if isinstance(_key, str):
                            _key = (key, _key)
                        else:
                            _key = (key, *_key)
                        result.append((_key, _val))
                return result

            return fast_iter()
        else:
            return super().items(
                include_nested=include_nested,
                leaves_only=leaves_only,
                is_leaf=is_leaf,
                sort=sort,
            )

    # Override: dense fast path over the dict of entries.
    def values(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
        *,
        sort: bool = False,
    ) -> Iterator[CompatibleType]:
        if not include_nested and not leaves_only:
            if not sort:
                return self._tensordict.values()
            else:

                def keyfunc(x):
                    return x[0]

                items = sorted(self._tensordict.items(), key=keyfunc)
                return list(zip(*items))[1] if items else ()
        else:
            return TensorDictBase.values(
                self,
                include_nested=include_nested,
                leaves_only=leaves_only,
                is_leaf=is_leaf,
                sort=sort,
            )


class _StateFromSource:
    """State of a view that reads another tensordict, ``self._source``.

    The view keeps no lock, device, shared or memmap flag, or names flag of its
    own: it reports those of its source, and a device assignment goes to the
    source. Subclasses set ``_source``.
    """

    # Override: the lock state lives on the source.
    @TensorDictBase.is_locked.getter
    def is_locked(self) -> bool:
        return self._source.is_locked

    # Override: the device is the source's.
    @property
    def device(self) -> torch.device | None:
        return self._source.device

    @device.setter
    def device(self, value: DeviceType) -> None:
        self._source.device = value

    # Override: the flag lives on the source.
    @property
    def _is_shared(self):
        return self._source._is_shared

    # Override: the flag lives on the source.
    @property
    def _is_memmap(self):
        return self._source._is_memmap

    # Override: the names are the source's.
    def _has_names(self):
        return self._source._has_names()

    # Override: the storage is the source's.
    def _storage_is_read_only(self) -> bool:
        return self._source._storage_is_read_only()

    # Override: the entries are the source's.
    @property
    def _has_non_tensor(self):
        return self._source._has_non_tensor


class _SubTensorDict(_NoShapeOps, _StateFromSource, TensorDictBase):
    """A TensorDict that only sees an index of the stored tensors."""

    _lazy = True
    _inplace_set = True
    _safe = False
    _masking_on_dense_copy = True
    _memmap_inplace_unsupported = True
    _pin_memory_unsupported = True
    # td[key] = value writes into an existing entry of the source
    _setitem_key_inplace = BEST_ATTEMPT_INPLACE

    def __init__(
        self,
        source: T,
        idx: IndexType,
        batch_size: Sequence[int] | None = None,
    ) -> None:
        if not _is_tensor_collection(type(source)):
            raise TypeError(
                f"Expected source to be a subclass of TensorDictBase, "
                f"got {type(source)}"
            )
        self._source = source
        idx = (
            (idx,)
            if not isinstance(
                idx,
                (
                    tuple,
                    list,
                ),
            )
            else tuple(idx)
        )
        if any(item is Ellipsis for item in idx):
            idx = convert_ellipsis_to_idx(idx, self._source.batch_size)
        self._batch_size = _getitem_batch_size(self._source.batch_size, idx)
        self.idx = idx

        if batch_size is not None and batch_size != self.batch_size:
            raise RuntimeError("batch_size does not match self.batch_size.")

    @property
    def _reads_return_copies(self):
        # the entries are views only for a basic index of the tensors that the
        # source holds: an index tensor or a mask, or a source that builds its
        # entries on each read (a lazy stack, a file), gives copies
        return not (
            _holds_its_tensors(self._source)
            and all(
                (type(idx) is int) or idx is None or isinstance(idx, slice)
                for idx in self.idx
            )
        )

    # Override: the names come from the source, at the index of the view.
    @TensorDictBase.names.getter
    def names(self):
        names = self._source._get_names_idx(self.idx)
        if names is None:
            return [None] * self.batch_dims
        return names

    # Override: a view cannot rename the dims of its source.
    def _set_names(self, names: Sequence[str] | None):
        if names is None:
            names = [None] * self.batch_dims
        if names[: self.batch_dims] == self.names:
            return
        raise RuntimeError(
            "Names of a subtensordict cannot be modified. Instantiate it as a TensorDict first."
        )

    # Override: a view cannot erase the names of its source.
    def _erase_names(self):
        raise RuntimeError(
            "Cannot erase names of a _SubTensorDict. Erase source TensorDict's names instead."
        )

    def _rename_subtds(self, names):
        for key in self.keys():
            if _is_tensor_collection(self.entry_class(key)):
                raise RuntimeError("Cannot rename nested sub-tensordict dimensions.")

    def _preallocate(self, key: NestedKey, value: CompatibleType) -> Self:
        return self._source.set(key, value)

    def _convert_inplace(self, inplace, key):
        has_key = key in self.keys()
        if inplace is not False:
            if inplace is True and not has_key:  # inplace could be None
                raise KeyError(
                    _KEY_ERROR.format(key, type(self).__name__, sorted(self.keys()))
                )
            inplace = has_key
        if not inplace and has_key:
            raise RuntimeError(
                "Calling `_SubTensorDict.set(key, value, inplace=False)` is "
                "prohibited for existing tensors. Consider calling "
                "_SubTensorDict.set_(...) or cloning your tensordict first."
            )
        elif not inplace and self.is_locked:
            raise RuntimeError(_LOCK_ERROR)
        return inplace

    def _set_str(
        self,
        key: NestedKey,
        value: dict[str, CompatibleType] | CompatibleType,
        *,
        inplace: bool,
        validated: bool,
        ignore_lock: bool = False,
        non_blocking: bool = False,
    ) -> Self:
        inplace = self._convert_inplace(inplace, key)
        # it is assumed that if inplace=False then the key doesn't exist. This is
        # checked in set method, but not here. responsibility lies with the caller
        # so that this method can have minimal overhead from runtime checks
        parent = self._source
        if not validated:
            value = self._validate_value(
                value, check_shape=True, non_blocking=non_blocking, key=key
            )
            validated = True
        if not inplace:
            if _is_tensor_collection(type(value)):
                # value has the shape of subtd[idx], so we want an expanded
                #  version value_expand such that value_expand[idx] has the
                #  shape of value
                value_expand = _expand_to_match_shape(
                    parent.batch_size,
                    value,
                    self.batch_dims,
                    self.device,
                    index=self.idx,
                )
                for _key, _tensor in value.items():
                    value_expand._set_str(
                        _key,
                        _expand_to_match_shape(
                            parent.batch_size,
                            _tensor,
                            self.batch_dims,
                            self.device,
                            index=self.idx,
                        ),
                        inplace=inplace,
                        validated=validated,
                        ignore_lock=ignore_lock,
                        non_blocking=non_blocking,
                    )
            else:
                value_expand = torch.zeros(
                    (
                        *parent.batch_size,
                        *_shape(value)[self.batch_dims :],
                    ),
                    dtype=value.dtype,
                    device=self.device,
                )
                if self._is_shared:
                    value_expand.share_memory_()
                elif self._is_memmap:
                    value_expand = MemoryMappedTensor.from_tensor(value_expand)
            parent._set_str(
                key,
                value_expand,
                inplace=False,
                validated=validated,
                ignore_lock=ignore_lock,
                non_blocking=non_blocking,
            )

        parent._set_at_str(
            key, value, self.idx, validated=validated, non_blocking=non_blocking
        )
        return self

    def _set_tuple(
        self,
        key: NestedKey,
        value: dict[str, CompatibleType] | CompatibleType,
        *,
        inplace: bool,
        validated: bool,
        non_blocking: bool = False,
    ) -> Self:
        if len(key) == 1:
            return self._set_str(
                key[0],
                value,
                inplace=inplace,
                validated=validated,
                non_blocking=non_blocking,
            )
        parent = self._source
        td = parent._get_str(key[0], None)
        if td is None:
            td = parent.select()
            parent._set_str(
                key[0], td, inplace=False, validated=True, non_blocking=non_blocking
            )
        _SubTensorDict(td, self.idx)._set_tuple(
            key[1:],
            value,
            inplace=inplace,
            validated=validated,
            non_blocking=non_blocking,
        )
        return self

    def _set_at_str(self, key, value, idx, *, validated, non_blocking: bool):
        tensor_in = self._get_str(key, NO_DEFAULT)
        if not validated:
            value = self._validate_value(
                _cast_scalar(value, tensor_in),
                check_shape=False,
                non_blocking=non_blocking,
            )
            validated = True
        if isinstance(idx, tuple) and len(idx) and isinstance(idx[0], tuple):
            warn(
                "Multiple indexing can lead to unexpected behaviours when "
                "setting items, for instance `td[idx1][idx2] = other` may "
                "not write to the desired location if idx1 is a list/tensor."
            )
            tensor_in = _sub_index(tensor_in, idx)
            tensor_in.copy_(value)
            tensor_out = tensor_in
        else:
            tensor_out = _set_item(
                tensor_in, idx, value, validated=validated, non_blocking=non_blocking
            )
        # make sure that the value is updated
        self._source._set_at_str(
            key, tensor_out, self.idx, validated=validated, non_blocking=non_blocking
        )
        return self

    def _set_at_tuple(self, key, value, idx, *, validated, non_blocking: bool):
        if len(key) == 1:
            return self._set_at_str(
                key[0], value, idx, validated=validated, non_blocking=non_blocking
            )
        if key[0] not in self.keys():
            # this won't work
            raise KeyError(f"key {key} not found in set_at_ with tensordict {self}.")
        else:
            td = self._get_str(key[0], NO_DEFAULT)
        td._set_at_tuple(
            key[1:], value, idx, validated=validated, non_blocking=non_blocking
        )
        return self

    # @_cache_while_locked  # noqa: B019
    def keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
        *,
        sort: bool = False,
    ) -> _TensorDictKeysView:
        return self._source.keys(
            include_nested=include_nested,
            leaves_only=leaves_only,
            is_leaf=is_leaf,
            sort=sort,
        )

    def entry_class(self, key: NestedKey) -> type:
        # Let the source resolve the type: it may avoid reading the entry
        # (e.g. PersistentTensorDict only inspects the storage node).
        try:
            source_type = self._source.entry_class(key)
        except NotImplementedError:
            source_type = type(self._source.get(key))
        if _is_tensor_collection(source_type):
            return type(self)
        return source_type

    # Override: writes at self.idx of the source, in place.
    def _stack_onto_(self, list_item: list[CompatibleType], dim: int) -> _SubTensorDict:
        self._source._stack_onto_at_(list_item, dim=dim, idx=self.idx)
        return self

    # Override: a view cannot change its leaves in place nor its batch size:
    # it returns itself on its own device, and converts a to_tensordict() copy
    # otherwise.
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
            raise TypeError(
                "Cannot send a _SubTensorDict instance to device/dtype inplace."
            )
        if (
            device is not None
            and dtype is None
            and convert_to_format is None
            and device == self.device
            and (batch_size is None or torch.Size(batch_size) == self.batch_size)
        ):
            return self
        return self.to_tensordict().to(*args, non_blocking=non_blocking, **kwargs)

    def _get_non_tensor(self, key: NestedKey, default=NO_DEFAULT):
        out = super()._get_non_tensor(key, default=default)

        if isinstance(out, _SubTensorDict) and is_non_tensor(out._source):
            return out._source
        return out

    def _get_str(self, key, default, **kwargs):
        if key in self.keys() and _is_tensor_collection(self.entry_class(key)):
            data = self._source._get_str(key, NO_DEFAULT, **kwargs)
            if _pass_through(data):
                return data[self.idx]
            return _SubTensorDict(data, self.idx)
        return self._source._get_at_str(key, self.idx, default=default, **kwargs)

    def _get_tuple(self, key, default, **kwargs):
        return self._source._get_at_tuple(key, self.idx, default=default, **kwargs)

    # Override: writes the leaves that the view has into its source, in place.
    def _update_entries(
        self,
        input_dict_or_td: dict[str, CompatibleType] | TensorCollection,
        *,
        clone: bool,
        inplace: bool,
        non_blocking: bool,
        keys_to_update: list[NestedKey] | None,
        is_leaf: Callable[[Type], bool],
        update_batch_size: bool,
        ignore_lock: bool,
    ) -> _SubTensorDict:
        if getattr(self._source, "_has_exclusive_keys", False):
            raise RuntimeError(
                "Cannot use _SubTensorDict.update with a LazyStackedTensorDict that has exclusive keys."
            )
        keys = set(self.keys(False))
        for input_key, value in input_dict_or_td.items():
            key = _unravel_key_to_tuple(input_key)
            firstkey, subkey = key[0], key[1:]
            if keys_to_update and not any(
                firstkey == ktu if isinstance(ktu, str) else firstkey == ktu[0]
                for ktu in keys_to_update
            ):
                continue
            if clone and hasattr(value, "clone"):
                value = value.clone()
            elif clone:
                value = tree_map(torch.clone, value)
            elif value is self and (keys_to_update is None or firstkey not in keys):
                # with keys_to_update, a merge of self into a sub-tensordict
                # can leave out the entry that holds the sub-tensordict
                raise ValueError(_SELF_NESTING_ERROR.format(input_key))
            # the key must be a string by now. Let's check if it is present
            if firstkey in keys:
                target_class = self.entry_class(firstkey)
                if _is_tensor_collection(target_class):
                    target = self._source.get(firstkey)._get_sub_tensordict(self.idx)
                    if len(subkey):
                        sub_keys_to_update = _prune_selected_keys(
                            keys_to_update, firstkey
                        )
                        target.update(
                            {subkey: value},
                            inplace=inplace,
                            keys_to_update=sub_keys_to_update,
                            non_blocking=non_blocking,
                            is_leaf=is_leaf,
                            update_batch_size=update_batch_size,
                            ignore_lock=ignore_lock,
                        )
                        continue
                    elif isinstance(value, dict) or _is_tensor_collection(type(value)):
                        sub_keys_to_update = _prune_selected_keys(
                            keys_to_update, firstkey
                        )
                        target.update(
                            value,
                            inplace=inplace,
                            keys_to_update=sub_keys_to_update,
                            non_blocking=non_blocking,
                            update_batch_size=update_batch_size,
                            ignore_lock=ignore_lock,
                        )
                        continue
                    raise ValueError(
                        f"Tried to replace a tensordict with an incompatible object of type {type(value)}"
                    )
                else:
                    self._set_tuple(
                        key,
                        value,
                        inplace=True,
                        validated=False,
                        non_blocking=non_blocking,
                    )
            else:
                self._set_tuple(
                    key,
                    value,
                    inplace=BEST_ATTEMPT_INPLACE if inplace else False,
                    validated=False,
                    non_blocking=non_blocking,
                )
        return self

    # Override: writes into the source at the index of the view.
    def _update_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | TensorCollection,
        *,
        clone: bool,
        non_blocking: bool,
        keys_to_update: list[tuple[str, ...]] | None,
    ) -> _SubTensorDict:
        return self._update_source_at_(
            input_dict_or_td,
            self.idx,
            clone=clone,
            non_blocking=non_blocking,
            keys_to_update=keys_to_update,
        )

    def _update_source_at_(
        self,
        input_dict: dict[str, CompatibleType] | TensorCollection,
        idx: IndexType,
        *,
        clone: bool = False,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
    ) -> _SubTensorDict:
        """Writes the leaves of ``input_dict`` into the source at ``idx``, an index of the source.

        ``update_at_(input_dict, idx, discard_idx_attr=True)`` calls it, and
        :meth:`_update_` with the index of the view.
        """
        if keys_to_update is not None:
            if len(keys_to_update) == 0:
                return self
            keys_to_update = [
                _unravel_key_to_tuple(key) for key in unravel_key_list(keys_to_update)
            ]
        if isinstance(input_dict, dict):
            input_dict = _flatten_nested_dicts(input_dict)
        for key, value in input_dict.items():
            key = _unravel_key_to_tuple(key)
            if keys_to_update:
                value = _select_entry(key, value, keys_to_update)
                if value is None:
                    continue
            if not _is_accepted_class(type(value)):
                raise TypeError(
                    "Expected value to be a Tensor, a TensorDictBase or a "
                    f"tensorclass but got {type(value)}"
                )
            if clone:
                value = value.clone()
            self._source._set_at_tuple(
                key, value, idx, non_blocking=non_blocking, validated=False
            )
        return self

    def get_parent_tensordict(self) -> Self:
        if not isinstance(self._source, TensorDictBase):
            raise TypeError(
                f"_SubTensorDict was initialized with a source of type"
                f" {type(self._source).__name__}, "
                "parent tensordict not accessible"
            )
        return self._source

    # Override: deletes the entry of the source.
    def _del(self, key: tuple[str, ...]) -> None:
        self._source.del_(key)

    def _clone(self, recurse: bool = True) -> _SubTensorDict:
        """Clones the _SubTensorDict.

        Args:
            recurse (bool, optional): if ``True`` (default), a regular
                :class:`~.tensordict.TensorDict` instance will be created from the :class:`~.tensordict._SubTensorDict`.
                Otherwise, another :class:`~.tensordict._SubTensorDict` with identical content
                will be returned.

        Examples:
            >>> data = TensorDict({"a": torch.arange(4).reshape(2, 2,)}, batch_size=[2, 2])
            >>> sub_data = data._get_sub_tensordict([0,])
            >>> print(sub_data)
            _SubTensorDict(
                fields={
                    a: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.int64, is_shared=False)},
                batch_size=torch.Size([2]),
                device=None,
                is_shared=False)
            >>> # the data of both subtensordict is the same
            >>> print(data.get("a").data_ptr(), sub_data.get("a").data_ptr())
            140183705558208 140183705558208
            >>> sub_data_clone = sub_data.clone(recurse=True)
            >>> print(sub_data_clone)
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.int64, is_shared=False)},
                batch_size=torch.Size([2]),
                device=None,
                is_shared=False)
            >>. print(sub_data.get("a").data_ptr())
            140183705558208
            >>> sub_data_clone = sub_data.clone(recurse=False)
            >>> print(sub_data_clone)
            _SubTensorDict(
                fields={
                    a: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.int64, is_shared=False)},
                batch_size=torch.Size([2]),
                device=None,
                is_shared=False)
            >>> print(sub_data.get("a").data_ptr())
            140183705558208
        """
        if not recurse:
            return _SubTensorDict(
                source=self._source._clone(recurse=False), idx=self.idx
            )
        return self.to_tensordict()

    # Override: keeps accepting, and ignoring, the inplace keyword argument
    # of the expand that _SubTensorDict had.
    def expand(self, *args: int, inplace: bool = False) -> Self:
        return super().expand(*args)

    # Override: a view of another tensordict cannot be built from a dict.
    @classmethod
    def from_dict(
        cls,
        input_dict,
        *others,
        auto_batch_size: bool = False,
        batch_size=None,
        device=None,
        batch_dims=None,
        names=None,
    ):
        raise NotImplementedError(f"from_dict not implemented for {cls.__name__}.")

    # Override: the source checks its lock and the keys, and renames.
    def rename_key_(
        self, old_key: NestedKey, new_key: NestedKey, safe: bool = False
    ) -> _SubTensorDict:
        self._source.rename_key_(old_key, new_key, safe=safe)
        return self

    # Override: a view over another tensordict cannot detach in place.
    def _detach_(self) -> None:
        raise RuntimeError("Detaching a sub-tensordict in-place cannot be done.")

    # Override: writes each filled entry with set_, which raises on a file
    # opened read-only, where the base write-back would warn and skip it.
    # Delete in 0.17, when in-place writes to read-only storage raise
    # everywhere.
    def _masked_fill_(self, mask: Tensor, value: float | bool) -> Self:
        for key, item in self.items():
            self.set_(key, item.masked_fill(expand_as_right(mask, item), value))
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
    ) -> Self:
        if prefix is not None:

            def save_metadata(prefix=prefix, self=self):
                prefix = Path(prefix)
                if not prefix.exists():
                    os.makedirs(prefix, exist_ok=True)
                with open(prefix / "meta.json", "wb") as f:
                    from tensordict._utils_key_json import json_dumps

                    metadata_json = json_dumps(
                        {
                            "_type": str(type(self)),
                            "index": _index_to_str(self.idx),
                        }
                    )
                    if isinstance(metadata_json, str):
                        metadata_json = metadata_json.encode("utf-8")
                    f.write(metadata_json)

            if executor is None:
                save_metadata()
            else:
                futures.append(executor.submit(save_metadata))

        _source = self._source._memmap_(
            prefix=prefix / "_source" if prefix is not None else None,
            copy_existing=copy_existing,
            executor=executor,
            futures=futures,
            inplace=inplace,
            like=like,
            share_non_tensor=share_non_tensor,
            existsok=existsok,
            robust_key=robust_key,
        )
        if not inplace:
            result = _SubTensorDict(_source, idx=self.idx)
        else:
            result = self
        return result

    @classmethod
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
        index = _str_to_index(metadata["index"])
        if out is not None:
            if not isinstance(out, _SubTensorDict):
                raise TypeError(
                    f"Loading a _SubTensorDict into a {type(out).__name__} output is not supported. "
                    "Provide a _SubTensorDict as `out` or None."
                )
            TensorDict.load_memmap(
                prefix / "_source",
                device=device,
                out=out._source,
                robust_key=robust_key,
                allow_pickle=allow_pickle,
                mode=mode,
            )
            return out
        return _SubTensorDict(
            TensorDict.load_memmap(
                prefix / "_source",
                device=device,
                robust_key=robust_key,
                allow_pickle=allow_pickle,
                mode=mode,
            ),
            index,
        )

    # Override: a view over another tensordict cannot be shared.
    def _share_memory_(self) -> None:
        raise RuntimeError(
            "Casting a sub-tensordict values to shared memory cannot be done."
        )

    # Override: only the parent tensordict can be unlocked, so a raise in
    # _propagate_unlock would also stop the unlock_ of a tensordict that
    # holds this view.
    @_as_context_manager("is_locked")
    def unlock_(self) -> Self:
        if self.is_locked:
            raise RuntimeError(
                "Cannot unlock a _SubTensorDict. Unlock the parent tensordict instead."
            )
        return self

    # Override: the lock is the source's; a view holds nothing to unlock.
    def _propagate_unlock(self):
        return []

    def _remove_lock(self, lock_id):
        raise RuntimeError(
            "Cannot unlock a _SubTensorDict. Unlock the parent tensordict instead."
        )

    # Override: only the parent tensordict can be locked.
    def _propagate_lock(self, lock_ids=None, *, is_compiling):
        raise RuntimeError(
            "Cannot lock a _SubTensorDict. Lock the parent tensordict instead."
        )

    def _create_nested_str(self, key):
        # this may fail with a sub-sub tensordict
        out = self._source.empty()
        self._source._set_str(
            key, out, inplace=False, validated=True, non_blocking=False
        )
        # the id of out changes
        return self._get_str(key, default=NO_DEFAULT)

    def _cast_reduction(
        self,
        *,
        reduction_name,
        dim=NO_DEFAULT,
        keepdim=NO_DEFAULT,
        tuple_ok=True,
        **kwargs,
    ):
        try:
            td = self.to_tensordict()
        except Exception:
            raise RuntimeError(
                f"{reduction_name} requires this object to be cast to a regular TensorDict. "
                f"If you need {type(self).__name__} to support {reduction_name}, help us by filing an issue"
                f" on github!"
            )
        return td._cast_reduction(
            reduction_name=reduction_name,
            dim=dim,
            keepdim=keepdim,
            tuple_ok=tuple_ok,
            **kwargs,
        )

    _add_batch_dim = TensorDict._add_batch_dim

    def _index_tensordict(
        self,
        index: IndexType,
        new_batch_size: torch.Size | None = None,
        names: Sequence[str] | None = None,
    ) -> _SubTensorDict:
        # we ignore the names and new_batch_size which are only provided for
        # efficiency purposes
        return self._get_sub_tensordict(index)

    def _remove_batch_dim(self, *args, **kwargs):
        raise NotImplementedError

    def _maybe_remove_batch_dim(self, *args, **kwargs):
        raise NotImplementedError


###########################
# Keys utils


class _TensorDictKeysView:
    """The view of the keys that :meth:`~tensordict.TensorDictBase.keys` returns for nested or filtered keys.

    It holds a reference to the tensordict, so it shows the keys that are
    added later. With ``include_nested=True``, a membership check accepts
    nested keys: ``("a", "b") in td.keys(include_nested=True)`` is ``True``
    if ``td["a", "b"]`` exists. (The plain ``td.keys()`` view accepts only
    top-level string keys.)

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> td = TensorDict(
        ...     a=TensorDict(b=torch.zeros(1, 2), batch_size=[1, 2]),
        ...     c=torch.zeros(1),
        ...     batch_size=[1],
        ... )
        >>> keys = td.keys(include_nested=True)
        >>> ("a", "b") in keys, ("a", "c") in keys
        (True, False)
        >>> td["d"] = torch.zeros(1)
        >>> list(keys)
        [('a', 'b'), 'a', 'c', 'd']
    """

    def __init__(
        self,
        tensordict: T,
        include_nested: bool,
        leaves_only: bool,
        is_leaf: Callable[[Type], bool] | None = None,
        sort: bool = False,
    ) -> None:
        self.tensordict = tensordict
        self.include_nested = include_nested
        self.leaves_only = leaves_only
        if is_leaf is None:
            is_leaf = _default_is_leaf
        self.is_leaf = is_leaf
        self.sort = sort

    def __iter__(self) -> Iterator[str | tuple[str, ...]]:
        # Not a generator: iterating returns the underlying iterator directly,
        # which saves a generator frame per key.
        if self.sort:

            def keyfunc(key):
                return ".".join(key) if isinstance(key, tuple) else key

            if self.include_nested and type(self.tensordict) is TensorDict:
                # sorted() reads every key anyway: list them in one walk.
                keys = self._nested_key_list(self.tensordict, None, [])
            else:
                keys = self._iter()
            return iter(sorted(keys, key=keyfunc))
        return self._iter()

    def _iter(self) -> Iterator[str | tuple[str, ...]]:
        if self.include_nested:
            return self._iter_helper(self.tensordict)
        if self.leaves_only:
            return self._iter_leaves()
        return iter(self._keys())

    def _iter_leaves(self) -> Iterator[str]:
        for key in self._keys():
            target_class = self.tensordict.entry_class(key)
            if not self.is_leaf(target_class):
                continue
            yield key

    def _iter_helper(
        self, tensordict: T, prefix: tuple | None = None
    ) -> Iterable[str | tuple[str, ...]]:
        # Yields the keys of the first level as str and the nested keys as tuples.
        leaves_only = self.leaves_only
        is_leaf = self.is_leaf
        for key, value in self._items(tensordict):
            cls = type(value)
            if cls is Tensor:
                # A plain tensor is never a tensor collection: skip the checks.
                if not leaves_only or is_leaf is _default_is_leaf or is_leaf(cls):
                    yield key if prefix is None else prefix + (key,)
                continue
            full_key = (key,) if prefix is None else prefix + (key,)
            if cls is TensorDict:
                # Skip the checks below: a TensorDict is a tensor collection
                # to recurse into, and not a leaf for the default is_leaf.
                yield from self._iter_helper(value, prefix=full_key)
                if not leaves_only or (
                    is_leaf is not _default_is_leaf and is_leaf(cls)
                ):
                    yield key if prefix is None else full_key
                continue
            while cls is list:
                # For lazy stacks
                value = value[0]
                cls = type(value)
            if _is_tensor_collection(cls):
                # Don't recurse into non-tensor or pass-through values
                if not _pass_through(value):
                    yield from self._iter_helper(value, prefix=full_key)
            if not leaves_only or is_leaf(cls):
                yield key if prefix is None else full_key

    def _nested_key_list(
        self, tensordict: T, prefix: tuple | None, result: list
    ) -> list[str | tuple[str, ...]]:
        # Appends to result the keys that _iter_helper() yields, in its order,
        # in one walk without a generator per level. For the callers that read
        # every key before they return.
        leaves_only = self.leaves_only
        is_leaf = self.is_leaf
        for key, value in self._items(tensordict):
            cls = type(value)
            if cls is Tensor:
                if not leaves_only or is_leaf is _default_is_leaf or is_leaf(cls):
                    result.append(key if prefix is None else prefix + (key,))
                continue
            full_key = (key,) if prefix is None else prefix + (key,)
            if cls is TensorDict:
                # Skip the checks below: a TensorDict is a tensor collection
                # to recurse into, and not a leaf for the default is_leaf.
                self._nested_key_list(value, full_key, result)
                if not leaves_only or (
                    is_leaf is not _default_is_leaf and is_leaf(cls)
                ):
                    result.append(key if prefix is None else full_key)
                continue
            while cls is list:
                value = value[0]
                cls = type(value)
            if _is_tensor_collection(cls):
                if not _pass_through(value):
                    self._nested_key_list(value, full_key, result)
            if not leaves_only or is_leaf(cls):
                result.append(key if prefix is None else full_key)
        return result

    def __len__(self) -> int:
        if self.include_nested and type(self.tensordict) is TensorDict:
            return len(self._nested_key_list(self.tensordict, None, []))
        return sum(1 for _ in self)

    def _items(
        self, tensordict: TensorDictBase | None = None
    ) -> Iterable[tuple[NestedKey, CompatibleType]]:
        if tensordict is None:
            tensordict = self.tensordict
        if type(tensordict) is TensorDict:
            return tensordict._tensordict.items()
        if is_tensorclass(tensordict):
            tensordict = tensordict._tensordict
        if isinstance(tensordict, TensorDict):
            return tensordict._tensordict.items()
        # Imported lazily: TypedTensorDict delegates its storage to this module.
        from tensordict.typedtensordict import TypedTensorDict

        if isinstance(tensordict, TypedTensorDict):
            return self._items(tensordict._source)
        from tensordict.nn import TensorDictParams

        if isinstance(tensordict, TensorDictParams):
            assert tensordict._param_td is not None
            return tensordict._param_td.items()
        from tensordict._lazy import (
            _CustomOpTensorDict,
            _iter_items_lazystack,
            LazyStackedTensorDict,
        )

        if isinstance(tensordict, LazyStackedTensorDict):
            return _iter_items_lazystack(tensordict, return_none_for_het_values=True)
        if isinstance(tensordict, _CustomOpTensorDict):
            # it's possible that a TensorDict contains a nested LazyStackedTensorDict,
            # or _CustomOpTensorDict, so as we iterate through the contents we need to
            # be careful to not rely on tensordict._tensordict existing.
            return (
                (key, tensordict._get_str(key, NO_DEFAULT))
                for key in tensordict._source.keys()
            )
        from tensordict.persistent import PersistentTensorDict

        if isinstance(tensordict, PersistentTensorDict):
            return (
                (key, tensordict._get_str(key, NO_DEFAULT)) for key in tensordict.keys()
            )
        raise NotImplementedError(type(tensordict))

    def _keys(self) -> _TensorDictKeysView:
        return self.tensordict._tensordict.keys()

    def __contains__(self, key: NestedKey) -> bool:
        key = _unravel_key_to_tuple(key)
        if not key:
            raise TypeError(_NON_STR_KEY_ERR)

        if isinstance(key, str):
            if key in self._keys():
                if self.leaves_only:
                    # TODO: make this faster for LazyStacked without compromising regular
                    return not _is_tensor_collection(
                        type(self.tensordict._get_str(key, NO_DEFAULT))
                    )
                return True
            return False
        else:
            # thanks to _unravel_key_to_tuple we know the key is a tuple
            if len(key) == 1:
                return key[0] in self._keys()
            elif self.include_nested:
                item_root = self.tensordict._get_str(key[0], default=None)
                if item_root is not None:
                    entry_type = type(item_root)
                    if issubclass(entry_type, Tensor):
                        return False
                    # TODO: make this faster for LazyStacked without compromising regular
                    _is_tensordict = _is_tensor_collection(entry_type)
                    if _is_tensordict:
                        # # this will call _unravel_key_to_tuple many times
                        # return key[1:] in self.tensordict._get_str(key[0], NO_DEFAULT).keys(include_nested=self.include_nested)
                        # this won't call _unravel_key_to_tuple but requires to get the default which can be suboptimal
                        if len(key) >= 3:
                            leaf_td = item_root._get_tuple(key[1:-1], None)
                            if leaf_td is None or (
                                not _is_tensor_collection(type(leaf_td))
                            ):
                                return False
                        else:
                            leaf_td = item_root
                        return key[-1] in leaf_td.keys()
                return False
            # A nested key is not among the keys of a view without nesting
            return False

    def __repr__(self):
        include_nested = f"include_nested={self.include_nested}"
        leaves_only = f"leaves_only={self.leaves_only}"
        return f"{type(self).__name__}({list(self)},\n{indent(include_nested, 4 * ' ')},\n{indent(leaves_only, 4 * ' ')})"


def _is_stack_of_batchless_non_tensors(dest) -> bool:
    # A NonTensorStack whose NonTensorData leaves have no batch dims, which
    # maybe_to_stack() would rebuild unchanged. The members of a stack are
    # NonTensorData (tensorclasses) or NonTensorStacks.
    if not is_non_tensor(dest) or is_tensorclass(dest):
        return False
    batch_dims = dest.batch_dims
    if batch_dims == 1:
        return True
    members = dest.tensordicts
    # read the class attribute: is_tensorclass is slower, once per member
    if any(getattr(type(td), "_is_tensorclass", False) for td in members):
        return False
    return batch_dims == 2 or all(map(_is_stack_of_batchless_non_tensors, members))


def _index_to_str(index: IndexType) -> Any:
    # Tag containers explicitly: JSON round-trips tuples as lists, so an
    # untagged encoding cannot distinguish a tuple of indices from an
    # advanced (list) index when loading back.
    if isinstance(index, tuple):
        return ("tuple", [_index_to_str(elt) for elt in index])
    if isinstance(index, list):
        return ("list", [_index_to_str(elt) for elt in index])
    if isinstance(index, slice):
        return ("slice", {"start": index.start, "stop": index.stop, "step": index.step})
    if isinstance(index, range):
        return ("range", {"start": index.start, "stop": index.stop, "step": index.step})
    if isinstance(index, Tensor):
        return ("tensor", index.tolist(), str(index.device))
    if index is Ellipsis:
        return ("ellipsis",)
    return index


def _str_to_index(index: Any) -> IndexType:
    if isinstance(index, (tuple, list)):
        if not len(index):
            return ()
        tag = index[0]
        if tag == "tuple":
            return tuple(_str_to_index(elt) for elt in index[1])
        if tag == "list":
            return [_str_to_index(elt) for elt in index[1]]
        if tag == "slice":
            content = index[1]
            return slice(content["start"], content["stop"], content["step"])
        if tag == "range":
            content = index[1]
            return range(content["start"], content["stop"], content["step"])
        if tag == "tensor":
            content, device = index[1:]
            return torch.tensor(content, device=device)
        if tag == "ellipsis":
            return Ellipsis
        # Legacy (untagged) metadata: a sequence of indices saved as a tuple.
        return tuple(_str_to_index(elt) for elt in index)
    return index


_register_tensor_class(TensorDict)
_set_tensordict_type(TensorDict)
_set_subtensordict_type(_SubTensorDict)
_register_tensor_class(_SubTensorDict)


from tensordict._td_functions import (  # noqa: F401
    cat,
    from_consolidated,
    from_module,
    from_modules,
    from_pytree,
    fromkeys,
    lazy_stack,
    load,
    load_memmap,
    maybe_dense_stack,
    memmap,
    save,
    stack,
)
from tensordict._td_memmap import (
    _populate_empty,
    _populate_memmap,
    _populate_storage,
    _save_metadata,
    _update_metadata,
)

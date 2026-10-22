# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Conversions to and from other containers, file formats and nn.Module parameters of :class:`~tensordict.TensorDictBase`.

The methods live on a mixin that ``tensordict.base`` imports before it
defines ``TensorDictBase``, so this module imports the helpers it needs from
``tensordict.base`` without needing the class itself. A method that uses
``TensorDictBase`` at run time imports it locally. The comment above the
mixin imports in ``tensordict/base.py`` gives the other rules.
"""

from __future__ import annotations

import collections
import importlib.util
import numbers
import sys
import uuid
import warnings
import weakref
from collections import UserDict
from copy import copy
from typing import Any, List, Literal, Sequence, Type, TYPE_CHECKING

import numpy as np
import tensordict.base as _td_base
import torch
from tensordict._datasets import to_mds
from tensordict._nestedkey import NestedKey
from tensordict._tabular import (
    _columns_to_tensordict,
    _dataframe_to_tensordict,
    _read_csv,
    _read_json,
    _read_parquet,
    _tensordict_to_dataframe,
    _write_csv,
    _write_json,
    _write_parquet,
)
from tensordict.base import (
    __base__setattr__,
    _has_h5,
    _has_top_level_exclusive_keys,
    _is_leaf_nontensor,
    _is_tensor_collection,
    _maybe_preserve_module_state,
    _PACKAGE_DIR,
    _raise_on_exclusive_keys,
    _set_tensor_dict,
    _wrap_extra_state,
    Buffer,
    is_compiling,
    is_tensor_collection,
    NO_DEFAULT,
    Self,
)
from tensordict.utils import (
    _as_context_manager,
    _is_dataclass as is_dataclass,
    _is_list_tensor_compatible,
    _is_namedtuple,
    _is_namedtuple_class,
    _is_non_tensor,
    _is_unbatched,
    _maybe_correct_neg_dim,
    _set_max_batch_size,
    _user_stacklevel,
    _warn_batch_dims_ignored,
    is_non_tensor,
    LinkedList,
    set_lazy_legacy,
)
from torch import nn, Tensor
from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX
from torch.nn.parameter import UninitializedTensorMixin
from torch.nn.utils._named_member_accessor import swap_tensor
from torch.utils._pytree import is_structseq_instance, tree_map

if TYPE_CHECKING:
    from tensordict.base import TensorDictBase

try:
    from functorch import dim as ftdim
except ImportError:
    from tensordict.utils import _ftdim_mock as ftdim


def _int_batch_dims(batch_dims):
    # for backward compatibility, from_any ignores a batch_dims that is not an
    # int for dataclasses and h5py files
    if isinstance(batch_dims, numbers.Integral) and not isinstance(batch_dims, bool):
        return batch_dims
    return None


def _dict_holds_sequence(value: dict) -> bool:
    # from_pytree turns a list or tuple inside a dict into a nested TensorDict,
    # where from_any (used for namedtuple fields on main) keeps a tensor
    return any(
        _dict_holds_sequence(val)
        if isinstance(val, dict)
        else isinstance(val, (list, tuple)) and not _is_namedtuple(val)
        for val in value.values()
    )


def _holds_empty(value) -> bool:
    # from_pytree gives an empty dict or namedtuple a _pytree_type entry, so
    # auto_batch_size finds no batch dims; from_any leaves it empty, as on main
    if _is_namedtuple(value):
        value = value._asdict()
    elif not isinstance(value, dict):
        return False
    return not value or any(_holds_empty(val) for val in value.values())


def _pytree_field(value) -> bool:
    # a namedtuple field (or tuple element) that from_pytree converts so that
    # to_pytree can rebuild it, and that from_any would convert to the same
    # entries, without the _pytree_type ones
    return (
        (isinstance(value, dict) and not _dict_holds_sequence(value))
        or _is_namedtuple(value)
    ) and not _holds_empty(value)


def _dict_to_namedtuple(dictionary: dict, dest_cls: type | None, root: bool = False):
    # the conversion of to_namedtuple(dest_cls=...), at a nested level unless root
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = _dict_to_namedtuple(value, dest_cls)
    cls = dest_cls
    if cls is not None and not root and hasattr(cls, "_fields"):
        # a nested level uses dest_cls only if dest_cls takes its keys
        keys = set(dictionary)
        defaults = getattr(cls, "_field_defaults", {})
        if not (keys <= set(cls._fields) and set(cls._fields) - keys <= set(defaults)):
            cls = None
    if cls is None:
        cls = collections.namedtuple("GenericDict", dictionary.keys())
    return cls(**dictionary)


def _pytree_dict_key(key):
    # Reads a dict key that from_pytree stored. Once the non-tensor entry has
    # batch dims (e.g. after stack or unsqueeze), td["key"] returns a list, so
    # read .data. memmap saves tuples as JSON lists, which are not hashable.
    if is_non_tensor(key):
        key = key.data
    if isinstance(key, list):
        return tuple(_pytree_dict_key(k) for k in key)
    return key


def _warn_structseq_keys(obj) -> None:
    """Warns that ``from_any`` will key a structseq by its field names."""
    # point to the first frame outside tensordict: the user's call
    frame, stacklevel = sys._getframe(1), 2
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame, stacklevel = frame.f_back, stacklevel + 1
    obj_type = type(obj)
    positional = [str(i) for i in range(len(obj))]
    named = list(obj_type.__match_args__)
    warnings.warn(
        f"from_any converts a {obj_type.__module__}.{obj_type.__name__} like a "
        f"plain tuple, with the keys {positional}. Starting with TensorDict "
        f"0.17, the default will change to its field names {named}, as "
        "from_namedtuple gives. Pass it to from_namedtuple to get the field "
        "names now, or convert it with tuple() to keep the keys of a tuple.",
        FutureWarning,
        stacklevel=stacklevel,
    )


def _swap_extra_state(module, value, return_swap):
    """Loads the extra state ``value`` into ``module`` and returns the old one if ``return_swap``.

    ``value`` is the extra state that ``from_module(use_state_dict=True)``
    saves with ``get_extra_state()``. It is loaded as ``load_state_dict``
    does.
    """
    out = None
    if return_swap:
        # Clone the tensors: set_extra_state may copy into them.
        out = _wrap_extra_state(
            tree_map(
                lambda x: x.clone() if isinstance(x, Tensor) else x,
                module.get_extra_state(),
            )
        )
    if is_tensor_collection(value):
        value = value.to_dict()
    module.set_extra_state(value)
    return out


def _swap_into_module(td, module, memo, non_blocking, preserve_module_state):
    """Writes the leaves of ``td`` into ``module`` and returns the old ones in a dict.

    The fast path of ``_to_module`` for a module with the native
    ``__setattr__``, with ``return_swap=True``, not in place, without
    ``use_state_dict`` and outside dynamo. It recurses into the submodules
    without going through ``_to_module``, whose checks cost more than the swap
    of the leaves.
    """
    hooks = memo["hooks"]
    __dict__ = module.__dict__
    _parameters = __dict__["_parameters"]
    _buffers = __dict__["_buffers"]
    _modules = __dict__["_modules"]
    swap = memo[weakref.ref(module)] = {}
    for key, value in td.items():
        if key == _EXTRA_STATE_KEY_SUFFIX:
            local_out = _swap_extra_state(module, value, True)
        elif isinstance(value, (Tensor, ftdim.Tensor)):
            local_out = _set_tensor_dict(
                __dict__,
                _parameters,
                _buffers,
                hooks,
                module,
                key,
                value,
                False,
                return_swap=True,
                preserve_module_state=preserve_module_state,
            )
        else:
            child = _modules[key]
            local_out = memo.get(weakref.ref(child), NO_DEFAULT)
            if local_out is NO_DEFAULT:
                if (
                    type(value)._to_module is _Conversion._to_module
                    and type(child).__setattr__ is __base__setattr__
                    and not _is_tensor_collection(type(child))
                ):
                    local_out = _swap_into_module(
                        value, child, memo, non_blocking, preserve_module_state
                    )
                else:
                    local_out = value._to_module(
                        child,
                        swap_dest={},
                        memo=memo,
                        non_blocking=non_blocking,
                        preserve_module_state=preserve_module_state,
                        is_dynamo=False,
                    )
            else:
                memo["shared"] = True
        swap[key] = local_out
    return swap


def _replaced_entries(
    swap: dict,
    td,
    entries: list,
    skip_wrappers: bool = False,
    seen: dict | None = None,
    source=None,
) -> list:
    """Lists the ``(sub-tensordict, key, value)`` of ``swap`` that are not the values of ``td``.

    With ``skip_wrappers``, a value that views the storage of the value of
    ``td`` with the same shape, strides, offset and dtype, such as the
    ``nn.Parameter`` that ``to_module`` wraps around a plain tensor, is not
    listed, unless both require grad: the gradient of the value of ``td``
    then lands on the wrapper, which is written back.

    With a ``seen`` dict, given when the module uses a submodule more than
    once, the submodule holds the values of its first entry in ``td``, the
    ``source`` of the later entries: these values and their wrappers are not
    listed for the later entries either, which keep their own values.
    """
    for key, val in swap.items():
        if key == _EXTRA_STATE_KEY_SUFFIX:
            # Restored with set_extra_state, nothing to write.
            continue
        if isinstance(val, dict):
            sub = td._get_str(key, default=NO_DEFAULT)
            # to_module swaps a shared submodule once and gives each of its
            # entries the same dict; it sets memo["shared"] when it does.
            _replaced_entries(
                val,
                sub,
                entries,
                skip_wrappers,
                seen,
                None if seen is None else seen.setdefault(id(val), sub),
            )
            continue
        current = td._get_str(key, None)
        if current is not val and not (
            skip_wrappers
            and isinstance(current, Tensor)
            and isinstance(val, Tensor)
            and not (current.requires_grad and val.requires_grad)
            and torch._C._is_alias_of(val, current)
            and val.shape == current.shape
            and val.stride() == current.stride()
            and val.storage_offset() == current.storage_offset()
            and val.dtype == current.dtype
        ):
            if source is not None and source is not td:
                current = source._get_str(key, None)
                if current is val or (
                    isinstance(current, Tensor)
                    and isinstance(val, Tensor)
                    and torch._C._is_alias_of(val, current)
                    and val.shape == current.shape
                    and val.stride() == current.stride()
                    and val.storage_offset() == current.storage_offset()
                    and val.dtype == current.dtype
                ):
                    continue
            entries.append((td, key, val))
    return entries


def _restore_module(swap, args, kwargs, out):
    """Writes ``swap`` back into the module when ``with out.to_module(...)`` exits.

    The tensors that the module holds at the exit and that are not in ``out``
    (for example the materialized parameters of a lazy module) are written
    into ``out``. The parameters and buffers that ``to_module`` wrapped around
    the leaves of ``out`` are not, so that ``out`` keeps its own leaves.
    ``out`` is unlocked only if there are any, as unlocking and relocking it
    costs more than the swap.
    """
    hooks = torch.nn.modules.module._global_parameter_registration_hooks
    memo = {"hooks": tuple(hooks.values())}
    current = swap._to_module(*args, **{**kwargs, "memo": memo}, swap_dest={})
    if not isinstance(current, dict):
        return current
    seen = {} if "shared" in memo else None
    replaced = _replaced_entries(current, out, [], True, seen)
    if replaced:
        non_blocking = kwargs.get("non_blocking", False)
        with out.unlock_():
            for td, key, val in replaced:
                td._set_str(
                    key, val, inplace=False, validated=True, non_blocking=non_blocking
                )
    return out


class _Conversion:
    """Conversions to and from other containers, file formats and nn.Module parameters."""

    @classmethod
    def from_list(
        cls,
        input,
        *,
        auto_batch_size: bool | None = None,
        batch_size: torch.Size | None = None,
        device: torch.device | None = None,
        batch_dims: int | None = None,
        names: list[str] | None = None,
        lazy: bool | None = None,
    ) -> Self:
        """Builds a tensordict by stacking a list of dicts or tensor collections.

        Each dict is converted with :meth:`from_dict` and the results are
        stacked along a new leading dimension. A tensordict or tensorclass in
        the list is stacked as it is: the keyword arguments do not apply to
        it. When the result does not get the ``batch_size`` or ``device``
        asked for, a ``FutureWarning`` is emitted: starting with TensorDict
        0.17, the default will change to applying them to the result, which
        raises if the batch size does not fit, as for a list of dicts.

        Args:
            input (list): the dicts or tensor collections to stack.

        Keyword Args:
            auto_batch_size (bool, optional): if ``True``, sets the batch size
                of each element to the largest leading shape that all its
                entries share. Defaults to ``None``, which behaves like
                ``False`` unless ``batch_dims`` is given.
            batch_size (torch.Size or list of int, optional): the batch size of
                the result, with the length of the list first. Each element
                gets ``batch_size[1:]``. Defaults to ``None``.
            device (torch.device, str or int, optional): the device of the
                result. Leaves on another device are moved to it. ``None``
                leaves each leaf where it is, and the result has no device.
                Defaults to ``None``.
            batch_dims (int, optional): the maximum number of batch dimensions
                of the result, the list dimension included. Turns on
                ``auto_batch_size`` when that is ``None``. Exclusive with
                ``batch_size``. Defaults to ``None``.
            names (list of str, optional): the names of the batch dimensions
                of the result, one per dimension, with the name of the list
                dimension first. Defaults to ``None``.
            lazy (bool, optional): if ``True``, the result is a lazy stack;
                if ``False``, a dense stack. Defaults to ``None``, which builds
                a dense stack when the elements have the same shape and a lazy
                stack otherwise. See :ref:`stacking`.

        Raises:
            ValueError: if ``batch_size[0]`` is not the length of the list.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict.from_list(
            ...     [{"a": torch.zeros(3)}, {"a": torch.zeros(3)}],
            ...     batch_size=[2, 3],
            ...     names=["x", "y"],
            ... )
            >>> td.batch_size
            torch.Size([2, 3])
            >>> td.names
            ['x', 'y']

        """
        from tensordict.base import TensorDictBase

        if lazy is None:
            stack = cls.maybe_dense_stack
        elif lazy:
            stack = cls.lazy_stack
        else:
            stack = torch.stack
        if batch_size is not None:
            if isinstance(batch_size, int):
                batch_size = torch.Size([batch_size])
            if batch_size[0] != len(input):
                raise ValueError(
                    f"The provided batch size ({batch_size}) does not match the length of the list ({len(input)})."
                )
            bsz = batch_size[1:]
        else:
            bsz = None
        if batch_dims is not None:
            batch_dims -= 1
        member_names = names[1:] if names is not None else None
        if cls is TensorDictBase:
            from tensordict import TensorDict

            cls = TensorDict
        input = [
            (
                cls.from_dict(
                    d,
                    auto_batch_size=auto_batch_size,
                    batch_size=bsz,
                    batch_dims=batch_dims,
                    device=device,
                    names=member_names,
                )
                if not is_tensor_collection(d)
                else d
            )
            for d in input
        ]
        out = stack(input)
        ignored = []
        if batch_size is not None and out.batch_size != torch.Size(batch_size):
            ignored.append("batch_size")
        if device is not None:
            from tensordict._td import _parse_device

            if out.device != _parse_device(device, memoize=False)[0]:
                ignored.append("device")
        if ignored:
            # the tensor collections of the list skip from_dict (#2979)
            warnings.warn(
                f"from_list ignores {' and '.join(ignored)} for the tensor "
                "collections of the list: the result keeps the batch size "
                "and device of their stack. Starting with TensorDict 0.17, "
                "the default will change to applying them to the result, "
                "which raises if the batch size does not fit, as for a list "
                "of dicts.",
                FutureWarning,
                stacklevel=_user_stacklevel(),
            )
        if names is not None:
            # the stack does not name the list dimension, and from_dict does
            # not apply names with auto_batch_size: name the result
            if len(names) == out.batch_dims:
                out.names = names
            else:
                warnings.warn(
                    f"from_list ignores names={names}, whose length is not the "
                    "number of batch dims of the result, today. Starting with TensorDict "
                    "0.17, the default will change to raising a ValueError, "
                    "as for a batch_size of another length.",
                    FutureWarning,
                    stacklevel=2,
                )
        return out

    def from_dict_instance(
        self,
        input_dict,
        *,
        auto_batch_size: bool | None = None,
        batch_size=None,
        device=None,
        batch_dims=None,
        names=None,
    ):
        """Builds a tensordict from a dictionary, with the types of the entries of this tensordict.

        Works like :meth:`~tensordict.TensorDict.from_dict`, except for the
        nested dicts: a nested dict whose key is a nested tensordict or
        tensorclass of this tensordict is converted to that type, recursively.
        The other nested dicts become :class:`~tensordict.TensorDict`
        instances. Use it to turn the output of :meth:`~.to_dict` back into
        tensorclasses. The values of this tensordict are not used, and the
        result is a :class:`~tensordict.TensorDict`.

        Args:
            input_dict (dict): the dictionary to convert. Nested dicts become
                nested entries.

        Keyword Args:
            auto_batch_size (bool, optional): if ``True``, sets the batch size
                to the largest leading shape that all the entries share.
                Defaults to ``None``, which behaves like ``False`` unless
                ``batch_dims`` is given.
            batch_size (torch.Size or list of int, optional): the batch size of
                the result: the leading dimensions that all its entries
                share. Defaults to ``None``, which gives an empty batch size
                unless the batch size is computed.
            device (torch.device, str or int, optional): the device of the
                result. Leaves on another device are moved to it. ``None``
                leaves each leaf where it is, and the result has no device.
                Defaults to ``None``.
            batch_dims (int, optional): with ``auto_batch_size``, the maximum
                number of batch dimensions. Turns on ``auto_batch_size`` when
                that is ``None``. Exclusive with ``batch_size``. Defaults to
                ``None``.
            names (list of str, optional): the names of the batch dimensions,
                one per dimension. Used only with ``batch_size``: when the
                batch size is empty or computed, the names are dropped.
                Defaults to ``None``.

        Raises:
            ValueError: if both ``batch_size`` and ``batch_dims`` are given.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict, tensorclass
            >>> @tensorclass
            ... class MyClass:
            ...     x: torch.Tensor
            >>> td = TensorDict(a=torch.zeros(()), b=MyClass(x=torch.zeros(())))
            >>> data = {"a": torch.ones(()), "b": {"x": torch.ones(())}}
            >>> type(td.from_dict_instance(data)["b"]).__name__
            'MyClass'
            >>> type(TensorDict.from_dict(data)["b"]).__name__
            'TensorDict'

        """
        if batch_dims is not None and batch_size is not None:
            raise ValueError(
                "Cannot pass both batch_size and batch_dims to `from_dict`."
            )
        from tensordict import TensorDict

        batch_size_set = torch.Size(()) if batch_size is None else batch_size
        if is_compiling():
            input_dict = type(input_dict)(input_dict)
        else:
            input_dict = copy(input_dict)
        for key, value in list(input_dict.items()):
            if isinstance(value, (dict,)):
                cur_value = self.get(key)
                if cur_value is not None:
                    input_dict[key] = cur_value.from_dict_instance(
                        value,
                        device=device,
                        auto_batch_size=False,
                    )
                    continue
                else:
                    # we don't know if another tensor of smaller size is coming
                    # so we can't be sure that the batch-size will still be valid later
                    input_dict[key] = TensorDict.from_dict(
                        value,
                        device=device,
                        auto_batch_size=False,
                    )
            else:
                input_dict[key] = TensorDict.from_any(
                    value,
                    auto_batch_size=False,
                )

        out = TensorDict._from_dict_impl(
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
    def from_pytree(
        cls,
        pytree,
        *,
        batch_size: torch.Size | None = None,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
    ):
        """Converts a pytree to a tensordict that :meth:`~.to_pytree` converts back.

        Each list, tuple, named tuple and dict of the pytree becomes a nested
        tensordict, and its type is stored in a ``"_pytree_type"`` entry so
        that :meth:`~.to_pytree` can rebuild the pytree. The elements of a list
        or tuple get the keys ``"0"``, ``"1"``, ... The subclasses of these
        types, such as :class:`collections.OrderedDict`,
        :class:`collections.defaultdict` and the ``torch.return_types``
        results of ``torch.max`` and the like, are accepted too. A
        ``defaultdict`` is rebuilt without its ``default_factory``.

        A tensor or a tensor collection is returned as it is. Python ints and
        floats and NumPy arrays become tensors, so :meth:`~.to_pytree` returns
        them as tensors. A dict key that is not a :class:`NestedKey` (such as an
        int or an object) is stored in a :class:`~tensordict.NonTensorData`
        entry next to its value. Other types raise a ``NotImplementedError``.

        Args:
            pytree: the pytree to convert: a list, tuple, named tuple or dict,
                nested to any depth, whose leaves are tensors, tensor
                collections, Python numbers or NumPy arrays.

        Keyword Args:
            batch_size (torch.Size or list of int, optional): the batch size of
                the result: the leading dimensions that all its entries share.
                Defaults to ``None``, which gives an empty batch size unless
                ``auto_batch_size=True``.
            auto_batch_size (bool, optional): if ``True``, sets the batch size
                to the largest leading shape that all the entries share.
                Defaults to ``False``.
            batch_dims (int, optional): with ``auto_batch_size``, the maximum
                number of batch dimensions. Ignored without
                ``auto_batch_size``, with a ``FutureWarning`` when it would
                change the result. Starting with TensorDict 0.17, the default
                will change to turning on ``auto_batch_size`` when
                ``batch_dims`` is given, as ``from_dict`` does. Defaults to
                ``None``.

        .. note::
            A module ``state_dict()`` round-trips through ``from_pytree``, and its
            :meth:`~.memmap` output loads with ``load_memmap(path, allow_pickle=False)``.
            The state dicts of an optimizer and of a learning-rate scheduler hold
            ``None``, ``str`` values and Python numbers, which ``from_pytree`` rejects
            (``None`` and ``str``) or returns as tensors (numbers): save them with
            ``torch.save(sd, path)`` and load them with ``torch.load(path, weights_only=True)``,
            which round-trips them exactly without unpickling arbitrary objects.

        .. note::
            The list fields of a named tuple, and some of its dict and tuple
            fields, are converted with :meth:`~.from_any` and do not
            round-trip through :meth:`~.to_pytree`:

            - a list of numbers comes back as a tensor;
            - an empty tuple, an empty dict or named tuple, and a dict that
              holds a str or a list come back as a ``GenericDict`` named
              tuple;
            - a dict that holds a tuple, a tuple subclass (such as a
              ``torch.return_types`` result), and a tuple that holds anything
              other than tensors, Python numbers, dicts and named tuples make
              :meth:`~.to_pytree` raise a ``ValueError``.

            The other dict, named tuple and tuple fields come back with their
            own type.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> pytree = {"obs": torch.zeros(3, 4), "pair": (torch.zeros(3), torch.ones(3))}
            >>> td = TensorDict.from_pytree(pytree, auto_batch_size=True)
            >>> td.batch_size
            torch.Size([3])
            >>> sorted(td.keys()), sorted(td["pair"].keys())
            (['_pytree_type', 'obs', 'pair'], ['0', '1', '_pytree_type'])
            >>> type(td.to_pytree()["pair"]).__name__
            'tuple'

            A pytree with lists, dicts, a named tuple and a dict key that is
            not a string round-trips:

            >>> class Key:
            ...     pass
            >>> key = Key()
            >>> pytree = (
            ...     [torch.arange(3), torch.zeros(2)],
            ...     {"tensor": torch.ones(2), key: torch.arange(2), "list": [1, 2, 3]},
            ...     {"named_tuple": TensorDict(two=torch.ones(1)).to_namedtuple()},
            ... )
            >>> pytree_recon = TensorDict.from_pytree(pytree).to_pytree()
            >>> def check(v1, v2):
            ...     assert (v1 == v2).all()
            >>> _ = torch.utils._pytree.tree_map(check, pytree, pytree_recon)
            >>> key in pytree_recon[1]
            True

        """
        if is_tensor_collection(pytree):
            return pytree
        if isinstance(pytree, (torch.Tensor,)):
            return pytree

        from tensordict._td import TensorDict
        from tensordict.tensorclass import NonTensorData

        result = None
        if _is_namedtuple(pytree):
            # Dict, namedtuple and tuple fields go through from_pytree, so that
            # to_pytree can rebuild them. The other fields, the dicts that hold
            # a list or tuple, the tuples that hold a list, tuple or str, the
            # fields that hold an empty dict, namedtuple or tuple, and the
            # fields that from_pytree does not take (such as a dict holding a
            # str) go through from_any, as on main.
            nested = {}
            for key, value in pytree._asdict().items():
                if _pytree_field(value) or (
                    type(value) is tuple
                    and value
                    and all(
                        isinstance(elt, (torch.Tensor, int, float))
                        or _pytree_field(elt)
                        for elt in value
                    )
                ):
                    try:
                        nested[key] = cls.from_pytree(value)
                    except NotImplementedError:
                        pass
            result = TensorDict.from_namedtuple(named_tuple=pytree._replace(**nested))
            if batch_size is not None:
                # parsed as the TensorDict constructor does, so that ints work
                result.batch_size = TensorDict._parse_batch_size(None, batch_size)
            result["_pytree_type"] = type(pytree)
        elif isinstance(pytree, (list, tuple)):
            source = {str(i): cls.from_pytree(elt) for i, elt in enumerate(pytree)}
            source["_pytree_type"] = type(pytree)
            result = TensorDict(source, batch_size=batch_size)
        elif isinstance(pytree, dict):
            source = {}
            for key, item in pytree.items():
                if isinstance(key, NestedKey):
                    source[key] = cls.from_pytree(item)
                else:
                    subs_key = "<NON_NESTED>" + str(uuid.uuid1())
                    if not isinstance(key, torch.Tensor):
                        # The constructor would turn numbers into tensors
                        key = NonTensorData(key)
                    source[subs_key] = TensorDict(
                        {"value": cls.from_pytree(item), "key": key}
                    )
            source["_pytree_type"] = type(pytree)
            result = TensorDict(source, batch_size=batch_size)
        if result is not None:
            if auto_batch_size:
                result.auto_batch_size_(batch_dims)
            elif batch_dims is not None:
                # auto_batch_size applies after batch_size here: it does not raise
                _warn_batch_dims_ignored("from_pytree", result, batch_dims, None)
            return result
        if isinstance(pytree, (int, float, np.ndarray)):
            return torch.as_tensor(pytree)
        raise NotImplementedError(f"Unknown type {type(pytree)}.")

    def to_pytree(self):
        """Rebuilds the pytree that :meth:`~.from_pytree` converted to this tensordict.

        If the tensordict has no ``"_pytree_type"`` entry, that is, if it was
        not created by :meth:`~.from_pytree`, this method returns ``self``.
        A nested tensordict without that entry is kept as a tensordict in the
        pytree.

        .. seealso:: :meth:`~.from_pytree` for the supported types.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict.from_pytree({"a": [torch.zeros(2), torch.ones(2)]})
            >>> td.to_pytree()
            {'a': [tensor([0., 0.]), tensor([1., 1.])]}

        """
        _pytree_type = self._get_str("_pytree_type", default=None)
        if _pytree_type is None:
            return self
        _pytree_type = _pytree_type.data
        items = {key: val for (key, val) in self.items() if key != "_pytree_type"}
        items = {
            key: val if not is_tensor_collection(val) else val.to_pytree()
            for key, val in items.items()
        }
        if _is_namedtuple_class(_pytree_type):
            # Build this level only: the nested levels were rebuilt above, and
            # to_namedtuple(dest_cls=...) would apply the class to them too.
            fields = {}
            for key, val in items.items():
                if _is_non_tensor(type(val)):
                    val = val.to_dict()
                elif is_tensor_collection(val):
                    # from_any built it (e.g. for a dict holding a str), not
                    # from_pytree: convert it as to_namedtuple did on main
                    val = _dict_to_namedtuple(
                        val.to_dict(retain_none=False), _pytree_type
                    )
                fields[key] = val
            return _pytree_type(**fields)
        if issubclass(_pytree_type, (list, tuple)):
            # a list, a tuple or a subclass, such as the torch.return_types
            # structseqs, which take a sequence
            return _pytree_type([items[str(i)] for i in range(len(items))])
        if issubclass(_pytree_type, dict):
            result = {}
            for key, val in items.items():
                if key.startswith("<NON_NESTED>"):
                    key, val = _pytree_dict_key(val.get("key")), val["value"]
                    # The entry holding key and value has no _pytree_type, so
                    # the conversion above left the value as it is
                    if is_tensor_collection(val):
                        val = val.to_pytree()
                result[key] = val
            if _pytree_type is not dict:
                # an OrderedDict, or a defaultdict, which from_pytree records
                # without its default_factory
                typed = _pytree_type()
                typed.update(result)
                result = typed
            return result
        raise NotImplementedError(f"unknown type {_pytree_type}")

    @classmethod
    def from_h5(
        cls,
        filename,
        *,
        mode: str = "r",
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        batch_size: torch.Size | None = None,
        non_tensor_as_stack: bool = False,
    ):
        """Opens an HDF5 file as a :class:`~tensordict.PersistentTensorDict`.

        The data stays in the file: entries are read when they are accessed.
        Requires ``h5py``. See :ref:`saving` and
        :class:`~tensordict.PersistentTensorDict`.

        Args:
            filename (str or path): the path to the HDF5 file.

        Keyword Args:
            mode (str, optional): how the file is opened, as in
                :class:`h5py.File`: ``"r"`` (read only), ``"r+"`` (read and
                write), and so on. Defaults to ``"r"``.
            auto_batch_size (bool, optional): if ``True``, sets the batch size
                to the largest leading shape that all the entries share.
                Exclusive with ``batch_size``. Defaults to ``False``.
            batch_dims (int, optional): with ``auto_batch_size``, the maximum
                number of batch dimensions. Ignored without
                ``auto_batch_size``, with a ``FutureWarning`` when it would
                change the result. Starting with TensorDict 0.17, the default
                will change to turning on ``auto_batch_size`` when
                ``batch_dims`` is given, as ``from_dict`` does. Defaults to
                ``None``.
            batch_size (torch.Size or list of int, optional): the batch size of
                the result: the leading dimensions that all its entries share.
                Exclusive with ``auto_batch_size``. Defaults to ``None``, which
                gives each nested tensordict the largest batch size that its
                entries share.
            non_tensor_as_stack (bool, optional): if ``True``, a non-tensor
                dataset that tensordict did not write, such as a string column
                with one string per row, reads as a
                :class:`~tensordict.NonTensorStack` of its rows when its
                leading dimensions equal the batch size. Otherwise it reads as
                one :class:`~tensordict.NonTensorData` that holds the whole
                dataset. Defaults to ``False``.

        Returns:
            A :class:`~tensordict.PersistentTensorDict` backed by the file.

        Raises:
            TypeError: if both ``auto_batch_size=True`` and ``batch_size``
                are given.

        Examples:
            >>> import os
            >>> import tempfile
            >>> import torch
            >>> from tensordict import TensorDict
            >>> data = TensorDict(key1=torch.zeros(3), key2=torch.ones(3), batch_size=[3])
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     path = os.path.join(tmpdir, "file.h5")
            ...     _ = data.to_h5(path)
            ...     td = TensorDict.from_h5(path)
            ...     print(td)
            PersistentTensorDict(batch_size=[3], fields={key1: f32[], key2: f32[]})
        """
        from tensordict.persistent import PersistentTensorDict

        if not issubclass(cls, PersistentTensorDict):
            cls = PersistentTensorDict
        result = cls._open_h5(
            filename,
            mode=mode,
            batch_size=batch_size,
            non_tensor_as_stack=non_tensor_as_stack,
        )
        if auto_batch_size:
            if batch_size is not None:
                raise TypeError(cls._CONFLICTING_BATCH_SIZES.format("from_h5"))
            result.auto_batch_size_(batch_dims=batch_dims)
        elif batch_dims is not None:
            _warn_batch_dims_ignored("from_h5", result, batch_dims, batch_size)
        return result

    @classmethod
    def from_zarr(
        cls,
        filename,
        *,
        mode: str = "r",
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        batch_size: torch.Size | None = None,
        non_tensor_as_stack: bool = False,
        allow_pickle: bool | None = None,
    ):
        """Opens a zarr store as a :class:`~tensordict.PersistentTensorDict`.

        The data stays in the store: entries are read when they are accessed.
        Requires ``zarr>=3.0``. See :ref:`saving` and
        :class:`~tensordict.PersistentTensorDict`.

        Args:
            filename (str, path or zarr store): the path to the zarr store (a
                directory), or a ``zarr.abc.store.Store`` instance (e.g. a
                ``zarr.storage.ZipStore``).

        Keyword Args:
            mode (str, optional): how the store is opened: ``"r"`` (read
                only), ``"r+"`` (read and write), and so on, as in
                :func:`zarr.open_group`. Defaults to ``"r"``.
            auto_batch_size (bool, optional): if ``True``, sets the batch size
                to the largest leading shape that all the entries share.
                Exclusive with ``batch_size``. Defaults to ``False``.
            batch_dims (int, optional): with ``auto_batch_size``, the maximum
                number of batch dimensions. Ignored without
                ``auto_batch_size``, with a ``FutureWarning`` when it would
                change the result. Starting with TensorDict 0.17, the default
                will change to turning on ``auto_batch_size`` when
                ``batch_dims`` is given, as ``from_dict`` does. Defaults to
                ``None``.
            batch_size (torch.Size or list of int, optional): the batch size of
                the result: the leading dimensions that all its entries share.
                Exclusive with ``auto_batch_size``. Defaults to ``None``, which
                reads the batch size and the names of the batch dimensions that :meth:`~.to_zarr` stored, or, for
                a store written by another tool, gives each nested tensordict
                the largest batch size that its entries share.
            non_tensor_as_stack (bool, optional): if ``True``, a non-tensor
                array that tensordict did not write, such as a string column
                with one string per row, reads as a
                :class:`~tensordict.NonTensorStack` of its rows when its
                leading dimensions equal the batch size. Otherwise it reads as
                one :class:`~tensordict.NonTensorData` that holds the whole
                array. Defaults to ``False``.
            allow_pickle (bool, optional): whether to decode pickled
                non-tensor values. ``False`` rejects them before decoding.
                ``True`` permits them: use it only for data you trust.
                ``None`` permits them with a ``FutureWarning`` when read.
                Starting with TensorDict 0.17, the default will change to
                ``allow_pickle=False``. Tensor and JSON entries are unaffected.
                Defaults to ``None``.

        Returns:
            A :class:`~tensordict.PersistentTensorDict` backed by the store.

        Raises:
            TypeError: if both ``auto_batch_size=True`` and ``batch_size``
                are given.

        Examples:
            >>> import os
            >>> import tempfile
            >>> import torch
            >>> from tensordict import TensorDict
            >>> data = TensorDict(key1=torch.zeros(3), key2=torch.ones(3), batch_size=[3])
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     path = os.path.join(tmpdir, "store.zarr")
            ...     _ = data.to_zarr(path)
            ...     td = TensorDict.from_zarr(path)
            ...     print(td)
            PersistentTensorDict(batch_size=[3], fields={key1: f32[], key2: f32[]})
        """
        from tensordict.persistent import PersistentTensorDict

        if not issubclass(cls, PersistentTensorDict):
            cls = PersistentTensorDict
        result = cls._open_zarr(
            filename,
            mode=mode,
            batch_size=batch_size,
            non_tensor_as_stack=non_tensor_as_stack,
            allow_pickle=allow_pickle,
        )
        if auto_batch_size:
            if batch_size is not None:
                raise TypeError(cls._CONFLICTING_BATCH_SIZES.format("from_zarr"))
            result.auto_batch_size_(batch_dims=batch_dims)
        elif batch_dims is not None:
            _warn_batch_dims_ignored("from_zarr", result, batch_dims, batch_size)
        return result

    @classmethod
    def from_schema(
        cls,
        schema: dict[str, tuple[list[int] | torch.Size, torch.dtype]],
        *,
        batch_size: Sequence[int] | torch.Size | None = None,
        storage: str | None = None,
        device=None,
        **kwargs,
    ) -> TensorDictBase:
        """Allocates a zero-filled tensordict from a schema of shapes and dtypes.

        Each entry of ``schema`` maps a key to an ``(element_shape, dtype)``
        pair, and the entry gets the shape ``[*batch_size, *element_shape]``.
        ``storage`` selects where the zeros are stored, so that a buffer on
        disk or on a server can be created without building it in memory
        first.

        Args:
            schema (dict): maps each key (a ``str``) to a pair
                ``(element_shape, dtype)``, where ``element_shape`` (a list of
                int or a ``torch.Size``) is the shape of the entry after the
                batch dimensions.

        Keyword Args:
            batch_size (torch.Size or list of int, optional): the batch size of
                the result, prepended to every element shape. Defaults to
                ``None``, an empty batch size.
            storage (str, optional): where the entries are stored:

                - ``None``: a :class:`~tensordict.TensorDict` of regular
                  tensors.
                - ``"memmap"``: memory-mapped tensors, in the directory given
                  by ``prefix=<dir>`` (see :meth:`~.memmap_like`).
                - ``"h5"``: a :class:`~tensordict.PersistentTensorDict` in the
                  HDF5 file given by ``filename=<path>``.
                - ``"zarr"``: a :class:`~tensordict.PersistentTensorDict` in
                  the zarr store given by ``filename=<path or store>``
                  (requires ``zarr>=3.0``).
                - ``"shared"``: tensors in shared memory (see
                  :meth:`~.share_memory_`).
                - ``"redis"`` or ``"dragonfly"``: a
                  :class:`~tensordict.store.TensorDictStore` on a server,
                  through :meth:`TensorDictStore.from_schema
                  <tensordict.store.TensorDictStore.from_schema>`.

                Defaults to ``None``.
            device (torch.device, str or int, optional): the device of the
                result. The ``"memmap"`` storage ignores it. Defaults to
                ``None``.
            **kwargs: passed to the backend: ``prefix`` and the other
                arguments of :meth:`~.memmap_like` for ``"memmap"``,
                ``filename`` for ``"h5"`` and ``"zarr"``, ``host``, ``port``
                and the like for ``"redis"`` and ``"dragonfly"``.

        Returns:
            A new tensordict of the type that ``storage`` selects, filled with
            zeros.

        Raises:
            ValueError: if ``storage`` is not one of the values above.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict.from_schema(
            ...     {"obs": ([84, 84, 3], torch.uint8),
            ...      "reward": ([], torch.float32)},
            ...     batch_size=[1000],
            ... )
            >>> td["obs"].shape
            torch.Size([1000, 84, 84, 3])

            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as d:
            ...     td_mm = TensorDict.from_schema(
            ...         {"obs": ([4], torch.float32)},
            ...         batch_size=[8],
            ...         storage="memmap",
            ...         prefix=d,
            ...     )
            ...     print(td_mm.is_memmap())
            True

        """
        from tensordict._td import TensorDict

        if batch_size is None:
            batch_size = torch.Size(())
        else:
            batch_size = torch.Size(batch_size)

        def _full_shape(elem_shape):
            return [*batch_size, *elem_shape]

        if storage is None:
            source = {
                key: torch.zeros(_full_shape(es), dtype=dt, device=device)
                for key, (es, dt) in schema.items()
            }
            return TensorDict(source, batch_size=batch_size, device=device)

        if storage == "memmap":
            source = {
                key: torch.zeros((), dtype=dt).expand(_full_shape(es))
                for key, (es, dt) in schema.items()
            }
            td = TensorDict(source, batch_size=batch_size)
            prefix = kwargs.pop("prefix", None)
            return td.memmap_like(prefix, **kwargs)

        if storage == "h5":
            from tensordict.persistent import PersistentTensorDict

            source = {
                key: torch.zeros((), dtype=dt).expand(_full_shape(es))
                for key, (es, dt) in schema.items()
            }
            filename = kwargs.pop("filename")
            return PersistentTensorDict.from_dict(
                source, filename, batch_size=batch_size, device=device, **kwargs
            )

        if storage == "zarr":
            from tensordict.persistent import PersistentTensorDict

            source = {
                key: torch.zeros((), dtype=dt).expand(_full_shape(es))
                for key, (es, dt) in schema.items()
            }
            filename = kwargs.pop("filename")
            return PersistentTensorDict.from_dict(
                source,
                filename,
                batch_size=batch_size,
                device=device,
                backend="zarr",
                **kwargs,
            )

        if storage == "shared":
            source = {
                key: torch.zeros(_full_shape(es), dtype=dt, device=device)
                for key, (es, dt) in schema.items()
            }
            return TensorDict(
                source, batch_size=batch_size, device=device
            ).share_memory_()

        if storage in ("redis", "dragonfly"):
            from tensordict.store._store import TensorDictStore

            return TensorDictStore.from_schema(
                schema,
                batch_size=batch_size,
                backend=storage,
                device=device,
                **kwargs,
            )

        raise ValueError(
            f"Unknown storage backend {storage!r}. Expected one of "
            f"None, 'memmap', 'h5', 'zarr', 'shared', 'redis', 'dragonfly'."
        )

    # Module interaction
    @classmethod
    def from_module(
        cls,
        module,
        as_module: bool = False,
        lock: bool = True,
        use_state_dict: bool = False,
    ):
        """Returns a tensordict of the parameters and buffers of a module.

        The leaves are the parameters and buffers of the module, not copies:
        an in-place change to a leaf changes the module. Use
        ``TensorDict.from_module(module).data.clone()`` to get independent
        tensors. Submodules become sub-tensordicts: ``module.layers[0].weight``
        is the entry ``("layers", "0", "weight")``. Write the parameters back
        with :meth:`~.to_module`.

        The result is a :class:`~tensordict.TensorDict` (or a
        :class:`~tensordict.nn.TensorDictParams` with ``as_module=True``),
        whatever class this method is called on.

        .. note::
            This method locks the result by default (``lock=True``), but its
            override :meth:`TensorDict.from_module
            <tensordict.TensorDict.from_module>` does not.

        Args:
            module (nn.Module): the module to get the parameters from.
            as_module (bool, optional): if ``True``, returns a
                :class:`~tensordict.nn.TensorDictParams`, which can be stored
                as a submodule of a :class:`torch.nn.Module`. Defaults to
                ``False``.
            lock (bool, optional): if ``True``, the result is locked (see
                :ref:`keys-locking`). Defaults to ``True``.
            use_state_dict (bool, optional): if ``True``, reads the module's
                ``state_dict()``, so that its state-dict hooks run, and
                unflattens it into the tree of the submodules. The leaves are
                then the tensors of the state dict, not the parameters.
                Defaults to ``False``.

        Examples:
            >>> from tensordict import TensorDict
            >>> from torch import nn
            >>> module = nn.TransformerDecoder(
            ...     decoder_layer=nn.TransformerDecoderLayer(nhead=4, d_model=4),
            ...     num_layers=1
            ... )
            >>> params = TensorDict.from_module(module)
            >>> print(params["layers", "0", "linear1"])
            TensorDict(batch_size=[], fields={bias: f32[2048], weight: f32[2048,4]})

        """
        from tensordict import TensorDict

        return TensorDict.from_module(
            module, as_module=as_module, lock=lock, use_state_dict=use_state_dict
        )

    @classmethod
    def from_modules(
        cls,
        *modules,
        as_module: bool = False,
        lock: bool = True,
        use_state_dict: bool = False,
        lazy_stack: bool = False,
        expand_identical: bool = False,
    ):
        """Stacks the parameters and buffers of several modules in one tensordict.

        The result has one batch dimension, with one element per module, and
        the structure that :meth:`~.from_module` gives. Use it to run an
        ensemble of modules that have the same structure with
        :func:`torch.vmap` and :meth:`~.to_module` (see the example).

        The default, dense result holds new parameters and buffers: the
        stacks of those of the modules, with the same ``requires_grad``. Pass
        these new parameters to the optimizer: gradients reach them, not the
        parameters of the modules. With ``lazy_stack=True``, the result is a
        lazy stack of the parameters of the modules, which receive the
        gradients.

        Args:
            *modules (nn.Module): the modules to get the parameters from. With
                a dense stack, they must have the same parameters and buffers,
                with the same shapes.

        Keyword Args:
            as_module (bool, optional): if ``True``, returns a
                :class:`~tensordict.nn.TensorDictParams`, which can be stored
                as a submodule of a :class:`torch.nn.Module`. Exclusive with
                ``lazy_stack``. Defaults to ``False``.
            lock (bool, optional): if ``True``, the result is locked (see
                :ref:`keys-locking`). Defaults to ``True``.
            use_state_dict (bool, optional): if ``True``, reads the
                ``state_dict()`` of each module, so that its state-dict hooks
                run. Defaults to ``False``.
            lazy_stack (bool, optional): if ``True``, returns a
                :class:`~tensordict.LazyStackedTensorDict` of the parameters of
                the modules; if ``False``, stacks them into new parameters.
                Exclusive with ``as_module``, and not compatible with lazy
                modules. Defaults to ``False``.

                .. warning::
                    A lazy stack stacks the parameters each time an entry is
                    read with :meth:`~.get`, which costs memory (``N`` times
                    the size of the parameters, more if a graph is built) and
                    time, and the optimizer holds ``N`` times more parameters.
                    Use ``lazy_stack=True`` only when you need to train the
                    parameters of the modules themselves.

            expand_identical (bool, optional): if ``True``, a parameter that is
                the same object in every module is expanded instead of
                stacked, so that the result shares its memory. Ignored when
                ``lazy_stack=True``. Defaults to ``False``.

        Raises:
            RuntimeError: if ``lazy_stack=True`` and ``as_module=True``, or if
                ``lazy_stack=True`` and a module has uninitialized (lazy)
                parameters.

        Examples:
            >>> import torch
            >>> from torch import nn
            >>> from tensordict import TensorDict
            >>> _ = torch.manual_seed(0)
            >>> modules = [nn.Linear(3, 4) for _ in range(2)]
            >>> params = TensorDict.from_modules(*modules)
            >>> params.batch_size, params["weight"].shape
            (torch.Size([2]), torch.Size([2, 4, 3]))

            Run the two modules on one input with :func:`torch.vmap`, through a
            module on the ``"meta"`` device that holds no data. The gradients
            reach the stacked parameters, not those of the modules:

            >>> empty_module = nn.Linear(3, 4, device="meta")
            >>> def exec_module(params, x):
            ...     with params.to_module(empty_module):
            ...         return empty_module(x)
            >>> x = torch.randn(3)
            >>> y = torch.vmap(exec_module, (0, None))(params, x)
            >>> y.shape
            torch.Size([2, 4])
            >>> y.sum().backward()
            >>> params["weight"].grad is not None, modules[0].weight.grad is None
            (True, True)

            With ``lazy_stack=True``, the gradients reach the parameters of
            the modules:

            >>> params = TensorDict.from_modules(*modules, lazy_stack=True)
            >>> type(params).__name__
            'LazyStackedTensorDict'
            >>> y = torch.vmap(exec_module, (0, None))(params, x)
            >>> y.sum().backward()
            >>> modules[0].weight.grad is not None
            True

        """
        param_list = [
            cls.from_module(module, use_state_dict=use_state_dict) for module in modules
        ]
        if lazy_stack:
            from tensordict._lazy import LazyStackedTensorDict

            for param in param_list:
                if any(
                    isinstance(tensor, UninitializedTensorMixin)
                    for tensor in param.values(True, True)
                ):
                    raise RuntimeError(
                        "lasy_stack=True is not compatible with lazy modules."
                    )
            params = LazyStackedTensorDict.lazy_stack(param_list)
        elif expand_identical:
            from tensordict._torch_func import _stack_uninit_params

            # Check the keys
            #  If not expand_identical, `stack` takes care of that check but
            #  here we use apply which will ignore keys that are in one TD but not another
            sets = [set(param.keys(True, True)) for param in param_list]
            for set_ in sets[1:]:
                if set_ != sets[0]:
                    raise ValueError(
                        f"All key sets must match. "
                        f"Got {set_.symmetric_difference(sets[0])} in one but not another."
                    )

            def maybe_stack(*params):
                param = params[0]
                if isinstance(param, UninitializedTensorMixin):
                    return _stack_uninit_params(params, 0)
                if len(set(params)) == 1:
                    return param.expand((len(params), *param.shape))
                result = torch.stack(params)
                if isinstance(param, nn.Parameter):
                    return nn.Parameter(result.detach(), param.requires_grad)
                return Buffer(result)

            params = param_list[0]._fast_apply(
                maybe_stack,
                *param_list[1:],
                batch_size=torch.Size([len(param_list), *param_list[0].batch_size]),
            )
        else:
            with set_lazy_legacy(False), torch.no_grad():
                params = torch.stack(param_list)

            # Make sure params are params, buffers are buffers
            def make_param(param, orig_param):
                if isinstance(param, UninitializedTensorMixin):
                    return param
                if isinstance(orig_param, nn.Parameter):
                    return nn.Parameter(param.detach(), orig_param.requires_grad)
                return Buffer(param)

            params = params._fast_apply(make_param, param_list[0], propagate_lock=True)
        if as_module:
            from tensordict.nn import TensorDictParams

            params = TensorDictParams(params, no_convert=True)
        if lock:
            params.lock_()
        return params

    @_as_context_manager()
    def to_module(
        self,
        module: nn.Module,
        *,
        inplace: bool | None = None,
        return_swap: bool = True,
        swap_dest=None,
        use_state_dict: bool = False,
        non_blocking: bool = False,
        preserve_module_state: bool = True,
        memo=None,  # deprecated
    ):
        """Writes the content of the tensordict into the parameters and buffers of a module, recursively.

        The keys of the tensordict must match the names of the module
        attributes, as :meth:`~.from_module` gives them. A nested key is a
        submodule: ``("layers", "0", "weight")`` is ``module.layers[0].weight``.

        ``to_module`` can also be used as a context manager, to populate a
        module with other parameters for the duration of a ``with`` block (see
        the second example). When the block exits, the original parameters are
        written back, also if the block raises an exception.

        Args:
            module (nn.Module): a module to write the parameters into.

        Keyword Args:
            inplace (bool, optional): if ``True``, copies the values into the
                existing parameters and buffers of the module, which keep
                their identity. Otherwise the leaves of the tensordict replace
                them. Defaults to ``None``, which behaves like ``False``.
                Cannot be passed with ``use_state_dict=True``.
            return_swap (bool, optional): if ``True``, returns the parameters
                and buffers that the module held before the call. Defaults to
                ``True``.
            swap_dest (TensorDictBase, optional): with ``return_swap``, a
                tensordict to write the old values into and to return.
                Defaults to ``None``, which returns a new tensordict.
            use_state_dict (bool, optional): if ``True``, loads the values
                through the state-dict API, so that the module's state-dict
                hooks run. Defaults to ``False``.
            non_blocking (bool, optional): if ``True``, copies between devices
                can run asynchronously with respect to the host. Defaults to
                ``False``.
            preserve_module_state (bool, optional): if ``True``, existing
                :class:`~torch.nn.Parameter` and buffer registrations are
                preserved when writing tensor leaves to ``module``: parameters
                remain parameters with their original ``requires_grad`` value,
                and buffers remain registered buffers. Leaves that are not
                parameters but require gradients, or that are batched by
                :func:`~torch.vmap`, are registered as they are, so that
                gradients still reach them. If ``False``, a leaf that is not a
                :class:`~torch.nn.Parameter` replaces the attribute as it is,
                which can deregister an existing parameter (the historical
                behavior). Defaults to ``True``.

                .. versionchanged:: 0.15
                    ``preserve_module_state=None``, deprecated in 0.14, is no
                    longer accepted.

            memo: no longer accepted: passing it raises a ``RuntimeError``.

        Returns:
            A tensordict of the values that the module held before the call
            (or ``swap_dest``) if ``return_swap`` is ``True``, ``None``
            otherwise.

        Examples:
            >>> import torch
            >>> from torch import nn
            >>> from tensordict import TensorDict
            >>> module = nn.Linear(3, 4)
            >>> params = TensorDict.from_module(module)
            >>> zeros = params.data.clone().zero_()
            >>> old_params = zeros.to_module(module)
            >>> bool((module.weight == 0).all())
            True
            >>> _ = old_params.to_module(module)  # restores the original values
            >>> bool((module.weight != 0).any())
            True

            Use the tensordict as a context manager to make a functional call.
            When the block exits, the original parameters are written back:

            >>> with zeros.to_module(module):
            ...     y = module(torch.ones(2, 3))  # runs with zeroed parameters
            >>> bool((y == 0).all())
            True
            >>> bool((module.weight != 0).any())
            True

            The original parameters are written back if the block raises an
            exception too:

            >>> try:
            ...     with zeros.to_module(module):
            ...         raise ValueError("stop")
            ... except ValueError:
            ...     pass
            >>> bool((module.weight != 0).any())
            True

        """
        if memo is not None:
            raise RuntimeError("memo cannot be passed to the public to_module anymore.")
        if preserve_module_state is None:
            raise TypeError(
                "to_module(..., preserve_module_state=None) is no longer "
                "supported. Pass True to preserve the parameter and buffer "
                "registrations of the module, or False for the historical "
                "replacement behavior."
            )
        hooks = torch.nn.modules.module._global_parameter_registration_hooks
        memo = {"hooks": tuple(hooks.values())}
        return self._to_module(
            module=module,
            inplace=inplace,
            return_swap=return_swap,
            swap_dest=swap_dest,
            memo=memo,
            use_state_dict=use_state_dict,
            non_blocking=non_blocking,
            preserve_module_state=preserve_module_state,
        )

    def _to_module(
        self,
        module: nn.Module,
        *,
        inplace: bool | None = None,
        return_swap: bool = True,
        swap_dest=None,
        memo=None,
        use_state_dict: bool = False,
        non_blocking: bool = False,
        preserve_module_state: bool = True,
        is_dynamo: bool | None = None,
    ):
        from tensordict.base import TensorDictBase

        if is_dynamo is None:
            is_dynamo = is_compiling()

        if not use_state_dict and isinstance(module, TensorDictBase):
            if return_swap:
                swap = module.copy()
                module._param_td = getattr(self, "_param_td", self)
                return swap
            else:
                module.update(self)
                return

        if (
            return_swap
            and not inplace
            and not use_state_dict
            and not is_dynamo
            and (swap_dest is None or isinstance(swap_dest, dict))
            and type(module).__setattr__ is __base__setattr__
        ):
            _swap = _swap_into_module(
                self, module, memo, non_blocking, preserve_module_state
            )
            if swap_dest is None:
                return self._new_unsafe(_swap, batch_size=torch.Size(()))
            return _swap

        hooks = memo["hooks"]
        if return_swap:
            _swap = {}
            if not is_dynamo:
                memo[weakref.ref(module)] = _swap

        if use_state_dict:
            if inplace is not None:
                raise RuntimeError(
                    "inplace argument cannot be passed when use_state_dict=True."
                )
            # execute module's pre-hooks
            state_dict = self.flatten_keys(".")
            prefix = ""
            strict = True
            local_metadata = {}
            missing_keys = []
            unexpected_keys = []
            error_msgs = []
            for hook in module._load_state_dict_pre_hooks.values():
                hook(
                    state_dict,
                    prefix,
                    local_metadata,
                    strict,
                    missing_keys,
                    unexpected_keys,
                    error_msgs,
                )

            def convert_type(x, y):
                if isinstance(y, nn.Parameter):
                    return nn.Parameter(x)
                if isinstance(y, Buffer):
                    return Buffer(x)
                return x

            input = state_dict.unflatten_keys(".")._fast_apply(
                convert_type, self, propagate_lock=True
            )
        else:
            input = self
            inplace = bool(inplace)

        # we use __dict__ directly to avoid the getattr/setattr overhead whenever we can
        if not is_dynamo and type(module).__setattr__ is __base__setattr__:
            # if type(module).__setattr__ is __base__setattr__:
            __dict__ = module.__dict__
            _parameters = __dict__["_parameters"]
            _buffers = __dict__["_buffers"]
        else:
            __dict__ = None

        for key, value in input.items():
            if key == _EXTRA_STATE_KEY_SUFFIX:
                local_out = _swap_extra_state(module, value, return_swap)
            elif isinstance(value, (Tensor, ftdim.Tensor)):
                # For Dynamo, we use regular set/delattr as we're not
                #  much afraid by overhead (and dynamo doesn't like those
                #  hacks we're doing).
                if __dict__ is not None:
                    # if setattr is the native nn.Module.setattr, we can rely on _set_tensor_dict
                    local_out = _set_tensor_dict(
                        __dict__,
                        _parameters,
                        _buffers,
                        hooks,
                        module,
                        key,
                        value,
                        inplace,
                        return_swap=return_swap,
                        preserve_module_state=preserve_module_state,
                    )
                else:
                    if not inplace:
                        value = _maybe_preserve_module_state(
                            module,
                            key,
                            value,
                            preserve_module_state=preserve_module_state,
                        )
                        local_out = swap_tensor(module, key, value)
                    else:
                        new_val = local_out
                        if return_swap:
                            local_out = local_out.clone()
                        new_val.data.copy_(value.data, non_blocking=non_blocking)
            else:
                if __dict__ is not None:
                    child = __dict__["_modules"][key]
                else:
                    child = module._modules.get(key)

                if not is_dynamo:
                    local_out = memo.get(weakref.ref(child), NO_DEFAULT)

                if is_dynamo or local_out is NO_DEFAULT:
                    local_out = value._to_module(
                        child,
                        inplace=inplace,
                        return_swap=return_swap,
                        swap_dest={},  # we'll be calling update later
                        memo=memo,
                        use_state_dict=use_state_dict,
                        non_blocking=non_blocking,
                        preserve_module_state=preserve_module_state,
                        is_dynamo=is_dynamo,
                    )
                else:
                    memo["shared"] = True

            if return_swap:
                _swap[key] = local_out

        if return_swap:
            if isinstance(swap_dest, dict):
                return _swap
            elif swap_dest is not None:
                for td, key, val in _replaced_entries(_swap, swap_dest, []):
                    td._set_str(
                        key,
                        val,
                        inplace=False,
                        validated=True,
                        non_blocking=non_blocking,
                    )
                return swap_dest
            else:
                return self._new_unsafe(_swap, batch_size=torch.Size(()))

    @classmethod
    def fromkeys(cls, keys: List[NestedKey], value: Any = 0):
        """Creates a tensordict with the same value for every key, like :meth:`dict.fromkeys`.

        The result is a :class:`~tensordict.TensorDict` with an empty batch
        size. A nested key creates nested tensordicts. A Python number becomes
        a tensor for each key, and a tensor is stored as it is: every key then
        holds the same tensor object.

        Args:
            keys (list of NestedKey): the keys of the result.
            value (Any, optional): the value of every key: a tensor, a number,
                or any value that the :class:`~tensordict.TensorDict`
                constructor takes. Defaults to ``0``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict.fromkeys(["a", ("b", "c")], torch.zeros(3))
            >>> td["b", "c"]
            tensor([0., 0., 0.])
            >>> td["a"] is td["b", "c"]
            True

        """
        from tensordict._td import TensorDict

        return TensorDict(dict.fromkeys(keys, value), batch_size=[])

    to_mds = to_mds

    def _convert_to_tensordict(
        self, dict_value: dict[str, Any], non_blocking: bool | None = None
    ) -> Self:
        TensorDict = _td_base._TENSORDICT_TYPE

        return TensorDict(
            dict_value,
            batch_size=self.batch_size,
            device=self.device,
            names=self._maybe_names(),
            lock=self.is_locked,
            non_blocking=non_blocking,
        )

    def to_tensordict(self, *, retain_none: bool | None = None) -> Self:
        """Returns a :class:`~tensordict.TensorDict` copy of this tensordict.

        Every tensor is cloned, and every nested tensordict and tensorclass is
        converted to a :class:`~tensordict.TensorDict`, recursively. Use it to
        get a regular, unlocked tensordict in memory from a lazy stack, a
        memory-mapped or locked tensordict, a
        :class:`~tensordict.PersistentTensorDict` or a tensorclass. The batch
        size, device and names of the batch dimensions are kept, with one
        exception: if the tensordicts of a lazy stack have no names, the
        result has no names either, even when the stack dimension has a name
        (from ``stack_dim_name`` or the ``names`` setter). Name the dimension
        again with :meth:`~.refine_names`, as in the example.

        Keyword Args:
            retain_none (bool, optional): if ``True``, the ``None`` fields of
                tensorclass instances are written as entries of the result.
                Defaults to ``None``, which behaves like ``False``: the
                ``None`` fields are dropped.

        Returns:
            A new :class:`~tensordict.TensorDict` with copies of the values.

        Examples:
            >>> import torch
            >>> from tensordict import LazyStackedTensorDict, TensorDict
            >>> stack = LazyStackedTensorDict(
            ...     TensorDict(a=torch.zeros(3)), TensorDict(a=torch.ones(3)), stack_dim=0)
            >>> td = stack.to_tensordict()
            >>> type(td).__name__, td.batch_size
            ('TensorDict', torch.Size([2]))
            >>> stack.names = ["s"]
            >>> stack.to_tensordict().names
            [None]
            >>> stack.to_tensordict().refine_names("s").names
            ['s']

        """
        return self._to_tensordict(retain_none=retain_none)

    def _to_tensordict(self, *, retain_none: bool | None) -> TensorDictBase:
        """Builds the dense copy that :meth:`to_tensordict` returns.

        ``retain_none`` is the argument of :meth:`to_tensordict`. The base
        implementation clones each tensor and converts each nested tensordict
        and tensorclass with its own :meth:`to_tensordict`. A store overrides
        it to read all the leaves in one round trip; a ``NonTensorStack``,
        which is already the dense form of its data, returns itself.
        """
        from tensordict import TensorDict

        return TensorDict(
            {
                key: (
                    value.clone()
                    if not _is_tensor_collection(type(value))
                    else (
                        value
                        if is_non_tensor(value)
                        else (
                            value.clone()
                            if _is_unbatched(value)
                            else value.to_tensordict(retain_none=retain_none)
                        )
                    )
                )
                for key, value in self.items(is_leaf=_is_leaf_nontensor)
            },
            device=self.device,
            batch_size=self.batch_size,
            names=self._maybe_names(),
        )

    def to_lazystack(self, dim: int = 0):
        """Returns a lazy stack of the slices of the tensordict along ``dim``.

        The tensordict is unbound along ``dim`` and the slices are stacked in
        a :class:`~tensordict.LazyStackedTensorDict` (see :ref:`stacking`).
        The slices are views: writing into a tensordict of the stack in place
        writes into the original data. The batch size does not change. A lazy
        stack along ``dim`` is returned as it is, and a lazy stack along
        another dimension is restacked, which changes its stack dimension.

        Args:
            dim (int, optional): the dimension to stack along. Defaults to
                ``0``.

        Returns:
            A lazy stack with ``stack_dim == dim``, or ``self`` when the
            tensordict is already one.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(2, 3), batch_size=(2, 3))
            >>> stack = td.to_lazystack()
            >>> type(stack).__name__, stack.batch_size, stack.stack_dim
            ('LazyStackedTensorDict', torch.Size([2, 3]), 0)
            >>> stack.to_lazystack(1).stack_dim
            1
            >>> stack.to_lazystack(0) is stack
            True

        """
        from tensordict import lazy_stack, LazyStackedTensorDict
        from tensordict.tensorclass import _is_tensorclass

        dim = _maybe_correct_neg_dim(dim, ndim=self.ndim, shape=None)
        if (isinstance(self, LazyStackedTensorDict) and self.stack_dim == dim) or (
            _is_tensorclass(type(self))
            and isinstance(self._tensordict, LazyStackedTensorDict)
            and self._tensordict.stack_dim == dim
        ):
            return self
        return lazy_stack(self.unbind(dim), dim=dim)

    def to_dict(
        self,
        *,
        retain_none: bool = True,
        convert_tensors: bool | Literal["numpy"] = False,
        tolist_first: bool = False,
    ) -> dict[str, Any]:
        """Returns a nested dictionary with the entries of the tensordict.

        Nested tensordicts and tensorclasses become nested dicts. The tensors
        are not copied unless ``convert_tensors`` is set. Non-tensor data is
        returned as its Python value.

        Keyword Args:
            retain_none (bool, optional): if ``True``, the ``None`` fields of
                tensorclasses are written in the dict as ``None``. If
                ``False``, they are dropped. Defaults to ``True``.
            convert_tensors (bool or str, optional): if ``True``, tensors are
                converted to (nested) Python lists with
                :meth:`torch.Tensor.tolist`. If ``"numpy"``, they are converted
                to NumPy arrays. If ``False``, they are kept as tensors.
                Defaults to ``False``.
            tolist_first (bool, optional): if ``True``, a nested tensordict is
                converted with :meth:`~.tolist`, so that a nested tensordict
                with batch dimensions becomes a list of dicts, one per element.
                Defaults to ``False``.

        Returns:
            A dict whose keys are those of the tensordict.

        Raises:
            RuntimeError: on a lazy stack whose tensordicts have different
                keys (see :ref:`stacking-variable-size`).

        .. seealso:: :meth:`~.tolist`, :meth:`~.numpy` and
            :meth:`~tensordict.TensorDict.from_dict`, which converts back.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>>
            >>> td = TensorDict(
            ...     a=torch.arange(6).view(2, 3),
            ...     b=TensorDict(c=torch.arange(4).reshape(2, 2), batch_size=(2, 2)),
            ...     batch_size=(2,)
            ... )
            >>> print(td.to_dict())
            {'a': tensor([[0, 1, 2],
                    [3, 4, 5]]), 'b': {'c': tensor([[0, 1],
                    [2, 3]])}}
            >>> print(td.to_dict(convert_tensors=True))
            {'a': [[0, 1, 2], [3, 4, 5]], 'b': {'c': [[0, 1], [2, 3]]}}

        """
        # With tolist_first, the members of a nested lazy stack are listed one
        # by one, so only exclusive keys at the top level would be lost.
        if self._has_exclusive_keys and (
            not tolist_first or _has_top_level_exclusive_keys(self)
        ):
            _raise_on_exclusive_keys("to_dict")
        result = {}
        for key, value in self.items():
            if _is_tensor_collection(type(value)):
                # NonTensorStack.data raises AttributeError when the stacked
                # values differ: such a stack is not None, so it is kept.
                if (
                    not retain_none
                    and _is_non_tensor(type(value))
                    and getattr(value, "data", NO_DEFAULT) is None
                ):
                    continue
                if tolist_first:
                    value = value.tolist(convert_tensors=convert_tensors)
                else:
                    value = value.to_dict(
                        retain_none=retain_none, convert_tensors=convert_tensors
                    )
            elif convert_tensors:
                if isinstance(value, torch.Tensor) and convert_tensors == "numpy":
                    value = value.numpy()
                elif hasattr(value, "tolist"):
                    value = value.tolist()
            result[key] = value
        return result

    def tolist(
        self,
        *,
        convert_nodes: bool = True,
        convert_tensors: bool | Literal["numpy"] = False,
        tolist_first: bool = False,
        as_linked_list: bool = False,
    ) -> List[Any]:
        """Returns the elements of the tensordict as nested lists, one level per batch dimension.

        The innermost elements are the tensordict indexed at every batch
        index, converted with :meth:`~.to_dict`: for batch size ``[2, 3]``,
        the result is a list of 2 lists of 3 dicts. A tensordict without
        batch dimensions returns one dict.

        Keyword Args:
            convert_nodes (bool, optional): if ``True``, the elements are
                converted to dicts. If ``False``, they are kept as
                tensordicts. Defaults to ``True``.
            convert_tensors (bool or str, optional): as in :meth:`~.to_dict`:
                if ``True``, tensors become Python lists, and if ``"numpy"``,
                NumPy arrays. Requires ``convert_nodes=True``. Defaults to
                ``False``.
            tolist_first (bool, optional): as in :meth:`~.to_dict`: if
                ``True``, a nested tensordict with batch dimensions becomes a
                list of dicts. Defaults to ``False``.
            as_linked_list (bool, optional): if ``True``, returns a
                :class:`~tensordict.utils.LinkedList`: assigning an element of
                the outer list (``result[0] = {...}``) writes it into the
                tensordict. Defaults to ``False``.

        Returns:
            A list (nested once per batch dimension), or a dict (or ``self``
            if ``convert_nodes=False``) when the tensordict has no batch
            dimensions.

        Raises:
            TypeError: if ``convert_tensors`` is set and ``convert_nodes=False``.

        .. seealso:: :meth:`~.to_dict`.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(
            ...     a=torch.arange(4).view(2, 2),
            ...     b=TensorDict(c=torch.arange(8).view(2, 2, 2), batch_size=(2, 2, 2)),
            ...     batch_size=(2,),
            ... )
            >>> td.tolist()
            [{'a': tensor([0, 1]), 'b': {'c': tensor([[0, 1],
                    [2, 3]])}}, {'a': tensor([2, 3]), 'b': {'c': tensor([[4, 5],
                    [6, 7]])}}]
            >>> td.tolist(convert_tensors=True)
            [{'a': [0, 1], 'b': {'c': [[0, 1], [2, 3]]}}, {'a': [2, 3], 'b': {'c': [[4, 5], [6, 7]]}}]
            >>> td.tolist(tolist_first=True, convert_tensors=True)[0]
            {'a': [0, 1], 'b': [[{'c': 0}, {'c': 1}], [{'c': 2}, {'c': 3}]]}
            >>> type(td.tolist(convert_nodes=False)[0]).__name__
            'TensorDict'

        """
        if convert_tensors and not convert_nodes:
            raise TypeError("convert_tensors requires convert_nodes to be set to True")
        if not self.batch_dims:
            if convert_nodes:
                return self.to_dict(
                    convert_tensors=convert_tensors, tolist_first=tolist_first
                )
            return self

        q = collections.deque()
        result = []
        q.append((self, result))
        while len(q):
            val, _result = q.popleft()
            vals = val.unbind(0)
            if val.ndim == 1:
                if convert_nodes:
                    vals = [
                        v.to_dict(
                            convert_tensors=convert_tensors, tolist_first=tolist_first
                        )
                        for v in vals
                    ]
                else:
                    vals = list(vals)
                _result.extend(vals)
            else:
                for local_val in vals:
                    local_res = []
                    _result.append(local_res)
                    q.append((local_val, local_res))
        if as_linked_list:
            return LinkedList(result, td=self)
        return result

    def numpy(self) -> np.ndarray | dict[str, Any]:
        """Returns a nested dict of NumPy arrays with the entries of the tensordict.

        Each tensor becomes a NumPy array with :meth:`torch.Tensor.numpy`,
        so the arrays share memory with the tensors, and a tensor that
        requires gradients or is not on the CPU raises a ``RuntimeError`` or
        ``TypeError``: call :meth:`~.detach` or :meth:`~.cpu` first. A nested
        tensor becomes a tuple of arrays. Non-tensor data is returned as its
        Python value. The ``None`` fields of tensorclasses are dropped.

        .. seealso:: :meth:`~.to_struct_array` to convert to a NumPy
            structured array, and :meth:`~.to_dict`.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> data = TensorDict({"a": {"b": torch.zeros(()), "c": "a string!"}})
            >>> data.numpy()
            {'a': {'b': array(0., dtype=float32), 'c': 'a string!'}}

        """
        as_dict = self.to_dict(retain_none=False)

        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                if x.is_nested:
                    return tuple(_x.numpy() for _x in x)
                return x.numpy()
            if hasattr(x, "numpy"):
                return x.numpy()
            return x

        return torch.utils._pytree.tree_map(to_numpy, as_dict)

    def to_namedtuple(self, dest_cls: type | None = None) -> Any:
        """Converts the tensordict to a named tuple, recursively.

        Each nested tensordict becomes a nested named tuple. Without
        ``dest_cls``, each level is an instance of a new named tuple class,
        ``GenericDict``, whose fields are the keys. The ``None`` fields of
        tensorclasses are dropped.

        On a tensorclass, the top level is a plain :class:`tuple`, also with
        ``dest_cls``: it holds the values of ``to_dict(retain_none=False)``,
        in the same order, without the field names. The nested tensordicts
        and tensorclasses in it are still named tuples. A plain tuple of
        tensors can be pickled and loaded with
        ``torch.load(weights_only=True)``, while a ``GenericDict`` cannot be
        pickled, since its class is made at each call. To keep the field
        names, call ``to_dict()``, or ``to_tensordict().to_namedtuple()``.

        Args:
            dest_cls (type, optional): the named tuple class of the result. A
                nested tensordict is converted to ``dest_cls`` too if
                ``dest_cls`` takes its keys, and to a ``GenericDict``
                otherwise. Defaults to ``None``.

        .. seealso:: :meth:`~tensordict.TensorDict.from_namedtuple`, which
            converts back.

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> data = TensorDict({
            ...     "a_tensor": torch.zeros((3)),
            ...     "nested": {"a_tensor": torch.zeros((3)), "a_string": "zero!"}}, [3])
            >>> data.to_namedtuple()
            GenericDict(a_tensor=tensor([0., 0., 0.]), nested=GenericDict(a_tensor=tensor([0., 0., 0.]), a_string='zero!'))

            On a tensorclass, the top level is a plain tuple:

            >>> from tensordict import tensorclass
            >>> @tensorclass
            ... class Data:
            ...     a_tensor: torch.Tensor
            ...     a_string: str
            >>> Data(a_tensor=torch.zeros(3), a_string="zero!", batch_size=[3]).to_namedtuple()
            (tensor([0., 0., 0.]), 'zero!')

        """
        return _dict_to_namedtuple(self.to_dict(retain_none=False), dest_cls, root=True)

    @classmethod
    def from_any(
        cls,
        obj,
        *,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
    ):
        """Converts a dict, tuple, dataclass, array or file to a tensordict, recursively.

        ``from_any`` picks the conversion from the type of ``obj``, and
        applies it to the values inside ``obj`` too. It accepts more types
        than the :class:`~tensordict.TensorDict` constructor, at the cost of
        some overhead and of opinionated choices about how each type maps to
        entries:

        - a dict or :class:`collections.UserDict`: :meth:`~.from_dict`;
        - a named tuple: :meth:`~.from_namedtuple`;
        - a tuple: :meth:`~.from_tuple`, with the keys ``"0"``, ``"1"``, ...
          This includes the results of ``torch.return_types`` (``max``,
          ``sort``, ``topk``, ...), which also emit a :class:`FutureWarning`.
          Starting with TensorDict 0.17, the default will change to their
          field names (``"values"``, ``"indices"``, ...), as
          :meth:`~.from_namedtuple` gives;
        - a dataclass instance: :meth:`~.from_dataclass` (the result is a
          tensordict, not a tensorclass). A dataclass type whose fields all
          have defaults gives a tensordict of the defaults too, with a
          :class:`FutureWarning`. Starting with TensorDict 0.17, the default
          will change to returning the type unchanged, as for any other
          class;
        - a NumPy structured array: :meth:`~.from_struct_array`;
        - a pandas DataFrame: :meth:`~.from_pandas`;
        - an open :class:`h5py.File`: a
          :class:`~tensordict.PersistentTensorDict` (see :meth:`~.from_h5`).

        A tensor or a tensor collection is returned as it is. A list becomes a
        tensor if :func:`torch.tensor` takes it, and a
        :class:`~tensordict.NonTensorStack` otherwise. Any other object is
        returned unchanged.

        Args:
            obj: the object to convert.

        Keyword Args:
            auto_batch_size (bool, optional): if ``True``, sets the batch size
                to the largest leading shape that all the entries share.
                Exclusive with ``batch_size``, except for a dict, where
                ``batch_size`` wins, and for an :class:`h5py.File`, where
                ``auto_batch_size`` wins. Defaults to ``False``.
            batch_dims (int, optional): with ``auto_batch_size``, the maximum
                number of batch dimensions. Ignored without
                ``auto_batch_size``, with a ``FutureWarning`` when it would
                change the result. Starting with TensorDict 0.17, the default
                will change to turning on ``auto_batch_size`` when
                ``batch_dims`` is given, as ``from_dict`` does. Defaults to
                ``None``.
            device (torch.device, str or int, optional): the device of the
                result. Leaves on another device are moved to it. ``None``
                leaves each leaf where it is, and the result has no device.
                Defaults to ``None``.
            batch_size (torch.Size or list of int, optional): the batch size of
                the result: the leading dimensions that all its entries share.
                Exclusive with ``auto_batch_size``, except for a dict, where
                ``batch_size`` wins and no batch size is computed, and for an
                :class:`h5py.File`, where ``auto_batch_size`` wins and
                ``batch_size`` is ignored. Defaults to ``None``.

        Returns:
            A tensordict (a :class:`~tensordict.PersistentTensorDict` for an
            HDF5 file), a tensor or a :class:`~tensordict.NonTensorStack` for
            a list, or ``obj`` itself for the other types.

        Examples:
            >>> import dataclasses
            >>> import torch
            >>> from tensordict import TensorDict
            >>> @dataclasses.dataclass
            ... class Data:
            ...     obs: torch.Tensor
            ...     extra: tuple
            >>> data = Data(obs=torch.zeros(3, 4), extra=(torch.ones(3), {"x": torch.ones(3)}))
            >>> td = TensorDict.from_any(data, auto_batch_size=True)
            >>> td.batch_size
            torch.Size([3])
            >>> sorted(td.keys(include_nested=True, leaves_only=True), key=str)
            [('extra', '0'), ('extra', '1', 'x'), 'obs']
            >>> TensorDict.from_any([1, 2, 3])
            tensor([1, 2, 3])

        """
        if type(obj) is Tensor or is_tensor_collection(obj):
            # Conversions from non-tensor data must be done manually
            # if is_non_tensor(obj):
            #     from tensordict.tensorclass import LazyStackedTensorDict
            #     if isinstance(obj, LazyStackedTensorDict):
            #         return obj
            #     return cls.from_any(obj.data, auto_batch_size=auto_batch_size)
            return obj
        if isinstance(obj, (dict, UserDict)):
            result = cls.from_dict(
                obj if isinstance(obj, dict) else dict(obj),
                auto_batch_size=auto_batch_size,
                batch_dims=batch_dims,
                device=device,
                batch_size=batch_size,
            )
            if batch_dims is not None and not auto_batch_size:
                _warn_batch_dims_ignored("from_any", result, batch_dims, batch_size)
            return result
        if (
            isinstance(obj, np.ndarray)
            and hasattr(obj.dtype, "names")
            and obj.dtype.names is not None
        ):
            return cls.from_struct_array(
                obj,
                auto_batch_size=auto_batch_size,
                batch_dims=batch_dims,
                device=device,
                batch_size=batch_size,
            )
        if isinstance(obj, tuple):
            if _is_namedtuple(obj):
                return cls.from_namedtuple(
                    obj,
                    auto_batch_size=auto_batch_size,
                    batch_dims=batch_dims,
                    device=device,
                    batch_size=batch_size,
                )
            if is_structseq_instance(obj):
                _warn_structseq_keys(obj)
            return cls.from_tuple(
                obj,
                auto_batch_size=auto_batch_size,
                batch_dims=batch_dims,
                device=device,
                batch_size=batch_size,
            )
        if isinstance(obj, list):
            if _is_list_tensor_compatible(obj)[0]:
                return torch.tensor(obj)
            else:
                from tensordict.tensorclass import NonTensorStack

                return NonTensorStack.from_list(obj)
        if is_dataclass(obj):
            dataclass_batch_dims = _int_batch_dims(batch_dims)
            if auto_batch_size and dataclass_batch_dims is not None:
                try:
                    return cls.from_dataclass(
                        obj,
                        auto_batch_size=auto_batch_size,
                        batch_dims=dataclass_batch_dims,
                        device=device,
                        batch_size=batch_size,
                    )
                except Exception:
                    # for backward compatibility, a batch_dims that cannot be
                    # applied, for example to a nested NonTensorStack, is ignored
                    pass
            result = cls.from_dataclass(
                obj,
                auto_batch_size=auto_batch_size,
                device=device,
                batch_size=batch_size,
            )
            if batch_dims is not None and not auto_batch_size:
                _warn_batch_dims_ignored("from_any", result, batch_dims, batch_size)
            return result
        if not is_compiling() and importlib.util.find_spec("pandas") is not None:
            import pandas as pd

            if isinstance(obj, pd.DataFrame):
                return cls.from_pandas(
                    obj,
                    auto_batch_size=auto_batch_size,
                    batch_dims=batch_dims,
                    device=device,
                    batch_size=batch_size,
                )
        if _has_h5:
            import h5py

            if isinstance(obj, h5py.File):
                from tensordict import TensorDict
                from tensordict.persistent import PersistentTensorDict

                if not auto_batch_size and batch_size is not None:
                    try:
                        result = PersistentTensorDict(
                            group=obj,
                            batch_size=TensorDict._parse_batch_size(None, batch_size),
                        )
                    except Exception:
                        # for backward compatibility, a batch size that is not a
                        # size or that the file does not take is ignored
                        pass
                    else:
                        if batch_dims is not None:
                            _warn_batch_dims_ignored(
                                "from_any", result, batch_dims, batch_size
                            )
                        return result
                obj = PersistentTensorDict(group=obj)
                if auto_batch_size:
                    obj.auto_batch_size_(batch_dims=_int_batch_dims(batch_dims))
                elif batch_dims is not None:
                    _warn_batch_dims_ignored("from_any", obj, batch_dims, batch_size)
                return obj
        return obj

    @classmethod
    def from_tuple(
        cls,
        obj,
        *,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
    ):
        """Converts a tuple to a tensordict with the keys ``"0"``, ``"1"``, ...

        Each element is converted with :meth:`~.from_any`, so a nested tuple
        or dict becomes a nested tensordict and a number becomes a tensor.

        Args:
            obj (tuple): the tuple to convert.

        Keyword Args:
            auto_batch_size (bool, optional): if ``True``, sets the batch size
                to the largest leading shape that all the entries share.
                Exclusive with ``batch_size``. Defaults to ``False``.
            batch_dims (int, optional): with ``auto_batch_size``, the maximum
                number of batch dimensions. Ignored without
                ``auto_batch_size``, with a ``FutureWarning`` when it would
                change the result. Starting with TensorDict 0.17, the default
                will change to turning on ``auto_batch_size`` when
                ``batch_dims`` is given, as ``from_dict`` does. Defaults to
                ``None``.
            device (torch.device, str or int, optional): the device of the
                result. Leaves on another device are moved to it. ``None``
                leaves each leaf where it is, and the result has no device.
                Defaults to ``None``.
            batch_size (torch.Size or list of int, optional): the batch size of
                the result: the leading dimensions that all its entries share.
                It applies to the nested tensordicts too. Exclusive with
                ``auto_batch_size``. Defaults to ``None``, an empty batch size.

        Returns:
            A :class:`~tensordict.TensorDict`.

        Raises:
            TypeError: if both ``auto_batch_size=True`` and ``batch_size``
                are given.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict.from_tuple((torch.zeros(3, 4), 1), auto_batch_size=True)
            >>> td["0"].shape, td["1"], td.batch_size
            (torch.Size([3, 4]), tensor(1), torch.Size([]))
            >>> td = TensorDict.from_tuple((torch.zeros(3, 4), torch.ones(3)), auto_batch_size=True)
            >>> td.batch_size
            torch.Size([3])

        """
        from tensordict import TensorDict

        result = TensorDict(
            {
                str(i): cls.from_any(item, batch_size=batch_size, device=device)
                for i, item in enumerate(obj)
            },
            batch_size=batch_size,
            device=device,
        )
        if auto_batch_size:
            if batch_size is not None:
                raise TypeError(cls._CONFLICTING_BATCH_SIZES.format("from_tuple"))
            result.auto_batch_size_(batch_dims=batch_dims)
        elif batch_dims is not None:
            _warn_batch_dims_ignored("from_tuple", result, batch_dims, batch_size)
        return result

    @classmethod
    def from_dataclass(
        cls,
        dataclass,
        *,
        dest_cls: Type | None = None,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        as_tensorclass: bool = False,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
    ):
        """Converts a dataclass instance to a tensordict, with one entry per field.

        Each field value is converted with :meth:`~.from_any`, so a nested
        dataclass or dict becomes a nested tensordict and a number becomes a
        tensor. With ``as_tensorclass=True``, the conversion is that of the
        function :func:`~tensordict.from_dataclass` instead, which returns a
        tensorclass.

        Args:
            dataclass: the dataclass instance to convert. With
                ``as_tensorclass=True``, it can also be a dataclass type.
                Without it, a dataclass type whose fields all have defaults
                gives a tensordict of the defaults, with a
                :class:`FutureWarning`. Starting with TensorDict 0.17, the
                default will change to raising a :class:`TypeError` for it.

        Keyword Args:
            dest_cls (type, optional): with ``as_tensorclass=True``, the
                tensorclass to convert to. Ignored otherwise. Defaults to
                ``None``, which creates a new tensorclass with the fields of
                the dataclass.
            auto_batch_size (bool, optional): if ``True``, sets the batch size
                to the largest leading shape that all the entries share.
                Exclusive with ``batch_size``. Defaults to ``False``.
            batch_dims (int, optional): with ``auto_batch_size``, the maximum
                number of batch dimensions. Ignored without
                ``auto_batch_size``, with a ``FutureWarning`` when it would
                change the result. Starting with TensorDict 0.17, the default
                will change to turning on ``auto_batch_size`` when
                ``batch_dims`` is given, as ``from_dict`` does. Defaults to
                ``None``.
            as_tensorclass (bool, optional): if ``True``, returns a
                tensorclass instance (or a tensorclass type, for a dataclass
                type) through :func:`~tensordict.from_dataclass`. Defaults to
                ``False``.
            device (torch.device, str or int, optional): the device of the
                result. Leaves on another device are moved to it. ``None``
                leaves each leaf where it is, and the result has no device.
                Defaults to ``None``.
            batch_size (torch.Size or list of int, optional): the batch size of
                the result: the leading dimensions that all its entries share.
                Exclusive with ``auto_batch_size``. Defaults to ``None``, an
                empty batch size.

        Returns:
            A :class:`~tensordict.TensorDict`, or a tensorclass instance or
            type if ``as_tensorclass=True``.

        Raises:
            TypeError: if ``dataclass`` is not a dataclass and
                ``as_tensorclass=False``, if it is a dataclass type with a
                field that cannot be read from the class (a field without a
                default, or a field of a tensorclass type) and
                ``as_tensorclass=False``, or if both ``auto_batch_size=True``
                and ``batch_size`` are given.

        Examples:
            >>> import dataclasses
            >>> import torch
            >>> from tensordict import TensorDict
            >>> @dataclasses.dataclass
            ... class Data:
            ...     obs: torch.Tensor
            ...     reward: torch.Tensor
            >>> data = Data(obs=torch.zeros(3, 4), reward=torch.ones(3))
            >>> td = TensorDict.from_dataclass(data, auto_batch_size=True)
            >>> type(td).__name__, td.batch_size
            ('TensorDict', torch.Size([3]))
            >>> tc = TensorDict.from_dataclass(data, as_tensorclass=True, batch_size=[3])
            >>> type(tc).__name__, tc.batch_size
            ('Data_tc', torch.Size([3]))

        """
        if as_tensorclass:
            from tensordict.tensorclass import from_dataclass

            return from_dataclass(
                dataclass,
                auto_batch_size=auto_batch_size,
                dest_cls=dest_cls,
                batch_dims=batch_dims,
                batch_size=batch_size,
                device=device,
            )
        from dataclasses import fields

        from tensordict import TensorDict
        from tensordict.tensorclass import _FieldGetter

        if not is_dataclass(dataclass):
            raise TypeError(
                f"Expected a dataclass input, got a {type(dataclass)} input instead."
            )
        # The field getters of a tensorclass are not defaults.
        if isinstance(dataclass, type) and not all(
            hasattr(dataclass, field.name)
            and not isinstance(getattr(dataclass, field.name), _FieldGetter)
            for field in fields(dataclass)
        ):
            raise TypeError(
                f"from_dataclass() got the dataclass type {dataclass.__name__}, "
                "whose fields cannot all be read from the class. Pass an "
                "instance, or as_tensorclass=True to convert the type."
            )
        if isinstance(dataclass, type):
            warnings.warn(
                f"The dataclass type {dataclass.__name__} was converted to a "
                "tensordict of its field defaults. Starting with TensorDict "
                "0.17, the default will change to raising a TypeError in "
                "from_dataclass, and to keeping the type as it is in from_any "
                "and from_dict, as for any other class. Pass an instance to "
                "convert the defaults.",
                FutureWarning,
                stacklevel=_user_stacklevel(),
            )
        source = {}
        for field in fields(dataclass):
            source[field.name] = cls.from_any(
                getattr(dataclass, field.name), device=device, batch_size=batch_size
            )
        result = TensorDict(source, device=device, batch_size=batch_size)
        if auto_batch_size:
            if batch_size is not None:
                raise TypeError(cls._CONFLICTING_BATCH_SIZES.format("from_dataclass"))
            result.auto_batch_size_(batch_dims=batch_dims)
        elif batch_dims is not None:
            _warn_batch_dims_ignored("from_dataclass", result, batch_dims, batch_size)
        return result

    @classmethod
    def from_namedtuple(
        cls,
        named_tuple,
        *,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
    ):
        """Converts a named tuple to a tensordict whose keys are its field names.

        Each field value is converted with :meth:`~.from_any`, so a nested
        named tuple or dict becomes a nested tensordict and a number becomes a
        tensor. The ``torch.return_types`` results of ``torch.max``,
        ``torch.sort`` and the like are accepted too. :meth:`~.to_namedtuple`
        converts back.

        Args:
            named_tuple (NamedTuple): the named tuple to convert.

        Keyword Args:
            auto_batch_size (bool, optional): if ``True``, sets the batch size
                to the largest leading shape that all the entries share.
                Exclusive with ``batch_size``. Defaults to ``False``.
            batch_dims (int, optional): with ``auto_batch_size``, the maximum
                number of batch dimensions. Ignored without
                ``auto_batch_size``, with a ``FutureWarning`` when it would
                change the result. Starting with TensorDict 0.17, the default
                will change to turning on ``auto_batch_size`` when
                ``batch_dims`` is given, as ``from_dict`` does. Defaults to
                ``None``.
            device (torch.device, str or int, optional): the device of the
                result. Leaves on another device are moved to it. ``None``
                leaves each leaf where it is, and the result has no device.
                Defaults to ``None``.
            batch_size (torch.Size or list of int, optional): the batch size of
                the result: the leading dimensions that all its entries share.
                It applies to the nested tensordicts too. Exclusive with
                ``auto_batch_size``. Defaults to ``None``, an empty batch size.

        Returns:
            A :class:`~tensordict.TensorDict`.

        Raises:
            TypeError: if both ``auto_batch_size=True`` and ``batch_size``
                are given.

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> data = TensorDict({
            ...     "a_tensor": torch.zeros((3)),
            ...     "nested": {"a_tensor": torch.zeros((3)), "a_string": "zero!"}}, [3])
            >>> nt = data.to_namedtuple()
            >>> nt
            GenericDict(a_tensor=tensor([0., 0., 0.]), nested=GenericDict(a_tensor=tensor([0., 0., 0.]), a_string='zero!'))
            >>> td = TensorDict.from_namedtuple(nt, auto_batch_size=True)
            >>> td.batch_size, td["nested", "a_string"]
            (torch.Size([3]), 'zero!')
            >>> sorted(TensorDict.from_namedtuple(torch.tensor([2, 0, 1]).max(0)).keys())
            ['indices', 'values']

        """
        from tensordict import TensorDict

        def namedtuple_to_dict(namedtuple_obj):
            # A namedtuple. Test for _asdict, not _fields: under the Dynamo of
            # torch 2.11, a structseq has _fields, but no _asdict.
            if isinstance(namedtuple_obj, tuple) and hasattr(namedtuple_obj, "_asdict"):
                namedtuple_obj = namedtuple_obj._asdict()
            elif is_structseq_instance(namedtuple_obj):
                # torch.return_types (the results of max, sort, topk, ...)
                # are structseqs: named fields, but no _fields or _asdict.
                namedtuple_obj = {
                    name: getattr(namedtuple_obj, name)
                    for name in type(namedtuple_obj).__match_args__
                }
            for key, value in namedtuple_obj.items():
                namedtuple_obj[key] = cls.from_any(
                    value, device=device, batch_size=batch_size
                )
            return dict(namedtuple_obj)

        result = TensorDict(
            namedtuple_to_dict(named_tuple), device=device, batch_size=batch_size
        )
        if auto_batch_size:
            if batch_size is not None:
                raise TypeError(cls._CONFLICTING_BATCH_SIZES.format("from_namedtuple"))
            result.auto_batch_size_(batch_dims=batch_dims)
        elif batch_dims is not None:
            _warn_batch_dims_ignored("from_namedtuple", result, batch_dims, batch_size)
        return result

    @classmethod
    def from_struct_array(
        cls,
        struct_array: np.ndarray,
        *,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
    ) -> Self:
        """Converts a NumPy structured array to a tensordict, with one entry per field.

        The batch size is the shape of the array, and each entry holds a
        field. The conversion is zero-copy: the tensors share memory with the
        array, so an in-place change to the array changes the tensordict, and
        the other way around. Two cases copy the data: a ``device`` other
        than the CPU, and a field whose item size does not divide the size of
        a record, such as the ``"i8"`` field of a packed array of ``"i8"`` and
        ``"f4"`` fields (12 bytes per record), which torch cannot view. String
        fields become non-tensor data. :meth:`~.to_struct_array` converts
        back.

        Args:
            struct_array (np.ndarray): the structured array to convert.

        Keyword Args:
            auto_batch_size (bool, optional): if ``True``, sets the batch size
                to the largest leading shape that all the entries share,
                which can add the dimensions of array fields. Exclusive with
                ``batch_size``. Defaults to ``False``.
            batch_dims (int, optional): with ``auto_batch_size``, the maximum
                number of batch dimensions. Ignored without
                ``auto_batch_size``, with a ``FutureWarning`` when it would
                change the result. Starting with TensorDict 0.17, the default
                will change to turning on ``auto_batch_size`` when
                ``batch_dims`` is given, as ``from_dict`` does. Defaults to
                ``None``.
            device (torch.device, str or int, optional): the device of the
                result. Leaves on another device are moved to it. ``None``
                leaves each leaf where it is, and the result has no device.
                Defaults to ``None``.
            batch_size (torch.Size or list of int, optional): the batch size of
                the result: the leading dimensions that all its entries share.
                Exclusive with ``auto_batch_size``. Defaults to ``None``,
                which uses the shape of the array.

        Returns:
            A tensordict of the class this method is called on
            (:class:`~tensordict.TensorDict` for ``TensorDictBase``).

        Raises:
            TypeError: if both ``auto_batch_size=True`` and ``batch_size``
                are given.

        Examples:
            >>> import numpy as np
            >>> from tensordict import TensorDict
            >>> x = np.array(
            ...     [("Rex", 9, 81.0), ("Fido", 3, 27.0)],
            ...     dtype=[("name", "U10"), ("age", "i4"), ("weight", "f4")],
            ... )
            >>> td = TensorDict.from_struct_array(x)
            >>> td.batch_size, td["age"]
            (torch.Size([2]), tensor([9, 3], dtype=torch.int32))
            >>> x["age"] += 1  # the tensordict shares memory with the array
            >>> td["age"]
            tensor([10,  4], dtype=torch.int32)
            >>> bool((td.to_struct_array() == x).all())
            True

        """
        from tensordict.base import TensorDictBase

        if cls is TensorDictBase:
            from tensordict._td import TensorDict

            cls = TensorDict
        td: Self = cls(
            {name: struct_array[name] for name in struct_array.dtype.names},
            batch_size=struct_array.shape if batch_size is None else batch_size,
            device=device,
        )
        if auto_batch_size:
            if batch_size is not None:
                raise TypeError(
                    cls._CONFLICTING_BATCH_SIZES.format("from_struct_array")
                )
            td.auto_batch_size_(batch_dims=batch_dims)
        elif batch_dims is not None:
            _warn_batch_dims_ignored("from_struct_array", td, batch_dims, batch_size)
        return td

    def to_struct_array(self) -> np.ndarray:
        """Converts the tensordict to a NumPy structured array, with one field per entry.

        The shape of the array is the batch size. A nested tensordict becomes
        a nested structured field, and the feature dimensions of a tensor
        become the shape of its field. String values become ``"U"`` fields of
        at least 10 characters. Unlike :meth:`~.from_struct_array`, this
        method copies the data: the array does not share memory with the
        tensordict.

        Returns:
            A ``np.ndarray`` with a structured dtype.

        .. seealso:: :meth:`~.from_struct_array`, which converts back, and
            :meth:`~.numpy` to convert to a dict of NumPy arrays.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({'a': torch.tensor([1, 2, 3]), 'b': torch.tensor([4.0, 5.0, 6.0])}, batch_size=[3])
            >>> arr = td.to_struct_array()
            >>> print(arr)
            [(1, 4.) (2, 5.) (3, 6.)]

        """
        from tensordict.utils import _TORCH_TO_NUMPY_DTYPE_DICT

        keys, vals = zip(*self.items())
        _vals = []
        for v in vals:
            if is_tensor_collection(v):
                if is_non_tensor(v):
                    from tensordict import NonTensorDataBase

                    _vals.append(
                        v.data if isinstance(v, NonTensorDataBase) else v.tolist()
                    )
                    continue
                _vals.append(v.to_struct_array())
                continue
            _vals.append(v)
        vals = _vals
        del _vals
        vals = tuple(v if not is_non_tensor(v) else v.data for v in vals)

        # Convert values to numpy arrays and handle string inputs
        processed_vals = []
        for v in vals:
            if isinstance(v, torch.Tensor):
                processed_vals.append(v.detach().cpu().numpy())
            elif isinstance(v, (list, tuple, str)):
                # Handle lists/tuples which may contain strings, or strings
                processed_vals.append(np.array(v))
            else:
                # Keep other types as-is (already numpy arrays, etc.)
                processed_vals.append(v)
        vals = processed_vals

        def _get_dtype(val):
            if isinstance(val, np.ndarray):
                if val.dtype.kind in ["U", "S"]:  # Unicode or byte strings
                    # Calculate appropriate string length
                    if val.size > 0:
                        max_len = max(len(str(item)) for item in val.flat)
                        return (
                            f"U{max(10, max_len)}"  # At least U10, but longer if needed
                        )
                    return "U10"
                elif val.ndim > self.ndim:
                    # For arrays with more dimensions than batch dims, we need to specify shape
                    extra_shape = val.shape[self.ndim :]
                    return (val.dtype, extra_shape)
                return val.dtype
            elif isinstance(val, torch.Tensor):
                return _TORCH_TO_NUMPY_DTYPE_DICT.get(val.dtype, val.dtype)
            else:
                return "U10"

        dtype = [(key, _get_dtype(val)) for key, val in zip(keys, vals)]

        if self.ndim:
            # For multi-dimensional tensordicts, we need to create structured arrays properly
            # Reshape each value to have batch dimensions first, then flatten the batch dimensions
            batch_shape = self.shape
            batch_size = int(np.prod(batch_shape))

            # Reshape and prepare data for structured array
            reshaped_vals = []
            for val in vals:
                if isinstance(val, np.ndarray):
                    # Ensure the array has the right batch shape
                    if val.shape[: self.ndim] == batch_shape:
                        # Flatten batch dimensions
                        new_shape = (batch_size,) + val.shape[self.ndim :]
                        reshaped_vals.append(val.reshape(new_shape))
                    else:
                        # If shapes don't match, try to broadcast
                        try:
                            reshaped_vals.append(
                                np.broadcast_to(
                                    val, batch_shape + val.shape[self.ndim :]
                                ).reshape((batch_size,) + val.shape[self.ndim :])
                            )
                        except ValueError:
                            reshaped_vals.append(val)
                else:
                    reshaped_vals.append(val)

            # Create structured array
            result = np.empty(batch_size, dtype=dtype)
            for key, val in zip(keys, reshaped_vals):
                if isinstance(val, np.ndarray) and val.shape[0] == batch_size:
                    result[key] = val
                else:
                    result[key] = val

            # Reshape back to original batch shape
            return result.reshape(batch_shape)

        # For scalar case, create structured array properly
        result = np.empty((), dtype=dtype)
        for key, val in zip(keys, vals):
            if isinstance(val, np.ndarray) and val.ndim == 0:
                result[key] = val.item()
            elif isinstance(val, (np.ndarray, torch.Tensor)) and val.size == 1:
                result[key] = val.item()
            else:
                result[key] = val
        return result

    @classmethod
    def from_pandas(
        cls,
        dataframe,
        *,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
        separator: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Converts a pandas DataFrame to a tensordict, with one entry per column.

        The batch size is ``[num_rows]``. Numeric and boolean columns become
        tensors, a string or object column becomes one
        :class:`~tensordict.NonTensorData` that holds the whole column, a
        categorical column becomes the tensor of its codes, and a datetime
        column becomes an ``int64`` tensor of nanoseconds.
        :meth:`~.to_pandas` converts back.

        .. note::
            A string or object column is one value for the whole batch, so
            ``td[i]["name"]`` gives the values of every row, not the value of
            row ``i``. To get one value per row, set the column again as a
            list: ``td["name"] = df["name"].tolist()`` stores a
            :class:`~tensordict.NonTensorStack`, whose ``td[i]["name"]`` is
            the value of row ``i``. It builds one object per row, about 20
            microseconds each, so it is much slower than reading the column
            for large frames.

        Args:
            dataframe (pd.DataFrame): the DataFrame to convert.

        Keyword Args:
            auto_batch_size (bool, optional): if ``True``, sets the batch size
                to the largest leading shape that all the entries share.
                Exclusive with ``batch_size``. Defaults to ``False``.
            batch_dims (int, optional): with ``auto_batch_size``, the maximum
                number of batch dimensions. Ignored without
                ``auto_batch_size``, with a ``FutureWarning`` when it would
                change the result. Starting with TensorDict 0.17, the default
                will change to turning on ``auto_batch_size`` when
                ``batch_dims`` is given, as ``from_dict`` does. Defaults to
                ``None``.
            device (torch.device, str or int, optional): the device of the
                result. Leaves on another device are moved to it. ``None``
                leaves each leaf where it is, and the result has no device.
                Defaults to ``None``.
            batch_size (torch.Size or list of int, optional): the batch size of
                the result: the leading dimensions that all its entries share.
                Exclusive with ``auto_batch_size``. Defaults to ``None``,
                which gives ``[num_rows]``.
            separator (str, optional): if given, splits the column names on it
                to build nested entries: with ``separator="."``, the column
                ``"obs.x"`` becomes the entry ``("obs", "x")``. Defaults to
                ``None``, which keeps the column names as keys.
            dtype (torch.dtype, optional): if given, casts the numeric and
                boolean columns to it. Defaults to ``None``, which keeps the
                dtype of each column.

        Returns:
            A tensordict of the class this method is called on
            (:class:`~tensordict.TensorDict` for ``TensorDictBase``).

        Examples:
            >>> from tensordict import TensorDict
            >>> import pandas as pd
            >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
            >>> td = TensorDict.from_pandas(df)
            >>> print(td)
            TensorDict(batch_size=[3], fields={a: i64[], b: f64[]})
        """
        from tensordict.base import TensorDictBase

        if cls is TensorDictBase:
            from tensordict._td import TensorDict

            cls = TensorDict

        result = _dataframe_to_tensordict(
            dataframe,
            cls=cls,
            device=device,
            batch_size=batch_size,
            separator=separator,
            dtype=dtype,
        )
        if auto_batch_size:
            if batch_size is not None:
                raise TypeError(cls._CONFLICTING_BATCH_SIZES.format("from_pandas"))
            result.auto_batch_size_(batch_dims=batch_dims)
        elif batch_dims is not None:
            _warn_batch_dims_ignored("from_pandas", result, batch_dims, batch_size)
        return result

    def to_pandas(self, *, separator: str | None = None):
        """Converts the tensordict to a pandas DataFrame, with one column per leaf.

        Each leaf must hold one value per row: a tensor of shape
        ``[num_rows]``, or non-tensor data. Tensors are copied to the CPU and
        converted to NumPy arrays. Non-tensor data becomes a column of Python
        values. Requires pandas.

        A nested tensorclass gives one column per leaf with a ``separator``,
        as a nested tensordict does. Without a ``separator``, where a nested
        tensordict raises, a nested tensorclass becomes one column of
        objects that holds the tensorclass of each row. It also stays one
        such column with a ``separator`` if it has no leaves, or a leaf of
        more than one dimension.

        Keyword Args:
            separator (str, optional): if given, joins the keys of nested
                entries with it to build the column names: with
                ``separator="."``, the entry ``("obs", "x")`` becomes the
                column ``"obs.x"``. Required when the tensordict has nested
                tensordicts. Defaults to ``None``.

        Returns:
            A ``pd.DataFrame`` with one row per element of the tensordict.

        Raises:
            ValueError: if the tensordict has a nested tensordict and no
                ``separator``, or if a tensor has more than one dimension.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.arange(3), "b": torch.zeros(3)}, [3])
            >>> df = td.to_pandas()
            >>> print(df)
               a    b
            0  0  0.0
            1  1  0.0
            2  2  0.0

            A nested tensorclass without a ``separator``:

            >>> from tensordict import tensorclass
            >>> @tensorclass
            ... class Point:
            ...     x: torch.Tensor
            ...     y: torch.Tensor
            >>> td = TensorDict(p=Point(x=torch.zeros(3), y=torch.ones(3), batch_size=[3]), batch_size=[3])
            >>> df = td.to_pandas()
            >>> df["p"].dtype, type(df["p"][0]).__name__, df["p"][0].y
            (dtype('O'), 'Point', tensor(1.))
            >>> list(td.to_pandas(separator=".").columns)
            ['p.x', 'p.y']
        """
        return _tensordict_to_dataframe(self, separator=separator)

    @classmethod
    def from_csv(
        cls,
        path,
        *,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
        separator: str | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,
    ) -> Self:
        """Reads a CSV file into a tensordict, with one entry per column.

        Requires pandas or pyarrow: the file is read with
        :func:`pandas.read_csv` when pandas is installed, and with
        ``pyarrow.csv.read_csv`` otherwise. The columns are converted as in
        :meth:`~.from_pandas`. :meth:`~.to_csv` writes such a file. Through
        pandas, a string column is one value for the whole batch (see the
        note in :meth:`~.from_pandas`); through pyarrow, it is a
        :class:`~tensordict.NonTensorStack` with one value per row.

        Args:
            path (str or Path): the path of the CSV file.

        Keyword Args:
            auto_batch_size (bool, optional): if ``True``, sets the batch size
                to the largest leading shape that all the entries share.
                Exclusive with ``batch_size``. Defaults to ``False``.
            batch_dims (int, optional): with ``auto_batch_size``, the maximum
                number of batch dimensions. Ignored without
                ``auto_batch_size``, with a ``FutureWarning`` when it would
                change the result. Starting with TensorDict 0.17, the default
                will change to turning on ``auto_batch_size`` when
                ``batch_dims`` is given, as ``from_dict`` does. Defaults to
                ``None``.
            device (torch.device, str or int, optional): the device of the
                result. Leaves on another device are moved to it. ``None``
                leaves each leaf where it is, and the result has no device.
                Defaults to ``None``.
            batch_size (torch.Size or list of int, optional): the batch size of
                the result: the leading dimensions that all its entries share.
                Exclusive with ``auto_batch_size``. Defaults to ``None``,
                which gives ``[num_rows]``.
            separator (str, optional): if given, splits the column names on it
                to build nested entries: with ``separator="."``, the column
                ``"obs.x"`` becomes the entry ``("obs", "x")``. Defaults to
                ``None``, which keeps the column names as keys.
            dtype (torch.dtype, optional): if given, casts the numeric and
                boolean columns to it. Defaults to ``None``, which keeps the
                dtype of each column.
            **kwargs: passed to the CSV reader.

        Returns:
            A tensordict of the class this method is called on
            (:class:`~tensordict.TensorDict` for ``TensorDictBase``).

        Examples:
            >>> import os
            >>> import tempfile
            >>> import torch
            >>> from tensordict import TensorDict
            >>> data = TensorDict(obs=torch.arange(3.0), reward=torch.ones(3), batch_size=[3])
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     path = os.path.join(tmpdir, "data.csv")
            ...     data.to_csv(path)
            ...     td = TensorDict.from_csv(path)
            ...     td32 = TensorDict.from_csv(path, dtype=torch.float32)
            >>> td.batch_size, td["obs"].dtype, td32["obs"].dtype
            (torch.Size([3]), torch.float64, torch.float32)
        """
        from tensordict.base import TensorDictBase

        if cls is TensorDictBase:
            from tensordict._td import TensorDict

            cls = TensorDict

        columns, num_rows = _read_csv(path, **kwargs)
        result = _columns_to_tensordict(
            columns,
            cls=cls,
            device=device,
            batch_size=batch_size,
            separator=separator,
            dtype=dtype,
            num_rows=num_rows,
        )
        if auto_batch_size:
            if batch_size is not None:
                raise TypeError(cls._CONFLICTING_BATCH_SIZES.format("from_csv"))
            result.auto_batch_size_(batch_dims=batch_dims)
        elif batch_dims is not None:
            _warn_batch_dims_ignored("from_csv", result, batch_dims, batch_size)
        return result

    def to_csv(self, path, *, separator: str | None = None, **kwargs):
        """Writes the tensordict to a CSV file, with one column per leaf.

        The tensordict is converted with :meth:`~.to_pandas`, so the same
        rules apply, and written with :meth:`pandas.DataFrame.to_csv`
        without the index. Requires pandas. :meth:`~.from_csv` reads the
        file back.

        Args:
            path (str or Path): the path of the CSV file.

        Keyword Args:
            separator (str, optional): if given, joins the keys of nested
                entries with it to build the column names: with
                ``separator="."``, the entry ``("obs", "x")`` becomes the
                column ``"obs.x"``. Required when the tensordict has nested
                tensordicts. Defaults to ``None``.
            **kwargs: passed to :meth:`pandas.DataFrame.to_csv`.

        Raises:
            ValueError: as :meth:`~.to_pandas`.

        Examples:
            >>> import os
            >>> import tempfile
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.arange(2), b=TensorDict(c=torch.zeros(2)), batch_size=[2])
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     path = os.path.join(tmpdir, "data.csv")
            ...     td.to_csv(path, separator=".")
            ...     with open(path) as f:
            ...         print(f.read())
            a,b.c
            0,0.0
            1,0.0
            <BLANKLINE>

        """
        _write_csv(self, path, separator=separator, **kwargs)

    @classmethod
    def from_parquet(
        cls,
        path,
        *,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
        separator: str | None = None,
        dtype: torch.dtype | None = None,
        columns: list[str] | None = None,
        **kwargs,
    ) -> Self:
        """Reads a Parquet file into a tensordict, with one entry per column.

        Requires pyarrow or pandas: the file is read with
        ``pyarrow.parquet.read_table`` when pyarrow is installed, and with
        :func:`pandas.read_parquet` otherwise. The columns are converted as in
        :meth:`~.from_pandas`. :meth:`~.to_parquet` writes such a file. Through
        pyarrow, a string column is a :class:`~tensordict.NonTensorStack`
        with one value per row; through pandas, it is one value for the whole
        batch (see the note in :meth:`~.from_pandas`).

        Args:
            path (str or Path): the path of the Parquet file.

        Keyword Args:
            auto_batch_size (bool, optional): if ``True``, sets the batch size
                to the largest leading shape that all the entries share.
                Exclusive with ``batch_size``. Defaults to ``False``.
            batch_dims (int, optional): with ``auto_batch_size``, the maximum
                number of batch dimensions. Ignored without
                ``auto_batch_size``, with a ``FutureWarning`` when it would
                change the result. Starting with TensorDict 0.17, the default
                will change to turning on ``auto_batch_size`` when
                ``batch_dims`` is given, as ``from_dict`` does. Defaults to
                ``None``.
            device (torch.device, str or int, optional): the device of the
                result. Leaves on another device are moved to it. ``None``
                leaves each leaf where it is, and the result has no device.
                Defaults to ``None``.
            batch_size (torch.Size or list of int, optional): the batch size of
                the result: the leading dimensions that all its entries share.
                Exclusive with ``auto_batch_size``. Defaults to ``None``,
                which gives ``[num_rows]``.
            separator (str, optional): if given, splits the column names on it
                to build nested entries: with ``separator="."``, the column
                ``"obs.x"`` becomes the entry ``("obs", "x")``. Defaults to
                ``None``, which keeps the column names as keys.
            dtype (torch.dtype, optional): if given, casts the numeric and
                boolean columns to it. Defaults to ``None``, which keeps the
                dtype of each column.
            columns (list of str, optional): the columns to read. Defaults to
                ``None``, which reads all the columns.
            **kwargs: passed to the Parquet reader.

        Returns:
            A tensordict of the class this method is called on
            (:class:`~tensordict.TensorDict` for ``TensorDictBase``).

        Examples:
            >>> import os
            >>> import tempfile
            >>> import torch
            >>> from tensordict import TensorDict
            >>> data = TensorDict(obs=torch.arange(3.0), reward=torch.ones(3), batch_size=[3])
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     path = os.path.join(tmpdir, "data.parquet")
            ...     data.to_parquet(path)
            ...     td = TensorDict.from_parquet(path)
            ...     td_obs = TensorDict.from_parquet(path, columns=["obs"])
            >>> sorted(td.keys()), sorted(td_obs.keys())
            (['obs', 'reward'], ['obs'])
        """
        from tensordict.base import TensorDictBase

        if cls is TensorDictBase:
            from tensordict._td import TensorDict

            cls = TensorDict

        col_dict, num_rows = _read_parquet(path, columns=columns, **kwargs)
        result = _columns_to_tensordict(
            col_dict,
            cls=cls,
            device=device,
            batch_size=batch_size,
            separator=separator,
            dtype=dtype,
            num_rows=num_rows,
        )
        if auto_batch_size:
            if batch_size is not None:
                raise TypeError(cls._CONFLICTING_BATCH_SIZES.format("from_parquet"))
            result.auto_batch_size_(batch_dims=batch_dims)
        elif batch_dims is not None:
            _warn_batch_dims_ignored("from_parquet", result, batch_dims, batch_size)
        return result

    def to_parquet(self, path, *, separator: str | None = None, **kwargs):
        """Writes the tensordict to a Parquet file, with one column per leaf.

        The columns are built as in :meth:`~.to_pandas`, so the same rules
        apply. Requires pyarrow or pandas. When pyarrow is installed, the
        file is written with ``pyarrow.parquet.write_table``, from a pandas
        DataFrame if pandas is installed and from the columns directly
        otherwise. With pandas alone, the file is written with
        :meth:`pandas.DataFrame.to_parquet`, which then needs fastparquet.
        :meth:`~.from_parquet` reads the file back.

        Args:
            path (str or Path): the path of the Parquet file.

        Keyword Args:
            separator (str, optional): if given, joins the keys of nested
                entries with it to build the column names: with
                ``separator="."``, the entry ``("obs", "x")`` becomes the
                column ``"obs.x"``. Required when the tensordict has nested
                tensordicts. Defaults to ``None``.
            **kwargs: passed to the Parquet writer.

        Raises:
            ValueError: as :meth:`~.to_pandas`.

        Examples:
            >>> import os
            >>> import tempfile
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.arange(2), b=TensorDict(c=torch.zeros(2)), batch_size=[2])
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     path = os.path.join(tmpdir, "data.parquet")
            ...     td.to_parquet(path, separator=".")
            ...     td2 = TensorDict.from_parquet(path, separator=".")
            >>> td2["b", "c"]
            tensor([0., 0.])

        """
        _write_parquet(self, path, separator=separator, **kwargs)

    @classmethod
    def from_json(
        cls,
        path,
        *,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
        separator: str | None = None,
        dtype: torch.dtype | None = None,
        lines: bool = False,
        **kwargs,
    ) -> Self:
        """Reads a JSON file of records into a tensordict, with one entry per field.

        The file holds an array of records (``[{"a": 1, "b": 2.0}, ...]``),
        or one record per line with ``lines=True`` (JSON Lines). Each field
        becomes an entry with one value per record. The file is read with
        :func:`pandas.read_json` when pandas is installed, and with the
        :mod:`json` module otherwise. Through pandas, a string field is one
        value for the whole batch (see the note in :meth:`~.from_pandas`);
        through the :mod:`json` module, it is a
        :class:`~tensordict.NonTensorStack` with one value per record. To
        convert a single JSON object, load
        it with :func:`json.load` and pass it to :meth:`~.from_dict`.
        :meth:`~.to_json` writes such a file.

        Args:
            path (str or Path): the path of the JSON file.

        Keyword Args:
            auto_batch_size (bool, optional): if ``True``, sets the batch size
                to the largest leading shape that all the entries share.
                Exclusive with ``batch_size``. Defaults to ``False``.
            batch_dims (int, optional): with ``auto_batch_size``, the maximum
                number of batch dimensions. Ignored without
                ``auto_batch_size``, with a ``FutureWarning`` when it would
                change the result. Starting with TensorDict 0.17, the default
                will change to turning on ``auto_batch_size`` when
                ``batch_dims`` is given, as ``from_dict`` does. Defaults to
                ``None``.
            device (torch.device, str or int, optional): the device of the
                result. Leaves on another device are moved to it. ``None``
                leaves each leaf where it is, and the result has no device.
                Defaults to ``None``.
            batch_size (torch.Size or list of int, optional): the batch size of
                the result: the leading dimensions that all its entries share.
                Exclusive with ``auto_batch_size``. Defaults to ``None``,
                which gives ``[num_rows]``.
            separator (str, optional): if given, splits the column names on it
                to build nested entries: with ``separator="."``, the column
                ``"obs.x"`` becomes the entry ``("obs", "x")``. Defaults to
                ``None``, which keeps the column names as keys.
            dtype (torch.dtype, optional): if given, casts the numeric and
                boolean columns to it. Defaults to ``None``, which keeps the
                dtype of each column.
            lines (bool, optional): if ``True``, reads the file as JSON Lines,
                one record per line. Defaults to ``False``.
            **kwargs: passed to :func:`pandas.read_json`. Ignored without
                pandas.

        Returns:
            A tensordict of the class this method is called on
            (:class:`~tensordict.TensorDict` for ``TensorDictBase``).

        Examples:
            >>> import os
            >>> import tempfile
            >>> import torch
            >>> from tensordict import TensorDict
            >>> data = TensorDict(obs=torch.arange(3.0), reward=torch.ones(3), batch_size=[3])
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     path = os.path.join(tmpdir, "data.json")
            ...     data.to_json(path)
            ...     td = TensorDict.from_json(path)
            ...     path_lines = os.path.join(tmpdir, "data.jsonl")
            ...     data.to_json(path_lines, lines=True)
            ...     td_lines = TensorDict.from_json(path_lines, lines=True)
            >>> td.batch_size, td_lines.batch_size
            (torch.Size([3]), torch.Size([3]))
        """
        from tensordict.base import TensorDictBase

        if cls is TensorDictBase:
            from tensordict._td import TensorDict

            cls = TensorDict

        columns, num_rows = _read_json(path, lines=lines, **kwargs)
        result = _columns_to_tensordict(
            columns,
            cls=cls,
            device=device,
            batch_size=batch_size,
            separator=separator,
            dtype=dtype,
            num_rows=num_rows,
        )
        if auto_batch_size:
            if batch_size is not None:
                raise TypeError(cls._CONFLICTING_BATCH_SIZES.format("from_json"))
            result.auto_batch_size_(batch_dims=batch_dims)
        elif batch_dims is not None:
            _warn_batch_dims_ignored("from_json", result, batch_dims, batch_size)
        return result

    def to_json(
        self,
        path,
        *,
        separator: str | None = None,
        lines: bool = False,
        **kwargs,
    ):
        """Writes the tensordict to a JSON file, as an array of records.

        Each element of the tensordict becomes one record, with one field per
        leaf, as in :meth:`~.to_pandas`. With pandas installed, the file is
        written with :meth:`pandas.DataFrame.to_json` (``orient="records"``),
        and otherwise with the :mod:`json` module. :meth:`~.from_json` reads
        the file back.

        Args:
            path (str or Path): the path of the JSON file.

        Keyword Args:
            separator (str, optional): if given, joins the keys of nested
                entries with it to build the column names: with
                ``separator="."``, the entry ``("obs", "x")`` becomes the
                column ``"obs.x"``. Required when the tensordict has nested
                tensordicts. Defaults to ``None``.
            lines (bool, optional): if ``True``, writes JSON Lines, one record
                per line. Defaults to ``False``.
            **kwargs: passed to :meth:`pandas.DataFrame.to_json`. Ignored
                without pandas.

        Examples:
            >>> import os
            >>> import tempfile
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.arange(2), b=TensorDict(c=torch.zeros(2)), batch_size=[2])
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     path = os.path.join(tmpdir, "data.json")
            ...     td.to_json(path, separator=".")
            ...     with open(path) as f:
            ...         print(f.read())
            [{"a":0,"b.c":0.0},{"a":1,"b.c":0.0}]

        """
        _write_json(self, path, separator=separator, lines=lines, **kwargs)

    def to_h5(
        self,
        filename,
        **kwargs,
    ) -> Any:
        """Writes the tensordict to a new HDF5 file and returns a tensordict backed by it.

        Each leaf becomes a dataset of the file, and each nested tensordict a
        group. An existing file at ``filename`` is overwritten. The batch size
        and the names of the batch dimensions are kept. Requires ``h5py``.
        :meth:`~.from_h5` opens the file again. See :ref:`saving`.

        Args:
            filename (str or path): the path of the HDF5 file.
            **kwargs: passed to :meth:`h5py.Group.create_dataset`, for example
                ``compression="gzip"``.

        Returns:
            A :class:`~tensordict.PersistentTensorDict` backed by the new file.

        Examples:
            >>> import os
            >>> import tempfile
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(1000), b={"c": torch.zeros(1000, 3)}, batch_size=[1000])
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     td_h5 = td.to_h5(os.path.join(tmpdir, "data.h5"), compression="gzip")
            ...     print(type(td_h5).__name__, td_h5.batch_size, td_h5["b", "c"].shape)
            ...     td_h5.close()
            PersistentTensorDict torch.Size([1000]) torch.Size([1000, 3])

        """
        from tensordict.persistent import PersistentTensorDict

        out = PersistentTensorDict.from_dict(
            self,
            filename=filename,
            **kwargs,
        )
        if self._has_names():
            out.names = self.names
        return out

    def to_zarr(
        self,
        filename,
        *,
        allow_pickle: bool | None = None,
        **kwargs,
    ) -> Any:
        """Writes the tensordict to a new zarr store and returns a tensordict backed by it.

        Each leaf becomes an array of the store, and each nested tensordict a
        group. An existing store at ``filename`` is overwritten. The batch
        size and the names of the batch dimensions are saved in the store
        attributes, so that :meth:`~.from_zarr` restores them. Requires
        ``zarr>=3.0``. See :ref:`saving`.

        Args:
            filename (str, path or zarr store): the path of the zarr store (a
                directory), or a ``zarr.abc.store.Store`` instance (e.g. a
                ``zarr.storage.ZipStore``).
            **kwargs: passed to :meth:`zarr.Group.create_array`. By default
                each tensor is stored as one uncompressed chunk. Pass
                ``chunks=...`` or ``compressors=...`` to change that (for
                example to read rows without loading the whole array, or to
                compress on disk). ``chunks`` applies to the leading
                dimensions of each leaf, and the trailing dimensions are left
                whole: ``chunks=(16,)`` chunks every leaf along its first
                dimension in blocks of 16.

        Keyword Args:
            allow_pickle (bool, optional): the read policy of the returned
                tensordict for pickled non-tensor values. ``False`` rejects
                them. ``None`` and ``True`` read them, since this call wrote
                them. Opening the store again with :meth:`~.from_zarr` uses
                the policy of that call. Defaults to ``None``.

        Returns:
            A :class:`~tensordict.PersistentTensorDict` backed by the new store.

        Examples:
            >>> import os
            >>> import tempfile
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(1000), b={"c": torch.zeros(1000, 3)}, batch_size=[1000])
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     td_zarr = td.to_zarr(os.path.join(tmpdir, "store.zarr"), chunks=(100,))
            ...     print(type(td_zarr).__name__, td_zarr.batch_size, td_zarr["b", "c"].shape)
            PersistentTensorDict torch.Size([1000]) torch.Size([1000, 3])

        """
        from tensordict.persistent import PersistentTensorDict

        return PersistentTensorDict._from_dict(
            self,
            filename=filename,
            backend="zarr",
            names=self.names if self._has_names() else None,
            allow_pickle=allow_pickle,
            **kwargs,
        )

    def to_store(
        self,
        *,
        backend: str = "redis",
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        unix_socket_path: str | None = None,
        prefix: str = "tensordict",
        device=None,
        **kwargs,
    ) -> Any:
        """Uploads the tensordict to a key-value server and returns a tensordict backed by it.

        The entries are written to a Redis or Dragonfly server, under the key
        namespace ``prefix``. The result reads and writes the server. A lazy
        stack is uploaded in chunks, without building the dense stack in
        memory, and gives a :class:`~tensordict.store.LazyStackedTensorDictStore`.
        Requires the ``redis`` Python package and a running server. See
        :ref:`storage`.

        Keyword Args:
            backend (str, optional): the server type: ``"redis"`` or
                ``"dragonfly"``. Defaults to ``"redis"``.
            host (str, optional): the server hostname. Defaults to
                ``"localhost"``.
            port (int, optional): the server port. Defaults to ``6379``.
            db (int, optional): the database number. Defaults to ``0``.
            unix_socket_path (str, optional): the path of a Unix domain socket
                to connect through instead of ``host`` and ``port``. Defaults
                to ``None``.
            prefix (str, optional): the key namespace of the data on the
                server. Defaults to ``"tensordict"``.
            device (torch.device, str or int, optional): the device of the
                tensors that the result reads. Defaults to ``None``, which
                uses the device of the tensordict.
            **kwargs: passed to the connection of the client.

        Returns:
            A :class:`~tensordict.store.TensorDictStore`, or a
            :class:`~tensordict.store.LazyStackedTensorDictStore` for a lazy
            stack.

        Examples:
            The example needs a Redis server on ``localhost:6379``, and the
            last line a Dragonfly server on the host ``dragonfly-host``, so it
            is not run when the documentation is tested.

            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"obs": torch.randn(10, 84)}, [10])
            >>> store_td = td.to_store(host="localhost")  # doctest: +SKIP
            >>> store_td["obs"].shape  # doctest: +SKIP
            torch.Size([10, 84])
            >>>
            >>> # Using Dragonfly instead of Redis
            >>> store_td = td.to_store(backend="dragonfly", host="dragonfly-host")  # doctest: +SKIP
        """
        from tensordict.store._store import TensorDictStore

        return TensorDictStore.from_tensordict(
            self,
            backend=backend,
            host=host,
            port=port,
            db=db,
            unix_socket_path=unix_socket_path,
            prefix=prefix,
            device=device,
            **kwargs,
        )

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import collections
import functools
import numbers
import os
import re
import textwrap
import warnings
from collections import defaultdict
from collections.abc import MutableMapping
from copy import copy, deepcopy
from numbers import Number
from pathlib import Path
from textwrap import indent
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Iterator,
    OrderedDict,
    Sequence,
    Union,
)
from warnings import warn

import numpy as np

import torch
from tensordict._tensordict import _unravel_key_to_tuple
from tensordict.memmap import memmap_tensor_as_tensor, MemmapTensor
from tensordict.utils import (
    _device,
    _dtype,
    _GENERIC_NESTED_ERR,
    _get_item,
    _getitem_batch_size,
    _is_shared,
    _is_tensorclass,
    _NON_STR_KEY_ERR,
    _NON_STR_KEY_TUPLE_ERR,
    _set_item,
    _shape,
    _StringOnlyDict,
    _sub_index,
    as_decorator,
    cache,
    convert_ellipsis_to_idx,
    DeviceType,
    erase_cache,
    expand_as_right,
    expand_right,
    IndexType,
    int_generator,
    is_tensorclass,
    lock_blocked,
    NestedKey,
    prod,
)
from torch import distributed as dist, Tensor
from torch.utils._pytree import tree_map

try:
    from torch.jit._shape_functions import infer_size_impl
except ImportError:
    from tensordict.utils import infer_size_impl

# from torch.utils._pytree import _register_pytree_node


_has_functorch = False
try:
    try:
        from functorch._C import is_batchedtensor
    except ImportError:
        from torch._C._functorch import (
            _add_batch_dim,
            _remove_batch_dim,
            is_batchedtensor,
        )

    _has_functorch = True
except ImportError:
    _has_functorch = False

    def is_batchedtensor(tensor: Tensor) -> bool:
        """Placeholder for the functorch function."""
        return False


try:
    from torchrec import KeyedJaggedTensor

    _has_torchrec = True
except ImportError as err:
    _has_torchrec = False

    class KeyedJaggedTensor:  # noqa: D103, D101
        pass

    TORCHREC_ERR = str(err)

NO_DEFAULT = "_no_default_"


class _BEST_ATTEMPT_INPLACE:
    def __bool__(self):
        raise NotImplementedError


BEST_ATTEMPT_INPLACE = _BEST_ATTEMPT_INPLACE()

# some complex string used as separator to concatenate and split keys in
# distributed frameworks
DIST_SEPARATOR = ".-|-."
TD_HANDLED_FUNCTIONS: dict[Callable, Callable] = {}
CompatibleType = Union[
    Tensor,
    MemmapTensor,
]  # None? # leaves space for TensorDictBase

if _has_torchrec:
    CompatibleType = Union[
        Tensor,
        MemmapTensor,
        KeyedJaggedTensor,
    ]
_STR_MIXED_INDEX_ERROR = "Received a mixed string-non string index. Only string-only or string-free indices are supported."

_HEURISTIC_EXCLUDED = (Tensor, tuple, list, set, dict, np.ndarray)

_TENSOR_COLLECTION_MEMO = {}


def _is_tensor_collection(datatype):
    out = _TENSOR_COLLECTION_MEMO.get(datatype, None)
    if out is None:
        if issubclass(datatype, TensorDictBase):
            out = True
        elif _is_tensorclass(datatype):
            out = True
        else:
            out = False
        _TENSOR_COLLECTION_MEMO[datatype] = out
    return out


def is_tensor_collection(datatype: type | Any) -> bool:
    """Checks if a data object or a type is a tensor container from the tensordict lib.

    Returns:
        ``True`` if the input is a TensorDictBase subclass, a tensorclass or an istance of these.
        ``False`` otherwise.

    Examples:
        >>> is_tensor_collection(TensorDictBase)  # True
        >>> is_tensor_collection(TensorDict({}, []))  # True
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


def is_memmap(datatype: type | Any) -> bool:
    """Returns ``True`` if the class is a subclass of :class:`~.MemmapTensor` or the object an instance of it."""
    return (
        issubclass(datatype, MemmapTensor)
        if isinstance(datatype, type)
        else isinstance(datatype, MemmapTensor)
    )


class _TensorDictKeysView:
    """A Key view for TensorDictBase instance.

    _TensorDictKeysView is returned when accessing tensordict.keys() and holds a
    reference to the original TensorDict. This class enables us to support nested keys
    when performing membership checks and when iterating over keys.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict

        >>> td = TensorDict(
        >>>     {"a": TensorDict({"b": torch.rand(1, 2)}, [1, 2]), "c": torch.rand(1)},
        >>>     [1],
        >>> )

        >>> assert "a" in td.keys()
        >>> assert ("a",) in td.keys()
        >>> assert ("a", "b") in td.keys()
        >>> assert ("a", "c") not in td.keys()

        >>> assert set(td.keys()) == {("a", "b"), "c"}
    """

    def __init__(
        self,
        tensordict: TensorDictBase,
        include_nested: bool,
        leaves_only: bool,
    ) -> None:
        self.tensordict = tensordict
        self.include_nested = include_nested
        self.leaves_only = leaves_only

    def __iter__(self) -> Iterable[str] | Iterable[tuple[str, ...]]:
        if not self.include_nested:
            if self.leaves_only:
                for key in self._keys():
                    target_class = self.tensordict.entry_class(key)
                    if _is_tensor_collection(target_class):
                        continue
                    yield key
            else:
                yield from self._keys()
        else:
            yield from (
                key if len(key) > 1 else key[0]
                for key in self._iter_helper(self.tensordict)
            )

    def _iter_helper(
        self, tensordict: TensorDictBase, prefix: str | None = None
    ) -> Iterable[str] | Iterable[tuple[str, ...]]:
        for key, value in self._items(tensordict):
            full_key = self._combine_keys(prefix, key)
            cls = value.__class__
            if self.include_nested and (
                _is_tensor_collection(cls) or issubclass(cls, KeyedJaggedTensor)
            ):
                subkeys = tuple(self._iter_helper(value, prefix=full_key))
                yield from subkeys
            if not self.leaves_only or not _is_tensor_collection(cls):
                yield full_key

    def _combine_keys(self, prefix: tuple | None, key: str) -> tuple:
        if prefix is not None:
            return prefix + (key,)
        return (key,)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def _items(
        self, tensordict: TensorDict | None = None
    ) -> Iterable[tuple[NestedKey, CompatibleType]]:
        if tensordict is None:
            tensordict = self.tensordict
        if isinstance(tensordict, TensorDict) or is_tensorclass(tensordict):
            return tensordict._tensordict.items()
        elif isinstance(tensordict, LazyStackedTensorDict):
            return _iter_items_lazystack(tensordict)
        elif isinstance(tensordict, KeyedJaggedTensor):
            return tuple((key, tensordict[key]) for key in tensordict.keys())
        elif isinstance(tensordict, _CustomOpTensorDict):
            # it's possible that a TensorDict contains a nested LazyStackedTensorDict,
            # or _CustomOpTensorDict, so as we iterate through the contents we need to
            # be careful to not rely on tensordict._tensordict existing.
            return (
                (key, tensordict._get_str(key, NO_DEFAULT))
                for key in tensordict._source.keys()
            )

    def _keys(self) -> _TensorDictKeysView:
        return self.tensordict._tensordict.keys()

    def __contains__(self, key: NestedKey) -> bool:
        key = _unravel_key_to_tuple(key)
        if not key:
            raise TypeError(_NON_STR_KEY_ERR)

        if isinstance(key, str):
            if key in self._keys():
                if self.leaves_only:
                    return not _is_tensor_collection(self.tensordict.entry_class(key))
                return True
            return False
        else:
            # thanks to _unravel_key_to_tuple we know the key is a tuple
            if len(key) == 1:
                return key[0] in self._keys()
            elif self.include_nested:
                if key[0] in self._keys():
                    entry_type = self.tensordict.entry_class(key[0])
                    if entry_type in (Tensor, MemmapTensor):
                        return False
                    if entry_type is KeyedJaggedTensor:
                        if len(key) > 2:
                            return False
                        return key[1] in self.tensordict.get(key[0]).keys()
                    _is_tensordict = _is_tensor_collection(entry_type)
                    if _is_tensordict:
                        # # this will call _unravel_key_to_tuple many times
                        # return key[1:] in self.tensordict._get_str(key[0], NO_DEFAULT).keys(include_nested=self.include_nested)
                        # this won't call _unravel_key_to_tuple but requires to get the default which can be suboptimal
                        leaf_td = self.tensordict._get_tuple(key[:-1], None)
                        if leaf_td is None or (
                            not _is_tensor_collection(leaf_td.__class__)
                            and not isinstance(leaf_td, KeyedJaggedTensor)
                        ):
                            return False
                        return key[-1] in leaf_td.keys()
                return False
            # this is reached whenever there is more than one key but include_nested is False
            if all(isinstance(subkey, str) for subkey in key):
                raise TypeError(_NON_STR_KEY_TUPLE_ERR)

    def __repr__(self):
        include_nested = f"include_nested={self.include_nested}"
        leaves_only = f"leaves_only={self.leaves_only}"
        return f"{self.__class__.__name__}({list(self)},\n{indent(include_nested, 4*' ')},\n{indent(leaves_only, 4*' ')})"


def _renamed_inplace_method(fn):
    def wrapper(*args, **kwargs):
        warn(
            f"{fn.__name__.rstrip('_')} has been deprecated, use {fn.__name__} instead"
        )
        return fn(*args, **kwargs)

    return wrapper


class TensorDictBase(MutableMapping):
    """TensorDictBase is an abstract parent class for TensorDicts, a torch.Tensor data container."""

    LOCK_ERROR = (
        "Cannot modify locked TensorDict. For in-place modification, consider "
        "using the `set_()` method and make sure the key is present."
    )
    KEY_ERROR = 'key "{}" not found in {} with ' "keys {}"

    def __new__(cls, *args: Any, **kwargs: Any) -> TensorDictBase:
        self = super().__new__(cls)
        self._safe = kwargs.get("_safe", False)
        self._lazy = kwargs.get("_lazy", False)
        self._inplace_set = kwargs.get("_inplace_set", False)
        self.is_meta = kwargs.get("is_meta", False)
        self._is_locked = kwargs.get("_is_locked", False)
        self._cache = None
        self._last_op = None
        self.__last_op_queue = None
        return self

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict[str, Any]) -> dict[str, Any]:
        self.__dict__.update(state)

    @staticmethod
    def from_module(module):
        """Copies the params and buffers of a module in a tensordict.

        Examples:
            >>> from torch import nn
            >>> module = nn.TransformerDecoder(
            ...     decoder_layer=nn.TransformerDecoderLayer(nhead=4, d_model=4),
            ...     num_layers=1)
            >>> params = TensorDict.from_module(module)
            >>> print(params["layers", "0", "linear1"])
            TensorDict(
                fields={
                    bias: Parameter(shape=torch.Size([2048]), device=cpu, dtype=torch.float32, is_shared=False),
                    weight: Parameter(shape=torch.Size([2048, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
        """
        td = TensorDict(dict(module.named_parameters()), [])
        td.update(dict(module.named_buffers()))
        td = td.unflatten_keys(".")
        td.lock_()
        return td

    @property
    def shape(self) -> torch.Size:
        """See :obj:`TensorDictBase.batch_size`."""
        return self.batch_size

    @property
    @abc.abstractmethod
    def batch_size(self) -> torch.Size:
        """Shape of (or batch_size) of a TensorDict.

        The shape of a tensordict corresponds to the common N first
        dimensions of the tensors it contains, where N is an arbitrary
        number. The TensorDict shape is controlled by the user upon
        initialization (i.e. it is not inferred from the tensor shapes) and
        it should not be changed dynamically.

        Returns:
            a torch.Size object describing the TensorDict batch size.

        """
        raise NotImplementedError

    def _erase_cache(self):
        self._cache = None

    @property
    @abc.abstractmethod
    def names(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _erase_names(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _rename_subtds(self, value):
        # renames all the sub-tensordicts dimension according to value.
        # If value has less dimensions than the TD, the rest is just assumed to be None
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

    def refine_names(self, *names):
        """Refines the dimension names of self according to names.

        Refining is a special case of renaming that “lifts” unnamed dimensions.
        A None dim can be refined to have any name; a named dim can only be
        refined to have the same name.

        Because named tensors can coexist with unnamed tensors, refining names
        gives a nice way to write named-tensor-aware code that works with both
        named and unnamed tensors.

        names may contain up to one Ellipsis (...). The Ellipsis is expanded
        greedily; it is expanded in-place to fill names to the same length as
        self.dim() using names from the corresponding indices of self.names.

        Returns: the tensordict with dimensions named accordingly.

        """
        # replace ellipsis if any
        names_copy = copy(names)
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
        self.names = names
        # we also need to rename the sub-tensordicts
        # self._rename_subtds(self.names)
        return self

    def rename(self, *names, **rename_map):
        clone = self.clone(recurse=False)
        if len(names) == 1 and names[0] is None:
            clone.names = None
        if rename_map and names:
            raise ValueError(
                "Passed both a name map and a name list. Only one is accepted."
            )
        elif not rename_map and not names:
            raise ValueError(
                "Neither a name map nor a name list was passed. "
                "Only one is accepted."
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
        if len(names) == 1 and names[0] is None:
            self.names = None
        if rename_map and names:
            raise ValueError(
                "Passed both a name map and a name list. " "Only one is accepted."
            )
        elif not rename_map and not names and self.batch_dims:
            raise ValueError(
                "Neither a name map nor a name list was passed. "
                "Only one is accepted."
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
            self.names = cnames
        else:
            self.names = names
        return self

    @abc.abstractmethod
    def _has_names(self):
        raise NotImplementedError

    @property
    def _last_op_queue(self):
        last_op_queue = self.__last_op_queue
        if last_op_queue is None:
            last_op_queue = self.__last_op_queue = collections.deque()
        return last_op_queue

    def size(self, dim: int | None = None) -> torch.Size | int:
        """Returns the size of the dimension indicated by :obj:`dim`.

        If dim is not specified, returns the batch_size (or shape) of the TensorDict.

        """
        if dim is None:
            return self.batch_size
        return self.batch_size[dim]

    @property
    def requires_grad(self) -> bool:
        return any(v.requires_grad for v in self.values())

    def _batch_size_setter(self, new_batch_size: torch.Size) -> None:
        if new_batch_size == self.batch_size:
            return
        if self._lazy:
            raise RuntimeError(
                "modifying the batch size of a lazy repesentation of a "
                "tensordict is not permitted. Consider instantiating the "
                "tensordict first by calling `td = td.to_tensordict()` before "
                "resetting the batch size."
            )
        if not isinstance(new_batch_size, torch.Size):
            new_batch_size = torch.Size(new_batch_size)
        for key in self.keys():
            if _is_tensor_collection(self.entry_class(key)):
                tensordict = self.get(key)
                if len(tensordict.batch_size) < len(new_batch_size):
                    # document as edge case
                    tensordict.batch_size = new_batch_size
                    self._set_str(key, tensordict, inplace=True, validated=True)
        self._check_new_batch_size(new_batch_size)
        self._change_batch_size(new_batch_size)
        if self._has_names():
            names = self.names
            if len(names) < len(new_batch_size):
                self.names = names + [None] * (len(new_batch_size) - len(names))
            else:
                self.names = names[: self.batch_dims]

    @property
    def batch_dims(self) -> int:
        """Length of the tensordict batch size.

        Returns:
            int describing the number of dimensions of the tensordict.

        """
        return len(self.batch_size)

    def ndimension(self) -> int:
        return self.batch_dims

    @property
    def ndim(self) -> int:
        return self.batch_dims

    def dim(self) -> int:
        return self.batch_dims

    @property
    @abc.abstractmethod
    def device(self) -> torch.device | None:
        """Device of a TensorDict.

        If the TensorDict has a specified device, all
        tensors of a tensordict must live on the same device. If the TensorDict device
        is None, then different values can be located on different devices.

        Returns:
            torch.device object indicating the device where the tensors
            are placed, or None if TensorDict does not have a device.

        """
        raise NotImplementedError

    @device.setter
    @abc.abstractmethod
    def device(self, value: DeviceType) -> None:
        raise NotImplementedError

    def clear_device_(self) -> TensorDictBase:
        """Clears the device of the tensordict.

        Returns: self

        """
        self._device = None
        for value in self.values():
            if _is_tensor_collection(value.__class__):
                value.clear_device_()
        return self

    clear_device = _renamed_inplace_method(clear_device_)

    def is_shared(self) -> bool:
        """Checks if tensordict is in shared memory.

        If a TensorDict instance is in shared memory, any new tensor written
        in it will be placed in shared memory. If a TensorDict is created with
        tensors that are all in shared memory, this does not mean that it will be
        in shared memory (as a new tensor may not be in shared memory).
        Only if one calls `tensordict.share_memory_()` or places the tensordict
        on a device where the content is shared will the tensordict be considered
        in shared memory.

        This is always True for CUDA tensordicts, except when stored as
        MemmapTensors.

        """
        if self.device and not self._is_memmap:
            return self.device.type == "cuda" or self._is_shared
        return self._is_shared

    def state_dict(self) -> OrderedDict[str, Any]:
        out = collections.OrderedDict()
        for key, item in self.apply(memmap_tensor_as_tensor).items():
            out[key] = (
                item if not _is_tensor_collection(item.__class__) else item.state_dict()
            )
        if "__batch_size" in out:
            raise KeyError(
                "Cannot retrieve the state_dict of a TensorDict with `'__batch_size'` key"
            )
        if "__device" in out:
            raise KeyError(
                "Cannot retrieve the state_dict of a TensorDict with `'__batch_size'` key"
            )
        out["__batch_size"] = self.batch_size
        out["__device"] = self.device
        return out

    def load_state_dict(self, state_dict: OrderedDict[str, Any]) -> TensorDictBase:
        # copy since we'll be using pop
        state_dict = copy(state_dict)
        self.batch_size = state_dict.pop("__batch_size")
        device = state_dict.pop("__device")
        if device is not None:
            self.to(device)
        for key, item in state_dict.items():
            if isinstance(item, dict):
                self.set(
                    key,
                    self.get(key, default=TensorDict({}, [])).load_state_dict(item),
                    inplace=True,
                )
            else:
                self.set(key, item, inplace=True)
        return self

    def is_memmap(self) -> bool:
        """Checks if tensordict is stored with MemmapTensors."""
        return self._is_memmap

    def numel(self) -> int:
        """Total number of elements in the batch."""
        return max(1, prod(self.batch_size))

    def _check_batch_size(self) -> None:
        bs = [value.shape[: self.batch_dims] for value in self.values()] + [
            self.batch_size
        ]
        if len(set(bs)) > 1:
            raise RuntimeError(
                f"batch_size are incongruent, got {list(set(bs))}, "
                f"-- expected {self.batch_size}"
            )

    def _check_is_shared(self) -> bool:
        raise NotImplementedError(f"{self.__class__.__name__}")

    def _check_device(self) -> None:
        raise NotImplementedError(f"{self.__class__.__name__}")

    @abc.abstractmethod
    def entry_class(self, key: NestedKey) -> type:
        """Returns the class of an entry, avoiding a call to `isinstance(td.get(key), type)`."""
        raise NotImplementedError(f"{self.__class__.__name__}")

    def set(
        self, key: NestedKey, item: CompatibleType, inplace: bool = False, **kwargs: Any
    ) -> TensorDictBase:
        """Sets a new key-value pair.

        Args:
            key (str, tuple of str): name of the key to be set.
                If tuple of str it is equivalent to chained calls of getattr
            item (torch.Tensor): value to be stored in the tensordict
            inplace (bool, optional): if True and if a key matches an existing
                key in the tensordict, then the update will occur in-place
                for that key-value pair. Default is :obj:`False`.

        Returns:
            self

        """
        key = _unravel_key_to_tuple(key)
        # inplace is loose here, but for set_ it is constraining. We translate it
        # to None to tell _set_str and others to drop it if the key isn't found
        inplace = BEST_ATTEMPT_INPLACE if inplace else False
        return self._set_tuple(key, item, inplace=inplace, validated=False)

    def _convert_inplace(self, inplace, key):
        if inplace is not False:
            has_key = key in self.keys()
            if inplace is True and not has_key:  # inplace could be None
                raise KeyError(
                    TensorDictBase.KEY_ERROR.format(
                        key, self.__class__.__name__, sorted(self.keys())
                    )
                )
            inplace = has_key
        return inplace

    @abc.abstractmethod
    def _set_str(self, key, value, *, inplace, validated):
        ...

    @abc.abstractmethod
    def _set_tuple(self, key, value, *, inplace, validated):
        ...

    def set_at_(
        self, key: NestedKey, value: CompatibleType, idx: IndexType
    ) -> TensorDictBase:
        """Sets the values in-place at the index indicated by :obj:`idx`.

        Args:
            key (str, tuple of str): key to be modified.
            value (torch.Tensor): value to be set at the index `idx`
            idx (int, tensor or tuple): index where to write the values.

        Returns:
            self

        """
        key = _unravel_key_to_tuple(key)
        return self._set_at_tuple(key, value, idx, validated=False)

    @abc.abstractmethod
    def _set_at_str(self, key, value, idx, *, validated):
        ...

    @abc.abstractmethod
    def _set_at_tuple(self, key, value, idx, *, validated):
        ...

    def set_(
        self,
        key: NestedKey,
        item: CompatibleType,
    ) -> TensorDictBase:
        """Sets a value to an existing key while keeping the original storage.

        Args:
            key (str): name of the value
            item (torch.Tensor): value to be stored in the tensordict

        Returns:
            self

        """
        key = _unravel_key_to_tuple(key)
        return self._set_tuple(key, item, inplace=True, validated=False)

    @abc.abstractmethod
    def _stack_onto_(
        self,
        key: str,
        list_item: list[CompatibleType],
        dim: int,
    ) -> TensorDictBase:
        """Stacks a list of values onto an existing key while keeping the original storage.

        Args:
            key (str): name of the value
            list_item (list of torch.Tensor): value to be stacked and stored in the tensordict.
            dim (int): dimension along which the tensors should be stacked.

        Returns:
            self

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    def gather_and_stack(self, dst: int) -> TensorDictBase | None:
        """Gathers tensordicts from various workers and stacks them onto self in the destination worker.

        Args:
            dst (int): the rank of the destination worker where :func:`gather_and_stack` will be called.

        Example:
            >>> from torch import multiprocessing as mp
            >>> from tensordict import TensorDict
            >>> import torch
            >>>
            >>> def client():
            ...     torch.distributed.init_process_group(
            ...         "gloo",
            ...         rank=1,
            ...         world_size=2,
            ...         init_method=f"tcp://localhost:10003",
            ...     )
            ...     # Create a single tensordict to be sent to server
            ...     td = TensorDict(
            ...         {("a", "b"): torch.randn(2),
            ...          "c": torch.randn(2)}, [2]
            ...     )
            ...     td.gather_and_stack(0)
            ...
            >>> def server():
            ...     torch.distributed.init_process_group(
            ...         "gloo",
            ...         rank=0,
            ...         world_size=2,
            ...         init_method=f"tcp://localhost:10003",
            ...     )
            ...     # Creates the destination tensordict on server.
            ...     # The first dim must be equal to world_size-1
            ...     td = TensorDict(
            ...         {("a", "b"): torch.zeros(2),
            ...          "c": torch.zeros(2)}, [2]
            ...     ).expand(1, 2).contiguous()
            ...     td.gather_and_stack(0)
            ...     assert td["a", "b"] != 0
            ...     print("yuppie")
            ...
            >>> if __name__ == "__main__":
            ...     mp.set_start_method("spawn")
            ...
            ...     main_worker = mp.Process(target=server)
            ...     secondary_worker = mp.Process(target=client)
            ...
            ...     main_worker.start()
            ...     secondary_worker.start()
            ...
            ...     main_worker.join()
            ...     secondary_worker.join()
        """
        output = (
            [None for _ in range(dist.get_world_size())]
            if dst == dist.get_rank()
            else None
        )
        dist.gather_object(self, output, dst=dst)
        if dst == dist.get_rank():
            # remove self from output
            output = [item for i, item in enumerate(output) if i != dst]
            self.update(torch.stack(output, 0), inplace=True)
            return self
        return None

    def send(self, dst: int, init_tag: int = 0, pseudo_rand: bool = False) -> None:
        """Sends the content of a tensordict to a distant worker.

        Args:
            dst (int): the rank of the destination worker where the content
                should be sent.
            init_tag (int): the initial tag to be used to mark the tensors.
                Note that this will be incremented by as much as the number of
                tensors contained in the TensorDict.
            pseudo_rand (bool): if True, the sequence of tags will be pseudo-
                random, allowing to send multiple data from different nodes
                without overlap. Notice that the generation of these pseudo-random
                numbers is expensive (1e-5 sec/number), meaning that it could
                slow down the runtime of your algorithm.
                Defaults to ``False``.

        Example:
            >>> from torch import multiprocessing as mp
            >>> from tensordict import TensorDict
            >>> import torch
            >>>
            >>>
            >>> def client():
            ...     torch.distributed.init_process_group(
            ...         "gloo",
            ...         rank=1,
            ...         world_size=2,
            ...         init_method=f"tcp://localhost:10003",
            ...     )
            ...
            ...     td = TensorDict(
            ...         {
            ...             ("a", "b"): torch.randn(2),
            ...             "c": torch.randn(2, 3),
            ...             "_": torch.ones(2, 1, 5),
            ...         },
            ...         [2],
            ...     )
            ...     td.send(0)
            ...
            >>>
            >>> def server(queue):
            ...     torch.distributed.init_process_group(
            ...         "gloo",
            ...         rank=0,
            ...         world_size=2,
            ...         init_method=f"tcp://localhost:10003",
            ...     )
            ...     td = TensorDict(
            ...         {
            ...             ("a", "b"): torch.zeros(2),
            ...             "c": torch.zeros(2, 3),
            ...             "_": torch.zeros(2, 1, 5),
            ...         },
            ...         [2],
            ...     )
            ...     td.recv(1)
            ...     assert (td != 0).all()
            ...     queue.put("yuppie")
            ...
            >>>
            >>> if __name__=="__main__":
            ...     queue = mp.Queue(1)
            ...     main_worker = mp.Process(target=server, args=(queue,))
            ...     secondary_worker = mp.Process(target=client)
            ...
            ...     main_worker.start()
            ...     secondary_worker.start()
            ...     out = queue.get(timeout=10)
            ...     assert out == "yuppie"
            ...     main_worker.join()
            ...     secondary_worker.join()

        """
        self._send(dst, _tag=init_tag - 1, pseudo_rand=pseudo_rand)

    def _send(self, dst: int, _tag: int = -1, pseudo_rand: bool = False) -> int:
        for key in self.sorted_keys:
            value = self._get_str(key, NO_DEFAULT)
            if isinstance(value, Tensor):
                pass
            elif _is_tensor_collection(value.__class__):
                _tag = value._send(dst, _tag=_tag, pseudo_rand=pseudo_rand)
                continue
            elif isinstance(value, MemmapTensor):
                value = value.as_tensor()
            else:
                raise NotImplementedError(f"Type {type(value)} is not supported.")
            if not pseudo_rand:
                _tag += 1
            else:
                _tag = int_generator(_tag + 1)
            dist.send(value, dst=dst, tag=_tag)

        return _tag

    def recv(self, src: int, init_tag: int = 0, pseudo_rand: bool = False) -> int:
        """Receives the content of a tensordict and updates content with it.

        Check the example in the `send` method for context.

        Args:
            src (int): the rank of the source worker.
            init_tag (int): the ``init_tag`` used by the source worker.
            pseudo_rand (bool): if True, the sequence of tags will be pseudo-
                random, allowing to send multiple data from different nodes
                without overlap. Notice that the generation of these pseudo-random
                numbers is expensive (1e-5 sec/number), meaning that it could
                slow down the runtime of your algorithm.
                This value must match the one passed to :func:`send`.
                Defaults to ``False``.

        """
        return self._recv(src, _tag=init_tag - 1, pseudo_rand=pseudo_rand)

    def _recv(self, src: int, _tag: int = -1, pseudo_rand: bool = False) -> int:
        for key in self.sorted_keys:
            value = self._get_str(key, NO_DEFAULT)
            if isinstance(value, Tensor):
                pass
            elif _is_tensor_collection(value.__class__):
                _tag = value._recv(src, _tag=_tag, pseudo_rand=pseudo_rand)
                continue
            elif isinstance(value, MemmapTensor):
                value = value.as_tensor()
            else:
                raise NotImplementedError(f"Type {type(value)} is not supported.")
            if not pseudo_rand:
                _tag += 1
            else:
                _tag = int_generator(_tag + 1)
            dist.recv(value, src=src, tag=_tag)
            self._set_str(key, value, inplace=True, validated=True)

        return _tag

    def isend(self, dst: int, init_tag: int = 0, pseudo_rand: bool = False) -> int:
        """Sends the content of the tensordict asynchronously.

        Args:
            dst (int): the rank of the destination worker where the content
                should be sent.
            init_tag (int): the initial tag to be used to mark the tensors.
                Note that this will be incremented by as much as the number of
                tensors contained in the TensorDict.
            pseudo_rand (bool): if True, the sequence of tags will be pseudo-
                random, allowing to send multiple data from different nodes
                without overlap. Notice that the generation of these pseudo-random
                numbers is expensive (1e-5 sec/number), meaning that it could
                slow down the runtime of your algorithm.
                Defaults to ``False``.

        Example:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> from torch import multiprocessing as mp
            >>> def client():
            ...     torch.distributed.init_process_group(
            ...         "gloo",
            ...         rank=1,
            ...         world_size=2,
            ...         init_method=f"tcp://localhost:10003",
            ...     )
            ...
            ...     td = TensorDict(
            ...         {
            ...             ("a", "b"): torch.randn(2),
            ...             "c": torch.randn(2, 3),
            ...             "_": torch.ones(2, 1, 5),
            ...         },
            ...         [2],
            ...     )
            ...     td.isend(0)
            ...
            >>>
            >>> def server(queue, return_premature=True):
            ...     torch.distributed.init_process_group(
            ...         "gloo",
            ...         rank=0,
            ...         world_size=2,
            ...         init_method=f"tcp://localhost:10003",
            ...     )
            ...     td = TensorDict(
            ...         {
            ...             ("a", "b"): torch.zeros(2),
            ...             "c": torch.zeros(2, 3),
            ...             "_": torch.zeros(2, 1, 5),
            ...         },
            ...         [2],
            ...     )
            ...     out = td.irecv(1, return_premature=return_premature)
            ...     if return_premature:
            ...         for fut in out:
            ...             fut.wait()
            ...     assert (td != 0).all()
            ...     queue.put("yuppie")
            ...
            >>>
            >>> if __name__ == "__main__":
            ...     queue = mp.Queue(1)
            ...     main_worker = mp.Process(
            ...         target=server,
            ...         args=(queue, )
            ...         )
            ...     secondary_worker = mp.Process(target=client)
            ...
            ...     main_worker.start()
            ...     secondary_worker.start()
            ...     out = queue.get(timeout=10)
            ...     assert out == "yuppie"
            ...     main_worker.join()
            ...     secondary_worker.join()

        """
        return self._isend(dst, init_tag - 1, pseudo_rand=pseudo_rand)

    def _isend(
        self,
        dst: int,
        _tag: int = -1,
        _futures: list[torch.Future] | None = None,
        pseudo_rand: bool = False,
    ) -> int:
        root = False
        if _futures is None:
            root = True
            _futures = []
        for key in self.sorted_keys:
            value = self._get_str(key, NO_DEFAULT)
            if _is_tensor_collection(value.__class__):
                _tag = value._isend(
                    dst, _tag=_tag, pseudo_rand=pseudo_rand, _futures=_futures
                )
                continue
            elif isinstance(value, Tensor):
                pass
            elif isinstance(value, MemmapTensor):
                value = value.as_tensor()
            else:
                raise NotImplementedError(f"Type {type(value)} is not supported.")
            if not pseudo_rand:
                _tag += 1
            else:
                _tag = int_generator(_tag + 1)
            _future = dist.isend(value, dst=dst, tag=_tag)
            _futures.append(_future)
        if root:
            for _future in _futures:
                _future.wait()
        return _tag

    def irecv(
        self,
        src: int,
        return_premature: bool = False,
        init_tag: int = 0,
        pseudo_rand: bool = False,
    ) -> tuple[int, list[torch.Future]] | list[torch.Future] | None:
        """Receives the content of a tensordict and updates content with it asynchronously.

        Check the example in the `isend` method for context.

        Args:
            src (int): the rank of the source worker.
            return_premature (bool): if ``True``, returns a list of futures to wait
                upon until the tensordict is updated. Defaults to ``False``,
                i.e. waits until update is completed withing the call.
            init_tag (int): the ``init_tag`` used by the source worker.
            pseudo_rand (bool): if True, the sequence of tags will be pseudo-
                random, allowing to send multiple data from different nodes
                without overlap. Notice that the generation of these pseudo-random
                numbers is expensive (1e-5 sec/number), meaning that it could
                slow down the runtime of your algorithm.
                This value must match the one passed to :func:`isend`.
                Defaults to ``False``.

        Returns:
            if ``return_premature=True``, a list of futures to wait
                upon until the tensordict is updated.
        """
        return self._irecv(
            src, return_premature, _tag=init_tag - 1, pseudo_rand=pseudo_rand
        )

    def _irecv(
        self,
        src: int,
        return_premature: bool = False,
        _tag: int = -1,
        _future_list: list[torch.Future] = None,
        pseudo_rand: bool = False,
    ) -> tuple[int, list[torch.Future]] | list[torch.Future] | None:
        root = False
        if _future_list is None:
            _future_list = []
            root = True

        for key in self.sorted_keys:
            value = self._get_str(key, NO_DEFAULT)
            if _is_tensor_collection(value.__class__):
                _tag, _future_list = value._irecv(
                    src,
                    _tag=_tag,
                    _future_list=_future_list,
                    pseudo_rand=pseudo_rand,
                )
                continue
            elif isinstance(value, MemmapTensor):
                value = value.as_tensor()
            elif isinstance(value, Tensor):
                pass
            else:
                raise NotImplementedError(f"Type {type(value)} is not supported.")
            if not pseudo_rand:
                _tag += 1
            else:
                _tag = int_generator(_tag + 1)
            _future_list.append(dist.irecv(value, src=src, tag=_tag))
        if not root:
            return _tag, _future_list
        elif return_premature:
            return _future_list
        else:
            for future in _future_list:
                future.wait()
            return

    def reduce(self, dst, op=dist.ReduceOp.SUM, async_op=False, return_premature=False):
        """Reduces the tensordict across all machines.

        Only the process with ``rank`` dst is going to receive the final result.

        """
        return self._reduce(dst, op, async_op, return_premature)

    def _reduce(
        self,
        dst,
        op=dist.ReduceOp.SUM,
        async_op=False,
        return_premature=False,
        _future_list=None,
    ):
        root = False
        if _future_list is None:
            _future_list = []
            root = True
        for key in self.sorted_keys:
            value = self._get_str(key, NO_DEFAULT)
            if _is_tensor_collection(value.__class__):
                _future_list = value._reduce(
                    dst=dst,
                    op=op,
                    async_op=async_op,
                    _future_list=_future_list,
                )
                continue
            elif isinstance(value, MemmapTensor):
                value = value.as_tensor()
            elif isinstance(value, Tensor):
                pass
            else:
                raise NotImplementedError(f"Type {type(value)} is not supported.")
            _future_list.append(dist.reduce(value, dst=dst, op=op, async_op=async_op))
        if not root:
            return _future_list
        elif async_op and return_premature:
            return _future_list
        elif async_op:
            for future in _future_list:
                future.wait()
            return

    def _stack_onto_at_(
        self,
        key: str,
        list_item: list[CompatibleType],
        dim: int,
        idx: IndexType,
    ) -> TensorDictBase:
        """Similar to _stack_onto_ but on a specific index. Only works with regular TensorDicts."""
        raise RuntimeError(
            f"Cannot call _stack_onto_at_ with {self.__class__.__name__}. "
            "This error is probably caused by a call to a lazy operation before stacking. "
            "Make sure your sub-classed tensordicts are turned into regular tensordicts by calling to_tensordict() "
            "before calling __getindex__ and stack."
        )

    def _default_get(
        self, key: str, default: str | CompatibleType = NO_DEFAULT
    ) -> CompatibleType:
        if default is not NO_DEFAULT:
            return default
        else:
            # raise KeyError
            raise KeyError(
                TensorDictBase.KEY_ERROR.format(
                    key, self.__class__.__name__, sorted(self.keys())
                )
            )

    def get(
        self, key: NestedKey, default: str | CompatibleType = NO_DEFAULT
    ) -> CompatibleType:
        """Gets the value stored with the input key.

        Args:
            key (str, tuple of str): key to be queried. If tuple of str it is
                equivalent to chained calls of getattr.
            default: default value if the key is not found in the tensordict.

        """
        key = _unravel_key_to_tuple(key)
        if not key:
            raise KeyError(_GENERIC_NESTED_ERR)
        return self._get_tuple(key, default=default)

    @abc.abstractmethod
    def _get_str(self, key, default):
        ...

    @abc.abstractmethod
    def _get_tuple(self, key, default):
        ...

    def get_item_shape(self, key: NestedKey):
        """Returns the shape of the entry."""
        return self.get(key).shape

    def pop(
        self, key: NestedKey, default: str | CompatibleType = NO_DEFAULT
    ) -> CompatibleType:
        key = _unravel_key_to_tuple(key)
        if not key:
            raise KeyError(_GENERIC_NESTED_ERR)
        try:
            # using try/except for get/del is suboptimal, but
            # this is faster that checkink if key in self keys
            out = self.get(key, default)
            self.del_(key)
        except KeyError as err:
            # if default provided, 'out' value will return, else raise error
            if default == NO_DEFAULT:
                raise KeyError(
                    f"You are trying to pop key `{key}` which is not in dict "
                    f"without providing default value."
                ) from err
        return out

    def apply_(self, fn: Callable, *others) -> TensorDictBase:
        """Applies a callable to all values stored in the tensordict and re-writes them in-place.

        Args:
            fn (Callable): function to be applied to the tensors in the
                tensordict.
            *others (sequence of TensorDictBase, optional): the other
                tensordicts to be used.

        Returns:
            self or a copy of self with the function applied

        """
        return self.apply(fn, *others, inplace=True)

    def apply(
        self,
        fn: Callable,
        *others: TensorDictBase,
        batch_size: Sequence[int] | None = None,
        device: torch.device | None = None,
        names: Sequence[str] | None = None,
        inplace: bool = False,
        **constructor_kwargs,
    ) -> TensorDictBase:
        """Applies a callable to all values stored in the tensordict and sets them in a new tensordict.

        The apply method will return an TensorDict instance, regardless of the
        input type. To keep the same type, one can execute

          >>> out = td.clone(False).update(td.apply(...))

        Args:
            fn (Callable): function to be applied to the tensors in the
                tensordict.
            *others (TensorDictBase instances, optional): if provided, these
                tensordicts should have a structure matching the one of the
                current tensordict. The :obj:`fn` argument should receive as many
                inputs as the number of tensordicts, including the one where apply is
                being called.
            batch_size (sequence of int, optional): if provided,
                the resulting TensorDict will have the desired batch_size.
                The :obj:`batch_size` argument should match the batch_size after
                the transformation. This is a keyword only argument.
            device (torch.device, optional): the resulting device, if any.
            names (list of str, optional): the new dimension names, in case the
                batch_size is modified.
            inplace (bool, optional): if True, changes are made in-place.
                Default is False. This is a keyword only argument.
            **constructor_kwargs: additional keyword arguments to be passed to the
                TensorDict constructor.

        Returns:
            a new tensordict with transformed_in tensors.

        Example:
            >>> td = TensorDict({"a": -torch.ones(3), "b": {"c": torch.ones(3)}}, batch_size=[3])
            >>> td_1 = td.apply(lambda x: x+1)
            >>> assert (td["a"] == 0).all()
            >>> assert (td["b", "c"] == 2).all()
            >>> td_2 = td.apply(lambda x, y: x+y, td)
            >>> assert (td_2["a"] == -2).all()
            >>> assert (td_2["b", "c"] == 2).all()
        """
        if inplace:
            out = self
        elif batch_size is not None:
            out = TensorDict(
                {},
                batch_size=torch.Size(batch_size),
                names=names,
                device=self.device if not device else device,
                _run_checks=False,
                **constructor_kwargs,
            )
        else:
            out = TensorDict(
                {},
                batch_size=self.batch_size,
                device=self.device if not device else device,
                names=self.names if self._has_names() else None,
                _run_checks=False,
                **constructor_kwargs,
            )

        is_locked = out.is_locked
        if not inplace and is_locked:
            out.unlock_()

        for key, item in self.items():
            _others = [_other.get(key) for _other in others]
            if _is_tensor_collection(item.__class__):
                item_trsf = item.apply(
                    fn,
                    *_others,
                    inplace=inplace,
                    batch_size=batch_size,
                    device=device,
                    **constructor_kwargs,
                )
            else:
                item_trsf = fn(item, *_others)
            if item_trsf is not None:
                # if `self` is a `SubTensorDict` we want to process the input,
                # hence we call `set` rather than `_set_str`.
                if isinstance(self, SubTensorDict):
                    out.set(key, item_trsf, inplace=inplace)
                else:
                    out._set_str(
                        key,
                        item_trsf,
                        inplace=BEST_ATTEMPT_INPLACE if inplace else False,
                        validated=False,
                    )

        if not inplace and is_locked:
            out.lock_()
        return out

    @cache  # noqa: B019
    def _add_batch_dim(self, *, in_dim, vmap_level):
        if self.is_memmap():
            td = self.cpu().as_tensor()
        else:
            td = self
        out = TensorDict(
            {
                key: value._add_batch_dim(in_dim=in_dim, vmap_level=vmap_level)
                if is_tensor_collection(value)
                else _add_batch_dim(value, in_dim, vmap_level)
                for key, value in td.items()
            },
            batch_size=[b for i, b in enumerate(td.batch_size) if i != in_dim],
            names=[name for i, name in enumerate(td.names) if i != in_dim],
        )
        return out

    @cache  # noqa: B019
    def _remove_batch_dim(self, vmap_level, batch_size, out_dim):
        new_batch_size = list(self.batch_size)
        new_batch_size.insert(out_dim, batch_size)
        new_names = list(self.names)
        new_names.insert(out_dim, None)
        out = TensorDict(
            {
                key: value._remove_batch_dim(
                    vmap_level=vmap_level, batch_size=batch_size, out_dim=out_dim
                )
                if is_tensor_collection(value)
                else _remove_batch_dim(value, vmap_level, batch_size, out_dim)
                for key, value in self.items()
            },
            batch_size=new_batch_size,
            names=new_names,
        )
        return out

    def as_tensor(self):
        """Calls as_tensor on all the tensors contained in the object.

        This is reserved to classes that contain exclusively MemmapTensors,
        and will raise an exception in all other cases.

        """
        try:
            return self.apply(lambda x: x.as_tensor())
        except AttributeError as err:
            raise AttributeError(
                f"{self.__class__.__name__} does not have an 'as_tensor' method "
                f"because at least one of its tensors does not support this method."
            ) from err

    def update(
        self,
        input_dict_or_td: dict[str, CompatibleType] | TensorDictBase,
        clone: bool = False,
        inplace: bool = False,
    ) -> TensorDictBase:
        """Updates the TensorDict with values from either a dictionary or another TensorDict.

        Args:
            input_dict_or_td (TensorDictBase or dict): Does not keyword arguments
                (unlike :obj:`dict.update()`).
            clone (bool, optional): whether the tensors in the input (
                tensor) dict should be cloned before being set. Default is
                `False`.
            inplace (bool, optional): if True and if a key matches an existing
                key in the tensordict, then the update will occur in-place
                for that key-value pair. Default is :obj:`False`.
            **kwargs: keyword arguments for the :obj:`TensorDict.set` method

        Returns:
            self

        """
        if input_dict_or_td is self:
            # no op
            return self
        keys = set(self.keys(False))
        for key, value in list(input_dict_or_td.items()):
            if clone and hasattr(value, "clone"):
                value = value.clone()
            if isinstance(key, tuple):
                key, subkey = key[0], key[1:]
            else:
                subkey = []
            # the key must be a string by now. Let's check if it is present
            if key in keys:
                target_type = self.entry_class(key)
                if _is_tensor_collection(target_type):
                    target = self.get(key)
                    if len(subkey):
                        target.update({subkey: value})
                        continue
                    elif isinstance(value, (dict,)) or _is_tensor_collection(
                        value.__class__
                    ):
                        if isinstance(value, LazyStackedTensorDict) and not isinstance(
                            target, LazyStackedTensorDict
                        ):
                            self.set(
                                key,
                                LazyStackedTensorDict(
                                    *target.unbind(value.stack_dim),
                                    stack_dim=value.stack_dim,
                                ).update(value),
                            )
                        else:
                            target.update(value)
                        continue
            if len(subkey):
                self.set((key, *subkey), value, inplace=inplace)
            else:
                self.set(key, value, inplace=inplace)
        return self

    def update_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | TensorDictBase,
        clone: bool = False,
    ) -> TensorDictBase:
        """Updates the TensorDict in-place with values from either a dictionary or another TensorDict.

        Unlike TensorDict.update, this function will
        throw an error if the key is unknown to the TensorDict

        Args:
            input_dict_or_td (TensorDictBase or dict): Does not keyword
                arguments (unlike :obj:`dict.update()`).
            clone (bool, optional): whether the tensors in the input (
                tensor) dict should be cloned before being set. Default is
                `False`.

        Returns:
            self

        """
        if input_dict_or_td is self:
            # no op
            return self
        for key, value in input_dict_or_td.items():
            # if not isinstance(value, _accepted_classes):
            #     raise TypeError(
            #         f"Expected value to be one of types {_accepted_classes} "
            #         f"but got {type(value)}"
            #     )
            if clone:
                value = value.clone()
            self.set_(key, value)
        return self

    def update_at_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | TensorDictBase,
        idx: IndexType,
        clone: bool = False,
    ) -> TensorDictBase:
        """Updates the TensorDict in-place at the specified index with values from either a dictionary or another TensorDict.

        Unlike  TensorDict.update, this function will throw an error if the key is unknown to the TensorDict.

        Args:
            input_dict_or_td (TensorDictBase or dict): Does not keyword arguments
                (unlike :obj:`dict.update()`).
            idx (int, torch.Tensor, iterable, slice): index of the tensordict
                where the update should occur.
            clone (bool, optional): whether the tensors in the input (
                tensor) dict should be cloned before being set. Default is
                `False`.

        Returns:
            self

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3, 4, 5),
            ...    'b': torch.zeros(3, 4, 10)}, batch_size=[3, 4])
            >>> td.update_at_(
            ...    TensorDict(source={'a': torch.ones(1, 4, 5),
            ...        'b': torch.ones(1, 4, 10)}, batch_size=[1, 4]),
            ...    slice(1, 2))
            TensorDict(
                fields={
                    a: Tensor(torch.Size([3, 4, 5]), dtype=torch.float32),
                    b: Tensor(torch.Size([3, 4, 10]), dtype=torch.float32)},
                batch_size=torch.Size([3, 4]),
                device=None,
                is_shared=False)

        """
        for key, value in input_dict_or_td.items():
            if not isinstance(value, _ACCEPTED_CLASSES):
                raise TypeError(
                    f"Expected value to be one of types {_ACCEPTED_CLASSES} "
                    f"but got {type(value)}"
                )
            if clone:
                value = value.clone()
            self.set_at_(key, value, idx)
        return self

    def _convert_to_tensor(self, array: np.ndarray) -> Tensor | MemmapTensor:
        if isinstance(array, np.bool_):
            array = array.item()
        if isinstance(array, list):
            array = np.asarray(array)
        return torch.as_tensor(array, device=self.device)

    def _convert_to_tensordict(self, dict_value: dict[str, Any]) -> TensorDictBase:
        return TensorDict(
            dict_value,
            batch_size=self.batch_size,
            device=self.device,
            _is_shared=self._is_shared,
            _is_memmap=self._is_memmap,
        )

    def _validate_key(self, key: NestedKey) -> NestedKey:
        key = _unravel_key_to_tuple(key)
        if not key:
            raise KeyError(_GENERIC_NESTED_ERR)
        return key

    def _validate_value(
        self,
        value: CompatibleType | dict[str, CompatibleType],
        *,
        check_shape: bool = True,
    ) -> CompatibleType | dict[str, CompatibleType]:
        cls = value.__class__
        is_tc = _is_tensor_collection(cls)
        if is_tc or issubclass(cls, _ACCEPTED_CLASSES):
            pass
        elif issubclass(cls, dict):
            value = self._convert_to_tensordict(value)
            is_tc = True
        else:
            try:
                value = self._convert_to_tensor(value)
            except ValueError as err:
                raise ValueError(
                    f"TensorDict conversion only supports tensorclasses, tensordicts,"
                    f" numeric scalars and tensors. Got {type(value)}"
                ) from err
        bs = self.batch_size
        if check_shape and bs and _shape(value)[: len(bs)] != bs:
            # if TensorDict, let's try to map it to the desired shape
            if is_tc:
                value = value.clone(recurse=False)
                value.batch_size = self.batch_size
            else:
                raise RuntimeError(
                    f"batch dimension mismatch, got self.batch_size"
                    f"={self.batch_size} and value.shape[:self.batch_dims]"
                    f"={_shape(value)[: self.batch_dims]} with value {value}"
                )
        device = self.device
        if device is not None and value.device != device:
            value = value.to(device, non_blocking=True)
        if is_tc and check_shape:
            has_names = self._has_names()
            if has_names and value.names[: self.ndim] != self.names:
                value = value.clone(False).refine_names(*self.names)
            elif not has_names and value._has_names():
                self.names = value.names[: self.batch_dims]

        return value

    @abc.abstractmethod
    def pin_memory(self) -> TensorDictBase:
        """Calls :obj:`pin_memory` on the stored tensors."""
        raise NotImplementedError(f"{self.__class__.__name__}")

    def items(
        self, include_nested: bool = False, leaves_only: bool = False
    ) -> Iterator[tuple[str, CompatibleType]]:
        """Returns a generator of key-value pairs for the tensordict."""
        # check the conditions once only
        if include_nested and leaves_only:
            for k in self.keys():
                val = self._get_str(k, NO_DEFAULT)
                if _is_tensor_collection(val.__class__):
                    yield from (
                        (_unravel_key_to_tuple((k, _key)), _val)
                        for _key, _val in val.items(
                            include_nested=include_nested, leaves_only=leaves_only
                        )
                    )
                else:
                    yield k, val
        elif include_nested:
            for k in self.keys():
                val = self._get_str(k, NO_DEFAULT)
                yield k, val
                if _is_tensor_collection(val.__class__):
                    yield from (
                        (_unravel_key_to_tuple((k, _key)), _val)
                        for _key, _val in val.items(
                            include_nested=include_nested, leaves_only=leaves_only
                        )
                    )
        elif leaves_only:
            for k in self.keys():
                val = self._get_str(k, NO_DEFAULT)
                if not _is_tensor_collection(val.__class__):
                    yield k, val
        else:
            for k in self.keys():
                yield k, self._get_str(k, NO_DEFAULT)

    def values(
        self, include_nested: bool = False, leaves_only: bool = False
    ) -> Iterator[CompatibleType]:
        """Returns a generator representing the values for the tensordict."""
        # check the conditions once only
        if include_nested and leaves_only:
            for k in self.keys():
                val = self._get_str(k, NO_DEFAULT)
                if _is_tensor_collection(val.__class__):
                    yield from val.values(
                        include_nested=include_nested, leaves_only=leaves_only
                    )
                else:
                    yield val
        elif include_nested:
            for k in self.keys():
                val = self._get_str(k, NO_DEFAULT)
                yield val
                if _is_tensor_collection(val.__class__):
                    yield from val.values(
                        include_nested=include_nested, leaves_only=leaves_only
                    )
        elif leaves_only:
            for k in self.keys():
                val = self._get_str(k, NO_DEFAULT)
                if not _is_tensor_collection(val.__class__):
                    yield val
        else:
            for k in self.keys():
                yield self._get_str(k, NO_DEFAULT)

    @abc.abstractmethod
    def keys(
        self, include_nested: bool = False, leaves_only: bool = False
    ) -> _TensorDictKeysView:
        """Returns a generator of tensordict keys."""
        raise NotImplementedError(f"{self.__class__.__name__}")

    @property
    @cache  # noqa: B019
    def sorted_keys(self) -> list[NestedKey]:
        """Returns the keys sorted in alphabetical order.

        Does not support extra argument.

        If the TensorDict is locked, the keys are cached until the tensordict
        is unlocked.

        """
        return sorted(self.keys())

    def expand(self, *shape: int) -> TensorDictBase:
        """Expands each tensors of the tensordict according to the torch.expand function.

        In practice, this amends to: :obj:`tensor.expand(*shape, *tensor.shape)`.

        Supports iterables to specify the shape

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3, 4, 5),
            ...     'b': torch.zeros(3, 4, 10)}, batch_size=[3, 4])
            >>> td_expand = td.expand(10, 3, 4)
            >>> assert td_expand.shape == torch.Size([10, 3, 4])
            >>> assert td_expand.get("a").shape == torch.Size([10, 3, 4, 5])

        """
        d = {}
        tensordict_dims = self.batch_dims

        if len(shape) == 1 and isinstance(shape[0], Sequence):
            shape = tuple(shape[0])

        # new shape dim check
        if len(shape) < len(self.shape):
            raise RuntimeError(
                "the number of sizes provided ({shape_dim}) must be greater or equal to the number of "
                "dimensions in the TensorDict ({tensordict_dim})".format(
                    shape_dim=len(shape), tensordict_dim=tensordict_dims
                )
            )

        # new shape compatability check
        for old_dim, new_dim in zip(self.batch_size, shape[-tensordict_dims:]):
            if old_dim != 1 and new_dim != old_dim:
                raise RuntimeError(
                    "Incompatible expanded shape: The expanded shape length at non-singleton dimension should be same "
                    "as the original length. target_shape = {new_shape}, existing_shape = {old_shape}".format(
                        new_shape=shape, old_shape=self.batch_size
                    )
                )
        for key, value in self.items():
            tensor_dims = len(value.shape)
            last_n_dims = tensor_dims - tensordict_dims
            if last_n_dims > 0:
                d[key] = value.expand((*shape, *value.shape[-last_n_dims:]))
            else:
                d[key] = value.expand(shape)
        return TensorDict(
            source=d,
            batch_size=torch.Size(shape),
            device=self.device,
            _run_checks=False,
        )

    def flatten(self, start_dim=0, end_dim=-1):
        """Flattens all the tensors of a tensordict.

        Args:
            start_dim (int) – the first dim to flatten
            end_dim (int) – the last dim to flatten

        Examples:
            >>> td = TensorDict({"a": torch.arange(60).view(3, 4, 5), "b": torch.arange(12).view(3, 4)}, [3, 4])
            >>> td_flat = td.flatten(0, 1)
            >>> td_flat.batch_size
            torch.Size([12])
            >>> td_flat["a"]
            tensor([[ 0,  1,  2,  3,  4],
                    [ 5,  6,  7,  8,  9],
                    [10, 11, 12, 13, 14],
                    [15, 16, 17, 18, 19],
                    [20, 21, 22, 23, 24],
                    [25, 26, 27, 28, 29],
                    [30, 31, 32, 33, 34],
                    [35, 36, 37, 38, 39],
                    [40, 41, 42, 43, 44],
                    [45, 46, 47, 48, 49],
                    [50, 51, 52, 53, 54],
                    [55, 56, 57, 58, 59]])
            >>> td_flat["b"]
            tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

        """
        if end_dim < 0:
            end_dim = self.ndim + end_dim
            if end_dim < 0:
                raise ValueError(
                    f"Incompatible end_dim {end_dim} for tensordict with shape {self.shape}."
                )
        if end_dim <= start_dim:
            raise ValueError(
                "The end dimension must be strictly greater than the start dim."
            )

        def flatten(tensor):
            return torch.flatten(tensor, start_dim, end_dim)

        nelt = prod(self.batch_size[start_dim : end_dim + 1])
        if start_dim > 0:
            batch_size = (
                list(self.batch_size)[:start_dim]
                + [nelt]
                + list(self.batch_size[end_dim + 1 :])
            )
        else:
            batch_size = [nelt] + list(self.batch_size[end_dim + 1 :])
        out = self.apply(flatten, batch_size=batch_size)
        if self._has_names():
            names = [
                name
                for i, name in enumerate(self.names)
                if (i < start_dim or i > end_dim)
            ]
            names.insert(start_dim, None)
            out.names = names
        return out

    def unflatten(self, dim, unflattened_size):
        """Unflattens a tensordict dim expanding it to a desired shape.

        Args:
            dim (int): specifies the dimension of the input tensor to be unflattened.
            unflattened_size (shape): is the new shape of the unflattened dimension of the tensordict.

        Examples:
            >>> td = TensorDict({"a": torch.arange(60).view(3, 4, 5), "b": torch.arange(12).view(3, 4)}, [3, 4])
            >>> td_flat = td.flatten(0, 1)
            >>> td_unflat = td_flat.unflatten(0, [3, 4])
            >>> assert (td == td_unflat).all()
        """
        if dim < 0:
            dim = self.ndim + dim
            if dim < 0:
                raise ValueError(
                    f"Incompatible dim {dim} for tensordict with shape {self.shape}."
                )

        def unflatten(tensor):
            return torch.unflatten(
                tensor,
                dim,
                unflattened_size,
            )

        if dim > 0:
            batch_size = (
                list(self.batch_size)[:dim]
                + list(unflattened_size)
                + list(self.batch_size[dim + 1 :])
            )
        else:
            batch_size = list(unflattened_size) + list(self.batch_size[1:])
        out = self.apply(unflatten, batch_size=batch_size)
        if self._has_names():
            names = copy(self.names)
            for _ in range(len(unflattened_size) - 1):
                names.insert(dim, None)
            out.names = names
        return out

    def __enter__(self):
        self._last_op_queue.append(self._last_op)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _last_op = self._last_op_queue.pop()
        if _last_op is not None:
            last_op, (args, kwargs) = _last_op
            if last_op is self.__class__.lock_.__name__:
                return self.unlock_()
            elif last_op is self.__class__.unlock_.__name__:
                return self.lock_()
            else:
                raise NotImplementedError(f"Unrecognised function {last_op}.")
        return self

    def __bool__(self) -> bool:
        raise ValueError("Converting a tensordict to boolean value is not permitted")

    def __ne__(self, other: object) -> TensorDictBase:
        """XOR operation over two tensordicts, for evey key.

        The two tensordicts must have the same key set.

        Args:
            other (TensorDictBase, dict, or float): the value to compare against.

        Returns:
            a new TensorDict instance with all tensors are boolean
            tensors of the same shape as the original tensors.

        """
        if _is_tensorclass(other.__class__):
            return other != self
        if isinstance(other, (dict,)) or _is_tensor_collection(other.__class__):
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                raise KeyError(
                    f"keys in {self} and {other} mismatch, got {keys1} and {keys2}"
                )
            d = {}
            for key, item1 in self.items():
                d[key] = item1 != other.get(key)
            return TensorDict(batch_size=self.batch_size, source=d, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return TensorDict(
                {key: value != other for key, value in self.items()},
                self.batch_size,
                device=self.device,
            )
        return True

    def __eq__(self, other: object) -> TensorDictBase:
        """Compares two tensordicts against each other, for every key. The two tensordicts must have the same key set.

        Returns:
            a new TensorDict instance with all tensors are boolean
            tensors of the same shape as the original tensors.

        """
        if is_tensorclass(other):
            return other == self
        if isinstance(other, (dict,)) or _is_tensor_collection(other.__class__):
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                raise KeyError(f"keys in tensordicts mismatch, got {keys1} and {keys2}")
            d = {}
            for key, item1 in self.items():
                d[key] = item1 == other.get(key)
            return TensorDict(batch_size=self.batch_size, source=d, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return TensorDict(
                {key: value == other for key, value in self.items()},
                self.batch_size,
                device=self.device,
            )
        return False

    @abc.abstractmethod
    def del_(self, key: str) -> TensorDictBase:
        """Deletes a key of the tensordict.

        Args:
            key (str): key to be deleted

        Returns:
            self

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    @abc.abstractmethod
    def select(
        self, *keys: str, inplace: bool = False, strict: bool = True
    ) -> TensorDictBase:
        """Selects the keys of the tensordict and returns an new tensordict with only the selected keys.

        The values are not copied: in-place modifications a tensor of either
        of the original or new tensordict will result in a change in both
        tensordicts.

        Args:
            *keys (str): keys to select
            inplace (bool): if True, the tensordict is pruned in place.
                Default is :obj:`False`.
            strict (bool, optional): whether selecting a key that is not present
                will return an error or not. Default: :obj:`True`.

        Returns:
            A new tensordict with the selected keys only.

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    def exclude(self, *keys: str, inplace: bool = False) -> TensorDictBase:
        target = self if inplace else self.clone(recurse=False)
        for key in keys:
            if key in self.keys(True):
                del target[key]
        return target

    def copy_(self, tensordict: TensorDictBase) -> TensorDictBase:
        """See :obj:`TensorDictBase.update_`."""
        return self.update_(tensordict)

    def copy_at_(self, tensordict: TensorDictBase, idx: IndexType) -> TensorDictBase:
        """See :obj:`TensorDictBase.update_at_`."""
        return self.update_at_(tensordict, idx)

    def get_at(
        self, key: NestedKey, idx: IndexType, default: CompatibleType = NO_DEFAULT
    ) -> CompatibleType:
        """Get the value of a tensordict from the key `key` at the index `idx`.

        Args:
            key (str, tuple of str): key to be retrieved.
            idx (int, slice, torch.Tensor, iterable): index of the tensor.
            default (torch.Tensor): default value to return if the key is
                not present in the tensordict.

        Returns:
            indexed tensor.

        """
        key = _unravel_key_to_tuple(key)
        if not key:
            raise KeyError(_GENERIC_NESTED_ERR)
        # must be a tuple
        return self._get_at_tuple(key, idx, default)

    def _get_at_str(self, key, idx, default):
        out = self._get_str(key, default)
        if out is default:
            return out
        return out[idx]

    def _get_at_tuple(self, key, idx, default):
        out = self._get_tuple(key, default)
        if out is default:
            return out
        return out[idx]

    @abc.abstractmethod
    def share_memory_(self) -> TensorDictBase:
        """Places all the tensors in shared memory.

        The TensorDict is then locked, meaning that the only writing operations that
        can be executed must be done in-place.
        Once the tensordict is unlocked, the share_memory attribute is turned to False,
        because cross-process identity is not guaranteed anymore.

        Returns:
            self.

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    @abc.abstractmethod
    def memmap_(
        self, prefix: str | None = None, copy_existing: bool = False
    ) -> TensorDictBase:
        """Writes all tensors onto a MemmapTensor.

        Args:
            prefix (str): directory prefix where the memmap tensors will have to
                be stored.
            copy_existing (bool): If False (default), an exception will be raised if an
                entry in the tensordict is already a MemmapTensor but is not saved in
                the correct location according to prefix. If True, any MemmapTensors
                that are not in the correct location are copied to the new location.

        The TensorDict is then locked, meaning that the only writing operations that
        can be executed must be done in-place.
        Once the tensordict is unlocked, the memmap attribute is turned to False,
        because cross-process identity is not guaranteed anymore.

        Returns:
            self.

        Note:
            Serialising in this fashion might be slow with deeply nested tensordicts, so
            we do not recommend calling this method inside a training loop.
        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    def memmap_like(self, prefix: str | None = None) -> TensorDictBase:
        """Creates an empty Memory-mapped tensordict with the same content shape as the current one.

        Args:
            prefix (str): directory prefix where the memmap tensors will have to
                be stored.

        The resulting TensorDict will be locked and ``is_memmap() = True``,
        meaning that the only writing operations that can be executed must be done in-place.
        Once the tensordict is unlocked, the memmap attribute is turned to False,
        because cross-process identity is not guaranteed anymore.

        Returns:
            a new ``TensorDict`` instance with data stored as memory-mapped tensors.

        """
        if prefix is not None:
            prefix = Path(prefix)
            if not prefix.exists():
                os.makedirs(prefix, exist_ok=True)
            torch.save(
                {"batch_size": self.batch_size, "device": self.device},
                prefix / "meta.pt",
            )
        if not self.keys():
            raise Exception(
                "memmap_like() must be called when the TensorDict is (partially) "
                "populated. Set a tensor first."
            )
        tensordict = TensorDict(
            {},
            self.batch_size,
            device=self.device,
            names=self.names if self._has_names() else None,
        )
        for key, value in self.items():
            if _is_tensor_collection(value.__class__):
                if prefix is not None:
                    # ensure subdirectory exists
                    os.makedirs(prefix / key, exist_ok=True)
                    tensordict[key] = value.memmap_like(
                        prefix=prefix / key,
                    )
                    torch.save(
                        {"batch_size": value.batch_size, "device": value.device},
                        prefix / key / "meta.pt",
                    )
                else:
                    tensordict[key] = value.memmap_like()
                continue
            else:
                tensordict[key] = MemmapTensor.empty_like(
                    value,
                    filename=str(prefix / f"{key}.memmap")
                    if prefix is not None
                    else None,
                )
            if prefix is not None:
                torch.save(
                    {
                        "shape": value.shape,
                        "device": value.device,
                        "dtype": value.dtype,
                    },
                    prefix / f"{key}.meta.pt",
                )
        tensordict._is_memmap = True
        tensordict.lock_()
        return tensordict

    @abc.abstractmethod
    def detach_(self) -> TensorDictBase:
        """Detach the tensors in the tensordict in-place.

        Returns:
            self.

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    def detach(self) -> TensorDictBase:
        """Detach the tensors in the tensordict.

        Returns:
            a new tensordict with no tensor requiring gradient.

        """
        return self.apply(lambda x: x.detach())

    def to_h5(
        self,
        filename,
        **kwargs,
    ):
        """Converts a tensordict to a PersistentTensorDict with the h5 backend.

        Args:
            filename (str or path): path to the h5 file.
            device (torch.device or compatible, optional): the device where to
                expect the tensor once they are returned. Defaults to ``None``
                (on cpu by default).
            **kwargs: kwargs to be passed to :meth:`h5py.File.create_dataset`.

        Returns:
            A :class:`PersitentTensorDict` instance linked to the newly created file.

        Examples:
            >>> import tempfile
            >>> import timeit
            >>>
            >>> from tensordict import TensorDict, MemmapTensor
            >>> td = TensorDict({
            ...     "a": MemmapTensor(1_000_000),
            ...     "b": {"c": MemmapTensor(1_000_000, 3)},
            ... }, [1_000_000])
            >>>
            >>> file = tempfile.NamedTemporaryFile()
            >>> td_h5 = td.to_h5(file.name, compression="gzip", compression_opts=9)
            >>> print(td_h5)
            PersistentTensorDict(
                fields={
                    a: Tensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: PersistentTensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([1000000, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([1000000]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([1000000]),
                device=None,
                is_shared=False)


        """
        from .persistent import PersistentTensorDict

        out = PersistentTensorDict.from_dict(
            self,
            filename=filename,
            **kwargs,
        )
        if self._has_names():
            out.names = self.names
        return out

    def to_tensordict(self):
        """Returns a regular TensorDict instance from the TensorDictBase.

        Returns:
            a new TensorDict object containing the same values.

        """
        return TensorDict(
            {
                key: value.clone()
                if not _is_tensor_collection(value.__class__)
                else value.to_tensordict()
                for key, value in self.items()
            },
            device=self.device,
            batch_size=self.batch_size,
            names=self.names if self._has_names() else None,
        )

    def zero_(self) -> TensorDictBase:
        """Zeros all tensors in the tensordict in-place."""
        for key in self.keys():
            self.fill_(key, 0)
        return self

    def unbind(self, dim: int) -> tuple[TensorDictBase, ...]:
        """Returns a tuple of indexed tensordicts unbound along the indicated dimension.

        Resulting tensordicts will share the storage of the initial tensordict.

        """
        if dim < 0:
            dim = self.batch_dims + dim
        batch_size = torch.Size([s for i, s in enumerate(self.batch_size) if i != dim])
        names = None
        if self._has_names():
            names = copy(self.names)
            names = [name for i, name in enumerate(names) if i != dim]
        out = []
        unbind_self_dict = {key: tensor.unbind(dim) for key, tensor in self.items()}
        for _idx in range(self.batch_size[dim]):
            td = TensorDict(
                {key: tensor[_idx] for key, tensor in unbind_self_dict.items()},
                batch_size=batch_size,
                _run_checks=False,
                device=self.device,
                _is_memmap=False,
                _is_shared=False,
                names=names,
            )
            out.append(td)
            if self.is_shared():
                out[-1].share_memory_()
            elif self.is_memmap():
                out[-1].memmap_()
        return tuple(out)

    def chunk(self, chunks: int, dim: int = 0) -> tuple[TensorDictBase, ...]:
        """Splits a tendordict into the specified number of chunks, if possible.

        Each chunk is a view of the input tensordict.

        Args:
            chunks (int): number of chunks to return
            dim (int, optional): dimension along which to split the
                tensordict. Default is 0.

        """
        if chunks < 1:
            raise ValueError(
                f"chunks must be a strictly positive integer, got {chunks}."
            )
        indices = []
        _idx_start = 0
        if chunks > 1:
            interval = _idx_end = self.batch_size[dim] // chunks
        else:
            interval = _idx_end = self.batch_size[dim]
        for c in range(chunks):
            indices.append(slice(_idx_start, _idx_end))
            _idx_start = _idx_end
            _idx_end = _idx_end + interval if c < chunks - 2 else self.batch_size[dim]
        if dim < 0:
            dim = len(self.batch_size) + dim
        return tuple(self[(*[slice(None) for _ in range(dim)], idx)] for idx in indices)

    def clone(self, recurse: bool = True) -> TensorDictBase:
        """Clones a TensorDictBase subclass instance onto a new TensorDictBase subclass of the same type.

        To create a TensorDict instance from any other TensorDictBase subtype, call the :meth:`~.to_tensordict` method
        instead.

        Args:
            recurse (bool, optional): if True, each tensor contained in the
                TensorDict will be copied too. Default is `True`.

        .. note::
          For some TensorDictBase subtypes, such as :class:`SubTensorDict`, cloning
          recursively makes little sense (in this specific case it would involve
          copying the parent tensordict too). In those cases, :meth:`~.clone` will
          fall back onto :meth:`~.to_tensordict`.

        """
        raise NotImplementedError

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
        if func not in TD_HANDLED_FUNCTIONS or not all(
            issubclass(t, (Tensor, TensorDictBase)) for t in types
        ):
            return NotImplemented
        return TD_HANDLED_FUNCTIONS[func](*args, **kwargs)

    @abc.abstractmethod
    def to(self, dest: DeviceType | type | torch.Size, **kwargs) -> TensorDictBase:
        """Maps a TensorDictBase subclass either on a new device or to another TensorDictBase subclass (if permitted).

        Casting tensors to a new dtype is not allowed, as tensordicts are not bound to contain a single
        tensor dtype.

        Args:
            dest (device, size or TensorDictBase subclass): destination of the
                tensordict. If it is a torch.Size object, the batch_size
                will be updated provided that it is compatible with the
                stored tensors.

        Returns:
            a new tensordict. If device indicated by dest differs from
            the tensordict device, this is a no-op.

        """
        raise NotImplementedError

    def _check_new_batch_size(self, new_size: torch.Size) -> None:
        n = len(new_size)
        for key, tensor in self.items():
            if _shape(tensor)[:n] != new_size:
                raise RuntimeError(
                    f"the tensor {key} has shape {_shape(tensor)} which "
                    f"is incompatible with the new shape {new_size}."
                )

    @abc.abstractmethod
    def _change_batch_size(self, new_size: torch.Size) -> None:
        raise NotImplementedError

    def cpu(self) -> TensorDictBase:
        """Casts a tensordict to CPU."""
        return self.to("cpu")

    def cuda(self, device: int = None) -> TensorDictBase:
        """Casts a tensordict to a cuda device (if not already on it)."""
        if device is None:
            return self.to(torch.device("cuda"))
        return self.to(f"cuda:{device}")

    def _create_nested_str(self, key):
        self._set_str(key, self.select(), inplace=False, validated=True)

    def _create_nested_tuple(self, key):
        self._create_nested_str(key[0])
        if len(key) > 1:
            td = self._get_str(key[0], NO_DEFAULT)
            td._create_nested_tuple(key[1:])

    @lock_blocked
    def create_nested(self, key):
        """Creates a nested tensordict of the same shape, device and dim names as the current tensordict.

        If the value already exists, it will be overwritten by this operation.
        This operation is blocked in locked tensordicts.

        Examples:
            >>> data = TensorDict({}, [3, 4, 5])
            >>> data.create_nested("root")
            >>> data.create_nested(("some", "nested", "value"))
            >>> nested = data.get(("some", "nested", "value"))
        """
        key = _unravel_key_to_tuple(key)
        self._create_nested_tuple(key)
        return self

    @abc.abstractmethod
    def masked_fill_(self, mask: Tensor, value: float | bool) -> TensorDictBase:
        """Fills the values corresponding to the mask with the desired value.

        Args:
            mask (boolean torch.Tensor): mask of values to be filled. Shape
                must match tensordict shape.
            value: value to used to fill the tensors.

        Returns:
            self

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3, 4)},
            ...     batch_size=[3])
            >>> mask = torch.tensor([True, False, False])
            >>> _ = td.masked_fill_(mask, 1.0)
            >>> td.get("a")
            tensor([[1., 1., 1., 1.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.]])
        """
        raise NotImplementedError

    @abc.abstractmethod
    def masked_fill(self, mask: Tensor, value: float | bool) -> TensorDictBase:
        """Out-of-place version of masked_fill.

        Args:
            mask (boolean torch.Tensor): mask of values to be filled. Shape
                must match tensordict shape.
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

    def masked_select(self, mask: Tensor) -> TensorDictBase:
        """Masks all tensors of the TensorDict and return a new TensorDict instance with similar keys pointing to masked values.

        Args:
            mask (torch.Tensor): boolean mask to be used for the tensors.
                Shape must match the TensorDict batch_size.

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3, 4)},
            ...    batch_size=[3])
            >>> mask = torch.tensor([True, False, False])
            >>> td_mask = td.masked_select(mask)
            >>> td_mask.get("a")
            tensor([[0., 0., 0., 0.]])

        """
        d = {}
        mask_expand = mask
        while mask_expand.ndimension() > self.batch_dims:
            mndim = mask_expand.ndimension()
            mask_expand = mask_expand.squeeze(-1)
            if mndim == mask_expand.ndimension():  # no more squeeze
                break
        for key, value in self.items():
            d[key] = value[mask_expand]
        dim = int(mask.sum().item())
        other_dim = self.shape[mask.ndim :]
        return TensorDict(
            device=self.device, source=d, batch_size=torch.Size([dim, *other_dim])
        )

    @abc.abstractmethod
    def is_contiguous(self) -> bool:
        """Returns a boolean indicating if all the tensors are contiguous."""
        raise NotImplementedError

    @abc.abstractmethod
    def contiguous(self) -> TensorDictBase:
        """Returns a new tensordict of the same type with contiguous values (or self if values are already contiguous)."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Returns a dictionary with key-value pairs matching those of the tensordict."""
        return {
            key: value.to_dict() if _is_tensor_collection(value.__class__) else value
            for key, value in self.items()
        }

    def unsqueeze(self, dim: int) -> TensorDictBase:
        """Unsqueeze all tensors for a dimension comprised in between `-td.batch_dims` and `td.batch_dims` and returns them in a new tensordict.

        Args:
            dim (int): dimension along which to unsqueeze

        """
        if dim < 0:
            dim = self.batch_dims + dim + 1

        if (dim > self.batch_dims) or (dim < 0):
            raise RuntimeError(
                f"unsqueezing is allowed for dims comprised between "
                f"`-td.batch_dims` and `td.batch_dims` only. Got "
                f"dim={dim} with a batch size of {self.batch_size}."
            )
        return _UnsqueezedTensorDict(
            source=self,
            custom_op="unsqueeze",
            inv_op="squeeze",
            custom_op_kwargs={"dim": dim},
            inv_op_kwargs={"dim": dim},
        )

    def squeeze(self, dim: int | None = None) -> TensorDictBase:
        """Squeezes all tensors for a dimension comprised in between `-td.batch_dims+1` and `td.batch_dims-1` and returns them in a new tensordict.

        Args:
            dim (Optional[int]): dimension along which to squeeze. If dim is None, all singleton dimensions will be squeezed. dim is None by default.

        """
        if dim is None:
            size = self.size()
            if len(self.size()) == 1 or size.count(1) == 0:
                return self
            first_singleton_dim = size.index(1)

            squeezed_dict = _SqueezedTensorDict(
                source=self,
                custom_op="squeeze",
                inv_op="unsqueeze",
                custom_op_kwargs={"dim": first_singleton_dim},
                inv_op_kwargs={"dim": first_singleton_dim},
            )
            return squeezed_dict.squeeze(dim=None)

        if dim < 0:
            dim = self.batch_dims + dim

        if self.batch_dims and (dim >= self.batch_dims or dim < 0):
            raise RuntimeError(
                f"squeezing is allowed for dims comprised between 0 and "
                f"td.batch_dims only. Got dim={dim} and batch_size"
                f"={self.batch_size}."
            )

        if dim >= self.batch_dims or self.batch_size[dim] != 1:
            return self
        return _SqueezedTensorDict(
            source=self,
            custom_op="squeeze",
            inv_op="unsqueeze",
            custom_op_kwargs={"dim": dim},
            inv_op_kwargs={"dim": dim},
        )

    def reshape(
        self,
        *shape: int,
        size: list | tuple | torch.Size | None = None,
    ) -> TensorDictBase:
        """Returns a contiguous, reshaped tensor of the desired shape.

        Args:
            *shape (int): new shape of the resulting tensordict.
            size: iterable

        Returns:
            A TensorDict with reshaped keys

        """
        if len(shape) == 0 and size is not None:
            return self.reshape(*size)
        elif len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
            return self.reshape(*shape[0])
        elif not isinstance(shape, torch.Size):
            shape = torch.Size(shape)

        d = {}
        for key, item in self.items():
            d[key] = item.reshape((*shape, *item.shape[self.ndimension() :]))
        if d:
            batch_size = d[key].shape[: len(shape)]
        else:
            if any(not isinstance(i, int) or i < 0 for i in shape):
                raise RuntimeError(
                    "Implicit reshaping is not permitted with empty " "tensordicts"
                )
            batch_size = torch.Size(shape)
        return TensorDict(d, batch_size, device=self.device, _run_checks=False)

    def split(self, split_size: int | list[int], dim: int = 0) -> list[TensorDictBase]:
        """Splits each tensor in the TensorDict with the specified size in the given dimension, like `torch.split`.

        Returns a list of TensorDict with the view of split chunks of items. Nested TensorDicts will remain nested.

        The list of TensorDict maintains the original order of the tensor chunks.

        Args:
            split_size (int or List(int)): size of a single chunk or list of sizes for each chunk
            dim (int): dimension along which to split the tensor

        Returns:
            A list of TensorDict with specified size in given dimension.

        """
        batch_sizes = []
        if self.batch_dims == 0:
            raise RuntimeError("TensorDict with empty batch size is not splittable")
        if not (-self.batch_dims <= dim < self.batch_dims):
            raise IndexError(
                f"Dimension out of range (expected to be in range of [-{self.batch_dims}, {self.batch_dims - 1}], but got {dim})"
            )
        if dim < 0:
            dim += self.batch_dims
        if isinstance(split_size, int):
            rep, remainder = divmod(self.batch_size[dim], split_size)
            rep_shape = torch.Size(
                [
                    split_size if idx == dim else size
                    for (idx, size) in enumerate(self.batch_size)
                ]
            )
            batch_sizes = [rep_shape for _ in range(rep)]
            if remainder:
                batch_sizes.append(
                    torch.Size(
                        [
                            remainder if dim_idx == dim else dim_size
                            for (dim_idx, dim_size) in enumerate(self.batch_size)
                        ]
                    )
                )
        elif isinstance(split_size, list) and all(
            isinstance(element, int) for element in split_size
        ):
            if sum(split_size) != self.batch_size[dim]:
                raise RuntimeError(
                    f"Split method expects split_size to sum exactly to {self.batch_size[dim]} (tensor's size at dimension {dim}), but got split_size={split_size}"
                )
            for i in split_size:
                batch_sizes.append(
                    torch.Size(
                        [
                            i if dim_idx == dim else dim_size
                            for (dim_idx, dim_size) in enumerate(self.batch_size)
                        ]
                    )
                )
        else:
            raise TypeError(
                "split(): argument 'split_size' must be int or list of ints"
            )
        dictionaries = [{} for _ in range(len(batch_sizes))]
        for key, item in self.items():
            split_tensors = torch.split(item, split_size, dim)
            for idx, split_tensor in enumerate(split_tensors):
                dictionaries[idx][key] = split_tensor
        names = None
        if self._has_names():
            names = copy(self.names)
        return [
            TensorDict(
                dictionaries[i],
                batch_sizes[i],
                device=self.device,
                names=names,
                _run_checks=False,
                _is_shared=self.is_shared(),
                _is_memmap=self.is_memmap(),
            )
            for i in range(len(dictionaries))
        ]

    def gather(
        self, dim: int, index: Tensor, out: TensorDictBase | None = None
    ) -> TensorDictBase:
        """Gathers values along an axis specified by `dim`.

        Args:
            dim (int): the dimension along which collect the elements
            index (torch.Tensor): a long tensor which number of dimension matches
                the one of the tensordict with only one dimension differring between
                the two (the gathering dimension). Its elements refer to the
                index to be gathered along the required dimension.
            out (TensorDictBase, optional): a destination tensordict. It must
                have the same shape as the index.

        Examples:
            >>> td = TensorDict(
            ...     {"a": torch.randn(3, 4, 5),
            ...      "b": TensorDict({"c": torch.zeros(3, 4, 5)}, [3, 4, 5])},
            ...     [3, 4])
            >>> index = torch.randint(4, (3, 2))
            >>> td_gather = td.gather(dim=1, index=index)
            >>> print(td_gather)
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3, 2, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([3, 2, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([3, 2, 5]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3, 2]),
                device=None,
                is_shared=False)

        Gather keeps the dimension names.

        Examples:
            >>> td.names = ["a", "b"]
            >>> td_gather = td.gather(dim=1, index=index)
            >>> td_gather.names
            ["a", "b"]
        """
        return torch.gather(self, dim, index, out=out)

    def view(
        self,
        *shape: int,
        size: list | tuple | torch.Size | None = None,
    ) -> TensorDictBase:
        """Returns a tensordict with views of the tensors according to a new shape, compatible with the tensordict batch_size.

        Args:
            *shape (int): new shape of the resulting tensordict.
            size: iterable

        Returns:
            a new tensordict with the desired batch_size.

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3,4,5),
            ...    'b': torch.zeros(3,4,10,1)}, batch_size=torch.Size([3, 4]))
            >>> td_view = td.view(12)
            >>> print(td_view.get("a").shape)  # torch.Size([12, 5])
            >>> print(td_view.get("b").shape)  # torch.Size([12, 10, 1])
            >>> td_view = td.view(-1, 4, 3)
            >>> print(td_view.get("a").shape)  # torch.Size([1, 4, 3, 5])
            >>> print(td_view.get("b").shape)  # torch.Size([1, 4, 3, 10, 1])

        """
        if len(shape) == 0 and size is not None:
            return self.view(*size)
        elif len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
            return self.view(*shape[0])
        elif not isinstance(shape, torch.Size):
            shape = infer_size_impl(shape, self.numel())
            shape = torch.Size(shape)
        if shape == self.shape:
            return self
        return _ViewedTensorDict(
            source=self,
            custom_op="view",
            inv_op="view",
            custom_op_kwargs={"size": shape},
            inv_op_kwargs={"size": self.batch_size},
        )

    def transpose(self, dim0, dim1):
        """Returns a tensordit that is a transposed version of input. The given dimensions ``dim0`` and ``dim1`` are swapped.

        In-place or out-place modifications of the transposed tensordict will
        impact the original tensordict too as the memory is shared and the operations
        are mapped back on the original tensordict.

        Examples:
            >>> tensordict = TensorDict({"a": torch.randn(3, 4, 5)}, [3, 4])
            >>> tensordict_transpose = tensordict.transpose(0, 1)
            >>> print(tensordict_transpose.shape)
            torch.Size([4, 3])
            >>> tensordict_transpose.set("b",, torch.randn(4, 3))
            >>> print(tensordict.get("b").shape)
            torch.Size([3, 4])
        """
        if dim0 < 0:
            dim0 = self.ndim + dim0
        if dim1 < 0:
            dim1 = self.ndim + dim1
        if any((dim0 < 0, dim1 < 0)):
            raise ValueError(
                "The provided dimensions are incompatible with the tensordict batch-size."
            )
        if dim0 == dim1:
            return self
        return _TransposedTensorDict(
            source=self,
            custom_op="transpose",
            inv_op="transpose",
            custom_op_kwargs={"dim0": dim0, "dim1": dim1},
            inv_op_kwargs={"dim0": dim0, "dim1": dim1},
        )

    def permute(
        self,
        *dims_list: int,
        dims: list[int] | None = None,
    ) -> TensorDictBase:
        """Returns a view of a tensordict with the batch dimensions permuted according to dims.

        Args:
            *dims_list (int): the new ordering of the batch dims of the tensordict. Alternatively,
                a single iterable of integers can be provided.
            dims (list of int): alternative way of calling permute(...).

        Returns:
            a new tensordict with the batch dimensions in the desired order.

        Examples:
            >>> tensordict = TensorDict({"a": torch.randn(3, 4, 5)}, [3, 4])
            >>> print(tensordict.permute([1, 0]))
            PermutedTensorDict(
                source=TensorDict(
                    fields={
                        a: Tensor(torch.Size([3, 4, 5]), dtype=torch.float32)},
                    batch_size=torch.Size([3, 4]),
                    device=cpu,
                    is_shared=False),
                op=permute(dims=[1, 0]))
            >>> print(tensordict.permute(1, 0))
            PermutedTensorDict(
                source=TensorDict(
                    fields={
                        a: Tensor(torch.Size([3, 4, 5]), dtype=torch.float32)},
                    batch_size=torch.Size([3, 4]),
                    device=cpu,
                    is_shared=False),
                op=permute(dims=[1, 0]))
            >>> print(tensordict.permute(dims=[1, 0]))
            PermutedTensorDict(
                source=TensorDict(
                    fields={
                        a: Tensor(torch.Size([3, 4, 5]), dtype=torch.float32)},
                    batch_size=torch.Size([3, 4]),
                    device=cpu,
                    is_shared=False),
                op=permute(dims=[1, 0]))
        """
        if len(dims_list) == 0:
            dims_list = dims
        elif len(dims_list) == 1 and not isinstance(dims_list[0], int):
            dims_list = dims_list[0]
        if len(dims_list) != len(self.shape):
            raise RuntimeError(
                f"number of dims don't match in permute (got {len(dims_list)}, expected {len(self.shape)}"
            )

        if not len(dims_list) and not self.batch_dims:
            return self
        if np.array_equal(dims_list, range(self.batch_dims)):
            return self
        min_dim, max_dim = -self.batch_dims, self.batch_dims - 1
        seen = [False for dim in range(max_dim + 1)]
        for idx in dims_list:
            if idx < min_dim or idx > max_dim:
                raise IndexError(
                    f"dimension out of range (expected to be in range of [{min_dim}, {max_dim}], but got {idx})"
                )
            if seen[idx]:
                raise RuntimeError("repeated dim in permute")
            seen[idx] = True

        return _PermutedTensorDict(
            source=self,
            custom_op="permute",
            inv_op="permute",
            custom_op_kwargs={"dims": dims_list},
            inv_op_kwargs={"dims": dims_list},
        )

    def __repr__(self) -> str:
        fields = _td_fields(self)
        field_str = indent(f"fields={{{fields}}}", 4 * " ")
        batch_size_str = indent(f"batch_size={self.batch_size}", 4 * " ")
        device_str = indent(f"device={self.device}", 4 * " ")
        is_shared_str = indent(f"is_shared={self.is_shared()}", 4 * " ")
        string = ",\n".join([field_str, batch_size_str, device_str, is_shared_str])
        return f"{type(self).__name__}(\n{string})"

    def all(self, dim: int = None) -> bool | TensorDictBase:
        """Checks if all values are True/non-null in the tensordict.

        Args:
            dim (int, optional): if None, returns a boolean indicating
                whether all tensors return `tensor.all() == True`
                If integer, all is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.

        """
        if dim is not None and (dim >= self.batch_dims or dim < -self.batch_dims):
            raise RuntimeError(
                "dim must be greater than or equal to -tensordict.batch_dims and "
                "smaller than tensordict.batch_dims"
            )
        if dim is not None:
            if dim < 0:
                dim = self.batch_dims + dim

            names = None
            if self._has_names():
                names = copy(self.names)
                names = [name for i, name in enumerate(names) if i != dim]

            return TensorDict(
                source={key: value.all(dim=dim) for key, value in self.items()},
                batch_size=[b for i, b in enumerate(self.batch_size) if i != dim],
                device=self.device,
                names=names,
            )
        return all(value.all() for value in self.values())

    def any(self, dim: int = None) -> bool | TensorDictBase:
        """Checks if any value is True/non-null in the tensordict.

        Args:
            dim (int, optional): if None, returns a boolean indicating
                whether all tensors return `tensor.any() == True`.
                If integer, all is called upon the dimension specified if
                and only if this dimension is compatible with
                the tensordict shape.

        """
        if dim is not None and (dim >= self.batch_dims or dim < -self.batch_dims):
            raise RuntimeError(
                "dim must be greater than or equal to -tensordict.batch_dims and "
                "smaller than tensordict.batch_dims"
            )
        if dim is not None:
            if dim < 0:
                dim = self.batch_dims + dim

            names = None
            if self._has_names():
                names = copy(self.names)
                names = [name for i, name in enumerate(names) if i != dim]

            return TensorDict(
                source={key: value.any(dim=dim) for key, value in self.items()},
                batch_size=[b for i, b in enumerate(self.batch_size) if i != dim],
                device=self.device,
                names=names,
            )
        return any([value.any() for value in self.values()])

    def get_sub_tensordict(self, idx: IndexType) -> TensorDictBase:
        """Returns a SubTensorDict with the desired index."""
        return SubTensorDict(source=self, idx=idx)

    def __iter__(self) -> Generator:
        if not self.batch_dims:
            raise StopIteration
        length = self.batch_size[0]
        for i in range(length):
            yield self[i]

    @cache  # noqa: B019
    def flatten_keys(
        self, separator: str = ".", inplace: bool = False
    ) -> TensorDictBase:
        to_flatten = []
        existing_keys = self.keys(include_nested=True)
        for key, value in self.items():
            key_split = tuple(key.split(separator))
            if isinstance(value, TensorDictBase):
                to_flatten.append(key)
            elif (
                separator in key
                and key_split in existing_keys
                and not _is_tensor_collection(self.entry_class(key_split))
            ):
                raise KeyError(
                    f"Flattening keys in tensordict collides with existing key '{key}'"
                )

        if inplace:
            for key in to_flatten:
                inner_tensordict = self.get(key).flatten_keys(
                    separator=separator, inplace=inplace
                )
                for inner_key, inner_item in inner_tensordict.items():
                    self.set(separator.join([key, inner_key]), inner_item)
            for key in to_flatten:
                del self[key]
            return self
        else:
            tensordict_out = TensorDict(
                {},
                batch_size=self.batch_size,
                device=self.device,
                _run_checks=False,
                _is_shared=self.is_shared(),
                _is_memmap=self.is_memmap(),
                names=self.names,
            )
            for key, value in self.items():
                if key in to_flatten:
                    inner_tensordict = self.get(key).flatten_keys(
                        separator=separator, inplace=inplace
                    )
                    for inner_key, inner_item in inner_tensordict.items():
                        tensordict_out.set(separator.join([key, inner_key]), inner_item)
                else:
                    tensordict_out.set(key, value)
            return tensordict_out

    @cache  # noqa: B019
    def unflatten_keys(
        self, separator: str = ".", inplace: bool = False
    ) -> TensorDictBase:
        to_unflatten = defaultdict(list)
        for key in self.keys():
            if separator in key[1:-1]:
                split_key = key.split(separator)
                to_unflatten[split_key[0]].append((key, separator.join(split_key[1:])))

        if not inplace:
            out = TensorDict(
                {
                    key: value
                    for key, value in self.items()
                    if separator not in key[1:-1]
                },
                batch_size=self.batch_size,
                device=self.device,
                _run_checks=False,
                _is_shared=self.is_shared(),
                _is_memmap=self.is_memmap(),
                names=self.names,
            )
        else:
            out = self

        keys = set(out.keys())
        for key, list_of_keys in to_unflatten.items():
            if key in keys:
                raise KeyError(
                    "Unflattening key(s) in tensordict will override existing unflattened key"
                )

            tensordict = TensorDict(
                {},
                batch_size=self.batch_size,
                device=self.device,
                names=self.names,
            )
            if key in self.keys():
                tensordict.update(self[key])
            for old_key, new_key in list_of_keys:
                value = self.get(old_key)
                tensordict[new_key] = value
                if inplace:
                    del self[old_key]
            out._set_str(
                key,
                tensordict.unflatten_keys(separator=separator),
                inplace=False,
                validated=True,
            )
        return out

    def __len__(self) -> int:
        """Returns the length of first dimension, if there is, otherwise 0."""
        return self.shape[0] if self.batch_dims else 0

    def __contains__(self, key: NestedKey) -> bool:
        # by default a Mapping will implement __contains__ by calling __getitem__ and
        # returning False if a KeyError is raised, True otherwise. TensorDict has a
        # complex __getitem__ method since we support more than just retrieval of values
        # by key, and so this can be quite inefficient, particularly if values are
        # evaluated lazily on access. Hence we don't support use of __contains__ and
        # direct the user to use TensorDict.keys() instead
        raise NotImplementedError(
            "TensorDict does not support membership checks with the `in` keyword. If "
            "you want to check if a particular key is in your TensorDict, please use "
            "`key in tensordict.keys()` instead."
        )

    def _get_names_idx(self, idx):
        if not self._has_names():
            names = None
        else:

            def is_boolean(idx):
                if isinstance(idx, tuple) and len(idx) == 1:
                    return is_boolean(idx[0])
                if hasattr(idx, "dtype") and idx.dtype is torch.bool:
                    return idx.ndim
                return None

            num_boolean_dim = is_boolean(idx)
            names = self.names
            if num_boolean_dim:
                names = [None] + names[num_boolean_dim:]
            else:

                # def is_int(subidx):
                #     if isinstance(subidx, Number):
                #         return True
                #     if isinstance(subidx, Tensor) and len(subidx.shape) == 0:
                #         return True
                #     return False

                if not isinstance(idx, tuple):
                    idx = (idx,)
                if len(idx) < self.ndim:
                    idx = (*idx, Ellipsis)
                idx_names = convert_ellipsis_to_idx(idx, self.batch_size)
                # this will convert a [None, :, :, 0, None, 0] in [None, 0, 1, None, 3]
                count = 0
                idx_to_take = []
                for _idx in idx_names:
                    if _idx is None:
                        idx_to_take.append(None)
                    elif _is_number(_idx):
                        count += 1
                    else:
                        idx_to_take.append(count)
                        count += 1
                names = [names[i] if i is not None else None for i in idx_to_take]
        return names

    def _index_tensordict(self, index: IndexType) -> TensorDictBase:
        batch_size = self.batch_size
        if (
            not batch_size
            and index is not None
            and (not isinstance(index, tuple) or any(idx is not None for idx in index))
        ):
            raise RuntimeError(
                f"indexing a tensordict with td.batch_dims==0 is not permitted. Got index {index}."
            )
        names = self._get_names_idx(index)
        return TensorDict(
            source={key: _get_item(item, index) for key, item in self.items()},
            batch_size=_getitem_batch_size(batch_size, index),
            device=self.device,
            names=names,
            _run_checks=False,
            _is_shared=self.is_shared(),
            _is_memmap=self.is_memmap(),
        )

    def __getitem__(self, index: IndexType) -> TensorDictBase:
        """Indexes all tensors according to the provided index.

        Returns a new tensordict where the values share the storage of the
        original tensors (even when the index is a torch.Tensor).
        Any in-place modification to the resulting tensordict will
        impact the parent tensordict too.

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3,4,5)},
            ...     batch_size=torch.Size([3, 4]))
            >>> subtd = td[torch.zeros(1, dtype=torch.long)]
            >>> assert subtd.shape == torch.Size([1,4])
            >>> subtd.set("a", torch.ones(1,4,5))
            >>> print(td.get("a"))  # first row is full of 1
            >>> # Warning: this will not work as expected
            >>> subtd.get("a")[:] = 2.0
            >>> print(td.get("a"))  # values have not changed

        """
        istuple = isinstance(index, tuple)
        if istuple or isinstance(index, str):
            idx_unravel = _unravel_key_to_tuple(index)
            if idx_unravel:
                return self._get_tuple(idx_unravel, NO_DEFAULT)
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
        if all(isinstance(idx, slice) and idx == slice(None) for idx in index):
            return self

        return self._index_tensordict(index)

    __getitems__ = __getitem__

    def __setitem__(
        self,
        index: IndexType,
        value: TensorDictBase | dict | numbers.Number | CompatibleType,
    ) -> None:

        istuple = isinstance(index, tuple)
        if istuple or isinstance(index, str):
            # try:
            index_unravel = _unravel_key_to_tuple(index)
            if index_unravel:
                self._set_tuple(
                    index_unravel,
                    value,
                    inplace=BEST_ATTEMPT_INPLACE
                    if isinstance(self, SubTensorDict)
                    else False,
                    validated=False,
                )
                return

            # if any(isinstance(sub_index, (list, range)) for sub_index in index):
            #     index = tuple(
            #         torch.tensor(sub_index, device=self.device)
            #         if isinstance(sub_index, (list, range))
            #         else sub_index
            #         for sub_index in index
            #     )

        if index is Ellipsis or (isinstance(index, tuple) and Ellipsis in index):
            index = convert_ellipsis_to_idx(index, self.batch_size)
        # elif isinstance(index, (list, range)):
        #     index = torch.tensor(index, device=self.device)

        if isinstance(value, (TensorDictBase, dict)):
            indexed_bs = _getitem_batch_size(self.batch_size, index)
            if isinstance(value, dict):
                value = TensorDict(
                    value, batch_size=indexed_bs, device=self.device, _run_checks=False
                )
            if value.batch_size != indexed_bs:
                # try to expand
                try:
                    value = value.expand(indexed_bs)
                except RuntimeError as err:
                    raise RuntimeError(
                        f"indexed destination TensorDict batch size is {indexed_bs} "
                        f"(batch_size = {self.batch_size}, index={index}), "
                        f"which differs from the source batch size {value.batch_size}"
                    ) from err

            keys = set(self.keys())
            if any(key not in keys for key in value.keys()):
                subtd = self.get_sub_tensordict(index)
            for key, item in value.items():
                if key in keys:
                    self._set_at_str(key, item, index, validated=False)
                else:
                    subtd.set(key, item)
        else:
            for key in self.keys():
                self.set_at_(key, value, index)

    def __delitem__(self, index: IndexType) -> TensorDictBase:
        # if isinstance(index, str):
        return self.del_(index)
        # raise IndexError(f"Index has to a string but received {index}.")

    @abc.abstractmethod
    def rename_key_(
        self, old_key: str, new_key: str, safe: bool = False
    ) -> TensorDictBase:
        """Renames a key with a new string.

        Args:
            old_key (str): key to be renamed
            new_key (str): new name
            safe (bool, optional): if True, an error is thrown when the new
                key is already present in the TensorDict.

        Returns:
            self

        """
        raise NotImplementedError

    def fill_(self, key: NestedKey, value: float | bool) -> TensorDictBase:
        """Fills a tensor pointed by the key with the a given value.

        Args:
            key (str): key to be remaned
            value (Number, bool): value to use for the filling

        Returns:
            self

        """
        key = _unravel_key_to_tuple(key)
        data = self._get_tuple(key, NO_DEFAULT)
        if _is_tensor_collection(data.__class__):
            data.apply_(lambda x: x.fill_(value))
            # self._set(key, tensordict, inplace=True)
        else:
            data = data.fill_(value)
            self._set_tuple(key, data, inplace=True, validated=True)
        return self

    def empty(self, recurse=False) -> TensorDictBase:
        """Returns a new, empty tensordict with the same device and batch size.

        Args:
            recurse (bool, optional): if ``True``, the entire structure of the TensorDict
                will be rerproduced without content. Otherwise, only the root
                will be duplicated. Defaults to ``False``.

        """
        if not recurse:
            return self.select()
        return self.exclude(*self.keys(True, True))

    def is_empty(self) -> bool:
        for _ in self.keys():
            return False
        return True

    def setdefault(
        self, key: NestedKey, default: CompatibleType, inplace: bool = False
    ) -> CompatibleType:
        """Insert key with a value of default if key is not in the dictionary.

        Return the value for key if key is in the dictionary, else default.

        Args:
            key (str): the name of the value.
            default (torch.Tensor): value to be stored in the tensordict if the key is
                not already present.

        Returns:
            The value of key in the tensordict. Will be default if the key was not
            previously set.

        """
        if key not in self.keys(include_nested=isinstance(key, tuple)):
            self.set(key, default, inplace=inplace)
        return self.get(key)

    @property
    def is_locked(self) -> bool:
        return self._is_locked

    @is_locked.setter
    def is_locked(self, value: bool) -> None:
        if value:
            self.lock_()
        else:
            self.unlock_()

    def _lock_propagate(self, lock_ids=None):
        """Registers the parent tensordict that handles the lock."""
        self._is_locked = True
        is_root = lock_ids is None
        if is_root:
            lock_ids = set()
        self._lock_id = self._lock_id.union(lock_ids)
        lock_ids = lock_ids.union({id(self)})
        _locked_tensordicts = []
        for key in self.keys():
            if _is_tensor_collection(self.entry_class(key)):
                dest = self.get(key)
                dest._lock_propagate(lock_ids)
                _locked_tensordicts.append(dest)
        if is_root:
            self._locked_tensordicts = _locked_tensordicts
        else:
            self._locked_tensordicts += _locked_tensordicts

    @as_decorator("is_locked")
    def lock_(self) -> TensorDictBase:
        if self.is_locked:
            return self
        self._lock_propagate()
        return self

    lock = _renamed_inplace_method(lock_)

    def _remove_lock(self, lock_id):
        self._lock_id = self._lock_id - {lock_id}
        if self._locked_tensordicts:
            for td in self._locked_tensordicts:
                td._remove_lock(lock_id)

    @erase_cache
    def _propagate_unlock(self, lock_ids=None):
        if lock_ids is not None:
            self._lock_id.difference_update(lock_ids)
        else:
            lock_ids = set()
        self._is_locked = False

        unlocked_tds = [self]
        lock_ids.add(id(self))
        for key in self.keys():
            if _is_tensor_collection(self.entry_class(key)):
                dest = self.get(key)
                unlocked_tds.extend(dest._propagate_unlock(lock_ids))
        self._locked_tensordicts = []

        self._is_shared = False
        self._is_memmap = False
        return unlocked_tds

    @as_decorator("is_locked")
    def unlock_(self) -> TensorDictBase:
        unlock_tds = self._propagate_unlock()
        for td in unlock_tds:
            if len(td._lock_id):
                self.lock_()
                raise RuntimeError(
                    "Cannot unlock a tensordict that is part of a locked graph. "
                    "Unlock the root tensordict first. If the tensordict is part of multiple graphs, "
                    "group the graphs under a common tensordict an unlock this root. "
                )
        return self

    unlock = _renamed_inplace_method(unlock_)

    def __del__(self):
        for td in self._locked_tensordicts:
            td._remove_lock(id(self))

    def is_floating_point(self):
        for item in self.values(include_nested=True, leaves_only=True):
            if not item.is_floating_point():
                return False
        else:
            return True

    def double(self):
        r"""Casts all tensors to ``torch.bool``."""
        return self.apply(lambda x: x.double())

    def float(self):
        r"""Casts all tensors to ``torch.float``."""
        return self.apply(lambda x: x.float())

    def int(self):
        r"""Casts all tensors to ``torch.int``."""
        return self.apply(lambda x: x.int())

    def bool(self):
        r"""Casts all tensors to ``torch.bool``."""
        return self.apply(lambda x: x.bool())

    def half(self):
        r"""Casts all tensors to ``torch.half``."""
        return self.apply(lambda x: x.half())

    def bfloat16(self):
        r"""Casts all tensors to ``torch.bfloat16``."""
        return self.apply(lambda x: x.bfloat16())

    def type(self, dst_type):
        r"""Casts all tensors to :attr:`dst_type`.

        Args:
            dst_type (type or string): the desired type

        """
        return self.apply(lambda x: x.type(dst_type))


_ACCEPTED_CLASSES = [
    Tensor,
    MemmapTensor,
    TensorDictBase,
]
if _has_torchrec:
    _ACCEPTED_CLASSES += [KeyedJaggedTensor]
_ACCEPTED_CLASSES = tuple(_ACCEPTED_CLASSES)


class TensorDict(TensorDictBase):
    """A batched dictionary of tensors.

    TensorDict is a tensor container where all tensors are stored in a
    key-value pair fashion and where each element shares at least the
    following features:
    - memory location (shared, memory-mapped array, ...);
    - batch size (i.e. n^th first dimensions).

    Additionally, if the tensordict has a specified device, then each element
    must share that device.

    TensorDict instances support many regular tensor operations as long as
    they are dtype-independent (as a TensorDict instance can contain tensors
    of many different dtypes). Those operations include (but are not limited
    to):

    - operations on shape: when a shape operation is called (indexing,
      reshape, view, expand, transpose, permute,
      unsqueeze, squeeze, masking etc), the operations is done as if it
      was done on a tensor of the same shape as the batch size then
      expended to the right, e.g.:

        >>> td = TensorDict({'a': torch.zeros(3,4,5)}, batch_size=[3, 4])
        >>> # returns a TensorDict of batch size [3, 4, 1]
        >>> td_unsqueeze = td.unsqueeze(-1)
        >>> # returns a TensorDict of batch size [12]
        >>> td_view = td.view(-1)
        >>> # returns a tensor of batch size [12, 4]
        >>> a_view = td.view(-1).get("a")

    - casting operations: a TensorDict can be cast on a different device
      or another TensorDict type using

        >>> td_cpu = td.to("cpu")
        >>> td_savec = td.to(SavedTensorDict)  # TensorDict saved on disk
        >>> dictionary = td.to_dict()

      A call of the `.to()` method with a dtype will return an error.

    - Cloning, contiguous

    - Reading: `td.get(key)`, `td.get_at(key, index)`

    - Content modification: :obj:`td.set(key, value)`, :obj:`td.set_(key, value)`,
      :obj:`td.update(td_or_dict)`, :obj:`td.update_(td_or_dict)`, :obj:`td.fill_(key,
      value)`, :obj:`td.rename_key_(old_name, new_name)`, etc.

    - Operations on multiple tensordicts: `torch.cat(tensordict_list, dim)`,
      `torch.stack(tensordict_list, dim)`, `td1 == td2` etc.

    Args:
        source (TensorDict or dictionary): a data source. If empty, the
            tensordict can be populated subsequently.
        batch_size (iterable of int, optional): a batch size for the
            tensordict. The batch size is immutable and can only be modified
            by calling operations that create a new TensorDict. Unless the
            source is another TensorDict, the batch_size argument must be
            provided as it won't be inferred from the data.
        device (torch.device or compatible type, optional): a device for the
            TensorDict.
        names (lsit of str, optional): the names of the dimensions of the
            tensordict. If provided, its length must match the one of the
            ``batch_size``. Defaults to ``None`` (no dimension name, or ``None``
            for every dimension).

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> source = {'random': torch.randn(3, 4),
        ...     'zeros': torch.zeros(3, 4, 5)}
        >>> batch_size = [3]
        >>> td = TensorDict(source, batch_size)
        >>> print(td.shape)  # equivalent to td.batch_size
        torch.Size([3])
        >>> td_unqueeze = td.unsqueeze(-1)
        >>> print(td_unqueeze.get("zeros").shape)
        torch.Size([3, 1, 4, 5])
        >>> print(td_unqueeze[0].shape)
        torch.Size([1])
        >>> print(td_unqueeze.view(-1).shape)
        torch.Size([3])
        >>> print((td.clone()==td).all())
        True

    """

    __slots__ = (
        "_tensordict",
        "_batch_size",
        "_is_shared",
        "_is_memmap",
        "_device",
        "_is_locked",
        "_td_dim_names",
        "_lock_id",
        "_locked_tensordicts",
        "_cache",
        "_last_op",
        "__last_op_queue",
    )

    def __new__(cls, *args: Any, **kwargs: Any) -> TensorDict:
        cls._is_shared = False
        cls._is_memmap = False
        cls._td_dim_names = None
        return super().__new__(cls, *args, _safe=True, _lazy=False, **kwargs)

    def __init__(
        self,
        source: TensorDictBase | dict[str, CompatibleType],
        batch_size: Sequence[int] | torch.Size | int | None = None,
        device: DeviceType | None = None,
        names: Sequence[str] | None = None,
        _run_checks: bool = True,
        _is_shared: bool | None = False,
        _is_memmap: bool | None = False,
    ) -> None:
        self._lock_id = set()
        self._locked_tensordicts = []

        self._is_shared = _is_shared
        self._is_memmap = _is_memmap
        if device is not None and isinstance(device, (int, str)):
            device = torch.device(device)
        self._device = device

        if not _run_checks:
            _tensordict: dict = _StringOnlyDict()
            self._batch_size = batch_size
            for key, value in source.items():
                if isinstance(value, dict):
                    value = TensorDict(
                        value,
                        batch_size=self._batch_size,
                        device=self._device,
                        _run_checks=_run_checks,
                        _is_shared=_is_shared,
                        _is_memmap=_is_memmap,
                    )
                _tensordict[key] = value
            self._tensordict = _tensordict
            self._td_dim_names = names
        else:
            self._tensordict = _StringOnlyDict()
            if not isinstance(source, (TensorDictBase, dict)):
                raise ValueError(
                    "A TensorDict source is expected to be a TensorDictBase "
                    f"sub-type or a dictionary, found type(source)={type(source)}."
                )
            self._batch_size = self._parse_batch_size(source, batch_size)
            self.names = names

            if source is not None:
                for key, value in source.items():
                    self.set(key, value)

    @classmethod
    def from_dict(cls, input_dict, batch_size=None, device=None, batch_dims=None):
        """Returns a TensorDict created from a dictionary or another :class:`TensorDict`.

        If ``batch_size`` is not specified, returns the maximum batch size possible.

        This function works on nested dictionaries too, or can be used to determine the
        batch-size of a nested tensordict.

        Args:
            input_dict (dictionary, optional): a dictionary to use as a data source
                (nested keys compatible).
            batch_size (iterable of int, optional): a batch size for the tensordict.
            device (torch.device or compatible type, optional): a device for the TensorDict.
            batch_dims (int, optional): the ``batch_dims`` (ie number of leading dimensions
                to be considered for ``batch_size``). Exclusinve with ``batch_size``.
                Note that this is the __maximum__ number of batch dims of the tensordict,
                a smaller number is tolerated.

        Examples:
            >>> input_dict = {"a": torch.randn(3, 4), "b": torch.randn(3)}
            >>> print(TensorDict.from_dict(input_dict))
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
            >>> print(TensorDict.from_dict(input_dict))
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
            >>> # we can also use this to work out the batch sie of a tensordict
            >>> input_td = TensorDict({"a": torch.randn(3), "b": {"c": torch.randn(3, 4)}}, [])
            >>> print(TensorDict.from_dict(input_td))
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
        if batch_dims is not None and batch_size is not None:
            raise ValueError(
                "Cannot pass both batch_size and batch_dims to `from_dict`."
            )

        batch_size_set = [] if batch_size is None else batch_size
        for key, value in list(input_dict.items()):
            if isinstance(value, (dict,)):
                # we don't know if another tensor of smaller size is coming
                # so we can't be sure that the batch-size will still be valid later
                input_dict[key] = TensorDict.from_dict(
                    value, batch_size=[], device=device, batch_dims=None
                )
        # _run_checks=False breaks because a tensor may have the same batch-size as the tensordict
        out = cls(
            input_dict,
            batch_size=batch_size_set,
            device=device,
        )
        if batch_size is None:
            _set_max_batch_size(out, batch_dims)
        else:
            out.batch_size = batch_size
        return out

    @staticmethod
    def _parse_batch_size(
        source: TensorDictBase | dict,
        batch_size: Sequence[int] | torch.Size | int | None = None,
    ) -> torch.Size:
        try:
            return torch.Size(batch_size)
        except Exception as err:
            if isinstance(batch_size, Number):
                return torch.Size([batch_size])
            elif isinstance(source, TensorDictBase):
                return source.batch_size
            raise ValueError(
                "batch size was not specified when creating the TensorDict "
                "instance and it could not be retrieved from source."
            ) from err

    @property
    def batch_dims(self) -> int:
        return len(self.batch_size)

    @batch_dims.setter
    def batch_dims(self, value: int) -> None:
        raise RuntimeError(
            f"Setting batch dims on {self.__class__.__name__} instances is "
            f"not allowed."
        )

    def _has_names(self):
        return self._td_dim_names is not None

    def _erase_names(self):
        self._td_dim_names = None

    @property
    def names(self):
        names = self._td_dim_names
        if names is None:
            return [None for _ in range(self.batch_dims)]
        return names

    @names.setter
    def names(self, value):
        # we don't run checks on types for efficiency purposes
        if value is None:
            self._erase_names()
            return
        num_none = sum(v is None for v in value)
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

    def _rename_subtds(self, names):
        if names is None:
            for item in self._tensordict.values():
                if _is_tensor_collection(item.__class__):
                    item._erase_names()
            return
        for item in self._tensordict.values():
            if _is_tensor_collection(item.__class__):
                item_names = item.names
                td_names = list(names) + item_names[len(names) :]
                item.rename_(*td_names)

    @property
    def device(self) -> torch.device | None:
        """Device of the tensordict.

        Returns `None` if device hasn't been provided in the constructor or set via `tensordict.to(device)`.

        """
        return self._device

    @device.setter
    def device(self, value: DeviceType) -> None:
        raise RuntimeError(
            "device cannot be set using tensordict.device = device, "
            "because device cannot be updated in-place. To update device, use "
            "tensordict.to(new_device), which will return a new tensordict "
            "on the new device."
        )

    @property
    def batch_size(self) -> torch.Size:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_size: torch.Size) -> None:
        self._batch_size_setter(new_size)

    def _change_batch_size(self, new_size: torch.Size) -> None:
        if not hasattr(self, "_orig_batch_size"):
            self._orig_batch_size = self.batch_size
        elif self._orig_batch_size == new_size:
            del self._orig_batch_size
        self._batch_size = new_size

    # Checks
    def _check_is_shared(self) -> bool:
        share_list = [_is_shared(value) for value in self.values()]
        if any(share_list) and not all(share_list):
            shared_str = ", ".join(
                [f"{key}: {_is_shared(value)}" for key, value in self.items()]
            )
            raise RuntimeError(
                f"tensors must be either all shared or not, but mixed "
                f"features is not allowed. "
                f"Found: {shared_str}"
            )
        return all(share_list) and len(share_list) > 0

    def _check_is_memmap(self) -> bool:
        memmap_list = [is_memmap(self.entry_class(key)) for key in self.keys()]
        if any(memmap_list) and not all(memmap_list):
            memmap_str = ", ".join(
                [f"{key}: {is_memmap(self.entry_class(key))}" for key in self.keys()]
            )
            raise RuntimeError(
                f"tensors must be either all MemmapTensor or not, but mixed "
                f"features is not allowed. "
                f"Found: {memmap_str}"
            )
        return all(memmap_list) and len(memmap_list) > 0

    def _check_device(self) -> None:
        devices = {value.device for value in self.values()}
        if self.device is not None and len(devices) >= 1 and devices != {self.device}:
            raise RuntimeError(
                f"TensorDict.device is {self._device}, but elements have "
                f"device values {devices}. If TensorDict.device is set then "
                "all elements must share that device."
            )

    # def _index_tensordict(self, idx: IndexType) -> TensorDictBase:
    #     names = self._get_names_idx(idx)
    #     self_copy = copy(self)
    #     # self_copy = self.clone(False)
    #     self_copy._tensordict = {
    #         key: _get_item(item, idx) for key, item in self.items()
    #     }
    #     self_copy._batch_size = _getitem_batch_size(self_copy.batch_size, idx)
    #     self_copy._device = self.device
    #     self_copy.names = names
    #     return self_copy

    def pin_memory(self) -> TensorDictBase:
        def pin_mem(tensor):
            return tensor.pin_memory()

        return self.apply(pin_mem)

    def expand(self, *shape: int) -> TensorDictBase:
        """Expands every tensor with `(*shape, *tensor.shape)` and returns the same tensordict with new tensors with expanded shapes.

        Supports iterables to specify the shape.

        """
        d = {}
        tensordict_dims = self.batch_dims

        if len(shape) == 1 and isinstance(shape[0], Sequence):
            shape = tuple(shape[0])

        # new shape dim check
        if len(shape) < len(self.shape):
            raise RuntimeError(
                "the number of sizes provided ({shape_dim}) must be greater or equal to the number of "
                "dimensions in the TensorDict ({tensordict_dim})".format(
                    shape_dim=len(shape), tensordict_dim=tensordict_dims
                )
            )

        # new shape compatability check
        for old_dim, new_dim in zip(self.batch_size, shape[-tensordict_dims:]):
            if old_dim != 1 and new_dim != old_dim:
                raise RuntimeError(
                    "Incompatible expanded shape: The expanded shape length at non-singleton dimension should be same "
                    "as the original length. target_shape = {new_shape}, existing_shape = {old_shape}".format(
                        new_shape=shape, old_shape=self.batch_size
                    )
                )

        for key, value in self.items():
            tensor_dims = len(value.shape)
            last_n_dims = tensor_dims - tensordict_dims
            if last_n_dims > 0:
                d[key] = value.expand(*shape, *value.shape[-last_n_dims:])
            else:
                d[key] = value.expand(*shape)
        out = TensorDict(
            source=d,
            batch_size=torch.Size(shape),
            device=self.device,
            _run_checks=False,
        )
        if self._td_dim_names is not None:
            out.refine_names(..., *self.names)
        return out

    def _set_str(
        self,
        key: NestedKey,
        value: dict[str, CompatibleType] | CompatibleType,
        *,
        inplace: bool,
        validated: bool,
    ) -> TensorDictBase:
        best_attempt = inplace is BEST_ATTEMPT_INPLACE
        inplace = self._convert_inplace(inplace, key)
        if not validated:
            value = self._validate_value(value, check_shape=True)
        if not inplace:
            if self.is_locked:
                raise RuntimeError(TensorDictBase.LOCK_ERROR)
            self._tensordict[key] = value
        else:
            try:
                dest = self._get_str(key, default=NO_DEFAULT)
                if best_attempt and _is_tensor_collection(dest.__class__):
                    dest.update(value, inplace=True)
                else:
                    dest.copy_(value)
            except KeyError as err:
                raise err
            except Exception as err:
                raise ValueError(
                    f"Failed to update '{key}' in tensordict {self}"
                ) from err
        return self

    def _set_tuple(
        self,
        key: NestedKey,
        value: dict[str, CompatibleType] | CompatibleType,
        *,
        inplace: bool,
        validated: bool,
    ) -> TensorDictBase:
        if len(key) == 1:
            return self._set_str(key[0], value, inplace=inplace, validated=validated)
        td = self._get_str(key[0], None)
        if td is None:
            self._create_nested_str(key[0])
            td = self._get_str(key[0], NO_DEFAULT)
            inplace = False
        elif not _is_tensor_collection(td.__class__):
            raise RuntimeError(
                f"The entry {key[0]} is already present in " f"tensordict {self}."
            )
        td._set_tuple(key[1:], value, inplace=inplace, validated=validated)
        return self

    def _set_at_str(self, key, value, idx, *, validated):
        if not validated:
            value = self._validate_value(value, check_shape=False)
            validated = True
        tensor_in = self._get_str(key, NO_DEFAULT)

        if isinstance(idx, tuple) and len(idx) and isinstance(idx[0], tuple):
            warn(
                "Multiple indexing can lead to unexpected behaviours when "
                "setting items, for instance `td[idx1][idx2] = other` may "
                "not write to the desired location if idx1 is a list/tensor."
            )
            tensor_in = _sub_index(tensor_in, idx)
            tensor_in.copy_(value)
        else:
            _set_item(tensor_in, idx, value, validated=validated)

        return self

    def _set_at_tuple(self, key, value, idx, *, validated):
        if len(key) == 1:
            return self._set_at_str(key[0], value, idx, validated=validated)
        if key[0] not in self.keys():
            # this won't work
            raise KeyError(f"key {key} not found in set_at_ with tensordict {self}.")
        else:
            td = self._get_str(key[0], NO_DEFAULT)
        td._set_at_tuple(key[1:], value, idx, validated=validated)
        return self

    def del_(self, key: str) -> TensorDictBase:
        key = _unravel_key_to_tuple(key)
        if len(key) > 1:
            td, subkey = _get_leaf_tensordict(self, key)
            td.del_(subkey)
            return self

        del self._tensordict[key[0]]
        return self

    @lock_blocked
    def rename_key_(
        self, old_key: str, new_key: str, safe: bool = False
    ) -> TensorDictBase:
        # these checks are not perfect, tuples that are not tuples of strings or empty
        # tuples could go through but (1) it will raise an error anyway and (2)
        # those checks are expensive when repeated often.
        if not isinstance(old_key, (str, tuple)):
            raise TypeError(
                f"Expected old_name to be a string or a tuple of strings but found {type(old_key)}"
            )
        if not isinstance(new_key, (str, tuple)):
            raise TypeError(
                f"Expected new_name to be a string or a tuple of strings but found {type(new_key)}"
            )
        if safe and (new_key in self.keys(include_nested=True)):
            raise KeyError(f"key {new_key} already present in TensorDict.")

        if isinstance(new_key, str):
            self._set_str(new_key, self.get(old_key), inplace=False, validated=True)
        else:
            self._set_tuple(new_key, self.get(old_key), inplace=False, validated=True)
        self.del_(old_key)
        return self

    rename_key = _renamed_inplace_method(rename_key_)

    def _stack_onto_(
        self, key: str, list_item: list[CompatibleType], dim: int
    ) -> TensorDict:
        if not isinstance(key, str):
            raise ValueError("_stack_onto_ expects string keys.")
        torch.stack(list_item, dim=dim, out=self._get_str(key, NO_DEFAULT))
        return self

    def entry_class(self, key: NestedKey) -> type:
        return type(self.get(key))

    def _stack_onto_at_(
        self,
        key: str,
        list_item: list[CompatibleType],
        dim: int,
        idx: IndexType,
    ) -> TensorDict:
        if isinstance(idx, tuple) and len(idx) == 1:
            idx = idx[0]
        if isinstance(idx, (int, slice)) or (
            isinstance(idx, tuple)
            and all(isinstance(_idx, (int, slice)) for _idx in idx)
        ):
            torch.stack(list_item, dim=dim, out=self._tensordict[key][idx])
        else:
            raise ValueError(
                f"Cannot stack onto an indexed tensor with index {idx} "
                f"as its storage differs."
            )
        return self

    def _get_str(self, key, default):
        first_key = key
        out = self._tensordict.get(first_key, None)
        if out is None:
            return self._default_get(first_key, default)
        return out

    def _get_tuple(self, key, default):
        first = self._get_str(key[0], default)
        if len(key) == 1 or first is default:
            return first
        try:
            if isinstance(first, KeyedJaggedTensor):
                if len(key) != 2:
                    raise ValueError(f"Got too many keys for a KJT: {key}.")
                return first[key[1]]
            else:
                return first._get_tuple(key[1:], default=default)
        except AttributeError as err:
            if "has no attribute" in str(err):
                raise ValueError(
                    f"Expected a TensorDictBase instance but got {type(first)} instead"
                    f" for key '{key[1:]}' in tensordict:\n{self}."
                )

    def share_memory_(self) -> TensorDictBase:
        if self.is_memmap():
            raise RuntimeError(
                "memmap and shared memory are mutually exclusive features."
            )
        if self.device is not None and self.device.type == "cuda":
            # cuda tensors are shared by default
            return self
        for value in self.values():
            # no need to consider MemmapTensors here as we have checked that this is not a memmap-tensordict
            if (
                isinstance(value, Tensor)
                and value.device.type == "cpu"
                or _is_tensor_collection(value.__class__)
            ):
                value.share_memory_()
        self._is_shared = True
        self.lock_()
        return self

    def detach_(self) -> TensorDictBase:
        for value in self.values():
            value.detach_()
        return self

    def memmap_(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
    ) -> TensorDictBase:
        if prefix is not None:
            prefix = Path(prefix)
            if not prefix.exists():
                os.makedirs(prefix, exist_ok=True)
            torch.save(
                {"batch_size": self.batch_size, "device": self.device},
                prefix / "meta.pt",
            )
        if self.is_shared() and self.device.type == "cpu":
            raise RuntimeError(
                "memmap and shared memory are mutually exclusive features."
            )
        if not self._tensordict.keys():
            raise Exception(
                "memmap_() must be called when the TensorDict is (partially) "
                "populated. Set a tensor first."
            )
        for key, value in self.items():
            if value.requires_grad:
                raise Exception(
                    "memmap is not compatible with gradients, one of Tensors has requires_grad equals True"
                )
            if _is_tensor_collection(value.__class__):
                if prefix is not None:
                    # ensure subdirectory exists
                    os.makedirs(prefix / key, exist_ok=True)
                    self._tensordict[key] = value.memmap_(
                        prefix=prefix / key, copy_existing=copy_existing
                    )
                    torch.save(
                        {"batch_size": value.batch_size, "device": value.device},
                        prefix / key / "meta.pt",
                    )
                else:
                    self._tensordict[key] = value.memmap_()
                continue
            elif isinstance(value, MemmapTensor):
                if (
                    # user didn't specify location
                    prefix is None
                    # file is already in the correct location
                    or str(prefix / f"{key}.memmap") == value.filename
                ):
                    self._tensordict[key] = value
                elif copy_existing:
                    # user did specify location and memmap is in wrong place, so we copy
                    self._tensordict[key] = MemmapTensor.from_tensor(
                        value, filename=str(prefix / f"{key}.memmap")
                    )
                else:
                    # memmap in wrong location and copy is disallowed
                    raise RuntimeError(
                        "TensorDict already contains MemmapTensors saved to a location "
                        "incompatible with prefix. Either move the location of the "
                        "MemmapTensors, or allow automatic copying with "
                        "copy_existing=True"
                    )
            else:
                self._tensordict[key] = MemmapTensor.from_tensor(
                    value,
                    filename=str(prefix / f"{key}.memmap")
                    if prefix is not None
                    else None,
                )
            if prefix is not None:
                torch.save(
                    {
                        "shape": value.shape,
                        "device": value.device,
                        "dtype": value.dtype,
                    },
                    prefix / f"{key}.meta.pt",
                )
        self._is_memmap = True
        self.lock_()
        return self

    @classmethod
    def load_memmap(cls, prefix: str) -> TensorDictBase:
        prefix = Path(prefix)
        metadata = torch.load(prefix / "meta.pt")
        out = cls({}, batch_size=metadata["batch_size"], device=metadata["device"])

        for path in prefix.glob("**/*meta.pt"):
            key = path.parts[len(prefix.parts) :]
            if path.name == "meta.pt":
                if path == prefix / "meta.pt":
                    # skip prefix / "meta.pt" as we've already read it
                    continue
                key = key[:-1]  # drop "meta.pt" from key
                metadata = torch.load(path)
                if key in out.keys(include_nested=True):
                    out[key].batch_size = metadata["batch_size"]
                    device = metadata["device"]
                    if device is not None:
                        out[key] = out[key].to(device)
                else:
                    out[key] = cls(
                        {}, batch_size=metadata["batch_size"], device=metadata["device"]
                    )
            else:
                leaf, *_ = key[-1].rsplit(".", 2)  # remove .meta.pt suffix
                key = (*key[:-1], leaf)
                metadata = torch.load(path)
                out[key] = MemmapTensor(
                    *metadata["shape"],
                    device=metadata["device"],
                    dtype=metadata["dtype"],
                    filename=str(path.parent / f"{leaf}.memmap"),
                )

        return out

    def to(self, dest: DeviceType | torch.Size | type, **kwargs: Any) -> TensorDictBase:
        if isinstance(dest, type) and issubclass(dest, TensorDictBase):
            if isinstance(self, dest):
                return self
            td = dest(source=self, **kwargs)
            if self._td_dim_names is not None:
                td.names = self._td_dim_names
            return td
        elif isinstance(dest, (torch.device, str, int)):
            # must be device
            dest = torch.device(dest)
            if self.device is not None and dest == self.device:
                return self

            def to(tensor):
                return tensor.to(dest, **kwargs)

            return self.apply(to, device=dest)
        elif isinstance(dest, torch.Size):
            self.batch_size = dest
            return self
        elif dest is None:
            return self
        else:
            raise NotImplementedError(
                f"dest must be a string, torch.device or a TensorDict "
                f"instance, {dest} not allowed"
            )

    def masked_fill_(self, mask: Tensor, value: float | int | bool) -> TensorDictBase:
        for item in self.values():
            mask_expand = expand_as_right(mask, item)
            item.masked_fill_(mask_expand, value)
        return self

    def masked_fill(self, mask: Tensor, value: float | bool) -> TensorDictBase:
        td_copy = self.clone()
        return td_copy.masked_fill_(mask, value)

    def is_contiguous(self) -> bool:
        return all([value.is_contiguous() for _, value in self.items()])

    def clone(self, recurse: bool = True) -> TensorDictBase:
        return TensorDict(
            source={key: _clone_value(value, recurse) for key, value in self.items()},
            batch_size=self.batch_size,
            device=self.device,
            names=copy(self._td_dim_names),
            _run_checks=False,
            _is_shared=self.is_shared() if not recurse else False,
            _is_memmap=self.is_memmap() if not recurse else False,
        )

    def contiguous(self) -> TensorDictBase:
        if not self.is_contiguous():
            return self.clone()
        return self

    def select(
        self, *keys: NestedKey, inplace: bool = False, strict: bool = True
    ) -> TensorDictBase:
        source = {}
        if len(keys):
            keys_to_select = None
            for key in keys:
                if isinstance(key, str):
                    subkey = []
                else:
                    key, subkey = key[0], key[1:]
                try:
                    source[key] = self.get(key)
                    if len(subkey):
                        if keys_to_select is None:
                            # delay creation of defaultdict
                            keys_to_select = defaultdict(list)
                        keys_to_select[key].append(subkey)
                except KeyError as err:
                    if not strict:
                        continue
                    else:
                        raise KeyError(f"select failed to get key {key}") from err
            if keys_to_select is not None:
                for key, val in keys_to_select.items():
                    source[key] = source[key].select(
                        *val, strict=strict, inplace=inplace
                    )

        out = TensorDict(
            device=self.device,
            batch_size=self.batch_size,
            source=source,
            # names=self.names if self._has_names() else None,
            names=self._td_dim_names,
            _run_checks=False,
            _is_memmap=self._is_memmap,
            _is_shared=self._is_shared,
        )
        if inplace:
            self._tensordict = out._tensordict
            return self
        return out

    def keys(
        self, include_nested: bool = False, leaves_only: bool = False
    ) -> _TensorDictKeysView:
        if not include_nested and not leaves_only:
            return self._tensordict.keys()
        else:
            return self._nested_keys(
                include_nested=include_nested, leaves_only=leaves_only
            )

    # @cache  # noqa: B019
    def _nested_keys(
        self, include_nested: bool = False, leaves_only: bool = False
    ) -> _TensorDictKeysView:
        return _TensorDictKeysView(
            self, include_nested=include_nested, leaves_only=leaves_only
        )

    def __getstate__(self):
        return {
            slot: getattr(self, slot)
            for slot in self.__slots__
            if slot not in ("_last_op", "_cache", "__last_op_queue")
        }

    def __setstate__(self, state):
        for slot, value in state.items():
            setattr(self, slot, value)
        self._cache = None
        self._last_op = collections.deque()

    # some custom methods for efficiency
    def items(
        self, include_nested: bool = False, leaves_only: bool = False
    ) -> Iterator[tuple[str, CompatibleType]]:
        if not include_nested and not leaves_only:
            return self._tensordict.items()
        else:
            return super().items(include_nested=include_nested, leaves_only=leaves_only)

    def values(
        self, include_nested: bool = False, leaves_only: bool = False
    ) -> Iterator[tuple[str, CompatibleType]]:
        if not include_nested and not leaves_only:
            return self._tensordict.values()
        else:
            return super().values(
                include_nested=include_nested, leaves_only=leaves_only
            )


class _ErrorInteceptor:
    """Context manager for catching errors and modifying message.

    Intended for use with stacking / concatenation operations applied to TensorDicts.

    """

    DEFAULT_EXC_MSG = "Expected all tensors to be on the same device"

    def __init__(
        self,
        key: NestedKey,
        prefix: str,
        exc_msg: str | None = None,
        exc_type: type[Exception] | None = None,
    ) -> None:
        self.exc_type = exc_type if exc_type is not None else RuntimeError
        self.exc_msg = exc_msg if exc_msg is not None else self.DEFAULT_EXC_MSG
        self.prefix = prefix
        self.key = key

    def _add_key_to_error_msg(self, msg: str) -> str:
        if msg.startswith(self.prefix):
            return f'{self.prefix} "{self.key}" /{msg[len(self.prefix):]}'
        return f'{self.prefix} "{self.key}". {msg}'

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, _):
        if exc_type is self.exc_type and (
            self.exc_msg is None or self.exc_msg in str(exc_value)
        ):
            exc_value.args = (self._add_key_to_error_msg(str(exc_value)),)


def _nested_keys_to_dict(keys: Iterator[NestedKey]) -> dict[str, Any]:
    nested_keys = {}
    for key in keys:
        if isinstance(key, str):
            nested_keys.setdefault(key, {})
        else:
            d = nested_keys
            for subkey in key:
                d = d.setdefault(subkey, {})
    return nested_keys


def _dict_to_nested_keys(
    nested_keys: dict[NestedKey, NestedKey], prefix: tuple[str, ...] = ()
) -> tuple[str, ...]:
    for key, subkeys in nested_keys.items():
        if subkeys:
            yield from _dict_to_nested_keys(subkeys, prefix=(*prefix, key))
        elif prefix:
            yield (*prefix, key)
        else:
            yield key


def _default_hook(td: TensorDictBase, key: tuple[str, ...]) -> None:
    """Used to populate a tensordict.

    For example, ``td.set(("a", "b"))`` may require to create ``"a"``.

    """
    out = td.get(key[0], None)
    if out is None:
        td._create_nested_str(key[0])
        out = td._get_str(key[0], None)
    return out


def _get_leaf_tensordict(
    tensordict: TensorDictBase, key: tuple[str, ...], hook: Callable = None
) -> tuple[TensorDictBase, str]:
    # utility function for traversing nested tensordicts
    # hook should return the default value for tensordit.get(key)
    while len(key) > 1:
        if hook is not None:
            tensordict = hook(tensordict, key)
        else:
            tensordict = tensordict.get(key[0])
        key = key[1:]
    return tensordict, key[0]


def implements_for_td(torch_function: Callable) -> Callable[[Callable], Callable]:
    """Register a torch function override for TensorDict."""

    @functools.wraps(torch_function)
    def decorator(func: Callable) -> Callable:
        TD_HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


# @implements_for_td(torch.testing.assert_allclose) TODO
def assert_allclose_td(
    actual: TensorDictBase,
    expected: TensorDictBase,
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = True,
    msg: str = "",
) -> bool:
    """Compares two tensordicts and raise an exception if their content does not match exactly."""
    if not _is_tensor_collection(actual.__class__) or not _is_tensor_collection(
        expected.__class__
    ):
        raise TypeError("assert_allclose inputs must be of TensorDict type")
    set1 = set(actual.keys())
    set2 = set(expected.keys())
    if not (len(set1.difference(set2)) == 0 and len(set2) == len(set1)):
        raise KeyError(
            "actual and expected tensordict keys mismatch, "
            f"keys {(set1 - set2).union(set2 - set1)} appear in one but not "
            f"the other."
        )
    keys = sorted(actual.keys(), key=str)
    for key in keys:
        input1 = actual.get(key)
        input2 = expected.get(key)
        if _is_tensor_collection(input1.__class__):
            assert_allclose_td(input1, input2, rtol=rtol, atol=atol)
            continue

        mse = (input1.to(torch.float) - input2.to(torch.float)).pow(2).sum()
        mse = mse.div(input1.numel()).sqrt().item()

        default_msg = f"key {key} does not match, got mse = {mse:4.4f}"
        msg = "\t".join([default_msg, msg]) if len(msg) else default_msg
        if isinstance(input1, MemmapTensor):
            input1 = input1._tensor
        if isinstance(input2, MemmapTensor):
            input2 = input2._tensor
        torch.testing.assert_close(
            input1, input2, rtol=rtol, atol=atol, equal_nan=equal_nan, msg=msg
        )
    return True


@implements_for_td(torch.unbind)
def _unbind(
    td: TensorDictBase, *args: Any, **kwargs: Any
) -> tuple[TensorDictBase, ...]:
    return td.unbind(*args, **kwargs)


@implements_for_td(torch.gather)
def _gather(
    input: TensorDictBase,
    dim: int,
    index: Tensor,
    *,
    sparse_grad: bool = False,
    out: TensorDictBase | None = None,
) -> TensorDictBase:
    if sparse_grad:
        raise NotImplementedError(
            "sparse_grad=True not implemented for torch.gather(tensordict, ...)"
        )
    # the index must have as many dims as the tensordict
    if not len(index):
        raise RuntimeError("Cannot use torch.gather with an empty index")
    dim_orig = dim
    if dim < 0:
        dim = input.batch_dims + dim
    if dim > input.batch_dims - 1 or dim < 0:
        raise RuntimeError(
            f"Cannot gather tensordict with shape {input.shape} along dim {dim_orig}."
        )

    def _gather_tensor(tensor, dest=None):
        index_expand = index
        while index_expand.ndim < tensor.ndim:
            index_expand = index_expand.unsqueeze(-1)
        target_shape = list(tensor.shape)
        target_shape[dim] = index_expand.shape[dim]
        index_expand = index_expand.expand(target_shape)
        out = torch.gather(tensor, dim, index_expand, out=dest)
        return out

    if out is None:

        names = input.names if input._has_names() else None

        return TensorDict(
            {key: _gather_tensor(value) for key, value in input.items()},
            batch_size=index.shape,
            names=names,
        )
    TensorDict(
        {key: _gather_tensor(value, out[key]) for key, value in input.items()},
        batch_size=index.shape,
    )
    return out


@implements_for_td(torch.full_like)
def _full_like(td: TensorDictBase, fill_value: float, **kwargs: Any) -> TensorDictBase:
    td_clone = td.clone()
    for key in td_clone.keys():
        td_clone.fill_(key, fill_value)
    if "dtype" in kwargs:
        raise ValueError("Cannot pass dtype to full_like with TensorDict")
    if "device" in kwargs:
        td_clone = td_clone.to(kwargs.pop("device"))
    if len(kwargs):
        raise RuntimeError(
            f"keyword arguments {list(kwargs.keys())} are not "
            f"supported with full_like with TensorDict"
        )
    return td_clone


@implements_for_td(torch.zeros_like)
def _zeros_like(td: TensorDictBase, **kwargs: Any) -> TensorDictBase:
    td_clone = td.apply(torch.zeros_like)
    if "dtype" in kwargs:
        raise ValueError("Cannot pass dtype to full_like with TensorDict")
    if "device" in kwargs:
        td_clone = td_clone.to(kwargs.pop("device"))
    if len(kwargs):
        raise RuntimeError(
            f"keyword arguments {list(kwargs.keys())} are not "
            f"supported with full_like with TensorDict"
        )
    return td_clone


@implements_for_td(torch.ones_like)
def _ones_like(td: TensorDictBase, **kwargs: Any) -> TensorDictBase:
    td_clone = td.clone()
    for key in td_clone.keys():
        td_clone.fill_(key, 1.0)
    if "device" in kwargs:
        td_clone = td_clone.to(kwargs.pop("device"))
    if len(kwargs):
        raise RuntimeError(
            f"keyword arguments {list(kwargs.keys())} are not "
            f"supported with full_like with TensorDict"
        )
    return td_clone


@implements_for_td(torch.empty_like)
def _empty_like(td: TensorDictBase, *args, **kwargs) -> TensorDictBase:
    try:
        tdclone = td.clone()
    except Exception as err:
        raise RuntimeError(
            "The tensordict passed to torch.empty_like cannot be "
            "cloned, preventing empty_like to be called. "
            "Consider calling tensordict.to_tensordict() first."
        ) from err

    return tdclone.apply_(lambda x: torch.empty_like(x, *args, **kwargs))


@implements_for_td(torch.clone)
def _clone(td: TensorDictBase, *args: Any, **kwargs: Any) -> TensorDictBase:
    return td.clone(*args, **kwargs)


@implements_for_td(torch.squeeze)
def _squeeze(td: TensorDictBase, *args: Any, **kwargs: Any) -> TensorDictBase:
    return td.squeeze(*args, **kwargs)


@implements_for_td(torch.unsqueeze)
def _unsqueeze(td: TensorDictBase, *args: Any, **kwargs: Any) -> TensorDictBase:
    return td.unsqueeze(*args, **kwargs)


@implements_for_td(torch.masked_select)
def _masked_select(td: TensorDictBase, *args: Any, **kwargs: Any) -> TensorDictBase:
    return td.masked_select(*args, **kwargs)


@implements_for_td(torch.permute)
def _permute(td: TensorDictBase, dims: Sequence[int]) -> TensorDictBase:
    return td.permute(*dims)


@implements_for_td(torch.cat)
def _cat(
    list_of_tensordicts: Sequence[TensorDictBase],
    dim: int = 0,
    device: DeviceType | None = None,
    out: TensorDictBase | None = None,
) -> TensorDictBase:
    if not list_of_tensordicts:
        raise RuntimeError("list_of_tensordicts cannot be empty")

    batch_size = list(list_of_tensordicts[0].batch_size)
    if dim < 0:
        dim = len(batch_size) + dim
    if dim >= len(batch_size):
        raise RuntimeError(
            f"dim must be in the range 0 <= dim < len(batch_size), got dim"
            f"={dim} and batch_size={batch_size}"
        )
    batch_size[dim] = sum([td.batch_size[dim] for td in list_of_tensordicts])
    batch_size = torch.Size(batch_size)

    # check that all tensordict match
    keys = _check_keys(list_of_tensordicts, strict=True)
    if out is None:
        out = {}
        for key in keys:
            with _ErrorInteceptor(
                key, "Attempted to concatenate tensors on different devices at key"
            ):
                out[key] = torch.cat(
                    [td._get_str(key, NO_DEFAULT) for td in list_of_tensordicts], dim
                )
        if device is None:
            device = list_of_tensordicts[0].device
            for td in list_of_tensordicts[1:]:
                if device == td.device:
                    continue
                else:
                    device = None
                    break
        names = None
        if list_of_tensordicts[0]._has_names():
            names = list_of_tensordicts[0].names
        return TensorDict(
            out, device=device, batch_size=batch_size, _run_checks=False, names=names
        )
    else:
        if out.batch_size != batch_size:
            raise RuntimeError(
                "out.batch_size and cat batch size must match, "
                f"got out.batch_size={out.batch_size} and batch_size"
                f"={batch_size}"
            )

        for key in keys:
            with _ErrorInteceptor(
                key, "Attempted to concatenate tensors on different devices at key"
            ):
                if isinstance(out, TensorDict):
                    torch.cat(
                        [td.get(key) for td in list_of_tensordicts],
                        dim,
                        out=out.get(key),
                    )
                else:
                    out.set_(
                        key, torch.cat([td.get(key) for td in list_of_tensordicts], dim)
                    )
        return out


@implements_for_td(torch.stack)
def _stack(
    list_of_tensordicts: Sequence[TensorDictBase],
    dim: int = 0,
    device: DeviceType | None = None,
    out: TensorDictBase | None = None,
    strict: bool = False,
    contiguous: bool = False,
) -> TensorDictBase:
    if not list_of_tensordicts:
        raise RuntimeError("list_of_tensordicts cannot be empty")
    batch_size = list_of_tensordicts[0].batch_size
    if dim < 0:
        dim = len(batch_size) + dim + 1

    for td in list_of_tensordicts[1:]:
        if td.batch_size != list_of_tensordicts[0].batch_size:
            raise RuntimeError(
                "stacking tensordicts requires them to have congruent batch sizes, "
                f"got td1.batch_size={td.batch_size} and td2.batch_size="
                f"{list_of_tensordicts[0].batch_size}"
            )

    # check that all tensordict match
    keys = _check_keys(list_of_tensordicts)

    if out is None:
        device = list_of_tensordicts[0].device
        if contiguous:
            out = {}
            for key in keys:
                with _ErrorInteceptor(
                    key, "Attempted to stack tensors on different devices at key"
                ):
                    out[key] = torch.stack(
                        [_tensordict.get(key) for _tensordict in list_of_tensordicts],
                        dim,
                    )

            return TensorDict(
                out,
                batch_size=LazyStackedTensorDict._compute_batch_size(
                    batch_size, dim, len(list_of_tensordicts)
                ),
                device=device,
                _run_checks=False,
            )
        else:
            out = LazyStackedTensorDict(
                *list_of_tensordicts,
                stack_dim=dim,
            )
    else:
        batch_size = list(batch_size)
        batch_size.insert(dim, len(list_of_tensordicts))
        batch_size = torch.Size(batch_size)

        if out.batch_size != batch_size:
            raise RuntimeError(
                "out.batch_size and stacked batch size must match, "
                f"got out.batch_size={out.batch_size} and batch_size"
                f"={batch_size}"
            )

        out_keys = set(out.keys())
        if strict:
            in_keys = set(keys)
            if len(out_keys - in_keys) > 0:
                raise RuntimeError(
                    "The output tensordict has keys that are missing in the "
                    "tensordict that has to be written: {out_keys - in_keys}. "
                    "As per the call to `stack(..., strict=True)`, this "
                    "is not permitted."
                )
            elif len(in_keys - out_keys) > 0:
                raise RuntimeError(
                    "The resulting tensordict has keys that are missing in "
                    f"its destination: {in_keys - out_keys}. As per the call "
                    "to `stack(..., strict=True)`, this is not permitted."
                )

        for key in keys:
            if key in out_keys:
                out._stack_onto_(
                    key,
                    [_tensordict.get(key) for _tensordict in list_of_tensordicts],
                    dim,
                )
            else:
                with _ErrorInteceptor(
                    key, "Attempted to stack tensors on different devices at key"
                ):
                    out.set(
                        key,
                        torch.stack(
                            [
                                _tensordict.get(key)
                                for _tensordict in list_of_tensordicts
                            ],
                            dim,
                        ),
                        inplace=True,
                    )

    return out


def pad(
    tensordict: TensorDictBase, pad_size: Sequence[int], value: float = 0.0
) -> TensorDictBase:
    """Pads all tensors in a tensordict along the batch dimensions with a constant value, returning a new tensordict.

    Args:
         tensordict (TensorDict): The tensordict to pad
         pad_size (Sequence[int]): The padding size by which to pad some batch
            dimensions of the tensordict, starting from the first dimension and
            moving forward. [len(pad_size) / 2] dimensions of the batch size will
            be padded. For example to pad only the first dimension, pad has the form
            (padding_left, padding_right). To pad two dimensions,
            (padding_left, padding_right, padding_top, padding_bottom) and so on.
            pad_size must be even and less than or equal to twice the number of batch dimensions.
         value (float, optional): The fill value to pad by, default 0.0

    Returns:
        A new TensorDict padded along the batch dimensions

    Examples:
        >>> from tensordict import TensorDict
        >>> from tensordict.tensordict import pad
        >>> import torch
        >>> td = TensorDict({'a': torch.ones(3, 4, 1),
        ...     'b': torch.ones(3, 4, 1, 1)}, batch_size=[3, 4])
        >>> dim0_left, dim0_right, dim1_left, dim1_right = [0, 1, 0, 2]
        >>> padded_td = pad(td, [dim0_left, dim0_right, dim1_left, dim1_right], value=0.0)
        >>> print(padded_td.batch_size)
        torch.Size([4, 6])
        >>> print(padded_td.get("a").shape)
        torch.Size([4, 6, 1])
        >>> print(padded_td.get("b").shape)
        torch.Size([4, 6, 1, 1])

    """
    if len(pad_size) > 2 * len(tensordict.batch_size):
        raise RuntimeError(
            "The length of pad_size must be <= 2 * the number of batch dimensions"
        )

    if len(pad_size) % 2:
        raise RuntimeError("pad_size must have an even number of dimensions")

    new_batch_size = list(tensordict.batch_size)
    for i in range(len(pad_size)):
        new_batch_size[i // 2] += pad_size[i]

    reverse_pad = pad_size[::-1]
    for i in range(0, len(reverse_pad), 2):
        reverse_pad[i], reverse_pad[i + 1] = reverse_pad[i + 1], reverse_pad[i]

    out = TensorDict(
        {}, torch.Size(new_batch_size), device=tensordict.device, _run_checks=False
    )
    for key, tensor in tensordict.items():
        cur_pad = reverse_pad
        if len(pad_size) < len(_shape(tensor)) * 2:
            cur_pad = [0] * (len(_shape(tensor)) * 2 - len(pad_size)) + reverse_pad

        if _is_tensor_collection(tensor.__class__):
            padded = pad(tensor, pad_size, value)
        else:
            padded = torch.nn.functional.pad(tensor, cur_pad, value=value)
        out.set(key, padded)

    return out


def pad_sequence(
    list_of_tensordicts: Sequence[TensorDictBase],
    batch_first: bool = True,
    padding_value: float = 0.0,
    out: TensorDictBase | None = None,
    device: DeviceType | None = None,
    return_mask: bool | None = False,
) -> TensorDictBase:
    """Pads a list of tensordicts in order for them to be stacked together in a contiguous format.

    Args:
        list_of_tensordicts (List[TensorDictBase]): the list of instances to pad and stack.
        batch_first (bool, optional): the ``batch_first`` correspondant of :func:`torch.nn.utils.rnn.pad_sequence`.
            Defaults to ``True``.
        padding_value (number, optional): the padding value. Defaults to ``0.0``.
        out (TensorDictBase, optional): if provided, the destination where the data will be
            written.
        device (device compatible type, optional): if provded, the device where the
            TensorDict output will be created.
        return_mask (bool, optional): if ``True``, a "mask" entry will be returned.
            It contains the mask of valid values in the stacked tensordict.

    Examples:
        >>> list_td = [
        ...     TensorDict({"a": torch.zeros((3,))}, []),
        ...     TensorDict({"a": torch.zeros((4,))}, []),
        ...     ]
        >>> padded_td = pad_sequence(list_td)
        >>> print(padded_td)
        TensorDict(
            fields={
                a: Tensor(shape=torch.Size([2, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
    """
    if not list_of_tensordicts:
        raise RuntimeError("list_of_tensordicts cannot be empty")
    # check that all tensordict match
    if return_mask:
        list_of_tensordicts = [
            td.clone(False).set("mask", torch.ones(td.shape, dtype=torch.bool))
            for td in list_of_tensordicts
        ]
    keys = _check_keys(list_of_tensordicts, leaves_only=True, include_nested=True)
    shape = max(len(td) for td in list_of_tensordicts)
    if shape == 0:
        shape = [
            len(list_of_tensordicts),
        ]
    elif batch_first:
        shape = [len(list_of_tensordicts), shape]
    else:
        shape = [shape, len(list_of_tensordicts)]
    if out is None:
        out = TensorDict(
            {}, batch_size=torch.Size(shape), device=device, _run_checks=False
        )
        for key in keys:
            try:
                out.set(
                    key,
                    torch.nn.utils.rnn.pad_sequence(
                        [td.get(key) for td in list_of_tensordicts],
                        batch_first=batch_first,
                        padding_value=padding_value,
                    ),
                )
            except Exception as err:
                raise RuntimeError(f"pad_sequence failed for key {key}") from err
        return out
    else:
        for key in keys:
            out.set_(
                key,
                torch.nn.utils.rnn.pad_sequence(
                    [td.get(key) for td in list_of_tensordicts],
                    batch_first=batch_first,
                    padding_value=padding_value,
                ),
            )
        return out


@functools.wraps(pad_sequence)
def pad_sequence_ts(*args, **kwargs):
    """Warning: this function will soon be deprecated. Please use pad_sequence instead."""
    warnings.warn(
        "pad_sequence_ts will soon be deprecated in favour of pad_sequence. Please use the latter instead."
    )
    return pad_sequence(*args, **kwargs)


@implements_for_td(torch.split)
def _split(
    td: TensorDict, split_size_or_sections: int | list[int], dim: int = 0
) -> list[TensorDictBase]:
    return td.split(split_size_or_sections, dim)


class SubTensorDict(TensorDictBase):
    """A TensorDict that only sees an index of the stored tensors.

    By default, indexing a tensordict with an iterable will result in a
    SubTensorDict. This is done such that a TensorDict indexed with
    non-contiguous index (e.g. a Tensor) will still point to the original
    memory location (unlike regular indexing of tensors).

    Examples:
        >>> from tensordict import TensorDict, SubTensorDict
        >>> source = {'random': torch.randn(3, 4, 5, 6),
        ...    'zeros': torch.zeros(3, 4, 1, dtype=torch.bool)}
        >>> batch_size = torch.Size([3, 4])
        >>> td = TensorDict(source, batch_size)
        >>> td_index = td[:, 2]
        >>> print(type(td_index), td_index.shape)
        <class 'tensordict.tensordict.TensorDict'> \
torch.Size([3])
        >>> td_index = td[slice(None), slice(None)]
        >>> print(type(td_index), td_index.shape)
        <class 'tensordict.tensordict.TensorDict'> \
torch.Size([3, 4])
        >>> td_index = td.get_sub_tensordict((slice(None), torch.tensor([0, 2], dtype=torch.long)))
        >>> print(type(td_index), td_index.shape)
        <class 'tensordict.tensordict.SubTensorDict'> \
torch.Size([3, 2])
        >>> _ = td_index.fill_('zeros', 1)
        >>> # the indexed tensors are updated with Trues
        >>> print(td.get('zeros'))
        tensor([[[ True],
                 [False],
                 [ True],
                 [False]],
        <BLANKLINE>
                [[ True],
                 [False],
                 [ True],
                 [False]],
        <BLANKLINE>
                [[ True],
                 [False],
                 [ True],
                 [False]]])

    """

    def __new__(cls, *args: Any, **kwargs: Any) -> SubTensorDict:
        cls._is_shared = False
        cls._is_memmap = False
        return super().__new__(cls, _safe=False, _lazy=True, _inplace_set=True)

    def __init__(
        self,
        source: TensorDictBase,
        idx: IndexType,
        batch_size: Sequence[int] | None = None,
    ) -> None:
        if not _is_tensor_collection(source.__class__):
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

    # @staticmethod
    # def _convert_range(idx):
    #     return tuple(list(_idx) if isinstance(_idx, range) else _idx for _idx in idx)

    @staticmethod
    def _convert_ellipsis(idx, shape):
        if any(_idx is Ellipsis for _idx in idx):
            new_idx = []
            cursor = -1
            for _idx in idx:
                if _idx is Ellipsis:
                    if cursor == len(idx) - 1:
                        # then we can just skip
                        continue
                    n_upcoming = len(idx) - cursor - 1
                    while cursor < len(shape) - n_upcoming:
                        cursor += 1
                        new_idx.append(slice(None))
                else:
                    new_idx.append(_idx)
            return tuple(new_idx)
        return idx

    def exclude(self, *keys: str, inplace: bool = False) -> TensorDictBase:
        if inplace:
            return super().exclude(*keys, inplace=True)
        return self.to_tensordict().exclude(*keys, inplace=True)

    @property
    def batch_size(self) -> torch.Size:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_size: torch.Size) -> None:
        self._batch_size_setter(new_size)

    @property
    def names(self):
        names = self._source._get_names_idx(self.idx)
        if names is None:
            return [None] * self.batch_dims
        return names

    @names.setter
    def names(self, value):
        raise RuntimeError(
            "Names of a subtensordict cannot be modified. Instantiate the tensordict first."
        )

    def _has_names(self):
        return self._source._has_names()

    def _erase_names(self):
        raise RuntimeError(
            "Cannot erase names of a SubTensorDict. Erase source TensorDict's names instead."
        )

    def _rename_subtds(self, names):
        for key in self.keys():
            if _is_tensor_collection(self.entry_class(key)):
                raise RuntimeError("Cannot rename nested sub-tensordict dimensions.")

    @property
    def device(self) -> None | torch.device:
        return self._source.device

    @device.setter
    def device(self, value: DeviceType) -> None:
        self._source.device = value

    def _preallocate(self, key: str, value: CompatibleType) -> TensorDictBase:
        return self._source.set(key, value)

    def _convert_inplace(self, inplace, key):
        has_key = key in self.keys()
        if inplace is not False:
            if inplace is True and not has_key:  # inplace could be None
                raise KeyError(
                    TensorDictBase.KEY_ERROR.format(
                        key, self.__class__.__name__, sorted(self.keys())
                    )
                )
            inplace = has_key
        if not inplace and has_key:
            raise RuntimeError(
                "Calling `SubTensorDict.set(key, value, inplace=False)` is "
                "prohibited for existing tensors. Consider calling "
                "SubTensorDict.set_(...) or cloning your tensordict first."
            )
        elif not inplace and self.is_locked:
            raise RuntimeError(TensorDictBase.LOCK_ERROR)
        return inplace

    def _set_str(
        self,
        key: NestedKey,
        value: dict[str, CompatibleType] | CompatibleType,
        *,
        inplace: bool,
        validated: bool,
    ) -> TensorDictBase:
        inplace = self._convert_inplace(inplace, key)
        # it is assumed that if inplace=False then the key doesn't exist. This is
        # checked in set method, but not here. responsibility lies with the caller
        # so that this method can have minimal overhead from runtime checks
        parent = self._source
        if not validated:
            value = self._validate_value(value, check_shape=True)
            validated = True
        if not inplace:
            if _is_tensor_collection(value.__class__):
                value_expand = _expand_to_match_shape(
                    parent.batch_size, value, self.batch_dims, self.device
                )
                for _key, _tensor in value.items():
                    value_expand[_key] = _expand_to_match_shape(
                        parent.batch_size, _tensor, self.batch_dims, self.device
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
                if self.is_shared() and self.device.type == "cpu":
                    value_expand.share_memory_()
                elif self.is_memmap():
                    value_expand = MemmapTensor.from_tensor(value_expand)

            parent._set_str(key, value_expand, inplace=False, validated=validated)

        parent._set_at_str(key, value, self.idx, validated=validated)
        return self

    def _set_tuple(
        self,
        key: NestedKey,
        value: dict[str, CompatibleType] | CompatibleType,
        *,
        inplace: bool,
        validated: bool,
    ) -> TensorDictBase:
        if len(key) == 1:
            return self._set_str(key[0], value, inplace=inplace, validated=validated)
        parent = self._source
        td = parent._get_str(key[0], None)
        if td is None:
            td = parent.select()
            parent._set_str(key[0], td, inplace=False, validated=True)
        SubTensorDict(td, self.idx)._set_tuple(
            key[1:], value, inplace=inplace, validated=validated
        )
        return self

    def _set_at_str(self, key, value, idx, *, validated):
        tensor_in = self._get_str(key, NO_DEFAULT)
        if not validated:
            value = self._validate_value(value, check_shape=False)
            validated = True
        if isinstance(idx, tuple) and len(idx) and isinstance(idx[0], tuple):
            warn(
                "Multiple indexing can lead to unexpected behaviours when "
                "setting items, for instance `td[idx1][idx2] = other` may "
                "not write to the desired location if idx1 is a list/tensor."
            )
            tensor_in = _sub_index(tensor_in, idx)
            tensor_in.copy_(value)
        else:
            _set_item(tensor_in, idx, value, validated=validated)
        # make sure that the value is updated
        self._source._set_at_str(key, tensor_in, self.idx, validated=validated)
        return self

    def _set_at_tuple(self, key, value, idx, *, validated):
        if len(key) == 1:
            return self._set_at_str(key[0], value, idx, validated=validated)
        if key[0] not in self.keys():
            # this won't work
            raise KeyError(f"key {key} not found in set_at_ with tensordict {self}.")
        else:
            td = self._get_str(key[0], NO_DEFAULT)
        td._set_at_tuple(key[1:], value, idx, validated=validated)
        return self

    # @cache  # noqa: B019
    def keys(
        self, include_nested: bool = False, leaves_only: bool = False
    ) -> _TensorDictKeysView:
        return self._source.keys(include_nested=include_nested, leaves_only=leaves_only)

    def entry_class(self, key: NestedKey) -> type:
        source_type = type(self._source.get(key))
        if _is_tensor_collection(source_type):
            return self.__class__
        return source_type

    def _stack_onto_(
        self, key: str, list_item: list[CompatibleType], dim: int
    ) -> SubTensorDict:
        self._source._stack_onto_at_(key, list_item, dim=dim, idx=self.idx)
        return self

    def to(self, dest: DeviceType | torch.Size | type, **kwargs: Any) -> TensorDictBase:
        if isinstance(dest, type) and issubclass(dest, TensorDictBase):
            if isinstance(self, dest):
                return self
            out = dest(
                source=self.clone(),
            )
            if self._has_names():
                out.names = self.names
            return out
        elif isinstance(dest, (torch.device, str, int)):
            dest = torch.device(dest)
            # try:
            if self.device is not None and dest == self.device:
                return self
            td = self.to_tensordict().to(dest, **kwargs)
            # must be device
            return td

        elif isinstance(dest, torch.Size):
            self.batch_size = dest
            return self
        elif dest is None:
            return self
        else:
            raise NotImplementedError(
                f"dest must be a string, torch.device or a TensorDict "
                f"instance, {dest} not allowed"
            )

    def _change_batch_size(self, new_size: torch.Size) -> None:
        if not hasattr(self, "_orig_batch_size"):
            self._orig_batch_size = self.batch_size
        elif self._orig_batch_size == new_size:
            del self._orig_batch_size
        self._batch_size = new_size

    def get(
        self,
        key: NestedKey,
        default: Tensor | str | None = NO_DEFAULT,
    ) -> CompatibleType:
        return self._source.get_at(key, self.idx, default=default)

    def _get_str(self, key, default):
        if key in self.keys() and _is_tensor_collection(self.entry_class(key)):
            return SubTensorDict(self._source._get_str(key, NO_DEFAULT), self.idx)
        return self._source._get_at_str(key, self.idx, default=default)

    def _get_tuple(self, key, default):
        return self._source._get_at_tuple(key, self.idx, default=default)

    def update(
        self,
        input_dict_or_td: dict[str, CompatibleType] | TensorDictBase,
        clone: bool = False,
        inplace: bool = False,
        **kwargs,
    ) -> SubTensorDict:
        if input_dict_or_td is self:
            # no op
            return self
        keys = set(self.keys(False))
        for key, value in input_dict_or_td.items():
            if clone and hasattr(value, "clone"):
                value = value.clone()
            else:
                value = tree_map(torch.clone, value)
            key = _unravel_key_to_tuple(key)
            firstkey, subkey = key[0], key[1:]
            # the key must be a string by now. Let's check if it is present
            if firstkey in keys:
                target_class = self.entry_class(firstkey)
                if _is_tensor_collection(target_class):
                    target = self._source.get(firstkey).get_sub_tensordict(self.idx)
                    if len(subkey):
                        target._set_tuple(subkey, value, inplace=False, validated=False)
                        continue
                    elif isinstance(value, dict) or _is_tensor_collection(
                        value.__class__
                    ):
                        target.update(value)
                        continue
                    raise ValueError(
                        f"Tried to replace a tensordict with an incompatible object of type {type(value)}"
                    )
                else:
                    self._set_tuple(key, value, inplace=True, validated=False)
            else:
                self._set_tuple(
                    key,
                    value,
                    inplace=BEST_ATTEMPT_INPLACE if inplace else False,
                    validated=False,
                )
        return self

    def update_(
        self,
        input_dict: dict[str, CompatibleType] | TensorDictBase,
        clone: bool = False,
    ) -> SubTensorDict:
        return self.update_at_(
            input_dict, idx=self.idx, discard_idx_attr=True, clone=clone
        )

    def update_at_(
        self,
        input_dict: dict[str, CompatibleType] | TensorDictBase,
        idx: IndexType,
        discard_idx_attr: bool = False,
        clone: bool = False,
    ) -> SubTensorDict:
        for key, value in input_dict.items():
            if not isinstance(value, tuple(_ACCEPTED_CLASSES)):
                raise TypeError(
                    f"Expected value to be one of types {_ACCEPTED_CLASSES} "
                    f"but got {type(value)}"
                )
            if clone:
                value = value.clone()
            key = _unravel_key_to_tuple(key)
            if discard_idx_attr:
                self._source._set_at_tuple(
                    key,
                    value,
                    idx,
                    validated=False,
                )
            else:
                self._set_at_tuple(key, value, idx, validated=False)
        return self

    def get_parent_tensordict(self) -> TensorDictBase:
        if not isinstance(self._source, TensorDictBase):
            raise TypeError(
                f"SubTensorDict was initialized with a source of type"
                f" {self._source.__class__.__name__}, "
                "parent tensordict not accessible"
            )
        return self._source

    def del_(self, key: str) -> TensorDictBase:
        self._source = self._source.del_(key)
        return self

    def clone(self, recurse: bool = True) -> SubTensorDict:
        """Clones the SubTensorDict.

        Args:
            recurse (bool, optional): if ``True`` (default), a regular
                :class:`TensorDict` instance will be created from the :class:`SubTensorDict`.
                Otherwise, another :class:`SubTensorDict` with identical content
                will be returned.

        Examples:
            >>> data = TensorDict({"a": torch.arange(4).reshape(2, 2,)}, batch_size=[2, 2])
            >>> sub_data = data.get_sub_tensordict([0,])
            >>> print(sub_data)
            SubTensorDict(
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
            SubTensorDict(
                fields={
                    a: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.int64, is_shared=False)},
                batch_size=torch.Size([2]),
                device=None,
                is_shared=False)
            >>> print(sub_data.get("a").data_ptr())
            140183705558208
        """
        if not recurse:
            return SubTensorDict(source=self._source.clone(recurse=False), idx=self.idx)
        return self.to_tensordict()

    def is_contiguous(self) -> bool:
        return all(value.is_contiguous() for value in self.values())

    def contiguous(self) -> TensorDictBase:
        if self.is_contiguous():
            return self
        return TensorDict(
            batch_size=self.batch_size,
            source={key: value for key, value in self.items()},
            device=self.device,
            names=self.names,
            _run_checks=False,
        )

    def select(
        self, *keys: str, inplace: bool = False, strict: bool = True
    ) -> TensorDictBase:
        if inplace:
            self._source = self._source.select(*keys, strict=strict)
            return self
        return self._source.select(*keys, strict=strict)[self.idx]

    def expand(self, *shape: int, inplace: bool = False) -> TensorDictBase:
        if len(shape) == 1 and isinstance(shape[0], Sequence):
            shape = tuple(shape[0])
        return self.apply(
            lambda x: x.expand((*shape, *x.shape[self.ndim :])), batch_size=shape
        )

    def is_shared(self) -> bool:
        return self._source.is_shared()

    def is_memmap(self) -> bool:
        return self._source.is_memmap()

    def rename_key_(
        self, old_key: str, new_key: str, safe: bool = False
    ) -> SubTensorDict:
        self._source.rename_key_(old_key, new_key, safe=safe)
        return self

    rename_key = _renamed_inplace_method(rename_key_)

    def pin_memory(self) -> TensorDictBase:
        self._source.pin_memory()
        return self

    def detach_(self) -> TensorDictBase:
        raise RuntimeError("Detaching a sub-tensordict in-place cannot be done.")

    def masked_fill_(self, mask: Tensor, value: float | bool) -> TensorDictBase:
        for key, item in self.items():
            self.set_(key, torch.full_like(item, value))
        return self

    def masked_fill(self, mask: Tensor, value: float | bool) -> TensorDictBase:
        td_copy = self.clone()
        return td_copy.masked_fill_(mask, value)

    def memmap_(
        self, prefix: str | None = None, copy_existing: bool = False
    ) -> TensorDictBase:
        raise RuntimeError(
            "Converting a sub-tensordict values to memmap cannot be done."
        )

    def share_memory_(self) -> TensorDictBase:
        raise RuntimeError(
            "Casting a sub-tensordict values to shared memory cannot be done."
        )

    @property
    def is_locked(self) -> bool:
        return self._source.is_locked

    @is_locked.setter
    def is_locked(self, value) -> bool:
        if value:
            self.lock_()
        else:
            self.unlock_()

    @as_decorator("is_locked")
    def lock_(self) -> TensorDictBase:
        # we can't lock sub-tensordicts because that would mean that the
        # parent tensordict cannot be modified either.
        if not self.is_locked:
            raise RuntimeError(
                "Cannot lock a SubTensorDict. Lock the parent tensordict instead."
            )
        return self

    @as_decorator("is_locked")
    def unlock_(self) -> TensorDictBase:
        if self.is_locked:
            raise RuntimeError(
                "Cannot unlock a SubTensorDict. Unlock the parent tensordict instead."
            )
        return self

    def _remove_lock(self, lock_id):
        raise RuntimeError(
            "Cannot unlock a SubTensorDict. Unlock the parent tensordict instead."
        )

    def _lock_propagate(self, lock_ids=None):
        raise RuntimeError(
            "Cannot lock a SubTensorDict. Lock the parent tensordict instead."
        )

    lock = _renamed_inplace_method(lock_)
    unlock = _renamed_inplace_method(unlock_)

    def __del__(self):
        pass


def merge_tensordicts(*tensordicts: TensorDictBase) -> TensorDictBase:
    """Merges tensordicts together."""
    if len(tensordicts) < 2:
        raise RuntimeError(
            f"at least 2 tensordicts must be provided, got" f" {len(tensordicts)}"
        )
    d = tensordicts[0].to_dict()
    batch_size = tensordicts[0].batch_size
    for td in tensordicts[1:]:
        d.update(td.to_dict())
        if td.batch_dims < len(batch_size):
            batch_size = td.batch_size
    return TensorDict(d, batch_size, device=td.device, _run_checks=False)


class _LazyStackedTensorDictKeysView(_TensorDictKeysView):
    tensordict: LazyStackedTensorDict

    def __len__(self) -> int:
        return len(self._keys())

    def _keys(self) -> list[str]:
        return self.tensordict._key_list()

    def __contains__(self, item):
        item = _unravel_key_to_tuple(item)
        if item[0] in self.tensordict._iterate_over_keys():
            if self.leaves_only:
                return not _is_tensor_collection(self.tensordict.entry_class(item[0]))
            has_first_key = True
        else:
            has_first_key = False
        if not has_first_key or len(item) == 1:
            return has_first_key
        # otherwise take the long way
        return all(
            item[1:]
            in tensordict.get(item[0]).keys(self.include_nested, self.leaves_only)
            for tensordict in self.tensordict.tensordicts
        )


class LazyStackedTensorDict(TensorDictBase):
    """A Lazy stack of TensorDicts.

    When stacking TensorDicts together, the default behaviour is to put them
    in a stack that is not instantiated.
    This allows to seamlessly work with stacks of tensordicts with operations
    that will affect the original tensordicts.

    Args:
         *tensordicts (TensorDict instances): a list of tensordict with
            same batch size.
         stack_dim (int): a dimension (between `-td.ndimension()` and
            `td.ndimension()-1` along which the stack should be performed.
         hook_out (callable, optional): a callable to execute after :meth:`~.get`.
         hook_in (callable, optional): a callable to execute before :meth:`~.set`.

    Examples:
        >>> from tensordict import TensorDict
        >>> import torch
        >>> tds = [TensorDict({'a': torch.randn(3, 4)}, batch_size=[3])
        ...     for _ in range(10)]
        >>> td_stack = torch.stack(tds, -1)
        >>> print(td_stack.shape)
        torch.Size([3, 10])
        >>> print(td_stack.get("a").shape)
        torch.Size([3, 10, 4])
        >>> print(td_stack[:, 0] is tds[0])
        True

    """

    def __new__(cls, *args: Any, **kwargs: Any) -> LazyStackedTensorDict:
        cls._td_dim_name = None
        return super().__new__(cls, *args, _safe=False, _lazy=True, **kwargs)

    def __init__(
        self,
        *tensordicts: TensorDictBase,
        stack_dim: int = 0,
        hook_out: callable | None = None,
        hook_in: callable | None = None,
        batch_size: Sequence[int] | None = None,  # TODO: remove
    ) -> None:
        self._is_shared = False
        self._is_memmap = False
        self._is_locked = None

        # sanity check
        N = len(tensordicts)
        if not N:
            raise RuntimeError(
                "at least one tensordict must be provided to "
                "StackedTensorDict to be instantiated"
            )
        if not isinstance(tensordicts[0], TensorDictBase):
            raise TypeError(
                f"Expected input to be TensorDictBase instance"
                f" but got {type(tensordicts[0])} instead."
            )
        if stack_dim < 0:
            raise RuntimeError(
                f"stack_dim must be non negative, got stack_dim={stack_dim}"
            )
        _batch_size = tensordicts[0].batch_size
        device = tensordicts[0].device

        for td in tensordicts[1:]:
            if not isinstance(td, TensorDictBase):
                raise TypeError(
                    "Expected all inputs to be TensorDictBase instances but got "
                    f"{type(td)} instead."
                )
            _bs = td.batch_size
            _device = td.device
            if device != _device:
                raise RuntimeError(f"devices differ, got {device} and {_device}")
            if _bs != _batch_size:
                raise RuntimeError(
                    f"batch sizes in tensordicts differs, StackedTensorDict "
                    f"cannot be created. Got td[0].batch_size={_batch_size} "
                    f"and td[i].batch_size={_bs} "
                )
        self.tensordicts: list[TensorDictBase] = list(tensordicts)
        self.stack_dim = stack_dim
        self._batch_size = self._compute_batch_size(_batch_size, stack_dim, N)
        self.hook_out = hook_out
        self.hook_in = hook_in
        if batch_size is not None and batch_size != self.batch_size:
            raise RuntimeError("batch_size does not match self.batch_size.")

    @property
    def device(self) -> torch.device | None:
        # devices might have changed, so we check that they're all the same
        device_set = {td.device for td in self.tensordicts}
        if len(device_set) != 1:
            raise RuntimeError(
                f"found multiple devices in {self.__class__.__name__}:" f" {device_set}"
            )
        device = self.tensordicts[0].device
        return device

    @device.setter
    def device(self, value: DeviceType) -> None:
        for t in self.tensordicts:
            t.device = value

    @property
    def batch_size(self) -> torch.Size:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_size: torch.Size) -> None:
        return self._batch_size_setter(new_size)

    @property
    @cache  # noqa
    def names(self):
        names = list(self.tensordicts[0].names)
        for td in self.tensordicts[1:]:
            if names != td.names:
                raise ValueError(
                    f"Not all dim names match, got {names} and {td.names}."
                )
        names.insert(self.stack_dim, self._td_dim_name)
        return names

    @names.setter
    @erase_cache  # a nested lazy stacked tensordict is not apparent to the root
    def names(self, value):
        if value is None:
            for td in self.tensordicts:
                td.names = None
            self._td_dim_name = None
        else:
            names_c = list(value)
            name = names_c[self.stack_dim]
            self._td_dim_name = name
            del names_c[self.stack_dim]
            for td in self.tensordicts:
                if td._check_dim_name(name):
                    # TODO: should reset names here
                    raise ValueError(f"The dimension name {name} is already taken.")
                td.rename_(*names_c)

    def _rename_subtds(self, names):
        # remove the name of the stack dim
        names = list(names)
        del names[self.stack_dim]
        for td in self.tensordicts:
            td.names = names

    def _has_names(self):
        return all(td._has_names() for td in self.tensordicts)

    def _erase_names(self):
        self._td_dim_name = None
        for td in self.tensordicts:
            td._erase_names()

    def get_item_shape(self, key):
        """Gets the shape of an item in the lazy stack.

        Heterogeneous dimensions are returned as -1.

        This implementation is inefficient as it will attempt to stack the items
        to compute their shape, and should only be used for printing.
        """
        try:
            item = self.get(key)
            return item.shape
        except RuntimeError as err:
            if re.match(r"Found more than one unique shape in the tensors", str(err)):
                shape = None
                for td in self.tensordicts:
                    if shape is None:
                        shape = list(td.get_item_shape(key))
                    else:
                        _shape = td.get_item_shape(key)
                        if len(shape) != len(_shape):
                            shape = [-1]
                            return torch.Size(shape)
                        shape = [
                            s1 if s1 == s2 else -1 for (s1, s2) in zip(shape, _shape)
                        ]
                shape.insert(self.stack_dim, len(self.tensordicts))
                return torch.Size(shape)
            else:
                raise err

    def is_shared(self) -> bool:
        are_shared = [td.is_shared() for td in self.tensordicts]
        are_shared = [value for value in are_shared if value is not None]
        if not len(are_shared):
            return None
        if any(are_shared) and not all(are_shared):
            raise RuntimeError(
                f"tensordicts shared status mismatch, got {sum(are_shared)} "
                f"shared tensordicts and "
                f"{len(are_shared) - sum(are_shared)} non shared tensordict "
            )
        return all(are_shared)

    def is_memmap(self) -> bool:
        are_memmap = [td.is_memmap() for td in self.tensordicts]
        if any(are_memmap) and not all(are_memmap):
            raise RuntimeError(
                f"tensordicts memmap status mismatch, got {sum(are_memmap)} "
                f"memmap tensordicts and "
                f"{len(are_memmap) - sum(are_memmap)} non memmap tensordict "
            )
        return all(are_memmap)

    @staticmethod
    def _compute_batch_size(
        batch_size: torch.Size, stack_dim: int, N: int
    ) -> torch.Size:
        s = list(batch_size)
        s.insert(stack_dim, N)
        return torch.Size(s)

    def _set_str(
        self,
        key: NestedKey,
        value: dict[str, CompatibleType] | CompatibleType,
        *,
        inplace: bool,
        validated: bool,
    ) -> TensorDictBase:
        try:
            inplace = self._convert_inplace(inplace, key)
        except KeyError as e:
            raise KeyError(
                "setting a value in-place on a stack of TensorDict is only "
                "permitted if all members of the stack have this key in "
                "their register."
            ) from e
        if not validated:
            value = self._validate_value(value)
            validated = True
        if self.hook_in is not None:
            value = self.hook_in(value)
        values = value.unbind(self.stack_dim)
        for tensordict, item in zip(self.tensordicts, values):
            tensordict._set_str(key, item, inplace=inplace, validated=validated)
        return self

    def _set_tuple(
        self,
        key: NestedKey,
        value: dict[str, CompatibleType] | CompatibleType,
        *,
        inplace: bool,
        validated: bool,
    ) -> TensorDictBase:
        if len(key) == 1:
            return self._set_str(key[0], value, inplace=inplace, validated=validated)
        # if inplace is not False:  # inplace could be None
        #     # we don't want to end up in the situation where one tensordict has
        #     # inplace=True and another one inplace=False because inplace was loose.
        #     # Worse could be writing with inplace=True up until some level then to
        #     # realize the key is missing in one td, raising an exception and having
        #     # messed up the data. Hence we must start by checking if the key
        #     # is present.
        #     has_key = key in self.keys(True)
        #     if inplace is True and not has_key:  # inplace could be None
        #         raise KeyError(
        #             TensorDictBase.KEY_ERROR.format(
        #                 key, self.__class__.__name__, sorted(self.keys())
        #             )
        #         )
        #     inplace = has_key
        if not validated:
            value = self._validate_value(value)
            validated = True
        if self.hook_in is not None:
            value = self.hook_in(value)
        values = value.unbind(self.stack_dim)
        for tensordict, item in zip(self.tensordicts, values):
            tensordict._set_tuple(key, item, inplace=inplace, validated=validated)
        return self

    def _set_at_str(self, key, value, idx, *, validated):
        item = self._get_str(key, NO_DEFAULT)
        if not validated:
            value = self._validate_value(value, check_shape=False)
            validated = True
        if self.hook_in is not None:
            value = self.hook_in(value)
        item[idx] = value
        self._set_str(key, item, inplace=True, validated=validated)
        return self

    def _set_at_tuple(self, key, value, idx, *, validated):
        if len(key) == 1:
            return self._set_at_str(key[0], value, idx, validated=validated)
        # get the "last" tds
        tds = []
        for td in self.tensordicts:
            tds.append(td.get(key[:-1]))
        # build only a single lazy stack from it
        # (if the stack is a stack of stacks this won't be awesomely efficient
        # but then we'd need to splut the value (which we can do) and recompute
        # the sub-index for each td, which is a different story!
        td = LazyStackedTensorDict(
            *tds, stack_dim=self.stack_dim, hook_out=self.hook_out, hook_in=self.hook_in
        )
        if not validated:
            value = self._validate_value(value, check_shape=False)
            validated = True
        if self.hook_in is not None:
            value = self.hook_in(value)
        item = td._get_str(key, NO_DEFAULT)
        item[idx] = value
        td._set_str(key, item, inplace=True, validated=True)
        return self

    def unsqueeze(self, dim: int) -> TensorDictBase:
        if dim < 0:
            dim = self.batch_dims + dim + 1

        if (dim > self.batch_dims) or (dim < 0):
            raise RuntimeError(
                f"unsqueezing is allowed for dims comprised between "
                f"`-td.batch_dims` and `td.batch_dims` only. Got "
                f"dim={dim} with a batch size of {self.batch_size}."
            )
        if dim <= self.stack_dim:
            stack_dim = self.stack_dim + 1
        else:
            dim = dim - 1
            stack_dim = self.stack_dim
        return LazyStackedTensorDict(
            *(tensordict.unsqueeze(dim) for tensordict in self.tensordicts),
            stack_dim=stack_dim,
        )

    def squeeze(self, dim: int | None = None) -> TensorDictBase:
        """Squeezes all tensors for a dimension comprised in between `-td.batch_dims+1` and `td.batch_dims-1` and returns them in a new tensordict.

        Args:
            dim (Optional[int]): dimension along which to squeeze. If dim is None, all singleton dimensions will be squeezed. dim is None by default.

        """
        if dim is None:
            size = self.size()
            if len(self.size()) == 1 or size.count(1) == 0:
                return self
            first_singleton_dim = size.index(1)
            return self.squeeze(first_singleton_dim).squeeze()

        if dim < 0:
            dim = self.batch_dims + dim

        if self.batch_dims and (dim >= self.batch_dims or dim < 0):
            raise RuntimeError(
                f"squeezing is allowed for dims comprised between 0 and "
                f"td.batch_dims only. Got dim={dim} and batch_size"
                f"={self.batch_size}."
            )

        if dim >= self.batch_dims or self.batch_size[dim] != 1:
            return self
        if dim == self.stack_dim:
            return self.tensordicts[0]
        elif dim < self.stack_dim:
            stack_dim = self.stack_dim - 1
        else:
            dim = dim - 1
            stack_dim = self.stack_dim
        return LazyStackedTensorDict(
            *(tensordict.squeeze(dim) for tensordict in self.tensordicts),
            stack_dim=stack_dim,
        )

    def unbind(self, dim: int) -> tuple[TensorDictBase, ...]:
        if dim < 0:
            dim = self.batch_dims + dim
        if dim == self.stack_dim:
            return tuple(self.tensordicts)
        else:
            return super().unbind(dim)

    def _stack_onto_(
        self,
        key: str,
        list_item: list[CompatibleType],
        dim: int,
    ) -> TensorDictBase:
        if dim == self.stack_dim:
            for source, tensordict_dest in zip(list_item, self.tensordicts):
                tensordict_dest.set_(key, source)
        else:
            # we must stack and unbind, there is no way to make it more efficient
            self.set_(key, torch.stack(list_item, dim))
        return self

    @cache  # noqa: B019
    def _get_str(
        self,
        key: NestedKey,
        default: str | CompatibleType = NO_DEFAULT,
    ) -> CompatibleType:

        # we can handle the case where the key is a tuple of length 1
        tensors = []
        for td in self.tensordicts:
            tensors.append(td._get_str(key, default=default))
            if (
                tensors[-1] is default
                and not isinstance(
                    default, (MemmapTensor, KeyedJaggedTensor, torch.Tensor)
                )
                and not is_tensor_collection(default)
            ):
                # then we consider this default as non-stackable and return prematurly
                return default
        try:
            out = torch.stack(tensors, self.stack_dim)
            if _is_tensor_collection(out.__class__):
                if self._td_dim_name is not None:
                    out._td_dim_name = self._td_dim_name
                if isinstance(out, LazyStackedTensorDict):
                    # then it's a LazyStackedTD
                    out.hook_out = self.hook_out
                    out.hook_in = self.hook_in
                else:
                    # then it's a tensorclass
                    out._tensordict.hook_out = self.hook_out
                    out._tensordict.hook_in = self.hook_in
            elif self.hook_out is not None:
                out = self.hook_out(out)
            return out
        except RuntimeError as err:
            if "stack expects each tensor to be equal size" in str(err):
                shapes = {_shape(tensor) for tensor in tensors}
                raise RuntimeError(
                    f"Found more than one unique shape in the tensors to be "
                    f"stacked ({shapes}). This is likely due to a modification "
                    f"of one of the stacked TensorDicts, where a key has been "
                    f"updated/created with an uncompatible shape. If the entries "
                    f"are intended to have a different shape, use the get_nestedtensor "
                    f"method instead."
                )
            else:
                raise err

    def _get_tuple(self, key, default):
        first = self._get_str(key[0], None)
        if first is None:
            return self._default_get(first, default)
        if len(key) == 1:
            return first
        try:
            if isinstance(first, KeyedJaggedTensor):
                if len(key) != 2:
                    raise ValueError(f"Got too many keys for a KJT: {key}.")
                return first[key[-1]]
            else:
                return first._get_tuple(key[1:], default=default)
        except AttributeError as err:
            if "has no attribute" in str(err):
                raise ValueError(
                    f"Expected a TensorDictBase instance but got {type(first)} instead"
                    f" for key '{key[1:]}' in tensordict:\n{self}."
                )

    @cache  # noqa: B019
    def _add_batch_dim(self, *, in_dim, vmap_level):
        if self.is_memmap():
            td = torch.stack([td.cpu().as_tensor() for td in self.tensordicts], 0)
        else:
            td = self
        if in_dim < 0:
            in_dim = self.ndim + in_dim
        if in_dim == self.stack_dim:
            return self._cached_add_batch_dims(td, in_dim=in_dim, vmap_level=vmap_level)
        if in_dim < td.stack_dim:
            # then we'll stack along a dim before
            stack_dim = td.stack_dim - 1
        else:
            in_dim = in_dim - 1
            stack_dim = td.stack_dim
        tds = [
            td.apply(
                lambda _arg: _add_batch_dim(_arg, in_dim, vmap_level),
                batch_size=[b for i, b in enumerate(td.batch_size) if i != in_dim],
                names=[name for i, name in enumerate(td.names) if i != in_dim],
            )
            for td in td.tensordicts
        ]
        return LazyStackedTensorDict(*tds, stack_dim=stack_dim)

    @classmethod
    def _cached_add_batch_dims(cls, td, in_dim, vmap_level):
        # we return a stack with hook_out, and hack the batch_size and names
        # Per se it is still a LazyStack but the stacking dim is "hidden" from
        # the outside
        out = td.clone(False)

        def hook_out(tensor, in_dim=in_dim, vmap_level=vmap_level):
            return _add_batch_dim(tensor, in_dim, vmap_level)

        n = len(td.tensordicts)

        def hook_in(
            tensor,
            out_dim=in_dim,
            batch_size=n,
            vmap_level=vmap_level,
        ):
            return _remove_batch_dim(tensor, vmap_level, batch_size, out_dim)

        out.hook_out = hook_out
        out.hook_in = hook_in
        out._batch_size = torch.Size(
            [dim for i, dim in enumerate(out._batch_size) if i != out.stack_dim]
        )
        return out

    @cache  # noqa: B019
    def _remove_batch_dim(self, vmap_level, batch_size, out_dim):
        if self.hook_out is not None:
            # this is the hacked version. We just need to remove the hook_out and
            # reset a proper batch size
            return LazyStackedTensorDict(
                *self.tensordicts,
                stack_dim=out_dim,
            )
            # return self._cache_remove_batch_dim(vmap_level=vmap_level, batch_size=batch_size, out_dim=out_dim)
        else:
            # we must call _remove_batch_dim on all tensordicts
            # batch_size: size of the batch when we unhide it.
            # out_dim: dimension where the output will be found
            new_batch_size = list(self.batch_size)
            new_batch_size.insert(out_dim, batch_size)
            new_names = list(self.names)
            new_names.insert(out_dim, None)
            # rebuild the lazy stack
            # the stack dim is the same if the out_dim is past it, but it
            # must be incremented by one otherwise.
            # In the first case, the out_dim must be decremented by one
            if out_dim > self.stack_dim:
                stack_dim = self.stack_dim
                out_dim = out_dim - 1
            else:
                stack_dim = self.stack_dim + 1
            out = LazyStackedTensorDict(
                *[
                    td._remove_batch_dim(
                        vmap_level=vmap_level, batch_size=batch_size, out_dim=out_dim
                    )
                    for td in self.tensordicts
                ],
                stack_dim=stack_dim,
            )
        return out

    def get_nestedtensor(
        self,
        key: NestedKey,
        default: str | CompatibleType = NO_DEFAULT,
    ) -> CompatibleType:
        """Returns a nested tensor when stacking cannot be achieved.

        Args:
            key (NestedKey): the entry to nest.
            default (Any, optiona): the default value to return in case the key
                isn't in all sub-tensordicts.

                .. note:: In case the default is a tensor, this method will attempt
                  the construction of a nestedtensor with it. Otherwise, the default
                  value will be returned.

        Examples:
            >>> td0 = TensorDict({"a": torch.zeros(4), "b": torch.zeros(4)}, [])
            >>> td1 = TensorDict({"a": torch.ones(5)}, [])
            >>> td = torch.stack([td0, td1], 0)
            >>> a = td.get_nestedtensor("a")
            >>> # using a tensor as default uses this default to build the nested tensor
            >>> b = td.get_nestedtensor("b", default=torch.ones(4))
            >>> assert (a == b).all()
            >>> # using anything else as default returns the default
            >>> b2 = td.get_nestedtensor("b", None)
            >>> assert b2 is None

        """
        # disallow getting nested tensor if the stacking dimension is not 0
        if self.stack_dim != 0:
            raise RuntimeError(
                "Because nested tensors can only be stacked along their first "
                "dimension, LazyStackedTensorDict.get_nestedtensor can only be called "
                "when the stack_dim is 0."
            )

        # we can handle the case where the key is a tuple of length 1
        key = _unravel_key_to_tuple(key)
        subkey = key[0]
        if len(key) > 1:
            tensordict = self.get(subkey, default)
            if tensordict is default:
                return default
            return tensordict.get_nestedtensor(key[1:], default=default)
        tensors = [td.get(subkey, default=default) for td in self.tensordicts]
        if not isinstance(default, torch.Tensor) and any(
            tensor is default for tensor in tensors
        ):
            # we don't stack but return the default
            return default
        return torch.nested.nested_tensor(tensors)

    def is_contiguous(self) -> bool:
        return False

    def contiguous(self) -> TensorDictBase:
        source = {key: value.contiguous() for key, value in self.items()}
        batch_size = self.batch_size
        device = self.device
        out = TensorDict(
            source=source,
            batch_size=batch_size,
            device=device,
            names=self.names,
            _run_checks=False,
        )
        return out

    def clone(self, recurse: bool = True) -> TensorDictBase:
        if recurse:
            # This could be optimized using copy but we must be careful with
            # metadata (_is_shared etc)
            out = LazyStackedTensorDict(
                *[td.clone() for td in self.tensordicts],
                stack_dim=self.stack_dim,
            )
        else:
            out = LazyStackedTensorDict(
                *[td.clone(recurse=False) for td in self.tensordicts],
                stack_dim=self.stack_dim,
            )
        if self._td_dim_name is not None:
            out._td_dim_name = self._td_dim_name
        return out

    def pin_memory(self) -> TensorDictBase:
        for td in self.tensordicts:
            td.pin_memory()
        return self

    def to(self, dest: DeviceType | type, **kwargs) -> TensorDictBase:
        if isinstance(dest, type) and issubclass(dest, TensorDictBase):
            if isinstance(self, dest):
                return self
            kwargs.update({"batch_size": self.batch_size})
            out = dest(source=self, **kwargs)
            # if self._td_dim_name is not None:
            # TODO: define a _has_names util to quickly check and avoid this
            out.names = self.names
            return out
        elif isinstance(dest, (torch.device, str, int)):
            dest = torch.device(dest)
            if self.device is not None and dest == self.device:
                return self
            td = self.to_tensordict().to(dest, **kwargs)
            return td

        elif isinstance(dest, torch.Size):
            self.batch_size = dest
        elif dest is None:
            return self
        else:
            raise NotImplementedError(
                f"dest must be a string, torch.device or a TensorDict "
                f"instance, {dest} not allowed"
            )

    def _check_new_batch_size(self, new_size: torch.Size) -> None:
        if len(new_size) <= self.stack_dim:
            raise RuntimeError(
                "Changing the batch_size of a LazyStackedTensorDicts can only "
                "be done with sizes that are at least as long as the "
                "stacking dimension."
            )
        super()._check_new_batch_size(new_size)

    def _change_batch_size(self, new_size: torch.Size) -> None:
        if not hasattr(self, "_orig_batch_size"):
            self._orig_batch_size = self.batch_size
        elif self._orig_batch_size == new_size:
            del self._orig_batch_size
        self._batch_size = new_size

    def keys(
        self, include_nested: bool = False, leaves_only: bool = False
    ) -> _LazyStackedTensorDictKeysView:
        keys = _LazyStackedTensorDictKeysView(
            self, include_nested=include_nested, leaves_only=leaves_only
        )
        return keys

    valid_keys = keys

    # def _iterate_over_keys(self) -> None:
    #     for key in self.tensordicts[0].keys():
    #         if all(key in td.keys() for td in self.tensordicts):
    #             yield key
    def _iterate_over_keys(self) -> None:
        # this is about 20x faster than the version above
        yield from self._key_list()

    @cache  # noqa: B019
    def _key_list(self):
        keys = set(self.tensordicts[0].keys())
        for td in self.tensordicts[1:]:
            keys = keys.intersection(td.keys())
        return sorted(keys, key=str)

    def entry_class(self, key: NestedKey) -> type:
        data_type = type(self.tensordicts[0].get(key))
        if _is_tensor_collection(data_type):
            return LazyStackedTensorDict
        return data_type

    def apply_(self, fn: Callable, *others):
        for i, td in enumerate(self.tensordicts):
            idx = (slice(None),) * self.stack_dim + (i,)
            td.apply_(fn, *[other[idx] for other in others])
        return self

    def apply(
        self,
        fn: Callable,
        *others: TensorDictBase,
        batch_size: Sequence[int] | None = None,
        device: torch.device | None = None,
        names: Sequence[str] | None = None,
        inplace: bool = False,
        **constructor_kwargs,
    ) -> TensorDictBase:
        if inplace:
            if any(arg for arg in (batch_size, device, names, constructor_kwargs)):
                raise ValueError(
                    "Cannot pass other arguments to LazyStackedTensorDict.apply when inplace=True."
                )
            return self.apply_(fn, *others)
        else:
            if batch_size is not None:
                return super().apply(
                    fn,
                    *others,
                    batch_size=batch_size,
                    device=device,
                    names=names,
                    **constructor_kwargs,
                )
            others = (other.unbind(self.stack_dim) for other in others)
            out = LazyStackedTensorDict(
                *(
                    td.apply(fn, *oth, device=device)
                    for td, *oth in zip(self.tensordicts, *others)
                ),
                stack_dim=self.stack_dim,
            )
            if names is not None:
                out.names = names
            return out

    def select(
        self, *keys: str, inplace: bool = False, strict: bool = False
    ) -> LazyStackedTensorDict:
        # the following implementation keeps the hidden keys in the tensordicts
        tensordicts = [
            td.select(*keys, inplace=inplace, strict=strict) for td in self.tensordicts
        ]
        if inplace:
            return self
        return LazyStackedTensorDict(*tensordicts, stack_dim=self.stack_dim)

    def exclude(self, *keys: str, inplace: bool = False) -> LazyStackedTensorDict:
        tensordicts = [
            tensordict.exclude(*keys, inplace=inplace)
            for tensordict in self.tensordicts
        ]
        if inplace:
            self.tensordicts = tensordicts
            return self
        return torch.stack(tensordicts, dim=self.stack_dim)

    def __setitem__(self, item: IndexType, value: TensorDictBase) -> TensorDictBase:
        if isinstance(item, (list, range)):
            item = torch.tensor(item, device=self.device)
        if isinstance(item, tuple) and any(
            isinstance(sub_index, (list, range)) for sub_index in item
        ):
            item = tuple(
                torch.tensor(sub_index, device=self.device)
                if isinstance(sub_index, (list, range))
                else sub_index
                for sub_index in item
            )
        if (isinstance(item, Tensor) and item.dtype is torch.bool) or (
            isinstance(item, tuple)
            and any(
                isinstance(_item, Tensor) and _item.dtype is torch.bool
                for _item in item
            )
        ):
            raise RuntimeError(
                "setting values to a LazyStackTensorDict using boolean values is not supported yet. "
                "If this feature is needed, feel free to raise an issue on github."
            )
        if isinstance(item, Tensor):
            # e.g. item.shape = [1, 2, 3] and stack_dim == 2
            if item.ndimension() >= self.stack_dim + 1:
                items = item.unbind(self.stack_dim)
                values = value.unbind(self.stack_dim)
                for td, _item, sub_td in zip(self.tensordicts, items, values):
                    td[_item] = sub_td
            else:
                values = value.unbind(self.stack_dim)
                for td, sub_td in zip(self.tensordicts, values):
                    td[item] = sub_td
            return self
        return super().__setitem__(item, value)

    def __contains__(self, item: IndexType) -> bool:
        if isinstance(item, TensorDictBase):
            return any(item is td for td in self.tensordicts)
        return super().__contains__(item)

    def __getitem__(self, index: IndexType) -> TensorDictBase:
        if isinstance(index, (tuple, str)):
            index_key = _unravel_key_to_tuple(index)
            if index_key:
                return self._get_tuple(index_key, NO_DEFAULT)

        index_dict = _convert_index_lazystack(index, self.stack_dim, self.batch_size)
        if index_dict is None:
            # the we use a sub-tensordict
            return self.get_sub_tensordict(index)
        td_index = index_dict["remaining_index"]
        stack_index = index_dict["stack_index"]
        new_stack_dim = index_dict["new_stack_dim"]
        if new_stack_dim is not None:
            if isinstance(stack_index, slice):
                # we can't iterate but we can index the list directly
                out = LazyStackedTensorDict(
                    *[td[td_index] for td in self.tensordicts[stack_index]],
                    stack_dim=new_stack_dim,
                )
            elif isinstance(stack_index, (list, range)):
                # then we can iterate
                out = LazyStackedTensorDict(
                    *[self.tensordicts[idx][td_index] for idx in stack_index],
                    stack_dim=new_stack_dim,
                )
            elif isinstance(stack_index, Tensor):
                # td_index is a nested tuple that mimics the shape of stack_index
                def _nested_stack(t: list, stack_idx: Tensor, td_index):
                    if stack_idx.ndim:
                        assert len(td_index) == len(stack_idx)
                        out = LazyStackedTensorDict(
                            *[
                                _nested_stack(t, _idx, td_index[i])
                                for i, _idx in enumerate(stack_idx.unbind(0))
                            ],
                            stack_dim=new_stack_dim,
                        )
                        return out
                    return t[stack_idx][td_index]

                # print(index, td_index, stack_index)
                out = _nested_stack(self.tensordicts, stack_index, td_index)
            else:
                raise TypeError("Invalid index used for stack dimension.")
            out._td_dim_name = self._td_dim_name
            return out
        return self.tensordicts[stack_index][td_index]

    def __eq__(self, other):

        if is_tensorclass(other):
            return other == self
        if isinstance(other, (dict,)) or _is_tensor_collection(other.__class__):
            if (
                isinstance(other, LazyStackedTensorDict)
                and other.stack_dim == self.stack_dim
            ):
                if self.shape != other.shape:
                    raise RuntimeError(
                        "Cannot compare LazyStackedTensorDict instances of different shape."
                    )
                # in this case, we iterate over the tensordicts
                return torch.stack(
                    [
                        td1 == td2
                        for td1, td2 in zip(self.tensordicts, other.tensordicts)
                    ],
                    self.stack_dim,
                )
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                raise KeyError(f"keys in tensordicts mismatch, got {keys1} and {keys2}")
            d = {}
            for key, item1 in self.items():
                d[key] = item1 == other.get(key)
            return TensorDict(batch_size=self.batch_size, source=d, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return torch.stack(
                [td == other for td in self.tensordicts],
                self.stack_dim,
            )
        return False

    def __ne__(self, other):

        if is_tensorclass(other):
            return other != self
        if isinstance(other, (dict,)) or _is_tensor_collection(other.__class__):
            if (
                isinstance(other, LazyStackedTensorDict)
                and other.stack_dim == self.stack_dim
            ):
                if self.shape != other.shape:
                    raise RuntimeError(
                        "Cannot compare LazyStackedTensorDict instances of different shape."
                    )
                # in this case, we iterate over the tensordicts
                return torch.stack(
                    [
                        td1 != td2
                        for td1, td2 in zip(self.tensordicts, other.tensordicts)
                    ],
                    self.stack_dim,
                )
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                raise KeyError(f"keys in tensordicts mismatch, got {keys1} and {keys2}")
            d = {}
            for key, item1 in self.items():
                d[key] = item1 != other.get(key)
            return TensorDict(batch_size=self.batch_size, source=d, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return torch.stack(
                [td != other for td in self.tensordicts],
                self.stack_dim,
            )
        return True

    def all(self, dim: int = None) -> bool | TensorDictBase:
        if dim is not None and (dim >= self.batch_dims or dim < -self.batch_dims):
            raise RuntimeError(
                "dim must be greater than or equal to -tensordict.batch_dims and "
                "smaller than tensordict.batch_dims"
            )
        if dim is not None:
            # TODO: we need to adapt this to LazyStackedTensorDict too
            if dim < 0:
                dim = self.batch_dims + dim
            return TensorDict(
                source={key: value.all(dim=dim) for key, value in self.items()},
                batch_size=[b for i, b in enumerate(self.batch_size) if i != dim],
                device=self.device,
            )
        return all(value.all() for value in self.tensordicts)

    def any(self, dim: int = None) -> bool | TensorDictBase:
        if dim is not None and (dim >= self.batch_dims or dim < -self.batch_dims):
            raise RuntimeError(
                "dim must be greater than or equal to -tensordict.batch_dims and "
                "smaller than tensordict.batch_dims"
            )
        if dim is not None:
            # TODO: we need to adapt this to LazyStackedTensorDict too
            if dim < 0:
                dim = self.batch_dims + dim
            return TensorDict(
                source={key: value.any(dim=dim) for key, value in self.items()},
                batch_size=[b for i, b in enumerate(self.batch_size) if i != dim],
                device=self.device,
            )
        return any(value.any() for value in self.tensordicts)

    def _send(self, dst: int, _tag: int = -1, pseudo_rand: bool = False) -> int:
        for td in self.tensordicts:
            _tag = td._send(dst, _tag=_tag, pseudo_rand=pseudo_rand)
        return _tag

    def _isend(
        self,
        dst: int,
        _tag: int = -1,
        _futures: list[torch.Future] | None = None,
        pseudo_rand: bool = False,
    ) -> int:

        if _futures is None:
            is_root = True
            _futures = []
        else:
            is_root = False
        for td in self.tensordicts:
            _tag = td._isend(dst, _tag=_tag, pseudo_rand=pseudo_rand, _futures=_futures)
        if is_root:
            for future in _futures:
                future.wait()
        return _tag

    def _recv(self, src: int, _tag: int = -1, pseudo_rand: bool = False) -> int:
        for td in self.tensordicts:
            _tag = td._recv(src, _tag=_tag, pseudo_rand=pseudo_rand)
        return _tag

    def _irecv(
        self,
        src: int,
        return_premature: bool = False,
        _tag: int = -1,
        _future_list: list[torch.Future] = None,
        pseudo_rand: bool = False,
    ) -> tuple[int, list[torch.Future]] | list[torch.Future] | None:
        root = False
        if _future_list is None:
            _future_list = []
            root = True
        for td in self.tensordicts:
            _tag, _future_list = td._irecv(
                src=src,
                return_premature=return_premature,
                _tag=_tag,
                _future_list=_future_list,
                pseudo_rand=pseudo_rand,
            )

        if not root:
            return _tag, _future_list
        elif return_premature:
            return _future_list
        else:
            for future in _future_list:
                future.wait()
            return

    def del_(self, key: str, **kwargs: Any) -> TensorDictBase:
        ids = set()
        cur_len = len(ids)
        is_deleted = False
        error = None
        for td in self.tensordicts:
            # checking that the td has not been processed yet.
            # It could be that not all sub-tensordicts have the appropriate
            # entry but one must have it (or an error is thrown).
            tdid = id(td)
            ids.add(tdid)
            new_cur_len = len(ids)
            if new_cur_len == cur_len:
                continue
            cur_len = new_cur_len
            try:
                td.del_(key, **kwargs)
                is_deleted = True
            except KeyError as err:
                error = err
                continue
        if not is_deleted:
            # we know err is defined because LazyStackedTensorDict cannot be empty
            raise error
        return self

    def pop(
        self, key: NestedKey, default: str | CompatibleType = NO_DEFAULT
    ) -> CompatibleType:

        # using try/except for get/del is suboptimal, but
        # this is faster that checkink if key in self keys
        key = _unravel_key_to_tuple(key)
        if len(key) == 1:
            key = key[0]
        present = False
        if isinstance(key, tuple):
            if key in self.keys(True):
                present = True
                value = self._get_tuple(key, NO_DEFAULT)
        elif key in self.keys():
            present = True
            value = self._get_str(key, NO_DEFAULT)
        if present:
            self.del_(key)
        elif default is not NO_DEFAULT:
            value = default
        else:
            raise KeyError(
                f"You are trying to pop key `{key}` which is not in dict "
                f"without providing default value."
            )
        return value

    def share_memory_(self) -> TensorDictBase:
        for td in self.tensordicts:
            td.share_memory_()
        self._is_shared = True
        self.lock_()
        return self

    def detach_(self) -> TensorDictBase:
        for td in self.tensordicts:
            td.detach_()
        return self

    def memmap_(
        self, prefix: str | None = None, copy_existing: bool = False
    ) -> TensorDictBase:
        if prefix is not None:
            prefix = Path(prefix)
            if not prefix.exists():
                os.makedirs(prefix, exist_ok=True)
            torch.save({"stack_dim": self.stack_dim}, prefix / "meta.pt")
        for i, td in enumerate(self.tensordicts):
            td.memmap_(
                prefix=(prefix / str(i)) if prefix is not None else None,
                copy_existing=copy_existing,
            )
        self._is_memmap = True
        self.lock_()
        return self

    def memmap_like(
        self,
        prefix: str | None = None,
    ) -> TensorDictBase:
        tds = []
        if prefix is not None:
            prefix = Path(prefix)
            if not prefix.exists():
                os.makedirs(prefix, exist_ok=True)
            torch.save({"stack_dim": self.stack_dim}, prefix / "meta.pt")
        for i, td in enumerate(self.tensordicts):
            td_like = td.memmap_like(
                prefix=(prefix / str(i)) if prefix is not None else None,
            )
            tds.append(td_like)
        td_out = torch.stack(tds, self.stack_dim)
        td_out._is_memmap = True
        td_out.lock_()
        return td_out

    @classmethod
    def load_memmap(cls, prefix: str) -> LazyStackedTensorDict:
        prefix = Path(prefix)
        tensordicts = []
        i = 0
        while (prefix / str(i)).exists():
            tensordicts.append(TensorDict.load_memmap(prefix / str(i)))
            i += 1

        metadata = torch.load(prefix / "meta.pt")
        return cls(*tensordicts, stack_dim=metadata["stack_dim"])

    def expand(self, *shape: int, inplace: bool = False) -> TensorDictBase:
        if len(shape) == 1 and isinstance(shape[0], Sequence):
            shape = tuple(shape[0])
        stack_dim = len(shape) + self.stack_dim - self.ndimension()
        new_shape_tensordicts = [v for i, v in enumerate(shape) if i != stack_dim]
        tensordicts = [td.expand(*new_shape_tensordicts) for td in self.tensordicts]
        if inplace:
            self.tensordicts = tensordicts
            self.stack_dim = stack_dim
            return self
        return torch.stack(tensordicts, stack_dim)

    def update(
        self, input_dict_or_td: TensorDictBase, clone: bool = False, **kwargs: Any
    ) -> TensorDictBase:
        if input_dict_or_td is self:
            # no op
            return self

        if (
            isinstance(input_dict_or_td, LazyStackedTensorDict)
            and input_dict_or_td.stack_dim == self.stack_dim
        ):
            if not input_dict_or_td.shape[self.stack_dim] == len(self.tensordicts):
                raise ValueError(
                    "cannot update stacked tensordicts with different shapes."
                )
            for td_dest, td_source in zip(
                self.tensordicts, input_dict_or_td.tensordicts
            ):
                td_dest.update(td_source)
            return self

        keys = self.keys(False)
        for key, value in input_dict_or_td.items():
            if clone and hasattr(value, "clone"):
                value = value.clone()
            else:
                value = tree_map(torch.clone, value)
            if isinstance(key, tuple):
                key, subkey = key[0], key[1:]
            else:
                subkey = ()
            # the key must be a string by now. Let's check if it is present
            if key in keys:
                target_class = self.entry_class(key)
                if _is_tensor_collection(target_class):
                    if isinstance(value, dict):
                        value_unbind = TensorDict(
                            value, self.batch_size, _run_checks=False
                        ).unbind(self.stack_dim)
                    else:
                        value_unbind = value.unbind(self.stack_dim)
                    for t, _value in zip(self.tensordicts, value_unbind):
                        if len(subkey):
                            t.update({key: {subkey: _value}})
                        else:
                            t.update({key: _value})
                    continue
            if len(subkey):
                self.set((key, *subkey), value, **kwargs)
            else:
                self.set(key, value, **kwargs)
        return self

    def update_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | TensorDictBase,
        clone: bool = False,
        **kwargs: Any,
    ) -> TensorDictBase:
        if input_dict_or_td is self:
            # no op
            return self
        if (
            isinstance(input_dict_or_td, LazyStackedTensorDict)
            and input_dict_or_td.stack_dim == self.stack_dim
        ):
            if not input_dict_or_td.shape[self.stack_dim] == len(self.tensordicts):
                raise ValueError(
                    "cannot update stacked tensordicts with different shapes."
                )
            for td_dest, td_source in zip(
                self.tensordicts, input_dict_or_td.tensordicts
            ):
                td_dest.update_(td_source)
            return self
        for key, value in input_dict_or_td.items():
            if not isinstance(value, tuple(_ACCEPTED_CLASSES)):
                raise TypeError(
                    f"Expected value to be one of types {_ACCEPTED_CLASSES} "
                    f"but got {type(value)}"
                )
            if clone:
                value = value.clone()
            self.set_(key, value, **kwargs)
        return self

    def rename_key_(
        self, old_key: str, new_key: str, safe: bool = False
    ) -> TensorDictBase:
        def sort_keys(element):
            if isinstance(element, tuple):
                return "_-|-_".join(element)
            return element

        for td in self.tensordicts:
            td.rename_key_(old_key, new_key, safe=safe)
        return self

    rename_key = _renamed_inplace_method(rename_key_)

    def masked_fill_(self, mask: Tensor, value: float | bool) -> TensorDictBase:
        mask_unbind = mask.unbind(dim=self.stack_dim)
        for _mask, td in zip(mask_unbind, self.tensordicts):
            td.masked_fill_(_mask, value)
        return self

    def masked_fill(self, mask: Tensor, value: float | bool) -> TensorDictBase:
        td_copy = self.clone()
        return td_copy.masked_fill_(mask, value)

    @lock_blocked
    def insert(self, index: int, tensordict: TensorDictBase) -> None:
        """Insert a TensorDict into the stack at the specified index.

        Analogous to list.insert. The inserted TensorDict must have compatible
        batch_size and device. Insertion is in-place, nothing is returned.

        Args:
            index (int): The index at which the new TensorDict should be inserted.
            tensordict (TensorDictBase): The TensorDict to be inserted into the stack.

        """
        if not isinstance(tensordict, TensorDictBase):
            raise TypeError(
                "Expected new value to be TensorDictBase instance but got "
                f"{type(tensordict)} instead."
            )

        batch_size = self.tensordicts[0].batch_size
        device = self.tensordicts[0].device

        _batch_size = tensordict.batch_size
        _device = tensordict.device

        if device != _device:
            raise ValueError(
                f"Devices differ: stack has device={device}, new value has "
                f"device={_device}."
            )
        if _batch_size != batch_size:
            raise ValueError(
                f"Batch sizes in tensordicts differs: stack has "
                f"batch_size={batch_size}, new_value has batch_size={_batch_size}."
            )

        self.tensordicts.insert(index, tensordict)

        N = len(self.tensordicts)
        self._batch_size = self._compute_batch_size(batch_size, self.stack_dim, N)

    @lock_blocked
    def append(self, tensordict: TensorDictBase) -> None:
        """Append a TensorDict onto the stack.

        Analogous to list.append. The appended TensorDict must have compatible
        batch_size and device. The append operation is in-place, nothing is returned.

        Args:
            tensordict (TensorDictBase): The TensorDict to be appended onto the stack.

        """
        self.insert(len(self.tensordicts), tensordict)

    @property
    def is_locked(self) -> bool:
        if self._is_locked is not None:
            # if tensordicts have been locked through this Lazy stack, then we can
            # trust this lazy stack to contain the info.
            # In all other cases we must check
            return self._is_locked
        # If any of the tensordicts is not locked, we assume that the lazy stack
        # is not locked either. Caching is then disabled and
        for td in self.tensordicts:
            if not td.is_locked:
                return False
        else:
            # In this case, all tensordicts were locked before the lazy stack
            # was created and they were not locked through the lazy stack.
            # This means we cannot cache the value because this lazy stack
            # if not part of the graph. We don't want it to be part of the graph
            # because this object being locked is only a side-effect.
            # Calling self.lock_() here could however speed things up.
            return True

    @is_locked.setter
    def is_locked(self, value: bool) -> None:
        if value:
            self.lock_()
        else:
            self.unlock_()

    @property
    def _lock_id(self):
        """Ids of all tensordicts that need to be unlocked for this to be unlocked."""
        _lock_id = set()
        for tensordict in self.tensordicts:
            _lock_id = _lock_id.union(tensordict._lock_id)
        _lock_id = _lock_id - {id(self)}
        return _lock_id

    def _lock_propagate(self, lock_ids=None):
        """Registers the parent tensordict that handles the lock."""
        self._is_locked = True
        _locked_tensordicts = []
        is_root = lock_ids is None
        if is_root:
            lock_ids = set()
        lock_ids = lock_ids.union({id(self)})
        for dest in self.tensordicts:
            dest._lock_propagate(lock_ids)
            _locked_tensordicts.append(dest)

    def _remove_lock(self, lock_id):
        for td in self.tensordicts:
            td._remove_lock(lock_id)

    @erase_cache
    def _propagate_unlock(self, lock_ids=None):
        # we can't set _is_locked to False because after it's unlocked, anything
        # can happen to a child tensordict.
        self._is_locked = None
        if lock_ids is None:
            lock_ids = set()

        unlocked_tds = [self]
        lock_ids.add(id(self))
        for dest in self.tensordicts:
            unlocked_tds.extend(dest._propagate_unlock(lock_ids))

        self._is_shared = False
        self._is_memmap = False
        return unlocked_tds

    def __del__(self):
        if self._is_locked is None:
            # then we can reliably say that this lazy stack is not part of
            # the tensordicts graphs
            return
        # this can be a perf bottleneck
        for td in self.tensordicts:
            td._remove_lock(id(self))

    lock_ = TensorDictBase.lock_
    lock = _renamed_inplace_method(lock_)

    unlock_ = TensorDictBase.unlock_
    unlock = _renamed_inplace_method(unlock_)


class _CustomOpTensorDict(TensorDictBase):
    """Encodes lazy operations on tensors contained in a TensorDict."""

    def __new__(cls, *args: Any, **kwargs: Any) -> _CustomOpTensorDict:
        return super().__new__(cls, *args, _safe=False, _lazy=True, **kwargs)

    def __init__(
        self,
        source: TensorDictBase,
        custom_op: str,
        inv_op: str | None = None,
        custom_op_kwargs: dict | None = None,
        inv_op_kwargs: dict | None = None,
        batch_size: Sequence[int] | None = None,
    ) -> None:
        self._is_shared = source.is_shared()
        self._is_memmap = source.is_memmap()

        if not isinstance(source, TensorDictBase):
            raise TypeError(
                f"Expected source to be a TensorDictBase isntance, "
                f"but got {type(source)} instead."
            )
        self._source = source
        self.custom_op = custom_op
        self.inv_op = inv_op
        self.custom_op_kwargs = custom_op_kwargs if custom_op_kwargs is not None else {}
        self.inv_op_kwargs = inv_op_kwargs if inv_op_kwargs is not None else {}
        self._batch_size = None
        if batch_size is not None and batch_size != self.batch_size:
            raise RuntimeError("batch_size does not match self.batch_size.")

    def _update_custom_op_kwargs(self, source_tensor: Tensor) -> dict[str, Any]:
        """Allows for a transformation to be customized for a certain shape, device or dtype.

        By default, this is a no-op on self.custom_op_kwargs

        Args:
            source_tensor: corresponding Tensor

        Returns:
            a dictionary with the kwargs of the operation to execute
            for the tensor

        """
        return self.custom_op_kwargs

    def _update_inv_op_kwargs(self, source_tensor: Tensor) -> dict[str, Any]:
        """Allows for an inverse transformation to be customized for a certain shape, device or dtype.

        By default, this is a no-op on self.inv_op_kwargs

        Args:
            source_tensor: corresponding tensor

        Returns:
            a dictionary with the kwargs of the operation to execute for
            the tensor

        """
        return self.inv_op_kwargs

    def entry_class(self, key: NestedKey) -> type:
        return type(self._source.get(key))

    @property
    def device(self) -> torch.device | None:
        return self._source.device

    @device.setter
    def device(self, value: DeviceType) -> None:
        self._source.device = value

    @property
    def batch_size(self) -> torch.Size:
        if self._batch_size is None:
            self._batch_size = getattr(
                torch.zeros(self._source.batch_size, device="meta"), self.custom_op
            )(**self.custom_op_kwargs).shape
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_size: torch.Size) -> None:
        self._batch_size_setter(new_size)

    def _has_names(self):
        return self._source._has_names()

    def _erase_names(self):
        raise RuntimeError(
            f"Cannot erase names of a {type(self)}. "
            f"Erase source TensorDict's names instead."
        )

    def _rename_subtds(self, names):
        for key in self.keys():
            if _is_tensor_collection(self.entry_class(key)):
                raise RuntimeError(
                    "Cannot rename dimensions of a lazy TensorDict with "
                    "nested collections. Convert the instance to a regular "
                    "tensordict by using the `to_tensordict()` method first."
                )

    def _change_batch_size(self, new_size: torch.Size) -> None:
        if not hasattr(self, "_orig_batch_size"):
            self._orig_batch_size = self.batch_size
        elif self._orig_batch_size == new_size:
            del self._orig_batch_size
        self._batch_size = new_size

    def _get_str(self, key, default):
        tensor = self._source._get_str(key, default)
        if tensor is default:
            return tensor
        return self._transform_value(tensor)

    def _get_tuple(self, key, default):
        tensor = self._source._get_tuple(key, default)
        if tensor is default:
            return tensor
        return self._transform_value(tensor)

    def _transform_value(self, item):
        return getattr(item, self.custom_op)(**self._update_custom_op_kwargs(item))

    def _set_str(self, key, value, *, inplace: bool, validated: bool):
        if not validated:
            value = self._validate_value(value, check_shape=True)
            validated = True
        value = getattr(value, self.inv_op)(**self._update_inv_op_kwargs(value))
        self._source._set_str(key, value, inplace=inplace, validated=validated)
        return self

    def _set_tuple(self, key, value, *, inplace: bool, validated: bool):
        if len(key) == 1:
            return self._set_str(key[0], value, inplace=inplace, validated=validated)
        source = self._source._get_str(key[0], None)
        if source is None:
            self._source._create_nested_str(key[0])
            source = self._source._get_str(key[0], NO_DEFAULT)
        type(self)(
            source,
            custom_op=self.custom_op,
            inv_op=self.inv_op,
            custom_op_kwargs=self._update_custom_op_kwargs(source),
            inv_op_kwargs=self._update_inv_op_kwargs(source),
        )._set_tuple(key[1:], value, inplace=inplace, validated=validated)
        return self

    def _set_at_str(self, key, value, idx, *, validated):
        transformed_tensor, original_tensor = self._get_str(
            key, NO_DEFAULT
        ), self._source._get_str(key, NO_DEFAULT)
        if transformed_tensor.data_ptr() != original_tensor.data_ptr():
            raise RuntimeError(
                f"{self} original tensor and transformed_in do not point to the "
                f"same storage. Setting values in place is not currently "
                f"supported in this setting, consider calling "
                f"`td.clone()` before `td.set_at_(...)`"
            )
        transformed_tensor[idx] = value
        return self

    def _set_at_tuple(self, key, value, idx, *, validated):
        transformed_tensor, original_tensor = self._get_tuple(
            key, NO_DEFAULT
        ), self._source._get_tuple(key, NO_DEFAULT)
        if transformed_tensor.data_ptr() != original_tensor.data_ptr():
            raise RuntimeError(
                f"{self} original tensor and transformed_in do not point to the "
                f"same storage. Setting values in place is not currently "
                f"supported in this setting, consider calling "
                f"`td.clone()` before `td.set_at_(...)`"
            )
        if not validated:
            value = self._validate_value(value, check_shape=False)

        transformed_tensor[idx] = value
        return self

    def _stack_onto_(
        self,
        key: str,
        list_item: list[CompatibleType],
        dim: int,
    ) -> TensorDictBase:
        raise RuntimeError(
            f"stacking tensordicts is not allowed for type {type(self)}"
            f"consider calling 'to_tensordict()` first"
        )

    def __repr__(self) -> str:
        custom_op_kwargs_str = ", ".join(
            [f"{key}={value}" for key, value in self.custom_op_kwargs.items()]
        )
        indented_source = textwrap.indent(f"source={self._source}", "\t")
        return (
            f"{self.__class__.__name__}(\n{indented_source}, "
            f"\n\top={self.custom_op}({custom_op_kwargs_str}))"
        )

    # @cache  # noqa: B019
    def keys(
        self, include_nested: bool = False, leaves_only: bool = False
    ) -> _TensorDictKeysView:
        return self._source.keys(include_nested=include_nested, leaves_only=leaves_only)

    def select(
        self, *keys: str, inplace: bool = False, strict: bool = True
    ) -> _CustomOpTensorDict:
        if inplace:
            self._source.select(*keys, inplace=inplace, strict=strict)
            return self
        self_copy = copy(self)
        self_copy._source = self_copy._source.select(*keys, strict=strict)
        return self_copy

    def exclude(self, *keys: str, inplace: bool = False) -> TensorDictBase:
        if inplace:
            return super().exclude(*keys, inplace=True)
        return TensorDict(
            {key: value.clone() for key, value in self.items()},
            batch_size=self.batch_size,
            device=self.device,
            _run_checks=False,
            _is_memmap=self.is_memmap(),
            _is_shared=self.is_shared(),
        ).exclude(*keys, inplace=True)

    def clone(self, recurse: bool = True) -> TensorDictBase:
        """Clones the Lazy TensorDict.

        Args:
            recurse (bool, optional): if ``True`` (default), a regular
                :class:`TensorDict` instance will be returned.
                Otherwise, another :class:`SubTensorDict` with identical content
                will be returned.
        """
        if not recurse:
            return type(self)(
                source=self._source.clone(False),
                custom_op=self.custom_op,
                inv_op=self.inv_op,
                custom_op_kwargs=self.custom_op_kwargs,
                inv_op_kwargs=self.inv_op_kwargs,
                batch_size=self.batch_size,
            )
        return self.to_tensordict()

    def is_contiguous(self) -> bool:
        return all([value.is_contiguous() for _, value in self.items()])

    def contiguous(self) -> TensorDictBase:
        if self.is_contiguous():
            return self
        return self.to(TensorDict)

    def rename_key_(
        self, old_key: str, new_key: str, safe: bool = False
    ) -> _CustomOpTensorDict:
        self._source.rename_key_(old_key, new_key, safe=safe)
        return self

    rename_key = _renamed_inplace_method(rename_key_)

    def del_(self, key: str) -> _CustomOpTensorDict:
        self._source = self._source.del_(key)
        return self

    def to(self, dest: DeviceType | type, **kwargs) -> TensorDictBase:
        if isinstance(dest, type) and issubclass(dest, TensorDictBase):
            if isinstance(self, dest):
                return self
            out = dest(source=self)
            if self._td_dim_names is not None:
                out.names = self._td_dim_names
            return out
        elif isinstance(dest, (torch.device, str, int)):
            if self.device is not None and torch.device(dest) == self.device:
                return self
            td = self._source.to(dest, **kwargs)
            self_copy = copy(self)
            self_copy._source = td
            return self_copy
        elif dest is None:
            return self
        else:
            raise NotImplementedError(
                f"dest must be a string, torch.device or a TensorDict "
                f"instance, {dest} not allowed"
            )

    def pin_memory(self) -> _CustomOpTensorDict:
        self._source.pin_memory()
        return self

    def detach_(self) -> _CustomOpTensorDict:
        self._source.detach_()
        return self

    def masked_fill_(self, mask: Tensor, value: float | bool) -> _CustomOpTensorDict:
        for key, item in self.items():
            val = self._source.get(key)
            mask_exp = expand_right(
                mask, list(mask.shape) + list(val.shape[self._source.batch_dims :])
            )
            mask_proc_inv = getattr(mask_exp, self.inv_op)(
                **self._update_inv_op_kwargs(item)
            )
            val[mask_proc_inv] = value
            self._source.set(key, val)
        return self

    def masked_fill(self, mask: Tensor, value: float | bool) -> TensorDictBase:
        td_copy = self.clone()
        return td_copy.masked_fill_(mask, value)

    def memmap_(
        self, prefix: str | None = None, copy_existing: bool = False
    ) -> _CustomOpTensorDict:
        self._source.memmap_(prefix=prefix, copy_existing=copy_existing)
        if prefix is not None:
            prefix = Path(prefix)
            metadata = torch.load(prefix / "meta.pt")
            metadata["custom_op"] = self.custom_op
            metadata["inv_op"] = self.inv_op
            metadata["custom_op_kwargs"] = self.custom_op_kwargs
            metadata["inv_op_kwargs"] = self.inv_op_kwargs
            torch.save(metadata, prefix / "meta.pt")

        self._is_memmap = True
        self.lock_()
        return self

    @classmethod
    def load_memmap(cls, prefix: str) -> _CustomOpTensorDict:
        prefix = Path(prefix)
        source = TensorDict.load_memmap(prefix)
        metadata = torch.load(prefix / "meta.pt")
        return cls(
            source,
            custom_op=metadata["custom_op"],
            inv_op=metadata["inv_op"],
            custom_op_kwargs=metadata["custom_op_kwargs"],
            inv_op_kwargs=metadata["inv_op_kwargs"],
        )

    def share_memory_(self) -> _CustomOpTensorDict:
        self._source.share_memory_()
        self._is_shared = True
        self.lock_()
        return self

    @property
    def _td_dim_names(self):
        # we also want for _td_dim_names to be accurate
        if self._source._td_dim_names is None:
            return None
        return self.names

    @property
    def is_locked(self) -> bool:
        return self._source.is_locked

    @is_locked.setter
    def is_locked(self, value) -> bool:
        if value:
            self.lock_()
        else:
            self.unlock_()

    @as_decorator("is_locked")
    def lock_(self) -> TensorDictBase:
        self._source.lock_()
        return self

    @erase_cache
    @as_decorator("is_locked")
    def unlock_(self) -> TensorDictBase:
        self._source.unlock_()
        return self

    def _remove_lock(self, lock_id):
        return self._source._remove_lock(lock_id)

    @erase_cache
    def _lock_propagate(self, lock_ids):
        return self._source._lock_propagate(lock_ids)

    lock = _renamed_inplace_method(lock_)
    unlock = _renamed_inplace_method(unlock_)

    def __del__(self):
        pass

    @property
    def sorted_keys(self):
        return self._source.sorted_keys


class _UnsqueezedTensorDict(_CustomOpTensorDict):
    """A lazy view on an unsqueezed TensorDict.

    When calling `tensordict.unsqueeze(dim)`, a lazy view of this operation is
    returned such that the following code snippet works without raising an
    exception:

        >>> assert tensordict.unsqueeze(dim).squeeze(dim) is tensordict

    Examples:
        >>> from tensordict import TensorDict
        >>> import torch
        >>> td = TensorDict({'a': torch.randn(3, 4)}, batch_size=[3])
        >>> td_unsqueeze = td.unsqueeze(-1)
        >>> print(td_unsqueeze.shape)
        torch.Size([3, 1])
        >>> print(td_unsqueeze.squeeze(-1) is td)
        True
    """

    def squeeze(self, dim: int | None) -> TensorDictBase:
        if dim is not None and dim < 0:
            dim = self.batch_dims + dim
        if dim == self.custom_op_kwargs.get("dim"):
            return self._source
        return super().squeeze(dim)

    def _stack_onto_(
        self,
        key: str,
        list_item: list[CompatibleType],
        dim: int,
    ) -> TensorDictBase:
        unsqueezed_dim = self.custom_op_kwargs["dim"]
        diff_to_apply = 1 if dim < unsqueezed_dim else 0
        list_item_unsqueeze = [
            item.squeeze(unsqueezed_dim - diff_to_apply) for item in list_item
        ]
        return self._source._stack_onto_(key, list_item_unsqueeze, dim)

    @property
    def names(self):
        names = copy(self._source.names)
        dim = self.custom_op_kwargs.get("dim")
        names.insert(dim, None)
        return names

    @names.setter
    def names(self, value):
        if value[: self.batch_dims] == self.names:
            return
        raise RuntimeError(
            "Names of a lazy tensordict cannot be modified. Call to_tensordict() first."
        )


class _SqueezedTensorDict(_CustomOpTensorDict):
    """A lazy view on a squeezed TensorDict.

    See the `UnsqueezedTensorDict` class documentation for more information.

    """

    def unsqueeze(self, dim: int) -> TensorDictBase:
        if dim < 0:
            dim = self.batch_dims + dim + 1
        inv_op_dim = self.inv_op_kwargs.get("dim")
        if inv_op_dim < 0:
            inv_op_dim = self.batch_dims + inv_op_dim + 1
        if dim == inv_op_dim:
            return self._source
        return super().unsqueeze(dim)

    def _stack_onto_(
        self,
        key: str,
        list_item: list[CompatibleType],
        dim: int,
    ) -> TensorDictBase:
        squeezed_dim = self.custom_op_kwargs["dim"]
        # dim=0, squeezed_dim=2, [3, 4, 5] [3, 4, 1, 5] [[4, 5], [4, 5], [4, 5]] => unsq 1
        # dim=1, squeezed_dim=2, [3, 4, 5] [3, 4, 1, 5] [[3, 5], [3, 5], [3, 5], [3, 4]] => unsq 1
        # dim=2, squeezed_dim=2, [3, 4, 5] [3, 4, 1, 5] [[3, 4], [3, 4], ...] => unsq 2
        diff_to_apply = 1 if dim < squeezed_dim else 0
        list_item_unsqueeze = [
            item.unsqueeze(squeezed_dim - diff_to_apply) for item in list_item
        ]
        return self._source._stack_onto_(key, list_item_unsqueeze, dim)

    @property
    def names(self):
        names = copy(self._source.names)
        dim = self.custom_op_kwargs["dim"]
        if self._source.batch_size[dim] == 1:
            del names[dim]
        return names

    @names.setter
    def names(self, value):
        if value[: self.batch_dims] == self.names:
            return
        raise RuntimeError(
            "Names of a lazy tensordict cannot be modified. Call to_tensordict() first."
        )


class _ViewedTensorDict(_CustomOpTensorDict):
    def _update_custom_op_kwargs(self, source_tensor: Tensor) -> dict[str, Any]:
        new_dim_list = list(self.custom_op_kwargs.get("size"))
        new_dim_list += list(source_tensor.shape[self._source.batch_dims :])
        new_dim = torch.Size(new_dim_list)
        new_dict = deepcopy(self.custom_op_kwargs)
        new_dict.update({"size": new_dim})
        return new_dict

    def _update_inv_op_kwargs(self, tensor: Tensor) -> dict:
        size = list(self.inv_op_kwargs.get("size"))
        size += list(_shape(tensor)[self.batch_dims :])
        new_dim = torch.Size(size)
        new_dict = deepcopy(self.inv_op_kwargs)
        new_dict.update({"size": new_dim})
        return new_dict

    def view(
        self, *shape: int, size: list | tuple | torch.Size | None = None
    ) -> TensorDictBase:
        if len(shape) == 0 and size is not None:
            return self.view(*size)
        elif len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
            return self.view(*shape[0])
        elif not isinstance(shape, torch.Size):
            shape = infer_size_impl(shape, self.numel())
            shape = torch.Size(shape)
        if shape == self._source.batch_size:
            return self._source
        return super().view(*shape)

    @property
    def names(self):
        return [None] * self.ndim

    @names.setter
    def names(self, value):
        raise RuntimeError(
            "Names of a lazy tensordict cannot be modified. Call to_tensordict() first."
        )


class _TransposedTensorDict(_CustomOpTensorDict):
    """A lazy view on a TensorDict with two batch dimensions transposed.

    When calling `tensordict.permute(dims_list, dim)`, a lazy view of this operation is
    returned such that the following code snippet works without raising an
    exception:

        >>> assert tensordict.transpose(dims_list, dim).transpose(dims_list, dim) is tensordict

    """

    def transpose(self, dim0, dim1) -> TensorDictBase:
        if dim0 < 0:
            dim0 = self.ndim + dim0
        if dim1 < 0:
            dim1 = self.ndim + dim1
        if any((dim0 < 0, dim1 < 0)):
            raise ValueError(
                "The provided dimensions are incompatible with the tensordict batch-size."
            )
        if dim0 == dim1:
            return self
        dims = (self.inv_op_kwargs.get("dim0"), self.inv_op_kwargs.get("dim1"))
        if dim0 in dims and dim1 in dims:
            return self._source
        return super().permute(dim0, dim1)

    def add_missing_dims(
        self, num_dims: int, batch_dims: tuple[int, ...]
    ) -> tuple[int, ...]:
        dim_diff = num_dims - len(batch_dims)
        all_dims = list(range(num_dims))
        for i, x in enumerate(batch_dims):
            if x < 0:
                x = x - dim_diff
            all_dims[i] = x
        return tuple(all_dims)

    def _update_custom_op_kwargs(self, source_tensor: Tensor) -> dict[str, Any]:
        return self.custom_op_kwargs

    def _update_inv_op_kwargs(self, tensor: Tensor) -> dict[str, Any]:
        return self.custom_op_kwargs

    def _stack_onto_(
        self,
        key: str,
        list_item: list[CompatibleType],
        dim: int,
    ) -> TensorDictBase:

        trsp = self.custom_op_kwargs["dim0"], self.custom_op_kwargs["dim1"]
        if dim == trsp[0]:
            dim = trsp[1]
        elif dim == trsp[1]:
            dim = trsp[0]

        list_permuted_items = []
        for item in list_item:
            list_permuted_items.append(item.transpose(*trsp))
        self._source._stack_onto_(key, list_permuted_items, dim)
        return self

    @property
    def names(self):
        names = copy(self._source.names)
        dim0 = self.custom_op_kwargs["dim0"]
        dim1 = self.custom_op_kwargs["dim1"]
        names = [
            names[dim0] if i == dim1 else names[dim1] if i == dim0 else name
            for i, name in enumerate(names)
        ]
        return names

    @names.setter
    def names(self, value):
        raise RuntimeError(
            "Names of a lazy tensordict cannot be modified. Call to_tensordict() first."
        )


class _PermutedTensorDict(_CustomOpTensorDict):
    """A lazy view on a TensorDict with the batch dimensions permuted.

    When calling `tensordict.permute(dims_list, dim)`, a lazy view of this operation is
    returned such that the following code snippet works without raising an
    exception:

        >>> assert tensordict.permute(dims_list, dim).permute(dims_list, dim) is tensordict

    Examples:
        >>> from tensordict import TensorDict
        >>> import torch
        >>> td = TensorDict({'a': torch.randn(4, 5, 6, 9)}, batch_size=[3])
        >>> td_permute = td.permute(dims=(2, 1, 0))
        >>> print(td_permute.shape)
        torch.Size([6, 5, 4])
        >>> print(td_permute.permute(dims=(2, 1, 0)) is td)
        True

    """

    def permute(
        self,
        *dims_list: int,
        dims: Sequence[int] | None = None,
    ) -> TensorDictBase:
        if len(dims_list) == 0:
            dims_list = dims
        elif len(dims_list) == 1 and not isinstance(dims_list[0], int):
            dims_list = dims_list[0]
        if len(dims_list) != len(self.shape):
            raise RuntimeError(
                f"number of dims don't match in permute (got {len(dims_list)}, expected {len(self.shape)}"
            )
        if not len(dims_list) and not self.batch_dims:
            return self
        if np.array_equal(dims_list, range(self.batch_dims)):
            return self
        if np.array_equal(np.argsort(dims_list), self.inv_op_kwargs.get("dims")):
            return self._source
        return super().permute(*dims_list)

    def add_missing_dims(
        self, num_dims: int, batch_dims: tuple[int, ...]
    ) -> tuple[int, ...]:
        # Adds the feature dimensions to the permute dims
        dim_diff = num_dims - len(batch_dims)
        all_dims = list(range(num_dims))
        for i, x in enumerate(batch_dims):
            if x < 0:
                x = x - dim_diff
            all_dims[i] = x
        return tuple(all_dims)

    def _update_custom_op_kwargs(self, source_tensor: Tensor) -> dict[str, Any]:
        new_dims = self.add_missing_dims(
            len(source_tensor.shape), self.custom_op_kwargs["dims"]
        )
        kwargs = deepcopy(self.custom_op_kwargs)
        kwargs.update({"dims": new_dims})
        return kwargs

    def _update_inv_op_kwargs(self, tensor: Tensor) -> dict[str, Any]:
        new_dims = self.add_missing_dims(
            self._source.batch_dims + len(_shape(tensor)[self.batch_dims :]),
            self.custom_op_kwargs["dims"],
        )
        kwargs = deepcopy(self.custom_op_kwargs)
        kwargs.update({"dims": tuple(np.argsort(new_dims))})
        return kwargs

    def _stack_onto_(
        self,
        key: str,
        list_item: list[CompatibleType],
        dim: int,
    ) -> TensorDictBase:
        permute_dims = self.custom_op_kwargs["dims"]
        inv_permute_dims = np.argsort(permute_dims)
        new_dim = [i for i, v in enumerate(inv_permute_dims) if v == dim][0]
        inv_permute_dims = [p for p in inv_permute_dims if p != dim]
        inv_permute_dims = np.argsort(np.argsort(inv_permute_dims))

        list_permuted_items = []
        for item in list_item:
            perm = list(inv_permute_dims) + list(
                range(self.batch_dims - 1, item.ndimension())
            )
            list_permuted_items.append(item.permute(*perm))
        self._source._stack_onto_(key, list_permuted_items, new_dim)
        return self

    @property
    def names(self):
        names = copy(self._source.names)
        return [names[i] for i in self.custom_op_kwargs["dims"]]

    @names.setter
    def names(self, value):
        if value[: self.batch_dims] == self.names:
            return
        raise RuntimeError(
            "Names of a lazy tensordict cannot be modified. Call to_tensordict() first."
        )


def _get_repr(tensor: Tensor) -> str:
    s = ", ".join(
        [
            f"shape={_shape(tensor)}",
            f"device={_device(tensor)}",
            f"dtype={_dtype(tensor)}",
            f"is_shared={_is_shared(tensor)}",
        ]
    )
    return f"{tensor.__class__.__name__}({s})"


def _get_repr_custom(cls, shape, device, dtype, is_shared) -> str:
    s = ", ".join(
        [
            f"shape={shape}",
            f"device={device}",
            f"dtype={dtype}",
            f"is_shared={is_shared}",
        ]
    )
    return f"{cls.__name__}({s})"


def _make_repr(key: str, item: CompatibleType, tensordict: TensorDictBase) -> str:
    if _is_tensor_collection(type(item)):
        return f"{key}: {repr(tensordict.get(key))}"
    return f"{key}: {_get_repr(item)}"


def _td_fields(td: TensorDictBase) -> str:
    strs = []
    for key in td.keys():
        try:
            item = td.get(key)
            strs.append(_make_repr(key, item, td))
        except RuntimeError as err:
            if re.match(r"Found more than one unique shape in the tensors", str(err)):
                # we know td is lazy stacked and the key is a leaf
                # so we can get the shape and escape the error
                shape = td.get_item_shape(key)
                tensor = td.tensordicts[0].get(key)
                if isinstance(tensor, TensorDictBase):
                    substr = _td_fields(tensor)
                else:
                    substr = _get_repr_custom(
                        tensor.__class__,
                        shape=shape,
                        device=tensor.device,
                        dtype=tensor.dtype,
                        is_shared=tensor.is_shared(),
                    )
                strs.append(f"{key}: {substr}")
            else:
                raise err

    return indent(
        "\n" + ",\n".join(sorted(strs)),
        4 * " ",
    )


def _check_keys(
    list_of_tensordicts: Sequence[TensorDictBase],
    strict: bool = False,
    include_nested: bool = False,
    leaves_only: bool = False,
) -> set[str]:
    if not len(list_of_tensordicts):
        return set()
    keys: set[str] = set(
        list_of_tensordicts[0].keys(
            include_nested=include_nested, leaves_only=leaves_only
        )
    )
    for td in list_of_tensordicts[1:]:
        k = td.keys(include_nested=include_nested, leaves_only=leaves_only)
        if not strict:
            keys = keys.intersection(k)
        else:
            if set(k) != keys:
                raise KeyError(
                    f"got keys {keys} and {set(td.keys())} which are " f"incompatible"
                )
    return keys


def _expand_to_match_shape(
    parent_batch_size: torch.Size,
    tensor: Tensor,
    self_batch_dims: int,
    self_device: DeviceType,
) -> Tensor | TensorDictBase:
    if hasattr(tensor, "dtype"):
        return torch.zeros(
            (
                *parent_batch_size,
                *_shape(tensor)[self_batch_dims:],
            ),
            dtype=tensor.dtype,
            device=self_device,
        )
    else:
        # tensordict
        out = TensorDict(
            {},
            [*parent_batch_size, *_shape(tensor)[self_batch_dims:]],
            device=self_device,
        )
        return out


def make_tensordict(
    input_dict: dict[str, CompatibleType] | None = None,
    batch_size: Sequence[int] | torch.Size | int | None = None,
    device: DeviceType | None = None,
    **kwargs: CompatibleType,  # source
) -> TensorDict:
    """Returns a TensorDict created from the keyword arguments or an input dictionary.

    If ``batch_size`` is not specified, returns the maximum batch size possible.

    This function works on nested dictionaries too, or can be used to determine the
    batch-size of a nested tensordict.

    Args:
        input_dict (dictionary, optional): a dictionary to use as a data source
            (nested keys compatible).
        **kwargs (TensorDict or torch.Tensor): keyword arguments as data source
            (incompatible with nested keys).
        batch_size (iterable of int, optional): a batch size for the tensordict.
        device (torch.device or compatible type, optional): a device for the TensorDict.

    Examples:
        >>> input_dict = {"a": torch.randn(3, 4), "b": torch.randn(3)}
        >>> print(make_tensordict(input_dict))
        TensorDict(
            fields={
                a: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                b: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> # alternatively
        >>> td = make_tensordict(**input_dict)
        >>> # nested dict: the nested TensorDict can have a different batch-size
        >>> # as long as its leading dims match.
        >>> input_dict = {"a": torch.randn(3), "b": {"c": torch.randn(3, 4)}}
        >>> print(make_tensordict(input_dict))
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
        >>> # we can also use this to work out the batch sie of a tensordict
        >>> input_td = TensorDict({"a": torch.randn(3), "b": {"c": torch.randn(3, 4)}}, [])
        >>> print(make_tensordict(input_td))
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
    if input_dict is not None:
        kwargs.update(input_dict)
    return TensorDict.from_dict(kwargs, batch_size=batch_size, device=device)


def _set_max_batch_size(source: TensorDictBase, batch_dims=None):
    """Updates a tensordict with its maximium batch size."""
    tensor_data = list(source.values())

    for val in tensor_data:
        if _is_tensor_collection(val.__class__):
            _set_max_batch_size(val, batch_dims=batch_dims)
    batch_size = []
    if not tensor_data:  # when source is empty
        source.batch_size = batch_size
        return
    curr_dim = 0
    while True:
        if tensor_data[0].dim() > curr_dim:
            curr_dim_size = tensor_data[0].size(curr_dim)
        else:
            source.batch_size = batch_size
            return
        for tensor in tensor_data[1:]:
            if tensor.dim() <= curr_dim or tensor.size(curr_dim) != curr_dim_size:
                source.batch_size = batch_size
                return
        if batch_dims is None or len(batch_size) < batch_dims:
            batch_size.append(curr_dim_size)
        curr_dim += 1


def _iter_items_lazystack(
    tensordict: LazyStackedTensorDict,
) -> Iterator[tuple[str, CompatibleType]]:
    return tensordict.items()


def _clone_value(value: CompatibleType, recurse: bool) -> CompatibleType:
    if recurse:
        return value.clone()
    elif _is_tensor_collection(value.__class__):
        return value.clone(recurse=False)
    else:
        return value


def _is_number(item):
    if isinstance(item, Number):
        return True
    if isinstance(item, Tensor) and item.ndim == 0:
        return True
    return False


def _expand_index(index, batch_size):
    len_index = sum(True for idx in index if idx is not None)
    if len_index > len(batch_size):
        raise ValueError
    if len_index < len(batch_size):
        index = index + (slice(None),) * (len(batch_size) - len_index)
    return index


def _broadcast_tensors(index):
    # tensors and range need to be broadcast
    tensors = {
        i: tensor if isinstance(tensor, Tensor) else torch.tensor(tensor)
        for i, tensor in enumerate(index)
        if isinstance(tensor, (range, list, np.ndarray, torch.Tensor))
    }
    if tensors:
        shape = torch.broadcast_shapes(*[tensor.shape for tensor in tensors.values()])
        tensors = {i: tensor.expand(shape) for i, tensor in tensors.items()}
        index = tuple(
            idx if i not in tensors else tensors[i] for i, idx in enumerate(index)
        )
    return index


def _convert_index_lazystack(index, stack_dim, batch_size):
    out = {
        "remaining_index": None,
        "stack_index": None,
        "new_stack_dim": None,
    }
    index = convert_ellipsis_to_idx(index, batch_size)
    if not isinstance(index, tuple):
        index = (index,)
    if any(
        isinstance(idx, (MemmapTensor, Tensor)) and idx.dtype == torch.bool
        for idx in index
    ):
        return None
    index = _expand_index(index, batch_size)
    index = _broadcast_tensors(index)
    # find the index corresponding to the stack dim
    stack_dim_index = 0
    new_stack_dim = 0
    count = -1
    has_first_tensor = False
    while True:
        if index[stack_dim_index] is None:
            stack_dim_index += 1
            new_stack_dim += 1
            continue

        count += 1
        if count == stack_dim:
            remaining_index = tuple(
                idx for i, idx in enumerate(index) if i != stack_dim_index
            )
            stack_index = index[stack_dim_index]
            break
        if isinstance(index[stack_dim_index], int) or (
            isinstance(index[stack_dim_index], (Tensor, np.ndarray))
            and (not index[stack_dim_index].ndim or has_first_tensor)
        ):
            new_stack_dim_incr = 0
        else:
            new_stack_dim_incr = 1
            has_first_tensor = has_first_tensor or isinstance(
                index[stack_dim_index], (Tensor, np.ndarray)
            )
        new_stack_dim += new_stack_dim_incr
        stack_dim_index += 1

    out["stack_index"] = stack_index
    if not isinstance(stack_index, int):
        out["new_stack_dim"] = new_stack_dim
    if isinstance(stack_index, Tensor):
        # we build one index for each resulting item
        # all the tensors are indexed along the same integer
        def _dispatch(remaining_index, stack_index, i=None):
            if i is not None:
                remaining_index = tuple(
                    idx[i] if isinstance(idx, Tensor) else idx
                    for idx in remaining_index
                )
            if isinstance(stack_index, list):
                out = []
                for j, _stack_index in enumerate(stack_index):
                    out.append(_dispatch(remaining_index, _stack_index, j))
                return tuple(out)
            return remaining_index

        remaining_index = _dispatch(remaining_index, stack_index.tolist())
    out["remaining_index"] = remaining_index
    return out

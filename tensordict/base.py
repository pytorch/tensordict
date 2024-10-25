# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import collections
import concurrent.futures
import contextlib
import enum
import gc
import importlib
import os.path
import queue
import uuid
import warnings
import weakref
from collections.abc import MutableMapping

from concurrent.futures import Future, ThreadPoolExecutor, wait
from copy import copy
from functools import partial, wraps
from pathlib import Path
from textwrap import indent
from threading import Thread
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    OrderedDict,
    overload,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import orjson as json
import torch

from tensordict._contextlib import LAST_OP_MAPS
from tensordict._nestedkey import NestedKey
from tensordict.memmap import MemoryMappedTensor
from tensordict.utils import (
    _as_context_manager,
    _CloudpickleWrapper,
    _DTYPE2STRDTYPE,
    _GENERIC_NESTED_ERR,
    _is_non_tensor,
    _is_number,
    _is_tensorclass,
    _KEY_ERROR,
    _lock_warn,
    _make_dtype_promotion,
    _parse_to,
    _pin_mem,
    _PIN_MEM_TIMEOUT,
    _prefix_last_key,
    _proc_init,
    _prune_selected_keys,
    _rebuild_njt_from_njt,
    _set_max_batch_size,
    _shape,
    _split_tensordict,
    _td_fields,
    _unravel_key_to_tuple,
    _zip_strict,
    cache,
    convert_ellipsis_to_idx,
    DeviceType,
    erase_cache,
    implement_for,
    IndexType,
    infer_size_impl,
    int_generator,
    is_namedtuple,
    is_namedtuple_class,
    is_non_tensor,
    lazy_legacy,
    lock_blocked,
    prod,
    set_lazy_legacy,
    strtobool,
    TensorDictFuture,
    unravel_key,
    unravel_key_list,
)
from torch import distributed as dist, multiprocessing as mp, nn, Tensor
from torch.nn.parameter import Parameter, UninitializedTensorMixin
from torch.utils._pytree import tree_map

try:
    from torch.compiler import is_dynamo_compiling
except ImportError:  # torch 2.0
    from torch._dynamo import is_compiling as is_dynamo_compiling

try:
    from torch import _foreach_copy_
except ImportError:
    _foreach_copy_ = None

try:
    from torch.nn.parameter import Buffer
except ImportError:
    from tensordict.utils import Buffer


# NO_DEFAULT is used as a placeholder whenever the default is not provided.
# Using None is not an option since `td.get(key)` is a valid usage.
class _NoDefault(enum.IntEnum):
    ZERO = 0


NO_DEFAULT = _NoDefault.ZERO
assert not NO_DEFAULT
T = TypeVar("T", bound="TensorDictBase")


class _BEST_ATTEMPT_INPLACE:
    def __bool__(self):
        # we use an exception to exit when running `inplace = BEST_ATTEMPT_INPLACE if inplace else False`
        # more than once
        raise NotImplementedError


_has_mps = torch.backends.mps.is_available()
_has_cuda = torch.cuda.is_available()

BEST_ATTEMPT_INPLACE = _BEST_ATTEMPT_INPLACE()

# some complex string used as separator to concatenate and split keys in
# distributed frameworks
CompatibleType = Union[Tensor,]

_STR_MIXED_INDEX_ERROR = "Received a mixed string-non string index. Only string-only or string-free indices are supported."

_HEURISTIC_EXCLUDED = (Tensor, tuple, list, set, dict, np.ndarray)

if "TD_GET_DEFAULTS_TO_NONE" in os.environ:
    _GET_DEFAULTS_TO_NONE = strtobool(os.environ["TD_GET_DEFAULTS_TO_NONE"])
else:
    _GET_DEFAULTS_TO_NONE = None


def set_get_defaults_to_none(set_to_none: bool = True):
    """Sets the default of `get` to `None` and silences deprecation warnings during calls to `get` that result in a `KeyError`.

    This can also be controlled via the environment variable ``TD_GET_DEFAULTS_TO_NONE``.

    Args:
        set_to_none (bool): whether the default of `get` should be `None`, or should `get` raise a `KeyError` if
            no `default` is passed and the key is absent from the `TensorDict`.
            Defaults to `True`.

    """
    global _GET_DEFAULTS_TO_NONE
    _GET_DEFAULTS_TO_NONE = set_to_none


def get_defaults_to_none(set_to_none: bool = True):
    """Returns the status of `get` default value."""
    return _GET_DEFAULTS_TO_NONE


class _RecordDeviceTransfer:
    """A class that records the device transfers during a TensorDict initialization."""

    def __init__(self):
        self.marked = False
        self._has_transfer = False

    def mark(self):
        if self.marked:
            raise RuntimeError("Can only mark one TensorDict at a time.")
        self.marked = True
        self._has_transfer = False

    def unmark(self):
        self.marked = False
        self._has_transfer = False

    def record_transfer(self, device):
        # Adds a device to all markers
        self._has_transfer = True

    def has_transfer(self):
        return self._has_transfer


_device_recorder = _RecordDeviceTransfer()


class TensorDictBase(MutableMapping):
    """TensorDictBase is an abstract parent class for TensorDicts, a torch.Tensor data container."""

    _safe: bool = False
    _lazy: bool = False
    _inplace_set: bool = False
    is_meta: bool = False
    _is_locked: bool = False
    _cache: bool = None
    _is_non_tensor: bool = False
    _memmap_prefix = None
    _stream: torch.cuda.Stream | None = None

    def __bool__(self) -> bool:
        raise RuntimeError("Converting a tensordict to boolean value is not permitted")

    @abc.abstractmethod
    def __ne__(self, other: object) -> T:
        """NOT operation over two tensordicts, for evey key.

        The two tensordicts must have the same key set.

        Args:
            other (TensorDictBase, dict, or float): the value to compare against.

        Returns:
            a new TensorDict instance with all tensors are boolean
            tensors of the same shape as the original tensors.

        """
        ...

    @abc.abstractmethod
    def __xor__(self, other: TensorDictBase | float):
        """XOR operation over two tensordicts, for evey key.

        The two tensordicts must have the same key set.

        Args:
            other (TensorDictBase, dict, or float): the value to compare against.

        Returns:
            a new TensorDict instance with all tensors are boolean
            tensors of the same shape as the original tensors.

        """
        ...

    @abc.abstractmethod
    def __or__(self, other: TensorDictBase | torch.Tensor) -> T:
        """OR operation over two tensordicts, for evey key.

        The two tensordicts must have the same key set.

        Args:
            other (TensorDictBase, dict, or float): the value to compare against.

        Returns:
            a new TensorDict instance with all tensors are boolean
            tensors of the same shape as the original tensors.

        """
        ...

    @abc.abstractmethod
    def __eq__(self, other: object) -> T:
        """Compares two tensordicts against each other, for every key. The two tensordicts must have the same key set.

        Returns:
            a new TensorDict instance with all tensors are boolean
            tensors of the same shape as the original tensors.

        """
        ...

    @abc.abstractmethod
    def __ge__(self, other: object) -> T:
        """Compares two tensordicts against each other using the "greater or equal" operator, for every key. The two tensordicts must have the same key set.

        Returns:
            a new TensorDict instance with all tensors are boolean
            tensors of the same shape as the original tensors.

        """
        ...

    @abc.abstractmethod
    def __gt__(self, other: object) -> T:
        """Compares two tensordicts against each other using the "greater than" operator, for every key. The two tensordicts must have the same key set.

        Returns:
            a new TensorDict instance with all tensors are boolean
            tensors of the same shape as the original tensors.

        """
        ...

    @abc.abstractmethod
    def __le__(self, other: object) -> T:
        """Compares two tensordicts against each other using the "lower or equal" operator, for every key. The two tensordicts must have the same key set.

        Returns:
            a new TensorDict instance with all tensors are boolean
            tensors of the same shape as the original tensors.

        """
        ...

    @abc.abstractmethod
    def __lt__(self, other: object) -> T:
        """Compares two tensordicts against each other using the "lower than" operator, for every key. The two tensordicts must have the same key set.

        Returns:
            a new TensorDict instance with all tensors are boolean
            tensors of the same shape as the original tensors.

        """
        ...

    def __repr__(self) -> str:
        try:
            fields = _td_fields(self)
            field_str = indent(f"fields={{{fields}}}", 4 * " ")
            batch_size_str = indent(f"batch_size={self.batch_size}", 4 * " ")
            device_str = indent(f"device={self.device}", 4 * " ")
            is_shared_str = indent(f"is_shared={self.is_shared()}", 4 * " ")
            string = ",\n".join([field_str, batch_size_str, device_str, is_shared_str])
        except AttributeError:
            # When using torch.compile, an exception may be raised with a tensordict object
            #  that has no attribute (no _tensordict or no _batch_size).
            #  To get the proper erro message and not an attribute error raised during __repr__,
            #  we simply default to '...' when trying to print the TD content.
            string = "..."
        return f"{type(self).__name__}(\n{string})"

    def __iter__(self) -> Generator:
        """Iterates over the first shape-dimension of the tensordict."""
        if not self.batch_dims:
            raise StopIteration
        yield from self.unbind(0)

    def __len__(self) -> int:
        """Returns the length of first dimension, if there is, otherwise 0."""
        batch_size = self.batch_size
        if not batch_size:
            return 0
        return batch_size[0]

    def __contains__(self, key: NestedKey) -> bool:
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

    def __getitem__(self, index: IndexType) -> Any:
        """Indexes all tensors according to the provided index.

        The index can be a (nested) key or any valid shape index given the
        tensordict batch size.

        If the index is a nested key and the result is a :class:`~tensordict.NonTensorData`
        object, the content of the non-tensor is returned.

        Examples:
            >>> td = TensorDict({"root": torch.arange(2), ("nested", "entry"): torch.arange(2)}, [2])
            >>> td["root"]
            torch.tensor([0, 1])
            >>> td["nested", "entry"]
            torch.tensor([0, 1])
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
        if all(isinstance(idx, slice) and idx == slice(None) for idx in index):
            return self

        return self._index_tensordict(index)

    # this is necessary for data collectors for instance, otherwise indexing
    # will always be achieved one element at a time.
    __getitems__ = __getitem__

    def _get_sub_tensordict(self, idx: IndexType) -> T:
        """Returns a _SubTensorDict with the desired index."""
        from tensordict._td import _SubTensorDict

        return _SubTensorDict(source=self, idx=idx)

    @abc.abstractmethod
    def __setitem__(
        self,
        index: IndexType,
        value: Any,
    ) -> None: ...

    def __delitem__(self, key: NestedKey) -> T:
        return self.del_(key)

    def __getstate__(self):
        result = dict(self.__dict__)
        for key in ("_last_op", "_cache", "__lock_parents_weakrefs"):
            result.pop(key, None)
        return result

    def __setstate__(self, state):
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
            return NotImplemented
        return TD_HANDLED_FUNCTIONS[func](*args, **kwargs)

    @abc.abstractmethod
    def all(self, dim: int = None) -> bool | TensorDictBase:
        """Checks if all values are True/non-null in the tensordict.

        Args:
            dim (int, optional): if ``None``, returns a boolean indicating
                whether all tensors return `tensor.all() == True`
                If integer, all is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.

        """
        ...

    @abc.abstractmethod
    def any(self, dim: int = None) -> bool | TensorDictBase:
        """Checks if any value is True/non-null in the tensordict.

        Args:
            dim (int, optional): if ``None``, returns a boolean indicating
                whether all tensors return `tensor.any() == True`.
                If integer, all is called upon the dimension specified if
                and only if this dimension is compatible with
                the tensordict shape.

        """
        ...

    def isfinite(self) -> T:
        """Returns a new tensordict with boolean elements representing if each element is finite or not.

        Real values are finite when they are not NaN, negative infinity, or infinity. Complex values are finite when both their real and imaginary parts are finite.

        """
        keys, vals = self._items_list(True, True)
        vals = [val.isfinite() for val in vals]
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def isnan(self) -> T:
        """Returns a new tensordict with boolean elements representing if each element of input is NaN or not.

        Complex values are considered NaN when either their real and/or imaginary part is NaN.

        """
        keys, vals = self._items_list(True, True)
        vals = [val.isnan() for val in vals]
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def isneginf(self) -> T:
        """Tests if each element of input is negative infinity or not."""
        keys, vals = self._items_list(True, True)
        vals = [val.isneginf() for val in vals]
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def isposinf(self) -> T:
        """Tests if each element of input is negative infinity or not."""
        keys, vals = self._items_list(True, True)
        vals = [val.isposinf() for val in vals]
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def isreal(self) -> T:
        """Returns a new tensordict with boolean elements representing if each element of input is real-valued or not."""
        keys, vals = self._items_list(True, True)
        vals = [val.isreal() for val in vals]
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def amin(
        self,
        dim: int | NO_DEFAULT = NO_DEFAULT,
        keepdim: bool = False,
        *,
        reduce: bool | None = None,
    ) -> TensorDictBase | torch.Tensor:  # noqa: D417
        """Returns the minimum values of all elements in the input tensordict.

        Same as :meth:`~.min` with ``return_indices=False``.
        """
        return self._cast_reduction(
            reduction_name="amin",
            dim=dim,
            keepdim=keepdim,
            further_reduce=reduce,
            tuple_ok=False,
            values_only=True,
            call_on_nested=False,
        )

    def min(
        self,
        dim: int | NO_DEFAULT = NO_DEFAULT,
        keepdim: bool = False,
        *,
        reduce: bool | None = None,
        return_indices: bool = True,
    ) -> TensorDictBase | torch.Tensor:  # noqa: D417
        """Returns the minimum values of all elements in the input tensordict.

        Args:
            dim (int, optional): if ``None``, returns a dimensionless
                tensordict containing the min value of all leaves (if this can be computed).
                If integer, `min` is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.
            keepdim (bool): whether the output tensor has dim retained or not.

        Keyword Args:
            reduce (bool, optional): if ``True``, the reduciton will occur across all TensorDict values
                and a single reduced tensor will be returned.
                Defaults to ``False``.
            return_argmins (bool, optional): :func:`~torch.min` returns a named tuple with values and indices
                when the ``dim`` argument is passed. The ``TensorDict`` equivalent of this is to return a tensorclass
                with entries ``"values"`` and ``"indices"`` with idendical structure within. Defaults to ``True``.

        """
        result = self._cast_reduction(
            reduction_name="min",
            dim=dim,
            keepdim=keepdim,
            further_reduce=reduce,
            tuple_ok=False,
            values_only=not return_indices,
            call_on_nested=False,
        )
        if dim is not NO_DEFAULT and return_indices:
            # Split the tensordict
            from .return_types import min

            values_dict = {}
            indices_dict = {}
            for key in result.keys(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS):
                if key[-1] == "values":
                    values_dict[key] = key[:-1]
                else:
                    indices_dict[key] = key[:-1]
            return min(
                *result.split_keys(values_dict, indices_dict),
                batch_size=result.batch_size,
            )
        return result

    def amax(
        self,
        dim: int | NO_DEFAULT = NO_DEFAULT,
        keepdim: bool = False,
        *,
        reduce: bool | None = None,
    ) -> TensorDictBase | torch.Tensor:  # noqa: D417
        """Returns the maximum values of all elements in the input tensordict.

        Same as :meth:`~.max` with ``return_indices=False``.
        """
        return self._cast_reduction(
            reduction_name="amax",
            dim=dim,
            keepdim=keepdim,
            further_reduce=reduce,
            tuple_ok=False,
            values_only=True,
            call_on_nested=False,
        )

    def max(
        self,
        dim: int | NO_DEFAULT = NO_DEFAULT,
        keepdim: bool = False,
        *,
        reduce: bool | None = None,
        return_indices: bool = True,
    ) -> TensorDictBase | torch.Tensor:  # noqa: D417
        """Returns the maximum values of all elements in the input tensordict.

        Args:
            dim (int, optional): if ``None``, returns a dimensionless
                tensordict containing the max value of all leaves (if this can be computed).
                If integer, `max` is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.
            keepdim (bool): whether the output tensor has dim retained or not.

        Keyword Args:
            reduce (bool, optional): if ``True``, the reduciton will occur across all TensorDict values
                and a single reduced tensor will be returned.
                Defaults to ``False``.
            return_argmins (bool, optional): :func:`~torch.max` returns a named tuple with values and indices
                when the ``dim`` argument is passed. The ``TensorDict`` equivalent of this is to return a tensorclass
                with entries ``"values"`` and ``"indices"`` with idendical structure within. Defaults to ``True``.

        """
        result = self._cast_reduction(
            reduction_name="max",
            dim=dim,
            keepdim=keepdim,
            further_reduce=reduce,
            tuple_ok=False,
            values_only=not return_indices,
            call_on_nested=False,
        )
        if dim is not NO_DEFAULT and return_indices:
            # Split the tensordict
            from .return_types import max

            values_dict = {}
            indices_dict = {}
            for key in result.keys(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS):
                if key[-1] == "values":
                    values_dict[key] = key[:-1]
                else:
                    indices_dict[key] = key[:-1]
            return max(
                *result.split_keys(values_dict, indices_dict),
                batch_size=result.batch_size,
            )
        return result

    def cummin(
        self,
        dim: int,
        *,
        reduce: bool | None = None,
        return_indices: bool = True,
    ) -> TensorDictBase | torch.Tensor:  # noqa: D417
        """Returns the cumulative minimum values of all elements in the input tensordict.

        Args:
            dim (int): integer representing the dimension along which to perform the cummin operation.

        Keyword Args:
            reduce (bool, optional): if ``True``, the reduciton will occur across all TensorDict values
                and a single reduced tensor will be returned.
                Defaults to ``False``.
            return_argmins (bool, optional): :func:`~torch.cummin` returns a named tuple with values and indices
                when the ``dim`` argument is passed. The ``TensorDict`` equivalent of this is to return a tensorclass
                with entries ``"values"`` and ``"indices"`` with idendical structure within. Defaults to ``True``.

        """
        result = self._cast_reduction(
            reduction_name="cummin",
            dim=dim,
            further_reduce=reduce,
            tuple_ok=False,
            values_only=not return_indices,
            call_on_nested=False,
            batch_size=self.batch_size,
        )
        if dim is not NO_DEFAULT and return_indices:
            # Split the tensordict
            from .return_types import cummin

            values_dict = {}
            indices_dict = {}
            for key in result.keys(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS):
                if key[-1] == "values":
                    values_dict[key] = key[:-1]
                else:
                    indices_dict[key] = key[:-1]
            return cummin(
                *result.split_keys(values_dict, indices_dict),
                batch_size=result.batch_size,
            )
        return result

    def cummax(
        self,
        dim: int,
        *,
        reduce: bool | None = None,
        return_indices: bool = True,
    ) -> TensorDictBase | torch.Tensor:  # noqa: D417
        """Returns the cumulative maximum values of all elements in the input tensordict.

        Args:
            dim (int): integer representing the dimension along which to perform the cummax operation.

        Keyword Args:
            reduce (bool, optional): if ``True``, the reduciton will occur across all TensorDict values
                and a single reduced tensor will be returned.
                Defaults to ``False``.
            return_argmins (bool, optional): :func:`~torch.cummax` returns a named tuple with values and indices
                when the ``dim`` argument is passed. The ``TensorDict`` equivalent of this is to return a tensorclass
                with entries ``"values"`` and ``"indices"`` with idendical structure within. Defaults to ``True``.

        """
        result = self._cast_reduction(
            reduction_name="cummax",
            dim=dim,
            further_reduce=reduce,
            tuple_ok=False,
            values_only=not return_indices,
            call_on_nested=False,
            batch_size=self.batch_size,
        )
        if dim is not NO_DEFAULT and return_indices:
            # Split the tensordict
            from .return_types import cummax

            values_dict = {}
            indices_dict = {}
            for key in result.keys(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS):
                if key[-1] == "values":
                    values_dict[key] = key[:-1]
                else:
                    indices_dict[key] = key[:-1]
            return cummax(
                *result.split_keys(values_dict, indices_dict),
                batch_size=result.batch_size,
            )
        return result

    def mean(
        self,
        dim: int | Tuple[int] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool | None = None,
    ) -> TensorDictBase | torch.Tensor:  # noqa: D417
        """Returns the mean value of all elements in the input tensordict.

        Args:
            dim (int, tuple of int, optional): if ``None``, returns a dimensionless
                tensordict containing the mean value of all leaves (if this can be computed).
                If integer or tuple of integers, `mean` is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.
            keepdim (bool): whether the output tensor has dim retained or not.

        Keyword Args:
            dtype (torch.dtype, optional): the desired data type of returned tensor.
                If specified, the input tensor is casted to dtype before the operation is performed.
                This is useful for preventing data type overflows. Default: ``None``.
            reduce (bool, optional): if ``True``, the reduciton will occur across all TensorDict values
                and a single reduced tensor will be returned.
                Defaults to ``False``.

        """
        # if dim is NO_DEFAULT and not keepdim:
        #     dim = None
        #     keepdim = False
        return self._cast_reduction(
            reduction_name="mean",
            dim=dim,
            keepdim=keepdim,
            dtype=dtype,
            further_reduce=reduce,
        )

    def nanmean(
        self,
        dim: int | Tuple[int] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool | None = None,
    ) -> TensorDictBase | torch.Tensor:  # noqa: D417
        """Returns the mean of all non-NaN elements in the input tensordict.

        Args:
            dim (int, tuple of int, optional): if ``None``, returns a dimensionless
                tensordict containing the mean value of all leaves (if this can be computed).
                If integer or tuple of integers, `mean` is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.
            keepdim (bool): whether the output tensor has dim retained or not.

        Keyword Args:
            dtype (torch.dtype, optional): the desired data type of returned tensor.
                If specified, the input tensor is casted to dtype before the operation is performed.
                This is useful for preventing data type overflows. Default: ``None``.
            reduce (bool, optional): if ``True``, the reduciton will occur across all TensorDict values
                and a single reduced tensor will be returned.
                Defaults to ``False``.

        """
        return self._cast_reduction(
            reduction_name="nanmean",
            keepdim=keepdim,
            dim=dim,
            dtype=dtype,
            further_reduce=reduce,
        )

    def prod(
        self,
        dim: int | Tuple[int] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool | None = None,
    ) -> TensorDictBase | torch.Tensor:  # noqa: D417
        """Returns the produce of values of all elements in the input tensordict.

        Args:
            dim (int, tuple of int, optional): if ``None``, returns a dimensionless
                tensordict containing the prod value of all leaves (if this can be computed).
                If integer or tuple of integers, `prod` is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.
            keepdim (bool): whether the output tensor has dim retained or not.

        Keyword Args:
            dtype (torch.dtype, optional): the desired data type of returned tensor.
                If specified, the input tensor is casted to dtype before the operation is performed.
                This is useful for preventing data type overflows. Default: ``None``.
            reduce (bool, optional): if ``True``, the reduciton will occur across all TensorDict values
                and a single reduced tensor will be returned.
                Defaults to ``False``.

        """
        result = self._cast_reduction(
            reduction_name="prod",
            dim=dim,
            keepdim=False,
            tuple_ok=False,
            dtype=dtype,
            further_reduce=reduce,
        )
        if keepdim:
            if isinstance(dim, tuple):
                dim = dim[0]
            if dim not in (None, NO_DEFAULT):
                result = result.unsqueeze(dim)
            else:
                result = result.reshape([1 for _ in self.shape])
        return result

    def sum(
        self,
        dim: int | Tuple[int] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool | None = None,
    ) -> TensorDictBase | torch.Tensor:  # noqa: D417
        """Returns the sum value of all elements in the input tensordict.

        Args:
            dim (int, tuple of int, optional): if ``None``, returns a dimensionless
                tensordict containing the sum value of all leaves (if this can be computed).
                If integer or tuple of integers, `sum` is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.
            keepdim (bool): whether the output tensor has dim retained or not.

        Keyword Args:
            dtype (torch.dtype, optional): the desired data type of returned tensor.
                If specified, the input tensor is casted to dtype before the operation is performed.
                This is useful for preventing data type overflows. Default: ``None``.
            reduce (bool, optional): if ``True``, the reduciton will occur across all TensorDict values
                and a single reduced tensor will be returned.
                Defaults to ``False``.

        """
        return self._cast_reduction(
            reduction_name="sum",
            dim=dim,
            keepdim=keepdim,
            dtype=dtype,
            further_reduce=reduce,
        )

    def nansum(
        self,
        dim: int | Tuple[int] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool | None = None,
    ) -> TensorDictBase | torch.Tensor:  # noqa: D417
        """Returns the sum of all non-NaN elements in the input tensordict.

        Args:
            dim (int, tuple of int, optional): if ``None``, returns a dimensionless
                tensordict containing the sum value of all leaves (if this can be computed).
                If integer or tuple of integers, `sum` is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.
            keepdim (bool): whether the output tensor has dim retained or not.

        Keyword Args:
            dtype (torch.dtype, optional): the desired data type of returned tensor.
                If specified, the input tensor is casted to dtype before the operation is performed.
                This is useful for preventing data type overflows. Default: ``None``.
            reduce (bool, optional): if ``True``, the reduciton will occur across all TensorDict values
                and a single reduced tensor will be returned.
                Defaults to ``False``.

        """
        return self._cast_reduction(
            reduction_name="nansum",
            dim=dim,
            keepdim=keepdim,
            dtype=dtype,
            further_reduce=reduce,
        )

    def std(
        self,
        dim: int | Tuple[int] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        correction: int = 1,
        reduce: bool | None = None,
    ) -> TensorDictBase | torch.Tensor:  # noqa: D417
        """Returns the standard deviation value of all elements in the input tensordict.

        Args:
            dim (int, tuple of int, optional): if ``None``, returns a dimensionless
                tensordict containing the sum value of all leaves (if this can be computed).
                If integer or tuple of integers, `std` is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.
            keepdim (bool): whether the output tensor has dim retained or not.

        Keyword Args:
            correction (int): difference between the sample size and sample degrees of freedom.
                Defaults to Bessel's correction, correction=1.
            reduce (bool, optional): if ``True``, the reduciton will occur across all TensorDict values
                and a single reduced tensor will be returned.
                Defaults to ``False``.

        """
        return self._cast_reduction(
            reduction_name="std",
            dim=dim,
            keepdim=keepdim,
            correction=correction,
            further_reduce=reduce,
        )

    def var(
        self,
        dim: int | Tuple[int] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        correction: int = 1,
        reduce: bool | None = None,
    ) -> TensorDictBase | torch.Tensor:  # noqa: D417
        """Returns the variance value of all elements in the input tensordict.

        Args:
            dim (int, tuple of int, optional): if ``None``, returns a dimensionless
                tensordict containing the sum value of all leaves (if this can be computed).
                If integer or tuple of integers, `var` is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.
            keepdim (bool): whether the output tensor has dim retained or not.

        Keyword Args:
            correction (int): difference between the sample size and sample degrees of freedom.
                Defaults to Bessel's correction, correction=1.
            reduce (bool, optional): if ``True``, the reduciton will occur across all TensorDict values
                and a single reduced tensor will be returned.
                Defaults to ``False``.

        """
        return self._cast_reduction(
            reduction_name="var",
            dim=dim,
            keepdim=keepdim,
            correction=correction,
            further_reduce=reduce,
        )

    @abc.abstractmethod
    def _cast_reduction(
        self,
        *,
        reduction_name,
        dim=NO_DEFAULT,
        keepdim=NO_DEFAULT,
        dtype,
        tuple_ok=True,
        further_reduce: bool,
        **kwargs,
    ): ...

    def auto_batch_size_(self, batch_dims: int | None = None) -> T:
        """Sets the maximum batch-size for the tensordict, up to an optional batch_dims.

        Args:
            batch_dims (int, optional): if provided, the batch-size will be at
                most ``batch_dims`` long.

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
        _set_max_batch_size(self, batch_dims)
        return self

    def auto_device_(self) -> T:
        """Automatically sets the device, if it is unique.

        Returns: self with the edited ``device`` attribute.

        """
        devices = {
            value.device
            for value in self.values(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS)
            if value.device is not None
        }
        if len(devices) == 1:
            self.clear_device_()
            self._set_device(list(devices)[0])
        else:
            self.clear_device_()
        return self

    @classmethod
    @abc.abstractmethod
    def from_dict(
        cls,
        input_dict,
        batch_size: torch.Size | None = None,
        device: torch.device | None = None,
        batch_dims: int | None = None,
        names: List[str] | None = None,
    ):
        """Returns a TensorDict created from a dictionary or another :class:`~.tensordict.TensorDict`.

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
            names (list of str, optional): the dimension names of the tensordict.

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
        ...

    @classmethod
    def _from_dict_validated(cls, *args, **kwargs):
        """A faster version of from_dict when the values have been validated.

        By default, falls back on :meth:`~.from_dict`.
        """
        return cls.from_dict(*args, **kwargs)

    @abc.abstractmethod
    def from_dict_instance(
        self,
        input_dict,
        batch_size=None,
        device=None,
        batch_dims=None,
        names: List[str] | None = None,
    ):
        """Instance method version of :meth:`~tensordict.TensorDict.from_dict`.

        Unlike :meth:`~tensordict.TensorDict.from_dict`, this method will
        attempt to keep the tensordict types within the existing tree (for
        any existing leaf).

        Examples:
            >>> from tensordict import TensorDict, tensorclass
            >>> import torch
            >>>
            >>> @tensorclass
            >>> class MyClass:
            ...     x: torch.Tensor
            ...     y: int
            >>>
            >>> td = TensorDict({"a": torch.randn(()), "b": MyClass(x=torch.zeros(()), y=1)})
            >>> print(td.from_dict_instance(td.to_dict()))
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: MyClass(
                        x=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                        y=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> print(td.from_dict(td.to_dict()))
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            x: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                            y: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)

        """
        ...

    @classmethod
    def from_pytree(
        cls,
        pytree,
        *,
        batch_size: torch.Size | None = None,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
    ):
        """Converts a pytree to a TensorDict instance.

        This method is designed to keep the pytree nested structure as much as possible.

        Additional non-tensor keys are added to keep track of each level's identity, providing
        a built-in pytree-to-tensordict bijective transform API.

        Accepted classes currently include lists, tuples, named tuples and dict.

        .. note::
            For dictionaries, non-NestedKey keys are registered separately as :class:`~tensordict.NonTensorData`
            instances.

        .. note::
            Tensor-castable types (such as int, float or np.ndarray) will be converted to torch.Tensor instances.
            Note that this transformation is surjective: transforming back the tensordict to a pytree will not
            recover the original types.

        Examples:
            >>> # Create a pytree with tensor leaves, and one "weird"-looking dict key
            >>> class WeirdLookingClass:
            ...     pass
            ...
            >>> weird_key = WeirdLookingClass()
            >>> # Make a pytree with tuple, lists, dict and namedtuple
            >>> pytree = (
            ...     [torch.randint(10, (3,)), torch.zeros(2)],
            ...     {
            ...         "tensor": torch.randn(
            ...             2,
            ...         ),
            ...         "td": TensorDict({"one": 1}),
            ...         weird_key: torch.randint(10, (2,)),
            ...         "list": [1, 2, 3],
            ...     },
            ...     {"named_tuple": TensorDict({"two": torch.ones(1) * 2}).to_namedtuple()},
            ... )
            >>> # Build a TensorDict from that pytree
            >>> td = TensorDict.from_pytree(pytree)
            >>> # Recover the pytree
            >>> pytree_recon = td.to_pytree()
            >>> # Check that the leaves match
            >>> def check(v1, v2):
            >>>     assert (v1 == v2).all()
            >>>
            >>> torch.utils._pytree.tree_map(check, pytree, pytree_recon)
            >>> assert weird_key in pytree_recon[1]

        """
        if is_tensor_collection(pytree):
            return pytree
        if isinstance(pytree, (torch.Tensor,)):
            return pytree

        from tensordict._td import TensorDict

        result = None
        if is_namedtuple(pytree):
            result = TensorDict.from_namedtuple(named_tuple=pytree)
            if batch_dims is not None:
                result.batch_size = batch_size
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
                    source[subs_key] = TensorDict(
                        {"value": cls.from_pytree(item), "key": key}
                    )
            source["_pytree_type"] = type(pytree)
            result = TensorDict(source, batch_size=batch_size)
        if result is not None:
            if auto_batch_size:
                result.auto_batch_size_(batch_dims)
            return result
        if isinstance(pytree, (int, float, np.ndarray)):
            return torch.as_tensor(pytree)
        raise NotImplementedError(f"Unknown type {type(pytree)}.")

    def to_pytree(self):
        """Converts a tensordict to a PyTree.

        If the tensordict was not created from a pytree, this method just returns ``self`` without modification.

        See :meth:`~.from_pytree` for more information and examples.

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
        if _pytree_type in (list, tuple):
            return _pytree_type((items[str(i)] for i in range(len(items))))
        if _pytree_type is dict:
            items = dict(
                (
                    (
                        (val["key"], val["value"])
                        if key.startswith("<NON_NESTED>")
                        else (key, val)
                    )
                    for (key, val) in items.items()
                )
            )
            return items
        if is_namedtuple_class(_pytree_type):
            from tensordict._td import TensorDict

            return TensorDict(items).to_namedtuple(dest_cls=_pytree_type)
        raise NotImplementedError(f"unknown type {_pytree_type}")

    @classmethod
    def from_h5(cls, filename, mode="r"):
        """Creates a PersistentTensorDict from a h5 file.

        This function will automatically determine the batch-size for each nested
        tensordict.

        Args:
            filename (str): the path to the h5 file.
            mode (str, optional): reading mode. Defaults to ``"r"``.
        """
        from tensordict.persistent import PersistentTensorDict

        return PersistentTensorDict.from_h5(filename, mode=mode)

    # Module interaction
    @classmethod
    def from_module(
        cls,
        module,
        as_module: bool = False,
        lock: bool = True,
        use_state_dict: bool = False,
    ):
        """Copies the params and buffers of a module in a tensordict.

        Args:
            module (nn.Module): the module to get the parameters from.
            as_module (bool, optional): if ``True``, a :class:`~tensordict.nn.TensorDictParams`
                instance will be returned which can be used to store parameters
                within a :class:`torch.nn.Module`. Defaults to ``False``.
            lock (bool, optional): if ``True``, the resulting tensordict will be locked.
                Defaults to ``True``.
            use_state_dict (bool, optional): if ``True``, the state-dict from the
                module will be used and unflattened into a TensorDict with
                the tree structure of the model. Defaults to ``False``.

                .. note::
                    This is particularly useful when state-dict hooks have to be used.

        Examples:
            >>> from torch import nn
            >>> module = nn.TransformerDecoder(
            ...     decoder_layer=nn.TransformerDecoderLayer(nhead=4, d_model=4),
            ...     num_layers=1
            ... )
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
        ...

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
        """Retrieves the parameters of several modules for ensebmle learning/feature of expects applications through vmap.

        Args:
            modules (sequence of nn.Module): the modules to get the parameters from.
                If the modules differ in their structure, a lazy stack is needed
                (see the ``lazy_stack`` argument below).

        Keyword Args:
            as_module (bool, optional): if ``True``, a :class:`~tensordict.nn.TensorDictParams`
                instance will be returned which can be used to store parameters
                within a :class:`torch.nn.Module`. Defaults to ``False``.
            lock (bool, optional): if ``True``, the resulting tensordict will be locked.
                Defaults to ``True``.
            use_state_dict (bool, optional): if ``True``, the state-dict from the
                module will be used and unflattened into a TensorDict with
                the tree structure of the model. Defaults to ``False``.

                .. note::
                    This is particularly useful when state-dict hooks have to be used.

            lazy_stack (bool, optional): whether parameters should be densly or
                lazily stacked. Defaults to ``False`` (dense stack).

                .. note::
                    ``lazy_stack`` and ``as_module`` are exclusive features.

                .. warning::
                    There is a crucial difference between lazy and non-lazy outputs
                    in that non-lazy output will reinstantiate parameters with the
                    desired batch-size, while ``lazy_stack`` will just represent
                    the parameters as lazily stacked. This means that whilst the
                    original parameters can safely be passed to an optimizer
                    when ``lazy_stack=True``, the new parameters need to be passed
                    when it is set to ``True``.

                .. warning::
                    Whilst it can be tempting to use a lazy stack to keep the
                    orignal parameter references, remember that lazy stack
                    perform a stack each time :meth:`~.get` is called. This will
                    require memory (N times the size of the parameters, more if a
                    graph is built) and time to be computed.
                    It also means that the optimizer(s) will contain more
                    parameters, and operations like :meth:`~torch.optim.Optimizer.step`
                    or :meth:`~torch.optim.Optimizer.zero_grad` will take longer
                    to be executed. In general, ``lazy_stack`` should be reserved
                    to very few use cases.

            expand_identical (bool, optional): if ``True`` and the same parameter (same
                identity) is being stacked to itself, an expanded version of this parameter
                will be returned instead. This argument is ignored when ``lazy_stack=True``.

        Examples:
            >>> from torch import nn
            >>> from tensordict import TensorDict
            >>> torch.manual_seed(0)
            >>> empty_module = nn.Linear(3, 4, device="meta")
            >>> n_models = 2
            >>> modules = [nn.Linear(3, 4) for _ in range(n_models)]
            >>> params = TensorDict.from_modules(*modules)
            >>> print(params)
            TensorDict(
                fields={
                    bias: Parameter(shape=torch.Size([2, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                    weight: Parameter(shape=torch.Size([2, 4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([2]),
                device=None,
                is_shared=False)
            >>> # example of batch execution
            >>> def exec_module(params, x):
            ...     with params.to_module(empty_module):
            ...         return empty_module(x)
            >>> x = torch.randn(3)
            >>> y = torch.vmap(exec_module, (0, None))(params, x)
            >>> assert y.shape == (n_models, 4)
            >>> # since lazy_stack = False, backprop leaves the original params untouched
            >>> y.sum().backward()
            >>> assert params["weight"].grad.norm() > 0
            >>> assert modules[0].weight.grad is None

        With ``lazy_stack=True``, things are slightly different:

            >>> params = TensorDict.from_modules(*modules, lazy_stack=True)
            >>> print(params)
            LazyStackedTensorDict(
                fields={
                    bias: Tensor(shape=torch.Size([2, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                    weight: Tensor(shape=torch.Size([2, 4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
                exclusive_fields={
                },
                batch_size=torch.Size([2]),
                device=None,
                is_shared=False,
                stack_dim=0)
            >>> # example of batch execution
            >>> y = torch.vmap(exec_module, (0, None))(params, x)
            >>> assert y.shape == (n_models, 4)
            >>> y.sum().backward()
            >>> assert modules[0].weight.grad is not None


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
        memo=None,  # deprecated
    ):
        """Writes the content of a TensorDictBase instance onto a given nn.Module attributes, recursively.

        Args:
            module (nn.Module): a module to write the parameters into.

        Keyword Args:
            inplace (bool, optional): if ``True``, the parameters or tensors
                in the module are updated in-place. Defaults to ``False``.
            return_swap (bool, optional): if ``True``, the old parameter configuration
                will be returned. Defaults to ``False``.
            swap_dest (TensorDictBase, optional): if ``return_swap`` is ``True``,
                the tensordict where the swap should be written.
            use_state_dict (bool, optional): if ``True``, state-dict API will be
                used to load the parameters (including the state-dict hooks).
                Defaults to ``False``.
            non_blocking (bool, optional): if ``True`` and this copy is between
                different devices, the copy may occur asynchronously with respect
                to the host.

        Examples:
            >>> from torch import nn
            >>> module = nn.TransformerDecoder(
            ...     decoder_layer=nn.TransformerDecoderLayer(nhead=4, d_model=4),
            ...     num_layers=1)
            >>> params = TensorDict.from_module(module)
            >>> params.zero_()
            >>> params.to_module(module)
            >>> assert (module.layers[0].linear1.weight == 0).all()
        """
        if memo is not None:
            raise RuntimeError("memo cannot be passed to the public to_module anymore.")
        hooks = getattr(
            torch.nn.modules.module, "_global_parameter_registration_hooks", {}
        )
        memo = {"hooks": tuple(hooks.values())}
        return self._to_module(
            module=module,
            inplace=inplace,
            return_swap=return_swap,
            swap_dest=swap_dest,
            memo=memo,
            use_state_dict=use_state_dict,
            non_blocking=non_blocking,
        )

    @abc.abstractmethod
    def _to_module(
        self,
        module,
        *,
        inplace: bool | None = None,
        return_swap: bool = True,
        swap_dest=None,
        memo=None,
        use_state_dict: bool = False,
        non_blocking: bool = False,
    ): ...

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
        ...

    def size(self, dim: int | None = None) -> torch.Size | int:
        """Returns the size of the dimension indicated by ``dim``.

        If ``dim`` is not specified, returns the ``batch_size`` attribute of the TensorDict.

        """
        if dim is None:
            return self.batch_size
        return self.batch_size[dim]

    @property
    def data(self):
        """Returns a tensordict containing the .data attributes of the leaf tensors."""
        return self._data()

    @property
    def grad(self):
        """Returns a tensordict containing the .grad attributes of the leaf tensors."""
        return self._grad()

    def data_ptr(self, *, storage: bool = False):
        """Returns the data_ptr of the tensordict leaves.

        This can be useful to check if two tensordicts share the same ``data_ptr()``.

        Keyword Args:
            storage (bool, optional): if ``True``, `tensor.untyped_storage().data_ptr()` will be called
                instead. Defaults to ``False``.

        Examples:
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.randn(2), b=torch.randn(2), batch_size=[2])
            >>> assert (td0.data_ptr() == td.data_ptr()).all()

        .. note:: :class:`~tensordict.LazyStackedTensorDict` instances will be displayed as nested tensordicts to
            reflect the true ``data_ptr()`` of their leaves:

                >>> td0 = TensorDict(a=torch.randn(2), b=torch.randn(2), batch_size=[2])
                >>> td1 = TensorDict(a=torch.randn(2), b=torch.randn(2), batch_size=[2])
                >>> td = TensorDict.lazy_stack([td0, td1])
                >>> td.data_ptr()
                TensorDict(
                    fields={
                        0: TensorDict(
                            fields={
                                a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                                b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=cpu,
                            is_shared=False),
                        1: TensorDict(
                            fields={
                                a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                                b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=cpu,
                            is_shared=False)},
                    batch_size=torch.Size([]),
                    device=cpu,
                    is_shared=False)

        """
        if storage:

            def func(x):
                return x.untyped_storage().data_ptr()

        else:

            def func(x):
                return x.data_ptr()

        from tensordict import TensorDict

        return TensorDict(
            {
                key: func(val)
                for key, val in self.items(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS)
            },
            device=torch.device("cpu"),
        )

    @grad.setter
    def grad(self, grad):
        def set_grad(x, grad):
            if x.grad is None:
                x.grad = grad
            else:
                x.grad.copy_(grad)

        self._fast_apply(set_grad, grad)

    def zero_grad(self, set_to_none: bool = True) -> T:
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

    @cache  # noqa
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
                "modifying the batch size of a lazy representation of a "
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

    @cache  # noqa: B019
    def _depth(self):
        depth = 0
        for key in self.keys(True, True, is_leaf=_is_leaf_nontensor):
            if isinstance(key, tuple):
                depth = max(depth, len(key) - 1)
        return depth

    @overload
    def expand(self, *shape: int) -> T: ...

    @overload
    def expand(self, shape: torch.Size) -> T: ...

    @abc.abstractmethod
    def expand(self, *args: int | torch.Size) -> T:
        """Expands each tensor of the tensordict according to the :func:`~torch.expand` function, ignoring the feature dimensions.

        Supports iterables to specify the shape.

        Examples:
            >>> td = TensorDict({
            ...     'a': torch.zeros(3, 4, 5),
            ...     'b': torch.zeros(3, 4, 10)}, batch_size=[3, 4])
            >>> td_expand = td.expand(10, 3, 4)
            >>> assert td_expand.shape == torch.Size([10, 3, 4])
            >>> assert td_expand.get("a").shape == torch.Size([10, 3, 4, 5])

        """
        ...

    def expand_as(self, other: TensorDictBase | torch.Tensor) -> TensorDictBase:
        """Broadcasts the shape of the tensordict to the shape of `other` and expands it accordingly.

        If the input is a tensor collection (tensordict or tensorclass),
        the leaves will be expanded on a one-to-one basis.

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> td0 = TensorDict({
            ...     "a": torch.ones(3, 1, 4),
            ...     "b": {"c": torch.ones(3, 2, 1, 4)}},
            ...     batch_size=[3],
            ... )
            >>> td1 = TensorDict({
            ...     "a": torch.zeros(2, 3, 5, 4),
            ...     "b": {"c": torch.zeros(2, 3, 2, 6, 4)}},
            ...     batch_size=[2, 3],
            ... )
            >>> expanded = td0.expand_as(td1)
            >>> assert (expanded==1).all()
            >>> print(expanded)
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([2, 3, 5, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([2, 3, 2, 6, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([2, 3]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([2, 3]),
                device=None,
                is_shared=False)

        """
        if _is_tensor_collection(type(other)):
            return self.apply(
                lambda x, y: x.expand_as(y), other, batch_size=other.batch_size
            )
        return self.expand(other.shape)

    def new_zeros(
        self,
        *size: torch.Size,
        dtype: torch.dtype = None,
        device: DeviceType = NO_DEFAULT,
        requires_grad: bool = False,
        layout: torch.layout = torch.strided,
        pin_memory: bool = None,
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
            **kwargs,
        )

    def new_ones(
        self,
        *size: torch.Size,
        dtype: torch.dtype = None,
        device: DeviceType = NO_DEFAULT,
        requires_grad: bool = False,
        layout: torch.layout = torch.strided,
        pin_memory: bool = None,
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
            **kwargs,
        )

    def new_empty(
        self,
        *size: torch.Size,
        dtype: torch.dtype = None,
        device: DeviceType = NO_DEFAULT,
        requires_grad: bool = False,
        layout: torch.layout = torch.strided,
        pin_memory: bool = None,
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
        pin_memory: bool = None,
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
            return getattr(tensor, funcname)(
                size,
                dtype=dtype,
                requires_grad=requires_grad,
                layout=layout,
                **kwargs,
            )

        return self._fast_apply(
            func, call_on_nested=True, device=device, batch_size=size
        )

    def new_tensor(
        self,
        data: torch.Tensor | TensorDictBase,
        *,
        dtype: torch.dtype = None,
        device: DeviceType = NO_DEFAULT,
        requires_grad: bool = False,
        pin_memory: bool | None = None,
    ):  # noqa: D417
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

    def unbind(self, dim: int) -> tuple[T, ...]:
        """Returns a tuple of indexed tensordicts, unbound along the indicated dimension.

        Examples:
            >>> td = TensorDict({
            ...     'x': torch.arange(12).reshape(3, 4),
            ... }, batch_size=[3, 4])
            >>> td0, td1, td2 = td.unbind(0)
            >>> td0['x']
            tensor([0, 1, 2, 3])
            >>> td1['x']
            tensor([4, 5, 6, 7])

        """
        batch_dims = self.batch_dims
        if dim < -batch_dims or dim >= batch_dims:
            raise RuntimeError(
                f"the dimension provided ({dim}) is beyond the tensordict dimensions ({self.ndim})."
            )
        if dim < 0:
            dim = batch_dims + dim
        results = self._unbind(dim)
        if self._is_memmap or self._is_shared:
            for result in results:
                result.lock_()
        return results

    @abc.abstractmethod
    def _unbind(self, dim: int) -> tuple[T, ...]: ...

    def chunk(self, chunks: int, dim: int = 0) -> tuple[TensorDictBase, ...]:
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
        if chunks < 1:
            raise ValueError(
                f"chunks must be a strictly positive integer, got {chunks}."
            )
        # fall back on split, using upper rounding
        split_size = -(self.batch_size[dim] // -chunks)
        return self.split(split_size, dim=dim)

    @overload
    def unsqueeze(self, dim: int) -> T: ...

    @_as_context_manager()
    def unsqueeze(self, *args, **kwargs):
        """Unsqueezes all tensors for a dimension comprised in between `-td.batch_dims` and `td.batch_dims` and returns them in a new tensordict.

        Args:
            dim (int): dimension along which to unsqueeze

        Examples:
            >>> td = TensorDict({
            ...     'x': torch.arange(24).reshape(3, 4, 2),
            ... }, batch_size=[3, 4])
            >>> td = td.unsqueeze(-2)
            >>> td.shape
            torch.Size([3, 1, 4])
            >>> td.get("x").shape
            torch.Size([3, 1, 4, 2])

        This operation can be used as a context manager too. Changes to the original
        tensordict will occur out-place, i.e. the content of the original tensors
        will not be altered. This also assumes that the tensordict is not locked
        (otherwise, unlocking the tensordict is necessary).

            >>> td = TensorDict({
            ...     'x': torch.arange(24).reshape(3, 4, 2),
            ... }, batch_size=[3, 4])
            >>> with td.unsqueeze(-2) as tds:
            ...     tds.set("y", torch.zeros(3, 1, 4))
            >>> assert td.get("y").shape == [3, 4]

        """
        _lazy_legacy = lazy_legacy()

        if _lazy_legacy:
            return self._legacy_unsqueeze(*args, **kwargs)
        else:
            result = self._unsqueeze(*args, **kwargs)
            if result._is_memmap or result._is_shared:
                result.lock_()
            return result

    @abc.abstractmethod
    def _unsqueeze(self, dim): ...

    def _legacy_unsqueeze(self, dim: int) -> T:
        if dim < 0:
            dim = self.batch_dims + dim + 1

        if (dim > self.batch_dims) or (dim < 0):
            raise RuntimeError(
                f"unsqueezing is allowed for dims comprised between "
                f"`-td.batch_dims` and `td.batch_dims` only. Got "
                f"dim={dim} with a batch size of {self.batch_size}."
            )
        from tensordict._lazy import _UnsqueezedTensorDict

        return _UnsqueezedTensorDict(
            source=self,
            custom_op="unsqueeze",
            inv_op="squeeze",
            custom_op_kwargs={"dim": dim},
            inv_op_kwargs={"dim": dim},
        )

    @overload
    def squeeze(self, dim: int | None = None) -> T: ...

    @_as_context_manager()
    def squeeze(self, *args, **kwargs):
        """Squeezes all tensors for a dimension in between `-self.batch_dims+1` and `self.batch_dims-1` and returns them in a new tensordict.

        Args:
            dim (Optional[int]): dimension along which to squeeze. If dim is
                ``None``, all singleton dimensions will be squeezed.
                Defaults to ``None``.

        Examples:
            >>> td = TensorDict({
            ...     'x': torch.arange(24).reshape(3, 1, 4, 2),
            ... }, batch_size=[3, 1, 4])
            >>> td = td.squeeze()
            >>> td.shape
            torch.Size([3, 4])
            >>> td.get("x").shape
            torch.Size([3, 4, 2])

        This operation can be used as a context manager too. Changes to the original
        tensordict will occur out-place, i.e. the content of the original tensors
        will not be altered. This also assumes that the tensordict is not locked
        (otherwise, unlocking the tensordict is necessary). This functionality is
        *not* compatible with implicit squeezing.

            >>> td = TensorDict({
            ...     'x': torch.arange(24).reshape(3, 1, 4, 2),
            ... }, batch_size=[3, 1, 4])
            >>> with td.squeeze(1) as tds:
            ...     tds.set("y", torch.zeros(3, 4))
            >>> assert td.get("y").shape == [3, 1, 4]

        """
        _lazy_legacy = lazy_legacy()

        if _lazy_legacy:
            return self._legacy_squeeze(*args, **kwargs)
        else:
            result = self._squeeze(*args, **kwargs)
            if result._is_memmap or result._is_shared:
                result.lock_()
            return result

    @abc.abstractmethod
    def _squeeze(self, dim=None): ...

    def _legacy_squeeze(self, dim: int | None = None) -> T:
        from tensordict._lazy import _SqueezedTensorDict

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

    @overload
    def reshape(self, *shape: int): ...

    @overload
    def reshape(self, shape: list | tuple): ...

    @abc.abstractmethod
    def reshape(
        self,
        *args,
        **kwargs,
    ) -> T:
        """Returns a contiguous, reshaped tensor of the desired shape.

        Args:
            *shape (int): new shape of the resulting tensordict.

        Returns:
            A TensorDict with reshaped keys

        Examples:
            >>> td = TensorDict({
            ...     'x': torch.arange(12).reshape(3, 4),
            ... }, batch_size=[3, 4])
            >>> td = td.reshape(12)
            >>> print(td['x'])
            torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        """
        ...

    def cat_tensors(
        self,
        *keys: NestedKey,
        out_key: NestedKey,
        dim: int = 0,
        keep_entries: bool = False,
    ) -> T:
        """Concatenates entries into a new entry and possibly remove the original values.

        Args:
            keys (sequence of NestedKey): entries to concatenate.

        Keyword Argument:
            out_key (NestedKey): new key name for the concatenated inputs.
            keep_entries (bool, optional): if ``False``, entries in ``keys`` will be deleted.
                Defaults to ``False``.
            dim (int, optional): the dimension along which the concatenation must occur.
                Defaults to ``0``.

        Returns: self

        Examples:
            >>> td = TensorDict(a=torch.zeros(1), b=torch.ones(1))
            >>> td.cat_tensors("a", "b", out_key="c")
            >>> assert "a" not in td
            >>> assert (td["c"] == torch.tensor([0, 1])).all()

        """
        if keep_entries:
            entries = [self.get(key) for key in keys]
        else:
            entries = [self.pop(key) for key in keys]
        return self.set(out_key, torch.cat(entries, dim=dim))

    def stack_tensors(
        self,
        *keys: NestedKey,
        out_key: NestedKey,
        dim: int = 0,
        keep_entries: bool = False,
    ) -> T:
        """Stacks entries into a new entry and possibly remove the original values.

        Args:
            keys (sequence of NestedKey): entries to stack.

        Keyword Argument:
            out_key (NestedKey): new key name for the stacked inputs.
            keep_entries (bool, optional): if ``False``, entries in ``keys`` will be deleted.
                Defaults to ``False``.
            dim (int, optional): the dimension along which the stack must occur.
                Defaults to ``0``.

        Returns: self

        Examples:
            >>> td = TensorDict(a=torch.zeros(()), b=torch.ones(()))
            >>> td.stack_tensors("a", "b", out_key="c")
            >>> assert "a" not in td
            >>> assert (td["c"] == torch.tensor([0, 1])).all()

        """
        if keep_entries:
            entries = [self.get(key) for key in keys]
        else:
            entries = [self.pop(key) for key in keys]
        return self.set(out_key, torch.stack(entries, dim=dim))

    def cat_from_tensordict(
        self,
        dim: int = 0,
        *,
        sorted: bool | List[NestedKey] | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:  # noqa: D417
        """Concatenates all entries of a tensordict in a single tensor.

        Args:
            dim (int, optional): the dimension along which the entries should be concatenated.

        Keyword Args:
            sorted (bool or list of NestedKeys): if ``True``, the entries will be concatenated in alphabetical order.
                If ``False`` (default), the dict order will be used. Alternatively, a list of key names can be provided
                and the tensors will be concatenated accordingly. This incurs some overhead as the list of keys will
                be checked against the list of leaf names in the tensordict.
            out (torch.Tensor, optional): an optional destination tensor for the cat operation.

        """
        if sorted in (None, False):
            tensors = list(self.values(True, True))
        elif sorted in (True,):
            tensors = list(self.values(True, True, sort=True))
        else:
            keys = unravel_key_list(sorted)
            if set(keys) != set(self.keys(True, True)):
                raise RuntimeError(
                    "The provided set of keys differs from the tensordict list of keys."
                )
            tensors = [self.get(key) for key in keys]
        return torch.cat(tensors, dim, out=out)

    def stack_from_tensordict(
        self,
        dim: int = 0,
        *,
        sorted: bool | List[NestedKey] | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:  # noqa: D417
        """Stacks all entries of a tensordict in a single tensor.

        Args:
            dim (int, optional): the dimension along which the entries should be stacked.

        Keyword Args:
            sorted (bool or list of NestedKeys): if ``True``, the entries will be stacked in alphabetical order.
                If ``False`` (default), the dict order will be used. Alternatively, a list of key names can be provided
                and the tensors will be stacked accordingly. This incurs some overhead as the list of keys will
                be checked against the list of leaf names in the tensordict.
            out (torch.Tensor, optional): an optional destination tensor for the stack operation.

        """
        if sorted in (None, False):
            tensors = list(self.values(True, True))
        elif sorted in (True,):
            tensors = list(self.values(True, True, sort=True))
        else:
            keys = unravel_key_list(sorted)
            if set(keys) != set(self.keys(True, True)):
                raise RuntimeError(
                    "The provided set of keys differs from the tensordict list of keys."
                )
            tensors = [self.get(key) for key in keys]
        return torch.stack(tensors, dim, out=out)

    @classmethod
    def stack(cls, input, dim=0, *, out=None):
        """Stacks tensordicts into a single tensordict along the given dimension.

        This call is equivalent to calling :func:`torch.stack` but is compatible with torch.compile.

        """
        from tensordict._torch_func import _stack

        if not _is_tensor_collection(type(input[0])):
            return torch.stack(input, dim, out=out)
        return _stack(input, dim, out=out)

    @classmethod
    def cat(cls, input, dim=0, *, out=None):
        """Concatenates tensordicts into a single tensordict along the given dimension.

        This call is equivalent to calling :func:`torch.cat` but is compatible with torch.compile.

        """
        from tensordict._torch_func import _cat

        if not _is_tensor_collection(type(input[0])):
            return torch.cat(input, dim, out=out)
        return _cat(input, dim, out=out)

    @classmethod
    def lazy_stack(cls, input, dim=0, *, out=None, **kwargs):
        """Creates a lazy stack of tensordicts.

        See :meth:`~tensordict.LazyStackTensorDict.lazy_stack` for details.
        """
        from tensordict._lazy import LazyStackedTensorDict

        return LazyStackedTensorDict.lazy_stack(input, dim=dim, out=out, **kwargs)

    @classmethod
    def maybe_dense_stack(cls, input, dim=0, *, out=None, **kwargs):
        """Attempts to make a dense stack of tensordicts, and falls back on lazy stack when required..

        See :meth:`~tensordict.LazyStackTensorDict.maybe_dense_stack` for details.
        """
        from tensordict._lazy import LazyStackedTensorDict

        return LazyStackedTensorDict.maybe_dense_stack(
            input, dim=dim, out=out, **kwargs
        )

    @abc.abstractmethod
    def split(self, split_size: int | list[int], dim: int = 0) -> list[TensorDictBase]:
        """Splits each tensor in the TensorDict with the specified size in the given dimension, like `torch.split`.

        Returns a list of ``TensorDict`` instances with the view of split chunks of items.

        Args:
            split_size (int or List(int)): size of a single chunk or list of sizes for each chunk.
            dim (int): dimension along which to split the tensor.

        Returns:
            A list of TensorDict with specified size in given dimension.

        Examples:
            >>> td = TensorDict({
            ...     'x': torch.arange(12).reshape(3, 4),
            ... }, batch_size=[3, 4])
            >>> td0, td1 = td.split([1, 2], dim=0)
            >>> print(td0['x'])
            torch.Tensor([[0, 1, 2, 3]])
        """
        ...

    def gather(self, dim: int, index: Tensor, out: T | None = None) -> T:
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

    @overload
    def view(self, *shape: int): ...

    @overload
    def view(self, dtype): ...

    @overload
    def view(self, shape: torch.Size): ...

    @abc.abstractmethod
    def _view(
        self,
        *args,
        **kwargs,
    ) -> T: ...

    @_as_context_manager()
    def view(
        self,
        *shape: int,
        size: list | tuple | torch.Size | None = None,
        batch_size: torch.Size | None = None,
    ):
        """Returns a tensordict with views of the tensors according to a new shape, compatible with the tensordict batch_size.

        Alternatively, a dtype can be provided as a first unnamed argument. In that case, all tensors will be viewed
        with the according dtype. Note that this assume that the new shapes will be compatible with the provided dtype.
        See :meth:`~torch.view` for more information on dtype views.

        Args:
            *shape (int): new shape of the resulting tensordict.
            dtype (torch.dtype): alternatively, a dtype to use to represent the tensor content.
            size: iterable

        Keyword Args:
            batch_size (torch.Size, optional): if a dtype is provided, the batch-size can be reset using this
                keyword argument. If the ``view`` is called with a shape, this is without effect.

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
        if len(shape) == 1 and isinstance(shape[0], torch.dtype):
            dtype = shape[0]
            return self._view_dtype(dtype=dtype, batch_size=batch_size)
        _lazy_legacy = lazy_legacy()

        if _lazy_legacy:
            return self._legacy_view(*shape, size=size)
        else:
            result = self._view(size=size) if size is not None else self._view(*shape)
            if result._is_shared or result._is_memmap:
                result.lock_()
            return result

    def _view_dtype(self, *, dtype, batch_size):
        # We use apply because we want to check the shapes
        return self.apply(lambda x: x.view(dtype), batch_size=batch_size)

    def _legacy_view(
        self,
        *shape: int,
        size: list | tuple | torch.Size | None = None,
    ) -> T:
        if len(shape) == 0 and size is not None:
            return self.view(*size)
        elif len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
            return self.view(*shape[0])
        elif not isinstance(shape, torch.Size):
            shape = infer_size_impl(shape, self.numel())
            shape = torch.Size(shape)
        if shape == self.shape:
            return self
        from tensordict._lazy import _ViewedTensorDict

        return _ViewedTensorDict(
            source=self,
            custom_op="view",
            inv_op="view",
            custom_op_kwargs={"size": shape},
            inv_op_kwargs={"size": self.batch_size},
        )

    @_as_context_manager()
    def transpose(self, dim0, dim1):
        """Returns a tensordict that is a transposed version of input. The given dimensions ``dim0`` and ``dim1`` are swapped.

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
        _lazy_legacy = lazy_legacy()

        if _lazy_legacy:
            return self._legacy_transpose(dim0, dim1)
        else:
            ndim = self.ndim
            if dim0 < 0:
                dim0 = ndim + dim0
            if dim1 < 0:
                dim1 = ndim + dim1
            if dim0 < 0 or dim1 < 0 or dim0 >= ndim or dim1 >= ndim:
                raise ValueError(
                    "dim0 and dim1 must be within the range of the number of dimensions."
                )
            dim0, dim1 = min(dim0, dim1), max(dim0, dim1)
            if dim0 == dim1:
                return self
            result = self._transpose(dim0, dim1)
            if result._is_shared or result._is_memmap:
                result.lock_()
            return result

    @abc.abstractmethod
    def _transpose(self, dim0, dim1): ...

    def _legacy_transpose(self, dim0, dim1):
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
        from tensordict._lazy import _TransposedTensorDict

        return _TransposedTensorDict(
            source=self,
            custom_op="transpose",
            inv_op="transpose",
            custom_op_kwargs={"dim0": dim0, "dim1": dim1},
            inv_op_kwargs={"dim0": dim0, "dim1": dim1},
        )

    @overload
    def permute(self, *dims: int): ...

    @overload
    def permute(self, dims: list | tuple): ...

    @_as_context_manager()
    def permute(self, *args, **kwargs):
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
        _lazy_legacy = lazy_legacy()

        if _lazy_legacy:
            return self._legacy_permute(*args, **kwargs)
        else:
            result = self._permute(*args, **kwargs)
            if result._is_shared or result._is_memmap:
                result.lock_()
            return result

    @abc.abstractmethod
    def _permute(
        self,
        *args,
        **kwargs,
    ): ...

    def _legacy_permute(
        self,
        *dims_list: int,
        dims: list[int] | None = None,
    ) -> T:
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

        from tensordict._lazy import _PermutedTensorDict

        return _PermutedTensorDict(
            source=self,
            custom_op="permute",
            inv_op="permute",
            custom_op_kwargs={"dims": list(map(int, dims_list))},
            inv_op_kwargs={"dims": list(map(int, dims_list))},
        )

    # Cache functionality
    def _erase_cache(self):
        self._cache = None

    # Dim names functionality
    @property
    @abc.abstractmethod
    def names(self):
        """The dimension names of the tensordict.

        The names can be set at construction time using the ``names`` argument.

        See also :meth:`~.refine_names` for details on how to set the names after
        construction.
        """
        ...

    def _get_names_idx(self, idx):
        if not self._has_names():
            return None

        def is_boolean(idx):
            try:
                from functorch import dim as ftdim

            except ImportError:
                from tensordict.utils import _ftdim_mock as ftdim

            if isinstance(idx, ftdim.Dim):
                return None
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
            if not isinstance(idx, tuple):
                idx = (idx,)
            if len([_idx for _idx in idx if _idx is not None]) < self.ndim:
                idx = (*idx, Ellipsis)
            idx_names = convert_ellipsis_to_idx(idx, self.batch_size)
            # this will convert a [None, :, :, 0, None, 0] in [None, 0, 1, None, 3]
            count = 0
            idx_to_take = []
            no_more_tensors = False
            for _idx in idx_names:
                if _idx is None:
                    idx_to_take.append(None)
                elif _is_number(_idx):
                    count += 1
                elif isinstance(_idx, (torch.Tensor, np.ndarray)):
                    if not no_more_tensors:
                        idx_to_take.extend([count] * _idx.ndim)
                        count += 1
                        no_more_tensors = True
                    else:
                        # skip this one
                        count += 1
                else:
                    idx_to_take.append(count)
                    count += 1
            names = [names[i] if i is not None else None for i in idx_to_take]
        if all(name is None for name in names):
            return None
        return names

    @abc.abstractmethod
    def _erase_names(self):
        """Erases the dimension names from a tensordict."""
        ...

    @abc.abstractmethod
    def _rename_subtds(self, value):
        """Renames all the sub-tensordicts dimension according to value.

        If value has less dimensions than the TD, the rest is just assumed to be None.
        """
        ...

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

    def refine_names(self, *names) -> T:
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
        """Same as :meth:`~.rename`, but executes the renaming in-place.

        Examples:
            >>> td = TensorDict({}, batch_size=[1, 2, 3 ,4])
            >>> td.names = list("abcd")
            >>> assert td.rename_(c="g")
            >>> assert td.names == list("abgd")
        """
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
    def _has_names(self): ...

    def _maybe_names(self):
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
        ...

    @device.setter
    @abc.abstractmethod
    def device(self, value: DeviceType) -> None: ...

    @lock_blocked
    def clear(self) -> T:
        """Erases the content of the tensordict."""
        for key in list(self.keys()):
            del self[key]
        return self

    @classmethod
    def fromkeys(cls, keys: List[NestedKey], value: Any = 0):
        """Creates a tensordict from a list of keys and a single value.

        Args:
            keys (list of NestedKey): An iterable specifying the keys of the new dictionary.
            value (compatible type, optional): The value for all keys. Defaults to ``0``.
        """
        from tensordict._td import TensorDict

        return TensorDict(dict.fromkeys(keys, value), batch_size=[])

    @abc.abstractmethod
    def popitem(self) -> Tuple[NestedKey, CompatibleType]:
        """Removes the item that was last inserted into the TensorDict.

        ``popitem`` will only return non-nested values.
        """
        ...

    def clear_device_(self) -> T:
        """Clears the device of the tensordict.

        Returns: self

        """
        self._device = None
        for value in self.values():
            if _is_tensor_collection(type(value)):
                value.clear_device_()
        return self

    def _set_device(self, device: torch.device) -> T:
        self._device = device
        for value in self.values():
            if _is_tensor_collection(type(value)):
                value._set_device(device=device)
        return self

    @cache  # noqa: B019
    def param_count(self, *, count_duplicates: bool = True) -> int:
        """Counts the number of parameters (total number of indexable items), accounting for tensors only.

        Keyword Args:
            count_duplicates (bool): Whether to count duplicated tensor as independent or not.
                If ``False``, only strictly identical tensors will be discarded (same views but different
                ids from a common base tensor will be counted twice). Defaults to `True` (each tensor is assumed
                to be a single copy).

        """
        vals = self._values_list(True, True)
        total = 0
        if not count_duplicates:
            vals = set(vals)
        for v in vals:
            total += v.numel()
        return total

    @cache  # noqa: B019
    def bytes(self, *, count_duplicates: bool = True) -> int:
        """Counts the number of bytes of the contained tensors.

        Keyword Args:
            count_duplicates (bool): Whether to count duplicated tensor as independent or not.
                If ``False``, only strictly identical tensors will be discarded (same views but different
                ids from a common base tensor will be counted twice). Defaults to `True` (each tensor is assumed
                to be a single copy).

        """
        set_of_tensors = set() if not count_duplicates else []

        def add(tensor):
            if count_duplicates:
                set_of_tensors.append(tensor)
            else:
                set_of_tensors.add(tensor)

        def count_bytes(tensor):
            if tensor.is_nested:
                if not tensor.layout == torch.jagged:
                    raise RuntimeError(
                        "NTs that are not jagged are not supported by the bytes method. Please use the jagged layout instead "
                        "or raise and issue on https://github.com/pytorch/tensordict/issues instead."
                    )
                attrs, ctx = tensor.__tensor_flatten__()
                for attr in attrs:
                    t = getattr(tensor, attr)
                    count_bytes(t)
                return
            if isinstance(tensor, torch.Tensor):
                if isinstance(tensor, MemoryMappedTensor):
                    add(tensor)
                    return
                if type(tensor) in (Tensor, Parameter, Buffer):
                    pass
                elif hasattr(tensor, "__tensor_flatten__"):
                    attrs, ctx = tensor.__tensor_flatten__()
                    for attr in attrs:
                        t = getattr(tensor, attr)
                        count_bytes(t)
                    return
                else:
                    warnings.warn(
                        "The sub-tensor doesn't ot have a __tensor_flatten__ attribute, making it "
                        "impossible to count the bytes it contains. Falling back on regular count.",
                        category=UserWarning,
                    )
                    count_bytes(torch.as_tensor(tensor))
                    return

                grad = getattr(tensor, "grad", None)
                if grad is not None:
                    count_bytes(grad)
                    count_bytes(tensor.data)
                else:
                    add(tensor)
                return

        vals = self._values_list(True, True)
        for v in vals:
            count_bytes(v)
        total = 0
        for tensor in set_of_tensors:
            total += tensor.numel() * tensor.dtype.itemsize
        return total

    def pin_memory(self, num_threads: int | None = None, inplace: bool = False) -> T:
        """Calls :meth:`~torch.Tensor.pin_memory` on the stored tensors.

        Args:
            num_threads (int or str): if provided, the number of threads to use
                to call ``pin_memory`` on the leaves. Defaults to ``None``, which sets a high
                number of threads in :class:`~concurrent.futures.ThreadPoolExecutor(max_workers=None)`.
                To execute all the calls to :meth:`~torch.Tensor.pin_memory` on the main thread, pass
                ``num_threads=0``.
            inplace (bool, optional): if ``True``, the tensordict is modified in-place.
                Defaults to ``False``.

        """
        return self._fast_apply(
            lambda x: x.pin_memory(),
            num_threads=num_threads,
            inplace=inplace,
            propagate_lock=True,
        )

    def pin_memory_(self, num_threads: int | str = 0) -> T:
        """Calls :meth:`~torch.Tensor.pin_memory` on the stored tensors and returns the TensorDict modifies in-place.

        Args:
            num_threads (int or str): if provided, the number of threads to use
                to call ``pin_memory`` on the leaves. If ``"auto"`` is passed, the
                number of threads is automatically determined.

        """
        return self.pin_memory(num_threads=num_threads, inplace=True)

    def cpu(self, **kwargs) -> T:
        """Casts a tensordict to CPU.

        This function also supports all the keyword arguments of :meth:`~.to`.
        """
        return self.to("cpu", **kwargs)

    def cuda(self, device: int = None, **kwargs) -> T:
        """Casts a tensordict to a cuda device (if not already on it).

        Args:
            device (int, optional): if provided, the cuda device on which the
                tensor should be cast.

        This function also supports all the keyword arguments of :meth:`~.to`.

        """
        if device is None:
            return self.to(torch.device("cuda"))
        return self.to(f"cuda:{device}", **kwargs)

    @property
    def is_cuda(self):
        return self.device is not None and self.device.type == "cuda"

    @property
    def is_cpu(self):
        return self.device is not None and self.device.type == "cpu"

    # Serialization functionality
    def state_dict(
        self,
        destination=None,
        prefix="",
        keep_vars=False,
        flatten=False,
    ) -> OrderedDict[str, Any]:
        """Produces a state_dict from the tensordict.

        The structure of the state-dict will still be nested, unless ``flatten`` is set to ``True``.

        A tensordict state-dict contains all the tensors and meta-data needed
        to rebuild the tensordict (names are currently not supported).

        Args:
            destination (dict, optional): If provided, the state of tensordict will
                be updated into the dict and the same object is returned.
                Otherwise, an ``OrderedDict`` will be created and returned.
                Default: ``None``.
            prefix (str, optional): a prefix added to tensor
                names to compose the keys in state_dict. Default: ``''``.
            keep_vars (bool, optional): by default the :class:`torch.Tensor` items
                returned in the state dict are detached from autograd. If it's
                set to ``True``, detaching will not be performed.
                Default: ``False``.
            flatten (bool, optional): whether the structure should be flattened
                with the ``"."`` character or not.
                Defaults to ``False``.

        Examples:
            >>> data = TensorDict({"1": 1, "2": 2, "3": {"3": 3}}, [])
            >>> sd = data.state_dict()
            >>> print(sd)
            OrderedDict([('1', tensor(1)), ('2', tensor(2)), ('3', OrderedDict([('3', tensor(3)), ('__batch_size', torch.Size([])), ('__device', None)])), ('__batch_size', torch.Size([])), ('__device', None)])
            >>> sd = data.state_dict(flatten=True)
            OrderedDict([('1', tensor(1)), ('2', tensor(2)), ('3.3', tensor(3)), ('__batch_size', torch.Size([])), ('__device', None)])

        """
        out = collections.OrderedDict()
        source = self
        if flatten:
            source = source.flatten_keys(".")
        for key, item in source.items():
            if not _is_tensor_collection(type(item)):
                if not keep_vars:
                    out[prefix + key] = item.detach().clone()
                else:
                    out[prefix + key] = item
            else:
                out[prefix + key] = item.state_dict(keep_vars=keep_vars)
        if "__batch_size" in out:
            raise KeyError(
                "Cannot retrieve the state_dict of a TensorDict with `'__batch_size'` key"
            )
        if "__device" in out:
            raise KeyError(
                "Cannot retrieve the state_dict of a TensorDict with `'__batch_size'` key"
            )
        out[prefix + "__batch_size"] = source.batch_size
        out[prefix + "__device"] = source.device
        if destination is not None:
            destination.update(out)
            return destination
        return out

    def load_state_dict(
        self,
        state_dict: OrderedDict[str, Any],
        strict=True,
        assign=False,
        from_flatten=False,
    ) -> T:
        """Loads a state-dict, formatted as in :meth:`~.state_dict`, into the tensordict.

        Args:
            state_dict (OrderedDict): the state_dict of to be copied.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this tensordict's
                :meth:`torch.nn.Module.state_dict` function. Default: ``True``
            assign (bool, optional): whether to assign items in the state
                dictionary to their corresponding keys in the tensordict instead
                of copying them inplace into the tensordict's current tensors.
                When ``False``, the properties of the tensors in the current
                module are preserved while when ``True``, the properties of the
                Tensors in the state dict are preserved.
                Default: ``False``
            from_flatten (bool, optional): if ``True``, the input state_dict is
                assumed to be flattened.
                Defaults to ``False``.

        Examples:
            >>> data = TensorDict({"1": 1, "2": 2, "3": {"3": 3}}, [])
            >>> data_zeroed = TensorDict({"1": 0, "2": 0, "3": {"3": 0}}, [])
            >>> sd = data.state_dict()
            >>> data_zeroed.load_state_dict(sd)
            >>> print(data_zeroed["3", "3"])
            tensor(3)
            >>> # with flattening
            >>> data_zeroed = TensorDict({"1": 0, "2": 0, "3": {"3": 0}}, [])
            >>> data_zeroed.load_state_dict(data.state_dict(flatten=True), from_flatten=True)
            >>> print(data_zeroed["3", "3"])
            tensor(3)


        """
        if from_flatten:
            self_flatten = self.flatten_keys(".")
            self_flatten.load_state_dict(state_dict, strict=strict, assign=assign)
            if not assign:
                # modifications are done in-place so we should be fine returning self
                return self
            else:
                # run a check over keys, if we any key with a '.' in name we're doomed
                DOT_ERROR = "Cannot use load_state_dict(..., from_flatten=True, assign=True) when some keys contain a dot character."
                for key in self.keys(True, True):
                    if isinstance(key, tuple):
                        for subkey in key:
                            if "." in subkey:
                                raise RuntimeError(DOT_ERROR)
                    elif "." in key:
                        raise RuntimeError(DOT_ERROR)
                return self.update(self_flatten.unflatten_keys("."))

        # copy since we'll be using pop
        state_dict = copy(state_dict)
        batch_size = state_dict.pop("__batch_size")
        device = state_dict.pop("__device", None)

        if strict and set(state_dict.keys()) != set(self.keys()):
            set_sd = set(state_dict.keys())
            set_td = set(self.keys())

            # if there are keys in state-dict that point to an empty tensordict
            # or if the local tensordicts are empty, we can skip
            def _is_empty_dict(sd, key=None):
                if key is not None:
                    if not isinstance(sd[key], dict):
                        return False
                    return _is_empty_dict(sd[key])
                for key, item in sd.items():
                    if key in ("__batch_size", "__device"):
                        continue
                    if isinstance(item, dict):
                        if not _is_empty_dict(item):
                            return False
                        continue
                    return False
                else:
                    return True

            def check_is_empty(target, key):
                item = target.get(key)
                if not is_tensor_collection(item) or not item.is_empty():
                    return False
                return True

            if not all(check_is_empty(self, key) for key in set_td - set_sd) or not all(
                _is_empty_dict(state_dict, key) for key in set_sd - set_td
            ):
                raise RuntimeError(
                    "Cannot load state-dict because the key sets don't match: got "
                    f"state_dict extra keys \n{set_sd - set_td}\n and tensordict extra keys\n{set_td - set_sd}\n"
                )

        self.batch_size = batch_size
        if device is not None and self.device is not None and device != self.device:
            raise RuntimeError("Loading data from another device is not yet supported.")

        for key, item in state_dict.items():
            if isinstance(item, dict):
                dest = self.get(key, None)
                if dest is None:
                    dest = self.empty()
                dest.load_state_dict(item, assign=assign, strict=strict)
                self.set(
                    key,
                    dest,
                    inplace=not assign,
                )
            else:
                self.set(key, item, inplace=not assign)
        return self

    def is_shared(self) -> bool:
        """Checks if tensordict is in shared memory.

        If a TensorDict instance is in shared memory, it is locked (entries cannot
        be renamed, removed or added). If a ``TensorDict`` is created with
        tensors that are all in shared memory, this does __not__ mean that ``is_shared``
        will return ``True`` (as a new tensor may or may not be in shared memory).
        Only if one calls `tensordict.share_memory_()` or places the tensordict
        on a device where the content is shared by default (eg, ``"cuda"``)
        will the tensordict be considered in shared memory.

        This is always ``True`` for tensordicts on a CUDA device.

        """
        if self.device and not self._is_memmap:
            return self.device.type == "cuda" or self._is_shared
        return self._is_shared

    def is_memmap(self) -> bool:
        """Checks if tensordict is memory-mapped.

        If a TensorDict instance is memory-mapped, it is locked (entries cannot
        be renamed, removed or added). If a ``TensorDict`` is created with
        tensors that are all memory-mapped, this does __not__ mean that ``is_memmap``
        will return ``True`` (as a new tensor may or may not be memory-mapped).
        Only if one calls `tensordict.memmap_()` will the tensordict be
        considered as memory-mapped.

        This is always ``True`` for tensordicts on a CUDA device.

        """
        return self._is_memmap

    @abc.abstractmethod
    def share_memory_(self) -> T:
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
        ...

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
    ) -> T: ...

    def densify(self, layout: torch.layout = torch.strided):
        """Attempts to represent the lazy stack with contiguous tensors (plain tensors or nested).

        Keyword Args:
            layout (torch.layout): the layout of the nested tensors, if any. Defaults to
                :class:`~torch.strided`.

        """
        any_set = False
        out_dict = {}
        for key, val in self.items():
            if is_tensor_collection(val):
                val_dense = val.densify(layout=layout)
                any_set = any_set | (val_dense is not val)
                val = val_dense
            out_dict[key] = val
        if any_set:
            result = self.empty()
            for key, val in out_dict.items():
                result._set_str(key, val, validated=True, inplace=False)
            return result
        return self

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

    # Generic method to get a class metadata
    def _reduce_get_metadata(self):
        return {
            "device": str(self.device) if self.device is not None else None,
            "names": self.names,
            "batch_size": list(self.batch_size),
            "is_locked": self._is_locked,
        }

    # @cache  # noqa: B019
    def _reduce_vals_and_metadata(self, *, dtype=NO_DEFAULT, requires_metadata):
        """Returns a nested dictionary of metadata, a flat Dict[NestedKey, Tensor] containing tensor data and a list of tensor sizes."""
        if dtype is NO_DEFAULT:
            dtype = self.dtype
        need_padding = dtype is None
        # If the dtype is not unique (self.dtype is None) then we need the metadata
        # because we need a custom unpickler
        requires_metadata = requires_metadata | need_padding

        if requires_metadata:
            # metadata is nested
            metadata_dict = {
                "cls": type(self).__name__,
                "non_tensors": {},
                "leaves": {},
                "cls_metadata": self._reduce_get_metadata(),
            }
        else:
            metadata_dict = None

        # flat_key_values is flat
        flat_key_values = {}

        flat_size = []
        start = 0

        def add_single_value(value, key, metadata_dict, dtype, shape, flat_size):
            nonlocal start
            n = value.element_size() * value.numel()
            if need_padding:
                pad = n % 8
                if pad != 0:
                    pad = 8 - pad
            else:
                pad = 0
            flat_size.append(n + pad)
            stop = start + flat_size[-1]
            if requires_metadata:
                metadata_dict["leaves"][key] = (
                    _DTYPE2STRDTYPE[dtype],
                    list(shape),
                    # _DEVICE2STRDEVICE[device],
                    start,
                    stop,
                    pad,
                )
            start = stop

        def assign(
            key,
            value,
            track_key=(),
            metadata_dict=metadata_dict,
            flat_size=flat_size,
        ):
            total_key = key if isinstance(key, tuple) else (key,)
            total_key = track_key + total_key
            cls = type(value)
            if issubclass(cls, torch.Tensor):
                pass
            elif _is_non_tensor(cls):
                if requires_metadata:
                    metadata_dict["non_tensors"][key] = (
                        value.data,
                        list(value.batch_size),
                    )
                return
            elif _is_tensor_collection(cls):
                metadata_dict_key = None
                if requires_metadata:
                    metadata_dict_key = metadata_dict[key] = {
                        "cls": cls.__name__,
                        "non_tensors": {},
                        "leaves": {},
                        "cls_metadata": value._reduce_get_metadata(),
                    }
                local_assign = partial(
                    assign,
                    track_key=total_key,
                    metadata_dict=metadata_dict_key,
                    flat_size=flat_size,
                )
                value._fast_apply(
                    local_assign,
                    named=True,
                    nested_keys=True,
                    call_on_nested=True,
                    is_leaf=_NESTED_TENSORS_AS_LISTS_NONTENSOR,
                )
                return
            # Tensors: DTensor, nested and then regular
            if hasattr(value, "full_tensor"):
                raise NotImplementedError("DTensor is not supported yet")
            if getattr(value, "is_nested", False):
                if value.layout is torch.jagged:
                    # Get the values
                    values = value._values
                    shape = [v if isinstance(v, int) else -1 for v in values.shape]
                    # Get the offsets
                    offsets = value._offsets
                    # Get the lengths
                    lengths = value._lengths

                    # Now we're saving the two tensors
                    # We will rely on the fact that the writing order is preserved in python dict
                    # (since python 3.7). Later, we will read the NJT then the NJT offset in that order
                    # to do the allocation.
                    flat_key_values[_prefix_last_key(total_key, "<NJT>")] = value
                    flat_size.append(0)
                    flat_key_values[_prefix_last_key(total_key, "<NJT_VALUES>")] = (
                        values
                    )
                    add_single_value(
                        values,
                        _prefix_last_key(key, "<NJT_VALUES>"),
                        metadata_dict,
                        values.dtype,
                        shape,
                        flat_size,
                    )
                    # Lengths
                    if lengths is not None:
                        flat_key_values[
                            _prefix_last_key(total_key, "<NJT_LENGTHS>")
                        ] = lengths
                        add_single_value(
                            lengths,
                            _prefix_last_key(key, "<NJT_LENGTHS>"),
                            metadata_dict,
                            lengths.dtype,
                            lengths.shape,
                            flat_size,
                        )
                    # Offsets
                    flat_key_values[_prefix_last_key(total_key, "<NJT_OFFSETS>")] = (
                        offsets
                    )
                    add_single_value(
                        offsets,
                        _prefix_last_key(key, "<NJT_OFFSETS>"),
                        metadata_dict,
                        offsets.dtype,
                        offsets.shape,
                        flat_size,
                    )

                else:
                    raise NotImplementedError(
                        "NST is not supported, please use layout=torch.jagged when building the nested tensor."
                    )
                return
            flat_key_values[total_key] = value
            add_single_value(
                value,
                key,
                metadata_dict,
                value.dtype,
                value.shape,
                # value.device,
                flat_size,
            )

        self._fast_apply(
            assign,
            named=True,
            call_on_nested=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS_NONTENSOR,
            filter_empty=True,
        )
        return metadata_dict, flat_key_values, flat_size, need_padding

    def consolidate(
        self,
        filename: Path | str | None = None,
        *,
        num_threads=0,
        device: torch.device | None = None,
        non_blocking: bool = False,
        inplace: bool = False,
        return_early: bool = False,
        use_buffer: bool = False,
        share_memory: bool = False,
        pin_memory: bool = False,
        metadata: bool = False,
    ) -> None:
        """Consolidates the tensordict content in a single storage for fast serialization.

        Args:
            filename (Path, optional): an optional file path for a memory-mapped tensor
                to use as a storage for the tensordict.

        Keyword Args:
            num_threads (integer, optional): the number of threads to use for populating
                the storage.
            device (torch.device, optional): an optional device where the storage must be
                instantiated.
            non_blocking (bool, optional): ``non_blocking`` argument passed to :meth:`~torch.Tensor.copy_`.
            inplace (bool, optional): if ``True``, the resulting tensordict is the same
                as ``self`` with updated values. Defaults to ``False``.
            return_early (bool, optional): if ``True`` and ``num_threads>0``,
                the method will return a future of the tensordict. The resulting
                tensordict can be queried using `future.result()`.
            use_buffer (bool, optional): if ``True`` and a filename is passed, an intermediate
                local buffer will be created in shared memory, and the data will be copied at
                the storage location as a last step. This may be faster than writing directly
                to a distant physical memory (e.g., NFS).
                Defaults to ``False``.
            share_memory (bool, optional): if ``True``, the storage will be placed in shared memory.
                Defaults to ``False``.
            pin_memory (bool, optional): whether the consolidated data should be placed in pinned
                memory. Defaults to ``False``.
            metadata (bool, optional): if ``True``, the metadata will be stored alongisde the
                common storage. If a filename is provided, this is without effect.
                Storing the metadata can be useful when one wants to control how serialization
                is achieved, as TensorDict handles the pickling/unpickling of consolidated TDs
                differently if the metadata is or isn't available.

        .. note::
            If the tensordict is already consolidated, all arguments are ignored and ``self``
            is returned. Call :meth:`~.contiguous` to re-consolidate.

        Examples:
            >>> import pickle
            >>> import tempfile
            >>> import torch
            >>> import tqdm
            >>> from torch.utils.benchmark import Timer
            >>> from tensordict import TensorDict
            >>> data = TensorDict({"a": torch.zeros(()), "b": {"c": torch.zeros(())}})
            >>> data_consolidated = data.consolidate()
            >>> # check that the data has a single data_ptr()
            >>> assert torch.tensor([
            ...     v.untyped_storage().data_ptr() for v in data_c.values(True, True)
            ... ]).unique().numel() == 1
            >>> # Serializing the tensordict will be faster with data_consolidated
            >>> with open("data.pickle", "wb") as f:
            ...    print("regular", Timer("pickle.dump(data, f)", globals=globals()).adaptive_autorange())
            >>> with open("data_c.pickle", "wb") as f:
            ...     print("consolidated", Timer("pickle.dump(data_consolidated, f)", globals=globals()).adaptive_autorange())


        """
        if self.is_consolidated():
            return self

        (
            metadata_dict,
            flat_dict,
            flat_size,
            need_padding,
        ) = self._reduce_vals_and_metadata(
            requires_metadata=filename is not None or metadata, dtype=None
        )
        filesize = sum(flat_size)
        device = torch.device(device) if device is not None else None
        if filename is None:
            storage = torch.empty(
                filesize,
                dtype=torch.uint8,
                device=device if device else self.device,
                pin_memory=pin_memory,
            )
            if share_memory and not (
                device is not None and device.type == "cuda"
            ):  # cuda device is always shared
                storage.share_memory_()
        else:
            # Convert the dict to json
            metadata_dict_json = json.dumps(metadata_dict)
            # Represent as a tensor
            metadata_dict_json = torch.as_tensor(
                bytearray(metadata_dict_json), dtype=torch.uint8
            )
            len_metadata = torch.tensor(
                [metadata_dict_json.numel()], dtype=torch.int64
            ).view(torch.uint8)

            if device not in (torch.device("cpu"), None):
                raise RuntimeError(
                    "device and filename are mutually exclusive arguments."
                )
            suffix = len_metadata.numel() + metadata_dict_json.numel()
            if not use_buffer:
                total_storage = torch.from_file(
                    str(filename),
                    size=filesize + suffix,
                    dtype=torch.uint8,
                    shared=True,
                    # needed when device ctx differs
                    device=torch.device("cpu"),
                )
            else:
                total_storage = MemoryMappedTensor.empty(
                    shape=(filesize + suffix,),
                    dtype=torch.uint8,
                )

            total_storage[-8:] = len_metadata
            total_storage[-8 - metadata_dict_json.numel() : -8] = metadata_dict_json
            storage = total_storage[:-suffix]
            # assert len(storage.untyped_storage()) == filesize

        offsets = torch.tensor([0] + flat_size).cumsum(0).tolist()

        def view_old_as_new(v, oldv):
            v = v.view(oldv.dtype)
            if v.numel() > oldv.numel():
                return v[: oldv.numel()].view(oldv.shape)
            return v.view(oldv.shape)

        if num_threads > 0:

            def assign(
                *,
                k,
                v,
                start,
                stop,
                njts,
                storage=storage,
                non_blocking=non_blocking,
            ):
                """Reads a slice of the storage and assigns the resulting tensor in flat_dict."""
                # v may need padding
                if k[-1].startswith("<NJT>"):
                    njts[k] = v
                    return
                v_pad = v.view(-1).view(torch.uint8)
                exp_length = stop - start
                pad = exp_length - v_pad.numel()
                if pad:
                    v_pad = torch.cat([v_pad, v_pad.new_zeros(pad)])
                storage[start:stop].copy_(v_pad, non_blocking=non_blocking)

                storage_slice = storage[start:stop]
                shape, dtype = v.shape, v.dtype
                new_v = storage_slice.view(dtype)
                if pad:
                    new_v = new_v[: v.numel()]
                new_v = new_v.view(shape)
                flat_dict[k] = new_v

            njts = {}
            if num_threads > 1:
                executor = ThreadPoolExecutor(num_threads)
                r = []
                for i, (k, v) in enumerate(flat_dict.items()):
                    r.append(
                        executor.submit(
                            assign,
                            k=k,
                            v=v,
                            start=offsets[i],
                            stop=offsets[i + 1],
                            njts=njts,
                        )
                    )
                if not return_early:
                    wait(r)
                else:
                    # TODO: We'd need to merge the second half of this function to make this a thing
                    raise NotImplementedError(
                        "return_early is not implemented yet for `consolidate`."
                    )
            else:
                for i, (k, v) in enumerate(flat_dict.items()):
                    assign(
                        k=k,
                        v=v,
                        start=offsets[i],
                        stop=offsets[i + 1],
                        njts=njts,
                    )
            for njt_key, njt in njts.items():
                newkey = njt_key[:-1] + (njt_key[-1].replace("<NJT>", ""),)
                njt_key_values = njt_key[:-1] + (
                    njt_key[-1].replace("<NJT>", "<NJT_VALUES>"),
                )
                njt_key_offset = njt_key[:-1] + (
                    njt_key[-1].replace("<NJT>", "<NJT_OFFSETS>"),
                )
                njt_key_lengths = njt_key[:-1] + (
                    njt_key[-1].replace("<NJT>", "<NJT_LENGTHS>"),
                )
                val = _rebuild_njt_from_njt(
                    njt,
                    values=flat_dict.pop(njt_key_values),
                    offsets=flat_dict.pop(njt_key_offset),
                    lengths=flat_dict.pop(njt_key_lengths, None),
                )
                del flat_dict[njt_key]
                flat_dict[newkey] = val

            if non_blocking and device.type != "cuda":
                # sync if needed
                self._sync_all()
        else:

            def _view_and_pad(tensor):
                result = tensor.view(-1).view(torch.uint8)
                # result must always have a multiple of 8 elements
                pad = 0
                if need_padding:
                    pad = result.numel() % 8
                    if pad != 0:
                        result = torch.cat([result, result.new_zeros(8 - pad)])
                return result, pad

            items = []
            for v in flat_dict.values():
                if v.is_nested:
                    continue
                if v.device != storage.device:
                    v = v.to(storage.device, non_blocking=non_blocking)
                stride = v.stride()
                if (stride and stride[-1] != 1) or v.storage_offset():
                    v = v.clone(memory_format=torch.contiguous_format)
                v, pad = _view_and_pad(v)
                items.append(v)
            if non_blocking and device.type != "cuda":
                # sync if needed
                self._sync_all()
            torch.cat(items, out=storage)
            for v, (k, oldv) in _zip_strict(
                storage.split(flat_size), list(flat_dict.items())
            ):
                if not k[-1].startswith("<"):
                    flat_dict[k] = view_old_as_new(v, oldv)
                elif k[-1].startswith("<NJT>"):
                    # NJT/NT always comes before offsets/shapes
                    nt = oldv
                    assert not v.numel()
                    nt_lengths = None
                    del flat_dict[k]
                elif k[-1].startswith("<NJT_VALUES>"):
                    nt_vaues = view_old_as_new(v, oldv)
                    del flat_dict[k]
                elif k[-1].startswith("<NJT_LENGTHS>"):
                    nt_lengths = view_old_as_new(v, oldv)
                    del flat_dict[k]
                elif k[-1].startswith("<NJT_OFFSETS>"):
                    newk = k[:-1] + (k[-1].replace("<NJT_OFFSETS>", ""),)
                    nt_offsets = view_old_as_new(v, oldv)
                    del flat_dict[k]

                    val = _rebuild_njt_from_njt(
                        nt, values=nt_vaues, offsets=nt_offsets, lengths=nt_lengths
                    )

                    flat_dict[newk] = val

                    # delete the nested value to make sure that if there was an
                    # ordering mismatch we wouldn't be looking at the value key of
                    # another nested tensor.
                    del nt, nt_vaues, nt_offsets, nt_lengths
                else:
                    flat_dict[k] = view_old_as_new(v, oldv)

        def assign_val(key, val):
            if isinstance(key, str):
                key = (key,)
            return flat_dict.get(key, val)

        if filename is None:
            device = self.device
        elif not inplace:
            device = torch.device("cpu")
        elif self.device is not None and self.device != torch.device("cpu"):
            self.clear_device_()
            device = None
        else:
            device = None
        result = self._fast_apply(
            assign_val,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS_NONTENSOR,
            out=self if inplace else None,
            device=device,
        )
        result._consolidated = {"storage": storage, "metadata": metadata_dict}
        if filename is not None:
            if use_buffer:
                with open(filename, "w+b") as f:
                    f.write(total_storage._handler.buffer)
            # with open(Path(filename).with_suffix(".json"), "wb") as f:
            #     metadata_dict["size"] = filesize
            #     f.write(json.dumps(metadata_dict))
        return result

    @classmethod
    def from_consolidated(cls, filename):
        # with open(Path(filename).with_suffix(".json"), "rb") as f:
        #     metadata = json.loads(f.read())
        file = torch.from_file(
            str(filename),
            dtype=torch.uint8,
            size=os.path.getsize(filename),
            # needed when device ctx differs
            device=torch.device("cpu"),
        )
        metadata_size = file[-8:].clone().view(torch.int64)
        metadata = file[-metadata_size - 8 : -8]
        metadata = json.loads(bytes(metadata.tolist()))

        from ._reductions import _rebuild_tensordict_files_consolidated

        return _rebuild_tensordict_files_consolidated(
            metadata, file[: -metadata_size - 8]
        )

    def is_consolidated(self):
        """Checks if a TensorDict has a consolidated storage."""
        return hasattr(self, "_consolidated")

    def memmap_(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        *,
        num_threads: int = 0,
        return_early: bool = False,
        share_non_tensor: bool = False,
        existsok: bool = True,
    ) -> T:
        """Writes all tensors onto a corresponding memory-mapped Tensor, in-place.

        Args:
            prefix (str): directory prefix where the memory-mapped tensors will
                be stored. The directory tree structure will mimic the tensordict's.
            copy_existing (bool): If False (default), an exception will be raised if an
                entry in the tensordict is already a tensor stored on disk
                with an associated file, but is not saved in the correct
                location according to prefix.
                If ``True``, any existing Tensor will be copied to the new location.

        Keyword Args:
            num_threads (int, optional): the number of threads used to write the memmap
                tensors. Defaults to `0`.
            return_early (bool, optional): if ``True`` and ``num_threads>0``,
                the method will return a future of the tensordict. The resulting
                tensordict can be queried using `future.result()`.
            share_non_tensor (bool, optional): if ``True``, the non-tensor data will be
                shared between the processes and writing operation (such as inplace update
                or set) on any of the workers within a single node will update the value
                on all other workers. If the number of non-tensor leaves is high (e.g.,
                sharing large stacks of non-tensor data) this may result in OOM or similar
                errors. Defaults to ``False``.
            existsok (bool, optional): if ``False``, an exception will be raised if a tensor already
                exists in the same path. Defaults to ``True``.

        The TensorDict is then locked, meaning that any writing operations that
        isn't in-place will throw an exception (eg, rename, set or remove an
        entry).
        Once the tensordict is unlocked, the memory-mapped attribute is turned to ``False``,
        because cross-process identity is not guaranteed anymore.

        Returns:
            self if ``return_early=False``, otherwise a :class:`~tensordict.utils.TensorDictFuture` instance.

        Note:
            Serialising in this fashion might be slow with deeply nested tensordicts, so
            it is not recommended to call this method inside a training loop.
        """
        prefix = Path(prefix) if prefix is not None else self._memmap_prefix
        if num_threads > 1:
            with (
                ThreadPoolExecutor(max_workers=num_threads)
                if not return_early
                else contextlib.nullcontext()
            ) as executor:
                if return_early:
                    executor = ThreadPoolExecutor(max_workers=num_threads)
                futures = []
                result = self._memmap_(
                    prefix=prefix,
                    copy_existing=copy_existing,
                    executor=executor,
                    futures=futures,
                    inplace=True,
                    like=False,
                    share_non_tensor=share_non_tensor,
                    existsok=existsok,
                )
                if not return_early:
                    concurrent.futures.wait(futures)
                    return result
                else:
                    return TensorDictFuture(futures, result)
        return self._memmap_(
            prefix=prefix,
            copy_existing=copy_existing,
            inplace=True,
            futures=None,
            executor=None,
            like=False,
            share_non_tensor=share_non_tensor,
            existsok=existsok,
        ).lock_()

    @abc.abstractmethod
    def make_memmap(
        self,
        key: NestedKey,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
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

        Keyword arguments:
            dtype (torch.dtype, optional): the dtype of the new tensor.

        Returns:
            A new memory mapped tensor.

        """
        ...

    @abc.abstractmethod
    def make_memmap_from_storage(
        self,
        key: NestedKey,
        storage: torch.UntypedStorage,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
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

        Keyword arguments:
            dtype (torch.dtype, optional): the dtype of the new tensor.

        Returns:
            A new memory mapped tensor with the given storage.

        """
        ...

    @abc.abstractmethod
    def make_memmap_from_tensor(
        self, key: NestedKey, tensor: torch.Tensor, *, copy_data: bool = True
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

        Keyword arguments:
            copy_data (bool, optionaL): if ``False``, the new tensor will share the metadata of the input such as
                shape and dtype, but the content will be empty. Defaults to ``True``.

        Returns:
            A new memory mapped tensor with the given storage.

        """
        ...

    def save(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        *,
        num_threads: int = 0,
        return_early: bool = False,
        share_non_tensor: bool = False,
    ) -> T:
        """Saves the tensordict to disk.

        This function is a proxy to :meth:`~.memmap`.
        """
        return self.memmap(
            prefix=prefix,
            copy_existing=copy_existing,
            num_threads=num_threads,
            return_early=return_early,
            share_non_tensor=share_non_tensor,
        )

    dumps = save

    def memmap(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        *,
        num_threads: int = 0,
        return_early: bool = False,
        share_non_tensor: bool = False,
        existsok: bool = True,
    ) -> T:
        """Writes all tensors onto a corresponding memory-mapped Tensor in a new tensordict.

        Args:
            prefix (str): directory prefix where the memory-mapped tensors will
                be stored. The directory tree structure will mimic the tensordict's.
            copy_existing (bool): If False (default), an exception will be raised if an
                entry in the tensordict is already a tensor stored on disk
                with an associated file, but is not saved in the correct
                location according to prefix.
                If ``True``, any existing Tensor will be copied to the new location.

        Keyword Args:
            num_threads (int, optional): the number of threads used to write the memmap
                tensors. Defaults to `0`.
            return_early (bool, optional): if ``True`` and ``num_threads>0``,
                the method will return a future of the tensordict.
            share_non_tensor (bool, optional): if ``True``, the non-tensor data will be
                shared between the processes and writing operation (such as inplace update
                or set) on any of the workers within a single node will update the value
                on all other workers. If the number of non-tensor leaves is high (e.g.,
                sharing large stacks of non-tensor data) this may result in OOM or similar
                errors. Defaults to ``False``.
            existsok (bool, optional): if ``False``, an exception will be raised if a tensor already
                exists in the same path. Defaults to ``True``.

        The TensorDict is then locked, meaning that any writing operations that
        isn't in-place will throw an exception (eg, rename, set or remove an
        entry).
        Once the tensordict is unlocked, the memory-mapped attribute is turned to ``False``,
        because cross-process identity is not guaranteed anymore.

        Returns:
            A new tensordict with the tensors stored on disk if ``return_early=False``,
            otherwise a :class:`~tensordict.utils.TensorDictFuture` instance.

        Note:
            Serialising in this fashion might be slow with deeply nested tensordicts, so
            it is not recommended to call this method inside a training loop.
        """
        prefix = Path(prefix) if prefix is not None else self._memmap_prefix

        if num_threads > 1:
            with (
                ThreadPoolExecutor(max_workers=num_threads)
                if not return_early
                else contextlib.nullcontext()
            ) as executor:
                if return_early:
                    executor = ThreadPoolExecutor(max_workers=num_threads)
                futures = []
                result = self._memmap_(
                    prefix=prefix,
                    copy_existing=copy_existing,
                    executor=executor,
                    futures=futures,
                    inplace=False,
                    like=False,
                    share_non_tensor=share_non_tensor,
                    existsok=existsok,
                )
                if not return_early:
                    concurrent.futures.wait(futures)
                    return result
                else:
                    return TensorDictFuture(futures, result)

        return self._memmap_(
            prefix=prefix,
            copy_existing=copy_existing,
            inplace=False,
            executor=None,
            like=False,
            futures=None,
            share_non_tensor=share_non_tensor,
            existsok=existsok,
        ).lock_()

    def memmap_like(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        *,
        existsok: bool = True,
        num_threads: int = 0,
        return_early: bool = False,
        share_non_tensor: bool = False,
    ) -> T:
        """Creates a contentless Memory-mapped tensordict with the same shapes as the original one.

        Args:
            prefix (str): directory prefix where the memory-mapped tensors will
                be stored. The directory tree structure will mimic the tensordict's.
            copy_existing (bool): If False (default), an exception will be raised if an
                entry in the tensordict is already a tensor stored on disk
                with an associated file, but is not saved in the correct
                location according to prefix.
                If ``True``, any existing Tensor will be copied to the new location.

        Keyword Args:
            num_threads (int, optional): the number of threads used to write the memmap
                tensors. Defaults to `0`.
            return_early (bool, optional): if ``True`` and ``num_threads>0``,
                the method will return a future of the tensordict.
            share_non_tensor (bool, optional): if ``True``, the non-tensor data will be
                shared between the processes and writing operation (such as inplace update
                or set) on any of the workers within a single node will update the value
                on all other workers. If the number of non-tensor leaves is high (e.g.,
                sharing large stacks of non-tensor data) this may result in OOM or similar
                errors. Defaults to ``False``.
            existsok (bool, optional): if ``False``, an exception will be raised if a tensor already
                exists in the same path. Defaults to ``True``.

        The TensorDict is then locked, meaning that any writing operations that
        isn't in-place will throw an exception (eg, rename, set or remove an
        entry).
        Once the tensordict is unlocked, the memory-mapped attribute is turned to ``False``,
        because cross-process identity is not guaranteed anymore.

        Returns:
            A new ``TensorDict`` instance with data stored as memory-mapped tensors if ``return_early=False``,
            otherwise a :class:`~tensordict.utils.TensorDictFuture` instance.

        .. note::
            This is the recommended method to write a set of large buffers
            on disk, as :meth:`~.memmap_()` will copy the information, which can
            be slow for large content.

        Examples:
            >>> td = TensorDict({
            ...     "a": torch.zeros((3, 64, 64), dtype=torch.uint8),
            ...     "b": torch.zeros(1, dtype=torch.int64),
            ... }, batch_size=[]).expand(1_000_000)  # expand does not allocate new memory
            >>> buffer = td.memmap_like("/path/to/dataset")

        """
        prefix = Path(prefix) if prefix is not None else self._memmap_prefix
        if num_threads > 1:
            with (
                ThreadPoolExecutor(max_workers=num_threads)
                if not return_early
                else contextlib.nullcontext()
            ) as executor:
                if return_early:
                    executor = ThreadPoolExecutor(max_workers=num_threads)
                futures = []
                # we create an empty copy of self
                # This is because calling MMapTensor.from_tensor(mmap_tensor) does nothing
                # if both are in filesystem
                input = self.apply(
                    lambda x: torch.empty((), device=x.device, dtype=x.dtype).expand(
                        x.shape
                    )
                )
                result = input._memmap_(
                    prefix=prefix,
                    copy_existing=copy_existing,
                    executor=executor,
                    futures=futures,
                    inplace=False,
                    like=True,
                    share_non_tensor=share_non_tensor,
                    existsok=existsok,
                )
                if not return_early:
                    concurrent.futures.wait(futures)
                    return result
                else:
                    return TensorDictFuture(futures, result)
        input = self.apply(
            lambda x: torch.empty((), device=x.device, dtype=x.dtype).expand(x.shape)
        )
        return input._memmap_(
            prefix=prefix,
            copy_existing=copy_existing,
            inplace=False,
            like=True,
            executor=None,
            futures=None,
            share_non_tensor=share_non_tensor,
            existsok=existsok,
        ).lock_()

    @classmethod
    def load(cls, prefix: str | Path, *args, **kwargs) -> T:
        """Loads a tensordict from disk.

        This class method is a proxy to :meth:`~.load_memmap`.
        """
        return cls.load_memmap(prefix, *args, **kwargs)

    def load_(self, prefix: str | Path, *args, **kwargs):
        """Loads a tensordict from disk within the current tensordict.

        This class method is a proxy to :meth:`~.load_memmap_`.
        """
        return self.load_memmap_(prefix, *args, **kwargs)

    @classmethod
    def load_memmap(
        cls,
        prefix: str | Path,
        device: torch.device | None = None,
        non_blocking: bool = False,
        *,
        out: TensorDictBase | None = None,
    ) -> T:
        """Loads a memory-mapped tensordict from disk.

        Args:
            prefix (str or Path to folder): the path to the folder where the
                saved tensordict should be fetched.
            device (torch.device or equivalent, optional): if provided, the
                data will be asynchronously cast to that device.
                Supports `"meta"` device, in which case the data isn't loaded
                but a set of empty "meta" tensors are created. This is
                useful to get a sense of the total model size and structure
                without actually opening any file.
            non_blocking (bool, optional): if ``True``, synchronize won't be
                called after loading tensors on device. Defaults to ``False``.
            out (TensorDictBase, optional): optional tensordict where the data
                should be written.

        Examples:
            >>> from tensordict import TensorDict
            >>> td = TensorDict.fromkeys(["a", "b", "c", ("nested", "e")], 0)
            >>> td.memmap("./saved_td")
            >>> td_load = TensorDict.load_memmap("./saved_td")
            >>> assert (td == td_load).all()

        This method also allows loading nested tensordicts.

        Examples:
            >>> nested = TensorDict.load_memmap("./saved_td/nested")
            >>> assert nested["e"] == 0

        A tensordict can also be loaded on "meta" device or, alternatively,
        as a fake tensor.

        Examples:
            >>> import tempfile
            >>> td = TensorDict({"a": torch.zeros(()), "b": {"c": torch.zeros(())}})
            >>> with tempfile.TemporaryDirectory() as path:
            ...     td.save(path)
            ...     td_load = TensorDict.load_memmap(path, device="meta")
            ...     print("meta:", td_load)
            ...     from torch._subclasses import FakeTensorMode
            ...     with FakeTensorMode():
            ...         td_load = TensorDict.load_memmap(path)
            ...         print("fake:", td_load)
            meta: TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=meta, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([]), device=meta, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=meta,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=meta,
                is_shared=False)
            fake: TensorDict(
                fields={
                    a: FakeTensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: FakeTensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=cpu,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=cpu,
                is_shared=False)

        """
        prefix = Path(prefix)

        metadata = _load_metadata(prefix)
        type_name = metadata["_type"]
        if type_name != str(cls):
            import tensordict

            for other_cls in tensordict.base._ACCEPTED_CLASSES:
                if str(other_cls) == type_name:
                    return other_cls._load_memmap(prefix, metadata)
            else:
                raise RuntimeError(
                    f"Could not find name {type_name} in {tensordict.base._ACCEPTED_CLASSES}. "
                    f"Did you call _register_tensor_class(cls) on {type_name}?"
                )
        if device is not None:
            device = torch.device(device)
        out = cls._load_memmap(prefix, metadata, device=device, out=out)
        if (
            not non_blocking
            and device is not None
            and device.type not in ("meta", "cuda")
        ):
            out._sync_all()
        return out

    def load_memmap_(
        self,
        prefix: str | Path,
    ):
        """Loads the content of a memory-mapped tensordict within the tensordict where ``load_memmap_`` is called.

        See :meth:`~tensordict.TensorDictBase.load_memmap` for more info.
        """
        is_memmap = self.is_memmap()
        with self.unlock_() if is_memmap else contextlib.nullcontext():
            self.load_memmap(prefix=prefix, device=self.device, out=self)
        if is_memmap:
            self.memmap_()
        return self

    def memmap_refresh_(self):
        """Refreshes the content of the memory-mapped tensordict if it has a :attr:`~tensordict.TensorDict.saved_path`.

        This method will raise an exception if no path is associated with it.

        """
        if not self.is_memmap() or self._memmap_prefix is None:
            raise RuntimeError(
                "Cannot refresh a TensorDict that is not memory mapped or has no path associated."
            )
        return self.load_memmap_(prefix=self.saved_path)

    @classmethod
    @abc.abstractmethod
    def _load_memmap(
        cls,
        prefix: Path,
        metadata: dict,
        device: torch.device | None = None,
        *,
        out=None,
    ): ...

    # Key functionality: set, get, set_, set_at_, update, update_
    @abc.abstractmethod
    def entry_class(self, key: NestedKey) -> type:
        """Returns the class of an entry, possibly avoiding a call to `isinstance(td.get(key), type)`.

        This method should be preferred to ``tensordict.get(key).shape`` whenever
        :meth:`.get` can be expensive to execute.

        """
        ...

    def set(
        self,
        key: NestedKey,
        item: CompatibleType,
        inplace: bool = False,
        *,
        non_blocking: bool = False,
        **kwargs: Any,
    ) -> T:
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
            >>> td = TensorDict({}, batch_size[3, 4])
            >>> td.set("x", torch.randn(3, 4))
            >>> y = torch.randn(3, 4, 5)
            >>> td.set("y", y, inplace=True) # works, even if 'y' is not present yet
            >>> td.set("y", torch.zeros_like(y), inplace=True)
            >>> assert (y==0).all() # y values are overwritten
            >>> td.set("y", torch.ones(5), inplace=True) # raises an exception as shapes mismatch

        """
        key = _unravel_key_to_tuple(key)
        # inplace is loose here, but for set_ it is constraining. We translate it
        # to None to tell _set_str and others to drop it if the key isn't found
        inplace = BEST_ATTEMPT_INPLACE if inplace else False
        return self._set_tuple(
            key, item, inplace=inplace, validated=False, non_blocking=non_blocking
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
    ): ...

    @abc.abstractmethod
    def _set_tuple(self, key, value, *, inplace, validated, non_blocking: bool): ...

    @lock_blocked
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
            NonTensorData(
                data='a string!',
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)

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
            NonTensorData(
                data='a string!',
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)

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
            data = getattr(value, "data", None)
            if data is None:
                return value.tolist()
            return data
        return value

    def filter_non_tensor_data(self) -> T:
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
    ) -> T:
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
            >>> td = TensorDict({}, batch_size[3, 4])
            >>> x = torch.randn(3, 4)
            >>> td.set("x", x)
            >>> td.set_at_("x", value=torch.ones(1, 4), index=slice(1))
            >>> assert (x[0] == 1).all()
        """
        key = _unravel_key_to_tuple(key)
        return self._set_at_tuple(
            key, value, index, validated=False, non_blocking=non_blocking
        )

    @abc.abstractmethod
    def _set_at_str(self, key, value, idx, *, validated, non_blocking: bool): ...

    @abc.abstractmethod
    def _set_at_tuple(self, key, value, idx, *, validated, non_blocking: bool): ...

    def set_(
        self,
        key: NestedKey,
        item: CompatibleType,
        *,
        non_blocking: bool = False,
    ) -> T:
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
            >>> td = TensorDict({}, batch_size[3, 4])
            >>> x = torch.randn(3, 4)
            >>> td.set("x", x)
            >>> td.set_("x", torch.zeros_like(x))
            >>> assert (x == 0).all()

        """
        key = _unravel_key_to_tuple(key)
        return self._set_tuple(
            key, item, inplace=True, validated=False, non_blocking=non_blocking
        )

    # Stack functionality
    @abc.abstractmethod
    def _stack_onto_(
        self,
        list_item: list[CompatibleType],
        dim: int,
    ) -> T:
        """Stacks a list of values onto an existing key while keeping the original storage.

        Args:
            key (str): name of the value
            list_item (list of torch.Tensor): value to be stacked and stored in the tensordict.
            dim (int): dimension along which the tensors should be stacked.

        Returns:
            self

        """
        ...

    def _stack_onto_at_(
        self,
        key: NestedKey,
        list_item: list[CompatibleType],
        dim: int,
        idx: IndexType,
    ) -> T:
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

    def get(self, key: NestedKey, *args, **kwargs) -> CompatibleType:
        """Gets the value stored with the input key.

        Args:
            key (str, tuple of str): key to be queried. If tuple of str it is
                equivalent to chained calls of getattr.
            default: default value if the key is not found in the tensordict.

                .. warning::
                    Currently, if a key is not present in the tensordict and no default
                    is passed, a `KeyError` is raised. From v0.7, this behaviour will be changed
                    and a `None` value will be returned instead. To adopt the new behaviour,
                    set the environment variable `export TD_GET_DEFAULTS_TO_NONE='1'` or call
                    :func`~tensordict.set_get_defaults_to_none`.

        Examples:
            >>> td = TensorDict({"x": 1}, batch_size=[])
            >>> td.get("x")
            tensor(1)
            >>> set_get_defaults_to_none(False) # Current default behaviour
            >>> td.get("y") # Raises KeyError
            >>> set_get_defaults_to_none(True)
            >>> td.get("y")
            None
        """
        key = _unravel_key_to_tuple(key)
        if not key:
            raise KeyError(_GENERIC_NESTED_ERR.format(key))
        # Find what the default is
        has_default = False
        if args:
            default = args[0]
            if len(args) > 1 or kwargs:
                raise TypeError("only one (keyword) argument is allowed.")
            has_default = True
        elif kwargs:
            default = kwargs.pop("default")
            if args or kwargs:
                raise TypeError("only one (keyword) argument is allowed.")
            has_default = True
        elif _GET_DEFAULTS_TO_NONE:
            default = None
        else:
            default = NO_DEFAULT
        try:
            return self._get_tuple(key, default=default)
        except KeyError:
            if _GET_DEFAULTS_TO_NONE is None and not has_default:
                # We raise an exception AND a warning because we want the user to know that this exception will
                # not be raised in the future
                warnings.warn(
                    "The entry you have queried with `get` is not present in the tensordict. "
                    "Currently, this raises an exception. "
                    "To align with `dict.get`, this behaviour will be changed in v0.7 and a `None` value will "
                    "be returned instead (no error will be raised). "
                    "To suppress this warning and use the new behaviour (recommended), call `tensordict.set_get_defaults_to_none(True)` or set the env variable `export TD_GET_DEFAULTS_TO_NONE='1'`. "
                    "To suppress this warning and keep the old behaviour, call `tensordict.set_get_defaults_to_none(False)` or set the env variable `export TD_GET_DEFAULTS_TO_NONE='0'`.",
                    category=DeprecationWarning,
                )
            raise

    @abc.abstractmethod
    def _get_str(self, key, default): ...

    @abc.abstractmethod
    def _get_tuple(self, key, default): ...

    def _get_tuple_maybe_non_tensor(self, key, default):
        result = self._get_tuple(key, default)
        if is_non_tensor(result):
            result_data = getattr(result, "data", NO_DEFAULT)
            if result_data is NO_DEFAULT:
                return result.tolist()
            return result_data
        return result

    def get_at(
        self, key: NestedKey, index: IndexType, default: CompatibleType = NO_DEFAULT
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
        key = _unravel_key_to_tuple(key)
        if not key:
            raise KeyError(_GENERIC_NESTED_ERR.format(key))
        # must be a tuple
        return self._get_at_tuple(key, index, default)

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

    def get_item_shape(self, key: NestedKey):
        """Returns the shape of the entry, possibly avoiding recurring to :meth:`~.get`."""
        return _shape(self.get(key))

    def update(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        clone: bool = False,
        inplace: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
        is_leaf: Callable[[Type], bool] | None = None,
    ) -> T:
        """Updates the TensorDict with values from either a dictionary or another TensorDict.

        Args:
            input_dict_or_td (TensorDictBase or dict): input data to be written
                in self.
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
            >>> assert td['a'] is not other_td['a']

        """
        if input_dict_or_td is self:
            # no op
            return self
        if is_leaf is None:
            is_leaf = _is_leaf_nontensor
        if keys_to_update is not None:
            if len(keys_to_update) == 0:
                return self
            keys_to_update = unravel_key_list(keys_to_update)
        for key, value in input_dict_or_td.items():
            key = _unravel_key_to_tuple(key)
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
                if not is_leaf(type(target)):
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
                        )
                        continue
                    elif isinstance(value, (dict,)) or _is_tensor_collection(
                        type(value)
                    ):
                        from tensordict._lazy import LazyStackedTensorDict

                        if isinstance(value, LazyStackedTensorDict) and not isinstance(
                            target, LazyStackedTensorDict
                        ):
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
                            )
                        continue
            self._set_tuple(
                key,
                value,
                inplace=BEST_ATTEMPT_INPLACE if inplace else False,
                validated=False,
                non_blocking=non_blocking,
            )
        return self

    def update_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        clone: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
    ) -> T:
        """Updates the TensorDict in-place with values from either a dictionary or another TensorDict.

        Unlike :meth:`~.update`, this function will throw an error if the key is unknown to ``self``.

        Args:
            input_dict_or_td (TensorDictBase or dict): input data to be written
                in self.
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

        """
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

            def inplace_update(name, dest, source):
                if source is None:
                    return None
                name = _unravel_key_to_tuple(name)
                for key in keys_to_update:
                    if key == name[: len(key)]:
                        dest.copy_(source, non_blocking=non_blocking)

        else:
            # Fastest route using _foreach_copy_
            keys, vals = self._items_list(True, True)
            new_keys, other_val = input_dict_or_td._items_list(
                True, True, sorting_keys=keys, default="intersection"
            )
            if len(new_keys) and _foreach_copy_ is not None:
                if len(other_val) != len(vals):
                    vals = dict(zip(keys, vals))
                    vals = [vals[k] for k in new_keys]
                _foreach_copy_(vals, other_val, non_blocking=non_blocking)
                return self
            named = False

            def inplace_update(dest, source):
                if source is None:
                    return None
                dest.copy_(source, non_blocking=non_blocking)

        self._apply_nest(
            inplace_update,
            input_dict_or_td,
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
    ) -> T:
        """Updates the TensorDict in-place at the specified index with values from either a dictionary or another TensorDict.

        Unlike  TensorDict.update, this function will throw an error if the key is unknown to the TensorDict.

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
                    a: Tensor(torch.Size([3, 4, 5]), dtype=torch.float32),
                    b: Tensor(torch.Size([3, 4, 10]), dtype=torch.float32)},
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
            if not isinstance(value, _ACCEPTED_CLASSES):
                raise TypeError(
                    f"Expected value to be one of types {_ACCEPTED_CLASSES} "
                    f"but got {type(value)}"
                )
            if clone:
                value = value.clone()
            self.set_at_((firstkey, *nextkeys), value, idx, non_blocking=non_blocking)
        return self

    def replace(self, *args, **kwargs):
        """Creates a shallow copy of the tensordict where entries have been replaced.

        Accepts one unnamed argument which must be a dictionary of a :class:`~tensordict.TensorDictBase` subclass.
        Additionally, first-level entries can be updated with the named keyword arguments.

        Returns:
            a copy of ``self`` with updated entries if the input is non-empty. If an empty dict or no dict is provided
            and the kwargs are empty, ``self`` is returned.

        """
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

    @lock_blocked
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

    def copy_(self, tensordict: T, non_blocking: bool = False) -> T:
        """See :obj:`TensorDictBase.update_`.

        The non-blocking argument will be ignored and is just present for
        compatibility with :func:`torch.Tensor.copy_`.
        """
        return self.update_(tensordict, non_blocking=non_blocking)

    def copy_at_(self, tensordict: T, idx: IndexType, non_blocking: bool = False) -> T:
        """See :obj:`TensorDictBase.update_at_`."""
        return self.update_at_(tensordict, idx, non_blocking=non_blocking)

    def is_empty(self) -> bool:
        """Checks if the tensordict contains any leaf."""
        for _ in self.keys(True, True):
            return False
        return True

    # Dict features: setdefault, items, values, keys, ...
    def setdefault(
        self, key: NestedKey, default: CompatibleType, inplace: bool = False
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
            is_leaf: an optional callable that indicates if a class is to be considered a
                leaf or not.

        Keyword Args:
            sort (bool, optional): whether the keys should be sorted. For nested keys,
                the keys are sorted according to their joined name (ie, ``("a", "key")`` will
                be counted as ``"a.key"`` for sorting). Be mindful that sorting may incur
                significant overhead when dealing with large tensordicts.
                Defaults to ``False``.

        """
        if sort:
            yield from sorted(
                self.items(
                    include_nested=include_nested,
                    leaves_only=leaves_only,
                    is_leaf=is_leaf,
                ),
                key=lambda item: (
                    item[0] if isinstance(item[0], str) else ".".join(item[0])
                ),
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
                        if not is_non_tensor(cls):
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
            is_leaf: an optional callable that indicates if a class is to be considered a
                leaf or not.

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
                        if not is_non_tensor(cls):
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

    @cache  # noqa: B019
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
            if is_dynamo_compiling():
                key_to_index = {key: i for i, key in enumerate(keys)}
                return [vals[key_to_index[key]] for key in sorting_keys]
            else:
                source = dict(zip(keys, vals))
                return [source[key] for key in sorting_keys]

    @cache  # noqa: B019
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
            if is_dynamo_compiling():
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
        if isinstance(default, str) and default == "intersection":
            new_keys = [
                key for key in sorting_keys if key in set(keys)
            ]  # intersection does not keep the sorting
        else:
            new_keys = list(set(sorting_keys).union(keys))
        if is_dynamo_compiling():
            ...
        else:
            source = dict(zip(keys, vals))
            vals = [source.get(key, default) for key in new_keys]
            return new_keys, vals

    @cache  # noqa: B019
    def _grad(self):
        result = self._fast_apply(
            lambda x: x.grad, propagate_lock=True, filter_empty=True
        )
        return result

    @cache  # noqa: B019
    def _data(self):
        result = self._fast_apply(lambda x: x.data, propagate_lock=True)
        return result

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
            is_leaf: an optional callable that indicates if a class is to be considered a
                leaf or not.

        Keyword Args:
            sort (bool, optional): whether the keys shoulbe sorted. For nested keys,
                the keys are sorted according to their joined name (ie, ``("a", "key")`` will
                be counted as ``"a.key"`` for sorting). Be mindful that sorting may incur
                significant overhead when dealing with large tensordicts.
                Defaults to ``False``.

        Examples:
            >>> from tensordict import TensorDict
            >>> data = TensorDict({"0": 0, "1": {"2": 2}}, batch_size=[])
            >>> data.keys()
            ['0', '1']
            >>> list(data.keys(leaves_only=True))
            ['0']
            >>> list(data.keys(include_nested=True, leaves_only=True))
            ['0', '1', ('1', '2')]
        """
        ...

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
        key = _unravel_key_to_tuple(key)
        if not key:
            raise KeyError(_GENERIC_NESTED_ERR.format(key))
        try:
            # using try/except for get/del is suboptimal, but
            # this is faster that checkink if key in self keys
            out = self.get(key, default)
            self.del_(key)
        except KeyError as err:
            # if default provided, 'out' value will return, else raise error
            if default is NO_DEFAULT:
                raise KeyError(
                    f"You are trying to pop key `{key}` which is not in dict "
                    f"without providing default value. "
                    f"Keys={self.keys(include_nested=isinstance(key, tuple))}."
                ) from err
        return out

    @property
    @cache  # noqa: B019
    def sorted_keys(self) -> list[NestedKey]:
        """Returns the keys sorted in alphabetical order.

        Does not support extra arguments.

        If the TensorDict is locked, the keys are cached until the tensordict
        is unlocked for faster execution.

        """
        return sorted(self.keys())

    @_as_context_manager()
    def flatten(self, start_dim=0, end_dim=-1):
        """Flattens all the tensors of a tensordict.

        Args:
            start_dim (int): the first dim to flatten
            end_dim (int): the last dim to flatten

        Examples:
            >>> td = TensorDict({
            ...     "a": torch.arange(60).view(3, 4, 5),
            ...     "b": torch.arange(12).view(3, 4)}, batch_size=[3, 4])
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
        if start_dim < 0:
            start_dim = self.ndim + start_dim
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
        # TODO: check that this works with nested tds of different batch size
        if self._has_names():
            names = [
                name
                for i, name in enumerate(self.names)
                if (i < start_dim or i > end_dim)
            ]
            names.insert(start_dim, None)
        else:
            names = None
        out = self._fast_apply(
            flatten, batch_size=batch_size, propagate_lock=True, names=names
        )
        return out

    @_as_context_manager()
    def unflatten(self, dim, unflattened_size):
        """Unflattens a tensordict dim expanding it to a desired shape.

        Args:
            dim (int): specifies the dimension of the input tensor to be
                unflattened.
            unflattened_size (shape): is the new shape of the unflattened
                dimension of the tensordict.

        Examples:
            >>> td = TensorDict({
            ...     "a": torch.arange(60).view(3, 4, 5),
            ...     "b": torch.arange(12).view(3, 4)},
            ...     batch_size=[3, 4])
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
        # TODO: check that this works with nested tds of different batch size
        out = self._fast_apply(unflatten, batch_size=batch_size, propagate_lock=True)
        if self._has_names():
            names = copy(self.names)
            for _ in range(len(unflattened_size) - 1):
                names.insert(dim, None)
            out.names = names
        return out

    @abc.abstractmethod
    def rename_key_(
        self, old_key: NestedKey, new_key: NestedKey, safe: bool = False
    ) -> T:
        """Renames a key with a new string and returns the same tensordict with the updated key name.

        Args:
            old_key (str or nested key): key to be renamed.
            new_key (str or nested key): new name of the entry.
            safe (bool, optional): if ``True``, an error is thrown when the new
                key is already present in the TensorDict.

        Returns:
            self

        """
        ...

    @abc.abstractmethod
    def del_(self, key: NestedKey) -> T:
        """Deletes a key of the tensordict.

        Args:
            key (NestedKey): key to be deleted

        Returns:
            self

        """
        ...

    # Distributed functionality
    def gather_and_stack(
        self, dst: int, group: "dist.ProcessGroup" | None = None
    ) -> T | None:
        """Gathers tensordicts from various workers and stacks them onto self in the destination worker.

        Args:
            dst (int): the rank of the destination worker where :func:`gather_and_stack` will be called.
            group (torch.distributed.ProcessGroup, optional): if set, the specified process group
                will be used for communication. Otherwise, the default process group
                will be used.
                Defaults to ``None``.

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
            [None for _ in range(dist.get_world_size(group=group))]
            if dst == dist.get_rank(group=group)
            else None
        )
        dist.gather_object(self, output, dst=dst, group=group)
        if dst == dist.get_rank(group=group):
            # remove self from output
            output = [item for i, item in enumerate(output) if i != dst]
            self.update(torch.stack(output, 0), inplace=True)
            return self
        return None

    def send(
        self,
        dst: int,
        *,
        group: "dist.ProcessGroup" | None = None,
        init_tag: int = 0,
        pseudo_rand: bool = False,
    ) -> None:  # noqa: D417
        """Sends the content of a tensordict to a distant worker.

        Args:
            dst (int): the rank of the destination worker where the content
                should be sent.

        Keyword Args:
            group (torch.distributed.ProcessGroup, optional): if set, the specified process group
                will be used for communication. Otherwise, the default process group
                will be used.
                Defaults to ``None``.
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
        self._send(dst, _tag=init_tag - 1, pseudo_rand=pseudo_rand, group=group)

    def _send(
        self,
        dst: int,
        _tag: int = -1,
        pseudo_rand: bool = False,
        group: "dist.ProcessGroup" | None = None,
    ) -> int:
        for key in self.sorted_keys:
            value = self._get_str(key, NO_DEFAULT)
            if isinstance(value, Tensor):
                pass
            elif _is_tensor_collection(type(value)):
                _tag = value._send(dst, _tag=_tag, pseudo_rand=pseudo_rand, group=group)
                continue
            else:
                raise NotImplementedError(f"Type {type(value)} is not supported.")
            if not pseudo_rand:
                _tag += 1
            else:
                _tag = int_generator(_tag + 1)
            dist.send(value, dst=dst, tag=_tag, group=group)

        return _tag

    def recv(
        self,
        src: int,
        *,
        group: "dist.ProcessGroup" | None = None,
        init_tag: int = 0,
        pseudo_rand: bool = False,
    ) -> int:  # noqa: D417
        """Receives the content of a tensordict and updates content with it.

        Check the example in the `send` method for context.

        Args:
            src (int): the rank of the source worker.

        Keyword Args:
            group (torch.distributed.ProcessGroup, optional): if set, the specified process group
                will be used for communication. Otherwise, the default process group
                will be used.
                Defaults to ``None``.
            init_tag (int): the ``init_tag`` used by the source worker.
            pseudo_rand (bool): if True, the sequence of tags will be pseudo-
                random, allowing to send multiple data from different nodes
                without overlap. Notice that the generation of these pseudo-random
                numbers is expensive (1e-5 sec/number), meaning that it could
                slow down the runtime of your algorithm.
                This value must match the one passed to :func:`send`.
                Defaults to ``False``.
        """
        return self._recv(src, _tag=init_tag - 1, pseudo_rand=pseudo_rand, group=group)

    def _recv(
        self,
        src: int,
        _tag: int = -1,
        pseudo_rand: bool = False,
        group: "dist.ProcessGroup" | None = None,
        non_blocking: bool = False,
    ) -> int:
        for key in self.sorted_keys:
            value = self._get_str(key, NO_DEFAULT)
            if isinstance(value, Tensor):
                pass
            elif _is_tensor_collection(type(value)):
                _tag = value._recv(src, _tag=_tag, pseudo_rand=pseudo_rand, group=group)
                continue
            else:
                raise NotImplementedError(f"Type {type(value)} is not supported.")
            if not pseudo_rand:
                _tag += 1
            else:
                _tag = int_generator(_tag + 1)
            dist.recv(value, src=src, tag=_tag, group=group)
            self._set_str(
                key, value, inplace=True, validated=True, non_blocking=non_blocking
            )

        return _tag

    def isend(
        self,
        dst: int,
        *,
        group: "dist.ProcessGroup" | None = None,
        init_tag: int = 0,
        pseudo_rand: bool = False,
    ) -> int:  # noqa: D417
        """Sends the content of the tensordict asynchronously.

        Args:
            dst (int): the rank of the destination worker where the content
                should be sent.

        Keyword Args:
            group (torch.distributed.ProcessGroup, optional): if set, the specified process group
                will be used for communication. Otherwise, the default process group
                will be used.
                Defaults to ``None``.
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
        return self._isend(dst, _tag=init_tag - 1, pseudo_rand=pseudo_rand, group=group)

    def _isend(
        self,
        dst: int,
        _tag: int = -1,
        _futures: list[torch.Future] | None = None,
        pseudo_rand: bool = False,
        group: "dist.ProcessGroup" | None = None,
    ) -> int:
        root = False
        if _futures is None:
            root = True
            _futures = []
        for key in self.sorted_keys:
            value = self._get_str(key, NO_DEFAULT)
            if _is_tensor_collection(type(value)):
                _tag = value._isend(
                    dst,
                    _tag=_tag,
                    pseudo_rand=pseudo_rand,
                    _futures=_futures,
                    group=group,
                )
                continue
            elif isinstance(value, Tensor):
                pass
            else:
                raise NotImplementedError(f"Type {type(value)} is not supported.")
            if not pseudo_rand:
                _tag += 1
            else:
                _tag = int_generator(_tag + 1)
            _future = dist.isend(value, dst=dst, tag=_tag, group=group)
            _futures.append(_future)
        if root:
            for _future in _futures:
                _future.wait()
        return _tag

    def irecv(
        self,
        src: int,
        *,
        group: "dist.ProcessGroup" | None = None,
        return_premature: bool = False,
        init_tag: int = 0,
        pseudo_rand: bool = False,
    ) -> tuple[int, list[torch.Future]] | list[torch.Future] | None:
        """Receives the content of a tensordict and updates content with it asynchronously.

        Check the example in the :meth:`~.isend` method for context.

        Args:
            src (int): the rank of the source worker.

        Keyword Args:
            group (torch.distributed.ProcessGroup, optional): if set, the specified process group
                will be used for communication. Otherwise, the default process group
                will be used.
                Defaults to ``None``.
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
            src,
            return_premature=return_premature,
            _tag=init_tag - 1,
            pseudo_rand=pseudo_rand,
            group=group,
        )

    def _irecv(
        self,
        src: int,
        return_premature: bool = False,
        _tag: int = -1,
        _future_list: list[torch.Future] = None,
        pseudo_rand: bool = False,
        group: "dist.ProcessGroup" | None = None,
    ) -> tuple[int, list[torch.Future]] | list[torch.Future] | None:
        root = False
        if _future_list is None:
            _future_list = []
            root = True

        for key in self.sorted_keys:
            value = self._get_str(key, NO_DEFAULT)
            if _is_tensor_collection(type(value)):
                _tag, _future_list = value._irecv(
                    src,
                    _tag=_tag,
                    _future_list=_future_list,
                    pseudo_rand=pseudo_rand,
                    group=group,
                )
                continue
            elif isinstance(value, Tensor):
                pass
            else:
                raise NotImplementedError(f"Type {type(value)} is not supported.")
            if not pseudo_rand:
                _tag += 1
            else:
                _tag = int_generator(_tag + 1)
            _future_list.append(dist.irecv(value, src=src, tag=_tag, group=group))
        if not root:
            return _tag, _future_list
        elif return_premature:
            return _future_list
        else:
            for future in _future_list:
                future.wait()
            return

    def reduce(
        self,
        dst,
        op=None,
        async_op=False,
        return_premature=False,
        group=None,
    ):
        """Reduces the tensordict across all machines.

        Only the process with ``rank`` dst is going to receive the final result.

        """
        if op is None:
            op = dist.ReduceOp.SUM
        return self._reduce(dst, op, async_op, return_premature, group=group)

    def _reduce(
        self,
        dst,
        op=None,
        async_op=False,
        return_premature=False,
        _future_list=None,
        group=None,
    ):
        if op is None:
            op = dist.ReduceOp.SUM
        root = False
        if _future_list is None:
            _future_list = []
            root = True
        for key in self.sorted_keys:
            value = self._get_str(key, NO_DEFAULT)
            if _is_tensor_collection(type(value)):
                _future_list = value._reduce(
                    dst=dst,
                    op=op,
                    async_op=async_op,
                    _future_list=_future_list,
                )
                continue
            elif isinstance(value, Tensor):
                pass
            else:
                raise NotImplementedError(f"Type {type(value)} is not supported.")
            _future_list.append(
                dist.reduce(value, dst=dst, op=op, async_op=async_op, group=group)
            )
        if not root:
            return _future_list
        elif async_op and return_premature:
            return _future_list
        elif async_op:
            for future in _future_list:
                future.wait()
            return

    # Apply and map functionality
    def apply_(self, fn: Callable, *others, **kwargs) -> T:
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
    ) -> T | None:
        """Applies a callable to all values stored in the tensordict and sets them in a new tensordict.

        The callable signature must be ``Callable[Tuple[Tensor, ...], Optional[Union[Tensor, TensorDictBase]]]``.

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
                if left untouched by the function.
                Defaults to ``False`` for backward compatibility.
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
    ) -> T | None:
        """Applies a key-conditioned callable to all values stored in the tensordict and sets them in a new atensordict.

        The callable signature must be ``Callable[Tuple[str, Tensor, ...], Optional[Union[Tensor, TensorDictBase]]]``.

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
                empty data structures won't be created and destroyed. Defaults to
                ``False`` for backward compatibility.
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

    @abc.abstractmethod
    def _multithread_apply_flat(
        self,
        fn: Callable,
        *others: T,
        call_on_nested: bool = False,
        default: Any = NO_DEFAULT,
        named: bool = False,
        nested_keys: bool = False,
        prefix: tuple = (),
        is_leaf: Callable = None,
        executor: ThreadPoolExecutor,
        futures: List[Future],
        local_futures: List,
    ) -> None: ...

    @abc.abstractmethod
    def _multithread_rebuild(
        self,
        *,
        batch_size: Sequence[int] | None = None,
        device: torch.device | None = NO_DEFAULT,
        names: Sequence[str] | None = NO_DEFAULT,
        inplace: bool = False,
        checked: bool = False,
        out: TensorDictBase | None = None,
        filter_empty: bool = False,
        executor: ThreadPoolExecutor,
        futures: List[Future],
        local_futures: List,
        subs_results: Dict[Future, Any] | None = None,
        multithread_set: bool = False,  # Experimental
        **constructor_kwargs,
    ) -> None: ...

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
        is_leaf: Callable = None,
        out: TensorDictBase | None = None,
        num_threads: int,
        call_when_done: Callable | None = None,
        **constructor_kwargs,
    ) -> T | None:
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

    @abc.abstractmethod
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
        is_leaf: Callable = None,
        out: TensorDictBase | None = None,
        **constructor_kwargs,
    ) -> T | None: ...

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
        is_leaf: Callable = None,
        propagate_lock: bool = False,
        out: TensorDictBase | None = None,
        num_threads: int = 0,
        checked: bool = True,
        **constructor_kwargs,
    ) -> T | None:
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

    def map(
        self,
        fn: Callable[[TensorDictBase], TensorDictBase | None],
        dim: int = 0,
        num_workers: int | None = None,
        *,
        out: TensorDictBase | None = None,
        chunksize: int | None = None,
        num_chunks: int | None = None,
        pool: mp.Pool | None = None,
        generator: torch.Generator | None = None,
        max_tasks_per_child: int | None = None,
        worker_threads: int = 1,
        index_with_generator: bool = False,
        pbar: bool = False,
        mp_start_method: str | None = None,
    ):
        """Maps a function to splits of the tensordict across one dimension.

        This method will apply a function to a tensordict instance by chunking
        it in tensordicts of equal size and dispatching the operations over the
        desired number of workers.

        The function signature should be ``Callabe[[TensorDict], Union[TensorDict, Tensor]]``.
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
                Signatures similar to ``Callabe[[TensorDict], Union[TensorDict, Tensor]]``
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
        fn: Callable[[TensorDictBase], TensorDictBase | None],
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
    ):
        """Maps a function to splits of the tensordict across one dimension iteratively.

        This is the iterable version of :meth:`~TensorDictBase.map`.

        This method will apply a function to a tensordict instance by chunking
        it in tensordicts of equal size and dispatching the operations over the
        desired number of workers. It will yield the results one at a time.

        The function signature should be ``Callabe[[TensorDict], Union[TensorDict, Tensor]]``.
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
                Signatures similar to ``Callabe[[TensorDict], Union[TensorDict, Tensor]]``
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
        dim_orig = dim
        if dim < 0:
            dim = self.ndim + dim
        if dim < 0 or dim >= self.ndim:
            raise ValueError(f"Got incompatible dimension {dim_orig}")

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
                    out = LazyStackedTensorDict.maybe_dense_stack(imaplist, dim)
                else:
                    out = torch.cat(imaplist, dim)
            return out

    # Stream
    def record_stream(self, stream: torch.cuda.Stream):
        """Marks the tensordict as having been used by this stream.

        When the tensordict is deallocated, ensure the tensor memory is not reused for other tensors until all work
        queued on stream at the time of deallocation is complete.

        See :meth:`~torch.Tensor.record_stream` for more information.`

        """
        if self._stream is not None and self._stream != stream:
            raise RuntimeError(
                "A stream is already associated with this TensorDict instance."
            )
        self._stream = stream

        def record(tensor):
            tensor.record_stream(stream)

        self._fast_apply(record, filter_empty=True)
        return self

    # point-wise arithmetic ops
    def __add__(self, other: TensorDictBase | torch.Tensor) -> T:
        return self.add(other)

    def __iadd__(self, other: TensorDictBase | torch.Tensor) -> T:
        return self.add_(other)

    def __abs__(self):
        return self.abs()

    def __truediv__(self, other: TensorDictBase | torch.Tensor) -> T:
        return self.div(other)

    def __itruediv__(self, other: TensorDictBase | torch.Tensor) -> T:
        return self.div_(other)

    def __mul__(self, other: TensorDictBase | torch.Tensor) -> T:
        return self.mul(other)

    def __imul__(self, other: TensorDictBase | torch.Tensor) -> T:
        return self.mul_(other)

    def __sub__(self, other: TensorDictBase | torch.Tensor) -> T:
        return self.sub(other)

    def __isub__(self, other: TensorDictBase | torch.Tensor) -> T:
        return self.sub_(other)

    def __pow__(self, other: TensorDictBase | torch.Tensor) -> T:
        return self.pow(other)

    def __ipow__(self, other: TensorDictBase | torch.Tensor) -> T:
        return self.pow_(other)

    def abs(self) -> T:
        """Computes the absolute value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_abs(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def abs_(self) -> T:
        """Computes the absolute value of each element of the TensorDict in-place."""
        torch._foreach_abs_(self._values_list(True, True))
        return self

    def acos(self) -> T:
        """Computes the :meth:`~torch.acos` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_acos(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def acos_(self) -> T:
        """Computes the :meth:`~torch.acos` value of each element of the TensorDict in-place."""
        torch._foreach_acos_(self._values_list(True, True))
        return self

    def exp(self) -> T:
        """Computes the :meth:`~torch.exp` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_exp(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def exp_(self) -> T:
        """Computes the :meth:`~torch.exp` value of each element of the TensorDict in-place."""
        torch._foreach_exp_(self._values_list(True, True))
        return self

    def neg(self) -> T:
        """Computes the :meth:`~torch.neg` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_neg(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def neg_(self) -> T:
        """Computes the :meth:`~torch.neg` value of each element of the TensorDict in-place."""
        torch._foreach_neg_(self._values_list(True, True))
        return self

    def reciprocal(self) -> T:
        """Computes the :meth:`~torch.reciprocal` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_reciprocal(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def reciprocal_(self) -> T:
        """Computes the :meth:`~torch.reciprocal` value of each element of the TensorDict in-place."""
        torch._foreach_reciprocal_(self._values_list(True, True))
        return self

    def sigmoid(self) -> T:
        """Computes the :meth:`~torch.sigmoid` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_sigmoid(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def sigmoid_(self) -> T:
        """Computes the :meth:`~torch.sigmoid` value of each element of the TensorDict in-place."""
        torch._foreach_sigmoid_(self._values_list(True, True))
        return self

    def sign(self) -> T:
        """Computes the :meth:`~torch.sign` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_sign(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def sign_(self) -> T:
        """Computes the :meth:`~torch.sign` value of each element of the TensorDict in-place."""
        torch._foreach_sign_(self._values_list(True, True))
        return self

    def sin(self) -> T:
        """Computes the :meth:`~torch.sin` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_sin(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def sin_(self) -> T:
        """Computes the :meth:`~torch.sin` value of each element of the TensorDict in-place."""
        torch._foreach_sin_(self._values_list(True, True))
        return self

    def sinh(self) -> T:
        """Computes the :meth:`~torch.sinh` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_sinh(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def sinh_(self) -> T:
        """Computes the :meth:`~torch.sinh` value of each element of the TensorDict in-place."""
        torch._foreach_sinh_(self._values_list(True, True))
        return self

    def tan(self) -> T:
        """Computes the :meth:`~torch.tan` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_tan(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def tan_(self) -> T:
        """Computes the :meth:`~torch.tan` value of each element of the TensorDict in-place."""
        torch._foreach_tan_(self._values_list(True, True))
        return self

    def tanh(self) -> T:
        """Computes the :meth:`~torch.tanh` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_tanh(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def tanh_(self) -> T:
        """Computes the :meth:`~torch.tanh` value of each element of the TensorDict in-place."""
        torch._foreach_tanh_(self._values_list(True, True))
        return self

    def trunc(self) -> T:
        """Computes the :meth:`~torch.trunc` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_trunc(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def trunc_(self) -> T:
        """Computes the :meth:`~torch.trunc` value of each element of the TensorDict in-place."""
        torch._foreach_trunc_(self._values_list(True, True))
        return self

    @implement_for("torch", None, "2.4")
    def norm(
        self,
        *,
        out=None,
        dtype: torch.dtype | None = None,
    ):
        """Computes the norm of each tensor in the tensordict.

        Keyword Args:
            out (TensorDict, optional): the output tensordict.
            dtype (torch.dtype, optional): the output dtype (torch>=2.4).

        """
        keys, vals = self._items_list(True, True, collapse=True)
        if dtype is not None:
            raise RuntimeError("dtype must be None for torch <= 2.3")
        vals = torch._foreach_norm(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            batch_size=[],
            propagate_lock=True,
            out=out,
        )

    @implement_for("torch", "2.4")
    def norm(  # noqa: F811
        self,
        *,
        out=None,
        dtype: torch.dtype | None = None,
    ):
        """Computes the norm of each tensor in the tensordict.

        Keyword Args:
            out (TensorDict, optional): the output tensordict.
            dtype (torch.dtype, optional): the output dtype (torch>=2.4).

        """
        keys, vals = self._items_list(True, True, collapse=True)
        vals = torch._foreach_norm(vals, dtype=dtype)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            batch_size=[],
            propagate_lock=True,
            out=out,
        )

    def lgamma(self) -> T:
        """Computes the :meth:`~torch.lgamma` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_lgamma(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def lgamma_(self) -> T:
        """Computes the :meth:`~torch.lgamma` value of each element of the TensorDict in-place."""
        torch._foreach_lgamma_(self._values_list(True, True))
        return self

    def frac(self) -> T:
        """Computes the :meth:`~torch.frac` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_frac(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def frac_(self) -> T:
        """Computes the :meth:`~torch.frac` value of each element of the TensorDict in-place."""
        torch._foreach_frac_(self._values_list(True, True))
        return self

    def expm1(self) -> T:
        """Computes the :meth:`~torch.expm1` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_expm1(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def expm1_(self) -> T:
        """Computes the :meth:`~torch.expm1` value of each element of the TensorDict in-place."""
        torch._foreach_expm1_(self._values_list(True, True))
        return self

    def log(self) -> T:
        """Computes the :meth:`~torch.log` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_log(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def log_(self) -> T:
        """Computes the :meth:`~torch.log` value of each element of the TensorDict in-place."""
        torch._foreach_log_(self._values_list(True, True))
        return self

    def log10(self) -> T:
        """Computes the :meth:`~torch.log10` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_log10(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def log10_(self) -> T:
        """Computes the :meth:`~torch.log10` value of each element of the TensorDict in-place."""
        torch._foreach_log10_(self._values_list(True, True))
        return self

    def log1p(self) -> T:
        """Computes the :meth:`~torch.log1p` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_log1p(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def log1p_(self) -> T:
        """Computes the :meth:`~torch.log1p` value of each element of the TensorDict in-place."""
        torch._foreach_log1p_(self._values_list(True, True))
        return self

    def log2(self) -> T:
        """Computes the :meth:`~torch.log2` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_log2(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def log2_(self) -> T:
        """Computes the :meth:`~torch.log2` value of each element of the TensorDict in-place."""
        torch._foreach_log2_(self._values_list(True, True))
        return self

    def ceil(self) -> T:
        """Computes the :meth:`~torch.ceil` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_ceil(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def ceil_(self) -> T:
        """Computes the :meth:`~torch.ceil` value of each element of the TensorDict in-place."""
        torch._foreach_ceil_(self._values_list(True, True))
        return self

    def floor(self) -> T:
        """Computes the :meth:`~torch.floor` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_floor(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def floor_(self) -> T:
        """Computes the :meth:`~torch.floor` value of each element of the TensorDict in-place."""
        torch._foreach_floor_(self._values_list(True, True))
        return self

    def round(self) -> T:
        """Computes the :meth:`~torch.round` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_round(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def round_(self) -> T:
        """Computes the :meth:`~torch.round` value of each element of the TensorDict in-place."""
        torch._foreach_round_(self._values_list(True, True))
        return self

    def erf(self) -> T:
        """Computes the :meth:`~torch.erf` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_erf(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def erf_(self) -> T:
        """Computes the :meth:`~torch.erf` value of each element of the TensorDict in-place."""
        torch._foreach_erf_(self._values_list(True, True))
        return self

    def erfc(self) -> T:
        """Computes the :meth:`~torch.erfc` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_erfc(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def erfc_(self) -> T:
        """Computes the :meth:`~torch.erfc` value of each element of the TensorDict in-place."""
        torch._foreach_erfc_(self._values_list(True, True))
        return self

    def asin(self) -> T:
        """Computes the :meth:`~torch.asin` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_asin(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def asin_(self) -> T:
        """Computes the :meth:`~torch.asin` value of each element of the TensorDict in-place."""
        torch._foreach_asin_(self._values_list(True, True))
        return self

    def atan(self) -> T:
        """Computes the :meth:`~torch.atan` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_atan(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def atan_(self) -> T:
        """Computes the :meth:`~torch.atan` value of each element of the TensorDict in-place."""
        torch._foreach_atan_(self._values_list(True, True))
        return self

    def cos(self) -> T:
        """Computes the :meth:`~torch.cos` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_cos(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def cos_(self) -> T:
        """Computes the :meth:`~torch.cos` value of each element of the TensorDict in-place."""
        torch._foreach_cos_(self._values_list(True, True))
        return self

    def cosh(self) -> T:
        """Computes the :meth:`~torch.cosh` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_cosh(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def cosh_(self) -> T:
        """Computes the :meth:`~torch.cosh` value of each element of the TensorDict in-place."""
        torch._foreach_cosh_(self._values_list(True, True))
        return self

    def _clone_recurse(self) -> TensorDictBase:  # noqa: D417
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
        result = self._fast_apply(
            lambda name, val: items.pop(name, None),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=False,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    def add(
        self,
        other: TensorDictBase | torch.Tensor,
        *,
        alpha: float | None = None,
        default: str | CompatibleType | None = None,
    ) -> TensorDictBase:  # noqa: D417
        r"""Adds :attr:`other`, scaled by :attr:`alpha`, to ``self``.

        .. math::
            \text{{out}}_i = \text{{input}}_i + \text{{alpha}} \times \text{{other}}_i

        Args:
            other (TensorDictBase or torch.Tensor): the tensor or TensorDict to add to ``self``.

        Keyword Args:
            alpha (Number, optional): the multiplier for :attr:`other`.
            default (torch.Tensor or str, optional): the default value to use for exclusive entries.
                If none is provided, the two tensordicts key list must match exactly.
                If ``default="intersection"`` is passed, only the intersecting key sets will be considered
                and other keys will be ignored.
                In all other cases, ``default`` will be used for all missing entries on both sides of the
                operation.

        """
        keys, vals = self._items_list(True, True)
        if _is_tensor_collection(type(other)):
            new_keys, other_val = other._items_list(
                True, True, sorting_keys=keys, default=default
            )
            if default is not None:
                as_dict = dict(zip(keys, vals))
                vals = [as_dict.get(key, default) for key in new_keys]
                keys = new_keys
        else:
            other_val = other
        if alpha is not None:
            vals = torch._foreach_add(vals, other_val, alpha=alpha)
        else:
            vals = torch._foreach_add(vals, other_val)
        items = dict(zip(keys, vals))
        result = self._fast_apply(
            lambda name, val: items.pop(name, None),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    def add_(self, other: TensorDictBase | float, *, alpha: float | None = None):
        """In-place version of :meth:`~.add`.

        .. note::
            In-place ``add`` does not support ``default`` keyword argument.
        """
        if _is_tensor_collection(type(other)):
            keys, vals = self._items_list(True, True)
            other_val = other._values_list(True, True, sorting_keys=keys)
        else:
            vals = self._values_list(True, True)
            other_val = other
        if alpha is not None:
            torch._foreach_add_(vals, other_val, alpha=alpha)
        else:
            torch._foreach_add_(vals, other_val)
        return self

    def lerp(
        self,
        end: TensorDictBase | torch.Tensor,
        weight: TensorDictBase | torch.Tensor | float,
    ):
        r"""Does a linear interpolation of two tensors :attr:`start` (given by ``self``) and :attr:`end` based on a scalar or tensor :attr:`weight`.

        .. math::
            \text{out}_i = \text{start}_i + \text{weight}_i \times (\text{end}_i - \text{start}_i)

        The shapes of :attr:`start` and :attr:`end` must be
        broadcastable. If :attr:`weight` is a tensor, then
        the shapes of :attr:`weight`, :attr:`start`, and :attr:`end` must be broadcastable.

        Args:
            end (TensorDict): the tensordict with the ending points.
            weight (TensorDict, tensor or float): the weight for the interpolation formula.

        """
        keys, vals = self._items_list(True, True)
        if _is_tensor_collection(type(end)):
            end_val = end._values_list(True, True)
        else:
            end_val = end
        if isinstance(weight, (float, torch.Tensor)):
            weight_val = weight
        elif _is_tensor_collection(type(weight)):
            weight_val = weight._values_list(True, True)
        else:
            weight_val = weight
        vals = torch._foreach_lerp(vals, end_val, weight_val)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def lerp_(self, end: TensorDictBase | float, weight: TensorDictBase | float):
        """In-place version of :meth:`~.lerp`."""
        if _is_tensor_collection(type(end)):
            end_val = end._values_list(True, True)
        else:
            end_val = end
        if isinstance(weight, (float, torch.Tensor)):
            weight_val = weight
        elif _is_tensor_collection(type(weight)):
            weight_val = weight._values_list(True, True)
        else:
            weight_val = weight
        torch._foreach_lerp_(self._values_list(True, True), end_val, weight_val)
        return self

    def addcdiv(
        self,
        other1: TensorDictBase | torch.Tensor,
        other2: TensorDictBase | torch.Tensor,
        value: float | None = 1,
    ):  # noqa: D417
        r"""Performs the element-wise division of :attr:`other1` by :attr:`other2`, multiplies the result by the scalar :attr:`value` and adds it to ``self``.

        .. math::
            \text{out}_i = \text{input}_i + \text{value} \times \frac{\text{tensor1}_i}{\text{tensor2}_i}

        The shapes of the elements of ``self``, :attr:`other1`, and :attr:`other2` must be
        broadcastable.

        For inputs of type `FloatTensor` or `DoubleTensor`, :attr:`value` must be
        a real number, otherwise an integer.

        Args:
            other1 (TensorDict or Tensor): the numerator tensordict (or tensor)
            tensor2 (TensorDict or Tensor): the denominator tensordict (or tensor)

        Keyword Args:
            value (Number, optional): multiplier for :math:`\text{tensor1} / \text{tensor2}`
        """
        keys, vals = self._items_list(True, True)
        if _is_tensor_collection(type(other1)):
            other1_val = other1._values_list(True, True)
        else:
            other1_val = other1
        if _is_tensor_collection(type(other2)):
            other2_val = other2._values_list(True, True)
        else:
            other2_val = other2
        vals = torch._foreach_addcdiv(vals, other1_val, other2_val, value=value)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def addcdiv_(self, other1, other2, *, value: float | None = 1):
        """The in-place version of :meth:`~.addcdiv`."""
        if _is_tensor_collection(type(other1)):
            other1_val = other1._values_list(True, True)
        else:
            other1_val = other1
        if _is_tensor_collection(type(other2)):
            other2_val = other2._values_list(True, True)
        else:
            other2_val = other2
        torch._foreach_addcdiv_(
            self._values_list(True, True), other1_val, other2_val, value=value
        )
        return self

    def addcmul(self, other1, other2, *, value: float | None = 1):  # noqa: D417
        r"""Performs the element-wise multiplication of :attr:`other1` by :attr:`other2`, multiplies the result by the scalar :attr:`value` and adds it to ``self``.

        .. math::
            \text{out}_i = \text{input}_i + \text{value} \times \text{other1}_i \times \text{other2}_i

        The shapes of ``self``, :attr:`other1`, and :attr:`other2` must be
        broadcastable.

        For inputs of type `FloatTensor` or `DoubleTensor`, :attr:`value` must be
        a real number, otherwise an integer.

        Args:
            other1 (TensorDict or Tensor): the tensordict or tensor to be multiplied
            other2 (TensorDict or Tensor): the tensordict or tensor to be multiplied

        Keyword Args:
            value (Number, optional): multiplier for :math:`other1 .* other2`
        """
        keys, vals = self._items_list(True, True)
        if _is_tensor_collection(type(other1)):
            other1_val = other1._values_list(True, True)
        else:
            other1_val = other1
        if _is_tensor_collection(type(other2)):
            other2_val = other2._values_list(True, True)
        else:
            other2_val = other2
        vals = torch._foreach_addcmul(vals, other1_val, other2_val, value=value)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def addcmul_(self, other1, other2, *, value: float | None = 1):
        """The in-place version of :meth:`~.addcmul`."""
        if _is_tensor_collection(type(other1)):
            other1_val = other1._values_list(True, True)
        else:
            other1_val = other1
        if _is_tensor_collection(type(other2)):
            other2_val = other2._values_list(True, True)
        else:
            other2_val = other2
        torch._foreach_addcmul_(
            self._values_list(True, True), other1_val, other2_val, value=value
        )
        return self

    def sub(
        self,
        other: TensorDictBase | float,
        *,
        alpha: float | None = None,
        default: str | CompatibleType | None = None,
    ):  # noqa: D417
        r"""Subtracts :attr:`other`, scaled by :attr:`alpha`, from ``self``.

        .. math::
            \text{{out}}_i = \text{{input}}_i - \text{{alpha}} \times \text{{other}}_i

        Supports broadcasting,
        type promotion, and integer, float, and complex inputs.

        Args:
            other (TensorDict, Tensor or Number): the tensor or number to subtract from ``self``.

        Keyword Args:
            alpha (Number): the multiplier for :attr:`other`.
            default (torch.Tensor or str, optional): the default value to use for exclusive entries.
                If none is provided, the two tensordicts key list must match exactly.
                If ``default="intersection"`` is passed, only the intersecting key sets will be considered
                and other keys will be ignored.
                In all other cases, ``default`` will be used for all missing entries on both sides of the
                operation.

        """
        keys, vals = self._items_list(True, True)
        if _is_tensor_collection(type(other)):
            new_keys, other_val = other._items_list(
                True, True, sorting_keys=keys, default=default
            )
            if default is not None:
                as_dict = dict(zip(keys, vals))
                vals = [as_dict.get(key, default) for key in new_keys]
                keys = new_keys
        else:
            other_val = other
        if alpha is not None:
            vals = torch._foreach_sub(vals, other_val, alpha=alpha)
        else:
            vals = torch._foreach_sub(vals, other_val)
        items = dict(zip(keys, vals))
        result = self._fast_apply(
            lambda name, val: items.pop(name, None),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    def sub_(self, other: TensorDictBase | float, alpha: float | None = None):
        """In-place version of :meth:`~.sub`.

        .. note::
            In-place ``sub`` does not support ``default`` keyword argument.

        """
        if _is_tensor_collection(type(other)):
            keys, vals = self._items_list(True, True)
            other_val = other._values_list(True, True, sorting_keys=keys)
        else:
            vals = self._values_list(True, True)
            other_val = other
        if alpha is not None:
            torch._foreach_sub_(vals, other_val, alpha=alpha)
        else:
            torch._foreach_sub_(vals, other_val)
        return self

    def mul_(self, other: TensorDictBase | torch.Tensor) -> T:
        """In-place version of :meth:`~.mul`.

        .. note::
            Inplace ``mul`` does not support ``default`` keyword argument.

        """
        if _is_tensor_collection(type(other)):
            keys, vals = self._items_list(True, True)
            other_val = other._values_list(True, True, sorting_keys=keys)
        else:
            vals = self._values_list(True, True)
            other_val = other
        torch._foreach_mul_(vals, other_val)
        return self

    def mul(
        self,
        other: TensorDictBase | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> T:  # noqa: D417
        r"""Multiplies :attr:`other` to ``self``.

        .. math::
            \text{{out}}_i = \text{{input}}_i \times \text{{other}}_i

        Supports broadcasting, type promotion, and integer, float, and complex inputs.

        Args:
            other (TensorDict, Tensor or Number): the tensor or number to subtract from ``self``.

        Keyword Args:
            default (torch.Tensor or str, optional): the default value to use for exclusive entries.
                If none is provided, the two tensordicts key list must match exactly.
                If ``default="intersection"`` is passed, only the intersecting key sets will be considered
                and other keys will be ignored.
                In all other cases, ``default`` will be used for all missing entries on both sides of the
                operation.

        """
        keys, vals = self._items_list(True, True)
        if _is_tensor_collection(type(other)):
            new_keys, other_val = other._items_list(
                True, True, sorting_keys=keys, default=default
            )
            if default is not None:
                as_dict = dict(zip(keys, vals))
                vals = [as_dict.get(key, default) for key in new_keys]
                keys = new_keys
        else:
            other_val = other
        vals = torch._foreach_mul(vals, other_val)
        items = dict(zip(keys, vals))
        result = self._fast_apply(
            lambda name, val: items.pop(name, None),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    def maximum_(self, other: TensorDictBase | torch.Tensor) -> T:
        """In-place version of :meth:`~.maximum`.

        .. note::
            Inplace ``maximum`` does not support ``default`` keyword argument.

        """
        if _is_tensor_collection(type(other)):
            keys, vals = self._items_list(True, True)
            other_val = other._values_list(True, True, sorting_keys=keys)
        else:
            vals = self._values_list(True, True)
            other_val = other
        torch._foreach_maximum_(vals, other_val)
        return self

    def maximum(
        self,
        other: TensorDictBase | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> T:  # noqa: D417
        """Computes the element-wise maximum of ``self`` and :attr:`other`.

        Args:
            other (TensorDict or Tensor): the other input tensordict or tensor.

        Keyword Args:
            default (torch.Tensor or str, optional): the default value to use for exclusive entries.
                If none is provided, the two tensordicts key list must match exactly.
                If ``default="intersection"`` is passed, only the intersecting key sets will be considered
                and other keys will be ignored.
                In all other cases, ``default`` will be used for all missing entries on both sides of the
                operation.

        """
        keys, vals = self._items_list(True, True)
        if _is_tensor_collection(type(other)):
            new_keys, other_val = other._items_list(
                True, True, sorting_keys=keys, default=default
            )
            if default is not None:
                as_dict = dict(zip(keys, vals))
                vals = [as_dict.get(key, default) for key in new_keys]
                keys = new_keys
        else:
            other_val = other
        vals = torch._foreach_maximum(vals, other_val)
        items = dict(zip(keys, vals))
        result = self._fast_apply(
            lambda name, val: items.pop(name, None),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    def minimum_(self, other: TensorDictBase | torch.Tensor) -> T:
        """In-place version of :meth:`~.minimum`.

        .. note::
            Inplace ``minimum`` does not support ``default`` keyword argument.

        """
        if _is_tensor_collection(type(other)):
            keys, vals = self._items_list(True, True)
            other_val = other._values_list(True, True, sorting_keys=keys)
        else:
            vals = self._values_list(True, True)
            other_val = other
        torch._foreach_minimum_(vals, other_val)
        return self

    def minimum(
        self,
        other: TensorDictBase | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> T:  # noqa: D417
        """Computes the element-wise minimum of ``self`` and :attr:`other`.

        Args:
            other (TensorDict or Tensor): the other input tensordict or tensor.

        Keyword Args:
            default (torch.Tensor or str, optional): the default value to use for exclusive entries.
                If none is provided, the two tensordicts key list must match exactly.
                If ``default="intersection"`` is passed, only the intersecting key sets will be considered
                and other keys will be ignored.
                In all other cases, ``default`` will be used for all missing entries on both sides of the
                operation.

        """
        keys, vals = self._items_list(True, True)
        if _is_tensor_collection(type(other)):
            new_keys, other_val = other._items_list(
                True, True, sorting_keys=keys, default=default
            )
            if default is not None:
                as_dict = dict(zip(keys, vals))
                vals = [as_dict.get(key, default) for key in new_keys]
                keys = new_keys
        else:
            other_val = other
        vals = torch._foreach_minimum(vals, other_val)
        items = dict(zip(keys, vals))
        result = self._fast_apply(
            lambda name, val: items.pop(name, None),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    def clamp_max_(self, other: TensorDictBase | torch.Tensor) -> T:
        """In-place version of :meth:`~.clamp_max`.

        .. note::
            Inplace ``clamp_max`` does not support ``default`` keyword argument.

        """
        if _is_tensor_collection(type(other)):
            keys, vals = self._items_list(True, True)
            other_val = other._values_list(True, True, sorting_keys=keys)
        else:
            vals = self._values_list(True, True)
            other_val = other
        torch._foreach_clamp_max_(vals, other_val)
        return self

    def clamp_max(
        self,
        other: TensorDictBase | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> T:  # noqa: D417
        """Clamps the elements of ``self`` to :attr:`other` if they're superior to that value.

        Args:
            other (TensorDict or Tensor): the other input tensordict or tensor.

        Keyword Args:
            default (torch.Tensor or str, optional): the default value to use for exclusive entries.
                If none is provided, the two tensordicts key list must match exactly.
                If ``default="intersection"`` is passed, only the intersecting key sets will be considered
                and other keys will be ignored.
                In all other cases, ``default`` will be used for all missing entries on both sides of the
                operation.

        """
        keys, vals = self._items_list(True, True)
        if _is_tensor_collection(type(other)):
            new_keys, other_val = other._items_list(
                True, True, sorting_keys=keys, default=default
            )
            if default is not None:
                as_dict = dict(zip(keys, vals))
                vals = [as_dict.get(key, default) for key in new_keys]
                keys = new_keys
        else:
            other_val = other
        vals = torch._foreach_clamp_max(vals, other_val)
        items = dict(zip(keys, vals))
        result = self._fast_apply(
            lambda name, val: items.pop(name, None),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    def clamp_min_(self, other: TensorDictBase | torch.Tensor) -> T:
        """In-place version of :meth:`~.clamp_min`.

        .. note::
            Inplace ``clamp_min`` does not support ``default`` keyword argument.

        """
        if _is_tensor_collection(type(other)):
            keys, vals = self._items_list(True, True)
            other_val = other._values_list(True, True, sorting_keys=keys)
        else:
            vals = self._values_list(True, True)
            other_val = other
        torch._foreach_clamp_min_(vals, other_val)
        return self

    def clamp_min(
        self,
        other: TensorDictBase | torch.Tensor,
        default: str | CompatibleType | None = None,
    ) -> T:  # noqa: D417
        """Clamps the elements of ``self`` to :attr:`other` if they're inferior to that value.

        Args:
            other (TensorDict or Tensor): the other input tensordict or tensor.

        Keyword Args:
            default (torch.Tensor or str, optional): the default value to use for exclusive entries.
                If none is provided, the two tensordicts key list must match exactly.
                If ``default="intersection"`` is passed, only the intersecting key sets will be considered
                and other keys will be ignored.
                In all other cases, ``default`` will be used for all missing entries on both sides of the
                operation.

        """
        keys, vals = self._items_list(True, True)
        if _is_tensor_collection(type(other)):
            new_keys, other_val = other._items_list(
                True, True, sorting_keys=keys, default=default
            )
            if default is not None:
                as_dict = dict(zip(keys, vals))
                vals = [as_dict.get(key, default) for key in new_keys]
                keys = new_keys
        else:
            other_val = other
        vals = torch._foreach_clamp_min(vals, other_val)
        items = dict(zip(keys, vals))
        result = self._fast_apply(
            lambda name, val: items.pop(name, None),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    def pow_(self, other: TensorDictBase | torch.Tensor) -> T:
        """In-place version of :meth:`~.pow`.

        .. note::
            Inplace ``pow`` does not support ``default`` keyword argument.

        """
        if _is_tensor_collection(type(other)):
            keys, vals = self._items_list(True, True)
            other_val = other._values_list(True, True, sorting_keys=keys)
        else:
            vals = self._values_list(True, True)
            other_val = other
        torch._foreach_pow_(vals, other_val)
        return self

    def pow(
        self,
        other: TensorDictBase | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> T:  # noqa: D417
        r"""Takes the power of each element in ``self`` with :attr:`other` and returns a tensor with the result.

        :attr:`other` can be either a single ``float`` number, a `Tensor` or a ``TensorDict``.

        When :attr:`other` is a tensor, the shapes of :attr:`input`
        and :attr:`other` must be broadcastable.

        Args:
            other (float, tensor or tensordict): the exponent value

        Keyword Args:
            default (torch.Tensor or str, optional): the default value to use for exclusive entries.
                If none is provided, the two tensordicts key list must match exactly.
                If ``default="intersection"`` is passed, only the intersecting key sets will be considered
                and other keys will be ignored.
                In all other cases, ``default`` will be used for all missing entries on both sides of the
                operation.

        """
        keys, vals = self._items_list(True, True)
        if _is_tensor_collection(type(other)):
            new_keys, other_val = other._items_list(
                True, True, sorting_keys=keys, default=default
            )
            if default is not None:
                as_dict = dict(zip(keys, vals))
                vals = [as_dict.get(key, default) for key in new_keys]
                keys = new_keys
        else:
            other_val = other
        vals = torch._foreach_pow(vals, other_val)
        items = dict(zip(keys, vals))
        result = self._fast_apply(
            lambda name, val: items.pop(name, None),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    def div_(self, other: TensorDictBase | torch.Tensor) -> T:
        """In-place version of :meth:`~.div`.

        .. note::
            Inplace ``div`` does not support ``default`` keyword argument.

        """
        if _is_tensor_collection(type(other)):
            keys, vals = self._items_list(True, True)
            other_val = other._values_list(True, True, sorting_keys=keys)
        else:
            vals = self._values_list(True, True)
            other_val = other
        torch._foreach_div_(vals, other_val)
        return self

    def div(
        self,
        other: TensorDictBase | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> T:  # noqa: D417
        r"""Divides each element of the input ``self`` by the corresponding element of :attr:`other`.

        .. math::
            \text{out}_i = \frac{\text{input}_i}{\text{other}_i}

        Supports broadcasting, type promotion and integer, float, tensordict or tensor inputs.
        Always promotes integer types to the default scalar type.

        Args:
            other (TensorDict, Tensor or Number): the divisor.

        Keyword Args:
            default (torch.Tensor or str, optional): the default value to use for exclusive entries.
                If none is provided, the two tensordicts key list must match exactly.
                If ``default="intersection"`` is passed, only the intersecting key sets will be considered
                and other keys will be ignored.
                In all other cases, ``default`` will be used for all missing entries on both sides of the
                operation.

        """
        keys, vals = self._items_list(True, True)
        if _is_tensor_collection(type(other)):
            new_keys, other_val = other._items_list(
                True, True, sorting_keys=keys, default=default
            )
            if default is not None:
                as_dict = dict(zip(keys, vals))
                vals = [as_dict.get(key, default) for key in new_keys]
                keys = new_keys
        else:
            other_val = other
        vals = torch._foreach_div(vals, other_val)
        items = dict(zip(keys, vals))
        result = self._fast_apply(
            lambda name, val: items.pop(name, None),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    def sqrt_(self):
        """In-place version of :meth:`~.sqrt`."""
        torch._foreach_sqrt_(self._values_list(True, True))
        return self

    def sqrt(self):
        """Computes the element-wise square root of ``self``."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_sqrt(vals)
        items = dict(zip(keys, vals))
        return self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    # Functorch compatibility
    @abc.abstractmethod
    @cache  # noqa: B019
    def _add_batch_dim(self, *, in_dim, vmap_level): ...

    @abc.abstractmethod
    @cache  # noqa: B019
    def _remove_batch_dim(self, vmap_level, batch_size, out_dim): ...

    @abc.abstractmethod
    @cache  # noqa: B019
    def _maybe_remove_batch_dim(self, funcname, vmap_level, batch_size, out_dim): ...

    # Validation and checks
    def _convert_to_tensor(
        self, array: Any
    ) -> Tensor | "NonTensorData" | TensorDictBase:  # noqa: F821
        # We are sure that array is not a dict or anything in _ACCEPTED_CLASSES
        castable = None
        if isinstance(array, (float, int, bool)):
            castable = True
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
            return torch.as_tensor(array, device=self.device)
        else:
            from tensordict.tensorclass import NonTensorData

            return NonTensorData(
                data=array,
                batch_size=self.batch_size,
                device=self.device,
                names=self._maybe_names(),
            )

    @abc.abstractmethod
    def _convert_to_tensordict(self, dict_value: dict[str, Any]) -> T: ...

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

    @abc.abstractmethod
    def _check_is_shared(self) -> bool: ...

    def _check_new_batch_size(self, new_size: torch.Size) -> None:
        batch_dims = len(new_size)
        for key, tensor in self.items():
            if _shape(tensor)[:batch_dims] != new_size:
                raise RuntimeError(
                    f"the tensor {key} has shape {_shape(tensor)} which "
                    f"is incompatible with the batch-size {new_size}."
                )

    @abc.abstractmethod
    def _check_device(self, *, raise_exception: bool = True) -> None | bool: ...

    def _validate_key(self, key: NestedKey) -> NestedKey:
        key = _unravel_key_to_tuple(key)
        if not key:
            raise KeyError(_GENERIC_NESTED_ERR.format(key))
        return key

    def _validate_value(
        self,
        value: CompatibleType | dict[str, CompatibleType],
        non_blocking: bool = False,
        *,
        check_shape: bool = True,
    ) -> CompatibleType | dict[str, CompatibleType]:
        cls = type(value)
        is_tc = None
        if issubclass(cls, dict):
            # We use non-blocking if someone's watching or if non-blocking is explicitly passed
            value = self._convert_to_tensordict(
                value, non_blocking=_device_recorder.marked or non_blocking
            )
            is_tc = True
        elif not issubclass(cls, _ACCEPTED_CLASSES):
            # If cls is not a tensor
            try:
                value = self._convert_to_tensor(value)
            except ValueError as err:
                raise ValueError(
                    f"TensorDict conversion only supports tensorclasses, tensordicts,"
                    f" numeric scalars and tensors. Got {type(value)}"
                ) from err
        batch_size = self.batch_size
        check_shape = check_shape and self.batch_size
        if (
            check_shape
            and batch_size
            and _shape(value)[: self.batch_dims] != batch_size
        ):
            # if TensorDict, let's try to map it to the desired shape
            if is_tc is None:
                is_tc = _is_tensor_collection(cls)
            if is_tc:
                # we must clone the value before not to corrupt the data passed to set()
                value = value.clone(recurse=False)
                value.batch_size = self.batch_size
            else:
                raise RuntimeError(
                    f"batch dimension mismatch, got self.batch_size"
                    f"={self.batch_size} and value.shape={_shape(value)}."
                )
        device = self.device
        if device is not None and value.device != device:
            if _device_recorder.marked and device.type != "cuda":
                _device_recorder.record_transfer(device)
            value = value.to(device, non_blocking=non_blocking)
        if check_shape:
            if is_tc is None:
                is_tc = _is_tensor_collection(cls)
            if not is_tc:
                return value
            has_names = self._has_names()
            # we do our best to match the dim names of the value and the
            # container.
            if has_names and value.names[: self.batch_dims] != self.names:
                # we clone not to corrupt the value
                value = value.clone(False).refine_names(*self.names)
            elif not has_names and value._has_names():
                self.names = value.names[: self.batch_dims]
        return value

    def __enter__(self):
        if not hasattr(self, "_last_op_queue"):
            self._last_op_queue = collections.deque()
        self._last_op_queue.append(self._last_op)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # During exit, updates mustn't be made in-place as the source and dest
        # storage location can be identical, resulting in a RuntimeError
        if exc_type is not None and issubclass(exc_type, Exception):
            return False
        _last_op = self._last_op_queue.pop()
        if _last_op is not None:
            last_op, (args, kwargs, out) = _last_op
            # TODO: transpose, flatten etc. as decorator should lock the content to make sure that no key is
            #  added or deleted
            _inv_caller = LAST_OP_MAPS.get(last_op)
            if _inv_caller is not None:
                return _inv_caller(self, args, kwargs, out)
            else:
                raise NotImplementedError(f"Unrecognised function {last_op}.")
        return self

    # Clone, select, exclude, empty
    def select(self, *keys: NestedKey, inplace: bool = False, strict: bool = True) -> T:
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
    ) -> T: ...

    def exclude(self, *keys: NestedKey, inplace: bool = False) -> T:
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
    ) -> T: ...

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

    def to_tensordict(self, *, retain_none: bool | None = None) -> T:
        """Returns a regular TensorDict instance from the TensorDictBase.

        Args:
            retain_none (bool): if ``True``, the ``None`` values from tensorclass instances
                will be written in the tensordict.
                Otherwise they will be discarded. Default: ``True``.

                .. note:: from v0.8, the default value will be switched to ``False``.

        Returns:
            a new TensorDict object containing the same values.

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
                        else value.to_tensordict(retain_none=retain_none)
                    )
                )
                for key, value in self.items(is_leaf=_is_leaf_nontensor)
            },
            device=self.device,
            batch_size=self.batch_size,
            names=self._maybe_names(),
        )

    def clone(self, recurse: bool = True, **kwargs) -> T:
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
    def _clone(self, recurse: bool = False): ...

    def copy(self):
        """Return a shallow copy of the tensordict (ie, copies the structure but not the data).

        Equivalent to `TensorDictBase.clone(recurse=False)`
        """
        return self.clone(recurse=False)

    def to_padded_tensor(self, padding=0.0, mask_key: NestedKey | None = None):
        """Converts all nested tensors to a padded version and adapts the batch-size accordingly.

        Args:
            padding (float): the padding value for the tensors in the tensordict.
                Defaults to ``0.0``.
            mask_key (NestedKey, optional): if provided, the key where a
                mask for valid values will be written.
                Will result in an error if the heterogeneous dimension
                isn't part of the tensordict batch-size.
                Defaults to ``None``

        """
        batch_size = self.batch_size
        if any(shape == -1 for shape in batch_size):
            new_batch_size = []
        else:
            new_batch_size = None
            if mask_key is not None:
                raise RuntimeError(
                    "mask_key should only be provided if the "
                    "heterogenous dimension is part of the batch-size."
                )
        padded_names = []

        def to_padded(name, x):
            if x.is_nested:
                padded_names.append(name)
                return torch.nested.to_padded_tensor(x, padding=padding)
            return x

        result = self._apply_nest(
            to_padded,
            batch_size=new_batch_size,
            named=True,
            nested_keys=True,
        )
        if new_batch_size is not None:
            result = result.auto_batch_size_(batch_dims=self.batch_dims)

            if mask_key:
                # take the first of the padded keys
                padded_key = padded_names[0]
                # write the mask
                val = self.get(padded_key)
                val = torch.nested.to_padded_tensor(
                    torch.ones_like(val, dtype=torch.bool), padding=False
                )
                if val.ndim > result.ndim:
                    val = val.flatten(result.ndim, -1)[..., -1].clone()
                result.set(mask_key, val)
        return result

    def as_tensor(self):
        def as_tensor(tensor):
            try:
                return tensor.as_tensor()
            except AttributeError:
                return tensor

        return self._fast_apply(as_tensor, propagate_lock=True)

    def to_dict(self, *, retain_none: bool = True) -> dict[str, Any]:
        """Returns a dictionary with key-value pairs matching those of the tensordict.

        Args:
            retain_none (bool): if ``True``, the ``None`` values from tensorclass instances
                will be written in the dictionary.
                Otherwise, they will be discarded. Default: ``True``.

        """
        result = {}
        for key, value in self.items():
            if _is_tensor_collection(type(value)):
                if (
                    not retain_none
                    and _is_non_tensor(type(value))
                    and value.data is None
                ):
                    continue
                value = value.to_dict(retain_none=retain_none)
            result[key] = value
        return result

    def numpy(self):
        """Converts a tensordict to a (possibly nested) dictionary of numpy arrays.

        Non-tensor data is exposed as such.

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> data = TensorDict({"a": {"b": torch.zeros(()), "c": "a string!"}})
            >>> print(data)
            TensorDict(
                fields={
                    a: TensorDict(
                        fields={
                            b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                            c: NonTensorData(data=a string!, batch_size=torch.Size([]), device=None)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> print(data.numpy())
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

    def to_namedtuple(self, dest_cls: type | None = None):
        """Converts a tensordict to a namedtuple.

        Args:
            dest_cls (Type, optional): an optional namedtuple class to use.

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> data = TensorDict({
            ...     "a_tensor": torch.zeros((3)),
            ...     "nested": {"a_tensor": torch.zeros((3)), "a_string": "zero!"}}, [3])
            >>> data.to_namedtuple()
            GenericDict(a_tensor=tensor([0., 0., 0.]), nested=GenericDict(a_tensor=tensor([0., 0., 0.]), a_string='zero!'))

        """

        def dict_to_namedtuple(dictionary):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    dictionary[key] = dict_to_namedtuple(value)
            cls = (
                collections.namedtuple("GenericDict", dictionary.keys())
                if dest_cls is None
                else dest_cls
            )
            return cls(**dictionary)

        return dict_to_namedtuple(self.to_dict(retain_none=False))

    @classmethod
    def from_namedtuple(cls, named_tuple, *, auto_batch_size: bool = False):
        """Converts a namedtuple to a TensorDict recursively.

        Keyword Args:
            auto_batch_size (bool, optional): if ``True``, the batch size will be computed automatically.
                Defaults to ``False``.

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> data = TensorDict({
            ...     "a_tensor": torch.zeros((3)),
            ...     "nested": {"a_tensor": torch.zeros((3)), "a_string": "zero!"}}, [3])
            >>> nt = data.to_namedtuple()
            >>> print(nt)
            GenericDict(a_tensor=tensor([0., 0., 0.]), nested=GenericDict(a_tensor=tensor([0., 0., 0.]), a_string='zero!'))
            >>> TensorDict.from_namedtuple(nt, auto_batch_size=True)
            TensorDict(
                fields={
                    a_tensor: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                    nested: TensorDict(
                        fields={
                            a_string: NonTensorData(data=zero!, batch_size=torch.Size([3]), device=None),
                            a_tensor: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([3]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3]),
                device=None,
                is_shared=False)

        """
        from tensordict import TensorDict

        def namedtuple_to_dict(namedtuple_obj):
            if is_namedtuple(namedtuple_obj):
                namedtuple_obj = namedtuple_obj._asdict()

            else:
                from torch.return_types import cummax, cummin, max, min

                if isinstance(namedtuple_obj, (min, cummin, max, cummax)):
                    namedtuple_obj = {
                        "values": namedtuple_obj.values,
                        "indices": namedtuple_obj.indices,
                    }
            for key, value in namedtuple_obj.items():
                if is_namedtuple(value):
                    namedtuple_obj[key] = namedtuple_to_dict(value)
            return dict(namedtuple_obj)

        result = TensorDict(namedtuple_to_dict(named_tuple))
        if auto_batch_size:
            result.auto_batch_size_()
        return result

    @classmethod
    def from_struct_array(
        cls, struct_array: np.ndarray, device: torch.device | None = None
    ) -> T:
        """Converts a structured numpy array to a TensorDict.

        The content of the resulting TensorDict will share the same memory content as the numpy array (it is a zero-copy
        operation). Changing values of the structured numpy array in-place will affect the content of the TensorDict.

        Examples:
            >>> x = np.array(
            ...     [("Rex", 9, 81.0), ("Fido", 3, 27.0)],
            ...     dtype=[("name", "U10"), ("age", "i4"), ("weight", "f4")],
            ... )
            >>> td = TensorDict.from_struct_array(x)
            >>> x_recon = td.to_struct_array()
            >>> assert (x_recon == x).all()
            >>> assert x_recon.shape == x.shape
            >>> # Try modifying x age field and check effect on td
            >>> x["age"] += 1
            >>> assert (td["age"] == np.array([10, 4])).all()

        """
        if cls is TensorDictBase:
            from tensordict._td import TensorDict

            cls = TensorDict
        td = cls(
            {name: struct_array[name] for name in struct_array.dtype.names},
            batch_size=struct_array.shape,
            device=device,
        )
        return td

    def to_struct_array(self):
        """Converts a tensordict to a numpy structured array.

        In a :meth:`.from_struct_array` - :meth:`.to_struct_array` loop, the content of the input and output
        arrays should match. However, `to_struct_array` will not keep the memory content of the original arrays.

        See :meth:`.from_struct_array` for more information.

        """
        from .utils import TORCH_TO_NUMPY_DTYPE_DICT

        keys, vals = zip(*self.items())
        _vals = []
        for v in vals:
            if is_tensor_collection(v):
                if is_non_tensor(v):
                    _vals.append(v.data)
                    continue
                _vals.append(v.to_struct_array())
                continue
            _vals.append(v)
        vals = _vals
        del _vals
        vals = tuple(v if not is_non_tensor(v) else v.data for v in vals)
        dtype = [
            (key, TORCH_TO_NUMPY_DTYPE_DICT.get(val.dtype, val.dtype))
            for key, val in zip(keys, vals)
        ]
        if self.ndim:
            return np.array(list(zip(*vals)), dtype=dtype)
        return np.array(vals, dtype=dtype)

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
            A :class:`~.tensordict.PersitentTensorDict` instance linked to the newly created file.

        Examples:
            >>> import tempfile
            >>> import timeit
            >>>
            >>> from tensordict import TensorDict, MemoryMappedTensor
            >>> td = TensorDict({
            ...     "a": MemoryMappedTensor.from_tensor(torch.zeros(()).expand(1_000_000)),
            ...     "b": {"c": MemoryMappedTensor.from_tensor(torch.zeros(()).expand(1_000_000, 3))},
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
        from tensordict.persistent import PersistentTensorDict

        out = PersistentTensorDict.from_dict(
            self,
            filename=filename,
            **kwargs,
        )
        if self._has_names():
            out.names = self.names
        return out

    def empty(
        self, recurse=False, *, batch_size=None, device=NO_DEFAULT, names=None
    ) -> T:  # noqa: D417
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
            # simply exclude the leaves
            result = self._exclude(*self.keys(True, True), set_shared=False)
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

    # Filling
    def zero_(self) -> T:
        """Zeros all tensors in the tensordict in-place."""

        def fn(item):
            item.zero_()

        self._fast_apply(fn=fn, call_on_nested=True, propagate_lock=True)
        return self

    def fill_(self, key: NestedKey, value: float | bool) -> T:
        """Fills a tensor pointed by the key with a given scalar value.

        Args:
            key (str or nested key): entry to be filled.
            value (Number or bool): value to use for the filling.

        Returns:
            self

        """
        key = _unravel_key_to_tuple(key)
        data = self._get_tuple(key, NO_DEFAULT)
        if _is_tensor_collection(type(data)):
            data._fast_apply(lambda x: x.fill_(value), inplace=True)
        else:
            data = data.fill_(value)
            self._set_tuple(key, data, inplace=True, validated=True, non_blocking=False)
        return self

    # Masking
    @abc.abstractmethod
    def masked_fill_(self, mask: Tensor, value: float | bool) -> T:
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
        ...

    @abc.abstractmethod
    def masked_fill(self, mask: Tensor, value: float | bool) -> T:
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
        ...

    def where(self, condition, other, *, out=None, pad=None):  # noqa: D417
        """Return a ``TensorDict`` of elements selected from either self or other, depending on condition.

        Args:
            condition (BoolTensor): When ``True`` (nonzero), yields ``self``,
                otherwise yields ``other``.
            other (TensorDictBase or Scalar): value (if ``other`` is a scalar)
                or values selected at indices where condition is ``False``.

        Keyword Args:
            out (TensorDictBase, optional): the output ``TensorDictBase`` instance.
            pad (scalar, optional): if provided, missing keys from the source
                or destination tensordict will be written as `torch.where(mask, self, pad)`
                or `torch.where(mask, pad, other)`. Defaults to ``None``, ie
                missing keys are not tolerated.

        """
        ...

    @abc.abstractmethod
    def masked_select(self, mask: Tensor) -> T:
        """Masks all tensors of the TensorDict and return a new TensorDict instance with similar keys pointing to masked values.

        Args:
            mask (torch.Tensor): boolean mask to be used for the tensors.
                Shape must match the TensorDict ``batch_size``.

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3, 4)},
            ...    batch_size=[3])
            >>> mask = torch.tensor([True, False, False])
            >>> td_mask = td.masked_select(mask)
            >>> td_mask.get("a")
            tensor([[0., 0., 0., 0.]])

        """
        ...

    @abc.abstractmethod
    def _change_batch_size(self, new_size: torch.Size) -> None: ...

    @abc.abstractmethod
    def is_contiguous(self) -> bool:
        """Returns a boolean indicating if all the tensors are contiguous."""
        ...

    @abc.abstractmethod
    def contiguous(self) -> T:
        """Returns a new tensordict of the same type with contiguous values (or self if values are already contiguous)."""
        ...

    @cache  # noqa: B019
    @_as_context_manager()
    def flatten_keys(
        self,
        separator: str = ".",
        inplace: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
    ) -> T:
        """Converts a nested tensordict into a flat one, recursively.

        The TensorDict type will be lost and the result will be a simple TensorDict instance.

        Args:
            separator (str, optional): the separator between the nested items.
            inplace (bool, optional): if ``True``, the resulting tensordict will
                have the same identity as the one where the call has been made.
                Defaults to ``False``.
            is_leaf (callable, optional): a callable over a class type returning
                a bool indicating if this class has to be considered as a leaf.

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

    @cache  # noqa: B019
    @_as_context_manager()
    def unflatten_keys(self, separator: str = ".", inplace: bool = False) -> T:
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
            if not is_dynamo_compiling():
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
        strict: bool = True,
        reproduce_struct: bool = False,
    ):
        """Splits the tensordict in subsets given one or more set of keys.

        The method will return ``N+1`` tensordicts, where ``N`` is the number of
        the arguments provided.

        Args:
            key_sets (sequence of Dict[in_key, out_key] or list of keys): the various splits.
            inplace (bool, optional): if ``True``, the keys are removed from ``self``
                in-place. Defaults to ``False``.
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
        else:
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
                    del self[key]
                except KeyError:
                    # We're good if strict is False
                    if strict:
                        raise
            last_out = self
        if not reproduce_struct:
            last_out.filter_empty_()
        outs.append(last_out)
        return tuple(outs)

    @abc.abstractmethod
    def _index_tensordict(
        self,
        index: IndexType,
        new_batch_size: torch.Size | None = None,
        names: List[str] | None = None,
    ) -> T: ...

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

        for value in self.values():
            if _is_tensor_collection(type(value)):
                value._propagate_lock(lock_parents_weakrefs, is_compiling=is_compiling)

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
    def lock_(self) -> T:
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
        is_compiling = is_dynamo_compiling()
        if is_compiling:
            _lock_warn()
        self._propagate_lock(is_compiling=is_compiling)
        return self

    @erase_cache
    def _propagate_unlock(self):
        # if we end up here, we can clear the graph associated with this td
        self._is_locked = False

        self._is_shared = False
        self._is_memmap = False

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
    def unlock_(self) -> T:
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

    # Conversion (device or dtype)
    @overload
    def to(
        self: T,
        device: Optional[Union[int, device]] = ...,
        dtype: Optional[Union[torch.device, str]] = ...,
        non_blocking: bool = ...,
    ) -> T: ...

    @overload
    def to(self: T, dtype: Union[torch.device, str], non_blocking: bool = ...) -> T: ...

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool = ...) -> T: ...

    @overload
    def to(self: T, *, other: T, non_blocking: bool = ...) -> T: ...

    @overload
    def to(self: T, *, batch_size: torch.Size) -> T: ...

    def _to_cuda_with_pin_mem(
        self, *, num_threads, device="cuda", non_blocking=None, to: Callable
    ):
        if self.is_empty():
            return self.to(device)
        keys, vals = self._items_list(
            leaves_only=True, include_nested=True, is_leaf=_NESTED_TENSORS_AS_LISTS
        )
        lkeys = len(keys)
        q_in = queue.SimpleQueue()
        q_out = queue.SimpleQueue()
        threads = []
        items = {}
        for key, val in _zip_strict(keys, vals):
            q_in.put_nowait((key, val))
        for _ in range(min(num_threads, lkeys)):
            thread = Thread(target=_pin_mem, args=(q_in, q_out))
            thread.start()
            threads.append(thread)
        try:
            while len(items) < lkeys:
                keyval = q_out.get(timeout=_PIN_MEM_TIMEOUT)
                if not isinstance(keyval, tuple):
                    raise keyval
                key, val = keyval
                items[key] = to(val)
        finally:
            for thread in threads:
                thread.join(timeout=_PIN_MEM_TIMEOUT)
        result = self._fast_apply(
            lambda name, val: items.get(name, val),
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            device=device,
        )
        return result

    def to(self, *args, **kwargs) -> T:
        """Maps a TensorDictBase subclass either on another device, dtype or to another TensorDictBase subclass (if permitted).

        Casting tensors to a new dtype is not allowed, as tensordicts are not bound to contain a single
        tensor dtype.

        Args:
            device (torch.device, optional): the desired device of the tensordict.
            dtype (torch.dtype, optional): the desired floating point or complex dtype of
                the tensordict.
            tensor (torch.Tensor, optional): Tensor whose dtype and device are the desired
                dtype and device for all tensors in this TensorDict.

        Keyword Args:
            non_blocking (bool, optional): whether the operations should be blocking.
            memory_format (torch.memory_format, optional): the desired memory
                format for 4D parameters and buffers in this tensordict.
            batch_size (torch.Size, optional): resulting batch-size of the
                output tensordict.
            other (TensorDictBase, optional): TensorDict instance whose dtype
                and device are the desired dtype and device for all tensors
                in this TensorDict.

                .. note::
                    Since :class:`~tensordict.TensorDictBase` instances do not have
                    a dtype, the dtype is gathered from the example leaves.
                    If there are more than one dtype, then no dtype
                    casting is undertook.

            non_blocking_pin (bool, optional): if ``True``, the tensors are pinned before
                being sent to device. This will be done asynchronously but can be
                controlled via the ``num_threads`` argument.

                .. note::
                    Calling ``tensordict.pin_memory().to("cuda")`` will usually
                    be much slower than ``tensordict.to("cuda", non_blocking_pin=True)`` as
                    the pin_memory is called asynchronously in the second case.
                    Multithreaded ``pin_memory`` will usually be beneficial if the tensors
                    are large and numerous: when there are too few tensors to be sent,
                    the overhead of spawning threads and collecting data outweighs the benefits
                    of multithreading, and if the tensors are small the overhead of iterating
                    over a long list is also prohibitively large.

            num_threads (int or None, optional): if ``non_blocking_pin=True``, the number
                of threads to be used for ``pin_memory``. By default,
                ``max(1, torch.get_num_threads())`` threads will be spawn.
                ``num_threads=0`` will cancel any
                multithreading for the `pin_memory()` calls.

        Returns:
            a new tensordict instance if the device differs from the tensordict
            device and/or if the dtype is passed. The same tensordict otherwise.
            ``batch_size`` only modifications are done in-place.

        .. note::
            If the TensorDict is consolidated, the resulting TensorDict will be consolidated too.
            Each new tensor will be a view on the consolidated storage cast to the desired device.

        Examples:
            >>> data = TensorDict({"a": 1.0}, [], device=None)
            >>> data_cuda = data.to("cuda:0")  # casts to cuda
            >>> data_int = data.to(torch.int)  # casts to int
            >>> data_cuda_int = data.to("cuda:0", torch.int)  # multiple casting
            >>> data_cuda = data.to(torch.randn(3, device="cuda:0"))  # using an example tensor
            >>> data_cuda = data.to(other=TensorDict({}, [], device="cuda:0"))  # using a tensordict example
        """
        non_blocking = kwargs.pop("non_blocking", None)

        (
            device,
            dtype,
            _,
            convert_to_format,
            batch_size,
            non_blocking_pin,
            num_threads,
        ) = _parse_to(*args, **kwargs)
        result = self

        if device is not None and dtype is None and device == self.device:
            return result

        if self.is_consolidated() and dtype is None:
            return self._to_consolidated(
                device=device,
                pin_memory=non_blocking_pin,
                num_threads=num_threads,
                non_blocking=non_blocking,
            )

        if non_blocking is None:
            sub_non_blocking = True
            non_blocking = False
        else:
            sub_non_blocking = non_blocking

        if convert_to_format is not None:

            def to(tensor):
                return tensor.to(
                    device,
                    dtype,
                    non_blocking=sub_non_blocking,
                    convert_to_format=convert_to_format,
                )

        else:

            def to(tensor):
                return tensor.to(
                    device=device, dtype=dtype, non_blocking=sub_non_blocking
                )

        apply_kwargs = {}
        if device is not None or dtype is not None:
            if non_blocking_pin and num_threads != 0:
                if num_threads is None:
                    num_threads = max(1, torch.get_num_threads() // 2)
                result = self._to_cuda_with_pin_mem(
                    num_threads=num_threads, to=to, device=device
                )
            else:
                apply_kwargs["device"] = device if device is not None else self.device
                apply_kwargs["batch_size"] = batch_size
                if non_blocking_pin:

                    def to_pinmem(tensor, _to=to):
                        return to(tensor.pin_memory())

                    result = result._fast_apply(
                        to_pinmem, propagate_lock=True, **apply_kwargs
                    )
                else:
                    result = result._fast_apply(to, propagate_lock=True, **apply_kwargs)
        if batch_size is not None:
            result.batch_size = batch_size
        if (
            device is not None
            and sub_non_blocking
            and not non_blocking
            and device.type != "cuda"
        ):
            self._sync_all()
        return result

    def _to_consolidated(self, *, device, pin_memory, num_threads, non_blocking):
        if num_threads is None:
            # unspecified num_threads should mean 0
            num_threads = 0
        storage = self._consolidated["storage"]
        if pin_memory:
            storage = storage.pin_memory()
        storage_cast = storage.to(device, non_blocking=True)
        untyped_storage = storage_cast.untyped_storage()

        def set_(x):
            if x.is_nested:
                from torch._subclasses.fake_tensor import FakeTensor
                from torch._subclasses.functional_tensor import FunctionalTensor
                from torch.nested._internal.nested_tensor import (
                    _tensor_symint_registry,
                    NestedTensor,
                )
                from torch.nested._internal.ops import extract_kwargs

                if x.layout != torch.jagged:
                    raise RuntimeError(
                        "to(device) with nested tensors that do not have a jagged layout is not implemented yet. "
                        "Please raise an issue on GitHub."
                    )
                kwargs = extract_kwargs(x)
                values = x._values
                lengths = x._lengths
                offsets = x._offsets
                kwargs["offsets"] = set_(offsets)
                if lengths is not None:
                    kwargs["lengths"] = set_(lengths)
                    ragged_source = lengths
                else:
                    ragged_source = offsets
                new_thing = kwargs.get("lengths", kwargs.get("offsets"))
                if isinstance(new_thing, (FakeTensor, FunctionalTensor)):
                    from torch._subclasses.functional_tensor import (
                        mb_unwrap_functional_tensor,
                    )

                    # Temporary hack until we have the union find
                    tgt = mb_unwrap_functional_tensor(new_thing)
                    src = mb_unwrap_functional_tensor(ragged_source)
                    tgt.nested_int_memo = src.nested_int_memo
                else:
                    _tensor_symint_registry[new_thing] = _tensor_symint_registry[
                        ragged_source
                    ]

                return NestedTensor(
                    set_(values),
                    **kwargs,
                )
            storage_offset = x.storage_offset()
            stride = x.stride()
            return x.new_empty(0, device=device).set_(
                untyped_storage,
                size=x.shape,
                stride=stride,
                storage_offset=storage_offset,
            )

        result = self._fast_apply(
            set_, device=torch.device(device), num_threads=num_threads
        )
        result._consolidated = {"storage": storage_cast}
        if "metadata" in self._consolidated:
            # faster than deepcopy
            def copy_dict(d):
                return {
                    k: v if not isinstance(v, dict) else copy_dict(v)
                    for k, v in d.items()
                }

            result._consolidated["metadata"] = copy_dict(self._consolidated["metadata"])
        if non_blocking in (False, None):
            if device.type == "cuda" and non_blocking is False:
                # sending to CUDA force sync
                cuda_device = device
            elif storage.device.type == "cuda":
                # sending from cuda: need sync unless intentionally not asked for
                cuda_device = storage.device.type
            else:
                cuda_device = None
            if cuda_device is not None:
                torch.cuda.current_stream(cuda_device).synchronize()

        return result

    def _sync_all(self):
        if _has_cuda:
            # TODO: dynamo doesn't like torch.cuda.is_initialized
            if not is_dynamo_compiling() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
        elif _has_mps:
            mps = getattr(torch, "mps", None)
            if mps is not None:
                mps.synchronize()

    def is_floating_point(self):
        for item in self.values(include_nested=True, leaves_only=True):
            if not item.is_floating_point():
                return False
        else:
            return True

    def double(self):
        r"""Casts all tensors to ``torch.bool``."""
        return self._fast_apply(lambda x: x.double(), propagate_lock=True)

    def float(self):
        r"""Casts all tensors to ``torch.float``."""
        return self._fast_apply(lambda x: x.float(), propagate_lock=True)

    def int(self):
        r"""Casts all tensors to ``torch.int``."""
        return self._fast_apply(lambda x: x.int(), propagate_lock=True)

    def bool(self):
        r"""Casts all tensors to ``torch.bool``."""
        return self._fast_apply(lambda x: x.bool(), propagate_lock=True)

    def half(self):
        r"""Casts all tensors to ``torch.half``."""
        return self._fast_apply(lambda x: x.half(), propagate_lock=True)

    def type(self, dst_type):
        r"""Casts all tensors to :attr:`dst_type`.

        Args:
            dst_type (type or string): the desired type

        """
        return self._fast_apply(lambda x: x.type(dst_type))

    # Gradient compatibility
    @property
    def requires_grad(self) -> bool:
        return any(v.requires_grad for v in self.values())

    def requires_grad_(self, requires_grad=True) -> T:
        """Change if autograd should record operations on this tensor: sets this tensor's requires_grad attribute in-place.

        Returns this tensordict.

        Args:
            requires_grad (bool, optional): whether or not autograd should record operations on this tensordict.
                Defaults to ``True``.

        """
        for val in self._values_list(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS):
            val.requires_grad_(requires_grad)
        return self

    @abc.abstractmethod
    def detach_(self) -> T:
        """Detach the tensors in the tensordict in-place.

        Returns:
            self.

        """
        ...

    @cache  # noqa: B019
    def detach(self) -> T:
        """Detach the tensors in the tensordict.

        Returns:
            a new tensordict with no tensor requiring gradient.

        """
        return self._fast_apply(
            lambda x: x.detach(),
            propagate_lock=True,
        )

    @_make_dtype_promotion
    def bfloat16(self): ...

    @_make_dtype_promotion
    def complex128(self): ...

    @_make_dtype_promotion
    def complex32(self): ...

    @_make_dtype_promotion
    def complex64(self): ...

    @_make_dtype_promotion
    def float16(self): ...

    @_make_dtype_promotion
    def float32(self): ...

    @_make_dtype_promotion
    def float64(self): ...

    @_make_dtype_promotion
    def int16(self): ...

    @_make_dtype_promotion
    def int32(self): ...

    @_make_dtype_promotion
    def int64(self): ...

    @_make_dtype_promotion
    def int8(self): ...

    @_make_dtype_promotion
    def qint32(self): ...

    @_make_dtype_promotion
    def qint8(self): ...

    @_make_dtype_promotion
    def quint4x2(self): ...

    @_make_dtype_promotion
    def quint8(self): ...

    @_make_dtype_promotion
    def uint16(self): ...

    @_make_dtype_promotion
    def uint32(self): ...

    @_make_dtype_promotion
    def uint64(self): ...

    @_make_dtype_promotion
    def uint8(self): ...


_ACCEPTED_CLASSES = (
    Tensor,
    TensorDictBase,
)


def _register_tensor_class(cls):
    global _ACCEPTED_CLASSES
    _ACCEPTED_CLASSES = set(_ACCEPTED_CLASSES)
    _ACCEPTED_CLASSES.add(cls)
    _ACCEPTED_CLASSES = tuple(_ACCEPTED_CLASSES)


_TENSOR_COLLECTION_MEMO = {}


def _is_tensor_collection(datatype: type) -> bool:
    is_dynamo = is_dynamo_compiling()
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
    return not _is_tensor_collection(cls)


def _is_leaf_nontensor(cls: Type) -> bool:
    if _is_tensor_collection(cls):
        return _is_non_tensor(cls)
    # if issubclass(cls, KeyedJaggedTensor):
    #     return False
    return issubclass(cls, torch.Tensor)


def _load_metadata(prefix: Path):
    filepath = prefix / "meta.json"
    with open(filepath, "rb") as json_metadata:
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

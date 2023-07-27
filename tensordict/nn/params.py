# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import inspect
import numbers
import re
from copy import copy
from functools import wraps
from typing import Any, Callable, Sequence

import torch

from tensordict import TensorDictBase
from tensordict.tensordict import (
    CompatibleType,
    lock_blocked,
    NO_DEFAULT,
    TD_HANDLED_FUNCTIONS,
    TensorDict,
)
from tensordict.utils import DeviceType, erase_cache, IndexType, NestedKey
from torch import nn, Tensor
from torch.utils._pytree import tree_map


def _get_args_dict(func, args, kwargs):
    signature = inspect.signature(func)
    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()

    args_dict = dict(bound_arguments.arguments)
    return args_dict


def _maybe_make_param(tensor):
    if (
        isinstance(tensor, Tensor)
        and not isinstance(tensor, nn.Parameter)
        and tensor.dtype in (torch.float, torch.double, torch.half)
    ):
        tensor = nn.Parameter(tensor)
    return tensor


class _unlock_and_set:
    def __new__(cls, *args, **kwargs):
        if len(args) and callable(args[0]):
            return cls(**kwargs)(args[0])
        return super().__new__(cls)

    def __init__(self, **only_for_kwargs):
        self.only_for_kwargs = only_for_kwargs

    def __call__(self, func):
        name = func.__name__

        @wraps(func)
        def new_func(_self, *args, **kwargs):
            if self.only_for_kwargs:
                arg_dict = _get_args_dict(func, (_self, *args), kwargs)
                for kwarg, exp_value in self.only_for_kwargs.items():
                    cur_val = arg_dict.get(kwarg, NO_DEFAULT)
                    if cur_val != exp_value:
                        # escape
                        meth = getattr(_self._param_td, name)
                        out = meth(*args, **kwargs)
                        return out
            args = tree_map(_maybe_make_param, args)
            kwargs = tree_map(_maybe_make_param, kwargs)
            with _self._param_td.unlock_():
                meth = getattr(_self._param_td, name)
                out = meth(*args, **kwargs)
            _self.__dict__["_parameters"] = _self._param_td.flatten_keys("_").to_dict()
            if out is _self._param_td:
                return _self
            return out

        return new_func


def _fallback(func):
    name = func.__name__

    @wraps(func)
    def new_func(self, *args, **kwargs):
        out = getattr(self._param_td, name)(*args, **kwargs)
        if out is self._param_td:
            return self
        return out

    return new_func


def _fallback_property(func):
    name = func.__name__

    @wraps(func)
    def new_func(self):
        out = getattr(self._param_td, name)
        if out is self._param_td:
            return self
        return out

    return property(new_func)


def _replace(func):
    name = func.__name__

    @wraps(func)
    def new_func(self, *args, **kwargs):
        out = getattr(self._param_td, name)(*args, **kwargs)
        if out is self._param_td:
            return self
        self._param_td = out
        return self

    return new_func


def _carry_over(func):
    name = func.__name__

    @wraps(func)
    def new_func(self, *args, **kwargs):
        out = getattr(self._param_td, name)(*args, **kwargs)
        return TensorDictParams(out, no_convert=True)

    return new_func


class TensorDictParams(TensorDictBase, nn.Module):
    r"""Holds a TensorDictBase instance full of parameters.

    This class exposes the contained parameters to a parent nn.Module
    such that iterating over the parameters of the module also iterates over
    the leaves of the tensordict.

    Indexing works exactly as the indexing of the wrapped tensordict.
    The parameter names will be registered within this module using :meth:`~.TensorDict.flatten_keys("_")`.
    Therefore, the result of :meth:`~.named_parameters()` and the content of the
    tensordict will differ slightly in term of key names.

    Any operation that sets a tensor in the tensordict will be augmented by
    a :class:`torch.nn.Parameter` conversion.

    Args:
        parameters (TensorDictBase): a tensordict to represent as parameters.
            Values will be converted to parameters unless ``no_convert=True``.

    Keyword Args:
        no_convert (bool): if ``True``, no conversion to ``nn.Parameter`` will occur.
            Defaults to ``False``.

    Examples:
        >>> from torch import nn
        >>> from tensordict import TensorDict
        >>> module = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 4))
        >>> params = TensorDict.from_module(module)
        >>> params.lock_()
        >>> p = TensorDictParams(params)
        >>> print(p)
        TensorDictParams(params=TensorDict(
            fields={
                0: TensorDict(
                    fields={
                        bias: Parameter(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
                        weight: Parameter(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=None,
                    is_shared=False),
                1: TensorDict(
                    fields={
                        bias: Parameter(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
                        weight: Parameter(shape=torch.Size([4, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False))
        >>> class CustomModule(nn.Module):
        ...     def __init__(self, params):
        ...         super().__init__()
        ...         self.params = params
        >>> m = CustomModule(p)
        >>> # the wrapper supports assignment and values are turned in Parameter
        >>> m.params['other'] = torch.randn(3)
        >>> assert isinstance(m.params['other'], nn.Parameter)

    """

    def __init__(self, parameters: TensorDictBase, *, no_convert=False):
        super().__init__()
        self._param_td = parameters
        if not no_convert:
            self._param_td = self._param_td.apply(
                lambda x: _maybe_make_param(x)
            ).lock_()
        self._parameters = parameters.flatten_keys("_").to_dict()
        self._is_locked = False
        self._locked_tensordicts = []
        self.__last_op_queue = None

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
        if func not in TDPARAM_HANDLED_FUNCTIONS or not all(
            issubclass(t, (Tensor, TensorDictBase)) for t in types
        ):
            return NotImplemented
        return TDPARAM_HANDLED_FUNCTIONS[func](*args, **kwargs)

    @classmethod
    def _flatten_key(cls, key):
        def make_valid_identifier(s):
            # Replace invalid characters with underscores
            s = re.sub(r"\W|^(?=\d)", "_", s)

            # Ensure the string starts with a letter or underscore
            if not s[0].isalpha() and s[0] != "_":
                s = "_" + s

            return s

        key_flat = "_".join(key)
        if not key_flat.isidentifier():
            key_flat = make_valid_identifier(key_flat)
        return key_flat

    @lock_blocked
    @_unlock_and_set
    def __setitem__(
        self,
        index: IndexType,
        value: TensorDictBase | dict | numbers.Number | CompatibleType,
    ) -> None:
        ...

    @lock_blocked
    @_unlock_and_set
    def set(
        self, key: NestedKey, item: CompatibleType, inplace: bool = False, **kwargs: Any
    ) -> TensorDictBase:
        ...

    def update(
        self,
        input_dict_or_td: dict[str, CompatibleType] | TensorDictBase,
        clone: bool = False,
        inplace: bool = False,
    ) -> TensorDictBase:
        if isinstance(input_dict_or_td, TensorDictBase):
            input_dict_or_td = input_dict_or_td.apply(_maybe_make_param)
        else:
            input_dict_or_td = tree_map(_maybe_make_param, input_dict_or_td)
        with self._param_td.unlock_():
            TensorDictBase.update(self, input_dict_or_td, clone=clone, inplace=inplace)
        return self

    @lock_blocked
    @_unlock_and_set
    def pop(
        self, key: NestedKey, default: str | CompatibleType = NO_DEFAULT
    ) -> CompatibleType:
        ...

    @lock_blocked
    @_unlock_and_set
    def rename_key_(
        self, old_key: str, new_key: str, safe: bool = False
    ) -> TensorDictBase:
        ...

    @_unlock_and_set
    def apply_(self, fn: Callable, *others) -> TensorDictBase:
        ...

    @_unlock_and_set(inplace=True)
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
        ...

    @_fallback
    def get(
        self, key: NestedKey, default: str | CompatibleType = NO_DEFAULT
    ) -> CompatibleType:
        ...

    @_fallback
    def __getitem__(self, index: IndexType) -> TensorDictBase:
        ...

    def to(self, dest: DeviceType | type | torch.Size, **kwargs) -> TensorDictBase:
        params = self._param_td.to(dest)
        if params is self._param_td:
            return self
        return TensorDictParams(params)

    def cpu(self):
        params = self._param_td.cpu()
        if params is self._param_td:
            return self
        return TensorDictParams(params)

    def cuda(self, device=None):
        params = self._param_td.cuda(device=device)
        if params is self._param_td:
            return self
        return TensorDictParams(params)

    def clone(self, recurse: bool = True) -> TensorDictBase:
        return TensorDictParams(self._param_td.clone(recurse=recurse))

    @_fallback
    def chunk(self, chunks: int, dim: int = 0) -> tuple[TensorDictBase, ...]:
        ...

    @_fallback
    def unbind(self, dim: int) -> tuple[TensorDictBase, ...]:
        ...

    @_fallback
    def to_tensordict(self):
        ...

    @_fallback
    def to_h5(
        self,
        filename,
        **kwargs,
    ):
        ...

    def __hash__(self):
        return hash((id(self), id(self._param_td)))

    @_fallback
    def __eq__(self, other: object) -> TensorDictBase:
        ...

    @_fallback
    def __ne__(self, other: object) -> TensorDictBase:
        ...

    def __getattr__(self, item: str) -> Any:
        try:
            return getattr(self._param_td, item)
        except AttributeError:
            return super().__getattr__(item)

    @_fallback
    def _change_batch_size(self, *args, **kwargs):
        ...

    @_fallback
    def _erase_names(self, *args, **kwargs):
        ...

    # @_unlock_and_set  # we need this as one sub-module could call _get_str, get a td and want to modify it
    @_fallback
    def _get_str(self, *args, **kwargs):
        ...

    # @_unlock_and_set
    @_fallback
    def _get_tuple(self, *args, **kwargs):
        ...

    @_fallback
    def _has_names(self, *args, **kwargs):
        ...

    @_unlock_and_set
    def _rename_subtds(self, *args, **kwargs):
        ...

    @_unlock_and_set
    def _set_at_str(self, *args, **kwargs):
        ...

    @_fallback
    def _set_at_tuple(self, *args, **kwargs):
        ...

    @_unlock_and_set
    def _set_str(self, *args, **kwargs):
        ...

    @_unlock_and_set
    def _set_tuple(self, *args, **kwargs):
        ...

    @_unlock_and_set
    def _create_nested_str(self, *args, **kwargs):
        ...

    @_fallback
    def _stack_onto_(self, *args, **kwargs):
        ...

    @_fallback_property
    def batch_size(self) -> torch.Size:
        ...

    @_fallback
    def contiguous(self, *args, **kwargs):
        ...

    @lock_blocked
    @_unlock_and_set
    def del_(self, *args, **kwargs):
        ...

    @_fallback
    def detach_(self, *args, **kwargs):
        ...

    @_fallback_property
    def device(self):
        ...

    @_fallback
    def entry_class(self, *args, **kwargs):
        ...

    @_fallback
    def is_contiguous(self, *args, **kwargs):
        ...

    @_fallback
    def keys(self, *args, **kwargs):
        ...

    @_fallback
    def masked_fill(self, *args, **kwargs):
        ...

    @_fallback
    def masked_fill_(self, *args, **kwargs):
        ...

    def memmap_(
        self, prefix: str | None = None, copy_existing: bool = False
    ) -> TensorDictBase:
        raise RuntimeError("Cannot build a memmap TensorDict in-place.")

    @_fallback_property
    def names(self):
        ...

    @_fallback
    def pin_memory(self, *args, **kwargs):
        ...

    @_unlock_and_set
    def select(self, *args, **kwargs):
        ...

    @_fallback
    def share_memory_(self, *args, **kwargs):
        ...

    @property
    def is_locked(self) -> bool:
        # Cannot be locked
        return self._is_locked

    @is_locked.setter
    def is_locked(self, value):
        self._is_locked = bool(value)

    @_fallback_property
    def is_shared(self) -> bool:
        ...

    @_fallback_property
    def is_memmap(self) -> bool:
        ...

    @_fallback_property
    def shape(self) -> torch.Size:
        ...

    @erase_cache
    def _propagate_unlock(self, lock_ids=None):
        if lock_ids is not None:
            self._lock_id.difference_update(lock_ids)
        else:
            lock_ids = set()
        self._is_locked = False

        unlocked_tds = [self]
        lock_ids.add(id(self))
        self._locked_tensordicts = []

        self._is_shared = False
        self._is_memmap = False
        return unlocked_tds

    unlock_ = TensorDict.unlock_
    lock_ = TensorDict.lock_

    @property
    def data(self):
        return self.apply(lambda x: x.data)

    @_unlock_and_set(inplace=True)
    def flatten_keys(
        self, separator: str = ".", inplace: bool = False
    ) -> TensorDictBase:
        ...

    @_unlock_and_set(inplace=True)
    def unflatten_keys(
        self, separator: str = ".", inplace: bool = False
    ) -> TensorDictBase:
        ...

    @_unlock_and_set(inplace=True)
    def exclude(self, *keys: str, inplace: bool = False) -> TensorDictBase:
        ...

    @_carry_over
    def transpose(self, dim0, dim1):
        ...

    @_carry_over
    def permute(
        self,
        *dims_list: int,
        dims: list[int] | None = None,
    ) -> TensorDictBase:
        ...

    @_carry_over
    def squeeze(self, dim: int | None = None) -> TensorDictBase:
        ...

    @_carry_over
    def unsqueeze(self, dim: int) -> TensorDictBase:
        ...

    @_unlock_and_set
    def create_nested(self, key):
        ...

    def __repr__(self):
        return f"TensorDictParams(params={self._param_td})"


TDPARAM_HANDLED_FUNCTIONS = copy(TD_HANDLED_FUNCTIONS)


def implements_for_tdparam(torch_function: Callable) -> Callable[[Callable], Callable]:
    """Register a torch function override for TensorDictParams."""

    @functools.wraps(torch_function)
    def decorator(func: Callable) -> Callable:
        TDPARAM_HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


@implements_for_tdparam(torch.empty_like)
def _empty_like(td: TensorDictBase, *args, **kwargs) -> TensorDictBase:
    try:
        tdclone = td.clone()
    except Exception as err:
        raise RuntimeError(
            "The tensordict passed to torch.empty_like cannot be "
            "cloned, preventing empty_like to be called. "
            "Consider calling tensordict.to_tensordict() first."
        ) from err
    return tdclone.data.apply_(lambda x: torch.empty_like(x, *args, **kwargs))

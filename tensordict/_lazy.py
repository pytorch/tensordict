# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import numbers
import os
import re
import textwrap
import weakref
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from copy import copy, deepcopy
from functools import wraps
from pathlib import Path
from textwrap import indent
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    OrderedDict,
    Sequence,
    Tuple,
    Type,
)

import numpy as np

import orjson as json
import torch

from tensordict.memmap import MemoryMappedTensor
from torch.nn.utils.rnn import pad_sequence

try:
    from functorch import dim as ftdim

    _has_funcdim = True
except ImportError:
    from tensordict.utils import _ftdim_mock as ftdim

    _has_funcdim = False
from tensordict._td import _SubTensorDict, _TensorDictKeysView, TensorDict
from tensordict.base import (
    _is_leaf_nontensor,
    _is_tensor_collection,
    _NESTED_TENSORS_AS_LISTS,
    _NESTED_TENSORS_AS_LISTS_NONTENSOR,
    _register_tensor_class,
    BEST_ATTEMPT_INPLACE,
    CompatibleType,
    is_tensor_collection,
    NO_DEFAULT,
    T,
    TensorDictBase,
)
from tensordict.utils import (
    _as_context_manager,
    _broadcast_tensors,
    _check_is_flatten,
    _check_is_unflatten,
    _get_shape_from_args,
    _getitem_batch_size,
    _infer_size_impl,
    _is_number,
    _maybe_correct_neg_dim,
    _parse_to,
    _recursive_unbind_list,
    _renamed_inplace_method,
    _shape,
    _td_fields,
    _unravel_key_to_tuple,
    _zip_strict,
    cache,
    convert_ellipsis_to_idx,
    DeviceType,
    erase_cache,
    expand_right,
    IndexType,
    infer_size_impl,
    is_non_tensor,
    is_tensorclass,
    list_to_stack,
    lock_blocked,
    NestedKey,
    unravel_key_list,
)
from torch import Tensor


_has_functorch = False
try:
    try:
        from torch._C._functorch import (  # @manual=fbcode//caffe2:torch
            _add_batch_dim,
            _remove_batch_dim,
            is_batchedtensor,
        )
    except ImportError:
        from functorch._C import is_batchedtensor  # @manual=fbcode//caffe2/functorch:_C

    _has_functorch = True
except ImportError:
    _has_functorch = False

    def is_batchedtensor(tensor: Tensor) -> bool:
        """Placeholder for the functorch function."""
        return False


class _LazyStackedTensorDictKeysView(_TensorDictKeysView):
    tensordict: LazyStackedTensorDict

    def __len__(self) -> int:
        return len(self._keys())

    def _keys(self) -> list[str]:
        result = self.tensordict._key_list()
        if self.is_leaf in (
            _NESTED_TENSORS_AS_LISTS,
            _NESTED_TENSORS_AS_LISTS_NONTENSOR,
        ):
            return [
                (key, str(i))
                for key in result
                for i in range(len(self.tensordict.tensordicts))
            ]
        return result

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

    def __repr__(self):
        return f"{type(self).__name__}({tuple(self)})"


def _fails_exclusive_keys(func):
    @wraps(func)
    def newfunc(self, *args, **kwargs):
        if self._has_exclusive_keys:
            raise RuntimeError(
                f"the method {func.__name__} cannot complete when there are exclusive keys."
            )
        parent_func = getattr(TensorDictBase, func.__name__, None)
        if parent_func is None:
            parent_func = getattr(TensorDict, func.__name__)
        return parent_func(self, *args, **kwargs)

    return newfunc


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
         stack_dim_name (str, optional): the name of the stack dimension.
            Defaults to ``None``.
        strict_shape (bool, optional): if ``True``, every tensordict's shapes must match.
            Defaults to ``False``.

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

    .. note:: Lazy stacks support assignment via lists. For consistency, the lists should be
        presented as `tensor.tolist()` data structure. This means that the length of the first
        level of the nested lists should match the first dimension of the lazy stack (whether or
        not this is the stack dimension).

            >>> td = LazyStackedTensorDict(TensorDict(), TensorDict(), stack_dim=0)
            >>> td["a"] = [torch.ones(2), torch.zeros(1)]
            >>> assert td[1]["a"] == torch.zeros(1)
            >>> td["b"] = ["a string", "another string"]
            >>> assert td[1]["b"] == "another string"

    .. note:: When using the :meth:`~.get` method, one can pass `as_nested_tensor`, `as_padded_tensor`
        or the `as_list` arguments to control how the data should be presented if the dimensions of the
        tensors mismatch. When passed, the nesting/padding will occur regardless of whether the
        dimensions mismatch or not.

    """

    _is_vmapped: bool = False
    _device: torch.device | None = None

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Callable:
        from tensordict._torch_func import LAZY_TD_HANDLED_FUNCTIONS

        if func in LAZY_TD_HANDLED_FUNCTIONS:
            if kwargs is None:
                kwargs = {}
            if func not in LAZY_TD_HANDLED_FUNCTIONS or not all(
                issubclass(t, (Tensor, TensorDictBase)) for t in types
            ):
                return NotImplemented
            return LAZY_TD_HANDLED_FUNCTIONS[func](*args, **kwargs)
        else:
            return super().__torch_function__(func, types, args, kwargs)

    _td_dim_name = None
    _safe = False
    _lazy = True

    def __init__(
        self,
        *tensordicts: T,
        stack_dim: int = 0,
        hook_out: callable | None = None,
        hook_in: callable | None = None,
        batch_size: Sequence[int] | None = None,
        device: torch.device | None = None,
        names: Sequence[str] | None = None,
        stack_dim_name: str | None = None,
        strict_shape: bool = False,
    ) -> None:
        self._is_locked = None

        # sanity check
        num_tds = len(tensordicts)
        batch_size = torch.Size(batch_size) if batch_size is not None else None
        if not num_tds:
            # create an empty tensor
            td0 = TensorDict(batch_size=batch_size, device=device, names=names)
            self._device = torch.device(device) if device is not None else None
        else:
            td0 = tensordicts[0]
            device = td0.device
        if stack_dim < 0:
            ndim = td0.ndim
            try:
                stack_dim = _maybe_correct_neg_dim(stack_dim, ndim=ndim + 1, shape=None)
            except Exception:
                raise RuntimeError(
                    f"Couldn't infer stack dim from negative value, got stack_dim={stack_dim}"
                )
        self.stack_dim = stack_dim
        self._reset_batch_size(td0, tensordicts, device, num_tds, strict_shape)
        if stack_dim > len(self.batch_size):
            raise RuntimeError(
                f"Stack dim {stack_dim} is too big for batch size {self.batch_size}."
            )

        self.tensordicts: list[TensorDictBase] = list(tensordicts)
        self.hook_out = hook_out
        self.hook_in = hook_in
        if batch_size is not None and batch_size != self.batch_size and num_tds != 0:
            raise RuntimeError(
                f"batch_size does not match self.batch_size: {batch_size} vs {self.batch_size}."
            )
        if stack_dim_name is not None:
            self._td_dim_name = stack_dim_name

    @classmethod
    def _new_lazy_unsafe(
        cls,
        *tensordicts: T,
        stack_dim: int = 0,
        hook_out: callable | None = None,
        hook_in: callable | None = None,
        batch_size: Sequence[int] | None = None,
        device: torch.device | None = None,
        names: Sequence[str] | None = None,
        stack_dim_name: str | None = None,
        strict_shape: bool = False,
    ) -> None:
        self = cls.__new__(cls)
        self._is_locked = None

        # sanity check
        num_tds = len(tensordicts)
        batch_size = torch.Size(batch_size) if batch_size is not None else None
        if not num_tds:
            # create an empty tensor
            td0 = TensorDict(batch_size=batch_size, device=device, names=names)
            self._device = torch.device(device) if device is not None else None
        else:
            td0 = tensordicts[0]
            # device = td0.device
        _batch_size = td0.batch_size

        for td in tensordicts[1:]:
            _bs = td.batch_size
            if _bs != _batch_size:
                _batch_size = torch.Size(
                    [s if _bs[i] == s else -1 for i, s in enumerate(_batch_size)]
                )
        self.tensordicts: list[TensorDictBase] = list(tensordicts)
        self.stack_dim = stack_dim
        self._batch_size = self._compute_batch_size(_batch_size, stack_dim, num_tds)
        self.hook_out = hook_out
        self.hook_in = hook_in
        if stack_dim_name is not None:
            self._td_dim_name = stack_dim_name
        return self

    # These attributes should never be set
    @property
    def _is_shared(self):
        return all(td._is_shared for td in self.tensordicts)

    @property
    def _is_memmap(self):
        return all(td._is_memmap for td in self.tensordicts)

    @property
    @cache  # noqa: B019
    def _has_exclusive_keys(self):
        keys = None
        for td in self.tensordicts:
            _keys = set(td.keys(True, True))
            if keys is None:
                keys = _keys
            else:
                if keys != _keys:
                    return True
        else:
            return False

    @_fails_exclusive_keys
    def to_dict(
        self, *, retain_none: bool = True, convert_tensors: bool = False
    ) -> dict[str, Any]: ...

    def _reduce_get_metadata(self):
        metadata = {}
        metadata["stack_dim"] = self.stack_dim
        metadata["stack_dim_name"] = self._td_dim_name
        metadata["is_locked"] = self.is_locked
        return metadata

    @classmethod
    def from_dict(
        cls,
        input_dict: List[Dict[NestedKey, Any]],
        *other,
        auto_batch_size: bool = False,
        batch_size=None,
        device=None,
        batch_dims=None,
        stack_dim_name=None,
        stack_dim=0,
    ):
        return cls._new_lazy_unsafe(
            *(
                TensorDict.from_dict(
                    input_dict[str(i)],
                    *other,
                    auto_batch_size=auto_batch_size,
                    device=device,
                    batch_dims=batch_dims,
                    batch_size=batch_size,
                )
                for i in range(len(input_dict))
            ),
            stack_dim=stack_dim,
            stack_dim_name=stack_dim_name,
        )

    @_fails_exclusive_keys
    def state_dict(
        self,
        destination=None,
        prefix="",
        keep_vars=False,
        flatten=False,
    ) -> OrderedDict[str, Any]: ...

    @_fails_exclusive_keys
    def flatten_keys(
        self,
        separator: str = ".",
        inplace: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
    ) -> T: ...

    @_fails_exclusive_keys
    def unflatten_keys(self, separator: str = ".", inplace: bool = False) -> T: ...

    @property
    def device(self) -> torch.device | None:
        # devices might have changed, so we check that they're all the same
        if self.tensordicts:
            device = self.tensordicts[0].device
            for td in self.tensordicts:
                if device != td.device:
                    return None
            return device
        return self._device

    @device.setter
    def device(self, value: DeviceType) -> None:
        if not self.tensordicts:
            self._device = torch.device(value) if value is not None else value
            return
        for t in self.tensordicts:
            t.device = value

    def clear_device_(self) -> T:
        for td in self.tensordicts:
            td.clear_device_()
        return self

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
            if re.match(
                r"Failed to stack tensors within a tensordict",
                str(err),
            ):
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
                            s1 if s1 == s2 else -1
                            for (s1, s2) in _zip_strict(shape, _shape)
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
        return are_memmap[0]

    def _reset_batch_size(
        self,
        td0: TensorDictBase,
        tensordicts: list[TensorDictBase],
        device: torch.device,
        num_tds: int,
        strict_shape: bool,
    ):
        _batch_size = td0.batch_size
        stack_dim = self.stack_dim

        for td in tensordicts[1:]:
            if not is_tensor_collection(td):
                raise TypeError(
                    "Expected all inputs to be TensorDictBase instances but got "
                    f"{type(td)} instead."
                )
            _bs = td.batch_size
            _device = td.device
            if device != _device:
                raise RuntimeError(f"devices differ, got {device} and {_device}")
            if _bs != _batch_size:
                if strict_shape or len(_bs) != len(_batch_size):
                    raise RuntimeError(
                        f"batch sizes in tensordicts differs, LazyStackedTensorDict "
                        f"cannot be created. Got td[0].batch_size={_batch_size} "
                        f"and td[i].batch_size={_bs}. If the length match and you wish "
                        f"to stack these tensordicts, set strict_shape to False."
                    )
                else:
                    _batch_size = torch.Size(
                        [s if _bs[i] == s else -1 for i, s in enumerate(_batch_size)]
                    )
        self._batch_size = self._compute_batch_size(_batch_size, stack_dim, num_tds)

    @staticmethod
    def _compute_batch_size(
        batch_size: torch.Size, stack_dim: int, num_tds: int
    ) -> torch.Size:
        s = list(batch_size)
        s.insert(stack_dim, num_tds)
        return torch.Size(s)

    def _set_str(
        self,
        key: NestedKey,
        value: dict[str, CompatibleType] | CompatibleType,
        *,
        inplace: bool,
        validated: bool,
        ignore_lock: bool = False,
        non_blocking: bool = False,
    ) -> T:
        try:
            inplace = self._convert_inplace(inplace, key)
        except KeyError as e:
            raise KeyError(
                "setting a value in-place on a stack of TensorDict is only "
                "permitted if all members of the stack have this key in "
                "their register."
            ) from e
        if not validated:
            value = self._validate_value(
                value,
                non_blocking=non_blocking,
                check_shape=not (isinstance(value, list) and list_to_stack()),
            )
            validated = True
        if self._is_vmapped:
            value = self.hook_in(value)
        if isinstance(value, list):
            if self.stack_dim == 0:
                values = list(value)
            else:
                values = _recursive_unbind_list(value, self.stack_dim)
        else:
            values = value.unbind(self.stack_dim)
        for tensordict, item in _zip_strict(self.tensordicts, values):
            tensordict._set_str(
                key,
                item,
                inplace=inplace,
                validated=validated,
                ignore_lock=ignore_lock,
                non_blocking=non_blocking,
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
    ) -> T:
        if len(key) == 1:
            return self._set_str(
                key[0],
                value,
                inplace=inplace,
                validated=validated,
                non_blocking=non_blocking,
            )
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
        #                 key, type(self).__name__, sorted(self.keys())
        #             )
        #         )
        #     inplace = has_key
        if not validated:
            value = self._validate_value(value, non_blocking=non_blocking)
            validated = True
        if self._is_vmapped:
            value = self.hook_in(value)
        values = value.unbind(self.stack_dim)
        for tensordict, item in _zip_strict(self.tensordicts, values):
            tensordict._set_tuple(
                key,
                item,
                inplace=inplace,
                validated=validated,
                non_blocking=non_blocking,
            )
        return self

    def _split_index(self, index):
        """Given a tuple index, split it in as many indices as the number of tensordicts.

        Returns:
            a dictionary with {index-of-td: index-within-td}
            the number of single dim indices until stack dim
            a boolean indicating if the index along the stack dim is an integer
        """
        if not isinstance(index, tuple):
            index = (index,)
        index = convert_ellipsis_to_idx(index, self.batch_size)
        index = _broadcast_tensors(index)
        out = []
        num_single = 0
        num_none = 0
        isinteger = False
        is_nd_tensor = False
        cursor = 0  # the dimension cursor
        selected_td_idx = torch.arange(len(self.tensordicts))
        has_bool = False
        num_squash = 0
        encountered_tensor = False
        for i, idx in enumerate(index):  # noqa: B007
            cursor_incr = 1
            # if idx is None:
            #     idx = True
            if idx is None or idx is True:
                out.append(None)
                num_none += cursor <= self.stack_dim
                continue
            if cursor == self.stack_dim:
                # we need to check which tds need to be indexed
                if isinstance(idx, ftdim.Dim):
                    raise ValueError(
                        "Cannot index a lazy stacked tensordict along the stack dimension with "
                        "a first-class dimension index. Consider consolidating the tensordict first "
                        "using `tensordict.contiguous()`."
                    )
                elif isinstance(idx, slice) or _is_number(idx):
                    selected_td_idx = range(len(self.tensordicts))[idx]
                    if not isinstance(selected_td_idx, range):
                        isinteger = True
                        selected_td_idx = [selected_td_idx]
                elif isinstance(idx, torch.Tensor):
                    if idx.dtype == torch.bool:
                        # we mark that we need to dispatch the indices across stack idx
                        has_bool = True
                        # split mask along dim
                        individual_masks = idx = idx.unbind(0)
                        selected_td_idx = range(len(self.tensordicts))
                        out.append(idx)
                        split_dim = self.stack_dim - num_single
                        mask_loc = i
                    else:
                        is_nd_tensor = True
                        if not encountered_tensor:
                            # num_single -= idx.ndim - 1
                            encountered_tensor = True
                        else:
                            num_single += 1
                        selected_td_idx = idx
                        # out.append(idx.unbind(0))
                else:
                    raise TypeError(f"Invalid index type: {type(idx)}.")
            else:
                if _is_number(idx) and cursor < self.stack_dim:
                    num_single += 1
                if _is_number(idx) or isinstance(
                    idx,
                    (
                        ftdim.Dim,
                        slice,
                    ),
                ):
                    out.append(idx)
                elif isinstance(idx, torch.Tensor):
                    if idx.dtype == torch.bool:
                        cursor_incr = idx.ndim
                        if cursor < self.stack_dim:
                            num_squash += cursor_incr - 1
                        if (
                            cursor < self.stack_dim
                            and cursor + cursor_incr > self.stack_dim
                        ):
                            # we mark that we need to dispatch the indices across stack idx
                            has_bool = True
                            # split mask along dim
                            # relative_stack_dim = self.stack_dim - cursor - cursor_incr
                            individual_masks = idx = idx.unbind(0)
                            selected_td_idx = range(self.shape[i])
                            split_dim = cursor - num_single
                            mask_loc = i
                    elif cursor < self.stack_dim:
                        # we know idx is not a single integer, so it must have
                        # a dimension. We play with num_single, reducing it
                        # by the number of dims of idx: if idx has 3 dims, our
                        # indexed tensor will have 2 more dimensions, going in
                        # the opposite direction of indexing with a single integer,
                        # smth[torch.tensor(1)].ndim = smth.ndim-1
                        # smth[torch.tensor([1])].ndim = smth.ndim
                        # smth[torch.tensor([[1]])].ndim = smth.ndim+1
                        if not encountered_tensor:
                            num_single -= idx.ndim - 1
                            encountered_tensor = True
                        else:
                            num_single += 1
                    out.append(idx)
                else:
                    raise TypeError(f"Invalid index type: {type(idx)}.")
            cursor += cursor_incr
        if has_bool:
            out = tuple(
                tuple(idx if not isinstance(idx, tuple) else idx[i] for idx in out)
                for i in selected_td_idx
            )
            return {
                "index_dict": {i: out[i] for i in selected_td_idx},
                "num_single": num_single,
                "isinteger": isinteger,
                "has_bool": has_bool,
                "individual_masks": individual_masks,
                "split_dim": split_dim,
                "mask_loc": mask_loc,
                "is_nd_tensor": is_nd_tensor,
                "num_none": num_none,
                "num_squash": num_squash,
            }
        elif is_nd_tensor:

            def isindexable(idx):
                if isinstance(idx, torch.Tensor):
                    if idx.dtype == torch.bool:
                        return False
                    return True
                if isinstance(idx, (tuple, list, range)):
                    return True
                return False

            def outer_list(tensor_index, tuple_index):
                """Converts a tensor and a tuple to a nested list where each leaf is a (int, index) tuple where the index only points to one element."""
                if isinstance(tensor_index, torch.Tensor):
                    list_index = tensor_index.tolist()
                else:
                    list_index = tensor_index
                list_result = []

                def index_tuple_index(i, convert=False):
                    for idx in tuple_index:
                        if isindexable(idx):
                            if convert:
                                yield int(idx[i])
                            else:
                                yield idx[i]
                        else:
                            yield idx

                for i, idx in enumerate(list_index):
                    if isinstance(idx, int):
                        list_result.append(
                            (idx, tuple(index_tuple_index(i, convert=True)))
                        )
                    elif isinstance(idx, list):
                        list_result.append(outer_list(idx, tuple(index_tuple_index(i))))
                    else:
                        raise NotImplementedError
                return list_result

            return {
                "index_dict": outer_list(selected_td_idx, out),
                "num_single": num_single,
                "isinteger": isinteger,
                "has_bool": has_bool,
                "is_nd_tensor": is_nd_tensor,
                "num_none": num_none,
                "num_squash": num_squash,
            }
        return {
            "index_dict": {i: tuple(out) for i in selected_td_idx},
            "num_single": num_single,
            "isinteger": isinteger,
            "has_bool": has_bool,
            "is_nd_tensor": is_nd_tensor,
            "num_none": num_none,
            "num_squash": num_squash,
        }

    def _set_at_str(self, key, value, index, *, validated, non_blocking: bool):
        if not validated:
            value = self._validate_value(
                value, check_shape=False, non_blocking=non_blocking
            )
            validated = True
        if self._is_vmapped:
            value = self.hook_in(value)
        split_index = self._split_index(index)
        converted_idx = split_index["index_dict"]
        num_single = split_index["num_single"]
        isinteger = split_index["isinteger"]
        has_bool = split_index["has_bool"]
        num_squash = split_index.get("num_squash", 0)
        num_none = split_index.get("num_none", 0)
        is_nd_tensor = split_index.get("is_nd_tensor", False)
        if isinteger:
            # this will break if the index along the stack dim is [0] or :1 or smth
            for i, _idx in converted_idx.items():
                self.tensordicts[i]._set_at_str(
                    key, value, _idx, validated=validated, non_blocking=non_blocking
                )
            return self
        if is_nd_tensor:
            unbind_dim = self.stack_dim - num_single + num_none - num_squash
            value_unbind = value.unbind(unbind_dim)

            def set_at_str(converted_idx):
                for i, item in enumerate(converted_idx):
                    if isinstance(item, list):
                        set_at_str(item)
                    else:
                        _value = value_unbind[i]
                        stack_idx, idx = item
                        self.tensordicts[stack_idx]._set_at_str(
                            key,
                            _value,
                            idx,
                            validated=validated,
                            non_blocking=non_blocking,
                        )

            set_at_str(converted_idx)
            return self
        elif not has_bool:
            unbind_dim = self.stack_dim - num_single + num_none - num_squash
            value_unbind = value.unbind(unbind_dim)
            for (i, _idx), _value in _zip_strict(
                converted_idx.items(),
                value_unbind,
            ):
                self.tensordicts[i]._set_at_str(
                    key, _value, _idx, validated=validated, non_blocking=non_blocking
                )
        else:
            # we must split, not unbind
            mask_unbind = split_index["individual_masks"]
            split_dim = split_index["split_dim"]
            splits = [_mask_unbind.sum().item() for _mask_unbind in mask_unbind]
            value_unbind = value.split(splits, split_dim)
            if mask_unbind[0].ndim == 0:
                # we can return a stack
                for (i, _idx), mask, _value in _zip_strict(
                    converted_idx.items(),
                    mask_unbind,
                    value_unbind,
                ):
                    if mask.any():
                        self.tensordicts[i]._set_at_str(
                            key,
                            _value,
                            _idx,
                            validated=validated,
                            non_blocking=non_blocking,
                        )
            else:
                for (i, _idx), _value in _zip_strict(
                    converted_idx.items(), value_unbind
                ):
                    self_idx = (slice(None),) * split_index["mask_loc"] + (i,)
                    self[self_idx]._set_at_str(
                        key,
                        _value,
                        _idx,
                        validated=validated,
                        non_blocking=non_blocking,
                    )

    def _set_at_tuple(self, key, value, idx, *, validated, non_blocking: bool):
        if len(key) == 1:
            return self._set_at_str(
                key[0], value, idx, validated=validated, non_blocking=non_blocking
            )
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
            value = self._validate_value(
                value, check_shape=False, non_blocking=non_blocking
            )
            validated = True
        if self._is_vmapped:
            value = self.hook_in(value)
        item = td._get_str(key, NO_DEFAULT)
        item[idx] = value
        td._set_str(key, item, inplace=True, validated=True, non_blocking=non_blocking)
        return self

    def _legacy_unsqueeze(self, dim: int) -> T:
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
        return type(self)(
            *(tensordict.unsqueeze(dim) for tensordict in self.tensordicts),
            stack_dim=stack_dim,
            stack_dim_name=self._td_dim_name,
        )

    def _legacy_squeeze(self, dim: int | None = None) -> T:
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
        return type(self)(
            *(tensordict.squeeze(dim) for tensordict in self.tensordicts),
            stack_dim=stack_dim,
            stack_dim_name=self._td_dim_name,
        )

    def _unbind(self, dim: int) -> tuple[TensorDictBase, ...]:
        if dim == self.stack_dim:
            return tuple(self.tensordicts)
        else:
            # return a stack of unbound tensordicts
            out = []
            new_dim = dim if dim < self.stack_dim else dim - 1
            new_stack_dim = (
                self.stack_dim if dim > self.stack_dim else self.stack_dim - 1
            )
            for td in self.tensordicts:
                out.append(td._unbind(new_dim))
            return tuple(
                self.lazy_stack(vals, new_stack_dim) for vals in _zip_strict(*out)
            )

    def _stack_onto_(
        self,
        list_item: list[CompatibleType],
        dim: int,
    ) -> T:
        if dim == self.stack_dim:
            for source, tensordict_dest in _zip_strict(list_item, self.tensordicts):
                tensordict_dest.update_(source)
        else:
            for i, td in enumerate(list_item):
                idx = (slice(None),) * dim + (i,)
                self.update_at_(td, idx)
        return self

    def _maybe_get_list(self, key):
        vals = []
        for td in self.tensordicts:
            if isinstance(td, LazyStackedTensorDict):
                val = td._maybe_get_list(key)
            else:
                val = td._get_str(key, None)
                if _is_tensor_collection(type(val)):
                    return self._get_str(key, NO_DEFAULT)
                elif val is None:
                    return None
            vals.append(val)
        return vals

    def get(
        self,
        key: NestedKey,
        *args,
        as_list: bool = False,
        as_padded_tensor: bool = False,
        as_nested_tensor: bool = False,
        padding_side: str = "right",
        layout: torch.layout = None,
        padding_value: float | int | bool = 0.0,
        **kwargs,
    ) -> CompatibleType:
        """Gets the value stored with the input key.

        Args:
            key (str, tuple of str): key to be queried. If tuple of str it is
                equivalent to chained calls of getattr.
            default: default value if the key is not found in the tensordict. Defaults to ``None``.

                .. warning::
                    Previously, if a key was not present in the tensordict and no default
                    was passed, a `KeyError` was raised. From v0.7, this behaviour has been changed
                    and a `None` value is returned instead (in accordance with the what dict.get behavior).
                    To adopt the old behavior, set the environment variable `export TD_GET_DEFAULTS_TO_NONE='0'` or call
                    :func`~tensordict.set_get_defaults_to_none(False)`.

        Keyword Args:
            as_list (bool, optional): if ``True``, ragged tensors will be returned as list.
                Exclusive with `as_padded_tensor` and `as_nested_tensor`.
                Defaults to ``False``.
            as_padded_tensor (bool, optional):  if ``True``, ragged tensors will be returned as padded tensors.
                The padding value can be controlled via the `padding_value` keyword argument, and the padding
                side via the `padding_side` argument.
                Exclusive with `as_list` and `as_nested_tensor`.
                Defaults to ``False``.
            as_nested_tensor (bool, optional): if ``True``, ragged tensors will be returned as list.
                Exclusive with `as_list` and `as_padded_tensor`.
                The layout can be controlled via the `torch.layout` argument.
                Defaults to ``False``.
            layout (torch.layout, optional): the layout when `as_nested_tensor=True`.
            padding_side (str): The side of padding. Must be `"left"` or `"right"`. Defaults to `"right"`.
            padding_value (scalar or bool, optional): The padding value. Defaults to 0.0.

        Examples:
            >>> from tensordict import TensorDict, lazy_stack
            >>> import torch
            >>> td = lazy_stack([
            ...     TensorDict({"x": torch.ones(1,)}),
            ...     TensorDict({"x": torch.ones(2,) * 2}),
            ... ])
            >>> td.get("x", as_nested_tensor=True)
            NestedTensor(size=(2, j1), offsets=tensor([0, 1, 3]), contiguous=True)
            >>> td.get("x", as_padded_tensor=True)
            tensor([[1., 0.],
                    [2., 2.]])

        """
        return super().get(
            key,
            *args,
            as_list=as_list,
            as_padded_tensor=as_padded_tensor,
            as_nested_tensor=as_nested_tensor,
            padding_side=padding_side,
            layout=layout,
            padding_value=padding_value,
            **kwargs,
        )

    @cache  # noqa: B019
    def _get_str(
        self,
        key: NestedKey,
        default: Any = NO_DEFAULT,
        *,
        as_list: bool = False,
        as_padded_tensor: bool = False,
        as_nested_tensor: bool = False,
        padding_side: str = "right",
        layout: torch.layout = None,
        padding_value: float | int | bool = 0.0,
    ) -> CompatibleType:
        # we can handle the case where the key is a tuple of length 1
        tensors = []
        for td in self.tensordicts:
            tensors.append(td._get_str(key, default=default))
            if (
                tensors[-1] is default
                and not isinstance(default, torch.Tensor)
                and not is_tensor_collection(default)
            ):
                # then we consider this default as non-stackable and return prematurly
                return default
        try:
            out = self.lazy_stack(
                tensors,
                self.stack_dim,
                stack_dim_name=self._td_dim_name,
                as_list=as_list,
                as_padded_tensor=as_padded_tensor,
                as_nested_tensor=as_nested_tensor,
                padding_side=padding_side,
                layout=layout,
                padding_value=padding_value,
            )
            if _is_tensor_collection(type(out)):
                if isinstance(out, LazyStackedTensorDict):
                    # then it's a LazyStackedTD
                    out.hook_out = self.hook_out
                    out.hook_in = self.hook_in
                    out._is_vmapped = self._is_vmapped
                    incr = 0 if not self._is_vmapped else 1
                    out._batch_size = (
                        self._batch_size
                        + out.batch_size[(len(self._batch_size) + incr) :]
                    )
                elif is_tensorclass(out):
                    # then it's a tensorclass
                    out._tensordict.hook_out = self.hook_out
                    out._tensordict.hook_in = self.hook_in
                    out._tensordict._is_vmapped = self._is_vmapped
                    incr = 0 if not self._is_vmapped else 1
                    out._tensordict._batch_size = (
                        self._batch_size
                        + out._tensordict.batch_size[(len(self._batch_size) + incr) :]
                    )
                else:
                    raise RuntimeError
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

    def _get_tuple(self, key, default, **kwargs):
        first = self._get_str(key[0], None, **kwargs)
        if first is None:
            return self._default_get(key[0], default)
        if len(key) == 1:
            return first
        try:
            return first._get_tuple(key[1:], default=default, **kwargs)
        except AttributeError as err:
            if "has no attribute" in str(err):
                raise ValueError(
                    f"Expected a TensorDictBase instance but got {type(first)} instead"
                    f" for key '{key[1:]}' in tensordict:\n{self}."
                )

    @classmethod
    def lazy_stack(
        cls,
        items: Sequence[TensorDictBase],
        dim: int = 0,
        *,
        device: DeviceType | None = None,
        out: T | None = None,
        stack_dim_name: str | None = None,
        strict_shape: bool = False,
        as_list: bool = False,
        as_padded_tensor: bool = False,
        as_nested_tensor: bool = False,
        padding_side: str = "right",
        layout: torch.layout | None = None,
        padding_value: float | int | bool = 0.0,
    ) -> T:  # noqa: D417
        """Stacks tensordicts in a LazyStackedTensorDict.

        Args:
            items (Sequence of TensorDictBase instances): A sequence of TensorDictBase
                instances to stack.
            dim (int, optional): the dim along which to perform the lazy stack.
                Defaults to 0.

        Keyword Args:
            device (torch.device, optional): a device to set in the `LazyStackedTensorDict`
                in case it cannot be inferred from the tensordict list (e.g., the list is empty).
            out (TensorDictBase, optional): a `LazyStackedTensorDict` where to write the data.
            stack_dim_name (str, optional): a name for the stacked dimension.
            strict_shape (bool, optional): if ``True``, every tensordict's shapes must match.
                Defaults to ``False``.
            as_list (bool, optional): if ``True``, ragged tensors will be returned as list.
                Exclusive with `as_padded_tensor` and `as_nested_tensor`.
                Defaults to ``False``.
            as_padded_tensor (bool, optional):  if ``True``, ragged tensors will be returned as padded tensors.
                The padding value can be controlled via the `padding_value` keyword argument, and the padding
                side via the `padding_side` argument.
                Exclusive with `as_list` and `as_nested_tensor`.
                Defaults to ``False``.
            as_nested_tensor (bool, optional): if ``True``, ragged tensors will be returned as list.
                Exclusive with `as_list` and `as_padded_tensor`.
                The layout can be controlled via the `torch.layout` argument.
                Defaults to ``False``.
            layout (torch.layout, optional): the layout when `as_nested_tensor=True`.
            padding_side (str): The side of padding. Must be `"left"` or `"right"`. Defaults to `"right"`.
            padding_value (scalar or bool, optional): The padding value. Defaults to 0.0.

        """
        if not items:
            raise RuntimeError("items cannot be empty")

        if all(isinstance(item, torch.Tensor) for item in items):
            # This must be implemented here and not in _get_str because we want to leverage this check
            special_return = sum((as_list, as_padded_tensor, as_nested_tensor))
            if special_return > 1:
                raise TypeError(
                    "as_list, as_padded_tensor and as_nested_tensor are exclusive."
                )
            elif special_return:
                if as_padded_tensor:
                    return pad_sequence(
                        items,
                        padding_value=padding_value,
                        padding_side=padding_side,
                        batch_first=True,
                    )
                if as_nested_tensor:
                    if layout is None:
                        layout = torch.jagged
                    return torch.nested.as_nested_tensor(items, layout=layout)
                if as_list:
                    return items
            try:
                return torch.stack(items, dim=dim, out=out)
            except RuntimeError as err:
                raise RuntimeError(
                    "Failed to stack tensors within a tensordict. You can use nested tensors, "
                    "padded tensors or return lists via specialized keyword arguments. "
                    "Check the TensorDict.lazy_stack documentation!"
                ) from err
        if all(is_non_tensor(tensordict) for tensordict in items):
            # Non-tensor data (Data or Stack) are stacked using NonTensorStack
            # If the content is identical (not equal but same id) this does not
            # require additional memory.
            from .tensorclass import NonTensorStack

            return NonTensorStack(*items, stack_dim=dim)
        if all(
            is_tensorclass(item) and type(item) == type(items[0])  # noqa: E721
            for item in items
        ):
            lazy_stack = cls.lazy_stack(
                [item._tensordict for item in items],
                dim=dim,
                out=out,
                stack_dim_name=stack_dim_name,
            )
            # we take the first non_tensordict by convention
            return type(items[0])._from_tensordict(
                tensordict=lazy_stack, non_tensordict=items[0]._non_tensordict
            )

        batch_size = items[0].batch_size
        if dim < 0:
            dim = len(batch_size) + dim + 1

        if strict_shape:
            for td in items[1:]:
                if td.batch_size != items[0].batch_size:
                    raise RuntimeError(
                        "stacking tensordicts requires them to have congruent batch sizes, "
                        f"got td1.batch_size={td.batch_size} and td2.batch_size="
                        f"{items[0].batch_size}"
                    )

        if out is None:
            # We need to handle tensordicts with exclusive keys and tensordicts with
            # mismatching shapes.
            # The first case is handled within _check_keys which fails if keys
            # don't match exactly.
            # The second requires a check over the tensor shapes.
            return LazyStackedTensorDict(
                *items,
                stack_dim=dim,
                stack_dim_name=stack_dim_name,
                strict_shape=strict_shape,
                device=device,
            )
        else:
            batch_size = list(batch_size)
            batch_size.insert(dim, len(items))
            batch_size = torch.Size(batch_size)

            if out.batch_size != batch_size:
                raise RuntimeError(
                    "out.batch_size and stacked batch size must match, "
                    f"got out.batch_size={out.batch_size} and batch_size"
                    f"={batch_size}"
                )

            try:
                out._stack_onto_(items, dim)
            except KeyError as err:
                raise err
        return out

    @classmethod
    def maybe_dense_stack(
        cls,
        items: Sequence[TensorDictBase],
        dim: int = 0,
        out: T | None = None,
        strict: bool = False,
    ) -> T:
        """Stacks tensors or tensordicts densly if possible, or onto a LazyStackedTensorDict otherwise.

        Examples:
            >>> td0 = TensorDict({"a": 0}, [])
            >>> td1 = TensorDict({"b": 0}, [])
            >>> LazyStackedTensorDict.maybe_dense_stack([td0, td0])  # returns a TensorDict with shape [2]
            >>> LazyStackedTensorDict.maybe_dense_stack([td0, td1])  # returns a LazyStackedTensorDict with shape [2]
            >>> LazyStackedTensorDict.maybe_dense_stack(list(torch.randn(2)))  # returns a torch.Tensor with shape [2]
        """
        from ._torch_func import _stack

        return _stack(items, dim=dim, out=out, strict=strict, maybe_dense_stack=True)

    @cache  # noqa: B019
    def _add_batch_dim(self, *, in_dim, vmap_level):
        if self.is_memmap():
            td = LazyStackedTensorDict.lazy_stack(
                [td.cpu().as_tensor() for td in self.tensordicts], 0
            )
        else:
            td = self
        if in_dim < 0:
            in_dim = self.ndim + in_dim
        if in_dim == self.stack_dim:
            result = self._cached_add_batch_dims(
                td, in_dim=in_dim, vmap_level=vmap_level
            )
        else:
            if in_dim < td.stack_dim:
                # then we'll stack along a dim before
                stack_dim = td.stack_dim - 1
            else:
                in_dim = in_dim - 1
                stack_dim = td.stack_dim

            def addbatchdim(_arg):
                return _add_batch_dim(_arg, in_dim, vmap_level)

            tds = [
                td._fast_apply(
                    addbatchdim,
                    batch_size=[b for i, b in enumerate(td.batch_size) if i != in_dim],
                    names=(
                        [name for i, name in enumerate(td.names) if i != in_dim]
                        if self._has_names()
                        else None
                    ),
                )
                for td in td.tensordicts
            ]
            result = LazyStackedTensorDict(*tds, stack_dim=stack_dim)
        if self.is_locked:
            result.lock_()
        return result

    @classmethod
    def _cached_add_batch_dims(cls, td, in_dim, vmap_level):
        # we return a stack with hook_out, and hack the batch_size and names
        # Per se it is still a LazyStack but the stacking dim is "hidden" from
        # the outside
        out = td.copy()

        def hook_out(tensor, in_dim=in_dim, vmap_level=vmap_level):
            if _is_tensor_collection(type(tensor)):
                return tensor._add_batch_dim(in_dim=in_dim, vmap_level=vmap_level)
            return _add_batch_dim(tensor, in_dim, vmap_level)

        n = len(td.tensordicts)

        def hook_in(
            tensor,
            out_dim=in_dim,
            batch_size=n,
            vmap_level=vmap_level,
        ):
            if _is_tensor_collection(type(tensor)):
                return tensor._remove_batch_dim(vmap_level, batch_size, out_dim)
            return _remove_batch_dim(tensor, vmap_level, batch_size, out_dim)

        out.hook_out = hook_out
        out.hook_in = hook_in
        out._is_vmapped = True
        out._batch_size = torch.Size(
            [dim for i, dim in enumerate(out._batch_size) if i != out.stack_dim]
        )
        return out

    @cache  # noqa: B019
    def _remove_batch_dim(self, vmap_level, batch_size, out_dim):
        if self.hook_out is not None:
            # this is the hacked version. We just need to remove the hook_out and
            # reset a proper batch size
            result = LazyStackedTensorDict(
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
            result = LazyStackedTensorDict(
                *[
                    td._remove_batch_dim(
                        vmap_level=vmap_level, batch_size=batch_size, out_dim=out_dim
                    )
                    for td in self.tensordicts
                ],
                stack_dim=stack_dim,
            )
        if self.is_locked:
            result.lock_()
        return result

    @cache  # noqa: B019
    def _maybe_remove_batch_dim(self, funcname, vmap_level, batch_size, out_dim):
        if self.hook_out is not None:
            # this is the hacked version. We just need to remove the hook_out and
            # reset a proper batch size
            result = LazyStackedTensorDict(
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
            result = LazyStackedTensorDict(
                *[
                    td._maybe_remove_batch_dim(
                        funcname,
                        vmap_level=vmap_level,
                        batch_size=batch_size,
                        out_dim=out_dim,
                    )
                    for td in self.tensordicts
                ],
                stack_dim=stack_dim,
            )
        if self.is_locked:
            result.lock_()
        return result

    def get_nestedtensor(
        self,
        key: NestedKey,
        default: Any = NO_DEFAULT,
        *,
        layout: torch.layout | None = None,
    ) -> CompatibleType:
        """Returns a nested tensor when stacking cannot be achieved.

        Args:
            key (NestedKey): the entry to nest.
            default (Any, optiona): the default value to return in case the key
                isn't in all sub-tensordicts.

                .. note::
                    In case the default is a tensor, this method will attempt
                    the construction of a nestedtensor with it. Otherwise, the default
                    value will be returned.

        Keyword Args:
            layout (torch.layout, optional): the layout for the nested tensor.

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
            return tensordict.get_nestedtensor(key[1:], default=default, layout=layout)
        tensors = [td.get(subkey, default=default) for td in self.tensordicts]
        if not isinstance(default, torch.Tensor) and any(
            tensor is default for tensor in tensors
        ):
            # we don't stack but return the default
            return default
        return torch.nested.nested_tensor(tensors, layout=layout)

    def is_contiguous(self) -> bool:
        return False

    def contiguous(self) -> T:
        source = {key: value.contiguous() for key, value in self.items()}
        batch_size = self.batch_size
        device = self.device
        out = TensorDict._new_unsafe(
            source=source,
            batch_size=batch_size,
            device=device,
            names=self.names if self._has_names() else None,
            lock=self.is_locked,
        )
        return out

    def densify(self, *, layout: torch.layout = torch.strided):
        """Attempts to represent the lazy stack with contiguous tensors (plain tensors or nested).

        Keyword Args:
            layout (torch.layout): the layout of the nested tensors, if any. Defaults to
                :class:`~torch.strided`.

        """
        result = TensorDict._new_unsafe(
            batch_size=self.batch_size, device=self.device, names=self.names
        )
        for key in self._exclusive_keys():
            list_of_entries = [
                td._get_str(key, default=None) for td in self.tensordicts
            ]
            is_tensor = all(
                isinstance(item, torch.Tensor) or item is None
                for item in list_of_entries
            )
            if is_tensor:
                shapes = {
                    tensor.shape if tensor is not None else None
                    for tensor in list_of_entries
                }
                if None in shapes:
                    # There must be at least one non-None value
                    a_shape = None
                    while a_shape is None:
                        a_shape = shapes.pop()
                    if not a_shape:
                        raise RuntimeError(
                            f"Cannot densify a tensordict with values with empty shape and exclusive keys: got shape {a_shape}."
                        )
                    none_shape = a_shape[:-1] + (0,)
                    for tensor in list_of_entries:
                        if tensor is not None:
                            a_tensor = tensor.new_zeros(none_shape)
                            break
                    list_of_entries = [
                        tensor if tensor is not None else a_tensor
                        for tensor in list_of_entries
                    ]
                    shapes.update({a_shape, none_shape})
                if len(shapes) == 1:
                    tensor = torch.stack(list_of_entries, self.stack_dim)
                else:
                    if self.stack_dim == 0:
                        tensor = torch.nested.nested_tensor(
                            list_of_entries, layout=layout
                        )
                    else:
                        raise NotImplementedError(
                            f"stack_dim is {self.stack_dim} but not 0. Densify canot be done."
                        )
            else:
                tensor = self._get_str(key, None)
                if tensor is not None:
                    tensor = tensor.densify(layout=layout)
                else:
                    from tensordict import NonTensorData

                    tensor = NonTensorData(None)
            result._set_str(key, tensor, validated=True, inplace=False)
        return result

    def empty(
        self, recurse=False, *, batch_size=None, device=NO_DEFAULT, names=None
    ) -> T:
        name = None
        if batch_size is not None and (
            self.stack_dim
            and batch_size[: self.stack_dim] != self.batch_size[: self.stack_dim]
        ):
            return TensorDict.empty(
                self,
                recurse=recurse,
                batch_size=batch_size,
                device=device if device is not NO_DEFAULT else self.device,
                names=names if names is not None else None,
            )
        if names is not None:
            if len(names) > self.stack_dim:
                name = names[self.stack_dim]
            names = [name for i, name in enumerate(names) if i != self.stack_dim]
        if batch_size is not None:
            batch_size = torch.Size(
                [b for i, b in enumerate(batch_size) if i != self.stack_dim]
            )
        return type(self)(
            *[
                td.empty(
                    recurse=recurse, batch_size=batch_size, device=device, names=names
                )
                for td in self.tensordicts
            ],
            stack_dim=self.stack_dim,
            stack_dim_name=name,
        )

    def _clone(self, recurse: bool = True) -> T:
        if recurse:
            # This could be optimized using copy but we must be careful with
            # metadata (_is_shared etc)
            result = type(self)(
                *[td._clone() for td in self.tensordicts],
                stack_dim=self.stack_dim,
                stack_dim_name=self._td_dim_name,
            )
        else:
            result = type(self)(
                *[td._clone(recurse=False) for td in self.tensordicts],
                stack_dim=self.stack_dim,
                stack_dim_name=self._td_dim_name,
            )
        return result

    def to(self, *args, **kwargs) -> T:
        if kwargs.get("batch_size") is not None:
            raise TypeError("Cannot pass batch-size to a LazyStackedTensorDict.")
        return super().to(*args, **kwargs)

    def _check_new_batch_size(self, new_size: torch.Size) -> None:
        if len(new_size) <= self.stack_dim:
            raise RuntimeError(
                "Changing the batch_size of a LazyStackedTensorDicts can only "
                "be done with sizes that are at least as long as the "
                "stacking dimension."
            )
        super()._check_new_batch_size(new_size)

    def _change_batch_size(self, new_size: torch.Size) -> None:
        self._batch_size = new_size

    def keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
        *,
        sort: bool = False,
    ) -> _LazyStackedTensorDictKeysView:
        keys = _LazyStackedTensorDictKeysView(
            self,
            include_nested=include_nested,
            leaves_only=leaves_only,
            is_leaf=is_leaf,
            sort=sort,
        )
        return keys

    def values(
        self,
        include_nested=False,
        leaves_only=False,
        is_leaf=None,
        *,
        sort: bool = False,
    ):
        if is_leaf not in (
            _NESTED_TENSORS_AS_LISTS,
            _NESTED_TENSORS_AS_LISTS_NONTENSOR,
        ):
            yield from super().values(
                include_nested=include_nested,
                leaves_only=leaves_only,
                is_leaf=is_leaf,
                sort=sort,
            )
        else:
            for td in self.tensordicts:
                yield from td.values(
                    include_nested=include_nested,
                    leaves_only=leaves_only,
                    is_leaf=is_leaf,
                    sort=sort,
                )

    def items(
        self,
        include_nested=False,
        leaves_only=False,
        is_leaf=None,
        *,
        sort: bool = False,
    ):
        if is_leaf not in (
            _NESTED_TENSORS_AS_LISTS,
            _NESTED_TENSORS_AS_LISTS_NONTENSOR,
        ):
            yield from super().items(
                include_nested=include_nested,
                leaves_only=leaves_only,
                is_leaf=is_leaf,
                sort=sort,
            )
        else:
            for i, td in enumerate(self.tensordicts):
                for key, val in td.items(
                    include_nested=include_nested,
                    leaves_only=leaves_only,
                    is_leaf=is_leaf,
                    sort=sort,
                ):
                    if isinstance(key, str):
                        key = (str(i), key)
                    else:
                        key = (str(i), *key)
                    yield key, val

    valid_keys = keys

    def non_tensor_items(self, include_nested: bool = False):
        """Returns all non-tensor leaves, maybe recursively."""
        items = self.tensordicts[0].non_tensor_items(include_nested=include_nested)
        return tuple(
            (
                key,
                torch.stack(
                    [val0, *[td.get(key) for td in self.tensordicts[1:]]],
                    self.stack_dim,
                ),
            )
            for (key, val0) in items
        )

    def _iterate_over_keys(self) -> None:
        # this is about 20x faster than the version above
        yield from self._key_list()

    @cache  # noqa: B019
    def _key_list(self):
        if not self.tensordicts:
            return []
        keys = set(self.tensordicts[0].keys())
        for td in self.tensordicts[1:]:
            keys = keys.intersection(td.keys())
        return sorted(keys, key=str)

    @lock_blocked
    def popitem(self) -> Tuple[NestedKey, CompatibleType]:
        key, val = self.tensordicts[0].popitem()
        vals = [val]
        for i, td in enumerate(self.tensordicts[1:]):
            val = td.pop(key, None)
            if val is not None:
                vals.append(val)
            else:
                for j in range(i + 1):
                    self.tensordicts[j].set(key, vals[j])
                raise RuntimeError(f"Could not find key {key} in all tensordicts.")
        return key, torch.stack(vals, dim=self.stack_dim)

    def entry_class(self, key: NestedKey) -> type:
        data_type = type(self.tensordicts[0].get(key))
        if _is_tensor_collection(data_type):
            return LazyStackedTensorDict
        return data_type

    def apply_(self, fn: Callable, *others, **kwargs):
        others = (other.unbind(self.stack_dim) for other in others)
        for td, *_others in _zip_strict(self.tensordicts, *others):
            td._fast_apply(fn, *_others, inplace=True, propagate_lock=True, **kwargs)
        return self

    def _multithread_apply_nest(self, *args, **kwargs):
        if kwargs.get("batch_size") is not None:
            raise RuntimeError(
                f"batch_size cannot be specified for {type(self).__name__}._multithread_apply_nest."
            )
        return super()._multithread_apply_nest(*args, **kwargs)

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
        others = (other.unbind(self.stack_dim) for other in others)
        if (
            call_on_nested
            and named
            and is_leaf
            in (_NESTED_TENSORS_AS_LISTS, _NESTED_TENSORS_AS_LISTS_NONTENSOR)
        ):
            # When calling on nested with name and the name includes the TD index, we
            # want to call the function on each td.
            # If we were not keeping track of the TD's index, names would be the same for all
            # tds and there's a risk that values would collide.
            # nested_keys is irrelevant when named + call_on_nested are both true.
            for i, (td, *oth) in enumerate(_zip_strict(self.tensordicts, *others)):
                key = prefix + (str(i),)
                if len(key) == 1:
                    key = key[0]
                futures.append(executor.submit(fn, key, td, *oth))
                local_futures.append(futures[-1])
        else:
            for i, (td, *oth) in enumerate(_zip_strict(self.tensordicts, *others)):
                local_futures.append([])
                td._multithread_apply_flat(
                    fn,
                    *oth,
                    call_on_nested=call_on_nested,
                    default=default,
                    named=named,
                    nested_keys=nested_keys,
                    prefix=(
                        prefix + (str(i),)
                        if is_leaf
                        in (
                            _NESTED_TENSORS_AS_LISTS,
                            _NESTED_TENSORS_AS_LISTS_NONTENSOR,
                        )
                        else prefix
                    ),
                    is_leaf=is_leaf,
                    executor=executor,
                    futures=futures,
                    local_futures=local_futures[-1],
                )

    def _multithread_rebuild(
        self,
        *,
        # We know batch_size is None, this has been checked earlier
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
    ) -> None:
        if inplace and any(
            arg for arg in (batch_size, device, names, constructor_kwargs)
        ):
            raise ValueError(
                "Cannot pass other arguments to LazyStackedTensorDict.apply when inplace=True."
            )
        if out is not None:
            if not isinstance(out, LazyStackedTensorDict):
                raise ValueError(
                    "out must be a LazyStackedTensorDict instance in lazy_stack.apply(..., out=out)."
                )
            out = out.tensordicts
        results = []
        for i, (td, local_future) in enumerate(
            _zip_strict(self.tensordicts, local_futures)
        ):
            local_out = out[i] if out is not None else None
            # Each local_future points to a list of futures for a single tensordict
            local_out = td._multithread_rebuild(
                batch_size=batch_size,
                device=device,
                names=names,
                inplace=inplace,
                checked=checked,
                out=local_out,
                filter_empty=filter_empty,
                executor=executor,
                futures=futures,
                local_futures=local_future,
                subs_results=subs_results,
                multithread_set=multithread_set,
            )
            results.append(local_out)
        if filter_empty and all(r is None for r in results):
            return
        if not inplace:
            out = type(self)(
                *results,
                stack_dim=self.stack_dim,
                stack_dim_name=self._td_dim_name,
            )
        else:
            out = self
        if names is not NO_DEFAULT:
            out.names = names
        return out

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
        is_leaf: Callable | None = None,
        out: TensorDictBase | None = None,
        **constructor_kwargs,
    ) -> T | None:
        if inplace and any(
            arg for arg in (batch_size, device, names, constructor_kwargs)
        ):
            raise ValueError(
                "Cannot pass other arguments to LazyStackedTensorDict.apply when inplace=True. Got args "
                f"batch_size={batch_size}, device={device}, names={names}, constructor_kwargs={constructor_kwargs}"
            )
        if out is not None:
            if not isinstance(out, LazyStackedTensorDict):
                raise ValueError(
                    "out must be a LazyStackedTensorDict instance in lazy_stack.apply(..., out=out)."
                )
            out = out.tensordicts
        elif batch_size is not None:
            # any op that modifies the batch-size will result in a regular TensorDict
            batch_size = torch.Size(batch_size)
            out = TensorDict._new_unsafe(
                {},
                batch_size=batch_size,
                device=device if device is not NO_DEFAULT else self.device,
                names=names if names else self._maybe_names(),
            )
            return TensorDict._apply_nest(
                self,
                fn,
                *others,
                batch_size=batch_size,
                device=device,
                names=names,
                checked=checked,
                call_on_nested=call_on_nested,
                default=default,
                named=named,
                nested_keys=nested_keys,
                prefix=prefix,
                inplace=inplace,
                filter_empty=filter_empty,
                is_leaf=is_leaf,
                out=out,
                **constructor_kwargs,
            )

        others = (other.unbind(self.stack_dim) for other in others)
        if (
            call_on_nested
            and named
            and is_leaf
            in (_NESTED_TENSORS_AS_LISTS, _NESTED_TENSORS_AS_LISTS_NONTENSOR)
        ):
            # When calling on nested with name and the name includes the TD index, we
            # want to call the function on each td.
            # If we were not keeping track of the TD's index, names would be the same for all
            # tds and there's a risk that values would collide.
            # nested_keys is irrelevant when named + call_on_nested are both true.
            results = []
            for i, (td, *oth) in enumerate(_zip_strict(self.tensordicts, *others)):
                key = prefix + (str(i),)
                if len(key) == 1:
                    key = key[0]
                results.append(fn(key, td, *oth))
        else:
            results = [
                td._apply_nest(
                    fn,
                    *oth,
                    checked=checked,
                    device=device,
                    call_on_nested=call_on_nested,
                    default=default,
                    named=named,
                    nested_keys=nested_keys,
                    prefix=(
                        prefix + (str(i),)
                        if is_leaf
                        in (
                            _NESTED_TENSORS_AS_LISTS,
                            _NESTED_TENSORS_AS_LISTS_NONTENSOR,
                        )
                        else prefix
                    ),
                    inplace=inplace,
                    filter_empty=filter_empty,
                    is_leaf=is_leaf,
                    out=out[i] if out is not None else None,
                )
                for i, (td, *oth) in enumerate(_zip_strict(self.tensordicts, *others))
            ]
        if all(r is None for r in results) and filter_empty in (None, True):
            return
        if not inplace:
            if not results or any(r is not None for r in results):
                try:
                    out = type(self)(
                        *results,
                        stack_dim=self.stack_dim,
                        stack_dim_name=self._td_dim_name,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to reconstruct the lazy stack of tensordicts with class: {type(self)}. "
                        f"One common issue is that the outputs of apply are a mix of None and non-None "
                        f"values. Check that the outputs of apply() are all None or all non-None. "
                        f"Otherwise, please report this bug on tensordict github."
                    ) from e
            else:
                out = None
        else:
            out = self
        if names is not NO_DEFAULT:
            out.names = names
        return out

    def _select(
        self,
        *keys: NestedKey,
        inplace: bool = False,
        strict: bool = False,
        set_shared: bool = True,
    ) -> LazyStackedTensorDict:
        # the following implementation keeps the hidden keys in the tensordicts
        tensordicts = [
            td._select(*keys, inplace=inplace, strict=strict, set_shared=set_shared)
            for td in self.tensordicts
        ]
        if inplace:
            return self
        result = self._new_lazy_unsafe(
            *tensordicts, stack_dim=self.stack_dim, stack_dim_name=self._td_dim_name
        )
        return result

    def _exclude(
        self, *keys: NestedKey, inplace: bool = False, set_shared: bool = True
    ) -> LazyStackedTensorDict:
        tensordicts = [
            tensordict._exclude(*keys, inplace=inplace, set_shared=set_shared)
            for tensordict in self.tensordicts
        ]
        if inplace:
            self.tensordicts = tensordicts
            return self
        result = type(self)(
            *tensordicts, stack_dim=self.stack_dim, stack_dim_name=self._td_dim_name
        )
        return result

    def __setitem__(self, index: IndexType, value: T) -> T:
        if isinstance(index, (tuple, str)):
            # try:
            index_unravel = _unravel_key_to_tuple(index)
            if index_unravel:
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

            if any(
                isinstance(sub_index, (list, range, np.ndarray)) for sub_index in index
            ):
                index = tuple(
                    (
                        torch.as_tensor(sub_index, device=self.device)
                        if isinstance(sub_index, (list, range, np.ndarray))
                        else sub_index
                    )
                    for sub_index in index
                )

        if index is Ellipsis or (isinstance(index, tuple) and Ellipsis in index):
            index = convert_ellipsis_to_idx(index, self.batch_size)
        elif isinstance(index, (list, range)):
            index = torch.as_tensor(index, device=self.device)

        if is_tensor_collection(value) or isinstance(value, dict):
            indexed_bs = _getitem_batch_size(self.batch_size, index)
            if isinstance(value, dict):
                value = TensorDict._new_unsafe(
                    value, batch_size=indexed_bs, device=self.device
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
            split_index = self._split_index(index)
            converted_idx = split_index["index_dict"]
            num_single = split_index["num_single"]
            isinteger = split_index["isinteger"]
            has_bool = split_index["has_bool"]
            num_squash = split_index.get("num_squash", 0)
            num_none = split_index.get("num_none", 0)
            is_nd_tensor = split_index.get("is_nd_tensor", False)
            if isinteger:
                # this will break if the index along the stack dim is [0] or :1 or smth
                for i, _idx in converted_idx.items():
                    if _idx == ():
                        self.tensordicts[i].update(value, inplace=True)
                    else:
                        self.tensordicts[i][_idx] = value
                return self
            if is_nd_tensor:
                unbind_dim = self.stack_dim - num_single + num_none - num_squash

                # converted_idx is a nested list with (int, index) items
                def assign(converted_idx, value=value):
                    value = value.unbind(unbind_dim)
                    for i, item in enumerate(converted_idx):
                        if isinstance(item, list):
                            assign(item)
                        else:
                            stack_item, idx = item
                            if idx == ():
                                self.tensordicts[stack_item] = value[i]
                            else:
                                self.tensordicts[stack_item][idx] = value[i]

                assign(converted_idx)
                return self
            if not has_bool:
                unbind_dim = self.stack_dim - num_single + num_none - num_squash
                value_unbind = value.unbind(unbind_dim)
                for (i, _idx), _value in _zip_strict(
                    converted_idx.items(),
                    value_unbind,
                ):
                    if _idx == ():
                        self.tensordicts[i].update(_value, inplace=True)
                    else:
                        self.tensordicts[i][_idx] = _value
            else:
                # we must split, not unbind
                mask_unbind = split_index["individual_masks"]
                split_dim = split_index["split_dim"]
                splits = [_mask_unbind.sum().item() for _mask_unbind in mask_unbind]
                value_unbind = value.split(splits, split_dim)
                if mask_unbind[0].ndim == 0:
                    # we can return a stack
                    for (i, _idx), mask, _value in _zip_strict(
                        converted_idx.items(),
                        mask_unbind,
                        value_unbind,
                    ):
                        if mask.any():
                            self.tensordicts[i][_idx] = _value
                else:
                    for (i, _idx), _value in _zip_strict(
                        converted_idx.items(), value_unbind
                    ):
                        self_idx = (slice(None),) * split_index["mask_loc"] + (i,)
                        self[self_idx][_idx] = _value
        else:
            for key in self.keys():
                self.set_at_(key, value, index)

    def __contains__(self, item: IndexType) -> bool:
        if isinstance(item, TensorDictBase):
            return any(item is td for td in self.tensordicts)
        return super().__contains__(item)

    def __getitem__(self, index: IndexType) -> Any:
        if isinstance(index, (tuple, str)):
            index_key = _unravel_key_to_tuple(index)
            if index_key:
                leaf = self._get_tuple(index_key, NO_DEFAULT)
                if is_non_tensor(leaf):
                    # Only lazy stacks of non tensors are actually tensordict instances
                    if isinstance(leaf, TensorDictBase):
                        return leaf.tolist()
                    return leaf.data
                return leaf
        split_index = self._split_index(index)
        converted_idx = split_index["index_dict"]
        isinteger = split_index["isinteger"]
        has_bool = split_index["has_bool"]
        is_nd_tensor = split_index["is_nd_tensor"]
        num_single = split_index.get("num_single", 0)
        num_none = split_index.get("num_none", 0)
        num_squash = split_index.get("num_squash", 0)
        if has_bool:
            mask_unbind = split_index["individual_masks"]
            cat_dim = split_index["mask_loc"] - num_single
            result = []
            if mask_unbind[0].ndim == 0:
                # we can return a stack
                for (i, _idx), mask in _zip_strict(converted_idx.items(), mask_unbind):
                    if mask.any():
                        if mask.all() and self.tensordicts[i].ndim == 0:
                            result.append(self.tensordicts[i])
                        else:
                            result.append(self.tensordicts[i][_idx])
                            result[-1] = result[-1].squeeze(cat_dim)
                if not result:
                    batch_size = _getitem_batch_size(self.batch_size, index)
                else:
                    batch_size = None
                return self._new_lazy_unsafe(
                    *result,
                    stack_dim=cat_dim,
                    device=self.device,
                    names=self.names,
                    batch_size=batch_size,
                )
            else:
                for i, _idx in converted_idx.items():
                    self_idx = (slice(None),) * split_index["mask_loc"] + (i,)
                    result.append(self[self_idx][_idx])
                return torch.cat(result, cat_dim)
        elif is_nd_tensor:
            new_stack_dim = self.stack_dim - num_single + num_none

            def recompose(converted_idx, stack_dim=new_stack_dim):
                stack = []
                for item in converted_idx:
                    if isinstance(item, list):
                        stack.append(recompose(item, stack_dim=stack_dim))
                    else:
                        stack_elt, idx = item
                        if idx != ():
                            stack.append(self.tensordicts[stack_elt][idx])
                        else:
                            stack.append(self.tensordicts[stack_elt])

                # TODO: this produces multiple dims with the same name
                result = LazyStackedTensorDict.lazy_stack(
                    stack, stack_dim, stack_dim_name=self._td_dim_name
                )
                if self.is_locked:
                    result.lock_()
                return result

            return recompose(converted_idx)
        else:
            if isinteger:
                for (
                    i,
                    _idx,
                ) in (
                    converted_idx.items()
                ):  # for convenience but there's only one element
                    result = self.tensordicts[i]
                    if _idx is not None and _idx != ():
                        result = result[_idx]
                    return result
            else:
                result = []
                new_stack_dim = self.stack_dim - num_single + num_none - num_squash
                for i, _idx in converted_idx.items():
                    if _idx == ():
                        result.append(self.tensordicts[i])
                    else:
                        result.append(self.tensordicts[i][_idx])
                result = LazyStackedTensorDict.lazy_stack(
                    result, new_stack_dim, stack_dim_name=self._td_dim_name
                )
                if self.is_locked:
                    result.lock_()
                return result

    def __eq__(self, other):
        return self._dispatch_comparison(other, "__eq__", "__eq__", default=False)

    def __ne__(self, other):
        return self._dispatch_comparison(other, "__ne__", "__ne__", default=True)

    def __or__(self, other):
        return self._dispatch_comparison(other, "__or__", "__or__", default=NO_DEFAULT)

    def __xor__(self, other):
        return self._dispatch_comparison(
            other, "__xor__", "__xor__", default=NO_DEFAULT
        )

    def __ge__(self, other):
        return self._dispatch_comparison(other, "__ge__", "__le__", default=NO_DEFAULT)

    def __gt__(self, other):
        return self._dispatch_comparison(other, "__gt__", "__lt__", default=NO_DEFAULT)

    def __le__(self, other):
        return self._dispatch_comparison(other, "__le__", "__ge__", default=NO_DEFAULT)

    def __lt__(self, other):
        return self._dispatch_comparison(other, "__lt__", "__gt__", default=NO_DEFAULT)

    def _dispatch_comparison(self, other, comparison_str, inverse_str, default):
        if is_tensorclass(other):
            return getattr(other, inverse_str)(self)
        if isinstance(other, (dict,)):
            # we may want to broadcast it instead
            other = TensorDict.from_dict(other, batch_size=self.batch_size)
        if _is_tensor_collection(type(other)):
            if other.batch_size != self.batch_size:
                if self.ndim < other.ndim:
                    self_expand = self.expand(other.batch_size)
                elif self.ndim > other.ndim:
                    other = other.expand(self.batch_size)
                    self_expand = self
                else:
                    raise RuntimeError(
                        f"Could not compare tensordicts with shapes {self.shape} and {other.shape}"
                    )
            else:
                self_expand = self
            out = []
            for td0, td1 in _zip_strict(
                self_expand.tensordicts, other.unbind(self_expand.stack_dim)
            ):
                out.append(getattr(td0, comparison_str)(td1))
            return LazyStackedTensorDict.lazy_stack(out, self.stack_dim)
        if isinstance(other, (numbers.Number, Tensor)):
            return LazyStackedTensorDict.lazy_stack(
                [getattr(td, comparison_str)(other) for td in self.tensordicts],
                self.stack_dim,
            )
        if default is NO_DEFAULT:
            raise ValueError(
                f"Incompatible value {type(other)} for op {comparison_str}."
            )
        return default

    def _cast_reduction(
        self,
        *,
        reduction_name,
        dim=NO_DEFAULT,
        keepdim=NO_DEFAULT,
        tuple_ok=True,
        further_reduce: bool,
        **kwargs,
    ):
        if further_reduce:
            if dim is NO_DEFAULT:
                # It is not very memory-efficient to do this, but it's the easiest to cover all use cases
                agglomerate = [
                    val.contiguous().flatten()
                    for val in self._values_list(
                        True, True, is_leaf=_NESTED_TENSORS_AS_LISTS
                    )
                ]
                agglomerate = torch.cat(agglomerate, dim=-1)
                return getattr(torch, reduction_name)(agglomerate)
            elif dim == "feature":

                def proc_val(val):
                    val = val.contiguous()
                    if val.ndim > self.ndim:
                        val = val.flatten(self.ndim, -1)
                    else:
                        val = val.unsqueeze(-1)
                    return val

                agglomerate = [
                    proc_val(val)
                    for val in self.values(
                        True,
                        True,
                    )
                ]
                dim = -1
                cat_dim = -1
                keepdim = False
            else:
                agglomerate = [
                    val.contiguous().unsqueeze(self.stack_dim)
                    for val in self.values(True, True)
                ]
                cat_dim = self.stack_dim
            agglomerate = torch.cat(agglomerate, dim=cat_dim)
            return getattr(torch, reduction_name)(agglomerate, dim=dim, keepdim=keepdim)

        try:
            td: TensorDict = self.to_tensordict()
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
            further_reduce=further_reduce,
            **kwargs,
        )

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

    def _send(
        self,
        dst: int,
        _tag: int = -1,
        pseudo_rand: bool = False,
        group: "torch.distributed.ProcessGroup" | None = None,
    ) -> int:
        for td in self.tensordicts:
            _tag = td._send(dst, _tag=_tag, pseudo_rand=pseudo_rand, group=group)
        return _tag

    def _isend(
        self,
        dst: int,
        _tag: int = -1,
        _futures: list[torch.Future] | None = None,
        pseudo_rand: bool = False,
        group: "torch.distributed.ProcessGroup" | None = None,
    ) -> int:
        if _futures is None:
            is_root = True
            _futures = []
        else:
            is_root = False
        for td in self.tensordicts:
            _tag = td._isend(
                dst, _tag=_tag, pseudo_rand=pseudo_rand, _futures=_futures, group=group
            )
        if is_root:
            for future in _futures:
                future.wait()
        return _tag

    def _recv(
        self,
        src: int,
        _tag: int = -1,
        pseudo_rand: bool = False,
        group: "torch.distributed.ProcessGroup" | None = None,
    ) -> int:
        for td in self.tensordicts:
            _tag = td._recv(src, _tag=_tag, pseudo_rand=pseudo_rand, group=group)
        return _tag

    def _irecv(
        self,
        src: int,
        return_premature: bool = False,
        _tag: int = -1,
        _future_list: list[torch.Future] = None,
        pseudo_rand: bool = False,
        group: "torch.distributed.ProcessGroup" | None = None,
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
                group=group,
            )

        if not root:
            return _tag, _future_list
        elif return_premature:
            return _future_list
        else:
            for future in _future_list:
                future.wait()
            return

    @lock_blocked
    def del_(self, key: NestedKey, **kwargs: Any) -> T:
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

    def pop(self, key: NestedKey, default: Any = NO_DEFAULT) -> CompatibleType:
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

    def share_memory_(self) -> T:
        for td in self.tensordicts:
            td.share_memory_()
        self.lock_()
        return self

    def detach_(self) -> T:
        for td in self.tensordicts:
            td.detach_()
        return self

    def _memmap_(
        self,
        *,
        prefix: str | None = None,
        copy_existing: bool = False,
        executor=None,
        futures=None,
        inplace=True,
        like=False,
        share_non_tensor,
        existsok,
    ) -> T:
        if prefix is not None:
            prefix = Path(prefix)

            def save_metadata(prefix=prefix, self=self):
                prefix = Path(prefix)
                if not prefix.exists():
                    os.makedirs(prefix, exist_ok=True)
                with open(prefix / "meta.json", "wb") as f:
                    f.write(
                        json.dumps(
                            {"_type": str(type(self)), "stack_dim": self.stack_dim}
                        )
                    )

            if executor is None:
                save_metadata()
            else:
                futures.append(executor.submit(save_metadata))

        results = []
        for i, td in enumerate(self.tensordicts):
            results.append(
                td._memmap_(
                    prefix=(prefix / str(i)) if prefix is not None else None,
                    copy_existing=copy_existing,
                    executor=executor,
                    futures=futures,
                    inplace=inplace,
                    like=like,
                    share_non_tensor=share_non_tensor,
                    existsok=existsok,
                )
            )
        if not inplace:
            results = LazyStackedTensorDict.lazy_stack(results, dim=self.stack_dim)
        else:
            results = self
        results._device = torch.device("cpu")
        return results

    @classmethod
    def _load_memmap(
        cls,
        prefix: str,
        metadata: dict,
        device: torch.device | None = None,
        *,
        out=None,
        **kwargs,
    ) -> LazyStackedTensorDict:
        tensordicts = []
        i = 0
        stack_dim = metadata["stack_dim"]
        if out is not None:
            out = out.unbind(stack_dim)
        while (prefix / str(i)).exists():
            tensordicts.append(
                TensorDict.load_memmap(
                    prefix / str(i),
                    device=device,
                    **kwargs,
                    non_blocking=True,
                    out=out[i] if out is not None else None,
                )
            )
            i += 1
        return cls(*tensordicts, stack_dim=stack_dim, **kwargs)

    def make_memmap(
        self,
        key: NestedKey,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> MemoryMappedTensor:
        raise RuntimeError(
            "Making a memory-mapped tensor after instantiation isn't currently allowed for LazyStack as "
            "it can't return a contiguous view of the lazy stacked tensors. "
            "If this feature is required, open an issue on GitHub to trigger a discussion on the topic!"
        )

    def make_memmap_from_storage(
        self,
        key: NestedKey,
        storage: torch.UntypedStorage,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> MemoryMappedTensor:
        raise RuntimeError(
            "Making a memory-mapped tensor after instantiation isn't currently allowed for LazyStack as "
            "it can't return a contiguous view of the lazy stacked tensors. "
            "If this feature is required, open an issue on GitHub to trigger a discussion on the topic!"
        )

    def make_memmap_from_tensor(
        self, key: NestedKey, tensor: torch.Tensor, *, copy_data: bool = True
    ) -> MemoryMappedTensor:
        raise RuntimeError(
            "Making a memory-mapped tensor after instantiation isn't currently allowed for LazyStack as "
            "it can't return a contiguous view of the lazy stacked tensors. "
            "If this feature is required, open an issue on GitHub to trigger a discussion on the topic!"
        )

    def expand(self, *args: int, inplace: bool = False) -> T:
        if len(args) == 1 and isinstance(args[0], Sequence):
            shape = tuple(args[0])
        else:
            shape = args
        stack_dim = len(shape) + self.stack_dim - self.ndimension()
        new_shape_tensordicts = [v for i, v in enumerate(shape) if i != stack_dim]
        tensordicts = [td.expand(new_shape_tensordicts) for td in self.tensordicts]
        if inplace:
            self.tensordicts = tensordicts
            self.stack_dim = stack_dim
            return self
        return LazyStackedTensorDict.maybe_dense_stack(tensordicts, dim=stack_dim)

    @lock_blocked
    def update(
        self,
        input_dict_or_td: T,
        clone: bool = False,
        *,
        keys_to_update: Sequence[NestedKey] | None = None,
        non_blocking: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
        update_batch_size: bool = False,
        **kwargs: Any,
    ) -> T:
        # This implementation of update is compatible with exclusive keys
        # as well as vmapped lazy stacks.
        # We iterate over the tensordicts rather than iterating over the keys,
        # which requires stacking and unbinding but is also not robust to missing keys.
        if input_dict_or_td is self:
            # no op
            return self
        if is_leaf is None:
            is_leaf = _is_leaf_nontensor
        if isinstance(input_dict_or_td, dict):
            input_dict_or_td = TensorDict.from_dict(
                input_dict_or_td, batch_size=self.batch_size
            )

        if keys_to_update is not None:
            keys_to_update = unravel_key_list(keys_to_update)
            if len(keys_to_update) == 0:
                return self

        if (
            isinstance(input_dict_or_td, LazyStackedTensorDict)
            and input_dict_or_td.stack_dim == self.stack_dim
        ):
            tds = list(self.tensordicts)
            if len(input_dict_or_td.tensordicts) > len(self.tensordicts):
                tds.extend(
                    [td.copy() for td in input_dict_or_td.tensordicts[len(tds) :]]
                )
            elif len(input_dict_or_td.tensordicts) != len(self.tensordicts):
                if update_batch_size:
                    keys_source = set(input_dict_or_td.keys(True))
                    keys_dest = set(self.keys(True))
                    if not keys_dest.issubset(keys_source):
                        raise RuntimeError(
                            "Some keys of the dest tensordict are not present in the source "
                            "during update with mismatching batch-size. "
                            f"batch_size of source={input_dict_or_td.batch_size}, batch_size of dest={self.batch_size}, "
                            f"keys in dest but not in source: {{{keys_dest - keys_source}}}."
                        )
                    self.__init__(
                        *input_dict_or_td.tensordicts,
                        stack_dim=self.stack_dim,
                        hook_out=self.hook_out,
                        hook_in=self.hook_in,
                        stack_dim_name=self._td_dim_name,
                    )
                    return self

                else:
                    raise ValueError(
                        "cannot update stacked tensordicts with different shapes when update_batch_size=False."
                    )
            for td_dest, td_source in _zip_strict(tds, input_dict_or_td.tensordicts):
                td_dest.update(
                    td_source,
                    clone=clone,
                    keys_to_update=keys_to_update,
                    non_blocking=non_blocking,
                    is_leaf=is_leaf,
                    **kwargs,
                )
            return self

        if self.hook_in is not None:
            self_upd = self.hook_in(self)
            input_dict_or_td = self.hook_in(input_dict_or_td)
        else:
            self_upd = self
        # Then we can decompose the tensordict along its stack dim
        if input_dict_or_td.ndim <= self_upd.stack_dim or input_dict_or_td.batch_size[
            self_upd.stack_dim
        ] != len(self_upd.tensordicts):
            try:
                # if the batch-size does not permit unbinding, let's first try to reset the batch-size.
                input_dict_or_td = input_dict_or_td.copy()
                batch_size = self_upd.batch_size
                if self_upd.hook_out is not None:
                    batch_size = list(batch_size)
                    batch_size.insert(self_upd.stack_dim, len(self_upd.tensordicts))
                input_dict_or_td.batch_size = batch_size
            except RuntimeError as err:
                raise ValueError(
                    "cannot update stacked tensordicts with different shapes."
                ) from err
        for td_dest, td_source in _zip_strict(
            self_upd.tensordicts, input_dict_or_td.unbind(self_upd.stack_dim)
        ):
            td_dest.update(
                td_source,
                clone=clone,
                keys_to_update=keys_to_update,
                is_leaf=is_leaf,
                **kwargs,
            )
        if self.hook_out is not None:
            self_upd = self.hook_out(self_upd)
        else:
            self_upd = self
        return self_upd

    def update_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | TensorDictBase,
        clone: bool = False,
        *,
        non_blocking: bool = False,
        **kwargs: Any,
    ) -> T:
        if input_dict_or_td is self:
            # no op
            return self
        if not is_tensor_collection(input_dict_or_td):
            input_dict_or_td = TensorDict.from_dict(
                input_dict_or_td, batch_dims=self.batch_dims
            )
            if input_dict_or_td.batch_dims <= self.stack_dim:
                raise RuntimeError(
                    f"Built tensordict with ndim={input_dict_or_td.ndim} does not have enough dims."
                )
        if input_dict_or_td.batch_size[self.stack_dim] != len(self.tensordicts):
            raise ValueError("cannot update stacked tensordicts with different shapes.")
        for td_dest, td_source in _zip_strict(
            self.tensordicts, input_dict_or_td.unbind(self.stack_dim)
        ):
            td_dest.update_(td_source, clone=clone, non_blocking=non_blocking, **kwargs)
        return self

    def update_at_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | TensorDictBase,
        index: IndexType,
        clone: bool = False,
        *,
        non_blocking: bool = False,
    ) -> T:
        if not _is_tensor_collection(type(input_dict_or_td)):
            input_dict_or_td = TensorDict.from_dict(
                input_dict_or_td, batch_size=self.batch_size
            )
        split_index = self._split_index(index)
        converted_idx = split_index["index_dict"]
        num_single = split_index["num_single"]
        isinteger = split_index["isinteger"]
        if isinteger:
            # this will break if the index along the stack dim is [0] or :1 or smth
            for i, _idx in converted_idx.items():
                self.tensordicts[i].update_at_(
                    input_dict_or_td,
                    _idx,
                    non_blocking=non_blocking,
                )
            return self
        unbind_dim = self.stack_dim - num_single
        for (i, _idx), _value in _zip_strict(
            converted_idx.items(),
            input_dict_or_td.unbind(unbind_dim),
        ):
            self.tensordicts[i].update_at_(
                _value,
                _idx,
                non_blocking=non_blocking,
            )
        return self

    def rename_key_(
        self, old_key: NestedKey, new_key: NestedKey, safe: bool = False
    ) -> T:
        for td in self.tensordicts:
            td.rename_key_(old_key, new_key, safe=safe)
        return self

    rename_key = _renamed_inplace_method(rename_key_)

    def where(self, condition, other, *, out=None, pad=None):
        if condition.ndim < self.ndim:
            condition = expand_right(condition, self.batch_size)
        condition = condition.unbind(self.stack_dim)
        if _is_tensor_collection(type(other)) or (
            isinstance(other, Tensor)
            and other.shape[: self.stack_dim] == self.shape[: self.stack_dim]
        ):
            other = other.unbind(self.stack_dim)

            def where(td, cond, other, pad):
                if cond.numel() > 1:
                    return td.where(cond, other, pad=pad)
                return other if not cond else td

            result = LazyStackedTensorDict.lazy_stack(
                [
                    where(td, cond, _other, pad=pad)
                    for td, cond, _other in _zip_strict(
                        self.tensordicts, condition, other
                    )
                ],
                self.stack_dim,
            )
        else:
            result = LazyStackedTensorDict.lazy_stack(
                [
                    td.where(cond, other, pad=pad)
                    for td, cond in _zip_strict(self.tensordicts, condition)
                ],
                self.stack_dim,
            )
        # We should not pass out to stack because this will overwrite the tensors in-place, but
        # we don't want that
        if out is not None:
            out.update(result)
            return out
        return result

    def masked_fill_(self, mask: Tensor, value: float | bool) -> T:
        mask_unbind = mask.unbind(dim=self.stack_dim)
        for _mask, td in _zip_strict(mask_unbind, self.tensordicts):
            td.masked_fill_(_mask, value)
        return self

    def masked_fill(self, mask: Tensor, value: float | bool) -> T:
        td_copy = self.clone()
        return td_copy.masked_fill_(mask, value)

    @lock_blocked
    def insert(self, index: int, tensordict: T) -> None:
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
        if self.tensordicts:
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
        else:
            batch_size = tensordict.batch_size

        self.tensordicts.insert(index, tensordict)

        N = len(self.tensordicts)
        self._batch_size = self._compute_batch_size(batch_size, self.stack_dim, N)

    @lock_blocked
    def append(self, tensordict: T) -> None:
        """Append a TensorDict onto the stack.

        Analogous to list.append. The appended TensorDict must have compatible
        batch_size and device. The append operation is in-place, nothing is returned.

        Args:
            tensordict (TensorDictBase): The TensorDict to be appended onto the stack.

        """
        self.insert(len(self.tensordicts), tensordict)

    @lock_blocked
    def extend(self, tensordict: list[T] | T) -> None:
        """Extends the lazy stack with new tensordicts."""
        if _is_tensor_collection(type(tensordict)):
            tensordict = list(tensordict.unbind(self.stack_dim))
        if any(not isinstance(tensordict, TensorDictBase) for tensordict in tensordict):
            raise TypeError(
                "Expected new value to be TensorDictBase instance but got "
                f"{[type(tensordict) for tensordict in tensordict]} instead."
            )
        if self.tensordicts:
            batch_size = self.tensordicts[0].batch_size
            device = self.tensordicts[0].device

            for _td in tensordict:
                _batch_size = _td.batch_size
                _device = _td.device

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
        else:
            batch_size = tensordict.batch_size

        self.tensordicts.extend(tensordict)

        N = len(self.tensordicts)
        self._batch_size = self._compute_batch_size(batch_size, self.stack_dim, N)

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
            if not self.tensordicts:
                return False
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
    def _lock_parents_weakrefs(self):
        """Weakrefs of all tensordicts that need to be unlocked for this to be unlocked."""
        _lock_parents_weakrefs = []
        for tensordict in self.tensordicts:
            _lock_parents_weakrefs = (
                _lock_parents_weakrefs + tensordict._lock_parents_weakrefs
            )
        _lock_parents_weakrefs = [
            item for item in _lock_parents_weakrefs if item is not weakref.ref(self)
        ]
        return _lock_parents_weakrefs

    def _propagate_lock(self, lock_parents_weakrefs=None, *, is_compiling):
        """Registers the parent tensordict that handles the lock."""
        self._is_locked = True
        if not is_compiling:
            is_root = lock_parents_weakrefs is None
            if is_root:
                lock_parents_weakrefs = []

            lock_parents_weakrefs = copy(lock_parents_weakrefs) + [weakref.ref(self)]
        for dest in self.tensordicts:
            dest._propagate_lock(lock_parents_weakrefs, is_compiling=is_compiling)

    @erase_cache
    def _propagate_unlock(self):
        # we can't set _is_locked to False because after it's unlocked, anything
        # can happen to a child tensordict.
        self._is_locked = None
        sub_tds = defaultdict()
        for child in self.tensordicts:
            # we want to make sure that if the same child is present twice in the
            # stack we won't iterate multiple times over it
            sub_tds[id(child)] = child._propagate_unlock() + [child]
        sub_tds = [item for value in sub_tds.values() for item in value]
        return sub_tds

    def __repr__(self):
        fields = _td_fields(self)
        field_str = indent(f"fields={{{fields}}}", 4 * " ")
        exclusive_fields_str = indent(
            f"exclusive_fields={{{self._repr_exclusive_fields()}}}", 4 * " "
        )
        batch_size_str = indent(f"batch_size={self.batch_size}", 4 * " ")
        device_str = indent(f"device={self.device}", 4 * " ")
        is_shared_str = indent(f"is_shared={self.is_shared()}", 4 * " ")
        stack_dim = indent(f"stack_dim={self.stack_dim}", 4 * " ")
        string = ",\n".join(
            [
                field_str,
                exclusive_fields_str,
                batch_size_str,
                device_str,
                is_shared_str,
                stack_dim,
            ]
        )
        return f"{type(self).__name__}(\n{string})"

    def _exclusive_keys(self):
        return {key for td in self.tensordicts for key in td.keys()}

    def _repr_exclusive_fields(self):
        keys = set(self.keys())
        exclusive_keys = [
            _td_fields(td, [k for k in td.keys() if k not in keys])
            for td in self.tensordicts
        ]
        exclusive_key_str = ",\n".join(
            [
                indent(f"{i} ->{line}", 4 * " ")
                for i, line in enumerate(exclusive_keys)
                if line != "\n"
            ]
        )

        return "\n" + exclusive_key_str

    def _view(self, *args, raise_if_not_view: bool = True, **kwargs) -> T:
        shape = _get_shape_from_args(*args, **kwargs)
        if any(dim < 0 for dim in shape):
            shape = _infer_size_impl(shape, self.numel())

        # Then we just need to reorganize the lazy stack
        shape = torch.Size(shape)
        is_flatten, (i, j) = _check_is_flatten(
            shape, self.batch_size, return_flatten_dim=True
        )
        if is_flatten:
            # we need to get a flat representation of all the elements from dim i to j, starting from j
            tds = [self]
            for _ in range(i, j + 1):
                # for k in range(j, i-1, -1):
                # unbind along k
                tds = [_td for local_td in tds for _td in local_td.unbind(i)]
            # the dim along which to stack is the first, ie, i
            tds = self._new_lazy_unsafe(*tds, stack_dim=i)
            if self.is_locked:
                return tds.lock_()
            return tds

        is_unflatten, (i, j) = _check_is_unflatten(
            shape, self.batch_size, return_flatten_dim=True
        )
        if is_unflatten:
            # we are going to organize our list of (A*B*C) elements in a nested list of (A * (B * (C))) elements
            tds = self
            for k in range(i, j):
                tds = self._new_lazy_unsafe(
                    *list(tds.chunk(shape[k], dim=k)), stack_dim=k
                )
            if self.is_locked:
                return tds.lock_()
            return tds
        if raise_if_not_view:
            raise RuntimeError(
                "Cannot call `view` on a lazy stacked tensordict. Call `reshape` instead."
            )
        return TensorDict.reshape(self, shape)

    def reshape(
        self,
        *args,
        **kwargs,
    ) -> T:
        return self._view(*args, raise_if_not_view=False, **kwargs)

    def flatten(self, start_dim=0, end_dim=-1):
        end_dim = _maybe_correct_neg_dim(end_dim, shape=self.batch_size)
        start_dim = _maybe_correct_neg_dim(start_dim, shape=self.batch_size)
        new_shape = [
            s for i, s in enumerate(self.batch_size) if i < start_dim or i > end_dim
        ]
        new_shape.insert(start_dim, -1)
        return self.view(new_shape)

    def unflatten(self, dim, unflattened_size):
        dim = _maybe_correct_neg_dim(dim, shape=self.batch_size)
        new_shape = self.batch_size
        if dim == 0:
            new_shape = torch.Size(unflattened_size) + new_shape[1:]
        else:
            new_shape = (
                new_shape[:dim] + torch.Size(unflattened_size) + new_shape[dim + 1 :]
            )
        return self.view(new_shape)

    def _transpose(self, dim0, dim1):
        if self._is_vmapped:
            raise RuntimeError("cannot call transpose within vmap.")
        if dim0 == self.stack_dim:
            # we know dim0 and dim1 are sorted so dim1 comes after dim0
            # example: shape = [5, 4, 3, 2, 1], stack_dim=1, dim0=1, dim1=4
            # resulting shape: [5, 1, 3, 2, 4]
            if dim1 == dim0 + 1:
                result = type(self)(
                    *self.tensordicts, stack_dim=dim1, stack_dim_name=self._td_dim_name
                )
            else:
                result = type(self)(
                    *(td.transpose(dim0, dim1 - 1) for td in self.tensordicts),
                    stack_dim=dim1,
                    stack_dim_name=self._td_dim_name,
                )
        elif dim1 == self.stack_dim:
            # example: shape = [5, 4, 3, 2, 1], stack_dim=3, dim0=1, dim1=3
            # resulting shape: [5, 2, 3, 4, 1]
            if dim0 + 1 == dim1:
                result = type(self)(
                    *self.tensordicts, stack_dim=dim0, stack_dim_name=self._td_dim_name
                )
            else:
                result = type(self)(
                    *(td.transpose(dim0 + 1, dim1) for td in self.tensordicts),
                    stack_dim=dim0,
                    stack_dim_name=self._td_dim_name,
                )
        else:
            dim0 = dim0 if dim0 < self.stack_dim else dim0 - 1
            dim1 = dim1 if dim1 < self.stack_dim else dim1 - 1
            result = type(self)(
                *(td.transpose(dim0, dim1) for td in self.tensordicts),
                stack_dim=self.stack_dim,
                stack_dim_name=self._td_dim_name,
            )
        return result

    def _repeat(self, *repeats: int) -> TensorDictBase:
        repeats = list(repeats)
        r_dim = repeats.pop(self.stack_dim)
        tds = [td.repeat(*repeats) for td in self.tensordicts]
        tds = [td for _ in range(r_dim) for td in tds]
        return type(self)(
            *tds,
            stack_dim=self.stack_dim,
            stack_dim_name=self._td_dim_name,
            hook_in=self.hook_in,
            hook_out=self.hook_out,
        )

    def repeat_interleave(
        self, repeats: torch.Tensor | int, dim: int = None, *, output_size: int = None
    ) -> TensorDictBase:
        if self.ndim == 0:
            return self.unsqueeze(0).repeat_interleave(
                repeats=repeats, dim=dim, output_size=output_size
            )
        if dim is None:
            if self.ndim > 1:
                return self.reshape(-1).repeat_interleave(repeats, dim=0)
            return self.repeat_interleave(repeats, dim=0)
        dim_corrected = dim if dim >= 0 else self.ndim + dim
        if not (dim_corrected >= 0):
            raise ValueError(
                f"dim {dim} is out of range for tensordict with shape {self.shape}."
            )
        if dim_corrected == self.stack_dim:
            new_list_of_tds = [t for t in self.tensordicts for _ in range(repeats)]
            result = type(self)(
                *new_list_of_tds,
                stack_dim=self.stack_dim,
                stack_dim_name=self._td_dim_name,
                hook_out=self.hook_out,
                hook_in=self.hook_in,
            )
        else:
            dim_corrected = (
                dim_corrected if dim_corrected < self.stack_dim else dim_corrected - 1
            )
            result = type(self)(
                *(
                    td.repeat_interleave(
                        repeats=repeats, dim=dim_corrected, output_size=output_size
                    )
                    for td in self.tensordicts
                ),
                stack_dim=self.stack_dim,
                stack_dim_name=self._td_dim_name,
                hook_in=self.hook_in,
                hook_out=self.hook_out,
            )
        return result

    def _permute(
        self,
        *args,
        **kwargs,
    ):
        dims_list = _get_shape_from_args(*args, kwarg_name="dims", **kwargs)
        dims_list = [dim if dim >= 0 else self.ndim + dim for dim in dims_list]
        dims_list_sort = np.argsort(dims_list)
        # find the new stack dim
        stack_dim = dims_list_sort[self.stack_dim]
        # remove that dim from the dims_list
        dims_list = [
            d if d < self.stack_dim else d - 1 for d in dims_list if d != self.stack_dim
        ]
        result = LazyStackedTensorDict.lazy_stack(
            [td.permute(dims_list) for td in self.tensordicts],
            stack_dim,
            stack_dim_name=self._td_dim_name,
        )
        return result

    def _squeeze(self, dim=None):
        if dim is not None:
            new_dim = dim
            if new_dim < 0:
                new_dim = self.batch_dims + new_dim
            if new_dim > self.batch_dims - 1 or new_dim < 0:
                raise RuntimeError(
                    f"The dim provided to squeeze is incompatible with the tensordict shape: dim={dim} and batch_size={self.batch_size}."
                )
            dim = new_dim
            if self.batch_size[dim] != 1:
                return self
            if dim == self.stack_dim:
                return self.tensordicts[0]
            if dim > self.stack_dim:
                dim = dim - 1
                stack_dim = self.stack_dim
            else:
                stack_dim = self.stack_dim - 1
            result = LazyStackedTensorDict.lazy_stack(
                [td.squeeze(dim) for td in self.tensordicts],
                stack_dim,
                stack_dim_name=self._td_dim_name,
            )
        else:
            result = self
            for dim in range(self.batch_dims - 1, -1, -1):
                if self.batch_size[dim] == 1:
                    result = result.squeeze(dim)
        return result

    def _unsqueeze(self, dim):
        new_dim = dim
        if new_dim < 0:
            new_dim = self.batch_dims + new_dim + 1
        if new_dim > self.batch_dims or new_dim < 0:
            raise RuntimeError(
                f"The dim provided to unsqueeze is incompatible with the tensordict shape: dim={dim} and batch_size={self.batch_size}."
            )
        dim = new_dim
        if dim > self.stack_dim:
            dim = dim - 1
            stack_dim = self.stack_dim
        else:
            stack_dim = self.stack_dim + 1
        result = LazyStackedTensorDict.lazy_stack(
            [td.unsqueeze(dim) for td in self.tensordicts],
            stack_dim,
            stack_dim_name=self._td_dim_name,
        )
        return result

    def split(self, split_size: int | list[int], dim: int = 0) -> list[TensorDictBase]:
        dim = _maybe_correct_neg_dim(dim, shape=self.shape)
        if dim == self.stack_dim:
            if isinstance(split_size, int):
                split_size = [split_size] * -(len(self.tensordicts) // -split_size)
                split_size[-1] = len(self.tensordicts) - sum(split_size[:-1])

            def iter_across_tds():
                start = 0
                for s in split_size:
                    if s == 0:
                        batch_size = list(self._batch_size)
                        batch_size[self.stack_dim] = 0
                        yield LazyStackedTensorDict(
                            batch_size=batch_size,
                            device=self.device,
                            stack_dim=self.stack_dim,
                        )
                        continue
                    stop = start + s
                    yield self._new_lazy_unsafe(
                        *self.tensordicts[slice(start, stop)], stack_dim=self.stack_dim
                    )
                    start = stop

            return tuple(iter_across_tds())
        tds = []
        split_dim = dim if dim < self.stack_dim else dim - 1
        for td in self.tensordicts:
            tds.append(td.split(split_size, split_dim))
        return tuple(
            self._new_lazy_unsafe(*tds, stack_dim=self.stack_dim)
            for tds in _zip_strict(*tds)
        )

    lock_ = TensorDictBase.lock_
    lock = _renamed_inplace_method(lock_)

    unlock_ = TensorDictBase.unlock_
    unlock = _renamed_inplace_method(unlock_)

    _check_device = TensorDict._check_device
    _check_is_shared = TensorDict._check_is_shared
    _convert_to_tensordict = TensorDict._convert_to_tensordict
    _index_tensordict = TensorDict._index_tensordict
    masked_select = TensorDict.masked_select
    _to_module = TensorDict._to_module
    from_dict_instance = TensorDict.from_dict_instance


class _CustomOpTensorDict(TensorDictBase):
    """Encodes lazy operations on tensors contained in a TensorDict."""

    _safe = False
    _lazy = True

    def __init__(
        self,
        source: T,
        custom_op: str,
        inv_op: str | None = None,
        custom_op_kwargs: dict | None = None,
        inv_op_kwargs: dict | None = None,
        batch_size: Sequence[int] | None = None,
    ) -> None:

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

    # These attributes should never be set
    @property
    @cache  # noqa
    def _is_shared(self):
        return self._source._is_shared

    @property
    @cache  # noqa
    def _is_memmap(self):
        return self._source._is_memmap

    def is_empty(self) -> bool:
        return self._source.is_empty()

    def is_memmap(self) -> bool:
        return self._source.is_memmap()

    def is_shared(self) -> bool:
        return self._source.is_shared()

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

    @classmethod
    def from_dict(
        cls, input_dict, batch_size=None, device=None, batch_dims=None, names=None
    ):
        raise NotImplementedError(f"from_dict not implemented for {cls.__name__}.")

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
            f"Cannot erase names of a {type(self).__name__}. "
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
        self._batch_size = new_size

    def _get_str(self, key, default, **kwargs):
        tensor = self._source._get_str(key, default, **kwargs)
        if tensor is default:
            return tensor
        return self._transform_value(tensor)

    def _get_tuple(self, key, default, **kwargs):
        tensor = self._source._get_tuple(key, default, **kwargs)
        if tensor is default:
            return tensor
        return self._transform_value(tensor)

    def _transform_value(self, item):
        return getattr(item, self.custom_op)(**self._update_custom_op_kwargs(item))

    def _set_str(
        self,
        key,
        value,
        *,
        inplace: bool,
        validated: bool,
        ignore_lock: bool = False,
        non_blocking: bool = False,
    ):
        if not validated:
            value = self._validate_value(
                value, check_shape=True, non_blocking=non_blocking
            )
            validated = True
        value = getattr(value, self.inv_op)(**self._update_inv_op_kwargs(value))
        self._source._set_str(
            key,
            value,
            inplace=inplace,
            validated=validated,
            ignore_lock=ignore_lock,
            non_blocking=non_blocking,
        )
        return self

    def _set_tuple(
        self, key, value, *, inplace: bool, validated: bool, non_blocking: bool
    ):
        if len(key) == 1:
            return self._set_str(
                key[0],
                value,
                inplace=inplace,
                validated=validated,
                non_blocking=non_blocking,
            )
        source = self._source._get_str(key[0], None)
        if source is None:
            source = self._source._create_nested_str(key[0])
        nested = type(self)(
            source,
            custom_op=self.custom_op,
            inv_op=self.inv_op,
            custom_op_kwargs=self._update_custom_op_kwargs(source),
            inv_op_kwargs=self._update_inv_op_kwargs(source),
        )
        nested._set_tuple(
            key[1:],
            value,
            inplace=inplace,
            validated=validated,
            non_blocking=non_blocking,
        )
        return self

    def _set_at_str(self, key, value, idx, *, validated, non_blocking: bool):
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

    def _set_at_tuple(self, key, value, idx, *, validated, non_blocking: bool):
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
            value = self._validate_value(
                value, check_shape=False, non_blocking=non_blocking
            )

        transformed_tensor[idx] = value
        return self

    def _stack_onto_(
        self,
        list_item: list[CompatibleType],
        dim: int,
    ) -> T:
        raise RuntimeError(
            f"stacking tensordicts is not allowed for type {type(self).__name__}"
            f"consider calling 'to_tensordict()` first"
        )

    def __repr__(self) -> str:
        custom_op_kwargs_str = ", ".join(
            [f"{key}={value}" for key, value in self.custom_op_kwargs.items()]
        )
        indented_source = textwrap.indent(f"source={self._source}", "\t")
        return (
            f"{type(self).__name__}(\n{indented_source}, "
            f"\n\top={self.custom_op}({custom_op_kwargs_str}))"
        )

    # @cache  # noqa: B019
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

    def _select(
        self,
        *keys: NestedKey,
        inplace: bool = False,
        strict: bool = True,
        set_shared: bool = True,
    ) -> _CustomOpTensorDict:
        if inplace:
            raise RuntimeError("Cannot call select inplace on a lazy tensordict.")
        return self.to_tensordict(retain_none=True)._select(
            *keys, inplace=False, strict=strict, set_shared=set_shared
        )

    def _exclude(
        self, *keys: NestedKey, inplace: bool = False, set_shared: bool = True
    ) -> _CustomOpTensorDict:
        if inplace:
            raise RuntimeError("Cannot call exclude inplace on a lazy tensordict.")
        return self.to_tensordict()._exclude(
            *keys, inplace=False, set_shared=set_shared
        )

    def _clone(self, recurse: bool = True) -> T:
        """Clones the Lazy TensorDict.

        Args:
            recurse (bool, optional): if ``True`` (default), a regular
                :class:`~.tensordict.TensorDict` instance will be returned.
                Otherwise, another :class:`~.tensordict.SubTensorDict` with identical content
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

    def contiguous(self) -> T:
        def contiguous(x):
            return x.contiguous()

        return self._fast_apply(contiguous, propagate_lock=True)

    def rename_key_(
        self, old_key: NestedKey, new_key: NestedKey, safe: bool = False
    ) -> _CustomOpTensorDict:
        self._source.rename_key_(old_key, new_key, safe=safe)
        return self

    rename_key = _renamed_inplace_method(rename_key_)

    @lock_blocked
    def del_(self, key: NestedKey) -> _CustomOpTensorDict:
        self._source = self._source.del_(key)
        return self

    def to(self, *args, **kwargs) -> T:
        non_blocking = kwargs.pop("non_blocking", None)
        (
            device,
            dtype,
            _,
            convert_to_format,
            batch_size,
            pin_memory,
            num_threads,
            inplace,
        ) = _parse_to(*args, **kwargs)
        if inplace:
            raise TypeError(f"Cannot use inplace=True with {type(self).__name__}.to().")

        if batch_size is not None:
            raise TypeError(f"Cannot pass batch-size to {type(self).__name__}.to().")
        result = self

        if device is not None and dtype is None and device == self.device:
            return result

        td = self._source.to(*args, non_blocking=non_blocking, **kwargs)
        self_copy = copy(self)
        self_copy._source = td
        return self_copy

    def pin_memory(
        self, *, num_threads: int | str = 0, inplace: bool = False
    ) -> _CustomOpTensorDict:
        _source = self._source.pin_memory(num_threads=num_threads, inplace=inplace)
        if not inplace:
            return type(self)(
                source=_source,
                custom_op=self.custom_op,
                inv_op=self.inv_op,
                custom_op_kwargs=self.custom_op_kwargs,
                inv_op_kwargs=self.inv_op_kwargs,
                batch_size=self.batch_size,
            )
        return self

    @lock_blocked
    def popitem(self) -> Tuple[NestedKey, CompatibleType]:
        key, val = self._source.popitem()
        return key, self._transform_value(val)

    def detach_(self) -> _CustomOpTensorDict:
        self._source.detach_()
        return self

    def where(self, condition, other, *, out=None, pad=None):
        return self.to_tensordict().where(
            condition=condition, other=other, out=out, pad=pad
        )

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

    def masked_fill(self, mask: Tensor, value: float | bool) -> T:
        td_copy = self.clone()
        return td_copy.masked_fill_(mask, value)

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
    ) -> T:
        def save_metadata(data: TensorDictBase, filepath, metadata=None):
            if metadata is None:
                metadata = {}
            metadata.update(
                {
                    "shape": list(data.shape),
                    "device": str(data.device),
                    "_type": str(type(data)),
                    "custom_op": data.custom_op,
                    "inv_op": data.inv_op,
                    "custom_op_kwargs": data.custom_op_kwargs,
                    "inv_op_kwargs": data.inv_op_kwargs,
                }
            )
            with open(filepath, "wb") as json_metadata:
                json_metadata.write(json.dumps(metadata))

        if prefix is not None:
            prefix = Path(prefix)
            if not prefix.exists():
                os.makedirs(prefix, exist_ok=True)
            metadata = {}

        dest_source = self._source._memmap_(
            prefix=None if prefix is None else prefix / "_source",
            copy_existing=copy_existing,
            executor=executor,
            futures=futures,
            inplace=inplace,
            like=like,
            share_non_tensor=share_non_tensor,
            existsok=existsok,
        )
        if not inplace:
            dest = type(self)(
                dest_source,
                custom_op=self.custom_op,
                inv_op=self.inv_op,
                custom_op_kwargs=self.custom_op_kwargs,
                inv_op_kwargs=self.inv_op_kwargs,
                batch_size=self.batch_size,
            )
        else:
            dest = self

        if prefix is not None:
            if executor is None:
                save_metadata(
                    dest,
                    prefix / "meta.json",
                    metadata=metadata,
                )
            else:
                futures.append(
                    executor.submit(save_metadata, dest, prefix / "meta.json", metadata)
                )
        return dest

    @classmethod
    def _load_memmap(cls, prefix: str, metadata: dict, **kwargs) -> _CustomOpTensorDict:
        custom_op = metadata.pop("custom_op")
        inv_op = metadata.pop("inv_op")
        custom_op_kwargs = metadata.pop("custom_op_kwargs")
        inv_op_kwargs = metadata.pop("inv_op_kwargs")

        source = TensorDict.load_memmap(prefix / "_source", **kwargs, non_blocking=True)

        return cls(
            source,
            custom_op=custom_op,
            inv_op=inv_op,
            custom_op_kwargs=custom_op_kwargs,
            inv_op_kwargs=inv_op_kwargs,
        )

    def make_memmap(
        self,
        key: NestedKey,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> MemoryMappedTensor:
        raise RuntimeError(
            "Making a memory-mapped tensor after instantiation isn't currently allowed for lazy tensordicts."
            "If this feature is required, open an issue on GitHub to trigger a discussion on the topic!"
        )

    def make_memmap_from_storage(
        self,
        key: NestedKey,
        storage: torch.UntypedStorage,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> MemoryMappedTensor:
        raise RuntimeError(
            "Making a memory-mapped tensor after instantiation isn't currently allowed for lazy tensordicts."
            "If this feature is required, open an issue on GitHub to trigger a discussion on the topic!"
        )

    def make_memmap_from_tensor(
        self, key: NestedKey, tensor: torch.Tensor, *, copy_data: bool = True
    ) -> MemoryMappedTensor:
        raise RuntimeError(
            "Making a memory-mapped tensor after instantiation isn't currently allowed for lazy tensordicts."
            "If this feature is required, open an issue on GitHub to trigger a discussion on the topic!"
        )

    def share_memory_(self) -> _CustomOpTensorDict:
        self._source.share_memory_()
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

    @_as_context_manager("is_locked")
    def lock_(self) -> T:
        self._source.lock_()
        return self

    @erase_cache
    @_as_context_manager("is_locked")
    def unlock_(self) -> T:
        self._source.unlock_()
        return self

    def _remove_lock(self, lock_id):
        return self._source._remove_lock(lock_id)

    @erase_cache
    def _propagate_lock(self, lock_ids, *, is_compiling):
        return self._source._propagate_lock(lock_ids, is_compiling=is_compiling)

    @erase_cache
    def _propagate_unlock(self):
        return self._source._propagate_unlock()

    lock = _renamed_inplace_method(lock_)
    unlock = _renamed_inplace_method(unlock_)

    def __del__(self):
        pass

    @property
    def sorted_keys(self):
        return self._source.sorted_keys

    def _view(self, *args, **kwargs):
        raise RuntimeError(
            "Cannot call `view` on a lazy tensordict. Call `reshape` instead."
        )

    def _transpose(self, dim0, dim1):
        raise RuntimeError(
            "Cannot call `transpose` on a lazy tensordict. Make it dense before calling this method by calling `to_tensordict`."
        )

    def _permute(
        self,
        *args,
        **kwargs,
    ):
        raise RuntimeError(
            "Cannot call `permute` on a lazy tensordict. Make it dense before calling this method by calling `to_tensordict`."
        )

    def _squeeze(self, dim=None):
        raise RuntimeError(
            "Cannot call `squeeze` on a lazy tensordict. Make it dense before calling this method by calling `to_tensordict`."
        )

    def _unsqueeze(self, dim):
        raise RuntimeError(
            "Cannot call `unsqueeze` on a lazy tensordict. Make it dense before calling this method by calling `to_tensordict`."
        )

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

    __xor__ = TensorDict.__xor__
    __or__ = TensorDict.__or__
    __eq__ = TensorDict.__eq__
    __ne__ = TensorDict.__ne__
    __ge__ = TensorDict.__ge__
    __gt__ = TensorDict.__gt__
    __le__ = TensorDict.__le__
    __lt__ = TensorDict.__lt__
    __setitem__ = TensorDict.__setitem__
    _add_batch_dim = TensorDict._add_batch_dim
    _check_device = TensorDict._check_device
    _check_is_shared = TensorDict._check_is_shared
    _convert_to_tensordict = TensorDict._convert_to_tensordict
    _index_tensordict = TensorDict._index_tensordict

    _apply_nest = TensorDict._apply_nest
    _get_names_idx = TensorDict._get_names_idx
    _maybe_remove_batch_dim = TensorDict._maybe_remove_batch_dim
    _multithread_apply_flat = TensorDict._multithread_apply_flat
    _multithread_rebuild = TensorDict._multithread_rebuild
    _remove_batch_dim = TensorDict._remove_batch_dim
    _to_module = TensorDict._to_module
    _unbind = TensorDict._unbind
    all = TensorDict.all
    any = TensorDict.any
    expand = TensorDict.expand
    from_dict_instance = TensorDict.from_dict_instance
    masked_select = TensorDict.masked_select
    _repeat = TensorDict._repeat
    repeat_interleave = TensorDict.repeat_interleave
    reshape = TensorDict.reshape
    split = TensorDict.split


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

    def _legacy_squeeze(self, dim: int | None) -> T:
        if dim is not None and dim < 0:
            dim = self.batch_dims + dim
        if dim == self.custom_op_kwargs.get("dim"):
            return self._source
        return super()._legacy_squeeze(dim)

    def _stack_onto_(
        self,
        list_item: list[CompatibleType],
        dim: int,
    ) -> T:
        unsqueezed_dim = self.custom_op_kwargs["dim"]
        diff_to_apply = 1 if dim < unsqueezed_dim else 0
        list_item_unsqueeze = [
            item.squeeze(unsqueezed_dim - diff_to_apply) for item in list_item
        ]
        return self._source._stack_onto_(list_item_unsqueeze, dim)

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

    def _legacy_unsqueeze(self, dim: int) -> T:
        if dim < 0:
            dim = self.batch_dims + dim + 1
        inv_op_dim = self.inv_op_kwargs.get("dim")
        if inv_op_dim < 0:
            inv_op_dim = self.batch_dims + inv_op_dim + 1
        if dim == inv_op_dim:
            return self._source
        return super()._legacy_unsqueeze(dim)

    def _stack_onto_(
        self,
        list_item: list[CompatibleType],
        dim: int,
    ) -> T:
        squeezed_dim = self.custom_op_kwargs["dim"]
        # dim=0, squeezed_dim=2, [3, 4, 5] [3, 4, 1, 5] [[4, 5], [4, 5], [4, 5]] => unsq 1
        # dim=1, squeezed_dim=2, [3, 4, 5] [3, 4, 1, 5] [[3, 5], [3, 5], [3, 5], [3, 4]] => unsq 1
        # dim=2, squeezed_dim=2, [3, 4, 5] [3, 4, 1, 5] [[3, 4], [3, 4], ...] => unsq 2
        diff_to_apply = 1 if dim < squeezed_dim else 0
        list_item_unsqueeze = [
            item.unsqueeze(squeezed_dim - diff_to_apply) for item in list_item
        ]
        return self._source._stack_onto_(list_item_unsqueeze, dim)

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

    def _legacy_view(
        self, *shape: int, size: list | tuple | torch.Size | None = None
    ) -> T:
        if len(shape) == 0 and size is not None:
            return self._legacy_view(*size)
        elif len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
            return self._legacy_view(*shape[0])
        elif not isinstance(shape, torch.Size):
            shape = infer_size_impl(shape, self.numel())
            shape = torch.Size(shape)
        if shape == self._source.batch_size:
            return self._source
        return super()._legacy_view(*shape)

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

    def _legacy_transpose(self, dim0, dim1) -> T:
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
        return super()._legacy_transpose(dim0, dim1)

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
        list_item: list[CompatibleType],
        dim: int,
    ) -> T:
        trsp = self.custom_op_kwargs["dim0"], self.custom_op_kwargs["dim1"]
        if dim == trsp[0]:
            dim = trsp[1]
        elif dim == trsp[1]:
            dim = trsp[0]

        list_permuted_items = []
        for item in list_item:
            list_permuted_items.append(item.transpose(*trsp))
        self._source._stack_onto_(list_permuted_items, dim)
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

    def _legacy_permute(
        self,
        *dims_list: int,
        dims: Sequence[int] | None = None,
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
        if np.array_equal(np.argsort(dims_list), self.inv_op_kwargs.get("dims")):
            return self._source
        return super()._legacy_permute(*dims_list)

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
        list_item: list[CompatibleType],
        dim: int,
    ) -> T:
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
        self._source._stack_onto_(list_permuted_items, new_dim)
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


def _iter_items_lazystack(
    tensordict: LazyStackedTensorDict, return_none_for_het_values: bool = False
) -> Iterator[tuple[str, CompatibleType]]:
    for key in tensordict.tensordicts[0].keys():
        values = tensordict._maybe_get_list(key)
        if values is not None:
            yield key, values


_register_tensor_class(LazyStackedTensorDict)
_register_tensor_class(_CustomOpTensorDict)
_register_tensor_class(_PermutedTensorDict)
_register_tensor_class(_SqueezedTensorDict)
_register_tensor_class(_UnsqueezedTensorDict)
_register_tensor_class(_TransposedTensorDict)
_register_tensor_class(_ViewedTensorDict)

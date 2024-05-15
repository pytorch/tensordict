# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import numbers
import os
from collections import defaultdict
from copy import copy
from numbers import Number
from pathlib import Path
from textwrap import indent
from typing import Any, Callable, Iterable, Iterator, List, Sequence, Tuple, Type
from warnings import warn

import numpy as np
import torch

try:
    from functorch import dim as ftdim

    _has_funcdim = True
except ImportError:
    from tensordict.utils import _ftdim_mock as ftdim

    _has_funcdim = False

from tensordict.base import (
    _ACCEPTED_CLASSES,
    _default_is_leaf,
    _is_tensor_collection,
    _load_metadata,
    _register_tensor_class,
    BEST_ATTEMPT_INPLACE,
    CompatibleType,
    is_tensor_collection,
    NO_DEFAULT,
    T,
    TensorDictBase,
)

from tensordict.memmap import MemoryMappedTensor
from tensordict.utils import (
    _add_batch_dim_pre_hook,
    _BatchedUninitializedBuffer,
    _BatchedUninitializedParameter,
    _clone_value,
    _expand_to_match_shape,
    _get_item,
    _get_leaf_tensordict,
    _get_shape_from_args,
    _getitem_batch_size,
    _index_preserve_data_ptr,
    _is_number,
    _is_shared,
    _is_tensorclass,
    _KEY_ERROR,
    _LOCK_ERROR,
    _NON_STR_KEY_ERR,
    _NON_STR_KEY_TUPLE_ERR,
    _parse_to,
    _prune_selected_keys,
    _set_item,
    _set_max_batch_size,
    _shape,
    _STRDTYPE2DTYPE,
    _StringOnlyDict,
    _sub_index,
    _unravel_key_to_tuple,
    as_decorator,
    Buffer,
    cache,
    convert_ellipsis_to_idx,
    DeviceType,
    expand_as_right,
    IndexType,
    is_non_tensor,
    is_tensorclass,
    KeyedJaggedTensor,
    lock_blocked,
    NestedKey,
    unravel_key,
    unravel_key_list,
)
from torch import Tensor
from torch.jit._shape_functions import infer_size_impl
from torch.utils._pytree import tree_map

_register_tensor_class(ftdim.Tensor)

__base__setattr__ = torch.nn.Module.__setattr__

_has_mps = torch.backends.mps.is_available()
_has_cuda = torch.cuda.is_available()
_has_functorch = False
try:
    try:
        from torch._C._functorch import (
            _add_batch_dim,
            _remove_batch_dim,
            is_batchedtensor,
        )
    except ImportError:
        from functorch._C import is_batchedtensor

    _has_functorch = True
except ImportError:
    _has_functorch = False

    def is_batchedtensor(tensor: Tensor) -> bool:
        """Placeholder for the functorch function."""
        return False


class TensorDict(TensorDictBase):
    """A batched dictionary of tensors.

    TensorDict is a tensor container where all tensors are stored in a
    key-value pair fashion and where each element shares the same first ``N``
    leading dimensions shape, where is an arbitrary number with ``N >= 0``.

    Additionally, if the tensordict has a specified device, then each element
    must share that device.

    TensorDict instances support many regular tensor operations with the notable
    exception of algebraic operations:

    - operations on shape: when a shape operation is called (indexing,
      reshape, view, expand, transpose, permute,
      unsqueeze, squeeze, masking etc), the operations is done as if it
      was executed on a tensor of the same shape as the batch size then
      expended to the right, e.g.:

        >>> td = TensorDict({'a': torch.zeros(3, 4, 5)}, batch_size=[3, 4])
        >>> # returns a TensorDict of batch size [3, 4, 1]:
        >>> td_unsqueeze = td.unsqueeze(-1)
        >>> # returns a TensorDict of batch size [12]
        >>> td_view = td.view(-1)
        >>> # returns a tensor of batch size [12, 4]
        >>> a_view = td.view(-1).get("a")

    - casting operations: a TensorDict can be cast on a different device using

        >>> td_cpu = td.to("cpu")
        >>> dictionary = td.to_dict()

      A call of the `.to()` method with a dtype will return an error.

    - Cloning (:meth:`~TensorDictBase.clone`), contiguous (:meth:`~TensorDictBase.contiguous`);

    - Reading: `td.get(key)`, `td.get_at(key, index)`

    - Content modification: :obj:`td.set(key, value)`, :obj:`td.set_(key, value)`,
      :obj:`td.update(td_or_dict)`, :obj:`td.update_(td_or_dict)`, :obj:`td.fill_(key,
      value)`, :obj:`td.rename_key_(old_name, new_name)`, etc.

    - Operations on multiple tensordicts: `torch.cat(tensordict_list, dim)`,
      `torch.stack(tensordict_list, dim)`, `td1 == td2`, `td.apply(lambda x+y, other_td)` etc.

    Args:
        source (TensorDict or Dict[NestedKey, Union[Tensor, TensorDictBase]]): a
            data source. If empty, the tensordict can be populated subsequently.
        batch_size (iterable of int, optional): a batch size for the
            tensordict. The batch size can be modified subsequently as long
            as it is compatible with its content.
            If not batch-size is provided, an empty batch-size is assumed (it
            is not inferred automatically from the data). To automatically set
            the batch-size, refer to :meth:`~.auto_batch_size_`.
        device (torch.device or compatible type, optional): a device for the
            TensorDict. If provided, all tensors will be stored on that device.
            If not, tensors on different devices are allowed.
        names (lsit of str, optional): the names of the dimensions of the
            tensordict. If provided, its length must match the one of the
            ``batch_size``. Defaults to ``None`` (no dimension name, or ``None``
            for every dimension).
        non_blocking (bool, optional): if ``True`` and a device is passed, the tensordict
            is delivered without synchronization. This is the fastest option but is only
            safe when casting from cpu to cuda (otherwise a synchronization call must be
            implemented by the user).
            If ``False`` is passed, every tensor movement will be done synchronously.
            If ``None`` (default), the device casting will be done asynchronously but
            a synchronization will be executed after creation if required. This option
            should generally be faster than ``False`` and potentially slower than ``True``.
        lock (bool, optional): if ``True``, the resulting tensordict will be
            locked.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> source = {'random': torch.randn(3, 4),
        ...     'zeros': torch.zeros(3, 4, 5)}
        >>> batch_size = [3]
        >>> td = TensorDict(source, batch_size=batch_size)
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

    _td_dim_names = None
    _is_shared = False
    _is_memmap = False
    _has_exclusive_keys = False

    def __init__(
        self,
        source: T | dict[str, CompatibleType] = None,
        batch_size: Sequence[int] | torch.Size | int | None = None,
        device: DeviceType | None = None,
        names: Sequence[str] | None = None,
        non_blocking: bool = None,
        lock: bool = False,
        _run_checks: bool = True,
    ) -> None:
        has_device = False
        sub_non_blocking = False
        if device is not None:
            has_device = True
            if non_blocking is None:
                sub_non_blocking = True
                non_blocking = False
            else:
                sub_non_blocking = non_blocking
            device = torch.device(device)
            if _has_mps:
                # With MPS, an explicit sync is required
                sub_non_blocking = True
        self._device = device

        self._tensordict = _tensordict = _StringOnlyDict()
        if not _run_checks:
            self._batch_size = batch_size
            if source:  # faster than calling items
                for key, value in source.items():
                    if isinstance(value, dict):
                        value = TensorDict(
                            value,
                            batch_size=self._batch_size,
                            device=self._device,
                            _run_checks=_run_checks,
                            non_blocking=sub_non_blocking,
                        )
                    _tensordict[key] = value
            self._td_dim_names = names
        else:
            if source is None:
                source = {}
            if not isinstance(source, (TensorDictBase, dict)):
                raise ValueError(
                    "A TensorDict source is expected to be a TensorDictBase "
                    f"sub-type or a dictionary, found type(source)={type(source)}."
                )
            self._batch_size = self._parse_batch_size(source, batch_size)
            self.names = names

            for key, value in source.items():
                self.set(key, value, non_blocking=sub_non_blocking)
        if not non_blocking and sub_non_blocking and has_device:
            self._sync_all()
        if lock:
            self.lock_()

    @classmethod
    def from_module(
        cls,
        module: torch.nn.Module,
        as_module: bool = False,
        lock: bool = False,
        use_state_dict: bool = False,
    ):
        result = cls._from_module(
            module=module, as_module=as_module, use_state_dict=use_state_dict
        )
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
    ):
        from tensordict.nn import TensorDictParams

        if isinstance(module, TensorDictParams):
            return module
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
                destination[name] = buffer

        if use_state_dict:
            for hook in module._state_dict_hooks.values():
                hook_result = hook(module, destination, prefix, local_metadata)
                if hook_result is not None:
                    destination = hook_result
        destination = TensorDict(destination, batch_size=[])
        for name, submodule in module._modules.items():
            if submodule is not None:
                subtd = cls._from_module(
                    module=submodule,
                    as_module=False,
                    use_state_dict=use_state_dict,
                    prefix=prefix + name + ".",
                )
                destination._set_str(
                    name, subtd, validated=True, inplace=False, non_blocking=False
                )

        if as_module:
            from tensordict.nn.params import TensorDictParams

            return TensorDictParams(destination, no_convert=True)
        return destination

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
        return True

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
    ):

        if not use_state_dict and isinstance(module, TensorDictBase):
            if return_swap:
                swap = module.copy()
                module._param_td = getattr(self, "_param_td", self)
                return swap
            else:
                module.update(self)
                return

        # we use __dict__ directly to avoid the getattr/setattr overhead whenever we can
        __dict__ = module.__dict__

        hooks = memo["hooks"]
        if return_swap:
            _swap = {}
            memo[id(module)] = _swap

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
                if isinstance(y, torch.nn.Parameter):
                    return torch.nn.Parameter(x)
                if isinstance(y, Buffer):
                    return Buffer(x)
                return x

            input = state_dict.unflatten_keys(".")._fast_apply(
                convert_type, self, propagate_lock=True
            )
        else:
            input = self
            inplace = bool(inplace)

        for key, value in input.items():
            if isinstance(value, (Tensor, ftdim.Tensor)):
                if module.__class__.__setattr__ is __base__setattr__:
                    # if setattr is the native nn.Module.setattr, we can rely on _set_tensor_dict
                    local_out = _set_tensor_dict(
                        __dict__, hooks, module, key, value, inplace
                    )
                else:
                    if return_swap:
                        local_out = getattr(module, key)
                    if not inplace:
                        # use specialized __setattr__ if needed
                        delattr(module, key)
                        setattr(module, key, value)
                    else:
                        new_val = local_out
                        if return_swap:
                            local_out = local_out.clone()
                        new_val.data.copy_(value.data, non_blocking=non_blocking)
            else:
                if value.is_empty():
                    # if there is at least one key, we must populate the module.
                    # Otherwise, we just go to the next key
                    continue
                child = __dict__["_modules"][key]
                local_out = memo.get(id(child), NO_DEFAULT)

                if local_out is NO_DEFAULT:
                    # if isinstance(child, TensorDictBase):
                    #     # then child is a TensorDictParams
                    #     from tensordict.nn import TensorDictParams
                    #
                    #     local_out = child
                    #     if not isinstance(value, TensorDictParams):
                    #         value = TensorDictParams(value, no_convert=True)
                    #     __dict__["_modules"][key] = value
                    # else:
                    local_out = value._to_module(
                        child,
                        inplace=inplace,
                        return_swap=return_swap,
                        swap_dest={},  # we'll be calling update later
                        memo=memo,
                        use_state_dict=use_state_dict,
                        non_blocking=non_blocking,
                    )

            if return_swap:
                _swap[key] = local_out
        if return_swap:
            if isinstance(swap_dest, dict):
                return _swap
            elif swap_dest is not None:

                def _quick_set(swap_dict, swap_td):
                    for key, val in swap_dict.items():
                        if isinstance(val, dict):
                            _quick_set(val, swap_td._get_str(key, default=NO_DEFAULT))
                        elif swap_td._get_str(key, None) is not val:
                            swap_td._set_str(
                                key,
                                val,
                                inplace=False,
                                validated=True,
                                non_blocking=non_blocking,
                            )

                _quick_set(_swap, swap_dest)
                return swap_dest
            else:
                return TensorDict(_swap, batch_size=[], _run_checks=False)

    def __ne__(self, other: object) -> T | bool:
        if _is_tensorclass(other):
            return other != self
        if isinstance(other, (dict,)):
            other = self.from_dict_instance(other)
        if _is_tensor_collection(other.__class__):
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

    def __xor__(self, other: object) -> T | bool:
        if _is_tensorclass(other):
            return other ^ self
        if isinstance(other, (dict,)):
            other = self.from_dict_instance(other)
        if _is_tensor_collection(other.__class__):
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                raise KeyError(
                    f"keys in {self} and {other} mismatch, got {keys1} and {keys2}"
                )
            d = {}
            for key, item1 in self.items():
                d[key] = item1 ^ other.get(key)
            return TensorDict(batch_size=self.batch_size, source=d, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return TensorDict(
                {key: value ^ other for key, value in self.items()},
                self.batch_size,
                device=self.device,
            )
        return True

    def __or__(self, other: object) -> T | bool:
        if _is_tensorclass(other):
            return other | self
        if isinstance(other, (dict,)):
            other = self.from_dict_instance(other)
        if _is_tensor_collection(other.__class__):
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                raise KeyError(
                    f"keys in {self} and {other} mismatch, got {keys1} and {keys2}"
                )
            d = {}
            for key, item1 in self.items():
                d[key] = item1 | other.get(key)
            return TensorDict(batch_size=self.batch_size, source=d, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return TensorDict(
                {key: value | other for key, value in self.items()},
                self.batch_size,
                device=self.device,
            )
        return False

    def __eq__(self, other: object) -> T | bool:
        if is_tensorclass(other):
            return other == self
        if isinstance(other, (dict,)):
            other = self.from_dict_instance(other)
        if _is_tensor_collection(other.__class__):
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                keys1 = sorted(
                    keys1,
                    key=lambda key: "".join(key) if isinstance(key, tuple) else key,
                )
                keys2 = sorted(
                    keys2,
                    key=lambda key: "".join(key) if isinstance(key, tuple) else key,
                )
                raise KeyError(f"keys in tensordicts mismatch, got {keys1} and {keys2}")
            d = {}
            for key, item1 in self.items():
                d[key] = item1 == other.get(key)
            return TensorDict(source=d, batch_size=self.batch_size, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return TensorDict(
                {key: value == other for key, value in self.items()},
                self.batch_size,
                device=self.device,
            )
        return False

    def __ge__(self, other: object) -> T | bool:
        if is_tensorclass(other):
            return other <= self
        if isinstance(other, (dict,)):
            other = self.from_dict_instance(other)
        if _is_tensor_collection(other.__class__):
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                keys1 = sorted(
                    keys1,
                    key=lambda key: "".join(key) if isinstance(key, tuple) else key,
                )
                keys2 = sorted(
                    keys2,
                    key=lambda key: "".join(key) if isinstance(key, tuple) else key,
                )
                raise KeyError(f"keys in tensordicts mismatch, got {keys1} and {keys2}")
            d = {}
            for key, item1 in self.items():
                d[key] = item1 >= other.get(key)
            return TensorDict(source=d, batch_size=self.batch_size, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return TensorDict(
                {key: value >= other for key, value in self.items()},
                self.batch_size,
                device=self.device,
            )
        return False

    def __gt__(self, other: object) -> T | bool:
        if is_tensorclass(other):
            return other < self
        if isinstance(other, (dict,)):
            other = self.from_dict_instance(other)
        if _is_tensor_collection(other.__class__):
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                keys1 = sorted(
                    keys1,
                    key=lambda key: "".join(key) if isinstance(key, tuple) else key,
                )
                keys2 = sorted(
                    keys2,
                    key=lambda key: "".join(key) if isinstance(key, tuple) else key,
                )
                raise KeyError(f"keys in tensordicts mismatch, got {keys1} and {keys2}")
            d = {}
            for key, item1 in self.items():
                d[key] = item1 > other.get(key)
            return TensorDict(source=d, batch_size=self.batch_size, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return TensorDict(
                {key: value > other for key, value in self.items()},
                self.batch_size,
                device=self.device,
            )
        return False

    def __le__(self, other: object) -> T | bool:
        if is_tensorclass(other):
            return other >= self
        if isinstance(other, (dict,)):
            other = self.from_dict_instance(other)
        if _is_tensor_collection(other.__class__):
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                keys1 = sorted(
                    keys1,
                    key=lambda key: "".join(key) if isinstance(key, tuple) else key,
                )
                keys2 = sorted(
                    keys2,
                    key=lambda key: "".join(key) if isinstance(key, tuple) else key,
                )
                raise KeyError(f"keys in tensordicts mismatch, got {keys1} and {keys2}")
            d = {}
            for key, item1 in self.items():
                d[key] = item1 <= other.get(key)
            return TensorDict(source=d, batch_size=self.batch_size, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return TensorDict(
                {key: value <= other for key, value in self.items()},
                self.batch_size,
                device=self.device,
            )
        return False

    def __lt__(self, other: object) -> T | bool:
        if is_tensorclass(other):
            return other > self
        if isinstance(other, (dict,)):
            other = self.from_dict_instance(other)
        if _is_tensor_collection(other.__class__):
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                keys1 = sorted(
                    keys1,
                    key=lambda key: "".join(key) if isinstance(key, tuple) else key,
                )
                keys2 = sorted(
                    keys2,
                    key=lambda key: "".join(key) if isinstance(key, tuple) else key,
                )
                raise KeyError(f"keys in tensordicts mismatch, got {keys1} and {keys2}")
            d = {}
            for key, item1 in self.items():
                d[key] = item1 < other.get(key)
            return TensorDict(source=d, batch_size=self.batch_size, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return TensorDict(
                {key: value < other for key, value in self.items()},
                self.batch_size,
                device=self.device,
            )
        return False

    def __setitem__(
        self,
        index: IndexType,
        value: T | dict | numbers.Number | CompatibleType,
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
                    if isinstance(self, _SubTensorDict)
                    else False,
                    validated=False,
                    non_blocking=False,
                )
                return

        # we must use any and because using Ellipsis in index can break with some indices
        if index is Ellipsis or (
            isinstance(index, tuple) and any(idx is Ellipsis for idx in index)
        ):
            index = convert_ellipsis_to_idx(index, self.batch_size)

        if isinstance(value, (TensorDictBase, dict)):
            indexed_bs = _getitem_batch_size(self.batch_size, index)
            if isinstance(value, dict):
                value = self.from_dict_instance(value, batch_size=indexed_bs)
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
            if any(key not in keys for key in value.keys()):
                subtd = self._get_sub_tensordict(index)
            for key, item in value.items():
                if key in keys:
                    self._set_at_str(
                        key, item, index, validated=False, non_blocking=False
                    )
                else:
                    subtd.set(key, item, inplace=True, non_blocking=False)
        else:
            for key in self.keys():
                self.set_at_(key, value, index)

    def all(self, dim: int = None) -> bool | TensorDictBase:
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

    def _cast_reduction(
        self,
        *,
        reduction_name,
        dim=NO_DEFAULT,
        keepdim=NO_DEFAULT,
        tuple_ok=True,
        **kwargs,
    ):
        def proc_dim(dim, tuple_ok=True):
            if dim is None:
                return dim
            if isinstance(dim, tuple):
                if tuple_ok:
                    return tuple(_d for d in dim for _d in proc_dim(d, tuple_ok=False))
                return dim
            if dim >= self.batch_dims or dim < -self.batch_dims:
                raise RuntimeError(
                    "dim must be greater than or equal to -tensordict.batch_dims and "
                    "smaller than tensordict.batch_dims"
                )
            if dim < 0:
                return (self.batch_dims + dim,)
            return (dim,)

        if dim is not NO_DEFAULT:
            dim = proc_dim(dim, tuple_ok=tuple_ok)
            if not tuple_ok:
                dim = dim[0]
        if dim is not NO_DEFAULT or keepdim:
            names = None
            if self._has_names():
                names = copy(self.names)
                if not keepdim and isinstance(dim, tuple):
                    names = [name for i, name in enumerate(names) if i not in dim]
                else:
                    names = [name for i, name in enumerate(names) if i != dim]
            if dim is not NO_DEFAULT:
                kwargs["dim"] = dim
            if keepdim is not NO_DEFAULT:
                kwargs["keepdim"] = keepdim

            def reduction(val):
                result = getattr(val, reduction_name)(
                    **kwargs,
                )
                return result

            if dim not in (None, NO_DEFAULT):
                if not keepdim:
                    if isinstance(dim, tuple):
                        batch_size = [
                            b for i, b in enumerate(self.batch_size) if i not in dim
                        ]
                    else:
                        batch_size = [
                            b for i, b in enumerate(self.batch_size) if i != dim
                        ]
                else:
                    if isinstance(dim, tuple):
                        batch_size = [
                            b if i not in dim else 1
                            for i, b in enumerate(self.batch_size)
                        ]
                    else:
                        batch_size = [
                            b if i != dim else 1 for i, b in enumerate(self.batch_size)
                        ]

            else:
                batch_size = [1 for b in self.batch_size]

            return self._fast_apply(
                reduction,
                call_on_nested=True,
                batch_size=torch.Size(batch_size),
                device=self.device,
                names=names,
            )

        def reduction(val):
            return getattr(val, reduction_name)(**kwargs)

        return self._fast_apply(
            reduction,
            call_on_nested=True,
            batch_size=torch.Size([]),
            device=self.device,
            names=None,
        )

    def _apply_nest(
        self,
        fn: Callable,
        *others: T,
        batch_size: Sequence[int] | None = None,
        device: torch.device | None = NO_DEFAULT,
        names: Sequence[str] | None = None,
        inplace: bool = False,
        checked: bool = False,
        call_on_nested: bool = False,
        default: Any = NO_DEFAULT,
        named: bool = False,
        nested_keys: bool = False,
        prefix: tuple = (),
        filter_empty: bool | None = None,
        is_leaf: Callable = None,
        **constructor_kwargs,
    ) -> T | None:
        if inplace:
            result = self
            is_locked = result.is_locked
        elif batch_size is not None:

            def make_result():
                return TensorDict(
                    {},
                    batch_size=torch.Size(batch_size),
                    names=names,
                    device=self.device if device is NO_DEFAULT else device,
                    _run_checks=False,
                    **constructor_kwargs,
                )

            result = None
            is_locked = False
        else:

            def make_result():
                return TensorDict(
                    {},
                    batch_size=self.batch_size,
                    device=self.device if device is NO_DEFAULT else device,
                    names=self.names if self._has_names() else None,
                    _run_checks=False,
                    **constructor_kwargs,
                )

            result = None
            is_locked = False

        any_set = False
        if is_leaf is None:
            is_leaf = _default_is_leaf

        for key, item in self.items():
            if (
                not call_on_nested
                and not is_leaf(item.__class__)
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
                    **constructor_kwargs,
                )
            else:
                _others = [_other._get_str(key, default=default) for _other in others]
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
                if isinstance(self, _SubTensorDict):
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
            warn(
                "Your resulting tensordict has no leaves but you did not specify filter_empty=False. "
                "Currently, this returns an empty tree (filter_empty=True), but from v0.5 it will return "
                "a None unless filter_empty=False. "
                "To silence this warning, set filter_empty to the desired value in your call to `apply`.",
                category=DeprecationWarning,
            )
        if result is None:
            result = make_result()

        if not inplace and is_locked:
            result.lock_()
        return result

    # Functorch compatibility
    @cache  # noqa: B019
    def _add_batch_dim(self, *, in_dim, vmap_level):
        td = self

        def _add_batch_dim_wrapper(key, value):
            if is_tensor_collection(value):
                return value._add_batch_dim(in_dim=in_dim, vmap_level=vmap_level)

            if isinstance(
                value, (_BatchedUninitializedParameter, _BatchedUninitializedBuffer)
            ):
                value.in_dim = in_dim
                value.vmap_level = vmap_level
                return value
            return _add_batch_dim(value, in_dim, vmap_level)

        out = TensorDict(
            {key: _add_batch_dim_wrapper(key, value) for key, value in td.items()},
            batch_size=torch.Size(
                [b for i, b in enumerate(td.batch_size) if i != in_dim]
            ),
            names=[name for i, name in enumerate(td.names) if i != in_dim],
            _run_checks=False,
            lock=self.is_locked,
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
            lock=self.is_locked,
        )
        return out

    def _convert_to_tensordict(self, dict_value: dict[str, Any]) -> T:
        return TensorDict(
            dict_value,
            batch_size=self.batch_size,
            device=self.device,
        )

    def _index_tensordict(
        self,
        index: IndexType,
        new_batch_size: torch.Size | None = None,
        names: List[str] | None = None,
    ) -> T:
        batch_size = self.batch_size
        batch_dims = len(batch_size)
        if (
            not batch_size
            and index is not None
            and (not isinstance(index, tuple) or any(idx is not None for idx in index))
        ):
            raise RuntimeError(
                f"indexing a tensordict with td.batch_dims==0 is not permitted. Got index {index}."
            )
        if new_batch_size is not None:
            batch_size = new_batch_size
        else:
            batch_size = _getitem_batch_size(batch_size, index)

        if names is None:
            names = self._get_names_idx(index)

        source = {}
        for key, item in self.items():
            if isinstance(item, TensorDict):
                # this is the simplest case, we can pre-compute the batch size easily
                new_batch_size = batch_size + item.batch_size[batch_dims:]
                source[key] = item._index_tensordict(
                    index, new_batch_size=new_batch_size
                )
            else:
                source[key] = _get_item(item, index)
        result = TensorDict(
            source=source,
            batch_size=batch_size,
            device=self.device,
            names=names,
            _run_checks=False,
            # lock=self.is_locked,
        )
        if self._is_memmap and _index_preserve_data_ptr(index):
            result._is_memmap = True
            result.lock_()
        elif self._is_shared and _index_preserve_data_ptr(index):
            result._is_shared = True
            result.lock_()
        return result

    def expand(self, *args, **kwargs) -> T:
        tensordict_dims = self.batch_dims
        shape = _get_shape_from_args(*args, **kwargs)

        # new shape dim check
        if len(shape) < len(self.shape):
            raise RuntimeError(
                f"the number of sizes provided ({len(shape)}) must be greater or equal to the number of "
                f"dimensions in the TensorDict ({tensordict_dims})"
            )

        # new shape compatability check
        for old_dim, new_dim in zip(self.batch_size, shape[-tensordict_dims:]):
            if old_dim != 1 and new_dim != old_dim:
                raise RuntimeError(
                    "Incompatible expanded shape: The expanded shape length at non-singleton dimension should be same "
                    f"as the original length. target_shape = {shape}, existing_shape = {self.batch_size}"
                )

        def _expand(tensor):
            tensor_shape = tensor.shape
            tensor_dims = len(tensor_shape)
            last_n_dims = tensor_dims - tensordict_dims
            if last_n_dims > 0:
                new_shape = (*shape, *tensor_shape[-last_n_dims:])
            else:
                new_shape = shape
            return tensor.expand(new_shape)

        names = [None] * (len(shape) - tensordict_dims) + self.names
        return self._fast_apply(
            _expand,
            batch_size=shape,
            call_on_nested=True,
            names=names,
            propagate_lock=True,
        )

    def _unbind(self, dim: int):
        batch_size = torch.Size([s for i, s in enumerate(self.batch_size) if i != dim])
        names = None
        if self._has_names():
            names = copy(self.names)
            names = [name for i, name in enumerate(names) if i != dim]
        device = self.device

        is_shared = self._is_shared
        is_memmap = self._is_memmap

        def empty():
            result = TensorDict(
                {}, batch_size=batch_size, names=names, _run_checks=False, device=device
            )
            result._is_shared = is_shared
            result._is_memmap = is_memmap
            return result

        tds = tuple(empty() for _ in range(self.batch_size[dim]))

        def unbind(key, val, tds=tds):
            unbound = (
                val.unbind(dim)
                if not isinstance(val, TensorDictBase)
                # tensorclass is also unbound using plain unbind
                else val._unbind(dim)
            )
            for td, _val in zip(tds, unbound):
                td._set_str(
                    key, _val, validated=True, inplace=False, non_blocking=False
                )

        for key, val in self.items():
            unbind(key, val)
        return tds

    def split(self, split_size: int | list[int], dim: int = 0) -> list[TensorDictBase]:
        # we must use slices to keep the storage of the tensors
        WRONG_TYPE = "split(): argument 'split_size' must be int or list of ints"
        batch_size = self.batch_size
        batch_sizes = []
        batch_dims = len(batch_size)
        if dim < 0:
            dim = len(batch_size) + dim
        if dim >= batch_dims or dim < 0:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [-{self.batch_dims}, {self.batch_dims - 1}], but got {dim})"
            )
        max_size = batch_size[dim]
        if isinstance(split_size, int):
            idx0 = 0
            idx1 = min(max_size, split_size)
            split_sizes = [slice(idx0, idx1)]
            batch_sizes.append(
                torch.Size(
                    tuple(
                        d if i != dim else idx1 - idx0 for i, d in enumerate(batch_size)
                    )
                )
            )
            while idx1 < max_size:
                idx0 = idx1
                idx1 = min(max_size, idx1 + split_size)
                split_sizes.append(slice(idx0, idx1))
                batch_sizes.append(
                    torch.Size(
                        tuple(
                            d if i != dim else idx1 - idx0
                            for i, d in enumerate(batch_size)
                        )
                    )
                )
        elif isinstance(split_size, (list, tuple)):
            if len(split_size) == 0:
                raise RuntimeError("Insufficient number of elements in split_size.")
            try:
                idx0 = 0
                idx1 = split_size[0]
                split_sizes = [slice(idx0, idx1)]
                batch_sizes.append(
                    torch.Size(
                        tuple(
                            d if i != dim else idx1 - idx0
                            for i, d in enumerate(batch_size)
                        )
                    )
                )
                for idx in split_size[1:]:
                    idx0 = idx1
                    idx1 = min(max_size, idx1 + idx)
                    split_sizes.append(slice(idx0, idx1))
                    batch_sizes.append(
                        torch.Size(
                            tuple(
                                d if i != dim else idx1 - idx0
                                for i, d in enumerate(batch_size)
                            )
                        )
                    )
            except TypeError:
                raise TypeError(WRONG_TYPE)

            if idx1 < batch_size[dim]:
                raise RuntimeError(
                    f"Split method expects split_size to sum exactly to {self.batch_size[dim]} (tensor's size at dimension {dim}), but got split_size={split_size}"
                )
        else:
            raise TypeError(WRONG_TYPE)
        index = (slice(None),) * dim
        names = self.names
        return tuple(
            self._index_tensordict(index + (ss,), new_batch_size=bs, names=names)
            for ss, bs in zip(split_sizes, batch_sizes)
        )

    def masked_select(self, mask: Tensor) -> T:
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

    def _view(
        self,
        *args,
        **kwargs,
    ) -> T:
        shape = _get_shape_from_args(*args, **kwargs)
        if any(dim < 0 for dim in shape):
            shape = infer_size_impl(shape, self.numel())
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

    def reshape(
        self,
        *args,
        **kwargs,
    ) -> T:
        shape = _get_shape_from_args(*args, **kwargs)
        if any(dim < 0 for dim in shape):
            shape = infer_size_impl(shape, self.numel())
            shape = torch.Size(shape)
        if torch.Size(shape) == self.shape:
            return self
        batch_dims = self.batch_dims

        def _reshape(tensor):
            return tensor.reshape((*shape, *tensor.shape[batch_dims:]))

        return self._fast_apply(
            _reshape, batch_size=shape, call_on_nested=True, propagate_lock=True
        )

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

    def _permute(self, *args, **kwargs):
        dims_list = _get_shape_from_args(*args, kwarg_name="dims", **kwargs)
        dims_list = [dim if dim >= 0 else self.ndim + dim for dim in dims_list]
        if any(dim < 0 or dim >= self.ndim for dim in dims_list):
            raise ValueError(
                "Received an permutation order incompatible with the tensordict shape."
            )
        # note: to allow this to work recursively, we must allow permutation order with fewer elements than dims,
        # as long as this list is complete.
        if not np.array_equal(sorted(dims_list), range(len(dims_list))):
            raise ValueError(
                f"Cannot compute the permutation, got dims={dims_list} but expected a permutation of {list(range(len(dims_list)))}."
            )
        if not len(dims_list) and not self.batch_dims:
            return self
        if np.array_equal(dims_list, range(len(dims_list))):
            return self

        def _permute(tensor):
            return tensor.permute(*dims_list, *range(len(dims_list), tensor.ndim))

        batch_size = self.batch_size
        batch_size = [batch_size[p] for p in dims_list] + list(
            batch_size[len(dims_list) :]
        )
        if self._has_names():
            names = self.names
            names = [names[i] for i in dims_list]
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
            names = list(self.names)
            batch_size, names = zip(
                *[(size, name) for size, name in zip(batch_size, names) if size != 1]
            )
            batch_size = torch.Size(batch_size)
            if batch_size == self.batch_size:
                return self

            # we only want to squeeze dimensions lower than the batch dim, and view
            # is the perfect op for this
            def _squeeze(tensor):
                return tensor.view(*batch_size, *tensor.shape[self.batch_dims :])

            return self._fast_apply(
                _squeeze,
                batch_size=batch_size,
                names=names,
                inplace=False,
                call_on_nested=True,
                propagate_lock=True,
            )
        # make the dim positive
        if dim < 0:
            newdim = self.batch_dims + dim
        else:
            newdim = dim

        if (newdim >= self.batch_dims) or (newdim < 0):
            raise RuntimeError(
                f"squeezing is allowed for dims comprised between "
                f"`-td.batch_dims` and `td.batch_dims - 1` only. Got "
                f"dim={dim} with a batch size of {self.batch_size}."
            )
        if batch_size[dim] != 1:
            return self
        batch_size = list(batch_size)
        batch_size.pop(dim)
        batch_size = list(batch_size)
        names = list(self.names)
        names.pop(dim)

        result = self._fast_apply(
            lambda x: x.squeeze(newdim),
            batch_size=batch_size,
            names=names,
            inplace=False,
            call_on_nested=True,
            propagate_lock=True,
        )
        self._maybe_set_shared_attributes(result)
        return result

    def _unsqueeze(self, dim):
        # make the dim positive
        if dim < 0:
            newdim = self.batch_dims + dim + 1
        else:
            newdim = dim

        if (newdim > self.batch_dims) or (newdim < 0):
            raise RuntimeError(
                f"unsqueezing is allowed for dims comprised between "
                f"`-td.batch_dims - 1` and `td.batch_dims` only. Got "
                f"dim={dim} with a batch size of {self.batch_size}."
            )
        batch_size = list(self.batch_size)
        batch_size.insert(newdim, 1)
        batch_size = torch.Size(batch_size)

        names = copy(self.names)
        names.insert(newdim, None)

        def _unsqueeze(tensor):
            return tensor.unsqueeze(newdim)

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
    def from_dict(cls, input_dict, batch_size=None, device=None, batch_dims=None):
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

        batch_size_set = torch.Size(()) if batch_size is None else batch_size
        input_dict = copy(input_dict)
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

    def from_dict_instance(
        self, input_dict, batch_size=None, device=None, batch_dims=None
    ):
        if batch_dims is not None and batch_size is not None:
            raise ValueError(
                "Cannot pass both batch_size and batch_dims to `from_dict`."
            )
        from tensordict import TensorDict

        batch_size_set = torch.Size(()) if batch_size is None else batch_size
        input_dict = copy(input_dict)
        for key, value in list(input_dict.items()):
            if isinstance(value, (dict,)):
                cur_value = self.get(key, None)
                if cur_value is not None:
                    input_dict[key] = cur_value.from_dict_instance(
                        value, batch_size=[], device=device, batch_dims=None
                    )
                    continue
                # we don't know if another tensor of smaller size is coming
                # so we can't be sure that the batch-size will still be valid later
                input_dict[key] = TensorDict.from_dict(
                    value, batch_size=[], device=device, batch_dims=None
                )
        out = TensorDict.from_dict(
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
        source: T | dict,
        batch_size: Sequence[int] | torch.Size | int | None = None,
    ) -> torch.Size:
        try:
            return torch.Size(batch_size)
        except Exception:
            if batch_size is None:
                return torch.Size([])
            elif isinstance(batch_size, Number):
                return torch.Size([batch_size])
            elif isinstance(source, TensorDictBase):
                return source.batch_size
            raise ValueError(
                "batch size was not specified when creating the TensorDict "
                "instance and it could not be retrieved from source."
            )

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

    def _get_names_idx(self, idx):
        if not self._has_names():
            names = None
        else:

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
        return names

    @names.setter
    def names(self, value):
        # we don't run checks on types for efficiency purposes
        if value is None:
            self._rename_subtds(value)
            self._erase_names()
            return
        value = list(value)
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
                if _is_tensor_collection(type(item)):
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

    def _check_device(self) -> None:
        devices = {value.device for value in self.values()}
        if self.device is not None and len(devices) >= 1 and devices != {self.device}:
            raise RuntimeError(
                f"TensorDict.device is {self._device}, but elements have "
                f"device values {devices}. If TensorDict.device is set then "
                "all elements must share that device."
            )

    @lock_blocked
    def popitem(self) -> Tuple[NestedKey, CompatibleType]:
        return self._tensordict.popitem()

    def pin_memory(self) -> T:
        def pin_mem(tensor):
            return tensor.pin_memory()

        return self._fast_apply(pin_mem, propagate_lock=True)

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
        if inplace is not False:
            best_attempt = inplace is BEST_ATTEMPT_INPLACE
            inplace = self._convert_inplace(inplace, key)
        if not validated:
            value = self._validate_value(value, check_shape=True)
        if not inplace:
            if self._is_locked and not ignore_lock:
                raise RuntimeError(_LOCK_ERROR)
            self._tensordict[key] = value
        else:
            try:
                dest = self._get_str(key, default=NO_DEFAULT)
                if best_attempt and _is_tensor_collection(dest.__class__):
                    dest.update(value, inplace=True, non_blocking=non_blocking)
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
            except KeyError as err:
                raise err
            except Exception as err:
                raise ValueError(
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
    ) -> T:
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
        elif not _is_tensor_collection(td.__class__):
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

    def _set_at_str(self, key, value, idx, *, validated, non_blocking: bool):
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
            tensor_in.copy_(value, non_blocking=non_blocking)
        else:
            tensor_out = _set_item(
                tensor_in, idx, value, validated=validated, non_blocking=non_blocking
            )
            if tensor_in is not tensor_out:
                if self._is_shared or self._is_memmap:
                    raise RuntimeError(
                        "You're attempting to update a leaf in-place with a shared "
                        "tensordict, but the new value does not match the previous. "
                        "If you're using NonTensorData, see the class documentation "
                        "to see how to properly pre-allocate memory in shared contexts."
                    )
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

    @lock_blocked
    def del_(self, key: NestedKey) -> T:
        key = _unravel_key_to_tuple(key)
        if len(key) > 1:
            td, subkey = _get_leaf_tensordict(self, key)
            td.del_(subkey)
            return self

        del self._tensordict[key[0]]
        return self

    @lock_blocked
    def rename_key_(
        self, old_key: NestedKey, new_key: NestedKey, safe: bool = False
    ) -> T:
        # these checks are not perfect, tuples that are not tuples of strings or empty
        # tuples could go through but (1) it will raise an error anyway and (2)
        # those checks are expensive when repeated often.
        if old_key == new_key:
            return self
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
            self._set_str(
                new_key,
                self.get(old_key),
                inplace=False,
                validated=True,
                non_blocking=False,
            )
        else:
            self._set_tuple(
                new_key,
                self.get(old_key),
                inplace=False,
                validated=True,
                non_blocking=False,
            )
        self.del_(old_key)
        return self

    def _stack_onto_(self, list_item: list[CompatibleType], dim: int) -> TensorDict:
        # if not isinstance(key, str):
        #     raise ValueError("_stack_onto_ expects string keys.")
        for key in self.keys():
            vals = [item._get_str(key, None) for item in list_item]
            if all(v is None for v in vals):
                continue
            dest = self._get_str(key, NO_DEFAULT)
            torch.stack(
                vals,
                dim=dim,
                out=dest,
            )
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
            vals = [td._get_str(key, NO_DEFAULT) for td in list_item]
            if all(v is None for v in vals):
                continue
            v = self._get_str(key, NO_DEFAULT)
            v_idx = v[idx]
            if v.data_ptr() != v_idx.data_ptr():
                raise IndexError(
                    f"Index {idx} is incompatible with stack(..., out=data) as the storages of the indexed tensors differ."
                )
            torch.stack(vals, dim=dim, out=v_idx)
            # raise ValueError(
            #     f"Cannot stack onto an indexed tensor with index {idx} "
            #     f"as its storage differs."
            # )
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
            return first._get_tuple(key[1:], default=default)
        except AttributeError as err:
            if "has no attribute" in str(err):
                raise ValueError(
                    f"Expected a TensorDictBase instance but got {type(first)} instead"
                    f" for key '{key[1:]}' in tensordict:\n{self}."
                )

    def share_memory_(self) -> T:
        if self.is_memmap():
            raise RuntimeError(
                "memmap and shared memory are mutually exclusive features."
            )
        if self.device is not None and self.device.type == "cuda":
            # cuda tensors are shared by default
            return self
        for value in self.values():
            if (
                isinstance(value, Tensor)
                and value.device.type == "cpu"
                or _is_tensor_collection(value.__class__)
            ):
                value.share_memory_()
        self._is_shared = True
        self.lock_()
        return self

    def detach_(self) -> T:
        for value in self.values():
            value.detach_()
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
    ) -> T:

        if prefix is not None:
            prefix = Path(prefix)
            if not prefix.exists():
                os.makedirs(prefix, exist_ok=True)
            metadata = {}
        if inplace and self._is_shared:
            raise RuntimeError(
                "memmap and shared memory are mutually exclusive features."
            )
        dest = (
            self
            if inplace
            else TensorDict(
                {},
                batch_size=self.batch_size,
                names=self.names if self._has_names() else None,
                device=torch.device("cpu"),
            )
        )

        # We must set these attributes before memmapping because we need the metadata
        # to match the tensordict content.
        if inplace:
            self._is_memmap = True
            self._is_shared = False  # since they are mutually exclusive
            self._device = torch.device("cpu")
        else:
            dest._is_memmap = True
            dest._is_shared = False  # since they are mutually exclusive

        for key, value in self.items():
            type_value = type(value)
            if _is_tensor_collection(type_value):
                dest._tensordict[key] = value._memmap_(
                    prefix=prefix / key if prefix is not None else None,
                    copy_existing=copy_existing,
                    executor=executor,
                    futures=futures,
                    inplace=inplace,
                    like=like,
                    share_non_tensor=share_non_tensor,
                )
                if prefix is not None:
                    _update_metadata(
                        metadata=metadata, key=key, value=value, is_collection=True
                    )
                continue
            else:

                if executor is None:
                    _populate_memmap(
                        dest=dest,
                        value=value,
                        key=key,
                        copy_existing=copy_existing,
                        prefix=prefix,
                        like=like,
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
    ) -> T:
        if metadata["device"] == "None":
            metadata["device"] = None
        else:
            metadata["device"] = torch.device(metadata["device"])
        metadata["shape"] = torch.Size(metadata["shape"])

        if out is None:
            result = cls(
                {},
                batch_size=metadata.pop("shape"),
                device=metadata.pop("device") if device is None else device,
            )
        else:
            result = out

        paths = set()
        for key, entry_metadata in metadata.items():
            if not isinstance(entry_metadata, dict):
                # there can be other metadata
                continue
            type_value = entry_metadata.get("type", None)
            if type_value is not None:
                paths.add(key)
                continue
            dtype = entry_metadata.get("dtype", None)
            shape = entry_metadata.get("shape", None)
            if (
                not (prefix / f"{key}.memmap").exists()
                or dtype is None
                or shape is None
            ):
                # invalid dict means
                continue
            if (
                device is None or device != torch.device("meta")
            ) and not torch._guards.active_fake_mode():
                if entry_metadata.get("is_nested", False):
                    # The shape is the shape of the shape, get the shape from it
                    shape = MemoryMappedTensor.from_filename(
                        (prefix / f"{key}.memmap").with_suffix(".shape.memmap"),
                        shape=shape,
                        dtype=torch.long,
                    )
                else:
                    shape = torch.Size(shape)
                tensor = MemoryMappedTensor.from_filename(
                    dtype=_STRDTYPE2DTYPE[dtype],
                    shape=shape,
                    filename=str(prefix / f"{key}.memmap"),
                )
                if device is not None:
                    tensor = tensor.to(device, non_blocking=True)
            else:
                tensor = torch.zeros(
                    torch.Size(shape),
                    device=device,
                    dtype=_STRDTYPE2DTYPE[dtype],
                )
            result._set_str(
                key,
                tensor,
                validated=True,
                inplace=False,
                non_blocking=False,
            )
        # iterate over folders and load them
        for path in prefix.iterdir():
            if path.is_dir() and path.parts[-1] in paths:
                key = path.parts[-1]  # path.parts[len(prefix.parts) :]
                existing_elt = result._get_str(key, default=None)
                if existing_elt is not None:
                    existing_elt.load_memmap_(path)
                else:
                    result._set_str(
                        key,
                        TensorDict.load_memmap(path, device=device, non_blocking=True),
                        inplace=False,
                        validated=False,
                    )
        result._memmap_prefix = prefix
        return result

    def _make_memmap_subtd(self, key):
        """Creates a sub-tensordict given a tuple key."""
        result = self
        for key_str in key:
            result_tmp = result._get_str(key_str, default=None)
            if result_tmp is None:
                result_tmp = result.empty()
                if result._memmap_prefix is not None:
                    result_tmp.memmap_(prefix=result._memmap_prefix / key_str)
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

    def make_memmap(
        self,
        key: NestedKey,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> MemoryMappedTensor:
        if not self.is_memmap():
            raise RuntimeError(
                "Can only make a memmap tensor within a memory-mapped tensordict."
            )

        key = unravel_key(key)
        if isinstance(key, tuple):
            last_node = self._make_memmap_subtd(key[:-1])
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

    def make_memmap_from_storage(
        self,
        key: NestedKey,
        storage: torch.UntypedStorage,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> MemoryMappedTensor:
        if not self.is_memmap():
            raise RuntimeError(
                "Can only make a memmap tensor within a memory-mapped tensordict."
            )

        key = unravel_key(key)
        if isinstance(key, tuple):
            last_node = self._make_memmap_subtd(key[:-1])
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

    def make_memmap_from_tensor(
        self, key: NestedKey, tensor: torch.Tensor, *, copy_data: bool = True
    ) -> MemoryMappedTensor:
        if not self.is_memmap():
            raise RuntimeError(
                "Can only make a memmap tensor within a memory-mapped tensordict."
            )

        key = unravel_key(key)
        if isinstance(key, tuple):
            last_node = self._make_memmap_subtd(key[:-1])
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

    def to(self, *args, **kwargs: Any) -> T:
        non_blocking = kwargs.pop("non_blocking", None)
        device, dtype, _, convert_to_format, batch_size = _parse_to(*args, **kwargs)
        result = self

        if device is not None and dtype is None and device == self.device:
            return result

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
            apply_kwargs["device"] = device if device is not None else self.device
            apply_kwargs["batch_size"] = batch_size
            result = result._fast_apply(to, propagate_lock=True, **apply_kwargs)
        elif batch_size is not None:
            result.batch_size = batch_size
        if device is not None and sub_non_blocking and not non_blocking:
            self._sync_all()
        return result

    def where(self, condition, other, *, out=None, pad=None):
        if _is_tensor_collection(other.__class__):

            def func(tensor, _other, key):
                if tensor is None:
                    if pad is not None:
                        tensor = _other
                        _other = pad
                    else:
                        raise KeyError(
                            f"Key {key} not found and no pad value provided."
                        )
                    cond = expand_as_right(~condition, tensor)
                elif _other is None:
                    if pad is not None:
                        _other = pad
                    else:
                        raise KeyError(
                            f"Key {key} not found and no pad value provided."
                        )
                    cond = expand_as_right(condition, tensor)
                else:
                    cond = expand_as_right(condition, tensor)
                return torch.where(
                    condition=cond,
                    input=tensor,
                    other=_other,
                )

            result = self.empty() if out is None else out
            other_keys = set(other.keys())
            # we turn into a list because out could be = to self!
            for key in list(self.keys()):
                tensor = self._get_str(key, default=NO_DEFAULT)
                _other = other._get_str(key, default=None)
                if _is_tensor_collection(type(tensor)):
                    _out = None if out is None else out._get_str(key, None)
                    if _other is None:
                        _other = tensor.empty()
                    val = tensor.where(
                        condition=condition, other=_other, out=_out, pad=pad
                    )
                else:
                    val = func(tensor, _other, key)
                result._set_str(
                    key, val, inplace=False, validated=True, non_blocking=False
                )
                other_keys.discard(key)
            for key in other_keys:
                tensor = None
                _other = other._get_str(key, default=NO_DEFAULT)
                if _is_tensor_collection(type(_other)):
                    try:
                        tensor = _other.empty()
                    except NotImplementedError:
                        # H5 tensordicts do not support select()
                        tensor = _other.to_tensordict().empty()
                    val = _other.where(
                        condition=~condition, other=tensor, out=None, pad=pad
                    )
                else:
                    val = func(tensor, _other, key)
                result._set_str(
                    key, val, inplace=False, validated=True, non_blocking=False
                )
            return result
        else:
            if out is None:

                def func(tensor):
                    return torch.where(
                        condition=expand_as_right(condition, tensor),
                        input=tensor,
                        other=other,
                    )

                return self._fast_apply(func, propagate_lock=True)
            else:

                def func(tensor, _out):
                    return torch.where(
                        condition=expand_as_right(condition, tensor),
                        input=tensor,
                        other=other,
                        out=_out,
                    )

                return self._fast_apply(func, out, propagate_lock=True)

    def masked_fill_(self, mask: Tensor, value: float | int | bool) -> T:
        for item in self.values():
            mask_expand = expand_as_right(mask, item)
            item.masked_fill_(mask_expand, value)
        return self

    def masked_fill(self, mask: Tensor, value: float | bool) -> T:
        td_copy = self.clone()
        return td_copy.masked_fill_(mask, value)

    def is_contiguous(self) -> bool:
        return all([value.is_contiguous() for _, value in self.items()])

    def _clone(self, recurse: bool = True) -> T:
        result = TensorDict(
            source={key: _clone_value(value, recurse) for key, value in self.items()},
            batch_size=self.batch_size,
            device=self.device,
            names=copy(self._td_dim_names),
            _run_checks=False,
        )
        # If this is uncommented, a shallow copy of a shared/memmap will be shared and locked too
        # This may be undesirable, not sure if this should be the default behaviour
        # (one usually does a copy to modify it).
        # if not recurse:
        #     self._maybe_set_shared_attributes(result)
        return result

    def contiguous(self) -> T:
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

    def empty(self, recurse=False) -> T:
        if not recurse:
            return TensorDict(
                device=self._device,
                batch_size=self._batch_size,
                source={},
                names=self._td_dim_names,
                _run_checks=False,
            )
        return super().empty(recurse=recurse)

    def _select(
        self,
        *keys: NestedKey,
        inplace: bool = False,
        strict: bool = True,
        set_shared: bool = True,
    ) -> T:
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

                val = self._get_str(key, default=None if not strict else NO_DEFAULT)
                if val is None:
                    continue
                source[key] = val
                if len(subkey):
                    if keys_to_select is None:
                        # delay creation of defaultdict
                        keys_to_select = defaultdict(list)
                    keys_to_select[key].append(subkey)

            if keys_to_select is not None:
                for key, val in keys_to_select.items():
                    source[key] = source[key]._select(
                        *val, strict=strict, inplace=inplace, set_shared=set_shared
                    )

        result = TensorDict(
            device=self.device,
            batch_size=self.batch_size,
            source=source,
            # names=self.names if self._has_names() else None,
            names=self._td_dim_names,
            _run_checks=False,
        )
        if inplace:
            self._tensordict = result._tensordict
            return self
        # If this is uncommented, a shallow copy of a shared/memmap will be shared and locked too
        # This may be undesirable, not sure if this should be the default behaviour
        # (one usually does a copy to modify it).
        # if set_shared:
        #     self._maybe_set_shared_attributes(result)
        return result

    def _exclude(
        self, *keys: NestedKey, inplace: bool = False, set_shared: bool = True
    ) -> T:
        # faster than Base.exclude
        if not len(keys):
            return self.copy() if not inplace else self
        if not inplace:
            _tensordict = copy(self._tensordict)
        else:
            _tensordict = self._tensordict
        keys_to_exclude = None
        for key in keys:
            key = unravel_key(key)
            if isinstance(key, str):
                _tensordict.pop(key, None)
            else:
                if keys_to_exclude is None:
                    # delay creation of defaultdict
                    keys_to_exclude = defaultdict(list)
                if key[0] in self._tensordict:
                    keys_to_exclude[key[0]].append(key[1:])
        if keys_to_exclude is not None:
            for key, cur_keys in keys_to_exclude.items():
                val = _tensordict.get(key, None)
                if val is not None:
                    val = val._exclude(
                        *cur_keys, inplace=inplace, set_shared=set_shared
                    )
                    if not inplace:
                        _tensordict[key] = val
        if inplace:
            return self
        result = TensorDict(
            _tensordict,
            batch_size=self.batch_size,
            device=self.device,
            names=self.names if self._has_names() else None,
            _run_checks=False,
        )
        # If this is uncommented, a shallow copy of a shared/memmap will be shared and locked too
        # This may be undesirable, not sure if this should be the default behaviour
        # (one usually does a copy to modify it).
        # if set_shared:
        #     self._maybe_set_shared_attributes(result)
        return result

    def keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
    ) -> _TensorDictKeysView:
        if not include_nested and not leaves_only:
            return self._tensordict.keys()
        else:
            return self._nested_keys(
                include_nested=include_nested, leaves_only=leaves_only, is_leaf=is_leaf
            )

    @cache  # noqa: B019
    def _nested_keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
    ) -> _TensorDictKeysView:
        return _TensorDictKeysView(
            self,
            include_nested=include_nested,
            leaves_only=leaves_only,
            is_leaf=is_leaf,
        )

    def __getstate__(self):
        result = {
            key: val
            for key, val in self.__dict__.items()
            if key
            not in ("_last_op", "_cache", "__last_op_queue", "__lock_parents_weakrefs")
        }
        return result

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)
        self._cache = None
        self.__last_op_queue = None
        self._last_op = None
        if self._is_locked:
            # this can cause avoidable overhead, as we will be locking the leaves
            # then locking their parent, and the parent of the parent, every
            # time re-locking tensordicts that have already been locked.
            # To avoid this, we should lock only at the root, but it isn't easy
            # to spot what the root is...
            self._is_locked = False
            self.lock_()

    # some custom methods for efficiency
    def items(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
    ) -> Iterator[tuple[str, CompatibleType]]:
        if not include_nested and not leaves_only:
            return self._tensordict.items()
        elif include_nested and leaves_only:
            is_leaf = _default_is_leaf if is_leaf is None else is_leaf

            def fast_iter():
                for k, val in self._tensordict.items():
                    if not is_leaf(val.__class__):
                        yield from (
                            ((k, *((_key,) if isinstance(_key, str) else _key)), _val)
                            for _key, _val in val.items(
                                include_nested=include_nested,
                                leaves_only=leaves_only,
                                is_leaf=is_leaf,
                            )
                        )
                    else:
                        yield k, val

            return fast_iter()
        else:
            return super().items(
                include_nested=include_nested, leaves_only=leaves_only, is_leaf=is_leaf
            )

    def values(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
    ) -> Iterator[tuple[str, CompatibleType]]:
        if not include_nested and not leaves_only:
            return self._tensordict.values()
        else:
            return super().values(
                include_nested=include_nested,
                leaves_only=leaves_only,
                is_leaf=is_leaf,
            )


class _SubTensorDict(TensorDictBase):
    """A TensorDict that only sees an index of the stored tensors."""

    _lazy = True
    _inplace_set = True
    _safe = False

    def __init__(
        self,
        source: T,
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

    # These attributes should never be set
    @property
    def _is_shared(self):
        return self._source._is_shared

    @property
    def _is_memmap(self):
        return self._source._is_memmap

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
            "Names of a subtensordict cannot be modified. Instantiate it as a TensorDict first."
        )

    def _has_names(self):
        return self._source._has_names()

    def _erase_names(self):
        raise RuntimeError(
            "Cannot erase names of a _SubTensorDict. Erase source TensorDict's names instead."
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

    def _preallocate(self, key: NestedKey, value: CompatibleType) -> T:
        return self._source.set(key, value)

    def _convert_inplace(self, inplace, key):
        has_key = key in self.keys()
        if inplace is not False:
            if inplace is True and not has_key:  # inplace could be None
                raise KeyError(
                    _KEY_ERROR.format(key, self.__class__.__name__, sorted(self.keys()))
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

    from_dict_instance = TensorDict.from_dict_instance

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
                    value_expand._set_str(
                        _key,
                        _expand_to_match_shape(
                            parent.batch_size, _tensor, self.batch_dims, self.device
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
    ) -> T:
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

    # @cache  # noqa: B019
    def keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
    ) -> _TensorDictKeysView:
        return self._source.keys(
            include_nested=include_nested, leaves_only=leaves_only, is_leaf=is_leaf
        )

    def entry_class(self, key: NestedKey) -> type:
        source_type = type(self._source.get(key))
        if _is_tensor_collection(source_type):
            return self.__class__
        return source_type

    def _stack_onto_(self, list_item: list[CompatibleType], dim: int) -> _SubTensorDict:
        self._source._stack_onto_at_(list_item, dim=dim, idx=self.idx)
        return self

    def to(self, *args, **kwargs: Any) -> T:
        device, dtype, non_blocking, convert_to_format, batch_size = _parse_to(
            *args, **kwargs
        )
        result = self

        if device is not None and dtype is None and device == self.device:
            return result
        return self.to_tensordict().to(*args, **kwargs)

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

    def _get_non_tensor(self, key: NestedKey, default=NO_DEFAULT):
        out = super()._get_non_tensor(key, default=default)

        if isinstance(out, _SubTensorDict) and is_non_tensor(out._source):
            return out._source
        return out

    def _get_str(self, key, default):
        if key in self.keys() and _is_tensor_collection(self.entry_class(key)):
            data = self._source._get_str(key, NO_DEFAULT)
            if is_non_tensor(data):
                return data[self.idx]
            return _SubTensorDict(data, self.idx)
        return self._source._get_at_str(key, self.idx, default=default)

    def _get_tuple(self, key, default):
        return self._source._get_at_tuple(key, self.idx, default=default)

    def update(
        self,
        input_dict_or_td: dict[str, CompatibleType] | TensorDictBase,
        clone: bool = False,
        inplace: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
        **kwargs,
    ) -> _SubTensorDict:
        if input_dict_or_td is self:
            # no op
            return self

        if getattr(self._source, "_has_exclusive_keys", False):
            raise RuntimeError(
                "Cannot use _SubTensorDict.update with a LazyStackedTensorDict that has exclusive keys."
            )
        if keys_to_update is not None:
            if len(keys_to_update) == 0:
                return self
            keys_to_update = unravel_key_list(keys_to_update)
        keys = set(self.keys(False))
        for key, value in input_dict_or_td.items():
            key = _unravel_key_to_tuple(key)
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
                            inplace=False,
                            keys_to_update=sub_keys_to_update,
                            non_blocking=non_blocking,
                        )
                        continue
                    elif isinstance(value, dict) or _is_tensor_collection(
                        value.__class__
                    ):
                        sub_keys_to_update = _prune_selected_keys(
                            keys_to_update, firstkey
                        )
                        target.update(
                            value,
                            keys_to_update=sub_keys_to_update,
                            non_blocking=non_blocking,
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

    def update_(
        self,
        input_dict: dict[str, CompatibleType] | TensorDictBase,
        clone: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
    ) -> _SubTensorDict:
        return self.update_at_(
            input_dict,
            idx=self.idx,
            discard_idx_attr=True,
            clone=clone,
            keys_to_update=keys_to_update,
            non_blocking=non_blocking,
        )

    def update_at_(
        self,
        input_dict: dict[str, CompatibleType] | TensorDictBase,
        idx: IndexType,
        *,
        discard_idx_attr: bool = False,
        clone: bool = False,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
    ) -> _SubTensorDict:
        if keys_to_update is not None:
            if len(keys_to_update) == 0:
                return self
            keys_to_update = unravel_key_list(keys_to_update)
        for key, value in input_dict.items():
            key = _unravel_key_to_tuple(key)
            firstkey, _ = key[0], key[1:]
            if keys_to_update and not any(
                firstkey == ktu if isinstance(ktu, str) else firstkey == ktu[0]
                for ktu in keys_to_update
            ):
                continue
            if not isinstance(value, tuple(_ACCEPTED_CLASSES)):
                raise TypeError(
                    f"Expected value to be one of types {_ACCEPTED_CLASSES} "
                    f"but got {type(value)}"
                )
            if clone:
                value = value.clone()
            if discard_idx_attr:
                self._source._set_at_tuple(
                    key,
                    value,
                    idx,
                    non_blocking=non_blocking,
                    validated=False,
                )
            else:
                self._set_at_tuple(
                    key, value, idx, validated=False, non_blocking=non_blocking
                )
        return self

    def get_parent_tensordict(self) -> T:
        if not isinstance(self._source, TensorDictBase):
            raise TypeError(
                f"_SubTensorDict was initialized with a source of type"
                f" {self._source.__class__.__name__}, "
                "parent tensordict not accessible"
            )
        return self._source

    @lock_blocked
    def del_(self, key: NestedKey) -> T:
        self._source = self._source.del_(key)
        return self

    @lock_blocked
    def popitem(self) -> Tuple[NestedKey, CompatibleType]:
        raise NotImplementedError(
            f"popitem not implemented for class {type(self).__name__}."
        )

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

    def is_contiguous(self) -> bool:
        return all(value.is_contiguous() for value in self.values())

    def contiguous(self) -> T:
        return TensorDict(
            batch_size=self.batch_size,
            source={key: value.contiguous() for key, value in self.items()},
            device=self.device,
            names=self.names,
            _run_checks=False,
        )

    def _select(
        self,
        *keys: NestedKey,
        inplace: bool = False,
        strict: bool = True,
        set_shared: bool = True,
    ) -> T:
        if inplace:
            raise RuntimeError("Cannot call select inplace on a lazy tensordict.")
        return self.to_tensordict()._select(
            *keys, inplace=False, strict=strict, set_shared=set_shared
        )

    def _exclude(
        self, *keys: NestedKey, inplace: bool = False, set_shared: bool = True
    ) -> T:
        if inplace:
            raise RuntimeError("Cannot call exclude inplace on a lazy tensordict.")
        return self.to_tensordict()._exclude(
            *keys, inplace=False, set_shared=set_shared
        )

    def expand(self, *args: int, inplace: bool = False) -> T:
        if len(args) == 1 and isinstance(args[0], Sequence):
            shape = tuple(args[0])
        else:
            shape = args
        return self._fast_apply(
            lambda x: x.expand((*shape, *x.shape[self.ndim :])),
            batch_size=shape,
            propagate_lock=True,
        )

    def is_shared(self) -> bool:
        return self._source.is_shared()

    def is_memmap(self) -> bool:
        return self._source.is_memmap()

    def rename_key_(
        self, old_key: NestedKey, new_key: NestedKey, safe: bool = False
    ) -> _SubTensorDict:
        self._source.rename_key_(old_key, new_key, safe=safe)
        return self

    def pin_memory(self) -> T:
        self._source.pin_memory()
        return self

    def detach_(self) -> T:
        raise RuntimeError("Detaching a sub-tensordict in-place cannot be done.")

    def where(self, condition, other, *, out=None, pad=None):
        return self.to_tensordict().where(
            condition=condition, other=other, out=out, pad=pad
        )

    def masked_fill_(self, mask: Tensor, value: float | bool) -> T:
        for key, item in self.items():
            self.set_(key, torch.full_like(item, value))
        return self

    def masked_fill(self, mask: Tensor, value: float | bool) -> T:
        td_copy = self.clone()
        return td_copy.masked_fill_(mask, value)

    def memmap_(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        num_threads: int = 0,
    ) -> T:
        raise RuntimeError(
            "Converting a sub-tensordict values to memmap cannot be done."
        )

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
    ) -> T:
        if prefix is not None:

            def save_metadata(prefix=prefix, self=self):
                prefix = Path(prefix)
                if not prefix.exists():
                    os.makedirs(prefix, exist_ok=True)
                with open(prefix / "meta.json", "w") as f:
                    json.dump(
                        {
                            "_type": str(self.__class__),
                            "index": _index_to_str(self.idx),
                        },
                        f,
                    )

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
        )
        if not inplace:
            result = _SubTensorDict(_source, idx=self.idx)
        else:
            result = self
        return result

    @classmethod
    def _load_memmap(
        cls, prefix: Path, metadata: dict, device: torch.device | None = None
    ):
        index = metadata["index"]
        return _SubTensorDict(
            TensorDict.load_memmap(prefix / "_source", device=device),
            _str_to_index(index),
        )

    def make_memmap(
        self,
        key: NestedKey,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> MemoryMappedTensor:
        raise RuntimeError(
            "Making a memory-mapped tensor after instantiation isn't currently allowed for _SubTensorDict."
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
            "Making a memory-mapped tensor after instantiation isn't currently allowed for _SubTensorDict."
            "If this feature is required, open an issue on GitHub to trigger a discussion on the topic!"
        )

    def make_memmap_from_tensor(
        self, key: NestedKey, tensor: torch.Tensor, *, copy_data: bool = True
    ) -> MemoryMappedTensor:
        raise RuntimeError(
            "Making a memory-mapped tensor after instantiation isn't currently allowed for _SubTensorDict."
            "If this feature is required, open an issue on GitHub to trigger a discussion on the topic!"
        )

    def share_memory_(self) -> T:
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
    def lock_(self) -> T:
        # we can't lock sub-tensordicts because that would mean that the
        # parent tensordict cannot be modified either.
        if not self.is_locked:
            raise RuntimeError(
                "Cannot lock a _SubTensorDict. Lock the parent tensordict instead."
            )
        return self

    @as_decorator("is_locked")
    def unlock_(self) -> T:
        if self.is_locked:
            raise RuntimeError(
                "Cannot unlock a _SubTensorDict. Unlock the parent tensordict instead."
            )
        return self

    def _remove_lock(self, lock_id):
        raise RuntimeError(
            "Cannot unlock a _SubTensorDict. Unlock the parent tensordict instead."
        )

    def _propagate_lock(self, lock_ids=None):
        raise RuntimeError(
            "Cannot lock a _SubTensorDict. Lock the parent tensordict instead."
        )

    def __del__(self):
        pass

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
                f"If you need {type(self)} to support {reduction_name}, help us by filing an issue"
                f" on github!"
            )
        return td._cast_reduction(
            reduction_name=reduction_name,
            dim=dim,
            keepdim=keepdim,
            tuple_ok=tuple_ok,
            **kwargs,
        )

    # TODO: check these implementations
    __eq__ = TensorDict.__eq__
    __ne__ = TensorDict.__ne__
    __ge__ = TensorDict.__ge__
    __gt__ = TensorDict.__gt__
    __le__ = TensorDict.__le__
    __lt__ = TensorDict.__lt__
    __setitem__ = TensorDict.__setitem__
    __xor__ = TensorDict.__xor__
    __or__ = TensorDict.__or__
    _check_device = TensorDict._check_device
    _check_is_shared = TensorDict._check_is_shared
    all = TensorDict.all
    any = TensorDict.any
    masked_select = TensorDict.masked_select
    memmap_like = TensorDict.memmap_like
    reshape = TensorDict.reshape
    split = TensorDict.split
    _to_module = TensorDict._to_module
    _unbind = TensorDict._unbind

    def _view(self, *args, **kwargs):
        raise RuntimeError(
            "Cannot call `view` on a sub-tensordict. Call `reshape` instead."
        )

    def _transpose(self, dim0, dim1):
        raise RuntimeError(
            "Cannot call `transpose` on a sub-tensordict. Make it dense before calling this method by calling `to_tensordict`."
        )

    def _permute(
        self,
        *args,
        **kwargs,
    ):
        raise RuntimeError(
            "Cannot call `permute` on a sub-tensordict. Make it dense before calling this method by calling `to_tensordict`."
        )

    def _squeeze(self, dim=None):
        raise RuntimeError(
            "Cannot call `squeeze` on a sub-tensordict. Make it dense before calling this method by calling `to_tensordict`."
        )

    def _unsqueeze(self, dim):
        raise RuntimeError(
            "Cannot call `unsqueeze` on a sub-tensordict. Make it dense before calling this method by calling `to_tensordict`."
        )

    _add_batch_dim = TensorDict._add_batch_dim

    _apply_nest = TensorDict._apply_nest
    # def _apply_nest(self, *args, **kwargs):
    #     return self.to_tensordict()._apply_nest(*args, **kwargs)
    _convert_to_tensordict = TensorDict._convert_to_tensordict

    _get_names_idx = TensorDict._get_names_idx

    def _index_tensordict(self, index, new_batch_size=None, names=None):
        # we ignore the names and new_batch_size which are only provided for
        # efficiency purposes
        return self._get_sub_tensordict(index)

    def _remove_batch_dim(self, *args, **kwargs):
        raise NotImplementedError


###########################
# Keys utils


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
        tensordict: T,
        include_nested: bool,
        leaves_only: bool,
        is_leaf: Callable[[Type], bool] = None,
    ) -> None:
        self.tensordict = tensordict
        self.include_nested = include_nested
        self.leaves_only = leaves_only
        if is_leaf is None:
            is_leaf = _default_is_leaf
        self.is_leaf = is_leaf

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
        self, tensordict: T, prefix: str | None = None
    ) -> Iterable[str] | Iterable[tuple[str, ...]]:
        for key, value in self._items(tensordict):
            full_key = self._combine_keys(prefix, key)
            cls = value.__class__
            while cls is list:
                # For lazy stacks
                value = value[0]
                cls = value.__class__
            is_leaf = self.is_leaf(cls)
            if self.include_nested and not is_leaf:
                yield from self._iter_helper(value, prefix=full_key)
            if not self.leaves_only or is_leaf:
                yield full_key

    def _combine_keys(self, prefix: tuple | None, key: NestedKey) -> tuple:
        if prefix is not None:
            return prefix + (key,)
        return (key,)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def _items(
        self, tensordict: TensorDictBase | None = None
    ) -> Iterable[tuple[NestedKey, CompatibleType]]:
        if tensordict is None:
            tensordict = self.tensordict
        if isinstance(tensordict, TensorDict) or is_tensorclass(tensordict):
            return tensordict._tensordict.items()
        from tensordict.nn import TensorDictParams

        if isinstance(tensordict, TensorDictParams):
            return tensordict._param_td.items()
        if isinstance(tensordict, KeyedJaggedTensor):
            return tuple((key, tensordict[key]) for key in tensordict.keys())
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
                        type(self.tensordict._get_str(key))
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
                    elif entry_type is KeyedJaggedTensor:
                        if len(key) > 2:
                            return False
                        return key[1] in item_root.keys()
                    # TODO: make this faster for LazyStacked without compromising regular
                    _is_tensordict = _is_tensor_collection(entry_type)
                    if _is_tensordict:
                        # # this will call _unravel_key_to_tuple many times
                        # return key[1:] in self.tensordict._get_str(key[0], NO_DEFAULT).keys(include_nested=self.include_nested)
                        # this won't call _unravel_key_to_tuple but requires to get the default which can be suboptimal
                        if len(key) >= 3:
                            leaf_td = item_root._get_tuple(key[1:-1], None)
                            if leaf_td is None or (
                                not _is_tensor_collection(leaf_td.__class__)
                                and not isinstance(leaf_td, KeyedJaggedTensor)
                            ):
                                return False
                        else:
                            leaf_td = item_root
                        return key[-1] in leaf_td.keys()
                return False
            # this is reached whenever there is more than one key but include_nested is False
            if all(isinstance(subkey, str) for subkey in key):
                raise TypeError(_NON_STR_KEY_TUPLE_ERR)

    def __repr__(self):
        include_nested = f"include_nested={self.include_nested}"
        leaves_only = f"leaves_only={self.leaves_only}"
        return f"{self.__class__.__name__}({list(self)},\n{indent(include_nested, 4*' ')},\n{indent(leaves_only, 4*' ')})"


def _set_tensor_dict(  # noqa: F811
    module_dict,
    hooks,
    module: torch.nn.Module,
    name: str,
    tensor: torch.Tensor,
    inplace: bool,
) -> None:
    """Simplified version of torch.nn.utils._named_member_accessor."""
    was_buffer = False
    out = module_dict["_parameters"].pop(name, None)  # type: ignore[assignment]
    if out is None:
        out = module_dict["_buffers"].pop(name, None)
        was_buffer = out is not None
    if out is None:
        out = module_dict.pop(name)
    if inplace:
        # swap tensor and out after updating out
        out_tmp = out.clone()
        out.data.copy_(tensor.data)
        tensor = out
        out = out_tmp

    if isinstance(tensor, torch.nn.Parameter):
        for hook in hooks:
            output = hook(module, name, tensor)
            if output is not None:
                tensor = output
        module_dict["_parameters"][name] = tensor

        if isinstance(
            tensor, (_BatchedUninitializedParameter, _BatchedUninitializedBuffer)
        ):
            module.register_forward_pre_hook(
                _add_batch_dim_pre_hook(), with_kwargs=True
            )

    elif was_buffer and isinstance(tensor, torch.Tensor):
        module_dict["_buffers"][name] = tensor
    else:
        module_dict[name] = tensor
    return out


def _index_to_str(index):
    if isinstance(index, tuple):
        return tuple(_index_to_str(elt) for elt in index)
    if isinstance(index, slice):
        return ("slice", {"start": index.start, "stop": index.stop, "step": index.step})
    if isinstance(index, range):
        return ("range", {"start": index.start, "stop": index.stop, "step": index.step})
    if isinstance(index, Tensor):
        return ("tensor", index.tolist(), str(index.device))
    return index


def _str_to_index(index):
    if isinstance(index, tuple):
        if not len(index):
            return index
        if index[0] == "slice":
            index = index[1]
            return slice(index["start"], index["stop"], index["step"])
        if index[0] == "range":
            index = index[1]
            return range(index["start"], index["stop"], index["step"])
        if index[0] == "tensor":
            index, device = index[1:]
            return torch.tensor(index, device=device)
        return tuple(_index_to_str(elt) for elt in index)
    return index


_register_tensor_class(TensorDict)
_register_tensor_class(_SubTensorDict)


def _save_metadata(data: TensorDictBase, prefix: Path, metadata=None):
    """Saves the metadata of a memmap tensordict on disk."""
    filepath = prefix / "meta.json"
    if metadata is None:
        metadata = {}
    metadata.update(
        {
            "shape": list(data.shape),
            "device": str(data.device),
            "_type": str(data.__class__),
        }
    )
    with open(filepath, "w") as json_metadata:
        json.dump(metadata, json_metadata)


# user did specify location and memmap is in wrong place, so we copy
def _populate_memmap(*, dest, value, key, copy_existing, prefix, like):
    filename = None if prefix is None else str(prefix / f"{key}.memmap")
    if value.is_nested:
        shape = value._nested_tensor_size()
        # Make the shape a memmap tensor too
        if prefix is not None:
            shape_filename = Path(filename)
            shape_filename = shape_filename.with_suffix(".shape.memmap")
            MemoryMappedTensor.from_tensor(
                shape,
                filename=shape_filename,
                copy_existing=copy_existing,
                existsok=True,
                copy_data=True,
            )
    else:
        shape = None
    memmap_tensor = MemoryMappedTensor.from_tensor(
        value.data if value.requires_grad else value,
        filename=filename,
        copy_existing=copy_existing,
        existsok=True,
        copy_data=not like,
        shape=shape,
    )
    dest._tensordict[key] = memmap_tensor
    return memmap_tensor


def _populate_empty(
    *,
    dest,
    key,
    shape,
    dtype,
    prefix,
):
    filename = None if prefix is None else str(prefix / f"{key}.memmap")
    if isinstance(shape, torch.Tensor):
        # Make the shape a memmap tensor too
        if prefix is not None:
            shape_filename = Path(filename)
            shape_filename = shape_filename.with_suffix(".shape.memmap")
            MemoryMappedTensor.from_tensor(
                shape,
                filename=shape_filename,
                existsok=True,
                copy_data=True,
            )
    memmap_tensor = MemoryMappedTensor.empty(
        shape=shape,
        dtype=dtype,
        filename=filename,
        existsok=True,
    )
    dest._tensordict[key] = memmap_tensor
    return memmap_tensor


def _populate_storage(
    *,
    dest,
    key,
    shape,
    dtype,
    prefix,
    storage,
):
    filename = None if prefix is None else str(prefix / f"{key}.memmap")
    if isinstance(shape, torch.Tensor):
        # Make the shape a memmap tensor too
        if prefix is not None:
            shape_filename = Path(filename)
            shape_filename = shape_filename.with_suffix(".shape.memmap")
            MemoryMappedTensor.from_tensor(
                shape,
                filename=shape_filename,
                existsok=True,
                copy_data=True,
            )
    memmap_tensor = MemoryMappedTensor.from_storage(
        storage=storage,
        shape=shape,
        dtype=dtype,
        filename=filename,
    )
    dest._tensordict[key] = memmap_tensor
    return memmap_tensor


def _update_metadata(*, metadata, key, value, is_collection):
    if not is_collection:
        metadata[key] = {
            "device": str(value.device),
            "shape": list(value.shape)
            if not value.is_nested
            else value._nested_tensor_size().shape,
            "dtype": str(value.dtype),
            "is_nested": value.is_nested,
        }
    else:
        metadata[key] = {
            "type": type(value).__name__,
        }

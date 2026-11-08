# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc
import builtins
import dataclasses
import enum
import multiprocessing.pool
import types
from abc import abstractmethod
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterator,
    Literal,
    OrderedDict,
    overload,
    Sequence,
    TYPE_CHECKING,
    TypeAlias,
    TypeVar,
)

import numpy as np

import torch
import torch.distributed as dist
from _typeshed import Incomplete
from streaming import MDSWriter
from tensordict._base.shape import PackLayout
from tensordict._indexing import convert_ellipsis_to_idx as convert_ellipsis_to_idx
from tensordict._nestedkey import NestedKey as NestedKey
from tensordict._ucxx import TensorDictPipe
from tensordict.memmap import MemoryMappedTensor as MemoryMappedTensor
from tensordict.tensorclass import TensorClass
from tensordict.utils import (
    _cache_while_locked as _cache_while_locked,
    _erase_cache_first as _erase_cache_first,
    _infer_size_impl as _infer_size_impl,
    _int_generator as _int_generator,
    _is_namedtuple as _is_namedtuple,
    _is_namedtuple_class as _is_namedtuple_class,
    _lock_blocked as _lock_blocked,
    _strtobool as _strtobool,
    Buffer as Buffer,
    DeviceType as DeviceType,
    IndexType as IndexType,
    KEY_SEPARATOR as KEY_SEPARATOR,
    lazy_legacy as lazy_legacy,
    set_lazy_legacy as set_lazy_legacy,
    TensorDictFuture as TensorDictFuture,
    unravel_key as unravel_key,
    unravel_key_list as unravel_key_list,
)
from torch import nn, Tensor
from typing_extensions import dataclass_transform, deprecated

class _NoDefault(enum.IntEnum):
    ZERO = 0

NO_DEFAULT: Incomplete

class _BEST_ATTEMPT_INPLACE:
    def __bool__(self) -> bool: ...

BEST_ATTEMPT_INPLACE: Incomplete
CompatibleType: TypeAlias = Tensor

T = TypeVar("T", bound="TensorCollection")
_TC = TypeVar("_TC", bound="TensorCollection | TensorClass")

if TYPE_CHECKING:
    from typing_extensions import Self
else:
    Self = Any

class TensorCollection:
    def __init__(
        self,
        *args,
        batch_size: Sequence[builtins.int] | torch.Size | builtins.int | None = None,
        device: DeviceType | None = None,
        names: Sequence[str] | None = None,
        non_blocking: builtins.bool | None = None,
        lock: builtins.bool = False,
        **kwargs,
    ) -> None: ...
    @property
    def is_meta(self) -> builtins.bool: ...
    def __bool__(self) -> builtins.bool: ...
    def __ne__(self, other: object) -> Self: ...  # type: ignore[override]
    def __xor__(self, other: TensorCollection | builtins.float): ...
    def __or__(self, other: TensorCollection | torch.Tensor) -> Self: ...
    def __eq__(self, other: object) -> Self: ...  # type: ignore[override]
    def __ge__(self, other: object) -> Self: ...
    def __gt__(self, other: object) -> Self: ...
    def __le__(self, other: object) -> Self: ...
    def __lt__(self, other: object) -> Self: ...
    def __deepcopy__(self, memodict=...): ...
    def __iter__(self) -> Iterator[TensorCollection]: ...
    def __len__(self) -> builtins.int: ...
    def __contains__(self, key: NestedKey) -> builtins.bool: ...
    @overload
    def __getitem__(self, index: str | tuple[str, ...]) -> Any: ...
    @overload
    def __getitem__(self, index: IndexType) -> Self: ...
    __getitems__ = __getitem__

    def __setitem__(self, index: IndexType, value: Any) -> None: ...
    def __delitem__(self, key: NestedKey) -> Self: ...
    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[builtins.type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Callable: ...
    def all(
        self, dim: builtins.int | None = None
    ) -> builtins.bool | TensorCollection: ...
    def any(
        self, dim: builtins.int | None = None
    ) -> builtins.bool | TensorCollection: ...
    def isfinite(self) -> Self: ...
    def isnan(self) -> Self: ...
    def isneginf(self) -> Self: ...
    def isposinf(self) -> Self: ...
    def isreal(self) -> Self: ...
    @overload
    def amin(
        self,
        dim: builtins.int | _NoDefault = ...,
        keepdim: builtins.bool = False,
    ) -> Self: ...
    @overload
    def amin(
        self,
        dim: builtins.int | _NoDefault = ...,
        keepdim: builtins.bool = False,
        *,
        reduce: builtins.bool,
    ) -> Self | torch.Tensor: ...
    @overload
    def min(
        self,
        dim: builtins.int | _NoDefault = ...,
        keepdim: builtins.bool = False,
        *,
        return_indices: builtins.bool = True,
    ) -> Self: ...
    @overload
    def min(
        self,
        dim: builtins.int | _NoDefault = ...,
        keepdim: builtins.bool = False,
        *,
        reduce: builtins.bool,
        return_indices: builtins.bool = True,
    ) -> Self | torch.Tensor: ...
    @overload
    def amax(
        self,
        dim: builtins.int | _NoDefault = ...,
        keepdim: builtins.bool = False,
    ) -> Self: ...
    @overload
    def amax(
        self,
        dim: builtins.int | _NoDefault = ...,
        keepdim: builtins.bool = False,
        *,
        reduce: builtins.bool,
    ) -> Self | torch.Tensor: ...
    @overload
    def max(
        self,
        dim: builtins.int | _NoDefault = ...,
        keepdim: builtins.bool = False,
        *,
        return_indices: builtins.bool = True,
    ) -> Self: ...
    @overload
    def max(
        self,
        dim: builtins.int | _NoDefault = ...,
        keepdim: builtins.bool = False,
        *,
        reduce: builtins.bool,
        return_indices: builtins.bool = True,
    ) -> Self | torch.Tensor: ...
    @overload
    def cummin(
        self, dim: builtins.int, *, return_indices: builtins.bool = True
    ) -> Self: ...
    @overload
    def cummin(
        self,
        dim: builtins.int,
        *,
        reduce: builtins.bool,
        return_indices: builtins.bool = True,
    ) -> Self | torch.Tensor: ...
    @overload
    def cummax(
        self, dim: builtins.int, *, return_indices: builtins.bool = True
    ) -> Self: ...
    @overload
    def cummax(
        self,
        dim: builtins.int,
        *,
        reduce: builtins.bool,
        return_indices: builtins.bool = True,
    ) -> Self | torch.Tensor: ...
    @overload
    def mean(
        self,
        dim: builtins.int | tuple[builtins.int] = ...,
        keepdim: builtins.bool = ...,
        *,
        dtype: torch.dtype | None = None,
    ) -> Self: ...
    @overload
    def mean(
        self,
        dim: builtins.int | tuple[builtins.int] = ...,
        keepdim: builtins.bool = ...,
        *,
        dtype: torch.dtype | None = None,
        reduce: builtins.bool,
    ) -> Self | torch.Tensor: ...
    @overload
    def mean(
        self,
        dim: builtins.int | tuple[builtins.int] = ...,
        keepdim: builtins.bool = ...,
        *,
        dtype: torch.dtype | None = None,
        reduce: builtins.bool | None = None,
        key_transform: Callable[[NestedKey], NestedKey] | None = None,
    ) -> Self | torch.Tensor: ...
    @overload
    def nanmean(
        self,
        dim: builtins.int | tuple[builtins.int] = ...,
        keepdim: builtins.bool = ...,
        *,
        dtype: torch.dtype | None = None,
    ) -> Self: ...
    @overload
    def nanmean(
        self,
        dim: builtins.int | tuple[builtins.int] = ...,
        keepdim: builtins.bool = ...,
        *,
        dtype: torch.dtype | None = None,
        reduce: builtins.bool,
    ) -> Self | torch.Tensor: ...
    @overload
    def prod(
        self,
        dim: builtins.int | tuple[builtins.int] = ...,
        keepdim: builtins.bool = ...,
        *,
        dtype: torch.dtype | None = None,
    ) -> Self: ...
    @overload
    def prod(
        self,
        dim: builtins.int | tuple[builtins.int] = ...,
        keepdim: builtins.bool = ...,
        *,
        dtype: torch.dtype | None = None,
        reduce: builtins.bool,
    ) -> Self | torch.Tensor: ...
    @overload
    def sum(
        self,
        dim: builtins.int | tuple[builtins.int] = ...,
        keepdim: builtins.bool = ...,
        *,
        dtype: torch.dtype | None = None,
    ) -> Self: ...
    @overload
    def sum(
        self,
        dim: builtins.int | tuple[builtins.int] = ...,
        keepdim: builtins.bool = ...,
        *,
        dtype: torch.dtype | None = None,
        reduce: builtins.bool,
    ) -> Self | torch.Tensor: ...
    @overload
    def sum(
        self,
        dim: builtins.int | tuple[builtins.int] = ...,
        keepdim: builtins.bool = ...,
        *,
        dtype: torch.dtype | None = None,
        reduce: builtins.bool | None = None,
        key_transform: Callable[[NestedKey], NestedKey] | None = None,
    ) -> Self | torch.Tensor: ...
    @overload
    def nansum(
        self,
        dim: builtins.int | tuple[builtins.int] = ...,
        keepdim: builtins.bool = ...,
        *,
        dtype: torch.dtype | None = None,
    ) -> Self: ...
    @overload
    def nansum(
        self,
        dim: builtins.int | tuple[builtins.int] = ...,
        keepdim: builtins.bool = ...,
        *,
        dtype: torch.dtype | None = None,
        reduce: builtins.bool,
    ) -> Self | torch.Tensor: ...
    @overload
    def std(
        self,
        dim: builtins.int | tuple[builtins.int] = ...,
        keepdim: builtins.bool = ...,
        *,
        correction: builtins.int = 1,
    ) -> Self: ...
    @overload
    def std(
        self,
        dim: builtins.int | tuple[builtins.int] = ...,
        keepdim: builtins.bool = ...,
        *,
        correction: builtins.int = 1,
        reduce: builtins.bool,
    ) -> Self | torch.Tensor: ...
    @overload
    def std(
        self,
        dim: builtins.int | tuple[builtins.int] = ...,
        keepdim: builtins.bool = ...,
        *,
        correction: builtins.int = 1,
        reduce: builtins.bool | None = None,
        key_transform: Callable[[NestedKey], NestedKey] | None = None,
    ) -> Self | torch.Tensor: ...
    @overload
    def var(
        self,
        dim: builtins.int | tuple[builtins.int] = ...,
        keepdim: builtins.bool = ...,
        *,
        correction: builtins.int = 1,
    ) -> Self: ...
    @overload
    def var(
        self,
        dim: builtins.int | tuple[builtins.int] = ...,
        keepdim: builtins.bool = ...,
        *,
        correction: builtins.int = 1,
        reduce: builtins.bool,
    ) -> Self | torch.Tensor: ...
    @overload
    def var(
        self,
        dim: builtins.int | tuple[builtins.int] = ...,
        keepdim: builtins.bool = ...,
        *,
        correction: builtins.int = 1,
        reduce: builtins.bool | None = None,
        key_transform: Callable[[NestedKey], NestedKey] | None = None,
    ) -> Self | torch.Tensor: ...
    @overload
    def quantile(
        self,
        q: builtins.float | torch.Tensor,
        dim: builtins.int | tuple[builtins.int] = ...,
        keepdim: builtins.bool = ...,
        *,
        interpolation: str = "linear",
    ) -> Self: ...
    @overload
    def quantile(
        self,
        q: builtins.float | torch.Tensor,
        dim: builtins.int | tuple[builtins.int] = ...,
        keepdim: builtins.bool = ...,
        *,
        interpolation: str = "linear",
        reduce: builtins.bool,
    ) -> Self | torch.Tensor: ...
    @overload
    def quantile(
        self,
        q: builtins.float | torch.Tensor,
        dim: builtins.int | tuple[builtins.int] = ...,
        keepdim: builtins.bool = ...,
        *,
        interpolation: str = "linear",
        reduce: builtins.bool | None = None,
        key_transform: Callable[[NestedKey], NestedKey] | None = None,
    ) -> Self | torch.Tensor: ...
    def auto_batch_size_(self, batch_dims: builtins.int | None = None) -> Self: ...
    def auto_device_(self) -> Self: ...
    @classmethod
    def from_dataclass(
        cls,
        dataclass,
        *,
        auto_batch_size: builtins.bool = False,
        as_tensorclass: builtins.bool = False,
    ): ...
    @classmethod
    def from_any(cls, obj, *, auto_batch_size: builtins.bool = False): ...
    @classmethod
    def from_pandas(
        cls,
        dataframe,
        *,
        auto_batch_size: builtins.bool = False,
        batch_dims: builtins.int | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
        separator: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self: ...
    @classmethod
    def from_csv(
        cls,
        path,
        *,
        auto_batch_size: builtins.bool = False,
        batch_dims: builtins.int | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
        separator: str | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,
    ) -> Self: ...
    @classmethod
    def from_parquet(
        cls,
        path,
        *,
        auto_batch_size: builtins.bool = False,
        batch_dims: builtins.int | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
        separator: str | None = None,
        dtype: torch.dtype | None = None,
        columns: list[str] | None = None,
        **kwargs,
    ) -> Self: ...
    @classmethod
    def from_json(
        cls,
        path,
        *,
        auto_batch_size: builtins.bool = False,
        batch_dims: builtins.int | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
        separator: str | None = None,
        dtype: torch.dtype | None = None,
        lines: builtins.bool = False,
        **kwargs,
    ) -> Self: ...
    @classmethod
    def from_dict(
        cls,
        input_dict,
        *,
        batch_size: torch.Size | None = None,
        device: torch.device | None = None,
        batch_dims: builtins.int | None = None,
        names: list[str] | None = None,
    ): ...
    def from_dict_instance(
        self,
        input_dict,
        batch_size: Incomplete | None = None,
        device: Incomplete | None = None,
        batch_dims: Incomplete | None = None,
        names: list[str] | None = None,
    ): ...
    @classmethod
    def from_pytree(
        cls,
        pytree,
        *,
        batch_size: torch.Size | None = None,
        auto_batch_size: builtins.bool = False,
        batch_dims: builtins.int | None = None,
    ): ...
    def to_pytree(self): ...
    @classmethod
    def from_h5(cls, filename, mode: str = "r"): ...
    @classmethod
    def from_zarr(
        cls,
        filename,
        *,
        mode: str = "r",
        auto_batch_size: builtins.bool = False,
        batch_dims: builtins.int | None = None,
        batch_size: torch.Size | None = None,
        non_tensor_as_stack: builtins.bool = False,
        allow_pickle: builtins.bool | None = None,
    ): ...
    @classmethod
    def from_schema(
        cls,
        schema: dict[str, tuple[list[builtins.int] | torch.Size, torch.dtype]],
        *,
        batch_size: Sequence[builtins.int] | torch.Size | None = None,
        storage: str | None = None,
        device: DeviceType | None = None,
        **kwargs: Any,
    ) -> Self: ...
    @classmethod
    def from_module(
        cls,
        module,
        as_module: builtins.bool = False,
        lock: builtins.bool = True,
        use_state_dict: builtins.bool = False,
    ): ...
    @classmethod
    def from_modules(
        cls,
        *modules,
        as_module: builtins.bool = False,
        lock: builtins.bool = True,
        use_state_dict: builtins.bool = False,
        lazy_stack: builtins.bool = False,
        expand_identical: builtins.bool = False,
    ): ...
    def to_module(
        self,
        module: nn.Module,
        *,
        inplace: builtins.bool | None = None,
        return_swap: builtins.bool = True,
        swap_dest: Incomplete | None = None,
        use_state_dict: builtins.bool = False,
        non_blocking: builtins.bool = False,
        preserve_module_state: builtins.bool = True,
        memo: Incomplete | None = None,
    ): ...
    @property
    def shape(self) -> torch.Size: ...
    @shape.setter
    def shape(self, value) -> torch.Size: ...
    @property
    def batch_size(self) -> torch.Size: ...
    def size(self, dim: builtins.int | None = None) -> torch.Size | builtins.int: ...
    @property
    def data(self) -> Self: ...
    @property
    def grad(self) -> Self: ...
    @grad.setter
    def grad(self, grad) -> None: ...
    def data_ptr(self, *, storage: builtins.bool = False): ...
    def zero_grad(self, set_to_none: builtins.bool = True) -> Self: ...
    @property
    def dtype(self): ...
    @property
    def batch_dims(self) -> builtins.int: ...
    def ndimension(self) -> builtins.int: ...
    @property
    def ndim(self) -> builtins.int: ...
    def dim(self) -> builtins.int: ...
    def numel(self) -> builtins.int: ...
    @property
    def depth(self) -> builtins.int: ...
    @overload
    def expand(self, *shape: builtins.int) -> Self: ...
    @overload
    def expand(self, shape: torch.Size) -> Self: ...
    def expand_as(self, other: TensorCollection | torch.Tensor) -> Self: ...
    def new_zeros(
        self,
        *size: torch.Size,
        dtype: torch.dtype = None,
        device: DeviceType = ...,
        requires_grad: builtins.bool = False,
        layout: torch.layout = ...,
        pin_memory: builtins.bool | None = None,
        empty_lazy: builtins.bool = False,
    ): ...
    def new_ones(
        self,
        *size: torch.Size,
        dtype: torch.dtype = None,
        device: DeviceType = ...,
        requires_grad: builtins.bool = False,
        layout: torch.layout = ...,
        pin_memory: builtins.bool | None = None,
        empty_lazy: builtins.bool = False,
    ): ...
    def new_empty(
        self,
        *size: torch.Size,
        dtype: torch.dtype = None,
        device: DeviceType = ...,
        requires_grad: builtins.bool = False,
        layout: torch.layout = ...,
        pin_memory: builtins.bool | None = None,
        empty_lazy: builtins.bool = False,
    ): ...
    def new_full(
        self,
        size: torch.Size,
        fill_value,
        *,
        dtype: torch.dtype = None,
        device: DeviceType = ...,
        requires_grad: builtins.bool = False,
        layout: torch.layout = ...,
        pin_memory: builtins.bool | None = None,
        empty_lazy: builtins.bool = False,
    ): ...
    def new_tensor(
        self,
        data: torch.Tensor | TensorCollection | np.ndarray | Sequence | builtins.float,
        *,
        dtype: torch.dtype = None,
        device: DeviceType = ...,
        requires_grad: builtins.bool = False,
        pin_memory: builtins.bool | None = None,
    ): ...
    def unbind(self, dim: builtins.int) -> tuple[TensorCollection, ...]: ...
    def chunk(
        self, chunks: builtins.int, dim: builtins.int = 0
    ) -> tuple[TensorCollection, ...]: ...
    def unsqueeze(self, dim: builtins.int) -> Self: ...
    def squeeze(self, dim: builtins.int | None = None) -> Self: ...
    @overload
    def reshape(self, *shape: builtins.int) -> Self: ...
    @overload
    def reshape(self, shape: list | tuple) -> Self: ...
    def repeat_interleave(
        self,
        repeats: torch.Tensor | builtins.int,
        dim: builtins.int | None = None,
        *,
        output_size: builtins.int | None = None,
    ) -> Self: ...
    def repeat(self, *repeats: builtins.int) -> Self: ...
    def cat_tensors(
        self,
        *keys: NestedKey,
        out_key: NestedKey,
        dim: builtins.int = 0,
        keep_entries: builtins.bool = False,
    ) -> Self: ...
    def stack_tensors(
        self,
        *keys: NestedKey,
        out_key: NestedKey,
        dim: builtins.int = 0,
        keep_entries: builtins.bool = False,
    ) -> Self: ...
    def cat_from_tensordict(
        self,
        dim: builtins.int = 0,
        *,
        sorted: builtins.bool | list[NestedKey] | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def pack(
        self, dim: builtins.int = -1, *, layout: PackLayout | None = None
    ) -> tuple[torch.Tensor, PackLayout]: ...
    def stack_from_tensordict(
        self,
        dim: builtins.int = 0,
        *,
        sorted: builtins.bool | list[NestedKey] | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    # The result has the type of the inputs, not of cls: a list of tensors
    # gives a tensor. The last overload keeps other inputs, such as instances
    # of a @tensorclass decorator class, typed Any rather than rejected.
    @overload
    @classmethod
    def stack(
        cls, input: Sequence[_TC], dim: builtins.int = 0, *, out: _TC | None = None
    ) -> _TC: ...
    @overload
    @classmethod
    def stack(
        cls,
        input: Sequence[Tensor],
        dim: builtins.int = 0,
        *,
        out: Tensor | None = None,
    ) -> Tensor: ...
    @overload
    @classmethod
    def stack(
        cls, input: Sequence[Any], dim: builtins.int = 0, *, out: Any = None
    ) -> Any: ...
    @overload
    @classmethod
    def cat(
        cls, input: Sequence[_TC], dim: builtins.int = 0, *, out: _TC | None = None
    ) -> _TC: ...
    @overload
    @classmethod
    def cat(
        cls,
        input: Sequence[Tensor],
        dim: builtins.int = 0,
        *,
        out: Tensor | None = None,
    ) -> Tensor: ...
    @overload
    @classmethod
    def cat(
        cls, input: Sequence[Any], dim: builtins.int = 0, *, out: Any = None
    ) -> Any: ...
    @classmethod
    def lazy_stack(
        cls, input, dim: builtins.int = 0, *, out: Incomplete | None = None, **kwargs
    ): ...
    @classmethod
    def maybe_dense_stack(
        cls, input, dim: builtins.int = 0, *, out: Incomplete | None = None, **kwargs
    ): ...
    def split(
        self, split_size: builtins.int | list[builtins.int], dim: builtins.int = 0
    ) -> tuple[TensorCollection, ...]: ...
    def gather(
        self, dim: builtins.int, index: Tensor, out: T | None = None
    ) -> Self: ...
    @overload
    def view(self, *shape: builtins.int): ...
    @overload
    def view(self, dtype) -> Self: ...
    @overload
    def view(self, shape: torch.Size): ...
    def transpose(self, dim0, dim1): ...
    def swapaxes(self, axis0: builtins.int, axis1: builtins.int): ...
    def swapdims(self, dim0: builtins.int, dim1: builtins.int): ...
    def flip(self, dims: builtins.int | tuple[builtins.int, ...]): ...
    def fliplr(self): ...
    def flipud(self): ...
    def roll(
        self,
        shifts: builtins.int | tuple[builtins.int, ...],
        dims: builtins.int | tuple[builtins.int, ...] | None = None,
    ): ...
    def rot90(
        self, k: builtins.int = 1, dims: tuple[builtins.int, builtins.int] = (0, 1)
    ): ...
    def narrow(self, dim: builtins.int, start: builtins.int, length: builtins.int): ...
    def tile(self, dims: tuple[builtins.int, ...]): ...
    def broadcast_to(self, shape: tuple[builtins.int, ...]): ...
    def atleast_1d(self): ...
    def atleast_2d(self): ...
    def atleast_3d(self): ...
    def movedim(
        self,
        source: builtins.int | tuple[builtins.int, ...],
        destination: builtins.int | tuple[builtins.int, ...],
    ): ...
    def moveaxis(
        self,
        source: builtins.int | tuple[builtins.int, ...],
        destination: builtins.int | tuple[builtins.int, ...],
    ): ...
    def pad(
        self,
        pad_size: Sequence[builtins.int],
        value: builtins.float = 0.0,
        inplace: builtins.bool = False,
        safe: builtins.bool = True,
    ) -> Self: ...
    @overload
    def permute(self, *dims: builtins.int): ...
    @overload
    def permute(self, dims: list | tuple): ...
    @property
    def names(self): ...
    def refine_names(self, *names) -> Self: ...
    def rename(self, *names, **rename_map): ...
    def rename_(self, *names, **rename_map): ...
    @property
    def device(self) -> torch.device | None: ...
    @device.setter
    def device(self, value: DeviceType) -> torch.device | None: ...
    def clear(self) -> Self: ...
    def clear_refs_for_compile_(self) -> Self: ...
    @classmethod
    def fromkeys(cls, keys: list[NestedKey], value: Any = 0): ...
    def popitem(self) -> tuple[NestedKey, CompatibleType]: ...
    def clear_device_(self) -> Self: ...
    def param_count(
        self, *, count_duplicates: builtins.bool = True
    ) -> builtins.int: ...
    def bytes(self, *, count_duplicates: builtins.bool = True) -> builtins.int: ...
    def pin_memory(
        self, num_threads: builtins.int | None = None, inplace: builtins.bool = False
    ) -> Self: ...
    def pin_memory_(self, num_threads: builtins.int = 0) -> Self: ...
    def cpu(self, **kwargs) -> Self: ...
    def cuda(
        self, device: builtins.int | str | torch.device | None = None, **kwargs
    ) -> Self: ...
    @property
    def is_cuda(self): ...
    @property
    def is_cpu(self): ...
    def state_dict(
        self,
        destination: Incomplete | None = None,
        prefix: str = "",
        keep_vars: builtins.bool = False,
        flatten: builtins.bool = True,
    ) -> OrderedDict[str, Any]: ...
    def load_state_dict(
        self,
        state_dict: OrderedDict[str, Any],
        strict: builtins.bool = True,
        assign: builtins.bool = False,
        from_flatten: builtins.bool | None = None,
    ) -> Self: ...
    def is_shared(self) -> builtins.bool: ...
    def is_memmap(self) -> builtins.bool: ...
    def share_memory_(self) -> Self: ...
    def densify(self, layout: torch.layout = ...): ...
    @property
    def saved_path(self): ...
    def consolidate(
        self,
        filename: Path | str | None = None,
        *,
        num_threads: builtins.int = 0,
        device: torch.device | None = None,
        non_blocking: builtins.bool = False,
        inplace: builtins.bool = False,
        return_early: builtins.bool = False,
        use_buffer: builtins.bool = False,
        share_memory: builtins.bool = False,
        pin_memory: builtins.bool = False,
        metadata: builtins.bool = False,
    ) -> Self: ...
    @classmethod
    def from_consolidated(cls, filename): ...
    def is_consolidated(self): ...
    def memmap_(
        self,
        prefix: str | Path | None = None,
        copy_existing: builtins.bool = False,
        *,
        num_threads: builtins.int = 0,
        return_early: builtins.bool = False,
        share_non_tensor: builtins.bool = False,
        existsok: builtins.bool = True,
        robust_key: builtins.bool | None = True,
        atomic: builtins.bool = False,
    ) -> Self: ...
    def make_memmap(
        self,
        key: NestedKey,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        robust_key: builtins.bool | None = True,
    ) -> MemoryMappedTensor: ...
    def make_memmap_from_storage(
        self,
        key: NestedKey,
        storage: torch.UntypedStorage,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        robust_key: builtins.bool | None = True,
    ) -> MemoryMappedTensor: ...
    def make_memmap_from_tensor(
        self,
        key: NestedKey,
        tensor: torch.Tensor,
        *,
        copy_data: builtins.bool = True,
        robust_key: builtins.bool | None = True,
    ) -> MemoryMappedTensor: ...
    def save(
        self,
        prefix: str | Path | None = None,
        copy_existing: builtins.bool = False,
        *,
        num_threads: builtins.int = 0,
        return_early: builtins.bool = False,
        share_non_tensor: builtins.bool = False,
        robust_key: builtins.bool | None = True,
        archive: builtins.bool | None = None,
        compression: str | builtins.int | None = None,
        atomic: builtins.bool = False,
    ) -> Self: ...
    def dumps(
        self,
        prefix: str | Path | None = None,
        copy_existing: builtins.bool = False,
        *,
        num_threads: builtins.int = 0,
        return_early: builtins.bool = False,
        share_non_tensor: builtins.bool = False,
        robust_key: builtins.bool | None = True,
        archive: builtins.bool | None = None,
        compression: str | builtins.int | None = None,
        atomic: builtins.bool = False,
    ) -> Self: ...
    def memmap(
        self,
        prefix: str | Path | None = None,
        copy_existing: builtins.bool = False,
        *,
        num_threads: builtins.int = 0,
        return_early: builtins.bool = False,
        share_non_tensor: builtins.bool = False,
        existsok: builtins.bool = True,
        robust_key: builtins.bool | None = True,
        archive: builtins.bool | None = None,
        compression: str | builtins.int | None = None,
        atomic: builtins.bool = False,
    ) -> Self: ...
    def memmap_like(
        self,
        prefix: str | Path | None = None,
        copy_existing: builtins.bool = False,
        *,
        existsok: builtins.bool = True,
        num_threads: builtins.int = 0,
        return_early: builtins.bool = False,
        share_non_tensor: builtins.bool = False,
        robust_key: builtins.bool | None = True,
        archive: builtins.bool | None = None,
    ) -> Self: ...
    @classmethod
    def load(
        cls,
        prefix: str | Path,
        *args,
        robust_key: builtins.bool | None = True,
        **kwargs,
    ) -> Self: ...
    def load_(
        self,
        prefix: str | Path,
        *args,
        robust_key: builtins.bool | None = True,
        **kwargs,
    ): ...
    @classmethod
    def load_memmap(
        cls,
        prefix: str | Path,
        device: torch.device | None = None,
        non_blocking: builtins.bool = False,
        *,
        out: TensorCollection | None = None,
        robust_key: builtins.bool | None = True,
        allow_pickle: builtins.bool | None = None,
        mode: str | None = None,
    ) -> Self: ...
    def load_memmap_(
        self,
        prefix: str | Path,
        robust_key: builtins.bool | None = True,
        *,
        allow_pickle: builtins.bool | None = None,
        mode: str | None = None,
    ): ...
    def memmap_refresh_(self, *, allow_pickle: builtins.bool | None = None): ...
    def entry_class(self, key: NestedKey) -> builtins.type: ...
    def set(
        self,
        key: NestedKey,
        item: CompatibleType,
        inplace: builtins.bool = False,
        *,
        non_blocking: builtins.bool = False,
        **kwargs: Any,
    ) -> Self: ...
    def set_non_tensor(self, key: NestedKey, value: Any): ...
    def get_non_tensor(self, key: NestedKey, default=...): ...
    def is_non_tensor(self, key: NestedKey) -> builtins.bool: ...
    def filter_non_tensor_data(self) -> Self: ...
    def filter_empty_(self): ...
    def set_at_(
        self,
        key: NestedKey,
        value: CompatibleType,
        index: IndexType,
        *,
        non_blocking: builtins.bool = False,
    ) -> Self: ...
    def set_(
        self,
        key: NestedKey,
        item: CompatibleType,
        *,
        non_blocking: builtins.bool = False,
    ) -> Self: ...
    @overload
    def get(self, key): ...
    @overload
    def get(self, key, default): ...
    @overload
    def get_at(self, key, index): ...
    @overload
    def get_at(self, key, index, default): ...
    def get_item_shape(self, key: NestedKey): ...
    def update(
        self,
        input_dict_or_td: Mapping[Any, Any] | T,
        clone: builtins.bool = False,
        inplace: builtins.bool = False,
        *,
        non_blocking: builtins.bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
        is_leaf: Callable[[builtins.type], builtins.bool] | None = None,
        update_batch_size: builtins.bool = False,
        ignore_lock: builtins.bool = False,
    ) -> Self: ...
    def update_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        clone: builtins.bool = False,
        *,
        non_blocking: builtins.bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
    ) -> Self: ...
    def update_at_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        idx: IndexType,
        clone: builtins.bool = False,
        *,
        non_blocking: builtins.bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
    ) -> Self: ...
    def replace(self, *args, **kwargs): ...
    def create_nested(self, key): ...
    def copy_(self, tensordict: T, non_blocking: builtins.bool = False) -> Self: ...
    def copy_at_(
        self,
        tensordict: T,
        idx: IndexType,
        non_blocking: builtins.bool = False,
        *,
        fast: builtins.bool = True,
    ) -> Self: ...
    def is_empty(self) -> builtins.bool: ...
    def setdefault(
        self, key: NestedKey, default: CompatibleType, inplace: builtins.bool = False
    ) -> CompatibleType: ...
    def items(
        self,
        include_nested: builtins.bool = False,
        leaves_only: builtins.bool = False,
        is_leaf: Incomplete | None = None,
        *,
        sort: builtins.bool = False,
    ) -> Iterator[tuple[str, CompatibleType]]: ...
    def non_tensor_items(self, include_nested: builtins.bool = False): ...
    def values(
        self,
        include_nested: builtins.bool = False,
        leaves_only: builtins.bool = False,
        is_leaf: Incomplete | None = None,
        *,
        sort: builtins.bool = False,
    ) -> Iterator[CompatibleType]: ...
    def keys(
        self,
        include_nested: builtins.bool = False,
        leaves_only: builtins.bool = False,
        is_leaf: Callable[[builtins.type], builtins.bool] | None = None,
        *,
        sort: builtins.bool = False,
    ): ...
    def pop(self, key: NestedKey, default: Any = ...) -> CompatibleType: ...
    @property
    def sorted_keys(self) -> list[NestedKey]: ...
    def flatten(self, start_dim: builtins.int = 0, end_dim: builtins.int = -1): ...
    def unflatten(self, dim, unflattened_size): ...
    def _transform_keys(
        self, key_transform: Callable[[NestedKey], NestedKey]
    ) -> Self: ...
    def rename_key_(
        self, old_key: NestedKey, new_key: NestedKey, safe: builtins.bool = False
    ) -> Self: ...
    def rename_keys(
        self, mapping: Mapping[NestedKey, NestedKey], *, strict: builtins.bool = True
    ) -> Self: ...
    def del_(self, key: NestedKey) -> Self: ...
    def gather_and_stack(
        self, dst: builtins.int, group: dist.ProcessGroup | None = None
    ) -> Self | None: ...
    def send(
        self,
        dst: builtins.int | None = None,
        *,
        group: dist.ProcessGroup | None = None,
        group_dst: builtins.int | None = None,
        init_tag: builtins.int = 0,
        pseudo_rand: builtins.bool = False,
        consolidated: builtins.bool = False,
    ) -> None: ...
    def recv(
        self,
        src: builtins.int | None = None,
        *,
        group: dist.ProcessGroup | None = None,
        group_src: builtins.int | None = None,
        init_tag: builtins.int = 0,
        pseudo_rand: builtins.bool = False,
        consolidated: builtins.bool = False,
    ) -> builtins.int: ...
    @classmethod
    def from_remote_init(
        cls,
        src: builtins.int,
        group: dist.ProcessGroup | None = None,
        device: torch.device | None = None,
        use_broadcast: builtins.bool = False,
    ) -> Self: ...
    def init_remote(
        self,
        dst: builtins.int | None = None,
        group: dist.ProcessGroup | None = None,
        device: torch.device | None = None,
        use_broadcast: builtins.bool = False,
    ): ...
    def isend(
        self,
        dst: builtins.int | None = None,
        *,
        group: dist.ProcessGroup | None = None,
        group_dst: builtins.int | None = None,
        init_tag: builtins.int = 0,
        pseudo_rand: builtins.bool = False,
        return_early: builtins.bool = False,
        consolidated: builtins.bool = False,
    ) -> builtins.int | list[dist.Work] | None: ...
    def irecv(
        self,
        src: builtins.int | None = None,
        *,
        group: dist.ProcessGroup | None = None,
        group_src: builtins.int | None = None,
        return_premature: builtins.bool = False,
        init_tag: builtins.int = 0,
        pseudo_rand: builtins.bool = False,
        consolidated: builtins.bool = False,
    ) -> tuple[builtins.int, list[torch.Future]] | list[torch.Future] | None: ...
    async def asend(self, dst: TensorDictPipe) -> None: ...
    async def arecv(
        self,
        src: TensorDictPipe,
        *,
        device: torch.device | str | None = None,
    ) -> TensorCollection: ...
    def reduce(
        self,
        dst,
        op: Incomplete | None = None,
        async_op: builtins.bool = False,
        return_premature: builtins.bool = False,
        group: Incomplete | None = None,
        *,
        bucket_bytes: builtins.int | None = None,
    ): ...
    def broadcast(
        self,
        src: builtins.int,
        *,
        group: dist.ProcessGroup | None = None,
        device: torch.device | str | None = None,
    ) -> Self: ...
    def all_reduce(
        self,
        op: Incomplete | None = None,
        *,
        group: dist.ProcessGroup | None = None,
        async_op: builtins.bool = False,
        bucket_bytes: builtins.int | None = None,
    ) -> None: ...
    def all_gather(
        self,
        *,
        group: dist.ProcessGroup | None = None,
    ) -> list: ...
    def scatter(
        self,
        src: builtins.int,
        tensordicts: list | None = None,
        *,
        group: dist.ProcessGroup | None = None,
        device: torch.device | str | None = None,
    ) -> Self: ...
    def apply_(self, fn: Callable, *others, **kwargs) -> Self: ...
    @overload
    def apply(
        self,
        fn: Callable[..., Tensor | TensorCollection | TensorClass],
        *others: T,
        batch_size: Sequence[builtins.int] | None = None,
        device: torch.device | None = ...,
        names: Sequence[str] | None = ...,
        inplace: builtins.bool = False,
        default: Any = ...,
        filter_empty: Literal[False] | None = None,
        propagate_lock: builtins.bool = False,
        call_on_nested: builtins.bool = False,
        is_leaf: Callable[[builtins.type], builtins.bool] | None = None,
        out: TensorCollection | None = None,
        **constructor_kwargs,
    ) -> Self: ...
    @overload
    def apply(
        self,
        fn: Callable,
        *others: T,
        batch_size: Sequence[builtins.int] | None = None,
        device: torch.device | None = ...,
        names: Sequence[str] | None = ...,
        inplace: builtins.bool = False,
        default: Any = ...,
        filter_empty: Literal[False],
        propagate_lock: builtins.bool = False,
        call_on_nested: builtins.bool = False,
        is_leaf: Callable[[builtins.type], builtins.bool] | None = None,
        out: TensorCollection | None = None,
        **constructor_kwargs,
    ) -> Self: ...
    @overload
    def apply(
        self,
        fn: Callable,
        *others: T,
        batch_size: Sequence[builtins.int] | None = None,
        device: torch.device | None = ...,
        names: Sequence[str] | None = ...,
        inplace: builtins.bool = False,
        default: Any = ...,
        filter_empty: builtins.bool | None = None,
        propagate_lock: builtins.bool = False,
        call_on_nested: builtins.bool = False,
        is_leaf: Callable[[builtins.type], builtins.bool] | None = None,
        out: TensorCollection | None = None,
        **constructor_kwargs,
    ) -> Self | None: ...
    @overload
    def named_apply(
        self,
        fn: Callable[..., Tensor | TensorCollection | TensorClass],
        *others: T,
        nested_keys: builtins.bool = False,
        batch_size: Sequence[builtins.int] | None = None,
        device: torch.device | None = ...,
        names: Sequence[str] | None = ...,
        inplace: builtins.bool = False,
        default: Any = ...,
        filter_empty: Literal[False] | None = None,
        propagate_lock: builtins.bool = False,
        call_on_nested: builtins.bool = False,
        is_leaf: Callable[[builtins.type], builtins.bool] | None = None,
        out: TensorCollection | None = None,
        **constructor_kwargs,
    ) -> Self: ...
    @overload
    def named_apply(
        self,
        fn: Callable,
        *others: T,
        nested_keys: builtins.bool = False,
        batch_size: Sequence[builtins.int] | None = None,
        device: torch.device | None = ...,
        names: Sequence[str] | None = ...,
        inplace: builtins.bool = False,
        default: Any = ...,
        filter_empty: Literal[False],
        propagate_lock: builtins.bool = False,
        call_on_nested: builtins.bool = False,
        is_leaf: Callable[[builtins.type], builtins.bool] | None = None,
        out: TensorCollection | None = None,
        **constructor_kwargs,
    ) -> Self: ...
    @overload
    def named_apply(
        self,
        fn: Callable,
        *others: T,
        nested_keys: builtins.bool = False,
        batch_size: Sequence[builtins.int] | None = None,
        device: torch.device | None = ...,
        names: Sequence[str] | None = ...,
        inplace: builtins.bool = False,
        default: Any = ...,
        filter_empty: builtins.bool | None = None,
        propagate_lock: builtins.bool = False,
        call_on_nested: builtins.bool = False,
        is_leaf: Callable[[builtins.type], builtins.bool] | None = None,
        out: TensorCollection | None = None,
        **constructor_kwargs,
    ) -> Self | None: ...
    def map(
        self,
        fn: Callable[[TensorCollection], TensorCollection | None],
        dim: builtins.int = 0,
        num_workers: builtins.int | None = None,
        *,
        out: TensorCollection | None = None,
        chunksize: builtins.int | None = None,
        num_chunks: builtins.int | None = None,
        pool: multiprocessing.pool.Pool | None = None,
        generator: torch.Generator | None = None,
        max_tasks_per_child: builtins.int | None = None,
        worker_threads: builtins.int = 1,
        index_with_generator: builtins.bool = False,
        pbar: builtins.bool = False,
        mp_start_method: str | None = None,
    ): ...
    def map_iter(
        self,
        fn: Callable[[TensorCollection], TensorCollection | None],
        dim: builtins.int = 0,
        num_workers: builtins.int | None = None,
        *,
        shuffle: builtins.bool = False,
        chunksize: builtins.int | None = None,
        num_chunks: builtins.int | None = None,
        pool: multiprocessing.pool.Pool | None = None,
        generator: torch.Generator | None = None,
        max_tasks_per_child: builtins.int | None = None,
        worker_threads: builtins.int = 1,
        index_with_generator: builtins.bool = True,
        pbar: builtins.bool = False,
        mp_start_method: str | None = None,
    ): ...
    def record_stream(self, stream: torch.cuda.Stream): ...
    def __add__(
        self, other: TensorCollection | torch.Tensor | builtins.float
    ) -> Self: ...
    def __iadd__(
        self, other: TensorCollection | torch.Tensor | builtins.float
    ) -> Self: ...
    def __radd__(
        self, other: TensorCollection | torch.Tensor | builtins.float
    ) -> Self: ...
    def __rsub__(
        self, other: TensorCollection | torch.Tensor | builtins.float
    ) -> Self: ...
    def __rmul__(
        self, other: TensorCollection | torch.Tensor | builtins.float
    ) -> Self: ...
    def __rtruediv__(
        self, other: TensorCollection | torch.Tensor | builtins.float
    ) -> Self: ...
    def __neg__(self) -> Self: ...
    def __abs__(self) -> Self: ...
    def __truediv__(
        self, other: TensorCollection | torch.Tensor | builtins.float
    ) -> Self: ...
    def __itruediv__(
        self, other: TensorCollection | torch.Tensor | builtins.float
    ) -> Self: ...
    def __mod__(
        self, other: TensorCollection | torch.Tensor | builtins.float
    ) -> Self: ...
    def __mul__(
        self, other: TensorCollection | torch.Tensor | builtins.float
    ) -> Self: ...
    def __imul__(
        self, other: TensorCollection | torch.Tensor | builtins.float
    ) -> Self: ...
    def __sub__(
        self, other: TensorCollection | torch.Tensor | builtins.float
    ) -> Self: ...
    def __isub__(
        self, other: TensorCollection | torch.Tensor | builtins.float
    ) -> Self: ...
    def __pow__(
        self, other: TensorCollection | torch.Tensor | builtins.float
    ) -> Self: ...
    def __ipow__(
        self, other: TensorCollection | torch.Tensor | builtins.float
    ) -> Self: ...
    def abs(self) -> Self: ...
    def abs_(self) -> Self: ...
    def acos(self) -> Self: ...
    def acos_(self) -> Self: ...
    def exp(self) -> Self: ...
    def exp_(self) -> Self: ...
    def neg(self) -> Self: ...
    def neg_(self) -> Self: ...
    def reciprocal(self) -> Self: ...
    def reciprocal_(self) -> Self: ...
    def sigmoid(self) -> Self: ...
    def sigmoid_(self) -> Self: ...
    def sign(self) -> Self: ...
    def sign_(self) -> Self: ...
    def sin(self) -> Self: ...
    def sin_(self) -> Self: ...
    def sinh(self) -> Self: ...
    def sinh_(self) -> Self: ...
    def tan(self) -> Self: ...
    def tan_(self) -> Self: ...
    def tanh(self) -> Self: ...
    def tanh_(self) -> Self: ...
    def trunc(self) -> Self: ...
    def trunc_(self) -> Self: ...
    def lgamma(self) -> Self: ...
    def lgamma_(self) -> Self: ...
    def frac(self) -> Self: ...
    def frac_(self) -> Self: ...
    def expm1(self) -> Self: ...
    def expm1_(self) -> Self: ...
    def log(self) -> Self: ...
    def log_(self) -> Self: ...
    def log10(self) -> Self: ...
    def log10_(self) -> Self: ...
    def log1p(self) -> Self: ...
    def log1p_(self) -> Self: ...
    def log2(self) -> Self: ...
    def log2_(self) -> Self: ...
    def ceil(self) -> Self: ...
    def ceil_(self) -> Self: ...
    def floor(self) -> Self: ...
    def floor_(self) -> Self: ...
    def round(self) -> Self: ...
    def round_(self) -> Self: ...
    def erf(self) -> Self: ...
    def erf_(self) -> Self: ...
    def erfc(self) -> Self: ...
    def erfc_(self) -> Self: ...
    def asin(self) -> Self: ...
    def asin_(self) -> Self: ...
    def atan(self) -> Self: ...
    def atan_(self) -> Self: ...
    def cos(self) -> Self: ...
    def cos_(self) -> Self: ...
    def cosh(self) -> Self: ...
    def cosh_(self) -> Self: ...
    def add(
        self,
        other: TensorCollection | torch.Tensor,
        *,
        alpha: builtins.float | None = None,
        default: str | CompatibleType | None = None,
    ) -> Self: ...
    def add_(
        self,
        other: TensorCollection | builtins.float,
        *,
        alpha: builtins.float | None = None,
    ): ...
    def lerp(
        self,
        end: TensorCollection | torch.Tensor,
        weight: TensorCollection | torch.Tensor | builtins.float,
    ): ...
    def lerp_(
        self,
        end: TensorCollection | torch.Tensor,
        weight: TensorCollection | torch.Tensor | builtins.float,
    ): ...
    def addcdiv(
        self,
        other1: TensorCollection | torch.Tensor,
        other2: TensorCollection | torch.Tensor,
        value: builtins.float | None = 1,
    ): ...
    def addcdiv_(self, other1, other2, *, value: builtins.float | None = 1): ...
    def addcmul(self, other1, other2, *, value: builtins.float | None = 1): ...
    def addcmul_(self, other1, other2, *, value: builtins.float | None = 1): ...
    def sub(
        self,
        other: TensorCollection | builtins.float,
        *,
        alpha: builtins.float | None = None,
        default: str | CompatibleType | None = None,
    ): ...
    def rsub(
        self,
        other: TensorCollection | builtins.float,
        *,
        alpha: builtins.float | None = None,
        default: str | CompatibleType | None = None,
    ): ...
    def sub_(
        self,
        other: TensorCollection | builtins.float,
        alpha: builtins.float | None = None,
    ): ...
    def mod(self, other: TensorCollection | torch.Tensor) -> Self: ...
    def mul_(self, other: TensorCollection | torch.Tensor) -> Self: ...
    def mul(
        self,
        other: TensorCollection | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> Self: ...
    def maximum_(self, other: TensorCollection | torch.Tensor) -> Self: ...
    def maximum(
        self,
        other: TensorCollection | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> Self: ...
    def minimum_(self, other: TensorCollection | torch.Tensor) -> Self: ...
    def minimum(
        self,
        other: TensorCollection | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> Self: ...
    def clamp(
        self,
        min: TensorCollection | torch.Tensor | builtins.float | None = None,
        max: TensorCollection | torch.Tensor | builtins.float | None = None,
        *,
        out=None,
    ) -> Self: ...
    def logsumexp(self, dim=None, keepdim=False, *, out=None): ...
    def clamp_max_(self, other: TensorCollection | torch.Tensor) -> Self: ...
    def clamp_max(
        self,
        other: TensorCollection | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> Self: ...
    def clamp_min_(self, other: TensorCollection | torch.Tensor) -> Self: ...
    def clamp_min(
        self,
        other: TensorCollection | torch.Tensor,
        default: str | CompatibleType | None = None,
    ) -> Self: ...
    def pow_(self, other: TensorCollection | torch.Tensor) -> Self: ...
    def pow(
        self,
        other: TensorCollection | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> Self: ...
    def div_(self, other: TensorCollection | torch.Tensor) -> Self: ...
    def div(
        self,
        other: TensorCollection | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> Self: ...
    def sqrt_(self) -> Self: ...
    def sqrt(self) -> Self: ...
    def __enter__(self): ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ): ...
    def select(
        self,
        *keys: NestedKey,
        inplace: builtins.bool = False,
        strict: builtins.bool = True,
        as_tensordict: builtins.bool = False,
    ) -> Self: ...
    def exclude(self, *keys: NestedKey, inplace: builtins.bool = False) -> Self: ...
    def to_tensordict(self, *, retain_none: builtins.bool | None = None) -> Self: ...
    def clone(self, recurse: builtins.bool = True, **kwargs) -> Self: ...
    def copy(self) -> Self: ...
    def to_padded_tensor(
        self, padding: builtins.float = 0.0, mask_key: NestedKey | None = None
    ) -> Self: ...
    def as_tensor(self) -> Self: ...
    def to_lazystack(self, dim: builtins.int = 0) -> Self: ...
    def to_mds(
        self,
        *,
        out: str | tuple[str, str],
        columns: dict[str, str] | None = None,
        writer: MDSWriter | None = None,
        finish_writer: builtins.bool | None = None,
    ) -> None: ...
    def to_dict(
        self,
        *,
        retain_none: builtins.bool = True,
        convert_tensors: builtins.bool | Literal["numpy"] = False,
        tolist_first: builtins.bool = False,
    ) -> dict[str, Any]: ...
    def to_pandas(self, *, separator: str | None = None) -> Any: ...
    def to_csv(self, path, *, separator: str | None = None, **kwargs) -> None: ...
    def to_parquet(self, path, *, separator: str | None = None, **kwargs) -> None: ...
    def to_json(
        self,
        path,
        *,
        separator: str | None = None,
        lines: builtins.bool = False,
        **kwargs,
    ) -> None: ...
    @classmethod
    def from_list(
        cls,
        input,
        *,
        auto_batch_size: builtins.bool | None = None,
        batch_size: torch.Size | None = None,
        device: torch.device | None = None,
        batch_dims: builtins.int | None = None,
        names: list[str] | None = None,
        lazy: builtins.bool | None = None,
    ) -> Self: ...
    def tolist(
        self,
        *,
        convert_nodes: builtins.bool = True,
        convert_tensors: builtins.bool | Literal["numpy"] = False,
        tolist_first: builtins.bool = False,
        as_linked_list: builtins.bool = False,
    ) -> list[Any]: ...
    def numpy(self) -> np.ndarray | dict[str, Any]: ...
    def to_namedtuple(self, dest_cls: builtins.type | None = None) -> Any: ...
    @classmethod
    def from_namedtuple(
        cls, named_tuple, *, auto_batch_size: builtins.bool = False
    ): ...
    def from_tuple(
        cls,
        obj,
        *,
        auto_batch_size: builtins.bool = False,
        batch_dims: builtins.int | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
    ): ...
    def logical_and(
        self,
        other: TensorCollection | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> Self: ...
    def bitwise_and(
        self,
        other: TensorCollection | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> Self: ...
    @classmethod
    def from_struct_array(
        cls, struct_array: np.ndarray, device: torch.device | None = None
    ) -> Self: ...
    def to_struct_array(self) -> np.ndarray: ...
    def to_h5(self, filename, **kwargs) -> Any: ...
    def to_zarr(
        self, filename, *, allow_pickle: builtins.bool | None = None, **kwargs
    ) -> Any: ...
    def to_store(
        self,
        *,
        backend: str = "redis",
        host: str = "localhost",
        port: builtins.int = 6379,
        db: builtins.int = 0,
        unix_socket_path: str | None = None,
        prefix: str = "tensordict",
        device: Incomplete | None = None,
        **kwargs,
    ) -> Any: ...
    def empty(
        self,
        recurse: builtins.bool = False,
        *,
        batch_size: Incomplete | None = None,
        device=...,
        names: Incomplete | None = None,
    ) -> Self: ...
    def zero_(self) -> Self: ...
    def fill_(self, key: NestedKey, value: builtins.float | builtins.bool) -> Self: ...
    def masked_fill_(
        self, mask: Tensor, value: builtins.float | builtins.bool
    ) -> Self: ...
    def masked_fill(
        self, mask: Tensor, value: builtins.float | builtins.bool
    ) -> Self: ...
    def where(
        self,
        condition,
        other,
        *,
        out: Incomplete | None = None,
        pad: Incomplete | None = None,
        update_batch_size: builtins.bool = False,
    ) -> Self: ...
    def masked_select(
        self, mask: Tensor, *, inplace: builtins.bool = False
    ) -> Self: ...
    def is_contiguous(self) -> builtins.bool: ...
    def contiguous(self) -> Self: ...
    def flatten_keys(
        self,
        separator: str = KEY_SEPARATOR,
        inplace: builtins.bool = False,
        is_leaf: Callable[[builtins.type], builtins.bool] | None = None,
    ) -> Self: ...
    def unflatten_keys(
        self, separator: str = KEY_SEPARATOR, inplace: builtins.bool = False
    ) -> Self: ...
    def tensor_split(
        self,
        indices_or_sections: builtins.int
        | list[builtins.int]
        | tuple[builtins.int, ...]
        | torch.Tensor,
        dim: builtins.int = 0,
    ) -> tuple[Self, ...]: ...
    def split_keys(
        self,
        *key_sets,
        inplace: builtins.bool = False,
        strict: builtins.bool = True,
        reproduce_struct: builtins.bool = False,
    ): ...
    def separates(
        self,
        *keys: NestedKey,
        default: Any = ...,
        strict: builtins.bool = True,
        filter_empty: builtins.bool = True,
    ) -> Self: ...
    def norm(
        self,
        *,
        out=None,
        dtype: torch.dtype | None = None,
        key_transform: Callable[[NestedKey], NestedKey] | None = None,
    ): ...
    def softmax(self, dim: builtins.int, dtype: torch.dtype | None = None): ...
    @property
    def is_locked(self) -> builtins.bool: ...
    @is_locked.setter
    def is_locked(self, value: builtins.bool) -> None: ...
    def lock_(self) -> Self: ...
    def unlock_(self) -> Self: ...
    @overload
    def to(
        self,
        device: DeviceType | None = ...,
        dtype: torch.dtype | None = ...,
        non_blocking: builtins.bool = ...,
        inplace: builtins.bool = False,
        *,
        cast_filter: Callable | None = ...,
    ) -> Self: ...
    @overload
    def to(
        self,
        dtype: torch.dtype,
        non_blocking: builtins.bool = ...,
        *,
        cast_filter: Callable | None = ...,
    ) -> Self: ...
    @overload
    def to(
        self,
        tensor: Tensor,
        non_blocking: builtins.bool = ...,
        *,
        cast_filter: Callable | None = ...,
    ) -> Self: ...
    @overload
    def to(
        self,
        *,
        other: T,
        non_blocking: builtins.bool = ...,
        cast_filter: Callable | None = ...,
    ) -> Self: ...
    @overload
    def to(self, *, batch_size: torch.Size) -> Self: ...
    def attrs(
        self,
        *,
        fields: Sequence[str] = ("device", "dtype", "shape"),
        num_threads: builtins.int | None = None,
    ) -> Self: ...
    def is_floating_point(self) -> builtins.bool: ...
    def double(self, *, cast_filter: Callable | None = ...) -> Self: ...
    def float(self, *, cast_filter: Callable | None = ...) -> Self: ...
    def int(self, *, cast_filter: Callable | None = ...) -> Self: ...
    def bool(self, *, cast_filter: Callable | None = ...) -> Self: ...
    def half(self, *, cast_filter: Callable | None = ...) -> Self: ...
    def type(self, dst_type, *, cast_filter: Callable | None = ...): ...
    @property
    def requires_grad(self) -> builtins.bool: ...
    def requires_grad_(self, requires_grad: builtins.bool = True) -> Self: ...
    def backward(
        self,
        gradient: TensorCollection | None = None,
        retain_graph: builtins.bool | None = None,
        create_graph: builtins.bool = False,
        inputs: TensorCollection | Sequence[Tensor] | None = None,
    ) -> None: ...
    def detach_(self) -> Self: ...
    def detach(self) -> Self: ...
    def bfloat16(self, *, cast_filter: Callable | None = ...) -> Self: ...
    def complex128(self, *, cast_filter: Callable | None = ...) -> Self: ...
    def complex32(self, *, cast_filter: Callable | None = ...) -> Self: ...
    def complex64(self, *, cast_filter: Callable | None = ...) -> Self: ...
    def float16(self, *, cast_filter: Callable | None = ...) -> Self: ...
    def float32(self, *, cast_filter: Callable | None = ...) -> Self: ...
    def float64(self, *, cast_filter: Callable | None = ...) -> Self: ...
    def int16(self, *, cast_filter: Callable | None = ...) -> Self: ...
    def int32(self, *, cast_filter: Callable | None = ...) -> Self: ...
    def int64(self, *, cast_filter: Callable | None = ...) -> Self: ...
    def int8(self, *, cast_filter: Callable | None = ...) -> Self: ...
    @deprecated(
        "TensorDictBase.qint32() is deprecated and will be removed in TensorDict 0.18. "
        "Use td.apply(lambda x: torch.quantize_per_tensor(x, scale, zero_point, "
        "torch.qint32)) instead."
    )
    def qint32(self, *, cast_filter: Callable | None = ...) -> Self: ...
    @deprecated(
        "TensorDictBase.qint8() is deprecated and will be removed in TensorDict 0.18. "
        "Use td.apply(lambda x: torch.quantize_per_tensor(x, scale, zero_point, "
        "torch.qint8)) instead."
    )
    def qint8(self, *, cast_filter: Callable | None = ...) -> Self: ...
    @deprecated(
        "TensorDictBase.quint4x2() is deprecated and will be removed in TensorDict 0.18. "
        "Use td.apply(lambda x: torch.quantize_per_tensor(x, scale, zero_point, "
        "torch.quint4x2)) instead."
    )
    def quint4x2(self, *, cast_filter: Callable | None = ...) -> Self: ...
    @deprecated(
        "TensorDictBase.quint8() is deprecated and will be removed in TensorDict 0.18. "
        "Use td.apply(lambda x: torch.quantize_per_tensor(x, scale, zero_point, "
        "torch.quint8)) instead."
    )
    def quint8(self, *, cast_filter: Callable | None = ...) -> Self: ...
    def uint16(self, *, cast_filter: Callable | None = ...) -> Self: ...
    def uint32(self, *, cast_filter: Callable | None = ...) -> Self: ...
    def uint64(self, *, cast_filter: Callable | None = ...) -> Self: ...
    def uint8(self, *, cast_filter: Callable | None = ...) -> Self: ...

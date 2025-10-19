# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import dataclasses
import enum
import types
from abc import abstractmethod
from collections.abc import MutableMapping
from pathlib import Path
from typing import (
    Any,
    Callable,
    dataclass_transform,
    Generator,
    Iterator,
    Literal,
    OrderedDict,
    overload,
    Sequence,
    Tuple,
    Type,
    TYPE_CHECKING,
    TypeVar,
)

import numpy as np

import torch
import torch.distributed as dist
from _typeshed import Incomplete
from tensordict import TensorDictBase
from tensordict._contextlib import LAST_OP_MAPS as LAST_OP_MAPS
from tensordict._nestedkey import NestedKey
from tensordict._tensorcollection import TensorCollection
from tensordict.memmap import MemoryMappedTensor as MemoryMappedTensor
from tensordict.utils import (
    Buffer as Buffer,
    cache as cache,
    convert_ellipsis_to_idx as convert_ellipsis_to_idx,
    DeviceType as DeviceType,
    erase_cache as erase_cache,
    implement_for as implement_for,
    IndexType as IndexType,
    infer_size_impl as infer_size_impl,
    int_generator as int_generator,
    is_namedtuple as is_namedtuple,
    is_namedtuple_class as is_namedtuple_class,
    lazy_legacy as lazy_legacy,
    lock_blocked as lock_blocked,
    prod as prod,
    set_lazy_legacy as set_lazy_legacy,
    strtobool as strtobool,
    TensorDictFuture as TensorDictFuture,
    unravel_key as unravel_key,
    unravel_key_list as unravel_key_list,
)
from torch import multiprocessing as mp, nn, Tensor

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any

class _NoDefault(enum.IntEnum):
    ZERO = 0

NO_DEFAULT: Incomplete

class _BEST_ATTEMPT_INPLACE:
    def __bool__(self) -> bool: ...

BEST_ATTEMPT_INPLACE: Incomplete
from typing import TypeAlias

CompatibleType: TypeAlias = Tensor

T = TypeVar("T", bound="TensorDictBase")
# Use Any for methods that don't have T in their parameters but return T
T_Any = TypeVar("T_Any", bound="TensorDictBase")

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any

class TensorClass:
    _autocast: bool = False
    _nocast: bool = False
    _frozen: bool = False

    @overload
    def __class_getitem__(cls, item: Literal["autocast"]) -> Type["TensorClass"]: ...
    @overload
    def __class_getitem__(cls, item: Literal["nocast"]) -> Type["TensorClass"]: ...
    @overload
    def __class_getitem__(cls, item: Literal["frozen"]) -> Type["TensorClass"]: ...
    @overload
    def __class_getitem__(cls, item: Literal["tensor_only"]) -> Type["TensorClass"]: ...
    @overload
    def __class_getitem__(cls, item: Literal["shadow"]) -> Type["TensorClass"]: ...
    @overload
    def __class_getitem__(
        cls,
        item: tuple[
            Literal["autocast", "nocast", "frozen", "tensor_only", "shadow"], ...
        ],
    ) -> Type["TensorClass"]: ...
    def __init__(
        self,
        *args,
        batch_size: Sequence[int] | torch.Size | int | None = None,
        device: DeviceType | None = None,
        names: Sequence[str] | None = None,
        non_blocking: bool | None = None,
        lock: bool = False,
        **kwargs,
    ) -> None: ...
    @property
    def is_meta(self) -> bool: ...
    def __bool__(self) -> bool: ...
    def __ne__(self, other: object) -> Self: ...
    def __xor__(self, other: TensorCollection | float) -> Self: ...
    def __or__(self, other: TensorCollection | torch.Tensor) -> Self: ...
    def __eq__(self, other: object) -> Self: ...
    def __ge__(self, other: object) -> Self: ...
    def __gt__(self, other: object) -> Self: ...
    def __le__(self, other: object) -> Self: ...
    def __lt__(self, other: object) -> Self: ...
    def __deepcopy__(self, memodict={}) -> Self: ...
    def __iter__(self) -> Generator: ...
    def __len__(self) -> int: ...
    def __contains__(self, key: NestedKey) -> bool: ...
    def __getitem__(
        self, index: IndexType
    ) -> Self | Tensor | TensorCollection | Any: ...
    __getitems__ = __getitem__

    def __setitem__(self, index: IndexType, value: Any) -> None: ...
    def __delitem__(self, key: NestedKey) -> Any: ...
    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Callable: ...
    def all(self, dim: int | None = None) -> bool | TensorDictBase: ...
    def any(self, dim: int | None = None) -> bool | TensorDictBase: ...
    def isfinite(self) -> Any: ...
    def isnan(self) -> Any: ...
    def isneginf(self) -> Any: ...
    def isposinf(self) -> Any: ...
    def isreal(self) -> Any: ...
    @overload
    def amin(
        self,
        dim: int | NO_DEFAULT = ...,
        keepdim: bool = False,
    ) -> Self: ...
    @overload
    def amin(
        self,
        dim: int | NO_DEFAULT = ...,
        keepdim: bool = False,
        *,
        reduce: bool,
    ) -> Self | torch.Tensor: ...
    def amin(
        self,
        dim: int | NO_DEFAULT = ...,
        keepdim: bool = False,
        *,
        reduce: bool | None = None,
    ) -> Self | torch.Tensor: ...
    @overload
    def min(
        self,
        dim: int | NO_DEFAULT = ...,
        keepdim: bool = False,
        *,
        return_indices: bool = True,
    ) -> Self: ...
    @overload
    def min(
        self,
        dim: int | NO_DEFAULT = ...,
        keepdim: bool = False,
        *,
        reduce: bool,
        return_indices: bool = True,
    ) -> Self | torch.Tensor: ...
    def min(
        self,
        dim: int | NO_DEFAULT = ...,
        keepdim: bool = False,
        *,
        reduce: bool | None = None,
        return_indices: bool = True,
    ) -> Self | torch.Tensor: ...
    @overload
    def amax(
        self,
        dim: int | NO_DEFAULT = ...,
        keepdim: bool = False,
    ) -> Self: ...
    @overload
    def amax(
        self,
        dim: int | NO_DEFAULT = ...,
        keepdim: bool = False,
        *,
        reduce: bool,
    ) -> Self | torch.Tensor: ...
    def amax(
        self,
        dim: int | NO_DEFAULT = ...,
        keepdim: bool = False,
        *,
        reduce: bool | None = None,
    ) -> Self | torch.Tensor: ...
    @overload
    def max(
        self,
        dim: int | NO_DEFAULT = ...,
        keepdim: bool = False,
        *,
        return_indices: bool = True,
    ) -> Self: ...
    @overload
    def max(
        self,
        dim: int | NO_DEFAULT = ...,
        keepdim: bool = False,
        *,
        reduce: bool,
        return_indices: bool = True,
    ) -> Self | torch.Tensor: ...
    def max(
        self,
        dim: int | NO_DEFAULT = ...,
        keepdim: bool = False,
        *,
        reduce: bool | None = None,
        return_indices: bool = True,
    ) -> Self | torch.Tensor: ...
    @overload
    def cummin(self, dim: int, *, return_indices: bool = True) -> Self: ...
    @overload
    def cummin(
        self, dim: int, *, reduce: bool, return_indices: bool = True
    ) -> Self | torch.Tensor: ...
    def cummin(
        self, dim: int, *, reduce: bool | None = None, return_indices: bool = True
    ) -> Self | torch.Tensor: ...
    @overload
    def cummax(self, dim: int, *, return_indices: bool = True) -> Self: ...
    @overload
    def cummax(
        self, dim: int, *, reduce: bool, return_indices: bool = True
    ) -> Self | torch.Tensor: ...
    def cummax(
        self, dim: int, *, reduce: bool | None = None, return_indices: bool = True
    ) -> Self | torch.Tensor: ...
    @overload
    def mean(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        dtype: torch.dtype | None = None,
    ) -> Self: ...
    @overload
    def mean(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool,
    ) -> Self | torch.Tensor: ...
    def mean(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool | None = None,
    ) -> Self | torch.Tensor: ...
    @overload
    def nanmean(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        dtype: torch.dtype | None = None,
    ) -> Self: ...
    @overload
    def nanmean(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool,
    ) -> Self | torch.Tensor: ...
    def nanmean(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool | None = None,
    ) -> Self | torch.Tensor: ...
    @overload
    def prod(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        dtype: torch.dtype | None = None,
    ) -> Self: ...
    @overload
    def prod(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool,
    ) -> Self | torch.Tensor: ...
    def prod(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool | None = None,
    ) -> Self | torch.Tensor: ...
    @overload
    def sum(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        dtype: torch.dtype | None = None,
    ) -> Self: ...
    @overload
    def sum(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool,
    ) -> Self | torch.Tensor: ...
    def sum(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool | None = None,
    ) -> Self | torch.Tensor: ...
    @overload
    def nansum(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        dtype: torch.dtype | None = None,
    ) -> Self: ...
    @overload
    def nansum(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool,
    ) -> Self | torch.Tensor: ...
    def nansum(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool | None = None,
    ) -> Self | torch.Tensor: ...
    @overload
    def std(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        correction: int = 1,
    ) -> Self: ...
    @overload
    def std(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        correction: int = 1,
        reduce: bool,
    ) -> Self | torch.Tensor: ...
    def std(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        correction: int = 1,
        reduce: bool | None = None,
    ) -> Self | torch.Tensor: ...
    @overload
    def var(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        correction: int = 1,
    ) -> Self: ...
    @overload
    def var(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        correction: int = 1,
        reduce: bool,
    ) -> Self | torch.Tensor: ...
    def var(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        correction: int = 1,
        reduce: bool | None = None,
    ) -> Self | torch.Tensor: ...
    @overload
    def quantile(
        self,
        q: float | torch.Tensor,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        interpolation: str = "linear",
    ) -> Self: ...
    @overload
    def quantile(
        self,
        q: float | torch.Tensor,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        interpolation: str = "linear",
        reduce: bool,
    ) -> Self | torch.Tensor: ...
    def quantile(
        self,
        q: float | torch.Tensor,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        interpolation: str = "linear",
        reduce: bool | None = None,
    ) -> Self | torch.Tensor: ...
    def auto_batch_size_(self, batch_dims: int | None = None) -> Self: ...
    def auto_device_(self) -> Self: ...
    @classmethod
    def from_dataclass(
        cls, dataclass, *, auto_batch_size: bool = False, as_tensorclass: bool = False
    ) -> Self: ...
    @classmethod
    def from_any(cls, obj, *, auto_batch_size: bool = False) -> Self: ...
    @classmethod
    def from_dict(
        cls,
        input_dict,
        *,
        batch_size: Sequence[int] | torch.Size | int | None = None,
        device: torch.device | None = None,
        batch_dims: int | None = None,
        names: list[str] | None = None,
    ) -> Self: ...
    def from_dict_instance(
        self,
        input_dict,
        batch_size: Incomplete | None = None,
        device: Incomplete | None = None,
        batch_dims: Incomplete | None = None,
        names: list[str] | None = None,
    ) -> Self: ...
    @classmethod
    def from_pytree(
        cls,
        pytree,
        *,
        batch_size: torch.Size | None = None,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
    ) -> Self: ...
    def to_pytree(self) -> Self: ...
    @classmethod
    def from_h5(cls, filename, mode: str = "r") -> Self: ...
    @classmethod
    def from_module(
        cls,
        module,
        as_module: bool = False,
        lock: bool = True,
        use_state_dict: bool = False,
    ) -> Self: ...
    @classmethod
    def from_modules(
        cls,
        *modules,
        as_module: bool = False,
        lock: bool = True,
        use_state_dict: bool = False,
        lazy_stack: bool = False,
        expand_identical: bool = False,
    ) -> Self: ...
    def to_module(
        self,
        module: nn.Module,
        *,
        inplace: bool | None = None,
        return_swap: bool = True,
        swap_dest: Incomplete | None = None,
        use_state_dict: bool = False,
        non_blocking: bool = False,
        memo: Incomplete | None = None,
    ) -> Self: ...
    @property
    def shape(self) -> torch.Size: ...
    @shape.setter
    def shape(self, value) -> torch.Size: ...
    @property
    def batch_size(self) -> torch.Size: ...
    @batch_size.setter
    def batch_size(self, new_size: Sequence[int] | torch.Size) -> None: ...
    def size(self, dim: int | None = None) -> torch.Size | int: ...
    @property
    def data(self) -> Self: ...
    @property
    def grad(self) -> Self: ...
    @grad.setter
    def grad(self, grad) -> None: ...
    def data_ptr(self, *, storage: bool = False) -> int: ...
    def zero_grad(self, set_to_none: bool = True) -> Self: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def batch_dims(self) -> int: ...
    @batch_dims.setter
    def batch_dims(self, value: int) -> None: ...
    def ndimension(self) -> int: ...
    @property
    def ndim(self) -> int: ...
    def dim(self) -> int: ...
    def numel(self) -> int: ...
    @property
    def depth(self) -> int: ...
    @overload
    def expand(self, *shape: int) -> Self: ...
    @overload
    def expand(self, shape: torch.Size | Sequence[int]) -> Self: ...
    def expand_as(self, other: TensorCollection | torch.Tensor) -> Self: ...
    def new_zeros(
        self,
        *size: torch.Size,
        dtype: torch.dtype = None,
        device: DeviceType = ...,
        requires_grad: bool = False,
        layout: torch.layout = ...,
        pin_memory: bool | None = None,
        empty_lazy: bool = False,
    ) -> Self: ...
    def new_ones(
        self,
        *size: torch.Size,
        dtype: torch.dtype = None,
        device: DeviceType = ...,
        requires_grad: bool = False,
        layout: torch.layout = ...,
        pin_memory: bool | None = None,
        empty_lazy: bool = False,
    ) -> Self: ...
    def new_empty(
        self,
        *size: torch.Size,
        dtype: torch.dtype = None,
        device: DeviceType = ...,
        requires_grad: bool = False,
        layout: torch.layout = ...,
        pin_memory: bool | None = None,
        empty_lazy: bool = False,
    ) -> Self: ...
    def new_full(
        self,
        size: torch.Size,
        fill_value,
        *,
        dtype: torch.dtype = None,
        device: DeviceType = ...,
        requires_grad: bool = False,
        layout: torch.layout = ...,
        pin_memory: bool | None = None,
        empty_lazy: bool = False,
    ) -> Self: ...
    def new_tensor(
        self,
        data: torch.Tensor | TensorDictBase,
        *,
        dtype: torch.dtype = None,
        device: DeviceType = ...,
        requires_grad: bool = False,
        pin_memory: bool | None = None,
    ) -> Self: ...
    def unbind(self, dim: int) -> tuple[T, ...]: ...
    def chunk(self, chunks: int, dim: int = 0) -> tuple[Self, ...]: ...
    def unsqueeze(self, dim: int) -> Self: ...
    def squeeze(self, dim: int | None = None) -> Self: ...
    @overload
    def reshape(self, *shape: int) -> Self: ...
    @overload
    def reshape(self, shape: Sequence[int] | torch.Size) -> Self: ...
    def repeat_interleave(
        self,
        repeats: torch.Tensor | int,
        dim: int | None = None,
        *,
        output_size: int | None = None,
    ) -> Self: ...
    def repeat(self, *repeats: int) -> Self: ...
    def cat_tensors(
        self,
        *keys: NestedKey,
        out_key: NestedKey,
        dim: int = 0,
        keep_entries: bool = False,
    ) -> Self: ...
    def stack_tensors(
        self,
        *keys: NestedKey,
        out_key: NestedKey,
        dim: int = 0,
        keep_entries: bool = False,
    ) -> Self: ...
    def cat_from_tensordict(
        self,
        dim: int = 0,
        *,
        sorted: bool | list[NestedKey] | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def stack_from_tensordict(
        self,
        dim: int = 0,
        *,
        sorted: bool | list[NestedKey] | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    @classmethod
    def stack(cls, input, dim: int = 0, *, out: Incomplete | None = None) -> Self: ...
    @classmethod
    def cat(cls, input, dim: int = 0, *, out: Incomplete | None = None) -> Self: ...
    @classmethod
    def lazy_stack(
        cls, input, dim: int = 0, *, out: Incomplete | None = None, **kwargs
    ) -> Self: ...
    @classmethod
    def maybe_dense_stack(
        cls, input, dim: int = 0, *, out: Incomplete | None = None, **kwargs
    ) -> Self: ...
    def split(self, split_size: int | list[int], dim: int = 0) -> list[Self]: ...
    def gather(self, dim: int, index: Tensor, out: T | None = None) -> Self: ...
    @overload
    def view(self, *shape: int) -> Self: ...
    @overload
    def view(self, dtype) -> Self: ...
    @overload
    def view(self, shape: torch.Size) -> Self: ...
    def view(
        self,
        *shape: int,
        size: Sequence[int] | torch.Size | None = None,
        batch_size: torch.Size | None = None,
    ) -> Self: ...
    def transpose(self, dim0, dim1) -> Self: ...
    @overload
    def permute(self, *dims: int) -> Self: ...
    @overload
    def permute(self, dims: Sequence[int]) -> Self: ...
    @property
    def names(self) -> list[str]: ...
    @names.setter
    def names(self, value: Sequence[str]) -> None: ...
    def refine_names(self, *names: str) -> Self: ...
    def rename(self, *names: str, **rename_map: str) -> Self: ...
    def rename_(self, *names: str, **rename_map: str) -> Self: ...
    @property
    def device(self) -> torch.device | None: ...
    @device.setter
    def device(self, value: DeviceType) -> torch.device | None: ...
    def clear(self) -> Self: ...
    def clear_refs_for_compile_(self) -> Self: ...
    @classmethod
    def fromkeys(cls, keys: list[NestedKey], value: Any = 0) -> Self: ...
    def popitem(self) -> tuple[NestedKey, CompatibleType]: ...
    def clear_device_(self) -> Self: ...
    def param_count(self, *, count_duplicates: bool = True) -> int: ...
    def bytes(self, *, count_duplicates: bool = True) -> int: ...
    def pin_memory(
        self, num_threads: int | None = None, inplace: bool = False
    ) -> Self: ...
    def pin_memory_(self, num_threads: int | str = 0) -> Self: ...
    def cpu(self, **kwargs) -> Self: ...
    def cuda(self, device: int | None = None, **kwargs) -> Self: ...
    @property
    def is_cuda(self) -> bool: ...
    @property
    def is_cpu(self) -> bool: ...
    def state_dict(
        self,
        destination: Incomplete | None = None,
        prefix: str = "",
        keep_vars: bool = False,
        flatten: bool = False,
    ) -> OrderedDict[str, Any]: ...
    def load_state_dict(
        self,
        state_dict: OrderedDict[str, Any],
        strict: bool = True,
        assign: bool = False,
        from_flatten: bool = False,
    ) -> Self: ...
    def is_shared(self) -> bool: ...
    def is_memmap(self) -> bool: ...
    def share_memory_(self) -> Self: ...
    def densify(self, layout: torch.layout = ...) -> Self: ...
    @property
    def saved_path(self) -> Self: ...
    def consolidate(
        self,
        filename: Path | str | None = None,
        *,
        num_threads: int = 0,
        device: torch.device | None = None,
        non_blocking: bool = False,
        inplace: bool = False,
        return_early: bool = False,
        use_buffer: bool = False,
        share_memory: bool = False,
        pin_memory: bool = False,
        metadata: bool = False,
    ) -> Self: ...
    @classmethod
    def from_consolidated(cls, filename) -> Self: ...
    def is_consolidated(self) -> bool: ...
    def memmap_(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        *,
        num_threads: int = 0,
        return_early: bool = False,
        share_non_tensor: bool = False,
        existsok: bool = True,
        robust_key: bool | None = None,
    ) -> Self: ...
    def make_memmap(
        self,
        key: NestedKey,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        robust_key: bool | None = None,
    ) -> MemoryMappedTensor: ...
    def make_memmap_from_storage(
        self,
        key: NestedKey,
        storage: torch.UntypedStorage,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        robust_key: bool | None = None,
    ) -> MemoryMappedTensor: ...
    def make_memmap_from_tensor(
        self,
        key: NestedKey,
        tensor: torch.Tensor,
        *,
        copy_data: bool = True,
        robust_key: bool | None = None,
    ) -> MemoryMappedTensor: ...
    def save(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        *,
        num_threads: int = 0,
        return_early: bool = False,
        share_non_tensor: bool = False,
        robust_key: bool | None = None,
    ) -> Self: ...
    def dumps(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        *,
        num_threads: int = 0,
        return_early: bool = False,
        share_non_tensor: bool = False,
        robust_key: bool | None = None,
    ) -> Self: ...
    def memmap(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        *,
        num_threads: int = 0,
        return_early: bool = False,
        share_non_tensor: bool = False,
        existsok: bool = True,
        robust_key: bool | None = None,
    ) -> Self: ...
    def memmap_like(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        *,
        existsok: bool = True,
        num_threads: int = 0,
        return_early: bool = False,
        share_non_tensor: bool = False,
        robust_key: bool | None = None,
    ) -> Self: ...
    @classmethod
    def load(cls, prefix: str | Path, *args, **kwargs) -> Self: ...
    def load_(self, prefix: str | Path, *args, **kwargs) -> Self: ...
    @classmethod
    def load_memmap(
        cls,
        prefix: str | Path,
        device: torch.device | None = None,
        non_blocking: bool = False,
        *,
        out: TensorCollection | None = None,
        robust_key: bool | None = None,
    ) -> Self: ...
    def load_memmap_(
        self, prefix: str | Path, robust_key: bool | None = None
    ) -> Self: ...
    def memmap_refresh_(self) -> Self: ...
    def entry_class(self, key: NestedKey) -> type: ...
    def set(
        self,
        key: NestedKey,
        item: CompatibleType,
        inplace: bool = False,
        *,
        non_blocking: bool = False,
        **kwargs: Any,
    ) -> Self: ...
    def set_non_tensor(self, key: NestedKey, value: Any) -> Self: ...
    def get_non_tensor(self, key: NestedKey, default=...) -> CompatibleType: ...
    def filter_non_tensor_data(self) -> Self: ...
    def filter_empty_(self) -> Self: ...
    def set_at_(
        self,
        key: NestedKey,
        value: CompatibleType,
        index: IndexType,
        *,
        non_blocking: bool = False,
    ) -> Self: ...
    def set_(
        self, key: NestedKey, item: CompatibleType, *, non_blocking: bool = False
    ) -> Self: ...
    @overload
    def get(self, key: NestedKey) -> CompatibleType: ...
    @overload
    def get(self, key: NestedKey, default: CompatibleType | Any) -> CompatibleType: ...
    @overload
    def get(self, key: NestedKey, *args, **kwargs) -> CompatibleType: ...
    @overload
    def get(
        self,
        key: NestedKey,
        *,
        as_list: bool = False,
        as_padded_tensor: bool = False,
        as_nested_tensor: bool = False,
        padding_side: str = "right",
        layout: torch.layout | None = None,
        padding_value: float | int | bool = 0.0,
        **kwargs,
    ) -> CompatibleType: ...
    @overload
    def get_at(self, key: NestedKey, index: IndexType) -> CompatibleType: ...
    @overload
    def get_at(
        self, key: NestedKey, index: IndexType, default: CompatibleType | Any
    ) -> CompatibleType: ...
    def get_at(
        self,
        key: NestedKey,
        *args,
        **kwargs,
    ) -> CompatibleType: ...
    def get_item_shape(self, key: NestedKey) -> torch.Size: ...
    def update(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        clone: bool = False,
        inplace: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
        is_leaf: Callable[[type], bool] | None = None,
        update_batch_size: bool = False,
        ignore_lock: bool = False,
    ) -> Self: ...
    def update_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        clone: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
    ) -> Self: ...
    def update_at_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        idx: IndexType,
        clone: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
    ) -> Self: ...
    def replace(self, *args, **kwargs) -> Self: ...
    def create_nested(self, key: NestedKey) -> Self: ...
    def copy_(self, tensordict: T, non_blocking: bool = False) -> Self: ...
    def copy_at_(
        self, tensordict: T, idx: IndexType, non_blocking: bool = False
    ) -> Self: ...
    def is_empty(self) -> bool: ...
    def setdefault(
        self, key: NestedKey, default: CompatibleType, inplace: bool = False
    ) -> CompatibleType: ...
    def items(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Incomplete | None = None,
        *,
        sort: bool = False,
    ) -> Iterator[tuple[str, CompatibleType]]: ...
    def non_tensor_items(
        self, include_nested: bool = False
    ) -> Iterator[tuple[str, Any]]: ...
    def values(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Incomplete | None = None,
        *,
        sort: bool = False,
    ) -> Iterator[CompatibleType]: ...
    def keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[type], bool] | None = None,
        *,
        sort: bool = False,
    ) -> Iterator[NestedKey]: ...
    def pop(self, key: NestedKey, default: Any = ...) -> CompatibleType: ...
    @property
    def sorted_keys(self) -> list[NestedKey]: ...
    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Self: ...
    def unflatten(self, dim: int, unflattened_size: torch.Size) -> Self: ...
    def rename_key_(
        self, old_key: NestedKey, new_key: NestedKey, safe: bool = False
    ) -> Self: ...
    def del_(self, key: NestedKey) -> Self: ...
    def gather_and_stack(
        self, dst: int, group: "dist.ProcessGroup" | None = None
    ) -> Self | None: ...
    def send(
        self,
        dst: int,
        *,
        group: dist.ProcessGroup | None = None,
        init_tag: int = 0,
        pseudo_rand: bool = False,
    ) -> None: ...
    def recv(
        self,
        src: int,
        *,
        group: dist.ProcessGroup | None = None,
        init_tag: int = 0,
        pseudo_rand: bool = False,
    ) -> int: ...
    @classmethod
    def from_remote_init(
        cls: T,
        src: int,
        group: "ProcessGroup" | None = None,  # noqa: F821
        device: torch.device | None = None,
    ) -> T: ...
    def init_remote(
        self,
        dst: int,
        group: "ProcessGroup" | None = None,  # noqa: F821
        device: torch.device | None = None,
    ) -> Self: ...
    def isend(
        self,
        dst: int,
        *,
        group: "dist.ProcessGroup" | None = None,  # noqa: F821
        init_tag: int = 0,
        pseudo_rand: bool = False,
    ) -> int: ...
    def irecv(
        self,
        src: int,
        *,
        group: dist.ProcessGroup | None = None,
        return_premature: bool = False,
        init_tag: int = 0,
        pseudo_rand: bool = False,
    ) -> tuple[int, list[torch.Future]] | list[torch.Future] | None: ...
    def reduce(
        self,
        dst,
        op: Incomplete | None = None,
        async_op: bool = False,
        return_premature: bool = False,
        group: Incomplete | None = None,
    ) -> Self: ...
    def apply_(self, fn: Callable, *others, **kwargs) -> Self: ...
    def apply(
        self,
        fn: Callable,
        *others: T,
        batch_size: Sequence[int] | None = None,
        device: torch.device | None = ...,
        names: Sequence[str] | None = ...,
        inplace: bool = False,
        default: Any = ...,
        filter_empty: bool | None = None,
        propagate_lock: bool = False,
        call_on_nested: bool = False,
        out: TensorCollection | None = None,
        **constructor_kwargs,
    ) -> Self | None: ...
    def named_apply(
        self,
        fn: Callable,
        *others: T,
        nested_keys: bool = False,
        batch_size: Sequence[int] | None = None,
        device: torch.device | None = ...,
        names: Sequence[str] | None = ...,
        inplace: bool = False,
        default: Any = ...,
        filter_empty: bool | None = None,
        propagate_lock: bool = False,
        call_on_nested: bool = False,
        out: TensorCollection | None = None,
        **constructor_kwargs,
    ) -> Self | None: ...
    def map(
        self,
        fn: Callable[[TensorDictBase], TensorDictBase | None],
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
    ) -> Self: ...
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
    ) -> Self: ...
    def record_stream(self, stream: torch.cuda.Stream) -> Self: ...
    def __add__(self, other: TensorCollection | torch.Tensor) -> Self: ...
    def __iadd__(self, other: TensorCollection | torch.Tensor) -> Self: ...
    def __abs__(self) -> Self: ...
    def __truediv__(self, other: TensorCollection | torch.Tensor) -> Self: ...
    def __itruediv__(self, other: TensorCollection | torch.Tensor) -> Self: ...
    def __mod__(self, other: TensorCollection | torch.Tensor) -> Self: ...
    def __mul__(self, other: TensorCollection | torch.Tensor) -> Self: ...
    def __imul__(self, other: TensorCollection | torch.Tensor) -> Self: ...
    def __sub__(self, other: TensorCollection | torch.Tensor) -> Self: ...
    def __isub__(self, other: TensorCollection | torch.Tensor) -> Self: ...
    def __pow__(self, other: TensorCollection | torch.Tensor) -> Self: ...
    def __ipow__(self, other: TensorCollection | torch.Tensor) -> Self: ...
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
        alpha: float | None = None,
        default: str | CompatibleType | None = None,
    ) -> Self: ...
    def add_(
        self, other: TensorCollection | float, *, alpha: float | None = None
    ) -> Self: ...
    def lerp(
        self,
        end: TensorCollection | torch.Tensor,
        weight: TensorCollection | torch.Tensor | float,
    ) -> Self: ...
    def lerp_(
        self, end: TensorCollection | float, weight: TensorCollection | float
    ) -> Self: ...
    def addcdiv(
        self,
        other1: TensorCollection | torch.Tensor,
        other2: TensorCollection | torch.Tensor,
        value: float | None = 1,
    ) -> Self: ...
    def addcdiv_(self, other1, other2, *, value: float | None = 1) -> Self: ...
    def addcmul(self, other1, other2, *, value: float | None = 1) -> Self: ...
    def addcmul_(self, other1, other2, *, value: float | None = 1) -> Self: ...
    def sub(
        self,
        other: TensorCollection | float,
        *,
        alpha: float | None = None,
        default: str | CompatibleType | None = None,
    ) -> Self: ...
    def rsub(
        self,
        other: TensorCollection | float,
        *,
        alpha: float | None = None,
        default: str | CompatibleType | None = None,
    ) -> Self: ...
    def sub_(
        self, other: TensorCollection | float, alpha: float | None = None
    ) -> Self: ...
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
        min: TensorCollection | torch.Tensor = None,
        max: TensorCollection | torch.Tensor = None,
        *,
        out=None,
    ) -> Self: ...
    def logsumexp(self, dim=None, keepdim=False, *, out=None) -> Self: ...
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
    def __enter__(self) -> Any: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None: ...
    def select(
        self, *keys: NestedKey, inplace: bool = False, strict: bool = True
    ) -> Self: ...
    def exclude(self, *keys: NestedKey, inplace: bool = False) -> Self: ...
    def to_tensordict(self, *, retain_none: bool | None = None) -> Self: ...
    def clone(self, recurse: bool = True, **kwargs) -> Self: ...
    def copy(self) -> Self: ...
    def to_padded_tensor(
        self, padding: float = 0.0, mask_key: NestedKey | None = None
    ) -> Self: ...
    def as_tensor(self) -> Self: ...
    def to_lazystack(self, dim: int = 0) -> Self: ...
    def to_dict(
        self,
        *,
        retain_none: bool = True,
        convert_tensors: bool | Literal["numpy"] = False,
        tolist_first: bool = False,
    ) -> dict[str, Any]: ...
    def to_mds(
        self,
        *,
        out: str | tuple[str, str],
        columns: dict[str, str] | None = None,
        writer: "MDSWriter" | None = None,
    ) -> None: ...
    @classmethod
    def from_list(
        cls,
        input,
        *,
        auto_batch_size: bool | None = None,
        batch_size: torch.Size | None = None,
        device: torch.device | None = None,
        batch_dims: int | None = None,
        names: List[str] | None = None,
        lazy: bool | None = None,
    ) -> Self: ...
    def tolist(
        self,
        *,
        convert_nodes: bool = True,
        convert_tensors: bool | Literal["numpy"] = False,
        tolist_first: bool = False,
        as_linked_list: bool = False,
    ) -> list[Any]: ...
    def numpy(self) -> np.ndarray | dict[str, Any]: ...
    def to_namedtuple(self, dest_cls: type | None = None) -> Any: ...
    @classmethod
    def _from_tensordict(
        cls,
        tensordict: TensorCollection,
        non_tensordict: dict | None = None,
        safe: bool = True,
    ) -> Self: ...
    @classmethod
    def from_tensordict(
        cls,
        tensordict: TensorCollection,
        non_tensordict: dict | None = None,
        safe: bool = True,
    ) -> Self: ...
    def _from_tensordict_with_copy(self, tensordict: TensorCollection) -> Self: ...
    def _from_tensordict_with_none(self, tensordict: TensorCollection) -> Self: ...
    @classmethod
    def from_namedtuple(cls, named_tuple, *, auto_batch_size: bool = False) -> Self: ...
    def from_tuple(
        cls,
        obj,
        *,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
    ) -> Self: ...
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
    def to_h5(self, filename: str, **kwargs) -> Any: ...
    def empty(
        self,
        recurse: bool = False,
        *,
        batch_size: Incomplete | None = None,
        device=...,
        names: Incomplete | None = None,
    ) -> Self: ...
    def zero_(self) -> Self: ...
    def fill_(self, key: NestedKey, value: float | bool) -> Self: ...
    def masked_fill_(self, mask: Tensor, value: float | bool) -> Self: ...
    def masked_fill(self, mask: Tensor, value: float | bool) -> Self: ...
    def where(
        self,
        condition,
        other,
        *,
        out: Incomplete | None = None,
        pad: Incomplete | None = None,
        update_batch_size: bool = False,
    ) -> Self: ...
    def masked_select(self, mask: Tensor) -> Self: ...
    def is_contiguous(self) -> bool: ...
    def contiguous(self) -> Self: ...
    def flatten_keys(
        self,
        separator: str = ".",
        inplace: bool = False,
        is_leaf: Callable[[type], bool] | None = None,
    ) -> Self: ...
    def unflatten_keys(self, separator: str = ".", inplace: bool = False) -> Self: ...
    def tensor_split(
        self,
        indices_or_sections: int | list[int] | tuple[int, ...] | torch.Tensor,
        dim: int = 0,
    ) -> tuple[Self, ...]: ...
    def split_keys(
        self,
        *key_sets,
        inplace: bool = False,
        strict: bool = True,
        reproduce_struct: bool = False,
    ) -> tuple[Self, ...]: ...
    def separates(
        self,
        *keys: NestedKey,
        default: Any = NO_DEFAULT,
        strict: bool = True,
        filter_empty: bool = True,
    ) -> Self: ...
    def norm(
        self,
        *,
        out=None,
        dtype: torch.dtype | None = None,
    ) -> Self: ...
    def softmax(self, dim: int, dtype: torch.dtype | None = None) -> Self: ...
    @property
    def is_locked(self) -> bool: ...
    @is_locked.setter
    def is_locked(self, value: bool) -> None: ...
    def lock_(self) -> Self: ...
    def unlock_(self) -> Self: ...
    @overload
    def to(
        self,
        device: int | torch.device | None = ...,
        dtype: torch.dtype | None = ...,
        non_blocking: bool = ...,
        inplace: bool = False,
    ) -> Self: ...
    @overload
    def to(self, dtype: torch.dtype, non_blocking: bool = ...) -> Self: ...
    @overload
    def to(self, tensor: Tensor, non_blocking: bool = ...) -> Self: ...
    @overload
    def to(self, *, other: T, non_blocking: bool = ...) -> Self: ...
    @overload
    def to(self, *, batch_size: torch.Size) -> Self: ...
    def to(self, *args, **kwargs) -> Self: ...
    def is_floating_point(self) -> bool: ...
    def double(self) -> Self: ...
    def float(self) -> Self: ...
    def int(self) -> Self: ...
    def bool(self) -> Self: ...
    def half(self) -> Self: ...
    def type(self, dst_type: torch.dtype) -> Self: ...
    @property
    def requires_grad(self) -> bool: ...
    def requires_grad_(self, requires_grad: bool = True) -> Self: ...
    def detach_(self) -> Self: ...
    def detach(self) -> Self: ...
    def bfloat16(self) -> Self: ...
    def complex128(self) -> Self: ...
    def complex32(self) -> Self: ...
    def complex64(self) -> Self: ...
    def float16(self) -> Self: ...
    def float32(self) -> Self: ...
    def float64(self) -> Self: ...
    def int16(self) -> Self: ...
    def int32(self) -> Self: ...
    def int64(self) -> Self: ...
    def int8(self) -> Self: ...
    def qint32(self) -> Self: ...
    def qint8(self) -> Self: ...
    def quint4x2(self) -> Self: ...
    def quint8(self) -> Self: ...
    def uint16(self) -> Self: ...
    def uint32(self) -> Self: ...
    def uint64(self) -> Self: ...
    def uint8(self) -> Self: ...

class NonTensorDataBase(TensorClass): ...
class NonTensorData(NonTensorDataBase): ...
class MetaData(NonTensorDataBase): ...
class NonTensorStack(TensorDictBase): ...

@dataclass_transform()
def tensorclass(
    cls: T = None,
    /,
    *,
    autocast: bool = False,
    frozen: bool = False,
    nocast: bool = False,
    shadow: bool = False,
    tensor_only: bool = False,
) -> T: ...
def is_non_tensor(obj) -> bool: ...
def from_dataclass(
    obj: Any,
    *,
    dest_cls: Type | None = None,
    auto_batch_size: bool = False,
    batch_dims: int | None = None,
    batch_size: torch.Size | None = None,
    frozen: bool = False,
    autocast: bool = False,
    nocast: bool = False,
    inplace: bool = False,
    shadow: bool = False,
    tensor_only: bool = False,
    device: torch.device | None = None,
) -> Any: ...

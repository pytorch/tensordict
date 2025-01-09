# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import dataclasses
import enum
from abc import abstractmethod
from collections.abc import MutableMapping
from pathlib import Path
from typing import (
    Any,
    Callable,
    dataclass_transform,
    Generator,
    Iterator,
    OrderedDict,
    overload,
    Sequence,
    TypeVar,
)

import numpy as np

import torch
from _typeshed import Incomplete
from tensordict import TensorDictBase
from tensordict._contextlib import LAST_OP_MAPS as LAST_OP_MAPS
from tensordict._nestedkey import NestedKey as NestedKey
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
    is_non_tensor as is_non_tensor,
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

class _NoDefault(enum.IntEnum):
    ZERO = 0

NO_DEFAULT: Incomplete

class _BEST_ATTEMPT_INPLACE:
    def __bool__(self) -> bool: ...

BEST_ATTEMPT_INPLACE: Incomplete
CompatibleType = Tensor

T = TypeVar("T", bound="TensorDictBase")

@dataclasses.dataclass
class TensorClass:
    _autocast: bool = False
    _nocast: bool = False
    _frozen: bool = False
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
    def __ne__(self, other: object) -> T: ...
    def __xor__(self, other: TensorDictBase | float): ...
    def __or__(self, other: TensorDictBase | torch.Tensor) -> T: ...
    def __eq__(self, other: object) -> T: ...
    def __ge__(self, other: object) -> T: ...
    def __gt__(self, other: object) -> T: ...
    def __le__(self, other: object) -> T: ...
    def __lt__(self, other: object) -> T: ...
    def __iter__(self) -> Generator: ...
    def __len__(self) -> int: ...
    def __contains__(self, key: NestedKey) -> bool: ...
    def __getitem__(self, index: IndexType) -> Any: ...
    __getitems__ = __getitem__

    def __setitem__(self, index: IndexType, value: Any) -> None: ...
    def __delitem__(self, key: NestedKey) -> T: ...
    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Callable: ...
    def all(self, dim: int = None) -> bool | TensorDictBase: ...
    def any(self, dim: int = None) -> bool | TensorDictBase: ...
    def isfinite(self) -> T: ...
    def isnan(self) -> T: ...
    def isneginf(self) -> T: ...
    def isposinf(self) -> T: ...
    def isreal(self) -> T: ...
    def amin(
        self,
        dim: int | NO_DEFAULT = ...,
        keepdim: bool = False,
        *,
        reduce: bool | None = None,
    ) -> TensorDictBase | torch.Tensor: ...
    def min(
        self,
        dim: int | NO_DEFAULT = ...,
        keepdim: bool = False,
        *,
        reduce: bool | None = None,
        return_indices: bool = True,
    ) -> TensorDictBase | torch.Tensor: ...
    def amax(
        self,
        dim: int | NO_DEFAULT = ...,
        keepdim: bool = False,
        *,
        reduce: bool | None = None,
    ) -> TensorDictBase | torch.Tensor: ...
    def max(
        self,
        dim: int | NO_DEFAULT = ...,
        keepdim: bool = False,
        *,
        reduce: bool | None = None,
        return_indices: bool = True,
    ) -> TensorDictBase | torch.Tensor: ...
    def cummin(
        self, dim: int, *, reduce: bool | None = None, return_indices: bool = True
    ) -> TensorDictBase | torch.Tensor: ...
    def cummax(
        self, dim: int, *, reduce: bool | None = None, return_indices: bool = True
    ) -> TensorDictBase | torch.Tensor: ...
    def mean(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool | None = None,
    ) -> TensorDictBase | torch.Tensor: ...
    def nanmean(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool | None = None,
    ) -> TensorDictBase | torch.Tensor: ...
    def prod(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool | None = None,
    ) -> TensorDictBase | torch.Tensor: ...
    def sum(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool | None = None,
    ) -> TensorDictBase | torch.Tensor: ...
    def nansum(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool | None = None,
    ) -> TensorDictBase | torch.Tensor: ...
    def std(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        correction: int = 1,
        reduce: bool | None = None,
    ) -> TensorDictBase | torch.Tensor: ...
    def var(
        self,
        dim: int | tuple[int] = ...,
        keepdim: bool = ...,
        *,
        correction: int = 1,
        reduce: bool | None = None,
    ) -> TensorDictBase | torch.Tensor: ...
    def auto_batch_size_(self, batch_dims: int | None = None) -> T: ...
    def auto_device_(self) -> T: ...
    @classmethod
    def from_dataclass(
        cls, dataclass, *, auto_batch_size: bool = False, as_tensorclass: bool = False
    ): ...
    @classmethod
    def from_any(cls, obj, *, auto_batch_size: bool = False): ...
    @classmethod
    def from_dict(
        cls,
        input_dict,
        *,
        batch_size: torch.Size | None = None,
        device: torch.device | None = None,
        batch_dims: int | None = None,
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
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
    ): ...
    def to_pytree(self): ...
    @classmethod
    def from_h5(cls, filename, mode: str = "r"): ...
    @classmethod
    def from_module(
        cls,
        module,
        as_module: bool = False,
        lock: bool = True,
        use_state_dict: bool = False,
    ): ...
    @classmethod
    def from_modules(
        cls,
        *modules,
        as_module: bool = False,
        lock: bool = True,
        use_state_dict: bool = False,
        lazy_stack: bool = False,
        expand_identical: bool = False,
    ): ...
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
    ): ...
    @property
    def shape(self) -> torch.Size: ...
    @shape.setter
    def shape(self, value) -> torch.Size: ...
    @property
    def batch_size(self) -> torch.Size: ...
    def size(self, dim: int | None = None) -> torch.Size | int: ...
    @property
    def data(self): ...
    @property
    def grad(self): ...
    def data_ptr(self, *, storage: bool = False): ...
    @grad.setter
    def grad(self, grad) -> None: ...
    def zero_grad(self, set_to_none: bool = True) -> T: ...
    @property
    def dtype(self): ...
    @property
    def batch_dims(self) -> int: ...
    def ndimension(self) -> int: ...
    @property
    def ndim(self) -> int: ...
    def dim(self) -> int: ...
    def numel(self) -> int: ...
    @property
    def depth(self) -> int: ...
    @overload
    def expand(self, *shape: int) -> T: ...
    @overload
    def expand(self, shape: torch.Size) -> T: ...
    def expand_as(self, other: TensorDictBase | torch.Tensor) -> TensorDictBase: ...
    def new_zeros(
        self,
        *size: torch.Size,
        dtype: torch.dtype = None,
        device: DeviceType = ...,
        requires_grad: bool = False,
        layout: torch.layout = ...,
        pin_memory: bool = None,
    ): ...
    def new_ones(
        self,
        *size: torch.Size,
        dtype: torch.dtype = None,
        device: DeviceType = ...,
        requires_grad: bool = False,
        layout: torch.layout = ...,
        pin_memory: bool = None,
    ): ...
    def new_empty(
        self,
        *size: torch.Size,
        dtype: torch.dtype = None,
        device: DeviceType = ...,
        requires_grad: bool = False,
        layout: torch.layout = ...,
        pin_memory: bool = None,
    ): ...
    def new_full(
        self,
        size: torch.Size,
        fill_value,
        *,
        dtype: torch.dtype = None,
        device: DeviceType = ...,
        requires_grad: bool = False,
        layout: torch.layout = ...,
        pin_memory: bool = None,
    ): ...
    def new_tensor(
        self,
        data: torch.Tensor | TensorDictBase,
        *,
        dtype: torch.dtype = None,
        device: DeviceType = ...,
        requires_grad: bool = False,
        pin_memory: bool | None = None,
    ): ...
    def unbind(self, dim: int) -> tuple[T, ...]: ...
    def chunk(self, chunks: int, dim: int = 0) -> tuple[TensorDictBase, ...]: ...
    @overload
    def unsqueeze(self, dim: int) -> T: ...
    @overload
    def squeeze(self, dim: int | None = None) -> T: ...
    @overload
    def reshape(self, *shape: int): ...
    @overload
    def reshape(self, shape: list | tuple): ...
    def repeat_interleave(
        self, repeats: torch.Tensor | int, dim: int = None, *, output_size: int = None
    ) -> TensorDictBase: ...
    def repeat(self, *repeats: int) -> TensorDictBase: ...
    def cat_tensors(
        self,
        *keys: NestedKey,
        out_key: NestedKey,
        dim: int = 0,
        keep_entries: bool = False,
    ) -> T: ...
    def stack_tensors(
        self,
        *keys: NestedKey,
        out_key: NestedKey,
        dim: int = 0,
        keep_entries: bool = False,
    ) -> T: ...
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
    def stack(cls, input, dim: int = 0, *, out: Incomplete | None = None): ...
    @classmethod
    def cat(cls, input, dim: int = 0, *, out: Incomplete | None = None): ...
    @classmethod
    def lazy_stack(
        cls, input, dim: int = 0, *, out: Incomplete | None = None, **kwargs
    ): ...
    @classmethod
    def maybe_dense_stack(
        cls, input, dim: int = 0, *, out: Incomplete | None = None, **kwargs
    ): ...
    def split(
        self, split_size: int | list[int], dim: int = 0
    ) -> list[TensorDictBase]: ...
    def gather(self, dim: int, index: Tensor, out: T | None = None) -> T: ...
    @overload
    def view(self, *shape: int): ...
    @overload
    def view(self, dtype) -> T: ...
    @overload
    def view(self, shape: torch.Size): ...
    def view(
        self,
        *shape: int,
        size: list | tuple | torch.Size | None = None,
        batch_size: torch.Size | None = None,
    ): ...
    def transpose(self, dim0, dim1): ...
    @overload
    def permute(self, *dims: int): ...
    @overload
    def permute(self, dims: list | tuple): ...
    @property
    def names(self): ...
    def refine_names(self, *names) -> T: ...
    def rename(self, *names, **rename_map): ...
    def rename_(self, *names, **rename_map): ...
    @property
    def device(self) -> torch.device | None: ...
    @device.setter
    def device(self, value: DeviceType) -> torch.device | None: ...
    def clear(self) -> T: ...
    @classmethod
    def fromkeys(cls, keys: list[NestedKey], value: Any = 0): ...
    def popitem(self) -> tuple[NestedKey, CompatibleType]: ...
    def clear_device_(self) -> T: ...
    def param_count(self, *, count_duplicates: bool = True) -> int: ...
    def bytes(self, *, count_duplicates: bool = True) -> int: ...
    def pin_memory(
        self, num_threads: int | None = None, inplace: bool = False
    ) -> T: ...
    def pin_memory_(self, num_threads: int | str = 0) -> T: ...
    def cpu(self, **kwargs) -> T: ...
    def cuda(self, device: int = None, **kwargs) -> T: ...
    @property
    def is_cuda(self): ...
    @property
    def is_cpu(self): ...
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
    ) -> T: ...
    def is_shared(self) -> bool: ...
    def is_memmap(self) -> bool: ...
    def share_memory_(self) -> T: ...
    def densify(self, layout: torch.layout = ...): ...
    @property
    def saved_path(self): ...
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
    ) -> T: ...
    @classmethod
    def from_consolidated(cls, filename): ...
    def is_consolidated(self): ...
    def memmap_(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        *,
        num_threads: int = 0,
        return_early: bool = False,
        share_non_tensor: bool = False,
        existsok: bool = True,
    ) -> T: ...
    def make_memmap(
        self,
        key: NestedKey,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> MemoryMappedTensor: ...
    def make_memmap_from_storage(
        self,
        key: NestedKey,
        storage: torch.UntypedStorage,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> MemoryMappedTensor: ...
    def make_memmap_from_tensor(
        self, key: NestedKey, tensor: torch.Tensor, *, copy_data: bool = True
    ) -> MemoryMappedTensor: ...
    def save(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        *,
        num_threads: int = 0,
        return_early: bool = False,
        share_non_tensor: bool = False,
    ) -> T: ...
    def dumps(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        *,
        num_threads: int = 0,
        return_early: bool = False,
        share_non_tensor: bool = False,
    ) -> T: ...
    def memmap(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        *,
        num_threads: int = 0,
        return_early: bool = False,
        share_non_tensor: bool = False,
        existsok: bool = True,
    ) -> T: ...
    def memmap_like(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        *,
        existsok: bool = True,
        num_threads: int = 0,
        return_early: bool = False,
        share_non_tensor: bool = False,
    ) -> T: ...
    @classmethod
    def load(cls, prefix: str | Path, *args, **kwargs) -> T: ...
    def load_(self, prefix: str | Path, *args, **kwargs): ...
    @classmethod
    def load_memmap(
        cls,
        prefix: str | Path,
        device: torch.device | None = None,
        non_blocking: bool = False,
        *,
        out: TensorDictBase | None = None,
    ) -> T: ...
    def load_memmap_(self, prefix: str | Path): ...
    def memmap_refresh_(self): ...
    def entry_class(self, key: NestedKey) -> type: ...
    def set(
        self,
        key: NestedKey,
        item: CompatibleType,
        inplace: bool = False,
        *,
        non_blocking: bool = False,
        **kwargs: Any,
    ) -> T: ...
    def set_non_tensor(self, key: NestedKey, value: Any): ...
    def get_non_tensor(self, key: NestedKey, default=...): ...
    def filter_non_tensor_data(self) -> T: ...
    def filter_empty_(self): ...
    def set_at_(
        self,
        key: NestedKey,
        value: CompatibleType,
        index: IndexType,
        *,
        non_blocking: bool = False,
    ) -> T: ...
    def set_(
        self, key: NestedKey, item: CompatibleType, *, non_blocking: bool = False
    ) -> T: ...
    def get(self, key: NestedKey, *args, **kwargs) -> CompatibleType: ...
    def get_at(
        self, key: NestedKey, index: IndexType, default: CompatibleType = ...
    ) -> CompatibleType: ...
    def get_item_shape(self, key: NestedKey): ...
    def update(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        clone: bool = False,
        inplace: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
        is_leaf: Callable[[type], bool] | None = None,
    ) -> T: ...
    def update_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        clone: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
    ) -> T: ...
    def update_at_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        idx: IndexType,
        clone: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
    ) -> T: ...
    def replace(self, *args, **kwargs): ...
    def create_nested(self, key): ...
    def copy_(self, tensordict: T, non_blocking: bool = False) -> T: ...
    def copy_at_(
        self, tensordict: T, idx: IndexType, non_blocking: bool = False
    ) -> T: ...
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
    def non_tensor_items(self, include_nested: bool = False): ...
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
    ): ...
    def pop(self, key: NestedKey, default: Any = ...) -> CompatibleType: ...
    @property
    def sorted_keys(self) -> list[NestedKey]: ...
    def flatten(self, start_dim: int = 0, end_dim: int = -1): ...
    def unflatten(self, dim, unflattened_size): ...
    def rename_key_(
        self, old_key: NestedKey, new_key: NestedKey, safe: bool = False
    ) -> T: ...
    def del_(self, key: NestedKey) -> T: ...
    def gather_and_stack(
        self, dst: int, group: "dist.ProcessGroup" | None = None
    ) -> T | None: ...
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
    def isend(
        self,
        dst: int,
        *,
        group: dist.ProcessGroup | None = None,
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
    ): ...
    def apply_(self, fn: Callable, *others, **kwargs) -> T: ...
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
        out: TensorDictBase | None = None,
        **constructor_kwargs,
    ) -> T | None: ...
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
        out: TensorDictBase | None = None,
        **constructor_kwargs,
    ) -> T | None: ...
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
    ): ...
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
    ): ...
    def record_stream(self, stream: torch.cuda.Stream): ...
    def __add__(self, other: TensorDictBase | torch.Tensor) -> T: ...
    def __iadd__(self, other: TensorDictBase | torch.Tensor) -> T: ...
    def __abs__(self): ...
    def __truediv__(self, other: TensorDictBase | torch.Tensor) -> T: ...
    def __itruediv__(self, other: TensorDictBase | torch.Tensor) -> T: ...
    def __mul__(self, other: TensorDictBase | torch.Tensor) -> T: ...
    def __imul__(self, other: TensorDictBase | torch.Tensor) -> T: ...
    def __sub__(self, other: TensorDictBase | torch.Tensor) -> T: ...
    def __isub__(self, other: TensorDictBase | torch.Tensor) -> T: ...
    def __pow__(self, other: TensorDictBase | torch.Tensor) -> T: ...
    def __ipow__(self, other: TensorDictBase | torch.Tensor) -> T: ...
    def abs(self) -> T: ...
    def abs_(self) -> T: ...
    def acos(self) -> T: ...
    def acos_(self) -> T: ...
    def exp(self) -> T: ...
    def exp_(self) -> T: ...
    def neg(self) -> T: ...
    def neg_(self) -> T: ...
    def reciprocal(self) -> T: ...
    def reciprocal_(self) -> T: ...
    def sigmoid(self) -> T: ...
    def sigmoid_(self) -> T: ...
    def sign(self) -> T: ...
    def sign_(self) -> T: ...
    def sin(self) -> T: ...
    def sin_(self) -> T: ...
    def sinh(self) -> T: ...
    def sinh_(self) -> T: ...
    def tan(self) -> T: ...
    def tan_(self) -> T: ...
    def tanh(self) -> T: ...
    def tanh_(self) -> T: ...
    def trunc(self) -> T: ...
    def trunc_(self) -> T: ...
    def lgamma(self) -> T: ...
    def lgamma_(self) -> T: ...
    def frac(self) -> T: ...
    def frac_(self) -> T: ...
    def expm1(self) -> T: ...
    def expm1_(self) -> T: ...
    def log(self) -> T: ...
    def log_(self) -> T: ...
    def log10(self) -> T: ...
    def log10_(self) -> T: ...
    def log1p(self) -> T: ...
    def log1p_(self) -> T: ...
    def log2(self) -> T: ...
    def log2_(self) -> T: ...
    def ceil(self) -> T: ...
    def ceil_(self) -> T: ...
    def floor(self) -> T: ...
    def floor_(self) -> T: ...
    def round(self) -> T: ...
    def round_(self) -> T: ...
    def erf(self) -> T: ...
    def erf_(self) -> T: ...
    def erfc(self) -> T: ...
    def erfc_(self) -> T: ...
    def asin(self) -> T: ...
    def asin_(self) -> T: ...
    def atan(self) -> T: ...
    def atan_(self) -> T: ...
    def cos(self) -> T: ...
    def cos_(self) -> T: ...
    def cosh(self) -> T: ...
    def cosh_(self) -> T: ...
    def add(
        self,
        other: TensorDictBase | torch.Tensor,
        *,
        alpha: float | None = None,
        default: str | CompatibleType | None = None,
    ) -> TensorDictBase: ...
    def add_(self, other: TensorDictBase | float, *, alpha: float | None = None): ...
    def lerp(
        self,
        end: TensorDictBase | torch.Tensor,
        weight: TensorDictBase | torch.Tensor | float,
    ): ...
    def lerp_(self, end: TensorDictBase | float, weight: TensorDictBase | float): ...
    def addcdiv(
        self,
        other1: TensorDictBase | torch.Tensor,
        other2: TensorDictBase | torch.Tensor,
        value: float | None = 1,
    ): ...
    def addcdiv_(self, other1, other2, *, value: float | None = 1): ...
    def addcmul(self, other1, other2, *, value: float | None = 1): ...
    def addcmul_(self, other1, other2, *, value: float | None = 1): ...
    def sub(
        self,
        other: TensorDictBase | float,
        *,
        alpha: float | None = None,
        default: str | CompatibleType | None = None,
    ): ...
    def sub_(self, other: TensorDictBase | float, alpha: float | None = None): ...
    def mul_(self, other: TensorDictBase | torch.Tensor) -> T: ...
    def mul(
        self,
        other: TensorDictBase | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> T: ...
    def maximum_(self, other: TensorDictBase | torch.Tensor) -> T: ...
    def maximum(
        self,
        other: TensorDictBase | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> T: ...
    def minimum_(self, other: TensorDictBase | torch.Tensor) -> T: ...
    def minimum(
        self,
        other: TensorDictBase | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> T: ...
    def clamp(
        self,
        min: TensorDictBase | torch.Tensor = None,
        max: TensorDictBase | torch.Tensor = None,
        *,
        out=None,
    ): ...
    def logsumexp(self, dim=None, keepdim=False, *, out=None): ...
    def clamp_max_(self, other: TensorDictBase | torch.Tensor) -> T: ...
    def clamp_max(
        self,
        other: TensorDictBase | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> T: ...
    def clamp_min_(self, other: TensorDictBase | torch.Tensor) -> T: ...
    def clamp_min(
        self,
        other: TensorDictBase | torch.Tensor,
        default: str | CompatibleType | None = None,
    ) -> T: ...
    def pow_(self, other: TensorDictBase | torch.Tensor) -> T: ...
    def pow(
        self,
        other: TensorDictBase | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> T: ...
    def div_(self, other: TensorDictBase | torch.Tensor) -> T: ...
    def div(
        self,
        other: TensorDictBase | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> T: ...
    def sqrt_(self): ...
    def sqrt(self): ...
    def __enter__(self): ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ): ...
    def select(
        self, *keys: NestedKey, inplace: bool = False, strict: bool = True
    ) -> T: ...
    def exclude(self, *keys: NestedKey, inplace: bool = False) -> T: ...
    def to_tensordict(self, *, retain_none: bool | None = None) -> T: ...
    def clone(self, recurse: bool = True, **kwargs) -> T: ...
    def copy(self): ...
    def to_padded_tensor(
        self, padding: float = 0.0, mask_key: NestedKey | None = None
    ): ...
    def as_tensor(self): ...
    def to_dict(self, *, retain_none: bool = True) -> dict[str, Any]: ...
    def numpy(self): ...
    def to_namedtuple(self, dest_cls: type | None = None): ...
    @classmethod
    def from_namedtuple(cls, named_tuple, *, auto_batch_size: bool = False): ...
    def from_tuple(
        cls,
        obj,
        *,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
    ): ...
    def logical_and(
        self,
        other: TensorDictBase | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> TensorDictBase: ...
    def bitwise_and(
        self,
        other: TensorDictBase | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> TensorDictBase: ...
    @classmethod
    def from_struct_array(
        cls, struct_array: np.ndarray, device: torch.device | None = None
    ) -> T: ...
    def to_struct_array(self): ...
    def to_h5(self, filename, **kwargs): ...
    def empty(
        self,
        recurse: bool = False,
        *,
        batch_size: Incomplete | None = None,
        device=...,
        names: Incomplete | None = None,
    ) -> T: ...
    def zero_(self) -> T: ...
    def fill_(self, key: NestedKey, value: float | bool) -> T: ...
    def masked_fill_(self, mask: Tensor, value: float | bool) -> T: ...
    def masked_fill(self, mask: Tensor, value: float | bool) -> T: ...
    def where(
        self,
        condition,
        other,
        *,
        out: Incomplete | None = None,
        pad: Incomplete | None = None,
    ) -> T: ...
    def masked_select(self, mask: Tensor) -> T: ...
    def is_contiguous(self) -> bool: ...
    def contiguous(self) -> T: ...
    def flatten_keys(
        self,
        separator: str = ".",
        inplace: bool = False,
        is_leaf: Callable[[type], bool] | None = None,
    ) -> T: ...
    def unflatten_keys(self, separator: str = ".", inplace: bool = False) -> T: ...
    def split_keys(
        self,
        *key_sets,
        inplace: bool = False,
        strict: bool = True,
        reproduce_struct: bool = False,
    ): ...
    def separates(
        self,
        *keys: NestedKey,
        default: Any = NO_DEFAULT,
        strict: bool = True,
        filter_empty: bool = True,
    ) -> T: ...
    def norm(
        self,
        *,
        out=None,
        dtype: torch.dtype | None = None,
    ): ...
    def softmax(self, dim: int, dtype: torch.dtype | None = None): ...
    @property
    def is_locked(self) -> bool: ...
    @is_locked.setter
    def is_locked(self, value: bool) -> None: ...
    def lock_(self) -> T: ...
    def unlock_(self) -> T: ...
    @overload
    def to(
        self,
        device: int | device | None = ...,
        dtype: torch.device | str | None = ...,
        non_blocking: bool = ...,
        inplace: bool = False,
    ) -> T: ...
    @overload
    def to(self, dtype: torch.device | str, non_blocking: bool = ...) -> T: ...
    @overload
    def to(self, tensor: Tensor, non_blocking: bool = ...) -> T: ...
    @overload
    def to(self, *, other: T, non_blocking: bool = ...) -> T: ...
    @overload
    def to(self, *, batch_size: torch.Size) -> T: ...
    def to(self, *args, **kwargs) -> T: ...
    def is_floating_point(self): ...
    def double(self): ...
    def float(self): ...
    def int(self): ...
    def bool(self): ...
    def half(self): ...
    def type(self, dst_type): ...
    @property
    def requires_grad(self) -> bool: ...
    def requires_grad_(self, requires_grad: bool = True) -> T: ...
    def detach_(self) -> T: ...
    def detach(self) -> T: ...
    def bfloat16(self) -> T: ...
    def complex128(self) -> T: ...
    def complex32(self) -> T: ...
    def complex64(self) -> T: ...
    def float16(self) -> T: ...
    def float32(self) -> T: ...
    def float64(self) -> T: ...
    def int16(self) -> T: ...
    def int32(self) -> T: ...
    def int64(self) -> T: ...
    def int8(self) -> T: ...
    def qint32(self) -> T: ...
    def qint8(self) -> T: ...
    def quint4x2(self) -> T: ...
    def quint8(self) -> T: ...
    def uint16(self) -> T: ...
    def uint32(self) -> T: ...
    def uint64(self) -> T: ...
    def uint8(self) -> T: ...

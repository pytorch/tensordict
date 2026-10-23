# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
from abc import ABCMeta
from collections.abc import Callable, Sequence
from dataclasses import InitVar
from typing import Any, ClassVar, Generic, Literal, overload, Type, TypeVar

if sys.version_info >= (3, 11):
    from typing import dataclass_transform, Self
else:
    from typing_extensions import dataclass_transform, Self

import torch
from tensordict._nestedkey import NestedKey
from tensordict.base import TensorDictBase
from tensordict.utils import DeviceType

_SetT = TypeVar("_SetT", contravariant=True)
_GetT = TypeVar("_GetT", covariant=True)

class _InitField(Generic[_SetT, _GetT]):
    # A descriptor-typed dataclass field: __init__ and assignment take a
    # _SetT, and an instance reads a _GetT, as the TensorDictBase properties
    # do (#3305). An InitVar field would make pyright reject the reads.
    def __get__(self, instance: object, owner: Any = None) -> _GetT: ...
    def __set__(self, instance: object, value: _SetT) -> None: ...

@dataclass_transform(kw_only_default=True)
class _TypedTensorDictMeta(ABCMeta):
    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        shadow: bool | None = None,
        frozen: bool | None = None,
        autocast: bool | None = None,
        nocast: bool | None = None,
        tensor_only: bool | None = None,
        **kwargs: Any,
    ) -> type: ...
    @overload
    def __getitem__(cls, item: Literal["shadow"]) -> Type["TypedTensorDict"]: ...
    @overload
    def __getitem__(cls, item: Literal["frozen"]) -> Type["TypedTensorDict"]: ...
    @overload
    def __getitem__(cls, item: Literal["autocast"]) -> Type["TypedTensorDict"]: ...
    @overload
    def __getitem__(cls, item: Literal["nocast"]) -> Type["TypedTensorDict"]: ...
    @overload
    def __getitem__(cls, item: Literal["tensor_only"]) -> Type["TypedTensorDict"]: ...
    @overload
    def __getitem__(
        cls,
        item: tuple[
            Literal["shadow", "frozen", "autocast", "nocast", "tensor_only"], ...
        ],
    ) -> Type["TypedTensorDict"]: ...

class TypedTensorDict(TensorDictBase, metaclass=_TypedTensorDictMeta):
    batch_size: _InitField[Sequence[int] | torch.Size | int | None, torch.Size] = ...
    device: _InitField[DeviceType | None, torch.device | None] = ...  # type: ignore[override]
    names: _InitField[Sequence[str] | None, Any] = ...
    non_blocking: InitVar[bool | None] = None
    lock: InitVar[bool] = False

    _shadow: ClassVar[bool]
    _frozen: ClassVar[bool]
    _autocast: ClassVar[bool]
    _nocast: ClassVar[bool]
    _tensor_only: ClassVar[bool]

    __expected_keys__: ClassVar[frozenset[str]]
    __required_keys__: ClassVar[frozenset[str]]
    __optional_keys__: ClassVar[frozenset[str]]
    __field_defaults__: ClassVar[dict[str, Any]]

    @classmethod
    def from_schema(
        cls,
        schema: dict[str, tuple[list[int] | torch.Size, torch.dtype]],
        *,
        batch_size: Sequence[int] | torch.Size | None = None,
        storage: str | None = None,
        device: DeviceType | None = None,
        **kwargs: Any,
    ) -> Self: ...
    @classmethod
    def from_tensordict(cls, td: TensorDictBase, *, check: bool = True) -> Self: ...
    @classmethod
    def from_dict(
        cls,
        input_dict,
        *,
        auto_batch_size: bool | None = None,
        batch_size: Sequence[int] | torch.Size | None = None,
        device: DeviceType | None = None,
        batch_dims: int | None = None,
        names: Sequence[str] | None = None,
        check: bool = True,
    ) -> Self: ...

    # TensorDictBase declares these methods abstract. TypedTensorDict and its
    # metaclass define them; most delegate to the tensordict that holds the data.
    def keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[type], bool] | None = None,
        *,
        sort: bool = False,
    ) -> Any: ...
    def entry_class(self, key: NestedKey) -> type: ...
    def _add_batch_dim(self, *args: Any, **kwargs: Any) -> Any: ...
    def _clone(self, *args: Any, **kwargs: Any) -> Any: ...
    def _get_str(self, *args: Any, **kwargs: Any) -> Any: ...
    def _maybe_remove_batch_dim(self, *args: Any, **kwargs: Any) -> Any: ...
    def _memmap_(self, *args: Any, **kwargs: Any) -> Any: ...
    def _permute(self, *args: Any, **kwargs: Any) -> Any: ...
    def _remove_batch_dim(self, *args: Any, **kwargs: Any) -> Any: ...
    def _set_at_str(self, *args: Any, **kwargs: Any) -> Any: ...
    def _set_at_tuple(self, *args: Any, **kwargs: Any) -> Any: ...
    def _set_str(self, *args: Any, **kwargs: Any) -> Any: ...
    def _set_tuple(self, *args: Any, **kwargs: Any) -> Any: ...
    def _squeeze(self, *args: Any, **kwargs: Any) -> Any: ...
    def _transpose(self, *args: Any, **kwargs: Any) -> Any: ...
    def _unsqueeze(self, *args: Any, **kwargs: Any) -> Any: ...
    def _view(self, *args: Any, **kwargs: Any) -> Any: ...

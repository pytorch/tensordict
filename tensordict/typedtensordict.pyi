# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
from collections.abc import Sequence
from dataclasses import InitVar
from typing import Any, ClassVar, Literal, overload, Type

if sys.version_info >= (3, 11):
    from typing import dataclass_transform, Self
else:
    from typing_extensions import dataclass_transform, Self

import torch
from tensordict._td import TensorDict
from tensordict.utils import DeviceType

@dataclass_transform(kw_only_default=True)
class _TypedTensorDictMeta(type):
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

class TypedTensorDict(TensorDict, metaclass=_TypedTensorDictMeta):
    batch_size: InitVar[Sequence[int] | torch.Size | int | None] = None
    device: InitVar[DeviceType | None] = None
    names: InitVar[Sequence[str] | None] = None
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

    def __init__(
        self,
        *,
        batch_size: Sequence[int] | torch.Size | int | None = None,
        device: DeviceType | None = None,
        names: Sequence[str] | None = None,
        non_blocking: bool | None = None,
        lock: bool = False,
        **kwargs: Any,
    ) -> None: ...
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

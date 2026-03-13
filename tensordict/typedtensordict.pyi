# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from typing import Any, dataclass_transform, Literal, overload, Type, TYPE_CHECKING

import torch
from tensordict._td import TensorDict
from tensordict.utils import DeviceType

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any

@dataclass_transform()
class _TypedTensorDictMeta(type(TensorDict)):
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
    def __getitem__(cls, item: Any) -> Type["TypedTensorDict"]: ...

class TypedTensorDict(TensorDict, metaclass=_TypedTensorDictMeta):
    _shadow: bool
    _frozen: bool
    _autocast: bool
    _nocast: bool
    _tensor_only: bool

    __expected_keys__: frozenset[str]
    __required_keys__: frozenset[str]
    __optional_keys__: frozenset[str]

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

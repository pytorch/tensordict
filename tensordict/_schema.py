# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Schema dataclasses for TensorDict structure description.

These frozen dataclasses represent the immutable structure of a TensorDict,
enabling torch.compile to use a single hash guard instead of per-key guards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class TensorSpec:
    """Specification for a single tensor within a TensorDict.

    This captures the immutable metadata of a tensor (dtype, device, non-batch shape)
    while allowing batch dimensions to remain dynamic.

    Attributes:
        dtype: The tensor's dtype (e.g., torch.float32).
        device: The tensor's device (e.g., torch.device("cuda:0")).
        shape_suffix: The non-batch dimensions of the tensor shape.
            For a tensor with shape [batch, time, 128, 64] and batch_dims=2,
            shape_suffix would be (128, 64).
    """

    dtype: torch.dtype
    device: torch.device
    shape_suffix: tuple[int, ...]

    def __hash__(self) -> int:
        # device needs special handling since torch.device isn't always hashable consistently
        device_key = (str(self.device),) if self.device is not None else (None,)
        return hash((self.dtype, device_key, self.shape_suffix))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorSpec):
            return NotImplemented
        return (
            self.dtype == other.dtype
            and str(self.device) == str(other.device)
            and self.shape_suffix == other.shape_suffix
        )


@dataclass(frozen=True)
class TensorDictSchema:
    """Schema describing the structure of a TensorDict.

    This frozen dataclass captures the complete structure of a TensorDict including:
    - All keys (sorted for deterministic ordering)
    - Nested TensorDict schemas
    - Tensor specifications (dtype, device, non-batch shape)
    - Number of batch dimensions (sizes are dynamic)
    - Device constraint

    When a TensorDict has a frozen schema, torch.compile can use a single
    hash guard on the schema instead of generating per-key guards for
    key presence, ordering, and tensor metadata.

    Attributes:
        keys: Sorted tuple of all keys at this level.
        nested_schemas: Tuple of (key, TensorDictSchema) pairs for nested TensorDicts.
        tensor_specs: Tuple of (key, TensorSpec) pairs for leaf tensors.
        batch_dims: Number of leading dimensions that are dynamic (batch/time).
        device: Device constraint for the TensorDict, or None if unconstrained.
    """

    keys: tuple[str, ...]
    nested_schemas: tuple[tuple[str, "TensorDictSchema"], ...]
    tensor_specs: tuple[tuple[str, TensorSpec], ...]
    batch_dims: int
    device: torch.device | None

    def __hash__(self) -> int:
        # device needs special handling
        device_key = str(self.device) if self.device is not None else None
        return hash(
            (
                self.keys,
                self.nested_schemas,
                self.tensor_specs,
                self.batch_dims,
                device_key,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorDictSchema):
            return NotImplemented
        # Compare device as strings for consistency
        self_device = str(self.device) if self.device is not None else None
        other_device = str(other.device) if other.device is not None else None
        return (
            self.keys == other.keys
            and self.nested_schemas == other.nested_schemas
            and self.tensor_specs == other.tensor_specs
            and self.batch_dims == other.batch_dims
            and self_device == other_device
        )

    def __repr__(self) -> str:
        lines = [f"TensorDictSchema(batch_dims={self.batch_dims}, device={self.device})"]
        lines.append(f"  keys: {self.keys}")
        if self.tensor_specs:
            lines.append("  tensors:")
            for key, spec in self.tensor_specs:
                lines.append(
                    f"    {key}: dtype={spec.dtype}, device={spec.device}, "
                    f"shape_suffix={spec.shape_suffix}"
                )
        if self.nested_schemas:
            lines.append("  nested:")
            for key, schema in self.nested_schemas:
                nested_repr = repr(schema).replace("\n", "\n    ")
                lines.append(f"    {key}: {nested_repr}")
        return "\n".join(lines)


# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, TYPE_CHECKING

import torch
from tensordict._nestedkey import NestedKey
from tensordict._tensorcollection import TensorCollection

if TYPE_CHECKING:
    from typing import Self

    from tensordict.base import T

__all__ = [
    "cat",
    "from_consolidated",
    "from_module",
    "from_modules",
    "from_pytree",
    "fromkeys",
    "lazy_stack",
    "load",
    "load_memmap",
    "maybe_dense_stack",
    "memmap",
    "save",
    "stack",
]


def _tensordict_cls():
    from tensordict._td import TensorDict

    return TensorDict


def from_module(
    module,
    as_module: bool = False,
    lock: bool = True,
    use_state_dict: bool = False,
):
    """Copies the params and buffers of a module in a tensordict."""
    return _tensordict_cls().from_module(
        module=module, as_module=as_module, lock=lock, use_state_dict=use_state_dict
    )


def from_modules(
    *modules,
    as_module: bool = False,
    lock: bool = True,
    use_state_dict: bool = False,
    lazy_stack: bool = False,
    expand_identical: bool = False,
):
    """Retrieves the parameters of several modules for vmap."""
    return _tensordict_cls().from_modules(
        *modules,
        lazy_stack=lazy_stack,
        expand_identical=expand_identical,
        lock=lock,
        use_state_dict=use_state_dict,
        as_module=as_module,
    )


def from_pytree(
    pytree,
    *,
    batch_size: torch.Size | None = None,
    auto_batch_size: bool = False,
    batch_dims: int | None = None,
):
    """Converts a pytree to a TensorDict instance."""
    return _tensordict_cls().from_pytree(
        pytree,
        batch_size=batch_size,
        auto_batch_size=auto_batch_size,
        batch_dims=batch_dims,
    )


def stack(input: Sequence["T"], dim: int = 0, *, out=None) -> "T":
    """Stacks tensordicts into a single tensordict along the given dimension."""
    return torch.stack(input, dim=dim, out=out)


def lazy_stack(input: Sequence["T"], dim: int = 0, *, out=None) -> "T":
    """Creates a lazy stack of tensordicts."""
    return _tensordict_cls().lazy_stack(input, dim=dim, out=out)


def cat(input: Sequence["T"], dim: int = 0, *, out=None) -> "T":
    """Concatenates tensordicts into a single tensordict along the given dimension."""
    return torch.cat(input, dim=dim, out=out)


def maybe_dense_stack(input: Sequence["T"], dim: int = 0, *, out=None, **kwargs) -> "T":
    """Attempts to make a dense stack and falls back on lazy stack when required."""
    return _tensordict_cls().maybe_dense_stack(input, dim=dim, out=out, **kwargs)


def fromkeys(keys: list[NestedKey], value: Any = 0):
    """Creates a tensordict from a list of keys and a single value."""
    return _tensordict_cls().fromkeys(keys=keys, value=value)


def from_consolidated(filename):
    """Reconstructs a tensordict from a consolidated file."""
    return _tensordict_cls().from_consolidated(filename)


def load(
    prefix: str | Path,
    device: torch.device | None = None,
    non_blocking: bool = False,
    *,
    out: TensorCollection | None = None,
    robust_key: bool | None = None,
) -> "Self":
    """Loads a tensordict from disk."""
    return load_memmap(
        prefix=prefix,
        device=device,
        non_blocking=non_blocking,
        out=out,
        robust_key=robust_key,
    )


def load_memmap(
    prefix: str | Path,
    device: torch.device | None = None,
    non_blocking: bool = False,
    *,
    out: TensorCollection | None = None,
    robust_key: bool | None = None,
) -> "Self":
    """Loads a memory-mapped tensordict from disk."""
    return _tensordict_cls().load_memmap(
        prefix=prefix,
        device=device,
        non_blocking=non_blocking,
        out=out,
        robust_key=robust_key,
    )


def save(
    data: TensorCollection,
    prefix: str | None = None,
    copy_existing: bool = False,
    *,
    num_threads: int = 0,
    return_early: bool = False,
    share_non_tensor: bool = False,
    robust_key: bool | None = None,
) -> None:
    """Saves the tensordict to disk."""
    return data.memmap(
        prefix=prefix,
        copy_existing=copy_existing,
        num_threads=num_threads,
        return_early=return_early,
        share_non_tensor=share_non_tensor,
        robust_key=robust_key,
    )


def memmap(
    data: TensorCollection,
    prefix: str | None = None,
    copy_existing: bool = False,
    *,
    num_threads: int = 0,
    return_early: bool = False,
    share_non_tensor: bool = False,
    robust_key: bool | None = None,
) -> "Self":
    """Writes all tensors onto memory-mapped tensors in a new tensordict."""
    return data.memmap(
        prefix=prefix,
        copy_existing=copy_existing,
        num_threads=num_threads,
        return_early=return_early,
        share_non_tensor=share_non_tensor,
        robust_key=robust_key,
    )


for _name in __all__:
    globals()[_name].__module__ = "tensordict._td"

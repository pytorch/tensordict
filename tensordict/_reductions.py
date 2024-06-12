# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import copyreg
from multiprocessing.reduction import ForkingPickler

import torch

from tensordict._lazy import LazyStackedTensorDict
from tensordict._td import TensorDict
from tensordict.base import (
    _NESTED_TENSORS_AS_LISTS_NONTENSOR,
    NO_DEFAULT,
)


def _rebuild_tensordict_files(cls, keys, rebuilds, args, device, shape, names):
    values = [
        rebuild(*_args) if callable(rebuild) else rebuild
        for rebuild, _args in zip(rebuilds, args)
    ]
    return cls(
        dict(zip(keys, values)),
        device=device,
        batch_size=shape,
        names=names,
        # _run_checks=False,
    )


def _rebuild_tensordict_files_consolidated(
    cls, metadata, rebuild, args, device, shape, names
):
    storage = rebuild(*args)
    non_tensor = metadata.pop("<DEL>non_tensors<DEL>")
    d = non_tensor
    for key, (dtype, local_shape, start, stop) in metadata.items():
        d[key] = storage[start:stop].view(dtype).view(local_shape)
    return cls.from_dict(d, batch_size=shape, device=device, names=names)


def _reduce_td(data: TensorDict):
    metadata = data.device, data.shape, data.names
    consolidated = getattr(data, "_consolidated", None)
    if consolidated:
        storage = consolidated["storage"]
        storge_metadata = consolidated["metadata"]
        rebuild, args = torch.multiprocessing.reductions.reduce_tensor(storage)
        return (
            _rebuild_tensordict_files_consolidated,
            (type(data), storge_metadata, rebuild, args) + metadata,
        )

    keys, values = data._items_list(
        True, True, is_leaf=_NESTED_TENSORS_AS_LISTS_NONTENSOR, collapse=True
    )
    rebuilds, args = zip(
        *(
            torch.multiprocessing.reductions.reduce_tensor(value)
            if isinstance(value, torch.Tensor)
            else (value.data, None)
            for value in values
        )
    )
    return (_rebuild_tensordict_files, (type(data), keys, rebuilds, args) + metadata)


ForkingPickler.register(TensorDict, _reduce_td)

copyreg.pickle(TensorDict, _reduce_td)


def _rebuild_lazytd_files(cls, keys, rebuilds, args, device, shape, names, stack_dim):
    n = shape[stack_dim]
    shape = torch.Size((b for i, b in enumerate(shape) if i != stack_dim))
    dim_name = names[stack_dim] if names is not None else None
    names = (
        [name for i, name in enumerate(names) if i != stack_dim]
        if names is not None
        else names
    )
    td = _rebuild_tensordict_files(
        TensorDict, keys, rebuilds, args, device, shape, names
    )
    return cls(
        *[td._get_str(str(i), default=NO_DEFAULT) for i in range(n)],
        stack_dim=stack_dim,
        stack_dim_name=dim_name,
    )


def _rebuild_lazytd_files_consolidated(
    cls, metadata, rebuild, args, device, shape, names, stack_dim
):
    n = shape[stack_dim]
    shape = torch.Size((b for i, b in enumerate(shape) if i != stack_dim))
    dim_name = names[stack_dim] if names is not None else None
    names = (
        [name for i, name in enumerate(names) if i != stack_dim]
        if names is not None
        else names
    )
    td = _rebuild_tensordict_files_consolidated(
        TensorDict, metadata, rebuild, args, device, shape, names
    )
    return cls(
        *[td._get_str(str(i), default=NO_DEFAULT) for i in range(n)],
        stack_dim=stack_dim,
        stack_dim_name=dim_name,
    )


def _reduce_lazytd(data: LazyStackedTensorDict):
    metadata = data.device, data.shape, data.names, data.stack_dim
    consolidated = getattr(data, "_consolidated", None)
    if consolidated:
        storage = consolidated["storage"]
        storge_metadata = consolidated["metadata"]
        rebuild, args = torch.multiprocessing.reductions.reduce_tensor(storage)
        return (
            _rebuild_lazytd_files_consolidated,
            (type(data), storge_metadata, rebuild, args) + metadata,
        )

    keys, values = data._items_list(
        True, True, is_leaf=_NESTED_TENSORS_AS_LISTS_NONTENSOR, collapse=True
    )
    rebuilds, args = zip(
        *(
            torch.multiprocessing.reductions.reduce_tensor(value)
            if isinstance(value, torch.Tensor)
            else (value.data, None)
            for value in values
        )
    )
    return (_rebuild_lazytd_files, (type(data), keys, rebuilds, args) + metadata)


ForkingPickler.register(LazyStackedTensorDict, _reduce_lazytd)

copyreg.pickle(LazyStackedTensorDict, _reduce_lazytd)

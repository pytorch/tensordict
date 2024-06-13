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

CLS_MAP = {
    "TensorDict": TensorDict,
    "LazyStackedTensorDict": LazyStackedTensorDict,
}


def _rebuild_tensordict_files(keys, rebuilds_args, metadata_dict):
    rebuilds = dict(zip(keys, rebuilds_args))

    def from_metadata(metadata=metadata_dict, prefix=None):
        non_tensor = metadata.pop("non_tensors")
        leaves = metadata.pop("leaves")
        cls = metadata.pop("cls")
        cls_metadata = metadata.pop("cls_metadata")

        d = non_tensor
        for key, _ in leaves.items():
            total_key = (key,) if prefix is None else prefix + (key,)
            rebuild, args = rebuilds[total_key]
            d[key] = rebuild(*args)
        for k, v in metadata.items():
            # Each remaining key is a tuple pointing to a sub-tensordict
            d[k] = from_metadata(
                v, prefix=prefix + (k,) if prefix is not None else (k,)
            )
        result = CLS_MAP[cls].from_dict(d, **cls_metadata)
        return result

    return from_metadata()


def _rebuild_tensordict_files_consolidated(
    metadata,
    rebuild,
    args,
):
    storage = rebuild(*args)

    def from_metadata(metadata=metadata, prefix=None):
        non_tensor = metadata.pop("non_tensors")
        leaves = metadata.pop("leaves")
        cls = metadata.pop("cls")
        cls_metadata = metadata.pop("cls_metadata")

        d = non_tensor
        for key, (dtype, local_shape, _, start, stop) in leaves.items():
            d[key] = storage[start:stop].view(dtype).view(local_shape)
        for k, v in metadata.items():
            # Each remaining key is a tuple pointing to a sub-tensordict
            d[k] = from_metadata(
                v, prefix=prefix + (k,) if prefix is not None else (k,)
            )
        result = CLS_MAP[cls].from_dict(d, **cls_metadata)
        return result

    return from_metadata()


def _reduce_td(data: TensorDict):
    consolidated = getattr(data, "_consolidated", None)
    if consolidated:
        storage = consolidated["storage"]
        storge_metadata = consolidated["metadata"]
        rebuild, args = torch.multiprocessing.reductions.reduce_tensor(storage)
        return (
            _rebuild_tensordict_files_consolidated,
            (storge_metadata, rebuild, args),
        )

    metadata_dict, flat_key_values, _ = data._reduce_vals_and_metadata()
    keys, rebuilds_args = zip(
        *(
            (key, torch.multiprocessing.reductions.reduce_tensor(value))
            for key, value in flat_key_values.items()
        )
    )
    return (_rebuild_tensordict_files, (keys, rebuilds_args, metadata_dict))


ForkingPickler.register(TensorDict, _reduce_td)

copyreg.pickle(TensorDict, _reduce_td)

ForkingPickler.register(LazyStackedTensorDict, _reduce_td)

copyreg.pickle(LazyStackedTensorDict, _reduce_td)

#
# def _rebuild_lazytd_files(cls, keys, rebuilds, args, device, shape, names, stack_dim):
#     n = shape[stack_dim]
#     shape = torch.Size((b for i, b in enumerate(shape) if i != stack_dim))
#     dim_name = names[stack_dim] if names is not None else None
#     names = (
#         [name for i, name in enumerate(names) if i != stack_dim]
#         if names is not None
#         else names
#     )
#     td = _rebuild_tensordict_files(
#         TensorDict, keys, rebuilds, args, device, shape, names
#     )
#     return cls(
#         *[td._get_str(str(i), default=NO_DEFAULT) for i in range(n)],
#         stack_dim=stack_dim,
#         stack_dim_name=dim_name,
#     )
#
#
# def _rebuild_lazytd_files_consolidated(
#     cls, metadata, rebuild, args, device, shape, names, stack_dim
# ):
#     n = shape[stack_dim]
#     shape = torch.Size((b for i, b in enumerate(shape) if i != stack_dim))
#     dim_name = names[stack_dim] if names is not None else None
#     names = (
#         [name for i, name in enumerate(names) if i != stack_dim]
#         if names is not None
#         else names
#     )
#     td = _rebuild_tensordict_files_consolidated(
#         TensorDict, metadata, rebuild, args, device, shape, names
#     )
#     return cls(
#         *[td._get_str(str(i), default=NO_DEFAULT) for i in range(n)],
#         stack_dim=stack_dim,
#         stack_dim_name=dim_name,
#     )
#
#
# def _reduce_lazytd(data: LazyStackedTensorDict):
#     metadata = data.device, data.shape, data.names, data.stack_dim
#     consolidated = getattr(data, "_consolidated", None)
#     if consolidated:
#         storage = consolidated["storage"]
#         storge_metadata = consolidated["metadata"]
#         rebuild, args = torch.multiprocessing.reductions.reduce_tensor(storage)
#         return (
#             _rebuild_lazytd_files_consolidated,
#             (type(data), storge_metadata, rebuild, args) + metadata,
#         )
#
#     metadata_dict, flat_key_values, _ = data._reduce_vals_and_metadata()
#     rebuilds, args = zip(
#         *(
#             torch.multiprocessing.reductions.reduce_tensor(value)
#             if isinstance(value, torch.Tensor)
#             else (value.data, None)
#             for value in values
#         )
#     )
#     return (_rebuild_lazytd_files, (type(data), keys, rebuilds, args) + metadata)
#
#
# ForkingPickler.register(LazyStackedTensorDict, _reduce_lazytd)
#
# copyreg.pickle(LazyStackedTensorDict, _reduce_lazytd)

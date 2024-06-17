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
from tensordict.utils import _nt_from_tensor_shape, _STRDTYPE2DTYPE

CLS_MAP = {
    "TensorDict": TensorDict,
    "LazyStackedTensorDict": LazyStackedTensorDict,
}


def _rebuild_tensordict_files(flat_key_values, metadata_dict):
    def from_metadata(metadata=metadata_dict, prefix=None):
        non_tensor = metadata.pop("non_tensors")
        leaves = metadata.pop("leaves")
        cls = metadata.pop("cls")
        cls_metadata = metadata.pop("cls_metadata")
        is_locked = cls_metadata.pop("is_locked", False)

        d = non_tensor
        for key, _ in leaves.items():
            total_key = (key,) if prefix is None else prefix + (key,)
            if total_key[-1].startswith("<NJT>") or total_key[-1].startswith("<NT>"):
                nested_values = flat_key_values[total_key]
                continue
            if total_key[-1].startswith("<NJT_OFFSETS"):
                offsets = flat_key_values[total_key]
                key = key.replace("<NJT_OFFSETS>", "")
                value = torch.nested.nested_tensor_from_jagged(nested_values, offsets)
                del nested_values
            elif total_key[-1].startswith("<NT_SHAPES"):
                shapes = flat_key_values[total_key]
                key = key.replace("<NT_SHAPES>", "")
                value = _nt_from_tensor_shape(nested_values, shapes)
                del nested_values
            else:
                value = flat_key_values[total_key]
            d[key] = value
        for k, v in metadata.items():
            # Each remaining key is a tuple pointing to a sub-tensordict
            d[k] = from_metadata(
                v, prefix=prefix + (k,) if prefix is not None else (k,)
            )
        result = CLS_MAP[cls].from_dict(d, **cls_metadata)
        if is_locked:
            result.lock_()
        return result

    return from_metadata()


def _rebuild_tensordict_files_consolidated(
    metadata,
    storage,
):
    def from_metadata(metadata=metadata, prefix=None):
        non_tensor = metadata.pop("non_tensors")
        leaves = metadata.pop("leaves")
        cls = metadata.pop("cls")
        cls_metadata = metadata.pop("cls_metadata")
        is_locked = cls_metadata.pop("is_locked", False)
        # size can be there to tell what the size of the file is
        _ = metadata.pop("size", None)

        d = non_tensor
        for key, (dtype, local_shape, _, start, stop, pad) in leaves.items():
            dtype = _STRDTYPE2DTYPE[dtype]
            # device = torch.device(device)
            local_shape = torch.Size(local_shape)
            value = storage[start:stop].view(dtype)
            if value.numel() > local_shape.numel():
                print(pad, value.numel(), local_shape.numel())
                value = value[: local_shape.numel()]
            value = value.view(local_shape)
            if key.startswith("<NJT>") or key.startswith("<NT>"):
                nested_values = value
                continue
            elif key.startswith("<NJT_OFFSETS>"):
                offsets = value
                value = torch.nested.nested_tensor_from_jagged(nested_values, offsets)
                key = key.replace("<NJT_OFFSETS>", "")
            elif key.startswith("<NT_SHAPES>"):
                shapes = value
                value = _nt_from_tensor_shape(nested_values, shapes)
                key = key.replace("<NT_SHAPES>", "")
            d[key] = value
        for k, v in metadata.items():
            # Each remaining key is a tuple pointing to a sub-tensordict
            d[k] = from_metadata(
                v, prefix=prefix + (k,) if prefix is not None else (k,)
            )
        result = CLS_MAP[cls].from_dict(d, **cls_metadata)
        if is_locked:
            result = result.lock_()
        return result

    return from_metadata()


def _reduce_td(data: TensorDict):
    consolidated = getattr(data, "_consolidated", None)
    if consolidated:
        storage = consolidated["storage"]
        storge_metadata = consolidated["metadata"]
        return (
            _rebuild_tensordict_files_consolidated,
            (storge_metadata, storage),
        )

    metadata_dict, flat_key_values, _, _ = data._reduce_vals_and_metadata()
    return (_rebuild_tensordict_files, (flat_key_values, metadata_dict))


ForkingPickler.register(TensorDict, _reduce_td)

copyreg.pickle(TensorDict, _reduce_td)

ForkingPickler.register(LazyStackedTensorDict, _reduce_td)

copyreg.pickle(LazyStackedTensorDict, _reduce_td)

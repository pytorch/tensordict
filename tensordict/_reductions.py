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

from tensordict.tensorclass import NonTensorData
from tensordict.utils import _STRDTYPE2DTYPE

CLS_MAP = {
    "TensorDict": TensorDict,
    "LazyStackedTensorDict": LazyStackedTensorDict,
}


def _rebuild_tensordict_files(flat_key_values, metadata_dict, is_shared: bool = False):
    def from_metadata(metadata=metadata_dict, prefix=None):
        non_tensor = metadata.pop("non_tensors")
        leaves = metadata.pop("leaves")
        cls = metadata.pop("cls")
        cls_metadata = metadata.pop("cls_metadata")
        is_locked = cls_metadata.pop("is_locked", False)

        d = {
            key: NonTensorData(data, batch_size=batch_size)
            for (key, (data, batch_size)) in non_tensor.items()
        }
        for key, _ in leaves.items():
            total_key = (key,) if prefix is None else prefix + (key,)
            if total_key[-1].startswith("<NJT>"):
                nested_values = flat_key_values[total_key]
                continue
            if total_key[-1].startswith("<NJT_OFFSETS"):
                offsets = flat_key_values[total_key]
                key = key.replace("<NJT_OFFSETS>", "")
                value = torch.nested.nested_tensor_from_jagged(nested_values, offsets)
                del nested_values
            else:
                value = flat_key_values[total_key]
            d[key] = value
        for k, v in metadata.items():
            # Each remaining key is a tuple pointing to a sub-tensordict
            d[k] = from_metadata(
                v, prefix=prefix + (k,) if prefix is not None else (k,)
            )
        result = CLS_MAP[cls]._from_dict_validated(d, **cls_metadata)
        if is_locked:
            result.lock_()
        # if is_shared:
        #     result._is_shared = is_shared
        return result

    return from_metadata()


def _rebuild_tensordict_files_shared(flat_key_values, metadata_dict):
    return _rebuild_tensordict_files(flat_key_values, metadata_dict, is_shared=True)


def _rebuild_tensordict_files_consolidated(
    metadata,
    storage,
):
    def from_metadata(metadata=metadata, prefix=None):
        metadata = dict(metadata)
        non_tensor = metadata.pop("non_tensors")
        leaves = metadata.pop("leaves")
        cls = metadata.pop("cls")
        cls_metadata = metadata.pop("cls_metadata")
        is_locked = cls_metadata.pop("is_locked", False)
        # size can be there to tell what the size of the file is
        _ = metadata.pop("size", None)

        d = {
            key: NonTensorData(data, batch_size=batch_size)
            for (key, (data, batch_size)) in non_tensor.items()
        }
        for key, (dtype, local_shape, _, start, stop, pad) in leaves.items():
            dtype = _STRDTYPE2DTYPE[dtype]
            # device = torch.device(device)
            local_shape = torch.Size(local_shape)
            value = storage[start:stop].view(dtype)
            if pad:
                value = value[: local_shape.numel()]
            value = value.view(local_shape)
            if key.startswith("<NJT>"):
                nested_values = value
                continue
            elif key.startswith("<NJT_OFFSETS>"):
                offsets = value
                value = torch.nested.nested_tensor_from_jagged(nested_values, offsets)
                key = key.replace("<NJT_OFFSETS>", "")
            d[key] = value
        for k, v in metadata.items():
            # Each remaining key is a tuple pointing to a sub-tensordict
            d[k] = from_metadata(
                v, prefix=prefix + (k,) if prefix is not None else (k,)
            )
        result = CLS_MAP[cls]._from_dict_validated(d, **cls_metadata)
        # result._is_shared = storage.is_shared()
        if is_locked:
            result = result.lock_()
        return result

    return from_metadata()

def _make_td(cls, state):
    td = cls.__new__(cls)
    for key, val in state.items():
        setattr(td, key, val)
    return td


def _reduce_td(data: TensorDict):
    consolidated = getattr(data, "_consolidated", None)
    if consolidated and consolidated["metadata"] is not None:
        storage = consolidated["storage"]
        storge_metadata = consolidated["metadata"]
        return (
            _rebuild_tensordict_files_consolidated,
            (storge_metadata, storage),
        )
    # The reason we can't use this is that pytorch unpickler requires the dtypes to match for a single
    # storage.
    # Checking the dtype locally doesn't work because this reduction is also being called for sub-tds
    # of bigger consolidated TDs where the dtypes mismatch globally, but not locally.
    # Note also that we could say that for non-consolidated TDs we'll just use the regular reduce
    # but if one does `pipe.send(td.consolidate()[0])`, the td being sent over is consolidated but
    # lacks the _consolidated field, so this will fall back on regular pickle and PT will complain about
    # the storage pointing to tensors of different dtypes.
    # return (_make_td, (type(data), data.__getstate__(),))

    metadata_dict, flat_key_values, _, _ = data._reduce_vals_and_metadata(
        requires_metadata=True
    )
    return (_rebuild_tensordict_files, (flat_key_values, metadata_dict))


ForkingPickler.register(TensorDict, _reduce_td)

copyreg.pickle(TensorDict, _reduce_td)

ForkingPickler.register(LazyStackedTensorDict, _reduce_td)

copyreg.pickle(LazyStackedTensorDict, _reduce_td)

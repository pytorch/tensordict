# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import copyreg
from multiprocessing import reduction

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
        for key in leaves.keys():
            total_key = (key,) if prefix is None else prefix + (key,)
            if total_key[-1].startswith("<NJT>"):
                nested_values = flat_key_values[total_key]
                nested_lengths = None
                continue
            if total_key[-1].startswith("<NJT_LENGTHS>"):
                nested_lengths = flat_key_values[total_key]
                continue
            elif total_key[-1].startswith("<NJT_OFFSETS"):
                offsets = flat_key_values[total_key]
                key = key.replace("<NJT_OFFSETS>", "")
                value = torch.nested.nested_tensor_from_jagged(
                    nested_values, offsets=offsets, lengths=nested_lengths
                )
                del nested_values
                del nested_lengths
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
        consolidated = {"storage": storage, "metadata": metadata}
        metadata = dict(metadata)
        non_tensor = metadata.pop("non_tensors")
        leaves = metadata.pop("leaves")
        cls = metadata.pop("cls")
        cls_metadata = dict(metadata.pop("cls_metadata"))
        is_locked = cls_metadata.pop("is_locked", False)
        # size can be there to tell what the size of the file is
        _ = metadata.pop("size", None)

        d = {
            key: NonTensorData(data, batch_size=batch_size)
            for (key, (data, batch_size)) in non_tensor.items()
        }
        for key, (dtype, local_shape, start, stop, pad) in leaves.items():
            dtype = _STRDTYPE2DTYPE[dtype]
            # device = torch.device(device)
            local_shape = torch.Size(local_shape)
            value = storage[start:stop].view(dtype)
            if pad:
                value = value[: local_shape.numel()]
            value = value.view(local_shape)
            if key.startswith("<NJT>"):
                raise RuntimeError
            elif key.startswith("<NJT_VALUES>"):
                nested_values = value
                nested_lengths = None
                continue
            elif key.startswith("<NJT_LENGTHS>"):
                nested_lengths = value
                continue
            elif key.startswith("<NJT_OFFSETS>"):
                from torch.nested._internal.nested_tensor import NestedTensor

                offsets = value
                value = NestedTensor(
                    nested_values, offsets=offsets, lengths=nested_lengths
                )
                key = key.replace("<NJT_OFFSETS>", "")
            d[key] = value
        for k, v in metadata.items():
            # Each remaining key is a tuple pointing to a sub-tensordict
            d[k] = from_metadata(
                v, prefix=prefix + (k,) if prefix is not None else (k,)
            )
        result = CLS_MAP[cls]._from_dict_validated(d, **cls_metadata)
        if is_locked:
            result = result.lock_()
        result._consolidated = consolidated
        return result

    return from_metadata()


def _make_td(cls, state):
    td = cls.__new__(cls)
    td.__setstate__(state)
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

    # This is faster than the solution below.
    return (
        _make_td,
        (
            type(data),
            data.__getstate__(),
        ),
    )
    # metadata_dict, flat_key_values, _, _ = data._reduce_vals_and_metadata(
    #     requires_metadata=True
    # )
    # return (_rebuild_tensordict_files, (flat_key_values, metadata_dict))


reduction.register(TensorDict, _reduce_td)

copyreg.pickle(TensorDict, _reduce_td)

reduction.register(LazyStackedTensorDict, _reduce_td)

copyreg.pickle(LazyStackedTensorDict, _reduce_td)

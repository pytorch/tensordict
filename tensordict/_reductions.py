# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import copyreg
import queue
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
    _nt_values_and_keys = queue.Queue()
    _nt_lengths = queue.Queue()
    _nt_offsets = queue.Queue()

    def from_metadata(metadata=metadata_dict, prefix=None):
        metadata = dict(metadata)

        _ = metadata.pop("njt_values_start", None)
        _ = metadata.pop("njt_lengths_start", None)
        _ = metadata.pop("njt_offsets_start", None)

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
                total_key = total_key[:-1] + total_key[-1].replace("<NJT>", "")
                _nt_values_and_keys.put((nested_values, total_key))
                continue
            if total_key[-1].startswith("<NJT_LENGTHS>"):
                nested_lengths = flat_key_values[total_key]
                _nt_lengths.put(nested_lengths)
                continue
            elif total_key[-1].startswith("<NJT_OFFSETS"):
                offsets = flat_key_values[total_key]
                _nt_offsets.put(offsets)
                continue
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

    result = from_metadata()
    # Then assign the nested tensors
    while not _nt_values_and_keys.empty():
        vals, key = _nt_values_and_keys.get()
        lengths = _nt_lengths.get()
        offsets = _nt_offsets.get()
        value = torch.nested.nested_tensor_from_jagged(
            vals, offsets=offsets, lengths=lengths
        )
        result._set_tuple(key, value, inplace=False, validated=True)

    return result


def _rebuild_tensordict_files_shared(flat_key_values, metadata_dict):
    return _rebuild_tensordict_files(flat_key_values, metadata_dict, is_shared=True)


def _rebuild_tensordict_files_consolidated(
    metadata,
    storage,
):
    _nt_values_and_keys = queue.Queue()
    _nt_lengths = queue.Queue()
    _nt_offsets = queue.Queue()

    def from_metadata(metadata=metadata, prefix=None):
        consolidated = {"storage": storage, "metadata": metadata}
        metadata = dict(metadata)

        _ = metadata.pop("njt_values_start", None)
        _ = metadata.pop("njt_lengths_start", None)
        _ = metadata.pop("njt_offsets_start", None)

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
                key = key.replace("<NJT_VALUES>", "")
                if prefix:
                    total_key = prefix + (key,)
                else:
                    total_key = (key,)
                _nt_values_and_keys.put((value, total_key))
                continue
            elif key.startswith("<NJT_LENGTHS>"):
                _nt_lengths.put(value)
                continue
            elif key.startswith("<NJT_OFFSETS>"):
                _nt_offsets.put(value)
                if _nt_offsets.qsize() > _nt_lengths.qsize():
                    _nt_lengths.put(None)
                continue
            d[key] = value
        for key, val in metadata.items():
            # Each remaining key is a tuple pointing to a sub-tensordict
            d[key] = from_metadata(
                val, prefix=prefix + (key,) if prefix is not None else (key,)
            )
        result = CLS_MAP[cls]._from_dict_validated(d, **cls_metadata)
        if is_locked:
            result = result.lock_()
        result._consolidated = consolidated
        return result

    result = from_metadata()
    # Then assign the nested tensors
    while not _nt_values_and_keys.empty():
        vals, key = _nt_values_and_keys.get()
        lengths = _nt_lengths.get()
        offsets = _nt_offsets.get()
        value = torch.nested.nested_tensor_from_jagged(
            vals, offsets=offsets, lengths=lengths
        )
        result._set_tuple(key, value, inplace=False, validated=True)

    return result


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


ForkingPickler.register(TensorDict, _reduce_td)

copyreg.pickle(TensorDict, _reduce_td)

ForkingPickler.register(LazyStackedTensorDict, _reduce_td)

copyreg.pickle(LazyStackedTensorDict, _reduce_td)

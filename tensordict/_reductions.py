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


def _rebuild_tensordict_files(flat_key_values, metadata_dict):

    def from_metadata(metadata=metadata_dict, prefix=None):
        non_tensor = metadata.pop("non_tensors")
        leaves = metadata.pop("leaves")
        cls = metadata.pop("cls")
        cls_metadata = metadata.pop("cls_metadata")

        d = non_tensor
        for key, _ in leaves.items():
            total_key = (key,) if prefix is None else prefix + (key,)
            d[key] = flat_key_values[total_key]
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
    storage,
):

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
        return (
            _rebuild_tensordict_files_consolidated,
            (storge_metadata, storage),
        )

    metadata_dict, flat_key_values, _ = data._reduce_vals_and_metadata()
    return (_rebuild_tensordict_files, (flat_key_values, metadata_dict))


ForkingPickler.register(TensorDict, _reduce_td)

copyreg.pickle(TensorDict, _reduce_td)

ForkingPickler.register(LazyStackedTensorDict, _reduce_td)

copyreg.pickle(LazyStackedTensorDict, _reduce_td)

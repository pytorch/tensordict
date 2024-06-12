# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import copyreg
from multiprocessing.reduction import ForkingPickler

import torch
from tensordict._td import TensorDict


def _rebuild_tensordict_files(cls, keys, rebuilds, args, device, shape, names):
    values = [rebuild(*_args) for rebuild, _args in zip(rebuilds, args)]
    return cls(
        dict(zip(keys, values)),
        device=device,
        batch_size=shape,
        names=names,
        _run_checks=False,
    )


def _rebuild_tensordict_files_consolidated(
    cls, metadata, rebuild, args, device, shape, names
):
    storage = rebuild(*args)
    d = {}
    for key, (dtype, local_shape, start, stop) in metadata.items():
        d[key] = storage[start:stop].view(dtype).view(local_shape)
    return cls.from_dict(d, batch_size=shape, device=device, names=names)


def _reduce_td(data):
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

    keys, values = data._items_list(True, True)
    rebuilds, args = zip(
        *(torch.multiprocessing.reductions.reduce_tensor(value) for value in values)
    )
    return (_rebuild_tensordict_files, (type(data), keys, rebuilds, args) + metadata)


ForkingPickler.register(TensorDict, _reduce_td)

copyreg.pickle(TensorDict, _reduce_td)

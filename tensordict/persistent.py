# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Persistent tensordicts (H5 and others)."""

import h5py
import torch
from tensordict.tensordict import CompatibleType, NO_DEFAULT, TensorDictBase
from tensordict.utils import IndexType


class PersistentTensorDict(TensorDictBase):
    """Persistent TensorDict implementation.

    Args:
        batch_size (torch.Size or compatible): the tensordict batch size.
        filename (str, optional):
        group (h5py.Group, optional):
        mode (str, optional):
        backend (str, optional):
        device (torch.device or compatible, optional):

    """

    def __init__(
        self,
        *,
        batch_size,
        filename=None,
        group=None,
        mode="r",
        backend="h5",
        device=None,
    ):
        super().__init__()
        if backend != "h5":
            raise NotImplementedError
        if filename is not None and group is None:
            self.file = h5py.File(filename, mode)
        elif group is not None:
            self.file = group
        else:
            raise RuntimeError
        self._batch_size = torch.Size(batch_size)
        self._device = device
        self._is_shared = False
        self._check_batch_size(self._batch_size)

    def _check_batch_size(self, batch_size) -> None:
        for key in self.keys(True, True):
            if isinstance(key, str):
                pass
            elif len(key) == 1:
                key = key[0]
            else:
                key = "/".join(key)
            size = self.file[key].shape
            if torch.Size(size[: len(batch_size)]) != batch_size:
                raise RuntimeError(
                    f"batch size and array size mismatch: array.shape={size}, batch_size={batch_size}."
                )

    def get(self, key, default=NO_DEFAULT):
        if isinstance(key, tuple):
            key = "/".join(key)
        array = self.file[key]
        if isinstance(array, (h5py.Dataset,)):
            if self.device is not None:
                device = self.device
            else:
                device = torch.device("cpu")
            return torch.as_tensor(array, device=device)
        else:
            return PersistentTensorDict(group=array, batch_size=self.batch_size)

    def get_at(
        self, key: str, idx: IndexType, default: CompatibleType = NO_DEFAULT
    ) -> CompatibleType:
        if isinstance(key, tuple):
            key = "/".join(key)
        array = self.file[key]
        if isinstance(array, (h5py.Dataset,)):
            if self.device is not None:
                device = self.device
            else:
                device = torch.device("cpu")
            # indexing must be done before converting to tensor.
            # `get_at` is there to save us.
            return torch.as_tensor(array[idx], device=device)
        else:
            return PersistentTensorDict(
                group=array, batch_size=self.batch_size
            ).get_sub_tensordict(idx)

    def __getitem__(self, item):
        if isinstance(item, str) or (
            isinstance(item, tuple) and all(isinstance(val, str) for val in item)
        ):
            return self.get(item)
        return self.get_sub_tensordict(item)

    def keys(self, include_nested=False, leaves_only=False):
        if include_nested:
            if leaves_only:
                for key in self.file.visit(lambda key: key.split("/")):
                    if isinstance(self.file[key], (h5py.Dataset,)):
                        yield key
            else:
                yield from self.file.visit(lambda key: key.split("/"))
        else:
            yield from self.file.keys()

    def _change_batch_size(self):
        raise NotImplementedError

    def _stack_onto_(self):
        raise NotImplementedError

    @property
    def batch_size(self):
        return self._batch_size

    def contiguous(self):
        raise NotImplementedError

    def del_(self):
        raise NotImplementedError

    def detach_(self):
        raise NotImplementedError

    @property
    def device(self):
        return self._device

    def entry_class(self):
        raise NotImplementedError

    def is_contiguous(self):
        raise NotImplementedError

    def masked_fill(self):
        raise NotImplementedError

    def masked_fill_(self):
        raise NotImplementedError

    def memmap_(self):
        raise NotImplementedError

    def pin_memory(self):
        raise NotImplementedError

    def rename_key(self):
        raise NotImplementedError

    def select(self):
        raise NotImplementedError

    def set_(self):
        raise NotImplementedError

    def set_at_(self):
        raise NotImplementedError

    def share_memory_(self):
        raise NotImplementedError

    def to(self):
        raise NotImplementedError

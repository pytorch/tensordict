from __future__ import annotations

import functools
import mmap
import os

import sys
import tempfile
import warnings
from copy import copy, deepcopy
from multiprocessing import util
from multiprocessing.context import reduction
from pathlib import Path
from sys import getrefcount
from tempfile import _TemporaryFileWrapper
from typing import Any, Callable, Sequence

import numpy as np
import torch

from tensordict.utils import (
    _getitem_batch_size,
    convert_ellipsis_to_idx,
    DeviceType,
    IndexType,
    NUMPY_TO_TORCH_DTYPE_DICT,
    prod,
    TORCH_TO_NUMPY_DTYPE_DICT,
)
from torch.multiprocessing.reductions import ForkingPickler


class MemoryMappedTensor(torch.Tensor):
    filename: str | Path
    handler: FileHandler
    _clear: bool
    index: Any
    parent_shape: torch.Size

    def __new__(cls, tensor_or_file, handler=None, dtype=None, shape=None, index=None):
        if isinstance(tensor_or_file, str):
            return cls.from_filename(
                tensor_or_file,
                dtype,
                shape,
                index,
            )
        elif handler is not None:
            return cls.from_handler(
                handler,
                dtype,
                shape,
                index,
            )
        return super().__new__(cls, tensor_or_file)

    def __init__(self, tensor_or_file, handler=None, dtype=None, shape=None):
        ...

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def from_tensor(
        cls, tensor, transfer_ownership=False, dir=None, prefix=None, filename=None
    ):
        if isinstance(tensor, MemoryMappedTensor):
            if transfer_ownership:
                raise RuntimeError(
                    "from_tensor(memmap_tensor, transfer_ownership=True) is not permitted, as this method will "
                    "simply return the original MemmapTensor instance."
                )
            elif dir is None and (
                filename is None
                or Path(filename).absolute() == Path(tensor.filename).absolute()
            ):
                # either location was not specified, or memmap is already in the
                # correct location, so just return the MemmapTensor unmodified
                return tensor
        elif isinstance(tensor, np.ndarray):
            raise TypeError(
                "Convert input to torch.Tensor before calling MemoryMappedTensor.from_tensor."
            )
        if tensor.requires_grad:
            raise RuntimeError(
                "MemoryMappedTensor.from_tensor is incompatible with tensor.requires_grad."
            )
        shape = tensor.shape
        if filename is None:
            if tensor.dtype.is_floating_point:
                size = torch.finfo(tensor.dtype).bits // 8 * shape.numel()
            elif tensor.dtype.is_complex:
                raise ValueError(
                    "Complex-valued tensors are not supported by MemoryMappedTensor."
                )
            elif tensor.dtype == torch.bool:
                size = shape.numel()
            else:
                # assume integer
                size = torch.iinfo(tensor.dtype).bits // 8 * shape.numel()
            handler = FileHandler(size)
            out = torch.frombuffer(memoryview(handler.buffer), dtype=tensor.dtype)
            out = torch.reshape(out, shape)
            out = cls(out)
        else:
            handler = None
            out = cls(
                torch.from_file(
                    filename, shared=True, dtype=tensor.dtype, size=shape.numel()
                ).view(tensor.shape)
            )
        out.handler = handler
        out.filename = filename
        out.index = None
        out.parent_shape = tensor.shape
        out.copy_(tensor)
        return out

    # def __setstate__(self, state: dict[str, Any]) -> None:
    #     filename = state["filename"]
    #     handler = state['handler']
    #     if filename is not None:
    #         return self.from_filename(filename, state['dtype'], state['shape'])
    #     else:
    #         return self.from_handler(handler, state['dtype'], state['shape'])

    @classmethod
    def from_filename(cls, filename, dtype, shape, index):
        tensor = torch.from_file(
            filename, shared=True, dtype=dtype, size=shape.numel()
        ).view(shape)
        if index is not None:
            tensor = tensor[index]
        out = cls(tensor)
        out.filename = filename
        out.handler = None
        out.index = index
        out.parent_shape = shape
        return out

    @classmethod
    def from_handler(cls, handler, dtype, shape, index):
        out = torch.frombuffer(memoryview(handler.buffer), dtype=dtype)
        out = torch.reshape(out, shape)
        if index is not None:
            out = out[index]
        out = cls(out)
        out.filename = None
        out.handler = handler
        out.index = index
        out.parent_shape = shape
        return out

    def __reduce__(self):
        if getattr(self, "handler", None) is not None:
            return type(self).from_handler, (
                self.handler,
                self.dtype,
                self.parent_shape,
                self.index,
            )
        elif getattr(self, "filename", None) is not None:
            return type(self).from_filename, (
                self.filename,
                self.dtype,
                self.parent_shape,
                self.index,
            )
        else:
            raise RuntimeError

    def __getitem__(self, item):
        out = super().__getitem__(item)
        if out.data_ptr() == self.data_ptr():
            out = MemoryMappedTensor(out)
            assert isinstance(out, MemoryMappedTensor)
            out.handler = self.handler
            out.filename = self.filename
            out.index = item
            out.parent_shape = self.parent_shape
        return out


class FileHandler:
    if sys.platform == "linux":
        _dir_candidates = ["/dev/shm"]
    else:
        _dir_candidates = []

    def __init__(self, size, fd=-1, filename=None):
        # borrowed from mp.heap
        self.size = size
        # if filename is None:
        if fd == -1:
            self.fd, name = tempfile.mkstemp(
                prefix="pym-%d-" % os.getpid(), dir=self._choose_dir(size)
            )
            # self.filename = name
            os.unlink(name)
            util.Finalize(self, os.close, (self.fd,))
            os.ftruncate(self.fd, size)
        else:
            self.fd = fd
        # else:
        #     self.filename = filename
        self.buffer = mmap.mmap(self.fd, self.size)

    def _choose_dir(self, size):
        # Choose a non-storage backed directory if possible,
        # to improve performance
        for d in self._dir_candidates:
            st = os.statvfs(d)
            if st.f_bavail * st.f_frsize >= size:  # enough free space?
                return d
        tmpdir = util.get_temp_dir()
        return tmpdir


def reduce_handler(handler):
    if handler.fd == -1:
        raise ValueError(
            "Handler is unpicklable because " "forking was enabled when it was created"
        )
    return rebuild_handler, (handler.size, reduction.DupFd(handler.fd))


def rebuild_handler(size, dupfd):
    detached = dupfd.detach()
    return FileHandler(size, detached)


reduction.register(FileHandler, reduce_handler)


def reduce_memmap(memmap_tensor):
    return memmap_tensor.__reduce__()


ForkingPickler.register(MemoryMappedTensor, reduce_memmap)

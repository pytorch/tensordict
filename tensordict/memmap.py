# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools

import mmap
import os

import sys
import tempfile
from multiprocessing import reduction, util
from pathlib import Path
from typing import Any, Callable, overload

import numpy as np
import torch

from tensordict.utils import _shape, implement_for, IndexType, NESTED_TENSOR_ERR


class MemoryMappedTensor(torch.Tensor):
    """A Memory-mapped Tensor.

    Supports filenames or file handlers.

    The main advantage of MemoryMappedTensor resides in its serialization methods,
    which ensure that the tensor is passed through queues or RPC remote calls without
    any copy.

    .. note::
        When used within RPC settings, the filepath should be accessible to both nodes.
        If it isn't the behaviour of passing a MemoryMappedTensor from one worker
        to another is undefined.

    MemoryMappedTensor supports multiple construction methods.

    Examples:
          >>> # from an existing tensor
          >>> tensor = torch.randn(3)
          >>> with tempfile.NamedTemporaryFile() as file:
          ...     memmap_tensor = MemoryMappedTensor.from_tensor(tensor, filename=file.name)
          ...     assert memmap_tensor.filename is not None
          >>> # if no filename is passed, a handler is used
          >>> tensor = torch.randn(3)
          >>> memmap_tensor = MemoryMappedTensor.from_tensor(tensor, filename=file.name)
          >>> assert memmap_tensor.filename is None
          >>> # one can create an empty tensor too
          >>> with tempfile.NamedTemporaryFile() as file:
          ...     memmap_tensor_empty = MemoryMappedTensor.empty_like(tensor, filename=file.name)
          >>> with tempfile.NamedTemporaryFile() as file:
          ...     memmap_tensor_zero = MemoryMappedTensor.zeros_like(tensor, filename=file.name)
          >>> with tempfile.NamedTemporaryFile() as file:
          ...     memmap_tensor = MemoryMappedTensor.ones_like(tensor, filename=file.name)
    """

    _filename: str | Path = None
    _handler: _FileHandler = None
    _clear: bool
    index: Any
    parent_shape: torch.Size

    def __new__(
        cls,
        source,
        *,
        dtype=None,
        shape=None,
        index=None,
        device=None,
        handler=None,
        filename=None,
    ):
        if device is not None and torch.device(device).type != "cpu":
            raise ValueError(f"{cls} device must be cpu!")
        if isinstance(source, str):
            if filename is not None:
                raise TypeError("Duplicated filename argument.")
            filename = source
            source = None
        if filename is not None:
            if dtype is not None:
                raise TypeError("Cannot pass new dtype if source is provided.")
            result = cls.from_tensor(
                torch.as_tensor(source),
                filename=filename,
                # dtype=dtype,
                shape=shape,
                # index=index,
            )
            if index is not None:
                return result[index]
            return result
        elif isinstance(source, torch.StorageBase):
            return cls.from_storage(
                source,
                dtype=dtype,
                shape=shape,
                index=index,
                device=device,
                handler=handler,
                filename=filename,
            )
        elif handler is not None:
            return cls.from_handler(
                handler,
                dtype,
                shape,
                index,
            )
        return super().__new__(cls, source)

    def __init__(
        self,
        source,
        *,
        handler=None,
        dtype=None,
        shape=None,
        device=None,
        filename=None,
    ): ...

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def from_tensor(
        cls,
        input,
        *,
        filename: Path | str = None,
        existsok: bool = False,
        copy_existing: bool = False,
        copy_data: bool = True,
        shape: torch.Size | None = None,
    ):  # noqa: D417
        """Creates a MemoryMappedTensor with the same content as another tensor.

        If the tensor is already a MemoryMappedTensor the original tensor is
        returned if the `filename` argument is `None` or if the two paths match.
        In all other cases, a new :class:`MemoryMappedTensor` is produced.

        Args:
            input (torch.Tensor): the tensor which content must be copied onto
                the MemoryMappedTensor.

        Keyword Args:
            filename (path to a file): the path to the file where the tensor
                should be stored. If none is provided, a file handler is used
                instead.
            existsok (bool, optional): if ``True``, the file will overwrite
                an existing file. Defaults to ``False``.
            copy_existing (bool, optional): if ``True`` and the provided input
                is a MemoryMappedTensor with an associated filename, copying
                the content to the new location is permitted. Otherwise, an
                exception is thrown. This behaviour exists to prevent
                inadvertently duplicating data on disk.
            copy_data (bool, optional): if ``True``, the content of the tensor
                will be copied on the storage. Defaults to ``True``.
            shape (torch.Size or torch.Tensor): a shape to override the tensor
                shape. If a tensor is passed, it must represent the nested shapes of a
                nested tensor.
        """
        if isinstance(input, MemoryMappedTensor):
            if (filename is None and input._filename is None) or (
                input._filename is not None
                and filename is not None
                and Path(filename).absolute() == Path(input.filename).absolute()
            ):
                # either location was not specified, or memmap is already in the
                # correct location, so just return the MemmapTensor unmodified
                return input
            elif not copy_existing and (
                input._filename is not None
                and filename is not None
                and Path(filename).absolute() != Path(input.filename).absolute()
            ):
                raise RuntimeError(
                    f"A filename was provided but the tensor already has a file associated "
                    f"({input.filename}). "
                    f"To copy the tensor onto the new location, pass copy_existing=True."
                )
        elif isinstance(input, np.ndarray):
            raise TypeError(
                "Convert input to torch.Tensor before calling MemoryMappedTensor.from_tensor."
            )
        if input.requires_grad:
            raise RuntimeError(
                "MemoryMappedTensor.from_tensor is incompatible with tensor.requires_grad."
            )
        if shape is None:
            shape = _shape(input, nested_shape=True)
        if isinstance(shape, torch.Tensor):
            shape_numel = shape.prod(-1).sum()
        elif isinstance(shape, torch.Size):
            shape_numel = shape.numel()
        else:
            shape_numel = torch.Size(shape).numel()
        if filename is None:
            if input.dtype.is_floating_point:
                size = torch.finfo(input.dtype).bits // 8 * shape_numel
            elif input.dtype.is_complex:
                raise ValueError(
                    "Complex-valued tensors are not supported by MemoryMappedTensor."
                )
            elif input.dtype == torch.bool:
                size = shape_numel
            else:
                # assume integer
                size = torch.iinfo(input.dtype).bits // 8 * shape_numel
            handler = _FileHandler(size)
            if isinstance(shape, torch.Tensor):
                func_offset_stride = getattr(
                    torch, "_nested_compute_contiguous_strides_offsets", None
                )
                if func_offset_stride is not None:
                    offsets_strides = func_offset_stride(shape)
                else:
                    raise RuntimeError(NESTED_TENSOR_ERR)
                result = torch.frombuffer(memoryview(handler.buffer), dtype=input.dtype)
                if copy_data:
                    result.untyped_storage().copy_(input.untyped_storage())
                result = torch._nested_view_from_buffer(
                    result,
                    shape,
                    *offsets_strides,
                )
            else:
                result = torch.frombuffer(memoryview(handler.buffer), dtype=input.dtype)
                result = result.view(shape)
            result = cls(result)
        else:
            handler = None
            if not existsok and os.path.exists(str(filename)):
                raise RuntimeError(f"The file {filename} already exists.")
            result = torch.from_file(
                str(filename),
                shared=True,
                dtype=input.dtype,
                size=shape_numel,
                # needed when device ctx differs
                device=torch.device("cpu"),
            )
            if isinstance(shape, torch.Tensor):
                func_offset_stride = getattr(
                    torch, "_nested_compute_contiguous_strides_offsets", None
                )
                if func_offset_stride is not None:
                    offsets_strides = func_offset_stride(shape)
                else:
                    raise RuntimeError(NESTED_TENSOR_ERR)
                if copy_data:
                    result.untyped_storage().copy_(input.untyped_storage())
                result = torch._nested_view_from_buffer(
                    result,
                    shape,
                    *offsets_strides,
                )
            else:
                result = result.view(shape)
            result = cls(result)
        result._handler = handler
        result.filename = filename
        result.index = None
        result.parent_shape = shape
        if copy_data:
            if hasattr(input, "full_tensor"):
                # for DTensors, cheaper than importing DTensor every time
                input = input.full_tensor()
            if not result.is_nested:
                result.copy_(input)
        return result

    @classmethod
    def from_storage(
        cls,
        storage,
        *,
        shape: torch.Size | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        index: IndexType | None = None,
        filename: Path | str = None,
        handler: _handler = None,
    ):
        if getattr(storage, "filename", None) is not None:
            if filename is None:
                filename = storage.filename
            elif str(storage.filename) != str(filename):
                raise RuntimeError(
                    "Providing a storage with an associated filename that differs from the filename argument is not permitted unless filename=None. "
                    f"Got filename={str(filename)}, storage.filename={str(storage.filename)}"
                )
        tensor = torch.tensor(storage, dtype=dtype, device=device)
        if shape is not None:
            if isinstance(shape, torch.Tensor):
                func_offset_stride = getattr(
                    torch, "_nested_compute_contiguous_strides_offsets", None
                )
                if func_offset_stride is not None:
                    offsets_strides = func_offset_stride(shape)
                else:
                    raise RuntimeError(
                        "The PyTorch version isn't compatible with memmap "
                        "nested tensors. Please upgrade to a more recent "
                        "version."
                    )
                tensor = torch._nested_view_from_buffer(
                    tensor,
                    shape,
                    *offsets_strides,
                )
            else:
                tensor = tensor.view(shape)

        tensor = cls(tensor)
        if filename is not None:
            tensor.filename = filename
        elif handler is not None:
            tensor._handler = handler
        if index is not None:
            return tensor[index]
        return tensor

    @property
    def filename(self):
        """The filename of the tensor, if it has one.

        Raises an exception otherwise.
        """
        filename = self._filename
        if filename is None:
            raise RuntimeError("The MemoryMappedTensor has no file associated.")
        return filename

    @filename.setter
    def filename(self, value):
        if value is None and self._filename is None:
            return
        value = str(Path(value).absolute())
        if self._filename is not None and value != self._filename:
            raise RuntimeError(
                "the MemoryMappedTensor has already a filename associated."
            )
        self._filename = value

    @classmethod
    def empty_like(cls, input, *, filename=None):
        # noqa: D417
        """Creates a tensor with no content but the same shape and dtype as the input tensor.

        Args:
            input (torch.Tensor): the tensor to use as an example.

        Keyword Args:
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
        """
        return cls.from_tensor(
            torch.zeros((), dtype=input.dtype, device=input.device).expand_as(input),
            filename=filename,
            copy_data=False,
        )

    @classmethod
    def full_like(cls, input, fill_value, *, filename=None):
        # noqa: D417
        """Creates a tensor with a single content indicated by the `fill_value` argument, but the same shape and dtype as the input tensor.

        Args:
            input (torch.Tensor): the tensor to use as an example.
            fill_value (float or equivalent): content of the tensor.

        Keyword Args:
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
        """
        return cls.from_tensor(
            torch.zeros((), dtype=input.dtype, device=input.device).expand_as(input),
            filename=filename,
            copy_data=False,
        ).fill_(fill_value)

    @classmethod
    def zeros_like(cls, input, *, filename=None):
        # noqa: D417
        """Creates a tensor with a 0-filled content, but the same shape and dtype as the input tensor.

        Args:
            input (torch.Tensor): the tensor to use as an example.

        Keyword Args:
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
        """
        return cls.from_tensor(
            torch.zeros((), dtype=input.dtype, device=input.device).expand_as(input),
            filename=filename,
            copy_data=False,
        ).fill_(0.0)

    @classmethod
    def ones_like(cls, input, *, filename=None):
        # noqa: D417
        """Creates a tensor with a 1-filled content, but the same shape and dtype as the input tensor.

        Args:
            input (torch.Tensor): the tensor to use as an example.

        Keyword Args:
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
        """
        return cls.from_tensor(
            torch.ones((), dtype=input.dtype, device=input.device).expand_as(input),
            filename=filename,
            copy_data=False,
        ).fill_(1.0)

    @classmethod
    @overload
    def ones(cls, *size, dtype=None, device=None, filename=None): ...

    @classmethod
    @overload
    def ones(cls, shape, *, dtype=None, device=None, filename=None): ...

    @classmethod
    def ones(cls, *args, **kwargs):
        # noqa: D417
        """Creates a tensor with a 1-filled content, specific shape, dtype and filename.

        Args:
            shape (integers or torch.Size): the shape of the tensor.

        Keyword Args:
            dtype (torch.dtype): the dtype of the tensor.
            device (torch.device): the device of the tensor. Only `None` and `"cpu"`
                are accepted, any other device will raise an exception.
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
            existsok (bool, optional): whether it is ok to overwrite an existing file.
                Defaults to ``False``.
        """
        shape, device, dtype, _, filename = _proc_args_const(*args, **kwargs)
        if device is not None:
            device = torch.device(device)
            if device.type != "cpu":
                raise RuntimeError("Only CPU tensors are supported.")
        result = torch.ones((), dtype=dtype, device=device)
        if isinstance(shape, torch.Tensor):
            return cls.empty(
                shape, device=device, dtype=dtype, filename=filename
            ).fill_(1)
        if shape:
            if isinstance(shape[0], (list, tuple)) and len(shape) == 1:
                shape = torch.Size(shape[0])
            else:
                shape = torch.Size(shape)
            result = result.expand(shape)
        return cls.from_tensor(
            result,
            filename=filename,
            existsok=kwargs.pop("existsok", False),
        )

    @classmethod
    @overload
    def zeros(cls, *size, dtype=None, device=None, filename=None): ...

    @classmethod
    @overload
    def zeros(cls, shape, *, dtype=None, device=None, filename=None): ...

    @classmethod
    def zeros(cls, *args, **kwargs):
        # noqa: D417
        """Creates a tensor with a 0-filled content, specific shape, dtype and filename.

        Args:
            shape (integers or torch.Size): the shape of the tensor.

        Keyword Args:
            dtype (torch.dtype): the dtype of the tensor.
            device (torch.device): the device of the tensor. Only `None` and `"cpu"`
                are accepted, any other device will raise an exception.
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
            existsok (bool, optional): whether it is ok to overwrite an existing file.
                Defaults to ``False``.
        """
        shape, device, dtype, _, filename = _proc_args_const(*args, **kwargs)
        if device is not None:
            device = torch.device(device)
            if device.type != "cpu":
                raise RuntimeError("Only CPU tensors are supported.")
        if isinstance(shape, torch.Tensor):
            return cls.empty(
                shape, device=device, dtype=dtype, filename=filename
            ).fill_(0)
        result = torch.zeros((), dtype=dtype, device=device)
        if shape:
            if isinstance(shape[0], (list, tuple)) and len(shape) == 1:
                shape = torch.Size(shape[0])
            else:
                shape = torch.Size(shape)
            result = result.expand(shape)
        result = cls.from_tensor(
            result,
            filename=filename,
            existsok=kwargs.pop("existsok", False),
        )
        return result

    @classmethod
    @overload
    def empty(cls, *size, dtype=None, device=None, filename=None): ...

    @classmethod
    @overload
    def empty(cls, shape, *, dtype=None, device=None, filename=None): ...

    @classmethod
    def empty(cls, *args, **kwargs):
        # noqa: D417
        """Creates a tensor with empty content, specific shape, dtype and filename.

        Args:
            shape (integers or torch.Size): the shape of the tensor.

        Keyword Args:
            dtype (torch.dtype): the dtype of the tensor.
            device (torch.device): the device of the tensor. Only `None` and `"cpu"`
                are accepted, any other device will raise an exception.
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
            existsok (bool, optional): whether it is ok to overwrite an existing file.
                Defaults to ``False``.
        """
        shape, device, dtype, _, filename = _proc_args_const(*args, **kwargs)
        if device is not None:
            device = torch.device(device)
            if device.type != "cpu":
                raise RuntimeError("Only CPU tensors are supported.")
        result = torch.zeros((), dtype=dtype, device=device)
        if isinstance(shape, torch.Tensor):
            # nested tensor
            shape_numel = shape.prod(-1).sum()

            if filename is None:
                if dtype.is_floating_point:
                    size = torch.finfo(dtype).bits // 8 * shape_numel
                elif dtype.is_complex:
                    raise ValueError(
                        "Complex-valued tensors are not supported by MemoryMappedTensor."
                    )
                elif dtype == torch.bool:
                    size = shape_numel
                else:
                    # assume integer
                    size = torch.iinfo(dtype).bits // 8 * shape_numel
                handler = _FileHandler(size)

                # buffer
                func_offset_stride = getattr(
                    torch, "_nested_compute_contiguous_strides_offsets", None
                )
                if func_offset_stride is not None:
                    offsets_strides = func_offset_stride(shape)
                else:
                    raise RuntimeError(NESTED_TENSOR_ERR)
                result = torch.frombuffer(memoryview(handler.buffer), dtype=dtype)
                result = torch._nested_view_from_buffer(
                    result,
                    shape,
                    *offsets_strides,
                )
                result = cls(result)
                result._handler = handler
                return result
            else:
                result = torch.from_file(
                    str(filename),
                    shared=True,
                    dtype=dtype,
                    size=shape_numel,
                    # needed when device ctx differs
                    device=torch.device("cpu"),
                )
                func_offset_stride = getattr(
                    torch, "_nested_compute_contiguous_strides_offsets", None
                )
                if func_offset_stride is not None:
                    offsets_strides = func_offset_stride(shape)
                else:
                    raise RuntimeError(NESTED_TENSOR_ERR)
                result = torch._nested_view_from_buffer(
                    result,
                    shape,
                    *offsets_strides,
                )
                result = cls(result)
                result.filename = filename
                return result
            return result

        if shape:
            if isinstance(shape[0], (list, tuple)) and len(shape) == 1:
                shape = torch.Size(shape[0])
            else:
                shape = torch.Size(shape)
            result = result.expand(shape)
        result = cls.from_tensor(
            result,
            filename=filename,
            copy_data=False,
            existsok=kwargs.pop("existsok", False),
        )
        return result

    @classmethod
    def empty_nested(cls, *args, **kwargs):
        # noqa: D417
        """Creates a tensor with empty content, specific shape, dtype and filename.

        Args:
            shape (nested_shape): the shapes of the tensors.

        Keyword Args:
            dtype (torch.dtype): the dtype of the tensor.
            device (torch.device): the device of the tensor. Only `None` and `"cpu"`
                are accepted, any other device will raise an exception.
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
            existsok (bool, optional): whether it is ok to overwrite an existing file.
                Defaults to ``False``.
        """
        shape = kwargs.pop("shape", args[0])
        args = (torch.Size([]), *args)
        _, device, dtype, _, filename = _proc_args_const(*args, **kwargs)
        if device is not None:
            device = torch.device(device)
            if device.type != "cpu":
                raise RuntimeError("Only CPU tensors are supported.")
        result = torch.zeros((), dtype=dtype, device=device)
        if shape:
            if isinstance(shape[0], (list, tuple)) and len(shape) == 1:
                shape = torch.Size(shape[0])
            else:
                shape = torch.Size(shape)
            result = result.expand(shape)
        result = cls.from_tensor(
            result,
            filename=filename,
            copy_data=False,
            existsok=kwargs.pop("existsok", False),
        )
        return result

    @classmethod
    @overload
    def full(cls, *size, fill_value, dtype=None, device=None, filename=None): ...

    @classmethod
    @overload
    def full(cls, shape, *, fill_value, dtype=None, device=None, filename=None): ...

    @classmethod
    def full(cls, *args, **kwargs):
        # noqa: D417
        """Creates a tensor with a single content specified by `fill_value`, specific shape, dtype and filename.

        Args:
            shape (integers or torch.Size): the shape of the tensor.

        Keyword Args:
            fill_value (float or equivalent): content of the tensor.
            dtype (torch.dtype): the dtype of the tensor.
            device (torch.device): the device of the tensor. Only `None` and `"cpu"`
                are accepted, any other device will raise an exception.
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
            existsok (bool, optional): whether it is ok to overwrite an existing file.
                Defaults to ``False``.
        """
        shape, device, dtype, fill_value, filename = _proc_args_const(*args, **kwargs)
        if device is not None:
            device = torch.device(device)
            if device.type != "cpu":
                raise RuntimeError("Only CPU tensors are supported.")
        result = torch.zeros((), dtype=dtype, device=device).fill_(fill_value)
        if shape:
            if isinstance(shape[0], (list, tuple)) and len(shape) == 1:
                shape = torch.Size(shape[0])
            else:
                shape = torch.Size(shape)
            result = result.expand(shape)
        return cls.from_tensor(
            result, filename=filename, existsok=kwargs.pop("existsok", False)
        )

    @classmethod
    def from_filename(cls, filename, dtype, shape, index=None):
        # noqa: D417
        """Loads a MemoryMappedTensor from a given filename.

        Args:
            filename (path or equivalent): the path to the file.
            dtype (torch.dtype): the dtype of the tensor.
            shape (torch.Size or torch.Tensor): the shape of the tensor. If
                a tensor is provided, it is assumed that the tensor is a nested_tensor
                instance.
            index (torch-compatible index type): an index to use to build the
                tensor.

        """
        writable = _is_writable(filename)

        if isinstance(shape, torch.Tensor):
            func_offset_stride = getattr(
                torch, "_nested_compute_contiguous_strides_offsets", None
            )
            if func_offset_stride is not None:
                offsets_strides = func_offset_stride(shape)
            else:
                raise RuntimeError(
                    "The PyTorch version isn't compatible with memmap "
                    "nested tensors. Please upgrade to a more recent "
                    "version."
                )
            tensor = torch.from_file(
                str(filename),
                shared=writable,
                dtype=dtype,
                size=shape.prod(-1).sum().int(),
                # needed when device ctx differs
                device=torch.device("cpu"),
            )
            tensor = torch._nested_view_from_buffer(
                tensor,
                shape,
                *offsets_strides,
            )
        else:
            shape = torch.Size(shape)
            # whether the file already existed
            tensor = torch.from_file(
                str(filename),
                shared=writable,
                dtype=dtype,
                size=shape.numel(),
                # needed when device ctx differs
                device=torch.device("cpu"),
            )
            tensor = tensor.view(shape)

        if index is not None:
            tensor = tensor[index]
        out = cls(tensor)
        out.filename = filename
        out._handler = None
        out.index = index
        out.parent_shape = shape
        return out

    @classmethod
    def from_handler(cls, handler, dtype, shape, index=None):
        # noqa: D417
        """Loads a MemoryMappedTensor from a given handler.

        Args:
            handler (compatible file handler): the handler for the tensor.
            dtype (torch.dtype): the dtype of the tensor.
            shape (torch.Size or torch.Tensor): the shape of the tensor. If
                a tensor is provided, it is assumed that the tensor is a nested_tensor
                instance.
            index (torch-compatible index type, optional): an index to use to build the
                tensor.

        """
        out = torch.frombuffer(memoryview(handler.buffer), dtype=dtype)
        if isinstance(shape, torch.Tensor):
            func_offset_stride = getattr(
                torch, "_nested_compute_contiguous_strides_offsets", None
            )
            if func_offset_stride is not None:
                offsets_strides = func_offset_stride(shape)
            else:
                raise RuntimeError(
                    "The PyTorch version isn't compatible with memmap "
                    "nested tensors. Please upgrade to a more recent "
                    "version."
                )
            out = torch._nested_view_from_buffer(
                out,
                shape,
                *offsets_strides,
            )
        else:
            shape = torch.Size(shape)
            out = torch.reshape(out, shape)

        if index is not None:
            out = out[index]
        out = cls(out)
        out.filename = None
        out._handler = handler
        out.index = index
        out.parent_shape = shape
        return out

    @property
    def _tensor(self):
        # for bc-compatibility with MemmapTensor, to be deprecated in v0.4
        return self

    def __setstate__(self, state):
        if "filename" in state:
            self.__dict__ = type(self).from_filename(**state).__dict__
        else:
            self.__dict__ = type(self).from_handler(**state).__dict__

    def __getstate__(self):
        if getattr(self, "_handler", None) is not None:
            return {
                "handler": self._handler,
                "dtype": self.dtype,
                "shape": list(self.parent_shape),
                "index": self.index,
            }
        elif getattr(self, "_filename", None) is not None:
            return {
                "filename": self._filename,
                "dtype": self.dtype,
                "shape": self.parent_shape,
                "index": self.index,
            }
        else:
            raise RuntimeError("Could not find handler or filename.")

    def __reduce_ex__(self, protocol):
        return self.__reduce__()

    def __reduce__(self):
        if getattr(self, "_handler", None) is not None:
            return type(self).from_handler, (
                self._handler,
                self.dtype,
                self.parent_shape,
                self.index,
            )
        elif getattr(self, "_filename", None) is not None:
            return type(self).from_filename, (
                self._filename,
                self.dtype,
                self.parent_shape,
                self.index,
            )
        else:
            raise RuntimeError("Could not find handler or filename.")

    @implement_for("torch", "2.0", None)
    def __getitem__(self, item: IndexType) -> torch.Tensor:
        try:
            out = super().__getitem__(item)
        except ValueError as err:
            if "is unbound" in str(err):
                raise ValueError(
                    "Using first class dimension indices with MemoryMappedTensor "
                    "isn't supported at the moment."
                ) from err
            raise
        if out.untyped_storage().data_ptr() == self.untyped_storage().data_ptr():
            out = self._index_wrap(out, item)
        return out

    @implement_for("torch", None, "2.0")
    def __getitem__(self, item: IndexType) -> torch.Tensor:  # noqa: F811
        try:
            out = super().__getitem__(item)
        except ValueError as err:
            if "is unbound" in str(err):
                raise ValueError(
                    "Using first class dimension indices with MemoryMappedTensor "
                    "isn't supported at the moment."
                ) from err
            raise
        if out.storage().data_ptr() == self.storage().data_ptr():
            out = self._index_wrap(out, item)
        return out

    def _index_wrap(self, tensor, item, check=False):
        if check:
            if tensor.storage().data_ptr() == self.storage().data_ptr():
                return self._index_wrap(tensor, item)
            return tensor
        tensor = MemoryMappedTensor(tensor)
        tensor._handler = getattr(self, "_handler", None)
        tensor.filename = getattr(self, "_filename", None)
        tensor.index = item
        tensor.parent_shape = getattr(self, "parent_shape", None)
        return tensor

    def unbind(self, dim):
        out = super().unbind(dim)
        if dim < 0:
            dim = self.ndim + dim
        index_base = (slice(None),) * dim
        return tuple(
            self._index_wrap(_out, index_base + (i,)) for i, _out in enumerate(out)
        )

    def chunk(self, chunks, dim=0):
        out = super().chunk(chunks, dim)
        return tuple(self._index_wrap(chunk, None, check=True) for chunk in out)


#####################
# File handler
# borrowed from mp.heap

if sys.platform == "win32":
    import _winapi

    class _FileHandler:
        _rand = tempfile._RandomNameSequence()

        def __init__(self, size):
            self.size = size
            for _ in range(100):
                name = "pym-%d-%s" % (os.getpid(), next(self._rand))
                buf = mmap.mmap(-1, size, tagname=name)
                if _winapi.GetLastError() == 0:
                    break
                # We have reopened a preexisting mmap.
                buf.close()
            else:
                raise FileExistsError("Cannot find name for new mmap")
            self.name = name
            self.buffer = buf
            self._state = (self.size, self.name)

        def __getstate__(self):
            from multiprocessing.context import assert_spawning

            assert_spawning(self)
            return self._state

        def __setstate__(self, state):
            self.size, self.name = self._state = state
            # Reopen existing mmap
            self.buffer = mmap.mmap(-1, self.size, tagname=self.name)
            # XXX Temporarily preventing buildbot failures while determining
            # XXX the correct long-term fix. See issue 23060
            # assert _winapi.GetLastError() == _winapi.ERROR_ALREADY_EXISTS

else:

    class _FileHandler:
        if sys.platform == "linux":
            _dir_candidates = ["/dev/shm"]
        else:
            _dir_candidates = []

        def __init__(self, size, fd=-1):
            self.size = size
            self.fd = fd
            if fd == -1:
                self.fd, name = tempfile.mkstemp(
                    prefix="pym-%d-" % os.getpid(), dir=self._choose_dir(size)
                )
                os.unlink(name)
                util.Finalize(self, os.close, (self.fd,))
                os.ftruncate(self.fd, size)
            self.buffer = mmap.mmap(self.fd, self.size)

        def _choose_dir(self, size):
            # Choose a non-storage backed directory if possible,
            # to improve performance
            for d in self._dir_candidates:
                st = os.statvfs(d)
                if st.f_bavail * st.f_frsize >= size:  # enough free space?
                    return d
            return util.get_temp_dir()

    def _reduce_handler(handler):
        if handler.fd == -1:
            raise ValueError(
                "Handler is unpicklable because "
                "forking was enabled when it was created"
            )
        return _rebuild_handler, (handler.size, reduction.DupFd(handler.fd))

    def _rebuild_handler(size, dupfd):
        detached = dupfd.detach()
        return _FileHandler(size, detached)

    reduction.register(_FileHandler, _reduce_handler)


def _reduce_memmap(memmap_tensor):
    return memmap_tensor.__reduce__()


reduction.register(MemoryMappedTensor, _reduce_memmap)


def _proc_args_const(*args, **kwargs):
    if len(args) > 0:
        # then the first (or the N first) args are the shape
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            shape = args[0]
        elif len(args) == 1 and not isinstance(args[0], int):
            shape = torch.Size(args[0])
        else:
            shape = torch.Size(args)
    else:
        # we should have a "shape" keyword arg
        shape = kwargs.pop("shape", None)
        if shape is None:
            raise TypeError("Could not find the shape argument in the arguments.")
        if not isinstance(shape, torch.Tensor):
            shape = torch.Size(shape)
    return (
        shape,
        kwargs.pop("device", None),
        kwargs.pop("dtype", None),
        kwargs.pop("fill_value", None),
        kwargs.pop("filename", None),
    )


# Torch functions

MEMMAP_HANDLED_FUNCTIONS: dict[Callable, Callable] = {}


def implements_for_memmap(torch_function: Callable) -> Callable[[Callable], Callable]:
    """Register a torch function override for MemoryMappedTensor."""

    @functools.wraps(torch_function)
    def decorator(func: Callable) -> Callable:
        MEMMAP_HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


@implements_for_memmap(torch.unbind)
def _unbind(tensor, dim):
    return tensor.unbind(dim)


@implements_for_memmap(torch.chunk)
def _chunk(input, chunks, dim=0):
    return input.chunk(chunks, dim=dim)


def _is_writable(file_path):
    file_path = str(file_path)
    if os.path.exists(file_path):
        return os.access(file_path, os.W_OK)
    # Assume that the file can be written in the directory
    return True

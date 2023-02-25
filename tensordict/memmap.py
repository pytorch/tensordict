# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
import os
import tempfile
import warnings
from copy import copy, deepcopy
from pathlib import Path
from tempfile import _TemporaryFileWrapper
from typing import Any, Callable, Sequence

import numpy as np
import torch

from tensordict.utils import (
    _getitem_batch_size,
    DeviceType,
    IndexType,
    NUMPY_TO_TORCH_DTYPE_DICT,
    prod,
    TORCH_TO_NUMPY_DTYPE_DICT,
)

__all__ = ["MemmapTensor", "set_transfer_ownership"]


NoneType = type(None)
EllipsisType = type(Ellipsis)


MEMMAP_HANDLED_FN: dict[Callable, Callable] = {}


def implements_for_memmap(torch_function: Callable) -> Callable:
    """Register a torch function override for ScalarTensor."""

    @functools.wraps(torch_function)
    def decorator(func: Callable) -> Callable:
        MEMMAP_HANDLED_FN[torch_function] = func
        return func

    return decorator


def to_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor


class MemmapTensor:
    """A torch.tensor interface with a np.memmap array.

    A temporary file is created and cleared once the object is out-of-scope.
    This class is aimed at being used for data transfer in between processes
    and remote workers that have access to
    a common storage, and as such it supports serialization and
    deserialization. It is possible to choose if the ownership is
    transferred upon serialization / deserialization: If ownership is not
    transferred (transfer_ownership=False, default), then the process where
    the MemmapTensor was created will be responsible of clearing it once it
    gets out of scope (in that process). Otherwise, the process that
    deserialize the MemmapTensor will be responsible of clearing the files
    once the object is out of scope.

    Supports (almost) all tensor operations.

    Args:
        *tensor_or_size (torch.Tensor, MemmapTensor, torch.Size or sequence of integers):
            If a size is provided (with a sequence of integers, a torch.Size object
            or a list/tuple of integers) it indicates the size of the MemmapTensor created.
            If a te is provided, its content will be stored on physical storage.
            If MemmapTensor, a new MemmapTensor is created and the same data is stored in it.
        device (torch.device or equivalent, optional): device where the loaded
            tensor will be sent. This should not be used with MemmapTensors
            created from torch.Tensor objects. Default is "cpu".
        dtype (torch.dtype, optional): dtype of the loaded tensor.
            This should not be used with MemmapTensors created from torch.Tensor
            objects. Default is :obj:`torch.get_default_dtype()`.
        transfer_ownership (bool, optional): affects the ownership after serialization:
            if True, the current process looses ownership immediately after
            serialization. If False, the current process keeps the ownership
            of the temporary file.
            Default: False.
        prefix (str or path, optional): *Deprecated* prefix of the file location. Should
            not be specified together with prefix.
        filename (str or path, optional): location of the underlying memory-map. Should
            not be specified together with prefix.

    Examples:
        >>> x = torch.ones(3,4)
        >>> x_memmap = MemmapTensor.from_tensor(x)
        >>> # indexing
        >>> x0 = x_memmap[0]
        >>> x0[:] = 2
        >>> assert (x_memmap[0]==2).all()
        >>>
        >>> # device
        >>> x = x.to('cuda:0')
        >>> x_memmap = MemmapTensor.from_tensor(x)
        >>> assert (x_memmap.clone()).device == torch.device('cuda:0')
        >>>
        >>> # operations
        >>> assert (x_memmap + 1 == x+1).all()
        >>> assert (x_memmap / 2 == x/2).all()
        >>> assert (x_memmap * 2 == x*2).all()
        >>>
        >>> # temp file clearance
        >>> filename = x_memmap.filename
        >>> assert os.path.isfile(filename)
        >>> del x_memmap
        >>> assert not os.path.isfile(filename)

    """

    requires_grad: bool = False

    def __init__(
        self,
        *size: int,
        device: DeviceType | None = None,
        dtype: torch.dtype | None = None,
        transfer_ownership: bool = False,
        prefix: str | None = None,
        filename: str | None = None,
        mode: str = "r+",
    ) -> None:
        self.idx = None
        self._memmap_array = None
        self.prefix = prefix
        self.is_meta = False

        if mode in ("r+", "w+", "c", "copyonwrite", "readwrite", "write"):
            self.mode = mode
        else:
            raise ValueError(
                'Accepted values for mode are "r+", "readwrite", "w+", "write", "c" or '
                '"copyonwrite". PyTorch does not support tensors backed by read-only '
                'NumPy arrays, so "r" and "readonly" are not supported.'
            )

        if prefix is not None:
            warnings.warn(
                "prefix has been deprecated. If you want to control the location of "
                "the MemmapTensor on disk, consider using filename instead.",
                stacklevel=2,
            )
            if filename is not None:
                raise ValueError("filename and prefix should not both be specified")

        # open the files in r+ mode so as to not overwrite any data that might exist
        # there. the actual memmap will be instantiated with user-supplied mode
        if filename is None:
            self.file = tempfile.NamedTemporaryFile(
                prefix=prefix, delete=False, mode="r+"
            )
        else:
            # if filename doesn't exist we must create it
            Path(filename).touch(exist_ok=True)
            self.file = open(filename, mode="r+")

        self.filename = self.file.name
        self.file.close()  # we close the file for now, but don't delete it

        if isinstance(size[0], (torch.Tensor, MemmapTensor, np.ndarray)):
            raise NotImplementedError(
                "Creating a Memmap array from a tensor is not permitted anymore. "
                "Call MemmapTensor.from_tensor(tensor) instead."
            )
        else:
            try:
                shape = (
                    torch.Size(list(size[0]))
                    if len(size) == 1 and not isinstance(size[0], int)
                    else torch.Size(list(size))
                )
            except TypeError:
                raise TypeError(
                    f"The *size must be either a single list or tuple of ints, or a sequence of ints. Got {size} instead."
                )
            device = device if device is not None else torch.device("cpu")
            dtype = dtype if dtype is not None else torch.get_default_dtype()
            self._init_shape(
                shape=shape,
                device=device,
                dtype=dtype,
                transfer_ownership=transfer_ownership,
            )
        if not hasattr(self, "_index"):
            self._index = None

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor | MemmapTensor | np.ndarray,
        transfer_ownership: bool = False,
        prefix: str | None = None,
        filename: str | None = None,
        mode: str = "r+",
    ) -> MemmapTensor:
        if isinstance(tensor, MemmapTensor):
            if transfer_ownership:
                raise RuntimeError(
                    "from_tensor(memmap_tensor, transfer_ownership=True) is not permitted, as this method will "
                    "simply return the original MemmapTensor instance."
                )
            elif prefix is None and (
                filename is None
                or Path(filename).absolute() == Path(tensor.filename).absolute()
            ):
                # either location was not specified, or memmap is already in the
                # correct location, so just return the MemmapTensor unmodified
                return tensor
        elif isinstance(tensor, np.ndarray):
            raise TypeError(
                "Convert input to torch.Tensor before calling MemmapTensor."
            )
        if tensor.requires_grad:
            raise RuntimeError(
                "MemmapTensor is incompatible with tensor.requires_grad."
            )
        device = tensor.device if hasattr(tensor, "device") else torch.device("cpu")
        dtype = (
            tensor.dtype
            if isinstance(tensor, (torch.Tensor, MemmapTensor))
            else NUMPY_TO_TORCH_DTYPE_DICT[np.dtype(tensor.dtype.name)]
        )
        shape = tensor.shape
        out = cls(
            shape,
            device=device,
            dtype=dtype,
            prefix=prefix,
            transfer_ownership=transfer_ownership,
            filename=filename,
            mode=mode,
        )
        out.copy_(tensor)
        return out

    @classmethod
    def empty_like(
        cls,
        tensor: torch.Tensor | MemmapTensor,
        transfer_ownership: bool = False,
        prefix: str | None = None,
        filename: str | None = None,
        mode: str = "r+",
    ) -> MemmapTensor:
        if isinstance(tensor, np.ndarray):
            raise TypeError(
                "Convert input to torch.Tensor before calling MemmapTensor."
            )
        device = tensor.device
        dtype = tensor.dtype
        shape = tensor.shape
        out = cls(
            shape,
            device=device,
            dtype=dtype,
            prefix=prefix,
            transfer_ownership=transfer_ownership,
            filename=filename,
            mode=mode,
        )
        return out

    @staticmethod
    def _create_memmap_with_index(memmap_tensor, index):
        memmap_copy = copy(memmap_tensor)
        if memmap_copy._index is None:
            memmap_copy._index = []
        else:
            # avoid extending someone else's index
            memmap_copy._index = deepcopy(memmap_copy._index)
        memmap_copy._index.append(index)
        memmap_copy.transfer_ownership = False
        memmap_copy._shape_indexed = None
        return memmap_copy

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]

    def _init_shape(
        self,
        shape: torch.Size,
        device: DeviceType,
        dtype: torch.dtype,
        transfer_ownership: bool,
    ):
        self._device = device
        self._shape = shape
        self._shape_indexed = None
        self.transfer_ownership = transfer_ownership
        self.np_shape = tuple(self._shape)
        self._dtype = dtype
        self._ndim = len(shape)
        self._numel = prod(shape)
        self._has_ownership = True
        self._had_ownership = True

        self._tensor_dir = torch.zeros(0, device=device, dtype=dtype).__dir__()
        self._save_item(shape)

    def _get_memmap_array(self) -> np.memmap:
        if self._memmap_array is None:
            self._memmap_array = np.memmap(
                self.filename,
                dtype=TORCH_TO_NUMPY_DTYPE_DICT[self.dtype],
                mode=self.mode,
                shape=self.np_shape,
            )
        return self._memmap_array

    def _set_memmap_array(self, value: np.memmap) -> None:
        self._memmap_array = value

    memmap_array = property(_get_memmap_array, _set_memmap_array)

    def _save_item(
        self,
        value: torch.Tensor | torch.Size | MemmapTensor | np.ndarray,
        idx: int | None = None,
    ):
        if isinstance(value, MemmapTensor):
            np_array = value.memmap_array
        elif isinstance(value, (torch.Tensor,)):
            np_array = value.cpu().numpy()
        elif isinstance(value, torch.Size):
            # create the memmap array on disk
            _ = self.memmap_array
            return
        else:
            np_array = value
        memmap_array = self.memmap_array
        if idx is None:
            memmap_array[:] = np_array
        else:
            memmap_array[idx] = np_array

    def _copy_item(self, filename: bytes | str) -> None:
        self.memmap_array[:] = np.memmap(
            filename,
            dtype=TORCH_TO_NUMPY_DTYPE_DICT[self.dtype],
            mode="r",
            shape=self.np_shape,
        )

    def _get_item(self, idx: IndexType, memmap_array: np.ndarray) -> np.ndarray:
        if isinstance(idx, torch.Tensor):
            idx = idx.cpu()
        elif isinstance(idx, tuple) and any(
            isinstance(sub_index, torch.Tensor) for sub_index in idx
        ):
            idx = tuple(
                sub_index.cpu() if isinstance(sub_index, torch.Tensor) else sub_index
                for sub_index in idx
            )
        memmap_array = memmap_array[idx]
        return memmap_array

    def _load_item(
        self,
        idx: int | tuple | list | None = None,
        memmap_array: np.ndarray | None = None,
        from_numpy: bool = False,
    ) -> torch.Tensor:
        if memmap_array is None:
            memmap_array = self.memmap_array
        if idx is not None:
            if not isinstance(idx, list):
                idx = [idx]
            for _idx in idx:
                memmap_array = self._get_item(_idx, memmap_array)
        out = self._np_to_tensor(memmap_array, from_numpy=from_numpy)
        if (
            idx is not None
            and not isinstance(idx, (int, np.integer, slice))
            and len(idx) == 1
            and not (isinstance(idx, torch.Tensor) and idx.dtype is torch.bool)
        ):  # and isinstance(idx, torch.Tensor) and len(idx) == 1:
            size = self.shape
            out = out.view(size)
        return out

    def _np_to_tensor(self, memmap_array: np.ndarray, from_numpy: bool) -> torch.Tensor:
        if from_numpy:
            return torch.from_numpy(memmap_array)
        return torch.as_tensor(memmap_array, device=self.device)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ):
        if kwargs is None:
            kwargs = {}
        if func not in MEMMAP_HANDLED_FN:
            args = tuple(a._tensor if hasattr(a, "_tensor") else a for a in args)
            ret = func(*args, **kwargs)
            return ret

        return MEMMAP_HANDLED_FN[func](*args, **kwargs)

    @property
    def _tensor(self) -> torch.Tensor:
        if not os.path.isfile(self.filename):
            # close ref to file if it has been deleted -- ensures all processes
            # loose access to a file once it's deleted
            # see https://stackoverflow.com/questions/44691030/numpy-memmap-with-file-deletion
            self._memmap_array = None
        return self._load_item(self._index)

    @property
    def _tensor_from_numpy(self) -> torch.Tensor:
        # a tensor created with `from_numpy` to make sure that changes are done in-place
        return self._load_item(from_numpy=True)

    def ndimension(self) -> int:
        return self._ndim

    def numel(self) -> int:
        return self._numel

    def clone(self) -> MemmapTensor:
        """Clones the MemmapTensor onto another tensor.

        Returns:
            a new torch.Tensor with the same data but a new storage.

        """
        return self._tensor.clone()

    def contiguous(self) -> torch.Tensor:
        """Copies the MemmapTensor onto a torch.Tensor object.

        Returns:
            a torch.Tensor instance with the data of the MemmapTensor
        stored on the desired device.

        """
        return self._tensor

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: DeviceType) -> None:
        self._device = torch.device(device)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def shape(self) -> torch.Size:
        if self._shape_indexed is None:
            size = self._shape
            idx = self._index if self._index is not None else []
            for _idx in idx:
                size = _getitem_batch_size(size, _idx)
            self._shape_indexed = size
        return self._shape_indexed

    def cpu(self) -> torch.Tensor:
        """Defines the device of the MemmapTensor as "cpu".

        Returns: a MemmapTensor where device has been modified in-place

        """
        self.device = torch.device("cpu")
        return self

    def cuda(self) -> torch.Tensor:
        """Defines the device of the MemmapTensor as "cuda".

        Returns: a MemmapTensor where device has been modified in-place

        """
        self.device = torch.device("cuda")
        return self

    def numpy(self) -> np.ndarray:
        return self._tensor.numpy()

    def copy_(self, other: torch.Tensor | MemmapTensor) -> MemmapTensor:
        if isinstance(other, MemmapTensor) and other.filename == self.filename:
            if not self.shape == other.shape:
                raise ValueError(
                    f"""Cannot copy a MemmapTensor of shape {other.shape} on a
MemmapTensor of shape {self.shape}."""
                )
            self._index = other._index
            return self
        self._save_item(other)
        return self

    def set_transfer_ownership(self, value: bool = True) -> MemmapTensor:
        """Controls whether the ownership will be transferred to another process upon serialization/deserialization.

        Args:
            value (bool): if True, the ownership will be transferred.
                Otherwise the process will keep ownership of the
                MemmapTensor temp file.
                Default = True

        Returns:
            the MemmapTensor

        """
        if not isinstance(value, bool):
            raise TypeError(
                f"value provided to set_transfer_ownership should be a "
                f"boolean, got {type(value)}"
            )
        self.transfer_ownership = value
        return self

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> MemmapTensor:
        warnings.warn(
            "calling deepcopy on a memmap tensor involves loading it in memory "
            "and recreating a memmap tensor from scratch (as no file destination "
            "can be passed to deepcopy(...).",
            stacklevel=2,
        )
        return MemmapTensor.from_tensor(self.clone())

    def __del__(self) -> None:
        if "_has_ownership" in self.__dir__() and self._has_ownership:
            if isinstance(self.file, tempfile._TemporaryFileWrapper):
                # only delete file if we created a temporary file. Otherwise file should
                # persist on disk
                os.unlink(self.filename)
            del self.file

    def __eq__(self, other: Any) -> torch.Tensor:
        return self._tensor == other

    def __ne__(self, other: Any) -> torch.Tensor:
        return self._tensor != other

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__dir__():
            return self.__getattribute__(
                attr
            )  # make sure that appropriate exceptions are raised

        if ("_tensor_dir" not in self.__dir__()) or (
            attr not in self.__getattribute__("_tensor_dir")
        ):
            raise AttributeError(f"{attr} not found")
        _tensor = self.__getattribute__("_tensor")
        return getattr(_tensor, attr)

    def masked_fill_(self, mask: torch.Tensor, value: float) -> MemmapTensor:
        self.memmap_array[mask.cpu().numpy()] = value
        return self

    def __len__(self) -> int:
        return self.shape[0] if len(self.shape) else 0

    def is_shared(self) -> bool:
        return False

    def __add__(self, other: float | MemmapTensor | torch.Tensor) -> torch.Tensor:
        return torch.add(self, other)

    def __truediv__(self, other: float | MemmapTensor | torch.Tensor) -> torch.Tensor:
        return torch.div(self, other)

    def __neg__(self: float | MemmapTensor | torch.Tensor) -> torch.Tensor:
        return torch.neg(self)

    def __sub__(self, other: float | MemmapTensor | torch.Tensor) -> torch.Tensor:
        return torch.sub(self, other)

    def __matmul__(self, other: float | MemmapTensor | torch.Tensor) -> torch.Tensor:
        return torch.matmul(self, other)

    def __mul__(self, other: float | MemmapTensor | torch.Tensor) -> torch.Tensor:
        return torch.mul(self, other)

    def __pow__(self, other: float | MemmapTensor | torch.Tensor) -> torch.Tensor:
        return torch.pow(self, other)

    def __repr__(self) -> str:
        return f"MemmapTensor(shape={self.shape}, device={self.device}, dtype={self.dtype})"

    def __getitem__(self, item: IndexType) -> torch.Tensor:
        # return self._load_item(memmap_array=self.memmap_array[item])#[item]
        # return self._load_item()[item]
        if isinstance(item, (NoneType, EllipsisType, int, np.integer, slice)):
            item = (item,)
        return MemmapTensor._create_memmap_with_index(self, item)

    def __setitem__(self, idx: IndexType, value: torch.Tensor) -> None:
        if self.device == torch.device("cpu"):
            self._load_item()[idx] = value
        else:
            if isinstance(idx, torch.Tensor):
                idx = idx.cpu()
            elif isinstance(idx, tuple) and any(
                isinstance(_idx, torch.Tensor) for _idx in idx
            ):
                idx = tuple(
                    _idx.cpu() if isinstance(_idx, torch.Tensor) else _idx
                    for _idx in idx
                )
            self.memmap_array[idx] = to_numpy(value)

    def __setstate__(self, state: dict[str, Any]) -> None:
        if state["file"] is None:
            # state["_had_ownership"] = state["_had_ownership"]
            # state["_has_ownership"] = delete
            # tmpfile = tempfile.NamedTemporaryFile(delete=False)
            # tmpfile.close()
            tmpfile = _TemporaryFileWrapper(None, state["filename"], delete=True)
            tmpfile.name = state["filename"]
            tmpfile._closer.name = state["filename"]
            state["file"] = tmpfile
        self.__dict__.update(state)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["file"] = None
        state["_memmap_array"] = None
        state["_fake"] = None
        state["_has_ownership"] = (
            state["transfer_ownership"] and state["_had_ownership"]
        )
        self._had_ownership = self._has_ownership
        # self._had_ownership = self._has_ownership = state["_had_ownership"]
        return state

    def __reduce__(self) -> tuple[Any, ...]:
        if self.transfer_ownership and self._has_ownership:
            self._has_ownership = False
            # those values should already be False
            # self.file.delete = False
            # self.file._closer.delete = False
        return super().__reduce__()

    def to(
        self,
        dest: DeviceType | torch.dtype,
        non_blocking: bool = False,
    ) -> torch.Tensor | MemmapTensor:
        """Maps a MemmapTensor to a given dtype or device.

        Args:
            dest (device indicator or torch.dtype): where to cast the
                MemmapTensor. For devices, this is a lazy operation
                (as the data is stored on physical memory). For dtypes, the
                tensor will be retrieved, mapped to the
                desired dtype and cast to a new MemmapTensor.
            non_blocking (bool, optional): no-op for MemmapTensors. Default: False.

        Returns: the same memmap-tensor with the changed device.

        """
        if isinstance(dest, (int, str, torch.device)):
            dest = torch.device(dest)
            self.device = dest
            return self
        elif isinstance(dest, torch.dtype):
            return MemmapTensor.from_tensor(self._tensor.to(dest))
        else:
            raise NotImplementedError(
                f"argument dest={dest} to MemmapTensor.to(dest) is not "
                f"handled. "
                f"Please provide a dtype or a device."
            )

    def unbind(self, dim: int) -> tuple[torch.Tensor, ...]:
        """Unbinds a MemmapTensor along the desired dimension.

        Args:
            dim (int): dimension along which the MemmapTensor will be split.

        Returns:
            A tuple of indexed MemmapTensors that share the same storage.

        """
        idx = [(*(slice(None) for _ in range(dim)), i) for i in range(self.shape[dim])]
        return tuple(self[_idx] for _idx in idx)

    def as_tensor(self) -> torch.Tensor:
        """Represents a MemmapTensor as a tensor, with the same storage (ie without any copy)."""
        if not self.device.type == "cpu":
            raise RuntimeError(
                f"memmap.as_tensor() can only be called with MemmapTensors stored on CPU. Got device={self.device}."
            )
        # TorchSnapshot doesn't know how to stream MemmapTensor, so we view MemmapTensor
        # as a Tensor for saving and loading purposes. This doesn't incur any copy.
        if self._index:
            indexed_memmap = self._get_item(self._index[0], self.memmap_array)
            for _idx in self._index[1:]:
                indexed_memmap = self._get_item(_idx, indexed_memmap)
            return tensor_from_memoryview(
                dtype=self.dtype,
                shape=list(self.shape),
                mv=memoryview(indexed_memmap),
            )
        return tensor_from_memoryview(
            dtype=self.dtype,
            shape=list(self.shape),
            mv=memoryview(self.memmap_array),
        )


def tensor_from_memoryview(
    mv: memoryview, dtype: torch.dtype, shape: Sequence[int]
) -> torch.Tensor:
    # From torchsnapshot
    # PyTorch issues a warning if the given memoryview is non-writable. This is
    # not a concern for torchsnapshot, as tensors created from non-writable
    # buffers are all read-only, intermediate tensors.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.reshape(torch.frombuffer(mv, dtype=dtype), shape)


def _stack(
    sequence_of_memmap: Sequence[MemmapTensor],
    dim: int,
    out: torch.Tensor | MemmapTensor | None = None,
) -> torch.Tensor:
    list_of_tensors = [
        a._tensor if isinstance(a, MemmapTensor) else a for a in sequence_of_memmap
    ]
    if isinstance(out, MemmapTensor):
        list_of_tensors = [tensor.cpu() for tensor in list_of_tensors]
        return torch.stack(list_of_tensors, dim, out=out._tensor_from_numpy)
    else:
        return torch.stack(list_of_tensors, dim, out=out)


implements_for_memmap(torch.stack)(_stack)


def _unbind(memmap: MemmapTensor, dim: int) -> tuple[torch.Tensor, ...]:
    return memmap.unbind(dim)


implements_for_memmap(torch.unbind)(_unbind)


def _tensor(memmap: MemmapTensor) -> torch.Tensor:
    return memmap._tensor


implements_for_memmap(torch.tensor)(_tensor)


def _cat(
    sequence_of_memmap: Sequence[MemmapTensor],
    dim: int,
    out: torch.Tensor | MemmapTensor | None = None,
) -> torch.Tensor:
    list_of_tensors = [
        a._tensor if isinstance(a, MemmapTensor) else a for a in sequence_of_memmap
    ]
    return torch.cat(list_of_tensors, dim, out=out)


implements_for_memmap(torch.cat)(_cat)


def set_transfer_ownership(memmap: MemmapTensor, value: bool = True) -> None:
    """Changes the transfer_ownership attribute of a MemmapTensor."""
    if isinstance(memmap, MemmapTensor):
        memmap.set_transfer_ownership(value)


def memmap_tensor_as_tensor(
    mem_map_tensor: torch.Tensor | MemmapTensor,
) -> torch.Tensor:
    if not isinstance(mem_map_tensor, MemmapTensor):
        return mem_map_tensor
    # TorchSnapshot doesn't know how to stream MemmapTensor, so we view MemmapTensor
    # as a Tensor for saving and loading purposes. This doesn't incur any copy.
    return tensor_from_memoryview(
        dtype=mem_map_tensor.dtype,
        shape=list(mem_map_tensor.shape),
        mv=memoryview(mem_map_tensor._memmap_array),
    )

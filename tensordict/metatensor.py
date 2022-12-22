# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
from numbers import Number
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from tensordict.memmap import MemmapTensor
from tensordict.utils import (
    _dtype as _dtype_fn,
    _getitem_batch_size,
    _shape as _shape_fn,
    DEVICE_TYPING,
    INDEX_TYPING,
)

try:
    from torchrec import KeyedJaggedTensor

    _has_torchrec = True
except ImportError as err:
    _has_torchrec = False

    class KeyedJaggedTensor:
        pass

    TORCHREC_ERR = str(err)

META_HANDLED_FUNCTIONS = {}


def implements_for_meta(torch_function) -> Callable:
    """Register a torch function override for ScalarTensor."""

    @functools.wraps(torch_function)
    def decorator(func):
        META_HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


class MetaTensor:
    """MetaTensor is a custom class that stores the meta-information about a tensor without requiring to access the tensor.

    This is intended to be used with tensors that have a high access cost.
    MetaTensor supports more operations than tensors on 'meta' device (
    `torch.tensor(..., device='meta')`).
    For instance, MetaTensor supports some operations on its shape and device,
    such as :obj:`mt.to(device)`, :obj:`mt.view(*new_shape)`, :obj:`mt.expand(
    *expand_shape)` etc.

    Args:
        shape (iterable of integers): shape of the tensor. If the first
            element of "shape" is a torch.Tensor, the
            MetaTensor is built with this tensor specs.
        device (int, str or torch.device): device on which the tensor is
            stored.
        dtype (torch.dtype): tensor dtype.
        requires_grad (bool): tensor requires_grad.

    Examples:
        >>> meta1 = MetaTensor(3,4, device=torch.device("cpu"))
        >>> meta2 = MetaTensor(torch.randn(3,4,device="cuda:0",
        ...    dtype=torch.double))
        >>> assert meta1.device != meta2.device
        >>> assert meta1.dtype != meta2.dtype
        >>> assert meta1.expand(2, 3, 4).shape == torch.Size([2, 3, 4])
        >>> assert torch.stack([MetaTensor(3,4) for _ in range(10)],
        ...    1).shape == torch.Size([3, 10, 4])
    """

    @classmethod
    def __new__(cls, *args, **kwargs):
        cls._shape = None
        cls._requires_grad = None
        cls._ndimension = None
        cls._device = None
        cls._dtype = None
        cls._numel = None
        cls._is_shared = None
        cls._is_memmap = None
        cls._is_kjt = None
        cls._is_tensordict = None
        cls._tensor = None
        cls._class_name = None
        return super().__new__(cls)

    @property
    def shape(self):
        _shape = self._shape
        if _shape is None:
            _shape = self._shape = _shape_fn(self._tensor)
        return _shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    @property
    def device(self):
        _device = self._device
        if _device is None:
            _device = self._device = self._tensor.device
        return _device

    @property
    def dtype(self):
        _dtype = self._dtype
        if _dtype is None and not self.is_tensordict():
            _dtype = self._dtype = _dtype_fn(self._tensor)
        return _dtype

    def is_tensordict(self):
        _is_tensordict = self._is_tensordict
        if _is_tensordict is None:
            _is_tensordict = self._is_tensordict = (
                not isinstance(self._tensor, torch.Tensor)
                and not self.is_memmap()
                and not self.is_kjt()
            )
        return _is_tensordict

    def is_memmap(self):
        _is_memmap = self._is_memmap
        if _is_memmap is None:
            _is_memmap = self._is_memmap = isinstance(self._tensor, MemmapTensor)
        return _is_memmap

    def is_kjt(self):
        _is_kjt = self._is_kjt
        if _is_kjt is None:
            _is_kjt = self._is_kjt = isinstance(self._tensor, KeyedJaggedTensor)
        return _is_kjt

    @property
    def _ndim(self):
        _ndimension = self._ndimension
        if _ndimension is None:
            self._ndimension = _ndimension = len(self.shape)
        return _ndimension

    @_ndim.setter
    def _ndim(self, value):
        self._ndimension = value

    def ndimension(self):
        return self._ndim

    def __init__(
        self,
        *shape: Union[int, torch.Tensor, "MemmapTensor"],
        device: Optional[DEVICE_TYPING] = "cpu",
        dtype: torch.dtype = None,
        _is_shared: Optional[bool] = None,
        _is_memmap: Optional[bool] = None,
        _is_tensordict: Optional[bool] = None,
        _is_kjt: Optional[bool] = None,
        _repr_tensordict: Optional[str] = None,
    ):
        tensor = None
        if len(shape) == 1 and not isinstance(shape[0], (Number,)):
            tensor = shape[0]
            self._tensor = tensor
            return

        if type(shape) is not torch.Size:
            shape = torch.Size(shape)
        self.shape = shape
        self._device = device
        self._dtype = dtype if dtype is not None else torch.get_default_dtype()
        self._ndim = len(shape)
        self._numel = None
        # TODO: is_shared is mutable, hence it should not be stored here
        self._is_shared = bool(_is_shared)
        self._is_memmap = bool(_is_memmap)
        self._is_kjt = bool(_is_kjt)
        self._is_tensordict = bool(_is_tensordict)
        self._repr_tensordict = _repr_tensordict

    @property
    def class_name(self):
        name = self._class_name
        if name:
            return name
        if self._is_tensordict:
            name = "TensorDict"
        elif self._is_memmap:
            name = "MemmapTensor"
        elif self._is_kjt:
            name = "KeyedJaggedTensor"
        elif self.is_shared() and self.device.type != "cuda":
            name = "SharedTensor"
        else:
            name = "Tensor"
        self.name = name
        return name

    def get_repr(self):
        if self.is_tensordict():
            return repr(self._tensor)
        else:
            return f"{self.class_name}({self.shape}, dtype={self.dtype})"

    def memmap_(self) -> MetaTensor:
        """Changes the storage of the MetaTensor to memmap.

        Returns:
            self

        """
        self._is_memmap = True
        self._class_name = "MemmapTensor"
        return self

    def share_memory_(self) -> MetaTensor:
        """Changes the storage of the MetaTensor to shared memory.

        Returns:
            self

        """
        self._is_shared = True
        if (
            self._class_name
            and not self.is_tensordict()
            and not self.is_kjt()
            and not self.is_memmap()
        ):
            self._class_name = (
                "SharedTensor"
                if (self.device is not None and self.device.type != "cuda")
                else "Tensor"
            )
        return self

    def is_shared(self) -> bool:
        if self._is_shared is None:
            self._is_shared = self._tensor.is_shared()
        return self._is_shared

    def numel(self) -> int:
        if self._numel is None:
            self._numel = np.prod(self.shape)
        return self._numel

    def clone(self) -> MetaTensor:
        """Clones the meta-tensor.

        Returns: a new MetaTensor with the same specs.

        """
        return MetaTensor(
            *self.shape,
            device=self.device,
            dtype=self.dtype,
            _is_shared=self.is_shared(),
            _is_memmap=self.is_memmap(),
            _is_tensordict=self.is_tensordict(),
            _repr_tensordict=repr(self),
        )

    def _to_meta(self) -> torch.Tensor:
        return torch.empty(*self.shape, dtype=self.dtype, device="meta")

    def __getitem__(self, item: INDEX_TYPING) -> MetaTensor:
        shape = _getitem_batch_size(self.shape, item)
        return MetaTensor(
            *shape,
            dtype=self.dtype,
            device=self.device,
            _is_shared=self.is_shared(),
            _is_memmap=self.is_memmap(),
            _is_tensordict=self.is_tensordict(),
            _repr_tensordict=repr(self),
        )

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types,
        args: Tuple = (),
        kwargs: Optional[dict] = None,
    ):
        if kwargs is None:
            kwargs = {}
        if func not in META_HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, MetaTensor)) for t in types
        ):
            return NotImplemented
        return META_HANDLED_FUNCTIONS[func](*args, **kwargs)

    def expand(self, *shape: int) -> MetaTensor:
        shape = torch.Size([*shape, *self.shape])
        return MetaTensor(
            *shape,
            device=self.device,
            dtype=self.dtype,
        )

    def __repr__(self) -> str:
        return (
            f"MetaTensor(shape={self.shape}, device={self.device}, "
            f"dtype={self.dtype})"
        )

    def unsqueeze(self, dim: int) -> MetaTensor:
        """Unsqueezes the meta-tensor along the desired dim."""
        clone = self.clone()
        new_shape = []
        shape = list(clone.shape)
        for i in range(len(shape) + 1):
            if i == dim:
                new_shape.append(1)
            else:
                new_shape.append(shape[0])
                shape = shape[1:]
        clone.shape = torch.Size(new_shape)
        return clone

    def squeeze(self, dim: Optional[int] = None) -> MetaTensor:
        """Squeezes the meta-tensor along the desired dim."""
        clone = self.clone()
        shape = clone.shape
        if dim is None:
            new_shape = [i for i in shape if i != 1]
        else:
            new_shape = []
            for i in range(len(shape)):
                if i == dim and shape[0] == 1:
                    shape = shape[1:]
                    continue
                else:
                    new_shape.append(shape[0])
                    shape = shape[1:]
        clone.shape = torch.Size(new_shape)
        return clone

    def permute(self, dims: int) -> MetaTensor:
        """Permutes the dims of the meta-tensor."""
        clone = self.clone()
        new_shape = [self.shape[dim] for dim in dims]
        clone.shape = torch.Size(new_shape)
        return clone

    def view(
        self,
        *shape: Sequence,
        size: Optional[Union[List, Tuple, torch.Size]] = None,
    ) -> MetaTensor:
        """Returns a view of a reshaped meta-tensor."""
        if len(shape) == 0 and size is not None:
            return self.view(*size)
        elif len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
            return self.view(*shape[0])
        elif not isinstance(shape, torch.Size):
            shape = torch.Size(shape)
        new_shape = torch.zeros(
            self.shape,
            device="meta",
        ).view(*shape)
        return MetaTensor(
            new_shape,
            device=self.device,
            dtype=self.dtype,
            _is_shared=self.is_shared(),
            _is_memmap=self.is_memmap(),
            _is_tensordict=self.is_tensordict(),
        )

    def to(self, dest):
        if isinstance(dest, torch.dtype):
            self_copy = MetaTensor(
                *self.shape,
                _is_memmap=self.is_memmap(),
                _is_shared=self.is_shared(),
                _is_tensordict=self.is_tensordict(),
                _is_kjt=self.is_kjt(),
                dtype=dest,
                device=self.device,
            )
        else:
            self_copy = MetaTensor(
                *self.shape,
                _is_memmap=self.is_memmap(),
                _is_shared=self.is_shared(),
                _is_tensordict=self.is_tensordict(),
                _is_kjt=self.is_kjt(),
                dtype=self.dtype,
                device=dest,
            )
        return self_copy


def _stack_meta(
    list_of_meta_tensors: Sequence[MetaTensor],
    dim: int = 0,
    dtype: torch.dtype = torch.float,
    device: DEVICE_TYPING = "cpu",
    safe: bool = False,
) -> MetaTensor:
    if not len(list_of_meta_tensors):
        raise RuntimeError("empty list of meta tensors is not supported")
    is_tensordict = list_of_meta_tensors[0].is_tensordict()
    shape = list_of_meta_tensors[0].shape
    if safe:
        for tensor in list_of_meta_tensors:
            if tensor.shape != shape:
                raise RuntimeError(
                    f"Stacking meta tensors of different shapes is not "
                    f"allowed, got shapes {shape} and {tensor.shape}"
                )
            if is_tensordict and not tensor.is_tensordict():
                raise RuntimeError(
                    "Stacking meta tensors from tensordict and non-tensordict "
                    "inputs is not allowed."
                )
            if tensor.dtype != dtype:
                raise TypeError(
                    f"Stacking meta tensors of different dtype is not "
                    f"allowed, got shapes {dtype} and {tensor.dtype}"
                )

    shape = list(shape)
    shape.insert(dim, len(list_of_meta_tensors))

    return MetaTensor(
        *shape,
        dtype=dtype,
        device=device,
        _is_tensordict=is_tensordict,
    )


@implements_for_meta(torch.stack)
def stack_meta(
    list_of_meta_tensors: Sequence[MetaTensor],
    dim: int = 0,
    safe: bool = False,
) -> MetaTensor:
    """Stacks similar meta-tensors into a single meta-tensor."""
    dtype = (
        list_of_meta_tensors[0].dtype
        if len(list_of_meta_tensors)
        else torch.get_default_dtype()
    )
    device = (
        list_of_meta_tensors[0].device
        if len(list_of_meta_tensors)
        else torch.device("cpu")
    )
    return _stack_meta(
        list_of_meta_tensors,
        dim=dim,
        dtype=dtype,
        device=device,
        safe=safe,
    )

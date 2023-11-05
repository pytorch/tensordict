# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch import memory_format
from torch.types import (
    _bool,
    _dtype,
    _layout,
    _device,
)


def empty_like(
    input: Tensor,
    *,
    memory_format: Optional[memory_format] = None,
    dtype: Optional[_dtype] = None,
    layout: Optional[_layout] = None,
    device: Optional[Union[_device, str, None]] = None,
    pin_memory: Optional[_bool] = False,
    requires_grad: Optional[_bool] = False,
    filename: Optional[str] = None
) -> Tensor:
    shape = input.shape
    if dtype is None:
        dtype = input.dtype
    if device is not None:
        device = torch.device(device)
        if device.type != "cpu":
            raise ValueError
    if filename is None:
        if dtype.is_floating_point:
            size = torch.finfo(dtype).bits // 8 * shape.numel()
        elif dtype.is_complex:
            raise ValueError(
                "Complex-valued tensors are not supported by memory-mapped tensors."
            )
        elif dtype == torch.bool:
            size = shape.numel()
        else:
            # assume integer
            size = torch.iinfo(dtype).bits // 8 * shape.numel()
        handler = FileHandler(size)
        if layout is not None:
            raise ValueError
        if pin_memory:
            raise ValueError
        out = torch.frombuffer(
            memoryview(handler.buffer), dtype=dtype,
            # layout=layout,
            device=device,
            # pin_memory=pin_memory,
            requires_grad=requires_grad
            )
        out = torch.reshape(out, shape)
    else:
        out = torch.from_file(
            str(filename),
            shared=True,
            dtype=dtype,
            size=shape.numel(),
            layout=layout,
            device=device, pin_memory=pin_memory, requires_grad=requires_grad
        ).view(input.shape)
    return out


def zeros_like(
    input: Tensor,
    *,
    memory_format: Optional[memory_format] = None,
    dtype: Optional[_dtype] = None,
    layout: Optional[_layout] = None,
    device: Optional[Union[_device, str, None]] = None,
    pin_memory: Optional[_bool] = False,
    requires_grad: Optional[_bool] = False,
    filename: Optional[str] = None
):
    return empty_like(input, memory_format=memory_format, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad, filename=filename).zero_()

def ones_like(
    input: Tensor,
    *,
    memory_format: Optional[memory_format] = None,
    dtype: Optional[_dtype] = None,
    layout: Optional[_layout] = None,
    device: Optional[Union[_device, str, None]] = None,
    pin_memory: Optional[_bool] = False,
    requires_grad: Optional[_bool] = False,
    filename: Optional[str] = None
):
    return empty_like(input, memory_format=memory_format, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad, filename=filename).fill_(1.0)

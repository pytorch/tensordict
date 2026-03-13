# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings

import torch


class UnbatchedTensor(torch.Tensor):
    """A torch.Tensor subclass whose shape is ignored during batch operations on a TensorDict.

    When stored in a TensorDict, shape operations (indexing, reshape, unsqueeze,
    squeeze, transpose, etc.) on the parent TensorDict will leave this tensor
    unchanged. Data operations (arithmetic, clone, to, etc.) work natively
    through standard PyTorch tensor dispatch.

    Example:
        >>> td = TensorDict(a=UnbatchedTensor(torch.randn(3, 4)), b=torch.randn(2, 3), batch_size=(2,))
        >>> td_reshaped = td.reshape((1, 2))
        >>> td_reshaped["a"] is td["a"]
        True

    Since UnbatchedTensor is a torch.Tensor subclass, retrieving it from a
    TensorDict returns a usable tensor directly:

    Example:
        >>> val = td["a"]
        >>> isinstance(val, torch.Tensor)
        True
        >>> isinstance(val, UnbatchedTensor)
        True

    **Indexed Assignment Behavior:**

    UnbatchedTensor is not affected by indexed assignment operations on the parent TensorDict.
    Since UnbatchedTensor does not follow batch dimensions, operations like ``td[:, :2] = other``
    will skip the UnbatchedTensor entries entirely.

    **Batch Size Computation:**

    UnbatchedTensor is excluded from ``auto_batch_size_()`` computation. The batch size is determined
    solely by regular tensors.

    """

    _pass_through = True

    @staticmethod
    def __new__(cls, data):
        if isinstance(data, cls):
            return data
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data)
        return torch.Tensor._make_subclass(cls, data)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        with torch._C.DisableTorchFunctionSubclass():
            result = func(*args, **kwargs)
        if isinstance(result, torch.Tensor):
            return result.as_subclass(cls)
        if isinstance(result, (tuple, list)):
            return type(result)(
                r.as_subclass(cls) if isinstance(r, torch.Tensor) else r for r in result
            )
        return result

    def __repr__(self):
        with torch._C.DisableTorchFunctionSubclass():
            tensor_repr = repr(self.as_subclass(torch.Tensor))
        return f"UnbatchedTensor({tensor_repr})"

    def __reduce_ex__(self, protocol):
        return (UnbatchedTensor, (self.as_subclass(torch.Tensor),))

    @classmethod
    def _stack_non_tensor(
        cls, list_of_non_tensor, dim: int = 0, raise_if_non_unique=False
    ):
        first = list_of_non_tensor[0]
        ptr = first.data_ptr()
        if any(other.data_ptr() != ptr for other in list_of_non_tensor[1:]):
            warnings.warn(
                "Stacking UnbatchedTensors with different data storage. "
                "Only the first element's data will be kept. "
                "UnbatchedTensor is shape-invariant; if you need different data "
                "per batch element, consider using a regular tensor.",
                stacklevel=2,
            )
        return first

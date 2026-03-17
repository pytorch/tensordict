# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings

import torch


def _has_wrapper_subclass_vmap_fix():
    """Check if MetaConverter handles wrapper subclass storage correctly.

    pytorch/pytorch#176977 fixes a cross-device storage error when
    _make_wrapper_subclass tensors are used as non-batched vmap inputs
    inside torch.compile. Without this fix, the MetaConverter's storage
    memo incorrectly reuses placeholder storages from wrapper subclasses,
    leading to a "cuda:0 vs meta" device mismatch.

    The fix wraps the storage memo block with
    ``if not t.is_traceable_wrapper_subclass:`` before ``s = t.storage``.
    We detect this specific pattern in the source code.
    """
    try:
        import inspect
        import re

        from torch._subclasses.meta_utils import MetaConverter

        src = inspect.getsource(MetaConverter.meta_tensor)
        return bool(
            re.search(
                r"not\s+t\.is_traceable_wrapper_subclass.*\n\s+s = t\.storage", src
            )
        )
    except Exception:
        return False


_HAS_WRAPPER_SUBCLASS_FIX = _has_wrapper_subclass_vmap_fix()


if _HAS_WRAPPER_SUBCLASS_FIX:
    # Fast path: _make_wrapper_subclass + __torch_dispatch__
    # Zero Dynamo overhead — no DisableTorchFunctionSubclass context manager
    # or as_subclass() calls appear in the FX graph.

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

        __torch_function__ = torch._C._disabled_torch_function_impl

        @staticmethod
        def __new__(cls, data):
            if isinstance(data, cls):
                return data
            if not isinstance(data, torch.Tensor):
                data = torch.as_tensor(data)
            kwargs = {}
            if data.layout == torch.strided:
                kwargs["strides"] = data.stride()
                kwargs["storage_offset"] = data.storage_offset()
            r = torch.Tensor._make_wrapper_subclass(
                cls,
                data.shape,
                dtype=data.dtype,
                layout=data.layout,
                device=data.device,
                requires_grad=data.requires_grad,
                **kwargs,
            )
            r._data = data
            return r

        def __tensor_flatten__(self):
            return ["_data"], {}

        @classmethod
        def __tensor_unflatten__(
            cls, inner_tensors, metadata, outer_size, outer_stride
        ):
            return cls(inner_tensors["_data"])

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}

            def unwrap(x):
                if isinstance(x, cls):
                    return x._data
                if isinstance(x, (list, tuple)):
                    return type(x)(unwrap(a) for a in x)
                return x

            result = func(*unwrap(args), **unwrap(kwargs))

            if isinstance(result, torch.Tensor):
                return cls(result)
            if isinstance(result, (tuple, list)):
                return type(result)(
                    cls(r) if isinstance(r, torch.Tensor) else r for r in result
                )
            return result

        def data_ptr(self):
            return self._data.data_ptr()

        def untyped_storage(self):
            return self._data.untyped_storage()

        def numpy(self, *, force=False):
            return self._data.numpy(force=force)

        def tolist(self):
            return self._data.tolist()

        def __repr__(self):
            return f"UnbatchedTensor({self._data!r})"

        def __reduce_ex__(self, protocol):
            return (UnbatchedTensor, (self._data,))

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

else:
    # Safe fallback: _make_subclass + __torch_function__
    # Works on all PyTorch versions but emits extra FX graph nodes
    # (DisableTorchFunctionSubclass enter/exit + as_subclass) under Dynamo.

    class UnbatchedTensor(torch.Tensor):  # type: ignore[no-redef]
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
                        r.as_subclass(cls) if isinstance(r, torch.Tensor) else r
                        for r in result
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

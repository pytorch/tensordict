# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import math
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


class _UnbatchedTensorMixin:
    """Shared Python protocol delegations for ``UnbatchedTensor`` variants."""

    def _base_tensor(self):
        raise NotImplementedError

    @property
    def batch_size(self) -> torch.Size:
        batch_size = getattr(self, "_batch_size", None)
        if batch_size is None:
            return torch.Size()
        return batch_size

    @batch_size.setter
    def batch_size(self, batch_size) -> None:
        self._batch_size = torch.Size(batch_size)

    def copy(self):
        out = type(self)(self._base_tensor())
        batch_size = getattr(self, "_batch_size", None)
        if batch_size is not None:
            out.batch_size = batch_size
        return out

    def _with_batch_size(self, batch_size):
        out = self.copy()
        out.batch_size = batch_size
        return out

    def _add_batch_dim(self, *, in_dim: int, vmap_level: int):
        batch_size = list(self.batch_size)
        if in_dim < 0 and batch_size:
            in_dim %= len(batch_size)
        out = self.copy()
        out.batch_size = batch_size[:in_dim] + batch_size[in_dim + 1 :]
        return out

    def _remove_batch_dim(self, vmap_level: int, batch_size: int, out_dim: int):
        return self._maybe_remove_batch_dim(
            vmap_level=vmap_level, batch_size=batch_size, out_dim=out_dim
        )

    def _maybe_remove_batch_dim(
        self,
        funcname=None,  # noqa: ANN001
        *,
        vmap_level: int,
        batch_size: int,
        out_dim: int | None,
    ):
        if out_dim is None:
            return self
        current_batch_size = list(self.batch_size)
        if out_dim < 0:
            out_dim %= len(current_batch_size) + 1
        current_batch_size.insert(out_dim, batch_size)
        out = self.copy()
        out.batch_size = current_batch_size
        return out

    @staticmethod
    def _batch_size_from_args(args, kwargs=None):
        def _find_batch_size(value):
            if isinstance(value, _UnbatchedTensorMixin):
                batch_size = getattr(value, "_batch_size", None)
                if batch_size is not None:
                    return batch_size
            elif isinstance(value, dict):
                for item in value.values():
                    batch_size = _find_batch_size(item)
                    if batch_size is not None:
                        return batch_size
            elif isinstance(value, (list, tuple)):
                for item in value:
                    batch_size = _find_batch_size(item)
                    if batch_size is not None:
                        return batch_size
            return None

        batch_size = _find_batch_size(args)
        if batch_size is None and kwargs is not None:
            batch_size = _find_batch_size(kwargs)
        return batch_size

    @classmethod
    def _wrap_result(cls, result, batch_size):
        def _wrap(tensor):
            out = cls(tensor)
            if batch_size is not None:
                out.batch_size = batch_size
            return out

        if isinstance(result, torch.Tensor):
            return _wrap(result)
        if isinstance(result, (tuple, list)):
            return type(result)(
                _wrap(item) if isinstance(item, torch.Tensor) else item
                for item in result
            )
        return result

    def __str__(self):
        return str(self._base_tensor())

    def __format__(self, format_spec):
        return self._base_tensor().__format__(format_spec)

    def __bool__(self):
        return bool(self._base_tensor())

    def __int__(self):
        return int(self._base_tensor())

    def __float__(self):
        return float(self._base_tensor())

    def __complex__(self):
        return complex(self._base_tensor())

    def __index__(self):
        return self._base_tensor().__index__()

    def __round__(self, ndigits=None):
        if ndigits is None:
            return round(self._base_tensor())
        return round(self._base_tensor(), ndigits)

    def __trunc__(self):
        return math.trunc(self._base_tensor())

    def __floor__(self):
        return math.floor(self._base_tensor())

    def __ceil__(self):
        return math.ceil(self._base_tensor())

    def __array__(self, dtype=None):
        return self._base_tensor().__array__(dtype)

    def item(self):
        return self._base_tensor().item()

    def numpy(self, *, force=False):
        return self._base_tensor().numpy(force=force)

    def tolist(self):
        return self._base_tensor().tolist()


if _HAS_WRAPPER_SUBCLASS_FIX:
    # Fast path: _make_wrapper_subclass + __torch_dispatch__
    # Zero Dynamo overhead — no DisableTorchFunctionSubclass context manager
    # or as_subclass() calls appear in the FX graph.

    class UnbatchedTensor(_UnbatchedTensorMixin, torch.Tensor):
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

        Args:
            data (torch.Tensor or tensor-like): payload tensor.
            batch_size (torch.Size compatible, optional): TensorDict-facing
                batch-size metadata used by ``torch.vmap``. The payload shape is
                left unchanged.

        """

        _pass_through = True

        __torch_function__ = torch._C._disabled_torch_function_impl

        @staticmethod
        def __new__(cls, data, batch_size=None):
            if isinstance(data, cls):
                if batch_size is not None:
                    return data._with_batch_size(batch_size)
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
            if batch_size is not None:
                r.batch_size = batch_size
            return r

        def _base_tensor(self):
            return self._data

        def __tensor_flatten__(self):
            return ["_data"], {"batch_size": getattr(self, "_batch_size", None)}

        @classmethod
        def __tensor_unflatten__(
            cls, inner_tensors, metadata, outer_size, outer_stride
        ):
            batch_size = None if metadata is None else metadata.get("batch_size")
            return cls(inner_tensors["_data"], batch_size=batch_size)

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}
            batch_size = cls._batch_size_from_args(args, kwargs)

            def unwrap(x):
                if isinstance(x, cls):
                    return x._data
                if isinstance(x, (list, tuple)):
                    return type(x)(unwrap(a) for a in x)
                return x

            result = func(*unwrap(args), **unwrap(kwargs))

            return cls._wrap_result(result, batch_size)

        def data_ptr(self):
            return self._data.data_ptr()

        def untyped_storage(self):
            return self._data.untyped_storage()

        def __repr__(self):
            return f"UnbatchedTensor({self._data!r})"

        def __reduce_ex__(self, protocol):
            return (UnbatchedTensor, (self._data, self.batch_size))

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

    class UnbatchedTensor(_UnbatchedTensorMixin, torch.Tensor):  # type: ignore[no-redef]
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

        Args:
            data (torch.Tensor or tensor-like): payload tensor.
            batch_size (torch.Size compatible, optional): TensorDict-facing
                batch-size metadata used by ``torch.vmap``. The payload shape is
                left unchanged.

        """

        _pass_through = True

        @staticmethod
        def __new__(cls, data, batch_size=None):
            if isinstance(data, cls):
                if batch_size is not None:
                    return data._with_batch_size(batch_size)
                return data
            if not isinstance(data, torch.Tensor):
                data = torch.as_tensor(data)
            out = torch.Tensor._make_subclass(cls, data)
            if batch_size is not None:
                out.batch_size = batch_size
            return out

        def _base_tensor(self):
            with torch._C.DisableTorchFunctionSubclass():
                return self.as_subclass(torch.Tensor)

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}
            batch_size = cls._batch_size_from_args(args, kwargs)
            with torch._C.DisableTorchFunctionSubclass():
                result = func(*args, **kwargs)
                if isinstance(result, torch.Tensor):
                    out = result.as_subclass(cls)
                    if batch_size is not None:
                        out.batch_size = batch_size
                    return out
                if isinstance(result, (tuple, list)):
                    result_out = []
                    for item in result:
                        if isinstance(item, torch.Tensor):
                            item = item.as_subclass(cls)
                            if batch_size is not None:
                                item.batch_size = batch_size
                        result_out.append(item)
                    return type(result)(result_out)
                return result

        def __repr__(self):
            with torch._C.DisableTorchFunctionSubclass():
                tensor_repr = repr(self.as_subclass(torch.Tensor))
            return f"UnbatchedTensor({tensor_repr})"

        def __reduce_ex__(self, protocol):
            return (UnbatchedTensor, (self.as_subclass(torch.Tensor), self.batch_size))

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

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Reductions over the batch dimensions or the leaves of :class:`~tensordict.TensorDictBase`.

The methods live on a mixin that ``tensordict.base`` imports before it
defines ``TensorDictBase``, so this module imports the helpers it needs from
``tensordict.base`` without needing the class itself. A method that uses
``TensorDictBase`` at run time imports it locally. The comment above the
mixin imports in ``tensordict/base.py`` gives the other rules.
"""

from __future__ import annotations

from typing import Callable, Literal, overload, Tuple

import torch
from tensordict._nestedkey import NestedKey
from tensordict._tensorcollection import TensorCollection
from tensordict.base import (
    _is_tensor_collection,
    _NESTED_TENSORS_AS_LISTS,
    NO_DEFAULT,
    Self,
)
from tensordict.utils import _maybe_correct_neg_dim


class _Reductions:
    """Reductions over the batch dimensions or the leaves."""

    def all(self, dim: int | None = None) -> bool | TensorCollection:
        """Checks if all values are True/non-null in the tensordict.

        Args:
            dim (int, optional): if ``None``, returns a boolean indicating
                whether all tensors return `tensor.all() == True`
                If integer, all is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.

        """
        from tensordict._td import TensorDict

        if dim is not None and (dim >= self.batch_dims or dim < -self.batch_dims):
            raise RuntimeError(
                "dim must be greater than or equal to -tensordict.batch_dims and "
                "smaller than tensordict.batch_dims"
            )
        if dim is not None:
            dim = _maybe_correct_neg_dim(dim, self.batch_size)

            names = None
            if self._has_names():
                names = [name for i, name in enumerate(self.names) if i != dim]

            return TensorDict(
                source={key: value.all(dim=dim) for key, value in self.items()},
                batch_size=[b for i, b in enumerate(self.batch_size) if i != dim],
                device=self.device,
                names=names,
            )
        return all(value.all() for value in self.values())

    def any(self, dim: int | None = None) -> bool | TensorCollection:
        """Checks if any value is True/non-null in the tensordict.

        Args:
            dim (int, optional): if ``None``, returns a boolean indicating
                whether all tensors return `tensor.any() == True`.
                If integer, all is called upon the dimension specified if
                and only if this dimension is compatible with
                the tensordict shape.

        """
        from tensordict._td import TensorDict

        if dim is not None and (dim >= self.batch_dims or dim < -self.batch_dims):
            raise RuntimeError(
                "dim must be greater than or equal to -tensordict.batch_dims and "
                "smaller than tensordict.batch_dims"
            )
        if dim is not None:
            dim = _maybe_correct_neg_dim(dim, self.batch_size)

            names = None
            if self._has_names():
                names = [name for i, name in enumerate(self.names) if i != dim]

            return TensorDict(
                source={key: value.any(dim=dim) for key, value in self.items()},
                batch_size=[b for i, b in enumerate(self.batch_size) if i != dim],
                device=self.device,
                names=names,
            )
        return any([value.any() for value in self.values()])

    @overload
    def amin(
        self,
        dim: int | NO_DEFAULT = NO_DEFAULT,
        keepdim: bool = False,
    ) -> Self: ...

    @overload
    def amin(
        self,
        dim: int | NO_DEFAULT = NO_DEFAULT,
        keepdim: bool = False,
        *,
        reduce: bool,
    ) -> Self | torch.Tensor: ...

    def amin(
        self,
        dim: int | NO_DEFAULT = NO_DEFAULT,
        keepdim: bool = False,
        *,
        reduce: bool | None = None,
    ) -> Self | torch.Tensor:  # noqa: D417
        """Returns the minimum values of all elements in the input tensordict.

        Same as :meth:`~.min` with ``return_indices=False``.
        """
        return self._cast_reduction(
            reduction_name="amin",
            dim=dim,
            keepdim=keepdim,
            further_reduce=reduce,
            tuple_ok=False,
            values_only=True,
            call_on_nested=False,
        )

    @overload
    def min(
        self,
        dim: int | NO_DEFAULT = NO_DEFAULT,
        keepdim: bool = False,
        *,
        return_indices: bool = True,
    ) -> Self: ...

    @overload
    def min(
        self,
        dim: int | NO_DEFAULT = NO_DEFAULT,
        keepdim: bool = False,
        *,
        reduce: bool,
        return_indices: bool = True,
    ) -> Self | torch.Tensor: ...

    def min(
        self,
        dim: int | NO_DEFAULT = NO_DEFAULT,
        keepdim: bool = False,
        *,
        reduce: bool | None = None,
        return_indices: bool = True,
    ) -> Self | torch.Tensor:  # noqa: D417
        """Returns the minimum values of all elements in the input tensordict.

        Args:
            dim (int, optional): if ``None``, returns a dimensionless
                tensordict containing the min value of all leaves (if this can be computed).
                If integer, `min` is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.
            keepdim (bool): whether the output tensor has dim retained or not.

        Keyword Args:
            reduce (bool, optional): if ``True``, the reduction will occur across all TensorDict values
                and a single reduced tensor will be returned.
                Defaults to ``False``.
            return_indices (bool, optional): :func:`~torch.min` returns a named tuple with values and indices
                when the ``dim`` argument is passed. The ``TensorDict`` equivalent of this is to return a named tuple
                with ``values`` and ``indices`` tensordicts of identical structure. If ``False``, only the values
                are returned. Defaults to ``True``.

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> td = TensorDict(
            ...     a=torch.randn(3, 4, 5),
            ...     b=TensorDict(
            ...         c=torch.randn(3, 4, 5, 6),
            ...         d=torch.randn(3, 4, 5),
            ...         batch_size=(3, 4, 5),
            ...     ),
            ...     batch_size=(3, 4)
            ... )
            >>> td.min(dim=0)
            torch.return_types.min(
            values=TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([4, 5, 6]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([4]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([4]),
                device=None,
                is_shared=False),
            indices=TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.int64, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([4, 5, 6]), device=cpu, dtype=torch.int64, is_shared=False),
                            d: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.int64, is_shared=False)},
                        batch_size=torch.Size([4]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([4]),
                device=None,
                is_shared=False))
            >>> td.min()
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> td.min(reduce=True)
            tensor(-2.9953)

        """
        result = self._cast_reduction(
            reduction_name="min",
            dim=dim,
            keepdim=keepdim,
            further_reduce=reduce,
            tuple_ok=False,
            values_only=not return_indices,
            call_on_nested=False,
        )
        if dim is not NO_DEFAULT and dim is not None and return_indices:
            # Split the tensordict
            from torch.return_types import min

            values_dict = {}
            indices_dict = {}
            for key in result.keys(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS):
                if key[-1] == "values":
                    values_dict[key] = key[:-1]
                else:
                    indices_dict[key] = key[:-1]
            return min(
                result.split_keys(values_dict, indices_dict)[:2],
            )
        return result

    @overload
    def amax(
        self,
        dim: int | NO_DEFAULT = NO_DEFAULT,
        keepdim: bool = False,
    ) -> Self: ...

    @overload
    def amax(
        self,
        dim: int | NO_DEFAULT = NO_DEFAULT,
        keepdim: bool = False,
        *,
        reduce: bool,
    ) -> Self | torch.Tensor: ...

    def amax(
        self,
        dim: int | NO_DEFAULT = NO_DEFAULT,
        keepdim: bool = False,
        *,
        reduce: bool | None = None,
    ) -> Self | torch.Tensor:  # noqa: D417
        """Returns the maximum values of all elements in the input tensordict.

        Same as :meth:`~.max` with ``return_indices=False``.
        """
        return self._cast_reduction(
            reduction_name="amax",
            dim=dim,
            keepdim=keepdim,
            further_reduce=reduce,
            tuple_ok=False,
            values_only=True,
            call_on_nested=False,
        )

    @overload
    def max(
        self,
        dim: int | NO_DEFAULT = NO_DEFAULT,
        keepdim: bool = False,
        *,
        return_indices: bool = True,
    ) -> Self: ...

    @overload
    def max(
        self,
        dim: int | NO_DEFAULT = NO_DEFAULT,
        keepdim: bool = False,
        *,
        reduce: bool,
        return_indices: bool = True,
    ) -> Self | torch.Tensor: ...

    def max(
        self,
        dim: int | NO_DEFAULT = NO_DEFAULT,
        keepdim: bool = False,
        *,
        reduce: bool | None = None,
        return_indices: bool = True,
    ) -> Self | torch.Tensor:  # noqa: D417
        """Returns the maximum values of all elements in the input tensordict.

        Args:
            dim (int, optional): if ``None``, returns a dimensionless
                tensordict containing the max value of all leaves (if this can be computed).
                If integer, `max` is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.
            keepdim (bool): whether the output tensor has dim retained or not.

        Keyword Args:
            reduce (bool, optional): if ``True``, the reduction will occur across all TensorDict values
                and a single reduced tensor will be returned.
                Defaults to ``False``.
            return_indices (bool, optional): :func:`~torch.max` returns a named tuple with values and indices
                when the ``dim`` argument is passed. The ``TensorDict`` equivalent of this is to return a named tuple
                with ``values`` and ``indices`` tensordicts of identical structure. If ``False``, only the values
                are returned. Defaults to ``True``.

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> td = TensorDict(
            ...     a=torch.randn(3, 4, 5),
            ...     b=TensorDict(
            ...         c=torch.randn(3, 4, 5, 6),
            ...         d=torch.randn(3, 4, 5),
            ...         batch_size=(3, 4, 5),
            ...     ),
            ...     batch_size=(3, 4)
            ... )
            >>> td.max(dim=0)
            torch.return_types.max(
            values=TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([4, 5, 6]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([4]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([4]),
                device=None,
                is_shared=False),
            indices=TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.int64, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([4, 5, 6]), device=cpu, dtype=torch.int64, is_shared=False),
                            d: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.int64, is_shared=False)},
                        batch_size=torch.Size([4]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([4]),
                device=None,
                is_shared=False))
            >>> td.max()
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> td.max(reduce=True)
            tensor(3.2942)

        """
        result = self._cast_reduction(
            reduction_name="max",
            dim=dim,
            keepdim=keepdim,
            further_reduce=reduce,
            tuple_ok=False,
            values_only=not return_indices,
            call_on_nested=False,
        )
        if dim is not NO_DEFAULT and dim is not None and return_indices:
            # Split the tensordict
            from torch.return_types import max

            values_dict = {}
            indices_dict = {}
            for key in result.keys(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS):
                if key[-1] == "values":
                    values_dict[key] = key[:-1]
                else:
                    indices_dict[key] = key[:-1]
            return max(
                result.split_keys(values_dict, indices_dict)[:2],
            )
        return result

    @overload
    def cummin(
        self,
        dim: int,
        *,
        return_indices: bool = True,
    ) -> Self: ...

    @overload
    def cummin(
        self,
        dim: int,
        *,
        reduce: bool,
        return_indices: bool = True,
    ) -> Self | torch.Tensor: ...

    def cummin(
        self,
        dim: int,
        *,
        reduce: bool | None = None,
        return_indices: bool = True,
    ) -> Self | torch.Tensor:  # noqa: D417
        """Returns the cumulative minimum values of all elements in the input tensordict.

        Args:
            dim (int): integer representing the dimension along which to perform the cummin operation.

        Keyword Args:
            reduce (bool, optional): if ``True``, the reduction will occur across all TensorDict values
                and a single reduced tensor will be returned.
                Defaults to ``False``.
            return_indices (bool, optional): :func:`~torch.cummin` returns a named tuple with values and indices
                when the ``dim`` argument is passed. The ``TensorDict`` equivalent of this is to return a named tuple
                with ``values`` and ``indices`` tensordicts of identical structure. If ``False``, only the values
                are returned. Defaults to ``True``.

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> td = TensorDict(
            ...     a=torch.randn(3, 4, 5),
            ...     b=TensorDict(
            ...         c=torch.randn(3, 4, 5, 6),
            ...         d=torch.randn(3, 4, 5),
            ...         batch_size=(3, 4, 5),
            ...     ),
            ...     batch_size=(3, 4)
            ... )
            >>> td.cummin(dim=0)
            torch.return_types.cummin(
            values=TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([3, 4, 5, 6]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([3, 4]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3, 4]),
                device=None,
                is_shared=False),
            indices=TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.int64, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([3, 4, 5, 6]), device=cpu, dtype=torch.int64, is_shared=False),
                            d: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.int64, is_shared=False)},
                        batch_size=torch.Size([3, 4]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3, 4]),
                device=None,
                is_shared=False))
            >>> td = TensorDict(
            ...     a=torch.randn(3, 4, 5),
            ...     b=TensorDict(
            ...         c=torch.randn(3, 4, 5),
            ...         d=torch.randn(3, 4, 5),
            ...         batch_size=(3, 4, 5),
            ...     ),
            ...     batch_size=(3, 4)
            ... )
            >>> td.cummin(reduce=True, dim=0)
            torch.return_types.cummin(...)

        """
        result = self._cast_reduction(
            reduction_name="cummin",
            dim=dim,
            further_reduce=reduce,
            tuple_ok=False,
            values_only=not return_indices,
            call_on_nested=False,
            batch_size=self.batch_size,
        )
        if isinstance(result, (torch.Tensor, torch.return_types.cummin)):
            return result
        if dim is not NO_DEFAULT and return_indices:
            # Split the tensordict
            from torch.return_types import cummin

            values_dict = {}
            indices_dict = {}
            for key in result.keys(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS):
                if key[-1] == "values":
                    values_dict[key] = key[:-1]
                else:
                    indices_dict[key] = key[:-1]
            return cummin(
                result.split_keys(values_dict, indices_dict)[:2],
            )
        return result

    @overload
    def cummax(
        self,
        dim: int,
        *,
        return_indices: bool = True,
    ) -> Self: ...

    @overload
    def cummax(
        self,
        dim: int,
        *,
        reduce: bool,
        return_indices: bool = True,
    ) -> Self | torch.Tensor: ...

    def cummax(
        self,
        dim: int,
        *,
        reduce: bool | None = None,
        return_indices: bool = True,
    ) -> Self | torch.Tensor:  # noqa: D417
        """Returns the cumulative maximum values of all elements in the input tensordict.

        Args:
            dim (int): integer representing the dimension along which to perform the cummax operation.

        Keyword Args:
            reduce (bool, optional): if ``True``, the reduction will occur across all TensorDict values
                and a single reduced tensor will be returned.
                Defaults to ``False``.
            return_indices (bool, optional): :func:`~torch.cummax` returns a named tuple with values and indices
                when the ``dim`` argument is passed. The ``TensorDict`` equivalent of this is to return a named tuple
                with ``values`` and ``indices`` tensordicts of identical structure. If ``False``, only the values
                are returned. Defaults to ``True``.

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> td = TensorDict(
            ...     a=torch.randn(3, 4, 5),
            ...     b=TensorDict(
            ...         c=torch.randn(3, 4, 5, 6),
            ...         d=torch.randn(3, 4, 5),
            ...         batch_size=(3, 4, 5),
            ...     ),
            ...     batch_size=(3, 4)
            ... )
            >>> td.cummax(dim=0)
            torch.return_types.cummax(
            values=TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([3, 4, 5, 6]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([3, 4]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3, 4]),
                device=None,
                is_shared=False),
            indices=TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.int64, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([3, 4, 5, 6]), device=cpu, dtype=torch.int64, is_shared=False),
                            d: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.int64, is_shared=False)},
                        batch_size=torch.Size([3, 4]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3, 4]),
                device=None,
                is_shared=False))
            >>> td = TensorDict(
            ...     a=torch.randn(3, 4, 5),
            ...     b=TensorDict(
            ...         c=torch.randn(3, 4, 5),
            ...         d=torch.randn(3, 4, 5),
            ...         batch_size=(3, 4, 5),
            ...     ),
            ...     batch_size=(3, 4)
            ... )
            >>> td.cummax(reduce=True, dim=0)
            torch.return_types.cummax(...)

        """
        result = self._cast_reduction(
            reduction_name="cummax",
            dim=dim,
            further_reduce=reduce,
            tuple_ok=False,
            values_only=not return_indices,
            call_on_nested=False,
            batch_size=self.batch_size,
        )
        if isinstance(result, (torch.Tensor, torch.return_types.cummax)):
            return result
        if dim is not NO_DEFAULT and return_indices:
            # Split the tensordict
            from torch.return_types import cummax

            values_dict = {}
            indices_dict = {}
            for key in result.keys(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS):
                if key[-1] == "values":
                    values_dict[key] = key[:-1]
                else:
                    indices_dict[key] = key[:-1]
            return cummax(
                result.split_keys(values_dict, indices_dict)[:2],
            )
        return result

    @overload
    def mean(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        dtype: torch.dtype | None = None,
    ) -> Self: ...

    @overload
    def mean(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool,
    ) -> Self | torch.Tensor: ...

    @overload
    def mean(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool | None = None,
        key_transform: Callable[[NestedKey], NestedKey] | None = None,
    ) -> Self | torch.Tensor: ...

    def mean(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool | None = None,
        key_transform: Callable[[NestedKey], NestedKey] | None = None,
    ) -> Self | torch.Tensor:  # noqa: D417
        """Returns the mean value of all elements in the input tensordict.

        Args:
            dim (int, tuple of int, str, optional): if ``None``, returns a dimensionless
                tensordict containing the mean value of all leaves (if this can be computed).
                If integer or tuple of integers, `mean` is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.
                Only the `"feature"` string is currently permitted. Using `dim="feature"` will
                achieve the reduction over all feature dimensions. If `reduce=True`, a tensor of the
                shape of the TensorDict's batch-size will be returned. Otherwise, a new tensordict
                with the same structure as ``self`` with reduced feature dimensions will be returned.
            keepdim (bool): whether the output tensor has dim retained or not.

        Keyword Args:
            dtype (torch.dtype, optional): the desired data type of returned tensor.
                If specified, the input tensor is casted to dtype before the operation is performed.
                This is useful for preventing data type overflows. Default: ``None``.
            reduce (bool, optional): if ``True``, the reduction will occur across all TensorDict values
                and a single reduced tensor will be returned.
                Defaults to ``False``.
            key_transform (Callable[[NestedKey], NestedKey], optional): A function to transform key names.
                If provided, all keys in the result will be transformed using this function.
                For string keys, the function receives a string. For tuple keys, it receives a tuple.
                Only applied when ``reduce=False``. Default: ``None``.

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> td = TensorDict(
            ...     a=torch.randn(3, 4, 5),
            ...     b=TensorDict(
            ...         c=torch.randn(3, 4, 5, 6),
            ...         d=torch.randn(3, 4, 5),
            ...         batch_size=(3, 4, 5),
            ...     ),
            ...     batch_size=(3, 4)
            ... )
            >>> td.mean(dim=0)
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([4, 5, 6]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([4, 5]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([4]),
                device=None,
                is_shared=False)
            >>> td.mean()
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> td.mean(reduce=True)
            tensor(-0.0547)
            >>> td.mean(dim="feature")
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([3, 4, 5]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3, 4]),
                device=None,
                is_shared=False)
            >>> td = TensorDict(
            ...     a=torch.ones(3, 4, 5),
            ...     b=TensorDict(
            ...         c=torch.ones(3, 4, 5),
            ...         d=torch.ones(3, 4, 5),
            ...         batch_size=(3, 4, 5),
            ...     ),
            ...     batch_size=(3, 4)
            ... )
            >>> td.mean(reduce=True, dim="feature")
            tensor([[1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.]])
            >>> td.mean(reduce=True, dim=0)
            tensor([[1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1.]])
            >>> # Using key_transform to add prefix to keys
            >>> td.mean(key_transform=lambda key: f"avg_{key}")
            TensorDict(
                fields={
                    avg_a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    avg_b: TensorDict(
                        fields={
                            avg_c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                            avg_d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)


        """
        # if dim is NO_DEFAULT and not keepdim:
        #     dim = None
        #     keepdim = False
        result = self._cast_reduction(
            reduction_name="mean",
            dim=dim,
            keepdim=keepdim,
            dtype=dtype,
            further_reduce=reduce,
        )
        if key_transform is not None and not reduce:
            result = result._transform_keys(key_transform)
        return result

    @overload
    def nanmean(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        dtype: torch.dtype | None = None,
    ) -> Self: ...

    @overload
    def nanmean(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool,
    ) -> Self | torch.Tensor: ...

    def nanmean(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool | None = None,
    ) -> Self | torch.Tensor:  # noqa: D417
        """Returns the mean of all non-NaN elements in the input tensordict.

        Args:
            dim (int, tuple of int, optional): if ``None``, returns a dimensionless
                tensordict containing the mean value of all leaves (if this can be computed).
                If integer or tuple of integers, `mean` is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.
                Only the `"feature"` string is currently permitted. Using `dim="feature"` will
                achieve the reduction over all feature dimensions. If `reduce=True`, a tensor of the
                shape of the TensorDict's batch-size will be returned. Otherwise, a new tensordict
                with the same structure as ``self`` with reduced feature dimensions will be returned.
            keepdim (bool): whether the output tensor has dim retained or not.

        Keyword Args:
            dtype (torch.dtype, optional): the desired data type of returned tensor.
                If specified, the input tensor is casted to dtype before the operation is performed.
                This is useful for preventing data type overflows. Default: ``None``.
            reduce (bool, optional): if ``True``, the reduction will occur across all TensorDict values
                and a single reduced tensor will be returned.
                Defaults to ``False``.

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> td = TensorDict(
            ...     a=torch.randn(3, 4, 5),
            ...     b=TensorDict(
            ...         c=torch.randn(3, 4, 5, 6),
            ...         d=torch.randn(3, 4, 5),
            ...         batch_size=(3, 4, 5),
            ...     ),
            ...     batch_size=(3, 4)
            ... )
            >>> td.nanmean(dim=0)
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([4, 5, 6]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([4, 5]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([4]),
                device=None,
                is_shared=False)
            >>> td.nanmean()
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> td.nanmean(reduce=True)
            tensor(-0.0547)
            >>> td.nanmean(dim="feature")
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([3, 4, 5]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3, 4]),
                device=None,
                is_shared=False)
            >>> td = TensorDict(
            ...     a=torch.ones(3, 4, 5),
            ...     b=TensorDict(
            ...         c=torch.ones(3, 4, 5),
            ...         d=torch.ones(3, 4, 5),
            ...         batch_size=(3, 4, 5),
            ...     ),
            ...     batch_size=(3, 4)
            ... )
            >>> td.nanmean(reduce=True, dim="feature")
            tensor([[1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.]])
            >>> td.nanmean(reduce=True, dim=0)
            tensor([[1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1.]])

        """
        return self._cast_reduction(
            reduction_name="nanmean",
            keepdim=keepdim,
            dim=dim,
            dtype=dtype,
            further_reduce=reduce,
        )

    @overload
    def prod(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        dtype: torch.dtype | None = None,
    ) -> Self: ...

    @overload
    def prod(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool,
    ) -> Self | torch.Tensor: ...

    def prod(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool | None = None,
    ) -> Self | torch.Tensor:  # noqa: D417
        """Returns the produce of values of all elements in the input tensordict.

        Args:
            dim (int, tuple of int, optional): if ``None``, returns a dimensionless
                tensordict containing the prod value of all leaves (if this can be computed).
                If integer or tuple of integers, `prod` is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.
                Only the `"feature"` string is currently permitted. Using `dim="feature"` will
                achieve the reduction over all feature dimensions. If `reduce=True`, a tensor of the
                shape of the TensorDict's batch-size will be returned. Otherwise, a new tensordict
                with the same structure as ``self`` with reduced feature dimensions will be returned.
            keepdim (bool): whether the output tensor has dim retained or not.

        Keyword Args:
            dtype (torch.dtype, optional): the desired data type of returned tensor.
                If specified, the input tensor is casted to dtype before the operation is performed.
                This is useful for preventing data type overflows. Default: ``None``.
            reduce (bool, optional): if ``True``, the reduction will occur across all TensorDict values
                and a single reduced tensor will be returned.
                Defaults to ``False``.

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> td = TensorDict(
            ...     a=torch.randn(3, 4, 5),
            ...     b=TensorDict(
            ...         c=torch.randn(3, 4, 5, 6),
            ...         d=torch.randn(3, 4, 5),
            ...         batch_size=(3, 4, 5),
            ...     ),
            ...     batch_size=(3, 4)
            ... )
            >>> td.prod(dim=0)
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([4, 5, 6]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([4, 5]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([4]),
                device=None,
                is_shared=False)
            >>> td.prod()
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> td.prod(reduce=True)
            tensor(-0.)
            >>> td.prod(dim="feature")
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([3, 4, 5]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3, 4]),
                device=None,
                is_shared=False)
            >>> td = TensorDict(
            ...     a=torch.ones(3, 4, 5),
            ...     b=TensorDict(
            ...         c=torch.ones(3, 4, 5),
            ...         d=torch.ones(3, 4, 5),
            ...         batch_size=(3, 4, 5),
            ...     ),
            ...     batch_size=(3, 4)
            ... )
            >>> td.prod(reduce=True, dim="feature")
            tensor([[1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.]])
            >>> td.prod(reduce=True, dim=0)
            tensor([[1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1.]])

        """
        result = self._cast_reduction(
            reduction_name="prod",
            dim=dim,
            keepdim=False,
            tuple_ok=False,
            dtype=dtype,
            further_reduce=reduce,
        )
        if keepdim:
            if isinstance(dim, tuple):
                dim = dim[0]
            if dim is not None and dim is not NO_DEFAULT:
                result = result.unsqueeze(dim)
                if not reduce and self._has_names():
                    # keep the name of the reduced dim, as sum and mean do
                    result.names = self.names
            else:
                result = result.reshape([1 for _ in self.shape])
        return result

    @overload
    def sum(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        dtype: torch.dtype | None = None,
    ) -> Self: ...

    @overload
    def sum(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool,
    ) -> Self | torch.Tensor: ...

    @overload
    def sum(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool | None = None,
        key_transform: Callable[[NestedKey], NestedKey] | None = None,
    ) -> Self | torch.Tensor: ...

    def sum(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool | None = None,
        key_transform: Callable[[NestedKey], NestedKey] | None = None,
    ) -> Self | torch.Tensor:  # noqa: D417
        """Returns the sum value of all elements in the input tensordict.

        Args:
            dim (int, tuple of int, optional): if ``None``, returns a dimensionless
                tensordict containing the sum value of all leaves (if this can be computed).
                If integer or tuple of integers, `sum` is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.
                Only the `"feature"` string is currently permitted. Using `dim="feature"` will
                achieve the reduction over all feature dimensions. If `reduce=True`, a tensor of the
                shape of the TensorDict's batch-size will be returned. Otherwise, a new tensordict
                with the same structure as ``self`` with reduced feature dimensions will be returned.
            keepdim (bool): whether the output tensor has dim retained or not.

        Keyword Args:
            dtype (torch.dtype, optional): the desired data type of returned tensor.
                If specified, the input tensor is casted to dtype before the operation is performed.
                This is useful for preventing data type overflows. Default: ``None``.
            reduce (bool, optional): if ``True``, the reduction will occur across all TensorDict values
                and a single reduced tensor will be returned.
                Defaults to ``False``.
            key_transform (Callable[[NestedKey], NestedKey], optional): A function to transform key names.
                If provided, all keys in the result will be transformed using this function.
                For string keys, the function receives a string. For tuple keys, it receives a tuple.
                Only applied when ``reduce=False``. Default: ``None``.

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> td = TensorDict(
            ...     a=torch.randn(3, 4, 5),
            ...     b=TensorDict(
            ...         c=torch.randn(3, 4, 5, 6),
            ...         d=torch.randn(3, 4, 5),
            ...         batch_size=(3, 4, 5),
            ...     ),
            ...     batch_size=(3, 4)
            ... )
            >>> td.sum(dim=0)
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([4, 5, 6]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([4, 5]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([4]),
                device=None,
                is_shared=False)
            >>> td.sum()
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> td.sum(reduce=True)
            tensor(-0.)
            >>> td.sum(dim="feature")
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([3, 4, 5]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3, 4]),
                device=None,
                is_shared=False)
            >>> td = TensorDict(
            ...     a=torch.ones(3, 4, 5),
            ...     b=TensorDict(
            ...         c=torch.ones(3, 4, 5),
            ...         d=torch.ones(3, 4, 5),
            ...         batch_size=(3, 4, 5),
            ...     ),
            ...     batch_size=(3, 4)
            ... )
            >>> td.sum(reduce=True, dim="feature")
            tensor([[15., 15., 15., 15.],
                    [15., 15., 15., 15.],
                    [15., 15., 15., 15.]])
            >>> td.sum(reduce=True, dim=0)
            tensor([[9., 9., 9., 9., 9.],
                    [9., 9., 9., 9., 9.],
                    [9., 9., 9., 9., 9.],
                    [9., 9., 9., 9., 9.]])

        """
        result = self._cast_reduction(
            reduction_name="sum",
            dim=dim,
            keepdim=keepdim,
            dtype=dtype,
            further_reduce=reduce,
        )
        if key_transform is not None and not reduce:
            result = result._transform_keys(key_transform)
        return result

    @overload
    def nansum(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        dtype: torch.dtype | None = None,
    ) -> Self: ...

    @overload
    def nansum(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool,
    ) -> Self | torch.Tensor: ...

    def nansum(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        dtype: torch.dtype | None = None,
        reduce: bool | None = None,
    ) -> Self | torch.Tensor:  # noqa: D417
        """Returns the sum of all non-NaN elements in the input tensordict.

        Args:
            dim (int, tuple of int, optional): if ``None``, returns a dimensionless
                tensordict containing the sum value of all leaves (if this can be computed).
                If integer or tuple of integers, `sum` is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.
                Only the `"feature"` string is currently permitted. Using `dim="feature"` will
                achieve the reduction over all feature dimensions. If `reduce=True`, a tensor of the
                shape of the TensorDict's batch-size will be returned. Otherwise, a new tensordict
                with the same structure as ``self`` with reduced feature dimensions will be returned.
            keepdim (bool): whether the output tensor has dim retained or not.

        Keyword Args:
            dtype (torch.dtype, optional): the desired data type of returned tensor.
                If specified, the input tensor is casted to dtype before the operation is performed.
                This is useful for preventing data type overflows. Default: ``None``.
            reduce (bool, optional): if ``True``, the reduction will occur across all TensorDict values
                and a single reduced tensor will be returned.
                Defaults to ``False``.

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> td = TensorDict(
            ...     a=torch.randn(3, 4, 5),
            ...     b=TensorDict(
            ...         c=torch.randn(3, 4, 5, 6),
            ...         d=torch.randn(3, 4, 5),
            ...         batch_size=(3, 4, 5),
            ...     ),
            ...     batch_size=(3, 4)
            ... )
            >>> td.nansum(dim=0)
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([4, 5, 6]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([4, 5]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([4]),
                device=None,
                is_shared=False)
            >>> td.nansum()
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> td.nansum(reduce=True)
            tensor(-0.)
            >>> td.nansum(dim="feature")
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([3, 4, 5]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3, 4]),
                device=None,
                is_shared=False)
            >>> td = TensorDict(
            ...     a=torch.ones(3, 4, 5),
            ...     b=TensorDict(
            ...         c=torch.ones(3, 4, 5),
            ...         d=torch.ones(3, 4, 5),
            ...         batch_size=(3, 4, 5),
            ...     ),
            ...     batch_size=(3, 4)
            ... )
            >>> td.nansum(reduce=True, dim="feature")
            tensor([[15., 15., 15., 15.],
                    [15., 15., 15., 15.],
                    [15., 15., 15., 15.]])
            >>> td.nansum(reduce=True, dim=0)
            tensor([[9., 9., 9., 9., 9.],
                    [9., 9., 9., 9., 9.],
                    [9., 9., 9., 9., 9.],
                    [9., 9., 9., 9., 9.]])

        """
        return self._cast_reduction(
            reduction_name="nansum",
            dim=dim,
            keepdim=keepdim,
            dtype=dtype,
            further_reduce=reduce,
        )

    @overload
    def std(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        correction: int = 1,
    ) -> Self: ...

    @overload
    def std(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        correction: int = 1,
        reduce: bool,
    ) -> Self | torch.Tensor: ...

    @overload
    def std(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        correction: int = 1,
        reduce: bool | None = None,
        key_transform: Callable[[NestedKey], NestedKey] | None = None,
    ) -> Self | torch.Tensor: ...

    def std(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        correction: int = 1,
        reduce: bool | None = None,
        key_transform: Callable[[NestedKey], NestedKey] | None = None,
    ) -> Self | torch.Tensor:  # noqa: D417
        """Returns the standard deviation value of all elements in the input tensordict.

        Args:
            dim (int, tuple of int, optional): if ``None``, returns a dimensionless
                tensordict containing the sum value of all leaves (if this can be computed).
                If integer or tuple of integers, `std` is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.
                Only the `"feature"` string is currently permitted. Using `dim="feature"` will
                achieve the reduction over all feature dimensions. If `reduce=True`, a tensor of the
                shape of the TensorDict's batch-size will be returned. Otherwise, a new tensordict
                with the same structure as ``self`` with reduced feature dimensions will be returned.
            keepdim (bool): whether the output tensor has dim retained or not.

        Keyword Args:
            correction (int): difference between the sample size and sample degrees of freedom.
                Defaults to Bessel's correction, correction=1.
            reduce (bool, optional): if ``True``, the reduction will occur across all TensorDict values
                and a single reduced tensor will be returned.
                Defaults to ``False``.

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> td = TensorDict(
            ...     a=torch.randn(3, 4, 5),
            ...     b=TensorDict(
            ...         c=torch.randn(3, 4, 5, 6),
            ...         d=torch.randn(3, 4, 5),
            ...         batch_size=(3, 4, 5),
            ...     ),
            ...     batch_size=(3, 4)
            ... )
            >>> td.std(dim=0)
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([4, 5, 6]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([4, 5]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([4]),
                device=None,
                is_shared=False)
            >>> td.std()
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> td.std(reduce=True)
            tensor(1.0006)
            >>> td.std(dim="feature")
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([3, 4, 5]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3, 4]),
                device=None,
                is_shared=False)
            >>> td = TensorDict(
            ...     a=torch.ones(3, 4, 5),
            ...     b=TensorDict(
            ...         c=torch.ones(3, 4, 5),
            ...         d=torch.ones(3, 4, 5),
            ...         batch_size=(3, 4, 5),
            ...     ),
            ...     batch_size=(3, 4)
            ... )
            >>> td.std(reduce=True, dim="feature")
            tensor([[0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.]])
            >>> td.std(reduce=True, dim=0)
            tensor([[0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0.]])

        """
        result = self._cast_reduction(
            reduction_name="std",
            dim=dim,
            keepdim=keepdim,
            correction=correction,
            further_reduce=reduce,
        )
        if key_transform is not None and not reduce:
            result = result._transform_keys(key_transform)
        return result

    @overload
    def var(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        correction: int = 1,
    ) -> Self: ...

    @overload
    def var(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        correction: int = 1,
        reduce: bool,
    ) -> Self | torch.Tensor: ...

    @overload
    def var(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        correction: int = 1,
        reduce: bool | None = None,
        key_transform: Callable[[NestedKey], NestedKey] | None = None,
    ) -> Self | torch.Tensor: ...

    def var(
        self,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        correction: int = 1,
        reduce: bool | None = None,
        key_transform: Callable[[NestedKey], NestedKey] | None = None,
    ) -> Self | torch.Tensor:  # noqa: D417
        """Returns the variance value of all elements in the input tensordict.

        Args:
            dim (int, tuple of int, optional): if ``None``, returns a dimensionless
                tensordict containing the sum value of all leaves (if this can be computed).
                If integer or tuple of integers, `var` is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.
                Only the `"feature"` string is currently permitted. Using `dim="feature"` will
                achieve the reduction over all feature dimensions. If `reduce=True`, a tensor of the
                shape of the TensorDict's batch-size will be returned. Otherwise, a new tensordict
                with the same structure as ``self`` with reduced feature dimensions will be returned.
            keepdim (bool): whether the output tensor has dim retained or not.

        Keyword Args:
            correction (int): difference between the sample size and sample degrees of freedom.
                Defaults to Bessel's correction, correction=1.
            reduce (bool, optional): if ``True``, the reduction will occur across all TensorDict values
                and a single reduced tensor will be returned.
                Defaults to ``False``.

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> td = TensorDict(
            ...     a=torch.randn(3, 4, 5),
            ...     b=TensorDict(
            ...         c=torch.randn(3, 4, 5, 6),
            ...         d=torch.randn(3, 4, 5),
            ...         batch_size=(3, 4, 5),
            ...     ),
            ...     batch_size=(3, 4)
            ... )
            >>> td.var(dim=0)
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([4, 5, 6]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([4, 5]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([4]),
                device=None,
                is_shared=False)
            >>> td.var()
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> td.var(reduce=True)
            tensor(1.0006)
            >>> td.var(dim="feature")
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([3, 4, 5]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3, 4]),
                device=None,
                is_shared=False)
            >>> td = TensorDict(
            ...     a=torch.ones(3, 4, 5),
            ...     b=TensorDict(
            ...         c=torch.ones(3, 4, 5),
            ...         d=torch.ones(3, 4, 5),
            ...         batch_size=(3, 4, 5),
            ...     ),
            ...     batch_size=(3, 4)
            ... )
            >>> td.var(reduce=True, dim="feature")
            tensor([[0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.]])
            >>> td.var(reduce=True, dim=0)
            tensor([[0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0.]])

        """
        result = self._cast_reduction(
            reduction_name="var",
            dim=dim,
            keepdim=keepdim,
            correction=correction,
            further_reduce=reduce,
        )
        if key_transform is not None and not reduce:
            result = result._transform_keys(key_transform)
        return result

    @overload
    def quantile(
        self,
        q: float | torch.Tensor,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        interpolation: str = "linear",
    ) -> Self: ...

    @overload
    def quantile(
        self,
        q: float | torch.Tensor,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        interpolation: str = "linear",
        reduce: bool,
    ) -> Self | torch.Tensor: ...

    @overload
    def quantile(
        self,
        q: float | torch.Tensor,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        interpolation: str = "linear",
        reduce: bool | None = None,
        key_transform: Callable[[NestedKey], NestedKey] | None = None,
    ) -> Self | torch.Tensor: ...

    def quantile(
        self,
        q: float | torch.Tensor,
        dim: int | Tuple[int] | Literal["feature"] = NO_DEFAULT,
        keepdim: bool = NO_DEFAULT,
        *,
        interpolation: str = "linear",
        reduce: bool | None = None,
        key_transform: Callable[[NestedKey], NestedKey] | None = None,
    ) -> Self | torch.Tensor:  # noqa: D417
        """Returns the q-th quantile of all elements in the input tensordict.

        Args:
            q (float or torch.Tensor): quantile to compute, must be between 0 and 1.
            dim (int, tuple of int, str, optional): if ``None``, returns a dimensionless
                tensordict containing the quantile value of all leaves (if this can be computed).
                If integer or tuple of integers, `quantile` is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.
                Only the `"feature"` string is currently permitted. Using `dim="feature"` will
                achieve the reduction over all feature dimensions. If `reduce=True`, a tensor of the
                shape of the TensorDict's batch-size will be returned. Otherwise, a new tensordict
                with the same structure as ``self`` with reduced feature dimensions will be returned.
            keepdim (bool): whether the output tensor has dim retained or not.

        Keyword Args:
            interpolation (str): interpolation method to use when the desired quantile lies
                between two data points. Options are 'linear', 'lower', 'higher', 'midpoint', and 'nearest'.
                Defaults to 'linear'.
            reduce (bool, optional): if ``True``, the reduction will occur across all TensorDict values
                and a single reduced tensor will be returned.
                Defaults to ``False``.
            key_transform (Callable[[NestedKey], NestedKey], optional): A function to transform key names.
                If provided, all keys in the result will be transformed using this function.
                For string keys, the function receives a string. For tuple keys, it receives a tuple.
                Only applied when ``reduce=False``. Default: ``None``.

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> td = TensorDict(
            ...     a=torch.randn(3, 4, 5),
            ...     b=TensorDict(
            ...         c=torch.randn(3, 4, 5, 6),
            ...         d=torch.randn(3, 4, 5),
            ...         batch_size=(3, 4, 5),
            ...     ),
            ...     batch_size=(3, 4)
            ... )
            >>> td.quantile(0.5, dim=0)  # median along dim 0
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([4, 5, 6]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([4, 5]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([4]),
                device=None,
                is_shared=False)
            >>> td.quantile(0.5)  # median of all elements
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> td.quantile(0.5, reduce=True)  # single median value
            tensor(0.1234)
            >>> td.quantile(0.5, dim="feature")  # median along feature dimensions
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([3, 4, 5]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3, 4]),
                device=None,
                is_shared=False)
            >>> # Multiple quantiles
            >>> td.quantile(torch.tensor([0.25, 0.5, 0.75]), dim=0)
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([3, 4, 5, 6]), device=cpu, dtype=torch.float32, is_shared=False),
                            d: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([3, 4, 5]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3, 4]),
                device=None,
                is_shared=False)

        """
        result = self._cast_reduction(
            reduction_name="quantile",
            dim=dim,
            keepdim=keepdim,
            q=q,
            interpolation=interpolation,
            further_reduce=reduce,
        )
        if key_transform is not None and not reduce:
            result = result._transform_keys(key_transform)
        return result

    def _cast_reduction(
        self,
        *,
        reduction_name,
        dim=NO_DEFAULT,
        keepdim=NO_DEFAULT,
        tuple_ok=True,
        further_reduce: bool,
        values_only: bool = True,
        call_on_nested: bool = True,
        batch_size=None,
        **kwargs,
    ):
        from tensordict._td import TensorDict

        if dim is None and not keepdim:
            # dim=None reduces over all the elements, as an omitted dim does
            dim = NO_DEFAULT
        if further_reduce:
            # It is not very memory-efficient to do this, but it's the easiest to cover all use cases
            if dim is NO_DEFAULT:
                agglomerate = [
                    val.contiguous().flatten()
                    for val in self._values_list(
                        True, True, is_leaf=_NESTED_TENSORS_AS_LISTS
                    )
                ]
                agglomerate = torch.cat(agglomerate, dim=0)
                if reduction_name == "quantile":
                    q = kwargs.pop("q")
                    return getattr(torch, reduction_name)(agglomerate, q, **kwargs)
                return getattr(torch, reduction_name)(agglomerate, **kwargs)
            else:
                agglomerate = list(
                    self._values_list(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS)
                )
                if dim == "feature":
                    agglomerate = [
                        (
                            val.flatten(self.ndim, -1)
                            if val.ndim > self.ndim
                            else val.unsqueeze(-1)
                        )
                        for val in agglomerate
                    ]
                    cat_dim = -1
                    dim = -1
                    keepdim = False
                elif isinstance(dim, tuple):
                    cat_dim = dim[0]
                else:
                    cat_dim = dim
                agglomerate = torch.cat(agglomerate, dim=cat_dim)
                kwargs_copy = {}
                if keepdim is not NO_DEFAULT:
                    kwargs_copy["keepdim"] = keepdim
                if reduction_name == "quantile":
                    q = kwargs.pop("q")
                    kwargs_copy.update(kwargs)
                    return getattr(torch, reduction_name)(
                        agglomerate, q, dim=dim, **kwargs_copy
                    )
                kwargs_copy.update(kwargs)
                return getattr(torch, reduction_name)(
                    agglomerate, dim=dim, **kwargs_copy
                )

        # With a 1-d tensor q, torch.quantile adds a first dim of size len(q)
        # to each leaf, so the batch size of the result starts with it too.
        q_dims = []
        if reduction_name == "quantile":
            q = kwargs["q"]
            if isinstance(q, torch.Tensor) and q.ndim == 1:
                q_dims = [q.shape[0]]

        # IMPORTANT: do not directly access batch_dims (or any other property)
        # via self.batch_dims otherwise a reference cycle is introduced
        def proc_dim(dim, batch_dims, tuple_ok=True):
            if dim is None:
                return dim
            if isinstance(dim, tuple):
                if tuple_ok:
                    return tuple(
                        _d
                        for d in dim
                        for _d in proc_dim(d, batch_dims, tuple_ok=False)
                    )
                return dim
            return (_maybe_correct_neg_dim(dim, None, batch_dims),)

        dim_needs_proc = (dim is not NO_DEFAULT) and (dim not in ("feature",))
        if dim_needs_proc:
            dim = proc_dim(dim, self.batch_dims, tuple_ok=tuple_ok)
            if not tuple_ok and dim is not None:
                dim = dim[0]
        if dim in ("feature",):
            if keepdim:
                raise TypeError("dim='feature' is incompatible with keepdim=True.")

            ndim = self.ndim

            def reduction(val):
                if _is_tensor_collection(type(val)):
                    local_dim = dim
                else:
                    if val.ndim > ndim:
                        val = val.flatten(ndim, -1)
                    else:
                        val = val.unsqueeze(-1)
                    local_dim = -1
                if reduction_name == "quantile":
                    # Make a copy of kwargs to avoid consuming q multiple times
                    kwargs_copy = kwargs.copy()
                    q = kwargs_copy.pop("q")
                    result = getattr(val, reduction_name)(q, local_dim, **kwargs_copy)
                else:
                    result = getattr(val, reduction_name)(dim=local_dim, **kwargs)
                if isinstance(result, tuple):
                    if values_only:
                        result = result.values
                    else:
                        return TensorDict.from_namedtuple(result)
                return result

            if self._has_names():
                names = [None] * len(q_dims) + list(self.names)
            else:
                names = None
            if not call_on_nested:
                raise RuntimeError(
                    f"reduction {reduction_name} must be called with call_on_nested=True when dim='feature'."
                )
            return self._fast_apply(
                reduction,
                call_on_nested=call_on_nested,
                batch_size=torch.Size([*q_dims, *self.batch_size]) if q_dims else None,
                device=self.device,
                names=names,
            )

        elif dim is not NO_DEFAULT or keepdim:
            names = None
            if self._has_names():
                if not keepdim and isinstance(dim, tuple):
                    names = [name for i, name in enumerate(self.names) if i not in dim]
                else:
                    names = [name for i, name in enumerate(self.names) if i != dim]
                names = [None] * len(q_dims) + names
            if dim is not NO_DEFAULT:
                kwargs["dim"] = dim
            if keepdim is not NO_DEFAULT:
                kwargs["keepdim"] = keepdim

            def reduction(val):
                if reduction_name == "quantile":
                    # Make a copy of kwargs to avoid consuming q multiple times
                    kwargs_copy = kwargs.copy()
                    q = kwargs_copy.pop("q")
                    # Handle dim parameter properly for quantile
                    if "dim" in kwargs_copy:
                        dim_val = kwargs_copy.pop("dim")
                        # torch.quantile doesn't support tuple dimensions, so we need to handle this
                        if isinstance(dim_val, tuple):
                            # For tuple dimensions, we'll use the first dimension
                            # This is a limitation of torch.quantile compared to other reductions
                            dim_val = dim_val[0]
                        result = getattr(val, reduction_name)(q, dim_val, **kwargs_copy)
                    else:
                        result = getattr(val, reduction_name)(q, **kwargs_copy)
                else:
                    result = getattr(val, reduction_name)(**kwargs)
                if isinstance(result, tuple):
                    if values_only:
                        result = result.values
                    else:
                        return TensorDict.from_namedtuple(result, batch_size=batch_size)
                return result

            if batch_size is not None:
                pass
            elif dim is not None and dim is not NO_DEFAULT:
                if not keepdim:
                    if isinstance(dim, tuple):
                        batch_size = [
                            b for i, b in enumerate(self.batch_size) if i not in dim
                        ]
                    else:
                        batch_size = [
                            b for i, b in enumerate(self.batch_size) if i != dim
                        ]
                else:
                    if isinstance(dim, tuple):
                        batch_size = [
                            b if i not in dim else 1
                            for i, b in enumerate(self.batch_size)
                        ]
                    else:
                        batch_size = [
                            b if i != dim else 1 for i, b in enumerate(self.batch_size)
                        ]

            else:
                batch_size = [1 for b in self.batch_size]

            return self._fast_apply(
                reduction,
                call_on_nested=call_on_nested,
                batch_size=torch.Size([*q_dims, *batch_size]),
                device=self.device,
                names=names,
            )

        def reduction(val):
            if reduction_name == "quantile":
                # Make a copy of kwargs to avoid consuming q multiple times
                kwargs_copy = kwargs.copy()
                q = kwargs_copy.pop("q")
                return getattr(val, reduction_name)(q, **kwargs_copy)
            return getattr(val, reduction_name)(**kwargs)

        return self._fast_apply(
            reduction,
            call_on_nested=True,
            batch_size=torch.Size([]),
            device=self.device,
            names=None,
        )

    def logsumexp(self, dim=None, keepdim=False, *, out=None):  # noqa: D417
        """Returns the log of summed exponentials of each row of the input tensordict in the given dimension ``dim``. The computation is numerically stabilized.

        If keepdim is ``True``, the output tensor is of the same size as input except in the dimension(s) ``dim`` where it is of size ``1``.
        Otherwise, ``dim`` is squeezed (see :func:`~torch.squeeze`), resulting in the output tensor having 1 (or len(dim)) fewer dimension(s).

        Args:
            dim (int or tuple of ints, optional): the dimension or dimensions to reduce. If ``None``, all batch dimensions of the
                tensordict are reduced.
            keepdim (bool): whether the output tensordict has dim retained or not.

        Keyword Args:
            out (TensorDictBase, optional): the output tensordict.

        """
        if isinstance(dim, int):
            if dim < 0:
                new_dim = (self.ndim + dim,)
            else:
                new_dim = (dim,)
        elif dim is not None:
            new_dim = tuple(self.ndim + _dim if _dim < 0 else _dim for _dim in dim)
        else:
            new_dim = tuple(range(self.ndim))
        if new_dim is not None and any((d < 0) or (d >= self.ndim) for d in new_dim):
            raise ValueError(
                f"The dimension {dim} is incompatible with a tensordict with batch_size {self.batch_size}."
            )
        batch_size = self.batch_size
        if keepdim:
            batch_size = torch.Size(
                [b if i not in new_dim else 1 for i, b in enumerate(batch_size)]
            )
        else:
            batch_size = torch.Size(
                [b for i, b in enumerate(batch_size) if i not in new_dim]
            )
        if out is not None:
            result = self._fast_apply(
                lambda x, y: torch.logsumexp(x, dim=new_dim, keepdim=keepdim, out=y),
                out,
                default=None,
                batch_size=batch_size,
            )
            return out.update(result)

        return self._fast_apply(
            lambda x: torch.logsumexp(x, dim=new_dim, keepdim=keepdim),
            batch_size=batch_size,
        )

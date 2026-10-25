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

import sys
import warnings
from typing import Callable, Iterable, Literal, overload, Tuple

import tensordict.base as _td_base
import torch
from tensordict._deprecation import warn_deprecated
from tensordict._nestedkey import NestedKey
from tensordict._tensorcollection import TensorCollection
from tensordict.base import (
    _is_tensor_collection,
    _NESTED_TENSORS_AS_LISTS,
    _PACKAGE_DIR,
    NO_DEFAULT,
    Self,
)
from tensordict.utils import _maybe_correct_neg_dim

# dim=() reduces over every element, as in torch.sum(x, dim=()), except in
# these reductions, which keep raising for it
_NO_EMPTY_DIM_REDUCTIONS = frozenset({"max", "min", "cummax", "cummin", "quantile"})


def _user_stacklevel() -> int:
    """Returns the ``stacklevel`` that points the caller's warning to the first frame outside tensordict."""
    frame, stacklevel = sys._getframe(1), 1
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame, stacklevel = frame.f_back, stacklevel + 1
    return stacklevel


def _warn_negative_dim_with_reduce(reduction_name: str, dim) -> None:
    """Warns that ``reduce=True`` counts a negative dim from the end of the leaves (#2044)."""
    warnings.warn(
        f"{reduction_name}(dim={dim}, reduce=True) counts a negative dim from "
        "the end of each leaf. Starting with TensorDict 0.17, the default will "
        "change to counting it from the end of the batch dims, as with "
        "reduce=False. Pass a non-negative dim to get the same result in every "
        "version.",
        FutureWarning,
        stacklevel=_user_stacklevel(),
    )


def _warn_tuple_dim(reduction_name: str, dim: tuple) -> None:
    """Warns that ``reduction_name`` reduces only the first dim of the tuple ``dim`` (#2044)."""
    if reduction_name in ("amax", "amin"):
        warnings.warn(
            f"{reduction_name}(dim={dim}) reduces only the first dim of the "
            "tuple, and does not check it against the batch dims. Starting with "
            "TensorDict 0.17, the default will change to reducing every dim of "
            f"the tuple, counted from the batch dims, as torch.{reduction_name} "
            "does.",
            FutureWarning,
            stacklevel=_user_stacklevel(),
        )
    else:
        warn_deprecated(
            f"Passing a tuple as the dim of {reduction_name}(), which reduces "
            "only its first dim,",
            removal="0.17",
            replacement=f"an int dim, as torch.{reduction_name} takes",
            stacklevel=_user_stacklevel(),
        )


def _check_has_tensor_leaves(reduction_name: str, leaves: list) -> None:
    # reduce=True reduces the tensor leaves together; without any, torch.cat
    # would raise an error that does not say why
    if not leaves:
        raise ValueError(
            f"{reduction_name}(reduce=True) reduces the tensor leaves together, "
            "and this tensordict has no tensor leaves."
        )


class _Reductions:
    """Reductions over the batch dimensions or the leaves."""

    def all(self, dim: int | None = None) -> bool | TensorCollection:
        """Returns whether every element of every leaf is ``True`` (nonzero).

        Args:
            dim (int, optional): a batch dimension to reduce. If given, returns a
                new tensordict of boolean leaves without that dimension. A negative
                ``dim`` counts from the end of the batch dimensions. Defaults to
                ``None``, which checks every element and returns a ``bool``.

        Returns:
            A ``bool``, or a new tensordict if ``dim`` is given.

        Raises:
            RuntimeError: if ``dim`` is not a batch dimension.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.tensor([[True, False], [True, True]]), batch_size=[2, 2])
            >>> td.all()
            False
            >>> td.all(dim=1)["a"]
            tensor([False,  True])
        """
        TensorDict = _td_base._TENSORDICT_TYPE

        if dim is not None and (dim >= self.batch_dims or dim < -self.batch_dims):
            raise RuntimeError(
                "dim must be greater than or equal to -tensordict.batch_dims and "
                "smaller than tensordict.batch_dims"
            )
        if dim is not None:
            dim = _maybe_correct_neg_dim(dim, self.batch_size)
            return TensorDict(
                source={key: value.all(dim=dim) for key, value in self.items()},
                batch_size=[b for i, b in enumerate(self.batch_size) if i != dim],
                device=self.device,
                names=self._reduced_names(dim),
            )
        return all(value.all() for value in self._reduction_parts())

    def any(self, dim: int | None = None) -> bool | TensorCollection:
        """Returns whether any element of any leaf is ``True`` (nonzero).

        Args:
            dim (int, optional): a batch dimension to reduce. If given, returns a
                new tensordict of boolean leaves without that dimension. A negative
                ``dim`` counts from the end of the batch dimensions. Defaults to
                ``None``, which checks every element and returns a ``bool``.

        Returns:
            A ``bool``, or a new tensordict if ``dim`` is given.

        Raises:
            RuntimeError: if ``dim`` is not a batch dimension.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.tensor([[False, False], [True, False]]), batch_size=[2, 2])
            >>> td.any()
            True
            >>> td.any(dim=1)["a"]
            tensor([False,  True])
        """
        TensorDict = _td_base._TENSORDICT_TYPE

        if dim is not None and (dim >= self.batch_dims or dim < -self.batch_dims):
            raise RuntimeError(
                "dim must be greater than or equal to -tensordict.batch_dims and "
                "smaller than tensordict.batch_dims"
            )
        if dim is not None:
            dim = _maybe_correct_neg_dim(dim, self.batch_size)
            return TensorDict(
                source={key: value.any(dim=dim) for key, value in self.items()},
                batch_size=[b for i, b in enumerate(self.batch_size) if i != dim],
                device=self.device,
                names=self._reduced_names(dim),
            )
        # A loop, not any() of a generator, which is slower when no part holds a True
        for value in self._reduction_parts():
            if value.any():
                return True
        return False

    def _reduction_parts(self) -> Iterable:
        """The parts whose ``all()`` or ``any()`` give the result of :meth:`all` or :meth:`any` without a dim.

        The base returns the values. :class:`~tensordict.LazyStackedTensorDict`
        returns its members, which need not stack into dense tensors.
        """
        return self.values()

    def _reduced_names(self, dim: int) -> list[str | None] | None:
        """The dim names of the result of :meth:`all` or :meth:`any` along the batch dim ``dim``, or ``None``.

        ``dim`` is already non-negative. The base drops the name of ``dim``.
        :class:`~tensordict.LazyStackedTensorDict` returns ``None`` when its
        members' names differ or the names left are not unique.
        """
        if self._has_names():
            return [name for i, name in enumerate(self.names) if i != dim]
        return None

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
        """Returns the minimum of every leaf, over all its elements or along a batch dimension.

        Like :meth:`~.min` with ``return_indices=False``: the result holds the
        values only.

        Args:
            dim (int, optional): the batch dimension to reduce. A negative ``dim``
                counts from the end of the batch dimensions. A tuple reduces only
                its first dimension and is not checked against the batch dimensions.
                Starting with TensorDict 0.17, the default will change to reducing
                every dimension of the tuple, as :func:`torch.amin` does. Defaults
                to ``None``, which reduces every element of each leaf, and the
                result has an empty batch size.
            keepdim (bool, optional): if ``True``, the reduced batch dimension is
                kept with size 1. Defaults to ``False``.

        Keyword Args:
            reduce (bool, optional): if ``True``, reduces all the leaves together
                and returns a single tensor. Without ``dim``, the result is a
                scalar. With ``dim``, the leaves are concatenated along ``dim``
                first, so they must have the same number of dimensions. With feature
                dimensions, a negative ``dim`` counts from the end of each leaf.
                Starting with TensorDict 0.17, the default will change to counting
                it from the end of the batch dimensions, as with ``reduce=False``.
                Defaults to ``False``.

        Returns:
            A new tensordict, or a tensor if ``reduce=True``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.tensor([[1.0, 5.0], [3.0, 2.0]]), batch_size=[2])
            >>> td.amin()["a"]
            tensor(1.)
            >>> td.amin(dim=0)["a"]
            tensor([1., 2.])
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
        """Returns the minimum of every leaf, over all its elements or along a batch dimension.

        Without ``dim``, the result is a tensordict of scalar leaves. With ``dim``,
        as with :func:`torch.min`, the result is a named tuple of two tensordicts of
        the same structure, ``values`` and ``indices``.

        Args:
            dim (int, optional): the batch dimension to reduce. A negative ``dim``
                counts from the end of the batch dimensions. A tuple reduces only
                its first dimension; passing one is deprecated and will be removed
                in TensorDict 0.17. Defaults to ``None``, which reduces every
                element of each leaf, and the result has an empty batch size.
            keepdim (bool, optional): if ``True``, the reduced batch dimension is
                kept with size 1. Needs an int ``dim``. Defaults to ``False``.

        Keyword Args:
            reduce (bool, optional): if ``True``, reduces all the leaves together
                and returns a single tensor. Without ``dim``, the result is a
                scalar. With ``dim``, the leaves are concatenated along ``dim``
                first, so they must have the same number of dimensions. With feature
                dimensions, a negative ``dim`` counts from the end of each leaf.
                Starting with TensorDict 0.17, the default will change to counting
                it from the end of the batch dimensions, as with ``reduce=False``.
                Defaults to ``False``.
            return_indices (bool, optional): if ``False`` and ``dim`` is given,
                returns only the values tensordict. Defaults to ``True``.

        Returns:
            A new tensordict, or a named tuple ``(values, indices)`` of
            tensordicts when ``dim`` is given and ``return_indices=True``. With
            ``reduce=True``, a tensor, or what :func:`torch.min` returns when
            ``dim`` is given.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> _ = torch.manual_seed(0)
            >>> td = TensorDict(
            ...     a=torch.randn(3, 4, 5),
            ...     b=TensorDict(c=torch.randn(3, 4, 5, 6), batch_size=(3, 4, 5)),
            ...     batch_size=(3, 4),
            ... )
            >>> td.min(dim=0).values.batch_size
            torch.Size([4])
            >>> td.min(dim=0).indices["b", "c"].dtype
            torch.int64
            >>> td.min()["a"].shape
            torch.Size([])
            >>> td.min(reduce=True)
            tensor(-2.8891)
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
        if isinstance(result, torch.return_types.min):
            # reduce=True with a dim returns what torch.min returns, as in
            # cummax and cummin
            return result
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
        """Returns the maximum of every leaf, over all its elements or along a batch dimension.

        Like :meth:`~.max` with ``return_indices=False``: the result holds the
        values only.

        Args:
            dim (int, optional): the batch dimension to reduce. A negative ``dim``
                counts from the end of the batch dimensions. A tuple reduces only
                its first dimension and is not checked against the batch dimensions.
                Starting with TensorDict 0.17, the default will change to reducing
                every dimension of the tuple, as :func:`torch.amax` does. Defaults
                to ``None``, which reduces every element of each leaf, and the
                result has an empty batch size.
            keepdim (bool, optional): if ``True``, the reduced batch dimension is
                kept with size 1. Defaults to ``False``.

        Keyword Args:
            reduce (bool, optional): if ``True``, reduces all the leaves together
                and returns a single tensor. Without ``dim``, the result is a
                scalar. With ``dim``, the leaves are concatenated along ``dim``
                first, so they must have the same number of dimensions. With feature
                dimensions, a negative ``dim`` counts from the end of each leaf.
                Starting with TensorDict 0.17, the default will change to counting
                it from the end of the batch dimensions, as with ``reduce=False``.
                Defaults to ``False``.

        Returns:
            A new tensordict, or a tensor if ``reduce=True``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.tensor([[1.0, 5.0], [3.0, 2.0]]), batch_size=[2])
            >>> td.amax()["a"]
            tensor(5.)
            >>> td.amax(dim=0)["a"]
            tensor([3., 5.])
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
        """Returns the maximum of every leaf, over all its elements or along a batch dimension.

        Without ``dim``, the result is a tensordict of scalar leaves. With ``dim``,
        as with :func:`torch.max`, the result is a named tuple of two tensordicts of
        the same structure, ``values`` and ``indices``.

        Args:
            dim (int, optional): the batch dimension to reduce. A negative ``dim``
                counts from the end of the batch dimensions. A tuple reduces only
                its first dimension; passing one is deprecated and will be removed
                in TensorDict 0.17. Defaults to ``None``, which reduces every
                element of each leaf, and the result has an empty batch size.
            keepdim (bool, optional): if ``True``, the reduced batch dimension is
                kept with size 1. Needs an int ``dim``. Defaults to ``False``.

        Keyword Args:
            reduce (bool, optional): if ``True``, reduces all the leaves together
                and returns a single tensor. Without ``dim``, the result is a
                scalar. With ``dim``, the leaves are concatenated along ``dim``
                first, so they must have the same number of dimensions. With feature
                dimensions, a negative ``dim`` counts from the end of each leaf.
                Starting with TensorDict 0.17, the default will change to counting
                it from the end of the batch dimensions, as with ``reduce=False``.
                Defaults to ``False``.
            return_indices (bool, optional): if ``False`` and ``dim`` is given,
                returns only the values tensordict. Defaults to ``True``.

        Returns:
            A new tensordict, or a named tuple ``(values, indices)`` of
            tensordicts when ``dim`` is given and ``return_indices=True``. With
            ``reduce=True``, a tensor, or what :func:`torch.max` returns when
            ``dim`` is given.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> _ = torch.manual_seed(0)
            >>> td = TensorDict(
            ...     a=torch.randn(3, 4, 5),
            ...     b=TensorDict(c=torch.randn(3, 4, 5, 6), batch_size=(3, 4, 5)),
            ...     batch_size=(3, 4),
            ... )
            >>> td.max(dim=0).values.batch_size
            torch.Size([4])
            >>> td.max(dim=0).indices["b", "c"].dtype
            torch.int64
            >>> td.max()["a"].shape
            torch.Size([])
            >>> td.max(reduce=True)
            tensor(4.1015)
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
        if isinstance(result, torch.return_types.max):
            # reduce=True with a dim returns what torch.max returns, as in
            # cummax and cummin
            return result
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
        """Returns the cumulative minimum of every leaf along a batch dimension.

        As with :func:`torch.cummin`, the result is a named tuple of two
        tensordicts, ``values`` and ``indices``, with the structure and the batch
        size of the tensordict.

        Args:
            dim (int): the batch dimension along which to compute it. A negative
                ``dim`` counts from the end of the batch dimensions. A tuple uses
                only its first dimension; passing one is deprecated and will be
                removed in TensorDict 0.17.

        Keyword Args:
            reduce (bool, optional): if ``True``, concatenates the leaves along
                ``dim``, which requires them to have the same number of dimensions,
                and returns what :func:`torch.cummin` returns for the result. With
                feature dimensions, a negative ``dim`` counts from the end of each
                leaf. Starting with TensorDict 0.17, the default will change to
                counting it from the end of the batch dimensions, as with
                ``reduce=False``. Defaults to ``False``.
            return_indices (bool, optional): if ``False``, returns only the values
                tensordict. Defaults to ``True``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.tensor([3.0, 1.0, 2.0]), batch_size=[3])
            >>> td.cummin(dim=0).values["a"]
            tensor([3., 1., 1.])
            >>> td.cummin(dim=0).indices["a"]
            tensor([0, 1, 1])
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
        """Returns the cumulative maximum of every leaf along a batch dimension.

        As with :func:`torch.cummax`, the result is a named tuple of two
        tensordicts, ``values`` and ``indices``, with the structure and the batch
        size of the tensordict.

        Args:
            dim (int): the batch dimension along which to compute it. A negative
                ``dim`` counts from the end of the batch dimensions. A tuple uses
                only its first dimension; passing one is deprecated and will be
                removed in TensorDict 0.17.

        Keyword Args:
            reduce (bool, optional): if ``True``, concatenates the leaves along
                ``dim``, which requires them to have the same number of dimensions,
                and returns what :func:`torch.cummax` returns for the result. With
                feature dimensions, a negative ``dim`` counts from the end of each
                leaf. Starting with TensorDict 0.17, the default will change to
                counting it from the end of the batch dimensions, as with
                ``reduce=False``. Defaults to ``False``.
            return_indices (bool, optional): if ``False``, returns only the values
                tensordict. Defaults to ``True``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.tensor([1.0, 3.0, 2.0]), batch_size=[3])
            >>> td.cummax(dim=0).values["a"]
            tensor([1., 3., 3.])
            >>> td.cummax(dim=0).indices["a"]
            tensor([0, 1, 1])
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
        """Returns the mean of every leaf, over all its elements or along batch dimensions.

        Args:
            dim (int, tuple of int or str, optional): the batch dimensions to
                reduce. A negative ``dim`` counts from the end of the batch
                dimensions. ``"feature"`` reduces the feature dimensions of each
                leaf instead, and the result keeps the batch size. Defaults to
                ``None``, which reduces every element of each leaf, and the result
                has an empty batch size.
            keepdim (bool, optional): if ``True``, the reduced batch dimensions are
                kept with size 1. Not allowed with ``dim="feature"``. Defaults to
                ``False``.

        Keyword Args:
            dtype (torch.dtype, optional): if given, each leaf is cast to this dtype
                before the reduction, which helps to prevent overflows. Defaults to
                ``None``.
            reduce (bool, optional): if ``True``, reduces all the leaves together
                and returns a single tensor. Without ``dim``, the result is a
                scalar. With an int ``dim``, the leaves are concatenated along
                ``dim`` first, so they must have the same number of dimensions. With
                ``dim="feature"``, the result has the batch size as its shape. With
                feature dimensions, a negative ``dim`` counts from the end of each
                leaf. Starting with TensorDict 0.17, the default will change to
                counting it from the end of the batch dimensions, as with
                ``reduce=False``. Defaults to ``False``.
            key_transform (Callable, optional): a function applied to the name of
                every entry of the result, at every level of nesting, such as
                ``lambda key: f"avg_{key}"``. Ignored when ``reduce=True``. Defaults
                to ``None``.

        Returns:
            A new tensordict, or a tensor if ``reduce=True``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> _ = torch.manual_seed(0)
            >>> td = TensorDict(
            ...     a=torch.randn(3, 4, 5),
            ...     b=TensorDict(c=torch.randn(3, 4, 5, 6), batch_size=(3, 4, 5)),
            ...     batch_size=(3, 4),
            ... )
            >>> td.mean(dim=0).batch_size
            torch.Size([4])
            >>> td.mean(dim=0)["b", "c"].shape
            torch.Size([4, 5, 6])
            >>> td.mean()["a"].shape
            torch.Size([])
            >>> td.mean(dim="feature")["a"].shape
            torch.Size([3, 4])
            >>> td.mean(reduce=True)
            tensor(0.0156)
            >>> list(td.mean(key_transform=lambda key: f"avg_{key}").keys(True, True))
            ['avg_a', ('avg_b', 'avg_c')]

            With ``reduce=True`` and an int ``dim``, the leaves are concatenated
            along ``dim`` before the reduction:

            >>> td = TensorDict(a=torch.ones(3, 4), b=torch.ones(3, 4), batch_size=(3, 4))
            >>> td.mean(dim=0, reduce=True)
            tensor([1., 1., 1., 1.])
            >>> td.mean(dim="feature", reduce=True)
            tensor([[1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.]])
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
        """Returns the mean of the non-NaN elements of every leaf.

        Works as :meth:`~.mean`, but ignores NaN values.

        Args:
            dim (int, tuple of int or str, optional): the batch dimensions to
                reduce. A negative ``dim`` counts from the end of the batch
                dimensions. ``"feature"`` reduces the feature dimensions of each
                leaf instead, and the result keeps the batch size. Defaults to
                ``None``, which reduces every element of each leaf, and the result
                has an empty batch size.
            keepdim (bool, optional): if ``True``, the reduced batch dimensions are
                kept with size 1. Not allowed with ``dim="feature"``. Defaults to
                ``False``.

        Keyword Args:
            dtype (torch.dtype, optional): if given, each leaf is cast to this dtype
                before the reduction, which helps to prevent overflows. Defaults to
                ``None``.
            reduce (bool, optional): if ``True``, reduces all the leaves together
                and returns a single tensor. Without ``dim``, the result is a
                scalar. With an int ``dim``, the leaves are concatenated along
                ``dim`` first, so they must have the same number of dimensions. With
                ``dim="feature"``, the result has the batch size as its shape. With
                feature dimensions, a negative ``dim`` counts from the end of each
                leaf. Starting with TensorDict 0.17, the default will change to
                counting it from the end of the batch dimensions, as with
                ``reduce=False``. Defaults to ``False``.

        Returns:
            A new tensordict, or a tensor if ``reduce=True``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.tensor([[1.0, float("nan")], [3.0, 4.0]]), batch_size=[2])
            >>> td.nanmean()["a"]
            tensor(2.6667)
            >>> td.mean()["a"]
            tensor(nan)
            >>> td.nanmean(dim=0)["a"]
            tensor([2., 4.])
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
        """Returns the product of every leaf, over all its elements or along a batch dimension.

        Args:
            dim (int or str, optional): the batch dimension to reduce. A negative
                ``dim`` counts from the end of the batch dimensions. ``"feature"``
                reduces the feature dimensions of each leaf instead, and the result
                keeps the batch size. A tuple reduces only its first dimension;
                passing one is deprecated and will be removed in TensorDict 0.17.
                Defaults to ``None``, which reduces every element of each leaf, and
                the result has an empty batch size.
            keepdim (bool, optional): if ``True``, the reduced batch dimensions are
                kept with size 1. Not allowed with ``dim="feature"``. Defaults to
                ``False``.

        Keyword Args:
            dtype (torch.dtype, optional): if given, each leaf is cast to this dtype
                before the reduction, which helps to prevent overflows. Defaults to
                ``None``.
            reduce (bool, optional): if ``True``, reduces all the leaves together
                and returns a single tensor. Without ``dim``, the result is a
                scalar. With an int ``dim``, the leaves are concatenated along
                ``dim`` first, so they must have the same number of dimensions. With
                ``dim="feature"``, the result has the batch size as its shape. With
                feature dimensions, a negative ``dim`` counts from the end of each
                leaf. Starting with TensorDict 0.17, the default will change to
                counting it from the end of the batch dimensions, as with
                ``reduce=False``. Defaults to ``False``.

        Returns:
            A new tensordict, or a tensor if ``reduce=True``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.tensor([[1.0, 2.0], [3.0, 4.0]]), batch_size=[2])
            >>> td.prod()["a"]
            tensor(24.)
            >>> td.prod(dim=0)["a"]
            tensor([3., 8.])
            >>> td.prod(dim="feature")["a"]
            tensor([ 2., 12.])
            >>> td.prod(reduce=True)
            tensor(24.)
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
        """Returns the sum of every leaf, over all its elements or along batch dimensions.

        Args:
            dim (int, tuple of int or str, optional): the batch dimensions to
                reduce. A negative ``dim`` counts from the end of the batch
                dimensions. ``"feature"`` reduces the feature dimensions of each
                leaf instead, and the result keeps the batch size. Defaults to
                ``None``, which reduces every element of each leaf, and the result
                has an empty batch size.
            keepdim (bool, optional): if ``True``, the reduced batch dimensions are
                kept with size 1. Not allowed with ``dim="feature"``. Defaults to
                ``False``.

        Keyword Args:
            dtype (torch.dtype, optional): if given, each leaf is cast to this dtype
                before the reduction, which helps to prevent overflows. Defaults to
                ``None``.
            reduce (bool, optional): if ``True``, reduces all the leaves together
                and returns a single tensor. Without ``dim``, the result is a
                scalar. With an int ``dim``, the leaves are concatenated along
                ``dim`` first, so they must have the same number of dimensions. With
                ``dim="feature"``, the result has the batch size as its shape. With
                feature dimensions, a negative ``dim`` counts from the end of each
                leaf. Starting with TensorDict 0.17, the default will change to
                counting it from the end of the batch dimensions, as with
                ``reduce=False``. Defaults to ``False``.
            key_transform (Callable, optional): a function applied to the name of
                every entry of the result, at every level of nesting, such as
                ``lambda key: f"sum_{key}"``. Ignored when ``reduce=True``. Defaults
                to ``None``.

        Returns:
            A new tensordict, or a tensor if ``reduce=True``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> _ = torch.manual_seed(0)
            >>> td = TensorDict(
            ...     a=torch.randn(3, 4, 5),
            ...     b=TensorDict(c=torch.randn(3, 4, 5, 6), batch_size=(3, 4, 5)),
            ...     batch_size=(3, 4),
            ... )
            >>> td.sum(dim=0).batch_size
            torch.Size([4])
            >>> td.sum(dim=0)["b", "c"].shape
            torch.Size([4, 5, 6])
            >>> td.sum()["a"].shape
            torch.Size([])
            >>> td.sum(dim="feature")["a"].shape
            torch.Size([3, 4])
            >>> td.sum(reduce=True)
            tensor(6.5324)
            >>> list(td.sum(key_transform=lambda key: f"sum_{key}").keys(True, True))
            ['sum_a', ('sum_b', 'sum_c')]

            With ``reduce=True`` and an int ``dim``, the leaves are concatenated
            along ``dim`` before the reduction:

            >>> td = TensorDict(a=torch.ones(3, 4), b=torch.ones(3, 4), batch_size=(3, 4))
            >>> td.sum(dim=0, reduce=True)
            tensor([6., 6., 6., 6.])
            >>> td.sum(dim="feature", reduce=True)
            tensor([[2., 2., 2., 2.],
                    [2., 2., 2., 2.],
                    [2., 2., 2., 2.]])
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
        """Returns the sum of the non-NaN elements of every leaf.

        Works as :meth:`~.sum`, but counts NaN values as zero.

        Args:
            dim (int, tuple of int or str, optional): the batch dimensions to
                reduce. A negative ``dim`` counts from the end of the batch
                dimensions. ``"feature"`` reduces the feature dimensions of each
                leaf instead, and the result keeps the batch size. Defaults to
                ``None``, which reduces every element of each leaf, and the result
                has an empty batch size.
            keepdim (bool, optional): if ``True``, the reduced batch dimensions are
                kept with size 1. Not allowed with ``dim="feature"``. Defaults to
                ``False``.

        Keyword Args:
            dtype (torch.dtype, optional): if given, each leaf is cast to this dtype
                before the reduction, which helps to prevent overflows. Defaults to
                ``None``.
            reduce (bool, optional): if ``True``, reduces all the leaves together
                and returns a single tensor. Without ``dim``, the result is a
                scalar. With an int ``dim``, the leaves are concatenated along
                ``dim`` first, so they must have the same number of dimensions. With
                ``dim="feature"``, the result has the batch size as its shape. With
                feature dimensions, a negative ``dim`` counts from the end of each
                leaf. Starting with TensorDict 0.17, the default will change to
                counting it from the end of the batch dimensions, as with
                ``reduce=False``. Defaults to ``False``.

        Returns:
            A new tensordict, or a tensor if ``reduce=True``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.tensor([[1.0, float("nan")], [3.0, 4.0]]), batch_size=[2])
            >>> td.nansum()["a"]
            tensor(8.)
            >>> td.sum()["a"]
            tensor(nan)
            >>> td.nansum(dim=0)["a"]
            tensor([4., 4.])
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
        """Returns the standard deviation of every leaf, over all its elements or along batch dimensions.

        Args:
            dim (int, tuple of int or str, optional): the batch dimensions to
                reduce. A negative ``dim`` counts from the end of the batch
                dimensions. ``"feature"`` reduces the feature dimensions of each
                leaf instead, and the result keeps the batch size. Defaults to
                ``None``, which reduces every element of each leaf, and the result
                has an empty batch size.
            keepdim (bool, optional): if ``True``, the reduced batch dimensions are
                kept with size 1. Not allowed with ``dim="feature"``. Defaults to
                ``False``.

        Keyword Args:
            correction (int, optional): the difference between the sample size and
                the degrees of freedom. Defaults to ``1``, Bessel's correction.
            reduce (bool, optional): if ``True``, reduces all the leaves together
                and returns a single tensor. Without ``dim``, the result is a
                scalar. With an int ``dim``, the leaves are concatenated along
                ``dim`` first, so they must have the same number of dimensions. With
                ``dim="feature"``, the result has the batch size as its shape. With
                feature dimensions, a negative ``dim`` counts from the end of each
                leaf. Starting with TensorDict 0.17, the default will change to
                counting it from the end of the batch dimensions, as with
                ``reduce=False``. Defaults to ``False``.
            key_transform (Callable, optional): a function applied to the name of
                every entry of the result, at every level of nesting, such as
                ``lambda key: f"std_{key}"``. Ignored when ``reduce=True``. Defaults
                to ``None``.

        Returns:
            A new tensordict, or a tensor if ``reduce=True``.

        .. note:: With ``dim="feature"`` and the default ``correction``, a leaf
            without feature dimensions gives NaN, and torch warns that the degrees
            of freedom are ``<= 0``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> _ = torch.manual_seed(0)
            >>> td = TensorDict(
            ...     a=torch.randn(3, 4, 5),
            ...     b=TensorDict(c=torch.randn(3, 4, 5, 6), batch_size=(3, 4, 5)),
            ...     batch_size=(3, 4),
            ... )
            >>> td.std(dim=0).batch_size
            torch.Size([4])
            >>> td.std(dim=0)["b", "c"].shape
            torch.Size([4, 5, 6])
            >>> td.std()["a"].shape
            torch.Size([])
            >>> td.std(dim="feature")["a"].shape
            torch.Size([3, 4])
            >>> td.std(reduce=True)
            tensor(1.0264)
            >>> list(td.std(key_transform=lambda key: f"std_{key}").keys(True, True))
            ['std_a', ('std_b', 'std_c')]

            With ``reduce=True`` and an int ``dim``, the leaves are concatenated
            along ``dim`` before the reduction:

            >>> td = TensorDict(a=torch.ones(3, 4), b=torch.ones(3, 4), batch_size=(3, 4))
            >>> td.std(dim=0, reduce=True)
            tensor([0., 0., 0., 0.])
            >>> td.std(dim="feature", reduce=True)
            tensor([[0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.]])
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
        """Returns the variance of every leaf, over all its elements or along batch dimensions.

        Args:
            dim (int, tuple of int or str, optional): the batch dimensions to
                reduce. A negative ``dim`` counts from the end of the batch
                dimensions. ``"feature"`` reduces the feature dimensions of each
                leaf instead, and the result keeps the batch size. Defaults to
                ``None``, which reduces every element of each leaf, and the result
                has an empty batch size.
            keepdim (bool, optional): if ``True``, the reduced batch dimensions are
                kept with size 1. Not allowed with ``dim="feature"``. Defaults to
                ``False``.

        Keyword Args:
            correction (int, optional): the difference between the sample size and
                the degrees of freedom. Defaults to ``1``, Bessel's correction.
            reduce (bool, optional): if ``True``, reduces all the leaves together
                and returns a single tensor. Without ``dim``, the result is a
                scalar. With an int ``dim``, the leaves are concatenated along
                ``dim`` first, so they must have the same number of dimensions. With
                ``dim="feature"``, the result has the batch size as its shape. With
                feature dimensions, a negative ``dim`` counts from the end of each
                leaf. Starting with TensorDict 0.17, the default will change to
                counting it from the end of the batch dimensions, as with
                ``reduce=False``. Defaults to ``False``.
            key_transform (Callable, optional): a function applied to the name of
                every entry of the result, at every level of nesting, such as
                ``lambda key: f"var_{key}"``. Ignored when ``reduce=True``. Defaults
                to ``None``.

        Returns:
            A new tensordict, or a tensor if ``reduce=True``.

        .. note:: With ``dim="feature"`` and the default ``correction``, a leaf
            without feature dimensions gives NaN, and torch warns that the degrees
            of freedom are ``<= 0``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> _ = torch.manual_seed(0)
            >>> td = TensorDict(
            ...     a=torch.randn(3, 4, 5),
            ...     b=TensorDict(c=torch.randn(3, 4, 5, 6), batch_size=(3, 4, 5)),
            ...     batch_size=(3, 4),
            ... )
            >>> td.var(dim=0).batch_size
            torch.Size([4])
            >>> td.var(dim=0)["b", "c"].shape
            torch.Size([4, 5, 6])
            >>> td.var()["a"].shape
            torch.Size([])
            >>> td.var(dim="feature")["a"].shape
            torch.Size([3, 4])
            >>> td.var(reduce=True)
            tensor(1.0535)
            >>> list(td.var(key_transform=lambda key: f"var_{key}").keys(True, True))
            ['var_a', ('var_b', 'var_c')]

            With ``reduce=True`` and an int ``dim``, the leaves are concatenated
            along ``dim`` before the reduction:

            >>> td = TensorDict(a=torch.ones(3, 4), b=torch.ones(3, 4), batch_size=(3, 4))
            >>> td.var(dim=0, reduce=True)
            tensor([0., 0., 0., 0.])
            >>> td.var(dim="feature", reduce=True)
            tensor([[0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.]])
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
        """Returns the ``q``-th quantile of every leaf, over all its elements or along a batch dimension.

        Args:
            q (float or torch.Tensor): the quantile, between 0 and 1. With a
                1-dimensional tensor of ``n`` quantiles, every leaf of the result
                gets a leading dimension of size ``n``, and so does the batch size
                when ``dim`` is given.
            dim (int or str, optional): the batch dimension to reduce. A negative
                ``dim`` counts from the end of the batch dimensions. ``"feature"``
                reduces the feature dimensions of each leaf instead, and the result
                keeps the batch size. A tuple reduces only its first dimension;
                passing one is deprecated and will be removed in TensorDict 0.17.
                Defaults to ``None``, which reduces every element of each leaf, and
                the result has an empty batch size.
            keepdim (bool, optional): if ``True``, the reduced batch dimensions are
                kept with size 1. Not allowed with ``dim="feature"``. Defaults to
                ``False``.

        Keyword Args:
            interpolation (str, optional): the method used when the quantile lies
                between two data points: ``"linear"``, ``"lower"``, ``"higher"``,
                ``"midpoint"`` or ``"nearest"``. Defaults to ``"linear"``.
            reduce (bool, optional): if ``True``, reduces all the leaves together
                and returns a single tensor. Without ``dim``, the result is a
                scalar. With an int ``dim``, the leaves are concatenated along
                ``dim`` first, so they must have the same number of dimensions. With
                ``dim="feature"``, the result has the batch size as its shape. With
                feature dimensions, a negative ``dim`` counts from the end of each
                leaf. Starting with TensorDict 0.17, the default will change to
                counting it from the end of the batch dimensions, as with
                ``reduce=False``. Defaults to ``False``.
            key_transform (Callable, optional): a function applied to the name of
                every entry of the result, at every level of nesting, such as
                ``lambda key: f"median_{key}"``. Ignored when ``reduce=True``.
                Defaults to ``None``.

        Returns:
            A new tensordict, or a tensor if ``reduce=True``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.arange(5.0), b=torch.arange(5.0) * 2, batch_size=[5])
            >>> td.quantile(0.5)["b"]
            tensor(4.)
            >>> td.quantile(torch.tensor([0.25, 0.75]), dim=0)["a"]
            tensor([1., 3.])
            >>> td.quantile(0.5, reduce=True)  # the median of the ten elements
            tensor(2.5000)
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
        TensorDict = _td_base._TENSORDICT_TYPE

        if (
            isinstance(dim, tuple)
            and not dim
            and reduction_name not in _NO_EMPTY_DIM_REDUCTIONS
        ):
            # dim=() reduces over all the elements, as in torch
            dim = None
        if dim is None and not keepdim:
            # dim=None reduces over all the elements, as an omitted dim does
            dim = NO_DEFAULT
        if further_reduce:
            # It is not very memory-efficient to do this, but it's the easiest to cover all use cases
            if dim is NO_DEFAULT:
                agglomerate = [
                    val.contiguous().flatten() for val in self._values_list(True, True)
                ]
                _check_has_tensor_leaves(reduction_name, agglomerate)
                agglomerate = torch.cat(agglomerate, dim=0)
                if reduction_name == "quantile":
                    q = kwargs.pop("q")
                    return getattr(torch, reduction_name)(agglomerate, q, **kwargs)
                return getattr(torch, reduction_name)(agglomerate, **kwargs)
            else:
                agglomerate = list(self._values_list(True, True))
                # With feature dims, a negative dim counts from the end of the
                # leaves, not of the batch dims
                warn_negative_dim = (
                    bool(agglomerate)
                    and agglomerate[0].ndim != self.ndim
                    and any(
                        isinstance(d, int) and d < 0
                        for d in (dim if isinstance(dim, tuple) else (dim,))
                    )
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
                _check_has_tensor_leaves(reduction_name, agglomerate)
                agglomerate = torch.cat(agglomerate, dim=cat_dim)
                kwargs_copy = {}
                if keepdim is not NO_DEFAULT:
                    kwargs_copy["keepdim"] = keepdim
                if reduction_name == "quantile":
                    q = kwargs.pop("q")
                    kwargs_copy.update(kwargs)
                    result = getattr(torch, reduction_name)(
                        agglomerate, q, dim=dim, **kwargs_copy
                    )
                else:
                    kwargs_copy.update(kwargs)
                    result = getattr(torch, reduction_name)(
                        agglomerate, dim=dim, **kwargs_copy
                    )
                if warn_negative_dim:
                    _warn_negative_dim_with_reduce(reduction_name, dim)
                return result

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
        # Only the first dim of the tuple is reduced, as is. amax and amin
        # give the same result as torch for one dim inside the batch dims.
        tuple_dim = None
        if isinstance(dim, tuple) and dim:
            if reduction_name in (
                "max",
                "min",
                "prod",
                "cummax",
                "cummin",
                "quantile",
            ) or (
                reduction_name in ("amax", "amin")
                and (len(dim) > 1 or not 0 <= dim[0] < self.batch_dims)
            ):
                tuple_dim = dim
        if dim_needs_proc:
            dim = proc_dim(dim, self.batch_dims, tuple_ok=tuple_ok)
            if not tuple_ok and dim is not None:
                dim = dim[0]
            elif reduction_name == "quantile" and dim:
                # torch.quantile takes one dim, so the leaves and the batch
                # size are reduced over the first dim of the tuple only
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
                if keepdim or batch_size is not None:
                    # keepdim=True and the cumulative ops (which pass
                    # batch_size) keep every batch dim, so every name
                    names = list(self.names)
                elif isinstance(dim, tuple):
                    names = [name for i, name in enumerate(self.names) if i not in dim]
                else:
                    names = [name for i, name in enumerate(self.names) if i != dim]
                names = [None] * len(q_dims) + names
            if dim is not NO_DEFAULT:
                kwargs["dim"] = dim
            elif reduction_name in ("sum", "mean"):
                # torch's sum and mean need a dim with keepdim
                kwargs["dim"] = None
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

            result = self._fast_apply(
                reduction,
                call_on_nested=call_on_nested,
                batch_size=torch.Size([*q_dims, *batch_size]),
                device=self.device,
                names=names,
            )
            if tuple_dim is not None:
                _warn_tuple_dim(reduction_name, tuple_dim)
            return result

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
        """Returns the log of the summed exponentials of every leaf along batch dimensions.

        The computation is numerically stable, as with :func:`torch.logsumexp`.
        Unlike :meth:`~.sum`, it never reduces the feature dimensions: without
        ``dim``, it reduces every batch dimension.

        Args:
            dim (int or tuple of int, optional): the batch dimensions to reduce. A
                negative ``dim`` counts from the end of the batch dimensions.
                Defaults to ``None``, which reduces all the batch dimensions.
            keepdim (bool, optional): if ``True``, the reduced batch dimensions are
                kept with size 1. Defaults to ``False``.

        Keyword Args:
            out (TensorDictBase, optional): the tensordict to write the result into,
                which is returned. Defaults to ``None``, which returns a new
                tensordict.

        Raises:
            ValueError: if ``dim`` is not a batch dimension.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(2, 3), batch_size=[2])
            >>> td.logsumexp()["a"]  # log(exp(0) + exp(0))
            tensor([0.6931, 0.6931, 0.6931])
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
        names = None
        if self._has_names():
            if keepdim:
                names = list(self.names)
            else:
                names = [name for i, name in enumerate(self.names) if i not in new_dim]
        if keepdim:
            batch_size = torch.Size(
                [b if i not in new_dim else 1 for i, b in enumerate(batch_size)]
            )
        else:
            batch_size = torch.Size(
                [b for i, b in enumerate(batch_size) if i not in new_dim]
            )
        if out is not None:

            def reduction(x, y):
                if _is_tensor_collection(type(x)):
                    return x.logsumexp(new_dim, keepdim=keepdim, out=y)
                return torch.logsumexp(x, dim=new_dim, keepdim=keepdim, out=y)

            result = self._fast_apply(
                reduction,
                out,
                default=None,
                call_on_nested=True,
                batch_size=batch_size,
                names=names,
            )
            return out.update(result)

        # Nested tensordicts reduce themselves, so that they get their own names
        return self._fast_apply(
            lambda x: x.logsumexp(new_dim, keepdim=keepdim),
            call_on_nested=True,
            batch_size=batch_size,
            names=names,
        )

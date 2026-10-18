# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Elementwise arithmetic, comparison and logical operations of :class:`~tensordict.TensorDictBase`.

The methods live on a mixin that ``tensordict.base`` imports before it
defines ``TensorDictBase``, so this module imports the helpers it needs from
``tensordict.base`` without needing the class itself. A method that uses
``TensorDictBase`` at run time imports it locally. The comment above the
mixin imports in ``tensordict/base.py`` gives the other rules.
"""

from __future__ import annotations

import numbers
import operator
from collections.abc import Callable
from functools import wraps
from typing import Any, TYPE_CHECKING

import tensordict.base as _td_base
import torch
from tensordict._nestedkey import NestedKey
from tensordict._tensorcollection import TensorCollection
from tensordict.base import (
    _apply_pointwise,
    _is_tensor_collection,
    _maybe_broadcast_other,
    _mixes_lazy_stacks,
    _NESTED_TENSORS_AS_LISTS,
    _write_back_inplace,
    CompatibleType,
    is_tensor_collection,
    NO_DEFAULT,
    Self,
)
from tensordict.utils import (
    _foreach,
    _is_unbatched,
    _maybe_correct_neg_dim,
    _mismatch_keys,
    _unravel_key_to_tuple,
    expand_as_right,
    is_tensorclass,
)
from torch import Tensor

if TYPE_CHECKING:
    from tensordict.base import TensorDictBase


def _values_by_key(other: TensorCollection, keys: list) -> list:
    """The tensor leaves of ``other`` under ``keys``, in that order."""
    other_keys, values = other._items_list(True, True)
    if other_keys == keys:
        # same keys in the same order, the common case: no reordering
        return values
    return other._values_list(True, True, sorting_keys=keys)


def _clamp_into_out(clamp: Callable) -> Callable:
    """Makes ``clamp(..., out=out)`` copy the result of ``clamp(...)`` into the leaves of ``out``.

    The bounds go on as they come: ``_maybe_broadcast_other`` broadcasts only
    the positional ones.
    """

    @wraps(clamp)
    def wrapper(self, *args, out=None, **kwargs):
        if out is None:
            return clamp(self, *args, **kwargs)
        if all(bound is None for bound in (*args, *kwargs.values())):
            raise ValueError(
                "clamp() with min/max=None isn't implemented with specified output."
            )
        return out.update_(clamp(self, *args, **kwargs))

    return wrapper


class _PointwiseOps:
    """Elementwise arithmetic, comparison and logical operations."""

    def __abs__(self) -> Self:
        """Returns ``abs(td)``, a new tensordict with the absolute value of every leaf.

        .. seealso:: :meth:`~.abs`
        """
        return self.abs()

    def __neg__(self) -> Self:
        """Returns ``-td``, a new tensordict with every leaf negated.

        .. seealso:: :meth:`~.neg`
        """
        return self.neg()

    def __ne__(self, other: Any) -> Self | bool:
        """Compares the tensordict with ``other`` leaf by leaf with ``!=``.

        Works as :meth:`~.__eq__`, except that an ``other`` of another type
        gives ``True``.

        Args:
            other (TensorDictBase, dict, torch.Tensor or Number): the value to
                compare with.

        Returns:
            A new tensordict of boolean leaves, or ``True``.
        """
        return self._compare(other, "__ne__", "__ne__", default=True)

    def __xor__(self, other: Any) -> Self | bool:
        """Computes ``self ^ other`` leaf by leaf and returns a new tensordict.

        The operation is a logical XOR for boolean leaves and a bitwise XOR for
        integer leaves. ``other`` is paired as in :meth:`~.__eq__`: a
        tensordict or ``dict`` must have the same keys, or a ``KeyError`` is
        raised.

        Args:
            other (TensorDictBase, dict, torch.Tensor or Number): the other
                operand.
        """
        return self._compare(other, "__xor__", "__xor__", default=True)

    def __rxor__(self, other: TensorCollection | torch.Tensor | float):
        """Computes ``other ^ self``, which is the same as ``self ^ other``."""
        return self.__xor__(other)

    def __or__(self, other: Any) -> Self | bool:
        """Computes ``self | other`` leaf by leaf and returns a new tensordict.

        The operation is a logical OR for boolean leaves and a bitwise OR for
        integer leaves. ``other`` is paired as in :meth:`~.__eq__`: a
        tensordict or ``dict`` must have the same keys, or a ``KeyError`` is
        raised.

        Args:
            other (TensorDictBase, dict, torch.Tensor or Number): the other
                operand.
        """
        return self._compare(other, "__or__", "__or__", default=False)

    def __ror__(self, other: TensorCollection | torch.Tensor) -> Self:
        """Computes ``other | self``, which is the same as ``self | other``."""
        return self | other

    def __invert__(self) -> Self:
        """Returns ``~td``, a new tensordict with ``~`` applied to every leaf.

        The operation is a logical NOT for boolean leaves and a bitwise NOT for
        integer leaves.
        """
        keys, vals = self._items_list(True, True)
        vals = [~v for v in vals]
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def __and__(self, other: TensorCollection | torch.Tensor | float) -> Self:
        """Computes ``self & other`` leaf by leaf and returns a new tensordict.

        The operation is a logical AND for boolean leaves and a bitwise AND for
        integer leaves. A tensordict ``other`` is paired with the tensordict key
        by key and must have the same keys, or a ``KeyError`` is raised.

        Args:
            other (TensorDictBase, torch.Tensor or Number): the other operand.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.tensor([1.0, 2.0, 3.0]), batch_size=[3])
            >>> ((td > 1) & (td < 3))["a"]
            tensor([False,  True, False])
            >>> (~(td > 1))["a"]
            tensor([ True, False, False])
        """
        if _mixes_lazy_stacks(self, (other,)):
            # as for | and ^: the foreach path below flattens a lazy stack
            # into keys such as ("0", "a"), which a dense operand lacks
            return self._compare(other, "__and__", "__and__", default=False)
        keys, vals = self._items_list(True, True)
        if _is_tensor_collection(type(other)):
            new_keys, other_val = other._items_list(True, True, sorting_keys=keys)
            vals = [(v1 & v2) for v1, v2 in zip(vals, other_val)]
        else:
            vals = [(v & other) for v in vals]
        items = dict(zip(keys, vals))

        def pop(name, val):
            return items.pop(name, None)

        result = self._fast_apply(
            pop,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    __rand__ = __and__

    def __eq__(self, other: Any) -> Self | bool:
        """Compares the tensordict with ``other`` leaf by leaf with ``==``.

        ``other`` can be a tensordict or a ``dict`` with the same keys, paired
        with the tensordict key by key (a ``KeyError`` is raised if the keys
        differ); a tensor, which is aligned with the batch dimensions of each
        leaf; or a number. A batch size that differs from the tensordict's is
        broadcast with it. An ``other`` of another type, such as ``None``,
        gives ``False``. The other comparisons (``!=``, ``<``, ``<=``, ``>``,
        ``>=``) work the same way.

        Args:
            other (TensorDictBase, dict, torch.Tensor or Number): the value to
                compare with.

        Returns:
            A new tensordict of boolean leaves, or ``False``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.tensor([1.0, 2.0]), b=torch.zeros(2), batch_size=[2])
            >>> (td == 1)["a"]
            tensor([ True, False])
            >>> (td == td.clone()).all()
            True
            >>> (td >= torch.tensor([2.0, 0.0]))["b"]
            tensor([False,  True])
            >>> td == None
            False
        """
        return self._compare(other, "__eq__", "__eq__", default=False)

    def __ge__(self, other: Any) -> Self | bool:
        """Compares the tensordict with ``other`` leaf by leaf with ``>=``.

        Works as :meth:`~.__eq__`.

        Args:
            other (TensorDictBase, dict, torch.Tensor or Number): the value to
                compare with.

        Returns:
            A new tensordict of boolean leaves, or ``False``.
        """
        return self._compare(other, "__ge__", "__le__", default=False)

    def __gt__(self, other: Any) -> Self | bool:
        """Compares the tensordict with ``other`` leaf by leaf with ``>``.

        Works as :meth:`~.__eq__`.

        Args:
            other (TensorDictBase, dict, torch.Tensor or Number): the value to
                compare with.

        Returns:
            A new tensordict of boolean leaves, or ``False``.
        """
        return self._compare(other, "__gt__", "__lt__", default=False)

    def __le__(self, other: Any) -> Self | bool:
        """Compares the tensordict with ``other`` leaf by leaf with ``<=``.

        Works as :meth:`~.__eq__`.

        Args:
            other (TensorDictBase, dict, torch.Tensor or Number): the value to
                compare with.

        Returns:
            A new tensordict of boolean leaves, or ``False``.
        """
        return self._compare(other, "__le__", "__ge__", default=False)

    def __lt__(self, other: Any) -> Self | bool:
        """Compares the tensordict with ``other`` leaf by leaf with ``<``.

        Works as :meth:`~.__eq__`.

        Args:
            other (TensorDictBase, dict, torch.Tensor or Number): the value to
                compare with.

        Returns:
            A new tensordict of boolean leaves, or ``False``.
        """
        return self._compare(other, "__lt__", "__gt__", default=False)

    def _compare(
        self, other: Any, op: str, reflected_op: str, default: bool
    ) -> TensorDictBase | bool:
        """Computes the comparison or logical ``op`` (``"__eq__"``, ``"__lt__"``, ``"__or__"``, ...) of ``self`` and ``other``.

        The comparison dunders call this with their ``op``, the op that gives
        the same result with the operands swapped (``reflected_op``), called
        on a tensorclass ``other``, and the value to return for an operand that
        is not a tensor, number, dict or tensor collection (``default``).

        The base broadcasts the batch shapes against a tensor or a tensor
        collection, then pairs the leaves by key into a new dense
        :class:`~tensordict.TensorDict`, and raises ``KeyError`` on mismatched
        key sets. :class:`~tensordict.LazyStackedTensorDict` compares each
        member and stacks the results lazily; it returns ``default`` only for
        ``==`` and ``!=``, and raises ``ValueError`` for the other ops.
        """
        TensorDict = _td_base._TENSORDICT_TYPE

        if (isinstance(other, Tensor) and other.ndim) or (
            _is_tensor_collection(type(other))
            and other.ndim
            and other.shape != self.shape
        ):
            # Broadcast, as _maybe_broadcast_other does for the arithmetic ops.
            # The dunders do not use it, so that a lazy stack reaches its own
            # _compare, which goes member by member, with any operand.
            shape = torch.broadcast_shapes(self.shape, other.shape)
            self_expand = self if shape == self.shape else self.expand(shape)
            if other.shape != shape:
                other = other.expand(shape)
            if isinstance(other, Tensor):
                return self_expand._fast_apply(
                    lambda x: getattr(x, op)(expand_as_right(other, x))
                )
            return getattr(self_expand, op)(other)
        compare = getattr(operator, op)
        if is_tensorclass(other):
            return getattr(operator, reflected_op)(other, self)
        if isinstance(other, (dict,)):
            other = self.from_dict_instance(other, auto_batch_size=False)
        if _is_tensor_collection(type(other)):
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                if op in ("__ne__", "__xor__", "__or__"):
                    raise KeyError(
                        f"keys in {self} and {other} mismatch, got {keys1} and {keys2}"
                    )
                _mismatch_keys(keys1, keys2)
            d = {}
            for key, item1 in self.items():
                d[key] = compare(item1, other.get(key))
            return TensorDict(source=d, batch_size=self.batch_size, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return TensorDict(
                {key: compare(value, other) for key, value in self.items()},
                self.batch_size,
                device=self.device,
            )
        return default

    def isfinite(self) -> Self:
        """Returns a new tensordict with :func:`torch.isfinite` applied to every leaf.

        The leaves of the result are ``True`` where an element is neither NaN
        nor infinite.
        """
        keys, vals = self._items_list(True, True)
        vals = [val.isfinite() for val in vals]
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def isnan(self) -> Self:
        """Returns a new tensordict with :func:`torch.isnan` applied to every leaf.

        The leaves of the result are ``True`` where an element is NaN.
        """
        keys, vals = self._items_list(True, True)
        vals = [val.isnan() for val in vals]
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def isneginf(self) -> Self:
        """Returns a new tensordict with :func:`torch.isneginf` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        vals = [val.isneginf() for val in vals]
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def isposinf(self) -> Self:
        """Returns a new tensordict with :func:`torch.isposinf` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        vals = [val.isposinf() for val in vals]
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    def isreal(self) -> Self:
        """Returns a new tensordict with :func:`torch.isreal` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        vals = [val.isreal() for val in vals]
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    # point-wise arithmetic ops
    def __add__(self, other: TensorCollection | torch.Tensor | float) -> Self:
        return self.add(other)

    def __radd__(self, other: TensorCollection | torch.Tensor | float) -> Self:
        return self.add(other)

    def __iadd__(self, other: TensorCollection | torch.Tensor | float) -> Self:
        return self.add_(other)

    def __truediv__(self, other: TensorCollection | torch.Tensor | float) -> Self:
        return self.div(other)

    def __itruediv__(self, other: TensorCollection | torch.Tensor | float) -> Self:
        return self.div_(other)

    def __rtruediv__(self, other: TensorCollection | torch.Tensor | float) -> Self:
        return self.reciprocal() * other

    def __mul__(self, other: TensorCollection | torch.Tensor | float) -> Self:
        return self.mul(other)

    def __mod__(self, other: TensorCollection | torch.Tensor | float) -> Self:
        return self.mod(other)

    def __rmul__(self, other: TensorCollection | torch.Tensor | float) -> Self:
        return self.mul(other)

    def __imul__(self, other: TensorCollection | torch.Tensor | float) -> Self:
        return self.mul_(other)

    def __sub__(self, other: TensorCollection | torch.Tensor | float) -> Self:
        return self.sub(other)

    def __isub__(self, other: TensorCollection | torch.Tensor | float) -> Self:
        return self.sub_(other)

    def __rsub__(self, other: TensorCollection | torch.Tensor | float) -> Self:
        return self.rsub(other)

    def __pow__(self, other: TensorCollection | torch.Tensor | float) -> Self:
        return self.pow(other)

    def __rpow__(self, other: TensorCollection | torch.Tensor | float) -> Self:
        raise NotImplementedError(
            "rpow isn't implemented for tensordict yet. Make sure both elements are wrapped "
            "in a tensordict for this to work."
        )

    def __ipow__(self, other: TensorCollection | torch.Tensor | float) -> Self:
        return self.pow_(other)

    def abs(self) -> Self:
        """Returns a new tensordict with :func:`torch.abs` applied to every leaf.

        The other unary operations (:meth:`~.exp`, :meth:`~.log`,
        :meth:`~.sqrt`, :meth:`~.sin`...) work the same way. Each has an
        in-place twin with a trailing underscore (:meth:`~.abs_`,
        :meth:`~.exp_`...) that writes into the leaves and returns ``self``.
        See :ref:`inplace-conventions`.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(
            ...     a=torch.tensor([-1.0, 2.0]),
            ...     b=TensorDict(c=torch.tensor([-3.0, 0.0]), batch_size=[2]),
            ...     batch_size=[2],
            ... )
            >>> td.abs()["b", "c"]
            tensor([3., 0.])
            >>> td.exp()["a"]
            tensor([0.3679, 7.3891])
            >>> td.abs_() is td
            True
            >>> td["a"]
            tensor([1., 2.])
        """
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("abs", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def abs_(self) -> Self:
        """Applies :func:`torch.abs` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("abs_", vals)
        return self

    def acos(self) -> Self:
        """Returns a new tensordict with :func:`torch.acos` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("acos", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def acos_(self) -> Self:
        """Applies :func:`torch.acos` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("acos_", vals)
        return self

    def exp(self) -> Self:
        """Returns a new tensordict with :func:`torch.exp` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("exp", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def exp_(self) -> Self:
        """Applies :func:`torch.exp` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("exp_", vals)
        return self

    def neg(self) -> Self:
        """Returns a new tensordict with :func:`torch.neg` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        # Empty tensordict (no tensors) - return a copy unchanged
        if not vals:
            return self.copy()
        if any(_is_unbatched(v) for v in vals):
            return self.apply(lambda x: -x)
        vals = _foreach("neg", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def neg_(self) -> Self:
        """Applies :func:`torch.neg` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        # Empty tensordict (no tensors) - nothing to do
        if not vals:
            return self
        if any(_is_unbatched(v) for v in vals):
            self.apply_(lambda x: x.neg_())
            return self
        _foreach("neg_", vals)
        return self

    def reciprocal(self) -> Self:
        """Returns a new tensordict with :func:`torch.reciprocal` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("reciprocal", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def reciprocal_(self) -> Self:
        """Applies :func:`torch.reciprocal` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("reciprocal_", vals)
        return self

    def sigmoid(self) -> Self:
        """Returns a new tensordict with :func:`torch.sigmoid` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("sigmoid", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def sigmoid_(self) -> Self:
        """Applies :func:`torch.sigmoid` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("sigmoid_", vals)
        return self

    def sign(self) -> Self:
        """Returns a new tensordict with :func:`torch.sign` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("sign", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def sign_(self) -> Self:
        """Applies :func:`torch.sign` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("sign_", vals)
        return self

    def sin(self) -> Self:
        """Returns a new tensordict with :func:`torch.sin` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("sin", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def sin_(self) -> Self:
        """Applies :func:`torch.sin` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("sin_", vals)
        return self

    def sinh(self) -> Self:
        """Returns a new tensordict with :func:`torch.sinh` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("sinh", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def sinh_(self) -> Self:
        """Applies :func:`torch.sinh` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("sinh_", vals)
        return self

    def tan(self) -> Self:
        """Returns a new tensordict with :func:`torch.tan` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("tan", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def tan_(self) -> Self:
        """Applies :func:`torch.tan` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("tan_", vals)
        return self

    def tanh(self) -> Self:
        """Returns a new tensordict with :func:`torch.tanh` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("tanh", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def tanh_(self) -> Self:
        """Applies :func:`torch.tanh` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("tanh_", vals)
        return self

    def trunc(self) -> Self:
        """Returns a new tensordict with :func:`torch.trunc` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("trunc", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def trunc_(self) -> Self:
        """Applies :func:`torch.trunc` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("trunc_", vals)
        return self

    def lgamma(self) -> Self:
        """Returns a new tensordict with :func:`torch.lgamma` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("lgamma", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def lgamma_(self) -> Self:
        """Applies :func:`torch.lgamma` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("lgamma_", vals)
        return self

    def frac(self) -> Self:
        """Returns a new tensordict with :func:`torch.frac` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("frac", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def frac_(self) -> Self:
        """Applies :func:`torch.frac` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("frac_", vals)
        return self

    def expm1(self) -> Self:
        """Returns a new tensordict with :func:`torch.expm1` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("expm1", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def expm1_(self) -> Self:
        """Applies :func:`torch.expm1` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("expm1_", vals)
        return self

    def log(self) -> Self:
        """Returns a new tensordict with :func:`torch.log` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("log", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def log_(self) -> Self:
        """Applies :func:`torch.log` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("log_", vals)
        return self

    def softmax(self, dim: int, dtype: torch.dtype | None = None):  # noqa: D417
        """Returns a new tensordict with :func:`torch.softmax` applied to every leaf along a batch dimension.

        Args:
            dim (int): the batch dimension along which softmax is computed. A
                negative ``dim`` counts from the end of the batch dimensions.
            dtype (torch.dtype, optional): if given, each leaf is cast to this
                dtype before the operation, which helps to prevent overflows.
                Defaults to ``None``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(x=torch.zeros(2, 3), batch_size=[2])
            >>> td.softmax(0)["x"]
            tensor([[0.5000, 0.5000, 0.5000],
                    [0.5000, 0.5000, 0.5000]])
        """
        if isinstance(dim, int):
            dim = _maybe_correct_neg_dim(dim, self.batch_size)
        else:
            raise ValueError(f"Expected dim of type int, got {type(dim)}.")
        return self._fast_apply(
            lambda x: torch.softmax(x, dim=dim, dtype=dtype),
        )

    def log10(self) -> Self:
        """Returns a new tensordict with :func:`torch.log10` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("log10", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def log10_(self) -> Self:
        """Applies :func:`torch.log10` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("log10_", vals)
        return self

    def log1p(self) -> Self:
        """Returns a new tensordict with :func:`torch.log1p` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("log1p", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def log1p_(self) -> Self:
        """Applies :func:`torch.log1p` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("log1p_", vals)
        return self

    def log2(self) -> Self:
        """Returns a new tensordict with :func:`torch.log2` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("log2", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def log2_(self) -> Self:
        """Applies :func:`torch.log2` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("log2_", vals)
        return self

    def ceil(self) -> Self:
        """Returns a new tensordict with :func:`torch.ceil` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("ceil", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def ceil_(self) -> Self:
        """Applies :func:`torch.ceil` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("ceil_", vals)
        return self

    def floor(self) -> Self:
        """Returns a new tensordict with :func:`torch.floor` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("floor", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def floor_(self) -> Self:
        """Applies :func:`torch.floor` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("floor_", vals)
        return self

    def round(self) -> Self:
        """Returns a new tensordict with :func:`torch.round` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("round", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def round_(self) -> Self:
        """Applies :func:`torch.round` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("round_", vals)
        return self

    def erf(self) -> Self:
        """Returns a new tensordict with :func:`torch.erf` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("erf", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def erf_(self) -> Self:
        """Applies :func:`torch.erf` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("erf_", vals)
        return self

    def erfc(self) -> Self:
        """Returns a new tensordict with :func:`torch.erfc` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("erfc", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def erfc_(self) -> Self:
        """Applies :func:`torch.erfc` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("erfc_", vals)
        return self

    def asin(self) -> Self:
        """Returns a new tensordict with :func:`torch.asin` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("asin", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def asin_(self) -> Self:
        """Applies :func:`torch.asin` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("asin_", vals)
        return self

    def atan(self) -> Self:
        """Returns a new tensordict with :func:`torch.atan` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("atan", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def atan_(self) -> Self:
        """Applies :func:`torch.atan` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("atan_", vals)
        return self

    def cos(self) -> Self:
        """Returns a new tensordict with :func:`torch.cos` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("cos", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def cos_(self) -> Self:
        """Applies :func:`torch.cos` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("cos_", vals)
        return self

    def cosh(self) -> Self:
        """Returns a new tensordict with :func:`torch.cosh` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("cosh", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def cosh_(self) -> Self:
        """Applies :func:`torch.cosh` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("cosh_", vals)
        return self

    @_maybe_broadcast_other("bitwise_and")
    def bitwise_and(
        self,
        other: TensorCollection | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> Self:  # noqa: D417
        r"""Returns a new tensordict with the bitwise AND of every leaf and ``other``.

        .. math::
            \text{out}_i = \text{input}_i \land \text{other}_i

        Args:
            other (TensorDictBase, torch.Tensor or Number): the other operand. A
                tensordict is paired with the tensordict key by key. A tensor is
                aligned with the batch dimensions of each leaf and expanded over
                its feature dimensions. Batch sizes that differ are broadcast
                together.

        Keyword Args:
            default (torch.Tensor or str, optional): what to do with the keys
                that only one of the two tensordicts has. ``None`` requires the
                same keys and raises a ``KeyError`` otherwise.
                ``"intersection"`` computes the keys that both have and drops
                the others. A tensor stands in for the missing entry on either
                side, and the result has the keys of both. Defaults to ``None``.
        """
        keys, vals = self._items_list(True, True)
        if _is_tensor_collection(type(other)):
            new_keys, other_val = other._items_list(
                True, True, sorting_keys=keys, default=default
            )
            if default is not None:
                as_dict = dict(zip(keys, vals))
                vals = [as_dict.get(key, default) for key in new_keys]
                keys = new_keys
            vals = [(v1.bitwise_and(v2)) for v1, v2 in zip(vals, other_val)]
        else:
            vals = [v.bitwise_and(other) for v in vals]
        items = dict(zip(keys, vals))

        def pop(name, val):
            return items.pop(name, None)

        result = self._fast_apply(
            pop,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    @_maybe_broadcast_other("logical_and")
    def logical_and(
        self,
        other: TensorCollection | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> Self:  # noqa: D417
        r"""Returns a new tensordict with the logical AND of every leaf and ``other``.

        Nonzero elements count as ``True``, and the leaves of the result are
        boolean.

        .. math::
            \text{out}_i = \text{input}_i \land \text{other}_i

        Args:
            other (TensorDictBase or torch.Tensor): the other operand. A
                tensordict is paired with the tensordict key by key. A tensor is
                aligned with the batch dimensions of each leaf and expanded over
                its feature dimensions. Batch sizes that differ are broadcast
                together.

        Keyword Args:
            default (torch.Tensor or str, optional): what to do with the keys
                that only one of the two tensordicts has. ``None`` requires the
                same keys and raises a ``KeyError`` otherwise.
                ``"intersection"`` computes the keys that both have and drops
                the others. A tensor stands in for the missing entry on either
                side, and the result has the keys of both. Defaults to ``None``.
        """
        keys, vals = self._items_list(True, True)
        if _is_tensor_collection(type(other)):
            new_keys, other_val = other._items_list(
                True, True, sorting_keys=keys, default=default
            )
            if default is not None:
                as_dict = dict(zip(keys, vals))
                vals = [as_dict.get(key, default) for key in new_keys]
                keys = new_keys
            vals = [(v1.logical_and(v2)) for v1, v2 in zip(vals, other_val)]
        else:
            vals = [v.logical_and(other) for v in vals]
        items = dict(zip(keys, vals))

        def pop(name, val):
            return items.pop(name, None)

        result = self._fast_apply(
            pop,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    @_maybe_broadcast_other("add")
    def add(
        self,
        other: TensorCollection | torch.Tensor,
        *,
        alpha: float | None = None,
        default: str | CompatibleType | None = None,
    ) -> Self:  # noqa: D417
        r"""Returns a new tensordict with ``other``, scaled by ``alpha``, added to every leaf.

        ``td + other`` and ``other + td`` call this method.

        .. math::
            \text{out}_i = \text{input}_i + \text{alpha} \times \text{other}_i

        Args:
            other (TensorDictBase, torch.Tensor or Number): the value to add. A
                tensordict is paired with the tensordict key by key. A tensor is
                aligned with the batch dimensions of each leaf and expanded over
                its feature dimensions. Batch sizes that differ are broadcast
                together.

        Keyword Args:
            alpha (Number, optional): the multiplier for ``other``. Defaults to
                ``None``, which adds ``other`` as it is.
            default (torch.Tensor or str, optional): what to do with the keys
                that only one of the two tensordicts has. ``None`` requires the
                same keys and raises a ``KeyError`` otherwise.
                ``"intersection"`` computes the keys that both have and drops
                the others. A tensor stands in for the missing entry on either
                side, and the result has the keys of both. Defaults to ``None``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(2, 3), b=torch.ones(2), batch_size=[2])
            >>> td.add(1, alpha=2)["b"]
            tensor([3., 3.])
            >>> td.add(torch.tensor([1.0, 2.0]))["a"]  # one value per batch element
            tensor([[1., 1., 1.],
                    [2., 2., 2.]])
            >>> other = TensorDict(b=torch.ones(2), c=torch.ones(2), batch_size=[2])
            >>> sorted(td.add(other, default="intersection").keys())
            ['b']
            >>> sorted(td.add(other, default=torch.zeros(())).keys())
            ['a', 'b', 'c']
        """
        keys, vals = self._items_list(True, True)
        # Empty tensordict (no tensors) - return a copy unchanged
        if not vals:
            return self.copy()
        if any(_is_unbatched(v) for v in vals):
            if _is_tensor_collection(type(other)):
                if alpha is not None:
                    return self.apply(lambda x, y: x.add(y, alpha=alpha), other)
                return self.apply(lambda x, y: x + y, other)
            else:
                if alpha is not None:
                    return self.apply(lambda x: x.add(other, alpha=alpha))
                return self.apply(lambda x: x + other)
        if _is_tensor_collection(type(other)):
            new_keys, other_val = other._items_list(
                True, True, sorting_keys=keys, default=default
            )
            if default is not None:
                as_dict = dict(zip(keys, vals))
                vals = [as_dict.get(key, default) for key in new_keys]
                keys = new_keys
        else:
            other_val = other
        if alpha is not None and isinstance(other_val, numbers.Number):
            # The _foreach overloads that take a number have no alpha
            other_val, alpha = other_val * alpha, None
        if alpha is not None:
            vals = _foreach("add", vals, other_val, alpha=alpha)
        else:
            vals = _foreach("add", vals, other_val)
        items = dict(zip(keys, vals))

        def pop(name, val):
            return items.pop(name, None)

        result = self._fast_apply(
            pop,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    @_write_back_inplace
    @_maybe_broadcast_other("add_")
    def add_(
        self,
        other: TensorCollection | torch.Tensor | float,
        *,
        alpha: float | None = None,
    ) -> Self:
        """Adds ``other``, scaled by ``alpha``, to every leaf in place and returns ``self``.

        The in-place version of :meth:`~.add`, called by ``td += other``. It
        has no ``default``: a tensordict ``other`` must have every key of the
        tensordict, and its other keys are ignored.

        Args:
            other (TensorDictBase, torch.Tensor or Number): the value to add,
                paired with the leaves as in :meth:`~.add`.

        Keyword Args:
            alpha (Number, optional): the multiplier for ``other``. Defaults to
                ``None``, which adds ``other`` as it is.
        """
        if _is_tensor_collection(type(other)):
            keys, vals = self._items_list(True, True)
            other_val = other._values_list(True, True, sorting_keys=keys)
        else:
            vals = self._values_list(True, True)
            other_val = other
        # Empty tensordict (no tensors) - nothing to do
        if not vals:
            return self
        if any(_is_unbatched(v) for v in vals):
            if _is_tensor_collection(type(other)):
                if alpha is not None:
                    self.apply_(lambda x, y: x.add_(y, alpha=alpha), other)
                else:
                    self.apply_(lambda x, y: x.add_(y), other)
            else:
                if alpha is not None:
                    self.apply_(lambda x: x.add_(other, alpha=alpha))
                else:
                    self.apply_(lambda x: x.add_(other))
            return self
        if alpha is not None and isinstance(other_val, numbers.Number):
            # The _foreach overloads that take a number have no alpha
            other_val, alpha = other_val * alpha, None
        if alpha is not None:
            _foreach("add_", vals, other_val, alpha=alpha)
        else:
            _foreach("add_", vals, other_val)
        return self

    @_maybe_broadcast_other("lerp", 2)
    def lerp(
        self,
        end: TensorCollection | torch.Tensor,
        weight: TensorCollection | torch.Tensor | float,
    ) -> Self:
        r"""Returns a new tensordict that interpolates linearly from every leaf (the start) to ``end``.

        .. math::
            \text{out}_i = \text{start}_i + \text{weight}_i \times (\text{end}_i - \text{start}_i)

        Args:
            end (TensorDictBase or torch.Tensor): the end points. A tensordict
                is paired with the tensordict key by key and must have all its
                keys. A tensor is aligned with the batch dimensions of each
                leaf and expanded over its feature dimensions.
            weight (TensorDictBase, torch.Tensor or float): the interpolation
                weight: a number for all the leaves, or a tensordict or a
                tensor paired like ``end``. One of ``end`` and ``weight`` can
                be a tensordict and the other a tensor.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> start = TensorDict(x=torch.zeros(2), batch_size=[2])
            >>> end = TensorDict(x=torch.full((2,), 10.0), batch_size=[2])
            >>> start.lerp(end, 0.25)["x"]
            tensor([2.5000, 2.5000])
        """
        if (isinstance(end, Tensor) and end.ndim) or (
            isinstance(weight, Tensor) and weight.ndim
        ):
            # passed by keyword: _maybe_broadcast_other only reads positional
            # arguments, and broadcasts them to the batch dims
            return self.lerp(end, weight)
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        if _is_tensor_collection(type(end)):
            end_val = _values_by_key(end, keys)
        else:
            # _foreach_lerp takes a list of ends
            end_val = [end] * len(vals)
        if isinstance(weight, (float, torch.Tensor)):
            weight_val = weight
        elif _is_tensor_collection(type(weight)):
            weight_val = _values_by_key(weight, keys)
        else:
            weight_val = weight
        vals = _foreach("lerp", vals, end_val, weight_val)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    def lerp_(
        self,
        end: TensorDictBase | torch.Tensor,
        weight: TensorDictBase | torch.Tensor | float,
    ):
        """Interpolates linearly from every leaf to ``end`` in place and returns ``self``.

        The in-place version of :meth:`~.lerp`.

        Args:
            end (TensorDictBase or torch.Tensor): the end points. A tensordict
                is paired with the tensordict key by key and must have all its
                keys. A tensor is aligned with the batch dimensions of each
                leaf and expanded over its feature dimensions.
            weight (TensorDictBase, torch.Tensor or float): the interpolation
                weight: a number for all the leaves, or a tensordict or a
                tensor paired like ``end``. One of ``end`` and ``weight`` can
                be a tensordict and the other a tensor.
        """
        if (isinstance(end, Tensor) and end.ndim) or (
            isinstance(weight, Tensor) and weight.ndim
        ):
            # broadcast the tensor to the batch dims, as lerp does. The other
            # calls skip _maybe_broadcast_other, which costs a few us per call.
            lerp_ = _maybe_broadcast_other("lerp_", 2)(type(self).lerp_)
            return lerp_(self, end, weight)
        if _mixes_lazy_stacks(self, (end, weight)):
            return _apply_pointwise(self, Tensor.lerp_, (end, weight), inplace=True)
        # the leaves of the tensordict operands are paired by key, as in add_
        keys, vals = self._items_list(True, True)
        if not vals:
            return self
        if _is_tensor_collection(type(end)):
            end_val = _values_by_key(end, keys)
        else:
            # _foreach_lerp_ takes a list of ends
            end_val = [end] * len(vals)
        if isinstance(weight, (float, torch.Tensor)):
            weight_val = weight
        elif _is_tensor_collection(type(weight)):
            weight_val = _values_by_key(weight, keys)
        else:
            weight_val = weight
        _foreach("lerp_", vals, end_val, weight_val)
        return self

    @_maybe_broadcast_other("addcdiv", 2)
    def addcdiv(
        self,
        other1: TensorDictBase | torch.Tensor,
        other2: TensorDictBase | torch.Tensor,
        value: float | None = 1,
    ) -> Self:  # noqa: D417
        r"""Returns a new tensordict with ``value * other1 / other2`` added to every leaf.

        .. math::
            \text{out}_i = \text{input}_i + \text{value} \times \frac{\text{other1}_i}{\text{other2}_i}

        Args:
            other1 (TensorDictBase or torch.Tensor): the numerator. A tensordict
                is paired with the tensordict key by key and must have all its
                keys. A tensor is aligned with the batch dimensions of each leaf
                and expanded over its feature dimensions.
            other2 (TensorDictBase or torch.Tensor): the denominator, paired
                like ``other1``. One of ``other1`` and ``other2`` can be a
                tensordict and the other a tensor.
            value (Number, optional): the multiplier for
                :math:`\text{other1} / \text{other2}`. Defaults to ``1``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(x=torch.ones(2), batch_size=[2])
            >>> td.addcdiv(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 2.0]), value=2)["x"]
            tensor([2., 3.])
        """
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        if _is_tensor_collection(type(other1)):
            other1_val = _values_by_key(other1, keys)
        else:
            other1_val = other1
        if _is_tensor_collection(type(other2)):
            other2_val = _values_by_key(other2, keys)
        else:
            other2_val = other2
        vals = _foreach("addcdiv", vals, other1_val, other2_val, value=value)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    @_maybe_broadcast_other("addcdiv_", 2)
    def addcdiv_(self, other1, other2, *, value: float | None = 1):
        r"""Adds ``value * other1 / other2`` to every leaf in place and returns ``self``.

        The in-place version of :meth:`~.addcdiv`.

        Args:
            other1 (TensorDictBase or torch.Tensor): the numerator, paired with
                the leaves as in :meth:`~.addcdiv`.
            other2 (TensorDictBase or torch.Tensor): the denominator, paired
                like ``other1``.

        Keyword Args:
            value (Number, optional): the multiplier for
                :math:`\text{other1} / \text{other2}`. Defaults to ``1``.
        """
        # the leaves of the tensordict operands are paired by key, as in add_
        keys, vals = self._items_list(True, True)
        if not vals:
            return self
        if _is_tensor_collection(type(other1)):
            other1_val = _values_by_key(other1, keys)
        else:
            other1_val = other1
        if _is_tensor_collection(type(other2)):
            other2_val = _values_by_key(other2, keys)
        else:
            other2_val = other2
        _foreach(
            "addcdiv_",
            vals,
            other1_val,
            other2_val,
            value=value,
        )
        return self

    @_maybe_broadcast_other("addcmul", 2)
    def addcmul(
        self,
        other1: TensorDictBase | torch.Tensor,
        other2: TensorDictBase | torch.Tensor,
        *,
        value: float | None = 1,
    ) -> Self:  # noqa: D417
        r"""Returns a new tensordict with ``value * other1 * other2`` added to every leaf.

        .. math::
            \text{out}_i = \text{input}_i + \text{value} \times \text{other1}_i \times \text{other2}_i

        Args:
            other1 (TensorDictBase or torch.Tensor): the first factor. A
                tensordict is paired with the tensordict key by key and must
                have all its keys. A tensor is aligned with the batch dimensions
                of each leaf and expanded over its feature dimensions.
            other2 (TensorDictBase or torch.Tensor): the second factor, paired
                like ``other1``. One of ``other1`` and ``other2`` can be a
                tensordict and the other a tensor.

        Keyword Args:
            value (Number, optional): the multiplier for
                :math:`\text{other1} \times \text{other2}`. Defaults to ``1``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(x=torch.ones(2), batch_size=[2])
            >>> td.addcmul(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 3.0]), value=2)["x"]
            tensor([ 7., 13.])
        """
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        if _is_tensor_collection(type(other1)):
            other1_val = _values_by_key(other1, keys)
        else:
            other1_val = other1
        if _is_tensor_collection(type(other2)):
            other2_val = _values_by_key(other2, keys)
        else:
            other2_val = other2
        vals = _foreach("addcmul", vals, other1_val, other2_val, value=value)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    @_write_back_inplace
    @_maybe_broadcast_other("addcmul_", 2)
    def addcmul_(self, other1, other2, *, value: float | None = 1):
        r"""Adds ``value * other1 * other2`` to every leaf in place and returns ``self``.

        The in-place version of :meth:`~.addcmul`.

        Args:
            other1 (TensorDictBase or torch.Tensor): the first factor, paired
                with the leaves as in :meth:`~.addcmul`.
            other2 (TensorDictBase or torch.Tensor): the second factor, paired
                like ``other1``.

        Keyword Args:
            value (Number, optional): the multiplier for
                :math:`\text{other1} \times \text{other2}`. Defaults to ``1``.
        """
        # the leaves of the tensordict operands are paired by key, as in add_
        keys, vals = self._items_list(True, True)
        if not vals:
            return self
        if _is_tensor_collection(type(other1)):
            other1_val = _values_by_key(other1, keys)
        else:
            other1_val = other1
        if _is_tensor_collection(type(other2)):
            other2_val = _values_by_key(other2, keys)
        else:
            other2_val = other2
        _foreach(
            "addcmul_",
            vals,
            other1_val,
            other2_val,
            value=value,
        )
        return self

    @_maybe_broadcast_other("sub")
    def sub(
        self,
        other: TensorDictBase | torch.Tensor | float,
        *,
        alpha: float | None = None,
        default: str | CompatibleType | None = None,
    ) -> Self:  # noqa: D417
        r"""Returns a new tensordict with ``other``, scaled by ``alpha``, subtracted from every leaf.

        ``td - other`` calls this method, and ``other - td`` calls
        :meth:`~.rsub`.

        .. math::
            \text{out}_i = \text{input}_i - \text{alpha} \times \text{other}_i

        Args:
            other (TensorDictBase, torch.Tensor or Number): the value to
                subtract. A tensordict is paired with the tensordict key by key.
                A tensor is aligned with the batch dimensions of each leaf and
                expanded over its feature dimensions. Batch sizes that differ
                are broadcast together.

        Keyword Args:
            alpha (Number, optional): the multiplier for ``other``. Defaults to
                ``None``, which subtracts ``other`` as it is.
            default (torch.Tensor or str, optional): what to do with the keys
                that only one of the two tensordicts has. ``None`` requires the
                same keys and raises a ``KeyError`` otherwise.
                ``"intersection"`` computes the keys that both have and drops
                the others. A tensor stands in for the missing entry on either
                side, and the result has the keys of both. Defaults to ``None``.

        """
        keys, vals = self._items_list(True, True)
        no_tensors = not vals
        if _is_tensor_collection(type(other)):
            new_keys, other_val = other._items_list(
                True, True, sorting_keys=keys, default=default
            )
            if default is not None:
                as_dict = dict(zip(keys, vals))
                vals = [as_dict.get(key, default) for key in new_keys]
                keys = new_keys
        else:
            other_val = other
        if no_tensors and not vals:
            # No tensor in self, and default fills none in from other.
            return self.copy()
        if alpha is not None and isinstance(other_val, numbers.Number):
            # The _foreach overloads that take a number have no alpha
            other_val, alpha = other_val * alpha, None
        if alpha is not None:
            vals = _foreach("sub", vals, other_val, alpha=alpha)
        else:
            vals = _foreach("sub", vals, other_val)
        items = dict(zip(keys, vals))

        def pop(name, val):
            return items.pop(name, None)

        result = self._fast_apply(
            pop,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    @_write_back_inplace
    @_maybe_broadcast_other("sub_")
    def sub_(
        self, other: TensorDictBase | torch.Tensor | float, alpha: float | None = None
    ):
        """Subtracts ``other``, scaled by ``alpha``, from every leaf in place and returns ``self``.

        The in-place version of :meth:`~.sub`, called by ``td -= other``. It
        has no ``default``: a tensordict ``other`` must have every key of the
        tensordict, and its other keys are ignored.

        Args:
            other (TensorDictBase, torch.Tensor or Number): the value to
                subtract, paired with the leaves as in :meth:`~.sub`.
            alpha (Number, optional): the multiplier for ``other``. Defaults to
                ``None``, which subtracts ``other`` as it is.
        """
        if _is_tensor_collection(type(other)):
            keys, vals = self._items_list(True, True)
            other_val = other._values_list(True, True, sorting_keys=keys)
        else:
            vals = self._values_list(True, True)
            other_val = other
        if not vals:
            return self
        if alpha is not None and isinstance(other_val, numbers.Number):
            # The _foreach overloads that take a number have no alpha
            other_val, alpha = other_val * alpha, None
        if alpha is not None:
            _foreach("sub_", vals, other_val, alpha=alpha)
        else:
            _foreach("sub_", vals, other_val)
        return self

    @_maybe_broadcast_other("rsub")
    def rsub(
        self,
        other: TensorDictBase | torch.Tensor | float,
        *,
        alpha: float | None = None,
        default: str | CompatibleType | None = None,
    ) -> Self:  # noqa: D417
        r"""Returns a new tensordict with every leaf, scaled by ``alpha``, subtracted from ``other``.

        ``other - td`` calls this method.

        .. math::
            \text{out}_i = \text{other}_i - \text{alpha} \times \text{self}_i

        Args:
            other (TensorDictBase, torch.Tensor or Number): the value from which
                the leaves are subtracted. A tensordict is paired with the
                tensordict key by key. A tensor is aligned with the batch
                dimensions of each leaf and expanded over its feature
                dimensions. Batch sizes that differ are broadcast together.

        Keyword Args:
            alpha (Number, optional): the multiplier for the leaves. Defaults to
                ``None``, which subtracts them as they are.
            default (torch.Tensor or str, optional): what to do with the keys
                that only one of the two tensordicts has. ``None`` requires the
                same keys and raises a ``KeyError`` otherwise.
                ``"intersection"`` computes the keys that both have and drops
                the others. A tensor stands in for the missing entry on either
                side, and the result has the keys of both. Defaults to ``None``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.full((2,), 10.0), batch_size=[2])
            >>> other = TensorDict(a=torch.ones(2), batch_size=[2])
            >>> td.rsub(other)["a"]
            tensor([-9., -9.])
            >>> td.rsub(other, alpha=2)["a"]  # 1 - 2 * 10
            tensor([-19., -19.])
            >>> td.rsub(torch.tensor([1.0, 2.0]))["a"]
            tensor([-9., -8.])

        """
        keys, vals = self._items_list(True, True)
        no_tensors = not vals
        if _is_tensor_collection(type(other)):
            new_keys, other_val = other._items_list(
                True, True, sorting_keys=keys, default=default
            )
            if default is not None:
                as_dict = dict(zip(keys, vals))
                vals = [as_dict.get(key, default) for key in new_keys]
                keys = new_keys
        else:
            other_val = other
        if no_tensors and not vals:
            # No tensor in self, and default fills none in from other.
            return self.copy()
        if alpha is not None:
            if _is_tensor_collection(type(other)):
                # torch.rsub(input, other, alpha=a) is torch.sub(other, input, alpha=a)
                vals = _foreach("sub", other_val, vals, alpha=alpha)
            else:
                # No _foreach op computes number - alpha * tensor in one kernel
                vals = _foreach("add", _foreach("mul", vals, -alpha), other_val)
        else:
            vals = _foreach("neg", _foreach("sub", vals, other_val))
        items = dict(zip(keys, vals))

        def pop(name, val):
            return items.pop(name, None)

        result = self._fast_apply(
            pop,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    @_maybe_broadcast_other("__mod__")
    def mod(self, other: TensorCollection | torch.Tensor) -> Self:
        """Returns a new tensordict with the remainder of every leaf divided by ``other``.

        ``td % other`` calls this method. The remainder has the sign of the
        divisor, as with the ``%`` operator of tensors.

        Args:
            other (TensorDictBase, torch.Tensor or Number): the divisor. A
                tensordict is paired with the tensordict key by key and must
                have the same keys. A tensor is aligned with the batch
                dimensions of each leaf and expanded over its feature
                dimensions.
        """
        keys, vals = self._items_list(True, True)
        if _is_tensor_collection(type(other)):
            new_keys, other_val = other._items_list(True, True, sorting_keys=keys)
        else:
            other_val = other
        if isinstance(other_val, list):
            vals = [val % other_val for val, other_val in zip(vals, other_val)]
        else:
            vals = [val % other_val for val in vals]
        items = dict(zip(keys, vals))

        def pop(name, val):
            return items.pop(name, None)

        result = self._fast_apply(
            pop,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    @_write_back_inplace
    @_maybe_broadcast_other("mul_")
    def mul_(self, other: TensorCollection | torch.Tensor) -> Self:
        """Multiplies every leaf by ``other`` in place and returns ``self``.

        The in-place version of :meth:`~.mul`, called by ``td *= other``. It
        has no ``default``: a tensordict ``other`` must have every key of the
        tensordict, and its other keys are ignored.

        Args:
            other (TensorDictBase, torch.Tensor or Number): the multiplier,
                paired with the leaves as in :meth:`~.mul`.
        """
        if _is_tensor_collection(type(other)):
            keys, vals = self._items_list(True, True)
            other_val = other._values_list(True, True, sorting_keys=keys)
        else:
            vals = self._values_list(True, True)
            other_val = other
        # Empty tensordict (no tensors) - nothing to do
        if not vals:
            return self
        if any(_is_unbatched(v) for v in vals):
            if _is_tensor_collection(type(other)):
                self.apply_(lambda x, y: x.mul_(y), other)
            else:
                self.apply_(lambda x: x.mul_(other))
            return self
        _foreach("mul_", vals, other_val)
        return self

    @_maybe_broadcast_other("mul")
    def mul(
        self,
        other: TensorCollection | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> Self:  # noqa: D417
        r"""Returns a new tensordict with every leaf multiplied by ``other``.

        ``td * other`` and ``other * td`` call this method.

        .. math::
            \text{out}_i = \text{input}_i \times \text{other}_i

        Args:
            other (TensorDictBase, torch.Tensor or Number): the multiplier. A
                tensordict is paired with the tensordict key by key. A tensor is
                aligned with the batch dimensions of each leaf and expanded over
                its feature dimensions. Batch sizes that differ are broadcast
                together.

        Keyword Args:
            default (torch.Tensor or str, optional): what to do with the keys
                that only one of the two tensordicts has. ``None`` requires the
                same keys and raises a ``KeyError`` otherwise.
                ``"intersection"`` computes the keys that both have and drops
                the others. A tensor stands in for the missing entry on either
                side, and the result has the keys of both. Defaults to ``None``.

        """
        keys, vals = self._items_list(True, True)
        # Empty tensordict (no tensors) - return a copy unchanged
        if not vals:
            return self.copy()
        if any(_is_unbatched(v) for v in vals):
            if _is_tensor_collection(type(other)):
                return self.apply(lambda x, y: x * y, other)
            else:
                return self.apply(lambda x: x * other)
        if _is_tensor_collection(type(other)):
            new_keys, other_val = other._items_list(
                True, True, sorting_keys=keys, default=default
            )
            if default is not None:
                as_dict = dict(zip(keys, vals))
                vals = [as_dict.get(key, default) for key in new_keys]
                keys = new_keys
        else:
            other_val = other
        vals = _foreach("mul", vals, other_val)
        items = dict(zip(keys, vals))

        def pop(name, val):
            return items.pop(name, None)

        result = self._fast_apply(
            pop,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    @_write_back_inplace
    def maximum_(self, other: TensorCollection | torch.Tensor) -> Self:
        """Replaces every leaf with its element-wise maximum with ``other`` in place and returns ``self``.

        The in-place version of :meth:`~.maximum`. It has no ``default``: a
        tensordict ``other`` must have every key of the tensordict, and its
        other keys are ignored.

        Args:
            other (TensorDictBase, torch.Tensor or Number): the other operand,
                paired with the leaves as in :meth:`~.maximum`. A tensor is
                aligned with the batch dimensions of each leaf, except when
                a leaf is an :class:`~tensordict.UnbatchedTensor`: then every
                leaf broadcasts with it from the right, as in torch.
        """
        if _mixes_lazy_stacks(self, (other,)):
            return _apply_pointwise(
                self, lambda x, y: torch.maximum(x, y, out=x), (other,), inplace=True
            )
        if _is_tensor_collection(type(other)):
            keys, vals = self._items_list(True, True)
            other_val = other._values_list(True, True, sorting_keys=keys)
        else:
            vals = self._values_list(True, True)
            other_val = other
        # Empty tensordict (no tensors) - nothing to do
        if not vals:
            return self
        if any(_is_unbatched(v) for v in vals):
            if _is_tensor_collection(type(other)):
                self.apply_(lambda x, y: x.clamp_min_(y), other)
            else:
                self.apply_(lambda x: x.clamp_min_(other))
            return self
        if isinstance(other, Tensor) and other.ndim:
            # broadcast the tensor to the batch dims, as maximum does
            clamp_min_ = _maybe_broadcast_other("clamp_min_")(type(self).clamp_min_)
            return clamp_min_(self, other)
        _foreach("maximum_", vals, other_val)
        return self

    @_maybe_broadcast_other("maximum")
    def maximum(
        self,
        other: TensorCollection | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> Self:  # noqa: D417
        """Returns a new tensordict with the element-wise maximum of every leaf and ``other``.

        Args:
            other (TensorDictBase, torch.Tensor or Number): the other operand. A
                tensordict is paired with the tensordict key by key. A tensor is
                aligned with the batch dimensions of each leaf and expanded over
                its feature dimensions. Batch sizes that differ are broadcast
                together.

        Keyword Args:
            default (torch.Tensor or str, optional): what to do with the keys
                that only one of the two tensordicts has. ``None`` requires the
                same keys and raises a ``KeyError`` otherwise.
                ``"intersection"`` computes the keys that both have and drops
                the others. A tensor stands in for the missing entry on either
                side, and the result has the keys of both. Defaults to ``None``.

        """
        keys, vals = self._items_list(True, True)
        # Empty tensordict (no tensors) - return a copy unchanged
        if not vals:
            return self.copy()
        if any(_is_unbatched(v) for v in vals):
            if _is_tensor_collection(type(other)):
                return self.apply(lambda x, y: torch.clamp_min(x, y), other)
            else:
                return self.apply(lambda x: torch.clamp_min(x, other))
        if _is_tensor_collection(type(other)):
            new_keys, other_val = other._items_list(
                True, True, sorting_keys=keys, default=default
            )
            if default is not None:
                as_dict = dict(zip(keys, vals))
                vals = [as_dict.get(key, default) for key in new_keys]
                keys = new_keys
        else:
            other_val = other
        vals = _foreach("maximum", vals, other_val)
        items = dict(zip(keys, vals))

        def pop(name, val):
            return items.pop(name, None)

        result = self._fast_apply(
            pop,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    @_write_back_inplace
    def minimum_(self, other: TensorCollection | torch.Tensor) -> Self:
        """Replaces every leaf with its element-wise minimum with ``other`` in place and returns ``self``.

        The in-place version of :meth:`~.minimum`. It has no ``default``: a
        tensordict ``other`` must have every key of the tensordict, and its
        other keys are ignored.

        Args:
            other (TensorDictBase, torch.Tensor or Number): the other operand,
                paired with the leaves as in :meth:`~.minimum`. A tensor is
                aligned with the batch dimensions of each leaf, except when
                a leaf is an :class:`~tensordict.UnbatchedTensor`: then every
                leaf broadcasts with it from the right, as in torch.
        """
        if _mixes_lazy_stacks(self, (other,)):
            return _apply_pointwise(
                self, lambda x, y: torch.minimum(x, y, out=x), (other,), inplace=True
            )
        if _is_tensor_collection(type(other)):
            keys, vals = self._items_list(True, True)
            other_val = other._values_list(True, True, sorting_keys=keys)
        else:
            vals = self._values_list(True, True)
            other_val = other
        # Empty tensordict (no tensors) - nothing to do
        if not vals:
            return self
        if any(_is_unbatched(v) for v in vals):
            if _is_tensor_collection(type(other)):
                self.apply_(lambda x, y: x.clamp_max_(y), other)
            else:
                self.apply_(lambda x: x.clamp_max_(other))
            return self
        if isinstance(other, Tensor) and other.ndim:
            # broadcast the tensor to the batch dims, as minimum does
            clamp_max_ = _maybe_broadcast_other("clamp_max_")(type(self).clamp_max_)
            return clamp_max_(self, other)
        _foreach("minimum_", vals, other_val)
        return self

    @_maybe_broadcast_other("minimum")
    def minimum(
        self,
        other: TensorCollection | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> Self:  # noqa: D417
        """Returns a new tensordict with the element-wise minimum of every leaf and ``other``.

        Args:
            other (TensorDictBase, torch.Tensor or Number): the other operand. A
                tensordict is paired with the tensordict key by key. A tensor is
                aligned with the batch dimensions of each leaf and expanded over
                its feature dimensions. Batch sizes that differ are broadcast
                together.

        Keyword Args:
            default (torch.Tensor or str, optional): what to do with the keys
                that only one of the two tensordicts has. ``None`` requires the
                same keys and raises a ``KeyError`` otherwise.
                ``"intersection"`` computes the keys that both have and drops
                the others. A tensor stands in for the missing entry on either
                side, and the result has the keys of both. Defaults to ``None``.

        """
        keys, vals = self._items_list(True, True)
        # Empty tensordict (no tensors) - return a copy unchanged
        if not vals:
            return self.copy()
        if any(_is_unbatched(v) for v in vals):
            if _is_tensor_collection(type(other)):
                return self.apply(lambda x, y: torch.clamp_max(x, y), other)
            else:
                return self.apply(lambda x: torch.clamp_max(x, other))
        if _is_tensor_collection(type(other)):
            new_keys, other_val = other._items_list(
                True, True, sorting_keys=keys, default=default
            )
            if default is not None:
                as_dict = dict(zip(keys, vals))
                vals = [as_dict.get(key, default) for key in new_keys]
                keys = new_keys
        else:
            other_val = other
        vals = _foreach("minimum", vals, other_val)
        items = dict(zip(keys, vals))

        def pop(name, val):
            return items.pop(name, None)

        result = self._fast_apply(
            pop,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    @_write_back_inplace
    def clamp_max_(self, other: TensorCollection | torch.Tensor) -> Self:
        """Clamps every leaf to at most ``other`` in place and returns ``self``.

        The in-place version of :meth:`~.clamp_max`. It has no ``default``: a
        tensordict ``other`` must have every key of the tensordict, and its
        other keys are ignored.

        Args:
            other (TensorDictBase, torch.Tensor or Number): the upper bound,
                paired with the leaves as in :meth:`~.clamp_max`. A tensor is
                aligned with the batch dimensions of each leaf, except when
                a leaf is an :class:`~tensordict.UnbatchedTensor`: then every
                leaf broadcasts with it from the right, as in torch.
        """
        if _mixes_lazy_stacks(self, (other,)):
            return _apply_pointwise(self, Tensor.clamp_max_, (other,), inplace=True)
        if _is_tensor_collection(type(other)):
            keys, vals = self._items_list(True, True)
            other_val = other._values_list(True, True, sorting_keys=keys)
        else:
            vals = self._values_list(True, True)
            other_val = other
        # Empty tensordict (no tensors) - nothing to do
        if not vals:
            return self
        if any(_is_unbatched(v) for v in vals):
            if _is_tensor_collection(type(other)):
                self.apply_(lambda x, y: x.clamp_max_(y), other)
            else:
                self.apply_(lambda x: x.clamp_max_(other))
            return self
        if isinstance(other, Tensor) and other.ndim:
            # broadcast the tensor to the batch dims, as clamp_max does
            clamp_max_ = _maybe_broadcast_other("clamp_max_")(type(self).clamp_max_)
            return clamp_max_(self, other)
        try:
            _foreach("clamp_max_", vals, other_val)
        except RuntimeError as err:
            if "isDifferentiableType" in str(err):
                raise RuntimeError(
                    "Attempted to execute _foreach_clamp_max_ with a differentiable tensor. "
                    "Use `td.apply(lambda x: x.clamp_max_(val)` instead."
                )
            raise
        return self

    @_maybe_broadcast_other("clamp_max")
    def clamp_max(
        self,
        other: TensorDictBase | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> Self:  # noqa: D417
        """Returns a new tensordict with every leaf clamped to at most ``other``.

        Args:
            other (TensorDictBase, torch.Tensor or Number): the upper bound. A
                tensordict is paired with the tensordict key by key. A tensor is
                aligned with the batch dimensions of each leaf and expanded over
                its feature dimensions. Batch sizes that differ are broadcast
                together.

        Keyword Args:
            default (torch.Tensor or str, optional): what to do with the keys
                that only one of the two tensordicts has. ``None`` requires the
                same keys and raises a ``KeyError`` otherwise.
                ``"intersection"`` computes the keys that both have and drops
                the others. A tensor stands in for the missing entry on either
                side, and the result has the keys of both. Defaults to ``None``.

        """
        keys, vals = self._items_list(True, True)
        # Empty tensordict (no tensors) - return a copy unchanged
        if not vals:
            return self.copy()
        if any(_is_unbatched(v) for v in vals):
            if _is_tensor_collection(type(other)):
                return self.apply(lambda x, y: x.clamp_max(y), other)
            else:
                return self.apply(lambda x: x.clamp_max(other))
        if _is_tensor_collection(type(other)):
            new_keys, other_val = other._items_list(
                True, True, sorting_keys=keys, default=default
            )
            if default is not None:
                as_dict = dict(zip(keys, vals))
                vals = [as_dict.get(key, default) for key in new_keys]
                keys = new_keys
        else:
            other_val = other
        try:
            vals = _foreach("clamp_max", vals, other_val)
        except RuntimeError as err:
            if "isDifferentiableType" in str(err):
                raise RuntimeError(
                    "Attempted to execute _foreach_clamp_max with a differentiable tensor. "
                    "Use `td.apply(lambda x: x.clamp_max(val)` instead."
                )
            raise
        items = dict(zip(keys, vals))

        def pop(name, val):
            return items.pop(name, None)

        result = self._fast_apply(
            pop,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    @_write_back_inplace
    def clamp_min_(self, other: TensorDictBase | torch.Tensor) -> Self:
        """Clamps every leaf to at least ``other`` in place and returns ``self``.

        The in-place version of :meth:`~.clamp_min`. It has no ``default``: a
        tensordict ``other`` must have every key of the tensordict, and its
        other keys are ignored.

        Args:
            other (TensorDictBase, torch.Tensor or Number): the lower bound,
                paired with the leaves as in :meth:`~.clamp_min`. A tensor is
                aligned with the batch dimensions of each leaf, except when
                a leaf is an :class:`~tensordict.UnbatchedTensor`: then every
                leaf broadcasts with it from the right, as in torch.
        """
        if _mixes_lazy_stacks(self, (other,)):
            return _apply_pointwise(self, Tensor.clamp_min_, (other,), inplace=True)
        if _is_tensor_collection(type(other)):
            keys, vals = self._items_list(True, True)
            other_val = other._values_list(True, True, sorting_keys=keys)
        else:
            vals = self._values_list(True, True)
            other_val = other
        # Empty tensordict (no tensors) - nothing to do
        if not vals:
            return self
        if any(_is_unbatched(v) for v in vals):
            if _is_tensor_collection(type(other)):
                self.apply_(lambda x, y: x.clamp_min_(y), other)
            else:
                self.apply_(lambda x: x.clamp_min_(other))
            return self
        if isinstance(other, Tensor) and other.ndim:
            # broadcast the tensor to the batch dims, as clamp_min does
            clamp_min_ = _maybe_broadcast_other("clamp_min_")(type(self).clamp_min_)
            return clamp_min_(self, other)
        try:
            _foreach("clamp_min_", vals, other_val)
        except RuntimeError as err:
            if "isDifferentiableType" in str(err):
                raise RuntimeError(
                    "Attempted to execute _foreach_clamp_min_ with a differentiable tensor. "
                    "Use `td.apply(lambda x: x.clamp_min_(val)` instead."
                )
            raise

        return self

    @_maybe_broadcast_other("clamp_min")
    def clamp_min(
        self,
        other: TensorDictBase | torch.Tensor,
        default: str | CompatibleType | None = None,
    ) -> Self:  # noqa: D417
        """Returns a new tensordict with every leaf clamped to at least ``other``.

        Args:
            other (TensorDictBase, torch.Tensor or Number): the lower bound. A
                tensordict is paired with the tensordict key by key. A tensor is
                aligned with the batch dimensions of each leaf and expanded over
                its feature dimensions. Batch sizes that differ are broadcast
                together.
            default (torch.Tensor or str, optional): what to do with the keys
                that only one of the two tensordicts has. ``None`` requires the
                same keys and raises a ``KeyError`` otherwise.
                ``"intersection"`` computes the keys that both have and drops
                the others. A tensor stands in for the missing entry on either
                side, and the result has the keys of both. Defaults to ``None``.

        """
        keys, vals = self._items_list(True, True)
        # Empty tensordict (no tensors) - return a copy unchanged
        if not vals:
            return self.copy()
        if any(_is_unbatched(v) for v in vals):
            if _is_tensor_collection(type(other)):
                return self.apply(lambda x, y: x.clamp_min(y), other)
            else:
                return self.apply(lambda x: x.clamp_min(other))
        if _is_tensor_collection(type(other)):
            new_keys, other_val = other._items_list(
                True, True, sorting_keys=keys, default=default
            )
            if default is not None:
                as_dict = dict(zip(keys, vals))
                vals = [as_dict.get(key, default) for key in new_keys]
                keys = new_keys
        else:
            other_val = other
        try:
            vals = _foreach("clamp_min", vals, other_val)
        except RuntimeError as err:
            if "isDifferentiableType" in str(err):
                raise RuntimeError(
                    "Attempted to execute _foreach_clamp_min with a differentiable tensor. "
                    "Use `td.apply(lambda x: x.clamp_min(val)` instead."
                )
            raise

        items = dict(zip(keys, vals))

        def pop(name, val):
            return items.pop(name, None)

        result = self._fast_apply(
            pop,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    @_clamp_into_out
    @_maybe_broadcast_other("clamp", 2)
    def clamp(
        self,
        min: TensorDictBase | torch.Tensor | float | None = None,
        max: TensorDictBase | torch.Tensor | float | None = None,
        *,
        out=None,
    ) -> Self:  # noqa: D417, W605
        r"""Returns a new tensordict with every leaf clamped into the range ``[min, max]``.

        .. math::
            y_i = \min(\max(x_i, \text{min}_i), \text{max}_i)

        Where ``min`` is greater than ``max``, the result is ``max``, as with
        :func:`torch.clamp`. At least one of ``min`` and ``max`` must be given;
        with only one bound, ``clamp`` calls :meth:`~.clamp_min` or
        :meth:`~.clamp_max`.

        Args:
            min (TensorDictBase, torch.Tensor, Number or None, optional): the
                lower bound. A tensordict is paired with the tensordict key by
                key. ``min`` and ``max`` must both be tensordicts, or neither.
                Defaults to ``None``, which sets no lower bound.
            max (TensorDictBase, torch.Tensor, Number or None, optional): the
                upper bound. ``min`` and ``max`` must both be tensordicts, or
                neither. Defaults to ``None``, which sets no upper bound.

        Keyword Args:
            out (TensorDictBase, optional): the tensordict to write the result
                into, and the one returned. The result is copied into its
                leaves, so ``out`` must have the keys and shapes of the
                result. Defaults to ``None``.

        Raises:
            ValueError: if one of ``min`` and ``max`` is a tensordict and the
                other is not, or if ``out`` is given and both ``min`` and
                ``max`` are ``None``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(x=torch.tensor([-2.0, 0.5, 3.0]), batch_size=[3])
            >>> td.clamp(0.0, 1.0)["x"]
            tensor([0.0000, 0.5000, 1.0000])
            >>> td.clamp(max=0.0)["x"]
            tensor([-2.,  0.,  0.])
        """
        # _clamp_into_out takes out: it never reaches this point
        if min is None:
            return self.clamp_max(max)
        if max is None:
            return self.clamp_min(min)

        is_tc_min = is_tensor_collection(min)
        is_tc_max = is_tensor_collection(max)

        if is_tc_min ^ is_tc_max:
            raise ValueError(
                "Mixed tensordict and non-tensordict min/max values are not authorized."
            )

        if is_tc_min and is_tc_max:
            return self._fast_apply(
                lambda x, low, high: x.clamp(low, high), min, max, default=None
            )
        return self._fast_apply(lambda x: x.clamp(min, max))

    @_write_back_inplace
    @_maybe_broadcast_other("pow_")
    def pow_(self, other: TensorDictBase | torch.Tensor) -> Self:
        """Raises every leaf to the power ``other`` in place and returns ``self``.

        The in-place version of :meth:`~.pow`, called by ``td **= other``. It
        has no ``default``: a tensordict ``other`` must have every key of the
        tensordict, and its other keys are ignored.

        Args:
            other (TensorDictBase, torch.Tensor or Number): the exponent,
                paired with the leaves as in :meth:`~.pow`.
        """
        if _is_tensor_collection(type(other)):
            keys, vals = self._items_list(True, True)
            other_val = other._values_list(True, True, sorting_keys=keys)
        else:
            vals = self._values_list(True, True)
            other_val = other
        if not vals:
            return self
        _foreach("pow_", vals, other_val)
        return self

    @_maybe_broadcast_other("pow")
    def pow(
        self,
        other: TensorDictBase | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> Self:  # noqa: D417
        r"""Returns a new tensordict with every leaf raised to the power ``other``.

        ``td ** other`` calls this method. ``other ** td`` with a number
        ``other`` raises a ``NotImplementedError``.

        Args:
            other (TensorDictBase, torch.Tensor or Number): the exponent. A
                tensordict is paired with the tensordict key by key. A tensor is
                aligned with the batch dimensions of each leaf and expanded over
                its feature dimensions. Batch sizes that differ are broadcast
                together.

        Keyword Args:
            default (torch.Tensor or str, optional): what to do with the keys
                that only one of the two tensordicts has. ``None`` requires the
                same keys and raises a ``KeyError`` otherwise.
                ``"intersection"`` computes the keys that both have and drops
                the others. A tensor stands in for the missing entry on either
                side, and the result has the keys of both. Defaults to ``None``.

        """
        keys, vals = self._items_list(True, True)
        no_tensors = not vals
        if _is_tensor_collection(type(other)):
            new_keys, other_val = other._items_list(
                True, True, sorting_keys=keys, default=default
            )
            if default is not None:
                as_dict = dict(zip(keys, vals))
                vals = [as_dict.get(key, default) for key in new_keys]
                keys = new_keys
        else:
            other_val = other
        if no_tensors and not vals:
            # No tensor in self, and default fills none in from other.
            return self.copy()
        vals = _foreach("pow", vals, other_val)
        items = dict(zip(keys, vals))

        def pop(name, val):
            return items.pop(name, None)

        result = self._fast_apply(
            pop,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    @_write_back_inplace
    @_maybe_broadcast_other("div_")
    def div_(self, other: TensorDictBase | torch.Tensor) -> Self:
        """Divides every leaf by ``other`` in place and returns ``self``.

        The in-place version of :meth:`~.div`, called by ``td /= other``. It
        has no ``default``: a tensordict ``other`` must have every key of the
        tensordict, and its other keys are ignored.

        Args:
            other (TensorDictBase, torch.Tensor or Number): the divisor,
                paired with the leaves as in :meth:`~.div`.
        """
        if _is_tensor_collection(type(other)):
            keys, vals = self._items_list(True, True)
            other_val = other._values_list(True, True, sorting_keys=keys)
        else:
            vals = self._values_list(True, True)
            other_val = other
        if not vals:
            return self
        _foreach("div_", vals, other_val)
        return self

    @_maybe_broadcast_other("div")
    def div(
        self,
        other: TensorDictBase | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> Self:  # noqa: D417
        r"""Returns a new tensordict with every leaf divided by ``other``.

        ``td / other`` calls this method. Integer leaves are promoted to the
        default floating-point dtype, as with :func:`torch.div`.

        .. math::
            \text{out}_i = \frac{\text{input}_i}{\text{other}_i}

        Args:
            other (TensorDictBase, torch.Tensor or Number): the divisor. A
                tensordict is paired with the tensordict key by key. A tensor is
                aligned with the batch dimensions of each leaf and expanded over
                its feature dimensions. Batch sizes that differ are broadcast
                together.

        Keyword Args:
            default (torch.Tensor or str, optional): what to do with the keys
                that only one of the two tensordicts has. ``None`` requires the
                same keys and raises a ``KeyError`` otherwise.
                ``"intersection"`` computes the keys that both have and drops
                the others. A tensor stands in for the missing entry on either
                side, and the result has the keys of both. Defaults to ``None``.

        """
        keys, vals = self._items_list(True, True)
        no_tensors = not vals
        if _is_tensor_collection(type(other)):
            new_keys, other_val = other._items_list(
                True, True, sorting_keys=keys, default=default
            )
            if default is not None:
                as_dict = dict(zip(keys, vals))
                vals = [as_dict.get(key, default) for key in new_keys]
                keys = new_keys
        else:
            other_val = other
        if no_tensors and not vals:
            # No tensor in self, and default fills none in from other.
            return self.copy()
        vals = _foreach("div", vals, other_val)
        items = dict(zip(keys, vals))

        def pop(name, val):
            return items.pop(name, None)

        result = self._fast_apply(
            pop,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            filter_empty=True,
            default=None,
        )
        if items:
            result.update(items)
        return result

    @_write_back_inplace
    def sqrt_(self) -> Self:
        """Applies :func:`torch.sqrt` to every leaf in place and returns ``self``."""
        vals = self._values_list(True, True)
        if vals:
            _foreach("sqrt_", vals)
        return self

    def sqrt(self) -> Self:
        """Returns a new tensordict with :func:`torch.sqrt` applied to every leaf."""
        keys, vals = self._items_list(True, True)
        if not vals:
            return self.copy()
        vals = _foreach("sqrt", vals)
        items = dict(zip(keys, vals))

        def get(name, val):
            return items.get(name, val)

        return self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
        )

    # Filling
    def zero_(self) -> Self:
        """Fills every leaf with zeros in place and returns ``self``."""
        if self._fills_without_reading:
            for key in self.keys():
                self._fill_((key,), 0)
            return self
        return self._zero_()

    @_write_back_inplace
    def _zero_(self) -> Self:
        """Zeros the leaves in place: :meth:`zero_` without ``_fills_without_reading``."""

        def fn(item):
            item.zero_()

        self._fast_apply(fn=fn, call_on_nested=True, propagate_lock=True)
        return self

    def fill_(self, key: NestedKey, value: float | bool) -> Self:
        """Fills the entry at ``key`` with ``value`` in place and returns ``self``.

        If ``key`` points to a nested tensordict, every leaf of it is filled.

        Args:
            key (NestedKey): the key of the entry to fill.
            value (Number or bool): the value to write.

        Raises:
            KeyError: if ``key`` is missing.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(2), b=TensorDict(c=torch.zeros(2)))
            >>> td.fill_("a", 1.0)["a"]
            tensor([1., 1.])
            >>> td.fill_("b", 2.0)["b", "c"]
            tensor([2., 2.])
        """
        self._fill_(_unravel_key_to_tuple(key), value)
        return self

    def _fill_(self, key: tuple[str, ...], value: float | bool) -> None:
        """Fills the entry at ``key``, a leaf or every leaf of a nested tensordict, with ``value``.

        Called by :meth:`fill_` with the key as a tuple; :meth:`fill_` returns
        ``self`` afterwards, and so does :meth:`zero_`, which calls it with
        ``0`` for each key when ``_fills_without_reading`` is set. A missing key
        raises a ``KeyError``. The base implementation reads the entry, fills
        it and writes it back; when ``_nested_reads_return_copies`` is set, it
        writes each leaf of a nested tensordict back under its full key.
        Override it when the storage can write a constant without reading the
        entry.
        """
        data = self._get_tuple(key, NO_DEFAULT)
        if not _is_tensor_collection(type(data)):
            data = data.fill_(value)
            self._set_tuple(key, data, inplace=True, validated=True, non_blocking=False)
        elif self._nested_reads_return_copies:
            # data is a new tensordict: its leaves are written back
            for subkey, leaf in data.items(True, True):
                self._set_tuple(
                    key + _unravel_key_to_tuple(subkey),
                    leaf.fill_(value),
                    inplace=True,
                    validated=True,
                    non_blocking=False,
                )
        else:

            def fill(x):
                return x.fill_(value)

            data._fast_apply(fill, inplace=True)

    def where(
        self,
        condition: Tensor,
        other: Tensor | TensorDictBase,
        *,
        out: TensorDictBase | None = None,
        pad: int | bool = None,
        update_batch_size: bool = False,
    ) -> Self:  # noqa: D417
        """Returns a tensordict with the leaves where ``condition`` is ``True`` and ``other`` elsewhere.

        :func:`torch.where` is applied to every leaf. ``condition`` is aligned
        with the batch dimensions of each leaf and expanded over its feature
        dimensions.

        Args:
            condition (torch.Tensor): a boolean mask over the leading batch
                dimensions.
            other (TensorDictBase, torch.Tensor or Number): the values to take
                where ``condition`` is ``False``. A tensordict is paired with
                the tensordict key by key. On a
                :class:`~tensordict.LazyStackedTensorDict`, a tensor ``other``
                that starts with the batch shape up to the stack dimension is
                split along that dimension, one piece per member, so it must
                have the stack size there: expand it to the batch shape first.

        Keyword Args:
            out (TensorDictBase, optional): the tensordict to write the result
                into, which is returned. Defaults to ``None``, which returns a
                new tensordict.
            pad (Number or bool, optional): the value that stands in for an
                entry that only one of the tensordict and a tensordict
                ``other`` has. Defaults to ``None``, which raises a
                ``KeyError`` for such an entry.
            update_batch_size (bool, optional): passed to :meth:`~.update`
                when the result is written into ``out``. It only matters when
                ``out`` is a lazy stack: if ``True``, its number of members
                follows the result. Defaults to ``False``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(x=torch.ones(2, 3), batch_size=[2])
            >>> td.where(torch.tensor([True, False]), 0.0)["x"]
            tensor([[1., 1., 1.],
                    [0., 0., 0.]])
            >>> other = TensorDict(y=torch.zeros(2), batch_size=[2])
            >>> sorted(td.where(torch.tensor([True, False]), other, pad=-1).keys())
            ['x', 'y']
        """
        if self._masking_on_dense_copy:
            return self.to_tensordict().where(
                condition=condition,
                other=other,
                out=out,
                pad=pad,
                update_batch_size=update_batch_size,
            )
        return self._where(
            condition, other, out=out, pad=pad, update_batch_size=update_batch_size
        )

    def _where(
        self,
        condition: Tensor,
        other: Tensor | TensorDictBase,
        *,
        out: TensorDictBase | None,
        pad: int | bool | None,
        update_batch_size: bool,
    ) -> TensorDictBase:
        """Storage-specific part of :meth:`where`, called when ``_masking_on_dense_copy`` is False.

        Takes the arguments of :meth:`where` as they are and returns its result,
        written into ``out`` when ``out`` is given. The base pairs the entries
        by key. :class:`~tensordict.LazyStackedTensorDict` splits ``condition``
        (and a tensordict or batch-shaped tensor ``other``) along the stack dim,
        runs :meth:`where` on each member and stacks the results lazily.
        """
        if _is_tensor_collection(type(other)):

            def func(tensor, _other, key):
                if tensor is None:
                    if pad is not None:
                        tensor = _other
                        _other = torch.tensor(
                            pad, dtype=_other.dtype, device=_other.device
                        )
                    else:
                        raise KeyError(
                            f"Key {key} not found and no pad value provided."
                        )
                    cond = expand_as_right(~condition, tensor)
                elif _other is None:
                    if pad is not None:
                        _other = torch.tensor(
                            pad, dtype=tensor.dtype, device=tensor.device
                        )
                    else:
                        raise KeyError(
                            f"Key {key} not found and no pad value provided."
                        )
                    cond = expand_as_right(condition, tensor)
                else:
                    cond = expand_as_right(condition, tensor)
                return torch.where(
                    condition=cond,
                    input=tensor,
                    other=_other,
                )

            result = self.empty() if out is None else out
            other_keys = set(other.keys())
            # we turn into a list because out could be = to self!
            for key in list(self.keys()):
                tensor = self._get_str(key, default=NO_DEFAULT)
                _other = other._get_str(key, default=None)
                if _is_tensor_collection(type(tensor)):
                    _out = None if out is None else out._get_str(key, None)
                    if _other is None:
                        _other = tensor.empty()
                    val = tensor.where(
                        condition=condition, other=_other, out=_out, pad=pad
                    )
                else:
                    val = func(tensor, _other, key)
                result._set_str(
                    key, val, inplace=False, validated=True, non_blocking=False
                )
                other_keys.discard(key)
            for key in other_keys:
                tensor = None
                _other = other._get_str(key, default=NO_DEFAULT)
                if _is_tensor_collection(type(_other)):
                    try:
                        tensor = _other.empty()
                    except NotImplementedError:
                        # H5 tensordicts do not support select()
                        tensor = _other.to_tensordict().empty()
                    val = _other.where(
                        condition=~condition, other=tensor, out=None, pad=pad
                    )
                else:
                    val = func(tensor, _other, key)
                result._set_str(
                    key, val, inplace=False, validated=True, non_blocking=False
                )
            return result
        else:

            def func(tensor):
                return torch.where(
                    condition=expand_as_right(condition, tensor),
                    input=tensor,
                    other=other,
                )

            result = self._fast_apply(func, propagate_lock=True)
            if out is not None:
                # As in LazyStackedTensorDict.where. A result without leaves
                # has nothing to write, and out may be locked.
                if not result.is_empty():
                    out.update(result, update_batch_size=update_batch_size)
                return out
            return result

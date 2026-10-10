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

import contextlib
import numbers
from typing import Any, TYPE_CHECKING

import torch
from tensordict._nestedkey import NestedKey
from tensordict._tensorcollection import TensorCollection
from tensordict.base import (
    _is_tensor_collection,
    _maybe_broadcast_other,
    _NESTED_TENSORS_AS_LISTS,
    CompatibleType,
    is_tensor_collection,
    NO_DEFAULT,
    Self,
)
from tensordict.utils import (
    _is_unbatched,
    _maybe_correct_neg_dim,
    _mismatch_keys,
    _unravel_key_to_tuple,
    is_tensorclass,
)
from torch import Tensor

if TYPE_CHECKING:
    from tensordict.base import TensorDictBase


class _PointwiseOps:
    """Elementwise arithmetic, comparison and logical operations."""

    def __abs__(self) -> Self:
        """Returns a new TensorDict instance with absolute values of all tensors.

        Returns:
            A new TensorDict instance with the same key set as the original,
            but with all tensors having their absolute values computed.

        .. seealso:: :meth:`~.abs`

        """
        return self.abs()

    def __neg__(self) -> Self:
        """Returns a new TensorDict instance with negated values of all tensors.

        Returns:
            A new TensorDict instance with the same key set as the original,
            but with all tensors having their values negated.

        .. seealso:: :meth:`~.neg`

        """
        return self.neg()

    @_maybe_broadcast_other("__ne__")
    def __ne__(self, other: Any) -> Self | bool:
        """NOT operation over two tensordicts, for evey key.

        The two tensordicts must have the same key set.

        Args:
            other (TensorDictBase, dict, or float): the value to compare against.

        Returns:
            a new TensorDict instance with all tensors are boolean
            tensors of the same shape as the original tensors.

        """
        from tensordict._td import TensorDict

        if is_tensorclass(other):
            return other != self
        if isinstance(other, (dict,)):
            other = self.from_dict_instance(other, auto_batch_size=False)
        if _is_tensor_collection(type(other)):
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                raise KeyError(
                    f"keys in {self} and {other} mismatch, got {keys1} and {keys2}"
                )
            d = {}
            for key, item1 in self.items():
                d[key] = item1 != other.get(key)
            return TensorDict(batch_size=self.batch_size, source=d, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return TensorDict(
                {key: value != other for key, value in self.items()},
                self.batch_size,
                device=self.device,
            )
        return True

    @_maybe_broadcast_other("__xor__")
    def __xor__(self, other: Any) -> Self | bool:
        """XOR operation over two tensordicts, for evey key.

        The two tensordicts must have the same key set.

        Args:
            other (TensorDictBase, dict, or float): the value to compare against.

        Returns:
            a new TensorDict instance with all tensors are boolean
            tensors of the same shape as the original tensors.

        """
        from tensordict._td import TensorDict

        if is_tensorclass(other):
            return other ^ self
        if isinstance(other, (dict,)):
            other = self.from_dict_instance(other, auto_batch_size=False)
        if _is_tensor_collection(type(other)):
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                raise KeyError(
                    f"keys in {self} and {other} mismatch, got {keys1} and {keys2}"
                )
            d = {}
            for key, item1 in self.items():
                d[key] = item1 ^ other.get(key)
            return TensorDict(batch_size=self.batch_size, source=d, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return TensorDict(
                {key: value ^ other for key, value in self.items()},
                self.batch_size,
                device=self.device,
            )
        return True

    def __rxor__(self, other: TensorCollection | torch.Tensor | float):
        """XOR operation over two tensordicts, for evey key.

        Wraps `__xor__` as it is assumed to be commutative.
        """
        return self.__xor__(other)

    @_maybe_broadcast_other("__or__")
    def __or__(self, other: Any) -> Self | bool:
        """OR operation over two tensordicts, for evey key.

        The two tensordicts must have the same key set.

        Args:
            other (TensorDictBase, dict, or float): the value to compare against.

        Returns:
            a new TensorDict instance with all tensors are boolean
            tensors of the same shape as the original tensors.

        """
        from tensordict._td import TensorDict

        if is_tensorclass(other):
            return other | self
        if isinstance(other, (dict,)):
            other = self.from_dict_instance(other, auto_batch_size=False)
        if _is_tensor_collection(type(other)):
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                raise KeyError(
                    f"keys in {self} and {other} mismatch, got {keys1} and {keys2}"
                )
            d = {}
            for key, item1 in self.items():
                d[key] = item1 | other.get(key)
            return TensorDict(batch_size=self.batch_size, source=d, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return TensorDict(
                {key: value | other for key, value in self.items()},
                self.batch_size,
                device=self.device,
            )
        return False

    def __ror__(self, other: TensorCollection | torch.Tensor) -> Self:
        """Right-side OR operation over two tensordicts, for evey key.

        This is a wrapper around `__or__` since it is assumed to be commutative.
        """
        return self | other

    def __invert__(self) -> Self:
        """Returns a new TensorDict instance with all tensors inverted (i.e., bitwise NOT operation).

        Returns:
            A new TensorDict instance with the same key set as the original,
            but with all tensors having their bits inverted.
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
        """Returns a new TensorDict instance with all tensors performing a logical or bitwise AND operation with the given value.

        Args:
            other: The value to perform the AND operation with.

        Returns:
            A new TensorDict instance with the same key set as the original,
            but with all tensors having performed a AND operation with the given value.
        """
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

    @_maybe_broadcast_other("__eq__")
    def __eq__(self, other: Any) -> Self | bool:
        """Compares two tensordicts against each other, for every key. The two tensordicts must have the same key set.

        Returns:
            a new TensorDict instance with all tensors are boolean
            tensors of the same shape as the original tensors.

        """
        from tensordict._td import TensorDict

        if is_tensorclass(other):
            return other == self
        if isinstance(other, (dict,)):
            other = self.from_dict_instance(other, auto_batch_size=False)
        if _is_tensor_collection(type(other)):
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                _mismatch_keys(keys1, keys2)
            d = {}
            for key, item1 in self.items():
                d[key] = item1 == other.get(key)
            return TensorDict(source=d, batch_size=self.batch_size, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return TensorDict(
                {key: value == other for key, value in self.items()},
                self.batch_size,
                device=self.device,
            )
        return False

    @_maybe_broadcast_other("__ge__")
    def __ge__(self, other: Any) -> Self | bool:
        """Compares two tensordicts against each other using the "greater or equal" operator, for every key. The two tensordicts must have the same key set.

        Returns:
            a new TensorDict instance with all tensors are boolean
            tensors of the same shape as the original tensors.

        """
        from tensordict._td import TensorDict

        if is_tensorclass(other):
            return other <= self
        if isinstance(other, (dict,)):
            other = self.from_dict_instance(other, auto_batch_size=False)
        if _is_tensor_collection(type(other)):
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                _mismatch_keys(keys1, keys2)
            d = {}
            for key, item1 in self.items():
                d[key] = item1 >= other.get(key)
            return TensorDict(source=d, batch_size=self.batch_size, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return TensorDict(
                {key: value >= other for key, value in self.items()},
                self.batch_size,
                device=self.device,
            )
        return False

    @_maybe_broadcast_other("__gt__")
    def __gt__(self, other: Any) -> Self | bool:
        """Compares two tensordicts against each other using the "greater than" operator, for every key. The two tensordicts must have the same key set.

        Returns:
            a new TensorDict instance with all tensors are boolean
            tensors of the same shape as the original tensors.

        """
        from tensordict._td import TensorDict

        if is_tensorclass(other):
            return other < self
        if isinstance(other, (dict,)):
            other = self.from_dict_instance(other, auto_batch_size=False)
        if _is_tensor_collection(type(other)):
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                _mismatch_keys(keys1, keys2)
            d = {}
            for key, item1 in self.items():
                d[key] = item1 > other.get(key)
            return TensorDict(source=d, batch_size=self.batch_size, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return TensorDict(
                {key: value > other for key, value in self.items()},
                self.batch_size,
                device=self.device,
            )
        return False

    @_maybe_broadcast_other("__le__")
    def __le__(self, other: Any) -> Self | bool:
        """Compares two tensordicts against each other using the "lower or equal" operator, for every key. The two tensordicts must have the same key set.

        Returns:
            a new TensorDict instance with all tensors are boolean
            tensors of the same shape as the original tensors.

        """
        from tensordict._td import TensorDict

        if is_tensorclass(other):
            return other >= self
        if isinstance(other, (dict,)):
            other = self.from_dict_instance(other, auto_batch_size=False)
        if _is_tensor_collection(type(other)):
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                _mismatch_keys(keys1, keys2)
            d = {}
            for key, item1 in self.items():
                d[key] = item1 <= other.get(key)
            return TensorDict(source=d, batch_size=self.batch_size, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return TensorDict(
                {key: value <= other for key, value in self.items()},
                self.batch_size,
                device=self.device,
            )
        return False

    @_maybe_broadcast_other("__lt__")
    def __lt__(self, other: Any) -> Self | bool:
        """Compares two tensordicts against each other using the "lower than" operator, for every key. The two tensordicts must have the same key set.

        Returns:
            a new TensorDict instance with all tensors are boolean
            tensors of the same shape as the original tensors.

        """
        from tensordict._td import TensorDict

        if is_tensorclass(other):
            return other > self
        if isinstance(other, (dict,)):
            other = self.from_dict_instance(other, auto_batch_size=False)
        if _is_tensor_collection(type(other)):
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                _mismatch_keys(keys1, keys2)
            d = {}
            for key, item1 in self.items():
                d[key] = item1 < other.get(key)
            return TensorDict(source=d, batch_size=self.batch_size, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return TensorDict(
                {key: value < other for key, value in self.items()},
                self.batch_size,
                device=self.device,
            )
        return False

    def isfinite(self) -> Self:
        """Returns a new tensordict with boolean elements representing if each element is finite or not.

        Real values are finite when they are not NaN, negative infinity, or infinity. Complex values are finite when both their real and imaginary parts are finite.

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
        """Returns a new tensordict with boolean elements representing if each element of input is NaN or not.

        Complex values are considered NaN when either their real and/or imaginary part is NaN.

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
        """Tests if each element of input is negative infinity or not."""
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
        """Tests if each element of input is negative infinity or not."""
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
        """Returns a new tensordict with boolean elements representing if each element of input is real-valued or not."""
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
        """Computes the absolute value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_abs(vals)
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

    def abs_(self) -> Self:
        """Computes the absolute value of each element of the TensorDict in-place."""
        torch._foreach_abs_(self._values_list(True, True))
        return self

    def acos(self) -> Self:
        """Computes the :meth:`~torch.acos` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_acos(vals)
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

    def acos_(self) -> Self:
        """Computes the :meth:`~torch.acos` value of each element of the TensorDict in-place."""
        torch._foreach_acos_(self._values_list(True, True))
        return self

    def exp(self) -> Self:
        """Computes the :meth:`~torch.exp` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_exp(vals)
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

    def exp_(self) -> Self:
        """Computes the :meth:`~torch.exp` value of each element of the TensorDict in-place."""
        torch._foreach_exp_(self._values_list(True, True))
        return self

    def neg(self) -> Self:
        """Computes the :meth:`~torch.neg` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        # Empty tensordict (no tensors) - return a copy unchanged
        if not vals:
            return self.copy()
        if any(_is_unbatched(v) for v in vals):
            return self.apply(lambda x: -x)
        vals = torch._foreach_neg(vals)
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

    def neg_(self) -> Self:
        """Computes the :meth:`~torch.neg` value of each element of the TensorDict in-place."""
        vals = self._values_list(True, True)
        # Empty tensordict (no tensors) - nothing to do
        if not vals:
            return self
        if any(_is_unbatched(v) for v in vals):
            self.apply_(lambda x: x.neg_())
            return self
        torch._foreach_neg_(vals)
        return self

    def reciprocal(self) -> Self:
        """Computes the :meth:`~torch.reciprocal` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_reciprocal(vals)
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

    def reciprocal_(self) -> Self:
        """Computes the :meth:`~torch.reciprocal` value of each element of the TensorDict in-place."""
        torch._foreach_reciprocal_(self._values_list(True, True))
        return self

    def sigmoid(self) -> Self:
        """Computes the :meth:`~torch.sigmoid` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_sigmoid(vals)
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

    def sigmoid_(self) -> Self:
        """Computes the :meth:`~torch.sigmoid` value of each element of the TensorDict in-place."""
        torch._foreach_sigmoid_(self._values_list(True, True))
        return self

    def sign(self) -> Self:
        """Computes the :meth:`~torch.sign` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_sign(vals)
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

    def sign_(self) -> Self:
        """Computes the :meth:`~torch.sign` value of each element of the TensorDict in-place."""
        torch._foreach_sign_(self._values_list(True, True))
        return self

    def sin(self) -> Self:
        """Computes the :meth:`~torch.sin` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_sin(vals)
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

    def sin_(self) -> Self:
        """Computes the :meth:`~torch.sin` value of each element of the TensorDict in-place."""
        torch._foreach_sin_(self._values_list(True, True))
        return self

    def sinh(self) -> Self:
        """Computes the :meth:`~torch.sinh` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_sinh(vals)
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

    def sinh_(self) -> Self:
        """Computes the :meth:`~torch.sinh` value of each element of the TensorDict in-place."""
        torch._foreach_sinh_(self._values_list(True, True))
        return self

    def tan(self) -> Self:
        """Computes the :meth:`~torch.tan` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_tan(vals)
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

    def tan_(self) -> Self:
        """Computes the :meth:`~torch.tan` value of each element of the TensorDict in-place."""
        torch._foreach_tan_(self._values_list(True, True))
        return self

    def tanh(self) -> Self:
        """Computes the :meth:`~torch.tanh` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_tanh(vals)
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

    def tanh_(self) -> Self:
        """Computes the :meth:`~torch.tanh` value of each element of the TensorDict in-place."""
        torch._foreach_tanh_(self._values_list(True, True))
        return self

    def trunc(self) -> Self:
        """Computes the :meth:`~torch.trunc` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_trunc(vals)
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

    def trunc_(self) -> Self:
        """Computes the :meth:`~torch.trunc` value of each element of the TensorDict in-place."""
        torch._foreach_trunc_(self._values_list(True, True))
        return self

    def lgamma(self) -> Self:
        """Computes the :meth:`~torch.lgamma` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_lgamma(vals)
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

    def lgamma_(self) -> Self:
        """Computes the :meth:`~torch.lgamma` value of each element of the TensorDict in-place."""
        torch._foreach_lgamma_(self._values_list(True, True))
        return self

    def frac(self) -> Self:
        """Computes the :meth:`~torch.frac` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_frac(vals)
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

    def frac_(self) -> Self:
        """Computes the :meth:`~torch.frac` value of each element of the TensorDict in-place."""
        torch._foreach_frac_(self._values_list(True, True))
        return self

    def expm1(self) -> Self:
        """Computes the :meth:`~torch.expm1` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_expm1(vals)
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

    def expm1_(self) -> Self:
        """Computes the :meth:`~torch.expm1` value of each element of the TensorDict in-place."""
        torch._foreach_expm1_(self._values_list(True, True))
        return self

    def log(self) -> Self:
        """Computes the :meth:`~torch.log` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_log(vals)
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

    def log_(self) -> Self:
        """Computes the :meth:`~torch.log` value of each element of the TensorDict in-place."""
        torch._foreach_log_(self._values_list(True, True))
        return self

    def softmax(self, dim: int, dtype: torch.dtype | None = None):  # noqa: D417
        """Apply a softmax function to the tensordict elements.

        Args:
            dim (int or tuple of ints): A tensordict dimension along which softmax will be computed.
            dtype (torch.dtype, optional): the desired data type of returned tensor.
                If specified, the input tensor is cast to dtype before the operation is performed.
                This is useful for preventing data type overflows.

        """
        if isinstance(dim, int):
            dim = _maybe_correct_neg_dim(dim, self.batch_size)
        else:
            raise ValueError(f"Expected dim of type int, got {type(dim)}.")
        return self._fast_apply(
            lambda x: torch.softmax(x, dim=dim, dtype=dtype),
        )

    def log10(self) -> Self:
        """Computes the :meth:`~torch.log10` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_log10(vals)
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

    def log10_(self) -> Self:
        """Computes the :meth:`~torch.log10` value of each element of the TensorDict in-place."""
        torch._foreach_log10_(self._values_list(True, True))
        return self

    def log1p(self) -> Self:
        """Computes the :meth:`~torch.log1p` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_log1p(vals)
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

    def log1p_(self) -> Self:
        """Computes the :meth:`~torch.log1p` value of each element of the TensorDict in-place."""
        torch._foreach_log1p_(self._values_list(True, True))
        return self

    def log2(self) -> Self:
        """Computes the :meth:`~torch.log2` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_log2(vals)
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

    def log2_(self) -> Self:
        """Computes the :meth:`~torch.log2` value of each element of the TensorDict in-place."""
        torch._foreach_log2_(self._values_list(True, True))
        return self

    def ceil(self) -> Self:
        """Computes the :meth:`~torch.ceil` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_ceil(vals)
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

    def ceil_(self) -> Self:
        """Computes the :meth:`~torch.ceil` value of each element of the TensorDict in-place."""
        torch._foreach_ceil_(self._values_list(True, True))
        return self

    def floor(self) -> Self:
        """Computes the :meth:`~torch.floor` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_floor(vals)
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

    def floor_(self) -> Self:
        """Computes the :meth:`~torch.floor` value of each element of the TensorDict in-place."""
        torch._foreach_floor_(self._values_list(True, True))
        return self

    def round(self) -> Self:
        """Computes the :meth:`~torch.round` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_round(vals)
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

    def round_(self) -> Self:
        """Computes the :meth:`~torch.round` value of each element of the TensorDict in-place."""
        torch._foreach_round_(self._values_list(True, True))
        return self

    def erf(self) -> Self:
        """Computes the :meth:`~torch.erf` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_erf(vals)
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

    def erf_(self) -> Self:
        """Computes the :meth:`~torch.erf` value of each element of the TensorDict in-place."""
        torch._foreach_erf_(self._values_list(True, True))
        return self

    def erfc(self) -> Self:
        """Computes the :meth:`~torch.erfc` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_erfc(vals)
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

    def erfc_(self) -> Self:
        """Computes the :meth:`~torch.erfc` value of each element of the TensorDict in-place."""
        torch._foreach_erfc_(self._values_list(True, True))
        return self

    def asin(self) -> Self:
        """Computes the :meth:`~torch.asin` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_asin(vals)
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

    def asin_(self) -> Self:
        """Computes the :meth:`~torch.asin` value of each element of the TensorDict in-place."""
        torch._foreach_asin_(self._values_list(True, True))
        return self

    def atan(self) -> Self:
        """Computes the :meth:`~torch.atan` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_atan(vals)
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

    def atan_(self) -> Self:
        """Computes the :meth:`~torch.atan` value of each element of the TensorDict in-place."""
        torch._foreach_atan_(self._values_list(True, True))
        return self

    def cos(self) -> Self:
        """Computes the :meth:`~torch.cos` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_cos(vals)
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

    def cos_(self) -> Self:
        """Computes the :meth:`~torch.cos` value of each element of the TensorDict in-place."""
        torch._foreach_cos_(self._values_list(True, True))
        return self

    def cosh(self) -> Self:
        """Computes the :meth:`~torch.cosh` value of each element of the TensorDict."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_cosh(vals)
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

    def cosh_(self) -> Self:
        """Computes the :meth:`~torch.cosh` value of each element of the TensorDict in-place."""
        torch._foreach_cosh_(self._values_list(True, True))
        return self

    @_maybe_broadcast_other("bitwise_and")
    def bitwise_and(
        self,
        other: TensorCollection | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> Self:  # noqa: D417
        r"""Performs a bitwise AND operation between ``self`` and :attr:`other`.

        .. math::
            \text{{out}}_i = \text{{input}}_i \land \text{{other}}_i

        Args:
            other (TensorDictBase or torch.Tensor): the tensor or TensorDict to perform the bitwise AND with.

        Keyword Args:
            default (torch.Tensor or str, optional): the default value to use for exclusive entries.
                If none is provided, the two tensordicts key list must match exactly.
                If ``default="intersection"`` is passed, only the intersecting key sets will be considered
                and other keys will be ignored.
                In all other cases, ``default`` will be used for all missing entries on both sides of the
                operation.
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
        r"""Performs a logical AND operation between ``self`` and :attr:`other`.

        .. math::
            \text{{out}}_i = \text{{input}}_i \land \text{{other}}_i

        Args:
            other (TensorDictBase or torch.Tensor): the tensor or TensorDict to perform the logical AND with.

        Keyword Args:
            default (torch.Tensor or str, optional): the default value to use for exclusive entries.
                If none is provided, the two tensordicts key list must match exactly.
                If ``default="intersection"`` is passed, only the intersecting key sets will be considered
                and other keys will be ignored.
                In all other cases, ``default`` will be used for all missing entries on both sides of the
                operation.
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
        r"""Adds :attr:`other`, scaled by :attr:`alpha`, to ``self``.

        .. math::
            \text{{out}}_i = \text{{input}}_i + \text{{alpha}} \times \text{{other}}_i

        Args:
            other (TensorDictBase or torch.Tensor): the tensor or TensorDict to add to ``self``.

        Keyword Args:
            alpha (Number, optional): the multiplier for :attr:`other`.
            default (torch.Tensor or str, optional): the default value to use for exclusive entries.
                If none is provided, the two tensordicts key list must match exactly.
                If ``default="intersection"`` is passed, only the intersecting key sets will be considered
                and other keys will be ignored.
                In all other cases, ``default`` will be used for all missing entries on both sides of the
                operation.

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
        if alpha is not None:
            vals = torch._foreach_add(vals, other_val, alpha=alpha)
        else:
            vals = torch._foreach_add(vals, other_val)
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

    @_maybe_broadcast_other("add_")
    def add_(
        self,
        other: TensorCollection | torch.Tensor | float,
        *,
        alpha: float | None = None,
    ) -> Self:
        """In-place version of :meth:`~.add`.

        .. note::
            In-place ``add`` does not support ``default`` keyword argument.
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
        if alpha is not None:
            torch._foreach_add_(vals, other_val, alpha=alpha)
        else:
            torch._foreach_add_(vals, other_val)
        return self

    @_maybe_broadcast_other("lerp", 2)
    def lerp(
        self,
        end: TensorCollection | torch.Tensor,
        weight: TensorCollection | torch.Tensor | float,
    ) -> Self:
        r"""Does a linear interpolation of two tensors :attr:`start` (given by ``self``) and :attr:`end` based on a scalar or tensor :attr:`weight`.

        .. math::
            \text{out}_i = \text{start}_i + \text{weight}_i \times (\text{end}_i - \text{start}_i)

        The shapes of :attr:`start` and :attr:`end` must be
        broadcastable. If :attr:`weight` is a tensor, then
        the shapes of :attr:`weight`, :attr:`start`, and :attr:`end` must be broadcastable.

        Args:
            end (TensorDict): the tensordict with the ending points.
            weight (TensorDict, tensor or float): the weight for the interpolation formula.

        """
        keys, vals = self._items_list(True, True)
        if _is_tensor_collection(type(end)):
            end_val = end._values_list(True, True)
        else:
            end_val = end
        if isinstance(weight, (float, torch.Tensor)):
            weight_val = weight
        elif _is_tensor_collection(type(weight)):
            weight_val = weight._values_list(True, True)
        else:
            weight_val = weight
        vals = torch._foreach_lerp(vals, end_val, weight_val)
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

    def lerp_(
        self,
        end: TensorDictBase | torch.Tensor | float,
        weight: TensorDictBase | torch.Tensor | float,
    ):
        """In-place version of :meth:`~.lerp`."""
        if _is_tensor_collection(type(end)):
            end_val = end._values_list(True, True)
        else:
            end_val = end
        if isinstance(weight, (float, torch.Tensor)):
            weight_val = weight
        elif _is_tensor_collection(type(weight)):
            weight_val = weight._values_list(True, True)
        else:
            weight_val = weight
        torch._foreach_lerp_(self._values_list(True, True), end_val, weight_val)
        return self

    @_maybe_broadcast_other("addcdiv", 2)
    def addcdiv(
        self,
        other1: TensorDictBase | torch.Tensor,
        other2: TensorDictBase | torch.Tensor,
        value: float | None = 1,
    ) -> Self:  # noqa: D417
        r"""Performs the element-wise division of :attr:`other1` by :attr:`other2`, multiplies the result by the scalar :attr:`value` and adds it to ``self``.

        .. math::
            \text{out}_i = \text{input}_i + \text{value} \times \frac{\text{other1}_i}{\text{other2}_i}

        The shapes of the elements of ``self``, :attr:`other1`, and :attr:`other2` must be
        broadcastable.

        For inputs of type `FloatTensor` or `DoubleTensor`, :attr:`value` must be
        a real number, otherwise an integer.

        Args:
            other1 (TensorDict or Tensor): the numerator tensordict (or tensor)
            other2 (TensorDict or Tensor): the denominator tensordict (or tensor)

        Keyword Args:
            value (Number, optional): multiplier for :math:`\text{other1} / \text{other2}`
        """
        keys, vals = self._items_list(True, True)
        if _is_tensor_collection(type(other1)):
            other1_val = other1._values_list(True, True)
        else:
            other1_val = other1
        if _is_tensor_collection(type(other2)):
            other2_val = other2._values_list(True, True)
        else:
            other2_val = other2
        vals = torch._foreach_addcdiv(vals, other1_val, other2_val, value=value)
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

    @_maybe_broadcast_other("addcdiv_", 2)
    def addcdiv_(self, other1, other2, *, value: float | None = 1):
        """The in-place version of :meth:`~.addcdiv`."""
        if _is_tensor_collection(type(other1)):
            other1_val = other1._values_list(True, True)
        else:
            other1_val = other1
        if _is_tensor_collection(type(other2)):
            other2_val = other2._values_list(True, True)
        else:
            other2_val = other2
        torch._foreach_addcdiv_(
            self._values_list(True, True), other1_val, other2_val, value=value
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
        r"""Performs the element-wise multiplication of :attr:`other1` by :attr:`other2`, multiplies the result by the scalar :attr:`value` and adds it to ``self``.

        .. math::
            \text{out}_i = \text{input}_i + \text{value} \times \text{other1}_i \times \text{other2}_i

        The shapes of ``self``, :attr:`other1`, and :attr:`other2` must be
        broadcastable.

        For inputs of type `FloatTensor` or `DoubleTensor`, :attr:`value` must be
        a real number, otherwise an integer.

        Args:
            other1 (TensorDict or Tensor): the tensordict or tensor to be multiplied
            other2 (TensorDict or Tensor): the tensordict or tensor to be multiplied

        Keyword Args:
            value (Number, optional): multiplier for :math:`other1 .* other2`
        """
        keys, vals = self._items_list(True, True)
        if _is_tensor_collection(type(other1)):
            other1_val = other1._values_list(True, True)
        else:
            other1_val = other1
        if _is_tensor_collection(type(other2)):
            other2_val = other2._values_list(True, True)
        else:
            other2_val = other2
        vals = torch._foreach_addcmul(vals, other1_val, other2_val, value=value)
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

    @_maybe_broadcast_other("addcmul_", 2)
    def addcmul_(self, other1, other2, *, value: float | None = 1):
        """The in-place version of :meth:`~.addcmul`."""
        if _is_tensor_collection(type(other1)):
            other1_val = other1._values_list(True, True)
        else:
            other1_val = other1
        if _is_tensor_collection(type(other2)):
            other2_val = other2._values_list(True, True)
        else:
            other2_val = other2
        torch._foreach_addcmul_(
            self._values_list(True, True), other1_val, other2_val, value=value
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
        r"""Subtracts :attr:`other`, scaled by :attr:`alpha`, from ``self``.

        .. math::
            \text{{out}}_i = \text{{input}}_i - \text{{alpha}} \times \text{{other}}_i

        Supports broadcasting,
        type promotion, and integer, float, and complex inputs.

        Args:
            other (TensorDict, Tensor or Number): the tensor or number to subtract from ``self``.

        Keyword Args:
            alpha (Number): the multiplier for :attr:`other`.
            default (torch.Tensor or str, optional): the default value to use for exclusive entries.
                If none is provided, the two tensordicts key list must match exactly.
                If ``default="intersection"`` is passed, only the intersecting key sets will be considered
                and other keys will be ignored.
                In all other cases, ``default`` will be used for all missing entries on both sides of the
                operation.

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
        else:
            other_val = other
        if alpha is not None:
            vals = torch._foreach_sub(vals, other_val, alpha=alpha)
        else:
            vals = torch._foreach_sub(vals, other_val)
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

    @_maybe_broadcast_other("sub_")
    def sub_(
        self, other: TensorDictBase | torch.Tensor | float, alpha: float | None = None
    ):
        """In-place version of :meth:`~.sub`.

        .. note::
            In-place ``sub`` does not support ``default`` keyword argument.

        """
        if _is_tensor_collection(type(other)):
            keys, vals = self._items_list(True, True)
            other_val = other._values_list(True, True, sorting_keys=keys)
        else:
            vals = self._values_list(True, True)
            other_val = other
        if alpha is not None:
            torch._foreach_sub_(vals, other_val, alpha=alpha)
        else:
            torch._foreach_sub_(vals, other_val)
        return self

    @_maybe_broadcast_other("rsub")
    def rsub(
        self,
        other: TensorDictBase | torch.Tensor | float,
        *,
        alpha: float | None = None,
        default: str | CompatibleType | None = None,
    ) -> Self:  # noqa: D417
        r"""Subtracts `self` from :attr:`other`, scaled by :attr:`alpha`, from ``self``.

        .. math::
            \text{{out}}_i = \text{{input}}_i - \text{{alpha}} \times \text{{other}}_i

        Supports broadcasting,
        type promotion, and integer, float, and complex inputs.

        Args:
            other (TensorDict, Tensor or Number): the tensor or number to subtract from ``self``.

        Keyword Args:
            alpha (Number): the multiplier for :attr:`other`.
            default (torch.Tensor or str, optional): the default value to use for exclusive entries.
                If none is provided, the two tensordicts key list must match exactly.
                If ``default="intersection"`` is passed, only the intersecting key sets will be considered
                and other keys will be ignored.
                In all other cases, ``default`` will be used for all missing entries on both sides of the
                operation.

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
        else:
            other_val = other
        if alpha is not None:
            vals = torch._foreach_neg(torch._foreach_sub(vals, other_val, alpha=alpha))
        else:
            vals = torch._foreach_neg(torch._foreach_sub(vals, other_val))
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
        """Computes the element-wise modulo of ``self`` and :attr:`other`.

        Args:
            other (TensorDict or Tensor): the other input tensordict or tensor.

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

    @_maybe_broadcast_other("mul_")
    def mul_(self, other: TensorCollection | torch.Tensor) -> Self:
        """In-place version of :meth:`~.mul`.

        .. note::
            Inplace ``mul`` does not support ``default`` keyword argument.

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
        torch._foreach_mul_(vals, other_val)
        return self

    @_maybe_broadcast_other("mul")
    def mul(
        self,
        other: TensorCollection | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> Self:  # noqa: D417
        r"""Multiplies :attr:`other` to ``self``.

        .. math::
            \text{{out}}_i = \text{{input}}_i \times \text{{other}}_i

        Supports broadcasting, type promotion, and integer, float, and complex inputs.

        Args:
            other (TensorDict, Tensor or Number): the tensor or number to subtract from ``self``.

        Keyword Args:
            default (torch.Tensor or str, optional): the default value to use for exclusive entries.
                If none is provided, the two tensordicts key list must match exactly.
                If ``default="intersection"`` is passed, only the intersecting key sets will be considered
                and other keys will be ignored.
                In all other cases, ``default`` will be used for all missing entries on both sides of the
                operation.

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
        vals = torch._foreach_mul(vals, other_val)
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

    def maximum_(self, other: TensorCollection | torch.Tensor) -> Self:
        """In-place version of :meth:`~.maximum`.

        .. note::
            Inplace ``maximum`` does not support ``default`` keyword argument.

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
                self.apply_(lambda x, y: x.maximum_(y), other)
            else:
                self.apply_(lambda x: x.maximum_(other))
            return self
        torch._foreach_maximum_(vals, other_val)
        return self

    @_maybe_broadcast_other("maximum")
    def maximum(
        self,
        other: TensorCollection | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> Self:  # noqa: D417
        """Computes the element-wise maximum of ``self`` and :attr:`other`.

        Args:
            other (TensorDict or Tensor): the other input tensordict or tensor.

        Keyword Args:
            default (torch.Tensor or str, optional): the default value to use for exclusive entries.
                If none is provided, the two tensordicts key list must match exactly.
                If ``default="intersection"`` is passed, only the intersecting key sets will be considered
                and other keys will be ignored.
                In all other cases, ``default`` will be used for all missing entries on both sides of the
                operation.

        """
        keys, vals = self._items_list(True, True)
        # Empty tensordict (no tensors) - return a copy unchanged
        if not vals:
            return self.copy()
        if any(_is_unbatched(v) for v in vals):
            if _is_tensor_collection(type(other)):
                return self.apply(lambda x, y: torch.maximum(x, y), other)
            else:
                return self.apply(lambda x: torch.maximum(x, other))
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
        vals = torch._foreach_maximum(vals, other_val)
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

    def minimum_(self, other: TensorCollection | torch.Tensor) -> Self:
        """In-place version of :meth:`~.minimum`.

        .. note::
            Inplace ``minimum`` does not support ``default`` keyword argument.

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
                self.apply_(lambda x, y: x.minimum_(y), other)
            else:
                self.apply_(lambda x: x.minimum_(other))
            return self
        torch._foreach_minimum_(vals, other_val)
        return self

    @_maybe_broadcast_other("minimum")
    def minimum(
        self,
        other: TensorCollection | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> Self:  # noqa: D417
        """Computes the element-wise minimum of ``self`` and :attr:`other`.

        Args:
            other (TensorDict or Tensor): the other input tensordict or tensor.

        Keyword Args:
            default (torch.Tensor or str, optional): the default value to use for exclusive entries.
                If none is provided, the two tensordicts key list must match exactly.
                If ``default="intersection"`` is passed, only the intersecting key sets will be considered
                and other keys will be ignored.
                In all other cases, ``default`` will be used for all missing entries on both sides of the
                operation.

        """
        keys, vals = self._items_list(True, True)
        # Empty tensordict (no tensors) - return a copy unchanged
        if not vals:
            return self.copy()
        if any(_is_unbatched(v) for v in vals):
            if _is_tensor_collection(type(other)):
                return self.apply(lambda x, y: torch.minimum(x, y), other)
            else:
                return self.apply(lambda x: torch.minimum(x, other))
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
        vals = torch._foreach_minimum(vals, other_val)
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

    def clamp_max_(self, other: TensorCollection | torch.Tensor) -> Self:
        """In-place version of :meth:`~.clamp_max`.

        .. note::
            Inplace ``clamp_max`` does not support ``default`` keyword argument.

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
                self.apply_(lambda x, y: x.clamp_max_(y), other)
            else:
                self.apply_(lambda x: x.clamp_max_(other))
            return self
        try:
            torch._foreach_clamp_max_(vals, other_val)
        except RuntimeError as err:
            if "isDifferentiableType" in str(err):
                raise RuntimeError(
                    "Attempted to execute _foreach_clamp_max_ with a differentiable tensor. "
                    "Use `td.apply(lambda x: x.clamp_max_(val)` instead."
                )
        return self

    @_maybe_broadcast_other("clamp_max")
    def clamp_max(
        self,
        other: TensorDictBase | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> Self:  # noqa: D417
        """Clamps the elements of ``self`` to :attr:`other` if they're superior to that value.

        Args:
            other (TensorDict or Tensor): the other input tensordict or tensor.

        Keyword Args:
            default (torch.Tensor or str, optional): the default value to use for exclusive entries.
                If none is provided, the two tensordicts key list must match exactly.
                If ``default="intersection"`` is passed, only the intersecting key sets will be considered
                and other keys will be ignored.
                In all other cases, ``default`` will be used for all missing entries on both sides of the
                operation.

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
            vals = torch._foreach_clamp_max(vals, other_val)
        except RuntimeError as err:
            if "isDifferentiableType" in str(err):
                raise RuntimeError(
                    "Attempted to execute _foreach_clamp_max with a differentiable tensor. "
                    "Use `td.apply(lambda x: x.clamp_max(val)` instead."
                )
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

    def clamp_min_(self, other: TensorDictBase | torch.Tensor) -> Self:
        """In-place version of :meth:`~.clamp_min`.

        .. note::
            Inplace ``clamp_min`` does not support ``default`` keyword argument.

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
                self.apply_(lambda x, y: x.clamp_min_(y), other)
            else:
                self.apply_(lambda x: x.clamp_min_(other))
            return self
        try:
            torch._foreach_clamp_min_(vals, other_val)
        except RuntimeError as err:
            if "isDifferentiableType" in str(err):
                raise RuntimeError(
                    "Attempted to execute _foreach_clamp_min_ with a differentiable tensor. "
                    "Use `td.apply(lambda x: x.clamp_min_(val)` instead."
                )

        return self

    @_maybe_broadcast_other("clamp_min")
    def clamp_min(
        self,
        other: TensorDictBase | torch.Tensor,
        default: str | CompatibleType | None = None,
    ) -> Self:  # noqa: D417
        """Clamps the elements of ``self`` to :attr:`other` if they're inferior to that value.

        Args:
            other (TensorDict or Tensor): the other input tensordict or tensor.

        Keyword Args:
            default (torch.Tensor or str, optional): the default value to use for exclusive entries.
                If none is provided, the two tensordicts key list must match exactly.
                If ``default="intersection"`` is passed, only the intersecting key sets will be considered
                and other keys will be ignored.
                In all other cases, ``default`` will be used for all missing entries on both sides of the
                operation.

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
            vals = torch._foreach_clamp_min(vals, other_val)
        except RuntimeError as err:
            if "isDifferentiableType" in str(err):
                raise RuntimeError(
                    "Attempted to execute _foreach_clamp_min with a differentiable tensor. "
                    "Use `td.apply(lambda x: x.clamp_min(val)` instead."
                )

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

    @_maybe_broadcast_other("clamp", 2)
    def clamp(
        self,
        min: TensorDictBase | torch.Tensor | float | None = None,
        max: TensorDictBase | torch.Tensor | float | None = None,
        *,
        out=None,
    ) -> Self:  # noqa: D417, W605
        r"""Clamps all elements in :attr:`self` into the range `[` :attr:`min`, :attr:`max` `]`.

        Letting min_value and max_value be :attr:`min` and :attr:`max`, respectively, this returns:

            .. math::
                y_i = \min(\max(x_i, \text{min\_value}_i), \text{max\_value}_i)

            If :attr:`min` is ``None``, there is no lower bound.
            Or, if :attr:`max` is ``None`` there is no upper bound.

        .. note::
            If :attr:`min` is greater than :attr:`max` :func:`torch.clamp(..., min, max) <torch.clamp>`
            sets all elements in :attr:`input` to the value of :attr:`max`.

        """
        if min is None:
            if out is not None:
                raise ValueError(
                    "clamp() with min/max=None isn't implemented with specified output."
                )
            return self.clamp_max(max)
        if max is None:
            if out is not None:
                raise ValueError(
                    "clamp() with min/max=None isn't implemented with specified output."
                )
            return self.clamp_min(min)

        is_tc_min = is_tensor_collection(min)
        is_tc_max = is_tensor_collection(max)

        if is_tc_min ^ is_tc_max:
            raise ValueError(
                "Mixed tensordict and non-tensordict min/max values are not authorized."
            )

        if out is None:
            if is_tc_min and is_tc_max:
                return self._fast_apply(
                    lambda x, low, high: x.clamp(low, high), min, max, default=None
                )
            return self._fast_apply(lambda x: x.clamp(min, max))
        if is_tc_min and is_tc_max:
            result = self._fast_apply(
                lambda x, y, low, high: x.clamp(low, high, out=y),
                out,
                min,
                max,
                default=None,
            )
        else:
            result = self._fast_apply(
                lambda x, y: x.clamp(min, max, out=y), out, default=None
            )
        with out.unlock_() if out.is_locked else contextlib.nullcontext():
            return out.update(result)

    @_maybe_broadcast_other("pow_")
    def pow_(self, other: TensorDictBase | torch.Tensor) -> Self:
        """In-place version of :meth:`~.pow`.

        .. note::
            Inplace ``pow`` does not support ``default`` keyword argument.

        """
        if _is_tensor_collection(type(other)):
            keys, vals = self._items_list(True, True)
            other_val = other._values_list(True, True, sorting_keys=keys)
        else:
            vals = self._values_list(True, True)
            other_val = other
        torch._foreach_pow_(vals, other_val)
        return self

    @_maybe_broadcast_other("pow")
    def pow(
        self,
        other: TensorDictBase | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> Self:  # noqa: D417
        r"""Takes the power of each element in ``self`` with :attr:`other` and returns a tensor with the result.

        :attr:`other` can be either a single ``float`` number, a `Tensor` or a ``TensorDict``.

        When :attr:`other` is a tensor, the shapes of :attr:`input`
        and :attr:`other` must be broadcastable.

        Args:
            other (float, tensor or tensordict): the exponent value

        Keyword Args:
            default (torch.Tensor or str, optional): the default value to use for exclusive entries.
                If none is provided, the two tensordicts key list must match exactly.
                If ``default="intersection"`` is passed, only the intersecting key sets will be considered
                and other keys will be ignored.
                In all other cases, ``default`` will be used for all missing entries on both sides of the
                operation.

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
        else:
            other_val = other
        vals = torch._foreach_pow(vals, other_val)
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

    @_maybe_broadcast_other("div_")
    def div_(self, other: TensorDictBase | torch.Tensor) -> Self:
        """In-place version of :meth:`~.div`.

        .. note::
            Inplace ``div`` does not support ``default`` keyword argument.

        """
        if _is_tensor_collection(type(other)):
            keys, vals = self._items_list(True, True)
            other_val = other._values_list(True, True, sorting_keys=keys)
        else:
            vals = self._values_list(True, True)
            other_val = other
        torch._foreach_div_(vals, other_val)
        return self

    @_maybe_broadcast_other("div")
    def div(
        self,
        other: TensorDictBase | torch.Tensor,
        *,
        default: str | CompatibleType | None = None,
    ) -> Self:  # noqa: D417
        r"""Divides each element of the input ``self`` by the corresponding element of :attr:`other`.

        .. math::
            \text{out}_i = \frac{\text{input}_i}{\text{other}_i}

        Supports broadcasting, type promotion and integer, float, tensordict or tensor inputs.
        Always promotes integer types to the default scalar type.

        Args:
            other (TensorDict, Tensor or Number): the divisor.

        Keyword Args:
            default (torch.Tensor or str, optional): the default value to use for exclusive entries.
                If none is provided, the two tensordicts key list must match exactly.
                If ``default="intersection"`` is passed, only the intersecting key sets will be considered
                and other keys will be ignored.
                In all other cases, ``default`` will be used for all missing entries on both sides of the
                operation.

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
        else:
            other_val = other
        vals = torch._foreach_div(vals, other_val)
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

    def sqrt_(self) -> Self:
        """In-place version of :meth:`~.sqrt`."""
        torch._foreach_sqrt_(self._values_list(True, True))
        return self

    def sqrt(self) -> Self:
        """Computes the element-wise square root of ``self``."""
        keys, vals = self._items_list(True, True)
        vals = torch._foreach_sqrt(vals)
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
        """Zeros all tensors in the tensordict in-place."""

        def fn(item):
            item.zero_()

        self._fast_apply(fn=fn, call_on_nested=True, propagate_lock=True)
        return self

    def fill_(self, key: NestedKey, value: float | bool) -> Self:
        """Fills a tensor pointed by the key with a given scalar value.

        Args:
            key (str or nested key): entry to be filled.
            value (Number or bool): value to use for the filling.

        Returns:
            self

        """
        key = _unravel_key_to_tuple(key)
        data = self._get_tuple(key, NO_DEFAULT)
        if _is_tensor_collection(type(data)):

            def fill(x):
                return x.fill_(value)

            data._fast_apply(fill, inplace=True)
        else:
            data = data.fill_(value)
            self._set_tuple(key, data, inplace=True, validated=True, non_blocking=False)
        return self

    def where(
        self,
        condition: Tensor,
        other: Tensor | TensorDictBase,
        *,
        out: TensorDictBase | None = None,
        pad: int | bool = None,
        update_batch_size: bool = False,
    ) -> Self:  # noqa: D417
        """Return a ``TensorDict`` of elements selected from either self or other, depending on condition.

        Args:
            condition (BoolTensor): When ``True`` (nonzero), yields ``self``,
                otherwise yields ``other``.
            other (TensorDictBase or Scalar): value (if ``other`` is a scalar)
                or values selected at indices where condition is ``False``.

        Keyword Args:
            out (TensorDictBase, optional): the output ``TensorDictBase`` instance.
            pad (scalar, optional): if provided, missing keys from the source
                or destination tensordict will be written as `torch.where(mask, self, pad)`
                or `torch.where(mask, pad, other)`. Defaults to ``None``, ie
                missing keys are not tolerated.
            update_batch_size (bool, optional): if ``True`` and ``out`` is provided, the batch size of the output will be
                updated to match the batch size of the condition. Defaults to ``False``.

        """
        ...

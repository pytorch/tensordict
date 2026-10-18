# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc


class _NestedKeyMeta(abc.ABCMeta):
    def __instancecheck__(self, instance):
        return isinstance(instance, str) or (
            isinstance(instance, tuple)
            and len(instance)
            and all(isinstance(subkey, NestedKey) for subkey in instance)
        )


class NestedKey(metaclass=_NestedKeyMeta):
    """The type of the keys of a tensordict: a string, or a tuple of strings for a nested entry.

    A nested key is either a string or a non-empty tuple of nested keys.
    ``td["a", "b"]`` and ``td[("a", "b")]`` read the entry ``"b"`` of the
    nested tensordict ``"a"``. Nested tuples are flattened: ``("a", ("b",))``
    is the same key as ``("a", "b")``. See :ref:`keys`.

    Use :class:`NestedKey` in type hints and in ``isinstance`` checks.

    Examples:
        >>> from tensordict import NestedKey, TensorDict
        >>> isinstance("a", NestedKey), isinstance(("a", ("b",)), NestedKey)
        (True, True)
        >>> isinstance(("a", 1), NestedKey), isinstance((), NestedKey)
        (False, False)
        >>> td = TensorDict({"a": {"b": 0}})
        >>> td["a", "b"] is td[("a", ("b",))]
        True

    """

    pass

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
    """An abstract class for nested keys.

    Nested keys are the generic key type accepted by TensorDict.

    A nested key is either a string or a non-empty tuple of NestedKeys instances.

    The NestedKey class supports instance checks.

    """

    pass

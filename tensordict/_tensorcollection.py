# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TypeVar

T = TypeVar("T", bound="TensorCollection")


class TensorCollection:
    """A base class for TensorDictBase and TensorClass.

    This is an abstract base class that provides the foundation for tensor collections
    like TensorDict and TensorClass. It serves as a common interface for all tensor
    collection types in the tensordict library.

    Use it in type hints and ``isinstance`` checks for "a tensordict or a
    tensorclass". Classes decorated with ``@tensorclass`` do not inherit from
    it, only subclasses of :class:`~tensordict.TensorClass` do; to check both,
    use :func:`~tensordict.is_tensor_collection`.

    Examples:
        >>> import torch
        >>> from tensordict import TensorClass, TensorCollection, TensorDict
        >>> class Data(TensorClass):
        ...     x: torch.Tensor
        >>> td = TensorDict(x=torch.zeros(3), batch_size=[3])
        >>> isinstance(td, TensorCollection)
        True
        >>> isinstance(Data(x=torch.zeros(3), batch_size=[3]), TensorCollection)
        True
        >>> isinstance(torch.zeros(3), TensorCollection)
        False
    """

    def __init__(self, *args, **kwargs):
        """Initialize the TensorCollection.

        This is an abstract base class and should not be instantiated directly.
        """
        if not (args or kwargs) and "_tensordict" in self.__dict__:
            # super().__init__() from the __init__ of a TensorClass subclass,
            # once the tensorclass constructor has built the instance.
            return
        raise NotImplementedError(
            "TensorCollection is an abstract base class and cannot be instantiated directly."
        )

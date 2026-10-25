# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class TensorCollection:
    """The abstract base class of tensordicts and tensorclasses.

    :class:`~tensordict.TensorDictBase` and :class:`~tensordict.TensorClass`
    inherit from it. It cannot be instantiated.

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
        """Raises a ``NotImplementedError``: the class cannot be instantiated."""
        if not (args or kwargs) and "_tensordict" in self.__dict__:
            # super().__init__() from the __init__ of a TensorClass subclass,
            # once the tensorclass constructor has built the instance.
            return
        raise NotImplementedError(
            "TensorCollection is an abstract base class and cannot be instantiated directly."
        )

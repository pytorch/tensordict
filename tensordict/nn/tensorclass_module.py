from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import cast, Generic, get_args, get_origin, TypeVar, Union

from tensordict._td import TensorDict
from tensordict.nn.common import dispatch, TensorDictModuleBase
from tensordict.tensorclass import TensorClass
from torch import nn, Tensor

__all__ = ["TensorClassModuleBase", "TensorClassModuleWrapper"]


def _tensor_class_keys(tensorclass_type: type[TensorClass]) -> list[tuple[str, ...]]:
    """Extract all keys from a TensorClass type, including nested keys.

    Args:
        tensorclass_type (type[TensorClass]): The TensorClass type to extract keys from.

    Returns:
        list[tuple[str, ...]]: A list of key tuples representing all fields in the TensorClass.

    """
    fields = dataclasses.fields(tensorclass_type)
    # field.type is a string under postponed annotations: use the resolved hints
    type_hints = getattr(tensorclass_type, "_type_hints", None) or {}
    keys: list[tuple[str, ...]] = []
    for field in fields:
        key = field.name
        field_type = type_hints.get(key, field.type)
        if isinstance(field_type, type) and issubclass(field_type, TensorClass):
            subkeys = _tensor_class_keys(cast(type[TensorClass], field_type))
            for subkey in subkeys:
                keys.append((key,) + subkey)
        else:
            keys.append((key,))
    return keys


InputTensorClass = TypeVar("InputTensorClass", bound=TensorClass)
OutputTensorClass = TypeVar("OutputTensorClass", bound=TensorClass)


class TensorClassModuleWrapper(TensorDictModuleBase):
    """Wraps a :class:`TensorClassModuleBase` so that it takes and returns a tensordict.

    The wrapper converts the input tensordict to the input class of the
    module, calls the module, and converts the output tensorclass back to a
    TensorDict. Its ``in_keys`` and ``out_keys`` are the fields of the input
    and output classes, with nested keys for nested tensorclasses.
    :meth:`TensorClassModuleBase.as_td_module` returns one.

    The wrapper returns a new TensorDict that holds only the output fields. It
    does not write into the input. In a
    :class:`~tensordict.nn.TensorDictSequential`, the modules after it cannot
    read the input keys, and by default the sequence returns only the
    outputs.

    Args:
        module (TensorClassModuleBase): the module to wrap. Its input and
            output types must be tensorclasses.

    Examples:
        >>> from tensordict import TensorDict
        >>> from tensordict.tensorclass import TensorClass
        >>> from tensordict.nn import TensorClassModuleBase
        >>> import torch
        >>>
        >>> class InputTC(TensorClass):
        ...     x: torch.Tensor
        ...
        >>> class OutputTC(TensorClass):
        ...     y: torch.Tensor
        ...
        >>> class MyModule(TensorClassModuleBase[InputTC, OutputTC]):
        ...     def forward(self, input: InputTC) -> OutputTC:
        ...         return OutputTC(y=input.x + 1, batch_size=input.batch_size)
        ...
        >>> module = MyModule()
        >>> td_module = module.as_td_module()
        >>> td = TensorDict({"x": torch.zeros(3)}, batch_size=[3])
        >>> result = td_module(td)
        >>> list(result.keys())
        ['y']
        >>> "y" in td
        False

    """

    def __init__(
        self, module: TensorClassModuleBase[InputTensorClass, OutputTensorClass]
    ) -> None:
        super().__init__()
        # Start in the module's mode. Set the flag only: ``train()`` would also
        # reset the mode of every submodule of ``module``.
        self.training = module.training
        self.tc_module = module
        self.in_keys = _tensor_class_keys(cast(type[TensorClass], module.input_type))
        self.out_keys = _tensor_class_keys(cast(type[TensorClass], module.output_type))

    @dispatch(auto_batch_size=False)
    def forward(self, tensordict: TensorDict, *args, **kwargs) -> TensorDict:
        """Converts ``tensordict`` to the input class, calls the module, and returns its output as a TensorDict.

        Args:
            tensordict (TensorDictBase): the input tensordict. Its entries that
                are not fields of the input class are ignored.
            *args: ignored.
            **kwargs: ignored.

        Returns:
            A new TensorDict with the fields of the output tensorclass.

        """
        input_type = self.tc_module.input_type
        # from_tensordict rejects the keys that are not fields of the input class
        extra_keys = [
            key for key in tensordict.keys() if key not in input_type.__expected_keys__
        ]
        if extra_keys:
            tensordict = tensordict.exclude(*extra_keys)
        return self.tc_module(input_type.from_tensordict(tensordict)).to_tensordict()


InputClass = TypeVar("InputClass", bound=Union[TensorClass, Tensor])
OutputClass = TypeVar("OutputClass", bound=Union[TensorClass, Tensor])


class TensorClassModuleBase(Generic[InputClass, OutputClass], ABC, nn.Module):
    """Base class of the modules that take and return a tensorclass.

    Subclass ``TensorClassModuleBase[InputClass, OutputClass]``, where each
    type is a :class:`~tensordict.TensorClass` subclass or
    :class:`torch.Tensor`, and implement :meth:`forward`. A type checker
    then knows the type of the input and output, and can report a wrong
    field name. The class reads both types from the subscript: a subclass
    without it has no ``input_type`` and ``output_type``.

    When both types are tensorclasses, :meth:`as_td_module` returns a
    module that takes and returns a tensordict, for example to put the
    module in a :class:`~tensordict.nn.TensorDictSequential`. See
    :ref:`tensorclass-guide`.

    Attributes:
        input_type (type): the input type, ``InputClass``.
        output_type (type): the output type, ``OutputClass``.

    Examples:
        >>> from tensordict.tensorclass import TensorClass
        >>> from tensordict.nn import TensorClassModuleBase
        >>> import torch
        >>>
        >>> class InputTC(TensorClass):
        ...     a: torch.Tensor
        ...     b: torch.Tensor
        ...
        >>> class OutputTC(TensorClass):
        ...     result: torch.Tensor
        ...
        >>> class AddModule(TensorClassModuleBase[InputTC, OutputTC]):
        ...     def forward(self, x: InputTC) -> OutputTC:
        ...         return OutputTC(
        ...             result=x.a + x.b,
        ...             batch_size=x.batch_size
        ...         )
        ...
        >>> module = AddModule()
        >>> input_tc = InputTC(a=torch.tensor([1.0]), b=torch.tensor([2.0]), batch_size=[1])
        >>> module(input_tc).result
        tensor([3.])

    """

    input_type: type[InputClass]
    output_type: type[OutputClass]

    def __init_subclass__(cls) -> None:
        """Reads ``input_type`` and ``output_type`` from the subscript of the base class."""
        super().__init_subclass__()
        for base in cls.__orig_bases__:  # type:ignore[attr-defined]
            origin = get_origin(base)
            if origin is TensorClassModuleBase:
                generic_args = get_args(base)
                if generic_args:
                    cls.input_type, cls.output_type = generic_args
                else:
                    raise ValueError(
                        "Generic input/output types not set in TensorClassModuleBase"
                    )

    @abstractmethod
    def forward(self, x: InputClass) -> OutputClass:
        """Computes the output of the module. Subclasses implement it.

        Args:
            x (InputClass): the input.

        Returns:
            An instance of ``OutputClass``.

        """
        ...

    def __call__(self, x: InputClass) -> OutputClass:
        """Calls the module on ``x``, with the hooks of :class:`torch.nn.Module`, and returns its output."""
        return cast("OutputClass", super().__call__(x))

    def as_td_module(self) -> TensorClassModuleWrapper:
        """Returns a :class:`TensorClassModuleWrapper` of the module, which takes and returns a tensordict.

        The wrapper is a :class:`~tensordict.nn.TensorDictModuleBase`. Its
        ``in_keys`` and ``out_keys`` are the fields of the input and output
        classes. It returns a new TensorDict with only the output fields (see
        :class:`TensorClassModuleWrapper`).

        Raises:
            ValueError: if ``input_type`` or ``output_type`` is not a
                tensorclass.

        Examples:
            >>> from tensordict import TensorDict
            >>> from tensordict.tensorclass import TensorClass
            >>> from tensordict.nn import TensorClassModuleBase
            >>> import torch
            >>>
            >>> class InputTC(TensorClass):
            ...     x: torch.Tensor
            ...
            >>> class OutputTC(TensorClass):
            ...     y: torch.Tensor
            ...
            >>> class MyModule(TensorClassModuleBase[InputTC, OutputTC]):
            ...     def forward(self, input: InputTC) -> OutputTC:
            ...         return OutputTC(y=input.x * 2, batch_size=input.batch_size)
            ...
            >>> module = MyModule()
            >>> td_module = module.as_td_module()
            >>> td = TensorDict({"x": torch.ones(3)}, batch_size=[3])
            >>> td_module(td)["y"]
            tensor([2., 2., 2.])

        """
        if not (
            issubclass(self.input_type, TensorClass)
            and issubclass(self.output_type, TensorClass)
        ):
            raise ValueError(
                "Only TensorClassModuleBase implementations with both input and "
                "output type as TensorClass can be converted to TensorDictModule"
            )
        return TensorClassModuleWrapper(self)  # type:ignore[arg-type,type-var]

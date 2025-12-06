from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import Field
from typing import Any, cast, Generic, get_args, get_origin, TypeVar, Union

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
    fields = cast("Iterable[Field[Any]]", tensorclass_type.fields())
    keys: list[tuple[str, ...]] = []
    for field in fields:
        key = field.name
        if issubclass(field.type, TensorClass):
            subkeys = _tensor_class_keys(cast(type[TensorClass], field.type))
            for subkey in subkeys:
                keys.append((key,) + subkey)
        else:
            keys.append((key,))
    return keys


InputTensorClass = TypeVar("InputTensorClass", bound=TensorClass)
OutputTensorClass = TypeVar("OutputTensorClass", bound=TensorClass)


class TensorClassModuleWrapper(TensorDictModuleBase):
    """Wrapper class for TensorClassModuleBase objects.

    This wrapper allows TensorClassModuleBase instances to be used in TensorDict-based
    workflows by handling the conversion between TensorDict and TensorClass representations.
    When called with a TensorDict, the wrapper converts it to a TensorClass, passes it through
    the wrapped module, and converts the output back to a TensorDict.

    Args:
        module (TensorClassModuleBase): The TensorClassModuleBase instance to wrap.

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
        >>> assert "y" in result

    """

    def __init__(
        self, module: TensorClassModuleBase[InputTensorClass, OutputTensorClass]
    ) -> None:
        super().__init__()
        self.tc_module = module
        self.in_keys = _tensor_class_keys(cast(type[TensorClass], module.input_type))
        self.out_keys = _tensor_class_keys(cast(type[TensorClass], module.output_type))

    @dispatch(auto_batch_size=False)
    def forward(self, tensordict: TensorDict, *args, **kwargs) -> TensorDict:
        """Forward pass converting TensorDict to TensorClass and back.

        Args:
            tensordict (TensorDict): Input tensordict.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorDict: Output tensordict.

        """
        return self.tc_module(
            self.tc_module.input_type.from_tensordict(tensordict)
        ).to_tensordict()


InputClass = TypeVar("InputClass", bound=Union[TensorClass, Tensor])
OutputClass = TypeVar("OutputClass", bound=Union[TensorClass, Tensor])


class TensorClassModuleBase(Generic[InputClass, OutputClass], ABC, nn.Module):
    """A TensorClassModuleBase is a base class for modules that operate on TensorClass instances.

    TensorClassModuleBase subclasses provide a type-safe way to define modules that work with TensorClass
    inputs and outputs. The class automatically extracts input and output type information from the
    generic type parameters.

    The module can be converted to a TensorDictModule using the :meth:`as_td_module`
    method, allowing it to be used in TensorDict-based workflows.

    Type Parameters:
        InputClass: The input type, must be a TensorClass or Tensor.
        OutputClass: The output type, must be a TensorClass or Tensor.

    Attributes:
        input_type (type[InputClass]): The input type class.
        output_type (type[OutputClass]): The output type class.

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
        >>> output = module(input_tc)
        >>> assert output.result == torch.tensor([3.0])

    """

    input_type: type[InputClass]
    output_type: type[OutputClass]

    def __init_subclass__(cls) -> None:
        """Initialize subclass by extracting type information from generic parameters."""
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
        """Forward pass of the module.

        Args:
            x (InputClass): Input instance.

        Returns:
            OutputClass: Output instance.

        """
        ...

    def __call__(self, x: InputClass) -> OutputClass:
        """Call the module's forward method.

        Args:
            x (InputClass): Input instance.

        Returns:
            OutputClass: Output instance.

        """
        return cast("OutputClass", super().__call__(x))

    def as_td_module(self) -> TensorClassModuleWrapper:
        """Convert this module to a TensorDictModule.

        This method wraps the TensorClassModuleBase in a TensorClassModuleWrapper,
        allowing it to be used with TensorDict inputs and outputs.

        Returns:
            TensorClassModuleWrapper: A wrapper that converts between TensorDict
                and TensorClass representations.

        Raises:
            ValueError: If either input_type or output_type is not a TensorClass.

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
            >>> result = td_module(td)
            >>> assert (result["y"] == 2).all()

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

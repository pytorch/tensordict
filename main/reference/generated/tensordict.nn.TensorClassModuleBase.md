# TensorClassModuleBase

*class*tensordict.nn.TensorClassModuleBase(**args: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*, ***kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*)

Bases: [`Generic`](https://docs.python.org/3/library/typing.html#typing.Generic)[`InputClass`, `OutputClass`], [`ABC`](https://docs.python.org/3/library/abc.html#abc.ABC), [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

A TensorClassModuleBase is a base class for modules that operate on TensorClass instances.

TensorClassModuleBase subclasses provide a type-safe way to define modules that work with TensorClass
inputs and outputs. The class automatically extracts input and output type information from the
generic type parameters.

The module can be converted to a TensorDictModule using the `as_td_module()`
method, allowing it to be used in TensorDict-based workflows.

Type Parameters:

InputClass: The input type, must be a TensorClass or Tensor.
OutputClass: The output type, must be a TensorClass or Tensor.

Variables:

- **input_type** ([*type*](https://docs.python.org/3/builtins/functions.html#type)*[**InputClass**]*) - The input type class.
- **output_type** ([*type*](https://docs.python.org/3/builtins/functions.html#type)*[**OutputClass**]*) - The output type class.

Examples

```
>>> from tensordict.tensorclass import TensorClass
>>> from tensordict.nn import TensorClassModuleBase
>>> import torch
>>>
>>> class InputTC(TensorClass):
... a: torch.Tensor
... b: torch.Tensor
...
>>> class OutputTC(TensorClass):
... result: torch.Tensor
...
>>> class AddModule(TensorClassModuleBase[InputTC, OutputTC]):
... def forward(self, x: InputTC) -> OutputTC:
... return OutputTC(
... result=x.a + x.b,
... batch_size=x.batch_size
... )
...
>>> module = AddModule()
>>> input_tc = InputTC(a=torch.tensor([1.0]), b=torch.tensor([2.0]), batch_size=[1])
>>> output = module(input_tc)
>>> assert output.result == torch.tensor([3.0])
```

as_td_module() → [TensorClassModuleWrapper](tensordict.nn.TensorClassModuleWrapper.html#tensordict.nn.TensorClassModuleWrapper)

Convert this module to a module that takes and returns a TensorDict.

This method wraps the TensorClassModuleBase in a
[`TensorClassModuleWrapper`](tensordict.nn.TensorClassModuleWrapper.html#tensordict.nn.TensorClassModuleWrapper), a
[`TensorDictModuleBase`](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) subclass. Its `in_keys`
and `out_keys` are the fields of the input and output classes. It
returns a new TensorDict with only the output fields (see
[`TensorClassModuleWrapper`](tensordict.nn.TensorClassModuleWrapper.html#tensordict.nn.TensorClassModuleWrapper)).

Returns:

A wrapper that converts between TensorDict

and TensorClass representations.

Return type:

[*TensorClassModuleWrapper*](tensordict.nn.TensorClassModuleWrapper.html#tensordict.nn.TensorClassModuleWrapper)

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - If either input_type or output_type is not a TensorClass.

Examples

```
>>> from tensordict import TensorDict
>>> from tensordict.tensorclass import TensorClass
>>> from tensordict.nn import TensorClassModuleBase
>>> import torch
>>>
>>> class InputTC(TensorClass):
... x: torch.Tensor
...
>>> class OutputTC(TensorClass):
... y: torch.Tensor
...
>>> class MyModule(TensorClassModuleBase[InputTC, OutputTC]):
... def forward(self, input: InputTC) -> OutputTC:
... return OutputTC(y=input.x * 2, batch_size=input.batch_size)
...
>>> module = MyModule()
>>> td_module = module.as_td_module()
>>> td = TensorDict({"x": torch.ones(3)}, batch_size=[3])
>>> result = td_module(td)
>>> assert (result["y"] == 2).all()
```

*abstract*forward(*x: InputClass*) → OutputClass

Forward pass of the module.

Parameters:

**x** (*InputClass*) - Input instance.

Returns:

Output instance.

Return type:

OutputClass
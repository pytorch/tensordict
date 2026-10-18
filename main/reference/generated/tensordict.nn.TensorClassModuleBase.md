# TensorClassModuleBase

*class*tensordict.nn.TensorClassModuleBase(**args: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*, ***kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*)

Bases: [`Generic`](https://docs.python.org/3/library/typing.html#typing.Generic)[`InputClass`, `OutputClass`], [`ABC`](https://docs.python.org/3/library/abc.html#abc.ABC), [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Base class of the modules that take and return a tensorclass.

Subclass `TensorClassModuleBase[InputClass, OutputClass]`, where each
type is a [`TensorClass`](tensordict.TensorClass.html#tensordict.TensorClass) subclass or
[`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), and implement `forward()`. A type checker
then knows the type of the input and output, and can report a wrong
field name. The class reads both types from the subscript: a subclass
without it has no `input_type` and `output_type`.

When both types are tensorclasses, `as_td_module()` returns a
module that takes and returns a tensordict, for example to put the
module in a [`TensorDictSequential`](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential). See
[TensorClass](../../tensorclass.html#tensorclass-guide).

Variables:

- **input_type** ([*type*](https://docs.python.org/3/builtins/functions.html#type)) - the input type, `InputClass`.
- **output_type** ([*type*](https://docs.python.org/3/builtins/functions.html#type)) - the output type, `OutputClass`.

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
>>> module(input_tc).result
tensor([3.])
```

as_td_module() → [TensorClassModuleWrapper](tensordict.nn.TensorClassModuleWrapper.html#tensordict.nn.TensorClassModuleWrapper)

Returns a [`TensorClassModuleWrapper`](tensordict.nn.TensorClassModuleWrapper.html#tensordict.nn.TensorClassModuleWrapper) of the module, which takes and returns a tensordict.

The wrapper is a [`TensorDictModuleBase`](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase). Its
`in_keys` and `out_keys` are the fields of the input and output
classes. It returns a new TensorDict with only the output fields (see
[`TensorClassModuleWrapper`](tensordict.nn.TensorClassModuleWrapper.html#tensordict.nn.TensorClassModuleWrapper)).

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if `input_type` or `output_type` is not a
 tensorclass.

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
>>> td_module(td)["y"]
tensor([2., 2., 2.])
```

*abstract*forward(*x: InputClass*) → OutputClass

Computes the output of the module. Subclasses implement it.

Parameters:

**x** (*InputClass*) - the input.

Returns:

An instance of `OutputClass`.
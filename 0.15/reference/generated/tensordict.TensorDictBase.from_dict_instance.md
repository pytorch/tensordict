# TensorDictBase.from_dict_instance

TensorDictBase.from_dict_instance(*input_dict*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size=None*, *device=None*, *batch_dims=None*, *names=None*)

Instance method version of [`from_dict()`](tensordict.TensorDictBase.from_dict.html#tensordict.TensorDictBase.from_dict).

Unlike [`from_dict()`](tensordict.TensorDictBase.from_dict.html#tensordict.TensorDictBase.from_dict), this method will
attempt to keep the tensordict types within the existing tree (for
any existing leaf).

Examples

```
>>> from tensordict import TensorDict, tensorclass
>>> import torch
>>>
>>> @tensorclass
... class MyClass:
... x: torch.Tensor
... y: int
>>>
>>> td = TensorDict({"a": torch.randn(()), "b": MyClass(x=torch.zeros(()), y=1)})
>>> print(td.from_dict_instance(td.to_dict()))
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 b: MyClass(
 x=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 y=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> print(td.from_dict(td.to_dict()))
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 b: TensorDict(
 fields={
 x: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 y: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
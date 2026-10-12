# TensorDictBase.numpy

TensorDictBase.numpy() → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray) | [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Converts a tensordict to a (possibly nested) dictionary of numpy arrays.

Non-tensor data is exposed as such.

Examples

```
>>> from tensordict import TensorDict
>>> import torch
>>> data = TensorDict({"a": {"b": torch.zeros(()), "c": "a string!"}})
>>> print(data)
TensorDict(
 fields={
 a: TensorDict(
 fields={
 b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 c: NonTensorData(data=a string!, batch_size=torch.Size([]), device=None)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> print(data.numpy())
{'a': {'b': array(0., dtype=float32), 'c': 'a string!'}}
```

seealso: [`to_struct_array()`](tensordict.TensorDictBase.to_struct_array.html#tensordict.TensorDictBase.to_struct_array) to convert to a struct array.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
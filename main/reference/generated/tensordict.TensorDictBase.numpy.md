# TensorDictBase.numpy

TensorDictBase.numpy() → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray) | [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Returns a nested dict of NumPy arrays with the entries of the tensordict.

Each tensor becomes a NumPy array with [`torch.Tensor.numpy()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numpy.html#torch.Tensor.numpy),
so the arrays share memory with the tensors, and a tensor that
requires gradients or is not on the CPU raises a `RuntimeError` or
`TypeError`: call [`detach()`](tensordict.TensorDictBase.detach.html#tensordict.TensorDictBase.detach) or [`cpu()`](tensordict.TensorDictBase.cpu.html#tensordict.TensorDictBase.cpu) first. A nested
tensor becomes a tuple of arrays. Non-tensor data is returned as its
Python value. The `None` fields of tensorclasses are dropped.

See also

[`to_struct_array()`](tensordict.TensorDictBase.to_struct_array.html#tensordict.TensorDictBase.to_struct_array) to convert to a NumPy
structured array, and [`to_dict()`](tensordict.TensorDictBase.to_dict.html#tensordict.TensorDictBase.to_dict).

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> data = TensorDict({"a": {"b": torch.zeros(()), "c": "a string!"}})
>>> data.numpy()
{'a': {'b': array(0., dtype=float32), 'c': 'a string!'}}
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
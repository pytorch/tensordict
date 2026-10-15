# TensorDictBase.narrow

TensorDictBase.narrow(*dim: [int](https://docs.python.org/3/builtins/functions.html#int)*, *start: [int](https://docs.python.org/3/builtins/functions.html#int)*, *length: [int](https://docs.python.org/3/builtins/functions.html#int)*)

Returns a new tensordict that is a narrowed version of the input.

The dimension dim is input from start to start + length.

Parameters:

- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - The dimension along which to narrow.
- **start** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - Starting index.
- **length** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - Length of the narrowed dimension.

Returns:

a new tensordict narrowed along the specified dimension.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.arange(6).view(2, 3)}, batch_size=[2, 3])
>>> print(td["a"])
tensor([[0, 1, 2],
 [3, 4, 5]])
>>> print(td.narrow(1, 1, 2)["a"])
tensor([[1, 2],
 [4, 5]])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
# TensorDictBase.narrow

TensorDictBase.narrow(*dim: [int](https://docs.python.org/3/builtins/functions.html#int)*, *start: [int](https://docs.python.org/3/builtins/functions.html#int)*, *length: [int](https://docs.python.org/3/builtins/functions.html#int)*)

Returns a tensordict narrowed along a batch dimension, as [`torch.Tensor.narrow()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.narrow.html#torch.Tensor.narrow) does.

The result keeps the elements `start` to `start + length - 1` of
`dim`. Its leaves are views of the original leaves, and it keeps the
names.

Parameters:

- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the batch dimension to narrow.
- **start** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the first element to keep.
- **length** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the number of elements to keep.

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
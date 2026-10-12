# TensorDictBase.tile

TensorDictBase.tile(*dims: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), ...]*)

Construct a tensordict by repeating the elements.

The dims argument specifies the number of repetitions in each dimension.

Parameters:

**dims** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of**ints*) - The number of repetitions per dimension.

Returns:

a new tensordict with elements repeated.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.arange(6).view(2, 3)}, batch_size=[2, 3])
>>> print(td["a"])
tensor([[0, 1, 2],
 [3, 4, 5]])
>>> print(td.tile((2, 1))["a"])
tensor([[0, 1, 2],
 [3, 4, 5],
 [0, 1, 2],
 [3, 4, 5]])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
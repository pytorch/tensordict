# TensorDictBase.rot90

TensorDictBase.rot90(*k: [int](https://docs.python.org/3/builtins/functions.html#int) = 1*, *dims: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), [int](https://docs.python.org/3/builtins/functions.html#int)] = (0, 1)*)

Rotate the tensordict by 90 degrees in the plane specified by dims.

Rotation direction is from the first towards the second axis.

Parameters:

- **k** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - Number of times to rotate. Default: 1.
- **dims** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of**two ints*) - The plane to rotate in. Default: (0, 1).

Returns:

a new tensordict rotated by 90 degrees.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.arange(6).view(2, 3)}, batch_size=[2, 3])
>>> print(td["a"])
tensor([[0, 1, 2],
 [3, 4, 5]])
>>> print(td.rot90()["a"])
tensor([[2, 5],
 [1, 4],
 [0, 3]])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
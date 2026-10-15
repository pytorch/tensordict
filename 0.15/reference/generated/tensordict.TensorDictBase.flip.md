# TensorDictBase.flip

TensorDictBase.flip(*dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), ...]*)

Reverse the order of elements in the tensordict along the given dimensions.

The shape of the tensordict is preserved, but the elements are reordered.

Parameters:

**dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of**ints*) - Dimensions to flip.

Returns:

a new tensordict with the dimensions flipped.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.arange(6).view(2, 3)}, batch_size=[2, 3])
>>> print(td["a"])
tensor([[0, 1, 2],
 [3, 4, 5]])
>>> print(td.flip(0)["a"])
tensor([[3, 4, 5],
 [0, 1, 2]])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
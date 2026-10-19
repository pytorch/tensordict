# TensorDictBase.tile

TensorDictBase.tile(*dims: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), ...]*)

Returns a tensordict whose batch is repeated, as [`torch.tile()`](https://docs.pytorch.org/docs/stable/generated/torch.tile.html#torch.tile) does.

If `dims` has fewer entries than the batch dimensions, it is padded
with leading ones. If it has more, the result gets new leading batch
dimensions. The feature dimensions are not repeated, the result has no
names, and the data is copied. See also [`repeat()`](tensordict.TensorDictBase.repeat.html#tensordict.TensorDictBase.repeat).

Parameters:

**dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)) - the number of repetitions along each
batch dimension.

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
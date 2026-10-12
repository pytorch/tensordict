# TensorDictBase.broadcast_to

TensorDictBase.broadcast_to(*shape: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), ...]*)

Broadcasts the tensordict to a new shape.

The new shape must be compatible with the original shape.

Parameters:

**shape** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of**ints*) - The desired shape.

Returns:

a new tensordict with the shape broadcast.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.arange(3)}, batch_size=[3])
>>> print(td.broadcast_to((2, 3)).shape)
torch.Size([2, 3])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
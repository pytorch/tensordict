# TensorDictBase.broadcast_to

TensorDictBase.broadcast_to(*shape: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), ...]*)

Returns the tensordict broadcast to the batch size `shape`, as [`torch.broadcast_to()`](https://docs.pytorch.org/docs/stable/generated/torch.broadcast_to.html#torch.broadcast_to) does.

As for [`expand()`](tensordict.TensorDictBase.expand.html#tensordict.TensorDictBase.expand), the new shape can add leading dimensions and
expand the dimensions of size 1, and `-1` keeps the size of a
dimension. The leaves of the result are views of the original leaves,
and the result has no names.

Parameters:

**shape** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)) - the new batch size.

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
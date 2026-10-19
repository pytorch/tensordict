# TensorDictBase.atleast_2d

TensorDictBase.atleast_2d()

Returns the tensordict with at least 2 batch dimensions.

If the tensordict already has 2 or more batch dimensions, it is
returned as it is. Otherwise, dimensions of size 1 are prepended to
reach 2 dimensions. The leaves of the result are views of the original
leaves.

It can be used as a context manager, as [`unsqueeze()`](tensordict.TensorDictBase.unsqueeze.html#tensordict.TensorDictBase.unsqueeze) can.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.zeros(3)}, batch_size=[3])
>>> print(td.atleast_2d().shape)
torch.Size([1, 3])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
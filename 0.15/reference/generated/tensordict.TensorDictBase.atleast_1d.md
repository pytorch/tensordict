# TensorDictBase.atleast_1d

TensorDictBase.atleast_1d()

Returns the tensordict with at least 1 batch dimension.

If the tensordict already has 1 or more batch dimensions, it is
returned as it is. Otherwise, a dimension of size 1 is prepended. The
leaves of the result are views of the original leaves.

It can be used as a context manager, as [`unsqueeze()`](tensordict.TensorDictBase.unsqueeze.html#tensordict.TensorDictBase.unsqueeze) can.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.zeros(3)}, batch_size=[])
>>> print(td.atleast_1d().shape)
torch.Size([1])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
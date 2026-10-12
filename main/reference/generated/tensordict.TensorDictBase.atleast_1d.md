# TensorDictBase.atleast_1d

TensorDictBase.atleast_1d()

Returns the tensordict with at least 1 batch dimension.

If the tensordict already has 1 or more batch dimensions, it is returned unchanged.
Otherwise, a dimension of size 1 is prepended.

Returns:

a tensordict with at least 1 batch dimension.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.randn(3)}, batch_size=[])
>>> print(td.atleast_1d().shape)
torch.Size([1])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
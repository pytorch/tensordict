# TensorDictBase.atleast_2d

TensorDictBase.atleast_2d()

Returns the tensordict with at least 2 batch dimensions.

If the tensordict already has 2 or more batch dimensions, it is returned unchanged.
Otherwise, dimensions of size 1 are prepended to reach 2 dimensions.

Returns:

a tensordict with at least 2 batch dimensions.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.randn(3)}, batch_size=[3])
>>> print(td.atleast_2d().shape)
torch.Size([1, 3])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
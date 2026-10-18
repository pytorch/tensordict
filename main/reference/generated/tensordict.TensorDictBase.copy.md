# TensorDictBase.copy

TensorDictBase.copy() → Self

Returns a shallow copy of the tensordict: a new structure that holds the same leaves.

The same as `td.clone(recurse=False)`. The nested tensordicts are new
objects, but the leaves are the same tensors: an in-place change to a
leaf shows in both.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), b={"c": torch.zeros(3)}, batch_size=[3])
>>> c = td.copy()
>>> c["a"] is td["a"], c["b"] is td["b"]
(True, False)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
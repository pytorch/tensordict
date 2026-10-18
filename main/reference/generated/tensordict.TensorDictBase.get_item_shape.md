# TensorDictBase.get_item_shape

TensorDictBase.get_item_shape(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*)

Returns the shape of the entry at `key`, without reading the entry when possible.

For a lazy stack, a dimension whose size differs between the stacked
entries has size `-1`.

Parameters:

**key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key of the entry.

Examples

```
>>> import torch
>>> from tensordict import TensorDict, lazy_stack
>>> td = lazy_stack([TensorDict(x=torch.zeros(2)), TensorDict(x=torch.zeros(3))])
>>> td.get_item_shape("x")
torch.Size([2, -1])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
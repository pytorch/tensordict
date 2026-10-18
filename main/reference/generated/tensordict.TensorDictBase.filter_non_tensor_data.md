# TensorDictBase.filter_non_tensor_data

TensorDictBase.filter_non_tensor_data() → Self

Returns a tensordict without the non-tensor entries, nested ones included.

The other leaves are not copied.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), b="text", n={"c": "text", "d": torch.zeros(3)}, batch_size=[3])
>>> list(td.filter_non_tensor_data().keys(include_nested=True, leaves_only=True))
['a', ('n', 'd')]
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
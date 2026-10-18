# TensorDictBase.names

*property*TensorDictBase.names

The dimension names of the tensordict, one per batch dimension.

A dimension without a name is `None`. The names can be set at
construction time using the `names` argument, or by assignment
(`td.names = [...]`). The names must be unique. Shape operations and
indexing keep the names of the dimensions that remain. Assigning
`None` removes the names. See [Batch size, shapes, and indexing](../../concepts/batch_size.html#batch-size).

See also

[`refine_names()`](tensordict.TensorDictBase.refine_names.html#tensordict.TensorDictBase.refine_names) and [`rename_()`](tensordict.TensorDictBase.rename_.html#tensordict.TensorDictBase.rename_).

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3, 4), batch_size=[3, 4], names=["time", None])
>>> td.names
['time', None]
>>> td.names = ["time", "batch"]
>>> td[0].names
['batch']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
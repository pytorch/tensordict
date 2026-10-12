# TensorDictBase.size

TensorDictBase.size(*dim: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [int](https://docs.python.org/3/builtins/functions.html#int)

Returns the batch size, or the size of one batch dimension.

Parameters:

**dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - a batch dimension. If `None`, the whole
[`batch_size`](tensordict.TensorDictBase.batch_size.html#tensordict.TensorDictBase.batch_size) is returned. Defaults to `None`.

Examples

```
>>> from tensordict import TensorDict
>>> td = TensorDict(batch_size=[3, 4])
>>> td.size()
torch.Size([3, 4])
>>> td.size(-1)
4
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
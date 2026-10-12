# TensorDictBase.to_tensordict

TensorDictBase.to_tensordict(***, *retain_none: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Returns a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) copy of this tensordict.

Every tensor is cloned, and every sub-tensordict and tensorclass is
converted to a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict), recursively. Use it to
get a regular, unlocked tensordict in memory from a lazy stack, a
memory-mapped or locked tensordict, a
[`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) or a tensorclass. The batch
size, device and dimension names are kept.

Keyword Arguments:

**retain_none** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the `None` fields of
tensorclass instances are written as entries of the result.
Defaults to `None`, which behaves like `False`: the
`None` fields are dropped.

Returns:

a new TensorDict object containing the same values.

Examples

```
>>> import torch
>>> from tensordict import LazyStackedTensorDict, TensorDict
>>> stack = LazyStackedTensorDict(
... TensorDict(a=torch.zeros(3)), TensorDict(a=torch.ones(3)), stack_dim=0)
>>> td = stack.to_tensordict()
>>> type(td).__name__, td.batch_size
('TensorDict', torch.Size([2]))
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
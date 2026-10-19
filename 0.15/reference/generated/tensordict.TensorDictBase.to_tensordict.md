# TensorDictBase.to_tensordict

TensorDictBase.to_tensordict(***, *retain_none: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Returns a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) copy of this tensordict.

Every tensor is cloned, and every nested tensordict and tensorclass is
converted to a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict), recursively. Use it to
get a regular, unlocked tensordict in memory from a lazy stack, a
memory-mapped or locked tensordict, a
[`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) or a tensorclass. The batch
size, device and names of the batch dimensions are kept, with one
exception: if the tensordicts of a lazy stack have no names, the
result has no names either, even when the stack dimension has a name
(from `stack_dim_name` or the `names` setter). Name the dimension
again with [`refine_names()`](tensordict.TensorDictBase.refine_names.html#tensordict.TensorDictBase.refine_names), as in the example.

Keyword Arguments:

**retain_none** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the `None` fields of
tensorclass instances are written as entries of the result.
Defaults to `None`, which behaves like `False`: the
`None` fields are dropped.

Returns:

A new [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) with copies of the values.

Examples

```
>>> import torch
>>> from tensordict import LazyStackedTensorDict, TensorDict
>>> stack = LazyStackedTensorDict(
... TensorDict(a=torch.zeros(3)), TensorDict(a=torch.ones(3)), stack_dim=0)
>>> td = stack.to_tensordict()
>>> type(td).__name__, td.batch_size
('TensorDict', torch.Size([2]))
>>> stack.names = ["s"]
>>> stack.to_tensordict().names
[None]
>>> stack.to_tensordict().refine_names("s").names
['s']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
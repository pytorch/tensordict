# TensorDictBase.entry_class

*abstract*TensorDictBase.entry_class(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*) → [type](https://docs.python.org/3/builtins/functions.html#type)

Returns the class of an entry, possibly without reading the entry.

It is the same as `type(td.get(key))`. Prefer it to that expression
when [`get()`](tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get) is expensive, for example on a lazy stack or a
tensordict stored on disk.

Parameters:

**key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key of the entry.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), b=TensorDict(c=torch.zeros(3)))
>>> td.entry_class("a")
<class 'torch.Tensor'>
>>> td.entry_class("b")
<class 'tensordict._td.TensorDict'>
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
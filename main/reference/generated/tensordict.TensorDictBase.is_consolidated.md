# TensorDictBase.is_consolidated

TensorDictBase.is_consolidated()

Whether the leaves of the tensordict are views into one storage made by [`consolidate()`](tensordict.TensorDictBase.consolidate.html#tensordict.TensorDictBase.consolidate).

Becomes `False` when the storage no longer matches the content: after
[`memmap_()`](tensordict.TensorDictBase.memmap_.html#tensordict.TensorDictBase.memmap_) moves the leaves out of it, after an in-place write
into a non-tensor entry, or after an entry is replaced, added or
deleted in an unpickled copy (which is not locked). In-place writes
into the tensor leaves keep it `True`. Call [`consolidate()`](tensordict.TensorDictBase.consolidate.html#tensordict.TensorDictBase.consolidate)
to build a new consolidated copy.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3))
>>> td.is_consolidated(), td.consolidate().is_consolidated()
(False, True)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
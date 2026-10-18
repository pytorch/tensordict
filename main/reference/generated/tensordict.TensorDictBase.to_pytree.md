# TensorDictBase.to_pytree

TensorDictBase.to_pytree()

Rebuilds the pytree that [`from_pytree()`](tensordict.TensorDictBase.from_pytree.html#tensordict.TensorDictBase.from_pytree) converted to this tensordict.

If the tensordict has no `"_pytree_type"` entry, that is, if it was
not created by [`from_pytree()`](tensordict.TensorDictBase.from_pytree.html#tensordict.TensorDictBase.from_pytree), this method returns `self`.
A nested tensordict without that entry is kept as a tensordict in the
pytree.

See also

[`from_pytree()`](tensordict.TensorDictBase.from_pytree.html#tensordict.TensorDictBase.from_pytree) for the supported types.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict.from_pytree({"a": [torch.zeros(2), torch.ones(2)]})
>>> td.to_pytree()
{'a': [tensor([0., 0.]), tensor([1., 1.])]}
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
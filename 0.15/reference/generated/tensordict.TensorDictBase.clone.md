# TensorDictBase.clone

TensorDictBase.clone(*recurse: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, ***kwargs*) → Self

Clones a TensorDictBase subclass instance onto a new TensorDictBase subclass of the same type.

To create a TensorDict instance from any other TensorDictBase subtype, call the [`to_tensordict()`](tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict) method
instead.

Parameters:

**recurse** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, each tensor contained in the
TensorDict will be copied too. Otherwise only the TensorDict
tree structure is copied: the sub-tensordicts are new objects,
but the leaves are the same tensors as in `self`. Defaults to
`True`.

Note

Unlike many other ops (pointwise arithmetic, shape operations, ...) `clone` does not inherit the
original lock attribute. This design choice is made such that a clone can be created to be modified,
which is the most frequent usage.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.zeros(3), "b": {"c": torch.ones(3)}}, [3])
>>> td_clone = td.clone()
>>> td_clone["a"] is td["a"]
False
>>> td_shallow = td.clone(recurse=False)
>>> td_shallow["a"] is td["a"], td_shallow["b"] is td["b"]
(True, False)
>>> td.lock_().clone().is_locked
False
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `clone`

![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict
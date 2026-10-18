# TensorDictBase.clone

TensorDictBase.clone(*recurse: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, ***kwargs*) → Self

Returns a copy of the tensordict, of the same type.

With `recurse=True`, every tensor leaf is copied. The result is not
locked, even if the tensordict is: a clone is usually made to be
modified. To get a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) from any other type,
call [`to_tensordict()`](tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict) instead.

Parameters:

- **recurse** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the leaves are copied too.
If `False`, only the tree of nested tensordicts is copied:
the nested tensordicts are new objects, and the leaves are the
same tensors as in the tensordict. Defaults to `True`.
- ****kwargs** - options of the class, for example `newfile`, the
destination file of a [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict)
cloned with `recurse=True`.

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
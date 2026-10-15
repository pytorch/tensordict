# TensorDictBase.is_locked

*property*TensorDictBase.is_locked*: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

`True` if the tensordict is locked.

A locked tensordict raises `RuntimeError` on any operation that adds,
removes or renames an entry, or that replaces a leaf with a new tensor.
Writes into the existing leaves still work. See [`lock_()`](tensordict.TensorDictBase.lock_.html#tensordict.TensorDictBase.lock_).

Setting `td.is_locked = True` calls [`lock_()`](tensordict.TensorDictBase.lock_.html#tensordict.TensorDictBase.lock_), and
`td.is_locked = False` calls [`unlock_()`](tensordict.TensorDictBase.unlock_.html#tensordict.TensorDictBase.unlock_).

Examples

```
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": 0})
>>> td.is_locked
False
>>> td.is_locked = True
>>> td.is_locked
True
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `is_locked`

![](../../_images/sphx_glr_tensordict_memory_thumb.png)

[Devices and memory-mapped tensors](../../tutorials/tensordict_memory.html)

Devices and memory-mapped tensors
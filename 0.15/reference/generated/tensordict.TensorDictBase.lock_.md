# TensorDictBase.lock

TensorDictBase.lock_() → Self

Locks the tensordict and its nested tensordicts, and returns it.

A locked tensordict raises `RuntimeError` on any operation that adds,
removes or renames an entry, or that replaces a leaf with a new tensor:
[`set()`](tensordict.TensorDictBase.set.html#tensordict.TensorDictBase.set), `__setitem__`, [`update()`](tensordict.TensorDictBase.update.html#tensordict.TensorDictBase.update), [`pop()`](tensordict.TensorDictBase.pop.html#tensordict.TensorDictBase.pop),
[`rename_key_()`](tensordict.TensorDictBase.rename_key_.html#tensordict.TensorDictBase.rename_key_), `del`. Writes into the existing leaves still
work: [`set_()`](tensordict.TensorDictBase.set_.html#tensordict.TensorDictBase.set_), [`update_()`](tensordict.TensorDictBase.update_.html#tensordict.TensorDictBase.update_), and `set(..., inplace=True)`.
Check the state with [`is_locked`](tensordict.TensorDictBase.is_locked.html#tensordict.TensorDictBase.is_locked). See [Locking and temporary changes](../../concepts/keys.html#keys-locking).

Can be used as a context manager: the tensordict is unlocked when the
`with` block exits, also if the block raises an exception.

Examples

```
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": 1, "b": 2, "c": 3}, batch_size=[])
>>> with td.lock_():
... assert td.is_locked
... try:
... td.set("d", 0) # adds a key: error
... except RuntimeError:
... print("td is locked!")
... try:
... del td["a"] # removes a key: error
... except RuntimeError:
... print("td is locked!")
... try:
... td.rename_key_("a", "d") # renames a key: error
... except RuntimeError:
... print("td is locked!")
... # These calls write into the existing leaves, so they work:
... _ = td.set("a", 0, inplace=True)
... _ = td.set_("a", 0)
... _ = td.update({"a": 0}, inplace=True)
... _ = td.update_({"a": 0})
td is locked!
td is locked!
td is locked!
>>> td.is_locked
False
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `lock_`

![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict
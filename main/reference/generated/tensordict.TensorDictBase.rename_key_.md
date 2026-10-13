# TensorDictBase.rename_key

TensorDictBase.rename_key_(*old_key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *new_key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *safe: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Renames a key with a new string and returns the same tensordict with the updated key name.

The value is not copied. A locked tensordict raises `RuntimeError`
(see [`lock_()`](tensordict.TensorDictBase.lock_.html#tensordict.TensorDictBase.lock_)). Until 0.17, a locked
[`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) or store renames the key and
warns instead.

Parameters:

- **old_key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - key to be renamed. A missing key raises
`KeyError`.
- **new_key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - new name of the entry.
- **safe** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a `KeyError` is raised when
the new key is already present in the TensorDict. If
`False`, the existing entry is overwritten. Defaults to
`False`.

Returns:

self

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.zeros(3), "b": {"c": torch.ones(3)}}, [3])
>>> td = td.rename_key_("a", "x")
>>> td = td.rename_key_(("b", "c"), ("b", "y"))
>>> list(td.keys(include_nested=True, leaves_only=True))
[('b', 'y'), 'x']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `rename_key_`

![](../../_images/sphx_glr_tensordict_keys_thumb.png)

[Getting, setting, and nesting keys](../../tutorials/tensordict_keys.html)

Getting, setting, and nesting keys
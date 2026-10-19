# TensorDictBase.rename_key

TensorDictBase.rename_key_(*old_key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *new_key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *safe: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Renames an entry of the tensordict in place and returns the tensordict.

The value is not copied. A locked tensordict raises `RuntimeError`
(see [`lock_()`](tensordict.TensorDictBase.lock_.html#tensordict.TensorDictBase.lock_)). Until TensorDict 0.17, a locked
[`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) or store renames the key and
warns instead.

Parameters:

- **old_key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key to rename. A missing key raises
`KeyError`.
- **new_key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the new key of the entry. An empty key raises
`KeyError`. A key inside `old_key`, such as `("a", "b")`
for `"a"`, moves the entry there, as
`td.set(new_key, td.pop(old_key))` does.
- **safe** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a `KeyError` is raised when
`new_key` is already present in the tensordict, unless it
lies inside `old_key`, which moves with the entry. If
`False`, the existing entry is overwritten. Defaults to
`False`.

Returns:

`self`.

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